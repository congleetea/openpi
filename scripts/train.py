import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """
    根据配置加载预训练权重的函数说明
    """
    # 使用配置中的weight_loader加载预训练权重
    loaded_params = loader.load(params_shape)
    # 验证加载的权重形状和数据类型是否与预期匹配
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 从加载的参数中移除jax.ShapeDtypeStruct。这确保只返回实际的加载参数。
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    '''
    初始化训练状态函数说明（包含权重加载机制）
    '''
    # 1. 创建优化器（包含学习率调度）
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    # 2. 定义初始化函数（用于创建训练状态）
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # 初始化模型（及其参数）。
        model = config.model.create(model_rng)

        # 将部分参数合并到模型中（权重加载的核心步骤）。这里是通过下面的_load_weights_and_validate加载的预训练权重,初始的时候要把这
        # 部分参数加进去作为初始的参数.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # 如果部分参数不是状态的子集，这将产生错误。
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        # 从模型中提取当前参数状态（包含所有层的参数）。
        params = nnx.state(model)   
        # 将冻结参数转换为bfloat16, 这是因为在训练过程中, 冻结的参数通常是不需要更新的, 而bfloat16可以在不损失精度的情况下, 减少内存占用.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # 这是训练恢复的时候执行的, init_train_state里面如果resume就会从checkpoint加载训练状态, 而不是加载预训练权重, 然后返回.
        return train_state_shape, state_sharding

    # 根据配置加载预训练权重
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 实际执行init进行初始化, 包括构建模型
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 释放部分参数缓冲区以优化内存。
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


'''
训练步骤函数详细说明
'''
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    '''
    单步训练函数：执行一次前向传播、计算损失、反向传播和参数更新
    '''
    # 将模型定义和参数合并，创建完整的可训练模型
    model = nnx.merge(state.model_def, state.params)
    # 设置模型为训练模式（启用dropout等训练特定操作）
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        '''
        损失函数：计算模型在给定观测和动作上的损失
        '''
        # 计算批次中每个样本的损失
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        # 返回平均损失
        return jnp.mean(chunked_loss)

    # 为当前训练步骤生成唯一的随机数种子
    train_rng = jax.random.fold_in(rng, state.step)
    # 解包批次数据为观测和动作
    observation, actions = batch

    # 创建DiffState，只对可训练参数计算梯度（忽略冻结参数）
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 计算损失和梯度（只对可训练参数求梯度）
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 过滤出可训练参数
    params = state.params.filter(config.trainable_filter)
    # 使用优化器更新参数和优化器状态
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    # 应用参数更新
    new_params = optax.apply_updates(params, updates)

    # 将更新后的参数应用到模型中
    nnx.update(model, new_params)
    # 从更新后的模型中提取参数
    new_params = nnx.state(model)

    # 创建新的训练状态，更新步数和参数
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    # 如果使用EMA（指数移动平均），更新EMA参数
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                # 计算EMA：new_ema = decay * old_ema + (1 - decay) * new_params
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 过滤出kernel参数（排除bias、scale、pos_embedding、input_embedding等）
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),  # 排除特定名称的参数
            lambda _, x: x.value.ndim > 1,  # 只选择维度大于1的参数（kernel参数）
        ),
    )
    # 收集训练指标信息
    info = {
        "loss": loss,  # 当前批次的平均损失
        "grad_norm": optax.global_norm(grads),  # 梯度的全局范数（用于监控训练稳定性）
        "param_norm": optax.global_norm(kernel_params),  # kernel参数的全局范数
    }
    return new_state, info


def main(config: _config.TrainConfig):
    '''
    主训练函数说明
    '''
    '''
    初始化工作
    '''
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 验证批量大小是否能被设备数量整除（分布式训练要求）
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 更新JAX编译缓存目录配置
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 初始化随机数生成器, 分成两个随机数, 一个用于训练, 一个用于初始化模型
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 创建分布式网格（用于FSDP分片）
    mesh = sharding.make_mesh(config.fsdp_devices)
    # 定义数据分片策略
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 定义复制分片策略
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化检查点管理器，处理覆盖/恢复逻辑
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    # 初始化Weights & Biases实验跟踪
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建分布式数据加载器
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    # 创建数据迭代器
    data_iter = iter(data_loader)
    # 获取第一个批次用于初始化验证
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 记录第一个批次的图像用于验证数据管道
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态（包括模型参数, 模型, 优化器状态等）
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    # 等待JAX操作完成
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果是恢复训练，从检查点恢复训练状态
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 编译训练步骤函数以优化性能
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),  # 输入分片策略
        out_shardings=(train_state_sharding, replicated_sharding),  # 输出分片策略
        donate_argnums=(1,),  # 优化内存使用
    )

    # 获取起始训练步数
    start_step = int(train_state.step)
    # 创建训练进度条, TQDM 是一个流行的 Python 进度条库，名称来源于阿拉伯语 "taqaddum"，意思是"progress"（进展）。它用于在长时间运行的循环或操作中显示进度条
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),  # 训练步数范围
        initial=start_step,  # 初始步数
        total=config.num_train_steps,  # 总训练步数
        dynamic_ncols=True,  # 动态列宽
    )

    # 初始化信息收集列表
    infos = []
    # 主训练循环
    for step in pbar:
        with sharding.set_mesh(mesh):  # 设置分布式网格上下文, 在这个上下文中, jax知道如何在分布式环境中分片数据和参数
            # 执行单步训练
            train_state, info = ptrain_step(train_rng, train_state, batch)
        # 收集训练信息（损失、梯度范数等）
        infos.append(info)
        # 定期记录日志
        if step % config.log_interval == 0:
            # 堆叠收集的信息
            stacked_infos = common_utils.stack_forest(infos)
            # 计算平均指标
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # 格式化信息字符串
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")  # 在进度条中显示信息
            wandb.log(reduced_info, step=step)  # 记录到wandb
            infos = []  # 清空信息收集列表
        # 获取下一个批次的数据
        batch = next(data_iter)

        # 检查点保存条件：满足间隔条件或训练结束时保存
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # 保存训练状态到检查点
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # 等待检查点管理器完成所有异步操作
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


'''
训练脚本整体流程说明
'''
# 程序入口点: 解析命令行参数并启动主训练函数
if __name__ == "__main__":
    main(_config.cli())
