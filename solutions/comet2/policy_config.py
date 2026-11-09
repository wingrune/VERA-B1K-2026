from dataclasses import replace
from pathlib import Path
from .config import CheckpointConfig, PolicyConfig, configure_runtime

def create_transform_pipeline(train_config, checkpoint, policy_config):
    from openpi import transforms
    from openpi.training import checkpoints
    from .preprocessing import B1KInputs, B1KOutputs, CheckpointNormalize, CheckpointUnnormalize
    data = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data.asset_id != checkpoint.asset_id:
        raise ValueError(f'Unexpected normalization asset {data.asset_id}')
    stats = checkpoints.load_norm_stats(Path(checkpoint.path) / 'assets', data.asset_id)


    robot_inputs = list(data.data_transforms.inputs)
    robot_outputs = list(data.data_transforms.outputs)
    if type(robot_inputs[0]).__name__ != 'B1KInputs' or type(robot_outputs[-1]).__name__ != 'B1KOutputs':
        raise ValueError('Pinned data transform contract changed')
    robot_inputs[0] = B1KInputs()
    robot_outputs[-1] = B1KOutputs(policy_config.action_dim)
    inputs = [transforms.InjectDefaultPrompt(None), *robot_inputs,
              CheckpointNormalize(stats), *data.model_transforms.inputs]
    outputs = [*data.model_transforms.outputs, CheckpointUnnormalize(stats), *robot_outputs]
    return inputs, outputs, stats

def training_config(config, checkpoint):
    configure_runtime()
    from openpi.training import config as training
    train = training.get_config(checkpoint.model_config)
    return replace(train, data=replace(train.data, repo_id=checkpoint.asset_id, robot_config_name=config.robot))

def create_policy(config: PolicyConfig, checkpoint: CheckpointConfig):

    train = training_config(config, checkpoint)
    import jax.numpy as jnp
    from openpi.models.model import restore_params
    from openpi.policies.policy import Policy
    root = Path(checkpoint.path)
    if not (root / 'params').is_dir() or not (root / 'assets').is_dir():
        raise ValueError(f'Expected Orbax params/ and assets/ under {root}')
    if train.model.action_horizon != config.model_horizon:
        raise ValueError('Checkpoint action horizon mismatch')
    model = train.model.load(restore_params(root / 'params', dtype=jnp.bfloat16))
    inputs, outputs, stats = create_transform_pipeline(train, checkpoint, config)

    policy = Policy(model, transforms=inputs, output_transforms=outputs,
                    metadata=train.policy_metadata, sample_kwargs=None, is_pytorch=False)
    policy.input_transforms = tuple(inputs)
    policy.output_transforms = tuple(outputs)
    policy.norm_stats = stats
    return policy
