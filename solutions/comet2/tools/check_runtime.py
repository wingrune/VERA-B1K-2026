import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'solutions/comet2'
sys.path.insert(0, str(ROOT))


def main():
    manifest = json.loads((HERE / 'runtime_manifest.json').read_text())
    if tuple(sys.version_info[:3]) != tuple(map(int, manifest['python'].split('.'))):
        raise RuntimeError('Unexpected Python version')
    for path, expected in manifest['files'].items():
        actual = hashlib.sha256((HERE / 'runtime' / path).read_bytes()).hexdigest()
        if actual != expected['sha256']:
            raise RuntimeError(f'Runtime source checksum mismatch: {path}')
    lock = (HERE / 'requirements.lock').read_bytes()
    if hashlib.sha256(lock).hexdigest() != manifest['requirements_sha256']:
        raise RuntimeError('Dependency lock checksum mismatch')
    requirements = lock.decode().splitlines()
    for requirement in requirements:
        name, version = requirement.split('===', 1)
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Runtime dependency mismatch: {name}')
    from solutions.comet2.config import PolicyConfig, CheckpointConfig, configure_runtime
    configure_runtime()
    from solutions.comet2.policy_config import training_config
    from openpi.models import pi0, tokenizer
    from openpi.training import checkpoints
    import openpi
    import openpi_client
    import lerobot
    for module in (openpi, openpi_client, lerobot):
        if not Path(module.__file__).resolve().is_relative_to(HERE / 'runtime'):
            raise RuntimeError(f'Unexpected module location: {module.__name__}')
    config = training_config(PolicyConfig(), CheckpointConfig(path=Path('/unused')))
    assert config.model.action_horizon == 32
    assert config.model.action_dim == 32
    assert config.model.pi05
    assert config.model.dtype == 'bfloat16'
    ids, mask = tokenizer.PaligemmaTokenizer(config.model.max_token_len).tokenize('turning_out_all_lights_before_sleep')
    assert ids.shape == mask.shape == (200,)
    print(json.dumps({'name': 'comet2', 'runtime_files': len(manifest['files']),
                      'dependencies': len(requirements), 'imports': 'passed', 'tokenizer': 'passed'}))


if __name__ == '__main__':
    main()
