"""Portable configuration for the preserved helper bridge; no simulator imports."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VARIANTS = {'h18': 'h30', 'h23': 'h23', 'h30': 'h30', 'h31': 'h32', 'h32': 'h32'}
PINS = {'h23': '6bd93c5af983cf5ec5e3191e73a5153896c46b56',
        'h30': 'dad862f0fd014a8ba8a37dab1f8249f30d9f12dc',
        'h32': 'bfabd3570a0eff2fedc092051a1a3b293629bb7a'}

def runtime(variant):
    return ROOT / 'runtime' / VARIANTS[variant]

def activate_imports(variant):
    selected = runtime(variant)
    for name in ('core', 'evo', 'tasks'):
        module = sys.modules.get(name)
        if module is not None:
            locations = list(getattr(module, '__path__', []))
            if not locations or any(not Path(p).resolve().is_relative_to(selected) for p in locations):
                raise RuntimeError(f'{name} belongs to another runtime; use a fresh process')
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))

@contextmanager
def profile_scope(output, variant='h30', backend='cuda'):
    if backend not in ('cpu', 'cuda'):
        raise ValueError('backend must be cpu or cuda')
    if variant not in VARIANTS:
        raise ValueError('Unknown preserved candidate')
    old = {k: v for k, v in os.environ.items() if k.startswith('BIMBO_ROSE_') or k == 'PIN'}
    for key in old:
        os.environ.pop(key, None)
    os.environ.update(BIMBO_ROSE_CONTROLLER=str(ROOT / 'controller'),
                      BIMBO_ROSE_CANDIDATE=str(ROOT / 'candidates' / variant),
                      BIMBO_ROSE_OUTPUT=str(Path(output).resolve()),
                      BIMBO_ROSE_PERCEPTION_BACKEND=backend,
                      PIN=PINS[VARIANTS[variant]])
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.startswith('BIMBO_ROSE_') or key == 'PIN':
                os.environ.pop(key, None)
        os.environ.update(old)
