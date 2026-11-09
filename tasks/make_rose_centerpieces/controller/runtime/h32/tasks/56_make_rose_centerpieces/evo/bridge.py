"""Trusted candidate binding; this module is outside the mutable program.

The original controller archive stays read-only and fully hash-verified.
This preparation module has not yet been qualified by remote/runtime tests.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

from evo.gate import check
from ..planners.fair.source import SourceSession
from ..planners.fair.planner import Planner as FrozenPlanner
from .facade import bind_helpers, bind_motion_helpers, release_owner, trusted_call

MANIFEST = Path(__file__).with_name("HELPER_PACKAGE.json")
MOTION_MANIFEST = Path(__file__).with_name("HELPER_PACKAGE_MOTION.json")
ODOMETRY_MANIFEST = Path(__file__).with_name("HELPER_PACKAGE_ODOMETRY.json")

def package_manifest(folder):
    if (Path(folder) / 'visual_odometry.py').is_file():
        return ODOMETRY_MANIFEST
    return MOTION_MANIFEST if (Path(folder) / 'pickup_recovery.py').is_file() else MANIFEST


def candidate_sources(folder):
    folder = Path(folder).resolve()
    manifest = json.loads(package_manifest(folder).read_text())
    expected = set(manifest["mutable_modules"]) | set(manifest["frozen_modules"])
    files = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    if set(files) != expected:
        raise ValueError("Candidate path set differs from approved helper package")
    if any(p.is_symlink() or folder not in p.resolve().parents for p in files.values()):
        raise ValueError("Candidate files must remain inside their package")
    for name, digest in manifest["frozen_modules"].items():
        if hashlib.sha256(files[name].read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen candidate interface changed: {name}")
    sources = {name: path.read_text() for name, path in files.items()}
    errors = check(sources)
    if errors:
        raise ValueError("Candidate gate rejected: " + "; ".join(errors))
    for name in manifest["mutable_modules"]:
        limits = manifest.get("module_source_limits", {}).get(name, {})
        max_lines = int(limits.get("max_lines", 330))
        max_tokens = int(limits.get("max_estimated_tokens", 4000))
        if len(sources[name].splitlines()) > max_lines or int(len(sources[name])/3.5)+1 > max_tokens:
            raise ValueError(f"Candidate module exceeds full-source prompt limits: {name}")
        attributes = set(manifest["candidate_attribute_allowlist"])
        tree = ast.parse(sources[name])
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr not in attributes:
                raise ValueError(f"Candidate attribute not approved: {name}:{node.lineno} {node.attr}")
            if isinstance(node, ast.Name) and node.id in {"setattr", "delattr", "vars", "dir", "type", "object"}:
                raise ValueError(f"Candidate introspection not approved: {name}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "getattr":
                parent = parents.get(node)
                if not isinstance(parent, ast.Call) or parent.func is not node:
                    raise ValueError(f"Candidate getattr must be called directly: {name}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if (len(node.args) < 2 or not isinstance(node.args[1], ast.Constant)
                        or node.args[1].value not in attributes):
                    raise ValueError(f"Candidate getattr must name an approved literal: {name}:{node.lineno}")
    return folder, sources


class CandidateSession(SourceSession):
    def __init__(self, source_root, output, candidate_dir):
        super().__init__(source_root, output)
        folder, sources = candidate_sources(candidate_dir)
        self.package_name = "_rose_candidate_" + uuid.uuid4().hex
        self.bound_modules = {}
        self.reference_class = None
        self.reference_binding = None
        try:
            spec = importlib.util.spec_from_file_location(
                self.package_name, folder / "__init__.py",
                submodule_search_locations=[str(folder)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[self.package_name] = package
            spec.loader.exec_module(package)
            facade = importlib.import_module(self.package_name + ".skills")
            # Original dependency modules are retained by SourceSession.scope.
            # No controller has been created before the aliases are replaced.
            with self.scope():
                if 'visual_odometry.py' in sources:
                    self._bind_reference_odometry(facade)
                collision = importlib.import_module("scene_collision")
                navigation = importlib.import_module("navigation")
                facade.ObservedSceneCollision = collision.ObservedSceneCollision
                facade.rot2, facade.wrap = navigation.rot2, navigation.wrap
                if 'transform' in sources['skills.py']:
                    # Read the already hash-verified pure helper without eagerly
                    # importing observation_policy before candidate aliases exist.
                    source = ast.parse((self.root / 'observation_policy.py').read_text())
                    function = next(node for node in source.body
                                    if isinstance(node, ast.FunctionDef) and node.name == 'transform')
                    namespace = {'np': importlib.import_module('numpy')}
                    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
                    exec(compile(code, '<verified-transform>', 'exec'), namespace)
                    facade.transform = namespace['transform']
                for name in sorted(sources):
                    if name in ("__init__.py", "skills.py"):
                        continue
                    module = importlib.import_module(self.package_name + "." + Path(name).stem)
                    self.bound_modules[Path(name).stem] = module
                bind_helpers(self.bound_modules, collision.ObservedSceneCollision, self.package_name)
                if 'pickup_recovery' in self.bound_modules:
                    bind_motion_helpers(self.bound_modules, self.package_name)
            # Parent scope restores aliases after each call, then re-installs
            # this map on the next call, including lazy imports in forward().
            self.names.update(self.bound_modules)
            self.modules.update(self.bound_modules)
        except BaseException:
            self.close()
            raise
        self.binding_receipt = {
            name: {"candidate_sha256": hashlib.sha256(sources[name + ".py"].encode()).hexdigest(),
                   "legacy_alias": name, "candidate_module": module.__name__}
            for name, module in self.bound_modules.items()}

    def _bind_reference_odometry(self, facade):
        """Capture the original image-only class under the selected import overlay."""
        source_path = self.root / 'visual_odometry.py'
        original = importlib.import_module('visual_odometry')
        if Path(original.__file__).resolve() != source_path:
            raise RuntimeError('Reference VisualOdometry did not load from verified source')
        source = source_path.read_bytes()
        effective = source
        loader = original.__spec__.loader
        if self.compute_backend == 'cuda':
            from ..planners.fair.cuda_overlay import Loader, rewrite
            if not isinstance(loader, Loader) or loader.backend is not self.cuda_finder.backend:
                raise RuntimeError('Reference VisualOdometry bypassed the selected CUDA overlay')
            effective = rewrite('visual_odometry', source.decode()).encode()
        reference_class = original.VisualOdometry
        if reference_class.__module__ != 'visual_odometry':
            raise RuntimeError('Reference VisualOdometry is a candidate alias')
        self.reference_class = reference_class
        binding = self.reference_binding = {
            'source_file': 'visual_odometry.py',
            'source_sha256': hashlib.sha256(source).hexdigest(),
            'effective_sha256': hashlib.sha256(effective).hexdigest(),
            'backend': self.compute_backend,
            'class_module': reference_class.__module__,
            'class_name': reference_class.__name__,
            'loader_class': type(loader).__name__,
            'captured_before_candidate_aliases': True,
            'factory_instances_created': 0,
            'scope': 'Original RGB-D feature/registration/image-anchor state; no controller or simulator',
        }

        def create_visual_odometry(interval=5):
            if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
                raise ValueError('VisualOdometry interval must be a positive integer')
            reference = reference_class(interval=interval)
            binding['factory_instances_created'] += 1
            return reference

        # The callable exposes no attributes. Its only result is an independent
        # original image-odometry instance, never a policy or environment facet.
        facade.create_visual_odometry = trusted_call(create_visual_odometry, self.package_name)

    def assert_odometry_binding(self, policy):
        if 'visual_odometry' not in self.bound_modules:
            return
        if policy.vo.__class__ is not self.bound_modules['visual_odometry'].VisualOdometry:
            raise RuntimeError('Policy did not bind candidate VisualOdometry before initialization')
        if hasattr(policy.vo, 'reference') and policy.vo.reference.__class__ is not self.reference_class:
            raise RuntimeError('Candidate reference is not the captured original VisualOdometry')

    def create(self):
        policy = super().create()
        with self.scope():
            for name, module in self.bound_modules.items():
                if sys.modules.get(name) is not module:
                    raise RuntimeError(f"Candidate legacy alias not active: {name}")
            shared = importlib.import_module('observation_policy')
            if shared.grasp_regions is not self.bound_modules['grasp_regions'].grasp_regions:
                raise RuntimeError('Shared grasp planner did not bind candidate grasp_regions')
            legacy = importlib.import_module("rose_policy")
            if legacy.HeldVaseView is not self.bound_modules["held_vase_view"].HeldVaseView:
                raise RuntimeError("Frozen policy did not bind candidate HeldVaseView")
            if legacy.PosePlanner is not self.bound_modules["pose_navigation"].PosePlanner:
                raise RuntimeError("Frozen policy did not bind candidate PosePlanner")
            if legacy.start_release_view is not self.bound_modules["release_view"].start_release_view:
                raise RuntimeError("Frozen policy did not bind candidate release recovery")
            if legacy.start_vertical_retreat is not self.bound_modules["retreat_recovery"].start_vertical_retreat:
                raise RuntimeError("Frozen policy did not bind candidate vertical recovery")
            if legacy.supported_aperture_minimum is not self.bound_modules["grasp_regions"].supported_aperture_minimum:
                raise RuntimeError("Frozen policy did not bind candidate grasp affordance")
            if 'pickup_recovery' in self.bound_modules:
                from .motion_overlay import install
                install(legacy.Policy, self.root / 'rose_policy.py',
                        json.loads(MOTION_MANIFEST.read_text()), self.bound_modules)
                from .observed_egress import install as install_observed_egress
                install_observed_egress(legacy.Policy,self.bound_modules['retreat_recovery'].check_egress_action)
            self.assert_odometry_binding(policy)
        return policy

    def close(self):
        release_owner(self.package_name)
        for name in list(sys.modules):
            if name == self.package_name or name.startswith(self.package_name + "."):
                sys.modules.pop(name, None)


class Planner(FrozenPlanner):
    """Keep existing observation conversion, reset, action checks and trace log."""
    def __init__(self, source_dir, output_dir, candidate_dir):
        self.session = CandidateSession(source_dir, output_dir, candidate_dir)
        self.controller = self.session.create()
        self.last_phase = None
        self.reason = "initialized controller helper candidate"

    def reset(self, seed=0):
        super().reset(seed)
        self.session.assert_odometry_binding(self.controller)

    def act(self, obs):
        from .observed_egress import run_with_command_history
        return run_with_command_history(self.controller, lambda: super(Planner, self).act(obs))

    def close(self):
        self.session.close()
