"""Runtime observation-only interfaces for mutable controller helpers.

Targets and callable bindings live in trusted module tables, not facade
attributes. This interface is paired with the static candidate gate and
process isolation; it does not claim an adversarial Python sandbox.
"""
import weakref

_OBJECTS = weakref.WeakKeyDictionary()
_CALLS = weakref.WeakKeyDictionary()

SCHEMAS = {
    "policy": {
        "values": {"action", "age", "child", "current_cameras", "current_q", "held", "parent",
                   "phase", "pose_index", "pose_route", "poses", "rose_releases", "seen", "step"},
        "methods": {"base", "bounds", "change"},
        "children": {"perception": "perception", "servo": "servo", "robot": "robot",
                     "dual": "dual", "paired": "paired"}, "writes": set()},
    "perception": {"values": {"models"}, "methods": set(), "children": {}, "writes": set()},
    "servo": {"values": {"indices", "target", "synchronized", "retain_target"},
              "methods": {"pose", "solve", "step"}, "children": {"kin": "kin"},
              "writes": {"target", "synchronized", "retain_target"}},
    "kin": {"values": set(), "methods": {"forward"}, "children": {}, "writes": set()},
    "robot": {"values": {"names", "radii", "extra_payloads"},
              "methods": {"centers", "margins", "valid"}, "children": {}, "writes": set()},
    "depth": {"values": {"log_odds", "res", "n", "z_min"}, "methods": {"xy", "ij"},
              "children": {}, "writes": set()},
    "dual": {"values": {"active", "right_held"}, "methods": set(), "children": {}, "writes": set()},
    "paired": {"values": {"active"}, "methods": set(), "children": {}, "writes": set()},
}


def unwrap(value):
    if isinstance(value, PolicyFacade):
        return _OBJECTS[value][0]
    return value


class TrustedCall:
    __slots__ = ("__weakref__",)

    def __getattribute__(self, name):
        raise AttributeError("Trusted callable exposes no attributes")

    def __call__(self, *args, **kwargs):
        function, _ = _CALLS[self]
        return function(*(unwrap(value) for value in args),
                        **{key: unwrap(value) for key, value in kwargs.items()})


def trusted_call(function, owner):
    call = TrustedCall()
    _CALLS[call] = (function, owner)
    return call


class PolicyFacade:
    __slots__ = ("__weakref__",)

    def __getattribute__(self, name):
        target, schema_name, owner = _OBJECTS[self]
        schema = SCHEMAS[schema_name]
        if name in schema["methods"]:
            return trusted_call(getattr(target, name), owner)
        if name in schema["children"]:
            child = getattr(target, name)
            return None if child is None else facet(child, schema["children"][name], owner)
        if name in schema["values"]:
            value = getattr(target, name)
            if schema_name in ("perception", "motion_perception") and name == "models":
                # Only canonical geometry/color used by these helper functions.
                return {category: {key: model[key] for key in ("xyz", "rgb", "bounds") if key in model}
                        for category, model in value.items()}
            return value
        raise AttributeError(f"Observation-only {schema_name} interface does not expose {name}")

    def __setattr__(self, name, value):
        target, schema_name, _ = _OBJECTS[self]
        if name not in SCHEMAS[schema_name]["writes"]:
            raise AttributeError(f"Observation-only {schema_name} interface does not permit writing {name}")
        setattr(target, name, value)


def facet(target, schema_name="policy", owner=None):
    if isinstance(target, PolicyFacade):
        return target
    result = PolicyFacade()
    _OBJECTS[result] = (target, schema_name, owner)
    return result


def release_owner(owner):
    for value, binding in list(_OBJECTS.items()):
        if binding[2] == owner:
            _OBJECTS.pop(value, None)
    for value, binding in list(_CALLS.items()):
        if binding[1] == owner:
            _CALLS.pop(value, None)


def bind_helpers(modules, collision_class, owner):
    """Wrap entry boundaries without changing candidate function bodies."""
    facade_module = modules["held_vase_view"].__package__ + ".skills"
    import sys
    sys.modules[facade_module].ObservedSceneCollision = trusted_call(collision_class, owner)

    for module_name, functions in {
            "held_vase_view": ("arm_ray_exposure", "plan_view"),
            "release_view": ("start_release_view", "step_release_view"),
            "retreat_recovery": ("start_vertical_retreat", "step_vertical_retreat", "check_egress_action")}.items():
        module = modules[module_name]
        # Each import bound this symbol before the facade was updated.
        module.ObservedSceneCollision = trusted_call(collision_class, owner)
        for name in functions:
            original = getattr(module, name)
            def boundary(policy, *args, _function=original, **kwargs):
                return _function(facet(policy, "policy", owner), *args, **kwargs)
            setattr(module, name, boundary)

    view = modules["held_vase_view"]
    original_view = view.HeldVaseView
    class HeldVaseView(original_view):
        def __init__(self, policy):
            super().__init__(facet(policy, "policy", owner))
    view.HeldVaseView = HeldVaseView

    navigation = modules["pose_navigation"]
    original_planner = navigation.PosePlanner
    class PosePlanner(original_planner):
        def __init__(self, depth, robot, servo, *args, **kwargs):
            super().__init__(facet(depth, "depth", owner), facet(robot, "robot", owner),
                             facet(servo, "servo", owner), *args, **kwargs)
    navigation.PosePlanner = PosePlanner


def bind_motion_helpers(modules, owner):
    """Expose only RGB-D/proprio-derived state for the expanded motion package."""
    motion_values = {'pickup_view_state','grasp_trials','pickup_retry_count','stage','grasp','motion_history','hand_history','stationary_taskspace',
        'close_history','close_ready','pickup_ready','pickup_confirmations','relative_history',
        'pickup_observation_steps','pickup_reacquire_command','pickup_reacquire_started',
        'pickup_reacquire_failed','left_pickup_confirmed','pregrasp_rose_world',
        'commanded_release_world','rose_assembly','vase_payload_bounds','nav',
        'unconfirmed_refit_debug','waypoints','waypoint','stable','release_pose','release_debug'}
    # Explicit estimator state, derived solely from RGB-D/proprio; no native pose.
    # Optional-odometry candidates may compensate its next prediction.
    if 'visual_odometry' in modules:
        motion_values.add('odom')
    SCHEMAS['motion_policy'] = {
        'values': SCHEMAS['policy']['values'] | motion_values,
        'methods': SCHEMAS['policy']['methods'] | {'center','plan_vase_placement'},
        'children': {**SCHEMAS['policy']['children'],'perception':'motion_perception','dual':'motion_dual'},
        'writes': motion_values | {'held','child','parent','pose_route'}}
    SCHEMAS['motion_perception'] = {
        **SCHEMAS['perception'],'values': {'models','active','excluded','vase_predicted'},'writes': {'active','vase_predicted'}}
    SCHEMAS['motion_dual'] = {
        **SCHEMAS['dual'],'methods': {'begin_right_place'},'children': {'perception':'motion_tracks'}}
    SCHEMAS['motion_tracks'] = {'values': {'hands','released'},'methods': {'mark_released'},'children': {},'writes': set()}
    for name, functions in {
        'pickup_recovery': ('update_pickup','prepare_reacquire','finish_reacquire'),
        'insertion': ('make_task_path','release_observed','record_rose_release'),
        'vase_transfer': ('plan_vase_placement','resume_after_rose_release')}.items():
        for function in functions:
            original = getattr(modules[name],function)
            def boundary(policy,*args,_function=original,**kwargs):
                return _function(facet(policy,'motion_policy',owner),*args,**kwargs)
            setattr(modules[name],function,boundary)
