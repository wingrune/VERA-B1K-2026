"""Bind task-owned mutable motion functions into the verified archived policy.

Only the approved blocks/methods are replaced. Observation conversion, physics,
logging and check_live_action remain trusted. No candidate receives this source.
"""
import ast
import hashlib
import textwrap
from pathlib import Path


def install(policy_class, source_path, manifest, modules):
    source = Path(source_path).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != manifest['motion_overlay_source_sha256']:
        raise RuntimeError('Motion overlay archive hash mismatch')
    from ..planners.fair.search_time import transform
    tree = ast.parse(transform("rose_policy", source))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Policy')
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_control')
    replacements = [(290,314,'stationary = _motion_pickup_update(self, s)'),
                    (344,349,'_motion_pickup_prepare(self)'),
                    (358,375,'action = _motion_pickup_finish(self, action, s)')]
    body = []
    matched = set()
    for node in original.body:
        replacement = next(((lo,hi,text) for lo,hi,text in replacements if lo <= node.lineno <= hi), None)
        if replacement:
            lo,hi,text = replacement
            if lo not in matched:
                body.extend(ast.parse(text).body)
                matched.add(lo)
        else:
            body.append(node)
    if matched != {lo for lo,_,_ in replacements}:
        raise RuntimeError('Motion overlay block boundaries not found')
    original.body = body
    # A separately compiled method lacks Python's implicit __class__ closure.
    class BaseForward(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'forward'
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == 'super'):
                return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='shared',ctx=ast.Load()),attr='Policy',ctx=ast.Load()),attr='forward',ctx=ast.Load()),args=[ast.Name(id='self',ctx=ast.Load()),*node.args],keywords=node.keywords),node)
            return node
    original = BaseForward().visit(original)
    module = ast.fix_missing_locations(ast.Module(body=[original],type_ignores=[]))
    namespace = dict(policy_class._forward_control.__globals__)
    namespace.update(_motion_pickup_update=modules['pickup_recovery'].update_pickup,
                     _motion_pickup_prepare=modules['pickup_recovery'].prepare_reacquire,
                     _motion_pickup_finish=modules['pickup_recovery'].finish_reacquire)
    exec(compile(module,str(source_path)+'::motion-overlay','exec'),namespace)
    policy_class._forward_control = namespace['_forward_control']
    for method,mod,func in [('make_task_path','insertion','make_task_path'),
                            ('release_observed','insertion','release_observed'),
                            ('_record_rose_release','insertion','record_rose_release'),
                            ('_resume_after_rose_release','vase_transfer','resume_after_rose_release'),
                            ('plan_vase_placement','vase_transfer','plan_vase_placement')]:
        setattr(policy_class,method,getattr(modules[mod],func))
