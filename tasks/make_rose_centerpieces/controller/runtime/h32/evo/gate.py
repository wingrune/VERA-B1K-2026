"""Short static gate (catches mistakes, NOT a security boundary: isolation is by process, planner gets only obs): imports, file access, network. Process isolation is the main protection;
the same rules are shown to the mutator in the prompt (RULES) so refusals stay rare."""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {
    "math", "numpy", "scipy", "dataclasses", "typing", "collections", "itertools", "functools",
    "enum", "heapq", "random", "json", "time", "logging", "copy", "__future__", "core.skills_lib",
}
# `core` is allowed only as `core.skills_lib`; relative imports must stay inside the planner package (level up to the depth of the module).
# getattr/__dict__/__class__/super() stay allowed: the old gate rejected such legitimate mutations.
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals",
                "__builtins__", "__loader__", "__spec__"}
BANNED_ATTRS = {"system", "popen", "socket", "urlopen", "request", "environ", "load", "loadtxt", "genfromtxt",
                "fromfile", "savetxt", "tofile", "loadmat", "FileHandler", "read_text", "write_text",
                "__subclasses__", "__globals__", "__builtins__"}

RULES = (
    "Gate rules: import only " + ", ".join(sorted(ALLOWED_IMPORTS)) + " and sibling modules of the planner "
    "(relative imports); no open/eval/exec/__import__, no os/sys/subprocess/socket/requests, no reading "
    "files or environment. Read data from planner/data via the provided loader only."
)


def check(modules: dict[str, str]) -> list[str]:
    """Return a list of violations (empty = pass)."""
    names = {p[:-3].replace("/", ".") for p in modules if p.endswith(".py")}
    local_roots = {n.split(".")[0] for n in names}
    allowed_roots = {x.split(".")[0] for x in ALLOWED_IMPORTS if x != "core.skills_lib"} | local_roots
    errs = []
    for path, src in modules.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            errs.append(f"{path}: syntax error line {e.lineno}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > path.count("/") + 1:   # a module in phases/ may use `..`, one in the root only `.`
                    errs.append(f"{path}:{node.lineno} relative import above the planner package")
                if node.level:
                    continue
                mods = [node.module or ""]
            else:
                mods = []
            for m in mods:
                root = m.split(".")[0]
                if root == "core":
                    ok = m == "core.skills_lib" or m.startswith("core.skills_lib.")
                else:
                    ok = root in allowed_roots
                if not ok:
                    errs.append(f"{path}:{node.lineno} import {m} not allowed")
            if isinstance(node, ast.Name) and node.id in BANNED_NAMES:  # any reference, so aliasing is caught
                errs.append(f"{path}:{node.lineno} name {node.id} not allowed")
            if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
                errs.append(f"{path}:{node.lineno} attribute {node.attr} not allowed")
    return errs
