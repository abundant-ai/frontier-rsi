"""The student's environment for multi-file repair: a flat Python project and five tools.

    ls()                    list the files
    view("name.py")         show a file with line numbers
    write("name.py")        replace or create a file; the whole new contents go in a ```python block after the call
    run_python("name.py")   run a file in the sandbox: what it printed, the traceback, the exit code
    submit()                finish; the project is graded as it is now

One tool call per reply, on its own line; text before it is free reasoning. The first call in a reply runs (calls
inside a fenced block do not count). Every reply uses one call from the budget, a reply without a valid call
included; when the budget is spent the project is graded as it is.

Grading (`grade`): every function listed in params["hidden"] is called, from the module it was defined in, on hidden
inputs in the sandbox (`sandbox.call_project`), and must return the expected value on all of them. Reward 1 or 0.
The visible tests (run_tests.py) are only a hint: editing them changes nothing.

It needs only `sandbox.py`. It follows the posttrain-kit's env interface: Env(params).reset() -> text,
Env.step(text) -> (observation, reward, done).

params (JSON):
    files       {name: source}: the project as the student first sees it
    hidden      [{"module", "func", "inputs": [[arg, ...], ...], "expected": [value, ...]}]  (encoded, sandbox.encode)
    max_calls   tool calls allowed
    timeout_s   wall-clock limit of one run_python
    graded      optional: "api" for projects whose other functions are helpers: the prompt then says that only
                the functions api.py imports are graded. It changes only the prompt: grade() calls exactly the
                functions in hidden, so list only those there.
    change      optional: an update task; the text of the new specification, shown in the prompt
    task        optional: "implement" (write the functions whose body is `raise NotImplementedError`) or
                "port" (update the callers of the module params["interface"], whose interface changed)
    interface   with task "port": the file name of the module whose interface changed
    (other keys are ignored here)
"""
from __future__ import annotations

import ast
import re

from rl_evalgen.sandbox import ALLOWED_IMPORTS, FILE_NAME, MAX_PROJECT_FILES, call_project, lib_tops, run_project, same

NAME = "multifile_repair"
TOOLS = ("ls", "view", "write", "run_python", "submit")
MAX_FILE_CHARS = 30_000
MAX_SHOWN = 6000            # characters of run output shown to the student
GRADE_TIMEOUT_S = 30.0

_CALL = re.compile(r"^[ \t]*`?[ \t]*(ls|view|write|run_python|submit)[ \t]*\((.*)\)[ \t]*`?[ \t]*$")
_FENCE_LINE = re.compile(r"^[ \t]*```")
_BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n?[ \t]*```", re.S)


_LIB_NAMES = {"numpy": "NumPy", "scipy": "SciPy", "torch": "PyTorch", "pandas": "pandas"}


def _graded_sentence(params: dict) -> str:
    """Which functions the hidden tests call: every function (the default), or, with params["graded"] == "api", the
    ones api.py imports (projects whose other functions are helpers, such as chained_repair's)."""
    if params.get("graded") == "api":
        return ("The project is graded with hidden tests: every function that api.py imports is called on many valid "
                "inputs (each docstring says which inputs are valid), and every call must return what the docstring "
                "describes. The other functions are helpers, tested through the functions that call them.")
    return ("The project is graded with hidden tests: every function in it is called on many valid inputs (each "
            "docstring says which inputs are valid), and every call must return what the docstring describes.")


def _intro(params: dict) -> str:
    """The task: fix bugs (the default); with params["change"], update the project to a new specification; with
    params["task"] "implement" or "port", write the stubbed functions or update the callers of a changed module."""
    if params.get("task") == "implement":
        return ("You are finishing a small Python project. Some of its functions are not written yet: their bodies are "
                "`raise NotImplementedError`. Each of them has a docstring that says exactly what it must return. Write "
                "them all, so that every function does what its docstring says; the functions already written are "
                "correct. Functions call each other, so a missing function also breaks the ones that use it.\n\n"
                + _graded_sentence(params) +
                " run_tests.py runs a few tests, but passing them does not mean the project is correct.\n\n")
    if params.get("task") == "port":
        mod = params["interface"]
        return (f"You are updating a small Python project. The module {mod} was just changed to a new interface: its "
                f"functions take or return different things than before, as its docstrings and code now say. The "
                f"rest of the project still uses the old interface, so parts of it no longer work. Update the other "
                f"modules so that they work with the new {mod}, and do not change {mod} itself: it is graded as it "
                f"is now. Every other function must keep doing what its docstring says.\n\n"
                + _graded_sentence(params) +
                " run_tests.py runs a few tests, but passing them does not mean the project is correct.\n\n")
    if params.get("change"):
        return ("You are updating a small Python project to a new specification. Each function's docstring says what "
                "it does now. The specification is changing as described below: update the project so that it follows "
                "the new specification. Where the change says nothing, the behavior stays as the docstrings describe."
                "\n\nThe change:\n\n" + params["change"].strip() + "\n\n"
                "The project is graded with hidden tests: every function that api.py imports is called on many inputs "
                "that are valid under the new specification, and every call must return what the new specification "
                "describes. The other functions are helpers, tested through the functions that call them. run_tests.py "
                "runs a few tests of the new specification, but passing them does not mean the project is correct.\n\n")
    return ("You are fixing bugs in a small Python project. Some of its functions do not do what their docstrings say. "
            "Find the bugs and fix them.\n\n" + _graded_sentence(params) +
            " run_tests.py runs a few tests, but passing them does not mean the project is correct.\n\n")


def instructions(params: dict) -> str:
    libs = lib_tops(params.get("libs", ()))
    mods = ", ".join(sorted(ALLOWED_IMPORTS - {"sys"}) + [f"{m} ({_LIB_NAMES[m]})" for m in libs])
    return (
        _intro(params) +
        "Tools. Use exactly one per reply, on a line of its own; you may think before it.\n"
        "  ls()                    list the files\n"
        "  view(\"name.py\")         show a file with line numbers\n"
        "  write(\"name.py\")        replace or create a file: put its whole new contents in a ```python block "
        "right after the call\n"
        "  run_python(\"name.py\")   run a file: shows what it printed, the traceback if it failed, and the exit code\n"
        "  submit()                finish: the project is graded as it is now\n\n"
        "Example of a write:\n"
        "write(\"notes.py\")\n```python\ndef double(x):\n    return 2 * x\n```\n\n"
        f"Rules: all files are in one folder and end in .py. Code may import only the project's own files and these "
        f"modules: {mods}. There is no file, network or process access. A run is stopped after "
        f"{params.get('timeout_s', 10):g} seconds. You can write and run your own scripts to test things. You have "
        f"{params['max_calls']} tool calls; when they run out, the project is graded as it is."
        + (f" Results are compared exactly, except that pandas Series and DataFrames must have the expected index, column "
           f"labels and kind of dtype (float, int, bool or other) in each column, arrays the expected kind of dtype and "
           f"shape, and float values may differ from the expected ones by a relative {params['rel_tol']:g}."
           if "pandas" in libs else
           f" Results are compared exactly, except that arrays and tensors must have the expected kind of dtype (float, int "
           f"or bool) and shape, and float values may differ from the expected ones by a relative {params['rel_tol']:g}."
           if libs else "")
    )


def parse_call(text: str) -> tuple[str | None, list | str]:
    """(tool, args) for the first tool call in a reply, or (None, what was wrong)."""
    lines = (text or "").split("\n")
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else _CALL.match(line)
        if not m:
            continue
        tool, inner = m.group(1), m.group(2).strip()
        args = _args(inner)
        if args is None:
            return None, f"could not read the arguments of {tool}({inner})"
        if tool in ("ls", "submit"):
            return tool, []
        if not args:
            return None, f"{tool} needs a file name, like {tool}(\"main.py\")"
        name = str(args[0])
        if tool == "write":
            rest = "\n".join(lines[i + 1:])
            b = _BLOCK.search(rest)
            if len(args) >= 2 and isinstance(args[1], str):
                return tool, [name, args[1]]
            if not b:
                return None, ("write needs the whole new file in a ```python block right after the call, like:\n"
                              "write(\"notes.py\")\n```python\ndef double(x):\n    return 2 * x\n```")
            return tool, [name, b.group(2)]
        return tool, [name]
    return None, "no tool call found"


def _args(inner: str) -> list | None:
    if not inner:
        return []
    try:
        node = ast.parse(f"f({inner})", mode="eval").body
        if not node.keywords and all(isinstance(a, ast.Constant) for a in node.args):
            return [a.value for a in node.args]
    except (SyntaxError, ValueError, RecursionError):
        pass
    bare = inner.strip().strip("'\"` ")            # view(main.py) without quotes
    return [bare] if FILE_NAME.match(bare) else None


def _numbered(src: str) -> str:
    lines = src.split("\n")
    w = len(str(len(lines)))
    return "\n".join(f"{k:>{w}} | {ln}" for k, ln in enumerate(lines, 1))


def _clip(text: str, limit: int = MAX_SHOWN) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters not shown ...]\n{text[-half:]}"


def grade(params: dict, files: dict) -> dict:
    """{"passed": bool, "functions": {func: bool}, "error": str | None} for a project's current files."""
    calls = [{"module": h["module"], "func": h["func"], "inputs": h["inputs"]} for h in params["hidden"]]
    out, why = call_project(files, calls, timeout_s=GRADE_TIMEOUT_S, libs=params.get("libs", ()))
    if out is None:
        return {"passed": False, "functions": {h["func"]: False for h in params["hidden"]}, "error": why}
    funcs = {}
    for h, r in zip(params["hidden"], out):
        tol = params.get("rel_tol", 1e-9)
        ok = "outcomes" in r and all(o[0] == 0 and same(o[1], want, tol) for o, want in zip(r["outcomes"], h["expected"]))
        funcs[h["func"]] = funcs.get(h["func"], True) and ok
    return {"passed": all(funcs.values()), "functions": funcs, "error": None}


class Env:
    def __init__(self, params: dict):
        self.p = params
        self.files = dict(params["files"])
        self.calls = 0
        self.result: dict | None = None        # set when the episode ends: grade() of the final files

    def reset(self) -> str:
        return instructions(self.p)

    def _finish(self, note: str):
        self.result = grade(self.p, self.files)
        return note, (1.0 if self.result["passed"] else 0.0), True

    def step(self, text: str):
        self.calls += 1
        left = self.p["max_calls"] - self.calls
        tool, args = parse_call(text)
        if tool == "submit":
            return self._finish("Submitted.")
        if tool is None:
            obs = (f"Error: {args}. Reply with exactly one tool call on its own line: ls(), view(\"name.py\"), "
                   f"write(\"name.py\") followed by a ```python block, run_python(\"name.py\") or submit().")
        else:
            obs = self._run(tool, args)
        if left <= 0:
            return self._finish(obs + "\n\nNo tool calls left: the project is graded as it is.")
        return f"{obs}\n\n[{left} tool call{'s' if left != 1 else ''} left]", 0.0, False

    def _run(self, tool: str, args: list) -> str:
        if tool == "ls":
            return "\n".join(f"{n}  ({len(s.splitlines())} lines)" for n, s in sorted(self.files.items())) or "(no files)"
        name = args[0]
        if tool in ("view", "run_python") and name not in self.files:
            return f"Error: no file named {name}. Files: {', '.join(sorted(self.files))}"
        if tool == "view":
            return f"{name}:\n{_numbered(self.files[name])}"
        if tool == "write":
            if not FILE_NAME.match(name):
                return "Error: a file name is letters, digits and _, ending in .py (no folders)"
            if len(args[1]) > MAX_FILE_CHARS:
                return f"Error: a file can have at most {MAX_FILE_CHARS} characters"
            if name not in self.files and len(self.files) >= MAX_PROJECT_FILES:
                return f"Error: at most {MAX_PROJECT_FILES} files"
            new = args[1].rstrip("\n") + "\n"
            existed = name in self.files
            self.files[name] = new
            return f"{'Replaced' if existed else 'Created'} {name} ({len(new.splitlines())} lines)."
        r = run_project(self.files, name, timeout_s=float(self.p.get("timeout_s", 10)), libs=self.p.get("libs", ()))
        if r.exit_code is None:
            why = {"timeout": f"stopped after {self.p.get('timeout_s', 10):g} seconds",
                   "out of memory": "stopped: it used too much memory"}.get(r.error, r.error)
            return f"$ python {name}\n{_clip(r.printed)}\nError: {why}".replace("\n\n", "\n")
        parts = [f"$ python {name}"]
        if r.printed:
            parts.append(_clip(r.printed.rstrip("\n")))
        if r.traceback:
            parts.append(_clip(r.traceback.rstrip("\n"), MAX_SHOWN // 2))
        parts.append(f"(exit code {r.exit_code})")
        return "\n".join(parts)


def selftest() -> None:
    assert parse_call("I will look.\nls()") == ("ls", [])
    assert parse_call("view(\"a.py\")\nrun_python(\"b.py\")") == ("view", ["a.py"])
    assert parse_call("view(a.py)") == ("view", ["a.py"])
    assert parse_call("`run_python('t.py')`") == ("run_python", ["t.py"])
    assert parse_call("Fix:\nwrite(\"a.py\")\n```python\nx = 1\ny = 2\n```\nthen run_python(\"t.py\")") == \
        ("write", ["a.py", "x = 1\ny = 2"])
    assert parse_call("```python\nls()\n```\nsubmit()") == ("submit", [])
    assert parse_call("write(\"a.py\")")[0] is None and parse_call("hello")[0] is None
    assert parse_call("view()")[0] is None and parse_call("view(1 + )")[0] is None

    lib = "def inc(x):\n    \"\"\"Return x plus one.\"\"\"\n    return x + 2\n"
    params = {"files": {"lib.py": lib, "run_tests.py": "from lib import inc\nprint(inc(1))\n"},
              "hidden": [{"module": "lib", "func": "inc", "inputs": [[1], [5]], "expected": [2, 6]}],
              "max_calls": 6, "timeout_s": 5}
    e = Env(params)
    assert "run_tests.py" in e.reset() and "6 tool calls" in e.reset()
    obs, r, done = e.step("ls()")
    assert "lib.py  (3 lines)" in obs and "[5 tool calls left]" in obs and (r, done) == (0.0, False)
    obs, _, _ = e.step("run_python(\"run_tests.py\")")
    assert "$ python run_tests.py\n3\n(exit code 0)" in obs, obs
    obs, _, _ = e.step("view(\"lib.py\")")
    assert "3 |     return x + 2" in obs, obs
    obs, _, _ = e.step("nothing here")
    assert obs.startswith("Error: no tool call found"), obs
    obs, _, _ = e.step("write(\"lib.py\")\n```python\ndef inc(x):\n    return x + 1\n```")
    assert obs.startswith("Replaced lib.py (2 lines)"), obs
    obs, r, done = e.step("submit()")
    assert (r, done) == (1.0, True) and e.result["passed"], (obs, e.result)

    # grading ignores the visible tests and calls the function where it was defined; the budget ends the episode
    e = Env(params)
    e.step("write(\"run_tests.py\")\n```python\nprint('all passed')\n```")
    obs, r, done = e.step("write(\"lib.py\")\n```python\nimport os\ndef inc(x):\n    return x + 1\n```")
    assert not done
    obs, _, _ = e.step("run_python(\"run_tests.py\")")
    assert "rejected: lib.py: import os" in obs, obs
    for _ in range(2):
        obs, r, done = e.step("ls()")
    assert not done
    obs, r, done = e.step("ls()")
    assert done and r == 0.0 and "No tool calls left" in obs and e.result["error"].startswith("rejected"), (obs, e.result)
    # a crash in a file shows a traceback with the student's file and line
    e = Env(params)
    e.step("write(\"lib.py\")\n```python\ndef inc(x):\n    return [x][1]\n```")
    obs, _, _ = e.step("run_python(\"run_tests.py\")")
    assert 'File "lib.py", line 2, in inc' in obs and "IndexError" in obs and "(exit code 1)" in obs, obs


if __name__ == "__main__":
    selftest()
    print("project_env selftest passed")
