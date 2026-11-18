"""Fix the bugs in a small multi-file Python project with tools (the engine: rl_evalgen/project_env.py).

The model answers each observation with one tool call on a line of its own: ls(), view("name.py"),
write("name.py") followed by the whole file in a ```python block, run_python("name.py") or submit(). Reward 1 when
every function passes its hidden tests at submit() (or when the tool calls run out), else 0.

Episodes are params dicts as project_env describes ({"files", "hidden", "max_calls", "timeout_s"}); the sample rows in
/app/data/samples.jsonl hold some. There is no generator here: generate() raises. expert(params) needs params["gold"]
({file: fixed source}), which the samples do not have.
"""
from rl_evalgen import project_env

NAME = "multifile_repair"
DIFFICULTIES = ["easy", "medium", "hard", "expert"]
HAS_GENERATOR = False
Env = project_env.Env
grade = project_env.grade


def generate(rng, difficulty):
    raise NotImplementedError("multifile_repair has no episode generator: build params dicts yourself "
                              "(see rl_evalgen/project_env.py and /app/data/samples.jsonl)")


def expert(params):
    """Tool calls that fix the project from params["gold"]: look around, run the tests, rewrite each file, submit."""
    if not params.get("gold"):
        raise ValueError("expert() needs params['gold']: {file name: fixed source}")
    acts = ["ls()", 'run_python("run_tests.py")']
    for name in sorted(params["gold"]):
        acts.append(f'view("{name}")')
        acts.append(f'write("{name}")\n```python\n{params["gold"][name].rstrip()}\n```')
    return acts + ['run_python("run_tests.py")', "submit()"]
