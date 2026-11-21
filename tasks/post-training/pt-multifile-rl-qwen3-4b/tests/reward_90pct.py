"""Turn the task grader's result into the rsi-bench reward (90% best-trial rule).

Runs after the task's own grader (see test.sh). Reads the raw metric the grader wrote to
/logs/verifier, then rewrites reward.json (and reward.txt if present):

  maximize: reward = clip((raw - 0.9*best) / (0.1*best), 0, 1)
  minimize: reward = clip((best/0.9 - raw) / (best/0.9 - best), 0, 1)

`best` is fixed in reward_90pct.json: the best valid model trial in the rsi-bench collection when
this task version was made. A run that matches or beats it gets 1; a run at or below 90% of its
performance gets 0. A run the grader marks invalid gets 0. The grader's own files are kept:
reward.json is copied to reward_grader.json first.

Standard library only. Usage: python3 reward_90pct.py [config] [verifier dir]
"""
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def score(raw, best, direction):
    if direction == 'maximize':
        x = (raw - 0.9 * best) / (0.1 * best)
    else:
        x = (best / 0.9 - raw) / (best / 0.9 - best)
    return min(1.0, max(0.0, x))


def raw_metric(cfg, grader, out):
    """Raw value and validity from the grader's files; (None, reason) when there is none."""
    if not isinstance(grader, dict):
        return None, 'no reward.json from the grader'
    if 'valid' in grader and not grader['valid']:
        return None, 'grader marked the run invalid'
    if cfg.get('require_passed') and grader.get('passed') is not True:
        return None, 'grader checks did not pass'
    m = cfg['metric']
    data = grader if m['file'] == 'reward.json' else load(out / m['file'])
    if not isinstance(data, dict):
        return None, f"no {m['file']}"
    if m['file'] != 'reward.json' and 'valid' in data and not data['valid']:
        return None, f"{m['file']} marks the run invalid"
    value = number(data.get(m['key']))
    if value is None:
        return None, f"no numeric {m['key']} in {m['file']}"
    value *= m.get('sign', 1)
    if cfg['direction'] == 'minimize' and value <= 0:
        return None, f'non-positive {m["key"]} for a lower-is-better metric'
    return value, None


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'reward_90pct.json'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/logs/verifier')
    cfg = json.loads(cfg_path.read_text())
    reward_json = out / 'reward.json'
    if not reward_json.exists() and not (out / 'reward.txt').exists():
        print('reward_90pct: the grader wrote no reward file; nothing to do')
        return
    grader = load(reward_json)
    if grader is None and (out / 'reward.txt').exists():
        try:
            grader = {'reward': float((out / 'reward.txt').read_text().strip())}
        except ValueError:
            grader = None
    if reward_json.exists():
        (out / 'reward_grader.json').write_bytes(reward_json.read_bytes())
    best = float(cfg['best_raw'])
    raw, reason = raw_metric(cfg, grader, out)
    reward = 0.0 if raw is None else score(raw, best, cfg['direction'])
    result = {'reward': reward, 'valid': 0 if raw is None else 1}
    if raw is not None:
        result['raw_metric'] = raw
    result['best_raw'] = best
    g = number((grader or {}).get('reward'))
    if g is not None:
        result['grader_reward'] = g
    tmp = out / 'reward.json.tmp'
    tmp.write_text(json.dumps(result) + '\n')
    os.replace(tmp, reward_json)
    if (out / 'reward.txt').exists():
        (out / 'reward.txt').write_text(f'{reward}\n')
    note = {'rule': cfg['formula'], 'direction': cfg['direction'], 'unit': cfg.get('unit'),
            'metric': cfg['metric'], 'best_raw': best, 'best_source': cfg.get('best_source'),
            'raw_metric': raw, 'reward': reward, 'invalid_reason': reason}
    (out / 'reward_90pct.json').write_text(json.dumps(note, indent=1) + '\n')
    print('reward_90pct:', json.dumps(result))


if __name__ == '__main__':
    main()
