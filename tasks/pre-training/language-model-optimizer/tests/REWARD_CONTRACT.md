# Signed raw rewards and disqualification

Valid optimizer results have `valid: 1` and a numeric `reward` equal to negative
mean held-out cross-entropy. Invalid submissions are DISQUALIFIED, not assigned
a baseline value: their reward artifact contains `valid: 0` and
`disqualified: 1`, with no `reward` key. They must be excluded from score
comparisons and training datasets of valid raw measurements.

The second field is deliberate. Oddish's `oddish.core.harbor_artifacts._extract_reward`
reads `rewards['reward']`; when that is absent, it falls back to the sole value
only if the dictionary has exactly one entry. The two-entry disqualification
therefore produces `None`, not zero. A one-entry `{"valid": 0}` would have been
wrong and was replaced before admitting any full cohort.

This behavior has been tested against the actual installed Oddish reader:

- `{"valid": 1, "reward": -4.25}` -> primary reward `-4.25`.
- `{"valid": 0, "disqualified": 1}` -> primary reward `None`.

`score.py` preserves disqualification under both raw and future linear scoring.
The offline verifier requires the same result as the saved artifact and checks
that Oddish has no numeric reward for a disqualified trial. No implicit
conversion of a missing scalar to zero is part of this measurement contract.
Assigning a finite arbitrary penalty instead would be a different convention
from the requested disqualification and would contaminate raw loss values.
