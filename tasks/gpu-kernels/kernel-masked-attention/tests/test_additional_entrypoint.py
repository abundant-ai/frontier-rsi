"""Load the restricted entrypoint for untimed correctness execution."""
from pathlib import Path
from test_additional_cases import Kernel
from test_restricted import load


def load_kernel(candidate: Path | None, plan: Path | None) -> Kernel:
    """Load numerical submission code through the existing restricted loader.

    Args:
        candidate: Frozen submitted Python file.
        plan: Unused for restricted-Python tasks; must be absent.

    Returns:
        The validated run callable.

    Raises:
        ValueError: The expected candidate path is absent or a plan is supplied.
    """
    if candidate is None or plan is not None:
        raise ValueError('A restricted candidate file is required')
    return load(candidate, ['run']).run
