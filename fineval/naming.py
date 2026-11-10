"""Current generator naming with stable identities for historical reproduction."""

from pathlib import Path

INDIGO_NAME = "InDiGO"

# Sampling streams were keyed by these identifiers in the published run records.
# Retain them so a display-name change cannot change the random indices.
_SEED_NAMES = {"InDiGO": "AIL", "indigo": "ail", "INDIGO": "AIL"}


def generator_seed_name(name: str) -> str:
    """Return the original stream identifier for a generator name."""
    return _SEED_NAMES.get(name, name)


def resolve_indigo_path(path: Path) -> Path:
    """Accept original archive input names when the current path is absent."""
    if path.exists():
        return path
    aliases = {
        "indigo_prices.parquet": "ail_prices.parquet",
        "indigo_synthetic_data": "ail_synthetic_data",
    }
    original = Path(*(aliases.get(part, part) for part in path.parts))
    return original if original.exists() else path
