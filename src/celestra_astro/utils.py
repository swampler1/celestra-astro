from __future__ import annotations

import re


###################################################################
# safe file names with aliases from SIMBAD
def safe_slug(s: str) -> str:
    """Return a filesystem-safe version of a string."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')
