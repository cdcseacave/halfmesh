import re

import halfmesh


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", halfmesh.version())


def test_dunder_version_matches_native():
    assert halfmesh.__version__ == halfmesh.version()


def test_package_reexports_every_binding():
    """A function bound in binding.cpp but missing from __init__.py would only be
    reachable as halfmesh._halfmesh.<name>."""
    native = {name for name in dir(halfmesh._halfmesh) if not name.startswith("_")}
    assert native == set(halfmesh.__all__) - {"__version__"}
