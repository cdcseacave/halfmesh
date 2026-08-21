import re

import halfmesh


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", halfmesh.version())


def test_dunder_version_matches_native():
    assert halfmesh.__version__ == halfmesh.version()
