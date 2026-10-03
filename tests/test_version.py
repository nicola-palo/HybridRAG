from importlib.metadata import version

import hybridrag


def test_declared_version_matches_installed_package() -> None:
    assert version("hybridrag") == hybridrag.__version__
