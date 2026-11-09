import missinggmm


def test_version_is_a_string() -> None:
    assert isinstance(missinggmm.__version__, str)
