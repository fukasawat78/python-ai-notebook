import pytest


numeric_datas = [
    (2, 5, 10),
    (3, 6, 18),
    (5, 9, 45)
]


@pytest.mark.parametrize("a, b, expect", numeric_datas)
def test_multiple(a, b, expect):
    assert a*b == expect