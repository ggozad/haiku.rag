import pytest

from evaluations.numbers import extract_numbers, numbers_close


class TestExtractNumbers:
    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            ("the answer is 127.4 dollars", [127.4]),
            ("$1,234.5", [1234.5]),
            ("margin was 50.3%", [50.3, 0.503]),
            ("loss of (123)", [-123.0]),
            ("a change of −1.9 million", [-1.9, -1.9e6]),
            ("no figures here", []),
        ],
        ids=[
            "plain",
            "currency-and-thousands",
            "percent-yields-both-readings",
            "parenthesised-negative",
            "unicode-minus",
            "no-numbers",
        ],
    )
    def test_extracts(self, text: str, numbers: list[float]) -> None:
        assert extract_numbers(text) == numbers

    def test_scale_word_adds_scaled_and_raw(self) -> None:
        numbers = extract_numbers("revenue of 1.2 billion")
        assert 1.2 in numbers
        assert 1.2e9 in numbers


@pytest.mark.parametrize(
    ("value", "target", "close"),
    [
        (127.4, 127.40, True),
        (100.9, 100.0, True),
        (102.0, 100.0, False),
        (0.0, 0.0, True),
        (0.1, 0.0, False),
    ],
    ids=[
        "within-tolerance",
        "just-inside",
        "outside-tolerance",
        "zero-target-exact",
        "zero-target-off",
    ],
)
def test_numbers_close(value: float, target: float, close: bool) -> None:
    assert numbers_close(value, target, 0.01) is close
