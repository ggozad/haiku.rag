from evaluations.numbers import extract_numbers, numbers_close


class TestExtractNumbers:
    def test_plain(self) -> None:
        assert extract_numbers("the answer is 127.4 dollars") == [127.4]

    def test_currency_and_thousands(self) -> None:
        assert extract_numbers("$1,234.5") == [1234.5]

    def test_percent_yields_both_readings(self) -> None:
        assert extract_numbers("margin was 50.3%") == [50.3, 0.503]

    def test_parenthesised_negative(self) -> None:
        assert extract_numbers("loss of (123)") == [-123.0]

    def test_unicode_minus(self) -> None:
        assert extract_numbers("a change of −1.9 million") == [-1.9, -1.9e6]

    def test_scale_word_adds_scaled_and_raw(self) -> None:
        numbers = extract_numbers("revenue of 1.2 billion")
        assert 1.2 in numbers
        assert 1.2e9 in numbers

    def test_no_numbers(self) -> None:
        assert extract_numbers("no figures here") == []


class TestNumbersClose:
    def test_within_tolerance(self) -> None:
        assert numbers_close(127.4, 127.40, 0.01)

    def test_just_inside(self) -> None:
        assert numbers_close(100.9, 100.0, 0.01)

    def test_outside_tolerance(self) -> None:
        assert not numbers_close(102.0, 100.0, 0.01)

    def test_zero_target_exact(self) -> None:
        assert numbers_close(0.0, 0.0, 0.01)
        assert not numbers_close(0.1, 0.0, 0.01)
