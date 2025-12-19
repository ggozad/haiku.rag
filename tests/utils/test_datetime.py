from datetime import UTC, datetime, timezone

import pytest

from haiku.rag.utils import parse_datetime, to_utc


class TestParseDateTime:
    def test_parse_iso8601_with_timezone(self):
        """Parse ISO 8601 datetime with timezone."""
        result = parse_datetime("2025-01-15T14:30:00+00:00")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
        assert result.second == 0
        assert result.tzinfo is not None

    def test_parse_iso8601_without_timezone(self):
        """Parse ISO 8601 datetime without timezone (naive)."""
        result = parse_datetime("2025-01-15T14:30:00")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30

    def test_parse_date_only(self):
        """Parse date-only string as start of day."""
        result = parse_datetime("2025-01-15")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_parse_various_formats(self):
        """Parse various datetime formats."""
        # ISO with Z suffix
        result = parse_datetime("2025-01-15T14:30:00Z")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

        # With milliseconds
        result = parse_datetime("2025-01-15T14:30:00.123")
        assert result.microsecond == 123000

    def test_parse_invalid_raises_value_error(self):
        """Invalid datetime string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_datetime("not-a-datetime")
        assert "Could not parse datetime" in str(exc_info.value)


class TestToUtc:
    def test_naive_datetime_assumes_local_and_converts(self):
        """Naive datetime is assumed local and converted to UTC."""
        naive = datetime(2025, 1, 15, 14, 30, 0)
        result = to_utc(naive)
        assert result.tzinfo == UTC

    def test_utc_datetime_unchanged(self):
        """UTC datetime is returned as-is."""
        utc_dt = datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC)
        result = to_utc(utc_dt)
        assert result == utc_dt
        assert result.tzinfo == UTC

    def test_other_timezone_converts_to_utc(self):
        """Datetime with other timezone is converted to UTC."""
        from datetime import timedelta

        # Create a datetime at UTC+5
        tz_plus5 = timezone(timedelta(hours=5))
        dt_plus5 = datetime(2025, 1, 15, 19, 30, 0, tzinfo=tz_plus5)

        result = to_utc(dt_plus5)

        # 19:30 UTC+5 = 14:30 UTC
        assert result.tzinfo == UTC
        assert result.hour == 14
        assert result.minute == 30
