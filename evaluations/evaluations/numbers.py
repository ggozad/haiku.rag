import re

_SCALE = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}

_NUMBER_RE = re.compile(r"[(-]?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_SCALED_RE = re.compile(
    r"([(-]?\$?\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|trillion)",
    re.IGNORECASE,
)


def _to_float(token: str) -> float | None:
    stripped = token.lstrip()
    negative = stripped.startswith("(") or stripped.startswith("-")
    cleaned = (
        token.replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
        .lstrip("-")
    )
    if not cleaned or cleaned == ".":
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def extract_numbers(text: str) -> list[float]:
    """Pull numeric values out of free-form text.

    Handles currency, thousands separators, trailing percent signs, and
    parenthesised negatives. Numbers qualified by a scale word ("1.2 million")
    contribute both the raw and the scaled value, so either phrasing can match.
    """
    text = text.replace("−", "-")  # normalize the typographic minus sign
    numbers: list[float] = []
    for token in _NUMBER_RE.findall(text):
        value = _to_float(token)
        if value is not None:
            numbers.append(value)
            if "%" in token:
                # Gold answers store ratios as either a percent (24.69) or a
                # decimal (0.935); offer both readings of a percent figure.
                numbers.append(value / 100)
    for number, scale in _SCALED_RE.findall(text):
        value = _to_float(number)
        if value is not None:
            numbers.append(value * _SCALE[scale.lower()])
    return numbers


def numbers_close(value: float, target: float, eps: float = 0.01) -> bool:
    """Whether ``value`` matches ``target`` within relative tolerance ``eps``."""
    if target == 0:
        return value == 0
    return abs(value - target) / abs(target) <= eps
