import asyncio
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from haiku.rag.config.models import ThrottleConfig


@dataclass(frozen=True)
class Conditions:
    """What the machine is doing. None where it cannot be read."""

    on_battery: bool | None = None
    load_per_core: float | None = None
    free_memory_percent: int | None = None


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def read_conditions() -> Conditions:
    """Power source and free memory from macOS, load from the OS."""
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    if sys.platform != "darwin":
        return Conditions(load_per_core=load)
    battery = _command("pmset", "-g", "batt")
    memory = _command("sysctl", "-n", "kern.memorystatus_level")
    return Conditions(
        on_battery=None if battery is None else "'Battery Power'" in battery,
        load_per_core=load,
        free_memory_percent=int(memory)
        if memory and memory.strip().isdigit()
        else None,
    )


class Throttle:
    """Holds supervised conversions back while the machine should be spared.

    Conditions are read at most every `check_interval_s`. A paused conversion
    waits before it starts; one running is never stopped.
    """

    def __init__(
        self,
        settings: ThrottleConfig,
        *,
        read: Callable[[], Conditions] = read_conditions,
    ):
        self.settings = settings
        self._read = read
        self._conditions = Conditions()
        self._read_at: float | None = None
        self._single = asyncio.Lock()

    @property
    def conditions(self) -> Conditions:
        return self._conditions

    @property
    def pause_reason(self) -> str | None:
        """Why conversions are held back now, or None."""
        c, s = self._conditions, self.settings
        if c.on_battery and s.on_battery == "pause":
            return "on battery"
        if (
            s.max_load is not None
            and c.load_per_core is not None
            and c.load_per_core > s.max_load
        ):
            return "system busy"
        if (
            s.min_free_memory_percent is not None
            and c.free_memory_percent is not None
            and c.free_memory_percent < s.min_free_memory_percent
        ):
            return "low memory"
        return None

    @property
    def slow(self) -> bool:
        """Whether conversions run one at a time."""
        return bool(self._conditions.on_battery) and self.settings.on_battery == "slow"

    async def refresh(self) -> None:
        """Read the conditions again once they are older than the interval."""
        now = time.monotonic()
        if (
            self._read_at is None
            or now - self._read_at >= self.settings.check_interval_s
        ):
            self._conditions = await asyncio.to_thread(self._read)
            self._read_at = now

    @asynccontextmanager
    async def gate(self) -> AsyncIterator[None]:
        """Admit one conversion once nothing holds it back, for its whole run."""
        await self.refresh()
        while self.pause_reason is not None:
            await asyncio.sleep(self.settings.check_interval_s)
            await self.refresh()
        if self.slow:
            async with self._single:
                yield
        else:
            yield
