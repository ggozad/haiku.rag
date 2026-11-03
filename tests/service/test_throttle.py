import asyncio
import subprocess

import pytest

from haiku.rag.config.models import ThrottleConfig
from haiku.rag.service import throttle as throttle_module
from haiku.rag.service.throttle import Conditions, Throttle, read_conditions


def _throttle(conditions: list[Conditions], **settings) -> Throttle:
    readings = iter(conditions)
    last = conditions[-1]
    return Throttle(
        ThrottleConfig(check_interval_s=0.01, **settings),
        read=lambda: next(readings, last),
    )


@pytest.mark.parametrize(
    ("conditions", "settings", "reason"),
    [
        (Conditions(), {}, None),
        (Conditions(on_battery=True), {"on_battery": "pause"}, "on battery"),
        (Conditions(on_battery=True), {"on_battery": "slow"}, None),
        (Conditions(load_per_core=0.9), {}, "system busy"),
        (Conditions(load_per_core=0.9), {"max_load": None}, None),
        (Conditions(free_memory_percent=5), {}, "low memory"),
        (Conditions(free_memory_percent=5), {"min_free_memory_percent": None}, None),
        (Conditions(free_memory_percent=50, load_per_core=0.1), {}, None),
    ],
)
async def test_pause_reasons(conditions, settings, reason):
    throttle = _throttle([conditions], **settings)
    await throttle.refresh()
    assert throttle.conditions == conditions
    assert throttle.pause_reason == reason


async def test_on_battery_conversions_run_one_at_a_time():
    throttle = _throttle([Conditions(on_battery=True)], on_battery="slow")
    running = []
    peak = []

    async def unit():
        async with throttle.gate():
            running.append(1)
            peak.append(len(running))
            await asyncio.sleep(0.02)
            running.pop()

    await asyncio.gather(unit(), unit(), unit())
    assert throttle.slow
    assert max(peak) == 1


async def test_on_power_conversions_run_together():
    throttle = _throttle([Conditions(on_battery=False)])
    running = []
    peak = []

    async def unit():
        async with throttle.gate():
            running.append(1)
            peak.append(len(running))
            await asyncio.sleep(0.02)
            running.pop()

    await asyncio.gather(unit(), unit())
    assert not throttle.slow
    assert max(peak) == 2


async def test_a_paused_conversion_waits_until_the_machine_is_free():
    throttle = _throttle(
        [Conditions(load_per_core=0.95), Conditions(load_per_core=0.95), Conditions()]
    )
    async with throttle.gate():
        assert throttle.pause_reason is None


async def test_conditions_are_read_at_most_every_interval():
    reads = []
    throttle = Throttle(
        ThrottleConfig(check_interval_s=60),
        read=lambda: reads.append(1) or Conditions(),
    )
    await throttle.refresh()
    await throttle.refresh()
    assert reads == [1]


def test_reading_conditions_off_macos(monkeypatch):
    monkeypatch.setattr(throttle_module.sys, "platform", "linux")
    monkeypatch.setattr(throttle_module.os, "getloadavg", lambda: (4.0, 0, 0))
    monkeypatch.setattr(throttle_module.os, "cpu_count", lambda: 8)
    assert read_conditions() == Conditions(load_per_core=0.5)


@pytest.mark.parametrize(
    ("batt", "memory", "expected"),
    [
        (
            "Now drawing from 'Battery Power'\n",
            "42\n",
            Conditions(on_battery=True, load_per_core=1.0, free_memory_percent=42),
        ),
        (
            "Now drawing from 'AC Power'\n",
            "n/a",
            Conditions(on_battery=False, load_per_core=1.0),
        ),
        (None, None, Conditions(load_per_core=1.0)),
    ],
)
def test_reading_conditions_on_macos(monkeypatch, batt, memory, expected):
    monkeypatch.setattr(throttle_module.sys, "platform", "darwin")
    monkeypatch.setattr(throttle_module.os, "getloadavg", lambda: (2.0, 0, 0))
    monkeypatch.setattr(throttle_module.os, "cpu_count", lambda: 2)
    outputs = {"pmset": batt, "sysctl": memory}
    monkeypatch.setattr(throttle_module, "_command", lambda *args: outputs[args[0]])
    assert read_conditions() == expected


def test_a_command_that_fails_reads_as_unknown(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(throttle_module.subprocess, "run", fail)
    assert throttle_module._command("pmset", "-g", "batt") is None
    monkeypatch.undo()
    assert throttle_module._command("echo", "hi") == "hi\n"
