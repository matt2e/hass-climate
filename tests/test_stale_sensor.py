"""Tests for neutralizing rooms whose temperature sensor goes stale.

A sensor can keep publishing its last numeric value long after it has actually
stopped updating (dead battery, lost connectivity) without ever going
unavailable. Its ``last_reported`` age is how long it has been silent, but
many battery sensors only report on change, so silence alone is ambiguous:

- While a room is being conditioned (unit cooling/heating, vent open) its
  temperature must move, so silence past SENSOR_STALE_CONDITIONING_TIMEOUT
  (30 min, counted from when conditioning began) means the sensor is dead.
- Otherwise a steady room can legitimately be silent for hours, so only
  SENSOR_STALE_IDLE_TIMEOUT (8 h) is treated as dead.

A stale room is neutralized (marked satisfied, vent closed) and stays that way
until the sensor reports again, so a dead sensor can't cycle the AC.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE
from homeassistant.util import dt as dt_util

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_hass, make_parent, make_state, setup_feedback_switches


def _living_cover_zero_calls(hass) -> list:
    """Cover calls that close the living-room vent to position 0."""
    return [
        c
        for c in hass.services.async_call.call_args_list
        if c[0][0] == "cover"
        and c[0][2]["entity_id"] == "cover.living_room_vent"
        and c[0][2]["position"] == 0
    ]


def _turn_off_calls(hass) -> list:
    return [
        c
        for c in hass.services.async_call.call_args_list
        if c[0][:2] == ("climate", "turn_off")
    ]


def _set_primary_lights_on(parent) -> None:
    """Mark every light-controlled room as lit so it stays PRIMARY."""
    for room in parent._rooms:
        if room.light_entity:
            parent._room_states[room.name].light_on = True


def _override_states(hass, overrides: dict) -> None:
    """Replace some entity states on an existing mock hass."""
    original_get = hass.states.get
    hass.states.get = lambda eid: overrides.get(eid) or original_get(eid)


def _make_stale_cooling_parent(
    *,
    stale_age: timedelta,
    conditioning_for: timedelta | None = None,
    real_climate_action: str = HVACAction.COOLING,
    living_cover_pos: int = 100,
):
    """Cooling parent where the living room reports a hot value.

    The living room sits above target (would demand cooling) and its sensor
    last reported ``stale_age`` ago. ``conditioning_for`` says how long the
    room has already been conditioned (None = not yet). The office and bedroom
    sit comfortably below target with fresh readings so they never generate
    demand — the living room is the only room under test.
    """
    hass = make_hass(
        room_temps={
            "sensor.living_room_temp": 25.0,
            "sensor.office_temp": 20.0,
            "sensor.bedroom_temp": 20.0,
        },
        stale_sensors={"sensor.living_room_temp": stale_age},
        cover_positions={
            "cover.living_room_vent": living_cover_pos,
            "cover.office_vent": 0,
            "cover.bedroom_vent": 0,
        },
        light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
        real_climate_action=real_climate_action,
        real_climate_state=HVACMode.COOL,
    )
    parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
    setup_feedback_switches(hass)
    _set_primary_lights_on(parent)
    living = parent._rooms[0]
    parent._room_states[living.name] = RoomState(
        mode=RoomMode.PRIMARY,
        is_satisfied=False,
        cover_pos=living_cover_pos,
        light_on=True,
        conditioning_since=(
            dt_util.utcnow() - conditioning_for
            if conditioning_for is not None
            else None
        ),
    )
    return hass, parent, living


def _assert_neutralized(hass, parent, living) -> None:
    state = parent._room_states[living.name]
    assert state.is_satisfied is True
    assert state.cover_pos == 0
    assert state.sensor_stale_warned is True
    assert parent.calculate_fan_speed() is None


def _assert_trusted(parent, living) -> None:
    state = parent._room_states[living.name]
    # 25 °C is above the 22 °C cooling target → normal cooling demand.
    assert state.is_satisfied is False
    assert state.cover_pos == 100
    assert state.sensor_stale_warned is False
    assert state.stale_report_at is None
    assert parent.calculate_fan_speed() is not None


class TestConditioningTimeout:
    """While the room is being cooled, silence is suspicious after 30 min."""

    @pytest.mark.asyncio
    async def test_within_timeout_holds_demand(self):
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=25), conditioning_for=timedelta(minutes=25)
        )

        await parent._async_control_real_climate()

        _assert_trusted(parent, living)
        assert _living_cover_zero_calls(hass) == []

    @pytest.mark.asyncio
    async def test_past_timeout_neutralizes_and_turns_off_ac(self):
        """Regression: a dead sensor kept the room cooling indefinitely."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=35), conditioning_for=timedelta(minutes=35)
        )

        await parent._async_control_real_climate()

        _assert_neutralized(hass, parent, living)
        assert len(_living_cover_zero_calls(hass)) == 1
        assert len(_turn_off_calls(hass)) == 1

    @pytest.mark.asyncio
    async def test_quiet_before_conditioning_gets_full_window(self):
        """A sensor quiet for hours isn't condemned the moment cooling starts.

        Silence is counted from when conditioning began, otherwise every room
        would be flagged on the first cycle after a steady night and the AC
        could never run.
        """
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(hours=3), conditioning_for=None
        )

        await parent._async_control_real_climate()

        _assert_trusted(parent, living)
        assert parent._room_states[living.name].conditioning_since is not None

    @pytest.mark.asyncio
    async def test_quiet_before_conditioning_then_silent_too_long(self):
        """...but the full window still applies once conditioning is underway."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(hours=3), conditioning_for=timedelta(minutes=31)
        )

        await parent._async_control_real_climate()

        _assert_neutralized(hass, parent, living)

    @pytest.mark.asyncio
    async def test_recent_report_during_long_conditioning(self):
        """A sensor that keeps reporting is trusted however long cooling runs."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=10), conditioning_for=timedelta(hours=2)
        )

        await parent._async_control_real_climate()

        _assert_trusted(parent, living)

    @pytest.mark.asyncio
    async def test_no_repeated_cover_calls(self):
        """Two past-timeout loops issue the cover-0 call at most once."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=35), conditioning_for=timedelta(minutes=35)
        )

        await parent._async_control_real_climate()
        await parent._async_control_real_climate()

        assert len(_living_cover_zero_calls(hass)) == 1

    @pytest.mark.asyncio
    async def test_warns_once_per_outage(self):
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=35), conditioning_for=timedelta(minutes=35)
        )

        with patch("custom_components.matt_thermostat.climate._LOGGER") as mock_logger:
            await parent._async_control_real_climate()
            await parent._async_control_real_climate()

        stale_warnings = [
            c for c in mock_logger.warning.call_args_list if "not reported" in c[0][0]
        ]
        assert len(stale_warnings) == 1

    @pytest.mark.asyncio
    async def test_custom_room_neutralizes(self):
        """A CUSTOM-mode room is neutralized on a stale sensor too."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=35), conditioning_for=timedelta(minutes=35)
        )
        # Put the living room's child thermostat into COOL so it becomes CUSTOM.
        parent._child_thermostats[living.name]._hvac_mode = HVACMode.COOL
        parent._room_states[living.name].mode = RoomMode.CUSTOM

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is True
        assert state.cover_pos == 0
        assert parent.calculate_fan_speed() is None

    @pytest.mark.asyncio
    async def test_secondary_room_vent_closed_when_stale(self):
        """A secondary room riding a primary-driven cycle is cut off too.

        The bedroom (secondary) sits above target so it would otherwise keep
        its vent open, cooling on a frozen reading.
        """
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 25.0,
                "sensor.office_temp": 20.0,
                "sensor.bedroom_temp": 25.0,
            },
            stale_sensors={"sensor.bedroom_temp": timedelta(minutes=35)},
            cover_positions={
                "cover.living_room_vent": 100,
                "cover.office_vent": 0,
                "cover.bedroom_vent": 100,
            },
            light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
            real_climate_action=HVACAction.COOLING,
            real_climate_state=HVACMode.COOL,
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        _set_primary_lights_on(parent)
        bedroom = parent._rooms[1]
        parent._room_states[bedroom.name] = RoomState(
            mode=RoomMode.SECONDARY,
            cover_pos=100,
            conditioning_since=dt_util.utcnow() - timedelta(minutes=35),
        )

        # The hot living room keeps the AC on, so secondary rooms are processed.
        await parent._async_control_real_climate()

        assert parent._room_states[bedroom.name].cover_pos == 0
        bedroom_cover_zero = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][0] == "cover"
            and c[0][2]["entity_id"] == "cover.bedroom_vent"
            and c[0][2]["position"] == 0
        ]
        assert len(bedroom_cover_zero) == 1


class TestIdleTimeout:
    """While nothing is being conditioned, only hours of silence is stale."""

    @pytest.mark.asyncio
    async def test_quiet_sensor_in_steady_room_still_drives_ac(self):
        """Regression: a report-on-change sensor quiet for hours left the AC
        idle with a primary room above target."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(hours=4),
            real_climate_action=HVACAction.OFF,
            living_cover_pos=0,
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is False
        assert state.sensor_stale_warned is False
        assert parent.calculate_fan_speed() is not None
        assert _turn_off_calls(hass) == []

    @pytest.mark.asyncio
    async def test_dead_sensor_neutralized_after_long_silence(self):
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(hours=9),
            real_climate_action=HVACAction.OFF,
            living_cover_pos=0,
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is True
        assert state.sensor_stale_warned is True
        assert parent.calculate_fan_speed() is None

    @pytest.mark.asyncio
    async def test_fan_cycling_is_not_conditioning(self):
        """Fan-only circulation barely moves temps, so it gets the idle timeout."""
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(hours=1), real_climate_action=HVACAction.COOLING
        )
        parent._fan_cycle_active = True

        parent._update_conditioning_clocks()

        assert parent._room_states[living.name].conditioning_since is None
        assert parent._room_temp(living) == 25.0


class TestStaleLatch:
    """A stale room stays neutralized until its sensor reports again."""

    @pytest.mark.asyncio
    async def test_stays_neutralized_after_ac_stops(self):
        """Closing the vent / stopping the AC must not revive a dead sensor.

        Without the latch the room drops back to the 8 h idle timeout, its
        frozen hot reading restarts the AC, and 30 min later it's cut off
        again — the runaway in slow motion.
        """
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(minutes=35), conditioning_for=timedelta(minutes=35)
        )
        await parent._async_control_real_climate()
        _assert_neutralized(hass, parent, living)

        # The AC is now off and the vent closed; the sensor is still silent.
        _override_states(
            hass,
            {
                "climate.real_ac": make_state(
                    HVACMode.OFF, {"hvac_action": HVACAction.OFF}
                )
            },
        )
        hass.services.async_call.reset_mock()

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.conditioning_since is None
        assert state.is_satisfied is True
        assert parent.calculate_fan_speed() is None
        assert parent._room_temp(living) is None
        set_mode_calls = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("climate", "set_hvac_mode")
        ]
        assert set_mode_calls == []

    @pytest.mark.asyncio
    async def test_fresh_report_clears_latch_and_restores_demand(self):
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(0),
            real_climate_action=HVACAction.OFF,
            living_cover_pos=0,
        )
        # Previously latched stale at an older report.
        state = parent._room_states[living.name]
        state.is_satisfied = True
        state.cover_pos = 0
        state.sensor_stale_warned = True
        state.stale_report_at = dt_util.utcnow() - timedelta(minutes=40)

        await parent._async_control_real_climate()

        assert state.stale_report_at is None
        assert state.sensor_stale_warned is False
        assert state.is_satisfied is False
        assert state.cover_pos == 100


class TestConditioningClock:
    def _parent(self, *, action=HVACAction.COOLING, cover_pos=100):
        hass, parent, living = _make_stale_cooling_parent(
            stale_age=timedelta(0),
            real_climate_action=action,
            living_cover_pos=cover_pos,
        )
        return hass, parent, living

    def test_starts_when_cooling_with_vent_open(self):
        hass, parent, living = self._parent()
        parent._update_conditioning_clocks()
        started = parent._room_states[living.name].conditioning_since
        assert started is not None

        # and holds its original start on later polls
        parent._update_conditioning_clocks()
        assert parent._room_states[living.name].conditioning_since == started

    def test_not_conditioning_with_vent_closed(self):
        hass, parent, living = self._parent(cover_pos=0)
        parent._update_conditioning_clocks()
        assert parent._room_states[living.name].conditioning_since is None

    @pytest.mark.parametrize(
        "action", [HVACAction.IDLE, HVACAction.OFF, HVACAction.FAN]
    )
    def test_not_conditioning_unless_unit_cools_or_heats(self, action):
        hass, parent, living = self._parent(action=action)
        parent._update_conditioning_clocks()
        assert parent._room_states[living.name].conditioning_since is None

    def test_resets_when_conditioning_stops(self):
        hass, parent, living = self._parent()
        parent._update_conditioning_clocks()
        _override_states(
            hass,
            {
                "climate.real_ac": make_state(
                    HVACMode.COOL, {"hvac_action": HVACAction.IDLE}
                )
            },
        )
        parent._update_conditioning_clocks()
        assert parent._room_states[living.name].conditioning_since is None

    def test_held_across_unit_dropout(self):
        """A momentary AC disconnect must not restart the 30 min window."""
        hass, parent, living = self._parent()
        parent._update_conditioning_clocks()
        started = parent._room_states[living.name].conditioning_since
        _override_states(hass, {"climate.real_ac": make_state(STATE_UNAVAILABLE)})
        parent._update_conditioning_clocks()
        assert parent._room_states[living.name].conditioning_since == started
