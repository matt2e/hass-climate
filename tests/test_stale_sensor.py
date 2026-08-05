"""Tests for neutralizing rooms whose temperature sensor goes stale.

A sensor can keep publishing its last numeric value long after it has actually
stopped updating (dead battery, lost connectivity) without ever going
unavailable. Home Assistant bumps ``last_reported`` on every write, so its age
is exactly how long the sensor has been silent. Once that exceeds
SENSOR_STALE_TIMEOUT (20 min) the frozen reading is no longer trusted and the
room is neutralized (marked satisfied, vent closed) so a demand room can no
longer pin the AC on. A fresh report restores normal control immediately.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import STATE_ON

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_hass, make_parent, setup_feedback_switches


def _living_cover_zero_calls(hass) -> list:
    """Cover calls that close the living-room vent to position 0."""
    return [
        c
        for c in hass.services.async_call.call_args_list
        if c[0][0] == "cover"
        and c[0][2]["entity_id"] == "cover.living_room_vent"
        and c[0][2]["position"] == 0
    ]


def _set_primary_lights_on(parent) -> None:
    """Mark every light-controlled room as lit so it stays PRIMARY."""
    for room in parent._rooms:
        if room.light_entity:
            parent._room_states[room.name].light_on = True


def _make_stale_cooling_parent(
    *,
    stale_sensor: str = "sensor.living_room_temp",
    stale_age: timedelta = timedelta(minutes=25),
    real_climate_action: str = HVACAction.COOLING,
):
    """Cooling parent where the living room reports a hot but stale value.

    The living room sits above target (would demand cooling) but its sensor
    last reported ``stale_age`` ago. The office and bedroom sit comfortably
    below target with fresh readings so they never generate demand — the
    living room is the only room under test.
    """
    hass = make_hass(
        room_temps={
            "sensor.living_room_temp": 25.0,
            "sensor.office_temp": 20.0,
            "sensor.bedroom_temp": 20.0,
        },
        stale_sensors={stale_sensor: stale_age},
        cover_positions={
            "cover.living_room_vent": 100,
            "cover.office_vent": 0,
            "cover.bedroom_vent": 0,
        },
        light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
        real_climate_action=real_climate_action,
    )
    parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
    setup_feedback_switches(hass)
    _set_primary_lights_on(parent)
    return hass, parent


class TestStaleSensor:
    @pytest.mark.asyncio
    async def test_within_timeout_holds_demand(self):
        """Sensor silent for < 20 min: the reading is still trusted."""
        hass, parent = _make_stale_cooling_parent(stale_age=timedelta(minutes=19))
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        # 25 °C is above the 22 °C cooling target → normal cooling demand.
        assert state.is_satisfied is False
        assert state.cover_pos == 100
        assert state.sensor_stale_warned is False
        assert _living_cover_zero_calls(hass) == []
        assert parent.calculate_fan_speed() is not None

    @pytest.mark.asyncio
    async def test_past_timeout_neutralizes_and_turns_off_ac(self):
        """Sensor silent for > 20 min: room is neutralized and the AC turns off.

        Direct regression test for a frozen-but-not-unavailable sensor pinning
        the AC on indefinitely.
        """
        hass, parent = _make_stale_cooling_parent()
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is True
        assert state.cover_pos == 0
        assert state.sensor_stale_warned is True
        assert len(_living_cover_zero_calls(hass)) == 1
        # With every room satisfied, the AC turns off.
        assert parent.calculate_fan_speed() is None
        turn_off_calls = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("climate", "turn_off")
        ]
        assert len(turn_off_calls) == 1

    @pytest.mark.asyncio
    async def test_recovery_restores_demand(self):
        """A fresh report clears the stale flag and resumes normal control."""
        hass, parent = _make_stale_cooling_parent(
            stale_age=timedelta(0), real_climate_action=HVACAction.IDLE
        )
        living = parent._rooms[0]
        # Previously neutralized after the sensor went stale.
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=True,
            cover_pos=0,
            light_on=True,
            sensor_stale_warned=True,
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.sensor_stale_warned is False
        # 25 °C is above the 22 °C cooling target → demand restored.
        assert state.is_satisfied is False
        assert state.cover_pos == 100

    @pytest.mark.asyncio
    async def test_no_repeated_cover_calls(self):
        """Two past-timeout loops issue the cover-0 call at most once."""
        hass, parent = _make_stale_cooling_parent()
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
        )

        await parent._async_control_real_climate()
        await parent._async_control_real_climate()

        assert len(_living_cover_zero_calls(hass)) == 1

    @pytest.mark.asyncio
    async def test_warns_once_per_outage(self):
        """A stale sensor is logged exactly once across repeated loops."""
        hass, parent = _make_stale_cooling_parent()
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
        )

        with patch("custom_components.matt_thermostat.climate._LOGGER") as mock_logger:
            await parent._async_control_real_climate()
            await parent._async_control_real_climate()

        stale_warnings = [
            c for c in mock_logger.warning.call_args_list if "not reported" in c[0][0]
        ]
        assert len(stale_warnings) == 1
        assert parent._room_states[living.name].sensor_stale_warned is True

    @pytest.mark.asyncio
    async def test_secondary_room_vent_closed_when_stale(self):
        """A secondary room's stuck vent is closed once its sensor goes stale.

        Secondary rooms can't pin the AC on, but a frozen sensor could leave
        their vent open and waste airflow while primaries run the AC.
        """
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 25.0,
                "sensor.office_temp": 20.0,
                "sensor.bedroom_temp": 21.0,
            },
            stale_sensors={"sensor.bedroom_temp": timedelta(minutes=25)},
            cover_positions={
                "cover.living_room_vent": 100,
                "cover.office_vent": 0,
                "cover.bedroom_vent": 100,
            },
            light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        _set_primary_lights_on(parent)
        bedroom = parent._rooms[1]
        parent._room_states[bedroom.name] = RoomState(
            mode=RoomMode.SECONDARY,
            cover_pos=100,
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

    @pytest.mark.asyncio
    async def test_custom_room_neutralizes(self):
        """A CUSTOM-mode room is neutralized on a stale sensor too."""
        hass, parent = _make_stale_cooling_parent()
        living = parent._rooms[0]
        # Put the living room's child thermostat into COOL so it becomes CUSTOM.
        parent._child_thermostats[living.name]._hvac_mode = HVACMode.COOL
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.CUSTOM,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is True
        assert state.cover_pos == 0
        assert parent.calculate_fan_speed() is None
