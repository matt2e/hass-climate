"""Tests for neutralizing rooms whose temperature sensor goes unavailable.

A demand room (primary/custom) with an unavailable sensor used to freeze at
is_satisfied=False and pin the AC on forever. After a 5-minute grace period the
room is now neutralized (marked satisfied, vent closed) so it stops holding the
AC on; brief dropouts under the grace window keep the room's demand.
"""

from __future__ import annotations

from datetime import datetime, timedelta

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


def _make_cooling_parent(
    *,
    unavailable_sensor: str = "sensor.living_room_temp",
    real_climate_action: str = HVACAction.COOLING,
):
    """Cooling parent where the living room's sensor is unavailable.

    The office and bedroom sit comfortably below target so they never
    generate demand — the living room is the only room under test.
    """
    hass = make_hass(
        room_temps={
            "sensor.office_temp": 20.0,
            "sensor.bedroom_temp": 20.0,
        },
        unavailable_sensors=[unavailable_sensor],
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


class TestUnavailableSensorGrace:
    @pytest.mark.asyncio
    async def test_within_grace_holds_demand(self):
        """Sensor out for < 5 min: room keeps demanding, not neutralized."""
        hass, parent = _make_cooling_parent()
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
            sensor_unavailable_since=datetime.now() - timedelta(minutes=2),
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is False
        assert state.sensor_unavailable_since is not None
        assert _living_cover_zero_calls(hass) == []
        # Room still holds the AC on.
        assert parent.calculate_fan_speed() is not None

    @pytest.mark.asyncio
    async def test_past_grace_neutralizes_and_turns_off_ac(self):
        """Sensor out for > 5 min: room is neutralized and the AC turns off.

        Direct regression test for the reported incident.
        """
        hass, parent = _make_cooling_parent()
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
            sensor_unavailable_since=datetime.now() - timedelta(minutes=6),
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is True
        assert state.cover_pos == 0
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
    async def test_timer_starts_on_first_dropout(self):
        """First unavailable read starts the timer without neutralizing."""
        hass, parent = _make_cooling_parent(real_climate_action=HVACAction.IDLE)
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
        )
        assert parent._room_states[living.name].sensor_unavailable_since is None

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.sensor_unavailable_since is not None
        # Not yet neutralized within the grace period.
        assert state.is_satisfied is False
        assert _living_cover_zero_calls(hass) == []

    @pytest.mark.asyncio
    async def test_recovery_clears_timer(self):
        """When the sensor returns, the timer clears and control resumes."""
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 25.0,
                "sensor.office_temp": 20.0,
                "sensor.bedroom_temp": 20.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.office_vent": 0,
                "cover.bedroom_vent": 0,
            },
            light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        _set_primary_lights_on(parent)
        living = parent._rooms[0]
        # Previously neutralized after a long dropout.
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=True,
            cover_pos=0,
            light_on=True,
            sensor_unavailable_since=datetime.now() - timedelta(minutes=6),
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.sensor_unavailable_since is None
        # 25 °C is above the 22 °C cooling target → demand restored.
        assert state.is_satisfied is False
        assert state.cover_pos == 100

    @pytest.mark.asyncio
    async def test_no_repeated_cover_calls(self):
        """Two past-grace loops issue the cover-0 call at most once."""
        hass, parent = _make_cooling_parent()
        living = parent._rooms[0]
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
            sensor_unavailable_since=datetime.now() - timedelta(minutes=6),
        )

        await parent._async_control_real_climate()
        await parent._async_control_real_climate()

        assert len(_living_cover_zero_calls(hass)) == 1

    @pytest.mark.asyncio
    async def test_secondary_room_vent_closed_past_grace(self):
        """A secondary room's stuck vent is closed once its sensor stays out.

        Secondary rooms can't pin the AC on, but a dead sensor could leave
        their vent open and waste airflow while primaries run the AC.
        """
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 25.0,
                "sensor.office_temp": 20.0,
            },
            unavailable_sensors=["sensor.bedroom_temp"],
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
            sensor_unavailable_since=datetime.now() - timedelta(minutes=6),
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
        """A CUSTOM-mode room is neutralized just like a PRIMARY one."""
        hass, parent = _make_cooling_parent()
        living = parent._rooms[0]
        # Put the living room's child thermostat into COOL so it becomes CUSTOM.
        parent._child_thermostats[living.name]._hvac_mode = HVACMode.COOL
        parent._room_states[living.name] = RoomState(
            mode=RoomMode.CUSTOM,
            is_satisfied=False,
            cover_pos=100,
            light_on=True,
            sensor_unavailable_since=datetime.now() - timedelta(minutes=6),
        )

        await parent._async_control_real_climate()

        state = parent._room_states[living.name]
        assert state.is_satisfied is True
        assert state.cover_pos == 0
        assert parent.calculate_fan_speed() is None
