"""Tests for the main control loop (_async_control_real_climate)."""

from __future__ import annotations

import pytest
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, STATE_OFF, STATE_ON

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_hass, make_parent, setup_feedback_switches


class TestControlLoopManualMode:
    @pytest.mark.asyncio
    async def test_manual_mode_resets_rooms_and_returns(self):
        hass = make_hass(
            manual=STATE_ON,
            room_temps={
                "sensor.living_room_temp": 24.0,
                "sensor.bedroom_temp": 22.0,
                "sensor.office_temp": 23.0,
            },
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)

        # Set some room state first
        parent._room_states[parent._rooms[0].name].is_satisfied = True

        await parent._async_control_real_climate()

        # All rooms should be reset
        for _name, state in parent._room_states.items():
            assert state.mode == RoomMode.DISABLED
            assert state.is_satisfied is False


class TestControlLoopNotPresent:
    @pytest.mark.asyncio
    async def test_not_present_disables_rooms_but_still_runs(self):
        hass = make_hass(
            presence=STATE_OFF,
            room_temps={
                "sensor.living_room_temp": 24.0,
                "sensor.bedroom_temp": 22.0,
                "sensor.office_temp": 23.0,
            },
            cover_positions={
                "cover.living_room_vent": 50,
                "cover.bedroom_vent": 50,
                "cover.office_vent": 50,
            },
            light_states={
                "light.living_room": STATE_OFF,
                "light.office": STATE_OFF,
            },
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)

        await parent._async_control_real_climate()

        # Should have called cover set_cover_position to close disabled room vents
        cover_calls = [
            c for c in hass.services.async_call.call_args_list if c[0][0] == "cover"
        ]
        assert len(cover_calls) > 0
        # All cover calls should set position to 0
        for c in cover_calls:
            assert c[0][2]["position"] == 0


class TestControlLoopCooling:
    @pytest.mark.asyncio
    async def test_cooling_sets_real_climate_temp(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 25.0,
                "sensor.bedroom_temp": 24.0,
                "sensor.office_temp": 25.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
            light_states={
                "light.living_room": STATE_ON,
                "light.office": STATE_ON,
            },
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        # Pre-set lights as on to make rooms primary
        for room in parent._rooms:
            if room.light_entity:
                parent._room_states[room.name].light_on = True

        await parent._async_control_real_climate()

        # Check climate.set_temperature was called
        set_temp_calls = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("climate", "set_temperature")
        ]
        assert len(set_temp_calls) == 1
        # target 22.0 → floor(22.0) + modifier(0.0) = 22.0
        assert set_temp_calls[0][0][2][ATTR_TEMPERATURE] == 22.0

    @pytest.mark.asyncio
    async def test_cooling_with_modifier(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 25.0,
                "sensor.bedroom_temp": 24.0,
                "sensor.office_temp": 25.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
            light_states={
                "light.living_room": STATE_ON,
                "light.office": STATE_ON,
            },
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_parent(
            hass,
            target_temp=22.5,
            hvac_mode=HVACMode.COOL,
            cooling_temp_modifier=-0.5,
        )
        setup_feedback_switches(hass)
        for room in parent._rooms:
            if room.light_entity:
                parent._room_states[room.name].light_on = True

        await parent._async_control_real_climate()

        set_temp_calls = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("climate", "set_temperature")
        ]
        # floor(22.5) + (-0.5) = 22 - 0.5 = 21.5
        assert set_temp_calls[0][0][2][ATTR_TEMPERATURE] == 21.5

    @pytest.mark.asyncio
    async def test_heating_with_modifier(self):
        hass = make_hass(
            bedtime=STATE_ON,
            room_temps={
                "sensor.living_room_temp": 19.0,
                "sensor.bedroom_temp": 19.0,
                "sensor.office_temp": 19.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
            light_states={
                "light.living_room": STATE_OFF,
                "light.office": STATE_OFF,
            },
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_parent(
            hass,
            target_temp=22.0,
            hvac_mode=HVACMode.HEAT,
            heating_temp_modifier=0.5,
        )
        setup_feedback_switches(hass)

        await parent._async_control_real_climate()

        set_temp_calls = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("climate", "set_temperature")
        ]
        # ceil(22.0) + 0.5 = 22.5
        assert set_temp_calls[0][0][2][ATTR_TEMPERATURE] == 22.5


class TestControlLoopTurnsOffAC:
    @pytest.mark.asyncio
    async def test_all_satisfied_turns_off_ac(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 22.0,
                "sensor.office_temp": 22.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
            light_states={
                "light.living_room": STATE_ON,
                "light.office": STATE_ON,
            },
            real_climate_action=HVACAction.COOLING,
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        # Mark all rooms as satisfied
        for room in parent._rooms:
            parent._room_states[room.name] = RoomState(
                mode=RoomMode.PRIMARY, is_satisfied=True
            )
            if room.light_entity:
                parent._room_states[room.name].light_on = True

        await parent._async_control_real_climate()

        turn_off_calls = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("climate", "turn_off")
        ]
        assert len(turn_off_calls) == 1


class TestSecondaryTemperature:
    def test_secondary_temp_cooling(self):
        parent = make_parent(target_temp=22.0, hvac_mode=HVACMode.COOL)
        assert parent._target_secondary_temp() == 24.0

    def test_secondary_temp_heating(self):
        parent = make_parent(target_temp=22.0, hvac_mode=HVACMode.HEAT)
        assert parent._target_secondary_temp() == 20.0

    def test_secondary_temp_cooling_capped_at_28(self):
        parent = make_parent(target_temp=27.0, hvac_mode=HVACMode.COOL)
        assert parent._target_secondary_temp() == 28.0

    def test_secondary_temp_heating_capped_at_16(self):
        parent = make_parent(target_temp=17.0, hvac_mode=HVACMode.HEAT)
        assert parent._target_secondary_temp() == 16.0

    def test_secondary_temp_off_returns_target(self):
        parent = make_parent(target_temp=22.0, hvac_mode=HVACMode.OFF)
        assert parent._target_secondary_temp() == 22.0


class TestHvacAction:
    def test_off_mode(self):
        parent = make_parent(hvac_mode=HVACMode.OFF)
        assert parent.hvac_action == HVACAction.OFF

    def test_cool_mode_active(self):
        hass = make_hass(real_climate_action=HVACAction.COOLING)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        assert parent.hvac_action == HVACAction.COOLING

    def test_heat_mode_active(self):
        hass = make_hass(real_climate_action=HVACAction.HEATING)
        parent = make_parent(hass, hvac_mode=HVACMode.HEAT)
        assert parent.hvac_action == HVACAction.HEATING

    def test_fan_only_active(self):
        hass = make_hass(real_climate_action=HVACAction.FAN)
        parent = make_parent(hass, hvac_mode=HVACMode.FAN_ONLY)
        assert parent.hvac_action == HVACAction.FAN

    def test_idle_when_device_not_active(self):
        hass = make_hass(real_climate_action=HVACAction.IDLE)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        assert parent.hvac_action == HVACAction.IDLE


class TestSetHvacMode:
    @pytest.mark.asyncio
    async def test_set_valid_mode(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 22.0,
                "sensor.office_temp": 22.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
            light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
        )
        parent = make_parent(hass, hvac_mode=HVACMode.OFF)
        setup_feedback_switches(hass)
        await parent.async_set_hvac_mode(HVACMode.COOL)
        assert parent.hvac_mode == HVACMode.COOL
        parent.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_set_invalid_mode_ignored(self):
        parent = make_parent(hvac_mode=HVACMode.COOL)
        await parent.async_set_hvac_mode(HVACMode.AUTO)
        assert parent.hvac_mode == HVACMode.COOL


class TestSetTemperature:
    @pytest.mark.asyncio
    async def test_set_temperature(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 22.0,
                "sensor.office_temp": 22.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
            light_states={"light.living_room": STATE_ON, "light.office": STATE_ON},
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        await parent.async_set_temperature(**{ATTR_TEMPERATURE: 24.0})
        assert parent.target_temperature == 24.0

    @pytest.mark.asyncio
    async def test_set_temperature_no_attr(self):
        parent = make_parent(target_temp=22.0)
        await parent.async_set_temperature(wrong_key=24.0)
        assert parent.target_temperature == 22.0


class TestMinMaxTemp:
    def test_custom_min_max(self):
        parent = make_parent()
        assert parent.min_temp == 16.0
        assert parent.max_temp == 30.0
