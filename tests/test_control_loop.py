"""Tests for the main control loop (_async_control_real_climate)."""

from __future__ import annotations

import pytest
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import (
    ATTR_TEMPERATURE,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)

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


class TestControlLoopSealsVentsWhenIdle:
    @pytest.mark.asyncio
    async def test_idle_closes_all_open_vents(self):
        """AC turns off with vents left open → every vent is sealed to 0."""
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 21.0,
                "sensor.bedroom_temp": 21.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions={
                "cover.living_room_vent": 100,
                "cover.bedroom_vent": 50,
                "cover.office_vent": 100,
            },
            light_states={
                "light.living_room": STATE_ON,
                "light.office": STATE_ON,
            },
            real_climate_action=HVACAction.COOLING,
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        setup_feedback_switches(hass)
        for room in parent._rooms:
            parent._room_states[room.name] = RoomState(
                mode=RoomMode.PRIMARY, is_satisfied=True, cover_pos=100
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

        # Every room should have its vent driven to 0 by the end of the cycle.
        for room in parent._rooms:
            closes = [
                c
                for c in hass.services.async_call.call_args_list
                if c[0][:2] == ("cover", "set_cover_position")
                and c[0][2]["entity_id"] == room.cover_entity
            ]
            assert closes, f"no cover call for {room.cover_entity}"
            assert closes[-1][0][2]["position"] == 0

    @pytest.mark.asyncio
    async def test_secondary_vent_sealed_when_ac_stops(self):
        """A secondary room left open from a prior cycle is sealed when idle."""
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 21.0,
                "sensor.bedroom_temp": 21.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions={
                "cover.living_room_vent": 0,
                "cover.bedroom_vent": 100,
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
        for room in parent._rooms:
            parent._room_states[room.name] = RoomState(
                mode=RoomMode.PRIMARY, is_satisfied=True
            )
            if room.light_entity:
                parent._room_states[room.name].light_on = True
        # Bedroom is the secondary "bathroom" left open by a prior on-cycle.
        parent._room_states["Bedroom"].cover_pos = 100

        await parent._async_control_real_climate()

        bedroom_closes = [
            c
            for c in hass.services.async_call.call_args_list
            if c[0][:2] == ("cover", "set_cover_position")
            and c[0][2]["entity_id"] == "cover.bedroom_vent"
        ]
        assert bedroom_closes
        assert bedroom_closes[-1][0][2]["position"] == 0

    @pytest.mark.asyncio
    async def test_no_redundant_cover_writes_when_all_closed(self):
        """With every vent already closed, the idle path issues no cover calls."""
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 21.0,
                "sensor.bedroom_temp": 21.0,
                "sensor.office_temp": 21.0,
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
        for room in parent._rooms:
            parent._room_states[room.name] = RoomState(
                mode=RoomMode.PRIMARY, is_satisfied=True
            )
            if room.light_entity:
                parent._room_states[room.name].light_on = True

        await parent._async_control_real_climate()

        cover_calls = [
            c for c in hass.services.async_call.call_args_list if c[0][0] == "cover"
        ]
        assert cover_calls == []


class TestTooHotWhileHeating:
    @pytest.mark.asyncio
    async def test_too_hot_above_target_within_tolerance_turns_off_ac(self):
        """Room at 19.4, target 19, heating mode, too_hot pressed → AC should turn off.

        The room is above the target but within the grace zone
        (target ± tolerance = 18.6–19.4). The 'too hot' opposing feedback
        should mark rooms satisfied and the AC should turn off, NOT snap the
        target temperature down.
        """
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 19.4,
                "sensor.bedroom_temp": 19.4,
                "sensor.office_temp": 19.4,
            },
            cover_positions={
                "cover.living_room_vent": 100,
                "cover.bedroom_vent": 0,
                "cover.office_vent": 100,
            },
            light_states={
                "light.living_room": STATE_ON,
                "light.office": STATE_ON,
            },
            real_climate_action=HVACAction.HEATING,
        )
        parent = make_parent(hass, target_temp=19.0, hvac_mode=HVACMode.HEAT)
        too_hot, _too_cold = setup_feedback_switches(hass)
        too_hot._is_on = True

        # Pre-set rooms as not satisfied (AC is actively heating)
        for room in parent._rooms:
            if room.light_entity:
                parent._room_states[room.name].light_on = True

        await parent._async_control_real_climate()

        # Target temp should NOT have been snapped down
        assert parent._target_temp == 19.0

        # AC should have been turned off (all rooms satisfied)
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


class TestDeviceActiveLatch:
    """`_is_device_active` holds its last trusted reading across AC dropouts.

    The real AC drops offline briefly but often. During a dropout the state
    object carries no attributes, so we must not recompute "is it running?" —
    we latch the last reading taken while the entity was available and hold it
    indefinitely.
    """

    def test_holds_active_reading_when_unavailable(self):
        parent = make_parent(
            make_hass(real_climate_action=HVACAction.COOLING),
            hvac_mode=HVACMode.COOL,
        )
        # A read while available latches "active".
        assert parent._is_device_active is True

        # AC drops offline: the entity goes unavailable, attributes vanish.
        parent.hass = make_hass(real_climate_state=STATE_UNAVAILABLE)
        assert parent._is_device_active is True

    def test_holds_inactive_reading_when_unknown(self):
        parent = make_parent(
            make_hass(real_climate_action=HVACAction.IDLE),
            hvac_mode=HVACMode.COOL,
        )
        assert parent._is_device_active is False

        parent.hass = make_hass(real_climate_state=STATE_UNKNOWN)
        assert parent._is_device_active is False

    def test_defaults_inactive_before_first_good_reading(self):
        parent = make_parent(
            make_hass(real_climate_state=STATE_UNAVAILABLE),
            hvac_mode=HVACMode.COOL,
        )
        assert parent._is_device_active is False

    def test_latch_updates_when_ac_comes_back(self):
        parent = make_parent(
            make_hass(real_climate_action=HVACAction.COOLING),
            hvac_mode=HVACMode.COOL,
        )
        assert parent._is_device_active is True

        # Dropout holds "active"...
        parent.hass = make_hass(real_climate_state=STATE_UNAVAILABLE)
        assert parent._is_device_active is True

        # ...then the AC returns, genuinely idle: the latch advances.
        parent.hass = make_hass(real_climate_action=HVACAction.IDLE)
        assert parent._is_device_active is False


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
