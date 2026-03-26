"""Tests for room mode calculation."""

from __future__ import annotations

from datetime import datetime, timedelta

from homeassistant.components.climate import HVACMode
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE

from custom_components.matt_thermostat.climate import RoomMode

from .conftest import make_hass, make_parent, make_rooms


class TestCalculateRoomMode:
    def test_no_presence_returns_disabled(self):
        hass = make_hass(presence=STATE_OFF)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = make_rooms()
        assert (
            parent._calculate_room_mode(rooms[0], presence=False) == RoomMode.DISABLED
        )

    def test_hvac_off_returns_disabled(self):
        parent = make_parent(hvac_mode=HVACMode.OFF)
        rooms = make_rooms()
        assert parent._calculate_room_mode(rooms[0], presence=True) == RoomMode.DISABLED

    def test_primary_room_no_light_entity(self):
        """Bedroom has no light entity, secondary by default, primary at bedtime."""
        hass = make_hass(bedtime=STATE_ON)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        # Bedroom: bedtime_mode=primary, no light
        assert parent._calculate_room_mode(rooms[1], presence=True) == RoomMode.PRIMARY

    def test_secondary_room_stays_secondary(self):
        hass = make_hass(bedtime=STATE_OFF)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        # Bedroom: standard_mode=secondary
        assert (
            parent._calculate_room_mode(rooms[1], presence=True) == RoomMode.SECONDARY
        )

    def test_primary_room_light_on_long_enough(self):
        """Room with light on for >2 min should be PRIMARY."""
        hass = make_hass(
            bedtime=STATE_OFF,
            light_states={"light.living_room": STATE_ON},
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        # Pre-set light state to have been on long enough
        parent._room_states[rooms[0].name].raw_light_on_at = datetime.now() - timedelta(
            minutes=3
        )
        parent._room_states[rooms[0].name].light_on = True

        result = parent._calculate_room_mode(rooms[0], presence=True)
        assert result == RoomMode.PRIMARY

    def test_primary_room_light_off_demotes_to_secondary(self):
        """Room with light off for >2 min should be SECONDARY."""
        hass = make_hass(
            bedtime=STATE_OFF,
            light_states={"light.living_room": STATE_OFF},
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        # Light has been off for a while
        parent._room_states[rooms[0].name].raw_light_off_at = (
            datetime.now() - timedelta(minutes=3)
        )
        parent._room_states[rooms[0].name].light_on = False

        result = parent._calculate_room_mode(rooms[0], presence=True)
        assert result == RoomMode.SECONDARY

    def test_primary_room_light_recently_turned_off_stays_primary(self):
        """Light off for <2 min — light_on stays True, room remains PRIMARY."""
        hass = make_hass(
            bedtime=STATE_OFF,
            light_states={"light.living_room": STATE_OFF},
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name].raw_light_off_at = (
            datetime.now() - timedelta(seconds=30)
        )
        parent._room_states[rooms[0].name].light_on = True

        result = parent._calculate_room_mode(rooms[0], presence=True)
        assert result == RoomMode.PRIMARY

    def test_bedtime_changes_mode(self):
        """Living Room: standard=primary, bedtime=secondary."""
        hass = make_hass(bedtime=STATE_ON)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        assert (
            parent._calculate_room_mode(rooms[0], presence=True) == RoomMode.SECONDARY
        )

    def test_bedtime_disables_office(self):
        """Office: bedtime_mode=disabled."""
        hass = make_hass(bedtime=STATE_ON)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        assert parent._calculate_room_mode(rooms[2], presence=True) == RoomMode.DISABLED

    def test_child_thermostat_override_to_custom(self):
        """When child thermostat is in COOL mode, room becomes CUSTOM."""
        hass = make_hass(bedtime=STATE_OFF)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        # Set living room child to COOL mode
        parent._child_thermostats["Living Room"]._hvac_mode = HVACMode.COOL
        assert parent._calculate_room_mode(rooms[0], presence=True) == RoomMode.CUSTOM

    def test_child_thermostat_auto_does_not_override(self):
        """When child thermostat is in AUTO mode, parent mode is used."""
        hass = make_hass(
            bedtime=STATE_OFF,
            light_states={"light.living_room": STATE_ON},
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._child_thermostats["Living Room"]._hvac_mode = HVACMode.AUTO
        parent._room_states[rooms[0].name].light_on = True
        parent._room_states[rooms[0].name].raw_light_on_at = datetime.now() - timedelta(
            minutes=3
        )
        result = parent._calculate_room_mode(rooms[0], presence=True)
        assert result == RoomMode.PRIMARY

    def test_light_unavailable_treated_as_off(self):
        hass = make_hass(
            bedtime=STATE_OFF,
            light_states={"light.living_room": STATE_UNAVAILABLE},
        )
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name].raw_light_off_at = (
            datetime.now() - timedelta(minutes=3)
        )
        parent._room_states[rooms[0].name].light_on = False
        result = parent._calculate_room_mode(rooms[0], presence=True)
        assert result == RoomMode.SECONDARY


class TestTransitionRoomToMode:
    def test_transition_to_same_mode_noop(self):
        parent = make_parent(hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name].mode = RoomMode.PRIMARY
        parent._room_states[rooms[0].name].is_satisfied = True
        parent._transition_room_to_mode(rooms[0], RoomMode.PRIMARY)
        # Should not have changed satisfaction
        assert parent._room_states[rooms[0].name].is_satisfied is True

    def test_transition_to_disabled_resets_satisfaction(self):
        """Transitioning from PRIMARY to DISABLED resets satisfaction."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 24.0})
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name].mode = RoomMode.PRIMARY
        parent._room_states[rooms[0].name].is_satisfied = True
        parent._transition_room_to_mode(rooms[0], RoomMode.DISABLED)
        assert parent._room_states[rooms[0].name].is_satisfied is False

    def test_transition_to_primary_cooling_satisfied(self):
        """Room at 21.0 with target 22.0 + hot_tolerance 0.4 → satisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 21.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._transition_room_to_mode(rooms[0], RoomMode.PRIMARY)
        assert parent._room_states[rooms[0].name].is_satisfied is True

    def test_transition_to_primary_cooling_not_satisfied(self):
        """Room at 25.0 with target 22.0 + hot_tolerance 0.4 → not satisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 25.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._transition_room_to_mode(rooms[0], RoomMode.PRIMARY)
        assert parent._room_states[rooms[0].name].is_satisfied is False

    def test_transition_to_primary_heating_satisfied(self):
        """Room at 23.0 with target 22.0 - cold_tolerance 0.4 → satisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 23.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        parent._transition_room_to_mode(rooms[0], RoomMode.PRIMARY)
        assert parent._room_states[rooms[0].name].is_satisfied is True

    def test_transition_sensor_unavailable_skips(self):
        """If sensor is unavailable, transition doesn't crash or change satisfaction."""
        hass = make_hass()
        # No temp set for living room sensor → states.get returns None
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name].is_satisfied = True
        parent._transition_room_to_mode(rooms[0], RoomMode.PRIMARY)
        # mode changed but satisfaction untouched because sensor unavailable
        assert parent._room_states[rooms[0].name].mode == RoomMode.PRIMARY
