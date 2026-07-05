"""Tests for _get_room_temp — averaging across a room's sensors."""

from __future__ import annotations

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from custom_components.matt_thermostat.climate import Room, RoomMode

from .conftest import make_hass, make_parent, make_state


def _room(sensors: list[str]) -> Room:
    return Room(
        name="Multi",
        sensor_entities=sensors,
        cover_entity="cover.multi_vent",
        light_entity=None,
        standard_mode=RoomMode.PRIMARY,
        bedtime_mode=RoomMode.PRIMARY,
        allows_override=False,
        is_overflow=False,
        vents=1,
    )


class TestGetRoomTemp:
    def test_single_sensor(self):
        hass = make_hass(room_temps={"sensor.a": 22.0})
        parent = make_parent(hass)
        assert parent._get_room_temp(_room(["sensor.a"])) == 22.0

    def test_average_of_multiple_sensors(self):
        hass = make_hass(room_temps={"sensor.a": 22.0, "sensor.b": 24.0})
        parent = make_parent(hass)
        assert parent._get_room_temp(_room(["sensor.a", "sensor.b"])) == 23.0

    def test_average_of_three_sensors(self):
        hass = make_hass(
            room_temps={"sensor.a": 21.0, "sensor.b": 22.0, "sensor.c": 26.0}
        )
        parent = make_parent(hass)
        assert (
            parent._get_room_temp(_room(["sensor.a", "sensor.b", "sensor.c"])) == 23.0
        )

    def test_unavailable_sensor_ignored(self):
        hass = make_hass(room_temps={"sensor.a": 22.0})
        parent = make_parent(hass)
        original_get = hass.states.get
        hass.states.get = lambda eid: (
            make_state(STATE_UNAVAILABLE) if eid == "sensor.b" else original_get(eid)
        )
        assert parent._get_room_temp(_room(["sensor.a", "sensor.b"])) == 22.0

    def test_unknown_sensor_ignored(self):
        hass = make_hass(room_temps={"sensor.a": 22.0})
        parent = make_parent(hass)
        original_get = hass.states.get
        hass.states.get = lambda eid: (
            make_state(STATE_UNKNOWN) if eid == "sensor.b" else original_get(eid)
        )
        assert parent._get_room_temp(_room(["sensor.a", "sensor.b"])) == 22.0

    def test_missing_sensor_ignored(self):
        hass = make_hass(room_temps={"sensor.a": 22.0})
        parent = make_parent(hass)
        assert parent._get_room_temp(_room(["sensor.a", "sensor.missing"])) == 22.0

    def test_no_usable_sensors_returns_none(self):
        hass = make_hass()
        parent = make_parent(hass)
        original_get = hass.states.get
        hass.states.get = lambda eid: (
            make_state(STATE_UNAVAILABLE) if eid == "sensor.a" else original_get(eid)
        )
        assert parent._get_room_temp(_room(["sensor.a", "sensor.missing"])) is None
