"""Tests for Room dataclass and RoomMode enum."""

import pytest

from custom_components.matt_thermostat.climate import Room, RoomMode


class TestRoomMode:
    def test_values(self):
        assert RoomMode.PRIMARY == "primary"
        assert RoomMode.SECONDARY == "secondary"
        assert RoomMode.DISABLED == "disabled"
        assert RoomMode.CUSTOM == "custom"


class TestRoomFromDict:
    def test_valid_primary_room(self):
        data = {
            "name": "Living Room",
            "sensor": "sensor.living_room_temp",
            "cover": "cover.living_room_vent",
            "light": "light.living_room",
            "mode": "primary",
            "bedtime_mode": "secondary",
            "allows_override": True,
            "is_overflow": False,
            "vents": 2,
        }
        room = Room.from_dict(data)
        assert room.name == "Living Room"
        assert room.sensor_entity == "sensor.living_room_temp"
        assert room.cover_entity == "cover.living_room_vent"
        assert room.light_entity == "light.living_room"
        assert room.standard_mode == RoomMode.PRIMARY
        assert room.bedtime_mode == RoomMode.SECONDARY
        assert room.allows_override is True
        assert room.is_overflow is False
        assert room.vents == 2

    def test_minimal_room(self):
        data = {
            "name": "Closet",
            "sensor": "sensor.closet_temp",
            "cover": "cover.closet_vent",
            "mode": "disabled",
            "bedtime_mode": "disabled",
        }
        room = Room.from_dict(data)
        assert room.name == "Closet"
        assert room.light_entity is None
        assert room.allows_override is False
        assert room.is_overflow is False
        assert room.vents == 1

    def test_missing_name_raises(self):
        with pytest.raises(ValueError):
            Room.from_dict(
                {
                    "sensor": "s",
                    "cover": "c",
                    "mode": "primary",
                    "bedtime_mode": "primary",
                }
            )

    def test_missing_cover_raises(self):
        with pytest.raises(ValueError, match="must have"):
            Room.from_dict(
                {
                    "name": "Room",
                    "sensor": "s",
                    "mode": "primary",
                    "bedtime_mode": "primary",
                }
            )

    def test_missing_sensor_raises(self):
        with pytest.raises(ValueError, match="must have"):
            Room.from_dict(
                {
                    "name": "Room",
                    "cover": "c",
                    "mode": "primary",
                    "bedtime_mode": "primary",
                }
            )

    def test_missing_mode_raises(self):
        with pytest.raises(ValueError, match="must have"):
            Room.from_dict(
                {"name": "Room", "sensor": "s", "cover": "c", "bedtime_mode": "primary"}
            )

    def test_missing_bedtime_mode_raises(self):
        with pytest.raises(ValueError, match="must have"):
            Room.from_dict(
                {"name": "Room", "sensor": "s", "cover": "c", "mode": "primary"}
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            Room.from_dict(
                {
                    "name": "Room",
                    "sensor": "s",
                    "cover": "c",
                    "mode": "turbo",
                    "bedtime_mode": "primary",
                }
            )

    def test_invalid_bedtime_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid bedtime mode"):
            Room.from_dict(
                {
                    "name": "Room",
                    "sensor": "s",
                    "cover": "c",
                    "mode": "primary",
                    "bedtime_mode": "turbo",
                }
            )

    def test_custom_mode_not_allowed_as_standard(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            Room.from_dict(
                {
                    "name": "Room",
                    "sensor": "s",
                    "cover": "c",
                    "mode": "custom",
                    "bedtime_mode": "primary",
                }
            )
