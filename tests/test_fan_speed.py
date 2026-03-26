"""Tests for fan speed calculation."""

from __future__ import annotations

from homeassistant.components.climate import HVACMode

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_parent


class TestCalculateFanSpeed:
    def test_no_unsatisfied_rooms_returns_none(self):
        parent = make_parent(hvac_mode=HVACMode.COOL)
        # All rooms disabled by default (RoomState starts with mode=DISABLED)
        assert parent.calculate_fan_speed() is None

    def test_unsatisfied_primary_room_auto(self):
        """One primary room with 2 vents at 100% in cooling = 2 * 0.5 = 1.0 → auto."""
        parent = make_parent(hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        # Living Room: primary, 2 vents
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=False
        )
        assert parent.calculate_fan_speed() == "auto"

    def test_unsatisfied_primary_rooms_medium_cooling(self):
        """Two primary rooms with enough vents for medium in cooling mode.

        Living Room: 2 vents at 100%, Office: 2 vents at 100%
        Bedroom: secondary 1 vent at 100%
        Total = 5 vents * 0.5 scale = 2.5 → still auto
        Need 6 vents for medium (6 * 0.5 = 3.0).
        """
        parent = make_parent(hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=False
        )
        parent._room_states[rooms[1].name] = RoomState(
            mode=RoomMode.SECONDARY, cover_pos=100, is_satisfied=True
        )
        parent._room_states[rooms[2].name] = RoomState(
            mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=False
        )
        # 5 vents * 0.5 = 2.5 → auto
        assert parent.calculate_fan_speed() == "auto"

    def test_medium_fan_heating(self):
        """In heating mode, scale is 1.0. 3 vents at 100% → medium."""
        parent = make_parent(hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=False
        )
        parent._room_states[rooms[1].name] = RoomState(
            mode=RoomMode.SECONDARY, cover_pos=100, is_satisfied=True
        )
        # Living room 2 vents + bedroom 1 vent = 3 vents * 1.0 = 3.0 → medium
        assert parent.calculate_fan_speed() == "medium"

    def test_high_fan_heating(self):
        """5+ vents at 1.0 scale → high."""
        parent = make_parent(hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        # All rooms primary, all open
        for room in rooms:
            parent._room_states[room.name] = RoomState(
                mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=False
            )
        # 2 + 1 + 2 = 5 vents * 1.0 = 5.0 → high
        assert parent.calculate_fan_speed() == "high"

    def test_half_open_vents(self):
        """Partially open covers contribute proportionally."""
        parent = make_parent(hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        # All rooms primary but at 50%
        for room in rooms:
            parent._room_states[room.name] = RoomState(
                mode=RoomMode.PRIMARY, cover_pos=50, is_satisfied=False
            )
        # (2*0.5 + 1*0.5 + 2*0.5) * 1.0 = 2.5 → auto
        assert parent.calculate_fan_speed() == "auto"

    def test_disabled_rooms_not_counted(self):
        parent = make_parent(hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.DISABLED, cover_pos=100, is_satisfied=False
        )
        assert parent.calculate_fan_speed() is None

    def test_satisfied_primary_rooms_dont_turn_on(self):
        """Even with vents open, if all primary rooms are satisfied, fan is off."""
        parent = make_parent(hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=True
        )
        assert parent.calculate_fan_speed() is None

    def test_custom_room_unsatisfied_turns_on(self):
        parent = make_parent(hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.CUSTOM, cover_pos=100, is_satisfied=False
        )
        assert parent.calculate_fan_speed() == "auto"

    def test_secondary_unsatisfied_does_not_turn_on(self):
        """Secondary rooms cannot trigger the AC by themselves."""
        parent = make_parent(hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        parent._room_states[rooms[1].name] = RoomState(
            mode=RoomMode.SECONDARY, cover_pos=100, is_satisfied=False
        )
        assert parent.calculate_fan_speed() is None

    def test_fan_only_uses_cooling_scale(self):
        """FAN_ONLY mode uses default scale (1.0), not the cooling 0.5 scale."""
        parent = make_parent(hvac_mode=HVACMode.FAN_ONLY)
        rooms = parent._rooms
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, cover_pos=100, is_satisfied=False
        )
        parent._room_states[rooms[1].name] = RoomState(
            mode=RoomMode.SECONDARY, cover_pos=100, is_satisfied=True
        )
        # 2 + 1 = 3 vents * 1.0 = 3.0 → medium
        assert parent.calculate_fan_speed() == "medium"
