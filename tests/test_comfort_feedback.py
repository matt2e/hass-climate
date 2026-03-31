"""Tests for comfort feedback (too hot / too cold) logic."""

from __future__ import annotations

import pytest
from homeassistant.components.climate import HVACMode

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_hass, make_parent, setup_feedback_switches


class TestCaptureAndResetFeedback:
    @pytest.mark.asyncio
    async def test_no_switches_returns_false_false(self):
        parent = make_parent()
        result = await parent._capture_and_reset_feedback()
        assert result == (False, False)

    @pytest.mark.asyncio
    async def test_too_hot_on_returns_true_false_and_resets(self):
        hass = make_hass()
        parent = make_parent(hass)
        too_hot, too_cold = setup_feedback_switches(hass)
        too_hot._is_on = True

        result = await parent._capture_and_reset_feedback()
        assert result == (True, False)
        assert too_hot.is_on is False

    @pytest.mark.asyncio
    async def test_too_cold_on_returns_false_true_and_resets(self):
        hass = make_hass()
        parent = make_parent(hass)
        too_hot, too_cold = setup_feedback_switches(hass)
        too_cold._is_on = True

        result = await parent._capture_and_reset_feedback()
        assert result == (False, True)
        assert too_cold.is_on is False

    @pytest.mark.asyncio
    async def test_both_on_cancels_out(self):
        hass = make_hass()
        parent = make_parent(hass)
        too_hot, too_cold = setup_feedback_switches(hass)
        too_hot._is_on = True
        too_cold._is_on = True

        result = await parent._capture_and_reset_feedback()
        assert result == (False, False)
        assert too_hot.is_on is False
        assert too_cold.is_on is False


class TestApplyComfortFeedback:
    """Test _apply_comfort_feedback logic."""

    def test_aligned_cooling_too_hot_forces_unsatisfied(self):
        """too_hot + cooling mode → force primary rooms unsatisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 23.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]  # Living Room
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=True
        )

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        assert parent._room_states[rooms[0].name].is_satisfied is False

    def test_aligned_cooling_too_hot_adjusts_temp_when_past_target(self):
        """too_hot + cooling but room already below target → snap temp down."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 21.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(mode=RoomMode.PRIMARY)

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        # Room is at 21.0 < target 22.0, so no room needs AC
        # → snap temp down: 22.0 → 21.7
        assert parent._target_temp == 21.7

    def test_aligned_heating_too_cold_forces_unsatisfied(self):
        """too_cold + heating → force primary rooms unsatisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 20.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=True
        )

        parent._apply_comfort_feedback(
            too_hot=False,
            too_cold=True,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        assert parent._room_states[rooms[0].name].is_satisfied is False

    def test_opposing_cooling_too_cold_marks_satisfied_when_below_target(self):
        """too_cold + cooling, room below target → mark satisfied (stop cooling)."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 21.5})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )

        parent._apply_comfort_feedback(
            too_hot=False,
            too_cold=True,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        assert parent._room_states[rooms[0].name].is_satisfied is True
        assert parent._target_temp == 22.0  # target unchanged

    def test_opposing_cooling_too_cold_adjusts_temp_and_satisfies_when_above_target(
        self,
    ):
        """too_cold + cooling, room above target → snap up and mark satisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 23.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(mode=RoomMode.PRIMARY)

        parent._apply_comfort_feedback(
            too_hot=False,
            too_cold=True,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        # 23.0 > 22.0 → no room past target (below) → snap up: 22.0 → 22.3
        assert parent._target_temp == 22.3
        # Still marked satisfied so cooling stops immediately
        assert parent._room_states[rooms[0].name].is_satisfied is True

    def test_opposing_heating_too_hot_marks_satisfied_when_in_range(self):
        """too_hot + heating, above target within tolerance → satisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 19.4})
        parent = make_parent(hass, target_temp=19.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        # 19.4 is within grace zone (19 - 0.4 to 19 + 0.4)
        # Should mark satisfied to stop heating, NOT snap target temp down
        assert parent._room_states[rooms[0].name].is_satisfied is True
        assert parent._target_temp == 19.0  # target should NOT change

    def test_opposing_heating_too_hot_cold_room_still_satisfies(self):
        """too_hot + heating, one room warm and another cold → still marks satisfied.

        A cold room elsewhere should not prevent 'too hot' from stopping heating.
        Since one room (19.4) is above the target (19.0), all rooms are marked
        satisfied without snapping the target.
        """
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 19.4,
                "sensor.office_temp": 18.0,
            },
        )
        parent = make_parent(hass, target_temp=19.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        living_room = rooms[0]  # Living Room
        office = rooms[2]  # Office
        primary_rooms = [living_room, office]
        parent._room_states[living_room.name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )
        parent._room_states[office.name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        assert parent._target_temp == 19.0  # target unchanged
        assert parent._room_states[living_room.name].is_satisfied is True
        assert parent._room_states[office.name].is_satisfied is True

    def test_opposing_heating_too_hot_above_target_satisfies_without_snap(self):
        """too_hot + heating, room above target → mark satisfied, no snap."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 24.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(mode=RoomMode.PRIMARY)

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        # 24.0 > 22.0 → room is past target → just satisfy, don't snap
        assert parent._target_temp == 22.0
        assert parent._room_states[rooms[0].name].is_satisfied is True

    def test_opposing_heating_too_hot_below_target_snaps_and_satisfies(self):
        """too_hot + heating, all rooms below target → snap down and satisfy."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 18.5})
        parent = make_parent(hass, target_temp=19.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        # 18.5 < 19.0 → no room past target → snap down: 19.0 → 18.7
        assert parent._target_temp == 18.7
        # Still marked satisfied so heating stops immediately
        assert parent._room_states[rooms[0].name].is_satisfied is True

    def test_opposing_no_valid_sensors_satisfies_without_snap(self):
        """too_hot + heating, all sensors unavailable → satisfy but don't snap."""
        hass = make_hass(room_temps={})  # no sensor data at all
        parent = make_parent(hass, target_temp=20.0, hvac_mode=HVACMode.HEAT)
        rooms = parent._rooms
        primary_rooms = [rooms[0]]
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=[],
            disabled_rooms=[],
        )
        # No valid readings → target must not change
        assert parent._target_temp == 20.0
        # Still marked satisfied so HVAC stops
        assert parent._room_states[rooms[0].name].is_satisfied is True


class TestAlignedFeedbackPromotesSecondary:
    """Aligned feedback promotes secondary rooms with light recently on."""

    def test_promotes_room_with_light_on_not_yet_delayed(self):
        from datetime import datetime

        hass = make_hass(room_temps={"sensor.living_room_temp": 25.0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        rooms = parent._rooms

        secondary_rooms = [rooms[0]]
        primary_rooms = []
        parent._room_states[rooms[0].name] = RoomState(
            mode=RoomMode.SECONDARY,
            light_on=False,
            raw_light_on_at=datetime.now(),
        )

        parent._apply_comfort_feedback(
            too_hot=True,
            too_cold=False,
            primary_rooms=primary_rooms,
            secondary_rooms=secondary_rooms,
            disabled_rooms=[],
        )

        # Room should have been promoted to primary
        assert rooms[0] in primary_rooms
        assert rooms[0] not in secondary_rooms
        assert parent._room_states[rooms[0].name].light_on is True
