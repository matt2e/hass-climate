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

    def test_opposing_cooling_too_cold_marks_satisfied_when_in_range(self):
        """too_cold + cooling, room within tolerance → mark satisfied."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 22.0})
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

    def test_opposing_cooling_too_cold_adjusts_temp_when_out_of_range(self):
        """too_cold + cooling, room well below target → ease off (snap up)."""
        hass = make_hass(room_temps={"sensor.living_room_temp": 20.0})
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
        # 20.0 < 22.0 - 0.4 = 21.6 → out of range → snap up: 22.0 → 22.3
        assert parent._target_temp == 22.3

    def test_opposing_heating_too_hot_adjusts_temp_down(self):
        """too_hot + heating, room above target+tolerance → ease off (snap down)."""
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
        # 24.0 > 22.0 + 0.4 = 22.4 → out of range → snap down: 22.0 → 21.7
        assert parent._target_temp == 21.7


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
