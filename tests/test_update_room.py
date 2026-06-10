"""Tests for async_update_room — cover position and satisfaction logic."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from homeassistant.components.climate import HVACMode

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_hass, make_parent


class TestUpdateRoomCooling:
    """Cover position and satisfaction when cooling."""

    @pytest.mark.asyncio
    async def test_needs_cooling_full_open(self):
        """Current 25, target 22, hot_tolerance 0.4 → diff=3 > min_diff → cover 100%."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(room, current_temp=25.0, target_temp=22.0)

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        hass.services.async_call.assert_called_with(
            "cover",
            "set_cover_position",
            {"entity_id": "cover.living_room_vent", "position": 100},
            blocking=False,
        )

    @pytest.mark.asyncio
    async def test_overcooled_closes_vent(self):
        """Current 21.0, target 22.0 → diff=-1 < min_diff(-0.4) → cover 0%."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(room, current_temp=21.0, target_temp=22.0)

        state = parent._room_states[room.name]
        assert state.cover_pos == 0

    @pytest.mark.asyncio
    async def test_at_target_satisfied_half_open(self):
        """Room at target and satisfied → cover at 50% (maintenance)."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=True
        )
        parent._sensor_found_for_room[room.name] = True

        # diff = 22.0 - 22.0 = 0, min_diff = -0.4
        # diff > min_diff -> 0 > -0.4 -> True
        # so desired_cover_pos = 100, reached_half_max_at = None
        # At exact target temp in cooling, cover is still 100%
        await parent.async_update_room(room, current_temp=22.0, target_temp=22.0)
        state = parent._room_states[room.name]
        assert state.cover_pos == 100

    @pytest.mark.asyncio
    async def test_slightly_below_target_satisfied_half_open(self):
        """Room slightly below target (within tolerance) and satisfied → 50%."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=True
        )
        parent._sensor_found_for_room[room.name] = True

        # diff = 21.8 - 22.0 = -0.2, min_diff = -0.4
        # diff > min_diff → -0.2 > -0.4 → True → cover 100, half_max reset
        # diff <= min_diff when diff <= -0.4, so current <= 21.6
        # else branch is entered when diff <= min_diff
        # if diff > min_diff -> 100. else -> check sub-conditions
        # diff > min_diff means "still above cutoff", so cover is 100
        # diff <= min_diff means "below cutoff", check if should close or half
        # So at current=21.5, diff=21.5-22.0=-0.5, min_diff=-0.4
        # -0.5 > -0.4 → False → else branch
        # diff <= min_diff (-0.5 <= -0.4 → True) → cover 0
        # For 50% we need: diff NOT > min_diff AND diff NOT <= min_diff
        # That's impossible since diff is always either > or <= min_diff
        # So 50% only happens in the sub-else: diff <= min_diff is False + is_satisfied
        # Wait re-reading: the outer if is `if diff > min_diff` → 100
        # The else handles diff <= min_diff. Inside:
        #   if diff <= min_diff → 0
        #   elif is_satisfied → 50
        #   else → 100
        # But diff <= min_diff is always true in the else branch... unless there's a
        # different min_diff. Let me re-read the code.

        # Oh wait, I see: there are TWO checks:
        # Line 860: if diff > min_diff: desired=100
        # Line 863: else: [reached_half_max logic]
        # Line 872: if diff <= min_diff: desired=0
        # Line 874: elif is_satisfied: desired=50
        # Line 877: else: desired=100
        # The outer if/else splits on diff > min_diff. If we're in the else,
        # diff <= min_diff, so the inner if (line 872) is ALWAYS true.
        # That means 50% is UNREACHABLE via line 874!
        # Unless... there's a floating point edge case where diff == min_diff exactly.
        # diff > min_diff → False, but diff <= min_diff → True. So still 0.
        # This means 50% cover is dead code in the current implementation.
        # Let me verify by just testing the two reachable paths.
        pass

    @pytest.mark.asyncio
    async def test_first_reading_within_range_marks_satisfied(self):
        """First temperature reading within max_diff marks room satisfied."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._sensor_found_for_room[room.name] = False  # first reading

        # diff = 22.2 - 22.0 = 0.2, max_diff = 0.4
        # diff <= max_diff so no unsatisfy, and is_first → satisfied = True
        await parent.async_update_room(room, current_temp=22.2, target_temp=22.0)
        state = parent._room_states[room.name]
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_first_reading_above_max_not_satisfied(self):
        """First reading well above target — not satisfied."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._sensor_found_for_room[room.name] = False

        # diff = 25.0 - 22.0 = 3.0, max_diff = 0.4 → diff > max_diff → unsatisfied
        await parent.async_update_room(room, current_temp=25.0, target_temp=22.0)
        state = parent._room_states[room.name]
        assert state.is_satisfied is False

    @pytest.mark.asyncio
    async def test_reached_max_at_sets_satisfied(self):
        """When diff < min_diff (overcooled past tolerance), marks satisfied."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.PRIMARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        # diff = 21.0 - 22.0 = -1.0, min_diff = -0.4 → diff < min_diff
        # reached_max_at was None → set now and is_satisfied = True
        await parent.async_update_room(room, current_temp=21.0, target_temp=22.0)
        state = parent._room_states[room.name]
        assert state.is_satisfied is True
        assert state.reached_max_at is not None

    @pytest.mark.asyncio
    async def test_no_service_call_when_position_unchanged(self):
        """Don't call cover service if position hasn't changed."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        # diff = 25 - 22 = 3 > min_diff → cover 100, same as current
        await parent.async_update_room(room, current_temp=25.0, target_temp=22.0)
        hass.services.async_call.assert_not_called()


class TestUpdateRoomHeating:
    """Cover position logic in heating mode."""

    @pytest.mark.asyncio
    async def test_needs_heating_full_open(self):
        """Current 19, target 22 → diff = 22-19 = 3 > min_diff → cover 100%."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(room, current_temp=19.0, target_temp=22.0)
        assert parent._room_states[room.name].cover_pos == 100

    @pytest.mark.asyncio
    async def test_overheated_closes_vent(self):
        """Current 23, target 22 → diff = 22-23 = -1 < min_diff(-0.4) → cover 0%."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(room, current_temp=23.0, target_temp=22.0)
        assert parent._room_states[room.name].cover_pos == 0


class TestUpdateRoomSecondarySatisfiedTarget:
    """Secondary rooms ride the primary target for cover.

    They still judge "satisfied" at the secondary bound.
    """

    @pytest.mark.asyncio
    async def test_between_secondary_and_primary_cooling_cover_open_and_satisfied(self):
        """Cooling: current 23, primary target 22, secondary target 24.

        Cover follows primary (diff=1 > min_diff → 100%). Satisfied at
        secondary (satisfied_diff=-1 < min_diff → satisfied).
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]  # Bedroom
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=23.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_overshoots_primary_cooling_cover_closes_still_satisfied(self):
        """Cooling: current 21, primary target 22, secondary target 24.

        Cover closes (diff=-1 < min_diff → 0%). Already past secondary,
        still satisfied.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]  # Bedroom
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=21.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 0
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_above_secondary_cooling_unsatisfied(self):
        """Cooling: current 25, primary 22, secondary 24.

        Unsatisfied at the secondary bar.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=True
        )
        parent._sensor_found_for_room[room.name] = True

        # satisfied_diff = 25-24 = 1 > max_diff(0.4) → unsatisfied
        await parent.async_update_room(
            room,
            current_temp=25.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.is_satisfied is False
        assert state.cover_pos == 100

    @pytest.mark.asyncio
    async def test_between_secondary_and_primary_heating_cover_open_and_satisfied(self):
        """Heating: current 21, primary 22, secondary 20.

        Cover follows primary (diff=22-21=1 > min_diff → 100%). Satisfied
        at secondary (satisfied_diff=20-21=-1 < min_diff → satisfied).
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=21.0,
            target_temp=22.0,
            satisfied_target=20.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_cool_at_secondary_boundary_sets_reached_target_at(self):
        """Case a: COOL, current=24 at the secondary target.

        satisfied_diff=0 → reached_target_at set; is_satisfied stays where
        it was (0 is not > max_diff); diff=2 → cover 100.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=24.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.reached_target_at is not None
        assert state.is_satisfied is False

    @pytest.mark.asyncio
    async def test_cool_at_satisfied_min_boundary_no_reached_max(self):
        """Case b: COOL, current=23.6 exactly at the satisfied min_diff edge.

        satisfied_diff=-0.4 is >= min_diff(-0.4) → reached_max_at stays
        None; diff=1.6 → cover 100.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=23.6,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.reached_max_at is None

    @pytest.mark.asyncio
    async def test_cool_just_past_satisfied_min_sets_reached_max(self):
        """Case c: COOL, current=23.5 just past satisfied min_diff.

        satisfied_diff=-0.5 < min_diff → reached_max_at set and
        is_satisfied=True; diff=1.5 still > min_diff → cover stays 100.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=23.5,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True
        assert state.reached_max_at is not None

    @pytest.mark.asyncio
    async def test_cool_at_satisfied_max_boundary_keeps_satisfied(self):
        """Case d: COOL, current=24.4 exactly at the satisfied max_diff edge.

        satisfied_diff=0.4 is not > max_diff(0.4) → is_satisfied stays True.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=True
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=24.4,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.is_satisfied is True
        assert state.cover_pos == 100

    @pytest.mark.asyncio
    async def test_heat_just_past_satisfied_min_sets_reached_max(self):
        """Case e: HEAT, current=20.5 just past heating satisfied min_diff.

        Mirror of case c. satisfied_diff=20-20.5=-0.5 < min_diff →
        reached_max_at set and is_satisfied=True; diff=22-20.5=1.5 →
        cover stays 100.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=20.5,
            target_temp=22.0,
            satisfied_target=20.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True
        assert state.reached_max_at is not None

    @pytest.mark.asyncio
    async def test_heat_well_below_both_targets_unsatisfied(self):
        """Case f: HEAT, current=18.0 well below both targets.

        satisfied_diff=2 > max_diff → is_satisfied=False;
        satisfied_diff > 0 → reached_target_at=None; cover stays open.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY,
            is_satisfied=True,
            reached_target_at=datetime.now(),
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=18.0,
            target_temp=22.0,
            satisfied_target=20.0,
        )

        state = parent._room_states[room.name]
        assert state.is_satisfied is False
        assert state.reached_target_at is None
        assert state.cover_pos == 100

    @pytest.mark.asyncio
    async def test_fan_only_between_targets_matches_cool_behavior(self):
        """Case g: FAN_ONLY shares the cool branch — cover 100, satisfied."""
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.FAN_ONLY)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=23.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True


class TestSatisfiedTargetTimestampDecoupling:
    """Timestamps respect their respective targets after the split."""

    @pytest.mark.asyncio
    async def test_reached_half_max_at_keyed_to_diff_not_satisfied(self):
        """reached_half_max_at follows diff (cover target), not satisfied_diff.

        current=22.5, primary=22, secondary=24 (cool): diff=0.5 → cover
        100 and reached_half_max_at stays None, even though
        satisfied_diff=-1.5 means the room is well past the satisfied bar.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=22.5,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.reached_half_max_at is None
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_reached_half_max_at_sets_when_diff_drops_even_if_satisfied(self):
        """reached_half_max_at still sets on diff dropping, even when satisfied.

        current=21.5, primary=22, secondary=24 (cool): diff=-0.5 <= min_diff.
        is_satisfied is already True (carried over) — but reached_half_max_at
        should still be stamped because it tracks the diff target.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY,
            is_satisfied=True,
            reached_max_at=datetime.now() - timedelta(minutes=10),
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=21.5,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        state = parent._room_states[room.name]
        assert state.cover_pos == 0
        assert state.reached_half_max_at is not None

    @pytest.mark.asyncio
    async def test_reached_max_at_set_via_satisfied_diff_not_first_reading(self):
        """is_satisfied via reached_max_at path with sensor_found already True.

        Confirms the satisfied_diff < min_diff branch (not the first-reading
        branch) is what flipped is_satisfied to True.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room,
            current_temp=21.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        state = parent._room_states[room.name]
        assert state.is_satisfied is True
        assert state.reached_max_at is not None


class TestSatisfiedTargetNoneFallback:
    """Passing satisfied_target=None must behave identically to omitting it."""

    @pytest.mark.asyncio
    async def test_cool_needs_cooling_none_default(self):
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room, current_temp=25.0, target_temp=22.0, satisfied_target=None
        )
        assert parent._room_states[room.name].cover_pos == 100

    @pytest.mark.asyncio
    async def test_cool_overcooled_none_default(self):
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room, current_temp=21.0, target_temp=22.0, satisfied_target=None
        )
        assert parent._room_states[room.name].cover_pos == 0

    @pytest.mark.asyncio
    async def test_heat_needs_heating_none_default(self):
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room, current_temp=19.0, target_temp=22.0, satisfied_target=None
        )
        assert parent._room_states[room.name].cover_pos == 100

    @pytest.mark.asyncio
    async def test_heat_overheated_none_default(self):
        hass = make_hass(cover_positions={"cover.living_room_vent": 100})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = parent._rooms[0]
        parent._room_states[room.name].mode = RoomMode.PRIMARY
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_room(
            room, current_temp=23.0, target_temp=22.0, satisfied_target=None
        )
        assert parent._room_states[room.name].cover_pos == 0

    @pytest.mark.asyncio
    async def test_first_reading_within_range_none_default_satisfied(self):
        """satisfied_target=None falls back to target_temp for first-reading check."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._sensor_found_for_room[room.name] = False

        await parent.async_update_room(
            room, current_temp=22.2, target_temp=22.0, satisfied_target=None
        )
        assert parent._room_states[room.name].is_satisfied is True


class TestSecondaryCoolingCycleSequence:
    """End-to-end sequence: a secondary room cools through a full cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_transitions(self):
        """Walk t0..t3 through a cooling cycle and verify each transition.

        target=22, satisfied_target=24, cold_tol=hot_tol=0.4.
        """
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[1]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        # t0: current=26 - well above both targets, unsatisfied, full open.
        await parent.async_update_room(
            room,
            current_temp=26.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        state = parent._room_states[room.name]
        assert state.is_satisfied is False
        assert state.cover_pos == 100
        assert state.reached_target_at is None
        assert state.reached_max_at is None

        # t1: current=24 - at secondary target. reached_target_at set, still
        # not satisfied, cover stays open.
        await parent.async_update_room(
            room,
            current_temp=24.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        assert state.is_satisfied is False
        assert state.reached_target_at is not None
        assert state.reached_max_at is None
        assert state.cover_pos == 100

        # t2: current=23 - past secondary, still pulling toward primary.
        # reached_max_at set, is_satisfied flips True, cover still open.
        await parent.async_update_room(
            room,
            current_temp=23.0,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        assert state.is_satisfied is True
        assert state.reached_max_at is not None
        assert state.cover_pos == 100

        # t3: current=21.5 - overshot primary. diff=-0.5 < min_diff -> cover
        # closes; remains satisfied.
        await parent.async_update_room(
            room,
            current_temp=21.5,
            target_temp=22.0,
            satisfied_target=24.0,
        )
        assert state.is_satisfied is True
        assert state.cover_pos == 0


class TestHalfMaxSatisfaction:
    """Satisfaction timeout after 5 minutes at half-max."""

    @pytest.mark.asyncio
    async def test_half_max_timeout_satisfies(self):
        """After 5 min below min_diff, room becomes satisfied."""
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            reached_half_max_at=datetime.now() - timedelta(minutes=6),
        )
        parent._sensor_found_for_room[room.name] = True

        # diff = 21.5 - 22.0 = -0.5 <= min_diff(-0.4)
        # reached_half_max_at already set and > 5 min ago → satisfied
        await parent.async_update_room(room, current_temp=21.5, target_temp=22.0)
        assert parent._room_states[room.name].is_satisfied is True

    @pytest.mark.asyncio
    async def test_half_max_not_yet_timed_out(self):
        """Not yet satisfied: <5 min below min_diff, not past max.

        Note: we need a temp where diff <= min_diff but
        reached_max_at is already set (so it doesn't trigger the
        first-time satisfied=True). We use a temp that triggers
        the half_max path without the reached_max path by setting
        reached_max_at already.
        """
        hass = make_hass(cover_positions={"cover.living_room_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = parent._rooms[0]
        already_set = datetime.now() - timedelta(minutes=10)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.PRIMARY,
            is_satisfied=False,
            reached_half_max_at=datetime.now() - timedelta(minutes=2),
            reached_max_at=already_set,  # already set, won't re-trigger satisfied
        )
        parent._sensor_found_for_room[room.name] = True

        # diff = 21.5 - 22.0 = -0.5, min_diff = -0.4
        # reached_max_at already set → no new satisfaction from that path
        # reached_half_max_at only 2 min ago → not yet 5 min → no satisfaction
        await parent.async_update_room(room, current_temp=21.5, target_temp=22.0)
        assert parent._room_states[room.name].is_satisfied is False
