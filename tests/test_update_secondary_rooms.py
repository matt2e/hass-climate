"""Tests for async_update_secondary_rooms — wiring of split targets."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.climate import HVACMode
from homeassistant.const import STATE_UNAVAILABLE

from custom_components.matt_thermostat.climate import RoomMode, RoomState

from .conftest import make_hass, make_parent, make_state


def _bedroom(parent):
    """The Bedroom room from the standard fixture (secondary, no light)."""
    return next(r for r in parent._rooms if r.name == "Bedroom")


class TestUpdateSecondaryRooms:
    """Integration through async_update_secondary_rooms."""

    @pytest.mark.asyncio
    async def test_passes_primary_target_for_cover_and_secondary_for_satisfied(self):
        """Cool, target=22, current=23 → cover open, satisfied at secondary."""
        hass = make_hass(
            room_temps={"sensor.bedroom_temp": 23.0},
            cover_positions={"cover.bedroom_vent": 0},
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = _bedroom(parent)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_secondary_rooms([room])

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_secondary_temp_cool_cap_at_28(self):
        """target=27 → secondary capped at 28; current=27.5 satisfies."""
        hass = make_hass(
            room_temps={"sensor.bedroom_temp": 27.5},
            cover_positions={"cover.bedroom_vent": 0},
        )
        parent = make_parent(hass, target_temp=27.0, hvac_mode=HVACMode.COOL)
        room = _bedroom(parent)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_secondary_rooms([room])

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_secondary_temp_heat_floor_at_16(self):
        """target=17 (heat) → secondary floored at 16; current=16.5 satisfies."""
        hass = make_hass(
            room_temps={"sensor.bedroom_temp": 16.5},
            cover_positions={"cover.bedroom_vent": 0},
        )
        parent = make_parent(hass, target_temp=17.0, hvac_mode=HVACMode.HEAT)
        room = _bedroom(parent)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        await parent.async_update_secondary_rooms([room])

        state = parent._room_states[room.name]
        assert state.cover_pos == 100
        assert state.is_satisfied is True

    @pytest.mark.asyncio
    async def test_secondary_temp_normal_offset_cool(self):
        """Cool target=22 → secondary=24 used end-to-end (no cap)."""
        hass = make_hass(
            room_temps={"sensor.bedroom_temp": 23.0},
            cover_positions={"cover.bedroom_vent": 0},
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = _bedroom(parent)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        with patch.object(parent, "async_update_room", new=AsyncMock()) as mock_update:
            await parent.async_update_secondary_rooms([room])

        mock_update.assert_awaited_once()
        kwargs = mock_update.await_args.kwargs
        assert kwargs["target_temp"] == 22.0
        assert kwargs["satisfied_target"] == 24.0

    @pytest.mark.asyncio
    async def test_secondary_temp_normal_offset_heat(self):
        """Heat target=22 → secondary=20 used end-to-end (no floor)."""
        hass = make_hass(
            room_temps={"sensor.bedroom_temp": 21.0},
            cover_positions={"cover.bedroom_vent": 0},
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.HEAT)
        room = _bedroom(parent)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        with patch.object(parent, "async_update_room", new=AsyncMock()) as mock_update:
            await parent.async_update_secondary_rooms([room])

        mock_update.assert_awaited_once()
        kwargs = mock_update.await_args.kwargs
        assert kwargs["target_temp"] == 22.0
        assert kwargs["satisfied_target"] == 20.0

    @pytest.mark.asyncio
    async def test_skips_unavailable_sensor(self):
        """STATE_UNAVAILABLE sensor → async_update_room not called for that room."""
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = _bedroom(parent)

        # Override the bedroom sensor to be unavailable.
        original_get = hass.states.get
        hass.states.get = lambda eid: (
            make_state(STATE_UNAVAILABLE)
            if eid == room.sensor_entity
            else original_get(eid)
        )

        with patch.object(parent, "async_update_room", new=AsyncMock()) as mock_update:
            await parent.async_update_secondary_rooms([room])

        mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_missing_sensor(self):
        """hass.states.get returns None → async_update_room not called."""
        hass = make_hass(cover_positions={"cover.bedroom_vent": 0})
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = _bedroom(parent)

        # No sensor entry → make_hass's get returns None for that entity.

        with patch.object(parent, "async_update_room", new=AsyncMock()) as mock_update:
            await parent.async_update_secondary_rooms([room])

        mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_secondary_rooms_each_processed(self):
        """Each room in the list gets its own async_update_room call."""
        hass = make_hass(
            room_temps={
                "sensor.bedroom_temp": 23.0,
                "sensor.office_temp": 25.0,
            },
            cover_positions={
                "cover.bedroom_vent": 0,
                "cover.office_vent": 0,
            },
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        bedroom = next(r for r in parent._rooms if r.name == "Bedroom")
        office = next(r for r in parent._rooms if r.name == "Office")
        for r in (bedroom, office):
            parent._room_states[r.name] = RoomState(
                mode=RoomMode.SECONDARY, is_satisfied=False
            )
            parent._sensor_found_for_room[r.name] = True

        with patch.object(parent, "async_update_room", new=AsyncMock()) as mock_update:
            await parent.async_update_secondary_rooms([bedroom, office])

        assert mock_update.await_count == 2
        rooms_called = [c.kwargs["room"] for c in mock_update.await_args_list]
        currents = [c.kwargs["current_temp"] for c in mock_update.await_args_list]
        assert rooms_called == [bedroom, office]
        assert currents == [23.0, 25.0]

    @pytest.mark.asyncio
    async def test_argument_wiring(self):
        """async_update_room is called with target_temp and satisfied_target kwargs."""
        hass = make_hass(
            room_temps={"sensor.bedroom_temp": 23.0},
            cover_positions={"cover.bedroom_vent": 0},
        )
        parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL)
        room = _bedroom(parent)
        parent._room_states[room.name] = RoomState(
            mode=RoomMode.SECONDARY, is_satisfied=False
        )
        parent._sensor_found_for_room[room.name] = True

        with patch.object(parent, "async_update_room", new=AsyncMock()) as mock_update:
            await parent.async_update_secondary_rooms([room])

        mock_update.assert_awaited_once_with(
            room=room,
            current_temp=23.0,
            target_temp=parent._target_temp,
            satisfied_target=parent._target_secondary_temp(),
        )
