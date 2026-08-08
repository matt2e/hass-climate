"""Tests for climate.async_setup_entry sourcing rooms from subentries.

The UI/config-entry setup path builds one Room per "room" subentry and adds a
ChildThermostat for each override room bound to that subentry, while the
ParentThermostat stays on the main entry. These tests drive async_setup_entry
against a mock config entry carrying room subentries and assert exactly that
wiring.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import UnitOfTemperature

from custom_components.matt_thermostat.child_thermostat import ChildThermostat
from custom_components.matt_thermostat.climate import (
    ParentThermostat,
    Room,
    RoomMode,
    async_setup_entry,
)
from custom_components.matt_thermostat.const import SUBENTRY_TYPE_ROOM

from .conftest import ROOMS_JSON, make_hass

ENTRY_ID = "cfg-entry"

# Minimal global options that satisfy CONFIG_ENTRY_SCHEMA_COMMON.
ENTRY_OPTIONS: dict[str, Any] = {
    "name": "Home Thermostat",
    "real_climate": "climate.real_ac",
    "presence": "input_boolean.presence",
    "manual": "input_boolean.manual",
    "bedtime": "input_boolean.bedtime",
}


class _AddEntitiesRecorder:
    """Stand-in for AddConfigEntryEntitiesCallback that records every call.

    Each recorded call keeps the entities added and the config_subentry_id
    they were bound to (None for the main entry).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Any], str | None]] = []

    def __call__(self, entities: Any, config_subentry_id: str | None = None) -> None:
        self.calls.append((list(entities), config_subentry_id))


def _room_subentries() -> dict[str, ConfigSubentry]:
    """Build one "room" subentry per room in the shared ROOMS_JSON fixture."""
    subentries: dict[str, ConfigSubentry] = {}
    for room in json.loads(ROOMS_JSON):
        subentry = ConfigSubentry(
            data=MappingProxyType(dict(room)),
            subentry_type=SUBENTRY_TYPE_ROOM,
            title=room["name"],
            unique_id=room["name"].lower().replace(" ", "_"),
        )
        subentries[subentry.subentry_id] = subentry
    return subentries


def _make_entry(subentries: dict[str, ConfigSubentry]) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = ENTRY_ID
    entry.options = dict(ENTRY_OPTIONS)
    entry.subentries = subentries
    return entry


async def _run_setup(
    subentries: dict[str, ConfigSubentry],
) -> _AddEntitiesRecorder:
    hass = make_hass()
    hass.config = MagicMock()
    hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
    recorder = _AddEntitiesRecorder()
    await async_setup_entry(hass, _make_entry(subentries), recorder)
    return recorder


def _added(recorder: _AddEntitiesRecorder, cls: type) -> list[tuple[Any, str | None]]:
    """Return (entity, subentry_id) pairs for entities of the given type."""
    return [
        (entity, subentry_id)
        for entities, subentry_id in recorder.calls
        for entity in entities
        if isinstance(entity, cls)
    ]


class TestChildThermostatsFromSubentries:
    @pytest.mark.asyncio
    async def test_one_child_per_override_room_bound_to_its_subentry(self):
        subentries = _room_subentries()
        recorder = await _run_setup(subentries)

        # Living Room and Office allow override; Bedroom does not.
        children = _added(recorder, ChildThermostat)
        by_name = {child._attr_name: subentry_id for child, subentry_id in children}
        assert set(by_name) == {"Living Room Thermostat", "Office Thermostat"}

        # Map subentry title -> id so we can assert each child bound to its own.
        title_to_id = {
            sub.title: sid
            for sid, sub in subentries.items()
            if sub.subentry_type == SUBENTRY_TYPE_ROOM
        }
        assert by_name["Living Room Thermostat"] == title_to_id["Living Room"]
        assert by_name["Office Thermostat"] == title_to_id["Office"]

    @pytest.mark.asyncio
    async def test_non_override_room_gets_no_child(self):
        recorder = await _run_setup(_room_subentries())
        children = _added(recorder, ChildThermostat)
        names = {child._attr_name for child, _ in children}
        assert "Bedroom Thermostat" not in names

    @pytest.mark.asyncio
    async def test_child_unique_ids_preserve_entry_and_room_name(self):
        recorder = await _run_setup(_room_subentries())
        children = _added(recorder, ChildThermostat)
        unique_ids = {child._attr_unique_id for child, _ in children}
        assert unique_ids == {
            f"{ENTRY_ID}-Living Room",
            f"{ENTRY_ID}-Office",
        }


class TestParentOnMainEntry:
    @pytest.mark.asyncio
    async def test_single_parent_added_without_subentry_id(self):
        recorder = await _run_setup(_room_subentries())
        parents = _added(recorder, ParentThermostat)
        assert len(parents) == 1
        _parent, subentry_id = parents[0]
        assert subentry_id is None

    @pytest.mark.asyncio
    async def test_parent_owns_every_room(self):
        recorder = await _run_setup(_room_subentries())
        (parent, _sid) = _added(recorder, ParentThermostat)[0]
        assert {room.name for room in parent._rooms} == {
            "Living Room",
            "Bedroom",
            "Office",
        }


class TestNonRoomSubentriesIgnored:
    @pytest.mark.asyncio
    async def test_other_subentry_types_do_not_produce_entities(self):
        subentries = _room_subentries()
        other = ConfigSubentry(
            data=MappingProxyType({}),
            subentry_type="not_a_room",
            title="Ignore me",
            unique_id="ignore",
        )
        subentries[other.subentry_id] = other

        recorder = await _run_setup(subentries)

        (parent, _sid) = _added(recorder, ParentThermostat)[0]
        # The non-room subentry must not have become a room.
        assert "Ignore me" not in {room.name for room in parent._rooms}
        # Still exactly two override children (Living Room + Office).
        assert len(_added(recorder, ChildThermostat)) == 2


class TestRoomFromSubentry:
    def test_reads_subentry_data_like_from_dict(self):
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
        room = Room.from_subentry(MappingProxyType(data))
        assert room == Room.from_dict(dict(data))
        assert room.standard_mode == RoomMode.PRIMARY
        assert room.bedtime_mode == RoomMode.SECONDARY
        assert room.allows_override is True
        assert room.vents == 2

    def test_reads_door_entity_from_subentry(self):
        """The optional door sensor round-trips from subentry data."""
        data = {
            "name": "Study",
            "sensor": "sensor.study_temp",
            "cover": "cover.study_vent",
            "mode": "secondary",
            "bedtime_mode": "disabled",
            "door": "binary_sensor.study_door",
        }
        room = Room.from_subentry(MappingProxyType(data))
        assert room.door_entity == "binary_sensor.study_door"
        assert room == Room.from_dict(dict(data))

    def test_door_entity_defaults_none_when_absent(self):
        data = {
            "name": "Study",
            "sensor": "sensor.study_temp",
            "cover": "cover.study_vent",
            "mode": "secondary",
            "bedtime_mode": "disabled",
        }
        assert Room.from_subentry(MappingProxyType(data)).door_entity is None

    def test_validates_mode_from_subentry(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            Room.from_subentry(
                MappingProxyType(
                    {
                        "name": "Bad",
                        "sensor": "sensor.bad",
                        "cover": "cover.bad",
                        "mode": "turbo",
                        "bedtime_mode": "primary",
                    }
                )
            )
