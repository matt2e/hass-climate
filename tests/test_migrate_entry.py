"""Tests for async_migrate_entry: CONF_ROOMS JSON blob -> room subentries.

Entries older than minor_version 3 kept their rooms as a single CONF_ROOMS
JSON array in options. Migration turns each room into a "room" ConfigSubentry,
strips CONF_ROOMS from options and bumps to minor_version 3. It must stay
resilient: malformed JSON or an individual bad room is logged and skipped, not
raised, so a broken legacy value can never leave the entry mid-migration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.matt_thermostat import async_migrate_entry
from custom_components.matt_thermostat.const import CONF_ROOMS, SUBENTRY_TYPE_ROOM

from .conftest import ROOMS_JSON


async def _migrate(
    options: dict[str, Any], *, minor_version: int = 2
) -> tuple[bool, list[Any], dict[str, Any], MagicMock]:
    """Run async_migrate_entry against a mock entry.

    Returns (result, added subentries, async_update_entry kwargs, hass).
    """
    hass = MagicMock()
    added: list[Any] = []
    hass.config_entries.async_add_subentry = MagicMock(
        side_effect=lambda _entry, subentry: added.append(subentry)
    )
    updates: dict[str, Any] = {}
    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda _entry, **kwargs: updates.update(kwargs)
    )

    entry = MagicMock()
    entry.version = 1
    entry.minor_version = minor_version
    entry.options = dict(options)

    result = await async_migrate_entry(hass, entry)
    return result, added, updates, hass


class TestSuccessfulMigration:
    @pytest.mark.asyncio
    async def test_creates_one_room_subentry_per_room(self):
        result, added, _updates, _hass = await _migrate({CONF_ROOMS: ROOMS_JSON})

        assert result is True
        assert [sub.title for sub in added] == ["Living Room", "Bedroom", "Office"]
        assert all(sub.subentry_type == SUBENTRY_TYPE_ROOM for sub in added)

    @pytest.mark.asyncio
    async def test_subentry_data_matches_legacy_room(self):
        _result, added, _updates, _hass = await _migrate({CONF_ROOMS: ROOMS_JSON})

        living = next(sub for sub in added if sub.title == "Living Room")
        assert living.data["sensor"] == "sensor.living_room_temp"
        assert living.data["mode"] == "primary"
        assert living.data["allows_override"] is True

    @pytest.mark.asyncio
    async def test_legacy_door_key_survives_migration(self):
        """A room's optional 'door' key rides through into the subentry data."""
        rooms = json.dumps(
            [
                {
                    "name": "Bedroom",
                    "sensor": "sensor.bedroom_temp",
                    "cover": "cover.bedroom_vent",
                    "mode": "secondary",
                    "bedtime_mode": "primary",
                    "door": "binary_sensor.bedroom_door",
                }
            ]
        )
        _result, added, _updates, _hass = await _migrate({CONF_ROOMS: rooms})
        assert added[0].data["door"] == "binary_sensor.bedroom_door"

    @pytest.mark.asyncio
    async def test_unique_ids_are_slugified(self):
        _result, added, _updates, _hass = await _migrate({CONF_ROOMS: ROOMS_JSON})
        assert [sub.unique_id for sub in added] == ["living_room", "bedroom", "office"]

    @pytest.mark.asyncio
    async def test_strips_rooms_from_options_and_bumps_version(self):
        _result, _added, updates, _hass = await _migrate(
            {CONF_ROOMS: ROOMS_JSON, "name": "Home"}
        )
        assert CONF_ROOMS not in updates["options"]
        assert updates["options"] == {"name": "Home"}
        assert updates["minor_version"] == 3


class TestDuplicateNames:
    @pytest.mark.asyncio
    async def test_duplicate_room_names_get_suffixed_unique_ids(self):
        rooms = json.dumps(
            [
                {
                    "name": "Den",
                    "sensor": "sensor.den_a",
                    "cover": "cover.den_a",
                    "mode": "primary",
                    "bedtime_mode": "disabled",
                },
                {
                    "name": "Den",
                    "sensor": "sensor.den_b",
                    "cover": "cover.den_b",
                    "mode": "secondary",
                    "bedtime_mode": "primary",
                },
            ]
        )
        _result, added, _updates, _hass = await _migrate({CONF_ROOMS: rooms})
        assert [sub.unique_id for sub in added] == ["den", "den_2"]
        # Both still land as subentries; only the id disambiguates.
        assert [sub.title for sub in added] == ["Den", "Den"]


class TestRobustness:
    @pytest.mark.asyncio
    async def test_malformed_json_bumps_version_without_raising(self):
        result, added, updates, _hass = await _migrate({CONF_ROOMS: "{not valid json"})

        assert result is True
        assert added == []
        assert updates["minor_version"] == 3
        assert CONF_ROOMS not in updates["options"]

    @pytest.mark.asyncio
    async def test_non_list_blob_bumps_version_without_raising(self):
        result, added, updates, _hass = await _migrate(
            {CONF_ROOMS: json.dumps({"not": "a list"})}
        )
        assert result is True
        assert added == []
        assert updates["minor_version"] == 3

    @pytest.mark.asyncio
    async def test_partially_bad_rooms_skip_only_the_bad_one(self):
        rooms = json.dumps(
            [
                {
                    "name": "Good",
                    "sensor": "sensor.good",
                    "cover": "cover.good",
                    "mode": "primary",
                    "bedtime_mode": "disabled",
                },
                # Missing 'cover' -> Room.from_dict raises -> skipped.
                {
                    "name": "Bad",
                    "sensor": "sensor.bad",
                    "mode": "primary",
                    "bedtime_mode": "disabled",
                },
            ]
        )
        result, added, updates, _hass = await _migrate({CONF_ROOMS: rooms})

        assert result is True
        assert [sub.title for sub in added] == ["Good"]
        assert updates["minor_version"] == 3

    @pytest.mark.asyncio
    async def test_missing_rooms_option_still_bumps_version(self):
        result, added, updates, _hass = await _migrate({"name": "Home"})

        assert result is True
        assert added == []
        assert updates["minor_version"] == 3
        assert updates["options"] == {"name": "Home"}


class TestAlreadyMigrated:
    @pytest.mark.asyncio
    async def test_minor_version_3_is_a_noop(self):
        result, added, updates, hass = await _migrate(
            {CONF_ROOMS: ROOMS_JSON}, minor_version=3
        )

        assert result is True
        assert added == []
        # Nothing rewritten: no subentries added and options untouched.
        assert updates == {}
        hass.config_entries.async_add_subentry.assert_not_called()
        hass.config_entries.async_update_entry.assert_not_called()
