"""Tests for the per-room config subentry flow and its schema.

These exercise the UI surface added when rooms moved from a single
CONF_ROOMS JSON blob to one "room" config subentry each: ROOM_SCHEMA
validation, RoomSubentryFlowHandler (add + reconfigure), the supported
subentry-type registration, and that CONF_ROOMS is gone from the global
config/options schemas.

There is no full Home Assistant flow-manager harness installed, so — like
the rest of this suite — the flow handler is driven directly: its handler
tuple, context and hass are set on the instance and its async_step_* methods
are called, matching how core exposes these callbacks.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigSubentry,
)
from homeassistant.data_entry_flow import FlowResultType

from custom_components.matt_thermostat.config_flow import (
    CONFIG_SCHEMA,
    OPTIONS_SCHEMA,
    ROOM_SCHEMA,
    ConfigFlowHandler,
    RoomSubentryFlowHandler,
)
from custom_components.matt_thermostat.const import CONF_ROOMS, SUBENTRY_TYPE_ROOM

# A full room, as the frontend would submit it (valid entity ids so the
# EntitySelectors accept them).
VALID_ROOM_INPUT: dict[str, Any] = {
    "name": "Kitchen",
    "sensor": "sensor.kitchen_temp",
    "cover": "cover.kitchen_vent",
    "light": "light.kitchen",
    "mode": "primary",
    "bedtime_mode": "disabled",
    "allows_override": True,
    "is_overflow": False,
    "vents": 2,
}

# The minimum a room needs: name, sensor, cover and both modes.
MINIMAL_ROOM_INPUT: dict[str, Any] = {
    "name": "Closet",
    "sensor": "sensor.closet_temp",
    "cover": "cover.closet_vent",
    "mode": "disabled",
    "bedtime_mode": "disabled",
}


def _add_handler(entry_id: str = "entry-1") -> RoomSubentryFlowHandler:
    """Return a handler wired up for the add (user) source."""
    handler = RoomSubentryFlowHandler()
    handler.handler = (entry_id, SUBENTRY_TYPE_ROOM)
    handler.context = {"source": SOURCE_USER}
    handler.flow_id = "test-flow"
    return handler


def _reconfigure_handler(
    hass: MagicMock, entry_id: str, subentry_id: str
) -> RoomSubentryFlowHandler:
    """Return a handler wired up for the reconfigure source."""
    handler = RoomSubentryFlowHandler()
    handler.hass = hass
    handler.handler = (entry_id, SUBENTRY_TYPE_ROOM)
    handler.context = {"source": SOURCE_RECONFIGURE, "subentry_id": subentry_id}
    handler.flow_id = "test-flow"
    return handler


def _entry_with_subentry(
    entry_id: str, subentry: ConfigSubentry
) -> tuple[MagicMock, MagicMock]:
    """Build a mock hass + config entry that owns a single subentry."""
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.subentries = {subentry.subentry_id: subentry}

    hass = MagicMock()
    hass.config_entries.async_get_known_entry = MagicMock(return_value=entry)
    hass.config_entries.async_update_subentry = MagicMock(return_value=True)
    return hass, entry


def _suggested_values(schema: vol.Schema) -> dict[str, Any]:
    """Extract the suggested_value hints add_suggested_values_to_schema set."""
    out: dict[str, Any] = {}
    for key in schema.schema:
        description = getattr(key, "description", None)
        if description and "suggested_value" in description:
            out[key.schema] = description["suggested_value"]
    return out


class TestRoomSchema:
    def test_defaults_applied_for_minimal_room(self):
        out = ROOM_SCHEMA(dict(MINIMAL_ROOM_INPUT))
        assert out["allows_override"] is False
        assert out["is_overflow"] is False
        assert out["vents"] == 1

    def test_light_is_optional_and_absent_when_omitted(self):
        out = ROOM_SCHEMA(dict(MINIMAL_ROOM_INPUT))
        assert "light" not in out

    def test_door_is_optional_and_absent_when_omitted(self):
        out = ROOM_SCHEMA(dict(MINIMAL_ROOM_INPUT))
        assert "door" not in out

    def test_door_accepted_when_provided(self):
        out = ROOM_SCHEMA({**MINIMAL_ROOM_INPUT, "door": "binary_sensor.closet_door"})
        assert out["door"] == "binary_sensor.closet_door"

    def test_full_room_round_trips(self):
        out = ROOM_SCHEMA(dict(VALID_ROOM_INPUT))
        assert out["name"] == "Kitchen"
        assert out["light"] == "light.kitchen"
        assert out["mode"] == "primary"
        assert out["allows_override"] is True
        assert out["vents"] == 2

    def test_invalid_mode_rejected(self):
        with pytest.raises(vol.Invalid):
            ROOM_SCHEMA({**MINIMAL_ROOM_INPUT, "mode": "turbo"})

    def test_invalid_bedtime_mode_rejected(self):
        with pytest.raises(vol.Invalid):
            ROOM_SCHEMA({**MINIMAL_ROOM_INPUT, "bedtime_mode": "turbo"})

    def test_missing_required_name_rejected(self):
        bad = {k: v for k, v in MINIMAL_ROOM_INPUT.items() if k != "name"}
        with pytest.raises(vol.Invalid, match="name"):
            ROOM_SCHEMA(bad)


class TestRoomSubentryFlowAdd:
    @pytest.mark.asyncio
    async def test_shows_add_form(self):
        handler = _add_handler()
        result = await handler.async_step_user()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
        assert result["data_schema"] is ROOM_SCHEMA

    @pytest.mark.asyncio
    async def test_creates_subentry_with_name_title_and_data(self):
        handler = _add_handler()
        result = await handler.async_step_user(dict(VALID_ROOM_INPUT))
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "Kitchen"
        assert result["data"] == VALID_ROOM_INPUT


class TestRoomSubentryFlowReconfigure:
    def _existing_subentry(self) -> ConfigSubentry:
        return ConfigSubentry(
            data=MappingProxyType(dict(VALID_ROOM_INPUT)),
            subentry_type=SUBENTRY_TYPE_ROOM,
            title="Kitchen",
            unique_id="kitchen",
        )

    @pytest.mark.asyncio
    async def test_shows_prefilled_reconfigure_form(self):
        subentry = self._existing_subentry()
        hass, _entry = _entry_with_subentry("entry-1", subentry)
        handler = _reconfigure_handler(hass, "entry-1", subentry.subentry_id)

        result = await handler.async_step_reconfigure()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "reconfigure"
        # The existing values are offered back as suggestions in the form.
        assert _suggested_values(result["data_schema"]) == VALID_ROOM_INPUT

    @pytest.mark.asyncio
    async def test_reconfigure_updates_subentry_and_aborts(self):
        subentry = self._existing_subentry()
        hass, entry = _entry_with_subentry("entry-1", subentry)
        handler = _reconfigure_handler(hass, "entry-1", subentry.subentry_id)

        new_input = {**VALID_ROOM_INPUT, "name": "Kitchenette", "vents": 3}
        result = await handler.async_step_reconfigure(new_input)

        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "reconfigure_successful"

        hass.config_entries.async_update_subentry.assert_called_once()
        _args, kwargs = hass.config_entries.async_update_subentry.call_args
        assert kwargs["entry"] is entry
        assert kwargs["subentry"] is subentry
        assert kwargs["title"] == "Kitchenette"
        assert kwargs["data"] == new_input


class TestSupportedSubentryTypes:
    def test_registers_room_handler(self):
        config_entry = MagicMock()
        supported = ConfigFlowHandler.async_get_supported_subentry_types(config_entry)
        assert supported == {SUBENTRY_TYPE_ROOM: RoomSubentryFlowHandler}


class TestGlobalSchema:
    def _schema_keys(self, schema: dict[Any, Any]) -> set[Any]:
        return {getattr(key, "schema", key) for key in schema}

    def test_options_schema_has_no_rooms_field(self):
        assert CONF_ROOMS not in self._schema_keys(OPTIONS_SCHEMA)

    def test_config_schema_has_no_rooms_field(self):
        assert CONF_ROOMS not in self._schema_keys(CONFIG_SCHEMA)

    def test_minor_version_is_three(self):
        assert ConfigFlowHandler.MINOR_VERSION == 3
