"""Tests for FeedbackSwitch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.matt_thermostat.const import DOMAIN
from custom_components.matt_thermostat.switch import FeedbackSwitch, async_setup_entry

from .conftest import make_hass


class TestFeedbackSwitch:
    def test_init(self):
        sw = FeedbackSwitch("Too Hot", "uid-hot")
        assert sw._attr_name == "Too Hot"
        assert sw._attr_unique_id == "uid-hot"
        assert sw.is_on is False

    @pytest.mark.asyncio
    async def test_turn_on(self):
        sw = FeedbackSwitch("Too Hot", "uid-hot")
        sw.async_write_ha_state = MagicMock()
        await sw.async_turn_on()
        assert sw.is_on is True
        sw.async_write_ha_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_off(self):
        sw = FeedbackSwitch("Too Hot", "uid-hot")
        sw.async_write_ha_state = MagicMock()
        sw._is_on = True
        await sw.async_turn_off()
        assert sw.is_on is False
        sw.async_write_ha_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_toggle_cycle(self):
        sw = FeedbackSwitch("Too Cold", "uid-cold")
        sw.async_write_ha_state = MagicMock()
        assert sw.is_on is False
        await sw.async_turn_on()
        assert sw.is_on is True
        await sw.async_turn_off()
        assert sw.is_on is False


class TestSwitchSetupEntry:
    @pytest.mark.asyncio
    async def test_creates_two_switches(self):
        hass = make_hass()
        config_entry = MagicMock()
        config_entry.entry_id = "test-entry-123"
        added = []
        async_add_entities = MagicMock(side_effect=lambda ents: added.extend(ents))

        await async_setup_entry(hass, config_entry, async_add_entities)

        assert len(added) == 2
        assert added[0]._attr_name == "Too Hot"
        assert added[1]._attr_name == "Too Cold"
        assert added[0]._attr_unique_id == "test-entry-123-too-hot"
        assert added[1]._attr_unique_id == "test-entry-123-too-cold"

    @pytest.mark.asyncio
    async def test_registers_in_hass_data(self):
        hass = make_hass()
        config_entry = MagicMock()
        config_entry.entry_id = "test-entry-123"
        async_add_entities = MagicMock()

        await async_setup_entry(hass, config_entry, async_add_entities)

        assert DOMAIN in hass.data
        entry_data = hass.data[DOMAIN]["test-entry-123"]
        assert "too_hot_switch" in entry_data
        assert "too_cold_switch" in entry_data
