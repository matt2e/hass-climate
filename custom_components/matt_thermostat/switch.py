"""Binary switches for Too Hot / Too Cold feedback."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class FeedbackSwitch(SwitchEntity):
    """A simple on/off switch for user comfort feedback."""

    _attr_should_poll = False

    def __init__(self, name: str, unique_id: str | None) -> None:
        """Initialize the switch."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._is_on = False
        self._on_turn_on_callback: Callable[[], Awaitable[None]] | None = None

    @property
    def is_on(self) -> bool:
        """Return true if the switch is on."""
        return self._is_on

    def set_on_turn_on_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set a callback to be called when the switch is turned on."""
        self._on_turn_on_callback = callback

    async def async_turn_on(self, **kwargs) -> None:
        """Turn the switch on."""
        self._is_on = True
        self.async_write_ha_state()
        if self._on_turn_on_callback is not None:
            await self._on_turn_on_callback()

    async def async_turn_off(self, **kwargs) -> None:
        """Turn the switch off."""
        self._is_on = False
        self.async_write_ha_state()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up switches from a config entry."""
    unique_id = config_entry.entry_id
    too_hot = FeedbackSwitch("Too Hot", f"{unique_id}-too-hot")
    too_cold = FeedbackSwitch("Too Cold", f"{unique_id}-too-cold")

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(unique_id, {})
    hass.data[DOMAIN][unique_id]["too_hot_switch"] = too_hot
    hass.data[DOMAIN][unique_id]["too_cold_switch"] = too_cold

    async_add_entities([too_hot, too_cold])
