"""Adds support for child thermostat units."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.climate import (
    PRESET_NONE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_TEMPERATURE,
    EVENT_HOMEASSISTANT_START,
    UnitOfTemperature,
)
from homeassistant.core import CoreState, Event, HomeAssistant, callback
from homeassistant.helpers.restore_state import RestoreEntity

_LOGGER = logging.getLogger(__name__)


class ChildThermostat(ClimateEntity, RestoreEntity):
    """Representation of a Child Thermostat device."""

    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        name: str,
        min_temp: float | None,
        max_temp: float | None,
        target_temp: float | None,
        precision: float | None,
        target_temperature_step: float | None,
        unit: UnitOfTemperature,
        unique_id: str | None,
    ) -> None:
        """Initialize the thermostat."""
        self._attr_name = name
        self._hvac_mode = HVACMode.AUTO
        self._hvac_action = HVACAction.OFF
        self._saved_target_temp = target_temp
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        self._attr_hvac_modes = [HVACMode.AUTO, HVACMode.COOL, HVACMode.HEAT]
        self._active = False
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE
        self._attr_preset_modes = [PRESET_NONE]

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        @callback
        def _async_startup(_: Event | None = None) -> None:
            """Init on startup."""

        if self.hass.state is CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Set default state to Auto
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.AUTO

    @property
    def precision(self) -> float:
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self) -> float:
        """Return the supported step of target temperature."""
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        # if a target_temperature_step is not defined, fallback to equal the precision
        return self.precision

    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        return self._hvac_action

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_child_state(
        self,
        parent_target_temperature: float,
        current_temperature: float,
        hvac_action: HVACAction,
    ) -> None:
        """Set current state for child thermometer."""
        if (
            self._hvac_action == hvac_action
            and self._attr_current_temperature == current_temperature
            and (
                self._hvac_mode in [HVACMode.HEAT, HVACMode.COOL]
                or self._target_temp == parent_target_temperature
            )
        ):
            # early exit as nothing has changed. This possibly avoids infinite loop as parent thermostat watches this state
            return

        if self._hvac_mode not in [HVACMode.HEAT, HVACMode.COOL]:
            self._target_temp = parent_target_temperature
        self._attr_current_temperature = current_temperature
        self._hvac_action = hvac_action
        self.async_write_ha_state()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.HEAT:
            self._hvac_mode = HVACMode.HEAT
        elif hvac_mode == HVACMode.COOL:
            self._hvac_mode = HVACMode.COOL
        elif hvac_mode == HVACMode.AUTO:
            self._hvac_mode = HVACMode.AUTO
        else:
            _LOGGER.error("Unrecognized hvac mode: %s", hvac_mode)
            return
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        self.async_write_ha_state()

    @property
    def min_temp(self) -> float:
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp
