"""Tests for ChildThermostat."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_TENTHS, UnitOfTemperature

from custom_components.matt_thermostat.child_thermostat import ChildThermostat

from .conftest import make_hass


def make_child(hass=None, target_temp=22.0) -> ChildThermostat:
    if hass is None:
        hass = make_hass()
    child = ChildThermostat(
        hass,
        name="Bedroom Thermostat",
        min_temp=16.0,
        max_temp=30.0,
        target_temp=target_temp,
        precision=PRECISION_TENTHS,
        target_temperature_step=0.1,
        unit=UnitOfTemperature.CELSIUS,
        unique_id="test-bedroom",
    )
    child.hass = hass
    child.async_write_ha_state = MagicMock()
    return child


class TestChildThermostatInit:
    def test_defaults(self):
        child = make_child()
        assert child.hvac_mode == HVACMode.AUTO
        assert child.hvac_action == HVACAction.OFF
        assert child.target_temperature == 22.0
        assert child.min_temp == 16.0
        assert child.max_temp == 30.0
        assert child.precision == PRECISION_TENTHS
        assert child.target_temperature_step == 0.1

    def test_none_min_max_falls_back_to_super(self):
        hass = make_hass()
        child = ChildThermostat(
            hass,
            name="Test",
            min_temp=None,
            max_temp=None,
            target_temp=22.0,
            precision=None,
            target_temperature_step=None,
            unit=UnitOfTemperature.CELSIUS,
            unique_id="test",
        )
        # super().min_temp and max_temp return HA defaults (7 and 35 for C)
        assert child.min_temp is not None
        assert child.max_temp is not None

    def test_none_precision_falls_back(self):
        hass = make_hass()
        # super().precision requires hass.config, so set it up
        hass.config = MagicMock()
        hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
        child = ChildThermostat(
            hass,
            name="Test",
            min_temp=16.0,
            max_temp=30.0,
            target_temp=22.0,
            precision=None,
            target_temperature_step=None,
            unit=UnitOfTemperature.CELSIUS,
            unique_id="test",
        )
        child.hass = hass
        # Falls back to super().precision
        assert child.precision is not None
        # target_temperature_step falls back to precision
        assert child.target_temperature_step == child.precision


class TestChildSetHvacMode:
    @pytest.mark.asyncio
    async def test_set_valid_modes(self):
        child = make_child()
        for mode in [HVACMode.HEAT, HVACMode.COOL, HVACMode.FAN_ONLY, HVACMode.AUTO]:
            await child.async_set_hvac_mode(mode)
            assert child.hvac_mode == mode
            child.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_set_invalid_mode_ignored(self):
        child = make_child()
        await child.async_set_hvac_mode(HVACMode.OFF)
        # Should stay AUTO (the default)
        assert child.hvac_mode == HVACMode.AUTO


class TestChildSetTemperature:
    @pytest.mark.asyncio
    async def test_set_temperature(self):
        child = make_child()
        await child.async_set_temperature(**{ATTR_TEMPERATURE: 25.0})
        assert child.target_temperature == 25.0
        child.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_set_temperature_no_temp_kwarg(self):
        child = make_child()
        await child.async_set_temperature(something_else=99)
        assert child.target_temperature == 22.0  # unchanged


class TestChildSetChildState:
    @pytest.mark.asyncio
    async def test_updates_state_in_auto_mode(self):
        child = make_child()
        await child.async_set_child_state(
            parent_target_temperature=23.0,
            current_temperature=21.5,
            hvac_action=HVACAction.COOLING,
        )
        assert child.target_temperature == 23.0
        assert child._attr_current_temperature == 21.5
        assert child.hvac_action == HVACAction.COOLING

    @pytest.mark.asyncio
    async def test_does_not_override_target_in_manual_mode(self):
        child = make_child(target_temp=20.0)
        await child.async_set_hvac_mode(HVACMode.COOL)
        child.async_write_ha_state.reset_mock()

        await child.async_set_child_state(
            parent_target_temperature=23.0,
            current_temperature=21.5,
            hvac_action=HVACAction.COOLING,
        )
        # Target should remain at 20.0 since child is in COOL mode
        assert child.target_temperature == 20.0
        assert child._attr_current_temperature == 21.5

    @pytest.mark.asyncio
    async def test_early_exit_no_change(self):
        child = make_child()
        await child.async_set_child_state(
            parent_target_temperature=23.0,
            current_temperature=21.5,
            hvac_action=HVACAction.COOLING,
        )
        child.async_write_ha_state.reset_mock()

        # Same call again — should early exit
        await child.async_set_child_state(
            parent_target_temperature=23.0,
            current_temperature=21.5,
            hvac_action=HVACAction.COOLING,
        )
        child.async_write_ha_state.assert_not_called()
