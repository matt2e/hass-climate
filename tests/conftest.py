"""Shared test fixtures for matt_thermostat tests."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import (
    PRECISION_TENTHS,
    STATE_OFF,
    STATE_ON,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, State

from custom_components.matt_thermostat.child_thermostat import ChildThermostat
from custom_components.matt_thermostat.climate import (
    ParentThermostat,
    Room,
)
from custom_components.matt_thermostat.const import DOMAIN
from custom_components.matt_thermostat.switch import FeedbackSwitch

ROOMS_JSON = json.dumps(
    [
        {
            "name": "Living Room",
            "sensor": "sensor.living_room_temp",
            "cover": "cover.living_room_vent",
            "light": "light.living_room",
            "mode": "primary",
            "bedtime_mode": "secondary",
            "allows_override": True,
            "is_overflow": False,
            "vents": 2,
        },
        {
            "name": "Bedroom",
            "sensor": "sensor.bedroom_temp",
            "cover": "cover.bedroom_vent",
            "light": None,
            "mode": "secondary",
            "bedtime_mode": "primary",
            "allows_override": False,
            "is_overflow": False,
            "vents": 1,
        },
        {
            "name": "Office",
            "sensor": "sensor.office_temp",
            "cover": "cover.office_vent",
            "light": "light.office",
            "mode": "primary",
            "bedtime_mode": "disabled",
            "allows_override": True,
            "is_overflow": False,
            "vents": 2,
        },
    ]
)


def make_state(state: str, attributes: dict[str, Any] | None = None) -> State:
    """Create a mock HA State object."""
    s = MagicMock(spec=State)
    s.state = state
    s.attributes = attributes or {}
    return s


def make_hass(
    *,
    presence: str = STATE_ON,
    bedtime: str = STATE_OFF,
    manual: str = STATE_OFF,
    room_temps: dict[str, float] | None = None,
    cover_positions: dict[str, int] | None = None,
    light_states: dict[str, str] | None = None,
    real_climate_action: str = HVACAction.IDLE,
    real_climate_state: str = HVACMode.OFF,
) -> MagicMock:
    """Build a mock HomeAssistant with controllable entity states."""
    hass = MagicMock(spec=HomeAssistant)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.data = {}

    temps = room_temps or {}
    covers = cover_positions or {}
    lights = light_states or {}

    state_map: dict[str, State] = {
        "input_boolean.presence": make_state(presence),
        "input_boolean.bedtime": make_state(bedtime),
        "input_boolean.manual": make_state(manual),
        "climate.real_ac": make_state(
            real_climate_state,
            {"hvac_action": real_climate_action},
        ),
    }

    for entity_id, temp in temps.items():
        state_map[entity_id] = make_state(str(temp))

    for entity_id, pos in covers.items():
        state_map[entity_id] = make_state("open", {"current_position": pos})

    for entity_id, light_st in lights.items():
        state_map[entity_id] = make_state(light_st)

    hass.states = MagicMock()
    hass.states.get = lambda eid: state_map.get(eid)

    return hass


def make_rooms() -> list[Room]:
    """Create the standard test room list."""
    data = json.loads(ROOMS_JSON)
    return [Room.from_dict(item) for item in data]


def make_child_thermostats(
    hass: MagicMock, rooms: list[Room]
) -> dict[str, ChildThermostat]:
    """Create child thermostats for rooms that allow override."""
    children = {}
    for room in rooms:
        if room.allows_override:
            child = ChildThermostat(
                hass,
                name=f"{room.name} Thermostat",
                min_temp=16.0,
                max_temp=30.0,
                target_temp=22.0,
                precision=PRECISION_TENTHS,
                target_temperature_step=0.1,
                unit=UnitOfTemperature.CELSIUS,
                unique_id=f"test-{room.name}",
            )
            child.hass = hass
            child.async_write_ha_state = MagicMock()
            children[room.name] = child
    return children


def make_parent(
    hass: MagicMock | None = None,
    *,
    rooms: list[Room] | None = None,
    children: dict[str, ChildThermostat] | None = None,
    target_temp: float = 22.0,
    hvac_mode: HVACMode = HVACMode.COOL,
    cold_tolerance: float = 0.4,
    hot_tolerance: float = 0.4,
    cooling_temp_modifier: float = 0.0,
    heating_temp_modifier: float = 0.0,
) -> ParentThermostat:
    """Create a ParentThermostat for testing."""
    if hass is None:
        hass = make_hass()
    if rooms is None:
        rooms = make_rooms()
    if children is None:
        children = make_child_thermostats(hass, rooms)

    parent = ParentThermostat(
        hass,
        name="Test Thermostat",
        real_climate_entity_id="climate.real_ac",
        presence_entity_id="input_boolean.presence",
        bedtime_entity_id="input_boolean.bedtime",
        manual_entity_id="input_boolean.manual",
        output_entity_id="input_text.output",
        min_temp=16.0,
        max_temp=30.0,
        target_temp=target_temp,
        min_cycle_duration=None,
        cold_tolerance=cold_tolerance,
        hot_tolerance=hot_tolerance,
        cooling_temp_modifier=cooling_temp_modifier,
        heating_temp_modifier=heating_temp_modifier,
        initial_hvac_mode=hvac_mode,
        precision=PRECISION_TENTHS,
        target_temperature_step=0.1,
        unit=UnitOfTemperature.CELSIUS,
        unique_id="test-entry",
        rooms=rooms,
        child_thermostats=children,
    )
    parent.hass = hass
    parent.async_write_ha_state = MagicMock()
    parent._active = True
    return parent


def setup_feedback_switches(hass: MagicMock, unique_id: str = "test-entry"):
    """Create and register feedback switches in hass.data."""
    too_hot = FeedbackSwitch("Too Hot", f"{unique_id}-too-hot")
    too_cold = FeedbackSwitch("Too Cold", f"{unique_id}-too-cold")
    too_hot.hass = hass
    too_cold.hass = hass
    too_hot.async_write_ha_state = MagicMock()
    too_cold.async_write_ha_state = MagicMock()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(unique_id, {})
    hass.data[DOMAIN][unique_id]["too_hot_switch"] = too_hot
    hass.data[DOMAIN][unique_id]["too_cold_switch"] = too_cold
    return too_hot, too_cold
