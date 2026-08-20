"""Adds support for matt thermostat units."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import voluptuous as vol
from homeassistant.components import input_text
from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PRESET_NONE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.components.climate import (
    PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    PRECISION_TENTHS,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
    AddEntitiesCallback,
)
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

from .child_thermostat import ChildThermostat
from .const import (
    CONF_BEDTIME,
    CONF_COLD_TOLERANCE,
    CONF_COOLING_TEMP_MODIFIER,
    CONF_HEATING_TEMP_MODIFIER,
    CONF_HOT_TOLERANCE,
    CONF_INITIAL_HVAC_MODE,
    CONF_MANUAL,
    CONF_MAX_TEMP,
    CONF_MIN_TEMP,
    CONF_OUTPUT_TEXT,
    CONF_PRESENCE,
    CONF_REAL_CLIMATE,
    CONF_ROOMS,
    CONF_TARGET_TEMP,
    DEFAULT_TEMP_MODIFIER,
    DEFAULT_TOLERANCE,
    DOMAIN,
    PLATFORMS,
    SUBENTRY_TYPE_ROOM,
)
from .switch import FeedbackSwitch

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Home Thermostat"

# how long a door must read continuously closed before a secondary room is
# disabled; opening the door re-enables the room immediately
DOOR_CLOSED_DELAY = timedelta(minutes=5)

# Fan cycling: when the AC is idle but a primary room is drifting toward a
# real cool/heat cycle and another room holds substantially better air, run
# the real unit in fan_only to circulate air and delay the compressor cycle.
FAN_CYCLE_SPREAD_ON = 4.0  # donor must be this much better off to start
FAN_CYCLE_SPREAD_OFF = 2.0  # stop once the spread collapses below this
FAN_CYCLE_RECOVERY_MARGIN = 0.3  # stop once the needy room is this far past target
FAN_CYCLE_MAX_RUNTIME = timedelta(minutes=45)  # hard cap per session
FAN_CYCLE_COOLDOWN = timedelta(minutes=30)  # no restart after capping out

# How long a room's temperature sensor may stay unavailable/unknown before we
# stop trusting the room and neutralize it (see _read_demand_room_temp). Brief
# dropouts under this window are tolerated so the room keeps its demand.
SENSOR_UNAVAILABLE_GRACE = timedelta(minutes=5)

# A sensor can keep publishing its last value long after it has actually
# stopped updating (dead battery, lost connectivity) without ever going
# unavailable. Once its state has not been written (last_reported) for longer
# than this, treat the reading as stale and neutralize the room until a fresh
# value arrives. The threshold sits above the slowest common battery-sensor
# heartbeat so healthy sensors in stable rooms are not false-triggered.
SENSOR_STALE_TIMEOUT = timedelta(minutes=20)


# Global config shared by the YAML platform and the UI config entry. The rooms
# field is deliberately excluded here because the two paths source rooms
# differently: YAML uses a single CONF_ROOMS JSON string, while the UI stores
# one "room" subentry per room (see async_setup_entry / async_setup_platform).
_COMMON_SCHEMA = {
    vol.Required(CONF_REAL_CLIMATE): cv.entity_id,
    vol.Required(CONF_PRESENCE): cv.entity_id,
    vol.Required(CONF_MANUAL): cv.entity_id,
    vol.Required(CONF_BEDTIME): cv.entity_id,
    vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
    vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
    vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
    vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
    vol.Optional(CONF_OUTPUT_TEXT): cv.string,
    vol.Optional(CONF_COOLING_TEMP_MODIFIER, default=DEFAULT_TEMP_MODIFIER): vol.Coerce(
        float
    ),
    vol.Optional(CONF_HEATING_TEMP_MODIFIER, default=DEFAULT_TEMP_MODIFIER): vol.Coerce(
        float
    ),
    vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
        [HVACMode.FAN_ONLY, HVACMode.COOL, HVACMode.HEAT, HVACMode.OFF]
    ),
    vol.Optional(CONF_UNIQUE_ID): cv.string,
}

# UI/config-entry options: rooms live in subentries, so CONF_ROOMS is absent.
CONFIG_ENTRY_SCHEMA_COMMON = vol.Schema(_COMMON_SCHEMA)

# Legacy YAML platform: rooms are still a single CONF_ROOMS JSON string.
PLATFORM_SCHEMA_COMMON = vol.Schema(
    {
        **_COMMON_SCHEMA,
        vol.Required(CONF_ROOMS): cv.string,
    }
)


PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend(PLATFORM_SCHEMA_COMMON.schema)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Initialize config entry.

    The UI path builds one Room per "room" config subentry, each with its own
    entity pickers and validation. The legacy YAML path (async_setup_platform)
    instead parses the CONF_ROOMS JSON string; that divergence is intentional
    so existing YAML setups keep working.
    """
    rooms: list[Room] = []
    room_subentry_ids: dict[str, str] = {}
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_ROOM:
            continue
        room = Room.from_subentry(subentry.data)
        rooms.append(room)
        room_subentry_ids[room.name] = subentry.subentry_id

    await _async_setup_config(
        hass,
        CONFIG_ENTRY_SCHEMA_COMMON(dict(config_entry.options)),
        config_entry.entry_id,
        rooms,
        async_add_entities,
        room_subentry_ids=room_subentry_ids,
    )


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the parent thermostat platform (legacy YAML)."""

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    # Legacy YAML keeps rooms as a single CONF_ROOMS JSON string; the UI path
    # now uses per-room subentries instead (see async_setup_entry).
    data = json.loads(config[CONF_ROOMS])
    if not isinstance(data, list):
        raise TypeError("Expected a list of rooms")
    rooms = [Room.from_dict(item) for item in data]

    await _async_setup_config(
        hass, config, config.get(CONF_UNIQUE_ID), rooms, async_add_entities
    )


async def _async_setup_config(
    hass: HomeAssistant,
    config: Mapping[str, Any],
    unique_id: str | None,
    rooms: list[Room],
    async_add_entities: AddEntitiesCallback | AddConfigEntryEntitiesCallback,
    room_subentry_ids: Mapping[str, str] | None = None,
) -> None:
    """Set up the parent thermostat platform from an already-built rooms list."""

    name: str = config[CONF_NAME]
    real_climate_entity_id: str = config[CONF_REAL_CLIMATE]
    bedtime_entity_id: str = config[CONF_BEDTIME]
    presence_entity_id: str = config[CONF_PRESENCE]
    manual_entity_id: str = config[CONF_MANUAL]
    output_entity_id: str | None = config.get(CONF_OUTPUT_TEXT)
    min_temp: float | None = config.get(CONF_MIN_TEMP)
    max_temp: float | None = config.get(CONF_MAX_TEMP)
    target_temp: float | None = config.get(CONF_TARGET_TEMP)
    cold_tolerance: float = config[CONF_COLD_TOLERANCE]
    hot_tolerance: float = config[CONF_HOT_TOLERANCE]
    cooling_temp_modifier: float = config.get(CONF_COOLING_TEMP_MODIFIER, 0.0)
    heating_temp_modifier: float = config.get(CONF_HEATING_TEMP_MODIFIER, 0.0)
    initial_hvac_mode: HVACMode | None = config.get(CONF_INITIAL_HVAC_MODE)
    unit = hass.config.units.temperature_unit

    precision: float = PRECISION_TENTHS
    target_temperature_step = 0.1

    # Create a map of room name to ChildThermostat for rooms that allow override
    child_thermostats: dict[str, ChildThermostat] = {
        room.name: ChildThermostat(
            hass,
            name=f"{room.name} Thermostat",
            min_temp=min_temp,
            max_temp=max_temp,
            target_temp=target_temp,
            precision=precision,
            target_temperature_step=target_temperature_step,
            unit=unit,
            unique_id=f"{unique_id}-{room.name}",
        )
        for room in rooms
        if room.allows_override
    }

    # Add child thermostats, which gives them each an entity id. On the UI path
    # each override room is a config subentry, so bind its child thermostat to
    # that subentry; the YAML path has no subentries and adds them plainly. The
    # unique_id is unchanged either way so existing entity ids / history survive.
    subentry_ids = room_subentry_ids or {}
    for room_name, child in child_thermostats.items():
        subentry_id = subentry_ids.get(room_name)
        if subentry_id is not None:
            async_add_entities([child], config_subentry_id=subentry_id)
        else:
            async_add_entities([child])

    parent = ParentThermostat(
        hass,
        name=name,
        real_climate_entity_id=real_climate_entity_id,
        presence_entity_id=presence_entity_id,
        bedtime_entity_id=bedtime_entity_id,
        manual_entity_id=manual_entity_id,
        output_entity_id=output_entity_id,
        min_temp=min_temp,
        max_temp=max_temp,
        target_temp=target_temp,
        cold_tolerance=cold_tolerance,
        hot_tolerance=hot_tolerance,
        cooling_temp_modifier=cooling_temp_modifier,
        heating_temp_modifier=heating_temp_modifier,
        initial_hvac_mode=initial_hvac_mode,
        precision=precision,
        target_temperature_step=target_temperature_step,
        unit=unit,
        unique_id=unique_id,
        rooms=rooms,
        child_thermostats=child_thermostats,
    )

    async_add_entities([parent])


class RoomMode(StrEnum):
    """Room modes."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DISABLED = "disabled"

    # internal modes
    CUSTOM = "custom"


@dataclass
class RoomState:
    """Room states."""

    mode: RoomMode = RoomMode.DISABLED
    cover_pos: int = 0
    reached_target_at: datetime | None = None
    reached_half_max_at: datetime | None = None
    reached_max_at: datetime | None = None
    # is true when room reaches max temp or stays above target temp
    # for long enough, and stays on until it falls below tolerance
    is_satisfied: bool = False
    # value delayed based on raw_light_on_at and raw_light_off_at
    light_on: bool = False
    raw_light_on_at: datetime | None = None
    raw_light_off_at: datetime | None = None
    # debounced door-closed value, driven by raw_door_closed_at
    door_closed: bool = False
    raw_door_closed_at: datetime | None = None
    # set only when the secondary gate converts SECONDARY -> DISABLED
    disabled_by_door: bool = False
    # tracks when the room's sensor first went unavailable; None while readable
    sensor_unavailable_since: datetime | None = None
    # true once an unavailable sensor past the grace period has been logged;
    # cleared when it reports again, so each outage warns exactly once
    sensor_unavailable_warned: bool = False
    # true once a stale sensor has been logged; cleared when it reports again,
    # so each outage warns exactly once
    sensor_stale_warned: bool = False


@dataclass
class Room:
    """A room configuration for the Matt Thermostat integration.

    Attributes:
        name: The name of the room.
        sensor_entity: The entity ID of the temperature sensor for the room.
        cover_entity: The entity ID of the cover (e.g., blinds) for the room.
        light_entity: The entity ID of the light for the room, or None if not used.
        standard_mode: The standard mode for the room.
        bedtime_mode: The bedtime mode for the room.
        allows_override: Whether custom thermostats are made for this room.
        door_entity: The entity ID of the door sensor for the room, or None if
            not used. When set and the (debounced) door reads closed, a room
            resolving to SECONDARY is disabled instead.
    """

    name: str
    sensor_entity: str
    cover_entity: str
    light_entity: str | None
    standard_mode: RoomMode
    bedtime_mode: RoomMode
    allows_override: bool
    is_overflow: bool
    vents: int
    door_entity: str | None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Room:
        """Read Room from json dict."""

        if "name" not in data:
            raise ValueError(
                "Room must have 'name' and 'cover' and 'mode' and 'sensor'"
            )
        name = str(data["name"])
        if (
            "cover" not in data
            or "mode" not in data
            or "sensor" not in data
            or "bedtime_mode" not in data
        ):
            raise ValueError(
                f"Room '{name}' must have 'cover' and "
                f"'mode' and 'sensor' and 'bedtime_mode'"
            )
        mode = str(data["mode"])
        if mode not in (RoomMode.PRIMARY, RoomMode.SECONDARY, RoomMode.DISABLED):
            raise ValueError(
                f"Invalid mode: {mode}, expected one of "
                f"{RoomMode.PRIMARY}, {RoomMode.SECONDARY}, "
                f"{RoomMode.DISABLED}"
            )
        bedtime_mode = str(data["bedtime_mode"])
        if bedtime_mode not in (
            RoomMode.PRIMARY,
            RoomMode.SECONDARY,
            RoomMode.DISABLED,
        ):
            raise ValueError(
                f"Invalid bedtime mode: {bedtime_mode}, "
                f"expected one of {RoomMode.PRIMARY}, "
                f"{RoomMode.SECONDARY}, {RoomMode.DISABLED}"
            )
        return Room(
            name=name,
            sensor_entity=str(data["sensor"]),
            cover_entity=str(data["cover"]),
            light_entity=data.get("light"),
            standard_mode=mode,
            bedtime_mode=bedtime_mode,
            allows_override=bool(data.get("allows_override", False)),
            is_overflow=bool(data.get("is_overflow", False)),
            vents=int(data.get("vents", 1)),
            door_entity=data.get("door"),
        )

    @staticmethod
    def from_subentry(data: Mapping[str, Any]) -> Room:
        """Build a Room from a config subentry's data mapping.

        The per-room ``CONF_*`` subentry keys deliberately match the legacy
        room-dict keys, so this reuses ``from_dict`` (and its RoomMode
        validation) rather than duplicating the field wiring.
        """
        return Room.from_dict(dict(data))


class ParentThermostat(ClimateEntity, RestoreEntity):
    """Representation of a Parent Thermostat device."""

    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        name: str,
        real_climate_entity_id: str,
        presence_entity_id: str,
        bedtime_entity_id: str,
        manual_entity_id: str,
        output_entity_id: str | None,
        min_temp: float | None,
        max_temp: float | None,
        target_temp: float | None,
        cold_tolerance: float,
        hot_tolerance: float,
        cooling_temp_modifier: float,
        heating_temp_modifier: float,
        initial_hvac_mode: HVACMode | None,
        precision: float | None,
        target_temperature_step: float | None,
        unit: UnitOfTemperature,
        unique_id: str | None,
        rooms: list[Room],
        child_thermostats: dict[str, ChildThermostat],
    ) -> None:
        """Initialize the thermostat."""
        self._attr_name = name
        self._rooms = rooms
        self._child_thermostats = child_thermostats
        self._real_climate_entity_id = real_climate_entity_id
        self._bedtime_entity_id = bedtime_entity_id
        self._presence_entity_id = presence_entity_id
        self._manual_entity_id = manual_entity_id
        self._output_entity_id = output_entity_id
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._cooling_temp_modifier = cooling_temp_modifier
        self._heating_temp_modifier = heating_temp_modifier
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        self._attr_hvac_modes = [
            HVACMode.OFF,
            HVACMode.FAN_ONLY,
            HVACMode.COOL,
            HVACMode.HEAT,
        ]
        self._active = False
        self._last_device_active = False
        self._fan_cycle_active = False
        self._fan_cycle_started_at: datetime | None = None
        self._fan_cycle_blocked_until: datetime | None = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._attr_supported_features = (
            ClimateEntityFeature.TARGET_TEMPERATURE
            | ClimateEntityFeature.TURN_OFF
            | ClimateEntityFeature.TURN_ON
        )
        self._attr_preset_modes = [PRESET_NONE]
        self._room_states: dict[str, RoomState] = {
            room.name: RoomState() for room in rooms
        }
        self._sensor_found_for_room: dict[str, bool] = {
            room.name: False for room in rooms
        }

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                (child.entity_id for child in self._child_thermostats.values()),
                self._async_child_thermostat_changed,
            )
        )

        self.async_on_remove(
            async_track_time_interval(
                self.hass,
                self._async_poll_for_changes,
                timedelta(seconds=30),
            )
        )

        # Register callback so turning on a feedback switch triggers
        # the control loop immediately instead of waiting for the next poll.
        too_hot_switch, too_cold_switch = self._get_feedback_switches()
        if too_hot_switch is not None:
            too_hot_switch.set_on_turn_on_callback(self._async_poll_for_changes)
        if too_cold_switch is not None:
            too_cold_switch.set_on_turn_on_callback(self._async_poll_for_changes)

        # Check If we have an old state
        if (old_state := await self.async_get_last_state()) is not None:
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    self._target_temp = 20
                    _LOGGER.warning(
                        "Undefined target temperature, falling back to %s",
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if (
                self.preset_modes
                and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes
            ):
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = HVACMode(old_state.state)

        else:
            # No previous state, try and restore defaults
            if self._target_temp is None:
                self._target_temp = 20.0
            _LOGGER.warning(
                "No previously saved temperature, setting to %s", self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

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
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE
        if self._fan_cycle_active:
            return HVACAction.FAN
        if self._hvac_mode == HVACMode.COOL:
            return HVACAction.COOLING
        if self._hvac_mode == HVACMode.HEAT:
            return HVACAction.HEATING
        if self._hvac_mode == HVACMode.FAN_ONLY:
            return HVACAction.FAN
        return HVACAction.IDLE

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode in {HVACMode.OFF, HVACMode.COOL, HVACMode.HEAT, HVACMode.FAN_ONLY}:
            self._hvac_mode = hvac_mode
            self._end_fan_cycle(clear_block=True)
        else:
            _LOGGER.error("Unrecognized hvac mode: %s", hvac_mode)
            return
        await self._async_control_real_climate(force=True)
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        await self._async_control_real_climate(force=True)
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

    async def _async_child_thermostat_changed(self, event: Any) -> None:
        """Called each time a child thermostat changes."""
        await self._async_control_real_climate()
        self.async_write_ha_state()

    async def _async_poll_for_changes(self, time: datetime | None = None) -> None:
        """Called each time we check for changes."""
        await self._async_control_real_climate()
        self.async_write_ha_state()

    async def _async_control_real_climate(self, force: bool = False) -> None:
        """Check if we need to turn heating on or off."""

        async with self._temp_lock:
            if not self._active and self._target_temp is not None:
                self._active = True

            if not self._active:
                return

            # _active is only set True once _target_temp is non-None, and
            # async_set_temperature never assigns None, so this is always safe.
            assert self._target_temp is not None

            presence_state = self.hass.states.get(self._presence_entity_id)
            presence = presence_state is not None and presence_state.state == STATE_ON
            manual_state = self.hass.states.get(self._manual_entity_id)
            manual = manual_state is not None and manual_state.state == STATE_ON

            if manual:
                self._end_fan_cycle()
                self._reset_all_room_states()
                return

            # --- Capture user comfort feedback ---
            too_hot, too_cold = await self._capture_and_reset_feedback()

            if not presence or self._hvac_mode == HVACMode.OFF:
                self._reset_all_room_states()
                # Intentionally do not return here — we still need to drive the
                # underlying AC and covers to their off/idle states. Child
                # thermostats may also be in a manual mode that requires the AC
                # to remain on even when the parent is away or set to OFF.

            # --- Control loop ---
            custom_rooms = []
            primary_rooms = []
            secondary_rooms = []
            disabled_rooms = []
            primary_current_temp: float | None = None
            most_extreme_temperature = self._target_temp

            # --- Put each room into the correct sub list ---
            for room in self._rooms:
                real_mode = self._calculate_room_mode(room, presence)

                self._transition_room_to_mode(room, real_mode)

                if real_mode == RoomMode.PRIMARY:
                    primary_rooms.append(room)
                elif real_mode == RoomMode.SECONDARY:
                    secondary_rooms.append(room)
                elif real_mode == RoomMode.CUSTOM:
                    custom_rooms.append(room)
                else:
                    disabled_rooms.append(room)

            # --- Apply user comfort feedback ---
            if (too_hot or too_cold) and self._hvac_mode in {
                HVACMode.COOL,
                HVACMode.HEAT,
                HVACMode.FAN_ONLY,
            }:
                self._apply_comfort_feedback(
                    too_hot,
                    too_cold,
                    primary_rooms,
                    secondary_rooms,
                    disabled_rooms,
                    # a fan-cycling unit reads as "active" but is not
                    # actually cooling/heating
                    is_active=self._is_device_active and not self._fan_cycle_active,
                )

            for room in disabled_rooms:
                await self.hass.services.async_call(
                    "cover",
                    "set_cover_position",
                    {"entity_id": room.cover_entity, "position": 0},
                    blocking=False,
                )

            for room in custom_rooms:
                child_thermo = self._child_thermostats.get(room.name)

                target_temp = child_thermo.target_temperature or self._target_temp
                if self._hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}:
                    most_extreme_temperature = min(
                        most_extreme_temperature, target_temp
                    )
                else:
                    most_extreme_temperature = max(
                        most_extreme_temperature, target_temp
                    )

                current_temp = await self._read_demand_room_temp(room)
                if current_temp is None:
                    continue

                await self.async_update_room(
                    room=room, current_temp=current_temp, target_temp=target_temp
                )

            for room in primary_rooms:
                current_temp = await self._read_demand_room_temp(room)
                if current_temp is None:
                    continue
                if primary_current_temp is None:
                    primary_current_temp = current_temp
                elif self._hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}:
                    primary_current_temp = max(primary_current_temp, current_temp)
                else:
                    primary_current_temp = min(primary_current_temp, current_temp)

                await self.async_update_room(
                    room=room, current_temp=current_temp, target_temp=self._target_temp
                )

            if self._hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}:
                most_extreme_temperature = (
                    math.floor(most_extreme_temperature) + self._cooling_temp_modifier
                )
            else:
                most_extreme_temperature = (
                    math.ceil(most_extreme_temperature) + self._heating_temp_modifier
                )

            # --- Apply AC mode, temp, and fan speed ---
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {
                    "entity_id": self._real_climate_entity_id,
                    ATTR_TEMPERATURE: most_extreme_temperature,
                },
                blocking=False,
            )

            if primary_current_temp is not None:
                self._attr_current_temperature = primary_current_temp
            else:
                # Still update current temperature for display when
                # rooms are disabled (e.g. not home or HVAC off) by
                # reading directly from primary-configured rooms
                display_temp: float | None = None
                for room in self._rooms:
                    if room.standard_mode != RoomMode.PRIMARY:
                        continue
                    sensor_state = self.hass.states.get(room.sensor_entity)
                    if sensor_state is None or sensor_state.state in (
                        STATE_UNAVAILABLE,
                        STATE_UNKNOWN,
                    ):
                        continue
                    temp = float(sensor_state.state)
                    display_temp = (
                        temp if display_temp is None else max(display_temp, temp)
                    )
                if display_temp is not None:
                    self._attr_current_temperature = display_temp

            fan_speed = self.calculate_fan_speed()
            if not fan_speed:
                donor_rooms = self._should_fan_cycle(
                    presence, primary_rooms, secondary_rooms, custom_rooms
                )
                if donor_rooms:
                    await self._async_fan_cycle(donor_rooms)
                else:
                    self._end_fan_cycle()
                    await self.hass.services.async_call(
                        "climate",
                        "turn_off",
                        {"entity_id": self._real_climate_entity_id},
                        blocking=False,
                    )
                    # no air is being pushed, so seal every vent to keep
                    # ambient air (e.g. shower steam) out of the ducts
                    await self._close_all_vents()
            else:
                if self._fan_cycle_active or not self._is_device_active:
                    # we are turning AC on (a fan-cycling unit counts as off),
                    # so mark all rooms as not satisfied
                    for room_state in self._room_states.values():
                        room_state.is_satisfied = False
                # a real cool/heat cycle moves the room temps the cooldown was
                # guarding against, so drop the block along with the session
                self._end_fan_cycle(clear_block=True)

                await self.hass.services.async_call(
                    "climate",
                    "set_fan_mode",
                    {"entity_id": self._real_climate_entity_id, "fan_mode": fan_speed},
                    blocking=False,
                )

                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {
                        "entity_id": self._real_climate_entity_id,
                        "hvac_mode": self._hvac_mode.value,
                    },
                    blocking=False,
                )

                # --- AC is on so activate secondary rooms as needed ---
                await self.async_update_secondary_rooms(secondary_rooms)

            await self._async_update_child_thermostats()
            await self._async_update_output()

    def _transition_room_to_mode(self, room: Room, mode: RoomMode) -> None:
        room_state = self._room_states[room.name]
        if room_state.mode == mode:
            return

        room_state.mode = mode

        sensor_state = self.hass.states.get(room.sensor_entity)
        if sensor_state is None or sensor_state.state in (
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
        ):
            return
        current_temp = float(sensor_state.state)

        # we need to update whether the room is satisfied based on
        # if it is in the target temp range
        if mode == RoomMode.DISABLED:
            room_state.is_satisfied = False
            return
        if mode == RoomMode.PRIMARY:
            target_temp = self._target_temp
        elif mode == RoomMode.CUSTOM:
            target_temp = self._child_thermostats[room.name].target_temperature
        elif mode == RoomMode.SECONDARY:
            target_temp = self._target_secondary_temp()
        else:
            return

        if self._hvac_mode == HVACMode.HEAT:
            room_state.is_satisfied = current_temp >= target_temp - self._cold_tolerance
        elif self._hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}:
            room_state.is_satisfied = current_temp <= target_temp + self._hot_tolerance

    def _reset_all_room_states(self) -> None:
        """Reset all room states to their initial values."""
        for name in self._room_states:
            self._room_states[name] = RoomState()

    def _calculate_room_mode(self, room: Room, presence: bool) -> RoomMode:
        """Calculate the current mode of a room."""
        if not presence or self._hvac_mode == HVACMode.OFF:
            return RoomMode.DISABLED

        if room.name in self._child_thermostats:
            child_thermo = self._child_thermostats[room.name]
            if child_thermo.hvac_mode in [
                HVACMode.HEAT,
                HVACMode.COOL,
                HVACMode.FAN_ONLY,
            ]:
                return RoomMode.CUSTOM

        room_state = self._room_states[room.name]
        self._update_door_state(room, room_state)
        # recomputed by _gate_secondary; cleared here so config/away disables
        # are never mislabeled as door-disabled
        room_state.disabled_by_door = False

        bedtime_state = self.hass.states.get(self._bedtime_entity_id)
        bedtime = bedtime_state is not None and bedtime_state.state == STATE_ON
        mode = room.bedtime_mode if bedtime else room.standard_mode

        if mode == RoomMode.SECONDARY:
            return self._gate_secondary(room_state)
        if mode == RoomMode.PRIMARY:
            if room.light_entity:
                light_state = self.hass.states.get(room.light_entity)
                if light_state is None or light_state.state in [
                    STATE_OFF,
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
                ]:
                    room_state.raw_light_on_at = None
                    if room_state.raw_light_off_at is None:
                        room_state.raw_light_off_at = datetime.now()
                    elif room_state.raw_light_off_at <= datetime.now() - timedelta(
                        minutes=2
                    ):
                        room_state.light_on = False
                else:
                    room_state.raw_light_off_at = None
                    if room_state.raw_light_on_at is None:
                        room_state.raw_light_on_at = datetime.now()
                    elif room_state.raw_light_on_at <= datetime.now() - timedelta(
                        minutes=2
                    ):
                        room_state.light_on = True

            if room.light_entity and not bedtime and not room_state.light_on:
                return self._gate_secondary(room_state)

            return RoomMode.PRIMARY

        return RoomMode.DISABLED

    def _update_door_state(self, room: Room, room_state: RoomState) -> None:
        """Debounce raw door sensor into room_state.door_closed.

        Fail-open: missing/unavailable/unknown sensor is treated as OPEN.
        Asymmetric: closing requires DOOR_CLOSED_DELAY of continuous-closed
        before it counts; opening re-enables immediately.
        """
        if not room.door_entity:
            room_state.door_closed = False
            return

        door_state = self.hass.states.get(room.door_entity)
        # device_class "door": off = closed. Everything else (on/open,
        # unavailable, unknown, missing) => treat as OPEN (fail-open).
        raw_closed = door_state is not None and door_state.state == STATE_OFF

        if raw_closed:
            if room_state.raw_door_closed_at is None:
                room_state.raw_door_closed_at = datetime.now()
            elif room_state.raw_door_closed_at <= datetime.now() - DOOR_CLOSED_DELAY:
                room_state.door_closed = True
        else:
            room_state.raw_door_closed_at = None
            room_state.door_closed = False

    def _gate_secondary(self, room_state: RoomState) -> RoomMode:
        """Disable a secondary room when its door is (debounced) closed."""
        if room_state.door_closed:
            room_state.disabled_by_door = True
            return RoomMode.DISABLED
        room_state.disabled_by_door = False
        return RoomMode.SECONDARY

    def _target_secondary_temp(self) -> float | None:
        if self.hvac_mode == HVACMode.HEAT:
            return max(self._target_temp - 2, 16)
        if self.hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}:
            return min(self._target_temp + 2, 28)
        return self._target_temp

    async def _read_demand_room_temp(self, room: Room) -> float | None:
        """Read a room's sensor for the airflow-gating loops.

        Returns the current temperature, or None if the sensor is
        unavailable/unknown or has gone stale. Tracks how long the sensor has
        been out and, once the grace period lapses, neutralizes the room so a
        dead sensor can no longer hold the AC on (primary/custom) or a vent
        open (secondary). A sensor that keeps its last numeric value but stops
        reporting (last_reported older than SENSOR_STALE_TIMEOUT) is treated
        the same way.
        """
        room_state = self._room_states[room.name]
        sensor_state = self.hass.states.get(room.sensor_entity)
        if sensor_state is None or sensor_state.state in (
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
        ):
            if room_state.sensor_unavailable_since is None:
                room_state.sensor_unavailable_since = datetime.now()
            elif (
                datetime.now() - room_state.sensor_unavailable_since
                >= SENSOR_UNAVAILABLE_GRACE
            ):
                if not room_state.sensor_unavailable_warned:
                    _LOGGER.warning(
                        "Room %s temperature sensor %s has been unavailable for "
                        "over %s; neutralizing the room until it reports again",
                        room.name,
                        room.sensor_entity,
                        SENSOR_UNAVAILABLE_GRACE,
                    )
                    room_state.sensor_unavailable_warned = True
                await self._neutralize_room(room)
            return None

        # A readable state means the sensor is no longer unavailable.
        room_state.sensor_unavailable_since = None
        room_state.sensor_unavailable_warned = False

        # last_reported is bumped on every write, even when the value is
        # unchanged, so its age is exactly how long the sensor has been silent.
        # It is tz-aware UTC, so compare against dt_util.utcnow() (naive
        # datetime.now() would raise). If it stays silent past the timeout,
        # stop trusting the frozen value and neutralize the room.
        last_reported = sensor_state.last_reported
        if (
            last_reported is not None
            and dt_util.utcnow() - last_reported >= SENSOR_STALE_TIMEOUT
        ):
            if not room_state.sensor_stale_warned:
                _LOGGER.warning(
                    "Room %s temperature sensor %s has not reported for over "
                    "%s; neutralizing the room until it updates again",
                    room.name,
                    room.sensor_entity,
                    SENSOR_STALE_TIMEOUT,
                )
                room_state.sensor_stale_warned = True
            await self._neutralize_room(room)
            return None

        room_state.sensor_stale_warned = False
        return float(sensor_state.state)

    async def _neutralize_room(self, room: Room) -> None:
        """Stop a room with a dead sensor from generating AC demand."""
        room_state = self._room_states[room.name]
        room_state.is_satisfied = True
        if room_state.cover_pos != 0:
            room_state.cover_pos = 0
            await self.hass.services.async_call(
                "cover",
                "set_cover_position",
                {"entity_id": room.cover_entity, "position": 0},
                blocking=False,
            )

    async def _close_all_vents(self) -> None:
        """Close every room's vent; the unit is delivering no air this cycle."""
        for room in self._rooms:
            room_state = self._room_states[room.name]
            # Skip based on the vent's physical position, not the in-memory
            # model: after a HA restart cover_pos defaults to 0 while a vent may
            # be left open, and external moves make the model stale. Trusting
            # the model would silently defeat this safety seal. Write the model
            # to 0 unconditionally so it stays authoritative next cycle.
            cover_state = self.hass.states.get(room.cover_entity)
            cover_pos = (
                cover_state.attributes.get("current_position", 0) if cover_state else 0
            )
            room_state.cover_pos = 0
            if cover_pos == 0:
                continue
            await self.hass.services.async_call(
                "cover",
                "set_cover_position",
                {"entity_id": room.cover_entity, "position": 0},
                blocking=False,
            )

    async def async_update_secondary_rooms(self, secondary_rooms: list[Room]) -> None:
        """Update secondary rooms when other rooms need the AC on."""
        target_temp_secondary = self._target_secondary_temp()
        for room in secondary_rooms:
            current_temp = await self._read_demand_room_temp(room)
            if current_temp is None:
                continue

            # Pull secondary rooms toward the primary target while the AC is
            # already running for primaries, but judge "satisfied" at the
            # looser secondary bound.
            await self.async_update_room(
                room=room,
                current_temp=current_temp,
                target_temp=self._target_temp,
                satisfied_target=target_temp_secondary,
            )

    async def async_update_room(
        self,
        room: Room,
        current_temp: float,
        target_temp: float,
        satisfied_target: float | None = None,
    ) -> None:
        """Update the target temperature for a room.

        target_temp drives cover position (how hard to pull the room toward
        a temperature). satisfied_target drives the is_satisfied bar (the
        comfort bound the user agreed to). For primary/custom rooms these
        match; for secondary rooms they differ so the room can ride a
        primary-driven cycle while still being judged "done" at its own
        looser bound.
        """
        if satisfied_target is None:
            satisfied_target = target_temp

        is_first_temp_reading = not self._sensor_found_for_room[room.name]
        if is_first_temp_reading:
            self._sensor_found_for_room[room.name] = True

        if self._hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}:
            # The more cooling needed the higher the diff
            diff = current_temp - target_temp
            satisfied_diff = current_temp - satisfied_target

            min_diff = -1 * self._cold_tolerance  # when to cut off cooling
            max_diff = self._hot_tolerance  # when to go full cooling
        else:
            # The more heating needed the higher the diff
            diff = target_temp - current_temp
            satisfied_diff = satisfied_target - current_temp
            min_diff = -1 * self._hot_tolerance  # when to cut off heating
            max_diff = self._cold_tolerance  # when to go full heating

        cover_state = self.hass.states.get(room.cover_entity)
        cover_pos = (
            cover_state.attributes.get("current_position", 0) if cover_state else 0
        )

        room_state = self._room_states[room.name]
        if satisfied_diff > max_diff:
            room_state.is_satisfied = False
        elif is_first_temp_reading:
            # When first launched, count any rooms within the range as satisfied
            room_state.is_satisfied = True

        if satisfied_diff > 0:
            room_state.reached_target_at = None
        elif room_state.reached_target_at is None:
            room_state.reached_target_at = datetime.now()

        if satisfied_diff >= min_diff:
            room_state.reached_max_at = None
        elif room_state.reached_max_at is None:
            room_state.reached_max_at = datetime.now()
            room_state.is_satisfied = True

        if diff > min_diff:
            desired_cover_pos = 100
            room_state.reached_half_max_at = None
        else:
            if room_state.reached_half_max_at is None:
                room_state.reached_half_max_at = datetime.now()
            elif not room_state.is_satisfied and (
                datetime.now() - room_state.reached_half_max_at
            ) >= timedelta(minutes=5):
                # spent enough time above half max, treat as satisfied
                room_state.is_satisfied = True

            if diff <= min_diff:
                desired_cover_pos = 0
            elif room_state.is_satisfied:
                desired_cover_pos = 50
            else:
                desired_cover_pos = 100

        room_state.cover_pos = desired_cover_pos
        if desired_cover_pos != cover_pos:
            await self.hass.services.async_call(
                "cover",
                "set_cover_position",
                {"entity_id": room.cover_entity, "position": desired_cover_pos},
                blocking=False,
            )

    def calculate_fan_speed(self) -> str | None:
        """Figure out what fan speed to use."""
        # Determine if HVAC should be on
        is_on = False
        vents = 0.0
        for room in self._rooms:
            state = self._room_states[room.name]
            if state.mode == RoomMode.DISABLED:
                continue

            vents += float(state.cover_pos) / 100.0 * float(room.vents)
            if (
                state.mode in [RoomMode.PRIMARY, RoomMode.CUSTOM]
                and not state.is_satisfied
            ):
                is_on = True

        if not is_on:
            return None

        scale = 0.5 if self._hvac_mode == HVACMode.COOL else 1.0

        if vents * scale >= 5.0:
            return "high"
        if vents * scale >= 3.0:
            return "medium"
        return "auto"

    def _room_temp(self, room: Room) -> float | None:
        """Read a room's current temperature, or None if it can't be trusted.

        Returns None when the sensor is unavailable/unknown or has gone stale
        (last_reported older than SENSOR_STALE_TIMEOUT), mirroring
        _read_demand_room_temp. A frozen reading from a dead sensor must not be
        trusted for fan-cycling decisions any more than for demand: otherwise a
        room the demand loop just neutralized would still look "needy" here and
        keep the unit fan-cycling instead of letting it turn off.
        """
        sensor_state = self.hass.states.get(room.sensor_entity)
        if sensor_state is None or sensor_state.state in (
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
        ):
            return None
        last_reported = sensor_state.last_reported
        if (
            last_reported is not None
            and dt_util.utcnow() - last_reported >= SENSOR_STALE_TIMEOUT
        ):
            return None
        return float(sensor_state.state)

    def _should_fan_cycle(
        self,
        presence: bool,
        primary_rooms: list[Room],
        secondary_rooms: list[Room],
        custom_rooms: list[Room],
    ) -> list[Room]:
        """Decide whether to circulate air by fan cycling the real unit.

        Only called when no primary/custom room demands a real cool/heat
        cycle. Fan cycling runs when a primary/custom room is at/past its
        target (drifting toward a real cycle) and another room holds
        substantially better air. Returns the donor rooms whose covers should
        open while cycling, or an empty list when fan cycling should not run.
        """
        if self._hvac_mode not in {HVACMode.COOL, HVACMode.HEAT}:
            return []
        if not presence:
            return []

        now = datetime.now()
        if (
            self._fan_cycle_blocked_until is not None
            and now < self._fan_cycle_blocked_until
        ):
            return []

        if self._fan_cycle_active:
            if (
                self._fan_cycle_started_at is not None
                and now - self._fan_cycle_started_at >= FAN_CYCLE_MAX_RUNTIME
            ):
                # room temps may never converge, so cap each session and
                # block an immediate restart
                self._fan_cycle_blocked_until = now + FAN_CYCLE_COOLDOWN
                return []
            # sticky: keep running until the needy room recovers past the
            # margin or the spread collapses
            needy_margin = FAN_CYCLE_RECOVERY_MARGIN
            required_spread = FAN_CYCLE_SPREAD_OFF
        else:
            needy_margin = 0.0
            required_spread = FAN_CYCLE_SPREAD_ON

        cooling = self._hvac_mode == HVACMode.COOL

        # find the worst-off "needy" room among primaries (parent target)
        # and custom rooms (child target)
        needy_candidates = [(room, self._target_temp) for room in primary_rooms]
        needy_candidates += [
            (
                room,
                self._child_thermostats[room.name].target_temperature
                or self._target_temp,
            )
            for room in custom_rooms
        ]
        worst_temp: float | None = None
        for room, target in needy_candidates:
            current_temp = self._room_temp(room)
            if current_temp is None or target is None:
                continue
            if cooling:
                if current_temp >= target - needy_margin:
                    worst_temp = (
                        current_temp
                        if worst_temp is None
                        else max(worst_temp, current_temp)
                    )
            elif current_temp <= target + needy_margin:
                worst_temp = (
                    current_temp
                    if worst_temp is None
                    else min(worst_temp, current_temp)
                )

        if worst_temp is None:
            return []

        donors = []
        for room in [*primary_rooms, *secondary_rooms, *custom_rooms]:
            current_temp = self._room_temp(room)
            if current_temp is None:
                continue
            if cooling:
                if current_temp <= worst_temp - required_spread:
                    donors.append(room)
            elif current_temp >= worst_temp + required_spread:
                donors.append(room)
        return donors

    async def _async_fan_cycle(self, donor_rooms: list[Room]) -> None:
        """Run the real unit in fan_only to circulate air between rooms."""
        if not self._fan_cycle_active:
            self._fan_cycle_active = True
            self._fan_cycle_started_at = datetime.now()

        await self.hass.services.async_call(
            "climate",
            "set_fan_mode",
            {"entity_id": self._real_climate_entity_id, "fan_mode": "auto"},
            blocking=False,
        )

        await self.hass.services.async_call(
            "climate",
            "set_hvac_mode",
            {
                "entity_id": self._real_climate_entity_id,
                "hvac_mode": HVACMode.FAN_ONLY.value,
            },
            blocking=False,
        )

        # open donor covers so air actually exchanges; needy rooms are
        # already open from the normal room update pass
        for room in donor_rooms:
            self._room_states[room.name].cover_pos = 100
            cover_state = self.hass.states.get(room.cover_entity)
            cover_pos = (
                cover_state.attributes.get("current_position", 0) if cover_state else 0
            )
            if cover_pos == 100:
                continue
            await self.hass.services.async_call(
                "cover",
                "set_cover_position",
                {"entity_id": room.cover_entity, "position": 100},
                blocking=False,
            )

    def _end_fan_cycle(self, clear_block: bool = False) -> None:
        """Clear fan cycling state; the caller drives the unit and covers.

        The runtime-cap cooldown (``_fan_cycle_blocked_until``) normally
        survives ending a session so an idle unit can't immediately restart
        cycling. Pass clear_block=True when the situation the cooldown guards
        against no longer applies — an hvac mode change or a real cool/heat
        cycle, both of which move the room temps the cooldown was scoped to.
        """
        self._fan_cycle_active = False
        self._fan_cycle_started_at = None
        if clear_block:
            self._fan_cycle_blocked_until = None

    async def _async_update_child_thermostats(self):
        """Update child thermostat state."""
        for room in self._rooms:
            child_thermo = self._child_thermostats.get(room.name)
            if child_thermo is None:
                continue

            sensor_state = self.hass.states.get(room.sensor_entity)
            if sensor_state is None or sensor_state.state in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                continue
            current_temp = float(sensor_state.state)
            cover_state = self.hass.states.get(room.cover_entity)
            cover_pos = (
                cover_state.attributes.get("current_position", 0) if cover_state else 0
            )
            if cover_pos == 0 and self.hvac_action == HVACAction.OFF:
                child_action = HVACAction.OFF
            elif cover_pos == 0:
                child_action = HVACAction.IDLE
            else:
                child_action = self.hvac_action

            if self._room_states[room.name].mode == RoomMode.SECONDARY:
                parent_target_temperature = self._target_secondary_temp()
            else:
                parent_target_temperature = self._target_temp

            await child_thermo.async_set_child_state(
                parent_target_temperature=parent_target_temperature,
                current_temperature=current_temp,
                hvac_action=child_action,
            )

    async def _async_update_output(self) -> None:
        """Update the output entity state."""
        if self._output_entity_id is None:
            return

        summaries = []
        now = datetime.now()

        for room in self._rooms:
            state = self._room_states.get(room.name)
            if not state:
                summaries.append(f"{room.name}: (no state)")
                continue

            # mode
            if state.mode == RoomMode.DISABLED:
                mode = "⊘" if state.disabled_by_door else "×"
            elif state.mode == RoomMode.PRIMARY:
                mode = "①"
            elif state.mode == RoomMode.SECONDARY:
                mode = "②"
            elif state.mode == RoomMode.CUSTOM:
                mode = "✐"
            else:
                mode = "?"

            # satisfied
            satisfied = "✓" if state.is_satisfied else "✗"

            # light
            if room.light_entity is not None:
                light = "◉" if state.light_on else "◎"
            else:
                light = ""

            # door
            if room.door_entity is not None:
                door = "▮" if state.door_closed else "▯"
            else:
                door = ""

            # when was target reached
            last_str = (
                f"{int((now - state.reached_target_at).total_seconds()) // 60}mins"
                if state.reached_target_at
                else ""
            )

            summaries.append(
                f"{room.name}: {mode} {satisfied}{light}{door} {last_str}".strip()
            )

        value = "   ".join(summaries)
        if self._fan_cycle_active:
            value = f"≋ {value}"

        await self.hass.services.async_call(
            input_text.DOMAIN,
            "set_value",
            {"entity_id": self._output_entity_id, "value": value},
            blocking=False,
        )

    def _apply_comfort_feedback(
        self,
        too_hot: bool,
        too_cold: bool,
        primary_rooms: list[Room],
        secondary_rooms: list[Room],
        disabled_rooms: list[Room],
        is_active: bool = False,
    ) -> None:
        """Apply too hot / too cold feedback to room states and target temp."""
        is_cooling = self._hvac_mode in {HVACMode.COOL, HVACMode.FAN_ONLY}
        # "aligned" = the feedback matches what the HVAC mode should fix
        # too_hot + cooling, or too_cold + heating
        aligned = (too_hot and is_cooling) or (too_cold and not is_cooling)

        if aligned:
            # Promote rooms whose light is on but hasn't passed the 2-min delay
            for room in list(secondary_rooms):
                room_state = self._room_states[room.name]
                if (
                    room.light_entity
                    and room_state.raw_light_on_at is not None
                    and not room_state.light_on
                ):
                    # Light is physically on but hasn't been on long enough
                    room_state.light_on = True
                    secondary_rooms.remove(room)
                    primary_rooms.append(room)
                    self._transition_room_to_mode(room, RoomMode.PRIMARY)

            # Check if any room's temp warrants forcing the AC on
            any_room_needs_ac = False
            for room in primary_rooms:
                sensor_state = self.hass.states.get(room.sensor_entity)
                if sensor_state is None or sensor_state.state in (
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
                ):
                    continue
                current_temp = float(sensor_state.state)
                target = self._target_temp
                if (is_cooling and current_temp >= target) or (
                    not is_cooling and current_temp <= target
                ):
                    any_room_needs_ac = True

            if any_room_needs_ac:
                # Force all primary/custom rooms to unsatisfied so AC turns on
                for room in primary_rooms:
                    self._room_states[room.name].is_satisfied = False
            else:
                # Rooms are already past target in the right direction, adjust temp
                # Make the system work harder: cool lower or heat higher
                if is_cooling:
                    self._snap_target_temp(-1)
                else:
                    self._snap_target_temp(+1)
        else:
            # Opposing feedback: too_hot + heating, or too_cold + cooling
            if is_active:
                # AC is actively running — check if rooms have passed the
                # target before deciding whether to also shift the target.
                any_past_target = False
                any_valid_reading = False
                for room in primary_rooms:
                    sensor_state = self.hass.states.get(room.sensor_entity)
                    if sensor_state is None or sensor_state.state in (
                        STATE_UNAVAILABLE,
                        STATE_UNKNOWN,
                    ):
                        continue
                    current_temp = float(sensor_state.state)
                    any_valid_reading = True
                    if (not is_cooling and current_temp > self._target_temp) or (
                        is_cooling and current_temp < self._target_temp
                    ):
                        any_past_target = True
                        break

                if not any_past_target and any_valid_reading:
                    # No room has reached the target yet — ease off
                    if is_cooling:
                        self._snap_target_temp(+1)
                    else:
                        self._snap_target_temp(-1)

                # Either way, mark all rooms satisfied so HVAC stops now
                for room in primary_rooms:
                    self._room_states[room.name].is_satisfied = True
            else:
                # AC is idle — user wants to shift the target for next cycle
                if is_cooling:
                    self._snap_target_temp(+1)
                else:
                    self._snap_target_temp(-1)

    def _get_feedback_switches(
        self,
    ) -> tuple[FeedbackSwitch | None, FeedbackSwitch | None]:
        """Get the too hot / too cold switches from hass.data."""
        domain_data = self.hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(self._attr_unique_id, {})
        return entry_data.get("too_hot_switch"), entry_data.get("too_cold_switch")

    async def _capture_and_reset_feedback(self) -> tuple[bool, bool]:
        """Read and reset the too hot / too cold switches.

        Returns (too_hot, too_cold). If both are on, returns (False, False).
        """
        too_hot_switch, too_cold_switch = self._get_feedback_switches()
        too_hot = too_hot_switch is not None and too_hot_switch.is_on
        too_cold = too_cold_switch is not None and too_cold_switch.is_on

        # Turn off whichever are on
        if too_hot and too_hot_switch is not None:
            await too_hot_switch.async_turn_off()
        if too_cold and too_cold_switch is not None:
            await too_cold_switch.async_turn_off()

        # Both on cancels out
        if too_hot and too_cold:
            return False, False

        return too_hot, too_cold

    def _snap_target_temp(self, direction: int) -> None:
        """Adjust target temp by snapping to next half-degree boundary.

        direction: +1 to increase, -1 to decrease.
        If currently on a whole or half degree, move by 0.3 in the given direction.
        Otherwise, snap to the next whole or half degree in the given direction.
        """
        temp = self._target_temp
        remainder = round(temp % 0.5, 2)
        on_boundary = remainder == 0.0
        if on_boundary:
            self._target_temp = round(temp + 0.3 * direction, 1)
        else:
            if direction > 0:
                self._target_temp = math.ceil(temp * 2) / 2
            else:
                self._target_temp = math.floor(temp * 2) / 2

    @property
    def _is_device_active(self) -> bool:
        """If the toggleable device is currently active.

        Always returns a bool: the latch (`_last_device_active`) starts False
        and only ever holds a bool, so dropouts return the last trusted
        reading rather than None.
        """

        climate_state = self.hass.states.get(self._real_climate_entity_id)
        if climate_state is None or climate_state.state in (
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
        ):
            # Momentary disconnect: the physical unit keeps idling/heating/
            # cooling as it was. During a dropout the state object carries no
            # attributes, so recomputing here would misread the AC. Hold the
            # last reading we trusted (indefinitely) instead of guessing.
            return self._last_device_active

        real_hvac_action = climate_state.attributes.get("hvac_action")
        self._last_device_active = real_hvac_action not in (
            HVACAction.IDLE,
            HVACAction.OFF,
        )
        return self._last_device_active
