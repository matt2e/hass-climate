"""Adds support for matt thermostat units."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import math
from typing import Any

import voluptuous as vol

from homeassistant.components import input_text
from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA,
    PRESET_NONE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_TENTHS,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import CoreState, Event, HomeAssistant, callback
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

from .const import (
    CONF_BEDTIME,
    CONF_COLD_TOLERANCE,
    CONF_HOT_TOLERANCE,
    CONF_MANUAL,
    CONF_MAX_TEMP,
    CONF_MIN_DUR,
    CONF_MIN_TEMP,
    CONF_OUTPUT_TEXT,
    CONF_PRESENCE,
    CONF_REAL_CLIMATE,
    CONF_ROOMS,
    DEFAULT_TOLERANCE,
    DOMAIN,
    PLATFORMS,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Home Thermostat"

CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_TARGET_TEMP = "target_temp"

PLATFORM_SCHEMA_COMMON = vol.Schema(
    {
        vol.Required(CONF_REAL_CLIMATE): cv.entity_id,
        vol.Required(CONF_PRESENCE): cv.entity_id,
        vol.Required(CONF_MANUAL): cv.entity_id,
        vol.Required(CONF_BEDTIME): cv.entity_id,
        vol.Required(CONF_ROOMS): cv.string,
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_OUTPUT_TEXT): cv.string,
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVACMode.COOL, HVACMode.HEAT, HVACMode.OFF]
        ),
        vol.Optional(CONF_UNIQUE_ID): cv.string,
    }
)


PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend(PLATFORM_SCHEMA_COMMON.schema)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Initialize config entry."""
    await _async_setup_config(
        hass,
        PLATFORM_SCHEMA_COMMON(dict(config_entry.options)),
        config_entry.entry_id,
        async_add_entities,
    )


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the parent thermostat platform."""

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    await _async_setup_config(
        hass, config, config.get(CONF_UNIQUE_ID), async_add_entities
    )


async def _async_setup_config(
    hass: HomeAssistant,
    config: Mapping[str, Any],
    unique_id: str | None,
    async_add_entities: AddEntitiesCallback | AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the parent thermostat platform."""

    name: str = config[CONF_NAME]
    real_climate_entity_id: str = config[CONF_REAL_CLIMATE]
    bedtime_entity_id: str = config[CONF_BEDTIME]
    presence_entity_id: str = config[CONF_PRESENCE]
    manual_entity_id: str = config[CONF_MANUAL]
    output_entity_id: str | None = config.get(CONF_OUTPUT_TEXT)
    min_temp: float | None = config.get(CONF_MIN_TEMP)
    max_temp: float | None = config.get(CONF_MAX_TEMP)
    target_temp: float | None = config.get(CONF_TARGET_TEMP)
    min_cycle_duration: timedelta | None = config.get(CONF_MIN_DUR)
    cold_tolerance: float = config[CONF_COLD_TOLERANCE]
    hot_tolerance: float = config[CONF_HOT_TOLERANCE]
    initial_hvac_mode: HVACMode | None = config.get(CONF_INITIAL_HVAC_MODE)
    unit = hass.config.units.temperature_unit

    precision: float | None = config.get(PRECISION_TENTHS)
    target_temperature_step = 0.1

    data = json.loads(config[CONF_ROOMS])
    if not isinstance(data, list):
        raise TypeError("Expected a list of rooms")
    rooms = [Room.from_dict(item) for item in data]

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

    # add child thermostats, which gives them each an entity id
    async_add_entities(list(child_thermostats.values()))

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
        min_cycle_duration=min_cycle_duration,
        cold_tolerance=cold_tolerance,
        hot_tolerance=hot_tolerance,
        initial_hvac_mode=initial_hvac_mode,
        precision=precision,
        target_temperature_step=target_temperature_step,
        unit=unit,
        unique_id=unique_id,
        rooms=rooms,
        child_thermostats=child_thermostats,
    )

    async_add_entities([parent])


class RoomMode(str, Enum):
    """Room modes."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DISABLED = "disabled"

    # internal modes
    CUSTOM = "custom"


class RoomState:
    """Room states."""

    mode: RoomMode = RoomMode.DISABLED
    cover_pos: int = 0
    reached_min_at: datetime | None = None
    reached_target_at: datetime | None = None
    reached_max_at: datetime | None = None

    # is true when room reaches max temp or stays above target temp for long enough, and stays on until it falls below tolerance
    is_satisfied: bool = False

    light_on: bool = (
        False  # value delayed based on raw_light_on_at and raw_light_off_at
    )
    raw_light_on_at: datetime | None = None
    raw_light_off_at: datetime | None = None


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
    """

    name: str
    sensor_entity: str
    cover_entity: str
    light_entity: str | None
    standard_mode: RoomMode
    bedtime_mode: RoomMode
    allows_override: bool
    is_overflow: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Room:
        """Read Room from json dict."""

        if "name" not in data:
            raise ValueError(
                "Room must have 'name' and 'cover' and 'mode' and 'sensor'"
            )
        name = str(data["name"])
        if "cover" not in data or "mode" not in data or "sensor" not in data:
            raise ValueError(f"Room '{name}' must have 'cover' and 'mode' and 'sensor'")
        mode = str(data["mode"])
        if mode not in (RoomMode.PRIMARY, RoomMode.SECONDARY, RoomMode.DISABLED):
            raise ValueError(
                f"Invalid mode: {mode}, expected one of {RoomMode.PRIMARY}, {RoomMode.SECONDARY}, {RoomMode.DISABLED}"
            )
        bedtime_mode = str(data["bedtime_mode"])
        if bedtime_mode not in (
            RoomMode.PRIMARY,
            RoomMode.SECONDARY,
            RoomMode.DISABLED,
        ):
            raise ValueError(
                f"Invalid bedtime mode: {bedtime_mode}, expected one of {RoomMode.PRIMARY}, {RoomMode.SECONDARY}, {RoomMode.DISABLED}"
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
        )


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
        min_cycle_duration: timedelta | None,
        cold_tolerance: float,
        hot_tolerance: float,
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
        self._min_cycle_duration = min_cycle_duration
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        self._attr_hvac_modes = [HVACMode.OFF, HVACMode.COOL, HVACMode.HEAT]
        self._active = False
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

        @callback
        def _async_startup(_: Event | None = None) -> None:
            """Init on startup."""
            # original code went here...

        if self.hass.state is CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

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
        if self._hvac_mode == HVACMode.COOL:
            return HVACAction.COOLING
        if self._hvac_mode == HVACMode.HEAT:
            return HVACAction.HEATING
        return HVACAction.IDLE

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.HEAT:
            self._hvac_mode = HVACMode.HEAT
        elif hvac_mode == HVACMode.COOL:
            self._hvac_mode = HVACMode.COOL
        elif hvac_mode == HVACMode.OFF:
            self._hvac_mode = HVACMode.OFF
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

            presence = self.hass.states.get(self._presence_entity_id).state == STATE_ON
            manual = self.hass.states.get(self._manual_entity_id).state == STATE_ON

            if manual:
                self._reset_all_room_states()
                return

            if not presence or self._hvac_mode == HVACMode.OFF:
                self._reset_all_room_states()
                await self.hass.services.async_call(
                    "climate",
                    "turn_off",
                    {"entity_id": self._real_climate_entity_id},
                    blocking=False,
                )
                return

            # --- Control loop ---
            custom_rooms = []
            primary_rooms = []
            secondary_rooms = []
            disabled_rooms = []
            lowest_primary_current_temp: float | None = None
            highest_target_temperature = self._target_temp

            # --- Put each room into the correct sub list ---
            for room in self._rooms:
                real_mode = self._calculate_room_mode(room)

                self._transition_room_to_mode(room, real_mode)

                if real_mode == RoomMode.PRIMARY:
                    primary_rooms.append(room)
                elif real_mode == RoomMode.SECONDARY:
                    secondary_rooms.append(room)
                elif real_mode == RoomMode.CUSTOM:
                    custom_rooms.append(room)
                else:
                    disabled_rooms.append(room)

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
                highest_target_temperature = max(
                    highest_target_temperature, target_temp
                )

                sensor_state = self.hass.states.get(room.sensor_entity)
                if sensor_state is None:
                    continue
                current_temp = float(sensor_state.state)

                await self.async_update_room(
                    room=room, current_temp=current_temp, target_temp=target_temp
                )

            for room in primary_rooms:
                sensor_state = self.hass.states.get(room.sensor_entity)
                if sensor_state is None:
                    continue
                current_temp = float(sensor_state.state)
                if (
                    lowest_primary_current_temp is None
                    or current_temp < lowest_primary_current_temp
                ):
                    lowest_primary_current_temp = current_temp

                await self.async_update_room(
                    room=room, current_temp=current_temp, target_temp=self._target_temp
                )

            # --- Apply AC mode, temp, and fan speed ---
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {
                    "entity_id": self._real_climate_entity_id,
                    ATTR_TEMPERATURE: math.ceil(highest_target_temperature),
                },
                blocking=False,
            )

            if lowest_primary_current_temp is not None:
                self._attr_current_temperature = lowest_primary_current_temp

            fan_speed = await self.calculate_fan_speed()
            if not fan_speed:
                await self.hass.services.async_call(
                    "climate",
                    "turn_off",
                    {"entity_id": self._real_climate_entity_id},
                    blocking=False,
                )
            else:
                device_active = self._is_device_active
                if device_active is not None and not device_active:
                    # we are turning AC on, so mark all rooms as not satisfied
                    for room_state in self._room_states.values():
                        room_state.is_satisfied = False

                await self.hass.services.async_call(
                    "climate",
                    "set_fan_mode",
                    {"entity_id": self._real_climate_entity_id, "fan_mode": fan_speed},
                    blocking=False,
                )
                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"entity_id": self._real_climate_entity_id, "hvac_mode": "heat"},
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
        if sensor_state is None:
            return
        current_temp = float(sensor_state.state)

        # we need to update whether the room is satisfied based on if it is in the target temp range
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

        room_state.is_satisfied = current_temp >= target_temp - self._cold_tolerance

    def _reset_all_room_states(self) -> None:
        """Reset all room states to their initial values."""
        for name, _ in self._room_states:
            self.room_states[name] = RoomState()

    def _calculate_room_mode(self, room: Room) -> RoomMode:
        """Calculate the current mode of a room."""
        if room.name in self._child_thermostats:
            child_thermo = self._child_thermostats[room.name]
            if child_thermo.hvac_mode in [HVACMode.HEAT, HVACMode.COOL]:
                return RoomMode.CUSTOM

        bedtime = self.hass.states.get(self._bedtime_entity_id).state == STATE_ON
        if bedtime:
            mode = room.bedtime_mode
        else:
            mode = room.standard_mode

        if mode == RoomMode.SECONDARY:
            return RoomMode.SECONDARY
        if mode == RoomMode.PRIMARY:
            room_state = self._room_states[room.name]
            if room.light_entity:
                if self.hass.states.get(room.light_entity).state in [
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
                return RoomMode.SECONDARY

            return RoomMode.PRIMARY

        return RoomMode.DISABLED

    def _target_secondary_temp(self) -> float | None:
        return max(self._target_temp - 2, 16)

    async def async_update_secondary_rooms(self, secondary_rooms: list[Room]) -> None:
        """Updates secondary rooms and is called when other rooms are needing the air con on."""
        target_temp_secondary = self._target_secondary_temp()
        for room in secondary_rooms:
            sensor_state = self.hass.states.get(room.sensor_entity)
            if sensor_state is None:
                continue
            current_temp = float(sensor_state.state)

            await self.async_update_room(
                room=room, current_temp=current_temp, target_temp=target_temp_secondary
            )

    async def async_update_room(
        self, room: Room, current_temp: float, target_temp: float
    ) -> None:
        """Update the target temperature for a room."""
        is_first_temp_reading = not self._sensor_found_for_room[room.name]
        if is_first_temp_reading:
            self._sensor_found_for_room[room.name] = True

        diff = target_temp - current_temp
        cover_state = self.hass.states.get(room.cover_entity)
        cover_pos = (
            cover_state.attributes.get("current_position", 0) if cover_state else 0
        )

        room_state = self._room_states[room.name]
        if diff > self._cold_tolerance:
            room_state.reached_min_at = None
            room_state.is_satisfied = False
        else:
            if is_first_temp_reading:
                # When first launched, count any rooms within the range as satisfied
                room_state.is_satisfied = True
            if room_state.reached_min_at is None:
                room_state.reached_min_at = datetime.now()

        if diff > 0:
            room_state.reached_target_at = None
        elif room_state.reached_target_at is None:
            room_state.reached_target_at = datetime.now()
        elif not room_state.is_satisfied and (
            datetime.now() - room_state.reached_target_at
        ) > timedelta(minutes=5):
            room_state.is_satisfied = True

        if diff > -1 * self._hot_tolerance:
            room_state.reached_max_at = None
        elif room_state.reached_max_at is None:
            room_state.reached_max_at = datetime.now()
            room_state.is_satisfied = True

        if diff > -0.5 * self._hot_tolerance:
            desired_cover_pos = 100
        elif diff <= -1 * self._hot_tolerance:
            desired_cover_pos = 0
        elif room_state.reached_target_at is not None and (
            datetime.now() - room_state.reached_target_at
        ) > timedelta(minutes=5):
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

    async def calculate_fan_speed(self) -> str | None:
        """Figure out what fan speed to use."""
        # Determine if HVAC should be on
        is_on = False
        overflow_room_state: RoomState | None = None
        below_target_count = 0
        for room in self._rooms:
            state = self._room_states[room.name]
            if state.mode == RoomMode.DISABLED:
                continue
            if room.is_overflow:
                overflow_room_state = state

            if state.reached_target_at is None:
                below_target_count += 1

            if (
                state.mode in [RoomMode.PRIMARY, RoomMode.CUSTOM]
                and not state.is_satisfied
            ):
                is_on = True

        if not is_on:
            return None

        if overflow_room_state.cover_pos == 100:
            # No fear of medium being too strong
            return "medium"

        return "auto"

    async def _async_update_child_thermostats(self):
        """Update child thermostat state."""
        for room in self._rooms:
            child_thermo = self._child_thermostats.get(room.name)
            if child_thermo is None:
                continue

            sensor_state = self.hass.states.get(room.sensor_entity)
            if sensor_state is None:
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
                mode = "×"
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

            # when was target reached
            last_str = (
                f"{(now - state.reached_target_at).seconds // 60}mins"
                if state.reached_target_at
                else ""
            )

            summaries.append(
                f"{room.name}: {mode} {satisfied}{light} {last_str}".strip()
            )

        await self.hass.services.async_call(
            input_text.DOMAIN,
            "set_value",
            {"entity_id": self._output_entity_id, "value": "   ".join(summaries)},
            blocking=False,
        )

    @property
    def _is_device_active(self) -> bool | None:
        """If the toggleable device is currently active."""

        climate_state = self.hass.states.get(self._real_climate_entity_id)
        if climate_state is None:
            return False

        real_hvac_action = climate_state.attributes.get("hvac_action")
        if real_hvac_action in (HVACAction.IDLE, HVACAction.OFF):
            return False

        return True


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
