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

    # add child thermostats and wait, which gives them each an entity id
    await async_add_entities(list(child_thermostats.values()))

    async_add_entities(
        [
            ParentThermostat(
                hass,
                name=name,
                real_climate_entity_id=real_climate_entity_id,
                presence_entity_id=presence_entity_id,
                bedtime_entity_id=bedtime_entity_id,
                manual_entity_id=manual_entity_id,
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
        ]
    )


class RoomMode(str, Enum):
    """Room modes."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DISABLED = "disabled"


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

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listeners
        # sensors = [room.sensor_entity for room in self._rooms]
        # self.async_on_remove(
        #     async_track_state_change_event(
        #         self.hass, sensors, self._async_sensor_changed
        #     )
        # )

        # lights = [
        #     room.light_entity for room in self._rooms if room.light_entity is not None
        # ]
        # self.async_on_remove(
        #     async_track_state_change_event(
        #         self.hass, lights, self._async_light_changed
        #     )
        # )

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
            # sensor_state = self.hass.states.get(self.sensor_entity_id)
            # if sensor_state and sensor_state.state not in (
            #     STATE_UNAVAILABLE,
            #     STATE_UNKNOWN,
            # ):
            #     self._async_update_temp(sensor_state)
            #     self.async_write_ha_state()
            # switch_state = self.hass.states.get(self.heater_entity_id)
            # if switch_state and switch_state.state not in (
            #     STATE_UNAVAILABLE,
            #     STATE_UNKNOWN,
            # ):
            #     self.hass.async_create_task(
            #         self._check_switch_initial_state(), eager_start=True
            #     )

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

    # async def _async_sensor_changed(self, event: Event[EventStateChangedData]) -> None:
    #     """Handle temperature changes."""
    #     new_state = event.data["new_state"]
    #     if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
    #         return

    #     self._async_update_temp(new_state)
    #     await self._async_control_heating()
    #     self.async_write_ha_state()

    # async def _check_switch_initial_state(self) -> None:
    #     """Prevent the device from keep running if HVACMode.OFF."""
    #     if self._hvac_mode == HVACMode.OFF and self._is_device_active:
    #         _LOGGER.warning(
    #             (
    #                 "The climate mode is OFF, but the switch device is ON. Turning off"
    #                 " device %s"
    #             ),
    #             self.heater_entity_id,
    #         )
    #         await self._async_heater_turn_off()

    # @callback
    # def _async_switch_changed(self, event: Event[EventStateChangedData]) -> None:
    #     """Handle heater switch state changes."""
    #     new_state = event.data["new_state"]
    #     old_state = event.data["old_state"]
    #     if new_state is None:
    #         return
    #     if old_state is None:
    #         self.hass.async_create_task(
    #             self._check_switch_initial_state(), eager_start=True
    #         )
    #     self.async_write_ha_state()

    # @callback
    # def _async_update_temp(self, state: State) -> None:
    #     """Update thermostat with latest state from sensor."""
    #     try:
    #         cur_temp = float(state.state)
    #         if not math.isfinite(cur_temp):
    #             raise ValueError(f"Sensor has illegal state {state.state}")
    #         self._cur_temp = cur_temp
    #     except ValueError as ex:
    #         _LOGGER.error("Unable to update from sensor: %s", ex)

    async def _async_child_thermostat_changed(self) -> None:
        """Called each time a child thermostat changes."""
        await self._async_control_real_climate()
        self.async_write_ha_state()

    async def _async_poll_for_changes(self, time: datetime | None = None) -> None:
        """Called each time we check for changes."""
        await self._async_control_real_climate()
        self.async_write_ha_state()

    async def _async_control_real_climate(self, force: bool = False) -> None:
        """Check if we need to turn heating on or off."""

        async def log(message: str):
            await self.hass.services.async_call(
                "logbook",
                "log",
                {
                    "name": "Custom AirCon",
                    "message": message,
                },
                blocking=False,
            )

        async with self._temp_lock:
            if not self._active and None not in (self._target_temp,):
                self._active = True

            if not self._active:
                return

            presence = self.hass.states.get(self._presence_entity_id)
            manual = self.hass.states.get(self._manual_entity_id)
            bedtime = self.hass.states.get(self._bedtime_entity_id) == STATE_ON

            if manual == STATE_ON:
                return

            if presence == STATE_OFF or self._hvac_mode == HVACMode.OFF:
                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"entity_id": self._real_climate_entity_id, "hvac_mode": "off"},
                    blocking=False,
                )
                return

            # --- Control loop ---
            fan_speed = "off"
            custom_rooms = []
            primary_rooms = []
            secondary_rooms = []
            disabled_rooms = []
            lowest_primary_current_temp: float | None = None
            highest_target_temperature = self._target_temp

            # --- Put each room into the correct sub list ---
            for room in self._rooms:
                if room.name in self._child_thermostats:
                    child_thermo = self._child_thermostats[room.name]
                    if child_thermo.hvac_mode in [HVACMode.HEAT, HVACMode.COOL]:
                        custom_rooms.append(room)
                        continue

                if bedtime:
                    mode = room.bedtime_mode
                else:
                    mode = room.standard_mode

                real_mode = RoomMode.DISABLED
                if mode == RoomMode.SECONDARY:
                    real_mode = RoomMode.SECONDARY
                elif mode == RoomMode.PRIMARY:
                    if room.light_entity and not bedtime:
                        if self.hass.states.get(room.light_entity) in [
                            STATE_OFF,
                            STATE_UNAVAILABLE,
                            STATE_UNKNOWN,
                        ]:
                            real_mode = RoomMode.SECONDARY
                        else:
                            real_mode = RoomMode.PRIMARY
                    else:
                        real_mode = RoomMode.PRIMARY

                if real_mode == RoomMode.PRIMARY:
                    primary_rooms.append(room)
                elif real_mode == RoomMode.SECONDARY:
                    secondary_rooms.append(room)
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
                child_thermo = room.child_thermostat

                target_temp = self._target_temp or self.target_temperature
                highest_target_temperature = max(
                    highest_target_temperature, target_temp
                )

                poss_current_temp = self.hass.states.get(room.sensor_entity)
                if poss_current_temp is None:
                    continue
                current_temp = float(poss_current_temp)

                is_active = await self.async_update_room(
                    room=room, current_temp=current_temp, target_temp=target_temp
                )
                if is_active:
                    fan_speed = "auto"

            for room in primary_rooms:
                poss_current_temp = self.hass.states.get(room.sensor_entity)
                if poss_current_temp is None:
                    continue
                current_temp = float(poss_current_temp)
                if (
                    lowest_primary_current_temp is None
                    or current_temp < lowest_primary_current_temp
                ):
                    lowest_primary_current_temp = current_temp

                is_active = await self.async_update_room(
                    room=room, current_temp=current_temp, target_temp=self._target_temp
                )
                if is_active:
                    fan_speed = "auto"

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

            if fan_speed == "off":
                await self.hass.services.async_call(
                    "climate",
                    "turn_off",
                    {"entity_id": self._real_climate_entity_id},
                    blocking=False,
                )
            else:
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

            await self.async_update_child_thermostats()

    async def async_update_secondary_rooms(self, secondary_rooms: list[Room]) -> None:
        """Updates secondary rooms and is called when other rooms are needing the air con on."""
        target_temp_secondary = max(self._target_temp - 2, 16)

        for room in secondary_rooms:
            poss_current_temp = self.hass.states.get(room.sensor_entity)
            if poss_current_temp is None:
                continue
            current_temp = float(poss_current_temp)

            await self.async_update_room(
                room=room, current_temp=current_temp, target_temp=target_temp_secondary
            )

    async def async_update_room(
        self, room: Room, current_temp: float, target_temp: float
    ) -> bool:
        """Update the target temperature for a room."""
        diff = self._target_temp - current_temp
        cover_state = self.hass.states.get(room.cover_entity)
        cover_pos = (
            cover_state.attributes.get("current_position", 0) if cover_state else 0
        )

        if diff > self._cold_tolerance or (
            -1 * self._hot_tolerance < diff <= self._cold_tolerance and cover_pos > 0
        ):
            if cover_pos < 100:
                await self.hass.services.async_call(
                    "cover",
                    "set_cover_position",
                    {"entity_id": room.cover_entity, "position": 100},
                    blocking=False,
                )
                # await log(f"{room.name}: 100% (goldilocks zone -> warming up)")
            return True
        if diff <= -1 * self._hot_tolerance:
            # await log(f"{room.name}: Reached target temperature + buffer")
            if cover_pos > 0:
                await self.hass.services.async_call(
                    "cover",
                    "set_cover_position",
                    {"entity_id": room.cover_entity, "position": 0},
                    blocking=False,
                )
            # await log(f"{room.name}: Goldilocks zone -> cooling down")
            return False

        return False

    async def async_update_child_thermostats(self):
        """Update child thermostat state."""
        for room in self._rooms:
            child_thermo = room.child_thermostat
            if child_thermo is None:
                continue

            poss_current_temp = self.hass.states.get(room.sensor_entity)
            if poss_current_temp is None:
                continue
            current_temp = float(poss_current_temp)
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

            await child_thermo.async_set_child_state(
                parent_target_temperature=self._target_temperature,
                current_temperature=current_temp,
                hvac_action=child_action,
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
        self._hvac_mode = HVACMode.OFF
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
        self._attr_supported_features = (
            ClimateEntityFeature.TARGET_TEMPERATURE
            | ClimateEntityFeature.TURN_OFF
            | ClimateEntityFeature.TURN_ON
        )
        self._attr_preset_modes = [PRESET_NONE]

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

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

        if self.hass.state is CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

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
