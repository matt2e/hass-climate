"""Config flow for Matt hygrostat."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import voluptuous as vol
from homeassistant.components import (
    binary_sensor,
    climate,
    cover,
    input_boolean,
    input_text,
    light,
    sensor,
)
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_NAME, DEGREE
from homeassistant.core import callback
from homeassistant.helpers import selector
from homeassistant.helpers.schema_config_entry_flow import (
    SchemaConfigFlowHandler,
    SchemaFlowFormStep,
)

from .climate import RoomMode
from .const import (
    CONF_ALLOWS_OVERRIDE,
    CONF_BEDTIME,
    CONF_BEDTIME_MODE,
    CONF_COLD_TOLERANCE,
    CONF_COOLING_TEMP_MODIFIER,
    CONF_COVER,
    CONF_DOOR,
    CONF_HEATING_TEMP_MODIFIER,
    CONF_HOT_TOLERANCE,
    CONF_IS_OVERFLOW,
    CONF_LIGHT,
    CONF_MANUAL,
    CONF_MAX_TEMP,
    CONF_MIN_TEMP,
    CONF_MODE,
    CONF_OUTPUT_TEXT,
    CONF_PRESENCE,
    CONF_REAL_CLIMATE,
    CONF_ROOM_NAME,
    CONF_SENSOR,
    CONF_VENTS,
    DEFAULT_TEMP_MODIFIER,
    DEFAULT_TOLERANCE,
    DOMAIN,
    SUBENTRY_TYPE_ROOM,
)

OPTIONS_SCHEMA = {
    vol.Optional(CONF_REAL_CLIMATE): selector.EntitySelector(
        selector.EntitySelectorConfig(domain=[climate.DOMAIN])
    ),
    vol.Required(
        CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE
    ): selector.NumberSelector(
        selector.NumberSelectorConfig(
            mode=selector.NumberSelectorMode.BOX, unit_of_measurement=DEGREE, step=0.1
        )
    ),
    vol.Required(
        CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE
    ): selector.NumberSelector(
        selector.NumberSelectorConfig(
            mode=selector.NumberSelectorMode.BOX, unit_of_measurement=DEGREE, step=0.1
        )
    ),
    vol.Optional(CONF_MIN_TEMP): selector.NumberSelector(
        selector.NumberSelectorConfig(
            mode=selector.NumberSelectorMode.BOX, unit_of_measurement=DEGREE, step=0.1
        )
    ),
    vol.Optional(CONF_MAX_TEMP): selector.NumberSelector(
        selector.NumberSelectorConfig(
            mode=selector.NumberSelectorMode.BOX, unit_of_measurement=DEGREE, step=0.1
        )
    ),
    vol.Required(CONF_PRESENCE): selector.EntitySelector(
        selector.EntitySelectorConfig(domain=[input_boolean.DOMAIN])
    ),
    vol.Required(CONF_BEDTIME): selector.EntitySelector(
        selector.EntitySelectorConfig(domain=[input_boolean.DOMAIN])
    ),
    vol.Required(CONF_MANUAL): selector.EntitySelector(
        selector.EntitySelectorConfig(domain=[input_boolean.DOMAIN])
    ),
    vol.Optional(CONF_OUTPUT_TEXT): selector.EntitySelector(
        selector.EntitySelectorConfig(domain=[input_text.DOMAIN])
    ),
    vol.Optional(
        CONF_COOLING_TEMP_MODIFIER, default=DEFAULT_TEMP_MODIFIER
    ): selector.NumberSelector(
        selector.NumberSelectorConfig(
            mode=selector.NumberSelectorMode.BOX, unit_of_measurement=DEGREE, step=0.1
        )
    ),
    vol.Optional(
        CONF_HEATING_TEMP_MODIFIER, default=DEFAULT_TEMP_MODIFIER
    ): selector.NumberSelector(
        selector.NumberSelectorConfig(
            mode=selector.NumberSelectorMode.BOX, unit_of_measurement=DEGREE, step=0.1
        )
    ),
}

CONFIG_SCHEMA = {
    vol.Required(CONF_NAME): selector.TextSelector(),
    **OPTIONS_SCHEMA,
}


CONFIG_FLOW = {
    "user": SchemaFlowFormStep(vol.Schema(CONFIG_SCHEMA)),
}

OPTIONS_FLOW = {
    "init": SchemaFlowFormStep(vol.Schema(OPTIONS_SCHEMA)),
}

# The three room modes a user may select. The RoomMode.CUSTOM value is an
# internal-only mode and is deliberately not offered here.
ROOM_MODE_OPTIONS = [
    selector.SelectOptionDict(value=mode.value, label=mode.value.title())
    for mode in (RoomMode.PRIMARY, RoomMode.SECONDARY, RoomMode.DISABLED)
]

# Schema for a single room, entered one room per config subentry. Defaults
# mirror those applied by the Room dataclass (Room.from_dict).
ROOM_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ROOM_NAME): selector.TextSelector(),
        vol.Required(CONF_SENSOR): selector.EntitySelector(
            selector.EntitySelectorConfig(domain=[sensor.DOMAIN])
        ),
        vol.Required(CONF_COVER): selector.EntitySelector(
            selector.EntitySelectorConfig(domain=[cover.DOMAIN])
        ),
        vol.Optional(CONF_LIGHT): selector.EntitySelector(
            selector.EntitySelectorConfig(domain=[light.DOMAIN])
        ),
        vol.Optional(CONF_DOOR): selector.EntitySelector(
            selector.EntitySelectorConfig(domain=[binary_sensor.DOMAIN])
        ),
        vol.Required(CONF_MODE): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=ROOM_MODE_OPTIONS,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Required(CONF_BEDTIME_MODE): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=ROOM_MODE_OPTIONS,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Required(CONF_ALLOWS_OVERRIDE, default=False): selector.BooleanSelector(),
        vol.Required(CONF_IS_OVERFLOW, default=False): selector.BooleanSelector(),
        vol.Required(CONF_VENTS, default=1): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1, step=1, mode=selector.NumberSelectorMode.BOX
            )
        ),
    }
)


class RoomSubentryFlowHandler(ConfigSubentryFlow):
    """Handle adding and reconfiguring a single room subentry."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a new room."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input[CONF_ROOM_NAME], data=user_input
            )
        return self.async_show_form(step_id="user", data_schema=ROOM_SCHEMA)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing room."""
        subentry = self._get_reconfigure_subentry()
        if user_input is not None:
            return self.async_update_and_abort(
                self._get_entry(),
                subentry,
                title=user_input[CONF_ROOM_NAME],
                data=user_input,
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(ROOM_SCHEMA, subentry.data),
        )


class ConfigFlowHandler(SchemaConfigFlowHandler, domain=DOMAIN):
    """Handle a config or options flow."""

    MINOR_VERSION = 3

    config_flow = CONFIG_FLOW
    options_flow = OPTIONS_FLOW

    def async_config_entry_title(self, options: Mapping[str, Any]) -> str:
        """Return config entry title."""
        return cast(str, options["name"])

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentry types supported by this integration."""
        return {SUBENTRY_TYPE_ROOM: RoomSubentryFlowHandler}
