"""Config flow for Matt hygrostat."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import voluptuous as vol

from homeassistant.components import climate, input_boolean, input_text
from homeassistant.const import CONF_NAME, DEGREE
from homeassistant.helpers import selector
from homeassistant.helpers.schema_config_entry_flow import (
    SchemaConfigFlowHandler,
    SchemaFlowFormStep,
)

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
    vol.Optional(CONF_MIN_DUR): selector.DurationSelector(
        selector.DurationSelectorConfig(allow_negative=False)
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
    vol.Required(CONF_ROOMS): selector.TextSelector(
        selector.TextSelectorConfig(multiline=True)
    ),
    vol.Optional(CONF_OUTPUT_TEXT): selector.EntitySelector(
        selector.EntitySelectorConfig(domain=[input_text.DOMAIN])
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


class ConfigFlowHandler(SchemaConfigFlowHandler, domain=DOMAIN):
    """Handle a config or options flow."""

    MINOR_VERSION = 2

    config_flow = CONFIG_FLOW
    options_flow = OPTIONS_FLOW

    def async_config_entry_title(self, options: Mapping[str, Any]) -> str:
        """Return config entry title."""
        return cast(str, options["name"])
