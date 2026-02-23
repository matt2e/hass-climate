"""Constants for the Matt Thermostat helper."""

from homeassistant.const import Platform

DOMAIN = "matt_thermostat"

PLATFORMS = [Platform.CLIMATE]

CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_REAL_CLIMATE = "real_climate"
CONF_PRESENCE = "presence"
CONF_BEDTIME = "bedtime"
CONF_MANUAL = "manual"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_MAX_TEMP = "max_temp"
CONF_MIN_TEMP = "min_temp"
CONF_MIN_DUR = "min_cycle_duration"
CONF_ROOMS = "rooms"
DEFAULT_TOLERANCE = 0.4
DEFAULT_TEMP_MODIFIER = 0.0
CONF_OUTPUT_TEXT = "output_text"
CONF_COOLING_TEMP_MODIFIER = "cooling_temp_modifier"
CONF_HEATING_TEMP_MODIFIER = "heating_temp_modifier"
