"""Constants for the Matt Thermostat helper."""

from homeassistant.const import Platform

DOMAIN = "matt_thermostat"

PLATFORMS = [Platform.CLIMATE, Platform.SWITCH]

CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_REAL_CLIMATE = "real_climate"
CONF_PRESENCE = "presence"
CONF_BEDTIME = "bedtime"
CONF_MANUAL = "manual"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_MAX_TEMP = "max_temp"
CONF_MIN_TEMP = "min_temp"
CONF_ROOMS = "rooms"
DEFAULT_TOLERANCE = 0.4
DEFAULT_TEMP_MODIFIER = 0.0
CONF_OUTPUT_TEXT = "output_text"
CONF_COOLING_TEMP_MODIFIER = "cooling_temp_modifier"
CONF_HEATING_TEMP_MODIFIER = "heating_temp_modifier"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_TARGET_TEMP = "target_temp"

# Per-room subentry fields (one room per subentry)
CONF_ROOM_NAME = "name"
CONF_SENSOR = "sensor"
CONF_COVER = "cover"
CONF_LIGHT = "light"
CONF_DOOR = "door"
CONF_MODE = "mode"
CONF_BEDTIME_MODE = "bedtime_mode"
CONF_ALLOWS_OVERRIDE = "allows_override"
CONF_IS_OVERFLOW = "is_overflow"
CONF_VENTS = "vents"

SUBENTRY_TYPE_ROOM = "room"
