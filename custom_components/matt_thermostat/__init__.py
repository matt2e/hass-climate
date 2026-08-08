"""The matt_thermostat component."""

import json
import logging
from types import MappingProxyType
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.util import slugify

from .climate import Room
from .const import CONF_ROOM_NAME, CONF_ROOMS, PLATFORMS, SUBENTRY_TYPE_ROOM

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(config_entry_update_listener))
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug(
        "Migrating from version %s.%s", config_entry.version, config_entry.minor_version
    )

    # minor_version 3 moved rooms out of the single CONF_ROOMS JSON blob and
    # into one "room" config subentry each. Only entries older than that still
    # carry the blob, so migrate them; a matching-or-newer minor_version (e.g.
    # after a downgrade) is left untouched so we never clobber a later schema.
    if config_entry.minor_version < 3:
        _migrate_rooms_to_subentries(hass, config_entry)

    _LOGGER.debug(
        "Migration to version %s.%s successful",
        config_entry.version,
        config_entry.minor_version,
    )

    return True


def _migrate_rooms_to_subentries(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Convert the legacy CONF_ROOMS JSON blob into per-room subentries.

    Each parsed room becomes a "room" ConfigSubentry whose data is the legacy
    room dict (the per-room CONF_* keys deliberately match the legacy keys, so
    Room.from_subentry reads it back unchanged). The blob is then dropped from
    options and the entry is bumped to minor_version 3.

    This is deliberately forgiving: a malformed blob or an individual bad room
    is logged and skipped rather than raised, so a broken legacy value can
    never leave the entry stuck in a migration-error state.
    """
    used_unique_ids: set[str] = set()

    for index, room in enumerate(_parse_legacy_rooms(entry.options.get(CONF_ROOMS))):
        try:
            # dict() rejects non-mapping list items; from_dict validates the
            # room the same way the runtime does and raises on bad data.
            room_data = dict(room)
            Room.from_dict(room_data)
            name = str(room_data[CONF_ROOM_NAME])
            unique_id = _unique_room_id(slugify(name) or "room", used_unique_ids)
            hass.config_entries.async_add_subentry(
                entry,
                ConfigSubentry(
                    data=MappingProxyType(room_data),
                    subentry_type=SUBENTRY_TYPE_ROOM,
                    title=name,
                    unique_id=unique_id,
                ),
            )
        except Exception as err:  # noqa: BLE001 - migration must skip, not brick
            _LOGGER.warning(
                "Skipping room #%s while migrating to per-room subentries: %s",
                index,
                err,
            )

    # Rooms now live in subentries; drop the legacy blob (the UI options schema
    # no longer allows it) and record that we reached minor_version 3.
    new_options = {k: v for k, v in entry.options.items() if k != CONF_ROOMS}
    hass.config_entries.async_update_entry(entry, options=new_options, minor_version=3)


def _parse_legacy_rooms(raw: Any) -> list[Any]:
    """Parse the legacy CONF_ROOMS JSON string into a list of room dicts.

    Returns an empty list (never raises) for a missing, empty or malformed
    value so migration can still bump the version with no subentries.
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        _LOGGER.warning(
            "Legacy '%s' option is not valid JSON; migrating with no rooms", CONF_ROOMS
        )
        return []
    if not isinstance(data, list):
        _LOGGER.warning(
            "Legacy '%s' option is not a list; migrating with no rooms", CONF_ROOMS
        )
        return []
    return data


def _unique_room_id(base: str, used: set[str]) -> str:
    """Return a subentry unique_id derived from base, unique within used.

    Duplicate room names slugify to the same base, so append a numeric suffix
    (_2, _3, ...) until the id is free, then reserve it.
    """
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


async def config_entry_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update listener, called when the config entry options are changed."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
