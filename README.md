# hass-climate

Custom climate management for Home Assistant.

`matt_thermostat` wraps a "real" climate/AC entity with a smarter parent
thermostat that drives the whole house from a set of per-room temperature
sensors and vents. Each room reports its own temperature and has a motorised
vent cover; the parent decides a target temperature, opens and closes vents,
and drives the underlying climate entity accordingly. Rooms can join in as
*primary* (they set the target), *secondary* (they follow along), or be
*disabled*, with a separate mode used at bedtime.

## Requirements

- **Home Assistant 2025.4.0 or newer.** Rooms are configured with [config
  subentries](https://developers.home-assistant.io/blog/2025/02/16/config-subentries/),
  and the reconfigure flow relies on `ConfigSubentryFlow` API that stabilised in
  2025.4. (Config subentries first landed in 2025.3, but this integration uses
  the reconfigure helpers added in the following release.)

## Installation

Install through [HACS](https://hacs.xyz/) as a custom repository, then restart
Home Assistant and add the **Matt Thermostat** integration from
*Settings → Devices & services → Add integration*. `hacs.json` pins the minimum
Home Assistant version, so HACS will not offer the integration on older cores.

## Configuring the thermostat

When you add the integration you configure the global, whole-home settings:

- **Real climate entity** – the underlying climate/AC unit that actually heats
  and cools.
- **Presence, bedtime and manual** – `input_boolean` helpers the thermostat
  watches to change behaviour (e.g. switching each room to its bedtime mode
  while *bedtime* is on).
- **Cold / hot tolerance** – how far the temperature must drift past the target
  before the thermostat reacts.
- **Cooling / heating temperature modifiers**, **min/max target temperature**,
  **minimum cycle duration** and an optional **output text** helper for
  publishing state.

These live on the main integration entry and can be changed later from the
integration's *Configure* option.

## Adding and managing rooms

Rooms are **no longer entered as a JSON array**. Each room is its own config
subentry, added and edited from tiles on the integration page.

### Add a room

On the integration page choose **Add room** and fill in the form:

| Field | Required | Description |
| --- | --- | --- |
| **Room name** | yes | A name for the room. Used to label the room's child thermostat and entities. |
| **Temperature sensor** | yes | The `sensor` that reflects the room's current temperature. |
| **Vent cover** | yes | The `cover` that opens and closes to control airflow into the room. |
| **Light** | no | An optional `light` used to detect whether the room is occupied. Leave empty if there is no light to track. |
| **Door sensor** | no | An optional door/contact `binary_sensor` (where *off* means closed). When set and the door reads closed, the room is disabled while it would otherwise be *secondary* — a closed internal door thermally decouples the room, so conditioning it is waste. Leave empty to never gate on a door. |
| **Mode** | yes | How the room participates: *primary* rooms drive the target temperature, *secondary* rooms follow along, and *disabled* rooms are excluded. |
| **Bedtime mode** | yes | The mode the room uses while bedtime is active, overriding the standard mode. |
| **Allow override thermostat** | no (default off) | Create a dedicated child thermostat for the room so its target temperature can be overridden independently. |
| **Overflow room** | no (default off) | Treat the room as an overflow room that opens its vents to relieve excess airflow. |
| **Vents** | yes (default 1) | Number of vents in the room. Used to weight how much airflow the room contributes. |

Because these are proper Home Assistant selectors, sensors, covers and lights
are picked from dropdowns and the mode fields only offer the valid
*primary / secondary / disabled* values — no more hand-editing JSON or
discovering typos at runtime.

### Reconfigure or remove a room

Each room appears as a tile on the integration page. Use its menu to
**Reconfigure** the room (the form is pre-filled with the current values) or to
**Delete** it. Saving or deleting reloads the integration so the change takes
effect immediately.

### Child thermostats

A room with **Allow override thermostat** enabled gets its own child climate
entity (named `<Room> Thermostat`) attached to that room's subentry, so you can
set an independent target for it. Rooms without override contribute their sensor
and vents to the parent only. The parent thermostat and its feedback switches
stay on the main integration entry.

## Migrating from the old JSON configuration

Earlier versions stored every room in a single multi-line JSON field. Existing
config entries are **migrated automatically** the first time the integration
loads after upgrading: each room in the JSON array becomes its own room
subentry and the old field is removed. Migration is deliberately forgiving — a
malformed blob or an individual bad room is logged and skipped rather than
blocking startup, so you may need to re-add any rooms that could not be
converted. No manual steps are required for well-formed configurations.
