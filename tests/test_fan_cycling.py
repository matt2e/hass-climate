"""Tests for fan cycling (circulating air with the real unit's fan)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.matt_thermostat.climate import (
    FAN_CYCLE_COOLDOWN,
    FAN_CYCLE_MAX_RUNTIME,
)

from .conftest import make_hass, make_parent, setup_feedback_switches

COVERS_CLOSED = {
    "cover.living_room_vent": 0,
    "cover.bedroom_vent": 0,
    "cover.office_vent": 0,
}
LIGHTS_ON = {
    "light.living_room": STATE_ON,
    "light.office": STATE_ON,
}


def make_cool_parent(hass, **kwargs):
    """Make a cool-mode parent with primary room lights pre-latched on."""
    parent = make_parent(hass, target_temp=22.0, hvac_mode=HVACMode.COOL, **kwargs)
    setup_feedback_switches(hass)
    for room in parent._rooms:
        if room.light_entity:
            parent._room_states[room.name].light_on = True
    return parent


def calls(hass, domain, service):
    return [
        c
        for c in hass.services.async_call.call_args_list
        if c[0][:2] == (domain, service)
    ]


class TestFanCycleStarts:
    @pytest.mark.asyncio
    async def test_starts_when_primary_at_target_and_donor_available(self):
        # Living room at target (satisfied but drifting), bedroom 4° cooler
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_cool_parent(hass)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is True
        assert parent._fan_cycle_started_at is not None
        assert not calls(hass, "climate", "turn_off")

        set_mode_calls = calls(hass, "climate", "set_hvac_mode")
        assert len(set_mode_calls) == 1
        assert set_mode_calls[0][0][2]["hvac_mode"] == HVACMode.FAN_ONLY.value

        set_fan_calls = calls(hass, "climate", "set_fan_mode")
        assert len(set_fan_calls) == 1
        assert set_fan_calls[0][0][2]["fan_mode"] == "auto"

    @pytest.mark.asyncio
    async def test_opens_donor_cover(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_cool_parent(hass)

        await parent._async_control_real_climate()

        donor_calls = [
            c
            for c in calls(hass, "cover", "set_cover_position")
            if c[0][2]["entity_id"] == "cover.bedroom_vent"
        ]
        assert donor_calls
        assert donor_calls[-1][0][2]["position"] == 100
        assert parent._room_states["Bedroom"].cover_pos == 100

    @pytest.mark.asyncio
    async def test_skips_donor_cover_already_open(self):
        # Donor vent already at 100 shouldn't get re-commanded every pass
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions={**COVERS_CLOSED, "cover.bedroom_vent": 100},
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_cool_parent(hass)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is True
        donor_calls = [
            c
            for c in calls(hass, "cover", "set_cover_position")
            if c[0][2]["entity_id"] == "cover.bedroom_vent"
        ]
        assert not donor_calls
        # state still reflects the desired open position
        assert parent._room_states["Bedroom"].cover_pos == 100

    @pytest.mark.asyncio
    async def test_starts_in_heat_mode_mirror(self):
        # Living room at target, bedroom 4° warmer
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 20.0,
                "sensor.bedroom_temp": 24.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_parent(hass, target_temp=20.0, hvac_mode=HVACMode.HEAT)
        setup_feedback_switches(hass)
        for room in parent._rooms:
            if room.light_entity:
                parent._room_states[room.name].light_on = True

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is True
        set_mode_calls = calls(hass, "climate", "set_hvac_mode")
        assert set_mode_calls[0][0][2]["hvac_mode"] == HVACMode.FAN_ONLY.value


class TestFanCycleDoesNotStart:
    @staticmethod
    def make_hass_with(living=22.0, bedroom=18.0, office=21.0, **kwargs):
        return make_hass(
            room_temps={
                "sensor.living_room_temp": living,
                "sensor.bedroom_temp": bedroom,
                "sensor.office_temp": office,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_spread_too_small(self):
        hass = self.make_hass_with(bedroom=19.0)  # only 3° spread
        parent = make_cool_parent(hass)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1
        assert not calls(hass, "climate", "set_hvac_mode")

    @pytest.mark.asyncio
    async def test_no_needy_primary(self):
        # All primaries below target, even with a huge spread
        hass = self.make_hass_with(living=21.5, bedroom=17.0)
        parent = make_cool_parent(hass)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1

    @pytest.mark.asyncio
    async def test_not_when_away(self):
        hass = self.make_hass_with(presence=STATE_OFF)
        parent = make_cool_parent(hass)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1

    @pytest.mark.asyncio
    async def test_not_when_parent_off(self):
        hass = self.make_hass_with()
        parent = make_cool_parent(hass)
        parent._hvac_mode = HVACMode.OFF

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1

    @pytest.mark.asyncio
    async def test_not_during_cooldown(self):
        hass = self.make_hass_with()
        parent = make_cool_parent(hass)
        parent._fan_cycle_blocked_until = datetime.now() + timedelta(minutes=10)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1


class TestFanCycleNeverDisplacesCooling:
    @pytest.mark.asyncio
    async def test_unsatisfied_primary_ends_fan_cycle_and_resets_rooms(self):
        # Living room drifted past tolerance while fan cycling: the cool
        # path must win, and the off→on room reset must still fire even
        # though the fan makes _is_device_active read True.
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 23.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.FAN,
        )
        parent = make_cool_parent(hass)
        parent._fan_cycle_active = True
        parent._fan_cycle_started_at = datetime.now()

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert parent._fan_cycle_started_at is None
        set_mode_calls = calls(hass, "climate", "set_hvac_mode")
        assert len(set_mode_calls) == 1
        assert set_mode_calls[0][0][2]["hvac_mode"] == HVACMode.COOL.value
        # primary rooms reset to not satisfied despite the device reading
        # active (Office was within range and would otherwise stay satisfied)
        assert parent._room_states["Living Room"].is_satisfied is False
        assert parent._room_states["Office"].is_satisfied is False


class TestFanCycleStickiness:
    @staticmethod
    def make_active_parent(living, bedroom, office=21.0):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": living,
                "sensor.bedroom_temp": bedroom,
                "sensor.office_temp": office,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.FAN,
        )
        parent = make_cool_parent(hass)
        parent._fan_cycle_active = True
        parent._fan_cycle_started_at = datetime.now()
        return hass, parent

    @pytest.mark.asyncio
    async def test_keeps_running_below_start_spread(self):
        # Spread shrank to 3° (below the 4° start bar, above the 2° stop bar)
        hass, parent = self.make_active_parent(living=22.0, bedroom=19.0)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is True
        assert not calls(hass, "climate", "turn_off")
        set_mode_calls = calls(hass, "climate", "set_hvac_mode")
        assert set_mode_calls[0][0][2]["hvac_mode"] == HVACMode.FAN_ONLY.value

    @pytest.mark.asyncio
    async def test_stops_when_spread_collapses(self):
        hass, parent = self.make_active_parent(living=22.0, bedroom=20.5)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1

    @pytest.mark.asyncio
    async def test_stops_when_primary_recovers_past_margin(self):
        # Worst primary at 21.6 < target - 0.3, so mission accomplished
        hass, parent = self.make_active_parent(living=21.6, bedroom=17.0, office=21.0)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1

    @pytest.mark.asyncio
    async def test_keeps_running_within_recovery_margin(self):
        # Primary slightly below target but within the 0.3 margin
        hass, parent = self.make_active_parent(living=21.8, bedroom=18.0)

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is True


class TestFanCycleRuntimeCap:
    @pytest.mark.asyncio
    async def test_max_runtime_stops_and_blocks_restart(self):
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.FAN,
        )
        parent = make_cool_parent(hass)
        parent._fan_cycle_active = True
        parent._fan_cycle_started_at = (
            datetime.now() - FAN_CYCLE_MAX_RUNTIME - timedelta(minutes=1)
        )

        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1
        assert parent._fan_cycle_blocked_until is not None
        assert parent._fan_cycle_blocked_until <= datetime.now() + FAN_CYCLE_COOLDOWN

        # A second pass during the cooldown must not restart it
        hass.services.async_call.reset_mock()
        await parent._async_control_real_climate()

        assert parent._fan_cycle_active is False
        assert len(calls(hass, "climate", "turn_off")) == 1

    @pytest.mark.asyncio
    async def test_hvac_mode_change_clears_block(self):
        # The cooldown is scoped to the room temps at the time it triggered;
        # an hvac mode change invalidates that, so the block must not carry over.
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 22.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_cool_parent(hass)
        parent._fan_cycle_blocked_until = datetime.now() + FAN_CYCLE_COOLDOWN

        await parent.async_set_hvac_mode(HVACMode.HEAT)

        assert parent._fan_cycle_blocked_until is None

    @pytest.mark.asyncio
    async def test_real_cool_cycle_clears_block(self):
        # A real compressor cycle moves the room temps the cooldown was
        # guarding against, so the block must not survive it.
        hass = make_hass(
            room_temps={
                "sensor.living_room_temp": 23.0,
                "sensor.bedroom_temp": 18.0,
                "sensor.office_temp": 21.0,
            },
            cover_positions=COVERS_CLOSED,
            light_states=LIGHTS_ON,
            real_climate_action=HVACAction.IDLE,
        )
        parent = make_cool_parent(hass)
        parent._fan_cycle_blocked_until = datetime.now() + FAN_CYCLE_COOLDOWN

        await parent._async_control_real_climate()

        # a real cool cycle ran (needy primary past tolerance)...
        set_mode_calls = calls(hass, "climate", "set_hvac_mode")
        assert set_mode_calls[-1][0][2]["hvac_mode"] == HVACMode.COOL.value
        # ...and the stale cooldown was dropped
        assert parent._fan_cycle_blocked_until is None


class TestFanCycleHvacAction:
    def test_reports_fan_while_cycling(self):
        hass = make_hass(real_climate_action=HVACAction.FAN)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        parent._fan_cycle_active = True
        assert parent.hvac_action == HVACAction.FAN

    def test_reports_cooling_when_not_cycling(self):
        hass = make_hass(real_climate_action=HVACAction.COOLING)
        parent = make_parent(hass, hvac_mode=HVACMode.COOL)
        assert parent.hvac_action == HVACAction.COOLING
