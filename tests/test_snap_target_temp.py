"""Tests for _snap_target_temp."""

from __future__ import annotations

from .conftest import make_parent


class TestSnapTargetTemp:
    def test_on_whole_degree_increase(self):
        parent = make_parent(target_temp=22.0)
        parent._snap_target_temp(+1)
        assert parent._target_temp == 22.3

    def test_on_whole_degree_decrease(self):
        parent = make_parent(target_temp=22.0)
        parent._snap_target_temp(-1)
        assert parent._target_temp == 21.7

    def test_on_half_degree_increase(self):
        parent = make_parent(target_temp=22.5)
        parent._snap_target_temp(+1)
        assert parent._target_temp == 22.8

    def test_on_half_degree_decrease(self):
        parent = make_parent(target_temp=22.5)
        parent._snap_target_temp(-1)
        assert parent._target_temp == 22.2

    def test_off_boundary_snap_up(self):
        """22.3 → snap up to 22.5."""
        parent = make_parent(target_temp=22.3)
        parent._snap_target_temp(+1)
        assert parent._target_temp == 22.5

    def test_off_boundary_snap_down(self):
        """22.3 → snap down to 22.0."""
        parent = make_parent(target_temp=22.3)
        parent._snap_target_temp(-1)
        assert parent._target_temp == 22.0

    def test_off_boundary_snap_up_from_below_half(self):
        """22.2 → snap up to 22.5."""
        parent = make_parent(target_temp=22.2)
        parent._snap_target_temp(+1)
        assert parent._target_temp == 22.5

    def test_off_boundary_snap_down_from_above_half(self):
        """22.7 → snap down to 22.5."""
        parent = make_parent(target_temp=22.7)
        parent._snap_target_temp(-1)
        assert parent._target_temp == 22.5

    def test_off_boundary_snap_up_from_above_half(self):
        """22.8 → snap up to 23.0."""
        parent = make_parent(target_temp=22.8)
        parent._snap_target_temp(+1)
        assert parent._target_temp == 23.0
