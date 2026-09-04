"""
Tests for PDDL domain validity and plan parsing.
Does NOT require a running robot or simulator.
"""

import os
import pytest
from planner.plan_parser import parse_plan, PrimitiveCall


def test_parse_pick_place_plan():
    raw = ["(pick red_cup table_a)", "(place red_cup table_b)"]
    result = parse_plan(raw)

    assert len(result) == 2
    assert result[0] == PrimitiveCall(name="pick", args=["red_cup", "table_a"])
    assert result[1] == PrimitiveCall(name="place", args=["red_cup", "table_b"])


def test_parse_empty_plan():
    assert parse_plan([]) == []


def test_parse_ignores_blank_lines():
    raw = ["(pick obj loc)", "", "  "]
    result = parse_plan(raw)
    assert len(result) == 1
