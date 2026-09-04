"""
Tests for oracle WorldState — pose resolution and PDDL init generation.
"""

from simulation.oracle.world_state import WorldState, ObjectState, Pose


def _make_world() -> WorldState:
    return WorldState(
        objects=[
            ObjectState(
                name="red_cup",
                pose=Pose(position=[0.5, 0.0, 0.2], orientation=[0, 0, 0, 1]),
                location="table_a",
            ),
            ObjectState(
                name="blue_box",
                pose=Pose(position=[0.6, 0.1, 0.2], orientation=[0, 0, 0, 1]),
                location="table_b",
            ),
        ],
        gripper_empty=True,
    )


def test_get_pose_known_object():
    ws = _make_world()
    pose = ws.get_pose("red_cup")
    assert pose is not None
    assert pose.position == [0.5, 0.0, 0.2]


def test_get_pose_unknown_object():
    ws = _make_world()
    assert ws.get_pose("nonexistent") is None


def test_pddl_init_facts():
    ws = _make_world()
    facts = ws.to_pddl_init()
    assert "(on red_cup table_a)" in facts
    assert "(on blue_box table_b)" in facts
    assert "(gripper-empty)" in facts
