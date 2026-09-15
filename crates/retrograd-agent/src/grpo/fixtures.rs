//! Trajectory fixtures shared by the selection and metrics tests.

use crate::trajectory::{Trajectory, TrajectoryGroup};

fn trajectory(truncated: bool) -> Trajectory {
    Trajectory {
        scenario_id: "s".into(),
        messages: vec![],
        tokens: vec![1, 2],
        old_logprobs: vec![-1.0],
        train_mask: vec![false, true],
        steps: vec![],
        reward: None,
        truncated,
        metadata: Default::default(),
    }
}

pub(super) fn group(truncations: &[bool]) -> TrajectoryGroup {
    TrajectoryGroup {
        group_id: 1,
        scenario_id: "s".into(),
        trajectories: truncations.iter().copied().map(trajectory).collect(),
    }
}
