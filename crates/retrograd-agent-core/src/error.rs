use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] retrograd_core::Error),
    #[error("invalid agent configuration: {0}")]
    Invalid(String),
    #[error("policy actor stopped before replying")]
    PolicyStopped,
    #[error("policy actor task failed: {0}")]
    PolicyTask(String),
    #[error("policy generation failed: {0}")]
    PolicyGeneration(String),
    #[error("tool error: {0}")]
    Tool(String),
    /// The world an episode was acting on stopped existing: a dead container, an
    /// unreachable daemon, a transport that broke mid-upload.
    ///
    /// Separate from [`Error::Tool`] because the two are told apart by *tools*,
    /// which turn a failed action into an observation the policy reads. A tool
    /// must never do that for this one: everything already collected is
    /// conditioned on a world that is gone, so the trajectory has to die instead
    /// of being scored and trained on. See [`Error::is_sandbox_failure`].
    #[error("sandbox failure: {0}")]
    Sandbox(String),
    #[error("reward backend error: {0}")]
    Reward(String),
}

/// Why a single rollout died. A group rollout survives the loss of individual
/// members, so the cause has to be counted rather than propagated; the split is
/// what tells "the environment is flaky" apart from "the policy is broken".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Tool,
    Policy,
    Other,
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// A world that is gone, as opposed to an action that failed.
    ///
    /// This is the predicate a `retrograd_tools::SessionTool` uses to decide what it may turn
    /// into an observation. A missing file is the policy's mistake and comes
    /// back as text; a lost connection to the daemon is not, and is propagated
    /// so [`Environment::step`](crate::Environment::step) can poison the lease
    /// and abandon the trajectory.
    pub fn is_sandbox_failure(&self) -> bool {
        matches!(self, Self::Sandbox(_))
    }

    pub fn sandbox(message: impl Into<String>) -> Self {
        Self::Sandbox(message.into())
    }

    pub fn failure_kind(&self) -> FailureKind {
        match self {
            // Both count as environment-side failures: the split the metric
            // exists for is "the environment is flaky" against "the policy is
            // broken", and a dead container is squarely the former.
            Self::Tool(_) | Self::Sandbox(_) => FailureKind::Tool,
            // Everything reaching the agent crate from `retrograd_core` comes
            // out of the trainer, so a core error is a policy-side failure.
            Self::Core(_)
            | Self::PolicyStopped
            | Self::PolicyTask(_)
            | Self::PolicyGeneration(_) => FailureKind::Policy,
            Self::Invalid(_) | Self::Reward(_) => FailureKind::Other,
        }
    }

    /// Copies an error so one batched failure can be attributed to every member
    /// of the decode batch it killed. `Error` is not `Clone` because
    /// `retrograd_core::Error` is not; a core error keeps its message and its
    /// [`FailureKind`] by becoming a `PolicyGeneration`.
    pub fn duplicate(&self) -> Self {
        match self {
            Self::Core(_) => Self::PolicyGeneration(self.to_string()),
            Self::Invalid(message) => Self::Invalid(message.clone()),
            Self::PolicyStopped => Self::PolicyStopped,
            Self::PolicyTask(message) => Self::PolicyTask(message.clone()),
            Self::PolicyGeneration(message) => Self::PolicyGeneration(message.clone()),
            Self::Tool(message) => Self::Tool(message.clone()),
            Self::Sandbox(message) => Self::Sandbox(message.clone()),
            Self::Reward(message) => Self::Reward(message.clone()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Back to the applicative facade, with the class preserved.
///
/// It lives here and not in `retrograd-core` because that crate knows nothing of
/// an agent - the orphan rule allows `impl From<LocalError> for ForeignError`,
/// and the direction of the dependency requires it.
///
/// What it replaces is `.map_err(|error| Error::invalid(error.to_string()))`,
/// written at twenty-two boundaries: the message survived and the *type* did
/// not, so a lost sandbox and a bad configuration reached the runner as the same
/// thing - an "invalid argument" that no argument caused. `Core` unwraps rather
/// than nests, so a core error that crossed into the agent stack and back is the
/// error it started as.
impl From<Error> for retrograd_core::Error {
    fn from(error: Error) -> Self {
        match error {
            Error::Core(error) => error,
            Error::Invalid(message) => retrograd_core::Error::config(message),
            // The world, the tools and the judge are things that ran and broke.
            Error::PolicyStopped
            | Error::PolicyTask(_)
            | Error::PolicyGeneration(_)
            | Error::Tool(_)
            | Error::Sandbox(_)
            | Error::Reward(_) => retrograd_core::Error::runtime(error.to_string()),
        }
    }
}

#[cfg(test)]
mod facade_tests {
    use super::*;

    /// A core error that crossed into the agent stack comes back as itself,
    /// prefix included - not wrapped in a second sentence.
    #[test]
    fn a_core_error_survives_the_round_trip() {
        let original = retrograd_core::Error::overflow("group size overflows usize");
        let crossed: Error = original.into();
        let back: retrograd_core::Error = crossed.into();
        assert_eq!(back.kind(), retrograd_core::ErrorKind::Overflow);
        assert_eq!(
            back.to_string(),
            "arithmetic overflow: group size overflows usize"
        );
    }

    /// A dead sandbox is not something the user typed, and must not be reported
    /// as one - avoiding the ambiguity of formatting the source error as text.
    #[test]
    fn a_sandbox_failure_is_not_a_user_error() {
        let error: retrograd_core::Error = Error::sandbox("container is gone").into();
        assert_eq!(error.kind(), retrograd_core::ErrorKind::Runtime);
        assert!(!error.is_user_error());
        assert!(
            error
                .to_string()
                .contains("sandbox failure: container is gone")
        );
    }

    #[test]
    fn an_agent_configuration_error_is_a_configuration_error() {
        let error: retrograd_core::Error =
            Error::invalid("agent.group_size must be at least 2").into();
        assert_eq!(error.kind(), retrograd_core::ErrorKind::Config);
        assert!(error.is_user_error());
    }
}
