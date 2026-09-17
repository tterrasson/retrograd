use std::path::PathBuf;

/// Startup failures [`ObserveSink::open`](crate::ObserveSink::open) reports.
/// Everything after the opening is a warning, never an error.
#[derive(Debug, thiserror::Error)]
pub enum ObserveError {
    #[error("observe.directory {}: {source}", path.display())]
    Directory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("the observe writer thread could not start: {0}")]
    Thread(std::io::Error),
}

impl From<ObserveError> for retrograd_core::Error {
    fn from(error: ObserveError) -> Self {
        match error {
            ObserveError::Directory { .. } => Self::config(error.to_string()),
            ObserveError::Thread(_) => Self::runtime(error.to_string()),
        }
    }
}
