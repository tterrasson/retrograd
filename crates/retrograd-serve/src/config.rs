//! The standalone server's minimal TOML configuration: the model, its
//! adapters, the KV pool geometry, the device, the auth, the queue, and the
//! head options of a decision model. Nothing about training - this file is
//! parsed here, not by the run-config crate.

use std::net::IpAddr;
use std::path::PathBuf;

use retrograd_core::Device;
use serde::Deserialize;

/// One LoRA checkpoint of the base model, served under `name`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterDecl {
    pub name: String,
    pub path: PathBuf,
}

/// The whole configuration of `retrograd-infer`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeConfig {
    /// The base GGUF.
    pub model: PathBuf,
    /// `[[adapters]]`: checkpoints of `model` served under their `name`,
    /// beside the base model under `base`.
    #[serde(default)]
    pub adapters: Vec<AdapterDecl>,
    /// `auto` (the default), `cpu` or `gpu`.
    #[serde(default)]
    pub device: Option<String>,
    /// Per-sequence context, in tokens. Default 8192.
    #[serde(default)]
    pub n_ctx: Option<u32>,
    /// Sequence slots of the KV pool. Absent: the serving default, 8.
    #[serde(default)]
    pub n_seq_max: Option<u32>,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Bearer token. Required when `host` is not a loopback address.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Chat requests waiting for admission. Default 8.
    #[serde(default)]
    pub queue: Option<usize>,
    /// `continuous` (the default) or `lockstep`.
    #[serde(default)]
    pub admission: Option<String>,
    /// Seconds a loaded model may sit unused before it is dropped. Default 300.
    #[serde(default)]
    pub idle_seconds: Option<u64>,
    /// Directory of the on-disk Responses store (one JSON per response, atomic
    /// writes). Absent: the bounded in-memory LRU, which never writes to disk.
    #[serde(default)]
    pub responses_store_dir: Option<PathBuf>,
    /// `[decision]`: what replaces the head settings of a decision model.
    /// Refused at load for a model that is not one.
    #[serde(default)]
    pub decision: DecisionSection,
}

/// `[decision]`. Every key is absent by default, and absent means the model's
/// own value - what llama-server serves from the same file.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSection {
    /// A `systemone` Jinja template rendered in place of the model's. It takes
    /// the same context and must keep one marker per option.
    #[serde(default)]
    pub template: Option<PathBuf>,
    /// Replaces `<arch>.decision.max_head_tokens` (marker readouts only).
    #[serde(default)]
    pub max_head_tokens: Option<usize>,
    /// Questions evaluated together in one forward pass, at most `n_seq_max`.
    /// Default 1. More is faster, and costs exactness: the answer to a
    /// question then depends on the other questions of its request, and
    /// drifts from llama-server's by up to 1e-1 on a sensitive model.
    #[serde(default)]
    pub pack: Option<usize>,
}

impl DecisionSection {
    pub fn options(&self) -> retrograd_openai::DecisionOptions {
        retrograd_openai::DecisionOptions {
            template: self.template.clone(),
            max_head_tokens: self.max_head_tokens,
            pack: self.pack,
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".into()
}

fn default_port() -> u16 {
    8000
}

impl ServeConfig {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|error| ConfigError::Read {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
        let config: Self = toml::from_str(&text).map_err(|error| ConfigError::Parse {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn device(&self) -> Result<Device, ConfigError> {
        match &self.device {
            None => Ok(Device::Auto),
            Some(value) => value.parse().map_err(|_| {
                ConfigError::Invalid(format!(
                    "device must be one of auto, cpu, gpu; got `{value}`"
                ))
            }),
        }
    }

    pub fn address(&self) -> Result<std::net::SocketAddr, ConfigError> {
        let host: IpAddr = match self.host.as_str() {
            "localhost" => IpAddr::from([127, 0, 0, 1]),
            host => host.parse().map_err(|_| {
                ConfigError::Invalid(format!(
                    "host must be an IP address or `localhost`; got `{host}`"
                ))
            })?,
        };
        Ok(std::net::SocketAddr::new(host, self.port))
    }

    pub fn admission(&self) -> Result<retrograd_infer::AdmissionMode, ConfigError> {
        match self.admission.as_deref() {
            None | Some("continuous") => Ok(retrograd_infer::AdmissionMode::Continuous),
            Some("lockstep") => Ok(retrograd_infer::AdmissionMode::Lockstep),
            Some(other) => Err(ConfigError::Invalid(format!(
                "admission must be `continuous` or `lockstep`; got `{other}`"
            ))),
        }
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx.unwrap_or(8192)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if !self.model.is_file() {
            return Err(ConfigError::Invalid(format!(
                "model {} does not exist",
                self.model.display()
            )));
        }
        let mut names = std::collections::HashSet::new();
        for adapter in &self.adapters {
            if adapter.name.trim().is_empty() {
                return Err(ConfigError::Invalid(
                    "an adapter name must not be empty".into(),
                ));
            }
            if adapter.name == retrograd_openai::BASE_MODEL_ID {
                return Err(ConfigError::Invalid(format!(
                    "an adapter name must not be `{}`: that id serves the base model",
                    retrograd_openai::BASE_MODEL_ID
                )));
            }
            if !names.insert(adapter.name.clone()) {
                return Err(ConfigError::Invalid(format!(
                    "two adapters are named `{}`",
                    adapter.name
                )));
            }
            if !adapter.path.is_file() {
                return Err(ConfigError::Invalid(format!(
                    "adapter `{}`: {} does not exist",
                    adapter.name,
                    adapter.path.display()
                )));
            }
        }
        if let Some(queue) = self.queue
            && queue == 0
        {
            return Err(ConfigError::Invalid(
                "queue must be greater than zero".into(),
            ));
        }
        if let Some(n_ctx) = self.n_ctx
            && n_ctx == 0
        {
            return Err(ConfigError::Invalid(
                "n_ctx must be greater than zero".into(),
            ));
        }
        if let Some(n_seq_max) = self.n_seq_max
            && n_seq_max == 0
        {
            return Err(ConfigError::Invalid(
                "n_seq_max must be greater than zero".into(),
            ));
        }
        if let Some(template) = &self.decision.template
            && !template.is_file()
        {
            return Err(ConfigError::Invalid(format!(
                "decision.template: {} does not exist",
                template.display()
            )));
        }
        if self.decision.max_head_tokens == Some(0) {
            return Err(ConfigError::Invalid(
                "decision.max_head_tokens must be greater than zero".into(),
            ));
        }
        if let Some(pack) = self.decision.pack {
            let slots = self
                .n_seq_max
                .unwrap_or(retrograd_openai::backend::DEFAULT_SERVING_SLOTS);
            // Widening u32 -> usize.
            if pack == 0 || pack > slots as usize {
                return Err(ConfigError::Invalid(format!(
                    "decision.pack must be between 1 and n_seq_max ({slots}); got {pack}"
                )));
            }
        }
        // Fail at load, not at first use.
        self.admission()?;
        self.device()?;
        self.check_network_rule()?;
        Ok(())
    }

    /// The one rule a command-line override can break: a model served without
    /// a key on an address other machines reach should be a decision, never a
    /// typo.
    pub fn check_network_rule(&self) -> Result<(), ConfigError> {
        check_network_rule(self.address()?, self.api_key.as_deref())
    }
}

pub(crate) fn check_network_rule(
    address: std::net::SocketAddr,
    api_key: Option<&str>,
) -> Result<(), ConfigError> {
    if api_key.is_some_and(|key| key.is_empty() || key.trim() != key) {
        return Err(ConfigError::Invalid(
            "api_key must not be empty or have surrounding whitespace".into(),
        ));
    }
    if !address.ip().is_loopback() && api_key.is_none() {
        return Err(ConfigError::Invalid(format!(
            "refusing to serve on {address} without api_key: the model would be open to the network"
        )));
    }
    Ok(())
}

/// The configuration file's failures, as the binary reports them.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read {path}: {reason}")]
    Read { path: PathBuf, reason: String },
    #[error("invalid configuration in {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(text: &str) -> Result<ServeConfig, ConfigError> {
        // A real file for the existence checks: /dev/null is not a regular file.
        let file = std::env::temp_dir().join(format!(
            "retrograd-serve-test-{}-{}.gguf",
            std::process::id(),
            text.len()
        ));
        std::fs::write(&file, b"test").expect("write the fixture file");
        let mut config: ServeConfig = toml::from_str(text).map_err(|error| ConfigError::Parse {
            path: PathBuf::from("test.toml"),
            message: error.to_string(),
        })?;
        config.model = file.clone();
        for adapter in &mut config.adapters {
            adapter.path = file.clone();
        }
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn a_minimal_config_parses_and_resolves_its_defaults() {
        let config = config(r#"model = "/dev/null""#).unwrap();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8000);
        assert_eq!(config.n_ctx(), 8192);
        assert_eq!(
            config.admission().unwrap(),
            retrograd_infer::AdmissionMode::Continuous
        );
        assert!(matches!(config.device().unwrap(), Device::Auto));
        assert!(config.adapters.is_empty());
    }

    #[test]
    fn adapters_and_knobs_parse() {
        let config = config(
            r#"
            model = "/dev/null"
            device = "cpu"
            n_ctx = 2048
            n_seq_max = 4
            host = "0.0.0.0"
            port = 9000
            api_key = "sk-test"
            queue = 16
            admission = "lockstep"
            idle_seconds = 60
            [[adapters]]
            name = "best"
            path = "/dev/null"
            "#,
        )
        .unwrap();
        assert_eq!(config.adapters.len(), 1);
        assert_eq!(config.adapters[0].name, "best");
        assert_eq!(config.n_ctx(), 2048);
        assert_eq!(
            config.admission().unwrap(),
            retrograd_infer::AdmissionMode::Lockstep
        );
        assert!(matches!(config.device().unwrap(), Device::Cpu));
        assert_eq!(config.address().unwrap().port(), 9000);
        assert_eq!(
            config.decision.options(),
            retrograd_openai::DecisionOptions::default()
        );
    }

    #[test]
    fn the_decision_section_parses_into_the_head_options() {
        let config = config(
            r#"
            model = "/dev/null"
            n_seq_max = 4
            [decision]
            max_head_tokens = 512
            pack = 4
            "#,
        )
        .unwrap();
        let options = config.decision.options();
        assert_eq!(options.max_head_tokens, Some(512));
        assert_eq!(options.pack, Some(4));
        assert_eq!(options.template, None);
    }

    #[test]
    fn the_refusals_name_the_field() {
        let base = r#"model = "/dev/null""#;
        for (text, expected) in [
            (
                format!("{base}\nadmission = \"turbo\""),
                "admission must be `continuous` or `lockstep`",
            ),
            (
                format!("{base}\nqueue = 0"),
                "queue must be greater than zero",
            ),
            (
                format!("{base}\nn_ctx = 0"),
                "n_ctx must be greater than zero",
            ),
            (
                format!("{base}\nn_seq_max = 0"),
                "n_seq_max must be greater than zero",
            ),
            (
                format!("{base}\n[decision]\nmax_head_tokens = 0"),
                "decision.max_head_tokens must be greater than zero",
            ),
            (
                format!("{base}\nn_seq_max = 2\n[decision]\npack = 3"),
                "decision.pack must be between 1 and n_seq_max (2)",
            ),
            (
                format!("{base}\n[decision]\ntemplate = \"/no/such/template.jinja\""),
                "decision.template: /no/such/template.jinja does not exist",
            ),
            (
                format!("{base}\ndevice = \"tpu\""),
                "device must be one of auto, cpu, gpu",
            ),
            (
                format!("{base}\nhost = \"example.org\""),
                "host must be an IP address",
            ),
            (format!("{base}\nhost = \"0.0.0.0\""), "without api_key"),
            (
                format!("{base}\napi_key = \"\""),
                "api_key must not be empty",
            ),
            (
                format!("{base}\napi_key = \" token \""),
                "surrounding whitespace",
            ),
            (
                format!("{base}\n[[adapters]]\nname = \"base\"\npath = \"/dev/null\""),
                "that id serves the base model",
            ),
        ] {
            let error = config(&text).unwrap_err().to_string();
            assert!(error.contains(expected), "{text} -> {error}");
        }
        // A duplicate adapter name is refused.
        let text = format!(
            "{base}\n[[adapters]]\nname = \"a\"\npath = \"/dev/null\"\n[[adapters]]\nname = \"a\"\npath = \"/dev/null\""
        );
        let error = config(&text).unwrap_err().to_string();
        assert!(error.contains("two adapters are named `a`"), "{error}");
    }
}
