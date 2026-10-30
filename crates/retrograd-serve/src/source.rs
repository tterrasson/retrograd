//! What the model ids of the standalone server name: the base model under
//! `base`, and each configured adapter under its own name. Every id is a
//! cold target - the serving session holds the base loaded and registers the
//! adapters beside it, so switching between them does not reload the base.

use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use retrograd_openai::{BASE_MODEL_ID, ModelCard, ModelSource, OpenAiError, Target, WeightsSpec};

use crate::config::ServeConfig;

/// The model ids of one configuration: `base` plus one id per adapter.
pub struct AdapterSource {
    base: WeightsSpec,
    adapters: Vec<(String, WeightsSpec)>,
    created: u64,
}

impl AdapterSource {
    pub fn from_config(config: &ServeConfig) -> Result<Self, crate::ServeError> {
        let device = config.device()?;
        let n_ctx = config.n_ctx();
        let n_seq_max = config.n_seq_max;
        let spec = |adapter: Option<std::path::PathBuf>| WeightsSpec {
            model: config.model.clone(),
            adapter,
            n_ctx,
            device,
            n_seq_max,
            chat_template_variables: None,
        };
        let base = spec(None);
        let adapters = config
            .adapters
            .iter()
            .map(|decl| (decl.name.clone(), spec(Some(decl.path.clone()))))
            .collect();
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0);
        Ok(Self {
            base,
            adapters,
            created,
        })
    }

    /// The weights to load at startup: the base, with the first adapter
    /// registered beside it when there is one.
    pub fn preload(&self) -> WeightsSpec {
        self.adapters
            .first()
            .map(|(_, spec)| spec.clone())
            .unwrap_or_else(|| self.base.clone())
    }
}

#[async_trait]
impl ModelSource for AdapterSource {
    async fn list(&self) -> Result<Vec<ModelCard>, OpenAiError> {
        let mut cards = vec![ModelCard::new(BASE_MODEL_ID, self.created)];
        cards.extend(
            self.adapters
                .iter()
                .map(|(name, _)| ModelCard::new(name, self.created)),
        );
        Ok(cards)
    }

    async fn resolve(&self, model: &str) -> Result<Target, OpenAiError> {
        if model == BASE_MODEL_ID {
            return Ok(Target::Cold {
                spec: self.base.clone(),
                fingerprint: BASE_MODEL_ID.to_owned(),
            });
        }
        let Some((name, spec)) = self.adapters.iter().find(|(name, _)| name == model) else {
            return Err(OpenAiError::model_not_found(
                model,
                format!(
                    "this server serves `{}` and the adapters {}",
                    BASE_MODEL_ID,
                    self.adapters
                        .iter()
                        .map(|(name, _)| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        };
        Ok(Target::Cold {
            spec: spec.clone(),
            fingerprint: name.clone(),
        })
    }
}
