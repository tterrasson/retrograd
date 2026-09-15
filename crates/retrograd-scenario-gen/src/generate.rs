use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use retrograd_agent_core::{Error, Result, Scenario};
use retrograd_config::ScenarioGenerationConfig;
use retrograd_core::hex_lower;
use retrograd_tools::{ToolCatalog, ToolSource};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};

use crate::client;
use crate::prompt::{PROMPT_VERSION, render_prompt};
use crate::types::{CatalogSummary, GenerationManifest};
use crate::validate::{NEAR_DUPLICATE_THRESHOLD, text_hash, validate_draft};
use crate::{GeneratedCorpus, RejectionCounts, ScenarioDraft};

/// Asks the configured model for `config.count` scenario drafts, in batches of
/// `config.batch_size`, validating and deduplicating as they arrive.
///
/// A batch that accepts nothing resets no counter but the retry one:
/// `max_retries` bounds how many *unproductive* batches in a row are
/// tolerated, not the total number of requests, so a generator that keeps
/// making progress is never cut off early. Generation fails once that budget
/// is exhausted with fewer than `config.count` accepted drafts.
pub async fn generate_scenarios(
    catalog: &ToolCatalog,
    config: &ScenarioGenerationConfig,
) -> Result<GeneratedCorpus> {
    config.validate()?;
    if catalog.tools.is_empty() && catalog.resources.is_empty() {
        return Err(Error::invalid(
            "cannot generate scenarios from an empty tool catalog",
        ));
    }
    // Check the complete catalogue budget before the first paid request.
    let initial_prompt = render_prompt(catalog, config, config.batch_size.min(config.count), &[])?;
    let mut accepted = Vec::<ScenarioDraft>::new();
    let mut rejected = RejectionCounts::default();
    let mut last_error = None;
    // `max_retries` bounds *unproductive* batches, not batches: counting every
    // request against it caps the corpus at `batch_size * (max_retries + 1)`,
    // so a perfectly ordinary `count = 100, batch_size = 10, max_retries = 2`
    // could never reach its target however well the generator answered.
    let mut retries = 0;
    while accepted.len() < config.count {
        let needed = config.count - accepted.len();
        let batch = needed.min(config.batch_size);
        let hashes = accepted
            .iter()
            .map(|draft| text_hash(&draft.task))
            .collect::<Vec<_>>();
        let prompt = render_prompt(catalog, config, batch, &hashes)?;
        let before = accepted.len();
        match client::request(config, &prompt.text, batch).await {
            Ok(drafts) => {
                for draft in drafts {
                    if accepted.len() == config.count {
                        break;
                    }
                    if validate_draft(
                        &draft,
                        catalog,
                        config.min_difficulty,
                        config.max_difficulty,
                        &accepted,
                        &mut rejected,
                    ) {
                        accepted.push(draft);
                    }
                }
            }
            Err(error) => last_error = Some(error),
        }
        if accepted.len() > before {
            // The batch made progress; the retry budget is for a generator that
            // has stopped producing anything usable, not for the corpus size.
            retries = 0;
            continue;
        }
        if retries == config.max_retries {
            break;
        }
        retries += 1;
        tokio::time::sleep(Duration::from_millis(200 * retries as u64)).await;
    }
    if accepted.len() != config.count {
        let suffix = last_error
            .map(|error| format!(": {error}"))
            .unwrap_or_default();
        return Err(Error::invalid(format!(
            "scenario generation accepted {} of {} after bounded retries{suffix}",
            accepted.len(),
            config.count
        )));
    }
    let mut scenarios = accepted
        .iter()
        .map(|draft| to_scenario(draft, catalog))
        .collect::<Vec<_>>();
    if config.shuffle {
        let seed = config.seed.unwrap_or(0);
        scenarios
            .sort_by_key(|scenario| hex_lower(&Sha256::digest(format!("{seed}|{}", scenario.id))));
    }
    let difficulty_distribution = accepted.iter().fold(BTreeMap::new(), |mut counts, draft| {
        *counts.entry(draft.difficulty.to_string()).or_insert(0) += 1;
        counts
    });
    let mut tool_coverage = catalog
        .tools
        .iter()
        .map(|entry| (entry.spec.name.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    for draft in &accepted {
        for name in &draft.expected_tools {
            *tool_coverage.entry(name.clone()).or_default() += 1;
        }
    }
    let mcp_servers = catalog
        .tools
        .iter()
        .filter_map(|entry| match &entry.source {
            ToolSource::Mcp { server } => Some(server.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    // Only the servers that actually declared themselves stateless: the
    // manifest is what a reader trusts to know which tools may be shared with
    // an environment, and the full server list would claim it of all of them.
    let stateless_shared = catalog
        .tools
        .iter()
        .filter_map(|entry| match &entry.source {
            ToolSource::Mcp { server } if !entry.stateful => Some(server.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let config_view = json!({
        "model": config.model, "base_url": config.base_url, "api_key_env": config.api_key_env,
        "count": config.count, "batch_size": config.batch_size, "timeout_secs": config.timeout_secs,
        "max_retries": config.max_retries, "seed": config.seed, "custom_instructions": config.custom_instructions,
        "min_difficulty": config.min_difficulty, "max_difficulty": config.max_difficulty,
        "max_catalog_bytes": config.max_catalog_bytes, "shuffle": config.shuffle,
    });
    let config_sha256 = hex_lower(&Sha256::digest(
        serde_json::to_vec(&config_view).expect("a Value always serializes"),
    ));
    let corpus_sha256 = crate::export::corpus_hash(&scenarios)?;
    let mut catalog_warnings = catalog.warnings.clone();
    for name in &mcp_servers {
        let prefix = format!("{name}__");
        for (tool, count) in tool_coverage
            .iter()
            .filter(|(tool, _)| tool.starts_with(&prefix))
        {
            if *count == 0 {
                catalog_warnings.push(format!(
                    "MCP tool '{tool}' was not cited by any generated scenario"
                ));
            }
        }
    }
    let manifest = GenerationManifest {
        schema_version: 1,
        generator_model: config.model.clone(),
        generator_base_url: config.base_url.clone(),
        prompt_version: PROMPT_VERSION.into(),
        prompt_sha256: initial_prompt.sha256,
        catalog_sha256: catalog.sha256.clone(),
        catalog: CatalogSummary {
            tools: catalog.tools.len(),
            session_tools: catalog.tools.iter().filter(|entry| entry.stateful).count(),
            mcp_servers,
            stateless_shared,
            resources: catalog.resources.len(),
            warnings: catalog_warnings,
        },
        config_sha256,
        corpus_sha256,
        requested: config.count,
        accepted: scenarios.len(),
        rejected,
        difficulty_distribution,
        tool_coverage,
        near_duplicate_threshold: NEAR_DUPLICATE_THRESHOLD,
    };
    Ok(GeneratedCorpus {
        scenarios,
        manifest,
    })
}

fn to_scenario(draft: &ScenarioDraft, catalog: &ToolCatalog) -> Scenario {
    let criteria = draft
        .success_criteria
        .iter()
        .map(|criterion| format!("- {}", criterion.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    let identity = format!("{}\n{}", crate::validate::normalized(&draft.task), criteria);
    let suffix = hex_lower(&Sha256::digest(identity.as_bytes()));
    let id = format!(
        "gen-{}-{}",
        &catalog.sha256[..catalog.sha256.len().min(8)],
        &suffix[..8]
    );
    let mut metadata = Map::new();
    metadata.insert("difficulty".into(), json!(draft.difficulty));
    metadata.insert("rubric".into(), json!(criteria));
    metadata.insert(
        "generation".into(),
        json!({
            "schema_version": 1,
            "catalog_sha256": catalog.sha256,
            "expected_tools": draft.expected_tools,
            "expected_resources": draft.expected_resources,
        }),
    );
    Scenario {
        id,
        system: Some("Use the available tools when useful.".into()),
        user: draft.task.trim().to_owned(),
        metadata,
    }
}

/// [`generate_scenarios`] on a dedicated current-thread runtime, for a caller
/// with no async context of its own (a CLI subcommand, notably).
pub fn generate_scenarios_blocking(
    catalog: &ToolCatalog,
    config: &ScenarioGenerationConfig,
) -> Result<GeneratedCorpus> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| Error::invalid(format!("scenario generation needs a runtime: {error}")))?;
    runtime.block_on(generate_scenarios(catalog, config))
}
