//! Small, dependency-free experiment metric exporters.
//!
//! A training run reports [`MetricEvent`]s to a [`MetricsBus`], which fans
//! each one out to every registered [`MetricsSink`]. Two sinks are provided:
//! [`TensorBoardSink`] writes a TFRecord event file TensorBoard can read
//! directly, and [`WandbExportSink`] writes a JSONL staging area for
//! `scripts/import_wandb.py` to upload later, since this crate cannot talk to
//! the W&B API itself without pulling in its dependencies.

use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use retrograd_core::{Error, Result, TrainMetrics};

/// Run-level configuration recorded once, at [`MetricEvent::RunStarted`].
#[derive(Clone, Debug, Serialize)]
pub struct RunMetadata {
    pub algorithm: String,
    pub model: String,
    pub train_data: String,
    pub eval_data: Option<String>,
    pub epochs: u32,
    pub learning_rate: f32,
    pub scheduler: String,
    pub warmup_steps: u64,
}

/// One point in a run's lifecycle, as reported to a [`MetricsSink`].
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum MetricEvent {
    RunStarted {
        metadata: RunMetadata,
    },
    Step {
        epoch: u32,
        global_step: u64,
        values: Vec<MetricValue>,
    },
    RunFinished {
        global_step: u64,
    },
    RunFailed {
        message: String,
    },
}

/// One named scalar measurement, e.g. `train/loss` at `0.42`. Names use a
/// `namespace/metric` convention so sinks can group related series.
#[derive(Clone, Debug, Serialize)]
pub struct MetricValue {
    pub name: Cow<'static, str>,
    pub value: f32,
}

/// A destination for [`MetricEvent`]s, e.g. TensorBoard or a W&B staging area.
pub trait MetricsSink {
    fn emit(&mut self, event: &MetricEvent) -> Result<()>;
}

/// Fans out each event to every registered [`MetricsSink`].
pub struct MetricsBus {
    sinks: Vec<Box<dyn MetricsSink>>,
}

impl Default for MetricsBus {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsBus {
    pub fn new() -> Self {
        Self { sinks: Vec::new() }
    }

    pub fn add(&mut self, sink: impl MetricsSink + 'static) {
        self.sinks.push(Box::new(sink));
    }

    /// Adds a sink whose type the caller has already erased. Needed by callers
    /// that receive sinks as data - a server handed a list of destinations
    /// cannot name their types.
    pub fn add_boxed(&mut self, sink: Box<dyn MetricsSink>) {
        self.sinks.push(sink);
    }

    /// Sends `event` to every sink, even if one fails. A sink is an
    /// independent destination, so a broken filesystem or exporter must not
    /// silently drop the event everywhere else; the first error is returned
    /// after the fan-out completes so callers still fail loudly.
    pub fn emit(&mut self, event: &MetricEvent) -> Result<()> {
        let mut first_error = None;
        for sink in &mut self.sinks {
            if let Err(error) = sink.emit(event) {
                // Exporters are independent destinations. Keep emitting to
                // the remaining sinks so one broken filesystem or exporter
                // cannot silently drop the event everywhere else; return the
                // first error after fan-out so callers still fail loudly.
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Builds the [`MetricValue`]s for one training step. Eval loss is included
/// only when `include_eval` is set and the value is finite, since an eval
/// pass that has not run yet reports `NaN` and must not be charted.
pub fn values_from_metrics(metrics: TrainMetrics, include_eval: bool) -> Vec<MetricValue> {
    let mut values = vec![
        MetricValue {
            name: "train/loss".into(),
            value: metrics.train_loss,
        },
        MetricValue {
            name: "system/tokens_per_second".into(),
            value: metrics.tokens_per_second,
        },
        MetricValue {
            name: "optimizer/learning_rate".into(),
            value: metrics.learning_rate,
        },
    ];
    if include_eval && metrics.eval_loss.is_finite() {
        values.push(MetricValue {
            name: "eval/loss".into(),
            value: metrics.eval_loss,
        });
    }
    values
}

/// TensorBoard Event-file writer for scalar summaries. The file follows the
/// TFRecord framing and Event/Summary protobuf wire format, so it needs no
/// TensorFlow dependency at runtime.
pub struct TensorBoardSink {
    writer: BufWriter<File>,
}

impl TensorBoardSink {
    /// Creates a fresh `run-<millis>-<pid>` directory under `root` and opens
    /// its event file. A new directory per run keeps concurrent runs, and
    /// repeated runs against the same `root`, from overwriting each other.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let run_dir = create_run_dir(root)?;
        let millis = now_millis();
        let path = run_dir.join(format!("events.out.tfevents.{millis}.retrograd"));
        let mut sink = Self {
            writer: BufWriter::new(File::create(path)?),
        };
        sink.write_event(
            0,
            &[
                0x1a, 13, b'b', b'r', b'a', b'i', b'n', b'.', b'E', b'v', b'e', b'n', b't', b':',
                b'2',
            ],
        )?;
        sink.writer.flush()?;
        Ok(sink)
    }

    fn write_event(&mut self, step: u64, payload: &[u8]) -> Result<()> {
        let mut event = Vec::with_capacity(24 + payload.len());
        event.push(0x09); // wall_time, fixed64
        event.extend_from_slice(&(now_millis() as f64 / 1000.0).to_le_bytes());
        event.push(0x10); // step
        put_varint(&mut event, step);
        event.extend_from_slice(payload);
        self.write_record(&event)
    }

    fn write_scalar(&mut self, step: u64, name: &str, value: f32) -> Result<()> {
        let mut scalar = Vec::new();
        scalar.push(0x0a); // Summary.Value.tag
        put_varint(&mut scalar, name.len() as u64);
        scalar.extend_from_slice(name.as_bytes());
        scalar.push(0x15); // Summary.Value.simple_value
        scalar.extend_from_slice(&value.to_le_bytes());

        let mut summary = Vec::new();
        summary.push(0x0a); // Summary.value
        put_varint(&mut summary, scalar.len() as u64);
        summary.extend_from_slice(&scalar);

        let mut payload = Vec::new();
        payload.push(0x2a); // Event.summary
        put_varint(&mut payload, summary.len() as u64);
        payload.extend_from_slice(&summary);
        self.write_event(step, &payload)
    }

    fn write_record(&mut self, data: &[u8]) -> Result<()> {
        let length = (data.len() as u64).to_le_bytes();
        self.writer.write_all(&length)?;
        self.writer
            .write_all(&masked_crc32c(&length).to_le_bytes())?;
        self.writer.write_all(data)?;
        self.writer.write_all(&masked_crc32c(data).to_le_bytes())?;
        Ok(())
    }
}

fn create_run_dir(root: &Path) -> Result<std::path::PathBuf> {
    let millis = now_millis();
    let pid = std::process::id();

    for attempt in 0..1000 {
        let suffix = if attempt == 0 {
            String::new()
        } else {
            format!("-{attempt}")
        };
        let path = root.join(format!("run-{millis}-{pid}{suffix}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }

    Err(Error::runtime(format!(
        "could not allocate a unique TensorBoard run directory under {}",
        root.display()
    )))
}

impl MetricsSink for TensorBoardSink {
    fn emit(&mut self, event: &MetricEvent) -> Result<()> {
        if let MetricEvent::Step {
            global_step,
            values,
            ..
        } = event
        {
            for value in values {
                self.write_scalar(*global_step, &value.name, value.value)?;
            }
            self.writer.flush()?;
        }
        Ok(())
    }
}

/// Portable staging area consumed by `scripts/import_wandb.py` after a run.
pub struct WandbExportSink {
    writer: BufWriter<File>,
}

impl WandbExportSink {
    /// Writes `run.json` (the run's [`RunMetadata`]) and opens `metrics.jsonl`
    /// under `dir`, ready to receive one [`MetricEvent`] per line.
    pub fn new(dir: impl AsRef<Path>, metadata: &RunMetadata) -> Result<Self> {
        fs::create_dir_all(dir.as_ref())?;
        let manifest = serde_json::to_vec_pretty(metadata)
            .map_err(|error| Error::runtime(format!("serialize W&B metadata: {error}")))?;
        fs::write(dir.as_ref().join("run.json"), manifest)?;
        Ok(Self {
            writer: BufWriter::new(File::create(dir.as_ref().join("metrics.jsonl"))?),
        })
    }
}

impl MetricsSink for WandbExportSink {
    fn emit(&mut self, event: &MetricEvent) -> Result<()> {
        serde_json::to_writer(&mut self.writer, event)
            .map_err(|error| Error::runtime(format!("serialize W&B metric event: {error}")))?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn masked_crc32c(data: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x82f63b78
            } else {
                crc >> 1
            };
        }
    }
    crc = !crc;
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct RecordingSink {
        events: Rc<RefCell<Vec<String>>>,
        fail: bool,
    }

    impl MetricsSink for RecordingSink {
        fn emit(&mut self, event: &MetricEvent) -> Result<()> {
            if self.fail {
                return Err(Error::runtime("sink failed"));
            }
            let kind = match event {
                MetricEvent::RunStarted { .. } => "started",
                MetricEvent::Step { .. } => "step",
                MetricEvent::RunFinished { .. } => "finished",
                MetricEvent::RunFailed { .. } => "failed",
            };
            self.events.borrow_mut().push(kind.into());
            Ok(())
        }
    }

    #[test]
    fn crc32c_matches_the_tfrecord_masking_example() {
        assert_eq!(masked_crc32c(b""), 0xa282ead8);
    }

    #[test]
    fn metric_values_include_eval_only_when_requested_and_finite() {
        let metrics = TrainMetrics {
            train_loss: 1.25,
            eval_loss: 2.5,
            tokens_per_second: 42.0,
            learning_rate: 3.0e-4,
            ..TrainMetrics::default()
        };
        let names = |values: Vec<MetricValue>| {
            values
                .into_iter()
                .map(|value| value.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(values_from_metrics(metrics, true)),
            [
                "train/loss",
                "system/tokens_per_second",
                "optimizer/learning_rate",
                "eval/loss",
            ]
        );
        assert!(!names(values_from_metrics(metrics, false)).contains(&"eval/loss".into()));
        assert!(
            !names(values_from_metrics(
                TrainMetrics {
                    eval_loss: f32::NAN,
                    ..metrics
                },
                true,
            ))
            .contains(&"eval/loss".into())
        );
    }

    #[test]
    fn metrics_bus_fans_out_and_propagates_sink_errors() {
        let first = Rc::new(RefCell::new(Vec::new()));
        let last = Rc::new(RefCell::new(Vec::new()));
        let mut bus = MetricsBus::new();
        bus.add(RecordingSink {
            events: first.clone(),
            fail: false,
        });
        bus.add(RecordingSink {
            events: Rc::new(RefCell::new(Vec::new())),
            fail: true,
        });
        bus.add(RecordingSink {
            events: last.clone(),
            fail: false,
        });
        let error = bus
            .emit(&MetricEvent::RunFinished { global_step: 2 })
            .unwrap_err();
        assert!(error.to_string().contains("sink failed"));
        assert_eq!(&*first.borrow(), &["finished"]);
        assert_eq!(
            &*last.borrow(),
            &["finished"],
            "a failed sink must not prevent later sinks from receiving the event"
        );
    }

    #[test]
    fn tensorboard_writes_valid_tfrecord_framing_and_scalar() {
        let dir = std::env::temp_dir().join("retrograd-tensorboard-test");
        let _ = fs::remove_dir_all(&dir);
        let mut sink = TensorBoardSink::new(&dir).unwrap();
        sink.emit(&MetricEvent::Step {
            epoch: 1,
            global_step: 3,
            values: vec![MetricValue {
                name: "train/loss".into(),
                value: 0.5,
            }],
        })
        .unwrap();
        let run_dirs: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(run_dirs.len(), 1);
        assert!(run_dirs[0].is_dir());
        let path = fs::read_dir(&run_dirs[0])
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let bytes = fs::read(path).unwrap();
        let mut offset = 0;
        let mut records = Vec::new();
        while offset < bytes.len() {
            let length_bytes: [u8; 8] = bytes[offset..offset + 8].try_into().unwrap();
            let length = u64::from_le_bytes(length_bytes) as usize;
            let length_crc = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
            assert_eq!(length_crc, masked_crc32c(&length_bytes));
            let payload_start = offset + 12;
            let payload_end = payload_start + length;
            let payload = &bytes[payload_start..payload_end];
            let payload_crc =
                u32::from_le_bytes(bytes[payload_end..payload_end + 4].try_into().unwrap());
            assert_eq!(payload_crc, masked_crc32c(payload));
            records.push(payload);
            offset = payload_end + 4;
        }
        assert_eq!(records.len(), 2, "version record plus one scalar record");
        assert!(
            records[1]
                .windows(b"train/loss".len())
                .any(|window| window == b"train/loss")
        );

        let _second_sink = TensorBoardSink::new(&dir).unwrap();
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wandb_export_writes_parseable_jsonl_and_manifest() {
        let dir = std::env::temp_dir().join("retrograd-wandb-export-test");
        let _ = fs::remove_dir_all(&dir);
        let metadata = RunMetadata {
            algorithm: "sft".into(),
            model: "model.gguf".into(),
            train_data: "train.jsonl".into(),
            eval_data: None,
            epochs: 1,
            learning_rate: 1e-4,
            scheduler: "constant".into(),
            warmup_steps: 0,
        };
        let mut sink = WandbExportSink::new(&dir, &metadata).unwrap();
        sink.emit(&MetricEvent::RunFinished { global_step: 2 })
            .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(manifest["algorithm"], "sft");
        assert_eq!(manifest["train_data"], "train.jsonl");
        let event: serde_json::Value = serde_json::from_str(
            fs::read_to_string(dir.join("metrics.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(event["event"], "run_finished");
        assert_eq!(event["global_step"], 2);
        let _ = fs::remove_dir_all(dir);
    }
}
