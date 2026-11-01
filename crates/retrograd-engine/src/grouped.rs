use retrograd_core::{Error, Result};

/// One question of a grouped training step: its tokens as the model reads
/// them, the positions its options are read at, and the target probability of
/// each option.
#[derive(Clone, Copy, Debug)]
pub struct GroupedQuestion<'a> {
    pub tokens: &'a [i32],
    /// The positions of the option markers, in the order of `targets`.
    pub reads: &'a [i32],
    /// The column of the question type in the embeddings output.
    pub column: i32,
    pub targets: &'a [f32],
}

/// One physical step of the grouped softmax cross-entropy: several questions,
/// each its own sequence, side by side in one micro-batch.
///
/// Built by [`GroupedBatch::pack`], which lays the questions out the way the
/// runtime takes them: sequence `k` is question `k` with its own positions
/// from zero, a token is an output exactly when a group reads it, and the
/// groups pad to the widest question with row `-1` and a zero target.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupedBatch {
    pub tokens: Vec<i32>,
    pub positions: Vec<i32>,
    pub seq_ids: Vec<i32>,
    pub outputs: Vec<i8>,
    pub n_groups: u32,
    pub n_opt_max: u32,
    pub rows: Vec<i32>,
    pub columns: Vec<i32>,
    pub targets: Vec<f32>,
    /// The scores are multiplied by this before the softmax.
    pub inv_temperature: f32,
    /// Physical steps per optimizer update, for the update this step belongs to.
    pub accumulation_steps: u32,
}

impl GroupedBatch {
    pub fn pack(
        questions: &[GroupedQuestion<'_>],
        inv_temperature: f32,
        accumulation_steps: u32,
    ) -> Result<Self> {
        if questions.is_empty() {
            return Err(Error::invalid(
                "a grouped batch needs at least one question",
            ));
        }
        if !(inv_temperature > 0.0 && inv_temperature.is_finite()) {
            return Err(Error::invalid(
                "a grouped batch's inverse temperature must be finite and positive",
            ));
        }
        if accumulation_steps == 0 {
            return Err(Error::invalid(
                "a grouped batch needs at least one physical step per update",
            ));
        }
        let n_opt_max = questions
            .iter()
            .map(|question| question.reads.len())
            .max()
            .unwrap_or(0);
        let n_groups = questions.len();
        let mut batch = Self {
            tokens: Vec::new(),
            positions: Vec::new(),
            seq_ids: Vec::new(),
            outputs: Vec::new(),
            n_groups: to_u32(n_groups, "question count")?,
            n_opt_max: to_u32(n_opt_max, "option count")?,
            rows: vec![-1; n_groups * n_opt_max],
            columns: Vec::with_capacity(n_groups),
            targets: vec![0.0; n_groups * n_opt_max],
            inv_temperature,
            accumulation_steps,
        };
        for (group, question) in questions.iter().enumerate() {
            let n_tokens = question.tokens.len();
            if question.reads.len() < 2 || question.reads.len() != question.targets.len() {
                return Err(Error::invalid(format!(
                    "grouped question {group} reads {} options for {} targets; a group needs two \
                     or more, one target each",
                    question.reads.len(),
                    question.targets.len()
                )));
            }
            let sum: f32 = question.targets.iter().sum();
            if question
                .targets
                .iter()
                .any(|target| !target.is_finite() || *target < 0.0)
                || (sum - 1.0).abs() > 1e-3
            {
                return Err(Error::invalid(format!(
                    "grouped question {group}'s targets must be probabilities summing to 1"
                )));
            }
            let offset = batch.tokens.len();
            let mut outputs = vec![0_i8; n_tokens];
            for (option, &read) in question.reads.iter().enumerate() {
                let position = usize::try_from(read)
                    .ok()
                    .filter(|&position| position < n_tokens)
                    .ok_or_else(|| {
                        Error::invalid(format!(
                            "grouped question {group} reads position {read} outside its \
                             {n_tokens} tokens"
                        ))
                    })?;
                if outputs[position] != 0 {
                    return Err(Error::invalid(format!(
                        "grouped question {group} reads position {read} twice"
                    )));
                }
                outputs[position] = 1;
                batch.rows[group * n_opt_max + option] =
                    to_i32(offset + position, "grouped batch row")?;
                batch.targets[group * n_opt_max + option] = question.targets[option];
            }
            let sequence = to_i32(group, "grouped batch sequence")?;
            for (position, &token) in question.tokens.iter().enumerate() {
                batch.tokens.push(token);
                batch
                    .positions
                    .push(to_i32(position, "grouped batch position")?);
                batch.seq_ids.push(sequence);
            }
            batch.outputs.extend(outputs);
            batch.columns.push(question.column);
        }
        Ok(batch)
    }

    /// The number of tokens, which must fit one micro-batch.
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::overflow(format!("{what} {value} exceeds u32")))
}

fn to_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::overflow(format!("{what} {value} exceeds i32")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_become_sequences_with_their_reads_as_rows() {
        let batch = GroupedBatch::pack(
            &[
                GroupedQuestion {
                    tokens: &[10, 11, 12, 13],
                    reads: &[1, 3],
                    column: 0,
                    targets: &[1.0, 0.0],
                },
                GroupedQuestion {
                    tokens: &[20, 21, 22],
                    reads: &[0, 1, 2],
                    column: 2,
                    targets: &[0.25, 0.25, 0.5],
                },
            ],
            1.0,
            1,
        )
        .unwrap();
        assert_eq!(batch.tokens, [10, 11, 12, 13, 20, 21, 22]);
        assert_eq!(batch.positions, [0, 1, 2, 3, 0, 1, 2]);
        assert_eq!(batch.seq_ids, [0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(batch.outputs, [0, 1, 0, 1, 1, 1, 1]);
        assert_eq!((batch.n_groups, batch.n_opt_max), (2, 3));
        assert_eq!(batch.rows, [1, 3, -1, 4, 5, 6]);
        assert_eq!(batch.targets, [1.0, 0.0, 0.0, 0.25, 0.25, 0.5]);
        assert_eq!(batch.columns, [0, 2]);
    }

    #[test]
    fn a_bad_question_is_refused_by_name() {
        let pack = |reads: &[i32], targets: &[f32]| {
            GroupedBatch::pack(
                &[GroupedQuestion {
                    tokens: &[1, 2, 3],
                    reads,
                    column: 0,
                    targets,
                }],
                1.0,
                1,
            )
        };
        assert!(pack(&[0, 3], &[0.5, 0.5]).is_err());
        assert!(pack(&[1, 1], &[0.5, 0.5]).is_err());
        assert!(pack(&[1, 2], &[0.5, 0.6]).is_err());
        assert!(pack(&[1], &[1.0]).is_err());
        assert!(pack(&[1, 2], &[0.5, 0.5]).is_ok());
    }
}
