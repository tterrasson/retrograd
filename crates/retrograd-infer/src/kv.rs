//! The prefix cache's data structure: a radix tree of resident prefixes.
//!
//! Each node is a token path from the root; a node *with a slot* owns a dormant
//! sequence slot whose KV is frozen exactly at the node's end position (its
//! resident span is `[0, depth)`), so a whole-sequence [`DecodeBackend::seq_cp`]
//! from it is exact for the attention KV *and* for the recurrent tail of a
//! hybrid model (the fork's recurrent `seq_cp` links the source's current tail,
//! which is the state at the boundary precisely because the slot is frozen
//! there). A node may instead be *swapped*: its slot was evicted under pressure
//! and only a host snapshot (full sequence state, flags 0) survives, restorable
//! with [`DecodeBackend::seq_load`].
//!
//! The key of the whole tree is [`KvCacheKey`]: model/tokenizer fingerprint,
//! weights/adapter generation, and the context settings that change
//! activations. When the backend reports a different key (every weight mutation
//! bumps the version), the scheduler drops the tree: no prefix computed under
//! other weights is ever reused.
//!
//! Reuse rule: a node is reused at its exact boundary via a whole-slot copy -
//! always safe. A *mid-sequence* reuse (hit shorter than the node's depth,
//! partial-range copy) is only offered when the backend's pool is unified (a
//! partial cross-stream copy asserts in the backend) and the model is dense (a
//! hybrid's recurrent state cannot be truncated to an arbitrary mid-sequence
//! position). Otherwise the scheduler falls back to the deepest exact
//! boundary, or to a full prefill (recompute).
//!
//! [`DecodeBackend::seq_cp`]: crate::DecodeBackend::seq_cp
//! [`DecodeBackend::seq_load`]: crate::DecodeBackend::seq_load

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// FNV-1a mix, the tree key's hash.
pub(crate) fn mix(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash = hash.wrapping_mul(1099511628211);
    hash
}

/// The key every prefix entry is valid under: which model/tokenizer, which
/// weights/adapter generation, which context settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheKey {
    pub fingerprint: u64,
    pub weights_version: u64,
    /// Hash of the context settings (pool geometry, unified flag).
    pub settings: u64,
}

impl KvCacheKey {
    pub fn new(fingerprint: u64, weights_version: u64, settings: u64) -> Self {
        Self {
            fingerprint,
            weights_version,
            settings,
        }
    }
}

/// A pin handle returned by [`PrefixTree::pin_at`]. Ids are never reused, so
/// a stale handle cannot release another node's pin.
pub type PinId = u64;

/// Index of the root node; it carries no edge, slot or snapshot.
const ROOT: usize = 0;

#[derive(Default)]
struct Node {
    /// Edge label from the parent (empty for the root).
    edge: Vec<i32>,
    /// Parent node index (the root is its own parent).
    parent: usize,
    /// End position of the path: number of tokens from the root.
    depth: usize,
    /// Dormant slot frozen at `depth` (span `[0, depth)`), if resident.
    slot: Option<i32>,
    /// Full host snapshot of the boundary state (flags 0), captured when the
    /// slot was evicted under pressure (swap).
    snapshot: Option<Arc<[u8]>>,
    /// LRU clock value of the last use.
    last_use: u64,
    /// Pin count; pinned nodes are never evicted.
    pins: u32,
    /// First token of the child edge -> child node index.
    children: BTreeMap<i32, usize>,
}

/// Counters the serving layer reads for the reuse ratio and the saturation
/// behavior (prefilled vs prompt cells, evictions, swaps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvStats {
    /// Groups admitted with a cache hit of at least one token.
    pub hits: u64,
    /// Groups admitted cold (full prefill).
    pub misses: u64,
    /// Prompt cells submitted to the backend (the part actually prefilled).
    pub prefilled_cells: u64,
    /// Prompt cells the groups asked for (full prompt lengths).
    pub prompt_cells: u64,
    /// Nodes evicted (slot released, with or without a swap).
    pub evictions: u64,
    /// Nodes swapped to a host snapshot instead of dropped.
    pub swaps: u64,
    /// Current node count / resident slots / host snapshot bytes.
    pub nodes: usize,
    pub resident_slots: usize,
    pub snapshot_bytes: u64,
}

impl KvStats {
    /// `prefilled / prompt` over the admitted groups: the share of prompt
    /// cells the cache did not serve.
    pub fn prefilled_ratio(&self) -> f64 {
        if self.prompt_cells == 0 {
            0.0
        } else {
            self.prefilled_cells as f64 / self.prompt_cells as f64
        }
    }
}

/// Length of the common head of an edge and the remaining prompt.
fn common_len(edge: &[i32], tokens: &[i32]) -> usize {
    edge.iter().zip(tokens).take_while(|(a, b)| a == b).count()
}

/// The radix tree of resident prefixes. Pure data plus decisions; every backend
/// operation is performed by the caller (the scheduler), which owns the
/// backend.
pub struct PrefixTree {
    nodes: Vec<Node>,
    /// Indices of pruned nodes, reused by the next insertion.
    free: Vec<usize>,
    /// Live pins: id -> node.
    pins: HashMap<PinId, usize>,
    next_pin: PinId,
    clock: u64,
    snapshot_bytes: u64,
}

impl PrefixTree {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::default()],
            free: Vec::new(),
            pins: HashMap::new(),
            next_pin: 1,
            clock: 0,
            snapshot_bytes: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn alloc(&mut self, node: Node) -> usize {
        match self.free.pop() {
            Some(index) => {
                self.nodes[index] = node;
                index
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        }
    }

    /// The longest prefix of `tokens` that has a usable node, as
    /// `(node, hit)` with `hit <= tokens.len()`. `partial_ok` allows a
    /// mid-sequence hit (`hit < node.depth`, served by a partial-range copy);
    /// an exact hit (`hit == node.depth`) is served by a whole-slot copy and is
    /// always allowed. `usable(node, hit)` is the caller's backend-side check
    /// (e.g. the resident span really is `[0, depth)`).
    pub fn longest_prefix(
        &self,
        tokens: &[i32],
        partial_ok: bool,
        usable: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        let mut cur = ROOT;
        let mut pos = 0_usize;
        loop {
            let depth = self.nodes[cur].depth;
            if depth > 0 && usable(cur, depth) {
                best = Some((cur, depth));
            }
            // The child that continues the prompt, and how much of its edge
            // the prompt shares.
            let next = tokens.get(pos).and_then(|first| {
                let child = *self.nodes[cur].children.get(first)?;
                Some((child, common_len(&self.nodes[child].edge, &tokens[pos..])))
            });
            if let Some((child, matched)) = next
                && matched == self.nodes[child].edge.len()
            {
                cur = child;
                pos += matched;
                continue;
            }
            // The walk stops here (the prompt diverges, ends inside an edge or
            // at a slotless branch). A mid-sequence hit is still possible: the
            // prompt shares `pos + matched` tokens with every path below the
            // continuing child (or `pos` with any child), served by a
            // partial-range copy [0, hit) from a slot-bearing node there.
            if partial_ok {
                let (child, matched) = match next {
                    Some(found) => (Some(found.0), found.1),
                    None => (self.nodes[cur].children.values().next().copied(), 0),
                };
                let hit = pos + matched;
                if let Some(child) = child
                    && hit > 0
                    && best.is_none_or(|(_, best_hit)| hit > best_hit)
                    && let Some(slot_node) = self.deepest_slot_in_subtree(child)
                    && usable(slot_node, hit)
                {
                    best = Some((slot_node, hit));
                }
            }
            return best;
        }
    }

    /// The deepest slot-bearing node in `node`'s subtree (the node included),
    /// or None. Used to serve a mid-sequence partial hit from a subtree whose
    /// branch node itself carries no slot.
    fn deepest_slot_in_subtree(&self, node: usize) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            if self.nodes[current].slot.is_some()
                && best.is_none_or(|b| self.nodes[current].depth > self.nodes[b].depth)
            {
                best = Some(current);
            }
            stack.extend(self.nodes[current].children.values());
        }
        best
    }

    /// Marks a node as just used (LRU clock).
    pub fn touch(&mut self, node: usize) {
        self.nodes[node].last_use = self.tick();
    }

    /// Inserts the full path `tokens` and attaches `slot` at its end node.
    /// Returns `(node, attached)`: `attached` is false when the end node
    /// already owned a slot, in which case the caller must release the passed
    /// slot itself.
    pub fn insert(&mut self, tokens: &[i32], slot: i32) -> (usize, bool) {
        let mut cur = ROOT;
        let mut pos = 0_usize;
        while pos < tokens.len() {
            let first = tokens[pos];
            match self.nodes[cur].children.get(&first).copied() {
                Some(child) => {
                    let matched = common_len(&self.nodes[child].edge, &tokens[pos..]);
                    if matched < self.nodes[child].edge.len() {
                        // Split the edge at `matched`: a new mid node takes the
                        // shared head, the old child moves under it with the tail.
                        let tail = self.nodes[child].edge.split_off(matched);
                        let head = std::mem::replace(&mut self.nodes[child].edge, tail);
                        let mid = self.alloc(Node {
                            edge: head,
                            parent: cur,
                            depth: self.nodes[cur].depth + matched,
                            ..Node::default()
                        });
                        let tail_first = self.nodes[child].edge[0];
                        self.nodes[mid].children.insert(tail_first, child);
                        self.nodes[child].parent = mid;
                        self.nodes[cur].children.insert(first, mid);
                        cur = mid;
                    } else {
                        cur = child;
                    }
                    pos += matched;
                }
                None => {
                    let new = self.alloc(Node {
                        edge: tokens[pos..].to_vec(),
                        parent: cur,
                        depth: tokens.len(),
                        ..Node::default()
                    });
                    self.nodes[cur].children.insert(first, new);
                    cur = new;
                    pos = tokens.len();
                }
            }
        }
        let attached = cur != ROOT && self.nodes[cur].slot.is_none();
        if attached {
            self.nodes[cur].slot = Some(slot);
            self.nodes[cur].last_use = self.tick();
        }
        (cur, attached)
    }

    /// Attaches a host snapshot to a node whose slot was just evicted (swap).
    pub fn attach_snapshot(&mut self, node: usize, snapshot: Arc<[u8]>) {
        self.snapshot_bytes += snapshot.len() as u64;
        if let Some(old) = self.nodes[node].snapshot.replace(snapshot) {
            self.snapshot_bytes = self.snapshot_bytes.saturating_sub(old.len() as u64);
        }
    }

    /// The node's full snapshot, if swapped.
    pub fn snapshot(&self, node: usize) -> Option<&Arc<[u8]>> {
        self.nodes[node].snapshot.as_ref()
    }

    /// Whether the node currently owns a dormant slot.
    pub fn slot(&self, node: usize) -> Option<i32> {
        self.nodes[node].slot
    }

    /// Detaches the slot (after the caller seq_rm'd it) without dropping the
    /// node: it becomes a swapped node if it has a snapshot.
    pub fn detach_slot(&mut self, node: usize) {
        self.nodes[node].slot = None;
    }

    /// The node's path depth (end position).
    pub fn depth(&self, node: usize) -> usize {
        self.nodes[node].depth
    }

    /// Pins the node at exactly `tokens` (which must exist in the tree).
    pub fn pin_at(&mut self, tokens: &[i32]) -> Option<PinId> {
        let node = self.find_exact(tokens)?;
        Some(self.pin_node(node))
    }

    /// Pins `node` so eviction never releases it.
    pub fn pin_node(&mut self, node: usize) -> PinId {
        self.nodes[node].pins += 1;
        self.nodes[node].last_use = self.tick();
        let id = self.next_pin;
        self.next_pin += 1;
        self.pins.insert(id, node);
        id
    }

    /// Releases one pin; returns false when the id is unknown or released.
    pub fn unpin(&mut self, id: PinId) -> bool {
        let Some(node) = self.pins.remove(&id) else {
            return false;
        };
        self.nodes[node].pins -= 1;
        true
    }

    fn find_exact(&self, tokens: &[i32]) -> Option<usize> {
        let mut cur = ROOT;
        let mut pos = 0_usize;
        while pos < tokens.len() {
            let child = *self.nodes[cur].children.get(&tokens[pos])?;
            let edge = &self.nodes[child].edge;
            if !tokens[pos..].starts_with(edge) {
                return None;
            }
            cur = child;
            pos += edge.len();
        }
        (cur != ROOT).then_some(cur)
    }

    /// The LRU eviction candidate: an unpinned node owning a slot, least
    /// recently used. Swapped-only nodes hold host bytes, not pool cells, and
    /// are dropped through [`Self::snapshot_eviction_candidate`] under byte
    /// pressure instead.
    pub fn eviction_candidate(&self) -> Option<usize> {
        self.lru(|node| node.slot.is_some())
    }

    /// The LRU swapped-only node (host snapshot bytes), for byte-budget
    /// pressure.
    pub fn snapshot_eviction_candidate(&self) -> Option<usize> {
        self.lru(|node| node.slot.is_none() && node.snapshot.is_some())
    }

    fn lru(&self, eligible: impl Fn(&Node) -> bool) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, node)| node.pins == 0 && eligible(node))
            .min_by_key(|(_, node)| node.last_use)
            .map(|(index, _)| index)
    }

    /// Drops a node's payload (the caller has already released its slot) and
    /// prunes it and its now-childless, payload-free, unpinned ancestors up to
    /// the root. Pruned indices are recycled by later insertions.
    pub fn remove(&mut self, node: usize) {
        if let Some(snap) = self.nodes[node].snapshot.take() {
            self.snapshot_bytes = self.snapshot_bytes.saturating_sub(snap.len() as u64);
        }
        self.nodes[node].slot = None;
        let mut child = node;
        while child != ROOT {
            let current = &self.nodes[child];
            if !current.children.is_empty()
                || current.slot.is_some()
                || current.snapshot.is_some()
                || current.pins > 0
            {
                break;
            }
            let parent = current.parent;
            let first = current.edge[0];
            self.nodes[parent].children.remove(&first);
            self.nodes[child] = Node::default();
            self.free.push(child);
            child = parent;
        }
    }

    /// All resident slots (to release them on invalidation or shutdown).
    pub fn resident_slots(&self) -> impl Iterator<Item = i32> + '_ {
        self.nodes.iter().filter_map(|node| node.slot)
    }

    /// Cells held by dormant slots, counted once per node: an upper bound,
    /// since slots copied from one another share their common prefix cells
    /// on a unified pool.
    pub fn resident_cells(&self) -> u64 {
        self.nodes
            .iter()
            .filter(|node| node.slot.is_some())
            .map(|node| node.depth as u64)
            .sum()
    }

    /// Drops every node and pin; returns the slots the caller must `seq_rm`.
    pub fn clear(&mut self) -> Vec<i32> {
        let slots = self.resident_slots().collect();
        self.nodes.truncate(1);
        self.nodes[ROOT] = Node::default();
        self.free.clear();
        self.pins.clear();
        self.snapshot_bytes = 0;
        slots
    }

    /// Reachable nodes, the root excluded.
    pub fn node_count(&self) -> usize {
        self.nodes.len() - 1 - self.free.len()
    }

    pub fn resident_slot_count(&self) -> usize {
        self.resident_slots().count()
    }

    pub fn snapshot_bytes(&self) -> u64 {
        self.snapshot_bytes
    }
}

impl Default for PrefixTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_with(paths: &[&[i32]]) -> PrefixTree {
        let mut tree = PrefixTree::new();
        for (i, path) in paths.iter().enumerate() {
            tree.insert(path, (i + 1) as i32);
        }
        tree
    }

    #[test]
    fn longest_prefix_walks_edges_and_splits_on_divergence() {
        let tree = tree_with(&[&[1, 2, 3, 4], &[1, 2, 9, 9]]);
        let all = |_: usize, _: usize| true;
        // Exact boundary of the first path.
        assert_eq!(
            tree.longest_prefix(&[1, 2, 3, 4, 5], true, &mut |n, h| all(n, h)),
            Some((1, 4))
        );
        // Diverges mid-edge of the second path [1,2,9,9]: with partial hits
        // allowed, the common prefix with that edge (depth 3) is offered; with
        // only exact boundaries, the best is depth 2.
        assert_eq!(
            tree.longest_prefix(&[1, 2, 9, 8], true, &mut |n, h| all(n, h))
                .map(|(_, h)| h),
            Some(3)
        );
        assert_eq!(
            tree.longest_prefix(&[1, 2, 9, 8], false, &mut |n, h| all(n, h))
                .map(|(_, h)| h),
            Some(2)
        );
        // Mid-sequence: prompt ends inside an edge -> partial hit at prompt len.
        let tree = tree_with(&[&[1, 2, 3, 4]]);
        assert_eq!(
            tree.longest_prefix(&[1, 2], true, &mut |_, _| true),
            Some((1, 2))
        );
        // Without partial_ok, the mid-sequence hit is not offered.
        assert_eq!(tree.longest_prefix(&[1, 2], false, &mut |_, _| true), None);
        // Diverging at a branch point with no matching child: the shared
        // prefix up to the branch is still a partial hit from a child slot.
        let tree = tree_with(&[&[1, 2, 3], &[1, 2, 9]]);
        let with_slots = |node: usize, _hit: usize| tree.nodes[node].slot.is_some();
        assert_eq!(
            tree.longest_prefix(&[1, 2, 7], true, &mut |n, h| with_slots(n, h))
                .map(|(_, h)| h),
            Some(2)
        );
        assert_eq!(
            tree.longest_prefix(&[1, 2, 7], false, &mut |n, h| with_slots(n, h)),
            None
        );
        // Unusable nodes are skipped for a deeper... here no deeper node.
        let tree = tree_with(&[&[1, 2, 3], &[1, 2, 3, 4]]);
        let res = tree.longest_prefix(&[1, 2, 3, 4, 5], false, &mut |node, _| node != 2);
        assert_eq!(res, Some((1, 3)));
    }

    #[test]
    fn insert_attaches_once_and_pin_protects_from_eviction() {
        let mut tree = tree_with(&[&[7, 7, 7]]);
        let (node, attached) = tree.insert(&[7, 7, 7], 42);
        assert!(!attached, "the node already owned a slot");
        assert_eq!(tree.slot(node), Some(1));
        let pin = tree.pin_at(&[7, 7, 7]).expect("node exists");
        assert_eq!(
            tree.eviction_candidate(),
            None,
            "pinned node is not evictable"
        );
        assert!(tree.unpin(pin));
        assert_eq!(tree.eviction_candidate(), Some(node));
        assert!(!tree.unpin(pin), "double unpin refused");
        assert_eq!(
            tree.pin_at(&[8, 8]),
            None,
            "unknown prefix cannot be pinned"
        );
    }

    #[test]
    fn clear_releases_every_slot_and_snapshot_bytes() {
        let mut tree = tree_with(&[&[1], &[1, 2]]);
        tree.attach_snapshot(2, Arc::from(vec![0u8; 100]));
        assert_eq!(tree.snapshot_bytes(), 100);
        let slots = tree.clear();
        assert_eq!(slots.len(), 2);
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.snapshot_bytes(), 0);
    }

    #[test]
    fn remove_prunes_up_to_the_root_and_recycles_indices() {
        let mut tree = tree_with(&[&[1, 2], &[5, 6]]);
        let (node, _) = tree.insert(&[1, 2], 9);
        tree.remove(node);
        // The top-level node is detached from the root, not left dangling.
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.longest_prefix(&[1, 2, 3], true, &mut |_, _| true), None);
        // Its index is reused by the next insertion.
        let (reused, attached) = tree.insert(&[7, 8], 3);
        assert!(attached);
        assert_eq!(reused, node);
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn stats_ratio_is_prefilled_over_prompt() {
        let stats = KvStats {
            prefilled_cells: 10,
            prompt_cells: 100,
            ..Default::default()
        };
        assert!((stats.prefilled_ratio() - 0.1).abs() < 1e-12);
        assert_eq!(KvStats::default().prefilled_ratio(), 0.0);
    }
}
