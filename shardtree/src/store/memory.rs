//! Implementation of an in-memory shard store with no persistence.

use std::collections::BTreeMap;
use std::convert::Infallible;

use incrementalmerkletree::Address;

use super::{Checkpoint, ShardStore};
use crate::{LocatedPrunableTree, PrunableTree};

/// An implementation of [`ShardStore`] that stores all state in memory.
///
/// State is not persisted anywhere, and will be lost when the struct is dropped.
///
/// Shards are stored sparsely in a [`BTreeMap`] keyed by shard index: only shards explicitly
/// inserted via [`ShardStore::put_shard`] are retained, and an index with no inserted shard reads
/// back as `None` (never as an empty shard). This matches the behaviour of persistent backends such
/// as the SQLite-backed store, and avoids materializing the run of empty shards below a high
/// starting index — e.g. for a wallet synced from a recent birthday, whose first populated shard
/// index is large.
#[derive(Debug)]
pub struct MemoryShardStore<H, C: Ord> {
    shards: BTreeMap<u64, LocatedPrunableTree<H>>,
    checkpoints: BTreeMap<C, Checkpoint>,
    cap: PrunableTree<H>,
}

impl<H, C: Ord> MemoryShardStore<H, C> {
    /// Constructs a new empty `MemoryShardStore`.
    pub fn empty() -> Self {
        Self {
            shards: BTreeMap::new(),
            checkpoints: BTreeMap::new(),
            cap: PrunableTree::empty(),
        }
    }
}

impl<H: Clone, C: Clone + Ord> ShardStore for MemoryShardStore<H, C> {
    type H = H;
    type CheckpointId = C;
    type Error = Infallible;

    fn get_shard(
        &self,
        shard_root: Address,
    ) -> Result<Option<LocatedPrunableTree<H>>, Self::Error> {
        Ok(self.shards.get(&shard_root.index()).cloned())
    }

    fn last_shard(&self) -> Result<Option<LocatedPrunableTree<H>>, Self::Error> {
        Ok(self.shards.values().next_back().cloned())
    }

    fn put_shard(&mut self, subtree: LocatedPrunableTree<H>) -> Result<(), Self::Error> {
        self.shards.insert(subtree.root_addr.index(), subtree);
        Ok(())
    }

    fn get_shard_roots(&self) -> Result<Vec<Address>, Self::Error> {
        Ok(self.shards.values().map(|s| s.root_addr).collect())
    }

    fn truncate_shards(&mut self, shard_index: u64) -> Result<(), Self::Error> {
        // Drop every shard with index >= `shard_index`.
        self.shards.split_off(&shard_index);
        Ok(())
    }

    fn get_cap(&self) -> Result<PrunableTree<H>, Self::Error> {
        Ok(self.cap.clone())
    }

    fn put_cap(&mut self, cap: PrunableTree<H>) -> Result<(), Self::Error> {
        self.cap = cap;
        Ok(())
    }

    fn add_checkpoint(
        &mut self,
        checkpoint_id: C,
        checkpoint: Checkpoint,
    ) -> Result<(), Self::Error> {
        self.checkpoints.insert(checkpoint_id, checkpoint);
        Ok(())
    }

    fn checkpoint_count(&self) -> Result<usize, Self::Error> {
        Ok(self.checkpoints.len())
    }

    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<Checkpoint>, Self::Error> {
        Ok(self.checkpoints.get(checkpoint_id).cloned())
    }

    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<(C, Checkpoint)>, Self::Error> {
        Ok(self
            .checkpoints
            .iter()
            .rev()
            .nth(checkpoint_depth)
            .map(|(id, c)| (id.clone(), c.clone())))
    }

    fn min_checkpoint_id(&self) -> Result<Option<C>, Self::Error> {
        Ok(self.checkpoints.keys().next().cloned())
    }

    fn max_checkpoint_id(&self) -> Result<Option<C>, Self::Error> {
        Ok(self.checkpoints.keys().last().cloned())
    }

    fn with_checkpoints<F>(&mut self, limit: usize, mut callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&C, &Checkpoint) -> Result<(), Self::Error>,
    {
        for (cid, checkpoint) in self.checkpoints.iter().take(limit) {
            callback(cid, checkpoint)?
        }

        Ok(())
    }

    fn for_each_checkpoint<F>(&self, limit: usize, mut callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&C, &Checkpoint) -> Result<(), Self::Error>,
    {
        for (cid, checkpoint) in self.checkpoints.iter().take(limit) {
            callback(cid, checkpoint)?
        }

        Ok(())
    }

    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &C,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut Checkpoint) -> Result<(), Self::Error>,
    {
        if let Some(c) = self.checkpoints.get_mut(checkpoint_id) {
            update(c)?;
            return Ok(true);
        }

        Ok(false)
    }

    fn remove_checkpoint(&mut self, checkpoint_id: &C) -> Result<(), Self::Error> {
        self.checkpoints.remove(checkpoint_id);
        Ok(())
    }

    fn truncate_checkpoints_retaining(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        let mut rest = self.checkpoints.split_off(checkpoint_id);
        if let Some(mut c) = rest.remove(checkpoint_id) {
            c.marks_removed.clear();
            self.checkpoints.insert(checkpoint_id.clone(), c);
        }
        Ok(())
    }
}
