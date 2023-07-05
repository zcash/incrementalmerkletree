use core::fmt::{self, Debug, Display};
use either::Either;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use tracing::trace;

use incrementalmerkletree::{
    frontier::NonEmptyFrontier, Address, Hashable, Level, MerklePath, Position, Retention,
};

#[cfg(feature = "legacy-api")]
use core::convert::TryFrom;

#[cfg(feature = "legacy-api")]
use incrementalmerkletree::witness::IncrementalWitness;

mod tree;
pub use self::tree::{LocatedTree, Node, Tree};

mod prunable;
pub use self::prunable::{
    IncompleteAt, InsertionError, LocatedPrunableTree, PrunableTree, QueryError, RetentionFlags,
};

pub mod memory;

#[cfg(any(bench, test, feature = "test-dependencies"))]
pub mod testing;

/// An enumeration of possible checkpoint locations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TreeState {
    /// Checkpoints of the empty tree.
    Empty,
    /// Checkpoint at a (possibly pruned) leaf state corresponding to the
    /// wrapped leaf position.
    AtPosition(Position),
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    tree_state: TreeState,
    marks_removed: BTreeSet<Position>,
}

impl Checkpoint {
    pub fn tree_empty() -> Self {
        Checkpoint {
            tree_state: TreeState::Empty,
            marks_removed: BTreeSet::new(),
        }
    }

    pub fn at_position(position: Position) -> Self {
        Checkpoint {
            tree_state: TreeState::AtPosition(position),
            marks_removed: BTreeSet::new(),
        }
    }

    pub fn from_parts(tree_state: TreeState, marks_removed: BTreeSet<Position>) -> Self {
        Checkpoint {
            tree_state,
            marks_removed,
        }
    }

    pub fn tree_state(&self) -> TreeState {
        self.tree_state
    }

    pub fn marks_removed(&self) -> &BTreeSet<Position> {
        &self.marks_removed
    }

    pub fn is_tree_empty(&self) -> bool {
        matches!(self.tree_state, TreeState::Empty)
    }

    pub fn position(&self) -> Option<Position> {
        match self.tree_state {
            TreeState::Empty => None,
            TreeState::AtPosition(pos) => Some(pos),
        }
    }
}

/// A capability for storage of fragment subtrees of the `ShardTree` type.
///
/// All fragment subtrees must have roots at level `SHARD_HEIGHT`
pub trait ShardStore {
    type H;
    type CheckpointId;
    type Error;

    /// Returns the subtree at the given root address, if any such subtree exists.
    fn get_shard(
        &self,
        shard_root: Address,
    ) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error>;

    /// Returns the subtree containing the maximum inserted leaf position.
    fn last_shard(&self) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error>;

    /// Inserts or replaces the subtree having the same root address as the provided tree.
    ///
    /// Implementations of this method MUST enforce the constraint that the root address
    /// of the provided subtree has level `SHARD_HEIGHT`.
    fn put_shard(&mut self, subtree: LocatedPrunableTree<Self::H>) -> Result<(), Self::Error>;

    /// Returns the vector of addresses corresponding to the roots of subtrees stored in this
    /// store.
    fn get_shard_roots(&self) -> Result<Vec<Address>, Self::Error>;

    /// Removes subtrees from the underlying store having root addresses at indices greater
    /// than or equal to that of the specified address.
    ///
    /// Implementations of this method MUST enforce the constraint that the root address
    /// provided has level `SHARD_HEIGHT`.
    fn truncate(&mut self, from: Address) -> Result<(), Self::Error>;

    /// A tree that is used to cache the known roots of subtrees in the "cap" - the top part of the
    /// tree, which contains parent nodes produced by hashing the roots of the individual shards.
    /// Nodes in the cap have levels in the range `SHARD_HEIGHT..DEPTH`. Note that the cap may be
    /// sparse, in the same way that individual shards may be sparse.
    fn get_cap(&self) -> Result<PrunableTree<Self::H>, Self::Error>;

    /// Persists the provided cap to the data store.
    fn put_cap(&mut self, cap: PrunableTree<Self::H>) -> Result<(), Self::Error>;

    /// Returns the identifier for the checkpoint with the lowest associated position value.
    fn min_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error>;

    /// Returns the identifier for the checkpoint with the highest associated position value.
    fn max_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error>;

    /// Adds a checkpoint to the data store.
    fn add_checkpoint(
        &mut self,
        checkpoint_id: Self::CheckpointId,
        checkpoint: Checkpoint,
    ) -> Result<(), Self::Error>;

    /// Returns the number of checkpoints maintained by the data store
    fn checkpoint_count(&self) -> Result<usize, Self::Error>;

    /// Returns the position of the checkpoint, if any, along with the number of subsequent
    /// checkpoints at the same position. Returns `None` if `checkpoint_depth == 0` or if
    /// insufficient checkpoints exist to seek back to the requested depth.
    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<(Self::CheckpointId, Checkpoint)>, Self::Error>;

    /// Returns the checkpoint corresponding to the specified checkpoint identifier.
    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<Checkpoint>, Self::Error>;

    /// Iterates in checkpoint ID order over the first `limit` checkpoints, applying the
    /// given callback to each.
    fn with_checkpoints<F>(&mut self, limit: usize, callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&Self::CheckpointId, &Checkpoint) -> Result<(), Self::Error>;

    /// Update the checkpoint having the given identifier by mutating it with the provided
    /// function, and persist the updated checkpoint to the data store.
    ///
    /// Returns `Ok(true)` if the checkpoint was found, `Ok(false)` if no checkpoint with the
    /// provided identifier exists in the data store, or an error if a storage error occurred.
    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut Checkpoint) -> Result<(), Self::Error>;

    /// Removes a checkpoint from the data store.
    fn remove_checkpoint(&mut self, checkpoint_id: &Self::CheckpointId) -> Result<(), Self::Error>;

    /// Removes checkpoints with identifiers greater than or equal to the given identifier
    fn truncate_checkpoints(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error>;
}

impl<S: ShardStore> ShardStore for &mut S {
    type H = S::H;
    type CheckpointId = S::CheckpointId;
    type Error = S::Error;

    fn get_shard(
        &self,
        shard_root: Address,
    ) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error> {
        S::get_shard(*self, shard_root)
    }

    fn last_shard(&self) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error> {
        S::last_shard(*self)
    }

    fn put_shard(&mut self, subtree: LocatedPrunableTree<Self::H>) -> Result<(), Self::Error> {
        S::put_shard(*self, subtree)
    }

    fn get_shard_roots(&self) -> Result<Vec<Address>, Self::Error> {
        S::get_shard_roots(*self)
    }

    fn get_cap(&self) -> Result<PrunableTree<Self::H>, Self::Error> {
        S::get_cap(*self)
    }

    fn put_cap(&mut self, cap: PrunableTree<Self::H>) -> Result<(), Self::Error> {
        S::put_cap(*self, cap)
    }

    fn truncate(&mut self, from: Address) -> Result<(), Self::Error> {
        S::truncate(*self, from)
    }

    fn min_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error> {
        S::min_checkpoint_id(self)
    }

    fn max_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error> {
        S::max_checkpoint_id(self)
    }

    fn add_checkpoint(
        &mut self,
        checkpoint_id: Self::CheckpointId,
        checkpoint: Checkpoint,
    ) -> Result<(), Self::Error> {
        S::add_checkpoint(self, checkpoint_id, checkpoint)
    }

    fn checkpoint_count(&self) -> Result<usize, Self::Error> {
        S::checkpoint_count(self)
    }

    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<(Self::CheckpointId, Checkpoint)>, Self::Error> {
        S::get_checkpoint_at_depth(self, checkpoint_depth)
    }

    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<Checkpoint>, Self::Error> {
        S::get_checkpoint(self, checkpoint_id)
    }

    fn with_checkpoints<F>(&mut self, limit: usize, callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&Self::CheckpointId, &Checkpoint) -> Result<(), Self::Error>,
    {
        S::with_checkpoints(self, limit, callback)
    }

    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut Checkpoint) -> Result<(), Self::Error>,
    {
        S::update_checkpoint_with(self, checkpoint_id, update)
    }

    fn remove_checkpoint(&mut self, checkpoint_id: &Self::CheckpointId) -> Result<(), Self::Error> {
        S::remove_checkpoint(self, checkpoint_id)
    }

    fn truncate_checkpoints(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        S::truncate_checkpoints(self, checkpoint_id)
    }
}

/// A sparse binary Merkle tree of the specified depth, represented as an ordered collection of
/// subtrees (shards) of a given maximum height.
///
/// This tree maintains a collection of "checkpoints" which represent positions, usually near the
/// front of the tree, that are maintained such that it's possible to truncate nodes to the right
/// of the specified position.
#[derive(Debug)]
pub struct ShardTree<S: ShardStore, const DEPTH: u8, const SHARD_HEIGHT: u8> {
    /// The vector of tree shards.
    store: S,
    /// The maximum number of checkpoints to retain before pruning.
    max_checkpoints: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardTreeError<S> {
    Query(QueryError),
    Insert(InsertionError),
    Storage(S),
}

impl<S> From<QueryError> for ShardTreeError<S> {
    fn from(err: QueryError) -> Self {
        ShardTreeError::Query(err)
    }
}

impl<S> From<InsertionError> for ShardTreeError<S> {
    fn from(err: InsertionError) -> Self {
        ShardTreeError::Insert(err)
    }
}

impl<S: fmt::Display> fmt::Display for ShardTreeError<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            ShardTreeError::Query(q) => Display::fmt(&q, f),
            ShardTreeError::Insert(i) => Display::fmt(&i, f),
            ShardTreeError::Storage(s) => {
                write!(
                    f,
                    "An error occurred persisting or retrieving tree data: {}",
                    s
                )
            }
        }
    }
}

impl<SE> std::error::Error for ShardTreeError<SE>
where
    SE: Debug + std::fmt::Display + std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self {
            ShardTreeError::Storage(e) => Some(e),
            _ => None,
        }
    }
}

impl<
        H: Hashable + Clone + PartialEq,
        C: Clone + Debug + Ord,
        S: ShardStore<H = H, CheckpointId = C>,
        const DEPTH: u8,
        const SHARD_HEIGHT: u8,
    > ShardTree<S, DEPTH, SHARD_HEIGHT>
{
    /// Creates a new empty tree.
    pub fn new(store: S, max_checkpoints: usize) -> Self {
        Self {
            store,
            max_checkpoints,
        }
    }

    /// Returns the root address of the tree.
    pub fn root_addr() -> Address {
        Address::from_parts(Level::from(DEPTH), 0)
    }

    /// Returns the fixed level of subtree roots within the vector of subtrees used as this tree's
    /// representation.
    pub fn subtree_level() -> Level {
        Level::from(SHARD_HEIGHT)
    }

    /// Returns the root address of the subtree that contains the specified position.
    pub fn subtree_addr(pos: Position) -> Address {
        Address::above_position(Self::subtree_level(), pos)
    }

    pub fn max_subtree_index() -> u64 {
        (0x1 << (DEPTH - SHARD_HEIGHT)) - 1
    }

    /// Returns the leaf value at the specified position, if it is a marked leaf.
    pub fn get_marked_leaf(
        &self,
        position: Position,
    ) -> Result<Option<H>, ShardTreeError<S::Error>> {
        Ok(self
            .store
            .get_shard(Self::subtree_addr(position))
            .map_err(ShardTreeError::Storage)?
            .and_then(|t| t.value_at_position(position).cloned())
            .and_then(|(v, r)| if r.is_marked() { Some(v) } else { None }))
    }

    /// Returns the positions of marked leaves in the tree.
    pub fn marked_positions(&self) -> Result<BTreeSet<Position>, ShardTreeError<S::Error>> {
        let mut result = BTreeSet::new();
        for subtree_addr in &self
            .store
            .get_shard_roots()
            .map_err(ShardTreeError::Storage)?
        {
            if let Some(subtree) = self
                .store
                .get_shard(*subtree_addr)
                .map_err(ShardTreeError::Storage)?
            {
                result.append(&mut subtree.marked_positions());
            }
        }
        Ok(result)
    }

    /// Inserts a new root into the tree at the given address.
    ///
    /// The level associated with the given address may not exceed `DEPTH`.
    /// This will return an error if the specified hash conflicts with any existing annotation.
    pub fn insert(&mut self, root_addr: Address, value: H) -> Result<(), ShardTreeError<S::Error>> {
        if root_addr.level() > Self::root_addr().level() {
            return Err(ShardTreeError::Insert(InsertionError::NotContained(
                root_addr,
            )));
        }

        let to_insert = LocatedTree {
            root_addr,
            root: Tree::leaf((value, RetentionFlags::EPHEMERAL)),
        };

        // The cap will retain nodes at the level of the shard roots or higher.
        if root_addr.level() >= Self::subtree_level() {
            let cap = LocatedTree {
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
                root_addr: Self::root_addr(),
            };

            cap.insert_subtree(to_insert.clone(), false)
                .map_err(ShardTreeError::Insert)
                .and_then(|(updated_cap, _)| {
                    self.store
                        .put_cap(updated_cap.root)
                        .map_err(ShardTreeError::Storage)
                })?;
        }

        if let Either::Left(shard_root_addr) = root_addr.context(Self::subtree_level()) {
            let shard = self
                .store
                .get_shard(shard_root_addr)
                .map_err(ShardTreeError::Storage)?
                .unwrap_or_else(|| LocatedTree {
                    root_addr: shard_root_addr,
                    root: Tree::empty(),
                });

            let updated_shard = shard
                .insert_subtree(to_insert, false)
                .map_err(ShardTreeError::Insert)
                .map(|(t, _)| t)?;

            self.store
                .put_shard(updated_shard)
                .map_err(ShardTreeError::Storage)?;
        }

        Ok(())
    }

    /// Append a single value at the first available position in the tree.
    ///
    /// Prefer to use [`Self::batch_insert`] when appending multiple values, as these operations
    /// require fewer traversals of the tree than are necessary when performing multiple sequential
    /// calls to [`Self::append`].
    pub fn append(
        &mut self,
        value: H,
        retention: Retention<C>,
    ) -> Result<(), ShardTreeError<S::Error>> {
        if let Retention::Checkpoint { id, .. } = &retention {
            if self
                .store
                .max_checkpoint_id()
                .map_err(ShardTreeError::Storage)?
                .as_ref()
                >= Some(id)
            {
                return Err(InsertionError::CheckpointOutOfOrder.into());
            }
        }

        let (append_result, position, checkpoint_id) =
            if let Some(subtree) = self.store.last_shard().map_err(ShardTreeError::Storage)? {
                if subtree.root.is_complete() {
                    let addr = subtree.root_addr;

                    if addr.index() < Self::max_subtree_index() {
                        LocatedTree::empty(addr.next_at_level()).append(value, retention)?
                    } else {
                        return Err(InsertionError::TreeFull.into());
                    }
                } else {
                    subtree.append(value, retention)?
                }
            } else {
                let root_addr = Address::from_parts(Self::subtree_level(), 0);
                LocatedTree::empty(root_addr).append(value, retention)?
            };

        self.store
            .put_shard(append_result)
            .map_err(ShardTreeError::Storage)?;
        if let Some(c) = checkpoint_id {
            self.store
                .add_checkpoint(c, Checkpoint::at_position(position))
                .map_err(ShardTreeError::Storage)?;
        }

        self.prune_excess_checkpoints()?;

        Ok(())
    }

    /// Add the leaf and ommers of the provided frontier as nodes within the subtree corresponding
    /// to the frontier's position, and update the cap to include the ommer nodes at levels greater
    /// than or equal to the shard height.
    pub fn insert_frontier_nodes(
        &mut self,
        frontier: NonEmptyFrontier<H>,
        leaf_retention: Retention<C>,
    ) -> Result<(), ShardTreeError<S::Error>> {
        let leaf_position = frontier.position();
        let subtree_root_addr = Address::above_position(Self::subtree_level(), leaf_position);
        trace!("Subtree containing nodes: {:?}", subtree_root_addr);

        let (updated_subtree, supertree) = self
            .store
            .get_shard(subtree_root_addr)
            .map_err(ShardTreeError::Storage)?
            .unwrap_or_else(|| LocatedTree::empty(subtree_root_addr))
            .insert_frontier_nodes(frontier, &leaf_retention)?;

        self.store
            .put_shard(updated_subtree)
            .map_err(ShardTreeError::Storage)?;

        if let Some(supertree) = supertree {
            let new_cap = LocatedTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            }
            .insert_subtree(supertree, leaf_retention.is_marked())?;

            self.store
                .put_cap(new_cap.0.root)
                .map_err(ShardTreeError::Storage)?;
        }

        if let Retention::Checkpoint { id, is_marked: _ } = leaf_retention {
            trace!("Adding checkpoint {:?} at {:?}", id, leaf_position);
            self.store
                .add_checkpoint(id, Checkpoint::at_position(leaf_position))
                .map_err(ShardTreeError::Storage)?;
        }

        self.prune_excess_checkpoints()?;
        Ok(())
    }

    /// Add the leaf and ommers of the provided witness as nodes within the subtree corresponding
    /// to the frontier's position, and update the cap to include the nodes of the witness at
    /// levels greater than or equal to the shard height. Also, if the witness spans multiple
    /// subtrees, update the subtree corresponding to the current witness "tip" accordingly.
    #[cfg(feature = "legacy-api")]
    pub fn insert_witness_nodes(
        &mut self,
        witness: IncrementalWitness<H, DEPTH>,
        checkpoint_id: S::CheckpointId,
    ) -> Result<(), ShardTreeError<S::Error>> {
        let leaf_position = witness.witnessed_position();
        let subtree_root_addr = Address::above_position(Self::subtree_level(), leaf_position);

        let shard = self
            .store
            .get_shard(subtree_root_addr)
            .map_err(ShardTreeError::Storage)?
            .unwrap_or_else(|| LocatedTree::empty(subtree_root_addr));

        let (updated_subtree, supertree, tip_subtree) =
            shard.insert_witness_nodes(witness, checkpoint_id)?;

        self.store
            .put_shard(updated_subtree)
            .map_err(ShardTreeError::Storage)?;

        if let Some(supertree) = supertree {
            let new_cap = LocatedTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            }
            .insert_subtree(supertree, true)?;

            self.store
                .put_cap(new_cap.0.root)
                .map_err(ShardTreeError::Storage)?;
        }

        if let Some(tip_subtree) = tip_subtree {
            let tip_subtree_addr = Address::above_position(
                Self::subtree_level(),
                tip_subtree.root_addr().position_range_start(),
            );

            let tip_shard = self
                .store
                .get_shard(tip_subtree_addr)
                .map_err(ShardTreeError::Storage)?
                .unwrap_or_else(|| LocatedTree::empty(tip_subtree_addr));

            self.store
                .put_shard(tip_shard.insert_subtree(tip_subtree, false)?.0)
                .map_err(ShardTreeError::Storage)?;
        }

        Ok(())
    }

    /// Put a range of values into the subtree to fill leaves starting from the given position.
    ///
    /// This operation will pad the tree until it contains enough subtrees to reach the starting
    /// position. It will fully consume the provided iterator, constructing successive subtrees
    /// until no more values are available. It aggressively prunes the tree as it goes, retaining
    /// only nodes that either have [`Retention::Marked`] retention, are required to construct a
    /// witness for such marked nodes, or that must be retained in order to make it possible to
    /// truncate the tree to any position with [`Retention::Checkpoint`] retention.
    ///
    /// This operation returns the final position at which a leaf was inserted, and the vector of
    /// [`IncompleteAt`] values that identify addresses at which [`Node::Nil`] nodes were
    /// introduced to the tree, as well as whether or not those newly introduced nodes will need to
    /// be filled with values in order to produce witnesses for inserted leaves with
    /// [`Retention::Marked`] retention.
    #[allow(clippy::type_complexity)]
    pub fn batch_insert<I: Iterator<Item = (H, Retention<C>)>>(
        &mut self,
        mut start: Position,
        values: I,
    ) -> Result<Option<(Position, Vec<IncompleteAt>)>, ShardTreeError<S::Error>> {
        let mut values = values.peekable();
        let mut subtree_root_addr = Self::subtree_addr(start);
        let mut max_insert_position = None;
        let mut all_incomplete = vec![];
        loop {
            if values.peek().is_some() {
                let mut res = self
                    .store
                    .get_shard(subtree_root_addr)
                    .map_err(ShardTreeError::Storage)?
                    .unwrap_or_else(|| LocatedTree::empty(subtree_root_addr))
                    .batch_insert(start, values)?
                    .expect(
                        "Iterator containing leaf values to insert was verified to be nonempty.",
                    );
                self.store
                    .put_shard(res.subtree)
                    .map_err(ShardTreeError::Storage)?;
                for (id, position) in res.checkpoints.into_iter() {
                    self.store
                        .add_checkpoint(id, Checkpoint::at_position(position))
                        .map_err(ShardTreeError::Storage)?;
                }

                values = res.remainder;
                subtree_root_addr = subtree_root_addr.next_at_level();
                max_insert_position = res.max_insert_position;
                start = max_insert_position.unwrap() + 1;
                all_incomplete.append(&mut res.incomplete);
            } else {
                break;
            }
        }

        self.prune_excess_checkpoints()?;
        Ok(max_insert_position.map(|p| (p, all_incomplete)))
    }

    /// Insert a tree by decomposing it into its `SHARD_HEIGHT` or smaller parts (if necessary)
    /// and inserting those at their appropriate locations.
    pub fn insert_tree(
        &mut self,
        tree: LocatedPrunableTree<H>,
    ) -> Result<Vec<IncompleteAt>, ShardTreeError<S::Error>> {
        let mut all_incomplete = vec![];
        for subtree in tree.decompose_to_level(Self::subtree_level()).into_iter() {
            let root_addr = subtree.root_addr;
            let contains_marked = subtree.root.contains_marked();
            let (new_subtree, mut incomplete) = self
                .store
                .get_shard(root_addr)
                .map_err(ShardTreeError::Storage)?
                .unwrap_or_else(|| LocatedTree::empty(root_addr))
                .insert_subtree(subtree, contains_marked)?;
            self.store
                .put_shard(new_subtree)
                .map_err(ShardTreeError::Storage)?;
            all_incomplete.append(&mut incomplete);
        }
        Ok(all_incomplete)
    }

    /// Adds a checkpoint at the rightmost leaf state of the tree.
    pub fn checkpoint(&mut self, checkpoint_id: C) -> Result<bool, ShardTreeError<S::Error>> {
        fn go<H: Hashable + Clone + PartialEq>(
            root_addr: Address,
            root: &PrunableTree<H>,
        ) -> Option<(PrunableTree<H>, Position)> {
            match root {
                Tree(Node::Parent { ann, left, right }) => {
                    let (l_addr, r_addr) = root_addr.children().unwrap();
                    go(r_addr, right).map_or_else(
                        || {
                            go(l_addr, left).map(|(new_left, pos)| {
                                (
                                    Tree::unite(
                                        l_addr.level(),
                                        ann.clone(),
                                        new_left,
                                        Tree(Node::Nil),
                                    ),
                                    pos,
                                )
                            })
                        },
                        |(new_right, pos)| {
                            Some((
                                Tree::unite(
                                    l_addr.level(),
                                    ann.clone(),
                                    left.as_ref().clone(),
                                    new_right,
                                ),
                                pos,
                            ))
                        },
                    )
                }
                Tree(Node::Leaf { value: (h, r) }) => Some((
                    Tree(Node::Leaf {
                        value: (h.clone(), *r | RetentionFlags::CHECKPOINT),
                    }),
                    root_addr.max_position(),
                )),
                Tree(Node::Nil) => None,
            }
        }

        // checkpoint identifiers at the tip must be in increasing order
        if self
            .store
            .max_checkpoint_id()
            .map_err(ShardTreeError::Storage)?
            .as_ref()
            >= Some(&checkpoint_id)
        {
            return Ok(false);
        }

        // Update the rightmost subtree to add the `CHECKPOINT` flag to the right-most leaf (which
        // need not be a level-0 leaf; it's fine to rewind to a pruned state).
        if let Some(subtree) = self.store.last_shard().map_err(ShardTreeError::Storage)? {
            if let Some((replacement, pos)) = go(subtree.root_addr, &subtree.root) {
                self.store
                    .put_shard(LocatedTree {
                        root_addr: subtree.root_addr,
                        root: replacement,
                    })
                    .map_err(ShardTreeError::Storage)?;
                self.store
                    .add_checkpoint(checkpoint_id, Checkpoint::at_position(pos))
                    .map_err(ShardTreeError::Storage)?;

                // early return once we've updated the tree state
                self.prune_excess_checkpoints()?;
                return Ok(true);
            }
        }

        self.store
            .add_checkpoint(checkpoint_id, Checkpoint::tree_empty())
            .map_err(ShardTreeError::Storage)?;

        // TODO: it should not be necessary to do this on every checkpoint,
        // but currently that's how the reference tree behaves so we're maintaining
        // those semantics for test compatibility.
        self.prune_excess_checkpoints()?;
        Ok(true)
    }

    fn prune_excess_checkpoints(&mut self) -> Result<(), ShardTreeError<S::Error>> {
        let checkpoint_count = self
            .store
            .checkpoint_count()
            .map_err(ShardTreeError::Storage)?;
        trace!(
            "Tree has {} checkpoints, max is {}",
            checkpoint_count,
            self.max_checkpoints,
        );
        if checkpoint_count > self.max_checkpoints {
            // Batch removals by subtree & create a list of the checkpoint identifiers that
            // will be removed from the checkpoints map.
            let mut checkpoints_to_delete = vec![];
            let mut clear_positions: BTreeMap<Address, BTreeMap<Position, RetentionFlags>> =
                BTreeMap::new();
            self.store
                .with_checkpoints(
                    checkpoint_count - self.max_checkpoints,
                    |cid, checkpoint| {
                        checkpoints_to_delete.push(cid.clone());

                        let mut clear_at = |pos, flags_to_clear| {
                            let subtree_addr = Self::subtree_addr(pos);
                            clear_positions
                                .entry(subtree_addr)
                                .and_modify(|to_clear| {
                                    to_clear
                                        .entry(pos)
                                        .and_modify(|flags| *flags |= flags_to_clear)
                                        .or_insert(flags_to_clear);
                                })
                                .or_insert_with(|| BTreeMap::from([(pos, flags_to_clear)]));
                        };

                        // clear the checkpoint leaf
                        if let TreeState::AtPosition(pos) = checkpoint.tree_state {
                            clear_at(pos, RetentionFlags::CHECKPOINT)
                        }

                        // clear the leaves that have been marked for removal
                        for unmark_pos in checkpoint.marks_removed.iter() {
                            clear_at(*unmark_pos, RetentionFlags::MARKED)
                        }

                        Ok(())
                    },
                )
                .map_err(ShardTreeError::Storage)?;

            trace!(
                "Removing checkpoints {:?}, pruning subtrees {:?}",
                checkpoints_to_delete,
                clear_positions,
            );

            // Prune each affected subtree
            for (subtree_addr, positions) in clear_positions.into_iter() {
                let cleared = self
                    .store
                    .get_shard(subtree_addr)
                    .map_err(ShardTreeError::Storage)?
                    .map(|subtree| subtree.clear_flags(positions));
                if let Some(cleared) = cleared {
                    self.store
                        .put_shard(cleared)
                        .map_err(ShardTreeError::Storage)?;
                }
            }

            // Now that the leaves have been pruned, actually remove the checkpoints
            for c in checkpoints_to_delete {
                self.store
                    .remove_checkpoint(&c)
                    .map_err(ShardTreeError::Storage)?;
            }
        }

        Ok(())
    }

    /// Truncates the tree, discarding all information after the checkpoint at the specified depth.
    ///
    /// This will also discard all checkpoints with depth <= the specified depth. Returns `true`
    /// if the truncation succeeds or has no effect, or `false` if no checkpoint exists at the
    /// specified depth.
    pub fn truncate_to_depth(
        &mut self,
        checkpoint_depth: usize,
    ) -> Result<bool, ShardTreeError<S::Error>> {
        if checkpoint_depth == 0 {
            Ok(true)
        } else if let Some((checkpoint_id, c)) = self
            .store
            .get_checkpoint_at_depth(checkpoint_depth)
            .map_err(ShardTreeError::Storage)?
        {
            self.truncate_removing_checkpoint_internal(&checkpoint_id, &c)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Truncates the tree, discarding all information after the specified checkpoint.
    ///
    /// This will also discard all checkpoints with depth <= the specified depth. Returns `true`
    /// if the truncation succeeds or has no effect, or `false` if no checkpoint exists for the
    /// specified checkpoint identifier.
    pub fn truncate_removing_checkpoint(
        &mut self,
        checkpoint_id: &C,
    ) -> Result<bool, ShardTreeError<S::Error>> {
        if let Some(c) = self
            .store
            .get_checkpoint(checkpoint_id)
            .map_err(ShardTreeError::Storage)?
        {
            self.truncate_removing_checkpoint_internal(checkpoint_id, &c)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn truncate_removing_checkpoint_internal(
        &mut self,
        checkpoint_id: &C,
        checkpoint: &Checkpoint,
    ) -> Result<(), ShardTreeError<S::Error>> {
        match checkpoint.tree_state {
            TreeState::Empty => {
                self.store
                    .truncate(Address::from_parts(Self::subtree_level(), 0))
                    .map_err(ShardTreeError::Storage)?;
                self.store
                    .truncate_checkpoints(checkpoint_id)
                    .map_err(ShardTreeError::Storage)?;
                self.store
                    .put_cap(Tree::empty())
                    .map_err(ShardTreeError::Storage)?;
            }
            TreeState::AtPosition(position) => {
                let subtree_addr = Self::subtree_addr(position);
                let replacement = self
                    .store
                    .get_shard(subtree_addr)
                    .map_err(ShardTreeError::Storage)?
                    .and_then(|s| s.truncate_to_position(position));

                let cap_tree = LocatedTree {
                    root_addr: Self::root_addr(),
                    root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
                };

                if let Some(truncated) = cap_tree.truncate_to_position(position) {
                    self.store
                        .put_cap(truncated.root)
                        .map_err(ShardTreeError::Storage)?;
                };

                if let Some(truncated) = replacement {
                    self.store
                        .truncate(subtree_addr)
                        .map_err(ShardTreeError::Storage)?;
                    self.store
                        .put_shard(truncated)
                        .map_err(ShardTreeError::Storage)?;
                    self.store
                        .truncate_checkpoints(checkpoint_id)
                        .map_err(ShardTreeError::Storage)?;
                }
            }
        }

        Ok(())
    }

    /// Computes the root of any subtree of this tree rooted at the given address, with the overall
    /// tree truncated to the specified position.
    ///
    /// The specified address is not required to be at any particular level, though it cannot
    /// exceed the level corresponding to the maximum depth of the tree. Nodes to the right of the
    /// given position, and parents of such nodes, will be replaced by the empty root for the
    /// associated level.
    ///
    /// Use [`Self::root_at_checkpoint`] to obtain the root of the overall tree.
    pub fn root(
        &self,
        address: Address,
        truncate_at: Position,
    ) -> Result<H, ShardTreeError<S::Error>> {
        assert!(Self::root_addr().contains(&address));

        // traverse the cap from root to leaf depth-first, either returning an existing
        // cached value for the node or inserting the computed value into the cache
        let (root, _) = self.root_internal(
            &LocatedPrunableTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            },
            address,
            truncate_at,
        )?;
        Ok(root)
    }

    pub fn root_caching(
        &mut self,
        address: Address,
        truncate_at: Position,
    ) -> Result<H, ShardTreeError<S::Error>> {
        let (root, updated_cap) = self.root_internal(
            &LocatedPrunableTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            },
            address,
            truncate_at,
        )?;
        if let Some(updated_cap) = updated_cap {
            self.store
                .put_cap(updated_cap)
                .map_err(ShardTreeError::Storage)?;
        }
        Ok(root)
    }

    // compute the root, along with an optional update to the cap
    #[allow(clippy::type_complexity)]
    fn root_internal(
        &self,
        cap: &LocatedPrunableTree<S::H>,
        // The address at which we want to compute the root hash
        target_addr: Address,
        // An inclusive lower bound for positions whose leaf values will be replaced by empty
        // roots.
        truncate_at: Position,
    ) -> Result<(H, Option<PrunableTree<H>>), ShardTreeError<S::Error>> {
        match &cap.root {
            Tree(Node::Parent { ann, left, right }) => {
                match ann {
                    Some(cached_root) if target_addr.contains(&cap.root_addr) => {
                        Ok((cached_root.as_ref().clone(), None))
                    }
                    _ => {
                        // Compute the roots of the left and right children and hash them together.
                        // We skip computation in any subtrees that will not have data included in
                        // the final result.
                        let (l_addr, r_addr) = cap.root_addr.children().unwrap();
                        let l_result = if r_addr.contains(&target_addr) {
                            None
                        } else {
                            Some(self.root_internal(
                                &LocatedPrunableTree {
                                    root_addr: l_addr,
                                    root: left.as_ref().clone(),
                                },
                                if l_addr.contains(&target_addr) {
                                    target_addr
                                } else {
                                    l_addr
                                },
                                truncate_at,
                            )?)
                        };
                        let r_result = if l_addr.contains(&target_addr) {
                            None
                        } else {
                            Some(self.root_internal(
                                &LocatedPrunableTree {
                                    root_addr: r_addr,
                                    root: right.as_ref().clone(),
                                },
                                if r_addr.contains(&target_addr) {
                                    target_addr
                                } else {
                                    r_addr
                                },
                                truncate_at,
                            )?)
                        };

                        // Compute the root value based on the child roots; these may contain the
                        // hashes of empty/truncated nodes.
                        let (root, new_left, new_right) = match (l_result, r_result) {
                            (Some((l_root, new_left)), Some((r_root, new_right))) => (
                                S::H::combine(l_addr.level(), &l_root, &r_root),
                                new_left,
                                new_right,
                            ),
                            (Some((l_root, new_left)), None) => (l_root, new_left, None),
                            (None, Some((r_root, new_right))) => (r_root, None, new_right),
                            (None, None) => unreachable!(),
                        };

                        let new_parent = Tree(Node::Parent {
                            ann: new_left
                                .as_ref()
                                .and_then(|l| l.node_value())
                                .zip(new_right.as_ref().and_then(|r| r.node_value()))
                                .map(|(l, r)| {
                                    // the node values of child nodes cannot contain the hashes of
                                    // empty nodes or nodes with positions greater than the
                                    Rc::new(S::H::combine(l_addr.level(), l, r))
                                }),
                            left: new_left.map_or_else(|| left.clone(), Rc::new),
                            right: new_right.map_or_else(|| right.clone(), Rc::new),
                        });

                        Ok((root, Some(new_parent)))
                    }
                }
            }
            Tree(Node::Leaf { value }) => {
                if truncate_at >= cap.root_addr.position_range_end()
                    && target_addr.contains(&cap.root_addr)
                {
                    // no truncation or computation of child subtrees of this leaf is necessary, just use
                    // the cached leaf value
                    Ok((value.0.clone(), None))
                } else {
                    // since the tree was truncated below this level, recursively call with an
                    // empty parent node to trigger the continued traversal
                    let (root, replacement) = self.root_internal(
                        &LocatedPrunableTree {
                            root_addr: cap.root_addr(),
                            root: Tree::parent(None, Tree::empty(), Tree::empty()),
                        },
                        target_addr,
                        truncate_at,
                    )?;

                    Ok((
                        root,
                        replacement.map(|r| r.reannotate_root(Some(Rc::new(value.0.clone())))),
                    ))
                }
            }
            Tree(Node::Nil) => {
                if cap.root_addr == target_addr
                    || cap.root_addr.level() == ShardTree::<S, DEPTH, SHARD_HEIGHT>::subtree_level()
                {
                    // We are at the leaf level or the target address; compute the root hash and
                    // return it as cacheable if it is not truncated.
                    let root = self.root_from_shards(target_addr, truncate_at)?;
                    Ok((
                        root.clone(),
                        if truncate_at >= cap.root_addr.position_range_end() {
                            // return the compute root as a new leaf to be cached if it contains no
                            // empty hashes due to truncation
                            Some(Tree::leaf((root, RetentionFlags::EPHEMERAL)))
                        } else {
                            None
                        },
                    ))
                } else {
                    // Compute the result by recursively walking down the tree. By replacing
                    // the current node with a parent node, the `Parent` handler will take care
                    // of the branching recursive calls.
                    self.root_internal(
                        &LocatedPrunableTree {
                            root_addr: cap.root_addr,
                            root: Tree::parent(None, Tree::empty(), Tree::empty()),
                        },
                        target_addr,
                        truncate_at,
                    )
                }
            }
        }
    }

    fn root_from_shards(
        &self,
        address: Address,
        truncate_at: Position,
    ) -> Result<H, ShardTreeError<S::Error>> {
        match address.context(Self::subtree_level()) {
            Either::Left(subtree_addr) => {
                // The requested root address is fully contained within one of the subtrees.
                Ok(if truncate_at <= address.position_range_start() {
                    H::empty_root(address.level())
                } else {
                    // get the child of the subtree with its root at `address`
                    self.store
                        .get_shard(subtree_addr)
                        .map_err(ShardTreeError::Storage)?
                        .ok_or_else(|| vec![subtree_addr])
                        .and_then(|subtree| {
                            subtree.subtree(address).map_or_else(
                                || Err(vec![address]),
                                |child| child.root_hash(truncate_at),
                            )
                        })
                        .map_err(QueryError::TreeIncomplete)?
                })
            }
            Either::Right(subtree_range) => {
                // The requested root requires hashing together the roots of several subtrees.
                let mut root_stack = vec![];
                let mut incomplete = vec![];

                for subtree_idx in subtree_range {
                    let subtree_addr = Address::from_parts(Self::subtree_level(), subtree_idx);
                    if truncate_at <= subtree_addr.position_range_start() {
                        break;
                    }

                    let subtree_root = self
                        .store
                        .get_shard(subtree_addr)
                        .map_err(ShardTreeError::Storage)?
                        .ok_or_else(|| vec![subtree_addr])
                        .and_then(|s| s.root_hash(truncate_at));

                    match subtree_root {
                        Ok(mut cur_hash) => {
                            if subtree_addr.index() % 2 == 0 {
                                root_stack.push((subtree_addr, cur_hash))
                            } else {
                                let mut cur_addr = subtree_addr;
                                while let Some((addr, hash)) = root_stack.pop() {
                                    if addr.parent() == cur_addr.parent() {
                                        cur_hash = H::combine(cur_addr.level(), &hash, &cur_hash);
                                        cur_addr = cur_addr.parent();
                                    } else {
                                        root_stack.push((addr, hash));
                                        break;
                                    }
                                }
                                root_stack.push((cur_addr, cur_hash));
                            }
                        }
                        Err(mut new_incomplete) => {
                            // Accumulate incomplete root information and continue, so that we can
                            // return the complete set of incomplete results.
                            incomplete.append(&mut new_incomplete);
                        }
                    }
                }

                if !incomplete.is_empty() {
                    return Err(ShardTreeError::Query(QueryError::TreeIncomplete(
                        incomplete,
                    )));
                }

                // Now hash with empty roots to obtain the root at maximum height
                if let Some((mut cur_addr, mut cur_hash)) = root_stack.pop() {
                    while let Some((addr, hash)) = root_stack.pop() {
                        while addr.level() > cur_addr.level() {
                            cur_hash = H::combine(
                                cur_addr.level(),
                                &cur_hash,
                                &H::empty_root(cur_addr.level()),
                            );
                            cur_addr = cur_addr.parent();
                        }
                        cur_hash = H::combine(cur_addr.level(), &hash, &cur_hash);
                        cur_addr = cur_addr.parent();
                    }

                    while cur_addr.level() < address.level() {
                        cur_hash = H::combine(
                            cur_addr.level(),
                            &cur_hash,
                            &H::empty_root(cur_addr.level()),
                        );
                        cur_addr = cur_addr.parent();
                    }

                    Ok(cur_hash)
                } else {
                    // if the stack is empty, we just return the default root at max height
                    Ok(H::empty_root(address.level()))
                }
            }
        }
    }

    /// Returns the position of the rightmost leaf inserted as of the given checkpoint.
    ///
    /// Returns the maximum leaf position if `checkpoint_depth == 0` (or `Ok(None)` in this
    /// case if the tree is empty) or an error if the checkpointed position cannot be restored
    /// because it has been pruned. Note that no actual level-0 leaf may exist at this position.
    pub fn max_leaf_position(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<Position>, ShardTreeError<S::Error>> {
        Ok(if checkpoint_depth == 0 {
            // TODO: This relies on the invariant that the last shard in the subtrees vector is
            // never created without a leaf then being added to it. However, this may be a
            // difficult invariant to maintain when adding empty roots, so perhaps we need a
            // better way of tracking the actual max position of the tree; we might want to
            // just store it directly.
            self.store
                .last_shard()
                .map_err(ShardTreeError::Storage)?
                .and_then(|t| t.max_position())
        } else {
            match self
                .store
                .get_checkpoint_at_depth(checkpoint_depth)
                .map_err(ShardTreeError::Storage)?
            {
                Some((_, c)) => Ok(c.position()),
                None => {
                    // There is no checkpoint at the specified depth, so we report it as pruned.
                    Err(QueryError::CheckpointPruned)
                }
            }?
        })
    }

    /// Computes the root of the tree as of the checkpointed position at the specified depth.
    ///
    /// Returns the root as of the most recently appended leaf if `checkpoint_depth == 0`. Note
    /// that if the most recently appended leaf is also a checkpoint, this will return the same
    /// result as `checkpoint_depth == 1`.
    pub fn root_at_checkpoint(
        &self,
        checkpoint_depth: usize,
    ) -> Result<H, ShardTreeError<S::Error>> {
        self.max_leaf_position(checkpoint_depth)?.map_or_else(
            || Ok(H::empty_root(Self::root_addr().level())),
            |pos| self.root(Self::root_addr(), pos + 1),
        )
    }

    pub fn root_at_checkpoint_caching(
        &mut self,
        checkpoint_depth: usize,
    ) -> Result<H, ShardTreeError<S::Error>> {
        self.max_leaf_position(checkpoint_depth)?.map_or_else(
            || Ok(H::empty_root(Self::root_addr().level())),
            |pos| self.root_caching(Self::root_addr(), pos + 1),
        )
    }

    /// Computes the witness for the leaf at the specified position.
    ///
    /// Returns the witness as of the most recently appended leaf if `checkpoint_depth == 0`. Note
    /// that if the most recently appended leaf is also a checkpoint, this will return the same
    /// result as `checkpoint_depth == 1`.
    pub fn witness(
        &self,
        position: Position,
        checkpoint_depth: usize,
    ) -> Result<MerklePath<H, DEPTH>, ShardTreeError<S::Error>> {
        let max_leaf_position = self.max_leaf_position(checkpoint_depth).and_then(|v| {
            v.ok_or_else(|| QueryError::TreeIncomplete(vec![Self::root_addr()]).into())
        })?;

        if position > max_leaf_position {
            Err(
                QueryError::NotContained(Address::from_parts(Level::from(0), position.into()))
                    .into(),
            )
        } else {
            let subtree_addr = Self::subtree_addr(position);

            // compute the witness for the specified position up to the subtree root
            let mut witness = self
                .store
                .get_shard(subtree_addr)
                .map_err(ShardTreeError::Storage)?
                .map_or_else(
                    || Err(QueryError::TreeIncomplete(vec![subtree_addr])),
                    |subtree| subtree.witness(position, max_leaf_position + 1),
                )?;

            // compute the remaining parts of the witness up to the root
            let root_addr = Self::root_addr();
            let mut cur_addr = subtree_addr;
            while cur_addr != root_addr {
                witness.push(self.root(cur_addr.sibling(), max_leaf_position + 1)?);
                cur_addr = cur_addr.parent();
            }

            Ok(MerklePath::from_parts(witness, position).unwrap())
        }
    }

    /// Computes the witness for the leaf at the specified position.
    ///
    /// This implementation will mutate the tree to cache intermediate root (ommer) values that are
    /// computed in the process of constructing the witness, so as to avoid the need to recompute
    /// those values from potentially large numbers of subtree roots in the future.
    pub fn witness_caching(
        &mut self,
        position: Position,
        checkpoint_depth: usize,
    ) -> Result<MerklePath<H, DEPTH>, ShardTreeError<S::Error>> {
        let max_leaf_position = self.max_leaf_position(checkpoint_depth).and_then(|v| {
            v.ok_or_else(|| QueryError::TreeIncomplete(vec![Self::root_addr()]).into())
        })?;

        if position > max_leaf_position {
            Err(
                QueryError::NotContained(Address::from_parts(Level::from(0), position.into()))
                    .into(),
            )
        } else {
            let subtree_addr = Address::above_position(Self::subtree_level(), position);

            // compute the witness for the specified position up to the subtree root
            let mut witness = self
                .store
                .get_shard(subtree_addr)
                .map_err(ShardTreeError::Storage)?
                .map_or_else(
                    || Err(QueryError::TreeIncomplete(vec![subtree_addr])),
                    |subtree| subtree.witness(position, max_leaf_position + 1),
                )?;

            // compute the remaining parts of the witness up to the root
            let root_addr = Self::root_addr();
            let mut cur_addr = subtree_addr;
            while cur_addr != root_addr {
                witness.push(self.root_caching(cur_addr.sibling(), max_leaf_position + 1)?);
                cur_addr = cur_addr.parent();
            }

            Ok(MerklePath::from_parts(witness, position).unwrap())
        }
    }

    /// Make a marked leaf at a position eligible to be pruned.
    ///
    /// If the checkpoint associated with the specified identifier does not exist because the
    /// corresponding checkpoint would have been more than `max_checkpoints` deep, the removal is
    /// recorded as of the first existing checkpoint and the associated leaves will be pruned when
    /// that checkpoint is subsequently removed.
    ///
    /// Returns `Ok(true)` if a mark was successfully removed from the leaf at the specified
    /// position, `Ok(false)` if the tree does not contain a leaf at the specified position or is
    /// not marked, or an error if one is produced by the underlying data store.
    pub fn remove_mark(
        &mut self,
        position: Position,
        as_of_checkpoint: Option<&C>,
    ) -> Result<bool, ShardTreeError<S::Error>> {
        match self
            .store
            .get_shard(Self::subtree_addr(position))
            .map_err(ShardTreeError::Storage)?
        {
            Some(shard)
                if shard
                    .value_at_position(position)
                    .iter()
                    .any(|(_, r)| r.is_marked()) =>
            {
                match as_of_checkpoint {
                    Some(cid)
                        if Some(cid)
                            >= self
                                .store
                                .min_checkpoint_id()
                                .map_err(ShardTreeError::Storage)?
                                .as_ref() =>
                    {
                        self.store
                            .update_checkpoint_with(cid, |checkpoint| {
                                checkpoint.marks_removed.insert(position);
                                Ok(())
                            })
                            .map_err(ShardTreeError::Storage)
                    }
                    _ => {
                        // if no checkpoint was provided, or if the checkpoint is too far in the past,
                        // remove the mark directly.
                        self.store
                            .put_shard(
                                shard.clear_flags(BTreeMap::from([(
                                    position,
                                    RetentionFlags::MARKED,
                                )])),
                            )
                            .map_err(ShardTreeError::Storage)?;
                        Ok(true)
                    }
                }
            }
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use proptest::prelude::*;

    use incrementalmerkletree::{
        frontier::NonEmptyFrontier,
        testing::{
            arb_operation, check_append, check_checkpoint_rewind, check_operations,
            check_remove_mark, check_rewind_remove_mark, check_root_hashes,
            check_witness_consistency, check_witnesses, complete_tree::CompleteTree, CombinedTree,
            SipHashable,
        },
        Address, Hashable, Level, Position, Retention,
    };

    use crate::{
        memory::MemoryShardStore,
        testing::{
            arb_char_str, arb_shardtree, check_shard_sizes, check_shardtree_insertion,
            check_witness_with_pruned_subtrees,
        },
        InsertionError, LocatedPrunableTree, RetentionFlags, ShardTree,
    };

    #[cfg(feature = "legacy-api")]
    use incrementalmerkletree::{frontier::CommitmentTree, witness::IncrementalWitness};

    #[cfg(feature = "legacy-api")]
    use crate::Tree;

    #[test]
    fn shardtree_insertion() {
        let tree: ShardTree<MemoryShardStore<String, u32>, 4, 3> =
            ShardTree::new(MemoryShardStore::empty(), 100);

        check_shardtree_insertion(tree)
    }

    #[test]
    fn shard_sizes() {
        let tree: ShardTree<MemoryShardStore<String, u32>, 4, 2> =
            ShardTree::new(MemoryShardStore::empty(), 100);

        check_shard_sizes(tree)
    }

    #[test]
    fn witness_with_pruned_subtrees() {
        let tree: ShardTree<MemoryShardStore<String, u32>, 6, 3> =
            ShardTree::new(MemoryShardStore::empty(), 100);

        check_witness_with_pruned_subtrees(tree)
    }

    fn new_tree(m: usize) -> ShardTree<MemoryShardStore<String, usize>, 4, 3> {
        ShardTree::new(MemoryShardStore::empty(), m)
    }

    #[test]
    fn append() {
        check_append(new_tree);
    }

    #[test]
    fn root_hashes() {
        check_root_hashes(new_tree);
    }

    #[test]
    fn witnesses() {
        check_witnesses(new_tree);
    }

    #[test]
    fn witness_consistency() {
        check_witness_consistency(new_tree);
    }

    #[test]
    fn checkpoint_rewind() {
        check_checkpoint_rewind(new_tree);
    }

    #[test]
    fn remove_mark() {
        check_remove_mark(new_tree);
    }

    #[test]
    fn rewind_remove_mark() {
        check_rewind_remove_mark(new_tree);
    }

    #[test]
    fn checkpoint_pruning_repeated() {
        // Create a tree with some leaves.
        let mut tree = new_tree(10);
        for c in 'a'..='c' {
            tree.append(c.to_string(), Retention::Ephemeral).unwrap();
        }

        // Repeatedly checkpoint the tree at the same position until the checkpoint cache
        // is full (creating a sequence of checkpoints in between which no new leaves were
        // appended to the tree).
        for i in 0..10 {
            assert_eq!(tree.checkpoint(i), Ok(true));
        }

        // Create one more checkpoint at the same position, causing the oldest in the
        // cache to be pruned.
        assert_eq!(tree.checkpoint(10), Ok(true));

        // Append a leaf to the tree and checkpoint it, causing the next oldest in the
        // cache to be pruned.
        assert_eq!(
            tree.append(
                'd'.to_string(),
                Retention::Checkpoint {
                    id: 11,
                    is_marked: false
                },
            ),
            Ok(()),
        );
    }

    // Combined tree tests
    #[allow(clippy::type_complexity)]
    fn new_combined_tree<H: Hashable + Ord + Clone + core::fmt::Debug>(
        max_checkpoints: usize,
    ) -> CombinedTree<
        H,
        usize,
        CompleteTree<H, usize, 4>,
        ShardTree<MemoryShardStore<H, usize>, 4, 3>,
    > {
        CombinedTree::new(
            CompleteTree::new(max_checkpoints),
            ShardTree::new(MemoryShardStore::empty(), max_checkpoints),
        )
    }

    #[test]
    fn combined_append() {
        check_append::<String, usize, _, _>(new_combined_tree);
    }

    #[test]
    fn combined_rewind_remove_mark() {
        check_rewind_remove_mark(new_combined_tree);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100000))]

        #[test]
        fn check_randomized_u64_ops(
            ops in proptest::collection::vec(
                arb_operation(
                    (0..32u64).prop_map(SipHashable),
                    (0u64..100).prop_map(Position::from)
                ),
                1..100
            )
        ) {
            let tree = new_combined_tree(100);
            let indexed_ops = ops.iter().enumerate().map(|(i, op)| op.map_checkpoint_id(|_| i)).collect::<Vec<_>>();
            check_operations(tree, &indexed_ops)?;
        }

        #[test]
        fn check_randomized_str_ops(
            ops in proptest::collection::vec(
                arb_operation(
                    (97u8..123).prop_map(|c| char::from(c).to_string()),
                    (0u64..100).prop_map(Position::from)
                ),
                1..100
            )
        ) {
            let tree = new_combined_tree(100);
            let indexed_ops = ops.iter().enumerate().map(|(i, op)| op.map_checkpoint_id(|_| i)).collect::<Vec<_>>();
            check_operations(tree, &indexed_ops)?;
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn check_shardtree_caching(
            (mut tree, _, marked_positions) in arb_shardtree(arb_char_str())
        ) {
            if let Some(max_leaf_pos) = tree.max_leaf_position(0).unwrap() {
                let max_complete_addr = Address::above_position(max_leaf_pos.root_level(), max_leaf_pos);
                let root = tree.root(max_complete_addr, max_leaf_pos + 1);
                let caching_root = tree.root_caching(max_complete_addr, max_leaf_pos + 1);
                assert_matches!(root, Ok(_));
                assert_eq!(root, caching_root);

                for pos in marked_positions {
                    let witness = tree.witness(pos, 0);
                    let caching_witness = tree.witness_caching(pos, 0);
                    assert_matches!(witness, Ok(_));
                    assert_eq!(witness, caching_witness);
                }
            }
        }
    }

    #[test]
    fn insert_frontier_nodes() {
        let mut frontier = NonEmptyFrontier::new("a".to_string());
        for c in 'b'..'z' {
            frontier.append(c.to_string());
        }

        let root_addr = Address::from_parts(Level::from(4), 1);
        let tree = LocatedPrunableTree::empty(root_addr);
        let result = tree.insert_frontier_nodes::<()>(frontier.clone(), &Retention::Ephemeral);
        assert_matches!(result, Ok(_));

        let mut tree1 = LocatedPrunableTree::empty(root_addr);
        for c in 'q'..'z' {
            let (t, _, _) = tree1
                .append::<()>(c.to_string(), Retention::Ephemeral)
                .unwrap();
            tree1 = t;
        }
        assert_matches!(
            tree1.insert_frontier_nodes::<()>(frontier.clone(), &Retention::Ephemeral),
            Ok(t) if t == result.unwrap()
        );

        let mut tree2 = LocatedPrunableTree::empty(root_addr);
        for c in 'a'..'i' {
            let (t, _, _) = tree2
                .append::<()>(c.to_string(), Retention::Ephemeral)
                .unwrap();
            tree2 = t;
        }
        assert_matches!(
            tree2.insert_frontier_nodes::<()>(frontier, &Retention::Ephemeral),
            Err(InsertionError::Conflict(_))
        );
    }

    #[test]
    fn insert_frontier_nodes_sub_shard_height() {
        let mut frontier = NonEmptyFrontier::new("a".to_string());
        for c in 'b'..='c' {
            frontier.append(c.to_string());
        }

        let root_addr = Address::from_parts(Level::from(3), 0);
        let tree = LocatedPrunableTree::empty(root_addr);
        let result = tree.insert_frontier_nodes::<()>(frontier.clone(), &Retention::Ephemeral);
        assert_matches!(result, Ok((ref _t, None)));

        if let Ok((t, None)) = result {
            // verify that the leaf at the tip is included
            assert_eq!(
                t.root.root_hash(root_addr, Position::from(3)),
                Ok("abc_____".to_string())
            );
        }
    }

    #[test]
    #[cfg(feature = "legacy-api")]
    fn insert_witness_nodes() {
        let mut base_tree = CommitmentTree::<String, 6>::empty();
        for c in 'a'..'h' {
            base_tree.append(c.to_string()).unwrap();
        }
        let mut witness = IncrementalWitness::from_tree(base_tree);
        for c in 'h'..'z' {
            witness.append(c.to_string()).unwrap();
        }

        let root_addr = Address::from_parts(Level::from(3), 0);
        let tree = LocatedPrunableTree::empty(root_addr);
        let result = tree.insert_witness_nodes(witness, 3usize);
        assert_matches!(result, Ok((ref _t, Some(ref _c), Some(ref _r))));

        if let Ok((t, Some(c), Some(r))) = result {
            // verify that we can find the "marked" leaf
            assert_eq!(
                t.root.root_hash(root_addr, Position::from(7)),
                Ok("abcdefg_".to_string())
            );

            assert_eq!(
                c.root,
                Tree::parent(
                    None,
                    Tree::parent(
                        None,
                        Tree::empty(),
                        Tree::leaf(("ijklmnop".to_string(), RetentionFlags::EPHEMERAL)),
                    ),
                    Tree::parent(
                        None,
                        Tree::leaf(("qrstuvwx".to_string(), RetentionFlags::EPHEMERAL)),
                        Tree::empty()
                    )
                )
            );

            assert_eq!(
                r.root
                    .root_hash(Address::from_parts(Level::from(3), 3), Position::from(25)),
                Ok("y_______".to_string())
            );
        }
    }

    #[test]
    #[cfg(feature = "legacy-api")]
    fn insert_witness_nodes_sub_shard_height() {
        let mut base_tree = CommitmentTree::<String, 6>::empty();
        for c in 'a'..='c' {
            base_tree.append(c.to_string()).unwrap();
        }
        let mut witness = IncrementalWitness::from_tree(base_tree);
        witness.append("d".to_string()).unwrap();

        let root_addr = Address::from_parts(Level::from(3), 0);
        let tree = LocatedPrunableTree::empty(root_addr);
        let result = tree.insert_witness_nodes(witness, 3usize);
        assert_matches!(result, Ok((ref _t, None, None)));

        if let Ok((t, None, None)) = result {
            // verify that we can find the "marked" leaf
            assert_eq!(
                t.root.root_hash(root_addr, Position::from(3)),
                Ok("abc_____".to_string())
            );
        }
    }
}
