//! A space-efficient implementation of the `Tree` interface.
//!
//! In this module, the term "ommer" is used as a gender-neutral term for
//! the sibling of a parent node in a binary tree.
use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::mem::size_of;
use std::ops::Range;

use super::{Address, Hashable, Level, Position, Source, Tree};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontierError {
    PositionMismatch { expected_ommers: usize },
    MaxDepthExceeded { depth: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathError {
    PositionNotMarked(Position),
    BridgeFusionError,
    FrontierAddressInvalid(Address),
    BridgeAddressInvalid(Address),
}

/// A `[NonEmptyFrontier]` is a reduced representation of a Merkle tree,
/// having either one or two leaf values, and then a set of hashes produced
/// by the reduction of previously appended leaf values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NonEmptyFrontier<H> {
    position: Position,
    leaf: H,
    ommers: Vec<H>,
}

impl<H> NonEmptyFrontier<H> {
    /// Constructs a new frontier with the specified value at position 0.
    pub fn new(leaf: H) -> Self {
        Self {
            position: 0.into(),
            leaf,
            ommers: vec![],
        }
    }

    pub fn from_parts(position: Position, leaf: H, ommers: Vec<H>) -> Result<Self, FrontierError> {
        let expected_ommers = position.past_ommer_count();
        if ommers.len() == expected_ommers {
            Ok(Self {
                position,
                leaf,
                ommers,
            })
        } else {
            Err(FrontierError::PositionMismatch { expected_ommers })
        }
    }

    /// Returns the position of the most recently appended leaf.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Returns the leaf most recently appended to the frontier
    pub fn leaf(&self) -> &H {
        &self.leaf
    }

    /// Returns the list of past hashes required to construct a witness for the
    /// leaf most recently appended to the frontier.
    pub fn ommers(&self) -> &[H] {
        &self.ommers
    }
}

impl<H: Hashable + Clone> NonEmptyFrontier<H> {
    /// Append a new leaf to the frontier, and recompute recompute ommers by hashing together full
    /// subtrees until an empty ommer slot is found.
    pub fn append(&mut self, leaf: H) {
        let prior_position = self.position;
        let prior_leaf = self.leaf.clone();
        self.position += 1;
        self.leaf = leaf;
        if self.position.is_odd() {
            // if the new position is odd, the current leaf will directly become
            // an ommer at level 0, and there is no other mutation made to the tree.
            self.ommers.insert(0, prior_leaf);
        } else {
            // if the new position is even, then the current leaf will be hashed
            // with the first ommer, and so forth up the tree.
            let new_root_level = self.position.root_level();

            let mut carry = Some((prior_leaf, 0.into()));
            let mut new_ommers = Vec::with_capacity(self.position.past_ommer_count());
            for (addr, source) in prior_position.witness_addrs(new_root_level) {
                if let Source::Past(i) = source {
                    if let Some((carry_ommer, carry_lvl)) = carry.as_ref() {
                        if *carry_lvl == addr.level() {
                            carry = Some((
                                H::combine(addr.level(), &self.ommers[i], carry_ommer),
                                addr.level() + 1,
                            ))
                        } else {
                            // insert the carry at the first empty slot; then the rest of the
                            // ommers will remain unchanged
                            new_ommers.push(carry_ommer.clone());
                            new_ommers.push(self.ommers[i].clone());
                            carry = None;
                        }
                    } else {
                        // when there's no carry, just push on the ommer value
                        new_ommers.push(self.ommers[i].clone());
                    }
                }
            }

            // we carried value out, so we need to push on one more ommer.
            if let Some((carry_ommer, _)) = carry {
                new_ommers.push(carry_ommer);
            }

            self.ommers = new_ommers;
        }
    }

    /// Generate the root of the Merkle tree by hashing against empty branches.
    pub fn root(&self, root_level: Option<Level>) -> H {
        let max_level = root_level.unwrap_or_else(|| self.position.root_level());
        self.position
            .witness_addrs(max_level)
            .fold(
                (self.leaf.clone(), Level::from(0)),
                |(digest, complete_lvl), (addr, source)| {
                    // fold up from complete_lvl to addr.level() pairing with empty roots; if
                    // complete_lvl == addr.level() this is just the complete digest to this point
                    let digest = complete_lvl
                        .iter_to(addr.level())
                        .fold(digest, |d, l| H::combine(l, &d, &H::empty_root(l)));

                    let res_digest = match source {
                        Source::Past(i) => H::combine(addr.level(), &self.ommers[i], &digest),
                        Source::Future => {
                            H::combine(addr.level(), &digest, &H::empty_root(addr.level()))
                        }
                    };

                    (res_digest, addr.level + 1)
                },
            )
            .0
    }

    /// Constructs a witness for the leaf at the tip of this
    /// frontier, given a source of node values that complement this frontier.
    pub fn witness<F>(&self, depth: u8, bridge_value_at: F) -> Result<Vec<H>, PathError>
    where
        F: Fn(Address) -> Option<H>,
    {
        // construct a complete trailing edge that includes the data from
        // the following frontier not yet included in the trailing edge.
        self.position()
            .witness_addrs(depth.into())
            .map(|(addr, source)| match source {
                Source::Past(i) => Ok(self.ommers[i].clone()),
                Source::Future => {
                    bridge_value_at(addr).ok_or(PathError::BridgeAddressInvalid(addr))
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

/// A possibly-empty Merkle frontier. Used when the
/// full functionality of a Merkle bridge is not necessary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frontier<H, const DEPTH: u8> {
    frontier: Option<NonEmptyFrontier<H>>,
}

impl<H, const DEPTH: u8> TryFrom<NonEmptyFrontier<H>> for Frontier<H, DEPTH> {
    type Error = FrontierError;
    fn try_from(f: NonEmptyFrontier<H>) -> Result<Self, FrontierError> {
        if f.position.root_level() <= Level::from(DEPTH) {
            Ok(Frontier { frontier: Some(f) })
        } else {
            Err(FrontierError::MaxDepthExceeded {
                depth: f.position.root_level().into(),
            })
        }
    }
}

impl<H, const DEPTH: u8> Frontier<H, DEPTH> {
    /// Constructs a new empty frontier.
    pub fn empty() -> Self {
        Self { frontier: None }
    }

    /// Constructs a new frontier from its constituent parts.
    ///
    /// Returns `None` if the new frontier would exceed the maximum
    /// allowed depth or if the list of ommers provided is not consistent
    /// with the position of the leaf.
    pub fn from_parts(position: Position, leaf: H, ommers: Vec<H>) -> Result<Self, FrontierError> {
        NonEmptyFrontier::from_parts(position, leaf, ommers).and_then(Self::try_from)
    }

    /// Return the wrapped NonEmptyFrontier reference, or None if
    /// the frontier is empty.
    pub fn value(&self) -> Option<&NonEmptyFrontier<H>> {
        self.frontier.as_ref()
    }

    /// Returns the amount of memory dynamically allocated for ommer
    /// values within the frontier.
    pub fn dynamic_memory_usage(&self) -> usize {
        self.frontier.as_ref().map_or(0, |f| {
            size_of::<usize>() + (f.ommers.capacity() + 1) * size_of::<H>()
        })
    }
}

impl<H: Hashable + Clone, const DEPTH: u8> crate::Frontier<H> for Frontier<H, DEPTH> {
    fn append(&mut self, value: &H) -> bool {
        if let Some(frontier) = self.frontier.as_mut() {
            if frontier.position().is_complete_subtree(DEPTH.into()) {
                false
            } else {
                frontier.append(value.clone());
                true
            }
        } else {
            self.frontier = Some(NonEmptyFrontier::new(value.clone()));
            true
        }
    }

    fn root(&self) -> H {
        self.frontier
            .as_ref()
            .map_or(H::empty_root(DEPTH.into()), |frontier| {
                frontier.root(Some(DEPTH.into()))
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleBridge<H> {
    /// The position of the final leaf in the frontier of the bridge that this bridge is the
    /// successor of, or None if this is the first bridge in a tree.
    prior_position: Option<Position>,
    /// The set of addresses for which we are waiting to discover the fragments.  The values of this
    /// set and the keys of the `need` map should always be disjoint. Also, this set should
    /// never contain an address for which the sibling value has been discovered; at that point,
    /// the address is replaced in this set with its parent and the address/sibling pair is stored
    /// in `fragments`.
    ///
    /// Another way to consider the contents of this set is that the values that exist in
    /// `fragments`, combined with the values in previous bridges' `fragments` and an original leaf
    /// node, already contain all the values needed to compute the value at the given address.
    /// Therefore, we are tracking that address as we do not yet have enough information to compute
    /// its sibling without filling the sibling subtree with empty nodes.
    tracking: BTreeSet<Address>,
    /// A map from addresses that were being tracked to the values of their fragments that have been
    /// discovered while scanning this bridge's range by adding leaves to the bridge's frontier.
    fragments: BTreeMap<Address, H>,
    /// The leading edge of the bridge.
    frontier: NonEmptyFrontier<H>,
}

impl<H> MerkleBridge<H> {
    /// Construct a new Merkle bridge containing only the specified
    /// leaf.
    pub fn new(value: H) -> Self {
        Self {
            prior_position: None,
            tracking: BTreeSet::new(),
            fragments: BTreeMap::new(),
            frontier: NonEmptyFrontier::new(value),
        }
    }

    /// Construct a new Merkle bridge from its constituent parts.
    pub fn from_parts(
        prior_position: Option<Position>,
        tracking: BTreeSet<Address>,
        fragments: BTreeMap<Address, H>,
        frontier: NonEmptyFrontier<H>,
    ) -> Self {
        Self {
            prior_position,
            tracking,
            fragments,
            frontier,
        }
    }

    /// Returns the position of the final leaf in the frontier of the
    /// bridge that this bridge is the successor of, or None
    /// if this is the first bridge in a tree.
    pub fn prior_position(&self) -> Option<Position> {
        self.prior_position
    }

    /// Returns the position of the most recently appended leaf.
    pub fn position(&self) -> Position {
        self.frontier.position()
    }

    /// Returns the set of internal node addresses that we're searching
    /// for the fragments for.
    pub fn tracking(&self) -> &BTreeSet<Address> {
        &self.tracking
    }

    /// Returns the set of internal node addresses that we're searching
    /// for the fragments for.
    pub fn fragments(&self) -> &BTreeMap<Address, H> {
        &self.fragments
    }

    /// Returns the non-empty frontier of this Merkle bridge.
    pub fn frontier(&self) -> &NonEmptyFrontier<H> {
        &self.frontier
    }

    /// Returns the value of the most recently appended leaf.
    pub fn current_leaf(&self) -> &H {
        self.frontier.leaf()
    }

    /// Checks whether this bridge is a valid successor for the specified
    /// bridge.
    pub fn can_follow(&self, prev: &Self) -> bool {
        self.prior_position == Some(prev.frontier.position())
    }

    /// Returns the range of positions observed by this bridge.
    pub fn position_range(&self) -> Range<Position> {
        Range {
            start: self.prior_position.unwrap_or(Position(0)),
            end: self.position() + 1,
        }
    }
}

impl<'a, H: Hashable + Ord + Clone + 'a> MerkleBridge<H> {
    /// Constructs a new bridge to follow this one. If mark_current_leaf is true, the successor
    /// will track the information necessary to create a witness for the leaf most
    /// recently appended to this bridge's frontier.
    #[must_use]
    pub fn successor(&self, mark_current_leaf: bool) -> Self {
        let mut result = Self {
            prior_position: Some(self.frontier.position()),
            tracking: self.tracking.clone(),
            fragments: BTreeMap::new(),
            frontier: self.frontier.clone(),
        };

        if mark_current_leaf {
            result.track_current_leaf();
        }

        result
    }

    fn track_current_leaf(&mut self) {
        self.tracking
            .insert(Address::from(self.frontier.position()).current_incomplete());
    }

    /// Advances this bridge's frontier by appending the specified node,
    /// and updates any auth path fragments being tracked if necessary.
    pub fn append(&mut self, value: H) {
        self.frontier.append(value);

        let mut found = vec![];
        for address in self.tracking.iter() {
            // We know that there will only ever be one address that we're
            // tracking at a given level, because as soon as we find a
            // value for the sibling of the address we're tracking, we
            // remove the tracked address and replace it the next parent
            // of that address for which we need to find a sibling.
            if self
                .frontier()
                .position()
                .is_complete_subtree(address.level())
            {
                let digest = self.frontier.root(Some(address.level()));
                self.fragments.insert(address.sibling(), digest);
                found.push(*address);
            }
        }

        for address in found {
            self.tracking.remove(&address);

            // The address of the next incomplete parent note for which
            // we need to find a sibling.
            let parent = address.next_incomplete_parent();
            assert!(!self.fragments.contains_key(&parent));
            self.tracking.insert(parent);
        }
    }

    /// Returns a single MerkleBridge that contains the aggregate information
    /// of this bridge and `next`, or None if `next` is not a valid successor
    /// to this bridge. The resulting Bridge will have the same state as though
    /// `self` had had every leaf used to construct `next` appended to it
    /// directly.
    fn fuse(&self, next: &Self) -> Option<Self> {
        if next.can_follow(self) {
            let fused = Self {
                prior_position: self.prior_position,
                tracking: next.tracking.clone(),
                fragments: self
                    .fragments
                    .iter()
                    .chain(next.fragments.iter())
                    .map(|(k, v)| (*k, v.clone()))
                    .collect(),
                frontier: next.frontier.clone(),
            };

            Some(fused)
        } else {
            None
        }
    }

    /// Returns a single MerkleBridge that contains the aggregate information
    /// of all the provided bridges (discarding internal frontiers) or None
    /// if any of the bridges are not valid successors to one another.
    fn fuse_all<T: Iterator<Item = &'a Self>>(mut iter: T) -> Option<Self> {
        let first = iter.next();
        iter.fold(first.cloned(), |acc, b| acc?.fuse(b))
    }

    /// If this bridge contains sufficient auth fragment information, construct an authentication
    /// path for the specified position by interleaving with values from the prior frontier. This
    /// method will panic if the position of the prior frontier does not match this bridge's prior
    /// position.
    fn witness(
        &self,
        depth: u8,
        prior_frontier: &NonEmptyFrontier<H>,
    ) -> Result<Vec<H>, PathError> {
        assert!(Some(prior_frontier.position()) == self.prior_position);

        prior_frontier.witness(depth, |addr| {
            let r = addr.position_range();
            if self.frontier.position() < r.start {
                Some(H::empty_root(addr.level))
            } else if r.contains(&self.frontier.position()) {
                Some(self.frontier.root(Some(addr.level())))
            } else {
                // the frontier's position is after the end of the requested
                // range, so the requested value should exist in a stored
                // fragment
                self.fragments.get(&addr).cloned()
            }
        })
    }

    fn retain(&mut self, witness_node_addrs: &BTreeSet<Address>) {
        // Prune away any fragments & tracking addresses we don't need
        self.tracking
            .retain(|addr| witness_node_addrs.contains(&addr.sibling()));
        self.fragments
            .retain(|addr, _| witness_node_addrs.contains(addr));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The number of bridges that will be retained in a rewind.
    bridges_len: usize,
    /// A flag indicating whether or not the current state of the tree
    /// had been marked at the time the checkpoint was created.
    is_marked: bool,
    /// A set of the positions that have been marked during the period that this
    /// checkpoint is the current checkpoint.
    marked: BTreeSet<Position>,
    /// When a mark is forgotten, if the index of the forgotten mark is <= bridge_idx we
    /// record it in the current checkpoint so that on rollback, we restore the forgotten
    /// marks to the BridgeTree's "saved" list. If the mark was newly created since the
    /// checkpoint, we don't need to remember when we forget it because both the mark
    /// creation and removal will be reverted in the rollback.
    forgotten: BTreeMap<Position, usize>,
}

impl Checkpoint {
    /// Creates a new checkpoint from its constituent parts.
    pub fn from_parts(
        bridges_len: usize,
        is_marked: bool,
        marked: BTreeSet<Position>,
        forgotten: BTreeMap<Position, usize>,
    ) -> Self {
        Self {
            bridges_len,
            is_marked,
            marked,
            forgotten,
        }
    }

    /// Creates a new empty checkpoint for the specified [`BridgeTree`] state.
    pub fn at_length(bridges_len: usize, is_marked: bool) -> Self {
        Checkpoint {
            bridges_len,
            is_marked,
            marked: BTreeSet::new(),
            forgotten: BTreeMap::new(),
        }
    }

    /// Returns the length of the [`prior_bridges`] vector of the [`BridgeTree`] to which
    /// this checkpoint refers.
    ///
    /// This is the number of bridges that will be retained in the event of a rewind to this
    /// checkpoint.
    pub fn bridges_len(&self) -> usize {
        self.bridges_len
    }

    /// Returns whether the current state of the tree had been marked at the point that
    /// this checkpoint was made.
    ///
    /// In the event of a rewind, the rewind logic will ensure that mark information is
    /// properly reconstituted for the checkpointed tree state.
    pub fn is_marked(&self) -> bool {
        self.is_marked
    }

    /// Returns a set of the positions that have been marked during the period that this
    /// checkpoint is the current checkpoint.
    pub fn marked(&self) -> &BTreeSet<Position> {
        &self.marked
    }

    /// Returns the set of previously-marked positions that have had their marks removed
    /// during the period that this checkpoint is the current checkpoint.
    pub fn forgotten(&self) -> &BTreeMap<Position, usize> {
        &self.forgotten
    }

    // A private convenience method that returns the root of the bridge corresponding to
    // this checkpoint at a specified depth, given the slice of bridges from which this checkpoint
    // was derived.
    fn root<H>(&self, bridges: &[MerkleBridge<H>], level: Level) -> H
    where
        H: Hashable + Clone + Ord,
    {
        if self.bridges_len == 0 {
            H::empty_root(level)
        } else {
            bridges[self.bridges_len - 1].frontier().root(Some(level))
        }
    }

    // A private convenience method that returns the position of the bridge corresponding
    // to this checkpoint, if the checkpoint is not for the empty bridge.
    fn position<H: Ord>(&self, bridges: &[MerkleBridge<H>]) -> Option<Position> {
        if self.bridges_len == 0 {
            None
        } else {
            Some(bridges[self.bridges_len - 1].position())
        }
    }

    // A private method that rewrites the indices of each forgotten marked record
    // using the specified rewrite function. Used during garbage collection.
    fn rewrite_indices<F: Fn(usize) -> usize>(&mut self, f: F) {
        self.bridges_len = f(self.bridges_len);
        for v in self.forgotten.values_mut() {
            *v = f(*v)
        }
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeTree<H, const DEPTH: u8> {
    /// The ordered list of Merkle bridges representing the history
    /// of the tree. There will be one bridge for each saved leaf.
    prior_bridges: Vec<MerkleBridge<H>>,
    /// The current (mutable) bridge at the tip of the tree.
    current_bridge: Option<MerkleBridge<H>>,
    /// A map from positions for which we wish to be able to compute a
    /// witness to index in the bridges vector.
    saved: BTreeMap<Position, usize>,
    /// A stack of bridge indices to which it's possible to rewind directly.
    checkpoints: Vec<Checkpoint>,
    /// The maximum number of checkpoints to retain. If this number is
    /// exceeded, the oldest checkpoint will be dropped when creating
    /// a new checkpoint.
    max_checkpoints: usize,
}

impl<H: Hashable + Ord + Debug, const DEPTH: u8> Debug for BridgeTree<H, DEPTH> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "BridgeTree {{\n  depth: {:?},\n  prior_bridges: {:?},\n  current_bridge: {:?},\n  saved: {:?},\n  checkpoints: {:?},\n  max_checkpoints: {:?}\n}}",
            DEPTH, self.prior_bridges, self.current_bridge, self.saved, self.checkpoints, self.max_checkpoints
        )
    }
}

/// Errors that can appear when validating the internal consistency of a `[MerkleBridge]`
/// value when constructing a bridge from its constituent parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeTreeError {
    IncorrectIncompleteIndex,
    InvalidMarkIndex(usize),
    PositionMismatch { expected: Position, found: Position },
    InvalidSavePoints,
    ContinuityError,
    CheckpointMismatch,
}

impl<H: Ord, const DEPTH: u8> BridgeTree<H, DEPTH> {
    /// Removes the oldest checkpoint. Returns true if successful and false if
    /// there are no checkpoints.
    fn drop_oldest_checkpoint(&mut self) -> bool {
        if self.checkpoints.is_empty() {
            false
        } else {
            self.checkpoints.remove(0);
            true
        }
    }

    /// Returns the prior bridges that make up this tree
    pub fn prior_bridges(&self) -> &[MerkleBridge<H>] {
        &self.prior_bridges
    }

    /// Returns the current bridge at the tip of this tree
    pub fn current_bridge(&self) -> &Option<MerkleBridge<H>> {
        &self.current_bridge
    }

    pub fn marked_indices(&self) -> &BTreeMap<Position, usize> {
        &self.saved
    }

    /// Returns the checkpoints to which this tree may be rewound.
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Returns the maximum number of checkpoints that will be maintained
    /// by the data structure. When this number of checkpoints is exceeded,
    /// the oldest checkpoints are discarded when creating new checkpoints.
    pub fn max_checkpoints(&self) -> usize {
        self.max_checkpoints
    }

    pub fn frontier(&self) -> Option<&NonEmptyFrontier<H>> {
        self.current_bridge.as_ref().map(|b| b.frontier())
    }
}

impl<H: Hashable + Ord + Clone, const DEPTH: u8> BridgeTree<H, DEPTH> {
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            prior_bridges: vec![],
            current_bridge: None,
            saved: BTreeMap::new(),
            checkpoints: vec![],
            max_checkpoints,
        }
    }

    fn check_consistency_internal(
        prior_bridges: &[MerkleBridge<H>],
        current_bridge: &Option<MerkleBridge<H>>,
        saved: &BTreeMap<Position, usize>,
        checkpoints: &[Checkpoint],
        max_checkpoints: usize,
    ) -> Result<(), BridgeTreeError> {
        // check that saved values correspond to bridges
        for (pos, i) in saved {
            if i >= &prior_bridges.len() {
                return Err(BridgeTreeError::InvalidMarkIndex(*i));
            }
            let found = prior_bridges[*i].position();
            if &found != pos {
                return Err(BridgeTreeError::PositionMismatch {
                    expected: *pos,
                    found,
                });
            }
        }

        if checkpoints.len() > max_checkpoints
            || checkpoints
                .iter()
                .any(|c| c.bridges_len > prior_bridges.len())
        {
            return Err(BridgeTreeError::CheckpointMismatch);
        }

        if !prior_bridges
            .iter()
            .zip(prior_bridges.iter().skip(1))
            .all(|(prev, next)| next.can_follow(prev))
        {
            return Err(BridgeTreeError::ContinuityError);
        }

        if !prior_bridges
            .last()
            .zip(current_bridge.as_ref())
            .map_or(true, |(prev, next)| next.can_follow(prev))
        {
            return Err(BridgeTreeError::ContinuityError);
        }

        Ok(())
    }

    /// Construct a new BridgeTree that will start recording changes from the state of
    /// the specified frontier.
    pub fn from_frontier(max_checkpoints: usize, frontier: NonEmptyFrontier<H>) -> Self {
        Self {
            prior_bridges: vec![],
            current_bridge: Some(MerkleBridge::from_parts(
                None,
                BTreeSet::new(),
                BTreeMap::new(),
                frontier,
            )),
            saved: BTreeMap::new(),
            checkpoints: vec![],
            max_checkpoints,
        }
    }

    pub fn from_parts(
        prior_bridges: Vec<MerkleBridge<H>>,
        current_bridge: Option<MerkleBridge<H>>,
        saved: BTreeMap<Position, usize>,
        checkpoints: Vec<Checkpoint>,
        max_checkpoints: usize,
    ) -> Result<Self, BridgeTreeError> {
        Self::check_consistency_internal(
            &prior_bridges,
            &current_bridge,
            &saved,
            &checkpoints,
            max_checkpoints,
        )?;
        Ok(BridgeTree {
            prior_bridges,
            current_bridge,
            saved,
            checkpoints,
            max_checkpoints,
        })
    }

    pub fn check_consistency(&self) -> Result<(), BridgeTreeError> {
        Self::check_consistency_internal(
            &self.prior_bridges,
            &self.current_bridge,
            &self.saved,
            &self.checkpoints,
            self.max_checkpoints,
        )
    }

    fn witness_inner(&self, position: Position, as_of_root: &H) -> Result<Vec<H>, PathError> {
        #[derive(Debug)]
        enum AuthBase<'a> {
            Current,
            Checkpoint(usize, &'a Checkpoint),
            NotFound,
        }

        let max_alt = Level(DEPTH);

        // Find the earliest checkpoint having a matching root, or the current
        // root if it matches and there is no earlier matching checkpoint.
        let auth_base = self
            .checkpoints
            .iter()
            .enumerate()
            .rev()
            .take_while(|(_, c)| c.position(&self.prior_bridges) >= Some(position))
            .filter(|(_, c)| &c.root(&self.prior_bridges, max_alt) == as_of_root)
            .last()
            .map(|(i, c)| AuthBase::Checkpoint(i, c))
            .unwrap_or_else(|| {
                if self.root(0).as_ref() == Some(as_of_root) {
                    AuthBase::Current
                } else {
                    AuthBase::NotFound
                }
            });

        let saved_idx = self
            .saved
            .get(&position)
            .or_else(|| {
                if let AuthBase::Checkpoint(i, _) = auth_base {
                    // The saved position might have been forgotten since the checkpoint,
                    // so look for it in each of the subsequent checkpoints' forgotten
                    // items.
                    self.checkpoints[i..].iter().find_map(|c| {
                        // restore the forgotten position, if that position was not also marked
                        // in the same checkpoint
                        c.forgotten
                            .get(&position)
                            .filter(|_| !c.marked.contains(&position))
                    })
                } else {
                    None
                }
            })
            .ok_or(PathError::PositionNotMarked(position))?;

        let prior_frontier = &self.prior_bridges[*saved_idx].frontier;

        // Fuse the following bridges to obtain a bridge that has all
        // of the data to the right of the selected value in the tree,
        // up to the specified checkpoint depth.
        let fuse_from = saved_idx + 1;
        let successor = match auth_base {
            AuthBase::Current => MerkleBridge::fuse_all(
                self.prior_bridges[fuse_from..]
                    .iter()
                    .chain(&self.current_bridge),
            ),
            AuthBase::Checkpoint(_, checkpoint) if fuse_from < checkpoint.bridges_len => {
                MerkleBridge::fuse_all(self.prior_bridges[fuse_from..checkpoint.bridges_len].iter())
            }
            AuthBase::Checkpoint(_, checkpoint) if fuse_from == checkpoint.bridges_len => {
                // The successor bridge should just be the empty successor to the
                // checkpointed bridge.
                if checkpoint.bridges_len > 0 {
                    Some(self.prior_bridges[checkpoint.bridges_len - 1].successor(false))
                } else {
                    None
                }
            }
            AuthBase::Checkpoint(_, _) => {
                // if the saved index is after the checkpoint, we can't generate
                // an auth path
                None
            }
            AuthBase::NotFound => None,
        }
        .ok_or(PathError::BridgeFusionError)?;

        successor.witness(DEPTH, prior_frontier)
    }
}

impl<H: Hashable + Ord + Clone, const DEPTH: u8> Tree<H> for BridgeTree<H, DEPTH> {
    fn append(&mut self, value: &H) -> bool {
        if let Some(bridge) = self.current_bridge.as_mut() {
            if bridge.frontier.position().is_complete_subtree(Level(DEPTH)) {
                false
            } else {
                bridge.append(value.clone());
                true
            }
        } else {
            self.current_bridge = Some(MerkleBridge::new(value.clone()));
            true
        }
    }

    fn root(&self, checkpoint_depth: usize) -> Option<H> {
        let root_level = Level(DEPTH);
        if checkpoint_depth == 0 {
            Some(
                self.current_bridge
                    .as_ref()
                    .map_or(H::empty_root(root_level), |bridge| {
                        bridge.frontier().root(Some(root_level))
                    }),
            )
        } else if self.checkpoints.len() >= checkpoint_depth {
            let checkpoint_idx = self.checkpoints.len() - checkpoint_depth;
            self.checkpoints
                .get(checkpoint_idx)
                .map(|c| c.root(&self.prior_bridges, root_level))
        } else {
            None
        }
    }

    fn current_position(&self) -> Option<Position> {
        self.current_bridge.as_ref().map(|b| b.position())
    }

    fn current_leaf(&self) -> Option<&H> {
        self.current_bridge.as_ref().map(|b| b.current_leaf())
    }

    fn mark(&mut self) -> Option<Position> {
        match self.current_bridge.take() {
            Some(mut cur_b) => {
                cur_b.track_current_leaf();
                let pos = cur_b.position();
                // If the latest bridge is a newly created checkpoint, the last prior
                // bridge will have the same position and all we need to do is mark
                // the checkpointed leaf as being saved.
                if self
                    .prior_bridges
                    .last()
                    .map_or(false, |prior_b| prior_b.position() == cur_b.position())
                {
                    // the current bridge has not been advanced, so we just need to make
                    // sure that we have are tracking the marked leaf
                    self.current_bridge = Some(cur_b);
                } else {
                    let successor = cur_b.successor(true);
                    self.prior_bridges.push(cur_b);
                    self.current_bridge = Some(successor);
                }

                self.saved
                    .entry(pos)
                    .or_insert(self.prior_bridges.len() - 1);

                // mark the position as having been marked in the current checkpoint
                if let Some(c) = self.checkpoints.last_mut() {
                    if !c.is_marked {
                        c.marked.insert(pos);
                    }
                }

                Some(pos)
            }
            None => None,
        }
    }

    fn marked_positions(&self) -> BTreeSet<Position> {
        self.saved.keys().cloned().collect()
    }

    fn get_marked_leaf(&self, position: Position) -> Option<&H> {
        self.saved
            .get(&position)
            .and_then(|idx| self.prior_bridges.get(*idx).map(|b| b.current_leaf()))
    }

    fn remove_mark(&mut self, position: Position) -> bool {
        if let Some(idx) = self.saved.remove(&position) {
            // If the position is one that has *not* just been marked since the last checkpoint,
            // then add it to the set of those forgotten during the current checkpoint span so that
            // it can be restored on rollback.
            if let Some(c) = self.checkpoints.last_mut() {
                if !c.marked.contains(&position) {
                    c.forgotten.insert(position, idx);
                }
            }
            true
        } else {
            false
        }
    }

    fn checkpoint(&mut self) {
        match self.current_bridge.take() {
            Some(cur_b) => {
                let is_marked = self.get_marked_leaf(cur_b.position()).is_some();

                // Do not create a duplicate bridge
                if self
                    .prior_bridges
                    .last()
                    .map_or(false, |pb| pb.position() == cur_b.position())
                {
                    self.current_bridge = Some(cur_b);
                } else {
                    self.current_bridge = Some(cur_b.successor(false));
                    self.prior_bridges.push(cur_b);
                }

                self.checkpoints
                    .push(Checkpoint::at_length(self.prior_bridges.len(), is_marked));
            }
            None => {
                self.checkpoints.push(Checkpoint::at_length(0, false));
            }
        }

        if self.checkpoints.len() > self.max_checkpoints {
            self.drop_oldest_checkpoint();
        }
    }

    fn rewind(&mut self) -> bool {
        match self.checkpoints.pop() {
            Some(mut c) => {
                // drop marked values at and above the checkpoint height;
                // we will re-mark if necessary.
                self.saved.append(&mut c.forgotten);
                self.saved.retain(|_, i| *i + 1 < c.bridges_len);
                self.prior_bridges.truncate(c.bridges_len);
                self.current_bridge = self.prior_bridges.last().map(|b| b.successor(c.is_marked));
                if c.is_marked {
                    self.mark();
                }
                true
            }
            None => false,
        }
    }

    fn witness(&self, position: Position, as_of_root: &H) -> Option<Vec<H>> {
        self.witness_inner(position, as_of_root).ok()
    }

    fn garbage_collect(&mut self) {
        // Only garbage collect once we have more bridges than the maximum number of
        // checkpoints; we cannot remove information that we might need to restore in
        // a rewind.
        if self.checkpoints.len() == self.max_checkpoints {
            let gc_len = self.checkpoints.first().unwrap().bridges_len;
            // Get a list of the leaf positions that we need to retain. This consists of
            // all the saved leaves, plus all the leaves that have been forgotten since
            // the most distant checkpoint to which we could rewind.
            let remember: BTreeSet<Position> = self
                .saved
                .keys()
                .chain(self.checkpoints.iter().flat_map(|c| c.forgotten.keys()))
                .cloned()
                .collect();

            let mut cur: Option<MerkleBridge<H>> = None;
            let mut merged = 0;
            let mut witness_node_addrs: BTreeSet<Address> = BTreeSet::new();
            for (i, next_bridge) in std::mem::take(&mut self.prior_bridges)
                .into_iter()
                .enumerate()
            {
                if let Some(cur_bridge) = cur {
                    let pos = cur_bridge.position();
                    let mut new_cur = if remember.contains(&pos) || i > gc_len {
                        // We need to remember cur_bridge; update its save index & put next_bridge
                        // on the chopping block
                        if let Some(idx) = self.saved.get_mut(&pos) {
                            *idx -= merged;
                        }

                        // Add the elements of the auth path to the set of addresses we should
                        // continue to track and retain information for
                        for (addr, source) in
                            cur_bridge.frontier.position().witness_addrs(Level(DEPTH))
                        {
                            if source == Source::Future {
                                witness_node_addrs.insert(addr);
                            }
                        }

                        self.prior_bridges.push(cur_bridge);
                        next_bridge
                    } else {
                        // We can fuse these bridges together because we don't need to
                        // remember next_bridge.
                        merged += 1;
                        cur_bridge.fuse(&next_bridge).unwrap()
                    };

                    new_cur.retain(&witness_node_addrs);
                    cur = Some(new_cur);
                } else {
                    // this case will only occur for the first bridge
                    cur = Some(next_bridge);
                }
            }

            // unwrap is safe because we know that prior_bridges was nonempty.
            if let Some(last_bridge) = cur {
                if let Some(idx) = self.saved.get_mut(&last_bridge.position()) {
                    *idx -= merged;
                }
                self.prior_bridges.push(last_bridge);
            }

            for c in self.checkpoints.iter_mut() {
                c.rewrite_indices(|idx| idx - merged);
            }
        }
        if let Err(e) = self.check_consistency() {
            panic!(
                "Consistency check failed after garbage collection with {:?}",
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::tests::{apply_operation, arb_operation};
    use crate::{Frontier, Tree};

    #[test]
    fn nonempty_frontier_root() {
        let mut frontier = NonEmptyFrontier::new("a".to_string());
        assert_eq!(frontier.root(None), "a");

        frontier.append("b".to_string());
        assert_eq!(frontier.root(None), "ab");

        frontier.append("c".to_string());
        assert_eq!(frontier.root(None), "abc_");
    }

    #[test]
    fn frontier_from_parts() {
        assert!(super::Frontier::<(), 1>::from_parts(0.into(), (), vec![]).is_ok());
        assert!(super::Frontier::<(), 1>::from_parts(1.into(), (), vec![()]).is_ok());
        assert!(super::Frontier::<(), 1>::from_parts(0.into(), (), vec![()]).is_err());
    }

    #[test]
    fn frontier_root() {
        let mut frontier: super::Frontier<String, 4> = super::Frontier::empty();
        assert_eq!(frontier.root().len(), 16);
        assert_eq!(frontier.root(), "________________");

        frontier.append(&"a".to_string());
        assert_eq!(frontier.root(), "a_______________");

        frontier.append(&"b".to_string());
        assert_eq!(frontier.root(), "ab______________");

        frontier.append(&"c".to_string());
        assert_eq!(frontier.root(), "abc_____________");
    }

    #[test]
    fn frontier_witness() {
        let mut frontier = NonEmptyFrontier::<String>::new("a".to_string());
        for c in 'b'..'h' {
            frontier.append(c.to_string());
        }
        let bridge_value_at = |addr: Address| match <u8>::from(addr.level()) {
            0 => Some("h".to_string()),
            3 => Some("xxxxxxxx".to_string()),
            _ => None,
        };

        assert_eq!(
            Ok(["h", "ef", "abcd", "xxxxxxxx"]
                .map(|v| v.to_string())
                .to_vec()),
            frontier.witness(4, bridge_value_at)
        );
    }

    #[test]
    fn tree_depth() {
        let mut tree = BridgeTree::<String, 3>::new(100);
        for c in 'a'..'i' {
            assert!(tree.append(&c.to_string()))
        }
        assert!(!tree.append(&'i'.to_string()));
    }

    fn arb_bridgetree<G: Strategy + Clone>(
        item_gen: G,
        max_count: usize,
    ) -> impl Strategy<Value = BridgeTree<G::Value, 8>>
    where
        G::Value: Hashable + Ord + Clone + Debug + 'static,
    {
        proptest::collection::vec(arb_operation(item_gen, 0..max_count), 0..max_count).prop_map(
            |ops| {
                let mut tree: BridgeTree<G::Value, 8> = BridgeTree::new(10);
                for op in ops {
                    apply_operation(&mut tree, op);
                }
                tree
            },
        )
    }

    proptest! {
        #[test]
        fn bridgetree_from_parts(
            tree in arb_bridgetree((97u8..123).prop_map(|c| char::from(c).to_string()), 100)
        ) {
            assert_eq!(
                BridgeTree::from_parts(
                    tree.prior_bridges.clone(),
                    tree.current_bridge.clone(),
                    tree.saved.clone(),
                    tree.checkpoints.clone(),
                    tree.max_checkpoints
                ),
                Ok(tree),
            );
        }

        #[test]
        fn prop_garbage_collect(
            tree in arb_bridgetree((97u8..123).prop_map(|c| char::from(c).to_string()), 100)
        ) {
            let mut tree_mut = tree.clone();
            // ensure we have enough checkpoints to not rewind past the state `tree` is in
            for _ in 0..10 {
                tree_mut.checkpoint();
            }

            tree_mut.garbage_collect();

            tree_mut.rewind();

            for pos in tree.saved.keys() {
                assert_eq!(
                    tree.witness(*pos, &tree.root(0).unwrap()),
                    tree_mut.witness(*pos, &tree.root(0).unwrap())
                );
            }
        }
    }

    #[test]
    fn drop_oldest_checkpoint() {
        let mut t = BridgeTree::<String, 6>::new(100);
        t.checkpoint();
        t.append(&"a".to_string());
        t.mark();
        t.append(&"b".to_string());
        t.append(&"c".to_string());
        assert!(
            t.drop_oldest_checkpoint(),
            "Checkpoint drop is expected to succeed"
        );
        assert!(!t.rewind(), "Rewind is expected to fail.");
    }

    #[test]
    fn root_hashes() {
        crate::tests::check_root_hashes(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn witnesss() {
        crate::tests::check_witnesss(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn checkpoint_rewind() {
        crate::tests::check_checkpoint_rewind(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn rewind_remove_mark() {
        crate::tests::check_rewind_remove_mark(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn garbage_collect() {
        let mut t = BridgeTree::<String, 7>::new(10);
        let mut to_unmark = vec![];
        let mut has_witness = vec![];
        for i in 0usize..100 {
            let elem: String = format!("{},", i);
            assert!(t.append(&elem), "Append should succeed.");
            if i % 5 == 0 {
                t.checkpoint();
            }
            if i % 7 == 0 {
                t.mark();
                if i > 0 && i % 2 == 0 {
                    to_unmark.push(Position::from(i));
                } else {
                    has_witness.push(Position::from(i));
                }
            }
            if i % 11 == 0 && !to_unmark.is_empty() {
                let pos = to_unmark.remove(0);
                t.remove_mark(pos);
            }
        }
        // 32 = 20 (checkpointed) + 14 (marked) - 2 (marked & checkpointed)
        assert_eq!(t.prior_bridges().len(), 20 + 14 - 2);
        let witnesss = has_witness
            .iter()
            .map(|pos| match t.witness_inner(*pos, &t.root(0).unwrap()) {
                Ok(path) => path,
                Err(e) => panic!("Failed to get auth path: {:?}", e),
            })
            .collect::<Vec<_>>();
        t.garbage_collect();
        // 20 = 32 - 10 (removed checkpoints) + 1 (not removed due to mark) - 3 (removed marks)
        assert_eq!(t.prior_bridges().len(), 32 - 10 + 1 - 3);
        let retained_witnesss = has_witness
            .iter()
            .map(|pos| {
                t.witness(*pos, &t.root(0).unwrap())
                    .expect("Must be able to get auth path")
            })
            .collect::<Vec<_>>();
        assert_eq!(witnesss, retained_witnesss);
    }

    #[test]
    fn garbage_collect_idx() {
        let mut tree: BridgeTree<String, 7> = BridgeTree::new(100);
        let empty_root = tree.root(0);
        tree.append(&"a".to_string());
        for _ in 0..100 {
            tree.checkpoint();
        }
        tree.garbage_collect();
        assert!(tree.root(0) != empty_root);
        tree.rewind();
        assert!(tree.root(0) != empty_root);
    }
}
