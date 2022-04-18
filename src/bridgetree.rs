//! A space-efficient implementation of the `Tree` interface.
//!
//! In this module, the term "ommer" is used as a gender-neutral term for
//! the sibling of a parent node in a binary tree.
use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::mem::size_of;

use super::{Altitude, Hashable, Position, Tree};

/// A set of leaves of a Merkle tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Leaf<A> {
    Left(A),
    Right(A, A),
}

impl<A> Leaf<A> {
    pub fn value(&self) -> &A {
        match self {
            Leaf::Left(a) => a,
            Leaf::Right(_, a) => a,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FrontierError {
    PositionMismatch { expected_ommers: usize },
    MaxDepthExceeded { altitude: Altitude },
}

/// A `[NonEmptyFrontier]` is a reduced representation of a Merkle tree,
/// having either one or two leaf values, and then a set of hashes produced
/// by the reduction of previously appended leaf values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NonEmptyFrontier<H> {
    position: Position,
    leaf: Leaf<H>,
    ommers: Vec<H>,
}

impl<H> NonEmptyFrontier<H> {
    /// Constructs a new frontier with the specified value at position 0.
    pub fn new(value: H) -> Self {
        Self {
            position: Position::zero(),
            leaf: Leaf::Left(value),
            ommers: vec![],
        }
    }

    pub fn from_parts(
        position: Position,
        leaf: Leaf<H>,
        ommers: Vec<H>,
    ) -> Result<Self, FrontierError> {
        let expected_ommers = position.ommer_altitudes().count();
        if expected_ommers == ommers.len() {
            Ok(Self {
                position,
                leaf,
                ommers,
            })
        } else {
            Err(FrontierError::PositionMismatch { expected_ommers })
        }
    }

    /// Returns the altitude of the highest ommer in the frontier.
    pub fn max_altitude(&self) -> Altitude {
        self.position.max_altitude()
    }

    /// Returns the position of the most recently appended leaf.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Returns the number of leaves that have been appended to this frontier.
    pub fn size(&self) -> usize {
        <usize>::try_from(self.position)
            .expect("The number of leaves must not exceed the representable range of a `usize`")
            + 1
    }

    pub fn leaf(&self) -> &Leaf<H> {
        &self.leaf
    }

    pub fn ommers(&self) -> &[H] {
        &self.ommers
    }

    /// Returns the value of the most recently appended leaf.
    pub fn leaf_value(&self) -> &H {
        match &self.leaf {
            Leaf::Left(v) | Leaf::Right(_, v) => v,
        }
    }
}

impl<H: Hashable + Clone> NonEmptyFrontier<H> {
    pub fn current_leaf(&self) -> &H {
        self.leaf.value()
    }

    /// of two nodes is full (if the current leaf before the append is a `Leaf::Right`)
    /// then recompute the ommers by hashing together full subtrees until an empty
    /// ommer slot is found.
    pub fn append(&mut self, value: H) {
        let mut carry = None;
        match &self.leaf {
            Leaf::Left(a) => {
                self.leaf = Leaf::Right(a.clone(), value);
            }
            Leaf::Right(a, b) => {
                carry = Some((H::combine(Altitude::zero(), a, b), Altitude::one()));
                self.leaf = Leaf::Left(value);
            }
        };

        if carry.is_some() {
            let mut new_ommers = Vec::with_capacity(self.position.altitudes_required().count());
            for (ommer, ommer_lvl) in self.ommers.iter().zip(self.position.ommer_altitudes()) {
                if let Some((carry_ommer, carry_lvl)) = carry.as_ref() {
                    if *carry_lvl == ommer_lvl {
                        carry = Some((H::combine(ommer_lvl, ommer, carry_ommer), ommer_lvl + 1))
                    } else {
                        // insert the carry at the first empty slot; then the rest of the
                        // ommers will remain unchanged
                        new_ommers.push(carry_ommer.clone());
                        new_ommers.push(ommer.clone());
                        carry = None;
                    }
                } else {
                    // when there's no carry, just push on the ommer value
                    new_ommers.push(ommer.clone());
                }
            }

            // we carried value out, so we need to push on one more ommer.
            if let Some((carry_ommer, _)) = carry {
                new_ommers.push(carry_ommer);
            }

            self.ommers = new_ommers;
        }

        self.position += 1;
    }

    /// Generate the root of the Merkle tree by hashing against empty branches.
    pub fn root(&self) -> H {
        Self::inner_root(self.position, &self.leaf, &self.ommers, None)
    }

    /// If the tree is full to the specified altitude, return the data
    /// required to witness a sibling at that altitude.
    pub fn witness(&self, sibling_altitude: Altitude) -> Option<H> {
        if sibling_altitude == Altitude::zero() {
            match &self.leaf {
                Leaf::Left(_) => None,
                Leaf::Right(_, a) => Some(a.clone()),
            }
        } else if self.position.is_complete(sibling_altitude) {
            // the "incomplete" subtree root is actually complete
            // if the tree is full to this altitude
            Some(Self::inner_root(
                self.position,
                &self.leaf,
                self.ommers.split_last().map_or(&[], |(_, s)| s),
                Some(sibling_altitude),
            ))
        } else {
            None
        }
    }

    /// If the tree is not full, generate the root of the incomplete subtree
    /// by hashing with empty branches
    pub fn witness_incomplete(&self, altitude: Altitude) -> Option<H> {
        if self.position.is_complete(altitude) {
            // if the tree is complete to this altitude, its hash should
            // have already been included in an auth fragment.
            None
        } else {
            Some(if altitude == Altitude::zero() {
                H::empty_leaf()
            } else {
                Self::inner_root(
                    self.position,
                    &self.leaf,
                    self.ommers.split_last().map_or(&[], |(_, s)| s),
                    Some(altitude),
                )
            })
        }
    }

    // returns
    fn inner_root(
        position: Position,
        leaf: &Leaf<H>,
        ommers: &[H],
        result_lvl: Option<Altitude>,
    ) -> H {
        let mut digest = match leaf {
            Leaf::Left(a) => H::combine(Altitude::zero(), a, &H::empty_leaf()),
            Leaf::Right(a, b) => H::combine(Altitude::zero(), a, b),
        };

        let mut complete_lvl = Altitude::one();
        for (ommer, ommer_lvl) in ommers.iter().zip(position.ommer_altitudes()) {
            // stop once we've reached the max altitude
            if result_lvl
                .iter()
                .any(|rl| *rl == complete_lvl || ommer_lvl >= *rl)
            {
                break;
            }

            digest = H::combine(
                ommer_lvl,
                ommer,
                // fold up to ommer.lvl pairing with empty roots; if
                // complete_lvl == ommer.lvl this is just the complete
                // digest to this point
                &complete_lvl
                    .iter_to(ommer_lvl)
                    .fold(digest, |d, l| H::combine(l, &d, &H::empty_root(l))),
            );

            complete_lvl = ommer_lvl + 1;
        }

        // if we've exhausted the ommers and still want more altitudes,
        // continue hashing against empty roots
        digest = complete_lvl
            .iter_to(result_lvl.unwrap_or(complete_lvl))
            .fold(digest, |d, l| H::combine(l, &d, &H::empty_root(l)));

        digest
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
        if f.position.max_altitude().0 <= DEPTH {
            Ok(Frontier { frontier: Some(f) })
        } else {
            Err(FrontierError::MaxDepthExceeded {
                altitude: f.position.max_altitude(),
            })
        }
    }
}

impl<H, const DEPTH: u8> Frontier<H, DEPTH> {
    /// Constructs a new empty frontier.
    pub fn empty() -> Self {
        Self { frontier: None }
    }

    /// Constructs a new non-empty frontier from its constituent parts.
    ///
    /// Returns `None` if the new frontier would exceed the maximum
    /// allowed depth or if the list of ommers provided is not consistent
    /// with the position of the leaf.
    pub fn from_parts(
        position: Position,
        leaf: Leaf<H>,
        ommers: Vec<H>,
    ) -> Result<Self, FrontierError> {
        NonEmptyFrontier::from_parts(position, leaf, ommers).and_then(Self::try_from)
    }

    /// Return the wrapped NonEmptyFrontier reference, or None if
    /// the frontier is empty.
    pub fn value(&self) -> Option<&NonEmptyFrontier<H>> {
        self.frontier.as_ref()
    }

    /// Returns the position of latest leaf appended to the frontier,
    /// if the frontier is nonempty.
    pub fn position(&self) -> Option<Position> {
        self.frontier.as_ref().map(|f| f.position)
    }

    /// Returns the amount of memory dynamically allocated for ommer
    /// values within the frontier.
    pub fn dynamic_memory_usage(&self) -> usize {
        self.frontier.as_ref().map_or(0, |f| {
            2 * size_of::<usize>() + f.ommers.capacity() * size_of::<H>()
        })
    }
}

impl<H: Hashable + Clone, const DEPTH: u8> crate::Frontier<H> for Frontier<H, DEPTH> {
    fn append(&mut self, value: &H) -> bool {
        if let Some(frontier) = self.frontier.as_mut() {
            if frontier.position().is_complete(Altitude(DEPTH)) {
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
            .map_or(H::empty_root(Altitude(DEPTH)), |frontier| {
                // fold from the current height, combining with empty branches,
                // up to the maximum height of the tree
                (frontier.max_altitude() + 1)
                    .iter_to(Altitude(DEPTH))
                    .fold(frontier.root(), |d, lvl| {
                        H::combine(lvl, &d, &H::empty_root(lvl))
                    })
            })
    }
}

/// Each AuthFragment stores part of the authentication path for the leaf at a
/// particular position.  Successive fragments may be concatenated to produce
/// the authentication path up to one less than the maximum altitude of the
/// Merkle frontier corresponding to the leaf at the specified position. Then,
/// the authentication path may be completed by hashing with empty roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthFragment<A> {
    /// The position of the leaf for which this path fragment is being constructed.
    position: Position,
    /// We track the total number of altitudes collected across all fragments
    /// constructed for the specified position separately from the length of
    /// the values vector because the values will usually be split across multiple
    /// fragments.
    altitudes_observed: usize,
    /// The subtree roots at altitudes required for the position that have not
    /// been included in preceding fragments.
    values: Vec<A>,
}

impl<A> AuthFragment<A> {
    /// Construct the new empty authentication path fragment for the specified
    /// position.
    pub fn new(position: Position) -> Self {
        Self {
            position,
            altitudes_observed: 0,
            values: vec![],
        }
    }

    /// Construct a fragment from its component parts. This cannot
    /// perform any meaningful validation that the provided values
    /// are valid.
    pub fn from_parts(position: Position, altitudes_observed: usize, values: Vec<A>) -> Self {
        Self {
            position,
            altitudes_observed,
            values,
        }
    }

    /// Construct the successor fragment for this fragment to produce a new empty fragment
    /// for the specified position.
    #[must_use]
    pub fn successor(&self) -> Self {
        Self {
            position: self.position,
            altitudes_observed: self.altitudes_observed,
            values: vec![],
        }
    }

    pub fn position(&self) -> Position {
        self.position
    }

    pub fn altitudes_observed(&self) -> usize {
        self.altitudes_observed
    }

    pub fn values(&self) -> &[A] {
        &self.values
    }

    pub fn is_complete(&self) -> bool {
        self.altitudes_observed >= self.position.altitudes_required().count()
    }

    pub fn next_required_altitude(&self) -> Option<Altitude> {
        self.position
            .all_altitudes_required()
            .nth(self.altitudes_observed)
    }
}

impl<A: Clone> AuthFragment<A> {
    pub fn fuse(&self, other: &Self) -> Option<Self> {
        if self.position == other.position
            && self.altitudes_observed + other.values.len() == other.altitudes_observed
        {
            Some(Self {
                position: self.position,
                altitudes_observed: other.altitudes_observed,
                values: self
                    .values
                    .iter()
                    .chain(other.values.iter())
                    .cloned()
                    .collect(),
            })
        } else {
            None
        }
    }
}

impl<H: Hashable + Clone + PartialEq> AuthFragment<H> {
    pub fn augment(&mut self, frontier: &NonEmptyFrontier<H>) {
        if let Some(altitude) = self.next_required_altitude() {
            if let Some(digest) = frontier.witness(altitude) {
                self.values.push(digest);
                self.altitudes_observed += 1;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleBridge<H: Ord> {
    /// The position of the final leaf in the frontier of the
    /// bridge that this bridge is the successor of, or None
    /// if this is the first bridge in a tree.
    prior_position: Option<Position>,
    /// Fragments of authorization path data for prior bridges,
    /// keyed by bridge index.
    auth_fragments: BTreeMap<Position, AuthFragment<H>>,
    /// The leading edge of the bridge.
    frontier: NonEmptyFrontier<H>,
}

impl<H: Ord> MerkleBridge<H> {
    /// Construct a new Merkle bridge containing only the specified
    /// leaf.
    pub fn new(value: H) -> Self {
        Self {
            prior_position: None,
            auth_fragments: BTreeMap::new(),
            frontier: NonEmptyFrontier::new(value),
        }
    }

    /// Construct a new Merkle bridge from its constituent parts.
    pub fn from_parts(
        prior_position: Option<Position>,
        auth_fragments: BTreeMap<Position, AuthFragment<H>>,
        frontier: NonEmptyFrontier<H>,
    ) -> Self {
        Self {
            prior_position,
            auth_fragments,
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

    /// Returns the fragments of authorization path data for prior bridges,
    /// keyed by bridge index.
    pub fn auth_fragments(&self) -> &BTreeMap<Position, AuthFragment<H>> {
        &self.auth_fragments
    }

    /// Returns the non-empty frontier of this Merkle bridge.
    pub fn frontier(&self) -> &NonEmptyFrontier<H> {
        &self.frontier
    }

    /// Returns the maximum altitude of this bridge's frontier.
    pub fn max_altitude(&self) -> Altitude {
        self.frontier.max_altitude()
    }

    /// Checks whether this bridge is a valid successor for the specified
    /// bridge.
    pub fn can_follow(&self, prev: &Self) -> bool {
        self.prior_position == Some(prev.frontier.position())
    }
}

impl<'a, H: Hashable + Ord + Clone + 'a> MerkleBridge<H> {
    /// Returns the current leaf.
    pub fn current_leaf(&self) -> &H {
        self.frontier.current_leaf()
    }

    /// Constructs a new bridge to follow this one. If witness_current_leaf is true, the successor
    /// will track the information necessary to create an authentication path for the leaf most
    /// recently appended to this bridge's frontier.
    #[must_use]
    pub fn successor(&self, witness_current_leaf: bool) -> Self {
        let result = Self {
            prior_position: Some(self.frontier.position()),
            auth_fragments: self
                .auth_fragments
                .iter()
                .map(|(k, v)| (*k, v.successor())) //TODO: filter_map and discard what we can
                .chain(if witness_current_leaf {
                    Some((
                        self.frontier.position(),
                        AuthFragment::new(self.frontier.position()),
                    ))
                } else {
                    None
                })
                .collect(),
            frontier: self.frontier.clone(),
        };

        result
    }

    /// Advances this bridge's frontier by appending the specified node,
    /// and updates any auth path fragments being tracked if necessary.
    pub fn append(&mut self, value: H) {
        self.frontier.append(value);

        for ext in self.auth_fragments.values_mut() {
            ext.augment(&self.frontier);
        }
    }

    /// Returns the Merkle root of this bridge's current frontier, as obtained
    /// by hashing against empty nodes.
    pub fn root(&self) -> H {
        self.frontier.root()
    }

    fn root_at_altitude(&self, alt: Altitude) -> H {
        // fold from the current height, combining with empty branches,
        // up to the specified altitude
        (self.max_altitude() + 1)
            .iter_to(alt)
            .fold(self.frontier.root(), |d, lvl| {
                H::combine(lvl, &d, &H::empty_root(lvl))
            })
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
                auth_fragments: self
                    .auth_fragments
                    .iter()
                    .map(|(k, ext)| {
                        // we only need to maintain & augment auth fragments that are in the current
                        // bridge, because we only need to complete the authentication path for the
                        // previous frontier, not the current one.
                        next.auth_fragments
                            .get(k)
                            .map_or((*k, ext.clone()), |next_ext| {
                                (
                                    *k,
                                    ext.fuse(next_ext)
                                        .expect("Found auth fragments at incompatible positions."),
                                )
                            })
                    })
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

    fn prune_auth_fragments(&mut self, to_remove: &BTreeSet<Position>) {
        self.auth_fragments.retain(|k, _| !to_remove.contains(k));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The number of bridges that will be retained in a rewind.
    bridges_len: usize,
    /// A flag indicating whether or not the current state of the tree
    /// had been witnessed at the time the checkpoint was created.
    is_witnessed: bool,
    /// A set of the positions that have been witnessed during the period that this
    /// checkpoint is the current checkpoint.
    witnessed: BTreeSet<Position>,
    /// When a witness is forgotten, if the index of the forgotten witness is <= bridge_idx we
    /// record it in the current checkpoint so that on rollback, we restore the forgotten
    /// witnesses to the BridgeTree's "saved" list. If the witness was newly created since the
    /// checkpoint, we don't need to remember when we forget it because both the witness
    /// creation and removal will be reverted in the rollback.
    forgotten: BTreeMap<Position, usize>,
}

impl Checkpoint {
    /// Creates a new checkpoint from its constituent parts.
    pub fn from_parts(
        bridges_len: usize,
        is_witnessed: bool,
        witnessed: BTreeSet<Position>,
        forgotten: BTreeMap<Position, usize>,
    ) -> Self {
        Self {
            bridges_len,
            is_witnessed,
            witnessed,
            forgotten,
        }
    }

    /// Creates a new empty checkpoint for the specified [`BridgeTree`] state.
    pub fn at_length(bridges_len: usize, is_witnessed: bool) -> Self {
        Checkpoint {
            bridges_len,
            is_witnessed,
            witnessed: BTreeSet::new(),
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

    /// Returns whether the current state of the tree had been witnessed at the point that
    /// this checkpoint was made.
    ///
    /// In the event of a rewind, the rewind logic will ensure that witness information is
    /// properly reconstituted for the checkpointed tree state.
    pub fn is_witnessed(&self) -> bool {
        self.is_witnessed
    }

    /// Returns a set of the positions that have been witnessed during the period that this
    /// checkpoint is the current checkpoint.
    pub fn witnessed(&self) -> &BTreeSet<Position> {
        &self.witnessed
    }

    /// Returns the set of previously-witnessed positions that have had their witnesses removed
    /// during the period that this checkpoint is the current checkpoint.
    pub fn forgotten(&self) -> &BTreeMap<Position, usize> {
        &self.forgotten
    }

    // A private convenience method that returns the root of the bridge corresponding to
    // this checkpoint at a specified depth, given the slice of bridges from which this checkpoint
    // was derived.
    fn root<H: Hashable + Clone + Ord>(
        &self,
        bridges: &[MerkleBridge<H>],
        altitude: Altitude,
    ) -> H {
        if self.bridges_len == 0 {
            H::empty_root(altitude)
        } else {
            bridges[self.bridges_len - 1].root_at_altitude(altitude)
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

    // A private method that rewrites the indices of each forgotten witness record
    // using the specified rewrite function. Used during garbage collection.
    fn rewrite_indices<F: Fn(usize) -> usize>(&mut self, f: F) {
        self.bridges_len = f(self.bridges_len);
        for v in self.forgotten.values_mut() {
            *v = f(*v)
        }
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeTree<H: Ord, const DEPTH: u8> {
    /// The ordered list of Merkle bridges representing the history
    /// of the tree. There will be one bridge for each saved leaf.
    prior_bridges: Vec<MerkleBridge<H>>,
    /// The current (mutable) bridge at the tip of the tree.
    current_bridge: Option<MerkleBridge<H>>,
    /// A map from positions for which we wish to be able to compute an
    /// authentication path to index in the bridges vector.
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
    InvalidWitnessIndex(usize),
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

    pub fn witnessed_indices(&self) -> &BTreeMap<Position, usize> {
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
                return Err(BridgeTreeError::InvalidWitnessIndex(*i));
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
            current_bridge: Some(MerkleBridge::from_parts(None, BTreeMap::new(), frontier)),
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
}

impl<H: Hashable + Ord + Clone + Debug, const DEPTH: u8> Tree<H> for BridgeTree<H, DEPTH> {
    fn append(&mut self, value: &H) -> bool {
        if let Some(bridge) = self.current_bridge.as_mut() {
            if bridge.frontier.position().is_complete(Altitude(DEPTH)) {
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
        let altitude = Altitude(DEPTH);
        if checkpoint_depth == 0 {
            Some(
                self.current_bridge
                    .as_ref()
                    .map_or(H::empty_root(altitude), |bridge| {
                        bridge.root_at_altitude(altitude)
                    }),
            )
        } else if self.checkpoints.len() >= checkpoint_depth {
            let checkpoint_idx = self.checkpoints.len() - checkpoint_depth;
            self.checkpoints
                .get(checkpoint_idx)
                .map(|c| c.root(&self.prior_bridges, altitude))
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

    fn witness(&mut self) -> Option<Position> {
        match self.current_bridge.take() {
            Some(mut cur_b) => {
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
                    // sure that we have an auth fragment tracking the witnessed leaf
                    cur_b
                        .auth_fragments
                        .entry(pos)
                        .or_insert_with(|| AuthFragment::new(pos));
                    self.current_bridge = Some(cur_b);
                } else {
                    let successor = cur_b.successor(true);
                    self.prior_bridges.push(cur_b);
                    self.current_bridge = Some(successor);
                }

                self.saved
                    .entry(pos)
                    .or_insert(self.prior_bridges.len() - 1);

                // mark the position as having been witnessed in the current checkpoint
                if let Some(c) = self.checkpoints.last_mut() {
                    if !c.is_witnessed {
                        c.witnessed.insert(pos);
                    }
                }

                Some(pos)
            }
            None => None,
        }
    }

    fn witnessed_positions(&self) -> BTreeSet<Position> {
        self.saved.keys().cloned().collect()
    }

    fn get_witnessed_leaf(&self, position: Position) -> Option<&H> {
        self.saved
            .get(&position)
            .and_then(|idx| self.prior_bridges.get(*idx).map(|b| b.current_leaf()))
    }

    fn remove_witness(&mut self, position: Position) -> bool {
        if let Some(idx) = self.saved.remove(&position) {
            // Stop tracking auth fragments for the removed position
            if let Some(cur_b) = self.current_bridge.as_mut() {
                cur_b.auth_fragments.remove(&position);
            }

            // If the position is one that has *not* just been witnessed since the last checkpoint,
            // then add it to the set of those forgotten during the current checkpoint span so that
            // it can be restored on rollback.
            if let Some(c) = self.checkpoints.last_mut() {
                if !c.witnessed.contains(&position) {
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
                let is_witnessed = self.get_witnessed_leaf(cur_b.position()).is_some();

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

                self.checkpoints.push(Checkpoint::at_length(
                    self.prior_bridges.len(),
                    is_witnessed,
                ));
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
                // drop witnessed values at and above the checkpoint height;
                // we will re-witness if necessary.
                self.saved.append(&mut c.forgotten);
                self.saved.retain(|_, i| *i + 1 < c.bridges_len);
                self.prior_bridges.truncate(c.bridges_len);
                self.current_bridge = self
                    .prior_bridges
                    .last()
                    .map(|b| b.successor(c.is_witnessed));
                if c.is_witnessed {
                    self.witness();
                }
                true
            }
            None => false,
        }
    }

    fn authentication_path(&self, position: Position, as_of_root: &H) -> Option<Vec<H>> {
        #[derive(Debug)]
        enum AuthBase<'a> {
            Current,
            Checkpoint(usize, &'a Checkpoint),
            NotFound,
        }

        let max_alt = Altitude(DEPTH);

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

        let saved_idx = self.saved.get(&position).or_else(|| {
            if let AuthBase::Checkpoint(i, _) = auth_base {
                // The saved position might have been forgotten since the checkpoint,
                // so look for it in each of the subsequent checkpoints' forgotten
                // items.
                self.checkpoints[i..].iter().find_map(|c| {
                    // restore the forgotten position, if that position was not also witnessed
                    // in the same checkpoint
                    c.forgotten
                        .get(&position)
                        .filter(|_| !c.witnessed.contains(&position))
                })
            } else {
                None
            }
        });

        saved_idx.and_then(|idx| {
            let frontier = &self.prior_bridges[*idx].frontier;

            // Fuse the following bridges to obtain a bridge that has all
            // of the data to the right of the selected value in the tree,
            // up to the specified checkpoint depth.
            let fuse_from = idx + 1;
            let fused = match auth_base {
                AuthBase::Current => MerkleBridge::fuse_all(
                    self.prior_bridges[fuse_from..]
                        .iter()
                        .chain(&self.current_bridge),
                ),
                AuthBase::Checkpoint(_, checkpoint) if fuse_from < checkpoint.bridges_len => {
                    MerkleBridge::fuse_all(
                        self.prior_bridges[fuse_from..checkpoint.bridges_len].iter(),
                    )
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
            };

            fused.map(|successor| {
                // construct a complete trailing edge that includes the data from
                // the following frontier not yet included in the trailing edge.
                let auth_fragment = successor.auth_fragments.get(&frontier.position());
                let rest_frontier = successor.frontier;

                let mut auth_values = auth_fragment.iter().flat_map(|auth_fragment| {
                    let last_altitude = auth_fragment.next_required_altitude();
                    let last_digest =
                        last_altitude.and_then(|lvl| rest_frontier.witness_incomplete(lvl));

                    // TODO: can we eliminate this .cloned()?
                    auth_fragment.values.iter().cloned().chain(last_digest)
                });

                let mut result = vec![];
                match &frontier.leaf {
                    Leaf::Left(_) => {
                        result.push(auth_values.next().unwrap_or_else(H::empty_leaf));
                    }
                    Leaf::Right(a, _) => {
                        result.push(a.clone());
                    }
                }

                for (ommer, ommer_lvl) in frontier
                    .ommers
                    .iter()
                    .zip(frontier.position.ommer_altitudes())
                {
                    for synth_lvl in (result.len() as u8)..(ommer_lvl.into()) {
                        result.push(
                            auth_values
                                .next()
                                .unwrap_or_else(|| H::empty_root(Altitude(synth_lvl))),
                        )
                    }
                    result.push(ommer.clone());
                }

                for synth_lvl in (result.len() as u8)..DEPTH {
                    result.push(
                        auth_values
                            .next()
                            .unwrap_or_else(|| H::empty_root(Altitude(synth_lvl))),
                    );
                }

                result
            })
        })
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
            let mut prune_fragment_positions: BTreeSet<Position> = BTreeSet::new();
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

                        self.prior_bridges.push(cur_bridge);
                        next_bridge
                    } else {
                        // We can fuse these bridges together because we don't need to
                        // remember next_bridge.
                        merged += 1;
                        prune_fragment_positions.insert(cur_bridge.frontier.position());
                        cur_bridge.fuse(&next_bridge).unwrap()
                    };

                    new_cur.prune_auth_fragments(&prune_fragment_positions);
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
            panic!("Consistency check failed with {:?} for tree {:?}", e, self);
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::tests::{apply_operation, arb_operation};
    use crate::Tree;

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
                    tree.authentication_path(*pos, &tree.root(0).unwrap()),
                    tree_mut.authentication_path(*pos, &tree.root(0).unwrap())
                );
            }
        }
    }

    #[test]
    fn bridge_root_hashes() {
        let mut bridge = MerkleBridge::<String>::new("a".to_string());
        assert_eq!(bridge.root(), "a_");

        bridge.append("b".to_string());
        assert_eq!(bridge.root(), "ab");

        bridge.append("c".to_string());
        assert_eq!(bridge.root(), "abc_");
    }

    #[test]
    fn drop_oldest_checkpoint() {
        let mut t = BridgeTree::<String, 6>::new(100);
        t.checkpoint();
        t.append(&"a".to_string());
        t.witness();
        t.append(&"b".to_string());
        t.append(&"c".to_string());
        assert!(
            t.drop_oldest_checkpoint(),
            "Checkpoint drop is expected to succeed"
        );
        assert!(!t.rewind(), "Rewind is expected to fail.");
    }

    #[test]
    fn frontier_from_parts() {
        assert!(
            super::Frontier::<(), 0>::from_parts(Position::zero(), Leaf::Left(()), vec![]).is_ok()
        );
        assert!(super::Frontier::<(), 0>::from_parts(
            Position::zero(),
            Leaf::Right((), ()),
            vec![]
        )
        .is_ok());
        assert!(
            super::Frontier::<(), 0>::from_parts(Position::zero(), Leaf::Left(()), vec![()])
                .is_err()
        );
    }

    #[test]
    fn root_hashes() {
        crate::tests::check_root_hashes(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn auth_paths() {
        crate::tests::check_auth_paths(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn checkpoint_rewind() {
        crate::tests::check_checkpoint_rewind(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn rewind_remove_witness() {
        crate::tests::check_rewind_remove_witness(BridgeTree::<String, 4>::new);
    }

    #[test]
    fn garbage_collect() {
        let mut t = BridgeTree::<String, 7>::new(10);
        let mut to_unwitness = vec![];
        let mut has_auth_path = vec![];
        for i in 0usize..100 {
            let elem: String = format!("{},", i);
            assert!(t.append(&elem), "Append should succeed.");
            if i % 5 == 0 {
                t.checkpoint();
            }
            if i % 7 == 0 {
                t.witness();
                if i > 0 && i % 2 == 0 {
                    to_unwitness.push(Position::from(i));
                } else {
                    has_auth_path.push(Position::from(i));
                }
            }
            if i % 11 == 0 && !to_unwitness.is_empty() {
                let pos = to_unwitness.remove(0);
                t.remove_witness(pos);
            }
        }
        // 32 = 20 (checkpointed) + 14 (witnessed) - 2 (witnessed & checkpointed)
        assert_eq!(t.prior_bridges().len(), 20 + 14 - 2);
        let auth_paths = has_auth_path
            .iter()
            .map(|pos| {
                t.authentication_path(*pos, &t.root(0).unwrap())
                    .expect("Must be able to get auth path")
            })
            .collect::<Vec<_>>();
        t.garbage_collect();
        // 20 = 32 - 10 (removed checkpoints) + 1 (not removed due to witness) - 3 (removed witnesses)
        assert_eq!(t.prior_bridges().len(), 32 - 10 + 1 - 3);
        let retained_auth_paths = has_auth_path
            .iter()
            .map(|pos| {
                t.authentication_path(*pos, &t.root(0).unwrap())
                    .expect("Must be able to get auth path")
            })
            .collect::<Vec<_>>();
        assert_eq!(auth_paths, retained_auth_paths);
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
