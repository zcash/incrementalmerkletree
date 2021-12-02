//! A space-efficient implementation of the `Tree` interface.
//!
//! In this module, the term "ommer" is used as a gender-neutral term for
//! the sibling of a parent node in a binary tree.
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem::size_of;

use super::{Altitude, Hashable, Position, Recording, Tree};

/// A set of leaves of a Merkle tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Leaf<A> {
    Left(A),
    Right(A, A),
}

impl<A> Leaf<A> {
    pub fn into_value(self) -> A {
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
        NonEmptyFrontier {
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
            Ok(NonEmptyFrontier {
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
}

impl<H: Clone> NonEmptyFrontier<H> {
    /// Returns the value of the most recently appended leaf.
    pub fn leaf_value(&self) -> &H {
        match &self.leaf {
            Leaf::Left(v) | Leaf::Right(_, v) => v,
        }
    }
}

impl<H: Hashable + Clone> NonEmptyFrontier<H> {
    /// Appends a new leaf value to the Merkle frontier. If the current leaf subtree
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
                carry = Some((H::combine(Altitude::zero(), &a, &b), Altitude::one()));
                self.leaf = Leaf::Left(value);
            }
        };

        if carry.is_some() {
            let mut new_ommers = Vec::with_capacity(self.position.altitudes_required().count());
            for (ommer, ommer_lvl) in self.ommers.iter().zip(self.position.ommer_altitudes()) {
                if let Some((carry_ommer, carry_lvl)) = carry.as_ref() {
                    if *carry_lvl == ommer_lvl {
                        carry = Some((H::combine(ommer_lvl, &ommer, &carry_ommer), ommer_lvl + 1))
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

        self.position.increment()
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
                &ommer,
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
        Frontier { frontier: None }
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
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the frontier would exceed the maximum
    /// allowed depth.
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

    /// Obtains the current root of this Merkle frontier.
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

/// Each AuthFragment stores part of the authentication path for the leaf at a particular position.
/// Successive fragments may be concatenated to produce the authentication path up to one less than
/// the maximum altitude of the Merkle frontier corresponding to the leaf at the specified
/// position. Then, the authentication path may be completed by hashing with empty roots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthFragment<A> {
    /// The position of the leaf for which this path fragment is being constructed.
    position: Position,
    /// We track the total number of altitudes collected across all fragments constructed for
    /// the specified position separately from the length of the values vector because the values
    /// will usually be split across multiple fragments.
    altitudes_observed: usize,
    /// The subtree roots at altitudes required for the position that have not been included in
    /// preceding fragments.
    values: Vec<A>,
}

impl<A> AuthFragment<A> {
    /// Construct the new empty authentication path fragment for the specified position.
    pub fn new(position: Position) -> Self {
        AuthFragment {
            position,
            altitudes_observed: 0,
            values: vec![],
        }
    }

    /// Construct a fragment from its component parts. This cannot
    /// perform any meaningful validation that the provided values
    /// are valid.
    pub fn from_parts(position: Position, altitudes_observed: usize, values: Vec<A>) -> Self {
        assert!(altitudes_observed <= values.len());
        AuthFragment {
            position,
            altitudes_observed,
            values,
        }
    }

    /// Construct the successor fragment for this fragment to produce a new empty fragment
    /// for the specified position.
    pub fn successor(&self) -> Self {
        AuthFragment {
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
            Some(AuthFragment {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleBridge<H> {
    /// The position of the final leaf in the frontier of the
    /// bridge that this bridge is the successor of, or None
    /// if this is the first bridge in a tree.
    prior_position: Option<Position>,
    /// Fragments of authorization path data for prior bridges,
    /// keyed by bridge index.
    auth_fragments: HashMap<usize, AuthFragment<H>>,
    frontier: NonEmptyFrontier<H>,
}

impl<H> MerkleBridge<H> {
    /// Construct a new Merkle bridge containing only the specified
    /// leaf.
    pub fn new(value: H) -> Self {
        MerkleBridge {
            prior_position: None,
            auth_fragments: HashMap::new(),
            frontier: NonEmptyFrontier::new(value),
        }
    }

    /// Construct a new Merkle bridge from its constituent parts.
    pub fn from_parts(
        prior_position: Option<Position>,
        auth_fragments: HashMap<usize, AuthFragment<H>>,
        frontier: NonEmptyFrontier<H>,
    ) -> Self {
        MerkleBridge {
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

    /// Returns the fragments of authorization path data for prior bridges,
    /// keyed by bridge index.
    pub fn auth_fragments(&self) -> &HashMap<usize, AuthFragment<H>> {
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
        self.prior_position
            .iter()
            .all(|p| *p == prev.frontier.position())
    }
}

impl<H: Hashable + Clone + PartialEq> MerkleBridge<H> {
    /// Constructs a new bridge to follow this one. The
    /// successor will track the information necessary to create an
    /// authentication path for the leaf most recently appended to
    /// this bridge's frontier.
    pub fn successor(&self, cur_idx: usize) -> Self {
        let result = MerkleBridge {
            prior_position: Some(self.frontier.position()),
            auth_fragments: self
                .auth_fragments
                .iter()
                .map(|(k, v)| (*k, v.successor())) //TODO: filter_map and discard what we can
                .chain(Some((cur_idx, AuthFragment::new(self.frontier.position()))))
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

    /// Returns the most recently appended leaf.
    pub fn leaf_value(&self) -> &H {
        self.frontier.leaf_value()
    }

    /// Returns a single MerkleBridge that contains the aggregate information
    /// of this bridge and `next`, or None if `next` is not a valid successor
    /// to this bridge. The resulting Bridge will have the same state as though
    /// `self` had had every leaf used to construct `next` appended to it
    /// directly.
    fn fuse(&self, next: &Self) -> Option<MerkleBridge<H>> {
        if next.can_follow(&self) {
            let fused = MerkleBridge {
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
    fn fuse_all(bridges: &[MerkleBridge<H>]) -> Option<MerkleBridge<H>> {
        let mut iter = bridges.iter();
        let first = iter.next();
        iter.fold(first.cloned(), |acc, b| acc?.fuse(b))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Checkpoint {
    /// Checkpoint of the empty bridge
    Empty,
    /// Checkpoint of a particular bridge state to which it is
    /// possible to rewind.
    AtIndex(usize),
}

impl Checkpoint {
    pub fn is_empty(&self) -> bool {
        matches!(&self, Checkpoint::Empty)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BridgeTree<H: Hash + Eq, const DEPTH: u8> {
    /// The ordered list of Merkle bridges representing the history
    /// of the tree. There will be one bridge for each saved leaf, plus
    /// the current bridge to the tip of the tree.
    bridges: Vec<MerkleBridge<H>>,
    /// An index from leaf digests to indices within the `bridges` vector.
    saved: HashMap<H, usize>,
    /// A stack of bridge indices to which it's possible to rewind directly.
    checkpoints: Vec<Checkpoint>,
    /// The maximum number of checkpoints to retain. If this number is
    /// exceeded, the oldest checkpoint will be dropped when creating
    /// a new checkpoint.
    max_checkpoints: usize,
}

impl<H: Hashable + Hash + Eq + Debug, const DEPTH: u8> Debug for BridgeTree<H, DEPTH> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "BridgeTree {{\n  depth: {:?},\n  bridges: {:?},\n saved: {:?},\n  checkpoints: {:?},\n  max_checkpoints: {:?}\n}}",
            DEPTH, self.bridges, self.saved, self.checkpoints, self.max_checkpoints
        )
    }
}

/// An internal-only enum used to indicate whether the `witness_internal`
/// method is being called in the context of marking a leaf as one for
/// which it needs to be able to construct a witness, or whether that
/// leaf is merely being marked as a checkpoint to which it is possible
/// to rewind.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum BoundaryType {
    Witness,
    Checkpoint,
}

/// Errors that can appear when validating the internal consistency of a `[MerkleBridge]`
/// value when constructing a bridge from its constituent parts.
#[derive(Debug, Clone)]
pub enum BridgeTreeError {
    IncorrectIncompleteIndex,
    InvalidWitnessIndex,
    InvalidSavePoints,
    ContinuityError,
    CheckpointMismatch,
}

impl<H: Hash + Eq, const DEPTH: u8> BridgeTree<H, DEPTH> {
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

    /// Returns the bridges that make up this tree
    pub fn bridges(&self) -> &[MerkleBridge<H>] {
        &self.bridges
    }

    /// Returns the map from witnessed leaf values to bridge indices.
    pub fn witnessable_leaves(&self) -> &HashMap<H, usize> {
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
        self.bridges.last().map(|b| b.frontier())
    }
}

impl<H: Hashable + Hash + Eq + Clone, const DEPTH: u8> BridgeTree<H, DEPTH> {
    pub fn new(max_checkpoints: usize) -> Self {
        BridgeTree {
            bridges: vec![],
            saved: HashMap::new(),
            checkpoints: vec![],
            max_checkpoints,
        }
    }

    pub fn from_parts(
        bridges: Vec<MerkleBridge<H>>,
        saved: HashMap<H, usize>,
        checkpoints: Vec<Checkpoint>,
        max_checkpoints: usize,
    ) -> Result<Self, BridgeTreeError> {
        // check that saved values correspond to bridges
        if saved
            .iter()
            .any(|(a, i)| i >= &bridges.len() || bridges[*i].frontier().leaf_value() != a)
        {
            Err(BridgeTreeError::InvalidWitnessIndex)
        } else if bridges.is_empty() {
            // it's okay to return the empty bridge, but only if there are no saved points
            // and no non-empty checkpoints
            if saved.is_empty() && checkpoints.iter().all(|c| c.is_empty()) {
                let mut result = Self::new(max_checkpoints);
                for _ in checkpoints.iter() {
                    // restore the correct number of empty checkpoints.
                    result.checkpoint()
                }
                Ok(result)
            } else {
                Err(BridgeTreeError::InvalidSavePoints)
            }
        // check continuity of bridge order
        } else if bridges
            .iter()
            .zip(bridges.iter().skip(1))
            .any(|(prev, next)| !next.can_follow(prev))
        {
            Err(BridgeTreeError::ContinuityError)
        // verify checkpoints to ensure continuity from bridge locations
        } else if checkpoints.len() > max_checkpoints
            || checkpoints.iter().any(|c| match c {
                Checkpoint::Empty => false,
                Checkpoint::AtIndex(i) => i >= &bridges.len(),
            })
        {
            Err(BridgeTreeError::CheckpointMismatch)
        } else {
            Ok(BridgeTree {
                bridges,
                saved,
                checkpoints,
                max_checkpoints,
            })
        }
    }

    fn witness_internal(&mut self, btype: BoundaryType) -> Option<usize> {
        let next = self.bridges.last().map(|current| {
            (
                current.leaf_value().clone(),
                current.successor(self.bridges.len() - 1),
            )
        });

        next.map(|(leaf, succ)| {
            let blen = self.bridges.len();

            // a duplicate frontier might occur when we observe a previously witnessed
            // value where that value was subsequently removed.
            let is_duplicate_frontier =
                blen > 1 && self.bridges[blen - 1].frontier == self.bridges[blen - 2].frontier;

            let save_idx = if is_duplicate_frontier {
                // By saving at `blen - 2` we effectively restore the original witness.
                // We do not push new duplicate frontiers.
                blen - 2
            } else {
                // only push the successor if its frontier is not a duplicate
                self.bridges.push(succ);
                blen - 1
            };

            if btype == BoundaryType::Witness {
                self.saved.entry(leaf).or_insert(save_idx);
            }

            save_idx
        })
    }
}

impl<H: Hashable + Hash + Eq + Clone, const DEPTH: u8> crate::Frontier<H> for BridgeTree<H, DEPTH> {
    fn append(&mut self, value: &H) -> bool {
        if let Some(bridge) = self.bridges.last_mut() {
            if bridge.frontier.position().is_complete(Altitude(DEPTH)) {
                false
            } else {
                bridge.append(value.clone());
                true
            }
        } else {
            self.bridges.push(MerkleBridge::new(value.clone()));
            true
        }
    }

    /// Obtains the current root of this Merkle tree.
    fn root(&self) -> H {
        self.bridges
            .last()
            .map_or(H::empty_root(Altitude(DEPTH)), |bridge| {
                // fold from the current height, combining with empty branches,
                // up to the maximum height of the tree
                (bridge.max_altitude() + 1)
                    .iter_to(Altitude(DEPTH))
                    .fold(bridge.root(), |d, lvl| {
                        H::combine(lvl, &d, &H::empty_root(lvl))
                    })
            })
    }
}

impl<H: Hashable + Hash + Eq + Clone, const DEPTH: u8> Tree<H> for BridgeTree<H, DEPTH> {
    type Recording = BridgeRecording<H, DEPTH>;

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns true if successful and false if the tree is empty.
    fn witness(&mut self) -> bool {
        self.witness_internal(BoundaryType::Witness).is_some()
    }

    /// Obtains an authentication path to the value specified in the tree.
    /// Returns `None` if there is no available authentication path to the
    /// specified value.
    fn authentication_path(&self, value: &H) -> Option<(Position, Vec<H>)> {
        self.saved.get(value).and_then(|idx| {
            let frontier = &self.bridges[*idx].frontier;

            // Fuse the following bridges to obtain a bridge that has all
            // of the data to the right of the selected value in the tree.
            // The unwrap here is safe because a witnessed leaf always
            // generates a subsequent bridge in the tree.
            MerkleBridge::fuse_all(&self.bridges[(idx + 1)..]).map(|fused| {
                // construct a complete trailing edge that includes the data from
                // the following frontier not yet included in the trailing edge.
                let auth_fragment = fused.auth_fragments.get(idx);
                let rest_frontier = fused.frontier;

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

                (frontier.position(), result)
            })
        })
    }

    /// Marks the specified tree state value as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if the value is not a known witness.
    fn remove_witness(&mut self, value: &H) -> bool {
        self.saved.remove(value).is_some()
    }

    /// Marks the current tree state as a checkpoint if it is not already a
    /// checkpoint.
    fn checkpoint(&mut self) {
        let new_checkpoint = self
            .witness_internal(BoundaryType::Checkpoint)
            .map_or(Checkpoint::Empty, |save_idx| Checkpoint::AtIndex(save_idx));
        self.checkpoints.push(new_checkpoint);

        if self.checkpoints.len() > self.max_checkpoints {
            self.drop_oldest_checkpoint();
        }
    }

    /// Rewinds the tree state to the previous checkpoint. This function will
    /// fail and return false if there is no previous checkpoint or in the event
    /// witness data would be destroyed in the process.
    fn rewind(&mut self) -> bool {
        match self.checkpoints.pop() {
            Some(Checkpoint::Empty) => {
                if self.saved.is_empty() {
                    self.bridges.truncate(0);
                    true
                } else {
                    self.checkpoints.push(Checkpoint::Empty);
                    false
                }
            }
            Some(Checkpoint::AtIndex(i)) => {
                // Do not rewind if doing so would remove a witness.
                // However, if the index to which we are rewinding is
                // a witnessed index, we can rewind and re-witness.
                match self
                    .saved
                    .values()
                    .filter(|saved_idx| *saved_idx >= &i)
                    .max()
                {
                    Some(saved_idx) if *saved_idx > i => {
                        // there is a witnessed value at a later position, so
                        // we restore the removed checkpoint and return failure
                        self.checkpoints.push(Checkpoint::AtIndex(i));
                        false
                    }
                    Some(saved_idx) if *saved_idx == i => {
                        // the position to which we are rewinding was previously
                        // witnessed, so we re-witness after truncation
                        self.bridges.truncate(i + 1);
                        self.witness();
                        true
                    }
                    _ => {
                        // no witnesses at positions later than the checkpoint,
                        // so we can just truncate.
                        self.bridges.truncate(i + 1);
                        // if the checkpoint removed was a duplicate, we need to
                        // restore the successor bridge so that future appends correctly
                        // affect the successor bridge
                        if self
                            .checkpoints
                            .last()
                            .iter()
                            .any(|c| **c == Checkpoint::AtIndex(i))
                        {
                            self.witness_internal(BoundaryType::Checkpoint);
                        }
                        true
                    }
                }
            }
            None => false,
        }
    }

    /// Start a recording of append operations performed on a tree.
    fn recording(&self) -> BridgeRecording<H, DEPTH> {
        BridgeRecording {
            bridge: self.bridges.last().cloned(),
        }
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the recording is incompatible with the current tree state.
    fn play(&mut self, recording: &BridgeRecording<H, DEPTH>) -> bool {
        let bridge_count = self.bridges.len();
        if bridge_count == 0 {
            if let Some(bridge) = &recording.bridge {
                self.bridges.push(bridge.clone());
                true
            } else {
                // nothing to do, but no incompatibilities here
                true
            }
        } else if let Some(bridge) = &recording.bridge {
            if bridge_count == 1 {
                self.bridges[0] = bridge.clone();
                true
            } else if bridge.can_follow(&self.bridges[bridge_count - 2]) {
                self.bridges[bridge_count - 1] = bridge.clone();
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub struct BridgeRecording<H, const DEPTH: u8> {
    bridge: Option<MerkleBridge<H>>,
}

impl<H: Hashable + Clone + PartialEq, const DEPTH: u8> Recording<H> for BridgeRecording<H, DEPTH> {
    fn append(&mut self, value: &H) -> bool {
        if let Some(bridge) = self.bridge.as_mut() {
            if bridge.frontier.position.is_complete(Altitude(DEPTH)) {
                false
            } else {
                bridge.append(value.clone());
                true
            }
        } else {
            self.bridge = Some(MerkleBridge::new(value.clone()));
            true
        }
    }

    fn play(&mut self, recording: &Self) -> bool {
        if let Some((current, next)) = self.bridge.as_ref().zip(recording.bridge.as_ref()) {
            if let Some(fused) = current.fuse(&next) {
                self.bridge = Some(fused);
                true
            } else {
                false
            }
        } else {
            self.bridge = recording.bridge.clone();
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Operation;
    use crate::tests::Operation::*;
    use crate::{Frontier, Tree};

    #[test]
    fn position_altitudes() {
        assert_eq!(Position(0).max_altitude(), Altitude(0));
        assert_eq!(Position(1).max_altitude(), Altitude(0));
        assert_eq!(Position(2).max_altitude(), Altitude(1));
        assert_eq!(Position(3).max_altitude(), Altitude(1));
        assert_eq!(Position(4).max_altitude(), Altitude(2));
        assert_eq!(Position(7).max_altitude(), Altitude(2));
        assert_eq!(Position(8).max_altitude(), Altitude(3));
    }

    #[test]
    fn tree_depth() {
        let mut tree = BridgeTree::<String, 3>::new(100);
        for c in 'a'..'i' {
            assert!(tree.append(&c.to_string()))
        }
        assert!(!tree.append(&'i'.to_string()));
    }

    #[test]
    fn root_hashes() {
        let mut bridge = MerkleBridge::<String>::new("a".to_string());
        assert_eq!(bridge.root(), "a_");

        bridge.append("b".to_string());
        assert_eq!(bridge.root(), "ab");

        bridge.append("c".to_string());
        assert_eq!(bridge.root(), "abc_");

        let mut tree = BridgeTree::<String, 4>::new(100);
        assert_eq!(tree.root(), "________________");

        tree.append(&"a".to_string());
        assert_eq!(tree.root().len(), 16);
        assert_eq!(tree.root(), "a_______________");

        tree.append(&"b".to_string());
        assert_eq!(tree.root(), "ab______________");

        tree.append(&"c".to_string());
        assert_eq!(tree.root(), "abc_____________");
    }

    #[test]
    fn auth_paths() {
        let mut tree = BridgeTree::<String, 4>::new(100);
        tree.append(&"a".to_string());
        tree.witness();
        assert_eq!(
            tree.authentication_path(&"a".to_string()),
            Some((
                Position::zero(),
                vec![
                    "_".to_string(),
                    "__".to_string(),
                    "____".to_string(),
                    "________".to_string()
                ]
            ))
        );

        tree.append(&"b".to_string());
        assert_eq!(
            tree.authentication_path(&"a".to_string()),
            Some((
                Position::zero(),
                vec![
                    "b".to_string(),
                    "__".to_string(),
                    "____".to_string(),
                    "________".to_string()
                ]
            ))
        );

        tree.append(&"c".to_string());
        tree.witness();
        assert_eq!(
            tree.authentication_path(&"c".to_string()),
            Some((
                Position::from(2),
                vec![
                    "_".to_string(),
                    "ab".to_string(),
                    "____".to_string(),
                    "________".to_string()
                ]
            ))
        );

        tree.append(&"d".to_string());
        assert_eq!(
            tree.authentication_path(&"c".to_string()),
            Some((
                Position::from(2),
                vec![
                    "d".to_string(),
                    "ab".to_string(),
                    "____".to_string(),
                    "________".to_string()
                ]
            ))
        );

        tree.append(&"e".to_string());
        assert_eq!(
            tree.authentication_path(&"c".to_string()),
            Some((
                Position::from(2),
                vec![
                    "d".to_string(),
                    "ab".to_string(),
                    "e___".to_string(),
                    "________".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        tree.append(&"a".to_string());
        tree.witness();
        for c in 'b'..'h' {
            tree.append(&c.to_string());
        }
        tree.witness();
        tree.append(&"h".to_string());

        assert_eq!(
            tree.authentication_path(&"a".to_string()),
            Some((
                Position::zero(),
                vec![
                    "b".to_string(),
                    "cd".to_string(),
                    "efgh".to_string(),
                    "________".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        tree.append(&"a".to_string());
        tree.witness();
        tree.append(&"b".to_string());
        tree.append(&"c".to_string());
        tree.append(&"d".to_string());
        tree.witness();
        tree.append(&"e".to_string());
        tree.witness();
        tree.append(&"f".to_string());
        tree.witness();
        tree.append(&"g".to_string());

        assert_eq!(
            tree.authentication_path(&"f".to_string()),
            Some((
                Position::from(5),
                vec![
                    "e".to_string(),
                    "g_".to_string(),
                    "abcd".to_string(),
                    "________".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        for c in 'a'..'l' {
            tree.append(&c.to_string());
        }
        tree.witness();
        tree.append(&'l'.to_string());

        assert_eq!(
            tree.authentication_path(&"k".to_string()),
            Some((
                Position::from(10),
                vec![
                    "l".to_string(),
                    "ij".to_string(),
                    "____".to_string(),
                    "abcdefgh".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        tree.append(&'a'.to_string());
        tree.witness();
        tree.checkpoint();
        tree.rewind();
        for c in 'b'..'f' {
            tree.append(&c.to_string());
        }
        tree.witness();
        for c in 'f'..'i' {
            tree.append(&c.to_string());
        }

        assert_eq!(
            tree.authentication_path(&"a".to_string()),
            Some((
                Position::zero(),
                vec![
                    "b".to_string(),
                    "cd".to_string(),
                    "efgh".to_string(),
                    "________".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        tree.append(&'a'.to_string());
        tree.witness();
        tree.remove_witness(&'a'.to_string());
        tree.checkpoint();
        tree.witness();
        tree.rewind();
        tree.checkpoint();
        tree.append(&'a'.to_string());

        assert_eq!(
            tree.authentication_path(&"a".to_string()),
            Some((
                Position::zero(),
                vec![
                    "a".to_string(),
                    "__".to_string(),
                    "____".to_string(),
                    "________".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        tree.append(&'a'.to_string());
        tree.append(&'b'.to_string());
        tree.append(&'c'.to_string());
        tree.witness();
        tree.append(&'d'.to_string());
        tree.append(&'e'.to_string());
        tree.append(&'f'.to_string());
        tree.append(&'g'.to_string());
        tree.witness();
        tree.checkpoint();
        tree.append(&'h'.to_string());
        tree.rewind();

        assert_eq!(
            tree.authentication_path(&"c".to_string()),
            Some((
                Position::from(2),
                vec![
                    "d".to_string(),
                    "ab".to_string(),
                    "efg_".to_string(),
                    "________".to_string()
                ]
            ))
        );

        let mut tree = BridgeTree::<String, 4>::new(100);
        for c in 'a'..'n' {
            tree.append(&c.to_string());
        }
        tree.witness();
        tree.append(&'n'.to_string());
        tree.witness();
        tree.append(&'o'.to_string());
        tree.append(&'p'.to_string());

        assert_eq!(
            tree.authentication_path(&"m".to_string()),
            Some((
                Position::from(12),
                vec![
                    "n".to_string(),
                    "op".to_string(),
                    "ijkl".to_string(),
                    "abcdefgh".to_string()
                ]
            ))
        );

        let ops = ('a'..='l')
            .into_iter()
            .map(|c| Append(c.to_string()))
            .chain(Some(Witness))
            .chain(Some(Append('m'.to_string())))
            .chain(Some(Append('n'.to_string())))
            .chain(Some(Authpath('l'.to_string())))
            .collect::<Vec<_>>();

        let mut tree = BridgeTree::<String, 4>::new(100);
        assert_eq!(
            Operation::apply_all(&ops, &mut tree),
            Some((
                Position::from(11),
                vec![
                    "k".to_string(),
                    "ij".to_string(),
                    "mn__".to_string(),
                    "abcdefgh".to_string()
                ]
            ))
        );
    }

    #[test]
    fn drop_oldest_checkpoint() {
        let mut t = BridgeTree::<String, 6>::new(100);
        t.checkpoint();
        t.append(&"a".to_string());
        t.witness();
        t.append(&"b".to_string());
        t.append(&"c".to_string());
        assert!(!t.rewind());
        assert!(t.drop_oldest_checkpoint());
    }

    #[test]
    fn checkpoint_rewind_0() {
        let mut t = BridgeTree::<String, 6>::new(100);
        t.append(&"a".to_string());
        t.append(&"b".to_string());
        t.checkpoint();
        t.append(&"c".to_string());
        t.witness();
        assert!(!t.rewind());
    }

    #[test]
    fn checkpoint_rewind_1() {
        let mut t = BridgeTree::<String, 6>::new(100);
        t.append(&"a".to_string());
        t.append(&"b".to_string());
        t.checkpoint();
        t.witness();
        t.witness();
        assert!(t.rewind());
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
}
