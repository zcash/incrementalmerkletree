//! # `incrementalmerkletree`
//!
//! Incremental Merkle Trees are fixed-depth Merkle trees with two primary
//! capabilities: appending (assigning a value to the next unused leaf and
//! advancing the tree) and obtaining the root of the tree. Importantly the tree
//! structure attempts to store the least amount of information necessary to
//! continue to function; other information should be pruned eagerly to avoid
//! waste when the tree state is encoded.
//!
//! ## Witnessing
//!
//! Merkle trees are typically used to show that a value exists in the tree via
//! an authentication path. We need an API that allows us to identify the
//! current leaf as a value we wish to compute authentication paths for even as
//! the tree continues to be appended to in the future; this is called
//! maintaining a witness. When we're later uninterested in such a leaf, we can
//! prune a witness and remove all unnecessary information from the structure as
//! a consequence.
//!
//! ## Checkpoints and Rollbacks
//!
//! The structure is not append-only in the strict sense. It is possible to
//! identify the current state of the tree as a "checkpoint" and to remove older
//! checkpoints that we're no longer interested in. It should be possible to
//! roll back to any previous checkpoint.

pub mod bridgetree;
mod sample;

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::num::TryFromIntError;
use std::ops::{Add, AddAssign, Sub};

/// A type-safe wrapper for indexing into "levels" of a binary tree, such that
/// nodes at altitude `0` are leaves, nodes at altitude `1` are parents
/// of nodes at altitude `0`, and so forth. This type is capable of
/// representing altitudes in trees containing up to 2^255 leaves.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Altitude(u8);

impl Altitude {
    /// Convenience method for returning the zero altitude.
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn one() -> Self {
        Self(1)
    }

    pub fn iter_to(self, other: Altitude) -> impl Iterator<Item = Self> {
        (self.0..other.0).into_iter().map(Altitude)
    }
}

impl Add<u8> for Altitude {
    type Output = Self;
    fn add(self, value: u8) -> Self {
        Self(self.0 + value)
    }
}

impl Sub<u8> for Altitude {
    type Output = Self;
    fn sub(self, value: u8) -> Self {
        Self(self.0 - value)
    }
}

impl From<u8> for Altitude {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl From<Altitude> for u8 {
    fn from(level: Altitude) -> u8 {
        level.0
    }
}

impl From<Altitude> for usize {
    fn from(level: Altitude) -> usize {
        level.0 as usize
    }
}

/// A type representing the position of a leaf in a Merkle tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Position(usize);

impl Position {
    /// Returns the position of the first leaf in the tree.
    pub fn zero() -> Self {
        Self(0)
    }

    /// Returns the altitude of the top of a binary tree containing
    /// a number of nodes equal to the next power of two greater than
    /// or equal to `self + 1`.
    fn max_altitude(&self) -> Altitude {
        Altitude(if self.0 == 0 {
            0
        } else {
            63 - self.0.leading_zeros() as u8
        })
    }

    /// Returns the altitude of each populated ommer.
    pub fn ommer_altitudes(&self) -> impl Iterator<Item = Altitude> + '_ {
        (0..=self.max_altitude().0)
            .into_iter()
            .filter_map(move |i| {
                if i != 0 && self.0 & (1 << i) != 0 {
                    Some(Altitude(i))
                } else {
                    None
                }
            })
    }

    /// Returns the altitude of each cousin and/or ommer required to construct
    /// an authentication path to the root of a merkle tree that has `self + 1`
    /// nodes.
    pub fn altitudes_required(&self) -> impl Iterator<Item = Altitude> + '_ {
        (0..=self.max_altitude().0)
            .into_iter()
            .filter_map(move |i| {
                if self.0 == 0 || self.0 & (1 << i) == 0 {
                    Some(Altitude(i))
                } else {
                    None
                }
            })
    }

    /// Returns the altitude of each cousin and/or ommer required to construct
    /// an authentication path to the root of a merkle tree containing 2^64
    /// nodes.
    pub fn all_altitudes_required(&self) -> impl Iterator<Item = Altitude> + '_ {
        (0..64).into_iter().filter_map(move |i| {
            if self.0 == 0 || self.0 & (1 << i) == 0 {
                Some(Altitude(i))
            } else {
                None
            }
        })
    }

    /// Returns whether the binary tree having `self` as the position of the
    /// rightmost leaf contains a perfect balanced tree of height
    /// `to_altitude + 1` that contains the aforesaid leaf, without requiring
    /// any empty leaves or internal nodes.
    pub fn is_complete(&self, to_altitude: Altitude) -> bool {
        for i in 0..(to_altitude.0) {
            if self.0 & (1 << i) == 0 {
                return false;
            }
        }
        true
    }
}

impl From<Position> for usize {
    fn from(p: Position) -> usize {
        p.0
    }
}

impl From<Position> for u64 {
    fn from(p: Position) -> Self {
        p.0 as u64
    }
}

impl Add<usize> for Position {
    type Output = Position;
    fn add(self, other: usize) -> Self {
        Position(self.0 + other)
    }
}

impl AddAssign<usize> for Position {
    fn add_assign(&mut self, other: usize) {
        self.0 += other
    }
}

impl From<usize> for Position {
    fn from(sz: usize) -> Self {
        Self(sz)
    }
}

impl TryFrom<u64> for Position {
    type Error = TryFromIntError;
    fn try_from(sz: u64) -> Result<Self, Self::Error> {
        <usize>::try_from(sz).map(Self)
    }
}

/// A trait describing the operations that make a value  suitable for inclusion in
/// an incremental merkle tree.
pub trait Hashable: Sized {
    fn empty_leaf() -> Self;

    fn combine(level: Altitude, a: &Self, b: &Self) -> Self;

    fn empty_root(level: Altitude) -> Self {
        Altitude::zero()
            .iter_to(level)
            .fold(Self::empty_leaf(), |v, lvl| Self::combine(lvl, &v, &v))
    }
}

/// A possibly-empty incremental Merkle frontier.
pub trait Frontier<H> {
    /// Appends a new value to the frontier at the next available slot.
    /// Returns true if successful and false if the frontier would exceed
    /// the maximum allowed depth.
    fn append(&mut self, value: &H) -> bool;

    /// Obtains the current root of this Merkle frontier by hashing
    /// against empty nodes up to the maximum height of the pruned
    /// tree that the frontier represents.
    fn root(&self) -> H;
}

/// A Merkle tree that supports incremental appends, witnessing of
/// leaf nodes, checkpoints and rollbacks.
pub trait Tree<H> {
    /// Appends a new value to the tree at the next available slot.
    /// Returns true if successful and false if the tree would exceed
    /// the maximum allowed depth.
    fn append(&mut self, value: &H) -> bool;

    /// Returns the most recently appended leaf value.
    fn current_position(&self) -> Option<Position>;

    /// Returns the most recently appended leaf value.
    fn current_leaf(&self) -> Option<&H>;

    /// Returns the leaf at the specified position if the tree can produce
    /// an authentication path for it.
    fn get_witnessed_leaf(&self, position: Position) -> Option<&H>;

    /// Marks the current leaf as one for which we're interested in producing
    /// an authentication path. Returns an optional value containing the
    /// current position if successful or if the current value was already
    /// marked, or None if the tree is empty.
    fn witness(&mut self) -> Option<Position>;

    /// Return a set of all the positions for which we have witnessed.
    fn witnessed_positions(&self) -> BTreeSet<Position>;

    /// Obtains the root of the Merkle tree at the specified checkpoint depth
    /// by hashing against empty nodes up to the maximum height of the tree.
    /// Returns `None` if there are not enough checkpoints available to reach the
    /// requested checkpoint depth.
    fn root(&self, checkpoint_depth: usize) -> Option<H>;

    /// Obtains an authentication path to the value at the specified position,
    /// as of the tree state corresponding to the given root.
    /// Returns `None` if there is no available authentication path to that
    /// position or if the root does not correspond to a checkpointed
    /// root of the tree.
    fn authentication_path(&self, position: Position, as_of_root: &H) -> Option<Vec<H>>;

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if we were already not maintaining a witness at this position.
    fn remove_witness(&mut self, position: Position) -> bool;

    /// Creates a new checkpoint for the current tree state. It is valid to
    /// have multiple checkpoints for the same tree state, and each `rewind`
    /// call will remove a single checkpoint.
    fn checkpoint(&mut self);

    /// Rewinds the tree state to the previous checkpoint, and then removes
    /// that checkpoint record. If there are multiple checkpoints at a given
    /// tree state, the tree state will not be altered until all checkpoints
    /// at that tree state have been removed using `rewind`. This function
    /// return false and leave the tree unmodified if no checkpoints exist.
    fn rewind(&mut self) -> bool;

    /// Remove state from the tree that no longer needs to be maintained
    /// because it is associated with checkpoints or witnesses that
    /// have been removed from the tree at positions deeper than those
    /// reachable by calls to `rewind`. It is always safe to implement
    /// this as a no-op operation
    fn garbage_collect(&mut self);
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(deprecated)]
    use std::collections::BTreeSet;
    use std::fmt::Debug;
    use std::hash::Hasher;
    use std::hash::SipHasher;

    use super::bridgetree::BridgeTree;
    use super::sample::{lazy_root, CompleteTree};
    use super::{Altitude, Hashable, Position, Tree};

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

    //
    // Types and utilities for shared example tests.
    //

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub(crate) struct SipHashable(pub(crate) u64);

    impl Hashable for SipHashable {
        fn empty_leaf() -> Self {
            SipHashable(0)
        }

        fn combine(_level: Altitude, a: &Self, b: &Self) -> Self {
            let mut hasher = SipHasher::new();
            hasher.write_u64(a.0);
            hasher.write_u64(b.0);
            SipHashable(hasher.finish())
        }
    }

    impl Hashable for String {
        fn empty_leaf() -> Self {
            "_".to_string()
        }

        fn combine(_: Altitude, a: &Self, b: &Self) -> Self {
            a.to_string() + b
        }
    }

    //
    // Shared example tests
    //

    pub(crate) fn check_root_hashes<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
        let mut tree = new_tree(100);
        assert_eq!(tree.root(0).unwrap(), "________________");

        tree.append(&"a".to_string());
        assert_eq!(tree.root(0).unwrap().len(), 16);
        assert_eq!(tree.root(0).unwrap(), "a_______________");

        tree.append(&"b".to_string());
        assert_eq!(tree.root(0).unwrap(), "ab______________");

        tree.append(&"c".to_string());
        assert_eq!(tree.root(0).unwrap(), "abc_____________");

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.checkpoint();
        t.witness();
        t.append(&"a".to_string());
        t.append(&"a".to_string());
        t.append(&"a".to_string());
        assert_eq!(t.root(0).unwrap(), "aaaa____________");
    }

    pub(crate) fn check_auth_paths<T: Tree<String> + std::fmt::Debug, F: Fn(usize) -> T>(
        new_tree: F,
    ) {
        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.witness();
        assert_eq!(
            tree.authentication_path(Position::from(0), &tree.root(0).unwrap()),
            Some(vec![
                "_".to_string(),
                "__".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"b".to_string());
        assert_eq!(
            tree.authentication_path(Position::zero(), &tree.root(0).unwrap()),
            Some(vec![
                "b".to_string(),
                "__".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"c".to_string());
        tree.witness();
        assert_eq!(
            tree.authentication_path(Position::from(2), &tree.root(0).unwrap()),
            Some(vec![
                "_".to_string(),
                "ab".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"d".to_string());
        assert_eq!(
            tree.authentication_path(Position::from(2), &tree.root(0).unwrap()),
            Some(vec![
                "d".to_string(),
                "ab".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"e".to_string());
        assert_eq!(
            tree.authentication_path(Position::from(2), &tree.root(0).unwrap()),
            Some(vec![
                "d".to_string(),
                "ab".to_string(),
                "e___".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.witness();
        for c in 'b'..'h' {
            tree.append(&c.to_string());
        }
        tree.witness();
        tree.append(&"h".to_string());

        assert_eq!(
            tree.authentication_path(Position::zero(), &tree.root(0).unwrap()),
            Some(vec![
                "b".to_string(),
                "cd".to_string(),
                "efgh".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
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
            tree.authentication_path(Position::from(5), &tree.root(0).unwrap()),
            Some(vec![
                "e".to_string(),
                "g_".to_string(),
                "abcd".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
        for c in 'a'..'l' {
            tree.append(&c.to_string());
        }
        tree.witness();
        tree.append(&'l'.to_string());

        assert_eq!(
            tree.authentication_path(Position::from(10), &tree.root(0).unwrap()),
            Some(vec![
                "l".to_string(),
                "ij".to_string(),
                "____".to_string(),
                "abcdefgh".to_string()
            ])
        );

        let mut tree = new_tree(100);
        tree.append(&'a'.to_string());
        tree.witness();
        tree.checkpoint();
        assert!(tree.rewind());
        for c in 'b'..'f' {
            tree.append(&c.to_string());
        }
        tree.witness();
        for c in 'f'..'i' {
            tree.append(&c.to_string());
        }

        assert_eq!(
            tree.authentication_path(Position::zero(), &tree.root(0).unwrap()),
            Some(vec![
                "b".to_string(),
                "cd".to_string(),
                "efgh".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
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
        assert!(tree.rewind());

        assert_eq!(
            tree.authentication_path(Position::from(2), &tree.root(0).unwrap()),
            Some(vec![
                "d".to_string(),
                "ab".to_string(),
                "efg_".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
        tree.append(&'a'.to_string());
        tree.append(&'b'.to_string());
        tree.witness();
        assert_eq!(
            tree.authentication_path(Position::from(0), &tree.root(0).unwrap()),
            None
        );

        let mut tree = new_tree(100);
        for c in 'a'..'n' {
            tree.append(&c.to_string());
        }
        tree.witness();
        tree.append(&'n'.to_string());
        tree.witness();
        tree.append(&'o'.to_string());
        tree.append(&'p'.to_string());

        assert_eq!(
            tree.authentication_path(Position::from(12), &tree.root(0).unwrap()),
            Some(vec![
                "n".to_string(),
                "op".to_string(),
                "ijkl".to_string(),
                "abcdefgh".to_string()
            ])
        );

        let ops = ('a'..='l')
            .into_iter()
            .map(|c| Append(c.to_string()))
            .chain(Some(Witness))
            .chain(Some(Append('m'.to_string())))
            .chain(Some(Append('n'.to_string())))
            .chain(Some(Authpath(11usize.into(), 0)))
            .collect::<Vec<_>>();

        let mut tree = new_tree(100);
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

    pub(crate) fn check_checkpoint_rewind<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
        let mut t = new_tree(100);
        assert!(!t.rewind());

        let mut t = new_tree(100);
        t.checkpoint();
        assert!(t.rewind());

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.checkpoint();
        t.append(&"b".to_string());
        t.witness();
        assert!(t.rewind());
        assert_eq!(Some(Position::from(0)), t.current_position());

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.witness();
        t.checkpoint();
        assert!(t.rewind());

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.checkpoint();
        t.witness();
        t.append(&"a".to_string());
        assert!(t.rewind());
        assert_eq!(Some(Position::from(0)), t.current_position());

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.checkpoint();
        t.checkpoint();
        assert!(t.rewind());
        t.append(&"b".to_string());
        assert!(t.rewind());
        t.append(&"b".to_string());
        assert_eq!(t.root(0).unwrap(), "ab______________");
    }

    pub(crate) fn check_rewind_remove_witness<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.witness();
        tree.checkpoint();
        assert!(tree.rewind());
        assert!(tree.remove_witness(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.checkpoint();
        tree.witness();
        assert!(tree.rewind());
        assert!(!tree.remove_witness(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.witness();
        tree.checkpoint();
        assert!(tree.remove_witness(0usize.into()));
        assert!(tree.rewind());
        assert!(tree.remove_witness(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.witness();
        assert!(tree.remove_witness(0usize.into()));
        tree.checkpoint();
        assert!(tree.rewind());
        assert!(!tree.remove_witness(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        assert!(!tree.remove_witness(0usize.into()));
        tree.checkpoint();
        assert!(tree.witness().is_some());
        assert!(tree.rewind());

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.checkpoint();
        assert!(tree.witness().is_some());
        assert!(tree.remove_witness(0usize.into()));
        assert!(tree.rewind());
        assert!(!tree.remove_witness(0usize.into()));

        // The following check_operations tests cover errors where the
        // test framework itself previously did not correctly handle
        // chain state restoration.

        let samples = vec![
            vec![append("x"), Checkpoint, Witness, Rewind, unwitness(0)],
            vec![
                append("d"),
                Checkpoint,
                Witness,
                unwitness(0),
                Rewind,
                unwitness(0),
            ],
            vec![
                append("o"),
                Checkpoint,
                Witness,
                Checkpoint,
                unwitness(0),
                Rewind,
                Rewind,
            ],
            vec![
                append("s"),
                Witness,
                append("m"),
                Checkpoint,
                unwitness(0),
                Rewind,
                unwitness(0),
                unwitness(0),
            ],
        ];

        for (i, sample) in samples.iter().enumerate() {
            let result = check_operations(sample);
            assert!(
                matches!(result, Ok(())),
                "Reference/Test mismatch at index {}: {:?}",
                i,
                result
            );
        }
    }

    //
    // Types and utilities for cross-verification property tests
    //

    #[derive(Clone)]
    pub struct CombinedTree<H: Hashable + Ord, const DEPTH: u8> {
        inefficient: CompleteTree<H>,
        efficient: BridgeTree<H, DEPTH>,
    }

    impl<H: Hashable + Ord + Clone, const DEPTH: u8> CombinedTree<H, DEPTH> {
        pub fn new() -> Self {
            CombinedTree {
                inefficient: CompleteTree::new(DEPTH.into(), 100),
                efficient: BridgeTree::new(100),
            }
        }
    }

    impl<H: Hashable + Ord + Clone + Debug, const DEPTH: u8> Tree<H> for CombinedTree<H, DEPTH> {
        fn append(&mut self, value: &H) -> bool {
            let a = self.inefficient.append(value);
            let b = self.efficient.append(value);
            assert_eq!(a, b);
            a
        }

        fn root(&self, checkpoint_depth: usize) -> Option<H> {
            let a = self.inefficient.root(checkpoint_depth);
            let b = self.efficient.root(checkpoint_depth);
            assert_eq!(a, b);
            a
        }

        fn current_position(&self) -> Option<Position> {
            let a = self.inefficient.current_position();
            let b = self.efficient.current_position();
            assert_eq!(a, b);
            a
        }

        fn current_leaf(&self) -> Option<&H> {
            let a = self.inefficient.current_leaf();
            let b = self.efficient.current_leaf();
            assert_eq!(a, b);
            a
        }

        fn get_witnessed_leaf(&self, position: Position) -> Option<&H> {
            let a = self.inefficient.get_witnessed_leaf(position);
            let b = self.efficient.get_witnessed_leaf(position);
            assert_eq!(a, b);
            a
        }

        fn witness(&mut self) -> Option<Position> {
            let a = self.inefficient.witness();
            let b = self.efficient.witness();
            assert_eq!(a, b);
            let apos = self.inefficient.witnessed_positions();
            let bpos = self.efficient.witnessed_positions();
            assert_eq!(apos, bpos);
            a
        }

        fn witnessed_positions(&self) -> BTreeSet<Position> {
            let a = self.inefficient.witnessed_positions();
            let b = self.efficient.witnessed_positions();
            assert_eq!(a, b);
            a
        }

        fn authentication_path(&self, position: Position, as_of_root: &H) -> Option<Vec<H>> {
            let a = self.inefficient.authentication_path(position, as_of_root);
            let b = self.efficient.authentication_path(position, as_of_root);
            assert_eq!(a, b);
            a
        }

        fn remove_witness(&mut self, position: Position) -> bool {
            let a = self.inefficient.remove_witness(position);
            let b = self.efficient.remove_witness(position);
            assert_eq!(a, b);
            a
        }

        fn checkpoint(&mut self) {
            self.inefficient.checkpoint();
            self.efficient.checkpoint();
        }

        fn rewind(&mut self) -> bool {
            let a = self.inefficient.rewind();
            let b = self.efficient.rewind();
            assert_eq!(a, b);
            a
        }

        fn garbage_collect(&mut self) {
            self.inefficient.garbage_collect();
            self.efficient.garbage_collect();
        }
    }

    //
    // Operations
    //

    #[derive(Clone, Debug)]
    pub enum Operation<A> {
        Append(A),
        CurrentPosition,
        CurrentLeaf,
        Witness,
        WitnessedLeaf(Position),
        WitnessedPositions,
        Unwitness(Position),
        Checkpoint,
        Rewind,
        Authpath(Position, usize),
        GarbageCollect,
    }

    use Operation::*;

    fn append(x: &str) -> Operation<String> {
        Operation::Append(x.to_string())
    }

    fn unwitness(pos: usize) -> Operation<String> {
        Operation::Unwitness(Position::from(pos))
    }

    fn authpath(pos: usize, depth: usize) -> Operation<String> {
        Operation::Authpath(Position::from(pos), depth)
    }

    impl<H: Hashable> Operation<H> {
        pub fn apply<T: Tree<H>>(&self, tree: &mut T) -> Option<(Position, Vec<H>)> {
            match self {
                Append(a) => {
                    assert!(tree.append(a), "append failed");
                    None
                }
                CurrentPosition => None,
                CurrentLeaf => None,
                Witness => {
                    assert!(tree.witness().is_some(), "witness failed");
                    None
                }
                WitnessedLeaf(_) => None,
                WitnessedPositions => None,
                Unwitness(p) => {
                    assert!(tree.remove_witness(*p), "remove witness failed");
                    None
                }
                Checkpoint => {
                    tree.checkpoint();
                    None
                }
                Rewind => {
                    assert!(tree.rewind(), "rewind failed");
                    None
                }
                Authpath(p, d) => tree
                    .root(*d)
                    .and_then(|root| tree.authentication_path(*p, &root))
                    .map(|xs| (*p, xs)),
                GarbageCollect => None,
            }
        }

        pub fn apply_all<T: Tree<H>>(
            ops: &[Operation<H>],
            tree: &mut T,
        ) -> Option<(Position, Vec<H>)> {
            let mut result = None;
            for op in ops {
                result = op.apply(tree);
            }
            result
        }
    }

    pub(crate) fn compute_root_from_auth_path<H: Hashable>(
        value: H,
        position: Position,
        path: &[H],
    ) -> H {
        let mut cur = value;
        let mut lvl = Altitude::zero();
        for (i, v) in path
            .iter()
            .enumerate()
            .map(|(i, v)| (((<usize>::from(position) >> i) & 1) == 1, v))
        {
            if i {
                cur = H::combine(lvl, v, &cur);
            } else {
                cur = H::combine(lvl, &cur, v);
            }
            lvl = lvl + 1;
        }
        cur
    }

    #[test]
    fn test_compute_root_from_auth_path() {
        let expected = SipHashable::combine(
            <Altitude>::from(2),
            &SipHashable::combine(
                Altitude::one(),
                &SipHashable::combine(Altitude::zero(), &SipHashable(0), &SipHashable(1)),
                &SipHashable::combine(Altitude::zero(), &SipHashable(2), &SipHashable(3)),
            ),
            &SipHashable::combine(
                Altitude::one(),
                &SipHashable::combine(Altitude::zero(), &SipHashable(4), &SipHashable(5)),
                &SipHashable::combine(Altitude::zero(), &SipHashable(6), &SipHashable(7)),
            ),
        );

        assert_eq!(
            compute_root_from_auth_path::<SipHashable>(
                SipHashable(0),
                Position::zero(),
                &[
                    SipHashable(1),
                    SipHashable::combine(Altitude::zero(), &SipHashable(2), &SipHashable(3)),
                    SipHashable::combine(
                        Altitude::one(),
                        &SipHashable::combine(Altitude::zero(), &SipHashable(4), &SipHashable(5)),
                        &SipHashable::combine(Altitude::zero(), &SipHashable(6), &SipHashable(7))
                    )
                ]
            ),
            expected
        );

        assert_eq!(
            compute_root_from_auth_path(
                SipHashable(4),
                <Position>::from(4),
                &[
                    SipHashable(5),
                    SipHashable::combine(Altitude::zero(), &SipHashable(6), &SipHashable(7)),
                    SipHashable::combine(
                        Altitude::one(),
                        &SipHashable::combine(Altitude::zero(), &SipHashable(0), &SipHashable(1)),
                        &SipHashable::combine(Altitude::zero(), &SipHashable(2), &SipHashable(3))
                    )
                ]
            ),
            expected
        );
    }

    #[test]
    fn test_auth_path_consistency() {
        let samples = vec![
            // Reduced examples
            vec![
                append("a"),
                append("b"),
                Checkpoint,
                Witness,
                authpath(0, 1),
            ],
            vec![
                append("c"),
                append("d"),
                Witness,
                Checkpoint,
                authpath(1, 1),
            ],
            vec![
                append("e"),
                Checkpoint,
                Witness,
                append("f"),
                authpath(0, 1),
            ],
            vec![
                append("g"),
                Witness,
                Checkpoint,
                unwitness(0),
                append("h"),
                authpath(0, 0),
            ],
            vec![
                append("i"),
                Checkpoint,
                Witness,
                unwitness(0),
                append("j"),
                authpath(0, 0),
            ],
            vec![
                append("i"),
                Witness,
                append("j"),
                Checkpoint,
                append("k"),
                authpath(0, 1),
            ],
            vec![
                append("l"),
                Checkpoint,
                Witness,
                Checkpoint,
                append("m"),
                Checkpoint,
                authpath(0, 2),
            ],
            vec![Checkpoint, append("n"), Witness, authpath(0, 1)],
            vec![
                append("a"),
                Witness,
                Checkpoint,
                unwitness(0),
                Checkpoint,
                append("b"),
                authpath(0, 1),
            ],
            vec![
                append("a"),
                Witness,
                append("b"),
                unwitness(0),
                Checkpoint,
                authpath(0, 0),
            ],
            vec![
                append("a"),
                Witness,
                Checkpoint,
                unwitness(0),
                Checkpoint,
                Rewind,
                append("b"),
                authpath(0, 0),
            ],
            vec![
                append("a"),
                Witness,
                Checkpoint,
                Checkpoint,
                Rewind,
                append("a"),
                unwitness(0),
                authpath(0, 1),
            ],
            // Unreduced examples
            vec![
                append("o"),
                append("p"),
                Witness,
                append("q"),
                Checkpoint,
                unwitness(1),
                authpath(1, 1),
            ],
            vec![
                append("r"),
                append("s"),
                append("t"),
                Witness,
                Checkpoint,
                unwitness(2),
                Checkpoint,
                authpath(2, 2),
            ],
            vec![
                append("u"),
                Witness,
                append("v"),
                append("w"),
                Checkpoint,
                unwitness(0),
                append("x"),
                Checkpoint,
                Checkpoint,
                authpath(0, 3),
            ],
        ];

        for (i, sample) in samples.iter().enumerate() {
            let result = check_operations(sample);
            assert!(
                matches!(result, Ok(())),
                "Reference/Test mismatch at index {}: {:?}",
                i,
                result
            );
        }
    }

    // These check_operations tests cover errors where the test framework itself previously did not
    // correctly handle chain state restoration.
    #[test]
    fn test_rewind_remove_witness_consistency() {
        let samples = vec![
            vec![append("x"), Checkpoint, Witness, Rewind, unwitness(0)],
            vec![
                append("d"),
                Checkpoint,
                Witness,
                unwitness(0),
                Rewind,
                unwitness(0),
            ],
            vec![
                append("o"),
                Checkpoint,
                Witness,
                Checkpoint,
                unwitness(0),
                Rewind,
                Rewind,
            ],
            vec![
                append("s"),
                Witness,
                append("m"),
                Checkpoint,
                unwitness(0),
                Rewind,
                unwitness(0),
                unwitness(0),
            ],
        ];
        for (i, sample) in samples.iter().enumerate() {
            let result = check_operations(sample);
            assert!(
                matches!(result, Ok(())),
                "Reference/Test mismatch at index {}: {:?}",
                i,
                result
            );
        }
    }

    use proptest::prelude::*;

    pub fn arb_operation<G: Strategy + Clone>(
        item_gen: G,
        pos_gen: impl Strategy<Value = usize> + Clone,
    ) -> impl Strategy<Value = Operation<G::Value>>
    where
        G::Value: Clone + 'static,
    {
        prop_oneof![
            item_gen.prop_map(Operation::Append),
            Just(Operation::Witness),
            prop_oneof![
                Just(Operation::CurrentLeaf),
                Just(Operation::CurrentPosition),
                Just(Operation::WitnessedPositions),
            ],
            Just(Operation::GarbageCollect),
            pos_gen
                .clone()
                .prop_map(|i| Operation::WitnessedLeaf(Position::from(i))),
            pos_gen
                .clone()
                .prop_map(|i| Operation::Unwitness(Position::from(i))),
            Just(Operation::Checkpoint),
            Just(Operation::Rewind),
            pos_gen.prop_flat_map(|i| (0usize..10)
                .prop_map(move |depth| Operation::Authpath(Position::from(i), depth))),
        ]
    }

    pub fn apply_operation<H, T: Tree<H>>(tree: &mut T, op: Operation<H>) {
        match op {
            Append(value) => {
                tree.append(&value);
            }
            Witness => {
                tree.witness();
            }
            Unwitness(position) => {
                tree.remove_witness(position);
            }
            Checkpoint => {
                tree.checkpoint();
            }
            Rewind => {
                tree.rewind();
            }
            CurrentPosition => {}
            CurrentLeaf => {}
            Authpath(_, _) => {}
            WitnessedLeaf(_) => {}
            WitnessedPositions => {}
            GarbageCollect => {}
        }
    }

    fn check_operations<H: Hashable + Ord + Clone + Debug>(
        ops: &[Operation<H>],
    ) -> Result<(), TestCaseError> {
        const DEPTH: u8 = 4;
        let mut tree = CombinedTree::<H, DEPTH>::new();

        let mut tree_size = 0;
        let mut tree_values: Vec<H> = vec![];
        // the number of leaves in the tree at the time that a checkpoint is made
        let mut tree_checkpoints: Vec<usize> = vec![];

        for op in ops {
            prop_assert_eq!(tree_size, tree_values.len());
            match op {
                Append(value) => {
                    if tree.append(value) {
                        prop_assert!(tree_size < (1 << DEPTH));
                        tree_size += 1;
                        tree_values.push(value.clone());
                    } else {
                        prop_assert_eq!(tree_size, 1 << DEPTH);
                    }
                }
                CurrentPosition => {
                    if let Some(pos) = tree.current_position() {
                        prop_assert!(tree_size > 0);
                        prop_assert_eq!(tree_size - 1, pos.into());
                    }
                }
                CurrentLeaf => {
                    prop_assert_eq!(tree_values.last(), tree.current_leaf());
                }
                Witness => {
                    if tree.witness().is_some() {
                        prop_assert!(tree_size != 0);
                    } else {
                        prop_assert_eq!(tree_size, 0);
                    }
                }
                WitnessedLeaf(position) => {
                    if tree.get_witnessed_leaf(*position).is_some() {
                        prop_assert!(<usize>::from(*position) < tree_size);
                    }
                }
                Unwitness(position) => {
                    tree.remove_witness(*position);
                }
                WitnessedPositions => {}
                Checkpoint => {
                    tree_checkpoints.push(tree_size);
                    tree.checkpoint();
                }
                Rewind => {
                    if tree.rewind() {
                        prop_assert!(!tree_checkpoints.is_empty());
                        let checkpointed_tree_size = tree_checkpoints.pop().unwrap();
                        tree_values.truncate(checkpointed_tree_size);
                        tree_size = checkpointed_tree_size;
                    }
                }
                Authpath(position, depth) => {
                    if let Some(path) = tree
                        .root(*depth)
                        .and_then(|r| tree.authentication_path(*position, &r))
                    {
                        let value: H = tree_values[<usize>::from(*position)].clone();
                        let tree_root = tree.root(*depth);

                        if tree_checkpoints.len() >= *depth {
                            let mut extended_tree_values = tree_values.clone();
                            if *depth > 0 {
                                // prune the tree back to the checkpointed size.
                                if let Some(checkpointed_tree_size) =
                                    tree_checkpoints.get(tree_checkpoints.len() - depth)
                                {
                                    extended_tree_values.truncate(*checkpointed_tree_size);
                                }
                            }
                            // extend the tree with empty leaves until it is full
                            extended_tree_values.resize(1 << DEPTH, H::empty_leaf());

                            // compute the root
                            let expected_root = lazy_root::<H>(extended_tree_values);
                            prop_assert_eq!(&tree_root.unwrap(), &expected_root);

                            prop_assert_eq!(
                                &compute_root_from_auth_path(value, *position, &path),
                                &expected_root
                            );
                        }
                    }
                }
                GarbageCollect => {}
            }
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100000))]

        #[test]
        fn check_randomized_u64_ops(
            ops in proptest::collection::vec(
                arb_operation((0..32u64).prop_map(SipHashable), 0usize..100),
                1..100
            )
        ) {
            check_operations(&ops)?;
        }

        #[test]
        fn check_randomized_str_ops(
            ops in proptest::collection::vec(
                arb_operation((97u8..123).prop_map(|c| char::from(c).to_string()), 0usize..100),
                1..100
            )
        ) {
            check_operations::<String>(&ops)?;
        }
    }
}
