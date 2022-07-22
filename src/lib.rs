//! # `incrementalmerkletree`
//!
//! Incremental Merkle Trees are fixed-depth Merkle trees with two primary
//! capabilities: appending (assigning a value to the next unused leaf and
//! advancing the tree) and obtaining the root of the tree. Importantly the tree
//! structure attempts to store the least amount of information necessary to
//! continue to function; other information should be pruned eagerly to avoid
//! waste when the tree state is encoded.
//!
//! ## Marking
//!
//! Merkle trees are typically used to show that a value exists in the tree via
//! a witness. We need an API that allows us to identify the
//! current leaf as a value we wish to compute witnesss for even as
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
use std::convert::{TryFrom, TryInto};
use std::num::TryFromIntError;
use std::ops::{Add, AddAssign, Range};

/// A type-safe wrapper for indexing into "levels" of a binary tree, such that
/// nodes at level `0` are leaves, nodes at level `1` are parents of nodes at
/// level `0`, and so forth. This type is capable of representing levels in
/// trees containing up to 2^255 leaves.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Level(u8);

impl Level {
    // TODO: replace with an instance for `Step<Level>` once `step_trait`
    // is stabilized
    pub fn iter_to(self, other: Level) -> impl Iterator<Item = Self> {
        (self.0..other.0).into_iter().map(Level)
    }
}

impl Add<u8> for Level {
    type Output = Self;
    fn add(self, value: u8) -> Self {
        Self(self.0 + value)
    }
}

impl From<u8> for Level {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl From<Level> for u8 {
    fn from(level: Level) -> u8 {
        level.0
    }
}

impl From<Level> for usize {
    fn from(level: Level) -> usize {
        level.0 as usize
    }
}

/// A type representing the position of a leaf in a Merkle tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Position(usize);

impl Position {
    /// Return whether the position is odd-valued.
    pub fn is_odd(&self) -> bool {
        self.0 & 0x1 == 1
    }

    /// Returns the minimum possible level of the root of a binary tree containing at least
    /// `self + 1` nodes.
    pub fn root_level(&self) -> Level {
        Level(64 - self.0.leading_zeros() as u8)
    }

    /// Returns the number of cousins and/or ommers required to construct an authentication
    /// path to the root of a merkle tree that has `self + 1` nodes.
    pub fn past_ommer_count(&self) -> usize {
        (0..self.root_level().0)
            .filter(|i| (self.0 >> i) & 0x1 == 1)
            .count()
    }

    /// Returns whether the binary tree having `self` as the position of the rightmost leaf
    /// contains a perfect balanced tree with a root at level `root_level` that contains the
    /// aforesaid leaf.
    pub fn is_complete_subtree(&self, root_level: Level) -> bool {
        !(0..(root_level.0)).any(|l| self.0 & (1 << l) == 0)
    }

    /// Returns an iterator over the addresses of nodes required to create a witness for this
    /// position, beginning with the sibling of the leaf at this position and ending with the
    /// sibling of the ancestor of the leaf at this position that is required to compute a root at
    /// the specified level.
    pub(crate) fn witness_addrs(
        &self,
        root_level: Level,
    ) -> impl Iterator<Item = (Address, Source)> {
        WitnessAddrsIter {
            root_level,
            current: Address::from(self),
            ommer_count: 0,
        }
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

/// The address of an internal node of the Merkle tree.
/// When `level == 0`, the index has the same value as the
/// position.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Address {
    level: Level,
    index: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// The sibling to the address can be derived from the incremental frontier
    /// at the contained ommer index
    Past(usize),
    /// The sibling to the address must be obtained from values discovered by
    /// the addition of more nodes to the tree
    Future,
}

impl Address {
    pub fn from_parts(level: Level, index: usize) -> Self {
        Address { level, index }
    }

    pub fn position_range(&self) -> Range<Position> {
        Range {
            start: (self.index << self.level.0).try_into().unwrap(),
            end: ((self.index + 1) << self.level.0).try_into().unwrap(),
        }
    }

    pub fn level(&self) -> Level {
        self.level
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn parent(&self) -> Address {
        Address {
            level: self.level + 1,
            index: self.index >> 1,
        }
    }

    pub fn sibling(&self) -> Address {
        Address {
            level: self.level,
            index: if self.index & 0x1 == 0 {
                self.index + 1
            } else {
                self.index - 1
            },
        }
    }

    pub fn is_complete_node(&self) -> bool {
        self.index & 0x1 == 1
    }

    pub fn current_incomplete(&self) -> Address {
        // find the first zero bit in the index, searching from the least significant bit
        let mut index = self.index;
        for level in self.level.0.. {
            if index & 0x1 == 1 {
                index >>= 1;
            } else {
                return Address {
                    level: Level(level),
                    index,
                };
            }
        }

        unreachable!("The loop will always terminate via return in at most 64 iterations.")
    }

    pub fn next_incomplete_parent(&self) -> Address {
        if self.is_complete_node() {
            self.current_incomplete()
        } else {
            let complete = Address {
                level: self.level,
                index: self.index + 1,
            };
            complete.current_incomplete()
        }
    }
}

impl From<Position> for Address {
    fn from(p: Position) -> Self {
        Address {
            level: 0.into(),
            index: p.into(),
        }
    }
}

impl<'a> From<&'a Position> for Address {
    fn from(p: &'a Position) -> Self {
        Address {
            level: 0.into(),
            index: (*p).into(),
        }
    }
}

impl From<Address> for Option<Position> {
    fn from(addr: Address) -> Self {
        if addr.level == 0.into() {
            Some(addr.index.into())
        } else {
            None
        }
    }
}

impl<'a> From<&'a Address> for Option<Position> {
    fn from(addr: &'a Address) -> Self {
        if addr.level == 0.into() {
            Some(addr.index.into())
        } else {
            None
        }
    }
}

#[must_use = "iterators are lazy and do nothing unless consumed"]
pub(crate) struct WitnessAddrsIter {
    root_level: Level,
    current: Address,
    ommer_count: usize,
}

impl Iterator for WitnessAddrsIter {
    type Item = (Address, Source);

    fn next(&mut self) -> Option<(Address, Source)> {
        if self.current.level() < self.root_level {
            let current = self.current;
            let source = if current.is_complete_node() {
                Source::Past(self.ommer_count)
            } else {
                Source::Future
            };

            self.current = current.parent();
            if matches!(source, Source::Past(_)) {
                self.ommer_count += 1;
            }

            Some((current.sibling(), source))
        } else {
            None
        }
    }
}

/// A trait describing the operations that make a value  suitable for inclusion in
/// an incremental merkle tree.
pub trait Hashable: Sized {
    fn empty_leaf() -> Self;

    fn combine(level: Level, a: &Self, b: &Self) -> Self;

    fn empty_root(level: Level) -> Self {
        Level(0)
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

/// A Merkle tree that supports incremental appends, marking of
/// leaf nodes for construction of witnesses, checkpoints and rollbacks.
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
    /// a witness for it.
    fn get_marked_leaf(&self, position: Position) -> Option<&H>;

    /// Marks the current leaf as one for which we're interested in producing
    /// a witness. Returns an optional value containing the
    /// current position if successful or if the current value was already
    /// marked, or None if the tree is empty.
    fn mark(&mut self) -> Option<Position>;

    /// Return a set of all the positions for which we have marked.
    fn marked_positions(&self) -> BTreeSet<Position>;

    /// Obtains the root of the Merkle tree at the specified checkpoint depth
    /// by hashing against empty nodes up to the maximum height of the tree.
    /// Returns `None` if there are not enough checkpoints available to reach the
    /// requested checkpoint depth.
    fn root(&self, checkpoint_depth: usize) -> Option<H>;

    /// Obtains a witness to the value at the specified position,
    /// as of the tree state corresponding to the given root.
    /// Returns `None` if there is no available witness to that
    /// position or if the root does not correspond to a checkpointed
    /// root of the tree.
    fn witness(&self, position: Position, as_of_root: &H) -> Option<Vec<H>>;

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a mark for. Returns true if successful and
    /// false if we were already not maintaining a mark at this position.
    fn remove_mark(&mut self, position: Position) -> bool;

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
    /// because it is associated with checkpoints or marks that
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
    use super::{Address, Hashable, Level, Position, Source, Tree};

    #[test]
    fn position_is_complete_subtree() {
        assert!(Position(0).is_complete_subtree(Level(0)));
        assert!(Position(1).is_complete_subtree(Level(1)));
        assert!(!Position(2).is_complete_subtree(Level(1)));
        assert!(!Position(2).is_complete_subtree(Level(2)));
        assert!(Position(3).is_complete_subtree(Level(2)));
        assert!(!Position(4).is_complete_subtree(Level(2)));
        assert!(Position(7).is_complete_subtree(Level(3)));
        assert!(Position(u32::MAX as usize).is_complete_subtree(Level(32)));
    }

    #[test]
    fn position_past_ommer_count() {
        assert_eq!(0, Position(0).past_ommer_count());
        assert_eq!(1, Position(1).past_ommer_count());
        assert_eq!(1, Position(2).past_ommer_count());
        assert_eq!(2, Position(3).past_ommer_count());
        assert_eq!(1, Position(4).past_ommer_count());
        assert_eq!(3, Position(7).past_ommer_count());
        assert_eq!(1, Position(8).past_ommer_count());
    }

    #[test]
    fn position_root_level() {
        assert_eq!(Level(0), Position(0).root_level());
        assert_eq!(Level(1), Position(1).root_level());
        assert_eq!(Level(2), Position(2).root_level());
        assert_eq!(Level(2), Position(3).root_level());
        assert_eq!(Level(3), Position(4).root_level());
        assert_eq!(Level(3), Position(7).root_level());
        assert_eq!(Level(4), Position(8).root_level());
    }

    #[test]
    fn current_incomplete() {
        let addr = |l, i| Address::from_parts(Level(l), i);
        assert_eq!(addr(0, 0), addr(0, 0).current_incomplete());
        assert_eq!(addr(1, 0), addr(0, 1).current_incomplete());
        assert_eq!(addr(0, 2), addr(0, 2).current_incomplete());
        assert_eq!(addr(2, 0), addr(0, 3).current_incomplete());
    }

    #[test]
    fn next_incomplete_parent() {
        let addr = |l, i| Address::from_parts(Level(l), i);
        assert_eq!(addr(1, 0), addr(0, 0).next_incomplete_parent());
        assert_eq!(addr(1, 0), addr(0, 1).next_incomplete_parent());
        assert_eq!(addr(2, 0), addr(0, 2).next_incomplete_parent());
        assert_eq!(addr(2, 0), addr(0, 3).next_incomplete_parent());
        assert_eq!(addr(3, 0), addr(2, 0).next_incomplete_parent());
        assert_eq!(addr(1, 2), addr(0, 4).next_incomplete_parent());
        assert_eq!(addr(3, 0), addr(1, 2).next_incomplete_parent());
    }

    #[test]
    fn position_witness_addrs() {
        use Source::*;
        let path_elem = |l, i, s| (Address::from_parts(Level(l), i), s);
        assert_eq!(
            vec![path_elem(0, 1, Future), path_elem(1, 1, Future)],
            Position(0).witness_addrs(Level(2)).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![path_elem(0, 3, Future), path_elem(1, 0, Past(0))],
            Position(2).witness_addrs(Level(2)).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                path_elem(0, 2, Past(0)),
                path_elem(1, 0, Past(1)),
                path_elem(2, 1, Future)
            ],
            Position(3).witness_addrs(Level(3)).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                path_elem(0, 5, Future),
                path_elem(1, 3, Future),
                path_elem(2, 0, Past(0)),
                path_elem(3, 1, Future)
            ],
            Position(4).witness_addrs(Level(4)).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                path_elem(0, 7, Future),
                path_elem(1, 2, Past(0)),
                path_elem(2, 0, Past(1)),
                path_elem(3, 1, Future)
            ],
            Position(6).witness_addrs(Level(4)).collect::<Vec<_>>()
        );
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

        fn combine(_level: Level, a: &Self, b: &Self) -> Self {
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

        fn combine(_: Level, a: &Self, b: &Self) -> Self {
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
        t.mark();
        t.append(&"a".to_string());
        t.append(&"a".to_string());
        t.append(&"a".to_string());
        assert_eq!(t.root(0).unwrap(), "aaaa____________");
    }

    pub(crate) fn check_witnesss<T: Tree<String> + std::fmt::Debug, F: Fn(usize) -> T>(
        new_tree: F,
    ) {
        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.mark();
        assert_eq!(
            tree.witness(Position(0), &tree.root(0).unwrap()),
            Some(vec![
                "_".to_string(),
                "__".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"b".to_string());
        assert_eq!(
            tree.witness(0.into(), &tree.root(0).unwrap()),
            Some(vec![
                "b".to_string(),
                "__".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"c".to_string());
        tree.mark();
        assert_eq!(
            tree.witness(Position(2), &tree.root(0).unwrap()),
            Some(vec![
                "_".to_string(),
                "ab".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"d".to_string());
        assert_eq!(
            tree.witness(Position(2), &tree.root(0).unwrap()),
            Some(vec![
                "d".to_string(),
                "ab".to_string(),
                "____".to_string(),
                "________".to_string()
            ])
        );

        tree.append(&"e".to_string());
        assert_eq!(
            tree.witness(Position(2), &tree.root(0).unwrap()),
            Some(vec![
                "d".to_string(),
                "ab".to_string(),
                "e___".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.mark();
        for c in 'b'..'h' {
            tree.append(&c.to_string());
        }
        tree.mark();
        tree.append(&"h".to_string());

        assert_eq!(
            tree.witness(0.into(), &tree.root(0).unwrap()),
            Some(vec![
                "b".to_string(),
                "cd".to_string(),
                "efgh".to_string(),
                "________".to_string()
            ])
        );

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.mark();
        tree.append(&"b".to_string());
        tree.append(&"c".to_string());
        tree.append(&"d".to_string());
        tree.mark();
        tree.append(&"e".to_string());
        tree.mark();
        tree.append(&"f".to_string());
        tree.mark();
        tree.append(&"g".to_string());

        assert_eq!(
            tree.witness(Position(5), &tree.root(0).unwrap()),
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
        tree.mark();
        tree.append(&'l'.to_string());

        assert_eq!(
            tree.witness(Position(10), &tree.root(0).unwrap()),
            Some(vec![
                "l".to_string(),
                "ij".to_string(),
                "____".to_string(),
                "abcdefgh".to_string()
            ])
        );

        let mut tree = new_tree(100);
        tree.append(&'a'.to_string());
        tree.mark();
        tree.checkpoint();
        assert!(tree.rewind());
        for c in 'b'..'f' {
            tree.append(&c.to_string());
        }
        tree.mark();
        for c in 'f'..'i' {
            tree.append(&c.to_string());
        }

        assert_eq!(
            tree.witness(0.into(), &tree.root(0).unwrap()),
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
        tree.mark();
        tree.append(&'d'.to_string());
        tree.append(&'e'.to_string());
        tree.append(&'f'.to_string());
        tree.append(&'g'.to_string());
        tree.mark();
        tree.checkpoint();
        tree.append(&'h'.to_string());
        assert!(tree.rewind());

        assert_eq!(
            tree.witness(Position(2), &tree.root(0).unwrap()),
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
        tree.mark();
        assert_eq!(tree.witness(Position(0), &tree.root(0).unwrap()), None);

        let mut tree = new_tree(100);
        for c in 'a'..'n' {
            tree.append(&c.to_string());
        }
        tree.mark();
        tree.append(&'n'.to_string());
        tree.mark();
        tree.append(&'o'.to_string());
        tree.append(&'p'.to_string());

        assert_eq!(
            tree.witness(Position(12), &tree.root(0).unwrap()),
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
            .chain(Some(Mark))
            .chain(Some(Append('m'.to_string())))
            .chain(Some(Append('n'.to_string())))
            .chain(Some(Authpath(11usize.into(), 0)))
            .collect::<Vec<_>>();

        let mut tree = new_tree(100);
        assert_eq!(
            Operation::apply_all(&ops, &mut tree),
            Some((
                Position(11),
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
        t.mark();
        assert!(t.rewind());
        assert_eq!(Some(Position(0)), t.current_position());

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.mark();
        t.checkpoint();
        assert!(t.rewind());

        let mut t = new_tree(100);
        t.append(&"a".to_string());
        t.checkpoint();
        t.mark();
        t.append(&"a".to_string());
        assert!(t.rewind());
        assert_eq!(Some(Position(0)), t.current_position());

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

    pub(crate) fn check_rewind_remove_mark<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.mark();
        tree.checkpoint();
        assert!(tree.rewind());
        assert!(tree.remove_mark(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.checkpoint();
        tree.mark();
        assert!(tree.rewind());
        assert!(!tree.remove_mark(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.mark();
        tree.checkpoint();
        assert!(tree.remove_mark(0usize.into()));
        assert!(tree.rewind());
        assert!(tree.remove_mark(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"e".to_string());
        tree.mark();
        assert!(tree.remove_mark(0usize.into()));
        tree.checkpoint();
        assert!(tree.rewind());
        assert!(!tree.remove_mark(0usize.into()));

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        assert!(!tree.remove_mark(0usize.into()));
        tree.checkpoint();
        assert!(tree.mark().is_some());
        assert!(tree.rewind());

        let mut tree = new_tree(100);
        tree.append(&"a".to_string());
        tree.checkpoint();
        assert!(tree.mark().is_some());
        assert!(tree.remove_mark(0usize.into()));
        assert!(tree.rewind());
        assert!(!tree.remove_mark(0usize.into()));

        // The following check_operations tests cover errors where the
        // test framework itself previously did not correctly handle
        // chain state restoration.

        let samples = vec![
            vec![append("x"), Checkpoint, Mark, Rewind, unmark(0)],
            vec![append("d"), Checkpoint, Mark, unmark(0), Rewind, unmark(0)],
            vec![
                append("o"),
                Checkpoint,
                Mark,
                Checkpoint,
                unmark(0),
                Rewind,
                Rewind,
            ],
            vec![
                append("s"),
                Mark,
                append("m"),
                Checkpoint,
                unmark(0),
                Rewind,
                unmark(0),
                unmark(0),
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

        fn get_marked_leaf(&self, position: Position) -> Option<&H> {
            let a = self.inefficient.get_marked_leaf(position);
            let b = self.efficient.get_marked_leaf(position);
            assert_eq!(a, b);
            a
        }

        fn mark(&mut self) -> Option<Position> {
            let a = self.inefficient.mark();
            let b = self.efficient.mark();
            assert_eq!(a, b);
            let apos = self.inefficient.marked_positions();
            let bpos = self.efficient.marked_positions();
            assert_eq!(apos, bpos);
            a
        }

        fn marked_positions(&self) -> BTreeSet<Position> {
            let a = self.inefficient.marked_positions();
            let b = self.efficient.marked_positions();
            assert_eq!(a, b);
            a
        }

        fn witness(&self, position: Position, as_of_root: &H) -> Option<Vec<H>> {
            let a = self.inefficient.witness(position, as_of_root);
            let b = self.efficient.witness(position, as_of_root);
            assert_eq!(a, b);
            a
        }

        fn remove_mark(&mut self, position: Position) -> bool {
            let a = self.inefficient.remove_mark(position);
            let b = self.efficient.remove_mark(position);
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
        Mark,
        MarkedLeaf(Position),
        MarkedPositions,
        Unmark(Position),
        Checkpoint,
        Rewind,
        Authpath(Position, usize),
        GarbageCollect,
    }

    use Operation::*;

    fn append(x: &str) -> Operation<String> {
        Operation::Append(x.to_string())
    }

    fn unmark(pos: usize) -> Operation<String> {
        Operation::Unmark(Position(pos))
    }

    fn authpath(pos: usize, depth: usize) -> Operation<String> {
        Operation::Authpath(Position(pos), depth)
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
                Mark => {
                    assert!(tree.mark().is_some(), "mark failed");
                    None
                }
                MarkedLeaf(_) => None,
                MarkedPositions => None,
                Unmark(p) => {
                    assert!(tree.remove_mark(*p), "remove mark failed");
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
                    .and_then(|root| tree.witness(*p, &root))
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

    pub(crate) fn compute_root_from_witness<H: Hashable>(
        value: H,
        position: Position,
        path: &[H],
    ) -> H {
        let mut cur = value;
        let mut lvl = 0.into();
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
    fn test_compute_root_from_witness() {
        let expected = SipHashable::combine(
            <Level>::from(2),
            &SipHashable::combine(
                Level(1),
                &SipHashable::combine(0.into(), &SipHashable(0), &SipHashable(1)),
                &SipHashable::combine(0.into(), &SipHashable(2), &SipHashable(3)),
            ),
            &SipHashable::combine(
                Level(1),
                &SipHashable::combine(0.into(), &SipHashable(4), &SipHashable(5)),
                &SipHashable::combine(0.into(), &SipHashable(6), &SipHashable(7)),
            ),
        );

        assert_eq!(
            compute_root_from_witness::<SipHashable>(
                SipHashable(0),
                0.into(),
                &[
                    SipHashable(1),
                    SipHashable::combine(0.into(), &SipHashable(2), &SipHashable(3)),
                    SipHashable::combine(
                        Level(1),
                        &SipHashable::combine(0.into(), &SipHashable(4), &SipHashable(5)),
                        &SipHashable::combine(0.into(), &SipHashable(6), &SipHashable(7))
                    )
                ]
            ),
            expected
        );

        assert_eq!(
            compute_root_from_witness(
                SipHashable(4),
                <Position>::from(4),
                &[
                    SipHashable(5),
                    SipHashable::combine(0.into(), &SipHashable(6), &SipHashable(7)),
                    SipHashable::combine(
                        Level(1),
                        &SipHashable::combine(0.into(), &SipHashable(0), &SipHashable(1)),
                        &SipHashable::combine(0.into(), &SipHashable(2), &SipHashable(3))
                    )
                ]
            ),
            expected
        );
    }

    #[test]
    fn test_witness_consistency() {
        let samples = vec![
            // Reduced examples
            vec![append("a"), append("b"), Checkpoint, Mark, authpath(0, 1)],
            vec![append("c"), append("d"), Mark, Checkpoint, authpath(1, 1)],
            vec![append("e"), Checkpoint, Mark, append("f"), authpath(0, 1)],
            vec![
                append("g"),
                Mark,
                Checkpoint,
                unmark(0),
                append("h"),
                authpath(0, 0),
            ],
            vec![
                append("i"),
                Checkpoint,
                Mark,
                unmark(0),
                append("j"),
                authpath(0, 0),
            ],
            vec![
                append("i"),
                Mark,
                append("j"),
                Checkpoint,
                append("k"),
                authpath(0, 1),
            ],
            vec![
                append("l"),
                Checkpoint,
                Mark,
                Checkpoint,
                append("m"),
                Checkpoint,
                authpath(0, 2),
            ],
            vec![Checkpoint, append("n"), Mark, authpath(0, 1)],
            vec![
                append("a"),
                Mark,
                Checkpoint,
                unmark(0),
                Checkpoint,
                append("b"),
                authpath(0, 1),
            ],
            vec![
                append("a"),
                Mark,
                append("b"),
                unmark(0),
                Checkpoint,
                authpath(0, 0),
            ],
            vec![
                append("a"),
                Mark,
                Checkpoint,
                unmark(0),
                Checkpoint,
                Rewind,
                append("b"),
                authpath(0, 0),
            ],
            vec![
                append("a"),
                Mark,
                Checkpoint,
                Checkpoint,
                Rewind,
                append("a"),
                unmark(0),
                authpath(0, 1),
            ],
            // Unreduced examples
            vec![
                append("o"),
                append("p"),
                Mark,
                append("q"),
                Checkpoint,
                unmark(1),
                authpath(1, 1),
            ],
            vec![
                append("r"),
                append("s"),
                append("t"),
                Mark,
                Checkpoint,
                unmark(2),
                Checkpoint,
                authpath(2, 2),
            ],
            vec![
                append("u"),
                Mark,
                append("v"),
                append("w"),
                Checkpoint,
                unmark(0),
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
    fn test_rewind_remove_mark_consistency() {
        let samples = vec![
            vec![append("x"), Checkpoint, Mark, Rewind, unmark(0)],
            vec![append("d"), Checkpoint, Mark, unmark(0), Rewind, unmark(0)],
            vec![
                append("o"),
                Checkpoint,
                Mark,
                Checkpoint,
                unmark(0),
                Rewind,
                Rewind,
            ],
            vec![
                append("s"),
                Mark,
                append("m"),
                Checkpoint,
                unmark(0),
                Rewind,
                unmark(0),
                unmark(0),
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
            Just(Operation::Mark),
            prop_oneof![
                Just(Operation::CurrentLeaf),
                Just(Operation::CurrentPosition),
                Just(Operation::MarkedPositions),
            ],
            Just(Operation::GarbageCollect),
            pos_gen
                .clone()
                .prop_map(|i| Operation::MarkedLeaf(Position(i))),
            pos_gen.clone().prop_map(|i| Operation::Unmark(Position(i))),
            Just(Operation::Checkpoint),
            Just(Operation::Rewind),
            pos_gen.prop_flat_map(
                |i| (0usize..10).prop_map(move |depth| Operation::Authpath(Position(i), depth))
            ),
        ]
    }

    pub fn apply_operation<H, T: Tree<H>>(tree: &mut T, op: Operation<H>) {
        match op {
            Append(value) => {
                tree.append(&value);
            }
            Mark => {
                tree.mark();
            }
            Unmark(position) => {
                tree.remove_mark(position);
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
            MarkedLeaf(_) => {}
            MarkedPositions => {}
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
                Mark => {
                    if tree.mark().is_some() {
                        prop_assert!(tree_size != 0);
                    } else {
                        prop_assert_eq!(tree_size, 0);
                    }
                }
                MarkedLeaf(position) => {
                    if tree.get_marked_leaf(*position).is_some() {
                        prop_assert!(<usize>::from(*position) < tree_size);
                    }
                }
                Unmark(position) => {
                    tree.remove_mark(*position);
                }
                MarkedPositions => {}
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
                    if let Some(path) = tree.root(*depth).and_then(|r| tree.witness(*position, &r))
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
                                &compute_root_from_witness(value, *position, &path),
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
