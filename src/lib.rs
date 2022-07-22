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

#[cfg(any(bench, test, feature = "test-dependencies"))]
pub mod testing;

#[cfg(test)]
pub(crate) mod tests {
    use super::{Address, Level, Position, Source};

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
}
