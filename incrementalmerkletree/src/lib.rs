//! Common types and utilities used in incremental Merkle tree implementations.

use serde::{Deserialize, Serialize};
use std::convert::{TryFrom, TryInto};
use std::num::TryFromIntError;
use std::ops::{Add, AddAssign, Range};

#[cfg(feature = "test-dependencies")]
pub mod testing;

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

/// The address of an internal node of the Merkle tree.
/// When `level == 0`, the index has the same value as the
/// position.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Address {
    level: Level,
    index: usize,
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

/// A trait describing the operations that make a type suitable for use as
/// a leaf or node value in a merkle tree.
pub trait Hashable: Sized {
    fn empty_leaf() -> Self;

    fn combine(level: Level, a: &Self, b: &Self) -> Self;

    fn empty_root(level: Level) -> Self {
        Level::from(0)
            .iter_to(level)
            .fold(Self::empty_leaf(), |v, lvl| Self::combine(lvl, &v, &v))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Address, Level, Position};

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
}
