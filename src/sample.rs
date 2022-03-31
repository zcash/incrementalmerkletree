//! Sample implementation of the Tree interface.
use super::{Altitude, Frontier, Hashable, Position, Tree};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct TreeState<H: Hashable> {
    leaves: Vec<H>,
    current_offset: usize,
    witnesses: BTreeSet<Position>,
    depth: usize,
}

impl<H: Hashable + Clone> TreeState<H> {
    /// Creates a new, empty binary tree of specified depth.
    #[cfg(test)]
    pub fn new(depth: usize) -> Self {
        Self {
            leaves: vec![H::empty_leaf(); 1 << depth],
            current_offset: 0,
            witnesses: BTreeSet::new(),
            depth,
        }
    }
}

impl<H: Hashable + Clone> Frontier<H> for TreeState<H> {
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H) -> bool {
        if self.current_offset == (1 << self.depth) {
            false
        } else {
            self.leaves[self.current_offset] = value.clone();
            self.current_offset += 1;
            true
        }
    }

    /// Obtains the current root of this Merkle tree.
    fn root(&self) -> H {
        lazy_root(self.leaves.clone())
    }
}

impl<H: Hashable + PartialEq + Clone> TreeState<H> {
    fn current_position(&self) -> Option<Position> {
        if self.current_offset == 0 {
            None
        } else {
            Some((self.current_offset - 1).into())
        }
    }

    /// Returns the leaf most recently appended to the tree
    fn current_leaf(&self) -> Option<&H> {
        self.current_position()
            .map(|p| &self.leaves[<usize>::from(p)])
    }

    /// Returns the leaf at the specified position if the tree can produce
    /// an authentication path for it.
    fn get_witnessed_leaf(&self, position: Position) -> Option<&H> {
        if self.witnesses.contains(&position) {
            self.leaves.get(<usize>::from(position))
        } else {
            None
        }
    }

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns the current position if the tree is non-empty.
    fn witness(&mut self) -> Option<Position> {
        self.current_position().map(|pos| {
            if !self.witnesses.contains(&pos) {
                self.witnesses.insert(pos);
            }
            pos
        })
    }

    /// Obtains an authentication path to the value at the specified position.
    /// Returns `None` if there is no available authentication path to that
    /// value.
    fn authentication_path(&self, position: Position) -> Option<Vec<H>> {
        if self.witnesses.contains(&position) {
            let mut path = vec![];

            let mut leaf_idx: usize = position.into();
            for bit in 0..self.depth {
                leaf_idx ^= 1 << bit;
                path.push(lazy_root::<H>(
                    self.leaves[leaf_idx..][0..(1 << bit)].to_vec(),
                ));
                leaf_idx &= usize::MAX << (bit + 1);
            }

            Some(path)
        } else {
            None
        }
    }

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if we were already not maintaining a witness at this position.
    fn remove_witness(&mut self, position: Position) -> bool {
        self.witnesses.remove(&position)
    }
}

#[derive(Clone, Debug)]
pub struct CompleteTree<H: Hashable> {
    tree_state: TreeState<H>,
    checkpoints: Vec<TreeState<H>>,
    max_checkpoints: usize,
}

impl<H: Hashable + Clone> CompleteTree<H> {
    /// Creates a new, empty binary tree of specified depth.
    #[cfg(test)]
    pub fn new(depth: usize, max_checkpoints: usize) -> Self {
        CompleteTree {
            tree_state: TreeState::new(depth),
            checkpoints: vec![],
            max_checkpoints,
        }
    }
}

impl<H: Hashable + Clone> Frontier<H> for CompleteTree<H> {
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H) -> bool {
        self.tree_state.append(value)
    }

    /// Obtains the current root of this Merkle tree.
    fn root(&self) -> H {
        self.tree_state.root()
    }
}

impl<H: Hashable + PartialEq + Clone> CompleteTree<H> {
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
}

impl<H: Hashable + PartialEq + Clone> Tree<H> for CompleteTree<H> {
    /// Returns the most recently appended leaf value.
    fn current_position(&self) -> Option<Position> {
        self.tree_state.current_position()
    }

    /// Returns the leaf most recently appended to the tree
    fn current_leaf(&self) -> Option<&H> {
        self.tree_state.current_leaf()
    }

    /// Returns the leaf at the specified position if the tree can produce
    /// an authentication path for it.
    fn get_witnessed_leaf(&self, position: Position) -> Option<&H> {
        self.tree_state.get_witnessed_leaf(position)
    }

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns the current position if the tree is non-empty.
    fn witness(&mut self) -> Option<Position> {
        self.tree_state.witness()
    }

    /// Obtains an authentication path to the value at the specified position.
    /// Returns `None` if there is no available authentication path to that
    /// value.
    fn authentication_path(&self, position: Position) -> Option<Vec<H>> {
        self.tree_state.authentication_path(position)
    }

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if we were already not maintaining a witness at this position.
    fn remove_witness(&mut self, position: Position) -> bool {
        self.tree_state.remove_witness(position)
    }

    /// Marks the current tree state as a checkpoint if it is not already a
    /// checkpoint.
    fn checkpoint(&mut self) {
        self.checkpoints.push(self.tree_state.clone());
        if self.checkpoints.len() > self.max_checkpoints {
            self.drop_oldest_checkpoint();
        }
    }

    /// Rewinds the tree state to the previous checkpoint. This function will
    /// return false and leave the tree unmodified if no checkpoints exist.
    fn rewind(&mut self) -> bool {
        if let Some(checkpointed_state) = self.checkpoints.pop() {
            self.tree_state = checkpointed_state;
            true
        } else {
            false
        }
    }
}

pub(crate) fn lazy_root<H: Hashable + Clone>(mut leaves: Vec<H>) -> H {
    //leaves are always at level zero, so we start there.
    let mut level = Altitude::zero();
    while leaves.len() != 1 {
        leaves = leaves
            .iter()
            .enumerate()
            .filter(|(i, _)| (i % 2) == 0)
            .map(|(_, a)| a)
            .zip(
                leaves
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i % 2) == 1)
                    .map(|(_, b)| b),
            )
            .map(|(a, b)| H::combine(level, a, b))
            .collect();
        level = level + 1;
    }

    leaves[0].clone()
}

#[cfg(test)]
mod tests {
    use crate::tests::{compute_root_from_auth_path, SipHashable};
    use crate::{Altitude, Frontier, Hashable, Position, Tree};
    use std::convert::TryFrom;

    use super::CompleteTree;

    #[test]
    fn correct_empty_root() {
        const DEPTH: u8 = 5;
        let mut expected = SipHashable(0u64);
        for lvl in 0u8..DEPTH {
            expected = SipHashable::combine(lvl.into(), &expected, &expected);
        }

        let tree = CompleteTree::<SipHashable>::new(DEPTH as usize, 100);
        assert_eq!(tree.root(), expected);
    }

    #[test]
    fn correct_root() {
        const DEPTH: usize = 3;
        let values = (0..(1 << DEPTH)).into_iter().map(SipHashable);

        let mut tree = CompleteTree::<SipHashable>::new(DEPTH, 100);
        for value in values {
            assert!(tree.append(&value));
        }
        assert!(!tree.append(&SipHashable(0)));

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

        assert_eq!(tree.root(), expected);
    }

    #[test]
    fn root_hashes() {
        crate::tests::check_root_hashes(|max_c| CompleteTree::<String>::new(4, max_c));
    }

    #[test]
    fn auth_paths() {
        crate::tests::check_auth_paths(|max_c| CompleteTree::<String>::new(4, max_c));
    }

    #[test]
    fn correct_auth_path() {
        const DEPTH: usize = 3;
        let values = (0..(1 << DEPTH)).into_iter().map(SipHashable);

        let mut tree = CompleteTree::<SipHashable>::new(DEPTH, 100);
        for value in values {
            assert!(tree.append(&value));
            tree.witness();
        }
        assert!(!tree.append(&SipHashable(0)));

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

        assert_eq!(tree.root(), expected);

        for i in 0u64..(1 << DEPTH) {
            let position = Position::try_from(i).unwrap();
            let path = tree.authentication_path(position).unwrap();
            assert_eq!(
                compute_root_from_auth_path(SipHashable(i), position, &path),
                expected
            );
        }
    }

    #[test]
    fn checkpoint_rewind() {
        crate::tests::check_checkpoint_rewind(|max_c| CompleteTree::<String>::new(4, max_c));
    }

    #[test]
    fn rewind_remove_witness() {
        crate::tests::check_rewind_remove_witness(|max_c| CompleteTree::<String>::new(4, max_c));
    }
}
