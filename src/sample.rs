use super::{Altitude, Frontier, Hashable, Position, Recording, Tree};
/// Sample implementation of the Tree interface.
use std::convert::TryInto;

#[derive(Clone, Debug)]
pub struct TreeState<H: Hashable> {
    leaves: Vec<H>,
    current_offset: usize,
    witnesses: Vec<(Position, H)>,
    depth: usize,
}

impl<H: Hashable + Clone> TreeState<H> {
    /// Creates a new, empty binary tree of specified depth.
    #[cfg(test)]
    pub fn new(depth: usize) -> Self {
        Self {
            leaves: vec![H::empty_leaf(); 1 << depth],
            current_offset: 0,
            witnesses: vec![],
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
    fn current_leaf(&self) -> Option<(Position, H)> {
        self.current_position()
            .map(|p| (p, self.leaves[<usize>::from(p)].clone()))
    }

    /// Returns whether a leaf with the specified position and value has been witnessed
    fn is_witnessed(&self, position: Position, value: &H) -> bool {
        self.witnesses
            .iter()
            .any(|(pos, v)| pos == &position && v == value)
    }

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns the current position and leaf value if the tree
    /// is non-empty.
    fn witness(&mut self) -> Option<(Position, H)> {
        self.current_leaf().map(|(pos, value)| {
            if !self.is_witnessed(pos, &value) {
                self.witnesses.push((pos, value.clone()));
            }
            (pos, value)
        })
    }

    /// Obtains an authentication path to the value specified in the tree.
    /// Returns `None` if there is no available authentication path to the
    /// specified value.
    fn authentication_path(&self, position: Position, value: &H) -> Option<Vec<H>> {
        self.witnesses
            .iter()
            .find(|(pos, v)| pos == &position && v == value)
            .map(|_| {
                let mut path = vec![];

                let mut leaf_idx: usize = position.into();
                for bit in 0..self.depth {
                    leaf_idx ^= 1 << bit;
                    path.push(lazy_root::<H>(
                        self.leaves[leaf_idx..][0..(1 << bit)].to_vec(),
                    ));
                    leaf_idx &= usize::MAX << (bit + 1);
                }

                path
            })
    }

    /// Marks the specified tree state value as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if the value is not a known witness.
    fn remove_witness(&mut self, position: Position, value: &H) -> bool {
        if let Some((witness_index, _)) = self
            .witnesses
            .iter()
            .enumerate()
            .find(|(_i, (pos, v))| pos == &position && v == value)
        {
            self.witnesses.remove(witness_index);
            true
        } else {
            false
        }
    }

    /// Start a recording of append operations performed on a tree.
    fn recording(&self) -> CompleteRecording<H> {
        CompleteRecording {
            start_position: self.current_offset,
            current_offset: self.current_offset,
            depth: self.depth,
            appends: vec![],
        }
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the recording is incompatible with the current tree state.
    fn play(&mut self, recording: &CompleteRecording<H>) -> bool {
        #[allow(clippy::suspicious_operation_groupings)]
        if recording.start_position == self.current_offset && self.depth == recording.depth {
            for val in recording.appends.iter() {
                self.append(val);
            }
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompleteTree<H: Hashable> {
    tree_state: TreeState<H>,
    checkpoints: Vec<(TreeState<H>, bool)>,
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
    type Recording = CompleteRecording<H>;

    /// Returns the most recently appended leaf value.
    fn current_position(&self) -> Option<Position> {
        self.tree_state.current_position()
    }

    /// Returns the leaf most recently appended to the tree
    fn current_leaf(&self) -> Option<(Position, H)> {
        self.tree_state.current_leaf()
    }

    /// Returns whether a leaf with the specified value has been witnessed
    fn is_witnessed(&self, position: Position, value: &H) -> bool {
        self.tree_state.is_witnessed(position, value)
    }

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns the current position and leaf value if the tree
    /// is non-empty.
    fn witness(&mut self) -> Option<(Position, H)> {
        self.tree_state.witness()
    }

    /// Obtains an authentication path to the value specified in the tree.
    /// Returns `None` if there is no available authentication path to the
    /// specified value.
    fn authentication_path(&self, position: Position, value: &H) -> Option<Vec<H>> {
        self.tree_state.authentication_path(position, value)
    }

    /// Marks the specified tree state value as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if the value is not a known witness.
    fn remove_witness(&mut self, position: Position, value: &H) -> bool {
        self.tree_state.remove_witness(position, value)
    }

    /// Marks the current tree state as a checkpoint if it is not already a
    /// checkpoint.
    fn checkpoint(&mut self) {
        let is_witnessed = self
            .tree_state
            .current_leaf()
            .into_iter()
            .any(|(p, l)| self.tree_state.is_witnessed(p, &l));
        self.checkpoints
            .push((self.tree_state.clone(), is_witnessed));
        if self.checkpoints.len() > self.max_checkpoints {
            self.drop_oldest_checkpoint();
        }
    }

    /// Rewinds the tree state to the previous checkpoint. This function will
    /// fail and return false if there is no previous checkpoint or in the event
    /// witness data would be destroyed in the process.
    fn rewind(&mut self) -> bool {
        if let Some((checkpointed_state, is_witnessed)) = self.checkpoints.pop() {
            // if there are any witnessed leaves in the current tree state
            // that would be removed, we don't rewind
            if self.tree_state.witnesses.iter().any(|&(pos, _)| {
                let offset: usize = (pos + 1).try_into().unwrap();
                offset > checkpointed_state.current_offset
                    || (offset == checkpointed_state.current_offset && !is_witnessed)
            }) {
                self.checkpoints.push((checkpointed_state, is_witnessed));
                false
            } else {
                self.tree_state = checkpointed_state;
                true
            }
        } else {
            false
        }
    }

    /// Start a recording of append operations performed on a tree.
    fn recording(&self) -> CompleteRecording<H> {
        self.tree_state.recording()
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the recording is incompatible with the current tree state.
    fn play(&mut self, recording: &CompleteRecording<H>) -> bool {
        self.tree_state.play(recording)
    }
}

#[derive(Clone)]
pub struct CompleteRecording<H: Hashable> {
    start_position: usize,
    current_offset: usize,
    depth: usize,
    appends: Vec<H>,
}

impl<H: Hashable + Clone> Recording<H> for CompleteRecording<H> {
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H) -> bool {
        if self.current_offset == (1 << self.depth) {
            false
        } else {
            self.appends.push(value.clone());
            self.current_offset += 1;

            true
        }
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the provided recording is incompatible with `Self`.
    fn play(&mut self, recording: &Self) -> bool {
        #[allow(clippy::suspicious_operation_groupings)]
        if self.current_offset == recording.start_position && self.depth == recording.depth {
            self.appends.extend_from_slice(&recording.appends);
            self.current_offset = recording.current_offset;
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
            let path = tree.authentication_path(position, &SipHashable(i)).unwrap();
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
