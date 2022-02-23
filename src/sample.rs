/// Sample implementation of the Tree interface.
use super::{Altitude, Frontier, Hashable, Position, Recording, Tree};

#[derive(Clone)]
pub struct CompleteTree<H: Hashable> {
    leaves: Vec<H>,
    current_position: usize,
    witnesses: Vec<(usize, H)>,
    checkpoints: Vec<usize>,
    depth: usize,
    max_checkpoints: usize,
}

impl<H: Hashable + Clone> CompleteTree<H> {
    /// Creates a new, empty binary tree of specified depth.
    #[cfg(test)]
    pub fn new(depth: usize, max_checkpoints: usize) -> Self {
        Self {
            leaves: vec![H::empty_leaf(); 1 << depth],
            current_position: 0,
            witnesses: vec![],
            checkpoints: vec![],
            depth,
            max_checkpoints,
        }
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

impl<H: Hashable + Clone> Frontier<H> for CompleteTree<H> {
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H) -> bool {
        if self.current_position == (1 << self.depth) {
            false
        } else {
            self.leaves[self.current_position] = value.clone();
            self.current_position += 1;
            true
        }
    }

    /// Obtains the current root of this Merkle tree.
    fn root(&self) -> H {
        lazy_root(self.leaves.clone())
    }
}

impl<H: Hashable + PartialEq + Clone> Tree<H> for CompleteTree<H> {
    type Recording = CompleteRecording<H>;

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns true if successful and false if the tree is empty.
    fn witness(&mut self) -> bool {
        if self.current_position == 0 {
            false
        } else {
            let value = self.leaves[self.current_position - 1].clone();
            if !self.witnesses.iter().any(|(_, v)| v == &value) {
                self.witnesses.push((self.current_position - 1, value));
            }
            true
        }
    }

    /// Obtains an authentication path to the value specified in the tree.
    /// Returns `None` if there is no available authentication path to the
    /// specified value.
    fn authentication_path(&self, value: &H) -> Option<(Position, Vec<H>)> {
        self.witnesses
            .iter()
            .find(|witness| witness.1 == *value)
            .map(|&(pos, _)| {
                let mut path = vec![];

                let mut index = pos;
                for bit in 0..self.depth {
                    index ^= 1 << bit;
                    path.push(lazy_root::<H>(self.leaves[index..][0..(1 << bit)].to_vec()));
                    index &= usize::MAX << (bit + 1);
                }

                (pos.into(), path)
            })
    }

    /// Marks the specified tree state value as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if the value is not a known witness.
    fn remove_witness(&mut self, value: &H) -> bool {
        if let Some((position, _)) = self
            .witnesses
            .iter()
            .enumerate()
            .find(|witness| (witness.1).1 == *value)
        {
            self.witnesses.remove(position);

            true
        } else {
            false
        }
    }

    /// Marks the current tree state as a checkpoint if it is not already a
    /// checkpoint.
    fn checkpoint(&mut self) {
        self.checkpoints.push(self.current_position);
        if self.checkpoints.len() > self.max_checkpoints {
            self.drop_oldest_checkpoint();
        }
    }

    /// Rewinds the tree state to the previous checkpoint. This function will
    /// fail and return false if there is no previous checkpoint or in the event
    /// witness data would be destroyed in the process.
    fn rewind(&mut self) -> bool {
        if let Some(checkpoint) = self.checkpoints.pop() {
            if self.witnesses.iter().any(|&(pos, _)| pos >= checkpoint) {
                self.checkpoints.push(checkpoint);
                return false;
            }

            self.current_position = checkpoint;
            if checkpoint != (1 << self.depth) {
                self.leaves[checkpoint..].fill(H::empty_leaf());
            }

            true
        } else {
            false
        }
    }

    /// Start a recording of append operations performed on a tree.
    fn recording(&self) -> CompleteRecording<H> {
        CompleteRecording {
            start_position: self.current_position,
            current_position: self.current_position,
            depth: self.depth,
            appends: vec![],
        }
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the recording is incompatible with the current tree state.
    fn play(&mut self, recording: &CompleteRecording<H>) -> bool {
        #[allow(clippy::suspicious_operation_groupings)]
        if recording.start_position == self.current_position && self.depth == recording.depth {
            for val in recording.appends.iter() {
                self.append(val);
            }
            true
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub struct CompleteRecording<H: Hashable> {
    start_position: usize,
    current_position: usize,
    depth: usize,
    appends: Vec<H>,
}

impl<H: Hashable + Clone> Recording<H> for CompleteRecording<H> {
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H) -> bool {
        if self.current_position == (1 << self.depth) {
            false
        } else {
            self.appends.push(value.clone());
            self.current_position += 1;

            true
        }
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the provided recording is incompatible with `Self`.
    fn play(&mut self, recording: &Self) -> bool {
        #[allow(clippy::suspicious_operation_groupings)]
        if self.current_position == recording.start_position && self.depth == recording.depth {
            self.appends.extend_from_slice(&recording.appends);
            self.current_position = recording.current_position;
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
    use crate::{Altitude, Frontier, Hashable, Tree};

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

        for i in 0..(1 << DEPTH) {
            let (position, path) = tree.authentication_path(&SipHashable(i)).unwrap();
            assert_eq!(
                compute_root_from_auth_path(SipHashable(i), position, &path),
                expected
            );
        }
    }
}
