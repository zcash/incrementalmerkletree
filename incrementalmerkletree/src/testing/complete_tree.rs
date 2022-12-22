//! Sample implementation of the Tree interface.
use std::cmp::min;
use std::collections::{BTreeSet, VecDeque};

use crate::{testing::Tree, Hashable, Level, Position};

pub(crate) fn root<H: Hashable + Clone>(leaves: &[H], depth: u8) -> H {
    let empty_leaf = H::empty_leaf();
    let mut leaves = leaves
        .iter()
        .chain(std::iter::repeat(&empty_leaf))
        .take(1 << depth)
        .cloned()
        .collect::<Vec<H>>();

    //leaves are always at level zero, so we start there.
    let mut level = Level::from(0);
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// The number of leaves that will be retained
    leaves_len: usize,
    /// A set of the positions that have been marked during the period that this
    /// checkpoint is the current checkpoint.
    marked: BTreeSet<Position>,
    /// When a mark is forgotten, we add it to the checkpoint's forgotten set but
    /// don't immediately remove it from the `marked` set; that removal occurs when
    /// the checkpoint is eventually dropped.
    forgotten: BTreeSet<Position>,
}

impl Checkpoint {
    pub fn at_length(leaves_len: usize) -> Self {
        Checkpoint {
            leaves_len,
            marked: BTreeSet::new(),
            forgotten: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompleteTree<H, const DEPTH: u8> {
    leaves: Vec<Option<H>>,
    marks: BTreeSet<Position>,
    checkpoints: VecDeque<Checkpoint>,
    max_checkpoints: usize,
}

impl<H: Hashable, const DEPTH: u8> CompleteTree<H, DEPTH> {
    /// Creates a new, empty binary tree
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            leaves: vec![],
            marks: BTreeSet::new(),
            checkpoints: VecDeque::from(vec![Checkpoint::at_length(0)]),
            max_checkpoints,
        }
    }

    fn append(&mut self, value: H) -> bool {
        if self.leaves.len() == (1 << DEPTH) {
            false
        } else {
            self.leaves.push(Some(value));
            true
        }
    }

    fn leaves_at_checkpoint_depth(&self, checkpoint_depth: usize) -> Option<usize> {
        if checkpoint_depth == 0 {
            Some(self.leaves.len())
        } else if checkpoint_depth <= self.checkpoints.len() {
            self.checkpoints
                .get(self.checkpoints.len() - checkpoint_depth)
                .map(|c| c.leaves_len)
        } else {
            None
        }
    }
}

impl<H: Hashable + PartialEq + Clone, const DEPTH: u8> CompleteTree<H, DEPTH> {
    /// Removes the oldest checkpoint. Returns true if successful and false if
    /// there are fewer than `self.max_checkpoints` checkpoints.
    fn drop_oldest_checkpoint(&mut self) -> bool {
        if self.checkpoints.len() > self.max_checkpoints {
            let c = self.checkpoints.pop_front().unwrap();
            for pos in c.forgotten.iter() {
                self.marks.remove(pos);
            }
            true
        } else {
            false
        }
    }
}

impl<H: Hashable + PartialEq + Clone + std::fmt::Debug, const DEPTH: u8> Tree<H>
    for CompleteTree<H, DEPTH>
{
    fn depth(&self) -> u8 {
        DEPTH
    }

    fn append(&mut self, value: H) -> bool {
        Self::append(self, value)
    }

    fn current_position(&self) -> Option<Position> {
        if self.leaves.is_empty() {
            None
        } else {
            Some((self.leaves.len() - 1).into())
        }
    }

    fn current_leaf(&self) -> Option<&H> {
        self.leaves.last().and_then(|opt: &Option<H>| opt.as_ref())
    }

    fn get_marked_leaf(&self, position: Position) -> Option<&H> {
        if self.marks.contains(&position) {
            self.leaves
                .get(usize::from(position))
                .and_then(|opt: &Option<H>| opt.as_ref())
        } else {
            None
        }
    }

    fn mark(&mut self) -> Option<Position> {
        match self.current_position() {
            Some(pos) => {
                if !self.marks.contains(&pos) {
                    self.marks.insert(pos);
                    self.checkpoints.back_mut().unwrap().marked.insert(pos);
                }
                Some(pos)
            }
            None => None,
        }
    }

    fn marked_positions(&self) -> BTreeSet<Position> {
        self.marks.clone()
    }

    fn root(&self, checkpoint_depth: usize) -> Option<H> {
        self.leaves_at_checkpoint_depth(checkpoint_depth)
            .and_then(|len| root(&self.leaves[0..len], DEPTH))
    }

    fn witness(&self, position: Position, checkpoint_depth: usize) -> Option<Vec<H>> {
        if self.marks.contains(&position) && checkpoint_depth <= self.checkpoints.len() {
            let checkpoint_idx = self.checkpoints.len() - checkpoint_depth;
            let len = if checkpoint_depth == 0 {
                self.leaves.len()
            } else {
                self.checkpoints[checkpoint_idx].leaves_len
            };

            if self
                .checkpoints
                .iter()
                .skip(checkpoint_idx)
                .any(|c| c.marked.contains(&position))
            {
                // The requested position was marked after the checkpoint was created, so we
                // cannot create a witness.
                None
            } else {
                let mut path = vec![];

                let mut leaf_idx: usize = position.into();
                for bit in 0..DEPTH {
                    leaf_idx ^= 1 << bit;
                    path.push(if leaf_idx < len {
                        let subtree_end = min(leaf_idx + (1 << bit), len);
                        root(&self.leaves[leaf_idx..subtree_end], bit)?
                    } else {
                        H::empty_root(Level::from(bit))
                    });
                    leaf_idx &= usize::MAX << (bit + 1);
                }

                Some(path)
            }
        } else {
            None
        }
    }

    fn remove_mark(&mut self, position: Position) -> bool {
        if self.marks.contains(&position) {
            self.checkpoints
                .back_mut()
                .unwrap()
                .forgotten
                .insert(position);
            true
        } else {
            false
        }
    }

    fn checkpoint(&mut self) {
        self.checkpoints
            .push_back(Checkpoint::at_length(self.leaves.len()));
        if self.checkpoints.len() > self.max_checkpoints {
            self.drop_oldest_checkpoint();
        }
    }

    fn rewind(&mut self) -> bool {
        if self.checkpoints.len() > 1 {
            let c = self.checkpoints.pop_back().unwrap();
            self.leaves.truncate(c.leaves_len);
            for pos in c.marked.iter() {
                self.marks.remove(pos);
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use super::CompleteTree;
    use crate::{
        testing::{
            check_checkpoint_rewind, check_rewind_remove_mark, check_root_hashes, check_witnesses,
            compute_root_from_witness, SipHashable, Tree,
        },
        Hashable, Level, Position,
    };

    #[test]
    fn correct_empty_root() {
        const DEPTH: u8 = 5;
        let mut expected = SipHashable(0u64);
        for lvl in 0u8..DEPTH {
            expected = SipHashable::combine(lvl.into(), &expected, &expected);
        }

        let tree = CompleteTree::<SipHashable, DEPTH>::new(100);
        assert_eq!(tree.root(0).unwrap(), expected);
    }

    #[test]
    fn correct_root() {
        const DEPTH: u8 = 3;
        let values = (0..(1 << DEPTH)).into_iter().map(SipHashable);

        let mut tree = CompleteTree::<SipHashable, DEPTH>::new(100);
        for value in values {
            assert!(tree.append(value));
        }
        assert!(!tree.append(SipHashable(0)));

        let expected = SipHashable::combine(
            Level::from(2),
            &SipHashable::combine(
                Level::from(1),
                &SipHashable::combine(Level::from(1), &SipHashable(0), &SipHashable(1)),
                &SipHashable::combine(Level::from(1), &SipHashable(2), &SipHashable(3)),
            ),
            &SipHashable::combine(
                Level::from(1),
                &SipHashable::combine(Level::from(1), &SipHashable(4), &SipHashable(5)),
                &SipHashable::combine(Level::from(1), &SipHashable(6), &SipHashable(7)),
            ),
        );

        assert_eq!(tree.root(0).unwrap(), expected);
    }

    #[test]
    fn root_hashes() {
        check_root_hashes(CompleteTree::<String, 4>::new);
    }

    #[test]
    fn witness() {
        check_witnesses(CompleteTree::<String, 4>::new);
    }

    #[test]
    fn correct_witness() {
        const DEPTH: u8 = 3;
        let values = (0..(1 << DEPTH)).into_iter().map(SipHashable);

        let mut tree = CompleteTree::<SipHashable, DEPTH>::new(100);
        for value in values {
            assert!(tree.append(value));
            tree.mark();
        }
        assert!(!tree.append(SipHashable(0)));

        let expected = SipHashable::combine(
            <Level>::from(2),
            &SipHashable::combine(
                Level::from(1),
                &SipHashable::combine(Level::from(1), &SipHashable(0), &SipHashable(1)),
                &SipHashable::combine(Level::from(1), &SipHashable(2), &SipHashable(3)),
            ),
            &SipHashable::combine(
                Level::from(1),
                &SipHashable::combine(Level::from(1), &SipHashable(4), &SipHashable(5)),
                &SipHashable::combine(Level::from(1), &SipHashable(6), &SipHashable(7)),
            ),
        );

        assert_eq!(tree.root(0).unwrap(), expected);

        for i in 0u64..(1 << DEPTH) {
            let position = Position::try_from(i).unwrap();
            let path = tree.witness(position, 0).unwrap();
            assert_eq!(
                compute_root_from_witness(SipHashable(i), position, &path),
                expected
            );
        }
    }

    #[test]
    fn checkpoint_rewind() {
        check_checkpoint_rewind(CompleteTree::<String, 4>::new);
    }

    #[test]
    fn rewind_remove_mark() {
        check_rewind_remove_mark(CompleteTree::<String, 4>::new);
    }
}
