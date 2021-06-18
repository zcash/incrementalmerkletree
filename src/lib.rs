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

pub mod sample;
pub mod efficient;

pub trait TreeHasher {
    type Digest: Clone + PartialEq + std::fmt::Debug;

    fn empty_leaf() -> Self::Digest;
    fn combine(a: &Self::Digest, b: &Self::Digest) -> Self::Digest;
}

pub trait Tree<H: TreeHasher> {
    /// The type of recordings that can be made of the operations of this tree.
    type Recording: Recording<H>;

    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H::Digest) -> bool;

    /// Obtains the current root of this Merkle tree.
    fn root(&self) -> H::Digest;

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns true if successful and false if the tree is empty.
    fn witness(&mut self) -> bool;

    /// Obtains an authentication path to the value specified in the tree.
    /// Returns `None` if there is no available authentication path to the
    /// specified value.
    fn authentication_path(&self, value: &H::Digest) -> Option<(usize, Vec<H::Digest>)>;

    /// Marks the specified tree state value as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if the value is not a known witness.
    fn remove_witness(&mut self, value: &H::Digest) -> bool;

    /// Marks the current tree state as a checkpoint if it is not already a
    /// checkpoint.
    fn checkpoint(&mut self);

    /// Rewinds the tree state to the previous checkpoint. This function will
    /// fail and return false if there is no previous checkpoint or in the event
    /// witness data would be destroyed in the process.
    fn rewind(&mut self) -> bool;

    /// Removes the oldest checkpoint. Returns true if successful and false if
    /// there are no checkpoints.
    fn pop_checkpoint(&mut self) -> bool;
   
    /// Start a recording of append operations performed on a tree.
    fn recording(&self) -> Self::Recording;

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the recording is incompatible with the current tree state.
    fn play(&mut self, recording: &Self::Recording) -> bool;
}

pub trait Recording<H: TreeHasher> {
    /// Appends a new value to the tree at the next available slot. Returns true
    /// if successful and false if the tree is full.
    fn append(&mut self, value: &H::Digest) -> bool;

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the provided recording is incompatible with `Self`.
    fn play(&mut self, recording: &Self) -> bool;
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]
    use super::*;
    use sample::*;
    use efficient::*;

    use std::hash::Hasher;
    use std::hash::SipHasher as Hash;

    #[derive(Clone)]
    pub struct CombinedTree<H: TreeHasher> {
        inefficient: CompleteTree<H>,
        efficient: EfficientTree<H>
    }

    impl<H: TreeHasher> CombinedTree<H> {
        pub fn new(depth: usize) -> Self {
            CombinedTree {
                inefficient: CompleteTree::new(depth),
                efficient: EfficientTree::new(depth),
            }
        }

        pub fn append(&mut self, value: &H::Digest) -> bool {
            let a = self.inefficient.append(value);
            let b = self.efficient.append(value);
            assert_eq!(a, b);
            a
        }

        /// Obtains the current root of this Merkle tree.
        pub fn root(&self) -> H::Digest {
            let a = self.inefficient.root();
            let b = self.efficient.root();
            assert_eq!(a, b);
            a
        }

        /// Marks the current tree state leaf as a value that we're interested in
        /// witnessing. Returns true if successful and false if the tree is empty.
        pub fn witness(&mut self) -> bool {
            let a = self.inefficient.witness();
            let b = self.efficient.witness();
            assert_eq!(a, b);
            a
        }

        /// Obtains an authentication path to the value specified in the tree.
        /// Returns `None` if there is no available authentication path to the
        /// specified value.
        pub fn authentication_path(&self, value: &H::Digest) -> Option<(usize, Vec<H::Digest>)> {
            let a = self.inefficient.authentication_path(value);
            let b = self.efficient.authentication_path(value);
            assert_eq!(a, b);
            a
        }

        /// Marks the specified tree state value as a value we're no longer
        /// interested in maintaining a witness for. Returns true if successful and
        /// false if the value is not a known witness.
        pub fn remove_witness(&mut self, value: &H::Digest) -> bool {
            let a = self.inefficient.remove_witness(value);
            let b = self.efficient.remove_witness(value);
            assert_eq!(a, b);
            a
        }

        /// Marks the current tree state as a checkpoint if it is not already a
        /// checkpoint.
        pub fn checkpoint(&mut self) {
            self.inefficient.checkpoint();
            self.efficient.checkpoint();
        }

        /// Rewinds the tree state to the previous checkpoint. This function will
        /// fail and return false if there is no previous checkpoint or in the event
        /// witness data would be destroyed in the process.
        pub fn rewind(&mut self) -> bool {
            let a = self.inefficient.rewind();
            let b = self.efficient.rewind();
            assert_eq!(a, b);
            a
        }

        /// Removes the oldest checkpoint. Returns true if successful and false if
        /// there are no checkpoints.
        pub fn pop_checkpoint(&mut self) -> bool {
            let a = self.inefficient.pop_checkpoint();
            let b = self.efficient.pop_checkpoint();
            assert_eq!(a, b);
            a
        }

        /// Start a recording of append operations performed on a tree.
        pub fn recording(&self) -> CombinedRecording<H> {
            CombinedRecording {
                inefficient: self.inefficient.recording(),
                efficient: self.efficient.recording()
            }
        }

        /// Plays a recording of append operations back. Returns true if successful
        /// and false if the recording is incompatible with the current tree state.
        pub fn play(&mut self, recording: &CombinedRecording<H>) -> bool {
            let a = self.inefficient.play(&recording.inefficient);
            let b = self.efficient.play(&recording.efficient);
            assert_eq!(a, b);
            a
        }
    }

    #[derive(Clone)]
    pub struct CombinedRecording<H: TreeHasher> {
        inefficient: CompleteRecording<H>,
        efficient: EfficientRecording<H>,
    }

    impl<H: TreeHasher> CombinedRecording<H> {
        pub fn append(&mut self, value: &H::Digest) -> bool {
            let a = self.inefficient.append(value);
            let b = self.efficient.append(value);
            assert_eq!(a, b);
            a
        }

        pub fn play(&mut self, recording: &Self) -> bool {
            let a = self.inefficient.play(&recording.inefficient);
            let b = self.efficient.play(&recording.efficient);
            assert_eq!(a, b);
            a
        }
    }

    impl TreeHasher for Hash {
        type Digest = u64;

        fn empty_leaf() -> Self::Digest {
            0
        }
        fn combine(a: &Self::Digest, b: &Self::Digest) -> Self::Digest {
            let mut hasher = Hash::new();
            hasher.write_u64(*a);
            hasher.write_u64(*b);
            hasher.finish()
        }
    }

    pub(crate) fn compute_root_from_auth_path<H: TreeHasher>(
        value: H::Digest,
        position: usize,
        path: &[H::Digest],
    ) -> H::Digest {
        let mut cur = value;
        for (i, v) in path
            .iter()
            .enumerate()
            .map(|(i, v)| (((position >> i) & 1) == 1, v))
        {
            if i {
                cur = H::combine(v, &cur);
            } else {
                cur = H::combine(&cur, v);
            }
        }
        cur
    }

    #[test]
    fn test_compute_root_from_auth_path() {
        let expected = Hash::combine(
            &Hash::combine(&Hash::combine(&0, &1), &Hash::combine(&2, &3)),
            &Hash::combine(&Hash::combine(&4, &5), &Hash::combine(&6, &7)),
        );

        assert_eq!(
            compute_root_from_auth_path::<Hash>(
                0,
                0,
                &[
                    1,
                    Hash::combine(&2, &3),
                    Hash::combine(&Hash::combine(&4, &5), &Hash::combine(&6, &7))
                ]
            ),
            expected
        );

        assert_eq!(
            compute_root_from_auth_path::<Hash>(
                4,
                4,
                &[
                    5,
                    Hash::combine(&6, &7),
                    Hash::combine(&Hash::combine(&0, &1), &Hash::combine(&2, &3))
                ]
            ),
            expected
        );
    }

    use proptest::prelude::*;

    #[derive(Clone, Debug)]
    enum Operation {
        Append(u64),
        Witness,
        Unwitness(u64),
        Checkpoint,
        Rewind,
        PopCheckpoint,
        Authpath(u64),
    }

    use Operation::*;

    prop_compose! {
        fn arb_operation()
                    (
                        opid in (0..7),
                        item in (0..32u64),
                    )
                    -> Operation
        {
            match opid {
                0 => Append(item),
                1 => Witness,
                2 => Unwitness(item),
                3 => Checkpoint,
                4 => Rewind,
                5 => PopCheckpoint,
                6 => Authpath(item),
                _ => unimplemented!()
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100000))]
        #[test]
        fn do_stuff(ops in proptest::collection::vec(arb_operation(), 1..100)) {
            const DEPTH: usize = 4;
            let mut tree = CombinedTree::<Hash>::new(DEPTH);

            let mut prevtrees = vec![];

            let mut tree_size = 0;
            let mut tree_values = vec![];
            let mut tree_checkpoints = vec![];
            let mut tree_witnesses: Vec<(usize, u64)> = vec![];

            for op in ops {
                assert_eq!(tree_size, tree_values.len());
                match op {
                    Append(value) => {
                        prevtrees.push((tree.clone(), tree.recording()));
                        if tree.append(&value) {
                            assert!(tree_size < (1 << DEPTH));
                            tree_size += 1;
                            tree_values.push(value);

                            for &mut (_, ref mut recording) in &mut prevtrees {
                                assert!(recording.append(&value));
                            }
                        } else {
                            assert!(tree_size == (1 << DEPTH));
                        }
                    }
                    Witness => {
                        if tree.witness() {
                            assert!(tree_size != 0);
                            tree_witnesses.push((tree_size - 1, *tree_values.last().unwrap()));
                        } else {
                            assert!(tree_size == 0);
                        }
                    }
                    Unwitness(value) => {
                        if tree.remove_witness(&value) {
                            if let Some((i, _)) = tree_witnesses.iter().enumerate().find(|v| (v.1).1 == value) {
                                tree_witnesses.remove(i);
                            } else {
                                panic!("witness should not have been removed");
                            }
                        } else {
                            if tree_witnesses.iter().find(|v| v.1 == value).is_some() {
                                panic!("witness should have been removed");
                            }
                        }
                    }
                    Checkpoint => {
                        tree_checkpoints.push(tree_size);
                        tree.checkpoint();
                    }
                    Rewind => {
                        prevtrees.truncate(0);

                        if tree.rewind() {
                            assert!(tree_checkpoints.len() > 0);
                            let checkpoint_location = tree_checkpoints.pop().unwrap();
                            for &(index, _) in tree_witnesses.iter() {
                                // index is the index in tree_values
                                // checkpoint_location is the size of the tree
                                // at the time of the checkpoint
                                // index should always be strictly smaller or
                                // else a witness would be erased!
                                assert!(index < checkpoint_location);
                            }
                            tree_values.truncate(checkpoint_location);
                            tree_size = checkpoint_location;
                        } else {
                            if tree_checkpoints.len() != 0 {
                                let checkpoint_location = *tree_checkpoints.last().unwrap();
                                assert!(tree_witnesses.iter().any(|&(index, _)| index >= checkpoint_location));
                            }
                        }
                    }
                    PopCheckpoint => {
                        if tree.pop_checkpoint() {
                            assert!(tree_checkpoints.len() > 0);
                            tree_checkpoints.remove(0);
                        } else {
                            assert!(tree_checkpoints.len() == 0);
                        }
                    }
                    Authpath(value) => {
                        if let Some((position, path)) = tree.authentication_path(&value) {
                            // must be the case that value was a witness
                            assert!(tree_witnesses.iter().any(|&(_, witness)| witness == value));

                            let mut extended_tree_values = tree_values.clone();
                            extended_tree_values.resize(1 << DEPTH, Hash::empty_leaf());
                            let expected_root = lazy_root::<Hash>(extended_tree_values);

                            let tree_root = tree.root();
                            assert_eq!(tree_root, expected_root);

                            assert_eq!(
                                compute_root_from_auth_path::<Hash>(value, position, &path),
                                expected_root
                            );
                        } else {
                            // must be the case that value wasn't a witness
                            for &(_, witness) in tree_witnesses.iter() {
                                assert!(witness != value);
                            }
                        }
                    }
                }
            }

            for (mut other_tree, other_recording) in prevtrees {
                assert!(other_tree.play(&other_recording));
                assert_eq!(tree.root(), other_tree.root());
            }
        }
    }
}
