mod complete_tree;

#[cfg(test)]
pub(crate) mod tests {
    use proptest::prelude::*;
    use std::collections::BTreeSet;
    use std::fmt::Debug;

    use crate::BridgeTree;
    use incrementalmerkletree::{
        testing::{arb_operation, Operation, Operation::*, SipHashable, Tree},
        Hashable, Level, Position,
    };

    use super::complete_tree::{lazy_root, CompleteTree};

    fn append(x: &str) -> Operation<String> {
        Operation::Append(x.to_string())
    }

    fn unmark(pos: usize) -> Operation<String> {
        Operation::Unmark(Position::from(pos))
    }

    fn witness(pos: usize, depth: usize) -> Operation<String> {
        Operation::Authpath(Position::from(pos), depth)
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
                Level::from(1),
                &SipHashable::combine(0.into(), &SipHashable(0), &SipHashable(1)),
                &SipHashable::combine(0.into(), &SipHashable(2), &SipHashable(3)),
            ),
            &SipHashable::combine(
                Level::from(1),
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
                        Level::from(1),
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
                Position::from(4),
                &[
                    SipHashable(5),
                    SipHashable::combine(0.into(), &SipHashable(6), &SipHashable(7)),
                    SipHashable::combine(
                        Level::from(1),
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
            vec![append("a"), append("b"), Checkpoint, Mark, witness(0, 1)],
            vec![append("c"), append("d"), Mark, Checkpoint, witness(1, 1)],
            vec![append("e"), Checkpoint, Mark, append("f"), witness(0, 1)],
            vec![
                append("g"),
                Mark,
                Checkpoint,
                unmark(0),
                append("h"),
                witness(0, 0),
            ],
            vec![
                append("i"),
                Checkpoint,
                Mark,
                unmark(0),
                append("j"),
                witness(0, 0),
            ],
            vec![
                append("i"),
                Mark,
                append("j"),
                Checkpoint,
                append("k"),
                witness(0, 1),
            ],
            vec![
                append("l"),
                Checkpoint,
                Mark,
                Checkpoint,
                append("m"),
                Checkpoint,
                witness(0, 2),
            ],
            vec![Checkpoint, append("n"), Mark, witness(0, 1)],
            vec![
                append("a"),
                Mark,
                Checkpoint,
                unmark(0),
                Checkpoint,
                append("b"),
                witness(0, 1),
            ],
            vec![
                append("a"),
                Mark,
                append("b"),
                unmark(0),
                Checkpoint,
                witness(0, 0),
            ],
            vec![
                append("a"),
                Mark,
                Checkpoint,
                unmark(0),
                Checkpoint,
                Rewind,
                append("b"),
                witness(0, 0),
            ],
            vec![
                append("a"),
                Mark,
                Checkpoint,
                Checkpoint,
                Rewind,
                append("a"),
                unmark(0),
                witness(0, 1),
            ],
            // Unreduced examples
            vec![
                append("o"),
                append("p"),
                Mark,
                append("q"),
                Checkpoint,
                unmark(1),
                witness(1, 1),
            ],
            vec![
                append("r"),
                append("s"),
                append("t"),
                Mark,
                Checkpoint,
                unmark(2),
                Checkpoint,
                witness(2, 2),
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
                witness(0, 3),
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
