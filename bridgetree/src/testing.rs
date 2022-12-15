#[cfg(test)]
pub(crate) mod tests {
    use proptest::prelude::*;
    use std::collections::BTreeSet;
    use std::fmt::Debug;

    use crate::BridgeTree;
    use incrementalmerkletree::{
        testing::{
            append_str, arb_operation, check_operations, complete_tree::CompleteTree, unmark,
            witness, Operation::*, SipHashable, Tree,
        },
        Hashable, Position,
    };

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

    #[test]
    fn test_witness_consistency() {
        let samples = vec![
            // Reduced examples
            vec![
                append_str("a"),
                append_str("b"),
                Checkpoint,
                Mark,
                witness(0, 1),
            ],
            vec![
                append_str("c"),
                append_str("d"),
                Mark,
                Checkpoint,
                witness(1, 1),
            ],
            vec![
                append_str("e"),
                Checkpoint,
                Mark,
                append_str("f"),
                witness(0, 1),
            ],
            vec![
                append_str("g"),
                Mark,
                Checkpoint,
                unmark(0),
                append_str("h"),
                witness(0, 0),
            ],
            vec![
                append_str("i"),
                Checkpoint,
                Mark,
                unmark(0),
                append_str("j"),
                witness(0, 0),
            ],
            vec![
                append_str("i"),
                Mark,
                append_str("j"),
                Checkpoint,
                append_str("k"),
                witness(0, 1),
            ],
            vec![
                append_str("l"),
                Checkpoint,
                Mark,
                Checkpoint,
                append_str("m"),
                Checkpoint,
                witness(0, 2),
            ],
            vec![Checkpoint, append_str("n"), Mark, witness(0, 1)],
            vec![
                append_str("a"),
                Mark,
                Checkpoint,
                unmark(0),
                Checkpoint,
                append_str("b"),
                witness(0, 1),
            ],
            vec![
                append_str("a"),
                Mark,
                append_str("b"),
                unmark(0),
                Checkpoint,
                witness(0, 0),
            ],
            vec![
                append_str("a"),
                Mark,
                Checkpoint,
                unmark(0),
                Checkpoint,
                Rewind,
                append_str("b"),
                witness(0, 0),
            ],
            vec![
                append_str("a"),
                Mark,
                Checkpoint,
                Checkpoint,
                Rewind,
                append_str("a"),
                unmark(0),
                witness(0, 1),
            ],
            // Unreduced examples
            vec![
                append_str("o"),
                append_str("p"),
                Mark,
                append_str("q"),
                Checkpoint,
                unmark(1),
                witness(1, 1),
            ],
            vec![
                append_str("r"),
                append_str("s"),
                append_str("t"),
                Mark,
                Checkpoint,
                unmark(2),
                Checkpoint,
                witness(2, 2),
            ],
            vec![
                append_str("u"),
                Mark,
                append_str("v"),
                append_str("w"),
                Checkpoint,
                unmark(0),
                append_str("x"),
                Checkpoint,
                Checkpoint,
                witness(0, 3),
            ],
        ];

        for (i, sample) in samples.iter().enumerate() {
            let tree = CombinedTree::<String, 4>::new();
            let result = check_operations(tree, 4, sample);
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
            vec![append_str("x"), Checkpoint, Mark, Rewind, unmark(0)],
            vec![
                append_str("d"),
                Checkpoint,
                Mark,
                unmark(0),
                Rewind,
                unmark(0),
            ],
            vec![
                append_str("o"),
                Checkpoint,
                Mark,
                Checkpoint,
                unmark(0),
                Rewind,
                Rewind,
            ],
            vec![
                append_str("s"),
                Mark,
                append_str("m"),
                Checkpoint,
                unmark(0),
                Rewind,
                unmark(0),
                unmark(0),
            ],
        ];
        for (i, sample) in samples.iter().enumerate() {
            let tree = CombinedTree::<String, 4>::new();
            let result = check_operations(tree, 4, sample);
            assert!(
                matches!(result, Ok(())),
                "Reference/Test mismatch at index {}: {:?}",
                i,
                result
            );
        }
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
            let tree = CombinedTree::<SipHashable, 4>::new();
            check_operations(tree, 4, &ops)?;
        }

        #[test]
        fn check_randomized_str_ops(
            ops in proptest::collection::vec(
                arb_operation((97u8..123).prop_map(|c| char::from(c).to_string()), 0usize..100),
                1..100
            )
        ) {
            let tree = CombinedTree::<String, 4>::new();
            check_operations(tree, 4, &ops)?;
        }
    }
}
