#[cfg(test)]
pub(crate) mod tests {
    use proptest::prelude::*;
    use std::fmt::Debug;

    use crate::BridgeTree;
    use incrementalmerkletree::{
        testing::{
            arb_operation, check_operations, check_rewind_remove_mark,
            check_rewind_remove_mark_consistency, complete_tree::CompleteTree, CombinedTree,
            SipHashable,
        },
        Hashable,
    };

    fn new_combined_tree<H: Hashable + Ord + Clone + Debug>(
        max_checkpoints: usize,
    ) -> CombinedTree<H, CompleteTree<H>, BridgeTree<H, 4>> {
        CombinedTree::new(
            CompleteTree::new(4, max_checkpoints),
            BridgeTree::<H, 4>::new(max_checkpoints),
        )
    }

    #[test]
    fn test_rewind_remove_mark() {
        check_rewind_remove_mark(new_combined_tree);
    }

    #[test]
    fn test_rewind_remove_mark_consistency() {
        check_rewind_remove_mark_consistency(new_combined_tree);
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
            let tree = new_combined_tree(100);
            check_operations(tree, 4, &ops)?;
        }

        #[test]
        fn check_randomized_str_ops(
            ops in proptest::collection::vec(
                arb_operation((97u8..123).prop_map(|c| char::from(c).to_string()), 0usize..100),
                1..100
            )
        ) {
            let tree = new_combined_tree(100);
            check_operations(tree, 4, &ops)?;
        }
    }
}
