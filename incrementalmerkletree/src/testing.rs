use crate::{Hashable, Level};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A possibly-empty incremental Merkle frontier.
pub trait Frontier<H> {
    /// Appends a new value to the frontier at the next available slot.
    /// Returns true if successful and false if the frontier would exceed
    /// the maximum allowed depth.
    fn append(&mut self, value: H) -> bool;

    /// Obtains the current root of this Merkle frontier by hashing
    /// against empty nodes up to the maximum height of the pruned
    /// tree that the frontier represents.
    fn root(&self) -> H;
}

impl Hashable for String {
    fn empty_leaf() -> Self {
        "_".to_string()
    }

    fn combine(_: Level, a: &Self, b: &Self) -> Self {
        a.to_string() + b
    }
}

impl<H> Hashable for Option<H>
where
    H: Hashable,
{
    fn empty_leaf() -> Self {
        Some(H::empty_leaf())
    }

    fn combine(l: Level, a: &Self, b: &Self) -> Self {
        match (a, b) {
            (Some(a), Some(b)) => Some(H::combine(l, a, b)),
            _ => None,
        }
    }

    fn combine_pairs<'a, I>(level: Level, pairs: I) -> Vec<Self>
    where
        Self: 'a,
        I: IntoIterator<Item = (&'a Self, &'a Self)>,
    {
        // A pair with a `None` child has the parent `None`. Only the pairs with two present
        // children go to `H::combine_pairs`.
        let pairs: Vec<Option<(&H, &H)>> = pairs
            .into_iter()
            .map(|(left, right)| left.as_ref().zip(right.as_ref()))
            .collect();
        let mut parents = H::combine_pairs(level, pairs.iter().flatten().copied()).into_iter();
        pairs
            .iter()
            .map(|pair| pair.and_then(|_| parents.next()))
            .collect()
    }
}

/// Checks that [`Hashable::combine_pairs`] agrees with [`Hashable::combine`] for the type `H`.
///
/// A crate that overrides `Hashable::combine_pairs` can call this function from its tests, with
/// enough pairs to reach its batch implementation.
///
/// - Preconditions: none.
/// - Inputs: `level` is the level of the nodes in `pairs`. Each item of `pairs` holds a left
///   child and then a right child.
/// - Outputs: none.
/// - Side effects: panics unless `H::combine_pairs` returns one node for each pair, in order,
///   and each node is equal to the result of `H::combine` for that pair.
pub fn check_combine_pairs<H>(level: Level, pairs: &[(H, H)])
where
    H: Hashable + PartialEq,
{
    let expected: Vec<H> = pairs
        .iter()
        .map(|(left, right)| H::combine(level, left, right))
        .collect();
    let actual = H::combine_pairs(level, pairs.iter().map(|(left, right)| (left, right)));
    assert_eq!(
        actual, expected,
        "`combine_pairs` must return the result of `combine` for each pair, in order (at {level:?})"
    );
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use proptest::prelude::*;

    use super::check_combine_pairs;
    use crate::Level;

    fn arb_node() -> impl Strategy<Value = String> {
        "[a-z]{1,3}"
    }

    proptest! {
        #[test]
        fn string_combine_pairs_agrees_with_combine(
            level in 0u8..64,
            pairs in proptest::collection::vec((arb_node(), arb_node()), 0..16),
        ) {
            check_combine_pairs::<String>(Level::from(level), &pairs);
        }

        #[test]
        fn option_combine_pairs_agrees_with_combine(
            level in 0u8..64,
            pairs in proptest::collection::vec(
                (proptest::option::of(arb_node()), proptest::option::of(arb_node())),
                0..16,
            ),
        ) {
            check_combine_pairs::<Option<String>>(Level::from(level), &pairs);
        }
    }
}
