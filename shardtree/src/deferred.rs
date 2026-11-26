//! Trees whose hashes are computed one level at a time.
//!
//! An operation that merges leaves must hash each pair of leaves that it merges. Whether two
//! leaves are merged depends on their retention flags, and not on their hashes. The types of
//! this module let an operation first decide which leaves to merge, and then compute the hashes
//! of each level with one call to [`Hashable::combine_pairs`].

use std::collections::BTreeMap;
use std::sync::Arc;

use incrementalmerkletree::{Hashable, Level};

use crate::{prunable::merged_leaf_flags, Node, PrunableTree, RetentionFlags, Tree};

/// The hash of a node. The hash is either held by a tree, or not yet computed.
pub(crate) enum DeferredHash<'a, H> {
    /// A hash that the source tree holds.
    Known(&'a H),
    /// The hash of the parent of a pair of nodes at `level`. `index` is the index of that pair
    /// among the pairs that a [`DeferredHashes`] holds for `level`.
    Combined { level: Level, index: usize },
}

impl<H> Clone for DeferredHash<'_, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H> Copy for DeferredHash<'_, H> {}

/// The hashes of a left node and of its right sibling.
type DeferredPair<'a, H> = (DeferredHash<'a, H>, DeferredHash<'a, H>);

/// The pairs of sibling nodes whose parents are not yet hashed.
pub(crate) struct DeferredHashes<'a, H> {
    /// The pairs of sibling nodes at each level, in the order in which they were added.
    pairs: BTreeMap<Level, Vec<DeferredPair<'a, H>>>,
}

impl<'a, H> DeferredHashes<'a, H> {
    /// Returns a value that holds no pairs.
    pub(crate) fn new() -> Self {
        DeferredHashes {
            pairs: BTreeMap::new(),
        }
    }

    /// Defers the hash of the parent of two sibling nodes.
    ///
    /// - Preconditions: `left` and `right` are the hashes of sibling nodes at `level`. A hash
    ///   that is not yet computed was returned by this method of `self`, for a level below
    ///   `level`.
    /// - Inputs: `level` is the level of the two nodes. `left` is the hash of the left node, and
    ///   `right` is the hash of the right node.
    /// - Outputs: the hash of the parent, which is not yet computed.
    /// - Side effects: adds the pair to `self`.
    pub(crate) fn combine(
        &mut self,
        level: Level,
        left: DeferredHash<'a, H>,
        right: DeferredHash<'a, H>,
    ) -> DeferredHash<'a, H> {
        let pairs = self.pairs.entry(level).or_default();
        let index = pairs.len();
        pairs.push((left, right));
        DeferredHash::Combined { level, index }
    }

    /// Computes the hash of the parent of each pair.
    ///
    /// - Preconditions: none.
    /// - Inputs: none.
    /// - Outputs: the computed hashes.
    /// - Side effects:
    ///   - Calls [`Hashable::combine_pairs`] once for each level that has a pair, in order of
    ///     level. Does not call [`Hashable::combine`].
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    pub(crate) fn compute(self) -> ComputedHashes<H>
    where
        H: Hashable,
    {
        let mut computed = ComputedHashes {
            parents: BTreeMap::new(),
        };
        // The levels are in increasing order, so each hash that a pair needs is computed
        // before the pair is hashed.
        for (level, pairs) in &self.pairs {
            let parents = H::combine_pairs(
                *level,
                pairs
                    .iter()
                    .map(|(left, right)| (computed.get(*left), computed.get(*right))),
            );
            assert_eq!(
                parents.len(),
                pairs.len(),
                "`Hashable::combine_pairs` must return exactly one node for each pair"
            );
            computed.parents.insert(*level, parents);
        }
        computed
    }
}

/// The hashes that [`DeferredHashes::compute`] computed.
pub(crate) struct ComputedHashes<H> {
    /// The parents of the pairs of each level, in the order of the pairs.
    parents: BTreeMap<Level, Vec<H>>,
}

impl<H> ComputedHashes<H> {
    /// Returns the hash that `hash` refers to.
    ///
    /// - Preconditions: if `hash` is not a known hash, it was returned by
    ///   [`DeferredHashes::combine`] of the value that these hashes were computed from.
    /// - Inputs: `hash` is the hash to look up.
    /// - Outputs: the known hash, or the computed hash of the pair that `hash` refers to.
    /// - Side effects: none.
    pub(crate) fn get<'r>(&'r self, hash: DeferredHash<'r, H>) -> &'r H {
        match hash {
            DeferredHash::Known(hash) => hash,
            DeferredHash::Combined { level, index } => self
                .parents
                .get(&level)
                .and_then(|parents| parents.get(index))
                .expect("the hash of each deferred pair is computed before it is read"),
        }
    }
}

/// A tree in which the hash of a leaf can be a hash that is not yet computed.
pub(crate) enum DeferredTree<'a, H> {
    /// A subtree of the source tree, without changes.
    Unchanged(&'a PrunableTree<H>),
    /// A leaf with the given hash and retention flags.
    Leaf(DeferredHash<'a, H>, RetentionFlags),
    /// A parent with the given annotation, left child and right child.
    Parent(
        &'a Option<Arc<H>>,
        Box<DeferredTree<'a, H>>,
        Box<DeferredTree<'a, H>>,
    ),
    /// A tree with no nodes.
    Nil,
}

impl<'a, H> DeferredTree<'a, H> {
    /// Returns `true` if this tree has no nodes.
    fn is_nil(&self) -> bool {
        match self {
            DeferredTree::Unchanged(tree) => tree.is_nil(),
            DeferredTree::Nil => true,
            DeferredTree::Leaf(..) | DeferredTree::Parent(..) => false,
        }
    }

    /// Returns the hash and the retention flags of this tree if it is a leaf.
    fn as_leaf(&self) -> Option<(DeferredHash<'a, H>, RetentionFlags)> {
        match self {
            DeferredTree::Unchanged(tree) => match &tree.0 {
                Node::Leaf {
                    value: (hash, flags),
                } => Some((DeferredHash::Known(hash), *flags)),
                Node::Parent { .. } | Node::Nil => None,
            },
            DeferredTree::Leaf(hash, flags) => Some((*hash, *flags)),
            DeferredTree::Parent(..) | DeferredTree::Nil => None,
        }
    }

    /// Unites two sibling trees as [`Tree::unite`] does, but defers the hash of a merged leaf.
    ///
    /// - Preconditions: `left` and `right` are siblings at `level`, and each hash in them that
    ///   is not yet computed was returned by `hashes`.
    /// - Inputs: `hashes` receives the pair to hash if one leaf replaces `left` and `right`.
    ///   `level` is the level of `left` and of `right`. `ann` is the annotation for a parent of
    ///   the two trees.
    /// - Outputs: the tree that [`Tree::unite`] returns for the same arguments, except that the
    ///   hash of a merged leaf is not yet computed.
    /// - Side effects: adds at most one pair to `hashes`.
    pub(crate) fn unite(
        hashes: &mut DeferredHashes<'a, H>,
        level: Level,
        ann: &'a Option<Arc<H>>,
        left: Self,
        right: Self,
    ) -> Self {
        if ann.is_none() && left.is_nil() && right.is_nil() {
            return DeferredTree::Nil;
        }
        let merged_leaf = left.as_leaf().zip(right.as_leaf()).and_then(
            |((left_hash, left_flags), (right_hash, right_flags))| {
                merged_leaf_flags(left_flags, right_flags)
                    .map(|flags| (left_hash, right_hash, flags))
            },
        );
        match merged_leaf {
            Some((left_hash, right_hash, flags)) => {
                DeferredTree::Leaf(hashes.combine(level, left_hash, right_hash), flags)
            }
            None => DeferredTree::Parent(ann, Box::new(left), Box::new(right)),
        }
    }

    /// Returns this tree with each of its hashes computed.
    ///
    /// - Preconditions: `hashes` was computed from the [`DeferredHashes`] that each hash of this
    ///   tree that is not yet computed was returned by.
    /// - Inputs: `hashes` holds the computed hashes.
    /// - Outputs: the tree.
    /// - Side effects: none.
    pub(crate) fn into_tree(self, hashes: &ComputedHashes<H>) -> PrunableTree<H>
    where
        H: Clone,
    {
        match self {
            DeferredTree::Unchanged(tree) => tree.clone(),
            DeferredTree::Leaf(hash, flags) => Tree::leaf((hashes.get(hash).clone(), flags)),
            DeferredTree::Parent(ann, left, right) => {
                Tree::parent(ann.clone(), left.into_tree(hashes), right.into_tree(hashes))
            }
            DeferredTree::Nil => Tree::empty(),
        }
    }
}
