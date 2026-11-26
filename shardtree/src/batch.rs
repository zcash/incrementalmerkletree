//! Helpers for inserting many leaves into a tree at once.

use std::{collections::BTreeMap, fmt, ops::Range};

use incrementalmerkletree::{Address, Hashable, Level, Position, Retention};
use tracing::trace;

use crate::{
    error::{InsertionError, ShardTreeError},
    store::{Checkpoint, ShardStore},
    IncompleteAt, LocatedPrunableTree, LocatedTree, PrunableTree, RetentionFlags, ShardTree, Tree,
};

impl<
        H: Hashable + Clone + PartialEq,
        C: Clone + fmt::Debug + Ord,
        S: ShardStore<H = H, CheckpointId = C>,
        const DEPTH: u8,
        const SHARD_HEIGHT: u8,
    > ShardTree<S, DEPTH, SHARD_HEIGHT>
{
    /// Put a range of values into the subtree to fill leaves starting from the given position.
    ///
    /// This operation will pad the tree until it contains enough subtrees to reach the starting
    /// position. It will fully consume the provided iterator, constructing successive subtrees
    /// until no more values are available. It aggressively prunes the tree as it goes, retaining
    /// only nodes that either have [`Retention::Marked`] retention, are required to construct a
    /// witness for such marked nodes, or that must be retained in order to make it possible to
    /// truncate the tree to any position with [`Retention::Checkpoint`] retention.
    ///
    /// This operation returns the final position at which a leaf was inserted, and the vector of
    /// [`IncompleteAt`] values that identify addresses at which [`Node::Nil`] nodes were
    /// introduced to the tree, as well as whether or not those newly introduced nodes will need to
    /// be filled with values in order to produce witnesses for inserted leaves with
    /// [`Retention::Marked`] retention.
    ///
    /// Each shard is checked and written in turn. If an error is returned, the shards written
    /// before the one that failed, and their checkpoints, remain in the store. Callers that need
    /// all-or-nothing behavior should perform the operation within a transaction of the
    /// underlying [`ShardStore`], and discard it on error.
    ///
    /// This method operates on a single thread. If you have parallelism available, consider using
    /// [`LocatedPrunableTree::from_iter`] and [`Self::insert_tree`] instead.
    ///
    /// [`Node::Nil`]: crate::tree::Node::Nil
    #[allow(clippy::type_complexity)]
    pub fn batch_insert<I>(
        &mut self,
        mut start: Position,
        values: I,
    ) -> Result<Option<(Position, Vec<IncompleteAt>)>, ShardTreeError<S::Error>>
    where
        I: Iterator<Item = (H, Retention<C>)>,
    {
        trace!("Batch inserting from {:?}", start);
        let mut values = values.peekable();
        let mut subtree_root_addr = Self::subtree_addr(start);
        let mut max_insert_position = None;
        let mut all_incomplete = vec![];
        loop {
            if values.peek().is_some() {
                let mut res = self
                    .store
                    .get_shard(subtree_root_addr)
                    .map_err(ShardTreeError::Storage)?
                    .unwrap_or_else(|| LocatedTree::empty(subtree_root_addr))
                    .batch_insert(start, values)?
                    .expect(
                        "Iterator containing leaf values to insert was verified to be nonempty.",
                    );
                self.put_shard_checked(res.subtree)?;
                for (id, position) in res.checkpoints.into_iter() {
                    self.store
                        .add_checkpoint(id, Checkpoint::at_position(position))
                        .map_err(ShardTreeError::Storage)?;
                }

                values = res.remainder;
                subtree_root_addr = subtree_root_addr.next_at_level();
                max_insert_position = Some(res.max_insert_position);
                start = res.max_insert_position + 1;
                all_incomplete.append(&mut res.incomplete);
            } else {
                break;
            }
        }

        self.prune_excess_checkpoints()?;
        Ok(max_insert_position.map(|p| (p, all_incomplete)))
    }
}

/// A type for the result of a batch insertion operation.
///
/// This result type contains the newly constructed tree, the addresses any new incomplete internal
/// nodes within that tree that were introduced as a consequence of that insertion, and the
/// remainder of the iterator that provided the inserted values.
#[derive(Debug)]
pub struct BatchInsertionResult<H, C, I>
where
    C: Ord,
    I: Iterator<Item = (H, Retention<C>)>,
{
    /// The updated tree after all insertions have been performed.
    pub subtree: LocatedPrunableTree<H>,
    /// A flag identifying whether the constructed subtree contains a marked node.
    pub contains_marked: bool,
    /// The vector of addresses of [`Node::Nil`] nodes that were inserted into the tree as part of
    /// the insertion operation, for nodes that are required in order to construct a witness for
    /// each inserted leaf with [`Retention::Marked`] retention.
    ///
    /// [`Node::Nil`]: crate::tree::Node::Nil
    pub incomplete: Vec<IncompleteAt>,
    /// The maximum position at which a leaf was inserted.
    pub max_insert_position: Position,
    /// The positions of all leaves with [`Retention::Checkpoint`] retention that were inserted.
    pub checkpoints: BTreeMap<C, Position>,
    /// The unconsumed remainder of the iterator from which leaves were inserted, if the tree
    /// was completely filled before the iterator was fully consumed.
    pub remainder: I,
}

impl<H> LocatedPrunableTree<H>
where
    H: Hashable + Clone + PartialEq,
{
    /// Append a values from an iterator, beginning at the first available position in the tree.
    ///
    /// Returns an error if the tree is full. If the position at the end of the iterator is outside
    /// of the subtree's range, the unconsumed part of the iterator will be returned as part of
    /// the result.
    pub fn batch_append<C, I>(
        &self,
        values: I,
    ) -> Result<Option<BatchInsertionResult<H, C, I>>, InsertionError>
    where
        C: Clone + Ord,
        I: Iterator<Item = (H, Retention<C>)>,
    {
        let append_position = self
            .max_position()
            .map(|p| p + 1)
            .unwrap_or_else(|| self.root_addr.position_range_start());
        self.batch_insert(append_position, values)
    }

    /// Put a range of values into the subtree by consuming the given iterator, starting at the
    /// specified position.
    ///
    /// The start position must exist within the position range of this subtree. If the position at
    /// the end of the iterator is outside of the subtree's range, the unconsumed part of the
    /// iterator will be returned as part of the result.
    ///
    /// Returns `Ok(None)` if the provided iterator is empty, `Ok(Some(BatchInsertionResult))` if
    /// values were successfully inserted, or an error if the start position provided is outside
    /// of this tree's position range or if a conflict with an existing subtree root is detected.
    pub fn batch_insert<C, I>(
        &self,
        start: Position,
        values: I,
    ) -> Result<Option<BatchInsertionResult<H, C, I>>, InsertionError>
    where
        C: Clone + Ord,
        I: Iterator<Item = (H, Retention<C>)>,
    {
        trace!("Batch inserting into {:?} from {:?}", self.root_addr, start);
        let subtree_range = self.root_addr.position_range();
        let contains_start = subtree_range.contains(&start);
        if contains_start {
            let position_range = Range {
                start,
                end: subtree_range.end,
            };
            Self::from_iter(position_range, self.root_addr.level(), values)
                .map(|mut res| {
                    let (subtree, mut incomplete) = self
                        .clone()
                        .insert_subtree(res.subtree, res.contains_marked)?;
                    res.subtree = subtree;
                    res.incomplete.append(&mut incomplete);
                    Ok(res)
                })
                .transpose()
        } else {
            Err(InsertionError::OutOfRange(start, subtree_range))
        }
    }

    /// Builds a [`LocatedPrunableTree`] from an iterator of level-0 leaves.
    ///
    /// This may be used in conjunction with [`ShardTree::insert_tree`] to support
    /// partially-parallelizable tree construction. Multiple subtrees may be constructed in
    /// parallel from iterators over (preferably, though not necessarily) disjoint leaf ranges, and
    /// [`ShardTree::insert_tree`] may be used to insert those subtrees into the [`ShardTree`] in
    /// arbitrary order.
    ///
    /// # Parameters:
    /// * `position_range` - The range of leaf positions at which values will be inserted. This
    ///   range is also used to place an upper bound on the number of items that will be consumed
    ///   from the `values` iterator.
    /// * `prune_below` - Nodes with [`Retention::Ephemeral`] retention that are not required to be retained
    ///   in order to construct a witness for a marked node or to make it possible to rewind to a
    ///   checkpointed node may be pruned so long as their address is at less than the specified
    ///   level.
    /// * `values` - The iterator of `(H, Retention)` pairs from which to construct the tree.
    pub fn from_iter<C, I>(
        position_range: Range<Position>,
        prune_below: Level,
        mut values: I,
    ) -> Option<BatchInsertionResult<H, C, I>>
    where
        C: Clone + Ord,
        I: Iterator<Item = (H, Retention<C>)>,
    {
        trace!(
            position_range = ?position_range,
            prune_below = ?prune_below,
            "Creating minimal tree for insertion"
        );

        // A stack of complete subtrees to be inserted as descendants into the subtree labeled
        // with the addresses at which they will be inserted, along with their root hashes.
        let mut fragments: Vec<(Self, bool)> = vec![];
        let mut position = position_range.start;
        let mut checkpoints: BTreeMap<C, Position> = BTreeMap::new();
        while position < position_range.end {
            if let Some((value, retention)) = values.next() {
                if let Retention::Checkpoint { id, .. } = &retention {
                    checkpoints.insert(id.clone(), position);
                }

                let rflags = RetentionFlags::from(retention);
                let mut subtree = LocatedTree {
                    root_addr: Address::from(position),
                    root: Tree::leaf((value.clone(), rflags)),
                };

                if position.is_right_child() {
                    // At right-hand positions, we are completing a subtree and so we unite
                    // fragments up the stack until we get the largest possible subtree
                    while let Some((potential_sibling, marked)) = fragments.pop() {
                        if potential_sibling.root_addr.parent() == subtree.root_addr.parent() {
                            subtree = unite(potential_sibling, subtree, prune_below);
                        } else {
                            // this is not a sibling node, so we push it back on to the stack
                            // and are done
                            fragments.push((potential_sibling, marked));
                            break;
                        }
                    }
                }

                fragments.push((subtree, rflags.is_marked()));
                position += 1;
            } else {
                break;
            }
        }
        trace!("Initial fragments: {:?}", fragments);

        if position > position_range.start {
            let last_position = position - 1;
            let minimal_tree_addr =
                Address::from(position_range.start).common_ancestor(&last_position.into());
            trace!("Building minimal tree at {:?}", minimal_tree_addr);
            build_minimal_tree(fragments, minimal_tree_addr, prune_below).map(
                |(to_insert, contains_marked, incomplete)| BatchInsertionResult {
                    subtree: to_insert,
                    contains_marked,
                    incomplete,
                    max_insert_position: last_position,
                    checkpoints,
                    remainder: values,
                },
            )
        } else {
            None
        }
    }

    /// Builds a [`LocatedPrunableTree`] from a slice of level-0 leaves.
    ///
    /// This method returns the same tree as [`Self::from_iter`] returns for the same leaves. It
    /// hashes the tree one level at a time, so a [`Hashable`] type that overrides
    /// [`Hashable::combine_pairs`] can hash many pairs in one call.
    ///
    /// - Preconditions: none.
    /// - Inputs:
    ///   - `start` is the position of the first leaf.
    ///   - `prune_below` is a level of the tree. This method prunes a node only if the address
    ///     of the node is below this level, as [`Self::from_iter`] does.
    ///   - `leaves` holds the hash and the retention of each leaf, in order of position.
    /// - Outputs:
    ///   - `None` if `leaves` is empty, or if the last leaf would have a position greater than
    ///     `u64::MAX`.
    ///   - Otherwise, the result that [`Self::from_iter`] returns for the same leaves at the
    ///     same positions. The `remainder` of the result has no items.
    /// - Side effects:
    ///   - Calls [`Hashable::combine_pairs`] at most once for each level below `prune_below`.
    ///     Does not call [`Hashable::combine`].
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    #[allow(clippy::type_complexity)]
    pub fn from_leaves<C>(
        start: Position,
        prune_below: Level,
        leaves: &[(H, Retention<C>)],
    ) -> Option<BatchInsertionResult<H, C, core::iter::Empty<(H, Retention<C>)>>>
    where
        C: Clone + Ord,
    {
        let first_position = u64::from(start);
        let leaf_count = u64::try_from(leaves.len()).ok()?;
        // This is `None` if there are no leaves, or if the last position is too great.
        let last_position = first_position.checked_add(leaf_count.checked_sub(1)?)?;

        let checkpoints: BTreeMap<C, Position> = leaves
            .iter()
            .zip(first_position..=last_position)
            .filter_map(|((_, retention), position)| match retention {
                Retention::Checkpoint { id, .. } => Some((id.clone(), Position::from(position))),
                _ => None,
            })
            .collect();

        // `from_iter` records for each complete subtree whether its last leaf is marked.
        let is_marked = |offset: usize| {
            leaves
                .get(offset)
                .is_some_and(|(_, retention)| RetentionFlags::from(retention).is_marked())
        };

        // `nodes` holds the root of each complete subtree at `level` that the leaves form, in
        // order of position. `first_index` and `last_index` are the indices, at `level`, of the
        // first and the last of these subtrees. `leaf_offsets` holds the offsets in `leaves` of
        // the leaves of these subtrees.
        let mut level = Level::from(0);
        let (mut first_index, mut last_index) = (first_position, last_position);
        let mut leaf_offsets = 0..leaves.len();
        let mut nodes: Vec<PrunableTree<H>> = leaves
            .iter()
            .map(|(value, retention)| Tree::leaf((value.clone(), RetentionFlags::from(retention))))
            .collect();

        // The complete subtrees that have no sibling among the leaves. `left_fragments` is in
        // order of position, and `right_fragments` is in reverse order of position.
        let mut left_fragments: Vec<(Self, bool)> = vec![];
        let mut right_fragments: Vec<(Self, bool)> = vec![];
        while !nodes.is_empty() {
            // Each subtree at `level` has this number of leaves.
            let subtree_size = leaf_offsets.len() / nodes.len();

            // If the last subtree is a left child, its right sibling is not among the leaves.
            if last_index % 2 == 0 {
                if let Some(root) = nodes.pop() {
                    let root_addr = Address::from_parts(level, last_index);
                    let marked = is_marked(leaf_offsets.end - 1);
                    right_fragments.push((LocatedTree { root_addr, root }, marked));
                    leaf_offsets.end -= subtree_size;
                }
            }
            // If the first subtree is a right child, its left sibling is not among the leaves.
            let mut siblings = nodes.into_iter();
            if first_index % 2 == 1 {
                if let Some(root) = siblings.next() {
                    let root_addr = Address::from_parts(level, first_index);
                    let marked = is_marked(leaf_offsets.start + subtree_size - 1);
                    left_fragments.push((LocatedTree { root_addr, root }, marked));
                    leaf_offsets.start += subtree_size;
                }
            }

            nodes = unite_pairs(level, prune_below, siblings.collect());
            first_index = first_index / 2 + first_index % 2;
            last_index = last_index.saturating_sub(1) / 2;
            level = level + 1;
        }
        let fragments: Vec<(Self, bool)> = left_fragments
            .into_iter()
            .chain(right_fragments.into_iter().rev())
            .collect();

        let last_position = Position::from(last_position);
        let minimal_tree_addr = Address::from(start).common_ancestor(&last_position.into());
        build_minimal_tree(fragments, minimal_tree_addr, prune_below).map(
            |(subtree, contains_marked, incomplete)| BatchInsertionResult {
                subtree,
                contains_marked,
                incomplete,
                max_insert_position: last_position,
                checkpoints,
                remainder: core::iter::empty(),
            },
        )
    }
}

/// Unites each pair of sibling trees into one tree at the next level.
///
/// - Preconditions: `children` holds an even number of trees, and none of them is
///   [`Node::Nil`]. Each tree at an even index is the left sibling of the tree after it.
/// - Inputs:
///   - `level` is the level of the root of each tree in `children`.
///   - `prune_below` is the level below which a pair of leaves can become one leaf.
///   - `children` holds the trees to unite, in order of position.
/// - Outputs: one tree for each pair, in order of position. Each tree is equal to the root that
///   [`unite`] returns for that pair and `prune_below`.
/// - Side effects:
///   - Calls [`Hashable::combine_pairs`] at most once. Does not call [`Hashable::combine`].
///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
///
/// [`Node::Nil`]: crate::tree::Node::Nil
fn unite_pairs<H>(
    level: Level,
    prune_below: Level,
    children: Vec<PrunableTree<H>>,
) -> Vec<PrunableTree<H>>
where
    H: Hashable + Clone + PartialEq,
{
    // For each pair that becomes one leaf: the two hashes to combine, and the flags of the leaf.
    let (pairs, _) = children.as_chunks::<2>();
    let mergeable: Vec<Option<(&H, &H, RetentionFlags)>> = pairs
        .iter()
        .map(|[left, right]| {
            if level < prune_below {
                PrunableTree::mergeable_leaves(left, right)
            } else {
                None
            }
        })
        .collect();

    let pair_count = mergeable.iter().flatten().count();
    let hashes = if pair_count == 0 {
        Vec::new()
    } else {
        H::combine_pairs(
            level,
            mergeable
                .iter()
                .flatten()
                .map(|(left, right, _)| (*left, *right)),
        )
    };
    assert_eq!(
        hashes.len(),
        pair_count,
        "`Hashable::combine_pairs` must return exactly one node for each pair"
    );

    let mut hashes = hashes.into_iter();
    let merged_leaves: Vec<Option<(H, RetentionFlags)>> = mergeable
        .into_iter()
        .map(|pair| pair.and_then(|(_, _, flags)| hashes.next().map(|hash| (hash, flags))))
        .collect();

    let mut children = children.into_iter();
    let pairs = core::iter::from_fn(|| Some((children.next()?, children.next()?)));
    merged_leaves
        .into_iter()
        .zip(pairs)
        .map(|(merged_leaf, (left, right))| match merged_leaf {
            Some(value) => Tree::leaf(value),
            None => Tree::parent(None, left, right),
        })
        .collect()
}

// Unite two subtrees by either adding a parent node, or a leaf containing the Merkle root
// of such a parent if both nodes are ephemeral leaves.
//
// `unite` is only called when both root addrs have the same parent.  `batch_insert` never
// constructs Nil nodes, so we don't create any incomplete root information here.
fn unite<H>(
    lroot: LocatedPrunableTree<H>,
    rroot: LocatedPrunableTree<H>,
    prune_below: Level,
) -> LocatedPrunableTree<H>
where
    H: Hashable + Clone + PartialEq,
{
    assert_eq!(lroot.root_addr.parent(), rroot.root_addr.parent());
    LocatedTree {
        root_addr: lroot.root_addr.parent(),
        root: if lroot.root_addr.level() < prune_below {
            Tree::unite(lroot.root_addr.level(), None, lroot.root, rroot.root)
        } else {
            Tree::parent(None, lroot.root, rroot.root)
        },
    }
}

/// Combines the given subtree with an empty sibling node to obtain the next level
/// subtree.
///
/// `expect_left_child` is set to a constant at each callsite, to ensure that this
/// function is only called on either the left-most or right-most subtree.
fn combine_with_empty<H>(
    root: LocatedPrunableTree<H>,
    expect_left_child: bool,
    incomplete: &mut Vec<IncompleteAt>,
    contains_marked: bool,
    prune_below: Level,
) -> LocatedPrunableTree<H>
where
    H: Hashable + Clone + PartialEq,
{
    assert_eq!(expect_left_child, root.root_addr.is_left_child());
    let sibling_addr = root.root_addr.sibling();
    incomplete.push(IncompleteAt {
        address: sibling_addr,
        required_for_witness: contains_marked,
    });
    let sibling = LocatedTree {
        root_addr: sibling_addr,
        root: Tree::empty(),
    };
    let (lroot, rroot) = if root.root_addr.is_left_child() {
        (root, sibling)
    } else {
        (sibling, root)
    };
    unite(lroot, rroot, prune_below)
}

// Builds a single tree from the provided stack of subtrees, which must be non-overlapping
// and in position order. Returns the resulting tree, a flag indicating whether the
// resulting tree contains a `MARKED` node, and the vector of [`IncompleteAt`] values for
// [`Node::Nil`] nodes that were introduced in the process of constructing the tree.
fn build_minimal_tree<H>(
    mut xs: Vec<(LocatedPrunableTree<H>, bool)>,
    root_addr: Address,
    prune_below: Level,
) -> Option<(LocatedPrunableTree<H>, bool, Vec<IncompleteAt>)>
where
    H: Hashable + Clone + PartialEq,
{
    // First, consume the stack from the right, building up a single tree
    // until we can't combine any more.
    if let Some((mut cur, mut contains_marked)) = xs.pop() {
        let mut incomplete = vec![];
        while let Some((top, top_marked)) = xs.pop() {
            while cur.root_addr.level() < top.root_addr.level() {
                cur = combine_with_empty(cur, true, &mut incomplete, top_marked, prune_below);
            }

            if cur.root_addr.level() == top.root_addr.level() {
                contains_marked = contains_marked || top_marked;
                if cur.root_addr.is_right_child() {
                    // We have a left child and a right child, so unite them.
                    cur = unite(top, cur, prune_below);
                } else {
                    // This is a left child, so we build it up one more level and then
                    // we've merged as much as we can from the right and need to work from
                    // the left
                    xs.push((top, top_marked));
                    cur = combine_with_empty(cur, true, &mut incomplete, top_marked, prune_below);
                    break;
                }
            } else {
                // top.root_addr.level < cur.root_addr.level, so we've merged as much as we
                // can from the right and now need to work from the left.
                xs.push((top, top_marked));
                break;
            }
        }

        // Ensure we can work from the left in a single pass by making this right-most subtree
        while cur.root_addr.level() + 1 < root_addr.level() {
            cur = combine_with_empty(cur, true, &mut incomplete, contains_marked, prune_below);
        }

        // push our accumulated max-height right hand node back on to the stack.
        xs.push((cur, contains_marked));

        // From the stack of subtrees, construct a single sparse tree that can be
        // inserted/merged into the existing tree
        let res_tree = xs.into_iter().fold(
            None,
            |acc: Option<LocatedPrunableTree<H>>, (next_tree, next_marked)| {
                if let Some(mut prev_tree) = acc {
                    // add nil branches to build up the left tree until we can merge it
                    // with the right
                    while prev_tree.root_addr.level() < next_tree.root_addr.level() {
                        contains_marked = contains_marked || next_marked;
                        prev_tree = combine_with_empty(
                            prev_tree,
                            false,
                            &mut incomplete,
                            next_marked,
                            prune_below,
                        );
                    }

                    Some(unite(prev_tree, next_tree, prune_below))
                } else {
                    Some(next_tree)
                }
            },
        );

        res_tree.map(|t| (t, contains_marked, incomplete))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use incrementalmerkletree::{Address, Level, Position, Retention};

    use super::{LocatedPrunableTree, RetentionFlags};
    use crate::{
        store::memory::MemoryShardStore,
        tree::tests::{leaf, nil, parent},
        BatchInsertionResult, ShardTree,
    };

    #[test]
    fn located_from_iter_non_sibling_adjacent() {
        let res = LocatedPrunableTree::from_iter::<(), _>(
            Position::from(3)..Position::from(5),
            Level::new(0),
            vec![
                ("d".to_string(), Retention::Ephemeral),
                ("e".to_string(), Retention::Ephemeral),
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(
            res.subtree,
            LocatedPrunableTree {
                root_addr: Address::from_parts(3.into(), 0),
                root: parent(
                    parent(
                        nil(),
                        parent(nil(), leaf(("d".to_string(), RetentionFlags::EPHEMERAL)))
                    ),
                    parent(
                        parent(leaf(("e".to_string(), RetentionFlags::EPHEMERAL)), nil()),
                        nil()
                    )
                )
            },
        );
    }

    #[test]
    fn located_insert() {
        let tree = LocatedPrunableTree::empty(Address::from_parts(Level::from(2), 0));
        let (base, _, _) = tree
            .append::<()>("a".to_string(), Retention::Ephemeral)
            .unwrap();
        assert_eq!(base.right_filled_root(), Ok("a___".to_string()));

        // Perform an in-order insertion.
        let (in_order, pos, _) = base
            .append::<()>("b".to_string(), Retention::Ephemeral)
            .unwrap();
        assert_eq!(pos, 1.into());
        assert_eq!(in_order.right_filled_root(), Ok("ab__".to_string()));

        // On the same tree, perform an out-of-order insertion.
        let out_of_order = base
            .batch_insert::<(), _>(
                Position::from(3),
                vec![("d".to_string(), Retention::Ephemeral)].into_iter(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            out_of_order.subtree,
            LocatedPrunableTree {
                root_addr: Address::from_parts(2.into(), 0),
                root: parent(
                    parent(leaf(("a".to_string(), RetentionFlags::EPHEMERAL)), nil()),
                    parent(nil(), leaf(("d".to_string(), RetentionFlags::EPHEMERAL)))
                )
            }
        );

        let complete = out_of_order
            .subtree
            .batch_insert::<(), _>(
                Position::from(1),
                vec![
                    ("b".to_string(), Retention::Ephemeral),
                    ("c".to_string(), Retention::Ephemeral),
                ]
                .into_iter(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(complete.subtree.right_filled_root(), Ok("abcd".to_string()));
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn build_insert_tree_and_batch_insert(
        leaves: Vec<(String, Retention<usize>)>,
    ) -> (
        ShardTree<MemoryShardStore<String, usize>, 6, 3>,
        ShardTree<MemoryShardStore<String, usize>, 6, 3>,
    ) {
        let max_checkpoints = 10;
        let start = Position::from(0);
        let end = start + leaves.len() as u64;

        // Construct a tree using `ShardTree::insert_tree`.
        let mut left = ShardTree::new(MemoryShardStore::empty(), max_checkpoints);
        if let Some(BatchInsertionResult {
            subtree,
            checkpoints,
            mut remainder,
            ..
        }) = LocatedPrunableTree::from_iter(start..end, 0.into(), leaves.clone().into_iter())
        {
            assert_eq!(remainder.next(), None);
            left.insert_tree(subtree, checkpoints).unwrap();
        }

        // Construct a tree using `ShardTree::batch_insert`.
        let mut right = ShardTree::new(MemoryShardStore::empty(), max_checkpoints);
        right.batch_insert(start, leaves.into_iter()).unwrap();

        (left, right)
    }

    #[test]
    fn batch_insert_matches_insert_tree() {
        {
            let (lhs, rhs) = build_insert_tree_and_batch_insert(vec![]);
            assert_eq!(lhs.max_leaf_position(None), Ok(None));
            assert_eq!(rhs.max_leaf_position(None), Ok(None));
        }

        for i in 0..64 {
            let num_leaves = i + 1;
            let leaves =
                std::iter::repeat_n(("a".into(), Retention::Ephemeral), num_leaves).collect();
            let expected_root = (0..64)
                .map(|c| if c < num_leaves { 'a' } else { '_' })
                .fold(String::with_capacity(64), |mut acc, c| {
                    acc.push(c);
                    acc
                });

            let (lhs, rhs) = build_insert_tree_and_batch_insert(leaves);
            assert_eq!(
                lhs.max_leaf_position(None),
                Ok(Some(Position::from(i as u64)))
            );
            assert_eq!(
                rhs.max_leaf_position(None),
                Ok(Some(Position::from(i as u64)))
            );

            assert_eq!(
                lhs.root_at_checkpoint_depth(None).unwrap().as_ref(),
                Some(&expected_root)
            );
            assert_eq!(
                rhs.root_at_checkpoint_depth(None).unwrap().as_ref(),
                Some(&expected_root)
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    use super::tests::build_insert_tree_and_batch_insert;
    use crate::testing::{arb_char_str, arb_leaves};

    proptest! {
        #[test]
        fn batch_insert_matches_insert_tree(
            leaves in arb_leaves(arb_char_str())
        ) {
            let (left, right) = build_insert_tree_and_batch_insert(leaves);

            // Check that the resulting trees are equal.
            assert_eq!(left.root_at_checkpoint_depth(None), right.root_at_checkpoint_depth(None));
        }
    }
}

/// Tests of [`LocatedPrunableTree::from_leaves`].
#[cfg(test)]
mod from_leaves_tests {
    use std::collections::BTreeMap;

    use incrementalmerkletree::{
        frontier::testing::{arb_test_node, TestNode},
        Address, Hashable, Level, Marking, Position, Retention,
    };
    use proptest::prelude::*;

    use super::{BatchInsertionResult, LocatedPrunableTree, RetentionFlags};
    use crate::{
        tests::{take_batch_only_calls, BatchOnly},
        tree::tests::{leaf, parent},
        IncompleteAt, LocatedTree,
    };

    /// The fields of a [`BatchInsertionResult`], without the remainder.
    type ResultParts<H, C> = (
        LocatedPrunableTree<H>,
        bool,
        Vec<IncompleteAt>,
        Position,
        BTreeMap<C, Position>,
    );

    /// Returns the fields of `result`, without the remainder. Two results that come from
    /// different types of leaf source can then be compared.
    fn result_parts<H, C, I>(
        result: Option<BatchInsertionResult<H, C, I>>,
    ) -> Option<ResultParts<H, C>>
    where
        C: Ord,
        I: Iterator<Item = (H, Retention<C>)>,
    {
        result.map(|result| {
            (
                result.subtree,
                result.contains_marked,
                result.incomplete,
                result.max_insert_position,
                result.checkpoints,
            )
        })
    }

    /// Returns a strategy for up to `max_count` leaves. Each kind of retention occurs. The
    /// identifier of each checkpoint is the index of its leaf.
    pub(super) fn arb_leaves_of_each_retention<H>(
        arb_leaf: H,
        max_count: usize,
    ) -> impl Strategy<Value = Vec<(H::Value, Retention<usize>)>>
    where
        H: Strategy,
    {
        let arb_marking = prop_oneof![
            Just(Marking::None),
            Just(Marking::Marked),
            Just(Marking::Reference),
        ];
        // `None` stands for a checkpoint with the marking that follows it.
        let arb_retention = prop_oneof![
            4 => Just(Some(Retention::Ephemeral)),
            1 => Just(Some(Retention::Marked)),
            1 => Just(Some(Retention::Reference)),
            2 => Just(None),
        ];
        proptest::collection::vec((arb_leaf, arb_retention, arb_marking), 0..=max_count).prop_map(
            |leaves| {
                leaves
                    .into_iter()
                    .enumerate()
                    .map(|(id, (leaf, retention, marking))| {
                        (
                            leaf,
                            retention.unwrap_or(Retention::Checkpoint { id, marking }),
                        )
                    })
                    .collect()
            },
        )
    }

    #[test]
    fn from_leaves_returns_none_for_no_leaves() {
        let result = LocatedPrunableTree::<String>::from_leaves::<()>(
            Position::from(3),
            Level::from(2),
            &[],
        );
        assert!(result.is_none());
    }

    #[test]
    fn from_leaves_returns_none_when_the_last_position_exceeds_u64_max() {
        let leaves = [
            ("a".to_string(), Retention::<()>::Ephemeral),
            ("b".to_string(), Retention::Ephemeral),
        ];
        let result =
            LocatedPrunableTree::from_leaves(Position::from(u64::MAX), Level::from(2), &leaves);
        assert!(result.is_none());
    }

    #[test]
    fn from_leaves_builds_a_tree_that_ends_at_position_u64_max() {
        let leaves = [
            ("a".to_string(), Retention::<()>::Ephemeral),
            ("b".to_string(), Retention::Ephemeral),
            ("c".to_string(), Retention::Ephemeral),
        ];
        let result =
            LocatedPrunableTree::from_leaves(Position::from(u64::MAX - 2), Level::from(2), &leaves)
                .expect("the three leaves have positions of at most `u64::MAX`");

        // The last two leaves are siblings. The first leaf is the right child of their uncle.
        assert_eq!(
            result.subtree,
            LocatedTree {
                root_addr: Address::from_parts(Level::from(2), u64::MAX >> 2),
                root: parent(
                    parent(
                        crate::tree::tests::nil(),
                        leaf(("a".to_string(), RetentionFlags::EPHEMERAL))
                    ),
                    leaf(("bc".to_string(), RetentionFlags::EPHEMERAL)),
                ),
            }
        );
        assert_eq!(result.max_insert_position, Position::from(u64::MAX));
    }

    #[test]
    fn from_leaves_keeps_marked_leaves_and_checkpoints() {
        let leaves = [
            ("a".to_string(), Retention::Ephemeral),
            ("b".to_string(), Retention::Marked),
            (
                "c".to_string(),
                Retention::Checkpoint {
                    id: 7,
                    marking: Marking::None,
                },
            ),
            ("d".to_string(), Retention::Ephemeral),
            ("e".to_string(), Retention::Ephemeral),
            ("f".to_string(), Retention::Ephemeral),
            ("g".to_string(), Retention::Ephemeral),
            (
                "h".to_string(),
                Retention::Checkpoint {
                    id: 8,
                    marking: Marking::None,
                },
            ),
        ];
        let mut result =
            LocatedPrunableTree::from_leaves(Position::from(0), Level::from(3), &leaves)
                .expect("there are leaves, and their positions are small");

        assert_eq!(
            result.subtree,
            LocatedTree {
                root_addr: Address::from_parts(Level::from(3), 0),
                root: parent(
                    parent(
                        // The marked leaf `b` is kept, so its sibling is kept too.
                        parent(
                            leaf(("a".to_string(), RetentionFlags::EPHEMERAL)),
                            leaf(("b".to_string(), RetentionFlags::MARKED)),
                        ),
                        // The checkpoint at `c` is at a left child, so `c` is not merged.
                        parent(
                            leaf(("c".to_string(), RetentionFlags::CHECKPOINT)),
                            leaf(("d".to_string(), RetentionFlags::EPHEMERAL)),
                        ),
                    ),
                    // The checkpoint at `h` is at the last position of each merged subtree.
                    leaf(("efgh".to_string(), RetentionFlags::CHECKPOINT)),
                ),
            }
        );
        assert_eq!(
            result.checkpoints,
            BTreeMap::from([(7, Position::from(2)), (8, Position::from(7))])
        );
        assert_eq!(result.max_insert_position, Position::from(7));
        assert_eq!(result.incomplete, vec![]);
        assert_eq!(result.remainder.next(), None);
    }

    #[test]
    fn from_leaves_hashes_each_level_with_one_call_to_combine_pairs() {
        // The leaves have the positions 5 to 15. Their complete subtrees are the leaf 5, the
        // leaves 6 to 7, and the leaves 8 to 15.
        let leaves: Vec<(BatchOnly, Retention<()>)> = (5..16)
            .map(|position: u64| (BatchOnly(position.to_string()), Retention::Ephemeral))
            .collect();

        take_batch_only_calls();
        let result = LocatedPrunableTree::from_leaves(Position::from(5), Level::from(16), &leaves)
            .expect("there are leaves, and their positions are small");

        assert_eq!(
            take_batch_only_calls(),
            vec![
                (Level::from(0), 5),
                (Level::from(1), 2),
                (Level::from(2), 1)
            ]
        );
        assert_eq!(
            result
                .subtree
                .subtree(Address::from_parts(Level::from(3), 1))
                .map(|subtree| subtree.root),
            Some(leaf((
                BatchOnly("2(1(0(89)0(1011))1(0(1213)0(1415)))".to_string()),
                RetentionFlags::EPHEMERAL
            )))
        );
    }

    /// A node whose `combine_pairs` returns one node too few.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DropsLastParent(String);

    impl Hashable for DropsLastParent {
        fn empty_leaf() -> Self {
            DropsLastParent(String::empty_leaf())
        }

        fn combine(level: Level, left: &Self, right: &Self) -> Self {
            DropsLastParent(String::combine(level, &left.0, &right.0))
        }

        fn combine_pairs<'a, I>(level: Level, pairs: I) -> Vec<Self>
        where
            Self: 'a,
            I: IntoIterator<Item = (&'a Self, &'a Self)>,
        {
            let mut parents: Vec<Self> = pairs
                .into_iter()
                .map(|(left, right)| Self::combine(level, left, right))
                .collect();
            parents.pop();
            parents
        }
    }

    #[test]
    #[should_panic(
        expected = "`Hashable::combine_pairs` must return exactly one node for each pair"
    )]
    fn from_leaves_panics_when_combine_pairs_drops_a_node() {
        let leaves = [
            (DropsLastParent("a".to_string()), Retention::<()>::Ephemeral),
            (DropsLastParent("b".to_string()), Retention::Ephemeral),
        ];
        let _ = LocatedPrunableTree::from_leaves(Position::from(0), Level::from(1), &leaves);
    }

    proptest! {
        #[test]
        fn from_leaves_matches_from_iter(
            start in 0u64..70,
            prune_below in 0u8..9,
            leaves in arb_leaves_of_each_retention(arb_test_node(), 70),
        ) {
            let start = Position::from(start);
            let end = start + leaves.len() as u64;
            let prune_below = Level::from(prune_below);

            let expected: Option<ResultParts<TestNode, usize>> = result_parts(
                LocatedPrunableTree::from_iter(start..end, prune_below, leaves.clone().into_iter()),
            );
            let actual = result_parts(LocatedPrunableTree::from_leaves(start, prune_below, &leaves));

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn from_leaves_calls_combine_pairs_at_most_once_for_each_level_below_prune_below(
            start in 0u64..70,
            prune_below in 0u8..9,
            leaves in arb_leaves_of_each_retention("[a-z]".prop_map(BatchOnly), 70),
        ) {
            take_batch_only_calls();
            // `BatchOnly::combine` panics, so this call fails if it hashes one pair at a time.
            let _ = LocatedPrunableTree::from_leaves(
                Position::from(start),
                Level::from(prune_below),
                &leaves,
            );

            let levels: Vec<u8> = take_batch_only_calls()
                .into_iter()
                .map(|(level, _)| u8::from(level))
                .collect();
            prop_assert!(levels.windows(2).all(|pair| pair[0] < pair[1]), "levels: {:?}", levels);
            prop_assert!(levels.iter().all(|level| *level < prune_below), "levels: {:?}", levels);
        }
    }
}

/// Tests of what [`LocatedPrunableTree::from_iter`] and [`LocatedPrunableTree::from_leaves`]
/// report about marked leaves.
#[cfg(test)]
mod marked_leaves_tests {
    use incrementalmerkletree::{Address, Level, Position, Retention};
    use proptest::prelude::*;

    use super::{
        from_leaves_tests::arb_leaves_of_each_retention, LocatedPrunableTree, RetentionFlags,
    };
    use crate::{testing::arb_char_str, IncompleteAt};

    /// The parts of a result that these tests examine: `subtree`, `contains_marked`, and
    /// `incomplete`.
    type Parts = (LocatedPrunableTree<String>, bool, Vec<IncompleteAt>);

    /// Returns the parts of the result of `from_iter`, and then of `from_leaves`, for `leaves`
    /// at the positions from `start`. Returns no parts if `leaves` is empty.
    fn results(start: u64, prune_below: u8, leaves: &[(String, Retention<usize>)]) -> Vec<Parts> {
        let start = Position::from(start);
        let end = start + leaves.len() as u64;
        let prune_below = Level::from(prune_below);
        let from_iter =
            LocatedPrunableTree::from_iter(start..end, prune_below, leaves.iter().cloned())
                .map(|result| (result.subtree, result.contains_marked, result.incomplete));
        let from_leaves = LocatedPrunableTree::from_leaves(start, prune_below, leaves)
            .map(|result| (result.subtree, result.contains_marked, result.incomplete));
        from_iter.into_iter().chain(from_leaves).collect()
    }

    /// Returns `count` leaves with the values `a`, `b`, and so on. The leaf at index `marked`
    /// is marked, and the other leaves are ephemeral.
    fn leaves_with_one_marked(count: u8, marked: Option<u8>) -> Vec<(String, Retention<usize>)> {
        (0..count)
            .map(|index| {
                let value = char::from(b'a' + index).to_string();
                if Some(index) == marked {
                    (value, Retention::Marked)
                } else {
                    (value, Retention::Ephemeral)
                }
            })
            .collect()
    }

    /// Returns the empty node at `position`, with the given flag.
    fn empty_leaf_at(position: u64, required_for_witness: bool) -> IncompleteAt {
        IncompleteAt {
            address: Address::from_parts(Level::from(0), position),
            required_for_witness,
        }
    }

    #[test]
    fn a_marked_leaf_that_is_not_the_last_leaf_of_its_subtree_is_reported() {
        // The leaves have the positions 0 to 3, and the leaf at position 1 is marked.
        let results = results(0, 2, &leaves_with_one_marked(4, Some(1)));

        assert_eq!(results.len(), 2);
        for (_, contains_marked, incomplete) in results {
            assert!(contains_marked);
            assert_eq!(incomplete, vec![]);
        }
    }

    #[test]
    fn an_empty_node_to_the_left_of_a_marked_leaf_is_required_for_its_witness() {
        // The leaves have the positions 1 to 3, and the leaf at position 1 is marked. The
        // node at position 0 is empty, and it is the sibling of the marked leaf.
        let results = results(1, 2, &leaves_with_one_marked(3, Some(0)));

        assert_eq!(results.len(), 2);
        for (_, contains_marked, incomplete) in results {
            assert!(contains_marked);
            assert_eq!(incomplete, vec![empty_leaf_at(0, true)]);
        }
    }

    #[test]
    fn an_empty_node_to_the_right_of_a_marked_leaf_is_required_for_its_witness() {
        // The leaves have the positions 0 to 2, so the node at position 3 is empty. The
        // witness of each of the three leaves needs the root of a subtree that holds that
        // node.
        for marked in 0..3 {
            let results = results(0, 2, &leaves_with_one_marked(3, Some(marked)));

            assert_eq!(results.len(), 2);
            for (_, contains_marked, incomplete) in results {
                assert!(contains_marked, "marked leaf at position {marked}");
                assert_eq!(
                    incomplete,
                    vec![empty_leaf_at(3, true)],
                    "marked leaf at position {marked}"
                );
            }
        }
    }

    #[test]
    fn no_empty_node_is_required_for_a_witness_when_no_leaf_is_marked() {
        // The leaves have the positions 1 to 2, so the nodes at positions 0 and 3 are empty.
        let results = results(1, 2, &leaves_with_one_marked(2, None));

        assert_eq!(results.len(), 2);
        for (_, contains_marked, incomplete) in results {
            assert!(!contains_marked);
            assert_eq!(
                incomplete,
                vec![empty_leaf_at(3, false), empty_leaf_at(0, false)]
            );
        }
    }

    proptest! {
        #[test]
        fn results_report_whether_any_leaf_is_marked(
            start in 0u64..70,
            prune_below in 0u8..9,
            leaves in arb_leaves_of_each_retention(arb_char_str(), 70),
        ) {
            let any_marked = leaves
                .iter()
                .any(|(_, retention)| RetentionFlags::from(retention).is_marked());

            for (subtree, contains_marked, incomplete) in results(start, prune_below, &leaves) {
                prop_assert_eq!(contains_marked, any_marked);
                // A marked leaf is never pruned, so the tree holds it.
                prop_assert_eq!(subtree.root().contains_marked(), any_marked);
                // The witness of a marked leaf needs the root of each subtree beside its
                // path, and each empty node of the tree is in one of those subtrees.
                for node in &incomplete {
                    prop_assert_eq!(node.required_for_witness, any_marked);
                }
                // The result lists each empty node of the tree once.
                let mut listed: Vec<Address> = incomplete.iter().map(|node| node.address).collect();
                listed.sort_by_key(|address| address.position_range_start());
                prop_assert_eq!(listed, subtree.incomplete_nodes());
            }
        }
    }
}
