use alloc::vec::Vec;
use core::mem::size_of;

use crate::{Address, Hashable, Level, MerklePath, Position, Source};

#[cfg(feature = "legacy-api")]
use {alloc::boxed::Box, alloc::collections::VecDeque, core::iter::repeat};

#[cfg(any(test, feature = "test-dependencies"))]
use {
    core::num::{NonZeroU64, NonZeroU8},
    rand::{
        distr::{Distribution, StandardUniform},
        Rng, RngExt,
    },
};

/// Validation errors that can occur during reconstruction of a Merkle frontier from
/// its constituent parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontierError {
    /// An error representing that the number of ommers provided in frontier construction does not
    /// the expected length of the ommers list given the position.
    PositionMismatch { expected_ommers: u8 },
    /// An error representing that the position and/or list of ommers provided to frontier
    /// construction would result in a frontier that exceeds the maximum statically allowed depth
    /// of the tree. `depth` is the minimum tree depth that would be required in order for that
    /// tree to contain the position in question.
    MaxDepthExceeded { depth: u8 },
}

/// A [`NonEmptyFrontier`] is a reduced representation of a Merkle tree, containing a single leaf
/// value, along with the vector of hashes produced by the reduction of previously appended leaf
/// values that will be required when producing a witness for the current leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonEmptyFrontier<H> {
    position: Position,
    leaf: H,
    ommers: Vec<H>,
}

impl<H> NonEmptyFrontier<H> {
    /// Constructs a new frontier with the specified value at position 0.
    pub fn new(leaf: H) -> Self {
        Self {
            position: 0.into(),
            leaf,
            ommers: vec![],
        }
    }

    /// Constructs a new frontier from its constituent parts.
    pub fn from_parts(position: Position, leaf: H, ommers: Vec<H>) -> Result<Self, FrontierError> {
        let expected_ommers = position.past_ommer_count();
        if ommers.len() == expected_ommers.into() {
            Ok(Self {
                position,
                leaf,
                ommers,
            })
        } else {
            Err(FrontierError::PositionMismatch { expected_ommers })
        }
    }

    /// Decomposes the frontier into its constituent parts
    pub fn into_parts(self) -> (Position, H, Vec<H>) {
        (self.position, self.leaf, self.ommers)
    }

    /// Returns the position of the most recently appended leaf.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Returns the leaf most recently appended to the frontier.
    pub fn leaf(&self) -> &H {
        &self.leaf
    }

    /// Returns the list of past hashes required to construct a witness for the
    /// leaf most recently appended to the frontier.
    pub fn ommers(&self) -> &[H] {
        &self.ommers
    }
}

impl<H> NonEmptyFrontier<H>
where
    H: Hashable + Clone,
{
    /// Append a new leaf to the frontier, and recompute ommers by hashing together full subtrees
    /// until an empty ommer slot is found.
    pub fn append(&mut self, leaf: H) {
        let prior_position = self.position;
        let prior_leaf = self.leaf.clone();
        self.position += 1;
        self.leaf = leaf;
        if self.position.is_right_child() {
            // if the new position is a right-hand leaf, the current leaf will directly become an
            // ommer at level 0, and there is no other mutation made to the tree.
            self.ommers.insert(0, prior_leaf);
        } else {
            // if the new position is even, then the current leaf will be hashed
            // with the first ommer, and so forth up the tree.
            let new_root_level = self.position.root_level();

            let mut carry = Some((prior_leaf, 0.into()));
            let mut new_ommers = Vec::with_capacity(self.position.past_ommer_count().into());
            for (addr, source) in prior_position.witness_addrs(new_root_level) {
                if let Source::Past(i) = source {
                    if let Some((carry_ommer, carry_lvl)) = carry.as_ref() {
                        if *carry_lvl == addr.level() {
                            carry = Some((
                                H::combine(addr.level(), &self.ommers[usize::from(i)], carry_ommer),
                                addr.level() + 1,
                            ))
                        } else {
                            // insert the carry at the first empty slot; then the rest of the
                            // ommers will remain unchanged
                            new_ommers.push(carry_ommer.clone());
                            new_ommers.push(self.ommers[usize::from(i)].clone());
                            carry = None;
                        }
                    } else {
                        // when there's no carry, just push on the ommer value
                        new_ommers.push(self.ommers[usize::from(i)].clone());
                    }
                }
            }

            // we carried value out, so we need to push on one more ommer.
            if let Some((carry_ommer, _)) = carry {
                new_ommers.push(carry_ommer);
            }

            self.ommers = new_ommers;
        }
    }

    /// Appends `leaves` to the tree, in order of position.
    ///
    /// The frontier that results is equal to the frontier that results from one call to
    /// [`NonEmptyFrontier::append`] for each leaf.
    ///
    /// - Preconditions: none.
    /// - Inputs: `leaves` are the leaves to add after the last leaf of the tree, in order.
    /// - Outputs: `true` if the frontier now includes every leaf of `leaves`. `false` if the
    ///   last leaf would have a position greater than `u64::MAX`. The frontier does not change
    ///   when the result is `false`.
    /// - Side effects:
    ///   - Calls [`Hashable::combine_pairs`] at most once for each level of the tree. Does not
    ///     call [`Hashable::combine`].
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    ///     The frontier does not change when this method panics.
    #[must_use = "the leaves are not appended if the result is `false`"]
    pub fn append_batch<I: IntoIterator<Item = H>>(&mut self, leaves: I) -> bool {
        self.append_collected(leaves.into_iter().collect(), None::<fn(Address, &[H])>)
    }

    /// Appends `leaves` to the tree, in order of position, and passes each node that these
    /// leaves complete to `visit`.
    ///
    /// A node is complete when the tree contains every leaf below it. `leaves` complete a node
    /// when the last leaf below that node is one of `leaves`. Each leaf completes itself.
    ///
    /// `visit` does not receive a node that was complete before this call. This is true even
    /// if no visitor ever received that node. That is the case for a node that
    /// [`NonEmptyFrontier::append`] or [`NonEmptyFrontier::append_batch`] completed, and for a
    /// node that was complete in a frontier that [`NonEmptyFrontier::from_parts`] made. A
    /// caller that must receive every complete node of a tree must use this method to add
    /// every leaf of that tree.
    ///
    /// The frontier that results is equal to the frontier that results from one call to
    /// [`NonEmptyFrontier::append`] for each leaf.
    ///
    /// - Preconditions: none.
    /// - Inputs:
    ///   - `leaves` are the leaves to add after the last leaf of the tree, in order.
    ///   - `visit` receives the completed nodes. In a call `visit(first, nodes)`, all of
    ///     `nodes` are at the level of `first`, and `nodes[i]` has the index
    ///     `first.index() + i`.
    /// - Outputs: `true` if the frontier now includes every leaf of `leaves`. `false` if the
    ///   last leaf would have a position greater than `u64::MAX`. The frontier does not change
    ///   when the result is `false`.
    /// - Side effects:
    ///   - Calls `visit` at most once for each level, in ascending order of level. Each call
    ///     passes every node of that level that `leaves` complete. The call for level 0
    ///     passes `leaves`. Does not call `visit` if `leaves` is empty or if the result is
    ///     `false`.
    ///   - Calls [`Hashable::combine_pairs`] at most once for each level of the tree. Does not
    ///     call [`Hashable::combine`].
    ///   - Hashes one more pair than [`NonEmptyFrontier::append_batch`] for each node above
    ///     level 0 that the last leaf completes. The frontier does not store those nodes, so
    ///     the next append computes them again.
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    ///     The frontier does not change when this method panics or when `visit` panics.
    ///   - `visit` receives the nodes of each level before the next level is hashed. If this
    ///     method panics, `visit` has already received the nodes of the levels below the
    ///     panic, although the frontier does not change. A visitor that stores nodes must
    ///     discard what it received in this call when the call panics.
    #[must_use = "the leaves are not appended if the result is `false`"]
    pub fn append_batch_visiting<I, V>(&mut self, leaves: I, visit: V) -> bool
    where
        I: IntoIterator<Item = H>,
        V: FnMut(Address, &[H]),
    {
        self.append_collected(leaves.into_iter().collect(), Some(visit))
    }

    /// Appends `leaves` to the tree, in order of position.
    ///
    /// - Preconditions: none.
    /// - Inputs: `leaves` are the leaves to add after the last leaf of the tree, in order.
    ///   `visit`, if it is present, receives each node that `leaves` complete.
    /// - Outputs: `true` if the frontier now includes every leaf of `leaves`. `false` if the
    ///   last leaf would have a position greater than `u64::MAX`. The frontier does not change
    ///   when the result is `false`.
    /// - Side effects: calls `visit` and [`Hashable::combine_pairs`] at most once for each
    ///   level. Panics if `Hashable::combine_pairs` does not return exactly one node for each
    ///   pair. The frontier does not change when this method panics.
    fn append_collected<V>(&mut self, leaves: Vec<H>, visit: Option<V>) -> bool
    where
        V: FnMut(Address, &[H]),
    {
        if leaves.is_empty() {
            return true;
        }
        match Self::with_appended(Some(self), leaves, visit) {
            Some(frontier) => {
                *self = frontier;
                true
            }
            None => false,
        }
    }

    /// Returns the positions of the first leaf and of the last leaf of a batch that is appended
    /// to a tree.
    ///
    /// - Preconditions: none.
    /// - Inputs: `prior` is the frontier of the tree before the append, or `None` if that tree
    ///   is empty. `leaf_count` is the number of leaves in the batch.
    /// - Outputs: the position of the first leaf of the batch, then the position of its last
    ///   leaf. `None` if `leaf_count` is zero, or if the last leaf would have a position
    ///   greater than `u64::MAX`.
    /// - Side effects: none.
    fn appended_positions(prior: Option<&Self>, leaf_count: usize) -> Option<(Position, Position)> {
        let first = match prior {
            Some(frontier) => u64::from(frontier.position).checked_add(1)?,
            None => 0,
        };
        let leaves_after_first = u64::try_from(leaf_count).ok()?.checked_sub(1)?;
        let last = first.checked_add(leaves_after_first)?;
        Some((Position::from(first), Position::from(last)))
    }

    /// Computes the frontier of the tree that results when `leaves` are appended to a tree.
    ///
    /// - Preconditions: none.
    /// - Inputs:
    ///   - `prior` is the frontier of the tree before the append, or `None` if that tree is
    ///     empty.
    ///   - `leaves` are the leaves to append, in order of position.
    ///   - `visit`, if it is present, receives each node that `leaves` complete. In a call
    ///     `visit(first, nodes)`, all of `nodes` are at the level of `first`, and `nodes[i]`
    ///     has the index `first.index() + i`.
    /// - Outputs: the frontier of the tree after the append. `None` if `leaves` is empty, or if
    ///   the last leaf would have a position greater than `u64::MAX`.
    /// - Side effects:
    ///   - Calls `visit` at most once for each level, in ascending order of level, with every
    ///     node of that level that `leaves` complete. Does not call `visit` if the result is
    ///     `None`.
    ///   - Calls [`Hashable::combine_pairs`] at most once for each level. Does not call
    ///     [`Hashable::combine`].
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    fn with_appended<V>(
        prior: Option<&Self>,
        mut leaves: Vec<H>,
        mut visit: Option<V>,
    ) -> Option<Self>
    where
        V: FnMut(Address, &[H]),
    {
        let (first_position, tip_position) = Self::appended_positions(prior, leaves.len())?;
        if let Some(visit) = visit.as_mut() {
            visit(Address::from(first_position), &leaves);
        }
        let tip_leaf = leaves.pop()?;

        // A frontier stores its last leaf, and one ommer for each complete subtree that holds
        // leaves before the last leaf. The loop below goes up the tree one level at a time and
        // adds complete nodes to those subtrees.
        //
        // At each level, `pending` holds the complete nodes of that level that do not have a
        // complete parent yet, in order of index. `pending_first` is the address of the first
        // of them. At level 0 these nodes are the last leaf of the prior tree, then every new
        // leaf except the last one.
        let mut pending: Vec<H> = prior
            .map(|frontier| frontier.leaf.clone())
            .into_iter()
            .chain(leaves)
            .collect();
        let mut pending_first =
            Address::from(prior.map_or(Position::from(0), |frontier| frontier.position));
        let mut prior_ommers = prior.map_or(&[][..], |frontier| &frontier.ommers).iter();
        let mut ommers: Vec<H> = Vec::with_capacity(tip_position.past_ommer_count().into());
        // The visitor also receives each ancestor that the new last leaf completes. The
        // frontier does not store those ancestors. `tip_ancestor` is the ancestor of the new
        // last leaf at the current level, if the new last leaf completes that ancestor.
        let mut tip_ancestor: Option<H> = visit.as_ref().map(|_| tip_leaf.clone());

        while !pending.is_empty() || tip_ancestor.is_some() {
            let level = pending_first.level();

            // If the first pending node is a right child, then its left sibling is the ommer
            // of the prior frontier at this level.
            let left_sibling = if pending_first.is_right_child() {
                prior_ommers.next()
            } else {
                None
            };
            let mut children: Vec<H> = left_sibling.cloned().into_iter().chain(pending).collect();
            // A last child that has no right sibling is the ommer of the new frontier at this
            // level.
            let ommer = if children.len() % 2 == 1 {
                children.pop()
            } else {
                None
            };
            // The new last leaf completes its ancestor at the next level only if the new
            // frontier has an ommer at this level. That ommer is the left child of the
            // ancestor.
            let tip_pair = ommer.as_ref().zip(tip_ancestor.as_ref());
            let has_tip_pair = tip_pair.is_some();

            let (child_pairs, _) = children.as_chunks::<2>();
            let pair_count = child_pairs.len() + tip_pair.iter().count();
            let mut parents = if pair_count == 0 {
                Vec::new()
            } else {
                H::combine_pairs(
                    level,
                    child_pairs
                        .iter()
                        .map(|[left, right]| (left, right))
                        .chain(tip_pair),
                )
            };
            assert_eq!(
                parents.len(),
                pair_count,
                "`Hashable::combine_pairs` must return exactly one node for each pair"
            );

            let parents_first = pending_first.parent();
            if let Some(visit) = visit.as_mut() {
                let (first, completed) = Self::completed_by_append(prior, parents_first, &parents);
                if !completed.is_empty() {
                    visit(first, completed);
                }
            }

            // The parent of the tip pair is the last of `parents`. It is not a pending node,
            // because the new last leaf stays out of the ommers.
            tip_ancestor = if has_tip_pair { parents.pop() } else { None };
            ommers.extend(ommer);
            pending = parents;
            pending_first = parents_first;
        }
        // The append did not reach the remaining ommers of the prior frontier.
        ommers.extend(prior_ommers.cloned());

        Some(NonEmptyFrontier {
            position: tip_position,
            leaf: tip_leaf,
            ommers,
        })
    }

    /// Selects the nodes that an append completed from the nodes that it computed at one level.
    ///
    /// An append computes the ancestor of the prior last leaf at each level that it reaches,
    /// because the nodes above need that ancestor. If the prior last leaf is the last leaf
    /// below that ancestor, then the ancestor was complete before the append. The frontier
    /// does not store such an ancestor, so the append computes it again, but the append did
    /// not complete it. Every other node that the append computes has its last leaf in the
    /// appended leaves.
    ///
    /// - Preconditions: `parents` are the nodes that the append computed at one level, in
    ///   order of index. `parents_first` is the address of `parents[0]`. If `prior` is present,
    ///   the last leaf of `prior` is below `parents_first`.
    /// - Inputs: `prior` is the frontier of the tree before the append, or `None` if that tree
    ///   is empty. `parents_first` and `parents` are as the preconditions describe.
    /// - Outputs: the nodes of `parents` that the append completed, after the address of the
    ///   first of them. These nodes are all of `parents`, or all of `parents` except
    ///   `parents[0]`.
    /// - Side effects: none.
    fn completed_by_append<'a>(
        prior: Option<&Self>,
        parents_first: Address,
        parents: &'a [H],
    ) -> (Address, &'a [H]) {
        let first_was_complete = prior
            .is_some_and(|frontier| frontier.position.is_complete_subtree(parents_first.level()));
        match parents.split_first() {
            Some((_, after_first)) if first_was_complete => {
                (parents_first.next_at_level(), after_first)
            }
            _ => (parents_first, parents),
        }
    }

    /// Generate the root of the Merkle tree by hashing against empty subtree roots.
    pub fn root(&self, root_level: Option<Level>) -> H {
        let max_level = root_level.unwrap_or_else(|| self.position.root_level());
        self.position
            .witness_addrs(max_level)
            .fold(
                (self.leaf.clone(), Level::from(0)),
                |(digest, complete_lvl), (addr, source)| {
                    // fold up from complete_lvl to addr.level() pairing with empty roots; if
                    // complete_lvl == addr.level() this is just the complete digest to this point
                    let digest = complete_lvl
                        .iter_to(addr.level())
                        .fold(digest, |d, l| H::combine(l, &d, &H::empty_root(l)));

                    let res_digest = match source {
                        Source::Past(i) => {
                            H::combine(addr.level(), &self.ommers[usize::from(i)], &digest)
                        }
                        Source::Future => {
                            H::combine(addr.level(), &digest, &H::empty_root(addr.level()))
                        }
                    };

                    (res_digest, addr.level() + 1)
                },
            )
            .0
    }

    /// Constructs a witness for the leaf at the tip of this frontier, given a source of node
    /// values that complement this frontier.
    ///
    /// If the `complement_nodes` function returns `None` when the value is requested at a given
    /// tree address, the address at which the failure occurs will be returned as an error.
    pub fn witness<F>(&self, depth: u8, complement_nodes: F) -> Result<Vec<H>, Address>
    where
        F: Fn(Address) -> Option<H>,
    {
        // construct a complete trailing edge that includes the data from
        // the following frontier not yet included in the trailing edge.
        self.position()
            .witness_addrs(depth.into())
            .map(|(addr, source)| match source {
                Source::Past(i) => Ok(self.ommers[usize::from(i)].clone()),
                Source::Future => complement_nodes(addr).ok_or(addr),
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(any(test, feature = "test-dependencies"))]
impl<H> NonEmptyFrontier<H>
where
    H: Hashable + Clone,
    StandardUniform: Distribution<H>,
{
    /// Generates a random frontier of a Merkle tree having the specified nonzero size.
    pub fn random_of_size<R>(rng: &mut R, tree_size: NonZeroU64) -> Self
    where
        R: Rng,
    {
        let position = (u64::from(tree_size) - 1).into();
        NonEmptyFrontier::from_parts(
            position,
            rng.random(),
            core::iter::repeat_with(|| rng.random())
                .take(position.past_ommer_count().into())
                .collect(),
        )
        .unwrap()
    }

    pub fn random_with_prior_subtree_roots<R>(
        rng: &mut R,
        tree_size: NonZeroU64,
        subtree_depth: NonZeroU8,
    ) -> (Vec<H>, Self)
    where
        R: Rng,
    {
        let prior_subtree_count: u64 = u64::from(tree_size) >> u8::from(subtree_depth);
        if prior_subtree_count > 0 {
            let prior_roots: Vec<H> = core::iter::repeat_with(|| rng.random())
                .take(prior_subtree_count as usize)
                .collect();

            let subtree_root_level = Level::from(u8::from(subtree_depth));

            // Generate replacement ommers for the random frontier from the prior subtree roots.
            let mut replacement_ommers: Vec<(Level, H)> = vec![];
            let mut roots_iter = prior_roots.iter();
            loop {
                if let Some(top) = replacement_ommers.pop() {
                    if let Some(prev) = replacement_ommers.pop() {
                        if top.0 == prev.0 {
                            // Combine, then continue the outer loop so that we eagerly combine as
                            // many values from the stack as we can before pushing more on.
                            replacement_ommers
                                .push((top.0 + 1, H::combine(top.0, &prev.1, &top.1)));
                            continue;
                        } else {
                            // We can't combine yet, so push `prev` back on. `top` will get pushed
                            // back on or consumed below.
                            replacement_ommers.push(prev);
                        }
                    }

                    if let Some(root) = roots_iter.next() {
                        if top.0 == subtree_root_level {
                            replacement_ommers.push((
                                subtree_root_level + 1,
                                H::combine(subtree_root_level, &top.1, root),
                            ));
                        } else {
                            replacement_ommers.push(top);
                            replacement_ommers.push((subtree_root_level, root.clone()));
                        }
                    } else {
                        // No more roots, so we just push `top` back on and break.
                        replacement_ommers.push(top);
                        break;
                    }
                } else if let Some(root) = roots_iter.next() {
                    replacement_ommers.push((subtree_root_level, root.clone()));
                } else {
                    break;
                }
            }

            let mut result = Self::random_of_size(rng, tree_size);
            let olen = result.ommers.len();
            for (idx, (_, ommer)) in replacement_ommers.into_iter().enumerate() {
                result.ommers[olen - (idx + 1)] = ommer;
            }

            (prior_roots, result)
        } else {
            (vec![], Self::random_of_size(rng, tree_size))
        }
    }
}

/// A possibly-empty Merkle frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frontier<H, const DEPTH: u8> {
    frontier: Option<NonEmptyFrontier<H>>,
}

impl<H, const DEPTH: u8> TryFrom<NonEmptyFrontier<H>> for Frontier<H, DEPTH> {
    type Error = FrontierError;
    fn try_from(f: NonEmptyFrontier<H>) -> Result<Self, FrontierError> {
        if f.position.root_level() <= Level::from(DEPTH) {
            Ok(Frontier { frontier: Some(f) })
        } else {
            Err(FrontierError::MaxDepthExceeded {
                depth: f.position.root_level().into(),
            })
        }
    }
}

impl<H, const DEPTH: u8> Frontier<H, DEPTH> {
    /// Constructs a new empty frontier.
    pub fn empty() -> Self {
        Self { frontier: None }
    }

    /// Constructs a new frontier from its constituent parts.
    ///
    /// Returns an error if the new frontier would exceed the maximum allowed depth or if the list
    /// of ommers provided is not consistent with the position of the leaf.
    pub fn from_parts(position: Position, leaf: H, ommers: Vec<H>) -> Result<Self, FrontierError> {
        NonEmptyFrontier::from_parts(position, leaf, ommers).and_then(Self::try_from)
    }

    /// Return the wrapped NonEmptyFrontier reference, or None if the frontier is empty.
    pub fn value(&self) -> Option<&NonEmptyFrontier<H>> {
        self.frontier.as_ref()
    }

    /// Consumes this wrapper and returns the underlying `Option<NonEmptyFrontier>`
    pub fn take(self) -> Option<NonEmptyFrontier<H>> {
        self.frontier
    }

    /// Returns the amount of memory dynamically allocated for ommer values within the frontier.
    pub fn dynamic_memory_usage(&self) -> usize {
        self.frontier.as_ref().map_or(0, |f| {
            size_of::<usize>() + (f.ommers.capacity() + 1) * size_of::<H>()
        })
    }

    /// Returns the size of the Merkle tree that this frontier corresponds to.
    pub fn tree_size(&self) -> u64 {
        self.frontier
            .as_ref()
            .map_or(0, |f| u64::from(f.position()) + 1)
    }
}

impl<H, const DEPTH: u8> Frontier<H, DEPTH>
where
    H: Hashable + Clone,
{
    /// Appends a new value to the frontier at the next available slot.
    /// Returns true if successful and false if the frontier would exceed
    /// the maximum allowed depth.
    pub fn append(&mut self, value: H) -> bool {
        if let Some(frontier) = self.frontier.as_mut() {
            if frontier.position().is_complete_subtree(DEPTH.into()) {
                false
            } else {
                frontier.append(value);
                true
            }
        } else {
            self.frontier = Some(NonEmptyFrontier::new(value));
            true
        }
    }

    /// Appends `leaves` to the tree, in order of position.
    ///
    /// The frontier that results is equal to the frontier that results from one call to
    /// [`Frontier::append`] for each leaf.
    ///
    /// - Preconditions: none.
    /// - Inputs: `leaves` are the leaves to add after the last leaf of the tree, in order.
    /// - Outputs: `true` if the frontier now includes every leaf of `leaves`. `false` if the
    ///   tree would have more than `2^DEPTH` leaves, or if the last leaf would have a position
    ///   greater than `u64::MAX`. The frontier does not change when the result is `false`.
    /// - Side effects:
    ///   - Calls [`Hashable::combine_pairs`] at most once for each level of the tree. Does not
    ///     call [`Hashable::combine`].
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    ///     The frontier does not change when this method panics.
    #[must_use = "the leaves are not appended if the result is `false`"]
    pub fn append_batch<I: IntoIterator<Item = H>>(&mut self, leaves: I) -> bool {
        self.append_collected(leaves.into_iter().collect(), None::<fn(Address, &[H])>)
    }

    /// Appends `leaves` to the tree, in order of position, and passes each node that these
    /// leaves complete to `visit`.
    ///
    /// A node is complete when the tree contains every leaf below it. `leaves` complete a node
    /// when the last leaf below that node is one of `leaves`. Each leaf completes itself.
    ///
    /// `visit` does not receive a node that was complete before this call. This is true even
    /// if no visitor ever received that node. That is the case for a node that
    /// [`Frontier::append`] or [`Frontier::append_batch`] completed, and for a node that was
    /// complete in a frontier that [`Frontier::from_parts`] made. A caller that must receive
    /// every complete node of a tree must use this method to add every leaf of that tree.
    ///
    /// The frontier that results is equal to the frontier that results from one call to
    /// [`Frontier::append`] for each leaf.
    ///
    /// - Preconditions: none.
    /// - Inputs:
    ///   - `leaves` are the leaves to add after the last leaf of the tree, in order.
    ///   - `visit` receives the completed nodes. In a call `visit(first, nodes)`, all of
    ///     `nodes` are at the level of `first`, and `nodes[i]` has the index
    ///     `first.index() + i`.
    /// - Outputs: `true` if the frontier now includes every leaf of `leaves`. `false` if the
    ///   tree would have more than `2^DEPTH` leaves, or if the last leaf would have a position
    ///   greater than `u64::MAX`. The frontier does not change when the result is `false`.
    /// - Side effects:
    ///   - Calls `visit` at most once for each level, in ascending order of level. Each call
    ///     passes every node of that level that `leaves` complete. The call for level 0
    ///     passes `leaves`. Does not call `visit` if `leaves` is empty or if the result is
    ///     `false`.
    ///   - Calls [`Hashable::combine_pairs`] at most once for each level of the tree. Does not
    ///     call [`Hashable::combine`].
    ///   - Hashes one more pair than [`Frontier::append_batch`] for each node above level 0
    ///     that the last leaf completes. The frontier does not store those nodes, so the next
    ///     append computes them again.
    ///   - Panics if `Hashable::combine_pairs` does not return exactly one node for each pair.
    ///     The frontier does not change when this method panics or when `visit` panics.
    ///   - `visit` receives the nodes of each level before the next level is hashed. If this
    ///     method panics, `visit` has already received the nodes of the levels below the
    ///     panic, although the frontier does not change. A visitor that stores nodes must
    ///     discard what it received in this call when the call panics.
    #[must_use = "the leaves are not appended if the result is `false`"]
    pub fn append_batch_visiting<I, V>(&mut self, leaves: I, visit: V) -> bool
    where
        I: IntoIterator<Item = H>,
        V: FnMut(Address, &[H]),
    {
        self.append_collected(leaves.into_iter().collect(), Some(visit))
    }

    /// Appends `leaves` to the tree, in order of position.
    ///
    /// - Preconditions: none.
    /// - Inputs: `leaves` are the leaves to add after the last leaf of the tree, in order.
    ///   `visit`, if it is present, receives each node that `leaves` complete.
    /// - Outputs: `true` if the frontier now includes every leaf of `leaves`. `false` if the
    ///   tree would have more than `2^DEPTH` leaves, or if the last leaf would have a position
    ///   greater than `u64::MAX`. The frontier does not change when the result is `false`.
    /// - Side effects: calls `visit` and [`Hashable::combine_pairs`] at most once for each
    ///   level. Does not call either of them if the result is `false`. Panics if
    ///   `Hashable::combine_pairs` does not return exactly one node for each pair. The frontier
    ///   does not change when this method panics.
    fn append_collected<V>(&mut self, leaves: Vec<H>, visit: Option<V>) -> bool
    where
        V: FnMut(Address, &[H]),
    {
        if leaves.is_empty() {
            return true;
        }
        let prior = self.frontier.as_ref();
        // A tree of depth `DEPTH` holds the positions that have a root level of at most `DEPTH`.
        let appended = NonEmptyFrontier::appended_positions(prior, leaves.len())
            .filter(|(_, tip_position)| tip_position.root_level() <= Level::from(DEPTH))
            .and_then(|_| NonEmptyFrontier::with_appended(prior, leaves, visit));
        match appended {
            Some(frontier) => {
                self.frontier = Some(frontier);
                true
            }
            None => false,
        }
    }

    /// Obtains the current root of this Merkle frontier by hashing
    /// against empty nodes up to the maximum height of the pruned
    /// tree that the frontier represents.
    pub fn root(&self) -> H {
        self.frontier.as_ref().map_or_else(
            || H::empty_root(DEPTH.into()),
            |frontier| frontier.root(Some(DEPTH.into())),
        )
    }

    /// Constructs a [`MerklePath`] to the leaf at the tip of this frontier, given a source of node
    /// values that complement this frontier.
    ///
    /// If the `complement_nodes` function returns `None` when the value is requested at a given
    /// tree address, the address at which the failure occurs will be returned as an error.
    ///
    /// Returns `Ok(Some(MerklePath))` if successful, `Ok(None)` if the frontier is empty,
    /// or an error containing the address of the failure.
    pub fn witness<F>(&self, complement_nodes: F) -> Result<Option<MerklePath<H, DEPTH>>, Address>
    where
        F: Fn(Address) -> Option<H>,
    {
        self.frontier
            .as_ref()
            .map(|f| {
                f.witness(DEPTH, complement_nodes).map(|path_elems| {
                    MerklePath::from_parts(path_elems, f.position())
                        .expect("Path length should be equal to frontier depth.")
                })
            })
            .transpose()
    }
}

#[cfg(any(test, feature = "test-dependencies"))]
impl<H, const DEPTH: u8> Frontier<H, DEPTH>
where
    H: Hashable + Clone,
    StandardUniform: Distribution<H>,
{
    /// Generates a random frontier of a Merkle tree having the specified size.
    pub fn random_of_size<R>(rng: &mut R, tree_size: u64) -> Self
    where
        R: Rng,
    {
        assert!(tree_size <= 2u64.checked_pow(DEPTH.into()).unwrap());
        Frontier {
            frontier: NonZeroU64::new(tree_size)
                .map(|sz| NonEmptyFrontier::random_of_size(rng, sz)),
        }
    }

    pub fn random_with_prior_subtree_roots<R>(
        rng: &mut R,
        tree_size: u64,
        subtree_depth: NonZeroU8,
    ) -> (Vec<H>, Self)
    where
        R: Rng,
    {
        assert!(tree_size <= 2u64.checked_pow(DEPTH.into()).unwrap());
        NonZeroU64::new(tree_size).map_or((vec![], Frontier::empty()), |tree_size| {
            let (prior_roots, frontier) =
                NonEmptyFrontier::random_with_prior_subtree_roots(rng, tree_size, subtree_depth);
            (
                prior_roots,
                Frontier {
                    frontier: Some(frontier),
                },
            )
        })
    }
}

#[cfg(feature = "legacy-api")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy-api")))]
pub struct PathFiller<H> {
    queue: VecDeque<H>,
}

#[cfg(feature = "legacy-api")]
impl<H> PathFiller<H>
where
    H: Hashable,
{
    pub fn empty() -> Self {
        PathFiller {
            queue: VecDeque::new(),
        }
    }

    pub fn new(queue: VecDeque<H>) -> Self {
        Self { queue }
    }

    pub fn next(&mut self, level: Level) -> H {
        self.queue
            .pop_front()
            .unwrap_or_else(|| H::empty_root(level))
    }
}

/// A Merkle tree of note commitments.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(feature = "legacy-api")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy-api")))]
pub struct CommitmentTree<H, const DEPTH: u8> {
    pub(crate) left: Option<H>,
    pub(crate) right: Option<H>,
    pub(crate) parents: Vec<Option<H>>,
}

#[cfg(feature = "legacy-api")]
impl<H, const DEPTH: u8> CommitmentTree<H, DEPTH> {
    /// Creates an empty tree.
    pub fn empty() -> Self {
        CommitmentTree {
            left: None,
            right: None,
            parents: vec![],
        }
    }

    #[allow(clippy::result_unit_err)]
    pub fn from_parts(
        left: Option<H>,
        right: Option<H>,
        parents: Vec<Option<H>>,
    ) -> Result<Self, ()> {
        if parents.len() < usize::from(DEPTH) {
            Ok(CommitmentTree {
                left,
                right,
                parents,
            })
        } else {
            Err(())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn left(&self) -> &Option<H> {
        &self.left
    }

    pub fn right(&self) -> &Option<H> {
        &self.right
    }

    pub fn parents(&self) -> &Vec<Option<H>> {
        &self.parents
    }

    pub fn leaf(&self) -> Option<&H> {
        self.right.as_ref().or(self.left.as_ref())
    }

    pub fn ommers_iter(&self) -> Box<dyn Iterator<Item = &'_ H> + '_> {
        if self.right.is_some() {
            Box::new(
                self.left
                    .iter()
                    .chain(self.parents.iter().filter_map(|v| v.as_ref())),
            )
        } else {
            Box::new(self.parents.iter().filter_map(|v| v.as_ref()))
        }
    }

    /// Returns the number of leaf nodes in the tree.
    pub fn size(&self) -> usize {
        self.parents.iter().enumerate().fold(
            match (self.left.as_ref(), self.right.as_ref()) {
                (None, None) => 0,
                (Some(_), None) => 1,
                (Some(_), Some(_)) => 2,
                (None, Some(_)) => unreachable!(),
            },
            |acc, (i, p)| {
                // Treat occupation of parents array as a binary number
                // (right-shifted by 1)
                acc + if p.is_some() { 1 << (i + 1) } else { 0 }
            },
        )
    }

    pub(crate) fn is_complete(&self, depth: u8) -> bool {
        if depth == 0 {
            self.left.is_some() && self.right.is_none() && self.parents.is_empty()
        } else {
            self.left.is_some()
                && self.right.is_some()
                && self
                    .parents
                    .iter()
                    .chain(repeat(&None))
                    .take((depth - 1).into())
                    .all(|p| p.is_some())
        }
    }
}

#[cfg(feature = "legacy-api")]
impl<H, const DEPTH: u8> CommitmentTree<H, DEPTH>
where
    H: Hashable + Clone,
{
    pub fn from_frontier(frontier: &Frontier<H, DEPTH>) -> Self {
        frontier.value().map_or_else(Self::empty, |f| {
            let mut ommers_iter = f.ommers().iter().cloned();
            let (left, right) = if f.position().is_right_child() {
                (
                    ommers_iter
                        .next()
                        .expect("An ommer must exist if the frontier position is odd"),
                    Some(f.leaf().clone()),
                )
            } else {
                (f.leaf().clone(), None)
            };

            Self {
                left: Some(left),
                right,
                parents: (1u8..DEPTH)
                    .map(|i| {
                        if u64::from(f.position()) & (1 << i) == 0 {
                            None
                        } else {
                            ommers_iter.next()
                        }
                    })
                    .collect(),
            }
        })
    }

    pub fn to_frontier(&self) -> Frontier<H, DEPTH> {
        if self.size() == 0 {
            Frontier::empty()
        } else {
            let ommers_iter = self.parents.iter().filter_map(|v| v.as_ref()).cloned();
            let (leaf, ommers) = match (self.left.as_ref(), self.right.as_ref()) {
                (Some(a), None) => (a.clone(), ommers_iter.collect()),
                (Some(a), Some(b)) => (
                    b.clone(),
                    Some(a.clone()).into_iter().chain(ommers_iter).collect(),
                ),
                _ => unreachable!(),
            };

            // If a frontier cannot be successfully constructed from the
            // parts of a commitment tree, it is a programming error.
            Frontier::from_parts((self.size() - 1).try_into().unwrap(), leaf, ommers)
                .expect("Frontier should be constructable from CommitmentTree.")
        }
    }

    /// Adds a leaf node to the tree.
    ///
    /// Returns an error if the tree is full.
    #[allow(clippy::result_unit_err)]
    pub fn append(&mut self, node: H) -> Result<(), ()> {
        if self.is_complete(DEPTH) {
            // Tree is full
            return Err(());
        }

        match (&self.left, &self.right) {
            (None, _) => self.left = Some(node),
            (_, None) => self.right = Some(node),
            (Some(l), Some(r)) => {
                let mut combined = H::combine(0.into(), l, r);
                self.left = Some(node);
                self.right = None;

                for i in 0..DEPTH {
                    let i_usize = usize::from(i);
                    if i_usize < self.parents.len() {
                        if let Some(p) = &self.parents[i_usize] {
                            combined = H::combine((i + 1).into(), p, &combined);
                            self.parents[i_usize] = None;
                        } else {
                            self.parents[i_usize] = Some(combined);
                            break;
                        }
                    } else {
                        self.parents.push(Some(combined));
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns the current root of the tree.
    pub fn root(&self) -> H {
        self.root_at_depth(DEPTH, PathFiller::empty())
    }

    pub fn root_at_depth(&self, depth: u8, mut filler: PathFiller<H>) -> H {
        assert!(depth > 0);

        // 1) Hash left and right leaves together.
        //    - Empty leaves are used as needed.
        //    - Note that `filler.next` is side-effecting and so cannot be factored out.
        let leaf_root = H::combine(
            0.into(),
            &self
                .left
                .as_ref()
                .map_or_else(|| filler.next(0.into()), |n| n.clone()),
            &self
                .right
                .as_ref()
                .map_or_else(|| filler.next(0.into()), |n| n.clone()),
        );

        // 2) Extend the parents to the desired depth with None values, then hash from leaf to
        //    root. Roots of the empty subtrees are used as needed.
        self.parents
            .iter()
            .chain(repeat(&None))
            .take((depth - 1).into())
            .zip(0u8..)
            .fold(leaf_root, |root, (p, i)| {
                let level = Level::from(i + 1);
                match p {
                    Some(node) => H::combine(level, node, &root),
                    None => H::combine(level, &root, &filler.next(level)),
                }
            })
    }
}

#[cfg(any(all(test, feature = "std"), feature = "test-dependencies"))]
pub mod testing {
    use core::fmt::Debug;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use rand::{
        distr::{Distribution, StandardUniform},
        Rng, RngExt,
    };

    #[cfg(feature = "std")]
    use {core::hash::Hasher, std::collections::hash_map::DefaultHasher};

    use crate::{frontier::Frontier, Hashable, Level};

    impl<H, const DEPTH: u8> crate::testing::Frontier<H> for super::Frontier<H, DEPTH>
    where
        H: Hashable + Clone,
    {
        fn append(&mut self, value: H) -> bool {
            super::Frontier::append(self, value)
        }

        fn root(&self) -> H {
            super::Frontier::root(self)
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct TestNode(pub u64);

    #[cfg(feature = "std")]
    impl Hashable for TestNode {
        fn empty_leaf() -> Self {
            Self(0)
        }

        fn combine(level: Level, a: &Self, b: &Self) -> Self {
            let mut hasher = DefaultHasher::new();
            hasher.write_u8(level.into());
            hasher.write_u64(a.0);
            hasher.write_u64(b.0);
            Self(hasher.finish())
        }
    }

    impl Distribution<TestNode> for StandardUniform {
        fn sample<R>(&self, rng: &mut R) -> TestNode
        where
            R: Rng + ?Sized,
        {
            TestNode(rng.random())
        }
    }

    pub fn arb_test_node() -> impl Strategy<Value = TestNode> + Clone {
        any::<u64>().prop_map(TestNode)
    }

    pub fn arb_frontier<H, T, const DEPTH: u8>(
        min_size: usize,
        arb_node: T,
    ) -> impl Strategy<Value = Frontier<H, DEPTH>>
    where
        H: Hashable + Clone + Debug,
        T: Strategy<Value = H>,
    {
        assert!((1 << DEPTH) >= min_size + 100);
        vec(arb_node, min_size..(min_size + 100)).prop_map(move |v| {
            let mut frontier = Frontier::empty();
            for node in v.into_iter() {
                frontier.append(node);
            }
            frontier
        })
    }

    #[cfg(feature = "legacy-api")]
    use crate::frontier::CommitmentTree;

    #[cfg(feature = "legacy-api")]
    #[cfg_attr(docsrs, doc(cfg(feature = "legacy-api")))]
    pub fn arb_commitment_tree<H, T, const DEPTH: u8>(
        min_size: usize,
        arb_node: T,
    ) -> impl Strategy<Value = CommitmentTree<H, DEPTH>>
    where
        H: Hashable + Clone + Debug,
        T: Strategy<Value = H>,
    {
        assert!((1 << DEPTH) >= min_size + 100);
        vec(arb_node, min_size..(min_size + 100)).prop_map(move |v| {
            let mut tree = CommitmentTree::empty();
            for node in v.into_iter() {
                tree.append(node).unwrap();
            }
            tree.parents.resize_with((DEPTH - 1).into(), || None);
            tree
        })
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::{String, ToString};
    use core::cell::RefCell;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::{
        testing::{arb_test_node, TestNode},
        *,
    };

    #[cfg(feature = "legacy-api")]
    use super::testing::arb_commitment_tree;

    /// Generates leaf positions that give long carry chains: small positions, positions near a
    /// power of two, and positions over the whole range of `u64`.
    fn arb_position() -> impl Strategy<Value = Position> {
        prop_oneof![
            0u64..512,
            (1u32..=64, -256i128..256).prop_map(|(exponent, offset)| {
                let near_power_of_two = (1i128 << exponent) + offset;
                u64::try_from(near_power_of_two.clamp(0, i128::from(u64::MAX)))
                    .expect("the value was clamped to the range of u64")
            }),
            any::<u64>(),
        ]
        .prop_map(Position::from)
    }

    /// Generates a frontier at an arbitrary position. The node values are arbitrary: they are
    /// not the hashes of the leaves of a real tree.
    fn arb_nonempty_frontier<H, S>(arb_node: S) -> impl Strategy<Value = NonEmptyFrontier<H>>
    where
        H: Clone + core::fmt::Debug,
        S: Strategy<Value = H> + Clone,
    {
        arb_position().prop_flat_map(move |position| {
            let ommer_count = usize::from(position.past_ommer_count());
            (
                arb_node.clone(),
                proptest::collection::vec(arb_node.clone(), ommer_count),
            )
                .prop_map(move |(leaf, ommers)| {
                    NonEmptyFrontier::from_parts(position, leaf, ommers)
                        .expect("the number of ommers matches the position")
                })
        })
    }

    /// Returns the position of the last leaf after `leaf_count` leaves are appended to
    /// `frontier`, or `None` if that position is greater than `u64::MAX`.
    fn tip_after<H>(frontier: &NonEmptyFrontier<H>, leaf_count: usize) -> Option<Position> {
        u64::from(frontier.position())
            .checked_add(u64::try_from(leaf_count).ok()?)
            .map(Position::from)
    }

    /// Returns the address of each node that has its last leaf after `prior_tip` and not after
    /// `new_tip`, in ascending order of level and then of index. A `prior_tip` of `None` means
    /// that the tree was empty.
    fn nodes_completed_after(prior_tip: Option<Position>, new_tip: Position) -> Vec<Address> {
        // A tree of `size` leaves has `size >> level` complete nodes at `level`.
        let prior_size = prior_tip.map_or(0, |tip| u128::from(u64::from(tip)) + 1);
        let new_size = u128::from(u64::from(new_tip)) + 1;
        (0..=64u8)
            .flat_map(|level| {
                ((prior_size >> level)..(new_size >> level)).map(move |index| {
                    let index = u64::try_from(index).expect("a node index is less than 2^64");
                    Address::from_parts(Level::from(level), index)
                })
            })
            .collect()
    }

    /// The arguments of each call that a batch append made to its visitor.
    type VisitCalls<H> = Vec<(Address, Vec<H>)>;

    /// Returns each node in `calls` with its address, in the order of the calls.
    fn visited_nodes<H: Clone>(calls: &[(Address, Vec<H>)]) -> Vec<(Address, H)> {
        calls
            .iter()
            .flat_map(|(first, nodes)| {
                nodes.iter().enumerate().map(move |(offset, node)| {
                    let offset = u64::try_from(offset).expect("a slice length fits in u64");
                    let address = Address::from_parts(first.level(), first.index() + offset);
                    (address, node.clone())
                })
            })
            .collect()
    }

    /// Returns the address of each node in `calls`, in the order of the calls.
    fn visited_addresses<H: Clone>(calls: &[(Address, Vec<H>)]) -> Vec<Address> {
        visited_nodes(calls)
            .into_iter()
            .map(|(address, _)| address)
            .collect()
    }

    /// Returns whether each call in `calls` passed at least one node, and passed nodes of a
    /// higher level than the call before it.
    fn has_one_call_per_level<H>(calls: &[(Address, Vec<H>)]) -> bool {
        calls.iter().all(|(_, nodes)| !nodes.is_empty())
            && calls
                .windows(2)
                .all(|calls| calls[0].0.level() < calls[1].0.level())
    }

    /// Returns the number of nodes above level 0 in the complete subtrees of a tree of
    /// `leaf_count` leaves. This is `leaf_count` minus the number of one bits in `leaf_count`.
    fn complete_parent_count(leaf_count: u64) -> u64 {
        leaf_count - u64::from(leaf_count.count_ones())
    }

    /// Every start size and batch length up to past two carries of depth 6, against `append`
    #[test]
    fn append_batch_equals_append_per_leaf() {
        const DEPTH: u8 = 6;
        let leaf = |i: usize| format!("<{i}>");
        for start in 0..=40 {
            for batch in 0..=24 {
                let mut expected = super::Frontier::<String, DEPTH>::empty();
                for i in 0..start + batch {
                    assert!(expected.append(leaf(i)));
                }
                let mut batched = super::Frontier::<String, DEPTH>::empty();
                assert!(batched.append_batch((0..start).map(leaf)));
                assert!(batched.append_batch((start..start + batch).map(leaf)));
                assert_eq!(batched, expected, "{start} then {batch}");
                assert_eq!(batched.root(), expected.root(), "{start} then {batch}");
            }
        }
    }

    /// Three consecutive batches of every small length: every complete node visited exactly once,
    /// by value (a `String` node = its leaves concatenated), and the frontier = `append`'s
    #[test]
    fn append_batch_visiting_visits_every_complete_node_once() {
        const DEPTH: u8 = 5;
        let leaf = |i: u64| format!("<{i}>");
        // 3 × 10 leaves stays within a depth-5 tree's 32
        for a in 0..=10 {
            for b in 0..=10 {
                for c in 0..=10 {
                    let size = a + b + c;
                    let mut visited = alloc::collections::BTreeMap::new();
                    let mut frontier = super::Frontier::<String, DEPTH>::empty();
                    for range in [0..a, a..a + b, a + b..size] {
                        assert!(
                            frontier.append_batch_visiting(range.map(leaf), |first, nodes| {
                                for (index, node) in (first.index()..).zip(nodes) {
                                    let at = (u8::from(first.level()), index);
                                    let again = visited.insert(at, node.clone());
                                    assert_eq!(again, None, "{at:?} visited twice, {a}/{b}/{c}");
                                }
                            })
                        );
                    }

                    let mut expected = alloc::collections::BTreeMap::new();
                    for level in 0..=DEPTH {
                        for index in 0..size >> level {
                            let leaves = (index << level)..((index + 1) << level);
                            expected.insert((level, index), leaves.map(leaf).collect::<String>());
                        }
                    }
                    assert_eq!(visited, expected, "{a}/{b}/{c}");

                    let mut appended = super::Frontier::<String, DEPTH>::empty();
                    for i in 0..size {
                        assert!(appended.append(leaf(i)));
                    }
                    assert_eq!(frontier, appended, "{a}/{b}/{c}");
                }
            }
        }
    }

    #[test]
    fn append_batch_past_capacity_appends_nothing() {
        let leaves = |range: core::ops::Range<u8>| range.map(|i| i.to_string());
        let mut frontier = super::Frontier::<String, 3>::empty();
        assert!(frontier.append_batch(leaves(0..5)));
        let before = frontier.clone();
        assert!(!frontier.append_batch(leaves(5..9)));
        assert_eq!(frontier, before);
        assert!(frontier.append_batch(leaves(5..8)));
        assert_eq!(frontier.root(), "01234567");
        assert!(!frontier.append_batch(leaves(8..9)));
    }

    #[test]
    fn append_batch_depth_128_does_not_overflow_capacity() {
        let mut frontier = super::Frontier::<String, 128>::empty();
        assert!(frontier.append_batch(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(frontier.tree_size(), 2);
    }

    #[test]
    fn append_batch_rejects_addition_at_u64_max() {
        let max_position = Position::from(u64::MAX);
        let mut frontier = super::Frontier::<String, 64>::from_parts(
            max_position,
            "tip".to_string(),
            vec![String::new(); 64],
        )
        .expect("valid frontier");
        let before = frontier.clone();
        let mut visits = 0;
        assert!(!frontier.append_batch_visiting(vec!["next".to_string()], |_, _| visits += 1));
        assert_eq!(visits, 0);
        assert_eq!(frontier, before);
    }

    std::thread_local! {
        /// The level and the number of pairs of each call to `PairsOnly::combine_pairs` that
        /// the current thread made after the last call to `take_combine_pairs_calls`.
        static COMBINE_PAIRS_CALLS: RefCell<Vec<(Level, usize)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Returns the level and the number of pairs of each call to `PairsOnly::combine_pairs`
    /// that the current thread made after the last call to this function.
    fn take_combine_pairs_calls() -> Vec<(Level, usize)> {
        COMBINE_PAIRS_CALLS.with(|calls| calls.take())
    }

    /// A node that records each call to its `combine_pairs`, and whose `combine` panics. A test
    /// that uses it fails if the code under test hashes one pair at a time.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PairsOnly(String);

    impl Hashable for PairsOnly {
        fn empty_leaf() -> Self {
            PairsOnly(String::empty_leaf())
        }

        fn combine(_: Level, _: &Self, _: &Self) -> Self {
            unreachable!("hashed outside combine_pairs")
        }

        fn combine_pairs<'a, I>(level: Level, pairs: I) -> Vec<Self>
        where
            Self: 'a,
            I: IntoIterator<Item = (&'a Self, &'a Self)>,
        {
            let parents: Vec<Self> = pairs
                .into_iter()
                .map(|(left, right)| PairsOnly(String::combine(level, &left.0, &right.0)))
                .collect();
            COMBINE_PAIRS_CALLS.with(|calls| calls.borrow_mut().push((level, parents.len())));
            parents
        }

        fn empty_root(level: Level) -> Self {
            PairsOnly(String::empty_root(level))
        }
    }

    #[test]
    fn option_forwards_combine_pairs_for_pairs_of_present_nodes() {
        let node = |value: &str| Some(PairsOnly(value.to_string()));
        let (a, b, c, d) = (node("a"), node("b"), node("c"), node("d"));
        assert_eq!(
            Option::combine_pairs(Level::from(0), [(&a, &b), (&None, &c), (&c, &d)]),
            vec![node("ab"), None, node("cd")],
        );
    }

    #[test]
    fn append_batch_hashes_through_combine_pairs() {
        let mut frontier = super::Frontier::<PairsOnly, 5>::empty();
        let mut expected = super::Frontier::<String, 5>::empty();
        for batch in [13, 6, 1, 3] {
            let leaves: Vec<String> = (0..batch).map(|i| format!("<{batch}.{i}>")).collect();
            assert!(frontier.append_batch(leaves.iter().cloned().map(PairsOnly)));
            for leaf in leaves {
                assert!(expected.append(leaf));
            }

            let actual = frontier.value().expect("frontier is non-empty");
            let expected = expected.value().expect("frontier is non-empty");
            assert_eq!(actual.position(), expected.position(), "after {batch}");
            assert_eq!(actual.leaf().0, *expected.leaf(), "after {batch}");
            let actual_ommers: Vec<_> = actual.ommers().iter().map(|node| &node.0).collect();
            let expected_ommers: Vec<_> = expected.ommers().iter().collect();
            assert_eq!(actual_ommers, expected_ommers, "after {batch}");
        }
    }

    fn arb_pairs_only() -> impl Strategy<Value = PairsOnly> + Clone {
        "[a-z]".prop_map(PairsOnly)
    }

    proptest! {
        #[test]
        fn append_batch_gives_the_same_frontier_as_append_per_leaf(
            frontier in arb_nonempty_frontier(arb_test_node()),
            leaves in proptest::collection::vec(arb_test_node(), 0..200),
        ) {
            prop_assume!(tip_after(&frontier, leaves.len()).is_some());
            let mut expected = frontier.clone();
            for leaf in leaves.iter() {
                expected.append(*leaf);
            }

            let mut batched = frontier.clone();
            prop_assert!(batched.append_batch(leaves.clone()));
            prop_assert_eq!(&batched, &expected);

            let mut visiting = frontier.clone();
            let appended = visiting.append_batch_visiting(leaves, |_, _| {});
            prop_assert!(appended);
            prop_assert_eq!(&visiting, &expected);
        }

        #[test]
        fn append_batch_visiting_visits_each_node_that_ends_in_the_batch_once(
            frontier in arb_nonempty_frontier(arb_test_node()),
            leaves in proptest::collection::vec(arb_test_node(), 0..200),
        ) {
            let new_tip = tip_after(&frontier, leaves.len());
            prop_assume!(new_tip.is_some());
            let new_tip = new_tip.expect("the batch fits");

            let mut calls: VisitCalls<TestNode> = vec![];
            let mut visiting = frontier.clone();
            let appended = visiting.append_batch_visiting(leaves, |first, nodes| {
                calls.push((first, nodes.to_vec()))
            });
            prop_assert!(appended);

            prop_assert!(has_one_call_per_level(&calls));
            prop_assert_eq!(
                visited_addresses(&calls),
                nodes_completed_after(Some(frontier.position()), new_tip)
            );
        }

        #[test]
        fn append_batch_visiting_visits_the_same_nodes_when_a_batch_is_split_in_two(
            frontier in arb_nonempty_frontier(arb_test_node()),
            first_leaves in proptest::collection::vec(arb_test_node(), 0..100),
            second_leaves in proptest::collection::vec(arb_test_node(), 0..100),
        ) {
            prop_assume!(
                tip_after(&frontier, first_leaves.len() + second_leaves.len()).is_some()
            );
            let append_and_record =
                |frontier: &mut NonEmptyFrontier<TestNode>, leaves: Vec<TestNode>| {
                    let mut calls: VisitCalls<TestNode> = vec![];
                    assert!(frontier.append_batch_visiting(leaves, |first, nodes| {
                        calls.push((first, nodes.to_vec()))
                    }));
                    visited_nodes(&calls)
                };

            let mut joined = frontier.clone();
            let all_leaves = first_leaves.iter().chain(&second_leaves).copied().collect();
            let mut joined_visits = append_and_record(&mut joined, all_leaves);

            let mut split = frontier.clone();
            let mut split_visits = append_and_record(&mut split, first_leaves);
            split_visits.extend(append_and_record(&mut split, second_leaves));

            joined_visits.sort();
            split_visits.sort();
            prop_assert_eq!(joined_visits, split_visits);
            prop_assert_eq!(joined, split);
        }

        #[test]
        fn append_batch_hashes_each_new_node_once_with_one_call_per_level(
            frontier in arb_nonempty_frontier(arb_pairs_only()),
            leaves in proptest::collection::vec(arb_pairs_only(), 0..200),
        ) {
            let new_tip = tip_after(&frontier, leaves.len());
            prop_assume!(new_tip.is_some());
            let new_tip = u64::from(new_tip.expect("the batch fits"));
            let prior_tip = u64::from(frontier.position());

            take_combine_pairs_calls();
            let mut batched = frontier.clone();
            prop_assert!(batched.append_batch(leaves));
            let calls = take_combine_pairs_calls();

            // The frontier holds the complete subtrees of the leaves before its last leaf.
            let pair_count: usize = calls.iter().map(|(_, pair_count)| pair_count).sum();
            prop_assert_eq!(
                u64::try_from(pair_count).expect("a pair count fits in u64"),
                complete_parent_count(new_tip) - complete_parent_count(prior_tip)
            );
            prop_assert!(calls.iter().all(|(_, pair_count)| *pair_count > 0));
            prop_assert!(calls.windows(2).all(|calls| calls[0].0 < calls[1].0));
        }

        #[test]
        fn append_batch_visiting_hashes_each_new_node_once_with_one_call_per_level(
            frontier in arb_nonempty_frontier(arb_pairs_only()),
            leaves in proptest::collection::vec(arb_pairs_only(), 1..200),
        ) {
            let new_tip = tip_after(&frontier, leaves.len());
            prop_assume!(new_tip.is_some());
            let new_tip = u64::from(new_tip.expect("the batch fits"));
            let prior_tip = u64::from(frontier.position());

            take_combine_pairs_calls();
            let mut visiting = frontier.clone();
            let appended = visiting.append_batch_visiting(leaves, |_, _| {});
            prop_assert!(appended);
            let calls = take_combine_pairs_calls();

            // The visitor also receives each node that has the new last leaf as its last leaf.
            // There is one such node above level 0 for each trailing one bit of the position.
            let pair_count: usize = calls.iter().map(|(_, pair_count)| pair_count).sum();
            prop_assert_eq!(
                u64::try_from(pair_count).expect("a pair count fits in u64"),
                complete_parent_count(new_tip) - complete_parent_count(prior_tip)
                    + u64::from(new_tip.trailing_ones())
            );
            prop_assert!(calls.iter().all(|(_, pair_count)| *pair_count > 0));
            prop_assert!(calls.windows(2).all(|calls| calls[0].0 < calls[1].0));
        }

        #[test]
        fn frontier_append_batch_appends_every_leaf_or_none(
            start in 0u64..=64,
            leaf_count in 0u64..=80,
        ) {
            const DEPTH: u8 = 6;
            let leaf = |position: u64| format!("<{position}>");
            let mut expected = super::Frontier::<String, DEPTH>::empty();
            for position in 0..start {
                prop_assert!(expected.append(leaf(position)));
            }

            let mut calls: VisitCalls<String> = vec![];
            let mut frontier = expected.clone();
            let appended = frontier
                .append_batch_visiting((start..start + leaf_count).map(leaf), |first, nodes| {
                    calls.push((first, nodes.to_vec()))
                });

            if start + leaf_count <= 1 << DEPTH {
                prop_assert!(appended);
                for position in start..start + leaf_count {
                    prop_assert!(expected.append(leaf(position)));
                }
                prop_assert_eq!(&frontier, &expected);

                // A `String` node is the concatenation of the leaves below it.
                prop_assert!(has_one_call_per_level(&calls));
                let prior_tip = start.checked_sub(1).map(Position::from);
                let completed = (start + leaf_count)
                    .checked_sub(1)
                    .map_or(vec![], |tip| nodes_completed_after(prior_tip, Position::from(tip)));
                let expected_visits: Vec<(Address, String)> = completed
                    .into_iter()
                    .filter(|_| leaf_count > 0)
                    .map(|address| {
                        let positions = u64::from(address.position_range_start())
                            ..u64::from(address.position_range_end());
                        (address, positions.map(leaf).collect())
                    })
                    .collect();
                prop_assert_eq!(visited_nodes(&calls), expected_visits);
            } else {
                prop_assert!(!appended);
                prop_assert_eq!(&frontier, &expected);
                prop_assert!(calls.is_empty());
            }
        }
    }

    #[test]
    fn append_batch_visiting_leaves_the_frontier_unchanged_when_the_visitor_panics() {
        let leaf = |position: u64| format!("<{position}>");
        let mut frontier = NonEmptyFrontier::new(leaf(0));
        assert!(frontier.append_batch((1..12).map(leaf)));
        let before = frontier.clone();

        let result = catch_unwind(AssertUnwindSafe(|| {
            frontier.append_batch_visiting((12..17).map(leaf), |first, _| {
                if first.level() > Level::from(0) {
                    panic!("the visitor failed")
                }
            })
        }));

        assert!(result.is_err());
        assert_eq!(frontier, before);
    }

    /// A node with a `combine_pairs` that does not return the parent of the last pair.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DropsLastParent(String);

    impl Hashable for DropsLastParent {
        fn empty_leaf() -> Self {
            DropsLastParent(String::empty_leaf())
        }

        fn combine(level: Level, a: &Self, b: &Self) -> Self {
            DropsLastParent(String::combine(level, &a.0, &b.0))
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
    fn append_batch_panics_and_leaves_the_frontier_unchanged_when_combine_pairs_drops_a_node() {
        let node = |value: &str| DropsLastParent(value.to_string());
        let mut frontier = NonEmptyFrontier::new(node("a"));
        let before = frontier.clone();

        let result = catch_unwind(AssertUnwindSafe(|| {
            frontier.append_batch(["b", "c", "d", "e"].map(node))
        }));

        assert!(result.is_err());
        assert_eq!(frontier, before);
    }

    #[test]
    fn append_batch_visiting_visits_the_root_of_a_full_tree_of_depth_64() {
        let prior_tip = Position::from(u64::MAX - 1);
        let ommers: Vec<String> = (1..64).map(|level| format!("<{level}>")).collect();
        let mut frontier =
            super::Frontier::<String, 64>::from_parts(prior_tip, "y".to_string(), ommers)
                .expect("the number of ommers matches the position");

        let mut calls: VisitCalls<String> = vec![];
        assert!(
            frontier.append_batch_visiting(["z".to_string()], |first, nodes| calls
                .push((first, nodes.to_vec())))
        );

        assert!(has_one_call_per_level(&calls));
        assert_eq!(
            visited_addresses(&calls),
            nodes_completed_after(Some(prior_tip), Position::from(u64::MAX))
        );
        let root_address = Address::from_parts(Level::from(64), 0);
        let visited_root = visited_nodes(&calls)
            .into_iter()
            .find(|(address, _)| *address == root_address)
            .map(|(_, root)| root);
        // A `String` node is the concatenation of its children. The ommers are the left
        // children on the path from the root to the two leaves "y" and "z".
        let expected_root: String = (1..64)
            .rev()
            .map(|level| format!("<{level}>"))
            .chain(["yz".to_string()])
            .collect();
        assert_eq!(visited_root, Some(expected_root));
    }

    #[test]
    fn nonempty_frontier_append_batch_appends_nothing_past_position_u64_max() {
        for (position, leaf_count) in [(u64::MAX, 1u8), (u64::MAX - 1, 2), (u64::MAX - 5, 7)] {
            let position = Position::from(position);
            let ommers = vec![String::new(); position.past_ommer_count().into()];
            let mut frontier = NonEmptyFrontier::from_parts(position, "tip".to_string(), ommers)
                .expect("the number of ommers matches the position");
            let before = frontier.clone();
            let leaves = || (0..leaf_count).map(|leaf| leaf.to_string());

            assert!(!frontier.append_batch(leaves()));
            assert_eq!(frontier, before);

            let mut visits = 0;
            assert!(!frontier.append_batch_visiting(leaves(), |_, _| visits += 1));
            assert_eq!(visits, 0);
            assert_eq!(frontier, before);
        }
    }

    #[test]
    fn nonempty_frontier_root() {
        let mut frontier = NonEmptyFrontier::new("a".to_string());
        assert_eq!(frontier.root(None), "a");

        frontier.append("b".to_string());
        assert_eq!(frontier.root(None), "ab");

        frontier.append("c".to_string());
        assert_eq!(frontier.root(None), "abc_");
    }

    /// A node whose `empty_leaf` panics. A test that uses it fails if the code under test
    /// computes the root of an empty subtree.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NoEmptyLeaf(String);

    impl Hashable for NoEmptyLeaf {
        fn empty_leaf() -> Self {
            panic!("the root of an empty subtree was computed")
        }

        fn combine(level: Level, a: &Self, b: &Self) -> Self {
            NoEmptyLeaf(String::combine(level, &a.0, &b.0))
        }
    }

    #[test]
    fn frontier_root_of_a_full_tree_computes_no_empty_root() {
        let mut frontier = super::Frontier::<NoEmptyLeaf, 2>::empty();
        for leaf in ["a", "b", "c", "d"] {
            assert!(frontier.append(NoEmptyLeaf(leaf.to_string())));
        }
        assert_eq!(frontier.root(), NoEmptyLeaf("abcd".to_string()));
    }

    #[test]
    fn frontier_from_parts() {
        assert!(super::Frontier::<(), 1>::from_parts(0.into(), (), vec![]).is_ok());
        assert!(super::Frontier::<(), 1>::from_parts(1.into(), (), vec![()]).is_ok());
        assert!(super::Frontier::<(), 1>::from_parts(0.into(), (), vec![()]).is_err());
    }

    #[test]
    fn frontier_root() {
        let mut frontier: super::Frontier<String, 4> = super::Frontier::empty();
        assert_eq!(frontier.root().len(), 16);
        assert_eq!(frontier.root(), "________________");

        frontier.append("a".to_string());
        assert_eq!(frontier.root(), "a_______________");

        frontier.append("b".to_string());
        assert_eq!(frontier.root(), "ab______________");

        frontier.append("c".to_string());
        assert_eq!(frontier.root(), "abc_____________");
    }

    #[test]
    fn nonempty_frontier_witness() {
        let mut frontier = NonEmptyFrontier::<String>::new("a".to_string());
        for c in 'b'..='g' {
            frontier.append(c.to_string());
        }
        let bridge_value_at = |addr: Address| match <u8>::from(addr.level()) {
            0 => Some("h".to_string()),
            3 => Some("xxxxxxxx".to_string()),
            _ => None,
        };

        assert_eq!(
            Ok(["h", "ef", "abcd", "xxxxxxxx"]
                .map(|v| v.to_string())
                .to_vec()),
            frontier.witness(4, bridge_value_at)
        );
    }

    #[test]
    fn frontier_witness() {
        let mut frontier = Frontier::<String, 4>::empty();
        for c in 'a'..='g' {
            frontier.append(c.to_string());
        }

        assert_eq!(
            frontier
                .witness(|addr| Some(String::empty_root(addr.level())))
                .map(|maybe_p| maybe_p.map(|p| p.path_elems().to_vec())),
            Ok(Some(
                ["_", "ef", "abcd", "________"]
                    .map(|v| v.to_string())
                    .to_vec()
            )),
        );
    }

    #[test]
    #[cfg(feature = "legacy-api")]
    fn test_commitment_tree_complete() {
        let mut t: CommitmentTree<TestNode, 6> = CommitmentTree::empty();
        for n in 1u64..=32 {
            t.append(TestNode(n)).unwrap();
            // every tree of a power-of-two height is complete
            let is_complete = n.count_ones() == 1;
            let level = usize::BITS - 1 - n.leading_zeros(); //log2
            assert_eq!(
                is_complete,
                t.is_complete(level.try_into().unwrap()),
                "Tree {:?} {} complete at height {}",
                t,
                if is_complete {
                    "should be"
                } else {
                    "should not be"
                },
                n
            );
        }
    }

    #[test]
    #[cfg(feature = "legacy-api")]
    fn test_commitment_tree_roundtrip() {
        let ct = CommitmentTree {
            left: Some("a".to_string()),
            right: Some("b".to_string()),
            parents: vec![
                Some("c".to_string()),
                Some("d".to_string()),
                Some("e".to_string()),
                Some("f".to_string()),
                None,
                None,
                None,
            ],
        };

        let frontier: Frontier<String, 8> = ct.to_frontier();
        let ct0 = CommitmentTree::from_frontier(&frontier);
        assert_eq!(ct, ct0);
        let frontier0: Frontier<String, 8> = ct0.to_frontier();
        assert_eq!(frontier, frontier0);
    }

    #[test]
    fn test_random_frontier_structure() {
        let tree_size = (2u64.pow(4)) * 3 + 5;

        let mut f: Frontier<TestNode, 8> = Frontier::empty();
        for i in 0..tree_size {
            f.append(TestNode(i));
        }
        let f = f.frontier.expect("Frontier should not be empty.");

        let mut rng = ChaChaRng::seed_from_u64(0);
        let (prior_roots, f0) = Frontier::<TestNode, 8>::random_with_prior_subtree_roots(
            &mut rng,
            tree_size,
            NonZeroU8::new(4).unwrap(),
        );
        let f0 = f0.frontier.expect("Frontier should not be empty.");

        assert_eq!(prior_roots.len(), 3);
        assert_eq!(f.position, f0.position);
        assert_eq!(f.ommers.len(), f0.ommers.len());

        let expected_largest_ommer =
            TestNode::combine(Level::from(4), &prior_roots[0], &prior_roots[1]);
        assert_eq!(f0.ommers[f0.ommers.len() - 1], expected_largest_ommer);
        assert_eq!(f0.ommers[f0.ommers.len() - 2], prior_roots[2]);
    }

    #[cfg(feature = "legacy-api")]
    proptest! {
        #[test]
        fn prop_commitment_tree_roundtrip(ct in arb_commitment_tree(32, arb_test_node())) {
            let frontier: Frontier<TestNode, 8> = ct.to_frontier();
            let ct0 = CommitmentTree::from_frontier(&frontier);
            assert_eq!(ct, ct0);
            let frontier0: Frontier<TestNode, 8> = ct0.to_frontier();
            assert_eq!(frontier, frontier0);
        }

        #[test]
        fn prop_commitment_tree_roundtrip_str(ct in arb_commitment_tree::<_, _, 8>(32, any::<char>().prop_map(|c| c.to_string()))) {
            let frontier: Frontier<String, 8> = ct.to_frontier();
            let ct0 = CommitmentTree::from_frontier(&frontier);
            assert_eq!(ct, ct0);
            let frontier0: Frontier<String, 8> = ct0.to_frontier();
            assert_eq!(frontier, frontier0);
        }
    }
}
