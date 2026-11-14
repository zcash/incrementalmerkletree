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

    /// Generates a random frontier of a Merkle tree having the specified nonzero size, along
    /// with the roots of all complete subtrees of depth `subtree_depth` in that tree.
    ///
    /// The returned roots are in order of subtree index, and there are
    /// `tree_size >> subtree_depth` of them. The frontier agrees with these roots: its ommers
    /// at levels at or above `subtree_depth` are computed from them. If the frontier's leaf
    /// completes the last subtree, the root of that subtree is computed from the frontier's
    /// leaf and its ommers below `subtree_depth`.
    pub fn random_with_prior_subtree_roots<R>(
        rng: &mut R,
        tree_size: NonZeroU64,
        subtree_depth: NonZeroU8,
    ) -> (Vec<H>, Self)
    where
        R: Rng,
    {
        /// Returns the root of the complete subtree whose leaves at `level` are `nodes`.
        ///
        /// The length of `nodes` must be a power of two.
        fn complete_subtree_root<H: Hashable + Clone>(level: Level, nodes: &[H]) -> H {
            match nodes {
                [node] => node.clone(),
                _ => {
                    let (left, right) = nodes.split_at(nodes.len() / 2);
                    let child_level = level + (nodes.len().trailing_zeros() - 1) as u8;
                    H::combine(
                        child_level,
                        &complete_subtree_root(level, left),
                        &complete_subtree_root(level, right),
                    )
                }
            }
        }

        let depth = u32::from(u8::from(subtree_depth));
        let subtree_level = Level::from(u8::from(subtree_depth));
        let below_subtree_mask = 1u64.checked_shl(depth).map_or(u64::MAX, |bit| bit - 1);

        let mut result = Self::random_of_size(rng, tree_size);
        let position = u64::from(result.position);

        // The subtrees that end before the frontier's leaf. Their roots make the ommers at
        // and above `subtree_level`.
        let prior_subtree_count = position.checked_shr(depth).unwrap_or(0);
        let mut prior_roots: Vec<H> = core::iter::repeat_with(|| rng.random())
            .take(prior_subtree_count as usize)
            .collect();

        let mut upper_ommers = Vec::new();
        let mut remaining_roots = &prior_roots[..];
        for bit in (0..u64::BITS).rev() {
            if (prior_subtree_count >> bit) & 1 == 1 {
                let (block, rest) = remaining_roots.split_at(1usize << bit);
                upper_ommers.push(complete_subtree_root(subtree_level, block));
                remaining_roots = rest;
            }
        }

        let lower_ommer_count = (position & below_subtree_mask).count_ones() as usize;
        result.ommers.truncate(lower_ommer_count);
        result.ommers.extend(upper_ommers.into_iter().rev());

        // If the frontier's leaf completes a subtree, that subtree's root depends on the leaf
        // and on all ommers below `subtree_level`.
        if u64::from(tree_size) & below_subtree_mask == 0 {
            let completed_root = result.ommers[..lower_ommer_count]
                .iter()
                .zip(0u8..)
                .fold(result.leaf.clone(), |node, (ommer, level)| {
                    H::combine(level.into(), ommer, &node)
                });
            prior_roots.push(completed_root);
        }

        (prior_roots, result)
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

    /// Obtains the current root of this Merkle frontier by hashing
    /// against empty nodes up to the maximum height of the pruned
    /// tree that the frontier represents.
    pub fn root(&self) -> H {
        self.frontier
            .as_ref()
            .map_or(H::empty_root(DEPTH.into()), |frontier| {
                frontier.root(Some(DEPTH.into()))
            })
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
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    use super::{testing::TestNode, *};

    #[cfg(feature = "legacy-api")]
    use {
        super::testing::{arb_commitment_tree, arb_test_node},
        proptest::prelude::*,
    };

    #[test]
    fn nonempty_frontier_root() {
        let mut frontier = NonEmptyFrontier::new("a".to_string());
        assert_eq!(frontier.root(None), "a");

        frontier.append("b".to_string());
        assert_eq!(frontier.root(None), "ab");

        frontier.append("c".to_string());
        assert_eq!(frontier.root(None), "abc_");
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

    #[test]
    fn test_random_frontier_agrees_with_prior_subtree_roots() {
        const DEPTH: u8 = 8;

        /// Returns the root of the depth-`DEPTH` tree whose nodes at `level` are `nodes`,
        /// followed by empty nodes.
        fn root_from_level(level: Level, mut nodes: Vec<TestNode>) -> TestNode {
            if level == Level::from(DEPTH) {
                assert_eq!(nodes.len(), 1);
                nodes[0]
            } else {
                if nodes.len() % 2 == 1 {
                    nodes.push(TestNode::empty_root(level));
                }
                let parents = nodes
                    .chunks(2)
                    .map(|pair| TestNode::combine(level, &pair[0], &pair[1]))
                    .collect();
                root_from_level(level + 1, parents)
            }
        }

        let mut rng = ChaChaRng::seed_from_u64(0);
        for subtree_depth in 1..=DEPTH {
            let subtree_level = Level::from(subtree_depth);
            let below_subtree_mask = (1u64 << subtree_depth) - 1;
            for tree_size in 1..=(1u64 << DEPTH) {
                let (prior_roots, frontier) =
                    Frontier::<TestNode, DEPTH>::random_with_prior_subtree_roots(
                        &mut rng,
                        tree_size,
                        NonZeroU8::new(subtree_depth).unwrap(),
                    );
                assert_eq!(prior_roots.len() as u64, tree_size >> subtree_depth);

                // The nodes at `subtree_level`: the complete subtree roots, then the root of
                // the incomplete last subtree, if there is one.
                let mut subtree_nodes = prior_roots;
                let tip = frontier.value().unwrap();
                if tree_size & below_subtree_mask != 0 {
                    let position = u64::from(tip.position);
                    let lower_ommer_count = (position & below_subtree_mask).count_ones() as usize;
                    let partial_subtree = NonEmptyFrontier::from_parts(
                        Position::from(position & below_subtree_mask),
                        tip.leaf,
                        tip.ommers[..lower_ommer_count].to_vec(),
                    )
                    .unwrap();
                    subtree_nodes.push(partial_subtree.root(Some(subtree_level)));
                }

                assert_eq!(
                    frontier.root(),
                    root_from_level(subtree_level, subtree_nodes),
                    "tree_size {tree_size}, subtree_depth {subtree_depth}"
                );
            }
        }
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
