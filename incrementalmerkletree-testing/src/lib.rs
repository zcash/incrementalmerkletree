//! Traits and types used to permit comparison testing between tree implementations.

use core::fmt::Debug;
use core::marker::PhantomData;
use proptest::prelude::*;
use std::collections::BTreeSet;

use incrementalmerkletree::{Hashable, Level, Marking, Position, Retention};

pub mod complete_tree;

//
// Traits used to permit comparison testing between tree implementations.
//

/// A Merkle tree that supports incremental appends, marking of leaf nodes for construction of
/// witnesses, checkpoints and rollbacks.
pub trait Tree<H, C> {
    /// Returns the number of levels in the tree.
    fn depth(&self) -> u8;

    /// Appends a new value to the tree at the next available slot.
    ///
    /// Returns true if successful and false if the tree would exceed
    /// the maximum allowed number of levels in the tree.
    fn append(&mut self, value: H, retention: Retention<C>) -> bool;

    /// Returns the most recently appended leaf value.
    fn current_position(&self) -> Option<Position>;

    /// Returns the leaf at the specified position if the tree can produce a witness for it.
    fn get_marked_leaf(&self, position: Position) -> Option<H>;

    /// Return a set of all the positions for which we have marked.
    fn marked_positions(&self) -> BTreeSet<Position>;

    /// Obtains the root of the Merkle tree at the specified checkpoint depth by hashing against
    /// empty nodes up to the maximum height of the tree.
    ///
    /// Returns `None` if a checkpoint depth is provided but there are not enough checkpoints
    /// available to reach the requested checkpoint depth.
    ///
    /// ## Parameters
    /// - `checkpoint_depth`: A zero-based index into the checkpoints that have been added to the
    ///   tree, in reverse checkpoint identifier order. If `checkpoint_depth` is `None`, the root
    ///   is computed over all leaves that have been added to the tree.
    fn root(&self, checkpoint_depth: Option<usize>) -> Option<H>;

    /// Obtains a witness for the value at the specified leaf position, as of the tree state at the
    /// given checkpoint depth.
    ///
    /// Returns `None` if there is no witness information for the requested position or if no
    /// checkpoint is available at the specified depth.
    ///
    /// ## Parameters
    /// - `position`: The position of the leaf for which the witness is to be computed.
    /// - `checkpoint_depth`: A zero-based index over the checkpoints that have been added to the
    ///   tree, in reverse checkpoint identifier order.
    fn witness(&self, position: Position, checkpoint_depth: usize) -> Option<Vec<H>>;

    /// Marks the value at the specified position as a value we're no longer interested in
    /// maintaining a mark for.
    ///
    /// Returns true if successful and false if we were already not maintaining a mark at this
    /// position.
    ///
    /// ## Parameters
    /// - `position`: The position of the marked leaf.
    fn remove_mark(&mut self, position: Position) -> bool;

    /// Creates a new checkpoint for the current tree state, with the given checkpoint identifier.
    ///
    /// It is valid to have multiple checkpoints for the same tree state, and each `rewind` call
    /// will remove a single checkpoint. Returns `false` if the checkpoint identifier provided is
    /// less than or equal to the maximum checkpoint identifier among previously added checkpoints.
    ///
    /// ## Parameters
    /// - `id`: The identifier for the checkpoint being added to the tree.
    fn checkpoint(&mut self, id: C) -> bool;

    /// Returns the number of checkpoints in the tree.
    fn checkpoint_count(&self) -> usize;

    /// Rewinds the tree state to the checkpoint at the specified checkpoint depth.
    ///
    /// Returns `true` if the request was satisfied, or `false` if insufficient checkpoints were
    /// available to satisfy the request.
    ///
    /// ## Parameters
    /// - `checkpoint_depth`: A zero-based index over the checkpoints that have been added to the
    ///   tree, in reverse checkpoint identifier order. A checkpoint depth value of `0` removes all
    ///   data added to the tree after the most recently-added checkpoint. The checkpoint at the
    ///   specified depth is retained, but all data and metadata related to operations on the tree
    ///   that occurred after the checkpoint was created is discarded.
    fn rewind(&mut self, checkpoint_depth: usize) -> bool;
}

//
// Types and utilities for shared example tests.
//

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SipHashable(pub u64);

impl Hashable for SipHashable {
    fn empty_leaf() -> Self {
        SipHashable(0)
    }

    fn combine(_level: Level, a: &Self, b: &Self) -> Self {
        #![allow(deprecated)]
        use std::hash::{Hasher, SipHasher};

        let mut hasher = SipHasher::new();
        hasher.write_u64(a.0);
        hasher.write_u64(b.0);
        SipHashable(hasher.finish())
    }
}

//
// Operations
//

#[derive(Clone, Debug)]
pub enum Operation<A, C> {
    Append(A, Retention<C>),
    CurrentPosition,
    MarkedLeaf(Position),
    MarkedPositions,
    Unmark(Position),
    Checkpoint(C),
    Rewind(usize),
    Witness(Position, usize),
    GarbageCollect,
}

use Operation::*;

pub fn append_str<C>(x: &str, retention: Retention<C>) -> Operation<String, C> {
    Operation::Append(x.to_string(), retention)
}

pub fn unmark<H, C>(pos: u64) -> Operation<H, C> {
    Operation::Unmark(Position::from(pos))
}

pub fn witness<H, C>(pos: u64, depth: usize) -> Operation<H, C> {
    Operation::Witness(Position::from(pos), depth)
}

impl<H: Hashable + Clone, C: Clone> Operation<H, C> {
    pub fn apply<T: Tree<H, C>>(&self, tree: &mut T) -> Option<(Position, Vec<H>)> {
        match self {
            Append(a, r) => {
                assert!(tree.append(a.clone(), r.clone()), "append failed");
                None
            }
            CurrentPosition => None,
            MarkedLeaf(_) => None,
            MarkedPositions => None,
            Unmark(p) => {
                assert!(tree.remove_mark(*p), "remove mark failed");
                None
            }
            Checkpoint(id) => {
                tree.checkpoint(id.clone());
                None
            }
            Rewind(depth) => {
                assert_eq!(tree.checkpoint_count() > *depth, tree.rewind(*depth));
                None
            }
            Witness(p, d) => tree.witness(*p, *d).map(|xs| (*p, xs)),
            GarbageCollect => None,
        }
    }

    pub fn apply_all<T: Tree<H, C>>(
        ops: &[Operation<H, C>],
        tree: &mut T,
    ) -> Option<(Position, Vec<H>)> {
        let mut result = None;
        for op in ops {
            result = op.apply(tree);
        }
        result
    }

    pub fn map_checkpoint_id<D, F: Fn(&C) -> D>(&self, f: F) -> Operation<H, D> {
        match self {
            Append(a, r) => Append(a.clone(), r.map(f)),
            CurrentPosition => CurrentPosition,
            MarkedLeaf(l) => MarkedLeaf(*l),
            MarkedPositions => MarkedPositions,
            Unmark(p) => Unmark(*p),
            Checkpoint(id) => Checkpoint(f(id)),
            Rewind(depth) => Rewind(*depth),
            Witness(p, d) => Witness(*p, *d),
            GarbageCollect => GarbageCollect,
        }
    }
}

/// Returns a strategy for creating a uniformly-distributed [`Marking`]
/// value.
pub fn arb_marking() -> impl Strategy<Value = Marking> {
    prop_oneof![
        Just(Marking::Marked),
        Just(Marking::Reference),
        Just(Marking::None)
    ]
}

/// Returns a strategy for creating a uniformly-distributed [`Retention`]
/// value.
pub fn arb_retention() -> impl Strategy<Value = Retention<()>> {
    prop_oneof![
        Just(Retention::Ephemeral),
        arb_marking().prop_map(|marking| Retention::Checkpoint { id: (), marking }),
        Just(Retention::Marked),
    ]
}

pub fn arb_operation<G: Strategy + Clone>(
    item_gen: G,
    pos_gen: impl Strategy<Value = Position> + Clone,
) -> impl Strategy<Value = Operation<G::Value, ()>>
where
    G::Value: Clone + 'static,
{
    prop_oneof![
        (item_gen, arb_retention()).prop_map(|(i, r)| Operation::Append(i, r)),
        prop_oneof![
            Just(Operation::CurrentPosition),
            Just(Operation::MarkedPositions),
        ],
        Just(Operation::GarbageCollect),
        pos_gen.clone().prop_map(Operation::MarkedLeaf),
        pos_gen.clone().prop_map(Operation::Unmark),
        Just(Operation::Checkpoint(())),
        (0usize..10).prop_map(Operation::Rewind),
        pos_gen.prop_flat_map(|i| (0usize..10).prop_map(move |depth| Operation::Witness(i, depth))),
    ]
}

pub fn apply_operation<H, C, T: Tree<H, C>>(tree: &mut T, op: Operation<H, C>) {
    match op {
        Append(value, r) => {
            tree.append(value, r);
        }
        Unmark(position) => {
            tree.remove_mark(position);
        }
        Checkpoint(id) => {
            tree.checkpoint(id);
        }
        Rewind(depth) => {
            tree.rewind(depth);
        }
        CurrentPosition => {}
        Witness(_, _) => {}
        MarkedLeaf(_) => {}
        MarkedPositions => {}
        GarbageCollect => {}
    }
}

pub fn check_operations<H: Hashable + Ord + Clone + Debug, C: Clone, T: Tree<H, C>>(
    mut tree: T,
    ops: &[Operation<H, C>],
) -> Result<(), TestCaseError> {
    let mut tree_size = 0;
    let mut tree_values: Vec<H> = vec![];
    // the number of leaves in the tree at the time that a checkpoint is made
    let mut tree_checkpoints: Vec<usize> = vec![];

    for op in ops {
        prop_assert_eq!(tree_size, tree_values.len());
        match op {
            Append(value, r) => {
                if tree.append(value.clone(), r.clone()) {
                    prop_assert!(tree_size < (1 << tree.depth()));
                    tree_size += 1;
                    tree_values.push(value.clone());
                    if r.is_checkpoint() {
                        tree_checkpoints.push(tree_size);
                    }
                } else {
                    prop_assert_eq!(
                        tree_size,
                        tree.current_position()
                            .map_or(0, |p| usize::try_from(p).unwrap() + 1)
                    );
                }
            }
            CurrentPosition => {
                if let Some(pos) = tree.current_position() {
                    prop_assert!(tree_size > 0);
                    prop_assert_eq!(tree_size - 1, pos.try_into().unwrap());
                }
            }
            MarkedLeaf(position) => {
                if tree.get_marked_leaf(*position).is_some() {
                    prop_assert!(<usize>::try_from(*position).unwrap() < tree_size);
                }
            }
            Unmark(position) => {
                tree.remove_mark(*position);
            }
            MarkedPositions => {}
            Checkpoint(id) => {
                tree_checkpoints.push(tree_size);
                tree.checkpoint(id.clone());
            }
            Rewind(depth) => {
                if tree.rewind(*depth) {
                    let retained = tree_checkpoints.len() - depth;
                    if *depth > 0 {
                        // The last checkpoint will have been dropped, and the tree will have been
                        // truncated to a previous checkpoint.
                        tree_checkpoints.truncate(retained);
                    }

                    let checkpointed_tree_size = tree_checkpoints
                        .last()
                        .expect("at least one checkpoint must exist in order to be able to rewind");
                    tree_values.truncate(*checkpointed_tree_size);
                    tree_size = *checkpointed_tree_size;
                }
            }
            Witness(position, depth) => {
                if let Some(path) = tree.witness(*position, *depth) {
                    let value: H = tree_values[<usize>::try_from(*position).unwrap()].clone();
                    let tree_root = tree.root(Some(*depth)).expect(
                        "we must be able to compute a root anywhere we can compute a witness.",
                    );
                    let mut extended_tree_values = tree_values.clone();
                    // prune the tree back to the checkpointed size.
                    let checkpointed_tree_size =
                        tree_checkpoints[tree_checkpoints.len() - (depth + 1)];
                    extended_tree_values.truncate(checkpointed_tree_size);

                    // compute the root
                    let expected_root =
                        complete_tree::root::<H>(&extended_tree_values, tree.depth());
                    prop_assert_eq!(&tree_root, &expected_root);

                    prop_assert_eq!(
                        &compute_root_from_witness(value, *position, &path),
                        &expected_root
                    );
                }
            }
            GarbageCollect => {}
        }
    }

    Ok(())
}

pub fn compute_root_from_witness<H: Hashable>(value: H, position: Position, path: &[H]) -> H {
    let mut cur = value;
    let mut lvl = 0.into();
    for (i, v) in path
        .iter()
        .enumerate()
        .map(|(i, v)| (((<u64>::from(position) >> i) & 1) == 1, v))
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

//
// Types and utilities for cross-verification property tests
//

#[derive(Clone, Debug)]
pub struct CombinedTree<H, C, I: Tree<H, C>, E: Tree<H, C>> {
    inefficient: I,
    efficient: E,
    _phantom_h: PhantomData<H>,
    _phantom_c: PhantomData<C>,
}

impl<H: Hashable + Ord + Clone + Debug, C, I: Tree<H, C>, E: Tree<H, C>> CombinedTree<H, C, I, E> {
    pub fn new(inefficient: I, efficient: E) -> Self {
        assert_eq!(inefficient.depth(), efficient.depth());
        CombinedTree {
            inefficient,
            efficient,
            _phantom_h: PhantomData,
            _phantom_c: PhantomData,
        }
    }
}

impl<H: Hashable + Ord + Clone + Debug, C: Clone, I: Tree<H, C>, E: Tree<H, C>> Tree<H, C>
    for CombinedTree<H, C, I, E>
{
    fn depth(&self) -> u8 {
        self.inefficient.depth()
    }

    fn append(&mut self, value: H, retention: Retention<C>) -> bool {
        let a = self.inefficient.append(value.clone(), retention.clone());
        let b = self.efficient.append(value, retention);
        assert_eq!(a, b);
        a
    }

    fn root(&self, checkpoint_depth: Option<usize>) -> Option<H> {
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

    fn get_marked_leaf(&self, position: Position) -> Option<H> {
        let a = self.inefficient.get_marked_leaf(position);
        let b = self.efficient.get_marked_leaf(position);
        assert_eq!(a, b);
        a
    }

    fn marked_positions(&self) -> BTreeSet<Position> {
        let a = self.inefficient.marked_positions();
        let b = self.efficient.marked_positions();
        assert_eq!(a, b);
        a
    }

    fn witness(&self, position: Position, checkpoint_depth: usize) -> Option<Vec<H>> {
        let a = self.inefficient.witness(position, checkpoint_depth);
        let b = self.efficient.witness(position, checkpoint_depth);
        assert_eq!(a, b);
        a
    }

    fn remove_mark(&mut self, position: Position) -> bool {
        let a = self.inefficient.remove_mark(position);
        let b = self.efficient.remove_mark(position);
        assert_eq!(a, b);
        a
    }

    fn checkpoint(&mut self, id: C) -> bool {
        let a = self.inefficient.checkpoint(id.clone());
        let b = self.efficient.checkpoint(id);
        assert_eq!(a, b);
        a
    }

    fn checkpoint_count(&self) -> usize {
        let a = self.inefficient.checkpoint_count();
        let b = self.efficient.checkpoint_count();
        assert_eq!(a, b);
        a
    }

    fn rewind(&mut self, checkpoint_depth: usize) -> bool {
        let a = self.inefficient.rewind(checkpoint_depth);
        let b = self.efficient.rewind(checkpoint_depth);
        assert_eq!(a, b);
        a
    }
}

//
// Shared example tests
//

pub trait TestHashable: Hashable + Ord + Clone + Debug {
    fn from_u64(value: u64) -> Self;

    fn combine_all(depth: u8, values: &[u64]) -> Self {
        let values: Vec<Self> = values.iter().map(|v| Self::from_u64(*v)).collect();
        complete_tree::root(&values, depth)
    }
}

impl TestHashable for String {
    fn from_u64(value: u64) -> Self {
        ('a'..)
            .nth(
                value
                    .try_into()
                    .expect("we do not use test value indices larger than usize::MAX"),
            )
            .expect("we do not choose test value indices larger than the iterable character range")
            .to_string()
    }
}

pub trait TestCheckpoint: Ord + Clone + Debug {
    fn from_u64(value: u64) -> Self;
}

impl TestCheckpoint for usize {
    fn from_u64(value: u64) -> Self {
        value
            .try_into()
            .expect("we do not use test checkpoint identifiers greater than usize::MAX")
    }
}

trait TestTree<H: TestHashable, C: TestCheckpoint> {
    fn assert_root(&self, checkpoint_depth: Option<usize>, values: &[u64]);

    fn assert_append(&mut self, value: u64, retention: Retention<u64>);

    fn assert_checkpoint(&mut self, value: u64);
}

impl<H: TestHashable, C: TestCheckpoint, T: Tree<H, C>> TestTree<H, C> for T {
    fn assert_root(&self, checkpoint_depth: Option<usize>, values: &[u64]) {
        assert_eq!(
            self.root(checkpoint_depth).unwrap(),
            H::combine_all(self.depth(), values)
        );
    }

    fn assert_append(&mut self, value: u64, retention: Retention<u64>) {
        assert!(
            self.append(H::from_u64(value), retention.map(|id| C::from_u64(*id))),
            "append failed for value {}",
            value
        );
    }

    fn assert_checkpoint(&mut self, value: u64) {
        assert!(
            self.checkpoint(C::from_u64(value)),
            "checkpoint failed for value {}",
            value
        );
    }
}

/// This checks basic append and root computation functionality
pub fn check_root_hashes<H: TestHashable, C: TestCheckpoint, T: Tree<H, C>, F: Fn(usize) -> T>(
    new_tree: F,
) {
    use Retention::*;

    {
        let mut tree = new_tree(100);
        tree.assert_root(None, &[]);
        tree.assert_append(0, Ephemeral);
        tree.assert_root(None, &[0]);
        tree.assert_append(1, Ephemeral);
        tree.assert_root(None, &[0, 1]);
        tree.assert_append(2, Ephemeral);
        tree.assert_root(None, &[0, 1, 2]);
    }

    {
        let mut t = new_tree(100);
        t.assert_append(
            0,
            Retention::Checkpoint {
                id: 1,
                marking: Marking::Marked,
            },
        );
        for _ in 0..3 {
            t.assert_append(0, Ephemeral);
        }
        t.assert_root(None, &[0, 0, 0, 0]);
    }
}

/// This test expects a depth-4 tree and verifies that the tree reports itself as full after 2^4
/// appends.
pub fn check_append<H: TestHashable, C: TestCheckpoint, T: Tree<H, C>, F: Fn(usize) -> T>(
    new_tree: F,
) {
    use Retention::*;

    {
        let mut tree = new_tree(100);
        assert_eq!(tree.depth(), 4);

        // 16 appends should succeed
        for i in 0..16 {
            tree.assert_append(i, Ephemeral);
            assert_eq!(tree.current_position(), Some(Position::from(i)));
        }

        // 17th append should fail
        assert!(!tree.append(H::from_u64(16), Ephemeral));
    }

    {
        // The following checks a condition on state restoration in the case that an append fails.
        // We want to ensure that a failed append does not cause a loss of information.
        let ops = (0..17)
            .map(|i| Append(H::from_u64(i), Ephemeral))
            .collect::<Vec<_>>();
        let tree = new_tree(100);
        check_operations(tree, &ops).unwrap();
    }
}

pub fn check_witnesses<H: TestHashable, C: TestCheckpoint, T: Tree<H, C>, F: Fn(usize) -> T>(
    new_tree: F,
) {
    use Retention::*;

    {
        let mut tree = new_tree(100);
        tree.assert_append(0, Ephemeral);
        tree.assert_append(1, Marked);
        tree.checkpoint(C::from_u64(0));
        assert_eq!(tree.witness(Position::from(0), 0), None);
    }

    {
        let mut tree = new_tree(100);
        tree.assert_append(0, Marked);
        tree.checkpoint(C::from_u64(0));
        assert_eq!(
            tree.witness(Position::from(0), 0),
            Some(vec![
                H::empty_root(0.into()),
                H::empty_root(1.into()),
                H::empty_root(2.into()),
                H::empty_root(3.into())
            ])
        );

        tree.assert_append(1, Ephemeral);
        tree.checkpoint(C::from_u64(1));
        assert_eq!(
            tree.witness(0.into(), 0),
            Some(vec![
                H::from_u64(1),
                H::empty_root(1.into()),
                H::empty_root(2.into()),
                H::empty_root(3.into())
            ])
        );

        tree.assert_append(2, Marked);
        tree.checkpoint(C::from_u64(2));
        assert_eq!(
            tree.witness(Position::from(2), 0),
            Some(vec![
                H::empty_root(0.into()),
                H::combine_all(1, &[0, 1]),
                H::empty_root(2.into()),
                H::empty_root(3.into())
            ])
        );

        tree.assert_append(3, Ephemeral);
        tree.checkpoint(C::from_u64(3));
        assert_eq!(
            tree.witness(Position::from(2), 0),
            Some(vec![
                H::from_u64(3),
                H::combine_all(1, &[0, 1]),
                H::empty_root(2.into()),
                H::empty_root(3.into())
            ])
        );

        tree.assert_append(4, Ephemeral);
        tree.checkpoint(C::from_u64(4));
        assert_eq!(
            tree.witness(Position::from(2), 0),
            Some(vec![
                H::from_u64(3),
                H::combine_all(1, &[0, 1]),
                H::combine_all(2, &[4]),
                H::empty_root(3.into())
            ])
        );
    }

    {
        let mut tree = new_tree(100);
        tree.assert_append(0, Marked);
        for i in 1..6 {
            tree.assert_append(i, Ephemeral);
        }
        tree.assert_append(6, Marked);
        tree.assert_append(7, Ephemeral);
        tree.checkpoint(C::from_u64(0));

        assert_eq!(
            tree.witness(0.into(), 0),
            Some(vec![
                H::from_u64(1),
                H::combine_all(1, &[2, 3]),
                H::combine_all(2, &[4, 5, 6, 7]),
                H::empty_root(3.into())
            ])
        );
    }

    {
        let mut tree = new_tree(100);
        tree.assert_append(0, Marked);
        tree.assert_append(1, Ephemeral);
        tree.assert_append(2, Ephemeral);
        tree.assert_append(3, Marked);
        tree.assert_append(4, Marked);
        tree.assert_append(5, Marked);
        tree.assert_append(6, Ephemeral);
        tree.checkpoint(C::from_u64(0));

        assert_eq!(
            tree.witness(Position::from(5), 0),
            Some(vec![
                H::from_u64(4),
                H::combine_all(1, &[6]),
                H::combine_all(2, &[0, 1, 2, 3]),
                H::empty_root(3.into())
            ])
        );
    }

    {
        let mut tree = new_tree(100);
        for i in 0..10 {
            tree.assert_append(i, Ephemeral);
        }
        tree.assert_append(10, Marked);
        tree.assert_append(11, Ephemeral);
        tree.checkpoint(C::from_u64(0));

        assert_eq!(
            tree.witness(Position::from(10), 0),
            Some(vec![
                H::from_u64(11),
                H::combine_all(1, &[8, 9]),
                H::empty_root(2.into()),
                H::combine_all(3, &[0, 1, 2, 3, 4, 5, 6, 7])
            ])
        );
    }

    {
        let mut tree = new_tree(100);
        tree.assert_append(
            0,
            Checkpoint {
                id: 1,
                marking: Marking::Marked,
            },
        );
        assert!(tree.rewind(0));
        for i in 1..4 {
            tree.assert_append(i, Ephemeral);
        }
        tree.assert_append(4, Marked);
        for i in 5..8 {
            tree.assert_append(i, Ephemeral);
        }
        tree.checkpoint(C::from_u64(2));
        assert_eq!(
            tree.witness(0.into(), 0),
            Some(vec![
                H::from_u64(1),
                H::combine_all(1, &[2, 3]),
                H::combine_all(2, &[4, 5, 6, 7]),
                H::empty_root(3.into()),
            ])
        );
    }

    {
        let mut tree = new_tree(100);
        tree.assert_append(0, Ephemeral);
        tree.assert_append(1, Ephemeral);
        tree.assert_append(2, Marked);
        tree.assert_append(3, Ephemeral);
        tree.assert_append(4, Ephemeral);
        tree.assert_append(5, Ephemeral);
        tree.assert_append(
            6,
            Checkpoint {
                id: 1,
                marking: Marking::Marked,
            },
        );
        tree.assert_append(7, Ephemeral);
        assert!(tree.rewind(0));
        assert_eq!(
            tree.witness(Position::from(2), 0),
            Some(vec![
                H::from_u64(3),
                H::combine_all(1, &[0, 1]),
                H::combine_all(2, &[4, 5, 6]),
                H::empty_root(3.into())
            ])
        );
    }

    {
        let mut tree = new_tree(100);
        for i in 0..12 {
            tree.assert_append(i, Ephemeral);
        }
        tree.assert_append(12, Marked);
        tree.assert_append(13, Marked);
        tree.assert_append(14, Ephemeral);
        tree.assert_append(15, Ephemeral);
        tree.checkpoint(C::from_u64(0));

        assert_eq!(
            tree.witness(Position::from(12), 0),
            Some(vec![
                H::from_u64(13),
                H::combine_all(1, &[14, 15]),
                H::combine_all(2, &[8, 9, 10, 11]),
                H::combine_all(3, &[0, 1, 2, 3, 4, 5, 6, 7]),
            ])
        );
    }

    {
        let ops = (0..=11)
            .map(|i| Append(H::from_u64(i), Marked))
            .chain(Some(Append(H::from_u64(12), Ephemeral)))
            .chain(Some(Append(
                H::from_u64(13),
                Checkpoint {
                    id: C::from_u64(0),
                    marking: Marking::None,
                },
            )))
            .chain(Some(Witness(11u64.into(), 0)))
            .collect::<Vec<_>>();

        let mut tree = new_tree(100);
        assert_eq!(
            Operation::apply_all(&ops, &mut tree),
            Some((
                Position::from(11),
                vec![
                    H::from_u64(10),
                    H::combine_all(1, &[8, 9]),
                    H::combine_all(2, &[12, 13]),
                    H::combine_all(3, &[0, 1, 2, 3, 4, 5, 6, 7]),
                ]
            ))
        );
    }

    {
        let ops = vec![
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(1), Ephemeral),
            Append(H::from_u64(2), Ephemeral),
            Append(
                H::from_u64(3),
                Checkpoint {
                    id: C::from_u64(1),
                    marking: Marking::Marked,
                },
            ),
            Append(H::from_u64(4), Marked),
            Operation::Checkpoint(C::from_u64(2)),
            Append(
                H::from_u64(5),
                Checkpoint {
                    id: C::from_u64(3),
                    marking: Marking::None,
                },
            ),
            Append(
                H::from_u64(6),
                Checkpoint {
                    id: C::from_u64(4),
                    marking: Marking::None,
                },
            ),
            Append(
                H::from_u64(7),
                Checkpoint {
                    id: C::from_u64(5),
                    marking: Marking::None,
                },
            ),
            Witness(3u64.into(), 4),
        ];
        let mut tree = new_tree(100);
        assert_eq!(
            Operation::apply_all(&ops, &mut tree),
            Some((
                Position::from(3),
                vec![
                    H::from_u64(2),
                    H::combine_all(1, &[0, 1]),
                    H::combine_all(2, &[]),
                    H::combine_all(3, &[]),
                ]
            ))
        );
    }

    {
        let ops = vec![
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(
                H::from_u64(0),
                Checkpoint {
                    id: C::from_u64(1),
                    marking: Marking::Marked,
                },
            ),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(
                H::from_u64(0),
                Checkpoint {
                    id: C::from_u64(2),
                    marking: Marking::None,
                },
            ),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Witness(Position::from(3), 0),
        ];
        let mut tree = new_tree(100);
        assert_eq!(
            Operation::apply_all(&ops, &mut tree),
            Some((
                Position::from(3),
                vec![
                    H::from_u64(0),
                    H::combine_all(1, &[0, 0]),
                    H::combine_all(2, &[0, 0, 0, 0]),
                    H::combine_all(3, &[]),
                ]
            ))
        );
    }

    {
        let ops = vec![
            Append(H::from_u64(0), Marked),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Append(H::from_u64(0), Ephemeral),
            Operation::Checkpoint(C::from_u64(1)),
            Append(H::from_u64(0), Marked),
            Operation::Checkpoint(C::from_u64(2)),
            Operation::Checkpoint(C::from_u64(3)),
            Append(
                H::from_u64(0),
                Checkpoint {
                    id: C::from_u64(4),
                    marking: Marking::None,
                },
            ),
            Rewind(2),
            Witness(Position::from(7), 2),
        ];
        let mut tree = new_tree(100);
        assert_eq!(Operation::apply_all(&ops, &mut tree), None);
    }

    {
        let ops = vec![
            Append(H::from_u64(0), Marked),
            Append(H::from_u64(0), Ephemeral),
            Append(
                H::from_u64(0),
                Checkpoint {
                    id: C::from_u64(1),
                    marking: Marking::Marked,
                },
            ),
            Append(
                H::from_u64(0),
                Checkpoint {
                    id: C::from_u64(4),
                    marking: Marking::None,
                },
            ),
            Witness(Position::from(2), 1),
        ];
        let mut tree = new_tree(100);
        assert_eq!(
            Operation::apply_all(&ops, &mut tree),
            Some((
                Position::from(2),
                vec![
                    H::empty_leaf(),
                    H::combine_all(1, &[0, 0]),
                    H::combine_all(2, &[]),
                    H::combine_all(3, &[]),
                ]
            ))
        );
    }
}

pub fn check_checkpoint_rewind<C: TestCheckpoint, T: Tree<String, C>, F: Fn(usize) -> T>(
    new_tree: F,
) {
    let mut t = new_tree(100);
    assert!(!t.rewind(0));

    let mut t = new_tree(100);
    t.assert_checkpoint(1);
    assert!(t.rewind(0));
    assert!(!t.rewind(1));

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Ephemeral);
    t.assert_checkpoint(1);
    t.append("b".to_string(), Retention::Marked);
    assert_eq!(Some(Position::from(1)), t.current_position());
    assert!(t.rewind(0));
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Marked);
    t.assert_checkpoint(1);
    assert!(t.rewind(0));
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Marked);
    t.assert_checkpoint(1);
    t.append("a".to_string(), Retention::Ephemeral);
    assert!(t.rewind(0));
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Ephemeral);
    t.assert_checkpoint(1);
    t.assert_checkpoint(2);
    assert!(t.rewind(1));
    t.append("b".to_string(), Retention::Ephemeral);
    assert!(t.rewind(0));
    t.append("b".to_string(), Retention::Ephemeral);
    assert_eq!(t.root(None).unwrap(), "ab______________");
}

pub fn check_remove_mark<C: TestCheckpoint, T: Tree<String, C>, F: Fn(usize) -> T>(new_tree: F) {
    let samples = vec![
        vec![
            append_str("a", Retention::Ephemeral),
            append_str(
                "a",
                Retention::Checkpoint {
                    id: C::from_u64(1),
                    marking: Marking::Marked,
                },
            ),
            witness(1, 1),
        ],
        vec![
            append_str("a", Retention::Ephemeral),
            append_str("a", Retention::Ephemeral),
            append_str("a", Retention::Ephemeral),
            append_str("a", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(3),
            witness(3, 0),
        ],
    ];

    for (i, sample) in samples.iter().enumerate() {
        let result = check_operations(new_tree(100), sample);
        assert!(
            matches!(result, Ok(())),
            "Reference/Test mismatch at index {}: {:?}",
            i,
            result
        );
    }
}

pub fn check_rewind_remove_mark<C: TestCheckpoint, T: Tree<String, C>, F: Fn(usize) -> T>(
    new_tree: F,
) {
    // rewinding doesn't remove a mark
    let mut tree = new_tree(100);
    tree.append("e".to_string(), Retention::Marked);
    tree.assert_checkpoint(1);
    assert!(tree.rewind(0));
    assert!(tree.remove_mark(0u64.into()));

    // use a maximum number of checkpoints of 1
    let mut tree = new_tree(1);
    assert!(tree.append("e".to_string(), Retention::Marked));
    tree.assert_checkpoint(1);
    assert!(tree.marked_positions().contains(&0u64.into()));
    assert!(tree.append("f".to_string(), Retention::Ephemeral));
    // simulate a spend of `e` at `f`
    assert!(tree.remove_mark(0u64.into()));
    // even though the mark has been staged for removal, it's not gone yet
    assert!(tree.marked_positions().contains(&0u64.into()));
    tree.assert_checkpoint(2);
    // the newest checkpoint will have caused the oldest to roll off, and
    // so the forgotten node will be unmarked
    assert!(!tree.marked_positions().contains(&0u64.into()));
    assert!(!tree.remove_mark(0u64.into()));

    // The following check_operations tests cover errors where the
    // test framework itself previously did not correctly handle
    // chain state restoration.

    let samples = vec![
        vec![
            append_str("x", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            Rewind(0),
            unmark(0),
        ],
        vec![
            append_str("d", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            Rewind(0),
            unmark(0),
        ],
        vec![
            append_str("o", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            Checkpoint(C::from_u64(2)),
            unmark(0),
            Rewind(0),
            Rewind(1),
        ],
        vec![
            append_str("s", Retention::Marked),
            append_str("m", Retention::Ephemeral),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            Rewind(0),
            unmark(0),
            unmark(0),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            Rewind(0),
            append_str("a", Retention::Marked),
        ],
    ];

    for (i, sample) in samples.iter().enumerate() {
        let result = check_operations(new_tree(100), sample);
        assert!(
            matches!(result, Ok(())),
            "Reference/Test mismatch at index {}: {:?}",
            i,
            result
        );
    }
}

pub fn check_witness_consistency<C: TestCheckpoint, T: Tree<String, C>, F: Fn(usize) -> T>(
    new_tree: F,
) {
    let samples = vec![
        // Reduced examples
        vec![
            append_str("a", Retention::Ephemeral),
            append_str("b", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            witness(0, 1),
        ],
        vec![
            append_str("c", Retention::Ephemeral),
            append_str("d", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            witness(1, 1),
        ],
        vec![
            append_str("e", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            append_str("f", Retention::Ephemeral),
            witness(0, 1),
        ],
        vec![
            append_str("g", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            append_str("h", Retention::Ephemeral),
            witness(0, 0),
        ],
        vec![
            append_str("i", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            append_str("j", Retention::Ephemeral),
            witness(0, 0),
        ],
        vec![
            append_str("i", Retention::Marked),
            append_str("j", Retention::Ephemeral),
            Checkpoint(C::from_u64(1)),
            append_str("k", Retention::Ephemeral),
            witness(0, 1),
        ],
        vec![
            append_str("l", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            Checkpoint(C::from_u64(2)),
            append_str("m", Retention::Ephemeral),
            Checkpoint(C::from_u64(3)),
            witness(0, 2),
        ],
        vec![
            Checkpoint(C::from_u64(1)),
            append_str("n", Retention::Marked),
            witness(0, 1),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            Checkpoint(C::from_u64(2)),
            append_str("b", Retention::Ephemeral),
            witness(0, 1),
        ],
        vec![
            append_str("a", Retention::Marked),
            append_str("b", Retention::Ephemeral),
            unmark(0),
            Checkpoint(C::from_u64(1)),
            witness(0, 0),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            Checkpoint(C::from_u64(2)),
            Rewind(0),
            append_str("b", Retention::Ephemeral),
            witness(0, 0),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            Checkpoint(C::from_u64(2)),
            Rewind(1),
            append_str("a", Retention::Ephemeral),
            unmark(0),
            witness(0, 1),
        ],
        // Unreduced examples
        vec![
            append_str("o", Retention::Ephemeral),
            append_str("p", Retention::Marked),
            append_str("q", Retention::Ephemeral),
            Checkpoint(C::from_u64(1)),
            unmark(1),
            witness(1, 1),
        ],
        vec![
            append_str("r", Retention::Ephemeral),
            append_str("s", Retention::Ephemeral),
            append_str("t", Retention::Marked),
            Checkpoint(C::from_u64(1)),
            unmark(2),
            Checkpoint(C::from_u64(2)),
            witness(2, 2),
        ],
        vec![
            append_str("u", Retention::Marked),
            append_str("v", Retention::Ephemeral),
            append_str("w", Retention::Ephemeral),
            Checkpoint(C::from_u64(1)),
            unmark(0),
            append_str("x", Retention::Ephemeral),
            Checkpoint(C::from_u64(2)),
            Checkpoint(C::from_u64(3)),
            witness(0, 3),
        ],
    ];

    for (i, sample) in samples.iter().enumerate() {
        let result = check_operations(new_tree(100), sample);
        assert!(
            matches!(result, Ok(())),
            "Reference/Test mismatch at index {}: {:?}",
            i,
            result
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{compute_root_from_witness, SipHashable};
    use incrementalmerkletree::{Hashable, Level, Position};

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
}
