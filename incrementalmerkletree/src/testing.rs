//! Traits and types used to permit comparison testing between tree implementations.

use core::fmt::Debug;
use core::marker::PhantomData;
use proptest::prelude::*;
use std::collections::BTreeSet;

use crate::{Hashable, Level, Position, Retention};

pub mod complete_tree;

//
// Traits used to permit comparison testing between tree implementations.
//

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

/// A Merkle tree that supports incremental appends, marking of
/// leaf nodes for construction of witnesses, checkpoints and rollbacks.
pub trait Tree<H, C> {
    /// Returns the depth of the tree.
    fn depth(&self) -> u8;

    /// Appends a new value to the tree at the next available slot.
    /// Returns true if successful and false if the tree would exceed
    /// the maximum allowed depth.
    fn append(&mut self, value: H, retention: Retention<C>) -> bool;

    /// Returns the most recently appended leaf value.
    fn current_position(&self) -> Option<Position>;

    /// Returns the leaf at the specified position if the tree can produce
    /// a witness for it.
    fn get_marked_leaf(&self, position: Position) -> Option<&H>;

    /// Return a set of all the positions for which we have marked.
    fn marked_positions(&self) -> BTreeSet<Position>;

    /// Obtains the root of the Merkle tree at the specified checkpoint depth
    /// by hashing against empty nodes up to the maximum height of the tree.
    /// Returns `None` if there are not enough checkpoints available to reach the
    /// requested checkpoint depth.
    fn root(&self, checkpoint_depth: usize) -> Option<H>;

    /// Obtains a witness for the value at the specified leaf position, as of the tree state at the
    /// given checkpoint depth. Returns `None` if there is no witness information for the requested
    /// position or if no checkpoint is available at the specified depth.
    fn witness(&self, position: Position, checkpoint_depth: usize) -> Option<Vec<H>>;

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a mark for. Returns true if successful and
    /// false if we were already not maintaining a mark at this position.
    fn remove_mark(&mut self, position: Position) -> bool;

    /// Creates a new checkpoint for the current tree state.
    ///
    /// It is valid to have multiple checkpoints for the same tree state, and each `rewind` call
    /// will remove a single checkpoint. Returns `false` if the checkpoint identifier provided is
    /// less than or equal to the maximum checkpoint identifier observed.
    fn checkpoint(&mut self, id: C) -> bool;

    /// Rewinds the tree state to the previous checkpoint, and then removes that checkpoint record.
    ///
    /// If there are multiple checkpoints at a given tree state, the tree state will not be altered
    /// until all checkpoints at that tree state have been removed using `rewind`. This function
    /// will return false and leave the tree unmodified if no checkpoints exist.
    fn rewind(&mut self) -> bool;
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

impl Hashable for String {
    fn empty_leaf() -> Self {
        "_".to_string()
    }

    fn combine(_: Level, a: &Self, b: &Self) -> Self {
        a.to_string() + b
    }
}

impl<H: Hashable> Hashable for Option<H> {
    fn empty_leaf() -> Self {
        Some(H::empty_leaf())
    }

    fn combine(l: Level, a: &Self, b: &Self) -> Self {
        match (a, b) {
            (Some(a), Some(b)) => Some(H::combine(l, a, b)),
            _ => None,
        }
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
    Rewind,
    Witness(Position, usize),
    GarbageCollect,
}

use Operation::*;

pub fn append_str<C>(x: &str, retention: Retention<C>) -> Operation<String, C> {
    Operation::Append(x.to_string(), retention)
}

pub fn unmark<H, C>(pos: usize) -> Operation<H, C> {
    Operation::Unmark(Position::from(pos))
}

pub fn witness<H, C>(pos: usize, depth: usize) -> Operation<H, C> {
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
            Rewind => {
                assert!(tree.rewind(), "rewind failed");
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
            Rewind => Rewind,
            Witness(p, d) => Witness(*p, *d),
            GarbageCollect => GarbageCollect,
        }
    }
}

pub fn arb_retention() -> impl Strategy<Value = Retention<()>> {
    prop_oneof![
        Just(Retention::Ephemeral),
        any::<bool>().prop_map(|is_marked| Retention::Checkpoint { id: (), is_marked }),
        Just(Retention::Marked),
    ]
}

pub fn arb_operation<G: Strategy + Clone>(
    item_gen: G,
    pos_gen: impl Strategy<Value = usize> + Clone,
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
        pos_gen
            .clone()
            .prop_map(|i| Operation::MarkedLeaf(Position::from(i))),
        pos_gen
            .clone()
            .prop_map(|i| Operation::Unmark(Position::from(i))),
        Just(Operation::Checkpoint(())),
        Just(Operation::Rewind),
        pos_gen
            .prop_flat_map(|i| (0usize..10)
                .prop_map(move |depth| Operation::Witness(Position::from(i), depth))),
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
        Rewind => {
            tree.rewind();
        }
        CurrentPosition => {}
        Witness(_, _) => {}
        MarkedLeaf(_) => {}
        MarkedPositions => {}
        GarbageCollect => {}
    }
}

pub fn check_operations<H: Hashable + Ord + Clone, C: Clone, T: Tree<H, C>>(
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
                        tree.current_position().map_or(0, |p| usize::from(p) + 1)
                    );
                }
            }
            CurrentPosition => {
                if let Some(pos) = tree.current_position() {
                    prop_assert!(tree_size > 0);
                    prop_assert_eq!(tree_size - 1, pos.into());
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
            Checkpoint(id) => {
                tree_checkpoints.push(tree_size);
                tree.checkpoint(id.clone());
            }
            Rewind => {
                if tree.rewind() {
                    prop_assert!(!tree_checkpoints.is_empty());
                    let checkpointed_tree_size = tree_checkpoints.pop().unwrap();
                    tree_values.truncate(checkpointed_tree_size);
                    tree_size = checkpointed_tree_size;
                }
            }
            Witness(position, depth) => {
                if let Some(path) = tree.witness(*position, *depth) {
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

                        // compute the root
                        let expected_root =
                            complete_tree::root::<H>(&extended_tree_values, tree.depth());
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

pub fn compute_root_from_witness<H: Hashable>(value: H, position: Position, path: &[H]) -> H {
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

    fn get_marked_leaf(&self, position: Position) -> Option<&H> {
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

    fn rewind(&mut self) -> bool {
        let a = self.inefficient.rewind();
        let b = self.efficient.rewind();
        assert_eq!(a, b);
        a
    }
}

//
// Shared example tests
//

pub fn check_root_hashes<T: Tree<String, usize>, F: Fn(usize) -> T>(new_tree: F) {
    let mut tree = new_tree(100);
    assert_eq!(tree.root(0).unwrap(), "________________");

    tree.append("a".to_string(), Retention::Ephemeral);
    assert_eq!(tree.root(0).unwrap().len(), 16);
    assert_eq!(tree.root(0).unwrap(), "a_______________");

    tree.append("b".to_string(), Retention::Ephemeral);
    assert_eq!(tree.root(0).unwrap(), "ab______________");

    tree.append("c".to_string(), Retention::Ephemeral);
    assert_eq!(tree.root(0).unwrap(), "abc_____________");

    let mut t = new_tree(100);
    t.append(
        "a".to_string(),
        Retention::Checkpoint {
            id: 1,
            is_marked: true,
        },
    );
    t.append("a".to_string(), Retention::Ephemeral);
    t.append("a".to_string(), Retention::Ephemeral);
    t.append("a".to_string(), Retention::Ephemeral);
    assert_eq!(t.root(0).unwrap(), "aaaa____________");
}

/// This test expects a depth-4 tree and verifies that the tree reports itself as full after 2^4
/// appends.
pub fn check_append<T: Tree<String, usize> + std::fmt::Debug, F: Fn(usize) -> T>(new_tree: F) {
    use Retention::*;

    let mut tree = new_tree(100);
    assert_eq!(tree.depth(), 4);

    // 16 appends should succeed
    for i in 0..16 {
        assert!(tree.append(i.to_string(), Ephemeral));
        assert_eq!(tree.current_position(), Some(Position::from(i)));
    }

    // 17th append should fail
    assert!(!tree.append("16".to_string(), Ephemeral));

    // The following checks a condition on state restoration in the case that an append fails.
    // We want to ensure that a failed append does not cause a loss of information.
    let ops = (0..17)
        .map(|i| Append(i.to_string(), Ephemeral))
        .collect::<Vec<_>>();
    let tree = new_tree(100);
    check_operations(tree, &ops).unwrap();
}

pub fn check_witnesses<T: Tree<String, usize> + std::fmt::Debug, F: Fn(usize) -> T>(new_tree: F) {
    use Retention::*;

    let mut tree = new_tree(100);
    tree.append("a".to_string(), Marked);
    assert_eq!(
        tree.witness(Position::from(0), 0),
        Some(vec![
            "_".to_string(),
            "__".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append("b".to_string(), Ephemeral);
    assert_eq!(
        tree.witness(0.into(), 0),
        Some(vec![
            "b".to_string(),
            "__".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append("c".to_string(), Marked);
    assert_eq!(
        tree.witness(Position::from(2), 0),
        Some(vec![
            "_".to_string(),
            "ab".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append("d".to_string(), Ephemeral);
    assert_eq!(
        tree.witness(Position::from(2), 0),
        Some(vec![
            "d".to_string(),
            "ab".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append("e".to_string(), Ephemeral);
    assert_eq!(
        tree.witness(Position::from(2), 0),
        Some(vec![
            "d".to_string(),
            "ab".to_string(),
            "e___".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append("a".to_string(), Marked);
    for c in 'b'..'g' {
        tree.append(c.to_string(), Ephemeral);
    }
    tree.append("g".to_string(), Marked);
    tree.append("h".to_string(), Ephemeral);

    assert_eq!(
        tree.witness(0.into(), 0),
        Some(vec![
            "b".to_string(),
            "cd".to_string(),
            "efgh".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append("a".to_string(), Marked);
    tree.append("b".to_string(), Ephemeral);
    tree.append("c".to_string(), Ephemeral);
    tree.append("d".to_string(), Marked);
    tree.append("e".to_string(), Marked);
    tree.append("f".to_string(), Marked);
    tree.append("g".to_string(), Ephemeral);

    assert_eq!(
        tree.witness(Position::from(5), 0),
        Some(vec![
            "e".to_string(),
            "g_".to_string(),
            "abcd".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    for c in 'a'..'k' {
        assert!(tree.append(c.to_string(), Ephemeral));
    }
    assert!(tree.append('k'.to_string(), Marked));
    assert!(tree.append('l'.to_string(), Ephemeral));

    assert_eq!(
        tree.witness(Position::from(10), 0),
        Some(vec![
            "l".to_string(),
            "ij".to_string(),
            "____".to_string(),
            "abcdefgh".to_string()
        ])
    );

    let mut tree = new_tree(100);
    assert!(tree.append(
        'a'.to_string(),
        Checkpoint {
            id: 1,
            is_marked: true
        }
    ));
    assert!(tree.rewind());
    for c in 'b'..'e' {
        tree.append(c.to_string(), Ephemeral);
    }
    tree.append("e".to_string(), Marked);
    for c in 'f'..'i' {
        tree.append(c.to_string(), Ephemeral);
    }
    assert_eq!(
        tree.witness(0.into(), 0),
        Some(vec![
            "b".to_string(),
            "cd".to_string(),
            "efgh".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append('a'.to_string(), Ephemeral);
    tree.append('b'.to_string(), Ephemeral);
    tree.append('c'.to_string(), Marked);
    tree.append('d'.to_string(), Ephemeral);
    tree.append('e'.to_string(), Ephemeral);
    tree.append('f'.to_string(), Ephemeral);
    assert!(tree.append(
        'g'.to_string(),
        Checkpoint {
            id: 1,
            is_marked: true
        }
    ));
    tree.append('h'.to_string(), Ephemeral);
    assert!(tree.rewind());
    assert_eq!(
        tree.witness(Position::from(2), 0),
        Some(vec![
            "d".to_string(),
            "ab".to_string(),
            "efg_".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append('a'.to_string(), Ephemeral);
    tree.append('b'.to_string(), Marked);
    assert_eq!(tree.witness(Position::from(0), 0), None);

    let mut tree = new_tree(100);
    for c in 'a'..'m' {
        tree.append(c.to_string(), Ephemeral);
    }
    tree.append('m'.to_string(), Marked);
    tree.append('n'.to_string(), Marked);
    tree.append('o'.to_string(), Ephemeral);
    tree.append('p'.to_string(), Ephemeral);

    assert_eq!(
        tree.witness(Position::from(12), 0),
        Some(vec![
            "n".to_string(),
            "op".to_string(),
            "ijkl".to_string(),
            "abcdefgh".to_string()
        ])
    );

    let ops = ('a'..='l')
        .map(|c| Append(c.to_string(), Marked))
        .chain(Some(Append('m'.to_string(), Ephemeral)))
        .chain(Some(Append('n'.to_string(), Ephemeral)))
        .chain(Some(Witness(11usize.into(), 0)))
        .collect::<Vec<_>>();

    let mut tree = new_tree(100);
    assert_eq!(
        Operation::apply_all(&ops, &mut tree),
        Some((
            Position::from(11),
            vec![
                "k".to_string(),
                "ij".to_string(),
                "mn__".to_string(),
                "abcdefgh".to_string()
            ]
        ))
    );

    let ops = vec![
        Append("a".to_string(), Ephemeral),
        Append("b".to_string(), Ephemeral),
        Append("c".to_string(), Ephemeral),
        Append(
            "d".to_string(),
            Checkpoint {
                id: 1,
                is_marked: true,
            },
        ),
        Append("e".to_string(), Marked),
        Operation::Checkpoint(2),
        Append(
            "f".to_string(),
            Checkpoint {
                id: 3,
                is_marked: false,
            },
        ),
        Append(
            "g".to_string(),
            Checkpoint {
                id: 4,
                is_marked: false,
            },
        ),
        Append(
            "h".to_string(),
            Checkpoint {
                id: 5,
                is_marked: false,
            },
        ),
        Witness(3usize.into(), 5),
    ];
    let mut tree = new_tree(100);
    assert_eq!(
        Operation::apply_all(&ops, &mut tree),
        Some((
            Position::from(3),
            vec![
                "c".to_string(),
                "ab".to_string(),
                "____".to_string(),
                "________".to_string()
            ]
        ))
    );
    let ops = vec![
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append(
            "a".to_string(),
            Checkpoint {
                id: 1,
                is_marked: true,
            },
        ),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append(
            "a".to_string(),
            Checkpoint {
                id: 2,
                is_marked: false,
            },
        ),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Witness(Position(3), 1),
    ];
    let mut tree = new_tree(100);
    assert_eq!(
        Operation::apply_all(&ops, &mut tree),
        Some((
            Position::from(3),
            vec![
                "a".to_string(),
                "aa".to_string(),
                "aaaa".to_string(),
                "________".to_string()
            ]
        ))
    );

    let ops = vec![
        Append("a".to_string(), Marked),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Append("a".to_string(), Ephemeral),
        Operation::Checkpoint(1),
        Append("a".to_string(), Marked),
        Operation::Checkpoint(2),
        Operation::Checkpoint(3),
        Append(
            "a".to_string(),
            Checkpoint {
                id: 4,
                is_marked: false,
            },
        ),
        Rewind,
        Rewind,
        Witness(Position(7), 2),
    ];
    let mut tree = new_tree(100);
    assert_eq!(Operation::apply_all(&ops, &mut tree), None);

    let ops = vec![
        Append("a".to_string(), Marked),
        Append("a".to_string(), Ephemeral),
        Append(
            "a".to_string(),
            Checkpoint {
                id: 1,
                is_marked: true,
            },
        ),
        Append(
            "a".to_string(),
            Checkpoint {
                id: 4,
                is_marked: false,
            },
        ),
        Witness(Position(2), 2),
    ];
    let mut tree = new_tree(100);
    assert_eq!(
        Operation::apply_all(&ops, &mut tree),
        Some((
            Position::from(2),
            vec![
                "_".to_string(),
                "aa".to_string(),
                "____".to_string(),
                "________".to_string()
            ]
        ))
    );
}

pub fn check_checkpoint_rewind<T: Tree<String, usize>, F: Fn(usize) -> T>(new_tree: F) {
    let mut t = new_tree(100);
    assert!(!t.rewind());

    let mut t = new_tree(100);
    t.checkpoint(1);
    assert!(t.rewind());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Ephemeral);
    t.checkpoint(1);
    t.append("b".to_string(), Retention::Marked);
    assert!(t.rewind());
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Marked);
    t.checkpoint(1);
    assert!(t.rewind());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Marked);
    t.checkpoint(1);
    t.append("a".to_string(), Retention::Ephemeral);
    assert!(t.rewind());
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append("a".to_string(), Retention::Ephemeral);
    t.checkpoint(1);
    t.checkpoint(2);
    assert!(t.rewind());
    t.append("b".to_string(), Retention::Ephemeral);
    assert!(t.rewind());
    t.append("b".to_string(), Retention::Ephemeral);
    assert_eq!(t.root(0).unwrap(), "ab______________");
}

pub fn check_remove_mark<T: Tree<String, usize>, F: Fn(usize) -> T>(new_tree: F) {
    let samples = vec![
        vec![
            append_str("a", Retention::Ephemeral),
            append_str(
                "a",
                Retention::Checkpoint {
                    id: 1,
                    is_marked: true,
                },
            ),
            witness(1, 1),
        ],
        vec![
            append_str("a", Retention::Ephemeral),
            append_str("a", Retention::Ephemeral),
            append_str("a", Retention::Ephemeral),
            append_str("a", Retention::Marked),
            Checkpoint(1),
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

pub fn check_rewind_remove_mark<T: Tree<String, usize>, F: Fn(usize) -> T>(new_tree: F) {
    // rewinding doesn't remove a mark
    let mut tree = new_tree(100);
    tree.append("e".to_string(), Retention::Marked);
    tree.checkpoint(1);
    assert!(tree.rewind());
    assert!(tree.remove_mark(0usize.into()));

    // use a maximum number of checkpoints of 1
    let mut tree = new_tree(1);
    assert!(tree.append("e".to_string(), Retention::Marked));
    assert!(tree.checkpoint(1));
    assert!(tree.marked_positions().contains(&0usize.into()));
    assert!(tree.append("f".to_string(), Retention::Ephemeral));
    // simulate a spend of `e` at `f`
    assert!(tree.remove_mark(0usize.into()));
    // even though the mark has been staged for removal, it's not gone yet
    assert!(tree.marked_positions().contains(&0usize.into()));
    assert!(tree.checkpoint(2));
    // the newest checkpoint will have caused the oldest to roll off, and
    // so the forgotten node will be unmarked
    assert!(!tree.marked_positions().contains(&0usize.into()));
    assert!(!tree.remove_mark(0usize.into()));

    // The following check_operations tests cover errors where the
    // test framework itself previously did not correctly handle
    // chain state restoration.

    let samples = vec![
        vec![
            append_str("x", Retention::Marked),
            Checkpoint(1),
            Rewind,
            unmark(0),
        ],
        vec![
            append_str("d", Retention::Marked),
            Checkpoint(1),
            unmark(0),
            Rewind,
            unmark(0),
        ],
        vec![
            append_str("o", Retention::Marked),
            Checkpoint(1),
            Checkpoint(2),
            unmark(0),
            Rewind,
            Rewind,
        ],
        vec![
            append_str("s", Retention::Marked),
            append_str("m", Retention::Ephemeral),
            Checkpoint(1),
            unmark(0),
            Rewind,
            unmark(0),
            unmark(0),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(1),
            Rewind,
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

pub fn check_witness_consistency<T: Tree<String, usize>, F: Fn(usize) -> T>(new_tree: F) {
    let samples = vec![
        // Reduced examples
        vec![
            append_str("a", Retention::Ephemeral),
            append_str("b", Retention::Marked),
            Checkpoint(1),
            witness(0, 1),
        ],
        vec![
            append_str("c", Retention::Ephemeral),
            append_str("d", Retention::Marked),
            Checkpoint(1),
            witness(1, 1),
        ],
        vec![
            append_str("e", Retention::Marked),
            Checkpoint(1),
            append_str("f", Retention::Ephemeral),
            witness(0, 1),
        ],
        vec![
            append_str("g", Retention::Marked),
            Checkpoint(1),
            unmark(0),
            append_str("h", Retention::Ephemeral),
            witness(0, 0),
        ],
        vec![
            append_str("i", Retention::Marked),
            Checkpoint(1),
            unmark(0),
            append_str("j", Retention::Ephemeral),
            witness(0, 0),
        ],
        vec![
            append_str("i", Retention::Marked),
            append_str("j", Retention::Ephemeral),
            Checkpoint(1),
            append_str("k", Retention::Ephemeral),
            witness(0, 1),
        ],
        vec![
            append_str("l", Retention::Marked),
            Checkpoint(1),
            Checkpoint(2),
            append_str("m", Retention::Ephemeral),
            Checkpoint(3),
            witness(0, 2),
        ],
        vec![
            Checkpoint(1),
            append_str("n", Retention::Marked),
            witness(0, 1),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(1),
            unmark(0),
            Checkpoint(2),
            append_str("b", Retention::Ephemeral),
            witness(0, 1),
        ],
        vec![
            append_str("a", Retention::Marked),
            append_str("b", Retention::Ephemeral),
            unmark(0),
            Checkpoint(1),
            witness(0, 0),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(1),
            unmark(0),
            Checkpoint(2),
            Rewind,
            append_str("b", Retention::Ephemeral),
            witness(0, 0),
        ],
        vec![
            append_str("a", Retention::Marked),
            Checkpoint(1),
            Checkpoint(2),
            Rewind,
            append_str("a", Retention::Ephemeral),
            unmark(0),
            witness(0, 1),
        ],
        // Unreduced examples
        vec![
            append_str("o", Retention::Ephemeral),
            append_str("p", Retention::Marked),
            append_str("q", Retention::Ephemeral),
            Checkpoint(1),
            unmark(1),
            witness(1, 1),
        ],
        vec![
            append_str("r", Retention::Ephemeral),
            append_str("s", Retention::Ephemeral),
            append_str("t", Retention::Marked),
            Checkpoint(1),
            unmark(2),
            Checkpoint(2),
            witness(2, 2),
        ],
        vec![
            append_str("u", Retention::Marked),
            append_str("v", Retention::Ephemeral),
            append_str("w", Retention::Ephemeral),
            Checkpoint(1),
            unmark(0),
            append_str("x", Retention::Ephemeral),
            Checkpoint(2),
            Checkpoint(3),
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
    use crate::{
        testing::{compute_root_from_witness, SipHashable},
        Hashable, Level, Position,
    };

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
