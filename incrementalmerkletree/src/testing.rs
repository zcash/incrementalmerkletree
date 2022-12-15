//! Traits and types used to permit comparison testing between tree implementations.

use core::fmt::Debug;
use core::marker::PhantomData;
use proptest::prelude::*;
use std::collections::BTreeSet;

use crate::{Hashable, Level, Position};

pub mod complete_tree;

//
// Traits used to permit comparison testing between tree implementations.
//

/// A possibly-empty incremental Merkle frontier.
pub trait Frontier<H> {
    /// Appends a new value to the frontier at the next available slot.
    /// Returns true if successful and false if the frontier would exceed
    /// the maximum allowed depth.
    fn append(&mut self, value: &H) -> bool;

    /// Obtains the current root of this Merkle frontier by hashing
    /// against empty nodes up to the maximum height of the pruned
    /// tree that the frontier represents.
    fn root(&self) -> H;
}

/// A Merkle tree that supports incremental appends, marking of
/// leaf nodes for construction of witnesses, checkpoints and rollbacks.
pub trait Tree<H> {
    /// Appends a new value to the tree at the next available slot.
    /// Returns true if successful and false if the tree would exceed
    /// the maximum allowed depth.
    fn append(&mut self, value: &H) -> bool;

    /// Returns the most recently appended leaf value.
    fn current_position(&self) -> Option<Position>;

    /// Returns the most recently appended leaf value.
    fn current_leaf(&self) -> Option<&H>;

    /// Returns the leaf at the specified position if the tree can produce
    /// a witness for it.
    fn get_marked_leaf(&self, position: Position) -> Option<&H>;

    /// Marks the current leaf as one for which we're interested in producing
    /// a witness. Returns an optional value containing the
    /// current position if successful or if the current value was already
    /// marked, or None if the tree is empty.
    fn mark(&mut self) -> Option<Position>;

    /// Return a set of all the positions for which we have marked.
    fn marked_positions(&self) -> BTreeSet<Position>;

    /// Obtains the root of the Merkle tree at the specified checkpoint depth
    /// by hashing against empty nodes up to the maximum height of the tree.
    /// Returns `None` if there are not enough checkpoints available to reach the
    /// requested checkpoint depth.
    fn root(&self, checkpoint_depth: usize) -> Option<H>;

    /// Obtains a witness to the value at the specified position,
    /// as of the tree state corresponding to the given root.
    /// Returns `None` if there is no available witness to that
    /// position or if the root does not correspond to a checkpointed
    /// root of the tree.
    fn witness(&self, position: Position, as_of_root: &H) -> Option<Vec<H>>;

    /// Marks the value at the specified position as a value we're no longer
    /// interested in maintaining a mark for. Returns true if successful and
    /// false if we were already not maintaining a mark at this position.
    fn remove_mark(&mut self, position: Position) -> bool;

    /// Creates a new checkpoint for the current tree state. It is valid to
    /// have multiple checkpoints for the same tree state, and each `rewind`
    /// call will remove a single checkpoint.
    fn checkpoint(&mut self);

    /// Rewinds the tree state to the previous checkpoint, and then removes
    /// that checkpoint record. If there are multiple checkpoints at a given
    /// tree state, the tree state will not be altered until all checkpoints
    /// at that tree state have been removed using `rewind`. This function
    /// return false and leave the tree unmodified if no checkpoints exist.
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

//
// Operations
//

#[derive(Clone, Debug)]
pub enum Operation<A> {
    Append(A),
    CurrentPosition,
    CurrentLeaf,
    Mark,
    MarkedLeaf(Position),
    MarkedPositions,
    Unmark(Position),
    Checkpoint,
    Rewind,
    Authpath(Position, usize),
    GarbageCollect,
}

use Operation::*;

pub fn append_str(x: &str) -> Operation<String> {
    Operation::Append(x.to_string())
}

pub fn unmark<T>(pos: usize) -> Operation<T> {
    Operation::Unmark(Position::from(pos))
}

pub fn witness<T>(pos: usize, depth: usize) -> Operation<T> {
    Operation::Authpath(Position::from(pos), depth)
}

impl<H: Hashable> Operation<H> {
    pub fn apply<T: Tree<H>>(&self, tree: &mut T) -> Option<(Position, Vec<H>)> {
        match self {
            Append(a) => {
                assert!(tree.append(a), "append failed");
                None
            }
            CurrentPosition => None,
            CurrentLeaf => None,
            Mark => {
                assert!(tree.mark().is_some(), "mark failed");
                None
            }
            MarkedLeaf(_) => None,
            MarkedPositions => None,
            Unmark(p) => {
                assert!(tree.remove_mark(*p), "remove mark failed");
                None
            }
            Checkpoint => {
                tree.checkpoint();
                None
            }
            Rewind => {
                assert!(tree.rewind(), "rewind failed");
                None
            }
            Authpath(p, d) => tree
                .root(*d)
                .and_then(|root| tree.witness(*p, &root))
                .map(|xs| (*p, xs)),
            GarbageCollect => None,
        }
    }

    pub fn apply_all<T: Tree<H>>(ops: &[Operation<H>], tree: &mut T) -> Option<(Position, Vec<H>)> {
        let mut result = None;
        for op in ops {
            result = op.apply(tree);
        }
        result
    }
}

pub fn arb_operation<G: Strategy + Clone>(
    item_gen: G,
    pos_gen: impl Strategy<Value = usize> + Clone,
) -> impl Strategy<Value = Operation<G::Value>>
where
    G::Value: Clone + 'static,
{
    prop_oneof![
        item_gen.prop_map(Operation::Append),
        Just(Operation::Mark),
        prop_oneof![
            Just(Operation::CurrentLeaf),
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
        Just(Operation::Checkpoint),
        Just(Operation::Rewind),
        pos_gen
            .prop_flat_map(|i| (0usize..10)
                .prop_map(move |depth| Operation::Authpath(Position::from(i), depth))),
    ]
}

pub fn apply_operation<H, T: Tree<H>>(tree: &mut T, op: Operation<H>) {
    match op {
        Append(value) => {
            tree.append(&value);
        }
        Mark => {
            tree.mark();
        }
        Unmark(position) => {
            tree.remove_mark(position);
        }
        Checkpoint => {
            tree.checkpoint();
        }
        Rewind => {
            tree.rewind();
        }
        CurrentPosition => {}
        CurrentLeaf => {}
        Authpath(_, _) => {}
        MarkedLeaf(_) => {}
        MarkedPositions => {}
        GarbageCollect => {}
    }
}

pub fn check_operations<H: Hashable + Ord + Clone + Debug, T: Tree<H>>(
    mut tree: T,
    tree_depth: u8,
    ops: &[Operation<H>],
) -> Result<(), TestCaseError> {
    let mut tree_size = 0;
    let mut tree_values: Vec<H> = vec![];
    // the number of leaves in the tree at the time that a checkpoint is made
    let mut tree_checkpoints: Vec<usize> = vec![];

    for op in ops {
        prop_assert_eq!(tree_size, tree_values.len());
        match op {
            Append(value) => {
                if tree.append(value) {
                    prop_assert!(tree_size < (1 << tree_depth));
                    tree_size += 1;
                    tree_values.push(value.clone());
                } else {
                    prop_assert_eq!(tree_size, 1 << tree_depth);
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
                if let Some(path) = tree.root(*depth).and_then(|r| tree.witness(*position, &r)) {
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
                        extended_tree_values.resize(1 << tree_depth, H::empty_leaf());

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

pub fn lazy_root<H: Hashable + Clone>(mut leaves: Vec<H>) -> H {
    //leaves are always at level zero, so we start there.
    let mut level = Level::from(0);
    while leaves.len() != 1 {
        leaves = leaves
            .iter()
            .enumerate()
            .filter(|(i, _)| (i % 2) == 0)
            .map(|(_, a)| a)
            .zip(
                leaves
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i % 2) == 1)
                    .map(|(_, b)| b),
            )
            .map(|(a, b)| H::combine(level, a, b))
            .collect();
        level = level + 1;
    }

    leaves[0].clone()
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

#[derive(Clone)]
pub struct CombinedTree<H, I: Tree<H>, E: Tree<H>> {
    inefficient: I,
    efficient: E,
    _phantom: PhantomData<H>,
}

impl<H: Hashable + Ord + Clone + Debug, I: Tree<H>, E: Tree<H>> CombinedTree<H, I, E> {
    pub fn new(inefficient: I, efficient: E) -> Self {
        CombinedTree {
            inefficient,
            efficient,
            _phantom: PhantomData,
        }
    }
}

impl<H: Hashable + Ord + Clone + Debug, I: Tree<H>, E: Tree<H>> Tree<H> for CombinedTree<H, I, E> {
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
//
// Shared example tests
//

pub fn check_root_hashes<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
    let mut tree = new_tree(100);
    assert_eq!(tree.root(0).unwrap(), "________________");

    tree.append(&"a".to_string());
    assert_eq!(tree.root(0).unwrap().len(), 16);
    assert_eq!(tree.root(0).unwrap(), "a_______________");

    tree.append(&"b".to_string());
    assert_eq!(tree.root(0).unwrap(), "ab______________");

    tree.append(&"c".to_string());
    assert_eq!(tree.root(0).unwrap(), "abc_____________");

    let mut t = new_tree(100);
    t.append(&"a".to_string());
    t.checkpoint();
    t.mark();
    t.append(&"a".to_string());
    t.append(&"a".to_string());
    t.append(&"a".to_string());
    assert_eq!(t.root(0).unwrap(), "aaaa____________");
}

pub fn check_witnesses<T: Tree<String> + std::fmt::Debug, F: Fn(usize) -> T>(new_tree: F) {
    let mut tree = new_tree(100);
    tree.append(&"a".to_string());
    tree.mark();
    assert_eq!(
        tree.witness(Position::from(0), &tree.root(0).unwrap()),
        Some(vec![
            "_".to_string(),
            "__".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append(&"b".to_string());
    assert_eq!(
        tree.witness(0.into(), &tree.root(0).unwrap()),
        Some(vec![
            "b".to_string(),
            "__".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append(&"c".to_string());
    tree.mark();
    assert_eq!(
        tree.witness(Position::from(2), &tree.root(0).unwrap()),
        Some(vec![
            "_".to_string(),
            "ab".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append(&"d".to_string());
    assert_eq!(
        tree.witness(Position::from(2), &tree.root(0).unwrap()),
        Some(vec![
            "d".to_string(),
            "ab".to_string(),
            "____".to_string(),
            "________".to_string()
        ])
    );

    tree.append(&"e".to_string());
    assert_eq!(
        tree.witness(Position::from(2), &tree.root(0).unwrap()),
        Some(vec![
            "d".to_string(),
            "ab".to_string(),
            "e___".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append(&"a".to_string());
    tree.mark();
    for c in 'b'..'h' {
        tree.append(&c.to_string());
    }
    tree.mark();
    tree.append(&"h".to_string());

    assert_eq!(
        tree.witness(0.into(), &tree.root(0).unwrap()),
        Some(vec![
            "b".to_string(),
            "cd".to_string(),
            "efgh".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append(&"a".to_string());
    tree.mark();
    tree.append(&"b".to_string());
    tree.append(&"c".to_string());
    tree.append(&"d".to_string());
    tree.mark();
    tree.append(&"e".to_string());
    tree.mark();
    tree.append(&"f".to_string());
    tree.mark();
    tree.append(&"g".to_string());

    assert_eq!(
        tree.witness(Position::from(5), &tree.root(0).unwrap()),
        Some(vec![
            "e".to_string(),
            "g_".to_string(),
            "abcd".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    for c in 'a'..'l' {
        tree.append(&c.to_string());
    }
    tree.mark();
    tree.append(&'l'.to_string());

    assert_eq!(
        tree.witness(Position::from(10), &tree.root(0).unwrap()),
        Some(vec![
            "l".to_string(),
            "ij".to_string(),
            "____".to_string(),
            "abcdefgh".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append(&'a'.to_string());
    tree.mark();
    tree.checkpoint();
    assert!(tree.rewind());
    for c in 'b'..'f' {
        tree.append(&c.to_string());
    }
    tree.mark();
    for c in 'f'..'i' {
        tree.append(&c.to_string());
    }

    assert_eq!(
        tree.witness(0.into(), &tree.root(0).unwrap()),
        Some(vec![
            "b".to_string(),
            "cd".to_string(),
            "efgh".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append(&'a'.to_string());
    tree.append(&'b'.to_string());
    tree.append(&'c'.to_string());
    tree.mark();
    tree.append(&'d'.to_string());
    tree.append(&'e'.to_string());
    tree.append(&'f'.to_string());
    tree.append(&'g'.to_string());
    tree.mark();
    tree.checkpoint();
    tree.append(&'h'.to_string());
    assert!(tree.rewind());

    assert_eq!(
        tree.witness(Position::from(2), &tree.root(0).unwrap()),
        Some(vec![
            "d".to_string(),
            "ab".to_string(),
            "efg_".to_string(),
            "________".to_string()
        ])
    );

    let mut tree = new_tree(100);
    tree.append(&'a'.to_string());
    tree.append(&'b'.to_string());
    tree.mark();
    assert_eq!(
        tree.witness(Position::from(0), &tree.root(0).unwrap()),
        None
    );

    let mut tree = new_tree(100);
    for c in 'a'..'n' {
        tree.append(&c.to_string());
    }
    tree.mark();
    tree.append(&'n'.to_string());
    tree.mark();
    tree.append(&'o'.to_string());
    tree.append(&'p'.to_string());

    assert_eq!(
        tree.witness(Position::from(12), &tree.root(0).unwrap()),
        Some(vec![
            "n".to_string(),
            "op".to_string(),
            "ijkl".to_string(),
            "abcdefgh".to_string()
        ])
    );

    let ops = ('a'..='l')
        .into_iter()
        .map(|c| Append(c.to_string()))
        .chain(Some(Mark))
        .chain(Some(Append('m'.to_string())))
        .chain(Some(Append('n'.to_string())))
        .chain(Some(Authpath(11usize.into(), 0)))
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
}

pub fn check_checkpoint_rewind<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
    let mut t = new_tree(100);
    assert!(!t.rewind());

    let mut t = new_tree(100);
    t.checkpoint();
    assert!(t.rewind());

    let mut t = new_tree(100);
    t.append(&"a".to_string());
    t.checkpoint();
    t.append(&"b".to_string());
    t.mark();
    assert!(t.rewind());
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append(&"a".to_string());
    t.mark();
    t.checkpoint();
    assert!(t.rewind());

    let mut t = new_tree(100);
    t.append(&"a".to_string());
    t.checkpoint();
    t.mark();
    t.append(&"a".to_string());
    assert!(t.rewind());
    assert_eq!(Some(Position::from(0)), t.current_position());

    let mut t = new_tree(100);
    t.append(&"a".to_string());
    t.checkpoint();
    t.checkpoint();
    assert!(t.rewind());
    t.append(&"b".to_string());
    assert!(t.rewind());
    t.append(&"b".to_string());
    assert_eq!(t.root(0).unwrap(), "ab______________");
}

pub fn check_rewind_remove_mark<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
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
        let result = check_operations(new_tree(100), 4, sample);
        assert!(
            matches!(result, Ok(())),
            "Reference/Test mismatch at index {}: {:?}",
            i,
            result
        );
    }
}

pub fn check_witness_consistency<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
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
        let result = check_operations(new_tree(100), 4, sample);
        assert!(
            matches!(result, Ok(())),
            "Reference/Test mismatch at index {}: {:?}",
            i,
            result
        );
    }
}

pub fn check_rewind_remove_mark_consistency<T: Tree<String>, F: Fn(usize) -> T>(new_tree: F) {
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
        let result = check_operations(new_tree(100), 4, sample);
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
