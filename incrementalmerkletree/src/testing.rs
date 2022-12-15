use proptest::prelude::*;
use std::collections::BTreeSet;

use crate::{Hashable, Level, Position};

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
