use core::fmt::Debug;
use core::ops::{BitAnd, BitOr, Deref, Not};
use either::Either;
use std::collections::BTreeSet;
use std::rc::Rc;

use incrementalmerkletree::{Address, Hashable, Level, Position, Retention};

/// A type for flags that determine when and how leaves can be pruned from a tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RetentionFlags(u8);

impl BitOr for RetentionFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        RetentionFlags(self.0 | rhs.0)
    }
}

impl BitAnd for RetentionFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        RetentionFlags(self.0 & rhs.0)
    }
}

/// An leaf with `EPHEMERAL` retention can be pruned as soon as we are certain that it is not part
/// of the witness for a leaf with `CHECKPOINT` or `MARKED` retention.
pub static EPHEMERAL: RetentionFlags = RetentionFlags(0b00000000);

/// A leaf with `CHECKPOINT` retention can be pruned when there are more than `max_checkpoints`
/// additional checkpoint leaves, if it is not also a marked leaf.
pub static CHECKPOINT: RetentionFlags = RetentionFlags(0b00000001);

/// A leaf with `MARKED` retention can be pruned only as a consequence of an explicit deletion
/// action.
pub static MARKED: RetentionFlags = RetentionFlags(0b00000010);

impl RetentionFlags {
    pub fn is_checkpoint(&self) -> bool {
        (*self & CHECKPOINT) == CHECKPOINT
    }

    pub fn is_marked(&self) -> bool {
        (*self & MARKED) == MARKED
    }
}

impl<'a, C> From<&'a Retention<C>> for RetentionFlags {
    fn from(retention: &'a Retention<C>) -> Self {
        match retention {
            Retention::Ephemeral => EPHEMERAL,
            Retention::Checkpoint { is_marked, .. } => {
                if *is_marked {
                    CHECKPOINT | MARKED
                } else {
                    CHECKPOINT
                }
            }
            Retention::Marked => MARKED,
        }
    }
}

impl<C> From<Retention<C>> for RetentionFlags {
    fn from(retention: Retention<C>) -> Self {
        RetentionFlags::from(&retention)
    }
}

/// A mask that may be used to unset one or more retention flags.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RetentionMask(u8);

impl Not for RetentionFlags {
    type Output = RetentionMask;

    fn not(self) -> Self::Output {
        RetentionMask(!self.0)
    }
}

impl BitAnd<RetentionMask> for RetentionFlags {
    type Output = Self;

    fn bitand(self, rhs: RetentionMask) -> Self {
        RetentionFlags(self.0 & rhs.0)
    }
}

/// A "pattern functor" for a single layer of a binary tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node<C, A, V> {
    /// A parent node in the tree, annotated with a value of type `A` and with left and right
    /// children of type `C`.
    Parent { ann: A, left: C, right: C },
    /// A node of the tree that contains a value (usually a hash, sometimes with additional
    /// metadata) and that has no children.
    ///
    /// Note that leaf nodes may appear at any position in the tree; i.e. they may contain computed
    /// subtree root values and not just level-0 leaves.
    Leaf { value: V },
    /// The empty tree; a subtree or leaf for which no information is available.
    Nil,
}

impl<C, A, V> Node<C, A, V> {
    /// Returns whether or not this is the `Nil` tree.
    ///
    /// This is useful for cases where the compiler can automatically dereference an `Rc`, where
    /// one would otherwise need additional ceremony to make an equality check.
    pub fn is_nil(&self) -> bool {
        matches!(self, Node::Nil)
    }

    /// Returns the contained leaf value, if this is a leaf node.
    pub fn leaf_value(&self) -> Option<&V> {
        match self {
            Node::Parent { .. } => None,
            Node::Leaf { value } => Some(value),
            Node::Nil { .. } => None,
        }
    }

    pub fn annotation(&self) -> Option<&A> {
        match self {
            Node::Parent { ann, .. } => Some(ann),
            Node::Leaf { .. } => None,
            Node::Nil => None,
        }
    }

    /// Replaces the annotation on this node, if it is a `Node::Parent`; otherwise
    /// returns this node unaltered.
    pub fn reannotate(self, ann: A) -> Self {
        match self {
            Node::Parent { left, right, .. } => Node::Parent { ann, left, right },
            other => other,
        }
    }
}

/// An F-algebra for use with [`Tree::reduce`] for determining whether a tree has any `Nil` nodes.
///
/// Returns `true` if no [`Node::Nil`] nodes are present in the tree.
pub fn is_complete<A, V>(node: Node<bool, A, V>) -> bool {
    match node {
        Node::Parent { left, right, .. } => left && right,
        Node::Leaf { .. } => true,
        Node::Nil { .. } => false,
    }
}

/// An F-algebra for use with [`Tree::try_reduce`] for determining whether a tree has any `MARKED` nodes.
///
/// `Tree::try_reduce` is preferred for this operation because it allows us to short-circuit as
/// soon as we find a marked node. Returns [`Either::Left(())`] if a marked node exists,
/// [`Either::Right(())`] otherwise.
pub fn contains_marked<A, V>(node: Node<(), A, (V, RetentionFlags)>) -> Either<(), ()> {
    match node {
        Node::Parent { .. } => Either::Right(()),
        Node::Leaf { value: (_, r) } => {
            if r.is_marked() {
                Either::Left(())
            } else {
                Either::Right(())
            }
        }
        Node::Nil { .. } => Either::Right(()),
    }
}

/// An immutable binary tree with each of its nodes tagged with an annotation value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree<A, V>(Node<Rc<Tree<A, V>>, A, V>);

impl<A, V> Deref for Tree<A, V> {
    type Target = Node<Rc<Tree<A, V>>, A, V>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, V> Tree<A, V> {
    /// Replaces the annotation at the root of the tree, if the root is a `Node::Parent`; otherwise
    /// returns this tree unaltered.
    pub fn reannotate_root(self, ann: A) -> Tree<A, V> {
        Tree(self.0.reannotate(ann))
    }

    /// Returns a vector of the addresses of [`Node::Nil`] subtree roots within this tree.
    ///
    /// The given address must correspond to the root of this tree, or this method will
    /// yield incorrect results or may panic.
    pub fn incomplete(&self, root_addr: Address) -> Vec<Address> {
        match &self.0 {
            Node::Parent { left, right, .. } => {
                // We should never construct parent nodes where both children are Nil.
                // While we could handle that here, if we encountered that case it would
                // be indicative of a programming error elsewhere and so we assert instead.
                assert!(!(left.0.is_nil() && right.0.is_nil()));
                let (left_root, right_root) = root_addr
                    .children()
                    .expect("A parent node cannot appear at level 0");

                let mut left_incomplete = left.incomplete(left_root);
                let mut right_incomplete = right.incomplete(right_root);
                left_incomplete.append(&mut right_incomplete);
                left_incomplete
            }
            Node::Leaf { .. } => vec![],
            Node::Nil => vec![root_addr],
        }
    }
}

impl<A: Clone, V: Clone> Tree<A, V> {
    /// Folds over the tree from leaf to root with the given function.
    ///
    /// See [`is_complete`] for an example of a function that can be used with this method.
    /// This operation will visit every node of the tree. See [`try_reduce`] for a variant
    /// that can perform a depth-first, left-to-right traversal with the option to
    /// short-circuit.
    pub fn reduce<B, F: Fn(Node<B, A, V>) -> B>(&self, alg: &F) -> B {
        match &self.0 {
            Node::Parent { ann, left, right } => {
                let left_result = left.reduce(alg);
                let right_result = right.reduce(alg);
                alg(Node::Parent {
                    ann: ann.clone(),
                    left: left_result,
                    right: right_result,
                })
            }
            Node::Leaf { value } => alg(Node::Leaf {
                value: value.clone(),
            }),
            Node::Nil => alg(Node::Nil),
        }
    }

    /// Folds over the tree from leaf to root with the given function.
    ///
    /// This performs a left-to-right, depth-first traversal that halts on the first
    /// [`Either::Left`] result, or builds an [`Either::Right`] from the results computed at every
    /// node.
    pub fn try_reduce<L, R, F: Fn(Node<R, A, V>) -> Either<L, R>>(&self, alg: &F) -> Either<L, R> {
        match &self.0 {
            Node::Parent { ann, left, right } => left.try_reduce(alg).right_and_then(|l_value| {
                right.try_reduce(alg).right_and_then(move |r_value| {
                    alg(Node::Parent {
                        ann: ann.clone(),
                        left: l_value,
                        right: r_value,
                    })
                })
            }),
            Node::Leaf { value } => alg(Node::Leaf {
                value: value.clone(),
            }),
            Node::Nil => alg(Node::Nil),
        }
    }
}

type PrunableTree<H> = Tree<Option<Rc<H>>, (H, RetentionFlags)>;

impl<H: Hashable + Clone + PartialEq> PrunableTree<H> {
    /// Returns the the value if this is a leaf.
    pub fn leaf_value(&self) -> Option<&H> {
        self.0.leaf_value().map(|(h, _)| h)
    }

    /// Returns the cached root value with which the tree has been annotated for this node if it is
    /// available, otherwise return the value if this is a leaf.
    pub fn node_value(&self) -> Option<&H> {
        self.0.annotation().map_or_else(
            || self.leaf_value(),
            |rc_opt| rc_opt.as_ref().map(|rc| rc.as_ref()),
        )
    }

    /// Returns whether or not this tree is a leaf with `Marked` retention.
    pub fn is_marked_leaf(&self) -> bool {
        self.0
            .leaf_value()
            .map_or(false, |(_, retention)| retention.is_marked())
    }

    /// Returns the Merkle root of this tree, given the address of the root node, or
    /// a vector of the addresses of `Nil` nodes that inhibited the computation of
    /// such a root.
    ///
    /// ### Parameters:
    /// * `truncate_at` An inclusive lower bound on positions in the tree beyond which all leaf
    ///    values will be treated as `Nil`.
    pub fn root_hash(&self, root_addr: Address, truncate_at: Position) -> Result<H, Vec<Address>> {
        if truncate_at <= root_addr.position_range_start() {
            // we are in the part of the tree where we're generating empty roots,
            // so no need to inspect the tree
            Ok(H::empty_root(root_addr.level()))
        } else {
            match self {
                Tree(Node::Parent { ann, left, right }) => ann
                    .as_ref()
                    .filter(|_| truncate_at >= root_addr.position_range_end())
                    .map_or_else(
                        || {
                            // Compute the roots of the left and right children and hash them
                            // together.
                            let (l_addr, r_addr) = root_addr.children().unwrap();
                            accumulate_result_with(
                                left.root_hash(l_addr, truncate_at),
                                right.root_hash(r_addr, truncate_at),
                                |left_root, right_root| {
                                    H::combine(l_addr.level(), &left_root, &right_root)
                                },
                            )
                        },
                        |rc| {
                            // Since we have an annotation on the root, and we are not truncating
                            // within this subtree, we can just use the cached value.
                            Ok(rc.as_ref().clone())
                        },
                    ),
                Tree(Node::Leaf { value }) => {
                    if truncate_at >= root_addr.position_range_end() {
                        // no truncation of this leaf is necessary, just use it
                        Ok(value.0.clone())
                    } else {
                        // we have a leaf value that is a subtree root created by hashing together
                        // the roots of child subtrees, but truncation would require that that leaf
                        // value be "split" into its constituent parts, which we can't do so we
                        // return an error
                        Err(vec![root_addr])
                    }
                }
                Tree(Node::Nil) => Err(vec![root_addr]),
            }
        }
    }

    /// Returns a vector of the positions of [`Node::Leaf`] values in the tree having [`MARKED`]
    /// retention.
    ///
    /// Computing the set of marked positions requires a full traversal of the tree, and so should
    /// be considered to be a somewhat expensive operation.
    pub fn marked_positions(&self, root_addr: Address) -> BTreeSet<Position> {
        match &self.0 {
            Node::Parent { left, right, .. } => {
                // We should never construct parent nodes where both children are Nil.
                // While we could handle that here, if we encountered that case it would
                // be indicative of a programming error elsewhere and so we assert instead.
                assert!(!(left.0.is_nil() && right.0.is_nil()));
                let (left_root, right_root) = root_addr
                    .children()
                    .expect("A parent node cannot appear at level 0");

                let mut left_incomplete = left.marked_positions(left_root);
                let mut right_incomplete = right.marked_positions(right_root);
                left_incomplete.append(&mut right_incomplete);
                left_incomplete
            }
            Node::Leaf {
                value: (_, retention),
            } => {
                let mut result = BTreeSet::new();
                if root_addr.level() == 0.into() && retention.is_marked() {
                    result.insert(Position::from(root_addr.index()));
                }
                result
            }
            Node::Nil => BTreeSet::new(),
        }
    }

    /// Prunes the tree by hashing together ephemeral sibling nodes.
    ///
    /// `level` must be the level of the root of the node being pruned.
    pub fn prune(self, level: Level) -> Self {
        match self {
            Tree(Node::Parent { ann, left, right }) => Tree::unite(
                level,
                ann,
                left.as_ref().clone().prune(level - 1),
                right.as_ref().clone().prune(level - 1),
            ),
            other => other,
        }
    }

    /// Merge two subtrees having the same root address.
    ///
    /// The merge operation is checked to be strictly additive and returns an error if merging
    /// would cause information loss or if a conflict between root hashes occurs at a node. The
    /// returned error contains the address of the node where such a conflict occurred.
    pub fn merge_checked(self, root_addr: Address, other: Self) -> Result<Self, Address> {
        #[allow(clippy::type_complexity)]
        fn go<H: Hashable + Clone + PartialEq>(
            addr: Address,
            t0: PrunableTree<H>,
            t1: PrunableTree<H>,
        ) -> Result<PrunableTree<H>, Address> {
            // Require that any roots the we compute will not be default-filled by picking
            // a starting valid fill point that is outside the range of leaf positions.
            let no_default_fill = addr.position_range_end();
            match (t0, t1) {
                (Tree(Node::Nil), other) => Ok(other),
                (other, Tree(Node::Nil)) => Ok(other),
                (Tree(Node::Leaf { value: vl }), Tree(Node::Leaf { value: vr })) => {
                    if vl == vr {
                        Ok(Tree(Node::Leaf { value: vl }))
                    } else {
                        Err(addr)
                    }
                }
                (Tree(Node::Leaf { value }), parent) => {
                    // `parent` is statically known to be a `Node::Parent`
                    if parent
                        .root_hash(addr, no_default_fill)
                        .iter()
                        .all(|r| r == &value.0)
                    {
                        Ok(parent.reannotate_root(Some(Rc::new(value.0))))
                    } else {
                        Err(addr)
                    }
                }
                (parent, Tree(Node::Leaf { value })) => {
                    // `parent` is statically known to be a `Node::Parent`
                    if parent
                        .root_hash(addr, no_default_fill)
                        .iter()
                        .all(|r| r == &value.0)
                    {
                        Ok(parent.reannotate_root(Some(Rc::new(value.0))))
                    } else {
                        Err(addr)
                    }
                }
                (lparent, rparent) => {
                    let lroot = lparent.root_hash(addr, no_default_fill).ok();
                    let rroot = rparent.root_hash(addr, no_default_fill).ok();
                    // If both parents share the same root hash (or if one of them is absent),
                    // they can be merged
                    if lroot.zip(rroot).iter().all(|(l, r)| l == r) {
                        // using `if let` here to bind variables; we need to borrow the trees for
                        // root hash calculation but binding the children of the parent node
                        // interferes with binding a reference to the parent.
                        if let (
                            Tree(Node::Parent {
                                ann: lann,
                                left: ll,
                                right: lr,
                            }),
                            Tree(Node::Parent {
                                ann: rann,
                                left: rl,
                                right: rr,
                            }),
                        ) = (lparent, rparent)
                        {
                            let (l_addr, r_addr) = addr.children().unwrap();
                            Ok(Tree::unite(
                                addr.level() - 1,
                                lann.or(rann),
                                go(l_addr, ll.as_ref().clone(), rl.as_ref().clone())?,
                                go(r_addr, lr.as_ref().clone(), rr.as_ref().clone())?,
                            ))
                        } else {
                            unreachable!()
                        }
                    } else {
                        Err(addr)
                    }
                }
            }
        }

        go(root_addr, self, other)
    }

    /// Unite two nodes by either constructing a new parent node, or, if both nodes are ephemeral
    /// leaves or Nil, constructing a replacement root by hashing leaf values together (or a
    /// replacement `Nil` value).
    ///
    /// `level` must be the level of the two nodes that are being joined.
    fn unite(level: Level, ann: Option<Rc<H>>, left: Self, right: Self) -> Self {
        match (left, right) {
            (Tree(Node::Nil), Tree(Node::Nil)) => Tree(Node::Nil),
            (Tree(Node::Leaf { value: lv }), Tree(Node::Leaf { value: rv }))
                // we can prune right-hand leaves that are not marked; if a leaf
                // is a checkpoint then that information will be propagated to
                // the replacement leaf
                if lv.1 == EPHEMERAL && (rv.1 & MARKED) == EPHEMERAL =>
            {
                Tree(
                    Node::Leaf {
                        value: (H::combine(level, &lv.0, &rv.0), rv.1),
                    },
                )
            }
            (left, right) => Tree(
                Node::Parent {
                    ann,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
            ),
        }
    }
}

// We need an applicative functor for Result for this function so that we can correctly
// accumulate errors, but we don't have one so we just write a special- cased version here.
fn accumulate_result_with<A, B, C>(
    left: Result<A, Vec<Address>>,
    right: Result<B, Vec<Address>>,
    combine_success: impl FnOnce(A, B) -> C,
) -> Result<C, Vec<Address>> {
    match (left, right) {
        (Ok(a), Ok(b)) => Ok(combine_success(a, b)),
        (Err(mut xs), Err(mut ys)) => {
            xs.append(&mut ys);
            Err(xs)
        }
        (Ok(_), Err(xs)) => Err(xs),
        (Err(xs), Ok(_)) => Err(xs),
    }
}

#[cfg(any(bench, test, feature = "test-dependencies"))]
pub mod testing {
    use super::*;
    use incrementalmerkletree::Hashable;
    use proptest::prelude::*;
    use proptest::sample::select;

    pub fn arb_retention_flags() -> impl Strategy<Value = RetentionFlags> {
        select(vec![EPHEMERAL, CHECKPOINT, MARKED, MARKED | CHECKPOINT])
    }

    pub fn arb_tree<A: Strategy + Clone + 'static, V: Strategy + Clone + 'static>(
        arb_annotation: A,
        arb_leaf: V,
        depth: u32,
        size: u32,
    ) -> impl Strategy<Value = Tree<A::Value, V::Value>>
    where
        A::Value: Clone + 'static,
        V::Value: Hashable + Clone + 'static,
    {
        let leaf = prop_oneof![
            Just(Tree(Node::Nil)),
            arb_leaf.prop_map(|value| Tree(Node::Leaf { value }))
        ];

        leaf.prop_recursive(depth, size, 2, move |inner| {
            (arb_annotation.clone(), inner.clone(), inner).prop_map(|(ann, left, right)| {
                Tree(if left.is_nil() && right.is_nil() {
                    Node::Nil
                } else {
                    Node::Parent {
                        ann,
                        left: Rc::new(left),
                        right: Rc::new(right),
                    }
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Node, PrunableTree, Tree, EPHEMERAL, MARKED};
    use incrementalmerkletree::{Address, Level, Position};
    use std::collections::BTreeSet;
    use std::rc::Rc;

    #[test]
    fn tree_incomplete() {
        let t = Tree(Node::Parent {
            ann: (),
            left: Rc::new(Tree(Node::Nil)),
            right: Rc::new(Tree(Node::Leaf { value: "a" })),
        });
        assert_eq!(
            t.incomplete(Address::from_parts(Level::from(1), 0)),
            vec![Address::from_parts(Level::from(0), 0)]
        );

        let t0 = Tree(Node::Parent {
            ann: (),
            left: Rc::new(Tree(Node::Leaf { value: "b" })),
            right: Rc::new(t.clone()),
        });
        assert_eq!(
            t0.incomplete(Address::from_parts(Level::from(2), 1)),
            vec![Address::from_parts(Level::from(0), 6)]
        );

        let t1 = Tree(Node::Parent {
            ann: (),
            left: Rc::new(Tree(Node::Nil)),
            right: Rc::new(t),
        });
        assert_eq!(
            t1.incomplete(Address::from_parts(Level::from(2), 1)),
            vec![
                Address::from_parts(Level::from(1), 2),
                Address::from_parts(Level::from(0), 6)
            ]
        );
    }

    #[test]
    fn tree_root() {
        let t: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Leaf {
                value: ("a".to_string(), EPHEMERAL),
            })),
            right: Rc::new(Tree(Node::Leaf {
                value: ("b".to_string(), EPHEMERAL),
            })),
        });
        assert_eq!(
            t.root_hash(Address::from_parts(Level::from(1), 0), Position::from(2)),
            Ok("ab".to_string())
        );

        let t0: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Nil)),
            right: Rc::new(t.clone()),
        });
        assert_eq!(
            t0.root_hash(Address::from_parts(Level::from(2), 0), Position::from(4)),
            Err(vec![Address::from_parts(Level::from(1), 0)])
        );

        // Check root computation with truncation
        let t1: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(t),
            right: Rc::new(Tree(Node::Nil)),
        });
        assert_eq!(
            t1.root_hash(Address::from_parts(Level::from(2), 0), Position::from(2)),
            Ok("ab__".to_string())
        );
        assert_eq!(
            t1.root_hash(Address::from_parts(Level::from(2), 0), Position::from(3)),
            Err(vec![Address::from_parts(Level::from(1), 1)])
        );
    }

    #[test]
    fn tree_marked_positions() {
        let t: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Leaf {
                value: ("a".to_string(), EPHEMERAL),
            })),
            right: Rc::new(Tree(Node::Leaf {
                value: ("b".to_string(), MARKED),
            })),
        });
        assert_eq!(
            t.marked_positions(Address::from_parts(Level::from(1), 0)),
            BTreeSet::from([Position::from(1)])
        );

        let t0: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(t.clone()),
            right: Rc::new(t),
        });
        assert_eq!(
            t0.marked_positions(Address::from_parts(Level::from(2), 1)),
            BTreeSet::from([Position::from(5), Position::from(7)])
        );
    }

    #[test]
    fn tree_prune() {
        let t: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Leaf {
                value: ("a".to_string(), EPHEMERAL),
            })),
            right: Rc::new(Tree(Node::Leaf {
                value: ("b".to_string(), EPHEMERAL),
            })),
        });

        assert_eq!(
            t.clone().prune(Level::from(1)),
            Tree(Node::Leaf {
                value: ("ab".to_string(), EPHEMERAL)
            })
        );

        let t0: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Leaf {
                value: ("c".to_string(), MARKED),
            })),
            right: Rc::new(t),
        });
        assert_eq!(
            t0.prune(Level::from(2)),
            Tree(Node::Parent {
                ann: None,
                left: Rc::new(Tree(Node::Leaf {
                    value: ("c".to_string(), MARKED),
                },)),
                right: Rc::new(Tree(Node::Leaf {
                    value: ("ab".to_string(), EPHEMERAL)
                }))
            },)
        );
    }

    #[test]
    fn tree_merge_checked() {
        let t0: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Leaf {
                value: ("a".to_string(), EPHEMERAL),
            })),
            right: Rc::new(Tree(Node::Nil)),
        });

        let t1: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Nil)),
            right: Rc::new(Tree(Node::Leaf {
                value: ("b".to_string(), EPHEMERAL),
            })),
        });

        assert_eq!(
            t0.clone()
                .merge_checked(Address::from_parts(1.into(), 0), t1.clone()),
            Ok(Tree(Node::Leaf {
                value: ("ab".to_string(), EPHEMERAL)
            }))
        );

        let t2: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(Tree(Node::Leaf {
                value: ("c".to_string(), EPHEMERAL),
            })),
            right: Rc::new(Tree(Node::Nil)),
        });
        assert_eq!(
            t0.clone()
                .merge_checked(Address::from_parts(1.into(), 0), t2.clone()),
            Err(Address::from_parts(0.into(), 0))
        );

        let t3: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(t0),
            right: Rc::new(t2),
        });
        let t4: PrunableTree<String> = Tree(Node::Parent {
            ann: None,
            left: Rc::new(t1.clone()),
            right: Rc::new(t1),
        });

        assert_eq!(
            t3.merge_checked(Address::from_parts(2.into(), 0), t4),
            Ok(Tree(Node::Leaf {
                value: ("abcb".to_string(), EPHEMERAL)
            }))
        );
    }
}
