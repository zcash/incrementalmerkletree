use core::fmt::Debug;
use core::ops::Deref;
use either::Either;
use std::rc::Rc;

use incrementalmerkletree::Address;

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

#[cfg(any(bench, test, feature = "test-dependencies"))]
pub mod testing {
    use super::*;
    use incrementalmerkletree::Hashable;
    use proptest::prelude::*;

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
    use crate::{Node, Tree};
    use incrementalmerkletree::{Address, Level};
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
}
