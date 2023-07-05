use std::ops::Deref;
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

impl<'a, C: Clone, A: Clone, V: Clone> Node<C, &'a A, &'a V> {
    pub fn cloned(&self) -> Node<C, A, V> {
        match self {
            Node::Parent { ann, left, right } => Node::Parent {
                ann: (*ann).clone(),
                left: left.clone(),
                right: right.clone(),
            },
            Node::Leaf { value } => Node::Leaf {
                value: (*value).clone(),
            },
            Node::Nil => Node::Nil,
        }
    }
}

/// An immutable binary tree with each of its nodes tagged with an annotation value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree<A, V>(pub(crate) Node<Rc<Tree<A, V>>, A, V>);

impl<A, V> Deref for Tree<A, V> {
    type Target = Node<Rc<Tree<A, V>>, A, V>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, V> Tree<A, V> {
    pub fn empty() -> Self {
        Tree(Node::Nil)
    }

    pub fn leaf(value: V) -> Self {
        Tree(Node::Leaf { value })
    }

    pub fn parent(ann: A, left: Self, right: Self) -> Self {
        Tree(Node::Parent {
            ann,
            left: Rc::new(left),
            right: Rc::new(right),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_nil()
    }

    /// Replaces the annotation at the root of the tree, if the root is a `Node::Parent`; otherwise
    /// returns this tree unaltered.
    pub fn reannotate_root(self, ann: A) -> Self {
        Tree(self.0.reannotate(ann))
    }

    /// Returns `true` if no [`Node::Nil`] nodes are present in the tree, `false` otherwise.
    pub fn is_complete(&self) -> bool {
        match &self.0 {
            Node::Parent { left, right, .. } => {
                left.as_ref().is_complete() && right.as_ref().is_complete()
            }
            Node::Leaf { .. } => true,
            Node::Nil { .. } => false,
        }
    }

    /// Returns a vector of the addresses of [`Node::Nil`] subtree roots within this tree.
    ///
    /// The given address must correspond to the root of this tree, or this method will
    /// yield incorrect results or may panic.
    pub fn incomplete_nodes(&self, root_addr: Address) -> Vec<Address> {
        match &self.0 {
            Node::Parent { left, right, .. } => {
                // We should never construct parent nodes where both children are Nil.
                // While we could handle that here, if we encountered that case it would
                // be indicative of a programming error elsewhere and so we assert instead.
                assert!(!(left.0.is_nil() && right.0.is_nil()));
                let (left_root, right_root) = root_addr
                    .children()
                    .expect("A parent node cannot appear at level 0");

                let mut left_incomplete = left.incomplete_nodes(left_root);
                let mut right_incomplete = right.incomplete_nodes(right_root);
                left_incomplete.append(&mut right_incomplete);
                left_incomplete
            }
            Node::Leaf { .. } => vec![],
            Node::Nil => vec![root_addr],
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use incrementalmerkletree::{Address, Level};

    use super::{Node, Tree};

    pub(crate) fn str_leaf<A>(c: &str) -> Tree<A, String> {
        Tree(Node::Leaf {
            value: c.to_string(),
        })
    }

    pub(crate) fn nil<A, B>() -> Tree<A, B> {
        Tree::empty()
    }

    pub(crate) fn leaf<A, B>(value: B) -> Tree<A, B> {
        Tree::leaf(value)
    }

    pub(crate) fn parent<A: Default, B>(left: Tree<A, B>, right: Tree<A, B>) -> Tree<A, B> {
        Tree::parent(A::default(), left, right)
    }

    #[test]
    fn incomplete_nodes() {
        let t: Tree<(), String> = parent(nil(), str_leaf("a"));
        assert_eq!(
            t.incomplete_nodes(Address::from_parts(Level::from(1), 0)),
            vec![Address::from_parts(Level::from(0), 0)]
        );

        let t0 = parent(str_leaf("b"), t.clone());
        assert_eq!(
            t0.incomplete_nodes(Address::from_parts(Level::from(2), 1)),
            vec![Address::from_parts(Level::from(0), 6)]
        );

        let t1 = parent(nil(), t);
        assert_eq!(
            t1.incomplete_nodes(Address::from_parts(Level::from(2), 1)),
            vec![
                Address::from_parts(Level::from(1), 2),
                Address::from_parts(Level::from(0), 6)
            ]
        );
    }
}
