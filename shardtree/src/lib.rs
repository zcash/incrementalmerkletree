use bitflags::bitflags;
use core::convert::TryFrom;
use core::fmt::{self, Debug};
use core::ops::{Deref, Range};
use either::Either;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::rc::Rc;

use incrementalmerkletree::{
    frontier::NonEmptyFrontier, Address, Hashable, Level, MerklePath, Position, Retention,
};

#[cfg(feature = "legacy-api")]
use incrementalmerkletree::witness::IncrementalWitness;

bitflags! {
    pub struct RetentionFlags: u8 {
        /// An leaf with `EPHEMERAL` retention can be pruned as soon as we are certain that it is not part
        /// of the witness for a leaf with [`CHECKPOINT`] or [`MARKED`] retention.
        ///
        /// [`CHECKPOINT`]: RetentionFlags::CHECKPOINT
        /// [`MARKED`]: RetentionFlags::MARKED
        const EPHEMERAL = 0b00000000;

        /// A leaf with `CHECKPOINT` retention can be pruned when there are more than `max_checkpoints`
        /// additional checkpoint leaves, if it is not also a marked leaf.
        const CHECKPOINT = 0b00000001;

        /// A leaf with `MARKED` retention can be pruned only as a consequence of an explicit deletion
        /// action.
        const MARKED = 0b00000010;
    }
}

impl RetentionFlags {
    pub fn is_checkpoint(&self) -> bool {
        (*self & RetentionFlags::CHECKPOINT) == RetentionFlags::CHECKPOINT
    }

    pub fn is_marked(&self) -> bool {
        (*self & RetentionFlags::MARKED) == RetentionFlags::MARKED
    }
}

impl<'a, C> From<&'a Retention<C>> for RetentionFlags {
    fn from(retention: &'a Retention<C>) -> Self {
        match retention {
            Retention::Ephemeral => RetentionFlags::EPHEMERAL,
            Retention::Checkpoint { is_marked, .. } => {
                if *is_marked {
                    RetentionFlags::CHECKPOINT | RetentionFlags::MARKED
                } else {
                    RetentionFlags::CHECKPOINT
                }
            }
            Retention::Marked => RetentionFlags::MARKED,
        }
    }
}

impl<C> From<Retention<C>> for RetentionFlags {
    fn from(retention: Retention<C>) -> Self {
        RetentionFlags::from(&retention)
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
pub struct Tree<A, V>(Node<Rc<Tree<A, V>>, A, V>);

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

pub type PrunableTree<H> = Tree<Option<Rc<H>>, (H, RetentionFlags)>;

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

    /// Determines whether a tree has any [`Retention::Marked`] nodes.
    pub fn contains_marked(&self) -> bool {
        match &self.0 {
            Node::Parent { left, right, .. } => left.contains_marked() || right.contains_marked(),
            Node::Leaf { value: (_, r) } => r.is_marked(),
            Node::Nil => false,
        }
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

    /// Returns a vector of the positions of [`Node::Leaf`] values in the tree having
    /// [`MARKED`](RetentionFlags::MARKED) retention.
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
                (Tree(Node::Nil), other) | (other, Tree(Node::Nil)) => Ok(other),
                (Tree(Node::Leaf { value: vl }), Tree(Node::Leaf { value: vr })) => {
                    if vl == vr {
                        Ok(Tree(Node::Leaf { value: vl }))
                    } else {
                        Err(addr)
                    }
                }
                (Tree(Node::Leaf { value }), parent @ Tree(Node::Parent { .. }))
                | (parent @ Tree(Node::Parent { .. }), Tree(Node::Leaf { value })) => {
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
                if lv.1 == RetentionFlags::EPHEMERAL && (rv.1 & RetentionFlags::MARKED) == RetentionFlags::EPHEMERAL =>
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

/// A binary Merkle tree with its root at the given address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocatedTree<A, V> {
    root_addr: Address,
    root: Tree<A, V>,
}

impl<A, V> LocatedTree<A, V> {
    /// Constructs a new LocatedTree from its constituent parts
    pub fn from_parts(root_addr: Address, root: Tree<A, V>) -> Self {
        LocatedTree { root_addr, root }
    }

    /// Returns the root address of this tree.
    pub fn root_addr(&self) -> Address {
        self.root_addr
    }

    /// Returns a reference to the root of the tree.
    pub fn root(&self) -> &Tree<A, V> {
        &self.root
    }

    /// Returns a new [`LocatedTree`] with the provided value replacing the annotation of its root
    /// node, if that root node is a `Node::Parent`. Otherwise .
    pub fn reannotate_root(self, value: A) -> Self {
        LocatedTree {
            root_addr: self.root_addr,
            root: self.root.reannotate_root(value),
        }
    }

    /// Returns the set of incomplete subtree roots contained within this tree, ordered by
    /// increasing position.
    pub fn incomplete_nodes(&self) -> Vec<Address> {
        self.root.incomplete_nodes(self.root_addr)
    }

    /// Returns the maximum position at which a non-Nil leaf has been observed in the tree.
    ///
    /// Note that no actual leaf value may exist at this position, as it may have previously been
    /// pruned.
    pub fn max_position(&self) -> Option<Position> {
        fn go<A, V>(addr: Address, root: &Tree<A, V>) -> Option<Position> {
            match &root.0 {
                Node::Nil => None,
                Node::Leaf { .. } => Some(addr.position_range_end() - 1),
                Node::Parent { left, right, .. } => {
                    let (l_addr, r_addr) = addr.children().unwrap();
                    go(r_addr, right.as_ref()).or_else(|| go(l_addr, left.as_ref()))
                }
            }
        }

        go(self.root_addr, &self.root)
    }

    /// Returns the value at the specified position, if any.
    pub fn value_at_position(&self, position: Position) -> Option<&V> {
        fn go<A, V>(pos: Position, addr: Address, root: &Tree<A, V>) -> Option<&V> {
            match &root.0 {
                Node::Parent { left, right, .. } => {
                    let (l_addr, r_addr) = addr.children().unwrap();
                    if l_addr.position_range().contains(&pos) {
                        go(pos, l_addr, left)
                    } else {
                        go(pos, r_addr, right)
                    }
                }
                Node::Leaf { value } if addr.level() == Level::from(0) => Some(value),
                _ => None,
            }
        }

        if self.root_addr.position_range().contains(&position) {
            go(position, self.root_addr, &self.root)
        } else {
            None
        }
    }
}

impl<A: Default + Clone, V: Clone> LocatedTree<A, V> {
    /// Constructs a new empty tree with its root at the provided address.
    pub fn empty(root_addr: Address) -> Self {
        Self {
            root_addr,
            root: Tree(Node::Nil),
        }
    }

    /// Constructs a new tree consisting of a single leaf with the provided value, and the
    /// specified root address.
    pub fn with_root_value(root_addr: Address, value: V) -> Self {
        Self {
            root_addr,
            root: Tree(Node::Leaf { value }),
        }
    }

    /// Traverses this tree to find the child node at the specified address and returns it.
    ///
    /// Returns `None` if the specified address is not a descendant of this tree's root address, or
    /// if the tree is terminated by a [`Node::Nil`] or leaf node before the specified address can
    /// be reached.
    pub fn subtree(&self, addr: Address) -> Option<Self> {
        fn go<A: Clone, V: Clone>(
            root_addr: Address,
            root: &Tree<A, V>,
            addr: Address,
        ) -> Option<LocatedTree<A, V>> {
            if root_addr == addr {
                Some(LocatedTree {
                    root_addr,
                    root: root.clone(),
                })
            } else {
                match &root.0 {
                    Node::Parent { left, right, .. } => {
                        let (l_addr, r_addr) = root_addr.children().unwrap();
                        if l_addr.contains(&addr) {
                            go(l_addr, left.as_ref(), addr)
                        } else {
                            go(r_addr, right.as_ref(), addr)
                        }
                    }
                    _ => None,
                }
            }
        }

        if self.root_addr.contains(&addr) {
            go(self.root_addr, &self.root, addr)
        } else {
            None
        }
    }

    /// Decomposes this tree into the vector of its subtrees having height `level + 1`.
    ///
    /// If this root address of this tree is lower down in the tree than the level specified,
    /// the entire tree is returned as the sole element of the result vector.
    pub fn decompose_to_level(self, level: Level) -> Vec<Self> {
        fn go<A: Clone, V: Clone>(
            level: Level,
            root_addr: Address,
            root: Tree<A, V>,
        ) -> Vec<LocatedTree<A, V>> {
            if root_addr.level() == level {
                vec![LocatedTree { root_addr, root }]
            } else {
                match root.0 {
                    Node::Parent { left, right, .. } => {
                        let (l_addr, r_addr) = root_addr.children().unwrap();
                        let mut l_decomposed = go(
                            level,
                            l_addr,
                            Rc::try_unwrap(left).unwrap_or_else(|rc| (*rc).clone()),
                        );
                        let mut r_decomposed = go(
                            level,
                            r_addr,
                            Rc::try_unwrap(right).unwrap_or_else(|rc| (*rc).clone()),
                        );
                        l_decomposed.append(&mut r_decomposed);
                        l_decomposed
                    }
                    _ => vec![],
                }
            }
        }

        if level >= self.root_addr.level() {
            vec![self]
        } else {
            go(level, self.root_addr, self.root)
        }
    }
}

pub type LocatedPrunableTree<H> = LocatedTree<Option<Rc<H>>, (H, RetentionFlags)>;

/// A data structure describing the nature of a [`Node::Nil`] node in the tree that was introduced
/// as the consequence of an insertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncompleteAt {
    /// The address of the empty node.
    pub address: Address,
    /// A flag identifying whether or not the missing node is required in order to construct a
    /// witness for a node with [`MARKED`] retention.
    ///
    /// [`MARKED`]: RetentionFlags::MARKED
    pub required_for_witness: bool,
}

/// A type for the result of a batch insertion operation.
///
/// This result type contains the newly constructed tree, the addresses any new incomplete internal
/// nodes within that tree that were introduced as a consequence of that insertion, and the
/// remainder of the iterator that provided the inserted values.
#[derive(Debug)]
pub struct BatchInsertionResult<H, C: Ord, I: Iterator<Item = (H, Retention<C>)>> {
    /// The updated tree after all insertions have been performed.
    pub subtree: LocatedPrunableTree<H>,
    /// A flag identifying whether the constructed subtree contains a marked node.
    pub contains_marked: bool,
    /// The vector of addresses of [`Node::Nil`] nodes that were inserted into the tree as part of
    /// the insertion operation, for nodes that are required in order to construct a witness for
    /// each inserted leaf with [`Retention::Marked`] retention.
    pub incomplete: Vec<IncompleteAt>,
    /// The maximum position at which a leaf was inserted.
    pub max_insert_position: Option<Position>,
    /// The positions of all leaves with [`Retention::Checkpoint`] retention that were inserted.
    pub checkpoints: BTreeMap<C, Position>,
    /// The unconsumed remainder of the iterator from which leaves were inserted, if the tree
    /// was completely filled before the iterator was fully consumed.
    pub remainder: I,
}

/// An error prevented the insertion of values into the subtree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InsertionError {
    /// The caller attempted to insert a subtree into a tree that does not contain
    /// the subtree's root address.
    NotContained(Address),
    /// The start of the range of positions provided for insertion is not included
    /// in the range of positions within this subtree.
    OutOfRange(Position, Range<Position>),
    /// An existing root hash conflicts with the root hash of a node being inserted.
    Conflict(Address),
    /// An out-of-order checkpoint was detected
    ///
    /// Checkpoint identifiers must be in nondecreasing order relative to tree positions.
    CheckpointOutOfOrder,
    /// An append operation has exceeded the capacity of the tree.
    TreeFull,
    /// An input data structure had malformed data when attempting to insert a value
    /// at the given address
    InputMalformed(Address),
}

impl fmt::Display for InsertionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            InsertionError::NotContained(addr) => {
                write!(f, "Tree does not contain a root at address {:?}", addr)
            }
            InsertionError::OutOfRange(p, r) => {
                write!(
                    f,
                    "Attempted insertion point {:?} is not in range {:?}",
                    p, r
                )
            }
            InsertionError::Conflict(addr) => write!(
                f,
                "Inserted root conflicts with existing root at address {:?}",
                addr
            ),
            InsertionError::CheckpointOutOfOrder => {
                write!(f, "Cannot append out-of-order checkpoint identifier.")
            }
            InsertionError::TreeFull => write!(f, "Note commitment tree is full."),
            InsertionError::InputMalformed(addr) => {
                write!(f, "Input malformed for insertion at address {:?}", addr)
            }
        }
    }
}

impl std::error::Error for InsertionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Errors that may be returned in the process of querying a [`ShardTree`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    /// The caller attempted to query the value at an address within a tree that does not contain
    /// that address.
    NotContained(Address),
    /// A leaf required by a given checkpoint has been pruned, or is otherwise not accessible in
    /// the tree.
    CheckpointPruned,
    /// It is not possible to compute a root for one or more subtrees because they contain
    /// [`Node::Nil`] values at positions that cannot be replaced with default hashes.
    TreeIncomplete(Vec<Address>),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            QueryError::NotContained(addr) => {
                write!(f, "Tree does not contain a root at address {:?}", addr)
            }
            QueryError::CheckpointPruned => {
                write!(
                    f,
                    "The leaf corresponding to the requested checkpoint is not present in the tree."
                )
            }
            QueryError::TreeIncomplete(addrs) => {
                write!(
                    f,
                    "Unable to compute root; missing values for nodes {:?}",
                    addrs
                )
            }
        }
    }
}

impl std::error::Error for QueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Operations on [`LocatedTree`]s that are annotated with Merkle hashes.
impl<H: Hashable + Clone + PartialEq> LocatedPrunableTree<H> {
    /// Computes the root hash of this tree, truncated to the given position.
    ///
    /// If the tree contains any [`Node::Nil`] nodes corresponding to positions less than
    /// `truncate_at`, this will return an error containing the addresses of those nodes within the
    /// tree.
    pub fn root_hash(&self, truncate_at: Position) -> Result<H, Vec<Address>> {
        self.root.root_hash(self.root_addr, truncate_at)
    }

    /// Compute the root hash of this subtree, filling empty nodes along the rightmost path of the
    /// subtree with the empty root value for the given level.
    ///
    /// This should only be used for computing roots when it is known that no successor trees
    /// exist.
    ///
    /// If the tree contains any [`Node::Nil`] nodes that are to the left of filled nodes in the
    /// tree, this will return an error containing the addresses of those nodes.
    pub fn right_filled_root(&self) -> Result<H, Vec<Address>> {
        self.root_hash(
            self.max_position()
                .map_or_else(|| self.root_addr.position_range_start(), |pos| pos + 1),
        )
    }

    /// Returns the positions of marked leaves in the tree.
    pub fn marked_positions(&self) -> BTreeSet<Position> {
        fn go<H: Hashable + Clone + PartialEq>(
            root_addr: Address,
            root: &PrunableTree<H>,
            acc: &mut BTreeSet<Position>,
        ) {
            match &root.0 {
                Node::Parent { left, right, .. } => {
                    let (l_addr, r_addr) = root_addr.children().unwrap();
                    go(l_addr, left.as_ref(), acc);
                    go(r_addr, right.as_ref(), acc);
                }
                Node::Leaf { value } => {
                    if value.1.is_marked() && root_addr.level() == 0.into() {
                        acc.insert(Position::from(root_addr.index()));
                    }
                }
                _ => {}
            }
        }

        let mut result = BTreeSet::new();
        go(self.root_addr, &self.root, &mut result);
        result
    }

    /// Compute the witness for the leaf at the specified position.
    ///
    /// This tree will be truncated to the `truncate_at` position, and then empty roots
    /// corresponding to later positions will be filled by the [`Hashable::empty_root`]
    /// implementation for `H`.
    ///
    /// Returns either the witness for the leaf at the specified position, or an error that
    /// describes the causes of failure.
    pub fn witness(&self, position: Position, truncate_at: Position) -> Result<Vec<H>, QueryError> {
        // traverse down to the desired leaf position, and then construct
        // the authentication path on the way back up.
        fn go<H: Hashable + Clone + PartialEq>(
            root: &PrunableTree<H>,
            root_addr: Address,
            position: Position,
            truncate_at: Position,
        ) -> Result<Vec<H>, Vec<Address>> {
            match &root.0 {
                Node::Parent { left, right, .. } => {
                    let (l_addr, r_addr) = root_addr.children().unwrap();
                    if root_addr.level() > 1.into() {
                        let r_start = r_addr.position_range_start();
                        if position < r_start {
                            accumulate_result_with(
                                go(left.as_ref(), l_addr, position, truncate_at),
                                right.as_ref().root_hash(r_addr, truncate_at),
                                |mut witness, sibling_root| {
                                    witness.push(sibling_root);
                                    witness
                                },
                            )
                        } else {
                            // if the position we're witnessing is down the right-hand branch then
                            // we always set the truncation bound outside the range of leaves on the
                            // left, because we don't allow any empty nodes to the left
                            accumulate_result_with(
                                left.as_ref().root_hash(l_addr, r_start),
                                go(right.as_ref(), r_addr, position, truncate_at),
                                |sibling_root, mut witness| {
                                    witness.push(sibling_root);
                                    witness
                                },
                            )
                        }
                    } else {
                        // we handle the level 0 leaves here by adding the sibling of our desired
                        // leaf to the witness
                        if position.is_odd() {
                            if right.is_marked_leaf() {
                                left.leaf_value()
                                    .map(|v| vec![v.clone()])
                                    .ok_or_else(|| vec![l_addr])
                            } else {
                                Err(vec![l_addr])
                            }
                        } else if left.is_marked_leaf() {
                            // If we have the left-hand leaf and the right-hand leaf is empty, we
                            // can fill it with the empty leaf, but only if we are truncating at
                            // a position to the left of the current position
                            if truncate_at <= position + 1 {
                                Ok(vec![H::empty_leaf()])
                            } else {
                                right
                                    .leaf_value()
                                    .map_or_else(|| Err(vec![r_addr]), |v| Ok(vec![v.clone()]))
                            }
                        } else {
                            Err(vec![r_addr])
                        }
                    }
                }
                _ => {
                    // if we encounter a nil or leaf node, we were unable to descend
                    // to the leaf at the desired position.
                    Err(vec![root_addr])
                }
            }
        }

        if self.root_addr.position_range().contains(&position) {
            go(&self.root, self.root_addr, position, truncate_at)
                .map_err(QueryError::TreeIncomplete)
        } else {
            Err(QueryError::NotContained(self.root_addr))
        }
    }

    /// Prunes this tree by replacing all nodes that are right-hand children along the path
    /// to the specified position with [`Node::Nil`].
    ///
    /// The leaf at the specified position is retained. Returns the truncated tree if a leaf or
    /// subtree root with the specified position as its maximum position exists, or `None`
    /// otherwise.
    pub fn truncate_to_position(&self, position: Position) -> Option<Self> {
        fn go<H: Hashable + Clone + PartialEq>(
            position: Position,
            root_addr: Address,
            root: &PrunableTree<H>,
        ) -> Option<PrunableTree<H>> {
            match &root.0 {
                Node::Parent { ann, left, right } => {
                    let (l_child, r_child) = root_addr.children().unwrap();
                    if position < r_child.position_range_start() {
                        // we are truncating within the range of the left node, so recurse
                        // to the left to truncate the left child and then reconstruct the
                        // node with `Nil` as the right sibling
                        go(position, l_child, left.as_ref()).map(|left| {
                            Tree::unite(l_child.level(), ann.clone(), left, Tree(Node::Nil))
                        })
                    } else {
                        // we are truncating within the range of the right node, so recurse
                        // to the right to truncate the right child and then reconstruct the
                        // node with the left sibling unchanged
                        go(position, r_child, right.as_ref()).map(|right| {
                            Tree::unite(r_child.level(), ann.clone(), left.as_ref().clone(), right)
                        })
                    }
                }
                Node::Leaf { .. } => {
                    if root_addr.max_position() <= position {
                        Some(root.clone())
                    } else {
                        None
                    }
                }
                Node::Nil => None,
            }
        }

        if self.root_addr.position_range().contains(&position) {
            go(position, self.root_addr, &self.root).map(|root| LocatedTree {
                root_addr: self.root_addr,
                root,
            })
        } else {
            None
        }
    }

    /// Inserts a descendant subtree into this subtree, creating empty sibling nodes as necessary
    /// to fill out the tree.
    ///
    /// In the case that a leaf node would be replaced by an incomplete subtree, the resulting
    /// parent node will be annotated with the existing leaf value.
    ///
    /// Returns the updated tree, along with the addresses of any [`Node::Nil`] nodes that were
    /// inserted in the process of creating the parent nodes down to the insertion point, or an
    /// error if the specified subtree's root address is not in the range of valid descendants of
    /// the root node of this tree or if the insertion would result in a conflict between computed
    /// root hashes of complete subtrees.
    pub fn insert_subtree(
        &self,
        subtree: Self,
        contains_marked: bool,
    ) -> Result<(Self, Vec<IncompleteAt>), InsertionError> {
        // A function to recursively dig into the tree, creating a path downward and introducing
        // empty nodes as necessary until we can insert the provided subtree.
        #[allow(clippy::type_complexity)]
        fn go<H: Hashable + Clone + PartialEq>(
            root_addr: Address,
            into: &PrunableTree<H>,
            subtree: LocatedPrunableTree<H>,
            is_complete: bool,
            contains_marked: bool,
        ) -> Result<(PrunableTree<H>, Vec<IncompleteAt>), InsertionError> {
            // In the case that we are replacing a node entirely, we need to extend the
            // subtree up to the level of the node being replaced, adding Nil siblings
            // and recording the presence of those incomplete nodes when necessary
            let replacement = |ann: Option<Rc<H>>, mut node: LocatedPrunableTree<H>| {
                // construct the replacement node bottom-up
                let mut incomplete = vec![];
                while node.root_addr.level() < root_addr.level() {
                    incomplete.push(IncompleteAt {
                        address: node.root_addr.sibling(),
                        required_for_witness: contains_marked,
                    });
                    node = LocatedTree {
                        root_addr: node.root_addr.parent(),
                        root: if node.root_addr.is_right_child() {
                            Tree(Node::Parent {
                                ann: None,
                                left: Rc::new(Tree(Node::Nil)),
                                right: Rc::new(node.root),
                            })
                        } else {
                            Tree(Node::Parent {
                                ann: None,
                                left: Rc::new(node.root),
                                right: Rc::new(Tree(Node::Nil)),
                            })
                        },
                    };
                }
                (node.root.reannotate_root(ann), incomplete)
            };

            match into {
                Tree(Node::Nil) => Ok(replacement(None, subtree)),
                Tree(Node::Leaf { value: (value, _) }) => {
                    if root_addr == subtree.root_addr {
                        if is_complete {
                            // It is safe to replace the existing root unannotated, because we
                            // can always recompute the root from a complete subtree.
                            Ok((subtree.root, vec![]))
                        } else if subtree
                            .root
                            .0
                            .annotation()
                            .and_then(|ann| ann.as_ref())
                            .iter()
                            .all(|v| v.as_ref() == value)
                        {
                            Ok((
                                // at this point we statically know the root to be a parent
                                subtree.root.reannotate_root(Some(Rc::new(value.clone()))),
                                vec![],
                            ))
                        } else {
                            Err(InsertionError::Conflict(root_addr))
                        }
                    } else {
                        Ok(replacement(Some(Rc::new(value.clone())), subtree))
                    }
                }
                parent if root_addr == subtree.root_addr => {
                    // Merge the existing subtree with the subtree being inserted.
                    // A merge operation can't introduce any new incomplete roots.
                    parent
                        .clone()
                        .merge_checked(root_addr, subtree.root)
                        .map_err(InsertionError::Conflict)
                        .map(|tree| (tree, vec![]))
                }
                Tree(Node::Parent { ann, left, right }) => {
                    // In this case, we have an existing parent but we need to dig down farther
                    // before we can insert the subtree that we're carrying for insertion.
                    let (l_addr, r_addr) = root_addr.children().unwrap();
                    if l_addr.contains(&subtree.root_addr) {
                        let (new_left, incomplete) =
                            go(l_addr, left.as_ref(), subtree, is_complete, contains_marked)?;
                        Ok((
                            Tree::unite(
                                root_addr.level() - 1,
                                ann.clone(),
                                new_left,
                                right.as_ref().clone(),
                            ),
                            incomplete,
                        ))
                    } else {
                        let (new_right, incomplete) = go(
                            r_addr,
                            right.as_ref(),
                            subtree,
                            is_complete,
                            contains_marked,
                        )?;
                        Ok((
                            Tree::unite(
                                root_addr.level() - 1,
                                ann.clone(),
                                left.as_ref().clone(),
                                new_right,
                            ),
                            incomplete,
                        ))
                    }
                }
            }
        }

        let LocatedTree { root_addr, root } = self;
        if root_addr.contains(&subtree.root_addr) {
            let complete = subtree.root.is_complete();
            go(*root_addr, root, subtree, complete, contains_marked).map(|(root, incomplete)| {
                (
                    LocatedTree {
                        root_addr: *root_addr,
                        root,
                    },
                    incomplete,
                )
            })
        } else {
            Err(InsertionError::NotContained(subtree.root_addr))
        }
    }

    #[cfg(feature = "legacy-api")]
    fn combine_optional(
        opt_t0: Option<Self>,
        opt_t1: Option<Self>,
        contains_marked: bool,
    ) -> Result<Option<Self>, InsertionError> {
        match (opt_t0, opt_t1) {
            (Some(t0), Some(t1)) => {
                let into = LocatedTree {
                    root_addr: t0.root_addr().common_ancestor(&t1.root_addr()),
                    root: Tree::empty(),
                };

                into.insert_subtree(t0, contains_marked)
                    .and_then(|(into, _)| into.insert_subtree(t1, contains_marked))
                    .map(|(t, _)| Some(t))
            }
            (t0, t1) => Ok(t0.or(t1)),
        }
    }

    /// Append a single value at the first available position in the tree.
    ///
    /// Prefer to use [`Self::batch_append`] or [`Self::batch_insert`] when appending multiple
    /// values, as these operations require fewer traversals of the tree than are necessary when
    /// performing multiple sequential calls to [`Self::append`].
    pub fn append<C: Clone + Ord>(
        &self,
        value: H,
        retention: Retention<C>,
    ) -> Result<(Self, Position, Option<C>), InsertionError> {
        let checkpoint_id = if let Retention::Checkpoint { id, .. } = &retention {
            Some(id.clone())
        } else {
            None
        };

        self.batch_append(Some((value, retention)).into_iter())
            // We know that the max insert position will have been incremented by one.
            .and_then(|r| {
                let mut r = r.expect("We know the iterator to have been nonempty.");
                if r.remainder.next().is_some() {
                    Err(InsertionError::TreeFull)
                } else {
                    Ok((r.subtree, r.max_insert_position.unwrap(), checkpoint_id))
                }
            })
    }

    /// Append a values from an iterator, beginning at the first available position in the tree.
    ///
    /// Returns an error if the tree is full. If the position at the end of the iterator is outside
    /// of the subtree's range, the unconsumed part of the iterator will be returned as part of
    /// the result.
    pub fn batch_append<C: Clone + Ord, I: Iterator<Item = (H, Retention<C>)>>(
        &self,
        values: I,
    ) -> Result<Option<BatchInsertionResult<H, C, I>>, InsertionError> {
        let append_position = self
            .max_position()
            .map(|p| p + 1)
            .unwrap_or_else(|| self.root_addr.position_range_start());
        self.batch_insert(append_position, values)
    }

    /// Builds a [`LocatedPrunableTree`] from an iterator of level-0 leaves.
    ///
    /// This may be used in conjunction with [`ShardTree::insert_tree`] to support
    /// partially-parallelizable tree construction. Multiple subtrees may be constructed in
    /// parallel from iterators over (preferably, though not necessarily) disjoint leaf ranges, and
    /// [`ShardTree::insert_tree`] may be used to insert those subtrees into the `ShardTree` in
    /// arbitrary order.
    ///
    /// * `position_range` - The range of leaf positions at which values will be inserted. This
    ///   range is also used to place an upper bound on the number of items that will be consumed
    ///   from the `values` iterator.
    /// * `prune_below` - Nodes with [`Retention::Ephemeral`] retention that are not required to be retained
    ///   in order to construct a witness for a marked node or to make it possible to rewind to a
    ///   checkpointed node may be pruned so long as their address is at less than the specified
    ///   level.
    /// * `values` The iterator of `(H, [`Retention`])` pairs from which to construct the tree.
    pub fn from_iter<C: Clone + Ord, I: Iterator<Item = (H, Retention<C>)>>(
        position_range: Range<Position>,
        prune_below: Level,
        mut values: I,
    ) -> Option<BatchInsertionResult<H, C, I>> {
        // Unite two subtrees by either adding a parent node, or a leaf containing the Merkle root
        // of such a parent if both nodes are ephemeral leaves.
        //
        // `unite` is only called when both root addrs have the same parent.  `batch_insert` never
        // constructs Nil nodes, so we don't create any incomplete root information here.
        fn unite<H: Hashable + Clone + PartialEq>(
            lroot: LocatedPrunableTree<H>,
            rroot: LocatedPrunableTree<H>,
            prune_below: Level,
        ) -> LocatedTree<Option<Rc<H>>, (H, RetentionFlags)> {
            LocatedTree {
                root_addr: lroot.root_addr.parent(),
                root: if lroot.root_addr.level() < prune_below {
                    Tree::unite(lroot.root_addr.level(), None, lroot.root, rroot.root)
                } else {
                    Tree(Node::Parent {
                        ann: None,
                        left: Rc::new(lroot.root),
                        right: Rc::new(rroot.root),
                    })
                },
            }
        }

        // Builds a single tree from the provided stack of subtrees, which must be non-overlapping
        // and in position order. Returns the resulting tree, a flag indicating whether the
        // resulting tree contains a `MARKED` node, and the vector of [`IncompleteAt`] values for
        // [`Node::Nil`] nodes that were introduced in the process of constructing the tree.
        fn build_minimal_tree<H: Hashable + Clone + PartialEq>(
            mut xs: Vec<(LocatedPrunableTree<H>, bool)>,
            prune_below: Level,
        ) -> Option<(LocatedPrunableTree<H>, bool, Vec<IncompleteAt>)> {
            // First, consume the stack from the right, building up a single tree
            // until we can't combine any more.
            if let Some((mut cur, mut contains_marked)) = xs.pop() {
                let mut incomplete = vec![];
                while let Some((top, top_marked)) = xs.pop() {
                    while cur.root_addr.level() < top.root_addr.level() {
                        let sibling_addr = cur.root_addr.sibling();
                        incomplete.push(IncompleteAt {
                            address: sibling_addr,
                            required_for_witness: top_marked,
                        });
                        cur = unite(
                            cur,
                            LocatedTree {
                                root_addr: sibling_addr,
                                root: Tree(Node::Nil),
                            },
                            prune_below,
                        );
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
                            let sibling_addr = cur.root_addr.sibling();
                            incomplete.push(IncompleteAt {
                                address: sibling_addr,
                                required_for_witness: top_marked,
                            });
                            cur = unite(
                                cur,
                                LocatedTree {
                                    root_addr: sibling_addr,
                                    root: Tree(Node::Nil),
                                },
                                prune_below,
                            );
                            break;
                        }
                    } else {
                        // top.root_addr.level < cur.root_addr.level, so we've merged as much as we
                        // can from the right and now need to work from the left.
                        xs.push((top, top_marked));
                        break;
                    }
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
                                let sibling_addr = prev_tree.root_addr.sibling();
                                contains_marked = contains_marked || next_marked;
                                incomplete.push(IncompleteAt {
                                    address: sibling_addr,
                                    required_for_witness: next_marked,
                                });
                                prev_tree = unite(
                                    LocatedTree {
                                        root_addr: sibling_addr,
                                        root: Tree(Node::Nil),
                                    },
                                    prev_tree,
                                    prune_below,
                                );
                            }

                            // at this point, prev_tree.level == next_tree.level
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
                    root: Tree(Node::Leaf {
                        value: (value.clone(), rflags),
                    }),
                };

                if position.is_odd() {
                    // At odd positions, we are completing a subtree and so we unite fragments
                    // up the stack until we get the largest possible subtree
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

        build_minimal_tree(fragments, prune_below).map(
            |(to_insert, contains_marked, incomplete)| BatchInsertionResult {
                subtree: to_insert,
                contains_marked,
                incomplete,
                max_insert_position: Some(position - 1),
                checkpoints,
                remainder: values,
            },
        )
    }

    /// Put a range of values into the subtree by consuming the given iterator, starting at the
    /// specified position.
    ///
    /// The start position must exist within the position range of this subtree. If the position at
    /// the end of the iterator is outside of the subtree's range, the unconsumed part of the
    /// iterator will be returned as part of the result.
    ///
    /// Returns `Ok(None)` if the provided iterator is empty, `Ok(Some<BatchInsertionResult>)` if
    /// values were successfully inserted, or an error if the start position provided is outside
    /// of this tree's position range or if a conflict with an existing subtree root is detected.
    pub fn batch_insert<C: Clone + Ord, I: Iterator<Item = (H, Retention<C>)>>(
        &self,
        start: Position,
        values: I,
    ) -> Result<Option<BatchInsertionResult<H, C, I>>, InsertionError> {
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

    // Constructs a pair of trees contains the leaf and ommers of the given frontier.
    // The first element of the result is a tree with its root at a level less than
    // or equal to `split_at`; the second element is a tree with its leaves at level
    // `split_at` that is only returned if the frontier contains sufficient data to
    // fill the first tree to the `split_at` level.
    fn from_frontier<C>(
        frontier: NonEmptyFrontier<H>,
        leaf_retention: &Retention<C>,
        split_at: Level,
    ) -> (Self, Option<Self>) {
        let (position, leaf, ommers) = frontier.into_parts();
        Self::from_frontier_parts(position, leaf, ommers.into_iter(), leaf_retention, split_at)
    }

    // Permits construction of a subtree from legacy `CommitmentTree` data that may
    // have inaccurate position information (e.g. in the case that the tree is the
    // cursor for an `IncrementalWitness`).
    fn from_frontier_parts<C>(
        position: Position,
        leaf: H,
        mut ommers: impl Iterator<Item = H>,
        leaf_retention: &Retention<C>,
        split_at: Level,
    ) -> (Self, Option<Self>) {
        let mut addr = Address::from(position);
        let mut subtree = Tree(Node::Leaf {
            value: (leaf, leaf_retention.into()),
        });

        while addr.level() < split_at {
            if addr.index() & 0x1 == 0 {
                // the current address is a left child, so create a parent with
                // an empty right-hand tree
                subtree = Tree::parent(None, subtree, Tree::empty());
            } else if let Some(left) = ommers.next() {
                // the current address corresponds to a right child, so create a parent that
                // takes the left sibling to that child from the ommers
                subtree =
                    Tree::parent(None, Tree::leaf((left, RetentionFlags::EPHEMERAL)), subtree);
            } else {
                break;
            }

            addr = addr.parent();
        }

        let located_subtree = LocatedTree {
            root_addr: addr,
            root: subtree,
        };

        let located_supertree = if located_subtree.root_addr().level() == split_at {
            let mut addr = located_subtree.root_addr();
            let mut supertree = None;
            for left in ommers {
                // build up the left-biased tree until we get a right-hand node
                while addr.index() & 0x1 == 0 {
                    supertree = supertree.map(|t| Tree::parent(None, t, Tree::empty()));
                    addr = addr.parent();
                }

                // once we have a right-hand root, add a parent with the current ommer as the
                // left-hand sibling
                supertree = Some(Tree::parent(
                    None,
                    Tree::leaf((left, RetentionFlags::EPHEMERAL)),
                    supertree.unwrap_or_else(Tree::empty),
                ));
                addr = addr.parent();
            }

            supertree.map(|t| LocatedTree {
                root_addr: addr,
                root: t,
            })
        } else {
            // if there were not enough ommers available from the frontier to reach the address
            // of the root of this tree, there is no contribution to the cap
            None
        };

        (located_subtree, located_supertree)
    }

    #[cfg(feature = "legacy-api")]
    fn from_witness_filled_nodes(
        leaf_addr: Address,
        mut filled: impl Iterator<Item = H>,
        split_at: Level,
    ) -> (Self, Option<Self>) {
        // add filled nodes to the subtree; here, we do not need to worry about
        // whether or not these nodes can be invalidated by a rewind
        let mut addr = leaf_addr;
        let mut subtree = Tree::empty();
        while addr.level() < split_at {
            if addr.index() & 0x1 == 0 {
                // the current  root is a left child, so take the right sibling from the
                // filled iterator
                if let Some(right) = filled.next() {
                    // once we have a right-hand node, add a parent with the current tree
                    // as the left-hand sibling
                    subtree = Tree::parent(
                        None,
                        subtree,
                        Tree::leaf((right.clone(), RetentionFlags::EPHEMERAL)),
                    );
                } else {
                    break;
                }
            } else {
                // the current address is for a right child, so add an empty left sibling
                subtree = Tree::parent(None, Tree::empty(), subtree);
            }

            addr = addr.parent();
        }

        let subtree = LocatedTree {
            root_addr: addr,
            root: subtree,
        };

        // add filled nodes to the supertree
        let supertree = if addr.level() == split_at {
            let mut supertree = None;
            for right in filled {
                // build up the right-biased tree until we get a left-hand node
                while addr.index() & 0x1 == 1 {
                    supertree = supertree.map(|t| Tree::parent(None, Tree::empty(), t));
                    addr = addr.parent();
                }

                // once we have a left-hand root, add a parent with the current ommer as the right-hand sibling
                supertree = Some(Tree::parent(
                    None,
                    supertree.unwrap_or_else(PrunableTree::empty),
                    Tree::leaf((right.clone(), RetentionFlags::EPHEMERAL)),
                ));
                addr = addr.parent();
            }

            supertree.map(|t| LocatedTree {
                root_addr: addr,
                root: t,
            })
        } else {
            None
        };

        (subtree, supertree)
    }

    /// Inserts leaves and subtree roots from the provided frontier into this tree, up to the level
    /// of this tree's root.
    ///
    /// Returns the updated tree, along with a `LocatedPrunableTree` containing only the remainder
    /// of the frontier's ommers that had addresses at levels greater than the root of this tree.
    ///
    /// Returns an error in the following cases:
    /// * the leaf node of `frontier` is at a position that is not contained within this tree's
    ///   position range
    /// * a conflict occurs where an ommer of the frontier being inserted does not match the
    ///   existing value for that node
    pub fn insert_frontier_nodes<C>(
        &self,
        frontier: NonEmptyFrontier<H>,
        leaf_retention: &Retention<C>,
    ) -> Result<(Self, Option<Self>), InsertionError> {
        let subtree_range = self.root_addr.position_range();
        if subtree_range.contains(&frontier.position()) {
            let leaf_is_marked = leaf_retention.is_marked();
            let (subtree, supertree) =
                Self::from_frontier(frontier, leaf_retention, self.root_addr.level());

            let subtree = self.insert_subtree(subtree, leaf_is_marked)?.0;

            Ok((subtree, supertree))
        } else {
            Err(InsertionError::OutOfRange(
                frontier.position(),
                subtree_range,
            ))
        }
    }

    /// Insert the nodes belonging to the given incremental witness to this tree, truncating the
    /// witness to the given position.
    #[cfg(feature = "legacy-api")]
    pub fn insert_witness_nodes<C, const DEPTH: u8>(
        &self,
        witness: IncrementalWitness<H, DEPTH>,
        checkpoint_id: C,
    ) -> Result<(Self, Option<Self>, Option<Self>), InsertionError> {
        let subtree_range = self.root_addr.position_range();
        if subtree_range.contains(&witness.witnessed_position()) {
            // construct the subtree and cap based on the frontier containing the
            // witnessed position
            let (past_subtree, past_supertree) = self.insert_frontier_nodes::<C>(
                witness.tree().to_frontier().take().unwrap(),
                &Retention::Marked,
            )?;

            // construct subtrees from the `filled` nodes of the witness
            let (future_subtree, future_supertree) = Self::from_witness_filled_nodes(
                Address::from(witness.witnessed_position()),
                witness.filled().iter().cloned(),
                self.root_addr.level(),
            );

            // construct subtrees from the `cursor` part of the witness
            let cursor_trees = witness.cursor().as_ref().filter(|c| c.size() > 0).map(|c| {
                Self::from_frontier_parts(
                    witness.tip_position(),
                    c.leaf()
                        .cloned()
                        .expect("Cannot have an empty leaf for a non-empty tree"),
                    c.ommers_iter().cloned(),
                    &Retention::Checkpoint {
                        id: checkpoint_id,
                        is_marked: false,
                    },
                    self.root_addr.level(),
                )
            });

            let (subtree, _) = past_subtree.insert_subtree(future_subtree, true)?;

            let supertree =
                LocatedPrunableTree::combine_optional(past_supertree, future_supertree, true)?;

            Ok(if let Some((cursor_sub, cursor_super)) = cursor_trees {
                let (complete_subtree, fragment) =
                    if subtree.root_addr().contains(&cursor_sub.root_addr()) {
                        // the cursor subtree can be absorbed into the current subtree
                        (subtree.insert_subtree(cursor_sub, false)?.0, None)
                    } else {
                        // the cursor subtree must be maintained separately
                        (subtree, Some(cursor_sub))
                    };

                let complete_supertree =
                    LocatedPrunableTree::combine_optional(supertree, cursor_super, false)?;

                (complete_subtree, complete_supertree, fragment)
            } else {
                (subtree, supertree, None)
            })
        } else {
            Err(InsertionError::OutOfRange(
                witness.witnessed_position(),
                subtree_range,
            ))
        }
    }

    /// Clears the specified retention flags at all positions specified, pruning any branches
    /// that no longer need to be retained.
    pub fn clear_flags(&self, to_clear: BTreeMap<Position, RetentionFlags>) -> Self {
        fn go<H: Hashable + Clone + PartialEq>(
            to_clear: &[(Position, RetentionFlags)],
            root_addr: Address,
            root: &PrunableTree<H>,
        ) -> PrunableTree<H> {
            if to_clear.is_empty() {
                // nothing to do, so we just return the root
                root.clone()
            } else {
                match root {
                    Tree(Node::Parent { ann, left, right }) => {
                        let (l_addr, r_addr) = root_addr.children().unwrap();

                        let p = to_clear.partition_point(|(p, _)| p < &l_addr.position_range_end());
                        Tree::unite(
                            l_addr.level(),
                            ann.clone(),
                            go(&to_clear[0..p], l_addr, left),
                            go(&to_clear[p..], r_addr, right),
                        )
                    }
                    Tree(Node::Leaf { value: (h, r) }) => {
                        // When we reach a leaf, we should be down to just a single position
                        // which should correspond to the last level-0 child of the address's
                        // subtree range; if it's a checkpoint this will always be the case for
                        // a partially-pruned branch, and if it's a marked node then it will
                        // be a level-0 leaf.
                        match to_clear {
                            [(pos, flags)] => {
                                assert_eq!(*pos, root_addr.max_position());
                                Tree(Node::Leaf {
                                    value: (h.clone(), *r & !*flags),
                                })
                            }
                            _ => {
                                panic!("Tree state inconsistent with checkpoints.");
                            }
                        }
                    }
                    Tree(Node::Nil) => Tree(Node::Nil),
                }
            }
        }

        let to_clear = to_clear.into_iter().collect::<Vec<_>>();
        Self {
            root_addr: self.root_addr,
            root: go(&to_clear, self.root_addr, &self.root),
        }
    }
}

/// An enumeration of possible checkpoint locations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TreeState {
    /// Checkpoints of the empty tree.
    Empty,
    /// Checkpoint at a (possibly pruned) leaf state corresponding to the
    /// wrapped leaf position.
    AtPosition(Position),
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    tree_state: TreeState,
    marks_removed: BTreeSet<Position>,
}

impl Checkpoint {
    pub fn tree_empty() -> Self {
        Checkpoint {
            tree_state: TreeState::Empty,
            marks_removed: BTreeSet::new(),
        }
    }

    pub fn at_position(position: Position) -> Self {
        Checkpoint {
            tree_state: TreeState::AtPosition(position),
            marks_removed: BTreeSet::new(),
        }
    }

    pub fn from_parts(tree_state: TreeState, marks_removed: BTreeSet<Position>) -> Self {
        Checkpoint {
            tree_state,
            marks_removed,
        }
    }

    pub fn tree_state(&self) -> TreeState {
        self.tree_state
    }

    pub fn marks_removed(&self) -> &BTreeSet<Position> {
        &self.marks_removed
    }

    pub fn is_tree_empty(&self) -> bool {
        matches!(self.tree_state, TreeState::Empty)
    }

    pub fn position(&self) -> Option<Position> {
        match self.tree_state {
            TreeState::Empty => None,
            TreeState::AtPosition(pos) => Some(pos),
        }
    }
}

/// A capability for storage of fragment subtrees of the `ShardTree` type.
///
/// All fragment subtrees must have roots at level `SHARD_HEIGHT`
pub trait ShardStore {
    type H;
    type CheckpointId;
    type Error;

    /// Returns the subtree at the given root address, if any such subtree exists.
    fn get_shard(
        &self,
        shard_root: Address,
    ) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error>;

    /// Returns the subtree containing the maximum inserted leaf position.
    fn last_shard(&self) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error>;

    /// Inserts or replaces the subtree having the same root address as the provided tree.
    ///
    /// Implementations of this method MUST enforce the constraint that the root address
    /// of the provided subtree has level `SHARD_HEIGHT`.
    fn put_shard(&mut self, subtree: LocatedPrunableTree<Self::H>) -> Result<(), Self::Error>;

    /// Returns the vector of addresses corresponding to the roots of subtrees stored in this
    /// store.
    fn get_shard_roots(&self) -> Result<Vec<Address>, Self::Error>;

    /// Removes subtrees from the underlying store having root addresses at indices greater
    /// than or equal to that of the specified address.
    ///
    /// Implementations of this method MUST enforce the constraint that the root address
    /// provided has level `SHARD_HEIGHT`.
    fn truncate(&mut self, from: Address) -> Result<(), Self::Error>;

    /// A tree that is used to cache the known roots of subtrees in the "cap" of nodes between
    /// `SHARD_HEIGHT` and `DEPTH` that are otherwise not directly represented in the tree.  
    ///
    /// This tree only contains data that is expected to be stable; i.e. any node whose value
    /// might be altered as a consequence of the discovery of additional information from
    /// branches of the tree will be `Nil` in this cache.
    fn get_cap(&self) -> Result<PrunableTree<Self::H>, Self::Error>;

    /// Persists the provided cap to the data store.
    fn put_cap(&mut self, cap: PrunableTree<Self::H>) -> Result<(), Self::Error>;

    /// Returns the identifier for the checkpoint with the lowest associated position value.
    fn min_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error>;

    /// Returns the identifier for the checkpoint with the highest associated position value.
    fn max_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error>;

    /// Adds a checkpoint to the data store.
    fn add_checkpoint(
        &mut self,
        checkpoint_id: Self::CheckpointId,
        checkpoint: Checkpoint,
    ) -> Result<(), Self::Error>;

    /// Returns the number of checkpoints maintained by the data store
    fn checkpoint_count(&self) -> Result<usize, Self::Error>;

    /// Returns the position of the checkpoint, if any, along with the number of subsequent
    /// checkpoints at the same position. Returns `None` if `checkpoint_depth == 0` or if
    /// insufficient checkpoints exist to seek back to the requested depth.
    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<(Self::CheckpointId, Checkpoint)>, Self::Error>;

    /// Returns the checkpoint corresponding to the specified checkpoint identifier.
    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<Checkpoint>, Self::Error>;

    /// Iterates in checkpoint ID order over the first `limit` checkpoints, applying the
    /// given callback to each.
    fn with_checkpoints<F>(&mut self, limit: usize, callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&Self::CheckpointId, &Checkpoint) -> Result<(), Self::Error>;

    /// Update the checkpoint having the given identifier by mutating it with the provided
    /// function, and persist the updated checkpoint to the data store.
    ///
    /// Returns `Ok(true)` if the checkpoint was found, `Ok(false)` if no checkpoint with the
    /// provided identifier exists in the data store, or an error if a storage error occurred.
    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut Checkpoint) -> Result<(), Self::Error>;

    /// Removes a checkpoint from the data store.
    fn remove_checkpoint(&mut self, checkpoint_id: &Self::CheckpointId) -> Result<(), Self::Error>;

    /// Removes checkpoints with identifiers greater than or equal to the given identifier
    fn truncate_checkpoints(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error>;
}

impl<S: ShardStore> ShardStore for &mut S {
    type H = S::H;
    type CheckpointId = S::CheckpointId;
    type Error = S::Error;

    fn get_shard(
        &self,
        shard_root: Address,
    ) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error> {
        S::get_shard(*self, shard_root)
    }

    fn last_shard(&self) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error> {
        S::last_shard(*self)
    }

    fn put_shard(&mut self, subtree: LocatedPrunableTree<Self::H>) -> Result<(), Self::Error> {
        S::put_shard(*self, subtree)
    }

    fn get_shard_roots(&self) -> Result<Vec<Address>, Self::Error> {
        S::get_shard_roots(*self)
    }

    fn get_cap(&self) -> Result<PrunableTree<Self::H>, Self::Error> {
        S::get_cap(*self)
    }

    fn put_cap(&mut self, cap: PrunableTree<Self::H>) -> Result<(), Self::Error> {
        S::put_cap(*self, cap)
    }

    fn truncate(&mut self, from: Address) -> Result<(), Self::Error> {
        S::truncate(*self, from)
    }

    fn min_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error> {
        S::min_checkpoint_id(self)
    }

    fn max_checkpoint_id(&self) -> Result<Option<Self::CheckpointId>, Self::Error> {
        S::max_checkpoint_id(self)
    }

    fn add_checkpoint(
        &mut self,
        checkpoint_id: Self::CheckpointId,
        checkpoint: Checkpoint,
    ) -> Result<(), Self::Error> {
        S::add_checkpoint(self, checkpoint_id, checkpoint)
    }

    fn checkpoint_count(&self) -> Result<usize, Self::Error> {
        S::checkpoint_count(self)
    }

    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<(Self::CheckpointId, Checkpoint)>, Self::Error> {
        S::get_checkpoint_at_depth(self, checkpoint_depth)
    }

    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<Checkpoint>, Self::Error> {
        S::get_checkpoint(self, checkpoint_id)
    }

    fn with_checkpoints<F>(&mut self, limit: usize, callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&Self::CheckpointId, &Checkpoint) -> Result<(), Self::Error>,
    {
        S::with_checkpoints(self, limit, callback)
    }

    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut Checkpoint) -> Result<(), Self::Error>,
    {
        S::update_checkpoint_with(self, checkpoint_id, update)
    }

    fn remove_checkpoint(&mut self, checkpoint_id: &Self::CheckpointId) -> Result<(), Self::Error> {
        S::remove_checkpoint(self, checkpoint_id)
    }

    fn truncate_checkpoints(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        S::truncate_checkpoints(self, checkpoint_id)
    }
}

#[derive(Debug)]
pub struct MemoryShardStore<H, C: Ord> {
    shards: Vec<LocatedPrunableTree<H>>,
    checkpoints: BTreeMap<C, Checkpoint>,
    cap: PrunableTree<H>,
}

impl<H, C: Ord> MemoryShardStore<H, C> {
    pub fn empty() -> Self {
        Self {
            shards: vec![],
            checkpoints: BTreeMap::new(),
            cap: PrunableTree::empty(),
        }
    }
}

impl<H: Clone, C: Clone + Ord> ShardStore for MemoryShardStore<H, C> {
    type H = H;
    type CheckpointId = C;
    type Error = Infallible;

    fn get_shard(
        &self,
        shard_root: Address,
    ) -> Result<Option<LocatedPrunableTree<H>>, Self::Error> {
        let shard_idx =
            usize::try_from(shard_root.index()).expect("SHARD_HEIGHT > 64 is unsupported");
        Ok(self.shards.get(shard_idx).cloned())
    }

    fn last_shard(&self) -> Result<Option<LocatedPrunableTree<H>>, Self::Error> {
        Ok(self.shards.last().cloned())
    }

    fn put_shard(&mut self, subtree: LocatedPrunableTree<H>) -> Result<(), Self::Error> {
        let subtree_addr = subtree.root_addr;
        for subtree_idx in
            self.shards.last().map_or(0, |s| s.root_addr.index() + 1)..=subtree_addr.index()
        {
            self.shards.push(LocatedTree {
                root_addr: Address::from_parts(subtree_addr.level(), subtree_idx),
                root: Tree(Node::Nil),
            })
        }

        let shard_idx =
            usize::try_from(subtree_addr.index()).expect("SHARD_HEIGHT > 64 is unsupported");
        self.shards[shard_idx] = subtree;
        Ok(())
    }

    fn get_shard_roots(&self) -> Result<Vec<Address>, Self::Error> {
        Ok(self.shards.iter().map(|s| s.root_addr).collect())
    }

    fn truncate(&mut self, from: Address) -> Result<(), Self::Error> {
        let shard_idx = usize::try_from(from.index()).expect("SHARD_HEIGHT > 64 is unsupported");
        self.shards.truncate(shard_idx);
        Ok(())
    }

    fn get_cap(&self) -> Result<PrunableTree<H>, Self::Error> {
        Ok(self.cap.clone())
    }

    fn put_cap(&mut self, cap: PrunableTree<H>) -> Result<(), Self::Error> {
        self.cap = cap;
        Ok(())
    }

    fn add_checkpoint(
        &mut self,
        checkpoint_id: C,
        checkpoint: Checkpoint,
    ) -> Result<(), Self::Error> {
        self.checkpoints.insert(checkpoint_id, checkpoint);
        Ok(())
    }

    fn checkpoint_count(&self) -> Result<usize, Self::Error> {
        Ok(self.checkpoints.len())
    }

    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<Checkpoint>, Self::Error> {
        Ok(self.checkpoints.get(checkpoint_id).cloned())
    }

    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<(C, Checkpoint)>, Self::Error> {
        Ok(if checkpoint_depth == 0 {
            None
        } else {
            self.checkpoints
                .iter()
                .rev()
                .nth(checkpoint_depth - 1)
                .map(|(id, c)| (id.clone(), c.clone()))
        })
    }

    fn min_checkpoint_id(&self) -> Result<Option<C>, Self::Error> {
        Ok(self.checkpoints.keys().next().cloned())
    }

    fn max_checkpoint_id(&self) -> Result<Option<C>, Self::Error> {
        Ok(self.checkpoints.keys().last().cloned())
    }

    fn with_checkpoints<F>(&mut self, limit: usize, mut callback: F) -> Result<(), Self::Error>
    where
        F: FnMut(&C, &Checkpoint) -> Result<(), Self::Error>,
    {
        for (cid, checkpoint) in self.checkpoints.iter().take(limit) {
            callback(cid, checkpoint)?
        }

        Ok(())
    }

    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &C,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut Checkpoint) -> Result<(), Self::Error>,
    {
        if let Some(c) = self.checkpoints.get_mut(checkpoint_id) {
            update(c)?;
            return Ok(true);
        }

        Ok(false)
    }

    fn remove_checkpoint(&mut self, checkpoint_id: &C) -> Result<(), Self::Error> {
        self.checkpoints.remove(checkpoint_id);
        Ok(())
    }

    fn truncate_checkpoints(&mut self, checkpoint_id: &C) -> Result<(), Self::Error> {
        self.checkpoints.split_off(checkpoint_id);
        Ok(())
    }
}

/// A sparse binary Merkle tree of the specified depth, represented as an ordered collection of
/// subtrees (shards) of a given maximum height.
///
/// This tree maintains a collection of "checkpoints" which represent positions, usually near the
/// front of the tree, that are maintained such that it's possible to truncate nodes to the right
/// of the specified position.
#[derive(Debug)]
pub struct ShardTree<S: ShardStore, const DEPTH: u8, const SHARD_HEIGHT: u8> {
    /// The vector of tree shards.
    store: S,
    /// The maximum number of checkpoints to retain before pruning.
    max_checkpoints: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardTreeError<S> {
    Query(QueryError),
    Insert(InsertionError),
    Storage(S),
}

impl<S> From<QueryError> for ShardTreeError<S> {
    fn from(err: QueryError) -> Self {
        ShardTreeError::Query(err)
    }
}

impl<S> From<InsertionError> for ShardTreeError<S> {
    fn from(err: InsertionError) -> Self {
        ShardTreeError::Insert(err)
    }
}

impl<S: fmt::Display> fmt::Display for ShardTreeError<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            ShardTreeError::Query(q) => {
                write!(f, "{}", q)
            }
            ShardTreeError::Insert(i) => {
                write!(f, "{}", i)
            }
            ShardTreeError::Storage(s) => {
                write!(
                    f,
                    "An error occurred persisting or retrieving tree data: {}",
                    s
                )
            }
        }
    }
}

impl<SE> std::error::Error for ShardTreeError<SE>
where
    SE: Debug + std::fmt::Display + std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self {
            ShardTreeError::Storage(e) => Some(e),
            _ => None,
        }
    }
}

impl<
        H: Hashable + Clone + PartialEq,
        C: Clone + Ord,
        S: ShardStore<H = H, CheckpointId = C>,
        const DEPTH: u8,
        const SHARD_HEIGHT: u8,
    > ShardTree<S, DEPTH, SHARD_HEIGHT>
{
    /// Creates a new empty tree.
    pub fn new(store: S, max_checkpoints: usize) -> Self {
        Self {
            store,
            max_checkpoints,
        }
    }

    /// Returns the root address of the tree.
    pub fn root_addr() -> Address {
        Address::from_parts(Level::from(DEPTH), 0)
    }

    /// Returns the fixed level of subtree roots within the vector of subtrees used as this tree's
    /// representation.
    pub fn subtree_level() -> Level {
        Level::from(SHARD_HEIGHT)
    }

    /// Returns the root address of the subtree that contains the specified position.
    pub fn subtree_addr(pos: Position) -> Address {
        Address::above_position(Self::subtree_level(), pos)
    }

    pub fn max_subtree_index() -> u64 {
        (0x1 << (DEPTH - SHARD_HEIGHT)) - 1
    }

    /// Returns the leaf value at the specified position, if it is a marked leaf.
    pub fn get_marked_leaf(
        &self,
        position: Position,
    ) -> Result<Option<H>, ShardTreeError<S::Error>> {
        Ok(self
            .store
            .get_shard(Self::subtree_addr(position))
            .map_err(ShardTreeError::Storage)?
            .and_then(|t| t.value_at_position(position).cloned())
            .and_then(|(v, r)| if r.is_marked() { Some(v) } else { None }))
    }

    /// Returns the positions of marked leaves in the tree.
    pub fn marked_positions(&self) -> Result<BTreeSet<Position>, ShardTreeError<S::Error>> {
        let mut result = BTreeSet::new();
        for subtree_addr in &self
            .store
            .get_shard_roots()
            .map_err(ShardTreeError::Storage)?
        {
            if let Some(subtree) = self
                .store
                .get_shard(*subtree_addr)
                .map_err(ShardTreeError::Storage)?
            {
                result.append(&mut subtree.marked_positions());
            }
        }
        Ok(result)
    }

    /// Inserts a new root into the tree at the given address.
    ///
    /// This will pad from the left until the tree's subtrees vector contains enough trees to reach
    /// the specified address, which must be at the [`Self::subtree_level`] level. If a subtree
    /// already exists at this address, its root will be annotated with the specified hash value.
    ///
    /// This will return an error if the specified hash conflicts with any existing annotation.
    pub fn put_root(&mut self, addr: Address, value: H) -> Result<(), ShardTreeError<S::Error>> {
        let updated_subtree = match self
            .store
            .get_shard(addr)
            .map_err(ShardTreeError::Storage)?
        {
            Some(s) if !s.root.is_nil() => s.root.node_value().map_or_else(
                || {
                    Ok(Some(
                        s.clone().reannotate_root(Some(Rc::new(value.clone()))),
                    ))
                },
                |v| {
                    if v == &value {
                        // the existing root is already correctly annotated, so no need to
                        // do anything
                        Ok(None)
                    } else {
                        // the provided value conflicts with the existing root value
                        Err(InsertionError::Conflict(addr))
                    }
                },
            ),
            _ => {
                // there is no existing subtree root, so construct a new one.
                Ok(Some(LocatedTree {
                    root_addr: addr,
                    root: Tree(Node::Leaf {
                        value: (value, RetentionFlags::EPHEMERAL),
                    }),
                }))
            }
        }?;

        if let Some(s) = updated_subtree {
            self.store.put_shard(s).map_err(ShardTreeError::Storage)?;
        }

        Ok(())
    }

    /// Append a single value at the first available position in the tree.
    ///
    /// Prefer to use [`Self::batch_insert`] when appending multiple values, as these operations
    /// require fewer traversals of the tree than are necessary when performing multiple sequential
    /// calls to [`Self::append`].
    pub fn append(
        &mut self,
        value: H,
        retention: Retention<C>,
    ) -> Result<(), ShardTreeError<S::Error>> {
        if let Retention::Checkpoint { id, .. } = &retention {
            if self
                .store
                .max_checkpoint_id()
                .map_err(ShardTreeError::Storage)?
                .as_ref()
                >= Some(id)
            {
                return Err(InsertionError::CheckpointOutOfOrder.into());
            }
        }

        let (append_result, position, checkpoint_id) =
            if let Some(subtree) = self.store.last_shard().map_err(ShardTreeError::Storage)? {
                if subtree.root.is_complete() {
                    let addr = subtree.root_addr;

                    if addr.index() < Self::max_subtree_index() {
                        LocatedTree::empty(addr.next_at_level()).append(value, retention)?
                    } else {
                        return Err(InsertionError::TreeFull.into());
                    }
                } else {
                    subtree.append(value, retention)?
                }
            } else {
                let root_addr = Address::from_parts(Self::subtree_level(), 0);
                LocatedTree::empty(root_addr).append(value, retention)?
            };

        self.store
            .put_shard(append_result)
            .map_err(ShardTreeError::Storage)?;
        if let Some(c) = checkpoint_id {
            self.store
                .add_checkpoint(c, Checkpoint::at_position(position))
                .map_err(ShardTreeError::Storage)?;
        }

        self.prune_excess_checkpoints()?;

        Ok(())
    }

    /// Add the leaf and ommers of the provided frontier as nodes within the subtree corresponding
    /// to the frontier's position, and update the cap to include the ommer nodes at levels greater
    /// than or equal to the shard height.
    pub fn insert_frontier_nodes(
        &mut self,
        frontier: NonEmptyFrontier<H>,
        leaf_retention: Retention<C>,
    ) -> Result<(), ShardTreeError<S::Error>> {
        let leaf_position = frontier.position();
        let subtree_root_addr = Address::above_position(Self::subtree_level(), leaf_position);

        let (updated_subtree, supertree) = self
            .store
            .get_shard(subtree_root_addr)
            .map_err(ShardTreeError::Storage)?
            .unwrap_or_else(|| LocatedTree::empty(subtree_root_addr))
            .insert_frontier_nodes(frontier, &leaf_retention)?;

        self.store
            .put_shard(updated_subtree)
            .map_err(ShardTreeError::Storage)?;

        if let Some(supertree) = supertree {
            let new_cap = LocatedTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            }
            .insert_subtree(supertree, leaf_retention.is_marked())?;

            self.store
                .put_cap(new_cap.0.root)
                .map_err(ShardTreeError::Storage)?;
        }

        if let Retention::Checkpoint { id, is_marked: _ } = leaf_retention {
            self.store
                .add_checkpoint(id, Checkpoint::at_position(leaf_position))
                .map_err(ShardTreeError::Storage)?;
        }

        self.prune_excess_checkpoints()?;
        Ok(())
    }

    /// Add the leaf and ommers of the provided witness as nodes within the subtree corresponding
    /// to the frontier's position, and update the cap to include the nodes of the witness at
    /// levels greater than or equal to the shard height. Also, if the witness spans multiple
    /// subtrees, update the subtree corresponding to the current witness "tip" accordingly.
    #[cfg(feature = "legacy-api")]
    pub fn insert_witness_nodes(
        &mut self,
        witness: IncrementalWitness<H, DEPTH>,
        checkpoint_id: S::CheckpointId,
    ) -> Result<(), ShardTreeError<S::Error>> {
        let leaf_position = witness.witnessed_position();
        let subtree_root_addr = Address::above_position(Self::subtree_level(), leaf_position);

        let shard = self
            .store
            .get_shard(subtree_root_addr)
            .map_err(ShardTreeError::Storage)?
            .unwrap_or_else(|| LocatedTree::empty(subtree_root_addr));

        let (updated_subtree, supertree, tip_subtree) =
            shard.insert_witness_nodes(witness, checkpoint_id)?;

        self.store
            .put_shard(updated_subtree)
            .map_err(ShardTreeError::Storage)?;

        if let Some(supertree) = supertree {
            let new_cap = LocatedTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            }
            .insert_subtree(supertree, true)?;

            self.store
                .put_cap(new_cap.0.root)
                .map_err(ShardTreeError::Storage)?;
        }

        if let Some(tip_subtree) = tip_subtree {
            let tip_subtree_addr = Address::above_position(
                Self::subtree_level(),
                tip_subtree.root_addr().position_range_start(),
            );

            let tip_shard = self
                .store
                .get_shard(tip_subtree_addr)
                .map_err(ShardTreeError::Storage)?
                .unwrap_or_else(|| LocatedTree::empty(tip_subtree_addr));

            self.store
                .put_shard(tip_shard.insert_subtree(tip_subtree, false)?.0)
                .map_err(ShardTreeError::Storage)?;
        }

        Ok(())
    }

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
    #[allow(clippy::type_complexity)]
    pub fn batch_insert<I: Iterator<Item = (H, Retention<C>)>>(
        &mut self,
        mut start: Position,
        values: I,
    ) -> Result<Option<(Position, Vec<IncompleteAt>)>, ShardTreeError<S::Error>> {
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
                self.store
                    .put_shard(res.subtree)
                    .map_err(ShardTreeError::Storage)?;
                for (id, position) in res.checkpoints.into_iter() {
                    self.store
                        .add_checkpoint(id, Checkpoint::at_position(position))
                        .map_err(ShardTreeError::Storage)?;
                }

                values = res.remainder;
                subtree_root_addr = subtree_root_addr.next_at_level();
                max_insert_position = res.max_insert_position;
                start = max_insert_position.unwrap() + 1;
                all_incomplete.append(&mut res.incomplete);
            } else {
                break;
            }
        }

        self.prune_excess_checkpoints()?;
        Ok(max_insert_position.map(|p| (p, all_incomplete)))
    }

    /// Insert a tree by decomposing it into its `SHARD_HEIGHT` or smaller parts (if necessary)
    /// and inserting those at their appropriate locations.
    pub fn insert_tree(
        &mut self,
        tree: LocatedPrunableTree<H>,
    ) -> Result<Vec<IncompleteAt>, ShardTreeError<S::Error>> {
        let mut all_incomplete = vec![];
        for subtree in tree.decompose_to_level(Self::subtree_level()).into_iter() {
            let root_addr = subtree.root_addr;
            let contains_marked = subtree.root.contains_marked();
            let (new_subtree, mut incomplete) = self
                .store
                .get_shard(root_addr)
                .map_err(ShardTreeError::Storage)?
                .unwrap_or_else(|| LocatedTree::empty(root_addr))
                .insert_subtree(subtree, contains_marked)?;
            self.store
                .put_shard(new_subtree)
                .map_err(ShardTreeError::Storage)?;
            all_incomplete.append(&mut incomplete);
        }
        Ok(all_incomplete)
    }

    /// Adds a checkpoint at the rightmost leaf state of the tree.
    pub fn checkpoint(&mut self, checkpoint_id: C) -> Result<bool, ShardTreeError<S::Error>> {
        fn go<H: Hashable + Clone + PartialEq>(
            root_addr: Address,
            root: &PrunableTree<H>,
        ) -> Option<(PrunableTree<H>, Position)> {
            match root {
                Tree(Node::Parent { ann, left, right }) => {
                    let (l_addr, r_addr) = root_addr.children().unwrap();
                    go(r_addr, right).map_or_else(
                        || {
                            go(l_addr, left).map(|(new_left, pos)| {
                                (
                                    Tree::unite(
                                        l_addr.level(),
                                        ann.clone(),
                                        new_left,
                                        Tree(Node::Nil),
                                    ),
                                    pos,
                                )
                            })
                        },
                        |(new_right, pos)| {
                            Some((
                                Tree::unite(
                                    l_addr.level(),
                                    ann.clone(),
                                    left.as_ref().clone(),
                                    new_right,
                                ),
                                pos,
                            ))
                        },
                    )
                }
                Tree(Node::Leaf { value: (h, r) }) => Some((
                    Tree(Node::Leaf {
                        value: (h.clone(), *r | RetentionFlags::CHECKPOINT),
                    }),
                    root_addr.max_position(),
                )),
                Tree(Node::Nil) => None,
            }
        }

        // checkpoint identifiers at the tip must be in increasing order
        if self
            .store
            .max_checkpoint_id()
            .map_err(ShardTreeError::Storage)?
            .as_ref()
            >= Some(&checkpoint_id)
        {
            return Ok(false);
        }

        // Update the rightmost subtree to add the `CHECKPOINT` flag to the right-most leaf (which
        // need not be a level-0 leaf; it's fine to rewind to a pruned state).
        if let Some(subtree) = self.store.last_shard().map_err(ShardTreeError::Storage)? {
            if let Some((replacement, pos)) = go(subtree.root_addr, &subtree.root) {
                self.store
                    .put_shard(LocatedTree {
                        root_addr: subtree.root_addr,
                        root: replacement,
                    })
                    .map_err(ShardTreeError::Storage)?;
                self.store
                    .add_checkpoint(checkpoint_id, Checkpoint::at_position(pos))
                    .map_err(ShardTreeError::Storage)?;

                // early return once we've updated the tree state
                self.prune_excess_checkpoints()?;
                return Ok(true);
            }
        }

        self.store
            .add_checkpoint(checkpoint_id, Checkpoint::tree_empty())
            .map_err(ShardTreeError::Storage)?;

        // TODO: it should not be necessary to do this on every checkpoint,
        // but currently that's how the reference tree behaves so we're maintaining
        // those semantics for test compatibility.
        self.prune_excess_checkpoints()?;
        Ok(true)
    }

    fn prune_excess_checkpoints(&mut self) -> Result<(), ShardTreeError<S::Error>> {
        let checkpoint_count = self
            .store
            .checkpoint_count()
            .map_err(ShardTreeError::Storage)?;
        if checkpoint_count > self.max_checkpoints {
            // Batch removals by subtree & create a list of the checkpoint identifiers that
            // will be removed from the checkpoints map.
            let mut checkpoints_to_delete = vec![];
            let mut clear_positions: BTreeMap<Address, BTreeMap<Position, RetentionFlags>> =
                BTreeMap::new();
            self.store
                .with_checkpoints(
                    checkpoint_count - self.max_checkpoints,
                    |cid, checkpoint| {
                        checkpoints_to_delete.push(cid.clone());

                        let mut clear_at = |pos, flags_to_clear| {
                            let subtree_addr = Self::subtree_addr(pos);
                            clear_positions
                                .entry(subtree_addr)
                                .and_modify(|to_clear| {
                                    to_clear
                                        .entry(pos)
                                        .and_modify(|flags| *flags |= flags_to_clear)
                                        .or_insert(flags_to_clear);
                                })
                                .or_insert_with(|| BTreeMap::from([(pos, flags_to_clear)]));
                        };

                        // clear the checkpoint leaf
                        if let TreeState::AtPosition(pos) = checkpoint.tree_state {
                            clear_at(pos, RetentionFlags::CHECKPOINT)
                        }

                        // clear the leaves that have been marked for removal
                        for unmark_pos in checkpoint.marks_removed.iter() {
                            clear_at(*unmark_pos, RetentionFlags::MARKED)
                        }

                        Ok(())
                    },
                )
                .map_err(ShardTreeError::Storage)?;

            // Prune each affected subtree
            for (subtree_addr, positions) in clear_positions.into_iter() {
                let cleared = self
                    .store
                    .get_shard(subtree_addr)
                    .map_err(ShardTreeError::Storage)?
                    .map(|subtree| subtree.clear_flags(positions));
                if let Some(cleared) = cleared {
                    self.store
                        .put_shard(cleared)
                        .map_err(ShardTreeError::Storage)?;
                }
            }

            // Now that the leaves have been pruned, actually remove the checkpoints
            for c in checkpoints_to_delete {
                self.store
                    .remove_checkpoint(&c)
                    .map_err(ShardTreeError::Storage)?;
            }
        }

        Ok(())
    }

    /// Truncates the tree, discarding all information after the checkpoint at the specified depth.
    ///
    /// This will also discard all checkpoints with depth <= the specified depth. Returns `true`
    /// if the truncation succeeds or has no effect, or `false` if no checkpoint exists at the
    /// specified depth.
    pub fn truncate_to_depth(
        &mut self,
        checkpoint_depth: usize,
    ) -> Result<bool, ShardTreeError<S::Error>> {
        if checkpoint_depth == 0 {
            Ok(true)
        } else if let Some((checkpoint_id, c)) = self
            .store
            .get_checkpoint_at_depth(checkpoint_depth)
            .map_err(ShardTreeError::Storage)?
        {
            self.truncate_removing_checkpoint_internal(&checkpoint_id, &c)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Truncates the tree, discarding all information after the specified checkpoint.
    ///
    /// This will also discard all checkpoints with depth <= the specified depth. Returns `true`
    /// if the truncation succeeds or has no effect, or `false` if no checkpoint exists for the
    /// specified checkpoint identifier.
    pub fn truncate_removing_checkpoint(
        &mut self,
        checkpoint_id: &C,
    ) -> Result<bool, ShardTreeError<S::Error>> {
        match self
            .store
            .get_checkpoint(checkpoint_id)
            .map_err(ShardTreeError::Storage)?
        {
            Some(c) => {
                self.truncate_removing_checkpoint_internal(checkpoint_id, &c)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn truncate_removing_checkpoint_internal(
        &mut self,
        checkpoint_id: &C,
        checkpoint: &Checkpoint,
    ) -> Result<(), ShardTreeError<S::Error>> {
        match checkpoint.tree_state {
            TreeState::Empty => {
                self.store
                    .truncate(Address::from_parts(Self::subtree_level(), 0))
                    .map_err(ShardTreeError::Storage)?;
                self.store
                    .truncate_checkpoints(checkpoint_id)
                    .map_err(ShardTreeError::Storage)?;
                self.store
                    .put_cap(Tree::empty())
                    .map_err(ShardTreeError::Storage)?;
            }
            TreeState::AtPosition(position) => {
                let subtree_addr = Self::subtree_addr(position);
                let replacement = self
                    .store
                    .get_shard(subtree_addr)
                    .map_err(ShardTreeError::Storage)?
                    .and_then(|s| s.truncate_to_position(position));

                let cap_tree = LocatedTree {
                    root_addr: Self::root_addr(),
                    root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
                };

                if let Some(truncated) = cap_tree.truncate_to_position(position) {
                    self.store
                        .put_cap(truncated.root)
                        .map_err(ShardTreeError::Storage)?;
                };

                if let Some(truncated) = replacement {
                    self.store
                        .truncate(subtree_addr)
                        .map_err(ShardTreeError::Storage)?;
                    self.store
                        .put_shard(truncated)
                        .map_err(ShardTreeError::Storage)?;
                    self.store
                        .truncate_checkpoints(checkpoint_id)
                        .map_err(ShardTreeError::Storage)?;
                }
            }
        }

        Ok(())
    }

    /// Computes the root of any subtree of this tree rooted at the given address, with the overall
    /// tree truncated to the specified position.
    ///
    /// The specified address is not required to be at any particular level, though it cannot
    /// exceed the level corresponding to the maximum depth of the tree. Nodes to the right of the
    /// given position, and parents of such nodes, will be replaced by the empty root for the
    /// associated level.
    ///
    /// Use [`Self::root_at_checkpoint`] to obtain the root of the overall tree.
    pub fn root(
        &self,
        address: Address,
        truncate_at: Position,
    ) -> Result<H, ShardTreeError<S::Error>> {
        assert!(Self::root_addr().contains(&address));

        // traverse the cap from root to leaf depth-first, either returning an existing
        // cached value for the node or inserting the computed value into the cache
        let (root, _) = self.root_internal(
            &LocatedPrunableTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            },
            address,
            truncate_at,
        )?;
        Ok(root)
    }

    pub fn root_caching(
        &mut self,
        address: Address,
        truncate_at: Position,
    ) -> Result<H, ShardTreeError<S::Error>> {
        let (root, updated_cap) = self.root_internal(
            &LocatedPrunableTree {
                root_addr: Self::root_addr(),
                root: self.store.get_cap().map_err(ShardTreeError::Storage)?,
            },
            address,
            truncate_at,
        )?;
        if let Some(updated_cap) = updated_cap {
            self.store
                .put_cap(updated_cap)
                .map_err(ShardTreeError::Storage)?;
        }
        Ok(root)
    }

    // compute the root, along with an optional update to the cap
    #[allow(clippy::type_complexity)]
    fn root_internal(
        &self,
        cap: &LocatedPrunableTree<S::H>,
        // The address at which we want to compute the root hash
        target_addr: Address,
        // An inclusive lower bound for positions whose leaf values will be replaced by empty
        // roots.
        truncate_at: Position,
    ) -> Result<(H, Option<PrunableTree<H>>), ShardTreeError<S::Error>> {
        match &cap.root {
            Tree(Node::Parent { ann, left, right }) => {
                match ann {
                    Some(cached_root) if target_addr.contains(&cap.root_addr) => {
                        Ok((cached_root.as_ref().clone(), None))
                    }
                    _ => {
                        // Compute the roots of the left and right children and hash them together.
                        // We skip computation in any subtrees that will not have data included in
                        // the final result.
                        let (l_addr, r_addr) = cap.root_addr.children().unwrap();
                        let l_result = if r_addr.contains(&target_addr) {
                            None
                        } else {
                            Some(self.root_internal(
                                &LocatedPrunableTree {
                                    root_addr: l_addr,
                                    root: left.as_ref().clone(),
                                },
                                if l_addr.contains(&target_addr) {
                                    target_addr
                                } else {
                                    l_addr
                                },
                                truncate_at,
                            )?)
                        };
                        let r_result = if l_addr.contains(&target_addr) {
                            None
                        } else {
                            Some(self.root_internal(
                                &LocatedPrunableTree {
                                    root_addr: r_addr,
                                    root: right.as_ref().clone(),
                                },
                                if r_addr.contains(&target_addr) {
                                    target_addr
                                } else {
                                    r_addr
                                },
                                truncate_at,
                            )?)
                        };

                        // Compute the root value based on the child roots; these may contain the
                        // hashes of empty/truncated nodes.
                        let (root, new_left, new_right) = match (l_result, r_result) {
                            (Some((l_root, new_left)), Some((r_root, new_right))) => (
                                S::H::combine(l_addr.level(), &l_root, &r_root),
                                new_left,
                                new_right,
                            ),
                            (Some((l_root, new_left)), None) => (l_root, new_left, None),
                            (None, Some((r_root, new_right))) => (r_root, None, new_right),
                            (None, None) => unreachable!(),
                        };

                        let new_parent = Tree(Node::Parent {
                            ann: new_left
                                .as_ref()
                                .and_then(|l| l.node_value())
                                .zip(new_right.as_ref().and_then(|r| r.node_value()))
                                .map(|(l, r)| {
                                    // the node values of child nodes cannot contain the hashes of
                                    // empty nodes or nodes with positions greater than the
                                    Rc::new(S::H::combine(l_addr.level(), l, r))
                                }),
                            left: new_left.map_or_else(|| left.clone(), Rc::new),
                            right: new_right.map_or_else(|| right.clone(), Rc::new),
                        });

                        Ok((root, Some(new_parent)))
                    }
                }
            }
            Tree(Node::Leaf { value }) => {
                if truncate_at >= cap.root_addr.position_range_end()
                    && target_addr.contains(&cap.root_addr)
                {
                    // no truncation or computation of child subtrees of this leaf is necessary, just use
                    // the cached leaf value
                    Ok((value.0.clone(), None))
                } else {
                    // since the tree was truncated below this level, recursively call with an
                    // empty parent node to trigger the continued traversal
                    let (root, replacement) = self.root_internal(
                        &LocatedPrunableTree {
                            root_addr: cap.root_addr(),
                            root: Tree::parent(None, Tree::empty(), Tree::empty()),
                        },
                        target_addr,
                        truncate_at,
                    )?;

                    Ok((
                        root,
                        replacement.map(|r| r.reannotate_root(Some(Rc::new(value.0.clone())))),
                    ))
                }
            }
            Tree(Node::Nil) => {
                if cap.root_addr == target_addr
                    || cap.root_addr.level() == ShardTree::<S, DEPTH, SHARD_HEIGHT>::subtree_level()
                {
                    // We are at the leaf level or the target address; compute the root hash and
                    // return it as cacheable if it is not truncated.
                    let root = self.root_from_shards(target_addr, truncate_at)?;
                    Ok((
                        root.clone(),
                        if truncate_at >= cap.root_addr.position_range_end() {
                            // return the compute root as a new leaf to be cached if it contains no
                            // empty hashes due to truncation
                            Some(Tree::leaf((root, RetentionFlags::EPHEMERAL)))
                        } else {
                            None
                        },
                    ))
                } else {
                    // Compute the result by recursively walking down the tree. By replacing
                    // the current node with a parent node, the `Parent` handler will take care
                    // of the branching recursive calls.
                    self.root_internal(
                        &LocatedPrunableTree {
                            root_addr: cap.root_addr,
                            root: Tree::parent(None, Tree::empty(), Tree::empty()),
                        },
                        target_addr,
                        truncate_at,
                    )
                }
            }
        }
    }

    fn root_from_shards(
        &self,
        address: Address,
        truncate_at: Position,
    ) -> Result<H, ShardTreeError<S::Error>> {
        match address.context(Self::subtree_level()) {
            Either::Left(subtree_addr) => {
                // The requested root address is fully contained within one of the subtrees.
                Ok(if truncate_at <= address.position_range_start() {
                    H::empty_root(address.level())
                } else {
                    // get the child of the subtree with its root at `address`
                    self.store
                        .get_shard(subtree_addr)
                        .map_err(ShardTreeError::Storage)?
                        .ok_or_else(|| vec![subtree_addr])
                        .and_then(|subtree| {
                            subtree.subtree(address).map_or_else(
                                || Err(vec![address]),
                                |child| child.root_hash(truncate_at),
                            )
                        })
                        .map_err(QueryError::TreeIncomplete)?
                })
            }
            Either::Right(subtree_range) => {
                // The requested root requires hashing together the roots of several subtrees.
                let mut root_stack = vec![];
                let mut incomplete = vec![];

                for subtree_idx in subtree_range {
                    let subtree_addr = Address::from_parts(Self::subtree_level(), subtree_idx);
                    if truncate_at <= subtree_addr.position_range_start() {
                        break;
                    }

                    let subtree_root = self
                        .store
                        .get_shard(subtree_addr)
                        .map_err(ShardTreeError::Storage)?
                        .ok_or_else(|| vec![subtree_addr])
                        .and_then(|s| s.root_hash(truncate_at));

                    match subtree_root {
                        Ok(mut cur_hash) => {
                            if subtree_addr.index() % 2 == 0 {
                                root_stack.push((subtree_addr, cur_hash))
                            } else {
                                let mut cur_addr = subtree_addr;
                                while let Some((addr, hash)) = root_stack.pop() {
                                    if addr.parent() == cur_addr.parent() {
                                        cur_hash = H::combine(cur_addr.level(), &hash, &cur_hash);
                                        cur_addr = cur_addr.parent();
                                    } else {
                                        root_stack.push((addr, hash));
                                        break;
                                    }
                                }
                                root_stack.push((cur_addr, cur_hash));
                            }
                        }
                        Err(mut new_incomplete) => {
                            // Accumulate incomplete root information and continue, so that we can
                            // return the complete set of incomplete results.
                            incomplete.append(&mut new_incomplete);
                        }
                    }
                }

                if !incomplete.is_empty() {
                    return Err(ShardTreeError::Query(QueryError::TreeIncomplete(
                        incomplete,
                    )));
                }

                // Now hash with empty roots to obtain the root at maximum height
                if let Some((mut cur_addr, mut cur_hash)) = root_stack.pop() {
                    while let Some((addr, hash)) = root_stack.pop() {
                        while addr.level() > cur_addr.level() {
                            cur_hash = H::combine(
                                cur_addr.level(),
                                &cur_hash,
                                &H::empty_root(cur_addr.level()),
                            );
                            cur_addr = cur_addr.parent();
                        }
                        cur_hash = H::combine(cur_addr.level(), &hash, &cur_hash);
                        cur_addr = cur_addr.parent();
                    }

                    while cur_addr.level() < address.level() {
                        cur_hash = H::combine(
                            cur_addr.level(),
                            &cur_hash,
                            &H::empty_root(cur_addr.level()),
                        );
                        cur_addr = cur_addr.parent();
                    }

                    Ok(cur_hash)
                } else {
                    // if the stack is empty, we just return the default root at max height
                    Ok(H::empty_root(address.level()))
                }
            }
        }
    }

    /// Returns the position of the rightmost leaf inserted as of the given checkpoint.
    ///
    /// Returns the maximum leaf position if `checkpoint_depth == 0` (or `Ok(None)` in this
    /// case if the tree is empty) or an error if the checkpointed position cannot be restored
    /// because it has been pruned. Note that no actual level-0 leaf may exist at this position.
    pub fn max_leaf_position(
        &self,
        checkpoint_depth: usize,
    ) -> Result<Option<Position>, ShardTreeError<S::Error>> {
        Ok(if checkpoint_depth == 0 {
            // TODO: This relies on the invariant that the last shard in the subtrees vector is
            // never created without a leaf then being added to it. However, this may be a
            // difficult invariant to maintain when adding empty roots, so perhaps we need a
            // better way of tracking the actual max position of the tree; we might want to
            // just store it directly.
            self.store
                .last_shard()
                .map_err(ShardTreeError::Storage)?
                .and_then(|t| t.max_position())
        } else {
            match self
                .store
                .get_checkpoint_at_depth(checkpoint_depth)
                .map_err(ShardTreeError::Storage)?
            {
                Some((_, c)) => Ok(c.position()),
                None => {
                    // There is no checkpoint at the specified depth, so we report it as pruned.
                    Err(QueryError::CheckpointPruned)
                }
            }?
        })
    }

    /// Computes the root of the tree as of the checkpointed position at the specified depth.
    ///
    /// Returns the root as of the most recently appended leaf if `checkpoint_depth == 0`. Note
    /// that if the most recently appended leaf is also a checkpoint, this will return the same
    /// result as `checkpoint_depth == 1`.
    pub fn root_at_checkpoint(
        &self,
        checkpoint_depth: usize,
    ) -> Result<H, ShardTreeError<S::Error>> {
        self.max_leaf_position(checkpoint_depth)?.map_or_else(
            || Ok(H::empty_root(Self::root_addr().level())),
            |pos| self.root(Self::root_addr(), pos + 1),
        )
    }

    pub fn root_at_checkpoint_caching(
        &mut self,
        checkpoint_depth: usize,
    ) -> Result<H, ShardTreeError<S::Error>> {
        self.max_leaf_position(checkpoint_depth)?.map_or_else(
            || Ok(H::empty_root(Self::root_addr().level())),
            |pos| self.root_caching(Self::root_addr(), pos + 1),
        )
    }

    /// Computes the witness for the leaf at the specified position.
    ///
    /// Returns the witness as of the most recently appended leaf if `checkpoint_depth == 0`. Note
    /// that if the most recently appended leaf is also a checkpoint, this will return the same
    /// result as `checkpoint_depth == 1`.
    pub fn witness(
        &self,
        position: Position,
        checkpoint_depth: usize,
    ) -> Result<MerklePath<H, DEPTH>, ShardTreeError<S::Error>> {
        let max_leaf_position = self.max_leaf_position(checkpoint_depth).and_then(|v| {
            v.ok_or_else(|| QueryError::TreeIncomplete(vec![Self::root_addr()]).into())
        })?;

        if position > max_leaf_position {
            Err(
                QueryError::NotContained(Address::from_parts(Level::from(0), position.into()))
                    .into(),
            )
        } else {
            let subtree_addr = Self::subtree_addr(position);

            // compute the witness for the specified position up to the subtree root
            let mut witness = self
                .store
                .get_shard(subtree_addr)
                .map_err(ShardTreeError::Storage)?
                .map_or_else(
                    || Err(QueryError::TreeIncomplete(vec![subtree_addr])),
                    |subtree| subtree.witness(position, max_leaf_position + 1),
                )?;

            // compute the remaining parts of the witness up to the root
            let root_addr = Self::root_addr();
            let mut cur_addr = subtree_addr;
            while cur_addr != root_addr {
                witness.push(self.root(cur_addr.sibling(), max_leaf_position + 1)?);
                cur_addr = cur_addr.parent();
            }

            Ok(MerklePath::from_parts(witness, position).unwrap())
        }
    }

    pub fn witness_caching(
        &mut self,
        position: Position,
        checkpoint_depth: usize,
    ) -> Result<MerklePath<H, DEPTH>, ShardTreeError<S::Error>> {
        let max_leaf_position = self.max_leaf_position(checkpoint_depth).and_then(|v| {
            v.ok_or_else(|| QueryError::TreeIncomplete(vec![Self::root_addr()]).into())
        })?;

        if position > max_leaf_position {
            Err(
                QueryError::NotContained(Address::from_parts(Level::from(0), position.into()))
                    .into(),
            )
        } else {
            let subtree_addr = Address::above_position(Self::subtree_level(), position);

            // compute the witness for the specified position up to the subtree root
            let mut witness = self
                .store
                .get_shard(subtree_addr)
                .map_err(ShardTreeError::Storage)?
                .map_or_else(
                    || Err(QueryError::TreeIncomplete(vec![subtree_addr])),
                    |subtree| subtree.witness(position, max_leaf_position + 1),
                )?;

            // compute the remaining parts of the witness up to the root
            let root_addr = Self::root_addr();
            let mut cur_addr = subtree_addr;
            while cur_addr != root_addr {
                witness.push(self.root_caching(cur_addr.sibling(), max_leaf_position + 1)?);
                cur_addr = cur_addr.parent();
            }

            Ok(MerklePath::from_parts(witness, position).unwrap())
        }
    }

    /// Make a marked leaf at a position eligible to be pruned.
    ///
    /// If the checkpoint associated with the specified identifier does not exist because the
    /// corresponding checkpoint would have been more than `max_checkpoints` deep, the removal is
    /// recorded as of the first existing checkpoint and the associated leaves will be pruned when
    /// that checkpoint is subsequently removed.
    ///
    /// Returns `Ok(true)` if a mark was successfully removed from the leaf at the specified
    /// position, `Ok(false)` if the tree does not contain a leaf at the specified position or is
    /// not marked, or an error if one is produced by the underlying data store.
    pub fn remove_mark(
        &mut self,
        position: Position,
        as_of_checkpoint: Option<&C>,
    ) -> Result<bool, ShardTreeError<S::Error>> {
        match self
            .store
            .get_shard(Self::subtree_addr(position))
            .map_err(ShardTreeError::Storage)?
        {
            Some(shard)
                if shard
                    .value_at_position(position)
                    .iter()
                    .any(|(_, r)| r.is_marked()) =>
            {
                match as_of_checkpoint {
                    Some(cid)
                        if Some(cid)
                            >= self
                                .store
                                .min_checkpoint_id()
                                .map_err(ShardTreeError::Storage)?
                                .as_ref() =>
                    {
                        self.store
                            .update_checkpoint_with(cid, |checkpoint| {
                                checkpoint.marks_removed.insert(position);
                                Ok(())
                            })
                            .map_err(ShardTreeError::Storage)
                    }
                    _ => {
                        // if no checkpoint was provided, or if the checkpoint is too far in the past,
                        // remove the mark directly.
                        self.store
                            .put_shard(
                                shard.clear_flags(BTreeMap::from([(
                                    position,
                                    RetentionFlags::MARKED,
                                )])),
                            )
                            .map_err(ShardTreeError::Storage)?;
                        Ok(true)
                    }
                }
            }
            _ => Ok(false),
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
    use incrementalmerkletree::{testing, Hashable};
    use proptest::bool::weighted;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::sample::select;

    pub fn arb_retention_flags() -> impl Strategy<Value = RetentionFlags> + Clone {
        select(vec![
            RetentionFlags::EPHEMERAL,
            RetentionFlags::CHECKPOINT,
            RetentionFlags::MARKED,
            RetentionFlags::MARKED | RetentionFlags::CHECKPOINT,
        ])
    }

    pub fn arb_tree<A: Strategy + Clone + 'static, V: Strategy + 'static>(
        arb_annotation: A,
        arb_leaf: V,
        depth: u32,
        size: u32,
    ) -> impl Strategy<Value = Tree<A::Value, V::Value>> + Clone
    where
        A::Value: Clone + 'static,
        V::Value: Clone + 'static,
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

    pub fn arb_prunable_tree<H: Strategy + Clone + 'static>(
        arb_leaf: H,
        depth: u32,
        size: u32,
    ) -> impl Strategy<Value = PrunableTree<H::Value>> + Clone
    where
        H::Value: Clone + 'static,
    {
        arb_tree(
            proptest::option::of(arb_leaf.clone().prop_map(Rc::new)),
            (arb_leaf, arb_retention_flags()),
            depth,
            size,
        )
    }

    /// Constructs a random shardtree of size up to 2^6 with shards of size 2^3. Returns the tree,
    /// along with vectors of the checkpoint and mark positions.
    pub fn arb_shardtree<H: Strategy + Clone>(
        arb_leaf: H,
    ) -> impl Strategy<
        Value = (
            ShardTree<MemoryShardStore<H::Value, usize>, 6, 3>,
            Vec<Position>,
            Vec<Position>,
        ),
    >
    where
        H::Value: Hashable + Clone + PartialEq,
    {
        vec(
            (arb_leaf, weighted(0.1), weighted(0.2)),
            0..=(2usize.pow(6)),
        )
        .prop_map(|leaves| {
            let mut tree = ShardTree::new(MemoryShardStore::empty(), 10);
            let mut checkpoint_positions = vec![];
            let mut marked_positions = vec![];
            tree.batch_insert(
                Position::from(0),
                leaves
                    .into_iter()
                    .enumerate()
                    .map(|(id, (leaf, is_marked, is_checkpoint))| {
                        (
                            leaf,
                            match (is_checkpoint, is_marked) {
                                (false, false) => Retention::Ephemeral,
                                (true, is_marked) => {
                                    let pos = Position::try_from(id).unwrap();
                                    checkpoint_positions.push(pos);
                                    if is_marked {
                                        marked_positions.push(pos);
                                    }
                                    Retention::Checkpoint { id, is_marked }
                                }
                                (false, true) => {
                                    marked_positions.push(Position::try_from(id).unwrap());
                                    Retention::Marked
                                }
                            },
                        )
                    }),
            )
            .unwrap();
            (tree, checkpoint_positions, marked_positions)
        })
    }

    pub fn arb_char_str() -> impl Strategy<Value = String> + Clone {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        (0usize..chars.len()).prop_map(move |i| chars.get(i..=i).unwrap().to_string())
    }

    impl<
            H: Hashable + Ord + Clone,
            C: Clone + Ord + core::fmt::Debug,
            S: ShardStore<H = H, CheckpointId = C>,
            const DEPTH: u8,
            const SHARD_HEIGHT: u8,
        > testing::Tree<H, C> for ShardTree<S, DEPTH, SHARD_HEIGHT>
    where
        S::Error: std::fmt::Debug,
    {
        fn depth(&self) -> u8 {
            DEPTH
        }

        fn append(&mut self, value: H, retention: Retention<C>) -> bool {
            match ShardTree::append(self, value, retention) {
                Ok(_) => true,
                Err(ShardTreeError::Insert(InsertionError::TreeFull)) => false,
                Err(other) => panic!("append failed due to error: {:?}", other),
            }
        }

        fn current_position(&self) -> Option<Position> {
            match ShardTree::max_leaf_position(self, 0) {
                Ok(v) => v,
                Err(err) => panic!("current position query failed: {:?}", err),
            }
        }

        fn get_marked_leaf(&self, position: Position) -> Option<H> {
            match ShardTree::get_marked_leaf(self, position) {
                Ok(v) => v,
                Err(err) => panic!("marked leaf query failed: {:?}", err),
            }
        }

        fn marked_positions(&self) -> BTreeSet<Position> {
            match ShardTree::marked_positions(&self) {
                Ok(v) => v,
                Err(err) => panic!("marked positions query failed: {:?}", err),
            }
        }

        fn root(&self, checkpoint_depth: usize) -> Option<H> {
            match ShardTree::root_at_checkpoint(self, checkpoint_depth) {
                Ok(v) => Some(v),
                Err(err) => panic!("root computation failed: {:?}", err),
            }
        }

        fn witness(&self, position: Position, checkpoint_depth: usize) -> Option<Vec<H>> {
            match ShardTree::witness(self, position, checkpoint_depth) {
                Ok(p) => Some(p.path_elems().to_vec()),
                Err(ShardTreeError::Query(
                    QueryError::NotContained(_) | QueryError::TreeIncomplete(_) | QueryError::CheckpointPruned
                )) => None,
                Err(err) => panic!("witness computation failed: {:?}", err),
            }
        }

        fn remove_mark(&mut self, position: Position) -> bool {
            let max_checkpoint = self
                .store
                .max_checkpoint_id()
                .unwrap_or_else(|err| panic!("checkpoint retrieval failed: {:?}", err));

            match ShardTree::remove_mark(self, position, max_checkpoint.as_ref()) {
                Ok(result) => result,
                Err(err) => panic!("mark removal failed: {:?}", err),
            }
        }

        fn checkpoint(&mut self, checkpoint_id: C) -> bool {
            ShardTree::checkpoint(self, checkpoint_id).unwrap()
        }

        fn rewind(&mut self) -> bool {
            ShardTree::truncate_to_depth(self, 1).unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        testing::{arb_char_str, arb_shardtree},
        IncompleteAt, InsertionError, LocatedPrunableTree, LocatedTree, MemoryShardStore, Node,
        PrunableTree, QueryError, RetentionFlags, ShardTree, ShardTreeError, Tree,
    };

    use assert_matches::assert_matches;
    use incrementalmerkletree::{
        frontier::NonEmptyFrontier,
        testing::{
            arb_operation, check_append, check_checkpoint_rewind, check_operations,
            check_rewind_remove_mark, check_root_hashes, check_witnesses,
            complete_tree::CompleteTree, CombinedTree, SipHashable,
        },
        Address, Hashable, Level, Position, Retention,
    };
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    #[cfg(feature = "legacy-api")]
    use incrementalmerkletree::{frontier::CommitmentTree, witness::IncrementalWitness};

    fn str_leaf<A>(c: &str) -> Tree<A, String> {
        Tree(Node::Leaf {
            value: c.to_string(),
        })
    }

    fn nil<A, B>() -> Tree<A, B> {
        Tree::empty()
    }

    fn leaf<A, B>(value: B) -> Tree<A, B> {
        Tree::leaf(value)
    }

    fn parent<A: Default, B>(left: Tree<A, B>, right: Tree<A, B>) -> Tree<A, B> {
        Tree::parent(A::default(), left, right)
    }

    #[test]
    fn tree_incomplete_nodes() {
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

    #[test]
    fn tree_root() {
        let t: PrunableTree<String> = parent(
            leaf(("a".to_string(), RetentionFlags::EPHEMERAL)),
            leaf(("b".to_string(), RetentionFlags::EPHEMERAL)),
        );

        assert_eq!(
            t.root_hash(Address::from_parts(Level::from(1), 0), Position::from(2)),
            Ok("ab".to_string())
        );

        let t0 = parent(nil(), t.clone());
        assert_eq!(
            t0.root_hash(Address::from_parts(Level::from(2), 0), Position::from(4)),
            Err(vec![Address::from_parts(Level::from(1), 0)])
        );

        // Check root computation with truncation
        let t1 = parent(t, nil());
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
    fn located_prunable_tree_insert_subtree() {
        let t: LocatedPrunableTree<String> = LocatedTree {
            root_addr: Address::from_parts(3.into(), 1),
            root: parent(
                leaf(("abcd".to_string(), RetentionFlags::EPHEMERAL)),
                parent(nil(), leaf(("gh".to_string(), RetentionFlags::EPHEMERAL))),
            ),
        };

        assert_eq!(
            t.insert_subtree(
                LocatedTree {
                    root_addr: Address::from_parts(1.into(), 6),
                    root: parent(leaf(("e".to_string(), RetentionFlags::MARKED)), nil())
                },
                true
            ),
            Ok((
                LocatedTree {
                    root_addr: Address::from_parts(3.into(), 1),
                    root: parent(
                        leaf(("abcd".to_string(), RetentionFlags::EPHEMERAL)),
                        parent(
                            parent(leaf(("e".to_string(), RetentionFlags::MARKED)), nil()),
                            leaf(("gh".to_string(), RetentionFlags::EPHEMERAL))
                        )
                    )
                },
                vec![]
            ))
        );
    }

    #[test]
    fn located_prunable_tree_witness() {
        let t: LocatedPrunableTree<String> = LocatedTree {
            root_addr: Address::from_parts(3.into(), 0),
            root: parent(
                leaf(("abcd".to_string(), RetentionFlags::EPHEMERAL)),
                parent(
                    parent(
                        leaf(("e".to_string(), RetentionFlags::MARKED)),
                        leaf(("f".to_string(), RetentionFlags::EPHEMERAL)),
                    ),
                    leaf(("gh".to_string(), RetentionFlags::EPHEMERAL)),
                ),
            ),
        };

        assert_eq!(
            t.witness(4.into(), 8.into()),
            Ok(vec!["f", "gh", "abcd"]
                .into_iter()
                .map(|s| s.to_string())
                .collect())
        );
        assert_eq!(
            t.witness(4.into(), 6.into()),
            Ok(vec!["f", "__", "abcd"]
                .into_iter()
                .map(|s| s.to_string())
                .collect())
        );
        assert_eq!(
            t.witness(4.into(), 7.into()),
            Err(QueryError::TreeIncomplete(vec![Address::from_parts(
                1.into(),
                3
            )]))
        );
    }

    #[test]
    fn tree_marked_positions() {
        let t: PrunableTree<String> = parent(
            leaf(("a".to_string(), RetentionFlags::EPHEMERAL)),
            leaf(("b".to_string(), RetentionFlags::MARKED)),
        );
        assert_eq!(
            t.marked_positions(Address::from_parts(Level::from(1), 0)),
            BTreeSet::from([Position::from(1)])
        );

        let t0 = parent(t.clone(), t);
        assert_eq!(
            t0.marked_positions(Address::from_parts(Level::from(2), 1)),
            BTreeSet::from([Position::from(5), Position::from(7)])
        );
    }

    #[test]
    fn tree_prune() {
        let t: PrunableTree<String> = parent(
            leaf(("a".to_string(), RetentionFlags::EPHEMERAL)),
            leaf(("b".to_string(), RetentionFlags::EPHEMERAL)),
        );

        assert_eq!(
            t.clone().prune(Level::from(1)),
            leaf(("ab".to_string(), RetentionFlags::EPHEMERAL))
        );

        let t0 = parent(leaf(("c".to_string(), RetentionFlags::MARKED)), t);
        assert_eq!(
            t0.prune(Level::from(2)),
            parent(
                leaf(("c".to_string(), RetentionFlags::MARKED)),
                leaf(("ab".to_string(), RetentionFlags::EPHEMERAL))
            )
        );
    }

    #[test]
    fn tree_merge_checked() {
        let t0: PrunableTree<String> =
            parent(leaf(("a".to_string(), RetentionFlags::EPHEMERAL)), nil());

        let t1: PrunableTree<String> =
            parent(nil(), leaf(("b".to_string(), RetentionFlags::EPHEMERAL)));

        assert_eq!(
            t0.clone()
                .merge_checked(Address::from_parts(1.into(), 0), t1.clone()),
            Ok(leaf(("ab".to_string(), RetentionFlags::EPHEMERAL)))
        );

        let t2: PrunableTree<String> =
            parent(leaf(("c".to_string(), RetentionFlags::EPHEMERAL)), nil());
        assert_eq!(
            t0.clone()
                .merge_checked(Address::from_parts(1.into(), 0), t2.clone()),
            Err(Address::from_parts(0.into(), 0))
        );

        let t3: PrunableTree<String> = parent(t0, t2);
        let t4: PrunableTree<String> = parent(t1.clone(), t1);

        assert_eq!(
            t3.merge_checked(Address::from_parts(2.into(), 0), t4),
            Ok(leaf(("abcb".to_string(), RetentionFlags::EPHEMERAL)))
        );
    }

    #[test]
    fn located_tree() {
        let l = parent(str_leaf("a"), str_leaf("b"));
        let r = parent(str_leaf("c"), str_leaf("d"));

        let t: LocatedTree<(), String> = LocatedTree {
            root_addr: Address::from_parts(2.into(), 1),
            root: parent(l.clone(), r.clone()),
        };

        assert_eq!(t.max_position(), Some(7.into()));
        assert_eq!(t.value_at_position(5.into()), Some(&"b".to_string()));
        assert_eq!(t.value_at_position(8.into()), None);
        assert_eq!(t.subtree(Address::from_parts(0.into(), 1)), None);
        assert_eq!(t.subtree(Address::from_parts(3.into(), 0)), None);

        let subtree_addr = Address::from_parts(1.into(), 3);
        assert_eq!(
            t.subtree(subtree_addr),
            Some(LocatedTree {
                root_addr: subtree_addr,
                root: r.clone()
            })
        );

        assert_eq!(
            t.decompose_to_level(1.into()),
            vec![
                LocatedTree {
                    root_addr: Address::from_parts(1.into(), 2),
                    root: l,
                },
                LocatedTree {
                    root_addr: Address::from_parts(1.into(), 3),
                    root: r,
                }
            ]
        );
    }

    #[test]
    fn located_prunable_tree_insert() {
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

    #[test]
    fn shardtree_insertion() {
        let mut tree: ShardTree<MemoryShardStore<String, usize>, 4, 3> =
            ShardTree::new(MemoryShardStore::empty(), 100);

        assert_matches!(
            tree.batch_insert(
                Position::from(1),
                vec![
                    ("b".to_string(), Retention::Checkpoint { id: 1, is_marked: false }),
                    ("c".to_string(), Retention::Ephemeral),
                    ("d".to_string(), Retention::Marked),
                ].into_iter()
            ),
            Ok(Some((pos, incomplete))) if
                pos == Position::from(3) &&
                incomplete == vec![
                    IncompleteAt {
                        address: Address::from_parts(Level::from(0), 0),
                        required_for_witness: true
                    },
                    IncompleteAt {
                        address: Address::from_parts(Level::from(2), 1),
                        required_for_witness: true
                    }
                ]
        );

        assert_matches!(
            tree.root_at_checkpoint(1),
            Err(ShardTreeError::Query(QueryError::TreeIncomplete(v))) if v == vec![Address::from_parts(Level::from(0), 0)]
        );

        assert_matches!(
            tree.batch_insert(
                Position::from(0),
                vec![
                    ("a".to_string(), Retention::Ephemeral),
                ].into_iter()
            ),
            Ok(Some((pos, incomplete))) if
                pos == Position::from(0) &&
                incomplete == vec![]
        );

        assert_matches!(
            tree.root_at_checkpoint(0),
            Ok(h) if h == *"abcd____________"
        );

        assert_matches!(
            tree.root_at_checkpoint(1),
            Ok(h) if h == *"ab______________"
        );

        assert_matches!(
            tree.batch_insert(
                Position::from(10),
                vec![
                    ("k".to_string(), Retention::Ephemeral),
                    ("l".to_string(), Retention::Checkpoint { id: 2, is_marked: false }),
                    ("m".to_string(), Retention::Ephemeral),
                ].into_iter()
            ),
            Ok(Some((pos, incomplete))) if
                pos == Position::from(12) &&
                incomplete == vec![
                    IncompleteAt {
                        address: Address::from_parts(Level::from(0), 13),
                        required_for_witness: false
                    },
                    IncompleteAt {
                        address: Address::from_parts(Level::from(1), 7),
                        required_for_witness: false
                    },
                    IncompleteAt {
                        address: Address::from_parts(Level::from(1), 4),
                        required_for_witness: false
                    },
                ]
        );

        assert_matches!(
            tree.root_at_checkpoint(0),
            // The (0, 13) and (1, 7) incomplete subtrees are
            // not considered incomplete here because they appear
            // at the tip of the tree.
            Err(ShardTreeError::Query(QueryError::TreeIncomplete(xs))) if xs == vec![
                Address::from_parts(Level::from(2), 1),
                Address::from_parts(Level::from(1), 4),
            ]
        );

        assert_matches!(tree.truncate_to_depth(1), Ok(true));
    }

    #[test]
    fn shard_sizes() {
        let mut tree: ShardTree<MemoryShardStore<String, usize>, 4, 2> =
            ShardTree::new(MemoryShardStore::empty(), 100);
        for c in 'a'..'p' {
            tree.append(c.to_string(), Retention::Ephemeral).unwrap();
        }

        assert_eq!(tree.store.shards.len(), 4);
        assert_eq!(
            tree.store.shards[3].max_position(),
            Some(Position::from(14))
        );
    }

    #[test]
    fn append() {
        check_append(|m| {
            ShardTree::<MemoryShardStore<String, usize>, 4, 3>::new(MemoryShardStore::empty(), m)
        });
    }

    #[test]
    fn root_hashes() {
        check_root_hashes(|m| {
            ShardTree::<MemoryShardStore<String, usize>, 4, 3>::new(MemoryShardStore::empty(), m)
        });
    }

    #[test]
    fn witnesses() {
        check_witnesses(|m| {
            ShardTree::<MemoryShardStore<String, usize>, 4, 3>::new(MemoryShardStore::empty(), m)
        });
    }

    #[test]
    fn checkpoint_rewind() {
        check_checkpoint_rewind(|m| {
            ShardTree::<MemoryShardStore<String, usize>, 4, 3>::new(MemoryShardStore::empty(), m)
        });
    }

    #[test]
    fn rewind_remove_mark() {
        check_rewind_remove_mark(|m| {
            ShardTree::<MemoryShardStore<String, usize>, 4, 3>::new(MemoryShardStore::empty(), m)
        });
    }

    // Combined tree tests
    #[allow(clippy::type_complexity)]
    fn new_combined_tree<H: Hashable + Ord + Clone + core::fmt::Debug>(
        max_checkpoints: usize,
    ) -> CombinedTree<
        H,
        usize,
        CompleteTree<H, usize, 4>,
        ShardTree<MemoryShardStore<H, usize>, 4, 3>,
    > {
        CombinedTree::new(
            CompleteTree::new(max_checkpoints),
            ShardTree::new(MemoryShardStore::empty(), max_checkpoints),
        )
    }

    #[test]
    fn combined_append() {
        check_append(new_combined_tree);
    }

    #[test]
    fn combined_rewind_remove_mark() {
        check_rewind_remove_mark(new_combined_tree);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100000))]

        #[test]
        fn check_randomized_u64_ops(
            ops in proptest::collection::vec(
                arb_operation(
                    (0..32u64).prop_map(SipHashable),
                    (0u64..100).prop_map(Position::from)
                ),
                1..100
            )
        ) {
            let tree = new_combined_tree(100);
            let indexed_ops = ops.iter().enumerate().map(|(i, op)| op.map_checkpoint_id(|_| i)).collect::<Vec<_>>();
            check_operations(tree, &indexed_ops)?;
        }

        #[test]
        fn check_randomized_str_ops(
            ops in proptest::collection::vec(
                arb_operation(
                    (97u8..123).prop_map(|c| char::from(c).to_string()),
                    (0u64..100).prop_map(Position::from)
                ),
                1..100
            )
        ) {
            let tree = new_combined_tree(100);
            let indexed_ops = ops.iter().enumerate().map(|(i, op)| op.map_checkpoint_id(|_| i)).collect::<Vec<_>>();
            check_operations(tree, &indexed_ops)?;
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn check_shardtree_caching(
            (mut tree, _, marked_positions) in arb_shardtree(arb_char_str())
        ) {
            if let Some(max_leaf_pos) = tree.max_leaf_position(0).unwrap() {
                let max_complete_addr = Address::above_position(max_leaf_pos.root_level(), max_leaf_pos);
                let root = tree.root(max_complete_addr, max_leaf_pos + 1);
                let caching_root = tree.root_caching(max_complete_addr, max_leaf_pos + 1);
                assert_matches!(root, Ok(_));
                assert_eq!(root, caching_root);

                for pos in marked_positions {
                    let witness = tree.witness(pos, 0);
                    let caching_witness = tree.witness_caching(pos, 0);
                    assert_matches!(witness, Ok(_));
                    assert_eq!(witness, caching_witness);
                }
            }
        }
    }

    #[test]
    fn insert_frontier_nodes() {
        let mut frontier = NonEmptyFrontier::new("a".to_string());
        for c in 'b'..'z' {
            frontier.append(c.to_string());
        }

        let root_addr = Address::from_parts(Level::from(4), 1);
        let tree = LocatedPrunableTree::empty(root_addr);
        let result = tree.insert_frontier_nodes::<()>(frontier.clone(), &Retention::Ephemeral);
        assert_matches!(result, Ok(_));

        let mut tree1 = LocatedPrunableTree::empty(root_addr);
        for c in 'q'..'z' {
            let (t, _, _) = tree1
                .append::<()>(c.to_string(), Retention::Ephemeral)
                .unwrap();
            tree1 = t;
        }
        assert_matches!(
            tree1.insert_frontier_nodes::<()>(frontier.clone(), &Retention::Ephemeral),
            Ok(t) if t == result.unwrap()
        );

        let mut tree2 = LocatedPrunableTree::empty(root_addr);
        for c in 'a'..'i' {
            let (t, _, _) = tree2
                .append::<()>(c.to_string(), Retention::Ephemeral)
                .unwrap();
            tree2 = t;
        }
        assert_matches!(
            tree2.insert_frontier_nodes::<()>(frontier, &Retention::Ephemeral),
            Err(InsertionError::Conflict(_))
        );
    }

    #[test]
    #[cfg(feature = "legacy-api")]
    fn insert_witness_nodes() {
        let mut base_tree = CommitmentTree::<String, 6>::empty();
        for c in 'a'..'h' {
            base_tree.append(c.to_string()).unwrap();
        }
        let mut witness = IncrementalWitness::from_tree(base_tree);
        for c in 'h'..'z' {
            witness.append(c.to_string()).unwrap();
        }

        let root_addr = Address::from_parts(Level::from(3), 0);
        let tree = LocatedPrunableTree::empty(root_addr);
        let result = tree.insert_witness_nodes(witness, 3usize);
        assert_matches!(result, Ok((ref _t, Some(ref _c), Some(ref _r))));

        if let Ok((t, Some(c), Some(r))) = result {
            // verify that we can find the "marked" leaf
            assert_eq!(
                t.root.root_hash(root_addr, Position::from(7)),
                Ok("abcdefg_".to_string())
            );

            assert_eq!(
                c.root,
                Tree::parent(
                    None,
                    Tree::parent(
                        None,
                        Tree::empty(),
                        Tree::leaf(("ijklmnop".to_string(), RetentionFlags::EPHEMERAL)),
                    ),
                    Tree::parent(
                        None,
                        Tree::leaf(("qrstuvwx".to_string(), RetentionFlags::EPHEMERAL)),
                        Tree::empty()
                    )
                )
            );

            assert_eq!(
                r.root
                    .root_hash(Address::from_parts(Level::from(3), 3), Position::from(25)),
                Ok("y_______".to_string())
            );
        }
    }
}
