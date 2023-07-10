use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::rc::Rc;

use bitflags::bitflags;
use incrementalmerkletree::{
    frontier::NonEmptyFrontier, Address, Hashable, Level, Position, Retention,
};
use tracing::trace;

use crate::{LocatedTree, Node, Tree};

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
                    if vl.0 == vr.0 {
                        // Merge the flags together.
                        Ok(Tree(Node::Leaf {
                            value: (vl.0, vl.1 | vr.1),
                        }))
                    } else {
                        trace!(left = ?vl.0, right = ?vr.0, "Merge conflict for leaves");
                        Err(addr)
                    }
                }
                (Tree(Node::Leaf { value }), parent @ Tree(Node::Parent { .. }))
                | (parent @ Tree(Node::Parent { .. }), Tree(Node::Leaf { value })) => {
                    let parent_hash = parent.root_hash(addr, no_default_fill);
                    if parent_hash.iter().all(|r| r == &value.0) {
                        Ok(parent.reannotate_root(Some(Rc::new(value.0))))
                    } else {
                        trace!(leaf = ?value, node = ?parent_hash, "Merge conflict for leaf into node");
                        Err(addr)
                    }
                }
                (lparent, rparent) => {
                    let lroot = lparent.root_hash(addr, no_default_fill).ok();
                    let rroot = rparent.root_hash(addr, no_default_fill).ok();
                    // If both parents share the same root hash (or if one of them is absent),
                    // they can be merged
                    if lroot.iter().zip(&rroot).all(|(l, r)| l == r) {
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
                        trace!(left = ?lroot, right = ?rroot, "Merge conflict for nodes");
                        Err(addr)
                    }
                }
            }
        }

        trace!(this = ?self, other = ?other, "Merging subtrees");
        go(root_addr, self, other)
    }

    /// Unite two nodes by either constructing a new parent node, or, if both nodes are ephemeral
    /// leaves or Nil, constructing a replacement root by hashing leaf values together (or a
    /// replacement `Nil` value).
    ///
    /// `level` must be the level of the two nodes that are being joined.
    pub(crate) fn unite(level: Level, ann: Option<Rc<H>>, left: Self, right: Self) -> Self {
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

impl std::error::Error for InsertionError {}

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

impl std::error::Error for QueryError {}

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
                        if position.is_right_child() {
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

            trace!(
                "Node at {:?} contains subtree at {:?}",
                root_addr,
                subtree.root_addr(),
            );
            match into {
                Tree(Node::Nil) => Ok(replacement(None, subtree)),
                Tree(Node::Leaf { value: (value, _) }) => {
                    if root_addr == subtree.root_addr {
                        if is_complete {
                            // It is safe to replace the existing root unannotated, because we
                            // can always recompute the root from a complete subtree.
                            Ok((subtree.root, vec![]))
                        } else if subtree.root.node_value().iter().all(|v| v == &value) {
                            Ok((
                                // at this point we statically know the root to be a parent
                                subtree.root.reannotate_root(Some(Rc::new(value.clone()))),
                                vec![],
                            ))
                        } else {
                            trace!(
                                cur_root = ?value,
                                new_root = ?subtree.root.node_value(),
                                "Insertion conflict",
                            );
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
        trace!(
            position_range = ?position_range,
            prune_below = ?prune_below,
            "Creating minimal tree for insertion"
        );

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
            assert_eq!(lroot.root_addr.parent(), rroot.root_addr.parent());
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

        /// Combines the given subtree with an empty sibling node to obtain the next level
        /// subtree.
        ///
        /// `expect_left_child` is set to a constant at each callsite, to ensure that this
        /// function is only called on either the left-most or right-most subtree.
        fn combine_with_empty<H: Hashable + Clone + PartialEq>(
            root: LocatedPrunableTree<H>,
            expect_left_child: bool,
            incomplete: &mut Vec<IncompleteAt>,
            contains_marked: bool,
            prune_below: Level,
        ) -> LocatedPrunableTree<H> {
            assert_eq!(expect_left_child, root.root_addr.is_left_child());
            let sibling_addr = root.root_addr.sibling();
            incomplete.push(IncompleteAt {
                address: sibling_addr,
                required_for_witness: contains_marked,
            });
            let sibling = LocatedTree {
                root_addr: sibling_addr,
                root: Tree(Node::Nil),
            };
            let (lroot, rroot) = if root.root_addr.is_left_child() {
                (root, sibling)
            } else {
                (sibling, root)
            };
            unite(lroot, rroot, prune_below)
        }

        // Builds a single tree from the provided stack of subtrees, which must be non-overlapping
        // and in position order. Returns the resulting tree, a flag indicating whether the
        // resulting tree contains a `MARKED` node, and the vector of [`IncompleteAt`] values for
        // [`Node::Nil`] nodes that were introduced in the process of constructing the tree.
        fn build_minimal_tree<H: Hashable + Clone + PartialEq>(
            mut xs: Vec<(LocatedPrunableTree<H>, bool)>,
            root_addr: Address,
            prune_below: Level,
        ) -> Option<(LocatedPrunableTree<H>, bool, Vec<IncompleteAt>)> {
            // First, consume the stack from the right, building up a single tree
            // until we can't combine any more.
            if let Some((mut cur, mut contains_marked)) = xs.pop() {
                let mut incomplete = vec![];
                while let Some((top, top_marked)) = xs.pop() {
                    while cur.root_addr.level() < top.root_addr.level() {
                        cur =
                            combine_with_empty(cur, true, &mut incomplete, top_marked, prune_below);
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
                            cur = combine_with_empty(
                                cur,
                                true,
                                &mut incomplete,
                                top_marked,
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

                // Ensure we can work from the left in a single pass by making this right-most subtree
                while cur.root_addr.level() + 1 < root_addr.level() {
                    cur = combine_with_empty(
                        cur,
                        true,
                        &mut incomplete,
                        contains_marked,
                        prune_below,
                    );
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
                                contains_marked = contains_marked || next_marked;
                                prev_tree = combine_with_empty(
                                    prev_tree,
                                    false,
                                    &mut incomplete,
                                    next_marked,
                                    prune_below,
                                );
                            }

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

                if position.is_right_child() {
                    // At right-hand positions, we are completing a subtree and so we unite
                    // fragments up the stack until we get the largest possible subtree
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
        trace!("Initial fragments: {:?}", fragments);

        if position > position_range.start {
            let last_position = position - 1;
            let minimal_tree_addr =
                Address::from(position_range.start).common_ancestor(&last_position.into());
            trace!("Building minimal tree at {:?}", minimal_tree_addr);
            build_minimal_tree(fragments, minimal_tree_addr, prune_below).map(
                |(to_insert, contains_marked, incomplete)| BatchInsertionResult {
                    subtree: to_insert,
                    contains_marked,
                    incomplete,
                    max_insert_position: Some(last_position),
                    checkpoints,
                    remainder: values,
                },
            )
        } else {
            None
        }
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
        trace!("Batch inserting into {:?} from {:?}", self.root_addr, start);
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

    // Constructs a pair of trees that contain the leaf and ommers of the given frontier. The first
    // element of the result is a tree with its root at a level less than or equal to `split_at`;
    // the second element is a tree with its leaves at level `split_at` that is only returned if
    // the frontier contains sufficient data to fill the first tree to the `split_at` level.
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
    pub(crate) fn from_frontier_parts<C>(
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
            if addr.is_left_child() {
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
                while addr.is_left_child() {
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
                        trace!(
                            "In {:?}, partitioned: {:?} {:?}",
                            root_addr,
                            &to_clear[0..p],
                            &to_clear[p..],
                        );
                        Tree::unite(
                            l_addr.level(),
                            ann.clone(),
                            go(&to_clear[0..p], l_addr, left),
                            go(&to_clear[p..], r_addr, right),
                        )
                    }
                    Tree(Node::Leaf { value: (h, r) }) => {
                        trace!("In {:?}, clearing {:?}", root_addr, to_clear);
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use incrementalmerkletree::{Address, Level, Position, Retention};

    use super::{LocatedPrunableTree, PrunableTree, QueryError, RetentionFlags};
    use crate::tree::{
        tests::{leaf, nil, parent},
        LocatedTree,
    };

    #[test]
    fn root() {
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
    fn marked_positions() {
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
    fn prune() {
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
    fn merge_checked() {
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
    fn merge_checked_flags() {
        let t0: PrunableTree<String> = leaf(("a".to_string(), RetentionFlags::EPHEMERAL));
        let t1: PrunableTree<String> = leaf(("a".to_string(), RetentionFlags::MARKED));
        let t2: PrunableTree<String> = leaf(("a".to_string(), RetentionFlags::CHECKPOINT));

        assert_eq!(
            t0.merge_checked(Address::from_parts(1.into(), 0), t1.clone()),
            Ok(t1.clone()),
        );

        assert_eq!(
            t1.merge_checked(Address::from_parts(1.into(), 0), t2),
            Ok(leaf((
                "a".to_string(),
                RetentionFlags::MARKED | RetentionFlags::CHECKPOINT,
            ))),
        );
    }

    #[test]
    fn located_insert_subtree() {
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
    fn located_insert_subtree_leaf_overwrites() {
        let t: LocatedPrunableTree<String> = LocatedTree {
            root_addr: Address::from_parts(2.into(), 1),
            root: parent(leaf(("a".to_string(), RetentionFlags::MARKED)), nil()),
        };

        assert_eq!(
            t.insert_subtree(
                LocatedTree {
                    root_addr: Address::from_parts(1.into(), 2),
                    root: leaf(("b".to_string(), RetentionFlags::EPHEMERAL)),
                },
                false,
            ),
            Ok((
                LocatedTree {
                    root_addr: Address::from_parts(2.into(), 1),
                    root: parent(leaf(("b".to_string(), RetentionFlags::EPHEMERAL)), nil()),
                },
                vec![],
            )),
        );
    }

    #[test]
    fn located_witness() {
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
    fn located_from_iter_non_sibling_adjacent() {
        let res = LocatedPrunableTree::from_iter::<(), _>(
            Position::from(3)..Position::from(5),
            Level::new(0),
            vec![
                ("d".to_string(), Retention::Ephemeral),
                ("e".to_string(), Retention::Ephemeral),
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(
            res.subtree,
            LocatedPrunableTree {
                root_addr: Address::from_parts(3.into(), 0),
                root: parent(
                    parent(
                        nil(),
                        parent(nil(), leaf(("d".to_string(), RetentionFlags::EPHEMERAL)))
                    ),
                    parent(
                        parent(leaf(("e".to_string(), RetentionFlags::EPHEMERAL)), nil()),
                        nil()
                    )
                )
            },
        );
    }

    #[test]
    fn located_insert() {
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
}
