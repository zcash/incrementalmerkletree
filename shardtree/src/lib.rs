use core::convert::Infallible;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{BitAnd, BitOr, Deref, Not, Range};
use either::Either;
use std::collections::{BTreeMap, BTreeSet};
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

/// A binary Merkle tree with its root at the given address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocatedTree<A, V> {
    root_addr: Address,
    root: Tree<A, V>,
}

impl<A, V> LocatedTree<A, V> {
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
    pub fn incomplete(&self) -> Vec<Address> {
        self.root.incomplete(self.root_addr)
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

type LocatedPrunableTree<H> = LocatedTree<Option<Rc<H>>, (H, RetentionFlags)>;

/// A data structure describing the nature of a [`Node::Nil`] node in the tree that was introduced
/// as the consequence of an insertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncompleteAt {
    /// The address of the empty node.
    pub address: Address,
    /// A flag identifying whether or not the missing node is required in order to construct a
    /// witness for a node with [`MARKED`] retention.
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
    /// each inserted leaf with [`MARKED`] retention.
    pub incomplete: Vec<IncompleteAt>,
    /// The maximum position at which a leaf was inserted.
    pub max_insert_position: Option<Position>,
    /// The positions of all leaves with [`CHECKPOINT`] retention that were inserted.
    pub checkpoints: BTreeMap<C, Position>,
    /// The unconsumed remainder of the iterator from which leaves were inserted, if the tree
    /// was completely filled before the iterator was fully consumed.
    pub remainder: I,
}

/// An error prevented the insertion of values into the subtree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InsertionError<S> {
    /// The caller attempted to insert a subtree into a tree that does not contain
    /// the subtree's root address.
    NotContained,
    /// The start of the range of positions provided for insertion is not included
    /// in the range of positions within this subtree.
    OutOfRange(Range<Position>),
    /// An existing root hash conflicts with the root hash of a node being inserted.
    Conflict(Address),
    /// An out-of-order checkpoint was detected
    ///
    /// Checkpoint identifiers must be in nondecreasing order relative to tree positions.
    CheckpointOutOfOrder,
    /// An append operation has exceeded the capacity of the tree.
    TreeFull,
    /// An error was produced by the underlying [`ShardStore`]
    Storage(S),
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
    /// This tree will be truncated to the `truncate_at` position, and then empty
    /// empty roots corresponding to later positions will be filled by [`H::empty_root`].
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
                            // can fill it with the empty leaf, but only if `fill_start` is None or
                            // it is located at `position + 1`.
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
    /// The leaf at the specified position is retained.
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
    pub fn insert_subtree<E>(
        &self,
        subtree: Self,
        contains_marked: bool,
    ) -> Result<(Self, Vec<IncompleteAt>), InsertionError<E>> {
        // A function to recursively dig into the tree, creating a path downward and introducing
        // empty nodes as necessary until we can insert the provided subtree.
        #[allow(clippy::type_complexity)]
        fn go<H: Hashable + Clone + PartialEq, E>(
            root_addr: Address,
            into: &PrunableTree<H>,
            subtree: LocatedPrunableTree<H>,
            is_complete: bool,
            contains_marked: bool,
        ) -> Result<(PrunableTree<H>, Vec<IncompleteAt>), InsertionError<E>> {
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
            let complete = subtree.root.reduce(&is_complete);
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
            Err(InsertionError::NotContained)
        }
    }

    /// Append a single value at the first available position in the tree.
    ///
    /// Prefer to use [`Self::batch_append`] or [`Self::batch_insert`] when appending multiple
    /// values, as these operations require fewer traversals of the tree than are necessary when
    /// performing multiple sequential calls to [`Self::append`].
    pub fn append<C: Clone + Ord, E>(
        &self,
        value: H,
        retention: Retention<C>,
    ) -> Result<(Self, Position, Option<C>), InsertionError<E>> {
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
    pub fn batch_append<C: Clone + Ord, I: Iterator<Item = (H, Retention<C>)>, E>(
        &self,
        values: I,
    ) -> Result<Option<BatchInsertionResult<H, C, I>>, InsertionError<E>> {
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
    /// * `prune_below` - Nodes with [`EPHEMERAL`] retention that are not required to be retained
    ///   in order to construct a witness for a marked node or to make it possible to rewind to a
    ///   checkpointed node may be pruned so long as their address is at less than the specified
    ///   level.
    /// * `values` The iterator of `(H, Retention)` pairs from which to construct the tree.
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
    pub fn batch_insert<C: Clone + Ord, I: Iterator<Item = (H, Retention<C>)>, E>(
        &self,
        start: Position,
        values: I,
    ) -> Result<Option<BatchInsertionResult<H, C, I>>, InsertionError<E>> {
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
            Err(InsertionError::OutOfRange(subtree_range))
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
/// All fragment subtrees must have roots at level `SHARD_HEIGHT - 1`
pub trait ShardStore<H> {
    type Error;

    /// Returns the subtree at the given root address, if any such subtree exists.
    fn get_shard(&self, shard_root: Address) -> Option<&LocatedPrunableTree<H>>;

    /// Returns the subtree containing the maximum inserted leaf position.
    fn last_shard(&self) -> Option<&LocatedPrunableTree<H>>;

    /// Inserts or replaces the subtree having the same root address as the provided tree.
    ///
    /// Implementations of this method MUST enforce the constraint that the root address
    /// of the provided subtree has level `SHARD_HEIGHT - 1`.
    fn put_shard(&mut self, subtree: LocatedPrunableTree<H>) -> Result<(), Self::Error>;

    /// Returns the vector of addresses corresponding to the roots of subtrees stored in this
    /// store.
    fn get_shard_roots(&self) -> Vec<Address>;

    /// Removes subtrees from the underlying store having root addresses at indices greater
    /// than or equal to that of the specified address.
    ///
    /// Implementations of this method MUST enforce the constraint that the root address
    /// provided has level `SHARD_HEIGHT - 1`.
    fn truncate(&mut self, from: Address) -> Result<(), Self::Error>;
}

impl<H> ShardStore<H> for Vec<LocatedPrunableTree<H>> {
    type Error = Infallible;

    fn get_shard(&self, shard_root: Address) -> Option<&LocatedPrunableTree<H>> {
        self.get(shard_root.index())
    }

    fn last_shard(&self) -> Option<&LocatedPrunableTree<H>> {
        self.last()
    }

    fn put_shard(&mut self, subtree: LocatedPrunableTree<H>) -> Result<(), Self::Error> {
        let subtree_addr = subtree.root_addr;
        for subtree_idx in self.last().map_or(0, |s| s.root_addr.index() + 1)..=subtree_addr.index()
        {
            self.push(LocatedTree {
                root_addr: Address::from_parts(subtree_addr.level(), subtree_idx),
                root: Tree(Node::Nil),
            })
        }
        self[subtree_addr.index()] = subtree;
        Ok(())
    }

    fn get_shard_roots(&self) -> Vec<Address> {
        self.iter().map(|s| s.root_addr).collect()
    }

    fn truncate(&mut self, from: Address) -> Result<(), Self::Error> {
        self.truncate(from.index());
        Ok(())
    }
}

/// A left-dense, sparse binary Merkle tree of the specified depth, represented as a vector of
/// subtrees (shards) of the given maximum height.
///
/// This tree maintains a collection of "checkpoints" which represent positions, usually near the
/// front of the tree, that are maintained such that it's possible to truncate nodes to the right
/// of the specified position.
#[derive(Debug)]
pub struct ShardTree<H, C: Ord, S: ShardStore<H>, const DEPTH: u8, const SHARD_HEIGHT: u8> {
    /// The vector of tree shards.
    store: S,
    /// The maximum number of checkpoints to retain before pruning.
    max_checkpoints: usize,
    /// A map from position to the count of checkpoints at this position.
    checkpoints: BTreeMap<C, Checkpoint>,
    // /// A tree that is used to cache the known roots of subtrees in the "cap" of nodes between
    // /// `SHARD_HEIGHT` and `DEPTH` that are otherwise not directly represented in the tree.  This
    // /// cache is automatically updated when computing roots and witnesses. Leaf nodes are empty
    // /// because the annotation slot is consistently used to store the subtree hashes at each node.
    // cap_cache: Tree<Option<Rc<H>>, ()>
    _hash_type: PhantomData<H>,
}

impl<
        H: Hashable + Clone + PartialEq,
        C: Clone + Ord + core::fmt::Debug,
        S: ShardStore<H>,
        const DEPTH: u8,
        const SHARD_HEIGHT: u8,
    > ShardTree<H, C, S, DEPTH, SHARD_HEIGHT>
{
    /// Creates a new empty tree.
    pub fn new(store: S, max_checkpoints: usize, initial_checkpoint_id: C) -> Self {
        Self {
            store,
            max_checkpoints,
            checkpoints: BTreeMap::from([(initial_checkpoint_id, Checkpoint::tree_empty())]),
            //cap_cache: Tree(None, ())
            _hash_type: PhantomData,
        }
    }

    /// Returns the root address of the tree.
    pub fn root_addr() -> Address {
        Address::from_parts(Level::from(DEPTH), 0)
    }

    /// Returns the fixed level of subtree roots within the vector of subtrees used as this tree's
    /// representation.
    pub fn subtree_level() -> Level {
        Level::from(SHARD_HEIGHT - 1)
    }

    /// Returns the position and checkpoint count for each checkpointed position in the tree.
    pub fn checkpoints(&self) -> &BTreeMap<C, Checkpoint> {
        &self.checkpoints
    }

    /// Returns the leaf value at the specified position, if it is a marked leaf.
    pub fn get_marked_leaf(&self, position: Position) -> Option<&H> {
        self.store
            .get_shard(Address::above_position(Self::subtree_level(), position))
            .and_then(|t| t.value_at_position(position))
            .and_then(|(v, r)| if r.is_marked() { Some(v) } else { None })
    }

    /// Returns the positions of marked leaves in the tree.
    pub fn marked_positions(&self) -> BTreeSet<Position> {
        let mut result = BTreeSet::new();
        for subtree_addr in &self.store.get_shard_roots() {
            if let Some(subtree) = self.store.get_shard(*subtree_addr) {
                result.append(&mut subtree.marked_positions());
            }
        }
        result
    }

    /// Inserts a new root into the tree at the given address.
    ///
    /// This will pad from the left until the tree's subtrees vector contains enough trees to reach
    /// the specified address, which must be at the [`Self::subtree_level`] level. If a subtree
    /// already exists at this address, its root will be annotated with the specified hash value.
    ///
    /// This will return an error if the specified hash conflicts with any existing annotation.
    pub fn put_root(&mut self, addr: Address, value: H) -> Result<(), InsertionError<S::Error>> {
        let updated_subtree = match self.store.get_shard(addr) {
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
                        value: (value, EPHEMERAL),
                    }),
                }))
            }
        }?;

        if let Some(s) = updated_subtree {
            self.store.put_shard(s).map_err(InsertionError::Storage)?;
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
    ) -> Result<(), InsertionError<S::Error>> {
        if let Retention::Checkpoint { id, .. } = &retention {
            if self.checkpoints.keys().last() >= Some(id) {
                return Err(InsertionError::CheckpointOutOfOrder);
            }
        }

        let (append_result, position, checkpoint_id) =
            if let Some(subtree) = self.store.last_shard() {
                if subtree.root.reduce(&is_complete) {
                    let addr = subtree.root_addr;

                    if addr.index() + 1 >= 0x1 << (SHARD_HEIGHT - 1) {
                        return Err(InsertionError::OutOfRange(addr.position_range()));
                    } else {
                        LocatedTree::empty(addr.next_at_level()).append(value, retention)?
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
            .map_err(InsertionError::Storage)?;
        if let Some(c) = checkpoint_id {
            self.checkpoints
                .insert(c, Checkpoint::at_position(position));
        }

        self.prune_excess_checkpoints()
            .map_err(InsertionError::Storage)?;

        Ok(())
    }

    fn prune_excess_checkpoints(&mut self) -> Result<(), S::Error> {
        if self.checkpoints.len() > self.max_checkpoints {
            // Batch removals by subtree & create a list of the checkpoint identifiers that
            // will be removed from the checkpoints map.
            let mut checkpoints_to_delete = vec![];
            let mut clear_positions: BTreeMap<Address, BTreeMap<Position, RetentionFlags>> =
                BTreeMap::new();
            for (cid, checkpoint) in self
                .checkpoints
                .iter()
                .take(self.checkpoints.len() - self.max_checkpoints)
            {
                checkpoints_to_delete.push(cid.clone());

                // clear the checkpoint leaf
                if let TreeState::AtPosition(pos) = checkpoint.tree_state {
                    let subtree_addr = Address::above_position(Self::subtree_level(), pos);
                    clear_positions
                        .entry(subtree_addr)
                        .and_modify(|to_clear| {
                            to_clear
                                .entry(pos)
                                .and_modify(|flags| *flags = *flags | CHECKPOINT)
                                .or_insert(CHECKPOINT);
                        })
                        .or_insert_with(|| BTreeMap::from([(pos, CHECKPOINT)]));
                }

                // clear the leaves that have been marked for removal
                for unmark_pos in checkpoint.marks_removed.iter() {
                    let subtree_addr = Address::above_position(Self::subtree_level(), *unmark_pos);
                    clear_positions
                        .entry(subtree_addr)
                        .and_modify(|to_clear| {
                            to_clear
                                .entry(*unmark_pos)
                                .and_modify(|flags| *flags = *flags | MARKED)
                                .or_insert(MARKED);
                        })
                        .or_insert_with(|| BTreeMap::from([(*unmark_pos, MARKED)]));
                }
            }

            // Prune each affected subtree
            for (subtree_addr, positions) in clear_positions.into_iter() {
                let cleared = self
                    .store
                    .get_shard(subtree_addr)
                    .map(|subtree| subtree.clear_flags(positions));
                if let Some(cleared) = cleared {
                    self.store.put_shard(cleared)?;
                }
            }

            // Now that the leaves have been pruned, actually remove the checkpoints
            for c in checkpoints_to_delete {
                self.checkpoints.remove(&c);
            }
        }

        Ok(())
    }

    /// Returns the position of the checkpoint, if any, along with the number of subsequent
    /// checkpoints at the same position. Returns `None` if `checkpoint_depth == 0` or if
    /// insufficient checkpoints exist to seek back to the requested depth.
    pub fn checkpoint_at_depth(&self, checkpoint_depth: usize) -> Option<(&C, &Checkpoint)> {
        if checkpoint_depth == 0 {
            None
        } else {
            self.checkpoints.iter().rev().nth(checkpoint_depth - 1)
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
    ) -> Result<Option<Position>, QueryError> {
        if checkpoint_depth == 0 {
            // TODO: This relies on the invariant that the last shard in the subtrees vector is
            // never created without a leaf then being added to it. However, this may be a
            // difficult invariant to maintain when adding empty roots, so perhaps we need a
            // better way of tracking the actual max position of the tree; we might want to
            // just store it directly.
            Ok(self.store.last_shard().and_then(|t| t.max_position()))
        } else {
            match self.checkpoint_at_depth(checkpoint_depth) {
                Some((_, c)) => Ok(c.position()),
                None => {
                    // There is no checkpoint at the specified depth, so we report it as pruned.
                    Err(QueryError::CheckpointPruned)
                }
            }
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
    use crate::{
        LocatedPrunableTree, LocatedTree, Node, PrunableTree, QueryError, ShardStore, ShardTree,
        Tree, EPHEMERAL, MARKED,
    };
    use core::convert::Infallible;
    use incrementalmerkletree::{
        testing::{self, check_append, complete_tree::CompleteTree, CombinedTree},
        Address, Hashable, Level, Position, Retention,
    };
    use std::collections::BTreeSet;
    use std::rc::Rc;

    fn nil<A, B>() -> Tree<A, B> {
        Tree(Node::Nil)
    }

    fn str_leaf<A>(c: &str) -> Tree<A, String> {
        Tree(Node::Leaf {
            value: c.to_string(),
        })
    }

    fn leaf<A, B>(value: B) -> Tree<A, B> {
        Tree(Node::Leaf { value })
    }

    fn parent<A: Default, B>(left: Tree<A, B>, right: Tree<A, B>) -> Tree<A, B> {
        Tree(Node::Parent {
            ann: A::default(),
            left: Rc::new(left),
            right: Rc::new(right),
        })
    }

    #[test]
    fn tree_incomplete() {
        let t: Tree<(), String> = parent(nil(), str_leaf("a"));
        assert_eq!(
            t.incomplete(Address::from_parts(Level::from(1), 0)),
            vec![Address::from_parts(Level::from(0), 0)]
        );

        let t0 = parent(str_leaf("b"), t.clone());
        assert_eq!(
            t0.incomplete(Address::from_parts(Level::from(2), 1)),
            vec![Address::from_parts(Level::from(0), 6)]
        );

        let t1 = parent(nil(), t);
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
        let t: PrunableTree<String> = parent(
            leaf(("a".to_string(), EPHEMERAL)),
            leaf(("b".to_string(), EPHEMERAL)),
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
                leaf(("abcd".to_string(), EPHEMERAL)),
                parent(nil(), leaf(("gh".to_string(), EPHEMERAL))),
            ),
        };

        assert_eq!(
            t.insert_subtree::<Infallible>(
                LocatedTree {
                    root_addr: Address::from_parts(1.into(), 6),
                    root: parent(leaf(("e".to_string(), MARKED)), nil())
                },
                true
            ),
            Ok((
                LocatedTree {
                    root_addr: Address::from_parts(3.into(), 1),
                    root: parent(
                        leaf(("abcd".to_string(), EPHEMERAL)),
                        parent(
                            parent(leaf(("e".to_string(), MARKED)), nil()),
                            leaf(("gh".to_string(), EPHEMERAL))
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
                leaf(("abcd".to_string(), EPHEMERAL)),
                parent(
                    parent(
                        leaf(("e".to_string(), MARKED)),
                        leaf(("f".to_string(), EPHEMERAL)),
                    ),
                    leaf(("gh".to_string(), EPHEMERAL)),
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

    type VecShardStore<H> = Vec<LocatedPrunableTree<H>>;

    #[test]
    fn tree_marked_positions() {
        let t: PrunableTree<String> = parent(
            leaf(("a".to_string(), EPHEMERAL)),
            leaf(("b".to_string(), MARKED)),
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
            leaf(("a".to_string(), EPHEMERAL)),
            leaf(("b".to_string(), EPHEMERAL)),
        );

        assert_eq!(
            t.clone().prune(Level::from(1)),
            leaf(("ab".to_string(), EPHEMERAL))
        );

        let t0 = parent(leaf(("c".to_string(), MARKED)), t);
        assert_eq!(
            t0.prune(Level::from(2)),
            parent(
                leaf(("c".to_string(), MARKED)),
                leaf(("ab".to_string(), EPHEMERAL))
            )
        );
    }

    #[test]
    fn tree_merge_checked() {
        let t0: PrunableTree<String> = parent(leaf(("a".to_string(), EPHEMERAL)), nil());

        let t1: PrunableTree<String> = parent(nil(), leaf(("b".to_string(), EPHEMERAL)));

        assert_eq!(
            t0.clone()
                .merge_checked(Address::from_parts(1.into(), 0), t1.clone()),
            Ok(leaf(("ab".to_string(), EPHEMERAL)))
        );

        let t2: PrunableTree<String> = parent(leaf(("c".to_string(), EPHEMERAL)), nil());
        assert_eq!(
            t0.clone()
                .merge_checked(Address::from_parts(1.into(), 0), t2.clone()),
            Err(Address::from_parts(0.into(), 0))
        );

        let t3: PrunableTree<String> = parent(t0, t2);
        let t4: PrunableTree<String> = parent(t1.clone(), t1);

        assert_eq!(
            t3.merge_checked(Address::from_parts(2.into(), 0), t4),
            Ok(leaf(("abcb".to_string(), EPHEMERAL)))
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
            .append::<(), Infallible>("a".to_string(), Retention::Ephemeral)
            .unwrap();
        assert_eq!(base.right_filled_root(), Ok("a___".to_string()));

        // Perform an in-order insertion.
        let (in_order, pos, _) = base
            .append::<(), Infallible>("b".to_string(), Retention::Ephemeral)
            .unwrap();
        assert_eq!(pos, 1.into());
        assert_eq!(in_order.right_filled_root(), Ok("ab__".to_string()));

        // On the same tree, perform an out-of-order insertion.
        let out_of_order = base
            .batch_insert::<(), _, Infallible>(
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
                    parent(leaf(("a".to_string(), EPHEMERAL)), nil()),
                    parent(nil(), leaf(("d".to_string(), EPHEMERAL)))
                )
            }
        );

        let complete = out_of_order
            .subtree
            .batch_insert::<(), _, Infallible>(
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

    impl<
            H: Hashable + Ord + Clone,
            C: Clone + Ord + core::fmt::Debug,
            S: ShardStore<H>,
            const DEPTH: u8,
            const SHARD_HEIGHT: u8,
        > testing::Tree<H, C> for ShardTree<H, C, S, DEPTH, SHARD_HEIGHT>
    {
        fn depth(&self) -> u8 {
            DEPTH
        }

        fn append(&mut self, value: H, retention: Retention<C>) -> bool {
            ShardTree::append(self, value, retention).is_ok()
        }

        fn current_position(&self) -> Option<Position> {
            ShardTree::max_leaf_position(self, 0).ok().flatten()
        }

        fn get_marked_leaf(&self, _position: Position) -> Option<&H> {
            todo!()
        }

        fn marked_positions(&self) -> BTreeSet<Position> {
            todo!()
        }

        fn root(&self, _checkpoint_depth: usize) -> Option<H> {
            todo!()
        }

        fn witness(&self, _position: Position, _checkpoint_depth: usize) -> Option<Vec<H>> {
            todo!()
        }

        fn remove_mark(&mut self, _position: Position) -> bool {
            todo!()
        }

        fn checkpoint(&mut self, _checkpoint_id: C) -> bool {
            todo!()
        }

        fn rewind(&mut self) -> bool {
            todo!()
        }
    }

    #[test]
    fn append() {
        check_append(|m| {
            ShardTree::<String, usize, VecShardStore<String>, 4, 3>::new(vec![], m, 0)
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
        ShardTree<H, usize, VecShardStore<H>, 4, 3>,
    > {
        CombinedTree::new(
            CompleteTree::new(max_checkpoints, 0),
            ShardTree::new(vec![], max_checkpoints, 0),
        )
    }

    #[test]
    fn combined_append() {
        check_append(new_combined_tree);
    }
}
