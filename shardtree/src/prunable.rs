use std::collections::BTreeSet;
use std::rc::Rc;

use bitflags::bitflags;
use incrementalmerkletree::{Address, Hashable, Level, Position, Retention};

use crate::{accumulate_result_with, Node, Tree};

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use incrementalmerkletree::{Address, Level, Position};

    use super::{PrunableTree, RetentionFlags};
    use crate::tree::tests::{leaf, nil, parent};

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
}
