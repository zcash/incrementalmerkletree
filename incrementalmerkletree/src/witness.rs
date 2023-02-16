use std::convert::TryInto;
use std::iter::repeat;

use crate::{
    frontier::{CommitmentTree, PathFiller},
    Hashable, Level,
};

/// A path from a position in a particular commitment tree to the root of that tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerklePath<H, const DEPTH: u8> {
    auth_path: Vec<(H, bool)>,
    position: u64,
}

impl<H, const DEPTH: u8> MerklePath<H, DEPTH> {
    pub fn auth_path(&self) -> &[(H, bool)] {
        &self.auth_path
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    /// Constructs a Merkle path directly from a path and position.
    pub fn from_path(auth_path: Vec<(H, bool)>, position: u64) -> Result<Self, ()> {
        if auth_path.len() == usize::from(DEPTH) {
            Ok(MerklePath {
                auth_path,
                position,
            })
        } else {
            Err(())
        }
    }
}

impl<H: Hashable, const DEPTH: u8> MerklePath<H, DEPTH> {
    /// Returns the root of the tree corresponding to this path applied to `leaf`.
    pub fn root(&self, leaf: H) -> H {
        self.auth_path
            .iter()
            .enumerate()
            .fold(leaf, |root, (i, (p, leaf_is_on_right))| {
                let level = u8::try_from(i)
                    .expect("Parents list length may not exceed what is representable by an u8")
                    .into();
                match leaf_is_on_right {
                    false => H::combine(level, &root, p),
                    true => H::combine(level, p, &root),
                }
            })
    }
}

/// An updatable witness to a path from a position in a particular [`CommitmentTree`].
///
/// Appending the same commitments in the same order to both the original
/// [`CommitmentTree`] and this `IncrementalWitness` will result in a witness to the path
/// from the target position to the root of the updated tree.
///
/// # Examples
///
/// ```
/// use incrementalmerkletree::{
///     frontier::{CommitmentTree, testing::TestNode},
///     witness::IncrementalWitness,
/// };
///
/// let mut tree = CommitmentTree::<TestNode, 8>::empty();
///
/// tree.append(TestNode(0));
/// tree.append(TestNode(1));
/// let mut witness = IncrementalWitness::from_tree(tree.clone());
/// assert_eq!(witness.position(), 1);
/// assert_eq!(tree.root(), witness.root());
///
/// let next = TestNode(2);
/// tree.append(next.clone());
/// witness.append(next);
/// assert_eq!(tree.root(), witness.root());
/// ```
#[derive(Clone, Debug)]
pub struct IncrementalWitness<H, const DEPTH: u8> {
    tree: CommitmentTree<H, DEPTH>,
    filled: Vec<H>,
    cursor_depth: u8,
    cursor: Option<CommitmentTree<H, DEPTH>>,
}

impl<H, const DEPTH: u8> IncrementalWitness<H, DEPTH> {
    /// Creates an `IncrementalWitness` for the most recent commitment added to the given
    /// [`CommitmentTree`].
    pub fn from_tree(tree: CommitmentTree<H, DEPTH>) -> IncrementalWitness<H, DEPTH> {
        IncrementalWitness {
            tree,
            filled: vec![],
            cursor_depth: 0,
            cursor: None,
        }
    }

    /// Returns the position of the witnessed leaf node in the commitment tree.
    pub fn position(&self) -> usize {
        self.tree.size() - 1
    }

    /// Finds the next "depth" of an unfilled subtree.
    fn next_depth(&self) -> u8 {
        let mut skip: u8 = self
            .filled
            .len()
            .try_into()
            .expect("Merkle tree depths may not exceed the bounds of a u8");

        if self.tree.left.is_none() {
            if skip > 0 {
                skip -= 1;
            } else {
                return 0;
            }
        }

        if self.tree.right.is_none() {
            if skip > 0 {
                skip -= 1;
            } else {
                return 0;
            }
        }

        let mut d = 1;
        for p in &self.tree.parents {
            if p.is_none() {
                if skip > 0 {
                    skip -= 1;
                } else {
                    return d;
                }
            }
            d += 1;
        }

        d + skip
    }
}

impl<H: Hashable + Clone, const DEPTH: u8> IncrementalWitness<H, DEPTH> {
    fn filler(&self) -> PathFiller<H> {
        let cursor_root = self
            .cursor
            .as_ref()
            .map(|c| c.root_inner(self.cursor_depth, PathFiller::empty()));

        PathFiller {
            queue: self.filled.iter().cloned().chain(cursor_root).collect(),
        }
    }

    /// Tracks a leaf node that has been added to the underlying tree.
    ///
    /// Returns an error if the tree is full.
    #[allow(clippy::result_unit_err)]
    pub fn append(&mut self, node: H) -> Result<(), ()> {
        if let Some(mut cursor) = self.cursor.take() {
            cursor.append(node).expect("cursor should not be full");
            if cursor.is_complete(self.cursor_depth) {
                self.filled
                    .push(cursor.root_inner(self.cursor_depth, PathFiller::empty()));
            } else {
                self.cursor = Some(cursor);
            }
        } else {
            self.cursor_depth = self.next_depth();
            if self.cursor_depth >= DEPTH {
                // Tree is full
                return Err(());
            }

            if self.cursor_depth == 0 {
                self.filled.push(node);
            } else {
                let mut cursor = CommitmentTree::empty();
                cursor.append(node).expect("cursor should not be full");
                self.cursor = Some(cursor);
            }
        }

        Ok(())
    }

    /// Returns the current root of the tree corresponding to the witness.
    pub fn root(&self) -> H {
        self.root_inner(DEPTH)
    }

    fn root_inner(&self, depth: u8) -> H {
        self.tree.root_inner(depth, self.filler())
    }

    /// Returns the current witness, or None if the tree is empty.
    pub fn path(&self) -> Option<MerklePath<H, DEPTH>> {
        self.path_inner(DEPTH)
    }

    fn path_inner(&self, depth: u8) -> Option<MerklePath<H, DEPTH>> {
        let mut filler = self.filler();
        let mut auth_path = Vec::new();

        if let Some(node) = &self.tree.left {
            if self.tree.right.is_some() {
                auth_path.push((node.clone(), true));
            } else {
                auth_path.push((filler.next(0.into()), false));
            }
        } else {
            // Can't create an authentication path for the beginning of the tree
            return None;
        }

        for (i, p) in self
            .tree
            .parents
            .iter()
            .chain(repeat(&None))
            .take((depth - 1).into())
            .enumerate()
        {
            auth_path.push(match p {
                Some(node) => (node.clone(), true),
                None => (filler.next(Level::from((i + 1) as u8)), false),
            });
        }

        assert_eq!(auth_path.len(), usize::from(depth));

        MerklePath::from_path(auth_path, self.position() as u64).ok()
    }
}
