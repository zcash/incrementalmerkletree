# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Tree::get_witnessed_leaf` has been added to allow a user to query for the
  leaf value of witnessed leaves by their position in the tree.

### Changed

Changes to top-level types and traits:

- The `Tree` trait has substantial changes in this release. Hashes are no longer used
  in identifying nodes when generating authentication paths and removing witnesses;
  instead, these operations now use position information exclusively.
- `Tree::authentication_path` and `Tree::remove_witness` have both been changed to only
  take a `position` parameter, instead of both the leaf value and the position.
- `Tree::current_leaf` and `Tree::witness` have both been changed to only return the leaf
  value, instead of both the leaf value and the position.

Changes to the `bridgetree` module:

- The type of `BridgeTree::saved` and `Checkpoint::forgotten` have been changed from 
  `BTreeMap<(Position, H), usize>` to `BTreeMap<Position, usize>`. This change
  is also reflected in the rturn type of the `BridgeTree::witnessed_indices` method.
- The `Checkpoint` type is no longer parameterized by `H`.
- `BridgeTree::bridges` has been split into two parts: 
  - `BridgeTree::prior_bridges` now tracks past bridges not including the current frontier.
  - `BridgeTree::current_bridge` now tracks current mutable frontier.
- The signature of `BridgeTree::from_parts` has been modified to reflect these changes.
- A bug in `BridgeTree::garbage_collect` has been fixed. This bug caused garbage
  collection to in some cases incorrectly rewrite checkpointed bridge lengths, resulting
  in a condition where a rewind could panic after a GC operation.

### Removed

- The `Tree::Recording` associated type and the relate `record`/`play` infrastructure has
  been removed; it was unused and added unnecessary complexity.

## [0.3.0-beta.1] - 2022-03-22

### Added

Additions to top-level types and traits:

- `Position` trait impls
  - `From<Position>` impl for `usize`
  - `Add<usize>` impl for `Position`
  - `AddAssign<usize>` impl for `Position`
  - `TryFrom<u64>` impl for `Position`
- Additions to the `Tree` trait
  - `current_leaf` method returns the position of the latest note appended to the
    tree along with the leaf value, if any.
  - `current_position` method returns the position of the latest note appended to the
    tree, if any.
  - `is_witnessed` method returns a boolean value indicating whether a leaf value
    has been marked for later construction of an authentication path at a specified
    position in the tree.

Additions to the `bridgetree` module:

- Added `MerkleBridge` methods
  - `position` returns the position of the tip of the frontier.
  - `current_leaf` returns the leaf value most recently appended to the frontier.
- Added `Leaf::Value` that returns the value of the leaf by reference.
- `PartialEq` and `Eq` impls for `AuthFragment`, `MerkleBridge`, and `BridgeTreeError`
- A number of constructor and accessor methods for `Checkpoint` have
  been added to facilitate serialization.
- `BridgeTree::garbage_collect` is a new method that prunes data for removed
  witnesses and inaccessable checkpoints beyond the max rewind depth.

### Changed

Changes to top-level types and traits:

- `Position` now wraps a `usize` rather than a `u64`
- The following `Tree` methods now take an additional `Position` argument
  to support cases where leaf values may appear at multiple positions in the tree.
  - `authentication_path` 
  - `remove_witness` 
- The `witness` method now returns a tuple of the witnessed position and leaf hash,
  or `None` if the tree is empty, instead of a boolean value.
- The `Clone` constraint is removed for the `NonEmptyFrontier::leaf_value` method.

Changes to the `bridgetree` module:

- Most `MerkleBridge` methods now require the `H` type to be ordered.
- Changed the return type of `MerkleBridge::auth_fragments` from 
  `HashMap<usize, AuthFragment>` to `BTreeMap<Position, AuthFragment>`
- `MerkleBridge::successor` arguments have changed. 
- `Checkpoint` is now a struct, rather than an enum type, and contains
  information about witnessed nodes that have been marked for removal since the
  last checkpoint. 
- `BridgeTree` now requires an `Ord` constraint for its leaf type, rather than a
  `Hash` constraint.

### Removed

- `TryFrom<Position>` impl for `usize`
- `bridgetree::Leaf::into_value` removed in favor of `bridgtree::leaf::value`
- `bridgetree::MerkleBridge::leaf_value` method has been removed as a duplicate.
- `bridgetree::BridgeTree::witnessable_leaves` has been replaced by
  `bridgetree::BridgeTree::witnessed_indices`

## [0.2.0] - 2022-03-22
Initial release!
