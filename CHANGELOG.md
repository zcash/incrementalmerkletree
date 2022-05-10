# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2022-05-10

### Added

- `incrementalmerkletree::bridgetree`:
  - `Checkpoint::witnessed` returns the set of positions that have been marked with
    `BridgeTree::witness` while this checkpoint was the current checkpoint.

### Changed

- `incrementalmerkletree`:
  - `Tree::authentication_path` has been changed to take an `as_of_root` parameter.
    This allows computation of the authentication path as of previous tree states, in
    addition to the previous behavior which only allowed computation of the path as of the
    most recent tree state. The provided `as_of_root` value must be equal to either the
    current root of the tree, or to the root of the tree at a previous checkpoint that
    contained a note at the given position. 

### Removed

- `incrementalmerkletree::bridgetree`:
  - `Checkpoint::rewrite_indices` was an internal utility method that had inadvertently
    been made a part of the public API.

## [0.3.0-beta.2] - 2022-04-06

### Added
- `incrementalmerkletree`:
  - `Tree::get_witnessed_leaf`, to allow a user to query for the leaf value of
    witnessed leaves by their position in the tree.
  - `Tree::witnessed_positions`, to allow a user to query for all positions that
    have been witnessed.
  - `Tree::garbage_collect`, to prune checkpoint and removed witness information 
    that is no longer reachable by rewinds of the tree.
- `incrementalmerkletree::bridgetree`:
  - `NonEmptyFrontier::current_leaf`
  - `BridgeTree::from_frontier`
  - `BridgeTree::check_consistency`

### Changed

- `incrementalmerkletree`:
  - The `Tree` trait has substantial changes in this release. Hashes are no longer used
    in identifying nodes when generating authentication paths and removing witnesses;
    instead, these operations now use position information exclusively.
  - `Tree::authentication_path` and `Tree::remove_witness` have both been changed to only
    take a `position` parameter, instead of both the leaf value and the position.
  - `Tree::current_leaf` and `Tree::witness` have both been changed to only return the leaf
    value, instead of both the leaf value and the position.
- `incrementalmerkletree::bridgetree`:
  - The type of `BridgeTree::saved` and `Checkpoint::forgotten` have been changed from 
    `BTreeMap<(Position, H), usize>` to `BTreeMap<Position, usize>`. This change
    is also reflected in the rturn type of the `BridgeTree::witnessed_indices` method.
  - The `Checkpoint` type is no longer parameterized by `H`.
  - `BridgeTree::bridges` has been split into two parts: 
    - `BridgeTree::prior_bridges` now tracks past bridges not including the current frontier.
    - `BridgeTree::current_bridge` now tracks current mutable frontier.
  - The signature of `BridgeTree::from_parts` has been modified to reflect these changes.

### Removed
- `incrementalmerkletree`:
  - `Tree::is_witnessed` (use `Tree::get_witnessed_leaf(...).is_some()` instead).
  - The recording / playback infrastructure, which was unused and added
    unnecessary complexity:
    - The `Tree::Recording` associated type.
    - `Tree::{recording, play}`
    - `Recording`
- `Tree::is_witnessed` has been removed (use `Tree::get_witnessed_leaf(...).is_some()`
  instead).

### Fixed

- A bug in `BridgeTree::garbage_collect` that caused garbage collection to in some 
  cases incorrectly rewrite checkpointed bridge lengths, resulting in a condition 
  where a rewind could panic after a GC operation.

## [0.3.0-beta.1] - 2022-03-22

### Added
- `incrementalmerkletree`:
  - New trait impls
    - `impl From<Position> for usize`
    - `impl Add<usize> for Position`
    - `impl AddAssign<usize> for Position`
    - `impl TryFrom<u64> for Position`
    - `impl Hash for Position`
  - New trait methods:
    - `Tree::current_leaf`, which returns the position of the latest note
      appended to the tree along with the leaf value, if any.
    - `Tree::current_position`, which returns the position of the latest note
      appended to the tree, if any.
    - `Tree::is_witnessed`, which returns a boolean value indicating whether a
      leaf value has been marked for later construction of an authentication
      path at a specified position in the tree.
- `incrementalmerkletree::bridgetree`:
  - `MerkleBridge::position`, which returns the position of the tip of the
    frontier.
  - `MerkleBridge::current_leaf`, which returns the leaf value most recently
    appended to the frontier.
  - `Leaf::value`, which returns the value of the leaf by reference.
  - `impl {PartialEq, Eq} for {AuthFragment, MerkleBridge, BridgeTreeError}`
  - Various constructor and accessor methods for `Checkpoint` to facilitate
    serialization:
    - `Checkpoint::{from_parts, at_length}`
    - `Checkpoint::{bridges_len, is_witnessed, forgotten}`
    - `Checkpoint::rewrite_indices`
  - `BridgeTree::witnessed_indices`
  - `BridgeTree::garbage_collect`, which prunes data for removed witnesses and
    inaccessable checkpoints beyond the max rewind depth.

### Changed
- `incrementalmerkletree`:
  - `Position` now wraps a `usize` rather than a `u64`, affecting its layout (as
    exposed by `repr(transparent)`).
  - The following methods now take an additional `Position` argument to support
    cases where leaf values may appear at multiple positions in the tree:
    - `Tree::authentication_path`
    - `Tree::remove_witness`
  - `Tree::witness` now returns a tuple of the witnessed position and leaf hash,
    or `None` if the tree is empty, instead of a boolean value.
- `incrementalmerkletree::bridgetree`:
  - Most `MerkleBridge` methods now require `H: Ord`.
  - Changed the return type of `MerkleBridge::auth_fragments` from 
    `HashMap<usize, AuthFragment>` to `BTreeMap<Position, AuthFragment>`; the
    `saved` argument to `BridgeTree::from_parts` is similarly altered.
  - `MerkleBridge::successor` now takes a boolean argument for tracking the most
    recently appended leaf, instead of the previous `cur_idx` argument.
  - `Checkpoint` is now a struct, rather than an enum type, and contains
    information about witnessed nodes that have been marked for removal since the
    last checkpoint.
  - `BridgeTree` now requires an `Ord` constraint for its leaf type, rather than a
    `Hash` constraint.

### Removed
- `incrementalmerkletree`:
  - `Position::increment`
  - `impl TryFrom<Position> for usize` (use the `From` impl instead).
- `incrementalmerkletree::bridgetree`:
  - `Leaf::into_value` (use `Leaf::value` instead).
  - `MerkleBridge::leaf_value` (use `MerkleBridge::current_leaf` instead).
  - `H: Clone` bound on `NonEmptyFrontier::<H>::leaf_value`.
  - `BridgeTree::witnessable_leaves` (use `BridgeTree::witnessed_indices` instead).

## [0.2.0] - 2022-03-22

v0.2.0 is essentially a complete rewrite relative to v0.1.0, and should be considered
the first usable release.  

## [0.1.0] - 2021-06-23
Initial release!
