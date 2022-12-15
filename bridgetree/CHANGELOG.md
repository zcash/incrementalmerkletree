# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- The `testing` module has been removed in favor of depending on
  `incrementalmerkletree::testing`.

## [bridgetree-v0.2.0] - 2022-05-10

The `bridgetree` crate is a fork of `incrementalmerkletree`, with the contents
of the `bridgetree` module moved to the crate root. As such, a number of things
have been significantly refactored and/or reorganized. In the following
documentation, types and operations that have been removed are referred to by
their original paths; types that have been moved to the root module are
referred to by their new location.

### Changed relative to `incrementalmerkletree-v0.3.0`

- The `MerkleBridge` type has been substantially refactored to avoid storing duplicate
  ommer values.

- `Position::is_complete` has been renamed to `Position::is_complete_subtree`.
- The return type of `NonEmptyFrontier::leaf` has changed.
- The arguments to `NonEmptyFrontier::new` have changed.
- `NonEmptyFrontier::root` now takes an optional `root_level` argument.
- `NonEmptyFrontier::leaf` now always returns the leaf hash.
- `NonEmptyFrontier::ommers`' return values now include the sibling of the leaf,
  if the frontier's leaf position is odd. That is, the leaf is now always a leaf,
  instead of a potentially-half-full subtree.
- The arguments to `Frontier::from_parts` have changed.
- The `Ord` bound on the `H` type parameter to `MerkleBridge<H>` has been removed.
- `Altitude` has been renamed to `Level`
- `witness` is now used as the name of the operation to construct the witness for a leaf.
  We now use `mark` to refer to the process of marking a node for which we may later wish
  to construct a witness.
  - `BridgeTree::witness` has been renamed to `BridgeTree::mark`
  - `BridgeTree::witnessed_positions` has been renamed to `BridgeTree::marked_positions`
  - `BridgeTree::get_witnessed_leaf` has been renamed to `BridgeTree::get_marked_leaf`
  - `BridgeTree::remove_witness` has been renamed to `BridgeTree::remove_mark`
  - `BridgeTree::authentication_path` has been renamed to `BridgeTree::witness`
  - `BridgeTree::witnessed` has been renamed to `BridgeTree::marked`
  - `BridgeTree::witnessed_indices` has been renamed to `BridgeTree::marked_indices`

The following types have been moved from the `bridgetree` module of
`incrementalmerkletree` to the crate root:

- `NonEmptyFrontier`
- `Frontier`
- `MerkleBridge`
- `BridgeTree`

### Added relative to `incrementalmerkletree-v0.3.0`

- `NonEmptyFrontier::value_at`
- `NonEmptyFrontier::authentication_path`
- `ContinuityError` and `WitnessingError` error types for reporting errors in
  constructing witnesses.
- `MerkleBridge::position_range`
- `MerkleBridge::tracking`
- `MerkleBridge::ommers`
- `MerkleBridge::current_leaf`
- `Position::is_odd`
- `Position::ommer_index`
- `Position::root_level`
- `Position::past_ommer_count`
- `Address` A type used to uniquely identify node locations within a binary tree.

A `test-dependencies` feature has been added. This makes available a `testing`
module to users of this crate, which contains `proptest` generators for types
from this crate as well as a number of tools for comparison testing between
`Tree` implementations.  The `Frontier` and `Tree` traits have been moved to
the `testing` module, as there is not another good use case for polymorphism
over tree implementations; the API of `Tree` is excessively specialized to the
`BridgeTree` use patterns case.

The `Tree` interface reflects the renaming of `witness` to `mark` described above:
  - `Tree::witness` has been renamed to `Tree::mark`
  - `Tree::witnessed_positions` has been renamed to `Tree::marked_positions`
  - `Tree::get_witnessed_leaf` has been renamed to `Tree::get_marked_leaf`
  - `Tree::remove_witness` has been renamed to `Tree::remove_mark`
  - `Tree::authentication_path` has been renamed to `Tree::witness`

### Removed relative to `incrementalmerkletree-0.3.0`

- `bridgetree::Leaf`
- `bridgetree::AuthFragment`
- `NonEmptyFrontier::size`
- `NonEmptyFrontier::max_altitude`
- `NonEmptyFrontier::current_leaf`
- `NonEmptyFrontier::witness`
- `MerkleBridge::root`
- `MerkleBridge::root_at_altitude`
- `MerkleBridge::auth_fragments`
- `BridgeTree::check_consistency`
- `Position::altitudes_required`
- `Position::all_altitudes_required`
- `Position::auth_path`
- `Position::max_altitude`
- `Position::ommer_altitudes`
- `impl Sub<u8> for Altitude`
