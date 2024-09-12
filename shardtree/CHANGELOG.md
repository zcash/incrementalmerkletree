# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- `shardtree::store::ShardStore::for_each_checkpoint`

## [0.4.0] - 2024-08-12

This is a bugfix release that fixes a couple of subtle problems related to
pruning in the presence of inserted `Frontier` nodes. See the `Removed` and
`Fixed` sections below for additional details.

### Added
- `shardtree::tree::Tree::{is_leaf, map, try_map, empty_pruned}`
- `shardtree::tree::LocatedTree::{map, try_map}`
- `shardtree::prunable::PrunableTree::{has_computable_root, is_full}`
- `shardtree::prunable::LocatedPrunableTree::{max_position}`

### Changed
- Updated to use `incrementalmerkletree` v0.6.

### Removed
- `shardtree::tree::LocatedTree::max_position` did not behave correctly regarding
  annotated parent nodes. Use `LocatedPrunableTree::max_position` instead.
- `shardtree::tree::Tree::is_complete` was somewhat misleadingly named, as `Nil`
  nodes that were inserted as a consequence of insertion after pruning could be
  interpreted as rendering the tree incomplete. Use `PrunableTree::has_computable_root`
  instead to determine whether it is possible to compute the root of a tree.

### Fixed
- Fixes an error that could occur if an inserted `Frontier` node was
  interpreted as a node that had actually had its value observed as though it
  had been inserted using the ordinary tree insertion methods.
- Fixes an error in an internal method that could result in subtree root
  annotation data being discarded when pruning a `Parent` node having
  `Nil` nodes for both its left and right children.

## [0.3.1] - 2024-04-03

### Fixed
- Fixes a missing transitive dependency when using the `test-dependencies` feature flag.

## [0.3.0] - 2024-03-25

### Added
- `ShardTree::{store, store_mut}`
- `ShardTree::insert_frontier`

### Changed
- `shardtree::error::InsertionError` has new variant `MarkedRetentionInvalid`

## [0.2.0] - 2023-11-07

### Added
- `ShardTree::{root_at_checkpoint_id, root_at_checkpoint_id_caching}`
- `ShardTree::{witness_at_checkpoint_id, witness_at_checkpoint_id_caching}`

### Changed
- `ShardTree::root_at_checkpoint` and `ShardTree::root_at_checkpoint_caching` have
  been renamed to `root_at_checkpoint_depth` and `root_at_checkpoint_depth_caching`,
  respectively.
- `ShardTree::witness` and `ShardTree::witness_caching` have
  been renamed to `witness_at_checkpoint_depth` and `witness_at_checkpoint_depth_caching`,
  respectively.

## [0.1.0] - 2023-09-08

Initial release!
