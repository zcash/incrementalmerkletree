# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Added
* `Shardtree::{root_at_checkpoint_id, root_at_checkpoint_id_caching}`
* `Shardtree::{witness_at_checkpoint_id, witness_at_checkpoint_id_caching}`

## Changed
* `Shardtree::root_at_checkpoint` and `Shardtree::root_at_checkpoint_caching` have
  been renamed to `root_at_checkpoint_depth` and `root_at_checkpoint_depth_caching`,
  respectively.
* `Shardtree::witness` and `Shardtree::witness_caching` have
  been renamed to `witness_at_checkpoint_depth` and `witness_at_checkpoint_depth_caching`,
  respectively.

## [0.1.0] - 2023-09-08

Initial release!
