# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.2.0-backcompat.0.8] - 2025-06-04

This is a backwards-compatible release based on 0.1.0 (it does not include the
major testing refactor from the final 0.2.0), to enable crates using the 0.1.0
test suite to migrate to later `incrementalmerkletree` versions.

### Changed
- Updated to `incrementalmerkletree 0.8`

## [0.3.0] - 2025-01-30

### Changed
- Updated to `incrementalmerkletree 0.8`

## [0.2.0] - 2024-10-04

This release includes a significant refactoring and rework of several methods
of the `incrementalmerkletree_testing::Tree` trait. Please read the notes for
this release carefully as the semantics of important methods have changed.
These changes may require changes to example tests that rely on this crate; in
particular, additional checkpoints may be required in circumstances where
rewind operations are being applied.

### Changed
- `incrementalmerkletree_testing::Tree`
  - Added method `Tree::checkpoint_count`.
  - `Tree::root` now takes its `checkpoint_depth` argument as `Option<usize>`
    instead of `usize`. Passing `None` to this method will now compute the root
    given all of the leaves in the tree; if a `Some` value is passed,
    implementations of this method must treat the wrapped `usize` as a reverse
    index into the checkpoints added to the tree, or return `None` if no
    checkpoint exists at the specified index. This effectively modifies this
    method to use zero-based indexing instead of a muddled 1-based indexing
    scheme.
  - `Tree::rewind` now takes an additional `checkpoint_depth` argument, which
    is non-optional. Rewinding the tree may now only be performed if there is
    a checkpoint at the specified depth to rewind to. This depth should be
    treated as a zero-based reverse index into the checkpoints of the tree.
    Rewinding no longer removes the checkpoint being rewound to; instead, it
    removes the effect all state changes to the tree resulting from
    operations performed since the checkpoint was created, but leaves the
    checkpoint itself in place.

## [0.1.0] - 2024-09-25
Initial release.
