//! Benchmarks for the operations of a [`ShardTree`] that a wallet uses when it scans blocks.
//!
//! Run them with `cargo bench -p shardtree`. Each argument that does not start with `--` is a
//! filter: a row is measured only if its workload or its operation contains one of the filters.
//!
//! For each operation, the output gives two kinds of number:
//!
//! - Counts of hashes. These numbers do not depend on the machine or on the cost of the hash
//!   function.
//!   - `combines` is the number of parent nodes that the operation hashes.
//!   - `in batches` is the number of those parent nodes that it hashes through
//!     [`Hashable::combine_pairs`]. It hashes each other parent node with its own call to
//!     [`Hashable::combine`].
//!   - `batches` is the number of calls to `Hashable::combine_pairs`.
//! - The time for the operation with a node type that has a cheap `combine`, and with a node
//!   type that has a costly `combine`. The first time shows the cost of the work on the tree
//!   structure. The second time shows the cost when hashing dominates.
//!
//! The first line of the output gives the measured cost of one `combine` for each node type.
//! To estimate the time for another hash function, multiply the number of `combines` by the
//! cost of that hash function, and add the time for the cheap node type.
//!
//! The node types of this benchmark hash a batch of pairs one pair at a time, so the times
//! show no gain from a batch. A hash function that has a faster batch implementation gains on
//! the parent nodes that `in batches` counts.
//!
//! For each workload, the output also counts the nodes of the tree that results. A parent node
//! with a stored root holds one more node value than a parent node without one.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::ops::Range;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use incrementalmerkletree::{Address, Hashable, Level, Marking, Position, Retention};
use shardtree::{
    store::{memory::MemoryShardStore, ShardStore},
    LocatedPrunableTree, Node, PrunableTree, ShardTree,
};

/// The depth of the benchmarked tree. This is the depth of a Zcash note commitment tree.
const DEPTH: u8 = 32;

/// The height of a shard of the benchmarked tree. This is the height that Zcash wallets use.
const SHARD_HEIGHT: u8 = 16;

/// The number of checkpoints that the benchmarked tree keeps. This is the number that Zcash
/// wallets use.
const MAX_CHECKPOINTS: usize = 100;

/// The number of extra rounds of work in one `combine` of the costly node type.
const COSTLY_ROUNDS: u32 = 8_000;

/// Counts of the hashes that a thread has computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct HashCounts {
    /// The number of parent nodes that were hashed.
    combines: u64,
    /// The number of those parent nodes that were hashed through `Hashable::combine_pairs`.
    in_batches: u64,
    /// The number of calls to `Hashable::combine_pairs`.
    batches: u64,
}

impl HashCounts {
    /// Returns the counts of the hashes that the thread computed after it had the counts
    /// `earlier`.
    fn since(self, earlier: HashCounts) -> HashCounts {
        HashCounts {
            combines: self.combines - earlier.combines,
            in_batches: self.in_batches - earlier.in_batches,
            batches: self.batches - earlier.batches,
        }
    }
}

thread_local! {
    /// The counts of the hashes that the current thread has computed.
    static HASH_COUNTS: Cell<HashCounts> = const {
        Cell::new(HashCounts {
            combines: 0,
            in_batches: 0,
            batches: 0,
        })
    };
}

/// A node value of 32 bytes. Its `combine` depends on the level and on the order of its two
/// arguments, and does `ROUNDS` extra rounds of work, so that the cost of one call can be set.
/// The result of `combine` does not depend on `ROUNDS`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BenchNode<const ROUNDS: u32>([u64; 4]);

/// Returns the node value for the parent of two node values.
///
/// - Preconditions: none.
/// - Inputs: `level` is the level of the two children. `left` and `right` are their values.
/// - Outputs: a value that depends on all three inputs and on the order of the children.
/// - Side effects: none.
fn mix(level: u8, left: &[u64; 4], right: &[u64; 4]) -> [u64; 4] {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ u64::from(level);
    let mut parent = [0u64; 4];
    for (limb, (l, r)) in parent.iter_mut().zip(left.iter().zip(right)) {
        state = (state ^ l)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
            .rotate_left(31)
            ^ r;
        state = (state ^ (state >> 29)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *limb = state;
    }
    parent
}

/// Does `rounds` rounds of arithmetic that the compiler cannot remove.
///
/// - Preconditions: none.
/// - Inputs: `rounds` is the number of rounds. `seed` is the starting value.
/// - Outputs: the value after the last round.
/// - Side effects: none.
fn spin(rounds: u32, seed: u64) -> u64 {
    (0..rounds).fold(seed | 1, |x, _| {
        x.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(27) ^ 0xD6E8_FEB8_6659_FD93
    })
}

/// Returns the root of an empty subtree at each level from 0 to 64.
///
/// The hash functions of Zcash note commitment trees keep such a table, so `empty_root` does
/// not count as a call to `combine`.
fn empty_roots() -> &'static [[u64; 4]] {
    static EMPTY_ROOTS: OnceLock<Vec<[u64; 4]>> = OnceLock::new();
    EMPTY_ROOTS.get_or_init(|| {
        (0..=64u8)
            .scan([0u64; 4], |root, level| {
                let current = *root;
                *root = mix(level, &current, &current);
                Some(current)
            })
            .collect()
    })
}

impl<const ROUNDS: u32> BenchNode<ROUNDS> {
    /// Returns the parent of two nodes, and does the extra rounds of work of this node type.
    ///
    /// - Preconditions: none.
    /// - Inputs: `level` is the level of the two children. `left` and `right` are the children.
    /// - Outputs: the parent node.
    /// - Side effects: none. This method does not count the hash.
    fn parent(level: Level, left: &Self, right: &Self) -> Self {
        let parent = mix(u8::from(level), &left.0, &right.0);
        black_box(spin(ROUNDS, parent[0]));
        BenchNode(parent)
    }
}

impl<const ROUNDS: u32> Hashable for BenchNode<ROUNDS> {
    fn empty_leaf() -> Self {
        BenchNode(empty_roots()[0])
    }

    fn combine(level: Level, left: &Self, right: &Self) -> Self {
        HASH_COUNTS.with(|counts| {
            let mut current = counts.get();
            current.combines += 1;
            counts.set(current);
        });
        Self::parent(level, left, right)
    }

    fn combine_pairs<'a, I>(level: Level, pairs: I) -> Vec<Self>
    where
        Self: 'a,
        I: IntoIterator<Item = (&'a Self, &'a Self)>,
    {
        let parents: Vec<Self> = pairs
            .into_iter()
            .map(|(left, right)| Self::parent(level, left, right))
            .collect();
        HASH_COUNTS.with(|counts| {
            let mut current = counts.get();
            current.combines += parents.len() as u64;
            current.in_batches += parents.len() as u64;
            current.batches += 1;
            counts.set(current);
        });
        parents
    }

    fn empty_root(level: Level) -> Self {
        BenchNode(empty_roots()[usize::from(u8::from(level)).min(64)])
    }
}

/// A sequence of leaves that stands for the note commitments of a range of blocks.
#[derive(Clone, Copy)]
struct Workload {
    /// The name of the workload in the output.
    name: &'static str,
    /// The number of leaves in each block. The last leaf of each block is a checkpoint.
    block_size: u64,
    /// The number of blocks whose leaves the wallet adds to the tree in one batch.
    blocks_per_batch: u64,
    /// The number of batches.
    batch_count: u64,
    /// One leaf in each `marked_every` leaves is marked, as a note of the wallet is. Zero means
    /// that no leaf is marked.
    marked_every: u64,
}

impl Workload {
    /// Returns the number of leaves in one batch.
    fn batch_size(&self) -> u64 {
        self.block_size * self.blocks_per_batch
    }

    /// Returns the number of leaves.
    fn leaf_count(&self) -> u64 {
        self.batch_size() * self.batch_count
    }

    /// Returns the retention of the leaf at `position`.
    ///
    /// - Preconditions: none.
    /// - Inputs: `position` is the position of the leaf.
    /// - Outputs: a checkpoint for the last leaf of a block, with the number of the block as
    ///   its identifier. Otherwise `Marked` for a marked leaf, and `Ephemeral` for any other
    ///   leaf.
    /// - Side effects: none.
    fn retention(&self, position: u64) -> Retention<u64> {
        let marked = self.marked_every != 0 && position % self.marked_every == 1;
        if position % self.block_size == self.block_size - 1 {
            Retention::Checkpoint {
                id: position / self.block_size,
                marking: if marked {
                    Marking::Marked
                } else {
                    Marking::None
                },
            }
        } else if marked {
            Retention::Marked
        } else {
            Retention::Ephemeral
        }
    }

    /// Returns the leaves at `positions`, each with its retention.
    fn leaves<const ROUNDS: u32>(
        &self,
        positions: Range<u64>,
    ) -> Vec<(BenchNode<ROUNDS>, Retention<u64>)> {
        positions
            .map(|position| {
                let value = [
                    position,
                    position.rotate_left(17) ^ 0xA5A5_A5A5_A5A5_A5A5,
                    !position,
                    position.wrapping_mul(0x2545_F491_4F6C_DD1D),
                ];
                (BenchNode(value), self.retention(position))
            })
            .collect()
    }

    /// Returns the ranges of positions of the batches, in order of position.
    fn batches(&self) -> Vec<Range<u64>> {
        (0..self.batch_count)
            .map(|batch| (batch * self.batch_size())..((batch + 1) * self.batch_size()))
            .collect()
    }

    /// Returns the identifier of the last checkpoint of the workload.
    fn last_checkpoint(&self) -> u64 {
        self.blocks_per_batch * self.batch_count - 1
    }

    /// Returns the position of the last marked leaf, if the workload has marked leaves.
    fn last_marked_position(&self) -> Option<u64> {
        let last_position = self.leaf_count() - 1;
        (self.marked_every != 0).then(|| last_position - ((last_position - 1) % self.marked_every))
    }

    /// Returns the number of shards that the leaves of the workload fill.
    fn complete_shard_count(&self) -> u64 {
        self.leaf_count() >> SHARD_HEIGHT
    }
}

/// The workloads that are measured. The sizes of the blocks are odd numbers, so that the
/// batches do not start at the boundaries of subtrees.
const WORKLOADS: &[Workload] = &[
    Workload {
        name: "sparse blocks",
        block_size: 3,
        blocks_per_batch: 1000,
        batch_count: 10,
        marked_every: 0,
    },
    Workload {
        name: "typical blocks",
        block_size: 13,
        blocks_per_batch: 100,
        batch_count: 20,
        marked_every: 1000,
    },
    Workload {
        name: "dense blocks",
        block_size: 501,
        blocks_per_batch: 10,
        batch_count: 12,
        marked_every: 1000,
    },
    Workload {
        name: "many marked leaves",
        block_size: 13,
        blocks_per_batch: 100,
        batch_count: 20,
        marked_every: 10,
    },
    Workload {
        name: "three shards",
        block_size: 13,
        blocks_per_batch: 100,
        batch_count: 160,
        marked_every: 1000,
    },
];

type BenchTree<const ROUNDS: u32> =
    ShardTree<MemoryShardStore<BenchNode<ROUNDS>, u64>, DEPTH, SHARD_HEIGHT>;

type Subtree<const ROUNDS: u32> = (
    LocatedPrunableTree<BenchNode<ROUNDS>>,
    BTreeMap<u64, Position>,
);

/// Returns an empty tree.
fn empty_tree<const ROUNDS: u32>() -> BenchTree<ROUNDS> {
    ShardTree::new(MemoryShardStore::empty(), MAX_CHECKPOINTS)
}

/// Builds one subtree for each batch of `workload`, as a wallet does before it calls
/// `ShardTree::insert_tree`.
fn build_subtrees<const ROUNDS: u32>(workload: &Workload) -> Vec<Subtree<ROUNDS>> {
    workload
        .batches()
        .into_iter()
        .map(|batch| {
            let positions = Position::from(batch.start)..Position::from(batch.end);
            let leaves = workload.leaves::<ROUNDS>(batch);
            let built = LocatedPrunableTree::from_iter(
                positions,
                Level::from(SHARD_HEIGHT),
                leaves.into_iter(),
            )
            .expect("each batch has at least one leaf");
            (built.subtree, built.checkpoints)
        })
        .collect()
}

/// Builds one subtree for each batch of `workload` from a slice of the leaves of the batch.
/// The subtrees are equal to the subtrees that `build_subtrees` returns.
fn build_subtrees_from_leaves<const ROUNDS: u32>(workload: &Workload) -> Vec<Subtree<ROUNDS>> {
    workload
        .batches()
        .into_iter()
        .map(|batch| {
            let start = Position::from(batch.start);
            let leaves = workload.leaves::<ROUNDS>(batch);
            let built = LocatedPrunableTree::from_leaves(start, Level::from(SHARD_HEIGHT), &leaves)
                .expect("each batch has at least one leaf");
            (built.subtree, built.checkpoints)
        })
        .collect()
}

/// Inserts `subtrees` into `tree`, in the order given.
fn insert_subtrees<const ROUNDS: u32>(
    mut tree: BenchTree<ROUNDS>,
    subtrees: Vec<Subtree<ROUNDS>>,
) -> BenchTree<ROUNDS> {
    for (subtree, checkpoints) in subtrees {
        tree.insert_tree(subtree, checkpoints)
            .expect("the subtrees do not conflict");
    }
    tree
}

/// Adds the leaves of `workload` to an empty tree with one `batch_insert` for each batch.
fn batch_insert<const ROUNDS: u32>(workload: &Workload) -> BenchTree<ROUNDS> {
    let mut tree = empty_tree();
    for batch in workload.batches() {
        let start = Position::from(batch.start);
        tree.batch_insert(start, workload.leaves::<ROUNDS>(batch).into_iter())
            .expect("the leaves do not conflict");
    }
    tree
}

/// Adds the leaves of `workload` to an empty tree with one `append` for each leaf.
fn append<const ROUNDS: u32>(workload: &Workload) -> BenchTree<ROUNDS> {
    let mut tree = empty_tree();
    for (leaf, retention) in workload.leaves::<ROUNDS>(0..workload.leaf_count()) {
        tree.append(leaf, retention)
            .expect("the tree has room for the leaf");
    }
    tree
}

/// Returns the address of the shard with the given index.
fn shard_address(index: u64) -> Address {
    Address::from_parts(Level::from(SHARD_HEIGHT), index)
}

/// Returns the root of each shard that the leaves of `workload` fill, in order of index.
///
/// - Preconditions: `tree` holds every leaf of `workload`.
/// - Inputs: `workload` is the workload. `tree` is the tree.
/// - Outputs: one root for each complete shard.
/// - Side effects: none.
fn complete_shard_roots<const ROUNDS: u32>(
    workload: &Workload,
    tree: &BenchTree<ROUNDS>,
) -> Vec<BenchNode<ROUNDS>> {
    (0..workload.complete_shard_count())
        .map(|index| {
            let address = shard_address(index);
            let Ok(shard) = tree.store().get_shard(address);
            shard
                .expect("the shard is stored")
                .root()
                .root_hash(address, address.position_range_end())
                .expect("the shard is complete")
        })
        .collect()
}

/// Returns an empty tree to which the root of each complete shard has been added, as a wallet
/// does when a server gives it the roots of the shards before it scans their blocks.
fn tree_with_shard_roots<const ROUNDS: u32>(roots: &[BenchNode<ROUNDS>]) -> BenchTree<ROUNDS> {
    let mut tree = empty_tree();
    for (index, root) in (0u64..).zip(roots) {
        tree.insert(shard_address(index), root.clone())
            .expect("the roots do not conflict");
    }
    tree
}

/// The numbers of nodes of each kind in a tree.
#[derive(Default)]
struct Census {
    /// The number of parent nodes.
    parents: u64,
    /// The number of parent nodes that store the root of their subtree.
    stored_roots: u64,
    /// The number of leaf nodes.
    leaves: u64,
}

impl Census {
    /// Adds the nodes of `tree` to the counts.
    fn count<const ROUNDS: u32>(&mut self, tree: &PrunableTree<BenchNode<ROUNDS>>) {
        match &**tree {
            Node::Parent { ann, left, right } => {
                self.parents += 1;
                self.stored_roots += u64::from(ann.is_some());
                self.count(left);
                self.count(right);
            }
            Node::Leaf { .. } => self.leaves += 1,
            Node::Nil => {}
        }
    }

    /// Counts the nodes of every shard of `tree`, and of its cap.
    fn of_tree<const ROUNDS: u32>(tree: &BenchTree<ROUNDS>) -> Self {
        let mut census = Census::default();
        let Ok(shard_addresses) = tree.store().get_shard_roots();
        for address in shard_addresses {
            let Ok(shard) = tree.store().get_shard(address);
            census.count(shard.expect("the shard is stored").root());
        }
        let Ok(cap) = tree.store().get_cap();
        census.count(&cap);
        census
    }
}

/// The result of the measurement of one operation.
struct Measurement {
    /// The counts of the hashes of one run of the operation.
    hash_counts: HashCounts,
    /// The median of the times of the runs.
    median: Duration,
}

/// Measures one operation.
///
/// - Preconditions: none.
/// - Inputs: `setup` makes the input for one run. `run` is the operation.
/// - Outputs: the counts of the hashes of one run of `run`, and the median of the times of the
///   runs. The times do not include `setup`, and do not include the time to drop the result
///   of `run`.
/// - Side effects: calls `setup` and `run` at least 3 times, and then until the runs have
///   taken one second in total or until there are 200 runs.
fn measure<S, T>(mut setup: impl FnMut() -> S, mut run: impl FnMut(S) -> T) -> Measurement {
    let mut times = Vec::new();
    let mut hash_counts = HashCounts::default();
    let mut total = Duration::ZERO;
    while times.len() < 3 || (total < Duration::from_secs(1) && times.len() < 200) {
        let input = setup();
        let counts_before = HASH_COUNTS.with(Cell::get);
        let start = Instant::now();
        let output = black_box(run(black_box(input)));
        let elapsed = start.elapsed();
        hash_counts = HASH_COUNTS.with(Cell::get).since(counts_before);
        drop(output);
        total += elapsed;
        times.push(elapsed);
    }
    times.sort();
    Measurement {
        hash_counts,
        median: times[times.len() / 2],
    }
}

/// The measurements for one workload and one node type.
struct WorkloadResults {
    /// The name and the measurement of each operation that was measured.
    operations: Vec<(&'static str, Measurement)>,
    /// The nodes of the tree that holds every leaf of the workload.
    census: Census,
    /// The first 8 bytes of the root of that tree at its last checkpoint.
    root: u64,
}

/// Measures every operation for one workload and one node type.
///
/// - Preconditions: none.
/// - Inputs: `workload` is the workload. `selected` tells whether to measure an operation,
///   given its name.
/// - Outputs: the measurements of the selected operations, and a description of the tree that
///   holds every leaf of the workload.
/// - Side effects: runs the operations.
fn measure_workload<const ROUNDS: u32>(
    workload: &Workload,
    selected: &dyn Fn(&str) -> bool,
) -> WorkloadResults {
    let subtrees = build_subtrees::<ROUNDS>(workload);
    assert!(
        subtrees == build_subtrees_from_leaves::<ROUNDS>(workload),
        "`from_leaves` builds the subtrees that `from_iter` builds"
    );
    let tree = batch_insert::<ROUNDS>(workload);
    let shard_roots = complete_shard_roots(workload, &tree);
    let checkpoint = workload.last_checkpoint();

    let mut operations = Vec::new();
    let mut add = |name: &'static str, measurement: &mut dyn FnMut() -> Measurement| {
        if selected(name) {
            operations.push((name, measurement()));
        }
    };

    add("from_iter, each batch", &mut || {
        measure(|| (), |()| build_subtrees::<ROUNDS>(workload))
    });
    add("from_leaves, each batch", &mut || {
        measure(|| (), |()| build_subtrees_from_leaves::<ROUNDS>(workload))
    });
    add("insert_tree, in order", &mut || {
        measure(
            || (empty_tree(), subtrees.clone()),
            |(tree, subtrees)| insert_subtrees(tree, subtrees),
        )
    });
    add("insert_tree, in reverse order", &mut || {
        measure(
            || (empty_tree(), subtrees.iter().rev().cloned().collect()),
            |(tree, subtrees)| insert_subtrees(tree, subtrees),
        )
    });
    if !shard_roots.is_empty() {
        add("insert_tree, shard roots known", &mut || {
            measure(
                || (tree_with_shard_roots(&shard_roots), subtrees.clone()),
                |(tree, subtrees)| insert_subtrees(tree, subtrees),
            )
        });
    }
    add("batch_insert, each batch", &mut || {
        measure(|| (), |()| batch_insert::<ROUNDS>(workload))
    });
    add("append, each leaf", &mut || {
        measure(|| (), |()| append::<ROUNDS>(workload))
    });
    add("root_at_checkpoint_id", &mut || {
        measure(
            || (),
            |()| {
                tree.root_at_checkpoint_id(&checkpoint)
                    .expect("the root can be computed")
                    .expect("the checkpoint exists")
            },
        )
    });
    if let Some(position) = workload.last_marked_position() {
        add("witness_at_checkpoint_id", &mut || {
            measure(
                || (),
                |()| {
                    tree.witness_at_checkpoint_id(Position::from(position), &checkpoint)
                        .expect("the witness can be computed")
                        .expect("the checkpoint exists")
                },
            )
        });
    }

    let root = tree
        .root_at_checkpoint_id(&checkpoint)
        .expect("the root can be computed")
        .expect("the checkpoint exists");
    WorkloadResults {
        operations,
        census: Census::of_tree(&tree),
        root: root.0[0],
    }
}

/// Formats a duration with three significant digits and a unit.
fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_secs_f64() * 1e9;
    let (value, unit) = if nanos < 1e3 {
        (nanos, "ns")
    } else if nanos < 1e6 {
        (nanos / 1e3, "µs")
    } else if nanos < 1e9 {
        (nanos / 1e6, "ms")
    } else {
        (nanos / 1e9, "s")
    };
    let decimals = if value < 10.0 {
        2
    } else if value < 100.0 {
        1
    } else {
        0
    };
    format!("{value:.decimals$} {unit}")
}

/// Measures the cost of one `combine` call for a node type.
fn combine_cost<const ROUNDS: u32>() -> Duration {
    let left = BenchNode::<ROUNDS>([1, 2, 3, 4]);
    let right = BenchNode::<ROUNDS>([5, 6, 7, 8]);
    let calls: u32 = 2_000;
    let measurement = measure(
        || (),
        |()| {
            (0..calls).fold(left.clone(), |node, _| {
                BenchNode::combine(Level::from(0), &node, &right)
            })
        },
    );
    measurement.median / calls
}

fn main() {
    // Cargo passes flags such as `--bench`. Every other argument is a filter.
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|argument| !argument.starts_with("--"))
        .collect();
    let matches =
        |text: &str| filters.is_empty() || filters.iter().any(|filter| text.contains(filter));

    println!(
        "cost of one combine: cheap node {}, costly node {}",
        format_duration(combine_cost::<0>()),
        format_duration(combine_cost::<COSTLY_ROUNDS>()),
    );

    for workload in WORKLOADS {
        let workload_selected = matches(workload.name);
        let selected = |operation: &str| workload_selected || matches(operation);
        let cheap = measure_workload::<0>(workload, &selected);
        if cheap.operations.is_empty() {
            continue;
        }
        let costly = measure_workload::<COSTLY_ROUNDS>(workload, &selected);
        assert_eq!(
            cheap.root, costly.root,
            "the root does not depend on the node type"
        );

        println!();
        println!(
            "{}: {} leaves, {} in each block, {} in each batch, {}",
            workload.name,
            workload.leaf_count(),
            workload.block_size,
            workload.batch_size(),
            match workload.marked_every {
                0 => "no marked leaves".to_string(),
                marked_every => format!("1 leaf in {marked_every} marked"),
            },
        );
        println!(
            "  tree: {} parent nodes, {} of them with a stored root, {} leaf nodes, root {:016x}",
            cheap.census.parents, cheap.census.stored_roots, cheap.census.leaves, cheap.root,
        );
        println!(
            "  {:<32} {:>10} {:>11} {:>8} {:>12} {:>12}",
            "operation", "combines", "in batches", "batches", "cheap node", "costly node"
        );
        for ((operation, cheap), (_, costly)) in cheap.operations.iter().zip(&costly.operations) {
            assert_eq!(
                cheap.hash_counts, costly.hash_counts,
                "the counts of the hashes do not depend on the node type"
            );
            println!(
                "  {:<32} {:>10} {:>11} {:>8} {:>12} {:>12}",
                operation,
                cheap.hash_counts.combines,
                cheap.hash_counts.in_batches,
                cheap.hash_counts.batches,
                format_duration(cheap.median),
                format_duration(costly.median),
            );
        }
    }
}
