use criterion::{criterion_group, criterion_main, Criterion};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use incrementalmerkletree::Address;
use shardtree::{testing::arb_tree, Node};

#[cfg(unix)]
use pprof::criterion::{Output, PProfProfiler};

// An algebra for computing the incomplete roots of a tree (the addresses at which nodes are
// `Nil`). This is used for benchmarking to determine the viability of "attribute grammars" for
// when you want to use `reduce` to compute a value that requires information to be passed top-down
// through the tree.
type RootFn = Box<dyn Fn(Address) -> Vec<Address>>;
pub fn incomplete_roots<V: 'static>(node: Node<RootFn, V>) -> RootFn {
    Box::new(move |addr| match &node {
        Node::Parent { left, right, .. } => {
            let (left_addr, right_addr) = addr
                .children()
                .expect("A parent node cannot appear at level 0");
            let mut left_result = left(left_addr);
            let mut right_result = right(right_addr);
            left_result.append(&mut right_result);
            left_result
        }
        Node::Leaf { .. } => vec![],
        Node::Nil { .. } => vec![addr],
    })
}

pub fn bench_shardtree(c: &mut Criterion) {
    {
        //let mut group = c.benchmark_group("shardtree-incomplete");

        let mut runner = TestRunner::deterministic();
        let input = arb_tree(Just(()), any::<String>(), 16, 4096)
            .new_tree(&mut runner)
            .unwrap()
            .current();
        println!(
            "Benchmarking with {} leaves.",
            input.reduce(
                &(|node| match node {
                    Node::Parent { left, right } => left + right,
                    Node::Leaf { .. } => 1,
                    Node::Nil => 0,
                })
            )
        );

        let input_root = Address::from_parts(
            input
                .reduce(
                    &(|node| match node {
                        Node::Parent { left, right } => std::cmp::max(left, right) + 1,
                        Node::Leaf { .. } => 0,
                        Node::Nil => 0,
                    }),
                )
                .into(),
            0,
        );

        c.bench_function("direct_recursion", |b| {
            b.iter(|| input.incomplete(input_root))
        });

        c.bench_function("reduce", |b| {
            b.iter(|| input.reduce(&incomplete_roots)(input_root))
        });
    }
}

#[cfg(unix)]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_shardtree
}

#[cfg(not(unix))]
criterion_group!(benches, bench_shardtree);

criterion_main!(benches);
