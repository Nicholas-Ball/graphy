use criterion::Criterion;
use graphy::graph::builder::GraphBuilder;
use std::hint::black_box;

pub fn benchmark(c: &mut Criterion) {
    // Create 2 node graph
    let mut builder2 = GraphBuilder::new();

    builder2
        .add_vertex("A")
        .add_vertex("B")
        .add_edge("A", "B", 1.0f32);
    let graph2 = builder2.build();

    // Create 4 node graph
    let mut builder4 = GraphBuilder::new();
    builder4
        .add_vertex("A")
        .add_vertex("B")
        .add_vertex("C")
        .add_vertex("D")
        .add_edge("A", "B", 1.0)
        .add_edge("A", "C", 1.0)
        .add_edge("B", "D", 1.0)
        .add_edge("C", "D", 1.0);
    let graph4 = builder4.build();

    // Create 8 node graph
    let mut builder8 = GraphBuilder::new();
    builder8
        .add_vertex("A")
        .add_vertex("B")
        .add_vertex("C")
        .add_vertex("D")
        .add_vertex("E")
        .add_vertex("F")
        .add_vertex("G")
        .add_vertex("H")
        .add_edge("A", "B", 1.0)
        .add_edge("A", "C", 1.0)
        .add_edge("B", "D", 1.0)
        .add_edge("C", "D", 1.0)
        .add_edge("D", "E", 1.0)
        .add_edge("E", "F", 1.0)
        .add_edge("F", "G", 1.0)
        .add_edge("G", "H", 1.0);
    let graph8 = builder8.build();

    c.bench_function("Laplacian 2 nodes", |b| {
        b.iter(|| {
            let lap = graph2.laplacian_matrix();
            black_box(lap);
        })
    });

    c.bench_function("Laplacian 4 nodes", |b| {
        b.iter(|| {
            let lap = graph4.laplacian_matrix();
            black_box(lap);
        })
    });

    c.bench_function("Laplacian 8 nodes", |b| {
        b.iter(|| {
            let lap = graph8.laplacian_matrix();
            black_box(lap);
        })
    });
}
