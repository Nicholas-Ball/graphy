use criterion::{criterion_group, criterion_main};

mod fiedler;
mod laplacian;
mod matrix_add;

criterion_group!(
    benches,
    laplacian::benchmark,
    fiedler::benchmark,
    matrix_add::benchmark
);
criterion_main!(benches);
