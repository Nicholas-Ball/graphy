use criterion::Criterion;
use faer::mat::Mat;
use std::hint::black_box;

pub fn benchmark(c: &mut Criterion) {
    // 2x2
    let a2 = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 1.0f32,
        (0, 1) => 2.0,
        (1, 0) => 3.0,
        (1, 1) => 4.0,
        _ => unreachable!(),
    });
    let b2 = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 5.0,
        (0, 1) => 6.0,
        (1, 0) => 7.0,
        (1, 1) => 8.0,
        _ => unreachable!(),
    });

    // 4x4
    let a4 = Mat::from_fn(4, 4, |i, j| {
        let data = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        data[i][j]
    });
    let b4 = Mat::from_fn(4, 4, |i, j| {
        let data = [
            [16.0, 15.0, 14.0, 13.0],
            [12.0, 11.0, 10.0, 9.0],
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
        ];
        data[i][j]
    });

    c.bench_function("Matrix Addition 2x2", |b| {
        b.iter(|| {
            let result = black_box(&a2) + black_box(&b2);
            black_box(result);
        })
    });

    c.bench_function("Matrix Addition 4x4", |b| {
        b.iter(|| {
            let result = black_box(&a4) + black_box(&b4);
            black_box(result);
        })
    });
}
