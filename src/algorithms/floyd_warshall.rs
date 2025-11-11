use crate::graph::Graph;
use std::fmt::Debug;
use std::hash::Hash;

impl<T: Eq + Clone + Hash, V> Graph<T, V>
where
    V: 'static
        + num_traits::Zero
        + Copy
        + std::cmp::PartialOrd
        + std::ops::Add<Output = V>
        + Debug
        + std::ops::SubAssign,
{
    /// Computes all-pairs shortest-path distances using the Floyd–Warshall algorithm.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph. Uses `get_rank()` for the number of vertices and `get_edge_value(i, j)` for initial edge weights.
    ///
    /// # Returns
    ///
    /// * `Vec<Vec<V>>` - A square matrix where `dist[i][j]` is the shortest-path cost from vertex `i` to vertex `j`.
    pub fn floyd_warshall(&self) -> Vec<Vec<V>> {
        let n = self.get_rank();
        let mut dist = vec![vec![V::zero(); n]; n];

        // Initialize distances based on edge values
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = self.adjacency_undirectional[(i, j)];
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let new_dist = dist[i][k] + dist[k][j];
                    if new_dist < dist[i][j] {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }

        dist
    }
}

#[cfg(test)]
/// Unit tests for the Floyd–Warshall routine and supporting test structures.
mod floyd_warshall_tests {
    use crate::graph::builder::GraphBuilder;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn mat_approx_eq(a: &[Vec<f64>], b: &[Vec<f64>], eps: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            if row_a.len() != row_b.len() {
                return false;
            }
            for (&x, &y) in row_a.iter().zip(row_b.iter()) {
                if !approx_eq(x, y, eps) {
                    return false;
                }
            }
        }
        true
    }

    /// Tests Floyd–Warshall on a small connected graph for correct shortest-path computation.
    #[test]
    fn test_floyd_warshall_basic() {
        // 0 1 4
        // 1 0 2
        // 4 2 0
        let mut builder = GraphBuilder::new();
        for v in 0usize..3 {
            builder.add_vertex(v);
        }
        builder.add_edge(0, 1, 1.0f64);
        builder.add_edge(1, 0, 1.0f64);
        builder.add_edge(1, 2, 2.0f64);
        builder.add_edge(2, 1, 2.0f64);
        builder.add_edge(0, 2, 4.0f64);
        builder.add_edge(2, 0, 4.0f64);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 2.0],
            vec![3.0, 2.0, 0.0],
        ];
        assert!(mat_approx_eq(&result, &expected, 1e-8));
    }

    /// Tests Floyd–Warshall on a disconnected graph to ensure infinite distances are preserved.
    #[test]
    fn test_floyd_warshall_disconnected() {
        // 0 inf
        // inf 0
        let inf = f64::INFINITY;
        let mut builder = GraphBuilder::new();
        for v in 0usize..2 {
            builder.add_vertex(v);
        }
        builder.add_edge(0, 1, inf);
        builder.add_edge(1, 0, inf);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![vec![0.0, inf], vec![inf, 0.0]];

        assert!(&result == &expected);
    }

    /// Tests Floyd–Warshall on a triangle graph for correct shortest-path computation.
    #[test]
    fn test_floyd_warshall_triangle() {
        // 0 3 8
        // 3 0 2
        // 8 2 0
        let mut builder = GraphBuilder::new();
        for v in 0usize..3 {
            builder.add_vertex(v);
        }
        builder.add_edge(0, 1, 3.0f64);
        builder.add_edge(1, 0, 3.0f64);
        builder.add_edge(1, 2, 2.0f64);
        builder.add_edge(2, 1, 2.0f64);
        builder.add_edge(0, 2, 8.0f64);
        builder.add_edge(2, 0, 8.0f64);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![
            vec![0.0, 3.0, 5.0],
            vec![3.0, 0.0, 2.0],
            vec![5.0, 2.0, 0.0],
        ];

        assert!(mat_approx_eq(&result, &expected, 1e-8));
    }

    /// Tests Floyd–Warshall on a single-vertex graph.
    #[test]
    fn test_floyd_warshall_single_vertex() {
        let mut builder: GraphBuilder<usize, f64> = GraphBuilder::new();
        builder.add_vertex(0usize);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![vec![0.0]];
        assert!(mat_approx_eq(&result, &expected, 1e-8));
    }

    /// Tests Floyd–Warshall on an empty graph.
    #[test]
    fn test_floyd_warshall_empty_graph() {
        let builder: GraphBuilder<usize, f64> = GraphBuilder::new();
        let graph = builder.build();
        let result = graph.floyd_warshall();
        let expected: Vec<Vec<f64>> = Vec::new();
        assert_eq!(result, expected);
    }
}
