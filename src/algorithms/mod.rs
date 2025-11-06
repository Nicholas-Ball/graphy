use std::ops::Mul;

use crate::graph::Graph;
use ndarray::Array2;
use ndarray_linalg::Eig;

impl<T, V: num_traits::identities::Zero + Mul<Output = T> + Copy + std::cmp::PartialOrd>
    Graph<T, V>
{
    /// Computes all-pairs shortest-path distances using the Floyd–Warshall algorithm.
    ///
    /// Inputs:
    /// - self: Reference to the graph. The implementation reads the graph size via `get_rank()`
    ///   and initial edge weights via `get_edge_value(i, j)`.
    ///
    /// Returns:
    /// - A square matrix `Vec<Vec<V>>` where `dist[i][j]` is the shortest-path cost from vertex
    ///   `i` to vertex `j` in the graph.
    ///
    /// Notes:
    /// - The algorithm runs in O(n^3) time where `n` is the number of vertices.
    /// - It assumes the distance type supports addition and comparison.
    pub fn floyd_warshall(&self) -> Vec<Vec<V>> {
        let n = self.get_rank();
        let mut dist = vec![vec![V::zero(); n]; n];

        // Initialize distances based on edge values
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = self.get_edge_value(i, j);
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    // Assuming V supports addition and comparison
                    let new_dist = dist[i][k] + dist[k][j];
                    if new_dist < dist[i][j] {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }

        dist
    }

    pub fn fiedler(&self) -> (V, Vec<V>) {
        // Find eignenvalues and eigenvectors of the Laplacian matrix
        // Return the second smallest eigenvalue and its corresponding eigenvector
        let e: Array2<V> = self.edges.into();

        e.eig()
            .expect("Failed to compute eigenvalues and eigenvectors")
            .1
            .column(1)
            .iter()
            .cloned()
            .collect::<Vec<V>>()
            .into_iter()
            .zip(
                e.eig()
                    .expect("Failed to compute eigenvalues and eigenvectors")
                    .0
                    .iter()
                    .cloned()
                    .skip(1),
            )
            .next()
            .expect("Graph must have at least two vertices");
    }
}

#[cfg(test)]
/// Unit tests for the Floyd–Warshall routine and supporting test structures.
mod tests {
    use super::*;

    /// Dummy graph implementation backed by an adjacency matrix, used for testing algorithms.
    struct TestGraph {
        edges: Vec<Vec<i32>>,
    }

    impl TestGraph {
        /// Returns the number of vertices in the test graph.
        ///
        /// Inputs:
        /// - self: Reference to the `TestGraph`.
        ///
        /// Returns:
        /// - The number of vertices (the dimension of the adjacency matrix).
        fn get_rank(&self) -> usize {
            self.edges.len()
        }
        /// Retrieves the edge weight from vertex `i` to vertex `j`.
        ///
        /// Inputs:
        /// - i: Source vertex index.
        /// - j: Target vertex index.
        ///
        /// Returns:
        /// - The edge weight as `i32`.
        fn get_edge_value(&self, i: usize, j: usize) -> i32 {
            self.edges[i][j]
        }
    }

    /// Extension trait providing a Floyd–Warshall routine for `TestGraph` used in unit tests.
    trait FloydWarshallExt {
        /// Computes all-pairs shortest-path distances for the test graph.
        ///
        /// Inputs:
        /// - self: Reference to the graph.
        ///
        /// Returns:
        /// - A square matrix of distances `Vec<Vec<i32>>`, where entry `[i][j]` is the shortest
        ///   path cost from vertex `i` to vertex `j`.
        fn floyd_warshall(&self) -> Vec<Vec<i32>>;
    }

    impl FloydWarshallExt for TestGraph {
        /// Implementation of the Floyd–Warshall algorithm for `TestGraph`, producing all-pairs
        /// shortest-path distances.
        ///
        /// Inputs:
        /// - self: Reference to the graph.
        ///
        /// Returns:
        /// - A square matrix of `i32` shortest-path distances.
        fn floyd_warshall(&self) -> Vec<Vec<i32>> {
            let n = self.get_rank();
            let mut dist = vec![vec![0i32; n]; n];

            // Initialize distances based on edge values
            for i in 0..n {
                for j in 0..n {
                    dist[i][j] = self.get_edge_value(i, j);
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

    /// Verifies basic Floyd–Warshall behavior on a small connected graph.
    #[test]
    fn test_floyd_warshall_basic() {
        // 0 1 4
        // 1 0 2
        // 4 2 0
        let edges = vec![vec![0, 1, 4], vec![1, 0, 2], vec![4, 2, 0]];
        let graph = TestGraph { edges };
        let result = graph.floyd_warshall();
        let expected = vec![vec![0, 1, 3], vec![1, 0, 2], vec![3, 2, 0]];
        assert_eq!(result, expected);
    }

    /// Ensures that disconnected vertices retain an effectively infinite distance.
    #[test]
    fn test_floyd_warshall_disconnected() {
        // 0 9999
        // 9999 0
        let inf = 9999;
        let edges = vec![vec![0, inf], vec![inf, 0]];
        let graph = TestGraph { edges };
        let result = graph.floyd_warshall();
        let expected = vec![vec![0, inf], vec![inf, 0]];
        assert_eq!(result, expected);
    }

    /// Checks shortest paths on a triangle graph for correctness.
    #[test]
    fn test_floyd_warshall_triangle() {
        // 0 3 8
        // 3 0 2
        // 8 2 0
        let edges = vec![vec![0, 3, 8], vec![3, 0, 2], vec![8, 2, 0]];
        let graph = TestGraph { edges };
        let result = graph.floyd_warshall();
        let expected = vec![vec![0, 3, 5], vec![3, 0, 2], vec![5, 2, 0]];
        assert_eq!(result, expected);
    }
}
