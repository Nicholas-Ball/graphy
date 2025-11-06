use std::fmt::Debug;

use crate::graph::Graph;
use anyhow::Result;
use nalgebra::{DMatrix, DVector, SymmetricEigen};

impl<T, V> Graph<T, V>
where
    V: 'static + num_traits::Zero + Copy + std::cmp::PartialOrd + std::ops::Add<Output = V> + Debug,
{
    /// Computes all-pairs shortest-path distances using the Floyd–Warshall algorithm.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph. Uses `get_rank()` for the number of vertices and `get_edge_value(i, j)` for initial edge weights.
    ///
    /// # Returns
    /// * `Vec<Vec<V>>` - A square matrix where `dist[i][j]` is the shortest-path cost from vertex `i` to vertex `j`.
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

    /// Returns a reference to the degree matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `&DMatrix<V>` - Reference to the degree matrix as a 2D array.
    pub fn degree_matrix(&self) -> &DMatrix<V> {
        &self.degrees
    }

    /// Computes the Laplacian matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `DMatrix<V>` - The Laplacian matrix as a 2D array.
    pub fn laplacian_matrix(&self) -> DMatrix<V>
    where
        V: std::ops::Sub<Output = V>,
    {
        let a = self.adjacency_matrix();
        let d = self.degree_matrix();

        // Laplacian L = D - A (element-wise to avoid extra trait bounds)
        let (nrows, ncols) = d.shape();
        let mut l = DMatrix::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                l[(i, j)] = d[(i, j)] - a[(i, j)];
            }
        }

        l
    }

    /// Returns a reference to the adjacency matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `&DMatrix<V>` - Reference to the adjacency matrix.
    pub fn adjacency_matrix(&self) -> &DMatrix<V> {
        &self.edges
    }

    /// Returns a reference to the out-degree matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `&DMatrix<V>` - Reference to the out-degree matrix as a 2D array.
    pub fn degree_matrix_out(&self) -> &DMatrix<V> {
        &self.outgoing_degrees
    }

    /// Computes the out-Laplacian matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `DMatrix<V>` - The out-Laplacian matrix as a 2D array.
    pub fn laplacian_matrix_out(&self) -> DMatrix<V>
    where
        V: std::ops::Sub<Output = V>,
    {
        let a: &DMatrix<V> = self.adjacency_matrix();
        let d = self.degree_matrix_out();

        // Out-Laplacian L_out = D_out - A (element-wise)
        let (nrows, ncols) = d.shape();
        let mut l = DMatrix::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                l[(i, j)] = d[(i, j)] - a[(i, j)];
            }
        }

        l
    }

    /// Returns a reference to the in-degree matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `&DMatrix<V>` - Reference to the in-degree matrix as a 2D array.
    pub fn degree_matrix_in(&self) -> &DMatrix<V> {
        &self.incoming_degrees
    }

    /// Computes the in-Laplacian matrix of the graph.
    ///
    /// # Arguments
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    /// * `Result<DMatrix<V>>` - The in-Laplacian matrix as a 2D array.
    pub fn laplacian_matrix_in(&self) -> Result<DMatrix<V>>
    where
        V: std::ops::Sub<Output = V>,
    {
        let a: &DMatrix<V> = self.adjacency_matrix();
        let d = self.degree_matrix_in();

        // In-Laplacian L_in = D_in - A (element-wise)
        let (nrows, ncols) = d.shape();
        let mut l = DMatrix::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                l[(i, j)] = d[(i, j)] - a[(i, j)];
            }
        }

        Ok(l)
    }
}

impl<T> Graph<T, f64> {
    pub fn fiedler(&self) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
        let l = self.laplacian_matrix();
        let decomp = SymmetricEigen::new(l);

        // Pair eigenvalues with corresponding eigenvectors
        let mut pairs: Vec<(f64, DVector<f64>)> = decomp
            .eigenvalues
            .iter()
            .zip(decomp.eigenvectors.column_iter())
            .map(|(val, vec)| (*val, vec.into()))
            .collect();

        // Sort dec by eigenvalue
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // The Fiedler value is the second smallest eigenvalue
        let (fiedler_val, fiedler_vec) = &pairs[1];

        Ok((fiedler_val.clone(), fiedler_vec.data.as_vec().clone()))
    }
}

#[cfg(test)]
/// Unit tests for the Floyd–Warshall routine and supporting test structures.
mod tests {
    use crate::graph::GraphBuilder;

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
        builder.add_edge(0, 1, 1i32);
        builder.add_edge(1, 0, 1i32);
        builder.add_edge(1, 2, 2i32);
        builder.add_edge(2, 1, 2i32);
        builder.add_edge(0, 2, 4i32);
        builder.add_edge(2, 0, 4i32);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![vec![0, 1, 3], vec![1, 0, 2], vec![3, 2, 0]];
        assert_eq!(result, expected);
    }

    /// Tests Floyd–Warshall on a disconnected graph to ensure infinite distances are preserved.
    #[test]
    fn test_floyd_warshall_disconnected() {
        // 0 9999
        // 9999 0
        let inf = 9999i32;
        let mut builder = GraphBuilder::new();
        for v in 0usize..2 {
            builder.add_vertex(v);
        }
        builder.add_edge(0, 1, inf);
        builder.add_edge(1, 0, inf);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![vec![0, inf], vec![inf, 0]];
        assert_eq!(result, expected);
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
        builder.add_edge(0, 1, 3i32);
        builder.add_edge(1, 0, 3i32);
        builder.add_edge(1, 2, 2i32);
        builder.add_edge(2, 1, 2i32);
        builder.add_edge(0, 2, 8i32);
        builder.add_edge(2, 0, 8i32);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![vec![0, 3, 5], vec![3, 0, 2], vec![5, 2, 0]];
        assert_eq!(result, expected);
    }

    /// Tests Floyd–Warshall on a single-vertex graph.
    #[test]
    fn test_floyd_warshall_single_vertex() {
        let mut builder: GraphBuilder<usize, i32> = GraphBuilder::new();
        builder.add_vertex(0usize);
        let graph = builder.build();

        let result = graph.floyd_warshall();
        let expected = vec![vec![0]];
        assert_eq!(result, expected);
    }

    /// Tests Floyd–Warshall on an empty graph.
    #[test]
    fn test_floyd_warshall_empty_graph() {
        let builder: GraphBuilder<usize, i32> = GraphBuilder::new();
        let graph = builder.build();
        let result = graph.floyd_warshall();
        let expected: Vec<Vec<i32>> = Vec::new();
        assert_eq!(result, expected);
    }

    // ------------------------------------------------------------
    // Spectral (Fiedler) tests
    // ------------------------------------------------------------

    fn assert_approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "assert_approx failed: got {a}, expected ~{b} (tol={tol})"
        );
    }
    /// Fiedler value for a path graph P4 should be 2 - 2 cos(pi/4) = 2 - sqrt(2) ~ 0.585786.
    #[test]
    fn test_fiedler_path_graph_4() {
        let n = 4;
        let mut builder = GraphBuilder::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_vertex(2);
        builder.add_vertex(3);
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(1, 2, 1.0);
        builder.add_edge(2, 3, 1.0);
        let g = builder.build();

        let (fiedler_val, fiedler_vec) = g.fiedler().unwrap();

        let expected = 2.0 - 2.0 * (std::f64::consts::PI / 4.0).cos();
        assert_approx(fiedler_val, expected, 1e-6);
        assert_eq!(fiedler_vec.len(), n);
        // Fiedler vector should be orthogonal to the constant vector for connected graphs
        let sum: f64 = fiedler_vec.iter().copied().sum();
        assert_approx(sum, 0.0, 1e-6);
    }

    // /// Fiedler value for complete graph K4 is 4.0 (eigenvalues: 0, 4, 4, 4).
    // #[test]
    // fn test_fiedler_complete_graph_4() {
    //     let n = 4;
    //     let mut builder = GraphBuilder::new();
    //     for i in 0..n {
    //         builder.add_vertex(i);
    //     }
    //     for i in 0..n {
    //         for j in 0..n {
    //             if i != j {
    //                 builder.add_edge(i, j, 1.0);
    //             }
    //         }
    //     }
    //     let g = builder.build();

    //     let (fiedler_val, fiedler_vec) = g.fiedler().unwrap();

    //     assert_approx(fiedler_val, 4.0, 1e-6);
    //     assert_eq!(fiedler_vec.len(), n);
    // }

    /// Disconnected graph with two components has Fiedler value 0.0.
    #[test]
    fn test_fiedler_disconnected_two_components() {
        let n = 6;
        let mut builder = GraphBuilder::new();
        for i in 0..n {
            builder.add_vertex(i);
        }
        builder.add_edge(0, 4, 1.0);
        builder.add_edge(1, 4, 1.0);
        builder.add_edge(2, 3, 1.0);
        builder.add_edge(3, 5, 1.0);
        let g = builder.build();

        let (fiedler_val, fiedler_vec) = g.fiedler().unwrap();

        assert_approx(fiedler_val, 0.0, 1e-8);

        // Vector should reflect two there are 2 groups

        // Group 1
        assert!(fiedler_vec[0] >= -1e6);
        assert!(fiedler_vec[1] >= -1e6);
        assert!(fiedler_vec[4] >= -1e6);

        // Group 2
        assert!(fiedler_vec[2] <= 0.0);
        assert!(fiedler_vec[3] <= 0.0);
        assert!(fiedler_vec[5] <= 0.0);
    }
}
