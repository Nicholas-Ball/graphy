use crate::graph::Graph;
use faer::mat::Mat;
use faer::traits::RealField;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Sub, SubAssign};

impl<
    T: Eq + Clone + Hash,
    V: Debug + PartialEq + Copy + num_traits::Zero + Sub<Output = V> + SubAssign + RealField + 'static,
> Graph<T, V>
{
    /// Computes the Laplacian matrix of the graph.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `Mat<V>` - The Laplacian matrix as a 2D array where each entry represents the difference between the degree matrix and the adjacency matrix.
    pub fn laplacian_matrix(&self) -> Mat<V> {
        let a = self.adjacency_matrix();
        let d = self.degree_matrix();

        let (nrows, ncols) = (d.nrows(), d.ncols());
        let mut l = Mat::zeros(nrows, ncols);

        for i in 0..nrows {
            for j in 0..ncols {
                l[(i, j)] = d[(i, j)] - a[(i, j)];
            }
        }

        l
    }

    /// Computes the out-Laplacian matrix of the graph for directed graphs.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `Mat<V>` - The out-Laplacian matrix as a 2D array where each entry represents the difference between the out-degree matrix and the adjacency matrix.
    pub fn laplacian_matrix_out(&self) -> Mat<V> {
        let a = self.adjacency_matrix_directed();
        let d = self.degree_matrix_out();

        let (nrows, ncols) = (d.nrows(), d.ncols());
        let mut l = Mat::zeros(nrows, ncols);

        for i in 0..nrows {
            for j in 0..ncols {
                l[(i, j)] = d[(i, j)] - a[(i, j)];
            }
        }

        l
    }

    /// Computes the in-Laplacian matrix of the graph for directed graphs.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `Mat<V>` - The in-Laplacian matrix as a 2D array where each entry represents the difference between the in-degree matrix and the adjacency matrix.
    pub fn laplacian_matrix_in(&self) -> Mat<V>
    where
        V: std::ops::Sub<Output = V>,
    {
        let a = self.adjacency_matrix_directed();
        let d = self.degree_matrix_in();

        let (nrows, ncols) = (d.nrows(), d.ncols());
        let mut l = Mat::zeros(nrows, ncols);

        for i in 0..nrows {
            for j in 0..ncols {
                l[(i, j)] = d[(i, j)] - a[(i, j)];
            }
        }

        l
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::builder::GraphBuilder;
    use faer::mat::Mat;

    fn mat_from_row_slice(nrows: usize, ncols: usize, data: &[f64]) -> Mat<f64> {
        assert_eq!(data.len(), nrows * ncols);
        let mut m = Mat::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                m[(i, j)] = data[i * ncols + j];
            }
        }
        m
    }

    /// Checks that each row of the given matrix sums to zero (within a small epsilon for floats).
    ///
    /// # Arguments
    ///
    /// * `matrix` - Reference to a Mat<f64> whose rows will be checked.
    fn assert_row_sums_zero(matrix: &Mat<f64>) {
        let eps = 1e-8;
        for i in 0..matrix.nrows() {
            let mut sum = 0.0;
            for j in 0..matrix.ncols() {
                sum += matrix[(i, j)];
            }
            assert!(
                (sum - 0.0).abs() < eps,
                "Row {} of Laplacian does not sum to zero. Row: {:?}, sum: {}",
                i,
                (0..matrix.ncols())
                    .map(|j| matrix[(i, j)])
                    .collect::<Vec<_>>(),
                sum
            );
        }
    }

    /// Checks that two matrices are approximately equal elementwise (for floats).
    fn assert_matrix_approx_eq(a: &Mat<f64>, b: &Mat<f64>, eps: f64, msg: &str) {
        assert_eq!(
            a.nrows(),
            b.nrows(),
            "Matrix row counts do not match: {:?}",
            msg
        );
        assert_eq!(
            a.ncols(),
            b.ncols(),
            "Matrix col counts do not match: {:?}",
            msg
        );
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() < eps,
                    "{}: Mismatch at ({}, {}): {} != {}",
                    msg,
                    i,
                    j,
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    /// Tests the Laplacian matrix for an undirected graph with two nodes and a single edge.
    ///
    /// Builds a simple undirected graph with two nodes A and B, adds an edge between them,
    /// and checks that the computed Laplacian matrix matches the expected result.
    #[test]
    fn test_laplacian_undirected_two_node_single_edge() {
        // Build a simple undirected graph with two nodes A-B.
        let mut g: GraphBuilder<&str, f64> = GraphBuilder::new();
        g.add_vertex("A");
        g.add_vertex("B");
        g.add_edge("A", "B", 1.0);

        let g = g.build();

        let lap = g.laplacian_matrix();

        // Expected Laplacian:
        // Degree matrix D = [[1,0],[0,1]]
        // Adjacency A = [[0,1],[1,0]]
        // L = D - A = [[1,-1],[-1,1]]
        let expected = mat_from_row_slice(2, 2, &[1.0, -1.0, -1.0, 1.0]);

        assert_matrix_approx_eq(
            &lap,
            &expected,
            1e-8,
            "Undirected Laplacian does not match expected.",
        );
        assert_row_sums_zero(&lap);
    }

    /// Tests the out-Laplacian and in-Laplacian matrices for a directed graph with a single edge.
    ///
    /// Builds a directed graph with two nodes A and B, adds an edge from A to B,
    /// and checks that the computed out-Laplacian and in-Laplacian matrices match the expected results.
    #[test]
    fn test_directed_out_in_laplacians_single_edge() {
        // Directed graph A -> B
        let mut g: GraphBuilder<&str, f64> = GraphBuilder::new();
        g.add_vertex("A");
        g.add_vertex("B");
        g.add_edge("A", "B", 1.0);

        let g = g.build();

        dbg!(g.adjacency_matrix_directed());

        let l_out = g.laplacian_matrix_out();
        let l_in = g.laplacian_matrix_in();

        println!("L_out: \n{:?}", l_out);
        println!("L_in: \n{:?}", l_in);

        // For adjacency A:
        // A = [[0,1],
        //      [0,0]]
        // Out-degree diag: [1,0]
        // In-degree  diag: [0,1]
        // L_out = D_out - A = [[1,-1],[0,0]]
        // L_in  = D_in  - A = [[0,-1],[0,1]]
        let expected_out = mat_from_row_slice(2, 2, &[1.0, -1.0, 0.0, 0.0]);
        let expected_in = mat_from_row_slice(2, 2, &[0.0, -1.0, 0.0, 1.0]);

        println!("Expected L_out: \n{:?}", expected_out);
        println!("Expected L_in: \n{:?}", expected_in);

        assert_matrix_approx_eq(
            &l_out,
            &expected_out,
            1e-8,
            "Out-Laplacian does not match expected.",
        );
        assert_matrix_approx_eq(
            &l_in,
            &expected_in,
            1e-8,
            "In-Laplacian does not match expected.",
        );

        // Row-sum zero checks (each row sum should still be zero).
        assert_row_sums_zero(&l_out);

        // Column sums for in-Laplacian should also be zero.
        let eps = 1e-8;
        for j in 0..l_in.ncols() {
            let mut sum = 0.0;
            for i in 0..l_in.nrows() {
                sum += l_in[(i, j)];
            }
            assert!(
                sum.abs() < eps,
                "Column {} of In-Laplacian does not sum to zero (sum: {}).",
                j,
                sum
            );
        }
    }

    /// Tests the Laplacian matrix for a graph with an isolated vertex.
    ///
    /// Builds a graph with three vertices (A, B, and C), connects A and B, leaves C isolated,
    /// and checks that the computed Laplacian matrix matches the expected result.
    #[test]
    fn test_laplacian_with_isolated_vertex() {
        // Graph with three vertices A-B connected, C isolated.
        let mut g: GraphBuilder<&str, f64> = GraphBuilder::new();
        g.add_vertex("A");
        g.add_vertex("B");
        g.add_vertex("C");
        g.add_edge("A", "B", 2.0);

        let g = g.build();

        let lap = g.laplacian_matrix();

        dbg!(&lap);

        // Adjacency (assuming ordering A,B,C):
        // [[0,2,0],
        //  [2,0,0],
        //  [0,0,0]]
        // Degrees: A=1, B=1, C=0
        // Laplacian:
        // [[ 1,-2, 0],
        //  [-2, 1, 0],
        //  [ 0, 0, 0]]
        let expected = mat_from_row_slice(3, 3, &[1.0, -2.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        assert_matrix_approx_eq(
            &lap,
            &expected,
            1e-8,
            "Laplacian with isolated vertex incorrect.",
        );
    }
}
