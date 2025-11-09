use crate::graph::Graph;
use nalgebra::DMatrix;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Sub, SubAssign};

impl<
    T: Eq + Clone + Hash,
    V: Debug + PartialEq + Copy + num_traits::Zero + Sub<Output = V> + SubAssign + 'static,
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
    /// * `DMatrix<V>` - The Laplacian matrix as a 2D array where each entry represents the difference between the degree matrix and the adjacency matrix.
    pub fn laplacian_matrix(&self) -> DMatrix<V> {
        let a = self.adjacency_matrix();
        let d = self.degree_matrix();

        // Laplacian L = D - A (element-wise)
        let l = d - a;

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
    /// * `DMatrix<V>` - The out-Laplacian matrix as a 2D array where each entry represents the difference between the out-degree matrix and the adjacency matrix.
    pub fn laplacian_matrix_out(&self) -> DMatrix<V> {
        let a: &DMatrix<V> = self.adjacency_matrix_directed();
        let d = self.degree_matrix_out();

        // Out-Laplacian L_out = D_out - A (element-wise)
        let l = d - a;

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
    /// * `DMatrix<V>` - The in-Laplacian matrix as a 2D array where each entry represents the difference between the in-degree matrix and the adjacency matrix.
    pub fn laplacian_matrix_in(&self) -> DMatrix<V>
    where
        V: std::ops::Sub<Output = V>,
    {
        let a: &DMatrix<V> = self.adjacency_matrix_directed();
        let d = self.degree_matrix_in();

        // In-Laplacian L_in = D_in - A (element-wise)
        let l = d - a;

        l
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::builder::GraphBuilder;

    use super::*;
    use nalgebra::DMatrix;

    /// Checks that each row of the given matrix sums to zero.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Reference to a DMatrix<i32> whose rows will be checked.
    fn assert_row_sums_zero(matrix: &DMatrix<i32>) {
        for i in 0..matrix.nrows() {
            let mut sum = 0;
            for j in 0..matrix.ncols() {
                sum += matrix[(i, j)];
            }
            assert_eq!(
                sum,
                0,
                "Row {} of Laplacian does not sum to zero. Row: {:?}",
                i,
                (0..matrix.ncols())
                    .map(|j| matrix[(i, j)])
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Tests the Laplacian matrix for an undirected graph with two nodes and a single edge.
    ///
    /// Builds a simple undirected graph with two nodes A and B, adds an edge between them,
    /// and checks that the computed Laplacian matrix matches the expected result.
    #[test]
    fn test_laplacian_undirected_two_node_single_edge() {
        // Build a simple undirected graph with two nodes A-B.
        let mut g: GraphBuilder<&str, i32> = GraphBuilder::new();
        g.add_vertex("A");
        g.add_vertex("B");
        // Since this is undirected, add edges both ways (if the Graph is directed internally).
        g.add_edge("A", "B", 1);

        let g = g.build();

        let lap = g.laplacian_matrix();

        // Expected Laplacian:
        // Degree matrix D = [[1,0],[0,1]]
        // Adjacency A = [[0,1],[1,0]]
        // L = D - A = [[1,-1],[-1,1]]
        let expected = DMatrix::from_row_slice(2, 2, &[1, -1, -1, 1]);

        assert_eq!(
            lap, expected,
            "Undirected Laplacian does not match expected."
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
        let mut g: GraphBuilder<&str, i32> = GraphBuilder::new();
        g.add_vertex("A");
        g.add_vertex("B");
        g.add_edge("A", "B", 1);

        let g = g.build();

        dbg!("adjacency: \n{}", g.adjacency_matrix_directed());

        let l_out = g.laplacian_matrix_out();
        let l_in = g.laplacian_matrix_in();

        println!("L_out: \n{}", l_out);
        println!("L_in: \n{}", l_in);

        // For adjacency A:
        // A = [[0,1],
        //      [0,0]]
        // Out-degree diag: [1,0]
        // In-degree  diag: [0,1]
        // L_out = D_out - A = [[1,-1],[0,0]]
        // L_in  = D_in  - A = [[0,-1],[0,1]]
        let expected_out = DMatrix::from_row_slice(2, 2, &[1, -1, 0, 0]);
        let expected_in = DMatrix::from_row_slice(2, 2, &[0, -1, 0, 1]);

        println!("Expected L_out: \n{}", expected_out);
        println!("Expected L_in: \n{}", expected_in);

        assert_eq!(
            l_out, expected_out,
            "Out-Laplacian does not match expected."
        );
        assert_eq!(l_in, expected_in, "In-Laplacian does not match expected.");

        // Row-sum zero checks (each row sum should still be zero).
        assert_row_sums_zero(&l_out.map(|v| v)); // map clone to ensure type i32

        // Column sums for in-Laplacian should also be zero.
        for j in 0..l_in.ncols() {
            let mut sum = 0;
            for i in 0..l_in.nrows() {
                sum += l_in[(i, j)];
            }
            assert_eq!(sum, 0, "Column {} of In-Laplacian does not sum to zero.", j);
        }
    }

    /// Tests the Laplacian matrix for a graph with an isolated vertex.
    ///
    /// Builds a graph with three vertices (A, B, and C), connects A and B, leaves C isolated,
    /// and checks that the computed Laplacian matrix matches the expected result.
    #[test]
    fn test_laplacian_with_isolated_vertex() {
        // Graph with three vertices A-B connected, C isolated.
        let mut g: GraphBuilder<&str, i32> = GraphBuilder::new();
        g.add_vertex("A");
        g.add_vertex("B");
        g.add_vertex("C");
        g.add_edge("A", "B", 2);

        let g = g.build();

        let lap = g.laplacian_matrix();

        // Adjacency (assuming ordering A,B,C):
        // [[0,2,0],
        //  [2,0,0],
        //  [0,0,0]]
        // Degrees: A=2, B=2, C=0
        // Laplacian:
        // [[ 2,-2, 0],
        //  [-2, 2, 0],
        //  [ 0, 0, 0]]
        let expected = DMatrix::from_row_slice(3, 3, &[2, -2, 0, -2, 2, 0, 0, 0, 0]);

        assert_eq!(lap, expected, "Laplacian with isolated vertex incorrect.");
        assert_row_sums_zero(&lap);
    }
}
