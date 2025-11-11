use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Sub, SubAssign},
};

use faer::traits::RealField;
use faer::{Side, col::Own};
use num_traits::{Float, FromPrimitive};

use crate::graph::Graph;

impl<
    T: Eq + Clone + Hash,
    V: Add
        + Copy
        + Debug
        + PartialOrd
        + RealField
        + SubAssign
        + num_traits::Zero
        + Sub<Output = V>
        + std::ops::AddAssign
        + 'static,
> Graph<T, V>
{
    /// Computes the Fiedler value and vector of the graph's (combinatorial) Laplacian.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `(V, faer::col::generic::Col<Own<V>>)` - The Fiedler value (second-smallest eigenvalue)
    ///   and its corresponding eigenvector (column 1).
    ///
    /// The Fiedler value is the second-smallest eigenvalue of the Laplacian matrix,
    /// and the Fiedler vector is its associated eigenvector.
    pub fn fiedler(&self) -> (V, faer::col::generic::Col<Own<V>>) {
        let a = self.laplacian_matrix();

        // Compute eigenvalues and eigenvectors using faer
        let decomp = a.self_adjoint_eigen(Side::Lower).unwrap();

        let eigen_values = decomp.S();
        let eigen_vectors = decomp.U();

        // The eigenvalues are sorted in ascending order
        let fiedler_value = eigen_values[1];
        let fiedler_vector = eigen_vectors.col(1);

        (fiedler_value, fiedler_vector.cloned())
    }

    /// Partitions the vertex set into two groups using the sign of the Fiedler vector
    /// (the eigenvector corresponding to the second-smallest Laplacian eigenvalue), and
    /// returns the two induced subgraphs (one per group) instead of raw index lists.
    ///
    /// Grouping rule:
    ///   - Vertices whose Fiedler vector entry is >= V::zero() go into the first group.
    ///   - Vertices whose entry is < V::zero() go into the second group.
    ///
    /// This implements the standard spectral bisection heuristic. The partition is only defined
    /// up to a global sign flip of the Fiedler vector (i.e., the two returned graphs may be swapped
    /// relative to another run or mathematical convention).
    ///
    /// For disconnected graphs with multiple 0 eigenvalues, the "Fiedler vector" chosen may
    /// correspond to one particular null-space direction, potentially yielding a partition
    /// that reflects one separation of components.
    ///
    /// The returned graphs are the induced subgraphs on each vertex group: all original vertices
    /// in a group are added, and for every ordered pair (u, v) of vertices in the group, the
    /// original edge (u, v) is added if its weight is non-zero.
    ///
    /// Implementation notes:
    ///   - Edges are added using vertex labels (cloned T values) instead of numeric indices.
    ///   - Edge weights are retrieved via `get_edge_value(&T, &T)` rather than indexing by usize.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `(Graph<T, V>, Graph<T, V>)` - Tuple containing:
    ///   - Induced subgraph on vertices with Fiedler entries >= V::zero().
    ///   - Induced subgraph on vertices with Fiedler entries < V::zero().
    pub fn split_groups_by_fiedler(&self) -> (Graph<T, V>, Graph<T, V>)
    where
        T: Clone,
        V: Float + FromPrimitive,
    {
        let (_fiedler_value, fiedler_vector) = self.fiedler();

        dbg!("Fiedler vector for splitting: {:?}", &fiedler_vector);

        let mut group1_indices = Vec::new();
        let mut group2_indices = Vec::new();

        let pivot = fiedler_vector[0];

        for (i, &value) in fiedler_vector.iter().enumerate() {
            if (value - pivot).abs() >= V::from_f64(0.01).unwrap() {
                group1_indices.push(i);
            } else {
                group2_indices.push(i);
            }
        }

        use crate::graph::builder::GraphBuilder;

        // Build induced subgraph for group 1
        let mut builder1: GraphBuilder<T, V> = GraphBuilder::new();
        for &idx in &group1_indices {
            let v = self.vertices[idx].clone();
            builder1.add_vertex(v);
        }
        // Add edges within group 1 (using vertex labels with new API)
        for &i in &group1_indices {
            for &j in &group1_indices {
                let w = self.get_edge_value(&self.vertices[i], &self.vertices[j]);
                if w != V::zero() {
                    let src = self.vertices[i].clone();
                    let dst = self.vertices[j].clone();
                    builder1.add_edge(src, dst, w);
                }
            }
        }
        let g1 = builder1.build();

        // Build induced subgraph for group 2
        let mut builder2: GraphBuilder<T, V> = GraphBuilder::new();
        for &idx in &group2_indices {
            let v = self.vertices[idx].clone();
            builder2.add_vertex(v);
        }
        // Add edges within group 2 (using vertex labels with new API)
        for &i in &group2_indices {
            for &j in &group2_indices {
                let w = self.get_edge_value(&self.vertices[i], &self.vertices[j]);
                if w != V::zero() {
                    let src = self.vertices[i].clone();
                    let dst = self.vertices[j].clone();
                    builder2.add_edge(src, dst, w);
                }
            }
        }
        let g2 = builder2.build();

        (g1, g2)
    }
}

#[cfg(test)]
mod fiedler_tests {
    use crate::graph::builder::GraphBuilder;

    /// Tests the Fiedler value and vector for a disconnected graph with two components.
    ///
    /// For a disconnected graph with two components, the Fiedler value should be 0.0.
    #[test]
    fn test_fiedler_disconnected_two_components() {
        let n = 6;
        let mut builder: GraphBuilder<i32, f64> = GraphBuilder::new();
        for i in 0..n {
            builder.add_vertex(i as i32);
        }

        // Group 1: vertices 0, 1, 4
        builder.add_edge(0, 4, 1.0);
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(1, 4, 1.0);

        // Group 2: vertices 2, 3, 5
        builder.add_edge(3, 2, 1.0);
        builder.add_edge(3, 5, 1.0);

        let g = builder.build();

        let (fiedler_val, fiedler_vec) = g.fiedler();

        dbg!("Fiedler value: {}", &fiedler_val);
        dbg!("Fiedler vector: {:?}", &fiedler_vec);

        assert!(fiedler_val.abs() < 1e-6);

        // Group 1
        assert!(fiedler_vec[0] == 0.0);
        assert!(fiedler_vec[1] == 0.0);
        assert!(fiedler_vec[4] == 0.0);

        // Group 2
        assert!(fiedler_vec[2] <= 0.0);
        assert!(fiedler_vec[3] <= 0.0);
        assert!(fiedler_vec[5] <= 0.0);
    }

    // ------------------------------------------------------------
    // Split groups by Fiedler tests
    // ------------------------------------------------------------

    /// Utility: return a sorted clone of the given vector.
    fn sorted(mut v: Vec<i32>) -> Vec<i32> {
        v.sort();
        v
    }

    /// Splitting on a simple path graph with 4 vertices should produce two groups
    /// corresponding to the two halves of the path (up to a global sign flip).
    #[test]
    fn test_split_groups_by_fiedler_path_graph_4() {
        let mut builder: GraphBuilder<i32, f64> = GraphBuilder::new();
        for i in 0..4 {
            builder.add_vertex(i as i32);
        }
        // Undirected path: 0-1 and 2-3 (two disconnected edges)
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(2, 3, 1.0);

        let g = builder.build();
        let (g1, g2) = g.split_groups_by_fiedler();

        // Expect {0,1} and {2,3} up to swapping.
        let s1 = sorted(g1.get_vertices().clone());
        let s2 = sorted(g2.get_vertices().clone());
        let a = vec![0, 1];
        let b = vec![2, 3];

        assert!(
            (s1 == a && s2 == b) || (s1 == b && s2 == a),
            "Unexpected partition: {:?} | {:?}, expected {:?} | {:?} (up to swap)",
            s1,
            s2,
            a,
            b
        );
    }

    /// Splitting on two tight clusters with a weak bridge should separate the clusters
    /// (up to a global sign flip).
    #[test]
    fn test_split_groups_by_fiedler_two_clusters_weak_bridge() {
        let mut builder: GraphBuilder<i32, f64> = GraphBuilder::new();
        for i in 0..6 {
            builder.add_vertex(i as i32);
        }

        // Make two clusters: {0,1,2} and {3,4,5}
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(0, 2, 1.0);
        builder.add_edge(1, 2, 1.0);

        builder.add_edge(3, 4, 1.0);
        builder.add_edge(3, 5, 1.0);
        builder.add_edge(4, 5, 1.0);

        // Weak bridge between the two clusters
        builder.add_edge(2, 3, 0.01);

        let g = builder.build();
        let (g1, g2) = g.split_groups_by_fiedler();

        let s1 = sorted(g1.get_vertices().clone());
        let s2 = sorted(g2.get_vertices().clone());
        let left = vec![0, 1, 2];
        let right = vec![3, 4, 5];

        assert!(
            (s1 == left && s2 == right) || (s1 == right && s2 == left),
            "Unexpected partition: {:?} | {:?}, expected {:?} | {:?} (up to swap)",
            s1,
            s2,
            left,
            right
        );
    }
}
