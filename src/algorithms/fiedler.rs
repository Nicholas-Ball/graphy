use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Sub, SubAssign},
};

use nalgebra::SymmetricEigen;

use crate::graph::Graph;

/// Implementation for spectral graph analysis for graphs with scalar edge weights of type V.
impl<
    T: Eq + Clone + Hash,
    V: Add
        + Copy
        + Debug
        + PartialOrd
        + SubAssign
        + num_traits::Zero
        + Sub<Output = V>
        + nalgebra::ComplexField<RealField = V>
        + 'static,
> Graph<T, V>
{
    /// Computes the Fiedler value and vector of the graph's (combinatorial) Laplacian.
    ///
    /// The Laplacian used is the (unnormalized) matrix L = D - A.
    /// This routine assumes:
    ///   - The underlying Laplacian is symmetric (i.e., the graph is treated as undirected
    ///     or its Laplacian is otherwise symmetric).
    ///   - `nalgebra::SymmetricEigen` returns eigenvalues in ascending order (which it does),
    ///     so the second-smallest eigenvalue is at index 1.
    ///
    /// Notes:
    ///   - For a connected graph, the smallest eigenvalue is 0 with multiplicity 1, and
    ///     the Fiedler value (algebraic connectivity) is the next eigenvalue (> 0).
    ///   - For a disconnected graph with c > 1 components, the 0 eigenvalue has multiplicity c,
    ///     so the "Fiedler value" (index 1) may also be 0. The corresponding eigenvector chosen
    ///     by the decomposition is one of the eigenvectors spanning the null space.
    ///   - This function will panic if the graph has fewer than 2 vertices because it directly
    ///     indexes eigenvalue/eigenvector at position 1.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `(V, Vec<V>)` - The Fiedler value (second-smallest eigenvalue) and its corresponding eigenvector (column 1).
    pub fn fiedler(&self) -> (V, Vec<V>) {
        let laplacian = self.laplacian_matrix();

        // Compute eigenvalues and eigenvectors
        let symmetric_eigen = SymmetricEigen::new(laplacian.clone());

        let eigenvalues = symmetric_eigen.eigenvalues;
        let eigenvectors = symmetric_eigen.eigenvectors;

        // Sort eigenpairs by eigenvalue (ascending)
        let mut idx: Vec<usize> = (0..eigenvalues.len()).collect();
        idx.sort_by(|&i, &j| {
            eigenvalues[i]
                .partial_cmp(&eigenvalues[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // The Fiedler value is the second smallest eigenvalue
        let k = idx[1];
        let fiedler_value = eigenvalues[k];
        let fiedler_vector = eigenvectors.column(k).iter().copied().collect();

        (fiedler_value, fiedler_vector)
    }

    /// Partitions the vertex set into two groups using the sign of the Fiedler vector
    /// (the eigenvector corresponding to the second-smallest Laplacian eigenvalue), and
    /// returns the two induced subgraphs (one per group) instead of raw index lists.
    ///
    /// Grouping rule:
    ///   - Vertices whose Fiedler vector entry is >= V::zero() go into the first group.
    ///   - Vertices whose entry is < V::zero() go into the second group.
    ///
    /// This implements a standard spectral bisection heuristic. The partition is only defined
    /// up to a global sign flip of the Fiedler vector (i.e., the two returned graphs may be swapped
    /// relative to an alternate run or mathematical convention).
    ///
    /// For disconnected graphs with multiple 0 eigenvalues, the "Fiedler vector" chosen may
    /// correspond to one particular null-space direction, potentially yielding a partition
    /// that reflects one separation of components.
    ///
    /// The returned graphs are the induced subgraphs on each vertex group: all original vertices
    /// in a group are added, and for every ordered pair (u, v) of vertices in the group, the
    /// original edge (u, v) is added if its weight is non-zero.
    ///
    /// Updated to reflect the new API:
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
    {
        let (_fiedler_value, fiedler_vector) = self.fiedler();

        let mut group1_indices = Vec::new();
        let mut group2_indices = Vec::new();

        for (i, &value) in fiedler_vector.iter().enumerate() {
            if value >= V::zero() {
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
    /// Disconnected graph with two components has Fiedler value 0.0.
    #[test]
    fn test_fiedler_disconnected_two_components() {
        let n = 6;
        let mut builder = GraphBuilder::new();
        for i in 0..n {
            builder.add_vertex(i);
        }

        // Group 1
        builder.add_edge(0, 4, 1.0);
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(1, 4, 1.0);

        // Group 2
        builder.add_edge(3, 2, 1.0);
        builder.add_edge(3, 5, 1.0);

        let g = builder.build();

        let (fiedler_val, fiedler_vec): (f32, Vec<f32>) = g.fiedler();

        dbg!("Fiedler value: {}", &fiedler_val);
        dbg!("Fiedler vector: {:?}", &fiedler_vec);

        assert!(fiedler_val.abs() < 1e-6);

        // Group 1
        assert!(fiedler_vec[0] <= 0.0);
        assert!(fiedler_vec[1] <= 0.0);
        assert!(fiedler_vec[4] <= 0.0);

        // Group 2
        assert!(fiedler_vec[2] == 0.0);
        assert!(fiedler_vec[3] == 0.0);
        assert!(fiedler_vec[5] == 0.0);
    }

    // ------------------------------------------------------------
    // Split groups by Fiedler tests
    // ------------------------------------------------------------

    /// Utility: return a sorted clone of the given vector.
    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    /// Splitting on a simple path graph with 4 vertices should produce two groups
    /// corresponding to the two halves of the path (up to a global sign flip).
    #[test]
    fn test_split_groups_by_fiedler_path_graph_4() {
        let mut builder: GraphBuilder<usize, f64> = GraphBuilder::new();
        for i in 0..4 {
            builder.add_vertex(i);
        }
        // Undirected path: 0-1 2-3
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
        let mut builder: GraphBuilder<usize, f64> = GraphBuilder::new();
        for i in 0..6 {
            builder.add_vertex(i);
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
