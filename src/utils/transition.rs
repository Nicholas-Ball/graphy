use crate::graph::Graph;
use faer::linalg::solvers::DenseSolveCore;
use faer::traits::RealField;
use std::hash::Hash;
use std::ops::Mul;

impl<T: Eq + Clone + Hash, V: Copy + Mul + RealField + 'static> Graph<T, V> {
    pub fn transition_probability_matrix(&self) -> faer::mat::Mat<V> {
        let a = self.adjacency_matrix();
        let d = self.degree_matrix();

        // invert degree matrix
        let invert = d.partial_piv_lu().inverse();

        // multiply inverted degree matrix by adjacency matrix
        let t = invert.mul(&a);

        t
    }
}
#[cfg(test)]
mod tests {
    use crate::graph::builder::GraphBuilder;

    use super::*;
    use faer::mat::Mat;
    use num_traits::FromPrimitive;

    // Helper: approximate equality for RealField
    fn approx_eq<V: RealField>(a: V, b: V, tol: V) -> bool {
        let diff = if a > b { a - b } else { b - a };
        diff <= tol
    }

    // Helper: assert each row sums to 1 (row-stochastic matrix)
    fn assert_row_stochastic<V: RealField + FromPrimitive + Copy>(m: &Mat<V>) {
        let (rows, cols) = m.shape();
        for i in 0..rows {
            let mut sum = V::zero();
            for j in 0..cols {
                sum = sum + m[(i, j)];
            }
            assert!(
                approx_eq(sum, V::one(), V::from_f64(1e-9).unwrap()),
                "Row {i} does not sum to 1: got {sum:?}"
            );
        }
    }

    // NOTE: These tests assume the existence of certain Graph construction APIs.
    // If your Graph implementation differs, adjust the setup portions accordingly.

    #[test]
    fn two_node_bidirectional_graph() {
        // Graph: 0 <-> 1
        // Expected transition matrix:
        // [0, 1]
        // [1, 0]
        let mut g: GraphBuilder<usize, f64> = GraphBuilder::new();
        g.add_vertex(0);
        g.add_vertex(1);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 0, 1.0);
        let g = g.build();

        let t = g.transition_probability_matrix();

        dbg!(&t);

        assert_eq!(t.nrows(), 2);
        assert_eq!(t.ncols(), 2);
        assert!(approx_eq(t[(0, 0)], 0.0, 1e-12));
        assert!(approx_eq(t[(0, 1)], 1.0, 1e-12));
        assert!(approx_eq(t[(1, 0)], 1.0, 1e-12));
        assert!(approx_eq(t[(1, 1)], 0.0, 1e-12));
        assert_row_stochastic(&t);
    }

    #[test]
    fn triangle_unweighted_graph() {
        // Graph: 0-1-2 fully connected (undirected, no self loops)
        // Adjacency rows each have two ones.
        // Expected transition probabilities per row: each non-self neighbor gets 0.5
        let mut gb: GraphBuilder<usize, f64> = GraphBuilder::new();
        gb.add_vertex(0);
        gb.add_vertex(1);
        gb.add_vertex(2);
        gb.add_edge(0, 1, 1.0);
        gb.add_edge(0, 2, 1.0);
        gb.add_edge(1, 0, 1.0);
        gb.add_edge(1, 2, 1.0);
        gb.add_edge(2, 0, 1.0);
        gb.add_edge(2, 1, 1.0);
        let g = gb.build();

        let t = g.transition_probability_matrix();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 3);

        let half = 0.5;
        // Row 0
        assert!(approx_eq(t[(0, 0)], 0.0, 1e-12));
        assert!(approx_eq(t[(0, 1)], half, 1e-12));
        assert!(approx_eq(t[(0, 2)], half, 1e-12));
        // Row 1
        assert!(approx_eq(t[(1, 0)], half, 1e-12));
        assert!(approx_eq(t[(1, 1)], 0.0, 1e-12));
        assert!(approx_eq(t[(1, 2)], half, 1e-12));
        // Row 2
        assert!(approx_eq(t[(2, 0)], half, 1e-12));
        assert!(approx_eq(t[(2, 1)], half, 1e-12));
        assert!(approx_eq(t[(2, 2)], 0.0, 1e-12));

        assert_row_stochastic(&t);
    }

    #[test]
    fn weighted_graph_probability_distribution() {
        let mut gb: GraphBuilder<usize, f64> = GraphBuilder::new();
        gb.add_vertex(0);
        gb.add_vertex(1);
        gb.add_vertex(2);
        gb.add_edge(0, 1, 2.0);
        gb.add_edge(0, 2, 1.0);

        let g = gb.build();

        let t = g.transition_probability_matrix();

        dbg!(&t);

        // Row 0 checks
        assert!(approx_eq(t[(0, 0)], 0.0, 1e-12));
        assert!(approx_eq(t[(0, 1)], 1.0, 1e-12));
        assert!(approx_eq(t[(0, 2)], 1.0 / 2.0, 1e-12));

        // Rows 1 and 2 should each be deterministic self loops due to added weight 1.0
        assert!(approx_eq(t[(1, 0)], 2.0, 1e-12));
        assert!(approx_eq(t[(1, 1)], 0.0, 1e-12));
        assert!(approx_eq(t[(1, 2)], 0.0, 1e-12));

        assert!(approx_eq(t[(2, 0)], 1.0, 1e-12));
        assert!(approx_eq(t[(2, 1)], 0.0, 1e-12));
        assert!(approx_eq(t[(2, 2)], 0.0, 1e-12));
    }

    #[test]
    fn self_loop_graph() {
        // Node 0 has a self loop and an edge to node 1.
        // If weights are both 1, probabilities from node 0: P(0->0)=0.5, P(0->1)=0.5
        let mut gb: GraphBuilder<usize, f64> = GraphBuilder::new();
        gb.add_vertex(0);
        gb.add_vertex(1);
        gb.add_edge(0, 0, 1.0);
        gb.add_edge(0, 1, 1.0);
        gb.add_edge(1, 0, 1.0); // to ensure node 1 has degree > 0
        let g = gb.build();

        let t = g.transition_probability_matrix();
        dbg!(&t);
        assert!(approx_eq(t[(0, 0)], 0.5, 1e-12));
        assert!(approx_eq(t[(0, 1)], 0.5, 1e-12));
        assert!(approx_eq(t[(1, 0)], 1.0, 1e-12));
        assert_row_stochastic(&t);
    }

    #[test]
    fn empty_graph_has_empty_matrix() {
        let gb: GraphBuilder<usize, f64> = GraphBuilder::new();
        let g = gb.build();
        let t = g.transition_probability_matrix();
        assert_eq!(t.nrows(), 0);
        assert_eq!(t.ncols(), 0);
    }

    #[test]
    fn matrix_matches_manual_computation() {
        // Build a small graph and manually compute D^{-1} A to compare.
        // Graph:
        // 0 -> 1 (1.0), 0 -> 2 (3.0)
        // 1 -> 2 (2.0)
        // 2 -> 0 (4.0)
        let mut gb: GraphBuilder<usize, f64> = GraphBuilder::new();
        gb.add_vertex(0);
        gb.add_vertex(1);
        gb.add_vertex(2);
        gb.add_edge(0, 1, 1.0);
        gb.add_edge(0, 2, 3.0);
        gb.add_edge(1, 2, 2.0);
        gb.add_edge(2, 0, 4.0);
        let g = gb.build();

        let a = g.adjacency_matrix();
        let d = g.degree_matrix();
        let manual = d.partial_piv_lu().inverse().mul(&a);
        let via_method = g.transition_probability_matrix();

        assert_eq!(manual.nrows(), via_method.nrows());
        assert_eq!(manual.ncols(), via_method.ncols());
        for i in 0..manual.nrows() {
            for j in 0..manual.ncols() {
                assert!(
                    approx_eq(manual[(i, j)], via_method[(i, j)], 1e-12),
                    "Mismatch at ({i},{j}): manual={}, method={}",
                    manual[(i, j)],
                    via_method[(i, j)]
                );
            }
        }
    }
}
