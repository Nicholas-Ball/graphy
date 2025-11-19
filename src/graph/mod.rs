pub mod builder;
pub mod edge;
pub mod vertex;

use std::collections::HashMap;

/// Extension providing a nalgebra-like `.shape()` on faer matrices for tests.
pub trait MatrixShape {
    fn shape(&self) -> (usize, usize);
}

impl<V> MatrixShape for faer::Mat<V> {
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

/// Represents a graph with vertex values of type `T` and edge values of type `V`.
///
/// # Fields
///
/// * `vertices` - A vector containing the values of the graph's vertices.
/// * `adjacency` - A 2D array (adjacency matrix) representing (undirected/symmetric) edge weights.
/// * `degrees`, `incoming_degrees`, `outgoing_degrees` - Diagonal matrices with UNWEIGHTED (in/out/total) degrees
///   expressed as natural counts (stored as `V`, currently floats).
/// * `index_map` - Maps vertex values to their internal contiguous indices.
/// edges: symmetric adjacency matrix (undirected)
/// adjacency_directed: asymmetric adjacency matrix (directed)
pub struct Graph<T, V> {
    pub vertices: Vec<T>,
    pub adjacency_undirectional: faer::Mat<V>,
    pub adjacency_directional: faer::Mat<V>,
    pub degrees: faer::Mat<V>,
    pub incoming_degrees: faer::Mat<V>,
    pub outgoing_degrees: faer::Mat<V>,
    pub index_map: HashMap<T, usize>,
}

impl<T, V> Graph<T, V> {
    /// Returns the number of vertices in the graph.
    ///
    /// # Returns
    ///
    /// The number of vertices (rank) in the graph.
    pub fn get_rank(&self) -> usize {
        self.vertices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;
    use num_traits::identities::Zero;

    #[test]
    fn test_add_vertex_and_indices() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        let idx1 = builder.vertices.len();
        builder.add_vertex(42);
        let idx2 = builder.vertices.len();
        builder.add_vertex(99);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        let g = builder.build();
        assert_eq!(g.vertices, vec![42, 99]);
        assert_eq!(g.adjacency_undirectional.shape(), (2, 2));
        assert_eq!(g.adjacency_undirectional[(0, 0)], 0.0);
        assert_eq!(g.adjacency_undirectional[(0, 1)], 0.0);
        assert_eq!(g.adjacency_undirectional[(1, 0)], 0.0);
        assert_eq!(g.adjacency_undirectional[(1, 1)], 0.0);
    }

    #[test]
    fn test_add_edges_and_matrix_values() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_vertex(2);
        let a = 0;
        let b = 1;
        let c = 2;
        builder.add_edge(a, b, 10.0);
        builder.add_edge(b, c, 20.0);
        builder.add_edge(c, a, 30.0);
        let g = builder.build();
        assert_eq!(g.vertices, vec![0, 1, 2]);
        assert_eq!(g.adjacency_undirectional.shape(), (3, 3));

        let ai = g.get_vertex_index(&a).unwrap();
        let bi = g.get_vertex_index(&b).unwrap();
        let ci = g.get_vertex_index(&c).unwrap();

        // adjacency is symmetric: store at both (to, from) and (from, to)
        assert_eq!(g.adjacency_undirectional[(bi, ai)], 10.0);
        assert_eq!(g.adjacency_undirectional[(ai, bi)], 10.0);
        assert_eq!(g.adjacency_undirectional[(ci, bi)], 20.0);
        assert_eq!(g.adjacency_undirectional[(bi, ci)], 20.0);
        assert_eq!(g.adjacency_undirectional[(ai, ci)], 30.0);
        assert_eq!(g.adjacency_undirectional[(ci, ai)], 30.0);

        // Check unset edges are zero
        for i in 0..3 {
            for j in 0..3 {
                let is_edge = (i == bi && j == ai)
                    || (i == ai && j == bi)
                    || (i == ci && j == bi)
                    || (i == bi && j == ci)
                    || (i == ai && j == ci)
                    || (i == ci && j == ai);
                if !is_edge {
                    assert_eq!(
                        g.adjacency_undirectional[(i, j)],
                        f64::zero(),
                        "Expected zero at ({}, {})",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_multiple_edges_last_assignment() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        let u = 0;
        let v = 1;
        builder.add_edge(u, v, 1.0);
        builder.add_edge(u, v, 2.0);
        let g = builder.build();
        let ui = g.get_vertex_index(&u).unwrap();
        let vi = g.get_vertex_index(&v).unwrap();
        // edges matrix stores at (to, from)
        assert_eq!(g.adjacency_undirectional[(vi, ui)], 1.0);
    }

    #[test]
    fn test_empty_graph() {
        let builder = GraphBuilder::<i32, f64>::new();
        let g = builder.build();
        assert_eq!(g.vertices.len(), 0);
        assert_eq!(g.adjacency_undirectional.shape(), (0, 0));
    }

    #[test]
    fn test_get_edge_value_and_vertex_value() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(7);
        builder.add_vertex(8);
        builder.add_edge(7, 8, 15.0);
        let g = builder.build();
        assert_eq!(g.get_edge_value(&8, &7), 15.0);
        let idx7 = g.get_vertex_index(&7).unwrap();
        let idx8 = g.get_vertex_index(&8).unwrap();
        assert_eq!(*g.get_vertex_value(idx7), 7);
        assert_eq!(*g.get_vertex_value(idx8), 8);
    }

    #[test]
    fn test_get_rank() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        for i in 0..5 {
            builder.add_vertex(i);
        }
        let g = builder.build();
        assert_eq!(g.get_rank(), 5);
    }

    #[test]
    fn test_default_builder() {
        let mut builder: GraphBuilder<i32, f64> = Default::default();
        let idx = builder.vertices.len();
        builder.add_vertex(123);
        assert_eq!(idx, 0);
        let g = builder.build();
        assert_eq!(g.vertices, vec![123]);
        assert_eq!(g.adjacency_undirectional.shape(), (1, 1));
        assert_eq!(g.adjacency_undirectional[(0, 0)], 0.0);
    }

    // New tests (adjusted for UNWEIGHTED degree counts)

    #[test]
    fn test_degree_matrices_for_simple_cycle() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        // Vertices: 0,1,2 with edges 0->1, 1->2, 2->0 (a directed 3-cycle)
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_vertex(2);
        builder.add_edge(0, 1, 10.0);
        builder.add_edge(1, 2, 20.0);
        builder.add_edge(2, 0, 30.0);
        let g = builder.build();

        // Unweighted counts: each vertex has 1 outgoing edge
        assert_eq!(
            g.outgoing_degrees[(
                g.get_vertex_index(&0).unwrap(),
                g.get_vertex_index(&0).unwrap()
            )],
            1.0
        );
        assert_eq!(
            g.outgoing_degrees[(
                g.get_vertex_index(&1).unwrap(),
                g.get_vertex_index(&1).unwrap()
            )],
            1.0
        );
        assert_eq!(
            g.outgoing_degrees[(
                g.get_vertex_index(&2).unwrap(),
                g.get_vertex_index(&2).unwrap()
            )],
            1.0
        );

        // Unweighted counts: each vertex has 1 incoming edge
        assert_eq!(
            g.incoming_degrees[(
                g.get_vertex_index(&0).unwrap(),
                g.get_vertex_index(&0).unwrap()
            )],
            1.0
        );
        assert_eq!(
            g.incoming_degrees[(
                g.get_vertex_index(&1).unwrap(),
                g.get_vertex_index(&1).unwrap()
            )],
            1.0
        );
        assert_eq!(
            g.incoming_degrees[(
                g.get_vertex_index(&2).unwrap(),
                g.get_vertex_index(&2).unwrap()
            )],
            1.0
        );

        // Total degree = in + out counts
        assert_eq!(
            g.degrees[(
                g.get_vertex_index(&0).unwrap(),
                g.get_vertex_index(&0).unwrap()
            )],
            2.0
        );
        assert_eq!(
            g.degrees[(
                g.get_vertex_index(&1).unwrap(),
                g.get_vertex_index(&1).unwrap()
            )],
            2.0
        );
        assert_eq!(
            g.degrees[(
                g.get_vertex_index(&2).unwrap(),
                g.get_vertex_index(&2).unwrap()
            )],
            2.0
        );
    }

    #[test]
    fn test_degrees_with_self_loop_and_parallel_edges() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        // Self-loop on 0
        builder.add_edge(0, 0, 99.0);
        // Two parallel edges 0->1 (count both)
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(0, 1, 2.0);
        // One edge 1->0
        builder.add_edge(1, 0, 3.0);

        let g = builder.build();

        let i0 = g.get_vertex_index(&0).unwrap();
        let i1 = g.get_vertex_index(&1).unwrap();

        // Outgoing: counts (loop + two parallel edges) and single edge from 1
        assert_eq!(g.outgoing_degrees[(i0, i0)], 2.0);
        assert_eq!(g.outgoing_degrees[(i1, i1)], 1.0);

        // Incoming: counts (loop + one from 1) and two from 0
        assert_eq!(g.incoming_degrees[(i0, i0)], 2.0);
        assert_eq!(g.incoming_degrees[(i1, i1)], 1.0);

        // Total degrees: counts
        assert_eq!(g.degrees[(i0, i0)], 2.0);
        assert_eq!(g.degrees[(i1, i1)], 1.0);

        // Adjacency is symmetric; last assignment between 0 and 1 was 1->0 with weight 3
        assert_eq!(g.adjacency_undirectional[(i1, i0)], 1.0);
        assert_eq!(g.adjacency_undirectional[(i0, i1)], 1.0);
        // Self-loop should be present
        assert_eq!(g.adjacency_undirectional[(i0, i0)], 99.0);
    }

    #[test]
    fn test_degree_matrices_shapes_empty_and_no_edges() {
        // Empty graph
        let builder_empty = GraphBuilder::<i32, f64>::new();
        let g_empty = builder_empty.build();
        assert_eq!(g_empty.degrees.shape(), (0, 0));
        assert_eq!(g_empty.incoming_degrees.shape(), (0, 0));
        assert_eq!(g_empty.outgoing_degrees.shape(), (0, 0));

        // Graph with 3 isolated vertices (no edges)
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(10);
        builder.add_vertex(20);
        builder.add_vertex(30);
        let g = builder.build();
        assert_eq!(g.degrees.shape(), (3, 3));
        assert_eq!(g.incoming_degrees.shape(), (3, 3));
        assert_eq!(g.outgoing_degrees.shape(), (3, 3));
        for i in 0..3 {
            assert_eq!(g.degrees[(i, i)], 0.0);
            assert_eq!(g.incoming_degrees[(i, i)], 0.0);
            assert_eq!(g.outgoing_degrees[(i, i)], 0.0);
        }
    }

    #[test]
    fn test_non_integer_weights_and_degrees_f64() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_edge(0, 1, 2.5);
        let g = builder.build();
        assert_eq!(g.adjacency_undirectional.shape(), (2, 2));
        let i0 = g.get_vertex_index(&0).unwrap();
        let i1 = g.get_vertex_index(&1).unwrap();
        // edges matrix stores at (to, from)
        assert_eq!(g.adjacency_undirectional[(i1, i0)], 2.5_f64);

        // Degree entries are UNWEIGHTED counts
        assert_eq!(g.outgoing_degrees[(i0, i0)], 1.0);
        assert_eq!(g.outgoing_degrees[(i1, i1)], 0.0);
        assert_eq!(g.incoming_degrees[(i0, i0)], 0.0);
        assert_eq!(g.incoming_degrees[(i1, i1)], 1.0);
        assert_eq!(g.degrees[(i0, i0)], 1.0);
        assert_eq!(g.degrees[(i1, i1)], 1.0);
    }

    #[test]
    fn test_generic_vertex_values() {
        let mut builder = GraphBuilder::<String, f64>::new();
        let alpha = "alpha".to_string();
        let beta = "beta".to_string();
        builder.add_vertex(alpha.clone());
        builder.add_vertex(beta.clone());
        builder.add_edge(alpha.clone(), beta.clone(), 7.0);
        let g = builder.build();
        assert_eq!(g.vertices.len(), 2);
        let alpha_idx = g.get_vertex_index(&alpha).unwrap();
        let beta_idx = g.get_vertex_index(&beta).unwrap();
        assert_eq!(g.get_vertex_value(alpha_idx), "alpha");
        assert_eq!(g.get_vertex_value(beta_idx), "beta");
        assert_eq!(g.get_edge_value_by_vertices(&beta, &alpha).unwrap(), 7.0);
        assert_eq!(g.get_edge_value_by_vertices(&alpha, &beta).unwrap(), 7.0);

        // Degree counts (one edge alpha->beta)
        assert_eq!(g.outgoing_degrees[(alpha_idx, alpha_idx)], 1.0);
        assert_eq!(g.incoming_degrees[(beta_idx, beta_idx)], 1.0);
        assert_eq!(g.degrees[(alpha_idx, alpha_idx)], 1.0);
        assert_eq!(g.degrees[(beta_idx, beta_idx)], 1.0);
    }

    #[test]
    fn test_degree_sum_equals_twice_edge_count_in_directed_graph() {
        let n = 10;
        let mut builder = GraphBuilder::<i32, f64>::new();
        for i in 0..n {
            builder.add_vertex(i as i32);
        }
        // Create a path: 0->1->2->...->9, edges = n-1
        for i in 0..(n - 1) {
            builder.add_edge(i as i32, (i + 1) as i32, 1.0);
        }
        let g = builder.build();
        let mut sum_deg = 0.0f64;
        let mut sum_out = 0.0f64;
        let mut sum_in = 0.0f64;
        for i in 0..n {
            sum_deg += g.degrees[(i, i)];
            sum_out += g.outgoing_degrees[(i, i)];
            sum_in += g.incoming_degrees[(i, i)];
        }
        let m = (n - 1) as f64;
        assert_eq!(
            sum_out, m,
            "sum of outgoing degrees should equal number of edges"
        );
        assert_eq!(
            sum_in, m,
            "sum of incoming degrees should equal number of edges"
        );
        assert_eq!(
            sum_deg,
            2.0 * m,
            "sum of total degrees should be twice the number of edges"
        );
    }

    #[test]
    fn test_get_edge_value_unset_is_zero() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        let g = builder.build();
        assert_eq!(g.get_edge_value(&0, &1), f64::zero());
        assert_eq!(g.get_edge_value(&1, &0), f64::zero());
    }

    #[test]
    fn test_incoming_outgoing_totals_consistency() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        for n in 0..4 {
            builder.add_vertex(n);
        }
        // Edges: 0->1, 0->2, 1->2, 3->0
        builder.add_edge(0, 1, 5.0);
        builder.add_edge(0, 2, 6.0);
        builder.add_edge(1, 2, 7.0);
        builder.add_edge(3, 0, 8.0);
        let g = builder.build();

        let mut sum_out = 0.0f64;
        let mut sum_in = 0.0f64;
        for i in 0..4 {
            sum_out += g.outgoing_degrees[(i, i)];
            sum_in += g.incoming_degrees[(i, i)];
        }
        assert_eq!(sum_out, 4.0); // total edges
        assert_eq!(sum_in, 4.0); // total edges

        // Spot-check some individual UNWEIGHTED degrees
        let i0 = g.get_vertex_index(&0).unwrap();
        let i2 = g.get_vertex_index(&2).unwrap();
        let i3 = g.get_vertex_index(&3).unwrap();
        assert_eq!(g.outgoing_degrees[(i0, i0)], 2.0); // 0->1, 0->2
        assert_eq!(g.incoming_degrees[(i2, i2)], 2.0); // from 0 and 1
        assert_eq!(g.incoming_degrees[(i0, i0)], 1.0); // from 3
        assert_eq!(g.outgoing_degrees[(i3, i3)], 1.0); // to 0
    }

    #[test]
    fn test_add_edge_by_values_and_lookup() {
        let mut builder = GraphBuilder::<&'static str, f64>::new();
        builder.add_vertex("A").add_vertex("B");
        builder.add_edge("A", "B", 11.0);
        builder.add_edge("A", "C", 7.0); // "C" auto-inserted
        let g = builder.build();
        let a_idx = g.get_vertex_index(&"A").unwrap();
        let b_idx = g.get_vertex_index(&"B").unwrap();
        let c_idx = g.get_vertex_index(&"C").unwrap();
        assert_eq!(g.adjacency_undirectional[(b_idx, a_idx)], 11.0);
        assert_eq!(g.adjacency_undirectional[(a_idx, b_idx)], 11.0);
        assert_eq!(g.adjacency_undirectional[(c_idx, a_idx)], 7.0);
        assert_eq!(g.get_edge_value_by_vertices(&"A", &"B").unwrap(), 11.0);
        assert_eq!(g.get_edge_value_by_vertices(&"C", &"A").unwrap(), 7.0);

        // Degree counts: A has 2 outgoing (A->B, A->C), B has 1 incoming, C has 1 incoming
        assert_eq!(g.outgoing_degrees[(a_idx, a_idx)], 2.0);
        assert_eq!(g.incoming_degrees[(b_idx, b_idx)], 1.0);
        assert_eq!(g.incoming_degrees[(c_idx, c_idx)], 1.0);
    }
}
