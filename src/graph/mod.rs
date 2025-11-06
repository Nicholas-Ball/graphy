use std::fmt::Debug;

use nalgebra::Scalar;

/// Represents a graph with vertex values of type `T` and edge values of type `V`.
///
/// # Fields
///
/// * `vertexes` - A vector containing the values of the graph's vertices.
/// * `edges` - A 2D array (adjacency matrix) representing the edge values between vertices.
/// edges: symmetric adjacency matrix (undirected)
/// adjacency_directed: asymmetric adjacency matrix (directed)
pub struct Graph<T, V: Copy + num_traits::identities::Zero + nalgebra::Scalar> {
    pub vertexes: Vec<T>,

    pub adjacency: nalgebra::DMatrix<V>,
    pub degrees: nalgebra::DMatrix<V>,
    pub incoming_degrees: nalgebra::DMatrix<V>,
    pub outgoing_degrees: nalgebra::DMatrix<V>,
}

/// A builder for constructing a `Graph`.
///
/// # Fields
///
/// * `vertexes` - A vector storing the values of vertices to be added to the graph.
/// * `edges` - A vector of tuples representing edges as (from, to, value).
#[derive(Default)]
pub struct GraphBuilder<T, V: num_traits::identities::Zero + Copy + nalgebra::Scalar> {
    vertexes: Vec<T>,
    edges: Vec<(usize, usize, V)>,
}

impl<T, V: num_traits::identities::Zero + Copy + num_traits::identities::One + nalgebra::Scalar>
    GraphBuilder<T, V>
{
    /// Creates a new, empty `GraphBuilder`.
    ///
    /// # Returns
    ///
    /// A new instance of `GraphBuilder` with no vertices or edges.
    pub fn new() -> Self {
        Self {
            vertexes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Adds a vertex to the graph.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to assign to the new vertex.
    ///
    /// # Returns
    ///
    /// The index of the newly added vertex.
    pub fn add_vertex(&mut self, value: T) -> &mut Self {
        self.vertexes.push(value);
        self
    }

    /// Adds an edge to the graph from one vertex to another with a specified value.
    ///
    /// # Arguments
    ///
    /// * `from` - The index of the source vertex.
    /// * `to` - The index of the destination vertex.
    /// * `value` - The value to assign to the edge.
    pub fn add_edge(&mut self, from: usize, to: usize, value: V) -> &mut Self {
        self.edges.push((from, to, value));
        self
    }

    pub fn build(self) -> Graph<T, V>
    where
        V: nalgebra::Scalar + num_traits::Zero + num_traits::One + Copy + std::ops::AddAssign,
    {
        let n = self.vertexes.len();
        let mut adjacency_matrix = nalgebra::DMatrix::<V>::zeros(n, n);
        let mut degree_matrix = nalgebra::DMatrix::<V>::zeros(n, n);
        let mut incoming_degree_matrix = nalgebra::DMatrix::<V>::zeros(n, n);
        let mut outgoing_degree_matrix = nalgebra::DMatrix::<V>::zeros(n, n);

        for (from, to, value) in self.edges.iter() {
            adjacency_matrix[(*to, *from)] = *value;
            adjacency_matrix[(*from, *to)] = *value;

            // Update weighted degrees
            outgoing_degree_matrix[(*from, *from)] += *value;
            incoming_degree_matrix[(*to, *to)] += *value;
            degree_matrix[(*from, *from)] += *value;
            degree_matrix[(*to, *to)] += *value;
        }

        Graph {
            vertexes: self.vertexes,
            adjacency: adjacency_matrix,
            degrees: degree_matrix,
            incoming_degrees: incoming_degree_matrix,
            outgoing_degrees: outgoing_degree_matrix,
        }
    }
}

impl<T, V: Copy + Debug + Scalar + num_traits::Zero> Graph<T, V> {
    /// Retrieves the value of the edge from one vertex to another.
    ///
    /// # Arguments
    ///
    /// * `from` - The index of the source vertex.
    /// * `to` - The index of the destination vertex.
    ///
    /// # Returns
    ///
    /// The value of the edge from `from` to `to`.
    pub fn get_edge_value(&self, from: usize, to: usize) -> V {
        self.adjacency[(from, to)]
    }

    /// Returns the number of vertices in the graph.
    ///
    /// # Returns
    ///
    /// The number of vertices (rank) in the graph.
    pub fn get_rank(&self) -> usize {
        self.vertexes.len()
    }

    /// Retrieves a reference to the value of a vertex by its index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the vertex.
    ///
    /// # Returns
    ///
    /// A reference to the value of the vertex at the given index.
    pub fn get_vertex_value(&self, index: usize) -> &T {
        &self.vertexes[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::identities::Zero;

    #[test]
    fn test_add_vertex_and_indices() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        let idx1 = builder.vertexes.len();
        builder.add_vertex(42);
        let idx2 = builder.vertexes.len();
        builder.add_vertex(99);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        let g = builder.build();
        assert_eq!(g.vertexes, vec![42, 99]);
        assert_eq!(g.adjacency.shape(), (2, 2));
        assert_eq!(g.adjacency[(0, 0)], 0);
        assert_eq!(g.adjacency[(0, 1)], 0);
        assert_eq!(g.adjacency[(1, 0)], 0);
        assert_eq!(g.adjacency[(1, 1)], 0);
    }

    #[test]
    fn test_add_edges_and_matrix_values() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_vertex(2);
        let a = 0;
        let b = 1;
        let c = 2;
        builder.add_edge(a, b, 10);
        builder.add_edge(b, c, 20);
        builder.add_edge(c, a, 30);
        let g = builder.build();
        assert_eq!(g.vertexes, vec![0, 1, 2]);
        assert_eq!(g.adjacency.shape(), (3, 3));
        // adjacency is symmetric: store at both (to, from) and (from, to)
        assert_eq!(g.adjacency[(b, a)], 10);
        assert_eq!(g.adjacency[(a, b)], 10);
        assert_eq!(g.adjacency[(c, b)], 20);
        assert_eq!(g.adjacency[(b, c)], 20);
        assert_eq!(g.adjacency[(a, c)], 30);
        assert_eq!(g.adjacency[(c, a)], 30);
        // Check unset edges are zero
        for i in 0..3 {
            for j in 0..3 {
                let is_edge = (i == b && j == a)
                    || (i == a && j == b)
                    || (i == c && j == b)
                    || (i == b && j == c)
                    || (i == a && j == c)
                    || (i == c && j == a);
                if !is_edge {
                    assert_eq!(
                        g.adjacency[(i, j)],
                        i32::zero(),
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
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(0);
        builder.add_vertex(0);
        let u = 0;
        let v = 1;
        builder.add_edge(u, v, 1);
        builder.add_edge(u, v, 2);
        let g = builder.build();
        // edges matrix stores at (to, from)
        assert_eq!(g.adjacency[(v, u)], 2);
    }

    #[test]
    fn test_empty_graph() {
        let builder = GraphBuilder::<i32, i32>::new();
        let g = builder.build();
        assert_eq!(g.vertexes.len(), 0);
        assert_eq!(g.adjacency.shape(), (0, 0));
    }

    #[test]
    fn test_get_edge_value_and_vertex_value() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(7);
        builder.add_vertex(8);
        let a = 0;
        let b = 1;
        builder.add_edge(a, b, 15);
        let g = builder.build();
        // get_edge_value uses the edges matrix which stores at (to, from)
        assert_eq!(g.get_edge_value(b, a), 15);
        assert_eq!(*g.get_vertex_value(a), 7);
        assert_eq!(*g.get_vertex_value(b), 8);
    }

    #[test]
    fn test_get_rank() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        for i in 0..5 {
            builder.add_vertex(i);
        }
        let g = builder.build();
        assert_eq!(g.get_rank(), 5);
    }

    #[test]
    fn test_default_builder() {
        let mut builder: GraphBuilder<i32, i32> = Default::default();
        let idx = builder.vertexes.len();
        builder.add_vertex(123);
        assert_eq!(idx, 0);
        let g = builder.build();
        assert_eq!(g.vertexes, vec![123]);
        assert_eq!(g.adjacency.shape(), (1, 1));
        assert_eq!(g.adjacency[(0, 0)], 0);
    }

    // New tests

    #[test]
    fn test_degree_matrices_for_simple_cycle() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        // Vertices: 0,1,2 with edges 0->1, 1->2, 2->0 (a directed 3-cycle)
        for v in 1..=3 {
            builder.add_vertex(v);
        }
        builder.add_edge(0, 1, 10);
        builder.add_edge(1, 2, 20);
        builder.add_edge(2, 0, 30);
        let g = builder.build();

        // Outgoing and incoming on diagonals are weighted by edge values
        assert_eq!(g.outgoing_degrees[(0, 0)], 10);
        assert_eq!(g.outgoing_degrees[(1, 1)], 20);
        assert_eq!(g.outgoing_degrees[(2, 2)], 30);

        assert_eq!(g.incoming_degrees[(0, 0)], 30);
        assert_eq!(g.incoming_degrees[(1, 1)], 10);
        assert_eq!(g.incoming_degrees[(2, 2)], 20);

        // Total degree = in + out (weighted)
        assert_eq!(g.degrees[(0, 0)], 40);
        assert_eq!(g.degrees[(1, 1)], 30);
        assert_eq!(g.degrees[(2, 2)], 50);
    }

    #[test]
    fn test_degrees_with_self_loop_and_parallel_edges() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(0);
        builder.add_vertex(1);
        // Self-loop on 0
        builder.add_edge(0, 0, 99);
        // Two parallel edges 0->1 (last value wins for edge matrix)
        builder.add_edge(0, 1, 1);
        builder.add_edge(0, 1, 2);
        // One edge 1->0
        builder.add_edge(1, 0, 3);

        let g = builder.build();

        // Outgoing: weighted sums
        assert_eq!(g.outgoing_degrees[(0, 0)], 99 + 1 + 2);
        assert_eq!(g.outgoing_degrees[(1, 1)], 3);

        // Incoming: weighted sums
        assert_eq!(g.incoming_degrees[(0, 0)], 99 + 3);
        assert_eq!(g.incoming_degrees[(1, 1)], 1 + 2);

        // Total degrees: weighted sums
        assert_eq!(g.degrees[(0, 0)], (99 + 1 + 2) + (99 + 3));
        assert_eq!(g.degrees[(1, 1)], 3 + (1 + 2));

        // Adjacency is symmetric; the last assignment between 0 and 1 was 1->0 with weight 3
        assert_eq!(g.adjacency[(1, 0)], 3);
        assert_eq!(g.adjacency[(0, 1)], 3);
        // Self-loop should be present
        assert_eq!(g.adjacency[(0, 0)], 99);
    }

    #[test]
    fn test_degree_matrices_shapes_empty_and_no_edges() {
        // Empty graph
        let builder_empty = GraphBuilder::<i32, i32>::new();
        let g_empty = builder_empty.build();
        assert_eq!(g_empty.degrees.shape(), (0, 0));
        assert_eq!(g_empty.incoming_degrees.shape(), (0, 0));
        assert_eq!(g_empty.outgoing_degrees.shape(), (0, 0));

        // Graph with 3 isolated vertices (no edges)
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(10);
        builder.add_vertex(20);
        builder.add_vertex(30);
        let g = builder.build();
        assert_eq!(g.degrees.shape(), (3, 3));
        assert_eq!(g.incoming_degrees.shape(), (3, 3));
        assert_eq!(g.outgoing_degrees.shape(), (3, 3));
        for i in 0..3 {
            assert_eq!(g.degrees[(i, i)], 0);
            assert_eq!(g.incoming_degrees[(i, i)], 0);
            assert_eq!(g.outgoing_degrees[(i, i)], 0);
        }
    }

    #[test]
    fn test_non_integer_weights_and_degrees_f64() {
        let mut builder = GraphBuilder::<i32, f64>::new();
        builder.add_vertex(1);
        builder.add_vertex(2);
        builder.add_edge(0, 1, 2.5);
        let g = builder.build();
        assert_eq!(g.adjacency.shape(), (2, 2));
        // edges matrix stores at (to, from)
        assert_eq!(g.adjacency[(1, 0)], 2.5_f64);

        // Degree entries are weighted by edge values (f64)
        assert_eq!(g.outgoing_degrees[(0, 0)], 2.5);
        assert_eq!(g.outgoing_degrees[(1, 1)], 0.0);
        assert_eq!(g.incoming_degrees[(0, 0)], 0.0);
        assert_eq!(g.incoming_degrees[(1, 1)], 2.5);
        assert_eq!(g.degrees[(0, 0)], 2.5);
        assert_eq!(g.degrees[(1, 1)], 2.5);
    }

    #[test]
    fn test_generic_vertex_values() {
        let mut builder = GraphBuilder::<String, i32>::new();
        builder.add_vertex("alpha".to_string());
        builder.add_vertex("beta".to_string());
        builder.add_edge(0, 1, 7);
        let g = builder.build();
        assert_eq!(g.vertexes.len(), 2);
        assert_eq!(g.get_vertex_value(0), "alpha");
        assert_eq!(g.get_vertex_value(1), "beta");
        // get_edge_value reads from edges (stored at (to, from))
        assert_eq!(g.get_edge_value(1, 0), 7);
    }

    #[test]
    fn test_degree_sum_equals_twice_edge_count_in_directed_graph() {
        let n = 10;
        let mut builder = GraphBuilder::<i32, i32>::new();
        for i in 0..n {
            builder.add_vertex(i as i32);
        }
        // Create a path: 0->1->2->...->9, edges = n-1
        for i in 0..(n - 1) {
            builder.add_edge(i, i + 1, 1);
        }
        let g = builder.build();
        let mut sum_deg = 0i32;
        let mut sum_out = 0i32;
        let mut sum_in = 0i32;
        for i in 0..n {
            sum_deg += g.degrees[(i, i)];
            sum_out += g.outgoing_degrees[(i, i)];
            sum_in += g.incoming_degrees[(i, i)];
        }
        let m = (n - 1) as i32;
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
            2 * m,
            "sum of total degrees should be twice the number of edges"
        );
    }

    #[test]
    fn test_get_edge_value_unset_is_zero() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(0);
        builder.add_vertex(0);
        let g = builder.build();
        assert_eq!(g.get_edge_value(0, 1), i32::zero());
        assert_eq!(g.get_edge_value(1, 0), i32::zero());
    }

    #[test]
    fn test_incoming_outgoing_totals_consistency() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        for _ in 0..4 {
            builder.add_vertex(0);
        }
        // Edges: 0->1, 0->2, 1->2, 3->0 with weights
        builder.add_edge(0, 1, 5);
        builder.add_edge(0, 2, 6);
        builder.add_edge(1, 2, 7);
        builder.add_edge(3, 0, 8);
        let g = builder.build();

        let mut sum_out = 0i32;
        let mut sum_in = 0i32;
        for i in 0..4 {
            sum_out += g.outgoing_degrees[(i, i)];
            sum_in += g.incoming_degrees[(i, i)];
        }
        assert_eq!(sum_out, 5 + 6 + 7 + 8);
        assert_eq!(sum_in, 5 + 6 + 7 + 8);

        // Spot-check some individual weighted degrees
        assert_eq!(g.outgoing_degrees[(0, 0)], 5 + 6); // 0->1 (5), 0->2 (6)
        assert_eq!(g.incoming_degrees[(2, 2)], 6 + 7); // from 0 (6) and 1 (7)
        assert_eq!(g.incoming_degrees[(0, 0)], 8); // from 3 (8)
        assert_eq!(g.outgoing_degrees[(3, 3)], 8); // to 0 (8)
    }
}
