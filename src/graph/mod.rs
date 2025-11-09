use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use nalgebra::Scalar;

/// Represents a graph with vertex values of type `T` and edge values of type `V`.
///
/// # Fields
///
/// * `vertices` - A vector containing the values of the graph's vertices.
/// * `adjacency` - A 2D array (adjacency matrix) representing (undirected/symmetric) edge weights.
/// * `degrees`, `incoming_degrees`, `outgoing_degrees` - Diagonal matrices with weighted (in/out/total) degrees.
/// * `index_map` - Maps vertex values to their internal contiguous indices.
/// edges: symmetric adjacency matrix (undirected)
/// adjacency_directed: asymmetric adjacency matrix (directed)
pub struct Graph<T: Eq + Hash + Clone, V: Copy + num_traits::identities::Zero + nalgebra::Scalar> {
    pub vertices: Vec<T>,
    pub adjacency: nalgebra::DMatrix<V>,
    pub degrees: nalgebra::DMatrix<V>,
    pub incoming_degrees: nalgebra::DMatrix<V>,
    pub outgoing_degrees: nalgebra::DMatrix<V>,
    pub index_map: HashMap<T, usize>,
}

/// A builder for constructing a `Graph`.
///
/// # Fields
///
/// * `vertices` - A vector storing the values of vertices to be added to the graph.
/// * `edges` - A vector of tuples representing edges as (from_index, to_index, value).
#[derive(Default)]
pub struct GraphBuilder<
    T: Eq + Hash + Clone,
    V: num_traits::identities::Zero + Copy + nalgebra::Scalar,
> {
    vertices: Vec<T>,
    edges: Vec<(usize, usize, V)>,
    index_map: HashMap<T, usize>,
}

impl<
    T: Eq + Hash + Clone,
    V: num_traits::identities::Zero + Copy + num_traits::identities::One + nalgebra::Scalar,
> GraphBuilder<T, V>
{
    /// Creates a new, empty `GraphBuilder`.
    ///
    /// # Returns
    ///
    /// A new instance of `GraphBuilder` with no vertices or edges.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            index_map: HashMap::new(),
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
    /// The builder (for chaining).
    ///
    /// Note: If the vertex value already exists it will NOT create a duplicate; the existing
    /// index is retained (idempotent insertion). This keeps T independent of positional indices.
    pub fn add_vertex(&mut self, value: T) -> &mut Self {
        if !self.index_map.contains_key(&value) {
            let idx = self.vertices.len();
            self.vertices.push(value.clone());
            self.index_map.insert(value, idx);
        }
        self
    }

    /// Adds an edge by vertex values `from_val` -> `to_val` with a specified weight.
    ///
    /// If either endpoint does not yet exist, it is inserted automatically.
    pub fn add_edge(&mut self, from_val: T, to_val: T, value: V) -> &mut Self {
        // Ensure vertices exist (idempotent).
        let from_idx = self.get_or_insert_index(from_val);
        let to_idx = self.get_or_insert_index(to_val);
        self.edges.push((from_idx, to_idx, value));
        self
    }

    /// Adds an edge by borrowing vertex values; vertices must already exist.
    pub fn add_edge_ref(&mut self, from: &T, to: &T, value: V) -> &mut Self {
        if let (Some(&from_idx), Some(&to_idx)) = (self.index_map.get(from), self.index_map.get(to))
        {
            self.edges.push((from_idx, to_idx, value));
        } else {
            panic!("Attempted to add_edge_ref with unknown vertex value(s).");
        }
        self
    }

    fn get_or_insert_index(&mut self, v: T) -> usize {
        if let Some(&idx) = self.index_map.get(&v) {
            idx
        } else {
            let idx = self.vertices.len();
            self.vertices.push(v.clone());
            self.index_map.insert(v, idx);
            idx
        }
    }

    pub fn build(self) -> Graph<T, V>
    where
        V: nalgebra::Scalar + num_traits::Zero + num_traits::One + Copy + std::ops::AddAssign,
    {
        let n = self.vertices.len();
        let mut adjacency_matrix = nalgebra::DMatrix::<V>::zeros(n, n);
        let mut degree_matrix = nalgebra::DMatrix::<V>::zeros(n, n);
        let mut incoming_degree_matrix = nalgebra::DMatrix::<V>::zeros(n, n);
        let mut outgoing_degree_matrix = nalgebra::DMatrix::<V>::zeros(n, n);

        for (from, to, value) in self.edges.iter() {
            // Store directed edge weight at (to, from)
            adjacency_matrix[(*to, *from)] = *value;
            adjacency_matrix[(*from, *to)] = *value;

            // Update weighted degrees
            outgoing_degree_matrix[(*from, *from)] += *value;
            incoming_degree_matrix[(*to, *to)] += *value;
            degree_matrix[(*from, *from)] += *value;
            degree_matrix[(*to, *to)] += *value;
        }

        Graph {
            vertices: self.vertices,
            adjacency: adjacency_matrix,
            degrees: degree_matrix,
            incoming_degrees: incoming_degree_matrix,
            outgoing_degrees: outgoing_degree_matrix,
            index_map: self.index_map,
        }
    }
}

impl<T: Eq + Hash + Clone, V: Copy + Debug + Scalar + num_traits::Zero> Graph<T, V> {
    /// Retrieves the value of the edge from one vertex index to another.
    ///
    /// # Arguments
    ///
    /// * `from` - The index of the source vertex.
    /// * `to` - The index of the destination vertex.
    ///
    /// # Returns
    ///
    /// The value of the edge from `from` to `to`.
    pub fn get_edge_value(&self, from: &T, to: &T) -> V {
        let idx = (
            *self.index_map.get(to).expect("Unknown 'to' vertex value."),
            *self
                .index_map
                .get(from)
                .expect("Unknown 'from' vertex value."),
        );

        self.adjacency[idx]
    }

    /// Retrieves the edge value using vertex values instead of indices.
    ///
    /// Returns Some(weight) if both vertices exist, otherwise None.
    pub fn get_edge_value_by_vertices(&self, from: &T, to: &T) -> Option<V> {
        let from_idx = self.index_map.get(from)?;
        let to_idx = self.index_map.get(to)?;
        Some(self.adjacency[(*to_idx, *from_idx)])
    }

    /// Returns the number of vertices in the graph.
    ///
    /// # Returns
    ///
    /// The number of vertices (rank) in the graph.
    pub fn get_rank(&self) -> usize {
        self.vertices.len()
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
        &self.vertices[index]
    }

    /// Returns the internal index of a vertex value, if present.
    pub fn get_vertex_index(&self, value: &T) -> Option<usize> {
        self.index_map.get(value).copied()
    }

    pub fn get_vertices(&self) -> &Vec<T> {
        &self.vertices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::identities::Zero;

    #[test]
    fn test_add_vertex_and_indices() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        let idx1 = builder.vertices.len();
        builder.add_vertex(42);
        let idx2 = builder.vertices.len();
        builder.add_vertex(99);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        let g = builder.build();
        assert_eq!(g.vertices, vec![42, 99]);
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
        assert_eq!(g.vertices, vec![0, 1, 2]);
        assert_eq!(g.adjacency.shape(), (3, 3));

        let ai = g.get_vertex_index(&a).unwrap();
        let bi = g.get_vertex_index(&b).unwrap();
        let ci = g.get_vertex_index(&c).unwrap();

        // adjacency is symmetric: store at both (to, from) and (from, to)
        assert_eq!(g.adjacency[(bi, ai)], 10);
        assert_eq!(g.adjacency[(ai, bi)], 10);
        assert_eq!(g.adjacency[(ci, bi)], 20);
        assert_eq!(g.adjacency[(bi, ci)], 20);
        assert_eq!(g.adjacency[(ai, ci)], 30);
        assert_eq!(g.adjacency[(ci, ai)], 30);

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
        builder.add_vertex(1);
        let u = 0;
        let v = 1;
        builder.add_edge(u, v, 1);
        builder.add_edge(u, v, 2);
        let g = builder.build();
        let ui = g.get_vertex_index(&u).unwrap();
        let vi = g.get_vertex_index(&v).unwrap();
        // edges matrix stores at (to, from)
        assert_eq!(g.adjacency[(vi, ui)], 2);
    }

    #[test]
    fn test_empty_graph() {
        let builder = GraphBuilder::<i32, i32>::new();
        let g = builder.build();
        assert_eq!(g.vertices.len(), 0);
        assert_eq!(g.adjacency.shape(), (0, 0));
    }

    #[test]
    fn test_get_edge_value_and_vertex_value() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        builder.add_vertex(7);
        builder.add_vertex(8);
        builder.add_edge(7, 8, 15);
        let g = builder.build();
        assert_eq!(g.get_edge_value(&8, &7), 15);
        let idx7 = g.get_vertex_index(&7).unwrap();
        let idx8 = g.get_vertex_index(&8).unwrap();
        assert_eq!(*g.get_vertex_value(idx7), 7);
        assert_eq!(*g.get_vertex_value(idx8), 8);
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
        let idx = builder.vertices.len();
        builder.add_vertex(123);
        assert_eq!(idx, 0);
        let g = builder.build();
        assert_eq!(g.vertices, vec![123]);
        assert_eq!(g.adjacency.shape(), (1, 1));
        assert_eq!(g.adjacency[(0, 0)], 0);
    }

    // New tests

    #[test]
    fn test_degree_matrices_for_simple_cycle() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        // Vertices: 0,1,2 with edges 0->1, 1->2, 2->0 (a directed 3-cycle)
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_vertex(2);
        builder.add_edge(0, 1, 10);
        builder.add_edge(1, 2, 20);
        builder.add_edge(2, 0, 30);
        let g = builder.build();

        // Outgoing and incoming on diagonals are weighted by edge values
        assert_eq!(
            g.outgoing_degrees[(
                g.get_vertex_index(&0).unwrap(),
                g.get_vertex_index(&0).unwrap()
            )],
            10
        );
        assert_eq!(
            g.outgoing_degrees[(
                g.get_vertex_index(&1).unwrap(),
                g.get_vertex_index(&1).unwrap()
            )],
            20
        );
        assert_eq!(
            g.outgoing_degrees[(
                g.get_vertex_index(&2).unwrap(),
                g.get_vertex_index(&2).unwrap()
            )],
            30
        );

        assert_eq!(
            g.incoming_degrees[(
                g.get_vertex_index(&0).unwrap(),
                g.get_vertex_index(&0).unwrap()
            )],
            30
        );
        assert_eq!(
            g.incoming_degrees[(
                g.get_vertex_index(&1).unwrap(),
                g.get_vertex_index(&1).unwrap()
            )],
            10
        );
        assert_eq!(
            g.incoming_degrees[(
                g.get_vertex_index(&2).unwrap(),
                g.get_vertex_index(&2).unwrap()
            )],
            20
        );

        // Total degree = in + out (weighted)
        assert_eq!(
            g.degrees[(
                g.get_vertex_index(&0).unwrap(),
                g.get_vertex_index(&0).unwrap()
            )],
            40
        );
        assert_eq!(
            g.degrees[(
                g.get_vertex_index(&1).unwrap(),
                g.get_vertex_index(&1).unwrap()
            )],
            30
        );
        assert_eq!(
            g.degrees[(
                g.get_vertex_index(&2).unwrap(),
                g.get_vertex_index(&2).unwrap()
            )],
            50
        );
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

        let i0 = g.get_vertex_index(&0).unwrap();
        let i1 = g.get_vertex_index(&1).unwrap();

        // Outgoing: weighted sums
        assert_eq!(g.outgoing_degrees[(i0, i0)], 99 + 1 + 2);
        assert_eq!(g.outgoing_degrees[(i1, i1)], 3);

        // Incoming: weighted sums
        assert_eq!(g.incoming_degrees[(i0, i0)], 99 + 3);
        assert_eq!(g.incoming_degrees[(i1, i1)], 1 + 2);

        // Total degrees: weighted sums
        assert_eq!(g.degrees[(i0, i0)], (99 + 1 + 2) + (99 + 3));
        assert_eq!(g.degrees[(i1, i1)], 3 + (1 + 2));

        // Adjacency is symmetric; the last assignment between 0 and 1 was 1->0 with weight 3
        assert_eq!(g.adjacency[(i1, i0)], 3);
        assert_eq!(g.adjacency[(i0, i1)], 3);
        // Self-loop should be present
        assert_eq!(g.adjacency[(i0, i0)], 99);
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
        builder.add_vertex(0);
        builder.add_vertex(1);
        builder.add_edge(0, 1, 2.5);
        let g = builder.build();
        assert_eq!(g.adjacency.shape(), (2, 2));
        let i0 = g.get_vertex_index(&0).unwrap();
        let i1 = g.get_vertex_index(&1).unwrap();
        // edges matrix stores at (to, from)
        assert_eq!(g.adjacency[(i1, i0)], 2.5_f64);

        // Degree entries are weighted by edge values (f64)
        assert_eq!(g.outgoing_degrees[(i0, i0)], 2.5);
        assert_eq!(g.outgoing_degrees[(i1, i1)], 0.0);
        assert_eq!(g.incoming_degrees[(i0, i0)], 0.0);
        assert_eq!(g.incoming_degrees[(i1, i1)], 2.5);
        assert_eq!(g.degrees[(i0, i0)], 2.5);
        assert_eq!(g.degrees[(i1, i1)], 2.5);
    }

    #[test]
    fn test_generic_vertex_values() {
        let mut builder = GraphBuilder::<String, i32>::new();
        let alpha = "alpha".to_string();
        let beta = "beta".to_string();
        builder.add_vertex(alpha.clone());
        builder.add_vertex(beta.clone());
        builder.add_edge(alpha.clone(), beta.clone(), 7);
        let g = builder.build();
        assert_eq!(g.vertices.len(), 2);
        let alpha_idx = g.get_vertex_index(&alpha).unwrap();
        let beta_idx = g.get_vertex_index(&beta).unwrap();
        assert_eq!(g.get_vertex_value(alpha_idx), "alpha");
        assert_eq!(g.get_vertex_value(beta_idx), "beta");
        assert_eq!(g.get_edge_value_by_vertices(&beta, &alpha).unwrap(), 7);
        assert_eq!(g.get_edge_value_by_vertices(&alpha, &beta).unwrap(), 7);
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
            builder.add_edge(i as i32, (i + 1) as i32, 1);
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
        builder.add_vertex(1);
        let g = builder.build();
        assert_eq!(g.get_edge_value(&0, &1), i32::zero());
        assert_eq!(g.get_edge_value(&1, &0), i32::zero());
    }

    #[test]
    fn test_incoming_outgoing_totals_consistency() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        for n in 0..4 {
            builder.add_vertex(n);
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
        let i0 = g.get_vertex_index(&0).unwrap();
        let i2 = g.get_vertex_index(&2).unwrap();
        let i3 = g.get_vertex_index(&3).unwrap();
        assert_eq!(g.outgoing_degrees[(i0, i0)], 5 + 6); // 0->1 (5), 0->2 (6)
        assert_eq!(g.incoming_degrees[(i2, i2)], 6 + 7); // from 0 (6) and 1 (7)
        assert_eq!(g.incoming_degrees[(i0, i0)], 8); // from 3 (8)
        assert_eq!(g.outgoing_degrees[(i3, i3)], 8); // to 0 (8)
    }

    #[test]
    fn test_add_edge_by_values_and_lookup() {
        let mut builder = GraphBuilder::<&'static str, i32>::new();
        builder.add_vertex("A").add_vertex("B");
        builder.add_edge("A", "B", 11);
        builder.add_edge("A", "C", 7); // "C" auto-inserted
        let g = builder.build();
        let a_idx = g.get_vertex_index(&"A").unwrap();
        let b_idx = g.get_vertex_index(&"B").unwrap();
        let c_idx = g.get_vertex_index(&"C").unwrap();
        assert_eq!(g.adjacency[(b_idx, a_idx)], 11);
        assert_eq!(g.adjacency[(a_idx, b_idx)], 11);
        assert_eq!(g.adjacency[(c_idx, a_idx)], 7);
        assert_eq!(g.get_edge_value_by_vertices(&"A", &"B").unwrap(), 11);
        assert_eq!(g.get_edge_value_by_vertices(&"C", &"A").unwrap(), 7);
    }
}
