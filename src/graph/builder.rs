use faer::traits::ComplexField;

use crate::graph::Graph;
use std::collections::HashMap;
use std::hash::Hash;

/// A builder for constructing a `Graph`.
///
/// # Fields
///
/// * `vertices` - A vector storing the values of vertices to be added to the graph.
/// * `edges` - A vector of tuples representing edges as (from_index, to_index, value).
#[derive(Default)]
pub struct GraphBuilder<T, V> {
    pub(crate) vertices: Vec<T>,
    pub(crate) edges: Vec<(usize, usize, V)>,
    pub(crate) index_map: HashMap<T, usize>,
}

impl<T: Eq + Hash + Clone, V> GraphBuilder<T, V> {
    /// Creates a new, empty `GraphBuilder`.
    ///
    /// # Returns
    ///
    /// * `GraphBuilder<T, V>` - A new instance of `GraphBuilder` with no vertices or edges.
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
    /// * `&mut Self` - The builder for chaining.
    ///
    /// If the vertex value already exists, it will not create a duplicate; the existing
    /// index is retained (idempotent insertion).
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
    /// # Arguments
    ///
    /// * `from_val` - The value of the source vertex.
    /// * `to_val` - The value of the destination vertex.
    /// * `value` - The weight of the edge.
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The builder for chaining.
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
    ///
    /// # Arguments
    ///
    /// * `from` - Reference to the value of the source vertex.
    /// * `to` - Reference to the value of the destination vertex.
    /// * `value` - The weight of the edge.
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The builder for chaining.
    ///
    /// # Panics
    ///
    /// Panics if either vertex does not exist in the builder.
    pub fn add_edge_ref(&mut self, from: &T, to: &T, value: V) -> &mut Self {
        if let (Some(&from_idx), Some(&to_idx)) = (self.index_map.get(from), self.index_map.get(to))
        {
            self.edges.push((from_idx, to_idx, value));
        } else {
            panic!("Attempted to add_edge_ref with unknown vertex value(s).");
        }
        self
    }

    /// Gets the index of a vertex, inserting it if it does not exist.
    ///
    /// # Arguments
    ///
    /// * `v` - The value of the vertex.
    ///
    /// # Returns
    ///
    /// * `usize` - The index of the vertex.
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

    /// Builds and returns a `Graph` from the builder.
    ///
    /// # Returns
    ///
    /// * `Graph<T, V>` - The constructed graph with all vertices and edges added.
    pub fn build(self) -> Graph<T, V>
    where
        V: num_traits::Zero
            + num_traits::One
            + Copy
            + std::ops::AddAssign
            + faer::traits::RealField,
    {
        let n = self.vertices.len();
        let mut adjacency_matrix = faer::Mat::<V>::zeros(n, n);
        let mut directed_adjacency_matrix = faer::Mat::<V>::zeros(n, n);
        let mut degree_matrix = faer::Mat::<V>::zeros(n, n);
        let mut incoming_degree_matrix = faer::Mat::<V>::zeros(n, n);
        let mut outgoing_degree_matrix = faer::Mat::<V>::zeros(n, n);

        let mut edges_evaluated = Vec::new();

        for (from, to, value) in self.edges.iter() {
            if *value == V::zero() {
                continue;
            }

            if !edges_evaluated.contains(&(*from, *to)) {
                // Directed adjacency
                directed_adjacency_matrix[(*from, *to)] = *value;

                // Update weighted degrees
                outgoing_degree_matrix[(*from, *from)] += V::one();
                incoming_degree_matrix[(*to, *to)] += V::one();
            }

            // validate that an edge already exists before adding to degree matrix
            if edges_evaluated.contains(&(*from, *to)) || edges_evaluated.contains(&(*to, *from)) {
                continue;
            }

            // Undirected adjacency (symmetric)
            adjacency_matrix[(*to, *from)] = *value;
            adjacency_matrix[(*from, *to)] = *value;

            degree_matrix[(*from, *from)] += V::one();

            if *from != *to {
                degree_matrix[(*to, *to)] += V::one();
            }

            edges_evaluated.push((*from, *to));
        }

        Graph {
            vertices: self.vertices,
            adjacency_undirectional: adjacency_matrix,
            adjacency_directional: directed_adjacency_matrix,
            degrees: degree_matrix,
            incoming_degrees: incoming_degree_matrix,
            outgoing_degrees: outgoing_degree_matrix,
            index_map: self.index_map,
        }
    }
}
