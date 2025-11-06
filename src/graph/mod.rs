use std::ops::Mul;

/// Represents a graph with vertex values of type `T` and edge values of type `V`.
///
/// # Fields
/// - `vertexes`: A vector containing the values of the graph's vertices.
/// - `edges`: A 2D array (adjacency matrix) representing the edge values between vertices.
pub struct Graph<T, V: Mul<Output = T> + Copy + num_traits::identities::Zero> {
    pub vertexes: Vec<T>,
    pub edges: ndarray::Array2<V>,
}

/// A builder for constructing a `Graph`.
///
/// # Fields
/// - `vertexes`: A vector storing the values of vertices to be added to the graph.
/// - `edges`: A vector of tuples representing edges as (from, to, value).
pub struct GraphBuilder<T, V: num_traits::identities::Zero + Mul<Output = T> + Copy> {
    vertexes: Vec<T>,
    edges: Vec<(usize, usize, V)>,
}

impl<T, V: num_traits::identities::Zero + Mul<Output = T> + Copy> GraphBuilder<T, V> {
    /// Creates a new, empty `GraphBuilder`.
    ///
    /// # Returns
    /// A new instance of `GraphBuilder` with no vertices or edges.
    pub fn new() -> Self {
        Self {
            vertexes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Adds a vertex to the graph with the given value.
    ///
    /// # Arguments
    /// * `value` - The value to assign to the new vertex.
    ///
    /// # Returns
    /// The index of the newly added vertex.
    pub fn add_vertex(&mut self, value: T) -> usize {
        self.vertexes.push(value);
        self.vertexes.len() - 1
    }

    /// Adds an edge to the graph from one vertex to another with a specified value.
    ///
    /// # Arguments
    /// * `from` - The index of the source vertex.
    /// * `to` - The index of the destination vertex.
    /// * `value` - The value to assign to the edge.
    pub fn add_edge(&mut self, from: usize, to: usize, value: V) {
        self.edges.push((from, to, value));
    }

    /// Builds and returns a `Graph` from the current state of the builder.
    ///
    /// # Returns
    /// A `Graph` containing all added vertices and edges.
    pub fn build(self) -> Graph<T, V> {
        let n = self.vertexes.len();
        let mut edge_matrix = ndarray::Array2::<V>::zeros((n, n));

        for (from, to, value) in self.edges.iter() {
            edge_matrix[[*from, *to]] = *value;
        }

        Graph {
            vertexes: self.vertexes,
            edges: edge_matrix,
        }
    }
}

impl<T, V: num_traits::identities::Zero + Mul<Output = T> + Copy> Default for GraphBuilder<T, V> {
    /// Returns a default, empty `GraphBuilder`.
    ///
    /// # Returns
    /// An empty `GraphBuilder`.
    fn default() -> Self {
        Self::new()
    }
}

impl<T, V: num_traits::identities::Zero + Mul<Output = T> + Copy> Graph<T, V> {
    /// Retrieves the value of the edge from one vertex to another.
    ///
    /// # Arguments
    /// * `from` - The index of the source vertex.
    /// * `to` - The index of the destination vertex.
    ///
    /// # Returns
    /// The value of the edge from `from` to `to`.
    pub fn get_edge_value(&self, from: usize, to: usize) -> V {
        self.edges[[from, to]]
    }

    /// Returns the number of vertices in the graph.
    ///
    /// # Returns
    /// The number of vertices (rank) in the graph.
    pub fn get_rank(&self) -> usize {
        self.vertexes.len()
    }

    /// Retrieves a reference to the value of a vertex by its index.
    ///
    /// # Arguments
    /// * `index` - The index of the vertex.
    ///
    /// # Returns
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
        let idx1 = builder.add_vertex(42);
        let idx2 = builder.add_vertex(99);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        let g = builder.build();
        assert_eq!(g.vertexes, vec![42, 99]);
        assert_eq!(g.edges.shape(), &[2, 2]);
        assert_eq!(g.edges[[0, 0]], 0);
        assert_eq!(g.edges[[0, 1]], 0);
        assert_eq!(g.edges[[1, 0]], 0);
        assert_eq!(g.edges[[1, 1]], 0);
    }

    #[test]
    fn test_add_edges_and_matrix_values() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        let a = builder.add_vertex(1);
        let b = builder.add_vertex(2);
        let c = builder.add_vertex(3);
        builder.add_edge(a, b, 10);
        builder.add_edge(b, c, 20);
        builder.add_edge(c, a, 30);
        let g = builder.build();
        assert_eq!(g.vertexes, vec![1, 2, 3]);
        assert_eq!(g.edges.shape(), &[3, 3]);
        assert_eq!(g.edges[[a, b]], 10);
        assert_eq!(g.edges[[b, c]], 20);
        assert_eq!(g.edges[[c, a]], 30);
        // Check unset edges are zero
        for i in 0..3 {
            for j in 0..3 {
                let expected = if (i == a && j == b) || (i == b && j == c) || (i == c && j == a) {
                    None
                } else {
                    Some(i32::zero())
                };
                if let Some(z) = expected {
                    assert_eq!(g.edges[[i, j]], z, "Expected zero at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn test_multiple_edges_last_assignment() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        let u = builder.add_vertex(0);
        let v = builder.add_vertex(0);
        builder.add_edge(u, v, 1);
        builder.add_edge(u, v, 2);
        let g = builder.build();
        assert_eq!(g.edges[[u, v]], 2);
    }

    #[test]
    fn test_empty_graph() {
        let builder = GraphBuilder::<i32, i32>::new();
        let g = builder.build();
        assert_eq!(g.vertexes.len(), 0);
        assert_eq!(g.edges.shape(), &[0, 0]);
    }

    #[test]
    fn test_get_edge_value_and_vertex_value() {
        let mut builder = GraphBuilder::<i32, i32>::new();
        let a = builder.add_vertex(7);
        let b = builder.add_vertex(8);
        builder.add_edge(a, b, 15);
        let g = builder.build();
        assert_eq!(g.get_edge_value(a, b), 15);
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
        let idx = builder.add_vertex(123);
        assert_eq!(idx, 0);
        let g = builder.build();
        assert_eq!(g.vertexes, vec![123]);
        assert_eq!(g.edges.shape(), &[1, 1]);
        assert_eq!(g.edges[[0, 0]], 0);
    }
}
