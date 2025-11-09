use crate::graph::Graph;

impl<T, V> Graph<T, V> {
    /// Returns a reference to the value of a vertex by its index.
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
    ///
    /// # Arguments
    ///
    /// * `value` - Reference to the value of the vertex.
    ///
    /// # Returns
    ///
    /// Some(index) if the vertex exists, otherwise None.
    pub fn get_vertex_index(&self, value: &T) -> Option<usize>
    where
        T: std::hash::Hash + Eq,
    {
        self.index_map.get(value).copied()
    }

    /// Returns a reference to the vector of all vertex values in the graph.
    ///
    /// # Returns
    ///
    /// Reference to the vector of vertex values.
    pub fn get_vertices(&self) -> &Vec<T> {
        &self.vertices
    }
}
