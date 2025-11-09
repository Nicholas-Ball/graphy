use std::hash::Hash;

use crate::graph::Graph;

impl<T: Hash + Eq, V: Copy> Graph<T, V> {
    /// Returns the value of the edge from one vertex to another.
    ///
    /// # Arguments
    ///
    /// * `from` - Reference to the value of the source vertex.
    /// * `to` - Reference to the value of the destination vertex.
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

        self.adjacency_undirectional[idx]
    }

    /// Returns the value of the edge between two vertices, if both vertices exist.
    ///
    /// # Arguments
    ///
    /// * `from` - Reference to the value of the source vertex.
    /// * `to` - Reference to the value of the destination vertex.
    ///
    /// # Returns
    ///
    /// Some(weight) if both vertices exist, otherwise None.
    pub fn get_edge_value_by_vertices(&self, from: &T, to: &T) -> Option<V> {
        let from_idx = self.index_map.get(from)?;
        let to_idx = self.index_map.get(to)?;
        Some(self.adjacency_undirectional[(*to_idx, *from_idx)])
    }
}
