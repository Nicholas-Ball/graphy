pub mod cheeger;
pub mod fiedler;
pub mod floyd_warshall;
pub mod hoffman_delsarte;

use std::{fmt::Debug, hash::Hash};

use crate::graph::Graph;
use nalgebra::DMatrix;

impl<T: Eq + Clone + Hash, V: Debug + PartialEq + Copy + num_traits::Zero + 'static> Graph<T, V> {
    /// Returns a reference to the degree matrix of the graph.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `&DMatrix<V>` - Reference to the degree matrix as a 2D array.
    pub fn degree_matrix(&self) -> &DMatrix<V> {
        &self.degrees
    }

    /// Returns a reference to the adjacency matrix of the graph.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `&DMatrix<V>` - Reference to the adjacency matrix.
    pub fn adjacency_matrix(&self) -> &DMatrix<V> {
        &self.adjacency_undirectional
    }

    pub fn adjacency_matrix_directed(&self) -> &DMatrix<V> {
        &self.adjacency_directional
    }

    /// Returns a reference to the out-degree matrix of the graph.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `&DMatrix<V>` - Reference to the out-degree matrix as a 2D array.
    pub fn degree_matrix_out(&self) -> &DMatrix<V> {
        &self.outgoing_degrees
    }

    /// Returns a reference to the in-degree matrix of the graph.
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph.
    ///
    /// # Returns
    ///
    /// * `&DMatrix<V>` - Reference to the in-degree matrix as a 2D array.
    pub fn degree_matrix_in(&self) -> &DMatrix<V> {
        &self.incoming_degrees
    }
}
