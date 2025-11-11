use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Sub, SubAssign};

use num_traits::FromPrimitive;

use crate::graph::Graph;

impl<
    T: Eq + Clone + Hash,
    V: Add
        + Copy
        + Debug
        + PartialOrd
        + SubAssign
        + num_traits::Zero
        + Sub<Output = V>
        + faer::traits::RealField
        + FromPrimitive
        + Ord
        + 'static,
> Graph<T, V>
{
    /// Computes the Hoffman-Delsarte bound for the graph.
    ///
    /// This function calculates an upper bound on the independence number of the graph
    /// using the eigenvalues of the normalized Laplacian matrix and the degrees of the graph.
    ///
    /// # Inputs
    ///
    /// * `&self` - A reference to the graph instance.
    ///
    /// # Outputs
    ///
    /// Returns a value of type `V` representing the Hoffman-Delsarte bound for the graph.
    pub fn hoffman_delsarte_bound(&self) -> V {
        let eigenvalues = self
            .laplacian_matrix()
            .self_adjoint_eigen(faer::Side::Lower)
            .unwrap();
        let lambda_max = eigenvalues.S()[self.get_rank() - 1];

        let max_degree = self
            .adjacency_undirectional
            .row_iter()
            .map(|row| row.iter().fold(V::zero(), |acc, &val| acc + val))
            .max()
            .unwrap_or(V::zero());

        let min_degree = self
            .adjacency_undirectional
            .row_iter()
            .map(|row| row.iter().fold(V::zero(), |acc, &val| acc + val))
            .min()
            .unwrap_or(V::zero());

        let n = V::from_usize(self.get_rank()).unwrap();
        let bound = n * (V::one() - (V::one() / lambda_max)) * max_degree / min_degree;

        bound
    }
}
