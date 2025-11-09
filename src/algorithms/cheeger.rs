use super::Graph;
pub use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Sub, SubAssign};

impl<
    T: Eq + Clone + Hash,
    V: Add
        + Copy
        + Debug
        + PartialOrd
        + SubAssign
        + num_traits::Zero
        + Sub<Output = V>
        + nalgebra::ComplexField<RealField = V>
        + 'static,
> Graph<T, V>
{
    /// Computes an estimated range for the Cheeger constant of the graph.
    ///
    /// This function calculates lower and upper bounds for the Cheeger constant
    /// using the second smallest eigenvalue (Fiedler value) of the graph's Laplacian.
    ///
    /// # Inputs
    ///
    /// * `&self` - A reference to the graph instance.
    ///
    /// # Outputs
    ///
    /// Returns a tuple `(V, V)` where:
    /// - The first element is the lower bound approximation of the Cheeger constant.
    /// - The second element is the upper bound approximation of the Cheeger constant.
    pub fn cheeger_range(&self) -> (V, V) {
        let lambda_2 = self.fiedler().0;

        // Cheeger low constant approximation
        let cheeger_const_lower = lambda_2 / V::from_f64(2.0).unwrap();

        // Cheeger upper constant approximation
        let cheeger_const_upper = (lambda_2 * V::from_f64(2.0).unwrap()).sqrt();

        (cheeger_const_lower, cheeger_const_upper)
    }
}
