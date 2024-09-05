use std::sync::Arc;

use ark_ff::PrimeField;
use ark_r1cs_std::{
    alloc::AllocVar,
    fields::{fp::FpVar, FieldVar},
    R1CSVar,
};
use ark_relations::r1cs::SynthesisError;

use super::params::GriffinParams;

#[derive(Clone, Debug)]
pub struct GriffinCircuit<F: PrimeField> {
    pub(crate) params: Arc<GriffinParams<F>>,
}

impl<F: PrimeField> GriffinCircuit<F> {
    pub fn new(params: &Arc<GriffinParams<F>>) -> Self {
        GriffinCircuit {
            params: Arc::clone(params),
        }
    }

    fn non_linear(&self, state: &[FpVar<F>]) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let cs = state.cs();
        let mut result = state.to_owned();
        // x0
        result[0] = FpVar::new_variable_with_inferred_mode(cs, || {
            Ok({
                {
                    let v = result[0].value().unwrap_or_default();
                    let mut res = F::one();
                    for &i in &self.params.d_inv {
                        res.square_in_place();
                        if i {
                            res *= v;
                        }
                    }
                    res
                }
            })
        })?;

        let mut sq = result[0].square()?;
        if self.params.d == 5 {
            sq = sq.square()?;
        }
        result[0].mul_equals(&sq, &state[0])?;

        // x1
        let mut sq = result[1].square()?;
        if self.params.d == 5 {
            sq = sq.square()?;
        }
        result[1] *= sq;

        let mut y01_i = result[1].clone();

        // rest of the state
        for i in 2..result.len() {
            y01_i += &result[0];
            let l = if i == 2 {
                y01_i.clone()
            } else {
                &y01_i + &state[i - 1]
            };
            let ab = &self.params.alpha_beta[i - 2];
            result[i] *= l.square()? + l * ab[0] + ab[1];
        }

        Ok(result)
    }
}

pub trait PermCircuit<F: PrimeField> {
    fn permute(&self, state: &[FpVar<F>]) -> Result<Vec<FpVar<F>>, SynthesisError>;
    fn hash(&self, message: &[FpVar<F>]) -> Result<FpVar<F>, SynthesisError>;
}

impl<F: PrimeField> PermCircuit<F> for GriffinCircuit<F> {
    fn permute(&self, state: &[FpVar<F>]) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let mut current_state = state.to_owned();
        current_state = self
            .params
            .mat
            .iter()
            .map(|row| current_state.iter().zip(row).map(|(a, b)| a * *b).sum())
            .collect();

        for r in 0..self.params.rounds {
            current_state = self.non_linear(&current_state)?;
            current_state = self
                .params
                .mat
                .iter()
                .map(|row| current_state.iter().zip(row).map(|(a, b)| a * *b).sum())
                .collect();
            if r < self.params.rounds - 1 {
                current_state = current_state
                    .iter()
                    .zip(&self.params.round_constants[r])
                    .map(|(c, rc)| c + *rc)
                    .collect();
            }
        }
        Ok(current_state)
    }

    fn hash(&self, message: &[FpVar<F>]) -> Result<FpVar<F>, SynthesisError> {
        let mut state = vec![FpVar::zero(); self.params.t];
        for chunk in message.chunks(self.params.rate) {
            for i in 0..chunk.len() {
                state[i] += &chunk[i];
            }
            state = self.permute(&state)?;
        }
        Ok(state[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_r1cs_std::{alloc::AllocVar, R1CSVar};
    use ark_relations::r1cs::ConstraintSystem;
    use rand::thread_rng;

    use crate::griffin::{
        griffin::{Griffin, Permutation},
        params::GriffinParams,
    };

    use super::{GriffinCircuit, PermCircuit};

    #[test]
    fn test() {
        let rng = &mut thread_rng();
        let params = Arc::new(GriffinParams::new(24, 5, 9));
        let griffin = Griffin::new(&params);
        let t = griffin.params.t;
        let x: Vec<Fr> = (0..t).map(|_| Fr::rand(rng)).collect();

        let y = griffin.hash(&x);

        let cs = ConstraintSystem::new_ref();
        let x_var = Vec::new_witness(cs.clone(), || Ok(x.clone())).unwrap();
        let y_var = GriffinCircuit::new(&params).hash(&x_var).unwrap();
        assert_eq!(y, y_var.value().unwrap());
        println!("{}", cs.num_constraints());
        assert!(cs.is_satisfied().unwrap());
    }
}
