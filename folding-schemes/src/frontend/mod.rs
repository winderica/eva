// pub mod circom;
use std::{
    cell::UnsafeCell,
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    marker::PhantomData,
    mem::size_of,
    rc::Rc,
    slice::from_raw_parts,
};

use crate::Error;
use ark_ff::{batch_inversion, PrimeField};
use ark_r1cs_std::{
    alloc::AllocVar,
    fields::{
        fp::{AllocatedFp, FpVar},
        FieldVar,
    },
    R1CSVar,
};
use ark_relations::{
    lc,
    r1cs::{ConstraintSystemRef, LcIndex, LinearCombination, SynthesisError, Variable},
};
use ark_std::{end_timer, fmt::Debug, start_timer};
use rayon::prelude::*;

pub type IntMap<K, V> = HashMap<K, V, BuildNoHashHasher<K>>;

pub type BuildNoHashHasher<T> = BuildHasherDefault<NoHashHasher<T>>;

pub struct NoHashHasher<T>(pub u64, PhantomData<T>);

impl<T> Default for NoHashHasher<T> {
    fn default() -> Self {
        NoHashHasher(0, PhantomData)
    }
}

impl<T> Clone for NoHashHasher<T> {
    fn clone(&self) -> Self {
        NoHashHasher(self.0, self.1)
    }
}

impl<T> Copy for NoHashHasher<T> {}

impl<F: PrimeField> Hasher for NoHashHasher<F> {
    #[inline]
    fn write(&mut self, n: &[u8]) {
        let n = unsafe {
            from_raw_parts(
                n as *const _ as *const u64,
                n.len() * size_of::<u8>() / size_of::<u64>(),
            )
        };
        for i in n {
            self.0 ^= i
                .wrapping_add(0x9e3779b97f4a7c15u64)
                .wrapping_add(self.0.wrapping_shl(6))
                .wrapping_add(self.0.wrapping_shr(2));
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {}

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct LookupArgumentRef<F: PrimeField>(Rc<UnsafeCell<LookupArgument<F>>>);

impl<F: PrimeField> LookupArgumentRef<F> {
    pub fn len(&self) -> usize {
        self.borrow().table.len()
    }

    pub fn set_table(&self, table: Vec<F>) {
        self.borrow_mut().set_table(table);
    }

    pub fn build_histo(&self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.borrow_mut().build_histo(cs)
    }

    pub fn generate_lookup_constraints(
        &self,
        cs: ConstraintSystemRef<F>,
        c: F,
    ) -> Result<(), SynthesisError> {
        self.borrow().generate_lookup_constraints(cs, c)
    }

    pub fn into_inner(self) -> Option<LookupArgument<F>> {
        Rc::try_unwrap(self.0).ok().map(|s| s.into_inner())
    }

    pub fn borrow(&self) -> &LookupArgument<F> {
        unsafe { &*self.0.get() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn borrow_mut(&self) -> &mut LookupArgument<F> {
        unsafe { &mut *self.0.get() }
    }
}

#[derive(Debug)]
pub struct LookupArgument<F: PrimeField> {
    pub table: Vec<F>,
    histo: Vec<(F, Variable)>,
}

impl<F: PrimeField> LookupArgument<F> {
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            histo: Vec::new(),
        }
    }

    pub fn new_ref() -> LookupArgumentRef<F> {
        LookupArgumentRef(Rc::new(UnsafeCell::new(Self::new())))
    }

    pub fn set_table(&mut self, table: Vec<F>) {
        self.table = table;
    }

    fn compute_lhs(
        &self,
        cs: ConstraintSystemRef<F>,
        c: &AllocatedFp<F>,
    ) -> Result<LinearCombination<F>, SynthesisError> {
        let mut new_lc = lc!();
        let c_value = c.value().unwrap_or_default();

        let timer = start_timer!(|| "Batch inversion");
        let mut entries = self
            .table
            .par_iter()
            .map(|i| c_value - i)
            .collect::<Vec<_>>();
        batch_inversion(&mut entries);
        end_timer!(timer);

        let timer = start_timer!(|| "Get LHS LC");
        for ((i, v), (e_val, e)) in self.table.iter().zip(entries.into_iter()).zip(&self.histo) {
            let inverse = cs.new_witness_variable(|| Ok(v * e_val))?;
            new_lc = new_lc + inverse;
            cs.enforce_constraint(
                lc!() + c.variable - (*i, Variable::One),
                inverse.into(),
                (*e).into(),
            )?;
        }
        end_timer!(timer);

        Ok(new_lc)
    }

    fn compute_rhs(
        &self,
        cs: ConstraintSystemRef<F>,
        c: &AllocatedFp<F>,
    ) -> Result<LinearCombination<F>, SynthesisError> {
        let mut cs = cs.borrow_mut().unwrap();

        let num_query_variables = cs.num_committed_variables - self.table.len();

        let c_value = c.value().unwrap_or_default();

        let timer = start_timer!(|| "Batch inversion");
        let mut queries_inv = cs.committed_assignment[0..num_query_variables]
            .par_iter()
            .map(|i| c_value - i)
            .collect::<Vec<_>>();
        batch_inversion(&mut queries_inv);
        end_timer!(timer);

        let timer = start_timer!(|| "Get RHS LC");
        if !cs.is_in_setup_mode() {
            cs.witness_assignment.extend_from_slice(&queries_inv);
        }
        let w_index = cs.num_witness_variables + cs.w_offset;
        let lc_index = cs.num_linear_combinations + cs.lc_offset;
        let new_lc = LinearCombination(
            (0..num_query_variables)
                .into_par_iter()
                .map(|i| (F::one(), Variable::Witness(w_index + i)))
                .collect(),
        );
        if cs.should_construct_matrices() {
            cs.a_constraints.extend_from_slice(
                &(0..num_query_variables)
                    .into_par_iter()
                    .map(|i| LcIndex(lc_index + i * 3))
                    .collect::<Vec<_>>(),
            );
            cs.b_constraints.extend_from_slice(
                &(0..num_query_variables)
                    .into_par_iter()
                    .map(|i| LcIndex(lc_index + i * 3 + 1))
                    .collect::<Vec<_>>(),
            );
            cs.c_constraints.extend_from_slice(
                &(0..num_query_variables)
                    .into_par_iter()
                    .map(|i| LcIndex(lc_index + i * 3 + 2))
                    .collect::<Vec<_>>(),
            );

            for j in 0..num_query_variables {
                let inverse = Variable::Witness(w_index + j);
                cs.lc_map.insert(
                    LcIndex(lc_index + j * 3),
                    LinearCombination(vec![
                        (-F::one(), Variable::Committed(cs.q_offset + j)),
                        (F::one(), c.variable),
                    ]),
                );
                cs.lc_map
                    .insert(LcIndex(lc_index + j * 3 + 1), inverse.into());
                cs.lc_map
                    .insert(LcIndex(lc_index + j * 3 + 2), Variable::One.into());
            }
            cs.num_linear_combinations += num_query_variables * 3;
        }

        cs.num_witness_variables += num_query_variables;
        cs.num_constraints += num_query_variables;
        end_timer!(timer);

        Ok(new_lc)
    }

    pub fn build_histo(&mut self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let mut histo = self
            .table
            .iter()
            .map(|&v| (v, 0u64))
            .collect::<IntMap<_, _>>();
        cs.borrow_mut()
            .unwrap()
            .committed_assignment
            .iter()
            .for_each(|&i| {
                histo.get_mut(&i).map(|c| *c += 1).unwrap();
            });
        let e = self
            .table
            .iter()
            .map(|i| {
                Ok((
                    F::from(histo[i]),
                    cs.new_committed_variable(|| Ok(F::from(histo[i])))?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.histo = e;
        Ok(())
    }

    pub fn generate_lookup_constraints(
        &self,
        cs: ConstraintSystemRef<F>,
        c: F,
    ) -> Result<(), SynthesisError> {
        if self.table.is_empty() {
            return Ok(());
        }

        let c = AllocatedFp::new_input(cs.clone(), || Ok(c))?;

        let timer = start_timer!(|| "Compute LHS of LogUp identity");
        let l = self.compute_lhs(cs.clone(), &c)?;
        end_timer!(timer);
        let timer = start_timer!(|| "Compute RHS of LogUp identity");
        let r = self.compute_rhs(cs.clone(), &c)?;
        end_timer!(timer);

        cs.enforce_constraint(l, lc!() + Variable::One, r)?;

        Ok(())
    }
}

/// FCircuit defines the trait of the circuit of the F function, which is the one being folded (ie.
/// inside the agmented F' function).
/// The parameter z_i denotes the current state, and z_{i+1} denotes the next state after applying
/// the step.
pub trait FCircuit<F: PrimeField>: Clone + Debug {
    type Params: Debug;
    type ExternalInputs;

    /// returns a new FCircuit instance
    fn new(params: Self::Params) -> Self;

    /// returns the number of elements in the state of the FCircuit, which corresponds to the
    /// FCircuit inputs.
    fn state_len(&self) -> usize;

    /// computes the next state values in place, assigning z_{i+1} into z_i, and computing the new
    /// z_{i+1}
    fn step_native(
        // this method uses self, so that each FCircuit implementation (and different frontends)
        // can hold a state if needed to store data to compute the next state.
        &self,
        i: usize,
        z_i: Vec<F>,
        external_inputs: &Self::ExternalInputs,
    ) -> Result<Vec<F>, Error>;

    /// generates the constraints for the step of F for the given z_i
    fn generate_step_constraints(
        // this method uses self, so that each FCircuit implementation (and different frontends)
        // can hold a state if needed to store data to generate the constraints.
        &self,
        cs: ConstraintSystemRef<F>,
        la: LookupArgumentRef<F>,
        i: usize,
        z_i: Vec<FpVar<F>>,
        external_inputs: &Self::ExternalInputs,
    ) -> Result<Vec<FpVar<F>>, SynthesisError>;
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget};
    use ark_relations::r1cs::{
        ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError,
    };
    use core::marker::PhantomData;

    /// CubicFCircuit is a struct that implements the FCircuit trait, for the R1CS example circuit
    /// from https://www.vitalik.ca/general/2016/12/10/qap.html, which checks `x^3 + x + 5 = y`. It
    /// has 2 public inputs which are used as the state. `z_i` is used as `x`, and `z_{i+1}` is
    /// used as `y`, and at the next step, `z_{i+1}` will be assigned to `z_i`, and a new `z+{i+1}`
    /// will be computted.
    #[derive(Clone, Copy, Debug)]
    pub struct CubicFCircuit<F: PrimeField> {
        _f: PhantomData<F>,
    }
    impl<F: PrimeField> FCircuit<F> for CubicFCircuit<F> {
        type Params = ();
        type ExternalInputs = ();
        fn new(_params: Self::Params) -> Self {
            Self { _f: PhantomData }
        }
        fn state_len(&self) -> usize {
            1
        }
        fn step_native(&self, _i: usize, z_i: Vec<F>, _: &Self::ExternalInputs) -> Result<Vec<F>, Error> {
            Ok(vec![z_i[0] * z_i[0] * z_i[0] + z_i[0] + F::from(5_u32)])
        }
        fn generate_step_constraints(
            &self,
            cs: ConstraintSystemRef<F>,
            la: LookupArgumentRef<F>,
            _i: usize,
            z_i: Vec<FpVar<F>>,
            _: &Self::ExternalInputs,
        ) -> Result<Vec<FpVar<F>>, SynthesisError> {
            la.set_table((0u32..256).map(F::from).collect());
            let five = FpVar::<F>::new_committed(cs.clone(), || Ok(F::from(5u32)))?;
            let z_i = z_i[0].clone();

            Ok(vec![&z_i * &z_i * &z_i + &z_i + &five])
        }
    }

    /// CustomFCircuit is a circuit that has the number of constraints specified in the
    /// `n_constraints` parameter. Note that the generated circuit will have very sparse matrices.
    #[derive(Clone, Copy, Debug)]
    pub struct CustomFCircuit<F: PrimeField> {
        _f: PhantomData<F>,
        pub n_constraints: usize,
    }
    impl<F: PrimeField> FCircuit<F> for CustomFCircuit<F> {
        type Params = usize;
        type ExternalInputs = ();

        fn new(params: Self::Params) -> Self {
            Self {
                _f: PhantomData,
                n_constraints: params,
            }
        }
        fn state_len(&self) -> usize {
            1
        }
        fn step_native(&self, _i: usize, z_i: Vec<F>, _: &Self::ExternalInputs) -> Result<Vec<F>, Error> {
            let mut z_i1 = F::one();
            for _ in 0..self.n_constraints - 1 {
                z_i1 *= z_i[0];
            }
            Ok(vec![z_i1])
        }
        fn generate_step_constraints(
            &self,
            cs: ConstraintSystemRef<F>,
            _la: LookupArgumentRef<F>,
            _i: usize,
            z_i: Vec<FpVar<F>>,
            _: &Self::ExternalInputs
        ) -> Result<Vec<FpVar<F>>, SynthesisError> {
            let mut z_i1 = FpVar::<F>::new_witness(cs.clone(), || Ok(F::one()))?;
            for _ in 0..self.n_constraints - 1 {
                z_i1 *= z_i[0].clone();
            }

            Ok(vec![z_i1])
        }
    }

    /// WrapperCircuit is a circuit that wraps any circuit that implements the FCircuit trait. This
    /// is used to test the `FCircuit.generate_step_constraints` method. This is a similar wrapping
    /// than the one done in the `AugmentedFCircuit`, but without adding all the extra constraints
    /// of the AugmentedF circuit logic, in order to run lighter tests when we're not interested in
    /// the the AugmentedF logic but in the wrapping of the circuits.
    pub struct WrapperCircuit<F: PrimeField, FC: FCircuit<F, ExternalInputs = ()>> {
        pub FC: FC, // F circuit
        pub z_i: Option<Vec<F>>,
        pub z_i1: Option<Vec<F>>,
    }
    impl<F, FC> ConstraintSynthesizer<F> for WrapperCircuit<F, FC>
    where
        F: PrimeField,
        FC: FCircuit<F, ExternalInputs = ()>,
    {
        fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
            let z_i = Vec::<FpVar<F>>::new_witness(cs.clone(), || {
                Ok(self.z_i.unwrap_or(vec![F::zero()]))
            })?;
            let z_i1 = Vec::<FpVar<F>>::new_input(cs.clone(), || {
                Ok(self.z_i1.unwrap_or(vec![F::zero()]))
            })?;
            let computed_z_i1 = self.FC.generate_step_constraints(
                cs.clone(),
                LookupArgument::new_ref(),
                0,
                z_i.clone(),
                &()
            )?;

            computed_z_i1.enforce_equal(&z_i1)?;
            Ok(())
        }
    }

    #[test]
    fn test_testfcircuit() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let F_circuit = CubicFCircuit::<Fr>::new(());

        let wrapper_circuit = WrapperCircuit::<Fr, CubicFCircuit<Fr>> {
            FC: F_circuit,
            z_i: Some(vec![Fr::from(3_u32)]),
            z_i1: Some(vec![Fr::from(35_u32)]),
        };
        wrapper_circuit.generate_constraints(cs.clone()).unwrap();
        assert_eq!(cs.num_constraints(), 3);
    }

    #[test]
    fn test_customtestfcircuit() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let n_constraints = 1000;
        let custom_circuit = CustomFCircuit::<Fr>::new(n_constraints);
        let z_i = vec![Fr::from(5_u32)];
        let wrapper_circuit = WrapperCircuit::<Fr, CustomFCircuit<Fr>> {
            FC: custom_circuit,
            z_i: Some(z_i.clone()),
            z_i1: Some(custom_circuit.step_native(0, z_i, &()).unwrap()),
        };
        wrapper_circuit.generate_constraints(cs.clone()).unwrap();
        assert_eq!(cs.num_constraints(), n_constraints);
    }
}
