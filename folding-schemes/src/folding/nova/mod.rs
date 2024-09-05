/// Implements the scheme described in [Nova](https://eprint.iacr.org/2021/370.pdf) and
/// [CycleFold](https://eprint.iacr.org/2023/1192.pdf).
use ark_crypto_primitives::{
    crh::{poseidon::CRH, CRHScheme},
    sponge::{poseidon::PoseidonConfig, Absorb},
};
use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField, ToConstraintField};
use ark_r1cs_std::{convert::ToConstraintFieldGadget, groups::GroupOpsBounds, prelude::CurveVar};
use ark_std::UniformRand;
use ark_std::{
    add_to_trace,
    rand::{thread_rng, Rng},
};
use ark_std::{end_timer, fmt::Debug, start_timer};
use ark_std::{One, Zero};
use core::marker::PhantomData;
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use std::{cmp::max, time::Instant};

use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};

use crate::commitment::CommitmentScheme;
use crate::folding::circuits::nonnative::{
    affine::nonnative_affine_to_field_elements, uint::nonnative_field_to_field_elements,
};
use crate::folding::nova::circuits::IO_LEN;
use crate::frontend::FCircuit;
use crate::frontend::LookupArgument;
use crate::utils::vec::is_zero_vec;
use crate::Error;
use crate::FoldingScheme;
use crate::MSM;
use crate::{
    ccs::r1cs::{extract_r1cs, extract_w_x, R1CS},
    MVM,
};

pub mod circuits;
pub mod cyclefold;
// pub mod decider_eth;
// pub mod decider_eth_circuit;
pub mod decider_pedersen;
pub mod nifs;
pub mod traits;

use circuits::{AugmentedFCircuit, ChallengeGadget, CF2};
use cyclefold::{CycleFoldChallengeGadget, CycleFoldCircuit};
use nifs::NIFS;
use traits::NovaR1CS;

#[cfg(test)]
use cyclefold::CF_IO_LEN;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RunningInstance<C: CurveGroup> {
    pub cmE: C,
    pub u: C::ScalarField,
    pub cmQ: C,
    pub cmW: C,
    pub x: Vec<C::ScalarField>,
}

impl<C: CurveGroup> RunningInstance<C> {
    pub fn dummy(io_len: usize) -> Self {
        Self {
            cmE: C::zero(),
            u: C::ScalarField::zero(),
            cmQ: C::zero(),
            cmW: C::zero(),
            x: vec![C::ScalarField::zero(); io_len],
        }
    }
}

impl<C: CurveGroup> RunningInstance<C>
where
    C::ScalarField: Absorb,
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    /// hash implements the committed instance hash compatible with the gadget implemented in
    /// nova/circuits.rs::CommittedInstanceVar.hash.
    /// Returns `H(i, z_0, z_i, U_i)`, where `i` can be `i` but also `i+1`, and `U_i` is the
    /// `CommittedInstance`.
    pub fn hash(
        &self,
        poseidon_config: &PoseidonConfig<C::ScalarField>,
        i: C::ScalarField,
        z_0: Vec<C::ScalarField>,
        z_i: Vec<C::ScalarField>,
    ) -> Result<C::ScalarField, Error> {
        CRH::<C::ScalarField>::evaluate(
            poseidon_config,
            vec![
                vec![i],
                z_0,
                z_i,
                vec![self.u],
                self.x.clone(),
                nonnative_affine_to_field_elements::<C>(self.cmE),
                nonnative_affine_to_field_elements::<C>(self.cmQ),
                nonnative_affine_to_field_elements::<C>(self.cmW),
            ]
            .concat(),
        )
        .map_err(|e| Error::Other(e.to_string()))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CycleFoldCommittedInstance<C: CurveGroup> {
    pub cmE: C,
    pub u: C::ScalarField,
    pub cmW: C,
    pub x: Vec<C::ScalarField>,
}

impl<C: CurveGroup> CycleFoldCommittedInstance<C> {
    pub fn dummy(io_len: usize) -> Self {
        Self {
            cmE: C::zero(),
            u: C::ScalarField::zero(),
            cmW: C::zero(),
            x: vec![C::ScalarField::zero(); io_len],
        }
    }
}

impl<C: CurveGroup> ToConstraintField<C::BaseField> for CycleFoldCommittedInstance<C>
where
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField + Absorb,
{
    fn to_field_elements(&self) -> Option<Vec<C::BaseField>> {
        let u = nonnative_field_to_field_elements(&self.u);
        let x = self
            .x
            .iter()
            .flat_map(nonnative_field_to_field_elements)
            .collect::<Vec<_>>();
        let (cmE_x, cmE_y, cmE_is_inf) = match self.cmE.into_affine().xy() {
            Some((x, y)) => (x, y, C::BaseField::zero()),
            None => (
                C::BaseField::zero(),
                C::BaseField::zero(),
                C::BaseField::one(),
            ),
        };
        let (cmW_x, cmW_y, cmW_is_inf) = match self.cmW.into_affine().xy() {
            Some((x, y)) => (x, y, C::BaseField::zero()),
            None => (
                C::BaseField::zero(),
                C::BaseField::zero(),
                C::BaseField::one(),
            ),
        };
        // Concatenate `cmE_is_inf` and `cmW_is_inf` to save constraints for CRHGadget::evaluate in the corresponding circuit
        let is_inf = cmE_is_inf.double() + cmW_is_inf;

        Some([u, x, vec![cmE_x, cmE_y, cmW_x, cmW_y, is_inf]].concat())
    }
}

impl<C: CurveGroup> CycleFoldCommittedInstance<C>
where
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField + Absorb,
{
    /// hash_cyclefold implements the committed instance hash compatible with the gadget implemented in
    /// nova/cyclefold.rs::CycleFoldCommittedInstanceVar.hash.
    /// Returns `H(U_i)`, where `U_i` is the `CommittedInstance` for CycleFold.
    pub fn hash_cyclefold(
        &self,
        poseidon_config: &PoseidonConfig<C::BaseField>,
    ) -> Result<C::BaseField, Error> {
        CRH::<C::BaseField>::evaluate(poseidon_config, self.to_field_elements().unwrap())
            .map_err(|e| Error::Other(e.to_string()))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CurrentInstance<C: CurveGroup> {
    pub cmE: C,
    pub u: C::ScalarField,
    pub cmQ: C,
    pub cmW: C,
    pub x: Vec<C::ScalarField>,
}

impl<C: CurveGroup> CurrentInstance<C> {
    pub fn dummy(io_len: usize) -> Self {
        Self {
            cmE: C::zero(),
            u: C::ScalarField::zero(),
            cmQ: C::zero(),
            cmW: C::zero(),
            x: vec![C::ScalarField::zero(); io_len],
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Witness<C: CurveGroup> {
    pub E: Vec<C::ScalarField>,
    pub rE: C::ScalarField,
    pub QW: Vec<C::ScalarField>,
    pub rQ: C::ScalarField,
    pub rW: C::ScalarField,
    Q_len: usize,
}

impl<C: CurveGroup> Witness<C> {
    pub fn new(mut qw: Vec<C::ScalarField>, r1cs: &R1CS<C::ScalarField>) -> Self {
        // note: at the current version, we don't use the blinding factors and we set them to 0
        // always.
        Self {
            E: vec![C::ScalarField::zero(); r1cs.A.n_rows],
            rE: C::ScalarField::zero(),
            QW: qw,
            rQ: C::ScalarField::zero(),
            rW: C::ScalarField::zero(),
            Q_len: r1cs.q,
        }
    }

    #[inline]
    pub fn q(&self) -> &[C::ScalarField] {
        &self.QW[..self.Q_len]
    }

    #[inline]
    pub fn w(&self) -> &[C::ScalarField] {
        &self.QW[self.Q_len..]
    }

    pub fn commit_running<CS: CommitmentScheme<C>>(
        &self,
        params: &CS::ProverParams,
        x: Vec<C::ScalarField>,
    ) -> Result<RunningInstance<C>, Error>
    where
        C::Config: MSM<C>,
    {
        let mut cmE = C::zero();
        if !is_zero_vec::<C::ScalarField>(&self.E) {
            cmE = CS::commit(params, &self.E, &self.rE)?;
        }
        Ok(RunningInstance {
            cmE,
            u: C::ScalarField::one(),
            cmQ: CS::commit(params, &self.q(), &self.rQ)?,
            cmW: CS::commit(params, &self.w(), &self.rW)?,
            x,
        })
    }

    pub fn commit_cyclefold<CS: CommitmentScheme<C>>(
        &self,
        params: &CS::ProverParams,
        x: Vec<C::ScalarField>,
    ) -> Result<CycleFoldCommittedInstance<C>, Error>
    where
        C::Config: MSM<C>,
    {
        let mut cmE = C::zero();
        if !is_zero_vec::<C::ScalarField>(&self.E) {
            cmE = CS::commit(params, &self.E, &self.rE)?;
        }
        assert_eq!(self.Q_len, 0);
        let cmW = CS::commit(params, &self.QW, &self.rW)?;
        Ok(CycleFoldCommittedInstance {
            cmE,
            u: C::ScalarField::one(),
            cmW,
            x,
        })
    }

    pub fn commit_current<CS: CommitmentScheme<C>>(
        &self,
        params: &CS::ProverParams,
        x: Vec<C::ScalarField>,
    ) -> Result<CurrentInstance<C>, Error>
    where
        C::Config: MSM<C>,
    {
        let mut cmE = C::zero();
        if !is_zero_vec::<C::ScalarField>(&self.E) {
            cmE = CS::commit(params, &self.E, &self.rE)?;
        }
        Ok(CurrentInstance {
            cmE,
            u: C::ScalarField::one(),
            cmQ: CS::commit(params, &self.q(), &self.rQ)?,
            cmW: CS::commit(params, &self.w(), &self.rW)?,
            x,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ProverParams<C1, C2, CS1, CS2>
where
    C1: CurveGroup,
    C1::Config: MSM<C1>,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    CS1: CommitmentScheme<C1>,
    CS2: CommitmentScheme<C2>,
{
    pub poseidon_config: PoseidonConfig<C1::ScalarField>,
    pub cs_params: CS1::ProverParams,
    pub cf_cs_params: CS2::ProverParams,
}

#[derive(Clone)]
pub struct VerifierParams<C1: CurveGroup, C2: CurveGroup> {
    pub poseidon_config: PoseidonConfig<C1::ScalarField>,
    pub r1cs: R1CS<C1::ScalarField>,
    pub cf_r1cs: R1CS<C2::ScalarField>,
}

/// Implements Nova+CycleFold's IVC, described in [Nova](https://eprint.iacr.org/2021/370.pdf) and
/// [CycleFold](https://eprint.iacr.org/2023/1192.pdf), following the FoldingScheme trait
#[derive(Clone)]
pub struct Nova<C1, GC1, C2, GC2, FC, CS1, CS2>
where
    C1: CurveGroup,
    C1::Config: MSM<C1>,
    GC1: CurveVar<C1, CF2<C1>> + ToConstraintFieldGadget<CF2<C1>>,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    GC2: CurveVar<C2, CF2<C2>>,
    FC: FCircuit<C1::ScalarField>,
    CS1: CommitmentScheme<C1>,
    CS2: CommitmentScheme<C2>,
{
    _gc1: PhantomData<GC1>,
    _c2: PhantomData<C2>,
    _gc2: PhantomData<GC2>,
    _cs1: PhantomData<CS1>,
    _cs2: PhantomData<CS2>,
    pub poseidon_config: PoseidonConfig<C1::ScalarField>,
    /// F circuit, the circuit that is being folded
    pub F: FC,
    pub i: C1::ScalarField,
    /// initial state
    pub z_0: Vec<C1::ScalarField>,
    /// current i-th state
    pub z_i: Vec<C1::ScalarField>,
    /// Nova instances
    pub w_i: Witness<C1>,
    pub u_i: CurrentInstance<C1>,
    pub W_i: Witness<C1>,
    pub U_i: RunningInstance<C1>,

    /// CycleFold running instance
    pub cf_W_i: Witness<C2>,
    pub cf_U_i: CycleFoldCommittedInstance<C2>,
}

impl<C1, GC1, C2, GC2, FC, CS1, CS2> FoldingScheme<C1, C2, FC>
    for Nova<C1, GC1, C2, GC2, FC, CS1, CS2>
where
    C1: CurveGroup,
    C1::Config: MSM<C1>,
    GC1: CurveVar<C1, CF2<C1>> + ToConstraintFieldGadget<CF2<C1>>,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    GC2: CurveVar<C2, CF2<C2>> + ToConstraintFieldGadget<CF2<C2>>,
    FC: FCircuit<C1::ScalarField>,
    CS1: CommitmentScheme<C1>,
    CS2: CommitmentScheme<C2>,
    <C1 as CurveGroup>::BaseField: PrimeField,
    <C2 as CurveGroup>::BaseField: PrimeField,
    C1::ScalarField: Absorb + MVM,
    C2::ScalarField: Absorb + MVM,
    C1: CurveGroup<BaseField = C2::ScalarField, ScalarField = C2::BaseField>,
    for<'a> &'a GC1: GroupOpsBounds<'a, C1, GC1>,
    for<'a> &'a GC2: GroupOpsBounds<'a, C2, GC2>,
{
    type PreprocessorParam = PoseidonConfig<C1::ScalarField>;
    type ProverParam = ProverParams<C1, C2, CS1, CS2>;
    type VerifierParam = VerifierParams<C1, C2>;
    type RunningCommittedInstanceWithWitness = (RunningInstance<C1>, Witness<C1>);
    type CurrentCommittedInstanceWithWitness = (CurrentInstance<C1>, Witness<C1>);
    type CFCommittedInstanceWithWitness = (CycleFoldCommittedInstance<C2>, Witness<C2>);

    fn preprocess<R: Rng>(
        poseidon_config: &Self::PreprocessorParam,
        step_circuit: &FC,
        rng: &mut R,
        external_inputs: &FC::ExternalInputs,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        // prepare the circuit to obtain its R1CS
        let cs = ConstraintSystem::<C1::ScalarField>::new_ref();
        let cs2 = ConstraintSystem::<C1::BaseField>::new_ref();

        let la = LookupArgument::new_ref();
        let augmented_F_circuit = AugmentedFCircuit::<C1, C2, GC2, FC>::empty(
            poseidon_config,
            la.clone(),
            step_circuit,
            external_inputs,
        );
        let cf_circuit = CycleFoldCircuit::<C1, GC1>::empty();

        augmented_F_circuit.generate_constraints(cs.clone())?;
        la.build_histo(cs.clone())?;
        la.generate_lookup_constraints(cs.clone(), C1::ScalarField::rand(rng))?;

        cs.finalize();
        let cs = cs.into_inner().ok_or(Error::NoInnerConstraintSystem)?;
        add_to_trace!(|| "R1CS", || format!(
            "{} constraints, {} variables (q: {}, w: {}, x: {}, u: 1)",
            cs.num_constraints,
            cs.num_committed_variables + cs.num_witness_variables + cs.num_instance_variables,
            cs.num_committed_variables,
            cs.num_witness_variables,
            cs.num_instance_variables - 1,
        ));
        let r1cs = extract_r1cs::<C1::ScalarField>(&cs);

        cf_circuit.generate_constraints(cs2.clone())?;
        cs2.finalize();
        let cs2 = cs2.into_inner().ok_or(Error::NoInnerConstraintSystem)?;
        let cf_r1cs = extract_r1cs::<C1::BaseField>(&cs2);

        let (cs_params, _) = CS1::setup(rng, max(r1cs.A.n_rows, r1cs.A.n_cols))?;
        let (cf_cs_params, _) = CS2::setup(rng, max(cf_r1cs.A.n_rows, cf_r1cs.A.n_cols))?;

        Ok((
            Self::ProverParam {
                poseidon_config: poseidon_config.clone(),
                cs_params,
                cf_cs_params,
            },
            Self::VerifierParam {
                poseidon_config: poseidon_config.clone(),
                r1cs,
                cf_r1cs,
            },
        ))
    }

    /// Initializes the Nova+CycleFold's IVC for the given parameters and initial state `z_0`.
    fn init(
        (pp, vp): &(Self::ProverParam, Self::VerifierParam),
        F: FC,
        z_0: Vec<C1::ScalarField>,
    ) -> Result<Self, Error> {
        // setup the dummy instances
        let (w_dummy, u_dummy) = vp.r1cs.dummy_current_instance();
        let (W_dummy, U_dummy) = vp.r1cs.dummy_running_instance();
        let (cf_w_dummy, cf_u_dummy) = vp.cf_r1cs.dummy_cyclefold_instance();

        // W_dummy=W_0 is a 'dummy witness', all zeroes, but with the size corresponding to the
        // R1CS that we're working with.
        Ok(Self {
            _gc1: PhantomData,
            _c2: PhantomData,
            _gc2: PhantomData,
            _cs1: PhantomData,
            _cs2: PhantomData,
            poseidon_config: pp.poseidon_config.clone(),
            F,
            i: C1::ScalarField::zero(),
            z_0: z_0.clone(),
            z_i: z_0,
            w_i: w_dummy,
            u_i: u_dummy,
            W_i: W_dummy,
            U_i: U_dummy,
            // cyclefold running instance
            cf_W_i: cf_w_dummy.clone(),
            cf_U_i: cf_u_dummy.clone(),
        })
    }

    /// Implements IVC.P of Nova+CycleFold
    fn prove_step(
        &mut self,
        (pp, vp): &(Self::ProverParam, Self::VerifierParam),
        external_inputs: &FC::ExternalInputs,
    ) -> Result<(), Error> {
        let augmented_F_circuit: AugmentedFCircuit<C1, C2, GC2, FC>;

        let timer = start_timer!(|| "Run step function natively");
        let i: BigUint = self.i.into();
        let i_usize = i.to_usize().unwrap();
        let z_i1 = self
            .F
            .step_native(i_usize, self.z_i.clone(), external_inputs)?;
        end_timer!(timer);

        let timer = start_timer!(|| "Compute primary cross term");
        // compute T and cmT for AugmentedFCircuit
        let (T, cmT) = NIFS::<C1, CS1>::compute_cmT(
            &pp.cs_params,
            &vp.r1cs,
            &self.W_i,
            &self.U_i,
            &self.w_i,
            &self.u_i,
        )?;
        end_timer!(timer);

        let timer = start_timer!(|| "Generate primary challenge");
        // r_bits is the r used to the RLC of the F' instances
        let r_bits = ChallengeGadget::<C1>::get_challenge_native(
            &self.poseidon_config,
            &self.U_i,
            &self.u_i,
            cmT,
        )?;
        let r_Fr = C1::ScalarField::from_bigint(BigInteger::from_bits_le(&r_bits))
            .ok_or(Error::OutOfBounds)?;
        let r_Fq = C1::BaseField::from_bigint(BigInteger::from_bits_le(&r_bits))
            .ok_or(Error::OutOfBounds)?;
        end_timer!(timer);

        let timer = start_timer!(|| "Fold primary instances and witnesses");
        // fold Nova instances
        let (W_i1, U_i1) = NIFS::<C1, CS1>::fold_instances(
            r_Fr, &self.W_i, &self.U_i, &self.w_i, &self.u_i, &T, cmT,
        )?;
        end_timer!(timer);

        let timer = start_timer!(|| "Hash primary instance");
        // folded instance output (public input, x)
        // u_{i+1}.x[0] = H(i+1, z_0, z_{i+1}, U_{i+1})
        let u_i1_x = U_i1.hash(
            &self.poseidon_config,
            self.i + C1::ScalarField::one(),
            self.z_0.clone(),
            z_i1.clone(),
        )?;
        end_timer!(timer);
        // u_{i+1}.x[1] = H(cf_U_{i+1})
        let cf_u_i1_x: C1::ScalarField;

        let la = LookupArgument::new_ref();

        if self.i == C1::ScalarField::zero() {
            let timer = start_timer!(|| "Hash CycleFold instance");
            cf_u_i1_x = self.cf_U_i.hash_cyclefold(&self.poseidon_config)?;
            end_timer!(timer);
            // base case
            let timer = start_timer!(|| "Build augmented circuit");
            augmented_F_circuit = AugmentedFCircuit::<C1, C2, GC2, FC> {
                _gc2: PhantomData,
                la: la.clone(),
                poseidon_config: self.poseidon_config.clone(),
                i: Some(C1::ScalarField::zero()), // = i=0
                i_usize: Some(0),
                z_0: Some(self.z_0.clone()), // = z_i
                z_i: Some(self.z_i.clone()),
                u_i_cmQ: Some(self.u_i.cmQ), // = dummy
                u_i_cmW: Some(self.u_i.cmW), // = dummy
                U_i: Some(self.U_i.clone()), // = dummy
                U_i1_cmE: Some(U_i1.cmE),
                U_i1_cmQ: Some(U_i1.cmQ),
                U_i1_cmW: Some(U_i1.cmW),
                cmT: Some(cmT),
                F: &self.F,
                x: Some(u_i1_x),
                cf1_u_i_cmW: None,
                cf2_u_i_cmW: None,
                cf3_u_i_cmW: None,
                cf_U_i: None,
                cf1_cmT: None,
                cf2_cmT: None,
                cf3_cmT: None,
                cf_x: Some(cf_u_i1_x),
                external_inputs,
            };
            end_timer!(timer);

            #[cfg(test)]
            NIFS::<C1, CS1>::verify_folded_instance(r_Fr, &self.U_i, &self.u_i, &U_i1, &cmT)?;
        } else {
            // CycleFold part:

            let timer = start_timer!(|| "Run and fold CycleFold circuits for Q");
            // fold self.cf_U_i + cfW_U -> folded running with cfW
            let (cfWa_u_i, cfWa_W_i1, cfWa_U_i1, cfWa_cmT) = self.fold_cyclefold_circuit(
                &vp.cf_r1cs,
                &pp.cf_cs_params,
                &self.cf_W_i,
                &self.cf_U_i,
                CycleFoldCircuit::<C1, GC1> {
                    _gc: PhantomData,
                    r_bits: Some(r_bits.clone()),
                    p1: Some(self.U_i.cmQ),
                    p2: Some(self.u_i.cmQ),
                    x: Some(
                        [
                            vec![r_Fq],
                            get_cm_coordinates(&self.U_i.cmQ),
                            get_cm_coordinates(&self.u_i.cmQ),
                            get_cm_coordinates(&U_i1.cmQ),
                        ]
                        .concat(),
                    ),
                },
            )?;
            end_timer!(timer);

            let timer = start_timer!(|| "Run and fold CycleFold circuits for W");
            let (cfWb_u_i, cfWb_W_i1, cfWb_U_i1, cfWb_cmT) = self.fold_cyclefold_circuit(
                &vp.cf_r1cs,
                &pp.cf_cs_params,
                &cfWa_W_i1,
                &cfWa_U_i1,
                CycleFoldCircuit::<C1, GC1> {
                    _gc: PhantomData,
                    r_bits: Some(r_bits.clone()),
                    p1: Some(self.U_i.cmW),
                    p2: Some(self.u_i.cmW),
                    x: Some(
                        [
                            vec![r_Fq],
                            get_cm_coordinates(&self.U_i.cmW),
                            get_cm_coordinates(&self.u_i.cmW),
                            get_cm_coordinates(&U_i1.cmW),
                        ]
                        .concat(),
                    ),
                },
            )?;
            end_timer!(timer);

            // fold [the output from folding self.cf_U_i + cfW_U] + cfE_U = folded_running_with_cfW + cfE
            let timer = start_timer!(|| "Run and fold CycleFold circuits for E");
            let (cfE_u_i, cfE_W_i1, cfE_U_i1, cfE_cmT) = self.fold_cyclefold_circuit(
                &vp.cf_r1cs,
                &pp.cf_cs_params,
                &cfWb_W_i1,
                &cfWb_U_i1,
                CycleFoldCircuit::<C1, GC1> {
                    _gc: PhantomData,
                    r_bits: Some(r_bits.clone()),
                    p1: Some(self.U_i.cmE),
                    p2: Some(cmT),
                    x: Some(
                        [
                            vec![r_Fq],
                            get_cm_coordinates(&self.U_i.cmE),
                            get_cm_coordinates(&cmT),
                            get_cm_coordinates(&U_i1.cmE),
                        ]
                        .concat(),
                    ),
                },
            )?;
            end_timer!(timer);

            let timer = start_timer!(|| "Hash CycleFold instance");
            cf_u_i1_x = cfE_U_i1.hash_cyclefold(&self.poseidon_config)?;
            end_timer!(timer);

            let timer = start_timer!(|| "Build augmented circuit");
            augmented_F_circuit = AugmentedFCircuit::<C1, C2, GC2, FC> {
                _gc2: PhantomData,
                la: la.clone(),
                poseidon_config: self.poseidon_config.clone(),
                i: Some(self.i),
                i_usize: Some(i_usize),
                z_0: Some(self.z_0.clone()),
                z_i: Some(self.z_i.clone()),
                u_i_cmQ: Some(self.u_i.cmQ),
                u_i_cmW: Some(self.u_i.cmW),
                U_i: Some(self.U_i.clone()),
                U_i1_cmE: Some(U_i1.cmE),
                U_i1_cmQ: Some(U_i1.cmQ),
                U_i1_cmW: Some(U_i1.cmW),
                cmT: Some(cmT),
                F: &self.F,
                x: Some(u_i1_x),
                // cyclefold values
                cf1_u_i_cmW: Some(cfWa_u_i.cmW),
                cf2_u_i_cmW: Some(cfWb_u_i.cmW),
                cf3_u_i_cmW: Some(cfE_u_i.cmW),
                cf_U_i: Some(self.cf_U_i.clone()),
                cf1_cmT: Some(cfWa_cmT),
                cf2_cmT: Some(cfWb_cmT),
                cf3_cmT: Some(cfE_cmT),
                cf_x: Some(cf_u_i1_x),
                external_inputs,
            };
            end_timer!(timer);

            self.cf_W_i = cfE_W_i1;
            self.cf_U_i = cfE_U_i1;

            #[cfg(test)]
            {
                vp.cf_r1cs
                    .check_relaxed_cyclefold_instance_relation(&self.cf_W_i, &self.cf_U_i)?;
            }
        }

        let timer = start_timer!(|| "Synthesize primary circuit");
        let cs = ConstraintSystem::<C1::ScalarField>::new_ref();
        #[cfg(not(test))]
        cs.set_mode(ark_relations::r1cs::SynthesisMode::Prove {
            construct_matrices: false,
        });

        augmented_F_circuit.generate_constraints(cs.clone())?;
        end_timer!(timer);

        let timer = start_timer!(|| "Build lookup histogram");
        la.build_histo(cs.clone())?;
        end_timer!(timer);

        let timer = start_timer!(|| "Commit to Q");
        let cmQ = CS1::commit(
            &pp.cs_params,
            &cs.borrow()
                .ok_or(Error::NoInnerConstraintSystem)?
                .committed_assignment,
            &C1::ScalarField::zero(),
        )?;
        end_timer!(timer);

        let t = cs.num_constraints();
        let timer = start_timer!(|| "Synthesize lookup constraints");
        la.generate_lookup_constraints(
            cs.clone(),
            CRH::evaluate(
                &self.poseidon_config,
                nonnative_affine_to_field_elements(cmQ),
            )
            .unwrap(),
        )?;
        end_timer!(timer);
        add_to_trace!(|| "Check lookup identity", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(test)]
        assert!(cs.is_satisfied().unwrap());

        let timer = start_timer!(|| "Extract primary variables");
        let cs = cs.into_inner().ok_or(Error::NoInnerConstraintSystem)?;
        let (w_i1, x_i1) = extract_w_x::<C1::ScalarField>(&cs);
        end_timer!(timer);

        let timer = start_timer!(|| "Commit to W");
        let cmW = CS1::commit(
            &pp.cs_params,
            &cs.witness_assignment,
            &C1::ScalarField::zero(),
        )?;
        end_timer!(timer);

        #[cfg(test)]
        if x_i1[0] != u_i1_x || x_i1[1] != cf_u_i1_x {
            return Err(Error::NotEqual);
        }

        #[cfg(test)]
        if x_i1.len() != IO_LEN {
            return Err(Error::NotExpectedLength(x_i1.len(), IO_LEN));
        }

        // set values for next iteration
        let timer = start_timer!(|| "Build Primary witnesses and instances");
        self.i += C1::ScalarField::one();
        self.z_i = z_i1;
        self.w_i = Witness::<C1>::new(w_i1, &vp.r1cs);
        self.u_i = CurrentInstance {
            cmE: C1::zero(),
            u: C1::ScalarField::one(),
            cmQ,
            cmW,
            x: x_i1,
        };
        self.W_i = W_i1;
        self.U_i = U_i1;
        end_timer!(timer);

        #[cfg(test)]
        {
            vp.r1cs
                .check_current_instance_relation(&self.w_i, &self.u_i)?;
            vp.r1cs
                .check_relaxed_running_instance_relation(&self.W_i, &self.U_i)?;
        }

        Ok(())
    }

    fn state(&self) -> Vec<C1::ScalarField> {
        self.z_i.clone()
    }
    fn instances(
        &self,
    ) -> (
        Self::RunningCommittedInstanceWithWitness,
        Self::CurrentCommittedInstanceWithWitness,
        Self::CFCommittedInstanceWithWitness,
    ) {
        (
            (self.U_i.clone(), self.W_i.clone()),
            (self.u_i.clone(), self.w_i.clone()),
            (self.cf_U_i.clone(), self.cf_W_i.clone()),
        )
    }

    /// Implements IVC.V of Nova+CycleFold
    fn verify(
        vp: &Self::VerifierParam,
        z_0: Vec<C1::ScalarField>, // initial state
        z_i: Vec<C1::ScalarField>, // last state
        num_steps: C1::ScalarField,
        running_instance: Self::RunningCommittedInstanceWithWitness,
        incoming_instance: Self::CurrentCommittedInstanceWithWitness,
        cyclefold_instance: Self::CFCommittedInstanceWithWitness,
    ) -> Result<(), Error> {
        let (U_i, W_i) = running_instance;
        let (u_i, w_i) = incoming_instance;
        let (cf_U_i, cf_W_i) = cyclefold_instance;

        if u_i.x.len() != IO_LEN || U_i.x.len() != IO_LEN {
            return Err(Error::IVCVerificationFail);
        }

        // check that u_i's output points to the running instance
        // u_i.X[0] == H(i, z_0, z_i, U_i)
        let expected_u_i_x = U_i.hash(&vp.poseidon_config, num_steps, z_0, z_i.clone())?;
        if expected_u_i_x != u_i.x[0] {
            return Err(Error::IVCVerificationFail);
        }
        // u_i.X[1] == H(cf_U_i)
        let expected_cf_u_i_x = cf_U_i.hash_cyclefold(&vp.poseidon_config)?;
        if expected_cf_u_i_x != u_i.x[1] {
            return Err(Error::IVCVerificationFail);
        }

        // check u_i.cmE==0, u_i.u==1 (=u_i is a un-relaxed instance)
        if !u_i.cmE.is_zero() || !u_i.u.is_one() {
            return Err(Error::IVCVerificationFail);
        }

        // check R1CS satisfiability
        vp.r1cs.check_current_instance_relation(&w_i, &u_i)?;
        // check RelaxedR1CS satisfiability
        vp.r1cs
            .check_relaxed_running_instance_relation(&W_i, &U_i)?;

        // check CycleFold RelaxedR1CS satisfiability
        vp.cf_r1cs
            .check_relaxed_cyclefold_instance_relation(&cf_W_i, &cf_U_i)?;

        Ok(())
    }
}


impl<C1, GC1, C2, GC2, FC, CS1, CS2> Nova<C1, GC1, C2, GC2, FC, CS1, CS2>
where
    C1: CurveGroup,
    C1::Config: MSM<C1>,
    GC1: CurveVar<C1, CF2<C1>> + ToConstraintFieldGadget<CF2<C1>>,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    GC2: CurveVar<C2, CF2<C2>> + ToConstraintFieldGadget<CF2<C2>>,
    FC: FCircuit<C1::ScalarField>,
    CS1: CommitmentScheme<C1>,
    CS2: CommitmentScheme<C2>,
    <C1 as CurveGroup>::BaseField: PrimeField,
    <C2 as CurveGroup>::BaseField: PrimeField,
    C1::ScalarField: Absorb + MVM,
    C2::ScalarField: Absorb + MVM,
    C1: CurveGroup<BaseField = C2::ScalarField, ScalarField = C2::BaseField>,
    for<'a> &'a GC1: GroupOpsBounds<'a, C1, GC1>,
    for<'a> &'a GC2: GroupOpsBounds<'a, C2, GC2>,
{
    // folds the given cyclefold circuit and its instances
    #[allow(clippy::type_complexity)]
    fn fold_cyclefold_circuit(
        &self,
        cf_r1cs: &R1CS<C2::ScalarField>,
        cf_cs_params: &CS2::ProverParams,
        cf_W_i: &Witness<C2>,                    // witness of the running instance
        cf_U_i: &CycleFoldCommittedInstance<C2>, // running instance
        cf_circuit: CycleFoldCircuit<C1, GC1>,
    ) -> Result<
        (
            CycleFoldCommittedInstance<C2>, // u_i
            Witness<C2>,                    // W_i1
            CycleFoldCommittedInstance<C2>, // U_i1
            C2,                             // cmT
        ),
        Error,
    > {
        let timer = start_timer!(|| "Synthesize CycleFold circuit");
        let cs2 = ConstraintSystem::<C1::BaseField>::new_ref();
        #[cfg(not(test))]
        cs2.set_mode(ark_relations::r1cs::SynthesisMode::Prove {
            construct_matrices: false,
        });
        cf_circuit.generate_constraints(cs2.clone())?;
        end_timer!(timer);

        let timer = start_timer!(|| "Extract CycleFold variables");
        let cs2 = cs2.into_inner().ok_or(Error::NoInnerConstraintSystem)?;
        let (cf_w_i, cf_x_i) = extract_w_x::<C1::BaseField>(&cs2);
        end_timer!(timer);

        #[cfg(test)]
        if cf_x_i.len() != CF_IO_LEN {
            return Err(Error::NotExpectedLength(cf_x_i.len(), CF_IO_LEN));
        }

        // fold cyclefold instances
        let timer = start_timer!(|| "Build CycleFold incoming witness");
        let cf_w_i = Witness::<C2>::new(cf_w_i, &cf_r1cs);
        end_timer!(timer);

        let timer = start_timer!(|| "Build CycleFold incoming instance");
        let cf_u_i = CycleFoldCommittedInstance {
            cmE: C2::zero(),
            u: C2::ScalarField::one(),
            cmW: {
                let timer = start_timer!(|| "Commit to W");
                let cmW = CS2::commit(cf_cs_params, &cf_w_i.QW, &cf_w_i.rW)?;
                end_timer!(timer);
                cmW
            },
            x: cf_x_i,
        };
        end_timer!(timer);

        // compute T* and cmT* for CycleFoldCircuit
        let timer = start_timer!(|| "Compute CycleFold cross term");
        let (cf_T, cf_cmT) = NIFS::<C2, CS2>::compute_cyclefold_cmT(
            cf_cs_params,
            cf_r1cs,
            &cf_w_i,
            &cf_u_i,
            &cf_W_i,
            &cf_U_i,
        )?;
        end_timer!(timer);

        let timer = start_timer!(|| "Generate CycleFold challenge");
        let cf_r_bits = CycleFoldChallengeGadget::<C2, GC2>::get_challenge_native(
            &self.poseidon_config,
            &cf_U_i,
            &cf_u_i,
            cf_cmT,
        )?;
        let cf_r_Fq = C1::BaseField::from_bigint(BigInteger::from_bits_le(&cf_r_bits))
            .ok_or(Error::OutOfBounds)?;
        end_timer!(timer);

        let timer = start_timer!(|| "Fold CycleFold instances and witnesses");
        let (cf_W_i1, cf_U_i1) = NIFS::<C2, CS2>::fold_cf_instances(
            cf_r_Fq, &cf_W_i, &cf_U_i, &cf_w_i, &cf_u_i, &cf_T, cf_cmT,
        )?;
        end_timer!(timer);

        #[cfg(test)]
        {
            cf_r1cs
                .check_cyclefold_instance_relation(&cf_w_i, &cf_u_i)?;
        }

        Ok((cf_u_i, cf_W_i1, cf_U_i1, cf_cmT))
    }
}

/// helper method to get the r1cs from the ConstraintSynthesizer
pub fn get_r1cs_from_cs<F: MVM>(circuit: impl ConstraintSynthesizer<F>) -> Result<R1CS<F>, Error> {
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();
    let cs = cs.into_inner().ok_or(Error::NoInnerConstraintSystem)?;
    let r1cs = extract_r1cs::<F>(&cs);
    Ok(r1cs)
}

/// returns the coordinates of a commitment point. This is compatible with the arkworks
/// GC.to_constraint_field()[..2]
pub(crate) fn get_cm_coordinates<C: CurveGroup>(cm: &C) -> Vec<C::BaseField> {
    let zero = (C::BaseField::zero(), C::BaseField::zero());
    let cm = cm.into_affine();
    let (cm_x, cm_y) = cm.xy().unwrap_or(zero);
    vec![cm_x, cm_y]
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::commitment::kzg::{ProverKey as KZGProverKey, KZG};
    use crate::commitment::pedersen::Pedersen;

    use crate::frontend::LookupArgumentRef;
    use crate::transcript::poseidon::poseidon_test_config;
    use ark_bn254::{constraints::GVar, Bn254, Fr, G1Projective as Projective};
    use ark_grumpkin::{constraints::GVar as GVar2, Projective as Projective2};
    use ark_poly_commit::kzg10::VerifierKey as KZGVerifierKey;
    use ark_r1cs_std::fields::fp::FpVar;
    use ark_r1cs_std::{alloc::AllocVar, fields::FieldVar};
    use ark_relations::r1cs::ConstraintSystemRef;
    use ark_relations::r1cs::SynthesisError;

    #[derive(Clone, Copy, Debug)]
    pub struct LookupCircuit<F: PrimeField> {
        _f: PhantomData<F>,
    }
    impl<F: PrimeField> FCircuit<F> for LookupCircuit<F> {
        type Params = ();
        type ExternalInputs = ();
        fn new(_params: Self::Params) -> Self {
            Self { _f: PhantomData }
        }
        fn state_len(&self) -> usize {
            1
        }
        fn step_native(
            &self,
            _i: usize,
            z_i: Vec<F>,
            _: &Self::ExternalInputs,
        ) -> Result<Vec<F>, Error> {
            Ok(vec![z_i[0] + F::from(5u32)])
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

            Ok(vec![&z_i[0] + five])
        }
    }

    /// This test tests the Nova+CycleFold IVC, and by consequence it is also testing the
    /// AugmentedFCircuit
    #[test]
    fn test_ivc() {
        let mut rng = ark_std::test_rng();
        let poseidon_config = poseidon_test_config::<Fr>();

        let F_circuit = LookupCircuit::<Fr>::new(());

        // run the test using Pedersen commitments on both sides of the curve cycle
        test_ivc_opt::<Pedersen<Projective>, Pedersen<Projective2>>(
            poseidon_config.clone(),
            F_circuit,
        );
        // run the test using KZG for the commitments on the main curve, and Pedersen for the
        // commitments on the secondary curve
        test_ivc_opt::<KZG<Bn254>, Pedersen<Projective2>>(poseidon_config, F_circuit);
    }

    // test_ivc allowing to choose the CommitmentSchemes
    fn test_ivc_opt<CS1: CommitmentScheme<Projective>, CS2: CommitmentScheme<Projective2>>(
        poseidon_config: PoseidonConfig<Fr>,
        F_circuit: LookupCircuit<Fr>,
    ) {
        type NOVA<CS1, CS2> =
            Nova<Projective, GVar, Projective2, GVar2, LookupCircuit<Fr>, CS1, CS2>;

        let params =
            NOVA::<CS1, CS2>::preprocess(&poseidon_config, &F_circuit, &mut thread_rng(), &())
                .unwrap();

        let z_0 = vec![Fr::from(3_u32)];
        let mut nova = NOVA::init(&params, F_circuit, z_0.clone()).unwrap();

        let num_steps: usize = 3;
        for _ in 0..num_steps {
            nova.prove_step(&params, &()).unwrap();
        }
        assert_eq!(Fr::from(num_steps as u32), nova.i);

        let (running_instance, incoming_instance, cyclefold_instance) = nova.instances();
        NOVA::<CS1, CS2>::verify(
            &params.1,
            z_0,
            nova.z_i,
            nova.i,
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )
        .unwrap();
    }
}
