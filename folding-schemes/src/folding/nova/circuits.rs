/// contains [Nova](https://eprint.iacr.org/2021/370.pdf) related circuits
use ark_crypto_primitives::crh::{
    poseidon::constraints::{CRHGadget, CRHParametersVar},
    CRHSchemeGadget,
};
use ark_crypto_primitives::sponge::{
    constraints::CryptographicSpongeVar,
    poseidon::{constraints::PoseidonSpongeVar, PoseidonConfig, PoseidonSponge},
    Absorb, CryptographicSponge,
};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{Field, PrimeField};
use ark_r1cs_std::{
    alloc::{AllocVar, AllocationMode},
    boolean::Boolean,
    convert::ToConstraintFieldGadget,
    eq::EqGadget,
    fields::{fp::FpVar, FieldVar},
    groups::GroupOpsBounds,
    prelude::CurveVar,
    R1CSVar,
};
use ark_relations::r1cs::{ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::{add_to_trace, end_timer, fmt::Debug, start_timer, One, Zero};
use core::{borrow::Borrow, marker::PhantomData};
use icicle_cuda_runtime::{memory::DeviceVec, stream::CudaStream};

use super::{
    cyclefold::{
        CycleFoldChallengeGadget, CycleFoldCommittedInstanceVar, NIFSFullGadget, CF_IO_LEN,
    },
    CurrentInstance, CycleFoldCommittedInstance, RunningInstance,
};
use crate::folding::circuits::nonnative::{
    affine::{nonnative_affine_to_field_elements, NonNativeAffineVar},
    uint::NonNativeUintVar,
};
use crate::frontend::{FCircuit, LookupArgumentRef};
use crate::{constants::N_BITS_RO, MSM};

/// CF1 represents the ConstraintField used for the main Nova circuit which is over E1::Fr, where
/// E1 is the main curve where we do the folding.
pub type CF1<C> = <<C as CurveGroup>::Affine as AffineRepr>::ScalarField;
/// CF2 represents the ConstraintField used for the CycleFold circuit which is over E2::Fr=E1::Fq,
/// where E2 is the auxiliary curve (from [CycleFold](https://eprint.iacr.org/2023/1192.pdf)
/// approach) where we check the folding of the commitments (elliptic curve points).
pub type CF2<C> = <<C as CurveGroup>::BaseField as Field>::BasePrimeField;

/// CommittedInstanceVar contains the u, x, cmE and cmW values which are folded on the main Nova
/// constraints field (E1::Fr, where E1 is the main curve). The peculiarity is that cmE and cmW are
/// represented non-natively over the constraint field.
#[derive(Debug, Clone)]
pub struct RunningInstanceVar<C: CurveGroup>
where
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    pub u: FpVar<C::ScalarField>,
    pub x: Vec<FpVar<C::ScalarField>>,
    pub cmE: NonNativeAffineVar<C>,
    pub cmQ: NonNativeAffineVar<C>,
    pub cmW: NonNativeAffineVar<C>,
}

impl<C> AllocVar<RunningInstance<C>, CF1<C>> for RunningInstanceVar<C>
where
    C: CurveGroup,
    <C as ark_ec::CurveGroup>::BaseField: PrimeField,
{
    fn new_variable<T: Borrow<RunningInstance<C>>>(
        cs: impl Into<Namespace<CF1<C>>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        f().and_then(|val| {
            let cs = cs.into();

            let u = FpVar::<C::ScalarField>::new_variable(cs.clone(), || Ok(val.borrow().u), mode)?;
            let x: Vec<FpVar<C::ScalarField>> =
                Vec::new_variable(cs.clone(), || Ok(val.borrow().x.clone()), mode)?;

            let cmE =
                NonNativeAffineVar::<C>::new_variable(cs.clone(), || Ok(val.borrow().cmE), mode)?;
            let cmQ =
                NonNativeAffineVar::<C>::new_variable(cs.clone(), || Ok(val.borrow().cmQ), mode)?;
            let cmW =
                NonNativeAffineVar::<C>::new_variable(cs.clone(), || Ok(val.borrow().cmW), mode)?;

            Ok(Self {
                u,
                x,
                cmE,
                cmQ,
                cmW,
            })
        })
    }
}

impl<C> RunningInstanceVar<C>
where
    C: CurveGroup,
    C::ScalarField: Absorb,
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    /// hash implements the committed instance hash compatible with the native implementation from
    /// CommittedInstance.hash.
    /// Returns `H(i, z_0, z_i, U_i)`, where `i` can be `i` but also `i+1`, and `U` is the
    /// `CommittedInstance`.
    /// Additionally it returns the vector of the field elements from the self parameters, so they
    /// can be reused in other gadgets avoiding recalculating (reconstraining) them.
    #[allow(clippy::type_complexity)]
    pub fn hash(
        self,
        crh_params: &CRHParametersVar<CF1<C>>,
        i: FpVar<CF1<C>>,
        z_0: Vec<FpVar<CF1<C>>>,
        z_i: Vec<FpVar<CF1<C>>>,
    ) -> Result<(FpVar<CF1<C>>, Vec<FpVar<CF1<C>>>), SynthesisError> {
        let U_vec = [
            vec![self.u],
            self.x,
            self.cmE.to_constraint_field()?,
            self.cmQ.to_constraint_field()?,
            self.cmW.to_constraint_field()?,
        ]
        .concat();
        let input = [vec![i], z_0, z_i, U_vec.clone()].concat();
        Ok((
            CRHGadget::<C::ScalarField>::evaluate(crh_params, &input)?,
            U_vec,
        ))
    }
}

/// CommittedInstanceVar contains the u, x, cmE and cmW values which are folded on the main Nova
/// constraints field (E1::Fr, where E1 is the main curve). The peculiarity is that cmE and cmW are
/// represented non-natively over the constraint field.
#[derive(Debug, Clone)]
pub struct CurrentInstanceVar<C: CurveGroup>
where
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    pub u: FpVar<C::ScalarField>,
    pub x: Vec<FpVar<C::ScalarField>>,
    pub cmE: NonNativeAffineVar<C>,
    pub cmQ: NonNativeAffineVar<C>,
    pub cmW: NonNativeAffineVar<C>,
}

impl<C> AllocVar<CurrentInstance<C>, CF1<C>> for CurrentInstanceVar<C>
where
    C: CurveGroup,
    <C as ark_ec::CurveGroup>::BaseField: PrimeField,
{
    fn new_variable<T: Borrow<CurrentInstance<C>>>(
        cs: impl Into<Namespace<CF1<C>>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        f().and_then(|val| {
            let cs = cs.into();

            let u = FpVar::<C::ScalarField>::new_variable(cs.clone(), || Ok(val.borrow().u), mode)?;
            let x: Vec<FpVar<C::ScalarField>> =
                Vec::new_variable(cs.clone(), || Ok(val.borrow().x.clone()), mode)?;

            let cmE =
                NonNativeAffineVar::<C>::new_variable(cs.clone(), || Ok(val.borrow().cmE), mode)?;
            let cmQ =
                NonNativeAffineVar::<C>::new_variable(cs.clone(), || Ok(val.borrow().cmQ), mode)?;
            let cmW =
                NonNativeAffineVar::<C>::new_variable(cs.clone(), || Ok(val.borrow().cmW), mode)?;

            Ok(Self {
                u,
                x,
                cmE,
                cmQ,
                cmW,
            })
        })
    }
}

impl<C> CurrentInstanceVar<C>
where
    C: CurveGroup,
    C::ScalarField: Absorb,
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    /// hash implements the committed instance hash compatible with the native implementation from
    /// CommittedInstance.hash.
    /// Returns `H(i, z_0, z_i, U_i)`, where `i` can be `i` but also `i+1`, and `U` is the
    /// `CommittedInstance`.
    /// Additionally it returns the vector of the field elements from the self parameters, so they
    /// can be reused in other gadgets avoiding recalculating (reconstraining) them.
    #[allow(clippy::type_complexity)]
    pub fn hash(
        self,
        crh_params: &CRHParametersVar<CF1<C>>,
        i: FpVar<CF1<C>>,
        z_0: Vec<FpVar<CF1<C>>>,
        z_i: Vec<FpVar<CF1<C>>>,
    ) -> Result<(FpVar<CF1<C>>, Vec<FpVar<CF1<C>>>), SynthesisError> {
        let U_vec = [
            vec![self.u],
            self.x,
            self.cmE.to_constraint_field()?,
            self.cmQ.to_constraint_field()?,
            self.cmW.to_constraint_field()?,
        ]
        .concat();
        let input = [vec![i], z_0, z_i, U_vec.clone()].concat();
        Ok((
            CRHGadget::<C::ScalarField>::evaluate(crh_params, &input)?,
            U_vec,
        ))
    }
}

/// Implements the circuit that does the checks of the Non-Interactive Folding Scheme Verifier
/// described in section 4 of [Nova](https://eprint.iacr.org/2021/370.pdf), where the cmE & cmW checks are
/// delegated to the NIFSCycleFoldGadget.
pub struct NIFSGadget<C: CurveGroup> {
    _c: PhantomData<C>,
}

impl<C: CurveGroup> NIFSGadget<C>
where
    C: CurveGroup,
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    pub fn fold_committed_instance(
        r: FpVar<CF1<C>>,
        ci1: RunningInstanceVar<C>, // U_i
        ci2: CurrentInstanceVar<C>, // u_i
    ) -> Result<RunningInstanceVar<C>, SynthesisError> {
        Ok(RunningInstanceVar {
            cmE: NonNativeAffineVar::new_constant(ConstraintSystemRef::None, C::zero())?,
            cmQ: NonNativeAffineVar::new_constant(ConstraintSystemRef::None, C::zero())?,
            cmW: NonNativeAffineVar::new_constant(ConstraintSystemRef::None, C::zero())?,
            // ci3.u = ci1.u + r * ci2.u
            u: ci1.u + &r * ci2.u,
            // ci3.x = ci1.x + r * ci2.x
            x: ci1
                .x
                .iter()
                .zip(ci2.x)
                .map(|(a, b)| a + &r * &b)
                .collect::<Vec<FpVar<CF1<C>>>>(),
        })
    }

    /// Implements the constraints for NIFS.V for u and x, since cm(E) and cm(W) are delegated to
    /// the CycleFold circuit.
    pub fn verify(
        r: FpVar<CF1<C>>,
        ci1: RunningInstanceVar<C>, // U_i
        ci2: CurrentInstanceVar<C>, // u_i
        ci3: RunningInstanceVar<C>, // U_{i+1}
    ) -> Result<(), SynthesisError> {
        let ci = Self::fold_committed_instance(r, ci1, ci2)?;

        ci.u.enforce_equal(&ci3.u)?;
        ci.x.enforce_equal(&ci3.x)?;

        Ok(())
    }
}

/// ChallengeGadget computes the RO challenge used for the Nova instances NIFS, it contains a
/// rust-native and a in-circuit compatible versions.
pub struct ChallengeGadget<C: CurveGroup> {
    _c: PhantomData<C>,
}
impl<C: CurveGroup> ChallengeGadget<C>
where
    C: CurveGroup,
    <C as CurveGroup>::BaseField: PrimeField,
    C::ScalarField: Absorb,
{
    pub fn get_challenge_native(
        poseidon_config: &PoseidonConfig<C::ScalarField>,
        U_i: &RunningInstance<C>,
        u_i: &CurrentInstance<C>,
        cmT: C,
    ) -> Result<Vec<bool>, SynthesisError> {
        let mut sponge = PoseidonSponge::<C::ScalarField>::new(poseidon_config);
        let input = vec![
            vec![U_i.u],
            U_i.x.clone(),
            nonnative_affine_to_field_elements::<C>(U_i.cmE),
            nonnative_affine_to_field_elements::<C>(U_i.cmQ),
            nonnative_affine_to_field_elements::<C>(U_i.cmW),
            vec![u_i.u],
            u_i.x.clone(),
            nonnative_affine_to_field_elements::<C>(u_i.cmE),
            nonnative_affine_to_field_elements::<C>(u_i.cmQ),
            nonnative_affine_to_field_elements::<C>(u_i.cmW),
            nonnative_affine_to_field_elements::<C>(cmT),
        ]
        .concat();
        sponge.absorb(&input);
        let bits = sponge.squeeze_bits(N_BITS_RO);
        Ok(bits)
    }

    pub fn get_challenge_device(
        stream_t: Option<&CudaStream>,
        stream_w: Option<&CudaStream>,
        poseidon_config: &PoseidonConfig<C::ScalarField>,
        U_i: &RunningInstance<C>,
        u_i: &mut CurrentInstance<C>,
        cmT: &DeviceVec<<C::Config as MSM<C>>::R>,
        cmW: &Option<DeviceVec<<C::Config as MSM<C>>::R>>,
    ) -> Result<(Vec<bool>, C), SynthesisError>
    where
        C::Config: MSM<C>,
    {
        let mut sponge = PoseidonSponge::<C::ScalarField>::new(poseidon_config);
        let r;
        let input = vec![
            vec![U_i.u],
            U_i.x.clone(),
            nonnative_affine_to_field_elements::<C>(U_i.cmE),
            nonnative_affine_to_field_elements::<C>(U_i.cmQ),
            nonnative_affine_to_field_elements::<C>(U_i.cmW),
            vec![u_i.u],
            u_i.x.clone(),
            nonnative_affine_to_field_elements::<C>(u_i.cmE),
            nonnative_affine_to_field_elements::<C>(u_i.cmQ),
            nonnative_affine_to_field_elements::<C>({
                if let Some(cmW) = cmW {
                    let cmW = <C::Config as MSM<C>>::retrieve_msm_result(stream_w, cmW);
                    u_i.cmW = cmW;
                }
                u_i.cmW
            }),
            nonnative_affine_to_field_elements::<C>({
                let cmT = <C::Config as MSM<C>>::retrieve_msm_result(stream_t, cmT);
                r = cmT;
                r
            }),
        ]
        .concat();
        sponge.absorb(&input);
        let bits = sponge.squeeze_bits(N_BITS_RO);
        Ok((bits, r))
    }

    // compatible with the native get_challenge_native
    pub fn get_challenge_gadget(
        cs: ConstraintSystemRef<C::ScalarField>,
        poseidon_config: &PoseidonConfig<C::ScalarField>,
        U_i_vec: Vec<FpVar<CF1<C>>>, // apready processed input, so we don't have to recompute these values
        u_i: CurrentInstanceVar<C>,
        cmT: NonNativeAffineVar<C>,
    ) -> Result<Vec<Boolean<C::ScalarField>>, SynthesisError> {
        let mut sponge = PoseidonSpongeVar::<C::ScalarField>::new(cs, poseidon_config);

        let input: Vec<FpVar<C::ScalarField>> = [
            U_i_vec,
            vec![u_i.u.clone()],
            u_i.x.clone(),
            u_i.cmE.to_constraint_field()?,
            u_i.cmQ.to_constraint_field()?,
            u_i.cmW.to_constraint_field()?,
            cmT.to_constraint_field()?,
        ]
        .concat();
        sponge.absorb(&input)?;
        let bits = sponge.squeeze_bits(N_BITS_RO)?;
        Ok(bits)
    }
}

pub const IO_LEN: usize = 3;

/// AugmentedFCircuit implements the F' circuit (augmented F) defined in
/// [Nova](https://eprint.iacr.org/2021/370.pdf) together with the extra constraints defined in
/// [CycleFold](https://eprint.iacr.org/2023/1192.pdf).
pub struct AugmentedFCircuit<
    'b,
    C1: CurveGroup,
    C2: CurveGroup,
    GC2: CurveVar<C2, CF2<C2>>,
    FC: FCircuit<CF1<C1>>,
> where
    for<'a> &'a GC2: GroupOpsBounds<'a, C2, GC2>,
{
    pub la: LookupArgumentRef<CF1<C1>>,
    pub _gc2: PhantomData<GC2>,
    pub poseidon_config: PoseidonConfig<CF1<C1>>,
    pub i: Option<CF1<C1>>,
    pub i_usize: Option<usize>,
    pub z_0: Option<Vec<C1::ScalarField>>,
    pub z_i: Option<Vec<C1::ScalarField>>,
    pub u_i_cmQ: Option<C1>,
    pub u_i_cmW: Option<C1>,
    pub U_i: Option<RunningInstance<C1>>,
    pub U_i1_cmE: Option<C1>,
    pub U_i1_cmQ: Option<C1>,
    pub U_i1_cmW: Option<C1>,
    pub cmT: Option<C1>,
    pub F: &'b FC, // F circuit

    // cyclefold verifier on C1
    // Here 'cf1, cf2' are for each of the CycleFold circuits, corresponding to the fold of cmW and
    // cmE respectively
    pub cf1_u_i_cmW: Option<C2>,                        // input
    pub cf2_u_i_cmW: Option<C2>,                        // input
    pub cf3_u_i_cmW: Option<C2>,                        // input
    pub cf_U_i: Option<CycleFoldCommittedInstance<C2>>, // input
    pub cf1_cmT: Option<C2>,
    pub cf2_cmT: Option<C2>,
    pub cf3_cmT: Option<C2>,

    pub external_inputs: &'b FC::ExternalInputs,
}

impl<'b, C1: CurveGroup, C2: CurveGroup, GC2: CurveVar<C2, CF2<C2>>, FC: FCircuit<CF1<C1>>>
    AugmentedFCircuit<'b, C1, C2, GC2, FC>
where
    for<'a> &'a GC2: GroupOpsBounds<'a, C2, GC2>,
{
    pub fn empty(
        poseidon_config: &PoseidonConfig<CF1<C1>>,
        la: LookupArgumentRef<CF1<C1>>,
        F_circuit: &'b FC,
        external_inputs: &'b FC::ExternalInputs,
    ) -> Self {
        Self {
            _gc2: PhantomData,
            la,
            poseidon_config: poseidon_config.clone(),
            i: None,
            i_usize: None,
            z_0: None,
            z_i: None,
            u_i_cmQ: None,
            u_i_cmW: None,
            U_i: None,
            U_i1_cmE: None,
            U_i1_cmQ: None,
            U_i1_cmW: None,
            cmT: None,
            F: F_circuit,
            // cyclefold values
            cf1_u_i_cmW: None,
            cf2_u_i_cmW: None,
            cf3_u_i_cmW: None,
            cf_U_i: None,
            cf1_cmT: None,
            cf2_cmT: None,
            cf3_cmT: None,
            external_inputs,
        }
    }
}

impl<'b, C1, C2, GC2, FC> AugmentedFCircuit<'b, C1, C2, GC2, FC>
where
    C1: CurveGroup,
    C2: CurveGroup,
    GC2: CurveVar<C2, CF2<C2>> + ToConstraintFieldGadget<CF2<C2>>,
    FC: FCircuit<CF1<C1>>,
    <C1 as CurveGroup>::BaseField: PrimeField,
    <C2 as CurveGroup>::BaseField: PrimeField,
    C1::ScalarField: Absorb,
    C2::ScalarField: Absorb,
    C1: CurveGroup<BaseField = C2::ScalarField, ScalarField = C2::BaseField>,
    for<'a> &'a GC2: GroupOpsBounds<'a, C2, GC2>,
{
    pub fn run(self, cs: ConstraintSystemRef<CF1<C1>>) -> Result<Vec<CF1<C1>>, SynthesisError> {
        let t = cs.num_constraints();
        let timer = start_timer!(|| "Variable creation");
        let i = FpVar::<CF1<C1>>::new_witness(cs.clone(), || {
            Ok(self.i.unwrap_or_else(CF1::<C1>::zero))
        })?;
        let z_0 = Vec::<FpVar<CF1<C1>>>::new_witness(cs.clone(), || {
            Ok(self
                .z_0
                .unwrap_or(vec![CF1::<C1>::zero(); self.F.state_len()]))
        })?;
        let z_i = Vec::<FpVar<CF1<C1>>>::new_witness(cs.clone(), || {
            Ok(self
                .z_i
                .unwrap_or(vec![CF1::<C1>::zero(); self.F.state_len()]))
        })?;

        let u_dummy = RunningInstance::dummy(IO_LEN);
        let u_i_cmQ =
            NonNativeAffineVar::new_witness(cs.clone(), || Ok(self.u_i_cmQ.unwrap_or(C1::zero())))?;
        let u_i_cmW =
            NonNativeAffineVar::new_witness(cs.clone(), || Ok(self.u_i_cmW.unwrap_or(C1::zero())))?;
        let U_i = RunningInstanceVar::<C1>::new_witness(cs.clone(), || {
            Ok(self.U_i.unwrap_or(u_dummy.clone()))
        })?;
        let U_i1_cmE = NonNativeAffineVar::new_witness(cs.clone(), || {
            Ok(self.U_i1_cmE.unwrap_or_else(C1::zero))
        })?;
        let U_i1_cmQ = NonNativeAffineVar::new_witness(cs.clone(), || {
            Ok(self.U_i1_cmQ.unwrap_or_else(C1::zero))
        })?;
        let U_i1_cmW = NonNativeAffineVar::new_witness(cs.clone(), || {
            Ok(self.U_i1_cmW.unwrap_or_else(C1::zero))
        })?;

        let cmT =
            NonNativeAffineVar::new_witness(cs.clone(), || Ok(self.cmT.unwrap_or_else(C1::zero)))?;

        let cf_u_dummy = CycleFoldCommittedInstance::dummy(CF_IO_LEN);
        let cf_U_i = CycleFoldCommittedInstanceVar::<C2, GC2>::new_witness(cs.clone(), || {
            Ok(self.cf_U_i.unwrap_or(cf_u_dummy.clone()))
        })?;
        let cf1_cmT = GC2::new_witness(cs.clone(), || Ok(self.cf1_cmT.unwrap_or_else(C2::zero)))?;
        let cf2_cmT = GC2::new_witness(cs.clone(), || Ok(self.cf2_cmT.unwrap_or_else(C2::zero)))?;
        let cf3_cmT = GC2::new_witness(cs.clone(), || Ok(self.cf3_cmT.unwrap_or_else(C2::zero)))?;

        let crh_params = CRHParametersVar::<C1::ScalarField>::new_constant(
            cs.clone(),
            self.poseidon_config.clone(),
        )?;

        let i_usize = self.i_usize.unwrap_or(0);
        let is_basecase = i.is_zero()?;
        end_timer!(timer);
        add_to_trace!(|| "Variable creation", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        // get z_{i+1} from the F circuit
        let t = cs.num_constraints();
        let timer = start_timer!(|| "Step circuit");
        let z_i1 = self.F.generate_step_constraints(
            cs.clone(),
            self.la.clone(),
            i_usize,
            z_i.clone(),
            &self.external_inputs,
        )?;
        end_timer!(timer);
        add_to_trace!(|| "Step circuit", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        let t = cs.num_constraints();
        // Primary Part
        // P.1. Compute u_i.x
        // u_i.x[0] = H(i, z_0, z_i, U_i)
        let timer = start_timer!(|| "Fold primary instances");
        let (u_i_x, U_i_vec) =
            U_i.clone()
                .hash(&crh_params, i.clone(), z_0.clone(), z_i.clone())?;
        // u_i.x[1] = H(cf_U_i)
        let (cf_u_i_x, cf_U_i_vec) = cf_U_i.clone().hash(&crh_params)?;
        // u_i.x[2] = H(cmW)
        let logup_challenge = CRHGadget::evaluate(&crh_params, &u_i_cmQ.to_constraint_field()?)?;

        // P.2. Construct u_i
        let u_i = CurrentInstanceVar {
            // u_i.cmE = cm(0)
            cmE: NonNativeAffineVar::new_constant(cs.clone(), C1::zero())?,
            // u_i.u = 1
            u: FpVar::one(),
            // u_i.cmW is provided by the prover as witness
            cmQ: u_i_cmQ,
            cmW: u_i_cmW,
            // u_i.x is computed in step 1
            x: vec![u_i_x, cf_u_i_x, logup_challenge],
        };

        // P.3. nifs.verify, obtains U_{i+1} by folding u_i & U_i .

        // compute r = H(u_i, U_i, cmT)
        let r_bits = ChallengeGadget::<C1>::get_challenge_gadget(
            cs.clone(),
            &self.poseidon_config,
            U_i_vec,
            u_i.clone(),
            cmT.clone(),
        )?;
        let r = Boolean::le_bits_to_fp(&r_bits)?;
        // Also convert r_bits to a `NonNativeFieldVar`
        let r_nonnat = {
            let mut bits = r_bits;
            bits.resize(C1::BaseField::MODULUS_BIT_SIZE as usize, Boolean::FALSE);
            NonNativeUintVar::from(&bits)
        };

        // Notice that NIFSGadget::fold_committed_instance does not fold cmE & cmW.
        // We set `U_i1.cmE` and `U_i1.cmW` to unconstrained witnesses `U_i1_cmE` and `U_i1_cmW`
        // respectively.
        // The correctness of them will be checked on the other curve.
        let mut U_i1 = NIFSGadget::<C1>::fold_committed_instance(r, U_i.clone(), u_i.clone())?;
        U_i1.cmE = U_i1_cmE;
        U_i1.cmQ = U_i1_cmQ;
        U_i1.cmW = U_i1_cmW;
        end_timer!(timer);
        add_to_trace!(|| "Fold primary instances", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        let t = cs.num_constraints();
        let timer = start_timer!(|| "Fold CycleFold instances");
        // CycleFold part
        // C.1. Compute cf1_u_i.x and cf2_u_i.x
        let cfWa_x = vec![
            r_nonnat.clone(),
            U_i.cmQ.x,
            U_i.cmQ.y,
            u_i.cmQ.x,
            u_i.cmQ.y,
            U_i1.cmQ.x.clone(),
            U_i1.cmQ.y.clone(),
        ];
        let cfWb_x = vec![
            r_nonnat.clone(),
            U_i.cmW.x,
            U_i.cmW.y,
            u_i.cmW.x,
            u_i.cmW.y,
            U_i1.cmW.x.clone(),
            U_i1.cmW.y.clone(),
        ];
        let cfE_x = vec![
            r_nonnat,
            U_i.cmE.x,
            U_i.cmE.y,
            cmT.x,
            cmT.y,
            U_i1.cmE.x.clone(),
            U_i1.cmE.y.clone(),
        ];

        // ensure that cf1_u & cf2_u have as public inputs the cmW & cmE from main instances U_i,
        // u_i, U_i+1 coordinates of the commitments
        // C.2. Construct `cf1_u_i` and `cf2_u_i`
        let cf1_u_i = CycleFoldCommittedInstanceVar {
            // cf1_u_i.cmE = 0
            cmE: GC2::zero(),
            // cf1_u_i.u = 1
            u: NonNativeUintVar::new_constant(cs.clone(), C1::BaseField::one())?,
            // cf1_u_i.cmW is provided by the prover as witness
            cmW: GC2::new_witness(cs.clone(), || Ok(self.cf1_u_i_cmW.unwrap_or(C2::zero())))?,
            // cf1_u_i.x is computed in step 1
            x: cfWa_x,
        };
        let cf2_u_i = CycleFoldCommittedInstanceVar {
            // cf1_u_i.cmE = 0
            cmE: GC2::zero(),
            // cf1_u_i.u = 1
            u: NonNativeUintVar::new_constant(cs.clone(), C1::BaseField::one())?,
            // cf1_u_i.cmW is provided by the prover as witness
            cmW: GC2::new_witness(cs.clone(), || Ok(self.cf2_u_i_cmW.unwrap_or(C2::zero())))?,
            // cf1_u_i.x is computed in step 1
            x: cfWb_x,
        };
        let cf3_u_i = CycleFoldCommittedInstanceVar {
            // cf2_u_i.cmE = 0
            cmE: GC2::zero(),
            // cf2_u_i.u = 1
            u: NonNativeUintVar::new_constant(cs.clone(), C1::BaseField::one())?,
            // cf2_u_i.cmW is provided by the prover as witness
            cmW: GC2::new_witness(cs.clone(), || Ok(self.cf3_u_i_cmW.unwrap_or(C2::zero())))?,
            // cf2_u_i.x is computed in step 1
            x: cfE_x,
        };

        // C.3. nifs.verify, obtains cf1_U_{i+1} by folding cf1_u_i & cf_U_i, and then cf_U_{i+1}
        // by folding cf2_u_i & cf1_U_{i+1}.

        // compute cf1_r = H(cf1_u_i, cf_U_i, cf1_cmT)
        // cf_r_bits is denoted by rho* in the paper.
        let cf1_r_bits = CycleFoldChallengeGadget::<C2, GC2>::get_challenge_gadget(
            cs.clone(),
            &self.poseidon_config,
            cf_U_i_vec,
            cf1_u_i.clone(),
            cf1_cmT.clone(),
        )?;
        // Convert cf1_r_bits to a `NonNativeFieldVar`
        let cf1_r_nonnat = {
            let mut bits = cf1_r_bits.clone();
            bits.resize(C1::BaseField::MODULUS_BIT_SIZE as usize, Boolean::FALSE);
            NonNativeUintVar::from(&bits)
        };
        // Fold cf1_u_i & cf_U_i into cf1_U_{i+1}
        let cf1_U_i1 = NIFSFullGadget::<C2, GC2>::fold_committed_instance(
            cf1_r_bits,
            cf1_r_nonnat,
            cf1_cmT,
            cf_U_i,
            cf1_u_i,
        )?;

        // same for cf2_r:
        let cf2_r_bits = CycleFoldChallengeGadget::<C2, GC2>::get_challenge_gadget(
            cs.clone(),
            &self.poseidon_config,
            cf1_U_i1.to_constraint_field()?,
            cf2_u_i.clone(),
            cf2_cmT.clone(),
        )?;
        let cf2_r_nonnat = {
            let mut bits = cf2_r_bits.clone();
            bits.resize(C1::BaseField::MODULUS_BIT_SIZE as usize, Boolean::FALSE);
            NonNativeUintVar::from(&bits)
        };
        let cf2_U_i1 = NIFSFullGadget::<C2, GC2>::fold_committed_instance(
            cf2_r_bits,
            cf2_r_nonnat,
            cf2_cmT,
            cf1_U_i1, // the output from NIFS.V(cf1_r, cf_U, cfE_u)
            cf2_u_i,
        )?;

        // same for cf3_r:
        let cf3_r_bits = CycleFoldChallengeGadget::<C2, GC2>::get_challenge_gadget(
            cs.clone(),
            &self.poseidon_config,
            cf2_U_i1.to_constraint_field()?,
            cf3_u_i.clone(),
            cf3_cmT.clone(),
        )?;
        let cf3_r_nonnat = {
            let mut bits = cf3_r_bits.clone();
            bits.resize(C1::BaseField::MODULUS_BIT_SIZE as usize, Boolean::FALSE);
            NonNativeUintVar::from(&bits)
        };
        let cf3_U_i1 = NIFSFullGadget::<C2, GC2>::fold_committed_instance(
            cf3_r_bits,
            cf3_r_nonnat,
            cf3_cmT,
            cf2_U_i1, // the output from NIFS.V(cf1_r, cf_U, cfE_u)
            cf3_u_i,
        )?;
        end_timer!(timer);
        add_to_trace!(|| "Fold CycleFold instances", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        // Back to Primary Part

        let t = cs.num_constraints();
        let timer = start_timer!(|| "Check public inputs");
        // P.4.a compute and check the first output of F'
        // Base case: u_{i+1}.x[0] == H((i+1, z_0, z_{i+1}, U_{\bot})
        // Non-base case: u_{i+1}.x[0] == H((i+1, z_0, z_{i+1}, U_{i+1})
        let (u_i1_x, _) = U_i1.hash(
            &crh_params,
            i + FpVar::<CF1<C1>>::one(),
            z_0.clone(),
            z_i1.clone(),
        )?;
        let (u_i1_x_base, _) = RunningInstanceVar::new_constant(cs.clone(), u_dummy)?.hash(
            &crh_params,
            FpVar::<CF1<C1>>::one(),
            z_0.clone(),
            z_i1.clone(),
        )?;
        let x = FpVar::new_input(cs.clone(), || {
            if i_usize == 0 {
                u_i1_x_base.value()
            } else {
                u_i1_x.value()
            }
        })?;
        x.enforce_equal(&is_basecase.select(&u_i1_x_base, &u_i1_x)?)?;

        // P.4.b compute and check the second output of F'
        // Base case: u_{i+1}.x[1] == H(cf_U_{\bot})
        // Non-base case: u_{i+1}.x[1] == H(cf_U_{i+1})
        let (cf_u_i1_x, _) = cf3_U_i1.clone().hash(&crh_params)?;
        let (cf_u_i1_x_base, _) =
            CycleFoldCommittedInstanceVar::<C2, GC2>::new_constant(cs.clone(), cf_u_dummy)?
                .hash(&crh_params)?;
        let cf_x = FpVar::new_input(cs.clone(), || {
            if i_usize == 0 {
                cf_u_i1_x_base.value()
            } else {
                cf_u_i1_x.value()
            }
        })?;
        cf_x.enforce_equal(&is_basecase.select(&cf_u_i1_x_base, &cf_u_i1_x)?)?;
        end_timer!(timer);
        add_to_trace!(|| "Check public inputs", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        z_i1.value()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_ff::BigInteger;
    use ark_grumpkin::{Fr, Projective};
    use ark_relations::r1cs::ConstraintSystem;
    use ark_std::UniformRand;

    use crate::commitment::pedersen::Pedersen;
    use crate::folding::nova::nifs::tests::prepare_simple_fold_inputs;
    use crate::folding::nova::nifs::NIFS;
    use crate::transcript::poseidon::poseidon_test_config;

    #[test]
    fn test_committed_instance_var() {
        let mut rng = ark_std::test_rng();

        let ci = RunningInstance::<Projective> {
            cmE: Projective::rand(&mut rng),
            u: Fr::rand(&mut rng),
            cmQ: Projective::rand(&mut rng),
            cmW: Projective::rand(&mut rng),
            x: vec![Fr::rand(&mut rng); 1],
        };

        let cs = ConstraintSystem::<Fr>::new_ref();
        let ciVar =
            RunningInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(ci.clone())).unwrap();
        assert_eq!(ciVar.u.value().unwrap(), ci.u);
        assert_eq!(ciVar.x.value().unwrap(), ci.x);
        // the values cmE and cmW are checked in the CycleFold's circuit
        // CommittedInstanceInCycleFoldVar in
        // nova::cyclefold::tests::test_committed_instance_cyclefold_var
    }

    #[test]
    fn test_nifs_gadget() {
        let (_, _, _, _, ci1, _, ci2, _, ci3, _, _, cmT, _, r_Fr) = prepare_simple_fold_inputs();

        let ci3_verifier = NIFS::<Projective, Pedersen<Projective>>::verify(r_Fr, &ci1, &ci2, &cmT);
        assert_eq!(ci3_verifier, ci3);

        let cs = ConstraintSystem::<Fr>::new_ref();

        let rVar = FpVar::<Fr>::new_witness(cs.clone(), || Ok(r_Fr)).unwrap();
        let ci1Var =
            RunningInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(ci1.clone())).unwrap();
        let ci2Var =
            CurrentInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(ci2.clone())).unwrap();
        let ci3Var =
            RunningInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(ci3.clone())).unwrap();

        NIFSGadget::<Projective>::verify(
            rVar.clone(),
            ci1Var.clone(),
            ci2Var.clone(),
            ci3Var.clone(),
        )
        .unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_committed_instance_hash() {
        let mut rng = ark_std::test_rng();
        let poseidon_config = poseidon_test_config::<Fr>();

        let i = Fr::from(3_u32);
        let z_0 = vec![Fr::from(3_u32)];
        let z_i = vec![Fr::from(3_u32)];
        let ci = RunningInstance::<Projective> {
            cmE: Projective::rand(&mut rng),
            u: Fr::rand(&mut rng),
            cmQ: Projective::rand(&mut rng),
            cmW: Projective::rand(&mut rng),
            x: vec![Fr::rand(&mut rng); 1],
        };

        // compute the CommittedInstance hash natively
        let h = ci
            .hash(&poseidon_config, i, z_0.clone(), z_i.clone())
            .unwrap();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let iVar = FpVar::<Fr>::new_witness(cs.clone(), || Ok(i)).unwrap();
        let z_0Var = Vec::<FpVar<Fr>>::new_witness(cs.clone(), || Ok(z_0.clone())).unwrap();
        let z_iVar = Vec::<FpVar<Fr>>::new_witness(cs.clone(), || Ok(z_i.clone())).unwrap();
        let ciVar =
            RunningInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(ci.clone())).unwrap();

        let crh_params = CRHParametersVar::<Fr>::new_constant(cs.clone(), poseidon_config).unwrap();

        // compute the CommittedInstance hash in-circuit
        let (hVar, _) = ciVar.hash(&crh_params, iVar, z_0Var, z_iVar).unwrap();
        assert!(cs.is_satisfied().unwrap());

        // check that the natively computed and in-circuit computed hashes match
        assert_eq!(hVar.value().unwrap(), h);
    }

    // checks that the gadget and native implementations of the challenge computation match
    #[test]
    fn test_challenge_gadget() {
        let mut rng = ark_std::test_rng();
        let poseidon_config = poseidon_test_config::<Fr>();

        let u_i = CurrentInstance::<Projective> {
            cmE: Projective::rand(&mut rng),
            u: Fr::rand(&mut rng),
            cmQ: Projective::rand(&mut rng),
            cmW: Projective::rand(&mut rng),
            x: vec![Fr::rand(&mut rng); 1],
        };
        let U_i = RunningInstance::<Projective> {
            cmE: Projective::rand(&mut rng),
            u: Fr::rand(&mut rng),
            cmQ: Projective::rand(&mut rng),
            cmW: Projective::rand(&mut rng),
            x: vec![Fr::rand(&mut rng); 1],
        };
        let cmT = Projective::rand(&mut rng);

        // compute the challenge natively
        let r_bits =
            ChallengeGadget::<Projective>::get_challenge_native(&poseidon_config, &U_i, &u_i, cmT)
                .unwrap();
        let r = Fr::from_bigint(BigInteger::from_bits_le(&r_bits)).unwrap();

        let cs = ConstraintSystem::<Fr>::new_ref();
        let u_iVar =
            CurrentInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(u_i.clone())).unwrap();
        let U_iVar =
            RunningInstanceVar::<Projective>::new_witness(cs.clone(), || Ok(U_i.clone())).unwrap();
        let cmTVar = NonNativeAffineVar::<Projective>::new_witness(cs.clone(), || Ok(cmT)).unwrap();

        // compute the challenge in-circuit
        let U_iVar_vec = [
            vec![U_iVar.u.clone()],
            U_iVar.x.clone(),
            U_iVar.cmE.to_constraint_field().unwrap(),
            U_iVar.cmQ.to_constraint_field().unwrap(),
            U_iVar.cmW.to_constraint_field().unwrap(),
        ]
        .concat();
        let r_bitsVar = ChallengeGadget::<Projective>::get_challenge_gadget(
            cs.clone(),
            &poseidon_config,
            U_iVar_vec,
            u_iVar,
            cmTVar,
        )
        .unwrap();
        assert!(cs.is_satisfied().unwrap());

        // check that the natively computed and in-circuit computed hashes match
        let rVar = Boolean::le_bits_to_fp(&r_bitsVar).unwrap();
        assert_eq!(rVar.value().unwrap(), r);
        assert_eq!(r_bitsVar.value().unwrap(), r_bits);
    }
}
