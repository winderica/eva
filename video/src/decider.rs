/// This file implements the onchain (Ethereum's EVM) decider.
use ark_bn254::Bn254;
use ark_crypto_primitives::crh::poseidon::constraints::CRHGadget;
use ark_crypto_primitives::crh::CRHSchemeGadget;
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField, ToConstraintField};
use ark_r1cs_std::{
    convert::ToConstraintFieldGadget, groups::GroupOpsBounds, prelude::CurveVar, R1CSVar as _,
};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_snark::SNARK;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::{add_to_trace, end_timer, start_timer, One, Zero};
use core::marker::PhantomData;

use folding_schemes::commitment::pedersen::{Pedersen, PedersenGadget};
use folding_schemes::commitment::{pedersen::Params as PedersenParams, CommitmentScheme};
use folding_schemes::folding::nova::cyclefold::{CycleFoldCommittedInstanceVar, CF_IO_LEN};
use folding_schemes::folding::nova::{ProverParams, VerifierParams};
use folding_schemes::folding::{
    circuits::nonnative::affine::NonNativeAffineVar,
    nova::{
        circuits::{CF2, IO_LEN},
        nifs::NIFS,
        CurrentInstance, CycleFoldCommittedInstance, Nova, RunningInstance,
    },
};
use folding_schemes::frontend::FCircuit;
use folding_schemes::{Decider as DeciderTrait, FoldingScheme, MVM};
use folding_schemes::{Error, MSM};

/// This file implements the onchain (Ethereum's EVM) decider circuit. For non-ethereum use cases,
/// other more efficient approaches can be used.
use ark_crypto_primitives::crh::poseidon::constraints::CRHParametersVar;
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_poly::Polynomial;
use ark_r1cs_std::{
    alloc::{AllocVar, AllocationMode},
    boolean::Boolean,
    convert::ToBitsGadget,
    eq::EqGadget,
    fields::{fp::FpVar, FieldVar},
    poly::{domain::Radix2DomainVar, evaluations::univariate::EvaluationsVar},
};

use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::log2;
use core::borrow::Borrow;

use folding_schemes::ccs::r1cs::R1CS;
use folding_schemes::folding::circuits::nonnative::uint::LimbVar;
use folding_schemes::folding::circuits::nonnative::{
    affine::nonnative_affine_to_field_elements, uint::NonNativeUintVar,
};
use folding_schemes::folding::nova::circuits::ChallengeGadget;
use folding_schemes::folding::nova::circuits::NIFSGadget;
use folding_schemes::folding::nova::{
    circuits::{CurrentInstanceVar, RunningInstanceVar, CF1},
    Witness,
};
use folding_schemes::transcript::{
    poseidon::{PoseidonTranscript, PoseidonTranscriptVar},
    Transcript, TranscriptVar,
};
use folding_schemes::utils::{
    gadgets::{MatrixGadget, SparseMatrixVar, VectorGadget},
    vec::poly_from_vec,
};

use num_bigint::BigUint;

#[derive(Debug, Clone)]
pub struct RelaxedR1CSGadget {}
impl RelaxedR1CSGadget {
    /// performs the RelaxedR1CS check for native variables (Az∘Bz==uCz+E)
    pub fn check_native<F: PrimeField>(
        r1cs: R1CSVar<F, F, FpVar<F>>,
        E: Vec<FpVar<F>>,
        u: FpVar<F>,
        z: Vec<FpVar<F>>,
    ) -> Result<(), SynthesisError> {
        let Az = r1cs.A.mul_vector(&z)?;
        let Bz = r1cs.B.mul_vector(&z)?;
        let Cz = r1cs.C.mul_vector(&z)?;
        let uCzE = Cz.mul_scalar(&u)?.add(&E)?;
        Az.iter()
            .zip(&Bz)
            .zip(uCzE)
            .try_for_each(|((a, b), c)| a.mul_equals(&b, &c))
    }

    /// performs the RelaxedR1CS check for non-native variables (Az∘Bz==uCz+E)
    pub fn check_nonnative<F: PrimeField, CF: PrimeField>(
        r1cs: R1CSVar<F, CF, NonNativeUintVar<CF>>,
        E: Vec<NonNativeUintVar<CF>>,
        u: NonNativeUintVar<CF>,
        z: Vec<NonNativeUintVar<CF>>,
    ) -> Result<(), SynthesisError> {
        // First we do addition and multiplication without mod F's order
        let Az = r1cs.A.mul_vector(&z)?;
        let Bz = r1cs.B.mul_vector(&z)?;
        let Cz = r1cs.C.mul_vector(&z)?;
        let uCzE = Cz.mul_scalar(&u)?.add(&E)?;
        let AzBz = Az.hadamard(&Bz)?;

        // Then we compare the results by checking if they are congruent
        // modulo the field order
        AzBz.into_iter()
            .zip(uCzE)
            .try_for_each(|(a, b)| a.enforce_congruent::<F>(&b))
    }
}

#[derive(Debug, Clone)]
pub struct R1CSVar<F: PrimeField, CF: PrimeField, FV: AllocVar<F, CF>> {
    _f: PhantomData<F>,
    _cf: PhantomData<CF>,
    _fv: PhantomData<FV>,
    pub A: SparseMatrixVar<F, CF, FV>,
    pub B: SparseMatrixVar<F, CF, FV>,
    pub C: SparseMatrixVar<F, CF, FV>,
}

impl<F, CF, FV> AllocVar<R1CS<F>, CF> for R1CSVar<F, CF, FV>
where
    F: PrimeField,
    CF: PrimeField,
    FV: AllocVar<F, CF>,
{
    fn new_variable<T: Borrow<R1CS<F>>>(
        cs: impl Into<Namespace<CF>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        _mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        f().and_then(|val| {
            let cs = cs.into();

            let A = SparseMatrixVar::<F, CF, FV>::new_constant(cs.clone(), &val.borrow().A)?;
            let B = SparseMatrixVar::<F, CF, FV>::new_constant(cs.clone(), &val.borrow().B)?;
            let C = SparseMatrixVar::<F, CF, FV>::new_constant(cs.clone(), &val.borrow().C)?;

            Ok(Self {
                _f: PhantomData,
                _cf: PhantomData,
                _fv: PhantomData,
                A,
                B,
                C,
            })
        })
    }
}

/// In-circuit representation of the Witness associated to the CommittedInstance.
#[derive(Debug, Clone)]
pub struct WitnessVar<C: CurveGroup> {
    pub E: Vec<FpVar<C::ScalarField>>,
    pub rE: FpVar<C::ScalarField>,
    pub Q: Vec<FpVar<C::ScalarField>>,
    pub rQ: FpVar<C::ScalarField>,
    pub W: Vec<FpVar<C::ScalarField>>,
    pub rW: FpVar<C::ScalarField>,
}

impl<C> AllocVar<Witness<C>, CF1<C>> for WitnessVar<C>
where
    C: CurveGroup,
    <C as ark_ec::CurveGroup>::BaseField: PrimeField,
{
    fn new_variable<T: Borrow<Witness<C>>>(
        cs: impl Into<Namespace<CF1<C>>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        f().and_then(|val| {
            let cs = cs.into();

            let E: Vec<FpVar<C::ScalarField>> =
                Vec::new_variable(cs.clone(), || Ok(val.borrow().E.clone()), mode)?;
            let rE =
                FpVar::<C::ScalarField>::new_variable(cs.clone(), || Ok(val.borrow().rE), mode)?;

            let Q: Vec<FpVar<C::ScalarField>> =
                Vec::new_variable(cs.clone(), || Ok(val.borrow().q().to_vec()), mode)?;
            let rQ =
                FpVar::<C::ScalarField>::new_variable(cs.clone(), || Ok(val.borrow().rQ), mode)?;

            let W: Vec<FpVar<C::ScalarField>> =
                Vec::new_variable(cs.clone(), || Ok(val.borrow().w().to_vec()), mode)?;
            let rW =
                FpVar::<C::ScalarField>::new_variable(cs.clone(), || Ok(val.borrow().rW), mode)?;

            Ok(Self {
                E,
                rE,
                Q,
                rQ,
                W,
                rW,
            })
        })
    }
}

/// In-circuit representation of the Witness associated to the CommittedInstance, but with
/// non-native representation, since it is used to represent the CycleFold witness.
#[derive(Debug, Clone)]
pub struct CycleFoldWitnessVar<C: CurveGroup> {
    pub E: Vec<NonNativeUintVar<CF2<C>>>,
    pub rE: NonNativeUintVar<CF2<C>>,
    pub W: Vec<NonNativeUintVar<CF2<C>>>,
    pub rW: NonNativeUintVar<CF2<C>>,
}

impl<C> AllocVar<Witness<C>, CF2<C>> for CycleFoldWitnessVar<C>
where
    C: CurveGroup,
    <C as ark_ec::CurveGroup>::BaseField: PrimeField,
{
    fn new_variable<T: Borrow<Witness<C>>>(
        cs: impl Into<Namespace<CF2<C>>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        f().and_then(|val| {
            let cs = cs.into();

            let E = Vec::new_variable(cs.clone(), || Ok(val.borrow().E.clone()), mode)?;
            let rE = NonNativeUintVar::new_variable(cs.clone(), || Ok(val.borrow().rE), mode)?;

            assert!(val.borrow().q().is_empty());

            let W = Vec::new_variable(cs.clone(), || Ok(val.borrow().QW.clone()), mode)?;
            let rW = NonNativeUintVar::new_variable(cs.clone(), || Ok(val.borrow().rW), mode)?;

            Ok(Self { E, rE, W, rW })
        })
    }
}

/// Circuit that implements the in-circuit checks needed for the onchain (Ethereum's EVM)
/// verification.
#[derive(Clone)]
pub struct DeciderEthCircuit<C1, GC1, C2, GC2>
where
    C1: CurveGroup,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    GC1: CurveVar<C1, CF2<C1>>,
    GC2: CurveVar<C2, CF2<C2>>,
{
    pub _gc1: PhantomData<GC1>,
    pub _gc2: PhantomData<GC2>,

    /// R1CS of the Augmented Function circuit
    pub r1cs: R1CS<C1::ScalarField>,
    /// R1CS of the CycleFold circuit
    pub cf_r1cs: R1CS<C2::ScalarField>,
    /// CycleFold PedersenParams over C2
    pub cf_pedersen_params: PedersenParams<C2>,
    pub poseidon_config: PoseidonConfig<CF1<C1>>,
    pub i: Option<CF1<C1>>,
    /// initial state
    pub z_0: Option<Vec<C1::ScalarField>>,
    /// Nova instances
    pub u_i: Option<CurrentInstance<C1>>,
    pub U_i: Option<RunningInstance<C1>>,
    pub W_i1: Option<Witness<C1>>,
    pub cmT: Option<C1>,
    pub r: Option<C1::ScalarField>,
    /// CycleFold running instance
    pub cf_U_i: Option<CycleFoldCommittedInstance<C2>>,
    pub cf_W_i: Option<Witness<C2>>,

    pub sigma: (C1::ScalarField, C2::ScalarField),
    pub vk: C2,
    pub h1: C1::ScalarField,
    pub h2: C1::ScalarField,
}
impl<C1, GC1, C2, GC2> DeciderEthCircuit<C1, GC1, C2, GC2>
where
    C1: CurveGroup,
    C1::Config: MSM<C1>,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    GC1: CurveVar<C1, CF2<C1>> + ToConstraintFieldGadget<CF2<C1>>,
    GC2: CurveVar<C2, CF2<C2>> + ToConstraintFieldGadget<CF2<C2>>,
    C1::ScalarField: Absorb + MVM,
    <C1 as CurveGroup>::BaseField: PrimeField,
{
    pub fn from_nova<FC: FCircuit<C1::ScalarField>, CS1, CS2>(
        nova: &Nova<C1, GC1, C2, GC2, FC, CS1, CS2>,
        (pp, vp): (ProverParams<C1, C2, CS1, CS2>, VerifierParams<C1, C2>),
        vk: C2,
        sigma: (C1::ScalarField, C2::ScalarField),
    ) -> Result<Self, Error>
    where
        CS1: CommitmentScheme<C1>,
        // enforce that the CS2 is Pedersen commitment scheme, since we're at Ethereum's EVM decider
        CS2: CommitmentScheme<C2, ProverParams = PedersenParams<C2>>,
    {
        // compute the U_{i+1}, W_{i+1}
        let (T, cmT) = NIFS::<C1, CS1>::compute_cmT(
            &pp.cs_params,
            &vp.r1cs,
            &nova.W_i.clone(),
            &nova.U_i.clone(),
            &nova.w_i.clone(),
            &nova.u_i.clone(),
        )?;
        let r_bits = ChallengeGadget::<C1>::get_challenge_native(
            &nova.poseidon_config,
            &nova.U_i,
            &nova.u_i,
            cmT,
        )?;
        let r_Fr = C1::ScalarField::from_bigint(BigInteger::from_bits_le(&r_bits))
            .ok_or(Error::OutOfBounds)?;
        let W_i1 =
            NIFS::<C1, CS1>::fold_witness(r_Fr, &nova.W_i, &nova.w_i, &T, C1::ScalarField::zero())?;

        Ok(Self {
            _gc1: PhantomData,
            _gc2: PhantomData,

            r1cs: vp.r1cs,
            cf_r1cs: vp.cf_r1cs,
            cf_pedersen_params: pp.cf_cs_params,
            poseidon_config: nova.poseidon_config.clone(),
            i: Some(nova.i),
            z_0: Some(nova.z_0.clone()),
            u_i: Some(nova.u_i.clone()),
            U_i: Some(nova.U_i.clone()),
            W_i1: Some(W_i1),
            cmT: Some(cmT),
            r: Some(r_Fr),
            cf_U_i: Some(nova.cf_U_i.clone()),
            cf_W_i: Some(nova.cf_W_i.clone()),
            vk,
            sigma,
            h1: nova.z_i[0],
            h2: nova.z_i[1],
        })
    }
}

impl<C1, GC1, C2, GC2> ConstraintSynthesizer<CF1<C1>> for DeciderEthCircuit<C1, GC1, C2, GC2>
where
    C1: CurveGroup,
    C1::Config: MSM<C1>,
    C2: CurveGroup,
    C2::Config: MSM<C2>,
    GC1: CurveVar<C1, CF2<C1>>,
    GC2: CurveVar<C2, CF2<C2>> + ToConstraintFieldGadget<CF2<C2>>,
    <C2 as CurveGroup>::BaseField: PrimeField,
    C1::ScalarField: Absorb,
    C1: CurveGroup<BaseField = C2::ScalarField, ScalarField = C2::BaseField>,
{
    fn generate_constraints(self, cs: ConstraintSystemRef<CF1<C1>>) -> Result<(), SynthesisError> {
        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let crh_params = CRHParametersVar::<C1::ScalarField>::new_constant(
            cs.clone(),
            self.poseidon_config.clone(),
        )?;
        let r1cs =
            R1CSVar::<C1::ScalarField, CF1<C1>, FpVar<CF1<C1>>>::new_witness(cs.clone(), || {
                Ok(self.r1cs.clone())
            })?;
        let cf_r1cs = R1CSVar::<C1::BaseField, CF1<C1>, NonNativeUintVar<CF1<C1>>>::new_witness(
            cs.clone(),
            || Ok(self.cf_r1cs.clone()),
        )?;

        let i =
            FpVar::<CF1<C1>>::new_input(cs.clone(), || Ok(self.i.unwrap_or_else(CF1::<C1>::zero)))?;
        let z_0 = Vec::<FpVar<CF1<C1>>>::new_input(cs.clone(), || {
            Ok(self.z_0.unwrap_or(vec![CF1::<C1>::zero()]))
        })?;
        let z_i = vec![
            FpVar::new_witness(cs.clone(), || Ok(self.h1))?,
            FpVar::new_input(cs.clone(), || Ok(self.h2))?,
        ];

        let g = GC2::new_constant(cs.clone(), C2::generator())?;
        let p = GC2::new_input(cs.clone(), || Ok(self.vk))?;
        let (px, py) = {
            let v = p.to_constraint_field()?;
            (v[0].clone(), v[1].clone())
        };
        let sigma_r = FpVar::new_witness(cs.clone(), || Ok(self.sigma.0))?;
        let sigma_s = Vec::<Boolean<_>>::new_witness(cs.clone(), || {
            Ok({
                let mut s = self.sigma.1.into_bigint().to_bits_le();
                s.resize(C2::ScalarField::MODULUS_BIT_SIZE as usize, false);
                s
            })
        })?;

        let U_i = self.U_i.unwrap_or(RunningInstance::dummy(IO_LEN));
        let U_i = RunningInstanceVar {
            u: FpVar::new_input(cs.clone(), || Ok(U_i.u))?,
            x: Vec::new_witness(cs.clone(), || Ok(U_i.x))?,
            cmQ: NonNativeAffineVar::new_input(cs.clone(), || Ok(U_i.cmQ))?,
            cmW: NonNativeAffineVar::new_input(cs.clone(), || Ok(U_i.cmW))?,
            cmE: NonNativeAffineVar::new_input(cs.clone(), || Ok(U_i.cmE))?,
        };
        let cf_U_i = CycleFoldCommittedInstanceVar::<C2, GC2>::new_witness(cs.clone(), || {
            Ok(self
                .cf_U_i
                .unwrap_or_else(|| CycleFoldCommittedInstance::<C2>::dummy(CF_IO_LEN)))
        })?;

        let u_i = self.u_i.unwrap_or(CurrentInstance::dummy(IO_LEN));
        let u_i_cmQ = NonNativeAffineVar::new_input(cs.clone(), || Ok(u_i.cmQ))?;
        let u_i_cmW = NonNativeAffineVar::new_input(cs.clone(), || Ok(u_i.cmW))?;

        let cmT =
            NonNativeAffineVar::new_input(cs.clone(), || Ok(self.cmT.unwrap_or_else(C1::zero)))?;
        let r = FpVar::new_input(cs.clone(), || Ok(self.r.unwrap_or_else(CF1::<C1>::zero)))?;

        let W_i1 = self.W_i1.unwrap_or(Witness::<C1>::new(
            vec![C1::ScalarField::zero(); self.r1cs.A.n_cols - IO_LEN - 1],
            &self.r1cs,
        ));
        let W_i1_QW = Vec::new_committed(cs.clone(), || Ok(W_i1.QW))?;
        let W_i1_E = Vec::new_committed(cs.clone(), || Ok(W_i1.E))?;

        let cf_W_i = self.cf_W_i.unwrap_or(Witness::<C2>::new(
            vec![C2::ScalarField::zero(); self.cf_r1cs.A.n_cols - 1 - self.cf_r1cs.l],
            &self.cf_r1cs,
        ));
        let E_bits = cf_W_i
            .E
            .iter()
            .map(|i| Vec::new_witness(cs.clone(), || Ok(i.into_bigint().to_bits_le())))
            .collect::<Result<Vec<_>, _>>()?;
        let W_bits = cf_W_i
            .QW
            .iter()
            .map(|i| Vec::new_witness(cs.clone(), || Ok(i.into_bigint().to_bits_le())))
            .collect::<Result<Vec<_>, _>>()?;

        let G = Vec::<GC2>::new_constant(cs.clone(), self.cf_pedersen_params.generators)?;
        let H = GC2::new_constant(cs.clone(), self.cf_pedersen_params.h)?;
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Create variables", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        {
            let e = CRHGadget::evaluate(&crh_params, &[sigma_r.clone(), px, py, z_i[0].clone()])?;
            let e = e.to_bits_le()?;
            (g.scalar_mul_le(sigma_s.iter())? - p.scalar_mul_le(e.iter())?)
                .to_constraint_field()?[0]
                .enforce_equal(&sigma_r)?;
        };
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Verify signature", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let (U_i, cf_U_i, u_i, U_i_vec) = {
            let (u_i_x, U_i_vec) = U_i.clone().hash(&crh_params, i, z_0, z_i)?;
            let (cf_u_i_x, _) = cf_U_i.clone().hash(&crh_params)?;

            let mut u_i = CurrentInstanceVar {
                u: FpVar::one(),
                x: vec![u_i_x, cf_u_i_x],
                cmQ: u_i_cmQ,
                cmW: u_i_cmW,
                cmE: NonNativeAffineVar::new_constant(cs.clone(), C1::zero())?,
            };
            u_i.x.push(CRHGadget::evaluate(
                &crh_params,
                &u_i.cmQ.to_constraint_field()?,
            )?);
            (U_i, cf_U_i, u_i, U_i_vec)
        };
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Reconstruct instances", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        Boolean::le_bits_to_fp(&ChallengeGadget::<C1>::get_challenge_gadget(
            cs.clone(),
            &self.poseidon_config,
            U_i_vec,
            u_i.clone(),
            cmT.clone(),
        )?)?
        .enforce_equal(&r)?;
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Check challenge", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let U_i1 = NIFSGadget::<C1>::fold_committed_instance(r, U_i.clone(), u_i.clone())?;
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Fold primary instances", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        {
            #[cfg(feature = "constraints")]
            let t = cs.num_constraints();
            let z = [&vec![U_i1.u.clone()][..], &U_i1.x, &W_i1_QW].concat();
            RelaxedR1CSGadget::check_native(r1cs, W_i1_E, U_i1.u, z)?;
            #[cfg(feature = "constraints")]
            add_to_trace!(|| "Check relaxed R1CS satisfiability", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        };
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Check primary instances", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        {
            #[cfg(feature = "constraints")]
            let t = cs.num_constraints();

            RelaxedR1CSGadget::check_nonnative(
                cf_r1cs,
                E_bits
                    .iter()
                    .map(|i| {
                        Ok(NonNativeUintVar(
                            i.chunks(NonNativeUintVar::<C2::ScalarField>::bits_per_limb())
                                .map(|i| {
                                    Ok(LimbVar {
                                        ub: (BigUint::one() << i.len()) - BigUint::one(),
                                        v: {
                                            let v = Boolean::le_bits_to_fp(i)?;
                                            let x = FpVar::new_witness(cs.clone(), || v.value())?;
                                            x.enforce_equal(&v)?;
                                            x
                                        },
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                cf_U_i.u.clone(),
                [
                    &vec![cf_U_i.u][..],
                    &cf_U_i.x,
                    &W_bits
                        .iter()
                        .map(|i| {
                            Ok(NonNativeUintVar(
                                i.chunks(NonNativeUintVar::<C2::ScalarField>::bits_per_limb())
                                    .map(|i| {
                                        Ok(LimbVar {
                                            ub: (BigUint::one() << i.len()) - BigUint::one(),
                                            v: {
                                                let v = Boolean::le_bits_to_fp(i)?;
                                                let x =
                                                    FpVar::new_witness(cs.clone(), || v.value())?;
                                                x.enforce_equal(&v)?;
                                                x
                                            },
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ]
                .concat(),
            )?;
            #[cfg(feature = "constraints")]
            add_to_trace!(|| "Check relaxed R1CS satisfiability", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));

            #[cfg(feature = "constraints")]
            let t = cs.num_constraints();
            PedersenGadget::<C2, GC2>::commit(
                H.clone(),
                G.clone(),
                E_bits,
                NonNativeUintVar::new_witness(cs.clone(), || Ok(cf_W_i.rE))?.to_bits_le()?,
            )?
            .enforce_equal(&cf_U_i.cmE)?;
            PedersenGadget::<C2, GC2>::commit(
                H,
                G,
                W_bits,
                NonNativeUintVar::new_witness(cs.clone(), || Ok(cf_W_i.rW))?.to_bits_le()?,
            )?
            .enforce_equal(&cf_U_i.cmW)?;
            #[cfg(feature = "constraints")]
            add_to_trace!(|| "Check commitments", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Check CycleFold instances", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        let cs = cs.borrow().unwrap();
        add_to_trace!(|| "R1CS", || format!(
            "{} constraints, {} variables (q: {}, w: {}, x: {}, u: 1)",
            cs.num_constraints,
            cs.num_committed_variables + cs.num_witness_variables + cs.num_instance_variables,
            cs.num_committed_variables,
            cs.num_witness_variables,
            cs.num_instance_variables - 1,
        ));

        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Proof<C1, S>
where
    C1: CurveGroup,
    S: SNARK<C1::ScalarField>,
{
    pub snark_proof: S::Proof,
    // cmT and r are values for the last fold, U_{i+1}=NIFS.V(r, U_i, u_i, cmT), and they are
    // checked in-circuit
    pub cmT: C1,
    pub r: C1::ScalarField,
}

/// Onchain Decider, for ethereum use cases
#[derive(Clone, Debug)]
pub struct Decider {}

impl Decider {
    pub fn prove<
        E: Pairing,
        GC1,
        GC2,
        C2: CurveGroup<BaseField = E::ScalarField, ScalarField = <E::G1 as CurveGroup>::BaseField>,
    >(
        pp: ark_groth16::ProvingKey<E>,
        mut rng: impl RngCore + CryptoRng,
        circuit: DeciderEthCircuit<E::G1, GC1, C2, GC2>,
    ) -> Result<(ark_groth16::Proof<E>, E::G1, E::ScalarField), Error>
    where
        GC1: CurveVar<E::G1, CF2<E::G1>> + ToConstraintFieldGadget<CF2<E::G1>>,
        GC2: CurveVar<C2, CF2<C2>> + ToConstraintFieldGadget<CF2<C2>>,
        C2::Config: MSM<C2>,
        <E::G1 as CurveGroup>::Config: MSM<E::G1>,
        C2::Config: MSM<C2>,
        E::ScalarField: Absorb,
    {
        let prove_timer = start_timer!(|| "Decider::Prove");
        let snark_pk = pp;

        let cmT = circuit.cmT.unwrap();
        let r = circuit.r.unwrap();

        let snark_proof = ark_groth16::Groth16::<E>::prove(&snark_pk, circuit, &mut rng)
            .map_err(|e| Error::Other(e.to_string()))?;

        end_timer!(prove_timer);

        add_to_trace!(|| "Proof size", || format!("{}", {
            snark_proof.0.serialized_size(Compress::Yes)
            + cmT.serialized_size(Compress::Yes)
            + r.serialized_size(Compress::Yes)
            + CF1::<E::G1>::zero().serialized_size(Compress::Yes) // U.u
            + E::G1::zero().serialized_size(Compress::Yes) // U.cmQ
            + E::G1::zero().serialized_size(Compress::Yes) // U.cmW
            + E::G1::zero().serialized_size(Compress::Yes) // U.cmE
            + E::G1::zero().serialized_size(Compress::Yes) // u.cmQ
            + E::G1::zero().serialized_size(Compress::Yes) // u.cmW
        }));

        Ok((snark_proof.0, cmT, r))
    }

    pub fn verify<
        E: Pairing,
        C2: CurveGroup<BaseField = E::ScalarField> + ToConstraintField<E::ScalarField>,
    >(
        vp: ark_groth16::PreparedVerifyingKey<E>,
        vk: C2,
        i: E::ScalarField,
        z_0: Vec<E::ScalarField>,
        h2: E::ScalarField,
        U: &RunningInstance<E::G1>,
        u: &CurrentInstance<E::G1>,
        (proof, cmT, r): (ark_groth16::Proof<E>, E::G1, E::ScalarField),
    ) -> Result<bool, Error>
    where
        E::BaseField: PrimeField,
        <E::G1 as CurveGroup>::Config: MSM<E::G1>,
        E::ScalarField: Absorb + MVM,
    {
        // if i <= C1::ScalarField::one() {
        //     return Err(Error::NotEnoughSteps);
        // }

        let verify_timer = start_timer!(|| "Decider::Verify");

        let nifs_timer = start_timer!(|| "Folding verification");
        let UU = NIFS::<E::G1, Pedersen<E::G1, false>>::fold_committed_instance(r, U, u, &cmT);
        end_timer!(nifs_timer);

        let snark_vk = vp;

        let inputs_timer = start_timer!(|| "Prepare inputs");
        let public_inputs = vec![
            vec![i],
            z_0,
            vec![h2],
            if vk.is_zero() {
                vec![Zero::zero(), One::one(), Zero::zero()]
            } else {
                let (x, y) = vk.into_affine().xy().unwrap();
                vec![x, y, One::one()]
            },
            vec![U.u],
            NonNativeAffineVar::inputize(U.cmQ)?,
            NonNativeAffineVar::inputize(U.cmW)?,
            NonNativeAffineVar::inputize(U.cmE)?,
            NonNativeAffineVar::inputize(u.cmQ)?,
            NonNativeAffineVar::inputize(u.cmW)?,
            NonNativeAffineVar::inputize(cmT)?,
            vec![r],
        ]
        .concat();

        let prepared_inputs = ark_groth16::Groth16::<E>::prepare_inputs(&snark_vk, &public_inputs)?;
        end_timer!(inputs_timer);

        let snark_timer = start_timer!(|| "CP-SNARK verification");
        let snark_v = ark_groth16::Groth16::<E>::verify_proof_with_prepared_inputs(
            &snark_vk,
            &(proof, vec![UU.cmQ.into(), UU.cmW.into(), UU.cmE.into()]),
            &prepared_inputs,
        )
        .map_err(|e| Error::Other(e.to_string()))?;
        end_timer!(snark_timer);

        end_timer!(verify_timer);

        if !snark_v {
            return Err(Error::SNARKVerificationFail);
        }

        Ok(true)
    }
}
