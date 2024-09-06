#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(clippy::upper_case_acronyms)]

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_r1cs_std::groups::CurveVar;
use ark_std::rand::{CryptoRng, Rng};
use ark_std::{end_timer, log2, start_timer};
use ark_std::{fmt::Debug, rand::RngCore, Zero};
use commitment::CommitmentScheme;
use folding::nova::circuits::CF2;
use folding::nova::decider_pedersen::DeciderEthCircuit;
use icicle_core::traits::ArkConvertible;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use rayon::prelude::*;
use thiserror::Error;
use utils::vec::CSRSparseMatrix;

use crate::frontend::FCircuit;

pub mod ccs;
pub mod commitment;
pub mod constants;
pub mod folding;
pub mod frontend;
pub mod transcript;
pub mod utils;

#[derive(Debug, Error)]
pub enum Error {
    // Wrappers on top of other errors
    #[error("ark_relations::r1cs::SynthesisError")]
    SynthesisError(#[from] ark_relations::r1cs::SynthesisError),
    #[error("ark_serialize::SerializationError")]
    SerializationError(#[from] ark_serialize::SerializationError),
    #[error("ark_poly_commit::Error")]
    PolyCommitError(#[from] ark_poly_commit::Error),
    #[error("crate::utils::espresso::virtual_polynomial::ArithErrors")]
    ArithError(#[from] utils::espresso::virtual_polynomial::ArithErrors),
    // #[error(transparent)]
    // ProtoGalaxy(folding::protogalaxy::ProtoGalaxyError),
    #[error("std::io::Error")]
    IOError(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),

    // Relation errors
    #[error("Relation not satisfied")]
    NotSatisfied,
    #[error("SNARK verification failed")]
    SNARKVerificationFail,
    #[error("IVC verification failed")]
    IVCVerificationFail,
    #[error("R1CS instance is expected to not be relaxed")]
    R1CSUnrelaxedFail,
    #[error("Could not find the inner ConstraintSystem")]
    NoInnerConstraintSystem,
    #[error("Sum-check prove failed: {0}")]
    SumCheckProveError(String),
    #[error("Sum-check verify failed: {0}")]
    SumCheckVerifyError(String),

    // Comparators errors
    #[error("Not equal")]
    NotEqual,
    #[error("Vectors should have the same length ({0}: {1}, {2}: {3})")]
    NotSameLength(String, usize, String, usize),
    #[error("Vector's length ({0}) is not the expected ({1})")]
    NotExpectedLength(usize, usize),
    #[error("Vector ({0}) length ({1}) is not a power of two")]
    NotPowerOfTwo(String, usize),
    #[error("Can not be empty")]
    Empty,
    #[error("Value out of bounds")]
    OutOfBounds,
    #[error("Could not construct the Evaluation Domain")]
    NewDomainFail,

    // Commitment errors
    #[error("Pedersen parameters length is not sufficient (generators.len={0} < vector.len={1} unsatisfied)")]
    PedersenParamsLen(usize, usize),
    #[error("Blinding factor not 0 for Commitment without hiding")]
    BlindingNotZero,
    #[error("Commitment verification failed")]
    CommitmentVerificationFail,

    // Other
    #[error("Randomness for blinding not found")]
    MissingRandomness,
    #[error("Missing value: {0}")]
    MissingValue(String),
    #[error("Feature '{0}' not supported yet")]
    NotSupportedYet(String),
    #[error("Feature '{0}' is not supported and it will not be")]
    NotSupported(String),
    #[error("max i-th step reached (usize limit reached)")]
    MaxStep,
    #[error("Circom Witness calculation error: {0}")]
    WitnessCalculationError(String),
    #[error("BigInt to PrimeField conversion error: {0}")]
    BigIntConversionError(String),
}

/// FoldingScheme defines trait that is implemented by the diverse folding schemes. It is defined
/// over a cycle of curves (C1, C2), where:
/// - C1 is the main curve, which ScalarField we use as our F for al the field operations
/// - C2 is the auxiliary curve, which we use for the commitments, whose BaseField (for point
/// coordinates) are in the C1::ScalarField.
/// In other words, C1.Fq == C2.Fr, and C1.Fr == C2.Fq.
pub trait FoldingScheme<C1: CurveGroup, C2: CurveGroup, FC>: Sized
where
    C1: CurveGroup<BaseField = C2::ScalarField, ScalarField = C2::BaseField>,
    C2::BaseField: PrimeField,
    FC: FCircuit<C1::ScalarField>,
{
    type PreprocessorParam;
    type ProverParam;
    type VerifierParam;
    type RunningCommittedInstanceWithWitness: Debug;
    type CurrentCommittedInstanceWithWitness: Debug;
    type CFCommittedInstanceWithWitness: Debug; // CycleFold CommittedInstance & Witness

    fn preprocess<R: Rng>(
        prep_param: &Self::PreprocessorParam,
        step_circuit: &FC,
        rng: &mut R,
        external_inputs: &FC::ExternalInputs,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn init(
        pp: &(Self::ProverParam, Self::VerifierParam),
        step_circuit: FC,
        z_0: Vec<C1::ScalarField>, // initial state
    ) -> Result<Self, Error>;

    fn prove_step(
        &mut self,
        pp: &(Self::ProverParam, Self::VerifierParam),
        external_inputs: &FC::ExternalInputs,
    ) -> Result<(), Error>;

    // returns the state at the current step
    fn state(&self) -> Vec<C1::ScalarField>;

    // returns the instances at the current step, in the following order:
    // (running_instance, incoming_instance, cyclefold_instance)
    fn instances(
        &self,
    ) -> (
        Self::RunningCommittedInstanceWithWitness,
        Self::CurrentCommittedInstanceWithWitness,
        Self::CFCommittedInstanceWithWitness,
    );

    fn verify(
        vp: &Self::VerifierParam,
        z_0: Vec<C1::ScalarField>, // initial state
        z_i: Vec<C1::ScalarField>, // last state
        // number of steps between the initial state and the last state
        num_steps: C1::ScalarField,
        running_instance: Self::RunningCommittedInstanceWithWitness,
        incoming_instance: Self::CurrentCommittedInstanceWithWitness,
        cyclefold_instance: Self::CFCommittedInstanceWithWitness,
    ) -> Result<(), Error>;
}

pub trait Decider<C1: CurveGroup, C2: CurveGroup, GC1, GC2, CS1, CS2>
where
    C1: CurveGroup<BaseField = C2::ScalarField, ScalarField = C2::BaseField>,
    C2::BaseField: PrimeField,
    C1::Config: MSM<C1>,
    C2::Config: MSM<C2>,
    GC1: CurveVar<C1, CF2<C1>>,
    GC2: CurveVar<C2, CF2<C2>>,
    CS1: CommitmentScheme<C1>,
    CS2: CommitmentScheme<C2>,
{
    type ProverParam: Clone;
    type Proof;
    type VerifierParam;
    type PublicInput: Debug;
    type CommittedInstanceWithWitness: Debug;
    type CurrentInstance: Clone + Debug;
    type RunningInstance: Clone + Debug;

    fn prove(
        pp: Self::ProverParam,
        rng: impl RngCore + CryptoRng,
        circuit: DeciderEthCircuit<C1, GC1, C2, GC2, CS1, CS2>,
    ) -> Result<Self::Proof, Error>;

    fn verify(
        vp: Self::VerifierParam,
        i: C1::ScalarField,
        z_0: Vec<C1::ScalarField>,
        z_i: Vec<C1::ScalarField>,
        running_instance: &Self::RunningInstance,
        incoming_instance: &Self::CurrentInstance,
        proof: Self::Proof,
        // returns `Result<bool, Error>` to differentiate between an error occurred while performing
        // the verification steps, and the verification logic of the scheme not passing.
    ) -> Result<bool, Error>;
}

pub trait MVM
where
    Self: PrimeField,
{
    fn prepare_matrix(csr: &CSRSparseMatrix<Self>) -> HybridMatrix;

    fn compute_t(
        stream: Option<&CudaStream>,
        a: &HybridMatrix,
        b: &HybridMatrix,
        c: &HybridMatrix,
        z1_u: &[Self],
        z1_x: &[Self],
        z1_wq: &[Self],
        z2_u: &[Self],
        z2_x: &[Self],
        z2_wq: &[Self],
        e: &DeviceVec<Self>,
        t: &mut DeviceVec<Self>,
    );

    fn alloc_vec(len: usize) -> DeviceVec<Self> {
        let mut ptr = DeviceVec::cuda_malloc(len).unwrap();
        let zeros = vec![Self::zero(); len];
        ptr.copy_from_host(HostSlice::from_slice(zeros.cast()))
            .unwrap();
        ptr
    }

    fn update_e(stream: Option<&CudaStream>, e: &mut DeviceVec<Self>, t: &DeviceVec<Self>, r: Self);

    fn retrieve_e(e: &DeviceVec<Self>) -> Vec<Self>;
}

pub trait MSM<C: CurveGroup> {
    type T;
    type R;

    fn precompute(generators: &[C::Affine]) -> icicle_cuda_runtime::memory::DeviceVec<Self::T>;

    fn var_msm(points: &[C::Affine], scalars: &[C::ScalarField], offset: usize) -> C;

    fn var_msm_precomputed(
        stream: Option<&CudaStream>,
        points: &icicle_cuda_runtime::memory::DeviceSlice<Self::T>,
        scalars: &[C::ScalarField],
        offset: usize,
        bitsize: Option<usize>,
    ) -> DeviceVec<Self::R>;

    fn var_msm_device_precomputed(
        stream: Option<&CudaStream>,
        points: &icicle_cuda_runtime::memory::DeviceSlice<Self::T>,
        scalars: &DeviceVec<C::ScalarField>,
        offset: usize,
    ) -> DeviceVec<Self::R>;

    fn retrieve_msm_result(stream: Option<&CudaStream>, result: &DeviceVec<Self::R>) -> C;
}

const PRECOMPUTE_FACTOR: usize = 4;
const C: i32 = 4;

impl MSM<ark_bn254::G1Projective> for ark_bn254::g1::Config {
    type T = icicle_bn254::curve::G1Affine;
    type R = icicle_bn254::curve::G1Projective;

    fn precompute(
        generators: &[ark_bn254::G1Affine],
    ) -> icicle_cuda_runtime::memory::DeviceVec<Self::T> {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            memory::{DeviceVec, HostSlice},
            stream::CudaStream,
        };
        let mut precomputed_points_d =
            DeviceVec::cuda_malloc(PRECOMPUTE_FACTOR * generators.len()).unwrap();
        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.precompute_factor = PRECOMPUTE_FACTOR as i32;
        // cfg.start = offset as i32;
        cfg.ctx.stream = &stream;
        cfg.is_async = true;
        cfg.c = max(log2(generators.len()) as i32 - C, 1);

        msm::precompute_points(
            HostSlice::from_slice(
                &generators
                    .to_vec()
                    .into_par_iter()
                    .map(icicle_bn254::curve::G1Affine::from_ark)
                    .collect::<Vec<_>>(),
            ),
            generators.len() as i32,
            &cfg,
            &mut precomputed_points_d,
        )
        .unwrap();
        stream
            .synchronize()
            .expect("Failed to synchronize CUDA stream");
        precomputed_points_d
    }

    fn var_msm(
        points: &[ark_bn254::G1Affine],
        scalars: &[ark_bn254::Fr],
        offset: usize,
    ) -> ark_bn254::G1Projective {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            memory::{DeviceVec, HostSlice},
            stream::CudaStream,
        };
        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        // cfg.start = offset as i32;
        cfg.ctx.stream = &stream;
        cfg.is_async = true;

        let mut msm_results =
            DeviceVec::<icicle_bn254::curve::G1Projective>::cuda_malloc(1).unwrap();
        msm::msm(
            HostSlice::from_slice(scalars.cast()),
            HostSlice::from_slice(
                &points
                    .iter()
                    .map(|&i| icicle_bn254::curve::G1Affine::from_ark(i))
                    .collect::<Vec<_>>(),
            ),
            &cfg,
            &mut msm_results[..],
        )
        .unwrap();
        let mut msm_host_result = vec![icicle_bn254::curve::G1Projective::zero(); 1];

        stream.synchronize().unwrap();
        msm_results
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
            .unwrap();
        stream.destroy().unwrap();
        msm_host_result[0].to_ark()
    }

    fn var_msm_precomputed(
        stream: Option<&CudaStream>,
        points: &icicle_cuda_runtime::memory::DeviceSlice<Self::T>,
        scalars: &[ark_bn254::Fr],
        offset: usize,
        bitsize: Option<usize>,
    ) -> DeviceVec<Self::R> {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
        let timer = start_timer!(|| "Start MSM over BN254");
        // let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.precompute_factor = PRECOMPUTE_FACTOR as i32;
        // cfg.start = offset as i32;
        if let Some(stream) = stream {
            cfg.ctx.stream = stream;
            cfg.is_async = true;
        }
        if let Some(bitsize) = bitsize {
            cfg.bitsize = bitsize as i32;
        }
        cfg.c = max(log2(points.len() / PRECOMPUTE_FACTOR) as i32 - C, 1);

        let mut msm_results = DeviceVec::<Self::R>::cuda_malloc(1).unwrap();
        msm::msm(
            HostSlice::from_slice(scalars.cast()),
            points,
            &cfg,
            &mut msm_results[..],
        )
        .unwrap();
        end_timer!(timer);

        msm_results
    }

    fn var_msm_device_precomputed(
        stream: Option<&CudaStream>,
        points: &icicle_cuda_runtime::memory::DeviceSlice<Self::T>,
        scalars: &DeviceVec<ark_bn254::Fr>,
        offset: usize,
    ) -> DeviceVec<Self::R> {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice};
        let timer = start_timer!(|| "Start MSM over BN254");
        // let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = false;
        cfg.precompute_factor = PRECOMPUTE_FACTOR as i32;
        // cfg.start = offset as i32;
        if let Some(stream) = stream {
            cfg.ctx.stream = stream;
            cfg.is_async = true;
        }
        cfg.c = max(log2(points.len() / PRECOMPUTE_FACTOR) as i32 - C, 1);

        let mut msm_results = DeviceVec::<Self::R>::cuda_malloc(1).unwrap();
        msm::msm(
            unsafe { std::mem::transmute::<_, &DeviceSlice<_>>(&scalars[..]) },
            points,
            &cfg,
            &mut msm_results[..],
        )
        .unwrap();
        end_timer!(timer);

        msm_results
    }

    fn retrieve_msm_result(
        stream: Option<&CudaStream>,
        msm_results: &DeviceVec<Self::R>,
    ) -> ark_bn254::G1Projective {
        if let Some(stream) = stream {
            let timer = start_timer!(|| "Synchronize");
            stream.synchronize().unwrap();
            end_timer!(timer);
        }

        let timer = start_timer!(|| "Get result from GPU");
        let mut msm_host_result = vec![Self::R::zero(); 1];
        msm_results
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
            .unwrap();
        let result = msm_host_result[0].to_ark();
        end_timer!(timer);

        result
    }

    // fn sparse_mvm(
    //     csr: &CudaSparseMatrix<ark_bn254::Fr>,
    //     z: &[ark_bn254::Fr],
    // ) -> Vec<ark_bn254::Fr> {
    //     use icicle_core::ntt::FieldImpl;
    //     use icicle_core::traits::ArkConvertible;
    //     use icicle_cuda_runtime::memory::HostSlice;

    //     let mut result = vec![icicle_bn254::curve::ScalarField::zero(); csr.row_ptr.len() - 1];

    //     let cfg = icicle_core::vec_ops::VecOpsConfig::default();

    //     icicle_core::vec_ops::mul_mat(
    //         HostSlice::from_slice(
    //             &z.iter()
    //                 .map(|&i| icicle_bn254::curve::ScalarField::from_ark(i))
    //                 .collect::<Vec<_>>(),
    //         ),
    //         HostSlice::from_slice(
    //             &csr.data
    //                 .iter()
    //                 .map(|&i| icicle_bn254::curve::ScalarField::from_ark(i))
    //                 .collect::<Vec<_>>(),
    //         ),
    //         HostSlice::from_slice(&csr.row_ptr),
    //         HostSlice::from_slice(&csr.col_idx),
    //         HostSlice::from_mut_slice(&mut result),
    //         &cfg,
    //     )
    //     .unwrap();
    //     result.iter().map(|&i| i.to_ark()).collect()
    // }
}

impl MVM for ark_bn254::Fr {
    fn prepare_matrix(csr: &CSRSparseMatrix<ark_bn254::Fr>) -> HybridMatrix {
        use icicle_core::ntt::FieldImpl;
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            device_context::{DeviceContext, DEFAULT_DEVICE_ID},
            memory::HostSlice,
        };

        let mut result = HybridMatrix::default();

        let ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        icicle_core::vec_ops::prepare_matrix(
            HostSlice::from_slice(csr.data.cast::<icicle_bn254::curve::ScalarField>()),
            HostSlice::from_slice(&csr.row_ptr),
            HostSlice::from_slice(&csr.col_idx),
            HostSlice::from_slice(&csr.sparse_to_original),
            HostSlice::from_slice(&csr.dense_to_original),
            &ctx,
            &mut result,
        )
        .unwrap();
        result
    }

    fn compute_t(
        stream: Option<&CudaStream>,
        a: &HybridMatrix,
        b: &HybridMatrix,
        c: &HybridMatrix,
        z1_u: &[Self],
        z1_x: &[Self],
        z1_wq: &[Self],
        z2_u: &[Self],
        z2_x: &[Self],
        z2_wq: &[Self],
        e: &DeviceVec<ark_bn254::Fr>,
        t: &mut DeviceVec<ark_bn254::Fr>,
    ) {
        use icicle_core::ntt::FieldImpl;
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            device_context::{DeviceContext, DEFAULT_DEVICE_ID},
            memory::{DeviceSlice, HostOrDeviceSlice, HostSlice},
        };

        let mut ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        if let Some(stream) = stream {
            ctx.stream = stream;
        }

        icicle_core::vec_ops::compute_t(
            &a, &b, &c,
            HostSlice::from_slice(z1_u.cast::<icicle_bn254::curve::ScalarField>()),
            HostSlice::from_slice(z1_x.cast::<icicle_bn254::curve::ScalarField>()),
            HostSlice::from_slice(z1_wq.cast::<icicle_bn254::curve::ScalarField>()),
            HostSlice::from_slice(z2_u.cast::<icicle_bn254::curve::ScalarField>()),
            HostSlice::from_slice(z2_x.cast::<icicle_bn254::curve::ScalarField>()),
            HostSlice::from_slice(z2_wq.cast::<icicle_bn254::curve::ScalarField>()),
            unsafe { std::mem::transmute::<_, &DeviceSlice<_>>(&e[..]) },
            &ctx,
            unsafe { std::mem::transmute::<_, &mut DeviceSlice<_>>(&mut t[..]) },
        )
        .unwrap();
    }

    fn update_e(
        stream: Option<&CudaStream>,
        e: &mut DeviceVec<Self>,
        t: &DeviceVec<Self>,
        r: Self,
    ) {
        let mut ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        if let Some(stream) = stream {
            ctx.stream = stream;
        }

        icicle_core::vec_ops::update_e(
            unsafe { std::mem::transmute::<_, &mut DeviceSlice<_>>(&mut e[..]) },
            unsafe { std::mem::transmute::<_, &DeviceSlice<_>>(&t[..]) },
            HostSlice::from_slice(vec![r].cast::<icicle_bn254::curve::ScalarField>()),
            &ctx,
        )
        .unwrap();
    }

    fn retrieve_e(e: &DeviceVec<Self>) -> Vec<Self> {
        let ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        let mut result = vec![Self::zero(); e.len()];

        icicle_core::vec_ops::return_e(
            unsafe {
                std::mem::transmute::<_, &DeviceSlice<icicle_bn254::curve::ScalarField>>(&e[..])
            },
            &ctx,
            HostSlice::from_mut_slice(result.cast_mut()),
        )
        .unwrap();

        result
    }
}

impl MSM<ark_grumpkin::Projective> for ark_grumpkin::GrumpkinConfig {
    type T = icicle_grumpkin::curve::G1Affine;
    type R = icicle_grumpkin::curve::G1Projective;

    fn precompute(
        generators: &[ark_grumpkin::Affine],
    ) -> icicle_cuda_runtime::memory::DeviceVec<Self::T> {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            memory::{DeviceVec, HostSlice},
            stream::CudaStream,
        };
        let mut precomputed_points_d =
            DeviceVec::cuda_malloc(PRECOMPUTE_FACTOR * generators.len()).unwrap();
        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.precompute_factor = PRECOMPUTE_FACTOR as i32;
        // cfg.start = offset as i32;
        cfg.ctx.stream = &stream;
        cfg.is_async = true;
        cfg.c = max(log2(generators.len()) as i32 - 4, 1);

        msm::precompute_points(
            HostSlice::from_slice(
                &generators
                    .to_vec()
                    .into_par_iter()
                    .map(icicle_grumpkin::curve::G1Affine::from_ark)
                    .collect::<Vec<_>>(),
            ),
            generators.len() as i32,
            &cfg,
            &mut precomputed_points_d,
        )
        .unwrap();
        stream
            .synchronize()
            .expect("Failed to synchronize CUDA stream");
        precomputed_points_d
    }

    fn var_msm(
        points: &[ark_grumpkin::Affine],
        scalars: &[ark_grumpkin::Fr],
        offset: usize,
    ) -> ark_grumpkin::Projective {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            memory::{DeviceVec, HostOrDeviceSlice, HostSlice},
            stream::CudaStream,
        };
        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        // cfg.start = offset as i32;
        cfg.ctx.stream = &stream;
        cfg.is_async = true;

        let mut msm_results =
            DeviceVec::<icicle_grumpkin::curve::G1Projective>::cuda_malloc(1).unwrap();
        msm::msm(
            HostSlice::from_slice(scalars.cast()),
            HostSlice::from_slice(
                &points
                    .par_iter()
                    .map(|&i| icicle_grumpkin::curve::G1Affine::from_ark(i))
                    .collect::<Vec<_>>(),
            ),
            &cfg,
            &mut msm_results[..],
        )
        .unwrap();
        let mut msm_host_result = vec![icicle_grumpkin::curve::G1Projective::zero(); 1];

        stream.synchronize().unwrap();
        msm_results
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
            .unwrap();
        stream.destroy().unwrap();
        msm_host_result[0].to_ark()
    }

    fn var_msm_precomputed(
        stream: Option<&CudaStream>,
        points: &icicle_cuda_runtime::memory::DeviceSlice<Self::T>,
        scalars: &[ark_grumpkin::Fr],
        offset: usize,
        bitsize: Option<usize>,
    ) -> DeviceVec<Self::R> {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
        let timer = start_timer!(|| "Start MSM over Grumpkin");
        // let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        cfg.precompute_factor = PRECOMPUTE_FACTOR as i32;
        // cfg.start = offset as i32;
        if let Some(stream) = stream {
            cfg.ctx.stream = stream;
            cfg.is_async = true;
        }
        if let Some(bitsize) = bitsize {
            cfg.bitsize = bitsize as i32;
        }
        cfg.c = max(log2(points.len() / PRECOMPUTE_FACTOR) as i32 - C, 1);

        let mut msm_results = DeviceVec::<Self::R>::cuda_malloc(1).unwrap();
        msm::msm(
            HostSlice::from_slice(scalars.cast()),
            points,
            &cfg,
            &mut msm_results[..],
        )
        .unwrap();
        end_timer!(timer);

        msm_results
    }

    fn var_msm_device_precomputed(
        stream: Option<&CudaStream>,
        points: &icicle_cuda_runtime::memory::DeviceSlice<Self::T>,
        scalars: &DeviceVec<ark_grumpkin::Fr>,
        offset: usize,
    ) -> DeviceVec<Self::R> {
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice};
        let timer = start_timer!(|| "Start MSM over Grumpkin");
        // let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = false;
        cfg.precompute_factor = PRECOMPUTE_FACTOR as i32;
        // cfg.start = offset as i32;
        if let Some(stream) = stream {
            cfg.ctx.stream = stream;
            cfg.is_async = true;
        }
        cfg.c = max(log2(points.len() / PRECOMPUTE_FACTOR) as i32 - C, 1);

        let mut msm_results = DeviceVec::<Self::R>::cuda_malloc(1).unwrap();
        msm::msm(
            unsafe { std::mem::transmute::<_, &DeviceSlice<_>>(&scalars[..]) },
            points,
            &cfg,
            &mut msm_results[..],
        )
        .unwrap();
        end_timer!(timer);

        msm_results
    }

    fn retrieve_msm_result(
        stream: Option<&CudaStream>,
        msm_results: &DeviceVec<Self::R>,
    ) -> ark_grumpkin::Projective {
        if let Some(stream) = stream {
            let timer = start_timer!(|| "Synchronize");
            stream.synchronize().unwrap();
            end_timer!(timer);
        }

        let timer = start_timer!(|| "Get result from GPU");
        let mut msm_host_result = vec![Self::R::zero(); 1];
        msm_results
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
            .unwrap();
        let result = msm_host_result[0].to_ark();
        end_timer!(timer);

        result
    }

    // fn sparse_mvm(
    //     csr: &HybridMatrix,
    //     z: &[ark_grumpkin::Fr],
    // ) -> Vec<ark_grumpkin::Fr> {
    //     use icicle_cuda_runtime::memory::HostSlice;

    //     let mut result = vec![ark_grumpkin::Fr::zero(); csr.row_ptr.len() - 1];

    //     let cfg = icicle_core::vec_ops::VecOpsConfig::default();

    //     icicle_core::vec_ops::mul_mat(
    //         HostSlice::from_slice(z.cast::<icicle_grumpkin::curve::ScalarField>()),
    //         HostSlice::from_slice(csr.data.cast()),
    //         HostSlice::from_slice(&csr.row_ptr),
    //         HostSlice::from_slice(&csr.col_idx),
    //         HostSlice::from_mut_slice(result.cast_mut()),
    //         &cfg,
    //     )
    //     .unwrap();
    //     result
    // }
}

impl MVM for ark_grumpkin::Fr {
    fn prepare_matrix(
        csr: &CSRSparseMatrix<ark_grumpkin::Fr>,
    ) -> HybridMatrix {
        use icicle_core::ntt::FieldImpl;
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            device_context::{DeviceContext, DEFAULT_DEVICE_ID},
            memory::HostSlice,
        };

        let mut result = HybridMatrix::default();

        let ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        icicle_core::vec_ops::prepare_matrix(
            HostSlice::from_slice(csr.data.cast::<icicle_grumpkin::curve::ScalarField>()),
            HostSlice::from_slice(&csr.row_ptr),
            HostSlice::from_slice(&csr.col_idx),
            HostSlice::from_slice(&csr.sparse_to_original),
            HostSlice::from_slice(&csr.dense_to_original),
            &ctx,
            &mut result,
        )
        .unwrap();
        result
    }

    fn compute_t(
        stream: Option<&CudaStream>,
        a: &HybridMatrix,
        b: &HybridMatrix,
        c: &HybridMatrix,
        z1_u: &[Self],
        z1_x: &[Self],
        z1_wq: &[Self],
        z2_u: &[Self],
        z2_x: &[Self],
        z2_wq: &[Self],
        e: &DeviceVec<ark_grumpkin::Fr>,
        t: &mut DeviceVec<ark_grumpkin::Fr>,
    ) {
        use icicle_core::ntt::FieldImpl;
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::{
            device_context::{DeviceContext, DEFAULT_DEVICE_ID},
            memory::{DeviceSlice, HostOrDeviceSlice, HostSlice},
        };

        let mut ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        if let Some(stream) = stream {
            ctx.stream = stream;
        }

        icicle_core::vec_ops::compute_t(
            &a, &b, &c,
            HostSlice::from_slice(z1_u.cast::<icicle_grumpkin::curve::ScalarField>()),
            HostSlice::from_slice(z1_x.cast::<icicle_grumpkin::curve::ScalarField>()),
            HostSlice::from_slice(z1_wq.cast::<icicle_grumpkin::curve::ScalarField>()),
            HostSlice::from_slice(z2_u.cast::<icicle_grumpkin::curve::ScalarField>()),
            HostSlice::from_slice(z2_x.cast::<icicle_grumpkin::curve::ScalarField>()),
            HostSlice::from_slice(z2_wq.cast::<icicle_grumpkin::curve::ScalarField>()),
            unsafe { std::mem::transmute::<_, &DeviceSlice<_>>(&e[..]) },
            &ctx,
            unsafe { std::mem::transmute::<_, &mut DeviceSlice<_>>(&mut t[..]) },
        )
        .unwrap();
    }

    fn update_e(
        stream: Option<&CudaStream>,
        e: &mut DeviceVec<ark_grumpkin::Fr>,
        t: &DeviceVec<ark_grumpkin::Fr>,
        r: ark_grumpkin::Fr,
    ) {
        let mut ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        if let Some(stream) = stream {
            ctx.stream = stream;
        }

        icicle_core::vec_ops::update_e(
            unsafe { std::mem::transmute::<_, &mut DeviceSlice<_>>(&mut e[..]) },
            unsafe { std::mem::transmute::<_, &DeviceSlice<_>>(&t[..]) },
            HostSlice::from_slice(vec![r].cast::<icicle_grumpkin::curve::ScalarField>()),
            &ctx,
        )
        .unwrap();
    }

    fn retrieve_e(e: &DeviceVec<Self>) -> Vec<Self> {
        let ctx = DeviceContext::default_for_device(DEFAULT_DEVICE_ID);

        let mut result = vec![Self::zero(); e.len()];

        icicle_core::vec_ops::return_e(
            unsafe {
                std::mem::transmute::<_, &DeviceSlice<icicle_grumpkin::curve::ScalarField>>(&e[..])
            },
            &ctx,
            HostSlice::from_mut_slice(result.cast_mut()),
        )
        .unwrap();

        result
    }
}

use icicle_core::vec_ops::HybridMatrix;
use std::cmp::max;
use std::mem::size_of;
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait CastSlice<From>: AsRef<[From]> {
    #[inline]
    fn cast<To>(&self) -> &[To] {
        let slice = self.as_ref();
        unsafe {
            from_raw_parts(
                slice as *const _ as *const To,
                slice.len() * size_of::<From>() / size_of::<To>(),
            )
        }
    }

    #[inline]
    fn cast_mut<To>(&mut self) -> &mut [To] {
        let slice = self.as_ref();
        unsafe {
            from_raw_parts_mut(
                slice as *const _ as *mut To,
                slice.len() * size_of::<From>() / size_of::<To>(),
            )
        }
    }
}

impl<From> CastSlice<From> for &[From] {}
impl<From> CastSlice<From> for [From] {}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ark_ec::VariableBaseMSM;
    use ark_ff::{FftField, Field, UniformRand};
    use ark_r1cs_std::alloc::AllocVar;
    use ark_r1cs_std::fields::fp::FpVar;
    use ark_r1cs_std::fields::FieldVar;
    use ark_relations::r1cs::ConstraintSynthesizer;
    use ark_std::test_rng;
    use commitment::pedersen::Pedersen;
    use folding::nova::get_r1cs_from_cs;
    use folding::nova::nifs::NIFS;
    use icicle_core::traits::ArkConvertible;
    use icicle_core::{
        curve::Curve,
        msm::{self, MSMConfig}
        ,
    };
    use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};
    use rand::thread_rng;

    use crate::utils::vec::{vec_add, vec_sub};

    use super::*;
    #[test]
    fn msm() {
        let m = 1 << 10;
        let n = 1 << 23;
        let rng = &mut test_rng();
        let points: Vec<_> = (0..m).map(|_| ark_bn254::G1Projective::rand(rng)).collect();
        let mut points = ark_bn254::G1Projective::normalize_batch(&points);
        while points.len() < n {
            points.extend_from_within(..);
        }
        let points_precomputed = ark_bn254::g1::Config::precompute(&points);
        for i in 0..5 {
            let mut scalars: Vec<ark_bn254::Fr> =
                (0..m).map(|_| ark_bn254::Fr::rand(rng)).collect();
            scalars[1] = ark_bn254::Fr::zero();
            while scalars.len() < n - n / 3 {
                scalars.extend_from_within(..);
            }
            // scalars.resize(n, ark_bn254::Fr::zero());
            let now = Instant::now();
            let res2 = ark_bn254::g1::Config::var_msm_precomputed(
                None,
                &points_precomputed,
                &scalars[..],
                0,
                None,
            );
            let res2 = ark_bn254::g1::Config::retrieve_msm_result(None, &res2);
            println!("{}", res2);
            println!("{:?}", now.elapsed());
            let now = Instant::now();
            let expected2 = ark_bn254::G1Projective::msm_unchecked(&points[..], &scalars[..]);
            println!("{}", expected2);
            println!("{:?}", now.elapsed());
            assert_eq!(res2, expected2);
        }
    }

    #[test]
    fn test() {
        for i in 0..33 {
            println!("{},", ark_pallas::Fr::get_root_of_unity(1u64 << i).unwrap());
        }
        println!();
        for i in 0..33 {
            println!(
                "{},",
                ark_pallas::Fr::get_root_of_unity(1u64 << i)
                    .unwrap()
                    .inverse()
                    .unwrap()
            );
        }

        let rng = &mut thread_rng();
        let a = ark_bn254::Fr::rand(rng);
        println!("{:?}", [a.into_bigint()].cast::<u8>());
        let b = icicle_bn254::curve::ScalarField::from_ark(a);
        println!("{:?}", [b].cast::<u8>());
    }

    #[test]
    fn msm2() {
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_ff::UniformRand;
        use ark_std::test_rng;
        use icicle_core::msm::{self, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_cuda_runtime::memory::HostSlice;
        let n = 1 << 18;
        let rng = &mut test_rng();
        let points = ark_bn254::G1Projective::normalize_batch(
            &(0..n)
                .map(|_| ark_bn254::G1Projective::rand(rng))
                .collect::<Vec<_>>(),
        );
        let points_icicle = points
            .clone()
            .into_iter()
            .map(icicle_bn254::curve::G1Affine::from_ark)
            .collect::<Vec<_>>();
        let scalars = (0..n).map(|_| ark_bn254::Fr::rand(rng)).collect::<Vec<_>>();
        let scalars_icicle = scalars
            .clone()
            .into_iter()
            .map(icicle_bn254::curve::ScalarField::from_ark)
            .collect::<Vec<_>>();
        let cfg = MSMConfig::default();

        let expected = ark_bn254::G1Projective::msm_unchecked(&points, &scalars);
        for i in 0..1000 {
            println!("{}", i);

            std::thread::sleep(std::time::Duration::from_millis(300));
            let result = {
                let mut msm_host_result = [icicle_bn254::curve::G1Projective::zero(); 1];
                msm::msm(
                    HostSlice::from_slice(&scalars_icicle),
                    HostSlice::from_slice(&points_icicle),
                    &cfg,
                    HostSlice::from_mut_slice(&mut msm_host_result[..]),
                )
                .unwrap();

                msm_host_result[0]
            };

            assert_eq!(
                result,
                icicle_bn254::curve::G1Projective::from_ark(expected)
            );
        }
    }

    struct TestCircuit {}

    impl<F: PrimeField> ConstraintSynthesizer<F> for TestCircuit {
        fn generate_constraints(
            self,
            cs: ark_relations::r1cs::ConstraintSystemRef<F>,
        ) -> ark_relations::r1cs::Result<()> {
            let mut r = FpVar::one();
            let mut s = FpVar::one();
            for i in 1u64..1000 {
                let x = FpVar::new_witness(cs.clone(), || Ok(F::from(i)))?;
                s += x.inverse()?;
                r *= &s.inverse()?;
            }
            Ok(())
        }
    }

    #[test]
    fn compute_t() {
        let rng = &mut test_rng();
        let r1cs = get_r1cs_from_cs::<ark_bn254::Fr>(TestCircuit {}).unwrap();
        let z1 = (0..r1cs.A.n_cols)
            .map(|_| ark_bn254::Fr::rand(rng))
            .collect::<Vec<_>>();
        let z2 = (0..r1cs.A.n_cols)
            .map(|_| ark_bn254::Fr::rand(rng))
            .collect::<Vec<_>>();
        let e1 = r1cs.eval_relation(&z1).unwrap();
        let e2 = r1cs.eval_relation(&z2).unwrap();
        let e = vec_add(&e1, &e2).unwrap();
        let now = Instant::now();
        let expected =
            NIFS::<ark_bn254::G1Projective, Pedersen<ark_bn254::G1Projective, false>>::compute_T(
                &r1cs, z1[0], z2[0], &z1, &z2,
            )
            .unwrap();
        println!("{:?}", now.elapsed());

        let mut t = ark_bn254::Fr::alloc_vec(r1cs.A.n_rows);
        let tmp_e = ark_bn254::Fr::alloc_vec(r1cs.A.n_rows);
        let now = Instant::now();

        ark_bn254::Fr::compute_t(
            None,
            &r1cs.A.cuda,
            &r1cs.B.cuda,
            &r1cs.C.cuda,
            &z1[..1],
            &z1[1..1 + r1cs.l],
            &z1[1 + r1cs.l..],
            &z2[..1],
            &z2[1..1 + r1cs.l],
            &z2[1 + r1cs.l..],
            &tmp_e,
            &mut t,
        );
        println!("{:?}", now.elapsed());
        let result = vec_sub(&ark_bn254::Fr::retrieve_e(&t), &e).unwrap();

        assert_eq!(result, expected);
    }
}
