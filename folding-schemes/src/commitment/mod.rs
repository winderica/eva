use ark_ec::CurveGroup;
use ark_std::fmt::Debug;
use ark_std::rand::RngCore;
use icicle_cuda_runtime::memory::DeviceVec;
use icicle_cuda_runtime::stream::CudaStream;

use crate::transcript::Transcript;
use crate::{Error, MSM};

pub mod pedersen;

/// CommitmentScheme defines the vector commitment scheme trait. Where `H` indicates if to use the
/// commitment in hiding mode or not.
pub trait CommitmentScheme<C: CurveGroup, const H: bool = false>: Clone + Debug
where
    C::Config: MSM<C>,
{
    type ProverParams: Clone;
    type VerifierParams: Clone;
    type Proof: Clone + Debug;
    type ProverChallenge: Clone + Debug;
    type Challenge: Clone + Debug;

    fn setup(
        rng: &mut impl RngCore,
        len: usize,
    ) -> Result<(Self::ProverParams, Self::VerifierParams), Error>;

    fn commit(
        stream: Option<&CudaStream>,
        params: &Self::ProverParams,
        v: &[C::ScalarField],
        blind: &C::ScalarField,
    ) -> Result<C, Error>;

    fn commit_device(
        stream: Option<&CudaStream>,
        params: &Self::ProverParams,
        v: &DeviceVec<C::ScalarField>,
        blind: &C::ScalarField,
    ) -> Result<C, Error>;

    fn prove(
        params: &Self::ProverParams,
        transcript: &mut impl Transcript<C>,
        cm: &C,
        v: &[C::ScalarField],
        blind: &C::ScalarField,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Error>;

    /// same as `prove` but instead of providing a Transcript to use, providing the already
    /// computed challenge
    fn prove_with_challenge(
        params: &Self::ProverParams,
        challenge: Self::ProverChallenge,
        v: &[C::ScalarField],
        blind: &C::ScalarField,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Error>;

    fn verify(
        params: &Self::VerifierParams,
        transcript: &mut impl Transcript<C>,
        cm: &C,
        proof: &Self::Proof,
    ) -> Result<(), Error>;

    /// same as `verify` but instead of providing a Transcript to use, providing the already
    /// computed challenge
    fn verify_with_challenge(
        params: &Self::VerifierParams,
        challenge: Self::Challenge,
        cm: &C,
        proof: &Self::Proof,
    ) -> Result<(), Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Projective as G1};
    use ark_crypto_primitives::sponge::{poseidon::PoseidonConfig, Absorb};
    use ark_std::Zero;
    use ark_std::{test_rng, UniformRand};

    use super::pedersen::Pedersen;
    use crate::transcript::{
        poseidon::{poseidon_test_config, PoseidonTranscript},
        Transcript,
    };

    #[test]
    fn test_homomorphic_property_using_Commitment_trait() {
        let mut rng = &mut test_rng();
        let poseidon_config = poseidon_test_config::<Fr>();
        let n: usize = 128;

        // set random vector for the test
        let v_1: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(rng)).take(n).collect();
        let v_2: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(rng)).take(n).collect();
        // set a random challenge for the random linear combination
        let r = Fr::rand(rng);

        // setup params for Pedersen & KZG
        let (pedersen_params, _) = Pedersen::<G1>::setup(&mut rng, n).unwrap();
        // let (kzg_pk, kzg_vk): (ProverKey<G1>, VerifierKey<Bn254>) =
        //     KZG::<Bn254>::setup(rng, n).unwrap();

        // test with Pedersen
        test_homomorphic_property_using_Commitment_trait_opt::<G1, Pedersen<G1>>(
            &poseidon_config,
            &pedersen_params,
            &pedersen_params,
            r,
            &v_1,
            &v_2,
        );
        // // test with IPA
        // test_homomorphic_property_using_Commitment_trait_opt::<G1, IPA<G1>>(
        //     &poseidon_config,
        //     &pedersen_params,
        //     &pedersen_params,
        //     r,
        //     &v_1,
        //     &v_2,
        // );
        // // test with KZG
        // test_homomorphic_property_using_Commitment_trait_opt::<G1, KZG<Bn254>>(
        //     &poseidon_config,
        //     &kzg_pk,
        //     &kzg_vk,
        //     r,
        //     &v_1,
        //     &v_2,
        // );
    }

    fn test_homomorphic_property_using_Commitment_trait_opt<
        C: CurveGroup,
        CS: CommitmentScheme<C>,
    >(
        poseidon_config: &PoseidonConfig<C::ScalarField>,
        prover_params: &CS::ProverParams,
        verifier_params: &CS::VerifierParams,
        r: C::ScalarField,
        v_1: &[C::ScalarField],
        v_2: &[C::ScalarField],
    ) where
        C::ScalarField: Absorb,
        C::Config: MSM<C>,
    {
        // compute the commitment of the two vectors using the given CommitmentScheme
        let cm_1 = CS::commit(None, prover_params, v_1, &C::ScalarField::zero()).unwrap();
        let cm_2 = CS::commit(None, prover_params, v_2, &C::ScalarField::zero()).unwrap();

        // random linear combination of the commitments and their witnesses (vectors v_i)
        let cm_3 = cm_1 + cm_2.mul(r);
        let v_3: Vec<C::ScalarField> = v_1.iter().zip(v_2).map(|(a, b)| *a + (r * b)).collect();

        // compute the proof of the cm_3
        let transcript_p = &mut PoseidonTranscript::<C>::new(poseidon_config);
        let proof = CS::prove(
            prover_params,
            transcript_p,
            &cm_3,
            &v_3,
            &C::ScalarField::zero(),
            None,
        )
        .unwrap();

        // verify the opening proof
        let transcript_v = &mut PoseidonTranscript::<C>::new(poseidon_config);
        CS::verify(verifier_params, transcript_v, &cm_3, &proof).unwrap();
    }
}
