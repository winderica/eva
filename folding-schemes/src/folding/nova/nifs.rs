use ark_crypto_primitives::sponge::Absorb;
use ark_ec::CurveGroup;
use ark_std::{cfg_into_iter, cfg_iter, end_timer, start_timer, Zero};
use rayon::prelude::*;
use std::marker::PhantomData;

use super::{CurrentInstance, CycleFoldCommittedInstance, RunningInstance, Witness};
use crate::ccs::r1cs::R1CS;
use crate::commitment::CommitmentScheme;
use crate::transcript::Transcript;
use crate::utils::vec::*;
use crate::{Error, MSM, MVM};

/// Implements the Non-Interactive Folding Scheme described in section 4 of
/// [Nova](https://eprint.iacr.org/2021/370.pdf)
pub struct NIFS<C: CurveGroup, CS: CommitmentScheme<C>>
where
    C::Config: MSM<C>,
{
    _c: PhantomData<C>,
    _cp: PhantomData<CS>,
}

impl<C: CurveGroup, CS: CommitmentScheme<C>> NIFS<C, CS>
where
    C::ScalarField: Absorb + MVM,
    C::Config: MSM<C>,
{
    // compute_T: compute cross-terms T
    pub fn compute_T(
        r1cs: &R1CS<C::ScalarField>,
        u1: C::ScalarField,
        u2: C::ScalarField,
        z1: &[C::ScalarField],
        z2: &[C::ScalarField],
    ) -> Result<Vec<C::ScalarField>, Error> {
        // this is parallelizable (for the future)
        let Az1 = mat_vec_mul_sparse(&r1cs.A, z1)?;
        let Bz1 = mat_vec_mul_sparse(&r1cs.B, z1)?;
        let Cz1 = mat_vec_mul_sparse(&r1cs.C, z1)?;
        let Az2 = mat_vec_mul_sparse(&r1cs.A, z2)?;
        let Bz2 = mat_vec_mul_sparse(&r1cs.B, z2)?;
        let Cz2 = mat_vec_mul_sparse(&r1cs.C, z2)?;

        Ok(cfg_into_iter!(Az1)
            .zip(Az2)
            .zip(Bz1)
            .zip(Bz2)
            .zip(Cz1)
            .zip(Cz2)
            .map(|(((((az1, az2), bz1), bz2), cz1), cz2)| {
                az1 * bz2 + az2 * bz1 - u1 * cz2 - u2 * cz1
            })
            .collect())
    }

    pub fn fold_witness(
        r: C::ScalarField,
        w1: &Witness<C>,
        w2: &Witness<C>,
        T: &[C::ScalarField],
        rT: C::ScalarField,
    ) -> Result<Witness<C>, Error> {
        let r2 = r * r;
        let E = cfg_iter!(&w1.E)
            .zip(T)
            .zip(&w2.E)
            .map(|((a, b), c)| *a + (r * b) + (r2 * c))
            .collect::<Vec<_>>();
        let rE = w1.rE + r * rT + r2 * w2.rE;
        let QW: Vec<C::ScalarField> = cfg_iter!(w1.QW)
            .zip(&w2.QW)
            .map(|(a, b)| *a + (r * b))
            .collect();
        let rQ = w1.rQ + r * w2.rQ;
        let rW = w1.rW + r * w2.rW;
        Ok(Witness::<C> {
            E,
            rE,
            QW,
            rW,
            rQ,
            Q_len: w1.Q_len,
        })
    }

    pub fn fold_committed_instance(
        r: C::ScalarField,
        ci1: &RunningInstance<C>, // U_i
        ci2: &CurrentInstance<C>, // u_i
        cmT: &C,
    ) -> RunningInstance<C> {
        let r2 = r * r;
        let cmE = ci1.cmE + cmT.mul(r) + ci2.cmE.mul(r2);
        let u = ci1.u + r * ci2.u;
        let cmQ = ci1.cmQ + ci2.cmQ.mul(r);
        let cmW = ci1.cmW + ci2.cmW.mul(r);
        let x = cfg_iter!(ci1.x)
            .zip(&ci2.x)
            .map(|(a, b)| *a + (r * b))
            .collect::<Vec<C::ScalarField>>();

        RunningInstance::<C> {
            cmE,
            u,
            cmQ,
            cmW,
            x,
        }
    }

    pub fn fold_cf_committed_instance(
        r: C::ScalarField,
        ci1: &CycleFoldCommittedInstance<C>, // U_i
        ci2: &CycleFoldCommittedInstance<C>, // u_i
        cmT: &C,
    ) -> CycleFoldCommittedInstance<C> {
        let r2 = r * r;
        let cmE = ci1.cmE + cmT.mul(r) + ci2.cmE.mul(r2);
        let u = ci1.u + r * ci2.u;
        let cmW = ci1.cmW + ci2.cmW.mul(r);
        let x = cfg_iter!(ci1.x)
            .zip(&ci2.x)
            .map(|(a, b)| *a + (r * b))
            .collect::<Vec<C::ScalarField>>();

        CycleFoldCommittedInstance::<C> { cmE, u, cmW, x }
    }

    /// NIFS.P is the consecutive combination of compute_cmT with fold_instances

    /// compute_cmT is part of the NIFS.P logic
    pub fn compute_cmT(
        cs_prover_params: &CS::ProverParams,
        r1cs: &R1CS<C::ScalarField>,
        w1: &Witness<C>,
        ci1: &RunningInstance<C>,
        w2: &Witness<C>,
        ci2: &CurrentInstance<C>,
    ) -> Result<(Vec<C::ScalarField>, C), Error> {
        let timer = start_timer!(|| "prepare variables");
        let z1: Vec<C::ScalarField> = [&[ci1.u], &ci1.x[..], &w1.QW[..]].concat();
        let z2: Vec<C::ScalarField> = [&[ci2.u], &ci2.x[..], &w2.QW[..]].concat();
        end_timer!(timer);

        let timer = start_timer!(|| "Compute T");
        // compute cross terms
        let T = C::ScalarField::compute_t(&r1cs.A.cuda, &r1cs.B.cuda, &r1cs.C.cuda, &z1, &z2);
        // let T = Self::compute_T(r1cs, ci1.u, ci2.u, &z1, &z2)?;
        end_timer!(timer);

        let timer = start_timer!(|| "Commit to T");
        // use r_T=0 since we don't need hiding property for cm(T)
        let cmT = CS::commit(cs_prover_params, &T, &C::ScalarField::zero())?;
        end_timer!(timer);

        Ok((T, cmT))
    }

    pub fn compute_cyclefold_cmT(
        cs_prover_params: &CS::ProverParams,
        r1cs: &R1CS<C::ScalarField>, // R1CS over C2.Fr=C1.Fq (here C=C2)
        w1: &Witness<C>,
        ci1: &CycleFoldCommittedInstance<C>,
        w2: &Witness<C>,
        ci2: &CycleFoldCommittedInstance<C>,
    ) -> Result<(Vec<C::ScalarField>, C), Error>
    where
        <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
    {
        let timer = start_timer!(|| "prepare variables");
        let z1: Vec<C::ScalarField> = [&[ci1.u], &ci1.x[..], &w1.QW[..]].concat();
        let z2: Vec<C::ScalarField> = [&[ci2.u], &ci2.x[..], &w2.QW[..]].concat();
        end_timer!(timer);

        let timer = start_timer!(|| "Compute T");
        // compute cross terms
        let T = C::ScalarField::compute_t(&r1cs.A.cuda, &r1cs.B.cuda, &r1cs.C.cuda, &z1, &z2);
        // let T = Self::compute_T(r1cs, ci1.u, ci2.u, &z1, &z2)?;
        end_timer!(timer);

        let timer = start_timer!(|| "Commit to T");
        // use r_T=0 since we don't need hiding property for cm(T)
        let cmT = CS::commit(cs_prover_params, &T, &C::ScalarField::zero())?;
        end_timer!(timer);

        Ok((T, cmT))
    }

    /// fold_instances is part of the NIFS.P logic described in
    /// [Nova](https://eprint.iacr.org/2021/370.pdf)'s section 4. It returns the folded Committed
    /// Instances and the Witness.
    pub fn fold_instances(
        r: C::ScalarField,
        w1: &Witness<C>,
        ci1: &RunningInstance<C>,
        w2: &Witness<C>,
        ci2: &CurrentInstance<C>,
        T: &[C::ScalarField],
        cmT: C,
    ) -> Result<(Witness<C>, RunningInstance<C>), Error> {
        let timer = start_timer!(|| "Fold witnesses");
        // fold witness
        // use r_T=0 since we don't need hiding property for cm(T)
        let w3 = NIFS::<C, CS>::fold_witness(r, w1, w2, T, C::ScalarField::zero())?;
        end_timer!(timer);

        let timer = start_timer!(|| "Fold instances");
        // fold committed instances
        let ci3 = NIFS::<C, CS>::fold_committed_instance(r, ci1, ci2, &cmT);
        end_timer!(timer);

        Ok((w3, ci3))
    }

    /// fold_instances is part of the NIFS.P logic described in
    /// [Nova](https://eprint.iacr.org/2021/370.pdf)'s section 4. It returns the folded Committed
    /// Instances and the Witness.
    pub fn fold_cf_instances(
        r: C::ScalarField,
        w1: &Witness<C>,
        ci1: &CycleFoldCommittedInstance<C>,
        w2: &Witness<C>,
        ci2: &CycleFoldCommittedInstance<C>,
        T: &[C::ScalarField],
        cmT: C,
    ) -> Result<(Witness<C>, CycleFoldCommittedInstance<C>), Error> {
        // fold witness
        // use r_T=0 since we don't need hiding property for cm(T)
        let w3 = NIFS::<C, CS>::fold_witness(r, w1, w2, T, C::ScalarField::zero())?;

        // fold committed instances
        let ci3 = NIFS::<C, CS>::fold_cf_committed_instance(r, ci1, ci2, &cmT);

        Ok((w3, ci3))
    }

    /// verify implements NIFS.V logic described in [Nova](https://eprint.iacr.org/2021/370.pdf)'s
    /// section 4. It returns the folded Committed Instance
    pub fn verify(
        // r comes from the transcript, and is a n-bit (N_BITS_CHALLENGE) element
        r: C::ScalarField,
        ci1: &RunningInstance<C>,
        ci2: &CurrentInstance<C>,
        cmT: &C,
    ) -> RunningInstance<C> {
        NIFS::<C, CS>::fold_committed_instance(r, ci1, ci2, cmT)
    }

    /// Verify committed folded instance (ci) relations. Notice that this method does not open the
    /// commitments, but just checks that the given committed instances (ci1, ci2) when folded
    /// result in the folded committed instance (ci3) values.
    pub fn verify_folded_instance(
        r: C::ScalarField,
        ci1: &RunningInstance<C>,
        ci2: &CurrentInstance<C>,
        ci3: &RunningInstance<C>,
        cmT: &C,
    ) -> Result<(), Error> {
        let expected = Self::fold_committed_instance(r, ci1, ci2, cmT);
        if ci3.cmE != expected.cmE
            || ci3.u != expected.u
            || ci3.cmQ != expected.cmQ
            || ci3.cmW != expected.cmW
            || ci3.x != expected.x
        {
            println!("{} {}", ci3.cmE, expected.cmE);
            println!("{} {}", ci3.u, expected.u);
            println!("{} {}", ci3.cmQ, expected.cmQ);
            println!("{} {}", ci3.cmW, expected.cmW);
            println!("{:?} {:?}", ci3.x, expected.x);
            return Err(Error::NotSatisfied);
        }
        Ok(())
    }

    pub fn prove_commitments(
        tr: &mut impl Transcript<C>,
        cs_prover_params: &CS::ProverParams,
        w: &Witness<C>,
        ci: &RunningInstance<C>,
        T: Vec<C::ScalarField>,
        cmT: &C,
    ) -> Result<[CS::Proof; 4], Error> {
        let cmE_proof = CS::prove(cs_prover_params, tr, &ci.cmE, &w.E, &w.rE, None)?;
        let cmQ_proof = CS::prove(cs_prover_params, tr, &ci.cmQ, &w.q(), &w.rQ, None)?;
        let cmW_proof = CS::prove(cs_prover_params, tr, &ci.cmW, &w.w(), &w.rW, None)?;
        let cmT_proof = CS::prove(cs_prover_params, tr, cmT, &T, &C::ScalarField::zero(), None)?; // cm(T) is committed with rT=0
        Ok([cmE_proof, cmQ_proof, cmW_proof, cmT_proof])
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
    use ark_ff::{BigInteger, PrimeField};
    use ark_grumpkin::{Fr, Projective};
    use ark_std::{ops::Mul, UniformRand};

    use crate::ccs::r1cs::tests::{get_test_r1cs, get_test_z};
    use crate::commitment::pedersen::{Params as PedersenParams, Pedersen};
    use crate::folding::nova::circuits::ChallengeGadget;
    use crate::folding::nova::traits::NovaR1CS;
    use crate::transcript::poseidon::{poseidon_test_config, PoseidonTranscript};
    use crate::MVM;

    #[allow(clippy::type_complexity)]
    pub(crate) fn prepare_simple_fold_inputs<C>() -> (
        PedersenParams<C>,
        PoseidonConfig<C::ScalarField>,
        R1CS<C::ScalarField>,
        Witness<C>,          // w1
        RunningInstance<C>,  // ci1
        Witness<C>,          // w2
        CurrentInstance<C>,  // ci2
        Witness<C>,          // w3
        RunningInstance<C>,  // ci3
        Vec<C::ScalarField>, // T
        C,                   // cmT
        Vec<bool>,           // r_bits
        C::ScalarField,      // r_Fr
    )
    where
        C: CurveGroup,
        C::Config: MSM<C>,
        <C as CurveGroup>::BaseField: PrimeField,
        C::ScalarField: Absorb + MVM,
    {
        let r1cs = get_test_r1cs();
        let z1 = get_test_z(3);
        let z2 = get_test_z(4);
        let (w1, x1) = r1cs.split_z(&z1);
        let (w2, x2) = r1cs.split_z(&z2);

        let w1 = Witness::<C>::new(w1.clone(), &r1cs);
        let w2 = Witness::<C>::new(w2.clone(), &r1cs);

        let mut rng = ark_std::test_rng();
        let (pedersen_params, _) = Pedersen::<C>::setup(&mut rng, r1cs.A.n_cols).unwrap();

        // compute committed instances
        let ci1 = w1
            .commit_running::<Pedersen<C>>(&pedersen_params, x1.clone())
            .unwrap();
        let ci2 = w2
            .commit_current::<Pedersen<C>>(&pedersen_params, x2.clone())
            .unwrap();

        // NIFS.P
        let (T, cmT) =
            NIFS::<C, Pedersen<C>>::compute_cmT(&pedersen_params, &r1cs, &w1, &ci1, &w2, &ci2)
                .unwrap();

        let poseidon_config = poseidon_test_config::<C::ScalarField>();

        let r_bits = ChallengeGadget::<C>::get_challenge_native(
            &poseidon_config,
            &ci1,
            &ci2,
            cmT,
        )
        .unwrap();
        let r_Fr = C::ScalarField::from_bigint(BigInteger::from_bits_le(&r_bits)).unwrap();

        let (w3, ci3) =
            NIFS::<C, Pedersen<C>>::fold_instances(r_Fr, &w1, &ci1, &w2, &ci2, &T, cmT).unwrap();

        (
            pedersen_params,
            poseidon_config,
            r1cs,
            w1,
            ci1,
            w2,
            ci2,
            w3,
            ci3,
            T,
            cmT,
            r_bits,
            r_Fr,
        )
    }

    // fold 2 dummy instances and check that the folded instance holds the relaxed R1CS relation
    #[test]
    fn test_nifs_fold_dummy() {
        let r1cs = get_test_r1cs::<Fr>();
        let z1 = get_test_z(3);
        let (w1, x1) = r1cs.split_z(&z1);

        let mut rng = ark_std::test_rng();
        let (pedersen_params, _) = Pedersen::<Projective>::setup(&mut rng, r1cs.A.n_cols).unwrap();

        let w_i = Witness::<Projective>::new(vec![Fr::zero(); w1.len()], &r1cs);
        let u_i = CurrentInstance::dummy(x1.len());
        let W_i = Witness::<Projective>::new(vec![Fr::zero(); w1.len()], &r1cs);
        let U_i = RunningInstance::dummy(x1.len());
        r1cs.check_relaxed_current_instance_relation(&w_i, &u_i)
            .unwrap();
        r1cs.check_relaxed_running_instance_relation(&W_i, &U_i)
            .unwrap();

        let r_Fr = Fr::from(3_u32);

        let (T, cmT) = NIFS::<Projective, Pedersen<Projective>>::compute_cmT(
            &pedersen_params,
            &r1cs,
            &W_i,
            &U_i,
            &w_i,
            &u_i,
        )
        .unwrap();
        let (W_i1, U_i1) = NIFS::<Projective, Pedersen<Projective>>::fold_instances(
            r_Fr, &W_i, &U_i, &w_i, &u_i, &T, cmT,
        )
        .unwrap();
        r1cs.check_relaxed_running_instance_relation(&W_i1, &U_i1)
            .unwrap();
    }

    // fold 2 instances into one
    #[test]
    fn test_nifs_one_fold() {
        let (pedersen_params, poseidon_config, r1cs, w1, ci1, w2, ci2, w3, ci3, T, cmT, _, r) =
            prepare_simple_fold_inputs();

        // NIFS.V
        let ci3_v = NIFS::<Projective, Pedersen<Projective>>::verify(r, &ci1, &ci2, &cmT);
        assert_eq!(ci3_v, ci3);

        // check that relations hold for the 2 inputted instances and the folded one
        r1cs.check_relaxed_running_instance_relation(&w1, &ci1)
            .unwrap();
        r1cs.check_relaxed_current_instance_relation(&w2, &ci2)
            .unwrap();
        r1cs.check_relaxed_running_instance_relation(&w3, &ci3)
            .unwrap();

        // check that folded commitments from folded instance (ci) are equal to folding the
        // use folded rE, rW to commit w3
        let ci3_expected = w3
            .commit_running::<Pedersen<Projective>>(&pedersen_params, ci3.x.clone())
            .unwrap();
        assert_eq!(ci3_expected.cmE, ci3.cmE);
        assert_eq!(ci3_expected.cmQ, ci3.cmQ);
        assert_eq!(ci3_expected.cmW, ci3.cmW);

        // next equalities should hold since we started from two cmE of zero-vector E's
        assert_eq!(ci3.cmE, cmT.mul(r));
        assert_eq!(w3.E, vec_scalar_mul(&T, &r));

        // NIFS.Verify_Folded_Instance:
        NIFS::<Projective, Pedersen<Projective>>::verify_folded_instance(r, &ci1, &ci2, &ci3, &cmT)
            .unwrap();

        // init Prover's transcript
        let mut transcript_p = PoseidonTranscript::<Projective>::new(&poseidon_config);
        // init Verifier's transcript
        let mut transcript_v = PoseidonTranscript::<Projective>::new(&poseidon_config);

        // prove the ci3.cmE, ci3.cmW, cmT commitments
        let cm_proofs = NIFS::<Projective, Pedersen<Projective>>::prove_commitments(
            &mut transcript_p,
            &pedersen_params,
            &w3,
            &ci3,
            T,
            &cmT,
        )
        .unwrap();

        // verify the ci3.cmE, ci3.cmW, cmT commitments
        assert_eq!(cm_proofs.len(), 4);
        Pedersen::<Projective>::verify(
            &pedersen_params,
            &mut transcript_v,
            &ci3.cmE,
            &cm_proofs[0].clone(),
        )
        .unwrap();
        Pedersen::<Projective>::verify(
            &pedersen_params,
            &mut transcript_v,
            &ci3.cmQ,
            &cm_proofs[1].clone(),
        )
        .unwrap();
        Pedersen::<Projective>::verify(
            &pedersen_params,
            &mut transcript_v,
            &ci3.cmW,
            &cm_proofs[2].clone(),
        )
        .unwrap();
        Pedersen::<Projective>::verify(
            &pedersen_params,
            &mut transcript_v,
            &cmT,
            &cm_proofs[3].clone(),
        )
        .unwrap();
    }

    #[test]
    fn test_nifs_fold_loop() {
        let r1cs = get_test_r1cs();
        let z = get_test_z(3);
        let (w, x) = r1cs.split_z(&z);

        let mut rng = ark_std::test_rng();
        let (pedersen_params, _) = Pedersen::<Projective>::setup(&mut rng, r1cs.A.n_cols).unwrap();

        // prepare the running instance
        let mut running_instance_w = Witness::<Projective>::new(w.clone(), &r1cs);
        let mut running_committed_instance = running_instance_w
            .commit_running::<Pedersen<Projective>>(&pedersen_params, x)
            .unwrap();

        r1cs.check_relaxed_running_instance_relation(
            &running_instance_w,
            &running_committed_instance,
        )
        .unwrap();

        let num_iters = 10;
        for i in 0..num_iters {
            // prepare the incoming instance
            let incoming_instance_z = get_test_z(i + 4);
            let (w, x) = r1cs.split_z(&incoming_instance_z);
            let incoming_instance_w = Witness::<Projective>::new(w.clone(), &r1cs);
            let incoming_committed_instance = incoming_instance_w
                .commit_current::<Pedersen<Projective>>(&pedersen_params, x)
                .unwrap();
            r1cs.check_relaxed_current_instance_relation(
                &incoming_instance_w,
                &incoming_committed_instance,
            )
            .unwrap();

            let r = Fr::rand(&mut rng); // folding challenge would come from the RO

            // NIFS.P
            let (T, cmT) = NIFS::<Projective, Pedersen<Projective>>::compute_cmT(
                &pedersen_params,
                &r1cs,
                &running_instance_w,
                &running_committed_instance,
                &incoming_instance_w,
                &incoming_committed_instance,
            )
            .unwrap();
            let (folded_w, _) = NIFS::<Projective, Pedersen<Projective>>::fold_instances(
                r,
                &running_instance_w,
                &running_committed_instance,
                &incoming_instance_w,
                &incoming_committed_instance,
                &T,
                cmT,
            )
            .unwrap();

            // NIFS.V
            let folded_committed_instance = NIFS::<Projective, Pedersen<Projective>>::verify(
                r,
                &running_committed_instance,
                &incoming_committed_instance,
                &cmT,
            );

            r1cs.check_relaxed_running_instance_relation(&folded_w, &folded_committed_instance)
                .unwrap();

            // set running_instance for next loop iteration
            running_instance_w = folded_w;
            running_committed_instance = folded_committed_instance;
        }
    }
}
