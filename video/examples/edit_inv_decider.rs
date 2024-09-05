use std::marker::PhantomData;
use std::{path::Path, sync::Arc};

use ark_bn254::Bn254;
use ark_crypto_primitives::crh::poseidon::CRH;
use ark_crypto_primitives::crh::CRHScheme;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_groth16::Groth16;
use ark_snark::SNARK;
use ark_std::{add_to_trace, end_timer, start_timer};
use rand::thread_rng;
use video::decider::{Decider, DeciderEthCircuit};
use video::griffin::params::GriffinParams;
use video::utils::srs_size;
use video::{parse_prover_data, EditEncodeCircuit, ExternalInputs};

use ark_bn254::{constraints::GVar, Fq, Fr, G1Projective as Projective};
use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
use ark_grumpkin::{constraints::GVar as GVar2, Projective as Projective2};
use folding_schemes::{
    commitment::pedersen::Pedersen, folding::nova::Nova,
    transcript::poseidon::poseidon_test_config, FoldingScheme,
};

type Op = video::edit::constraints::InvertColor;

const BLOCKS_PER_STEP: usize = 256;

type NOVA = Nova<
    Projective,
    GVar,
    Projective2,
    GVar2,
    EditEncodeCircuit<Fr, Op>,
    Pedersen<Projective>,
    Pedersen<Projective2>,
>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rng = &mut thread_rng();
    let sk = Fq::rand(rng);
    let vk = Projective2::generator() * sk;

    const W: usize = 352;
    const H: usize = 288;

    let (blocks, predictions, outputs, configs) = parse_prover_data(
        Path::new(env!("DATA_PATH")).join("foreman"),
        Path::new(env!("DATA_PATH")).join("foreman_inv"),
        None,
    )?;
    // let num_steps = 2;
    let num_steps = blocks.len() / BLOCKS_PER_STEP;

    let F_circuit = EditEncodeCircuit {
        _e: PhantomData,
        griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
    };

    let poseidon_config = poseidon_test_config();

    println!("Prepare Nova ProverParams & VerifierParams");
    let (pp, vp) = NOVA::preprocess(
        &poseidon_config,
        &F_circuit,
        rng,
        &ExternalInputs {
            blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
            predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
            outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
            encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
            edit_configs: vec![(); BLOCKS_PER_STEP],
        },
    )
    .unwrap();

    let pk = Groth16::<Bn254>::generate_random_parameters_with_reduction(
        DeciderEthCircuit::<Projective, GVar, Projective2, GVar2> {
            _gc1: std::marker::PhantomData,
            _gc2: std::marker::PhantomData,
            r1cs: vp.r1cs.clone(),
            cf_r1cs: vp.cf_r1cs.clone(),
            cf_pedersen_params: pp.cf_cs_params.clone(),
            poseidon_config: poseidon_config.clone(),
            i: None,
            z_0: Some(vec![Fr::rand(rng), Fr::rand(rng)]),
            u_i: None,
            U_i: None,
            W_i1: None,
            cmT: None,
            r: None,
            cf_U_i: None,
            cf_W_i: None,
            sigma: (Fr::rand(rng), Fq::rand(rng)),
            vk: Projective2::rand(rng),
            h1: Fr::rand(rng),
            h2: Fr::rand(rng),
        },
        vec![
            (
                &pp.cs_params.generators[..vp.r1cs.q],
                pp.cs_params.h.into_affine(),
            ),
            (
                &pp.cs_params.generators[..vp.r1cs.A.n_cols - 1 - vp.r1cs.l - vp.r1cs.q],
                pp.cs_params.h.into_affine(),
            ),
            (
                &pp.cs_params.generators[..vp.r1cs.A.n_rows],
                pp.cs_params.h.into_affine(),
            ),
        ],
        rng,
    )
    .unwrap();
    add_to_trace!(|| format!("SRS size"), || format!(
        "{}",
        srs_size(&pk, &pp.cs_params)
    ));
    if std::env::var("SETUP_ONLY").is_ok() {
        return Ok(());
    }

    let decider_vp = Groth16::<Bn254>::process_vk(&pk.vk).unwrap();
    let decider_pp = pk;

    let (circuit, num_steps, initial_state, last_state, running_instance, incoming_instance) = {
        let params = (pp, vp);

        let initial_state = vec![Fr::zero(), Fr::zero()];

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone()).unwrap();

        // compute a step of the IVC
        let timer = start_timer!(|| "IVC::prove");
        for i in 0..num_steps {
            let timer = start_timer!(|| format!("Nova::prove_step {}", i));
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                        predictions: predictions[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP]
                            .to_vec(),
                        outputs: outputs[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                        encode_configs: configs[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP]
                            .to_vec(),
                        edit_configs: vec![(); BLOCKS_PER_STEP],
                    },
                )
                .unwrap();
            end_timer!(timer);
        }
        end_timer!(timer);

        let last_state = folding_scheme.state();
        add_to_trace!(
            || format!("state at last ({}-th) step", num_steps),
            || format!("{:?}", last_state)
        );

        {
            let (running_instance, incoming_instance, cyclefold_instance) =
                folding_scheme.instances();

            println!("Run the Nova's IVC verifier");
            NOVA::verify(
                &params.1,
                initial_state.clone(),
                last_state.clone(), // latest state
                Fr::from(num_steps as u32),
                running_instance.clone(),
                incoming_instance.clone(),
                cyclefold_instance,
            )?;
        }

        let vk = Projective2::generator() * sk;
        let (px, py) = {
            let p = vk.into_affine();
            p.xy().unwrap_or((Fr::zero(), Fr::zero()))
        };
        let sigma = {
            let r = Fq::rand(rng);
            let rx = (Projective2::generator() * r)
                .into_affine()
                .x()
                .unwrap_or_default();
            let e = CRH::evaluate(&poseidon_config, [rx, px, py, folding_scheme.z_i[0]])?;
            (
                rx,
                r + sk * Fq::from_le_bytes_mod_order(&e.into_bigint().to_bytes_le()),
            )
        };

        let circuit = DeciderEthCircuit::<Projective, GVar, Projective2, GVar2>::from_nova(
            &folding_scheme,
            params,
            vk,
            sigma,
        )?;

        (
            circuit,
            folding_scheme.i,
            folding_scheme.z_0.clone(),
            folding_scheme.z_i.clone(),
            folding_scheme.U_i.clone(),
            folding_scheme.u_i.clone(),
        )
    };

    // decider proof generation
    let proof = Decider::prove(decider_pp, rng, circuit).unwrap();

    // decider proof verification
    let start = start_timer!(|| "Verify");
    let verified = Decider::verify(
        decider_vp,
        vk,
        num_steps,
        initial_state,
        last_state[1],
        &running_instance,
        &incoming_instance,
        proof,
    )
    .unwrap();
    assert!(verified);

    end_timer!(start);

    Ok(())
}
