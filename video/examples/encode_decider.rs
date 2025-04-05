#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(clippy::upper_case_acronyms)]

use std::fs::File;
use std::io::{BufReader, Read};
use std::marker::PhantomData;
use std::{path::Path, sync::Arc};

use ark_bn254::Bn254;
use ark_crypto_primitives::crh::poseidon::CRH;
use ark_crypto_primitives::crh::CRHScheme;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_groth16::Groth16;
use ark_serialize::CanonicalSerialize;
use ark_snark::SNARK;
use ark_std::{add_to_trace, end_timer, start_timer};
use folding_schemes::CastSlice;
use rand::thread_rng;
use rayon::prelude::*;
use video::decider::{Decider, DeciderEthCircuit};
use video::edit::constraints::NoOp;
use video::utils::srs_size;
use video::ExternalInputs;
use video::{
    encode::{MacroblockType, Matrix},
    griffin::params::GriffinParams,
    EditEncodeCircuit, EncodeConfig,
};

use ark_bn254::{constraints::GVar, Fq, Fr, G1Projective as Projective};
use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
use ark_grumpkin::{constraints::GVar as GVar2, Projective as Projective2};
use folding_schemes::{
    commitment::pedersen::Pedersen, folding::nova::Nova,
    transcript::poseidon::poseidon_test_config, FoldingScheme,
};

type Op = NoOp;

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
    let input_path = Path::new(env!("DATA_PATH")).join("bunny");
    let output_path = Path::new(env!("DATA_PATH")).join("bunny");
    let types_file = File::open(output_path.join("type_enc"))?;
    let orig_y_file = File::open(input_path.join("orig_y_enc"))?;
    let orig_u_file = File::open(input_path.join("orig_u_enc"))?;
    let orig_v_file = File::open(input_path.join("orig_v_enc"))?;
    let pred_y_file = File::open(output_path.join("pred_y_enc"))?;
    let pred_u_file = File::open(output_path.join("pred_u_enc"))?;
    let pred_v_file = File::open(output_path.join("pred_v_enc"))?;
    let result_y_file = File::open(output_path.join("coeff_y_enc"))?;
    let result_u_file = File::open(output_path.join("coeff_u_enc"))?;
    let result_v_file = File::open(output_path.join("coeff_v_enc"))?;

    // Calculate the total number of blocks based on file size
    let types_len = types_file.metadata()?.len() as usize;
    let num_blocks = types_len / 6;

    // Create buffered readers once
    let mut types_reader = BufReader::new(types_file);
    let mut orig_y_reader = BufReader::new(orig_y_file);
    let mut orig_u_reader = BufReader::new(orig_u_file);
    let mut orig_v_reader = BufReader::new(orig_v_file);
    let mut pred_y_reader = BufReader::new(pred_y_file);
    let mut pred_u_reader = BufReader::new(pred_u_file);
    let mut pred_v_reader = BufReader::new(pred_v_file);
    let mut result_y_reader = BufReader::new(result_y_file);
    let mut result_u_reader = BufReader::new(result_u_file);
    let mut result_v_reader = BufReader::new(result_v_file);

    let rng = &mut thread_rng();
    let sk = Fq::rand(rng);
    let vk = Projective2::generator() * sk;

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
            blocks: vec![Default::default(); BLOCKS_PER_STEP],
            predictions: vec![Default::default(); BLOCKS_PER_STEP],
            outputs: vec![Default::default(); BLOCKS_PER_STEP],
            encode_configs: vec![Default::default(); BLOCKS_PER_STEP],
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
            E: None,
            cf_E: None,
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

    let num_steps = num_blocks / BLOCKS_PER_STEP;
    // let num_steps = 2;

    let (circuit, num_steps, initial_state, last_state, running_instance, incoming_instance) = {
        let params = (pp, vp);

        let mut types = [0u8; BLOCKS_PER_STEP * 6];
        let mut orig_y = [0u8; BLOCKS_PER_STEP * 256];
        let mut orig_u = [0u8; BLOCKS_PER_STEP * 64];
        let mut orig_v = [0u8; BLOCKS_PER_STEP * 64];
        let mut pred_y = [0u8; BLOCKS_PER_STEP * 256];
        let mut pred_u = [0u8; BLOCKS_PER_STEP * 64];
        let mut pred_v = [0u8; BLOCKS_PER_STEP * 64];
        let mut result_y = [0u8; BLOCKS_PER_STEP * 256];
        let mut result_u = [0u8; BLOCKS_PER_STEP * 64];
        let mut result_v = [0u8; BLOCKS_PER_STEP * 64];

        let initial_state = vec![Fr::zero(), Fr::zero()];

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone()).unwrap();

        // compute a step of the IVC
        let timer = start_timer!(|| "IVC::prove");
        for i in 0..num_steps {
            let timer = start_timer!(|| format!("Read data for step {}", i));

            // Read each component from disk
            types_reader.read_exact(&mut types)?;
            orig_y_reader.read_exact(&mut orig_y)?;
            orig_u_reader.read_exact(&mut orig_u)?;
            orig_v_reader.read_exact(&mut orig_v)?;
            pred_y_reader.read_exact(&mut pred_y)?;
            pred_u_reader.read_exact(&mut pred_u)?;
            pred_v_reader.read_exact(&mut pred_v)?;
            result_y_reader.read_exact(&mut result_y)?;
            result_u_reader.read_exact(&mut result_u)?;
            result_v_reader.read_exact(&mut result_v)?;

            let blocks = orig_y
                .par_chunks_exact(256)
                .zip(orig_u.par_chunks_exact(64).zip(orig_v.par_chunks_exact(64)))
                .map(|(orig_y, (orig_u, orig_v))| {
                    (
                        Matrix::from_vec(orig_y.to_vec()),
                        Matrix::from_vec(orig_u.to_vec()),
                        Matrix::from_vec(orig_v.to_vec()),
                    )
                })
                .collect::<Vec<_>>();
            let predictions = pred_y
                .par_chunks_exact(256)
                .zip(pred_u.par_chunks_exact(64).zip(pred_v.par_chunks_exact(64)))
                .map(|(pred_y, (pred_u, pred_v))| {
                    (
                        Matrix::from_vec(pred_y.to_vec()),
                        Matrix::from_vec(pred_u.to_vec()),
                        Matrix::from_vec(pred_v.to_vec()),
                    )
                })
                .collect::<Vec<_>>();
            let outputs = result_y
                .cast::<i8>()
                .par_chunks_exact(256)
                .zip(
                    result_u
                        .cast::<i8>()
                        .par_chunks_exact(64)
                        .zip(result_v.cast::<i8>().par_chunks_exact(64)),
                )
                .map(|(result_y, (result_u, result_v))| {
                    (
                        Matrix::from_vec(result_y.to_vec()),
                        Matrix::from_vec(result_u.to_vec()),
                        Matrix::from_vec(result_v.to_vec()),
                    )
                })
                .collect::<Vec<_>>();
            let configs = types
                .par_chunks_exact(6)
                .map(|types| EncodeConfig {
                    y_is_8x8: match types[2] {
                        1 => panic!(),
                        _ => false,
                    },
                    slice_is_intra: types[0] == 2,
                    qp: types[4] as usize,
                    qpc: types[5] as usize,
                    mb_type: match types[1] {
                        1 => MacroblockType::P16x16,
                        2 => MacroblockType::P16x8,
                        3 => MacroblockType::P8x16,
                        8 => MacroblockType::P8x8,
                        9 => MacroblockType::I4x4,
                        10 => MacroblockType::I16x16,
                        13 => MacroblockType::I8x8,
                        _ => panic!("unknown type"),
                    },
                })
                .collect::<Vec<_>>();
            end_timer!(timer);

            let timer = start_timer!(|| format!("Nova::prove_step {}", i));
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks,
                        predictions,
                        outputs,
                        encode_configs: configs,
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

        let i = folding_scheme.i;
        let z_0 = folding_scheme.z_0.clone();
        let z_i = folding_scheme.z_i.clone();
        let U_i = folding_scheme.U_i.clone();

        let circuit = DeciderEthCircuit::<Projective, GVar, Projective2, GVar2>::from_nova(
            folding_scheme,
            params,
            vk,
            sigma,
        )?;

        let u_i = circuit.u_i.clone().unwrap();

        (circuit, i, z_0, z_i, U_i, u_i)
    };

    // decider proof generation
    let proof = Decider::prove(decider_pp, rng, circuit).unwrap();

    let mut v = vec![];
    decider_vp.serialize_compressed(&mut v).unwrap();
    vk.serialize_compressed(&mut v).unwrap();

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
