#![feature(test)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(clippy::upper_case_acronyms)]

use std::{
    fs::File,
    io::{BufReader, Read},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use crate::encode::traits::CastSlice;
use ark_crypto_primitives::sponge::Absorb;
use ark_ff::{BigInteger, PrimeField};
use ark_r1cs_std::{
    alloc::AllocVar,
    boolean::{AllocatedBool, Boolean},
    fields::{
        fp::{AllocatedFp, FpVar},
        FieldVar,
    },
    R1CSVar,
};
use ark_relations::r1cs::{
    ConstraintSystem, ConstraintSystemRef, LinearCombination, SynthesisError
    , Variable,
};
use ark_std::{end_timer, start_timer};
use edit::constraints::{EditConfig, EditConfigVar, EditGadget};
use encode::constants::{OFFSET_BITS, Q_BITS_4};
use folding_schemes::{
    frontend::{FCircuit, LookupArgument, LookupArgumentRef},
    Error,
};
use rayon::prelude::*;
use var::I64Var;

use crate::{
    encode::{
        constants::{BASE_OFFSET_BP_SLICE, BASE_OFFSET_I_SLICE},
        constraints::MatrixVar,
        MacroblockType, Matrix,
    },
    griffin::{
        constraints::{GriffinCircuit, PermCircuit},
        griffin::{Griffin, Permutation},
        params::GriffinParams,
    },
};

pub mod decider;
pub mod edit;
pub mod encode;
pub mod griffin;
pub mod utils;
pub mod var;

const SCALES: [[u64; 6]; 3] = [
    [13107, 11916, 10082, 9362, 8192, 7282],
    [8066, 7490, 6554, 5825, 5243, 4559],
    [5243, 4660, 4194, 3647, 3355, 2893],
];

pub const MB_BITS: usize = 8;
pub const COEFF_BITS: usize = 8;

#[derive(Clone, Debug)]
pub struct EncodeConfig {
    pub mb_type: MacroblockType,
    pub slice_is_intra: bool,
    pub y_is_8x8: bool,
    pub qp: usize,
    pub qpc: usize,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            mb_type: MacroblockType::default(),
            slice_is_intra: false,
            y_is_8x8: false,
            qp: 51,
            qpc: 51,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EditEncodeCircuit<F: PrimeField + Absorb, E: EditGadget> {
    pub _e: PhantomData<E>,
    pub griffin_params: Arc<GriffinParams<F>>,
}

fn bit_decompose(mut x: usize, n: usize) -> Vec<bool> {
    let mut v = vec![];
    for _ in 0..n {
        v.push(x & 1 == 1);
        x >>= 1;
    }
    v
}

fn select_scale<F: PrimeField>(
    v: &[Boolean<F>],
    scales: &[u64; 6],
) -> Result<I64Var<F>, SynthesisError> {
    let [a, b, c, d, e, f] = scales.map(|x| I64Var::constant(x));
    v[0].select(
        &v[1].select(&d, &v[2].select(&f, &b)?)?,
        &v[1].select(&c, &v[2].select(&e, &a)?)?,
    )
}

impl<F: PrimeField + Absorb, E: EditGadget> EditEncodeCircuit<F, E> {
    fn process_macroblock<const DUMMY: bool>(
        cs: ConstraintSystemRef<F>,
        la: LookupArgumentRef<F>,
        griffin: &GriffinCircuit<F>,
        encode_config: &EncodeConfig,
        edit_config: &E::Cfg,
        (y_block, u_block, v_block): &(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>),
        (y_pred, u_pred, v_pred): &(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>),
        (y_output, u_output, v_output): &(Matrix<i8, 16, 16>, Matrix<i8, 8, 8>, Matrix<i8, 8, 8>),
    ) -> Result<
        (
            (Option<Variable>, F),
            (Option<Variable>, F),
            (Option<Variable>, bool),
        ),
        SynthesisError,
    > {
        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let y_block_var = MatrixVar::new_committed(cs.clone(), || Ok(y_block))?;
        let u_block_var = MatrixVar::new_committed(cs.clone(), || Ok(u_block))?;
        let v_block_var = MatrixVar::new_committed(cs.clone(), || Ok(v_block))?;

        let edit_config = E::CfgVar::new_witness(cs.clone(), || Ok(edit_config))?;

        let qp_over_6_bits = Vec::<Boolean<_>>::new_witness(cs.clone(), || {
            Ok(bit_decompose(encode_config.qp / 6, 4))
        })?;
        let qpc_over_6_bits = Vec::<Boolean<_>>::new_witness(cs.clone(), || {
            Ok(bit_decompose(encode_config.qpc / 6, 4))
        })?;
        let qp_mod_6_bits = Vec::<Boolean<_>>::new_witness(cs.clone(), || {
            Ok(bit_decompose(encode_config.qp % 6, 3))
        })?;
        let qpc_mod_6_bits = Vec::<Boolean<_>>::new_witness(cs.clone(), || {
            Ok(bit_decompose(encode_config.qpc % 6, 3))
        })?;
        let two_to_qp_over_6 = I64Var::one().double()?.pow_le(&qp_over_6_bits)?;
        let two_to_qpc_over_6 = I64Var::one().double()?.pow_le(&qpc_over_6_bits)?;
        let scale = (
            select_scale(&qp_mod_6_bits, &SCALES[0])?,
            select_scale(&qp_mod_6_bits, &SCALES[1])?,
            select_scale(&qp_mod_6_bits, &SCALES[2])?,
        );
        let scalec = (
            select_scale(&qpc_mod_6_bits, &SCALES[0])?,
            select_scale(&qpc_mod_6_bits, &SCALES[1])?,
            select_scale(&qpc_mod_6_bits, &SCALES[2])?,
        );

        let is_intra = Boolean::new_witness(cs.clone(), || Ok(encode_config.slice_is_intra))?;

        let base_offset = is_intra.select(
            &I64Var::constant((BASE_OFFSET_I_SLICE)),
            &I64Var::constant((BASE_OFFSET_BP_SLICE)),
        )?;
        let offset = &base_offset * &two_to_qp_over_6 * (1 << (Q_BITS_4 - OFFSET_BITS));
        let offset_c = &base_offset * &two_to_qpc_over_6 * (1 << (Q_BITS_4 - OFFSET_BITS));

        let is_i16x16 = Boolean::new_witness(cs.clone(), || {
            Ok(encode_config.mb_type == MacroblockType::I16x16)
        })?;
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Variable creation", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let (y, u, v) = E::edit_circuit(&y_block_var, &u_block_var, &v_block_var, &edit_config)?;
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Edit", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        let (y_has_constant_encoding, u_has_constant_encoding, v_has_constant_encoding) =
            E::result_has_constant_encoding();

        let y_pred_var = if y_has_constant_encoding {
            MatrixVar::new_constant(cs.clone(), y_pred)?
        } else {
            MatrixVar::new_committed(cs.clone(), || Ok(y_pred))?
        };
        let u_pred_var = if u_has_constant_encoding {
            MatrixVar::new_constant(cs.clone(), u_pred)?
        } else {
            MatrixVar::new_committed(cs.clone(), || Ok(u_pred))?
        };
        let v_pred_var = if v_has_constant_encoding {
            MatrixVar::new_constant(cs.clone(), v_pred)?
        } else {
            MatrixVar::new_committed(cs.clone(), || Ok(v_pred))?
        };

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let y_output_var = if y_has_constant_encoding {
            MatrixVar::new_constant(cs.clone(), y_output)?
        } else {
            y.encode_luma_4x4::<DUMMY>(
                cs.clone(),
                la.clone(),
                &y_pred_var,
                &is_i16x16,
                &offset,
                &two_to_qp_over_6,
                &scale,
            )?
        };
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Encode Y", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let u_output_var = if u_has_constant_encoding {
            MatrixVar::new_constant(cs.clone(), u_output)?
        } else {
            u.encode_chroma::<DUMMY>(
                cs.clone(),
                la.clone(),
                &u_pred_var,
                &offset_c,
                &two_to_qpc_over_6,
                &scalec,
            )?
        };
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Encode U", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let v_output_var = if v_has_constant_encoding {
            MatrixVar::new_constant(cs.clone(), v_output)?
        } else {
            v.encode_chroma::<DUMMY>(
                cs.clone(),
                la.clone(),
                &v_pred_var,
                &offset_c,
                &two_to_qpc_over_6,
                &scalec,
            )?
        };
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Encode V", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let s1 = F::from(1 << MB_BITS);
        let s2 = F::from(1 << COEFF_BITS);
        let x = vec![]
            .into_iter()
            .chain(y_block_var.0)
            .chain(u_block_var.0)
            .chain(v_block_var.0)
            .collect::<Vec<_>>()
            .chunks(F::MODULUS_BIT_SIZE as usize / MB_BITS)
            .map(|c| {
                let mut r = FpVar::zero();
                for v in c {
                    r = r * s1 + v.to_fpvar();
                }
                r
            })
            .collect::<Vec<_>>();
        let h1 = griffin.hash(&x)?;
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Partial hash 1", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let x = vec![]
            .into_iter()
            .chain(y_pred_var.0)
            .chain(u_pred_var.0)
            .chain(v_pred_var.0)
            .collect::<Vec<_>>()
            .chunks(F::MODULUS_BIT_SIZE as usize / MB_BITS)
            .map(|c| {
                let mut r = FpVar::zero();
                for v in c {
                    r = r * s1 + v.to_fpvar();
                }
                r
            })
            .chain(
                vec![]
                    .into_iter()
                    .chain(y_output_var.0)
                    .chain(u_output_var.0)
                    .chain(v_output_var.0)
                    .collect::<Vec<_>>()
                    .chunks(F::MODULUS_BIT_SIZE as usize / COEFF_BITS)
                    .map(|c| {
                        let mut r = FpVar::zero();
                        for v in c {
                            r = r * s2 + v.to_fpvar() + F::from(128);
                        }
                        r
                    }),
            )
            .chain({
                vec![Boolean::le_bits_to_fp(
                    &[
                        vec![is_intra, is_i16x16],
                        qp_over_6_bits,
                        qpc_over_6_bits,
                        qp_mod_6_bits,
                        qpc_mod_6_bits,
                    ]
                    .concat(),
                )?]
            })
            .chain(edit_config.compactify())
            .collect::<Vec<_>>();
        let keep = edit_config.should_keep_circuit()?;
        let h2 = keep.select(&griffin.hash(&x)?, &FpVar::zero())?;
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Partial hash 2", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        let h1 = match h1 {
            FpVar::Var(h1) => (Some(h1.variable), h1.value()?),
            FpVar::Constant(h1) => (None, h1),
        };
        let h2 = match h2 {
            FpVar::Var(h2) => (Some(h2.variable), h2.value()?),
            FpVar::Constant(h2) => (None, h2),
        };
        let keep = match keep {
            Boolean::Var(keep) => (Some(keep.variable()), keep.value()?),
            Boolean::Constant(keep) => (None, keep),
        };

        Ok((h1, h2, keep))
    }
}

pub struct ExternalInputs<EditConfig: Default + Clone> {
    pub blocks: Vec<(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>)>,
    pub predictions: Vec<(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>)>,
    pub outputs: Vec<(Matrix<i8, 16, 16>, Matrix<i8, 8, 8>, Matrix<i8, 8, 8>)>,
    pub encode_configs: Vec<EncodeConfig>,
    pub edit_configs: Vec<EditConfig>,
}

impl<F: PrimeField + Absorb, E: EditGadget> FCircuit<F> for EditEncodeCircuit<F, E> {
    type Params = Self;
    type ExternalInputs = ExternalInputs<E::Cfg>;

    fn new(params: Self::Params) -> Self {
        params
    }

    fn state_len(&self) -> usize {
        2
    }

    /// computes the next state values in place, assigning z_{i+1} into z_i, and computing the new
    /// z_{i+1}
    fn step_native(
        &self,
        i: usize,
        z_i: Vec<F>,
        external_inputs: &Self::ExternalInputs,
    ) -> Result<Vec<F>, Error> {
        let Self::ExternalInputs {
            blocks,
            predictions,
            outputs,
            encode_configs,
            edit_configs,
        } = external_inputs;
        let griffin = Griffin::new(&self.griffin_params);
        let mut h1s = blocks
            .into_par_iter()
            .map(|(y_block, u_block, v_block)| {
                griffin.hash(
                    &vec![]
                        .into_iter()
                        .chain(y_block.iter())
                        .chain(u_block.iter())
                        .chain(v_block.iter())
                        .copied()
                        .collect::<Vec<_>>()
                        .chunks(F::MODULUS_BIT_SIZE as usize / MB_BITS)
                        .map(F::from_be_bytes_mod_order)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<F>>();
        h1s.push(z_i[0]);

        let mut h2s = predictions
            .into_par_iter()
            .zip(outputs.par_iter())
            .zip(encode_configs.par_iter())
            .zip(edit_configs.par_iter())
            .map(
                |(
                    (((y_pred, u_pred, v_pred), (y_output, u_output, v_output)), encode_config),
                    edit_config,
                )| {
                    let is_intra = encode_config.slice_is_intra;
                    let is_i16x16 = encode_config.mb_type == MacroblockType::I16x16;

                    if !edit_config.should_keep_native() {
                        return F::zero();
                    }

                    griffin.hash(
                        &vec![]
                            .into_iter()
                            .chain(y_pred.iter())
                            .chain(u_pred.iter())
                            .chain(v_pred.iter())
                            .copied()
                            .collect::<Vec<_>>()
                            .chunks(F::MODULUS_BIT_SIZE as usize / MB_BITS)
                            .chain(
                                vec![]
                                    .into_iter()
                                    .chain(y_output.iter())
                                    .chain(u_output.iter())
                                    .chain(v_output.iter())
                                    .map(|&i| u8::try_from(i as i16 + 128))
                                    .collect::<Result<Vec<_>, _>>()
                                    .unwrap()
                                    .chunks(F::MODULUS_BIT_SIZE as usize / COEFF_BITS),
                            )
                            .map(F::from_be_bytes_mod_order)
                            .chain({
                                let mut v = (encode_config.qpc % 6) as u64;
                                v = v * 8 + (encode_config.qp % 6) as u64;
                                v = v * 16 + (encode_config.qpc / 6) as u64;
                                v = v * 16 + (encode_config.qp / 6) as u64;
                                v = v * 2 + is_i16x16 as u64;
                                v = v * 2 + is_intra as u64;
                                vec![F::from(v)]
                            })
                            .chain(edit_config.compactify())
                            .collect::<Vec<_>>(),
                    )
                },
            )
            .collect::<Vec<F>>();
        h2s.push(z_i[1]);

        let h1 = griffin.hash(&h1s);
        let h2 = if edit_configs.iter().any(|cfg| cfg.should_keep_native()) {
            griffin.hash(&h2s)
        } else {
            z_i[1]
        };

        Ok(vec![h1, h2])
    }

    /// generates the constraints for the step of F for the given z_i
    fn generate_step_constraints(
        &self,
        cs: ConstraintSystemRef<F>,
        la: LookupArgumentRef<F>,
        i: usize,
        z_i: Vec<FpVar<F>>,
        external_inputs: &Self::ExternalInputs,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let Self::ExternalInputs {
            blocks,
            predictions,
            outputs,
            encode_configs,
            edit_configs,
        } = external_inputs;

        let griffin = GriffinCircuit::new(&self.griffin_params);
        let table = (0u32..(1 << MB_BITS)).map(F::from).collect::<Vec<_>>();
        la.set_table(table.clone());

        let mode = cs.borrow().unwrap().mode;

        let (w_offset0, q_offset0, lc_offset0) = {
            let cs = cs.borrow().unwrap();
            (
                cs.num_witness_variables,
                cs.num_committed_variables,
                cs.num_linear_combinations,
            )
        };

        let timer = start_timer!(|| "Synthesize dummy circuit");
        let (w_offset, q_offset, lc_offset, constraints) = {
            let la = LookupArgument::new_ref();
            la.set_table(table.clone());
            let cs = ConstraintSystemRef::new(ConstraintSystem::new_offset(0, 0, 0));
            cs.set_mode(mode);

            let _ = Self::process_macroblock::<true>(
                cs.clone(),
                la,
                &griffin,
                &encode_configs[0],
                &edit_configs[0],
                &blocks[0],
                &predictions[0],
                &outputs[0],
            )?;
            let cs = cs.into_inner().unwrap();

            (
                cs.num_witness_variables,
                cs.num_committed_variables,
                cs.num_linear_combinations,
                cs.num_constraints,
            )
        };
        end_timer!(timer);

        let timer = start_timer!(|| "Synthesize partial circuits");
        let result = blocks
            .par_iter()
            .zip(predictions.par_iter())
            .zip(outputs.par_iter())
            .zip(encode_configs.par_iter())
            .zip(edit_configs.par_iter())
            .enumerate()
            .map(
                |(j, ((((block, prediction), output), encode_config), edit_config))| {
                    let la: LookupArgumentRef<F> = LookupArgument::new_ref();
                    la.set_table(table.clone());
                    let cs = ConstraintSystemRef::new(ConstraintSystem::new_offset(
                        w_offset0 + w_offset * j,
                        q_offset0 + q_offset * j,
                        lc_offset0 + lc_offset * j,
                    ));
                    cs.set_mode(mode);

                    let (h1, h2, keep) = Self::process_macroblock::<false>(
                        cs.clone(),
                        la,
                        &griffin,
                        &encode_config,
                        &edit_config,
                        &block,
                        &prediction,
                        &output,
                    )?;

                    let cs = cs.into_inner().unwrap();
                    Ok((
                        (
                            cs.witness_assignment,
                            cs.committed_assignment,
                            cs.lc_map,
                            cs.a_constraints,
                            cs.b_constraints,
                            cs.c_constraints,
                        ),
                        h1,
                        h2,
                        keep,
                    ))
                },
            )
            .collect::<Result<Vec<_>, SynthesisError>>()?;
        end_timer!(timer);

        let timer = start_timer!(|| "Merge partial circuits");
        {
            let cs = cs.borrow_mut().unwrap();
            cs.num_committed_variables += q_offset * blocks.len();
            cs.num_witness_variables += w_offset * blocks.len();
            cs.num_constraints += constraints * blocks.len();
            cs.num_linear_combinations += lc_offset * blocks.len();
            cs.committed_assignment.reserve(q_offset * blocks.len());
            cs.witness_assignment.reserve(w_offset * blocks.len());
        }

        let mut h1s = vec![];
        let mut h2s = vec![];
        let mut keeps = vec![];
        for (
            (
                witness_assignment,
                committed_assignment,
                lc_map,
                a_constraints,
                b_constraints,
                c_constraints,
            ),
            h1,
            h2,
            keep,
        ) in result
        {
            {
                let cs = cs.borrow_mut().unwrap();
                cs.witness_assignment.extend_from_slice(&witness_assignment);
                cs.committed_assignment
                    .extend_from_slice(&committed_assignment);
                cs.lc_map.extend(lc_map);
                cs.a_constraints.extend_from_slice(&a_constraints);
                cs.b_constraints.extend_from_slice(&b_constraints);
                cs.c_constraints.extend_from_slice(&c_constraints);
            }
            let h1 = match h1.0 {
                Some(var) => FpVar::Var(AllocatedFp::new(Some(h1.1), var, cs.clone())),
                None => FpVar::Constant(h1.1),
            };
            let h2 = match h2.0 {
                Some(var) => FpVar::Var(AllocatedFp::new(Some(h2.1), var, cs.clone())),
                None => FpVar::Constant(h2.1),
            };
            let keep = match keep.0 {
                Some(var) => Boolean::Var(AllocatedBool::new(Some(keep.1), var, cs.clone())),
                None => Boolean::Constant(keep.1),
            };
            h1s.push(h1);
            h2s.push(h2);
            keeps.push(keep);
        }
        h1s.push(z_i[0].clone());
        h2s.push(z_i[1].clone());
        end_timer!(timer);

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let timer = start_timer!(|| "Compute final hash 1");
        let h1 = griffin.hash(&h1s)?;
        end_timer!(timer);
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Final hash 1", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let timer = start_timer!(|| "Compute final hash 2");
        let h2 = {
            let mut cs = ConstraintSystemRef::None;
            let mut value = 0u16;
            let mut constant = 0u16;
            let mut new_lc = vec![];

            for x in keeps {
                let x_cs = x.cs();
                match x {
                    Boolean::Constant(c) => {
                        constant += c as u16;
                    }
                    Boolean::Var(variable) => {
                        cs = cs.or(x_cs);
                        value += variable.value().unwrap() as u16;
                        new_lc.push((F::one(), variable.variable()));
                    }
                }
            }
            if cs.is_none() {
                FpVar::constant(F::from(constant))
            } else {
                FpVar::Var(AllocatedFp::new(
                    Some(F::from(constant + value)),
                    cs.new_lc(if cs.should_construct_matrices() {
                        LinearCombination(new_lc) + (F::from(constant), Variable::One)
                    } else {
                        LinearCombination::new()
                    })?,
                    cs,
                ))
            }
        }
        .is_zero()?
        .select(&z_i[1], &griffin.hash(&h2s)?)?;
        end_timer!(timer);
        #[cfg(feature = "constraints")]
        add_to_trace!(|| "Final hash 2", || format!(
            "{} constraints",
            cs.num_constraints() - t
        ));

        Ok(vec![h1, h2])
    }
}

pub fn parse_prover_data(
    input_path: PathBuf,
    output_path: PathBuf,
    n: Option<usize>,
) -> Result<
    (
        Vec<(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>)>,
        Vec<(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>)>,
        Vec<(Matrix<i8, 16, 16>, Matrix<i8, 8, 8>, Matrix<i8, 8, 8>)>,
        Vec<EncodeConfig>,
    ),
    Box<dyn std::error::Error>,
> {
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

    let mut types;
    let mut orig_y;
    let mut orig_u;
    let mut orig_v;
    let mut pred_y;
    let mut pred_u;
    let mut pred_v;
    let mut result_y;
    let mut result_u;
    let mut result_v;

    if let Some(n) = n {
        types = vec![0u8; n * 6];
        orig_y = vec![0u8; n * 256];
        orig_u = vec![0u8; n * 64];
        orig_v = vec![0u8; n * 64];
        pred_y = vec![0u8; n * 256];
        pred_u = vec![0u8; n * 64];
        pred_v = vec![0u8; n * 64];
        result_y = vec![0u8; n * 256];
        result_u = vec![0u8; n * 64];
        result_v = vec![0u8; n * 64];
    } else {
        types = vec![0u8; types_file.metadata()?.len() as usize];
        orig_y = vec![0u8; orig_y_file.metadata()?.len() as usize];
        orig_u = vec![0u8; orig_u_file.metadata()?.len() as usize];
        orig_v = vec![0u8; orig_v_file.metadata()?.len() as usize];
        pred_y = vec![0u8; pred_y_file.metadata()?.len() as usize];
        pred_u = vec![0u8; pred_u_file.metadata()?.len() as usize];
        pred_v = vec![0u8; pred_v_file.metadata()?.len() as usize];
        result_y = vec![0u8; result_y_file.metadata()?.len() as usize];
        result_u = vec![0u8; result_u_file.metadata()?.len() as usize];
        result_v = vec![0u8; result_v_file.metadata()?.len() as usize];
    }

    BufReader::new(types_file).read_exact(&mut types)?;
    BufReader::new(orig_y_file).read_exact(&mut orig_y)?;
    BufReader::new(orig_u_file).read_exact(&mut orig_u)?;
    BufReader::new(orig_v_file).read_exact(&mut orig_v)?;
    BufReader::new(pred_y_file).read_exact(&mut pred_y)?;
    BufReader::new(pred_u_file).read_exact(&mut pred_u)?;
    BufReader::new(pred_v_file).read_exact(&mut pred_v)?;
    BufReader::new(result_y_file).read_exact(&mut result_y)?;
    BufReader::new(result_u_file).read_exact(&mut result_u)?;
    BufReader::new(result_v_file).read_exact(&mut result_v)?;

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

    Ok((blocks, predictions, outputs, configs))
}

pub fn parse_recorder_data(
    input_path: PathBuf,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let orig_y = std::fs::read(input_path.join("orig_y_enc"))?;
    let orig_u = std::fs::read(input_path.join("orig_u_enc"))?;
    let orig_v = std::fs::read(input_path.join("orig_v_enc"))?;

    Ok(orig_y
        .par_chunks_exact(256)
        .zip(orig_u.par_chunks_exact(64).zip(orig_v.par_chunks_exact(64)))
        .map(|(orig_y, (orig_u, orig_v))| [orig_y, orig_u, orig_v].concat())
        .collect::<Vec<_>>())
}

pub fn parse_verifier_data(
    output_path: PathBuf,
) -> Result<
    (
        Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
        Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
        Vec<EncodeConfig>,
    ),
    Box<dyn std::error::Error>,
> {
    let types = std::fs::read(output_path.join("type_enc"))?;
    let pred_y = std::fs::read(output_path.join("pred_y_enc"))?;
    let pred_u = std::fs::read(output_path.join("pred_u_enc"))?;
    let pred_v = std::fs::read(output_path.join("pred_v_enc"))?;
    let result_y = std::fs::read(output_path.join("coeff_y_enc"))?;
    let result_u = std::fs::read(output_path.join("coeff_u_enc"))?;
    let result_v = std::fs::read(output_path.join("coeff_v_enc"))?;

    let predictions = pred_y
        .par_chunks_exact(256)
        .zip(pred_u.par_chunks_exact(64).zip(pred_v.par_chunks_exact(64)))
        .map(|(pred_y, (pred_u, pred_v))| (pred_y.to_vec(), pred_u.to_vec(), pred_v.to_vec()))
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
            Ok::<_, <u8 as TryFrom<i8>>::Error>((
                result_y
                    .iter()
                    .map(|&i| u8::try_from(i as i16 + 128))
                    .collect::<Result<Vec<_>, _>>()?,
                result_u
                    .iter()
                    .map(|&i| u8::try_from(i as i16 + 128))
                    .collect::<Result<Vec<_>, _>>()?,
                result_v
                    .iter()
                    .map(|&i| u8::try_from(i as i16 + 128))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
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

    Ok((predictions, outputs, configs))
}

#[cfg(test)]
pub mod tests {
    use std::{
        error::Error,
        fs::File,
        io::{BufReader, BufWriter, Read, Write},
        path::Path,
        time::Instant,
    };

    use super::*;
    use crate::encode::traits::CastSlice;
    use ark_bn254::{constraints::GVar, Fr, G1Projective as Projective};
    use ark_ff::Zero;
    use ark_grumpkin::{constraints::GVar as GVar2, Projective as Projective2};
    use edit::constraints::{
        Brightness, BrightnessCfg, Grayscale, InvertColor, MaskCfg, Masking, NoOp, Removing,
        RemovingCfg,
    };
    use folding_schemes::{
        commitment::{pedersen::Pedersen, CommitmentScheme},
        folding::nova::Nova,
        transcript::poseidon::poseidon_test_config,
        FoldingScheme,
    };
    use ndarray::Array2;
    use rand::thread_rng;

    #[test]
    fn test_write_noop_params() -> Result<(), Box<dyn Error>> {
        const BLOCKS_PER_STEP: usize = 256;

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, NoOp>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let (pk, vk) = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: vec![Default::default(); BLOCKS_PER_STEP],
                predictions: vec![Default::default(); BLOCKS_PER_STEP],
                outputs: vec![Default::default(); BLOCKS_PER_STEP],
                encode_configs: vec![Default::default(); BLOCKS_PER_STEP],
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        pk.cs_params;

        Ok(())
    }

    #[test]
    fn test_noop() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 1;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, NoOp>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            folding_scheme.prove_step(
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
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[test]
    fn test_bright() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_bright"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 1;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Brightness>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![BrightnessCfg::default(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            folding_scheme.prove_step(
                &params,
                &ExternalInputs {
                    blocks: blocks[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                    predictions: predictions[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP]
                        .to_vec(),
                    outputs: outputs[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                    encode_configs: configs[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP]
                        .to_vec(),
                    edit_configs: vec![BrightnessCfg(416); BLOCKS_PER_STEP],
                },
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[test]
    fn test_gray() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_gray"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 1;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Grayscale>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            folding_scheme.prove_step(
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
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[test]
    fn test_inv() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_inv"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 1;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, InvertColor>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            folding_scheme.prove_step(
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
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[test]
    fn test_crop() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        const WW: usize = 160;
        const HH: usize = 128;

        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_crop"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 1;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Removing>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![RemovingCfg(false); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        let x = 48;
        let y = 80;

        let mut j = 0;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            let mut p = vec![];
            let mut o = vec![];
            let mut e1 = vec![];
            let mut e2 = vec![];

            for v in i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP {
                let tt = v % (W / 16 * H / 16);
                let xx = tt % (W / 16) * 16;
                let yy = tt / (W / 16) * 16;
                let should_keep = xx >= x && xx < (x + WW) && yy >= y && yy < (y + HH);
                if should_keep {
                    p.push(predictions[j].clone());
                    o.push(outputs[j].clone());
                    e1.push(configs[j].clone());
                } else {
                    p.push(Default::default());
                    o.push(Default::default());
                    e1.push(Default::default());
                }
                e2.push(RemovingCfg(should_keep));
                if should_keep {
                    j += 1;
                }
            }

            folding_scheme.prove_step(
                &params,
                &ExternalInputs {
                    blocks: blocks[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                    predictions: p,
                    outputs: o,
                    encode_configs: e1,
                    edit_configs: e2,
                },
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[test]
    fn test_cut() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;

        let (mut blocks, mut predictions, mut outputs, mut configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_cut"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 20;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Removing>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![RemovingCfg(false); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        let s = 6;
        let e = 134;
        let mut j = 0;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            let mut p: Vec<(Matrix<u8, 16, 16>, Matrix<u8, 8, 8>, Matrix<u8, 8, 8>)> = vec![];
            let mut o = vec![];
            let mut e1 = vec![];
            let mut e2 = vec![];

            for v in i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP {
                let tt = v / (W / 16 * H / 16);
                let should_keep = tt >= s && tt < e;
                if should_keep {
                    p.push(predictions[j].clone());
                    o.push(outputs[j].clone());
                    e1.push(configs[j].clone());
                } else {
                    p.push(Default::default());
                    o.push(Default::default());
                    e1.push(Default::default());
                }
                e2.push(RemovingCfg(should_keep));
                if should_keep {
                    j += 1;
                }
            }

            folding_scheme.prove_step(
                &params,
                &ExternalInputs {
                    blocks: blocks[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                    predictions: p,
                    outputs: o,
                    encode_configs: e1,
                    edit_configs: e2,
                },
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[test]
    fn test_mask() -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;

        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_mask"),
            None,
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let num_steps = blocks.len() / BLOCKS_PER_STEP;
        // let num_steps = 20;
        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Masking>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![MaskCfg::default(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;
        let x = 96;
        let y = 80;
        let w = 176;
        let h = 144;

        // compute a step of the IVC
        for i in 0..num_steps {
            let start = Instant::now();

            folding_scheme.prove_step(
                &params,
                &ExternalInputs {
                    blocks: blocks[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                    predictions: predictions[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP]
                        .to_vec(),
                    outputs: outputs[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP].to_vec(),
                    encode_configs: configs[i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP]
                        .to_vec(),
                    edit_configs: (i * BLOCKS_PER_STEP..(i + 1) * BLOCKS_PER_STEP)
                        .map(|v| {
                            let tt = v % (W / 16 * H / 16);
                            let xx = tt % (W / 16) * 16;
                            let yy = tt / (W / 16) * 16;
                            MaskCfg(
                                Array2::from_shape_fn((16, 16), |(m, n)| {
                                    let xx = xx + m;
                                    let yy = yy + n;
                                    let b = xx >= x && xx < x + w && yy >= y && yy < y + h;
                                    (
                                        if b {
                                            (((xx - x) / 16 % 2 + (yy - y) / 16 % 2 + 3) * 32) as u8
                                        } else {
                                            0
                                        },
                                        !b,
                                    )
                                }),
                                Array2::from_shape_fn((8, 8), |(m, n)| {
                                    let xx = xx + m * 2;
                                    let yy = yy + n * 2;
                                    (128, !(xx >= x && xx < x + w && yy >= y && yy < y + h))
                                }),
                                Array2::from_shape_fn((8, 8), |(m, n)| {
                                    let xx = xx + m * 2;
                                    let yy = yy + n * 2;
                                    (128, !(xx >= x && xx < x + w && yy >= y && yy < y + h))
                                }),
                            )
                        })
                        .collect::<Vec<_>>(),
                },
            )?;

            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
        }
        println!(
            "state at last step (after {} iterations): {:?}",
            num_steps,
            folding_scheme.state()
        );

        let (running_instance, incoming_instance, cyclefold_instance) = folding_scheme.instances();

        println!("Run the Nova's IVC verifier");
        NOVA::verify(
            &params.1,
            initial_state.clone(),
            folding_scheme.state(), // latest state
            Fr::from(num_steps as u32),
            running_instance,
            incoming_instance,
            cyclefold_instance,
        )?;
        Ok(())
    }

    #[ignore]
    #[test]
    fn swap() -> Result<(), Box<dyn Error>> {
        fn redirect(i: File, o: File, n: usize) -> Result<(), Box<dyn Error>> {
            let m = i.metadata()?.len() as usize;
            assert_eq!(m % n, 0);
            let mut tmp1 = vec![0u8; n];
            let mut tmp2 = vec![0u8; n];
            let mut i = BufReader::new(i);
            let mut o = BufWriter::new(o);
            i.read_exact(&mut tmp1)?;
            o.write_all(&tmp1)?;

            let r = (m - n) / (2 * n);

            for _ in 0..r {
                i.read_exact(&mut tmp1)?;
                i.read_exact(&mut tmp2)?;
                o.write_all(&tmp2)?;
                o.write_all(&tmp1)?;
            }
            if r * 2 * n != m - n {
                i.read_exact(&mut tmp1)?;
                o.write_all(&tmp1)?;
            }
            Ok(())
        }

        fn redirect_compact(i: File, o: File, n: usize) -> Result<(), Box<dyn Error>> {
            let m = i.metadata()?.len() as usize;
            assert_eq!(m % n, 0);
            let mut tmp1 = vec![0u8; n];
            let mut tmp2 = vec![0u8; n];
            let mut i = BufReader::new(i);
            let mut o = BufWriter::new(o);
            i.read_exact(&mut tmp1)?;
            o.write_all(
                &tmp1
                    .cast::<i32>()
                    .iter()
                    .map(|&i| i8::try_from(i))
                    .collect::<Result<Vec<_>, _>>()?
                    .cast(),
            )?;

            let r = (m - n) / (2 * n);

            for _ in 0..r {
                i.read_exact(&mut tmp1)?;
                i.read_exact(&mut tmp2)?;
                o.write_all(
                    &tmp2
                        .cast::<i32>()
                        .iter()
                        .map(|&i| i8::try_from(i))
                        .collect::<Result<Vec<_>, _>>()?
                        .cast(),
                )?;
                o.write_all(
                    &tmp1
                        .cast::<i32>()
                        .iter()
                        .map(|&i| i8::try_from(i))
                        .collect::<Result<Vec<_>, _>>()?
                        .cast(),
                )?;
            }
            if r * 2 * n != m - n {
                i.read_exact(&mut tmp1)?;
                o.write_all(
                    &tmp1
                        .cast::<i32>()
                        .iter()
                        .map(|&i| i8::try_from(i))
                        .collect::<Result<Vec<_>, _>>()?
                        .cast(),
                )?;
            }
            Ok(())
        }

        for (f, ww, hh) in [
            ("bunny2", 1280, 720),
            // ("foreman", 352, 288),
            // ("foreman_bright", 352, 288),
            // ("foreman_crop", 160, 128),
            // ("foreman_cut", 352, 288),
            // ("foreman_gray", 352, 288),
            // ("foreman_inv", 352, 288),
            // ("foreman_mask", 352, 288),
        ] {
            let input_path = Path::new(env!("DATA_PATH"))
                .parent()
                .unwrap()
                .join("data_raw")
                .join(f);
            let output_path = Path::new(env!("DATA_PATH")).join(f);

            redirect(
                File::open(input_path.join("type_enc"))?,
                File::create(output_path.join("type_enc"))?,
                ww * hh / 256 * 6,
            )?;
            redirect(
                File::open(input_path.join("orig_y_enc"))?,
                File::create(output_path.join("orig_y_enc"))?,
                ww * hh,
            )?;
            redirect(
                File::open(input_path.join("orig_u_enc"))?,
                File::create(output_path.join("orig_u_enc"))?,
                ww * hh / 4,
            )?;
            redirect(
                File::open(input_path.join("orig_v_enc"))?,
                File::create(output_path.join("orig_v_enc"))?,
                ww * hh / 4,
            )?;
            redirect(
                File::open(input_path.join("pred_y_enc"))?,
                File::create(output_path.join("pred_y_enc"))?,
                ww * hh,
            )?;
            redirect(
                File::open(input_path.join("pred_u_enc"))?,
                File::create(output_path.join("pred_u_enc"))?,
                ww * hh / 4,
            )?;
            redirect(
                File::open(input_path.join("pred_v_enc"))?,
                File::create(output_path.join("pred_v_enc"))?,
                ww * hh / 4,
            )?;
            redirect_compact(
                File::open(input_path.join("coeff_y_enc"))?,
                File::create(output_path.join("coeff_y_enc"))?,
                ww * hh * 4,
            )?;
            redirect_compact(
                File::open(input_path.join("coeff_u_enc"))?,
                File::create(output_path.join("coeff_u_enc"))?,
                ww * hh,
            )?;
            redirect_compact(
                File::open(input_path.join("coeff_v_enc"))?,
                File::create(output_path.join("coeff_v_enc"))?,
                ww * hh,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod benches {
    extern crate test;
    use std::{
        env,
        error::Error

        ,
        path::Path
        ,
    };

    use super::*;
    use ark_bn254::{constraints::GVar, Fq, Fr, G1Projective as Projective};
    use ark_crypto_primitives::crh::{poseidon::CRH, CRHScheme};
    use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
    use ark_ff::{UniformRand, Zero};
    use ark_grumpkin::{constraints::GVar as GVar2, Projective as Projective2};
    use ark_relations::r1cs::ConstraintSystem;
    use edit::constraints::{
        Brightness, BrightnessCfg, Grayscale, InvertColor, MaskCfg, Masking, NoOp, Removing,
        RemovingCfg,
    };
    use folding_schemes::{
        ccs::r1cs::extract_r1cs,
        commitment::{pedersen::Pedersen, CommitmentScheme},
        folding::nova::{
            circuits::AugmentedFCircuit, nifs::NIFS, Nova,
        },
        transcript::poseidon::poseidon_test_config
        ,
        FoldingScheme, MVM,
    };
    use ndarray::Array2;
    use rand::thread_rng;

    #[bench]
    fn bench_sig_keygen(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        b.iter(|| {
            let sk = Fq::rand(rng);
            Projective2::generator() * sk
        });
        Ok(())
    }

    #[bench]
    fn bench_sig_sign(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();
        let poseidon_config = poseidon_test_config();
        let sk = Fq::rand(rng);
        let vk = Projective2::generator() * sk;
        let m = Fr::rand(rng);
        b.iter(|| {
            let (px, py) = {
                let p = vk.into_affine();
                p.xy().unwrap_or((Fr::zero(), Fr::zero()))
            };
            {
                let r = Fq::rand(rng);
                let rx = (Projective2::generator() * r)
                    .into_affine()
                    .x()
                    .unwrap_or_default();
                let e = CRH::evaluate(&poseidon_config, [rx, px, py, m]).unwrap();
                (
                    rx,
                    r + sk * Fq::from_le_bytes_mod_order(&e.into_bigint().to_bytes_le()),
                )
            }
        });
        Ok(())
    }

    #[bench]
    fn bench_cpu_compute_t(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();
        let blocks_per_step: usize = env::var("BLOCKS_PER_STEP").unwrap().parse().unwrap();

        let cs = ConstraintSystem::<Fr>::new_ref();
        let la = LookupArgument::new_ref();
        AugmentedFCircuit::<Projective, Projective2, GVar2, _>::empty(
            &poseidon_test_config(),
            la.clone(),
            &EditEncodeCircuit::<Fr, NoOp> {
                _e: PhantomData,
                griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
            },
            &ExternalInputs {
                blocks: vec![Default::default(); blocks_per_step],
                predictions: vec![Default::default(); blocks_per_step],
                outputs: vec![Default::default(); blocks_per_step],
                encode_configs: vec![Default::default(); blocks_per_step],
                edit_configs: vec![(); blocks_per_step],
            },
        )
        .run(cs.clone())?;
        la.build_histo(cs.clone())?;
        la.generate_lookup_constraints(cs.clone(), Fr::rand(rng))?;

        cs.finalize();
        let cs = cs.into_inner().unwrap();

        let r1cs = extract_r1cs(&cs);
        let z1 = (0..r1cs.A.n_cols)
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let z2 = (0..r1cs.A.n_cols)
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();

        b.iter(|| {
            NIFS::<Projective, Pedersen<Projective, false>>::compute_T(
                &r1cs, z1[0], z2[0], &z1, &z2,
            )
            .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_gpu_compute_t(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();
        let blocks_per_step: usize = env::var("BLOCKS_PER_STEP").unwrap().parse().unwrap();

        let cs = ConstraintSystem::<Fr>::new_ref();
        let la = LookupArgument::new_ref();
        AugmentedFCircuit::<Projective, Projective2, GVar2, _>::empty(
            &poseidon_test_config(),
            la.clone(),
            &EditEncodeCircuit::<Fr, NoOp> {
                _e: PhantomData,
                griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
            },
            &ExternalInputs {
                blocks: vec![Default::default(); blocks_per_step],
                predictions: vec![Default::default(); blocks_per_step],
                outputs: vec![Default::default(); blocks_per_step],
                encode_configs: vec![Default::default(); blocks_per_step],
                edit_configs: vec![(); blocks_per_step],
            },
        )
        .run(cs.clone())?;
        la.build_histo(cs.clone())?;
        la.generate_lookup_constraints(cs.clone(), Fr::rand(rng))?;

        cs.finalize();
        let cs = cs.into_inner().unwrap();

        let r1cs = extract_r1cs(&cs);
        let z1 = (0..r1cs.A.n_cols)
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let z2 = (0..r1cs.A.n_cols)
            .map(|_| Fr::rand(rng))
            .collect::<Vec<_>>();
        let mut t = ark_bn254::Fr::alloc_vec(r1cs.A.n_rows);
        let tmp_e = ark_bn254::Fr::alloc_vec(r1cs.A.n_rows);

        b.iter(|| {
            Fr::compute_t(None, &r1cs.A.cuda, &r1cs.B.cuda, &r1cs.C.cuda,             &z1[..1],
                          &z1[1..1 + r1cs.l],
                          &z1[1 + r1cs.l..],
                          &z2[..1],
                          &z2[1..1 + r1cs.l],
                          &z2[1 + r1cs.l..], &tmp_e, &mut t);
        });

        Ok(())
    }

    #[bench]
    fn bench_step_block(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman"),
            None,
        )?;
        let blocks_per_step: usize = env::var("BLOCKS_PER_STEP").unwrap().parse().unwrap();

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, NoOp>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..blocks_per_step].to_vec(),
                predictions: predictions[0..blocks_per_step].to_vec(),
                outputs: outputs[0..blocks_per_step].to_vec(),
                encode_configs: configs[0..blocks_per_step].to_vec(),
                edit_configs: vec![(); blocks_per_step],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        b.iter(|| {
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..blocks_per_step].to_vec(),
                        predictions: predictions[0..blocks_per_step].to_vec(),
                        outputs: outputs[0..blocks_per_step].to_vec(),
                        encode_configs: configs[0..blocks_per_step].to_vec(),
                        edit_configs: vec![(); blocks_per_step],
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_noop(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, NoOp>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        b.iter(|| {
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                        outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                        encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                        edit_configs: vec![(); BLOCKS_PER_STEP],
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_bright(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_bright"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Brightness>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![BrightnessCfg::default(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        b.iter(|| {
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                        outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                        encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                        edit_configs: vec![BrightnessCfg(416); BLOCKS_PER_STEP],
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_gray(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_gray"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Grayscale>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        b.iter(|| {
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                        outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                        encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                        edit_configs: vec![(); BLOCKS_PER_STEP],
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_inv(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_inv"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, InvertColor>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        b.iter(|| {
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                        outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                        encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                        edit_configs: vec![(); BLOCKS_PER_STEP],
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_crop(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        const WW: usize = 160;
        const HH: usize = 128;

        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_crop"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Removing>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![RemovingCfg(false); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        let x = 48;
        let y = 80;

        b.iter(|| {
            let mut j = 0;

            let mut p = vec![];
            let mut o = vec![];
            let mut e1 = vec![];
            let mut e2 = vec![];

            for v in 0..BLOCKS_PER_STEP {
                let tt = v % (W / 16 * H / 16);
                let xx = tt % (W / 16) * 16;
                let yy = tt / (W / 16) * 16;
                let should_keep = xx >= x && xx < (x + WW) && yy >= y && yy < (y + HH);
                if should_keep {
                    p.push(predictions[j].clone());
                    o.push(outputs[j].clone());
                    e1.push(configs[j].clone());
                } else {
                    p.push(Default::default());
                    o.push(Default::default());
                    e1.push(Default::default());
                }
                e2.push(RemovingCfg(should_keep));
                if should_keep {
                    j += 1;
                }
            }

            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: p,
                        outputs: o,
                        encode_configs: e1,
                        edit_configs: e2,
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_cut(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_cut"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Removing>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![RemovingCfg(false); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        let s = 6;
        let e = 134;

        b.iter(|| {
            let mut j = 0;

            let mut p = vec![];
            let mut o = vec![];
            let mut e1 = vec![];
            let mut e2 = vec![];

            for v in 0..BLOCKS_PER_STEP {
                let tt = v / (W / 16 * H / 16);
                let should_keep = tt >= s && tt < e;
                if should_keep {
                    p.push(predictions[j].clone());
                    o.push(outputs[j].clone());
                    e1.push(configs[j].clone());
                } else {
                    p.push(Default::default());
                    o.push(Default::default());
                    e1.push(Default::default());
                }
                e2.push(RemovingCfg(should_keep));
                if should_keep {
                    j += 1;
                }
            }

            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: p,
                        outputs: o,
                        encode_configs: e1,
                        edit_configs: e2,
                    },
                )
                .unwrap();
        });

        Ok(())
    }

    #[bench]
    fn bench_step_mask(b: &mut test::Bencher) -> Result<(), Box<dyn Error>> {
        const W: usize = 352;
        const H: usize = 288;
        let (blocks, predictions, outputs, configs) = parse_prover_data(
            Path::new(env!("DATA_PATH")).join("foreman"),
            Path::new(env!("DATA_PATH")).join("foreman_mask"),
            Some(256),
        )?;
        const BLOCKS_PER_STEP: usize = 256;

        let initial_state = vec![Fr::zero(), Fr::zero()];

        let F_circuit = EditEncodeCircuit {
            _e: PhantomData,
            griffin_params: Arc::new(GriffinParams::new(16, 5, 9)),
        };

        /// The idea here is that eventually we could replace the next line chunk that defines the
        /// `type NOVA = Nova<...>` by using another folding scheme that fulfills the `FoldingScheme`
        /// trait, and the rest of our code would be working without needing to be updated.
        type NOVA = Nova<
            Projective,
            GVar,
            Projective2,
            GVar2,
            EditEncodeCircuit<Fr, Masking>,
            Pedersen<Projective>,
            Pedersen<Projective2>,
        >;

        println!("Prepare Nova ProverParams & VerifierParams");
        let params = NOVA::preprocess(
            &poseidon_test_config(),
            &F_circuit,
            &mut thread_rng(),
            &ExternalInputs {
                blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                edit_configs: vec![MaskCfg::default(); BLOCKS_PER_STEP],
            },
        )?;

        println!("Initialize FoldingScheme");
        let mut folding_scheme = NOVA::init(&params, F_circuit, initial_state.clone())?;

        let x = 96;
        let y = 80;
        let w = 176;
        let h = 144;

        b.iter(|| {
            folding_scheme
                .prove_step(
                    &params,
                    &ExternalInputs {
                        blocks: blocks[0..BLOCKS_PER_STEP].to_vec(),
                        predictions: predictions[0..BLOCKS_PER_STEP].to_vec(),
                        outputs: outputs[0..BLOCKS_PER_STEP].to_vec(),
                        encode_configs: configs[0..BLOCKS_PER_STEP].to_vec(),
                        edit_configs: (0..BLOCKS_PER_STEP)
                            .map(|v| {
                                let tt = v % (W / 16 * H / 16);
                                let xx = tt % (W / 16) * 16;
                                let yy = tt / (W / 16) * 16;
                                MaskCfg(
                                    Array2::from_shape_fn((16, 16), |(m, n)| {
                                        let xx = xx + m;
                                        let yy = yy + n;
                                        let b = xx >= x && xx < x + w && yy >= y && yy < y + h;
                                        (
                                            if b {
                                                (((xx - x) / 16 % 2 + (yy - y) / 16 % 2 + 3) * 32)
                                                    as u8
                                            } else {
                                                0
                                            },
                                            !b,
                                        )
                                    }),
                                    Array2::from_shape_fn((8, 8), |(m, n)| {
                                        let xx = xx + m * 2;
                                        let yy = yy + n * 2;
                                        (128, !(xx >= x && xx < x + w && yy >= y && yy < y + h))
                                    }),
                                    Array2::from_shape_fn((8, 8), |(m, n)| {
                                        let xx = xx + m * 2;
                                        let yy = yy + n * 2;
                                        (128, !(xx >= x && xx < x + w && yy >= y && yy < y + h))
                                    }),
                                )
                            })
                            .collect::<Vec<_>>(),
                    },
                )
                .unwrap();
        });

        Ok(())
    }
}
