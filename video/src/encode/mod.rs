pub mod constants;
pub mod constraints;
pub mod traits;

use constants::*;

use std::borrow::{Borrow, BorrowMut};
use std::mem::size_of;
use std::ops::{Deref, Index, IndexMut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

use ark_ff::{BigInteger, Field, One, PrimeField};
use ark_r1cs_std::alloc::{AllocVar, AllocationMode};
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::{fields::FieldVar, R1CSVar};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::iterable::Iterable;
use ndarray::{array, s, Array, Array2};
use num_bigint::BigUint;
use rayon::prelude::*;

struct QuantParam {
    offset: u64,
    scale: u64,
}

impl QuantParam {
    fn new(offset: u64, scale: u64) -> Self {
        Self { offset, scale }
    }
}

impl From<(u64, u64)> for QuantParam {
    fn from((offset, scale): (u64, u64)) -> Self {
        Self { offset, scale }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MacroblockType {
    SKIP,
    P16x16,
    P16x8,
    P8x16,
    P8x8,
    #[default]
    I4x4,
    I8x8,
    I16x16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<const M: usize, const N: usize>(Array2<i16>);

impl<const M: usize, const N: usize> Deref for Matrix<M, N> {
    type Target = Array2<i16>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const M: usize, const N: usize> Index<(usize, usize)> for Matrix<M, N> {
    type Output = i16;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.0[[i, j]]
    }
}

impl<const M: usize, const N: usize> IndexMut<(usize, usize)> for Matrix<M, N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.0[[i, j]]
    }
}

impl<const M: usize, const N: usize> FromIterator<i16> for Matrix<M, N> {
    fn from_iter<I: IntoIterator<Item = i16>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<const M: usize, const N: usize> Default for Matrix<M, N> {
    fn default() -> Self {
        Self(Array2::zeros((M, N)))
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    fn new() -> Self {
        Self(Array2::zeros((M, N)))
    }

    pub fn from_vec(v: Vec<i16>) -> Self {
        Self(Array2::from_shape_vec((M, N), v).unwrap())
    }

    fn dct_region<const P: usize, const Q: usize>(
        &self,
        i_offset: usize,
        j_offset: usize,
    ) -> Matrix<P, Q> {
        match (P, Q) {
            (4, 4) => {
                let t = array![[1, 1, 1, 1], [2, 1, -1, -2], [1, -1, -1, 1], [1, -2, 2, -1],];
                Matrix::<P, Q>(
                    t.dot(&self.slice(s![i_offset..i_offset + 4, j_offset..j_offset + 4]))
                        .dot(&t.t()),
                )
            }
            (8, 8) => {
                let mut t = vec![];
                for i in 0..8 {
                    let p = &self.slice(s![i_offset + i, j_offset..]);
                    let a = [
                        &p[0] + &p[7],
                        &p[1] + &p[6],
                        &p[2] + &p[5],
                        &p[3] + &p[4],
                        &p[0] - &p[7],
                        &p[1] - &p[6],
                        &p[2] - &p[5],
                        &p[3] - &p[4],
                    ];
                    let b = [
                        &a[0] + &a[3],
                        &a[1] + &a[2],
                        &a[0] - &a[3],
                        &a[1] - &a[2],
                        &a[5] + &a[6] + &a[4] + (&a[4] >> 1),
                        &a[4] - &a[7] - &a[6] - (&a[6] >> 1),
                        &a[4] + &a[7] - &a[5] - (&a[5] >> 1),
                        &a[5] - &a[6] + &a[7] + (&a[7] >> 1),
                    ];
                    t.push(&b[0] + &b[1]);
                    t.push(&b[4] + (b[7] >> 2));
                    t.push(&b[2] + (b[3] >> 1));
                    t.push(&b[5] + (b[6] >> 2));
                    t.push(&b[0] - &b[1]);
                    t.push(&b[6] - (b[5] >> 2));
                    t.push((b[2] >> 1) - &b[3]);
                    t.push((b[4] >> 2) - &b[7]);
                }
                let mut r = Matrix::<P, Q>::new();
                for i in 0..8 {
                    let a = vec![
                        &t[i] + &t[i + 56],
                        &t[i + 8] + &t[i + 48],
                        &t[i + 16] + &t[i + 40],
                        &t[i + 24] + &t[i + 32],
                        &t[i] - &t[i + 56],
                        &t[i + 8] - &t[i + 48],
                        &t[i + 16] - &t[i + 40],
                        &t[i + 24] - &t[i + 32],
                    ];
                    let b = [
                        &a[0] + &a[3],
                        &a[1] + &a[2],
                        &a[0] - &a[3],
                        &a[1] - &a[2],
                        &a[5] + &a[6] + &a[4] + (a[4] >> 1),
                        &a[4] - &a[7] - &a[6] - (a[6] >> 1),
                        &a[4] + &a[7] - &a[5] - (a[5] >> 1),
                        &a[5] - &a[6] + &a[7] + (a[7] >> 1),
                    ];
                    r[(0, i)] = &b[0] + &b[1];
                    r[(1, i)] = &b[4] + (b[7] >> 2);
                    r[(2, i)] = &b[2] + (b[3] >> 1);
                    r[(3, i)] = &b[5] + (b[6] >> 2);
                    r[(4, i)] = &b[0] - &b[1];
                    r[(5, i)] = &b[6] - (b[5] >> 2);
                    r[(6, i)] = (b[2] >> 1) - &b[3];
                    r[(7, i)] = (b[4] >> 2) - &b[7];
                }
                r
            }
            _ => unimplemented!(),
        }
    }

    fn hadamard_quant(&self, qp_over_6: usize, param: &QuantParam) -> Self {
        match (M, N) {
            (4, 4) => {
                let t = array![[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1],];
                let r = t.dot(&self.0).dot(&t.t());

                Self::from_iter(r.into_iter().map(|v| {
                    let is_neg = v < 0;
                    let level = ((v.abs() as u64 * &param.scale + param.offset * 2)
                        >> (Q_BITS_4 + qp_over_6 + 1)) as i16;
                    if is_neg {
                        -level
                    } else {
                        level
                    }
                }))
            }
            (2, 2) => {
                let t = array![[1, 1], [1, -1]];
                let r = t.dot(&self.0).dot(&t.t());

                Self::from_iter(r.into_iter().map(|v| {
                    let is_neg = v < 0;
                    let level = ((v.abs() as u64 * &param.scale + param.offset * 2)
                        >> (Q_BITS_4 + qp_over_6 + 1)) as i16;
                    if is_neg {
                        -level
                    } else {
                        level
                    }
                }))
            }
            _ => unimplemented!(),
        }
    }

    fn quant(&self, qp_over_6: usize, params: &[QuantParam], ac_only: bool) -> Self {
        let q_bits = match (M, N) {
            (4, 4) => Q_BITS_4,
            (8, 8) => Q_BITS_8,
            _ => unimplemented!(),
        };

        match (M, N) {
            (4, 4) | (8, 8) => self
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i == 0 && ac_only {
                        return 0;
                    }
                    let param = &params[i];
                    let is_neg = v < 0;
                    let level = ((v.abs() as u64 * &param.scale + &param.offset)
                        >> (q_bits + qp_over_6)) as i16;
                    if is_neg {
                        -level
                    } else {
                        level
                    }
                })
                .collect(),
            _ => unimplemented!(),
        }
    }
}

fn iabs(x: i32) -> i32 {
    let y = x >> 15;
    (x ^ y) - y
}

fn quant_core(v: i16, m: i16, a: i32, d: usize) -> i16 {
    let level = (iabs(v as i32) * m as i32 + a) >> d;
    if v < 0 {
        -iabs(level) as i16
    } else {
        iabs(level) as i16
    }
}

impl Matrix<16, 16> {
    pub fn encode_luma_4x4(
        &self,
        pred: &Self,
        is_i16x16: bool,
        is_intra_slice: bool,
        qp: usize,
    ) -> Self {
        let (qp_over_6, qp_mod_6) = (qp / 6, qp % 6);
        let d = Q_BITS_4 + qp_over_6;

        let base_offset = if is_intra_slice {
            BASE_OFFSET_I_SLICE
        } else {
            BASE_OFFSET_BP_SLICE
        };
        let diff = self
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| a - b)
            .collect::<Self>();

        let offset = (base_offset << (qp_over_6 + Q_BITS_4 - OFFSET_BITS));
        let scale0 = [13107, 11916, 10082, 9362, 8192, 7282][qp_mod_6];
        let scale1 = [8066, 7490, 6554, 5825, 5243, 4559][qp_mod_6];
        let scale2 = [5243, 4660, 4194, 3647, 3355, 2893][qp_mod_6];

        let blocks = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| diff.dct_region::<4, 4>(i * 4, j * 4))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut dc_block = Matrix::<4, 4>::new();
        for i in 0..4 {
            for j in 0..4 {
                dc_block[(i, j)] = blocks[i][j][(0, 0)];
            }
        }
        let dc_block = {
            let mut t = vec![];
            for i in 0..4 {
                let row = &dc_block.row(i);
                t.push(&row[0] + &row[1] + &row[2] + &row[3]);
                t.push(&row[0] + &row[1] - &row[2] - &row[3]);
                t.push(&row[0] - &row[1] - &row[2] + &row[3]);
                t.push(&row[0] - &row[1] + &row[2] - &row[3]);
            }
            let mut block = Matrix::<4, 4>::new();
            for i in 0..4 {
                for (j, &v) in vec![
                    &t[i] + &t[i + 4] + &t[i + 8] + &t[i + 12],
                    &t[i] + &t[i + 4] - &t[i + 8] - &t[i + 12],
                    &t[i] - &t[i + 4] - &t[i + 8] + &t[i + 12],
                    &t[i] - &t[i + 4] + &t[i + 8] - &t[i + 12],
                ]
                .iter()
                .enumerate()
                {
                    let v = v >> 1;
                    block[(j, i)] = quant_core(v, scale0, offset as i32 * 2, d + 1);
                }
            }
            block
        };

        let blocks = (0..4)
            .flat_map(|i| {
                (0..4)
                    .map(|j| {
                        let mut r = blocks[i / 2 * 2 + j / 2][i % 2 * 2 + j % 2]
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                quant_core(
                                    v,
                                    match i {
                                        0 | 2 | 8 | 10 => scale0,
                                        5 | 7 | 13 | 15 => scale2,
                                        _ => scale1,
                                    },
                                    offset as i32,
                                    d,
                                )
                            })
                            .collect::<Vec<_>>();

                        if is_i16x16 {
                            r[0] = dc_block[(i / 2 * 2 + j / 2, i % 2 * 2 + j % 2)];
                        }

                        r
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .concat();

        Self::from_vec(blocks)
    }
}

impl Matrix<8, 8> {
    pub fn encode_chroma(
        &self,
        pred: &Self,
        is_i16x16: bool,
        is_intra_slice: bool,
        qp: usize,
    ) -> Self {
        let (qp_over_6, qp_mod_6) = (qp / 6, qp % 6);
        let d = Q_BITS_4 + qp_over_6;

        let base_offset = if is_intra_slice {
            BASE_OFFSET_I_SLICE
        } else {
            BASE_OFFSET_BP_SLICE
        };
        let diff = self
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| a - b)
            .collect::<Self>();

        let offset = (base_offset << (qp_over_6 + Q_BITS_4 - OFFSET_BITS));
        let scale0 = [13107, 11916, 10082, 9362, 8192, 7282][qp_mod_6];
        let scale1 = [8066, 7490, 6554, 5825, 5243, 4559][qp_mod_6];
        let scale2 = [5243, 4660, 4194, 3647, 3355, 2893][qp_mod_6];

        let blocks = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| diff.dct_region::<4, 4>(i * 4, j * 4))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut dc_block = Matrix::<2, 2>::new();
        for i in 0..2 {
            for j in 0..2 {
                dc_block[(i, j)] = blocks[i][j][(0, 0)].clone();
            }
        }
        let dc_block = {
            let p0 = &dc_block[(0, 0)] + &dc_block[(0, 1)];
            let p1 = &dc_block[(0, 0)] - &dc_block[(0, 1)];
            let p2 = &dc_block[(1, 0)] + &dc_block[(1, 1)];
            let p3 = &dc_block[(1, 0)] - &dc_block[(1, 1)];

            vec![&p0 + &p2, &p1 + &p3, &p0 - &p2, &p1 - &p3]
                .iter()
                .map(|&v| quant_core(v, scale0, offset as i32 * 2, d + 1))
                .collect::<Matrix<2, 2>>()
        };

        let blocks = (0..2)
            .flat_map(|i| {
                (0..2)
                    .map(|j| {
                        let mut r = blocks[i][j]
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                quant_core(
                                    v,
                                    match i {
                                        0 | 2 | 8 | 10 => scale0,
                                        5 | 7 | 13 | 15 => scale2,
                                        _ => scale1,
                                    },
                                    offset as i32,
                                    d,
                                )
                            })
                            .collect::<Vec<_>>();

                        r[0] = dc_block[(i, j)].clone();
                        r
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .concat();

        Self::from_vec(blocks)
    }
}

#[cfg(test)]
pub mod tests {
    use self::constraints::{MatrixVar, QuantParamVar};
    use self::traits::CastSlice;
    use super::*;
    use ark_pallas::Fr;
    use ark_r1cs_std::{alloc::AllocVar, R1CSVar};
    use ark_relations::r1cs::ConstraintSystem;
    use folding_schemes::frontend::LookupArgument;
    use rand::{thread_rng, Rng};
    use std::error::Error;
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::path::Path;

    #[test]
    fn test_dct_4x4() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let x = Matrix::<4, 4>::from_iter((0..16).map(|_| rng.gen::<u8>() as i16));
        let y = x.dct_region(0, 0);

        let x_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(y))?;
        x_var.dct_region(0, 0)?.enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_dct_8x8() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let x = Matrix::<8, 8>::from_iter((0..64).map(|_| rng.gen::<u8>() as i16));
        let y = x.dct_region(0, 0);

        let x_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(y))?;
        x_var.dct_region(0, 0)?.enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_hadamard_quant_2x2() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let param = QuantParam::new(174592, 8192);

        let x = Matrix::<2, 2>::from_iter((0..4).map(|_| rng.gen::<u8>() as i16));
        let y = x.hadamard_quant(4, &param);

        let x_var = MatrixVar::<FpVar<Fr>, 2, 2>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 2, 2>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .hadamard_quant(4, &QuantParamVar::constant(param))?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_hadamard_quant_4x4() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let param = QuantParam::new(174592, 8192);

        let x = Matrix::<4, 4>::from_iter((0..16).map(|_| rng.gen::<u8>() as i16));
        let y = x.hadamard_quant(4, &param);

        let x_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .hadamard_quant(4, &QuantParamVar::constant(param))?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_quant_ac_4x4() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let qp = 30;
        let offset = BASE_OFFSET_I_SLICE << (Q_BITS_4 + qp / 6 - OFFSET_BITS);
        let params = SCALES_4x4[qp % 6].map(|i| QuantParam::new(offset, i));

        let x = Matrix::<4, 4>::from_iter((0..16).map(|_| rng.gen::<i16>() as i16));
        let y = x.quant(qp / 6, &params, true);

        let x_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .quant(qp / 6, &Vec::new_constant(cs.clone(), params)?, true)?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_quant_ac_8x8() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let qp = 30;
        let offset = BASE_OFFSET_I_SLICE << (Q_BITS_8 + qp / 6 - OFFSET_BITS);
        let params = SCALES_8x8[qp % 6].map(|i| QuantParam::new(offset, i));

        let x = Matrix::<8, 8>::from_iter((0..64).map(|_| rng.gen::<i16>() as i16));
        let y = x.quant(qp / 6, &params, true);

        let x_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .quant(qp / 6, &Vec::new_constant(cs.clone(), params)?, true)?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_quant_4x4() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let qp = 30;
        let offset = BASE_OFFSET_I_SLICE << (Q_BITS_4 + qp / 6 - OFFSET_BITS);
        let params = SCALES_4x4[qp % 6].map(|i| QuantParam::new(offset, i));

        let x = Matrix::<4, 4>::from_iter((0..16).map(|_| rng.gen::<i16>() as i16));
        let y = x.quant(qp / 6, &params, false);

        let x_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .quant(qp / 6, &Vec::new_constant(cs.clone(), params)?, false)?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_quant_8x8() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let qp = 30;
        let offset = BASE_OFFSET_I_SLICE << (Q_BITS_8 + qp / 6 - OFFSET_BITS);
        let params = SCALES_8x8[qp % 6].map(|i| QuantParam::new(offset, i));

        let x = Matrix::<8, 8>::from_iter((0..64).map(|_| rng.gen::<i16>() as i16));
        let y = x.quant(qp / 6, &params, false);

        let x_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .quant(qp / 6, &Vec::new_constant(cs.clone(), params)?, false)?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }
}
