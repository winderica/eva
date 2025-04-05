pub mod constants;
pub mod constraints;
pub mod traits;

use constants::*;

use std::ops::{Deref, Index, IndexMut};

use ark_ff::{Fp, FpConfig, PrimeField, Zero};
use ndarray::{array, s, Array2};
use num_bigint::BigUint;
use num_traits::ToPrimitive;

pub struct QuantParam {
    offset: u64,
    scale: u64,
}

impl QuantParam {
    pub fn new(offset: u64, scale: u64) -> Self {
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
pub struct Matrix<T, const M: usize, const N: usize>(Array2<T>);

impl<T, const M: usize, const N: usize> Deref for Matrix<T, M, N> {
    type Target = Array2<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const M: usize, const N: usize> Index<(usize, usize)> for Matrix<T, M, N> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.0[[i, j]]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for Matrix<T, M, N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.0[[i, j]]
    }
}

impl<T, const M: usize, const N: usize> FromIterator<T> for Matrix<T, M, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<T: Clone + Zero, const M: usize, const N: usize> Default for Matrix<T, M, N> {
    fn default() -> Self {
        Self(Array2::zeros((M, N)))
    }
}

impl<T: Clone + Zero, const M: usize, const N: usize> Matrix<T, M, N> {
    fn new() -> Self {
        Self(Array2::zeros((M, N)))
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn from_vec(v: Vec<T>) -> Self {
        Self(Array2::from_shape_vec((M, N), v).unwrap())
    }
}

impl<P: FpConfig<X>, const X: usize, const M: usize, const N: usize> Matrix<Fp<P, X>, M, N> {
    pub fn to_i8(self) -> Matrix<i8, M, N> {
        Matrix::from_iter(self.0.into_iter().map(|i| {
            if i.into_bigint() < Fp::<P, X>::MODULUS_MINUS_ONE_DIV_TWO {
                let i: BigUint = i.into();
                i.to_i8().unwrap()
            } else {
                let i: BigUint = (-i).into();
                -(i.to_i8().unwrap())
            }
        }))
    }

    pub fn to_u8(self) -> Matrix<u8, M, N> {
        Matrix::from_iter(self.0.into_iter().map(|i| {
            let i: BigUint = i.into();
            i.to_u8().unwrap()
        }))
    }
}

impl<const M: usize, const N: usize> Matrix<i64, M, N> {
    pub fn to_i8(self) -> Matrix<i8, M, N> {
        Matrix::from_iter(self.0.into_iter().map(|i| i.try_into().unwrap()))
    }

    pub fn to_u8(self) -> Matrix<u8, M, N> {
        Matrix::from_iter(self.0.into_iter().map(|i| i.try_into().unwrap()))
    }
}

impl<const M: usize, const N: usize> Matrix<i16, M, N> {
    fn dct_region<const P: usize, const Q: usize>(
        &self,
        i_offset: usize,
        j_offset: usize,
    ) -> Matrix<i16, P, Q> {
        match (P, Q) {
            (4, 4) => {
                let t = array![[1, 1, 1, 1], [2, 1, -1, -2], [1, -1, -1, 1], [1, -2, 2, -1],];
                Matrix::<i16, P, Q>(
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
                let mut r = Matrix::<i16, P, Q>::new();
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

    pub fn hadamard_quant(&self, qp_over_6: usize, param: &QuantParam) -> Self {
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

    pub fn quant(&self, qp_over_6: usize, params: &[QuantParam], ac_only: bool) -> Self {
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

impl Matrix<i16, 16, 16> {
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

        let offset = base_offset << (qp_over_6 + Q_BITS_4 - OFFSET_BITS);
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

        let mut dc_block = Matrix::<i16, 4, 4>::new();
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
            let mut block = Matrix::<i16, 4, 4>::new();
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

impl Matrix<i16, 8, 8> {
    pub fn encode_chroma(
        &self,
        pred: &Self,
        _is_i16x16: bool,
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

        let offset = base_offset << (qp_over_6 + Q_BITS_4 - OFFSET_BITS);
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

        let mut dc_block = Matrix::<i16, 2, 2>::new();
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
                .collect::<Matrix<i16, 2, 2>>()
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
    use crate::var::I64Var;

    use self::constraints::{MatrixVar, QuantParamVar};
    use self::traits::CastSlice;
    use super::*;
    use ark_pallas::Fr;
    use ark_r1cs_std::boolean::Boolean;
    use ark_r1cs_std::eq::EqGadget;
    use ark_r1cs_std::fields::fp::FpVar;
    use ark_r1cs_std::{alloc::AllocVar, R1CSVar};
    use ark_relations::r1cs::ConstraintSystem;
    use folding_schemes::frontend::LookupArgument;
    use rand::{thread_rng, Rng};
    use rayon::prelude::*;
    use std::error::Error;
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::path::Path;

    #[test]
    fn test_dct_4x4() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let x = Matrix::<i16, 4, 4>::from_iter((0..16).map(|_| rng.gen::<u8>() as i16));
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

        let x = Matrix::<i16, 8, 8>::from_iter((0..64).map(|_| rng.gen::<u8>() as i16));
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

        let x = Matrix::<i16, 2, 2>::from_iter((0..4).map(|_| rng.gen::<u8>() as i16));
        let y = x.hadamard_quant(4, &param);

        let x_var = MatrixVar::<FpVar<Fr>, 2, 2>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 2, 2>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .hadamard_quant(4, &QuantParamVar::<FpVar<Fr>>::constant(param))?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test_hadamard_quant_4x4() -> Result<(), Box<dyn Error>> {
        let rng = &mut thread_rng();

        let cs = ConstraintSystem::<Fr>::new_ref();

        let param = QuantParam::new(174592, 8192);

        let x = Matrix::<i16, 4, 4>::from_iter((0..16).map(|_| rng.gen::<u8>() as i16));
        let y = x.hadamard_quant(4, &param);

        let x_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 4, 4>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .hadamard_quant(4, &QuantParamVar::<FpVar<Fr>>::constant(param))?
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

        let x = Matrix::<i16, 4, 4>::from_iter((0..16).map(|_| rng.gen::<i16>() as i16));
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

        let x = Matrix::<i16, 8, 8>::from_iter((0..64).map(|_| rng.gen::<i16>() as i16));
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

        let x = Matrix::<i16, 4, 4>::from_iter((0..16).map(|_| rng.gen::<i16>() as i16));
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

        let x = Matrix::<i16, 8, 8>::from_iter((0..64).map(|_| rng.gen::<i16>() as i16));
        let y = x.quant(qp / 6, &params, false);

        let x_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(x))?;
        let y_var = MatrixVar::<FpVar<Fr>, 8, 8>::new_witness(cs.clone(), || Ok(y))?;
        x_var
            .quant(qp / 6, &Vec::new_constant(cs.clone(), params)?, false)?
            .enforce_equal(&y_var)?;

        assert!(cs.is_satisfied().unwrap());
        Ok(())
    }

    #[test]
    fn test() -> Result<(), Box<dyn Error>> {
        let path = Path::new(env!("DATA_PATH")).join("bunny2");

        let mut type_reader = BufReader::new(File::open(path.join("type_enc")).unwrap());
        let mut orig_y_reader = BufReader::new(File::open(path.join("orig_y_enc")).unwrap());
        let mut orig_u_reader = BufReader::new(File::open(path.join("orig_u_enc")).unwrap());
        let mut orig_v_reader = BufReader::new(File::open(path.join("orig_v_enc")).unwrap());
        let mut pred_y_reader = BufReader::new(File::open(path.join("pred_y_enc")).unwrap());
        let mut pred_u_reader = BufReader::new(File::open(path.join("pred_u_enc")).unwrap());
        let mut pred_v_reader = BufReader::new(File::open(path.join("pred_v_enc")).unwrap());
        let mut result_y_reader = BufReader::new(File::open(path.join("coeff_y_enc")).unwrap());
        let mut result_u_reader = BufReader::new(File::open(path.join("coeff_u_enc")).unwrap());
        let mut result_v_reader = BufReader::new(File::open(path.join("coeff_v_enc")).unwrap());

        let mut orig_y = [0; 256];
        let mut orig_u = [0; 64];
        let mut orig_v = [0; 64];

        let mut i = 0;

        while let (Ok(_), Ok(_), Ok(_)) = (
            orig_y_reader.read_exact(&mut orig_y),
            orig_u_reader.read_exact(&mut orig_u),
            orig_v_reader.read_exact(&mut orig_v),
        ) {
            let mut pred_y = [0; 256];
            let mut pred_u = [0; 64];
            let mut pred_v = [0; 64];
            pred_y_reader.read_exact(&mut pred_y).unwrap();
            pred_u_reader.read_exact(&mut pred_u).unwrap();
            pred_v_reader.read_exact(&mut pred_v).unwrap();

            let mut types = [0; 6];
            type_reader.read_exact(&mut types).unwrap();
            let mb_type = match types[1] {
                1 => MacroblockType::P16x16,
                2 => MacroblockType::P16x8,
                3 => MacroblockType::P8x16,
                8 => MacroblockType::P8x8,
                9 => MacroblockType::I4x4,
                10 => MacroblockType::I16x16,
                13 => MacroblockType::I8x8,
                0 => MacroblockType::SKIP,
                _ => panic!("unknown type"),
            };

            if mb_type == MacroblockType::SKIP {
                println!("{} {}", types[1], types[3]);
                if types[0] == 1 && types[3] != 0 {
                } else {
                    i += 1;
                    continue;
                }
            }

            let mut result_y = [0; 256];
            let mut result_u = [0; 64];
            let mut result_v = [0; 64];
            result_y_reader.read_exact(&mut result_y).unwrap();
            result_u_reader.read_exact(&mut result_u).unwrap();
            result_v_reader.read_exact(&mut result_v).unwrap();
            let result_y = result_y
                .cast::<i8>()
                .iter()
                .map(|&i| i as i16)
                .collect::<Matrix<i16, 16, 16>>();
            let result_u = result_u
                .cast::<i8>()
                .iter()
                .map(|&i| i as i16)
                .collect::<Matrix<i16, 8, 8>>();
            let result_v = result_v
                .cast::<i8>()
                .iter()
                .map(|&i| i as i16)
                .collect::<Matrix<i16, 8, 8>>();

            let y_is_8x8 = types[2] == 1;
            assert!(!y_is_8x8);

            let is_i_slice = types[0] == 2;
            let qp = types[4] as usize;
            let qpc = types[5] as usize;
            let coeff_y = orig_y
                .iter()
                .map(|&i| i as i16)
                .collect::<Matrix<i16, 16, 16>>()
                .encode_luma_4x4(
                    &pred_y
                        .iter()
                        .map(|&i| i as i16)
                        .collect::<Matrix<i16, 16, 16>>(),
                    mb_type == MacroblockType::I16x16,
                    is_i_slice,
                    qp,
                );
            let coeff_u = orig_u
                .iter()
                .map(|&i| i as i16)
                .collect::<Matrix<i16, 8, 8>>()
                .encode_chroma(
                    &pred_u
                        .iter()
                        .map(|&i| i as i16)
                        .collect::<Matrix<i16, 8, 8>>(),
                    mb_type == MacroblockType::I16x16,
                    is_i_slice,
                    qpc,
                );
            let coeff_v = orig_v
                .iter()
                .map(|&i| i as i16)
                .collect::<Matrix<i16, 8, 8>>()
                .encode_chroma(
                    &pred_v
                        .iter()
                        .map(|&i| i as i16)
                        .collect::<Matrix<i16, 8, 8>>(),
                    mb_type == MacroblockType::I16x16,
                    is_i_slice,
                    qpc,
                );
            if coeff_y != result_y || coeff_u != result_u || coeff_v != result_v {
                for i in 0..16 {
                    for j in 0..16 {
                        if coeff_y[(i, j)] != result_y[(i, j)] {
                            println!("Y {} {} {} {}", i, j, coeff_y[(i, j)], result_y[(i, j)]);
                        }
                    }
                }
                for i in 0..8 {
                    for j in 0..8 {
                        if coeff_u[(i, j)] != result_u[(i, j)] {
                            println!("U {} {} {} {}", i, j, coeff_u[(i, j)], result_u[(i, j)]);
                        }
                    }
                }
                for i in 0..8 {
                    for j in 0..8 {
                        if coeff_v[(i, j)] != result_v[(i, j)] {
                            println!("V {} {} {} {}", i, j, coeff_v[(i, j)], result_v[(i, j)]);
                        }
                    }
                }
                println!(
                    "{} {} {:?} {} {} {} {}",
                    i, types[0], mb_type, types[2], types[3], qp, qpc,
                );
            }
            i += 1;
        }

        Ok(())
    }

    #[test]
    fn test_gadget() -> Result<(), Box<dyn Error>> {
        let scales = [
            [13107, 11916, 10082, 9362, 8192, 7282],
            [8066, 7490, 6554, 5825, 5243, 4559],
            [5243, 4660, 4194, 3647, 3355, 2893],
        ];

        let path = Path::new(env!("DATA_PATH")).join("foreman_cut");

        let types = std::fs::read(path.join("type_enc"))?;
        let orig_y = std::fs::read(path.join("orig_y_enc"))?;
        let orig_u = std::fs::read(path.join("orig_u_enc"))?;
        let orig_v = std::fs::read(path.join("orig_v_enc"))?;
        let pred_y = std::fs::read(path.join("pred_y_enc"))?;
        let pred_u = std::fs::read(path.join("pred_u_enc"))?;
        let pred_v = std::fs::read(path.join("pred_v_enc"))?;
        let result_y = std::fs::read(path.join("coeff_y_enc"))?;
        let result_u = std::fs::read(path.join("coeff_u_enc"))?;
        let result_v = std::fs::read(path.join("coeff_v_enc"))?;

        types
            .par_chunks_exact(6)
            .zip(
                orig_y
                    .par_chunks_exact(256)
                    .zip(orig_u.par_chunks_exact(64).zip(orig_v.par_chunks_exact(64))),
            )
            .zip(
                pred_y
                    .par_chunks_exact(256)
                    .zip(pred_u.par_chunks_exact(64).zip(pred_v.par_chunks_exact(64))),
            )
            .zip(
                result_y.cast::<i8>().par_chunks_exact(256).zip(
                    result_u
                        .cast::<i8>()
                        .par_chunks_exact(64)
                        .zip(result_v.cast::<i8>().par_chunks_exact(64)),
                ),
            )
            .enumerate()
            .take(5)
            .for_each(
                |(
                    i,
                    (
                        ((types, (orig_y, (orig_u, orig_v))), (pred_y, (pred_u, pred_v))),
                        (result_y, (result_u, result_v)),
                    ),
                )| {
                    let cs = ConstraintSystem::<Fr>::new_ref();
                    let la = LookupArgument::new_ref();
                    la.set_table((0u32..256).map(Fr::from).collect());

                    let mb_type = match types[1] {
                        1 => MacroblockType::P16x16,
                        2 => MacroblockType::P16x8,
                        3 => MacroblockType::P8x16,
                        8 => MacroblockType::P8x8,
                        9 => MacroblockType::I4x4,
                        10 => MacroblockType::I16x16,
                        13 => MacroblockType::I8x8,
                        0 => panic!("no skip"),
                        _ => panic!("unknown type"),
                    };

                    let qp = types[4] as usize;
                    let qpc = types[5] as usize;
                    let is_i_slice = types[0] == 2;

                    let two_to_qp_over_6 =
                        I64Var::new_witness(cs.clone(), || Ok(1 << (qp / 6))).unwrap();
                    let two_to_qpc_over_6 =
                        I64Var::new_witness(cs.clone(), || Ok(1 << (qpc / 6))).unwrap();

                    let scale = (
                        I64Var::new_witness(cs.clone(), || Ok(scales[0][qp % 6])).unwrap(),
                        I64Var::new_witness(cs.clone(), || Ok(scales[1][qp % 6])).unwrap(),
                        I64Var::new_witness(cs.clone(), || Ok(scales[2][qp % 6])).unwrap(),
                    );
                    let scalec = (
                        I64Var::new_witness(cs.clone(), || Ok(scales[0][qpc % 6])).unwrap(),
                        I64Var::new_witness(cs.clone(), || Ok(scales[1][qpc % 6])).unwrap(),
                        I64Var::new_witness(cs.clone(), || Ok(scales[2][qpc % 6])).unwrap(),
                    );

                    let base_offset = Boolean::new_witness(cs.clone(), || Ok(is_i_slice))
                        .unwrap()
                        .select(
                            &I64Var::constant(BASE_OFFSET_I_SLICE),
                            &I64Var::constant(BASE_OFFSET_BP_SLICE),
                        )
                        .unwrap();
                    let offset = &base_offset * &two_to_qp_over_6 * (1 << (Q_BITS_4 - OFFSET_BITS));
                    let offset_c =
                        &base_offset * &two_to_qpc_over_6 * (1 << (Q_BITS_4 - OFFSET_BITS));

                    let coeff_y = orig_y
                        .iter()
                        .map(|&i| I64Var::new_witness(cs.clone(), || Ok(i)))
                        .collect::<Result<MatrixVar<I64Var<Fr>, 16, 16>, _>>()
                        .unwrap()
                        .encode_luma_4x4::<true>(
                            cs.clone(),
                            la.clone(),
                            &pred_y
                                .iter()
                                .map(|&i| I64Var::new_witness(cs.clone(), || Ok(i)))
                                .collect::<Result<MatrixVar<I64Var<Fr>, 16, 16>, _>>()
                                .unwrap(),
                            &Boolean::new_witness(cs.clone(), || {
                                Ok(mb_type == MacroblockType::I16x16)
                            })
                            .unwrap(),
                            &offset,
                            &two_to_qp_over_6,
                            &scale,
                        )
                        .unwrap()
                        .value()
                        .unwrap()
                        .to_i8();
                    let coeff_u = orig_u
                        .iter()
                        .map(|&i| I64Var::new_witness(cs.clone(), || Ok(i)))
                        .collect::<Result<MatrixVar<I64Var<Fr>, 8, 8>, _>>()
                        .unwrap()
                        .encode_chroma::<true>(
                            cs.clone(),
                            la.clone(),
                            &pred_u
                                .iter()
                                .map(|&i| I64Var::new_witness(cs.clone(), || Ok(i)))
                                .collect::<Result<MatrixVar<I64Var<Fr>, 8, 8>, _>>()
                                .unwrap(),
                            &offset_c,
                            &two_to_qpc_over_6,
                            &scalec,
                        )
                        .unwrap()
                        .value()
                        .unwrap()
                        .to_i8();
                    let coeff_v = orig_v
                        .iter()
                        .map(|&i| I64Var::new_witness(cs.clone(), || Ok(i)))
                        .collect::<Result<MatrixVar<I64Var<Fr>, 8, 8>, _>>()
                        .unwrap()
                        .encode_chroma::<true>(
                            cs.clone(),
                            la.clone(),
                            &pred_v
                                .iter()
                                .map(|&i| I64Var::new_witness(cs.clone(), || Ok(i)))
                                .collect::<Result<MatrixVar<I64Var<Fr>, 8, 8>, _>>()
                                .unwrap(),
                            &offset_c,
                            &two_to_qpc_over_6,
                            &scalec,
                        )
                        .unwrap()
                        .value()
                        .unwrap()
                        .to_i8();
                    let result_y = result_y.iter().map(|&i| i).collect::<Matrix<i8, 16, 16>>();
                    let result_u = result_u.iter().map(|&i| i).collect::<Matrix<i8, 8, 8>>();
                    let result_v = result_v.iter().map(|&i| i).collect::<Matrix<i8, 8, 8>>();

                    if coeff_y != result_y || coeff_u != result_u || coeff_v != result_v {
                        for i in 0..16 {
                            for j in 0..16 {
                                if coeff_y[(i, j)] != result_y[(i, j)] {
                                    println!(
                                        "Y {} {} {} {}",
                                        i,
                                        j,
                                        coeff_y[(i, j)],
                                        result_y[(i, j)]
                                    );
                                }
                            }
                        }
                        for i in 0..8 {
                            for j in 0..8 {
                                if coeff_u[(i, j)] != result_u[(i, j)] {
                                    println!(
                                        "U {} {} {} {}",
                                        i,
                                        j,
                                        coeff_u[(i, j)],
                                        result_u[(i, j)]
                                    );
                                }
                            }
                        }
                        for i in 0..8 {
                            for j in 0..8 {
                                if coeff_v[(i, j)] != result_v[(i, j)] {
                                    println!(
                                        "V {} {} {} {}",
                                        i,
                                        j,
                                        coeff_v[(i, j)],
                                        result_v[(i, j)]
                                    );
                                }
                            }
                        }
                        println!(
                            "{} {} {:?} {} {} {} {}",
                            i, types[0], mb_type, types[2], types[3], qp, qpc,
                        );
                    }
                },
            );

        Ok(())
    }

    #[test]
    fn test_max() -> Result<(), Box<dyn Error>> {
        let path = Path::new(env!("DATA_PATH"));

        for f in [
            // "data",
            "foreman",
            "foreman_bright",
            "foreman_crop",
            "foreman_cut",
            "foreman_gray",
            "foreman_inv",
            "foreman_mask",
        ] {
            let path = path.parent().unwrap().join("data_raw").join(f);

            let mut result_y_reader = BufReader::new(File::open(path.join("coeff_y_enc")).unwrap());
            let mut result_u_reader = BufReader::new(File::open(path.join("coeff_u_enc")).unwrap());
            let mut result_v_reader = BufReader::new(File::open(path.join("coeff_v_enc")).unwrap());

            let mut result_y = [0; 256 * 4];
            let mut result_u = [0; 64 * 4];
            let mut result_v = [0; 64 * 4];

            let mut i = 0;

            loop {
                match (
                    result_y_reader.read_exact(&mut result_y),
                    result_u_reader.read_exact(&mut result_u),
                    result_v_reader.read_exact(&mut result_v),
                ) {
                    (Ok(_), Ok(_), Ok(_)) => {
                        let j = *result_y
                            .cast::<i32>()
                            .iter()
                            .chain(result_u.cast())
                            .chain(result_v.cast())
                            .min()
                            .unwrap();
                        if j < i {
                            i = j;
                        }
                    }
                    _ => break,
                }
            }
            println!("{}", i);
        }

        Ok(())
    }
}
