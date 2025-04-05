use std::borrow::Borrow;
use std::ops::{Deref, Index, IndexMut};

use ark_ff::{BigInteger, One, PrimeField};
use ark_r1cs_std::alloc::{AllocVar, AllocationMode};
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::{fields::FieldVar, R1CSVar};
use ark_relations::r1cs::{ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::log2;
use folding_schemes::frontend::LookupArgumentRef;
use ndarray::{s, Array2};
use num_bigint::BigUint;
use num_traits::AsPrimitive;

use crate::var::I64Var;
use crate::COEFF_BITS;

use super::{constants::*, Matrix, QuantParam};

pub struct QuantParamVar<T> {
    offset: T,
    scale: T,
}

impl<F: PrimeField> AllocVar<QuantParam, F> for QuantParamVar<I64Var<F>> {
    fn new_variable<T: Borrow<QuantParam>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();
        let param = f()?;
        let param = param.borrow();
        let offset = I64Var::new_variable(cs.clone(), || Ok(param.offset), mode)?;
        let scale = I64Var::new_variable(cs.clone(), || Ok(param.scale), mode)?;
        Ok(Self { offset, scale })
    }
}

impl<F: PrimeField> QuantParamVar<I64Var<F>> {
    pub fn constant<T: Into<QuantParam>>(param: T) -> Self {
        let param = param.into();
        let offset = I64Var::constant(param.offset);
        let scale = I64Var::constant(param.scale);
        Self { offset, scale }
    }
}

impl<F: PrimeField> AllocVar<QuantParam, F> for QuantParamVar<FpVar<F>> {
    fn new_variable<T: Borrow<QuantParam>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();
        let param = f()?;
        let param = param.borrow();
        let offset = FpVar::new_variable(cs.clone(), || Ok(F::from(param.offset)), mode)?;
        let scale = FpVar::new_variable(cs.clone(), || Ok(F::from(param.scale)), mode)?;
        Ok(Self { offset, scale })
    }
}

impl<F: PrimeField> QuantParamVar<FpVar<F>> {
    pub fn constant<T: Into<QuantParam>>(param: T) -> Self {
        let param = param.into();
        let offset = FpVar::constant(F::from(param.offset));
        let scale = FpVar::constant(F::from(param.scale));
        Self { offset, scale }
    }
}

#[derive(Clone)]
pub struct MatrixVar<T, const M: usize, const N: usize>(pub Array2<T>);

impl<T, const M: usize, const N: usize> Deref for MatrixVar<T, M, N> {
    type Target = Array2<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const M: usize, const N: usize> Index<(usize, usize)> for MatrixVar<T, M, N> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.0[[i, j]]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for MatrixVar<T, M, N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.0[[i, j]]
    }
}

impl<T, const M: usize, const N: usize> FromIterator<T> for MatrixVar<T, M, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<S: Copy, F: PrimeField + From<S>, const M: usize, const N: usize> AllocVar<Matrix<S, M, N>, F>
    for MatrixVar<FpVar<F>, M, N>
{
    fn new_variable<T: Borrow<Matrix<S, M, N>>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let cs = cs.into().cs();
        let matrix = f()?;
        matrix
            .borrow()
            .iter()
            .map(|&v| FpVar::new_variable(cs.clone(), || Ok(F::from(v)), mode))
            .collect()
    }
}

impl<S: Copy + AsPrimitive<i64>, F: PrimeField, const M: usize, const N: usize>
    AllocVar<Matrix<S, M, N>, F> for MatrixVar<I64Var<F>, M, N>
{
    fn new_variable<T: Borrow<Matrix<S, M, N>>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let cs = cs.into().cs();
        let matrix = f()?;
        matrix
            .borrow()
            .iter()
            .map(|&v| I64Var::new_variable(cs.clone(), || Ok(v), mode))
            .collect()
    }
}

impl<F: PrimeField, const M: usize, const N: usize> EqGadget<F> for MatrixVar<FpVar<F>, M, N> {
    fn is_eq(&self, other: &Self) -> Result<Boolean<F>, SynthesisError> {
        let mut r = Boolean::TRUE;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            r &= a.is_eq(b)?;
        }
        Ok(r)
    }

    fn enforce_equal(&self, other: &Self) -> Result<(), SynthesisError> {
        for (_j, (a, b)) in self.0.iter().zip(other.0.iter()).enumerate() {
            // print!("{}: ", j);
            // let i = a.value()?;
            // if i.into_bigint() < F::MODULUS_MINUS_ONE_DIV_TWO {
            //     print!("{} ", i.into_bigint().to_string());
            // } else {
            //     print!("-{} ", (-i).into_bigint().to_string());
            // }
            // let i = b.value()?;
            // if i.into_bigint() < F::MODULUS_MINUS_ONE_DIV_TWO {
            //     print!("{} ", i.into_bigint().to_string());
            // } else {
            //     print!("-{} ", (-i).into_bigint().to_string());
            // }
            // println!();
            a.enforce_equal(b)?;
        }
        Ok(())
    }
}

impl<F: PrimeField, const M: usize, const N: usize> EqGadget<F> for MatrixVar<I64Var<F>, M, N> {
    fn is_eq(&self, other: &Self) -> Result<Boolean<F>, SynthesisError> {
        let mut r = Boolean::TRUE;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            r &= a.is_eq(b)?;
        }
        Ok(r)
    }

    fn enforce_equal(&self, other: &Self) -> Result<(), SynthesisError> {
        for (_j, (a, b)) in self.0.iter().zip(other.0.iter()).enumerate() {
            // print!("{}: ", j);
            // let i = a.value()?;
            // if i.into_bigint() < F::MODULUS_MINUS_ONE_DIV_TWO {
            //     print!("{} ", i.into_bigint().to_string());
            // } else {
            //     print!("-{} ", (-i).into_bigint().to_string());
            // }
            // let i = b.value()?;
            // if i.into_bigint() < F::MODULUS_MINUS_ONE_DIV_TWO {
            //     print!("{} ", i.into_bigint().to_string());
            // } else {
            //     print!("-{} ", (-i).into_bigint().to_string());
            // }
            // println!();
            a.enforce_equal(b)?;
        }
        Ok(())
    }
}

impl<F: PrimeField, const M: usize, const N: usize> R1CSVar<F> for MatrixVar<FpVar<F>, M, N> {
    type Value = Matrix<F, M, N>;

    fn cs(&self) -> ConstraintSystemRef<F> {
        let mut cs = ConstraintSystemRef::None;
        for v in self.0.iter() {
            cs = cs.or(v.cs());
        }
        cs
    }

    fn value(&self) -> Result<Self::Value, SynthesisError> {
        Ok(Matrix::from_vec(
            self.0
                .iter()
                .map(|v| {
                    let i = v.value()?;
                    Ok(i)
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<F: PrimeField, const M: usize, const N: usize> R1CSVar<F> for MatrixVar<I64Var<F>, M, N> {
    type Value = Matrix<i64, M, N>;

    fn cs(&self) -> ConstraintSystemRef<F> {
        let mut cs = ConstraintSystemRef::None;
        for v in self.0.iter() {
            cs = cs.or(v.cs());
        }
        cs
    }

    fn value(&self) -> Result<Self::Value, SynthesisError> {
        Ok(Matrix::from_vec(
            self.0
                .iter()
                .map(|v| {
                    let i = v.value()?;
                    Ok(i)
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<F: PrimeField, const M: usize, const N: usize> R1CSVar<F> for MatrixVar<Boolean<F>, M, N> {
    type Value = Matrix<bool, M, N>;

    fn cs(&self) -> ConstraintSystemRef<F> {
        let mut cs = ConstraintSystemRef::None;
        for v in self.0.iter() {
            cs = cs.or(v.cs());
        }
        cs
    }

    fn value(&self) -> Result<Self::Value, SynthesisError> {
        Ok(Matrix::from_vec(
            self.0
                .iter()
                .map(|v| {
                    let i = v.value()?;
                    Ok(i)
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<F: PrimeField, const M: usize, const N: usize> MatrixVar<FpVar<F>, M, N> {
    pub fn regroup(
        self,
        width: usize,
        la: Option<LookupArgumentRef<F>>,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        if let Some(la) = la {
            for i in self.iter() {
                i.enforce_bit_length(width, BoundingMode::TightForUnknownRange, la.clone())?;
            }
        }
        let s = F::from(BigUint::one() << width);

        self.0
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(F::MODULUS_BIT_SIZE as usize / width)
            .map(|c| {
                let mut r = FpVar::zero();
                for v in c {
                    r = r * s + v;
                }
                Ok(r)
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<T, const M: usize, const N: usize> MatrixVar<T, M, N> {
    fn from_vec(v: Vec<T>) -> Self {
        Self(Array2::from_shape_vec((M, N), v).unwrap())
    }
}

impl<F: PrimeField, const M: usize, const N: usize> MatrixVar<FpVar<F>, M, N> {
    fn new() -> Self {
        Self(Array2::from_elem((M, N), FpVar::zero()))
    }

    pub fn dct_region<const P: usize, const Q: usize>(
        &self,
        i_offset: usize,
        j_offset: usize,
    ) -> Result<MatrixVar<FpVar<F>, P, Q>, SynthesisError> {
        match (P, Q) {
            (4, 4) => {
                let mut t = vec![];
                for i in 0..4 {
                    let p = &self.slice(s![i_offset + i, j_offset..]);
                    t.push(&p[0] + &p[1] + &p[2] + &p[3]);
                    t.push(&p[0].double()? + &p[1] - &p[2] - &p[3].double()?);
                    t.push(&p[0] - &p[1] - &p[2] + &p[3]);
                    t.push(&p[0] - &p[1].double()? + &p[2].double()? - &p[3]);
                }
                let mut r = MatrixVar::<FpVar<F>, P, Q>::new();
                for i in 0..4 {
                    r[(0, i)] = &t[i] + &t[i + 4] + &t[i + 8] + &t[i + 12];
                    r[(1, i)] = &t[i].double()? + &t[i + 4] - &t[i + 8] - &t[i + 12].double()?;
                    r[(2, i)] = &t[i] - &t[i + 4] - &t[i + 8] + &t[i + 12];
                    r[(3, i)] = &t[i] - &t[i + 4].double()? + &t[i + 8].double()? - &t[i + 12];
                }
                Ok(r)
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
                        &a[5] + &a[6] + &a[4] + rshift(&a[4], 8, 1)?,
                        &a[4] - &a[7] - &a[6] - rshift(&a[6], 8, 1)?,
                        &a[4] + &a[7] - &a[5] - rshift(&a[5], 8, 1)?,
                        &a[5] - &a[6] + &a[7] + rshift(&a[7], 8, 1)?,
                    ];
                    t.push(&b[0] + &b[1]);
                    t.push(&b[4] + rshift(&b[7], 11, 2)?);
                    t.push(&b[2] + rshift(&b[3], 9, 1)?);
                    t.push(&b[5] + rshift(&b[6], 11, 2)?);
                    t.push(&b[0] - &b[1]);
                    t.push(&b[6] - rshift(&b[5], 11, 2)?);
                    t.push(rshift(&b[2], 9, 1)? - &b[3]);
                    t.push(rshift(&b[4], 11, 2)? - &b[7]);
                }
                let mut r = MatrixVar::<FpVar<F>, P, Q>::new();
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
                        &a[5] + &a[6] + &a[4] + rshift(&a[4], 12, 1)?,
                        &a[4] - &a[7] - &a[6] - rshift(&a[6], 12, 1)?,
                        &a[4] + &a[7] - &a[5] - rshift(&a[5], 12, 1)?,
                        &a[5] - &a[6] + &a[7] + rshift(&a[7], 12, 1)?,
                    ];
                    r[(0, i)] = &b[0] + &b[1];
                    r[(1, i)] = &b[4] + rshift(&b[7], 14, 2)?;
                    r[(2, i)] = &b[2] + rshift(&b[3], 13, 1)?;
                    r[(3, i)] = &b[5] + rshift(&b[6], 14, 2)?;
                    r[(4, i)] = &b[0] - &b[1];
                    r[(5, i)] = &b[6] - rshift(&b[5], 14, 2)?;
                    r[(6, i)] = rshift(&b[2], 13, 1)? - &b[3];
                    r[(7, i)] = rshift(&b[4], 14, 2)? - &b[7];
                }
                Ok(r)
            }
            _ => unimplemented!(),
        }
    }

    pub fn hadamard_quant(
        &self,
        qp_over_6: usize,
        param: &QuantParamVar<FpVar<F>>,
    ) -> Result<Self, SynthesisError> {
        match (M, N) {
            (4, 4) => {
                let mut t = vec![];
                for i in 0..4 {
                    let row = &self.row(i);
                    t.push(&row[0] + &row[1] + &row[2] + &row[3]);
                    t.push(&row[0] + &row[1] - &row[2] - &row[3]);
                    t.push(&row[0] - &row[1] - &row[2] + &row[3]);
                    t.push(&row[0] - &row[1] + &row[2] - &row[3]);
                }
                let mut r = Self::new();
                for i in 0..4 {
                    for (j, v) in vec![
                        &t[i] + &t[i + 4] + &t[i + 8] + &t[i + 12],
                        &t[i] + &t[i + 4] - &t[i + 8] - &t[i + 12],
                        &t[i] - &t[i + 4] - &t[i + 8] + &t[i + 12],
                        &t[i] - &t[i + 4] + &t[i + 8] - &t[i + 12],
                    ]
                    .iter()
                    .enumerate()
                    {
                        let cs = v.cs();
                        let mode = if cs.is_none() {
                            AllocationMode::Constant
                        } else {
                            AllocationMode::Witness
                        };
                        let is_neg = {
                            Boolean::<F>::new_variable(
                                cs.clone(),
                                || {
                                    Ok(v.value().unwrap().into_bigint()
                                        > F::MODULUS_MINUS_ONE_DIV_TWO)
                                },
                                mode,
                            )?
                        };
                        let abs = is_neg.select(&v.negate()?, &v)?;
                        let bits = to_bits(&(abs * &param.scale + param.offset.double()?), 32)?;
                        let level = Boolean::le_bits_to_fp(&bits[Q_BITS_4 + qp_over_6 + 1..])?;
                        r[(j, i)] = is_neg.select(&level.negate()?, &level)?;
                    }
                }

                Ok(r)
            }
            (2, 2) => {
                let p0 = &self[(0, 0)] + &self[(0, 1)];
                let p1 = &self[(0, 0)] - &self[(0, 1)];
                let p2 = &self[(1, 0)] + &self[(1, 1)];
                let p3 = &self[(1, 0)] - &self[(1, 1)];

                vec![&p0 + &p2, &p1 + &p3, &p0 - &p2, &p1 - &p3]
                    .iter()
                    .map(|v| {
                        let cs = v.cs();
                        let mode = if cs.is_none() {
                            AllocationMode::Constant
                        } else {
                            AllocationMode::Witness
                        };
                        let is_neg = {
                            Boolean::<F>::new_variable(
                                cs.clone(),
                                || {
                                    Ok(v.value().unwrap().into_bigint()
                                        > F::MODULUS_MINUS_ONE_DIV_TWO)
                                },
                                mode,
                            )?
                        };
                        let bits = to_bits(
                            &(is_neg.select(&v.negate()?, &v)? * &param.scale
                                + param.offset.double()?),
                            32,
                        )?;
                        let level = Boolean::le_bits_to_fp(&bits[Q_BITS_4 + qp_over_6 + 1..])?;
                        is_neg.select(&level.negate()?, &level)
                    })
                    .collect()
            }
            _ => unimplemented!(),
        }
    }

    pub fn quant(
        &self,
        qp_over_6: usize,
        params: &[QuantParamVar<FpVar<F>>],
        ac_only: bool,
    ) -> Result<Self, SynthesisError> {
        let q_bits = match (M, N) {
            (4, 4) => Q_BITS_4,
            (8, 8) => Q_BITS_8,
            _ => unimplemented!(),
        };

        match (M, N) {
            (4, 4) | (8, 8) => self
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if i == 0 && ac_only {
                        return Ok(FpVar::zero());
                    }
                    let param = &params[i];
                    let cs = v.cs();
                    let mode = if cs.is_none() {
                        AllocationMode::Constant
                    } else {
                        AllocationMode::Witness
                    };
                    let is_neg = {
                        Boolean::<F>::new_variable(
                            cs.clone(),
                            || Ok(v.value().unwrap().into_bigint() > F::MODULUS_MINUS_ONE_DIV_TWO),
                            mode,
                        )?
                    };
                    let bits = to_bits(
                        &(is_neg.select(&v.negate()?, &v)? * &param.scale + &param.offset),
                        32,
                    )?;
                    let level = Boolean::le_bits_to_fp(&bits[q_bits + qp_over_6..])?;
                    is_neg.select(&level.negate()?, &level)
                })
                .collect(),
            _ => unimplemented!(),
        }
    }
}

impl<F: PrimeField, const M: usize, const N: usize> MatrixVar<I64Var<F>, M, N> {
    fn new() -> Self {
        Self(Array2::from_elem((M, N), I64Var::zero()))
    }

    pub fn dct_region<const P: usize, const Q: usize>(
        &self,
        i_offset: usize,
        j_offset: usize,
    ) -> Result<MatrixVar<I64Var<F>, P, Q>, SynthesisError> {
        match (P, Q) {
            (4, 4) => {
                let mut t = vec![];
                for i in 0..4 {
                    let p = &self.slice(s![i_offset + i, j_offset..]);
                    t.push(&p[0] + &p[1] + &p[2] + &p[3]);
                    t.push(&p[0].double()? + &p[1] - &p[2] - &p[3].double()?);
                    t.push(&p[0] - &p[1] - &p[2] + &p[3]);
                    t.push(&p[0] - &p[1].double()? + &p[2].double()? - &p[3]);
                }
                let mut r = MatrixVar::<I64Var<F>, P, Q>::new();
                for i in 0..4 {
                    r[(0, i)] = &t[i] + &t[i + 4] + &t[i + 8] + &t[i + 12];
                    r[(1, i)] = &t[i].double()? + &t[i + 4] - &t[i + 8] - &t[i + 12].double()?;
                    r[(2, i)] = &t[i] - &t[i + 4] - &t[i + 8] + &t[i + 12];
                    r[(3, i)] = &t[i] - &t[i + 4].double()? + &t[i + 8].double()? - &t[i + 12];
                }
                Ok(r)
            }
            _ => unimplemented!(),
        }
    }
}

fn quant_core<F: PrimeField, const X: usize, const Y_DC: bool>(
    la: LookupArgumentRef<F>,
    v: I64Var<F>,
    is_neg: Boolean<F>,
    qp_over_6: usize,
    two_to_qp_over_6: &I64Var<F>,
) -> Result<I64Var<F>, SynthesisError> {
    let cs = v.cs();
    let v_val = v.value()?;

    let q = I64Var::new_variable_with_inferred_mode(cs.clone(), || {
        if is_neg.value()? {
            Ok(-(v_val >> (Q_BITS_4 + X + qp_over_6)))
        } else {
            Ok(v_val >> (Q_BITS_4 + X + qp_over_6))
        }
    })?;
    let q_abs = I64Var::from(!is_neg).mac_enforce_bit_length::<false>(
        &q.double()?,
        &q.negate()?,
        COEFF_BITS - 1,
        BoundingMode::Loose,
        la.clone(),
    )? - &q;
    if Y_DC {
        let s = I64Var::new_variable(
            cs.clone(),
            || Ok((v_val >> 1) & ((1 << qp_over_6) - 1)),
            if cs.is_none() {
                AllocationMode::Constant
            } else {
                AllocationMode::Committed
            },
        )?;
        (two_to_qp_over_6 - I64Var::one() - &s).enforce_bit_length(
            8,
            BoundingMode::TightForUnknownRange,
            la.clone(),
        )?;
        let _ = (v.div(2)? - s).mac_enforce_bit_length::<true>(
            two_to_qp_over_6,
            &(q_abs * -(1 << (Q_BITS_4 + 1))),
            Q_BITS_4 + 1,
            BoundingMode::TightForUnknownRange,
            la.clone(),
        )?;
    } else {
        let _ = (v * (1 << 8)).mac_enforce_bit_length::<true>(
            two_to_qp_over_6,
            &(q_abs * -(1 << (Q_BITS_4 + X + 8))),
            Q_BITS_4 + X + 8,
            BoundingMode::TightForSmallAbs,
            la.clone(),
        )?;
    }

    Ok(q)
}

impl<F: PrimeField> MatrixVar<I64Var<F>, 16, 16> {
    pub fn encode_luma_4x4<const DUMMY: bool>(
        &self,
        _cs: ConstraintSystemRef<F>,
        la: LookupArgumentRef<F>,
        pred: &Self,
        is_i16x16: &Boolean<F>,
        offset: &I64Var<F>,
        two_to_qp_over_6: &I64Var<F>,
        (scale0, scale1, scale2): &(I64Var<F>, I64Var<F>, I64Var<F>),
    ) -> Result<Self, SynthesisError> {
        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let diff = self
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| a - b)
            .collect::<Self>();
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Diff", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let qp_over_6 = two_to_qp_over_6.value()? as usize;
        let qp_over_6 = log2(qp_over_6) as usize;

        let blocks = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| diff.dct_region::<4, 4>(i * 4, j * 4))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut dc_block = MatrixVar::<I64Var<F>, 4, 4>::new();
        for i in 0..4 {
            for j in 0..4 {
                dc_block[(i, j)] = blocks[i][j][(0, 0)].clone();
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
            let mut block = MatrixVar::<I64Var<F>, 4, 4>::new();
            for i in 0..4 {
                for (j, v) in vec![
                    &t[i] + &t[i + 4] + &t[i + 8] + &t[i + 12],
                    &t[i] + &t[i + 4] - &t[i + 8] - &t[i + 12],
                    &t[i] - &t[i + 4] - &t[i + 8] + &t[i + 12],
                    &t[i] - &t[i + 4] + &t[i + 8] - &t[i + 12],
                ]
                .into_iter()
                .enumerate()
                {
                    block[(j, i)] = v;
                }
            }
            block
        };
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Transform", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let mut coeffs = vec![];

        for i in 0..4 {
            for j in 0..4 {
                for (k, v) in blocks[i / 2 * 2 + j / 2][i % 2 * 2 + j % 2]
                    .iter()
                    .enumerate()
                {
                    if k == 0 {
                        let (is_neg, is_odd) = {
                            let cs = v.cs();
                            let v = if is_i16x16.value()? {
                                &dc_block[(i / 2 * 2 + j / 2, i % 2 * 2 + j % 2)]
                            } else {
                                v
                            }
                            .value()?;
                            let is_neg = v < 0;
                            let is_odd = v % 2 != 0;
                            (
                                Boolean::<F>::new_variable_with_inferred_mode(cs.clone(), || {
                                    Ok(is_neg)
                                })?,
                                Boolean::<F>::new_variable_with_inferred_mode(cs.clone(), || {
                                    Ok(is_odd)
                                })?,
                            )
                        };
                        let v = is_i16x16.select(
                            &(&dc_block[(i / 2 * 2 + j / 2, i % 2 * 2 + j % 2)]
                                - I64Var::from(is_odd)),
                            &v.double()?.double()?,
                        )?;
                        let v = is_neg.select(&v.negate()?, &v)? * scale0 + offset * 4;
                        coeffs.push(quant_core::<_, 2, true>(
                            la.clone(),
                            v,
                            is_neg,
                            qp_over_6,
                            two_to_qp_over_6,
                        )?);
                    } else {
                        let is_neg = Boolean::<F>::new_variable_with_inferred_mode(v.cs(), || {
                            Ok(v.value()? < 0)
                        })?;
                        let v = is_neg.select(&v.negate()?, &v)?
                            * match k {
                                0 | 2 | 8 | 10 => scale0,
                                5 | 7 | 13 | 15 => scale2,
                                _ => scale1,
                            }
                            + offset;
                        coeffs.push(quant_core::<_, 0, false>(
                            la.clone(),
                            v,
                            is_neg,
                            qp_over_6,
                            two_to_qp_over_6,
                        )?);
                    }
                }
            }
        }
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Quantization", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        Ok(Self::from_vec(coeffs))
    }

    // pub fn encode_luma_8x8(
    //     &self,
    //     pred: &Self,
    //     slice_is_intra: bool,
    //     qp_over_6: usize,
    //     qp_mod_6: usize,
    // ) -> Result<Self, SynthesisError> {
    //     let base_offset = if slice_is_intra {
    //         BASE_OFFSET_I_SLICE
    //     } else {
    //         BASE_OFFSET_BP_SLICE
    //     };
    //     let diff = self
    //         .iter()
    //         .zip(pred.iter())
    //         .map(|(a, b)| a - b)
    //         .collect::<Self>();
    //     let offset = base_offset << (Q_BITS_8 + qp_over_6 - OFFSET_BITS);
    //     let params = SCALES_8x8[qp_mod_6].map(|i| QuantParamVar::constant((offset, i)));

    //     let blocks = (0..4)
    //         .map(|pos| {
    //             Ok(diff
    //                 .dct_region::<8, 8>((pos / 2) * 8, (pos % 2) * 8)?
    //                 .quant(qp_over_6, &params, false)?
    //                 .0
    //                 .into_iter()
    //                 .collect::<Vec<_>>())
    //         })
    //         .collect::<Result<Vec<_>, _>>()?
    //         .concat();
    //     Ok(Self::from_vec(blocks))
    // }
}

impl<F: PrimeField> MatrixVar<I64Var<F>, 8, 8> {
    pub fn encode_chroma<const DUMMY: bool>(
        &self,
        _cs: ConstraintSystemRef<F>,
        la: LookupArgumentRef<F>,
        pred: &Self,
        offset: &I64Var<F>,
        two_to_qp_over_6: &I64Var<F>,
        (scale0, scale1, scale2): &(I64Var<F>, I64Var<F>, I64Var<F>),
    ) -> Result<Self, SynthesisError> {
        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let qp_over_6 = two_to_qp_over_6.value()? as usize;
        let qp_over_6 = log2(qp_over_6) as usize;

        let diff = self
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| a - b)
            .collect::<Self>();
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Diff", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let blocks = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| diff.dct_region::<4, 4>(i * 4, j * 4))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut dc_block = MatrixVar::<I64Var<F>, 2, 2>::new();
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

            MatrixVar::<I64Var<F>, 2, 2>::from_vec(vec![&p0 + &p2, &p1 + &p3, &p0 - &p2, &p1 - &p3])
        };
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Transform", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        #[cfg(feature = "constraints")]
        let t = cs.num_constraints();
        let mut coeffs = vec![];

        for i in 0..2 {
            for j in 0..2 {
                for (k, v) in blocks[i][j].iter().enumerate() {
                    if k == 0 {
                        let v = dc_block[(i, j)].clone();
                        let is_neg = {
                            Boolean::<F>::new_variable_with_inferred_mode(v.cs(), || {
                                Ok(v.value()? < 0)
                            })?
                        };
                        let v = is_neg.select(&v.negate()?, &v)? * scale0 + offset.double()?;
                        coeffs.push(quant_core::<_, 1, false>(
                            la.clone(),
                            v,
                            is_neg,
                            qp_over_6,
                            two_to_qp_over_6,
                        )?);
                    } else {
                        let is_neg = Boolean::<F>::new_variable_with_inferred_mode(v.cs(), || {
                            Ok(v.value()? < 0)
                        })?;
                        let v = is_neg.select(&v.negate()?, &v)?
                            * match k {
                                0 | 2 | 8 | 10 => scale0,
                                5 | 7 | 13 | 15 => scale2,
                                _ => scale1,
                            }
                            + offset;
                        coeffs.push(quant_core::<_, 0, false>(
                            la.clone(),
                            v,
                            is_neg,
                            qp_over_6,
                            two_to_qp_over_6,
                        )?);
                    }
                }
            }
        }
        #[cfg(feature = "constraints")]
        if DUMMY {
            add_to_trace!(|| "Quantization", || format!(
                "{} constraints",
                cs.num_constraints() - t
            ));
        }

        Ok(Self::from_vec(coeffs))
    }
}

fn to_bits<F: PrimeField>(v: &FpVar<F>, w: usize) -> Result<Vec<Boolean<F>>, SynthesisError> {
    let cs = v.cs();
    let mode = if cs.is_none() {
        AllocationMode::Constant
    } else {
        AllocationMode::Witness
    };
    let bits = {
        let bits = v.value().unwrap().into_bigint().to_bits_le();
        Vec::<Boolean<F>>::new_variable(cs.clone(), || Ok(&bits[..w]), mode)?
    };
    v.enforce_equal(&Boolean::le_bits_to_fp(&bits)?)?;
    Ok(bits)
}

fn rshift<F: PrimeField>(v: &FpVar<F>, w: usize, n: usize) -> Result<FpVar<F>, SynthesisError> {
    let cs = v.cs();
    let mode = if cs.is_none() {
        AllocationMode::Constant
    } else {
        AllocationMode::Witness
    };
    let is_neg = {
        Boolean::<F>::new_variable(
            cs.clone(),
            || Ok(v.value().unwrap().into_bigint() > F::MODULUS_MINUS_ONE_DIV_TWO),
            mode,
        )?
    };
    let abs = is_neg.select(&(v.negate()? + F::from(BigUint::one() << n) - F::one()), v)?;
    let bits = to_bits(&abs, w)?;
    let q = Boolean::le_bits_to_fp(&bits[n..])?;
    is_neg.select(&q.negate()?, &q)
}

pub fn rshift_abs<F: PrimeField>(
    v: &FpVar<F>,
    w: usize,
    n: usize,
) -> Result<(Boolean<F>, FpVar<F>), SynthesisError> {
    let cs = v.cs();
    let mode = if cs.is_none() {
        AllocationMode::Constant
    } else {
        AllocationMode::Witness
    };
    let is_neg = {
        Boolean::<F>::new_variable(
            cs.clone(),
            || Ok(v.value().unwrap().into_bigint() > F::MODULUS_MINUS_ONE_DIV_TWO),
            mode,
        )?
    };
    let abs = is_neg.select(&(v.negate()? + F::from(BigUint::one() << n) - F::one()), v)?;
    let bits = to_bits(&abs, w)?;
    let q = Boolean::le_bits_to_fp(&bits[n..])?;
    Ok((is_neg, q))
}

#[derive(PartialEq)]
pub enum BoundingMode {
    Loose,
    TightForSmallAbs,
    TightForUnknownRange,
}

pub trait BitDecompose<F: PrimeField>
where
    Self: Sized,
{
    /// Enforce that `self * multiplicand + addend` has at most `length` bits, i.e.,
    /// `0 <= self * multiplicand + addend < 2^length`, and return `self * multiplicand`.
    /// If `mode` is `BoundingMode::Loose`, then the result can only be guaranteed to be in the
    /// range `[0, 2^(length.next_multiple_of(LOOKUP_TABLE_BITS)))`.
    /// If `mode` is `BoundingMode::SmallAbsToTight`, then the absolute value of `self *
    /// multiplicand + addend` should be small, and the result can be guaranteed to be in the
    /// range `[0, 2^length)`. Note that in this case, we only need to know beforehand the
    /// absolute value of the result is small, as internally `mac_enforce_bit_length` will shift
    /// the result to the left by a small amount and then check the bit length of the shifted
    /// result. When the result is a negative number with small absolute value, the shifted
    /// result is still negative and has small absolute value, whose field representation is
    /// large and will be rejected.
    fn mac_enforce_bit_length<const DIV: bool>(
        &self,
        multiplicand: &Self,
        addend: &Self,
        length: usize,
        mode: BoundingMode,
        la: LookupArgumentRef<F>,
    ) -> Result<Self, SynthesisError>;

    /// Enforce that `self` has at most `length` bits, i.e., `0 <= self < 2^length`.
    /// If `mode` is `BoundingMode::Loose`, then `self` can only be guaranteed to be in the range
    /// `[0, 2^(length.next_multiple_of(LOOKUP_TABLE_BITS)))`.
    /// If `mode` is `BoundingMode::SmallAbsToTight`, then the absolute value of `self` should be
    /// small, and `self` can be guaranteed to be in the range `[0, 2^length)`.
    fn enforce_bit_length(
        &self,
        length: usize,
        mode: BoundingMode,
        la: LookupArgumentRef<F>,
    ) -> Result<(), SynthesisError>;
}

impl<F: PrimeField> BitDecompose<F> for FpVar<F> {
    fn mac_enforce_bit_length<const DIV: bool>(
        &self,
        multiplicand: &Self,
        addend: &Self,
        length: usize,
        mode: BoundingMode,
        la: LookupArgumentRef<F>,
    ) -> Result<Self, SynthesisError> {
        let cs = self.cs();

        let table_size = (la.len() as i64).ilog2() as usize;

        let num_chunks = (length + table_size - 1) / table_size;
        let extended_length = num_chunks * table_size;
        let scale = if mode == BoundingMode::TightForSmallAbs {
            F::from(BigUint::one() << (extended_length - length))
        } else {
            F::one()
        };

        let v = if DIV {
            (self.value()? / multiplicand.value()? + addend.value()?) * scale
        } else {
            (self.value()? * multiplicand.value()? + addend.value()?) * scale
        };

        if num_chunks == 1 {
            let result = FpVar::new_variable(
                cs.clone(),
                || Ok(v),
                if cs.is_none() {
                    AllocationMode::Constant
                } else {
                    AllocationMode::Committed
                },
            )?;
            let product = &result * scale.inverse().unwrap() - addend;
            if DIV {
                product.mul_equals(multiplicand, self)?;
            } else {
                self.mul_equals(multiplicand, &product)?;
            }

            if mode == BoundingMode::TightForUnknownRange && length != extended_length {
                result.enforce_bit_length(
                    table_size + length - extended_length,
                    BoundingMode::TightForSmallAbs,
                    la,
                )?;
            }

            return Ok(product);
        }

        let chunks = {
            let chunks = if table_size == 8 {
                v.into_bigint()
                    .to_bytes_le()
                    .into_iter()
                    .take(num_chunks)
                    .map(F::from)
                    .collect::<Vec<_>>()
            } else {
                v.into_bigint()
                    .to_bits_le()
                    .chunks(table_size)
                    .take(num_chunks)
                    .map(|chunk| F::from_bigint(F::BigInt::from_bits_le(chunk)).unwrap())
                    .collect::<Vec<_>>()
            };

            Vec::<FpVar<_>>::new_variable(
                cs.clone(),
                || Ok(chunks),
                if cs.is_none() {
                    AllocationMode::Constant
                } else {
                    AllocationMode::Committed
                },
            )?
        };

        let mut accumulated = FpVar::zero();
        for (i, v) in chunks.iter().enumerate() {
            accumulated = &accumulated + v * F::from(BigUint::one() << (i * table_size));
        }
        let product = &accumulated * scale.inverse().unwrap() - addend;
        if DIV {
            product.mul_equals(multiplicand, &self)?;
        } else {
            self.mul_equals(multiplicand, &product)?;
        }

        if mode == BoundingMode::TightForUnknownRange && length != extended_length {
            chunks[num_chunks - 1].enforce_bit_length(
                table_size + length - extended_length,
                BoundingMode::TightForSmallAbs,
                la,
            )?;
        }

        Ok(product)
    }

    fn enforce_bit_length(
        &self,
        length: usize,
        mode: BoundingMode,
        la: LookupArgumentRef<F>,
    ) -> Result<(), SynthesisError> {
        let _ =
            self.mac_enforce_bit_length::<false>(&FpVar::one(), &FpVar::zero(), length, mode, la)?;
        Ok(())
    }
}

impl<F: PrimeField> BitDecompose<F> for I64Var<F> {
    fn mac_enforce_bit_length<const DIV: bool>(
        &self,
        multiplicand: &Self,
        addend: &Self,
        length: usize,
        mode: BoundingMode,
        la: LookupArgumentRef<F>,
    ) -> Result<Self, SynthesisError> {
        let cs = self.cs();

        let table_size = (la.len() as i64).ilog2() as usize;

        let num_chunks = (length + table_size - 1) / table_size;
        let extended_length = num_chunks * table_size;
        let scale = if mode == BoundingMode::TightForSmallAbs {
            1 << (extended_length - length)
        } else {
            1
        };

        let v = if DIV {
            (self.value()? / multiplicand.value()? + addend.value()?) * scale
        } else {
            (self.value()? * multiplicand.value()? + addend.value()?) * scale
        };

        if num_chunks == 1 {
            let result = I64Var::new_variable(
                cs.clone(),
                || Ok(v),
                if cs.is_none() {
                    AllocationMode::Constant
                } else {
                    AllocationMode::Committed
                },
            )?;
            let product = &result.div(scale)? - addend;
            if DIV {
                product.mul_equals(multiplicand, self)?;
            } else {
                self.mul_equals(multiplicand, &product)?;
            }

            if mode == BoundingMode::TightForUnknownRange && length != extended_length {
                result.enforce_bit_length(
                    table_size + length - extended_length,
                    BoundingMode::TightForSmallAbs,
                    la,
                )?;
            }

            return Ok(product);
        }

        let chunks = {
            let chunks = if table_size == 8 {
                (v as u64)
                    .to_le_bytes()
                    .into_iter()
                    .take(num_chunks)
                    .map(|i| i as i64)
                    .collect::<Vec<_>>()
            } else {
                let mut chunks = vec![];
                let mut v = v as u64;
                for _ in 0..num_chunks {
                    chunks.push((v & ((1 << table_size) - 1)) as i64);
                    v >>= table_size;
                }
                chunks
            };

            Vec::<I64Var<_>>::new_variable(
                cs.clone(),
                || Ok(chunks),
                if cs.is_none() {
                    AllocationMode::Constant
                } else {
                    AllocationMode::Committed
                },
            )?
        };

        let mut accumulated = I64Var::zero();
        for (i, v) in chunks.iter().enumerate() {
            accumulated = &accumulated + v * (1 << (i * table_size));
        }
        let product = accumulated.div(scale)? - addend;
        if DIV {
            product.mul_equals(multiplicand, &self)?;
        } else {
            self.mul_equals(multiplicand, &product)?;
        }

        if mode == BoundingMode::TightForUnknownRange && length != extended_length {
            chunks[num_chunks - 1].enforce_bit_length(
                table_size + length - extended_length,
                BoundingMode::TightForSmallAbs,
                la,
            )?;
        }

        Ok(product)
    }

    fn enforce_bit_length(
        &self,
        length: usize,
        mode: BoundingMode,
        la: LookupArgumentRef<F>,
    ) -> Result<(), SynthesisError> {
        let _ = self.mac_enforce_bit_length::<false>(
            &I64Var::one(),
            &I64Var::zero(),
            length,
            mode,
            la,
        )?;
        Ok(())
    }
}
