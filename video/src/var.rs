use ark_ff::{BigInteger, PrimeField};
use ark_relations::{
    lc,
    r1cs::{ConstraintSystemRef, Namespace, SynthesisError, Variable},
};
use num_traits::AsPrimitive;

use core::borrow::Borrow;

use ark_r1cs_std::{
    boolean::AllocatedBool,
    fields::{
        fp::{AllocatedFp, FpVar},
        FieldOpsBounds, FieldVar,
    },
    impl_bounded_ops, impl_ops,
    prelude::*,
    Assignment,
};

/// Represents a variable in the constraint system whose
/// value can be an arbitrary field element.
#[derive(Debug, Clone)]
#[must_use]
pub struct AllocatedI64<F: PrimeField> {
    pub(crate) value: i64,
    /// The allocated variable corresponding to `self` in `self.cs`.
    pub variable: Variable,
    /// The constraint system that `self` was allocated in.
    pub cs: ConstraintSystemRef<F>,
    enable_lc: bool,
}

impl<F: PrimeField> AllocatedI64<F> {
    /// Constructs a new `AllocatedFp` from a (optional) value, a low-level
    /// Variable, and a `ConstraintSystemRef`.
    pub fn new(value: i64, variable: Variable, cs: ConstraintSystemRef<F>) -> Self {
        Self {
            value,
            variable,
            enable_lc: cs.should_construct_matrices(),
            cs,
        }
    }
}

/// Represent variables corresponding to a field element in `F`.
#[derive(Clone, Debug)]
#[must_use]
pub enum I64Var<F: PrimeField> {
    /// Represents a constant in the constraint system, which means that
    /// it does not have a corresponding variable.
    Constant(i64),
    /// Represents an allocated variable constant in the constraint system.
    Var(AllocatedI64<F>),
}

impl<F: PrimeField> R1CSVar<F> for I64Var<F> {
    type Value = i64;

    fn cs(&self) -> ConstraintSystemRef<F> {
        match self {
            Self::Constant(_) => ConstraintSystemRef::None,
            Self::Var(a) => a.cs.clone(),
        }
    }

    fn value(&self) -> Result<Self::Value, SynthesisError> {
        match self {
            Self::Constant(v) => Ok(*v),
            Self::Var(v) => v.value(),
        }
    }
}

impl<F: PrimeField> From<Boolean<F>> for I64Var<F> {
    fn from(other: Boolean<F>) -> Self {
        if let Boolean::Constant(b) = other {
            Self::Constant(b as i64)
        } else {
            // `other` is a variable
            let cs = other.cs();
            let variable = cs
                .new_lc(if cs.should_construct_matrices() {
                    other.lc()
                } else {
                    lc!()
                })
                .unwrap();
            Self::Var(AllocatedI64::new(
                other.value().unwrap_or_default() as i64,
                variable,
                cs,
            ))
        }
    }
}

impl<F: PrimeField> From<AllocatedI64<F>> for I64Var<F> {
    fn from(other: AllocatedI64<F>) -> Self {
        Self::Var(other)
    }
}

impl<'a, F: PrimeField> FieldOpsBounds<'a, i64, Self> for I64Var<F> {}
impl<'a, F: PrimeField> FieldOpsBounds<'a, i64, I64Var<F>> for &'a I64Var<F> {}

impl<F: PrimeField> AllocatedI64<F> {
    /// Constructs `Self` from a `Boolean`: if `other` is false, this outputs
    /// `zero`, else it outputs `one`.
    pub fn from(other: Boolean<F>) -> Self {
        let cs = other.cs();
        let variable = cs
            .new_lc(if cs.should_construct_matrices() {
                other.lc()
            } else {
                lc!()
            })
            .unwrap();
        Self::new(other.value().unwrap_or_default() as i64, variable, cs)
    }

    /// Returns the value assigned to `self` in the underlying constraint system
    /// (if a value was assigned).
    pub fn value(&self) -> Result<i64, SynthesisError> {
        Ok(self.value)
    }

    /// Outputs `self + other`.
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn add(&self, other: &Self) -> Self {
        let value = self.value + other.value;

        let variable = self
            .cs
            .new_lc(if self.enable_lc {
                lc!() + self.variable + other.variable
            } else {
                lc!()
            })
            .unwrap();
        AllocatedI64::new(value, variable, self.cs.clone())
    }

    /// Add many allocated Fp elements together.
    ///
    /// This does not create any constraints and only creates one linear
    /// combination.
    pub fn add_many<B: Borrow<Self>, I: Iterator<Item = B>>(iter: I) -> Self {
        let mut cs = ConstraintSystemRef::None;
        let mut value = 0;
        let mut new_lc = lc!();

        let mut num_iters = 0;
        for variable in iter {
            let variable = variable.borrow();
            if !variable.cs.is_none() {
                cs = cs.or(variable.cs.clone());
            }
            value += variable.value().unwrap();
            if variable.enable_lc {
                new_lc = new_lc + variable.variable;
            }
            num_iters += 1;
        }
        assert_ne!(num_iters, 0);

        let variable = cs.new_lc(new_lc).unwrap();

        AllocatedI64::new(value, variable, cs)
    }

    /// Outputs `self - other`.
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn sub(&self, other: &Self) -> Self {
        let value = self.value - other.value;

        let variable = self
            .cs
            .new_lc(if self.enable_lc {
                lc!() + self.variable - other.variable
            } else {
                lc!()
            })
            .unwrap();
        AllocatedI64::new(value, variable, self.cs.clone())
    }

    /// Outputs `self * other`.
    ///
    /// This requires *one* constraint.
    #[tracing::instrument(target = "r1cs")]
    pub fn mul(&self, other: &Self) -> Self {
        let product =
            AllocatedI64::new_witness(self.cs.clone(), || Ok(self.value * &other.value)).unwrap();
        if self.enable_lc {
            self.cs
                .enforce_constraint(
                    lc!() + self.variable,
                    lc!() + other.variable,
                    lc!() + product.variable,
                )
                .unwrap();
        } else {
            self.cs.borrow_mut().unwrap().num_constraints += 1;
        }

        product
    }

    /// Output `self + other`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn add_constant(&self, other: i64) -> Self {
        if other == 0 {
            self.clone()
        } else {
            let value = self.value + other;
            let variable = self
                .cs
                .new_lc(if self.enable_lc {
                    lc!() + self.variable + (F::from(other), Variable::One)
                } else {
                    lc!()
                })
                .unwrap();
            AllocatedI64::new(value, variable, self.cs.clone())
        }
    }

    /// Output `self - other`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn sub_constant(&self, other: i64) -> Self {
        self.add_constant(-other)
    }

    /// Output `self * other`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn mul_constant(&self, other: i64) -> Self {
        if other == 1 {
            self.clone()
        } else {
            let value = self.value * other;
            let variable = self
                .cs
                .new_lc(if self.enable_lc {
                    lc!() + (F::from(other), self.variable)
                } else {
                    lc!()
                })
                .unwrap();
            AllocatedI64::new(value, variable, self.cs.clone())
        }
    }

    /// Output `self / other`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn div_constant(&self, other: i64) -> Self {
        if other == 1 {
            self.clone()
        } else {
            let value = self.value / other;
            let variable = self
                .cs
                .new_lc(if self.enable_lc {
                    lc!() + (F::from(other).inverse().unwrap(), self.variable)
                } else {
                    lc!()
                })
                .unwrap();
            AllocatedI64::new(value, variable, self.cs.clone())
        }
    }

    /// Output `self + self`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn double(&self) -> Result<Self, SynthesisError> {
        let value = self.value * 2;
        let variable = self.cs.new_lc(if self.enable_lc {
            lc!() + self.variable + self.variable
        } else {
            lc!()
        })?;
        Ok(Self::new(value, variable, self.cs.clone()))
    }

    /// Output `-self`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn negate(&self) -> Self {
        let mut result = self.clone();
        result.negate_in_place();
        result
    }

    /// Sets `self = -self`
    ///
    /// This does not create any constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn negate_in_place(&mut self) -> &mut Self {
        self.value = -self.value;
        self.variable = self
            .cs
            .new_lc(if self.enable_lc {
                lc!() - self.variable
            } else {
                lc!()
            })
            .unwrap();
        self
    }

    /// Outputs `self * self`
    ///
    /// This requires *one* constraint.
    #[tracing::instrument(target = "r1cs")]
    pub fn square(&self) -> Result<Self, SynthesisError> {
        Ok(self.mul(self))
    }

    /// Enforces that `self * other = result`.
    ///
    /// This requires *one* constraint.
    #[tracing::instrument(target = "r1cs")]
    pub fn mul_equals(&self, other: &Self, result: &Self) -> Result<(), SynthesisError> {
        if self.enable_lc {
            self.cs.enforce_constraint(
                lc!() + self.variable,
                lc!() + other.variable,
                lc!() + result.variable,
            )?;
        } else {
            self.cs.borrow_mut().unwrap().num_constraints += 1;
        }
        Ok(())
    }

    /// Enforces that `self * self = result`.
    ///
    /// This requires *one* constraint.
    #[tracing::instrument(target = "r1cs")]
    pub fn square_equals(&self, result: &Self) -> Result<(), SynthesisError> {
        if self.enable_lc {
            self.cs.enforce_constraint(
                lc!() + self.variable,
                lc!() + self.variable,
                lc!() + result.variable,
            )?;
        } else {
            self.cs.borrow_mut().unwrap().num_constraints += 1;
        }
        Ok(())
    }

    /// Outputs the bit `self == other`.
    ///
    /// This requires two constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn is_eq(&self, other: &Self) -> Result<Boolean<F>, SynthesisError> {
        Ok(!self.is_neq(other)?)
    }

    /// Outputs the bit `self != other`.
    ///
    /// This requires two constraints.
    #[tracing::instrument(target = "r1cs")]
    pub fn is_neq(&self, other: &Self) -> Result<Boolean<F>, SynthesisError> {
        // We don't need to enforce `is_not_equal` to be boolean here;
        // see the comments above the constraints below for why.
        let is_not_equal = Boolean::from(AllocatedBool::new_witness_without_booleanity_check(
            self.cs.clone(),
            || Ok(self.value != other.value),
        )?);
        let multiplier = self.cs.new_witness_variable(|| {
            if is_not_equal.value()? {
                F::from(self.value - other.value).inverse().get()
            } else {
                Ok(F::one())
            }
        })?;

        // Completeness:
        // Case 1: self != other:
        // ----------------------
        //   constraint 1:
        //   (self - other) * multiplier = is_not_equal
        //   => (non_zero) * multiplier = 1 (satisfied, because multiplier = 1/(self -
        // other)
        //
        //   constraint 2:
        //   (self - other) * not(is_not_equal) = 0
        //   => (non_zero) * not(1) = 0
        //   => (non_zero) * 0 = 0
        //
        // Case 2: self == other:
        // ----------------------
        //   constraint 1:
        //   (self - other) * multiplier = is_not_equal
        //   => 0 * multiplier = 0 (satisfied, because multiplier = 1
        //
        //   constraint 2:
        //   (self - other) * not(is_not_equal) = 0
        //   => 0 * not(0) = 0
        //   => 0 * 1 = 0
        //
        // --------------------------------------------------------------------
        //
        // Soundness:
        // Case 1: self != other, but is_not_equal != 1.
        // --------------------------------------------
        //   constraint 2:
        //   (self - other) * not(is_not_equal) = 0
        //   => (non_zero) * (1 - is_not_equal) = 0
        //   => non_zero = 0 (contradiction) || 1 - is_not_equal = 0 (contradiction)
        //
        // Case 2: self == other, but is_not_equal != 0.
        // --------------------------------------------
        //   constraint 1:
        //   (self - other) * multiplier = is_not_equal
        //   0 * multiplier = is_not_equal != 0 (unsatisfiable)
        //
        // That is, constraint 1 enforces that if self == other, then `is_not_equal = 0`
        // and constraint 2 enforces that if self != other, then `is_not_equal = 1`.
        // Since these are the only possible two cases, `is_not_equal` is always
        // constrained to 0 or 1.
        if self.enable_lc {
            self.cs.enforce_constraint(
                lc!() + self.variable - other.variable,
                lc!() + multiplier,
                is_not_equal.lc(),
            )?;
            self.cs.enforce_constraint(
                lc!() + self.variable - other.variable,
                (!&is_not_equal).lc(),
                lc!(),
            )?;
        } else {
            self.cs.borrow_mut().unwrap().num_constraints += 2;
        }
        Ok(is_not_equal)
    }

    /// Enforces that self == other if `should_enforce.is_eq(&Boolean::TRUE)`.
    ///
    /// This requires one constraint.
    #[tracing::instrument(target = "r1cs")]
    pub fn conditional_enforce_equal(
        &self,
        other: &Self,
        should_enforce: &Boolean<F>,
    ) -> Result<(), SynthesisError> {
        if self.enable_lc {
            self.cs.enforce_constraint(
                lc!() + self.variable - other.variable,
                lc!() + should_enforce.lc(),
                lc!(),
            )?;
        } else {
            self.cs.borrow_mut().unwrap().num_constraints += 1;
        }
        Ok(())
    }

    /// Enforces that self != other if `should_enforce.is_eq(&Boolean::TRUE)`.
    ///
    /// This requires one constraint.
    #[tracing::instrument(target = "r1cs")]
    pub fn conditional_enforce_not_equal(
        &self,
        other: &Self,
        should_enforce: &Boolean<F>,
    ) -> Result<(), SynthesisError> {
        let multiplier = self.cs.new_witness_variable(|| {
            if should_enforce.value()? {
                F::from(self.value - other.value).inverse().get()
            } else {
                Ok(F::zero())
            }
        })?;

        // The high level logic is as follows:
        // We want to check that self - other != 0. We do this by checking that
        // (self - other).inverse() exists. In more detail, we check the following:
        // If `should_enforce == true`, then we set `multiplier = (self -
        // other).inverse()`, and check that (self - other) * multiplier == 1.
        // (i.e., that the inverse exists)
        //
        // If `should_enforce == false`, then we set `multiplier == 0`, and check that
        // (self - other) * 0 == 0, which is always satisfied.

        if self.enable_lc {
            self.cs.enforce_constraint(
                lc!() + self.variable - other.variable,
                lc!() + multiplier,
                should_enforce.lc(),
            )?;
        } else {
            self.cs.borrow_mut().unwrap().num_constraints += 1;
        }

        Ok(())
    }
}

impl<F: PrimeField> CondSelectGadget<F> for AllocatedI64<F> {
    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn conditionally_select(
        cond: &Boolean<F>,
        true_val: &Self,
        false_val: &Self,
    ) -> Result<Self, SynthesisError> {
        match cond {
            Boolean::Constant(true) => Ok(true_val.clone()),
            Boolean::Constant(false) => Ok(false_val.clone()),
            _ => {
                let cs = cond.cs();
                let result = Self::new_witness(cs.clone(), || {
                    cond.value()
                        .and_then(|c| Ok(if c { true_val } else { false_val }.value))
                })?;
                // a = self; b = other; c = cond;
                //
                // r = c * a + (1  - c) * b
                // r = b + c * (a - b)
                // c * (a - b) = r - b
                if cs.should_construct_matrices() {
                    cs.enforce_constraint(
                        cond.lc(),
                        lc!() + true_val.variable - false_val.variable,
                        lc!() + result.variable - false_val.variable,
                    )?;
                } else {
                    cs.borrow_mut().unwrap().num_constraints += 1;
                }

                Ok(result)
            }
        }
    }
}

impl<F: PrimeField, S: AsPrimitive<i64>> AllocVar<S, F> for AllocatedI64<F> {
    fn new_variable<T: Borrow<S>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();
        if mode == AllocationMode::Constant {
            let v = *f()?.borrow();
            let lc = cs.new_lc(if cs.should_construct_matrices() {
                lc!() + (F::from(v.as_()), Variable::One)
            } else {
                lc!()
            })?;
            Ok(Self::new(v.as_(), lc, cs))
        } else {
            let mut value = 0;
            let value_generator = || {
                let t = *f()?.borrow();
                value = t.as_();
                Ok(F::from(value))
            };
            let variable = match mode {
                AllocationMode::Input => cs.new_input_variable(value_generator)?,
                AllocationMode::Witness => cs.new_witness_variable(value_generator)?,
                AllocationMode::Committed => cs.new_committed_variable(value_generator)?,
                _ => unreachable!(),
            };
            Ok(Self::new(value, variable, cs))
        }
    }
}

impl<F: PrimeField> I64Var<F> {
    pub fn constant<T: AsPrimitive<i64>>(f: T) -> Self {
        Self::Constant(f.as_())
    }

    pub fn to_fpvar(&self) -> FpVar<F> {
        match self {
            Self::Var(var) => FpVar::Var(AllocatedFp::new(
                Some(F::from(var.value)),
                var.variable,
                var.cs.clone(),
            )),
            Self::Constant(c) => FpVar::Constant(F::from(*c)),
        }
    }

    pub fn zero() -> Self {
        Self::Constant(0)
    }

    pub fn one() -> Self {
        Self::Constant(1)
    }

    /// Returns a `Boolean` representing whether `self == Self::zero()`.
    pub fn is_zero(&self) -> Result<Boolean<F>, SynthesisError> {
        self.is_eq(&Self::zero())
    }

    /// Returns a `Boolean` representing whether `self == Self::one()`.
    pub fn is_one(&self) -> Result<Boolean<F>, SynthesisError> {
        self.is_eq(&Self::one())
    }

    #[tracing::instrument(target = "r1cs")]
    pub fn double(&self) -> Result<Self, SynthesisError> {
        match self {
            Self::Constant(c) => Ok(Self::Constant(c * 2)),
            Self::Var(v) => Ok(Self::Var(v.double()?)),
        }
    }

    #[tracing::instrument(target = "r1cs")]
    pub fn negate(&self) -> Result<Self, SynthesisError> {
        match self {
            Self::Constant(c) => Ok(Self::Constant(-c)),
            Self::Var(v) => Ok(Self::Var(v.negate())),
        }
    }

    #[tracing::instrument(target = "r1cs")]
    pub fn square(&self) -> Result<Self, SynthesisError> {
        match self {
            Self::Constant(c) => Ok(Self::Constant(c * c)),
            Self::Var(v) => Ok(Self::Var(v.square()?)),
        }
    }

    /// Enforce that `self * other == result`.
    #[tracing::instrument(target = "r1cs")]
    pub fn mul_equals(&self, other: &Self, result: &Self) -> Result<(), SynthesisError> {
        use I64Var::*;
        match (self, other, result) {
            (Constant(_), Constant(_), Constant(_)) => Ok(()),
            (Constant(_), Constant(_), _) | (Constant(_), Var(_), _) | (Var(_), Constant(_), _) => {
                result.enforce_equal(&(self * other))
            } // this multiplication should be free
            (Var(v1), Var(v2), Var(v3)) => v1.mul_equals(v2, v3),
            (Var(v1), Var(v2), Constant(f)) => {
                let cs = v1.cs.clone();
                let v3 = AllocatedI64::new_constant(cs, *f).unwrap();
                v1.mul_equals(v2, &v3)
            }
        }
    }

    /// Enforce that `self * self == result`.
    #[tracing::instrument(target = "r1cs")]
    pub fn square_equals(&self, result: &Self) -> Result<(), SynthesisError> {
        use I64Var::*;
        match (self, result) {
            (Constant(_), Constant(_)) => Ok(()),
            (Constant(f), Var(r)) => {
                let cs = r.cs.clone();
                let v = AllocatedI64::new_witness(cs, || Ok(*f))?;
                v.square_equals(&r)
            }
            (Var(v), Constant(f)) => {
                let cs = v.cs.clone();
                let r = AllocatedI64::new_witness(cs, || Ok(*f))?;
                v.square_equals(&r)
            }
            (Var(v1), Var(v2)) => v1.square_equals(v2),
        }
    }

    #[tracing::instrument(target = "r1cs")]
    pub fn div(&self, c: i64) -> Result<Self, SynthesisError> {
        match self {
            I64Var::Var(v) => Ok(I64Var::Var(v.div_constant(c))),
            I64Var::Constant(f) => Ok(I64Var::Constant(f / c)),
        }
    }

    pub fn pow_le(&self, bits: &[Boolean<F>]) -> Result<Self, SynthesisError> {
        let mut res = Self::one();
        let mut power = self.clone();
        for bit in bits {
            let tmp = res.clone() * &power;
            res = bit.select(&tmp, &res)?;
            power = power.square()?;
        }
        Ok(res)
    }
}

impl_ops!(
    I64Var<F>,
    i64,
    Add,
    add,
    AddAssign,
    add_assign,
    |this: &'a I64Var<F>, other: &'a I64Var<F>| {
        use I64Var::*;
        match (this, other) {
            (Constant(c1), Constant(c2)) => Constant(*c1 + *c2),
            (Constant(c), Var(v)) | (Var(v), Constant(c)) => Var(v.add_constant(*c)),
            (Var(v1), Var(v2)) => Var(v1.add(v2)),
        }
    },
    |this: &'a I64Var<F>, other: i64| { this + &I64Var::Constant(other) },
    F: PrimeField,
);

impl_ops!(
    I64Var<F>,
    i64,
    Sub,
    sub,
    SubAssign,
    sub_assign,
    |this: &'a I64Var<F>, other: &'a I64Var<F>| {
        use I64Var::*;
        match (this, other) {
            (Constant(c1), Constant(c2)) => Constant(*c1 - *c2),
            (Var(v), Constant(c)) => Var(v.sub_constant(*c)),
            (Constant(c), Var(v)) => Var(v.sub_constant(*c).negate()),
            (Var(v1), Var(v2)) => Var(v1.sub(v2)),
        }
    },
    |this: &'a I64Var<F>, other: i64| { this - &I64Var::Constant(other) },
    F: PrimeField
);

impl_ops!(
    I64Var<F>,
    i64,
    Mul,
    mul,
    MulAssign,
    mul_assign,
    |this: &'a I64Var<F>, other: &'a I64Var<F>| {
        use I64Var::*;
        match (this, other) {
            (Constant(c1), Constant(c2)) => Constant(*c1 * *c2),
            (Constant(c), Var(v)) | (Var(v), Constant(c)) => Var(v.mul_constant(*c)),
            (Var(v1), Var(v2)) => Var(v1.mul(v2)),
        }
    },
    |this: &'a I64Var<F>, other: i64| {
        if other == 0 {
            I64Var::zero()
        } else {
            this * &I64Var::Constant(other)
        }
    },
    F: PrimeField
);

/// *************************************************************************
/// *************************************************************************

impl<F: PrimeField> EqGadget<F> for I64Var<F> {
    #[tracing::instrument(target = "r1cs")]
    fn is_eq(&self, other: &Self) -> Result<Boolean<F>, SynthesisError> {
        match (self, other) {
            (Self::Constant(c1), Self::Constant(c2)) => Ok(Boolean::Constant(c1 == c2)),
            (Self::Constant(c), Self::Var(v)) | (Self::Var(v), Self::Constant(c)) => {
                let cs = v.cs.clone();
                let c = AllocatedI64::new_constant(cs, *c)?;
                c.is_eq(v)
            }
            (Self::Var(v1), Self::Var(v2)) => v1.is_eq(v2),
        }
    }

    #[tracing::instrument(target = "r1cs")]
    fn conditional_enforce_equal(
        &self,
        other: &Self,
        should_enforce: &Boolean<F>,
    ) -> Result<(), SynthesisError> {
        match (self, other) {
            (Self::Constant(_), Self::Constant(_)) => Ok(()),
            (Self::Constant(c), Self::Var(v)) | (Self::Var(v), Self::Constant(c)) => {
                let cs = v.cs.clone();
                let c = AllocatedI64::new_constant(cs, *c)?;
                c.conditional_enforce_equal(v, should_enforce)
            }
            (Self::Var(v1), Self::Var(v2)) => v1.conditional_enforce_equal(v2, should_enforce),
        }
    }

    #[tracing::instrument(target = "r1cs")]
    fn conditional_enforce_not_equal(
        &self,
        other: &Self,
        should_enforce: &Boolean<F>,
    ) -> Result<(), SynthesisError> {
        match (self, other) {
            (Self::Constant(_), Self::Constant(_)) => Ok(()),
            (Self::Constant(c), Self::Var(v)) | (Self::Var(v), Self::Constant(c)) => {
                let cs = v.cs.clone();
                let c = AllocatedI64::new_constant(cs, *c)?;
                c.conditional_enforce_not_equal(v, should_enforce)
            }
            (Self::Var(v1), Self::Var(v2)) => v1.conditional_enforce_not_equal(v2, should_enforce),
        }
    }
}

impl<F: PrimeField> CondSelectGadget<F> for I64Var<F> {
    #[tracing::instrument(target = "r1cs")]
    fn conditionally_select(
        cond: &Boolean<F>,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, SynthesisError> {
        match cond {
            &Boolean::Constant(true) => Ok(true_value.clone()),
            &Boolean::Constant(false) => Ok(false_value.clone()),
            _ => {
                match (true_value, false_value) {
                    (Self::Constant(t), Self::Constant(f)) => {
                        let is = AllocatedI64::from(cond.clone());
                        let not = AllocatedI64::from(!cond);
                        // cond * t + (1 - cond) * f
                        Ok(is.mul_constant(*t).add(&not.mul_constant(*f)).into())
                    }
                    (..) => {
                        let cs = cond.cs();
                        let true_value = match true_value {
                            Self::Constant(f) => AllocatedI64::new_constant(cs.clone(), *f)?,
                            Self::Var(v) => v.clone(),
                        };
                        let false_value = match false_value {
                            Self::Constant(f) => AllocatedI64::new_constant(cs, *f)?,
                            Self::Var(v) => v.clone(),
                        };
                        cond.select(&true_value, &false_value).map(Self::Var)
                    }
                }
            }
        }
    }
}

impl<F: PrimeField, S: AsPrimitive<i64>> AllocVar<S, F> for I64Var<F> {
    fn new_variable<T: Borrow<S>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        if mode == AllocationMode::Constant {
            let v = *f()?.borrow();
            Ok(Self::Constant(v.as_()))
        } else {
            AllocatedI64::new_variable(cs, f, mode).map(Self::Var)
        }
    }
}
