use ark_crypto_primitives::sponge::Absorb;
use ark_ec::{CurveGroup};
use ark_std::{One, Zero};

use super::{CurrentInstance, CycleFoldCommittedInstance, RunningInstance, Witness};
use crate::ccs::r1cs::R1CS;
use crate::{Error, MSM};

/// NovaR1CS extends R1CS methods with Nova specific methods
pub trait NovaR1CS<C: CurveGroup> {
    /// returns a dummy instance (Witness and CommittedInstance) for the current R1CS structure
    fn dummy_running_instance(&self) -> (Witness<C>, RunningInstance<C>);
    fn dummy_current_instance(&self) -> (Witness<C>, CurrentInstance<C>);
    fn dummy_cyclefold_instance(&self) -> (Witness<C>, CycleFoldCommittedInstance<C>);

    /// checks the R1CS relation (un-relaxed) for the given Witness and CommittedInstance.
    fn check_running_instance_relation(
        &self,
        W: &Witness<C>,
        U: &RunningInstance<C>,
    ) -> Result<(), Error>;

    /// checks the R1CS relation (un-relaxed) for the given Witness and CommittedInstance.
    fn check_current_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CurrentInstance<C>,
    ) -> Result<(), Error>;

    /// checks the R1CS relation (un-relaxed) for the given Witness and CommittedInstance.
    fn check_cyclefold_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CycleFoldCommittedInstance<C>,
    ) -> Result<(), Error>;

    /// checks the Relaxed R1CS relation (corresponding to the current R1CS) for the given Witness
    /// and CommittedInstance.
    fn check_relaxed_running_instance_relation(
        &self,
        W: &Witness<C>,
        U: &RunningInstance<C>,
    ) -> Result<(), Error>;

    /// checks the Relaxed R1CS relation (corresponding to the current R1CS) for the given Witness
    /// and CommittedInstance.
    fn check_relaxed_current_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CurrentInstance<C>,
    ) -> Result<(), Error>;

    /// checks the Relaxed R1CS relation (corresponding to the current R1CS) for the given Witness
    /// and CommittedInstance.
    fn check_relaxed_cyclefold_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CycleFoldCommittedInstance<C>,
    ) -> Result<(), Error>;
}

impl<C: CurveGroup> NovaR1CS<C> for R1CS<C::ScalarField>
where
    C::Config: MSM<C>,
    C::ScalarField: Absorb,
    <C as ark_ec::CurveGroup>::BaseField: ark_ff::PrimeField,
{
    fn dummy_running_instance(&self) -> (Witness<C>, RunningInstance<C>) {
        let w_len = self.A.n_cols - 1 - self.l;
        let w_dummy = Witness::<C>::new(vec![C::ScalarField::zero(); w_len], self);
        let u_dummy = RunningInstance::<C>::dummy(self.l);
        (w_dummy, u_dummy)
    }

    fn dummy_current_instance(&self) -> (Witness<C>, CurrentInstance<C>) {
        let w_len = self.A.n_cols - 1 - self.l;
        let w_dummy = Witness::<C>::new(vec![C::ScalarField::zero(); w_len], self);
        let u_dummy = CurrentInstance::<C>::dummy(self.l);
        (w_dummy, u_dummy)
    }

    fn dummy_cyclefold_instance(&self) -> (Witness<C>, CycleFoldCommittedInstance<C>) {
        let w_len = self.A.n_cols - 1 - self.l;
        let w_dummy = Witness::<C>::new(vec![C::ScalarField::zero(); w_len], self);
        let u_dummy = CycleFoldCommittedInstance::<C>::dummy(self.l);
        (w_dummy, u_dummy)
    }

    fn check_running_instance_relation(
        &self,
        W: &Witness<C>,
        U: &RunningInstance<C>,
    ) -> Result<(), Error> {
        if U.cmE != C::zero() || U.u != C::ScalarField::one() {
            return Err(Error::R1CSUnrelaxedFail);
        }

        let Z: Vec<C::ScalarField> = [vec![U.u], U.x.to_vec(), W.QW.to_vec()].concat();
        self.check_relation(&Z)
    }

    fn check_current_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CurrentInstance<C>,
    ) -> Result<(), Error> {
        if U.cmE != C::zero() || U.u != C::ScalarField::one() {
            return Err(Error::R1CSUnrelaxedFail);
        }

        let Z: Vec<C::ScalarField> = [vec![U.u], U.x.to_vec(), W.QW.to_vec()].concat();
        self.check_relation(&Z)
    }

    fn check_cyclefold_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CycleFoldCommittedInstance<C>,
    ) -> Result<(), Error> {
        if U.cmE != C::zero() || U.u != C::ScalarField::one() {
            return Err(Error::R1CSUnrelaxedFail);
        }

        let Z: Vec<C::ScalarField> = [vec![U.u], U.x.to_vec(), W.QW.to_vec()].concat();
        self.check_relation(&Z)
    }

    fn check_relaxed_running_instance_relation(
        &self,
        W: &Witness<C>,
        U: &RunningInstance<C>,
    ) -> Result<(), Error> {
        let mut rel_r1cs = self.clone().relax();
        rel_r1cs.u = U.u;
        rel_r1cs.E = W.E.clone();

        let Z: Vec<C::ScalarField> = [vec![U.u], U.x.to_vec(), W.QW.to_vec()].concat();
        rel_r1cs.check_relation(&Z)
    }

    fn check_relaxed_current_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CurrentInstance<C>,
    ) -> Result<(), Error> {
        let mut rel_r1cs = self.clone().relax();
        rel_r1cs.u = U.u;
        rel_r1cs.E = W.E.clone();

        let Z: Vec<C::ScalarField> = [vec![U.u], U.x.to_vec(), W.QW.to_vec()].concat();
        rel_r1cs.check_relation(&Z)
    }

    fn check_relaxed_cyclefold_instance_relation(
        &self,
        W: &Witness<C>,
        U: &CycleFoldCommittedInstance<C>,
    ) -> Result<(), Error> {
        let mut rel_r1cs = self.clone().relax();
        rel_r1cs.u = U.u;
        rel_r1cs.E = W.E.clone();

        let Z: Vec<C::ScalarField> = [vec![U.u], U.x.to_vec(), W.QW.to_vec()].concat();
        rel_r1cs.check_relation(&Z)
    }
}
