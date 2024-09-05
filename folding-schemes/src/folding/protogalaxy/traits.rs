use ark_crypto_primitives::sponge::Absorb;
use ark_ec::CurveGroup;

use super::CommittedInstance;
use crate::transcript::{poseidon::PoseidonTranscript, Transcript};
use crate::Error;

/// ProtoGalaxyTranscript extends [`Transcript`] with the method to absorb ProtoGalaxy's
/// CommittedInstance.
pub trait ProtoGalaxyTranscript<C: CurveGroup>: Transcript<C> {
    fn absorb_committed_instance(&mut self, ci: &CommittedInstance<C>) -> Result<(), Error> {
        self.absorb_point(&ci.phi)?;
        self.absorb_vec(&ci.betas);
        self.absorb(&ci.e);
        Ok(())
    }
}

// Implements ProtoGalaxyTranscript for PoseidonTranscript
impl<C: CurveGroup> ProtoGalaxyTranscript<C> for PoseidonTranscript<C> where C::ScalarField: Absorb {}
