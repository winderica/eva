use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_groth16::ProvingKey;
use ark_serialize::{CanonicalSerialize, Compress};
use folding_schemes::{commitment::pedersen::Params, MSM};

pub fn srs_size<E: Pairing>(pk: &ProvingKey<E>, ck: &Params<E::G1>) -> usize
where
    <E::G1 as CurveGroup>::Config: MSM<E::G1>,
{
    pk.common.beta_g1.serialized_size(Compress::Yes)
        + pk.common.delta_g1.serialized_size(Compress::Yes)
        + pk.common.eta_delta_inv_g1.serialized_size(Compress::Yes)
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.common.a_query.len()
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.common.b_g1_query.len()
        + E::G2Affine::zero().serialized_size(Compress::Yes) * pk.common.b_g2_query.len()
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.common.h_query.len()
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.common.l_query.len()
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.common.link_ek.p.len()
        + pk.vk.alpha_g1.serialized_size(Compress::Yes)
        + pk.vk.beta_g2.serialized_size(Compress::Yes)
        + pk.vk.gamma_g2.serialized_size(Compress::Yes)
        + pk.vk.delta_g2.serialized_size(Compress::Yes)
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.vk.gamma_abc_g1.0.len()
        + E::G1Affine::zero().serialized_size(Compress::Yes) * pk.vk.gamma_abc_g1.1.len()
        + pk.vk.eta_gamma_inv_g1.serialized_size(Compress::Yes)
        + pk.vk.link_pp.serialized_size(Compress::Yes)
        + E::G2Affine::zero().serialized_size(Compress::Yes) * pk.vk.link_vk.c.len()
        + pk.vk.link_vk.a.serialized_size(Compress::Yes)
        + E::G1Affine::zero().serialized_size(Compress::Yes) * ck.generators.len()
        + ck.h.serialized_size(Compress::Yes)
}
