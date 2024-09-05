use std::{path::Path, sync::Arc};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer, Zero};
use ndarray::Array2;
use rayon::prelude::*;
use video::{
    edit::constraints::{BrightnessCfg, EditConfig, MaskCfg, RemovingCfg},
    encode::MacroblockType,
    griffin::{
        griffin::{Griffin, Permutation},
        params::GriffinParams,
    },
    parse_verifier_data, COEFF_BITS, MB_BITS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (predictions, outputs, encode_configs) =
        parse_verifier_data(Path::new(env!("DATA_PATH")).join("foreman_crop"))?;
    let blocks_per_step = 256;
    const W: usize = 352;
    const H: usize = 288;
    const WW: usize = 160;
    const HH: usize = 128;

    let x = 48;
    let y = 80;

    const N: usize = 256;

    let mut j = 0;
    let mut p = vec![];
    let mut o = vec![];
    let mut e1 = vec![];
    let mut e2 = vec![];
    for i in 0..W / 16 * H / 16 * N / blocks_per_step {
        let should_keeps = (i * blocks_per_step..(i + 1) * blocks_per_step).map(|v| {
            let tt = v % (W / 16 * H / 16);
            let xx = tt % (W / 16) * 16;
            let yy = tt / (W / 16) * 16;
            xx >= x && xx < (x + WW) && yy >= y && yy < (y + HH)
        }).collect::<Vec<_>>();
        if should_keeps.contains(&true) {
            for should_keep in &should_keeps {
                if *should_keep {
                    p.push(predictions[j].clone());
                    o.push(outputs[j].clone());
                    e1.push(encode_configs[j].clone());
                    j += 1;
                } else {
                    p.push(Default::default());
                    o.push(Default::default());
                    e1.push(Default::default());
                }
            }
            e2.extend(should_keeps.into_iter().map(RemovingCfg));
        }
    }

    let predictions = p;
    let outputs = o;
    let encode_configs = e1;
    let edit_configs = e2;

    let num_steps = predictions.len() / blocks_per_step;

    let griffin = Griffin::new(&Arc::new(GriffinParams::new(16, 5, 9)));
    let mut h = Fr::zero();

    let timer = start_timer!(|| "Hash prediction macroblocks and quantized coefficients");
    for i in 0..num_steps {
        h = griffin.hash(
            &(i * blocks_per_step..(i + 1) * blocks_per_step)
                .into_par_iter()
                .map(|j| {
                    let (y_pred, u_pred, v_pred) = &predictions[j];
                    let (y_output, u_output, v_output) = &outputs[j];
                    let encode_config = &encode_configs[j];
                    let edit_config = &edit_configs[j];

                    if !edit_config.should_keep_native() {
                        return Fr::zero();
                    }

                    griffin.hash(
                        &[&y_pred[..], u_pred, v_pred]
                            .concat()
                            .chunks(Fr::MODULUS_BIT_SIZE as usize / MB_BITS)
                            .chain(
                                [&y_output[..], u_output, v_output]
                                    .concat()
                                    .chunks(Fr::MODULUS_BIT_SIZE as usize / COEFF_BITS),
                            )
                            .map(Fr::from_be_bytes_mod_order)
                            .chain({
                                let mut v = (encode_config.qpc % 6) as u64;
                                v = v * 8 + (encode_config.qp % 6) as u64;
                                v = v * 16 + (encode_config.qpc / 6) as u64;
                                v = v * 16 + (encode_config.qp / 6) as u64;
                                v = v * 2
                                    + (encode_config.mb_type == MacroblockType::I16x16) as u64;
                                v = v * 2 + encode_config.slice_is_intra as u64;
                                vec![Fr::from(v)]
                            })
                            .chain(edit_config.compactify())
                            .collect::<Vec<_>>(),
                    )
                })
                .chain(vec![h])
                .collect::<Vec<_>>(),
        )
    }
    end_timer!(timer);
    println!("{}", h);

    Ok(())
}
