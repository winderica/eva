use std::{path::Path, sync::Arc};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer, Zero};
use rayon::prelude::*;
use video::{
    edit::constraints::{EditConfig, RemovingCfg},
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
    // let edit_configs = vec![(); predictions.len()];
    // let edit_configs = vec![BrightnessCfg(416); predictions.len()];
    // let edit_configs = {
    //     const W: usize = 352;
    //     const H: usize = 288;
    //     let x = 96;
    //     let y = 80;
    //     let w = 176;
    //     let h = 144;
    //     (0..predictions.len())
    //         .into_par_iter()
    //         .map(|v| {
    //             let tt = v % (W / 16 * H / 16);
    //             let xx = tt % (W / 16) * 16;
    //             let yy = tt / (W / 16) * 16;
    //             MaskCfg(
    //                 Array2::from_shape_fn((16, 16), |(m, n)| {
    //                     let xx = xx + m;
    //                     let yy = yy + n;
    //                     (
    //                         (((xx - x) / 16 % 2 + (yy - y) / 16 % 2 + 3) * 32) as u8,
    //                         !(xx >= x && xx < x + w && yy >= y && yy < y + h),
    //                     )
    //                 }),
    //                 Array2::from_shape_fn((8, 8), |(m, n)| {
    //                     let xx = xx + m * 2;
    //                     let yy = yy + n * 2;
    //                     (128, !(xx >= x && xx < x + w && yy >= y && yy < y + h))
    //                 }),
    //                 Array2::from_shape_fn((8, 8), |(m, n)| {
    //                     let xx = xx + m * 2;
    //                     let yy = yy + n * 2;
    //                     (128, !(xx >= x && xx < x + w && yy >= y && yy < y + h))
    //                 }),
    //             )
    //         })
    //         .collect::<Vec<_>>()
    // };
    let edit_configs = {
        const W: usize = 352;
        const H: usize = 288;
        const WW: usize = 160;
        const HH: usize = 128;

        let x = 48;
        let y = 80;

        const N: usize = 256;
        (0..W * H * N)
            .into_par_iter()
            .map(|v| {
                let tt = v % (W / 16 * H / 16);
                let xx = tt % (W / 16) * 16;
                let yy = tt / (W / 16) * 16;
                RemovingCfg(xx >= x && xx < (x + WW) && yy >= y && yy < (y + HH))
            })
            .collect::<Vec<_>>()
    };
    let num_steps = predictions.len() / blocks_per_step;

    let griffin = Griffin::new(&Arc::new(GriffinParams::new(16, 5, 9)));
    let mut h = Fr::zero();

    let timer = start_timer!(|| "Hash prediction macroblocks and quantized coefficients");
    for i in 0..num_steps {
        if edit_configs[i * blocks_per_step..(i + 1) * blocks_per_step]
            .iter()
            .any(|cfg| cfg.should_keep_native())
        {
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
    }
    end_timer!(timer);
    println!("{}", h);

    Ok(())
}
