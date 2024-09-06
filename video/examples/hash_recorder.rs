use std::{path::Path, sync::Arc};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer, Zero};
use rayon::prelude::*;
use video::{
    griffin::{
        griffin::{Griffin, Permutation},
        params::GriffinParams,
    },
    parse_recorder_data, MB_BITS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let blocks = parse_recorder_data(Path::new(env!("DATA_PATH")).join("bunny"))?;
    let blocks_per_step = 256;
    let num_steps = blocks.len() / blocks_per_step;

    let griffin = Griffin::new(&Arc::new(GriffinParams::new(16, 5, 9)));
    let mut h = Fr::zero();

    let timer = start_timer!(|| "Hash original macroblocks");
    for i in 0..num_steps {
        h = griffin.hash(
            &(i * blocks_per_step..(i + 1) * blocks_per_step)
                .into_par_iter()
                .map(|j| {
                    griffin.hash(
                        &blocks[j]
                            .chunks(Fr::MODULUS_BIT_SIZE as usize / MB_BITS)
                            .map(Fr::from_be_bytes_mod_order)
                            .collect::<Vec<_>>(),
                    )
                })
                .chain(vec![h])
                .collect::<Vec<_>>(),
        );
    }
    end_timer!(timer);
    println!("{}", h);

    Ok(())
}
