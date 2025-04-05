pub mod constraints;

use crate::encode::Matrix;

impl<const M: usize, const N: usize> Matrix<u8, M, N> {
    pub fn invert_color(&self) -> Matrix<u8, M, N> {
        let mut inverted = self.clone();
        for i in 0..M {
            for j in 0..N {
                inverted[(i, j)] = 255 - inverted[(i, j)];
            }
        }
        inverted
    }
}

#[cfg(test)]
pub mod tests {
    use std::{cmp::min, error::Error, fs, path::Path};

    const W: usize = 352;
    const H: usize = 288;

    #[test]
    fn test_invert() -> Result<(), Box<dyn Error>> {
        let path = Path::new(env!("DATA_PATH"));
        let input = fs::read(path.join("foreman_cif.yuv"))?;

        let output = input.iter().map(|i| 255 - i).collect::<Vec<_>>();

        fs::write(path.join("foreman_cif_inv.yuv"), output)?;

        Ok(())
    }

    #[test]
    fn test_gray() -> Result<(), Box<dyn Error>> {
        let path = Path::new(env!("DATA_PATH"));
        let input = fs::read(path.join("foreman_cif.yuv"))?;

        let output = input
            .chunks_exact(W * H * 3 / 2)
            .flat_map(|i| [&i[..W * H], &[128; W * H / 2]].concat())
            .collect::<Vec<_>>();

        fs::write(path.join("foreman_cif_gray.yuv"), output)?;

        Ok(())
    }

    #[test]
    fn test_bright() -> Result<(), Box<dyn Error>> {
        let a = 13;
        let b = 8;

        let path = Path::new(env!("DATA_PATH"));
        let input = fs::read(path.join("foreman_cif.yuv"))?;

        let output = input
            .chunks_exact(W * H * 3 / 2)
            .flat_map(|i| {
                [
                    &i[..W * H]
                        .iter()
                        .map(|&i| min(i as u16 * a / b, 255) as u8)
                        .collect::<Vec<_>>(),
                    &i[W * H..],
                ]
                .concat()
            })
            .collect::<Vec<_>>();

        fs::write(path.join("foreman_cif_bright.yuv"), output)?;

        Ok(())
    }

    #[test]
    fn test_crop() -> Result<(), Box<dyn Error>> {
        let x = 48;
        let y = 80;
        let w = 160;
        let h = 128;

        let path = Path::new(env!("DATA_PATH"));
        let input = fs::read(path.join("foreman_cif.yuv"))?;

        let output = input
            .chunks_exact(W * H * 3 / 2)
            .flat_map(|i| {
                [
                    i[..W * H]
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % W >= x && i % W < x + w && i / W >= y && i / W < y + h)
                        .map(|(_, &v)| v)
                        .collect::<Vec<_>>(),
                    i[W * H..W * H * 5 / 4]
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| {
                            i % (W / 2) >= x / 2
                                && i % (W / 2) < x / 2 + w / 2
                                && i / (W / 2) >= y / 2
                                && i / (W / 2) < y / 2 + h / 2
                        })
                        .map(|(_, &v)| v)
                        .collect::<Vec<_>>(),
                    i[W * H * 5 / 4..]
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| {
                            i % (W / 2) >= x / 2
                                && i % (W / 2) < x / 2 + w / 2
                                && i / (W / 2) >= y / 2
                                && i / (W / 2) < y / 2 + h / 2
                        })
                        .map(|(_, &v)| v)
                        .collect::<Vec<_>>(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();

        fs::write(path.join("foreman_cif_crop.yuv"), output)?;

        Ok(())
    }

    #[test]
    fn test_cut() -> Result<(), Box<dyn Error>> {
        let s = 6;
        let e = 134;

        let path = Path::new(env!("DATA_PATH"));
        let input = fs::read(path.join("foreman_cif.yuv"))?;

        let output = input
            .chunks_exact(W * H * 3 / 2)
            .enumerate()
            .filter(|(i, _)| i >= &s && i < &e)
            .flat_map(|(_, i)| i.to_vec())
            .collect::<Vec<_>>();

        fs::write(path.join("foreman_cif_cut.yuv"), output)?;

        Ok(())
    }

    #[test]
    fn test_mask() -> Result<(), Box<dyn Error>> {
        let x = 96;
        let y = 80;
        let w = 176;
        let h = 144;

        let path = Path::new(env!("DATA_PATH"));
        let input = fs::read(path.join("foreman_cif.yuv"))?;

        let output = input
            .chunks_exact(W * H * 3 / 2)
            .flat_map(|i| {
                [
                    i[..W * H]
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            if i % W >= x && i % W < x + w && i / W >= y && i / W < y + h {
                                (((i % W - x) / 16 % 2 + (i / W - y) / 16 % 2 + 3) * 32) as u8
                            } else {
                                v
                            }
                        })
                        .collect::<Vec<_>>(),
                    i[W * H..W * H * 5 / 4]
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            if i % (W / 2) >= x / 2
                                && i % (W / 2) < x / 2 + w / 2
                                && i / (W / 2) >= y / 2
                                && i / (W / 2) < y / 2 + h / 2
                            {
                                128
                            } else {
                                v
                            }
                        })
                        .collect::<Vec<_>>(),
                    i[W * H * 5 / 4..]
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            if i % (W / 2) >= x / 2
                                && i % (W / 2) < x / 2 + w / 2
                                && i / (W / 2) >= y / 2
                                && i / (W / 2) < y / 2 + h / 2
                            {
                                128
                            } else {
                                v
                            }
                        })
                        .collect::<Vec<_>>(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();

        fs::write(path.join("foreman_cif_mask.yuv"), output)?;

        Ok(())
    }
}
