use std::sync::Arc;

use ark_ff::PrimeField;

use super::params::GriffinParams;

#[derive(Clone, Debug)]
pub struct Griffin<S: PrimeField> {
    pub(crate) params: Arc<GriffinParams<S>>,
}

impl<S: PrimeField> Griffin<S> {
    pub fn new(params: &Arc<GriffinParams<S>>) -> Self {
        Griffin {
            params: Arc::clone(params),
        }
    }

    fn affine_3(&self, input: &mut [S], round: usize) {
        // multiplication by circ(2 1 1) is equal to state + sum(state)
        let mut sum = input[0];
        input.iter().skip(1).for_each(|el| sum.add_assign(el));

        if round < self.params.rounds - 1 {
            for (el, rc) in input
                .iter_mut()
                .zip(self.params.round_constants[round].iter())
            {
                el.add_assign(&sum);
                el.add_assign(rc); // add round constant
            }
        } else {
            // no round constant
            for el in input.iter_mut() {
                el.add_assign(&sum);
            }
        }
    }

    fn affine_4(&self, input: &mut [S], round: usize) {
        let mut t_0 = input[0];
        t_0.add_assign(&input[1]);
        let mut t_1 = input[2];
        t_1.add_assign(&input[3]);
        let mut t_2 = input[1];
        t_2.double_in_place();
        t_2.add_assign(&t_1);
        let mut t_3 = input[3];
        t_3.double_in_place();
        t_3.add_assign(&t_0);
        let mut t_4 = t_1;
        t_4.double_in_place();
        t_4.double_in_place();
        t_4.add_assign(&t_3);
        let mut t_5 = t_0;
        t_5.double_in_place();
        t_5.double_in_place();
        t_5.add_assign(&t_2);
        let mut t_6 = t_3;
        t_6.add_assign(&t_5);
        let mut t_7 = t_2;
        t_7.add_assign(&t_4);
        input[0] = t_6;
        input[1] = t_5;
        input[2] = t_7;
        input[3] = t_4;

        if round < self.params.rounds - 1 {
            for (i, rc) in input
                .iter_mut()
                .zip(self.params.round_constants[round].iter())
            {
                i.add_assign(rc);
            }
        }
    }

    fn affine(&self, input: &mut [S], round: usize) {
        if self.params.t == 3 {
            self.affine_3(input, round);
            return;
        }
        if self.params.t == 4 {
            self.affine_4(input, round);
            return;
        }

        // first matrix
        let t4 = self.params.t / 4;
        for i in 0..t4 {
            let start_index = i * 4;
            let mut t_0 = input[start_index];
            t_0.add_assign(&input[start_index + 1]);
            let mut t_1 = input[start_index + 2];
            t_1.add_assign(&input[start_index + 3]);
            let mut t_2 = input[start_index + 1];
            t_2.double_in_place();
            t_2.add_assign(&t_1);
            let mut t_3 = input[start_index + 3];
            t_3.double_in_place();
            t_3.add_assign(&t_0);
            let mut t_4: S = t_1;
            t_4.double_in_place();
            t_4.double_in_place();
            t_4.add_assign(&t_3);
            let mut t_5 = t_0;
            t_5.double_in_place();
            t_5.double_in_place();
            t_5.add_assign(&t_2);
            input[start_index] = t_3 + t_5;
            input[start_index + 1] = t_5;
            input[start_index + 2] = t_2 + t_4;
            input[start_index + 3] = t_4;
        }

        // second matrix
        let mut stored = [S::zero(); 4];
        for l in 0..4 {
            stored[l] = input[l];
            for j in 1..t4 {
                stored[l].add_assign(&input[4 * j + l]);
            }
        }

        for i in 0..input.len() {
            input[i].add_assign(&stored[i % 4]);
            if round < self.params.rounds - 1 {
                input[i].add_assign(&self.params.round_constants[round][i]); // add round constant
            }
        }
    }

    fn non_linear(&self, input: &mut [S]) {
        // first two state words
        input[0] = {
            let mut res = S::one();
            for &i in &self.params.d_inv {
                res.square_in_place();
                if i {
                    res *= input[0];
                }
            }
            res
        };

        let mut state = input[1];

        input[1].square_in_place();
        match self.params.d {
            3 => {}
            5 => {
                input[1].square_in_place();
            }
            _ => panic!(),
        }
        input[1].mul_assign(&state);

        let mut y01_i = input[1];
        // rest of the state
        for i in 2..input.len() {
            y01_i += input[0];
            let l = if i == 2 { y01_i } else { y01_i + state };
            let ab = &self.params.alpha_beta[i - 2];
            state = input[i];
            input[i] *= l.square() + l * ab[0] + ab[1];
        }
    }
}

pub trait Permutation<S: PrimeField> {
    fn permute(&self, input: &mut [S]);
    fn hash(&self, input: &[S]) -> S;
}

impl<S: PrimeField> Permutation<S> for Griffin<S> {
    fn permute(&self, input: &mut [S]) {
        self.affine(input, self.params.rounds); // no RC

        for r in 0..self.params.rounds {
            self.non_linear(input);
            self.affine(input, r);
        }
    }

    fn hash(&self, message: &[S]) -> S {
        let mut state = vec![S::zero(); self.params.t];
        for chunk in message.chunks(self.params.rate) {
            for i in 0..chunk.len() {
                state[i] += &chunk[i];
            }
            self.permute(&mut state)
        }
        state[0]
    }
}

#[cfg(test)]
mod griffin_tests_bn256 {
    use super::*;
    use ark_bn254::Fr as Scalar;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    static TESTRUNS: usize = 5;

    #[test]
    fn consistent_perm() {
        let rng = &mut thread_rng();
        let griffin = Griffin::new(&Arc::new(GriffinParams::new(3, 5, 12)));
        let t = griffin.params.t;
        for _ in 0..TESTRUNS {
            let input1: Vec<Scalar> = (0..t).map(|_| Scalar::rand(rng)).collect();

            let mut input2: Vec<Scalar>;
            loop {
                input2 = (0..t).map(|_| Scalar::rand(rng)).collect();
                if input1 != input2 {
                    break;
                }
            }

            let mut perm1 = input1.clone();
            let mut perm2 = input1.clone();
            let mut perm3 = input2.clone();
            griffin.permute(&mut perm1);
            griffin.permute(&mut perm2);
            griffin.permute(&mut perm3);
            assert_eq!(perm1, perm2);
            assert_ne!(perm1, perm3);
        }
    }
}

#[cfg(test)]
mod griffin_affine_tests_bn256 {
    use super::*;
    use ark_bn254::Fr as Scalar;
    use ark_ff::{UniformRand, Zero};
    use rand::thread_rng;

    static TESTRUNS: usize = 5;

    fn matmul(input: &[Scalar], mat: &[Vec<Scalar>]) -> Vec<Scalar> {
        let t = mat.len();
        debug_assert!(t == input.len());
        let mut out = vec![Scalar::zero(); t];
        for row in 0..t {
            for (col, inp) in input.iter().enumerate() {
                let mut tmp = mat[row][col];
                tmp *= inp;
                out[row] += &tmp;
            }
        }
        out
    }

    fn affine_test(t: usize) {
        let rng = &mut thread_rng();
        let griffin_param = Arc::new(GriffinParams::<Scalar>::new(t, 5, 1));
        let griffin = Griffin::<Scalar>::new(&griffin_param);

        let mat = &griffin_param.mat;

        for _ in 0..TESTRUNS {
            let input: Vec<Scalar> = (0..t).map(|_| Scalar::rand(rng)).collect();

            // affine 1
            let output1 = matmul(&input, mat);
            let mut output2 = input.to_owned();
            griffin.affine(&mut output2, 1);
            assert_eq!(output1, output2);
        }
    }

    #[test]
    fn affine_3() {
        affine_test(3);
    }

    #[test]
    fn affine_4() {
        affine_test(4);
    }

    #[test]
    fn affine_8() {
        affine_test(8);
    }

    #[test]
    fn affine_60() {
        affine_test(60);
    }
}
