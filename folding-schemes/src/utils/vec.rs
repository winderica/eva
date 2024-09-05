use std::sync::Arc;

use ark_ff::PrimeField;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain,
};
pub use ark_relations::r1cs::Matrix as R1CSMatrix;
use ark_std::cfg_iter;
use rayon::prelude::*;

use crate::{Error, MVM};

#[derive(Clone)]
pub struct SparseMatrix<F: PrimeField> {
    pub n_rows: usize,
    pub n_cols: usize,
    /// coeffs = R1CSMatrix = Vec<Vec<(F, usize)>>, which contains each row and the F is the value
    /// of the coefficient and the usize indicates the column position
    pub coeffs: R1CSMatrix<F>,
    pub cuda: Arc<CudaSparseMatrix<F>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct CSRSparseMatrix<F> {
    pub data: Vec<F>,
    pub col_idx: Vec<i32>,
    pub row_ptr: Vec<i32>,
}

pub struct CudaSparseMatrix<F> {
    pub data: icicle_cuda_runtime::memory::DeviceVec<F>,
    pub row_ptr: icicle_cuda_runtime::memory::DeviceVec<i32>,
    pub col_idx: icicle_cuda_runtime::memory::DeviceVec<i32>,
}

impl<F: PrimeField> SparseMatrix<F> {
    pub fn new(n_rows: usize, n_cols: usize, coeffs: Vec<Vec<(F, usize)>>) -> Self where F: MVM {
        let csr = Self::to_csr(n_rows, n_cols, &coeffs);
        SparseMatrix {
            cuda: Arc::new(MVM::prepare_matrix(&csr)),
            n_rows,
            n_cols,
            coeffs,
        }
    }

    pub fn to_dense(&self) -> Vec<Vec<F>> {
        let mut r: Vec<Vec<F>> = vec![vec![F::zero(); self.n_cols]; self.n_rows];
        for (row_i, row) in self.coeffs.iter().enumerate() {
            for &(value, col_i) in row.iter() {
                r[row_i][col_i] = value;
            }
        }
        r
    }

    pub fn to_csr(n_rows: usize, n_cols: usize, coeffs: &[Vec<(F, usize)>]) -> CSRSparseMatrix<F> {
        let mut data = coeffs
            .iter()
            .flat_map(|row| row.iter().map(|(v, _)| *v))
            .collect::<Vec<_>>();
        let mut indices = coeffs
            .iter()
            .flat_map(|row| row.iter().map(|(_, i)| *i as i32))
            .collect::<Vec<_>>();
        let mut indptr = vec![0];
        let mut i = 0;
        for row in coeffs.iter() {
            i += row.len();
            indptr.push(i as i32);
        }
        data.shrink_to_fit();
        indices.shrink_to_fit();
        indptr.shrink_to_fit();

        CSRSparseMatrix {
            data,
            col_idx: indices,
            row_ptr: indptr,
        }
    }
}

pub fn dense_matrix_to_sparse<F: MVM>(m: Vec<Vec<F>>) -> SparseMatrix<F> {
    let mut coeffs = vec![];
    for m_row in m.iter() {
        let mut row: Vec<(F, usize)> = Vec::new();
        for (col_i, value) in m_row.iter().enumerate() {
            if !value.is_zero() {
                row.push((*value, col_i));
            }
        }
        coeffs.push(row);
    }
    SparseMatrix::new(m.len(), m[0].len(), coeffs)
}

pub fn vec_add<F: PrimeField>(a: &[F], b: &[F]) -> Result<Vec<F>, Error> {
    if a.len() != b.len() {
        return Err(Error::NotSameLength(
            "a.len()".to_string(),
            a.len(),
            "b.len()".to_string(),
            b.len(),
        ));
    }
    Ok(cfg_iter!(a).zip(b).map(|(x, y)| *x + y).collect())
}

pub fn vec_sub<F: PrimeField>(a: &[F], b: &[F]) -> Result<Vec<F>, Error> {
    if a.len() != b.len() {
        return Err(Error::NotSameLength(
            "a.len()".to_string(),
            a.len(),
            "b.len()".to_string(),
            b.len(),
        ));
    }
    Ok(cfg_iter!(a).zip(b).map(|(x, y)| *x - y).collect())
}

pub fn vec_scalar_mul<F: PrimeField>(vec: &[F], c: &F) -> Vec<F> {
    cfg_iter!(vec).map(|a| *a * c).collect()
}

pub fn is_zero_vec<F: PrimeField>(vec: &[F]) -> bool {
    cfg_iter!(vec).all(|a| a.is_zero())
}

pub fn mat_vec_mul<F: PrimeField>(M: &Vec<Vec<F>>, z: &[F]) -> Result<Vec<F>, Error> {
    if M.is_empty() {
        return Err(Error::Empty);
    }
    if M[0].len() != z.len() {
        return Err(Error::NotSameLength(
            "M[0].len()".to_string(),
            M[0].len(),
            "z.len()".to_string(),
            z.len(),
        ));
    }

    let mut r: Vec<F> = vec![F::zero(); M.len()];
    for (i, M_i) in M.iter().enumerate() {
        for (j, M_ij) in M_i.iter().enumerate() {
            r[i] += *M_ij * z[j];
        }
    }
    Ok(r)
}

pub fn mat_vec_mul_sparse<F: PrimeField>(M: &SparseMatrix<F>, z: &[F]) -> Result<Vec<F>, Error> {
    if M.n_cols != z.len() {
        return Err(Error::NotSameLength(
            "M.n_cols".to_string(),
            M.n_cols,
            "z.len()".to_string(),
            z.len(),
        ));
    }

    Ok(cfg_iter!(M.coeffs)
        .map(|row| {
            row.iter()
                .map(|&(value, col_i)| value * z[col_i])
                .sum::<F>()
        })
        .collect())
}

pub fn hadamard<F: PrimeField>(a: &[F], b: &[F]) -> Result<Vec<F>, Error> {
    if a.len() != b.len() {
        return Err(Error::NotSameLength(
            "a.len()".to_string(),
            a.len(),
            "b.len()".to_string(),
            b.len(),
        ));
    }
    Ok(cfg_iter!(a).zip(b).map(|(a, b)| *a * b).collect())
}

/// returns the interpolated polynomial of degree=v.len().next_power_of_two(), which passes through all
/// the given elements of v.
pub fn poly_from_vec<F: PrimeField>(v: Vec<F>) -> Result<DensePolynomial<F>, Error> {
    let D = GeneralEvaluationDomain::<F>::new(v.len()).ok_or(Error::NewDomainFail)?;
    Ok(Evaluations::from_vec_and_domain(v, D).interpolate())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_grumpkin::Fr;

    pub fn to_F_matrix<F: MVM>(M: Vec<Vec<usize>>) -> SparseMatrix<F> {
        dense_matrix_to_sparse(to_F_dense_matrix(M))
    }
    pub fn to_F_dense_matrix<F: PrimeField>(M: Vec<Vec<usize>>) -> Vec<Vec<F>> {
        M.iter()
            .map(|m| m.iter().map(|r| F::from(*r as u64)).collect())
            .collect()
    }
    pub fn to_F_vec<F: PrimeField>(z: Vec<usize>) -> Vec<F> {
        z.iter().map(|c| F::from(*c as u64)).collect()
    }

    #[test]
    fn test_dense_sparse_conversions() {
        let A = to_F_dense_matrix::<Fr>(vec![
            vec![0, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 1, 0, 0, 1, 0],
            vec![5, 0, 0, 0, 0, 1],
        ]);
        let A_sparse = dense_matrix_to_sparse(A.clone());
        assert_eq!(A_sparse.to_dense(), A);
    }

    // test mat_vec_mul & mat_vec_mul_sparse
    #[test]
    fn test_mat_vec_mul() {
        let A = to_F_matrix::<Fr>(vec![
            vec![0, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 1, 0, 0, 1, 0],
            vec![5, 0, 0, 0, 0, 1],
        ])
        .to_dense();
        let z = to_F_vec(vec![1, 3, 35, 9, 27, 30]);
        assert_eq!(mat_vec_mul(&A, &z).unwrap(), to_F_vec(vec![3, 9, 30, 35]));
        assert_eq!(
            mat_vec_mul_sparse(&dense_matrix_to_sparse(A), &z).unwrap(),
            to_F_vec(vec![3, 9, 30, 35])
        );

        let A = to_F_matrix::<Fr>(vec![vec![2, 3, 4, 5], vec![4, 8, 12, 14], vec![9, 8, 7, 6]]);
        let v = to_F_vec(vec![19, 55, 50, 3]);

        assert_eq!(
            mat_vec_mul(&A.to_dense(), &v).unwrap(),
            to_F_vec(vec![418, 1158, 979])
        );
        assert_eq!(
            mat_vec_mul_sparse(&A, &v).unwrap(),
            to_F_vec(vec![418, 1158, 979])
        );
    }

    #[test]
    fn test_hadamard_product() {
        let a = to_F_vec::<Fr>(vec![1, 2, 3, 4, 5, 6]);
        let b = to_F_vec(vec![7, 8, 9, 10, 11, 12]);
        assert_eq!(
            hadamard(&a, &b).unwrap(),
            to_F_vec(vec![7, 16, 27, 40, 55, 72])
        );
    }

    #[test]
    fn test_vec_add() {
        let a: Vec<Fr> = to_F_vec::<Fr>(vec![1, 2, 3, 4, 5, 6]);
        let b: Vec<Fr> = to_F_vec(vec![7, 8, 9, 10, 11, 12]);
        assert_eq!(
            vec_add(&a, &b).unwrap(),
            to_F_vec(vec![8, 10, 12, 14, 16, 18])
        );
    }
}
