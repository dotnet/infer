// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Math
{
#if LAPACK
    using ptrdiff_t = Int64;
    using System.Runtime.InteropServices;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Matrix implementations making use of Lapack
    /// </summary>
    unsafe public static class Lapack
    {
        const string dllName = "mkl_rt.dll";

        // http://software.intel.com/en-us/node/469230#00589A5E-742D-44A6-AEEF-DA54064AFEA5
        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern void dgeev(char* jobvl, char* jobvr, ptrdiff_t* n, double* a, ptrdiff_t* lda, double* wr, double* wi, double* vl, 
            ptrdiff_t *ldvl, double *vr, ptrdiff_t *ldvr, double *work, ptrdiff_t *lwork, ptrdiff_t *info);

        /// <summary>
        /// Computes the eigenvalues of a nonsymmetric matrix, overwriting the matrix.
        /// </summary>
        /// <param name="A">On input, an arbitrary matrix.  On output, undefined.</param>
        /// <param name="eigenvaluesReal">Modified to contain the real part of the eigenvalues.</param>
        /// <param name="eigenvaluesImag">Modified to contain the imaginary part of the eigenvalues.</param>
        public static void EigenvaluesInPlace(Matrix A, double[] eigenvaluesReal, double[] eigenvaluesImag)
        {
            char jobvl = 'N';
            char jobvr = 'N';
            ptrdiff_t n = A.Rows;
            ptrdiff_t ldvl = 1;
            ptrdiff_t ldvr = 1;
            ptrdiff_t lwork = 3 * n;
            double[] workArray = new double[lwork];
            ptrdiff_t info;

            fixed (double* sA = A.SourceArray, sR = eigenvaluesReal, sI = eigenvaluesImag, work = workArray)
            {
                dgeev(&jobvl, &jobvr, &n, sA, &n, sR, sI, null, &ldvl, null, &ldvr, work, &lwork, &info);
            }
            if (info > 0)
            {
                throw new Exception("dgeev failed\n");
            }
        }

        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern void dsyev(char* jobz, char* uplo, ptrdiff_t* n, double* a, ptrdiff_t* lda, double* w, double* work, ptrdiff_t* lwork, ptrdiff_t* info);

        /// <summary>
        /// Compute the eigenvectors and eigenvalues of a symmetric matrix in place
        /// </summary>
        /// <param name="A">On input, a symmetric matrix.  On output, the eigenvectors as rows.</param>
        /// <param name="eigenvalues">Modified to contain the eigenvalues</param>
        public static void EigenvectorsSymmetricInPlace(Matrix A, double[] eigenvalues)
        {
            char jobz = 'V';
            char uplo = 'U';
            ptrdiff_t rank = A.Rows;
            ptrdiff_t lda = rank;
            ptrdiff_t lwork;
            double[] workArray;
            ptrdiff_t info;

            lwork = 3 * A.Rows;
            workArray = new double[lwork];

            fixed (double* sA = A.SourceArray, sE = eigenvalues, work = workArray) {
                dsyev(&jobz, &uplo, &rank, sA, &lda, sE, work, &lwork, &info);
            }
            if (info > 0) {
                throw new Exception("dsysev failed\n");
            }
        }

        [DllImport(dllName, CallingConvention=CallingConvention.Cdecl, ExactSpelling=true, SetLastError=false)]
        internal static extern void dgemm(char* transa, char* transb,
                                                                        ptrdiff_t* m, ptrdiff_t* n, ptrdiff_t* k,
                                                                        double* alpha, double* A,
                                                                        ptrdiff_t* lda, double* B, ptrdiff_t* ldb,
                                                                        double* beta, double* C,
                                                                        ptrdiff_t* ldc);

        /// <summary>
        /// Sets the matrix result to the product of the two matrix A and matrix B
        /// </summary>
        /// <param name="result">The result matrix</param>
        /// <param name="A">The first matrix</param>
        /// <param name="B">The second matrix</param>
        public static void SetToProduct(Matrix result, Matrix A, Matrix B)
        {
            Assert.IsTrue(result.Rows == A.Rows);
            Assert.IsTrue(result.Cols == B.Cols);
            Assert.IsTrue(A.Cols == B.Rows);
            Assert.IsTrue(!object.ReferenceEquals(result, A));
            Assert.IsTrue(!object.ReferenceEquals(result, B));
            // Lapack is column-major, so we give it the implicitly transposed matrices and reverse their order:
            // A*B = (B'*A')'
            char trans = 'n';
            double zero = 0, one = 1;
            ptrdiff_t m = result.Cols, n = result.Rows, k = B.Rows, lda = B.Cols;
            // ms-help://MS.VSCC.v80/MS.MSDN.v80/MS.VisualStudio.v80.en/dv_csref/html/ec16fbb4-a24e-45f5-a763-9499d3fabe0a.htm
            fixed (double* s = result.SourceArray, sA = A.SourceArray, sB = B.SourceArray) {
                double* p = s, pA = sA, pB = sB;
                dgemm(&trans, &trans, &m, &n, &k, &one, pB, &lda, pA, &k, &zero, p, &m);
            }
        }

        [DllImport(dllName, CallingConvention=CallingConvention.Cdecl, ExactSpelling=true, SetLastError=false)]
        internal static extern void dgemv(char* trans, ptrdiff_t* m, ptrdiff_t* n,
                                                                        double* alpha, double* A, ptrdiff_t* lda,
                                                                        double* x, ptrdiff_t* incx, double* beta,
                                                                        double* y, ptrdiff_t* incy);

        /// <summary>
        /// Sets the vector result to the product of matrix A and vector b
        /// </summary>
        /// <param name="result">The resulting vector</param>
        /// <param name="A">The given matrix</param>
        /// <param name="v">The given vector</param>
        public static void SetToProduct(DenseVector result, Matrix A, DenseVector v)
        {
            Assert.IsTrue(result.Count == A.Rows);
            Assert.IsTrue(A.Cols == v.Count);
            Assert.IsTrue(!object.ReferenceEquals(result, v));
            // Lapack is column-major, so we tell it that A is transposed.
            char trans = 't';
            ptrdiff_t m = A.Cols, n = A.Rows;
            double zero = 0, one = 1;
            ptrdiff_t i_one = 1;
            fixed (double* s = result.SourceArray, sA = A.SourceArray, sv = v.SourceArray) {
                double* p = s, pA = sA, pv = sv;
                dgemv(&trans, &m, &n, &one, pA, &m, pv, &i_one, &zero, p, &i_one);
            }
        }

        // http://www.netlib.org/lapack/double/dpotrf.f
        // http://software.intel.com/en-us/node/468690
        [DllImport(dllName, CallingConvention=CallingConvention.Cdecl, ExactSpelling=false, SetLastError=false)]
        internal static extern int dpotrf(char* uplo, ptrdiff_t* n, double* A, ptrdiff_t* lda, ptrdiff_t* info);

        /// <summary>
        /// Performs a Cholesky decomposition in place on matrix A
        /// </summary>
        /// <param name="A">The given matrix</param>
        /// <returns>True if <paramref name="A"/> is positive definite, otherwise false.</returns>
        public static bool CholeskyInPlace(Matrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            // Lapack is column-major, so we request the upper triangle and implicitly transpose it.
            char uplo = 'U';
            ptrdiff_t lda = A.Rows, info;
            int rank = A.Rows;
            fixed (double* s = A.SourceArray) {
                double* p = s;
                dpotrf(&uplo, &lda, p, &lda, &info);
                if (info > 0) return false;
                // clear the upper triangle (it will have the original matrix contents)
                for (int i = 0; i < rank; i++) {
                    int row = i * rank; // expression lifting
                    for (int j = i + 1; j < rank; j++) {
                        p[row + j] = 0.0;
                    }
                }
            }
            return true;
        }

        // http://www.netlib.org/lapack/double/dpotri.f
        [DllImport(dllName, CallingConvention=CallingConvention.Cdecl, ExactSpelling=true, SetLastError=false)]
        internal static extern int dpotri(char* uplo, ptrdiff_t* n, double* A, ptrdiff_t* lda, ptrdiff_t* info);

        /// <summary>
        /// Performs a Cholsky decomposition in place on matrix A
        /// </summary>
        /// <param name="A">The given matrix</param>
        /// <returns></returns>
        public static bool SymmetricInverseInPlace(Matrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            // Lapack is column-major, so we request the upper triangle and implicitly transpose it.
            char uplo = 'U';
            ptrdiff_t lda = A.Rows, info;
            int rank = A.Rows;
            fixed (double* s = A.SourceArray) {
                double* p = s;
                dpotrf(&uplo, &lda, p, &lda, &info);
                if (info > 0) return false;
                dpotri(&uplo, &lda, p, &lda, &info);
                if (info > 0) return false;
                for (int i = 0; i < rank; i++) {
                    int row = i * rank; // expression lifting
                    for (int j = i + 1; j < rank; j++) {
                        p[row + j] = p[j * rank + i];
                    }
                }
            }
            return true;
        }

        // http://www.netlib.org/blas/dtrsm.f
        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern int dtrsm(char* side, char* uplo, char* transa, char* diag,
            ptrdiff_t* m, ptrdiff_t* n, double* alpha, double* a, ptrdiff_t* lda,
            double* b, ptrdiff_t* ldb);

        /// <summary>
        /// Sets b to inv(A)*b.
        /// </summary>
        /// <param name="b"></param>
        /// <param name="A"></param>
        public static void PredivideBy(DenseVector b, LowerTriangularMatrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == b.Count);
            // Lapack is column-major, so we tell it that A is upper triangular and transposed
            char side = 'L', uplo = 'U', trans = 'T', diag = 'N';
            double alpha = 1;
            ptrdiff_t m = b.Count, n = 1;
            fixed (double* sA = A.SourceArray, sB = b.SourceArray) {
                double* pA = sA, pB = sB;
                dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, pA, &m, pB, &m);
            }
        }
        /// <summary>
        /// Sets B to A\B.
        /// </summary>
        /// <param name="B"></param>
        /// <param name="A"></param>
        public static void PredivideBy(Matrix B, LowerTriangularMatrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == B.Rows);
            Assert.IsTrue(!object.ReferenceEquals(B, A));
            // Lapack is column-major, so we tell it that A is upper triangular and switch sides:
            // inv(A)*B = (B'*inv(A'))'
            char side = 'R', uplo = 'U', trans = 'N', diag = 'N';
            double alpha = 1;
            ptrdiff_t m = B.Cols, n = B.Rows;
            fixed (double* sA = A.SourceArray, sB = B.SourceArray) {
                double* pA = sA, pB = sB;
                dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, pA, &n, pB, &m);
            }
        }

        /// <summary>
        /// Pre-divides vector b by lower triangular matrix
        /// </summary>
        /// <param name="b"></param>
        /// <param name="A"></param>
        public static void PredivideByTranspose(DenseVector b, LowerTriangularMatrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == b.Count);
            // Lapack is column-major, so we tell it that A is upper triangular 
            // (and not transposed in this case)
            char side = 'L', uplo = 'U', trans = 'N', diag = 'N';
            double alpha = 1;
            ptrdiff_t m = b.Count, n = 1;
            fixed (double* sA = A.SourceArray, sB = b.SourceArray) {
                double* pA = sA, pB = sB;
                dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, pA, &m, pB, &m);
            }
        }
        /// <summary>
        /// Sets B to A'\B.
        /// </summary>
        /// <param name="B"></param>
        /// <param name="A"></param>
        public static void PredivideByTranspose(Matrix B, LowerTriangularMatrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == B.Rows);
            Assert.IsTrue(!object.ReferenceEquals(B, A));
            // Lapack is column-major, so we tell it that A is upper triangular and switch sides:
            // inv(A)*B = (B'*inv(A'))'
            char side = 'R', uplo = 'U', trans = 'T', diag = 'N';
            double alpha = 1;
            ptrdiff_t m = B.Cols, n = B.Rows;
            fixed (double* sA = A.SourceArray, sB = B.SourceArray) {
                double* pA = sA, pB = sB;
                dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, pA, &n, pB, &m);
            }
        }

        /// <summary>
        /// Pre-divides vector b by upper triangular matrix A
        /// </summary>
        /// <param name="b"></param>
        /// <param name="A"></param>
        public static void PredivideBy(DenseVector b, UpperTriangularMatrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == b.Count);
            // Lapack is column-major, so we tell it that A is lower triangular and transposed
            char side = 'L', uplo = 'L', trans = 'T', diag = 'N';
            double alpha = 1;
            ptrdiff_t m = b.Count, n = 1;
            fixed (double* sA = A.SourceArray, sB = b.SourceArray) {
                double* pA = sA, pB = sB;
                dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, pA, &m, pB, &m);
            }
        }
        /// <summary>
        /// Pre-divides matrix B by upper triangular matrix A
        /// </summary>
        /// <param name="B"></param>
        /// <param name="A"></param>
        public static void PredivideBy(Matrix B, UpperTriangularMatrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == B.Rows);
            Assert.IsTrue(!object.ReferenceEquals(B, A));
            // Lapack is column-major, so we tell it that A is lower triangular and switch sides:
            // inv(A)*B = (B'*inv(A'))'
            char side = 'R', uplo = 'L', trans = 'N', diag = 'N';
            double alpha = 1;
            ptrdiff_t m = B.Cols, n = B.Rows;
            fixed (double* sA = A.SourceArray, sB = B.SourceArray) {
                double* pA = sA, pB = sB;
                dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, pA, &n, pB, &m);
            }
        }
    }
#endif
}