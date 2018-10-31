// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Math
{
    using System.Runtime.Serialization;

    /// <summary>
    /// Class for lower triangular matrices
    /// </summary>
    [DataContract]
    public class LowerTriangularMatrix : Matrix
    {
        /// <summary>
        /// Default constructor just used for serialization.
        /// </summary>
        protected LowerTriangularMatrix()
        {
        }
        /// <summary>
        /// Constructs a zero matrix of the given dimensions.
        /// </summary>
        /// <param name="nRows">Number of rows >= 0.</param>
        /// <param name="nCols">Number of columns >= 0.</param>
        public LowerTriangularMatrix(int nRows, int nCols)
            : base(nRows, nCols)
        {
        }

        /// <summary>
        /// Constructs a matrix by referencing an array.
        /// </summary>
        /// <param name="data">Storage for the matrix elements.</param>
        /// <param name="nRows">Number of rows.</param>
        /// <param name="nCols">Number of columns.</param>
        public LowerTriangularMatrix(int nRows, int nCols, double[] data)
            : base(nRows, nCols, data)
        {
        }

        /// <summary>
        /// Constructs a matrix from data in a 2D array.
        /// </summary>
        /// <param name="data">2D array of elements.</param>
        /// <remarks>The 2D array is copied into new storage.
        /// The size of the matrix is taken from the array.
        /// </remarks>
        public LowerTriangularMatrix(double[,] data)
            : base(data)
        {
        }

        /// <summary>
        /// Creates a full clone of this instance (including the data)
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            LowerTriangularMatrix result = new LowerTriangularMatrix(rows, cols);
            result.SetTo(this);
            return result;
        }

        /// <summary>
        /// Checks that this instance is lower triangular
        /// </summary>
        /// <exclude/>
        [Conditional("DEBUG")]
        public void CheckLowerTriangular()
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = i + 1; j < cols; j++)
                {
                    Assert.IsTrue(this[i, j] == 0, "Matrix is not lower triangular");
                }
            }
        }

#if false
    /// <remarks>U is no longer a valid upper triangular matrix.</remarks>
        public static LowerTriangularMatrix TransposeInPlace(UpperTriangularMatrix U)
        {
            U.CheckUpperTriangular();
            LowerTriangularMatrix L = new LowerTriangularMatrix(U.Cols, U.Rows,
                                                                                                     U.SourceArray, U.Start);
            Assert.IsTrue(L.Rows == L.Cols);
            for (int i = 0; i < L.Rows; ++i) {
                for (int j = 0; j < i; ++j) {
                    L[i, j] = L[j, i];
                    L[j, i] = 0;
                }
            }
            L.CheckLowerTriangular();
            return L;
        }
#endif

        /// <summary>
        /// Modifies <c>this</c> to be the inverse of A.
        /// </summary>
        /// <param name="A">Can be the same object as <c>this</c></param>
        public void SetToInverse(LowerTriangularMatrix A)
        {
#if LAPACK
            if (object.ReferenceEquals(this, A)) {
                A = (LowerTriangularMatrix)A.Clone();
            }
            this.SetToIdentity();
            Lapack.PredivideBy(this, A);
#else
            // Reference: Golub and van Loan (1996)
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == this.Rows);
            for (int k = 0; k < rows; k++)
            {
                if (A[k, k] == 0)
                    throw new MatrixSingularException(A);
            }
            for (int j = 0; j < cols; j++)
            {
                int index1 = j;
                // k < j case
                for (int k = 0; k < j; k++)
                {
                    //B[k, j] = 0.0;
                    data[index1] = 0.0;
                    index1 += cols;
                }
                // k == j case
                int index2 = j + j*cols;
                double v = 1.0/A.data[index2];
                data[index1] = v;
                for (int i = j + 1; i < rows; i++)
                {
                    // B[i, j] = -v * A[i, j];
                    index1 += cols;
                    index2 += cols;
                    data[index1] = -v*A.data[index2];
                }
                // k > j case
                for (int k = j + 1; k < rows; k++)
                {
                    index1 = j + k*cols;
                    index2 = k + k*cols;
                    if (data[index1] != 0)
                    {
                        //double u = B[k,j]/A[k,k];
                        //B[k, j] = u;
                        // TM: this style of indexing may look ugly but it runs much faster
                        double u = data[index1]/A.data[index2];
                        data[index1] = u;
                        for (int i = k + 1; i < rows; i++)
                        {
                            // B[i, j] -= u * A[i, k];
                            index1 += cols;
                            index2 += cols;
                            data[index1] -= u*A.data[index2];
                        }
                    }
                }
            }
#endif
        }

        /// <summary>
        /// Returns the inverse of this lower triangular matrix
        /// </summary>
        /// <returns></returns>
        public new LowerTriangularMatrix Inverse()
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(rows, cols);
            L.SetToInverse(this);
            return L;
        }

#if false
    /// <summary>
    /// Set this lower triangular matrix to the transpose of an upper triangular matrix
    /// </summary>
    /// <param name="U">The matrix to transpose</param>
    /// <returns></returns>
        public LowerTriangularMatrix SetToTranspose(UpperTriangularMatrix U)
        {
            if (rows != U.Cols || cols != U.Rows) {
                throw new ArgumentException("Output matrix is not compatible with the transpose", "that");
            }
            U.CheckUpperTriangular();
            // upper triangle of this is assumed to be zero
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j <= i; ++j) {
                    this[i, j] = U[j, i];
                }
            }
            CheckLowerTriangular();
            return this;
        }
#endif

        /// <summary>
        /// Returns the transpose of this lower triangular matrix
        /// </summary>
        /// <returns></returns>
        public new UpperTriangularMatrix Transpose()
        {
            UpperTriangularMatrix U = new UpperTriangularMatrix(cols, rows);
            U.SetToTranspose(this);
            return U;
        }

        /// <summary>
        /// Returns the product of diagonal elements of this lower triangular matrix
        /// </summary>
        /// <returns><c>prod(diag(this))</c>.</returns>
        public double ProdDiag()
        {
            Assert.IsTrue(rows == cols);
            double prod = 1.0;
            for (int i = 0; i < rows; ++i)
            {
                prod *= this[i, i];
            }
            return (prod);
        }

        /// <summary>
        /// Returns the determinant of this lower-triangular matrix.
        /// </summary>
        /// <returns>The determinant of this.</returns>
        public new double Determinant()
        {
            CheckLowerTriangular();
            return ProdDiag();
        }

        /// <summary>
        /// Returns the sum of the logarithm of diagonal elements.
        /// </summary>
        /// <param name="ignoreInfinity">If true, +infinity on the diagonal is treated as 1.</param>
        /// <returns><c>sum(log(diag(this)))</c>.</returns>
        public double TraceLn(bool ignoreInfinity = false)
        {
            Assert.IsTrue(rows == cols);
            double sum = 0.0;
            if (ignoreInfinity)
            {
                for (int i = 0; i < rows; ++i)
                {
                    if (!double.IsPositiveInfinity(this[i, i]))
                        sum += System.Math.Log(this[i, i]);
                }
            }
            else
            {
                for (int i = 0; i < rows; ++i)
                {
                    sum += System.Math.Log(this[i, i]);
                }
            }
            return sum;
        }

        /// <summary>
        /// Returns the natural logarithm of the determinant of a lower-triangular matrix.
        /// </summary>
        /// <param name="ignoreInfinity">If true, +infinity on the diagonal is treated as 1.</param>
        /// <returns>The log-determinant of this.</returns>
        public double LogDeterminant(bool ignoreInfinity = false)
        {
            CheckLowerTriangular();
            return TraceLn(ignoreInfinity);
        }

        /// <summary>
        /// Gets the Cholesky decomposition L, such that L*L' = A.
        /// </summary>
        /// <param name="A">A symmetric matrix to decompose.</param>
        /// <returns>True if <paramref name="A"/> is positive definite, otherwise false.</returns>
        /// <remarks>
        /// The decomposition is a lower triangular matrix L, returned in <c>this</c>.
        /// <paramref name="A"/> must be symmetric, but need not be positive definite.
        /// If <paramref name="A"/> is positive semidefinite, 
        /// then L will satisfy L*L' = A.
        /// Otherwise, L will only approximately satisfy L*L' = A.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public bool SetToCholesky(Matrix A)
        {
            A.CheckSymmetry(nameof(A));
#if LAPACK
            SetTo(A);
            bool isPosDef = Lapack.CholeskyInPlace(this);
#else
            CheckCompatible(A, nameof(A));

            bool isPosDef = true;
            LowerTriangularMatrix L = this;
            // compute the Cholesky factor
            // Reference: Golub and van Loan (1996)
            for (int i = 0; i < A.Cols; i++)
            {
                // upper triangle
                for (int j = 0; j < i; j++) L[j, i] = 0;
                // lower triangle
                for (int j = i; j < A.Rows; j++)
                {
                    double sum = A[i, j];
                    int index1 = i*cols;
                    int index2 = j*cols;
                    for (int k = 0; k < i; k++)
                    {
                        //sum -= L[i, k] * L[j, k];
                        sum -= data[index1++]*data[index2++];
                    }
                    if (i == j)
                    {
                        // diagonal entry
                        if (sum <= 0)
                        {
                            isPosDef = false;
                            L[i, i] = 0;
                        }
                        else
                        {
                            L[i, i] = System.Math.Sqrt(sum);
                        }
                    }
                    else
                    {
                        // off-diagonal entry
                        if (L[i, i] > 0)
                        {
                            L[j, i] = sum/L[i, i];
                        }
                        else
                        {
                            L[j, i] = 0;
                        }
                    }
                }
            }
            CheckLowerTriangular();
#endif
            return isPosDef;
        }
    }
}