// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Math
{
    using System.Runtime.Serialization;

    /// <summary>
    /// Upper triangular matrix class
    /// </summary>
    [DataContract]
    public class UpperTriangularMatrix : Matrix
    {
        /// <summary>
        /// Default constructor just used for serialization.
        /// </summary>
        protected UpperTriangularMatrix()
        {
        }
        /// <summary>
        /// Constructs a zero matrix of the given dimensions.
        /// </summary>
        /// <param name="nRows">Number of rows >= 0.</param>
        /// <param name="nCols">Number of columns >= 0.</param>
        public UpperTriangularMatrix(int nRows, int nCols)
            : base(nRows, nCols)
        {
        }

        /// <summary>
        /// Constructs a matrix by referencing an array.
        /// </summary>
        /// <param name="nRows">Number of rows.</param>
        /// <param name="nCols">Number of columns.</param>
        /// <param name="data">Storage for the matrix elements.</param>
        public UpperTriangularMatrix(int nRows, int nCols, double[] data)
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
        public UpperTriangularMatrix(double[,] data)
            : base(data)
        {
        }

        /// <summary>
        /// Creates a full clone of this upper triangular matrix
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            UpperTriangularMatrix result = new UpperTriangularMatrix(rows, cols);
            result.SetTo(this);
            return result;
        }

        /// <summary>
        /// Check that this matrix is upper triangular
        /// </summary>
        /// <exclude/>
        public void CheckUpperTriangular()
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    Assert.IsTrue(this[i, j] == 0, "Matrix is not upper triangular");
                }
            }
        }

        /// <summary>
        /// Transposes a given lower triangular matrix in place.
        /// </summary>
        /// <param name="L">Matrix to transpose.  Contents are corrupted on exit.</param>
        /// <returns>An upper triangular wrapper of L's source array.</returns>
        /// <remarks>L is no longer a valid lower triangular matrix.</remarks>
        public static UpperTriangularMatrix TransposeInPlace(LowerTriangularMatrix L)
        {
            L.CheckLowerTriangular();
            UpperTriangularMatrix U = new UpperTriangularMatrix(L.Cols, L.Rows,
                                                                L.SourceArray);
            Assert.IsTrue(U.Rows == U.Cols);
            for (int i = 0; i < U.Rows; ++i)
            {
                for (int j = i + 1; j < U.Cols; ++j)
                {
                    U[i, j] = U[j, i];
                    U[j, i] = 0;
                }
            }
            U.CheckUpperTriangular();
            return U;
        }

#if false
        public UpperTriangularMatrix GetsTranspose(LowerTriangularMatrix L)
        {
            if (rows != L.Cols || cols != L.Rows) {
                throw new ArgumentException("Output matrix is not compatible with the transpose", "that");
            }
            L.CheckLowerTriangular();
            // lower triangle of this is assumed to be zero
            for (int i = 0; i < rows; ++i) {
                for (int j = i; j < cols; ++j) {
                    this[i, j] = L[j, i];
                }
            }
            CheckUpperTriangular();
            return this;
        }
#endif

        /// <summary>
        /// Return the transpose of this upper triangular matrix
        /// </summary>
        /// <returns></returns>
        public new LowerTriangularMatrix Transpose()
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(cols, rows);
            L.SetToTranspose(this);
            return L;
        }

        /// <summary>
        /// Returns the product of diagonal elements.
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
        /// Returns the determinant of this upper-triangular matrix.
        /// </summary>
        /// <returns>The determinant of this.</returns>
        public new double Determinant()
        {
            CheckUpperTriangular();
            return ProdDiag();
        }

        /// <summary>
        /// Returns the sum of the logarithm of diagonal elements.
        /// </summary>
        /// <returns><c>sum(log(diag(this)))</c>.</returns>
        public double TraceLn()
        {
            Assert.IsTrue(rows == cols);
            double sum = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                sum += System.Math.Log(this[i, i]);
            }
            return (sum);
        }

        /// <summary>
        /// Returns the natural logarithm of the determinant of this upper-triangular matrix.
        /// </summary>
        /// <returns>The log-determinant of this.</returns>
        public double LogDeterminant()
        {
            CheckUpperTriangular();
            return TraceLn();
        }
    }
}