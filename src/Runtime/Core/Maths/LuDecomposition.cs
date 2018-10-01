// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Class for calculating and doing operations with an LU decomposition
    /// </summary>
    public class LuDecomposition
    {
        /// <summary>
        /// Stores the permuted L in the lower triangle (whose diagonal is assumed to be 1), with U in the upper triangle and diagonal.
        /// </summary>
        public Matrix LU;

        private int[] pivots;
        private double parity = 1.0;

        /// <summary>
        /// Creates a new instance with a given number of rows and columns
        /// </summary>
        /// <param name="nRows">Number of rows</param>
        /// <param name="nCols">Number of columns</param>
        /// <remarks><para>
        /// Currently this only supports square matrices
        /// </para></remarks>
        public LuDecomposition(int nRows, int nCols)
        {
            // we currently only support square matrices
            Assert.IsTrue(nRows == nCols, "nRows and nCols do not match");
            LU = new Matrix(nRows, nCols);
            pivots = new int[nRows];
        }

        /// <summary>
        /// Constructs an instance and performs the decomposition on the given matrix.
        /// </summary>
        /// <param name="A"></param>
        public LuDecomposition(Matrix A)
            : this(A.Rows, A.Cols)
        {
            Decompose(A);
        }

        /// <summary>
        /// Creates performs and returns an LuDecomposition on a given matrix.
        /// </summary>
        /// <param name="A"></param>
        public static LuDecomposition InPlace(Matrix A)
        {
            Assert.IsTrue(A.Rows == A.Cols, "nRows and nCols do not match");
            LuDecomposition result = new LuDecomposition(A.Rows, A.Cols);
            result.LU = A;
            result.pivots = new int[A.Rows];
            result.Decompose();
            return result;
        }

        /// <summary>
        /// Decomposes the matrix A
        /// </summary>
        /// <param name="A"></param>
        public void Decompose(Matrix A)
        {
            LU.SetTo(A);
            Decompose();
        }

        /// <summary>
        /// Performs the decomposition for this instance
        /// </summary>
        public void Decompose()
        {
            // Origin: "Numerical Recipes in C" by Press et al (1992)
            double[] rowScale = new double[LU.Rows];
            int pivot = 0;

            /* Scan the rows for implicit scaling information */
            for (int i = 0; i < LU.Rows; ++i)
            {
                // largest = largest elt in this row
                double largest = 0.0;
                for (int j = 0; j < LU.Cols; ++j)
                {
                    double element = System.Math.Abs(LU[i, j]);
                    if (element > largest) largest = element;
                }
                if (largest < Double.Epsilon)
                    throw new MatrixSingularException(LU);
                rowScale[i] = 1.0/largest;
            }

            /* Use Crout's algorithm to organise the columns */
            for (int j = 0; j < LU.Cols; ++j)
            {
                for (int i = 0; i < j; ++i)
                {
                    double sum = LU[i, j];
                    for (int k = 0; k < i; ++k)
                    {
                        sum -= LU[i, k]*LU[k, j];
                    }
                    LU[i, j] = sum;
                }
                double largest = 0.0;
                for (int i = j; i < LU.Rows; ++i)
                {
                    double sum = LU[i, j];
                    for (int k = 0; k < j; ++k)
                    {
                        sum -= LU[i, k]*LU[k, j];
                    }
                    LU[i, j] = sum;

                    /* The best pivot is the largest one */
                    double pivotMerit = rowScale[i] * System.Math.Abs(sum);
                    if (pivotMerit >= largest)
                    {
                        largest = pivotMerit;
                        pivot = i;
                    }
                }

                // Swap rows to put the pivot on the diagonal
                if (j != pivot)
                {
                    for (int k = 0; k < LU.Cols; ++k)
                    {
                        double temp = LU[pivot, k];
                        LU[pivot, k] = LU[j, k];
                        LU[j, k] = temp;
                    }
                    rowScale[pivot] = rowScale[j];
                    /* We've switched parity when we swap rows! */
                    parity = -parity;
                }
                pivots[j] = pivot;

                /* If we put 0 on the diagonal, replace it with a small number */
                if (LU[j, j] == 0.0) LU[j, j] = Double.Epsilon;

                /* Now scale by the pivot element */
                if (j != LU.Cols - 1)
                {
                    double scale = 1.0/LU[j, j];
                    for (int i = j + 1; i < LU.Rows; ++i)
                    {
                        LU[i, j] *= scale;
                    }
                }
            }
        }

        /// <summary>
        /// Compute the determinant of the decomposed matrix. The
        /// decomposition is assumed to have been performed.
        /// </summary>
        /// <returns></returns>
        public double Determinant()
        {
            double det = parity;
            for (int i = 0; i < LU.Rows; ++i)
            {
                det *= LU[i, i];
            }
            return det;
        }

        /// <summary>
        /// Solves Ay = x for y, leaving the result in x.
        /// </summary>
        /// <param name="x"></param>
        public void Solve(Vector x)
        {
            Assert.IsTrue(x.Count == LU.Rows);
            int iDim = x.Count;
            int iNonZeroIndexB = -1;
            double sum = 0.0;

            for (int i = 0; i < iDim; ++i)
            {
                int pivot = pivots[i];

                /* If the rows are permuted, switch values on the RHS */
                sum = x[pivot];
                x[pivot] = x[i];

                if (iNonZeroIndexB != -1)
                {
                    /* Forward substitute */
                    for (int j = iNonZeroIndexB; j < i; ++j)
                    {
                        sum -= LU[i, j]*x[j];
                    }
                }
                else if (sum != 0.0)
                {
                    iNonZeroIndexB = i;
                }

                x[i] = sum;
            }

            /* Back-substitution */
            for (int i = iDim - 1; i >= 0; --i)
            {
                sum = x[i];
                for (int j = i + 1; j < iDim; ++j)
                {
                    sum -= LU[i, j]*x[j];
                }

                x[i] = sum/LU[i, i];
            }
        }
    }
}