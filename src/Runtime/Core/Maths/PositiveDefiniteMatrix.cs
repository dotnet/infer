// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Math
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// A subclass of Matrix with extra methods appropriate to positive-definite matrices.
    /// </summary>
    [Serializable]
    [DataContract]
    public class PositiveDefiniteMatrix : Matrix
    {
        /// <summary>
        /// Default constructor just used for serialization.
        /// </summary>
        protected PositiveDefiniteMatrix()
        {
        }

        /// <summary>
        /// Constructs a zero matrix of the given dimensions.
        /// </summary>
        /// <param name="rows">Number of rows >= 0.</param>
        /// <param name="cols">Number of columns >= 0.</param>
        public PositiveDefiniteMatrix(int rows, int cols)
            : base(rows, cols)
        {
        }

        /// <summary>
        /// Constructs a matrix from data in a 2D array.
        /// </summary>
        /// <param name="data">2D array of elements.</param>
        /// <remarks>The 2D array is copied into new storage.
        /// The size of the matrix is taken from the array.
        /// </remarks>
        [Construction("ToArray")]
        public PositiveDefiniteMatrix(double[,] data)
            : base(data)
        {
        }

        /// <summary>
        /// Constructs a matrix by referencing an array.
        /// </summary>
        /// <param name="data">Storage for the matrix elements.</param>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        public PositiveDefiniteMatrix(int rows, int cols, double[] data)
            : base(rows, cols, data)
        {
        }

        /// <summary>
        /// Constructs a positive-definite matrix type which references an existing matrix.
        /// </summary>
        /// <param name="A">A positive-definite matrix.</param>
        /// <remarks>This method is similar to a typecast, except it creates a new wrapper around the matrix.</remarks>
        public PositiveDefiniteMatrix(Matrix A)
            : this(A.Rows, A.Cols, A.SourceArray)
        {
        }

        /// <summary>
        /// Creates a positive-definite identity matrix of a given dimension
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        public new static PositiveDefiniteMatrix Identity(int dimension)
        {
            PositiveDefiniteMatrix id = new PositiveDefiniteMatrix(dimension, dimension);
            id.SetToIdentity();
            return id;
        }

        /// <summary>
        /// Creates a positive-definite identity matrix of a given dimension, scaled by a given value
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        public new static PositiveDefiniteMatrix IdentityScaledBy(int dimension, double scale)
        {
            PositiveDefiniteMatrix id = new PositiveDefiniteMatrix(dimension, dimension);
            id.SetToIdentityScaledBy(scale);
            return id;
        }

        /// <summary>
        /// Creates a full clone of this positive-definite matrix
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            PositiveDefiniteMatrix result = new PositiveDefiniteMatrix(rows, cols);
            result.SetTo(this);
            return result;
        }

        /// <summary>
        /// Sets this positive-definite matrix to the sum of two positive-definite matrices.
        /// Assumes compatible matrices
        /// </summary>
        /// <param name="a">First matrix, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second matrix, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public PositiveDefiniteMatrix SetToSum(PositiveDefiniteMatrix a, PositiveDefiniteMatrix b)
        {
            base.SetToSum(a, b);
            return (this);
        }

        /// <summary>
        /// Add two positive-definite matrices.
        /// </summary>
        /// <param name="a">First matrix.</param>
        /// <param name="b">Second matrix.</param>
        /// <returns>Their sum.</returns>
        public static PositiveDefiniteMatrix operator +(PositiveDefiniteMatrix a, PositiveDefiniteMatrix b)
        {
            PositiveDefiniteMatrix result = (PositiveDefiniteMatrix) a.Clone();
            result.SetToSum(result, b);
            return result;
        }

        /// <summary>
        /// Multiply matrix times scalar
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>A new matrix with entry [i,j] equal to a[i,j]*b</returns>
        public static PositiveDefiniteMatrix operator *(PositiveDefiniteMatrix a, double b)
        {
            PositiveDefiniteMatrix result = (PositiveDefiniteMatrix) a.Clone();
            result.Scale(b);
            return result;
        }

        /// <summary>
        /// Divide matrix by scalar
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>A new matrix with entry [i,j] equal to a[i,j]/b</returns>
        public static PositiveDefiniteMatrix operator /(PositiveDefiniteMatrix a, double b)
        {
            return a * (1 / b);
        }

        /// <summary>
        /// Multiply matrix times scalar
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>A new matrix with entry [i,j] equal to a*b[i,j]</returns>
        public static PositiveDefiniteMatrix operator *(double a, PositiveDefiniteMatrix b)
        {
            PositiveDefiniteMatrix result = (PositiveDefiniteMatrix)b.Clone();
            result.Scale(a);
            return result;
        }

        /// <summary>
        /// Returns the determinant of this positive-definite matrix.
        /// </summary>
        /// <returns>The determinant of <c>this</c>.</returns>
        /// <remarks>Throws a MatrixSingularException
        /// if the matrix is singular.</remarks>
        public new double Determinant()
        {
            return System.Math.Exp(LogDeterminant());
        }

        /// <summary>
        /// Returns the natural logarithm of the determinant of this positive-definite matrix.
        /// </summary>
        /// <param name="ignoreInfinity">If true, +infinity on the diagonal is treated as 1.</param>
        /// <returns>The log-determinant of <c>this</c>.</returns>
        public double LogDeterminant(bool ignoreInfinity = false)
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(rows, cols);
            return LogDeterminant(L, ignoreInfinity);
        }

        /// <summary>
        /// Returns the natural logarithm of the determinant of this positive-definite matrix
        /// where a lower triangular workspace is passed in.
        /// </summary>
        /// <param name="L">A temporary workspace, same size as <c>this</c>.</param>
        /// <param name="ignoreInfinity">If true, +infinity on the diagonal is treated as 1.</param>
        /// <returns>The log-determinant.</returns>
        /// <remarks>Throws a MatrixSingularException
        /// if the matrix is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public double LogDeterminant(LowerTriangularMatrix L, bool ignoreInfinity = false)
        {
            L.SetToCholesky(this);
            return 2*L.LogDeterminant(ignoreInfinity);
        }

        /// <summary>
        /// Returns the inverse of a positive-definite matrix.
        /// </summary>
        /// <returns>A new matrix which is the inverse of <c>this</c></returns>
        /// <remarks>
        /// Because <c>this</c> is positive definite, it must be 
        /// invertible, so this routine never throws MatrixSingularException.
        /// </remarks>
        /// <exception cref="PositiveDefiniteMatrixException">If <c>this</c> is not positive definite.</exception>
        public new PositiveDefiniteMatrix Inverse()
        {
            return (new PositiveDefiniteMatrix(Rows, Cols)).SetToInverse(this);
        }

        /// <summary>
        /// Sets this positive-definite matrix to inverse of a given positive-definite matrix.
        /// </summary>
        /// <param name="A">A symmetric positive-definite matrix, same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <exception cref="PositiveDefiniteMatrixException">If <paramref name="A"/> is not positive definite.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public PositiveDefiniteMatrix SetToInverse(PositiveDefiniteMatrix A)
        {
#if LAPACK
      this.SetTo(A);
      Lapack.SymmetricInverseInPlace(this);
      return this;
#else
            return SetToInverse(A, new LowerTriangularMatrix(rows, cols));
#endif
        }

        /// <summary>
        /// Sets this positive-definite matrix to inverse of a given positive-definite matrix
        /// where a lower triangular workspace is passed.
        /// </summary>
        /// <param name="A">A symmetric positive-definite matrix, same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="L">A workspace, same size as <paramref name="A"/>.</param>
        /// <returns><c>this</c></returns>
        /// <exception cref="PositiveDefiniteMatrixException">If <paramref name="A"/> is not positive definite.</exception>
        public PositiveDefiniteMatrix SetToInverse(PositiveDefiniteMatrix A, LowerTriangularMatrix L)
        {
            CheckCompatible(L, nameof(L));
            // Algorithm:
            // A = L*L'
            // inv(A) = inv(L')*inv(L)
            bool isPD = L.SetToCholesky(A);
            if (!isPD) throw new PositiveDefiniteMatrixException();
            L.SetToInverse(L);
            SetToOuterTranspose(L);
            return this;
        }

        /// <summary>
        /// Gets the Cholesky decomposition of the matrix (L*L' = A), replacing its contents.
        /// </summary>
        /// <param name="isPosDef">True if <c>this</c> is positive definite, otherwise false.</param>
        /// <returns>The Cholesky decomposition L.  If <c>this</c> is positive semidefinite, 
        /// then L will satisfy L*L' = A.
        /// Otherwise, L will only approximately satisfy L*L' = A.</returns>
        /// <remarks>
        /// <c>this</c> must be symmetric, but need not be positive definite.
        /// </remarks>
        public LowerTriangularMatrix CholeskyInPlace(out bool isPosDef)
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(rows, cols, data);
#if LAPACK
            isPosDef = Lapack.CholeskyInPlace(this);
#else
            isPosDef = L.SetToCholesky(this);
#endif
            return L;
        }
    }

    /// <summary>
    /// Exception thrown when a matrix is not positive definite.
    /// </summary>
    [Serializable]
    public class PositiveDefiniteMatrixException : Exception
    {
        /// <summary>
        /// Creates a new positive definite matrix exception
        /// </summary>
        public PositiveDefiniteMatrixException()
            : base("The matrix is not positive definite.")
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PositiveDefiniteMatrixException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public PositiveDefiniteMatrixException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PositiveDefiniteMatrixException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public PositiveDefiniteMatrixException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PositiveDefiniteMatrixException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected PositiveDefiniteMatrixException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}