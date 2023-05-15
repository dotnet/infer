// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;

using System;
using System.Runtime.Serialization;

using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Two-dimensional container of doubles.
    /// </summary>
    [Serializable]
    [DataContract]
    public class Matrix : IList<double>, IReadOnlyList<double>, ICloneable, SettableTo<Matrix>, CanSetAllElementsTo<double>
    {
#pragma warning disable 1591
        // Storage for the elements
        [DataMember]
        protected double[] data;
        [DataMember]
        protected readonly int rows;
        [DataMember]
        protected readonly int cols;
#pragma warning restore 1591

        /// <summary>Gets and sets an element.</summary>
        public double this[int row, int col]
        {
            get { return data[(row*cols) + col]; }

            set { data[(row*cols) + col] = value; }
        }

        /// <summary>Gets and sets an element.</summary>
        public double this[int index]
        {
            get { return data[index]; }

            set { data[index] = value; }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// A row of the matrix.
        /// </summary>
        /// <param name="row">An integer in [0,Rows-1].</param>
        /// <returns>If colStride == 1, then a Vector object pointing at one row of this matrix.  Otherwise a copy of the row.</returns>
        public Vector RowVector(int row)
        {
            if (row < 0) throw new ArgumentOutOfRangeException("row (" + row + ") < 0");
            if (row >= Rows) throw new ArgumentOutOfRangeException("row (" + row + ") > Rows-1");
            if (true)
            {
                return DenseVector.FromArrayReference(Cols, SourceArray, row*cols);
            }
            else
            {
                Vector v = Vector.Zero(Cols);
                for (int c = 0; c < Cols; c++)
                {
                    v[c] = this[row, c];
                }
                return v;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// The number of rows of the matrix.
        /// </summary>
        public int Rows
        {
            get { return rows; }
        }

        /// <summary>
        /// The number of columns of the matrix.
        /// </summary>
        public int Cols
        {
            get { return cols; }
        }

        /// <summary>
        /// Gets/sets the matrix's source array
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Performance", "CA1819:PropertiesShouldNotReturnArrays")]
        [IgnoreDataMember]
        public double[] SourceArray
        {
            get { return data; }
            set { data = value; }
        }

        /// <summary>
        /// Number of elements in the matrix
        /// </summary>
        public int Count
        {
            get { return rows*cols; }
        }

        #region IList methods

        /// <summary>
        /// Gets a typed enumerator for this matrix
        /// </summary>
        /// <returns></returns>
        public IEnumerator<double> GetEnumerator()
        {
            for (int i = 0; i < Count; i++) yield return this[i];
        }

        /// <summary>
        /// Gets an enumerator for this matrix
        /// </summary>
        /// <returns></returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Returns true if this matrix contains the given value
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Contains(double item)
        {
            for (int i = 0; i < Count; i++)
            {
                if (this[i] == item) return true;
            }
            return false;
        }

        /// <summary>
        /// Copies the values in this matrix to an array starting at a given index in the destination array
        /// </summary>
        /// <param name="array"></param>
        /// <param name="index"></param>
        public void CopyTo(double[] array, int index)
        {
            Array.Copy(data, 0, array, index, Count);
        }

        bool ICollection<double>.IsReadOnly
        {
            get { return false; }
        }

        void ICollection<double>.Add(double item)
        {
            throw new NotSupportedException();
        }

        void ICollection<double>.Clear()
        {
            throw new NotSupportedException();
        }

        bool ICollection<double>.Remove(double item)
        {
            throw new NotSupportedException();
        }

        /// <summary>
        /// Returns the first index of the given item if it exists in the matrix,
        /// otherwise returns -1
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public int IndexOf(double item)
        {
            for (int i = 0; i < Count; i++)
            {
                if (this[i] == item) return i;
            }
            return -1;
        }

        void IList<double>.Insert(int index, double item)
        {
            throw new NotSupportedException();
        }

        void IList<double>.RemoveAt(int index)
        {
            throw new NotSupportedException();
        }

        #endregion

        #region Constructors
        /// <summary>
        /// Default constructor just used for serialization.
        /// </summary>
        protected Matrix() { }

        // this constructor is preferred for serialization.
        /// <summary>
        /// Construct a zero matrix of the given dimensions.
        /// </summary>
        /// <param name="rows">Number of rows >= 0.</param>
        /// <param name="cols">Number of columns >= 0.</param>
        public Matrix(int rows, int cols)
        {
            if (rows < 0) throw new ArgumentOutOfRangeException("rows < 0");
            if (cols < 0) throw new ArgumentOutOfRangeException("cols < 0");
            data = new double[rows*cols];
            this.rows = rows;
            this.cols = cols;
            // Not required: start  = 0;
        }

        /// <summary>
        /// Construct a matrix from data in a 2D array.
        /// </summary>
        /// <param name="data">2D array of elements.</param>
        /// <remarks>The 2D array is copied into new storage.
        /// The size of the matrix is taken from the array.
        /// </remarks>
        [Construction("ToArray")]
        public Matrix(double[,] data)
            : this(data.GetLength(0), data.GetLength(1))
        {
            SetTo(data);
        }

        /// <summary>
        /// Construct a matrix from data in a 2D array.
        /// </summary>
        /// <param name="data">2D array of elements.</param>
        /// <remarks>The 2D array is copied into new storage.
        /// The size of the matrix is taken from the array.
        /// </remarks>
        public static Matrix FromArray(double[,] data)
        {
            return new Matrix(data);
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Matrix(Matrix that)
            : this(that.Rows, that.Cols)
        {
            SetTo(that);
        }

        /// <summary>
        /// Construct a matrix by referencing an array.
        /// </summary>
        /// <param name="data">Storage for the matrix elements.</param>
        /// <param name="nRows">Number of rows.</param>
        /// <param name="nCols">Number of columns.</param>
        public Matrix(int nRows, int nCols, double[] data)
        {
            if (nRows < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(nRows),
                                                      nRows,
                                                      $"{nameof(nRows)} cannot be negative");
            }
            if (nCols < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(nCols),
                                                      nCols,
                                                      $"{nameof(nCols)} cannot be negative");
            }

            this.data = data;
            cols = nCols;
            rows = nRows;
        }

        #endregion

        #region Copying

        /// <summary>
        /// Fully clones this matrix
        /// </summary>
        /// <returns></returns>
        public virtual object Clone()
        {
            return new Matrix(this);
        }

        /// <summary>
        /// Creates a 2-D arrays from this matrix
        /// </summary>
        /// <returns></returns>
        public double[,] ToArray()
        {
            double[,] result = new double[Rows,Cols];
            CopyTo(result);
            return result;
        }

        /// <summary>
        /// Copies matrix values to a two-dimensional array.
        /// </summary>
        /// <param name="values">An array whose <c>GetLength(0) >= Rows</c> and <c>GetLength(1) >= Cols</c>.</param>
        public void CopyTo(double[,] values)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    values[i, j] = this[i, j];
                }
            }
        }

        /// <summary>
        /// Copies values from another matrix.
        /// </summary>
        /// <param name="that">The second matrix, which must have the same size as this.</param>
        /// <returns>The mutated this matrix.</returns>
        public void SetTo(Matrix that)
        {
            if (!ReferenceEquals(this, that))
            {
                CheckCompatible(that, nameof(that));
                Array.Copy(that.data, 0, data, 0, Count);
            }
        }

        #endregion

        /// <summary>
        /// Sets all elements to a given value.
        /// </summary>
        /// <param name="value">The new value.</param>
        public void SetAllElementsTo(double value)
        {
            int count = Count;
            for (int i = 0; i < count; ++i)
            {
                this[i] = value;
            }
        }

        /// <summary>
        /// Tests if all elements are equal to a given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are equal to <paramref name="value"/>.</returns>
        public bool EqualsAll(double value)
        {
            int count = Count;
            for (int i = 0; i < count; ++i)
            {
                if (this[i] != value) return false;
            }
            return true;
        }

        /// <summary>
        /// Copies values from an array.
        /// </summary>
        /// <param name="values">An array whose length is at least <c>this.Count</c>.</param>
        public void SetTo(double[] values)
        {
            SetTo(values, 0);
        }

        /// <summary>
        /// Copies values from an array.
        /// </summary>
        /// <param name="values">An array whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="values"/>.</param>
        public void SetTo(double[] values, int startIndex)
        {
            int count = Count;
            for (int i = 0; i < count; ++i)
            {
                this[i] = values[i + startIndex];
            }
        }

        /// <summary>
        /// Copies values from a two-dimensional array.
        /// </summary>
        /// <param name="values">An array whose <c>GetLength(0) >= Rows</c> and <c>GetLength(1) >= Cols</c>.</param>
        public void SetTo(double[,] values)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this[i, j] = values[i, j];
                }
            }
        }

        /// <summary>
        /// max | this[i,j] - this[j,i] |
        /// </summary>
        public double SymmetryError()
        {
            if (rows != cols) return Double.PositiveInfinity;
            double max = 0;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = i + 1; j < cols; ++j)
                {
                    double x = this[i, j];
                    double y = this[j, i];
                    bool xnan = Double.IsNaN(x);
                    bool ynan = Double.IsNaN(y);
                    if (xnan != ynan)
                    {
                        return Double.PositiveInfinity;
                    }
                    else
                    {
                        // matching infinities will not change max
                        double diff = System.Math.Abs(x - y);
                        if (diff > max) max = diff;
                    }
                }
            }
            return max;
        }

        /// <summary>
        /// Set a[i,j] to the average (a[i,j]+a[j,i])/2
        /// </summary>
        /// <returns></returns>
        public void Symmetrize()
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = i + 1; j < cols; j++)
                {
                    double v = (this[i, j] + this[j, i])*0.5;
                    this[i, j] = v;
                    this[j, i] = v;
                }
            }
        }

        /// <summary>
        /// Sets the matrix to the identity.
        /// </summary>
        /// <remarks>The matrix must be square.</remarks>
        /// <returns>The mutated matrix.</returns>
        public Matrix SetToIdentity()
        {
            return SetToIdentityScaledBy(1.0);
        }

        /// <summary>
        /// Sets the matrix to the identity times a number.
        /// </summary>
        /// <param name="scale"></param>
        /// <returns>The mutated matrix.</returns>
        /// <remarks>The matrix must be square.</remarks>
        public Matrix SetToIdentityScaledBy(double scale)
        {
            if (rows != cols) throw new Exception("Matrix is rectangular.");
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    this[i, j] = ((i == j) ? scale : 0.0);
                }
            }
            return (this);
        }

        /// <summary>
        /// Creates an identity matrix of the specified size
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        public static Matrix Identity(int dimension)
        {
            Matrix id = new Matrix(dimension, dimension);
            id.SetToIdentity();
            return id;
        }

        /// <summary>
        /// Creates an identity matrix of the specified size, scaled by the specified value
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        public static Matrix IdentityScaledBy(int dimension, double scale)
        {
            Matrix id = new Matrix(dimension, dimension);
            id.SetToIdentityScaledBy(scale);
            return id;
        }

        /// <summary>
        /// Transposes the matrix.
        /// </summary>
        /// <returns>The transposed matrix.</returns>
        public Matrix Transpose()
        {
            Matrix that = new Matrix(cols, rows);
            that.SetToTranspose(this);
            return (that);
        }

        /// <summary>
        /// Sets the matrix to the transpose of another.
        /// </summary>
        /// <param name="that">The matrix to transpose.  Can be the same object as <c>this</c>.  <c>that.Count</c> must equal <c>this.Count</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the transposed matrix, and must already
        /// be the correct size.
        /// If <c>this</c> and <paramref name="that"/> are different objects but occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToTranspose(Matrix that)
        {
            if (rows != that.Cols || cols != that.Rows)
            {
                throw new ArgumentException("Output matrix is not compatible with the transpose", nameof(that));
            }
            if (ReferenceEquals(this.SourceArray, that.SourceArray))
            {
                // we must have rows == cols
                // in-place transposition of a square matrix.
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < i; ++j)
                    {
                        double temp = this[j, i];
                        this[j, i] = this[i, j];
                        this[i, j] = temp;
                    }
                }
            }
            else
            {
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        this[i, j] = that[j, i];
                    }
                }
            }
            return this;
        }

        /// <summary>
        /// Sets the matrix to a submatrix of another.
        /// </summary>
        /// <param name="that">Size must be at least <c>this.Rows+firstRow</c> by <c>this.Cols+firstColumn</c>.</param>
        /// <param name="firstRow">Index of the first row in <paramref name="that"/> to copy.</param>
        /// <param name="firstColumn">Index of the first column in <paramref name="that"/> to copy.</param>
        /// <returns><c>this</c></returns>
        public Matrix SetToSubmatrix(Matrix that, int firstRow, int firstColumn)
        {
            if (rows + firstRow > that.Rows) throw new ArgumentException("Rows (" + rows + ") + thatRow (" + firstRow + ") > that.Rows (" + that.Rows + ")");
            if (cols + firstColumn > that.Cols) throw new ArgumentException("Cols (" + cols + ") + thatColumn (" + firstColumn + ") > that.Cols (" + that.Cols + ")");
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this[i, j] = that[firstRow + i, firstColumn + j];
                }
            }
            return this;
        }

        /// <summary>
        /// Set a submatrix of this matrix to match another matrix.
        /// </summary>
        /// <param name="firstRow">Index of the first row in <c>this</c> to copy to.</param>
        /// <param name="firstColumn">Index of the first column in <c>this</c> to copy to.</param>
        /// <param name="that">Size is at most <c>this.Rows-thisRow</c> by <c>this.Cols-thisColumn</c>.</param>
        public void SetSubmatrix(int firstRow, int firstColumn, Matrix that)
        {
            if (firstRow + that.Rows > Rows) throw new ArgumentException("thisRow (" + firstRow + ") + that.Rows (" + that.Rows + ") > Rows (" + Rows + ")");
            if (firstColumn + that.Cols > Cols) throw new ArgumentException("thisColumn (" + firstColumn + ") + that.Cols (" + that.Cols + ") > Cols (" + Cols + ")");
            for (int i = 0; i < that.Rows; i++)
            {
                for (int j = 0; j < that.Cols; j++)
                {
                    this[i + firstRow, j + firstColumn] = that[i, j];
                }
            }
        }

        /// <summary>
        /// Sets a submatrix of this matrix to match another matrix.
        /// </summary>
        /// <param name="thisRow">Index of the first row in <c>this</c> to copy to.</param>
        /// <param name="thisColumn">Index of the first column in <c>this</c> to copy to.</param>
        /// <param name="that">Size is at most <c>this.Rows-firstRow</c> by <c>this.Cols-firstColumn</c>.</param>
        /// <param name="thatRow">Index of the first row in <paramref name="that"/> to copy.</param>
        /// <param name="thatColumn">Index of the first column in <paramref name="that"/> to copy.</param>
        /// <param name="numRows">The number of rows to copy</param>
        /// <param name="numColumns">The number of columns to copy</param>
        public void SetSubmatrix(int thisRow, int thisColumn, Matrix that, int thatRow, int thatColumn, int numRows, int numColumns)
        {
            if (numRows + thisRow > this.Rows) throw new ArgumentException("numRows (" + numRows + ") + thisRow (" + thisRow + ") > this.Rows (" + this.Rows + ")");
            if (numColumns + thisColumn > this.Cols)
                throw new ArgumentException("numColumns (" + numColumns + ") + thisColumn (" + thisColumn + ") > this.Cols (" + this.Cols + ")");
            if (numRows + thatRow > that.Rows) throw new ArgumentException("numRows (" + numRows + ") + thatRow (" + thatRow + ") > that.Rows (" + that.Rows + ")");
            if (numColumns + thatColumn > that.Cols)
                throw new ArgumentException("numColumns (" + numColumns + ") + thatColumn (" + thatColumn + ") > that.Cols (" + that.Cols + ")");
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numColumns; j++)
                {
                    this[i + thisRow, j + thisColumn] = that[i + thatRow, j + thatColumn];
                }
            }
        }

        /// <summary>
        /// Sets this matrix to a diagonal matrix with diagonal values specified in the given vector
        /// </summary>
        /// <param name="diag"></param>
        /// <returns></returns>
        public Matrix SetToDiagonal(Vector diag)
        {
            SetAllElementsTo(0);
            SetDiagonal(diag);
            return this;
        }

        /// <summary>
        /// Sets the diagonal elements to the values specified in the given vector
        /// </summary>
        /// <param name="diag"></param>
        /// <returns></returns>
        public Matrix SetDiagonal(Vector diag)
        {
            if (rows != cols) throw new Exception("Matrix is rectangular.");
            for (int i = 0; i < diag.Count; i++)
            {
                this[i, i] = diag[i];
            }
            return this;
        }

        /// <summary>
        /// Creates a new diagonal matrix with diagonal values specified in the given vector
        /// </summary>
        /// <param name="diag"></param>
        /// <returns></returns>
        public static Matrix FromDiagonal(Vector diag)
        {
            int count = diag.Count;
            Matrix m = new Matrix(count, count);
            m.SetDiagonal(diag);
            return m;
        }

        /// <summary>
        /// Creates a vector from the diagonal values in the matrix
        /// </summary>
        /// <returns></returns>
        public Vector Diagonal()
        {
            if (rows != cols) throw new Exception("Matrix is rectangular.");
            Vector diag = Vector.Zero(Rows);
            for (int i = 0; i < diag.Count; i++)
            {
                diag[i] = this[i, i];
            }
            return diag;
        }

        /// <summary>
        /// Enumerator which yields the diagonal elements of the matrix
        /// </summary>
        /// <returns></returns>
        public IEnumerable<double> EnumerateDiagonal()
        {
            if (rows != cols) throw new Exception("Matrix is rectangular.");
            for (int i = 0; i < Rows; i++)
            {
                yield return this[i, i];
            }
        }

        /// <summary>
        /// The sum of diagonal elements of a matrix product.
        /// </summary>
        /// <param name="a">A matrix of size n by m.</param>
        /// <param name="b">A matrix of size m by n.</param>
        /// <returns><c>sum_i sum_k a[i,k]*b[k,i]</c></returns>
        public static double TraceOfProduct(Matrix a, Matrix b)
        {
            Assert.IsTrue(a.Cols == b.Rows);
            Assert.IsTrue(a.Rows == b.Cols);
            double sum = 0;
            for (int i = 0; i < a.Rows; i++)
            {
                for (int k = 0; k < a.Cols; k++)
                {
                    if (a[i, k] != 0 && b[k, i] != 0) // avoid 0*Infinity
                        sum += a[i, k]*b[k, i];
                }
            }
            return sum;
        }

        /// <summary>
        /// The diagonal elements of a matrix product.
        /// </summary>
        /// <param name="a">A matrix of size n by m.</param>
        /// <param name="b">A matrix of size m by n.</param>
        /// <returns><c>v[i] = sum_k a[i,k]*b[k,i]</c></returns>
        public static Vector DiagonalOfProduct(Matrix a, Matrix b)
        {
            Assert.IsTrue(a.Cols == b.Rows);
            Assert.IsTrue(a.Rows == b.Cols);
            Vector result = Vector.Zero(a.Rows);
            for (int i = 0; i < result.Count; i++)
            {
                double sum = 0.0;
                for (int k = 0; k < a.Cols; k++)
                {
                    sum += a[i, k]*b[k, i];
                }
                result[i] = sum;
            }
            return result;
        }

#if false
    /// <summary>
    /// The diagonal elements of a matrix product.
    /// </summary>
    /// <param name="a">A matrix of size n by m.</param>
    /// <param name="bTranspose">A matrix of size n by m.</param>
    /// <returns><c>v[i] = sum_k a[i,k]*bTranspose[i,k]</c></returns>
        public static Vector DiagonalOfProductWithTranspose(Matrix a, Matrix bTranspose)
        {
            Assert.IsTrue(a.Cols == bTranspose.Cols);
            Assert.IsTrue(a.Rows == bTranspose.Rows);
            Vector result = new Vector(a.Rows);
            for (int i = 0; i < result.Count; i++) {
                double sum = 0.0;
                for (int k = 0; k < a.Cols; k++) {
                    sum += a[i, k] * bTranspose[i,k];
                }
                result[i] = sum;
            }
            return result;
        }
#endif

        #region Arithmetic operations

        /// <summary>
        /// Multiplies this matrix by a scalar.
        /// </summary>
        /// <param name="scale">The scalar.</param>
        /// <returns>this</returns>
        /// <remarks>this receives the product.
        /// This method is a synonym for SetToProduct(this, scale)
        /// </remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1719:ParameterNamesShouldNotMatchMemberNames", MessageId = "0#")]
        public Matrix Scale(double scale)
        {
            int count = Count;
            for (int i = 0; i < count; ++i)
            {
                this[i] *= scale;
            }
            return (this);
        }

        /// <summary>
        /// Multiplies each row of this matrix by a different scalar.
        /// </summary>
        /// <param name="rowScale">The ith element scales row i.</param>
        /// <returns>this</returns>
        /// <remarks>this receives the product.
        /// </remarks>
        public Matrix ScaleRows(Vector rowScale)
        {
            for (int i = 0; i < rows; i++)
            {
                double scale = rowScale[i];
                for (int j = 0; j < cols; j++)
                {
                    this[i, j] *= scale;
                }
            }
            return this;
        }

        /// <summary>
        /// Multiplies each column of this matrix by a different scalar.
        /// </summary>
        /// <param name="colScale">The ith element scales column i.</param>
        /// <returns>this</returns>
        /// <remarks>this receives the product.
        /// </remarks>
        public Matrix ScaleCols(Vector colScale)
        {
            for (int j = 0; j < cols; j++)
            {
                double scale = colScale[j];
                for (int i = 0; i < rows; i++)
                {
                    this[i, j] *= scale;
                }
            }
            return this;
        }

        /// <summary>
        /// Returns the median of the values in the matrix
        /// </summary>
        /// <returns></returns>
        public double Median()
        {
            return MMath.Median(data, 0, Count);
        }

        /// <summary>
        /// The element-wise product of two matrices.
        /// </summary>
        /// <param name="a">First matrix, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second matrix, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this[i,j] = a[i,j] * b[i,j]</c>.  <c>this</c> receives the product, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToElementwiseProduct(Matrix a, Matrix b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            for (int i = 0; i < Count; ++i)
            {
                this[i] = a[i]*b[i];
            }
            return (this);
        }

        /// <summary>
        /// The element-wise ratio of two matrices.
        /// </summary>
        /// <param name="a">First matrix, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second matrix, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this[i,j] = a[i,j] / b[i,j]</c>.  <c>this</c> receives the ratio, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToElementwiseRatio(Matrix a, Matrix b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            for (int i = 0; i < Count; ++i)
            {
                this[i] = a[i]/b[i];
            }
            return (this);
        }

        /// <summary>
        /// Gets a matrix times a scalar.
        /// </summary>
        /// <param name="m">A matrix, which must have the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="s">A scalar.</param>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="m"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToProduct(Matrix m, double s)
        {
            CheckCompatible(m, nameof(m));
            for (int i = 0; i < Count; ++i)
            {
                this[i] = s*m[i];
            }
            return this;
        }

        /// <summary>
        /// Multiplies every element of a matrix by a scalar.
        /// </summary>
        /// <param name="m">A matrix.</param>
        /// <param name="s">A scalar.</param>
        /// <returns>A new matrix with the product.</returns>
        public static Matrix operator *(Matrix m, double s)
        {
            Matrix result = (Matrix) m.Clone();
            result.SetToProduct(result, s);
            return (result);
        }

        /// <summary>
        /// Divides every element of a matrix by a scalar.
        /// </summary>
        /// <param name="m">A matrix.</param>
        /// <param name="s">A scalar.</param>
        /// <returns>A new matrix with the ratio.</returns>
        public static Matrix operator /(Matrix m, double s)
        {
            return m * (1 / s);
        }

        /// <summary>
        /// Modify <c>this</c> to be the product of two matrices.
        /// </summary>
        /// <param name="A">First matrix.  Cannot be <c>this</c>.</param>
        /// <param name="B">Second matrix.  Cannot be <c>this</c>.</param>
        /// <remarks><paramref name="A"/> and <paramref name="B"/> must have compatible dimensions.
        /// <c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="B"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToProduct(Matrix A, Matrix B)
        {
#if LAPACK
            Lapack.SetToProduct(this, A, B);
#else
            if(ReferenceEquals(this.SourceArray, A.SourceArray))
            {
                throw new ArgumentException($"{nameof(A)} is the same object as this.");
            }
            if(ReferenceEquals(this.SourceArray, B.SourceArray))
            {
                throw new ArgumentException($"{nameof(B)} is the same object as this.");
            }
            if (A.Cols != B.Rows)
            {
                throw new ArgumentException("Incompatible operand dimensions", nameof(A));
            }
            if (rows != A.Rows || cols != B.Cols)
            {
                throw new ArgumentException("Output matrix is not compatible with the product", nameof(A));
            }
            // data[row,col] = data[(row * cols) + col];
            int ACols = A.Cols;
            int BCols = B.Cols;
            double[] AData = A.data;
            double[] BData = B.data;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    double sum = 0.0;
                    int index1 = i*ACols;
                    int index2 = j;
                    for (int k = 0; k < A.Cols; ++k)
                    {
                        sum += AData[index1]*BData[index2];
                        // sum += A[i,k]*B[k,j]
                        index1++;
                        index2 += BCols;
                    }
                    this[i, j] = sum;
                }
            }
#endif
            return this;
        }

        /// <summary>
        /// Modify <c>this</c> to be the product of a matrix and its transpose (A*A').
        /// </summary>
        /// <param name="A">Matrix.  Cannot be <c>this</c>.</param>
        /// <remarks>
        /// <c>this</c> receives A*A' and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        /// <returns><c>this</c></returns>
        public Matrix SetToOuter(Matrix A)
        {
#if LAPACK
            this.SetToProduct(A, A.Transpose());
#else
            if(ReferenceEquals(this.SourceArray, A.SourceArray))
            {
                throw new ArgumentException($"{nameof(A)} is the same object as this.");
            }
            if (rows != A.Rows || cols != A.Rows)
            {
                throw new ArgumentException("Output matrix is not compatible with the product", nameof(A));
            }

            int ACols = A.Cols;
            int ARows = A.Rows;
            double[] AData = A.data;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    double sum = 0.0;
                    int index1 = i*ACols;
                    int index2 = j*ACols;
                    for (int k = 0; k < ACols; ++k)
                    {
                        //sum += A[i, k] * A[j, k];
                        sum += AData[index1++]*AData[index2++];
                    }
                    this[i, j] = sum;
                }
            }
#endif
            return this;
        }

        /// <summary>
        /// Returns the product of this matrix and its transpose
        /// </summary>
        /// <returns></returns>
        public PositiveDefiniteMatrix Outer()
        {
            PositiveDefiniteMatrix result = new PositiveDefiniteMatrix(Rows, Rows);
            result.SetToOuter(this);
            return result;
        }

        /// <summary>
        /// Modify <c>this</c> to be the product A'*A.
        /// </summary>
        /// <param name="A">Matrix.  Cannot be <c>this</c>.</param>
        /// <remarks>
        /// <c>this</c> receives A'*A and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        /// <returns><c>this</c></returns>
        public Matrix SetToOuterTranspose(Matrix A)
        {
#if LAPACK
            this.SetToProduct(A.Transpose(), A);
#else
            if(ReferenceEquals(this.SourceArray, A.SourceArray))
            {
                throw new ArgumentException($"{nameof(A)} is the same object as this.");
            }
            if (rows != A.Cols || cols != A.Cols)
            {
                throw new ArgumentException("Output matrix is not compatible with the product", nameof(A));
            }

            int ACols = A.Cols;
            int ARows = A.Rows;
            double[] AData = A.data;
            if (ACols < 40)
            {
                for (int i = 0; i < ACols; ++i)
                {
                    for (int j = 0; j <= i; ++j)
                    {
                        double sum = 0.0;
                        // TM: this looks ugly but runs much faster
                        int index1 = i;
                        int index2 = j;
                        for (int k = 0; k < ARows; ++k)
                        {
                            //sum += A[k, i] * A[k, j];                        
                            sum += AData[index1]*AData[index2];
                            index1 += ACols;
                            index2 += ACols;
                        }
                        this[i, j] = sum;
                        this[j, i] = sum;
                    }
                }
            }
            else
            {
                // blocked version
                // this[i,j] = sum_k A[k,i]*A[k,j]
                for (int i = 0; i < ACols; ++i)
                {
                    for (int j = 0; j < ACols; ++j)
                    {
                        this[i, j] = 0.0;
                    }
                }
                int colblock = 20;
                int rowblock = 20;
                int colblocks = (ACols + colblock - 1)/colblock;
                int rowblocks = (ARows + rowblock - 1)/rowblock;
                for (int ib = 0; ib < colblocks; ib++)
                {
                    int imax = System.Math.Min((ib + 1)* colblock, ACols);
                    for (int jb = 0; jb < colblocks; jb++)
                    {
                        int jmax = System.Math.Min((jb + 1)* colblock, ACols);
                        for (int kb = 0; kb < rowblocks; kb++)
                        {
                            int kmax = System.Math.Min((kb + 1)* rowblock, ARows);
                            for (int i = ib*colblock; i < imax; ++i)
                            {
                                for (int j = jb*colblock; j < jmax; ++j)
                                {
                                    double sum = this[i, j];
                                    // TM: this looks ugly but runs much faster
                                    int kmin = kb*rowblock;
                                    int start = kmin*ACols;
                                    int index1 = i + start;
                                    int index2 = j + start;
                                    for (int k = kmin; k < kmax; ++k)
                                    {
                                        //sum += A[k, i] * A[k, j];                        
                                        sum += AData[index1]*AData[index2];
                                        index1 += ACols;
                                        index2 += ACols;
                                    }
                                    this[i, j] = sum;
                                }
                            }
                        }
                    }
                }
            }
#endif
            return this;
        }

        /// <summary>
        /// Returns the transpose of this matrix times this matrix
        /// </summary>
        /// <returns></returns>
        public PositiveDefiniteMatrix OuterTranspose()
        {
            PositiveDefiniteMatrix result = new PositiveDefiniteMatrix(Cols, Cols);
            result.SetToOuterTranspose(this);
            return result;
        }

        /// <summary>
        /// Matrix product.
        /// </summary>
        /// <param name="A">First matrix.</param>
        /// <param name="B">Second matrix.</param>
        /// <returns>A new matrix with their product.</returns>
        public static Matrix operator *(Matrix A, Matrix B)
        {
            Matrix C = new Matrix(A.Rows, B.Cols);
            C.SetToProduct(A, B);
            return (C);
        }

        /// <summary>
        /// Premultiplies a vector by a matrix (i.e. A*x).
        /// </summary>
        /// <param name="A">A matrix.</param>
        /// <param name="x">A vector.</param>
        /// <returns>A new vector with the product.</returns>
        public static Vector operator *(Matrix A, Vector x)
        {
            Vector b = Vector.Zero(A.Rows);
            b.SetToProduct(A, x);
            return b;
        }

        /// <summary>
        /// Postmultiplies a vector by a matrix (i.e. x*A).
        /// </summary>
        /// <param name="x">A vector.</param>
        /// <param name="A">A matrix.</param>
        /// <returns>The new vector with the product.</returns>
        public static Vector operator *(Vector x, Matrix A)
        {
            Vector b = Vector.Zero(A.Cols);
            b.SetToProduct(x, A);
            return b;
        }

        /// <summary>
        /// Returns the outer product of two vectors.
        /// </summary>
        /// <param name="a">First vector, of length <c>this.Rows</c>.</param>
        /// <param name="b">Second vector, of length <c>this.Cols</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToOuter(Vector a, Vector b)
        {
            if (this.Rows != a.Count) throw new ArgumentException($"this.Rows ({this.Rows}) != a.Count ({a.Count})");
            if (this.Cols != b.Count) throw new ArgumentException($"this.Cols ({this.Cols}) != b.Count ({b.Count})");
            for (int i = 0; i < a.Count; ++i)
            {
                for (int j = 0; j < b.Count; ++j)
                {
                    this[i, j] = a[i]*b[j];
                }
            }
            return (this);
        }

        /// <summary>
        /// Returns a matrix plus the scaled outer product of two vectors.
        /// </summary>
        /// <param name="m">A matrix with the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="scale"></param>
        /// <param name="a">A vector with <c>a.Count == this.Rows</c></param>
        /// <param name="b">A vector with <c>b.Count == this.Cols</c></param>
        /// <returns>this[i,j] = m[i,j] + scale*a[i]*b[j]</returns>
        public Matrix SetToSumWithOuter(Matrix m, double scale, Vector a, Vector b)
        {
            Assert.IsTrue(Rows == m.Rows);
            Assert.IsTrue(Cols == m.Cols);
            Assert.IsTrue(this.Rows == a.Count);
            Assert.IsTrue(this.Cols == b.Count);
            for (int i = 0; i < a.Count; ++i)
            {
                for (int j = 0; j < b.Count; ++j)
                {
                    this[i, j] = m[i, j] + scale*a[i]*b[j];
                }
            }
            return (this);
        }

        /// <summary>
        /// Evaluates the product x'Ax (where ' is transposition).
        /// </summary>
        /// <param name="x">A vector whose length equals Rows.</param>
        /// <remarks><c>this</c> must be a square matrix with Rows == x.Count.</remarks>
        /// <returns>The above product.</returns>
        public double QuadraticForm(Vector x)
        {
            if (rows != cols) throw new Exception("Matrix is rectangular.");
            Assert.IsTrue(Rows == x.Count);
            double result = 0.0;
            int n = x.Count;
            for (int i = 0; i < n; ++i)
            {
                if (x[i] == 0) continue;
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                {
                    sum += this[i, j]*x[j];
                }
                result += sum*x[i];
            }
            return result;
        }

        /// <summary>
        /// Evaluates the product x'Ay (where ' is transposition).
        /// </summary>
        /// <param name="x">A vector whose length equals Rows.</param>
        /// <param name="y">A vector whose length equals Cows.</param>
        /// <returns>The above product.</returns>
        public double QuadraticForm(Vector x, Vector y)
        {
            Assert.IsTrue(Rows == x.Count);
            Assert.IsTrue(Cols == y.Count);
            double result = 0.0;
            int n = x.Count;
            int m = y.Count;
            for (int i = 0; i < n; ++i)
            {
                if (x[i] == 0) continue;
                double sum = 0.0;
                for (int j = 0; j < m; ++j)
                {
                    sum += this[i, j]*y[j];
                }
                result += sum*x[i];
            }
            return result;
        }

        /// <summary>
        /// Sets this matrix to sum of two other matrices. Assumes compatible matrices.
        /// </summary>
        /// <param name="a">First matrix, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second matrix, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this[i,j] = a[i,j] + b[i,j]</c>.  <c>this</c> receives the sum, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToSum(Matrix a, Matrix b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int count = Count;
            double[] adata = a.data;
            double[] bdata = b.data;
            for (int i = 0; i < count; i++)
            {
                data[i] = adata[i] + bdata[i];
            }
            return (this);
        }

        /// <summary>
        /// The sum of two matrices with scale factors.
        /// </summary>
        /// <param name="aScale">A scale factor.</param>
        /// <param name="a">First matrix, which must have the same size as <c>this</c>.</param>
        /// <param name="bScale">A scale factor.</param>
        /// <param name="b">Second matrix, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this[i,j] = aScale*a[i,j] + bScale*b[i,j]</c>.</remarks>
        public Matrix SetToSum(double aScale, Matrix a, double bScale, Matrix b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            for (int i = 0; i < Count; ++i)
            {
                this[i] = aScale*a[i] + bScale*b[i];
            }
            return (this);
        }

        /// <summary>
        /// Returns the sum of two matrices. Assumes compatible matrices.
        /// </summary>
        /// <param name="a">First matrix.</param>
        /// <param name="b">Second matrix.</param>
        /// <returns>Their sum.</returns>
        public static Matrix operator +(Matrix a, Matrix b)
        {
            Matrix result = (Matrix) a.Clone();
            result.SetToSum(result, b);
            return result;
        }

        /// <summary>
        /// Sets this matrix to the difference of two matrices. Assumes compatible matrices.
        /// </summary>
        /// <param name="a">First matrix, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second matrix, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the difference, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public Matrix SetToDifference(Matrix a, Matrix b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            for (int i = 0; i < Count; ++i)
            {
                this[i] = a[i] - b[i];
            }
            return (this);
        }

        /// <summary>
        /// Sets this matrix to a - bScale*b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="bScale"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public Matrix SetToDifference(Matrix a, double bScale, Matrix b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            for (int i = 0; i < Count; ++i)
            {
                this[i] = a[i] - bScale*b[i];
            }
            return (this);
        }

        /// <summary>
        /// Returns the difference of two matrices. Assumes compatible matrices.
        /// </summary>
        /// <param name="a">First matrix.</param>
        /// <param name="b">Second matrix.</param>
        /// <returns>Their difference.</returns>
        public static Matrix operator -(Matrix a, Matrix b)
        {
            Matrix result = (Matrix) a.Clone();
            result.SetToDifference(result, b);
            return result;
        }

        /// <summary>
        /// Returns the negation of a matrix.
        /// </summary>
        /// <param name="a">The matrix to negate.</param>
        /// <returns>The negation -a.</returns>
        public static Matrix operator -(Matrix a)
        {
            return ((Matrix) a.Clone()).Scale(-1);
        }

        #endregion

        #region Equality

        /// <summary>
        /// Equality operator.
        /// </summary>
        /// <param name="a">First matrix.</param>
        /// <param name="b">Second Matrix.</param>
        /// <returns>True if the matrices have the same size and element values.</returns>
        public static bool operator ==(Matrix a, Matrix b)
        {
            if (Object.ReferenceEquals(a, b))
            {
                return (true);
            }
            if (Object.ReferenceEquals(a, null) ||
                Object.ReferenceEquals(b, null))
            {
                return (false);
            }

            if ((a.Rows != b.Rows) || (a.Cols != b.Cols))
            {
                return false;
            }

            for (int i = 0; i < a.Count; ++i)
            {
                if (a[i] != b[i])
                {
                    return (false);
                }
            }

            return (true);
        }

        /// <summary>
        /// Inequality operator.
        /// </summary>
        /// <param name="a">First matrix.</param>
        /// <param name="b">Second matrix.</param>
        /// <returns>True if matrices are not equal.</returns>
        public static bool operator !=(Matrix a, Matrix b)
        {
            return (!(a == b));
        }

        /// <summary>
        /// Object comparator.
        /// </summary>
        /// <param name="obj">An object - must be an IntMatrix.</param>
        /// <returns>True if objects are equal.</returns>
        /// <exclude/>
        public override bool Equals(object obj)
        {
            Matrix that = obj as Matrix;
            if (Object.ReferenceEquals(that, null))
            {
                return false;
            }
            return (this == that);
        }

        /// <summary>
        /// Hash code generator.
        /// </summary>
        /// <returns>The hash code for the instance.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            for (int i = 0; i < Count; i++)
                hash = Hash.Combine(hash, this[i]);
            return hash;
        }

        /// <summary>
        /// Checks that a given matrix is the same size as this matrix.
        /// Throws an exception if not with the given string
        /// </summary>
        /// <param name="that">The matrix to check</param>
        /// <param name="paramName"></param>
        /// <exclude/>
        protected void CheckCompatible(Matrix that, string paramName)
        {
            if (rows != that.Rows || cols != that.Cols)
            {
                throw new ArgumentException("Matrices have different size", paramName);
            }
        }

        internal const double Tolerance = 1e-4;

        /// <summary>
        /// Checks this matrix for symmetry
        /// </summary>
        /// <param name="paramName"></param>
        /// <exclude/>
        [Conditional("DEBUG")]
        public void CheckSymmetry(string paramName)
        {
            double error = SymmetryError();
            Assert.IsTrue(error < Tolerance, String.Format(
                "Matrix is not symmetric (SymmetryError = {0})", error));
        }

        /// <summary>
        /// Returns the maximum absolute difference between matrix elements.
        /// </summary>
        /// <param name="that">The second matrix.</param>
        /// <returns>max(abs(this[i,j] - that[i,j])).  
        /// Matching infinities or NaNs do not count.  
        /// If this and that are not the same size, returns infinity.</returns>
        /// <remarks>This routine is typically used instead of Equals, since Equals is susceptible to roundoff errors.
        /// </remarks>
        public double MaxDiff(Matrix that)
        {
            if ((rows != that.Rows) || (cols != that.Cols))
            {
                return Double.PositiveInfinity;
            }

            double max = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    double x = this[i, j];
                    double y = that[i, j];
                    bool xnan = Double.IsNaN(x);
                    bool ynan = Double.IsNaN(y);
                    if (xnan != ynan)
                    {
                        return Double.PositiveInfinity;
                    }
                    else if (x != y)
                    {
                        // catches infinities
                        double diff = System.Math.Abs(x - y);
                        if (diff > max) max = diff;
                    }
                }
            }
            return max;
        }

        /// <summary>
        /// Returns the maximum relative difference between matrix elements.
        /// </summary>
        /// <param name="that">The second matrix.</param>
        /// <param name="rel">An offset to avoid division by zero.</param>
        /// <returns>max(abs(this[i,j] - that[i,j])/(abs(this[i,j]) + rel)).  
        /// Matching infinities or NaNs do not count.  
        /// If this and that are not the same size, returns infinity.</returns>
        /// <remarks>This routine is typically used instead of Equals, since Equals is susceptible to roundoff errors.
        /// </remarks>
        public double MaxDiff(Matrix that, double rel)
        {
            if ((rows != that.Rows) || (cols != that.Cols))
            {
                return Double.PositiveInfinity;
            }

            double max = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    double x = this[i, j];
                    double y = that[i, j];
                    bool xnan = Double.IsNaN(x);
                    bool ynan = Double.IsNaN(y);
                    if (xnan != ynan)
                    {
                        return Double.PositiveInfinity;
                    }
                    else if (x != y)
                    {
                        // catches infinities
                        double diff = System.Math.Abs(x - y)/(System.Math.Abs(x) + rel);
                        if (diff > max) max = diff;
                    }
                }
            }
            return max;
        }

        #endregion

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return ToString("g4");
        }

        /// <summary>
        /// Convert the matrix to a string, using a specified number format
        /// </summary>
        /// <param name="format">The format string for each matrix entry</param>
        /// <returns>A string</returns>
        public string ToString(string format)
        {
            if (cols == 0) return String.Format("[{0}x{1} Matrix]", rows, cols);
            string[][] lines = new string[2*cols - 1][];
            for (int j = 0; j < cols; ++j)
            {
                if (j > 0)
                {
                    lines[2*j - 1] = new string[1];
                    lines[2*j - 1][0] = " ";
                }
                lines[2*j] = new string[rows];
                for (int i = 0; i < rows; ++i)
                {
                    lines[2*j][i] = this[i, j].ToString(format);
                }
            }
            return StringUtil.JoinColumns(lines);
        }

        /// <summary>
        /// Parse a string (in the format produced by ToString) to recover a Matrix
        /// </summary>
        /// <param name="s">The string to parse</param>
        /// <returns>A matrix</returns>
        public static Matrix Parse(string s)
        {
            string[] lines = s.Split(new string[] {Environment.NewLine}, StringSplitOptions.RemoveEmptyEntries);
            int rows = lines.Length;
            int cols = 0;
            Matrix result = null;
            for (int i = 0; i < rows; i++)
            {
                string line = lines[i];
                string[] fields = line.Split(new char[] {' '}, StringSplitOptions.RemoveEmptyEntries);
                if (i == 0)
                {
                    cols = fields.Length;
                    result = new Matrix(rows, cols);
                }
                else if (cols != fields.Length) throw new ArgumentException(String.Format("line {0} has {1} columns but line 0 has {2} columns", i, fields.Length, cols));
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = double.Parse(fields[j]);
                }
            }
            return result;
        }

        #region Linear algebra

        /// <summary>
        /// Tests for positive-definiteness.
        /// </summary>
        /// <returns>True if the matrix is positive-definite, i.e. all eigenvalues > 0.</returns>
        public bool IsPositiveDefinite()
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(rows, cols);
            return (L.SetToCholesky(this));
        }

        /// <summary>
        /// Trace of a square matrix.
        /// </summary>
        /// <returns>The trace.</returns>
        public double Trace()
        {
            if (rows != cols) throw new Exception("Matrix is rectangular.");
            double sum = 0.0;
            for (int i = 0; i < rows; ++i)
            {
                sum += this[i, i];
            }
            return (sum);
        }

        /// <summary>
        /// Determinant of a square matrix
        /// </summary>
        /// <returns>The determinant</returns>
        public double Determinant()
        {
            return (new LuDecomposition(this)).Determinant();
        }

        /// <summary>
        /// Return the inverse of this matrix - not implemented yet
        /// </summary>
        /// <returns></returns>
        public Matrix Inverse()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Inner product of matrices.
        /// </summary>
        /// <param name="A">A matrix with the same size as this.</param>
        /// <returns>sum_ij A[i,j]*this[i,j].</returns>
        public double Inner(Matrix A)
        {
            Matrix B = this;
            Assert.IsTrue(A.Cols == B.Cols);
            Assert.IsTrue(A.Rows == B.Rows);
            double sum = 0.0;
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = 0; j < A.Cols; j++)
                {
                    sum += A[i, j]*B[i, j];
                }
            }
            return sum;
        }

        /// <summary>
        /// Set this matrix to the Kronecker product of two matrices
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        public void SetToKronecker(Matrix A, Matrix B)
        {
            if (this.Rows != A.Rows * B.Rows)
                throw new ArgumentException("Rows (" + Rows + ") != A.Rows * B.Rows (" + A.Rows * B.Rows + ")");
            if (this.Cols != A.Cols * B.Cols)
                throw new ArgumentException("Cols (" + Cols + ") != A.Cols * B.Cols (" + A.Cols * B.Cols + ")");
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = 0; j < A.Cols; j++)
                {
                    for (int k = 0; k < B.Rows; k++)
                    {
                        for (int l = 0; l < B.Cols; l++)
                        {
                            this[i * B.Rows + k, j * B.Cols + l] = A[i, j] * B[k, l];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Kronecker product with another matrix.
        /// </summary>
        /// <param name="A"></param>
        /// <returns>The Kronecker product</returns>
        public Matrix Kronecker(Matrix A)
        {
            Matrix result = new Matrix(this.Rows * A.Rows, this.Cols * A.Cols);
            result.SetToKronecker(this, A);
            return result;
        }

        /// <summary>
        /// Magnus and Neudecker's commutation matrix for a given size matrix
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <returns>The commutation matrix of size rows*cols by rows*cols</returns>
        public static Matrix Commutation(int rows, int cols)
        {
            int dim = rows * cols;
            Matrix result = new Matrix(dim, dim);
            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    int row = i + j * rows;
                    int i2 = row / cols;
                    int j2 = row % cols;
                    int col = i2 + j2 * rows;
                    result[row, col] = 1;
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the solution to AX=B, where A is an upper triangular matrix and
        /// B is a matrix of right-hand sides.    It is equivalent to the left-division X = A\B.
        /// </summary>
        /// <param name="A">An upper triangular matrix with A.Rows == this.Rows.  Cannot be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side matrix B, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public Matrix PredivideBy(UpperTriangularMatrix A)
        {
#if LAPACK
            Lapack.PredivideBy(this, A);
#else
            if (ReferenceEquals(this.SourceArray, A.SourceArray))
            {
                throw new ArgumentException("A is the same object as this.");
            }
            Matrix B = this;
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == B.Rows);

            for (int j = 0; j < B.Cols; j++)
            {
#if false
    // Golub and van Loan, Alg 3.1.4 on p90
                for(k = A.Rows-1; k >= 0; k--) {
                    if(A[k,k] == 0) throw new MatrixSingularException( A );
                    if(B[k,j] != 0) {
                        B[k,j] /= A[k,k];
                        for(i=0;i<k;i++) {
                            B[i,j] -= B[k,j] * A[i,k];
                        }
                    }
                }
#else
                for (int i = A.Rows - 1; i >= 0; i--)
                {
                    double sum = B[i, j];
                    for (int k = i + 1; k < A.Rows; k++)
                    {
                        sum -= A[i, k]*B[k, j];
                    }
                    if (A[i, i] == 0) throw new MatrixSingularException(A);
                    B[i, j] = sum/A[i, i];
                }
#endif
            }
#endif
            return this;
        }

        /// <summary>
        /// Gets the solution to A'X=B, where A is a lower triangular matrix.
        /// Equivalent to the left-division X = A'\B.
        /// </summary>
        /// <param name="A">A lower triangular matrix with A.Cols == this.Rows.  Cannot be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side matrix B, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public Matrix PredivideByTranspose(LowerTriangularMatrix A)
        {
#if LAPACK
            Lapack.PredivideByTranspose(this, A);
#else
            if (ReferenceEquals(this.SourceArray, A.SourceArray))
            {
                throw new ArgumentException("A is the same object as this.");
            }
            Matrix B = this;
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == B.Rows);

            for (int j = 0; j < B.Cols; j++)
            {
                for (int i = A.Rows - 1; i >= 0; i--)
                {
                    double sum = B[i, j];
                    for (int k = i + 1; k < A.Rows; k++)
                    {
                        sum -= A[k, i]*B[k, j];
                    }
                    if (A[i, i] == 0) throw new MatrixSingularException(A);
                    B[i, j] = sum/A[i, i];
                }
            }
#endif
            return this;
        }

        /// <summary>
        /// Gets the solution to AX=B, where A is a lower triangular matrix.
        /// Equivalent to the left-division X = A\B.
        /// </summary>
        /// <param name="A">A lower triangular matrix with A.Rows == this.Rows.  Cannot be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side matrix B, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public Matrix PredivideBy(LowerTriangularMatrix A)
        {
#if LAPACK
            Lapack.PredivideBy(this, A);
#else
            if (ReferenceEquals(this.SourceArray, A.SourceArray))
            {
                throw new ArgumentException("A is the same object as this.");
            }
            Matrix B = this;
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == B.Rows);

            for (int j = 0; j < B.Cols; j++)
            {
                for (int k = 0; k < A.Rows; k++)
                {
                    if (System.Math.Abs(A[k, k]) < double.Epsilon)
                    {
                        throw new MatrixSingularException(A);
                    }
                    if (B[k, j] != 0)
                    {
                        double u = B[k, j]/A[k, k];
                        B[k, j] = u;
                        for (int i = k + 1; i < A.Rows; i++)
                        {
                            B[i, j] -= u*A[i, k];
                        }
                    }
                }
            }
#endif
            return this;
        }

        /// <summary>
        /// Premultiply this matrix by the inverse of the given positive definite matrix
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public Matrix PredivideBy(PositiveDefiniteMatrix A)
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(A.Rows, A.Cols);
            bool isPD = L.SetToCholesky(A);
            if (!isPD)
                throw new PositiveDefiniteMatrixException();
            PredivideBy(L);
            PredivideByTranspose(L);
            return this;
        }


        /// <summary>
        /// Solve Y = X*A
        /// </summary>
        /// <param name="Y">Portrait matrix</param>
        /// <param name="X">Portrait matrix</param>
        /// <returns>The smallest squared singular value of X.  This is useful for detecting an ill-conditioned problem.</returns>
        public double SetToLeastSquares(Matrix Y, Matrix X)
        {
            Matrix result = this;
            if (Y.Rows != X.Rows)
                throw new ArgumentException("Y.Rows != X.Rows");
            Matrix V = new Matrix(X.Cols, X.Cols);
            Matrix US = (Matrix)X.Clone();
            V.SetToRightSingularVectors(US);
            double minSingularValueSqr = double.PositiveInfinity;
            for (int k = 0; k < US.Cols; k++)
            {
                double singularValueSqr = 0;
                for (int j = 0; j < US.Rows; j++)
                {
                    singularValueSqr += US[j, k] * US[j, k];
                }
                minSingularValueSqr = System.Math.Min(minSingularValueSqr, singularValueSqr);
                if (singularValueSqr > 0)
                {
                    for (int j = 0; j < V.Cols; j++)
                    {
                        V[j, k] /= singularValueSqr;
                    }
                }
            }
            // pinv(X) = V*inv(S)*U'
            Matrix USt = US.Transpose();
            var UStY = USt * Y;
            result.SetToProduct(V, UStY);
            // iterate again on the residual for higher accuracy
            var residual = X * result;
            residual.SetToDifference(Y, residual);
            UStY.SetToProduct(USt, residual);
            result.SetToSum(result, V * UStY);
            return minSingularValueSqr;
        }

        /// <summary>
        /// Set the columns to the eigenvectors of a symmetric matrix, modifying the input matrix to contain its eigenvalues on the diagonal.
        /// The input matrix = this'*modified*this.
        /// </summary>
        /// <param name="symmetricMatrix">A symmetric matrix, modified on return to contain its eigenvalues on the diagonal</param>
        public void SetToEigenvectorsOfSymmetric(Matrix symmetricMatrix)
        {
            if (Rows != Cols) throw new ArgumentException("this.Rows (" + this.Rows + ") != this.Cols (" + this.Cols + ")");
            if (Rows != symmetricMatrix.Rows) throw new ArgumentException("this.Rows (" + this.Rows + ") != symmetricMatrix.Rows (" + symmetricMatrix.Rows + ")");
            if (symmetricMatrix.Rows != symmetricMatrix.Cols)
                throw new ArgumentException("input matrix is not symmetric (Rows (" + symmetricMatrix.Rows + ") != Cols (" + symmetricMatrix.Cols + "))");
            Matrix A = symmetricMatrix;
            int n = Rows;
            SetToIdentity();
            // Jacobi iteration algorithm
            // Reference: Golub and van Loan (1996)
            // each iteration is O(n^3)
            for (int iter = 0; iter < n; iter++)
            {
                double offdiag = 0.0;
                for (int p = 0; p < n - 1; p++)
                {
                    for (int q = p + 1; q < n; q++)
                    {
                        if (A[p, q] == 0) continue;
                        offdiag += A[p, q]*A[p, q];
                        double c, s;
                        sym_schur2(A, p, q, out c, out s);
                        // V = J'*A*J
                        for (int i = 0; i < n; i++)
                        {
                            if (i == q) continue;
                            double x = A[i, p];
                            double y = A[i, q];
                            if (i == p)
                            {
                                // J'*[x y; y z]*J
                                double z = A[q, q];
                                double cc = c*c;
                                double ss = s*s;
                                double sc = s*c;
                                double scy2 = sc*y*2;
                                A[p, p] = cc*x - scy2 + ss*z; // c*c*x - s*c*y - s*c*y + s*s*z
                                A[p, q] = sc*(x - z) + (cc - ss)*y; // s*c*x + c*c*y - s*s*y - s*c*z
                                A[q, p] = A[p, q];
                                A[q, q] = ss*x + scy2 + cc*z; // s*s*x + s*c*y + s*c*y + c*c*z
                            }
                            else
                            {
                                A[i, p] = c*x - s*y;
                                A[p, i] = A[i, p];
                                A[i, q] = s*x + c*y;
                                A[q, i] = A[i, q];
                            }
                        }
                        // V = V*J
                        for (int i = 0; i < n; i++)
                        {
                            double x = this[i, p];
                            double y = this[i, q];
                            this[i, p] = c*x - s*y;
                            this[i, q] = s*x + c*y;
                        }
                    }
                }
                if (offdiag == 0) break;
            }
        }

        /// <summary>
        /// Set the columns to the right singular vectors of a matrix, modifying the input matrix to contain its left singular vectors scaled by the singular values.
        /// </summary>
        /// <param name="A">A portrait matrix with the same number of columns as <c>this</c>, modified on return to contain its left singular vectors (as columns) scaled by the singular values</param>
        /// <remarks><c>this</c> must be a square matrix.</remarks>
        public void SetToRightSingularVectors(Matrix A)
        {
            if (Cols != A.Cols)
                throw new ArgumentException("this.Cols (" + this.Cols + ") != A.Cols (" + A.Cols + ")");
            if (Rows != A.Cols)
                throw new ArgumentException("this.Rows (" + this.Rows + ") != A.Cols (" + A.Cols + ")");
            if(A.Rows < A.Cols)
                throw new ArgumentException("A.Rows (" + A.Rows + ") < A.Cols (" + A.Cols + ")");
            // One-sided Jacobi iteration algorithm
            // Reference: James Demmel, Kresimir Veselic (1992)
            // "Jacobi's method is more accurate than QR"
            SetToIdentity();
            var submatrix = new Matrix(2, 2);
            for (int iter = 0; iter < 100*Rows; iter++)
            {
                double maxOffDiag = 0;
                for (int p = 0; p < A.Cols - 1; p++)
                {
                    for (int q = p + 1; q < A.Cols; q++)
                    {
                        // form the (p,q) submatrix of A'A
                        submatrix.SetAllElementsTo(0);
                        for (int i = 0; i < A.Rows; i++)
                        {
                            double Aip = A[i, p];
                            double Aiq = A[i, q];
                            submatrix[0, 0] += Aip * Aip;
                            submatrix[1, 1] += Aiq * Aiq;
                            submatrix[0, 1] += Aip * Aiq;
                        }
                        if (submatrix[0, 1] == 0)
                            continue;
                        submatrix[1, 0] = submatrix[0, 1];
                        double offdiag;
                        if (submatrix[0, 0] == 0)
                        {
                            // special method that avoids numerical underflow
                            double maxAbs = 0;
                            for (int i = 0; i < A.Rows; i++)
                            {
                                maxAbs = System.Math.Max(maxAbs, System.Math.Abs(A[i, p]));
                            }
                            double sum01 = 0;
                            double sum00 = 0;
                            for (int i = 0; i < A.Rows; i++)
                            {
                                double Aip = A[i, p];
                                double Aiq = A[i, q];
                                double Aips = Aip / maxAbs;
                                sum00 += Aips * Aips;
                                sum01 += Aiq * Aips;
                            }
                            offdiag = (sum01*sum01) / (sum00 * submatrix[1, 1]);
                        }
                        else if (submatrix[1, 1] == 0)
                        {
                            // special method that avoids numerical underflow
                            double maxAbs = 0;
                            for (int i = 0; i < A.Rows; i++)
                            {
                                maxAbs = System.Math.Max(maxAbs, System.Math.Abs(A[i, q]));
                            }
                            double sum01 = 0;
                            double sum11 = 0;
                            for (int i = 0; i < A.Rows; i++)
                            {
                                double Aip = A[i, p];
                                double Aiq = A[i, q];
                                double Aiqs = Aiq / maxAbs;
                                sum11 += Aiqs * Aiqs;
                                sum01 += Aip * Aiqs;
                            }
                            offdiag = (sum01*sum01) / (sum11 * submatrix[0, 0]);
                        }
                        else
                        {
                            offdiag = System.Math.Abs(submatrix[0, 1] * submatrix[0, 1]) / (submatrix[0, 0] * submatrix[1, 1]);
                            //offdiag = Math.Abs(submatrix[0, 1]) / (Math.Sqrt(submatrix[0, 0]) * Math.Sqrt(submatrix[1, 1]));
                        }
                        maxOffDiag = System.Math.Max(maxOffDiag, offdiag);
                        double c, s;
                        sym_schur2(submatrix, 0, 1, out c, out s);
                        // update columns p and q of A
                        for (int i = 0; i < A.Rows; i++)
                        {
                            double x = A[i, p];
                            double y = A[i, q];
                            A[i, p] = c * x - s * y;
                            A[i, q] = s * x + c * y;
                        }
                        // update the right singular vectors
                        for (int i = 0; i < this.Rows; i++)
                        {
                            double x = this[i, p];
                            double y = this[i, q];
                            this[i, p] = c * x - s * y;
                            this[i, q] = s * x + c * y;
                        }
                    }
                }
                //Debug.WriteLine("maxOffDiag = {0}", maxOffDiag);
                if (maxOffDiag < 1e-30)
                    break;
            }
        }

        /// <summary>
        /// Symmetric Schur decomposition of a 2x2 submatrix
        /// </summary>
        /// <param name="A">A matrix with at least 2 rows and 2 columns</param>
        /// <param name="p">Index of the first row of the submatrix</param>
        /// <param name="q">Index of the second row of the submatrix</param>
        /// <param name="c">Upper left entry of the 2x2 factor</param>
        /// <param name="s">Upper right entry of the 2x2 factor</param>
        private void sym_schur2(Matrix A, int p, int q, out double c, out double s)
        {
            // should never be called with A[p,q] == 0
            if (A[p, q] != 0)
            {
                double tau = (A[q, q] - A[p, p]) / (2 * A[p, q]);
                double t;
                if (System.Math.Abs(tau) > 1e9)
                {
                    // In double precision, tau * tau + 1 == tau * tau, therefore sqrt(1 + tau * tau) == abs(tau)
                    t = 0.5 / tau;
                }
                else
                {
                    if (tau >= 0)
                        t = 1 / (tau + System.Math.Sqrt(1 + tau * tau));
                    else
                        t = 1 / (tau - System.Math.Sqrt(1 + tau * tau));
                }
                c = 1 / System.Math.Sqrt(1 + t * t);
                s = t * c;
            }
            else
            {
                c = 1;
                s = 0;
            }
        }

        /// <summary>
        /// Compute the eigenvalues of a square matrix, destroying the contents of the matrix.
        /// </summary>
        /// <param name="eigenvaluesReal">On output, the real part of the eigenvalues.</param>
        /// <param name="eigenvaluesImag">On output, the imaginary part of the eigenvalues.</param>
        public void EigenvaluesInPlace(double[] eigenvaluesReal, double[] eigenvaluesImag)
        {
#if LAPACK
            Lapack.EigenvaluesInPlace(this, eigenvaluesReal, eigenvaluesImag);
#else
            EigenvaluesInPlace(this, eigenvaluesReal, eigenvaluesImag);
#endif
        }

        // Apply the implicit QR algorithm from (Golub and van Loan, 1996)
        // with modifications suggested by:
        // "Numerical Methods and Software for General and Structured Eigenvalue Problems"
        // D. Kressner, PhD thesis 2004
        // http://www8.cs.umu.se/~kressner/kressner.pdf
        private static void EigenvaluesInPlace(Matrix A, double[] eigenvaluesReal, double[] eigenvaluesImag)
        {
            int n = A.Rows;
            // Reduce to upper Hessenberg form, overwriting A
            // This makes the lower triangle zero (except for the subdiagonal), without changing the eigenvalues
            for (int k = 0; k < n - 2; k++)
            {
                Vector x = Vector.Zero(n - k - 1);
                for (int i = 0; i < x.Count; i++)
                {
                    x[i] = A[i + k + 1, k];
                }
                Vector v = Vector.Zero(x.Count);
                double beta = Householder(x, v);
                HouseHolderLeft(A, k + 1, k, n - k, beta, v);
                HouseHolderRight(A, 0, n, k + 1, beta, v);
            }
            //Console.WriteLine("Hessenberg form:");
            //Console.WriteLine(A);
            //PermuteForEigenvalues(A);
            //Console.WriteLine("permuted:");
            //Console.WriteLine(A);
            Balance(A);
            //Console.WriteLine("after balancing:");
            //Console.WriteLine(A);
            // Iterate Fancis QR steps to zero out the subdiagonal elements, 
            // eventually arriving at an upper quasi-triangular matrix.
            // upper quasi-triangular = upper triangular except for 2x2 diagonal blocks
            for (int iter = 0; ;iter++)
            {
                if (iter >= 30*n)
                    throw new Exception(string.Format("EigenvaluesInPlace exceeded {0} iterations", iter));
                // Set small subdiagonal elements to zero
                double tol = 1e-15;
                int nonZeroStart = n;
                int nonZeroCount = 0;
                bool foundQRBlock = false;
                bool finishedQRBlock = false;
                for (int i = 0; i < n - 1; i++)
                {
                    if (System.Math.Abs(A[i + 1, i]) <= tol * (System.Math.Abs(A[i, i]) + System.Math.Abs(A[i + 1, i + 1])))
                    {
                        A[i + 1, i] = 0;
                        if (!foundQRBlock)
                            nonZeroCount = 0;
                        else
                            finishedQRBlock = true;
                    }
                    else if (!finishedQRBlock)
                    {
                        nonZeroCount++;
                        if (nonZeroCount > 1)
                            foundQRBlock = true; // found 2 consecutive nonzeros
                        else
                            nonZeroStart = i;
                    }
                }
                if (!foundQRBlock)
                    break;
                nonZeroCount++;
                //Console.WriteLine("nonZeroCount = {0}", nonZeroCount);
                Matrix B = new Matrix(nonZeroCount, nonZeroCount);
                B.SetToSubmatrix(A, nonZeroStart, nonZeroStart);
                FrancisQR(B, iter);
                A.SetSubmatrix(nonZeroStart, nonZeroStart, B);
                //Console.WriteLine("after QR:");
                //Console.WriteLine(A);
            }
            // Compute eigenvalues from the 1x1 and 2x2 diagonal blocks
            int start = 0;
            for (int i = 0; i < n - 1; i++)
            {
                if (A[i + 1, i] == 0)
                {
                    eigenvaluesReal[start] = A[i, i];
                    eigenvaluesImag[start] = 0;
                    start++;
                }
                else
                {
                    Eigenvalues2x2(A[i, i], A[i, i + 1], A[i + 1, i], A[i + 1, i + 1], eigenvaluesReal, eigenvaluesImag, start);
                    start += 2;
                    i++;
                }
            }
            if (start < n)
            {
                Assert.IsTrue(start == n - 1);
                eigenvaluesReal[start] = A[start, start];
                eigenvaluesImag[start] = 0;
            }
        }

        // Perform a Francis QR step (Algorithm 7.5.1 of Golub and van Loan, 1996)
        private static void FrancisQR(Matrix A, int iter)
        {
            int n = A.Rows;
            Assert.IsTrue(n >= 3);
            int m = n - 1;
            // compute the eigenvalues of the trailing 2x2 principal submatrix of A
            double A00 = A[m-1,m-1];
            double A01 = A[m-1,m];
            double A10 = A[m,m-1];
            double A11 = A[m,m];
            if(iter == 10) {
                // Wilkinson's exceptional shift
                const double DAT1 = 3.0 / 4.0, DAT2 = -0.4375;
                double S = System.Math.Abs(A10) + System.Math.Abs(A[m - 1, m-2]);
                A00 = DAT1*S + A11;
                A01 = DAT2*S;
                A10 = S;
                A11 = A00;
            }
            double s, t;
            s = A00 + A11;
            t = A00 * A11 - A01 * A10;
            if (n == m)
            {
                double tr = 0.5 * (A00 + A11);
                double det = (A00 - tr) * (A11 - tr) - A01 * A10;
                double rtdisc = System.Math.Sqrt(System.Math.Abs(det));
                double rt1r, rt2r, rt1i, rt2i;
                if (det < 0)
                {
                    rt1r = tr;
                    rt2r = rt1r;
                    rt1i = rtdisc;
                    rt2i = -rtdisc;
                }
                else
                {
                    // use Day's shift strategy
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if (System.Math.Abs(rt1r - A11) < System.Math.Abs(rt2r - A11))
                    {
                        rt2r = rt1r;
                    }
                    else
                    {
                        rt1r = rt2r;
                    }
                    rt1i = 0;
                    rt2i = 0;
                }
                s = rt1r + rt2r;
                t = rt1r * rt2r - rt1i * rt2i;
            }
            // s and t implicitly define the double shift
            double x = A[0, 0] * A[0, 0] + A[0, 1] * A[1, 0] - s * A[0, 0] + t;
            //double x = A[1, 0] * A[0, 1] + (A[0, 0] - rt1r) * (A[0, 0] - rt2r) - rt1i * rt2i;
            double y = A[1, 0] * (A[0, 0] + A[1, 1] - s);
            double z = A[1, 0] * A[2, 1];
            Vector x3 = Vector.Zero(3);
            Vector v3 = Vector.Zero(3);
            for (int k = 0; k <= n - 3; k++)
            {
                x3[0] = x;
                x3[1] = y;
                x3[2] = z;
                double beta = Householder(x3, v3);
                //if (k == 0)
                //    Console.WriteLine("beta = {0}, v3 = {1}", beta, v3);
                //if (k == 0 && Math.Abs(beta) < 1e-16)
                //    throw new Exception("beta = "+beta);
                int q = System.Math.Max(0, k-1);
                HouseHolderLeft(A, k, q, n - q, beta, v3);
                int r = System.Math.Min(k + 4, n);
                HouseHolderRight(A, 0, r, k, beta, v3);
                x = A[k + 1, k];
                y = A[k + 2, k];
                if (k < n - 3)
                    z = A[k + 3, k];
            }
            Vector x2 = Vector.Zero(2);
            x2[0] = x;
            x2[1] = y;
            Vector v2 = Vector.Zero(2);
            double beta2 = Householder(x2, v2);
            HouseHolderLeft(A, n - 2, n - 3, 3, beta2, v2);
            HouseHolderRight(A, 0, n, n - 2, beta2, v2);
        }

        // A = (I-beta*v*v')*A
        private static void HouseHolderLeft(Matrix A, int startRow, int startCol, int countCol, double beta, Vector v)
        {
            int endCol = startCol + countCol;
            for (int j = startCol; j < endCol; j++)
            {
                double sum = 0;
                for (int i = 0; i < v.Count; i++)
                {
                    sum += v[i] * A[startRow + i, j];
                }
                double bvA = sum * beta;
                for (int i = 0; i < v.Count; i++)
                {
                    A[startRow + i, j] -= v[i] * bvA;
                }
            }
        }

        // A = A*(I-beta*v*v')
        private static void HouseHolderRight(Matrix A, int startRow, int countRow, int startCol, double beta, Vector v)
        {
            int endRow = startRow + countRow;
            for (int i = startRow; i < endRow; i++)
            {
                double sum = 0;
                for (int j = 0; j < v.Count; j++)
                {
                    sum += v[j] * A[i, startCol + j];
                }
                double Abv = sum * beta;
                for (int j = 0; j < v.Count; j++)
                {
                    A[i, startCol + j] -= Abv * v[j];
                }
            }
        }

        // Compute a Householder vector (Algorithm 5.1.1 of Golub and van Loan, 1996)
        // so that the Householder matrix P = I-beta*vv' satisfies Px = norm(x)*e1
        private static double Householder(Vector x, Vector v)
        {
            v.SetTo(x);
            v[0] = 0;
            double sigma = v.Inner(v);
            if (sigma == 0)
            {
                v[0] = 1;
                return sigma;
            }
            else
            {
                double mu = System.Math.Sqrt(x[0] * x[0] + sigma);
                double v1;
                if (x[0] <= 0)
                    v1 = x[0] - mu;
                else
                    v1 = -sigma / (x[0] + mu);
                double v1sqr = v1 * v1;
                v.Scale(1.0 / v1);
                v[0] = 1;
                return 2 * v1sqr / (sigma + v1sqr);
            }
        }

        // Permute the rows and columns of A to help recover eigenvalues, without changing the eigenvalues.
        // Uses the Parlett-Reinsch algorithm as described by Kressner (2004).
        private static void PermuteForEigenvalues(Matrix A)
        {
            int n = A.Rows;
            int imin = 0;
            int imax = n;
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int i = imin; i < imax; i++)
                {
                    bool allzero = true;
                    for (int j = imin; j < imax; j++)
                    {
                        if (j == i)
                            continue;
                        if (A[i, j] != 0)
                            allzero = false;
                    }
                    if (allzero)
                    {
                        // swap rows/cols i and imax-1
                        SwapRows(A, i, imax - 1);
                        SwapCols(A, i, imax - 1);
                        imax--;
                        changed = true;
                        break;
                    }
                }
                for (int i = imin; i < imax; i++)
                {
                    bool allzero = true;
                    for (int j = imin; j < imax; j++)
                    {
                        if (j == i)
                            continue;
                        if (A[j, i] != 0)
                            allzero = false;
                    }
                    if (allzero)
                    {
                        // swap rows/cols i and imin
                        SwapRows(A, i, imin);
                        SwapCols(A, i, imin);
                        imin++;
                        changed = true;
                        break;
                    }
                }
            }
        }

        private static void SwapRows(Matrix A, int i, int k)
        {
            int m = A.Cols;
            for (int j = 0; j < m; j++)
            {
                double temp = A[i, j];
                A[i, j] = A[k, j];
                A[k, j] = temp;
            }
        }

        private static void SwapCols(Matrix A, int i, int k)
        {
            int n = A.Rows;
            for (int j = 0; j < n; j++)
            {
                double temp = A[j, i];
                A[j, i] = A[j, k];
                A[j, k] = temp;
            }
        }

        // Scale the rows and columns of A to help recover eigenvalues, without changing the eigenvalues.
        // Uses the Parlett-Reinsch algorithm as described by Kressner (2004).
        private static void Balance(Matrix A)
        {
            const double beta = 2;
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int j = 0; j < A.Cols; j++)
                {
                    double c = 0, r = 0;
                    for (int i = 0; i < A.Rows; i++)
                    {
                        if (i == j)
                            continue;
                        c += System.Math.Abs(A[i, j]);
                        r += System.Math.Abs(A[j, i]);
                    }
                    if (c == 0 || r == 0)
                        continue;
                    double sum = c + r;
                    double scale = 1;
                    while (c < r / beta)
                    {
                        c *= beta;
                        r /= beta;
                        scale *= beta;
                    }
                    while (c > r * beta)
                    {
                        c /= beta;
                        r *= beta;
                        scale /= beta;
                    }
                    if (c + r < 0.95 * sum)
                    {
                        changed = true;
                        for (int i = 0; i < A.Rows; i++)
                        {
                            A[i, j] *= scale;
                            A[j, i] /= scale;
                        }
                    }
                }
            }
        }

        // Compute the eigenvalues of a 2x2 matrix
        private static void Eigenvalues2x2(double a, double b, double c, double d,
            double[] eigenvaluesReal, double[] eigenvaluesImag, int start)
        {
            // [a b; c d] has characteristic polynomial (a-x)(d-x) - bc = x^2 -(a+d)x + ad-bc
            // roots are (a+d)/2 +/- sqrt((a-d)^2/4 + bc)
            if (b == 0 || c == 0)
            {
                eigenvaluesReal[start] = a;
                eigenvaluesReal[start + 1] = d;
                eigenvaluesImag[start] = 0;
                eigenvaluesImag[start + 1] = 0;
                return;
            }
            double p = 0.5 * (a - d);
            double bc = b * c;
            double z = p * p + bc;
            if (z >= 0)
            {
                // real eigenvalues
                double r1, r2;
                if (p > 0)
                {
                    double y = p + System.Math.Sqrt(z);
                    r1 = d + y;
                    r2 = d - bc / y;
                }
                else
                {
                    double y = System.Math.Sqrt(z) - p;
                    r1 = a + y;
                    r2 = a - bc / y;
                }
                eigenvaluesReal[start] = r1;
                eigenvaluesReal[start + 1] = r2;
                eigenvaluesImag[start] = 0;
                eigenvaluesImag[start + 1] = 0;
            }
            else
            {
                // complex eigenvalues
                double r = d + p;
                eigenvaluesReal[start] = r;
                eigenvaluesReal[start + 1] = r;
                double y = System.Math.Sqrt(-z);
                eigenvaluesImag[start] = y;
                eigenvaluesImag[start + 1] = -y;
            }
        }
        #endregion

        /// <summary>
        /// Returns the specified column as an array of doubles
        /// </summary>
        /// <param name="col"></param>
        /// <returns></returns>
        public double[] GetColumn(int col)
        {
            double[] result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = this[i, col];
            }
            return result;
        }

        /// <summary>
        /// Returns the specified row as an array of doubles
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public double[] GetRow(int row)
        {
            double[] result = new double[cols];
            for (int i = 0; i < cols; i++)
            {
                result[i] = this[row, i];
            }
            return result;
        }
    }

    /// <summary>
    /// Exception thrown when a singular matrix is encountered.
    /// </summary>
    [Serializable]
    public class MatrixSingularException : Exception
    {
        private Matrix offender;

        /// <summary>
        /// Initializes a new instance of the <see cref="MatrixSingularException"/> class.
        /// </summary>
        public MatrixSingularException()
        {
        }

        /// <summary>Gets the singular matrix that caused the exception.</summary>
        public Matrix Offender
        {
            get { return (offender); }
        }

        /// <summary>
        /// Construct the exception.
        /// </summary>
        /// <param name="m">The offending matrix.</param>
        public MatrixSingularException(Matrix m)
            : base("The matrix is singular")
        {
            offender = m;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatrixSingularException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public MatrixSingularException(string message, Exception inner)
            : base(message, inner)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatrixSingularException"/> class.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected MatrixSingularException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}