// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using System;
using System.Runtime.Serialization;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Math
{

    /// <summary>
    /// Defines sparsity settings for vectors. The sparsity handling has been designed to
    /// deal with large dimensional distributions such as Discrete and Dirichlet distributions.
    /// </summary>
    [Serializable]
    [DataContract]
    public class Sparsity
    {
        private static Sparsity sparse = new Sparsity {Storage = StorageType.Sparse};

        /// <summary>
        /// Static read-only property giving Sparsity settings for sparse, exact vectors.
        /// </summary>
        /// <remarks>A vector class which uses this sparsity will maintain a common background
        /// value. It is expected that only a small percentage of values will differ from this
        /// common value, and this enables the vector class to be efficient computationally and
        /// memory-wise. <see cref="SparseVector"/> uses this specification.</remarks>
        public static Sparsity Sparse
        {
            get { return sparse; }
        }

        private static Sparsity piecewise = new Sparsity {Storage = StorageType.Piecewise};

        /// <summary>
        /// Static read-only property giving Sparsity settings for piecewise, exact vectors.
        /// </summary>
        /// <remarks>A vector class which uses this sparsity is expected to have ranges of elements
        /// that have the same values. <see cref="PiecewiseVector"/> uses this specification.
        /// </remarks>
        public static Sparsity Piecewise
        {
            get { return piecewise; }
        }

        private static Sparsity dense = new Sparsity {Storage = StorageType.Dense};

        /// <summary>
        /// Static read-only property giving Sparsity settings for dense vectors.
        /// </summary>
        /// <remarks> <see cref="DenseVector"/> uses this specification.</remarks>
        public static Sparsity Dense
        {
            get { return dense; }
        }

        /// <summary>
        /// Static read-only property giving Sparsity settings for approximate vector with the given tolerance.
        /// </summary>
        /// <param name="tolerance">At what tolerance are vector values considered equal to the common value.</param>
        /// <returns></returns>
        /// <remarks><see cref="ApproximateSparseVector"/> supports this specification.</remarks>
        public static Sparsity ApproximateWithTolerance(double tolerance)
        {
            return new Sparsity {Storage = StorageType.Sparse, Tolerance = tolerance};
        }

        /// <summary>
        /// Static method to create a general sparsity specification
        /// </summary>
        /// <param name="storage">The storage method the vector should use</param>
        /// <param name="tolerance">The tolerance at which vector element values are considered equal to the common value.</param>
        /// <param name="countTolerance">The maximum allowed count of vector elements not set to the common value.</param>
        /// <returns></returns>
        [Construction("Storage", "Tolerance", "CountTolerance")]
        public static Sparsity FromSpec(StorageType storage, double tolerance, int countTolerance)
        {
            if (storage != StorageType.Sparse)
            {
                tolerance = 0.0;
                countTolerance = 0;
            }
            return new Sparsity
                {
                    Storage = storage,
                    Tolerance = tolerance,
                    CountTolerance = countTolerance
                };
        }

        [DataMember]
        private StorageType storage;

        /// <summary>
        /// The storage method used by this vector
        /// </summary>
        public StorageType Storage
        {
            get { return storage; }
            private set { storage = value; }
        }

        /// <summary>
        /// True if is the vector is dense
        /// </summary>
        public bool IsDense
        {
            get { return storage == StorageType.Dense; }
        }

        /// <summary>
        /// True if the vector is sparse (exact or approximate)
        /// </summary>
        public bool IsSparse
        {
            get { return storage == StorageType.Sparse; }
        }

        /// <summary>
        /// True if is the vector is piecewise
        /// </summary>
        public bool IsPiecewise
        {
            get { return storage == StorageType.Piecewise; }
        }

        /// <summary>
        /// True if the sparsity is approximate
        /// </summary>
        public bool IsApproximate
        {
            get { return (Tolerance > 0.0) || (CountTolerance > 0); }
        }

        /// <summary>
        /// True if the sparsity is exact
        /// </summary>
        public bool IsExact
        {
            get { return !IsApproximate; }
        }

        /// <summary>
        /// The tolerance at which vector element values are considered equal to the common value.
        /// </summary>
        [DataMember]
        public double Tolerance { get; private set; }

        /// <summary>
        /// The maximum allowed count of vector elements not set to the common value.
        /// </summary>
        [DataMember]
        public int CountTolerance { get; private set; }

        private Sparsity()
        {
        }

        /// <summary>
        /// Creates a vector of all zeros with these sparsity settings
        /// </summary>
        /// <param name="count">Size of vector</param>
        /// <returns></returns>
        internal Vector CreateZeroVector(int count)
        {
            if (IsDense)
                return DenseVector.Zero(count);
            else if (IsPiecewise)
                return PiecewiseVector.Zero(count);
            else if (IsExact)
                return SparseVector.Zero(count);
            else
                return ApproximateSparseVector.Zero(count, this);
        }

        /// <summary>
        /// Creates a constant vector with these sparsity settings
        /// </summary>
        /// <param name="count">Size of vector</param>
        /// <param name="value">Constant value</param>
        /// <returns></returns>
        internal Vector CreateConstantVector(int count, double value)
        {
            if (IsDense)
                return DenseVector.Constant(count, value);
            else if (IsPiecewise)
                return PiecewiseVector.Constant(count, value);
            else if (IsExact)
                return SparseVector.Constant(count, value);
            else
                return ApproximateSparseVector.Constant(count, value, this);
        }

        /// <summary>
        /// Instance description
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            if (IsDense) return "Sparsity.Dense";
            else if (IsPiecewise) return "Sparsity.Piecewise";
            else if (IsExact) return "Sparsity.Sparse";
            else return "Sparsity.ApproximateWithTolerance(" + Tolerance + ")";
        }

        /// <summary>
        /// Determines the equality of this instance with another
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            Sparsity that = obj as Sparsity;
            if (Object.ReferenceEquals(that, null))
                return false;
            if (Object.ReferenceEquals(this, that))
                return true;
            if (this.Storage != that.Storage ||
                this.Tolerance != that.Tolerance ||
                this.CountTolerance != that.CountTolerance)
                return false;
            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, Storage.GetHashCode());
            hash = Hash.Combine(hash, Tolerance);
            hash = Hash.Combine(hash, CountTolerance);
            return hash;
        }

        /// <summary>
        /// Equality operator.
        /// </summary>
        /// <param name="a">First sparsity.</param>
        /// <param name="b">Second sparsity.</param>
        /// <returns>True if the sparsity specifications are equal.</returns>
        public static bool operator ==(Sparsity a, Sparsity b)
        {
            if (Object.ReferenceEquals(a, b))
            {
                return (true);
            }
            if (Object.ReferenceEquals(a, null) || Object.ReferenceEquals(b, null))
                return (false);

            return a.Equals(b);
        }

        /// <summary>
        /// Inequality operator.
        /// </summary>
        /// <param name="a">First sparsity specification.</param>
        /// <param name="b">Second sparsity specification.</param>
        /// <returns>True if sparsity specifications are not equal.</returns>
        public static bool operator !=(Sparsity a, Sparsity b)
        {
            return (!(a == b));
        }
    }

    /// <summary>
    /// The type of storage used in a vector, which is 
    /// specified as part of the Sparsity class.
    /// </summary>
    public enum StorageType
    {
        /// <summary>
        /// The vector is stored as a dense array with memory allocated
        /// for each element.
        /// </summary>
        Dense,

        /// <summary>
        /// The vector is stored as a sparse array with memory allocated
        /// only for elements that do not have a particular common value.
        /// </summary>
        Sparse,

        /// <summary>
        /// The vector is stored as a set of pieces with each piece
        /// having a constant value, and all elements not in any piece 
        /// having a particular common value.
        /// </summary>
        Piecewise
    }

    /// <summary>
    /// Base class for vectors. <see cref="DenseVector"/>, <see cref="SparseVector"/>, and <see cref="ApproximateSparseVector"/>
    /// all inherit from this base class.
    /// </summary>
    /// <remarks>This class includes factory methods for instantiating Vectors of different <see cref="Sparsity"/>
    /// specifications. Beyond this initial construction, application code does not need to know the
    /// the sparsity, and most operations can be done via this general base class which will handle sparsity correctly.</remarks>
    [Serializable]
    [DataContract]
    public abstract class Vector : IList<double>, IReadOnlyList<double>, SettableTo<Vector>, ICloneable,
                                   CanSetAllElementsTo<double>, SettableToPower<Vector>, SettableToProduct<Vector>,
                                   SettableToWeightedSum<Vector>
    {
        [DataMember]
        private Sparsity sparsity = Sparsity.Dense;

        /// <summary>
        /// The <see cref="Sparsity"/> specification of this vector.
        /// </summary>
        public Sparsity Sparsity
        {
            get { return sparsity; }
            protected set { sparsity = value; }
        }

        /// <summary>
        /// True if this vector is dense
        /// </summary>
        public bool IsDense
        {
            get { return sparsity.IsDense; }
        }

        /// <summary>
        /// True if is this vector is sparse (exact or approximate)
        /// </summary>
        public bool IsSparse
        {
            get { return sparsity.IsSparse; }
        }

        /// <summary>
        /// True if this vector is approximate (sparse only)
        /// </summary>
        public bool IsApproximate
        {
            get { return sparsity.IsApproximate; }
        }

        /// <summary>
        /// True if this vector is exact (dense or sparse)
        /// </summary>
        public bool IsExact
        {
            get { return sparsity.IsExact; }
        }

        #region Factory methods and constructors

        /// <summary>
        /// Creates a dense vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <returns></returns>
        public static Vector Zero(int count)
        {
            return DenseVector.Zero(count);
        }

        /// <summary>
        /// Creates a vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification.</param>
        /// <returns></returns>
        public static Vector Zero(int count, Sparsity sparsity)
        {
            return sparsity.CreateZeroVector(count);
        }

        /// <summary>
        /// Create a dense vector of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <returns></returns>
        public static Vector Constant(int count, double value)
        {
            return DenseVector.Constant(count, value);
        }

        /// <summary>
        /// Create a vector of given length with elements all equal to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification.</param>
        /// <returns></returns>
        public static Vector Constant(int count, double value, Sparsity sparsity)
        {
            return sparsity.CreateConstantVector(count, value);
        }

        /// <summary>
        /// Create a vector as a copy of another vector.
        /// </summary>
        /// <param name="that">The source vector - can be dense or sparse</param>
        /// <remarks>Sparsity of created vector matches that of source vector</remarks>
        public static Vector Copy(Vector that)
        {
            return Vector.Copy(that, that.Sparsity);
        }

        /// <summary>
        /// Create a vector as a copy of another vector with a given target sparsity
        /// </summary>
        /// <param name="that">The source vector - can be dense or sparse</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification.</param>
        public static Vector Copy(Vector that, Sparsity sparsity)
        {
            if (sparsity.IsDense)
                return DenseVector.Copy(that);
            else if (sparsity.IsPiecewise)
                return PiecewiseVector.Copy(that);
            else if (sparsity.IsExact)
                return SparseVector.Copy(that);
            else
            {
                var asv = ApproximateSparseVector.Copy(that);
                asv.Sparsity = sparsity;
                return asv;
            }
        }

        /// <summary>
        /// Constructs a dense vector from an array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        public static Vector FromArray(params double[] data)
        {
            return DenseVector.FromArray(data);
        }

        /// <summary>
        /// Constructs a vector from an array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        public static Vector FromArray(double[] data, Sparsity sparsity)
        {
            if (sparsity.IsDense)
                return DenseVector.FromArray(data);
            else if (sparsity.IsPiecewise)
                return PiecewiseVector.FromArray(data);
            else if (sparsity.IsExact)
                return SparseVector.FromArray(data);
            else
            {
                var asv = ApproximateSparseVector.FromArray(data);
                asv.Sparsity = sparsity;
                return asv;
            }
        }

        /// <summary>
        /// Constructs a vector from part of an array.
        /// </summary>
        /// <param name="data">Storage for the vector elements.</param>
        /// <param name="count">The number of elements in the vector.</param>
        /// <param name="start">The starting index in the array for the vector elements.</param>
        /// <remarks><para>
        /// Throws an exception if Data is null, start &lt; 0, or count &lt; 0.
        /// </para></remarks>
        public static Vector FromArray(int count, double[] data, int start)
        {
            return DenseVector.FromArray(count, data, start);
        }

        /// <summary>
        /// Constructs a vector from a list. Maintains sparsity. 
        /// </summary>
        /// <param name="list">List to create vector from.</param>
        public static Vector FromList(IList<double> list)
        {
            if (list is SparseList<double>)
            {
                var result = SparseVector.Zero(list.Count);
                result.SetTo(list);
                return result;
            }
            else
            {
                var result = Vector.Zero(list.Count);
                result.SetTo(list);
                return result;
            }
        }

        /// <summary>
        /// Constructs a vector from part of an array.
        /// </summary>
        /// <param name="data">Storage for the vector elements.</param>
        /// <param name="count">The number of elements in the vector.</param>
        /// <param name="start">The starting index in the array for the vector elements.</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification.</param>
        /// <remarks><para>
        /// Throws an exception if Data is null, start &lt; 0, or count &lt; 0.
        /// </para></remarks>
        public static Vector FromArray(int count, double[] data, int start, Sparsity sparsity)
        {
            if (sparsity.IsDense)
                return DenseVector.FromArray(count, data, start);
            else if (sparsity.IsPiecewise)
                return PiecewiseVector.FromArray(count, data, start);
            else if (sparsity.IsExact)
                return SparseVector.FromArray(count, data, start);
            else
                return ApproximateSparseVector.FromArray(count, data, start);
        }

        #endregion

        #region IList<double> Members

        void IList<double>.Insert(int index, double item)
        {
            throw new NotSupportedException();
        }

        void IList<double>.RemoveAt(int index)
        {
            throw new NotSupportedException();
        }

        /// <summary>
        /// Returns the index of the first occurence of the given value in the array.
        /// Returns -1 if the value is not in the array
        /// </summary>
        /// <param name="item">The item to check for</param>
        /// <returns>Its index in the array</returns>
        public virtual int IndexOf(double item)
        {
            throw new NotImplementedException();
        }

        /// <summary>Gets and sets an element.</summary>
        public virtual double this[int index]
        {
            get { throw new NotImplementedException(); }
            set { throw new NotImplementedException(); }
        }

        /// <summary>
        /// Is read only
        /// </summary>
        public virtual bool IsReadOnly
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
        /// Returns true if the Vector contains the specified item value
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public virtual bool Contains(double item)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Copies this vector to the given array starting at the specified index
        /// in the target array
        /// </summary>
        /// <param name="array">The target array</param>
        /// <param name="index">The start index in the target array</param>
        public virtual void CopyTo(double[] array, int index)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Number of elements in vector
        /// </summary>
        public virtual int Count
        {
            get { throw new NotImplementedException(); }
            protected set { throw new NotImplementedException(); }
        }

        /// <summary>
        /// Gets a typed enumerator which yields the vector elements
        /// </summary>
        /// <returns></returns>
        public virtual IEnumerator<double> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        /// <summary>
        /// Checks that a given vector is the same size as this vector.
        /// Throws an exception if not with the given string
        /// </summary>
        /// <param name="that">The vector to check</param>
        /// <param name="paramName"></param>
        /// <exclude/>
        protected virtual void CheckCompatible(Vector that, string paramName)
        {
            if (Count != that.Count)
            {
                throw new ArgumentException("Vectors have different size", paramName);
            }
        }

        #region LINQ-like operators

        /// <summary>
        /// Tests if all elements in the vector satisfy the specified condition.
        /// </summary>
        /// <param name="fun">The condition for the elements to satisfy.</param>
        /// <returns>True if all elements satisfy the condition, false otherwise.</returns>
        public abstract bool All(Func<double, bool> fun);

        /// <summary>
        /// Test if all corresponding elements in this and that vector satisfy a condition
        /// </summary>
        /// <param name="that"></param>
        /// <param name="fun"></param>
        /// <returns></returns>
        public bool All(Vector that, Func<double, double, bool> fun)
        {
            return !Any(that, (x, y) => !fun(x, y));
        }

        /// <summary>
        /// Tests if any elements in the vector satisfy the specified condition.
        /// </summary>
        /// <param name="fun">The condition for the elements to satisfy.</param>
        /// <returns>True if any elements satisfy the condition, false otherwise.</returns>
        public abstract bool Any(Func<double, bool> fun);

        /// <summary>
        /// Test if any corresponding elements in this and that vector satisfy a condition
        /// </summary>
        /// <param name="that"></param>
        /// <param name="fun"></param>
        /// <returns></returns>
        public virtual bool Any(Vector that, Func<double, double, bool> fun)
        {
            if (that is DenseVector denseVector) return denseVector.Any(this, (x, y) => fun(y, x));
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns an enumeration over the indices of all elements which satisfy the specified condition.
        /// Indices are returned in sorted order.
        /// </summary>
        /// <param name="fun">A function to check if the condition is satisfied.</param>
        /// <returns>An enumeration over the indices of all elements which satisfy the specified condition.</returns>
        public IEnumerable<int> IndexOfAll(Func<double, bool> fun)
        {
            return this.FindAll(fun).Select(valueAtIndex => valueAtIndex.Index);
        }

        /// <summary>
        /// Returns an enumeration over the indices and values of all elements which satisfy the specified condition.
        /// Indices are returned in sorted order.
        /// </summary>
        /// <param name="fun">A function to check if the condition is satisfied.</param>
        /// <returns>An enumeration over the indices and values of all elements which satisfy the specified condition.</returns>
        public abstract IEnumerable<ValueAtIndex<double>> FindAll(Func<double, bool> fun);

        /// <summary>
        /// Returns the number of elements in the vector which satisfy a given condition.
        /// </summary>
        /// <param name="fun">The condition for the elements to satisfy.</param>
        /// <returns>The number of elements in the vector which satisfy the condition.</returns>
        public abstract int CountAll(Func<double, bool> fun);

        /// <summary>
        /// Returns the index of the first element that satisfies a given condition.
        /// </summary>
        /// <param name="fun">The condition for the element to satisfy.</param>
        /// <returns>The zero-based index of the first occurrence of an element that matches the conditions defined by match, if found; otherwise, -1.</returns>
        public abstract int FindFirstIndex(Func<double, bool> fun);

        /// <summary>
        /// Returns the index of the last element that satisfies a given condition.
        /// </summary>
        /// <param name="fun">The condition for the element to satisfy.</param>
        /// <returns>The zero-based index of the last occurrence of an element that matches the conditions defined by match, if found; otherwise, -1.</returns>
        public abstract int FindLastIndex(Func<double, bool> fun);

        #endregion

        #region Copying

        /// <summary>
        /// Converts this vector to an array of doubles
        /// </summary>
        /// <returns></returns>
        public virtual double[] ToArray()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Sets all elements to a given value.
        /// </summary>
        /// <param name="value">The new value.</param>
        // SetAll might be a better name.
        public virtual void SetAllElementsTo(double value)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Copies values from an array.
        /// </summary>
        /// <param name="values">An array whose length matches <c>this.Count</c>.</param>
        public virtual void SetTo(double[] values)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Copies values from a Vector to this vector
        /// </summary>
        /// <param name="that"></param>
        public virtual void SetTo(Vector that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Copies values from an Enumerable to this vector
        /// </summary>
        /// <param name="that"></param>
        public virtual void SetTo(IEnumerable<double> that)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Arithmetic operations

        /// <summary>
        /// Reduce method. Operates on this vector
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <returns></returns>
        public virtual double Reduce(double initial, Func<double, double, double> fun)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Reduce method. Operates on this vector and that vector
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="that">A second vector</param>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="that"/></param>
        /// <returns></returns>
        public virtual double Reduce(double initial, Vector that, Func<double, double, double, double> fun)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Reduce method. Operates on this vector and two other vectors
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="a">A second vector</param>
        /// <param name="b">A third vector</param>
        /// <param name="fun"></param>
        /// <returns></returns>
        public virtual double Reduce(double initial, Vector a, Vector b, Func<double, double, double, double, double> fun)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the sum of all elements.
        /// </summary>
        public virtual double Sum()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the sum of a function of all elements.
        /// </summary>
        public virtual double Sum(Converter<double, double> fun)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the sum of a function of this vector filtered by a function of a second vector.
        /// </summary>
        /// <param name="fun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="cond">Function to convert the elements of that vector to give the filter condition</param>
        /// <returns>The filtered and mapped sum</returns>
        public virtual double Sum(Converter<double, double> fun, Vector that, Converter<double, bool> cond)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            IEnumerator<double> thisEnum = GetEnumerator();
            IEnumerator<double> thatEnum = that.GetEnumerator();
            while (thisEnum.MoveNext() && thatEnum.MoveNext())
                if (cond(thatEnum.Current))
                    sum += fun(thisEnum.Current);
            return sum;
        }

        /// <summary>
        /// Returns the sum of over zero-based index * element.
        /// </summary>
        public virtual double SumI()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the sum of over square of index^2 times element.
        /// </summary>
        public virtual double SumISq()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the maximum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public virtual double Max()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the maximum of a function of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public virtual double Max(Converter<double, double> fun)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public virtual double Min()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the minimum of a function of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public virtual double Min(Converter<double, double> fun)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public virtual double Median()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the log of the sum of exponentials of the elements of the vector
        /// computed to high accuracy
        /// </summary>
        /// <returns></returns>
        public virtual double LogSumExp()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the index of the minimum element.
        /// </summary>
        /// <returns>The index of the minimum element.</returns>
        public virtual int IndexOfMinimum()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the index of the maximum element.
        /// </summary>
        /// <returns>The index of the maximum element.</returns>
        public virtual int IndexOfMaximum()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the index of the first element at which the sum of all elements so far is greater
        /// than a particular value. Useful for finding the median of a Discrete distribution.
        /// </summary>
        /// <param name="targetSum">The sum of interest</param>
        /// <returns>
        /// The index of the element where <paramref name="targetSum"/> is exceeded 
        /// or -1 if <paramref name="targetSum"/> cannot be exceeded.
        /// </returns>
        public virtual int IndexAtCumulativeSum(double targetSum)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the inner product of this vector with another vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns>Their inner product.</returns>
        public virtual double Inner(Vector that)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            IEnumerator<double> thisEnum = GetEnumerator();
            IEnumerator<double> thatEnum = that.GetEnumerator();
            while (thisEnum.MoveNext() && thatEnum.MoveNext())
                sum += thisEnum.Current*thatEnum.Current;
            return sum;
        }

        /// <summary>
        /// Returns the inner product of a function of this vector with a second vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public virtual double Inner(Vector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            IEnumerator<double> thisEnum = GetEnumerator();
            IEnumerator<double> thatEnum = that.GetEnumerator();
            while (thisEnum.MoveNext() && thatEnum.MoveNext())
                sum += thisEnum.Current*fun(thatEnum.Current);
            return sum;
        }

        /// <summary>
        /// Returns the inner product of a function of this vector with a function of a second vector.
        /// </summary>
        /// <param name="thisFun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="thatFun">Function to convert the elements of that vector</param>
        /// <returns>Their inner product.</returns>
        public virtual double Inner(Converter<double, double> thisFun, Vector that, Converter<double, double> thatFun)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            IEnumerator<double> thisEnum = GetEnumerator();
            IEnumerator<double> thatEnum = that.GetEnumerator();
            while (thisEnum.MoveNext() && thatEnum.MoveNext())
                sum += thisFun(thisEnum.Current)*thatFun(thatEnum.Current);
            return (sum);
        }

        /// <summary>
        /// Returns the outer product of this vector with another vector.
        /// </summary>
        /// <param name="that">Second vector.</param>
        /// <returns>Their outer product.</returns>
        public virtual Matrix Outer(Vector that)
        {
            Matrix outer = new Matrix(Count, that.Count);
            for (int i = 0; i < Count; ++i)
            {
                double v = this[i];
                for (int j = 0; j < that.Count; ++j)
                    outer[i, j] = v*that[j];
            }
            return outer;
        }

        /// <summary>
        /// Sets this vector to the diagonal of a matrix.
        /// </summary>
        /// <param name="m">A matrix with Rows==Cols==this.Count.</param>
        public virtual void SetToDiagonal(Matrix m)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Multiplies this vector by a scalar.
        /// </summary>
        /// <param name="scale">The scalar.</param>
        /// <returns></returns>
        /// <remarks>this receives the product.
        /// This method is a synonym for SetToProduct(this, scale)
        /// </remarks>
        public virtual void Scale(double scale)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region sum, difference etc

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of a given vector
        /// </summary>
        /// <param name="that">The given vector</param>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual Vector SetToFunction(Vector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            double[] fdata = Array.ConvertAll<double, double>(that.ToArray(), x => fun(x));
            SetTo(fdata);
            return this;
        }

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of two vectors
        /// </summary>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual Vector SetToFunction(Vector a, Vector b, Func<double, double, double> fun)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            double[] fdata = new double[a.Count];
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = 0;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                fdata[i++] = fun(aEnum.Current, bEnum.Current);
            SetTo(fdata);
            return this;
        }

        /// <summary>
        /// Sets this vector to the elementwise power of another vector.
        /// </summary>
        /// <param name="that">A vector, which must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="exponent">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="that"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual void SetToPower(Vector that, double exponent)
        {
            SetToFunction(that, x => System.Math.Pow(x, exponent));
        }

        /// <summary>
        /// Sets this vector to the elementwise product of two other vectors.
        /// </summary>
        /// <param name="a">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual void SetToProduct(Vector a, Vector b)
        {
            SetToFunction(a, b, (x, y) => x*y);
        }

        /// <summary>
        /// Set this vector to a linear combination of two other vectors
        /// </summary>
        /// <param name="aScale">The multiplier for vector a</param>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="bScale">The multiplier for vector b</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual void SetToSum(double aScale, Vector a, double bScale, Vector b)
        {
            SetToFunction(a, b, (x, y) => aScale*x + bScale*y);
        }

        /// <summary>
        /// Sets this vector to the elementwise sum of two other vectors.
        /// </summary>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToSum(Vector a, Vector b)
        {
            return SetToFunction(a, b, (x1, x2) => x1 + x2);
        }

        /// <summary>
        /// Sets this vector to another vector plus a scalar.
        /// </summary>
        /// <param name="a">A vector, which must have the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="b">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToSum(Vector a, double b)
        {
            return SetToFunction(a, x => x + b);
        }

        /// <summary>
        /// Sets this vector to the difference of two vectors
        /// </summary>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> receives the difference, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToDifference(Vector a, Vector b)
        {
            return SetToFunction(a, b, (x1, x2) => x1 - x2);
        }

        /// <summary>
        /// Set this vector to another vector minus a constant
        /// </summary>
        /// <param name="a">The other vector</param>
        /// <param name="b">The constant</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual Vector SetToDifference(Vector a, double b)
        {
            return SetToFunction(a, x => x - b);
        }

        /// <summary>
        /// Set this vector to a constant minus another vector
        /// </summary>
        /// <param name="a">The constant</param>
        /// <param name="b">The other vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual Vector SetToDifference(double a, Vector b)
        {
            return SetToFunction(b, x => a - x);
        }

        /// <summary>
        /// Sets this vector to a vector times a scalar.
        /// </summary>
        /// <param name="a">A vector, which must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToProduct(Vector a, double b)
        {
            return SetToFunction(a, x => x*b);
        }

        /// <summary>
        /// Sets this vector to the product of a vector by a matrix (i.e. x*A).
        /// </summary>
        /// <param name="x">A vector.  Cannot be <c>this</c>.</param>
        /// <param name="A">A matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="x"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToProduct(Vector x, Matrix A)
        {
            Vector d = Vector.Copy(this, Sparsity.Dense);
            d.SetToProduct(x, A);
            SetTo(d);
            return this;
        }

        /// <summary>
        /// Set this vector to the product of a matrix by a vector (i.e. A*x).
        /// </summary>
        /// <param name="A">A matrix.</param>
        /// <param name="x">A vector.  Cannot be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="x"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToProduct(Matrix A, Vector x)
        {
            Vector d = Vector.Copy(this, Sparsity.Dense);
            d.SetToProduct(A, x);
            SetTo(d);
            return this;
        }

        /// <summary>
        /// Sets this vector to the elementwise ratio of two other vectors.
        /// </summary>
        /// <param name="a">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public virtual Vector SetToRatio(Vector a, Vector b)
        {
            return SetToFunction(a, b, (x, y) => x/y);
        }

        #endregion

        #region Operator overloads

        /// <summary>
        /// Multiplies every element of a vector by a scalar.
        /// </summary>
        /// <param name="a">A vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>A new vector with the product.</returns>
        public static Vector operator *(Vector a, double b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToProduct(a, b);
            return (result);
        }

        /// <summary>
        /// Divides every element of a vector by a scalar.
        /// </summary>
        /// <param name="a">A vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>A new vector with the ratio.</returns>
        public static Vector operator /(Vector a, double b)
        {
            return a * (1 / b);
        }

        /// <summary>
        /// Multiply every element of this vector by a scalar.
        /// </summary>
        /// <param name="a">A vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>A new vector with the product.</returns>
        public static Vector operator *(double b, Vector a)
        {
            return a*b;
        }

        /// <summary>
        /// Returns a vector to some power.
        /// </summary>
        /// <param name="a">A vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>A new vector with this[i] = Math.Pow(a[i],b).</returns>
        public static Vector operator ^(Vector a, double b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToPower(a, b);
            return result;
        }

        /// <summary>
        /// Returns the elementwise product of two vectors.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>A new vector with the product.</returns>
        public static Vector operator *(Vector a, Vector b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Add a scalar to every element of a vector.
        /// </summary>
        /// <param name="a">A vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>A vector with the sum.</returns>
        public static Vector operator +(Vector a, double b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            return (Vector) result.SetToSum(a, b);
        }

        /// <summary>
        /// Returns the sum of two vectors.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>The sum.</returns>
        public static Vector operator +(Vector a, Vector b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToSum(a, b);
            return (result);
        }

        /// <summary>
        /// Returns the difference of two vectors
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>The difference.</returns>
        public static Vector operator -(Vector a, Vector b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToDifference(a, b);
            return (result);
        }

        /// <summary>
        /// Subtracts a scalar from each element of a vector.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>The difference.</returns>
        public static Vector operator -(Vector a, double b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToDifference(a, b);
            return (result);
        }

        /// <summary>
        /// Subtracts a scalar from each element of a vector.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">A scalar.</param>
        /// <returns>The difference.</returns>
        public static Vector operator -(double a, Vector b)
        {
            Vector result = Vector.Zero(b.Count, b.Sparsity);
            result.SetToDifference(a, b);
            return (result);
        }

        /// <summary>
        /// Returns a vector which is the unary negation of a vector.
        /// </summary>
        /// <param name="a">The vector to negate.</param>
        /// <returns>The negation of a.</returns>
        public static Vector operator -(Vector a)
        {
            Vector b = Vector.Copy(a);
            b.Scale(-1);
            return b;
        }

        /// <summary>
        /// Returns the ratio of two vectors
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>The difference.</returns>
        public static Vector operator /(Vector a, Vector b)
        {
            Vector result = Vector.Zero(a.Count, a.Sparsity);
            result.SetToRatio(a, b);
            return (result);
        }

        /// <summary>
        /// Returns the inner product of two vectors.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>The inner product.</returns>
        public static double InnerProduct(Vector a, Vector b)
        {
            return a.Inner(b);
        }

        #endregion

        #region Equality

        /// <summary>
        /// Equality operator.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>True if the vectors have the same size and element values.</returns>
        public static bool operator ==(Vector a, Vector b)
        {
            if (ReferenceEquals(a, b))
            {
                return true;
            }
            if (ReferenceEquals(a, null) || ReferenceEquals(b, null))
                return false;

            return a.Equals(b);
        }

        /// <summary>
        /// Inequality operator.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>True if vectors are not equal.</returns>
        public static bool operator !=(Vector a, Vector b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Determines object equality.
        /// </summary>
        /// <param name="obj">Another vector.</param>
        /// <returns>True if equal.</returns>
        /// <exclude/>
        public override bool Equals(object obj)
        {
            Vector that = obj as Vector;
            if (ReferenceEquals(that, null))
                return false;
            if (ReferenceEquals(this, that))
                return true;
            if (this.Count != that.Count)
                return false;
            for (int i = 0; i < Count; i++)
                if (this[i] != that[i])
                    return false;
            return true;
        }

        /// <summary>
        /// Tests if all elements are equal to a given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are equal to <paramref name="value"/>.</returns>
        public virtual bool EqualsAll(double value)
        {
            return All(x => x == value);
        }

        /// <summary>
        /// Tests if all elements are strictly greater than a given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are strictly greater than <paramref name="value"/>.</returns>
        public virtual bool GreaterThan(double value)
        {
            return All(x => x > value);
        }

        /// <summary>
        /// Tests if all elements are strictly less than a given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are strictly less than <paramref name="value"/>.</returns>
        public virtual bool LessThan(double value)
        {
            return All(x => x < value);
        }

        /// <summary>
        /// Tests if this vector is strictly greater than a second vector.
        /// </summary>
        /// <param name="that">The value to test against.</param>
        /// <returns>True if each element is strictly greater than the corresponding element of <paramref name="that"/>.</returns>
        public virtual bool GreaterThan(Vector that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Tests if this vector is strictly less than a second vector.
        /// </summary>
        /// <param name="that">The value to test against.</param>
        /// <returns>True if each element is strictly less than the corresponding element of <paramref name="that"/>.</returns>
        public virtual bool LessThan(Vector that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Tests if all elements are greater than or equal to a given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are greater than or equal to <paramref name="value"/>.</returns>
        public virtual bool GreaterThanOrEqual(double value)
        {
            return All(x => x >= value);
        }

        /// <summary>
        /// Tests if all elements are less than or equal to a given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are less than or equal to <paramref name="value"/>.</returns>
        public virtual bool LessThanOrEqual(double value)
        {
            return All(x => x <= value);
        }

        /// <summary>
        /// Tests if this vector is than or equal to a second vector.
        /// </summary>
        /// <param name="that">The value to test against.</param>
        /// <returns>True if each element is greater than or equal to the corresponding element of <paramref name="that"/>.</returns>
        public virtual bool GreaterThanOrEqual(Vector that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Tests if this vector is less than or equal to a second vector.
        /// </summary>
        /// <param name="that">The value to test against.</param>
        /// <returns>True if each element is strictly less than or equal to the corresponding element of <paramref name="that"/>.</returns>
        public virtual bool LessThanOrEqual(Vector that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>True if each element in the first vector is greater than the corresponding element in the second vector.</returns>
        public static bool operator >(Vector a, Vector b)
        {
            return a.GreaterThan(b);
        }

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">Vector.</param>
        /// <param name="value">Value to compare against.</param>
        /// <returns>True if each element is greater than given value.</returns>
        public static bool operator >(Vector a, double value)
        {
            return a.GreaterThan(value);
        }

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>True if each element in the first vector is less than the corresponding element in the second vector.</returns>
        public static bool operator <(Vector a, Vector b)
        {
            return a.LessThan(b);
        }

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">Vector.</param>
        /// <param name="value">Value to compare against.</param>
        /// <returns>True if each element is less than given value.</returns>
        public static bool operator <(Vector a, double value)
        {
            return a.LessThan(value);
        }

        /// <summary>
        /// Greater than or equal to operator.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>True if each element in the first vector is not less than the corresponding element in the second vector.</returns>
        public static bool operator >=(Vector a, Vector b)
        {
            return a.GreaterThanOrEqual(b);
        }

        /// <summary>
        /// Greater than or equal to operator.
        /// </summary>
        /// <param name="a">Vector.</param>
        /// <param name="value">Value to compare against.</param>
        /// <returns>True if each element is not less than given value.</returns>
        public static bool operator >=(Vector a, double value)
        {
            return a.GreaterThanOrEqual(value);
        }

        /// <summary>
        /// Less than or equal to operator.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>True if each element in the first vector is not greater than the corresponding element in the second vector.</returns>
        public static bool operator <=(Vector a, Vector b)
        {
            return a.LessThanOrEqual(b);
        }

        /// <summary>
        /// Less than or equal to operator.
        /// </summary>
        /// <param name="a">Vector.</param>
        /// <param name="value">Value to compare against.</param>
        /// <returns>True if each element is not greater than given value.</returns>
        public static bool operator <=(Vector a, double value)
        {
            return a.LessThanOrEqual(value);
        }

        /// <summary>
        /// Returns the maximum absolute difference between this vector and another vector.
        /// </summary>
        /// <param name="that">The second vector.</param>
        /// <returns><c>max(abs(this[i] - that[i]))</c>. 
        /// Matching infinities or NaNs do not count.  
        /// If <c>this</c> and <paramref name="that"/> are not the same size, returns infinity.</returns>
        /// <remarks>This routine is typically used instead of <c>Equals</c>, since <c>Equals</c> is susceptible to roundoff errors.
        /// </remarks>
        public virtual double MaxDiff(Vector that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the maximum relative difference between this vector and another.
        /// </summary>
        /// <param name="that">The second vector.</param>
        /// <param name="rel">An offset to avoid division by zero.</param>
        /// <returns><c>max(abs(this[i] - that[i])/(min(abs(this[i]),abs(that[i])) + rel))</c>. 
        /// Matching infinities or NaNs do not count.  
        /// If <c>this</c> and <paramref name="that"/> are not the same size, returns infinity.</returns>
        /// <remarks>This routine is typically used instead of <c>Equals</c>, since <c>Equals</c> is susceptible to roundoff errors.
        /// </remarks>
        public virtual double MaxDiff(Vector that, double rel)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Gets a hash code for the instance.
        /// </summary>
        /// <returns>The code.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            for (int i = 0; i < Count; i++)
                hash = Hash.Combine(hash, this[i]);
            return hash;
        }

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// </summary>
        /// <param name="format"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public virtual string ToString(string format, string delimiter)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// and a function for converting integers to display strings.
        /// </summary>
        public virtual string ToString(string format, string delimiter, Func<int, string> intToString)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Linear algebra

        /// <summary>
        /// Gets the solution to Ax=b, where A is an upper triangular matrix, and b is this vector.
        /// Equivalent to the left-division x = A\b.
        /// </summary>
        /// <param name="A">An upper triangular matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side vector b, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        public virtual Vector PredivideBy(UpperTriangularMatrix A)
        {
            var b = EnsureDense(this);
            SetTo(b.PredivideBy(A));
            return this;
        }

        /// <summary>
        /// Gets the solution to A'x=b, where A is a lower triangular matrix, and b is this vector.
        /// Equivalent to the left-division x = A'\b.
        /// </summary>
        /// <param name="A">A lower triangular matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side vector b, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        public virtual Vector PredivideByTranspose(LowerTriangularMatrix A)
        {
            var b = EnsureDense(this);
            SetTo(b.PredivideByTranspose(A));
            return this;
        }

        /// <summary>
        /// Gets the solution to Ax=b, where A is a lower triangular matrix, and b is this vector.
        /// Equivalent to the left-division x = A\b.
        /// </summary>
        /// <param name="A">A lower triangular matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side vector b, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        public virtual Vector PredivideBy(LowerTriangularMatrix A)
        {
            var b = EnsureDense(this);
            SetTo(b.PredivideBy(A));
            return this;
        }

        protected DenseVector EnsureDense(Vector vector)
        {
            if (vector.IsDense) return (DenseVector) vector;
            return DenseVector.Copy(vector);
        }

        /// <summary>
        /// Premultiply this vector by the inverse of a positive definite matrix
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public virtual Vector PredivideBy(PositiveDefiniteMatrix A)
        {
            var b = EnsureDense(this);
            SetTo(b.PredivideBy(A));
            return this;
        }

        #endregion

        #region Creation methods

        /// <summary>
        /// Appends an item to a vector - returns a new vector
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public virtual Vector Append(double item)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Return a new vector which is the concatenation of this vector and a second vector.
        /// </summary>
        /// <param name="second">Second vector</param>
        /// <returns></returns>
        public virtual Vector Append(Vector second)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Create a vector by concatenating two vectors.
        /// </summary>
        /// <param name="first">First vector</param>
        /// <param name="second">Second vector</param>
        /// <returns>A new vector with all elements of the first vector followed by all elements of the second vector.</returns>
        public static Vector Concat(Vector first, Vector second)
        {
            return first.Append(second);
        }

        /// <summary>
        /// Copy a subvector.
        /// </summary>
        /// <param name="source">A vector whose length is at least <c>count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="source"/> to copy.</param>
        /// <param name="count">The number of elements to copy.</param>
        /// <returns>A Vector of length <paramref name="count"/></returns>
        public static Vector Subvector(Vector source, int startIndex, int count)
        {
            return source.Subvector(startIndex, count);
        }

        /// <summary>
        /// Copies values from an array.
        /// </summary>
        /// <param name="values">An array whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="values"/> to copy.</param>
        public virtual void SetToSubarray(double[] values, int startIndex)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Copies value from a vector.
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public virtual void SetToSubvector(Vector that, int startIndex)
        {
            for (int i = 0, j = startIndex; i < Count; i++)
                this[i] = that[j++];
        }

        /// <summary>
        /// Create a subvector of this vector
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public virtual Vector Subvector(int startIndex, int count)
        {
            throw new NotImplementedException();
        }


        /// <summary>
        /// Set a subvector of this to another vector.
        /// </summary>
        /// <param name="startIndex">The index of the first element of this to copy to.</param>
        /// <param name="that">A vector whose length is at most <c>this.Count - startIndex</c>.</param>
        public virtual void SetSubvector(int startIndex, Vector that)
        {
            int end = startIndex + that.Count;
            for (int i = startIndex, j = 0; i < end; i++)
            {
                this[i] = that[j++];
            }
        }

        /// <summary>
        /// Clones this vector - return as a vector
        /// </summary>
        /// <returns></returns>
        public virtual Vector Clone()
        {
            throw new NotImplementedException();
        }

        object ICloneable.Clone()
        {
            return Clone();
        }

        #endregion

        /// <summary>
        /// String representation of vector with a specified format for each element
        /// </summary>
        /// <param name="format"></param>
        /// <returns></returns>
        public virtual string ToString(string format)
        {
            return ToString(format, " ");
        }

        /// <summary>
        /// Converts this sparse vector into a human readable string
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return ToString("g4");
        }
    }
}