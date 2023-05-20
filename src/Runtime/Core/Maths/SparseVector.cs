// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using System.Linq;
using SparseElement = Microsoft.ML.Probabilistic.Collections.ValueAtIndex<double>;

namespace Microsoft.ML.Probabilistic.Math
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// A one-dimensional vector of double values, optimised for the case where many of the 
    /// elements share a common value (which need not be zero).
    /// </summary>
    [Serializable]
    [DataContract]
    public class SparseVector : Vector, ISparseList<double>, ICloneable, CanSetAllElementsTo<double>, SettableTo<Vector>
    {
        [DataMember]
        private int count;

        #region Properties

        /// <summary>
        /// A list of the value and indices of elements which may not have the common value 
        /// (although they are not precluded from doing so).
        /// This list is kept sorted by index to allow efficient operations on the sparse vector.
        /// </summary>
        [DataMember]
        public List<SparseElement> SparseValues { get; protected set; }

        /// <summary>
        /// The value of all elements not mentioned explicitly as sparse values.
        /// </summary>
        [DataMember]
        public double CommonValue { get; protected set; }

        /// <summary>
        /// The number of elements in this vector
        /// </summary>
        public override int Count
        {
            get
            {
                return count;
            }
            protected set
            {
                count = value;
            }
        }

        /// <summary>
        /// The number of elements not equal to the common value
        /// </summary>
        public int SparseCount
        {
            get { return SparseValues == null ? 0 : SparseValues.Count; }
        }

        #endregion

        #region Factory methods and constructors

        /// <summary>
        /// Create a sparse vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <returns></returns>
        public new static SparseVector Zero(int count)
        {
            return new SparseVector(count);
        }

        /// <summary>
        /// Create a sparse vector of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <returns></returns>
        public new static SparseVector Constant(int count, double value)
        {
            SparseVector v = new SparseVector(count);
            v.SetAllElementsTo(value);
            return v;
        }

        /// <summary>
        /// Creator a sparse vector as a copy of another vector (which may not be sparse)
        /// </summary>
        /// <param name="that">The source vector - can be dense or sparse</param>
        public new static SparseVector Copy(Vector that)
        {
            SparseVector v = new SparseVector(that.Count);
            if (that is SparseVector)
                v.SetTo(that as SparseVector);
            else
                v.SetTo(that);
            return v;
        }

        /// <summary>
        /// Constructs a sparse vector from a dense array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        public new static SparseVector FromArray(params double[] data)
        {
            SparseVector v = new SparseVector(data.Length);
            v.SetTo(data);
            return v;
        }

        /// <summary>
        /// Constructs a sparse vector from a sorted list of sparse elements.
        /// </summary>
        /// <param name="count">Count for result</param>
        /// <param name="commonValue">Common value</param>
        /// <param name="sortedSparseValues">Sorted list of sparse elements</param>
        [Construction("Count", "CommonValue", "SparseValues")]
        public static SparseVector FromSparseValues(int count, double commonValue, List<SparseElement> sortedSparseValues)
        {
            SparseVector result = new SparseVector
                {
                    CommonValue = commonValue,
                    Count = count,
                    SparseValues = sortedSparseValues,
                };
            return result;
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
        public new static SparseVector FromArray(int count, double[] data, int start)
        {
            SparseVector v = new SparseVector(count);
            v.SetToSubarray(data, start);
            return v;
        }

        #endregion

        #region Append

        /// <summary>
        /// Appends an item to a vector - returns a new sparse vector
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public override Vector Append(double item)
        {
            SparseVector result = SparseVector.Constant(Count + 1, CommonValue);
            result.SparseValues.AddRange(SparseValues);
            result[Count] = item;
            return result;
        }

        /// <summary>
        /// Returns a new vector which appends a second vector to this vector
        /// </summary>
        /// <param name="second">Second vector</param>
        /// <returns></returns>
        /// <remarks>If the second vector is dense, or if the common values
        /// for the two vectors are different, then the result becomes dense</remarks>
        public override Vector Append(Vector second)
        {
            int jointCount = Count + second.Count;
            SparseVector seconds = second as SparseVector; // Will be null if not sparse
            if (seconds != null && seconds.CommonValue == CommonValue)
            {
                SparseVector result = SparseVector.Constant(jointCount, CommonValue);
                result.SparseValues.AddRange(SparseValues);
                foreach (SparseElement sel in seconds.SparseValues)
                    result[sel.Index + Count] = sel.Value;
                return result;
            }
            else
            {
                DenseVector result = DenseVector.Constant(jointCount, CommonValue);
                foreach (SparseElement sel in SparseValues)
                    result[sel.Index] = sel.Value;

                for (int i = Count, j = 0; i < jointCount; i++)
                    result[i] = second[j++];
                return result;
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Constructs a zero vector with the given number of elements.
        /// </summary>
        protected SparseVector()
        {
            Sparsity = Sparsity.Sparse;
        }

        /// <summary>
        /// Constructs a zero vector with the given number of elements.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        protected SparseVector(int count)
        {
            SparseValues = new List<SparseElement>();
            Count = count;
            Sparsity = Sparsity.Sparse;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        protected SparseVector(int count, double commonValue)
            : this(count)
        {
            CommonValue = commonValue;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value, except
        /// for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="sortedSparseValues">The list of sparse values, which must be sorted by index.</param>
        //[Construction("Count","CommonValue","SparseValues")]
        protected SparseVector(int count, double commonValue, List<SparseElement> sortedSparseValues)
            : this(count, commonValue)
        {
            SparseValues = sortedSparseValues;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that">the vector to copy into this new vector</param>
        protected SparseVector(SparseVector that)
        {
            SparseValues = new List<SparseElement>();
            Count = that.Count;
            Sparsity = Sparsity.Sparse;
            SetTo(that);
        }

        /// <summary>
        /// Creates a sparse vector from a list of doubles.
        /// </summary>
        /// <param name="dlist">the list of doubles</param>
        protected SparseVector(IList<double> dlist)
        {
            SparseValues = new List<SparseElement>();
            Count = dlist.Count;
            Sparsity = Sparsity.Sparse;
            SetTo(dlist);
        }

        #endregion

        #region Element-wise access

        /// <summary>Gets or sets an element at a given index.</summary>
        /// <param name="index">The index of an element.</param>
        /// <returns>The element at a given index.</returns>
        public override double this[int index]
        {
            get
            {
                if (index < 0 || index >= this.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                var k = this.GetSparseIndex(index);
                if (k < 0)
                {
                    return this.CommonValue;
                }

                return this.SparseValues[k].Value;
            }

            set
            {
                if (index < 0 || index >= this.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                var k = this.GetSparseIndex(index);
                if (value == this.CommonValue)
                {
                    if (k >= 0)
                    {
                        this.SparseValues.RemoveAt(k);
                    }
                }
                else if (k < 0)
                {
                    this.SparseValues.Insert(~k, new SparseElement {Index = index, Value = value});
                }
                else
                {
                    var sel = this.SparseValues[k];
                    sel.Value = value;
                    this.SparseValues[k] = sel;
                }
            }
        }

        #endregion

        #region Sparsity-related operations

        /// <summary>
        /// Gets the index into the sparse values array corresponding to an element index.
        /// If there is no sparse value at that index, returns the binary complement of the 
        /// index in the sparse array where such an element should be inserted to retain the
        /// sort order of the sparse array.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected int GetSparseIndex(int index)
        {
            return this.SparseValues.BinarySearch(new SparseElement {Index = index}, new ByIndexComparer<double>());
        }

        /// <summary>
        /// Returns true if there is at least one element which has the common value.
        /// </summary>
        public bool HasCommonElements
        {
            get { return Count > SparseValues.Count; }
        }


        /// <summary>
        /// Gets the dense index of the first common element.
        /// </summary>
        /// <returns>Returns the dense index of the first common element or -1 if there are no common elements</returns>
        public int GetFirstCommonIndex()
        {
            if (HasCommonElements)
            {
                int index = 0;
                foreach (SparseElement sel in SparseValues)
                {
                    if (index != sel.Index)
                        return index;
                    index++;
                }

                if (index < this.Count)
                {
                    return index;
                }
            }
            return -1;
        }

        #endregion

        #region Enumerators

        /// <summary>
        /// Gets a typed enumerator which yields the vector elements
        /// </summary>
        /// <returns></returns>
        public override IEnumerator<double> GetEnumerator()
        {
            int index = 0;
            foreach (SparseElement sel in SparseValues)
            {
                while (index++ < sel.Index)
                {
                    yield return CommonValue;
                }
                yield return sel.Value;
            }
            while (index++ < Count)
            {
                yield return CommonValue;
            }
        }

        /// <summary>
        /// Gets an enumerator which yields the vector elements
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Returns a sparse enumerator
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<double> GetSparseEnumerator()
        {
            return new SparseListEnumerator<double>(SparseValues, Count, CommonValue);
        }

        #endregion

        #region Cloning, SetTo operations

        /// <summary>
        /// Sets all elements to a given value.
        /// </summary>
        /// <param name="value">The new value.</param>
        public override void SetAllElementsTo(double value)
        {
            SparseValues.Clear();
            CommonValue = value;
        }

        /// <summary>
        /// Copies values from an array. The minimum value is used as the common value
        /// </summary>
        /// <param name="values">An array whose length matches <c>this.Count</c>.</param>
        public override void SetTo(double[] values)
        {
            double min = Double.PositiveInfinity;
            foreach (double d in values) min = System.Math.Min(min, d);
            SetAllElementsTo(min);
            for (int i = 0; i < Count; i++)
            {
                double val = values[i];
                if (val != min) SparseValues.Add(new SparseElement {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Copies values from a Vector to this vector.
        /// </summary>
        /// <param name="that"></param>
        /// <remarks> The source vector can be dense, in which case the
        /// minimum value is used as the common value.</remarks>
        public override void SetTo(Vector that)
        {
            if (that.IsSparse)
            {
                SetTo((SparseVector) that);
                return;
            }
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            double min = that.Min();
            SetAllElementsTo(min);
            for (int i = 0; i < that.Count; i++)
            {
                double val = that[i];
                if (val != min) SparseValues.Add(new SparseElement {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Copies values from a sparse vector to this sparse vector.
        /// </summary>
        /// <param name="that"></param>
        public virtual void SetTo(SparseVector that)
        {
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            CommonValue = that.CommonValue;
            SparseValues = new List<SparseElement>(that.SparseValues);
        }

        /// <summary>
        /// Copies values from a list of doubles which must have the same size as this vector.
        /// The 'common value' is set to the minimum value of the list.
        /// </summary>
        /// <param name="dlist"></param>
        public virtual void SetTo(IList<double> dlist)
        {
            double min = Double.PositiveInfinity;
            foreach (double d in dlist) min = System.Math.Min(min, d);
            SetTo(dlist, min);
        }

        /// <summary>
        /// Copies values from a list of doubles which must have the same size as this vector,
        /// using the specified common value.
        /// </summary>
        /// <param name="dlist">List of doubles</param>
        /// <param name="commonValue">Common value</param>
        public virtual void SetTo(IList<double> dlist, double commonValue)
        {
            SetAllElementsTo(commonValue);
            for (int i = 0; i < dlist.Count; i++)
            {
                double val = dlist[i];
                if (val != commonValue) SparseValues.Add(new SparseElement {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Copies values from an Enumerable to this vector
        /// </summary>
        /// <param name="that"></param>
        public override void SetTo(IEnumerable<double> that)
        {
            if (!Object.ReferenceEquals(this, that))
            {
                if (that is ISparseEnumerable<double>)
                {
                    var ise = (that as ISparseEnumerable<double>).GetSparseEnumerator();
                    var sel = new List<SparseElement>();
                    while (ise.MoveNext())
                        sel.Add(new ValueAtIndex<double>(ise.CurrentIndex, ise.Current));

                    CommonValue = ise.CommonValue;
                    SparseValues = sel;
                    Count = ise.CommonValueCount + SparseValues.Count;
                }
                else
                {
                    SetTo(that.ToArray());
                }
            }
        }

        /// <summary>
        /// Clones this vector - return as a vector
        /// </summary>
        /// <returns></returns>
        public override Vector Clone()
        {
            return new SparseVector(this);
        }

        /// <summary>
        /// Clones this vector - return as an object
        /// </summary>
        /// <returns></returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        #endregion

        #region Equality

        /// <inheritdoc/>
        public override bool Equals(object obj)
        {
            var that = obj as Vector;
            if (ReferenceEquals(this, that)) return true;
            if (ReferenceEquals(that, null))
                return false;
            if (this.Count != that.Count) return false;
            return this.MaxDiff(that) == 0;
        }

        /// <summary>
        /// Gets a hash code for the instance.
        /// </summary>
        /// <returns>The code.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, CommonValue);
            foreach (var sel in SparseValues) hash = Hash.Combine(hash, sel.GetHashCode());
            return hash;
        }

        /// <inheritdoc/>
        public override bool GreaterThan(Vector that)
        {
            return this.All(that, (x, y) => x > y);
        }

        /// <inheritdoc/>
        public override bool LessThan(Vector that)
        {
            return this.All(that, (x, y) => x < y);
        }

        /// <inheritdoc/>
        public override bool GreaterThanOrEqual(Vector that)
        {
            return this.All(that, (x, y) => x >= y);
        }

        /// <inheritdoc/>
        public override bool LessThanOrEqual(Vector that)
        {
            return this.All(that, (x, y) => x <= y);
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that)
        {
            if (Count != that.Count) return Double.PositiveInfinity;

            var absdiff = new SparseVector(Count);
            absdiff.SetToFunction(this, that, (a, b) => MMath.AbsDiffAllowingNaNs(a, b));
            return absdiff.Max();
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that, double rel)
        {
            var absdiff = new SparseVector(Count);
            absdiff.SetToFunction(this, that, (a, b) => MMath.AbsDiffAllowingNaNs(a, b, rel));
            return absdiff.Max();
        }

        #endregion

        #region LINQ-like operators (All, Any, FindAll etc.)

        /// <inheritdoc/>
        public override bool All(Func<double, bool> fun)
        {
            if (HasCommonElements)
            {
                // At least one element has the common value
                if (!fun(CommonValue)) return false;
            }
            foreach (SparseElement sel in SparseValues)
            {
                if (!fun(sel.Value)) return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool Any(Func<double, bool> fun)
        {
            if (HasCommonElements)
            {
                if (fun(CommonValue)) return true;
            }
            foreach (SparseElement sel in SparseValues)
            {
                if (fun(sel.Value)) return true;
            }
            return false;
        }

        /// <inheritdoc/>
        public override bool Any(Vector that, Func<double, double, bool> fun)
        {
            if (that.IsSparse)
            {
                return Any((SparseVector) that, fun);
            }
            return base.Any(that, fun);
        }

        /// <inheritdoc cref="Any(Vector, Func{double, double, bool})"/>
        public bool Any(SparseVector that, Func<double, double, bool> fun)
        {
            CheckCompatible(that, nameof(that));
            int sparseValueCount = 0;
            var ae = this.GetSparseEnumerator();
            var be = that.GetSparseEnumerator();
            bool aActive = ae.MoveNext();
            bool bActive = be.MoveNext();
            var aCommon = this.CommonValue;
            var bCommon = be.CommonValue;
            while (aActive || bActive)
            {
                if (aActive)
                {
                    // a is the earlier index
                    if ((!bActive) || ae.CurrentIndex < be.CurrentIndex)
                    {
                        if (fun(ae.Current, bCommon)) return true;
                        aActive = ae.MoveNext();
                        sparseValueCount++;
                        continue;
                    }
                    // a and b equal indices
                    if (bActive && ae.CurrentIndex == be.CurrentIndex)
                    {
                        if (fun(ae.Current, be.Current)) return true;
                        aActive = ae.MoveNext();
                        bActive = be.MoveNext();
                        sparseValueCount++;
                        continue;
                    }
                }
                if (bActive)
                {
                    // b is the earlier index
                    if ((!aActive) || be.CurrentIndex < ae.CurrentIndex)
                    {
                        if (fun(aCommon, be.Current)) return true;
                        bActive = be.MoveNext();
                        sparseValueCount++;
                        continue;
                    }
                }
            }
            int repeatCount = Count - sparseValueCount;
            if (repeatCount > 0)
                return fun(aCommon, bCommon);

            return false;
        }

        /// <inheritdoc/>
        public override IEnumerable<SparseElement> FindAll(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            bool funIsTrueForCommonValue = fun(this.CommonValue);
            int index = 0;
            foreach (var sparseElement in this.SparseValues)
            {
                for (; index < sparseElement.Index && funIsTrueForCommonValue; ++index)
                {
                    yield return new SparseElement(index, this.CommonValue);
                }

                if (fun(sparseElement.Value))
                {
                    yield return new SparseElement(sparseElement.Index, sparseElement.Value);
                }

                index = sparseElement.Index + 1;
            }

            for (; index < this.Count && funIsTrueForCommonValue; ++index)
            {
                yield return new SparseElement(index, this.CommonValue);
            }
        }

        /// <inheritdoc/>
        public override int CountAll(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            int result = 0;

            int commonValueCount = this.Count - this.SparseCount;
            if (commonValueCount > 0 && fun(this.CommonValue))
            {
                result += commonValueCount;
            }

            foreach (var sparseElement in this.SparseValues)
            {
                if (fun(sparseElement.Value))
                {
                    ++result;
                }
            }

            return result;
        }

        /// <inheritdoc/>
        public override int FindFirstIndex(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            bool funIsTrueForCommonValue = fun(this.CommonValue);
            int index = 0;
            int firstIndex = -1;
            foreach (var sparseElement in this.SparseValues)
            {
                if (index < sparseElement.Index && funIsTrueForCommonValue)
                {
                    firstIndex = index;
                    break;
                }

                if (fun(sparseElement.Value))
                {
                    firstIndex = sparseElement.Index;
                    break;
                }

                index = sparseElement.Index + 1;
            }

            if (index < this.Count && funIsTrueForCommonValue)
            {
                firstIndex = index;
            }

            return firstIndex;
        }

        /// <inheritdoc/>
        public override int FindLastIndex(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            bool funIsTrueForCommonValue = fun(this.CommonValue);
            int index = this.Count - 1;
            int lastIndex = -1;
            var sparseValuesInReverseOrder = Enumerable.Reverse(this.SparseValues);
            foreach (var sparseElement in sparseValuesInReverseOrder)
            {
                if (index > sparseElement.Index && funIsTrueForCommonValue)
                {
                    lastIndex = index;
                    break;
                }

                if (fun(sparseElement.Value))
                {
                    lastIndex = sparseElement.Index;
                    break;
                }

                index = sparseElement.Index - 1;
            }

            if (index >= 0 && funIsTrueForCommonValue)
            {
                lastIndex = index;
            }

            return lastIndex;
        }

        #endregion

        #region Min,max,sum operations

        /// <summary>
        /// Reduce method. Operates on this vector
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <returns></returns>
        public override double Reduce(double initial, Func<double, double, double> fun)
        {
            double result = initial;
            for (int i = 0; i < Count - SparseValues.Count; i++)
                result = fun(result, CommonValue);
            foreach (SparseElement sel in SparseValues)
                result = fun(result, sel.Value);
            return result;
        }

        /// <summary>
        /// Reduce method. Operates on this vector and that vector
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="that">A second vector</param>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="that"/></param>
        /// <returns></returns>
        public override double Reduce(double initial, Vector that, Func<double, double, double, double> fun)
        {
            double result = initial;
            IEnumerator<double> thisEnum = GetEnumerator();
            IEnumerator<double> thatEnum = that.GetEnumerator();

            while (thisEnum.MoveNext() && thatEnum.MoveNext())
                result = fun(result, thisEnum.Current, thatEnum.Current);
            return result;
        }

        /// <summary>
        /// Reduce method. Operates on this vector and that vector
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="that">A second vector</param>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="that"/></param>
        /// <returns></returns>
        public double Reduce(double initial, SparseVector that, Func<double, double, double, double> fun)
        {
            return base.Reduce(initial, that, fun);
        }

        /// <summary>
        /// Reduce method. Operates on this vector and two other vectors
        /// </summary>
        /// <param name="fun">Reduction function</param>
        /// <param name="initial">Initial value</param>
        /// <param name="a">A second vector</param>
        /// <param name="b">A third vector</param>
        /// <returns></returns>
        public double Reduce(double initial, SparseVector a, SparseVector b, Func<double, double, double, double, double> fun)
        {
            return base.Reduce(initial, a, b, fun);
        }

        /// <summary>
        /// Reduce method which can take advantage of sparse structure. Operates on this list
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        /// <remarks>This method does not take advantage of this list's sparseness.</remarks>
        public TRes Reduce<TRes>(TRes initial, Func<TRes, double, TRes> fun, Func<TRes, double, int, TRes> repeatedFun)
        {
            TRes result = initial;
            result = repeatedFun(result, CommonValue, Count - SparseValues.Count);
            foreach (var sel in SparseValues)
                result = fun(result, sel.Value);
            return result;
        }

        /// <summary>
        /// Reduce method which can take advantage of sparse structure. Operates on this list
        /// and another sparse collection
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="that">The other sparse collection</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        public TRes Reduce<TRes, T1>(TRes initial, ISparseEnumerable<T1> that, Func<TRes, double, T1, TRes> fun, Func<TRes, double, T1, int, TRes> repeatedFun)
        {
            TRes result = initial;
            int sparseValueCount = 0;
            var ae = this.GetSparseEnumerator();
            var be = that.GetSparseEnumerator();
            bool aActive = ae.MoveNext();
            bool bActive = be.MoveNext();
            var aCommon = this.CommonValue;
            var bCommon = be.CommonValue;
            while (aActive || bActive)
            {
                if (aActive)
                {
                    // a is the earlier index
                    if ((!bActive) || ae.CurrentIndex < be.CurrentIndex)
                    {
                        result = fun(result, ae.Current, bCommon);
                        aActive = ae.MoveNext();
                        sparseValueCount++;
                        continue;
                    }
                    // a and b equal indices
                    if (bActive && ae.CurrentIndex == be.CurrentIndex)
                    {
                        result = fun(result, ae.Current, be.Current);
                        aActive = ae.MoveNext();
                        bActive = be.MoveNext();
                        sparseValueCount++;
                        continue;
                    }
                }
                if (bActive)
                {
                    // b is the earlier index
                    if ((!aActive) || be.CurrentIndex < ae.CurrentIndex)
                    {
                        result = fun(result, aCommon, be.Current);
                        bActive = be.MoveNext();
                        sparseValueCount++;
                        continue;
                    }
                }
            }
            int repeatCount = Count - sparseValueCount;
            if (repeatCount > 0)
                result = repeatedFun(result, aCommon, bCommon, repeatCount);

            return result;
        }

        /// <summary>
        /// Reduce method which can take advantage of sparse structure. Operates on this list
        /// and two other sparse collections
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="b">A second sparse collection</param>
        /// <param name="c">A third sparse collection</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        public TRes Reduce<TRes, T1, T2>(TRes initial, ISparseEnumerable<T1> b, ISparseEnumerable<T2> c, Func<TRes, double, T1, T2, TRes> fun,
                                         Func<TRes, double, T1, T2, int, TRes> repeatedFun)
        {
            TRes result = initial;
            int sparseValueCount = 0;
            var ae = this.GetSparseEnumerator();
            var be = b.GetSparseEnumerator();
            var ce = c.GetSparseEnumerator();
            bool aActive = ae.MoveNext();
            bool bActive = be.MoveNext();
            bool cActive = ce.MoveNext();
            var aCommon = ae.CommonValue;
            var bCommon = be.CommonValue;
            var cCommon = ce.CommonValue;

            while (aActive || bActive || cActive)
            {
                if (aActive)
                {
                    if ((!bActive) || ae.CurrentIndex < be.CurrentIndex)
                    {
                        // a is the earliest index
                        if ((!cActive) || ae.CurrentIndex < ce.CurrentIndex)
                        {
                            result = fun(result, ae.Current, bCommon, cCommon);
                            aActive = ae.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                        // a and c are the earliest index
                        if (cActive && ae.CurrentIndex == ce.CurrentIndex)
                        {
                            result = fun(result, ae.Current, bCommon, ce.Current);
                            aActive = ae.MoveNext();
                            cActive = ce.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                    }
                    if (bActive && ae.CurrentIndex == be.CurrentIndex)
                    {
                        // a and b are the earliest index
                        if ((!cActive) || ae.CurrentIndex < ce.CurrentIndex)
                        {
                            result = fun(result, ae.Current, be.Current, cCommon);
                            aActive = ae.MoveNext();
                            bActive = be.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                        // a, b, and c are the earliest index
                        if (cActive && ae.CurrentIndex == ce.CurrentIndex)
                        {
                            result = fun(result, ae.Current, be.Current, ce.Current);
                            aActive = ae.MoveNext();
                            bActive = be.MoveNext();
                            cActive = ce.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                    }
                }
                if (bActive)
                {
                    if ((!aActive) || be.CurrentIndex < ae.CurrentIndex)
                    {
                        // b is the earliest index
                        if ((!cActive) || be.CurrentIndex < ce.CurrentIndex)
                        {
                            result = fun(result, aCommon, be.Current, cCommon);
                            bActive = be.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                        // b and c are the earliest index
                        if (cActive && be.CurrentIndex == ce.CurrentIndex)
                        {
                            result = fun(result, aCommon, be.Current, ce.Current);
                            bActive = be.MoveNext();
                            cActive = ce.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                    }
                }
                if (cActive)
                {
                    if ((!aActive) || ce.CurrentIndex < ae.CurrentIndex)
                    {
                        // c is the earliest index
                        if ((!bActive) || ce.CurrentIndex < be.CurrentIndex)
                        {
                            result = fun(result, aCommon, bCommon, ce.Current);
                            cActive = ce.MoveNext();
                            sparseValueCount++;
                            continue;
                        }
                    }
                }
            }

            int repeatCount = Count - sparseValueCount;
            if (repeatCount > 0)
                result = repeatedFun(result, aCommon, bCommon, cCommon, repeatCount);

            return result;
        }

        /// <summary>
        /// Returns the sum of all elements.
        /// </summary>
        public override double Sum()
        {
            double sum = (Count - SparseValues.Count)*CommonValue;
            foreach (var sel in SparseValues) sum += sel.Value;
            return sum;
        }

        /// <summary>
        /// Returns the sum of a function of all elements.
        /// </summary>
        /// <param name="fun">Conversion function</param>
        public override double Sum(Converter<double, double> fun)
        {
            double sum = (Count - SparseValues.Count)*fun(CommonValue);
            foreach (var sel in SparseValues) sum += fun(sel.Value);
            return sum;
        }

        /// <summary>
        /// Returns the sum of a function of this vector filtered by a function of a second vector.
        /// </summary>
        /// <param name="fun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="cond">Function to convert the elements of that vector to give the filter condition</param>
        /// <returns>The filtered and mapped sum</returns>
        public override double Sum(Converter<double, double> fun, Vector that, Converter<double, bool> cond)
        {
            if (that.IsSparse)
                return Sum(fun, (SparseVector) that, cond);
            return base.Sum(fun, that, cond);
        }

        /// <summary>
        /// Returns the sum of a function of this sparse vector filtered by a function of a second sparse vector.
        /// </summary>
        /// <param name="fun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="cond">Function to convert the elements of that vector to give the filter condition</param>
        /// <returns>The filtered and mapped sum</returns>
        public double Sum(Converter<double, double> fun, SparseVector that, Converter<double, bool> cond)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            int thisIndex = 0;
            int thatIndex = 0;
            SparseElement thisSel = this.GetSparseValue(thisIndex);
            SparseElement thatSel = that.GetSparseValue(thatIndex);
            double thisFunSelValue = (thisSel.Index == -1) ? 0.0 : fun(thisSel.Value);
            double thisFunCommonValue = fun(CommonValue);
            bool thatSelCond = (thatSel.Index == -1) ? false : cond(thatSel.Value);
            bool thatCommonCond = cond(that.CommonValue);
            int sparseCount = 0;
            while ((thisSel.Index != -1) || (thatSel.Index != -1))
            {
                if (((thisSel.Index < thatSel.Index) && (thisSel.Index != -1)) || (thatSel.Index == -1))
                {
                    if (thatCommonCond)
                        sum += thisFunSelValue;
                    thisIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    thisFunSelValue = (thisSel.Index == -1) ? 0.0 : fun(thisSel.Value);
                    sparseCount++;
                    continue;
                }
                if ((thatSel.Index < thisSel.Index) || (thisSel.Index == -1))
                {
                    if (thatSelCond)
                        sum += thisFunCommonValue;
                    thatIndex++;
                    thatSel = that.GetSparseValue(thatIndex);
                    thatSelCond = (thatSel.Index == -1) ? false : cond(thatSel.Value);
                    sparseCount++;
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (thisSel.Index == thatSel.Index)
                {
                    if (thatSelCond)
                        sum += thisFunSelValue;
                    thisIndex++;
                    thatIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    thisFunSelValue = (thisSel.Index == -1) ? 0.0 : fun(thisSel.Value);
                    thatSel = that.GetSparseValue(thatIndex);
                    thatSelCond = (thatSel.Index == -1) ? false : cond(thatSel.Value);
                    sparseCount++;
                }
            }
            if (sparseCount < Count)
                sum += thisFunCommonValue*(Count - sparseCount);
            return (sum);
        }

        /// <summary>
        /// Returns the sum of over zero-based index times element.
        /// </summary>
        public override double SumI()
        {
            double result = 0.5*(Count - 1)*Count*CommonValue;
            foreach (var sel in SparseValues) result += sel.Index*(sel.Value - CommonValue);
            return result;
        }

        /// <summary>
        /// Returns the sum of over square of index^2 times element.
        /// </summary>
        public override double SumISq()
        {
            double result = (Count - 1)*Count*(2.0*Count - 1.0)*CommonValue/6.0;
            foreach (var sel in SparseValues) result += sel.Index*sel.Index*(sel.Value - CommonValue);
            return result;
        }

        /// <summary>
        /// Returns the maximum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public override double Max()
        {
            double max = double.NegativeInfinity;
            if (HasCommonElements) max = CommonValue;
            foreach (var sel in SparseValues) max = System.Math.Max(max, sel.Value);
            return max;
        }

        /// <summary>
        /// Returns the maximum of a function of the elements in the vector
        /// </summary>
        /// <param name="fun">Conversion function</param>
        /// <returns></returns>
        public override double Max(Converter<double, double> fun)
        {
            double max = double.NegativeInfinity;
            if (HasCommonElements) max = fun(CommonValue);
            foreach (var sel in SparseValues) max = System.Math.Max(max, fun(sel.Value));
            return max;
        }

        /// <summary>
        /// Returns the median of all elements.
        /// </summary>
        public override double Median()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public override double Min()
        {
            double min = double.PositiveInfinity;
            if (HasCommonElements) min = CommonValue;
            foreach (var sel in SparseValues) min = System.Math.Min(min, sel.Value);
            return min;
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <param name="fun">Conversion function</param>
        /// <returns></returns>
        public override double Min(Converter<double, double> fun)
        {
            double min = double.PositiveInfinity;
            if (HasCommonElements) min = fun(CommonValue);
            foreach (var sel in SparseValues) min = System.Math.Min(min, fun(sel.Value));
            return min;
        }

        /// <summary>
        /// Returns the log of the sum of exponentials of the elements of the vector
        /// computed to high accuracy
        /// </summary>
        /// <returns></returns>
        public override double LogSumExp()
        {
            double max = Max();
            double Z = 0.0;
            if (HasCommonElements)
                Z += System.Math.Exp(CommonValue - max)*(Count - SparseValues.Count);
            foreach (var sel in SparseValues)
                Z += System.Math.Exp(sel.Value - max);
            return System.Math.Log(Z) + max;
        }

        /// <summary>
        /// Returns the index of the minimum element.
        /// </summary>
        /// <returns>The index of the minimum element.</returns>
        public override int IndexOfMinimum()
        {
            double min = double.PositiveInfinity;
            if (HasCommonElements) min = CommonValue;
            int pos = 0;

            foreach (var sel in SparseValues)
            {
                if (min > sel.Value)
                {
                    min = sel.Value;
                    pos = sel.Index;
                }
            }
            if (HasCommonElements && min == CommonValue)
                pos = GetFirstCommonIndex();
            return pos;
        }

        /// <summary>
        /// Returns the index of the maximum element.
        /// </summary>
        /// <returns>The index of the maximum element.</returns>
        public override int IndexOfMaximum()
        {
            double max = double.NegativeInfinity;
            if (HasCommonElements) max = CommonValue;
            int pos = 0;

            foreach (var sel in SparseValues)
            {
                if (max < sel.Value)
                {
                    max = sel.Value;
                    pos = sel.Index;
                }
            }
            if (HasCommonElements && max == CommonValue)
                pos = GetFirstCommonIndex();
            return pos;
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
        public override int IndexAtCumulativeSum(double targetSum)
        {
            const int TargetSumNotExceeded = -1;

            if (this.Count == 0)
            {
                return TargetSumNotExceeded;
            }

            int position = 0;
            int result = TargetSumNotExceeded;
            double sum = 0.0;
            double nextSum = 0.0;

            foreach (SparseElement sel in this.SparseValues)
            {
                // Add up the common values sitting between the sparse elements
                nextSum = (sel.Index - position)*this.CommonValue + sum;
                if (nextSum > targetSum)
                {
                    result = (int) ((targetSum - sum)/this.CommonValue) + position;
                    break;
                }

                // Add the current sparse element
                sum = nextSum + sel.Value;
                if (sum > targetSum)
                {
                    result = sel.Index;
                    break;
                }

                position = sel.Index + 1;
            }

            // Add up the remaining common elements
            if (result == TargetSumNotExceeded)
            {
                nextSum = (this.Count - position)*this.CommonValue + sum;
                if (nextSum > targetSum)
                {
                    result = (int) ((targetSum - sum)/this.CommonValue) + position;
                }
            }

            // The targeted sum was not reached
            return result;
        }

        /// <summary>
        /// Returns the inner product of this vector with another vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Vector that)
        {
            if (that.IsSparse)
                return Inner((SparseVector) that);
            return base.Inner(that);
        }

        /// <summary>
        /// Returns the inner product of this sparse vector with another sparse vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns>Their inner product.</returns>
        public double Inner(SparseVector that)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            int thisIndex = 0;
            int thatIndex = 0;
            SparseElement thisSel = this.GetSparseValue(thisIndex);
            SparseElement thatSel = that.GetSparseValue(thatIndex);
            int sparseCount = 0;
            while ((thisSel.Index != -1) || (thatSel.Index != -1))
            {
                if (((thisSel.Index < thatSel.Index) && (thisSel.Index != -1)) || (thatSel.Index == -1))
                {
                    sum += (thisSel.Value*that.CommonValue);
                    thisIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    sparseCount++;
                    continue;
                }
                if ((thatSel.Index < thisSel.Index) || (thisSel.Index == -1))
                {
                    sum += (this.CommonValue*thatSel.Value);
                    thatIndex++;
                    thatSel = that.GetSparseValue(thatIndex);
                    sparseCount++;
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (thisSel.Index == thatSel.Index)
                {
                    sum += (thisSel.Value*thatSel.Value);
                    thisIndex++;
                    thatIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    thatSel = that.GetSparseValue(thatIndex);
                    sparseCount++;
                }
            }
            if (sparseCount < Count)
                sum += (this.CommonValue*that.CommonValue*(Count - sparseCount));
            return (sum);
        }

        /// <summary>
        /// Returns the inner product of this vector with a function of a second vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Vector that, Converter<double, double> fun)
        {
            if (that.IsSparse)
                return Inner((SparseVector) that, fun);
            return base.Inner(that, fun);
        }

        /// <summary>
        /// Returns the inner product of this sparse vector with a function of a second sparse vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public double Inner(SparseVector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            int thisIndex = 0;
            int thatIndex = 0;
            SparseElement thisSel = this.GetSparseValue(thisIndex);
            SparseElement thatSel = that.GetSparseValue(thatIndex);
            double thatFunSelValue = (thatSel.Index == -1) ? 0.0 : fun(thatSel.Value);
            double thatFunCommonValue = fun(that.CommonValue);
            int sparseCount = 0;
            while ((thisSel.Index != -1) || (thatSel.Index != -1))
            {
                if (((thisSel.Index < thatSel.Index) && (thisSel.Index != -1)) || (thatSel.Index == -1))
                {
                    sum += (thisSel.Value*thatFunCommonValue);
                    thisIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    sparseCount++;
                    continue;
                }
                if ((thatSel.Index < thisSel.Index) || (thisSel.Index == -1))
                {
                    sum += (this.CommonValue*thatFunSelValue);
                    thatIndex++;
                    thatSel = that.GetSparseValue(thatIndex);
                    thatFunSelValue = (thatSel.Index == -1) ? 0.0 : fun(thatSel.Value);
                    sparseCount++;
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (thisSel.Index == thatSel.Index)
                {
                    sum += (thisSel.Value*thatFunSelValue);
                    thisIndex++;
                    thatIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    thatSel = that.GetSparseValue(thatIndex);
                    thatFunSelValue = (thatSel.Index == -1) ? 0.0 : fun(thatSel.Value);
                    sparseCount++;
                }
            }
            if (sparseCount < Count)
                sum += (this.CommonValue*thatFunCommonValue*(Count - sparseCount));
            return (sum);
        }

        /// <summary>
        /// Returns the inner product of a function of this vector with a function of a second vector.
        /// </summary>
        /// <param name="thisFun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="thatFun">Function to convert the elements of that vector</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Converter<double, double> thisFun, Vector that, Converter<double, double> thatFun)
        {
            if (that.IsSparse)
                return Inner(thisFun, (SparseVector) that, thatFun);
            return base.Inner(thisFun, that, thatFun);
        }

        /// <summary>
        /// Returns the inner product of a function of this sparse vector with a function of a second sparse vector.
        /// </summary>
        /// <param name="thisFun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="thatFun">Function to convert the elements of that vector</param>
        /// <returns>Their inner product.</returns>
        public double Inner(Converter<double, double> thisFun, SparseVector that, Converter<double, double> thatFun)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            int thisIndex = 0;
            int thatIndex = 0;
            SparseElement thisSel = this.GetSparseValue(thisIndex);
            SparseElement thatSel = that.GetSparseValue(thatIndex);
            double thisFunSelValue = (thisSel.Index == -1) ? 0.0 : thisFun(thisSel.Value);
            double thisFunCommonValue = thisFun(CommonValue);
            double thatFunSelValue = (thatSel.Index == -1) ? 0.0 : thatFun(thatSel.Value);
            double thatFunCommonValue = thatFun(that.CommonValue);
            int sparseCount = 0;
            while ((thisSel.Index != -1) || (thatSel.Index != -1))
            {
                if (((thisSel.Index < thatSel.Index) && (thisSel.Index != -1)) || (thatSel.Index == -1))
                {
                    sum += (thisFunSelValue*thatFunCommonValue);
                    thisIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    thisFunSelValue = (thisSel.Index == -1) ? 0.0 : thisFun(thisSel.Value);
                    sparseCount++;
                    continue;
                }
                if ((thatSel.Index < thisSel.Index) || (thisSel.Index == -1))
                {
                    sum += (thisFunCommonValue*thatFunSelValue);
                    thatIndex++;
                    thatSel = that.GetSparseValue(thatIndex);
                    thatFunSelValue = (thatSel.Index == -1) ? 0.0 : thatFun(thatSel.Value);
                    sparseCount++;
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (thisSel.Index == thatSel.Index)
                {
                    sum += (thisFunSelValue*thatFunSelValue);
                    thisIndex++;
                    thatIndex++;
                    thisSel = this.GetSparseValue(thisIndex);
                    thisFunSelValue = (thisSel.Index == -1) ? 0.0 : thisFun(thisSel.Value);
                    thatSel = that.GetSparseValue(thatIndex);
                    thatFunSelValue = (thatSel.Index == -1) ? 0.0 : thatFun(thatSel.Value);
                    sparseCount++;
                }
            }
            if (sparseCount < Count)
                sum += (thisFunCommonValue*thatFunCommonValue*(Count - sparseCount));
            return (sum);
        }

        /// <summary>
        /// Returns the outer product of this vector with another vector.
        /// </summary>
        /// <param name="that">Second vector.</param>
        /// <returns>Their outer product.</returns>
        public override Matrix Outer(Vector that)
        {
            if (that.IsSparse)
                return Outer((SparseVector) that);
            return base.Outer(that);
        }

        /// <summary>
        /// Returns the outer product of this sparse vector with another sparse vector.
        /// </summary>
        /// <param name="that">Second vector.</param>
        /// <returns>Their outer product.</returns>
        public Matrix Outer(SparseVector that)
        {
            Matrix outer = new Matrix(Count, that.Count);
            double common = CommonValue*that.CommonValue;
            outer.SetAllElementsTo(common);

            foreach (SparseElement thisSel in SparseValues)
            {
                int i = thisSel.Index;
                double vi = thisSel.Value;
                for (int j = 0; j < outer.Cols; j++)
                    outer[i, j] = vi*that.CommonValue;
            }
            foreach (SparseElement thatSel in that.SparseValues)
            {
                int j = thatSel.Index;
                double vj = thatSel.Value;
                for (int i = 0; i < outer.Rows; i++)
                    outer[i, j] = CommonValue*vj;
            }
            foreach (SparseElement thisSel in SparseValues)
            {
                int i = thisSel.Index;
                double vi = thisSel.Value;
                foreach (SparseElement thatSel in that.SparseValues)
                {
                    int j = thatSel.Index;
                    double vj = thatSel.Value;
                    outer[i, j] = vi*vj;
                }
            }
            return (outer);
        }

        /// <summary>
        /// Sets this vector to the diagonal of a matrix.
        /// </summary>
        /// <param name="m">A matrix with Rows==Cols==this.Count.</param>
        public override void SetToDiagonal(Matrix m)
        {
            double min = Double.PositiveInfinity;
            for (int i = 0; i < m.Rows; i++)
                min = System.Math.Min(min, m[i, i]);
            SetAllElementsTo(min);
            for (int i = 0; i < m.Rows; i++)
            {
                double val = m[i, i];
                if (val != min) SparseValues.Add(new SparseElement {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Multiplies this vector by a scalar.
        /// </summary>
        /// <param name="scale">The scalar.</param>
        /// <returns></returns>
        /// <remarks>this receives the product.
        /// This method is a synonym for SetToProduct(this, scale)
        /// </remarks>
        public override void Scale(double scale)
        {
            CommonValue = scale*CommonValue;
            List<SparseElement> sparseValues = new List<SparseElement>(SparseValues.Count);
            foreach (SparseElement se in SparseValues)
                sparseValues.Add(new SparseElement(se.Index, scale*se.Value));
            SparseValues = sparseValues;
        }

        #endregion

        #region IList support (interface implemented mainly for convenience, most operations are not supported)

        /// <summary>
        /// Returns true if the Vector contains the specified value
        /// </summary>
        /// <param name="value">The value to test for</param>
        /// <returns></returns>
        public override bool Contains(double value)
        {
            if (HasCommonElements)
            {
                // At least one element has the common value
                if (CommonValue == value) return true;
            }
            foreach (SparseElement sel in SparseValues)
            {
                if (sel.Value == value) return true;
            }
            return false;
        }

        /// <summary>
        /// Returns the index of the first occurence of the given value in the array.
        /// Returns -1 if the value is not in the array
        /// </summary>
        /// <param name="item">The item to check for</param>
        /// <returns>Its index in the array</returns>
        public override int IndexOf(double item)
        {
            if (HasCommonElements)
            {
                if (item == CommonValue)
                {
                    int idx = 0;
                    foreach (SparseElement sel in SparseValues)
                    {
                        if (sel.Index != idx) return idx;
                        idx++;
                    }
                }
            }
            foreach (SparseElement sel in SparseValues)
            {
                if (sel.Value == item) return sel.Index;
            }
            return -1;
        }

        /// <summary>
        /// Copies this vector to the given array starting at the specified index
        /// in the target array
        /// </summary>
        /// <param name="array">The target array</param>
        /// <param name="index">The start index in the target array</param>
        public override void CopyTo(double[] array, int index)
        {
            int j = index;
            for (int i = 0; i < Count; i++, j++)
                array[j] = CommonValue;
            foreach (SparseElement sel in SparseValues)
                array[index + sel.Index] = sel.Value;
        }

        #endregion

        #region General purpose function operators

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of a given vector
        /// </summary>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <param name="that">The given vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override Vector SetToFunction(Vector that, Converter<double, double> fun)
        {
            if (that.IsSparse)
                return SetToFunction((SparseVector) that, fun);
            return base.SetToFunction(that, fun);
        }

        /// <summary>
        /// Sets the elements of this sparse vector to a function of the elements of another sparse vector
        /// </summary>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <param name="that">The other vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual SparseVector SetToFunction(SparseVector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            CommonValue = fun(that.CommonValue);
            SparseValues = that.SparseValues.ConvertAll(x => new SparseElement {Index = x.Index, Value = fun(x.Value)});
            return this;
        }

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of two vectors
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override Vector SetToFunction(Vector a, Vector b, Func<double, double, double> fun)
        {
            if (a.Sparsity.IsSparse && b.Sparsity.IsSparse)
                return SetToFunction((SparseVector) a, (SparseVector) b, fun);
            return base.SetToFunction(a, b, fun);
        }

        /// <summary>
        /// Sets the elements of this sparse vector to a function of the elements of two other sparse vectors
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual SparseVector SetToFunction(SparseVector a, SparseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            List<SparseElement> newSparseValues;
            // SetToFunctionInPlace does an Insert which is expensive
            //if (ReferenceEquals(a, this) && !ReferenceEquals(b, this)) return SetToFunctionInPlace(b, fun);
            if (ReferenceEquals(a, this) || ReferenceEquals(b, this))
            {
                // This vector is one of the arguments, do not modify in place. Instead create
                // a new list to hold the sparse values of the results.

                // Set the new capacity conservatively, assuming the sparse values do not overlap
                newSparseValues = new List<SparseElement>(a.SparseValues.Count + b.SparseValues.Count);
            }
            else
            {
                // We can safely modify the sparse values list in place
                SparseValues.Clear();
                int targetCapacity = a.SparseValues.Count + b.SparseValues.Count;
                int currentCapacity = SparseValues.Capacity;
                if ((currentCapacity < targetCapacity/2) || (currentCapacity > targetCapacity*2))
                {
                    SparseValues.Capacity = targetCapacity;
                }
                newSparseValues = SparseValues;
            }
            // TODO: consider changing to use enumerators
            int aIndex = 0;
            int bIndex = 0;
            SparseElement aSel = a.GetSparseValue(aIndex);
            SparseElement bSel = b.GetSparseValue(bIndex);
            double newCommonValue = fun(a.CommonValue, b.CommonValue);
            while ((aSel.Index != -1) || (bSel.Index != -1))
            {
                if (((aSel.Index < bSel.Index) && (aSel.Index != -1)) || (bSel.Index == -1))
                {
                    newSparseValues.Add(new SparseElement {Index = aSel.Index, Value = fun(aSel.Value, b.CommonValue)});
                    aIndex++;
                    aSel = a.GetSparseValue(aIndex);
                    continue;
                }
                if ((bSel.Index < aSel.Index) || (aSel.Index == -1))
                {
                    newSparseValues.Add(new SparseElement {Index = bSel.Index, Value = fun(a.CommonValue, bSel.Value)});
                    bIndex++;
                    bSel = b.GetSparseValue(bIndex);
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (aSel.Index == bSel.Index)
                {
                    newSparseValues.Add(new SparseElement {Index = aSel.Index, Value = fun(aSel.Value, bSel.Value)});
                    aIndex++;
                    bIndex++;
                    aSel = a.GetSparseValue(aIndex);
                    bSel = b.GetSparseValue(bIndex);
                }
            }
            CommonValue = newCommonValue;
            SparseValues = newSparseValues;
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse vector to a function of the elements of this sparse vector and another sparse vectors
        /// x = fun(x,b)
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual SparseVector SetToFunctionInPlace(SparseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(b, nameof(b));
            if (ReferenceEquals(b, this))
            {
                throw new NotSupportedException("b must not be equal to this");
            }
            // TODO: consider changing to use enumerators
            int aIndex = 0;
            int bIndex = 0;
            SparseElement aSel = GetSparseValue(aIndex);
            SparseElement bSel = b.GetSparseValue(bIndex);
            double newCommonValue = fun(CommonValue, b.CommonValue);
            while ((aSel.Index != -1) || (bSel.Index != -1))
            {
                if (((aSel.Index < bSel.Index) && (aSel.Index != -1)) || (bSel.Index == -1))
                {
                    aSel.Value = fun(aSel.Value, b.CommonValue);
                    SparseValues[aIndex] = aSel;
                    aIndex++;
                    aSel = GetSparseValue(aIndex);
                    continue;
                }
                if ((bSel.Index < aSel.Index) || (aSel.Index == -1))
                {
                    SparseValues.Insert(aIndex, new SparseElement {Index = bSel.Index, Value = fun(CommonValue, bSel.Value)});
                    aIndex++;
                    bIndex++;
                    bSel = b.GetSparseValue(bIndex);
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (aSel.Index == bSel.Index)
                {
                    aSel.Value = fun(aSel.Value, bSel.Value);
                    SparseValues[aIndex] = aSel;
                    aIndex++;
                    bIndex++;
                    aSel = GetSparseValue(aIndex);
                    bSel = b.GetSparseValue(bIndex);
                }
            }
            CommonValue = newCommonValue;
            return this;
        }

        internal SparseElement GetSparseValue(int index)
        {
            return index < SparseValues.Count ? SparseValues[index] : SparseElement.NoElement;
        }

        #endregion

        #region Conversions (ToString, ToArray etc.)

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// </summary>
        /// <param name="format">Format of each element value</param>
        /// <param name="delimiter">Delimiter between sparse elements</param>
        /// <returns>A string</returns>
        public override string ToString(string format, string delimiter)
        {
            StringBuilder sb = new StringBuilder();
            if (SparseCount >= Count - 1)
            {
                for (int i = 0; i < Count; i++)
                {
                    if (i > 0) sb.Append(delimiter);
                    sb.Append(this[i].ToString(format));
                }
            }
            else
            {
                // output mimics the constructor, so that parameters do not need to be explained
                sb.Append("SparseVector(");
                sb.Append(Count);
                sb.Append(", ");
                sb.Append(CommonValue.ToString(format));
                int ct = 0;
                foreach (SparseElement sel in SparseValues)
                {
                    if (ct++ > 0) sb.Append(delimiter);
                    else sb.Append(", ");
                    sb.Append("[");
                    sb.Append(sel.Index);
                    sb.Append("] ");
                    sb.Append(sel.Value.ToString(format));
                }
                sb.Append(")");
            }
            return sb.ToString();
        }

        /// <summary>
        /// Converts this sparse vector to an array of doubles
        /// </summary>
        /// <returns></returns>
        public override double[] ToArray()
        {
            double[] result = new double[Count];
            for (int i = 0; i < result.Length; i++) result[i] = CommonValue;
            foreach (var sel in SparseValues) result[sel.Index] = sel.Value;
            return result;
        }

        /// <summary>
        /// Converts this sparse vector to an ordinary non-sparse vector
        /// </summary>
        /// <returns></returns>
        public Vector ToVector()
        {
            return Vector.FromArray(ToArray());
        }

        /// <summary>
        /// Converts the supplied list of doubles to a sparse vector, or does nothing
        /// if it is already a sparse vector.
        /// </summary>
        /// <param name="iList"></param>
        /// <returns>A sparse vector containing the list of doubles</returns>
        public static SparseVector AsSparseVector(IList<double> iList)
        {
            if (iList is SparseVector) return (SparseVector) iList;
            return new SparseVector(iList);
        }

        #endregion

        #region Subarray, subvector

        /// <summary>
        /// Copies values from an array. The minimum is used as the common value
        /// </summary>
        /// <param name="values">An array whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="values"/> to copy.</param>
        public override void SetToSubarray(double[] values, int startIndex)
        {
            double min = Double.PositiveInfinity;
            foreach (double d in values) min = System.Math.Min(min, d);
            SetAllElementsTo(min);
            int j = startIndex;
            for (int i = 0; i < Count; i++)
            {
                double val = values[j++];
                if (val != min) SparseValues.Add(new SparseElement {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Copies values from a vector. If the source vector is sparse, then the common value
        /// is set to the common value from the source vector. If the source vector
        /// is dense, then the common value is set to the minimum of the data in the source vector
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public override void SetToSubvector(Vector that, int startIndex)
        {
            if (that.IsSparse)
            {
                SetToSubvector((SparseVector) that, startIndex);
                return;
            }
            CommonValue = that.Min();
            base.SetToSubvector(that, startIndex);
        }

        /// <summary>
        /// Copies values from a sparse vector. The common value is set to the common value
        /// from the source vector.
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public virtual void SetToSubvector(SparseVector that, int startIndex)
        {
            CommonValue = that.CommonValue;
            SparseValues = new List<SparseElement>();
            foreach (SparseElement sel in that.SparseValues)
            {
                int index = sel.Index - startIndex;
                if (index >= Count)
                    break;
                if (index >= 0)
                    SparseValues.Add(new SparseElement(index, sel.Value));
            }
        }

        /// <summary>
        /// Create a subvector of this sparse vector.
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public override Vector Subvector(int startIndex, int count)
        {
            SparseVector result = new SparseVector(count);
            result.SetToSubvector(this, startIndex);
            return result;
        }

        /// <summary>
        /// Set a subvector of this sparse vector to another vector. The common value is
        /// not changed
        /// </summary>
        /// <param name="startIndex">The index of the first element of this to copy to.</param>
        /// <param name="that">A vector whose length is at most <c>this.Count - startIndex</c>.</param>
        public override void SetSubvector(int startIndex, Vector that)
        {
            if (startIndex + that.Count > Count)
                throw new ArgumentException("startIndex (" + startIndex + ") + that.Count (" + that.Count + ") > this.Count (" + this.Count + ")");
            SparseVector thats = that as SparseVector;
            if (thats != null && thats.CommonValue == CommonValue)
            {
                // Build up the new list of sparse elements
                List<SparseElement> newSV = new List<SparseElement>();
                int iorig = 0;
                for (; iorig < SparseValues.Count; iorig++)
                {
                    SparseElement sel = SparseValues[iorig];
                    if (sel.Index < startIndex)
                        newSV.Add(sel);
                    else
                        break;
                }
                foreach (SparseElement sel in thats.SparseValues)
                    newSV.Add(new SparseElement(sel.Index + startIndex, sel.Value));

                for (; iorig < SparseValues.Count; iorig++)
                {
                    SparseElement sel = SparseValues[iorig];
                    if (sel.Index >= startIndex + that.Count)
                        newSV.Add(sel);
                }
                SparseValues = newSV;
            }
            else
            {
                base.SetSubvector(startIndex, that);
            }
        }

        #endregion

        #region ICollection<double> Members

        public void Add(double item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            throw new NotImplementedException();
        }

        public bool Remove(double item)
        {
            throw new NotImplementedException();
        }

        #endregion
    }

    // Reference: Vaguely related to Chris Pal - Sparse Belief Propagation paper

    /// <summary>
    /// A one-dimensional vector of double values, optimised for the case where many of the 
    /// elements share a common value (which need not be zero) within some tolerance.
    /// </summary>
    [Serializable]
    [DataContract]
    public class ApproximateSparseVector : SparseVector, IList<double>, ICloneable, CanSetAllElementsTo<double>, SettableTo<Vector>
    {
        #region Properties

        private static double initialTolerance = 0.000001;

        /// <summary>
        /// The tolerance at which vector element values are considered equal to the common value.
        /// </summary>
        /// <remarks>By default this tolerance is set to 0.000001.</remarks>
        public double Tolerance
        {
            get { return Sparsity.Tolerance; }
        }

        /// <summary>
        /// The maximum allowed count of vector elements not set to the common value.
        /// </summary>
        /// <remarks>This is ignored if &lt;= 0 which is the default value.</remarks>
        public int CountTolerance
        {
            get { return Sparsity.CountTolerance; }
        }

        #endregion

        #region Factory methods and constructors

        /// <summary>
        /// Creates an approximate sparse vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <returns></returns>
        public new static ApproximateSparseVector Zero(int count)
        {
            return new ApproximateSparseVector(count);
        }

        /// <summary>
        /// Creates an approximate sparse vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="s">Sparsity</param>
        /// <returns></returns>
        public new static ApproximateSparseVector Zero(int count, Sparsity s)
        {
            return new ApproximateSparseVector(count, s);
        }


        /// <summary>
        /// Creates an approximate sparse vector of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <returns></returns>
        public new static ApproximateSparseVector Constant(int count, double value)
        {
            ApproximateSparseVector v = new ApproximateSparseVector(count);
            v.SetAllElementsTo(value);
            return v;
        }

        /// <summary>
        /// Creates an approximate sparse vector of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <param name="s">Sparsity</param>
        /// <returns></returns>
        public new static ApproximateSparseVector Constant(int count, double value, Sparsity s)
        {
            ApproximateSparseVector v = new ApproximateSparseVector(count, value, s);
            v.SetAllElementsTo(value);
            return v;
        }

        /// <summary>
        /// Creates an approximate sparse vector as a copy of another vector (which may not be sparse)
        /// </summary>
        /// <param name="that">The source vector - can be dense or sparse</param>
        public new static ApproximateSparseVector Copy(Vector that)
        {
            ApproximateSparseVector v = new ApproximateSparseVector(that.Count);
            if (that.IsApproximate) v.SetTo((ApproximateSparseVector) that);
            else if (that.IsSparse) v.SetTo((SparseVector) that);
            else v.SetTo(that);
            return v;
        }

        /// <summary>
        /// Constructs a sparse vector from a dense array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        public new static ApproximateSparseVector FromArray(params double[] data)
        {
            ApproximateSparseVector v = new ApproximateSparseVector(data.Length);
            v.SetTo(data);
            return v;
        }

        /// <summary>
        /// Constructs a sparse vector from a sorted list of sparse elements.
        /// </summary>
        /// <param name="count">Count for result</param>
        /// <param name="commonValue">Common value</param>
        /// <param name="s">Sparsity</param>
        /// <param name="sortedSparseValues">Sorted list of sparse elements</param>
        [Construction("Count", "CommonValue", "Sparsity", "SparseValues")]
        public static ApproximateSparseVector FromSparseValues(int count, double commonValue, Sparsity s, List<SparseElement> sortedSparseValues)
        {
            ApproximateSparseVector result = new ApproximateSparseVector
                {
                    CommonValue = commonValue,
                    Count = count,
                    SparseValues = sortedSparseValues,
                    Sparsity = s,
                };
            return result;
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
        public new static ApproximateSparseVector FromArray(int count, double[] data, int start)
        {
            ApproximateSparseVector v = new ApproximateSparseVector(count);
            v.SetToSubarray(data, start);
            return v;
        }

        #endregion

        /// <summary>
        /// Appends an item to a vector - returns a new sparse vector
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public override Vector Append(double item)
        {
            ApproximateSparseVector result = ApproximateSparseVector.Constant(Count + 1, CommonValue, Sparsity);
            result.SparseValues.AddRange(SparseValues);
            result[Count] = item;
            return result;
        }

        /// <summary>
        /// Returns a new vector which appends a second vector to this vector
        /// </summary>
        /// <param name="second">Second vector</param>
        /// <returns></returns>
        /// <remarks>If the second vector is dense, or if the common values
        /// for the two vectors are different, then the result becomes dense</remarks>
        public override Vector Append(Vector second)
        {
            int jointCount = Count + second.Count;
            SparseVector seconds = second as SparseVector; // Will be null if not sparse
            if (seconds != null && seconds.CommonValue == CommonValue)
            {
                ApproximateSparseVector result = ApproximateSparseVector.Constant(jointCount, CommonValue, Sparsity);
                result.SparseValues.AddRange(SparseValues);
                foreach (SparseElement sel in seconds.SparseValues)
                    result[sel.Index + Count] = sel.Value;
                return result;
            }
            else
            {
                DenseVector result = DenseVector.Constant(jointCount, CommonValue);
                foreach (SparseElement sel in SparseValues)
                    result[sel.Index] = sel.Value;

                for (int i = Count, j = 0; i < jointCount; i++)
                    result[i] = second[j++];
                return result;
            }
        }

        #region Constructors

        private ApproximateSparseVector()
        {
            Sparsity = Sparsity.ApproximateWithTolerance(initialTolerance);
        }

        /// <summary>
        /// Constructs a zero vector with the given number of elements.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        private ApproximateSparseVector(int count)
            : base(count)
        {
            Sparsity = Sparsity.ApproximateWithTolerance(initialTolerance);
        }

        /// <summary>
        /// Constructs a zero vector with the given number of elements and sparsity spec.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="s">Sparsity specification</param>
        private ApproximateSparseVector(int count, Sparsity s)
            : base(count)
        {
            Sparsity = s;
        }


        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        private ApproximateSparseVector(int count, double commonValue)
            : this(count)
        {
            CommonValue = commonValue;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="s">Sparsity</param>
        private ApproximateSparseVector(int count, double commonValue, Sparsity s)
            : this(count, s)
        {
            CommonValue = commonValue;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value, except
        /// for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="sortedSparseValues">The list of sparse values, which must be sorted by index.</param>
        private ApproximateSparseVector(int count, double commonValue, List<SparseElement> sortedSparseValues)
            : this(count, commonValue)
        {
            SparseValues = sortedSparseValues;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value, except
        /// for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="s">Sparsity</param>
        /// <param name="sortedSparseValues">The list of sparse values, which must be sorted by index.</param>
        private ApproximateSparseVector(int count, double commonValue, Sparsity s, List<SparseElement> sortedSparseValues)
            : this(count, commonValue, s)
        {
            SparseValues = sortedSparseValues;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that">the vector to copy into this new vector</param>
        private ApproximateSparseVector(ApproximateSparseVector that)
        {
            SparseValues = new List<SparseElement>();
            Count = that.Count;
            Sparsity = that.Sparsity;
            SetTo(that);
        }

        /// <summary>
        /// Creates a sparse vector from a list of doubles.
        /// </summary>
        /// <param name="dlist">the list of doubles</param>
        private ApproximateSparseVector(IList<double> dlist)
        {
            SparseValues = new List<SparseElement>();
            Count = dlist.Count;
            Sparsity = Sparsity.ApproximateWithTolerance(Tolerance);
            SetTo(dlist);
        }

        #endregion

        #region Element-wise access

        /// <summary>Gets or sets an element at a given index.</summary>
        /// <param name="index">The index of an element.</param>
        /// <returns>The element at a given index.</returns>
        public override double this[int index]
        {
            get
            {
                if (index < 0 || index >= this.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                var k = this.GetSparseIndex(index);
                if (k < 0)
                {
                    return this.CommonValue;
                }

                return this.SparseValues[k].Value;
            }

            set
            {
                if (index < 0 || index >= this.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                var k = this.GetSparseIndex(index);
                if (System.Math.Abs(value - this.CommonValue) <= this.Tolerance)
                {
                    if (k >= 0)
                    {
                        this.SparseValues.RemoveAt(k);
                    }
                }
                else if (k < 0)
                {
                    this.SparseValues.Insert(~k, new SparseElement { Index = index, Value = value});
                }
                else
                {
                    var sel = this.SparseValues[k];
                    sel.Value = value;
                    this.SparseValues[k] = sel;
                }
            }
        }

        private void AdjustToTolerance()
        {
            SparseValues.RemoveAll((SparseElement item) => System.Math.Abs(item.Value - CommonValue) <= Tolerance);
            AdjustToCountTolerance();
        }

        private void AdjustToCountTolerance()
        {
            if (CountTolerance > 0 && SparseValues.Count > CountTolerance)
            {
                SparseValues.Sort((a, b) => (a.Value > b.Value) ? -1 : (a.Value < b.Value) ? 1 : 0);
                SparseValues.RemoveRange(10, SparseValues.Count - CountTolerance);
                SparseValues.Sort((a, b) => a.Index - b.Index);
            }
        }

        #endregion

        #region Cloning, SetTo operations

        /// <summary>
        /// Copies values from an array. The minimum value is used as the common value
        /// </summary>
        /// <param name="values">An array whose length matches <c>this.Count</c>.</param>
        public override void SetTo(double[] values)
        {
            double min = Double.PositiveInfinity;
            foreach (double d in values) min = System.Math.Min(min, d);
            SetAllElementsTo(min);
            for (int i = 0; i < Count; i++)
            {
                double val = values[i];
                if (System.Math.Abs(val - min) > Tolerance) SparseValues.Add(new SparseElement { Index = i, Value = val});
            }
            if (CountTolerance > 0)
                AdjustToCountTolerance();
        }

        /// <summary>
        /// Copies values from a Vector to this vector.
        /// </summary>
        /// <param name="that"></param>
        /// <remarks> The source vector can be dense, in which case the
        /// minimum value is used as the common value.</remarks>
        public override void SetTo(Vector that)
        {
            if (that.IsSparse)
            {
                if (that.IsApproximate)
                    SetTo((ApproximateSparseVector) that);
                else
                    SetTo((SparseVector) that);
                return;
            }
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            double min = that.Min();
            SetAllElementsTo(min);
            for (int i = 0; i < Count; i++)
            {
                double val = that[i];
                if (System.Math.Abs(val - min) > Tolerance) SparseValues.Add(new SparseElement { Index = i, Value = val});
            }
            if (CountTolerance > 0)
                AdjustToCountTolerance();
        }

        /// <summary>
        /// Copies values from a sparse vector to this sparse vector.
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(ApproximateSparseVector that)
        {
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            CommonValue = that.CommonValue;
            Sparsity = that.Sparsity;
            SparseValues = new List<SparseElement>(that.SparseValues);
        }

        /// <summary>
        /// Copies values from a list of doubles which must have the same size as this vector,
        /// using the specified common value.
        /// </summary>
        /// <param name="dlist">List of doubles</param>
        /// <param name="commonValue">Common value</param>
        public override void SetTo(IList<double> dlist, double commonValue)
        {
            SetAllElementsTo(commonValue);
            for (int i = 0; i < Count; i++)
            {
                double val = dlist[i];
                if (System.Math.Abs(val - commonValue) > Tolerance) SparseValues.Add(new SparseElement { Index = i, Value = val});
            }
            if (CountTolerance > 0)
                AdjustToCountTolerance();
        }

        /// <summary>
        /// Set this vector to a collection
        /// </summary>
        /// <param name="that"></param>
        public override void SetTo(IEnumerable<double> that)
        {
            base.SetTo(that);
            AdjustToTolerance();
        }

        /// <summary>
        /// Clones this vector - return as a vector
        /// </summary>
        /// <returns></returns>
        public override Vector Clone()
        {
            return new ApproximateSparseVector(this);
        }

        /// <summary>
        /// Clones this vector - return as an object
        /// </summary>
        /// <returns></returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        #endregion

        #region Equality

        /// <summary>
        /// Gets a hash code for the instance.
        /// </summary>
        /// <returns>The code.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, CommonValue);
            hash = Hash.Combine(hash, Tolerance);
            hash = Hash.Combine(hash, CountTolerance);
            foreach (var sel in SparseValues) hash = Hash.Combine(hash, sel.GetHashCode());
            return hash;
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that)
        {
            if (Count != that.Count) return Double.PositiveInfinity;

            var absdiff = new ApproximateSparseVector(Count);
            absdiff.SetToFunction(this, that, (a, b) => MMath.AbsDiffAllowingNaNs(a, b));
            return absdiff.Max();
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that, double rel)
        {
            var absdiff = new ApproximateSparseVector(Count);
            absdiff.SetToFunction(this, that, (a, b) => MMath.AbsDiffAllowingNaNs(a, b, rel));
            return absdiff.Max();
        }

        #endregion

        /// <summary>
        /// Sets this vector to the diagonal of a matrix.
        /// </summary>
        /// <param name="m">A matrix with Rows==Cols==this.Count.</param>
        public override void SetToDiagonal(Matrix m)
        {
            base.SetToDiagonal(m);
            AdjustToTolerance();
        }

        /// <summary>
        /// Multiplies this vector by a scalar.
        /// </summary>
        /// <param name="scale">The scalar.</param>
        /// <returns></returns>
        /// <remarks>this receives the product.
        /// This method is a synonym for SetToProduct(this, scale)
        /// </remarks>
        public override void Scale(double scale)
        {
            base.Scale(scale);
            AdjustToTolerance();
        }

        #region IList support (interface implemented mainly for convenience, most operations are not supported)

        /// <summary>
        /// Returns true if the Vector contains the specified value up to tolerance
        /// </summary>
        /// <param name="value">The value to test for</param>
        /// <returns></returns>
        public override bool Contains(double value)
        {
            if (HasCommonElements)
            {
                // At least one element has the common value
                if (System.Math.Abs(CommonValue - value) <= Tolerance) return true;
            }
            foreach (SparseElement sel in SparseValues)
            {
                if (System.Math.Abs(sel.Value - value) <= Tolerance) return true;
            }
            return false;
        }

        /// <summary>
        /// Returns the index of the first occurence of the given value in the array.
        /// Returns -1 if the value is not in the array
        /// </summary>
        /// <param name="item">The item to check for</param>
        /// <returns>Its index in the array</returns>
        public override int IndexOf(double item)
        {
            if (HasCommonElements)
            {
                if (System.Math.Abs(CommonValue - item) <= Tolerance)
                {
                    int idx = 0;
                    foreach (SparseElement sel in SparseValues)
                    {
                        if (sel.Index != idx) return idx;
                        idx++;
                    }
                }
            }
            foreach (SparseElement sel in SparseValues)
            {
                if (System.Math.Abs(sel.Value - item) <= Tolerance) return sel.Index;
            }
            return -1;
        }

        #endregion

        #region General purpose function operators

        /// <summary>
        /// Sets the elements of this sparse vector to a function of the elements of another sparse vector
        /// </summary>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <param name="that">The other vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override SparseVector SetToFunction(SparseVector that, Converter<double, double> fun)
        {
            if (ReferenceEquals(that, this)) return SetToFunctionInPlace(fun);
            CommonValue = fun(that.CommonValue);
            SparseValues.Clear();
            foreach (SparseElement sel in that.SparseValues)
            {
                double newValue = fun(sel.Value);
                if (System.Math.Abs(newValue - CommonValue) <= Tolerance) continue;
                SparseValues.Add(new SparseElement {Index = sel.Index, Value = newValue});
            }
            AdjustToCountTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse vector to a function of themselves.
        /// </summary>
        /// <param name="fun"></param>
        /// <returns></returns>
        private SparseVector SetToFunctionInPlace(Converter<double, double> fun)
        {
            CommonValue = fun(CommonValue);
            int count = SparseValues.Count;
            for (int i = 0; i < count; i++)
            {
                var sel = SparseValues[i];
                sel.Value = fun(sel.Value);
                if (System.Math.Abs(sel.Value - CommonValue) <= Tolerance)
                {
                    SparseValues.RemoveAt(i);
                    i--;
                    count--;
                }
                else
                {
                    SparseValues[i] = sel;
                }
            }
            AdjustToCountTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse vector to a function of the elements of two other sparse vectors
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override SparseVector SetToFunction(SparseVector a, SparseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            if (ReferenceEquals(a, b)) return SetToFunction(a, x => fun(x, x));
            if (ReferenceEquals(a, this)) return SetToFunctionInPlace(b, fun);
            if (ReferenceEquals(b, this)) return SetToFunctionInPlace(a, (x, y) => fun(y, x));
            // We can safely modify the sparse values list in place
            SparseValues.Clear();
            int targetCapacity = a.SparseValues.Count + b.SparseValues.Count;
            int currentCapacity = SparseValues.Capacity;
            if ((currentCapacity < targetCapacity/2) || (currentCapacity > targetCapacity*2))
            {
                SparseValues.Capacity = targetCapacity;
            }
            // TODO: consider changing to use enumerators
            int aIndex = 0;
            int bIndex = 0;
            SparseElement aSel = a.GetSparseValue(aIndex);
            SparseElement bSel = b.GetSparseValue(bIndex);
            double newCommonValue = fun(a.CommonValue, b.CommonValue);
            while ((aSel.Index != -1) || (bSel.Index != -1))
            {
                if (((aSel.Index < bSel.Index) && (aSel.Index != -1)) || (bSel.Index == -1))
                {
                    double newValue = fun(aSel.Value, b.CommonValue);
                    if (System.Math.Abs(newValue - newCommonValue) > Tolerance)
                    {
                        SparseValues.Add(new SparseElement { Index = aSel.Index, Value = newValue});
                    }
                    aIndex++;
                    aSel = a.GetSparseValue(aIndex);
                    continue;
                }
                if ((bSel.Index < aSel.Index) || (aSel.Index == -1))
                {
                    double newValue = fun(a.CommonValue, bSel.Value);
                    if (System.Math.Abs(newValue - newCommonValue) > Tolerance)
                    {
                        SparseValues.Add(new SparseElement { Index = bSel.Index, Value = newValue});
                    }
                    bIndex++;
                    bSel = b.GetSparseValue(bIndex);
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (aSel.Index == bSel.Index)
                {
                    double newValue = fun(aSel.Value, bSel.Value);
                    if (System.Math.Abs(newValue - newCommonValue) > Tolerance)
                    {
                        SparseValues.Add(new SparseElement { Index = aSel.Index, Value = fun(aSel.Value, bSel.Value)});
                    }
                    aIndex++;
                    bIndex++;
                    aSel = a.GetSparseValue(aIndex);
                    bSel = b.GetSparseValue(bIndex);
                }
            }
            CommonValue = newCommonValue;
            AdjustToCountTolerance();
            return this;
        }

        #endregion

        /// <summary>
        /// Sets the elements of this sparse vector to a function of the elements of this sparse vector and another sparse vectors
        /// x = fun(x,b)
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override SparseVector SetToFunctionInPlace(SparseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(b, nameof(b));
            if (ReferenceEquals(b, this))
            {
                throw new NotSupportedException("b must not be equal to this");
            }
            // TODO: consider changing to use enumerators
            int aIndex = 0;
            int bIndex = 0;
            SparseElement aSel = GetSparseValue(aIndex);
            SparseElement bSel = b.GetSparseValue(bIndex);
            double newCommonValue = fun(CommonValue, b.CommonValue);
            while ((aSel.Index != -1) || (bSel.Index != -1))
            {
                if (((aSel.Index < bSel.Index) && (aSel.Index != -1)) || (bSel.Index == -1))
                {
                    aSel.Value = fun(aSel.Value, b.CommonValue);
                    if (System.Math.Abs(aSel.Value - newCommonValue) <= Tolerance)
                    {
                        SparseValues.RemoveAt(aIndex);
                    }
                    else
                    {
                        SparseValues[aIndex] = aSel;
                        aIndex++;
                    }
                    aSel = GetSparseValue(aIndex);
                    continue;
                }
                if ((bSel.Index < aSel.Index) || (aSel.Index == -1))
                {
                    double newValue = fun(CommonValue, bSel.Value);
                    if (System.Math.Abs(newValue - newCommonValue) > Tolerance)
                    {
                        SparseValues.Insert(aIndex, new SparseElement { Index = bSel.Index, Value = newValue});
                        aIndex++;
                    }
                    bIndex++;
                    bSel = b.GetSparseValue(bIndex);
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (aSel.Index == bSel.Index)
                {
                    aSel.Value = fun(aSel.Value, bSel.Value);
                    if (System.Math.Abs(aSel.Value - newCommonValue) <= Tolerance)
                    {
                        SparseValues.RemoveAt(aIndex);
                    }
                    else
                    {
                        SparseValues[aIndex] = aSel;
                        aIndex++;
                    }
                    bIndex++;
                    aSel = GetSparseValue(aIndex);
                    bSel = b.GetSparseValue(bIndex);
                }
            }
            CommonValue = newCommonValue;
            AdjustToCountTolerance();
            return this;
        }

        #region Conversions (ToString, ToArray etc.)

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// </summary>
        /// <param name="format"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public override string ToString(string format, string delimiter)
        {
            return "Approx" + base.ToString(format, delimiter);
        }

        #endregion

        /// <summary>
        /// Copies values from an array. The minimum is used as the common value
        /// </summary>
        /// <param name="values">An array whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="values"/> to copy.</param>
        public override void SetToSubarray(double[] values, int startIndex)
        {
            base.SetToSubarray(values, startIndex);
            AdjustToTolerance();
        }

        /// <summary>
        /// Copies values from a sparse vector. The common value is set to the common value
        /// from the source vector.
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public override void SetToSubvector(SparseVector that, int startIndex)
        {
            base.SetToSubvector(that, startIndex);
            AdjustToTolerance();
        }

        /// <summary>
        /// Create a subvector of this sparse vector.
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public override Vector Subvector(int startIndex, int count)
        {
            ApproximateSparseVector result = new ApproximateSparseVector(count, CommonValue, Sparsity);
            result.SetToSubvector(this, startIndex);
            return result;
        }

        /// <summary>
        /// Set a subvector of this sparse vector to another vector. The common value is
        /// not changed
        /// </summary>
        /// <param name="startIndex">The index of the first element of this to copy to.</param>
        /// <param name="that">A vector whose length is at most <c>this.Count - startIndex</c>.</param>
        public override void SetSubvector(int startIndex, Vector that)
        {
            base.SetSubvector(startIndex, that);
            AdjustToTolerance();
        }
    }
}