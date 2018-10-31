// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Utilities;
using System.Collections;
using Microsoft.ML.Probabilistic.Math;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Collections
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// A list which is optimised for the case where most of its elements share a common value.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [Serializable]
    [DataContract]
    public class SparseList<T> : ISparseList<T>, IList, ICloneable
    {
        #region Properties

        // Sparsity
        [DataMember]
        private Sparsity sparsity = Sparsity.Sparse;

        /// <summary>
        /// The <see cref="Sparsity"/> specification of this list.
        /// </summary>
        public Sparsity Sparsity
        {
            get { return sparsity; }
            protected set { sparsity = value; }
        }

        /// <summary>
        /// A list of the value and indices of elements which may not have the common value 
        /// (although they are not precluded from doing so).
        /// This list is kept sorted by index to allow efficient operations on the sparse list.
        /// </summary>
        [DataMember]
        public List<ValueAtIndex<T>> SparseValues { get; protected set; }

        /// <summary>
        /// The value of all elements not mentioned explicitly as sparse values.
        /// </summary>
        [DataMember]
        public T CommonValue { get; protected set; }

        [DataMember]
        private int count;

        /// <summary>
        /// Get or set the number of elements in this list.  Set the number of elements
        /// will grow or shrink the list accordingly.
        /// </summary>
        public int Count
        {
            get { return count; }
            set
            {
                if (value < count)
                {
                    throw new NotImplementedException("Shrinking sparse lists is not yet implemented.");
                }
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
        /// Create a sparse list of given length with elements all equal
        /// to the default value for the element type
        /// </summary>
        /// <param name="count">Number of elements in the list</param>
        /// <returns></returns>
        public static SparseList<T> FromSize(int count)
        {
            var v = new SparseList<T>(count, default(T));
            return v;
        }

        /// <summary>
        /// Create a sparse list of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in the list</param>
        /// <param name="value">Value for each element</param>
        /// <returns></returns>
        public static SparseList<T> Constant(int count, T value)
        {
            var v = new SparseList<T>(count, value);
            return v;
        }

        /// <summary>
        /// Creator a sparse list as a copy of another list (which may not be sparse)
        /// </summary>
        /// <param name="that">The source list - can be dense or sparse</param>
        public static SparseList<T> Copy(IList<T> that)
        {
            var v = new SparseList<T>(that.Count);
            v.SetTo(that);
            return v;
        }

        /// <summary>
        /// Constructs a sparse list from a sorted list of sparse elements.
        /// </summary>
        /// <param name="count">Count for result</param>
        /// <param name="commonValue">Common value</param>
        /// <param name="sortedSparseValues">Sorted list of sparse elements</param>
        [Construction("Count", "CommonValue", "SparseValues")]
        public static SparseList<T> FromSparseValues(int count, T commonValue,
                                                     List<ValueAtIndex<T>> sortedSparseValues)
        {
            var result = new SparseList<T>
                {
                    CommonValue = commonValue,
                    Count = count,
                    SparseValues = sortedSparseValues,
                };
            return result;
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Null constructor.
        /// </summary>
        protected SparseList()
        {
        }

        /// <summary>
        /// Constructs a sparse list with the given number of elements.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        protected SparseList(int count)
        {
            SparseValues = new List<ValueAtIndex<T>>();
            Count = count;
        }

        /// <summary>
        /// Constructs a sparse list of a given length and assigns all elements the given value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        protected SparseList(int count, T commonValue)
            : this(count)
        {
            CommonValue = commonValue;
        }

        /// <summary>
        /// Constructs a sparse list of a given length and assigns all elements the given value, except
        /// for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="sortedSparseValues">The list of sparse values, which must be sorted by index.</param>
        protected SparseList(int count, T commonValue, List<ValueAtIndex<T>> sortedSparseValues)
            : this(count, commonValue)
        {
            SparseValues = sortedSparseValues;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that">the sparse list to copy into this new sparse list</param>
        protected SparseList(ISparseList<T> that)
        {
            SparseValues = new List<ValueAtIndex<T>>();
            Count = that.Count;
            SetTo(that);
        }

        #endregion

        #region Element-wise access

        /// <summary>Gets or sets an element.</summary>
        public virtual T this[int index]
        {
            get
            {
                var k = GetSparseIndex(index);
                if (k < 0) return CommonValue;
                return SparseValues[k].Value;
            }
            set
            {
                var k = GetSparseIndex(index);
                if (Equals(CommonValue, value))
                {
                    if (k >= 0) SparseValues.RemoveAt(k);
                    return;
                }
                if (k < 0)
                {
                    SparseValues.Insert(~k, new ValueAtIndex<T> {Index = index, Value = value});
                }
                else
                {
                    var sel = SparseValues[k];
                    sel.Value = value;
                    SparseValues[k] = sel;
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
            return SparseValues.BinarySearch(new ValueAtIndex<T> {Index = index}, new ByIndexComparer<T>());
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
                foreach (var sel in SparseValues)
                {
                    if (index != sel.Index)
                        return index;
                    index++;
                }
            }
            return -1;
        }

        #endregion

        #region Enumerators

        /// <summary>
        /// Gets a typed enumerator which yields the list elements
        /// </summary>
        /// <returns></returns>
        public IEnumerator<T> GetEnumerator()
        {
            int index = 0;
            foreach (var sel in SparseValues)
            {
                while (index < sel.Index)
                {
                    index++;
                    yield return CommonValue;
                }
                index++;
                yield return sel.Value;
            }
            while (index < Count)
            {
                index++;
                yield return CommonValue;
            }
        }

        /// <summary>
        /// Gets an enumerator which yields the list elements
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region Cloning, SetTo operations

        /// <summary>
        /// Sets all elements to a given value.
        /// </summary>
        /// <param name="value">The new value.</param>
        public void SetAllElementsTo(T value)
        {
            SparseValues.Clear();
            CommonValue = value;
        }


        /// <summary>
        /// Copies values from another list.
        /// </summary>
        /// <param name="that"></param>
        /// <remarks> The source list can be dense, in which case 
        /// default(T) is used as the common value.</remarks>
        public void SetTo(IList<T> that)
        {
            if (that is SparseList<T>)
            {
                SetTo((SparseList<T>) that);
            }
            else if (Count == that.Count)
            {
                SetTo(that, default(T));
            }
            else
            {
                CommonValue = default(T);
                count = that.Count;
                SparseValues = that.Select((x, i) => new ValueAtIndex<T>(i, x)).ToList();
            }
        }

        /// <summary>
        /// Checks that a given list is the same size as this list.
        /// Throws an exception if not with the given string
        /// </summary>
        /// <param name="that">The list to check</param>
        /// <param name="paramName"></param>
        /// <exclude/>
        protected void CheckCompatible<T2>(IList<T2> that, string paramName)
        {
            Argument.CheckIfValid(that.Count == this.Count, string.Format("Size of {0} does not match size of this list. Expected {1}, got {2}.", paramName, this.Count, that.Count));
        }

        /// <summary>
        /// Copies values from a sparse list to this sparse list.
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(SparseList<T> that)
        {
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            CommonValue = that.CommonValue;
            SparseValues = new List<ValueAtIndex<T>>(that.SparseValues);
        }

        public void SetTo(IEnumerable<T> that)
        {
            if (Object.ReferenceEquals(this, that)) return;
            if (that is ISparseEnumerable<T>)
            {
                var ise = (that as ISparseEnumerable<T>).GetSparseEnumerator();
                var sel = new List<ValueAtIndex<T>>();
                while (ise.MoveNext())
                    sel.Add(new ValueAtIndex<T>(ise.CurrentIndex, ise.Current));

                CommonValue = ise.CommonValue;
                SparseValues = sel;
                count = ise.CommonValueCount + SparseValues.Count;
            }
            else
            {
                SetTo(that.ToList());
            }
        }


        /// <summary>
        /// Copies values from a list which must have the same size as this list,
        /// using the specified common value.
        /// </summary>
        /// <param name="dlist">The list to copy from</param>
        /// <param name="commonValue">Common value</param>
        public virtual void SetTo(IList<T> dlist, T commonValue)
        {
            SetAllElementsTo(commonValue);
            for (int i = 0; i < dlist.Count; i++)
            {
                T val = dlist[i];
                if (!Equals(val, commonValue)) SparseValues.Add(new ValueAtIndex<T> {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Clones this list - return as a sparse list.
        /// </summary>
        /// <returns></returns>
        public SparseList<T> Clone()
        {
            return new SparseList<T>(this);
        }

        /// <summary>
        /// Clones this list - return as an object
        /// </summary>
        /// <returns></returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        #endregion

        #region Equality

        /// <summary>
        /// Determines object equality.
        /// </summary>
        /// <param name="obj">Another (list) object.</param>
        /// <returns>True if equal.</returns>
        /// <exclude/>
        public override bool Equals(object obj)
        {
            var that = obj as IList<T>;
            if (ReferenceEquals(this, that)) return true;
            if (ReferenceEquals(that, null))
                return false;
            if (this.Count != that.Count) return false;
            var isEqual = SparseList<bool>.Constant(Count, false);
            return Reduce(true, that, (res, o1, o2) => res && Equals(o1, o2));
        }

        /// <summary>
        /// Gets a hash code for the instance.
        /// </summary>
        /// <returns>The code.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, CommonValue.GetHashCode());
            if (SparseValues != null)
                foreach (var sel in SparseValues) hash = Hash.Combine(hash, sel.GetHashCode());
            return hash;
        }

        /// <summary>
        /// Tests if all elements are equal to the given value.
        /// </summary>
        /// <param name="value">The value to test against.</param>
        /// <returns>True if all elements are equal to <paramref name="value"/>.</returns>
        public bool EqualsAll(T value)
        {
            return All(x => Equals(x, value));
        }

        #endregion

        #region LINQ-like operators (All, Any, IndexOfAll etc.)

        /// <summary>
        /// Tests if all elements in the list satisfy the specified condition.
        /// </summary>
        /// <param name="fun"></param>
        /// <returns></returns>
        public bool All(Converter<T, bool> fun)
        {
            if (HasCommonElements)
            {
                // At least one element has the common value
                if (!fun(CommonValue)) return false;
            }
            foreach (var sel in SparseValues)
            {
                if (!fun(sel.Value)) return false;
            }
            return true;
        }

        /// <summary>
        /// Tests if any elements in the list satisfy the specified condition.
        /// </summary>
        /// <param name="fun"></param>
        /// <returns></returns>
        public bool Any(Converter<T, bool> fun)
        {
            if (HasCommonElements)
            {
                if (fun(CommonValue)) return true;
            }
            foreach (var sel in SparseValues)
            {
                if (fun(sel.Value)) return true;
            }
            return false;
        }

        /// <summary>
        /// Returns an enumeration of the indices of all elements which satisfy the specified condition.
        /// Indices are returned in sorted order.
        /// </summary>
        /// <param name="fun"></param>
        /// <returns></returns>
        public IEnumerable<int> IndexOfAll(Converter<T, bool> fun)
        {
            if (fun(CommonValue))
            {
                for (int i = 0; i < Count; i++)
                {
                    if (fun(this[i])) yield return i;
                }
            }
            else
            {
                foreach (var sel in SparseValues)
                {
                    if (fun(sel.Value)) yield return sel.Index;
                }
            }
        }

        #endregion

        #region Reduce operation

        /// <summary>
        /// Reduce method. Operates on this list
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <returns></returns>
        /// <remarks>This method does not take advantage of this list's sparseness.</remarks>
        public TRes Reduce<TRes>(TRes initial, Func<TRes, T, TRes> fun)
        {
            TRes result = initial;
            for (int i = 0; i < Count - SparseValues.Count; i++)
                result = fun(result, CommonValue);
            foreach (var sel in SparseValues)
                result = fun(result, sel.Value);
            return result;
        }

        /// <summary>
        /// Reduce method which can take advantage of sparse structure. Operates on this list
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        /// <remarks>This method does not take advantage of this list's sparseness.</remarks>
        public TRes Reduce<TRes>(TRes initial, Func<TRes, T, TRes> fun, Func<TRes, T, int, TRes> repeatedFun)
        {
            TRes result = initial;
            result = repeatedFun(result, CommonValue, Count - SparseValues.Count);
            foreach (var sel in SparseValues)
                result = fun(result, sel.Value);
            return result;
        }

        /// <summary>
        /// Reduce method which can take advantage of sparse structure. Operates on this list
        /// and another sparse list
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="that">The other sparse list</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        public TRes Reduce<TRes, T1>(TRes initial, ISparseEnumerable<T1> that, Func<TRes, T, T1, TRes> fun, Func<TRes, T, T1, int, TRes> repeatedFun)
        {
            TRes result = initial;
            int sparseValueCount = 0;
            var ae = this.GetSparseEnumerator();
            var be = that.GetSparseEnumerator();
            bool aActive = ae.MoveNext();
            bool bActive = be.MoveNext();
            var aCommon = ae.CommonValue;
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
        /// and two other sparse lists
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="b">A second sparse list</param>
        /// <param name="c">A third sparse list</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        public TRes Reduce<TRes, T1, T2>(TRes initial, ISparseEnumerable<T1> b, ISparseEnumerable<T2> c, Func<TRes, T, T1, T2, TRes> fun,
                                         Func<TRes, T, T1, T2, int, TRes> repeatedFun)
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
        /// Reduce method. Operates on this list and another list.
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="that">A second list</param>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="that"/></param>
        /// <returns></returns>
        /// <remarks>This method does not take advantage of this list's sparseness.</remarks>
        public TRes Reduce<TRes, T2>(TRes initial, IEnumerable<T2> that, Func<TRes, T, T2, TRes> fun)
        {
            TRes result = initial;
            IEnumerator<T> thisEnum = GetEnumerator();
            IEnumerator<T2> thatEnum = that.GetEnumerator();

            while (thisEnum.MoveNext() && thatEnum.MoveNext())
                result = fun(result, thisEnum.Current, thatEnum.Current);
            return result;
        }

        /// <summary>
        /// Reduce method. Operates on this list and two other lists.
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="a">A second list</param>
        /// <param name="b">A third list</param>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="a"/> and <paramref name="b"/></param>
        /// <returns></returns>
        /// <remarks>This method does not take advantage of this list's sparseness.</remarks>
        public TRes Reduce<TRes, T1, T2>(TRes initial, IList<T1> a, IList<T2> b, Func<TRes, T, T1, T2, TRes> fun)
        {
            TRes result = initial;
            IEnumerator<T> thisEnum = GetEnumerator();
            IEnumerator<T1> aEnum = a.GetEnumerator();
            IEnumerator<T2> bEnum = b.GetEnumerator();

            while (thisEnum.MoveNext() && aEnum.MoveNext() && bEnum.MoveNext())
                result = fun(result, thisEnum.Current, aEnum.Current, bEnum.Current);
            return result;
        }

        #endregion

        #region IList support (interface implemented mainly for convenience, most operations are not supported)

        /// <summary>
        /// Returns true if this list contains the specified value
        /// </summary>
        /// <param name="value">The value to test for</param>
        /// <returns></returns>
        public bool Contains(T value)
        {
            if (HasCommonElements)
            {
                // At least one element has the common value
                if (Equals(CommonValue, value)) return true;
            }
            foreach (var sel in SparseValues)
            {
                if (Equals(sel.Value, value)) return true;
            }
            return false;
        }

        /// <summary>
        /// Returns the index of the first occurence of the given value in the list.
        /// Returns -1 if the value is not in the list
        /// </summary>
        /// <param name="item">The item to check for</param>
        /// <returns>Its index in the list</returns>
        public int IndexOf(T item)
        {
            if (HasCommonElements)
            {
                if (Equals(item, CommonValue))
                {
                    int idx = 0;
                    foreach (var sel in SparseValues)
                    {
                        if (sel.Index != idx) return idx;
                        idx++;
                    }
                }
            }
            foreach (var sel in SparseValues)
            {
                if (Equals(sel.Value, item)) return sel.Index;
            }
            return -1;
        }

        /// <summary>
        /// Sparse lists are not read only.
        /// </summary>
        public bool IsReadOnly
        {
            get { return false; }
        }

        /// <summary>
        /// Copies this sparse list to the given array starting at the specified index
        /// in the target array
        /// </summary>
        /// <param name="array">The target array</param>
        /// <param name="index">The start index in the target array</param>
        public void CopyTo(T[] array, int index)
        {
            int j = index;
            for (int i = 0; i < Count; i++, j++)
                array[j] = CommonValue;
            foreach (var sel in SparseValues)
                array[index + sel.Index] = sel.Value;
        }

        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="index"></param>
        /// <param name="item"></param>
        public void Insert(int index, T item)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="index"></param>
        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="item"></param>
        public void Add(T item)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Clears this list, setting it to zero length.
        /// </summary>
        public void Clear()
        {
            Count = 0;
            SparseValues.Clear();
        }

        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Remove(T item)
        {
            throw new NotImplementedException();
        }


        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        int IList.Add(object value)
        {
            throw new NotImplementedException();
        }


        bool IList.Contains(object value)
        {
            if (!(value is T)) return false;
            return Contains((T) value);
        }

        int IList.IndexOf(object value)
        {
            if (!(value is T)) return -1;
            return IndexOf((T) value);
        }

        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="index"></param>
        /// <param name="value"></param>
        void IList.Insert(int index, object value)
        {
            throw new NotImplementedException();
        }

        bool IList.IsFixedSize
        {
            get { return false; }
        }

        /// <summary>
        /// Not implemented
        /// </summary>
        /// <param name="value"></param>
        void IList.Remove(object value)
        {
            throw new NotImplementedException();
        }

        object IList.this[int index]
        {
            get { return this[index]; }
            set { this[index] = (T) value; }
        }


        void ICollection.CopyTo(Array array, int index)
        {
            CopyTo((T[]) array, index);
        }

        bool ICollection.IsSynchronized
        {
            get { return false; }
        }

        object ICollection.SyncRoot
        {
            get { return null; }
        }

        #endregion

        #region General purpose function operators

        /// <summary>
        /// Creates an array from the list
        /// </summary>
        /// <typeparam name="T2"></typeparam>
        /// <param name="that"></param>
        /// <returns></returns>
        protected T2[] ToArray<T2>(IList<T2> that)
        {
            var thatArray = new T2[that.Count];
            that.CopyTo(thatArray, 0);
            return thatArray;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of another sparse list
        /// </summary>
        /// <param name="fun">The function which maps from type T2 to type T</param>
        /// <param name="that">The other list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public SparseList<T> SetToFunction<T2>(SparseList<T2> that, Func<T2, T> fun)
        {
            CheckCompatible(that, nameof(that));
            CommonValue = fun(that.CommonValue);
            SparseValues = that.SparseValues.ConvertAll(x => new ValueAtIndex<T> {Index = x.Index, Value = fun(x.Value)});
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of a sparse collection
        /// </summary>
        /// <param name="fun">The function which maps from type T2 to type T</param>
        /// <param name="that">The other list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public SparseList<T> SetToFunction<T2>(ISparseEnumerable<T2> that, Func<T2, T> fun)
        {
            if (that is IList<T2>)
                CheckCompatible((IList<T2>) that, nameof(that));

            ISparseEnumerable<T> res = fun.Map(that) as ISparseEnumerable<T>; // Lazy
            var rese = res.GetSparseEnumerator();
            SparseValues.Clear();
            while (rese.MoveNext())
                SparseValues.Add(new ValueAtIndex<T>(rese.CurrentIndex, rese.Current));
            CommonValue = rese.CommonValue;
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of a collection
        /// </summary>
        /// <param name="fun">The function which maps from type T2 to type T</param>
        /// <param name="that">The other list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public SparseList<T> SetToFunction<T2>(IEnumerable<T2> that, Func<T2, T> fun)
        {
            if (that is SparseList<T2>)
                return SetToFunction((SparseList<T2>) that, fun);

            if (that is ISparseEnumerable<T2>)
                return SetToFunction((ISparseEnumerable<T2>) that, fun);

            List<T> fdata = new List<T>();
            var fmap = fun.Map(that);
            foreach (T elem in fmap)
                fdata.Add(elem);
            SetTo(ToArray(fdata));
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of two other sparse lists
        /// </summary>
        /// <param name="fun">The function which maps two elements to an element of this list</param>
        /// <param name="a">The first list</param>
        /// <param name="b">The second list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are all the same length</remarks>
        public SparseList<T> SetToFunction<T1, T2>(SparseList<T1> a, SparseList<T2> b, Func<T1, T2, T> fun)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            List<ValueAtIndex<T>> newSparseValues;
            if (ReferenceEquals(a, this) && !ReferenceEquals(b, this)) return SetToFunctionInPlace<T2>(b, (x, y) => fun((T1) (object) x, y));
            if (ReferenceEquals(a, this) || ReferenceEquals(b, this))
            {
                // This vector is one of the arguments, do not modify in place. Instead create
                // a new list to hold the sparse values of the results.

                // Set the new capacity conservatively, assuming the sparse values do not overlap
                newSparseValues = new List<ValueAtIndex<T>>(a.SparseValues.Count + b.SparseValues.Count);
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
            var aSel = a.GetSparseValue(aIndex);
            var bSel = b.GetSparseValue(bIndex);
            T newCommonValue = fun(a.CommonValue, b.CommonValue);
            while ((aSel.Index != -1) || (bSel.Index != -1))
            {
                if (((aSel.Index < bSel.Index) && (aSel.Index != -1)) || (bSel.Index == -1))
                {
                    newSparseValues.Add(new ValueAtIndex<T> {Index = aSel.Index, Value = fun(aSel.Value, b.CommonValue)});
                    aIndex++;
                    aSel = a.GetSparseValue(aIndex);
                    continue;
                }
                if ((bSel.Index < aSel.Index) || (aSel.Index == -1))
                {
                    newSparseValues.Add(new ValueAtIndex<T> {Index = bSel.Index, Value = fun(a.CommonValue, bSel.Value)});
                    bIndex++;
                    bSel = b.GetSparseValue(bIndex);
                    continue;
                }
                // If two indices are the same, apply operator to both
                if (aSel.Index == bSel.Index)
                {
                    newSparseValues.Add(new ValueAtIndex<T> {Index = aSel.Index, Value = fun(aSel.Value, bSel.Value)});
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
        /// Sets the elements of this sparse list to a function of the elements of two other sparse lists
        /// </summary>
        /// <param name="a">The first list</param>
        /// <param name="b">The second list</param>
        /// <param name="fun">The function which maps two elements to an element of this list</param>
        /// <returns></returns>
        public SparseList<T> SetToFunction<T1, T2>(
            ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, Func<T1, T2, T> fun)
        {
            if (a is IList<T1>)
                CheckCompatible((IList<T1>) a, nameof(a));
            if (b is IList<T2>)
                CheckCompatible((IList<T2>) b, nameof(b));

            ISparseEnumerable<T> res = fun.Map(a, b) as ISparseEnumerable<T>; // Lazy
            var rese = res.GetSparseEnumerator();
            List<ValueAtIndex<T>> newSparseValues;
            if (ReferenceEquals(a, this) || ReferenceEquals(b, this))
                newSparseValues = new List<ValueAtIndex<T>>();
            else
            {
                SparseValues.Clear();
                newSparseValues = SparseValues;
            }
            while (rese.MoveNext())
                newSparseValues.Add(new ValueAtIndex<T>(rese.CurrentIndex, rese.Current));
            CommonValue = rese.CommonValue;
            SparseValues = newSparseValues;
            return this;
        }

        /// <summary>
        /// Sets the elements of this list to a function of the elements of two collections
        /// </summary>
        /// <param name="fun">The function which maps the two elements of the other lists to an element of this list</param>
        /// <param name="a">The first list</param>
        /// <param name="b">The second list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public SparseList<T> SetToFunction<T1, T2>(IEnumerable<T1> a, IEnumerable<T2> b, Func<T1, T2, T> fun)
        {
            if ((a is SparseList<T1>) && (b is SparseList<T2>))
                return SetToFunction((SparseList<T1>) a, (SparseList<T2>) b, fun);

            if ((a is ISparseEnumerable<T1>) && (b is ISparseEnumerable<T2>))
                return SetToFunction((ISparseEnumerable<T1>) a, (ISparseEnumerable<T2>) b, fun);

            List<T> fdata = new List<T>();
            var fmap = fun.Map(a, b);
            foreach (T elem in fmap)
                fdata.Add(elem);
            SetTo(ToArray(fdata));
            return this;
        }


        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of three sparse collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="fun">The function which maps three elements to an element of this list</param>
        /// <returns></returns>
        public SparseList<T> SetToFunction<T1, T2, T3>(
            ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c, Func<T1, T2, T3, T> fun)
        {
            if (a is IList<T1>)
                CheckCompatible((IList<T1>) a, nameof(a));
            if (b is IList<T2>)
                CheckCompatible((IList<T2>) b, nameof(b));
            if (c is IList<T3>)
                CheckCompatible((IList<T3>) c, nameof(c));
            ISparseEnumerable<T> res = fun.Map(a, b, c) as ISparseEnumerable<T>; // Lazy
            var rese = res.GetSparseEnumerator();
            List<ValueAtIndex<T>> newSparseValues;
            if (ReferenceEquals(a, this) || ReferenceEquals(b, this) || ReferenceEquals(c, this))
                newSparseValues = new List<ValueAtIndex<T>>();
            else
            {
                SparseValues.Clear();
                newSparseValues = SparseValues;
            }
            while (rese.MoveNext())
                newSparseValues.Add(new ValueAtIndex<T>(rese.CurrentIndex, rese.Current));
            CommonValue = rese.CommonValue;
            SparseValues = newSparseValues;
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of three sparse collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="fun">The function which maps three elements to an element of this list</param>
        /// <returns></returns>
        public SparseList<T> SetToFunction<T1, T2, T3>(IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, Func<T1, T2, T3, T> fun)
        {
            if ((a is ISparseEnumerable<T1>) && (b is ISparseEnumerable<T2>))
                return SetToFunction((ISparseEnumerable<T1>) a, (ISparseEnumerable<T2>) b, (ISparseEnumerable<T3>) c, fun);

            List<T> fdata = new List<T>();
            var fmap = fun.Map(a, b, c);
            foreach (T elem in fmap)
                fdata.Add(elem);
            SetTo(ToArray(fdata));
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of three other collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="d">The fourth collection</param>
        /// <param name="fun">The function which maps four elements to an element of this list</param>
        /// <returns></returns>
        public SparseList<T> SetToFunction<T1, T2, T3, T4>(
            ISparseList<T1> a, ISparseList<T2> b, ISparseList<T3> c, ISparseList<T4> d, Func<T1, T2, T3, T4, T> fun)
        {
            if (a is IList<T1>)
                CheckCompatible((IList<T1>) a, nameof(a));
            if (b is IList<T2>)
                CheckCompatible((IList<T2>) b, nameof(b));
            if (c is IList<T3>)
                CheckCompatible((IList<T3>) c, nameof(c));
            if (d is IList<T4>)
                CheckCompatible((IList<T4>) d, nameof(d));

            ISparseEnumerable<T> res = fun.Map(a, b, c, d) as ISparseEnumerable<T>; // Lazy
            var rese = res.GetSparseEnumerator();
            List<ValueAtIndex<T>> newSparseValues;
            if (ReferenceEquals(a, this) || ReferenceEquals(b, this) || ReferenceEquals(c, this) || ReferenceEquals(d, this))
                newSparseValues = new List<ValueAtIndex<T>>();
            else
            {
                SparseValues.Clear();
                newSparseValues = SparseValues;
            }
            while (rese.MoveNext())
                newSparseValues.Add(new ValueAtIndex<T>(rese.CurrentIndex, rese.Current));
            CommonValue = rese.CommonValue;
            SparseValues = newSparseValues;
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of four sparse collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="d">The fourth collection</param>
        /// <param name="fun">The function which maps four elements to an element of this list</param>
        /// <returns></returns>
        public SparseList<T> SetToFunction<T1, T2, T3, T4>(IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d, Func<T1, T2, T3, T4, T> fun)
        {
            if ((a is ISparseEnumerable<T1>) && (b is ISparseEnumerable<T2>))
                return SetToFunction((ISparseEnumerable<T1>) a, (ISparseEnumerable<T2>) b, (ISparseEnumerable<T3>) c, (ISparseEnumerable<T4>) d, fun);

            List<T> fdata = new List<T>();
            var fmap = fun.Map(a, b, c, d);
            foreach (T elem in fmap)
                fdata.Add(elem);
            SetTo(ToArray(fdata));
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of this sparse list and another sparse list
        /// x = fun(x,b)
        /// </summary>
        /// <param name="fun">The function which maps (T,T1) to T</param>
        /// <param name="b">The other sparse list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public virtual SparseList<T> SetToFunctionInPlace<T1>(SparseList<T1> b, Func<T, T1, T> fun)
        {
            CheckCompatible(b, nameof(b));
            if (ReferenceEquals(b, this))
            {
                throw new NotSupportedException($"{nameof(b)} must not be equal to this");
            }
            // TODO: consider changing to use enumerators
            int aIndex = 0;
            int bIndex = 0;
            var aSel = GetSparseValue(aIndex);
            var bSel = b.GetSparseValue(bIndex);
            T newCommonValue = fun(CommonValue, b.CommonValue);
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
                    SparseValues.Insert(aIndex, new ValueAtIndex<T> {Index = bSel.Index, Value = fun(CommonValue, bSel.Value)});
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

        internal ValueAtIndex<T> GetSparseValue(int index)
        {
            return index < SparseValues.Count ? SparseValues[index] : ValueAtIndex<T>.NoElement;
        }

        #endregion

        #region Conversions (ToString, ToArray etc.)

        /// <summary>
        /// Converts this sparse list into a human readable string
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return ToString(",");
        }

        /// <summary>
        /// String representation of this list with a specified delimiter
        /// </summary>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public string ToString(string delimiter)
        {
            StringBuilder sb = new StringBuilder();
            if (HasCommonElements)
            {
                sb.Append("[0.." + (Count - 1) + "]=" + CommonValue);
                if (SparseValues.Count > 0) sb.Append(" except " + SparseValues.Count + " values ");
            }
            else
            {
                sb.Append("[Warning: dense sparse list] ");
            }
            int ct = 0;
            foreach (var sel in SparseValues)
            {
                if (ct++ > 0) sb.Append(delimiter);
                sb.Append(sel);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Converts this sparse list to an array
        /// </summary>
        /// <returns></returns>
        public T[] ToArray()
        {
            T[] result = new T[Count];
            CopyTo(result, 0);
            return result;
        }

        /// <summary>
        /// Converts this sparse list to an ordinary non-sparse list
        /// </summary>
        /// <returns></returns>
        public List<T> ToList()
        {
            return new List<T>(this);
        }

        #endregion

        #region ISparseEnumerable<T> Members

        /// <summary>
        /// Gets a sparse enumerator
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<T> GetSparseEnumerator()
        {
            return new SparseListEnumerator<T>(SparseValues, Count, CommonValue);
        }

        #endregion
    }

    /// <summary>
    /// Approximate Sparse List.
    /// </summary>
    /// <typeparam name="T">List element type. Must implement <see cref="Diffable"/>.</typeparam>
    [Serializable]
    [DataContract]
    public class ApproximateSparseList<T> : SparseList<T>, ISparseList<T>, IList<T>, IList, ICloneable
        where T : Diffable
    {
        /// <summary>
        /// The default tolerance for the approximate sparse list
        /// </summary>
        public static double DefaultTolerance = 0.000001;

        /// <summary>
        /// The tolerance for the approximation
        /// </summary>
        public double Tolerance
        {
            get { return Sparsity.Tolerance; }
            protected set { Sparsity = Sparsity.ApproximateWithTolerance(value); }
        }

        /// <summary>
        /// Create a sparse list of given length with elements all equal
        /// to the default value for the element type
        /// </summary>
        /// <param name="count">Number of elements in the list</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        /// <returns></returns>
        public static ApproximateSparseList<T> FromSize(int count, double tolerance)
        {
            var v = new ApproximateSparseList<T>(count, default(T), tolerance);
            return v;
        }

        /// <summary>
        /// Create a sparse list of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in the list</param>
        /// <param name="value">Value for each element</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        /// <returns></returns>
        public static ApproximateSparseList<T> Constant(int count, T value, double tolerance)
        {
            var v = new ApproximateSparseList<T>(count, value, tolerance);
            return v;
        }

        /// <summary>
        /// Constructs a sparse list from a sorted list of sparse elements.
        /// </summary>
        /// <param name="count">Count for result</param>
        /// <param name="commonValue">Common value</param>
        /// <param name="sortedSparseValues">Sorted list of sparse elements</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        [Construction("Count", "CommonValue", "SparseValues", "Tolerance")]
        public static SparseList<T> FromSparseValues(int count, T commonValue,
                                                     List<ValueAtIndex<T>> sortedSparseValues, double tolerance)
        {
            for (int i = 1; i < sortedSparseValues.Count; i++)
            {
                Assert.IsTrue(sortedSparseValues[i].Index > sortedSparseValues[i - 1].Index);
            }

            var result = new ApproximateSparseList<T>
                {
                    CommonValue = commonValue,
                    Count = count,
                    SparseValues = sortedSparseValues,
                    Tolerance = tolerance,
                };
            return result;
        }

        #region Constructors

        /// <summary>
        /// Null constructor.
        /// </summary>
        protected ApproximateSparseList() : this(DefaultTolerance)
        {
        }

        /// <summary>
        /// Constructs an approximate sparse list with the given number of elements
        /// and with default tolerance.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        protected ApproximateSparseList(int count) : this(count, DefaultTolerance)
        {
        }

        /// <summary>
        /// Constructs a sparse list of a given length and assigns all elements the given value.
        /// The tolerance is set to the default value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        protected ApproximateSparseList(int count, T commonValue) : this(count, commonValue, DefaultTolerance)
        {
        }

        /// <summary>
        /// Constructs a sparse list of a given length and assigns all elements the given value, except
        /// for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// The tolerance is set to the default value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="sortedSparseValues">The list of sparse values, which must be sorted by index.</param>
        protected ApproximateSparseList(int count, T commonValue, List<ValueAtIndex<T>> sortedSparseValues)
            : this(count, commonValue, sortedSparseValues, DefaultTolerance)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that">the sparse list to copy into this new sparse list</param>
        protected ApproximateSparseList(ISparseList<T> that) : this(that, DefaultTolerance)
        {
        }

        /// <summary>
        /// Constructs an approximate sparse list with the given tolerance
        /// </summary>
        /// <param name="tolerance">The tolerance for the approximation</param>
        protected ApproximateSparseList(double tolerance) : base()
        {
            this.Tolerance = tolerance;
        }

        /// <summary>
        /// Constructs an approximate sparse list with the given number of elements
        /// and with the given tolerance.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        protected ApproximateSparseList(int count, double tolerance)
            : base(count)
        {
            this.Tolerance = tolerance;
        }

        /// <summary>
        /// Constructs a sparse list of a given length and assigns all elements the given value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        protected ApproximateSparseList(int count, T commonValue, double tolerance)
            : base(count, commonValue)
        {
            this.Tolerance = tolerance;
        }

        /// <summary>
        /// Constructs a sparse list of a given length and assigns all elements the given value, except
        /// for the specified list of sparse values. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally after being passed in.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="sortedSparseValues">The list of sparse values, which must be sorted by index.</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        protected ApproximateSparseList(int count, T commonValue, List<ValueAtIndex<T>> sortedSparseValues, double tolerance)
            : base(count, commonValue, sortedSparseValues)
        {
            this.Tolerance = tolerance;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that">the sparse list to copy into this new sparse list</param>
        /// <param name="tolerance">The tolerance for the approximation</param>
        protected ApproximateSparseList(ISparseList<T> that, double tolerance)
            : base(that)
        {
            this.Tolerance = tolerance;
        }

        #endregion

        private void AdjustToTolerance()
        {
            SparseValues.RemoveAll(item => item.Value.MaxDiff(CommonValue) <= Sparsity.Tolerance);
        }

        #region Element-wise access

        /// <summary>Gets or sets an element.</summary>
        public override T this[int index]
        {
            get
            {
                var k = GetSparseIndex(index);
                if (k < 0) return CommonValue;
                return SparseValues[k].Value;
            }
            set
            {
                var k = GetSparseIndex(index);
                if (value.MaxDiff(CommonValue) <= Sparsity.Tolerance)
                {
                    if (k >= 0) SparseValues.RemoveAt(k);
                    return;
                }
                if (k < 0)
                {
                    SparseValues.Insert(~k, new ValueAtIndex<T> {Index = index, Value = value});
                }
                else
                {
                    var sel = SparseValues[k];
                    sel.Value = value;
                    SparseValues[k] = sel;
                }
            }
        }

        #endregion

        #region Cloning, SetTo operations

        /// <summary>
        /// Copies values from a sparse list to this sparse list.
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(ApproximateSparseList<T> that)
        {
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            CommonValue = that.CommonValue;
            SparseValues = new List<ValueAtIndex<T>>(that.SparseValues);
            Sparsity = that.Sparsity;
        }

        /// <summary>
        /// Copies values from a list which must have the same size as this list,
        /// using the specified common value.
        /// </summary>
        /// <param name="dlist">The list to copy from</param>
        /// <param name="commonValue">Common value</param>
        public override void SetTo(IList<T> dlist, T commonValue)
        {
            SetAllElementsTo(commonValue);
            for (int i = 0; i < dlist.Count; i++)
            {
                T val = dlist[i];
                if (val.MaxDiff(commonValue) > Sparsity.Tolerance) SparseValues.Add(new ValueAtIndex<T> {Index = i, Value = val});
            }
        }

        /// <summary>
        /// Clones this list - return as a sparse list.
        /// </summary>
        /// <returns></returns>
        public new ApproximateSparseList<T> Clone()
        {
            return new ApproximateSparseList<T>(this);
        }

        #endregion

        #region Equality

        /// <summary>
        /// Determines object equality.
        /// </summary>
        /// <param name="obj">Another object.</param>
        /// <returns>True if equal.</returns>
        /// <exclude/>
        public override bool Equals(object obj)
        {
            if (!base.Equals(obj))
                return false;

            ApproximateSparseList<T> that = obj as ApproximateSparseList<T>;
            if (that == null)
                return false;
            return Sparsity == that.Sparsity;
        }

        /// <summary>
        /// Gets a hash code for the instance.
        /// </summary>
        /// <returns>The code.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = base.GetHashCode();
            hash = Hash.Combine(hash, Sparsity.GetHashCode());
            return hash;
        }

        #endregion

        #region General purpose function operators

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of another sparse list
        /// </summary>
        /// <param name="fun">The function which maps from type T2 to type T</param>
        /// <param name="that">The other list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public new ApproximateSparseList<T> SetToFunction<T2>(SparseList<T2> that, Func<T2, T> fun)
        {
            base.SetToFunction(that, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of a sparse collection
        /// </summary>
        /// <param name="fun">The function which maps from type T2 to type T</param>
        /// <param name="that">The other list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public new ApproximateSparseList<T> SetToFunction<T2>(ISparseEnumerable<T2> that, Func<T2, T> fun)
        {
            base.SetToFunction(that, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of a collection
        /// </summary>
        /// <param name="fun">The function which maps from type T2 to type T</param>
        /// <param name="that">The other list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public new ApproximateSparseList<T> SetToFunction<T2>(IEnumerable<T2> that, Func<T2, T> fun)
        {
            base.SetToFunction(that, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this approximate sparse list to a function of the elements of
        /// two other sparse lists
        /// </summary>
        /// <param name="fun">The function which maps two elements to an element of this list</param>
        /// <param name="a">The first list</param>
        /// <param name="b">The second list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are all the same length</remarks>
        public new ApproximateSparseList<T> SetToFunction<T1, T2>(SparseList<T1> a, SparseList<T2> b, Func<T1, T2, T> fun)
        {
            base.SetToFunction(a, b, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of two other sparse lists
        /// </summary>
        /// <param name="a">The first list</param>
        /// <param name="b">The second list</param>
        /// <param name="fun">The function which maps two elements to an element of this list</param>
        /// <returns></returns>
        public new ApproximateSparseList<T> SetToFunction<T1, T2>(
            ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, Func<T1, T2, T> fun)
        {
            base.SetToFunction(a, b, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this list to a function of the elements of two collections
        /// </summary>
        /// <param name="fun">The function which maps the two elements of the other lists to an element of this list</param>
        /// <param name="a">The first list</param>
        /// <param name="b">The second list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public new ApproximateSparseList<T> SetToFunction<T1, T2>(IEnumerable<T1> a, IEnumerable<T2> b, Func<T1, T2, T> fun)
        {
            base.SetToFunction(a, b, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of three sparse collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="fun">The function which maps three elements to an element of this list</param>
        /// <returns></returns>
        public new ApproximateSparseList<T> SetToFunction<T1, T2, T3>(
            ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c, Func<T1, T2, T3, T> fun)
        {
            base.SetToFunction(a, b, c, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of three sparse collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="fun">The function which maps three elements to an element of this list</param>
        /// <returns></returns>
        public new ApproximateSparseList<T> SetToFunction<T1, T2, T3>(IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, Func<T1, T2, T3, T> fun)
        {
            base.SetToFunction(a, b, c, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of three other collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="d">The fourth collection</param>
        /// <param name="fun">The function which maps four elements to an element of this list</param>
        /// <returns></returns>
        public new ApproximateSparseList<T> SetToFunction<T1, T2, T3, T4>(
            ISparseList<T1> a, ISparseList<T2> b, ISparseList<T3> c, ISparseList<T4> d, Func<T1, T2, T3, T4, T> fun)
        {
            base.SetToFunction(a, b, c, d, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this sparse list to a function of the elements of four sparse collections
        /// </summary>
        /// <param name="a">The first collection</param>
        /// <param name="b">The second collection</param>
        /// <param name="c">The third collection</param>
        /// <param name="d">The fourth collection</param>
        /// <param name="fun">The function which maps four elements to an element of this list</param>
        /// <returns></returns>
        public new ApproximateSparseList<T> SetToFunction<T1, T2, T3, T4>(IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d,
                                                                          Func<T1, T2, T3, T4, T> fun)
        {
            base.SetToFunction(a, b, c, d, fun);
            AdjustToTolerance();
            return this;
        }

        /// <summary>
        /// Sets the elements of this approximate sparse list to a function of the elements
        /// of this sparse list and another sparse list x = fun(x,b)
        /// </summary>
        /// <param name="fun">The function which maps (T,T1) to T</param>
        /// <param name="b">The other sparse list</param>
        /// <returns></returns>
        /// <remarks>Assumes the lists are the same length</remarks>
        public new ApproximateSparseList<T> SetToFunctionInPlace<T1>(SparseList<T1> b, Func<T, T1, T> fun)
        {
            base.SetToFunctionInPlace(b, fun);
            AdjustToTolerance();
            return this;
        }

        #endregion
    }

    /// <summary>
    /// Exposes sparse lists. Inherits from <see cref="IList{T}"/>
    /// </summary>
    /// <typeparam name="T">Element type</typeparam>
    public interface ISparseList<T> : IList<T>, ISparseEnumerable<T>
    {
    }

    /// <summary>
    /// Contract for sparse enumeration
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ISparseEnumerator<T> : IEnumerator<T>
    {
        /// <summary>
        /// Current index. If past end of list, current index shows count
        /// </summary>
        int CurrentIndex { get; }

        /// <summary>
        /// Current common value count
        /// </summary>
        int CommonValueCount { get; }

        /// <summary>
        /// The common value for the sparse enumeration
        /// </summary>
        T CommonValue { get; }
    }

    /// <summary>
    /// Sparsely enumerable
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ISparseEnumerable<T> : IEnumerable<T>
    {
        /// <summary>
        /// Returns a sparse enumerator
        /// </summary>
        /// <returns></returns>
        ISparseEnumerator<T> GetSparseEnumerator();
    }

    /// <summary>
    /// Iterator class for sparse lists
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class SparseListEnumerator<T> : SparseEnumeratorBase<T>
    {
        private IList<ValueAtIndex<T>> sparseValues;
        private int totalCount;

        /// <summary>
        /// Constructs a sparse enumerator instance for a sparse list
        /// </summary>
        /// <param name="sparseList"></param>
        /// <param name="totalCount"></param>
        /// <param name="commonValue"></param>
        public SparseListEnumerator(IList<ValueAtIndex<T>> sparseList, int totalCount, T commonValue)
        {
            this.sparseValues = sparseList;
            this.commonValue = commonValue;
            this.totalCount = totalCount;
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (sparseValueCount < sparseValues.Count)
            {
                current = sparseValues[sparseValueCount].Value;
                index = sparseValues[sparseValueCount].Index;
                sparseValueCount++;
                return true;
            }
            index = totalCount;
            return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            index = -1;
            current = default(T);
            sparseValueCount = 0;
        }
    }

    /// <summary>
    /// Iterator base class for sparse enumeration
    /// </summary>
    /// <typeparam name="TRes"></typeparam>
    internal abstract class SparseEnumeratorBase<TRes> : ISparseEnumerator<TRes>
    {
        protected int index;
        protected TRes current;
        protected TRes commonValue;
        protected int sparseValueCount;

        #region ISparseEnumerator<TRes> Members

        /// <summary>
        /// Returns the index of the current sparse element
        /// </summary>
        public int CurrentIndex
        {
            get { return index; }
        }

        /// <summary>
        /// Returns the count of common values up to this point. Once <see cref="MoveNext"/> returns
        /// false, this property gives the total common value count
        /// </summary>
        public int CommonValueCount
        {
            get
            {
                if (index < 0)
                    throw new InvalidOperationException(
                        "The enumerator is positioned before the first or after the last element of the collection.");
                return index - sparseValueCount;
            }
        }

        /// <summary>
        /// Gets the common value for this sparse collection
        /// </summary>
        public TRes CommonValue
        {
            get { return commonValue; }
        }

        #endregion

        #region IEnumerator<TRes> Members

        /// <summary>
        /// Gets the current element in the collection.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">The enumerator is
        /// positioned before the first or after the last element of the collection.</exception>
        public TRes Current
        {
            get
            {
                if (index < 0)
                    throw new InvalidOperationException(
                        "The enumerator is positioned before the first or after the last element of the collection.");
                return current;
            }
        }

        #endregion

        #region IDisposable Members

        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            return;
        }

        #endregion

        #region IEnumerator Members

        object IEnumerator.Current
        {
            get { return (object) Current; }
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public abstract bool MoveNext();

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public virtual void Reset()
        {
            index = -1;
            current = default(TRes);
            sparseValueCount = 0;
        }

        #endregion
    }

    internal class ByIndexComparer<T> : IComparer<ValueAtIndex<T>>
    {
        public int Compare(ValueAtIndex<T> x, ValueAtIndex<T> y)
        {
            return x.Index.CompareTo(y.Index);
        }
    }
}