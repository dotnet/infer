// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1710:IdentifiersShouldHaveCorrectSuffix")]
    public interface ICursorArray<T> : IEnumerable<T>, IEnumerable
    {
        T this[int index] { get; }
        T this[params int[] index] { get; }
        void MoveTo(int index);
        void MoveTo(params int[] index);
        int Count { get; }
        IList<int> Lengths { get; }
        int Rank { get; }
        int GetLength(int dimension);
    }

    // does not compile if you say ICursorArray : ICursorArray<object>
    public interface ICursorArray : IEnumerable
    {
        object this[int index] { get; }
        object this[params int[] index] { get; }
        void MoveTo(int index);
        void MoveTo(params int[] index);
        int Count { get; }
        IList<int> Lengths { get; }
        int Rank { get; }
        int GetLength(int dimension);
    }

    /// <summary>
    /// A multidimensional array of objects which share a single storage block.
    /// </summary>
    /// <remarks><para>
    /// A CursorArray is meant to behave like an ordinary Array, while being
    /// more memory-efficient.  Instead of storing multiple instances of the
    /// same object type, it uses a single instance as a cursor over a block
    /// of data.  The cursor acts like a pointer which is targeted at
    /// the desired part of the array (via its Start property).
    /// </para><para>
    /// The CursorArray object does not hold a pointer to the actual source data.
    /// It is agnostic about the actual type and layout of the 
    /// data that the cursor is walking over, providing a large degree of 
    /// flexibility in the implementation of the cursor.
    /// </para></remarks>
    public class CursorArray<CursorType> : ICursor, ICursorArray<CursorType>, ICursorArray
        where CursorType : ICursor
    {
        // thanks to the where clause, cursor will automatically use the ICursor interface
        protected CursorType cursor;
        // last dimension varies fastest
        protected int[] dim;
        // count[d] = prod_(i > d) dim[i]
        protected int[] count;
        // stride[d] >= count[d] for all d
        protected int[] stride;
        protected int start;

        /// <summary>
        /// Position the cursor at a multidimensional index.
        /// </summary>
        /// <param name="index"></param>
        public void MoveTo(params int[] index)
        {
            int pos = start;
            for (int d = 0; d < dim.Length; d++)
            {
                pos += index[d]*stride[d];
            }
            cursor.Start = pos;
        }

        /// <summary>
        /// Retrieve an object by multidimensional index.
        /// </summary>
        /// <remarks>
        /// The result is a volatile cursor object, which becomes invalid
        /// on the next indexer call.  This can be a source of bugs, e.g.
        /// <c>f(a[i],a[j])</c> will not work.  If you want to save a result 
        /// across calls, you must make a ReferenceClone, as in:
        /// <c>f(a[i].ReferenceClone(), a[j])</c>.
        /// </remarks>
        public CursorType this[params int[] index]
        {
            get
            {
                MoveTo(index);
                return cursor;
            }
        }

        object ICursorArray.this[params int[] index]
        {
            get { return this[index]; }
        }

        /// <summary>
        /// Position the cursor at a linear index.
        /// </summary>
        /// <param name="index"></param>
        public void MoveTo(int index)
        {
            int pos = start;
            for (int d = 0; d < dim.Length; d++)
            {
                int index_d = index/count[d];
                index = index%count[d];
                pos += index_d*stride[d];
            }
            cursor.Start = pos;
            Assert.IsTrue(cursor.Start == pos);
        }

        /// <summary>
        /// Retrieve an object by linear index.
        /// </summary>
        /// <remarks><para>
        /// If the array is multidimensional, this will index the elements 
        /// sequentially in row-major order, i.e. the rightmost dimension varies
        /// fastest.
        /// </para><para>
        /// The result is a volatile cursor object, which becomes invalid
        /// on the next indexer call.  This can be a source of bugs, e.g.
        /// <c>f(a[i],a[j])</c> will not work.  If you want to save a result 
        /// across calls, you must make a ReferenceClone, as in:
        /// <c>f(a[i].ReferenceClone(), a[j])</c>.
        /// </para></remarks>
        public CursorType this[int index]
        {
            get
            {
                MoveTo(index);
                return cursor;
            }
        }

        object ICursorArray.this[int index]
        {
            get { return this[index]; }
        }

        public void LinearIndexToMultidimensionalIndex(int index, IList<int> indexList)
        {
            for (int d = 0; d < dim.Length; d++)
            {
                indexList[d] = index/count[d];
                index = index%count[d];
            }
        }

        /// <summary>
        /// The total number of structures across all dimensions
        /// </summary>
        public int Count
        {
            get { return dim[0]*count[0]; }
        }

        private void ComputeCount(IList<int> counts, int baseCount)
        {
            counts[dim.Length - 1] = baseCount;
            for (int d = dim.Length - 2; d >= 0; d--)
            {
                counts[d] = dim[d + 1]*counts[d + 1];
            }
        }

        /// <summary>
        /// The size of each dimension of the array.
        /// </summary>
        public IList<int> Lengths
        {
            get { return dim; }
            protected set
            {
                dim = new int[value.Count];
                value.CopyTo(dim, 0);
                // this violates the rules of a property, but it is convenient
                UpdateCounts();
            }
        }

        protected void UpdateCounts()
        {
            count = new int[dim.Length];
            ComputeCount(count, 1);
            stride = new int[dim.Length];
            // stride is set to the tightest packing of the dimensions
            ComputeCount(stride, cursor.Count);
        }

        #region Array methods

        /// <summary>
        /// The number of dimensions of the array.
        /// </summary>
        public int Rank
        {
            get { return dim.Length; }
        }

        public int GetLength(int dimension)
        {
            return dim[dimension];
        }

        #endregion

        public int[] Stride
        {
            get { return stride; }
        }

        #region ICursor methods

        int ICursor.Count
        {
            get { return Count*cursor.Count; }
        }

        public int Start
        {
            get { return start; }
            set { start = value; }
        }

        public void CreateSourceArray(int nRecords)
        {
            // a 'record' in this case is 'Count' cursor records.
            cursor.CreateSourceArray(Count*nRecords);
            start = cursor.Start;
            // stride is set to the tightest packing of the dimensions
            ComputeCount(stride, cursor.Count);
        }

        public virtual ICursor ReferenceClone()
        {
            // must use this[0] not cursor
            return new CursorArray<CursorType>((CursorType) this[0].ReferenceClone(), dim, stride);
        }

        #endregion

        #region Copying

        public virtual object Clone()
        {
            // must use this[0] not cursor
            return new CursorArray<CursorType>((CursorType) this[0].ReferenceClone(), dim);
        }

        #endregion

        #region IEnumerable methods

        public IEnumerator<CursorType> GetEnumerator()
        {
            for (int index = 0; index < Count; index++)
            {
                yield return this[index];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region Constructors

        public CursorArray(CursorType cursor)
        {
            this.cursor = cursor;
            start = cursor.Start;
        }

        public CursorArray(CursorType cursor, params int[] lengths)
            : this(cursor)
        {
            Lengths = lengths;
            CreateSourceArray(1);
        }

        // Strides are not necc decreasing, so the physical dimension order need 
        // not match the logical order.
        public CursorArray(CursorType cursor, IList<int> lengths, IList<int> stride)
            : this(cursor)
        {
            Assert.IsTrue(lengths.Count == stride.Count);
            Lengths = lengths;
            this.stride = new int[stride.Count];
            stride.CopyTo(this.stride, 0);
        }

        #endregion

        public bool IsCompatibleWith(Array that)
        {
            if (Rank != that.Rank) return false;
            for (int i = 0; i < Rank; i++)
            {
                if (GetLength(i) != that.GetLength(i)) return false;
            }
            return true;
        }

        public bool IsCompatibleWith(ICursorArray that)
        {
            return Lengths.ValueEquals(that.Lengths);
        }

        public void CheckCompatible(ICursorArray that)
        {
            if (!IsCompatibleWith(that))
                throw new InferRuntimeException("StructArrays are incompatible");
        }

        /// <summary>
        /// Invoke an action for each element of an array.
        /// </summary>
        /// <param name="action">A delegate which accesses the array cursor.</param>
        public void ForEach(Action action)
        {
            foreach (CursorType item in this)
            {
                action();
            }
        }

        /// <summary>
        /// Invoke an element-wise action across two arrays.
        /// </summary>
        /// <param name="that">An array of the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="action">A delegate which accesses the array cursors.</param>
        public void ForEach(ICursorArray that, Action action)
        {
            IEnumerator iter = that.GetEnumerator();
            foreach (CursorType item in this)
            {
                bool ok = iter.MoveNext();
                Assert.IsTrue(ok);
                action();
            }
        }

        /// <summary>
        /// Invoke an element-wise action across three arrays.
        /// </summary>
        /// <param name="a">An array of the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="b">An array of the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="action">A delegate which accesses the array cursors.</param>
        public void ForEach(ICursorArray a, ICursorArray b, Action action)
        {
            IEnumerator a_iter = a.GetEnumerator();
            IEnumerator b_iter = b.GetEnumerator();
            foreach (CursorType item in this)
            {
                bool a_ok = a_iter.MoveNext();
                bool b_ok = b_iter.MoveNext();
                Assert.IsTrue(a_ok);
                Assert.IsTrue(b_ok);
                action();
            }
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}