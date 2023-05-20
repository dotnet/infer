// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Math
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// A one-dimensional vector of double values, optimised for the case where many contiguous ranges
    /// of elements have the same value.
    /// </summary>
    [Serializable]
    [DataContract]
    public class PiecewiseVector : Vector, IList<double>, ICloneable, CanSetAllElementsTo<double>, SettableTo<Vector>
    {
        #region Properties
        [DataMember]
        private List<ConstantVector> pieces;

        [DataMember]
        private int count;

        /// <summary>
        /// A list of the pieces of this vector.
        /// </summary>
        public List<ConstantVector> Pieces
        {
            get { return pieces; }
        }

        /// <summary>
        /// The value of all elements which are not in any of the vector pieces.
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

        #endregion

        #region Factory methods and constructors

        /// <summary>
        /// Create a piecewise vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <returns></returns>
        public new static PiecewiseVector Zero(int count)
        {
            return new PiecewiseVector(count);
        }

        /// <summary>
        /// Create a piecewise vector of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <returns></returns>
        public new static PiecewiseVector Constant(int count, double value)
        {
            var v = new PiecewiseVector(count, value);
            return v;
        }

        /// <summary>
        /// Creator a piecewise vector as a copy of another vector (of any type)
        /// </summary>
        /// <param name="that">The source vector - can be dense or sparse</param>
        public new static PiecewiseVector Copy(Vector that)
        {
            var v = new PiecewiseVector(that.Count);
            if (that.Sparsity.IsPiecewise)
                v.SetTo((PiecewiseVector) that);
            else
                v.SetTo(that);
            return v;
        }

        /// <summary>
        /// Constructs a piecewise vector from a dense array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        public new static PiecewiseVector FromArray(params double[] data)
        {
            var v = new PiecewiseVector(data.Length);
            v.SetTo(data);
            return v;
        }

        /// <summary>
        /// Constructs a piecewise vector from a sorted list of subvectors, which will
        /// be used directly and not copied.
        /// </summary>
        /// <param name="count">Count for result</param>
        /// <param name="commonValue">Common value</param>
        /// <param name="sortedDisjointVectors">Sorted list of disjoint subvectors</param>
        [Construction("Count", "CommonValue", "Pieces")]
        public static PiecewiseVector FromSubvectors(int count, double commonValue, List<ConstantVector> sortedDisjointVectors)
        {
            PiecewiseVector result = new PiecewiseVector(count, commonValue, sortedDisjointVectors);
            return result;
        }

        #endregion

        #region Append

        /// <summary>
        /// Appends an item to a vector - returns a new piecewise vector
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public override Vector Append(double item)
        {
            var result = PiecewiseVector.Constant(Count + 1, CommonValue);
            result.pieces.AddRange(pieces);
            result[Count] = item;
            return result;
        }

        /// <summary>
        /// Returns a new vector which appends a second vector to this vector
        /// </summary>
        /// <param name="second">Second vector</param>
        /// <returns></returns>
        /// <remarks>If the second vector is dense then the result becomes dense</remarks>
        public override Vector Append(Vector second)
        {
            int jointCount = Count + second.Count;
            if (second.Sparsity.IsPiecewise)
            {
                var secondp = (PiecewiseVector) second;
                if (secondp.CommonValue == CommonValue)
                {
                    var result = PiecewiseVector.Constant(jointCount, CommonValue);
                    result.pieces.AddRange(pieces);
                    foreach (var sub in secondp.pieces)
                    {
                        var newvec = new ConstantVector(sub.Start + Count, sub.End + Count, sub.Value);
                        result.InsertPiece(newvec, result.pieces.Count);
                    }
                    return result;
                }
            }
            DenseVector result2 = DenseVector.Constant(jointCount, CommonValue);
            foreach (var sub in pieces)
                for (int i = sub.Start; i <= sub.End; i++) result2[i] = sub.Value;
            for (int i = Count, j = 0; i < jointCount; i++)
                result2[i] = second[j++];
            return result2;
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Constructs a zero vector with the given number of elements.
        /// </summary>
        protected PiecewiseVector()
        {
            Sparsity = Sparsity.Piecewise;
            pieces = new List<ConstantVector>();
        }

        /// <summary>
        /// Constructs a zero vector with the given number of elements.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        protected PiecewiseVector(int count) : this()
        {
            Count = count;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        protected PiecewiseVector(int count, double commonValue)
            : this(count)
        {
            CommonValue = commonValue;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value, except
        /// for the specified list of vectors. This list is stored internally as is
        /// so MUST be sorted by index and must not be modified externally.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        /// <param name="commonValue">The value to assign to all elements.</param>
        /// <param name="sortedVectors">The list of vectors, which must be disjoint and sorted by index.</param>
        protected PiecewiseVector(int count, double commonValue, List<ConstantVector> sortedVectors)
            : this(count, commonValue)
        {
            pieces = sortedVectors;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that">the vector to copy into this new vector</param>
        protected PiecewiseVector(PiecewiseVector that)
            : this(that.Count)
        {
            pieces = new List<ConstantVector>();
            SetTo(that);
        }

        /// <summary>
        /// Creates a piecewise vector from a list of doubles.
        /// </summary>
        /// <param name="dlist">the list of doubles</param>
        protected PiecewiseVector(IList<double> dlist) : this(dlist.Count)
        {
            pieces = new List<ConstantVector>();
            SetTo(dlist);
        }

        #endregion

        #region Element-wise access

        /// <summary>Gets and sets an element.</summary>
        public override double this[int index]
        {
            get
            {
                ConstantVector? vector = GetSubvector(index);
                if (!vector.HasValue) return CommonValue;
                return vector.Value.Value;
            }

            set
            {
                // todo: make this take less than linear time
                int i;
                for (i = 0; i < pieces.Count; i++)
                {
                    var sub = pieces[i];
                    if (sub.End < index) continue;
                    if (sub.Start <= index)
                    {
                        // In this piece

                        // Check if already at correct value
                        if (sub.Value == value) return;

                        var newvec = new ConstantVector(index, index, value);
                        if (sub.Start == index)
                        {
                            int j = InsertPiece(newvec, i);
                            if (sub.End == index)
                            {
                                pieces.RemoveAt(j + 1);
                            }
                            else
                            {
                                sub.Start++;
                                pieces[j + 1] = sub;
                            }
                        }
                        else
                        {
                            int oldSubEnd = sub.End;
                            sub.End = index - 1;
                            pieces[i] = sub;
                            int j = InsertPiece(newvec, i + 1);
                            if (index < oldSubEnd)
                            {
                                var newvec2 = new ConstantVector(index + 1, oldSubEnd, sub.Value);
                                InsertPiece(newvec2, j + 1);
                            }
                        }
                        return;
                    }
                    else
                    {
                        break;
                    }
                }
                // Not in an existing piece
                if (value == CommonValue) return;

                // Add a new singleton piece
                InsertPiece(new ConstantVector(index, index, value), i);
            }
        }

        #endregion

        #region Piecewise vector-related operations

        /// <summary>
        /// Gets the subvector that contains the specified index, or null if none.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected ConstantVector? GetSubvector(int index)
        {
            int i = pieces.BinarySearch(new ConstantVector {Start = index});
            if (i >= 0)
            {
                // there is a piece which starts at exactly 'index'
                // so return it.
                return pieces[i];
            }
            else
            {
                int k = ~i; // this is the index where a piece would be inserted
                if (k > 0) // there is a piece whose start index is below 'index'
                {
                    // get the closest piece whose start index is below 'index'
                    var piece = pieces[k - 1];
                    if (piece.Contains(index)) return piece;
                }
                return null;
            }
        }

        /// <summary>
        /// Gets the number of elements that don't belong to any subvector
        /// and so take the common value.
        /// </summary>
        protected int GetCommonValueCount()
        {
            int count = 0;
            for (int i = 0; i < this.pieces.Count; ++i)
            {
                count += this.pieces[i].Count;
            }

            return this.Count - count;
        }

        /// <summary>
        /// Gets whether there are any elements that don't belong to a subvector
        /// and so take the common value.
        /// </summary>
        public bool HasCommonElements()
        {
            int nextStartIndex = 0;
            for (int i = 0; i < this.pieces.Count; ++i)
            {
                if (this.pieces[i].Start != nextStartIndex)
                {
                    return true;
                }

                nextStartIndex = this.pieces[i].End + 1;
            }

            return nextStartIndex != this.Count;
        }

        /// <summary>
        /// Gets the dense index of the first common element.
        /// </summary>
        /// <returns>Returns the dense index of the first common element or -1 if there are no common elements</returns>
        public int GetFirstCommonIndex()
        {
            int index = 0;
            for (int i = 0; i < this.pieces.Count; ++i)
            {
                ConstantVector piece = this.pieces[i];
                if (!piece.Contains(index))
                {
                    return index;
                }

                index = piece.End + 1;
            }

            return index < this.Count ? index : -1;
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
            foreach (var sub in pieces)
            {
                while (index < sub.Start)
                {
                    yield return CommonValue;
                    index++;
                }
                while (index <= sub.End)
                {
                    yield return sub.Value;
                    index++;
                }
            }
            while (index < Count)
            {
                yield return CommonValue;
                index++;
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
        /// An action that takes a range and two values.
        /// </summary>
        /// <param name="start">The start of the range</param>
        /// <param name="end">The end of the range</param>
        /// <param name="value1">The first value</param>
        /// <param name="value2">The second value</param>
        protected delegate void RangeFunc(int start, int end, double value1, double value2);

        /// <summary>
        /// Applies a function to ranges in common between this vector and another
        /// and returns the number of elements with common values in both vectors.
        /// </summary>
        /// <param name="b"></param>
        /// <param name="func"></param>
        protected int ApplyRangeFunction(PiecewiseVector b, RangeFunc func)
        {
            CheckCompatible(b, nameof(b));
            var r1 = new ConstantVector();
            var r2 = new ConstantVector();
            int commonCount = 0;
            int lastIndex = 0;

            var enum1 = pieces.GetEnumerator();
            bool moveNext1 = enum1.MoveNext();
            if (moveNext1) r1 = enum1.Current;

            var enum2 = b.pieces.GetEnumerator();
            bool moveNext2 = enum2.MoveNext();
            if (moveNext2) r2 = enum2.Current;

            while (moveNext1 && moveNext2)
            {
                if (r1.End < r2.Start)
                {
                    // This range occurs entirely before that one, add it
                    func(r1.Start, r1.End, r1.Value, b.CommonValue);
                    if (r1.Start > lastIndex) commonCount += r1.Start - lastIndex;
                    lastIndex = r1.End + 1;
                    moveNext1 = enum1.MoveNext();
                    if (moveNext1) r1 = enum1.Current;
                    continue;
                }
                if (r2.End < r1.Start)
                {
                    // That range occurs entirely before this one, add it
                    func(r2.Start, r2.End, CommonValue, r2.Value);
                    if (r2.Start > lastIndex) commonCount += r2.Start - lastIndex;
                    lastIndex = r2.End + 1;
                    moveNext2 = enum2.MoveNext();
                    if (moveNext2) r2 = enum2.Current;
                    continue;
                }

                // Ranges overlap
                // Find the range that starts first
                int start = r1.Start;
                if (r1.Start < r2.Start)
                {
                    func(r1.Start, r2.Start - 1, r1.Value, b.CommonValue);
                    start = r2.Start;
                }
                else if (r1.Start > r2.Start)
                {
                    func(r2.Start, r1.Start - 1, CommonValue, r2.Value);
                }
                if (start > lastIndex) commonCount += start - lastIndex;

                // Find the range that ends first
                if (r1.End < r2.End)
                {
                    func(start, r1.End, r1.Value, r2.Value);
                    r2.Start = r1.End + 1;
                    lastIndex = r2.Start;
                    moveNext1 = enum1.MoveNext();
                    if (moveNext1) r1 = enum1.Current;
                }
                else if (r1.End > r2.End)
                {
                    func(start, r2.End, r1.Value, r2.Value);
                    r1.Start = r2.End + 1;
                    lastIndex = r1.Start;
                    moveNext2 = enum2.MoveNext();
                    if (moveNext2) r2 = enum2.Current;
                }
                else
                {
                    func(start, r1.End, r1.Value, r2.Value);
                    lastIndex = r1.End + 1;
                    moveNext1 = enum1.MoveNext();
                    moveNext2 = enum2.MoveNext();
                    if (moveNext1) r1 = enum1.Current;
                    if (moveNext2) r2 = enum2.Current;
                }
            }

            while (moveNext1)
            {
                func(r1.Start, r1.End, r1.Value, b.CommonValue);
                if (r1.Start > lastIndex) commonCount += r1.Start - lastIndex;
                lastIndex = r1.End + 1;
                moveNext1 = enum1.MoveNext();
                if (moveNext1) r1 = enum1.Current;
            }

            while (moveNext2)
            {
                func(r2.Start, r2.End, CommonValue, r2.Value);
                if (r2.Start > lastIndex) commonCount += r2.Start - lastIndex;
                lastIndex = r2.End + 1;
                moveNext2 = enum2.MoveNext();
                if (moveNext2) r2 = enum2.Current;
            }
            if (lastIndex < Count) commonCount += Count - lastIndex;
            return commonCount;
        }

        #endregion

        #region Cloning, SetTo operations

        /// <summary>
        /// Sets all elements to a given value.
        /// </summary>
        /// <param name="value">The new value.</param>
        public override void SetAllElementsTo(double value)
        {
            pieces.Clear();
            CommonValue = value;
        }

        /// <summary>
        /// Copies values from an array. The minimum value is used as the common value
        /// </summary>
        /// <param name="values">An array whose length matches <c>this.Count</c>.</param>
        public override void SetTo(double[] values)
        {
            double commonValue = Double.PositiveInfinity;
            foreach (double d in values) commonValue = System.Math.Min(commonValue, d);
            SetAllElementsTo(commonValue);
            for (int i = 0; i < values.Length; i++)
            {
                double val = values[i];
                if (val == commonValue) continue;
                InsertPiece(new ConstantVector(i, i, val), pieces.Count);
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
            if (that.Sparsity.IsPiecewise)
            {
                SetTo((PiecewiseVector) that);
                return;
            }
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            // todo: make efficient
            SetTo(that.ToArray());
        }

        /// <summary>
        /// Copies values from a piecewise vector to this piecewise vector.
        /// </summary>
        /// <param name="that"></param>
        public virtual void SetTo(PiecewiseVector that)
        {
            if (Object.ReferenceEquals(this, that)) return;
            CheckCompatible(that, nameof(that));
            CommonValue = that.CommonValue;
            pieces = new List<ConstantVector>(that.pieces);
        }

        /// <summary>
        /// Sets the vector to a constant value between the specified
        /// start and end indices inclusive (and zero elsewhere).
        /// </summary>
        /// <param name="start">The start index</param>
        /// <param name="end">The end index</param>
        /// <param name="value">The constant value</param>
        public void SetToConstantInRange(int start, int end, double value)
        {
            SetAllElementsTo(0);
            if (value != 0) pieces.Add(new ConstantVector(start, end, value));
        }

        /// <summary>
        /// Sets the vector to a constant value in multiple ranges (and zero elsewhere).
        /// The start and end points of the ranges are specified as consecutive pairs
        /// in a single enumerable which must therefore have even length.
        /// </summary>
        /// <param name="startEndPairs">Enumerable containing pairs of start and end values</param>
        /// <param name="value"></param>
        public void SetToConstantInRanges(IEnumerable<int> startEndPairs, double value)
        {
            if (value == 0)
            {
                SetAllElementsTo(0);
                return;
            }
            var newVectors = new List<ConstantVector>();
            var en = startEndPairs.GetEnumerator();
            while (en.MoveNext())
            {
                int start = en.Current;
                if (!en.MoveNext()) throw new ArgumentException("Start/end pair enumerable must have even length.");
                int end = en.Current;
                newVectors.Add(new ConstantVector(start, end, value));
            }
            newVectors.Sort();
            for (int i = 1; i < newVectors.Count; i++)
            {
                if (newVectors[i].End <= newVectors[i - 1].Start)
                {
                    throw new ArgumentException("Ranges must not overlap.");
                }
            }
            SetInternal(0, newVectors);
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
                if (val == commonValue) continue;
                InsertPiece(new ConstantVector(i, i, val), pieces.Count);
            }
        }

        /// <summary>
        /// Inserts a piece into the pieces list, merging it with
        /// the preceding piece if they are adjacent and have the same value.
        /// </summary>
        /// <remarks>
        /// The calling code must ensure that 'index' is set to ensure
        /// that 'pieces' remains in the correct sorted order.
        /// This code just handles any merging necessary.
        /// </remarks>
        /// <param name="sub">The piece to insert</param>
        /// <param name="index">The index to insert at</param>
        /// <returns>The index of the piece added or merged</returns>
        private int InsertPiece(ConstantVector sub, int index)
        {
            int i = sub.Start;
            if (index > 0)
            {
                var cv = pieces[index - 1];
                // Check whether to merge with the previous piece
                if ((cv.End == i - 1) && (cv.Value == sub.Value))
                {
                    cv.End = sub.End;
                    pieces[index - 1] = cv;
                    return index - 1;
                }
                // todo: check whether to merge with the next piece
            }
            pieces.Insert(index, sub);
            return index;
        }

        /// <summary>
        /// Copies values from an Enumerable to this vector
        /// </summary>
        /// <param name="that"></param>
        public override void SetTo(IEnumerable<double> that)
        {
            SetTo(that.ToArray());
        }

        /// <summary>
        /// Clones this vector - return as a vector
        /// </summary>
        /// <returns></returns>
        public override Vector Clone()
        {
            return new PiecewiseVector(this);
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
            if (ReferenceEquals(that, null)) return false;
            if (Count != that.Count) return false;
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
            foreach (var sel in pieces) hash = Hash.Combine(hash, sel.GetHashCode());
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

            var absdiff = new PiecewiseVector(Count);
            absdiff.SetToFunction(this, that, (a, b) => MMath.AbsDiffAllowingNaNs(a, b));
            return absdiff.Max();
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that, double rel)
        {
            var absdiff = new PiecewiseVector(Count);
            absdiff.SetToFunction(this, that, (a, b) => MMath.AbsDiffAllowingNaNs(a, b, rel));
            return absdiff.Max();
        }

        #endregion

        #region LINQ-like operators (All, Any, FindAll etc.)

        /// <inheritdoc/>
        public override bool All(Func<double, bool> fun)
        {
            if (this.HasCommonElements() && !fun(this.CommonValue))
            {
                return false;
            }

            for (int i = 0; i < this.pieces.Count; ++i)
            {
                if (!fun(this.pieces[i].Value))
                {
                    return false;
                }
            }

            return true;
        }

        /// <inheritdoc/>
        public override bool Any(Func<double, bool> fun)
        {
            if (this.HasCommonElements() && fun(this.CommonValue))
            {
                return true;
            }

            for (int i = 0; i < this.pieces.Count; ++i)
            {
                if (fun(this.pieces[i].Value))
                {
                    return true;
                }
            }

            return false;
        }

        /// <inheritdoc/>
        public override bool Any(Vector that, Func<double, double, bool> fun)
        {
            if (that.Sparsity.IsPiecewise)
            {
                return this.Any((PiecewiseVector)that, fun);
            }

            return base.Any(that, fun);
        }

        /// <inheritdoc cref="Any(Vector, Func{double, double, bool})"/>
        public bool Any(PiecewiseVector that, Func<double, double, bool> fun)
        {
            bool any = false;
            this.ApplyRangeFunction(that, (start, end, value1, value2) => { any |= fun(value1, value2); });
            return any;
        }

        /// <inheritdoc/>
        public override IEnumerable<ValueAtIndex<double>> FindAll(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            bool funIsTrueForCommonValue = fun(this.CommonValue);
            int index = 0;
            foreach (var constantVector in this.pieces)
            {
                for (; index < constantVector.Start && funIsTrueForCommonValue; ++index)
                {
                    yield return new ValueAtIndex<double>(index, this.CommonValue);
                }

                if (fun(constantVector.Value))
                {
                    for (index = constantVector.Start; index <= constantVector.End; ++index)
                    {
                        yield return new ValueAtIndex<double>(index, constantVector.Value);
                    }
                }

                index = constantVector.End + 1;
            }

            for (; index < this.Count && funIsTrueForCommonValue; index++)
            {
                yield return new ValueAtIndex<double>(index, this.CommonValue);
            }
        }

        /// <inheritdoc/>
        public override int CountAll(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            int totalPieceLength = 0;
            int result = 0;
            foreach (var constantVector in this.pieces)
            {
                totalPieceLength += constantVector.Count;
                if (fun(constantVector.Value))
                {
                    result += constantVector.Count;
                }
            }

            int commonValueCount = this.Count - totalPieceLength;
            if (commonValueCount > 0 && fun(this.CommonValue))
            {
                result += commonValueCount;
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
            foreach (var constantVector in this.pieces)
            {
                if (index < constantVector.Start && funIsTrueForCommonValue)
                {
                    firstIndex = index;
                    break;
                }

                if (fun(constantVector.Value))
                {
                    firstIndex = constantVector.Start;
                    break;
                }

                index = constantVector.End + 1;
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
            var piecesInReverseOrder = Enumerable.Reverse(this.Pieces);
            foreach (var constantVector in piecesInReverseOrder)
            {
                if (index > constantVector.End && funIsTrueForCommonValue)
                {
                    lastIndex = index;
                    break;
                }

                if (fun(constantVector.Value))
                {
                    lastIndex = constantVector.End;
                    break;
                }

                index = constantVector.Start - 1;
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
            int commonValueCount = this.GetCommonValueCount();
            for (int i = 0; i < commonValueCount; i++)
                result = fun(result, CommonValue);
            foreach (var sub in pieces)
            {
                for (int i = sub.Start; i <= sub.End; i++)
                {
                    result = fun(result, sub.Value);
                }
            }
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
        public double Reduce(double initial, PiecewiseVector that, Func<double, double, double, double> fun)
        {
            // todo: make more efficient
            return base.Reduce(initial, that, fun);
        }

        /// <summary>
        /// Reduce method which can take advantage of piecewise structure. Operates on this list
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="repeatedFun">Function which computes the reduction function applied multiple times</param>
        /// <returns></returns>
        /// <remarks>This method does not take advantage of this list's sparseness.</remarks>
        public TRes Reduce<TRes>(TRes initial, Func<TRes, double, int, TRes> repeatedFun)
        {
            TRes result = initial;
            result = repeatedFun(result, this.CommonValue, this.GetCommonValueCount());
            foreach (var sub in pieces)
                result = repeatedFun(result, sub.Value, sub.Count);
            return result;
        }

        /// <summary>
        /// Returns the sum of all elements.
        /// </summary>
        public override double Sum()
        {
            int count = 0;
            double sum = 0;
            foreach (var sub in pieces)
            {
                int ct = sub.Count;
                count += ct;
                sum += ct*sub.Value;
            }
            sum += (Count - count)*CommonValue;
            return sum;
        }

        /// <summary>
        /// Returns the sum of a function of all elements.
        /// </summary>
        /// <param name="fun">Conversion function</param>
        public override double Sum(Converter<double, double> fun)
        {
            int count = 0;
            double sum = 0;
            foreach (var sub in pieces)
            {
                int ct = sub.Count;
                count += ct;
                sum += ct*fun(sub.Value);
            }
            sum += (Count - count)*fun(CommonValue);
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
            if (that.Sparsity.IsPiecewise)
                return Sum(fun, (PiecewiseVector) that, cond);
            return base.Sum(fun, that, cond);
        }

        /// <summary>
        /// Returns the sum of a function of this piecewise vector filtered by a function of a second piecewise vector.
        /// </summary>
        /// <param name="fun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="cond">Function to convert the elements of that vector to give the filter condition</param>
        /// <returns>The filtered and mapped sum</returns>
        public double Sum(Converter<double, double> fun, PiecewiseVector that, Converter<double, bool> cond)
        {
            double sum = 0.0;
            int commonCount = ApplyRangeFunction(that,
                                                 (start, end, value1, value2) => { if (cond(value2)) sum += (1 + end - start)*fun(value1); });
            if (cond(that.CommonValue))
            {
                sum += commonCount*fun(CommonValue);
            }
            return sum;
        }

        /// <summary>
        /// Returns the sum of over zero-based index times element.
        /// </summary>
        public override double SumI()
        {
            double result = (Count - 1)*Count*CommonValue;
            foreach (var sub in pieces)
            {
                result += (sub.Value - CommonValue)*(((double)sub.End) + sub.Start)*sub.Count;
            }
            return 0.5*result;
        }

        /// <summary>
        /// Returns the sum of over square of index^2 times element.
        /// </summary>
        public override double SumISq()
        {
            double result = (Count - 1)*Count*(2.0*Count - 1.0)*CommonValue;
            foreach (var sub in pieces)
            {
                double endfun = sub.End*(sub.End + 1)*(2*sub.End + 1);
                double startfun = 0;
                if (sub.Start > 1) startfun = (sub.Start - 1)*sub.Start*(2*sub.Start - 1);
                result += (sub.Value - CommonValue)*(endfun - startfun);
            }
            return result/6.0;
        }

        /// <summary>
        /// Returns the maximum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public override double Max()
        {
            double max = double.NegativeInfinity;
            if (this.HasCommonElements()) max = CommonValue;
            foreach (var sub in pieces) max = System.Math.Max(max, sub.Value);
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
            if (this.HasCommonElements()) max = fun(CommonValue);
            foreach (var sub in pieces) max = System.Math.Max(max, fun(sub.Value));
            return max;
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public override double Min()
        {
            double min = double.PositiveInfinity;
            if (this.HasCommonElements()) min = CommonValue;
            foreach (var sub in pieces) min = System.Math.Min(min, sub.Value);
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
            if (this.HasCommonElements()) min = fun(CommonValue);
            foreach (var sub in pieces) min = System.Math.Min(min, fun(sub.Value));
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
            int ct = this.GetCommonValueCount();
            if (ct > 0)
                Z += System.Math.Exp(CommonValue - max)* ct;
            foreach (var sub in pieces)
                Z += System.Math.Exp(sub.Value - max)* sub.Count;
            return System.Math.Log(Z) + max;
        }

        /// <summary>
        /// Returns the index of the minimum element.
        /// </summary>
        /// <returns>The index of the minimum element.</returns>
        public override int IndexOfMinimum()
        {
            bool hasCommon = this.HasCommonElements();
            double minValue = hasCommon ? this.CommonValue : double.PositiveInfinity;
            int minPos = 0;
            for (int i = 0; i < this.pieces.Count; ++i)
            {
                ConstantVector piece = this.pieces[i];
                if (minValue > piece.Value)
                {
                    minValue = piece.Value;
                    minPos = piece.Start;
                }
            }

            if (minValue == this.CommonValue && hasCommon)
            {
                minPos = this.GetFirstCommonIndex();
            }

            return minPos;
        }

        /// <summary>
        /// Returns the index of the maximum element.
        /// </summary>
        /// <returns>The index of the maximum element.</returns>
        public override int IndexOfMaximum()
        {
            bool hasCommon = this.HasCommonElements();
            double maxValue = hasCommon ? this.CommonValue : double.NegativeInfinity;
            int maxPos = 0;
            for (int i = 0; i < this.pieces.Count; ++i)
            {
                ConstantVector piece = this.pieces[i];
                if (maxValue < piece.Value)
                {
                    maxValue = piece.Value;
                    maxPos = piece.Start;
                }
            }

            if (maxValue == this.CommonValue && hasCommon)
            {
                maxPos = this.GetFirstCommonIndex();
            }

            return maxPos;
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

            foreach (var sub in this.pieces)
            {
                // Add up the common values sitting between the pieces
                nextSum = (sub.Start - position)*this.CommonValue + sum;
                if (nextSum > targetSum)
                {
                    result = (int) ((targetSum - sum)/this.CommonValue) + position;
                    break;
                }

                // Add up the elements of the current piece
                sum = nextSum;
                nextSum += sub.Count*sub.Value;
                if (nextSum > targetSum)
                {
                    result = (int) ((targetSum - sum)/sub.Value) + sub.Start;
                    break;
                }

                sum = nextSum;
                position = sub.End + 1;
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
            if (that.Sparsity.IsPiecewise)
                return Inner((PiecewiseVector) that);
            return base.Inner(that);
        }

        /// <summary>
        /// Returns the inner product of this piecewise vector with another piecewise vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns>Their inner product.</returns>
        public double Inner(PiecewiseVector that)
        {
            double sum = 0.0;
            int ct = ApplyRangeFunction(that,
                                        (start, end, value1, value2) => sum += (1 + end - start)*(value1*value2));
            sum += ct*CommonValue*that.CommonValue;
            return sum;
        }

        /// <summary>
        /// Returns the inner product of this vector with a function of a second vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Vector that, Converter<double, double> fun)
        {
            if (that.Sparsity.IsPiecewise)
                return Inner((PiecewiseVector) that, fun);
            return base.Inner(that, fun);
        }

        /// <summary>
        /// Returns the inner product of this piecewise vector with a function of a second piecewise vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public double Inner(PiecewiseVector that, Converter<double, double> fun)
        {
            double sum = 0.0;
            int ct = ApplyRangeFunction(that,
                                        (start, end, value1, value2) => sum += (1 + end - start)*(value1*fun(value2)));
            sum += ct*CommonValue*fun(that.CommonValue);
            return sum;
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
            if (that.Sparsity.IsPiecewise)
                return Inner(thisFun, (PiecewiseVector) that, thatFun);

            return base.Inner(thisFun, that, thatFun);
        }

        /// <summary>
        /// Returns the inner product of a function of this piecewise vector with a function of a second piecewise vector.
        /// </summary>
        /// <param name="thisFun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="thatFun">Function to convert the elements of that vector</param>
        /// <returns>Their inner product.</returns>
        public double Inner(Converter<double, double> thisFun, PiecewiseVector that, Converter<double, double> thatFun)
        {
            double sum = 0.0;
            int ct = ApplyRangeFunction(that,
                                        (start, end, value1, value2) => sum += (1 + end - start)*(thisFun(value1)*thatFun(value2)));
            sum += ct*thisFun(CommonValue)*thatFun(that.CommonValue);
            return sum;
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
                if (val != min) this[i] = val;
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
            for (int i = 0; i < pieces.Count; i++)
            {
                var vec = pieces[i];
                vec.Value *= scale;
                pieces[i] = vec;
            }
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
            if (CommonValue == value)
            {
                // At least one element has the common value
                if (this.HasCommonElements()) return true;
            }
            foreach (var sub in pieces)
            {
                if (sub.Value == value) return true;
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
            int index = -1;
            if (item == CommonValue)
            {
                index = GetFirstCommonIndex();
            }
            foreach (var sub in pieces)
            {
                if (sub.Value == item)
                {
                    if (index == -1) return sub.Start;
                    return System.Math.Min(index, sub.Start);
                }
            }
            return index;
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
            foreach (var sub in pieces)
            {
                j = index + sub.Start;
                for (int i = 0; i < sub.Count; i++, j++)
                    array[j] = sub.Value;
            }
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
            if (that.Sparsity.IsPiecewise)
                return SetToFunction((PiecewiseVector) that, fun);
            return base.SetToFunction(that, fun);
        }

        /// <summary>
        /// Sets the elements of this piecewise vector to a function of the elements of another piecewise vector
        /// </summary>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <param name="that">The other vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual PiecewiseVector SetToFunction(PiecewiseVector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            CommonValue = fun(that.CommonValue);
            pieces = that.pieces.ConvertAll(x =>
                                            new ConstantVector(x.Start, x.End, fun(x.Value)));
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
            if ((a.Sparsity.IsPiecewise) && (b.Sparsity.IsPiecewise))
                return SetToFunction((PiecewiseVector) a, (PiecewiseVector) b, fun);
            return base.SetToFunction(a, b, fun);
        }

        /// <summary>
        /// Sets the elements of this piecewise vector to a function of the elements of two other piecewise vectors
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual PiecewiseVector SetToFunction(PiecewiseVector a, PiecewiseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(a, nameof(a));
            var newVectors = new List<ConstantVector>();
            double newCommonValue = fun(a.CommonValue, b.CommonValue);
            a.ApplyRangeFunction(b, (start, end, value1, value2) =>
            {
                double f = fun(value1, value2);
                if (f != newCommonValue)
                {
                    newVectors.Add(new ConstantVector(start, end, fun(value1, value2)));
                }
            });
            SetInternal(newCommonValue, newVectors);
            return this;
        }

        private void SetInternal(double newCommonValue, List<ConstantVector> newVectors)
        {
            SetAllElementsTo(newCommonValue);
            foreach (var vec in newVectors) InsertPiece(vec, pieces.Count);
        }

        /// <summary>
        /// Sets the elements of this piecewise vector to a function of the elements of this piecewise vector and another piecewise vectors
        /// x = fun(x,b)
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public virtual PiecewiseVector SetToFunctionInPlace(PiecewiseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(b, nameof(b));
            if (ReferenceEquals(b, this))
            {
                throw new NotSupportedException("b must not be equal to this");
            }
            throw new NotImplementedException("SetToFunctionInPlace() is not yet implemented");
        }

        #endregion

        #region Conversions (ToString, ToArray etc.)

        /// <summary>
        /// String representation of vector with a specified format for each element
        /// </summary>
        /// <returns></returns>
        public override string ToString(string format)
        {
            return ToString(format, ",");
        }

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// </summary>
        /// <param name="format"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public override string ToString(string format, string delimiter)
        {
            return ToString(format, delimiter, i => i.ToString(CultureInfo.InvariantCulture));
        }

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// and a function for converting integers to display strings.
        /// </summary>
        public override string ToString(string format, string delimiter, Func<int, string> intToString)
        {
            StringBuilder sb = new StringBuilder();
            if ((format == null) && (pieces.Count > 1)) sb.Append("[");
            int ct = 0;
            foreach (var sub in pieces)
            {
                if (ct++ > 0) sb.Append(delimiter);
                sb.Append(sub.ToString(format, intToString));
            }
            if (this.HasCommonElements() && (CommonValue != 0))
            {
                if (ct > 0) sb.Append(delimiter);
                if (format == null)
                {
                    sb.Append("?");
                }
                else
                {
                    sb.Append("[?]=" + CommonValue.ToString(format));
                }
            }
            if ((format == null) && (pieces.Count > 1)) sb.Append("]");
            return sb.ToString();
        }

        /// <summary>
        /// Converts this piecewise vector to an array of doubles
        /// </summary>
        /// <returns></returns>
        public override double[] ToArray()
        {
            double[] result = new double[Count];
            for (int i = 0; i < result.Length; i++) result[i] = CommonValue;
            foreach (var sub in pieces)
            {
                for (int i = sub.Start; i <= sub.End; i++)
                {
                    result[i] = sub.Value;
                }
            }
            return result;
        }

        /// <summary>
        /// Converts this piecewise vector to an ordinary dense vector
        /// </summary>
        /// <returns></returns>
        public DenseVector ToVector()
        {
            return DenseVector.FromArray(ToArray());
        }

        #endregion

        #region Subarray and subvector

        /// <summary>
        /// Copies values from a vector. If the source vector is piecewise, then the common value
        /// is set to the common value from the source vector. If the source vector
        /// is dense, then the common value is set to the minimum of the data in the source vector
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public override void SetToSubvector(Vector that, int startIndex)
        {
            if (that.Sparsity.IsPiecewise)
            {
                SetToSubvector((PiecewiseVector) that, startIndex);
                return;
            }
            CommonValue = that.Min();
            base.SetToSubvector(that, startIndex);
        }


        /// <summary>
        /// Copies values from a piecewise vector. The common value is set to the common value
        /// from the source vector.
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public virtual void SetToSubvector(PiecewiseVector that, int startIndex)
        {
            double newCommonValue = that.CommonValue;
            var newPieces = new List<ConstantVector>();
            foreach (var sub in that.pieces)
            {
                int start = sub.Start - startIndex;
                if (start >= Count) break;
                if (start < 0) continue;
                int end = sub.End - startIndex;
                if (end >= Count) end = Count - 1;
                newPieces.Add(new ConstantVector(start, end, sub.Value));
            }
            SetInternal(newCommonValue, newPieces);
        }

        /// <summary>
        /// Create a subvector of this piecewise vector.
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public override Vector Subvector(int startIndex, int count)
        {
            var result = new PiecewiseVector(count);
            result.SetToSubvector(this, startIndex);
            return result;
        }

        #endregion
    }

    /// <summary>
    /// A vector which has a constant value between its start and end indices.
    /// </summary>
    [DataContract]
    public struct ConstantVector : IComparable
    {
        /// <summary>
        /// The start index
        /// </summary>
        [DataMember]
        public int Start { get; set; }

        /// <summary>
        /// The end index
        /// </summary>
        [DataMember]
        public int End { get; set; }

        /// <summary>
        /// The value of the vector
        /// </summary>
        [DataMember]
        public double Value { get; set; }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <param name="value"></param>
        [Construction("Start", "End", "Value")]
        public ConstantVector(int start, int end, double value) : this()
        {
            this.Start = start;
            this.End = end;
            this.Value = value;
        }

        /// <summary>
        /// The number of elements in this vector
        /// </summary>
        public int Count
        {
            get { return 1 + End - Start; }
        }

        /// <summary>
        /// True if the index lies inside this vector.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public bool Contains(int index)
        {
            return index >= Start && index <= End;
        }

        /// <summary>
        /// String representation of this constant vector.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return ToString("g4", i => i.ToString(CultureInfo.InvariantCulture));
        }

        /// <summary>
        /// String representation of this constant vector using the supplied value format
        /// and function for converting ints to strings.
        /// </summary>
        /// <param name="format"></param>
        /// <param name="intToString"></param>
        /// <returns></returns>
        public string ToString(string format, Func<int, string> intToString)
        {
            var sb = new StringBuilder();
            if (format != null) sb.Append("[");
            if ((format == null) && (Value == 0)) sb.Append("^");
            sb.Append(intToString(Start));
            if (End > Start)
            {
                sb.Append("-");
                sb.Append(intToString(End));
            }
            if (format != null)
            {
                sb.Append("]=");
                sb.Append(Value.ToString(format));
            }
            return sb.ToString();
        }

        #region IComparable Members

        /// <summary>
        /// Compares this constant vector to another one.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public int CompareTo(object obj)
        {
            if (!(obj is ConstantVector)) return 0;
            var vec = (ConstantVector) obj;
            return Start.CompareTo(vec.Start);
        }

        #endregion
    }
}