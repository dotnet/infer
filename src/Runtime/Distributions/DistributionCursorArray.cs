// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions
{
    internal interface ReducibleTo<T>
    {
        /// <summary>
        /// Remove dimensions via multiplication.
        /// </summary>
        /// <param name="keep">The dimensions to keep.</param>
        /// <param name="result">A distribution or distribution array.</param>
        /// <returns>An action which will perform the reduction.</returns>
        /// <remarks>Each element of result will be a product over the dimensions not kept.
        /// Result must already be the correct size.
        /// If keep is empty, no dimensions are kept so the result is a single distribution.
        /// Otherwise, result is a distribution array whose dimensions are the kept dimensions.</remarks>
        void ReduceTo(int[] keep, ICursorArray<T> result);

        void ReduceTo(T result);
    }

    /// <summary>
    /// The distribution of an array of independent variables, or equivalently
    /// an array of distributions.
    /// </summary>
    /// <typeparam name="DistributionType"></typeparam>
    /// <typeparam name="DomainType"></typeparam>
    /// <remarks>
    /// This class supports all of the IDistribution methods, as well as 
    /// being a CursorArray<typeparamref name="DistributionType"/>.
    /// To support plates, it implements a ReduceTo method which removes 
    /// dimensions via multiplication.
    /// </remarks>
    // TODO: named dimensions (name can be any object, such as Plate)
    internal class DistributionCursorArray<DistributionType, DomainType> :
        CursorArray<DistributionType>, ICursor, IDistribution<DomainType[]>,
        SettableTo<DistributionCursorArray<DistributionType, DomainType>>,
        SettableToProduct<DistributionCursorArray<DistributionType, DomainType>>,
        ReducibleTo<DistributionType>
        where DistributionType : IDistribution<DomainType>, ICursor, SettableTo<DistributionType>, SettableToProduct<DistributionType>
    {
        #region IDistribution methods

        /// <summary>
        /// True if all elements are constant.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                foreach (DistributionType dist in this)
                {
                    if (!dist.IsPointMass) return false;
                }
                return true;
            }
        }

        public void SetToConstant()
        {
            foreach (DistributionType dist in this)
            {
                dist.Point = dist.Point;
            }
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public DomainType[] Point
        {
            get
            {
                DomainType[] result = new DomainType[Count];
                int i = 0;
                foreach (DistributionType dist in this)
                {
                    result[i++] = dist.Point;
                }
                // this only works if a distribution is stored sequentially in one array.
                // must be this[0] instead of cursor, since the Start property is important
                //CursorArray<ICursor> result = new CursorArray<ICursor>(this[0].Point, dim, stride);
                return result;
            }
            set
            {
                int i = 0;
                foreach (DistributionType dist in this)
                {
                    dist.Point = value[i++];
                }
            }
        }

        public bool IsCompatibleWith(object thatd)
        {
            DistributionCursorArray<DistributionType, DomainType> that = thatd as DistributionCursorArray<DistributionType, DomainType>;
            if (that == null) return false;
            if (!base.IsCompatibleWith(that)) return false;
            return true;
            //return cursor.IsCompatibleWith(that.cursor);
        }

        public void CheckCompatible(object that)
        {
            if (!IsCompatibleWith(that))
                throw new InferRuntimeException("DistributionArrays are incompatible");
        }

        public void SetToUniform()
        {
            foreach (DistributionType d in this)
            {
                d.SetToUniform();
            }
        }

        public bool IsUniform()
        {
            foreach (DistributionType d in this)
            {
                if (!d.IsUniform()) return false;
            }
            return true;
        }

        public double GetLogProb(DomainType[] x)
        {
            throw new NotSupportedException();
        }

#if false
        public T[] Sample(T[] result)
        {
            Assert.IsTrue(result.Length == Count);
            int i = 0;
            foreach (DistributionType d in this) {
                result[i++] = d.Sample();
            }
            return result;
        }
        public T[] Sample()
        {
            return Sample(new T[Count]);
        }
#endif

        public void SetTo(DistributionCursorArray<DistributionType, DomainType> that)
        {
            CheckCompatible(that);
            Action action = delegate() { cursor.SetTo(that.cursor); };
            ForEach(that, action);
        }

        // a and b may be the same object as this.
        public void SetToProduct(DistributionCursorArray<DistributionType, DomainType> a, DistributionCursorArray<DistributionType, DomainType> b)
        {
            CheckCompatible(a);
            CheckCompatible(b);
            Action action = delegate() { cursor.SetToProduct(a.cursor, b.cursor); };
            ForEach(a, b, action);
        }

        public override bool Equals(object thatd)
        {
            DistributionCursorArray<DistributionType, DomainType> that = thatd as DistributionCursorArray<DistributionType, DomainType>;
            if (that == null) return false;
            if (!IsCompatibleWith(that)) return false;
            IEnumerator<DistributionType> iter = that.GetEnumerator();
            foreach (DistributionType d in this)
            {
                bool ok = iter.MoveNext();
                Assert.IsTrue(ok);
                if (!d.Equals(iter.Current)) return false;
            }
            return true;
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            for (int i = 0; i < Rank; i++) hash = Hash.Combine(hash, dim[i]);
            foreach (DistributionType d in this)
            {
                hash = Hash.Combine(hash, d.GetHashCode());
            }
            return hash;
        }

        public double MaxDiff(object thatd)
        {
            DistributionCursorArray<DistributionType, DomainType> that = thatd as DistributionCursorArray<DistributionType, DomainType>;
            if (that == null) return Double.PositiveInfinity;
            if (!IsCompatibleWith(that)) return Double.PositiveInfinity;
            double max = 0;
            IEnumerator<DistributionType> iter = that.GetEnumerator();
            foreach (DistributionType d in this)
            {
                bool ok = iter.MoveNext();
                Assert.IsTrue(ok);
                double diff = ((Diffable) d).MaxDiff(iter.Current);
                if (diff > max) max = diff;
            }
            return max;
        }

        #endregion

        /// <summary>
        /// Apply a reduction action to produce a smaller array.
        /// </summary>
        public void ReduceTo(int[] keep, ICursorArray<DistributionType> result)
        {
            Assert.IsTrue(keep.Length == result.Rank);
            int[] index = new int[Rank];
            int[] result_index = new int[result.Rank];
            for (int i = 0; i < Count; i++)
            {
                LinearIndexToMultidimensionalIndex(i, index);
                int index_sum = 0;
                for (int d = 0; d < index.Length; d++)
                {
                    index_sum += index[d];
                }
                int result_index_sum = 0;
                for (int d = 0; d < keep.Length; d++)
                {
                    result_index[d] = index[keep[d]];
                    result_index_sum += result_index[d];
                }
                DistributionType d1 = this[index];
                DistributionType d2 = result[result_index];
                bool first = (result_index_sum == index_sum);
                if (first)
                {
                    d2.SetTo(d1);
                }
                else
                {
                    d2.SetToProduct(d2, d1);
                }
            }
        }

        /// <summary>
        /// Apply a reduction action to produce a single element distribution.
        /// </summary>
        public void ReduceTo(DistributionType result)
        {
            // FIXME: this could be more efficient
            bool first = true;
            foreach (DistributionType d in this)
            {
                if (first)
                {
                    result.SetTo(d);
                    first = false;
                }
                else
                {
                    result.SetToProduct(result, d);
                }
            }
        }

        public DistributionCursorArray(DistributionType cursor, params int[] lengths)
            : base(cursor, lengths)
        {
        }

        public DistributionCursorArray(DistributionType cursor, IList<int> lengths, IList<int> stride)
            : base(cursor, lengths, stride)
        {
        }

        /// <summary>
        /// Add dimensions to an array by replication.
        /// </summary>
        /// <param name="lengths">The result array dimensions.</param>
        /// <param name="newPosition">For each original dimension d, newPosition[d] is its index in the 
        /// result dimensions.  Length == this.Rank.</param>
        /// <returns>A new array which uses the same storage but a different cursor.</returns>
        public DistributionCursorArray<DistributionType, DomainType> Replicate(int[] lengths, int[] newPosition)
        {
            int[] newStride = new int[lengths.Length]; // newStride[i] = 0
            for (int i = 0; i < Rank; i++)
            {
                newStride[newPosition[i]] = stride[i];
                Assert.IsTrue(lengths[newPosition[i]] == dim[i]);
            }
            return new DistributionCursorArray<DistributionType, DomainType>
                ((DistributionType) this[0].ReferenceClone(), lengths, newStride);
        }

        /// <summary>
        /// Make a jagged array from a multidimensional array.
        /// </summary>
        /// <param name="isOuterDimension">For each original dimension d, 
        /// indicates whether it will be in the outer array. Length == this.Rank.</param>
        /// <returns>A jagged array [outer][inner] which uses the same storage but a different cursor.</returns>
        public DistributionCursorArray<DistributionCursorArray<DistributionType, DomainType>, DomainType[]>
            Split(IList<bool> isOuterDimension)
        {
            int outerRank = StringUtil.Sum(isOuterDimension);
            if (outerRank == 0) throw new ArgumentException("No outer dimensions chosen");
            int innerRank = Rank - outerRank;
            if (innerRank == 0) throw new ArgumentException("No inner dimensions left");
            int[] innerLengths = new int[innerRank];
            int[] innerStride = new int[innerRank];
            int[] outerLengths = new int[outerRank];
            int[] outerStride = new int[outerRank];
            int innerIndex = 0;
            int outerIndex = 0;
            int i = 0;
            foreach (bool isOuter in isOuterDimension)
            {
                if (isOuter)
                {
                    outerLengths[outerIndex] = dim[i];
                    outerStride[outerIndex] = stride[i];
                    outerIndex++;
                }
                else
                {
                    innerLengths[innerIndex] = dim[i];
                    innerStride[innerIndex] = stride[i];
                    innerIndex++;
                }
                i++;
            }
            DistributionCursorArray<DistributionType, DomainType> inner =
                new DistributionCursorArray<DistributionType, DomainType>
                    ((DistributionType) this[0].ReferenceClone(), innerLengths, innerStride);
            return new DistributionCursorArray<DistributionCursorArray<DistributionType, DomainType>, DomainType[]>
                (inner, outerLengths, outerStride);
        }

        #region Copying

        public override ICursor ReferenceClone()
        {
            // must use this[0] not cursor
            return new DistributionCursorArray<DistributionType, DomainType>((DistributionType) this[0].ReferenceClone(), dim, stride);
        }

        public override object Clone()
        {
            return new DistributionCursorArray<DistributionType, DomainType>((DistributionType) this[0].ReferenceClone(), dim);
        }

        #endregion

        /// <summary>
        /// Overrides ToString method
        /// </summary>
        /// <returns>String representation of instance</returns>
        public override string ToString()
        {
            return StringUtil.ArrayToString(this, count, new int[Rank]);
        }
    }
}