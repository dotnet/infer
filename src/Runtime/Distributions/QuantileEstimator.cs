// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Subsamples data to provide accurate estimation of quantiles.
    /// </summary>
    [Serializable, DataContract]
    public class QuantileEstimator : CanGetProbLessThan, CanGetQuantile
    {
        // Reference: 
        // "Quantiles over Data Streams: An Experimental Study"
        // Lu Wang, Ge Luo, Ke Yi, Graham Cormode
        // ACM-SIGMOD International Conference on Management of Data, 2013
        // http://dimacs.rutgers.edu/~graham/pubs/papers/nquantiles.pdf

        /// <summary>
        /// Stores a subset of added items, each with a different weight.
        /// All items in the same buffer have the same weight.
        /// Items in the lowest buffer have weight 2^lowestBufferHeight.
        /// Items in the next buffer have twice as much weight, and so on.
        /// To allow adding without overflow, the highest buffer must satisfy the following invariant:
        /// countInBuffer[bufferIndex] &lt;= buffers[bufferIndex].Length/2
        /// where bufferIndex = (lowestBufferIndex-1)
        /// </summary>
        [DataMember]
        double[][] buffers;
        [DataMember]
        int[] countInBuffer;
        [DataMember]
        int lowestBufferIndex;
        /// <summary>
        /// Always at least 0 and at most 30.
        /// </summary>
        [DataMember]
        int lowestBufferHeight;
        [DataMember]
        double nextSample;
        [DataMember]
        int reservoirCount;

        [DataMember]
        public readonly double MaximumError;

        /// <summary>
        /// Creates a new QuantileEstimator.
        /// </summary>
        /// <param name="maximumError">The allowed error in the return value of GetProbLessThan.  Must be greater than 0 and less than 1.  As a rule of thumb, set this to the reciprocal of the number of desired quantiles.</param>
        public QuantileEstimator(double maximumError)
        {
            if (maximumError <= 0) throw new ArgumentOutOfRangeException(nameof(maximumError), "maximumError <= 0");
            if (maximumError >= 1) throw new ArgumentOutOfRangeException(nameof(maximumError), "maximumError >= 1");
            this.MaximumError = maximumError;
            // maxError = 0.05 gives bufferCount = 6, bufferLength = 46
            double invError = 1 / maximumError;
            int bufferCount = 1 + Math.Max(1, (int)Math.Ceiling(Math.Log(invError, 2)));
            if (bufferCount < 2) throw new Exception("bufferCount < 2");
            int bufferLength = (int)Math.Ceiling(invError * Math.Sqrt(bufferCount - 1));
            if (bufferLength % 2 == 1) bufferLength++;
            buffers = Util.ArrayInit(bufferCount, i => new double[bufferLength]);
            countInBuffer = new int[bufferCount];
        }

        /// <summary>
        /// Returns the quantile rank of x.  This is a probability such that GetQuantile(probability) == x, whenever x is inside the support of the distribution.  May be discontinuous due to duplicates.
        /// </summary>
        public double GetProbLessThan(double x)
        {
            double lowerItem;
            double upperItem;
            long lowerIndex;
            long lowerWeight, minLowerWeight, upperWeight, minUpperWeight;
            long itemCount;
            GetAdjacentItems(x, out lowerItem, out upperItem, out lowerIndex, out lowerWeight, out upperWeight, out minLowerWeight, out minUpperWeight, out itemCount);
            if (lowerIndex < 0) return 0;
            if (x == lowerItem) return (double)(lowerIndex - lowerWeight + 1) / (itemCount - 1);
            // interpolate between the ranks of lowerItem and upperItem
            double frac = (x - lowerItem) / (upperItem - lowerItem);
            return (lowerIndex + frac) / (itemCount - 1);
        }

        /// <summary>
        /// Returns the largest value x such that GetProbLessThan(x) &lt;= probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns></returns>
        public double GetQuantile(double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1.0) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1.0");
            // compute the min and max of the retained items
            double lowerBound = double.PositiveInfinity, upperBound = double.NegativeInfinity;
            for (int bufferIndex = 0; bufferIndex < buffers.Length; bufferIndex++)
            {
                double[] buffer = buffers[bufferIndex];
                int count = countInBuffer[bufferIndex];
                for (int i = 0; i < count; i++)
                {
                    double item = buffer[i];
                    if (item < lowerBound) lowerBound = item;
                    if (item > upperBound) upperBound = item;
                }
            }
            if (probability == 1.0) return upperBound + MMath.Ulp(upperBound);
            if (double.IsPositiveInfinity(lowerBound)) throw new Exception("QuantileEstimator has no data");
            // use bisection
            while (true)
            {
                double x = (lowerBound + upperBound) / 2;
                double lowerItem;
                double upperItem;
                long lowerIndex;
                long lowerWeight, minLowerWeight, upperWeight, minUpperWeight;
                long itemCount;
                GetAdjacentItems(x, out lowerItem, out upperItem, out lowerIndex, out lowerWeight, out upperWeight, out minLowerWeight, out minUpperWeight, out itemCount);
                double scaledProbability = probability * (itemCount - 1);
                // probability of lowerItem ranges from (lowerIndex - lowerWeight + 1) / (itemCount - 1)
                // to lowerIndex / (itemCount - 1).
                if ((scaledProbability >= lowerIndex - lowerWeight + 1) && (scaledProbability < lowerIndex))
                    return lowerItem;
                // probability of upperItem ranges from (lowerIndex + 1) / (itemCount - 1)
                // to (lowerIndex + upperWeight) / (itemCount - 1)
                if ((scaledProbability >= lowerIndex + 1) && (scaledProbability <= lowerIndex + upperWeight))
                    return upperItem;
                // solve for frac in (lowerIndex + frac) / (itemCount - 1) == probability
                double frac = scaledProbability - lowerIndex;
                if (frac < 0)
                {
                    upperBound = x;
                }
                else if (frac > 1)
                {
                    lowerBound = x;
                }
                else
                {
                    return OuterQuantiles.GetQuantile(probability, lowerIndex, lowerItem, upperItem, itemCount);
                }
            }
        }

        public void Add(double item)
        {
            if (lowestBufferHeight == 0) AddAtHeight(item, lowestBufferHeight);
            else AddToSampler(item);
        }

        public void AddRange(IEnumerable<double> items)
        {
            foreach (var item in items) Add(item);
        }

        public void Add(double item, int weight)
        {
            int lowestBufferWeight = (1 << lowestBufferHeight);
            while (weight >= lowestBufferWeight)
            {
                int height = (int)Math.Floor(Math.Log(weight, 2));
                // This can change lowestBufferHeight.
                AddAtHeight(item, height);
                lowestBufferWeight = (1 << lowestBufferHeight);
                weight -= (1 << height);
            }
            AddToSampler(item, weight);
        }

        /// <summary>
        /// Add all samples stored in another QuantileEstimator, compacting them as necessary.
        /// </summary>
        /// <param name="that"></param>
        public void Add(QuantileEstimator that)
        {
            if (that == this) throw new ArgumentException("Argument is the same object as this", nameof(that));
            // Add the samples stored in all buffers, in increasing height.
            int height = that.lowestBufferHeight;
            for (int bufferIndex = that.lowestBufferIndex; ; height++)
            {
                double[] buffer = that.buffers[bufferIndex];
                int count = that.countInBuffer[bufferIndex];
                for (int i = 0; i < count; i++)
                {
                    AddAtHeight(buffer[i], height);
                }
                bufferIndex = (bufferIndex + 1) % that.buffers.Length;
                if (bufferIndex == that.lowestBufferIndex) break;
            }
            Add(that.nextSample, that.reservoirCount);
        }

        /// <summary>
        /// Divide all sample weights by 2.
        /// </summary>
        public void Deflate()
        {
            if (lowestBufferHeight == 0)
            {
                RaiseLowestHeight();
            }
            lowestBufferHeight--;
            reservoirCount /= 2;
        }

        public override string ToString()
        {
            return $"QuantileEstimator({MaximumError}, lowestBufferHeight = {lowestBufferHeight}, buffer sizes = {StringUtil.CollectionToString(countInBuffer, ",")})";
        }

        /// <summary>
        /// Returns true if that contains the same information as this.
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public bool ValueEquals(QuantileEstimator that)
        {
            return (this.MaximumError == that.MaximumError) &&
                Util.JaggedValueEquals(this.buffers, that.buffers) &&
                Util.ValueEquals(this.countInBuffer, that.countInBuffer) &&
                (this.lowestBufferIndex == that.lowestBufferIndex) &&
                (this.lowestBufferHeight == that.lowestBufferHeight) &&
                (this.nextSample == that.nextSample) &&
                (this.reservoirCount == that.reservoirCount);
        }

        /// <summary>
        /// Gets the total weight of retained items, which is an approximate count of the number of items added.
        /// </summary>
        /// <returns></returns>
        public ulong GetCount()
        {
            ulong itemCount = 0;
            ulong weight = (1UL << lowestBufferHeight);
            for (int bufferIndex = lowestBufferIndex; ; weight *= 2)
            {
                int count = countInBuffer[bufferIndex];
                itemCount += (ulong)count * weight;
                bufferIndex = (bufferIndex + 1) % buffers.Length;
                if (bufferIndex == lowestBufferIndex) break;
            }
            return itemCount;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="lowerItem">The largest item that is less than or equal to x.</param>
        /// <param name="upperItem">The smallest item that is greater than x.</param>
        /// <param name="lowerIndex"></param>
        /// <param name="lowerWeight">The total weight of items equal to lowerItem, excluding the item with lowest weight.</param>
        /// <param name="upperWeight">The total weight of items equal to upperItem.</param>
        /// <param name="minLowerWeight">The smallest weight of items equal to lowerItem.</param>
        /// <param name="minUpperWeight">The smallest weight of items equal to upperItem.</param>
        /// <param name="itemCount"></param>
        private void GetAdjacentItems(double x, out double lowerItem, out double upperItem, out long lowerIndex, out long lowerWeight, out long upperWeight, out long minLowerWeight, out long minUpperWeight, out long itemCount)
        {
            long lowerRank = 0;
            long weight = (1L << lowestBufferHeight);
            itemCount = 0;
            lowerItem = double.NegativeInfinity;
            upperItem = double.PositiveInfinity;
            lowerWeight = 0;
            upperWeight = 0;
            minLowerWeight = 0;
            minUpperWeight = 0;
            for (int bufferIndex = lowestBufferIndex; ; weight *= 2)
            {
                // count the number of items less than x
                double[] buffer = buffers[bufferIndex];
                int count = countInBuffer[bufferIndex];
                for (int i = 0; i < count; i++)
                {
                    double item = buffer[i];
                    if (item <= x)
                    {
                        lowerRank += weight;
                        if (item > lowerItem)
                        {
                            lowerItem = item;
                            lowerWeight = weight;
                            minLowerWeight = weight;
                        }
                        else if (item == lowerItem)
                        {
                            lowerWeight += weight;
                            if (weight < minLowerWeight)
                                minLowerWeight = weight;
                        }
                    }
                    else if (item < upperItem)
                    {
                        upperItem = item;
                        upperWeight = weight;
                        minUpperWeight = weight;
                    }
                    else if (item == upperItem)
                    {
                        upperWeight += weight;
                        if (weight < minUpperWeight)
                            minUpperWeight = weight;
                    }
                }
                itemCount += count * weight;
                bufferIndex = (bufferIndex + 1) % buffers.Length;
                if (bufferIndex == lowestBufferIndex) break;
            }
            if (itemCount == 0) throw new Exception("QuantileEstimator has no data");
            lowerIndex = lowerRank - 1;
        }

        private void AddToBuffer(int bufferIndex, double item)
        {
            double[] buffer = buffers[bufferIndex];
            int count = countInBuffer[bufferIndex];
            if (count == buffer.Length)
            {
                // buffer is full.
                CompactBuffer(bufferIndex);
                count = 0;
            }
            buffer[count] = item;
            count++;
            countInBuffer[bufferIndex] = count;
        }

        private void CompactBuffer(int bufferIndex)
        {
            double[] buffer = buffers[bufferIndex];
            int count = countInBuffer[bufferIndex];
            // move half of the items to the next buffer, and empty this buffer.
            int nextBufferIndex = (bufferIndex + 1) % buffers.Length;
            if (nextBufferIndex == lowestBufferIndex)
                throw new Exception("Out of buffers");
            Array.Sort(buffer, 0, count);
            int firstIndex = Rand.Int(2);
            for (int i = firstIndex; i < count; i += 2)
            {
                if (i + 2 >= count) countInBuffer[bufferIndex] = 0;
                AddToBuffer(nextBufferIndex, buffer[i]);
            }
            //Trace.WriteLine($"buffer sizes after compacting {bufferIndex}: {StringUtil.CollectionToString(countInBuffer, ",")}");
        }

        private void AddToSampler(double item)
        {
            // reservoir sampling
            reservoirCount = checked(reservoirCount + 1);
            if (Rand.Int(reservoirCount) == 0)
            {
                // item is the new sample
                nextSample = item;
            }
            int lowestBufferWeight = (1 << lowestBufferHeight);
            if (reservoirCount > lowestBufferWeight) throw new Exception("sampleWeight > lowestBufferWeight");
            if (reservoirCount == lowestBufferWeight)
            {
                // must do this before adding, since the sample may end up back in the reservoir.
                reservoirCount = 0;
                AddAtHeight(nextSample, lowestBufferHeight);
            }
        }

        /// <summary>
        /// Add a low-weight item to the sampling reservoir.
        /// </summary>
        /// <param name="item">Any number.</param>
        /// <param name="weight">At most lowestBufferWeight.</param>
        private void AddToSampler(double item, int weight)
        {
            // Reference: Section 3 of
            // "Optimal Quantile Approximation in Streams"
            // Zohar Karnin, Kevin Lang, Edo Liberty
            // 2016
            // https://arxiv.org/abs/1603.05346
            int lowestBufferWeight = (1 << lowestBufferHeight);
            int newCount = checked(reservoirCount + weight);
            if (newCount <= lowestBufferWeight)
            {
                reservoirCount = newCount;
                if (Rand.Int(reservoirCount) < weight)
                {
                    // item is the new sample
                    nextSample = item;
                }
                if (reservoirCount == lowestBufferWeight)
                {
                    // must do this before adding, since the sample may end up back in the reservoir.
                    reservoirCount = 0;
                    AddAtHeight(nextSample, lowestBufferHeight);
                }
            }
            else
            {
                if (weight > reservoirCount)
                {
                    // Output and discard item.
                    if (weight > lowestBufferWeight)
                    {
                        throw new ArgumentOutOfRangeException(nameof(weight), "weight > lowestBufferWeight");
                    }
                    if (Rand.Int(lowestBufferWeight) < weight)
                    {
                        AddAtHeight(item, lowestBufferHeight);
                    }
                }
                else
                {
                    // Output and discard nextSample.
                    double sample = nextSample;
                    int sampleWeight = reservoirCount;
                    // must do this before adding, since the sample may end up back in the reservoir.
                    nextSample = item;
                    reservoirCount = weight;
                    if (Rand.Int(lowestBufferWeight) < sampleWeight)
                    {
                        AddAtHeight(sample, lowestBufferHeight);
                    }
                }
            }
        }

        /// <summary>
        /// Add an item with weight 2^height to the appropriate buffer, or the sampling reservoir.  Requires the invariant and preserves the invariant.
        /// </summary>
        /// <param name="item"></param>
        /// <param name="height"></param>
        private void AddAtHeight(double item, int height)
        {
            if (height < lowestBufferHeight) AddToSampler(item, (1 << height));
            else
            {
                int heightDifference = height - lowestBufferHeight;
                while (heightDifference >= buffers.Length)
                {
                    // Compact the lower levels and advance the height until it matches.
                    RaiseLowestHeight();
                    heightDifference--;
                }
                // find the buffer with same height 
                int bufferIndex = (lowestBufferIndex + heightDifference) % buffers.Length;
                double[] buffer = buffers[bufferIndex];
                int count = countInBuffer[bufferIndex];
                // Make room in the buffer.
                if (count == buffer.Length)
                {
                    // This cannot cause overflow, due to the invariant, but may break the invariant.
                    CompactBuffer(bufferIndex);
                    MaintainInvariant();
                    if (height < lowestBufferHeight)
                    {
                        // lowestBufferHeight has changed.
                        AddAtHeight(item, height);
                        return;
                    }
                }
                if (heightDifference > 0)
                {
                    // if this is the highest buffer, check if adding would violate the invariant.
                    if (heightDifference == buffers.Length - 1 && count == buffer.Length / 2)
                    {
                        // Adding will violate the invariant.
                        // Compact the lowest buffer.  This cannot cause overflow, due to the invariant, but may break the invariant.
                        CompactBuffer(lowestBufferIndex);
                        count = countInBuffer[bufferIndex];
                        if (count == buffer.Length)
                        {
                            RaiseLowestHeight();
                        }
                        // Either the buffer has space, or the invariant holds.
                        AddToBuffer(bufferIndex, item);
                        MaintainInvariant();
                        return;
                    }
                }
                // this cannot cause a compaction.
                AddToBuffer(bufferIndex, item);
            }
        }

        /// <summary>
        /// Requires there to be at least one empty buffer.  Ensures the invariant.
        /// </summary>
        private void MaintainInvariant()
        {
            int bufferIndex = (lowestBufferIndex + buffers.Length - 1) % buffers.Length;
            double[] buffer = buffers[bufferIndex];
            int count = countInBuffer[bufferIndex];
            if (count > buffer.Length / 2)
            {
                RaiseLowestHeight();
            }
        }

        /// <summary>
        /// Requires the invariant or there to be at least one empty buffer.  Ensures the invariant.
        /// </summary>
        private void RaiseLowestHeight()
        {
            // Compacting the lowest buffer will always succeed because of the invariant or because there is at least one empty buffer.
            if (countInBuffer[lowestBufferIndex] > 0)
                CompactBuffer(lowestBufferIndex);
            // Since the lowest buffer is now empty, we re-purpose it as the new highest buffer.
            lowestBufferIndex = (lowestBufferIndex + 1) % buffers.Length;
            lowestBufferHeight++;
            if (lowestBufferHeight > 30) throw new Exception("Exceeded the limit on number of items");
        }
    }
}
