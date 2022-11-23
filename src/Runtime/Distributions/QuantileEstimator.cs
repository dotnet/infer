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
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Subsamples data to provide accurate estimation of quantiles.
    /// </summary>
    [Serializable, DataContract]
    public class QuantileEstimator : CanGetProbLessThan<double>, CanGetQuantile<double>
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
        SerializableRandom rand;

        [DataMember]
        public readonly double MaximumError;

        private int bufferLength
        {
            get
            {
                double invError = 1 / MaximumError;
                int bufferCount = buffers.Length;
                int length = (int)Math.Ceiling(invError * Math.Sqrt(bufferCount - 1));
                if (length % 2 == 1) length++;
                return length;
            }
        }

        /// <summary>
        /// 0 = point mass on each data point (i/n)
        /// 1 = interpolate (i/n + (i+1)/n)/2 = (i+0.5)/n
        /// 2 = interpolate i/(n-1)
        /// </summary>
        private readonly int InterpolationType = 0;

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
            buffers = new double[bufferCount][];
            countInBuffer = new int[bufferCount];

            rand = SerializableRandom.CreateNew(Rand.Int());
        }

        /// <summary>
        /// Sets the seed used for random number generation.
        /// </summary>
        /// <param name="seed">Specify the seed to use for random number generation.</param>
        public void SetRandomSeed(int seed) =>
            rand = SerializableRandom.CreateNew(seed);

        /// <summary>
        /// Returns the quantile rank of x.  This is a probability such that GetQuantile(probability) == x, whenever x is inside the support of the distribution.  May be discontinuous due to duplicates.
        /// </summary>
        public double GetProbLessThan(double x)
        {
            GetAdjacentItems(x, out double lowerItem, out double upperItem, out long lowerRank, out long lowerWeight, out long upperWeight, out long minLowerWeight, out long minUpperWeight, out long itemCount);
            if (lowerRank == 0) return 0;
            if (lowerRank == itemCount) return 1;
            if (InterpolationType == 0)
            {
                return (double)lowerRank / itemCount;
            }
            else if (InterpolationType == 1)
            {
                // interpolate between the ranks of lowerItem and upperItem
                // lowerItem has rank (lowerRank - 0.5) from above
                // upperItem has rank (lowerRank + 0.5) from below
                double frac = (x - lowerItem) / (upperItem - lowerItem);
                return (lowerRank - 0.5 + frac) / itemCount;
            }
            else
            {
                // interpolate between the ranks of lowerItem and upperItem
                // lowerItem has rank (lowerRank - 1)/(itemCount-1)  from above
                // upperItem has rank lowerRank/(itemCount-1) from below
                double frac = (x - lowerItem) / (upperItem - lowerItem);
                return (lowerRank - 1 + frac) / (itemCount - 1);
            }
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            return Math.Max(0.0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
        }

        /// <summary>
        /// Returns the largest value x such that GetProbLessThan(x) &lt;= probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns></returns>
        public double GetQuantile(double probability)
        {
            if (double.IsNaN(probability)) throw new ArgumentOutOfRangeException(nameof(probability), "probability is NaN");
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1.0) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1.0");
            // compute the min and max of the retained items
            double lowerBound = double.PositiveInfinity, upperBound = double.NegativeInfinity;
            bool countGreaterThanZero = false;
            for (int bufferIndex = 0; bufferIndex < buffers.Length; bufferIndex++)
            {
                double[] buffer = buffers[bufferIndex];
                int count = countInBuffer[bufferIndex];
                if (count > 0) countGreaterThanZero = true;
                for (int i = 0; i < count; i++)
                {
                    double item = buffer[i];
                    if (item < lowerBound) lowerBound = item;
                    if (item > upperBound) upperBound = item;
                }
            }
            if (probability == 1.0) return MMath.NextDouble(upperBound);
            if (probability == 0.0) return lowerBound;
            if (!countGreaterThanZero) throw new Exception("QuantileEstimator has no data");
            if (lowerBound == upperBound) return upperBound;
            // use bisection
            while (true)
            {
                double x = MMath.Average(lowerBound, upperBound);
                double lowerItem;
                double upperItem;
                long lowerRank;
                long lowerWeight, minLowerWeight, upperWeight, minUpperWeight;
                long itemCount;
                GetAdjacentItems(x, out lowerItem, out upperItem, out lowerRank, out lowerWeight, out upperWeight, out minLowerWeight, out minUpperWeight, out itemCount);
                if (InterpolationType == 0)
                {
                    double probabilityLessThanLowerItem = (double)(lowerRank - lowerWeight) / itemCount;
                    if (probability < probabilityLessThanLowerItem)
                    {
                        upperBound = MMath.PreviousDouble(lowerItem);
                    }
                    else
                    {
                        double probabilityLessThanUpperItem = (double)lowerRank / itemCount;
                        if (probability < probabilityLessThanUpperItem) return lowerItem;
                        double probabilityLessThanOrEqualUpperItem = (double)(lowerRank + upperWeight) / itemCount;
                        if (probability < probabilityLessThanOrEqualUpperItem) return upperItem;
                        lowerBound = MMath.NextDouble(upperItem);
                    }
                    if (lowerBound > upperBound) throw new Exception("lowerBound > upperBound");
                }
                else if (InterpolationType == 1)
                {
                    // Find frac such that (lowerRank - 0.5 + frac) / itemCount == probability
                    double scaledProbability = MMath.LargestDoubleProduct(probability, itemCount);
                    if (scaledProbability < 0.5) return lowerBound;
                    if (scaledProbability >= itemCount - 0.5) return upperBound;
                    // probability of lowerItem ranges from (lowerRank-lowerWeight+0.5) / itemCount
                    // to (lowerRank - 0.5) / itemCount
                    //if (scaledProbability == lowerRank - lowerWeight + 0.5) return lowerItem;
                    if ((scaledProbability > lowerRank - lowerWeight + 0.5) && (scaledProbability < lowerRank - 0.5))
                        return lowerItem;
                    // probability of upperItem ranges from (lowerRank + 0.5) / itemCount
                    // to (lowerRank + upperWeight - 0.5) / itemCount
                    if (scaledProbability == lowerRank + 0.5) return upperItem;
                    if ((scaledProbability > lowerRank + 0.5) && (scaledProbability < lowerRank + upperWeight - 0.5))
                        return upperItem;
                    double frac = scaledProbability - (lowerRank - 0.5);
                    if (frac < 0)
                    {
                        upperBound = MMath.PreviousDouble(x);
                    }
                    else if (frac > 1)
                    {
                        lowerBound = MMath.NextDouble(x);
                    }
                    else
                    {
                        return OuterQuantiles.GetQuantile(probability, lowerRank - 0.5, lowerItem, upperItem, itemCount + 1);
                    }
                }
                else
                {
                    double scaledProbability = MMath.LargestDoubleProduct(probability, itemCount - 1);
                    // probability of lowerItem ranges from (lowerRank-lowerWeight) / (itemCount - 1)
                    // to (lowerRank - 1) / (itemCount - 1).
                    if (scaledProbability == lowerRank - lowerWeight) return lowerItem;
                    if ((scaledProbability > lowerRank - lowerWeight) && (scaledProbability < lowerRank - 1))
                        return lowerItem;
                    // probability of upperItem ranges from lowerRank / (itemCount - 1)
                    // to (lowerRank + upperWeight-1) / (itemCount - 1)
                    if (scaledProbability == lowerRank) return upperItem;
                    if ((scaledProbability > lowerRank) && (scaledProbability < lowerRank + upperWeight - 1))
                        return upperItem;
                    // solve for frac in (lowerRank-1 + frac) / (itemCount - 1) == probability
                    double frac = scaledProbability - (lowerRank - 1);
                    if (frac < 0)
                    {
                        upperBound = MMath.PreviousDouble(x);
                    }
                    else if (frac > 1)
                    {
                        lowerBound = MMath.NextDouble(x);
                    }
                    else
                    {
                        return OuterQuantiles.GetQuantile(probability, lowerRank - 1, lowerItem, upperItem, itemCount);
                    }
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
                int bufferToFree = lowestBufferIndex;
                RaiseLowestHeight();
                if (countInBuffer[bufferToFree] > 0) throw new Exception("countInBuffer[bufferToFree] > 0");
                buffers[bufferToFree] = null;
            }
            lowestBufferHeight--;
            reservoirCount /= 2;
        }

        /// <summary>
        /// Multiply all sample weights by 2.
        /// </summary>
        public void Inflate()
        {
            if (lowestBufferHeight >= 30) throw new InvalidOperationException($"Cannot inflate when lowestBufferHeight = {lowestBufferHeight}");
            lowestBufferHeight++;
            reservoirCount *= 2;
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
                this.buffers.JaggedValueEquals(that.buffers) &&
                this.countInBuffer.ValueEquals(that.countInBuffer) &&
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
        /// <param name="lowerItem">The largest item that is less than x.</param>
        /// <param name="upperItem">The smallest item that is greater than or equal to x.</param>
        /// <param name="lowerRank">The total weight of items less than x.</param>
        /// <param name="lowerWeight">The total weight of items equal to lowerItem.</param>
        /// <param name="upperWeight">The total weight of items equal to upperItem.</param>
        /// <param name="minLowerWeight">The smallest weight of items equal to lowerItem.</param>
        /// <param name="minUpperWeight">The smallest weight of items equal to upperItem.</param>
        /// <param name="itemCount"></param>
        private void GetAdjacentItems(double x, out double lowerItem, out double upperItem, out long lowerRank, out long lowerWeight, out long upperWeight, out long minLowerWeight, out long minUpperWeight, out long itemCount)
        {
            lowerRank = 0;
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
                    if (item < x)
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
        }

        private double[] GetBuffer(int bufferIndex)
        {
            double[] buffer = buffers[bufferIndex];
            if (buffer == null)
            {
                buffers[bufferIndex] = buffer = new double[bufferLength];
            }
            return buffer;
        }

        private void AddToBuffer(int bufferIndex, double item)
        {
            double[] buffer = GetBuffer(bufferIndex);
            int count = countInBuffer[bufferIndex];
            if (count == buffer.Length)
            {
                // buffer is full.
                CompactBuffer(bufferIndex);
                count = 0;
            }
            if (double.IsNaN(item)) throw new ArgumentOutOfRangeException(nameof(item), item, "item is NaN");
            buffer[count] = item;
            count++;
            countInBuffer[bufferIndex] = count;
        }

        private void CompactBuffer(int bufferIndex)
        {
            int count = countInBuffer[bufferIndex];
            if (count == 0) return;
            double[] buffer = buffers[bufferIndex];
            // move half of the items to the next buffer, and empty this buffer.
            int nextBufferIndex = (bufferIndex + 1) % buffers.Length;
            if (nextBufferIndex == lowestBufferIndex)
                throw new Exception("Out of buffers");
            Array.Sort(buffer, 0, count);
            int firstIndex = rand.Next(2);
            if (count == 1 && firstIndex == 1) countInBuffer[bufferIndex] = 0;
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
            if (rand.Next(reservoirCount) == 0)
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
                if (rand.Next(reservoirCount) < weight)
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
                    if (rand.Next(lowestBufferWeight) < weight)
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
                    if (rand.Next(lowestBufferWeight) < sampleWeight)
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
                double[] buffer = GetBuffer(bufferIndex);
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
            int count = countInBuffer[bufferIndex];
            if (count > 0)
            {
                double[] buffer = buffers[bufferIndex];
                if (count > buffer.Length / 2)
                {
                    RaiseLowestHeight();
                }
            }
        }

        /// <summary>
        /// Requires the invariant or there to be at least one empty buffer.  Ensures the invariant.
        /// </summary>
        private void RaiseLowestHeight()
        {
            // Compacting the lowest buffer will always succeed because of the invariant or because there is at least one empty buffer.
            CompactBuffer(lowestBufferIndex);
            // Since the lowest buffer is now empty, we re-purpose it as the new highest buffer.
            lowestBufferIndex = (lowestBufferIndex + 1) % buffers.Length;
            lowestBufferHeight++;
            if (lowestBufferHeight > 30) throw new Exception("Exceeded the limit on number of items");
        }

        /// <summary>
        /// Similar to <see cref="Random"/> but serializable so that
        /// the quantile estimator can be serialized to JSON and still
        /// produces reproducible results.
        /// </summary>
        /// <remarks>
        /// Based on <see cref="Random"/>.
        /// </remarks>
        [Serializable, DataContract]
        private sealed class SerializableRandom
        {
            const int MBIG = Int32.MaxValue;
            const int MSEED = 161803398;

            [DataMember]
            int inext;

            [DataMember]
            int inextp;

            [DataMember]
            int[] seedArray = new int[56];

            public static SerializableRandom CreateNew(int seed)
            {
                var rand = new SerializableRandom();

                unchecked
                {
                    int ii;
                    int mj, mk;

                    int subtraction = (seed == Int32.MinValue) ? Int32.MaxValue : Math.Abs(seed);
                    mj = MSEED - subtraction;
                    rand.seedArray[55] = mj;
                    mk = 1;
                    for (int i = 1; i < 55; i++)
                    {
                        ii = (21 * i) % 55;
                        rand.seedArray[ii] = mk;
                        mk = mj - mk;
                        if (mk < 0) mk += MBIG;
                        mj = rand.seedArray[ii];
                    }
                    for (int k = 1; k < 5; k++)
                    {
                        for (int i = 1; i < 56; i++)
                        {
                            rand.seedArray[i] -= rand.seedArray[1 + (i + 30) % 55];
                            if (rand.seedArray[i] < 0) rand.seedArray[i] += MBIG;
                        }
                    }
                    rand.inext = 0;
                    rand.inextp = 21;

                    return rand;
                }
            }

            double Sample()
            {
                return (InternalSample() * (1.0 / MBIG));
            }

            private int InternalSample()
            {
                int retVal;
                int locINext = inext;
                int locINextp = inextp;

                if (++locINext >= 56) locINext = 1;
                if (++locINextp >= 56) locINextp = 1;

                retVal = seedArray[locINext] - seedArray[locINextp];

                if (retVal == MBIG) retVal--;
                if (retVal < 0) retVal += MBIG;

                seedArray[locINext] = retVal;

                inext = locINext;
                inextp = locINextp;

                return retVal;
            }

            public int Next()
            {
                int retVal;
                int locINext = inext;
                int locINextp = inextp;

                if (++locINext >= 56) locINext = 1;
                if (++locINextp >= 56) locINextp = 1;

                retVal = seedArray[locINext] - seedArray[locINextp];

                if (retVal == MBIG) retVal--;
                if (retVal < 0) retVal += MBIG;

                seedArray[locINext] = retVal;

                inext = locINext;
                inextp = locINextp;

                return retVal;
            }
            public int Next(int maxValue)
            {
                return (int)(Sample() * maxValue);
            }
        }
    }
}
