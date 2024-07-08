// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Class for accumulating weighted scalar observations and computing sample count and mean
    /// </summary>
    public class MeanAccumulator : Accumulator<double>, SettableTo<MeanAccumulator>, ICloneable
    {
        /// <summary>
        /// The sample mean
        /// </summary>
        public double Mean;

        /// <summary>
        /// Sample count
        /// </summary>
        public double Count;

        /// <summary>
        /// Adds an observation
        /// </summary>
        /// <param name="x"></param>
        public void Add(double x)
        {
            Add(x, 1);
        }

        /// <summary>
        /// Adds a weighted observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        public void Add(double x, double weight)
        {
            if (Count == 0)
            {
                Mean = x;
            }
            else
            {
                // This function ensures that the mean is within the bounds of the data and is monotonic in any single x.
                Mean = MMath.WeightedAverage(Count, Mean, weight, x);
            }
            Count += weight;
        }

        /// <summary>
        /// Adds all observations added to another accumulator.
        /// </summary>
        /// <param name="meanAccumulator"></param>
        public void Add(MeanAccumulator meanAccumulator)
        {
            Add(meanAccumulator.Mean, meanAccumulator.Count);
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            Count = 0;
            Mean = 0;
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MeanAccumulator value)
        {
            Mean = value.Mean;
            Count = value.Count;
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanAccumulator result = new MeanAccumulator();
            result.SetTo(this);
            return result;
        }

        public override string ToString()
        {
            return $"MeanAccumulator(Mean={Mean}, Count={Count})";
        }
    }

    /// <summary>
    /// Decorator of MeanVarianceAccumulator that does not add if any input is NaN.
    /// </summary>
    public class MeanAccumulatorSkipNaNs : Accumulator<double>, Accumulator<MeanAccumulatorSkipNaNs>, SettableTo<MeanAccumulatorSkipNaNs>, ICloneable
    {
        private MeanAccumulator mva = new MeanAccumulator();

        /// <summary>
        /// The sample mean
        /// </summary>
        public double Mean { get { return mva.Mean; } }

        /// <summary>
        /// Sample count
        /// </summary>
        public double Count { get { return mva.Count; } }

        /// <summary>
        /// Adds all items in another accumulator to this accumulator.
        /// </summary>
        /// <param name="that">Another estimator</param>
        public void Add(MeanAccumulatorSkipNaNs that)
        {
            mva.Add(that.mva);
        }

        /// <summary>
        /// Adds an observation
        /// </summary>
        /// <param name="x"></param>
        public void Add(double x)
        {
            if (!double.IsNaN(x))
                mva.Add(x);
        }

        /// <summary>
        /// Adds a weighted observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        public void Add(double x, double weight)
        {
            if (!double.IsNaN(x) && !double.IsNaN(weight))
                mva.Add(x, weight);
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            mva.Clear();
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MeanAccumulatorSkipNaNs value)
        {
            this.mva.SetTo(value.mva);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanAccumulatorSkipNaNs result = new MeanAccumulatorSkipNaNs();
            result.SetTo(this);
            return result;
        }
    }
}
