// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

using Microsoft.ML.Probabilistic.Distributions;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Class for accumulating weighted scalar observations and computing sample count and mean.
    /// Unlike <see cref="MeanAccumulator"/>, this class is associative but susceptible to overflow.
    /// </summary>
    public class MeanAccumulatorAssociative : Accumulator<double>, Accumulator<MeanAccumulatorAssociative>, SettableTo<MeanAccumulatorAssociative>, ICloneable
    {
        /// <summary>
        /// The sample sum
        /// </summary>
        public double Sum;

        /// <summary>
        /// Sample count
        /// </summary>
        public double Count;

        /// <summary>
        /// The sample sum
        /// </summary>
        public double Mean => Sum / Count;

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
            Sum += x;
            Count += weight;
        }

        /// <summary>
        /// Adds all observations added to another accumulator.
        /// </summary>
        /// <param name="meanAccumulator"></param>
        public void Add(MeanAccumulatorAssociative meanAccumulator)
        {
            Add(meanAccumulator.Sum, meanAccumulator.Count);
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            Count = 0;
            Sum = 0;
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MeanAccumulatorAssociative value)
        {
            Sum = value.Sum;
            Count = value.Count;
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanAccumulatorAssociative result = new MeanAccumulatorAssociative();
            result.SetTo(this);
            return result;
        }

        public override string ToString()
        {
            return $"MeanAccumulatorAssociative(Mean={Mean}, Count={Count})";
        }
    }

    /// <summary>
    /// Decorator of MeanVarianceAccumulator that does not add if any input is NaN.
    /// </summary>
    public class MeanAccumulatorAssociativeSkipNaNs : Accumulator<double>, Accumulator<MeanAccumulatorAssociativeSkipNaNs>, SettableTo<MeanAccumulatorAssociativeSkipNaNs>, ICloneable
    {
        public MeanAccumulatorAssociative meanAccumulator = new MeanAccumulatorAssociative();

        /// <summary>
        /// The sample mean
        /// </summary>
        public double Mean { get { return meanAccumulator.Mean; } }

        /// <summary>
        /// Sample count
        /// </summary>
        public double Count { get { return meanAccumulator.Count; } }

        /// <summary>
        /// Adds all items in another accumulator to this accumulator.
        /// </summary>
        /// <param name="that">Another estimator</param>
        public void Add(MeanAccumulatorAssociativeSkipNaNs that)
        {
            meanAccumulator.Add(that.meanAccumulator);
        }

        /// <summary>
        /// Adds an observation
        /// </summary>
        /// <param name="x"></param>
        public void Add(double x)
        {
            if (!double.IsNaN(x))
                meanAccumulator.Add(x);
        }

        /// <summary>
        /// Adds a weighted observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        public void Add(double x, double weight)
        {
            if (!double.IsNaN(x) && !double.IsNaN(weight))
                meanAccumulator.Add(x, weight);
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            meanAccumulator.Clear();
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MeanAccumulatorAssociativeSkipNaNs value)
        {
            this.meanAccumulator.SetTo(value.meanAccumulator);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanAccumulatorAssociativeSkipNaNs result = new MeanAccumulatorAssociativeSkipNaNs();
            result.SetTo(this);
            return result;
        }
    }
}
