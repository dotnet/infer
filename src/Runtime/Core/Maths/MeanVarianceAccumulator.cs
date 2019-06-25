// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using System;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Class for accumulating weighted noisy scalar observations,
    /// and computing sample count, mean, and variance
    /// </summary>
    public class MeanVarianceAccumulator : Accumulator<double>, SettableTo<MeanVarianceAccumulator>, ICloneable
    {
        /// <summary>
        /// The sample mean
        /// </summary>
        public double Mean;

        /// <summary>
        /// Sample variance
        /// </summary>
        public double Variance;

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
            // new_mean = mean + w/(count + w)*(x - mean)
            // new_var = count/(count + w)*(var + w/(count+w)*(x-mean)*(x-mean)')
            Count += weight;
            if (weight == Count)
            {
                // avoid numerical overflow
                Mean = x;
                Variance = 0;
            }
            else if(weight != 0)
            {
                double diff;
                if (x == Mean) diff = 0;  // avoid subtracting infinities
                else diff = x - Mean;
                double s = weight / Count;
                if (double.IsInfinity(Mean))
                    Mean = s * x + (1 - s) * Mean;
                else
                    Mean += s * diff;
                Variance += s * diff * diff;
                Variance = (1 - s) * Variance;
            }
        }

        /// <summary>
        /// Adds a noisy observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="noiseVariance"></param>
        /// <param name="weight"></param>
        public void Add(double x, double noiseVariance, double weight)
        {
            // new_cov = count/(count + w)*(cov + w/(count+w)*(x-mean)*(x-mean)') + w/(count+w)*noiseVariance
            Add(x, weight);
            if (Count == 0) return;
            double s = weight/Count;
            Variance += s*noiseVariance;
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            Count = 0;
            Mean = 0;
            Variance = 0;
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MeanVarianceAccumulator value)
        {
            Mean = value.Mean;
            Variance = value.Variance;
            Count = value.Count;
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanVarianceAccumulator result = new MeanVarianceAccumulator();
            result.SetTo(this);
            return result;
        }

        public override string ToString()
        {
            return $"MeanVarianceAccumulator(Mean={Mean}, Variance={Variance}, Count={Count})";
        }
    }

    /// <summary>
    /// Decorator of MeanVarianceAccumulator that does not add if any input is NaN.
    /// </summary>
    public class MeanVarianceAccumulatorSkipNaNs : Accumulator<double>, SettableTo<MeanVarianceAccumulatorSkipNaNs>, ICloneable
    {
        private MeanVarianceAccumulator mva = new MeanVarianceAccumulator();

        /// <summary>
        /// The sample mean
        /// </summary>
        public double Mean { get { return mva.Mean; } }

        /// <summary>
        /// Sample variance
        /// </summary>
        public double Variance { get { return mva.Variance; } }

        /// <summary>
        /// Sample count
        /// </summary>
        public double Count { get { return mva.Count; } }

        /// <summary>
        /// Adds an observation
        /// </summary>
        /// <param name="x"></param>
        public void Add(double x)
        {
            if(!double.IsNaN(x))
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
        /// Adds a noisy observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="noiseVariance"></param>
        /// <param name="weight"></param>
        public void Add(double x, double noiseVariance, double weight)
        {
            if(!double.IsNaN(x) && !double.IsNaN(noiseVariance) && !double.IsNaN(weight))
                mva.Add(x, noiseVariance, weight);
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
        public void SetTo(MeanVarianceAccumulatorSkipNaNs value)
        {
            this.mva.SetTo(value.mva);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanVarianceAccumulatorSkipNaNs result = new MeanVarianceAccumulatorSkipNaNs();
            result.SetTo(this);
            return result;
        }
    }

    /// <summary>
    /// Class for accumulating weighted noisy scalar observations,
    /// and computing sample count, mean, and variance
    /// </summary>
    public class MeanVarianceAccumulator2 : SettableTo<MeanVarianceAccumulator2>, ICloneable
    {
        /// <summary>
        /// The sample mean
        /// </summary>
        public double Mean;

        /// <summary>
        /// Sample variance
        /// </summary>
        public double Variance;

        /// <summary>
        /// Sample count
        /// </summary>
        public double LogCount = double.NegativeInfinity;

        /// <summary>
        /// Adds an observation
        /// </summary>
        /// <param name="x"></param>
        public void Add(double x)
        {
            Add(x, 0);
        }

        /// <summary>
        /// Adds a weighted observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="logWeight"></param>
        public void Add(double x, double logWeight)
        {
            // new_mean = mean + w/(count + w)*(x - mean)
            // new_var = count/(count + w)*(var + w/(count+w)*(x-mean)*(x-mean)')
            LogCount = MMath.LogSumExp(LogCount, logWeight);
            if (double.IsNegativeInfinity(LogCount)) return;
            double diff = x - Mean;
            double s = System.Math.Exp(logWeight - LogCount);
            Mean += s*diff;
            Variance += s*diff*diff;
            Variance = (1 - s)*Variance;
        }

        /// <summary>
        /// Adds a noisy observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="noiseVariance"></param>
        /// <param name="logWeight"></param>
        public void Add(double x, double noiseVariance, double logWeight)
        {
            // new_cov = count/(count + w)*(cov + w/(count+w)*(x-mean)*(x-mean)') + w/(count+w)*noiseVariance
            Add(x, logWeight);
            if (double.IsNegativeInfinity(LogCount)) return;
            double s = System.Math.Exp(logWeight - LogCount);
            Variance += s*noiseVariance;
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            LogCount = double.NegativeInfinity;
            Mean = 0;
            Variance = 0;
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MeanVarianceAccumulator2 value)
        {
            Mean = value.Mean;
            Variance = value.Variance;
            LogCount = value.LogCount;
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MeanVarianceAccumulator2 result = new MeanVarianceAccumulator2();
            result.SetTo(this);
            return result;
        }
    }
}