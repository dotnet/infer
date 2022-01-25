// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;

    public class TruncatedDistribution : ITruncatableDistribution<double>, CanGetLogProb<double>
    {
        public readonly double LowerBound, UpperBound;
        public readonly ITruncatableDistribution<double> Distribution;
        private readonly double LowerProbability, TotalProbability;

        public TruncatedDistribution(ITruncatableDistribution<double> distribution, double lowerBound, double upperBound)
        {
            if (lowerBound > upperBound) throw new ArgumentOutOfRangeException($"lowerBound ({lowerBound}) > upperBound ({upperBound})");
            this.Distribution = distribution;
            this.LowerBound = lowerBound;
            this.UpperBound = upperBound;
            this.TotalProbability = distribution.GetProbBetween(lowerBound, upperBound);
            this.LowerProbability = distribution.GetProbLessThan(lowerBound);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"TruncatedDistribution({Distribution}, {LowerBound}, {UpperBound})";
        }

        /// <inheritdoc/>
        public double GetProbLessThan(double x)
        {
            return Math.Min(1, Math.Max(0, Distribution.GetProbLessThan(x) - LowerProbability) / TotalProbability);
        }

        /// <inheritdoc/>
        public double GetQuantile(double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1");
            return Math.Min(UpperBound, Math.Max(LowerBound, Distribution.GetQuantile(probability * TotalProbability + LowerProbability)));
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            lowerBound = Math.Min(UpperBound, Math.Max(LowerBound, lowerBound));
            upperBound = Math.Min(UpperBound, Math.Max(LowerBound, upperBound));
            return Distribution.GetProbBetween(lowerBound, upperBound) / TotalProbability;
        }

        /// <inheritdoc/>
        public ITruncatableDistribution<double> Truncate(double lowerBound, double upperBound)
        {
            if (lowerBound > upperBound) throw new ArgumentOutOfRangeException($"lowerBound ({lowerBound}) > upperBound ({upperBound})");
            if (lowerBound > this.UpperBound) throw new ArgumentOutOfRangeException($"lowerBound ({lowerBound}) > this.UpperBound ({this.UpperBound})");
            if (upperBound < this.LowerBound) throw new ArgumentOutOfRangeException($"upperBound ({upperBound}) < this.LowerBound ({this.LowerBound})");
            return new TruncatedDistribution(Distribution, Math.Max(this.LowerBound, lowerBound), Math.Min(this.UpperBound, upperBound));
        }

        /// <inheritdoc/>
        public double GetLogProb(double value)
        {
            return ((CanGetLogProb<double>)Distribution).GetLogProb(value);
        }
    }
}
