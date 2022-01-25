// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;

    public class TruncatableDistribution<T> : ITruncatableDistribution<T>, CanGetLogProb<T>
    {
        public CanGetProbLessThan<T> CanGetProbLessThan { get; }

        public TruncatableDistribution(CanGetProbLessThan<T> canGetProbLessThan)
        {
            this.CanGetProbLessThan = canGetProbLessThan;
        }

        /// <inheritdoc/>
        public double GetProbBetween(T lowerBound, T upperBound)
        {
            return Math.Max(0.0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
        }

        /// <inheritdoc/>
        public double GetProbLessThan(T x)
        {
            return CanGetProbLessThan.GetProbLessThan(x);
        }

        /// <inheritdoc/>
        public T GetQuantile(double probability)
        {
            return ((CanGetQuantile<T>)CanGetProbLessThan).GetQuantile(probability);
        }

        /// <inheritdoc/>
        public ITruncatableDistribution<T> Truncate(T lowerBound, T upperBound)
        {
            return (ITruncatableDistribution<T>)new TruncatedDistribution((ITruncatableDistribution<double>)this, (double)(object)lowerBound, (double)(object)upperBound);
        }

        /// <inheritdoc/>
        public double GetLogProb(T value)
        {
            return ((CanGetLogProb<T>)CanGetProbLessThan).GetLogProb(value);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"TruncatableDistribution({CanGetProbLessThan})";
        }
    }
}
