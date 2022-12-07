using Microsoft.ML.Probabilistic.Math;

using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Represent an approximate distribution for which we can estimate the error of CDF.
    /// </summary>
    /// <remarks>
    /// Default implementation of this class considers the estimation as exact.
    /// </remarks>
    public class EstimatedDistribution : IEstimatedDistribution
    {
        public readonly ITruncatableDistribution<double> Distribution;

        public EstimatedDistribution(CanGetProbLessThan<double> distribution)
        {
            if (distribution is EstimatedDistribution estimatedDistribution)
            {
                // Avoids an extra layer of wrapping.
                this.Distribution = estimatedDistribution.Distribution;
            }
            else if (distribution is ITruncatableDistribution<double> truncatableDistribution)
            {
                this.Distribution = truncatableDistribution;
            }
            else
            {
                this.Distribution = new TruncatableDistribution<double>(distribution);
            }
        }

        /// <summary>
        /// Returns an instance of <see cref="EstimatedDistribution"/> representing the given <paramref name="distribution"/>
        /// as if it estimates the real distribution exactly.
        /// </summary>
        /// <param name="distribution"></param>
        /// <returns></returns>
        public static EstimatedDistribution NoError(CanGetProbLessThan<double> distribution)
            => new EstimatedDistribution(distribution);

        /// <inheritdoc/>
        public virtual double GetProbBetweenError(double approximateProb)
        {
            return 0.0; // the estimation is exact by default
        }

        /// <inheritdoc/>
        public virtual Interval GetExpectation(double maximumError, CancellationToken cancellationToken, Interval left, Interval right, bool preservesPoint, Func<Interval, Interval> function)
        {
            return Interval.GetExpectation(maximumError, cancellationToken, left, right, this, preservesPoint, function);
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound) => Distribution.GetProbBetween(lowerBound, upperBound);

        /// <inheritdoc/>
        public double GetProbLessThan(double x) => Distribution.GetProbLessThan(x);

        /// <inheritdoc/>
        public double GetQuantile(double probability) => Distribution.GetQuantile(probability);

        /// <inheritdoc/>
        public ITruncatableDistribution<double> Truncate(double lowerBound, double upperBound) => Distribution.Truncate(lowerBound, upperBound);

        /// <inheritdoc/>
        public override string ToString()
        {
            return this.Distribution.ToString();
        }
    }
}
