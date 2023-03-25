using Microsoft.ML.Probabilistic.Math;

using System;
using System.Threading;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Represent an approximate distribution for which we can estimate the error of CDF.
    /// </summary>
    public interface IEstimatedDistribution : ITruncatableDistribution<double>
    {
        /// <summary>
        /// Returns an absolute error for the given approximate probability of an interval (x,y),        
        /// in particular, returns error = |(cdf_appr(y) - cdf_a_appr(x)) - (cdf_exact(y) - cdf_exact(x))|,
        /// such that exactProb is within (approximateProb - error, approximateProb + error).
        /// </summary>
        /// <param name="approximateProb"></param>
        /// <returns></returns>
        double GetProbBetweenError(double approximateProb);

        /// <summary>
        /// Compute bounds on the expected value of a function over an interval whose endpoints are uncertain.
        /// </summary>
        /// <param name="maximumError"></param>
        /// <param name="cancellationToken"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="preservesPoint">If true, the output interval approaches a point as the input interval does.</param>
        /// <param name="function"></param>
        /// <returns></returns>
        Interval GetExpectation(double maximumError, CancellationToken cancellationToken, Interval left, Interval right, bool preservesPoint, Func<Interval, Interval> function);
    }
}
