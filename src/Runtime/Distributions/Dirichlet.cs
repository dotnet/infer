// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Linq;
    using System.Runtime.Serialization;
    using Math;
    using Collections;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A Dirichlet distribution on probability vectors.
    /// </summary>
    /// <remarks><para>
    /// The Dirichlet is a distribution on probability vectors.
    /// The formula for the distribution is p(x) = (Gamma(a)/prod_i Gamma(b_i)) prod_i x_i^{b_i-1}
    /// subject to the constraints x_i >= 0 and sum_i x_i = 1.
    /// The parameter a is the "total pseudo-count" and is shorthand for sum_i b_i.
    /// The vector b contains the pseudo-counts for each case i. The vector b can be sparse
    /// or dense; in many cases it is useful to give it a <see cref="Sparsity"/> specification of
    /// <see cref="Microsoft.ML.Probabilistic.Math.Sparsity.ApproximateWithTolerance(double)"/>.
    /// </para><para>
    /// The distribution is represented by the pair (TotalCount, PseudoCount).
    /// If TotalCount is infinity, the distribution is a point mass.  The Point property gives the mean.
    /// Otherwise TotalCount is always equal to PseudoCount.Sum().
    /// If distribution is uniform when all PseudoCounts = 1.
    /// If any PseudoCount &lt;= 0, the distribution is improper.
    /// In this case, the density is redefined to not include the Gamma terms, i.e.
    /// there is no normalizer.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public class Dirichlet : IDistribution<Vector>,
                             SettableTo<Dirichlet>, SettableToProduct<Dirichlet>, Diffable, SettableToUniform,
                             SettableToRatio<Dirichlet>, SettableToPower<Dirichlet>, SettableToWeightedSum<Dirichlet>, CanGetLogAverageOf<Dirichlet>,
                             CanGetLogAverageOfPower<Dirichlet>,
                             CanGetAverageLog<Dirichlet>, CanGetLogNormalizer,
                             Sampleable<Vector>, CanGetMean<Vector>, CanGetVariance<Vector>, CanGetMeanAndVariance<Vector, Vector>, CanSetMeanAndVariance<Vector, Vector>,
                             CanGetMode<Vector>
    {
        /// <summary>
        /// Gets the total count. If infinite, the distribution is a point mass.
        /// Otherwise, this is the sum of pseudo-counts
        /// </summary>
        [DataMember]
        public double TotalCount;

        /// <summary>
        /// Vector of pseudo-counts
        /// </summary>
        [DataMember]
        public Vector PseudoCount;

        /// <summary>
        /// Gets the dimension of this Dirichlet
        /// </summary>
        public int Dimension
        {
            get { return PseudoCount.Count; }
        }

        /// <summary>
        /// Gets the <see cref="Sparsity"/> specification of this Distribution.
        /// </summary>
        public Sparsity Sparsity
        {
            get { return PseudoCount.Sparsity; }
        }

        /// <summary>
        /// Whether the the distribution is proprer or not.
        /// It is proper if all pseudo-counts are > 0.
        /// </summary>
        /// <returns>true if proper, false otherwise</returns>
        public bool IsProper()
        {
            return PseudoCount > 0;
        }

        /// <summary>
        /// The most probable vector.
        /// </summary>
        /// <returns></returns>
        public Vector GetMode()
        {
            return GetMode(Vector.Zero(Dimension, Sparsity));
        }

        /// <summary>
        /// The most probable vector.
        /// </summary>
        /// <returns></returns>
        public Vector GetMode(Vector result)
        {
            if (IsPointMass)
            {
                result.SetTo(Point);
            }
            else
            {
                int countGe1 = PseudoCount.Count(c => (c >= 1));
                double sumGe1 = PseudoCount.Sum(c => (c >= 1) ? c : 0);
                if (countGe1 == 0)
                {
                    int index = PseudoCount.IndexOfMaximum();
                    result.SetAllElementsTo(0.0);
                    result[index] = 1.0;
                }
                else if (sumGe1 == countGe1)
                {
                    double scale = 1.0 / countGe1;
                    result.SetToFunction(PseudoCount, c => (c >= 1) ? scale : 0);
                }
                else
                {
                    double scale = 1.0 / (sumGe1 - countGe1);
                    result.SetToFunction(PseudoCount, c => (c > 1) ? (c - 1) * scale : 0);
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the expected value E(x)
        /// </summary>
        /// <returns>E(x)</returns>
        public Vector GetMean()
        {
            return GetMean(Vector.Zero(Dimension, Sparsity));
        }

        /// <summary>
        /// Gets the expected value E(x). Provide a vector to put the result
        /// </summary>
        /// <param name="result">Where to put E(x)</param>
        /// <returns>E(x)</returns>
        public Vector GetMean(Vector result)
        {
            if (IsPointMass)
            {
                result.SetTo(Point);
            }
            else if (!(PseudoCount >= 0))
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                result.SetToProduct(PseudoCount, 1.0 / TotalCount);
            }
            return result;
        }

        /// <summary>
        /// Gets the expected log value E(log(x))
        /// </summary>
        /// <returns>E(log(x))</returns>
        public Vector GetMeanLog()
        {
            return GetMeanLog(Vector.Zero(Dimension, Sparsity));
        }

        /// <summary>
        /// Gets the expected log value E(log(x)). Provide a vector to put the result
        /// </summary>
        /// <param name="result">Where to put E(log(x))</param>
        /// <returns>E(log(x))</returns>
        public Vector GetMeanLog(Vector result)
        {
            if (IsPointMass)
            {
                result.SetToFunction(Point, Math.Log);
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                var digammatot = MMath.Digamma(TotalCount);
                result.SetToFunction(PseudoCount, x => MMath.Digamma(x) - digammatot);
            }
            return result;
        }

        /// <summary>
        /// E[log prob[sample]]
        /// </summary>
        /// <param name="sample">a dimension of prob of interest</param>
        /// <returns>E[log prob[sample]]</returns>
        public double GetMeanLogAt(int sample)
        {
            if (IsPointMass)
            {
                return Math.Log(Point[sample]);
            }
            else
            {
                if (TotalCount > 0)
                {
                    double pseudoCountOfSample = PseudoCount[sample];
                    if (pseudoCountOfSample > 0)
                        return MMath.Digamma(pseudoCountOfSample) - MMath.Digamma(TotalCount);
                }
                throw new ImproperDistributionException(this);
            }
        }

        /// <summary>
        /// Computes E[p(x)^2] for each x.
        /// </summary>
        /// <returns></returns>
        public Vector GetMeanSquare()
        {
            Vector result = Vector.Zero(Dimension, Sparsity);
            if (IsPointMass)
            {
                result.SetToProduct(PseudoCount, PseudoCount);
            }
            else if (!(PseudoCount >= 0))
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double s = 1.0 / (TotalCount * (TotalCount + 1));
                result.SetToFunction(PseudoCount, x => x * (x + 1) * s);
            }
            return result;
        }

        /// <summary>
        /// Computes E[p(x)^3] for each x.
        /// </summary>
        /// <returns></returns>
        public Vector GetMeanCube()
        {
            Vector result = Vector.Zero(Dimension, Sparsity);
            if (IsPointMass)
            {
                result.SetToProduct(PseudoCount, PseudoCount);
                result.SetToProduct(result, PseudoCount);
            }
            else if (!(PseudoCount >= 0))
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double s = 1.0 / (TotalCount * (TotalCount + 1) * (TotalCount + 2));
                result.SetToFunction(PseudoCount, x => x * (x + 1) * (x + 2) * s);
            }
            return result;
        }

        /// <summary>
        /// Gets the variance var(p) = m*(1-m)/(1+s)
        /// </summary>
        /// <returns>The variance</returns>
        public Vector GetVariance()
        {
            Vector result = Vector.Zero(Dimension, Sparsity);
            if (IsPointMass) return result;
            else if (!(PseudoCount >= 0))
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double mult = 1.0 / TotalCount;
                double mult1 = 1.0 / (1.0 + TotalCount);
                result.SetToFunction(PseudoCount, x => (x * mult) * (1.0 - x * mult) * mult1);
                return result;
            }
        }

        /// <summary>
        /// Gets the mean E(p) = m/s and variance var(p) = m*(1-m)/(1+s)
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        public void GetMeanAndVariance(Vector mean, Vector variance)
        {
            if (IsPointMass)
            {
                mean.SetTo(Point);
                variance.SetAllElementsTo(0.0);
            }
            else if (!(PseudoCount >= 0))
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double mult = 1.0 / TotalCount;
                double mult1 = 1.0 / (1.0 + TotalCount);
                mean.SetToProduct(PseudoCount, mult);
                variance.SetToFunction(PseudoCount, x => x * mult * (1 - x * mult) * mult1);
            }
        }

        /// <summary>
        /// Sets the mean, and sets the precision to best match the given mean-squares.
        /// </summary>
        /// <param name="mean">Desired mean in each dimension.  Must be in [0,1] and sum to 1.</param>
        /// <param name="meanSquare">Desired meanSquare in each dimension.  Must be in [0,1].</param>
        /// <remarks>
        /// The resulting distribution will have the given mean but will only approximately match
        /// the meanSquare, since the Dirichlet does not have enough parameters.  The moment matching
        /// formula comes from:
        /// "Expectation-Propagation for the Generative Aspect Model",
        /// Thomas Minka and John Lafferty,
        /// Proceedings of the 18th Conference on Uncertainty in Artificial Intelligence, pp. 352-359, 2002
        /// http://research.microsoft.com/~minka/papers/aspect/
        /// </remarks>
        public void SetMeanAndMeanSquare(Vector mean, Vector meanSquare)
        {
            double sumMean = mean.Sum();
            double sumMeanSq = meanSquare.Sum();
            double sumSqMean = mean.Sum(x => x * x);
            double numer = sumMean - sumMeanSq;
            double denom = sumMeanSq - sumSqMean;
            if (denom == 0.0)
            {
                Point = mean;
            }
            else
            {
                TotalCount = numer / denom;
                PseudoCount.SetToProduct(mean, TotalCount);
            }
        }

        /// <summary>
        /// Sets the mean, and sets the precision to best match the given variances.
        /// </summary>
        /// <param name="mean">Desired mean in each dimension.  Must be in [0,1] and sum to 1.</param>
        /// <param name="variance">Desired variance in each dimension.  Must be non-negative.</param>
        /// <remarks>
        /// The resulting distribution will have the given mean but will only approximately match
        /// the variance, since the Dirichlet does not have enough parameters.  The moment matching
        /// formula comes from:
        /// "Expectation-Propagation for the Generative Aspect Model",
        /// Thomas Minka and John Lafferty,
        /// Proceedings of the 18th Conference on Uncertainty in Artificial Intelligence, pp. 352-359, 2002
        /// http://research.microsoft.com/~minka/papers/aspect/
        /// </remarks>
        public void SetMeanAndVariance(Vector mean, Vector variance)
        {
            double numer = mean.Sum(x => (x * (1 - x)));
            double denom = variance.Sum();
            if (denom == 0.0)
            {
                Point = mean;
            }
            else
            {
                TotalCount = numer / denom - 1;
                PseudoCount.SetToProduct(mean, TotalCount);
            }
        }

        /// <summary>
        /// Sets the mean and precision to best match the given derivatives at a point.
        /// </summary>
        /// <param name="x">A probability vector</param>
        /// <param name="dLogP">Desired derivative of log-density at x</param>
        /// <param name="ddLogP">Desired second derivative of log-density at x</param>
        /// <param name="forceProper">If true and both derivatives cannot be matched by a distribution with counts at least 1, match only the first.</param>
        /// <returns></returns>
        public void SetDerivatives(Vector x, Vector dLogP, Vector ddLogP, bool forceProper)
        {
            // dlogf/dxk = (alpha[k]-1)/x[k] - (alpha[N]-1)/x[N]
            // sum_{k=1}^{N-1} x[k] dlogf/dxk = (sum_k (alpha[k]-1)) - (alpha[N]-1)/x[N]
            // d^2 logf/dxk^2 = -(alpha[k]-1)/x[k]^2 - (alpha[N]-1)/x[N]^2
            // x[k] x[N] d^2 logf/dxk^2 = -(alpha[k]-1)*x[N]/x[k] - (alpha[N]-1)*x[k]/x[N]
            // x[k] * dlogf/dxk - x[k] x[N] d^2 logf/dxk^2 = 
            //   (alpha[k]-1) + (alpha[k]-1)*x[N]/x[k] - (alpha[N]-1)*x[k]/x[N] +(alpha[k]-1)*x[k]/x[N] = 
            //   (alpha[k]-1)*(x[k] + x[N])/x[k]
            // sum_{k=1}^{N-1} x[k] d^2 logf/dxk^2 = (sum_{k=1}^{N-1} -(alpha[k]-1)/x[k]) - (alpha[N]-1)/x[N]^2 + (alpha[N]-1)/x[N]
            // + sum_{k=1}^{N-1} dlogf/dxk = - (alpha[N]-1)/x[N] (N-1) - (alpha[N]-1)/x[N]^2 (1-x[N]) = -(alpha[N]-1)/x[N]*(N-2 + 1/x[N])
            // dlogf/dxk + x[k] d^2 logf/dxk^2 = - (alpha[N]-1)/x[N] -  x[k] (alpha[N]-1)/x[N]^2
            this.PseudoCount.SetToFunction(ddLogP, x, (ddLogPi, xi) => ddLogPi * xi);
            double sum = dLogP.Inner(this.PseudoCount);
            double xN = x[x.Count - 1];
            double alphaN = 1 - sum * xN / (x.Count - 2 + 1 / xN);
            if (alphaN < 1)
                alphaN = 1;
            this.PseudoCount.SetToFunction(x, dLogP, (xi, dLogPi) => 1 + xi * (dLogPi + (alphaN - 1) / xN));
            this.PseudoCount[x.Count - 1] = alphaN;
            this.TotalCount = this.PseudoCount.Sum();
        }

        /// <summary>
        /// Create a Dirichlet distribution with the given expected logarithms.
        /// </summary>
        /// <param name="meanLog">Desired expectation E[log(pk)] for each k.</param>
        /// <returns>A new Dirichlet where GetMeanLog == meanLog</returns>
        /// <remarks>
        /// This function is equivalent to maximum-likelihood estimation of a Dirichlet distribution
        /// from data given by sufficient statistics. 
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization.
        /// Uses the Newton algorithm described in "Estimating a Dirichlet distribution" by T. Minka, 2000.
        /// </remarks>
        public static Dirichlet FromMeanLog(Vector meanLog)
        {
            Dirichlet result = Dirichlet.Uniform(meanLog.Count, meanLog.Sparsity);
            result.SetMeanLog(meanLog);
            return result;
        }

        /// <summary>
        /// Set the Dirichlet parameters to produce the given expected logarithms.
        /// </summary>
        /// <param name="meanLog">Desired expectation E[log(pk)] for each k.</param>
        /// <remarks>
        /// This function is equivalent to maximum-likelihood estimation of a Dirichlet distribution
        /// from data given by sufficient statistics. 
        /// This function is significantly slower than the other setters since it
        /// involves nonlinear optimization.
        /// Uses the Newton algorithm described in "Estimating a Dirichlet distribution" by T. Minka, 2000.
        /// </remarks>
        public void SetMeanLog(Vector meanLog)
        {
            if (meanLog.Count != Dimension) throw new ArgumentException("meanLog.Count (" + meanLog.Count + ") != Dimension (" + Dimension + ")");
            PseudoCount.SetToFunction(PseudoCount, meanLog, (x, y) => Double.IsNegativeInfinity(y)
                                                                          ? 0.0
                                                                          : (Double.IsNaN(x) || Double.IsInfinity(x) || x <= 0) ? 1.0 : x);
            // TotalCount is not used
            EstimateNewton(PseudoCount, meanLog);
            TotalCount = PseudoCount.Sum();
        }

        /// <summary>
        /// Modifies PseudoCount to produce the given expected logarithms.
        /// </summary>
        /// <param name="PseudoCount">On input, the initial guess.  On output, the converged solution.</param>
        /// <param name="meanLog">May be -infinity.</param>
        public static void EstimateNewton(Vector PseudoCount, Vector meanLog)
        {
            int Dimension = PseudoCount.Count;
            Sparsity sparsity = PseudoCount.Sparsity;
            if (meanLog.Count != Dimension) throw new ArgumentException("meanLog.Count (" + meanLog.Count + ") != pseudoCount.Count (" + Dimension + ")");
            double TotalCount = PseudoCount.Sum();
            if (TotalCount == 0) return;
            if (TotalCount < 0) throw new Exception("TotalCount < 0");
            double oldLogLike = MMath.GammaLn(TotalCount);
            if (!(PseudoCount >= 0.0))
                throw new Exception("PseudoCount < 0");
            oldLogLike += PseudoCount.Inner(x => x - 1.0, meanLog, y => Double.IsNegativeInfinity(y) ? 0.0 : y);
            oldLogLike += PseudoCount.Sum(x => -MMath.GammaLn(x), meanLog, y => !Double.IsNegativeInfinity(y));
            // Work spaces
            Vector oldPseudoCount = Vector.Zero(Dimension, sparsity);
            Vector gradient = Vector.Zero(Dimension, sparsity);
            Vector trigammaPseudoCount = Vector.Zero(Dimension, sparsity);
            Vector inv_hessian = Vector.Zero(Dimension, sparsity);
            Vector gih = Vector.Zero(Dimension, sparsity);

            // lambda is an adaptive stepsize
            double lambda = 0.1;
            for (int iter = 0; iter < 100; iter++)
            {
                double oldTotalCount = TotalCount;
                oldPseudoCount.SetTo(PseudoCount);
                // TotalCount must always be in sync with PseudoCount
                if (TotalCount == 0) break;
                double digammaTotalCount = MMath.Digamma(TotalCount);
                double inv_trigammaTotalCount = 1 / MMath.Trigamma(TotalCount);
                trigammaPseudoCount.SetToFunction(PseudoCount, meanLog, (x, y) => Double.IsNegativeInfinity(y) ? Double.NegativeInfinity : MMath.Trigamma(x));
                gradient.SetToFunction(PseudoCount, meanLog, (x, y) => Double.IsNegativeInfinity(y) ? 0.0 : digammaTotalCount - MMath.Digamma(x) + y);
                while (true)
                {
                    inv_hessian.SetToFunction(trigammaPseudoCount, x => -1.0 / (x + lambda));
                    double sum_ih = inv_hessian.Sum();
                    double sum_gih = Vector.InnerProduct(inv_hessian, gradient);
                    double b = sum_gih / (inv_trigammaTotalCount + sum_ih);
                    // update PseudoCount, TotalCount, digammaPseudoCount, digammaTotalCount, logLike
                    double logLike = 0.0;
                    double activity = 0.0;
                    gih.SetToDifference(gradient, b);
                    gih.SetToProduct(gih, inv_hessian);
                    if (!(gih <= PseudoCount))
                        goto fail;
                    activity = gih.Max(Math.Abs);
                    PseudoCount.SetToDifference(PseudoCount, gih);
                    TotalCount = PseudoCount.Sum();
                    logLike += PseudoCount.Inner(x => x - 1.0, meanLog, y => Double.IsNegativeInfinity(y) ? 0.0 : y);
                    logLike += PseudoCount.Sum(x => -MMath.GammaLn(x), meanLog, y => !Double.IsNegativeInfinity(y));
                    if (TotalCount <= 0) throw new Exception("TotalCount <= 0");
                    logLike += MMath.GammaLn(TotalCount);
                    if (logLike > oldLogLike)
                    {
                        // success
                        lambda /= 10;
                        oldLogLike = logLike;
                        if (activity < 1e-10) return;
                        break;
                    }
                    fail:
                    // failed to increase likelihood
                    lambda *= 10;
                    if (lambda > 1e6) return; //throw new Exception("Newton iteration failed to converge");
                    TotalCount = oldTotalCount;
                    PseudoCount.SetTo(oldPseudoCount);
                }
            }
        }

        /// <summary>
        /// Clones this Dirichlet. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Dirichlet type</returns>
        public object Clone()
        {
            return new Dirichlet(this);
        }

        /// <summary>
        /// Sets/gets this distribution as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public Vector Point
        {
            get { return PseudoCount; }
            set
            {
                TotalCount = Double.PositiveInfinity;
                PseudoCount.SetTo(value);
            }
        }

        /// <summary>
        /// Whether this Dirichlet is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return Double.IsPositiveInfinity(TotalCount); }
        }

        /// <summary>
        /// The maximum difference between the parameters of this Dirichlet
        /// and that Dirichlet
        /// </summary>
        /// <param name="that">That Dirichlet</param>
        /// <returns>The maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object that)
        {
            Dirichlet thatd = that as Dirichlet;
            if (thatd == null) return Double.PositiveInfinity;
            return Math.Max(MMath.AbsDiff(TotalCount, thatd.TotalCount), PseudoCount.MaxDiff(thatd.PseudoCount));
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            Dirichlet that = thatd as Dirichlet;
            if (that == null) return false;
            return (MaxDiff(that) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(PseudoCount.GetHashCode(), TotalCount.GetHashCode());
        }

        /// <summary>
        /// Evaluates the log of the Dirichlet density function at the given Vector value
        /// </summary>
        /// <param name="value">Where to do the evaluation. Must be vector of positive real numbers</param>
        /// <returns>log(Dir(value;a,b))</returns>
        public double GetLogProb(Vector value)
        {
            if (IsPointMass)
            {
                return (value == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else
            {
                double p = -DirichletLn(PseudoCount);
                double init = -DirichletLn(PseudoCount);
                p += PseudoCount.Inner(x => x - 1.0, value, Math.Log);
                return p;
            }
        }

        /// <summary>
        /// Gets the log normalizer for the distribution
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            return DirichletLn(PseudoCount);
        }

        /// <summary>
        /// Computes the log Dirichlet function: <c>sum_i GammaLn(pseudoCount[i]) - GammaLn(sum_i pseudoCount[i])</c>
        /// </summary>
        /// <param name="pseudoCount">Vector of pseudo-counts.</param>
        /// <returns><c>sum_i GammaLn(pseudoCount[i]) - GammaLn(sum_i pseudoCount[i])</c></returns>
        /// <remarks>
        /// If any pseudoCount &lt;= 0, the result is defined to be 0.
        /// </remarks>
        public static double DirichletLn(Vector pseudoCount)
        {
            // If ANY value is <= 0.0, return 0.0
            if (!(pseudoCount > 0))
                return 0.0;
            double totalCount = pseudoCount.Sum();
            double result = pseudoCount.Sum(MMath.GammaLn);
            result -= MMath.GammaLn(totalCount);
            return result;
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Dirichlet that)
        {
            if (this.Dimension != that.Dimension)
                throw new ArgumentException(String.Format("that.Dimension ({0}) does not match this.Dimension ({1})", that.Dimension, this.Dimension));
            if (that.IsPointMass)
            {
                if (this.IsPointMass && Point.Equals(that.Point)) return 0.0;
                else return Double.NegativeInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double result = -DirichletLn(that.PseudoCount);
                if (IsPointMass)
                    result += that.PseudoCount.Inner(x => x - 1.0, this.Point, Math.Log);
                else
                {
                    result += that.PseudoCount.Inner(x => x - 1.0, this.PseudoCount, MMath.Digamma);
                    result += -(that.TotalCount - that.Dimension) * MMath.Digamma(TotalCount);
                }
                return result;
            }
        }

        /// <summary>
        /// Sets this Dirichlet instance to have the parameter values of another Dirichlet instance
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(Dirichlet value)
        {
            if (object.ReferenceEquals(value, null)) SetToUniform();
            else
            {
                TotalCount = value.TotalCount;
                PseudoCount.SetTo(value.PseudoCount);
            }
        }

        /// <summary>
        /// Sets the parameters to represent the product of two Dirichlets.
        /// </summary>
        /// <param name="a">The first Dirichlet.  May refer to <c>this</c>.</param>
        /// <param name="b">The second Dirichlet.  May refer to <c>this</c>.</param>
        /// <remarks>
        /// The result may not be proper, i.e. its parameters may be negative.
        /// For example, if you multiply Dirichlet(0.1,0.1) by itself you get Dirichlet(-0.8, -0.8).
        /// No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(Dirichlet a, Dirichlet b)
        {
            if (a.IsPointMass || b.IsUniform())
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                SetTo(a);
            }
            else if (b.IsPointMass || a.IsUniform())
            {
                SetTo(b);
            }
            else
            {
                // must catch uniform case above because the following code can produce round-off errors
                TotalCount = a.TotalCount + b.TotalCount - Dimension;
                PseudoCount.SetToFunction(a.PseudoCount, b.PseudoCount, (x, y) => x + y - 1);

                // Make pseudo-count dense if sparse count is too big.
                if (PseudoCount.IsSparse() && ((SparseVector)PseudoCount).SparseCount > 0.50 * Dimension)
                    PseudoCount = ((SparseVector)PseudoCount).ToVector();
                //PseudoCount.SetToSum(a.PseudoCount, b.PseudoCount);
                //PseudoCount.SetToSum(PseudoCount, -1);
            }
        }

        /// <summary>
        /// Creates a Dirichlet distribution which is the product of two Dirichlet distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting Dirichlet distribution</returns>
        public static Dirichlet operator *(Dirichlet a, Dirichlet b)
        {
            Dirichlet result = Dirichlet.Uniform(a.Dimension, a.Sparsity);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two Dirichlets.
        /// </summary>
        /// <param name="numerator">The numerator Dirichlet.  Can be the same object as this.</param>
        /// <param name="denominator">The denominator Dirichlet.  Can be the same object as this.</param>
        /// <param name="forceProper">If true, the PseudoCounts of the result are made >= 1, under the constraint that denominator*result has the same mean as numerator.</param>
        public void SetToRatio(Dirichlet numerator, Dirichlet denominator, bool forceProper = false)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        SetToUniform();
                    }
                    else
                    {
                        throw new DivideByZeroException();
                    }
                }
                else
                {
                    Point = numerator.Point;
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else if (forceProper && numerator.PseudoCount.Any(denominator.PseudoCount, (x, y) => (x < y)))
            {
                // constraint is: PseudoCount[i] >= 1, (PseudoCount[i]+denominator[i]-1)/(TotalCount+denominator.TotalCount-K) = numerator[i]/numerator.TotalCount
                // solution is: PseudoCount[i] = s*numerator[i] - denominator[i] + 1
                // where s = max(1, max_i denominator[i]/numerator[i])
                // note the scale of the numerator is irrelevant

                if (!numerator.IsProper())
                {
                    throw new ImproperDistributionException(numerator);
                }

                var ratio = denominator.PseudoCount / numerator.PseudoCount;
                double s = Math.Max(1.0, ratio.Max());

                TotalCount = s * numerator.TotalCount - denominator.TotalCount + Dimension;
                PseudoCount.SetToFunction(numerator.PseudoCount, denominator.PseudoCount, (x, y) => Math.Max(s * x - y + 1, 1));
            }
            else
            {
                TotalCount = numerator.TotalCount - denominator.TotalCount + Dimension;
                PseudoCount.SetToDifference(numerator.PseudoCount, denominator.PseudoCount);
                PseudoCount.SetToSum(PseudoCount, 1);
            }
        }

        /// <summary>
        /// Creates a Dirichlet distribution which is the ratio of two Dirichlet distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <returns>The resulting Dirichlet distribution</returns>
        public static Dirichlet operator /(Dirichlet numerator, Dirichlet denominator)
        {
            Dirichlet result = Dirichlet.Uniform(numerator.Dimension, numerator.Sparsity);
            result.SetToRatio(numerator, denominator, false);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the raising a Dirichlet to some power.
        /// </summary>
        /// <param name="dist">The Dirichlet</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Dirichlet dist, double exponent)
        {
            if (dist.IsPointMass)
            {
                if (exponent == 0)
                {
                    SetToUniform();
                }
                else if (exponent < 0)
                {
                    throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                }
                else
                {
                    Point = dist.Point;
                }
                return;
            }
            else
            {
                TotalCount = (dist.TotalCount - Dimension) * exponent + Dimension;
                PseudoCount.SetToFunction(dist.PseudoCount, x => ((x - 1) * exponent + 1));
                /*PseudoCount.SetToSum(dist.PseudoCount, -1);
                PseudoCount.Scale(exponent);
                PseudoCount.SetToSum(PseudoCount, 1);*/
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Dirichlet operator ^(Dirichlet dist, double exponent)
        {
            Dirichlet result = new Dirichlet(dist.Dimension, dist.Sparsity);
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// If true, <see cref="SetToSum"/> will use moment matching as described by Minka and Lafferty (2002).
        /// </summary>
        public static bool AllowImproperSum;

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="dist1">The first distribution.  Can be the same object as <c>this</c></param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist2">The second distribution.  Can be the same object as <c>this</c></param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, Dirichlet dist1, double weight2, Dirichlet dist2)
        {
            if (AllowImproperSum)
            {
                WeightedSum<Dirichlet>(this, Dimension, weight1, dist1, weight2, dist2, Sparsity);
                return;
            }
            if (weight1 + weight2 == 0) SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0) SetTo(dist2);
            else if (weight2 == 0) SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2)) SetTo(dist1);
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }
                else
                {
                    SetTo(dist1);
                }
            }
            else if (double.IsPositiveInfinity(weight2)) SetTo(dist2);
            else
            {
                Vector minCount = Vector.Zero(Dimension, Sparsity);
                if (dist1.IsPointMass)
                {
                    if (dist2.IsPointMass)
                    {
                        if (!dist1.Point.Equals(dist2.Point))
                            throw new AllZeroException("dist1.Point = " + dist1.Point + Environment.NewLine + "dist2.Point = " + dist2.Point);
                        Point = dist1.Point;
                        return;
                    }
                    else
                    {
                        minCount.SetTo(dist2.PseudoCount);
                    }
                }
                else if (dist2.IsPointMass)
                {
                    minCount.SetTo(dist1.PseudoCount);
                }
                else
                {
                    minCount.SetToFunction(dist1.PseudoCount, dist2.PseudoCount, (x, y) => Math.Min(x, y));
                }
                // algorithm: we choose the result to have the same mean and variance as the mixture
                // provided that all PseudoCounts are greater than the smallest PseudoCount in the mixture.
                // The result has the form (s*m[1],s*m[2],...)  where the mean m[k] is fixed and s satisfies
                // s*m[k] >= min_i dist[i].PseudoCount[k]    i.e.   s >= min[k]/m[k]  for all k
                // if weight2 < 0 then we want dist1[k] >= min(dist2[k],s*m[k])  i.e. s*m[k] <= dist1[k]  when dist1[k] < dist2[k]
                if (minCount.EqualsAll(0))
                {
                    PseudoCount.SetAllElementsTo(0);
                    TotalCount = 0;
                }
                else
                {
                    Dirichlet momentMatch = Dirichlet.Uniform(Dimension, Sparsity);
                    WeightedSum<Dirichlet>(momentMatch, Dimension, weight1, dist1, weight2, dist2, Sparsity);
                    Vector mean = momentMatch.GetMean();
                    Vector ratio = minCount / mean;
                    Vector pseudoCountDiff = dist1.PseudoCount - dist2.PseudoCount;
                    double bound;
                    bool boundViolated;
                    if (weight1 > 0)
                    {
                        if (weight2 > 0)
                        {
                            bound = ratio.Max();
                            boundViolated = (momentMatch.TotalCount < bound);
                        }
                        else
                        {
                            pseudoCountDiff.SetToFunction(pseudoCountDiff, ratio, (d, r) => d < 0 ? r : double.PositiveInfinity);
                            bound = pseudoCountDiff.Min();
                            boundViolated = (momentMatch.TotalCount > bound);
                        }
                    }
                    else
                    {
                        pseudoCountDiff.SetToFunction(pseudoCountDiff, ratio, (d, r) => d > 0 ? r : double.PositiveInfinity);
                        bound = pseudoCountDiff.Min();
                        boundViolated = (momentMatch.TotalCount > bound);
                    }
                    if (boundViolated)
                    {
                        TotalCount = bound;
                        PseudoCount.SetToProduct(mean, bound);
                    }
                    else
                    {
                        SetTo(momentMatch);
                    }
                }
            }
            if (dist1.IsProper() && dist2.IsProper() && !IsProper()) throw new ImproperDistributionException(this);
        }

        /// <summary>
        /// Static weighted sum method for distribution types for which both mean and variance
        /// can be got/set as Vectors
        /// </summary>
        /// <typeparam name="T">The distribution type</typeparam>
        /// <param name="result">The resulting distribution</param>
        /// <param name="dimension">The vector dimension</param>
        /// <param name="weight1">First weight</param>
        /// <param name="dist1">First distribution instance</param>
        /// <param name="weight2">Second weight</param>
        /// <param name="dist2">Second distribution instance</param>
        /// <param name="sparsity">Vector sparsity specification</param>
        /// <returns>Resulting distribution</returns>
        public static T WeightedSum<T>(T result, int dimension, double weight1, T dist1, double weight2, T dist2, Sparsity sparsity)
            where T : CanGetMeanAndVariance<Vector, Vector>, CanSetMeanAndVariance<Vector, Vector>, SettableToUniform, SettableTo<T>
        {
            if (weight1 + weight2 == 0) result.SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0) result.SetTo(dist2);
            else if (weight2 == 0) result.SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2)) result.SetTo(dist1);
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }
                else
                {
                    result.SetTo(dist1);
                }
            }
            else if (double.IsPositiveInfinity(weight2)) result.SetTo(dist2);
            else
            {
                // w = weight1/(weight1 + weight2)
                // m = w*m1 + (1-w)*m2
                // v+m^2 = w*(v1+m1^2) + (1-w)*(v2+m2^2)
                // v = w*v1 + (1-w)*v2 + w*(m1-m)^2 + (1-w)*(m2-m)^2
                Vector m1 = Vector.Zero(dimension, sparsity);
                Vector v1 = Vector.Zero(dimension, sparsity);
                Vector m2 = Vector.Zero(dimension, sparsity);
                Vector v2 = Vector.Zero(dimension, sparsity);
                Vector m = Vector.Zero(dimension, sparsity);
                Vector v = Vector.Zero(dimension, sparsity);
                dist1.GetMeanAndVariance(m1, v1);
                dist2.GetMeanAndVariance(m2, v2);
                if (m1.Equals(m2))
                {
                    // catch this to avoid roundoff errors
                    v.SetToSum(weight1, v1, weight2, v2);
                    v.Scale(1.0 / (weight1 + weight2));
                    result.SetMeanAndVariance(m1, v);
                }
                else
                {
                    double invZ = 1.0 / (weight1 + weight2);
                    m.SetToSum(weight1, m1, weight2, m2);
                    m.Scale(invZ);
                    // m1' = m1-m
                    m1.SetToDifference(m1, m);
                    m2.SetToDifference(m2, m);
                    v.SetToSum(weight1, v1, weight2, v2);
                    // m1' = (m1-m)^2
                    m1.SetToProduct(m1, m1);
                    v.SetToSum(1, v, weight1, m1);
                    m2.SetToProduct(m2, m2);
                    // v = w*v1 + (1-w)*v2 + w*(m1-m)^2 + (1-w)*(m2-m)^2
                    v.SetToSum(1, v, weight2, m2);
                    v.Scale(invZ);
                    result.SetMeanAndVariance(m, v);
                }
            }
            return result;
        }

        /// <summary>
        /// Sets the distribution to be uniform
        /// </summary>
        public void SetToUniform()
        {
            TotalCount = Dimension;
            PseudoCount.SetAllElementsTo(1);
        }

        /// <summary>
        /// Whether this instance is uniform (i.e. has unit pseudo-counts)
        /// </summary>
        /// <returns>true if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return (TotalCount == Dimension) && (PseudoCount.EqualsAll(1));
        }

        /// <summary>
        /// The log of the integral of the product of this Dirichlet and that Dirichlet
        /// </summary>
        /// <param name="that">That Dirichlet</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(Dirichlet that)
        {
            if (this.Dimension != that.Dimension)
                throw new ArgumentException(String.Format("that.Dimension ({0}) does not match this.Dimension ({1})", that.Dimension, this.Dimension));
            if (IsPointMass)
            {
                return that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                return GetLogProb(that.Point);
            }
            else
            {
                // int_p Dir(p;a1) Dir(p;a2) d_p = int_p Dir(p;a1+a2-1)*z
                // where z = (prod_i Choose(a1i+a2i-2,a1i-1)) * Gamma(sum(a1))*Gamma(sum(a2))/Gamma(sum(a1+a2-1))
                // TODO: remove this allocation
                Vector newCount = Vector.Zero(Dimension, Sparsity);
                newCount.SetToSum(PseudoCount, that.PseudoCount);
                newCount.SetToDifference(newCount, 1.0);
                return DirichletLn(newCount) - DirichletLn(PseudoCount) - DirichletLn(that.PseudoCount);
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Dirichlet that, double power)
        {
            if (IsPointMass)
            {
                return power * that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this * (that ^ power);
                return product.GetLogNormalizer() - this.GetLogNormalizer() - power * that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Samples from this Dirichlet distribution
        /// </summary>
        /// <returns>The sample Vector</returns>
        [Stochastic]
        public Vector Sample()
        {
            Sparsity resultSparsity = Sparsity;
            if (PseudoCount.IsSparse && ((SparseVector)PseudoCount).CommonValue != 0.0)
                resultSparsity = Sparsity.Dense;
            return Sample(Vector.Zero(Dimension, resultSparsity));
        }

        /// <summary>
        /// Samples from this Dirichlet distribution. Provide a Vector to place the result
        /// </summary>
        /// <param name="result">Where to place the resulting sample</param>
        /// <returns>result</returns>
        [Stochastic]
        public Vector Sample(Vector result)
        {
            if (IsPointMass)
            {
                result.SetTo(Point);
                return result;
            }
            else
            {
                return Sample(PseudoCount, result);
            }
        }

        /// <summary>
        /// Sample from a Dirichlet with specified pseudo-counts
        /// </summary>
        /// <param name="pseudoCount">The pseudo-count vector</param>
        /// <returns>A new Vector</returns>
        [Stochastic]
        public static Vector SampleFromPseudoCounts(Vector pseudoCount)
        {
            Vector result = Vector.Copy(pseudoCount);
            return Sample(pseudoCount, result);
        }

        /// <summary>
        /// Sample from a Dirichlet with specified pseudo-counts
        /// </summary>
        /// <param name="pseudoCount">The pseudo-count vector</param>
        /// <param name="result">Where to put the result</param>
        /// <returns>result</returns>
        [Stochastic]
        public static Vector Sample(Vector pseudoCount, Vector result)
        {
            return Rand.Dirichlet(pseudoCount, result);
        }

        /// <summary>
        /// Parameterless constructor required for serialization 
        /// </summary>
        private Dirichlet()
        {
        }

        /// <summary>
        /// Creates a uniform Dirichlet distribution with unit pseudo-counts.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        protected Dirichlet(int dimension)
            : this(dimension, 1)
        {
        }

        /// <summary>
        /// Creates a uniform Dirichlet distribution with unit pseudo-counts and a given dimension
        /// and <see cref="Sparsity"/>.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification. A specification of <see cref="Microsoft.ML.Probabilistic.Math.Sparsity.ApproximateWithTolerance(double)"/>
        /// is recommended for sparse problems.</param>
        protected Dirichlet(int dimension, Sparsity sparsity)
            : this(dimension, 1, sparsity)
        {
        }

        /// <summary>
        /// Creates a uniform Dirichlet distribution with the specified initial pseudo-count for each index.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="initialCount">Initial value for each pseudocount</param>
        protected Dirichlet(int dimension, double initialCount)
        {
            PseudoCount = Vector.Zero(dimension);
            PseudoCount.SetAllElementsTo(initialCount);
            TotalCount = dimension * initialCount;
        }

        /// <summary>
        /// Creates a uniform Dirichlet distribution with the specified dimension, initial pseudo-count
        /// and <see cref="Sparsity"/>.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="initialCount">Initial value for each pseudocount</param>
        /// <param name="sparsity">The <see cref="Sparsity"/> specification. A specification of <see cref="Microsoft.ML.Probabilistic.Math.Sparsity.ApproximateWithTolerance(double)"/>
        /// is recommended for sparse problems.</param>
        protected Dirichlet(int dimension, double initialCount, Sparsity sparsity)
        {
            PseudoCount = Vector.Zero(dimension, sparsity);
            PseudoCount.SetAllElementsTo(initialCount);
            TotalCount = dimension * initialCount;
        }

        /// <summary>
        /// Creates a Dirichlet distribution with the specified pseudo-counts.
        /// The pseudo-count vector can have any <see cref="Sparsity"/> specification.
        /// A specification of <see cref="Microsoft.ML.Probabilistic.Math.Sparsity.ApproximateWithTolerance(double)"/>
        /// is recommended for sparse problems, and the message functions used
        /// in inference will maintain that sparsity specification.
        /// </summary>
        /// <param name="pseudoCount">The vector of pseudo-counts</param>
        [Construction("PseudoCount")]
        public Dirichlet(Vector pseudoCount)
            : this(pseudoCount.Count, pseudoCount.Sparsity)
        {
            PseudoCount.SetTo(pseudoCount);
            TotalCount = PseudoCount.Sum();
        }

        /// <summary>
        /// Creates a Dirichlet distribution with the psecified pseudo-counts
        /// </summary>
        /// <param name="pseudoCount">An array of pseudo-counts</param>
        public Dirichlet(params double[] pseudoCount)
            : this(Vector.FromArray(pseudoCount))
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Dirichlet(Dirichlet that)
            : this(that.Dimension, that.Sparsity)
        {
            SetTo(that);
        }

        /// <summary>
        /// Creates a point-mass Dirichlet at the specified location
        /// </summary>
        /// <param name="mean">Where to locate the point-mass. All elements of the Vector must be positive</param>
        /// <returns>The created point mass Dirichlet</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static Dirichlet PointMass(Vector mean)
        {
            Dirichlet d = new Dirichlet(mean.Count, mean.Sparsity);
            d.Point = mean;
            return d;
        }

        /// <summary>
        /// Instantiates a uniform Dirichlet distribution
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <returns>A new uniform Dirichlet distribution</returns>
        [Skip]
        public static Dirichlet Uniform(int dimension)
        {
            Dirichlet result = new Dirichlet(dimension);
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Instantiates a uniform Dirichlet distribution of a given sparsity
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="sparsity">Sparsity</param>
        /// <returns>A new uniform Dirichlet distribution</returns>
        [Construction("Dimension", "Sparsity", UseWhen = "IsUniform"), Skip]
        public static Dirichlet Uniform(int dimension, Sparsity sparsity)
        {
            Dirichlet result = new Dirichlet(dimension, sparsity);
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a Dirichlet distribution with all pseudo-counts equal to initialCount.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="pseudoCount">The value for each pseudo-count</param>
        /// <returns>A new Dirichlet distribution</returns>
        public static Dirichlet Symmetric(int dimension, double pseudoCount)
        {
            return new Dirichlet(dimension, pseudoCount);
        }

        /// <summary>
        /// Creates a Dirichlet distribution of a given sparsity with all pseudo-counts equal to initialCount.
        /// </summary>
        /// <param name="dimension">Dimension</param>
        /// <param name="pseudoCount">The value for each pseudo-count</param>
        /// <param name="sparsity">Sparsity specification</param>
        /// <returns>A new Dirichlet distribution</returns>
        public static Dirichlet Symmetric(int dimension, double pseudoCount, Sparsity sparsity)
        {
            return new Dirichlet(dimension, pseudoCount, sparsity);
        }

        /// <summary>
        /// Creates a point-mass Dirichlet at the specified location
        /// </summary>
        /// <param name="mean">Where to locate the point-mass. All elements of the array must be positive</param>
        /// <returns>The created point mass Dirichlet</returns>
        public static Dirichlet PointMass(params double[] mean)
        {
            return PointMass(Vector.FromArray(mean));
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns>String representation of the instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            return ToString("g4");
        }

        public string ToString(string format)
        {
            return ToString(format, " ");
        }

        public string ToString(string format, string delimiter)
        {
            if (IsPointMass)
            {
                return "Dirichlet.PointMass(" + Point + ")";
            }
            return "Dirichlet(" + PseudoCount.ToString(format, delimiter) + ")";
        }
    }
}