// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// A Beta distribution over the interval [0,1].
    /// </summary>
    /// <remarks><para>
    /// The Beta is often used as a distribution on probability values.
    /// The formula for the distribution is <c>p(x) = (Gamma(trueCount+falseCount)/Gamma(trueCount)/Gamma(falseCount)) x^{trueCount-1} (1-x)^(falseCount-1)</c>
    /// subject to the constraint 0 &lt;= x &lt;= 1.
    /// </para><para>
    /// If trueCount = falseCount = 1, the distribution is uniform.
    /// If falseCount = infinity, the distribution is redefined to be a point mass on trueCount.
    /// When trueCount &lt;= 0 or falseCount &lt;= 0, the distribution is improper.
    /// In this case, the density is redefined to not include the Gamma terms, i.e.
    /// there is no normalizer.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public struct Beta : IDistribution<double>,
                         SettableTo<Beta>, SettableToProduct<Beta>, Diffable, SettableToUniform,
                         SettableToRatio<Beta>, SettableToPower<Beta>, SettableToWeightedSum<Beta>, CanGetLogAverageOf<Beta>, CanGetLogAverageOfPower<Beta>,
                         CanGetAverageLog<Beta>, CanGetLogNormalizer,
                         Sampleable<double>, CanGetMean<double>, CanGetVariance<double>, CanGetMeanAndVarianceOut<double, double>, CanSetMeanAndVariance<double, double>,
                         CanGetMode<double>, CanGetProbLessThan<double>
    {
        /// <summary>
        /// True count
        /// </summary>
        [DataMember]
        public double TrueCount;

        /// <summary>
        /// False count
        /// </summary>
        [DataMember]
        public double FalseCount;

        /// <summary>
        /// The sum of TrueCount and FalseCount.
        /// </summary>
        [IgnoreDataMember]
        public double TotalCount
        {
            get { return TrueCount + FalseCount; }
        }

        /// <summary>
        /// Whether the distribution is proper or not. It is improper
        /// when trueCount &lt;= 0 or falseCount &lt;= 0 in which case it
        /// cannot be normalised
        /// </summary>
        /// <returns>true if proper, false otherwise</returns>
        public bool IsProper()
        {
            return (TrueCount > 0) && (FalseCount > 0);
        }

        /// <summary>
        /// Compute the probability that a sample from this distribution is less than x.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns>The cumulative beta distribution at <paramref name="x"/></returns>
        public double GetProbLessThan(double x)
        {
            if (x < 0.0)
            {
                return 0.0;
            }
            else if (x > 1.0)
            {
                return 1.0;
            }
            else if (this.IsPointMass)
            {
                return (this.Point < x) ? 1.0 : 0.0;
            }
            else if (this.IsUniform())
            {
                return x;
            }
            else
                return MMath.Beta(x, this.TrueCount, this.FalseCount);
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            return System.Math.Max(0.0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
        }

        /// <summary>
        /// The most probable value.
        /// </summary>
        /// <returns></returns>
        public double GetMode()
        {
            if (IsPointMass)
            {
                return Point;
            }
            else if (TrueCount == 0 && TotalCount == 0)
            {
                return 0.5;
            }
            else if (IsUniform())
            {
                return 0.5;
            }
            else if (TrueCount < 1)
            {
                if (TrueCount < FalseCount)
                    return 0.0;
                else
                    return 1.0;
            }
            else if (FalseCount < 1)
            {
                return 1.0;
            }
            else
            {
                // TrueCount >= 1 and FalseCount >= 1 and TotalCount > 2
                double result = (TrueCount - 1)/(TotalCount - 2);
                if (result > 1)
                    return 1;
                else
                    return result;
            }
        }

        /// <summary>
        /// The expected value E[p].
        /// </summary>
        /// <returns><c>TrueCount/TotalCount</c></returns>
        /// <remarks>The result must be between 0 and 1.</remarks>
        public double GetMean()
        {
            if (IsPointMass)
            {
                return Point;
            }
            else if (TrueCount == 0 && TotalCount == 0)
            {
                return 0.5;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                return TrueCount/TotalCount;
            }
        }

        /// <summary>
        /// The expected square E[p^2].
        /// </summary>
        /// <returns></returns>
        public double GetMeanSquare()
        {
            if (IsPointMass)
            {
                return Point*Point;
            }
            else
            {
                return GetMean()*(TrueCount + 1)/(TotalCount + 1);
            }
        }

        /// <summary>
        /// The expected cube E[p^3].
        /// </summary>
        /// <returns></returns>
        public double GetMeanCube()
        {
            if (IsPointMass)
            {
                return Point*Point*Point;
            }
            else
            {
                return GetMeanSquare()*(TrueCount + 2)/(TotalCount + 2);
            }
        }

        /// <summary>
        /// The expected logarithm E[log(p)].
        /// </summary>
        /// <returns></returns>
        public double GetMeanLog()
        {
            if (IsPointMass)
            {
                return System.Math.Log(Point);
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                return MMath.Digamma(TrueCount) - MMath.Digamma(TotalCount);
            }
        }

        /// <summary>
        /// The expected logarithms E[log(p)] and E[log(1-p)].
        /// </summary>
        /// <param name="eLogP"></param>
        /// <param name="eLogOneMinusP"></param>
        public void GetMeanLogs(out double eLogP, out double eLogOneMinusP)
        {
            if (IsPointMass)
            {
                eLogP = System.Math.Log(Point);
                eLogOneMinusP = System.Math.Log(1 - Point);
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double d = MMath.Digamma(TotalCount);
                eLogP = MMath.Digamma(TrueCount) - d;
                eLogOneMinusP = MMath.Digamma(FalseCount) - d;
            }
        }

        /// <summary>
        /// The variance var(p).
        /// </summary>
        /// <returns></returns>
        public double GetVariance()
        {
            double mean, variance;
            GetMeanAndVariance(out mean, out variance);
            return variance;
        }

        /// <summary>
        /// Gets the mean and variance for this instance
        /// </summary>
        /// <param name="mean">Mean value - output</param>
        /// <param name="variance">Variance - output</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (IsPointMass)
            {
                mean = Point;
                variance = 0.0;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                mean = TrueCount/TotalCount;
                variance = mean*(1 - mean)/(TotalCount + 1);
            }
        }

        /// <summary>
        /// Sets the mean and variance for this instance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public void SetMeanAndVariance(double mean, double variance)
        {
            if (variance == 0.0)
            {
                Point = mean;
            }
            else if (mean < 0.0 || mean > 1.0)
            {
                throw new ArgumentException("Supplied mean is outside [0,1]", nameof(mean));
            }
            else if (variance < 0)
            {
                throw new ArgumentException("Supplied variance is negative", nameof(variance));
            }
            else
            {
                double totalCount = mean*(1 - mean)/variance - 1;
                TrueCount = totalCount*mean;
                FalseCount = totalCount*(1 - mean);
            }
        }



        /// <summary>
        /// Clones this Beta. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Beta type</returns>
        public object Clone()
        {
            return new Beta(this);
        }

        /// <summary>
        /// Gets/sets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get { return TrueCount; }
            set
            {
                FalseCount = Double.PositiveInfinity;
                TrueCount = value;
                if (TrueCount < 0.0 || TrueCount > 1.0) throw new ArgumentException("Supplied value is outside [0,1]", nameof(value));
            }
        }

        /// <summary>
        /// Whether the instance is a point mass beta distribution
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return Double.IsPositiveInfinity(FalseCount); }
        }

        /// <summary>
        /// The maximum 'difference' between the parameters of this instance and of that instance.
        /// </summary>
        /// <param name="thatd">That distribution</param>
        /// <returns>The resulting maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is Beta)) return Double.PositiveInfinity;
            Beta that = (Beta) thatd;
            // test for equality to catch infinities.
            return System.Math.Max(MMath.AbsDiff(that.TrueCount, TrueCount),
                            MMath.AbsDiff(that.FalseCount, FalseCount));
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="that">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object that)
        {
            return (MaxDiff(that) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(FalseCount.GetHashCode(), TrueCount);
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(Beta a, Beta b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(Beta a, Beta b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Sets this instance to be uniform
        /// </summary>
        public void SetToUniform()
        {
            TrueCount = 1;
            FalseCount = 1;
        }

        /// <summary>
        /// Whether the distribution is uniform
        /// </summary>
        /// <returns>True if uniform</returns>
        public bool IsUniform()
        {
            return (TrueCount == 1) && (FalseCount == 1);
        }

        /// <summary>
        /// Evaluates the logarithm of the density function
        /// </summary>
        /// <param name="value">Domain value</param>
        /// <returns>Log of the probability density for the given domain value</returns>
        public double GetLogProb(double value)
        {
            if (IsPointMass)
            {
                return (value == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else
            {
                double p = 0;
                // avoid 0*Inf in case value == 0 or 1
                if (TrueCount != 1) p += (TrueCount - 1)* System.Math.Log(value);
                if (FalseCount != 1) p += (FalseCount - 1)* System.Math.Log(1 - value);
                p -= BetaLn(TrueCount, FalseCount);
                return p;
            }
        }

        /// <summary>
        /// Gets the log normalizer for the distribution
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            return BetaLn(TrueCount, FalseCount);
        }

        /// <summary>
        /// Computes the log Beta function: <c>GammaLn(trueCount)+GammaLn(falseCount)-GammaLn(trueCount+falseCount)</c>.
        /// </summary>
        /// <param name="trueCount">Any real number.</param>
        /// <param name="falseCount">Any real number.</param>
        /// <returns><c>GammaLn(trueCount)+GammaLn(falseCount)-GammaLn(trueCount+falseCount)</c></returns>
        /// <remarks>
        /// If trueCount &lt;= 0 or falseCount &lt;= 0, the result is defined to be 0.
        /// </remarks>
        public static double BetaLn(double trueCount, double falseCount)
        {
            if (trueCount <= 0.0 || falseCount <= 0.0) return 0.0;
            double totalCount = trueCount + falseCount;
            return MMath.GammaLn(trueCount) + MMath.GammaLn(falseCount) - MMath.GammaLn(totalCount);
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Beta that)
        {
            // code is similar to GetLogProb
            // E[log Beta(p;b1,b0)] = (b1-1)*E[log(p)] + (b0-1)*E[log(1-p)] -BetaLn(b1,b0)
            if (that.IsPointMass)
            {
                if (this.IsPointMass && (this.Point == that.Point)) return 0.0;
                else return Double.NegativeInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                // that is not a point mass.
                double elogp, elog1minusp;
                GetMeanLogs(out elogp, out elog1minusp);
                double e = (that.TrueCount - 1)*elogp + (that.FalseCount - 1)*elog1minusp;
                e -= BetaLn(that.TrueCount, that.FalseCount);
                return e;
            }
        }

        /// <summary>
        /// Sets the parameters of this instance to the parameters of that instance
        /// </summary>
        /// <param name="value">that instance</param>
        public void SetTo(Beta value)
        {
            TrueCount = value.TrueCount;
            FalseCount = value.FalseCount;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two Betas.
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <remarks>
        /// The result may not be proper, i.e. its parameters may be negative.
        /// For example, if you multiply Beta(0.1,0.1) by itself you get Beta(-0.8, -0.8).
        /// No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(Beta a, Beta b)
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
                TrueCount = a.TrueCount + b.TrueCount - 1;
                FalseCount = a.FalseCount + b.FalseCount - 1;
            }
        }

        /// <summary>
        /// Static product operator. Create a Beta distribution which is the product of
        /// two Beta distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting Beta distribution</returns>
        public static Beta operator *(Beta a, Beta b)
        {
            Beta result = new Beta();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two Betas.
        /// </summary>
        /// <param name="numerator">The numerator distribution.  Can be the same object as this.</param>
        /// <param name="denominator">The denominator distribution.  Can be the same object as this.</param>
        /// <param name="forceProper">If true, the counts of the result are made >= 1, under the constraint that denominator*result has the same mean as numerator.</param>
        public void SetToRatio(Beta numerator, Beta denominator, bool forceProper = false)
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
            else if (forceProper && (numerator.TrueCount < denominator.TrueCount || numerator.FalseCount < denominator.FalseCount))
            {
                double s = 1;
                double ratio = denominator.TrueCount/numerator.TrueCount;
                if (ratio > s) s = ratio;
                double ratio2 = denominator.FalseCount/numerator.FalseCount;
                if (ratio2 > s) s = ratio2;
                TrueCount = s*numerator.TrueCount - denominator.TrueCount + 1;
                FalseCount = s*numerator.FalseCount - denominator.FalseCount + 1;
            }
            else
            {
                TrueCount = numerator.TrueCount - denominator.TrueCount + 1;
                FalseCount = numerator.FalseCount - denominator.FalseCount + 1;
            }
        }

        /// <summary>
        /// Static ratio operator. Create a Beta distribution which is the ratio of 
        /// two Beta distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <returns>The resulting Beta distribution</returns>
        public static Beta operator /(Beta numerator, Beta denominator)
        {
            Beta result = new Beta();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the a Beta raised to some power.
        /// </summary>
        /// <param name="dist">The distribution</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Beta dist, double exponent)
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
                TrueCount = exponent*(dist.TrueCount - 1) + 1;
                FalseCount = exponent*(dist.FalseCount - 1) + 1;
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Beta operator ^(Beta dist, double exponent)
        {
            Beta result = new Beta();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Property to allow an improper sum
        /// </summary>
        public static bool AllowImproperSum;

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, Beta dist1, double weight2, Beta dist2)
        {
            if (AllowImproperSum)
            {
                SetTo(Gaussian.WeightedSum<Beta>(this, weight1, dist1, weight2, dist2));
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
                double minTrue, minFalse;
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
                        minTrue = dist2.TrueCount;
                        minFalse = dist2.FalseCount;
                    }
                }
                else if (dist2.IsPointMass)
                {
                    minTrue = dist1.TrueCount;
                    minFalse = dist1.FalseCount;
                }
                else
                {
                    minTrue = System.Math.Min(dist1.TrueCount, dist2.TrueCount);
                    minFalse = System.Math.Min(dist1.FalseCount, dist2.FalseCount);
                }
                // algorithm: we choose the result to have the same mean and variance as the mixture
                // provided that all PseudoCounts are greater than the smallest PseudoCount in the mixture.
                // The result has the form (s*m,s*(1-m))  where the mean m is fixed and s satisfies
                // s*m >= min_i dist[i].TrueCount    i.e.   s >= minTrue/m
                // s*(1-m) >= min_i dist[i].FalseCount    i.e.   s >= minFalse/(1-m)
                // if weight2 < 0 then we want dist1[k] >= min(dist2[k],s*m[k])  i.e. s*m[k] <= dist1[k]  when dist1[k] < dist2[k]
                if (minTrue == 0 && minFalse == 0)
                {
                    TrueCount = 0;
                    FalseCount = 0;
                }
                else
                {
                    Beta momentMatch = Gaussian.WeightedSum<Beta>(this, weight1, dist1, weight2, dist2);
                    double mean = momentMatch.GetMean();
                    // minTrue > 0 and minFalse > 0 otherwise GetMean would have thrown an exception.
                    Assert.IsTrue(minTrue > 0);
                    Assert.IsTrue(minFalse > 0);
                    double boundTrue = minTrue/mean;
                    double boundFalse = minFalse/(1 - mean);
                    double bound;
                    bool boundViolated;
                    if (weight1 > 0)
                    {
                        if (weight2 > 0)
                        {
                            bound = System.Math.Max(boundTrue, boundFalse);
                            boundViolated = (momentMatch.TotalCount < bound);
                        }
                        else
                        {
                            bound = Double.PositiveInfinity;
                            if (dist1.TrueCount < dist2.TrueCount)
                            {
                                bound = boundTrue;
                            }
                            if (dist1.FalseCount < dist2.FalseCount)
                            {
                                bound = System.Math.Min(bound, boundFalse);
                            }
                            boundViolated = (momentMatch.TotalCount > bound);
                        }
                    }
                    else
                    {
                        // weight1 < 0
                        bound = Double.PositiveInfinity;
                        if (dist2.TrueCount < dist1.TrueCount)
                        {
                            bound = boundTrue;
                        }
                        if (dist2.FalseCount < dist1.FalseCount)
                        {
                            bound = System.Math.Min(bound, boundFalse);
                        }
                        boundViolated = (momentMatch.TotalCount > bound);
                    }
                    if (boundViolated)
                    {
                        TrueCount = bound*mean;
                        FalseCount = bound*(1 - mean);
                    }
                    else
                    {
                        SetTo(momentMatch);
                    }
                }
            }
        }

        /// <summary>
        /// The log of the integral of the product of this Beta and that Beta
        /// </summary>
        /// <param name="that">That beta</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(Beta that)
        {
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
                // int_p Beta(p;a1,a0) Beta(p;b1,b0) d_p = int_p Beta(p;a1+b1-1,a0+b0-1)*z
                // where z = Beta(a1+b1-1,a0+b0-1) / (Beta(a1,a0) Beta(b1,b0))
                double newTrueCount = TrueCount + that.TrueCount - 1;
                double newFalseCount = FalseCount + that.FalseCount - 1;
                //if (newTrueCount <= 0 || newFalseCount <= 0) throw new ArgumentException("The product is improper");
                return BetaLn(newTrueCount, newFalseCount)
                       - BetaLn(TrueCount, FalseCount) - BetaLn(that.TrueCount, that.FalseCount);
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Beta that, double power)
        {
            if (IsPointMass)
            {
                return power*that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this*(that ^ power);
                return product.GetLogNormalizer() - this.GetLogNormalizer() - power*that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Sample from this Beta distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample()
        {
            if (IsPointMass)
            {
                return Point;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                return Rand.Beta(TrueCount, FalseCount);
            }
        }
#pragma warning disable 1591
        [Stochastic]
        public double Sample(double result)
        {
            return Sample();
        }
#pragma warning restore 1591

        /// <summary>
        /// Static method to sample from a Beta distribution with the specified true and false counts
        /// </summary>
        /// <param name="trueCount">True count</param>
        /// <param name="falseCount">False count</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(double trueCount, double falseCount)
        {
            return Rand.Beta(trueCount, falseCount);
        }

        /// <summary>
        /// Static method to sample from a Beta distribution with the specified mean and variance 
        /// </summary>
        /// <param name="mean">The mean of the Beta distribution to sample from</param>
        /// <param name="variance">The variance of the Beta distribution to sample from</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "variance")]
        public static double SampleFromMeanAndVariance(double mean, double variance)
        {
            return Beta.FromMeanAndVariance(mean, variance).Sample();
        }

#if class
    /// <summary>
    /// Create a Beta(0,0) distribution.
    /// </summary>
        public Beta()
        {
        }
#endif

        /// <summary>
        /// Creates a Beta distribution with the given parameters.
        /// </summary>
        /// <param name="trueCount"></param>
        /// <param name="falseCount"></param>
        /// <remarks>
        /// The formula for the distribution is <c>p(x) = (Gamma(trueCount+falseCount)/Gamma(trueCount)/Gamma(falseCount)) x^{trueCount-1} (1-x)^(falseCount-1)</c>.
        /// </remarks>
        [Construction("TrueCount", "FalseCount")]
        public Beta(double trueCount, double falseCount)
        {
            TrueCount = trueCount;
            FalseCount = falseCount;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Beta(Beta that)
            : this(that.TrueCount, that.FalseCount)
        {
        }

        /// <summary>
        /// Creates a new Beta distribution from a specified mean and variance
        /// </summary>
        /// <param name="mean">mean</param>
        /// <param name="variance">variance</param>
        /// <returns>The new Beta distribution</returns>
        public static Beta FromMeanAndVariance(double mean, double variance)
        {
            Beta d = new Beta();
            d.SetMeanAndVariance(mean, variance);
            return d;
        }

        /// <summary>
        /// Constructs a Beta distribution with the given E[log(p)] and E[log(1-p)].
        /// </summary>
        /// <param name="eLogP">Desired expectation E[log(p)].</param>
        /// <param name="eLogOneMinusP">Desired expectation E[log(1-p)].</param>
        /// <returns>A new Beta distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Beta distribution
        /// from data given by sufficient statistics. So we want to maximize
        /// (a-1)eLogP + (b-1)eLogOneMinusP + Gamma(a+b) - Gamma(a) - Gamma(b)
        /// with respect to a, b where a=trueCount, b=falseCount.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization.
        /// </remarks>
        public static Beta FromMeanLogs(double eLogP, double eLogOneMinusP)
        {
            // This function should be named consistently with 'GetMeanLogs'.
            Dirichlet d = Dirichlet.Uniform(2);
            Vector pseudoCount = Vector.Zero(2);
            Vector meanLog = Vector.FromArray(new double[] {eLogP, eLogOneMinusP});
            pseudoCount.SetAllElementsTo(1);
            Dirichlet.EstimateNewton(pseudoCount, meanLog);
            return new Beta(pseudoCount[0], pseudoCount[1]);
        }

        /// <summary>
        /// Construct a Beta distribution whose pdf has the given derivatives at a point.
        /// </summary>
        /// <param name="x">A real number in [0,1]</param>
        /// <param name="dLogP">Desired derivative of log-density at x</param>
        /// <param name="ddLogP">Desired second derivative of log-density at x</param>
        /// <param name="forceProper">If true and both derivatives cannot be matched by a distribution with counts at least 1, match only the first.</param>
        /// <returns></returns>
        public static Beta FromDerivatives(double x, double dLogP, double ddLogP, bool forceProper)
        {
            // dlogf/dp = (trueCount-1)/p - (falseCount-1)/(1-p)
            // d^2 logf/dp^2 = -(trueCount-1)/p^2 - (falseCount-1)/(1-p)^2
            // in matrix form:
            // [1    -1      ] [(trueCount-1)/p (falseCount-1)/(1-p)]
            // [-1/p -1/(1-p)]
            // inv = 
            // [p      -p(1-p)]
            // [-(1-p) -p(1 - p)]
            double oneMinusX = 1 - x;
            double xOneMinusXddLogP = x * oneMinusX * ddLogP;
            double trueCount = 1 + x * (x * dLogP - xOneMinusXddLogP);
            double falseCount = 1 + oneMinusX * (-oneMinusX * dLogP - xOneMinusXddLogP);
            if (forceProper)
            {
                if (falseCount < 1)
                {
                    trueCount = 1 + dLogP * x;
                    falseCount = 1;
                }
                if (trueCount < 1)
                {
                    trueCount = 1;
                    falseCount = 1 - dLogP * oneMinusX;
                }
            }
            return new Beta(trueCount, falseCount);
        }

        /// <summary>
        /// Get the derivatives of the log-pdf at a point.
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="x"></param>
        /// <param name="dlogp">On exit, the first derivative.</param>
        /// <param name="ddlogp">On exit, the second derivative.</param>
        public static void GetDerivatives(Beta dist, double x, out double dlogp, out double ddlogp)
        {
            dist.GetDerivatives(x, out dlogp, out ddlogp);
        }

        /// <summary>
        /// Get the derivatives of the log-pdf at a point.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="dlogp">On exit, the first derivative.</param>
        /// <param name="ddlogp">On exit, the second derivative.</param>
        public void GetDerivatives(double x, out double dlogp, out double ddlogp)
        {
            dlogp = 0;
            ddlogp = 0;
            double trueCountMinus1 = this.TrueCount - 1;
            if (trueCountMinus1 != 0) // avoid 0/0
            {
                dlogp += trueCountMinus1 / x;
                ddlogp -= trueCountMinus1 / (x * x);
            }
            double falseCountMinus1 = this.FalseCount - 1;
            if (falseCountMinus1 != 0)
            {
                dlogp -= falseCountMinus1 / (1 - x);
                ddlogp -= falseCountMinus1 / ((1 - x) * (1 - x));
            }
        }

        /// <summary>
        /// Instantiates a uniform Beta distribution
        /// </summary>
        /// <returns>A new uniform Beta distribution</returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static Beta Uniform()
        {
            Beta result = new Beta();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Instantiates a point-mass Beta distribution
        /// </summary>
        /// <param name="mean">The domain value where the point mass occurs</param>
        /// <returns>A new point-mass Beta distribution at the specified location</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static Beta PointMass(double mean)
        {
            Beta d = new Beta();
            d.Point = mean;
            return d;
        }

        /// <summary>
        /// Override of the ToString method
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsPointMass)
            {
                return "Beta.PointMass(" + Point + ")";
            }
            else
            {
                string s = "Beta(" + TrueCount.ToString("g4") + "," + FalseCount.ToString("g4") + ")";
                if (IsProper())
                {
                    s += "[mean=" + GetMean().ToString("g4") + "]";
                }
                return s;
            }
        }
    }
}