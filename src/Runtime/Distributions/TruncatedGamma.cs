// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Factors.Attributes;
    using Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Utilities;

    /// <summary>
    /// A distribution over real numbers between an upper and lower bound.  If LowerBound=0 and UpperBound=Inf, it reduces to an ordinary Gamma distribution.
    /// </summary>
    /// <remarks>
    /// The distribution is parameterized by a Gamma and two real numbers (LowerBound, UpperBound).
    /// Between the bounds, the density is proportional to the Gamma.  Outside of the bounds, the density is zero.
    /// </remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Preview)]
    public struct TruncatedGamma : IDistribution<double>,
                                      SettableTo<TruncatedGamma>, Diffable,
                                      SettableToProduct<TruncatedGamma>, SettableToUniform,
                                      SettableToRatio<TruncatedGamma>, SettableToPower<TruncatedGamma>,
                                      Sampleable<double>, SettableToWeightedSum<TruncatedGamma>,
                                      CanGetMean<double>, CanGetVariance<double>, CanGetMeanAndVarianceOut<double, double>,
                                      CanGetLogNormalizer, CanGetLogAverageOf<TruncatedGamma>, CanGetLogAverageOfPower<TruncatedGamma>,
                                      CanGetAverageLog<TruncatedGamma>, CanGetMode<double>,
                                      CanGetProbLessThan<double>, CanGetQuantile<double>
    {
        /// <summary>
        /// Untruncated Gamma
        /// </summary>
        [DataMember]
        public Gamma Gamma;

        /// <summary>
        /// Lower bound
        /// </summary>
        [DataMember]
        public double LowerBound;

        /// <summary>
        /// Upper bound
        /// </summary>
        [DataMember]
        public double UpperBound;

        /// <summary>
        /// Create a new TruncatedGamma distribution equal to that
        /// </summary>
        /// <param name="that"></param>
        public TruncatedGamma(TruncatedGamma that)
            : this()
        {
            SetTo(that);
        }

        /// <summary>
        /// Create a truncated Gamma equivalent to a Gamma, i.e. with no truncation.
        /// </summary>
        /// <param name="that"></param>
        public TruncatedGamma(Gamma that)
        {
            this.Gamma = that;
            LowerBound = 0;
            UpperBound = Double.PositiveInfinity;
        }

        /// <summary>
        /// Create a truncated Gamma equivalent to a Gamma, i.e. with no truncation.
        /// </summary>
        /// <param name="Gamma"></param>
        public static TruncatedGamma FromGamma(Gamma Gamma)
        {
            return new TruncatedGamma(Gamma);
        }

        /// <summary>
        /// Create a truncated Gamma from a Gamma and bounds
        /// </summary>
        /// <param name="Gamma"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        [Construction("Gamma", "LowerBound", "UpperBound")]
        public TruncatedGamma(Gamma Gamma, double lowerBound, double upperBound)
        {
            LowerBound = lowerBound;
            UpperBound = upperBound;
            if (lowerBound == upperBound) this.Gamma = Gamma.PointMass(lowerBound);
            else this.Gamma = Gamma;
        }

        /// <summary>
        /// Create a truncated Gamma from untruncated (shape, scale) and bounds
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="scale"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        public TruncatedGamma(double shape, double scale, double lowerBound, double upperBound)
        {
            LowerBound = lowerBound;
            UpperBound = upperBound;
            if (lowerBound == upperBound) this.Gamma = Gamma.PointMass(lowerBound);
            else this.Gamma = Gamma.FromShapeAndScale(shape, scale);
        }

        /// <summary>
        /// Construct a uniform truncated Gamma. This is mathematically equivalent to
        /// a uniform Gamma. 
        /// </summary>
        /// <returns></returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static TruncatedGamma Uniform()
        {
            var result = new TruncatedGamma();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Create a point mass distribution.
        /// </summary>
        /// <param name="point">The location of the point mass</param>
        /// <returns>A new TruncatedGamma with all probability concentrated on the given point.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static TruncatedGamma PointMass(double point)
        {
            var result = new TruncatedGamma();
            result.Point = point;
            result.LowerBound = 0;
            result.UpperBound = double.PositiveInfinity;
            return result;
        }

        /// <summary>
        /// Get the Gamma with the same moments as this truncated Gamma. 
        /// </summary>
        /// <returns></returns>
        public Gamma ToGamma()
        {
            double m, v;
            GetMeanAndVariance(out m, out v);
            return Gamma.FromMeanAndVariance(m, v);
        }

        /// <summary>
        /// The maximum difference between the parameters of this distribution and that
        /// </summary>
        /// <param name="thatd"></param>
        /// <returns></returns>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is TruncatedGamma))
                return Double.PositiveInfinity;
            TruncatedGamma that = (TruncatedGamma)thatd;
            double diff1 = Gamma.MaxDiff(that.Gamma);
            double diff3 = MMath.AbsDiff(LowerBound, that.LowerBound);
            double diff4 = MMath.AbsDiff(UpperBound, that.UpperBound);
            return Math.Max(diff1, Math.Max(diff3, diff4));
        }

        /// <summary>
        /// True if this distribution has the same parameters as that
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns></returns>
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            return (MaxDiff(thatd) == 0.0);
        }

        /// <summary>
        /// A hash of the distribution parameter values
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = Gamma.GetHashCode();
            hash = Hash.Combine(hash, LowerBound);
            hash = Hash.Combine(hash, UpperBound);
            return hash;
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(TruncatedGamma a, TruncatedGamma b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(TruncatedGamma a, TruncatedGamma b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Make a deep copy of this distribution. 
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new TruncatedGamma(this);
        }

        /// <summary>
        /// Set this distribution to a point mass, or get its location
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get { return Gamma.Point; }
            set { Gamma.Point = value; }
        }

        /// <summary>
        /// True if the distribution is a point mass. 
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return Gamma.IsPointMass; }
        }

        /// <summary>
        /// Get the log probability density at value. 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetLogProb(double value)
        {
            if (value < LowerBound || value > UpperBound)
            {
                return double.NegativeInfinity;
            }
            else
            {
                double logZ = GetLogNormalizer();
                if (logZ < double.MinValue)
                {
                    return (value == GetMode()) ? 0.0 : double.NegativeInfinity;
                }
                double normalizer = this.Gamma.GetLogNormalizer();
                if (normalizer > double.MaxValue)
                {
                    return Gamma.GetLogProb(value, this.Gamma.Shape, this.Gamma.Rate, normalized: false) - logZ;
                }
                else
                {
                    return this.Gamma.GetLogProb(value) + (normalizer - logZ);
                }
            }
        }

        /// <summary>
        /// Gets the normalizer for the truncated Gamma density function
        /// </summary>
        /// <returns></returns>
        public double GetNormalizer()
        {
            if (IsProper() && !IsPointMass)
            {
                // Equivalent but less accurate:
                //return this.Gamma.GetProbLessThan(UpperBound) - this.Gamma.GetProbLessThan(LowerBound);
                return MMath.GammaProbBetween(this.Gamma.Shape, this.Gamma.Rate, LowerBound, UpperBound);
            }
            else
            {
                return 1.0;
            }
        }

        /// <summary>
        /// Gets the log of the normalizer for the truncated Gamma density function
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            // The output of this function can be checked against log(gammainc(shape, lowerBound*rate, UpperBound*rate, regularized=True)) in mpmath
            if (IsProper() && !IsPointMass)
            {
                double rl = (double)(this.Gamma.Rate * LowerBound);
                if (this.Gamma.Shape < 1 && rl > 0)
                {
                    // When Shape < 1, Gamma(Shape) > 1 so use the unregularized version to avoid underflow.
                    return Math.Log(MMath.GammaProbBetween(this.Gamma.Shape, this.Gamma.Rate, LowerBound, UpperBound, false)) - MMath.GammaLn(this.Gamma.Shape);
                }
                else
                {
                    return Math.Log(MMath.GammaProbBetween(this.Gamma.Shape, this.Gamma.Rate, LowerBound, UpperBound));
                }
            }
            else
            {
                return 0.0;
            }
        }

        /// <summary>
        /// Returns true if this distribution is proper
        /// </summary>
        /// <returns></returns>
        public bool IsProper()
        {
            return Gamma.IsProper();
        }

        /// <summary>
        /// Set this distribution equal to value. 
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(TruncatedGamma value)
        {
            Gamma.SetTo(value.Gamma);
            LowerBound = value.LowerBound;
            UpperBound = value.UpperBound;
        }

        /// <summary>
        /// Set this distribution equal to the product of a and b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public void SetToProduct(TruncatedGamma a, TruncatedGamma b)
        {
            LowerBound = Math.Max(a.LowerBound, b.LowerBound);
            UpperBound = Math.Min(a.UpperBound, b.UpperBound);
            if (LowerBound > UpperBound)
                throw new AllZeroException();
            if (LowerBound == UpperBound)
                Gamma.Point = LowerBound;
            else
                Gamma.SetToProduct(a.Gamma, b.Gamma);
        }

        /// <summary>
        /// Operator overload for product. 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static TruncatedGamma operator *(TruncatedGamma a, TruncatedGamma b)
        {
            TruncatedGamma result = new TruncatedGamma();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Set the distribution to uniform with infinite bounds
        /// </summary>
        public void SetToUniform()
        {
            Gamma.SetToUniform();
            LowerBound = 0;
            UpperBound = double.PositiveInfinity;
        }

        /// <summary>
        /// Asks whether this instance is uniform. If the upper and lower bounds are finite the distribution
        /// is not uniform. 
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return Gamma.IsUniform() && (LowerBound == 0) && double.IsPositiveInfinity(UpperBound);
        }

        /// <summary>
        /// Set this equal to numerator/denominator
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <param name="forceProper"></param>
        public void SetToRatio(TruncatedGamma numerator, TruncatedGamma denominator, bool forceProper = false)
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
                    if (denominator.LowerBound < numerator.Point && numerator.Point < denominator.UpperBound)
                        Point = numerator.Point;
                    else
                        throw new DivideByZeroException();
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else
            {
                if (numerator.LowerBound >= denominator.LowerBound && numerator.UpperBound <= denominator.UpperBound)
                {
                    LowerBound = numerator.LowerBound;
                    UpperBound = numerator.UpperBound;
                    Gamma.SetToRatio(numerator.Gamma, denominator.Gamma, forceProper);
                }
                else
                    throw new DivideByZeroException();
            }
        }

        /// <summary>
        /// Operator overload for division
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <returns></returns>
        public static TruncatedGamma operator /(TruncatedGamma numerator, TruncatedGamma denominator)
        {
            var result = new TruncatedGamma();
            result.SetToRatio(numerator, denominator);
            return result;
        }


        /// <summary>
        /// Set this equal to (dist)^exponent
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="exponent"></param>
        public void SetToPower(TruncatedGamma dist, double exponent)
        {
            if (exponent == 0)
            {
                SetToUniform();
            }
            else
            {
                if (exponent < 0 && !((dist.LowerBound == 0) && double.IsPositiveInfinity(dist.UpperBound)))
                    throw new DivideByZeroException("The exponent is negative and the bounds are finite");
                LowerBound = dist.LowerBound;
                UpperBound = dist.UpperBound;
                Gamma.SetToPower(dist.Gamma, exponent);
            }
        }

        /// <summary>
        /// Operator overload for exponentation
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="exponent"></param>
        /// <returns></returns>
        public static TruncatedGamma operator ^(TruncatedGamma dist, double exponent)
        {
            var result = new TruncatedGamma();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Sample from a TruncatedGamma distribution with the specified parameters
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(Gamma gamma, double lowerBound, double upperBound)
        {
            if (gamma.IsUniform()) return Rand.UniformBetween(lowerBound, upperBound);
            bool useQuantile = (gamma.Shape == 1);
            if (useQuantile)
            {
                return GetQuantile(gamma, lowerBound, upperBound, Rand.UniformBetween(0, 1));
            }
            else
            {
                double sample;
                do
                {
                    sample = gamma.Sample();
                } while (sample < lowerBound || sample > upperBound);
                return sample;
            }
        }

        /// <inheritdoc/>
        public double GetProbLessThan(double x)
        {
            if (this.IsPointMass)
            {
                return (this.Point < x) ? 1.0 : 0.0;
            }
            else
            {
                double totalProbability = Gamma.GetProbBetween(LowerBound, UpperBound);
                return Gamma.GetProbBetween(LowerBound, x) / totalProbability;
            }
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            double totalProbability = Gamma.GetProbBetween(LowerBound, UpperBound);
            return Gamma.GetProbBetween(Math.Max(LowerBound, lowerBound), Math.Min(UpperBound, upperBound)) / totalProbability;
        }

        public static double GetQuantile(Gamma gamma, double lowerBound, double upperBound, double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1");
            bool useBinarySearch = false;
            if (useBinarySearch)
            {
                // Binary search
                probability *= gamma.GetProbBetween(lowerBound, upperBound);
                double lowerX = lowerBound;
                double upperX = upperBound;
                while (lowerX < upperX)
                {
                    double average = MMath.Average(lowerX, upperX);
                    double p = gamma.GetProbBetween(lowerBound, average);
                    if (p == probability)
                    {
                        return average;
                    }
                    else if (p < probability)
                    {
                        lowerX = MMath.NextDouble(average);
                    }
                    else
                    {
                        upperX = MMath.PreviousDouble(average);
                    }
                }
                return lowerX;
            }
            else
            {
                double lowerProbability = gamma.GetProbLessThan(lowerBound);
                if (MMath.AreEqual(lowerProbability, 1)) throw new NotImplementedException();
                double totalProbability = gamma.GetProbBetween(lowerBound, upperBound);
                if (totalProbability + lowerProbability > 1.01) throw new Exception();
                double quantile = gamma.GetQuantile(probability * totalProbability + lowerProbability);
                return Math.Min(upperBound, Math.Max(lowerBound, quantile));
            }
        }

        /// <inheritdoc/>
        public double GetQuantile(double probability)
        {
            return GetQuantile(Gamma, LowerBound, UpperBound, probability);
        }

        /// <summary>
        /// Sample from the distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample()
        {
            if (IsPointMass)
            {
                return Point;
            }
            else
            {
                return Sample(Gamma, LowerBound, UpperBound);
            }
        }

        /// <summary>
        /// Sample from the distribution
        /// </summary>
        /// <param name="result">Ignored</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample(double result)
        {
            return Sample();
        }

        /// <summary>
        /// Get the mode (highest density point) of this distribution
        /// </summary>
        /// <returns></returns>
        public double GetMode()
        {
            return Math.Min(Math.Max(this.Gamma.GetMode(), this.LowerBound), this.UpperBound);
        }

        /// <summary>
        /// Returns the mean (first moment) of the distribution
        /// </summary>
        /// <returns></returns>
        public double GetMean()
        {
            double mean, variance;
            GetMeanAndVariance(out mean, out variance);
            return mean;
        }

        /// <summary>
        /// Get the variance of this distribution
        /// </summary>
        /// <returns></returns>
        public double GetVariance()
        {
            double mean, variance;
            GetMeanAndVariance(out mean, out variance);
            return variance;
        }

        /// <summary>
        /// Computes <c>GammaUpper(s,x)/(x^(s-1)*exp(-x)) - 1</c> to high accuracy
        /// </summary>
        /// <param name="s"></param>
        /// <param name="x">A real number gt;= 45 and gt; <paramref name="s"/>/0.99</param>
        /// <param name="regularized">If true, result is divided by <c>MMath.Gamma(s)</c></param>
        /// <returns></returns>
        public static double GammaUpperRatio(double s, double x, bool regularized = true)
        {
            if (s >= x * 0.99) throw new ArgumentOutOfRangeException(nameof(s), s, "s >= x*0.99");
            if (x < 45) throw new ArgumentOutOfRangeException(nameof(x), x, "x < 45");
            double term = (s - 1) / x;
            double sum = term;
            for (int i = 2; i < 1000; i++)
            {
                term *= (s - i) / x;
                double oldSum = sum;
                sum += term;
                if (MMath.AreEqual(sum, oldSum)) return regularized ? sum / MMath.Gamma(s) : sum;
            }
            throw new Exception($"GammaUpperRatio not converging for s={s:g17}, x={x:g17}, regularized={regularized}");
        }

        /// <summary>
        /// Gets E[log(x)] after truncation.
        /// </summary>
        public double GetMeanLog()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Get the mean and variance after truncation.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (this.Gamma.IsPointMass)
            {
                mean = this.Gamma.Point;
                variance = 0.0;
            }
            else if (!IsProper())
                throw new ImproperDistributionException(this);
            else
            {
                // Apply the recurrence GammaUpper(s+1,x,false) = s*GammaUpper(s,x,false) + x^s*exp(-x)
                // Z = GammaUpper(s,r*l,false) - GammaUpper(s,r*u,false)
                // E[x] = (GammaUpper(s+1,r*l,false) - GammaUpper(s+1,r*u,false))/r/Z
                //      = (s + ((r*l)^s*exp(-r*l) - (r*u)^s*exp(-r*u))/Z)/r
                double rl = this.Gamma.Rate * LowerBound;
                double ru = this.Gamma.Rate * UpperBound;
                double m = this.Gamma.Shape / this.Gamma.Rate;
                double offset, offset2;
                if (ru > double.MaxValue)
                {
                    double logZ = GetLogNormalizer();
                    if (logZ < double.MinValue)
                    {
                        mean = GetMode();
                        variance = 0.0;
                        return;
                    }
                    offset = Math.Exp(MMath.GammaUpperLogScale(this.Gamma.Shape, rl) - logZ);
                    offset2 = (rl - this.Gamma.Shape) / this.Gamma.Rate * offset;
                }
                else
                {
                    // This fails when GammaUpperScale underflows to 0
                    double Z = GetNormalizer();
                    if (Z == 0)
                    {
                        mean = GetMode();
                        variance = 0.0;
                        return;
                    }
                    double gammaUpperScaleLower = MMath.GammaUpperScale(this.Gamma.Shape, rl);
                    double gammaUpperScaleUpper = MMath.GammaUpperScale(this.Gamma.Shape, ru);
                    offset = (gammaUpperScaleLower - gammaUpperScaleUpper) / Z;
                    offset2 = ((rl - this.Gamma.Shape) / this.Gamma.Rate * gammaUpperScaleLower - (ru - this.Gamma.Shape) / this.Gamma.Rate * gammaUpperScaleUpper) / Z;
                }
                if (rl == this.Gamma.Shape) mean = LowerBound + offset / this.Gamma.Rate;
                else
                {
                    mean = (this.Gamma.Shape + offset) / this.Gamma.Rate;
                    if (mean < LowerBound) mean = MMath.NextDouble(mean);
                    if (mean < LowerBound) mean = MMath.NextDouble(mean);
                }
                if (mean > double.MaxValue) variance = mean;
                else if (offset2 > double.MaxValue) variance = double.PositiveInfinity;
                else variance = (m + offset2 + (1 - offset) * offset / this.Gamma.Rate) / this.Gamma.Rate;
            }
        }

        /// <summary>
        /// Computes E[x^power]
        /// </summary>
        /// <returns></returns>
        public double GetMeanPower(double power)
        {
            if (power == 0.0) return 1.0;
            else if (power == 1.0) return GetMean();
            else if (IsPointMass) return Math.Pow(Point, power);
            //else if (Rate == 0.0) return (power > 0) ? Double.PositiveInfinity : 0.0;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else if (this.Gamma.Shape <= -power && LowerBound == 0)
            {
                throw new ArgumentException("Cannot compute E[x^" + power + "] for " + this + " (shape <= " + (-power) + ")");
            }
            else if (power != 1)
            {
                double lowerBoundPower = Math.Pow(LowerBound, power);
                double upperBoundPower = Math.Pow(UpperBound, power);
                if (power < 0)
                {
                    (lowerBoundPower, upperBoundPower) = (upperBoundPower, lowerBoundPower);
                }
                // Large powers lead to overflow
                power = Math.Min(Math.Max(power, -1e300), 1e300);
                bool regularized = this.Gamma.Shape >= 1;
                double logZ = Math.Log(MMath.GammaProbBetween(this.Gamma.Shape, this.Gamma.Rate, LowerBound, UpperBound, regularized));
                if (logZ < double.MinValue)
                {
                    return Math.Pow(GetMode(), power);
                }
                double shapePlusPower = this.Gamma.Shape + power;
                if (MMath.AreEqual(this.Gamma.Shape, shapePlusPower) && this.Gamma.Shape > 1e15)
                {
                    // In this case, logProbBetween == logZ and digamma(shape) == log(shape) and MMath.RisingFactorialLn(this.Gamma.Shape, power) = power * log(this.Gamma.Shape)
                    double mean = this.Gamma.Shape / this.Gamma.Rate;
                    if (mean > 0)
                    {
                        // This version is the most accurate when applicable.
                        return Math.Pow(mean, power);
                    }
                    else
                    {
                        return Math.Exp(power * (Math.Log(this.Gamma.Shape) - Math.Log(this.Gamma.Rate)));
                    }
                }
                double logRate = Math.Log(this.Gamma.Rate);
                bool regularized2 = shapePlusPower >= 1;
                bool includeRatePower = true;
                double logProbBetween = Math.Log(MMath.GammaProbBetween(shapePlusPower, this.Gamma.Rate, LowerBound, UpperBound, regularized2));
                if (!regularized2 && logProbBetween > double.MaxValue && double.IsPositiveInfinity(UpperBound) && shapePlusPower < 0)
                {
                    logProbBetween = MMath.LogGammaUpper(shapePlusPower, logRate * power / shapePlusPower, Math.Log(LowerBound) + logRate * this.Gamma.Shape / shapePlusPower);
                    includeRatePower = false;
                }
                else if (!regularized2 && logProbBetween > double.MaxValue && shapePlusPower > 0)
                {
                    regularized2 = true;
                    logProbBetween = Math.Log(MMath.GammaProbBetween(shapePlusPower, this.Gamma.Rate, LowerBound, UpperBound, regularized2));
                }
                double correction;
                if (regularized2)
                {
                    // This formula cannot be used when shapePlusPower <= 0
                    if (regularized)
                    {
                        correction = MMath.RisingFactorialLn(this.Gamma.Shape, power);
                    }
                    else
                    {
                        correction = MMath.GammaLn(shapePlusPower);
                    }
                }
                else if (regularized)
                {
                    correction = -MMath.GammaLn(this.Gamma.Shape);
                }
                else 
                { 
                    correction = 0;
                }
                if (includeRatePower)
                {
                    correction -= power * logRate;
                }
                double result = Math.Exp(logProbBetween - logZ + correction);
                return Math.Max(lowerBoundPower, Math.Min(upperBoundPower, result));
            }
            else
            {
                // int_L^U x^p x^(a-1) b^a exp(-x*b)/Gamma(a) dx / Z
                // = int_L^U x^(a+p-1) b^(a+p) exp(-x*b)/Gamma(a+p) dx * Gamma(a+p)/Gamma(a)/b^p/Z
                double Z = GetNormalizer();
                if (Z == 0.0)
                {
                    return Math.Pow(GetMode(), power);
                }
                double shapePlusPower = this.Gamma.Shape + power;
                double Z1;
                //double gammaLnShapePlusPower = MMath.GammaLn(shapePlusPower);
                double gammaLnShape = MMath.GammaLn(this.Gamma.Shape);
                bool regularized = true; // (gammaLnShapePlusPower - gammaLnShape <= 700);
                if (regularized)
                {
                    // If shapePlusPower is large and Gamma.Rate * UpperBound is small, then this can lead to Inf * 0
                    Z1 = Math.Exp(MMath.RisingFactorialLn(this.Gamma.Shape, power)) *
                        MMath.GammaProbBetween(shapePlusPower, this.Gamma.Rate, LowerBound, UpperBound, regularized);
                }
                else
                {
                    Z1 = Math.Exp(-gammaLnShape) *
                        MMath.GammaProbBetween(shapePlusPower, this.Gamma.Rate, LowerBound, UpperBound, regularized);
                }
                return Z1 / (Math.Pow(this.Gamma.Rate, power) * Z);
            }
        }

        /// <summary>
        /// Get the logarithm of the average value of that distribution under this distribution, i.e. log(int this(x) that(x) dx)
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public double GetLogAverageOf(TruncatedGamma that)
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
                // neither this nor that is a point mass.
                TruncatedGamma product = this * that;
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(TruncatedGamma that, double power)
        {
            if (IsPointMass)
            {
                return power * that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0)
                    throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this * (that ^ power);
                return product.GetLogNormalizer() - this.GetLogNormalizer() - power * that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// A human-readable string containing the parameters of the distribution
        /// </summary>
        /// <returns>String representation of the instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsPointMass)
            {
                return "TruncatedGamma.PointMass(" + Point.ToString("g4") + ")";
            }
            else if (IsUniform())
            {
                return "TruncatedGamma.Uniform";
            }
            else
            {
                return "Truncated" + Gamma.ToString() + "[" + LowerBound.ToString("g4") + "<x<" + UpperBound.ToString("g4") + "]";
            }
        }

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, TruncatedGamma dist1, double weight2, TruncatedGamma dist2)
        {
            if (weight1 + weight2 == 0)
                SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0)
                SetTo(dist2);
            else if (weight2 == 0)
                SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.Equals(dist2))
                SetTo(dist1);
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
            else if (double.IsPositiveInfinity(weight2))
                SetTo(dist2);
            else if (dist1.LowerBound == dist2.LowerBound && dist1.UpperBound == dist2.UpperBound)
            {
                Gamma.SetToSum(weight1, dist1.Gamma, weight2, dist2.Gamma);
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Get the average logarithm of that distribution under this distribution, i.e. int this(x) log( that(x) ) dx
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public double GetAverageLog(TruncatedGamma that)
        {
            if (IsPointMass)
            {
                if (that.IsPointMass && this.Point == that.Point)
                    return 0.0;
                else
                    return that.GetLogProb(this.Point);
            }
            else if (this.LowerBound < that.LowerBound || this.UpperBound > that.UpperBound)
            {
                return double.NegativeInfinity;
            }
            else if (that.IsPointMass)
            {
                return Double.NegativeInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double m = GetMean();
                double meanLog = GetMeanLog();
                return (that.Gamma.Shape - 1) * meanLog - that.Gamma.Rate * m - that.GetLogNormalizer();
            }
        }
    }
}
