// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Factors;
    using Factors.Attributes;
    using Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Utilities;
    
    /// <summary>
    /// A distribution over real numbers between an upper and lower bound.  If both bounds are infinite, it reduces to an ordinary Gaussian distribution.
    /// </summary>
    /// <remarks>
    /// The distribution is parameterized by a Gaussian and two real numbers (LowerBound, UpperBound).
    /// Between the bounds, the density is proportional to the Gaussian.  Outside of the bounds, the density is zero.
    /// </remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Preview)]
    public struct TruncatedGaussian : IDistribution<double>,
                                      SettableTo<TruncatedGaussian>, Diffable,
                                      SettableToProduct<TruncatedGaussian>, SettableToUniform,
                                      SettableToRatio<TruncatedGaussian>, SettableToPower<TruncatedGaussian>,
                                      Sampleable<double>, SettableToWeightedSum<TruncatedGaussian>,
                                      CanGetMean<double>, CanGetVariance<double>, CanGetMeanAndVarianceOut<double, double>,
                                      CanGetLogNormalizer, CanGetLogAverageOf<TruncatedGaussian>,
                                      CanGetLogAverageOfPower<TruncatedGaussian>,
                                      CanGetAverageLog<TruncatedGaussian>,
                                      CanGetProbLessThan<double>, CanGetQuantile<double>, ITruncatableDistribution<double>
    {
        /// <summary>
        /// Untruncated Gaussian
        /// </summary>
        [DataMember]
        public Gaussian Gaussian;

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
        /// Create a new TruncatedGaussian distribution equal to that
        /// </summary>
        /// <param name="that"></param>
        public TruncatedGaussian(TruncatedGaussian that)
            : this()
        {
            SetTo(that);
        }

        /// <summary>
        /// Create a truncated Gaussian equivalent to a Gaussian, i.e. with no truncation.
        /// </summary>
        /// <param name="that"></param>
        public TruncatedGaussian(Gaussian that)
        {
            this.Gaussian = that;
            LowerBound = Double.NegativeInfinity;
            UpperBound = Double.PositiveInfinity;
        }

        /// <summary>
        /// Create a truncated Gaussian equivalent to a Gaussian, i.e. with no truncation.
        /// </summary>
        /// <param name="gaussian"></param>
        public static TruncatedGaussian FromGaussian(Gaussian gaussian)
        {
            return new TruncatedGaussian(gaussian);
        }

        /// <summary>
        /// Create a truncated Gaussian from a Gaussian and bounds
        /// </summary>
        /// <param name="gaussian"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        [Construction("Gaussian", "LowerBound", "UpperBound")]
        public TruncatedGaussian(Gaussian gaussian, double lowerBound, double upperBound)
        {
            this.Gaussian = gaussian;
            LowerBound = lowerBound;
            UpperBound = upperBound;
        }

        /// <summary>
        /// Create a truncated Gaussian from untruncated (mean, variance) and bounds
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        public TruncatedGaussian(double mean, double variance, double lowerBound, double upperBound)
        {
            this.Gaussian = Gaussian.FromMeanAndVariance(mean, variance);
            LowerBound = lowerBound;
            UpperBound = upperBound;
        }

        /// <summary>
        /// Construct a uniform truncated Gaussian. This is mathematically equivalent to
        /// a uniform Gaussian. 
        /// </summary>
        /// <returns></returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static TruncatedGaussian Uniform()
        {
            var result = new TruncatedGaussian();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Create a point mass distribution.
        /// </summary>
        /// <param name="point">The location of the point mass</param>
        /// <returns>A new TruncatedGaussian with all probability concentrated on the given point.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static TruncatedGaussian PointMass(double point)
        {
            var result = new TruncatedGaussian();
            result.Point = point;
            result.LowerBound = double.NegativeInfinity;
            result.UpperBound = double.PositiveInfinity;
            return result;
        }

        /// <summary>
        /// Get the Gaussian with the same moments as this truncated Gaussian. 
        /// </summary>
        /// <returns></returns>
        public Gaussian ToGaussian()
        {
            double m, v;
            GetMeanAndVariance(out m, out v);
            return Gaussian.FromMeanAndVariance(m, v);
        }

        /// <summary>
        /// The maximum difference between the parameters of this distribution and that
        /// </summary>
        /// <param name="thatd"></param>
        /// <returns></returns>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is TruncatedGaussian))
                return Double.PositiveInfinity;
            TruncatedGaussian that = (TruncatedGaussian)thatd;
            double diff1 = Gaussian.MaxDiff(that.Gaussian);
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
            int hash = Gaussian.GetHashCode();
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
        public static bool operator ==(TruncatedGaussian a, TruncatedGaussian b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(TruncatedGaussian a, TruncatedGaussian b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Make a deep copy of this distribution. 
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new TruncatedGaussian(this);
        }

        /// <summary>
        /// Set this distribution to a point mass, or get its location
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get { return Gaussian.Point; }
            set { Gaussian.Point = value; }
        }

        /// <summary>
        /// True if the distribution is a point mass. 
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return Gaussian.IsPointMass; }
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
            else if(IsPointMass)
            {
                return Gaussian.GetLogProb(value);
            }
            else
            {
                return (Gaussian.MeanTimesPrecision - 0.5 * value * Gaussian.Precision) * value - GetLogNormalizer();
            }
        }

        /// <summary>
        /// Gets the log of the normalizer for the Gaussian density function
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            if (IsProper())
            {
                return Gaussian.GetLogNormalizer() + Math.Log(Gaussian.GetProbBetween(LowerBound, UpperBound));
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
            return Gaussian.IsProper();
        }

        /// <summary>
        /// Set this distribution equal to value. 
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(TruncatedGaussian value)
        {
            Gaussian.SetTo(value.Gaussian);
            LowerBound = value.LowerBound;
            UpperBound = value.UpperBound;
        }

        /// <summary>
        /// Set this distribution equal to the product of a and b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public void SetToProduct(TruncatedGaussian a, TruncatedGaussian b)
        {
            LowerBound = Math.Max(a.LowerBound, b.LowerBound);
            UpperBound = Math.Min(a.UpperBound, b.UpperBound);
            if (LowerBound > UpperBound)
                throw new AllZeroException();
            Gaussian.SetToProduct(a.Gaussian, b.Gaussian);
        }

        /// <summary>
        /// Operator overload for product. 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static TruncatedGaussian operator *(TruncatedGaussian a, TruncatedGaussian b)
        {
            TruncatedGaussian result = new TruncatedGaussian();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Set the distribution to uniform with infinite bounds
        /// </summary>
        public void SetToUniform()
        {
            Gaussian.SetToUniform();
            LowerBound = double.NegativeInfinity;
            UpperBound = double.PositiveInfinity;
        }

        /// <summary>
        /// Asks whether this instance is uniform. If the upper and lower bounds are finite the distribution
        /// is not uniform. 
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return Gaussian.IsUniform() && double.IsNegativeInfinity(LowerBound) && double.IsPositiveInfinity(UpperBound);
        }

        /// <summary>
        /// Set this equal to numerator/denominator
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <param name="forceProper"></param>
        public void SetToRatio(TruncatedGaussian numerator, TruncatedGaussian denominator, bool forceProper = false)
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
                    if (denominator.LowerBound <= numerator.Point && numerator.Point <= denominator.UpperBound)
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
                    Gaussian.SetToRatio(numerator.Gaussian, denominator.Gaussian, forceProper);
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
        public static TruncatedGaussian operator /(TruncatedGaussian numerator, TruncatedGaussian denominator)
        {
            var result = new TruncatedGaussian();
            result.SetToRatio(numerator, denominator);
            return result;
        }


        /// <summary>
        /// Set this equal to (dist)^exponent
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="exponent"></param>
        public void SetToPower(TruncatedGaussian dist, double exponent)
        {
            if (exponent == 0)
            {
                SetToUniform();
            }
            else
            {
                if (exponent < 0 && !(double.IsNegativeInfinity(dist.LowerBound) && double.IsPositiveInfinity(dist.UpperBound)))
                    throw new DivideByZeroException("The exponent is negative and the bounds are finite");
                LowerBound = dist.LowerBound;
                UpperBound = dist.UpperBound;
                Gaussian.SetToPower(dist.Gaussian, exponent);
            }
        }

        /// <summary>
        /// Operator overload for exponentation
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="exponent"></param>
        /// <returns></returns>
        public static TruncatedGaussian operator ^(TruncatedGaussian dist, double exponent)
        {
            var result = new TruncatedGaussian();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Sample from a TruncatedGaussian distribution with the specified parameters
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(double mean, double precision, double lowerBound, double upperBound)
        {
            if (precision < 0)
                throw new ArgumentException("precision < 0 (" + precision + ")");
            if (precision == 0)
                return Rand.UniformBetween(lowerBound, upperBound);
            double sqrtPrec = Math.Sqrt(precision);
            double sd = 1.0 / sqrtPrec;
            // x*sd + mean > lowerBound
            // x > (lowerBound - mean)/sd
            double scaledLower = (lowerBound - mean) * sqrtPrec;
            double scaledUpper = (upperBound - mean) * sqrtPrec;
            return Rand.NormalBetween(scaledLower, scaledUpper) * sd + mean;
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
                return Sample(Gaussian.IsUniform() ? 0.0 : Gaussian.GetMean(), Gaussian.Precision, LowerBound, UpperBound);
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
        /// Returns the mean (first moment) of the distribution
        /// </summary>
        /// <returns></returns>
        public double GetMean()
        {
            double mean, var;
            GetMeanAndVariance(out mean, out var);
            return mean;
        }

        /// <summary>
        /// Get the mean and variance after truncation.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            // Use the DoubleIsBetweenOp functionality
            var result = IsBetweenGaussianOp.XAverageConditional(true, Gaussian, LowerBound, UpperBound);
            result *= Gaussian;
            result.GetMeanAndVariance(out mean, out variance);
        }

        /// <summary>
        /// Get the variance of this distribution
        /// </summary>
        /// <returns></returns>
        public double GetVariance()
        {
            double mean, var;
            GetMeanAndVariance(out mean, out var);
            return var;
        }

        /// <summary>
        /// Get the logarithm of the average value of that distribution under this distribution, i.e. log(int this(x) that(x) dx)
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public double GetLogAverageOf(TruncatedGaussian that)
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
                TruncatedGaussian product = this * that;
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(TruncatedGaussian that, double power)
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

        /// <inheritdoc/>
        public double GetProbLessThan(double x)
        {
            if (this.IsPointMass)
            {
                return (this.Point < x) ? 1.0 : 0.0;
            }
            else
            {
                double totalProbability = Gaussian.GetProbBetween(LowerBound, UpperBound);
                return Gaussian.GetProbBetween(LowerBound, x) / totalProbability;
            }
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            double totalProbability = Gaussian.GetProbBetween(LowerBound, UpperBound);
            return Gaussian.GetProbBetween(Math.Max(LowerBound, lowerBound), Math.Min(UpperBound, upperBound)) / totalProbability;
        }

        /// <inheritdoc/>
        public double GetQuantile(double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1");
            double totalProbability = Gaussian.GetProbBetween(LowerBound, UpperBound);
            double lowerProbability = Gaussian.GetProbLessThan(LowerBound);
            return Math.Min(UpperBound, Math.Max(LowerBound, Gaussian.GetQuantile(probability * totalProbability + lowerProbability)));
        }

        /// <inheritdoc/>
        public ITruncatableDistribution<double> Truncate(double lowerBound, double upperBound)
        {
            if (lowerBound > upperBound) throw new ArgumentOutOfRangeException($"lowerBound ({lowerBound}) > upperBound ({upperBound})");
            if (lowerBound > this.UpperBound) throw new ArgumentOutOfRangeException($"lowerBound ({lowerBound}) > this.UpperBound ({this.UpperBound})");
            if (upperBound < this.LowerBound) throw new ArgumentOutOfRangeException($"upperBound ({upperBound}) < this.LowerBound ({this.LowerBound})");
            return new TruncatedGaussian(Gaussian, Math.Max(LowerBound, lowerBound), Math.Min(UpperBound, upperBound));
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
                return "TruncatedGaussian.PointMass(" + Point.ToString("g4") + ")";
            }
            else if (IsUniform())
            {
                return "TruncatedGaussian.Uniform";
            }
            else
            {
                return "Truncated" + Gaussian.ToString() + "[" + LowerBound.ToString("g4") + "<x<" + UpperBound.ToString("g4") + "]";
            }
        }

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, TruncatedGaussian dist1, double weight2, TruncatedGaussian dist2)
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
                Gaussian.SetToSum(weight1, dist1.Gaussian, weight2, dist2.Gaussian);
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
        public double GetAverageLog(TruncatedGaussian that)
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
                double m, v;
                GetMeanAndVariance(out m, out v);
                return -0.5 * that.Gaussian.Precision * (m * m + v) + that.Gaussian.MeanTimesPrecision * m - that.GetLogNormalizer();
            }
        }
    }
}