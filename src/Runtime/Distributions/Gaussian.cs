// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a one-dimensional Gaussian distribution.
    /// </summary>
    /// <remarks><para>
    /// The distribution is represented by two parameters: MeanTimesPrecision and Precision.
    /// Precision is the inverse of the variance, so a Gaussian with mean m and variance v is
    /// represented as Precision = 1/v, MeanTimesPrecision = m/v.
    /// </para><para>
    /// Some special cases:
    /// If the Precision is zero, then the distribution is uniform.
    /// If the Precision is infinite, then the distribution is a point mass.  The Point property
    /// gives the location of the point mass.
    /// </para><para>
    /// The formula for the distribution is:
    /// <c>N(x;m,v) = 1/sqrt(2*pi*v) * exp(-(x-m)^2/(2v))</c>.
    /// When v=0, this reduces to delta(x-m).
    /// When v=infinity, the density is redefined to be 1.
    /// When v &lt; 0, the density is redefined to be <c>exp(-0.5*x^2*(1/v) + x*(m/v))</c>, 
    /// i.e. we drop the terms <c>exp(-m^2/(2v))/sqrt(2*pi*v)</c>.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public struct Gaussian : IDistribution<double>,
                             SettableTo<Gaussian>, SettableToProduct<Gaussian>, Diffable, SettableToUniform,
                             SettableToRatio<Gaussian>, SettableToPower<Gaussian>, SettableToWeightedSum<Gaussian>,
                             Sampleable<double>,
                             CanGetMean<double>, CanGetVariance<double>, CanGetMeanAndVarianceOut<double, double>,
                             CanSetMeanAndVariance<double, double>, CanGetLogAverageOf<Gaussian>, CanGetLogAverageOfPower<Gaussian>,
                             CanGetAverageLog<Gaussian>, CanGetLogNormalizer, CanGetMode<double>,
                             CanGetProbLessThan<double>, CanGetQuantile<double>, ITruncatableDistribution<double>
    {
        /// <summary>
        /// Mean times precision
        /// </summary>
        [DataMember]
        public double MeanTimesPrecision;

        /// <summary>
        /// Precision
        /// </summary>
        [DataMember]
        public double Precision;

        /// <summary>
        /// Gets the mean and variance, even if the distribution is improper
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        internal void GetMeanAndVarianceImproper(out double mean, out double variance)
        {
            if (IsPointMass)
            {
                variance = 0;
                mean = Point;
            }
            else
            {
                variance = 1.0 / Precision;
                if (Math.Abs(variance) > double.MaxValue) mean = 0;
                else mean = variance * MeanTimesPrecision;
            }
        }

        /// <summary>
        /// Gets the mean and variance
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (IsPointMass)
            {
                variance = 0;
                mean = Point;
            }
            else if (Precision == 0.0)
            {
                if (MeanTimesPrecision != 0)
                    throw new ImproperDistributionException(this);
                variance = Double.PositiveInfinity;
                mean = 0.0;
            }
            else if (Precision < 0.0)
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                variance = 1.0 / Precision;
                mean = MeanTimesPrecision / Precision; // must divide here to get best accuracy
            }
        }

        /// <summary>
        /// Sets the mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public void SetMeanAndVariance(double mean, double variance)
        {
            if (variance == 0 || double.IsInfinity(mean))
            {
                Point = mean;
            }
            else
            {
                double prec = 1.0 / variance;
                double meanTimesPrecision = prec * mean;
                if (prec > double.MaxValue)
                {
                    Point = mean;
                }
                else if (Math.Abs(meanTimesPrecision) > double.MaxValue)
                {
                    // This can happen when precision is too high.
                    // Lower the precision until meanTimesPrecision fits in the double-precision range.
                    MeanTimesPrecision = Math.Sign(mean) * double.MaxValue;
                    Precision = MeanTimesPrecision / mean;
                }
                else
                {
                    Precision = prec;
                    MeanTimesPrecision = meanTimesPrecision;
                }
            }
        }

        /// <summary>
        /// Gets the mean and precision
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="precision">Where to put the precision</param>
        public void GetMeanAndPrecision(out double mean, out double precision)
        {
            if (IsPointMass)
            {
                precision = Double.PositiveInfinity;
                mean = Point;
            }
            else if (Precision == 0.0)
            {
                precision = Precision;
                mean = 0.0;
            }
            else if (Precision < 0.0)
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                precision = Precision;
                mean = MeanTimesPrecision / Precision;
            }
        }

        /// <summary>
        /// Sets the mean and precision
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="precision">Precision</param>
        public void SetMeanAndPrecision(double mean, double precision)
        {
            if (double.IsPositiveInfinity(precision) || double.IsInfinity(mean))
            {
                Point = mean;
            }
            else
            {
                double meanTimesPrecision = precision * mean;
                if (Math.Abs(meanTimesPrecision) > double.MaxValue)
                {
                    // Lower the precision until meanTimesPrecision fits in the double-precision range.
                    MeanTimesPrecision = Math.Sign(mean) * double.MaxValue;
                    Precision = MeanTimesPrecision / mean;
                }
                else
                {
                    Precision = precision;
                    MeanTimesPrecision = meanTimesPrecision;
                }
            }
        }

        /// <summary>
        /// Gets the natural parameters of the distribution (mean time precision, and precision)
        /// </summary>
        /// <param name="meanTimesPrecision">Where to put the mean times precision</param>
        /// <param name="precision">Where to put the precision</param>
        public void GetNatural(out double meanTimesPrecision, out double precision)
        {
            meanTimesPrecision = this.MeanTimesPrecision;
            precision = this.Precision;
        }

        /// <summary>
        /// Sets the natural parameters of the distribution (mean time precision, and precision)
        /// </summary>
        /// <param name="meanTimesPrecision">Mean times precision</param>
        /// <param name="precision">Precision</param>
        public void SetNatural(double meanTimesPrecision, double precision)
        {
            this.MeanTimesPrecision = meanTimesPrecision;
            this.Precision = precision;
        }

        /// <summary>
        /// The most probable value
        /// </summary>
        /// <returns></returns>
        public double GetMode()
        {
            return GetMean();
        }

        /// <summary>
        /// Gets the expected value E(x)
        /// </summary>
        /// <returns>E(x)</returns>
        public double GetMean()
        {
            if (IsPointMass) return Point;
            else if (Precision <= 0.0) throw new ImproperDistributionException(this);
            else return MeanTimesPrecision / Precision;
        }

        /// <summary>
        /// Gets the variance
        /// </summary>
        /// <returns>Variance</returns>
        public double GetVariance()
        {
            if (IsPointMass) return 0.0;
            else if (Precision < 0.0) throw new ImproperDistributionException(this);
            else return 1.0 / Precision;
        }

        /// <inheritdoc/>
        public double GetProbLessThan(double x)
        {
            if (this.IsPointMass)
            {
                return (this.Point < x) ? 1.0 : 0.0;
            }
            else if (Precision <= 0.0)
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double mean = MeanTimesPrecision / Precision;
                double sqrtPrec = Math.Sqrt(Precision);
                // (x - m)/sqrt(v) = x/sqrt(v) - m/v * sqrt(v)
                return MMath.NormalCdf((x - mean) * sqrtPrec);
            }
        }

        /// <inheritdoc/>
        public double GetQuantile(double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException("probability < 0");
            if (probability > 1) throw new ArgumentOutOfRangeException("probability > 1");
            if (this.IsPointMass)
            {
                return (probability == 1.0) ? MMath.NextDouble(this.Point) : this.Point;
            }
            else if (Precision <= 0.0)
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double mean = MeanTimesPrecision / Precision;
                double sqrtPrec = Math.Sqrt(Precision);
                return MMath.NormalCdfInv(probability) / sqrtPrec + mean;
            }
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            double mean = GetMean();
            if (lowerBound > mean)
            {
                // same as below but more accurate
                // For a Gaussian, CDF(2*mean - x) = 1 - CDF(x)
                return Math.Max(0, GetProbLessThan(2 * mean - lowerBound) - GetProbLessThan(2 * mean - upperBound));
            }
            else
            {
                return Math.Max(0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
            }
        }

        /// <inheritdoc/>
        public ITruncatableDistribution<double> Truncate(double lowerBound, double upperBound)
        {
            if (lowerBound > upperBound) throw new ArgumentOutOfRangeException($"lowerBound ({lowerBound}) > upperBound ({upperBound})");
            return new TruncatedGaussian(this, lowerBound, upperBound);
        }

        /// <summary>
        /// Asks whether the instance is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return (Precision == Double.PositiveInfinity); }
        }

        /// <summary>
        /// Sets this instance to a point mass. The location of the
        /// point mass is the existing mean
        /// </summary>
        private void SetToPointMass()
        {
            Precision = Double.PositiveInfinity;
        }

        /// <summary>
        /// Sets/gets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get
            {
                // The accessor must succeed, even if the distribution is not a point mass.
                //Assert.IsTrue(IsPointMass, "The distribution is not a point mass");
                return MeanTimesPrecision;
            }
            set
            {
                SetToPointMass();
                MeanTimesPrecision = value;
            }
        }

        /// <summary>
        /// Sets this Gaussian instance to be a uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            Precision = 0;
            MeanTimesPrecision = 0;
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return (Precision == 0) && (MeanTimesPrecision == 0);
        }

        /// <summary>
        /// Asks whether this Gaussian instance is proper or not. A Gaussian distribution
        /// is proper only if Precision > 0.
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return (Precision > 0);
        }

        /// <summary>
        /// Evaluates the log of one-dimensional Gaussian density.
        /// </summary>
        /// <param name="x">Must be finite.</param>
        /// <param name="mean">Must be finite.</param>
        /// <param name="variance">Any real number.  May be zero or negative.</param>
        /// <remarks>
        /// <c>N(x;m,v) = 1/sqrt(2*pi*v) * exp(-(x-m)^2/(2v))</c>.
        /// When v=0, this reduces to delta(x-m).
        /// When v=infinity, the density is redefined to be 1.
        /// When v &lt; 0, the density is redefined to be <c>exp(-0.5*x^2*(1/v) + x*(m/v))</c>, 
        /// i.e. we drop the terms <c>exp(-m^2/(2v))/sqrt(2*pi*v)</c>.
        /// </remarks>
        /// <returns><c>log(N(x;mean,variance))</c></returns>
        public static double GetLogProb(double x, double mean, double variance)
        {
            if (variance == 0) return (x == mean) ? 0.0 : Double.NegativeInfinity;
            else if (Double.IsPositiveInfinity(variance)) return 0.0;
            else
            {
                if (variance > 0)
                {
                    double diff = x - mean;
                    return -0.5 * (Math.Log(variance) + diff * diff / variance) - MMath.LnSqrt2PI;
                }
                else
                {
                    return (mean - 0.5 * x) * x / variance;
                }
            }
        }

        /// <summary>
        /// Evaluates the log of this one-dimensional Gaussian density.
        /// </summary>
        /// <param name="x">Must be finite.</param>
        /// <returns><c>log(N(x;mean,variance))</c></returns>
        public double GetLogProb(double x)
        {
            if (IsPointMass)
            {
                return (x == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else if (IsUniform())
            {
                return 0;
            }
            else if (IsProper())
            {
                // This approach avoids rounding errors.
                // (x - mean)^2 * Precision = (x*Precision - MeanTimesPrecision)^2/Precision
                double diff = x * Precision - MeanTimesPrecision;
                return 0.5 * (Math.Log(Precision) - diff * (diff / Precision)) - MMath.LnSqrt2PI;
            }
            else
            {
                return (MeanTimesPrecision - 0.5 * x * Precision) * x;
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
                return MMath.LnSqrt2PI + 0.5 * (MeanTimesPrecision * MeanTimesPrecision / Precision - Math.Log(Precision));
            }
            else
            {
                return 0.0;
            }
        }

        /// <summary>
        /// Gets the expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Gaussian that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && this.Point == that.Point) return 0.0;
                else return Double.NegativeInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                // that.Precision != inf
                double mean, variance;
                GetMeanAndVariance(out mean, out variance);
                return that.GetLogProb(mean) - 0.5 * that.Precision * variance;
            }
        }

        /// <summary>
        /// Samples from a Gaussian distribution with the specified mean and precision
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(double mean, double precision)
        {
            if (precision <= 0) throw new ArgumentException("precision <= 0 (" + precision + ")");
            return Rand.Normal() / Math.Sqrt(precision) + mean;
        }

        /// <summary>
        /// Samples from this Gaussian distribution
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
                return Sample(MeanTimesPrecision / Precision, Precision);
            }
        }

        /// <summary>
        /// Samples from this Gaussian distribution. This override is only
        /// present to support the Sampleable interface
        /// </summary>
        /// <param name="result">Ignored</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample(double result)
        {
            return Sample();
        }

        /// <summary>
        /// Sets this Gaussian instance to have the parameter values of that Gaussian instance
        /// </summary>
        /// <param name="that">That Gaussian</param>
        public void SetTo(Gaussian that)
        {
            MeanTimesPrecision = that.MeanTimesPrecision;
            Precision = that.Precision;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two Gaussians.
        /// </summary>
        /// <param name="a">The first Gaussian</param>
        /// <param name="b">The second Gaussian</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(Gaussian a, Gaussian b)
        {
            if (a.IsPointMass)
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                Point = a.Point;
            }
            else if (b.IsPointMass)
            {
                Point = b.Point;
            }
            else
            {
                Precision = a.Precision + b.Precision;
                MeanTimesPrecision = a.MeanTimesPrecision + b.MeanTimesPrecision;
                if (Precision > double.MaxValue || Math.Abs(MeanTimesPrecision) > double.MaxValue)
                {
                    if (a.IsUniform()) SetTo(b);
                    else if (b.IsUniform()) SetTo(a);
                    else
                    {
                        // (am*ap + bm*bp)/(ap + bp) = am*w + bm*(1-w)
                        // w = 1/(1 + bp/ap)
                        double w = 1 / (1 + b.Precision / a.Precision);
                        Point = a.GetMean() * w + b.GetMean() * (1 - w);
                    }
                }
            }
        }

        /// <summary>
        /// Creates a new Gaussian which is the product of two other Gaussians
        /// </summary>
        /// <param name="a">First Gaussian</param>
        /// <param name="b">Second Gaussian</param>
        /// <returns>Result</returns>
        public static Gaussian operator *(Gaussian a, Gaussian b)
        {
            Gaussian result = new Gaussian();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two Gaussians, optionally forcing the precision to be non-negative.
        /// </summary>
        /// <param name="numerator">The numerator Gaussian.  May be the same as <c>this</c></param>
        /// <param name="denominator">The denominator Gaussian</param>
        /// <param name="forceProper">If true, the result will have non-negative precision, under the constraint that result*denominator has the same mean as numerator</param>
        public void SetToRatio(Gaussian numerator, Gaussian denominator, bool forceProper = false)
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
            else
            {
                if (forceProper && numerator.Precision < denominator.Precision)
                {
                    if (numerator.IsUniform())
                    {
                        MeanTimesPrecision = -denominator.MeanTimesPrecision;
                    }
                    else
                    {
                        // must not modify this before computing numerator.GetMean()
                        MeanTimesPrecision = numerator.GetMean() * denominator.Precision - denominator.MeanTimesPrecision;
                    }
                    Precision = 0;
                }
                else
                {
                    Precision = numerator.Precision - denominator.Precision;
                    MeanTimesPrecision = numerator.MeanTimesPrecision - denominator.MeanTimesPrecision;
                }
            }
        }

        /// <summary>
        /// Creates a new Gaussian which is the ratio of two other Gaussians
        /// </summary>
        /// <param name="numerator">numerator Gaussian</param>
        /// <param name="denominator">denominator Gaussian</param>
        /// <returns>Result</returns>
        public static Gaussian operator /(Gaussian numerator, Gaussian denominator)
        {
            Gaussian result = new Gaussian();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source Gaussian to some exponent.
        /// </summary>
        /// <param name="dist">The source Gaussian</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Gaussian dist, double exponent)
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
            Precision = dist.Precision * exponent;
            MeanTimesPrecision = dist.MeanTimesPrecision * exponent;
            if (Precision > double.MaxValue || Math.Abs(MeanTimesPrecision) > double.MaxValue)
            {
                Point = dist.GetMean();
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Gaussian operator ^(Gaussian dist, double exponent)
        {
            Gaussian result = new Gaussian();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Sets the mean and variance to match a Gaussian mixture.
        /// </summary>
        /// <param name="weight1">First weight</param>
        /// <param name="g1">First Gaussian</param>
        /// <param name="weight2">Second weight</param>
        /// <param name="g2">Second Gaussian</param>
        public void SetToSum(double weight1, Gaussian g1, double weight2, Gaussian g2)
        {
            SetTo(WeightedSum<Gaussian>(this, weight1, g1, weight2, g2));
        }

        /// <summary>
        /// Creates a distribution of the specified type which matches the mean and variance
        /// of a Gaussian mixture. The distribution type must implement <see cref="CanGetMeanAndVarianceOut&lt;Double, Double&gt;"/>,
        /// </summary>
        /// <see cref="CanSetMeanAndVariance&lt;Double, Double&gt;"/>, and <see cref="SettableToUniform"/>
        /// <typeparam name="T">Distribution type for the mixture</typeparam>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second distribution</param>
        /// <param name="result">Resulting distribution</param>
        public static T WeightedSum<T>(T result, double weight1, T dist1, double weight2, T dist2)
            where T : CanGetMeanAndVarianceOut<double, double>, CanSetMeanAndVariance<double, double>, SettableToUniform, SettableTo<T>
        {
            if (weight1 + weight2 == 0) result.SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            else if (weight1 == 0) result.SetTo(dist2);
            else if (weight2 == 0) result.SetTo(dist1);
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2)) throw new ArgumentException("both weights are infinity");
                else result.SetTo(dist1);
            }
            else if (double.IsPositiveInfinity(weight2)) result.SetTo(dist2);
            else
            {
                // w = weight1/(weight1 + weight2)
                // m = w*m1 + (1-w)*m2
                // v+m^2 = w*(v1+m1^2) + (1-w)*(v2+m2^2)
                // v = w*v1 + (1-w)*v2 + w*(m1-m)^2 + (1-w)*(m2-m)^2
                double m1, v1, m2, v2;
                dist1.GetMeanAndVariance(out m1, out v1);
                dist2.GetMeanAndVariance(out m2, out v2);
                if (m1 == m2)
                {
                    // catch this to avoid roundoff errors
                    if (v1 == v2) result.SetTo(dist1); // avoid roundoff errors
                    else
                    {
                        double v = (weight1 * v1 + weight2 * v2) / (weight1 + weight2);
                        result.SetMeanAndVariance(m1, v);
                    }
                }
                else
                {
                    double invZ = 1.0 / (weight1 + weight2);
                    double m = (weight1 * m1 + weight2 * m2) * invZ;
                    double dm1 = m1 - m;
                    double dm2 = m2 - m;
                    double v = (weight1 * v1 + weight2 * v2 + weight1 * dm1 * dm1 + weight2 * dm2 * dm2) * invZ;
                    result.SetMeanAndVariance(m, v);
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the integral of the product of two Gaussians.
        /// </summary>
        /// <param name="that"></param>
        /// <remarks>
        /// <c>this = N(x;m1,v1)</c>.
        /// <c>that = N(x;m2,v2)</c>.
        /// <c>int_(-infinity)^(infinity) N(x;m1,v1) N(x;m2,v2) dx = N(m1; m2, v1+v2)</c>.
        /// When improper, the density is redefined to be <c>exp(-0.5*x^2*(1/v) + x*(m/v))</c>, 
        /// i.e. we drop the terms <c>exp(-m^2/(2v))/sqrt(2*pi*v)</c>.
        /// </remarks>
        /// <returns>log(N(m1;m2,v1+v2)).</returns>
        public double GetLogAverageOf(Gaussian that)
        {
            if (IsPointMass)
            {
                return that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                return GetLogProb(that.Point);
            }
            else if (IsProper() && that.IsProper())
            {
                // neither this nor that is a point mass.
                // int_x N(x;m1,v1) N(x;m2,v2) dx = N(m1;m2,v1+v2)
                // (m1-m2)^2/(v1+v2) = (m1-m2)^2/(v1*v2*product.Prec)
                // (m1-m2)^2/(v1*v2) = (m1/(v1*v2) - m2/(v1*v2))^2 *v1*v2
                double m1, v1, m2, v2;
                this.GetMeanAndVariance(out m1, out v1);
                that.GetMeanAndVariance(out m2, out v2);
                return Gaussian.GetLogProb(m1, m2, v1 + v2);
            }
            else
            {
                // This is less accurate but works for improper distributions.
                Gaussian product = this * that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper.");
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Gaussian that, double power)
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
        /// The maximum difference between the parameters of this Gaussian
        /// and that Gaussian
        /// </summary>
        /// <param name="thatd">That Gaussian</param>
        /// <returns>The maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is Gaussian)) return Double.PositiveInfinity;
            Gaussian that = (Gaussian)thatd;
#if true
            double diff1 = MMath.AbsDiff(MeanTimesPrecision, that.MeanTimesPrecision);
            double diff2 = MMath.AbsDiff(Precision, that.Precision);
#else
      double diff1 = Math.Abs(GetMean() - that.GetMean());
      double diff2 = Math.Abs(Math.Sqrt(GetVariance()) - Math.Sqrt(that.GetVariance()));
#endif
            return Math.Max(diff1, diff2);
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            return (MaxDiff(thatd) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(MeanTimesPrecision.GetHashCode(), Precision.GetHashCode());
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(Gaussian a, Gaussian b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(Gaussian a, Gaussian b)
        {
            return !(a == b);
        }

#if class
    /// <summary>
    /// Create a uniform Gaussian distribution.
    /// </summary>
    public Gaussian()
    {
    }
#endif

        /// <summary>
        /// Creates a Gaussian distribution with specified mean and variance.
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public Gaussian(double mean, double variance)
        {
            // Initialise the object first, so that a call to an instance method is possible
            Precision = 0;
            MeanTimesPrecision = 0;

            this.SetMeanAndVariance(mean, variance);
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Gaussian(Gaussian that)
            : this()
        {
            SetTo(that);
        }

        /// <summary>
        /// Clones this Gaussian. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Gaussian type</returns>
        public object Clone()
        {
            return new Gaussian(this);
        }

        /// <summary>
        /// Creates a Gaussian distribution with given mean and variance.
        /// </summary>
        /// <param name="mean">The desired mean.</param>
        /// <param name="variance">The desired variance.</param>
        /// <returns>A new Gaussian distribution.</returns>
        public static Gaussian FromMeanAndVariance(double mean, double variance)
        {
            Gaussian result = new Gaussian();
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        /// <summary>
        /// Creates a Gaussian distribution with given mean and precision.
        /// </summary>
        /// <param name="mean">The desired mean.</param>
        /// <param name="precision">precision = 1/variance.</param>
        /// <returns>A new Gaussian distribution.</returns>
        public static Gaussian FromMeanAndPrecision(double mean, double precision)
        {
            Gaussian result = new Gaussian();
            result.SetMeanAndPrecision(mean, precision);
            return result;
        }

        /// <summary>
        /// Creates a new Gaussian distribution from its natural parameters
        /// (Mean times precision, and Precision)
        /// </summary>
        /// <param name="meanTimesPrecision">Mean time precision</param>
        /// <param name="precision">Precision</param>
        /// <returns>A new Gaussian distribution.</returns>
        [Construction("MeanTimesPrecision", "Precision")]
        public static Gaussian FromNatural(double meanTimesPrecision, double precision)
        {
            Gaussian result = new Gaussian();
            result.MeanTimesPrecision = meanTimesPrecision;
            result.Precision = precision;
            return result;
        }

        /// <summary>
        /// Construct a Gaussian distribution whose log-pdf has the given derivatives at a point.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="dlogp"></param>
        /// <param name="ddlogp"></param>
        /// <param name="forceProper">If true and both derivatives cannot be matched, match only the first.</param>
        /// <returns></returns>
        public static Gaussian FromDerivatives(double x, double dlogp, double ddlogp, bool forceProper)
        {
            // logp = -0.5*r*(x-m)^2 + const.
            // dlogp = -r*(x-m)
            // ddlogp = -r
            if (forceProper && ddlogp > 0)
            {
                ddlogp = 0;
            }
            if (ddlogp == 0) return Gaussian.FromNatural(dlogp, 0);
            if (double.IsInfinity(dlogp)) return Gaussian.PointMass(dlogp); // to avoid NaN
            double meanTimesPrecision = dlogp - ddlogp * x;
            if (Math.Abs(meanTimesPrecision) > double.MaxValue)
            {
                meanTimesPrecision = ddlogp * (dlogp / ddlogp - x);
            }
            return Gaussian.FromNatural(meanTimesPrecision, -ddlogp);
        }

        /// <summary>
        /// Get the derivatives of the log-pdf at a point.
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="x"></param>
        /// <param name="dlogp">On exit, the first derivative.</param>
        /// <param name="ddlogp">On exit, the second derivative.</param>
        public static void GetDerivatives(Gaussian dist, double x, out double dlogp, out double ddlogp)
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
            dlogp = this.MeanTimesPrecision - this.Precision * x;
            if (Math.Abs(dlogp) > double.MaxValue)
            {
                dlogp = this.Precision * (this.MeanTimesPrecision / this.Precision - x);
            }
            ddlogp = -this.Precision;
        }

        /// <summary>
        /// Creates a new uniform Gaussian distribution
        /// </summary>
        /// <returns>A new uniform Gaussian distribution.</returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static Gaussian Uniform()
        {
            Gaussian result = new Gaussian();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Create a new point mass Gaussian distribution at a specified location
        /// </summary>
        /// <param name="mean">The location for the point mass</param>
        /// <returns>A new point mass Gaussian distribution.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static Gaussian PointMass(double mean)
        {
            Gaussian result = new Gaussian();
            result.Point = mean;
            return result;
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

        /// <summary>
        /// Get a string representation of the distribution with a given number format
        /// </summary>
        /// <param name="format">Format string to use for each number</param>
        /// <returns>A string</returns>
        public string ToString(string format)
        {
            if (IsPointMass)
            {
                return "Gaussian.PointMass(" + Point.ToString(format) + ")";
            }
            else if (IsUniform())
            {
                return "Gaussian.Uniform";
            }
            else if (IsProper())
            {
                double mean, variance;
                GetMeanAndVariance(out mean, out variance);
                return "Gaussian(" + mean.ToString(format) + ", " + variance.ToString(format) + ")";
            }
            else
            {
                return "Gaussian(m/v=" + MeanTimesPrecision.ToString(format) + ", 1/v=" + Precision.ToString(format) + ")";
            }
        }
    }
}