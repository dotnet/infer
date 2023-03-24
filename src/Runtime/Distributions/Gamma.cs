// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Diagnostics;
    using System.Runtime.Serialization;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A Gamma distribution on positive reals.
    /// </summary>
    /// <remarks><para>
    /// The distribution is 
    /// <c>p(x) = x^(a-1)*exp(-x*b)*b^a/Gamma(a)</c>.
    /// In this code, the <c>a</c> parameter is called the "Shape" and the <c>b</c> parameter
    /// is called the "Rate".  The distribution is sometimes also parameterized by (shape,scale)
    /// where scale = 1/rate.
    /// The mean of the distribution is <c>shape/rate</c> and the variance is 
    /// <c>shape/rate^2</c>.
    /// </para><para>
    /// Special cases:
    /// When the shape is 1 and rate is 0, the distribution is uniform.
    /// When the shape is infinity, the distribution is a point mass and the density is delta(x-Point)
    /// where the Point property gives the mean.
    /// When a &lt;= 0 or b &lt;= 0 the <c>b^a/Gamma(a)</c> term is dropped.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public struct Gamma : IDistribution<double>, //ICursor,
                          SettableTo<Gamma>, SettableToProduct<Gamma>, Diffable, SettableToUniform,
                          SettableToRatio<Gamma>, SettableToPower<Gamma>, SettableToWeightedSum<Gamma>,
                          Sampleable<double>, CanGetMean<double>, CanGetVariance<double>,
                          CanGetMeanAndVarianceOut<double, double>, CanSetMeanAndVariance<double, double>,
                          CanGetLogAverageOf<Gamma>, CanGetLogAverageOfPower<Gamma>,
                          CanGetAverageLog<Gamma>, CanGetLogNormalizer, CanGetMode<double>,
                          CanGetProbLessThan<double>, CanGetQuantile<double>
    {
        /// <summary>
        /// Rate parameter for the distribution
        /// </summary>
        [DataMember]
        public double Rate;

        /// <summary>
        /// Shape parameter for the distribution
        /// </summary>
        [DataMember]
        public double Shape;

        /// <summary>
        /// The most probable value
        /// </summary>
        /// <returns></returns>
        public double GetMode()
        {
            if (IsPointMass)
                return Point;
            else if (Shape <= 1.0)
                return 0.0;
            else if (Rate == 0.0)
                return Double.PositiveInfinity;
            else
                return (Shape - 1) / Rate;
        }

        /// <summary>
        /// Gets the expected value E(x) - calculated as shape/rate
        /// </summary>
        /// <returns>E(x)</returns>
        public double GetMean()
        {
            if (IsPointMass) return Point;
            else if (Rate == 0.0) return Double.PositiveInfinity;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else return Shape / Rate;
        }

        /// <summary>
        /// Gets the variance - calculated as shape/rate^2
        /// </summary>
        /// <returns>Variance</returns>
        public double GetVariance()
        {
            if (IsPointMass) return 0;
            else if (Rate == 0.0) return Double.PositiveInfinity;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else return Shape / (Rate * Rate);
        }

        /// <summary>
        /// Gets the mean (shape/rate) and variance (shape/rate^2)
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (IsPointMass)
            {
                mean = Point;
                variance = 0;
            }
            else if (Rate == 0.0)
            {
                mean = Double.PositiveInfinity;
                variance = Double.PositiveInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                // 2 multiplies + 1 divide is faster than 2 divides.
                double ib = 1.0 / Rate;
                mean = Shape * ib;
                variance = mean * ib;
            }
        }

        /// <summary>
        /// Sets the mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public void SetMeanAndVariance(double mean, double variance)
        {
            if (variance == 0)
            {
                Point = mean;
            }
            else if (Double.IsPositiveInfinity(variance))
            {
                SetToUniform();
            }
            else if (variance < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(variance), variance, "variance < 0");
            }
            else
            {
                Rate = mean / variance;
                Shape = mean * Rate;
                if (Shape > double.MaxValue || Rate > double.MaxValue)
                {
                    // If the variance is so small that Shape overflows, treat the distribution as a point mass.
                    Point = mean;
                }
            }
        }

        /// <summary>
        /// Creates a new Gamma distribution from mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new Gamma instance</returns>
        public static Gamma FromMeanAndVariance(double mean, double variance)
        {
            Gamma g = new Gamma();
            g.SetMeanAndVariance(mean, variance);
            return g;
        }

        /// <summary>
        /// Sets the shape and rate (rate = 1/scale) parameters of the distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="rate">rate = 1/scale</param>
        public void SetShapeAndRate(double shape, double rate)
        {
            this.Shape = shape;
            this.Rate = rate;
            CheckForPointMass();
        }

        private void CheckForPointMass()
        {
            if (!IsPointMass && Rate > double.MaxValue)
            {
                Point = 0;
            }
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given shape and rate parameters.
        /// </summary>
        /// <param name="shape">shape</param>
        /// <param name="rate">rate = 1/scale</param>
        /// <returns>A new Gamma distribution.</returns>
        [Construction("Shape", "Rate")]
        public static Gamma FromShapeAndRate(double shape, double rate)
        {
            Gamma result = new Gamma();
            result.SetShapeAndRate(shape, rate);
            return result;
        }

        /// <summary>
        /// Gets the scale (1/rate)
        /// </summary>
        public double GetScale()
        {
            return 1.0 / Rate;
        }

        /// <summary>
        /// Gets the shape and scale (1/rate)
        /// </summary>
        /// <param name="shape">Where to put the shape</param>
        /// <param name="scale">Where to put the scale</param>
        public void GetShapeAndScale(out double shape, out double scale)
        {
            shape = this.Shape;
            scale = 1.0 / Rate;
        }

        /// <summary>
        /// Sets the shape and scale for this instance
        /// </summary>
        /// <param name="shape">Shape</param>
        /// <param name="scale">Scale</param>
        public void SetShapeAndScale(double shape, double scale)
        {
            if (double.IsPositiveInfinity(shape)) throw new ArgumentOutOfRangeException(nameof(shape), shape, "shape is infinite.  To create a point mass, set the Point property.");
            SetShapeAndRate(shape, 1.0 / scale);
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given shape and scale parameters.
        /// </summary>
        /// <param name="shape">shape</param>
        /// <param name="scale">scale</param>
        /// <returns>A new Gamma distribution.</returns>
        public static Gamma FromShapeAndScale(double shape, double scale)
        {
            Gamma result = new Gamma();
            result.SetShapeAndScale(shape, scale);
            return result;
        }

        /// <summary>
        /// Gets the logarithm of the expected value minus the expected logarithm, more accurately than directly computing <c>Math.Log(GetMean()) - GetMeanLog()</c>.
        /// </summary>
        /// <returns></returns>
        public double GetLogMeanMinusMeanLog()
        {
            if (IsPointMass) return 0;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else return LogMinusDigamma(Shape);
        }

        static readonly double largeShape = Math.Sqrt(Math.Sqrt(1.0 / 120 / MMath.Ulp1));

        private static double LogMinusDigamma(double shape)
        {
            if (shape > largeShape)
                // The next term in the series is -1/120/shape^4, which bounds the error.
                return (0.5 - 1.0 / 12 / shape) / shape;
            else
                return Math.Log(shape) - MMath.Digamma(shape);
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given mean and mean logarithm.
        /// </summary>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="meanLog">Desired expected logarithm.</param>
        /// <returns>A new Gamma distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Gamma distribution
        /// from data given by sufficient statistics.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization. The algorithm is a generalized Newton iteration, 
        /// described in "Estimating a Gamma distribution" by T. Minka, 2002.
        /// </remarks>
        public static Gamma FromMeanAndMeanLog(double mean, double meanLog)
        {
            return FromLogMeanMinusMeanLog(mean, Math.Log(mean) - meanLog);
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given mean and mean logarithm.
        /// </summary>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="logMeanMinusMeanLog">Logarithm of desired expected value minus desired expected logarithm.</param>
        /// <returns>A new Gamma distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Gamma distribution
        /// from data given by sufficient statistics.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization. The algorithm is a generalized Newton iteration, 
        /// described in "Estimating a Gamma distribution" by T. Minka, 2002.
        /// </remarks>
        public static Gamma FromLogMeanMinusMeanLog(double mean, double logMeanMinusMeanLog)
        {
            if (logMeanMinusMeanLog <= 0) return Gamma.PointMass(mean);
            double shape = 0.5 / logMeanMinusMeanLog;
            for (int iter = 0; iter < 100; iter++)
            {
                double g = LogMinusDigamma(shape) - logMeanMinusMeanLog;
                //Trace.WriteLine($"shape = {shape} g = {g}");
                if (MMath.AreEqual(g, 0)) break;
                shape /= 1 + g / (1 - shape * MMath.Trigamma(shape));
            }
            if (double.IsNaN(shape)) throw new InferRuntimeException("shape is nan");
            if (shape > double.MaxValue) return Gamma.PointMass(mean);
            return Gamma.FromShapeAndRate(shape, shape / mean);
        }

        /// <summary>
        /// Construct a Gamma distribution whose pdf has the given derivatives at a point.
        /// </summary>
        /// <param name="x">Cannot be negative</param>
        /// <param name="dLogP">Desired derivative of log-density at x</param>
        /// <param name="xdLogP">Desired derivative of log-density at x, times x</param>
        /// <param name="xxddLogP">Desired second derivative of log-density at x, times x squared</param>
        /// <param name="forceProper">If true and both derivatives cannot be matched by a proper distribution, match only the first.</param>
        /// <returns></returns>
        public static Gamma FromDerivatives(double x, double dLogP, double xdLogP, double xxddLogP, bool forceProper)
        {
            if (x < 0)
                throw new ArgumentOutOfRangeException(nameof(x), x, "x < 0");
            double a;
            if (double.IsPositiveInfinity(x))
            {
                if (xxddLogP < 0) return Gamma.PointMass(x);
                else if (xxddLogP == 0) a = 0.0;
                else if (forceProper)
                {
                    if (dLogP <= 0) return Gamma.FromShapeAndRate(1, -dLogP);
                    else return Gamma.PointMass(x);
                }
                else return Gamma.FromShapeAndRate(-x, -x - dLogP);
            }
            else
            {
                a = -xxddLogP;
                a = Math.Min(double.MaxValue, a);
            }
            if (forceProper)
            {
                if (dLogP <= 0)
                {
                    if (a < 0)
                        a = 0;
                }
                else
                {
                    // TODO: check that derivative of result is correct
                    double amin = xdLogP;
                    if (amin > double.MaxValue)
                    {
                        // we want a/x - b == dLogP > 0
                        // if a / x == dLogP == infinity then b = 0 works
                        // let a be the smallest value for which a / x == infinity                     
                        a = MMath.NextDouble(MMath.LargestDoubleProduct(double.MaxValue, x));
                        a = Math.Min(double.MaxValue, a);
                        Debug.Assert(a + 1 <= double.MaxValue);
                        return Gamma.FromShapeAndRate(a + 1, 0);
                    }
                    else if (a <= amin || double.IsPositiveInfinity(dLogP))
                    {
                        Debug.Assert(amin + 1 <= double.MaxValue);
                        return Gamma.FromShapeAndRate(amin + 1, 0);
                    }
                }
            }
            double b = ((a == 0) ? 0 : (a / x)) - dLogP;
            if (double.IsPositiveInfinity(b))
            {
                b = (a - xdLogP) / x;
            }

            if (forceProper)
            {
                // correct roundoff errors that might make b negative
                b = Math.Max(b, 0);
            }
            Debug.Assert(a + 1 <= double.MaxValue);
            return Gamma.FromShapeAndRate(a + 1, b);
        }

        /// <summary>
        /// Get the derivatives of the log-pdf at a point.
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="x"></param>
        /// <param name="dlogp">On exit, the first derivative.</param>
        /// <param name="ddlogp">On exit, the second derivative.</param>
        public static void GetDerivatives(Gamma dist, double x, out double dlogp, out double ddlogp)
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
            if (IsPointMass)
            {
                ddlogp = double.NegativeInfinity;
                if (x < Point) dlogp = double.PositiveInfinity;
                else if (x == Point) dlogp = 0;
                else dlogp = double.NegativeInfinity;
                return;
            }

            dlogp = -this.Rate;
            ddlogp = 0;
            double a = this.Shape - 1;
            if (a != 0) // avoid 0/0
            {
                ddlogp -= a / (x * x);
                dlogp += a / x;
                if (double.IsPositiveInfinity(dlogp))
                {
                    dlogp = (a - this.Rate * x) / x;
                }
            }
        }

        /// <summary>
        /// Sets the natural parameters of the distribution.
        /// </summary>
        /// <param name="shapeMinus1">The shape parameter - 1.</param>
        /// <param name="rate">rate = 1/scale</param>
        public void SetNatural(double shapeMinus1, double rate)
        {
            SetShapeAndRate(shapeMinus1 + 1, rate);
        }

        /// <summary>
        /// Computes E[log(x)]
        /// </summary>
        /// <returns></returns>
        public double GetMeanLog()
        {
            if (IsPointMass) return Math.Log(Point);
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else return MMath.Digamma(Shape) - Math.Log(Rate);
        }

        /// <summary>
        /// Computes E[1/x]
        /// </summary>
        /// <returns></returns>
        public double GetMeanInverse()
        {
            if (IsPointMass) return 1.0 / Point;
            //else if (Rate == 0.0) return 0.0;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else if (Shape <= 1.0)
            {
                throw new ArgumentException("Cannot compute E[1/x] for " + this + " (shape <= 1)");
            }
            else return Rate / (Shape - 1);
        }

        /// <summary>
        /// Computes E[x^power]
        /// </summary>
        /// <returns></returns>
        public double GetMeanPower(double power)
        {
            if (power == 0.0) return 1.0;
            else if (IsPointMass) return Math.Pow(Point, power);
            //else if (Rate == 0.0) return (power > 0) ? Double.PositiveInfinity : 0.0;
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else if (Shape <= -power)
            {
                throw new ArgumentException("Cannot compute E[x^" + power + "] for " + this + " (shape <= " + (-power) + ")");
            }
            else return Math.Exp(MMath.RisingFactorialLn(Shape, power) - power * Math.Log(Rate));
        }

        /// <inheritdoc/>
        public double GetProbLessThan(double x)
        {
            if (x < 0.0)
            {
                return 0.0;
            }
            else if (double.IsPositiveInfinity(x))
            {
                return 1.0;
            }
            else if (this.IsPointMass)
            {
                return (this.Point < x) ? 1.0 : 0.0;
            }
            else if (this.IsUniform())
            {
                throw new ImproperDistributionException(this);
            }
            else if (x*Rate < 0.25 && x*Rate*Shape < 1e-16)
            {
                // GammaLower =approx x^a/Gamma(a+1)
                return Math.Exp(Shape * (Math.Log(x) + Math.Log(Rate))) / MMath.Gamma(1 + Shape);
            }
            else
            {
                return MMath.GammaLower(Shape, x * Rate);
            }
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            return MMath.GammaProbBetween(Shape, Rate, lowerBound, upperBound);
        }

        /// <summary>
        /// Returns the value x such that GetProbLessThan(x) == probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns></returns>
        public double GetQuantile(double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), probability, "probability < 0");
            if (probability > 1) throw new ArgumentOutOfRangeException(nameof(probability), probability, "probability > 1");
            if (this.IsPointMass)
            {
                return (probability == 1.0) ? MMath.NextDouble(this.Point) : this.Point;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else if (MMath.AreEqual(probability, 0))
            {
                return 0;
            }
            else if (MMath.AreEqual(probability, 1))
            {
                return double.PositiveInfinity;
            }
            else if (Shape == 1)
            {
                // cdf is 1 - exp(-x*rate)
                return -Math.Log(1 - probability) / Rate;
            }
            else
            {
                // Binary search
                double lowerBound = 0;
                double upperBound = double.MaxValue;
                while (lowerBound < upperBound)
                {
                    double average = MMath.Average(lowerBound, upperBound);
                    double p = GetProbLessThan(average);
                    if (p == probability)
                    {
                        return average;
                    }
                    else if (p < probability)
                    {
                        lowerBound = MMath.NextDouble(average);
                    }
                    else
                    {
                        upperBound = MMath.PreviousDouble(average);
                    }
                }
                return lowerBound;
            }
        }

        /// <summary>
        /// Asks whether the instance is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return (Shape == Double.PositiveInfinity); }
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
                return Rate;
            }
            set
            {
                Shape = Double.PositiveInfinity;
                Rate = value;
            }
        }

        /// <summary>
        /// Sets this Gamma instance to be a uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            Shape = 1;
            Rate = 0;
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return (Shape == 1) && (Rate == 0);
        }

        /// <summary>
        /// Logarithm of the Gamma density function.
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">Rate parameter</param>
        /// <param name="normalized">If true, include the normalizer</param>
        /// <returns>log(Gamma(x;shape,rate))</returns>
        /// <remarks>
        /// The distribution is <c>p(x) = x^(a-1)*exp(-x*b)*b^a/Gamma(a)</c>.
        /// When a &lt;= 0 or b &lt;= 0 the <c>b^a/Gamma(a)</c> term is dropped.
        /// Thus if shape = 1 and rate = 0 the density is 1.
        /// </remarks>
        public static double GetLogProb(double x, double shape, double rate, bool normalized = true)
        {
            if (x < 0) return double.NegativeInfinity;
            if (x > double.MaxValue) // Avoid subtracting infinities below
            {
                if (rate > 0) return -x;
                else if (rate < 0) return x;
                // fall through when rate == 0
            }
            if (normalized) normalized = IsProper(shape, rate);
            if (shape > 1e10)
            {
                // In double precision, we can assume GammaLn(x) = (x-0.5)*log(x) - x + MMath.LnSqrt2PI for x > 1e10
                // Also log(1-1/x) = -1/x - 0.5/x^2  for x > 1e10
                // We compute the density in a way that ensures the maximum is at the mode returned by GetMode.
                double mode = (shape - 1) / rate; // cannot be zero
                double xOverMode = x / mode;
                if (xOverMode > double.MaxValue) return double.NegativeInfinity;
                else if (normalized)
                {
                    double logXOverMode = (xOverMode == 0) ? (Math.Log(x) - Math.Log(mode)) : Math.Log(xOverMode);
                    return (shape - 1) * (logXOverMode + (1 - xOverMode)) + (0.5 + 0.5 / shape) / shape + Math.Log(rate) - 0.5 * Math.Log(shape);
                }
                else if (shape != 1 && rate != 0 && x != 0)
                {
                    return (shape - 1) * (Math.Log(x) - xOverMode);
                }
            }
            double result = 0;
            if (shape != 1) result += (shape - 1) * Math.Log(x);
            if (rate != 0 && x != 0) result -= x * rate;
            if (normalized)
            {
                result += shape * Math.Log(rate) - MMath.GammaLn(shape);
            }
            return result;
        }

        /// <summary>
        /// Logarithm of this Gamma density function.
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <returns>log(Gamma(x;shape,rate))</returns>
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
            else
            {
                return GetLogProb(x, Shape, Rate);
            }
        }

        /// <summary>
        /// Gets log normalizer
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            if (IsProper() && !IsPointMass)
            {
                if (Shape > 1e10)
                {
                    // In double precision, we can assume GammaLn(x) = (x-0.5)*log(x) - x + MMath.LnSqrt2PI for x > 1e10
                    return (Shape - 0.5) * Math.Log(Shape / Rate) - Shape + MMath.LnSqrt2PI;
                }
                return MMath.GammaLn(Shape) - Shape * Math.Log(Rate);
            }
            else
            {
                return 0.0;
            }
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Gamma that)
        {
            if (this.IsPointMass)
            {
                return that.GetLogProb(Point);
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
                // that is not a point mass.
                double result = (that.Shape - 1) * GetMeanLog() - GetMean() * that.Rate;
                result -= that.GetLogNormalizer();
                return result;
            }
        }

        /// <summary>
        /// Asks whether this Gamma instance is proper or not. A Gamma distribution
        /// is proper only if Shape > 0 and Rate > 0.
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return (Shape > 0) && (Rate > 0);
        }

        /// <summary>
        /// Asks whether a Gamma distribution is proper or not. A Gamma distribution
        /// is proper only if Shape > 0 and Rate > 0.
        /// </summary>
        /// <param name="shape">shape parameter for the Gamma</param>
        /// <param name="rate">rate parameter for the Gamma</param>
        /// <returns>True if proper, false otherwise</returns>
        public static bool IsProper(double shape, double rate)
        {
            return (shape > 0) && (rate > 0);
        }

        /// <summary>
        /// The log of the integral of the product of this Gamma and that Gamma
        /// </summary>
        /// <param name="that">That Gamma</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(Gamma that)
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
                // int_x Ga(x;a1,b1) Ga(x;a2,b2) dx
                Gamma product = this * that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper");
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Gamma that, double power)
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
        /// Samples from this Gamma distribution
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
                return Rand.Gamma(Shape) / Rate;
            }
        }

        /// <summary>
        /// Samples from this Gamma distribution
        /// </summary>
        /// <param name="result">Ignored</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample(double result)
        {
            return Sample();
        }

        /// <summary>
        /// Samples from a Gamma distribution with given shape and scale
        /// </summary>
        /// <param name="shape">shape parameter</param>
        /// <param name="scale">scale parameter</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(double shape, double scale)
        {
            return Rand.Gamma(shape) * scale;
        }

        /// <summary>
        /// Samples from a Gamma distribution with given mean and variance
        /// </summary>
        /// <param name="mean">mean parameter</param>
        /// <param name="variance">variance parameter</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "variance")]
        public static double SampleFromMeanAndVariance(double mean, double variance)
        {
            return FromMeanAndVariance(mean, variance).Sample();
        }

        /// <summary>
        /// Sets this Gamma instance to have the parameter values of that Gamma instance
        /// </summary>
        /// <param name="that">That Gamma</param>
        public void SetTo(Gamma that)
        {
            Rate = that.Rate;
            Shape = that.Shape;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two Gammas.
        /// </summary>
        /// <param name="a">The first Gamma</param>
        /// <param name="b">The second Gamma</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(Gamma a, Gamma b)
        {
            if (a.IsPointMass)
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                Point = a.Point;
                return;
            }
            else if (b.IsPointMass)
            {
                Point = b.Point;
                return;
            }
            else
            {
                // avoid roundoff errors when a shape is below eps
                if (a.Shape < b.Shape)
                    Shape = a.Shape + (b.Shape - 1);
                else
                    Shape = b.Shape + (a.Shape - 1);
                Rate = a.Rate + b.Rate;
                if (Shape > double.MaxValue || Rate > double.MaxValue)
                {
                    // (as + bs - 1)/(ar + br) = (am*ar + bm*br - 1)/(ar + br) = am*w + bm*(1-w) - 1/(ar + br)
                    // w = ar/(ar + br)
                    double w = 1 / (1 + b.Rate / a.Rate);
                    Point = a.GetMean() * w + b.GetMean() * (1 - w) - 1 / Rate;
                }
            }
        }

        /// <summary>
        /// Creates a new Gamma which the product of two other Gammas
        /// </summary>
        /// <param name="a">First Gamma</param>
        /// <param name="b">Second Gamma</param>
        /// <returns>Result</returns>
        public static Gamma operator *(Gamma a, Gamma b)
        {
            Gamma result = new Gamma();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two Gammas
        /// </summary>
        /// <param name="numerator">The numerator Gamma.  Can be the same object as this.</param>
        /// <param name="denominator">The denominator Gamma.  Can be the same object as this.</param>
        /// <param name="forceProper">If true, the result has shape >= 1 and rate >= 0, under the constraint that result*denominator has the same mean as numerator</param>
        public void SetToRatio(Gamma numerator, Gamma denominator, bool forceProper = false)
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
                        throw new DivideByZeroException($"numerator {numerator} and denominator {denominator} are different point masses");
                    }
                }
                else
                {
                    Point = numerator.Point;
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException($"denominator {denominator} is a point mass but numerator {numerator} is not");
            }
            else if (forceProper && numerator.Rate == 0)
            {
                Rate = 0;
                Shape = 1 + Math.Max(0, numerator.Shape - denominator.Shape);
            }
            else if (forceProper && (numerator.Shape < denominator.Shape || numerator.Rate < denominator.Rate))
            {
                double mean = numerator.GetMean();
                double shape = mean * denominator.Rate + (1 - denominator.Shape);
                if (shape < 1)
                {
                    Rate = denominator.Shape / mean - denominator.Rate;
                    Shape = 1;
                }
                else
                {
                    Shape = shape;
                    Rate = 0;
                }
            }
            else
            {
                if (Math.Abs(denominator.Shape) > 10)
                {
                    Shape = (numerator.Shape - denominator.Shape) + 1;
                }
                else
                {
                    // avoid roundoff errors when numerator shape is below eps
                    Shape = numerator.Shape + (1 - denominator.Shape);
                }
                Rate = numerator.Rate - denominator.Rate;
            }
        }

        /// <summary>
        /// Creates a new Gamma which the ratio of two other Gammas
        /// </summary>
        /// <param name="numerator">numerator Gamma</param>
        /// <param name="denominator">denominator Gamma</param>
        /// <returns>Result</returns>
        public static Gamma operator /(Gamma numerator, Gamma denominator)
        {
            Gamma result = new Gamma();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source Gamma to some exponent.
        /// </summary>
        /// <param name="dist">The source Gamma</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Gamma dist, double exponent)
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
                if (Math.Abs(exponent) > 1e1)
                {
                    Shape = (dist.Shape - 1) * exponent + 1;
                }
                else
                {
                    Shape = dist.Shape * exponent + (1 - exponent);
                }
                Rate = dist.Rate * exponent;
                if (Shape > double.MaxValue || Rate > double.MaxValue)
                {
                    Point = dist.GetMode();
                }
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Gamma operator ^(Gamma dist, double exponent)
        {
            Gamma result = new Gamma();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Set the mean and variance to match the moments of a mixture of two Gammas.
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first Gamma</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second Gamma</param>
        public void SetToSum(double weight1, Gamma dist1, double weight2, Gamma dist2)
        {
            SetTo(Gaussian.WeightedSum<Gamma>(this, weight1, dist1, weight2, dist2));
        }

        /// <summary>
        /// The maximum difference between the parameters of this Gamma
        /// and that Gamma
        /// </summary>
        /// <param name="thatd">That Gamma</param>
        /// <returns>The maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is Gamma)) return Double.PositiveInfinity;
            Gamma that = (Gamma)thatd;
            return Math.Max(MMath.AbsDiff(Rate, that.Rate, 1e-15), MMath.AbsDiff(Shape, that.Shape, 1e-15));
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// Array of distribution requiring the distribution type to be a value type.
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
            return Hash.Combine(Rate.GetHashCode(), Shape.GetHashCode());
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(Gamma a, Gamma b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(Gamma a, Gamma b)
        {
            return !(a == b);
        }

#if class
        public Gamma()
        {
            SetToUniform();
        }
#endif

        /// <summary>
        /// Creates a Gamma distribution with given shape and scale parameters (scale = 1/rate) 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="scale">scale = 1/rate</param>
        /// <remarks>
        /// The distribution is <c>p(x) = x^(shape-1)*exp(-x/scale)/(scale^shape * Gamma(shape))</c>.
        /// </remarks>
        public Gamma(double shape, double scale)
        {
            Shape = shape;
            Rate = 0;
            SetShapeAndScale(shape, scale);
        }

        /// <summary>
        /// Constructs a Gamma distribution from its natural parameters.
        /// </summary>
        /// <param name="shapeMinus1">shape - 1</param>
        /// <param name="rate">rate = 1/scale</param>
        /// <returns>A new Gamma distribution</returns>
        public static Gamma FromNatural(double shapeMinus1, double rate)
        {
            Gamma result = new Gamma();
            result.SetShapeAndRate(shapeMinus1 + 1, rate);
            return result;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Gamma(Gamma that)
        {
            Shape = that.Shape;
            Rate = that.Rate;
        }

        /// <summary>
        /// Clones this Gamma. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Gamma type</returns>
        public object Clone()
        {
            return new Gamma(this);
        }

        /// <summary>
        /// Create a uniform Gamma distribution.
        /// </summary>
        /// <returns>A new uniform Gamma distribution</returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static Gamma Uniform()
        {
            Gamma result = new Gamma();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a point mass Gamma distribution
        /// </summary>
        /// <param name="mean">The location of the point mass</param>
        /// <returns>A new point mass Gamma distribution</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static Gamma PointMass(double mean)
        {
            Gamma result = new Gamma();
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
            if (IsPointMass)
            {
                return "Gamma.PointMass(" + Point.ToString("g4") + ")";
            }
            else if (IsUniform())
            {
                return "Gamma.Uniform";
            }
            else
            {
                GetShapeAndScale(out double shape, out double scale);
                string s = "Gamma(" + shape.ToString("g4") + ", " + scale.ToString("g4") + ")";
                if (IsProper())
                {
                    s += "[mean=" + GetMean().ToString("g4") + "]";
                }
                return s;
            }
        }
    }
}