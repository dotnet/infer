// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Left-truncated Com-Poisson distribution.
    /// </summary>
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class LeftTruncatedPoisson : IDistribution<int>,
                                        CanGetMean<double>,
                                        Sampleable<int>,
                                        SettableToProduct<LeftTruncatedPoisson>,
                                        SettableToRatio<LeftTruncatedPoisson>,
                                        SettableToPower<LeftTruncatedPoisson>,
                                        SettableToWeightedSum<LeftTruncatedPoisson>,
                                        SettableTo<LeftTruncatedPoisson>,
                                        SettableTo<Discrete>,
                                        SettableToPartialUniform<LeftTruncatedPoisson>,
                                        SettableToPartialUniform<Discrete>,
                                        CanGetLogAverageOf<LeftTruncatedPoisson>,
                                        CanGetAverageLog<LeftTruncatedPoisson>,
                                        CanGetLogNormalizer,
                                        ICanTruncateLeft<int>
    {
        /// <summary>
        /// Tolerance for numerical calculations.
        /// </summary>
        public const double Tolerance = 1e-10;

        /// <summary>
        /// The non-truncated distribution.
        /// </summary>
        [DataMember]
        private Poisson nonTruncatedDistribution;

        private Lazy<double> logNormalizer;
        private Lazy<double> meanLogFactorial;
        private Lazy<double> sumLogFactorial;

        /// <summary>
        /// Initializes a new instance of the <see cref="LeftTruncatedPoisson" /> class.
        /// </summary>
        /// <param name="rate">The rate parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="precision">The precision parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        [Construction("Rate", "Precision", "StartPoint")]
        public LeftTruncatedPoisson(double rate, double precision, int startPoint)
            : this(new Poisson(rate, precision), startPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LeftTruncatedPoisson" /> class.
        /// </summary>
        /// <param name="rate">
        /// The rate parameter of the underlying Poisson distribution. The precision parameter is assumed to be
        /// 1.0.
        /// </param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        public LeftTruncatedPoisson(double rate, int startPoint)
            : this(new Poisson(rate, 1.0), startPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LeftTruncatedPoisson" /> class.
        /// </summary>
        /// <param name="nonTruncatedDistribution">The non-truncated CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        public LeftTruncatedPoisson(Poisson nonTruncatedDistribution, int startPoint)
        {
            this.nonTruncatedDistribution = nonTruncatedDistribution;
            this.StartPoint = nonTruncatedDistribution.IsPointMass ? nonTruncatedDistribution.Point : startPoint;
            this.ResetCache();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LeftTruncatedPoisson" /> class.
        /// </summary>
        public LeftTruncatedPoisson()
        {
            this.SetToUniform();
        }

        /// <summary>
        /// The first non-zero point.
        /// </summary>
        [DataMember]
        public int StartPoint { get; private set; }

        /// <summary>
        /// Gets or sets the rate parameter of the COM-Poisson distribution, always >= 0.
        /// </summary>
        public double Rate
        {
            get => this.nonTruncatedDistribution.Rate;

            set
            {
                this.nonTruncatedDistribution.Rate = value;
                this.ResetCache();
            }
        }

        /// <summary>
        /// Gets or sets the precision parameter of the COM-Poisson distribution
        /// </summary>
        public double Precision
        {
            get => this.nonTruncatedDistribution.Precision;

            set
            {
                this.nonTruncatedDistribution.Precision = value;
                this.ResetCache();
            }
        }

        /// <summary>
        /// Returns the log normalizer of the instance.
        /// </summary>
        public double LogNormalizer => this.logNormalizer.Value;

        /// <summary>
        /// Returns the mean log factorial of the instance.
        /// </summary>
        public double MeanLogFactorial => this.meanLogFactorial.Value;

        /// <summary>
        /// Returns the log normalizer of the instance.
        /// </summary>
        public double SumLogFactorial => this.sumLogFactorial.Value;

        /// <summary>
        /// Gets a value indicating whether the current distribution represents a point mass.
        /// </summary>
        public bool IsPointMass => this.nonTruncatedDistribution.IsPointMass;

        /// <summary>
        /// Gets or sets the distribution as a point mass.
        /// </summary>
        public int Point
        {
            get => Math.Max(this.nonTruncatedDistribution.Point, this.StartPoint);

            set
            {
                this.nonTruncatedDistribution.Point = value;
                this.StartPoint = value;
                this.ResetCache();
            }
        }

        /// <summary>
        /// Creates a new truncated COM-Poisson which is the product of two other truncated COM-Poissons
        /// </summary>
        /// <param name="a">The first COM-Poisson</param>
        /// <param name="b">The second COM-Poisson</param>
        /// <returns>Result of the product.</returns>
        public static LeftTruncatedPoisson operator *(LeftTruncatedPoisson a, LeftTruncatedPoisson b)
        {
            var result = new LeftTruncatedPoisson();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Creates a new truncated COM-Poisson which is the ratio of two other truncated COM-Poissons
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <returns>Result of the ratio calculation.</returns>
        public static LeftTruncatedPoisson operator /(LeftTruncatedPoisson numerator, LeftTruncatedPoisson denominator)
        {
            var result = new LeftTruncatedPoisson();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Raises a truncated COM-Poisson to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist" /> raised to power <paramref name="exponent" />.</returns>
        public static LeftTruncatedPoisson operator ^(LeftTruncatedPoisson dist, double exponent)
        {
            var result = new LeftTruncatedPoisson();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Gets the log normalizer of a left-truncated Poisson.
        /// TODO: integrate this with Poisson.GetLogNormalizer.
        /// </summary>
        /// <param name="rate">Rate parameter.</param>
        /// <param name="prec">Precision parameter.</param>
        /// <param name="startPoint">Start point.</param>
        /// <returns>The log normalizer of the distribution.</returns>
        public static double GetLogNormalizer(double rate, double prec, int startPoint)
        {
            if (startPoint == 0)
            {
                return Poisson.GetLogNormalizer(rate, prec);
            }
            else if (prec == 0.0)
            {
                if (rate >= 1)
                {
                    return 0.0;
                }

                return (startPoint * Math.Log(rate)) - Math.Log(1.0 - rate);
            }
            else
            {
                // The following commented recursion becomes numerically unstable very quickly.
                // double lhs = GetLogNormalizer(rate, prec, startPoint - 1);
                // double rhs = (startPoint - 1) * Math.Log(rate) - prec * MMath.GammaLn(startPoint);
                // return MMath.LogDifferenceOfExp(lhs, rhs);
                // So do the infinite sum instead.
                const int MaxIterations = 1000;
                var logRate = Math.Log(rate);
                var maxIndex = MaxIterations + startPoint;

                // First term
                var logTerm = (startPoint * logRate) - (prec * MMath.GammaLn(startPoint + 1));
                var result = logTerm;
                int i;
                for (i = startPoint + 1; i < maxIndex; i++)
                {
                    var deltaLogTerm = logRate - (prec * Math.Log(i));
                    logTerm += deltaLogTerm;
                    result = MMath.LogSumExp(result, logTerm);
                    if (deltaLogTerm < 0)
                    {
                        // Truncation error.
                        // The rest of the series is sum_{i=0..infinity} exp(term + delta(i)*i)
                        // because delta is decreasing, we can bound the sum of the rest of the series
                        // by sum_{i=0..infinity} exp(term + delta*i) = exp(term)/(1 - exp(delta))
                        var r = Math.Exp(logTerm - result) / (1 - Math.Exp(deltaLogTerm));

                        // Truncation error in log domain: log(z+r)-log(z) = log(1 + r/z) =approx r/
                        if (r < Tolerance)
                        {
                            break;
                        }
                    }
                }

                if (i == maxIndex)
                {
                    throw new ArithmeticException("Unable to calculate log normalizer");
                }

                return result;
            }
        }

        /// <summary>
        /// Gets the log probability when some calculations have been done externally.
        /// </summary>
        /// <param name="value">The domain value.</param>
        /// <param name="logRate">The log of the rate.</param>
        /// <param name="precision">The precision.</param>
        /// <param name="logNormalizer">The log normalizer.</param>
        /// <returns>The log probability of the value.</returns>
        public static double GetLogProb(int value, double logRate, double precision, double logNormalizer) =>
            (value * logRate) - (precision * MMath.GammaLn(value + 1)) - logNormalizer;

        /// <summary>
        /// Initializes a new instance of the <see cref="LeftTruncatedPoisson" /> struct given a target
        /// mean and a start point.
        /// </summary>
        /// <param name="targetMean">The desired mean of the Truncated Poisson.</param>
        /// <param name="precision">The desired precision of the Truncated Poisson.</param>
        /// <param name="startPoint">The start point.</param>
        /// <param name="tolerance">Tolerance for mean.</param>
        /// <param name="maxIterations">Maximum iterations for binary search.</param>
        /// <returns>A new instance with the target mean.</returns>
        public static LeftTruncatedPoisson FromMeanAndStartPoint(
            double targetMean,
            double precision,
            int startPoint,
            double tolerance = Tolerance,
            int maxIterations = 100)
        {
            if (targetMean < startPoint)
            {
                throw new ArgumentException("Truncated Poisson: desired mean cannot be less than start point (M: {targetMean}, S: {startPoint})");
            }

            if (startPoint == 0)
            {
                return new LeftTruncatedPoisson(targetMean, startPoint);
            }

            if (precision == 0)
            {
                // Exact calculation.
                var numer = targetMean - startPoint;
                var denom = numer + 1.0;
                return new LeftTruncatedPoisson(numer / denom, 0.0, startPoint);
            }

            if (targetMean - startPoint < double.Epsilon)
            {
                return PointMass(startPoint);
            }

            // Avoid numerical issues
            if (targetMean < startPoint + tolerance)
            {
                targetMean = startPoint + tolerance;
            }

            // Do a binary search.
            var rateUpperBound = targetMean;
            var rateLowerBound = targetMean - startPoint;

            var current = new LeftTruncatedPoisson(targetMean, startPoint);

            var oldMean = double.MaxValue;
            for (var i = 0; i < maxIterations; i++)
            {
                var rateCurrent = 0.5 * (rateLowerBound + rateUpperBound);
                current = new LeftTruncatedPoisson(rateCurrent, precision, startPoint);
                var currentMean = current.GetMean();

                if (Math.Abs(currentMean - oldMean) < tolerance || Math.Abs(currentMean - targetMean) < tolerance)
                {
                    break;
                }

                if (currentMean < targetMean)
                {
                    rateLowerBound = rateCurrent;
                }
                else
                {
                    rateUpperBound = rateCurrent;
                }
            }

            return current;
        }

        /// <summary>
        /// Instantiates a uniform truncated Com-Poisson distribution
        /// </summary>
        /// <returns>A new uniform Com-Poisson distribution</returns>
        [Construction(UseWhen = "IsUniform")]
        [Skip]
        public static LeftTruncatedPoisson Uniform()
        {
            var result = new LeftTruncatedPoisson();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Instantiates a uniform truncated Com-Poisson distribution beyond a given start point.
        /// </summary>
        /// <param name="startPoint">Start point.</param>
        /// <returns>A new uniform Com-Poisson distribution</returns>
        public static LeftTruncatedPoisson TruncatedUniform(int startPoint)
        {
            var result = new LeftTruncatedPoisson();
            result.SetToPartialUniform(startPoint);
            result.ResetCache();
            return result;
        }

        /// <summary>
        /// Creates a truncated Com-Poisson distribution which only allows one value.
        /// </summary>
        /// <param name="value">The value of the point mass.</param>
        /// <returns>A point mass at the given value.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static LeftTruncatedPoisson PointMass(int value)
        {
            var result = new LeftTruncatedPoisson
            {
                Point = value,
                StartPoint = value
            };

            result.ResetCache();
            return result;
        }

        /// <summary>
        /// Computes sum_{x=start..infinity} log(x!) lambda^x / x!^nu
        /// </summary>
        /// <param name="lambda">The rate.</param>
        /// <param name="nu">The precision.</param>
        /// <param name="startPoint">The start point.</param>
        /// <returns>The sum log factorial.</returns>
        public static double GetSumLogFactorial(double lambda, double nu, int startPoint)
        {
            if (startPoint == 0)
            {
                return Poisson.GetSumLogFactorial(lambda, nu);
            }
            else if (double.IsPositiveInfinity(nu))
            {
                throw new ArgumentException("nu = Inf");
            }
            else if (lambda < 0)
            {
                throw new ArgumentException("lambda (" + lambda + ") < 0");
            }
            else if (nu < 0)
            {
                throw new ArgumentException("nu (" + nu + ") < 0");
            }
            else if (lambda == 0)
            {
                return 0.0;
            }

            const int MaxIters = 10000;
            var term2 = MMath.GammaLn(startPoint + 1);
            var term = Math.Exp((startPoint * Math.Log(lambda)) - (nu * term2));
            var result = term2 * term;
            int i;
            for (i = startPoint + 1; i < MaxIters; i++)
            {
                var oldterm2 = term2;
                term2 += Math.Log(i);
                var delta = lambda / Math.Pow(i, nu);
                term *= delta;
                result += term * term2;
                delta *= term2 / oldterm2; // delta = term*term2 / (oldterm*oldterm2)
                if ((i > 2) && (delta < 1))
                {
                    // delta is decreasing, so remainder of series can be bounded by
                    // sum_{i=0..infinity} term*term2*delta^i = term*term2/(1-delta)
                    var r = (term * term2) / result / (1 - delta);
                    if (r < Tolerance)
                    {
                        break;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Creates a cloned copy of this distribution.
        /// </summary>
        /// <returns>The copy.</returns>
        public object Clone() => new LeftTruncatedPoisson(this.Rate, this.Precision, this.StartPoint);

        /// <summary>
        /// Gets a value indicating how close this distribution is to another distribution.
        /// in terms of probabilities they assign to sequences.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>The max difference.</returns>
        public double MaxDiff(object that)
        {
            if (!(that is LeftTruncatedPoisson))
            {
                return double.PositiveInfinity;
            }

            var thatd = (LeftTruncatedPoisson)that;

            return Math.Max(this.nonTruncatedDistribution.MaxDiff(thatd.nonTruncatedDistribution), (double)Math.Abs(this.StartPoint - thatd.StartPoint));
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution is uniform.
        /// </summary>
        public bool IsUniform() => this.IsPartialUniform() && (this.StartPoint == 0);

        /// <summary>
        /// Asks whether this instance is proper or not. A truncated COM-Poisson distribution
        /// is proper if Rate >= 0 and (Precision > 0 or (Precision == 0 and Rate &lt; 1)).
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper() => this.nonTruncatedDistribution.IsProper();

        /// <summary>
        /// Sets the current distribution to be uniform.
        /// </summary>
        public void SetToUniform()
        {
            this.nonTruncatedDistribution.SetToUniform();
            this.StartPoint = 0;
            this.ResetCache();
        }

        /// <summary>
        /// Sets the current distribution to be uniform beyond a given start point.
        /// </summary>
        /// <param name="startPoint">The start point.</param>
        public void SetToPartialUniform(int startPoint)
        {
            this.nonTruncatedDistribution.SetToUniform();
            this.StartPoint = startPoint;
            this.ResetCache();
        }

        /// <summary>
        /// Sets the parameters to represent the product of two truncated COM-Poissons.
        /// </summary>
        /// <param name="a">The first truncated COM-Poisson</param>
        /// <param name="b">The second truncated COM-Poisson</param>
        public void SetToProduct(LeftTruncatedPoisson a, LeftTruncatedPoisson b)
        {
            this.nonTruncatedDistribution = a.nonTruncatedDistribution * b.nonTruncatedDistribution;
            this.StartPoint = Math.Max(a.StartPoint, b.StartPoint);
            this.ResetCache();
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two truncated COM-Poisson distributions
        /// </summary>
        /// <param name="numerator">The numerator truncated distribution</param>
        /// <param name="denominator">The denominator truncated distribution</param>
        /// <param name="forceProper">If true, the result has precision >= 0 and rate &lt;= 1</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToRatio(LeftTruncatedPoisson numerator, LeftTruncatedPoisson denominator, bool forceProper = false)
        {
            this.nonTruncatedDistribution = numerator.nonTruncatedDistribution / denominator.nonTruncatedDistribution;
            this.StartPoint = Math.Max(numerator.StartPoint, denominator.StartPoint);
            this.ResetCache();
        }

        /// <summary>
        /// Sets the parameters to represent the power of a truncated COM-Poisson to some exponent.
        /// </summary>
        /// <param name="dist">The source truncated COM-Poisson</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(LeftTruncatedPoisson dist, double exponent)
        {
            this.nonTruncatedDistribution = dist.nonTruncatedDistribution ^ exponent;
            this.StartPoint = dist.StartPoint;
            this.ResetCache();
        }

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second distribution</param>
        public void SetToSum(double weight1, LeftTruncatedPoisson dist1, double weight2, LeftTruncatedPoisson dist2)
        {
            var nonTrunc = default(Poisson);
            nonTrunc.SetToSum(weight1, dist1.nonTruncatedDistribution, weight2, dist2.nonTruncatedDistribution);
            this.nonTruncatedDistribution = nonTrunc;
            this.StartPoint = Math.Max(dist1.StartPoint, dist2.StartPoint);
            this.ResetCache();
        }

        /// <summary>
        /// Gets the log of the integral of the product of this truncated COM-Poisson with another truncated COM-Poisson.
        /// </summary>
        /// <param name="that">The other truncated COM-Poisson</param>
        public double GetLogAverageOf(LeftTruncatedPoisson that)
        {
            if (this.IsPointMass)
            {
                return that.GetLogProb(this.Point);
            }
            else if (that.IsPointMass)
            {
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = new LeftTruncatedPoisson();
                product.SetToProduct(this, that);

                return (this.GetLogNormalizer() + that.GetLogNormalizer()) - product.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Gets the expected logarithm of a truncated COM-Poisson under this truncated COM-Poisson.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns>
        ///     <c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c>
        /// </returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(LeftTruncatedPoisson that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && (this.Point == that.Point))
                {
                    return 0.0;
                }
                else
                {
                    return double.NegativeInfinity;
                }
            }
            else if (!this.IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else if (this.StartPoint < that.StartPoint)
            {
                return double.NegativeInfinity;
            }
            else
            {
                return (this.GetMean() * Math.Log(that.Rate)) - (that.Precision * this.GetMeanLogFactorial()) - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Gets the log of the normalizer of this distribution.
        /// </summary>
        /// <returns>The log of the normalizer.</returns>
        public double GetLogNormalizer()
        {
            return this.LogNormalizer;
        }

        /// <summary>
        /// Gets the log probability for a given domain value.
        /// </summary>
        /// <param name="value">The domain value.</param>
        /// <returns>The log probability of the value.</returns>
        public double GetLogProb(int value)
        {
            if (value < this.StartPoint)
            {
                return double.NegativeInfinity;
            }
            else if (this.IsPointMass)
            {
                return value == this.Point ? 0.0 : double.NegativeInfinity;
            }
            else if (this.IsPartialUniform())
            {
                return 0.0;
            }
            else
            {
                return GetLogProb(value, Math.Log(this.Rate), this.Precision, this.GetLogNormalizer());
            }
        }

        /// <summary>
        /// Emits a random sample from this distribution.
        /// </summary>
        /// <returns>A random sample.</returns>
        [Stochastic]
        public int Sample()
        {
            if (this.IsPointMass)
            {
                return this.Point;
            }

            // This is an inefficient rejection sampler which will not scale as
            // the start point gets large compared with the mean. If this is to be
            // used in anger we should revisit this.
            int result;
            do
            {
                result = Poisson.Sample(this.Rate, this.Precision);
            }
            while (result < this.StartPoint);

            return result;
        }

        /// <summary>
        /// Emits a random sample from this distribution.
        /// </summary>
        /// <param name="result">This argument is ignored.</param>
        /// <returns>A random sample.</returns>
        [Stochastic]
        public int Sample(int result) => this.Sample();

        /// <summary>
        /// Gets the mean of this distribution.
        /// </summary>
        /// <returns>The mean.</returns>
        public double GetMean()
        {
            if (this.Precision == 1.0)
            {
                if (this.StartPoint == 0)
                {
                    return this.nonTruncatedDistribution.GetMean();
                }
                else if (this.IsPointMass)
                {
                    return this.Point;
                }
                else
                {
                    var logMean = (Math.Log(this.Rate) + GetLogNormalizer(this.Rate, 1.0, this.StartPoint - 1)) - GetLogNormalizer(this.Rate, 1.0, this.StartPoint);

                    return Math.Exp(logMean);
                }
            }
            else if ((this.Precision == 0.0) && (this.Rate < 1.0))
            {
                if (this.StartPoint == 0)
                {
                    return this.nonTruncatedDistribution.GetMean();
                }
                else
                {
                    return ((this.Rate + this.StartPoint) - (this.Rate * this.StartPoint)) / (1.0 - this.Rate);
                }
            }
            else
            {
                throw new NotImplementedException("Calculation for mean is only available for Precision = 1, or Precision = 0 and Rate < 1");
            }
        }

        /// <inheritdoc />
        public void SetTo(LeftTruncatedPoisson value)
        {
            this.nonTruncatedDistribution.SetTo(value.nonTruncatedDistribution);
            this.StartPoint = value.StartPoint;
            this.ResetCache();
        }

        /// <inheritdoc />
        public void SetTo(Discrete value)
        {
            var startPoint = value.GetLogProbs().FindFirstIndex(prob => !double.IsNegativeInfinity(prob));
            var mean = value.GetMean();
            const double Delta = 0.01;
            if (mean - startPoint < Delta)
            {
                mean = startPoint + Delta;
            }

            var projection = FromMeanAndStartPoint(mean, 1.0, startPoint);
            this.SetTo(projection);
        }

        /// <summary>
        /// Overrides ToString method.
        /// </summary>
        /// <returns>String representation of instance.</returns>
        public override string ToString()
        {
            if (this.IsPointMass)
            {
                return "TruncPois.PointMass(" + this.Point + ")";
            }
            else if (this.IsPartialUniform())
            {
                if (this.StartPoint > 0)
                {
                    return "TruncPois.Uniform(" + this.StartPoint + ")";
                }
                else
                {
                    return "TruncPois.Uniform";
                }
            }
            else if (this.Precision == 1.0)
            {
                return string.Format("TruncPois:Rate={0:0.000},Start={1}(mean={2:0.00})", this.Rate, this.StartPoint, this.GetMean());
            }
            else
            {
                return string.Format("TruncPois:Rate={0:0.000},Prec={1:0.00},Start={2}", this.Rate, this.Precision, this.StartPoint);
            }
        }

        /// <summary>
        /// Set non-zero support to uniform
        /// </summary>
        public void SetToPartialUniform()
        {
            this.nonTruncatedDistribution.SetToUniform();
        }

        /// <inheritdoc />
        public void SetToPartialUniformOf(LeftTruncatedPoisson dist)
        {
            this.SetToPartialUniform();
            this.StartPoint = dist.StartPoint;
            this.ResetCache();
        }

        /// <inheritdoc />
        public void SetToPartialUniformOf(Discrete dist)
        {
            this.SetToPartialUniform();
            this.StartPoint = dist.GetLogProbs().FindFirstIndex(prob => !double.IsNegativeInfinity(prob));
            this.ResetCache();
        }

        /// <inheritdoc />
        public bool IsPartialUniform() => this.nonTruncatedDistribution.IsUniform();

        /// <inheritdoc />
        public int GetStartPoint()
        {
            return this.StartPoint;
        }

        /// <inheritdoc />
        public void TruncateLeft(int startPoint)
        {
            this.StartPoint = startPoint;
            this.ResetCache();
        }

        /// <summary>
        /// Resets the cache for this instance.
        /// </summary>
        internal void ResetCache()
        {
            this.logNormalizer = new Lazy<double>(() => GetLogNormalizer(this.Rate, this.Precision, this.StartPoint));
            this.meanLogFactorial = new Lazy<double>(this.GetMeanLogFactorial);
            this.sumLogFactorial = new Lazy<double>(() => GetSumLogFactorial(this.Rate, this.Precision, this.StartPoint));
        }

        /// <summary>
        /// Computes (sum_{x=start..infinity} log(x!) Rate^x / x!^Precision) / (sum_{x=start..infinity} Rate^x / x!^Precision )
        /// </summary>
        /// <returns>The mean log factorial.</returns>
        private double GetMeanLogFactorial()
        {
            if (this.IsPointMass)
            {
                return MMath.GammaLn(this.Point + 1);
            }
            else if (!this.IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                return this.SumLogFactorial / Math.Exp(this.LogNormalizer);
            }
        }
    }
}