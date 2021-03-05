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
    /// Left-and-right-truncated Com-Poisson distribution.
    /// </summary>
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class TruncatedPoisson : IDistribution<int>,
                                     CanGetMean<double>,
                                     Sampleable<int>,
                                     SettableToProduct<TruncatedPoisson>,
                                     SettableTo<TruncatedPoisson>,
                                     SettableTo<LeftTruncatedPoisson>,
                                     SettableTo<Discrete>,
                                     SettableToPartialUniform<TruncatedPoisson>,
                                     SettableToPartialUniform<Discrete>,
                                     CanGetLogAverageOf<TruncatedPoisson>,
                                     CanGetAverageLog<TruncatedPoisson>,
                                     CanGetLogNormalizer,
                                     ICanTruncateLeft<int>,
                                     ICanTruncateRight<int>
    {
        public const int PositiveInfinityEndPoint = int.MaxValue;

        /// <summary>
        /// The left-truncated distribution.
        /// </summary>
        [DataMember]
        private LeftTruncatedPoisson leftTruncatedDistribution;

        private Lazy<double> logNormalizer;

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="rate">The rate parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        public TruncatedPoisson(double rate, int startPoint)
            : this(rate, 1.0, startPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="rate">The rate parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="precision">The precision parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        public TruncatedPoisson(double rate, double precision, int startPoint)
            : this(new LeftTruncatedPoisson(rate, precision, startPoint), PositiveInfinityEndPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="rate">The rate parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="precision">The precision parameter of the underlying CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        /// <param name="endPoint">The index of the last non-zero term in the truncated distribution.</param>
        [Construction("Rate", "Precision", "StartPoint", "EndPoint")]
        public TruncatedPoisson(double rate, double precision, int startPoint, int endPoint)
            : this(new LeftTruncatedPoisson(rate, precision, startPoint), endPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="nonTruncatedDistribution">The underlying CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        public TruncatedPoisson(Poisson nonTruncatedDistribution, int startPoint)
            : this(new LeftTruncatedPoisson(nonTruncatedDistribution, startPoint))
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="truncatedDistribution">The underlying left-truncated CoM-Poisson distribution.</param>
        public TruncatedPoisson(LeftTruncatedPoisson truncatedDistribution)
            : this(truncatedDistribution, PositiveInfinityEndPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="nonTruncatedDistribution">The underlying CoM-Poisson distribution.</param>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        /// <param name="endPoint">The index of the last non-zero term in the truncated distribution.</param>
        public TruncatedPoisson(Poisson nonTruncatedDistribution, int startPoint, int endPoint)
            : this(new LeftTruncatedPoisson(nonTruncatedDistribution, startPoint), endPoint)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        /// <param name="leftTruncatedDistribution">The underlying left-truncated CoM-Poisson distribution.</param>
        /// <param name="endPoint">The index of the last non-zero term in the truncated distribution.</param>
        public TruncatedPoisson(LeftTruncatedPoisson leftTruncatedDistribution, int endPoint)
        {
            this.leftTruncatedDistribution = leftTruncatedDistribution;
            this.EndPoint = leftTruncatedDistribution.IsPointMass ? leftTruncatedDistribution.Point : endPoint;
            this.ResetCache();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> class.
        /// </summary>
        public TruncatedPoisson()
        {
            this.leftTruncatedDistribution = new LeftTruncatedPoisson();
            this.SetToUniform();
        }

        /// <summary>
        /// The last non-zero point.
        /// </summary>
        [DataMember]
        public int EndPoint { get; private set; }

        /// <summary>
        /// Gets or sets the rate parameter of the COM-Poisson distribution, always >= 0.
        /// </summary>
        public double Rate
        {
            get => this.leftTruncatedDistribution.Rate;

            set
            {
                this.leftTruncatedDistribution.Rate = value;
                this.ResetCache();
            }
        }

        /// <summary>
        /// Gets or sets the precision parameter of the COM-Poisson distribution
        /// </summary>
        public double Precision
        {
            get => this.leftTruncatedDistribution.Precision;

            set
            {
                this.leftTruncatedDistribution.Precision = value;
                this.ResetCache();
            }
        }

        /// <summary>
        /// The first non-zero point.
        /// </summary>
        public int StartPoint => this.leftTruncatedDistribution.StartPoint;

        /// <summary>
        /// Returns the log normalizer of the instance.
        /// </summary>
        public double LogNormalizer => this.logNormalizer.Value;

        /// <summary>
        /// Gets a value indicating whether the current distribution represents a point mass.
        /// </summary>
        public bool IsPointMass => this.leftTruncatedDistribution.IsPointMass || (this.StartPoint == this.EndPoint);

        /// <summary>
        /// Gets or sets the distribution as a point mass.
        /// </summary>
        public int Point
        {
            get => Math.Min(this.leftTruncatedDistribution.Point, this.EndPoint);

            set
            {
                this.leftTruncatedDistribution.Point = value;
                this.EndPoint = value;
                this.ResetCache();
            }
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution is right-truncated.
        /// </summary>
        public bool EndPointIsPositiveInfinity => this.EndPoint == PositiveInfinityEndPoint;

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> struct given a target
        /// mean, a start point and a end point.
        /// </summary>
        /// <param name="targetMean">The desired mean of the Truncated Poisson.</param>
        /// <param name="precision">The desired precision of the Truncated Poisson.</param>
        /// <param name="startPoint">The start point.</param>
        /// <param name="tolerance">Tolerance for mean.</param>
        /// <param name="maxIterations">Maximum iterations for binary search.</param>
        /// <returns>A new instance with the target mean.</returns>
        public static TruncatedPoisson FromMeanAndStartPoint(
            double targetMean,
            double precision,
            int startPoint,
            double tolerance = LeftTruncatedPoisson.Tolerance,
            int maxIterations = 100) =>
            FromMeanAndStartPointAndEndPoint(targetMean, precision, startPoint, PositiveInfinityEndPoint, tolerance, maxIterations);

        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedPoisson" /> struct given a target
        /// mean, a start point and a end point.
        /// </summary>
        /// <param name="targetMean">The desired mean of the Truncated Poisson.</param>
        /// <param name="precision">The desired precision of the Truncated Poisson.</param>
        /// <param name="startPoint">The start point.</param>
        /// <param name="endPoint">The end point.</param>
        /// <param name="tolerance">Tolerance for mean.</param>
        /// <param name="maxIterations">Maximum iterations for binary search.</param>
        /// <returns>A new instance with the target mean.</returns>
        public static TruncatedPoisson FromMeanAndStartPointAndEndPoint(
            double targetMean,
            double precision,
            int startPoint,
            int endPoint,
            double tolerance = LeftTruncatedPoisson.Tolerance,
            int maxIterations = 100)
        {
            if (targetMean < startPoint)
            {
                throw new ArgumentException(
                    $"Doubly Truncated Poisson: desired mean cannot be less than start point (M: {targetMean}, S: {startPoint}, E: {endPoint})");
            }

            if (targetMean > endPoint)
            {
                throw new ArgumentException($"Doubly Truncated Poisson: desired mean cannot be more than end point (M: {targetMean}, S: {startPoint}, E: {endPoint})");
            }

            if (endPoint == PositiveInfinityEndPoint)
            {
                var truncatedDistribution = LeftTruncatedPoisson.FromMeanAndStartPoint(targetMean, precision, startPoint, tolerance, maxIterations);
                return new TruncatedPoisson(truncatedDistribution, endPoint);
            }

            if (startPoint == endPoint || targetMean - startPoint < double.Epsilon)
            {
                return PointMass(startPoint);
            }

            // Avoid numerical issues
            if (targetMean < startPoint + tolerance)
            {
                targetMean = startPoint + tolerance;
            }

            // Do a binary search.
            double rateUpperBound;
            double rateLowerBound;
            TruncatedPoisson current;
            if (precision == 0)
            {
                rateUpperBound = 1.0;
                rateLowerBound = 0.0;
                current = new TruncatedPoisson(0.5, precision, startPoint, endPoint);
            }
            else
            {
                rateUpperBound = targetMean;
                rateLowerBound = targetMean - startPoint;
                current = new TruncatedPoisson(targetMean, 1.0, startPoint, endPoint);
            }

            var oldMean = double.MaxValue;
            for (var i = 0; i < maxIterations; i++)
            {
                var rateCurrent = 0.5 * (rateLowerBound + rateUpperBound);
                current = new TruncatedPoisson(rateCurrent, precision, startPoint, endPoint);
                var currentMean = current.GetMean();

                if (Math.Abs(currentMean - oldMean) < tolerance || Math.Abs(currentMean - targetMean) < tolerance)
                {
                    break;
                }

                oldMean = currentMean;

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
        /// Gets the log normalizer of a truncated Poisson.
        /// </summary>
        /// <param name="rate">Rate parameter.</param>
        /// <param name="prec">Precision parameter.</param>
        /// <param name="startPoint">Start point.</param>
        /// <param name="endPoint">End point.</param>
        /// <returns>The log normalizer of the distribution.</returns>
        public static double GetLogNormalizer(double rate, double prec, int startPoint, int endPoint)
        {
            if ((prec == 0.0) && (rate >= 1))
            {
                return 0.0;
            }

            return MMath.LogDifferenceOfExp(
                LeftTruncatedPoisson.GetLogNormalizer(rate, prec, startPoint),
                LeftTruncatedPoisson.GetLogNormalizer(rate, prec, endPoint + 1));
        }

        /// <summary>
        /// Creates a truncated Com-Poisson distribution which only allows one value.
        /// </summary>
        /// <param name="value">The value of the point mass.</param>
        /// <returns>A point mass at the given value.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static TruncatedPoisson PointMass(int value)
        {
            var leftTruncatedDistribution = LeftTruncatedPoisson.PointMass(value);
            var result = new TruncatedPoisson(leftTruncatedDistribution, value);
            return result;
        }

        /// <summary>
        /// Instantiates a uniform truncated Com-Poisson distribution beyond a given start point.
        /// </summary>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        /// <returns>A new uniform truncated Com-Poisson distribution</returns>
        public static TruncatedPoisson PartialUniform(int startPoint) => PartialUniform(startPoint, PositiveInfinityEndPoint);

        /// <summary>
        /// Instantiates a uniform truncated Com-Poisson distribution between given start and end points.
        /// </summary>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        /// <param name="endPoint">The index of the last non-zero term in the truncated distribution.</param>
        /// <returns>A new uniform truncated Com-Poisson distribution</returns>
        public static TruncatedPoisson PartialUniform(int startPoint, int endPoint)
        {
            var result = new TruncatedPoisson();
            result.SetToPartialUniform(startPoint, endPoint);
            return result;
        }

        /// <summary>
        /// Instantiates a uniform truncated Com-Poisson distribution
        /// </summary>
        /// <returns>A new uniform Com-Poisson distribution</returns>
        [Construction(UseWhen = "IsUniform")]
        [Skip]
        public static TruncatedPoisson Uniform()
        {
            var result = new TruncatedPoisson();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a cloned copy of this distribution.
        /// </summary>
        /// <returns>The copy.</returns>
        public object Clone() => new TruncatedPoisson(this.Rate, this.Precision, this.StartPoint, this.EndPoint);

        /// <summary>
        /// Gets the log of the integral of the product of this truncated COM-Poisson with another truncated COM-Poisson.
        /// </summary>
        /// <param name="that">The other truncated COM-Poisson</param>
        public double GetLogAverageOf(TruncatedPoisson that)
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
                var product = new TruncatedPoisson();
                product.SetToProduct(this, that);

                return (this.LogNormalizer + that.LogNormalizer) - product.LogNormalizer;
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
            if (this.EndPointIsPositiveInfinity)
            {
                return this.leftTruncatedDistribution.GetLogProb(value);
            }
            else if ((value > this.EndPoint) || (value < this.StartPoint))
            {
                return double.NegativeInfinity;
            }
            else if (this.IsPointMass)
            {
                return value == this.Point ? 0.0 : double.NegativeInfinity;
            }
            else
            {
                return LeftTruncatedPoisson.GetLogProb(value, Math.Log(this.Rate), this.Precision, this.GetLogNormalizer());
            }
        }

        /// <summary>
        /// Gets the mean of this distribution.
        /// </summary>
        /// <returns>The mean.</returns>
        public double GetMean()
        {
            if (this.EndPointIsPositiveInfinity)
            {
                return this.leftTruncatedDistribution.GetMean();
            }

            if (this.IsPointMass)
            {
                return this.Point;
            }

            if ((this.Precision == 0.0) && (this.Rate < 1))
            {
                var rateToPower = Math.Pow(this.Rate, (this.EndPoint + 1) - this.StartPoint);
                var numer = this.StartPoint - 1 - (this.EndPoint * rateToPower);
                var denom = 1 - rateToPower;

                return (numer / denom) + (1.0 / (1 - this.Rate));
            }
            else if (this.Precision == 1)
            {
                var logMean = (Math.Log(this.Rate) + GetLogNormalizer(this.Rate, this.Precision, this.StartPoint > 0 ? this.StartPoint - 1 : 0, this.EndPoint - 1))
                              - GetLogNormalizer(this.Rate, this.Precision, this.StartPoint, this.EndPoint);
                return Math.Exp(logMean);
            }
            else
            {
                throw new NotImplementedException("Calculation for mean is only available for Precision = 1, or Precision = 0 and Rate < 1");
            }
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution is uniform.
        /// </summary>
        public bool IsUniform() => this.leftTruncatedDistribution.IsUniform() && (this.EndPoint == int.MaxValue);

        /// <summary>
        /// Asks whether this instance is proper or not. A truncated COM-Poisson distribution
        /// is proper if Rate >= 0 and (Precision > 0 or (Precision == 0 and Rate &lt; 1)).
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper() => this.leftTruncatedDistribution.IsProper();

        /// <summary>
        /// Gets a value indicating how close this distribution is to another distribution.
        /// in terms of probabilities they assign to sequences.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>The max difference.</returns>
        public double MaxDiff(object that)
        {
            if (!(that is TruncatedPoisson))
            {
                return double.PositiveInfinity;
            }

            var thatd = (TruncatedPoisson)that;

            return Math.Max(this.leftTruncatedDistribution.MaxDiff(thatd.leftTruncatedDistribution), (double)Math.Abs(this.EndPoint - thatd.EndPoint));
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

            int result;
            do
            {
                result = this.leftTruncatedDistribution.Sample();
            }
            while (result > this.EndPoint);

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
        /// Sets the parameters to represent the product of two truncated COM-Poissons.
        /// </summary>
        /// <param name="a">The first truncated COM-Poisson</param>
        /// <param name="b">The second truncated COM-Poisson</param>
        public void SetToProduct(TruncatedPoisson a, TruncatedPoisson b)
        {
            if (Math.Max(a.StartPoint, b.StartPoint) > Math.Min(a.EndPoint, b.EndPoint))
            {
                throw new AllZeroException();
            }

            this.leftTruncatedDistribution = a.leftTruncatedDistribution * b.leftTruncatedDistribution;
            this.EndPoint = Math.Min(a.EndPoint, b.EndPoint);
            this.ResetCache();
        }

        /// <summary>
        /// Sets the current distribution to be uniform.
        /// </summary>
        public void SetToUniform()
        {
            this.leftTruncatedDistribution.SetToUniform();
            this.EndPoint = int.MaxValue;
            this.ResetCache();
        }

        /// <summary>
        /// Sets the current distribution to be uniform between given start and end points.
        /// </summary>
        /// <param name="startPoint">The index of the first non-zero term in the truncated distribution.</param>
        /// <param name="endPoint">The index of the last non-zero term in the truncated distribution.</param>
        public void SetToPartialUniform(int startPoint, int endPoint)
        {
            this.leftTruncatedDistribution.SetToPartialUniform(startPoint);
            this.EndPoint = endPoint;
            this.ResetCache();
        }

        /// <inheritdoc />
        public void SetTo(TruncatedPoisson value)
        {
            this.leftTruncatedDistribution.SetTo(value.leftTruncatedDistribution);
            this.EndPoint = value.EndPoint;
            this.ResetCache();
        }

        /// <inheritdoc />
        public void SetTo(LeftTruncatedPoisson value)
        {
            this.leftTruncatedDistribution.SetTo(value);
            this.EndPoint = int.MaxValue;
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
                if (startPoint >= this.EndPoint)
                {
                    this.Point = this.EndPoint;
                    return;
                }
                else
                {
                    mean = startPoint + Delta;
                }
            }

            var projection = FromMeanAndStartPointAndEndPoint(mean, 1.0, startPoint, this.EndPoint);
            this.SetTo(projection);
        }

        /// <summary>
        /// Overrides ToString method.
        /// </summary>
        /// <returns>String representation of instance.</returns>
        public override string ToString()
        {
            if (this.EndPointIsPositiveInfinity)
            {
                return this.leftTruncatedDistribution.ToString();
            }

            if (this.IsPointMass)
            {
                return "TruncPois.PointMass(" + this.Point + ")";
            }
            else if (this.IsPartialUniform())
            {
                return "TruncPois.Uniform(" + this.StartPoint + "," + this.EndPoint + ")";
            }
            else
            {
                return string.Format("TruncPois:Rate={0:0.000},Prec={1:0.00},Start={2}, End={3}", this.Rate, this.Precision, this.StartPoint, this.EndPoint);
            }
        }

        /// <inheritdoc />
        public double GetAverageLog(TruncatedPoisson that)
        {
            if (this.EndPointIsPositiveInfinity && that.EndPointIsPositiveInfinity)
            {
                return this.leftTruncatedDistribution.GetAverageLog(that.leftTruncatedDistribution);
            }

            throw new NotImplementedException();
        }

        /// <inheritdoc />
        public void SetToPartialUniform()
        {
            this.leftTruncatedDistribution.SetToPartialUniform();
            this.ResetCache();
        }

        /// <inheritdoc />
        public void SetToPartialUniformOf(TruncatedPoisson dist)
        {
            this.leftTruncatedDistribution.SetToPartialUniformOf(dist.leftTruncatedDistribution);
            this.EndPoint = dist.EndPoint;
            this.ResetCache();
        }

        /// <inheritdoc />
        public void SetToPartialUniformOf(Discrete dist)
        {
            this.leftTruncatedDistribution.SetToPartialUniformOf(dist);
            this.EndPoint = dist.GetLogProbs().FindLastIndex(prob => !double.IsNegativeInfinity(prob));
            this.ResetCache();
        }

        /// <inheritdoc />
        public bool IsPartialUniform() => this.leftTruncatedDistribution.IsPartialUniform();

        /// <inheritdoc />
        public int GetStartPoint()
        {
            return this.StartPoint;
        }

        /// <inheritdoc />
        public void TruncateLeft(int startPoint)
        {
            this.leftTruncatedDistribution.TruncateLeft(startPoint);
        }

        /// <inheritdoc />
        public int GetEndPoint()
        {
            return this.EndPoint;
        }

        /// <inheritdoc />
        public void TruncateRight(int endPoint)
        {
            this.EndPoint = endPoint;
            this.ResetCache();
        }

        /// <summary>
        /// Resets the cache for this instance.
        /// </summary>
        private void ResetCache()
        {
            this.leftTruncatedDistribution.ResetCache();
            this.logNormalizer = new Lazy<double>(() => this.EndPointIsPositiveInfinity ? this.leftTruncatedDistribution.LogNormalizer : GetLogNormalizer(this.Rate, this.Precision, this.StartPoint, this.EndPoint));
        }
    }
}