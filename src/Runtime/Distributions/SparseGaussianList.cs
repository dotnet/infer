// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Collections;
    using Math;
    using Factors.Attributes;

    /// <summary>
    /// Represents a sparse list of Gaussian distributions, optimized for the case where many share 
    /// the same parameter value. The class supports
    /// an approximation tolerance which allows elements close to the common value to be
    /// automatically reset to the common value. 
    /// </summary>
    [Serializable]
    [Quality(QualityBand.Stable)]
    [DataContract]
    public class SparseGaussianList : SparseDistributionList<Gaussian, double, SparseGaussianList>,
                                      IDistribution<ISparseList<double>>, Sampleable<ISparseList<double>>,
                                      SettableTo<SparseGaussianList>, SettableToProduct<SparseGaussianList>,
                                      SettableToPower<SparseGaussianList>, SettableToRatio<SparseGaussianList>,
                                      SettableToWeightedSum<SparseGaussianList>, CanGetLogAverageOf<SparseGaussianList>, CanGetLogAverageOfPower<SparseGaussianList>,
                                      CanGetAverageLog<SparseGaussianList>, CanGetMean<ISparseList<double>>, CanGetVariance<ISparseList<double>>
    {
        /// <summary>
        /// Initializes static members of the <see cref="SparseGaussianList"/> class.
        /// </summary>
        static SparseGaussianList()
        {
            SparseGaussianList.DefaultTolerance = 0.01;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SparseGaussianList"/> class.
        /// </summary>
        public SparseGaussianList() : base()
        {
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> class
        /// of a given size, with each element having a given mean and variance.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">The desired mean.</param>
        /// <param name="variance">The desired variance.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromMeanAndVariance(
            int size, double mean, double variance, double tolerance)
        {
            return SparseGaussianList.Constant(size, Gaussian.FromMeanAndVariance(mean, variance), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> class
        /// of a given size, with each element having a given mean and precision.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">The desired mean.</param>
        /// <param name="precision">The desired precision.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromMeanAndPrecision(
            int size, double mean, double precision, double tolerance)
        {
            return SparseGaussianList.Constant(size, Gaussian.FromMeanAndPrecision(mean, precision), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> class from sparse mean and precision lists.
        /// </summary>
        /// <param name="mean">The desired mean.</param>
        /// <param name="precision">The desired precision.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromMeanAndPrecision(
            ISparseList<double> mean, ISparseList<double> precision, double tolerance)
        {
            var result = FromSize(mean.Count, tolerance);
            result.SetToFunction(mean, precision, (m, p) => Gaussian.FromMeanAndPrecision(m, p));
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> class
        /// of a given size, with each element having the specified natural parameters.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="meanTimesPrecision">The mean times precision value.</param>
        /// <param name="precision">The precision value.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromNatural(int size, double meanTimesPrecision, double precision, double tolerance)
        {
            return SparseGaussianList.Constant(size, Gaussian.FromNatural(meanTimesPrecision, precision), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> 
        /// class of a given size, with each element having a given mean and variance.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">The desired mean.</param>
        /// <param name="variance">The desired variance.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromMeanAndVariance(
            int size, double mean, double variance)
        {
            return FromMeanAndVariance(size, mean, variance, SparseGaussianList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> class
        /// of a given size, with each element having a given mean and precision.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">The desired mean.</param>
        /// <param name="precision">The desired precision.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromMeanAndPrecision(
            int size, double mean, double precision)
        {
            return FromMeanAndPrecision(size, mean, precision, SparseGaussianList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGaussianList"/> class from sparse mean and precision lists.
        /// </summary>
        /// <param name="mean">The desired mean values.</param>
        /// <param name="precision">The desired precision values.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromMeanAndPrecision(
            ISparseList<double> mean, ISparseList<double> precision)
        {
            return FromMeanAndPrecision(mean, precision, SparseGaussianList.DefaultTolerance);
        }

        /// <summary>
        /// CReturns a new instance of the <see cref="SparseGaussianList"/> class
        /// of a given size, with each element having the specified natural parameters.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="meanTimesPrecision">The mean times precision value.</param>
        /// <param name="precision">The precision value.</param>
        /// <returns>The new <see cref="SparseGaussianList"/> instance.</returns>
        public static SparseGaussianList FromNatural(int size, double meanTimesPrecision, double precision)
        {
            return FromNatural(size, meanTimesPrecision, precision, SparseGaussianList.DefaultTolerance);
        }

        /// <summary>
        /// Samples from a list of Gaussian distributions with the specified vectors.
        /// of means and precisions
        /// </summary>
        /// <param name="means">Vector of means.</param>
        /// <param name="precs">Vector of precisions.</param>
        /// <returns>The sample.</returns>
        [Stochastic]
        public static ISparseList<double> Sample(ISparseList<double> means, ISparseList<double> precs)
        {
            SparseList<double> sample = SparseList<double>.FromSize(means.Count);
            sample.SetToFunction(means, precs, (m, p) => Gaussian.Sample(m, p));
            return sample;
        }

        /// <summary>
        /// Creates a <see cref="SparseGaussianList"/> distribution which is the product of two others.
        /// </summary>
        /// <param name="a">The first distribution.</param>
        /// <param name="b">The second distribution.</param>
        /// <returns>The resulting <see cref="SparseGaussianList"/> distribution.</returns>
        public static SparseGaussianList operator *(SparseGaussianList a, SparseGaussianList b)
        {
            var result = SparseGaussianList.FromSize(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseGaussianList"/> distribution which is the ratio of two others.
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <returns>The resulting <see cref="SparseGaussianList"/> distribution.</returns>
        public static SparseGaussianList operator /(SparseGaussianList numerator, SparseGaussianList denominator)
        {
            var result = SparseGaussianList.FromSize(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseGaussianList"/> distribution which is the power of another.
        /// </summary>
        /// <param name="dist">The other distribution.</param>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The resulting <see cref="SparseGaussianList"/> distribution.</returns>
        public static SparseGaussianList operator ^(SparseGaussianList dist, double exponent)
        {
            var result = SparseGaussianList.FromSize(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }
    }
}