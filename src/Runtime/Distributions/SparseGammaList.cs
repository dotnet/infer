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
    /// Represents a sparse list of Gamma distributions, optimized for the case where many share 
    /// the same parameter value. The class supports
    /// an approximation tolerance which allows elements close to the common value to be
    /// automatically reset to the common value. 
    /// </summary>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Stable)]
    public class SparseGammaList : SparseDistributionList<Gamma, double, SparseGammaList>,
                                   IDistribution<ISparseList<double>>, Sampleable<ISparseList<double>>,
                                   SettableTo<SparseGammaList>, SettableToProduct<SparseGammaList>,
                                   SettableToPower<SparseGammaList>, SettableToRatio<SparseGammaList>,
                                   SettableToWeightedSum<SparseGammaList>, CanGetLogAverageOf<SparseGammaList>, CanGetLogAverageOfPower<SparseGammaList>,
                                   CanGetAverageLog<SparseGammaList>, CanGetMean<ISparseList<double>>, CanGetVariance<ISparseList<double>>
    {
        /// <summary>
        /// Initializes static members of the <see cref="SparseGammaList"/> class.
        /// </summary>
        static SparseGammaList()
        {
            SparseGammaList.DefaultTolerance = 0.01;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SparseGammaList"/> class.
        /// </summary>
        public SparseGammaList() : base()
        {
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given mean and variance.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">The desired mean.</param>
        /// <param name="variance">The desired variance.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromMeanAndVariance(
            int size, double mean, double variance, double tolerance)
        {
            return SparseGammaList.Constant(size, Gamma.FromMeanAndVariance(mean, variance), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given mean and mean log.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="meanLog">Desired expected logarithm.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromMeanAndMeanLog(
            int size, double mean, double meanLog, double tolerance)
        {
            return SparseGammaList.Constant(size, Gamma.FromMeanAndMeanLog(mean, meanLog), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given shape and rate.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="shape">The shape value.</param>
        /// <param name="rate">The rate value (= 1/scale).</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndRate(
            int size, double shape, double rate, double tolerance)
        {
            return SparseGammaList.Constant(size, Gamma.FromShapeAndRate(shape, rate), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size with given shape and rate Vectors.
        /// </summary>
        /// <param name="shape">The shape vector.</param>
        /// <param name="rate">The rate vector (= 1/scale).</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndRate(
            Vector shape, Vector rate, double tolerance)
        {
            var result = FromSize(shape.Count, tolerance);
            result.SetToFunction(shape, rate, (s, r) => Gamma.FromShapeAndRate(s, r));
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given shape and scale.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="shape">The shape value.</param>
        /// <param name="scale">The scale value.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndScale(
            int size, double shape, double scale, double tolerance)
        {
            return SparseGammaList.Constant(size, Gamma.FromShapeAndScale(shape, scale), tolerance);
        }

        /// <summary>
        /// CReturns a new instance of the <see cref="SparseGammaList"/> class
        /// from sparse lists of shapes and scales.
        /// </summary>
        /// <param name="shape">The shape values.</param>
        /// <param name="scale">The scale values.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndScale(
            ISparseList<double> shape, ISparseList<double> scale, double tolerance)
        {
            var result = FromSize(shape.Count, tolerance);
            result.SetToFunction(shape, scale, (s, sc) => Gamma.FromShapeAndScale(s, sc));
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having the specified natural parameters.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="shapeMinus1">shape - 1.</param>
        /// <param name="rate">rate = 1/scale.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromNatural(int size, double shapeMinus1, double rate, double tolerance)
        {
            return SparseGammaList.Constant(size, Gamma.FromNatural(shapeMinus1, rate), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given mean and variance.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">The desired mean.</param>
        /// <param name="variance">The desired variance.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromMeanAndVariance(
            int size, double mean, double variance)
        {
            return FromMeanAndVariance(size, mean, variance, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given mean and mean log.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="meanLog">Desired expected logarithm.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromMeanAndMeanLog(
            int size, double mean, double meanLog)
        {
            return FromMeanAndMeanLog(size, mean, meanLog, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given shape and rate.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="shape">The shape value.</param>
        /// <param name="rate">The rate value (= 1/scale).</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndRate(
            int size, double shape, double rate)
        {
            return FromShapeAndRate(size, shape, rate, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class of a given size with given shape and rate Vectors.
        /// </summary>
        /// <param name="shape">The shape vector.</param>
        /// <param name="rate">The rate vector (= 1/scale).</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndRate(
            Vector shape, Vector rate)
        {
            return FromShapeAndRate(shape, rate, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having a given shape and scale.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="shape">The shape value.</param>
        /// <param name="scale">The scale value.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndScale(
            int size, double shape, double scale)
        {
            return FromShapeAndScale(size, shape, scale, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class from sparse lists of shapes and scales.
        /// </summary>
        /// <param name="shape">The shape values.</param>
        /// <param name="scale">The scale values.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromShapeAndScale(
            ISparseList<double> shape, ISparseList<double> scale)
        {
            return FromShapeAndScale(shape, scale, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseGammaList"/> class
        /// of a given size, with each element having the specified natural parameters.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="shapeMinus1">shape - 1.</param>
        /// <param name="rate">rate = 1/scale.</param>
        /// <returns>The new <see cref="SparseGammaList"/> instance.</returns>
        public static SparseGammaList FromNatural(int size, double shapeMinus1, double rate)
        {
            return FromNatural(size, shapeMinus1, rate, SparseGammaList.DefaultTolerance);
        }

        /// <summary>
        /// Samples from a list of Gamma distributions with the specified vectors.
        /// of shapes and rates
        /// </summary>
        /// <param name="shapes">Vector of shapes.</param>
        /// <param name="rates">Vector of rates.</param>
        /// <returns>The sample.</returns>
        [Stochastic]
        public static ISparseList<double> Sample(ISparseList<double> shapes, ISparseList<double> rates)
        {
            SparseList<double> sample = SparseList<double>.FromSize(shapes.Count);
            sample.SetToFunction(shapes, rates, (s, r) => Gamma.Sample(s, r));
            return sample;
        }

        /// <summary>
        /// Creates a <see cref="SparseGammaList"/> distribution which is the product of two others.
        /// </summary>
        /// <param name="a">The first distribution.</param>
        /// <param name="b">The second distribution.</param>
        /// <returns>The resulting <see cref="SparseGammaList"/> distribution.</returns>
        public static SparseGammaList operator *(SparseGammaList a, SparseGammaList b)
        {
            var result = SparseGammaList.FromSize(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseGammaList"/> distribution which is the ratio of two others.
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <returns>The resulting <see cref="SparseGammaList"/> distribution.</returns>
        public static SparseGammaList operator /(SparseGammaList numerator, SparseGammaList denominator)
        {
            var result = SparseGammaList.FromSize(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseGammaList"/> distribution which is the power of another.
        /// </summary>
        /// <param name="dist">The other distribution.</param>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The resulting <see cref="SparseGammaList"/> distribution.</returns>
        public static SparseGammaList operator ^(SparseGammaList dist, double exponent)
        {
            var result = SparseGammaList.FromSize(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }
    }
}