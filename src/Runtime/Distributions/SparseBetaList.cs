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
    /// Represents a sparse list of Beta distributions, optimized for the case where many share 
    /// the same parameter value. The class supports
    /// an approximation tolerance which allows elements close to the common value to be
    /// automatically reset to the common value. 
    /// </summary>
    [Serializable]
    [Quality(QualityBand.Stable)]
    [DataContract]
    public class SparseBetaList : SparseDistributionList<Beta, double, SparseBetaList>,
                                      IDistribution<ISparseList<double>>, Sampleable<ISparseList<double>>,
                                      SettableTo<SparseBetaList>, SettableToProduct<SparseBetaList>,
                                      SettableToPower<SparseBetaList>, SettableToRatio<SparseBetaList>,
                                      SettableToWeightedSum<SparseBetaList>, CanGetLogAverageOf<SparseBetaList>, CanGetLogAverageOfPower<SparseBetaList>,
                                      CanGetAverageLog<SparseBetaList>, CanGetMean<ISparseList<double>>, CanGetVariance<ISparseList<double>>
    {
        /// <summary>
        /// Initializes static members of the <see cref="SparseBetaList"/> class.
        /// </summary>
        static SparseBetaList()
        {
            SparseBetaList.DefaultTolerance = 0.01;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SparseBetaList"/> class.
        /// </summary>
        public SparseBetaList()
            : base()
        {
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBetaList"/> class of a given size, with each
        /// element having given true and false counts.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="trueCount">The true count.</param>
        /// <param name="falseCount">The false count.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>A new instance of the <see cref="SparseBetaList"/> class.</returns>
        public static SparseBetaList FromCounts(
            int size, double trueCount, double falseCount, double tolerance)
        {
            return SparseBetaList.Constant(size, new Beta(trueCount, falseCount));
        }

        /// <summary>
        /// Returns a new instance of instance of the <see cref="SparseBetaList"/> class
        /// from sparse lists of true and false counts.
        /// </summary>
        /// <param name="trueCounts">The true counts.</param>
        /// <param name="falseCounts">The false counts.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>A new instance of the <see cref="SparseBetaList"/> class.</returns>
        public static SparseBetaList FromCounts(
            ISparseList<double> trueCounts, ISparseList<double> falseCounts, double tolerance)
        {
            var result = FromSize(trueCounts.Count, tolerance);
            result.SetToFunction(trueCounts, falseCounts, (tc, fc) => new Beta(tc, fc));
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBetaList"/> class
        /// of a given size, with each element having given true and false counts.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="trueCount">The true count.</param>
        /// <param name="falseCount">The false count.</param>
        /// <returns>A new instance of the <see cref="SparseBetaList"/> class.</returns>
        public static SparseBetaList FromCounts(
            int size, double trueCount, double falseCount)
        {
            return FromCounts(size, trueCount, falseCount, SparseBetaList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBetaList"/> class
        /// from sparse lists of true and false counts.
        /// </summary>
        /// <param name="trueCounts">The true counts.</param>
        /// <param name="falseCounts">The false counts.</param>
        /// <returns>A new instance of the <see cref="SparseBetaList"/> class.</returns>
        public static SparseBetaList FromCounts(
            ISparseList<double> trueCounts, ISparseList<double> falseCounts)
        {
            return FromCounts(trueCounts, falseCounts, SparseBetaList.DefaultTolerance);
        }

        /// <summary>
        /// Samples from a <see cref="SparseBetaList"/> with the specified sparse lists
        /// of true counts and false counts.
        /// </summary>
        /// <param name="trueCounts">The true counts.</param>
        /// <param name="falseCounts">The false counts.</param>
        /// <returns>The sample.</returns>
        [Stochastic]
        public static ISparseList<double> Sample(ISparseList<double> trueCounts, ISparseList<double> falseCounts)
        {
            SparseList<double> sample = SparseList<double>.FromSize(trueCounts.Count);
            sample.SetToFunction(trueCounts, falseCounts, (tc, fc) => Beta.Sample(tc, fc));
            return sample;
        }

        /// <summary>
        /// Creates a <see cref="SparseBetaList"/> distribution which is the product of two others
        /// </summary>
        /// <param name="a">The first distribution.</param>
        /// <param name="b">The second distribution.</param>
        /// <returns>The resulting <see cref="SparseBetaList"/> distribution.</returns>
        public static SparseBetaList operator *(SparseBetaList a, SparseBetaList b)
        {
            var result = SparseBetaList.FromSize(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseBetaList"/> distribution which is the ratio of two others
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <returns>The resulting <see cref="SparseBetaList"/> distribution.</returns>
        public static SparseBetaList operator /(SparseBetaList numerator, SparseBetaList denominator)
        {
            var result = SparseBetaList.FromSize(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseBetaList"/> distribution which is the power of another.
        /// </summary>
        /// <param name="dist">The other distribution.</param>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The resulting <see cref="SparseBetaList"/> distribution.</returns>
        public static SparseBetaList operator ^(SparseBetaList dist, double exponent)
        {
            var result = SparseBetaList.FromSize(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }
    }
}