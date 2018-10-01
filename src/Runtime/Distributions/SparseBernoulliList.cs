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
    /// Represents a sparse list of Bernoulli distributions, optimized for the case where many share 
    /// the same parameter value. The class supports
    /// an approximation tolerance which allows elements close to the common value to be
    /// automatically reset to the common value. 
    /// </summary>
    [Serializable]
    [Quality(QualityBand.Stable)]
    [DataContract]
    public class SparseBernoulliList : SparseDistributionList<Bernoulli, bool, SparseBernoulliList>,
                                      IDistribution<ISparseList<bool>>, Sampleable<ISparseList<bool>>,
                                      SettableTo<SparseBernoulliList>, SettableToProduct<SparseBernoulliList>,
                                      SettableToPower<SparseBernoulliList>, SettableToRatio<SparseBernoulliList>,
                                      SettableToWeightedSum<SparseBernoulliList>, CanGetLogAverageOf<SparseBernoulliList>, CanGetLogAverageOfPower<SparseBernoulliList>,
                                      CanGetAverageLog<SparseBernoulliList>, CanGetMean<ISparseList<double>>, CanGetVariance<ISparseList<double>>
    {
        /// <summary>
        /// Initializes static members of the <see cref="SparseBernoulliList"/> class.
        /// </summary>
        static SparseBernoulliList()
        {
            SparseBernoulliList.DefaultTolerance = 0.01;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SparseBernoulliList"/> class
        /// </summary>
        public SparseBernoulliList()
            : base()
        {
        }
        
        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class of a given size, with each
        /// element having a given probability of true.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="probTrue">The desired probability of true.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromProbTrue(
            int size, double probTrue)
        {
            return FromProbTrue(size, probTrue, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class of a given size, with each
        /// element having a given log odds.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="logOdds">The desired log odds.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromLogOdds(
            int size, double logOdds)
        {
            return FromLogOdds(size, logOdds, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class, with each
        /// element having a given probability of true.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="probTrue">The desired probability of true.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromProbTrue(
            int size, double probTrue, double tolerance)
        {
            return SparseBernoulliList.Constant(size, new Bernoulli(probTrue), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class of a given size, with each
        /// element having a given log odds.
        /// </summary>
        /// <param name="size">The size of the list.</param>
        /// <param name="logOdds">The desired log odds.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromLogOdds(
            int size, double logOdds, double tolerance)
        {
            return SparseBernoulliList.Constant(size, Bernoulli.FromLogOdds(logOdds), tolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class of a given size
        /// from a sparse list of probability true.
        /// </summary>
        /// <param name="probTrue">The sparse list of probability of true.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromProbTrue(
            ISparseList<double> probTrue)
        {
            return SparseBernoulliList.FromProbTrue(probTrue, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class of a given size
        /// from a sparse list of log odds.
        /// </summary>
        /// <param name="logOdds">The sparse list of log odds.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromLogOdds(
            ISparseList<double> logOdds)
        {
            return SparseBernoulliList.FromLogOdds(logOdds, SparseBernoulliList.DefaultTolerance);
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class
        /// of a given size from a sparse list of probability true.
        /// </summary>
        /// <param name="probTrue">The sparse list of probability of true.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromProbTrue(
            ISparseList<double> probTrue, double tolerance)
        {
            var result = FromSize(probTrue.Count, tolerance);
            result.SetToFunction(probTrue, pt => new Bernoulli(pt));
            return result;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="SparseBernoulliList"/> class
        /// of a given size from a sparse list of log odds.
        /// </summary>
        /// <param name="logOdds">The sparse list of log odds.</param>
        /// <param name="tolerance">The tolerance for the approximation.</param>
        /// <returns>The new <see cref="SparseBernoulliList"/> instance.</returns>
        public static SparseBernoulliList FromLogOdds(
            ISparseList<double> logOdds, double tolerance)
        {
            var result = FromSize(logOdds.Count, tolerance);
            result.SetToFunction(logOdds, lodds => Bernoulli.FromLogOdds(lodds));
            return result;
        }
 
        /// <summary>
        /// Samples from a <see cref="SparseBernoulliList"/> with the specified probability of true.
        /// </summary>
        /// <param name="probTrue">Probability of true.</param>
        /// <returns>The sample.</returns>
        [Stochastic]
        public static ISparseList<bool> Sample(ISparseList<double> probTrue)
        {
            SparseList<bool> sample = SparseList<bool>.FromSize(probTrue.Count);
            sample.SetToFunction(probTrue, p => Bernoulli.Sample(p));
            return sample;
        }

        /// <summary>
        /// Creates a <see cref="SparseBernoulliList"/> distribution which is the product of two others.
        /// </summary>
        /// <param name="a">The first distribution.</param>
        /// <param name="b">The second distribution.</param>
        /// <returns>The resulting <see cref="SparseBernoulliList"/> distribution.</returns>
        public static SparseBernoulliList operator *(SparseBernoulliList a, SparseBernoulliList b)
        {
            var result = SparseBernoulliList.FromSize(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseBernoulliList"/> distribution which is the ratio of two others.
        /// </summary>
        /// <param name="numerator">The numerator.</param>
        /// <param name="denominator">The denominator.</param>
        /// <returns>The resulting <see cref="SparseBernoulliList"/> distribution.</returns>
        public static SparseBernoulliList operator /(SparseBernoulliList numerator, SparseBernoulliList denominator)
        {
            var result = SparseBernoulliList.FromSize(numerator.Dimension);
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Creates a <see cref="SparseBernoulliList"/> distribution which is the power of another.
        /// </summary>
        /// <param name="dist">The other distribution.</param>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The resulting <see cref="SparseBernoulliList"/> distribution.</returns>
        public static SparseBernoulliList operator ^(SparseBernoulliList dist, double exponent)
        {
            var result = SparseBernoulliList.FromSize(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }
    }
}