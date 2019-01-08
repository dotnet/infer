// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a factorized distribution over pairs of elements
    /// <c>P(x, y) = P(x)P(y)</c>. Pair elements <c>x</c> and <c>y</c> can be of different types.
    /// </summary>
    /// <typeparam name="TElement1">The type of a first element of a pair.</typeparam>
    /// <typeparam name="TElementDistribution1">The type of a distribution over <typeparamref name="TElement1"/>.</typeparam>
    /// <typeparam name="TElement2">The type of a second element of a pair.</typeparam>
    /// <typeparam name="TElementDistribution2">The type of a distribution over <typeparamref name="TElement2"/>.</typeparam>
    [Quality(QualityBand.Experimental)]
    public class PairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2> :
        PairDistributionBase<TElement1, TElementDistribution1, TElement2, TElementDistribution2, PairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2>>
        where TElementDistribution1 : IDistribution<TElement1>, CanGetLogAverageOf<TElementDistribution1>, SettableToProduct<TElementDistribution1>, SettableToPartialUniform<TElementDistribution1>, new()
        where TElementDistribution2 : IDistribution<TElement2>, CanGetLogAverageOf<TElementDistribution2>, SettableToProduct<TElementDistribution2>, SettableToPartialUniform<TElementDistribution2>, new()
    {
        /// <summary>
        /// Creates a distribution <c>Q(y, x) = P(x, y)</c>, where <c>P(x, y)</c> is the current distribution.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public PairDistribution<TElement2, TElementDistribution2, TElement1, TElementDistribution1> Transpose()
        {
            return PairDistribution<TElement2, TElementDistribution2, TElement1, TElementDistribution1>.FromFirstSecond(this.Second, this.First);
        }
    }

    /// <summary>
    /// This class can represent distributions over pairs of two types: a factorized <c>P(x, y) = P(x) P(y)</c>,
    /// and correlated <c>P(x, y) \propto P(x) P(y) I[x=y]</c>. Both elements of a pair must be of the same type.
    /// </summary>
    /// <typeparam name="TElement">The type of a pair element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over <typeparamref name="TElement"/>.</typeparam>
    public class PairDistribution<TElement, TElementDistribution> :
        PairDistributionBase<TElement, TElementDistribution, TElement, TElementDistribution, PairDistribution<TElement, TElementDistribution>>
        where TElementDistribution : IDistribution<TElement>, CanGetLogAverageOf<TElementDistribution>, SettableToProduct<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
    {
        /// <summary>
        /// Stores the product of distributions over the first and the the second element
        /// when the equality constraint is enabled.
        /// </summary>
        private Option<TElementDistribution> firstTimesSecond;

        /// <summary>
        /// Gets a value indicating whether the equality constraint is set on the distribution.
        /// </summary>
        public bool HasEqualityConstraint { get; private set; }

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public override Pair<Option<TElement>, Option<TElement>> Point
        {
            get => base.Point;

            set
            {
                base.Point = value;
                this.HasEqualityConstraint = false;
            }
        }

        /// <summary>
        /// Creates a distribution <c>P(x, y) \propto Q(x) R(y) I[x=y]</c>, where <c>Q(x)</c> and <c>R(y)</c> are given element distributions.
        /// </summary>
        /// <param name="firstElementDistribution">The marginal distribution of the first element of a pair.</param>
        /// <param name="secondElementDistribution">The marginal distribution of the second element of a pair.</param>
        /// <returns>The created distribution.</returns>
        public static PairDistribution<TElement, TElementDistribution> Constrained(
            TElementDistribution firstElementDistribution, TElementDistribution secondElementDistribution)
        {
            Argument.CheckIfValid(firstElementDistribution != null, nameof(firstElementDistribution));
            Argument.CheckIfValid(secondElementDistribution != null, nameof(secondElementDistribution));
            
            var result = new PairDistribution<TElement, TElementDistribution>
            {
                First = firstElementDistribution,
                Second = secondElementDistribution,
                HasEqualityConstraint = true,
            };

            result.firstTimesSecond = Distribution.Product<TElement, TElementDistribution>(
                result.First.Value, result.Second.Value);

            return result;
        }

        /// <summary>
        /// Creates a distribution <c>P(x, y) \propto P(x) P(y) I[x=y]</c>, where <c>P</c> is a given element distribution.
        /// </summary>
        /// <param name="elementDistribution">The marginal distribution over pair elements.</param>
        /// <returns>The created distribution.</returns>
        public static PairDistribution<TElement, TElementDistribution> Constrained(TElementDistribution elementDistribution)
        {
            return Constrained(elementDistribution, elementDistribution);
        }

        /// <summary>
        /// Creates a distribution <c>P(x, y) \propto I[x=y]</c>.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static PairDistribution<TElement, TElementDistribution> UniformConstrained()
        {
            return Constrained(Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Computes <c>P(y) = R(x, y)</c>, where <c>R(x, y)</c> is the current pair distribution,
        /// and <c>x</c> is a given element.
        /// </summary>
        /// <param name="first">The element to project.</param>
        /// <param name="result">The normalized projection result.</param>
        /// <returns>The logarithm of the scale for the projection result.</returns>
        public override double ProjectFirst(TElement first, out Option<TElementDistribution> result)
        {
            if (this.HasEqualityConstraint)
            {
                Debug.Assert(
                    this.First.HasValue && this.Second.HasValue,
                    "Cannot have a constrained pair distribution with a missing element distribution.");
                Debug.Assert(this.firstTimesSecond.HasValue, "Must have been computed.");

                double logAverageOf = this.firstTimesSecond.Value.GetLogProb(first);
                if (double.IsNegativeInfinity(logAverageOf))
                {
                    result = default(TElementDistribution);
                    return double.NegativeInfinity;
                }

                result = new TElementDistribution { Point = first };
                return logAverageOf;
            }

            return base.ProjectFirst(first, out result);
        }

        /// <summary>
        /// Computes <c>P(y) = sum_x Q(x) R(x, y)</c>, where <c>R(x, y)</c> is the current pair distribution,
        /// and <c>Q(x)</c> is a given element distribution.
        /// </summary>
        /// <param name="first">The element distribution to project.</param>
        /// <param name="result">The normalized projection result.</param>
        /// <returns>The logarithm of the scale for the projection result.</returns>
        public override double ProjectFirst(TElementDistribution first, out Option<TElementDistribution> result)
        {
            Argument.CheckIfNotNull(first, "first");

            if (first.IsPointMass)
            {
                return this.ProjectFirst(first.Point, out result);
            }
            
            if (this.HasEqualityConstraint)
            {
                Debug.Assert(
                    this.First.HasValue && this.Second.HasValue,
                    "Cannot have a constrained pair distribution with a missing element distribution.");
                Debug.Assert(this.firstTimesSecond.HasValue, "Must have been computed.");
                
                // TODO: can we introduce a single method that does both GetLogAverageOf and SetToProduct?
                double logAverageOf = Distribution.GetLogAverageOf<TElement, TElementDistribution>(
                    this.firstTimesSecond.Value, first, out var product);
                
                if (double.IsNegativeInfinity(logAverageOf))
                {
                    result = Option.None;
                    return double.NegativeInfinity;
                }
                
                result = product;
                return logAverageOf;    
            }

            return base.ProjectFirst(first, out result);
        }

        /// <summary>
        /// Checks whether the current distribution is uniform.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is uniform, <see langword="false"/> otherwise.</returns>
        public override bool IsUniform()
        {
            return !this.HasEqualityConstraint && base.IsUniform();
        }

        /// <summary>
        /// Replaces the current distribution with a uniform distribution.
        /// </summary>
        public override void SetToUniform()
        {
            base.SetToUniform();
            this.HasEqualityConstraint = false;
        }

        /// <summary>
        /// Creates a copy of the current distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public override object Clone()
        {
            var result = (PairDistribution<TElement, TElementDistribution>)base.Clone();
            result.HasEqualityConstraint = this.HasEqualityConstraint;
            result.firstTimesSecond =
                this.firstTimesSecond.HasValue
                    ? Option.Some((TElementDistribution)this.firstTimesSecond.Value.Clone())
                    : Option.None;
            return result;
        }

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <returns>
        /// A string that represents the distribution.
        /// </returns>
        public override string ToString()
        {
            if (this.HasEqualityConstraint)
            {
                return this.First + "©";
            }

            return base.ToString();
        }

        /// <summary>
        /// Returns the logarithm of the probability that the current distribution would draw the same sample
        /// as a given one.
        /// </summary>
        /// <param name="that">The given distribution.</param>
        /// <returns>The logarithm of the probability that distributions would draw the same sample.</returns>
        public override double GetLogAverageOf(PairDistribution<TElement, TElementDistribution> that)
        {
            Argument.CheckIfNotNull(that, "that");
            
            if (this.HasEqualityConstraint)
            {
                Debug.Assert(
                    this.First.HasValue &&!this.Second.HasValue &&
                    that.First.HasValue && that.Second.HasValue,
                    "Cannot have a constrained pair distribution with a missing element distribution.");

                double result = 0;
                var product = (TElementDistribution)this.firstTimesSecond.Value.Clone();
                result += product.GetLogAverageOf(that.First.Value);
                product.SetToProduct(product, that.First.Value);
                result += product.GetLogAverageOf(that.Second.Value);
                result -= that.First.Value.GetLogAverageOf(this.Second.Value);

                return result;
            }

            return base.GetLogAverageOf(that);
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="distribution">The distribution which support will be used to setup the current distribution.</param>
        public override void SetToPartialUniformOf(PairDistribution<TElement, TElementDistribution> distribution)
        {
            base.SetToPartialUniformOf(distribution);
            
            this.HasEqualityConstraint = distribution.HasEqualityConstraint;
            if (this.HasEqualityConstraint)
            {
                this.firstTimesSecond = Distribution.Product<TElement, TElementDistribution>(this.First.Value, this.Second.Value);
            }
        }

        /// <summary>
        /// Creates a distribution <c>Q(y, x) = P(x, y)</c>, where <c>P(x, y)</c> is the current distribution.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public PairDistribution<TElement, TElementDistribution> Transpose()
        {
            return new PairDistribution<TElement, TElementDistribution>
            {
                First = this.Second,
                Second = this.First,
                HasEqualityConstraint = this.HasEqualityConstraint,
                firstTimesSecond = this.firstTimesSecond
            };
        }
    }
}
