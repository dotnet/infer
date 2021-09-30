// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
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
    public class ImmutablePairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2> :
        ImmutablePairDistributionBase<TElement1, TElementDistribution1, TElement2, TElementDistribution2, ImmutablePairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2>>
        where TElementDistribution1 : IImmutableDistribution<TElement1, TElementDistribution1>, CanGetLogAverageOf<TElementDistribution1>, CanComputeProduct<TElementDistribution1>, CanCreatePartialUniform<TElementDistribution1>, new()
        where TElementDistribution2 : IImmutableDistribution<TElement2, TElementDistribution2>, CanGetLogAverageOf<TElementDistribution2>, CanComputeProduct<TElementDistribution2>, CanCreatePartialUniform<TElementDistribution2>, new()
    {
        /// <summary>
        /// Creates a distribution <c>Q(y, x) = P(x, y)</c>, where <c>P(x, y)</c> is the current distribution.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public ImmutablePairDistribution<TElement2, TElementDistribution2, TElement1, TElementDistribution1> Transpose()
        {
            return ImmutablePairDistribution<TElement2, TElementDistribution2, TElement1, TElementDistribution1>.FromFirstSecond(this.Second, this.First);
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override double GetLogProb((Option<TElement1>, Option<TElement2>) pair)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override ImmutablePairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2> Multiply(ImmutablePairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2> other)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override ImmutablePairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2> Sum(double weightThis, ImmutablePairDistribution<TElement1, TElementDistribution1, TElement2, TElementDistribution2> other, double weightOther)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// This class can represent distributions over pairs of two types: a factorized <c>P(x, y) = P(x) P(y)</c>,
    /// and correlated <c>P(x, y) \propto P(x) P(y) I[x=y]</c>. Both elements of a pair must be of the same type.
    /// </summary>
    /// <typeparam name="TElement">The type of a pair element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over <typeparamref name="TElement"/>.</typeparam>
    public class ImmutablePairDistribution<TElement, TElementDistribution> :
        ImmutablePairDistributionBase<TElement, TElementDistribution, TElement, TElementDistribution, ImmutablePairDistribution<TElement, TElementDistribution>>
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, new()
    {
        /// <summary>
        /// Stores the product of distributions over the first and the the second element
        /// when the equality constraint is enabled.
        /// </summary>
        private Option<TElementDistribution> firstTimesSecond;

        /// <summary>
        /// Gets a value indicating whether the equality constraint is set on the distribution.
        /// </summary>
        // Should only ever be set in factory methods.
        // TODO: replace with init-only property after switching to C# 9.0+
        public bool HasEqualityConstraint { get; private set; }

        /// <inheritdoc/>
        public override ImmutablePairDistribution<TElement, TElementDistribution> CreatePointMass(
            (Option<TElement>, Option<TElement>) point)
        {
            var result = base.CreatePointMass(point);
            result.HasEqualityConstraint = false;
            return result;
        }

        /// <summary>
        /// Creates a distribution <c>P(x, y) \propto Q(x) R(y) I[x=y]</c>, where <c>Q(x)</c> and <c>R(y)</c> are given element distributions.
        /// </summary>
        /// <param name="firstElementDistribution">The marginal distribution of the first element of a pair.</param>
        /// <param name="secondElementDistribution">The marginal distribution of the second element of a pair.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutablePairDistribution<TElement, TElementDistribution> Constrained(
            TElementDistribution firstElementDistribution, TElementDistribution secondElementDistribution)
        {
            Argument.CheckIfValid(firstElementDistribution != null, nameof(firstElementDistribution));
            Argument.CheckIfValid(secondElementDistribution != null, nameof(secondElementDistribution));
            
            var result = new ImmutablePairDistribution<TElement, TElementDistribution>
            {
                First = firstElementDistribution,
                Second = secondElementDistribution,
                HasEqualityConstraint = true,
            };

            result.firstTimesSecond = result.First.Value.Multiply(result.Second.Value);

            return result;
        }

        /// <summary>
        /// Creates a distribution <c>P(x, y) \propto P(x) P(y) I[x=y]</c>, where <c>P</c> is a given element distribution.
        /// </summary>
        /// <param name="elementDistribution">The marginal distribution over pair elements.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutablePairDistribution<TElement, TElementDistribution> Constrained(TElementDistribution elementDistribution)
        {
            return Constrained(elementDistribution, elementDistribution);
        }

        /// <summary>
        /// Creates a distribution <c>P(x, y) \propto I[x=y]</c>.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static ImmutablePairDistribution<TElement, TElementDistribution> UniformConstrained()
        {
            return Constrained(ElementDistribution1Factory.CreateUniform());
        }

        /// <inheritdoc/>
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

                result = ElementDistribution1Factory.CreatePointMass(first);
                return logAverageOf;
            }

            return base.ProjectFirst(first, out result);
        }

        /// <inheritdoc/>
        public override double ProjectFirst(TElementDistribution first, out Option<TElementDistribution> result)
        {
            Argument.CheckIfNotNull(first, nameof(first));

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

                double logAverageOf = firstTimesSecond.Value.GetLogAverageOf(first);
                
                if (double.IsNegativeInfinity(logAverageOf))
                {
                    result = Option.None;
                    return double.NegativeInfinity;
                }
                
                result = firstTimesSecond.Value.Multiply(first);
                return logAverageOf;    
            }

            return base.ProjectFirst(first, out result);
        }

        /// <inheritdoc/>
        public override bool IsUniform()
        {
            return !this.HasEqualityConstraint && base.IsUniform();
        }

        /// <inheritdoc/>
        public override ImmutablePairDistribution<TElement, TElementDistribution> CreateUniform()
        {
            var result = base.CreateUniform();
            result.HasEqualityConstraint = false;
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

        /// <inheritdoc/>
        public override double GetLogAverageOf(ImmutablePairDistribution<TElement, TElementDistribution> that)
        {
            Argument.CheckIfNotNull(that, nameof(that));
            
            if (this.HasEqualityConstraint)
            {
                Debug.Assert(
                    First.HasValue &&!Second.HasValue &&
                    that.First.HasValue && that.Second.HasValue,
                    "Cannot have a constrained pair distribution with a missing element distribution.");

                double result = 0;
                var product = firstTimesSecond.Value;
                result += product.GetLogAverageOf(that.First.Value);
                product = product.Multiply(that.First.Value);
                result += product.GetLogAverageOf(that.Second.Value);
                result -= that.First.Value.GetLogAverageOf(Second.Value);

                return result;
            }

            return base.GetLogAverageOf(that);
        }

        /// <inheritdoc/>
        public override ImmutablePairDistribution<TElement, TElementDistribution> CreatePartialUniform()
        {
            var result = base.CreatePartialUniform();
            result.HasEqualityConstraint = HasEqualityConstraint;
            if (HasEqualityConstraint)
            {
                result.firstTimesSecond = First.Value.Multiply(Second.Value);
            }

            return result;
        }

        /// <summary>
        /// Creates a distribution <c>Q(y, x) = P(x, y)</c>, where <c>P(x, y)</c> is the current distribution.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public ImmutablePairDistribution<TElement, TElementDistribution> Transpose()
        {
            return new ImmutablePairDistribution<TElement, TElementDistribution>
            {
                First = this.Second,
                Second = this.First,
                HasEqualityConstraint = this.HasEqualityConstraint,
                firstTimesSecond = this.firstTimesSecond
            };
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override double GetLogProb((Option<TElement>, Option<TElement>) pair)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override ImmutablePairDistribution<TElement, TElementDistribution> Multiply(ImmutablePairDistribution<TElement, TElementDistribution> other)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        /// <remarks>Not currently implemented.</remarks>
        public override ImmutablePairDistribution<TElement, TElementDistribution> Sum(double weightThis, ImmutablePairDistribution<TElement, TElementDistribution> other, double weightOther)
        {
            throw new NotImplementedException();
        }
    }
}
