// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// A base class for distributions over pairs of elements used to specify transducers.
    /// </summary>
    /// <typeparam name="TElement1">The type of a first element of a pair.</typeparam>
    /// <typeparam name="TElementDistribution1">The type of a distribution over <typeparamref name="TElement1"/>.</typeparam>
    /// <typeparam name="TElement2">The type of a second element of a pair.</typeparam>
    /// <typeparam name="TElementDistribution2">The type of a distribution over <typeparamref name="TElement2"/>.</typeparam>
    /// <typeparam name="TThis">The type of a concrete pair distribution class.</typeparam>
    /// <remarks>
    /// <p>
    /// Default interface implementations available in this class assume that the distribution is fully factorized:
    /// <c>P(x, y) = P(x) P(y)</c>, where <c>P(x)</c> is given by <see cref="First"/> and <c>P(y)</c> is given by second.
    /// Implementations of another kinds of pair distributions must override the interface implementation to ensure the correct behavior.
    /// </p>
    /// <p>
    /// One of the element distributions can be <see langword="null"/> to encode an epsilon input or an epsilon output
    /// in a transducer. Both distributions cannot be null at the same time: epsilon transition in transducers are specified by
    /// setting pair distributions to <see langword="null"/>.
    /// </p>
    /// </remarks>
    public abstract class ImmutablePairDistributionBase<TElement1, TElementDistribution1, TElement2, TElementDistribution2, TThis> :
        IImmutableDistribution<(Option<TElement1>, Option<TElement2>), TThis>, CanGetLogAverageOf<TThis>, CanComputeProduct<TThis>, SummableExactly<TThis>, CanCreatePartialUniform<TThis>
        where TElementDistribution1 : IImmutableDistribution<TElement1, TElementDistribution1>, CanGetLogAverageOf<TElementDistribution1>, CanComputeProduct<TElementDistribution1>, CanCreatePartialUniform<TElementDistribution1>, new()
        where TElementDistribution2 : IImmutableDistribution<TElement2, TElementDistribution2>, CanGetLogAverageOf<TElementDistribution2>, CanComputeProduct<TElementDistribution2>, CanCreatePartialUniform<TElementDistribution2>, new()
        where TThis : ImmutablePairDistributionBase<TElement1, TElementDistribution1, TElement2, TElementDistribution2, TThis>, new()
    {
        protected static readonly TElementDistribution1 ElementDistribution1Factory = new TElementDistribution1();
        protected static readonly TElementDistribution2 ElementDistribution2Factory = new TElementDistribution2();

        /// <summary>
        /// Gets or sets the marginal distribution over the first element in a pair.
        /// </summary>
        // Should only ever be set in factory methods.
        // TODO: replace with init-only property after switching to C# 9.0+
        public Option<TElementDistribution1> First { get; protected set; }

        /// <summary>
        /// Gets or sets the marginal distribution over the second element in a pair.
        /// </summary>
        public Option<TElementDistribution2> Second { get; protected set; }

        /// <inheritdoc/>
        public bool IsPointMass =>
            this.First.HasValue && this.First.Value.IsPointMass &&
            this.Second.HasValue && this.Second.Value.IsPointMass;

        /// <summary>
        /// Gets the point mass represented by the distribution.
        /// </summary>
        public virtual (Option<TElement1>, Option<TElement2>) Point
        {
            get
            {
                if (!this.IsPointMass)
                {
                    throw new InvalidOperationException("This distribution is not a point mass.");
                }

                return (this.First.Value.Point, this.Second.Value.Point);
            }
        }

        /// <inheritdoc/>
        public virtual TThis CreatePointMass((Option<TElement1>, Option<TElement2>) point) => new TThis()
        {
            First = ElementDistribution1Factory.CreatePointMass(point.Item1.Value),
            Second = ElementDistribution2Factory.CreatePointMass(point.Item2.Value)
        };

        /// <summary>
        /// Creates a pair distribution for an epsilon output transducer transition.
        /// </summary>
        /// <param name="first">The element distribution to weight the input element.</param>
        /// <returns>The created distribution.</returns>
        public static TThis FromFirst(Option<TElementDistribution1> first) =>
            first.HasValue ? new TThis { First = first } : null;

        /// <summary>
        /// Creates a pair distribution for an epsilon input transducer transition.
        /// </summary>
        /// <param name="second">The element distribution to weight the output element.</param>
        /// <returns>The created distribution.</returns>
        public static TThis FromSecond(Option<TElementDistribution2> second) =>
            second.HasValue ? new TThis { Second = second } : null;

        /// <summary>
        /// Creates a pair distribution for a transducer transition.
        /// </summary>
        /// <param name="first">The element distribution to weight the input element.</param>
        /// <param name="second">The element distribution to weight the output element.</param>
        /// <remarks>
        /// One of <paramref name="first"/> and <paramref name="second"/> can be <see langword="null"/>
        /// to encode an epsilon input or an epsilon output transducer transition, but not both.
        /// </remarks>
        /// <returns>The created distribution.</returns>
        public static TThis FromFirstSecond(Option<TElementDistribution1> first, Option<TElementDistribution2> second)
        {
            Argument.CheckIfValid(
                first.HasValue || second.HasValue,
                "At least one of the distributions must not be null.");
            
            return new TThis { First = first, Second = second };
        }

        /// <summary>
        /// Computes <c>P(y) = R(x, y)</c>, where <c>R(x, y)</c> is the current pair distribution,
        /// and <c>x</c> is a given element.
        /// </summary>
        /// <param name="first">The element to project.</param>
        /// <param name="result">The normalized projection result.</param>
        /// <returns>The logarithm of the scale for the projection result.</returns>
        public virtual double ProjectFirst(TElement1 first, out Option<TElementDistribution2> result)
        {
            if (!this.First.HasValue)
            {
                throw new InvalidOperationException("Cannot project on an epsilon transition.");
            }

            var logAverageOf = this.First.Value.GetLogProb(first);
            result = double.IsNegativeInfinity(logAverageOf) ? Option.None: this.Second;
            return logAverageOf;
        }

        /// <summary>
        /// Computes <c>P(y) = sum_x Q(x) R(x, y)</c>, where <c>R(x, y)</c> is the current pair distribution,
        /// and <c>Q(x)</c> is a given element distribution.
        /// </summary>
        /// <param name="first">The element distribution to project.</param>
        /// <param name="result">The normalized projection result.</param>
        /// <returns>The logarithm of the scale for the projection result.</returns>
        public virtual double ProjectFirst(TElementDistribution1 first, out Option<TElementDistribution2> result)
        {
            if (!this.First.HasValue)
            {
                throw new InvalidOperationException("Cannot project on an epsilon transition.");
            }
            
            var logAverageOf = first.GetLogAverageOf(this.First.Value);
            result = double.IsNegativeInfinity(logAverageOf) ? new TElementDistribution2() : this.Second;
            return logAverageOf;
        }

        /// <inheritdoc/>
        public virtual bool IsUniform() => this.First.Value.IsUniform() && this.Second.Value.IsUniform();

        /// <inheritdoc/>
        public virtual TThis CreateUniform() => new TThis()
        {
            First = ElementDistribution1Factory.CreateUniform(),
            Second = ElementDistribution2Factory.CreateUniform()
        };

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <returns>
        /// A string that represents the distribution.
        /// </returns>
        public override string ToString()
        {
            if (!this.First.HasValue)
            {
                return ">" + this.Second;
            }

            if (!this.Second.HasValue)
            {
                return this.First + ">";
            }

            return this.First + ">" + this.Second;
        }

        /// <summary>
        /// Creates a copy of the current distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public virtual object Clone() => this; // The type is immutable

        /// <inheritdoc/>
        public abstract double MaxDiff(object that);

        /// <inheritdoc/>
        public abstract double GetLogProb((Option<TElement1>, Option<TElement2>) pair);

        /// <inheritdoc/>
        public virtual double GetLogAverageOf(TThis that)
        {
            Argument.CheckIfValid(
                this.First.HasValue == that.First.HasValue,
                "The first element distribution must be either present or absent in both the argument and the current distribution.");
            Argument.CheckIfValid(
                this.Second.HasValue == that.Second.HasValue,
                "The first element distribution must be either present or absent in both the argument and the current distribution.");
            
            double result = 0;

            if (this.First.HasValue)
            {
                result += this.First.Value.GetLogAverageOf(that.First.Value);
            }

            if (this.Second.HasValue)
            {
                result += this.Second.Value.GetLogAverageOf(that.Second.Value);
            }

            return result;
        }

        /// <inheritdoc/>
        public abstract TThis Multiply(TThis other);

        /// <inheritdoc/>
        public abstract TThis Sum(double weightThis, TThis other, double weightOther);

        /// <inheritdoc/>
        public virtual TThis CreatePartialUniform() => new TThis()
        {
            First = First.HasValue ? Option.Some(First.Value.CreatePartialUniform()) : Option.None,
            Second = Second.HasValue ? Option.Some(Second.Value.CreatePartialUniform()) : Option.None
        };

        /// <inheritdoc/>
        public virtual bool IsPartialUniform()
        {
            return
                (!this.First.HasValue || this.First.Value.IsPartialUniform()) &&
                (!this.Second.HasValue || this.Second.Value.IsPartialUniform());
        }
    }
}