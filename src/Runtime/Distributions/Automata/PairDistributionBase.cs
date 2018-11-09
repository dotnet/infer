// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Core.Collections;
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
    public abstract class PairDistributionBase<TElement1, TElementDistribution1, TElement2, TElementDistribution2, TThis> :
        IDistribution<Pair<Option<TElement1>, Option<TElement2>>>, CanGetLogAverageOf<TThis>, SettableToProduct<TThis>, SettableToWeightedSumExact<TThis>, SettableToPartialUniform<TThis>
        where TElementDistribution1 : IDistribution<TElement1>, CanGetLogAverageOf<TElementDistribution1>, SettableToProduct<TElementDistribution1>, SettableToPartialUniform<TElementDistribution1>, new()
        where TElementDistribution2 : IDistribution<TElement2>, CanGetLogAverageOf<TElementDistribution2>, SettableToProduct<TElementDistribution2>, SettableToPartialUniform<TElementDistribution2>, new()
        where TThis : PairDistributionBase<TElement1, TElementDistribution1, TElement2, TElementDistribution2, TThis>, new()
    {
        /// <summary>
        /// Gets or sets the marginal distribution over the first element in a pair.
        /// </summary>
        public Option<TElementDistribution1> First { get; protected set; }

        /// <summary>
        /// Gets or sets the marginal distribution over the second element in a pair.
        /// </summary>
        public Option<TElementDistribution2> Second { get; protected set; }

        /// <summary>
        /// Gets a value indicating whether the current distribution represents a point mass.
        /// </summary>
        public bool IsPointMass =>
            this.First.HasValue && this.First.Value.IsPointMass &&
            this.Second.HasValue && this.Second.Value.IsPointMass;

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public virtual Pair<Option<TElement1>, Option<TElement2>> Point
        {
            get
            {
                if (!this.IsPointMass)
                {
                    throw new InvalidOperationException("This distribution is not a point mass.");
                }

                return new Pair<Option<TElement1>, Option<TElement2>>(this.First.Value.Point, this.Second.Value.Point);
            }

            set
            {
                this.First = new TElementDistribution1 { Point = value.First.Value };
                this.Second = new TElementDistribution2 { Point = value.Second.Value };
            }
        }

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

        /// <summary>
        /// Checks whether the current distribution is uniform.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is uniform, <see langword="false"/> otherwise.</returns>
        public virtual bool IsUniform() => this.First.Value.IsUniform() && this.Second.Value.IsUniform();

        /// <summary>
        /// Replaces the current distribution with a uniform distribution.
        /// </summary>
        public virtual void SetToUniform()
        {
            this.First = Distribution.CreateUniform<TElementDistribution1>();
            this.Second = Distribution.CreateUniform<TElementDistribution2>();
        }

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
        public virtual object Clone()
        {
            return new TThis
            {
                First =
                    this.First.HasValue
                        ? Option.Some((TElementDistribution1)this.First.Value.Clone())
                        : Option.None,
                Second =
                    this.Second.HasValue
                        ? Option.Some((TElementDistribution2)this.Second.Value.Clone())
                        : Option.None,
            };
        }

        /// <summary>
        /// Gets the maximum difference between the parameters of this distribution and a given one.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>The maximum difference.</returns>
        /// <remarks>Not currently implemented.</remarks>
        public virtual double MaxDiff(object that)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Gets the logarithm of the probability of a given element pair under this distribution.
        /// </summary>
        /// <param name="pair">The pair to get the probability for.</param>
        /// <returns>The logarithm of the probability of the pair.</returns>
        /// <remarks>Not currently implemented.</remarks>
        public virtual double GetLogProb(Pair<Option<TElement1>, Option<TElement2>> pair)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the logarithm of the probability that the current distribution would draw the same sample
        /// as a given one.
        /// </summary>
        /// <param name="that">The given distribution.</param>
        /// <returns>The logarithm of the probability that distributions would draw the same sample.</returns>
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

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions.
        /// </summary>
        /// <param name="distribution1">The first distribution.</param>
        /// <param name="distribution2">The second distribution.</param>
        /// <remarks>Not currently implemented.</remarks>
        public virtual void SetToProduct(TThis distribution1, TThis distribution2)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="distribution1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="distribution2">The second distribution.</param>
        /// <remarks>Not currently implemented.</remarks>
        public virtual void SetToSum(double weight1, TThis distribution1, double weight2, TThis distribution2)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        public virtual void SetToPartialUniform()
        {
            this.SetToPartialUniformOf((TThis)this);
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="distribution">The distribution which support will be used to setup the current distribution.</param>
        public virtual void SetToPartialUniformOf(TThis distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");

            this.First =
                distribution.First.HasValue
                    ? Option.Some(Distribution.CreatePartialUniform(distribution.First.Value))
                    : Option.None;
            this.Second =
                distribution.Second.HasValue
                    ? Option.Some(Distribution.CreatePartialUniform(distribution.Second.Value))
                    : Option.None;
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution is uniform over its support, <see langword="false"/> otherwise.</returns>
        public virtual bool IsPartialUniform()
        {
            return
                (!this.First.HasValue || this.First.Value.IsPartialUniform()) &&
                (!this.Second.HasValue || this.Second.Value.IsPartialUniform());
        }
    }
}