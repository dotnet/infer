// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// A base class for implementations of <see cref="ISequenceDistributionFormat"/> that use
    /// <see cref="IAutomatonFormat"/>.
    /// </summary>
    public abstract class AutomatonBasedSequenceDistributionFormatBase : ISequenceDistributionFormat
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AutomatonBasedSequenceDistributionFormatBase"/> class.
        /// </summary>
        /// <param name="automatonFormat">The underlying automaton format.</param>
        protected AutomatonBasedSequenceDistributionFormatBase(IAutomatonFormat automatonFormat)
        {
            Argument.CheckIfNotNull(automatonFormat, nameof(automatonFormat));

            this.AutomatonFormat = automatonFormat;
        }
        
        /// <summary>
        /// Gets the underlying automaton function format.
        /// </summary>
        public IAutomatonFormat AutomatonFormat { get; private set; }

        /// <summary>
        /// Uses <see cref="AutomatonFormat"/> on the weight function of <paramref name="sequenceDistribution"/>,
        /// or calls <see cref="ConvertPointMassToString{TSequence,TElement,TElementDistribution,TSequenceManipulator,TAutomaton,TWeightFunction,TWeightFunctionFactory,TSequenceDistribution}"/> if the distribution is a point mass.
        /// </summary>
        /// <typeparam name="TSequence">The type of sequences <paramref name="sequenceDistribution"/> is defined on.</typeparam>
        /// <typeparam name="TElement">The type of sequence elements of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TElementDistribution">The type of distributions over sequence elements of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate instances of <typeparamref name="TSequence"/>.</typeparam>
        /// <typeparam name="TAutomaton">The type of automata used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TWeightFunction">The type of weight functions used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TWeightFunctionFactory">The type of weight function factory used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TSequenceDistribution">The concrete type of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <param name="sequenceDistribution">The sequence distribution to convert to string.</param>
        /// <returns>The string representation of the <paramref name="sequenceDistribution"/>.</returns>
        public string ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>(
            TSequenceDistribution sequenceDistribution)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, Sampleable<TElement>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
            where TWeightFunction : WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.IWeightFunction<TWeightFunction>, new()
            where TWeightFunctionFactory : WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.IWeightFunctionFactory<TWeightFunction>, new()
            where TSequenceDistribution : SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>, new()
        {
            Argument.CheckIfNotNull(sequenceDistribution, nameof(sequenceDistribution));

            if (sequenceDistribution.IsPointMass)
            {
                return this.ConvertPointMassToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>(sequenceDistribution);
            }

            return this.AutomatonFormat.ConvertToString(sequenceDistribution.ToAutomaton());
        }

        /// <summary>
        /// Overridden in the derived classes to convert a point mass sequence distribution to a string.
        /// </summary>
        /// <typeparam name="TSequence">The type of sequences <paramref name="sequenceDistribution"/> is defined on.</typeparam>
        /// <typeparam name="TElement">The type of sequence elements of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TElementDistribution">The type of distributions over sequence elements of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate instances of <typeparamref name="TSequence"/>.</typeparam>
        /// <typeparam name="TAutomaton">The type of automata used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TWeightFunction">The type of weight functions used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TWeightFunctionFactory">The type of weight function factory used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TSequenceDistribution">The concrete type of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <param name="sequenceDistribution">The sequence distribution to convert to string.</param>
        /// <returns>The string representation of the <paramref name="sequenceDistribution"/>.</returns>
        protected abstract string ConvertPointMassToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>(
            TSequenceDistribution sequenceDistribution)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, Sampleable<TElement>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
            where TWeightFunction : WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.IWeightFunction<TWeightFunction>, new()
            where TWeightFunctionFactory : WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.IWeightFunctionFactory<TWeightFunction>, new()
            where TSequenceDistribution : SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>, new();
    }
}