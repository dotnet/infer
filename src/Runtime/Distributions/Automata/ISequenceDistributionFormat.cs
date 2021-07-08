// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// An interface for classes implementing various methods of representing sequence distributions as strings.
    /// </summary>
    public interface ISequenceDistributionFormat
    {
        /// <summary>
        /// Converts a given sequence distribution to a string.
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
        string ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>(
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
