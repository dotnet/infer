// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// An interface for classes implementing various methods of representing automata as strings.
    /// </summary>
    public interface IAutomatonFormat
    {
        /// <summary>
        /// Converts a given automaton to a string.
        /// </summary>
        /// <typeparam name="TSequence">The type of sequences <paramref name="automaton"/> is defined on.</typeparam>
        /// <typeparam name="TElement">The type of sequence elements of <paramref name="automaton"/>.</typeparam>
        /// <typeparam name="TElementDistribution">The type of distributions over sequence elements of <paramref name="automaton"/>.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate instances of <typeparamref name="TSequence"/>.</typeparam>
        /// <typeparam name="TAutomaton">The concrete type of <paramref name="automaton"/>.</typeparam>
        /// <param name="automaton">The automaton to convert to a string.</param>
        /// <returns>The string representation of <paramref name="automaton"/>.</returns>
        string ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>(
            Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> automaton)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>,
                SettableToPartialUniform<TElementDistribution>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new();

        /// <summary>
        /// Escapes a raw string, such that potential special characters can be represented in a way that the target can handle.
        /// </summary>
        /// <param name="rawString">Raw generated string.</param>
        /// <returns>An escaped string.</returns>
        string Escape(string rawString);
    }
}