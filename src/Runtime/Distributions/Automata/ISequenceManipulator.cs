// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    /// <summary>
    /// An interface for classes that provide functionality to manipulate sequences of a certain type.
    /// </summary>
    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The type of a sequence element.</typeparam>
    public interface ISequenceManipulator<TSequence, TElement>
        where TSequence : class, IEnumerable<TElement>
    {
        IEqualityComparer<TSequence> SequenceEqualityComparer { get; }

        /// <summary>
        /// Converts a given collection of elements to a sequence.
        /// </summary>
        /// <param name="elements">The collection of elements to convert to a sequence.</param>
        /// <returns>The sequence containing the elements.</returns>
        TSequence ToSequence(IEnumerable<TElement> elements);

        /// <summary>
        /// Gets the length of a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <returns>The length of the sequence.</returns>
        int GetLength(TSequence sequence);

        /// <summary>
        /// Gets the element at a given position in a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="index">The position.</param>
        /// <returns>The element at the given position in the sequence.</returns>
        TElement GetElement(TSequence sequence, int index);

        /// <summary>
        /// Creates a sequence by copying the first sequence and then appending the second sequence to it.
        /// </summary>
        /// <param name="sequence1">The first sequence.</param>
        /// <param name="sequence2">The second sequence.</param>
        /// <returns>The created sequence.</returns>
        TSequence Concat(TSequence sequence1, TSequence sequence2);
    }
}
