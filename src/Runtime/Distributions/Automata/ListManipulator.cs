// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Provides the ability to manipulate lists (classes that implement <see cref="IList{T}"/>).
    /// </summary>
    /// <typeparam name="TList">The type of a list.</typeparam>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    public class ListManipulator<TList, TElement> : ISequenceManipulator<TList, TElement>
        where TList : class, IList<TElement>, new()
    {
        /// <summary>
        /// Converts a given collection of elements to a list.
        /// </summary>
        /// <param name="elements">The collection of elements to convert to a sequence.</param>
        /// <returns>The list containing the elements.</returns>
        public TList ToSequence(IEnumerable<TElement> elements)
        {
            var result = new TList();
            result.AddRange(elements);
            return result;
        }

        /// <summary>
        /// Gets the length of a given list.
        /// </summary>
        /// <param name="sequence">The list.</param>
        /// <returns>The length of the list.</returns>
        public int GetLength(TList sequence) => sequence.Count;

        /// <summary>
        /// Gets the element at a given position in a given list.
        /// </summary>
        /// <param name="sequence">The list.</param>
        /// <param name="index">The position.</param>
        /// <returns>The element at the given position in the list.</returns>
        public TElement GetElement(TList sequence, int index) => sequence[index];

        /// <summary>
        /// Checks if given lists are equal.
        /// Lists are considered equal if they contain the same elements in the same order.
        /// </summary>
        /// <param name="sequence1">The first list.</param>
        /// <param name="sequence2">The second list.</param>
        /// <returns><see langword="true"/> if the lists are equal, <see langword="false"/> otherwise.</returns>
        public bool SequencesAreEqual(TList sequence1, TList sequence2) => Util.ValueEquals(sequence1, sequence2);

        /// <summary>
        /// Creates a list by copying the first list and then appending the second list to it.
        /// </summary>
        /// <param name="sequence1">The first list.</param>
        /// <param name="sequence2">The second list.</param>
        /// <returns>The created list.</returns>
        public TList Concat(TList sequence1, TList sequence2)
        {
            var result = new TList();
            result.AddRange(sequence1);
            result.AddRange(sequence2);
            return result;
        }
    }
}