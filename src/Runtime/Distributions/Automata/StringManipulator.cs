// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Provides the ability to manipulate strings.
    /// </summary>
    [Serializable]
    public class StringManipulator : ISequenceManipulator<string, char>
    {
        public IEqualityComparer<string> SequenceEqualityComparer { get; } =
            StringComparer.Ordinal;

        /// <summary>
        /// Converts a given sequence of characters to a string.
        /// </summary>
        /// <param name="elements">The sequence of characters.</param>
        /// <returns>The string.</returns>
        public string ToSequence(IEnumerable<char> elements)
        {
            return new string(elements.ToArray());
        }

        /// <summary>
        /// Gets the length of a given string.
        /// </summary>
        /// <param name="sequence">The string.</param>
        /// <returns>The length of the string.</returns>
        public int GetLength(string sequence)
        {
            return sequence.Length;
        }

        /// <summary>
        /// Gets the character at a given position in a given string.
        /// </summary>
        /// <param name="sequence">The string.</param>
        /// <param name="index">The position.</param>
        /// <returns>The character at the given position in the string.</returns>
        public char GetElement(string sequence, int index)
        {
            return sequence[index];
        }

        /// <summary>
        /// Creates a string by copying the first string and then appending the second string to it.
        /// </summary>
        /// <param name="sequence1">The first string.</param>
        /// <param name="sequence2">The second string.</param>
        /// <returns>The created string.</returns>
        public string Concat(string sequence1, string sequence2)
        {
            return sequence1 + sequence2;
        }
    }
}