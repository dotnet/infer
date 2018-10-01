// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    /// <summary>
    /// A set of functions to perform equality comparison of complex types.
    /// </summary>
    internal static class Comparers
    {
        /// <summary>
        /// Checks if two dictionaries are equal.
        /// </summary>
        /// <typeparam name="TKey">The type of the key in the dictionary.</typeparam>
        /// <typeparam name="TValue">The type of the value in the dictionary.</typeparam>
        /// <param name="dictionary1">The first dictionary.</param>
        /// <param name="dictionary2">The second dictionary.</param>
        /// <param name="valueComparer">The comparer of dictionary values.</param>
        /// <returns>True if the given dictionaries are equal, false otherwise.</returns>
        public static bool Dictionary<TKey, TValue>(
            IDictionary<TKey, TValue> dictionary1,
            IDictionary<TKey, TValue> dictionary2,
            Func<TValue, TValue, bool> valueComparer)
        {
            Debug.Assert(dictionary1 != null && dictionary2 != null, "Valid dictionaries should be provided.");

            if (dictionary1.Count != dictionary2.Count)
            {
                return false;
            }

            return dictionary1.Keys.All(
                key => dictionary2.ContainsKey(key) && valueComparer(dictionary1[key], dictionary2[key]));
        }

        /// <summary>
        /// Checks if two sets are equal.
        /// </summary>
        /// <typeparam name="T">The type of set elements.</typeparam>
        /// <param name="set1">The first set.</param>
        /// <param name="set2">The second set.</param>
        /// <returns>True if the given sets are equal, false otherwise.</returns>
        public static bool Set<T>(ISet<T> set1, ISet<T> set2)
        {
            Debug.Assert(set1 != null && set2 != null, "Valid sets should be provided.");

            if (set1.Count != set2.Count)
            {
                return false;
            }

            return set1.Any(v => set2.Contains(v));
        }

        /// <summary>
        /// Checks if two collections are equal, that is, contain the same elements in the same order.
        /// </summary>
        /// <typeparam name="T">The type of collection elements.</typeparam>
        /// <param name="collection1">The first collection.</param>
        /// <param name="collection2">The second collection.</param>
        /// <returns>True if the given collections are equal, false otherwise.</returns>
        public static bool Collection<T>(IEnumerable<T> collection1, IEnumerable<T> collection2)
        {
            Debug.Assert(collection1 != null && collection2 != null, "Valid collections should be provided.");

            return collection1.SequenceEqual(collection2);
        }

        /// <summary>
        /// Checks if two collections are equivalent, that is, contain the same elements.
        /// </summary>
        /// <typeparam name="T">The type of collection elements.</typeparam>
        /// <param name="collection1">The first collection.</param>
        /// <param name="collection2">The second collection.</param>
        /// <returns>True if the given collections are equivalent, false otherwise.</returns>
        public static bool CollectionEquivalence<T>(IEnumerable<T> collection1, IEnumerable<T> collection2)
        {
            Debug.Assert(collection1 != null && collection2 != null, "Valid collections should be provided.");

            ISet<T> set1 = new HashSet<T>(collection1);
            ISet<T> set2 = new HashSet<T>(collection2);
            return Set(set1, set2);
        }

        /// <summary>
        /// Checks if two values are equal.
        /// </summary>
        /// <typeparam name="T">The type of the value.</typeparam>
        /// <param name="value1">The first value.</param>
        /// <param name="value2">The second value.</param>
        /// <returns>True if the given values are equal, false otherwise.</returns>
        public static bool Generic<T>(T value1, T value2)
        {
            return object.Equals(value1, value2);
        }
    }
}
