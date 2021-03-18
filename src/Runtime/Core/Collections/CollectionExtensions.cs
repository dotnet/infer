// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Extension methods for ICollection
    /// </summary>
    public static class CollectionExtensions
    {
        /// <summary>
        /// Add multiple items to a collection
        /// </summary>
        /// <typeparam name="T">The item type</typeparam>
        /// <param name="collection">The collection to add to</param>
        /// <param name="items">The items to add</param>
        public static void AddRange<T>(this ICollection<T> collection, IEnumerable<T> items)
        {
            foreach (T item in items) collection.Add(item);
        }

        /// <summary>
        /// Test if a collection contains multiple items
        /// </summary>
        /// <typeparam name="T">The item type</typeparam>
        /// <param name="collection">The collection</param>
        /// <param name="items">The items to search for</param>
        /// <returns>True if the collection contains all items.</returns>
        public static bool ContainsAll<T>(this ICollection<T> collection, IEnumerable<T> items)
        {
            foreach (T item in items) if (!collection.Contains(item)) return false;
            return true;
        }

        /// <summary>
        /// Test if a collection contains any of multiple items
        /// </summary>
        /// <typeparam name="T">The item type</typeparam>
        /// <param name="collection">The collection</param>
        /// <param name="items">The items to search for</param>
        /// <returns>True if the collection contains any item in items.</returns>
        public static bool ContainsAny<T>(this ICollection<T> collection, IEnumerable<T> items)
        {
            foreach (T item in items) if (collection.Contains(item)) return true;
            return false;
        }
    }
}