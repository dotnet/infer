// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Distributions;
    using Math;
    using Attributes;
    using Utilities;

    /// <summary>
    /// Contains commonly used factor methods.
    /// </summary>
    [Quality(QualityBand.Stable)]
    public static class IndexingFactor
    {

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items we want to get.  Must all be different.</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices")]
        // Note the signature cannot use IReadOnlyList since IList does not implement IReadOnlyList
        public static T[] Subarray<T>(IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            return GetItems(array, indices);
        }

        /// <summary>
        /// Get multiple elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items we want to get</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices")]
        public static T[] GetItems<T>(IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[indices[i]];
            }
            return result;
        }
    }
}
