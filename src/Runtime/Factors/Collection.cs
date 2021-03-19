// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Attributes;
    using Utilities;

    /// <summary>
    /// Contains commonly used factor methods.
    /// </summary>
    [Quality(QualityBand.Stable)]
    public static class Collection
    {
        /// <summary>
        /// Get an element of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="index">The index to get</param>
        /// <returns>The item</returns>
        //[ParameterNames("item", "array", "index")]
        //public static T GetItem<T>(T[] array, int index) { return array[index]; }
        [Hidden]
        [ParameterNames("item", "array", "index")]
        public static T GetItem<T>(IReadOnlyList<T> array, int index)
        {
            return array[index];
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

        /// <summary>
        /// Get multiple elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items we want to get</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices")]
        public static T[][] GetJaggedItems<T>(IReadOnlyList<T> array, IReadOnlyList<IReadOnlyList<int>> indices)
        {
            return Util.ArrayInit(indices.Count, i =>
            {
                var indices_i = indices[i];
                return Util.ArrayInit(indices_i.Count, j =>
                    array[indices_i[j]]);
            });
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
        public static T[][][] GetDeepJaggedItems<T>(IReadOnlyList<T> array, IReadOnlyList<IReadOnlyList<IReadOnlyList<int>>> indices)
        {
            return Util.ArrayInit(indices.Count, i =>
            {
                var indices_i = indices[i];
                return Util.ArrayInit(indices_i.Count, j =>
                {
                    var indices_i_j = indices_i[j];
                    return Util.ArrayInit(indices_i_j.Count, k =>
                        array[indices_i_j[k]]);
                });
            });
        }

        /// <summary>
        /// Get multiple elements of a jagged array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of depth 1 indices for items we want to get</param>
        /// <param name="indices2">Array of depth 2 indices for items we want to get</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices", "indices2")]
        public static T[] GetItemsFromJagged<T>(IReadOnlyList<IReadOnlyList<T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[indices[i]][indices2[i]];
            }
            return result;
        }

        /// <summary>
        /// Get multiple elements of a jagged array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of depth 1 indices for items we want to get</param>
        /// <param name="indices2">Array of depth 2 indices for items we want to get</param>
        /// <param name="indices3">Array of depth 3 indices for items we want to get</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices", "indices2", "indices3")]
        public static T[] GetItemsFromDeepJagged<T>(IReadOnlyList<IReadOnlyList<IReadOnlyList<T>>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, IReadOnlyList<int> indices3)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[indices[i]][indices2[i]][indices3[i]];
            }
            return result;
        }

        /// <summary>
        /// Get multiple elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of depth 1 indices for items we want to get</param>
        /// <param name="indices2">Array of depth 2 indices for items we want to get</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices", "indices2")]
        public static T[][] GetJaggedItemsFromJagged<T>(IReadOnlyList<IReadOnlyList<T>> array, IReadOnlyList<IReadOnlyList<int>> indices, IReadOnlyList<IReadOnlyList<int>> indices2)
        {
            Assert.IsTrue(indices.Count == indices2.Count);
            return Util.ArrayInit(indices.Count, i =>
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                Assert.IsTrue(indices_i.Count == indices2_i.Count);
                return Util.ArrayInit(indices_i.Count, j =>
                    array[indices_i[j]][indices2_i[j]]);
            });
        }

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items we want to get.  Must all be different.</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices")]
        public static T[] Subarray<T>(IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            return GetItems(array, indices);
        }

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items to get.  Indices must all be different.</param>
        /// <param name="array2"></param>
        /// <returns>The items</returns>
        [ParameterNames("items", "array", "indices", "array2")]
        [Hidden]
        public static T[] Subarray2<T>(IReadOnlyList<T> array, IReadOnlyList<int> indices, IReadOnlyList<T> array2)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                if (indices[i] < 0)
                    result[i] = array2[indices[i]];
                else
                    result[i] = array[indices[i]];
            }
            return result;
        }

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items to get.  indices[i][j] must be different for different j and same i, but can match for different i.</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices")]
        public static T[][] JaggedSubarray<T>(IReadOnlyList<T> array, int[][] indices)
        {
            T[][] result = new T[indices.Length][];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = GetItems(array, indices[i]);
            }
            return result;
        }

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items to get.  indices[i][j] must be different for different j and same i, but can match for different i.</param>
        /// <param name="marginal"></param>
        /// <returns>The items</returns>
        [ParameterNames("items", "array", "indices", "marginal")]
        [Hidden]
        public static T[][] JaggedSubarrayWithMarginal<T>(IReadOnlyList<T> array, int[][] indices, out IReadOnlyList<T> marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Jagged array containing the indices of the elements to get.  The indices must all be different.  The shape of this array determines the shape of the result.</param>
        /// <returns>The items</returns>
        [Hidden]
        [ParameterNames("items", "array", "indices")]
        public static T[][] SplitSubarray<T>(IReadOnlyList<T> array, int[][] indices)
        {
            T[][] result = new T[indices.Length][];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = GetItems(array, indices[i]);
            }
            return result;
        }

        [Hidden]
        [ParameterNames("head", "array", "count", "tail")]
        public static T[] Split<T>(IReadOnlyList<T> array, int count, out T[] tail)
        {
            T[] head = new T[count];
            for (int i = 0; i < count; i++)
            {
                head[i] = array[i];
            }
            tail = new T[array.Count - count];
            for (int i = count; i < array.Count; i++)
            {
                tail[i - count] = array[i];
            }
            return head;
        }

        /// <summary>
        /// Get an element of a 2D array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="index1">The first index</param>
        /// <param name="index2">The second index</param>
        /// <returns>The item</returns>
        [Hidden]
        [ParameterNames("item", "array", "index1", "index2")]
        public static T GetItem2D<T>(T[,] array, int index1, int index2)
        {
            return array[index1, index2];
        }
    }
}
