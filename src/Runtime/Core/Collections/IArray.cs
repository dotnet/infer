// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Interface to an array of arbitrary rank.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IArray<T> : IList<T>
    {
        /// <summary>
        /// Get the number of dimensions of the array.
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// Get the size of a specified dimension of a multidimensional array.
        /// </summary>
        /// <param name="dimension">Zero-based dimension of the array.</param>
        /// <returns>The size of the specified dimension of the array.</returns>
        int GetLength(int dimension);

#if false
        /// <summary>
        /// Get or set an element of a multidimensional array.
        /// </summary>
        /// <param name="indices">Zero-based indices into the multidimensional array.</param>
        /// <returns>The element at the specified position.</returns>
        T this[params int[] indices] { get; set; }
#endif
    }

    /// <summary>
    /// Interface to a two-dimensional array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IArray2D<T> : IArray<T>
    {
        /// <summary>
        /// Get or set an element of a two-dimensional array.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        T this[int row, int column] { get; set; }
    }

    /// <summary>
    /// Interface to an object providing a constructor for new arrays.
    /// </summary>
    /// <typeparam name="ItemType"></typeparam>
    /// <typeparam name="ArrayType"></typeparam>
    public interface IArrayFactory<ItemType, ArrayType>
    {
        ArrayType CreateArray(int length, Func<int, ItemType> init);
    }
}