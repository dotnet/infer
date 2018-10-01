// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System.Collections;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;

    /// <summary>
    /// A faster alternative to <see cref="ReadOnlyCollection{T}"/>
    /// which allows calls to the underlying list to be inlined.
    /// </summary>
    /// <typeparam name="T">The type of a list element.</typeparam>
    public class ReadOnlyList<T> : IEnumerable<T>
    {
        /// <summary>
        /// The wrapped list.
        /// </summary>
        private readonly List<T> list;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadOnlyList{T}"/> class.
        /// </summary>
        /// <param name="list">The wrapped list.</param>
        public ReadOnlyList(List<T> list)
        {
            this.list = list;
        }

        /// <summary>
        /// Gets the number of elements in the list.
        /// </summary>
        public int Count
        {
            get
            {
                return this.list.Count;
            }
        }

        /// <summary>
        /// Gets the element with a specified index form the list.
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <returns>The element with the given index.</returns>
        public T this[int index]
        {
            get { return this.list[index]; }
        }

        /// <summary>
        /// Returns an enumerator that iterates through the list.
        /// </summary>
        /// <returns>
        /// An enumerator that iterates through the list.
        /// </returns>
        public IEnumerator<T> GetEnumerator()
        {
            return this.list.GetEnumerator();
        }

        /// <summary>
        /// Returns an enumerator that iterates through the list.
        /// </summary>
        /// <returns>
        /// An enumerator that iterates through the list.
        /// </returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}
