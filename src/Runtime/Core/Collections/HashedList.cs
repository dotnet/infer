// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Hash-indexed list.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <remarks>
    /// The user can supply a list and it will be automatically indexed.
    /// However, only changes made through this interface will be indexed properly.
    /// </remarks>
    public class HashedList<T> : IEnumerable<T>, IEnumerable //: IList<T>
    {
        private IList<T> list;
        private Dictionary<T, int> dictionary;

        public T this[int index]
        {
            get { return list[index]; }
            set
            {
                dictionary[value] = index;
                list[index] = value;
            }
        }

        public int IndexOf(T item)
        {
            return dictionary[item];
        }

        public void Insert(int index, T item)
        {
            throw new NotSupportedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotSupportedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public HashedList()
        {
            list = new List<T>();
            dictionary = new Dictionary<T, int>();
        }

        public HashedList(IList<T> list)
        {
            this.list = list;
            // FIXME
            dictionary = new Dictionary<T, int>();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}