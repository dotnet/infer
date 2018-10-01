// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    // The resulting ICollection is read-only.
    internal class JoinCollections<T> : ICollection<T>
    {
        private ICollection<ICollection<T>> collections;

        public JoinCollections(ICollection<ICollection<T>> collections)
        {
            this.collections = collections;
        }

        public JoinCollections(params ICollection<T>[] collections)
        {
            this.collections = collections;
        }

        #region IEnumerable<T> methods

        public IEnumerator<T> GetEnumerator()
        {
            foreach (ICollection<T> collection in collections)
            {
                foreach (T value in collection)
                {
                    yield return value;
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region ICollection<T> methods

        public int Count
        {
            get
            {
                int count = 0;
                foreach (ICollection<T> collection in collections)
                {
                    count += collection.Count;
                }
                return count;
            }
        }

        public bool IsReadOnly
        {
            get { return true; }
        }

        public void Add(T item)
        {
            throw new NotSupportedException();
        }

        public void Clear()
        {
            throw new NotSupportedException();
        }

        public bool Contains(T item)
        {
            foreach (ICollection<T> collection in collections)
            {
                if (collection.Contains(item)) return true;
            }
            return false;
        }

        public void CopyTo(T[] array, int index)
        {
            foreach (ICollection<T> collection in collections)
            {
                collection.CopyTo(array, index);
                index += collection.Count;
            }
        }

        public bool Remove(T item)
        {
            throw new NotSupportedException();
        }

        #endregion
    }
}