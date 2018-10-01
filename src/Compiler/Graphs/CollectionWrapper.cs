// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    // This class makes it easy to create custom List wrappers, by overriding
    // the appropriate methods.
    internal class CollectionWrapper<T, ListType> : ICollection<T>
        where ListType : ICollection<T>
    {
        protected ListType list;

        protected CollectionWrapper()
        {
        }

        public CollectionWrapper(ListType list)
        {
            this.list = list;
        }

        #region IEnumerable methods

        public virtual IEnumerator<T> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region ICollection methods

        public virtual int Count
        {
            get { return list.Count; }
        }

        public virtual bool IsReadOnly
        {
            get { return list.IsReadOnly; }
        }

        public virtual void CopyTo(T[] array, int index)
        {
            list.CopyTo(array, index);
        }

        public virtual void Add(T item)
        {
            list.Add(item);
        }

        public virtual void Clear()
        {
            list.Clear();
        }

        public virtual bool Contains(T item)
        {
            return list.Contains(item);
        }

        public virtual bool Remove(T item)
        {
            return list.Remove(item);
        }

        #endregion
    }
}