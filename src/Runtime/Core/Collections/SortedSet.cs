// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// A sorted collection of unique items.
    /// </summary>
    /// <typeparam name="T">The item type.</typeparam>
    /// <remarks>
    /// A sorted collection of items, all of which are different according
    /// to Equals.  null items are not allowed.  Adding a duplicate has no effect.
    /// Union, intersection, and superset are all supported via operator overloading.
    /// The items are stored in the keys of a SortedList, which does the sorting
    /// via its own comparer function.
    /// </remarks>
    public class SortedSet<T> : IList<T>, IReadOnlyList<T>, ICloneable
    {
        protected struct EmptyStruct
        {
        }

        private SortedList<T, EmptyStruct> dict;

        public SortedSet()
        {
            dict = new SortedList<T, EmptyStruct>();
        }

        public SortedSet(IComparer<T> comparer)
        {
            dict = new SortedList<T, EmptyStruct>(comparer);
        }

        public SortedSet(int capacity, IComparer<T> comparer)
        {
            dict = new SortedList<T, EmptyStruct>(capacity, comparer);
        }

        public static SortedSet<T> FromEnumerable(IEnumerable<T> collection)
        {
            SortedSet<T> result = new SortedSet<T>();
            result.AddRange(collection);
            return result;
        }

        public static SortedSet<T> FromEnumerable(IComparer<T> comparer, IEnumerable<T> collection)
        {
            SortedSet<T> result = new SortedSet<T>(comparer);
            result.AddRange(collection);
            return result;
        }

        public IComparer<T> Comparer
        {
            get { return dict.Comparer; }
        }

        public int Capacity
        {
            get { return dict.Capacity; }
            set { dict.Capacity = value; }
        }

        #region IEnumerable methods

        public virtual IEnumerator<T> GetEnumerator()
        {
            foreach (T item in dict.Keys)
            {
                yield return item;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region ICollection methods

        public virtual int Count
        {
            get { return dict.Count; }
        }

        public virtual bool IsReadOnly
        {
            get { return false; }
        }

        public virtual void Add(T item)
        {
            //if (!dict.ContainsKey(item)) dict.Add(item, null);
            dict[item] = new EmptyStruct();
        }

        public virtual void Clear()
        {
            dict.Clear();
        }

        public virtual bool Contains(T item)
        {
            return dict.ContainsKey(item);
        }

        public virtual void CopyTo(T[] array, int index)
        {
            throw new NotSupportedException();
        }

        public virtual bool Remove(T item)
        {
            return dict.Remove(item);
        }

        #endregion

        #region IList methods

        public T this[int index]
        {
            get { return dict.Keys[index]; }
            set { dict.Keys[index] = value; }
        }

        public int IndexOf(T item)
        {
            return dict.IndexOfKey(item);
        }

        public void Insert(int index, T item)
        {
            throw new NotSupportedException();
        }

        public void RemoveAt(int index)
        {
            dict.RemoveAt(index);
        }

        #endregion

        /// <summary>
        /// Add all items in a collection.
        /// </summary>
        /// <param name="list"></param>
        public void AddRange(IEnumerable<T> list)
        {
            foreach (T item in list)
            {
                Add(item);
            }
        }

        /// <summary>
        /// Test membership of all items in a collection.
        /// </summary>
        /// <param name="list"></param>
        /// <returns>result[i] is true iff the set contains list[i].</returns>
        public IList<bool> Contains(IList<T> list)
        {
            bool[] result = new bool[list.Count];
            int i = 0;
            foreach (T item in list)
            {
                result[i++] = Contains(item);
            }
            return result;
        }

        /// <summary>
        /// Test membership of all items in a collection.
        /// </summary>
        /// <param name="list"></param>
        /// <returns>true if the set contains any item in list.</returns>
        public bool ContainsAny(IEnumerable<T> list)
        {
            foreach (T item in list)
            {
                if (Contains(item)) return true;
            }
            return false;
        }

        /// <summary>
        /// Remove all items in a collection.
        /// </summary>
        /// <param name="list"></param>
        public void Remove(IEnumerable<T> list)
        {
            foreach (T item in list)
            {
                Remove(item);
            }
        }

        public object Clone()
        {
            SortedSet<T> result = new SortedSet<T>(Capacity, Comparer);
            result.AddRange(this);
            return result;
        }

        public override bool Equals(object that)
        {
            SortedSet<T> thatSet = that as SortedSet<T>;
            if (thatSet == null) return false;
            return this.ValueEquals(thatSet);
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            foreach (T item in this)
            {
                hash = Hash.Combine(hash, item.GetHashCode());
            }
            return hash;
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            int i = 0;
            foreach (T item in this)
            {
                if (i > 0) s.Append(" ");
                s.Append(item);
                i++;
            }
            return s.ToString();
        }

        #region Set operators

        public static SortedSet<T> operator |(SortedSet<T> a, SortedSet<T> b)
        {
            if (a == null) return b;
            if (b == null) return a;
            SortedSet<T> c = (SortedSet<T>) a.Clone();
            c.AddRange(b);
            return c;
        }

        public static SortedSet<T> Union(SortedSet<T> a, SortedSet<T> b)
        {
            return (a | b);
        }

        public static SortedSet<T> operator &(SortedSet<T> a, SortedSet<T> b)
        {
            if ((a == null) || (b == null)) return null;
            SortedSet<T> c = new SortedSet<T>(a.Capacity, a.Comparer);
            foreach (T item in a)
            {
                if (b.Contains(item)) c.Add(item);
            }
            return c;
        }

        public static SortedSet<T> Intersection(SortedSet<T> a, SortedSet<T> b)
        {
            return (a & b);
        }

        public static bool operator ==(SortedSet<T> a, SortedSet<T> b)
        {
            if (object.ReferenceEquals(a, null) != object.ReferenceEquals(b, null)) return false;
            if (object.ReferenceEquals(a, null)) return true;
            return a.Equals(b);
        }

        public static bool operator !=(SortedSet<T> a, SortedSet<T> b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Superset operator.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>True iff a is equal to or a superset of b.</returns>
        /// <remarks>null is treated as an empty set.</remarks>
        public static bool operator >=(SortedSet<T> a, SortedSet<T> b)
        {
            if (b == null) return true;
            if (a == null) return false; // because b is not null
            foreach (T item in b)
            {
                if (!a.Contains(item)) return false;
            }
            return true;
        }

        public static bool operator >(SortedSet<T> a, SortedSet<T> b)
        {
            return (a >= b) && (a != b);
        }

        /// <summary>
        /// Subset operator.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>True iff a is equal to or a subset of b.</returns>
        /// <remarks>null is treated as an empty set.</remarks>
        public static bool operator <=(SortedSet<T> a, SortedSet<T> b)
        {
            return (b >= a);
        }

        public static bool operator <(SortedSet<T> a, SortedSet<T> b)
        {
            return (a <= b) && (a != b);
        }

        #endregion
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}