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
    /// A collection of unique items.
    /// </summary>
    /// <typeparam name="T">The item type.</typeparam>
    /// <remarks>
    /// A collection of items, all of which are different according
    /// to Equals.  null items are not allowed.  Adding a duplicate has no effect.
    /// Union, intersection, and superset are all supported via operator overloading.
    /// The items are stored in the keys of a Dictionary that ensures uniqueness
    /// via its own comparer function.
    /// </remarks>
    public class Set<T> : ICollection<T>, ICloneable, IReadOnlyCollection<T>
    {
        protected struct EmptyStruct
        {
        }

        private Dictionary<T, EmptyStruct> dict;

        public Set()
        {
            dict = new Dictionary<T, EmptyStruct>();
        }

        public Set(IEqualityComparer<T> comparer)
        {
            dict = new Dictionary<T, EmptyStruct>(comparer);
        }

        public Set(int capacity, IEqualityComparer<T> comparer)
        {
            dict = new Dictionary<T, EmptyStruct>(capacity, comparer);
        }

        // this constructor causes confusion because an object can be both IEnumerable and IEqualityComparer
        //public Set(IEnumerable<T> collection) : this()
        //{
        //  Add(collection);
        //}
        public static Set<T> FromEnumerable(IEnumerable<T> collection)
        {
            Set<T> result = new Set<T>();
            result.AddRange(collection);
            return result;
        }

        public static Set<T> FromEnumerable(IEqualityComparer<T> comparer, IEnumerable<T> collection)
        {
            Set<T> result = new Set<T>(comparer);
            result.AddRange(collection);
            return result;
        }

        public Set<T> ConvertAll(Converter<T, T> converter)
        {
            Set<T> result = new Set<T>(Comparer);
            foreach (T item in this)
            {
                result.Add(converter(item));
            }
            return result;
        }

        public IEqualityComparer<T> Comparer
        {
            get { return dict.Comparer; }
        }

        public int Capacity
        {
            get { return dict.Count; }
            set
            {
                /* do nothing since Dictionary doesn't support it */
            }
        }

        #region IEnumerable methods

        public virtual IEnumerator<T> GetEnumerator()
        {
            return dict.Keys.GetEnumerator();
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
            EnumerableExtensions.CopyTo(this, array, index);
        }

        public virtual bool Remove(T item)
        {
            return dict.Remove(item);
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
        /// <returns>true if the set contains all items in list.</returns>
        public bool ContainsAll(IEnumerable<T> list)
        {
            foreach (T item in list)
            {
                if (!Contains(item)) return false;
            }
            return true;
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
            Set<T> result = new Set<T>(Capacity, Comparer);
            result.AddRange(this);
            return result;
        }

        public override bool Equals(object that)
        {
            Set<T> thatSet = that as Set<T>;
            if (thatSet == null) return false;
            return (Count == thatSet.Count) && ContainsAll(thatSet);
        }

        public override int GetHashCode()
        {
            return Hash.GetHashCodeAsSet(this, Comparer);
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

        public void IntersectWith(IEnumerable<T> items)
        {
            var s = Intersection(this, items);
            this.dict = s.dict;
        }

        #region Set operators

        public static Set<T> operator +(Set<T> a, Set<T> b)
        {
            return (a | b);
        }

        public static Set<T> operator |(Set<T> a, Set<T> b)
        {
            if (a == null) return b;
            if (b == null) return a;
            Set<T> c = (Set<T>) a.Clone();
            c.AddRange(b);
            return c;
        }

        public static Set<T> Union(Set<T> a, Set<T> b)
        {
            return (a | b);
        }

        public static Set<T> operator &(Set<T> a, IEnumerable<T> b)
        {
            if ((a == null) || (b == null)) return null;
            Set<T> c = new Set<T>(a.Capacity, a.Comparer);
            foreach (T item in b)
            {
                if (a.Contains(item)) c.Add(item);
            }
            return c;
        }

        public static Set<T> Intersection(Set<T> a, IEnumerable<T> b)
        {
            return (a & b);
        }

        public static Set<T> operator -(Set<T> a, IEnumerable<T> b)
        {
            Set<T> c = (Set<T>) a.Clone();
            c.Remove(b);
            return c;
        }

        public static Set<T> Difference(Set<T> a, IEnumerable<T> b)
        {
            return (a - b);
        }

        public static bool operator ==(Set<T> a, Set<T> b)
        {
            if (object.ReferenceEquals(a, null) != object.ReferenceEquals(b, null)) return false;
            if (object.ReferenceEquals(a, null)) return true;
            return a.Equals(b);
        }

        public static bool operator !=(Set<T> a, Set<T> b)
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
        public static bool operator >=(Set<T> a, Set<T> b)
        {
            if (b == null) return true;
            if (a == null) return false; // because b is not null
            foreach (T item in b)
            {
                if (!a.Contains(item)) return false;
            }
            return true;
        }

        public static bool operator >(Set<T> a, Set<T> b)
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
        public static bool operator <=(Set<T> a, Set<T> b)
        {
            return (b >= a);
        }

        public static bool operator <(Set<T> a, Set<T> b)
        {
            return (a <= b) && (a != b);
        }

        #endregion
    }
}

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif