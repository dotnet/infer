// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// A collection that provides efficient extraction of the minimum element.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class PriorityQueue<T>
    {
        public List<T> Items;

        public int Count
        {
            get { return Items.Count; }
        }

        public IComparer<T> Comparer;

        /// <summary>
        /// Raised when an item has changed position or been removed (-1).
        /// </summary>
        public event Action<T, int> Moved;

        public T this[int index]
        {
            get { return Items[index]; }
            set
            {
                Items[index] = value;
                Changed(index);
            }
        }

        public PriorityQueue()
            : this(Comparer<T>.Default)
        {
        }

        public PriorityQueue(IComparer<T> comparer)
        {
            Items = new List<T>();
            Comparer = comparer;
        }

        /// <summary>
        /// Create a priority queue filled with count items equal to default(T).
        /// </summary>
        /// <param name="count"></param>
        /// <param name="comparer"></param>
        public PriorityQueue(int count, IComparer<T> comparer)
        {
            Comparer = comparer;
            Items = new List<T>(count);
            for (int i = 0; i < count; i++)
            {
                Items.Add(default(T));
            }
        }

        /// <summary>
        /// Create a priority queue filled with count items equal to default(T).
        /// </summary>
        /// <param name="count"></param>
        public PriorityQueue(int count)
            : this(count, Comparer<T>.Default)
        {
        }

        /// <summary>
        /// Create a priority queue initialized with the contents of list.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="comparer"></param>
        /// <param name="moved"></param>
        public PriorityQueue(IEnumerable<T> list, IComparer<T> comparer, Action<T, int> moved)
        {
            Comparer = comparer;
            if (moved != null) Moved += moved;
            Items = new List<T>(list);
            SiftAll();
        }

        /// <summary>
        /// Create a priority queue initialized with the contents of list.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="comparer"></param>
        public PriorityQueue(IEnumerable<T> list, IComparer<T> comparer)
            : this(list, comparer, null)
        {
        }

        /// <summary>
        /// Create a priority queue initialized with the contents of list.
        /// </summary>
        /// <param name="list"></param>
        public PriorityQueue(IEnumerable<T> list)
            : this(list, Comparer<T>.Default)
        {
        }

        /// <summary>
        /// Create a priority queue initialized with the contents of list.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="moved"></param>
        public PriorityQueue(IEnumerable<T> list, Action<T, int> moved)
            : this(list, Comparer<T>.Default, moved)
        {
        }

        public void Clear()
        {
            Items.Clear();
        }

        public T ExtractMinimum()
        {
            T item = Items[0];
            RemoveAt(0);
            return item;
        }

        public int IndexOf(T item)
        {
            return Items.IndexOf(item);
        }

        public bool Contains(T item)
        {
            return Items.Contains(item);
        }

        /// <summary>
        /// Add a new item to the queue.
        /// </summary>
        /// <param name="item"></param>
        public void Add(T item)
        {
            Items.Add(item);
            SiftUp(Count - 1);
        }

        /// <summary>
        /// Add several new items to the queue.
        /// </summary>
        /// <param name="items"></param>
        public void AddRange(IEnumerable<T> items)
        {
            int start = Count;
            Items.AddRange(items);
            for (int i = start; i < Count; i++) SiftUp(i);
        }

        /// <summary>
        /// Reposition node i to restore the heap property.
        /// </summary>
        /// <param name="i"></param>
        public void Changed(int i)
        {
            if ((i > 0) && LessThan(i, Parent(i)))
            {
                SiftUp(i);
            }
            else
            {
                SiftDown(i);
            }
        }

        /// <summary>
        /// Remove the item at Items[i]
        /// </summary>
        /// <param name="i">Position in the Items array</param>
        public void RemoveAt(int i)
        {
            Moved?.Invoke(Items[i], -1);
            Items[i] = Items[Count - 1];
            Items.RemoveAt(Count - 1);
            if (i < Count) SiftDown(i);
        }

        protected int Parent(int index)
        {
            return ((index + 1) >> 1) - 1;
        }

        protected int Left(int index)
        {
            return ((index + 1) << 1) - 1;
        }

        protected int Right(int index)
        {
            return Left(index) + 1;
        }

        public bool LessThan(int i, int j)
        {
            return (Comparer.Compare(this[i], this[j]) < 0);
        }

        /// <summary>
        /// Move Items[i] upward until it is greater than or equal to its parent.
        /// </summary>
        /// <param name="i">An index into Items.</param>
        public void SiftUp(int i)
        {
            T item = Items[i];
            for (;
                (i > 0) && (Comparer.Compare(item, Items[Parent(i)]) < 0);
                i = Parent(i))
            {
                Items[i] = Items[Parent(i)];
                Moved?.Invoke(Items[i], i);
            }
            Items[i] = item;
            Moved?.Invoke(Items[i], i);
        }

        /// <summary>
        /// Move Items[i] downward until it is less than or equal to its children.
        /// </summary>
        /// <param name="i"></param>
        public void SiftDown(int i)
        {
            int left = Left(i);
            int right = Right(i);
            int smallest;
            // find the smallest of i and its two children.
            if ((left < Count) && LessThan(left, i))
            {
                smallest = left;
            }
            else
            {
                smallest = i;
            }
            if ((right < Count) && LessThan(right, smallest))
            {
                smallest = right;
            }
            if (smallest != i)
            {
                T item = Items[i];
                Items[i] = Items[smallest];
                Items[smallest] = item;
                SiftDown(smallest);
            }
            Moved?.Invoke(Items[i], i);
        }

        /// <summary>
        /// Rearrange all items to satisfy the heap property.
        /// </summary>
        public void SiftAll()
        {
            int half = Count/2;
            if (Moved != null)
            {
                for (int i = Count - 1; i >= half; i--) Moved(Items[i], i);
            }
            for (int i = half - 1; i >= 0; i--) SiftDown(i);
        }

        public override string ToString()
        {
            return StringUtil.ToString(Items);
        }

        public PriorityQueue<T> Clone()
        {
            return new PriorityQueue<T>(Items, Comparer);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}