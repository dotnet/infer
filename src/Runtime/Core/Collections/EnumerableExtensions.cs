// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Delegate for two argument predicate
    /// </summary>
    /// <typeparam name="T1">Type of first argument</typeparam>
    /// <typeparam name="T2">Type of second argument</typeparam>
    /// <param name="arg1">First argument</param>
    /// <param name="arg2">Second argument</param>
    /// <returns></returns>
    public delegate bool Predicate<T1, T2>(T1 arg1, T2 arg2);

    /// <summary>
    /// Delegate for three argument predicate
    /// </summary>
    /// <typeparam name="T1">Type of first argument</typeparam>
    /// <typeparam name="T2">Type of second argument</typeparam>
    /// <typeparam name="T3">Type of third argument</typeparam>
    /// <param name="arg1">First argument</param>
    /// <param name="arg2">Second argument</param>
    /// <param name="arg3">Third argument</param>
    /// <returns></returns>
    public delegate bool Predicate<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);

    /// <summary>
    /// Extension methods for IEnumerable
    /// </summary>
    public static class EnumerableExtensions
    {
        public static void ForEach<T>(this IEnumerable<T> list, Action<T> action)
        {
            foreach (T item in list)
            {
                action(item);
            }
        }

        public static void ForEach<T, U>(IEnumerable<T> list, IEnumerable<U> list2, Action<T, U> action)
        {
            IEnumerator<U> iter2 = list2.GetEnumerator();
            foreach (T item in list)
            {
                iter2.MoveNext();
                action(item, iter2.Current);
            }
        }

        public static void ForEach<T>(this IEnumerable list, Action<T> action)
        {
            foreach (T item in list)
            {
                action(item);
            }
        }

        public static HashSet<T> IntersectMany<T>(this IEnumerable<HashSet<T>> sets)
        {
            var setsBySize = sets.OrderBy(set => set.Count).ToList();
            if (setsBySize.Count == 0) return null;
            var res = new HashSet<T>(setsBySize[0]);

            for (var i = 1; i < setsBySize.Count; i++)
            {
                res.IntersectWith(setsBySize[i]);
            }

            return res;
        }

        public static void ForEach<T>(this IEnumerable<T> list, Action<int, T> action)
        {
            int index = 0;
            foreach (T item in list)
            {
                action(index++, item);
            }
        }

        public static void ForEach<T>(this IEnumerable list, Action<int, T> action)
        {
            int index = 0;
            foreach (T item in list)
            {
                action(index++, item);
            }
        }

        public static uint Sum(this IEnumerable<uint> source)
        {
            return source.Aggregate(0U, (a, b) => a + b);
        }

        public static uint Sum<TSource>(this IEnumerable<TSource> source, Func<TSource, uint> selector)
        {
            return Sum(source.Select(selector));
        }

        public static ulong Sum(this IEnumerable<ulong> source)
        {
            return source.Aggregate(0UL, (a, b) => a + b);
        }

        public static ulong Sum<TSource>(this IEnumerable<TSource> source, Func<TSource, ulong> selector)
        {
            return Sum(source.Select(selector));
        }

        public static bool TrueForAll<T>(IEnumerable<T> a, IEnumerable<T> b, Predicate<T, T> predicate)
        {
            IEnumerator<T> bIter = b.GetEnumerator();
            foreach (T itemA in a)
            {
                if (!bIter.MoveNext()) return false; // A longer than B
                T itemB = bIter.Current;
                if (!predicate(itemA, itemB)) return false;
            }
            if (bIter.MoveNext()) return false; // B longer than A
            return true;
        }

        public static bool JaggedValueEquals<T>(this IEnumerable<IEnumerable<T>> a, IEnumerable<IEnumerable<T>> b)
        {
            if (b == null) return false;
            else return TrueForAll(a, b, AreEqual);
        }

        public static bool ValueEquals<T>(this IEnumerable<T> a, IEnumerable<T> b)
        {
            return AreEqual(a, b);
        }

        public static bool AreEqual<T>(IEnumerable<T> a, IEnumerable<T> b)
        {
            if (a == null) return (b == null);
            else if (b == null) return false;
            else return TrueForAll(a, b, Util.AreEqual);
        }

        public static bool AreEqual(IEnumerable a, IEnumerable b)
        {
            if (a == null) return (b == null);
            else if (b == null) return false;
            else
            {
                IEnumerator bIter = b.GetEnumerator();
                foreach (object itemA in a)
                {
                    if (!bIter.MoveNext()) return false; // A longer than B
                    object itemB = bIter.Current;
                    if (itemA == null)
                    {
                        if (itemB != null) return false;
                    }
                    else if (!itemA.Equals(itemB)) return false;
                }
                if (bIter.MoveNext()) return false; // B longer than A
                return true;
            }
        }

        public static int IndexOf<T>(this IEnumerable<T> list, T value)
        {
            if (list is Array array) return Array.IndexOf(array, value);
            else if (value == null) return FindIndex(list, delegate(T item) { return item == null; });
            else return FindIndex(list, delegate(T item) { return value.Equals(item); });
        }

        public static int IndexOf<T>(this IEnumerable list, T value)
        {
            if (list is Array array) return Array.IndexOf(array, value);
            else if (value == null) return FindIndex<T>(list, delegate(T item) { return item == null; });
            else return FindIndex<T>(list, delegate(T item) { return value.Equals(item); });
        }

        /// <summary>
        /// Get the index of the first item where predicate returns true, or -1 if none
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <param name="predicate"></param>
        /// <returns></returns>
        public static int FindIndex<T>(this IEnumerable<T> list, Predicate<T> predicate)
        {
            int index = 0;
            foreach (T item in list)
            {
                if (predicate(item)) return index;
                index++;
            }
            return -1;
        }

        public static int FindIndex<T>(this IEnumerable list, Predicate<T> predicate)
        {
            int index = 0;
            foreach (T item in list)
            {
                if (predicate(item)) return index;
                index++;
            }
            return -1;
        }

        public static IEnumerable<int> FindAllIndex<T>(this IEnumerable<T> list, Predicate<T> predicate)
        {
            int index = 0;
            foreach (T item in list)
            {
                if (predicate(item)) yield return index;
                index++;
            }
        }

        public static IEnumerable<int> IndexOfAll<T>(this IEnumerable<T> list, T value)
        {
            return FindAllIndex(list, item => item.Equals(value));
        }

        /// <summary>
        /// Merges two sorted lists into one list in ascending order.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list1"></param>
        /// <param name="list2"></param>
        /// <param name="comparer"></param>
        /// <returns></returns>
        /// <remarks>
        /// If items from list1 and list2 are equal, the items from list1 are put first.
        /// </remarks>
        public static IEnumerable<T> Merge<T>(IEnumerable<T> list1, IEnumerable<T> list2, Comparison<T> comparer)
        {
            IEnumerator<T> iter1 = list1.GetEnumerator();
            IEnumerator<T> iter2 = list2.GetEnumerator();
            bool pending1 = iter1.MoveNext();
            bool pending2 = iter2.MoveNext();
            while (pending1)
            {
                if (pending2 && (comparer(iter1.Current, iter2.Current) > 0))
                {
                    // yield the smaller item
                    yield return iter2.Current;
                    pending2 = iter2.MoveNext();
                }
                else
                {
                    yield return iter1.Current;
                    pending1 = iter1.MoveNext();
                }
            }
            while (pending2)
            {
                yield return iter2.Current;
                pending2 = iter2.MoveNext();
            }
        }

        /// <summary>
        /// Skip n items from a stream, saving the skipped items to a list
        /// </summary>
        /// <typeparam name="T">The item type.</typeparam>
        /// <param name="stream">A stream</param>
        /// <param name="count">The number of items to skip from the front of the stream.</param>
        /// <param name="head">On return, the skipped items.</param>
        /// <returns>The rest of the stream.</returns>
        public static IEnumerable<T> Skip<T>(this IEnumerable<T> stream, int count, out List<T> head)
        {
            head = new List<T>();
            IEnumerator<T> iter = stream.GetEnumerator();
            for (; count > 0; count--)
            {
                if (!iter.MoveNext()) throw new ArgumentOutOfRangeException("IEnumerable has < " + count + " items");
                head.Add(iter.Current);
            }
            return FromEnumerator(iter);
        }

        public static IEnumerable<T> FromEnumerator<T>(IEnumerator<T> iter)
        {
            while (iter.MoveNext())
            {
                yield return iter.Current;
            }
        }

        public static void CopyTo<T>(this IEnumerable<T> list, T[] array, int index)
        {
            foreach (T item in list)
            {
                array[index++] = item;
            }
        }

        public static IReadOnlyList<T> ToReadOnlyList<T>(this IEnumerable<T> list)
        {
            return list.ToList();
        }

        public static IEnumerable<T> TakeRandom<T>(this IEnumerable<T> list, int count)
        {
            // Reservoir sampling
            List<T> values = new List<T>();
            int k = 0;
            foreach (var item in list)
            {
                if (k < count) values.Add(item);
                else
                {
                    int index = Rand.Int(k);
                    if (index < count)
                    {
                        values[index] = item;
                    }
                }
                k++;
            }
            return values;
        }
    }
}