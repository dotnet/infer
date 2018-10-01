// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Provides extension methods for IList that support sparse lists.
    /// </summary>
    public static class IListExtensions
    {
        /// <summary>
        /// Returns true if and only if this list is sparse
        /// </summary>
        /// <typeparam name="T">The type of this list</typeparam>
        /// <param name="source">The list</param>
        /// <returns></returns>
        public static bool IsSparse<T>(this IList<T> source)
        {
            return source is ISparseList<T>;
        }

        /// <summary>
        /// Similar to LINQ Select, except it takes and returns a list.
        /// If the supplied list is sparse, the returned list will also be sparse.
        /// </summary>
        /// <typeparam name="T">The type of this list</typeparam>
        /// <typeparam name="T2">The target element type</typeparam>
        /// <param name="source">This list</param>
        /// <param name="converter">The converter</param>
        /// <returns></returns>
        public static IList<T2> ListSelect<T, T2>(this IList<T> source, Func<T, T2> converter)
        {
            if (source is SparseList<T>)
            {
                var target = SparseList<T2>.FromSize(source.Count);
                target.SetToFunction<T>(source, converter);
                return target;
            }
            return source.Select(converter).ToList();
        }

        /// <summary>
        /// Sums the elements of this list.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static int ListSum(this IList<int> source)
        {
            if (source is SparseList<int>)
            {
                SparseList<int> sl = (SparseList<int>) source;
                int sum = sl.CommonValue*(sl.Count - sl.SparseValues.Count);
                foreach (var sel in sl.SparseValues) sum += sel.Value;
                return sum;
            }
            return source.Sum();
        }

        /// <summary>
        /// Sums the elements of this list, after transforming each element using the specified converter.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="converter">The converter</param>
        /// <returns></returns>
        public static double ListSum(this IList<int> source, Func<int, double> converter)
        {
            if (source is SparseList<int>)
            {
                var sl = (SparseList<int>) source;
                double sum = converter(sl.CommonValue)*(sl.Count - sl.SparseValues.Count);
                foreach (var sel in sl.SparseValues) sum += converter(sel.Value);
                return sum;
            }
            else
            {
                double sum = 0.0;
                foreach (int item in source)
                {
                    sum += converter(item);
                }
                return sum;
            }
        }

        /// <summary>
        /// Computes the inner product between an integer list and a double list of the same size.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="that"></param>
        /// <returns></returns>
        public static double Inner(this IList<int> source, IList<double> that)
        {
            if (source is SparseList<int>)
            {
                return source.ToVector().Inner(that.ToVector());
            }
            else
            {
                double sum = 0.0;
                for (int i = 0; i < source.Count; i++)
                {
                    sum += source[i]*that[i];
                }
                return sum;
            }
        }

        /// <summary>
        /// Reduces across a list.
        /// </summary>
        /// <typeparam name="T">Type of this list</typeparam>
        /// <typeparam name="TRes">Type of result</typeparam>
        /// <param name="source">This list</param>
        /// <param name="initial">Initial value for reduction</param>
        /// <param name="fun">The function</param>
        /// <returns></returns>
        public static TRes ListReduce<T, TRes>(this IList<T> source, TRes initial, Func<TRes, T, TRes> fun)
        {
            if (source is SparseList<T>)
            {
                var sl = (SparseList<T>) source;
                return sl.Reduce(initial, fun);
            }
            return source.Aggregate(initial, fun);
        }

        /// <summary>
        /// Reduces across a list and another collection
        /// </summary>
        /// <typeparam name="T">Type of this list</typeparam>
        /// <typeparam name="T2">Type of the other list</typeparam>
        /// <typeparam name="TRes">Type of result</typeparam>
        /// <param name="source">This list</param>
        /// <param name="secondList">The other collection</param>
        /// <param name="initial">Initial value for reduction</param>
        /// <param name="fun">The function</param>
        /// <returns></returns>
        public static TRes ListReduce<T, T2, TRes>(this IList<T> source, IEnumerable<T2> secondList,
                                                   TRes initial, Func<TRes, T, T2, TRes> fun)
        {
            if (source is SparseList<T>)
            {
                var sl = (SparseList<T>) source;
                return sl.Reduce(initial, secondList, fun);
            }
            throw new NotImplementedException("Two argument reduce not implemented for non-sparse lists");
        }

        /// <summary>
        /// Converts a list of doubles into a vector.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Vector ToVector(this IList<double> source)
        {
            if (source is Vector) return (Vector) source;
            if (source is SparseList<double>)
            {
                var sl = (SparseList<double>) source;
                return SparseVector.FromSparseValues(
                    sl.Count, sl.CommonValue,
                    sl.SparseValues
                    );
            }
            return Vector.FromArray(source.ToArray());
        }

        /// <summary>
        /// Converts a list of ints into a double vector.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Vector ToVector(this IList<int> source)
        {
            if (source is SparseList<int>)
            {
                var sl = (SparseList<int>) source;
                return SparseVector.FromSparseValues(
                    sl.Count, sl.CommonValue,
                    sl.SparseValues.Select(sel => new ValueAtIndex<double> {Index = sel.Index, Value = sel.Value}).ToList()
                    );
            }
            return Vector.FromArray(source.Select(x => (double) x).ToArray());
        }

        /// <summary>
        /// Sets this list from an enumerable
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="that"></param>
        public static void SetTo<T>(this IList<T> source, IEnumerable<T> that)
        {
            if (source is SparseList<T>)
                ((SparseList<T>) source).SetTo(that);
            else
            {
                var thatlst = that as IList<T>;
                if (thatlst == null)
                    thatlst = that.ToList();

                if (thatlst.Count != source.Count)
                {
                    source.Clear();
                    source.AddRange(that);
                    //throw new ArgumentException("Lists have different size");
                }
                else
                    for (int i = 0; i < source.Count; i++)
                        source[i] = thatlst[i];
            }
        }

        /// <summary>
        /// Sets this list from an enumerable (need separate implementation from
        /// generic version for interop with Vectors)
        /// </summary>
        /// <param name="source"></param>
        /// <param name="that"></param>
        public static void SetTo(this IList<double> source, IEnumerable<double> that)
        {
            if (source is Vector)
                ((Vector) source).SetTo(that);
            else
                source.SetTo<double>(that);
        }

        public static IReadOnlyList<T> AsReadOnly<T>(this IList<T> list)
        {
            return new ReadOnlyWrapper<T>(list);
        }
    }

    public class ReadOnlyWrapper<T> : IReadOnlyList<T>
    {
        IList<T> list;

        public ReadOnlyWrapper(IList<T> list)
        {
            this.list = list;
        }

        public T this[int index]
        {
            get
            {
                return list[index];
            }
        }

        public int Count
        {
            get
            {
                return list.Count;
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)list).GetEnumerator();
        }
    }
}