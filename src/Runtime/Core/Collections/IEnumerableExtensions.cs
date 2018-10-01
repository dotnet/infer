// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Provides extension methods for IEnumerable that support sparse collections.
    /// </summary>
    public static class IEnumerableExensions
    {
        /// <summary>
        /// Reduces across an enumerable, supporting sparse enumerables
        /// </summary>
        /// <typeparam name="T">Type of this collection</typeparam>
        /// <typeparam name="TRes">Type of result</typeparam>
        /// <param name="source">This collection</param>
        /// <param name="initial">Initial value for reduction</param>
        /// <param name="fun">The function</param>
        /// <param name="repeatedFun">The function when the same value enters the reduction several times.</param>
        /// <returns></returns>
        public static TRes EnumerableReduce<T, TRes>(this IEnumerable<T> source, TRes initial,
                                                     Func<TRes, T, TRes> fun, Func<TRes, T, int, TRes> repeatedFun)
        {
            if (source is ISparseEnumerable<T>)
            {
                TRes result = initial;
                var ise = source as ISparseEnumerable<T>;
                var isee = ise.GetSparseEnumerator();

                while (isee.MoveNext())
                    result = fun(result, isee.Current);
                // CommonValueCount is correct once sparse values are consumed
                if (isee.CommonValueCount > 0)
                    result = repeatedFun(result, isee.CommonValue, isee.CommonValueCount);
                return result;
            }
            return source.Aggregate(initial, fun);
        }

        public static double EnumerableSum<T>(this IEnumerable<T> source,
                                              Func<T, double> fun)
        {
            return source.EnumerableReduce(0.0, (res, current) => res + fun(current), (res, common, count) => res + count*fun(common));
        }

        public static IList<TResult> ListZip<TFirst, TSecond, TResult>(
            this IList<TFirst> first,
            IEnumerable<TSecond> second,
            Func<TFirst, TSecond, TResult> fun)
        {
            if (first is ISparseEnumerable<TFirst> && second is ISparseEnumerable<TSecond>)
            {
                var result = SparseList<TResult>.FromSize(first.Count);
                result.SetToFunction(first, second, fun);
                return result;
            }
            else
            {
                var result = new List<TResult>();
                using (IEnumerator<TFirst> e1 = first.GetEnumerator())
                using (IEnumerator<TSecond> e2 = second.GetEnumerator())
                    while (e1.MoveNext() && e2.MoveNext())
                        result.Add(fun(e1.Current, e2.Current));
                return result;
            }
        }

        public static IList<TResult> ListZip<TFirst, TSecond, TThird, TResult>(
            this IList<TFirst> first,
            IEnumerable<TSecond> second,
            IEnumerable<TThird> third,
            Func<TFirst, TSecond, TThird, TResult> fun)
        {
            if (first is ISparseEnumerable<TFirst>
                && second is ISparseEnumerable<TSecond>
                && third is ISparseEnumerable<TThird>)
            {
                var result = SparseList<TResult>.FromSize(first.Count);
                result.SetToFunction((SparseList<TFirst>) first, (SparseList<TSecond>) second, (SparseList<TThird>) third, fun);
                return result;
            }
            else
            {
                var result = new List<TResult>();
                using (IEnumerator<TFirst> e1 = first.GetEnumerator())
                using (IEnumerator<TSecond> e2 = second.GetEnumerator())
                using (IEnumerator<TThird> e3 = third.GetEnumerator())
                    while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext())
                        result.Add(fun(e1.Current, e2.Current, e3.Current));
                return result;
            }
        }

        public static IList<TResult> ListZip<TFirst, TSecond, TThird, TFourth, TResult>(
            this IList<TFirst> first,
            IList<TSecond> second,
            IList<TThird> third,
            IList<TFourth> fourth,
            Func<TFirst, TSecond, TThird, TFourth, TResult> fun)
            //where TResult : Diffable
        {
            if (first is ISparseList<TFirst>
                && second is ISparseList<TSecond>
                && third is ISparseList<TThird>
                && fourth is ISparseList<TFourth>)
            {
                var result = SparseList<TResult>.FromSize(first.Count);
                //var result = ApproximateSparseList<TResult>.FromSize(first.Count, 1e-6); 
                result.SetToFunction((ISparseList<TFirst>) first, (ISparseList<TSecond>) second, (ISparseList<TThird>) third, (ISparseList<TFourth>) fourth, fun);
                return result;
            }
            else
            {
                var result = new List<TResult>();
                using (IEnumerator<TFirst> e1 = first.GetEnumerator())
                using (IEnumerator<TSecond> e2 = second.GetEnumerator())
                using (IEnumerator<TThird> e3 = third.GetEnumerator())
                using (IEnumerator<TFourth> e4 = fourth.GetEnumerator())
                    while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext() && e4.MoveNext())
                        result.Add(fun(e1.Current, e2.Current, e3.Current, e4.Current));
                return result;
            }
        }
    }
}