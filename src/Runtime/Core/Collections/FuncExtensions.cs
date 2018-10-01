// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Provides extension methods for Func that support collection and sparse collections
    /// </summary>
    public static class FuncExtensions
    {
        /// <summary>
        /// Maps one enumerable sequence to another. Supports sparse representations.
        /// </summary>
        /// <typeparam name="T">Source sequence element type</typeparam>
        /// <typeparam name="TRes">Result sequence element type</typeparam>
        /// <param name="fun">The function</param>
        /// <param name="a">The source sequence</param>
        /// <returns></returns>
        public static IEnumerable<TRes> Map<T, TRes>(
            this Func<T, TRes> fun, IEnumerable<T> a)
        {
            if (a is ISparseEnumerable<T>)
                return new FuncSparseEnumerable<T, TRes>(fun, a as ISparseEnumerable<T>);
            else
                return new FuncEnumerable<T, TRes>(fun, a);
        }

        /// <summary>
        /// Maps two enumerable sequences to another. Supports sparse representations.
        /// </summary>
        /// <typeparam name="T1">First source sequence element type</typeparam>
        /// <typeparam name="T2">Second source sequence element type</typeparam>
        /// <typeparam name="TRes">Result sequence element type</typeparam>
        /// <param name="fun">The function</param>
        /// <param name="a">First source sequence</param>
        /// <param name="b">Second source sequence</param>
        /// <returns></returns>
        public static IEnumerable<TRes> Map<T1, T2, TRes>(
            this Func<T1, T2, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b)
        {
            if (a is ISparseEnumerable<T1> && b is ISparseEnumerable<T2>)
                return new FuncSparseEnumerable<T1, T2, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>);
            else
                return new FuncEnumerable<T1, T2, TRes>(fun, a, b);
        }

        /// <summary>
        /// Maps three enumerable sequences to another. Supports sparse representations.
        /// </summary>
        /// <typeparam name="T1">First source sequence element type</typeparam>
        /// <typeparam name="T2">Second source sequence element type</typeparam>
        /// <typeparam name="T3">Third source sequence element type</typeparam>
        /// <typeparam name="TRes">Result sequence element type</typeparam>
        /// <param name="fun">The function</param>
        /// <param name="a">First source sequence</param>
        /// <param name="b">Second source sequence</param>
        /// <param name="c">Second source sequence</param>
        /// <returns></returns>        
        public static IEnumerable<TRes> Map<T1, T2, T3, TRes>(
            this Func<T1, T2, T3, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c)
        {
            if (a is ISparseEnumerable<T1> && b is ISparseEnumerable<T2> && c is ISparseEnumerable<T3>)
                return new FuncSparseEnumerable<T1, T2, T3, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>, c as ISparseEnumerable<T3>);
            else
                return new FuncEnumerable<T1, T2, T3, TRes>(fun, a, b, c);
        }

        /// <summary>
        /// Maps four enumerable sequences to another. Supports sparse representations.
        /// </summary>
        /// <typeparam name="T1">First source sequence element type</typeparam>
        /// <typeparam name="T2">Second source sequence element type</typeparam>
        /// <typeparam name="T3">Third source sequence element type</typeparam>
        /// <typeparam name="T4">Fourth source sequence element type</typeparam>
        /// <typeparam name="TRes">Result sequence element type</typeparam>
        /// <param name="fun">The function</param>
        /// <param name="a">First source sequence</param>
        /// <param name="b">Third source sequence</param>
        /// <param name="c">Fourth source sequence</param>
        /// <param name="d">Fifth source sequence</param>
        /// <returns></returns>        
        public static IEnumerable<TRes> Map<T1, T2, T3, T4, TRes>(
            this Func<T1, T2, T3, T4, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d)
        {
            if (a is ISparseEnumerable<T1> && b is ISparseEnumerable<T2> && c is ISparseEnumerable<T3> && d is ISparseEnumerable<T4>)
                return new FuncSparseEnumerable<T1, T2, T3, T4, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>, c as ISparseEnumerable<T3>,
                                                                      d as ISparseEnumerable<T4>);
            else
                return new FuncEnumerable<T1, T2, T3, T4, TRes>(fun, a, b, c, d);
        }

        /// <summary>
        /// Maps four enumerable sequences to another. Supports sparse representations.
        /// </summary>
        /// <typeparam name="T1">First source sequence element type</typeparam>
        /// <typeparam name="T2">Second source sequence element type</typeparam>
        /// <typeparam name="T3">Third source sequence element type</typeparam>
        /// <typeparam name="T4">Fourth source sequence element type</typeparam>
        /// <typeparam name="T5">Fifth source sequence element type</typeparam>
        /// <typeparam name="TRes">Result sequence element type</typeparam>
        /// <param name="fun">The function</param>
        /// <param name="a">First source sequence</param>
        /// <param name="b">Second source sequence</param>
        /// <param name="c">Third source sequence</param>
        /// <param name="d">Fourth source sequence</param>
        /// <param name="e">Fifth source sequence</param>
        /// <returns></returns>        
        public static IEnumerable<TRes> Map<T1, T2, T3, T4, T5, TRes>(
            this Func<T1, T2, T3, T4, T5, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d, IEnumerable<T5> e)
        {
            if (a is ISparseEnumerable<T1> && b is ISparseEnumerable<T2> && c is ISparseEnumerable<T3> && d is ISparseEnumerable<T4> && e is ISparseEnumerable<T5>)
                return new FuncSparseEnumerable<T1, T2, T3, T4, T5, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>, c as ISparseEnumerable<T3>,
                                                                          d as ISparseEnumerable<T4>, e as ISparseEnumerable<T5>);
            else
                return new FuncEnumerable<T1, T2, T3, T4, T5, TRes>(fun, a, b, c, d, e);
        }
    }

    /// <summary>
    /// Sparse iterator class for a function of one sparse collection
    /// </summary>
    /// <typeparam name="T">Type of source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncSparseEnumerator<T, TRes> : SparseEnumeratorBase<TRes>
    {
        private ISparseEnumerator<T> aEnum;
        private Func<T, TRes> fun;

        public FuncSparseEnumerator(Func<T, TRes> fun, ISparseEnumerable<T> a)
        {
            aEnum = a.GetSparseEnumerator();
            this.fun = fun;
            this.commonValue = fun(aEnum.CommonValue);
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (!aEnum.MoveNext())
            {
                index = aEnum.CurrentIndex;
                return false;
            }

            index = aEnum.CurrentIndex;
            current = fun(aEnum.Current);
            sparseValueCount++;
            return true;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
        }
    }

    /// <summary>
    /// Sparse iterator class for a function of two sparse collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncSparseEnumerator<T1, T2, TRes> : SparseEnumeratorBase<TRes>
    {
        private ISparseEnumerator<T1> aEnum;
        private ISparseEnumerator<T2> bEnum;
        private Func<T1, T2, TRes> fun;
        private bool aActive, bActive;
        private T1 aCommon;
        private T2 bCommon;

        public FuncSparseEnumerator(Func<T1, T2, TRes> fun, ISparseEnumerable<T1> a, ISparseEnumerable<T2> b)
        {
            aEnum = a.GetSparseEnumerator();
            bEnum = b.GetSparseEnumerator();
            this.fun = fun;
            aCommon = aEnum.CommonValue;
            bCommon = bEnum.CommonValue;
            this.commonValue = fun(aEnum.CommonValue, bEnum.CommonValue);
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aActive || bActive)
            {
                if (aActive)
                {
                    // a is the earlier index
                    if ((!bActive) || aEnum.CurrentIndex < bEnum.CurrentIndex)
                    {
                        current = fun(aEnum.Current, bCommon);
                        index = aEnum.CurrentIndex;
                        aActive = aEnum.MoveNext();
                        sparseValueCount++;
                        return true;
                    }
                    // a and b equal indices
                    if (bActive && aEnum.CurrentIndex == bEnum.CurrentIndex)
                    {
                        current = fun(aEnum.Current, bEnum.Current);
                        index = aEnum.CurrentIndex;
                        aActive = aEnum.MoveNext();
                        bActive = bEnum.MoveNext();
                        sparseValueCount++;
                        return true;
                    }
                }
                if (bActive)
                {
                    // b is the earlier index
                    if ((!aActive) || bEnum.CurrentIndex < aEnum.CurrentIndex)
                    {
                        current = fun(aCommon, bEnum.Current);
                        index = bEnum.CurrentIndex;
                        bActive = bEnum.MoveNext();
                        sparseValueCount++;
                        return true;
                    }
                }
            }
            index = aEnum.CurrentIndex;
            return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
            aActive = aEnum.MoveNext();
            bActive = bEnum.MoveNext();
        }
    }

    /// <summary>
    /// Sparse iterator class for a function of three sparse collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="T3">Type of third source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncSparseEnumerator<T1, T2, T3, TRes> : SparseEnumeratorBase<TRes>
    {
        private ISparseEnumerator<T1> aEnum;
        private ISparseEnumerator<T2> bEnum;
        private ISparseEnumerator<T3> cEnum;
        private Func<T1, T2, T3, TRes> fun;
        private bool aActive, bActive, cActive;
        private T1 aCommon;
        private T2 bCommon;
        private T3 cCommon;

        public FuncSparseEnumerator(Func<T1, T2, T3, TRes> fun,
                                    ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c)
        {
            aEnum = a.GetSparseEnumerator();
            bEnum = b.GetSparseEnumerator();
            cEnum = c.GetSparseEnumerator();
            this.fun = fun;
            aCommon = aEnum.CommonValue;
            bCommon = bEnum.CommonValue;
            cCommon = cEnum.CommonValue;
            this.commonValue = fun(aEnum.CommonValue, bEnum.CommonValue, cEnum.CommonValue);
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aActive || bActive || cActive)
            {
                int min = int.MaxValue;
                if (aActive && min > aEnum.CurrentIndex) min = aEnum.CurrentIndex;
                if (bActive && min > bEnum.CurrentIndex) min = bEnum.CurrentIndex;
                if (cActive && min > cEnum.CurrentIndex) min = cEnum.CurrentIndex;
                index = min;
                T1 aVal = aCommon;
                T2 bVal = bCommon;
                T3 cVal = cCommon;
                if (aActive && aEnum.CurrentIndex == min)
                {
                    aVal = aEnum.Current;
                    aActive = aEnum.MoveNext();
                }
                if (bActive && bEnum.CurrentIndex == min)
                {
                    bVal = bEnum.Current;
                    bActive = bEnum.MoveNext();
                }
                if (cActive && cEnum.CurrentIndex == min)
                {
                    cVal = cEnum.Current;
                    cActive = cEnum.MoveNext();
                }
                current = fun(aVal, bVal, cVal);
                sparseValueCount++;
                return true;
            }
            index = aEnum.CurrentIndex;
            return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
            cEnum.Reset();
            aActive = aEnum.MoveNext();
            bActive = bEnum.MoveNext();
            cActive = cEnum.MoveNext();
        }
    }

    /// <summary>
    /// Sparse iterator class for a function of four sparse collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="T3">Type of third source collection</typeparam>
    /// <typeparam name="T4">Type of fourth source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncSparseEnumerator<T1, T2, T3, T4, TRes> : SparseEnumeratorBase<TRes>
    {
        private ISparseEnumerator<T1> aEnum;
        private ISparseEnumerator<T2> bEnum;
        private ISparseEnumerator<T3> cEnum;
        private ISparseEnumerator<T4> dEnum;
        private Func<T1, T2, T3, T4, TRes> fun;
        private bool aActive, bActive, cActive, dActive;
        private T1 aCommon;
        private T2 bCommon;
        private T3 cCommon;
        private T4 dCommon;

        public FuncSparseEnumerator(Func<T1, T2, T3, T4, TRes> fun,
                                    ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c, ISparseEnumerable<T4> d)
        {
            aEnum = a.GetSparseEnumerator();
            bEnum = b.GetSparseEnumerator();
            cEnum = c.GetSparseEnumerator();
            dEnum = d.GetSparseEnumerator();
            this.fun = fun;
            aCommon = aEnum.CommonValue;
            bCommon = bEnum.CommonValue;
            cCommon = cEnum.CommonValue;
            dCommon = dEnum.CommonValue;
            this.commonValue = fun(aEnum.CommonValue, bEnum.CommonValue, cEnum.CommonValue, dEnum.CommonValue);
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aActive || bActive || cActive || dActive)
            {
                int min = int.MaxValue;
                if (aActive && min > aEnum.CurrentIndex) min = aEnum.CurrentIndex;
                if (bActive && min > bEnum.CurrentIndex) min = bEnum.CurrentIndex;
                if (cActive && min > cEnum.CurrentIndex) min = cEnum.CurrentIndex;
                if (dActive && min > dEnum.CurrentIndex) min = dEnum.CurrentIndex;
                index = min;
                T1 aVal = aCommon;
                T2 bVal = bCommon;
                T3 cVal = cCommon;
                T4 dVal = dCommon;
                if (aActive && aEnum.CurrentIndex == min)
                {
                    aVal = aEnum.Current;
                    aActive = aEnum.MoveNext();
                }
                if (bActive && bEnum.CurrentIndex == min)
                {
                    bVal = bEnum.Current;
                    bActive = bEnum.MoveNext();
                }
                if (cActive && cEnum.CurrentIndex == min)
                {
                    cVal = cEnum.Current;
                    cActive = cEnum.MoveNext();
                }
                if (dActive && dEnum.CurrentIndex == min)
                {
                    dVal = dEnum.Current;
                    dActive = dEnum.MoveNext();
                }
                current = fun(aVal, bVal, cVal, dVal);
                sparseValueCount++;
                return true;
            }
            index = aEnum.CurrentIndex;
            return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
            cEnum.Reset();
            dEnum.Reset();
            aActive = aEnum.MoveNext();
            bActive = bEnum.MoveNext();
            cActive = cEnum.MoveNext();
            dActive = dEnum.MoveNext();
        }
    }

    /// <summary>
    /// Sparse iterator class for a function of four sparse collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="T3">Type of third source collection</typeparam>
    /// <typeparam name="T4">Type of fourth source collection</typeparam>
    /// <typeparam name="T5">Type of fifth source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncSparseEnumerator<T1, T2, T3, T4, T5, TRes> : SparseEnumeratorBase<TRes>
    {
        private ISparseEnumerator<T1> aEnum;
        private ISparseEnumerator<T2> bEnum;
        private ISparseEnumerator<T3> cEnum;
        private ISparseEnumerator<T4> dEnum;
        private ISparseEnumerator<T5> eEnum;
        private Func<T1, T2, T3, T4, T5, TRes> fun;
        private bool aActive, bActive, cActive, dActive, eActive;
        private T1 aCommon;
        private T2 bCommon;
        private T3 cCommon;
        private T4 dCommon;
        private T5 eCommon;

        public FuncSparseEnumerator(Func<T1, T2, T3, T4, T5, TRes> fun,
                                    ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c, ISparseEnumerable<T4> d, ISparseEnumerable<T5> e)
        {
            aEnum = a.GetSparseEnumerator();
            bEnum = b.GetSparseEnumerator();
            cEnum = c.GetSparseEnumerator();
            dEnum = d.GetSparseEnumerator();
            eEnum = e.GetSparseEnumerator();
            this.fun = fun;
            aCommon = aEnum.CommonValue;
            bCommon = bEnum.CommonValue;
            cCommon = cEnum.CommonValue;
            dCommon = dEnum.CommonValue;
            eCommon = eEnum.CommonValue;
            this.commonValue = fun(aEnum.CommonValue, bEnum.CommonValue, cEnum.CommonValue, dEnum.CommonValue, eEnum.CommonValue);
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next sparse element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aActive || bActive || cActive || dActive || eActive)
            {
                int min = int.MaxValue;
                if (aActive && min > aEnum.CurrentIndex) min = aEnum.CurrentIndex;
                if (bActive && min > bEnum.CurrentIndex) min = bEnum.CurrentIndex;
                if (cActive && min > cEnum.CurrentIndex) min = cEnum.CurrentIndex;
                if (dActive && min > dEnum.CurrentIndex) min = dEnum.CurrentIndex;
                if (eActive && min > eEnum.CurrentIndex) min = eEnum.CurrentIndex;
                index = min;
                T1 aVal = aCommon;
                T2 bVal = bCommon;
                T3 cVal = cCommon;
                T4 dVal = dCommon;
                T5 eVal = eCommon;
                if (aActive && aEnum.CurrentIndex == min)
                {
                    aVal = aEnum.Current;
                    aActive = aEnum.MoveNext();
                }
                if (bActive && bEnum.CurrentIndex == min)
                {
                    bVal = bEnum.Current;
                    bActive = bEnum.MoveNext();
                }
                if (cActive && cEnum.CurrentIndex == min)
                {
                    cVal = cEnum.Current;
                    cActive = cEnum.MoveNext();
                }
                if (dActive && dEnum.CurrentIndex == min)
                {
                    dVal = dEnum.Current;
                    dActive = dEnum.MoveNext();
                }
                if (eActive && eEnum.CurrentIndex == min)
                {
                    eVal = eEnum.Current;
                    eActive = eEnum.MoveNext();
                }
                current = fun(aVal, bVal, cVal, dVal, eVal);
                sparseValueCount++;
                return true;
            }
            index = aEnum.CurrentIndex;
            return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
            cEnum.Reset();
            dEnum.Reset();
            eEnum.Reset();
            aActive = aEnum.MoveNext();
            bActive = bEnum.MoveNext();
            cActive = cEnum.MoveNext();
            dActive = dEnum.MoveNext();
            eActive = eEnum.MoveNext();
        }
    }

    /// <summary>
    /// Enumerable generated as a function of the elements of another enumerable
    /// </summary>
    /// <typeparam name="T">Type for source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncEnumerable<T, TRes> : IEnumerable<TRes>
    {
        protected IEnumerable<T> a;
        protected Func<T, TRes> fun;

        /// <summary>
        /// Creates a new enumerator for a function of one variable to act on a collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The source collection</param>
        public FuncEnumerable(Func<T, TRes> fun, IEnumerable<T> a)
        {
            this.fun = fun;
            this.a = a;
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        public IEnumerator<TRes> GetEnumerator()
        {
            var aEnum = a.GetEnumerator();
            while (aEnum.MoveNext())
                yield return fun(aEnum.Current);
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// Enumerable generated as a function of the elements of two other enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncEnumerable<T1, T2, TRes> : IEnumerable<TRes>
    {
        protected IEnumerable<T1> a;
        protected IEnumerable<T2> b;
        protected Func<T1, T2, TRes> fun;

        /// <summary>
        /// Creates a new enumerator for a function of two variables to act on two collections
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        public FuncEnumerable(Func<T1, T2, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b)
        {
            this.fun = fun;
            this.a = a;
            this.b = b;
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        public IEnumerator<TRes> GetEnumerator()
        {
            var aEnum = a.GetEnumerator();
            var bEnum = b.GetEnumerator();
            while (aEnum.MoveNext() && bEnum.MoveNext())
                yield return fun(aEnum.Current, bEnum.Current);
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// Enumerable generated as a function of the elements of three other enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="T3">Type for third source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncEnumerable<T1, T2, T3, TRes> : IEnumerable<TRes>
    {
        protected IEnumerable<T1> a;
        protected IEnumerable<T2> b;
        protected IEnumerable<T3> c;
        protected Func<T1, T2, T3, TRes> fun;

        /// <summary>
        /// Creates a new enumerator for a function of one variable to act on a collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        public FuncEnumerable(Func<T1, T2, T3, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c)
        {
            this.fun = fun;
            this.a = a;
            this.b = b;
            this.c = c;
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        public IEnumerator<TRes> GetEnumerator()
        {
            var aEnum = a.GetEnumerator();
            var bEnum = b.GetEnumerator();
            var cEnum = c.GetEnumerator();
            while (aEnum.MoveNext() && bEnum.MoveNext() && cEnum.MoveNext())
                yield return fun(aEnum.Current, bEnum.Current, cEnum.Current);
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// Enumerable generated as a function of the elements of four other enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="T3">Type for third source enumerable</typeparam>
    /// <typeparam name="T4">Type for fourth source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncEnumerable<T1, T2, T3, T4, TRes> : IEnumerable<TRes>
    {
        protected IEnumerable<T1> a;
        protected IEnumerable<T2> b;
        protected IEnumerable<T3> c;
        protected IEnumerable<T4> d;
        protected Func<T1, T2, T3, T4, TRes> fun;

        /// <summary>
        /// Creates a new enumerator for a function of one variable to act on a collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        ///  <param name="d">The fourth source collection</param>
        public FuncEnumerable(Func<T1, T2, T3, T4, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d)
        {
            this.fun = fun;
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        public IEnumerator<TRes> GetEnumerator()
        {
            var aEnum = a.GetEnumerator();
            var bEnum = b.GetEnumerator();
            var cEnum = c.GetEnumerator();
            var dEnum = d.GetEnumerator();
            while (aEnum.MoveNext() && bEnum.MoveNext() && cEnum.MoveNext() && dEnum.MoveNext())
                yield return fun(aEnum.Current, bEnum.Current, cEnum.Current, dEnum.Current);
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// Enumerable generated as a function of the elements of five other enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="T3">Type for third source enumerable</typeparam>
    /// <typeparam name="T4">Type for fourth source enumerable</typeparam>
    ///  <typeparam name="T5">Type for fifth source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncEnumerable<T1, T2, T3, T4, T5, TRes> : IEnumerable<TRes>
    {
        protected IEnumerable<T1> a;
        protected IEnumerable<T2> b;
        protected IEnumerable<T3> c;
        protected IEnumerable<T4> d;
        protected IEnumerable<T5> e;
        protected Func<T1, T2, T3, T4, T5, TRes> fun;

        /// <summary>
        /// Creates a new enumerator for a function of one variable to act on a collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        /// <param name="d">The fourth source collection</param>
        /// <param name="e">The fifth source collection</param>
        public FuncEnumerable(Func<T1, T2, T3, T4, T5, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d, IEnumerable<T5> e)
        {
            this.fun = fun;
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
            this.e = e;
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        public IEnumerator<TRes> GetEnumerator()
        {
            var aEnum = a.GetEnumerator();
            var bEnum = b.GetEnumerator();
            var cEnum = c.GetEnumerator();
            var dEnum = d.GetEnumerator();
            var eEnum = e.GetEnumerator();
            while (aEnum.MoveNext() && bEnum.MoveNext() && cEnum.MoveNext() && dEnum.MoveNext() && eEnum.MoveNext())
                yield return fun(aEnum.Current, bEnum.Current, cEnum.Current, dEnum.Current, eEnum.Current);
        }

        /// <summary>
        /// Returns an enumerator
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// Sparse enumerable generated as a function of the elements of another sparse enumerable
    /// </summary>
    /// <typeparam name="T">Type for source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncSparseEnumerable<T, TRes> : FuncEnumerable<T, TRes>, ISparseEnumerable<TRes>
    {
        /// <summary>
        /// Creates a new enumerator for a function of one variable to act on a sparse collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The source collection</param>
        public FuncSparseEnumerable(Func<T, TRes> fun, ISparseEnumerable<T> a) :
            base(fun, a)
        {
        }

        /// <summary>
        /// Gets an enumerator over sparse values
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<TRes> GetSparseEnumerator()
        {
            return new FuncSparseEnumerator<T, TRes>(fun, a as ISparseEnumerable<T>);
        }
    }

    /// <summary>
    /// Sparse enumerable generated as a function of the elements of two other sparse enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncSparseEnumerable<T1, T2, TRes> : FuncEnumerable<T1, T2, TRes>, ISparseEnumerable<TRes>
    {
        /// <summary>
        /// Creates a new enumerator for a function of two variables to act on two sparse collections
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        public FuncSparseEnumerable(Func<T1, T2, TRes> fun, ISparseEnumerable<T1> a, ISparseEnumerable<T2> b) :
            base(fun, a, b)
        {
        }

        /// <summary>
        /// Gets an enumerator over sparse values
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<TRes> GetSparseEnumerator()
        {
            return new FuncSparseEnumerator<T1, T2, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>);
        }
    }


    /// <summary>
    /// Sparse enumerable generated as a function of the elements of three other sparse enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="T3">Type for third source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncSparseEnumerable<T1, T2, T3, TRes> : FuncEnumerable<T1, T2, T3, TRes>, ISparseEnumerable<TRes>
    {
        /// <summary>
        /// Creates a new enumerator for a function of three variables to act on three sparse collections
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        public FuncSparseEnumerable(Func<T1, T2, T3, TRes> fun, ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c) :
            base(fun, a, b, c)
        {
        }

        /// <summary>
        /// Gets an enumerator over sparse values
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<TRes> GetSparseEnumerator()
        {
            return new FuncSparseEnumerator<T1, T2, T3, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>, c as ISparseEnumerable<T3>);
        }
    }


    /// <summary>
    /// Sparse enumerable generated as a function of the elements of four other sparse enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="T3">Type for third source enumerable</typeparam>
    /// <typeparam name="T4">Type for fourth source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncSparseEnumerable<T1, T2, T3, T4, TRes> : FuncEnumerable<T1, T2, T3, T4, TRes>, ISparseEnumerable<TRes>
    {
        /// <summary>
        /// Creates a new enumerator for a function of three variables to act on three sparse collections
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        /// <param name="d">The fourth source collection</param>
        public FuncSparseEnumerable(Func<T1, T2, T3, T4, TRes> fun, ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c, ISparseEnumerable<T4> d) :
            base(fun, a, b, c, d)
        {
        }

        /// <summary>
        /// Gets an enumerator over sparse values
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<TRes> GetSparseEnumerator()
        {
            return new FuncSparseEnumerator<T1, T2, T3, T4, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>, c as ISparseEnumerable<T3>,
                                                                  d as ISparseEnumerable<T4>);
        }
    }


    /// <summary>
    /// Sparse enumerable generated as a function of the elements of four other sparse enumerables
    /// </summary>
    /// <typeparam name="T1">Type for first source enumerable</typeparam>
    /// <typeparam name="T2">Type for second source enumerable</typeparam>
    /// <typeparam name="T3">Type for third source enumerable</typeparam>
    /// <typeparam name="T4">Type for fourth source enumerable</typeparam>
    /// <typeparam name="T5">Type for fifth source enumerable</typeparam>
    /// <typeparam name="TRes">Type for result</typeparam>
    internal class FuncSparseEnumerable<T1, T2, T3, T4, T5, TRes> : FuncEnumerable<T1, T2, T3, T4, T5, TRes>, ISparseEnumerable<TRes>
    {
        /// <summary>
        /// Creates a new enumerator for a function of three variables to act on three sparse collections
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        /// <param name="d">The fourth source collection</param>
        /// <param name="e">The fifth source collection</param>
        public FuncSparseEnumerable(Func<T1, T2, T3, T4, T5, TRes> fun, ISparseEnumerable<T1> a, ISparseEnumerable<T2> b, ISparseEnumerable<T3> c, ISparseEnumerable<T4> d,
                                    ISparseEnumerable<T5> e) :
                                        base(fun, a, b, c, d, e)
        {
        }

        /// <summary>
        /// Gets an enumerator over sparse values
        /// </summary>
        /// <returns></returns>
        public ISparseEnumerator<TRes> GetSparseEnumerator()
        {
            return new FuncSparseEnumerator<T1, T2, T3, T4, T5, TRes>(fun, a as ISparseEnumerable<T1>, b as ISparseEnumerable<T2>, c as ISparseEnumerable<T3>,
                                                                      d as ISparseEnumerable<T4>, e as ISparseEnumerable<T5>);
        }
    }

#if false
    /// <summary>
    /// Base class for dense function iterators
    /// </summary>
    /// <typeparam name="TRes">Element type for result collection</typeparam>
    internal abstract class FuncDenseEnumeratorBase<TRes> : IEnumerator<TRes>
    {
        protected int index;
        protected TRes current;

    #region IEnumerator<T> Members

        /// <summary>
        /// Returns the current value
        /// </summary>
        public TRes Current
        {
            get
            {
                if (index < 0)
                    throw new InvalidOperationException(
                        "The enumerator is positioned before the first or after the last element of the collection.");
                return current;
            }
        }

        #endregion

    #region IDisposable Members
        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            return;
        }

        #endregion

    #region IEnumerator Members

        /// <summary>
        /// Returns the current value as an object
        /// </summary>
        object IEnumerator.Current
        {
            get { return (object)Current; }
        }

        /// <summary>
        /// Advances the enumerator to the next element of the list.
        /// </summary>
        /// <returns></returns>
        public abstract bool MoveNext();

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public virtual void Reset()
        {
            current = default(TRes);
            index = -1;
        }

        #endregion
    }
    /// <summary>
    /// Dense iterator class for a function of one collection
    /// </summary>
    /// <typeparam name="T">Type of first source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncDenseEnumerator<T, TRes> : FuncDenseEnumeratorBase<TRes>
    {
        private IEnumerator<T> aEnum;
        private Func<T, TRes> fun;
        /// <summary>
        /// Creates a new enumerator for a function of one variable to act on a collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The source collection</param>
        public FuncDenseEnumerator(Func<T, TRes> fun, IEnumerable<T> a)
        {
            this.fun = fun;
            aEnum = a.GetEnumerator();
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aEnum.MoveNext())
            {
                current = fun(aEnum.Current);
                index++;
                return true;
            }
            else
                return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
        }
    }


    /// <summary>
    /// Dense iterator class for a function of two collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncDenseEnumerator<T1, T2, TRes> : FuncDenseEnumeratorBase<TRes>
    {
        IEnumerator<T1> aEnum;
        IEnumerator<T2> bEnum;
        Func<T1, T2, TRes> fun;
        /// <summary>
        /// Creates a new enumerator for a function of two variables to act on two collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        public FuncDenseEnumerator(Func<T1, T2, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b)
        {
            this.fun = fun;
            aEnum = a.GetEnumerator();
            bEnum = b.GetEnumerator();
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aEnum.MoveNext() && bEnum.MoveNext())
            {
                current = fun(aEnum.Current, bEnum.Current);
                index++;
                return true;
            }
            else
                return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
        }
    }

    /// <summary>
    /// Dense iterator class for a function of three collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="T3">Type of second source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncDenseEnumerator<T1, T2, T3, TRes> : FuncDenseEnumeratorBase<TRes>
    {
        IEnumerator<T1> aEnum;
        IEnumerator<T2> bEnum;
        IEnumerator<T3> cEnum;
        Func<T1, T2, T3, TRes> fun;
        /// <summary>
        /// Creates a new enumerator for a function of three variables to act on two collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        public FuncDenseEnumerator(Func<T1, T2, T3, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c)
        {
            this.fun = fun;
            aEnum = a.GetEnumerator();
            bEnum = b.GetEnumerator();
            cEnum = c.GetEnumerator();
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aEnum.MoveNext() && bEnum.MoveNext() && cEnum.MoveNext())
            {
                current = fun(aEnum.Current, bEnum.Current, cEnum.Current);
                index++;
                return true;
            }
            else
                return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
            cEnum.Reset();
        }
    }


    /// <summary>
    /// Dense iterator class for a function of four collections
    /// </summary>
    /// <typeparam name="T1">Type of first source collection</typeparam>
    /// <typeparam name="T2">Type of second source collection</typeparam>
    /// <typeparam name="T3">Type of second source collection</typeparam>
    /// <typeparam name="T4">Type of fourth source collection</typeparam>
    /// <typeparam name="TRes">Type of result collection</typeparam>
    internal class FuncDenseEnumerator<T1, T2, T3, T4, TRes> : FuncDenseEnumeratorBase<TRes>
    {
        IEnumerator<T1> aEnum;
        IEnumerator<T2> bEnum;
        IEnumerator<T3> cEnum;
        IEnumerator<T4> dEnum;
        Func<T1, T2, T3, T4, TRes> fun;
        /// <summary>
        /// Creates a new enumerator for a function of four variables to act on two collection
        /// </summary>
        /// <param name="fun">The function</param>
        /// <param name="a">The first source collection</param>
        /// <param name="b">The second source collection</param>
        /// <param name="c">The third source collection</param>
        /// <param name="d">The fourth source collection</param>
        public FuncDenseEnumerator(Func<T1, T2, T3, T4, TRes> fun, IEnumerable<T1> a, IEnumerable<T2> b, IEnumerable<T3> c, IEnumerable<T4> d)
        {
            this.fun = fun;
            aEnum = a.GetEnumerator();
            bEnum = b.GetEnumerator();
            cEnum = c.GetEnumerator();
            dEnum = d.GetEnumerator();
            Reset();
        }

        /// <summary>
        /// Advances the enumerator to the next element of the list.
        /// </summary>
        /// <returns></returns>
        public override bool MoveNext()
        {
            if (aEnum.MoveNext() && bEnum.MoveNext() && cEnum.MoveNext())
            {
                current = fun(aEnum.Current, bEnum.Current, cEnum.Current, dEnum.Current);
                index++;
                return true;
            }
            else
                return false;
        }

        /// <summary>
        /// Resets this enumeration to the beginning
        /// </summary>
        public override void Reset()
        {
            base.Reset();
            aEnum.Reset();
            bEnum.Reset();
            cEnum.Reset();
            dEnum.Reset();
        }
    }
#endif
}