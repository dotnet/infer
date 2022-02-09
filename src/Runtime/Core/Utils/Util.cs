// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Utilities
{
    using System.Linq;
    using System.Linq.Expressions;

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Helpful methods used throughout the library.
    /// </summary>
    public static class Util
    {
        public static IEqualityComparer<T> GetEqualityComparer<T>()
        {
            if (typeof(T).IsArray)
            {
                Type elementType = typeof(T).GetElementType();
                MethodInfo method = new Func<IEqualityComparer<object>>(Util.GetEqualityComparer<object>).Method.GetGenericMethodDefinition();
                method = method.MakeGenericMethod(elementType);
                object elementComparer = Invoke(method, null);
                return (IEqualityComparer<T>)Activator.CreateInstance(typeof(ArrayComparer<>).MakeGenericType(elementType), elementComparer);
            }
            else return EqualityComparer<T>.Default;
        }

        // parameters may be null
        public static bool AreEqual<T>(T a, T b)
        {
            if (a == null) return (b == null);
            else return a.Equals(b);
        }

        public static string CollectionToString<T>(IEnumerable<T> list)
        {
            StringBuilder sb = new StringBuilder("(");
            foreach (object obj in list)
            {
                sb.Append(obj);
                sb.Append(",");
            }
            if (sb[sb.Length - 1] == ',') sb.Remove(sb.Length - 1, 1);
            sb.Append(")");
            return sb.ToString();
        }

        /// <summary>
        /// Set result to the product of all items in list.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="result"></param>
        /// <param name="list">Must have Count > 1.</param>
        public static void SetToProduct<T>(T result, IEnumerable<T> list)
            where T : SettableToProduct<T>
        {
            IEnumerator<T> iter = list.GetEnumerator();
            Assert.IsTrue(iter.MoveNext());
            T first = iter.Current;
            Assert.IsTrue(iter.MoveNext());
            T second = iter.Current;
            result.SetToProduct(first, second);
            while (iter.MoveNext())
            {
                result.SetToProduct(result, iter.Current);
            }
        }

        public static Type MakeArrayType(Type elementType, int rank)
        {
            if (rank == 0) return elementType;
            else if (rank == 1) return elementType.MakeArrayType(); // bizarrely, gives different result to MakeArrayType(1);
            else return elementType.MakeArrayType(rank);
        }

        /// <summary>
        /// Like <see cref="Array.CreateInstance(Type, int[])"/> but 5x faster.
        /// </summary>
        /// <typeparam name="T">Array element type.</typeparam>
        /// <param name="lengths"></param>
        /// <returns>A new array.</returns>
        public static Array CreateArray<T>(params int[] lengths)
        {
            if (lengths.Length == 1) return new T[lengths[0]];
            else if (lengths.Length == 2) return new T[lengths[0], lengths[1]];
            else if (lengths.Length == 3) return new T[lengths[0], lengths[1], lengths[2]];
            else if (lengths.Length == 4) return new T[lengths[0], lengths[1], lengths[2], lengths[3]];
            else if (lengths.Length == 5) return new T[lengths[0], lengths[1], lengths[2], lengths[3], lengths[4]];
            else return Array.CreateInstance(typeof(T), lengths);
        }

        /// <summary>
        /// Change the element type of an array type to another type.
        /// </summary>
        /// <param name="arrayType">A scalar, array, multidimensional array, or nested array type.</param>
        /// <param name="newElementType">Any type.</param>
        /// <param name="newRank">An integer greater than zero.</param>
        /// <returns>An array type whose element type is <paramref name="newElementType"/> and whose rank is <paramref name="newRank"/></returns>
        /// <remarks>
        /// For example, <c>ChangeElementTypeAndRank(typeof(int[,][]), typeof(double), 2)</c> returns
        /// <c>typeof(double[,])</c>.
        /// </remarks>
        public static Type ChangeElementTypeAndRank(Type arrayType, Type newElementType, int newRank)
        {
            if (newRank <= 0) throw new ArgumentException($"newRank <= 0");
            if (arrayType.IsArray)
            {
                return MakeArrayType(newElementType, newRank);
            }
            else if (arrayType.IsGenericType)
            {
                Type gtd = arrayType.GetGenericTypeDefinition();
                if (gtd == typeof(Distributions.DistributionRefArray<,>) ||
                    gtd == typeof(Distributions.DistributionStructArray<,>) ||
                    gtd == typeof(Distributions.DistributionRefArray2D<,>) ||
                    gtd == typeof(Distributions.DistributionStructArray2D<,>))
                {
                    Type elementDomainType = Distributions.Distribution.GetDomainType(newElementType);
                    if (newElementType.IsValueType)
                    {
                        if (newRank == 1)
                            return typeof(Distributions.DistributionStructArray<,>).MakeGenericType(newElementType, elementDomainType);
                        else if (newRank == 2)
                            return typeof(Distributions.DistributionStructArray2D<,>).MakeGenericType(newElementType, elementDomainType);
                        else
                            throw new ArgumentException($"newRank > 2");
                    }
                    else
                    {
                        if (newRank == 1)
                            return typeof(Distributions.DistributionRefArray<,>).MakeGenericType(newElementType, elementDomainType);
                        else if (newRank == 2)
                            return typeof(Distributions.DistributionRefArray2D<,>).MakeGenericType(newElementType, elementDomainType);
                        else
                            throw new ArgumentException($"newRank > 2");
                    }
                }
                else
                    throw new NotSupportedException();
            }
            else
                throw new NotSupportedException();
        }

        public static bool IsIList(Type type)
        {
            try
            {
                return type.IsArray || 
                    (type == typeof(System.Collections.IList)) ||
                    (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IList<>)) ||
                    (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IReadOnlyList<>)) ||
                    (type.GetInterface(typeof(IReadOnlyList<>).Name, false) != null) ||
                    (type.GetInterface(typeof(IList<>).Name, false) != null);
            }
            catch (AmbiguousMatchException)
            {
                return true;
            }
        }

        public static ParameterInfo[] GetParameters(MethodBase method)
        {
            var parameters = method.GetParameters();
            if (method.IsConstructor && typeof(Delegate).IsAssignableFrom(method.DeclaringType))
            {
                // Delegate constructors have a hidden second argument
                parameters = parameters.Take(1).ToArray();
            }
            return parameters;
        }

        public static Type GetElementType(Type type)
        {
            return GetElementType(type, out int rank);
        }

        public static Type GetElementType(Type type, out int rank)
        {
            if (type.IsArray)
            {
                rank = type.GetArrayRank();
                return type.GetElementType();
            }
            else if (type == typeof(System.Collections.IList))
            {
                rank = 1;
                return typeof(object);
            }
            else if (type.IsGenericType)
            {
                Type genericTypeDefinition = type.GetGenericTypeDefinition();
                if (genericTypeDefinition == typeof(IList<>) ||
                    genericTypeDefinition == typeof(IReadOnlyList<>))
                {
                    rank = 1;
                    return type.GetGenericArguments()[0];
                }
                // Fall through
            }
            // may throw AmbiguousMatchException
            Type face = type.GetInterface(typeof(IArray2D<>).Name, false);
            if (face != null)
            {
                rank = 2;
                return face.GetGenericArguments()[0];
            }
            else
            {
                face = type.GetInterface(typeof(IList<>).Name, false);
                if (face != null)
                {
                    rank = 1;
                    return face.GetGenericArguments()[0];
                }
                else
                {
                    face = type.GetInterface(typeof(IReadOnlyList<>).Name, false);
                    if (face != null)
                    {
                        rank = 1;
                        return face.GetGenericArguments()[0];
                    }
                    else
                    {
                        rank = 0;
                        return null;
                    }
                }
            }
        }

        /// <summary>
        /// Gets the innermost array element type of a given array type.
        /// </summary>
        /// <param name="arrayType">The array type.</param>
        /// <remarks>If the type is not an array, result will be the type itself.</remarks>
        /// <returns>The innermost array element type.</returns>
        public static Type GetInnermostElementType(Type arrayType)
        {
            Type elementType = arrayType;
            do
            {
                arrayType = elementType;
                elementType = GetElementType(arrayType);
            } while (elementType != null);

            return arrayType;
        }

        /// <summary>
        /// The number of indexing brackets needed to turn arrayType into innermostElementType.
        /// </summary>
        /// <param name="arrayType"></param>
        /// <param name="innermostElementType"></param>
        /// <returns>An integer between 0 and the full depth of arrayType.</returns>
        /// <remarks>
        /// For example, if <paramref name="arrayType"/> is <c>int[][]</c> and <paramref name="innermostElementType"/> is <c>int</c>, the
        /// result is 2. If <paramref name="arrayType"/> is <c>int[][]</c> and <paramref name="innermostElementType"/> is <c>int[]</c>, the
        /// result is 1. If <paramref name="arrayType"/> is <c>int[][]</c> and <paramref name="innermostElementType"/> is <c>int[][]</c>, the
        /// result is 0. 
        /// </remarks>
        public static int GetArrayDepth(Type arrayType, Type innermostElementType)
        {
            int depth = 0;
            while (!innermostElementType.IsAssignableFrom(arrayType))
            {
                depth++;
                int rank;
                Type elementType = Util.GetElementType(arrayType, out rank);
                if (elementType == null) throw new ArgumentException(arrayType + " is not an array type with innermost element type " + innermostElementType);
                arrayType = elementType;
            }
            return depth;
        }

        public static T[] ArrayInit<T>(int length, Converter<int, T> init)
        {
            T[] result = new T[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = init(i);
            }
            return result;
        }

        public static T[,] ArrayInit<T>(int length1, int length2, Func<int, int, T> init)
        {
            T[,] result = new T[length1, length2];
            for (int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    result[i, j] = init(i, j);
                }
            }
            return result;
        }

        /// <summary>
        /// Create an implicit array that calls a delegate whenever an element is read
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="length">The length of the array</param>
        /// <param name="getItem">The delegate to call on every element read</param>
        /// <returns>An IArray</returns>
        public static IArray<T> IArrayFromFunc<T>(int length, Func<int, T> getItem)
        {
            return new ArrayFromFunc<T>(length, getItem);
        }

        /// <summary>
        /// Create an implicit 2D array that calls a delegate whenever an element is read
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="length0">The length of the first dimension</param>
        /// <param name="length1">The length of the second dimension</param>
        /// <param name="getItem">The delegate to call on every element read</param>
        /// <returns>An IArray2D</returns>
        public static IArray2D<T> IArrayFromFunc<T>(int length0, int length1, Func<int, int, T> getItem)
        {
            return new ArrayFromFunc2D<T>(length0, length1, getItem);
        }

        /// <summary>
        /// Invoke a Delegate, preserving the stack trace of any exception thrown.
        /// </summary>
        /// <param name="del"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static object DynamicInvoke(Delegate del, params object[] parameters)
        {
            try
            {
                return del.DynamicInvoke(parameters);
            }
            catch (TargetInvocationException ex)
            {
                // To make the Visual Studio debugger stop at the inner exception, check "Enable Just My Code" in Debug->Options.
                // throw InnerException while preserving stack trace
                // https://stackoverflow.com/questions/57383/in-c-how-can-i-rethrow-innerexception-without-losing-stack-trace
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }

        /// <summary>
        /// Invoke a method, preserving the stack trace of any exception thrown.
        /// </summary>
        /// <param name="method"></param>
        /// <param name="target"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static object Invoke(MethodBase method, object target, params object[] parameters)
        {
            try
            {
                // constructors must be invoked specially
                if (method.IsConstructor) return ((ConstructorInfo)method).Invoke(parameters);
                else return method.Invoke(target, parameters);
            }
            catch (TargetInvocationException ex)
            {
                // To make the Visual Studio debugger stop at the inner exception, check "Enable Just My Code" in Debug->Options.
                // throw InnerException while preserving stack trace
                // https://stackoverflow.com/questions/57383/in-c-how-can-i-rethrow-innerexception-without-losing-stack-trace
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }

        /// <summary>
        /// Swaps the values of two given variables.
        /// </summary>
        /// <typeparam name="T">The type of the variables.</typeparam>
        /// <param name="value1">The first variable passed by reference.</param>
        /// <param name="value2">The second variable passed by reference.</param>
        public static void Swap<T>(ref T value1, ref T value2)
        {
            T temp = value1;
            value1 = value2;
            value2 = temp;
        }

        /// <summary>
        /// Gets the maximum k values from an enumerable (by using a heap).
        /// </summary>
        /// <typeparam name="T">The type of an element in the enumerable.</typeparam>
        /// <param name="values">The enumerable of values.</param>
        /// <param name="k">The number of values to get.</param>
        /// <param name="comparer">The comparer to use for ordering.</param>
        /// <returns>The list of the maximum k values.</returns>
        public static IEnumerable<T> GetMaxKValues<T>(IEnumerable<T> values, int k, IComparer<T> comparer = null)
        {
            if (k < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(k), "The number of elements to get must not be negative.");
            }

            if (k == 0)
            {
                return Enumerable.Empty<T>();
            }

            comparer = comparer ?? Comparer<T>.Default;
            var heap = new PriorityQueue<T>(comparer);

            foreach (var value in values)
            {
                if (heap.Count < k)
                {
                    heap.Add(value);
                }
                else if (comparer.Compare(value, heap[0]) > 0)
                {
                    heap.ExtractMinimum();
                    heap.Add(value);
                }
            }

            return heap.Items.OrderByDescending(x => x, comparer);
        }

        public static IEnumerable<IEnumerable<T>> CartesianProduct<T>(IEnumerable<IEnumerable<T>> listOfLists)
        {
            if (listOfLists.Any())
            {
                var a = listOfLists.First();
                var b = CartesianProduct(listOfLists.Skip(1));

                if (b.Any())
                {
                    foreach (var aElt in a)
                    {
                        var aEltList = new List<T> { aElt };
                        foreach (var bElt in b)
                        {
                            yield return aEltList.Concat(bElt);
                        }
                    }
                }
                else
                {
                    foreach (var aElt in a)
                    {
                        yield return new List<T> { aElt };
                    }
                }
            }

            yield break;
        }

        /// <summary>
        /// A faster versio of `new T()` when T is a generic type parameter.
        /// See: https://stackoverflow.com/a/1280832
        /// </summary>
        public static T New<T>()
            where T : new()
        {
            return NewFuncCache<T>.NewFunc();
        }

        private static class NewFuncCache<T>
            where T : new()
        {
            public static Expression<Func<T>> NewExpression = () => new T();
            public static Func<T> NewFunc = NewExpression.Compile();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}