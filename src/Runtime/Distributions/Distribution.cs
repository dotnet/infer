// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Reflection;

    using Math;
    using Utilities;
    using Factors.Attributes;

    /// <summary>
    /// Static class which implements useful functions on distributions.
    /// </summary>
    public static class Distribution
    {
        public static bool IsSettableTo(Type type, Type source)
        {
            if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(SettableTo<>)))
            {
                Type[] genArgs = type.GetGenericArguments();
                return genArgs[0].Equals(source);
            }
            foreach (Type face in type.GetInterfaces())
            {
                if (face.IsGenericType && (face.GetGenericTypeDefinition() == typeof(SettableTo<>)))
                {
                    Type[] genArgs = face.GetGenericArguments();
                    if (genArgs[0].Equals(source)) return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Determinines if this type is a distribution type
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool IsDistributionType(Type type)
        {
            if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(IDistribution<>))) return true;
            foreach (Type face in type.GetInterfaces())
            {
                if (face.IsGenericType && (face.GetGenericTypeDefinition() == typeof(IDistribution<>))) return true;
            }
            return false;
        }

        /// <summary>
        /// Determines whether the type, or any element, or any generic type parameter specification, etc
        /// is a distribution type
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool HasDistributionType(Type type)
        {
            if (type.IsArray)
                return HasDistributionType(type.GetElementType());

            if (type.IsGenericType)
            {
                Type[] gtypes = type.GetGenericArguments();
                foreach (Type gt in gtypes)
                    if (HasDistributionType(gt))
                        return true;
            }
            return IsDistributionType(type);
        }

        /// <summary>
        /// Returns the quality band for the distribution type
        /// </summary>
        /// <param name="type">The distribution type</param>
        /// <returns></returns>
        /// <remarks>This will throw an exception if there is no distribution associated with this type. Call
        /// <see cref="HasDistributionType"/> to check.</remarks>
        public static QualityBand GetQualityBand(Type type)
        {
            if (!HasDistributionType(type))
                throw new InferRuntimeException("Trying to get a quality band for a non-distribution type" + type.Name);

            if (type.IsArray)
                return GetQualityBand(type.GetElementType());

            QualityBand minQuality = QualityBand.Unknown;
            bool found = false;
            if (type.IsGenericType)
            {
                Type[] gtypes = type.GetGenericArguments();
                foreach (Type gt in gtypes)
                {
                    if (HasDistributionType(gt))
                    {
                        QualityBand qb = GetQualityBand(gt);
                        if ((!found) || qb < minQuality)
                            minQuality = qb;
                        found = true;
                    }
                }
            }
            if (IsDistributionType(type))
            {
                var qb = Quality.GetQualityBand(type);
                if ((!found) || qb < minQuality)
                    minQuality = qb;
            }
            return minQuality;
        }

        /// <summary>
        /// Gets the domain type of a distribution type, e.g. the domain type of 'Gaussian' is 'double'.
        /// </summary>
        /// <param name="distributionType"></param>
        /// <returns></returns>
        public static Type GetDomainType(Type distributionType)
        {
            try
            {
                Type face = distributionType.GetInterface(typeof(IDistribution<>).Name, false);
                if (face == null)
                {
                    if (distributionType.IsGenericType && distributionType.GetGenericTypeDefinition() == typeof(IDistribution<>)) face = distributionType;
                    else throw new ArgumentException(StringUtil.TypeToString(distributionType) + " does not implement IDistribution<>");
                }
                return face.GetGenericArguments()[0];
            }
            catch (AmbiguousMatchException)
            {
                // The distribution has multiple domain types - for now pick the last one declared.
                // TODO: work out what is the correct thing to do here
                Type tp = null;
                foreach (Type face in distributionType.GetInterfaces())
                {
                    if (!face.IsGenericType) continue;
                    if (face.GetGenericTypeDefinition() != typeof(IDistribution<>)) continue;
                    tp = face.GetGenericArguments()[0];
                }
                return tp;
            }
        }

        /// <summary>
        /// Convert a distribution from one type to another
        /// </summary>
        /// <typeparam name="TReturn">The desired distribution type</typeparam>
        /// <param name="distribution">The object to convert</param>
        /// <returns>The converted distribution</returns>
        public static TReturn ChangeType<TReturn>(object distribution)
        {
            Type fromType = distribution.GetType();
            Type toType = typeof(TReturn);
            if (toType.IsAssignableFrom(fromType))
            {
                return (TReturn)distribution;
            }
            if (toType.IsArray)
            {
                return Distribution.ToArray<TReturn>(distribution);
            }
            throw new ArgumentException("Cannot convert distribution type " + StringUtil.TypeToString(fromType) + " to type " + StringUtil.TypeToString(toType));
        }

        /// <summary>
        /// Makes a distribution array of a specified type and size
        /// </summary>
        /// <param name="elementType">Distribution type</param>
        /// <param name="rank">Number of dimensions</param>
        /// <returns></returns>
        public static Type MakeDistributionArrayType(Type elementType, int rank)
        {
            if (rank < 1) throw new ArgumentException("rank (" + rank + ") < 1");
            //return typeof(DistributionArray<>).MakeGenericType(elementType);
            if (rank == 1)
            {
                Type domainType = GetDomainType(elementType);
                if (elementType.IsValueType)
                {
                    return typeof(DistributionStructArray<,>).MakeGenericType(elementType, domainType);
                }
                else
                {
                    return typeof(DistributionRefArray<,>).MakeGenericType(elementType, domainType);
                }
            }
            else if (rank == 2)
            {
                Type domainType = GetDomainType(elementType);
                if (elementType.IsValueType)
                {
                    return typeof(DistributionStructArray2D<,>).MakeGenericType(elementType, domainType);
                }
                else
                {
                    return typeof(DistributionRefArray2D<,>).MakeGenericType(elementType, domainType);
                }
            }
            else
            {
                throw new ArgumentException("DistributionArray rank > 2 not yet implemented");
            }
        }

        public static Type MakeDistributionFileArrayType(Type elementType, int rank)
        {
            if (rank < 1) throw new ArgumentException("rank (" + rank + ") < 1");
            if (rank == 1)
            {
                Type domainType = GetDomainType(elementType);
                return typeof(DistributionFileArray<,>).MakeGenericType(elementType, domainType);
            }
            else
            {
                throw new ArgumentException("DistributionFileArray rank > 1 not yet implemented");
            }
        }

        /// <summary>
        /// Convert a distribution over an array variable to an array of element distributions.
        /// </summary>
        /// <typeparam name="ArrayType">A .NET array type, such as <c>Bernoulli[]</c> or <c>Gaussian[,][]</c>.  The array structure should match the domain of the distribution.</typeparam>
        /// <param name="distributionArray">A distribution over an array domain, such as <c>IDistribution&lt;bool[]&gt;</c> or <c>IDistribution&lt;double[,][]&gt;</c>.</param>
        /// <returns>An array of element distributions.</returns>
        /// <remarks>
        /// <typeparamref name="ArrayType"/> should match the array structure of the domain.  For example, if the input
        /// is <c>IDistribution&lt;bool[,]&gt;</c> then <typeparamref name="ArrayType"/> should be
        /// <c>Bernoulli[,]</c>.
        /// </remarks>
        public static ArrayType ToArray<ArrayType>(object distributionArray)
        {
            Type distributionType = distributionArray.GetType();
            MethodInfo method = new Func<object, ArrayType>(ToArray<object, ArrayType>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(distributionType, typeof(ArrayType));
            return (ArrayType)Util.Invoke(method, null, distributionArray);
        }

        /// <summary>
        /// Convert a distribution over an array variable to an array of element distributions.
        /// </summary>
        /// <typeparam name="DistributionType">The concrete type of distributionArray</typeparam>
        /// <typeparam name="ArrayType">A .NET array type, such as <c>Bernoulli[]</c> or <c>Gaussian[,][]</c>.  The array structure should match the domain of the distribution.</typeparam>
        /// <param name="distributionArray">A distribution over an array domain, such as <c>IDistribution&lt;bool[]&gt;</c> or <c>IDistribution&lt;double[,][]&gt;</c>.</param>
        /// <returns>An array of element distributions.</returns>
        /// <remarks>
        /// <typeparamref name="ArrayType"/> should match the array structure of the domain.  For example, if the input
        /// is <c>IDistribution&lt;bool[,]&gt;</c> then <typeparamref name="ArrayType"/> should be
        /// <c>Bernoulli[,]</c>.
        /// </remarks>
        internal static ArrayType ToArray<DistributionType, ArrayType>(DistributionType distributionArray)
        {
            Converter<DistributionType, ArrayType> converter = GetArrayConverter<DistributionType, ArrayType>();
            return converter(distributionArray);
        }

        /// <summary>
        /// Get a converter from a distribution over an array variable to an array of element distributions.
        /// </summary>
        /// <typeparam name="TInput">A distribution type over an array domain, such as <c>IDistribution&lt;bool[]&gt;</c></typeparam>
        /// <typeparam name="TOutput">An array type such as <c>Bernoulli[]</c></typeparam>
        /// <returns></returns>
        internal static Converter<TInput, TOutput> GetArrayConverter<TInput, TOutput>()
        {
            if (!typeof(TOutput).IsArray) throw new ArgumentException("TOutput (" + StringUtil.TypeToString(typeof(TOutput)) + ") must be an array type");
            Type elementType = Util.GetElementType(typeof(TInput));
            Type outputElementType = typeof(TOutput).GetElementType();
            if (!outputElementType.IsArray)
            {
                // delegate(TInput value) { return value.ToArray(); }
                MethodInfo method = typeof(TInput).GetMethod("ToArray", BindingFlags.Instance | BindingFlags.Public);
                if (method == null)
                {
                    string inputName = StringUtil.TypeToString(typeof(TInput));
                    string outputName = StringUtil.TypeToString(typeof(TOutput));
                    string message = inputName + " does not implement ToArray()";
                    throw new ArgumentException($"Cannot convert from {inputName} to {outputName}: {message}");
                }
                else if (!typeof(TOutput).IsAssignableFrom(method.ReturnType))
                {
                    string inputName = StringUtil.TypeToString(typeof(TInput));
                    string outputName = StringUtil.TypeToString(typeof(TOutput));
                    string returnTypeName = StringUtil.TypeToString(method.ReturnType);
                    string message = $"{inputName}.ToArray() returns {returnTypeName} instead of {outputName}";
                    throw new ArgumentException($"Cannot convert from {inputName} to {outputName}: {message}");
                }
                else
                {
                    return (Converter<TInput, TOutput>)Delegate.CreateDelegate(typeof(Converter<TInput, TOutput>), method);
                }
            }
            else
            {
                // delegate(TInput value) { return ToArray(itemConverter, value); }
                MethodInfo thisMethod = new Func<Converter<object, object>>(GetArrayConverter<object, object>).Method.GetGenericMethodDefinition();
                thisMethod = thisMethod.MakeGenericMethod(elementType, outputElementType);
                object itemConverter = Util.Invoke(thisMethod, null);
                MethodInfo method = typeof(TInput).GetMethod("ToArray", BindingFlags.Static | BindingFlags.FlattenHierarchy | BindingFlags.Public);
                method = method.MakeGenericMethod(outputElementType);
                return (Converter<TInput, TOutput>)Delegate.CreateDelegate(typeof(Converter<TInput, TOutput>), itemConverter, method);
            }
        }

        /// <summary>
        /// Convert an array of element distributions to a distribution over an array variable.
        /// </summary>
        /// <param name="arrayOfDistributions">Array of distributions</param>
        /// <returns>Distribution over an array variable</returns>
        public static object FromArray(object arrayOfDistributions)
        {
            Type arrType = arrayOfDistributions.GetType();
            if (arrType.IsArray)
            {
                object result = null;
                Type eltType = arrType.GetElementType();
                while (eltType.IsArray)
                    eltType = eltType.GetElementType();
                Type domainType = GetDomainType(eltType);
                Type distType = typeof(Distribution<>).MakeGenericType(domainType);
                MethodInfo[] staticMeths = distType.GetMethods(BindingFlags.Static | BindingFlags.Public);
                foreach (MethodInfo sm in staticMeths)
                {
                    if (sm.Name == "Array")
                    {
                        MethodInfo mi = sm.MakeGenericMethod(eltType);
                        if (mi.GetParameters()[0].ParameterType == arrType)
                        {
                            result = Util.Invoke(mi, null, arrayOfDistributions);
                            break;
                        }
                    }
                }
                return result;
            }
            else
                return arrayOfDistributions;
        }

#if false
    /// <summary>
    /// Convert a distribution array from one type to another
    /// </summary>
    /// <typeparam name="T">Type of source distribution array</typeparam>
    /// <typeparam name="U">Type of destination distribution array</typeparam>
    /// <param name="distributionArray"></param>
    /// <remarks>The element type of U is assumed to be constructable from the element type of T</remarks>
    /// <returns></returns>
        public static U ConvertArray<T, U>(T distributionArray)
            where T : CanGetDomainPrototype
        {
            Array dp = (Array)distributionArray.GetDomainPrototype();
            Type dpType = dp.GetType();
            Type dpEltType = dpType.GetElementType();
            Type TEltType = Util.GetElementType(typeof(T));
            Type UEltType = Util.GetElementType(typeof(U));
            Type distArrType = (JaggedArray.GetTypes(dpType, dpEltType, TEltType))[0];
            MethodInfo method = typeof(Distribution).GetMethod("ToArray", BindingFlags.Static|BindingFlags.Public);
            method = method.MakeGenericMethod(distArrType);
            Array tArray = (Array)method.Invoke(null, new object[] { distributionArray });
            var uArray = JaggedArray.ConvertToNew(
                tArray, TEltType, UEltType,
                delegate(object elt) { return Activator.CreateInstance(UEltType, elt); });
            return (U)FromArray(uArray);
        }
#endif

        /// <summary>
        /// Creates a uniform distribution of a specified type.
        /// </summary>
        /// <typeparam name="TDistribution">The distribution type.</typeparam>
        /// <returns>The created uniform distribution.</returns>
        public static TDistribution CreateUniform<TDistribution>()
            where TDistribution : SettableToUniform, new()
        {
            var result = new TDistribution();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a distribution which is uniform over the support of a specified distribution.
        /// </summary>
        /// <typeparam name="TDistribution">The distribution type.</typeparam>
        /// <param name="distribution">The distribution, which support will be used to create the result distribution.</param>
        /// <returns>The created distribution.</returns>
        public static TDistribution CreatePartialUniform<TDistribution>(TDistribution distribution)
            where TDistribution : SettableToPartialUniform<TDistribution>, new()
        {
            var result = new TDistribution();
            result.SetToPartialUniformOf(distribution);
            return result;
        }

        /// <summary>
        /// Sets result to value and returns result.
        /// </summary>
        /// <typeparam name="T">Type of the result</typeparam>
        /// <typeparam name="TValue">Type of the value</typeparam>
        /// <param name="result">The result</param>
        /// <param name="value">The value</param>
        /// <returns>result, or a newly allocated object.</returns>
        public static T SetTo<T, TValue>(T result, TValue value)
            where T : SettableTo<TValue>
        {
            if (result == null || result.Equals(default(T))) result = (T)((ICloneable)value).Clone();
            else result.SetTo(value);
            return result;
        }

        /// <summary>
        /// Sets result to value and returns result.
        /// </summary>
        /// <typeparam name="T">Type of the result</typeparam>
        /// <typeparam name="TValue">Type of the value</typeparam>
        /// <typeparam name="TDomain">Domain</typeparam>
        /// <param name="result">The result</param>
        /// <param name="value">The value</param>
        /// <returns>result, or a newly allocated object.</returns>
        public static T SetTo<T, TValue, TDomain>(T result, TValue value)
            where T : class, IDistribution<TDomain>, SettableTo<IDistribution<TDomain>>
            where TValue : IDistribution<TDomain>
        {
            result.SetTo(value);
            return result;
        }

        /// <summary>
        /// Product of all distributions in an array.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <returns><c>result</c>, unless it is not SettableTo&lt;T&gt; in which case an element of dists may be returned.</returns>
#if true
        public static T SetToProductOfAll<T, U>(T result, IReadOnlyList<U> dists)
#else
        public static T SetToProductOfAll<T, U, UList>(T result, UList dists)
            where UList : IList<U>  // this is significantly faster than IList<T> in the signature, for arrays
#endif
            where T : SettableToProduct<U>, SettableTo<U>, U, SettableToUniform
        {
            int count = dists.Count;
            if ((result == null || result.Equals(default(T)))
                && count > 0) result = (T)((ICloneable)dists[0]).Clone();
            if (count > 1)
            {
                result.SetToProduct(dists[0], dists[1]);
                for (int i = 2; i < count; i++)
                {
                    result.SetToProduct(result, dists[i]);
                }
            }
            else if (count == 0)
            {
                result.SetToUniform();
            }
            else if (count == 1)
            {
                result.SetTo(dists[0]);
            }
            return result;
        }

        /// <summary>
        /// Sets a distribution to the product of an array of ditributions
        /// </summary>
        /// <typeparam name="T">Domain type of the result distribution</typeparam>
        /// <typeparam name="U">Domain type of the array of distributions</typeparam>
        /// <param name="result">The result distribution</param>
        /// <param name="dists">The array of distributions</param>
        /// <returns></returns>
        public static T SetToProductOfAll<T, U>(T result, U[] dists)
            where T : SettableToProduct<U>, SettableTo<U>, U, SettableToUniform
        {
            int count = dists.Length;
            if ((result == null || result.Equals(default(T)))
                && count > 0) result = (T)((ICloneable)dists[0]).Clone();
            if (count > 1)
            {
                result.SetToProduct(dists[0], dists[1]);
                for (int i = 2; i < count; i++)
                {
                    result.SetToProduct(result, dists[i]);
                }
            }
            else if (count == 0)
            {
                result.SetToUniform();
            }
            else if (count == 1)
            {
                result.SetTo(dists[0]);
            }
            return result;
        }

        /// <summary>
        /// Multiplies result by all distributions in an array, except for one index.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <param name="index"></param>
        /// <returns><c>result</c></returns>
        public static T SetToProductWithAllExcept<T, U>(T result, IReadOnlyList<U> dists, int index)
            where T : SettableToProduct<T, U>
        {
            for (int i = 0; i < dists.Count; i++)
            {
                if (i == index) continue;
                result.SetToProduct(result, dists[i]);
            }
            return result;
        }

        /// <summary>
        /// Multiplies result by all distributions in an array, except for one index.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <param name="index"></param>
        /// <returns><c>result</c></returns>
        public static T SetToProductWithAllExcept<T, U>(T result, U[] dists, int index)
            where T : SettableToProduct<T, U>
        {
            for (int i = 0; i < dists.Length; i++)
            {
                if (i == index) continue;
                result.SetToProduct(result, dists[i]);
            }
            return result;
        }

        /// <summary>
        /// Multiplies result by all distributions in an array.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <returns><c>result</c></returns>
        public static T SetToProductWithAll<T, U>(T result, IReadOnlyList<U> dists)
            where T : SettableToProduct<T, U>
        {
            return SetToProductWithAllExcept<T, U>(result, dists, dists.Count);
        }

        /// <summary>
        /// Multiplies result by all distributions in an array.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <returns><c>result</c></returns>
        public static T SetToProductWithAll<T, U>(T result, U[] dists)
            where T : SettableToProduct<T, U>
        {
            return SetToProductWithAllExcept<T, U>(result, dists, dists.Length);
        }

        /// <summary>
        /// Multiplies result by all distributions in an array, except for one index.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <param name="index"></param>
        /// <returns><c>result</c></returns>
        public static T SetToProductWithAllExcept<T, U, TDomain>(T result, IList<U> dists, int index)
            where T : IDistribution<TDomain>, SettableToProduct<IDistribution<TDomain>, IDistribution<TDomain>>
            where U : IDistribution<TDomain>
        {
            for (int i = 0; i < dists.Count; i++)
            {
                if (i == index) continue;
                result.SetToProduct(result, dists[i]);
            }
            return result;
        }

        /// <summary>
        /// Multiplies result by all distributions in an array.
        /// </summary>
        /// <param name="result"></param>
        /// <param name="dists"></param>
        /// <returns><c>result</c></returns>
        public static T SetToProductWithAll<T, U, TDomain>(T result, IList<U> dists)
            where T : IDistribution<TDomain>, SettableToProduct<IDistribution<TDomain>, IDistribution<TDomain>>
            where U : IDistribution<TDomain>
        {
            return SetToProductWithAllExcept<T, U, TDomain>(result, dists, dists.Count);
        }

        /// <summary>
        /// Product of distributions in an array.
        /// </summary>
        /// <param name="result">A reference in which to place the results.</param>
        /// <param name="dists">An array of distributions.</param>
        /// <param name="count">The number of distributions in the array to multiply (starting from index 0).</param>
        /// <param name="index">An array index to omit in the multiplication.  If index == count, no index is omitted and all distributions are multiplied.</param>
        /// <returns><c>result</c>, unless it is not SettableTo&lt;T&gt; in which case an element of dists may be returned.</returns>
        public static T SetToProductOfAllExcept<T, U>(T result, IList<U> dists, int count, int index)
            where T : SettableToProduct<U>, SettableTo<U>, SettableToUniform, U
        {
            if (index < 0 || index > count) throw new ArgumentOutOfRangeException(nameof(index));
            if ((result == null || result.Equals(default(T)))
                && count > 0) result = (T)((ICloneable)dists[0]).Clone();
            if (count == 0)
            {
                result.SetToUniform();
            }
            else if (count == 1)
            {
                if (index == count)
                {
                    result.SetTo(dists[0]);
                }
                else
                {
                    result.SetToUniform();
                }
            }
            else if (count == 2)
            {
                if (index == count)
                {
                    result.SetToProduct(dists[0], dists[1]);
                }
                else
                {
                    result.SetTo(dists[1 - index]);
                }
            }
            else
            {
                // count >= 3
                if (index == 0)
                {
                    result.SetToProduct(dists[1], dists[2]);
                    for (int i = 3; i < count; i++)
                    {
                        result.SetToProduct(result, dists[i]);
                    }
                }
                else if (index == 1)
                {
                    result.SetToProduct(dists[0], dists[2]);
                    for (int i = 3; i < count; i++)
                    {
                        result.SetToProduct(result, dists[i]);
                    }
                }
                else
                {
                    result.SetToProduct(dists[0], dists[1]);
                    for (int i = 2; i < count; i++)
                    {
                        if (i == index) continue;
                        if (dists[i] != null)
                        {
                            result.SetToProduct(result, dists[i]);
                        }
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Log-probability that all distributions in the list would produce the same value.
        /// </summary>
        /// <param name="dists"></param>
        /// <returns><c>sum_x prod_i dists[i]&#8203;(x)</c></returns>
        public static double LogInnerProduct<T>(IList<T> dists)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, ICloneable
        {
            if (dists.Count <= 1) return 0.0;
            else
            {
                double result = dists[0].GetLogAverageOf(dists[1]);
                if (dists.Count > 2)
                {
                    // TODO: use a result buffer
                    T product = (T)dists[0].Clone();
                    for (int i = 2; i < dists.Count; i++)
                    {
                        product.SetToProduct(product, dists[i - 1]);
                        result += product.GetLogAverageOf(dists[i]);
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Max difference between two arrays of distributions
        /// </summary>
        /// <typeparam name="T">Domain type for fisrt array of distributions</typeparam>
        /// <typeparam name="U">Domain type for second array of distributions</typeparam>
        /// <param name="a">First array of distributions</param>
        /// <param name="b">Second array of distributions</param>
        /// <returns></returns>
        public static double MaxDiff<T, U>(T[] a, U[] b)
            where T : Diffable
        {
            try
            {
                double diff = 0.0;
                for (int i = 0; i < a.Length; i++)
                {
                    diff = Math.Max(diff, a[i].MaxDiff(b[i]));
                }
                return diff;
            }
            catch
            {
                return Double.PositiveInfinity;
            }
        }

        public static void SetTo<T1, T2>(T1[] result, T2[] values)
            where T1 : SettableTo<T2>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetTo(values[i]);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of result to the product of the corresponding distributions in two given
        /// arrays
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the two distribution arrays</typeparam>
        /// <param name="result">Result</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="b">The second distribution array</param>
        public static void SetToProduct<T1, T2>(T1[] result, T2[] a, T2[] b)
            where T1 : SettableToProduct<T2>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetToProduct(a[i], b[i]);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of the result to the product of the corresponding distributions in two given
        /// arrays.
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the numerator distribution arrays</typeparam>
        /// <typeparam name="T3">Element type of the denominator distribution array</typeparam>
        /// <typeparam name="TDomain">Domian type</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="b">The second distribution array</param>
        public static void SetToProduct<T1, T2, T3, TDomain>(T1[] result, T2[] a, T3[] b)
            where T1 : IDistribution<TDomain>, SettableToProduct<IDistribution<TDomain>, IDistribution<TDomain>>
            where T2 : IDistribution<TDomain>
            where T3 : IDistribution<TDomain>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetToProduct(a[i], b[i]);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of the result to the ratio of the corresponding distributions in two given
        /// arrays.
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the numerator and denominator distribution arrays</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        /// <param name="forceProper">Argument passed to T1.SetToRatio</param>
        public static void SetToRatio<T1, T2>(T1[] result, T2[] numerator, T2[] denominator, bool forceProper)
            where T1 : SettableToRatio<T2>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetToRatio(numerator[i], denominator[i], forceProper);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of the result to the ratio of the corresponding distributions in two given
        /// arrays.
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the numerator distribution arrays</typeparam>
        /// <typeparam name="T3">Element type of the denominator distribution array</typeparam>
        /// <typeparam name="TDomain">Domian type</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="numerator">The numerator distribution array</param>
        /// <param name="denominator">The denominator distribution array</param>
        public static void SetToRatio<T1, T2, T3, TDomain>(T1[] result, T2[] numerator, T3[] denominator)
            where T1 : IDistribution<TDomain>, SettableToRatio<IDistribution<TDomain>, IDistribution<TDomain>>
            where T2 : IDistribution<TDomain>
            where T3 : IDistribution<TDomain>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetToRatio(numerator[i], denominator[i]);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of the result to a power of the corresponding element in a source distribution array
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the source distribution array</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public static void SetToPower<T1, T2>(T1[] result, T2[] a, double exponent)
            where T1 : SettableToPower<T2>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetToPower(a[i], exponent);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of the result to a power of the corresponding element in a source distribution array
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the source distribution array</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="a">The source distribution array</param>
        /// <param name="exponent">The exponent</param>
        public static void SetToPowerLazy<T1, T2>(T1[] result, T2[] a, double exponent)
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                ((SettableToPower<T2>)item).SetToPower(a[i], exponent);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of result to a weighted sum of the corresponding distributions in two given
        /// arrays.
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the two source distribution arrays</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="b">The second distribution array</param>
        public static void SetToSum<T1, T2>(T1[] result, double weight1, T2[] a, double weight2, T2[] b)
            where T1 : SettableToWeightedSum<T2>
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                item.SetToSum(weight1, a[i], weight2, b[i]);
                result[i] = item;
            }
        }

        /// <summary>
        /// Sets each element of result to a weighted sum of the corresponding distributions in two given
        /// arrays.
        /// </summary>
        /// <typeparam name="T1">Element type of the result distribution array</typeparam>
        /// <typeparam name="T2">Element type of the two source distribution arrays</typeparam>
        /// <param name="result">Result distribution array</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="a">The first distribution array</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="b">The second distribution array</param>
        public static void SetToSumLazy<T1, T2>(T1[] result, double weight1, T2[] a, double weight2, T2[] b)
        {
            for (int i = 0; i < result.Length; i++)
            {
                T1 item = result[i];
                ((SettableToWeightedSum<T2>)item).SetToSum(weight1, a[i], weight2, b[i]);
                result[i] = item;
            }
        }

        /// <summary>
        /// Computes the log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <typeparam name="T">The distribution domain.</typeparam>
        /// <typeparam name="TDistribution">The type of a distribution.</typeparam>
        /// <param name="distribution1">The first distribution.</param>
        /// <param name="distribution2">The second distribution.</param>
        /// <param name="product">The product of the distributions will be returned in this argument. If the product is zero, the value is undefined.</param>
        /// <returns>
        /// The log-probability that two distributions would draw the same sample.
        /// </returns>
        public static double GetLogAverageOf<T, TDistribution>(
            TDistribution distribution1, TDistribution distribution2, out TDistribution product)
            where TDistribution : IDistribution<T>, SettableToProduct<TDistribution>, CanGetLogAverageOf<TDistribution>, new()
        {
            if (distribution2.IsPointMass)
            {
                product = distribution2;
                return distribution1.GetLogProb(product.Point);
            }

            if (distribution1.IsPointMass)
            {
                product = distribution1;
                return distribution2.GetLogProb(product.Point);
            }

            double logNormalizer = distribution1.GetLogAverageOf(distribution2);
            if (!double.IsNegativeInfinity(logNormalizer))
            {
                product = new TDistribution();
                product.SetToProduct(distribution1, distribution2);
            }
            else
            {
                product = default(TDistribution);
            }

            return logNormalizer;
        }

        /// <summary>
        /// Helper method that creates new distribution that is equal to product of 2 given distributions.
        /// </summary>
        public static TDistribution Product<T, TDistribution>(TDistribution distribution1, TDistribution distribution2)
            where TDistribution : IDistribution<T>, SettableToProduct<TDistribution>, new()
        {
            var result = new TDistribution();
            result.SetToProduct(distribution1, distribution2);
            return result;
        }

        /// <summary>
        /// The log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <typeparam name="T1">Element type of this distribution array</typeparam>
        /// <typeparam name="T2">Element type of that distribution array</typeparam>
        /// <param name="thisDist">This distribution array</param>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*that.Evaluate(x))</c></returns>
        /// <remarks>This can be considered a type of inner product between distributions.
        /// Another name might be "LogAverage" to go with "GetAverageLog".
        /// For an array, this specializes to:
        /// <c>sum_i Math.Log(sum_x this[i].Evaluate(x)*that[i].Evaluate(x))</c>
        /// = <c>sum_i this[i].GetLogAverageOf(that[i])</c>
        /// </remarks>
        public static double GetLogAverageOf<T1, T2>(T1[] thisDist, T2[] that)
            where T1 : CanGetLogAverageOf<T2>
        {
            double sum = 0.0;
            for (int i = 0; i < thisDist.Length; i++)
            {
                sum += thisDist[i].GetLogAverageOf(that[i]);
            }
            return sum;
        }

        public static double GetLogAverageOfPower<T1, T2>(T1[] thisDist, T2[] that, double exponent)
            where T1 : CanGetLogAverageOfPower<T2>
        {
            double sum = 0.0;
            for (int i = 0; i < thisDist.Length; i++)
            {
                sum += thisDist[i].GetLogAverageOfPower(that[i], exponent);
            }
            return sum;
        }

        public static double GetLogAverageOfPowerLazy<T1, T2>(T1[] thisDist, T2[] that, double exponent)
        {
            double sum = 0.0;
            for (int i = 0; i < thisDist.Length; i++)
            {
                sum += ((CanGetLogAverageOfPower<T2>)thisDist[i]).GetLogAverageOfPower(that[i], exponent);
            }
            return sum;
        }

        /// <summary>
        /// The log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <typeparam name="T1">Element type of this distribution array</typeparam>
        /// <typeparam name="T2">Element type of that distribution array</typeparam>
        /// <param name="thisDist">This distribution array</param>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x)*that.Evaluate(x))</c></returns>
        /// <remarks>This can be considered a type of inner product between distributions.
        /// Another name might be "LogAverage" to go with "GetAverageLog".
        /// For an array, this specializes to:
        /// <c>sum_i Math.Log(sum_x this[i].Evaluate(x)*that[i].Evaluate(x))</c>
        /// = <c>sum_i this[i].GetLogAverageOf(that[i])</c>
        /// </remarks>
        public static double GetLogAverageOfLazy<T1, T2>(T1[] thisDist, T2[] that)
        {
            double sum = 0.0;
            for (int i = 0; i < thisDist.Length; i++)
            {
                sum += ((CanGetLogAverageOf<T2>)thisDist[i]).GetLogAverageOf(that[i]);
            }
            return sum;
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <typeparam name="T">Type of this distribution</typeparam>
        /// <typeparam name="TValue">Type of the distribution to take the logarithm of</typeparam>
        /// <param name="thisDist">This distribution</param>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.
        /// For an array, this specializes to:
        /// <c>sum_i sum_x this[i].Evaluate(x)*Math.Log(that[i].Evaluate(x))</c>
        /// = <c>sum_i this[i].GetAverageLog(that[i])</c>
        /// </remarks>
        public static double GetAverageLog<T, TValue>(T[] thisDist, TValue[] that)
            where T : CanGetAverageLog<TValue>
        {
            double sum = 0.0;
            for (int i = 0; i < thisDist.Length; i++)
            {
                sum += thisDist[i].GetAverageLog(that[i]);
            }
            return sum;
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <typeparam name="T1">Element type of this distribution array</typeparam>
        /// <typeparam name="T2">Element type of thatd istribution array</typeparam>
        /// <param name="thisDist">This distribution array</param>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.
        /// For an array, this specializes to:
        /// <c>sum_i sum_x this[i].Evaluate(x)*Math.Log(that[i].Evaluate(x))</c>
        /// = <c>sum_i this[i].GetAverageLog(that[i])</c>
        /// </remarks>
        public static double GetAverageLogLazy<T1, T2>(T1[] thisDist, T2[] that)
        {
            double sum = 0.0;
            for (int i = 0; i < thisDist.Length; i++)
            {
                sum += ((CanGetAverageLog<T2>)thisDist[i]).GetAverageLog(that[i]);
            }
            return sum;
        }

        /// <summary>
        /// Creates a point mass at the specified location.
        /// </summary>
        /// <param name="point">The point at which to place the point mass</param>
        /// <returns>The PointMass object</returns>
        public static object PointMass<T>(T point)
        {
            return new PointMass<T>(point);
        }

        /// <summary>
        /// Set a distribution to a point mass
        /// </summary>
        /// <typeparam name="TDist">The distribution type</typeparam>
        /// <typeparam name="T">The domain type</typeparam>
        /// <param name="result">Where to put the result (for reference types)</param>
        /// <param name="point">The location of the point mass</param>
        /// <returns>A point mass distribution of the specified type and location</returns>
        public static TDist SetPoint<TDist, T>(TDist result, T point)
            where TDist : HasPoint<T>
        {
            result.Point = point;
            return result;
        }
    }

    /// <summary>
    /// Static class which implements useful functions on distributions.
    /// </summary>
    public static class Distribution<T>
    {
        /// <summary>
        /// Computes the log-probability that two distributions would draw the same sample.
        /// </summary>
        /// <typeparam name="TDistribution">The type of a distribution.</typeparam>
        /// <param name="distribution1">The first distribution.</param>
        /// <param name="distribution2">The second distribution.</param>
        /// <param name="product">The product of the distributions will be returned in this argument. If the product is zero, the value is undefined.</param>
        /// <returns>
        /// The log-probability that two distributions would draw the same sample.
        /// </returns>
        public static double GetLogAverageOf<TDistribution>(TDistribution distribution1, TDistribution distribution2, out TDistribution product)
            where TDistribution : IDistribution<T>, SettableToProduct<TDistribution>, CanGetLogAverageOf<TDistribution>, new()
        {
            return Distribution.GetLogAverageOf<T, TDistribution>(distribution1, distribution2, out product);
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="array">The distribution of each element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[]> Array<Distribution>(Distribution[] array)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 1);
            return (IDistribution<T[]>)Activator.CreateInstance(arrayType, array);
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="length">The length of the array.</param>
        /// <param name="init">A function providing the distribution of each array element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[]> Array<Distribution>(int length, Func<int, Distribution> init)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 1);
            return (IDistribution<T[]>)Activator.CreateInstance(arrayType, length, init);
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="array">The distribution of each element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[,]> Array<Distribution>(Distribution[,] array)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 2);
            return (IDistribution<T[,]>)Activator.CreateInstance(arrayType, array);
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="length1">The first dimension of the array.</param>
        /// <param name="length2">The second dimension of the array.</param>
        /// <param name="init">A function providing the distribution of each array element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[,]> Array<Distribution>(int length1, int length2, Func<int, int, Distribution> init)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 2);
            return (IDistribution<T[,]>)Activator.CreateInstance(arrayType, length1, length2, init);
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="array">The distribution of each element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[][]> Array<Distribution>(Distribution[][] array)
            where Distribution : IDistribution<T>
        {
            Type itemType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 1);
            MethodInfo method = new Func<Distribution[][], IDistribution<T[][]>>(Distribution<T>.Array11<Distribution, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof(Distribution), itemType);
            return (IDistribution<T[][]>)Util.Invoke(method, null, new object[] { array });
        }

        private static IDistribution<T[][]> Array11<Distribution, TItem>(Distribution[][] array)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(TItem), 1);
            return (IDistribution<T[][]>)Activator.CreateInstance(arrayType, array.Length, new Func<int, TItem>(
                                                                                                delegate (int i)
                                                                                                    { return (TItem)Activator.CreateInstance(typeof(TItem), array[i]); }));
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="array">The distribution of each element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[,][]> Array<Distribution>(Distribution[,][] array)
            where Distribution : IDistribution<T>
        {
            Type itemType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 1);
            MethodInfo method = new Func<Distribution[,][], IDistribution<T[,][]>>(Distribution<T>.Array21<Distribution, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof(Distribution), itemType);
            return (IDistribution<T[,][]>)Util.Invoke(method, null, new object[] { array });
        }

        private static IDistribution<T[,][]> Array21<Distribution, TItem>(Distribution[,][] array)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(TItem), 2);
            return (IDistribution<T[,][]>)Activator.CreateInstance(arrayType, array.GetLength(0), array.GetLength(1), new Func<int, int, TItem>(delegate (int i, int j)
            {
                return (TItem)Activator.CreateInstance(typeof(TItem), array[i, j]);
            }));
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="array">The distribution of each element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[][,]> Array<Distribution>(Distribution[][,] array)
            where Distribution : IDistribution<T>
        {
            Type itemType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 2);
            MethodInfo method = new Func<Distribution[][,], IDistribution<T[][,]>>(Distribution<T>.Array12<Distribution, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof(Distribution), itemType);
            return (IDistribution<T[][,]>)Util.Invoke(method, null, new object[] { array });
        }

        private static IDistribution<T[][,]> Array12<Distribution, TItem>(Distribution[][,] array)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(TItem), 1);
            return (IDistribution<T[][,]>)Activator.CreateInstance(arrayType, array.Length, new Func<int, TItem>(
                                                                                                 delegate (int i)
                                                                                                     { return (TItem)Activator.CreateInstance(typeof(TItem), array[i]); }));
        }

        /// <summary>
        /// Create a distribution over an array domain from independent distributions over the elements.
        /// </summary>
        /// <typeparam name="Distribution">Distribution type for an array element.</typeparam>
        /// <param name="array">The distribution of each element.</param>
        /// <returns>A single distribution object over the array domain.</returns>
        public static IDistribution<T[][][]> Array<Distribution>(Distribution[][][] array)
            where Distribution : IDistribution<T>
        {
            Type innerType = Distributions.Distribution.MakeDistributionArrayType(typeof(Distribution), 1);
            Type middleType = Distributions.Distribution.MakeDistributionArrayType(innerType, 1);
            MethodInfo method =
                new Func<Distribution[][][], IDistribution<T[][][]>>(Distribution<T>.Array111<Distribution, object, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof(Distribution), innerType, middleType);
            return (IDistribution<T[][][]>)Util.Invoke(method, null, new object[] { array });
        }

        private static IDistribution<T[][][]> Array111<Distribution, TInner, TMiddle>(Distribution[][][] array)
            where Distribution : IDistribution<T>
        {
            Type arrayType = Distributions.Distribution.MakeDistributionArrayType(typeof(TMiddle), 1);
            return (IDistribution<T[][][]>)Activator.CreateInstance(arrayType, array.Length, new Func<int, TMiddle>(delegate (int i)
            {
                return (TMiddle)Activator.CreateInstance(typeof(TMiddle), array[i].Length, new Func<int, TInner>(delegate (int j)
                {
                    return (TInner)Activator.CreateInstance(typeof(TInner), array[i][j]);
                }));
            }));
        }
    }
}