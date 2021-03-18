// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;
using System.Reflection;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimator for a DistributionArray type.
    /// </summary>
    /// <typeparam name="ItemEstimator">Type of estimator for each array element.</typeparam>
    /// <typeparam name="DistributionArray">Type of DistributionArray to estimate.</typeparam>
    /// <typeparam name="Distribution">Type of DistributionArray element.</typeparam>
    /// <typeparam name="Sample">Type of a SampleArray element - can be distributions.</typeparam>
    public class ArrayEstimator<ItemEstimator, DistributionArray, Distribution, Sample>
        : Estimator<DistributionArray>, Accumulator<Sample[]>,
          SettableTo<ArrayEstimator<ItemEstimator, DistributionArray, Distribution, Sample>>,
          ICloneable
        where DistributionArray : IList<Distribution>
        where ItemEstimator : Estimator<Distribution>, Accumulator<Sample>, SettableTo<ItemEstimator>, ICloneable
        where Distribution : SettableTo<Distribution>
    {
        /// <summary>
        /// The array of estimators
        /// </summary>
        protected ItemEstimator[] estimators;

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="length">Length of array</param>
        /// <param name="createEstimator">ItemEstimator for each array element</param>
        public ArrayEstimator(int length, Converter<int, ItemEstimator> createEstimator)
        {
            estimators = new ItemEstimator[length];
            for (int i = 0; i < length; i++)
            {
                estimators[i] = createEstimator(i);
            }
        }

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="estimators"></param>
        public ArrayEstimator(ItemEstimator[] estimators)
        {
            this.estimators = estimators;
        }

        /// <summary>
        /// Retrieve the estimated distributions
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public DistributionArray GetDistribution(DistributionArray result)
        {
            for (int i = 0; i < estimators.Length; i++)
            {
                result[i] = estimators[i].GetDistribution(result[i]);
            }
            return result;
        }

        /// <summary>
        /// Adds an array item to the estimator
        /// </summary>
        /// <param name="item"></param>
        public void Add(Sample[] item)
        {
            if (item.Length != estimators.Length) throw new ArgumentException("item.Length (" + item.Length + ") != estimators.Length (" + estimators.Length + ")");
            for (int i = 0; i < estimators.Length; i++)
            {
                estimators[i].Add(item[i]);
            }
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            for (int i = 0; i < estimators.Length; i++)
            {
                estimators[i].Clear();
            }
        }

        /// <summary>
        /// Set this ArrayEstimator to another ArrayEstimator
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(ArrayEstimator<ItemEstimator, DistributionArray, Distribution, Sample> value)
        {
            if (value.estimators.Length != estimators.Length)
                throw new ArgumentException("value.estimators.Length (" + value.estimators.Length + ") != estimators.Length (" + estimators.Length + ")");
            for (int i = 0; i < estimators.Length; i++)
            {
                estimators[i].SetTo(value.estimators[i]);
            }
        }

        /// <summary>
        /// Clones this ArrayEstimator
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new ArrayEstimator<ItemEstimator, DistributionArray, Distribution, Sample>(
                estimators.Length,
                delegate(int i) { return (ItemEstimator) estimators[i].Clone(); });
        }
    }

    /// <summary>
    /// Estimator for a DistributionArray type where the sample type is a distribution
    /// </summary>
    /// <typeparam name="ItemEstimator">Type of estimator for each array element.</typeparam>
    /// <typeparam name="DistributionArray">Type of DistributionArray to estimate.</typeparam>
    /// <typeparam name="Distribution">Type of DistributionArray element.</typeparam>
    public class ArrayEstimator<ItemEstimator, DistributionArray, Distribution>
        : Estimator<DistributionArray>, Accumulator<DistributionArray>,
          SettableTo<ArrayEstimator<ItemEstimator, DistributionArray, Distribution>>,
          ICloneable
        where DistributionArray : IArray<Distribution>
        where ItemEstimator : Estimator<Distribution>, Accumulator<Distribution>, SettableTo<ItemEstimator>, ICloneable
        where Distribution : SettableTo<Distribution>
    {
        /// <summary>
        /// The array of estimators
        /// </summary>
        protected ItemEstimator[] estimators;

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="length">Length of array</param>
        /// <param name="createEstimator">ItemEstimator for each array element</param>
        public ArrayEstimator(int length, Converter<int, ItemEstimator> createEstimator)
        {
            estimators = new ItemEstimator[length];
            for (int i = 0; i < length; i++)
            {
                estimators[i] = createEstimator(i);
            }
        }

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="estimators"></param>
        public ArrayEstimator(ItemEstimator[] estimators)
        {
            this.estimators = estimators;
        }

        /// <summary>
        /// Retrieve the estimated distributions
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public DistributionArray GetDistribution(DistributionArray result)
        {
            if (result.Count != estimators.Length) throw new ArgumentException("result.Count (" + result.Count + ") != estimators.Length (" + estimators.Length + ")");
            for (int i = 0; i < estimators.Length; i++)
            {
                result[i] = estimators[i].GetDistribution(result[i]);
            }
            return result;
        }

        /// <summary>
        /// Adds an array item to the estimator
        /// </summary>
        /// <param name="item"></param>
        public void Add(DistributionArray item)
        {
            for (int i = 0; i < estimators.Length; i++)
            {
                estimators[i].Add(item[i]);
            }
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            for (int i = 0; i < estimators.Length; i++)
            {
                estimators[i].Clear();
            }
        }

        /// <summary>
        /// Set this ArrayEstimator to another ArrayEstimator
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(ArrayEstimator<ItemEstimator, DistributionArray, Distribution> value)
        {
            for (int i = 0; i < estimators.Length; i++)
            {
                estimators[i].SetTo(value.estimators[i]);
            }
        }

        /// <summary>
        /// Clones this ArrayEstimator
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new ArrayEstimator<ItemEstimator, DistributionArray, Distribution>(
                estimators.Length,
                delegate(int i) { return (ItemEstimator) estimators[i].Clone(); });
        }
    }

    /// <summary>
    /// Estimator for a 2-D DistributionArray type.
    /// </summary>
    /// <typeparam name="ItemEstimator">Type of estimator for each array element.</typeparam>
    /// <typeparam name="DistributionArray">Type of DistributionArray to estimate.</typeparam>
    /// <typeparam name="Distribution">Type of a DistributionArray element.</typeparam>
    /// <typeparam name="Sample">Type of a SampleArray element - can be distributions.</typeparam>
    public class Array2DEstimator<ItemEstimator, DistributionArray, Distribution, Sample>
        : Estimator<DistributionArray>, Accumulator<Sample[,]>,
          SettableTo<Array2DEstimator<ItemEstimator, DistributionArray, Distribution, Sample>>,
          ICloneable
        where DistributionArray : IArray2D<Distribution>
        where ItemEstimator : Estimator<Distribution>, Accumulator<Sample>, SettableTo<ItemEstimator>, ICloneable
        where Distribution : SettableTo<Distribution>
    {
        /// <summary>
        /// The array of estimators
        /// </summary>
        protected ItemEstimator[,] estimators;

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="length1">Length of first dimension of array</param>
        /// <param name="length2">Length of second dimension of array</param>
        /// <param name="createEstimator">ItemEstimator for each array element</param>
        public Array2DEstimator(int length1, int length2, Func<int, int, ItemEstimator> createEstimator)
        {
            estimators = new ItemEstimator[length1,length2];
            for (int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    estimators[i, j] = createEstimator(i, j);
                }
            }
        }

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="estimators">ItemEstimator for each array element</param>
        public Array2DEstimator(ItemEstimator[,] estimators)
        {
            this.estimators = estimators;
        }

        /// <summary>
        /// Retrieve the estimated distributions
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public DistributionArray GetDistribution(DistributionArray result)
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    result[i, j] = estimators[i, j].GetDistribution(result[i, j]);
                }
            }
            return result;
        }

        /// <summary>
        /// Adds an array item to the estimator
        /// </summary>
        /// <param name="item"></param>
        public void Add(Sample[,] item)
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    estimators[i, j].Add(item[i, j]);
                }
            }
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    estimators[i, j].Clear();
                }
            }
        }

        /// <summary>
        /// Set this ArrayEstimator to another ArrayEstimator
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(Array2DEstimator<ItemEstimator, DistributionArray, Distribution, Sample> value)
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    estimators[i, j].SetTo(value.estimators[i, j]);
                }
            }
        }

        /// <summary>
        /// Clones this ArrayEstimator
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new Array2DEstimator<ItemEstimator, DistributionArray, Distribution, Sample>(
                estimators.GetLength(0), estimators.GetLength(1),
                delegate(int i, int j) { return (ItemEstimator) estimators[i, j].Clone(); });
        }
    }

    /// <summary>
    /// Estimator for a 2-D DistributionArray type, where the samples are distributions
    /// </summary>
    /// <typeparam name="ItemEstimator">Type of estimator for each array element.</typeparam>
    /// <typeparam name="DistributionArray">Type of DistributionArray to estimate.</typeparam>
    /// <typeparam name="Distribution">Type of a DistributionArray element.</typeparam>
    public class Array2DEstimator<ItemEstimator, DistributionArray, Distribution>
        : Estimator<DistributionArray>, Accumulator<DistributionArray>,
          SettableTo<Array2DEstimator<ItemEstimator, DistributionArray, Distribution>>,
          ICloneable
        where DistributionArray : IArray2D<Distribution>
        where ItemEstimator : Estimator<Distribution>, Accumulator<Distribution>, SettableTo<ItemEstimator>, ICloneable
        where Distribution : SettableTo<Distribution>
    {
        /// <summary>
        /// The array of estimators
        /// </summary>
        protected ItemEstimator[,] estimators;

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="length1">Length of first dimension of array</param>
        /// <param name="length2">Length of second dimension of array</param>
        /// <param name="createEstimator">ItemEstimator for each array element</param>
        public Array2DEstimator(int length1, int length2, Func<int, int, ItemEstimator> createEstimator)
        {
            estimators = new ItemEstimator[length1,length2];
            for (int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    estimators[i, j] = createEstimator(i, j);
                }
            }
        }

        /// <summary>
        /// Constructs an ArrayEstimator
        /// </summary>
        /// <param name="estimators">ItemEstimator for each array element</param>
        public Array2DEstimator(ItemEstimator[,] estimators)
        {
            this.estimators = estimators;
        }

        /// <summary>
        /// Retrieve the estimated distributions
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public DistributionArray GetDistribution(DistributionArray result)
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    result[i, j] = estimators[i, j].GetDistribution(result[i, j]);
                }
            }
            return result;
        }

        /// <summary>
        /// Adds an array item to the estimator
        /// </summary>
        /// <param name="item"></param>
        public void Add(DistributionArray item)
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    estimators[i, j].Add(item[i, j]);
                }
            }
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    estimators[i, j].Clear();
                }
            }
        }

        /// <summary>
        /// Set this ArrayEstimator to another ArrayEstimator
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(Array2DEstimator<ItemEstimator, DistributionArray, Distribution> value)
        {
            for (int i = 0; i < estimators.GetLength(0); i++)
            {
                for (int j = 0; j < estimators.GetLength(1); j++)
                {
                    estimators[i, j].SetTo(value.estimators[i, j]);
                }
            }
        }

        /// <summary>
        /// Clones this ArrayEstimator
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return new Array2DEstimator<ItemEstimator, DistributionArray, Distribution>(
                estimators.GetLength(0), estimators.GetLength(1),
                delegate(int i, int j) { return (ItemEstimator) estimators[i, j].Clone(); });
        }
    }

    /// <summary>
    /// Wraps a list of accumulators, adding each sample to all of them.
    /// </summary>
    /// <typeparam name="T">The type to accumulate</typeparam>
    public class AccumulatorList<T> : Accumulator<T>
    {
        /// <summary>
        /// The list of accumulators
        /// </summary>
        public List<Accumulator<T>> Accumulators = new List<Accumulator<T>>();

        /// <summary>
        /// Constructs an accumulator list
        /// </summary>
        /// <param name="accumulators"></param>
        public AccumulatorList(params Accumulator<T>[] accumulators)
        {
            this.Accumulators.AddRange(accumulators);
        }

        /// <summary>
        /// Add an item to all accumulators in the list
        /// </summary>
        /// <param name="item"></param>
        public void Add(T item)
        {
            foreach (Accumulator<T> acc in Accumulators)
            {
                acc.Add(item);
            }
        }

        /// <summary>
        /// Clear all accumulators in the list
        /// </summary>
        public void Clear()
        {
            foreach (Accumulator<T> acc in Accumulators)
            {
                acc.Clear();
            }
        }
    }

    /// <summary>
    /// Wraps an accumulator, discarding the first BurnIn samples.
    /// </summary>
    /// <typeparam name="T">The type to accumulate</typeparam>
    public class BurnInAccumulator<T> : Accumulator<T>
    {
        /// <summary>
        ///  Burn in
        /// </summary>
        public int BurnIn;

        /// <summary>
        /// Thin parameter
        /// </summary>
        public int Thin;

        /// <summary>
        /// Count
        /// </summary>
        public int Count;

        /// <summary>
        /// Accumulator
        /// </summary>
        public Accumulator<T> Accumulator;

        /// <summary>
        /// Constructs a burn-in accumulator
        /// </summary>
        /// <param name="burnIn"></param>
        /// <param name="thin"></param>
        /// <param name="accumulator"></param>
        public BurnInAccumulator(int burnIn, int thin, Accumulator<T> accumulator)
        {
            this.BurnIn = burnIn;
            this.Thin = thin;
            this.Accumulator = accumulator;
        }

        /// <summary>
        /// Adds a sample to the burn-in accumulator
        /// </summary>
        /// <param name="item"></param>
        public void Add(T item)
        {
            int diff = Count - BurnIn;
            if ((diff >= 0) && ((diff%Thin) == 0)) Accumulator.Add(item);
            Count++;
        }

        /// <summary>
        /// Clears the burn-in accumulator
        /// </summary>
        public void Clear()
        {
            Accumulator.Clear();
        }
    }

    /// <summary>
    /// Useful static methods relating to array estimators
    /// </summary>
    public class ArrayEstimator
    {
        /// <summary>
        /// Create an estimator for a given distribution
        /// </summary>
        /// <typeparam name="T">Type of distribution</typeparam>
        /// <typeparam name="TDomain">Type of domain</typeparam>
        /// <param name="dist">The distribution</param>
        /// <param name="accumDist">Whether the estimator should accumulate distributions or samples</param>
        /// <returns></returns>
        public static Estimator<T> CreateEstimator<T, TDomain>(T dist, bool accumDist)
            where T : IDistribution<TDomain>
        {
            // Check if distribution is scalar
            if (dist is PointMass<TDomain> || !typeof (TDomain).IsArray)
                return EstimatorFactory.Instance.CreateEstimator<T, TDomain>(dist);

            // Domain types at each depth
            Type[] domainTypes = JaggedArray.GetTypes<TDomain>();

            // Need to find the innermost distribution type - so walk the type parameters
            Type t = (typeof (T).GetGenericArguments())[0];
            while (t.IsGenericType)
                t = (t.GetGenericArguments())[0];
            // Leaf types
            Type leafDistributionType = t;
            Type leafDomainType = domainTypes[domainTypes.Length - 1];
            Type leafEstimatorType = EstimatorFactory.Instance.EstimatorType(leafDistributionType);
            if (leafEstimatorType == null)
                throw new InferRuntimeException(
                    String.Format("Cannot find Estimator class for {0}", leafDistributionType));

            // Distribution types at each depth when considered as a jagged array of distributions
            Type[] arrDistTypes = (JaggedArray.GetTypes(
                typeof (TDomain), leafDomainType, leafDistributionType));

            // Make the jagged array of distributions
            MethodInfo method = new Func<object, object>(Distribution.ToArray<object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(arrDistTypes[0]);
            Array distributions = (Array) Util.Invoke(method, null, dist);

            // Create jagged array of estimators - these will be converted to an ArrayEstimator instance
            method = new Func<Bernoulli, Estimator<Bernoulli>>(EstimatorFactory.Instance.CreateEstimator<Bernoulli, bool>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(leafDistributionType, leafDomainType);
            if (method == null)
                throw new InferRuntimeException(
                    String.Format("Cannot find Estimator class for {0}", leafDistributionType));
            Array estimators = JaggedArray.ConvertToNew(
                distributions, leafDistributionType, leafEstimatorType, elt => Util.Invoke(method, EstimatorFactory.Instance, elt));

            int depth = domainTypes.Length;
            if (depth > 4)
                throw new InferRuntimeException("Estimator arrays of depth greater than 3 not supported");

            // Get distribution array types
            Type[] distArrTypes = new Type[depth];
            Type[] estArrTypes = new Type[depth];
            distArrTypes[depth - 1] = arrDistTypes[depth - 1];
            estArrTypes[depth - 1] = leafEstimatorType;

            for (int i = depth - 2; i >= 0; i--)
            {
                int rank = arrDistTypes[i].GetArrayRank();
                distArrTypes[i] = Distribution.MakeDistributionArrayType(distArrTypes[i + 1], rank);
                if (accumDist)
                {
                    Type[] arrEstGenericArgs = new Type[3];
                    arrEstGenericArgs[0] = estArrTypes[i + 1];
                    arrEstGenericArgs[1] = distArrTypes[i];
                    arrEstGenericArgs[2] = distArrTypes[i + 1];
                    if (rank == 1)
                        estArrTypes[i] = typeof (ArrayEstimator<,,>).MakeGenericType(arrEstGenericArgs);
                    else if (rank == 2)
                        estArrTypes[i] = typeof (Array2DEstimator<,,>).MakeGenericType(arrEstGenericArgs);
                }
                else
                {
                    Type[] arrEstGenericArgs = new Type[4];
                    arrEstGenericArgs[0] = estArrTypes[i + 1];
                    arrEstGenericArgs[1] = distArrTypes[i];
                    arrEstGenericArgs[2] = distArrTypes[i + 1];
                    arrEstGenericArgs[3] = domainTypes[i + 1];
                    if (rank == 1)
                        estArrTypes[i] = typeof (ArrayEstimator<,,,>).MakeGenericType(arrEstGenericArgs);
                    else if (rank == 2)
                        estArrTypes[i] = typeof (Array2DEstimator<,,,>).MakeGenericType(arrEstGenericArgs);
                }
            }
            // Now create the estimator array
            return (Estimator<T>) FromArray(estimators, estArrTypes, leafDistributionType);
        }

        /// <summary>
        /// Get the estimator type for a distribution
        /// </summary>
        /// <param name="distType">Distribution type</param>
        /// <param name="accumDist">Accumulate distributions rather than samples</param>
        /// <returns></returns>
        public static Type GetEstimatorType(Type distType, bool accumDist)
        {
            Type domainType = Distribution.GetDomainType(distType);
            if (!domainType.IsArray)
                return EstimatorFactory.Instance.EstimatorType(distType);

            Type[] domainTypes = JaggedArray.GetTypes(domainType);
            // Need to find the innermost distribution type - so we need to walk the type parameters
            Type t = (distType.GetGenericArguments())[0];
            while (t.IsGenericType)
                t = (t.GetGenericArguments())[0];
            Type leafDistributionType = t;
            Type leafDomainType = domainTypes[domainTypes.Length - 1];
            Type leafEstimatorType = EstimatorFactory.Instance.EstimatorType(leafDistributionType);
            if (leafEstimatorType == null)
                throw new InferRuntimeException(
                    String.Format("Cannot find Estimator class for {0}", leafDistributionType));

            // Get array of distribution types
            Type[] arrDistTypes = (JaggedArray.GetTypes(
                domainType, leafDomainType, leafDistributionType));

            int depth = domainTypes.Length;
            // Get distribution array types
            Type[] distArrTypes = new Type[depth];
            Type[] arrayEstTypes = new Type[depth];
            distArrTypes[depth - 1] = arrDistTypes[depth - 1];
            arrayEstTypes[depth - 1] = leafEstimatorType;
            for (int i = depth - 2; i >= 0; i--)
            {
                int rank = arrDistTypes[i].GetArrayRank();
                distArrTypes[i] = Distribution.MakeDistributionArrayType(distArrTypes[i + 1], rank);
                if (accumDist)
                {
                    Type[] arrEstGenericArgs = new Type[3];
                    arrEstGenericArgs[0] = arrayEstTypes[i + 1];
                    arrEstGenericArgs[1] = distArrTypes[i];
                    arrEstGenericArgs[2] = distArrTypes[i + 1];
                    if (rank == 1)
                        arrayEstTypes[i] = typeof (ArrayEstimator<,,>).MakeGenericType(arrEstGenericArgs);
                    else if (rank == 2)
                        arrayEstTypes[i] = typeof (Array2DEstimator<,,>).MakeGenericType(arrEstGenericArgs);
                }
                else
                {
                    Type[] arrEstGenericArgs = new Type[4];
                    arrEstGenericArgs[0] = arrayEstTypes[i + 1];
                    arrEstGenericArgs[1] = distArrTypes[i];
                    arrEstGenericArgs[2] = distArrTypes[i + 1];
                    arrEstGenericArgs[3] = domainTypes[i + 1];
                    if (rank == 1)
                        arrayEstTypes[i] = typeof (ArrayEstimator<,,,>).MakeGenericType(arrEstGenericArgs);
                    else if (rank == 2)
                        arrayEstTypes[i] = typeof (Array2DEstimator<,,,>).MakeGenericType(arrEstGenericArgs);
                }
            }
            return arrayEstTypes[0];
        }

        /// <summary>
        /// Convert an array of element estimators to an estimator over an array variable.
        /// </summary>
        /// <param name="arrayOfEstimators">Array of estimators</param>
        /// <param name="estimatorTypes">Estimator types at each depth</param>
        /// <param name="distType">Distribution type</param>
        /// <returns>Estimator over an array variable</returns>
        private static object FromArray(object arrayOfEstimators, Type[] estimatorTypes, Type distType)
        {
            Type arrType = arrayOfEstimators.GetType();
            if (arrType.IsArray)
            {
                object result = null;
                Type eltType = arrType.GetElementType();
                while (eltType.IsArray)
                    eltType = eltType.GetElementType();

                Type estType = typeof (ArrayEstimator<>).MakeGenericType(distType);
                MethodInfo[] staticMeths = estType.GetMethods(BindingFlags.Static | BindingFlags.Public);
                foreach (MethodInfo sm in staticMeths)
                {
                    if (sm.Name == "Array")
                    {
                        MethodInfo mi = sm.MakeGenericMethod(estimatorTypes[0], eltType);
                        if (mi.GetParameters()[0].ParameterType == arrType)
                        {
                            result = Util.Invoke(mi, null, arrayOfEstimators, estimatorTypes);
                            break;
                        }
                    }
                }
                return result;
            }
            else
                return arrayOfEstimators;
        }
    }

    /// <summary>
    /// Static class which implements useful functions on estimator arrays.
    /// </summary>
    public static class ArrayEstimator<T>
    {
        /// <summary>
        /// Creates an estimator over an array domain from an estimator array over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="array">The estimator of each element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(TEstimator[] array, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(estArrTypes[0], array);
        }

        /// <summary>
        /// Creates an estimator over an array domain from an estimator over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="length">The length of the array.</param>
        /// <param name="init">A function providing the estimator of each array element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(int length, Converter<int, TEstimator> init, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(estArrTypes[0], length, init);
        }

        /// <summary>
        /// Creates an estimator over an array domain from independent estimators over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="array">The estimator of each element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(TEstimator[,] array, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(estArrTypes[0], array);
        }

        /// <summary>
        /// Creates an estimator over an array domain from independent estimators over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="length1">The first dimension of the array.</param>
        /// <param name="length2">The second dimension of the array.</param>
        /// <param name="init">A function providing the estimator of each array element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(int length1, int length2, Func<int, int, TEstimator> init, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(estArrTypes[0], length1, length2, init);
        }

        /// <summary>
        /// Creates an estimator over an array domain from independent estimators over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="array">The estimator of each element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(TEstimator[][] array, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            MethodInfo method = new Func<TEstimator[][], Type, TEstArray>(ArrayEstimator<T>.Array11<TEstArray, TEstimator, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof (TEstArray), typeof (TEstimator), estArrTypes[1]);
            return (TEstArray) Util.Invoke(method, null, array, estArrTypes[0]);
        }

        private static TEstArray Array11<TEstArray, TEstimator, TItem>(TEstimator[][] array, Type arrayType)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(arrayType, array.Length, new Converter<int, TItem>(
                                                                                     delegate(int i) { return (TItem) Activator.CreateInstance(typeof (TItem), array[i]); }));
        }

        /// <summary>
        /// Creates an estimator over an array domain from independent estimators over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="array">The estimator of each element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(TEstimator[,][] array, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            MethodInfo method = new Func<TEstimator[,][], Type, TEstArray>(ArrayEstimator<T>.Array21<TEstArray, TEstimator, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof (TEstArray), typeof (TEstimator), estArrTypes[1]);
            return (TEstArray) Util.Invoke(method, null, array, estArrTypes[0]);
        }

        private static TEstArray Array21<TEstArray, TEstimator, TItem>(TEstimator[,][] array, Type arrayType)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(arrayType, array.GetLength(0), array.GetLength(1), new Func<int, int, TItem>(
                                                                                                               delegate(int i, int j)
                                                                                                                   {
                                                                                                                       return
                                                                                                                           (TItem)
                                                                                                                           Activator.CreateInstance(typeof (TItem), array[i, j]);
                                                                                                                   }));
        }

        /// <summary>
        /// Creates an estimator over an array domain from independent estimators over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="array">The estimator of each element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(TEstimator[][,] array, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            MethodInfo method = new Func<TEstimator[][,], Type, TEstArray>(ArrayEstimator<T>.Array12<TEstArray, TEstimator, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof (TEstArray), typeof (TEstimator), estArrTypes[1]);
            return (TEstArray) Util.Invoke(method, null, array, estArrTypes[0]);
        }

        private static TEstArray Array12<TEstArray, TEstimator, TItem>(TEstimator[][,] array, Type arrayType)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(arrayType, array.Length, new Converter<int, TItem>(
                                                                                     delegate(int i) { return (TItem) Activator.CreateInstance(typeof (TItem), array[i]); }));
        }

        /// <summary>
        /// Creates an estimator over an array domain from independent estimators over the elements.
        /// </summary>
        /// <typeparam name="TEstArray">Type of estimator array</typeparam>
        /// <typeparam name="TEstimator">Estimator type for an array element.</typeparam>
        /// <param name="array">The estimator of each element.</param>
        /// <param name="estArrTypes">Types of estimator array at each depth</param>
        /// <returns>A single estimator object over the array domain.</returns>
        public static TEstArray Array<TEstArray, TEstimator>(TEstimator[][][] array, Type[] estArrTypes)
            where TEstimator : Estimator<T>
        {
            MethodInfo method =
                new Func<TEstimator[][][], Type, TEstArray>(ArrayEstimator<T>.Array111<TEstArray, TEstimator, object, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(typeof (TEstArray), typeof (TEstimator), estArrTypes[2], estArrTypes[1]);
            return (TEstArray) Util.Invoke(method, null, array, estArrTypes[0]);
        }

        private static TEstArray Array111<TEstArray, TEstimator, TInner, TMiddle>(TEstimator[][][] array, Type arrayType)
            where TEstimator : Estimator<T>
        {
            return (TEstArray) Activator.CreateInstance(arrayType, array.Length, new Converter<int, TMiddle>(
                                                                                     delegate(int i)
                                                                                         {
                                                                                             return
                                                                                                 (TMiddle)
                                                                                                 Activator.CreateInstance(typeof (TMiddle), array[i].Length,
                                                                                                                          new Converter<int, TInner>(
                                                                                                                              delegate(int j)
                                                                                                                                  {
                                                                                                                                      return
                                                                                                                                          (TInner)
                                                                                                                                          Activator.CreateInstance(
                                                                                                                                              typeof (TInner), array[i][j]);
                                                                                                                                  }));
                                                                                         }));
        }
    }
}