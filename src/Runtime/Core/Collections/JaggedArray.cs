// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;
using System.Collections;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Class that provides useful static methods for jagged arrays
    /// </summary>
    public static class JaggedArray
    {
        /// <summary>
        /// Gets the innermost non-array type
        /// </summary>
        /// <param name="jaggedType">Jagged array type</param>
        /// <returns></returns>
        public static Type GetInnermostType(Type jaggedType)
        {
            if (jaggedType.IsArray)
                return GetInnermostType(jaggedType.GetElementType());
            else
                return jaggedType;
        }

        /// <summary>
        /// Gets the innermost non-array type
        /// </summary>
        /// <typeparam name="JaggedType">Jagged array type</typeparam>
        /// <returns></returns>
        public static Type GetInnermostType<JaggedType>()
        {
            return GetInnermostType(typeof (JaggedType));
        }

        /// <summary>
        /// Gets the ranks of a jagged array type when considered an array 
        /// </summary>
        /// <param name="jaggedType">The jagged array type</param>
        /// <param name="leafType">The leaf type</param>
        /// <returns></returns>
        public static int[] GetRanks(Type jaggedType, Type leafType)
        {
            List<int> ranks = new List<int>();
            innerGetRanks(jaggedType, leafType, ranks);
            return ranks.ToArray();
        }

        /// <summary>
        /// Gets the ranks of a jagged array type when considered an array 
        /// over the specified leaf type
        /// </summary>
        /// <typeparam name="JaggedType">The jagged array type</typeparam>
        /// <typeparam name="LeafType">The leaf type</typeparam>
        /// <returns></returns>
        public static int[] GetRanks<JaggedType, LeafType>()
        {
            List<int> ranks = new List<int>();
            innerGetRanks(typeof (JaggedType), typeof (LeafType), ranks);
            return ranks.ToArray();
        }

        /// <summary>
        /// Gets the ranks of a jagged array down to the first non-array type
        /// </summary>
        /// <param name="jaggedType">The jagged array type</param>
        /// <returns></returns>
        public static int[] GetRanks(Type jaggedType)
        {
            Type LeafType = GetInnermostType(jaggedType);
            return GetRanks(jaggedType, LeafType);
        }

        /// <summary>
        /// Gets the ranks of a jagged array down to the first non-array type
        /// </summary>
        /// <typeparam name="JaggedType">Jagged array type</typeparam>
        /// <returns></returns>
        public static int[] GetRanks<JaggedType>()
        {
            Type LeafType = GetInnermostType(typeof (JaggedType));
            return GetRanks(typeof (JaggedType), LeafType);
        }

        private static void innerGetRanks(Type jaggedType, Type LeafType, List<int> ranks)
        {
            if (jaggedType == LeafType)
                return;
            else if (jaggedType.IsArray)
            {
                ranks.Add(jaggedType.GetArrayRank());
                Type nextArrayType = jaggedType.GetElementType();
                if (nextArrayType == LeafType)
                    return;
                else
                    innerGetRanks(nextArrayType, LeafType, ranks);
            }
            else
                throw new ArgumentException("Not a jagged array over the specified leaf type");
        }

        /// <summary>
        /// Gets the depth of the jagged array when considered an
        /// array over the specified leaf type
        /// </summary>
        /// <param name="jaggedType"></param>
        /// <param name="leafType">The leaf type</param>
        /// <returns></returns>
        public static int GetDepth(Type jaggedType, Type leafType)
        {
            return GetRanks(jaggedType, leafType).Length;
        }

        /// <summary>
        /// Gets the depth of the jagged array when considered an
        /// array over the specified leaf type
        /// </summary>
        /// <typeparam name="JaggedType">Jagge array type</typeparam>
        /// <typeparam name="LeafType">The leaf type</typeparam>
        /// <returns></returns>
        public static int GetDepth<JaggedType, LeafType>()
        {
            return GetRanks<JaggedType, LeafType>().Length;
        }

        /// <summary>
        /// Gets the depth of the jagged array
        /// </summary>
        /// <param name="jaggedType"></param>
        /// <returns></returns>
        public static int GetDepth(Type jaggedType)
        {
            return GetRanks(jaggedType).Length;
        }

        /// <summary>
        /// Gets the depth of the jagged array
        /// </summary>
        /// <typeparam name="JaggedType">Jagged array type</typeparam>
        /// <returns></returns>
        public static int GetDepth<JaggedType>()
        {
            return GetRanks(typeof (JaggedType)).Length;
        }


        /// <summary>
        /// Gets the types at each depth of a jagged array when
        /// considered as a jagged array over the specified leaf type
        /// </summary>
        /// <param name="jaggedType">Jagged array type</param>
        /// <param name="leafType">Leaf type</param>
        /// <returns></returns>
        public static Type[] GetTypes(Type jaggedType, Type leafType)
        {
            int[] ranks = JaggedArray.GetRanks(jaggedType, leafType);
            Type[] types = new Type[ranks.Length + 1];
            types[ranks.Length] = leafType;
            for (int i = ranks.Length - 1; i >= 0; i--)
                types[i] = Util.MakeArrayType(types[i + 1], ranks[i]);
            return types;
        }

        /// <summary>
        /// Gets the types at each depth of a jagged array
        /// </summary>
        /// <param name="jaggedType">Jagged array type</param>
        /// <returns></returns>
        public static Type[] GetTypes(Type jaggedType)
        {
            Type leafType = GetInnermostType(jaggedType);
            return GetTypes(jaggedType, leafType);
        }

        /// <summary>
        /// Gets jagged array types for the target leaf type to match
        /// the given jagged array considered as an array over the specified
        /// leaf type
        /// </summary>
        /// <param name="jaggedType">Jagged array type</param>
        /// <param name="leafType">Leaf type</param>
        /// <param name="targetLeafType">Desired leaf type</param>
        /// <returns></returns>
        public static Type[] GetTypes(Type jaggedType, Type leafType, Type targetLeafType)
        {
            int[] ranks = JaggedArray.GetRanks(jaggedType, leafType);
            Type[] types = new Type[ranks.Length + 1];
            types[ranks.Length] = targetLeafType;
            for (int i = ranks.Length - 1; i >= 0; i--)
                types[i] = Util.MakeArrayType(types[i + 1], ranks[i]);
            return types;
        }


        /// <summary>
        /// Gets the types at each depth of a jagged array when
        /// considered as a jagged array over the specified leaf type
        /// </summary>
        /// <typeparam name="JaggedType">Jagged array type</typeparam>
        /// <typeparam name="LeafType">Leaf type</typeparam>
        /// <returns></returns>
        public static Type[] GetTypes<JaggedType, LeafType>()
        {
            return GetTypes(typeof (JaggedType), typeof (LeafType));
        }

        /// <summary>
        /// Gets the types at each depth of a jagged array
        /// </summary>
        /// <typeparam name="JaggedType">Jagged array type</typeparam>
        /// <returns></returns>
        public static Type[] GetTypes<JaggedType>()
        {
            return GetTypes(typeof (JaggedType));
        }

        /// <summary>
        /// Gets jagged array types for the target leaf type to match
        /// the given jagged array considered as an array over the specified
        /// leaf type
        /// </summary>
        /// <typeparam name="JaggedType">Jagged array type</typeparam>
        /// <typeparam name="LeafType">Leaf type</typeparam>
        /// <typeparam name="TargetLeafType">Target leaf type</typeparam>
        /// <returns></returns>
        public static Type[] GetTypes<JaggedType, LeafType, TargetLeafType>()
        {
            return GetTypes(typeof (JaggedType), typeof (LeafType), typeof (TargetLeafType));
        }

        /// <summary>
        /// Iterates over the elements of a jagged array when considered as
        /// an array over the specified leaf type
        /// </summary>
        /// <param name="jaggedArray">The jagged array</param>
        /// <param name="leafType">The leaf type</param>
        /// <returns></returns>
        public static IEnumerable ElementIterator(IEnumerable jaggedArray, Type leafType)
        {
            foreach (object elt in jaggedArray)
            {
                if (elt == null || leafType.IsAssignableFrom(elt.GetType())) yield return elt;
                else if (elt is IEnumerable)
                {
                    foreach (object subElt in ElementIterator((IEnumerable) elt, leafType))
                        yield return subElt;
                }
            }
        }

        /// <summary>
        /// Iterates over the elements of a jagged array when considered as
        /// an array over the specified leaf type
        /// </summary>
        /// <typeparam name="LeafType">Leaf type</typeparam>
        /// <param name="jaggedArray">The jagged array</param>
        /// <returns></returns>
        public static IEnumerable ElementIterator<LeafType>(IEnumerable jaggedArray)
        {
            foreach (LeafType elt in ElementIterator(jaggedArray, typeof (LeafType)))
                yield return elt;
        }

        /// <summary>
        /// Iterates over the elements of a jagged array
        /// </summary>
        /// <param name="jaggedArray">The jagged array</param>
        /// <returns></returns>
        public static IEnumerable ElementIterator(IEnumerable jaggedArray)
        {
            Type leafType = GetInnermostType(jaggedArray.GetType());
            foreach (object elt in ElementIterator(jaggedArray, leafType))
                yield return elt;
        }

        /// <summary>
        /// Gets the total length of the jagged array when considered as
        /// an array over the specified leaf type.
        /// </summary>
        /// <param name="jaggedArray">The jagged array</param>
        /// <param name="leafType">The leaf type</param>
        /// <returns></returns>
        public static int GetLength(IEnumerable jaggedArray, Type leafType)
        {
            int res = 0;
            foreach (object elt in ElementIterator(jaggedArray, leafType))
                res++;
            return res;
        }

        /// <summary>
        /// Gets the total long length of the jagged array when considered as
        /// an array over the specified leaftype.
        /// </summary>
        /// <param name="jaggedArray">The jagged array</param>
        /// <param name="leafType">The leaf type</param>
        /// <returns></returns>
        public static long GetLongLength(IEnumerable jaggedArray, Type leafType)
        {
            long res = 0;
            foreach (object elt in ElementIterator(jaggedArray, leafType))
                res++;
            return res;
        }

        /// <summary>
        /// Gets the total length of the jagged array when considered as
        /// an array over the specified leaf type.
        /// </summary>
        /// <typeparam name="LeafType">The leaf type</typeparam>
        /// <param name="jaggedArray">The jagged array</param>
        /// <returns></returns>
        public static int GetLength<LeafType>(IEnumerable jaggedArray)
        {
            int res = 0;
            foreach (object elt in ElementIterator<LeafType>(jaggedArray))
                res++;
            return res;
        }

        /// <summary>
        /// Gets the total long length of the jagged array when considered as
        /// an array over the specified leaf type.
        /// </summary>
        /// <typeparam name="LeafType">The leaf type</typeparam>
        /// <param name="jaggedArray">The jagged array</param>
        /// <returns></returns>
        public static long GetLongLength<LeafType>(IEnumerable jaggedArray)
        {
            long res = 0;
            foreach (object elt in ElementIterator(jaggedArray))
                res++;
            return res;
        }

        /// <summary>
        /// Gets the total length of the jagged array
        /// </summary>
        /// <param name="jaggedArray">The jagged array</param>
        /// <returns></returns>
        public static int GetLength(IEnumerable jaggedArray)
        {
            int res = 0;
            foreach (object elt in ElementIterator(jaggedArray))
                res++;
            return res;
        }

        /// <summary>
        /// Gets the total long length of the jagged array
        /// </summary>
        /// <param name="jaggedArray">The jagged array</param>
        /// <returns></returns>
        public static long GetLongLength(IEnumerable jaggedArray)
        {
            long res = 0;
            foreach (object elt in ElementIterator(jaggedArray))
                res++;
            return res;
        }

        /// <summary>
        /// Delegate for jagged array element converter
        /// </summary>
        /// <param name="elt">Jagged array element</param>
        public delegate object ElementConverter(object elt);

        /// <summary>
        /// Delegate for jagged array element converter
        /// </summary>
        /// <typeparam name="SourceLeafType">Type of leaf elements in source array</typeparam>
        /// <typeparam name="TargetLeafType">Type of leaf elements in target array</typeparam>
        /// <param name="elt"></param>
        /// <returns></returns>
        public delegate TargetLeafType ElementConverter<SourceLeafType, TargetLeafType>(SourceLeafType elt);

        /// <summary>
        /// Delegate for jagged array element converter
        /// </summary>
        /// <param name="elt1">Jagged array element 1</param>
        /// <param name="elt2">Jagged array element 2</param>
        public delegate object ElementConverter2(object elt1, object elt2);

        /// <summary>
        /// Creates a jagged array with the same structure as another jagged array
        /// </summary>
        /// <param name="sourceArray">The source jagged array</param>
        /// <param name="sourceLeafType">The leaf type of the source jagged array</param>
        /// <param name="targetLeafType">The leaf type of the target jagged array</param>
        /// <param name="converter">Element converter</param>
        /// <returns></returns>
        public static Array ConvertToNew(
            Array sourceArray, Type sourceLeafType, Type targetLeafType, ElementConverter converter)
        {
            int[] ranks = JaggedArray.GetRanks(sourceArray.GetType(), sourceLeafType);
            // Get the new types at each depth
            Type[] types = new Type[ranks.Length + 1];
            types[ranks.Length] = targetLeafType;
            for (int i = ranks.Length - 1; i >= 0; i--)
                types[i] = Util.MakeArrayType(types[i + 1], ranks[i]);
            Array targetJaggedArray = innerConvertToNew(0, types, sourceArray, converter);
            return targetJaggedArray;
        }

        /// <summary>
        /// Creates a jagged array with the same structure as another jagged array
        /// </summary>
        /// <param name="sourceArray">The source jagged array</param>
        /// <param name="targetLeafType">The leaf type of the target jagged array</param>
        /// <param name="converter">Element converter</param>
        /// <returns></returns>
        public static Array ConvertToNew(
            Array sourceArray, Type targetLeafType, ElementConverter converter)
        {
            int[] ranks = JaggedArray.GetRanks(sourceArray.GetType());
            // Get the new types at each depth
            Type[] types = new Type[ranks.Length + 1];
            types[ranks.Length] = targetLeafType;
            for (int i = ranks.Length - 1; i >= 0; i--)
                types[i] = Util.MakeArrayType(types[i + 1], ranks[i]);
            Array targetJaggedArray = innerConvertToNew(0, types, sourceArray, converter);
            return targetJaggedArray;
        }

        /// <summary>
        /// Creates a jagged array with the same structure as another jagged array
        /// </summary>
        /// <typeparam name="SourceLeafType">Leaf type of the source jagged array</typeparam>
        /// <typeparam name="TargetLeafType">Leaf type of the target jagged array</typeparam>
        /// <param name="sourceArray">The source array</param>
        /// <param name="converter">The converter</param>
        /// <returns></returns>
        public static Array ConvertToNew<SourceLeafType, TargetLeafType>(
            Array sourceArray, ElementConverter<SourceLeafType, TargetLeafType> converter)
        {
            int[] ranks = JaggedArray.GetRanks(sourceArray.GetType(), typeof (SourceLeafType));
            // Get the new types at each depth
            Type[] types = new Type[ranks.Length + 1];
            types[ranks.Length] = typeof (TargetLeafType);
            for (int i = ranks.Length - 1; i >= 0; i--)
                types[i] = Util.MakeArrayType(types[i + 1], ranks[i]);
            Array targetJaggedArray = innerConvertToNew<SourceLeafType, TargetLeafType>(0, types, sourceArray, converter);
            return targetJaggedArray;
        }

        private static int[] GetMultiDimensions(Array arr)
        {
            int[] dims = new int[arr.Rank];
            for (int i = 0; i < arr.Rank; i++)
                dims[i] = arr.GetLength(i);

            return dims;
        }

        private static Array innerConvertToNew(
            int depth, Type[] types, Array sourceArray, ElementConverter converter)
        {
            bool atLeaf = (++depth == types.Length - 1);

            int[] dims = GetMultiDimensions(sourceArray);
            Array targetJaggedArray = Array.CreateInstance(types[depth], dims);
            switch (sourceArray.Rank)
            {
                case 1:
                    if (atLeaf)
                    {
                        for (int i = 0; i < sourceArray.Length; i++)
                            targetJaggedArray.SetValue(converter(sourceArray.GetValue(i)), i);
                    }
                    else
                    {
                        Array[] src = (Array[]) sourceArray;
                        Array[] tgt = (Array[]) targetJaggedArray;
                        for (int i = 0; i < src.Length; i++)
                            tgt[i] = innerConvertToNew(depth, types, src[i], converter);
                    }
                    break;
                case 2:
                    if (atLeaf)
                    {
                        for (int i = 0; i < sourceArray.GetLength(0); i++)
                            for (int j = 0; j < sourceArray.GetLength(1); j++)
                                targetJaggedArray.SetValue(converter(sourceArray.GetValue(i, j)), i, j);
                    }
                    else
                    {
                        Array[,] src = (Array[,]) sourceArray;
                        Array[,] tgt = (Array[,]) targetJaggedArray;
                        for (int i = 0; i < src.GetLength(0); i++)
                            for (int j = 0; j < src.GetLength(1); j++)
                                tgt[i, j] = innerConvertToNew(depth, types, src[i, j], converter);
                    }
                    break;

                default:
                    throw new ArgumentException("Multidimensional arrays of greater than rank 2 are not supported");
            }
            return targetJaggedArray;
        }

        private static Array innerConvertToNew<SourceLeafType, TargetLeafType>(
            int depth, Type[] types, Array sourceArray, ElementConverter<SourceLeafType, TargetLeafType> converter)
        {
            bool atLeaf = (++depth == types.Length - 1);

            int[] dims = GetMultiDimensions(sourceArray);
            Array targetJaggedArray = Array.CreateInstance(types[depth], dims);
            switch (sourceArray.Rank)
            {
                case 1:
                    if (atLeaf)
                    {
                        for (int i = 0; i < sourceArray.Length; i++)
                            targetJaggedArray.SetValue(converter((SourceLeafType) sourceArray.GetValue(i)), i);
                    }
                    else
                    {
                        Array[] src = (Array[]) sourceArray;
                        Array[] tgt = (Array[]) targetJaggedArray;
                        for (int i = 0; i < src.Length; i++)
                            tgt[i] = innerConvertToNew(depth, types, src[i], converter);
                    }
                    break;
                case 2:
                    if (atLeaf)
                    {
                        for (int i = 0; i < sourceArray.GetLength(0); i++)
                            for (int j = 0; j < sourceArray.GetLength(1); j++)
                                targetJaggedArray.SetValue(converter((SourceLeafType) sourceArray.GetValue(i, j)), i, j);
                    }
                    else
                    {
                        Array[,] src = (Array[,]) sourceArray;
                        Array[,] tgt = (Array[,]) targetJaggedArray;
                        for (int i = 0; i < src.GetLength(0); i++)
                            for (int j = 0; j < src.GetLength(1); j++)
                                tgt[i, j] = innerConvertToNew(depth, types, src[i, j], converter);
                    }
                    break;

                default:
                    throw new ArgumentException("Multidimensional arrays of greater than rank 2 are not supported");
            }
            return targetJaggedArray;
        }


        /// <summary>
        /// Sets the elements of a jagged array
        /// </summary>
        /// <param name="jaggedArray">The source jagged array</param>
        /// <param name="leafType">The leaf type of the jagged array</param>
        /// <param name="converter">Element converter</param>
        /// <returns></returns>
        public static void ConvertElements(
            Array jaggedArray, Type leafType, ElementConverter converter)
        {
            int leafDepth = GetDepth(jaggedArray.GetType(), leafType);
            innerConvertElements(0, leafDepth, jaggedArray, converter);
        }

        /// <summary>
        /// Sets the elements of a jagged array
        /// </summary>
        /// <param name="jaggedArray">The source jagged array</param>
        /// <param name="converter">Element converter</param>
        /// <returns></returns>
        public static void ConvertElements(
            Array jaggedArray, ElementConverter converter)
        {
            int leafDepth = GetDepth(jaggedArray.GetType());
            innerConvertElements(0, leafDepth, jaggedArray, converter);
        }

        private static void innerConvertElements(
            int depth, int leafDepth, Array jaggedArray, ElementConverter converter)
        {
            bool atLeaf = (++depth == leafDepth);
            switch (jaggedArray.Rank)
            {
                case 1:
                    if (atLeaf)
                    {
                        for (int i = 0; i < jaggedArray.Length; i++)
                            jaggedArray.SetValue(converter(jaggedArray.GetValue(i)), i);
                    }
                    else
                    {
                        Array[] ja = (Array[]) jaggedArray;
                        for (int i = 0; i < ja.Length; i++)
                            innerConvertElements(depth, leafDepth, ja[i], converter);
                    }
                    break;
                case 2:
                    if (atLeaf)
                    {
                        for (int i = 0; i < jaggedArray.GetLength(0); i++)
                            for (int j = 0; j < jaggedArray.GetLength(1); j++)
                                jaggedArray.SetValue(converter(jaggedArray.GetValue(i, j)), i, j);
                    }
                    else
                    {
                        Array[,] ja = (Array[,]) jaggedArray;
                        for (int i = 0; i < ja.GetLength(0); i++)
                            for (int j = 0; j < ja.GetLength(1); j++)
                                innerConvertElements(depth, leafDepth, ja[i, j], converter);
                    }
                    break;
                default:
                    throw new ArgumentException("Multidimensional arrays of greater than rank 2 are not supported");
            }
        }

        /// <summary>
        /// Sets the elements of a jagged array given another jagged
        /// </summary>
        /// <param name="targetArray">The target array - also acts as a source</param>
        /// <param name="sourceArray">The source array</param>
        /// <param name="leafType">The leaf type of the target array</param>
        /// <param name="converter">Element converter</param>
        /// <returns></returns>
        public static void ConvertElements2(
            Array targetArray, Array sourceArray, Type leafType, ElementConverter2 converter)
        {
            int leafDepth = GetDepth(targetArray.GetType(), leafType);
            innerConvertElements2(0, leafDepth, targetArray, sourceArray, converter);
        }

        /// <summary>
        /// Sets the elements of a jagged array given another jagged
        /// </summary>
        /// <param name="targetArray">The target array - also acts as a source</param>
        /// <param name="sourceArray">The source array</param>
        /// <param name="converter">Element converter</param>
        /// <returns></returns>
        public static void ConvertElements2(
            Array targetArray, Array sourceArray, ElementConverter2 converter)
        {
            int leafDepth = GetDepth(targetArray.GetType());
            innerConvertElements2(0, leafDepth, targetArray, sourceArray, converter);
        }

        private static void innerConvertElements2(
            int depth, int leafDepth, Array targetArray, Array sourceArray, ElementConverter2 converter)
        {
            bool atLeaf = (++depth == leafDepth);
            switch (targetArray.Rank)
            {
                case 1:
                    if (atLeaf)
                    {
                        for (int i = 0; i < targetArray.Length; i++)
                            targetArray.SetValue(converter(targetArray.GetValue(i), sourceArray.GetValue(i)), i);
                    }
                    else
                    {
                        Array[] ja1 = (Array[]) targetArray;
                        Array[] ja2 = (Array[]) sourceArray;
                        for (int i = 0; i < ja1.Length; i++)
                            innerConvertElements2(depth, leafDepth, ja1[i], ja2[i], converter);
                    }
                    break;
                case 2:
                    if (atLeaf)
                    {
                        for (int i = 0; i < targetArray.GetLength(0); i++)
                            for (int j = 0; j < targetArray.GetLength(1); j++)
                                targetArray.SetValue(converter(targetArray.GetValue(i, j), sourceArray.GetValue(i, j)), i, j);
                    }
                    else
                    {
                        Array[,] ja1 = (Array[,]) targetArray;
                        Array[,] ja2 = (Array[,]) sourceArray;
                        for (int i = 0; i < ja1.GetLength(0); i++)
                            for (int j = 0; j < ja1.GetLength(1); j++)
                                innerConvertElements2(depth, leafDepth, ja1[i, j], ja2[i, j], converter);
                    }
                    break;
                default:
                    throw new ArgumentException("Multidimensional arrays of greater than rank 2 are not supported");
            }
        }

        /// <summary>
        /// Delegate for jagged array element visitor
        /// </summary>
        /// <param name="elt">Jagged array element</param>
        public delegate void ElementAction(object elt);

        /// <summary>
        /// Visits all elements with the specified leaf type, and perform the action
        /// </summary>
        /// <param name="jaggedArray">Jagged array</param>
        /// <param name="leafType">Element type</param>
        /// <param name="action">Action delegate</param>
        public static void VisitElements(IEnumerable jaggedArray, Type leafType, ElementAction action)
        {
            foreach (object elt in ElementIterator(jaggedArray, leafType))
                action(elt);
        }

        /// <summary>
        /// Visits all elements and perform the action
        /// </summary>
        /// <param name="jaggedArray">Jagged array</param>
        /// <param name="action">Action delegate</param>
        public static void VisitElements(IEnumerable jaggedArray, ElementAction action)
        {
            foreach (object elt in ElementIterator(jaggedArray))
                action(elt);
        }

        /// <summary>
        /// Delegate for generic jagged array element visitor
        /// </summary>
        /// <param name="elt1">Jagged array element 1</param>
        /// <param name="elt2">Jagged array element 2</param>
        public delegate void ElementAction2(object elt1, object elt2);

        /// <summary>
        /// Visits all elements oftwo jagged arrays, and perform the action
        /// </summary>
        /// <param name="jaggedArray1">First jagged array</param>
        /// <param name="jaggedArray2">Second jagged array</param>
        /// <param name="leafType1">Leaf type of first jagged array element</param>
        /// <param name="leafType2">Leaf type of second jagged array element</param>
        /// <param name="action">The action to take</param>
        /// <remarks>There is no checking of compatibility between the two jagged arrays</remarks>
        public static void VisitElements2(
            IEnumerable jaggedArray1, IEnumerable jaggedArray2, Type leafType1, Type leafType2, ElementAction2 action)
        {
            IEnumerator en1 = JaggedArray.ElementIterator(jaggedArray1, leafType1).GetEnumerator();
            IEnumerator en2 = JaggedArray.ElementIterator(jaggedArray2, leafType2).GetEnumerator();
            while (en1.MoveNext() && en2.MoveNext())
                action(en1.Current, en2.Current);
        }

        /// <summary>
        /// Visits all elements oftwo jagged arrays, and perform the action
        /// </summary>
        /// <param name="jaggedArray1">First jagged array</param>
        /// <param name="jaggedArray2">Second jagged array</param>
        /// <param name="action">The action to take</param>
        /// <remarks>There is no checking of compatibility between the two jagged arrays</remarks>
        public static void VisitElements2(
            IEnumerable jaggedArray1, IEnumerable jaggedArray2, ElementAction2 action)
        {
            IEnumerator en1 = JaggedArray.ElementIterator(jaggedArray1).GetEnumerator();
            IEnumerator en2 = JaggedArray.ElementIterator(jaggedArray2).GetEnumerator();
            while (en1.MoveNext() && en2.MoveNext())
                action(en1.Current, en2.Current);
        }
    }
}