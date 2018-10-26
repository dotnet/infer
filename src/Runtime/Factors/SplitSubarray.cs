// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Distributions;
    using Math;
    using Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an array item.</typeparam>
    [FactorMethod(typeof(Factor), "SplitSubarray<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class SplitSubarrayOp<T>
    {
        public static double LogAverageFactor(IReadOnlyList<IReadOnlyList<T>> items, IReadOnlyList<T> array, IList<IList<int>> indices)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    if (!equalityComparer.Equals(items[i][j], array[indices[i][j]]))
                        return Double.NegativeInfinity;
                }
            }
            return 0.0;
        }

        public static double LogEvidenceRatio(IReadOnlyList<IReadOnlyList<T>> items, IReadOnlyList<T> array, IList<IList<int>> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        public static double AverageLogFactor(IReadOnlyList<IReadOnlyList<T>> items, IReadOnlyList<T> array, IList<IList<int>> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<IReadOnlyList<DistributionType>> items, IList<DistributionType> array, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogAverageOf<DistributionType>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = array[indices[i][j]];
                    z += value.GetLogAverageOf(items[i][j]);
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType, ItemType}(IList{ItemType}, IList{DistributionType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType, ItemType>(IList<ItemType> items, IList<DistributionType> array, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<DistributionType> array, IReadOnlyList<IReadOnlyList<T>> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = array[indices[i][j]];
                    z += value.GetLogProb(items[i][j]);
                }
            }
            return z;
        }

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(IList<DistributionType> array, IReadOnlyList<IReadOnlyList<T>> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
        {
            return LogAverageFactor<DistributionType>(array, items, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType, ItemType}(IList{DistributionType}, IList{ItemType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType, ItemType>(IList<DistributionType> array, IList<ItemType> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
            where ItemType : IList<T>
        {
            return 0.0;
        }

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<IReadOnlyList<DistributionType>> items, IReadOnlyList<T> array, IList<IList<int>> indices)
            where DistributionType : CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
                for (int j = 0; j < indices[i].Count; j++)
                    z += items[i][j].GetLogProb(array[indices[i][j]]);
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType, ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType, ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType, ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType, ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

#if true
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType, ItemArrayType, ItemType}(ItemArrayType, IList{DistributionType}, IList{IList{int}}, ItemArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemArrayType">The type of an incoming message from <c>items</c>.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<DistributionType, ItemArrayType, ItemType>(
            ItemArrayType items, IList<DistributionType> array, IList<IList<int>> indices, ItemArrayType to_items)
            where ItemArrayType : IList<ItemType>
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count == 0)
                return 0.0;
            if (items.Count == 1 && items[0].Count <= 1)
                return 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    int index = indices[i][j];
                    DistributionType value;
                    if (!productBefore.TryGetValue(index, out value))
                    {
                        value = (DistributionType)array[index].Clone();
                    }
                    z += value.GetLogAverageOf(items[i][j]);
                    value.SetToProduct(value, items[i][j]);
                    productBefore[index] = value;
                    z -= to_items[i][j].GetLogAverageOf(items[i][j]);
                }
            }
            return z;
        }
#else
    /// <summary>
    /// Evidence message for EP
    /// </summary>
    /// <param name="items">Incoming message from 'items'.</param>
    /// <param name="array">Incoming message from 'array'.</param>
    /// <param name="indices">Constant value for 'indices'.</param>
    /// <returns>Logarithm of the factor's contribution the EP model evidence</returns>
    /// <remarks><para>
    /// The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>.
    /// Adding up these values across all factors and variables gives the log-evidence estimate for EP.
    /// </para></remarks>
        public static double LogEvidenceRatio<DistributionType, ItemArrayType, ItemType>(ItemArrayType items, IList<DistributionType> array, IList<IList<int>> indices)
            where ItemArrayType : IList<ItemType>, ICloneable
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count == 0) return 0.0;
            if (items.Count == 1 && items[0].Count <= 1) return 0.0;
            bool firstTime = true;
            DistributionType uniform = default(DistributionType);
            ItemArrayType productBefore = (ItemArrayType)items.Clone();
            ItemArrayType productAfter = (ItemArrayType)items.Clone();
            Dictionary<int,KeyValuePair<int,int>> indexToItem = new Dictionary<int, KeyValuePair<int, int>>();
            for (int i = 0; i < indices.Count; i++) {
                for (int j = 0; j < indices[i].Count; j++) {
                    KeyValuePair<int,int> previousItem;
                    if (!indexToItem.TryGetValue(indices[i][j], out previousItem)) {
                        // no previous item with this index
                        productBefore[i][j] = array[indices[i][j]];
                    } else {
                        DistributionType temp = productBefore[i][j];
                        temp.SetToProduct(productBefore[previousItem.Key][previousItem.Value], items[previousItem.Key][previousItem.Value]);
                        productBefore[i][j] = temp;
                    }
                    z += productBefore[i][j].GetLogAverageOf(items[i][j]);
                    indexToItem[indices[i][j]] = new KeyValuePair<int, int>(i, j);
                    if (firstTime) {
                        uniform = (DistributionType)items[i][j].Clone();
                        uniform.SetToUniform();
                    }
                }
            }
            indexToItem.Clear();
            for (int i = indices.Count - 1; i >= 0; i--) {
                for (int j = indices[i].Length - 1; j >= 0; j--) {
                    KeyValuePair<int,int> itemAfter;
                    if (!indexToItem.TryGetValue(indices[i][j], out itemAfter)) {
                        // no item after with this index
                        productAfter[i][j] = uniform;
                    } else {
                        DistributionType temp = productAfter[i][j];
                        temp.SetToProduct(productAfter[itemAfter.Key][itemAfter.Value], items[itemAfter.Key][itemAfter.Value]);
                        productAfter[i][j] = temp;
                    }
                    DistributionType toItem = (DistributionType)items[i][j].Clone();
                    toItem.SetToProduct(productBefore[i][j], productAfter[i][j]);
                    z -= toItem.GetLogAverageOf(items[i][j]);
                    indexToItem[indices[i][j]] = new KeyValuePair<int,int>(i,j);
                }
            }
            return z;
        }
#endif

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<DistributionType, ItemType>(
            [SkipIfAllUniform] IReadOnlyList<DistributionType> array,
            IList<IList<int>> indices,
            int resultIndex,
            ItemType result)
            where ItemType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            int i = resultIndex;
            if (result.Count != indices[i].Count) throw new ArgumentException($"result.Count ({result.Count}) != indices[{i}].Count ({indices[i].Count})");
            // indices_i are all different
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetTo(array[indices_i[j]]);
                result[j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType, ItemType}(IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IReadOnlyList<IReadOnlyList<DistributionType>> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            if (items.Count != indices.Count) throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetToUniform();
            var indices_Count = indices.Count;
            for (int i = 0; i < indices_Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    DistributionType value = result[indices_i_j];
                    value.SetTo(items_i[j]);
                    result[indices_i_j] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType, ItemType}(IList{IList{int}}, IList{ItemType}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            IList<IList<int>> indices, IReadOnlyList<IReadOnlyList<T>> items, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            if (items.Count != indices.Count) throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetToUniform();
            var indices_Count = indices.Count;
            for (int i = 0; i < indices_Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    DistributionType value = result[indices_i_j];
                    value.Point = items_i[j];
                    result[indices_i_j] = value;
                }
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ItemsAverageLogarithm{DistributionType, ItemType, ResultType}(IList{DistributionType}, IList{IList{int}}, ResultType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="ResultType">The type of the outgoing message.</typeparam>
        public static ResultType ItemsAverageLogarithm<DistributionType, ItemType, ResultType>(
            [SkipIfAllUniform] IList<DistributionType> array, IList<IList<int>> indices, ResultType result)
            where ResultType : IList<ItemType>
            where ItemType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = result[i][j];
                    value.SetTo(array[indices[i][j]]);
                    result[i][j] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType, ItemType}(IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(
            [SkipIfAllUniform] IReadOnlyList<IReadOnlyList<DistributionType>> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType, ItemType}(IList{IList{int}}, IList{ItemType}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(
            IList<IList<int>> indices, IReadOnlyList<IReadOnlyList<T>> items, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(indices, items, result);
        }
    }
}
