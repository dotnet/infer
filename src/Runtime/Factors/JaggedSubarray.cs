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
    [FactorMethod(typeof(Factor), "JaggedSubarray<>", Default = true)]
    [Buffers("marginal")]
    [Quality(QualityBand.Mature)]
    public static class JaggedSubarrayOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogAverageFactor{ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where ItemType : IList<T>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where ItemType : IList<T>
        {
            return LogAverageFactor<ItemType>(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="AverageLogFactor{ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double AverageLogFactor<ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where ItemType : IList<T>
        {
            return LogAverageFactor<ItemType>(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType, ItemType}(IList{ItemType}, IList{DistributionType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<DistributionType, ItemType>(IList<ItemType> items, IList<DistributionType> array, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<DistributionType>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value;
                    if (!productBefore.TryGetValue(indices[i][j], out value))
                    {
                        value = (DistributionType)array[indices[i][j]].Clone();
                    }
                    z += value.GetLogAverageOf(items[i][j]);
                    value.SetToProduct(value, items[i][j]);
                    productBefore[indices[i][j]] = value;
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType, ItemType}(IList{DistributionType}, IList{ItemType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<DistributionType, ItemType>(IList<DistributionType> array, IList<ItemType> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
            where ItemType : IList<T>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value;
                    if (!productBefore.TryGetValue(indices[i][j], out value))
                    {
                        value = array[indices[i][j]];
                    }
                    z += value.GetLogProb(items[i][j]);
                    value.Point = items[i][j];
                    productBefore[indices[i][j]] = value;
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType, ItemType}(IList{DistributionType}, IList{ItemType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<DistributionType, ItemType>(IList<DistributionType> array, IList<ItemType> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
            where ItemType : IList<T>
        {
            return LogAverageFactor<DistributionType, ItemType>(array, items, indices);
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType, ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<DistributionType, ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="MarginalInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a message from <c>array</c>.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="Marginal{ArrayType, DistributionType, Object, ItemType}(ArrayType, IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a message from <c>array</c>.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [SkipIfAllUniform("array", "items")]
        public static ArrayType Marginal<ArrayType, DistributionType, ItemType>(
            ArrayType array, [NoInit] IList<ItemType> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            if(items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    DistributionType value = result[indices_i_j];
                    value.SetToProduct(value, items_i[j]);
                    result[indices_i_j] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, DistributionType, ItemType}(ArrayType, ItemType, ItemType, IList{IList{int}}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, DistributionType, ItemType>(
            ArrayType result,
            ItemType to_item,
            [SkipIfUniform] ItemType item, // SkipIfUniform on 'item' causes this line to be pruned when the backward messages aren't changing
            IList<IList<int>> indices,
            int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                var indices_i_j = indices_i[j];
                DistributionType value = result[indices_i_j];
                value.SetToProduct(to_item[j], item[j]);
                result[indices_i_j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, DistributionType, ItemType}(ItemType, ArrayType, ArrayType, IList{IList{int}}, int, ItemType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a message from <c>array</c>.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ArrayType, DistributionType, ItemType>(
            [Indexed, Cancels] ItemType items, // items dependency must be ignored for Sequential schedule
            [IgnoreDependency] ArrayType array,
            [SkipIfAllUniform] ArrayType marginal,
            IList<IList<int>> indices,
            int resultIndex,
            ItemType result)
            where ArrayType : IList<DistributionType>
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            if(result.Count != indices[i].Count)
                throw new ArgumentException($"result.Count ({result.Count}) != indices[{i}].Count ({indices[i].Count})");
            // indices_i are all different
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetToRatio(marginal[indices_i[j]], items[j], GaussianOp.ForceProper);
                result[j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType, ItemType}(IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType, ItemType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            if(items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
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
                    value.SetToProduct(value, items_i[j]);
                    result[indices_i_j] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType, ItemType}(IList{IList{int}}, IList{ItemType}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType, ItemType>(
            IList<IList<int>> indices, IList<ItemType> items, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<T>
            where DistributionType : HasPoint<T>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = result[indices[i][j]];
                    value.Point = items[i][j];
                    result[indices[i][j]] = value;
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
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType, ItemType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = result[indices[i][j]];
                    value.SetToProduct(value, items[i][j]);
                    result[indices[i][j]] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType, ItemType}(IList{IList{int}}, IList{ItemType}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType, ItemType>(
            IList<IList<int>> indices, IList<ItemType> items, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<T>
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType, ItemType>(indices, items, result);
        }
    }

    [FactorMethod(typeof(Factor), "JaggedSubarray<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class JaggedSubarrayOp_NoDivide<T>
    {
        public static ArrayType ItemsAverageConditional<DistributionType, ArrayType, ItemType>(
            ArrayType items, 
            [SkipIfAllUniform] IList<DistributionType> array, 
            IList<int[]> indices, 
            IList<DistributionType> to_array, 
            ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            if(result.Count != indices.Count)
                throw new ArgumentException($"result.Count ({result.Count}) != indices.Count ({indices.Count})");
            Dictionary<int, int[]> itemsOfArray = new Dictionary<int, int[]>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Length; j++)
                {
                    int index = indices[i][j];
                    int[] itemIndices;
                    if (!itemsOfArray.TryGetValue(index, out itemIndices))
                    {
                        itemIndices = Util.ArrayInit(indices.Count, k => -1);
                        itemsOfArray[index] = itemIndices;
                    }
                    itemIndices[i] = j;
                }
            }
            for (int i = 0; i < indices.Count; i++) {
                for (int j = 0; j < indices[i].Length; j++) {
                    DistributionType value = result[i][j];
                    int index = indices[i][j];
                    int[] itemIndices = itemsOfArray[index];
                    bool firstTime = true;
                    for (int other_i = 0; other_i < itemIndices.Length; other_i++)
                    {
                        if (other_i == i)
                            continue;
                        int itemIndex = itemIndices[other_i];
                        if (itemIndex == -1)
                            continue;
                        if (firstTime)
                        {
                            value.SetToProduct(array[index], items[other_i][itemIndex]);
                            firstTime = false;
                        }
                        else
                        {
                            value.SetToProduct(value, items[other_i][itemIndex]);
                        }
                    }
                    result[i][j] = value;
                }
            }
            return result;
        }
    }

    //// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an array item.</typeparam>
    [FactorMethod(typeof(Factor), "JaggedSubarrayWithMarginal<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class JaggedSubarrayWithMarginalOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogAverageFactor{ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where ItemType : IList<T>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where ItemType : IList<T>
        {
            return LogAverageFactor<ItemType>(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="AverageLogFactor{ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double AverageLogFactor<ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where ItemType : IList<T>
        {
            return LogAverageFactor<ItemType>(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType, ItemType}(IList{ItemType}, IList{DistributionType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<DistributionType, ItemType>(IList<ItemType> items, IList<DistributionType> array, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<DistributionType>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value;
                    if (!productBefore.TryGetValue(indices[i][j], out value))
                    {
                        value = (DistributionType)array[indices[i][j]].Clone();
                    }
                    z += value.GetLogAverageOf(items[i][j]);
                    value.SetToProduct(value, items[i][j]);
                    productBefore[indices[i][j]] = value;
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType, ItemType}(IList{ItemType}, IList{DistributionType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType, ItemType>(IList<ItemType> items, IList<DistributionType> array, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType, ItemType}(IList{DistributionType}, IList{ItemType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<DistributionType, ItemType>(IList<DistributionType> array, IList<ItemType> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
            where ItemType : IList<T>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value;
                    if (!productBefore.TryGetValue(indices[i][j], out value))
                    {
                        value = array[indices[i][j]];
                    }
                    z += value.GetLogProb(items[i][j]);
                    value.Point = items[i][j];
                    productBefore[indices[i][j]] = value;
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType, ItemType}(IList{DistributionType}, IList{ItemType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<DistributionType, ItemType>(IList<DistributionType> array, IList<ItemType> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
            where ItemType : IList<T>
        {
            return LogAverageFactor<DistributionType, ItemType>(array, items, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType, ItemType}(IList{DistributionType}, IList{ItemType}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType, ItemType>(IList<DistributionType> array, IList<ItemType> items, IList<IList<int>> indices)
            where DistributionType : IDistribution<T>, CanGetLogProb<T>
            where ItemType : IList<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType, ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<DistributionType, ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
                for (int j = 0; j < indices[i].Count; j++)
                    z += items[i][j].GetLogProb(array[indices[i][j]]);
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType, ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType, ItemType>(IList<ItemType> items, IList<T> array, IList<IList<int>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType, ItemType}(IList{ItemType}, IList{T}, IList{IList{int}})"]/*'/>
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
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType, ItemArrayType, ItemType}(ItemArrayType, IList{DistributionType}, IList{IList{int}}, ItemArrayType)"]/*'/>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="Marginal{ArrayType, DistributionType, Object, ItemType}(ArrayType, IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a message from <c>array</c>.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [SkipIfAllUniform("array", "items")]
        public static ArrayType MarginalAverageConditional<ArrayType, DistributionType, ItemType>(
            ArrayType array, [NoInit] IList<ItemType> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    DistributionType value = result[indices_i_j];
                    value.SetToProduct(value, items_i[j]);
                    result[indices_i_j] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="Marginal{ArrayType, DistributionType, Object, ItemType}(ArrayType, IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a message from <c>array</c>.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemArrayType">The type of an incoming message from <c>items</c>.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [SkipIfAllUniform("array", "items")]
        public static ArrayType MarginalAverageConditional<ArrayType, DistributionType, ItemArrayType, ItemType>(
            ArrayType array, [NoInit] IList<ItemType> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : HasPoint<T>
            where ItemType : IList<T>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, DistributionType, ItemType}(ArrayType, ItemType, ItemType, IList{IList{int}}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType MarginalIncrementItems<ArrayType, DistributionType, ItemType>(
            [Indexed, SkipIfUniform] ItemType item, // SkipIfUniform on 'item' causes this line to be pruned when the backward messages aren't changing
            [Indexed, Cancels] ItemType to_item,    // Cancels since updating to_item does not require recomputing the increment
            IList<IList<int>> indices,
            int resultIndex,
            ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                var indices_i_j = indices_i[j];
                DistributionType value = result[indices_i_j];
                value.SetToProduct(to_item[j], item[j]);
                result[indices_i_j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, DistributionType, ItemType}(ItemType, ArrayType, ArrayType, IList{IList{int}}, int, ItemType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a message from <c>array</c>.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ArrayType, DistributionType, ItemType>(
            [Indexed, Cancels] ItemType items, // items dependency must be ignored for Sequential schedule
            ArrayType array,
            [SkipIfAllUniform] ArrayType to_marginal,
            IList<IList<int>> indices,
            int resultIndex,
            ItemType result)
            where ArrayType : IList<DistributionType>
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            if(result.Count != indices[i].Count)
                throw new ArgumentException($"result.Count ({result.Count}) != indices[{i}].Count ({indices[i].Count})");
            // indices_i are all different
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetToRatio(to_marginal[indices_i[j]], items[j], GaussianOp.ForceProper);
                result[j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType, ItemType}(IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<ArrayType>(
            [Cancels] ArrayType array,
            [SkipIfAllUniform] ArrayType to_marginal,
            ArrayType result)
            where ArrayType : SettableToRatio<ArrayType>
        {
            result.SetToRatio(to_marginal, array, GaussianOp.ForceProper);
            return result;
        }

        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType MarginalIncrementArray<ArrayType>(
            [SkipIfUniform] ArrayType array,  // SkipIfUniform on 'array' causes this line to be pruned when the incoming message isn't changing
            [Cancels] ArrayType to_array,     // Cancels since updating to_array does not require recomputing the increment
            ArrayType result)
            where ArrayType : SettableToProduct<ArrayType>
        {
            result.SetToProduct(array, to_array);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType, ItemType}(IList{IList{int}}, IList{ItemType}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType, ItemType>(
            IList<IList<int>> indices, IList<ItemType> items, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<T>
            where DistributionType : HasPoint<T>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = result[indices[i][j]];
                    value.Point = items[i][j];
                    result[indices[i][j]] = value;
                }
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        public static ArrayType MarginalAverageLogarithm<ArrayType>(
            ArrayType array, ArrayType result)
            where ArrayType : SettableTo<ArrayType>
        {
            result.SetTo(array);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="ItemsAverageLogarithm{DistributionType, ItemType, ResultType}(IList{DistributionType}, IList{IList{int}}, ResultType)"]/*'/>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType, ItemType}(IList{ItemType}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType, ItemType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<int>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<DistributionType>
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException($"items.Count ({items.Count}) != indices.Count ({indices.Count})");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                for (int j = 0; j < indices[i].Count; j++)
                {
                    DistributionType value = result[indices[i][j]];
                    value.SetToProduct(value, items[i][j]);
                    result[indices[i][j]] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="JaggedSubarrayWithMarginalOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType, ItemType}(IList{IList{int}}, IList{ItemType}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType, ItemType>(
            IList<IList<int>> indices, IList<ItemType> items, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where ItemType : IList<T>
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType, ItemType>(indices, items, result);
        }
    }
}
