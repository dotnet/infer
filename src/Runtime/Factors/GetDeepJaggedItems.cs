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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetDeepJaggedItems<>", Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("marginal")]
    public static class GetDeepJaggedItemsOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogAverageFactor(IList{IList{IList{T}}}, IList{T}, IList{IList{IList{int}}})"]/*'/>
        public static double LogAverageFactor(IList<IList<IList<T>>> items, IList<T> array, IList<IList<IList<int>>> indices)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    for (int k = 0; k < indices_i_j.Count; k++)
                    {
                        if (!equalityComparer.Equals(items_i_j[k], array[indices_i_j[k]]))
                            return Double.NegativeInfinity;
                    }
                }
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogEvidenceRatio(IList{IList{IList{T}}}, IList{T}, IList{IList{IList{int}}})"]/*'/>
        public static double LogEvidenceRatio(IList<IList<IList<T>>> items, IList<T> array, IList<IList<IList<int>>> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="AverageLogFactor(IList{IList{IList{T}}}, IList{T}, IList{IList{IList{int}}})"]/*'/>
        public static double AverageLogFactor(IList<IList<IList<T>>> items, IList<T> array, IList<IList<IList<int>>> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogAverageFactor{ItemType, ItemType2,DistributionType}(IList{ItemType}, IList{DistributionType}, IList{IList{IList{int}}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<ItemType, ItemType2, DistributionType>(IList<ItemType> items, IList<DistributionType> array, IList<IList<IList<int>>> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<ItemType2>
            where ItemType2 : IList<DistributionType>
        {
            double z = 0.0;
            var productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    var indices_i_j_Count = indices_i_j.Count;
                    for (int k = 0; k < indices_i_j_Count; k++)
                    {
                        var indices_i_j_k = indices_i_j[k];
                        DistributionType value;
                        if (!productBefore.TryGetValue(indices_i_j_k, out value))
                        {
                            value = (DistributionType)array[indices_i_j_k].Clone();
                        }
                        var items_i_j_k = items_i_j[k];
                        z += value.GetLogAverageOf(items_i_j_k);
                        value.SetToProduct(value, items_i_j_k);
                        productBefore[indices_i_j_k] = value;
                    }
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="AverageLogFactor{ItemType, ItemType2, DistributionType}(IList{ItemType}, IList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<ItemType, ItemType2, DistributionType>(IList<ItemType> items, IList<DistributionType> array)
            where DistributionType : SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<ItemType2>
            where ItemType2 : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IList{IList{IList{T}}}, IList{DistributionType}, IList{IList{IList{int}}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<IList<IList<T>>> items, IList<DistributionType> array, IList<IList<IList<int>>> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            double z = 0.0;
            var productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    var indices_i_j_Count = indices_i_j.Count;
                    for (int k = 0; k < indices_i_j_Count; k++)
                    {
                        var indices_i_j_k = indices_i_j[k];
                        DistributionType value;
                        if (!productBefore.TryGetValue(indices_i_j_k, out value))
                        {
                            value = array[indices_i_j_k];
                        }
                        var items_i_j_k = items_i_j[k];
                        z += value.GetLogProb(items_i_j_k);
                        value.Point = items_i_j_k;
                        productBefore[indices_i_j_k] = value;
                    }
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IList{IList{IList{T}}}, IList{DistributionType}, IList{IList{IList{int}}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(IList<IList<IList<T>>> items, IList<DistributionType> array, IList<IList<IList<int>>> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return LogAverageFactor<DistributionType>(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IList{IList{IList{T}}}, IList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IList<IList<IList<T>>> items, IList<DistributionType> array)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogAverageFactor{ItemType, ItemType2, DistributionType}(IList{ItemType}, IList{T}, IList{IList{IList{int}}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<ItemType, ItemType2, DistributionType>(IList<ItemType> items, IList<T> array, IList<IList<IList<int>>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    var indices_i_j_Count = indices_i_j.Count;
                    for (int k = 0; k < indices_i_j_Count; k++)
                    {
                        var indices_i_j_k = indices_i_j[k];
                        z += items_i_j[k].GetLogProb(array[indices_i_j_k]);
                    }
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType, ItemType2,DistributionType}(IList{ItemType}, IList{T}, IList{IList{IList{int}}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<ItemType, ItemType2, DistributionType>(IList<ItemType> items, IList<T> array, IList<IList<IList<int>>> indices)
            where DistributionType : CanGetLogProb<T>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="AverageLogFactor{ItemType, ItemType2,DistributionType}(IList{ItemType}, IList{T})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<ItemType, ItemType2, DistributionType>(IList<ItemType> items, IList<T> array)
            where DistributionType : CanGetLogProb<T>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            return 0.0;
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType, ItemType2,DistributionType}(IList{ItemType}, IList{DistributionType}, IList{IList{IList{int}}}, IList{ItemType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<ItemType, ItemType2, DistributionType>(
            IList<ItemType> items, IList<DistributionType> array, IList<IList<IList<int>>> indices, IList<ItemType> to_items)
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count <= 1)
                return 0.0;
            var productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var to_items_i = to_items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    var to_items_i_j = to_items_i[j];
                    var indices_i_j_Count = indices_i_j.Count;
                    for (int k = 0; k < indices_i_j_Count; k++)
                    {
                        var indices_i_j_k = indices_i_j[k];
                        DistributionType value;
                        if (!productBefore.TryGetValue(indices_i_j_k, out value))
                        {
                            value = (DistributionType)array[indices_i_j_k].Clone();
                        }
                        var items_i_j_k = items_i_j[k];
                        z += value.GetLogAverageOf(items_i_j_k);
                        value.SetToProduct(value, items_i_j_k);
                        productBefore[indices_i_j_k] = value;
                        z -= to_items_i_j[k].GetLogAverageOf(items_i_j_k);
                    }
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="MarginalInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="Marginal{ArrayType,DistributionType}(ArrayType,ArrayType,ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [SkipIfAllUniform("array", "to_array")]
        [MultiplyAll]
        public static ArrayType Marginal<ArrayType, DistributionType>(
            ArrayType array, [NoInit] ArrayType to_array, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            for (int i = 0; i < result.Count; i++)
            {
                DistributionType value = result[i];
                value.SetToProduct(array[i], to_array[i]);
                result[i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, ItemType, ItemType2, DistributionType}(ArrayType, ItemType, ItemType, IList{IList{IList{int}}}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, ItemType, ItemType2, DistributionType>(
            ArrayType result, ItemType to_item, [SkipIfUniform] ItemType item, IList<IList<IList<int>>> indices, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                var indices_i_j = indices_i[j];
                var item_j = item[j];
                var to_item_j = to_item[j];
                var indices_i_j_Count = indices_i_j.Count;
                for (int k = 0; k < indices_i_j_Count; k++)
                {
                    var indices_i_j_k = indices_i_j[k];
                    DistributionType value = result[indices_i_j_k];
                    value.SetToProduct(to_item_j[k], item_j[k]);
                    result[indices_i_j_k] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, ItemType, ItemType2, DistributionType}(ItemType, ArrayType, ArrayType, IList{IList{IList{int}}}, int, ItemType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ArrayType, ItemType, ItemType2, DistributionType>(
            [Indexed, Cancels] ItemType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IList<IList<IList<int>>> indices,
            int resultIndex,
            ItemType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToRatio<DistributionType>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            Assert.IsTrue(result.Count == indices_i_Count, "result.Count != indices[i].Count");
            for (int j = 0; j < indices_i_Count; j++)
            {
                var indices_i_j = indices_i[j];
                var item_j = items[j];
                var indices_i_j_Count = indices_i_j.Count;
                for (int k = 0; k < indices_i_j_Count; k++)
                {
                    var indices_i_j_k = indices_i_j[k];
                    DistributionType value = result[j][k];
                    value.SetToRatio(marginal[indices_i_j_k], item_j[k]);
                    result[j][k] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IList{ItemType}, IList{IList{IList{int}}}, ArrayType)"]/*'/>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<ItemType, ItemType2, DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<IList<int>>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    var indices_i_j_Count = indices_i_j.Count;
                    for (int k = 0; k < indices_i_j_Count; k++)
                    {
                        var indices_i_j_k = indices_i_j[k];
                        DistributionType value = result[indices_i_j_k];
                        value.SetToProduct(value, items_i_j[k]);
                        result[indices_i_j_k] = value;
                    }
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IList{IList{IList{T}}}, IList{IList{IList{int}}}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<IList<IList<T>>> items, IList<IList<IList<int>>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var items_i_j = items_i[j];
                    var indices_i_j_Count = indices_i_j.Count;
                    for (int k = 0; k < indices_i_j_Count; k++)
                    {
                        var indices_i_j_k = indices_i_j[k];
                        DistributionType value = result[indices_i_j_k];
                        value.Point = items_i_j[k];
                        result[indices_i_j_k] = value;
                    }
                }
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="ItemsAverageLogarithm{ItemType, ItemType2, DistributionType}(IList{DistributionType}, IList{IList{IList{int}}}, int, ItemType)"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static ItemType ItemsAverageLogarithm<ItemType, ItemType2, DistributionType>(
            [SkipIfAllUniform] IList<DistributionType> array, IList<IList<IList<int>>> indices, int resultIndex, ItemType result)
            where DistributionType : SettableTo<DistributionType>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                var indices_i_j = indices_i[j];
                var indices_i_j_Count = indices_i_j.Count;
                for (int k = 0; k < indices_i_j_Count; k++)
                {
                    var indices_i_j_k = indices_i_j[k];
                    DistributionType value = result[j][k];
                    value.SetTo(array[indices_i_j_k]);
                    result[j][k] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="ArrayAverageLogarithm{ItemType, ItemType2, DistributionType, ArrayType}(IList{ItemType}, IList{IList{IList{int}}}, ArrayType)"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<ItemType, ItemType2, DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<IList<int>>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
            where ItemType2 : IList<DistributionType>
            where ItemType : IList<ItemType2>
        {
            return ArrayAverageConditional<ItemType, ItemType2, DistributionType, ArrayType>(items, indices, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IList{IList{IList{T}}}, IList{IList{IList{int}}}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(IList<IList<IList<T>>> items, IList<IList<IList<int>>> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }
    }
}
