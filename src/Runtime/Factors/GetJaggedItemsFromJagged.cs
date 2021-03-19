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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetJaggedItemsFromJagged<>", Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("marginal")]
    public static class GetJaggedItemsFromJaggedOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor(IList{IList{T}}, IList{IList{T}}, IList{IList{int}})"]/*'/>
        public static double LogAverageFactor(IList<IList<T>> items, IList<IList<T>> array, IList<IList<int>> indices, IList<IList<int>> indices2)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    if (!equalityComparer.Equals(items_i[j], array[indices_i[j]][indices2_i[j]]))
                        return Double.NegativeInfinity;
                }
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio(IList{IList{T}}, IList{IList{T}}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        public static double LogEvidenceRatio(IList<IList<T>> items, IList<IList<T>> array, IList<IList<int>> indices, IList<IList<int>> indices2)
        {
            return LogAverageFactor(items, array, indices, indices2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor(IList{IList{T}}, IList{IList{T}}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        public static double AverageLogFactor(IList<IList<T>> items, IList<IList<T>> array, IList<IList<int>> indices, IList<IList<int>> indices2)
        {
            return LogAverageFactor(items, array, indices, indices2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor{ItemType,DistributionType}(IList{ItemType}, IList{ItemType}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<ItemType, DistributionType>(IList<ItemType> items, IList<ItemType> array, IList<IList<int>> indices, IList<IList<int>> indices2)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<DistributionType>
        {
            double z = 0.0;
            var productBefore = new Dictionary<Tuple<int,int>, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var indices2_i_j = indices2_i[j];
                    var key = Tuple.Create(indices_i_j, indices2_i_j);
                    DistributionType value;
                    if (!productBefore.TryGetValue(key, out value))
                    {
                        value = (DistributionType)array[indices_i_j][indices2_i_j].Clone();
                    }
                    var items_i_j = items_i[j];
                    z += value.GetLogAverageOf(items_i_j);
                    value.SetToProduct(value, items_i_j);
                    productBefore[key] = value;
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor{ItemType,DistributionType}(IList{ItemType}, IList{ItemType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<ItemType, DistributionType>(IList<ItemType> items, IList<ItemType> array)
            where DistributionType : SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IList{IList{T}}, IList{ItemType}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<ItemType, DistributionType>(IList<IList<T>> items, IList<ItemType> array, IList<IList<int>> indices, IList<IList<int>> indices2)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            double z = 0.0;
            var productBefore = new Dictionary<Tuple<int,int>, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var indices2_i_j = indices2_i[j];
                    var key = Tuple.Create(indices_i_j, indices2_i_j);
                    DistributionType value;
                    if (!productBefore.TryGetValue(key, out value))
                    {
                        value = array[indices_i_j][indices2_i_j];
                    }
                    var items_i_j = items_i[j];
                    z += value.GetLogProb(items_i_j);
                    value.Point = items_i_j;
                    productBefore[key] = value;
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType,DistributionType}(IList{IList{T}}, IList{ItemType}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<ItemType, DistributionType>(IList<IList<T>> items, IList<ItemType> array, IList<IList<int>> indices, IList<IList<int>> indices2)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return LogAverageFactor<ItemType, DistributionType>(items, array, indices, indices2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor{ItemType,DistributionType}(IList{IList{T}}, IList{ItemType})"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<ItemType, DistributionType>(IList<IList<T>> items, IList<ItemType> array)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor{ItemType,DistributionType}(IList{ItemType}, IList{IList{T}}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogAverageFactor<ItemType, DistributionType>(IList<ItemType> items, IList<IList<T>> array, IList<IList<int>> indices, IList<IList<int>> indices2)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    z += items_i[j].GetLogProb(array[indices_i[j]][indices2_i[j]]);
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType,DistributionType}(IList{ItemType}, IList{IList{T}}, IList{IList{int}}, IList{IList{int}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<ItemType, DistributionType>(IList<ItemType> items, IList<IList<T>> array, IList<IList<int>> indices, IList<IList<int>> indices2)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor{ItemType,DistributionType}(IList{ItemType}, IList{IList{T}})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        [Skip]
        public static double AverageLogFactor<ItemType, DistributionType>(IList<ItemType> items, IList<IList<T>> array)
            where DistributionType : CanGetLogProb<T>
            where ItemType : IList<DistributionType>
        {
            return 0.0;
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio{ItemType,DistributionType}(IList{ItemType}, IList{ItemType}, IList{IList{int}}, IList{IList{int}}, IList{ItemType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static double LogEvidenceRatio<ItemType, DistributionType>(
            IList<ItemType> items, IList<ItemType> array, IList<IList<int>> indices, IList<IList<int>> indices2, IList<ItemType> to_items)
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
            where ItemType : IList<DistributionType>
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count <= 1)
                return 0.0;
            var productBefore = new Dictionary<Tuple<int,int>, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var to_items_i = to_items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var indices2_i_j = indices2_i[j];
                    var key = Tuple.Create(indices_i_j, indices2_i_j);
                    DistributionType value;
                    if (!productBefore.TryGetValue(key, out value))
                    {
                        value = (DistributionType)array[indices_i_j][indices2_i_j].Clone();
                    }
                    var items_i_j = items_i[j];
                    z += value.GetLogAverageOf(items_i_j);
                    value.SetToProduct(value, items_i_j);
                    productBefore[key] = value;
                    z -= to_items_i[j].GetLogAverageOf(items_i_j);
                }
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="MarginalInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="Marginal{ArrayType,DistributionType}(ArrayType,ArrayType,ArrayType)"]/*'/>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, ItemType, DistributionType}(ArrayType, ItemType, ItemType, IList{IList{int}}, IList{IList{int}}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, ItemType, DistributionType>(
            ArrayType result, ItemType to_item, [SkipIfUniform] ItemType item, IList<IList<int>> indices, IList<IList<int>> indices2, int resultIndex)
            where ArrayType : IList<ItemType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices2_i = indices2[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                var indices_i_j = indices_i[j];
                var indices2_i_j = indices2_i[j];
                DistributionType value = result[indices_i_j][indices2_i_j];
                value.SetToProduct(to_item[j], item[j]);
                result[indices_i_j][indices2_i_j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, ItemType, DistributionType}(ItemType, ArrayType, ArrayType, IList{IList{int}}, IList{IList{int}}, int, ItemType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ArrayType, ItemType, DistributionType>(
            [Indexed, Cancels] ItemType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IList<IList<int>> indices, 
            IList<IList<int>> indices2,
            int resultIndex,
            ItemType result)
            where ArrayType : IList<ItemType>
            where DistributionType : SettableToRatio<DistributionType>
            where ItemType : IList<DistributionType>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices2_i = indices2[i];
            var indices_i_Count = indices_i.Count;
            Assert.IsTrue(result.Count == indices_i_Count, "result.Count != indices[i].Count");
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetToRatio(marginal[indices_i[j]][indices2_i[j]], items[j]);
                result[j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageConditional{ItemType, DistributionType, ArrayType}(IList{ItemType}, IList{IList{int}}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<ItemType, DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<int>> indices, IList<IList<int>> indices2, ArrayType result)
            where ArrayType : IList<ItemType>, SettableToUniform
            where DistributionType : SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var indices2_i_j = indices2_i[j];
                    DistributionType value = result[indices_i_j][indices2_i_j];
                    value.SetToProduct(value, items_i[j]);
                    result[indices_i_j][indices2_i_j] = value;
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageConditional{ItemType, DistributionType, ArrayType}(IList{IList{T}}, IList{IList{int}}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static ArrayType ArrayAverageConditional<ItemType, DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<IList<T>> items, IList<IList<int>> indices, IList<IList<int>> indices2, ArrayType result)
            where ArrayType : IList<ItemType>, SettableToUniform
            where DistributionType : HasPoint<T>
            where ItemType : IList<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                var indices2_i = indices2[i];
                var items_i = items[i];
                var indices_i_Count = indices_i.Count;
                for (int j = 0; j < indices_i_Count; j++)
                {
                    var indices_i_j = indices_i[j];
                    var indices2_i_j = indices2_i[j];
                    DistributionType value = result[indices_i_j][indices2_i_j];
                    value.Point = items_i[j];
                    result[indices_i_j][indices2_i_j] = value;
                }
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="ItemsAverageLogarithm{ItemType, DistributionType}(IList{ItemType}, IList{IList{int}}, IList{IList{int}}, int, ItemType)"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static ItemType ItemsAverageLogarithm<ItemType, DistributionType>(
            [SkipIfAllUniform] IList<ItemType> array, IList<IList<int>> indices, IList<IList<int>> indices2, int resultIndex, ItemType result)
            where DistributionType : SettableTo<DistributionType>
            where ItemType : IList<DistributionType>
        {
            int i = resultIndex;
            var indices_i = indices[i];
            var indices2_i = indices2[i];
            var indices_i_Count = indices_i.Count;
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetTo(array[indices_i[j]][indices2_i[j]]);
                result[j] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageLogarithm{ItemType, DistributionType, ArrayType}(IList{ItemType}, IList{IList{int}},IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<ItemType, DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<ItemType> items, IList<IList<int>> indices, IList<IList<int>> indices2, ArrayType result)
            where ArrayType : IList<ItemType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
            where ItemType : IList<DistributionType>
        {
            return ArrayAverageConditional<ItemType, DistributionType, ArrayType>(items, indices, indices2, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageLogarithm{ItemType, DistributionType, ArrayType}(IList{IList{T}}, IList{IList{int}}, IList{IList{int}}, ArrayType)"]/*'/>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<ItemType, DistributionType, ArrayType>(IList<IList<T>> items, IList<IList<int>> indices, IList<IList<int>> indices2, ArrayType result)
            where ArrayType : IList<ItemType>, SettableToUniform
            where DistributionType : HasPoint<T>
            where ItemType : IList<DistributionType>
        {
            return ArrayAverageConditional<ItemType, DistributionType, ArrayType>(items, indices, indices2, result);
        }
    }
}
