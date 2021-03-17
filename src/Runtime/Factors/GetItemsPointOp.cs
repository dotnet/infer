// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System.Collections.Generic;
    using Math;
    using Utilities;
    using Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItems<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class GetItemsPointOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsPointOp{T}"]/message_doc[@name="ItemsAverageConditional{DistributionType}(IList{DistributionType}, IList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<DistributionType>(
            [SkipIfAllUniform] IList<DistributionType> array,
            IList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableTo<DistributionType>
        {
            // array will always be a point mass, so no division is needed.
            int i = resultIndex;
            result.SetTo(array[indices[i]]);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItemsFromJagged<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class GetItemsFromJaggedPointOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedPointOp{T}"]/message_doc[@name="ItemsAverageConditional{DistributionArrayType, DistributionType}(IList{DistributionArrayType}, IList{int}, IList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<DistributionArrayType, DistributionType>(
            [SkipIfAllUniform] IList<DistributionArrayType> array,
            IList<int> indices,
            IList<int> indices2,
            int resultIndex,
            DistributionType result)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            // array will always be a point mass, so no division is needed.
            int i = resultIndex;
            result.SetTo(array[indices[i]][indices2[i]]);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromDeepJaggedOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItemsFromDeepJagged<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class GetItemsFromDeepJaggedPointOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromDeepJaggedPointOp{T}"]/message_doc[@name="ItemsAverageConditional{DistributionArrayArrayType, DistributionArrayType, DistributionType}(IList{DistributionArrayArrayType}, IList{int}, IList{int}, IList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionArrayArrayType">The type of a distribution over depth 1 array elements.</typeparam>
        /// <typeparam name="DistributionArrayType">The type of a distribution over depth 2 array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over depth 3 array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<DistributionArrayArrayType, DistributionArrayType, DistributionType>(
            [SkipIfAllUniform] IList<DistributionArrayArrayType> array,
            IList<int> indices,
            IList<int> indices2,
            IList<int> indices3,
            int resultIndex,
            DistributionType result)
            where DistributionArrayArrayType : IList<DistributionArrayType>
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            // array will always be a point mass, so no division is needed.
            int i = resultIndex;
            result.SetTo(array[indices[i]][indices2[i]][indices3[i]]);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsPointOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetJaggedItems<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class GetJaggedItemsPointOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsPointOp{T}"]/message_doc[@name="ItemsAverageConditional{ItemType, DistributionType}(IList{DistributionType}, IList{IList{int}}, int, ItemType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ItemType, DistributionType>(
            [SkipIfAllUniform] IList<DistributionType> array,
            IList<IList<int>> indices,
            int resultIndex,
            ItemType result)
            where DistributionType : SettableTo<DistributionType>
            where ItemType : IList<DistributionType>
        {
            // array will always be a point mass, so no division is needed.
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            Assert.IsTrue(result.Count == indices_i_Count, "result.Count != indices[i].Count");
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetTo(array[indices_i[j]]);
                result[j] = value;
            }
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsPointOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetDeepJaggedItems<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class GetDeepJaggedItemsPointOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetDeepJaggedItemsPointOp{T}"]/message_doc[@name="ItemsAverageConditional{ItemType, ItemType2, DistributionType}(IList{DistributionType}, IList{IList{IList{int}}}, int, ItemType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        /// <typeparam name="ItemType2">The type of a sub-sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ItemType, ItemType2, DistributionType>(
            [SkipIfAllUniform] IList<DistributionType> array,
            IList<IList<IList<int>>> indices,
            int resultIndex,
            ItemType result)
            where DistributionType : SettableTo<DistributionType>
            where ItemType : IList<ItemType2>
            where ItemType2 : IList<DistributionType>
        {
            // array will always be a point mass, so no division is needed.
            int i = resultIndex;
            var indices_i = indices[i];
            var indices_i_Count = indices_i.Count;
            Assert.IsTrue(result.Count == indices_i_Count, "result.Count != indices[i].Count");
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
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedPointOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetJaggedItemsFromJagged<>", Default = false)]
    [Quality(QualityBand.Experimental)]
    public static class GetJaggedItemsFromJaggedPointOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetJaggedItemsFromJaggedPointOp{T}"]/message_doc[@name="ItemsAverageConditional{ItemType, DistributionType}(IList{ItemType}, IList{IList{int}}, IList{IList{int}}, int, ItemType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ItemType">The type of a sub-array.</typeparam>
        public static ItemType ItemsAverageConditional<ItemType, DistributionType>(
            [SkipIfAllUniform] IList<ItemType> array,
            IList<IList<int>> indices,
            IList<IList<int>> indices2,
            int resultIndex,
            ItemType result)
            where DistributionType : SettableTo<DistributionType>
            where ItemType : IList<DistributionType>
        {
            // array will always be a point mass, so no division is needed.
            int i = resultIndex;
            var indices_i = indices[i];
            var indices2_i = indices2[i];
            var indices_i_Count = indices_i.Count;
            Assert.IsTrue(result.Count == indices_i_Count, "result.Count != indices[i].Count");
            for (int j = 0; j < indices_i_Count; j++)
            {
                DistributionType value = result[j];
                value.SetTo(array[indices_i[j]][indices2_i[j]]);
                result[j] = value;
            }
            return result;
        }
    }
}
