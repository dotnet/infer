// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    [Hidden]
    internal static class ExperimentalFactor
    {
        [ParameterNames("items", "array", "indices", "dict")]
        public static T[] GetItemsWithDictionary<T>(IList<T> array, IList<string> indices, IDictionary<string, int> dict)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[dict[indices[i]]];
            }
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(ExperimentalFactor), "GetItemsWithDictionary<>")]
    [Quality(QualityBand.Experimental)]
    [Buffers("marginal")]
    internal static class GetItemsWithDictionaryOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/message_doc[@name="MarginalInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/message_doc[@name="Marginal{ArrayType, DistributionType}(ArrayType, IList{DistributionType}, IList{String}, IDictionary{String, int}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        [SkipIfAllUniform("array", "items")]
        public static ArrayType Marginal<ArrayType, DistributionType>(
            ArrayType array, [NoInit] IList<DistributionType> items, IList<string> indices, IDictionary<string, int> dict, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
            {
                int index = dict[indices[i]];
                DistributionType value = result[index];
                value.SetToProduct(value, items[i]);
                result[index] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, DistributionType}(ArrayType, DistributionType, DistributionType, IList{String}, IDictionary{String, int}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, DistributionType>(
            ArrayType result, DistributionType to_item, [SkipIfUniform] DistributionType item, IList<string> indices, IDictionary<string, int> dict, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            int i = resultIndex;
            int index = dict[indices[i]];
            DistributionType value = result[index];
            value.SetToProduct(to_item, item);
            result[index] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, DistributionType}(DistributionType, ArrayType, ArrayType, IList{String}, IDictionary{String, int}, int, DistributionType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        public static DistributionType ItemsAverageConditional<ArrayType, DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IList<string> indices,
            IDictionary<string, int> dict,
            int resultIndex,
            DistributionType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            int index = dict[indices[i]];
            result.SetToRatio(marginal[index], items);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IList{DistributionType}, IList{String}, IDictionary{String, int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items, IList<string> indices, IDictionary<string, int> dict, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                int index = dict[indices[i]];
                DistributionType value = result[index];
                value.SetToProduct(value, items[i]);
                result[index] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsWithDictionaryOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IList{T}, IList{String}, IDictionary{String, int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an item.</typeparam>
        /// <typeparam name="ArrayType">The type of an array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<T> items, IList<string> indices, IDictionary<string, int> dict, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                int index = dict[indices[i]];
                DistributionType value = result[index];
                value.Point = items[i];
                result[index] = value;
            }
            return result;
        }
    }
}
