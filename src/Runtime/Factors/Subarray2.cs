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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an array item.</typeparam>
    [FactorMethod(typeof(Collection), "Subarray2<>")]
    [Quality(QualityBand.Experimental)]
    public static class SubarrayOp2<T>
    {
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ResultType">The type of the outgoing message.</typeparam>
        //[SkipIfAllUniform("array", "array2")]
        public static ResultType ItemsAverageConditional<DistributionType, ResultType>(
            [SkipIfAllUniform] IList<DistributionType> array, 
            IList<int> indices,
            [SkipIfAllUniform] IList<DistributionType> array2,
            ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(result.Count == indices.Count, "result.Count != indices.Count");
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[i];
                if (indices[i] < 0)
                    value.SetTo(array2[i]);
                else
                    value.SetTo(array[indices[i]]);
                result[i] = value;
            }
            return result;
        }

        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType Array2AverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items,
            IList<int> indices,
            ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                if (indices[i] >= 0)
                    continue;
                DistributionType value = result[i];
                value.SetTo(items[i]);
                result[i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IList{DistributionType}, IList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items, 
            IList<int> indices, 
            ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                if (indices[i] < 0)
                    continue;
                DistributionType value = result[indices[i]];
                value.SetTo(items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IList{T}, IList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            IList<T> items, 
            IList<int> indices, 
            ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException(indices.Count + " indices were given to Subarray but the output array has length " + items.Count);
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                if (indices[i] < 0)
                    continue;
                DistributionType value = result[indices[i]];
                value.Point = items[i];
                result[indices[i]] = value;
            }
            return result;
        }
    }
}
