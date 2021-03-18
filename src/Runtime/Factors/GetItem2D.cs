// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItem2D<>")]
    [Quality(QualityBand.Stable)]
    public static class GetItem2DOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="LogAverageFactor{Distribution}(Distribution, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(Distribution item, [Fresh] Distribution to_item)
            where Distribution : CanGetLogAverageOf<Distribution>
        {
            return to_item.GetLogAverageOf(item);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="LogAverageFactor{Distribution}(T, IArray2D{Distribution}, int, int)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(T item, IArray2D<Distribution> array, int index1, int index2)
            where Distribution : CanGetLogProb<T>
        {
            return array[index1, index2].GetLogProb(item);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="LogEvidenceRatio{Distribution}(Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<Distribution>(Distribution item)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="LogEvidenceRatio{Distribution}(T, IArray2D{Distribution}, int, int)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogEvidenceRatio<Distribution>(T item, IArray2D<Distribution> array, int index1, int index2)
            where Distribution : CanGetLogProb<T>
        {
            return LogAverageFactor(item, array, index1, index2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="ItemAverageConditional{Distribution}(IArray2D{Distribution}, int, int, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageConditional<Distribution>([SkipIfAllUniform] IArray2D<Distribution> array, int index1, int index2, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index1, index2]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="ArrayAverageConditional{Distribution, DistributionArray}(Distribution, int, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>(
            [SkipIfUniform] Distribution item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index1, index2];
            value.SetTo(item);
            result[index1, index2] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="ArrayAverageConditional{Distribution, DistributionArray}(T, int, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>(
            T item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index1, index2];
            value.Point = item;
            result[index1, index2] = value;
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="ItemAverageLogarithm{Distribution}(IArray2D{Distribution}, int, int, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageLogarithm<Distribution>(
            [SkipIfAllUniform] IArray2D<Distribution> array, int index1, int index2, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index1, index2]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="ArrayAverageLogarithm{Distribution, DistributionArray}(Distribution, int, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>(
            [SkipIfUniform] Distribution item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            Distribution value = result[index1, index2];
            value.SetTo(item);
            result[index1, index2] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItem2DOp{T}"]/message_doc[@name="ArrayAverageLogarithm{Distribution, DistributionArray}(T, int, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>(
            T item, int index1, int index2, DistributionArray result)
            where DistributionArray : IArray2D<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index1, index2];
            value.Point = item;
            result[index1, index2] = value;
            return result;
        }
    }
}
