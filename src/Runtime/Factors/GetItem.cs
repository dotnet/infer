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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItem<>")]
    [Quality(QualityBand.Mature)]
    public static class GetItemOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="LogAverageFactor(T, IList{T}, int)"]/*'/>
        public static double LogAverageFactor(T item, IList<T> array, int index)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            return equalityComparer.Equals(item, array[index]) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="LogEvidenceRatio(T, IList{T}, int)"]/*'/>
        public static double LogEvidenceRatio(T item, IList<T> array, int index)
        {
            return LogAverageFactor(item, array, index);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="AverageLogFactor(T, IList{T}, int)"]/*'/>
        public static double AverageLogFactor(T item, IList<T> array, int index)
        {
            return LogAverageFactor(item, array, index);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="LogAverageFactor{Distribution}(Distribution, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(Distribution item, [Fresh] Distribution to_item)
            where Distribution : CanGetLogAverageOf<Distribution>
        {
            return to_item.GetLogAverageOf(item);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="LogAverageFactor{Distribution}(T, IList{Distribution}, int)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogAverageFactor<Distribution>(T item, IList<Distribution> array, int index)
            where Distribution : CanGetLogProb<T>
        {
            return array[index].GetLogProb(item);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="LogEvidenceRatio{Distribution}(Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<Distribution>(Distribution item) where Distribution : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="LogEvidenceRatio{Distribution}(T, IList{Distribution}, int)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static double LogEvidenceRatio<Distribution>(T item, IList<Distribution> array, int index)
            where Distribution : CanGetLogProb<T>
        {
            return LogAverageFactor(item, array, index);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ItemAverageConditional{Distribution}(IList{Distribution}, int, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageConditional<Distribution>([SkipIfAllUniform] IList<Distribution> array, int index, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ItemAverageConditionalInit{Distribution}(IList{Distribution})"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static Distribution ItemAverageConditionalInit<Distribution>([IgnoreDependency] IList<Distribution> array)
            where Distribution : ICloneable
        {
            return (Distribution)array[0].Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ArrayAverageConditional{Distribution, DistributionArray}(Distribution, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>([SkipIfUniform] Distribution item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index];
            value.SetTo(item);
            result[index] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ArrayAverageConditional{Distribution, DistributionArray}(T, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageConditional<Distribution, DistributionArray>(T item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index];
            value.Point = item;
            result[index] = value;
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ItemAverageLogarithm{Distribution}(IList{Distribution}, int, Distribution)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        public static Distribution ItemAverageLogarithm<Distribution>([SkipIfAllUniform] IList<Distribution> array, int index, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(array[index]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ItemAverageLogarithmInit{Distribution}(IList{Distribution})"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        [Skip]
        public static Distribution ItemAverageLogarithmInit<Distribution>([IgnoreDependency] IList<Distribution> array)
            where Distribution : ICloneable
        {
            return (Distribution)array[0].Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ArrayAverageLogarithm{Distribution, DistributionArray}(Distribution, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>([SkipIfUniform] Distribution item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : SettableTo<Distribution>
        {
            Distribution value = result[index];
            value.SetTo(item);
            result[index] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemOp{T}"]/message_doc[@name="ArrayAverageLogarithm{Distribution, DistributionArray}(T, int, DistributionArray)"]/*'/>
        /// <typeparam name="Distribution">The type of the distribution over an item.</typeparam>
        /// <typeparam name="DistributionArray">The type of the outgoing message.</typeparam>
        public static DistributionArray ArrayAverageLogarithm<Distribution, DistributionArray>(T item, int index, DistributionArray result)
            where DistributionArray : IList<Distribution>
            where Distribution : HasPoint<T>
        {
            // assume result is initialized to uniform.
            Distribution value = result[index];
            value.Point = item;
            result[index] = value;
            return result;
        }
    }
}
