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
    [FactorMethod(typeof(Collection), "Subarray<>")]
    [Quality(QualityBand.Mature)]
    public static class SubarrayOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogAverageFactor(IReadOnlyList{T}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        public static double LogAverageFactor(IReadOnlyList<T> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            return GetItemsOp<T>.LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio(IReadOnlyList{T}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        public static double LogEvidenceRatio(IReadOnlyList<T> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IReadOnlyList{T}, IReadOnlyList{DistributionType}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<T> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            AssertWhenDebugging.Distinct(indices);
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                z += array[indices[i]].GetLogProb(items[i]);
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{DistributionType}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where DistributionType : CanGetLogAverageOf<DistributionType>
        {
            AssertWhenDebugging.Distinct(indices);
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                z += array[indices[i]].GetLogAverageOf(items[i]);
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            return GetItemsOp<T>.LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IReadOnlyList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType>(IReadOnlyList<DistributionType> items) where DistributionType : CanGetLogAverageOf<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IReadOnlyList{T}, IReadOnlyList{DistributionType}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(IReadOnlyList<T> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ItemsAverageConditional{DistributionType, ResultType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, ResultType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ResultType">The type of the outgoing message.</typeparam>
        public static ResultType ItemsAverageConditional<DistributionType, ResultType>([SkipIfAllUniform] IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices, ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(result.Count == indices.Count, "result.Count != indices.Count");
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[i];
                value.SetTo(array[indices[i]]);
                result[i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ItemsAverageConditionalInit{TDist}(DistributionStructArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageConditionalInit<TDist>(
            [IgnoreDependency] DistributionStructArray<TDist, T> array, IReadOnlyList<int> indices)
            where TDist : struct,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ItemsAverageConditionalInit{TDist}(DistributionRefArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageConditionalInit<TDist>([IgnoreDependency] DistributionRefArray<TDist, T> array, IReadOnlyList<int> indices)
            where TDist : class,
                SettableTo<TDist>,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>([SkipIfAllUniform] IReadOnlyList<DistributionType> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            AssertWhenDebugging.Distinct(indices);
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetTo(items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IReadOnlyList{T}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(IReadOnlyList<T> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException(indices.Count + " indices were given to Subarray but the output array has length " + items.Count);
            AssertWhenDebugging.Distinct(indices);
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.Point = items[i];
                result[indices[i]] = value;
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ItemsAverageLogarithmInit{TDist}(DistributionStructArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageLogarithmInit<TDist>([IgnoreDependency] DistributionStructArray<TDist, T> array, IReadOnlyList<int> indices)
            where TDist : struct,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ItemsAverageLogarithmInit{TDist}(DistributionRefArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageLogarithmInit<TDist>([IgnoreDependency] DistributionRefArray<TDist, T> array, IReadOnlyList<int> indices)
            where TDist : class,
                SettableTo<TDist>,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ItemsAverageLogarithm{DistributionType, ResultType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, ResultType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ResultType">The type of the outgoing message.</typeparam>
        public static ResultType ItemsAverageLogarithm<DistributionType, ResultType>([SkipIfAllUniform] IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices, ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            return ItemsAverageConditional<DistributionType, ResultType>(array, indices, result);
        }

        [Skip]
        public static ResultType ItemsDeriv<ResultType>(ResultType result)
            where ResultType : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>([SkipIfAllUniform] IReadOnlyList<DistributionType> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubarrayOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IReadOnlyList{T}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(IReadOnlyList<T> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }
    }
}
