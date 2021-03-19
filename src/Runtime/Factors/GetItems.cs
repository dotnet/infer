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

    ///<include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItems<>", Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("marginal")]
    public static class GetItemsOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogAverageFactor(IReadOnlyList{T}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        public static double LogAverageFactor(IReadOnlyList<T> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            for (int i = 0; i < items.Count; i++)
            {
                if (!equalityComparer.Equals(items[i], array[indices[i]])) return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogEvidenceRatio(IReadOnlyList{T}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        public static double LogEvidenceRatio(IReadOnlyList<T> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="AverageLogFactor(IReadOnlyList{T}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        public static double AverageLogFactor(IReadOnlyList<T> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{DistributionType}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value;
                if (!productBefore.TryGetValue(indices[i], out value))
                {
                    value = (DistributionType) array[indices[i]].Clone();
                }
                z += value.GetLogAverageOf(items[i]);
                value.SetToProduct(value, items[i]);
                productBefore[indices[i]] = value;
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<DistributionType> array)
            where DistributionType : SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IReadOnlyList{T}, IReadOnlyList{DistributionType}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<T> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            double z = 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value;
                if (!productBefore.TryGetValue(indices[i], out value))
                {
                    value = array[indices[i]];
                }
                z += value.GetLogProb(items[i]);
                value.Point = items[i];
                productBefore[indices[i]] = value;
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IReadOnlyList{T}, IReadOnlyList{DistributionType}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(IReadOnlyList<T> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return LogAverageFactor<DistributionType>(items, array, indices);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IReadOnlyList{T}, IReadOnlyList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IReadOnlyList<T> items, IReadOnlyList<DistributionType> array)
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
                z += items[i].GetLogProb(array[indices[i]]);
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<T> array, IReadOnlyList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{T})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IReadOnlyList<DistributionType> items, IReadOnlyList<T> array)
            where DistributionType : CanGetLogProb<T>
        {
            return 0.0;
        }


#if true
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(
            IReadOnlyList<DistributionType> items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices, IReadOnlyList<DistributionType> to_items)
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count <= 1) return 0.0;
            Dictionary<int, DistributionType> productBefore = new Dictionary<int, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value;
                if (!productBefore.TryGetValue(indices[i], out value))
                {
                    value = (DistributionType) array[indices[i]].Clone();
                }
                z += value.GetLogAverageOf(items[i]);
                value.SetToProduct(value, items[i]);
                productBefore[indices[i]] = value;
                z -= to_items[i].GetLogAverageOf(items[i]);
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
        public static double LogEvidenceRatio<DistributionType, ArrayType>(ArrayType items, IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices)
            where ArrayType : IReadOnlyList<DistributionType>, ICloneable
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // result is LogAverageFactor - sum_i toItems[i].LogAverageOf(items[i])
            // we compute this efficiently by constructing toItems[i] by dynamic programming.
            // toItems[i] = productBefore[i] * productAfter[i].
            double z = 0.0;
            if (items.Count <= 1) return 0.0;
            DistributionType uniform = (DistributionType)items[0].Clone();
            uniform.SetToUniform();
            ArrayType productBefore = (ArrayType)items.Clone();
            ArrayType productAfter = (ArrayType)items.Clone();
            Dictionary<int,int> indexToItem = new Dictionary<int, int>();
            for (int i = 0; i < indices.Count; i++) {
                int previousItem;
                if (!indexToItem.TryGetValue(indices[i], out previousItem)) {
                    // no previous item with this index
                    productBefore[i] = array[indices[i]];
                } else {
                    DistributionType temp = productBefore[i];
                    temp.SetToProduct(productBefore[previousItem], items[previousItem]);
                    productBefore[i] = temp;
                }
                z += productBefore[i].GetLogAverageOf(items[i]);
                indexToItem[indices[i]] = i;
            }
            indexToItem.Clear();
            for (int i = indices.Count - 1; i >= 0; i--) {
                int itemAfter;
                if (!indexToItem.TryGetValue(indices[i], out itemAfter)) {
                    // no item after with this index
                    productAfter[i] = uniform;
                } else {
                    DistributionType temp = productAfter[i];
                    temp.SetToProduct(productAfter[itemAfter], items[itemAfter]);
                    productAfter[i] = temp;
                }
                DistributionType toItem = (DistributionType)items[i].Clone();
                toItem.SetToProduct(productBefore[i], productAfter[i]);
                z -= toItem.GetLogAverageOf(items[i]);
                indexToItem[indices[i]] = i;
            }
            return z;
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="MarginalInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="Marginal2{ArrayType, DistributionType}(ArrayType, IReadOnlyList{DistributionType}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [SkipIfAllUniform("array", "items")]
        public static ArrayType Marginal2<ArrayType, DistributionType>(
            ArrayType array, [NoInit] IReadOnlyList<DistributionType> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetTo(array);
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="Marginal{ArrayType, DistributionType}(ArrayType, ArrayType, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [SkipIfAllUniform("array", "to_array")]
        [MultiplyAll]
        public static ArrayType Marginal<ArrayType, DistributionType>(
            IReadOnlyList<DistributionType> array, [NoInit] IReadOnlyList<DistributionType> to_array, ArrayType result)
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, DistributionType}(ArrayType, DistributionType, DistributionType, IReadOnlyList{int}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, DistributionType>(
            ArrayType result, DistributionType to_item, [SkipIfUniform] DistributionType item, IReadOnlyList<int> indices, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            int i = resultIndex;
            DistributionType value = result[indices[i]];
            value.SetToProduct(to_item, item);
            result[indices[i]] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, DistributionType}(DistributionType, ArrayType, ArrayType, IReadOnlyList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<ArrayType, DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IReadOnlyList<int> indices,
            int resultIndex, 
            DistributionType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(marginal[indices[i]], items);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IReadOnlyList<DistributionType> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                DistributionType value = result[indices_i];
                value.SetToProduct(value, items[i]);
                result[indices_i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ArrayAverageConditional{DistributionType, ArrayType}(IReadOnlyList{T}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(
            [SkipIfAllUniform] IReadOnlyList<T> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                var indices_i = indices[i];
                DistributionType value = result[indices_i];
                value.Point = items[i];
                result[indices_i] = value;
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ItemsAverageLogarithm{DistributionType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageLogarithm<DistributionType>(
            [SkipIfAllUniform] IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices, int resultIndex, DistributionType result)
            where DistributionType : SettableTo<DistributionType>
        {
            int i = resultIndex;
            result.SetTo(array[indices[i]]);
            return result;
        }

        public static ResultType ItemsAverageLogarithm2<DistributionType, ResultType>(
            [SkipIfAllUniform] IReadOnlyList<DistributionType> array, IReadOnlyList<int> indices, ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[i];
                value.SetTo(array[indices[i]]);
                result[i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ItemsAverageLogarithmInit{TDist}(DistributionStructArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageLogarithmInit<TDist>(
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
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ItemsAverageLogarithmInit{TDist}(DistributionRefArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageLogarithmInit<TDist>(
            [IgnoreDependency] DistributionRefArray<TDist, T> array, IReadOnlyList<int> indices)
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
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IReadOnlyList{DistributionType}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(
            [SkipIfAllUniform] IReadOnlyList<DistributionType> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            return ArrayAverageConditional(items, indices, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IReadOnlyList{T}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(IReadOnlyList<T> items, IReadOnlyList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }
    }

    ///<include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/doc/*'/>
    /// <typeparam name="T">The type of a list element.</typeparam>
    [FactorMethod(typeof(Collection), "GetItems<>", Default = false)]
    [Buffers("partial")]
    [Quality(QualityBand.Mature)]
    public static class GetItemsOp2<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/message_doc[@name="ItemsAverageConditionalInit{TDist}(DistributionStructArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
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
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/message_doc[@name="ItemsAverageConditionalInit{TDist}(DistributionRefArray{TDist, T}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageConditionalInit<TDist>(
            [IgnoreDependency] DistributionRefArray<TDist, T> array, IReadOnlyList<int> indices)
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
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist) array[indices[i]].Clone());
        }

        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional2<DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [SkipIfAllUniform] IReadOnlyList<DistributionType> array,
            [Fresh] IReadOnlyList<DistributionType> to_array,
            IReadOnlyList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToProduct(to_array[indices[i]], array[indices[i]]);
            result.SetToRatio(result, items);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/message_doc[@name="ItemsAverageConditional{DistributionType}(DistributionType, IReadOnlyList{DistributionType}, IReadOnlyList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<DistributionType>(
            [Indexed] DistributionType partial,
            [SkipIfAllUniform] IReadOnlyList<DistributionType> array,
            IReadOnlyList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToProduct(partial, array[indices[i]]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/message_doc[@name="ArrayIncrement{DistributionType}(DistributionType, DistributionType, DistributionType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType ArrayIncrement<DistributionType>(DistributionType partial,
            [SkipIfUniform] DistributionType item,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>
        {
            result.SetToProduct(partial, item);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/message_doc[@name="PartialInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of a distribution array.</typeparam>
        [Skip]
        public static ArrayType PartialInit<ArrayType>(ArrayType items)
            where ArrayType : ICloneable, SettableToUniform
        {
            ArrayType result = (ArrayType)items.Clone();
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsOp2{T}"]/message_doc[@name="Partial{DistributionType}(DistributionType, IReadOnlyList{DistributionType}, IReadOnlyList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static DistributionType Partial<DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [Fresh] IReadOnlyList<DistributionType> to_array,
            IReadOnlyList<int> indices,
            int resultIndex,
            DistributionType result)
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(to_array[indices[i]], items);
            return result;
        }    
    }

    //[FactorMethod(typeof(Factor), "GetItems<>", Default=false)]
    [Buffers("marginal")]
    [Quality(QualityBand.Mature)]
    public static class GetItemsBufferOp2<T>
    {
        public static ArrayType MarginalInit<ArrayType, DistributionType>(ArrayType array, IReadOnlyList<DistributionType> items, IReadOnlyList<int> indices)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>, ICloneable
            where DistributionType : SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            ArrayType result = (ArrayType) array.Clone();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        public static ArrayType Marginal<ArrayType, DistributionType>(ArrayType marginal, DistributionType to_item, DistributionType item, IReadOnlyList<int> indices, int resultIndex)
            where ArrayType : IList<DistributionType>, SettableTo<ArrayType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            ArrayType result = marginal;
            int i = resultIndex;
            DistributionType value = result[indices[i]];
            value.SetToProduct(to_item, item);
            result[indices[i]] = value;
            return result;
        }

        public static DistributionType ItemsAverageConditional<ArrayType, DistributionType>([MatchingIndex] IReadOnlyList<DistributionType> items, [IgnoreDependency] IReadOnlyList<DistributionType> array,
                                                                                            [SkipIfAllUniform] ArrayType marginal, IReadOnlyList<int> indices, int resultIndex,
                                                                                            DistributionType result)
            where ArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(marginal[indices[i]], items[i]);
            return result;
        }
    }
}
