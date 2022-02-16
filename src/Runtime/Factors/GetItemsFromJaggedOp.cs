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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of an item.</typeparam>
    [FactorMethod(typeof(Collection), "GetItemsFromJagged<>", Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("marginal")]
    public static class GetItemsFromJaggedOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor(IList{T}, IList{T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        public static double LogAverageFactor(IList<T> items, IList<IList<T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            for (int i = 0; i < items.Count; i++)
            {
                if (!equalityComparer.Equals(items[i], array[indices[i]][indices2[i]]))
                    return Double.NegativeInfinity;
            }
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio(IList{T}, IList{T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        public static double LogEvidenceRatio(IList<T> items, IList<IList<T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
        {
            return LogAverageFactor(items, array, indices, indices2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor(IList{T}, IList{T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        public static double AverageLogFactor(IList<T> items, IList<IList<T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
        {
            return LogAverageFactor(items, array, indices, indices2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IList{DistributionType}, IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static double LogAverageFactor<DistributionArrayType, DistributionType>(IList<DistributionType> items, IList<DistributionArrayType> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : IDistribution<T>, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            double z = 0.0;
            var productBefore = new Dictionary<Tuple<int,int>, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var key = Tuple.Create(indices[i], indices2[i]);
                DistributionType value;
                if (!productBefore.TryGetValue(key, out value))
                {
                    value = (DistributionType)array[indices[i]][indices2[i]].Clone();
                }
                z += value.GetLogAverageOf(items[i]);
                value.SetToProduct(value, items[i]);
                productBefore[key] = value;
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IList{DistributionType}, IList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionArrayType, DistributionType>(IList<DistributionType> items, IList<DistributionArrayType> array)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IList{T}, IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static double LogAverageFactor<DistributionArrayType, DistributionType>(IList<T> items, IList<DistributionArrayType> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            double z = 0.0;
            var productBefore = new Dictionary<Tuple<int, int>, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var key = Tuple.Create(indices[i], indices2[i]);
                DistributionType value;
                if (!productBefore.TryGetValue(key, out value))
                {
                    value = array[indices[i]][indices2[i]];
                }
                z += value.GetLogProb(items[i]);
                value.Point = items[i];
                productBefore[key] = value;
            }
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IList{T}, IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionArrayType, DistributionType>(IList<T> items, IList<DistributionArrayType> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return LogAverageFactor<DistributionArrayType, DistributionType>(items, array, indices, indices2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IList{T}, IList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionArrayType, DistributionType>(IList<T> items, IList<DistributionArrayType> array)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : HasPoint<T>, CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogAverageFactor{DistributionType}(IList{DistributionType}, IList{T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<DistributionType> items, IList<IList<T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
            where DistributionType : CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
                z += items[i].GetLogProb(array[indices[i]][indices2[i]]);
            return z;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IList{DistributionType}, IList{T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType>(IList<DistributionType> items, IList<IList<T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
            where DistributionType : CanGetLogProb<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="AverageLogFactor{DistributionType}(IList{DistributionType}, IList{T})"]/*'/>
        /// <typeparam name="DistributionType">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static double AverageLogFactor<DistributionType>(IList<DistributionType> items, IList<IList<T>> array)
            where DistributionType : CanGetLogProb<T>
        {
            return 0.0;
        }


#if true
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="LogEvidenceRatio{DistributionType}(IList{DistributionType}, IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int}, IList{DistributionType})"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static double LogEvidenceRatio<DistributionArrayType, DistributionType>(
            IList<DistributionType> items, IList<DistributionArrayType> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, IList<DistributionType> to_items)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>, CanGetLogAverageOf<DistributionType>, ICloneable
        {
            // this code is adapted from GetItemsOp
            double z = 0.0;
            if (items.Count <= 1)
                return 0.0;
            var productBefore = new Dictionary<Tuple<int, int>, DistributionType>();
            for (int i = 0; i < indices.Count; i++)
            {
                var key = Tuple.Create(indices[i], indices2[i]);
                DistributionType value;
                if (!productBefore.TryGetValue(key, out value))
                {
                    value = (DistributionType)array[indices[i]][indices2[i]].Clone();
                }
                z += value.GetLogAverageOf(items[i]);
                value.SetToProduct(value, items[i]);
                productBefore[key] = value;
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
        public static double LogEvidenceRatio<DistributionType, ArrayType>(ArrayType items, IList<DistributionType> array, IList<int> indices)
            where ArrayType : IList<DistributionType>, ICloneable
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="MarginalInit{ArrayType}(ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        public static ArrayType MarginalInit<ArrayType>([SkipIfUniform] ArrayType array)
            where ArrayType : ICloneable
        {
            return (ArrayType)array.Clone();
        }

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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="MarginalIncrement{ArrayType, DistributionType}(ArrayType, DistributionType, DistributionType, IReadOnlyList{int}, IReadOnlyList{int}, int)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static ArrayType MarginalIncrement<ArrayType, DistributionArrayType, DistributionType>(
            ArrayType result, DistributionType to_item, [SkipIfUniform] DistributionType item, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, int resultIndex)
            where ArrayType : IList<DistributionArrayType>, SettableTo<ArrayType>
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            int i = resultIndex;
            DistributionType value = result[indices[i]][indices2[i]];
            value.SetToProduct(to_item, item);
            result[indices[i]][indices2[i]] = value;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ItemsAverageConditional{ArrayType, DistributionType}(DistributionType, ArrayType, ArrayType, IReadOnlyList{int}, IReadOnlyList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="ArrayType">The type of an array for the marginal.</typeparam>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static DistributionType ItemsAverageConditional<ArrayType, DistributionArrayType, DistributionType>(
            [Indexed, Cancels] DistributionType items,
            [IgnoreDependency] ArrayType array, // must have an (unused) 'array' argument to determine the type of 'marginal' buffer
            [SkipIfAllUniform] ArrayType marginal,
            IReadOnlyList<int> indices, IReadOnlyList<int> indices2,
            int resultIndex,
            DistributionType result)
            where ArrayType : IList<DistributionArrayType>
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableToRatio<DistributionType>
        {
            int i = resultIndex;
            result.SetToRatio(marginal[indices[i]][indices2[i]], items);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageConditional{ArrayType, DistributionArrayType, DistributionType}(IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static ArrayType ArrayAverageConditional<ArrayType, DistributionArrayType, DistributionType>(
            [SkipIfAllUniform] IList<DistributionType> items, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, ArrayType result)
            where ArrayType : IList<DistributionArrayType>, SettableToUniform
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]][indices2[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]][indices2[i]] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageConditional{ArrayType, DistributionArrayType, DistributionType}(IList{T}, IReadOnlyList{int}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static ArrayType ArrayAverageConditional<ArrayType, DistributionArrayType, DistributionType>(
            [SkipIfAllUniform] IList<T> items, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, ArrayType result)
            where ArrayType : IList<DistributionArrayType>, SettableToUniform
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : HasPoint<T>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]][indices2[i]];
                value.Point = items[i];
                result[indices[i]][indices2[i]] = value;
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ItemsAverageLogarithm{DistributionArrayType, DistributionType}(IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int}, int, DistributionType)"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        public static DistributionType ItemsAverageLogarithm<DistributionArrayType, DistributionType>(
            [SkipIfAllUniform] IList<DistributionArrayType> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, int resultIndex, DistributionType result)
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            int i = resultIndex;
            result.SetTo(array[indices[i]][indices2[i]]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ItemsAverageLogarithmInit{TDist}(DistributionStructArray{TDist, T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageLogarithmInit<TDist>(
            [IgnoreDependency] IList<DistributionStructArray<TDist, T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
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
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]][indices2[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ItemsAverageLogarithmInit{TDist}(DistributionRefArray{TDist, T}, IReadOnlyList{int}, IReadOnlyList{int})"]/*'/>
        /// <typeparam name="TDist">The type of a distribution over array elements.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageLogarithmInit<TDist>(
            [IgnoreDependency] IList<DistributionRefArray<TDist, T>> array, IReadOnlyList<int> indices, IReadOnlyList<int> indices2)
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
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]][indices2[i]].Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IList{DistributionType}, IReadOnlyList{int}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionArrayType, DistributionType, ArrayType>(
            [SkipIfAllUniform] IList<DistributionType> items, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, ArrayType result)
            where ArrayType : IList<DistributionArrayType>, SettableToUniform
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : SettableToUniform, SettableToProduct<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]][indices2[i]];
                value.SetToProduct(value, items[i]);
                result[indices[i]][indices2[i]] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GetItemsFromJaggedOp{T}"]/message_doc[@name="ArrayAverageLogarithm{DistributionType, ArrayType}(IList{T}, IReadOnlyList{int}, IReadOnlyList{int}, ArrayType)"]/*'/>
        /// <typeparam name="DistributionArrayType">The type of a distribution over array elements.</typeparam>
        /// <typeparam name="DistributionType">The type of a distribution over inner array elements.</typeparam>
        /// <typeparam name="ArrayType">The type of the resulting array.</typeparam>
        public static ArrayType ArrayAverageLogarithm<ArrayType, DistributionArrayType, DistributionType>(IList<T> items, IReadOnlyList<int> indices, IReadOnlyList<int> indices2, ArrayType result)
            where ArrayType : IList<DistributionArrayType>, SettableToUniform
            where DistributionArrayType : IList<DistributionType>
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<ArrayType, DistributionArrayType, DistributionType>(items, indices, indices2, result);
        }
    }
}
