// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    [FactorMethod(typeof(Factor), "Split<>")]
    [Quality(QualityBand.Preview)]
    public static class SplitOp<T>
    {
        public static double LogEvidenceRatio(IList<T> array, IList<T> head, int count, IList<T> tail)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            for (int i = 0; i < count; i++)
            {
                if (!equalityComparer.Equals(array[i], head[i]))
                    return double.NegativeInfinity;
            }
            for (int i = 0; i < tail.Count; i++)
            {
                if (!equalityComparer.Equals(array[i + count], tail[i]))
                    return double.NegativeInfinity;
            }
            return 0.0;
        }

        [Skip]
        public static double LogEvidenceRatio<ItemType>(IList<ItemType> array, IList<ItemType> head, int count, IList<ItemType> tail)
            where ItemType : IDistribution<T>
        {
            return 0.0;
        }

        public static double LogEvidenceRatio<ItemType>(IList<ItemType> array, IList<T> head, int count, IList<T> tail)
            where ItemType : CanGetLogProb<T>
        {
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                sum += array[i].GetLogProb(head[i]);
            }
            for (int i = 0; i < tail.Count; i++)
            {
                sum += array[i + count].GetLogProb(tail[i]);
            }
            return sum;
        }

        public static double LogEvidenceRatio<ItemType>(IList<ItemType> array, IList<ItemType> head, int count, IList<T> tail)
            where ItemType : CanGetLogProb<T>
        {
            double sum = 0;
            for (int i = 0; i < tail.Count; i++)
            {
                sum += array[i + count].GetLogProb(tail[i]);
            }
            return sum;
        }

        public static double LogEvidenceRatio<ItemType>(IList<ItemType> array, IList<T> head, int count, IList<ItemType> tail)
            where ItemType : CanGetLogProb<T>
        {
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                sum += array[i].GetLogProb(head[i]);
            }
            return sum;
        }

        public static ArrayType ArrayAverageConditional<ArrayType, ItemType>(IList<ItemType> head, int count, IList<ItemType> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>
        {
            // Vijay
            if (result != null && !(typeof(IList<SettableTo<ItemType>>).IsAssignableFrom(result.GetType())))
            {
                throw new InvalidOperationException($"result is {result.GetType()}, but expecting: {typeof(IList<SettableTo<ItemType>>)}");
            }

            for (int i = 0; i < count; i++)
            {
                ItemType item = result[i];
                item.SetTo(head[i]);
                result[i] = item;
            }
            for (int i = count; i < result.Count; i++)
            {
                ItemType item = result[i];
                item.SetTo(tail[i - count]);
                result[i] = item;                
            }
            return result;
        }

        public static ArrayType ArrayAverageConditional<ArrayType, ItemType>(IList<T> head, int count, IList<ItemType> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>, HasPoint<T>
        {
            for (int i = 0; i < count; i++)
            {
                ItemType item = result[i];
                item.Point = head[i];
                result[i] = item;
            }
            for (int i = count; i < result.Count; i++)
            {
                ItemType item = result[i];
                item.SetTo(tail[i - count]);
                result[i] = item;
            }
            return result;
        }

        public static ArrayType ArrayAverageConditional<ArrayType, ItemType>(IList<ItemType> head, int count, IList<T> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>, HasPoint<T>
        {
            for (int i = 0; i < count; i++)
            {
                ItemType item = result[i];
                item.SetTo(head[i]);
                result[i] = item;
            }
            for (int i = count; i < result.Count; i++)
            {
                ItemType item = result[i];
                item.Point = tail[i - count];
                result[i] = item;
            }
            return result;
        }

        public static ArrayType ArrayAverageConditional<ArrayType, ItemType>(IList<T> head, int count, IList<T> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : HasPoint<T>
        {
            for (int i = 0; i < count; i++)
            {
                ItemType item = result[i];
                item.Point = head[i];
                result[i] = item;
            }
            for (int i = count; i < result.Count; i++)
            {
                ItemType item = result[i];
                item.Point = tail[i - count];
                result[i] = item;
            }
            return result;
        }

        public static ArrayType HeadAverageConditional<ArrayType, ItemType>(IList<ItemType> array, int count, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>
        {
            if (result.Count != count)
                throw new ArgumentException($"result.Count ({result.Count}) != count ({count})");
            for (int i = 0; i < count; i++)
            {
                ItemType item = result[i];
                item.SetTo(array[i]);
                result[i] = item;
            }
            return result;
        }

        public static ArrayType TailAverageConditional<ArrayType, ItemType>(IList<ItemType> array, int count, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>
        {
            int tailCount = array.Count - count;
            if (result.Count != tailCount)
                throw new ArgumentException($"result.Count ({result.Count}) != array.Length ({array.Count}) - count ({count})");
            for (int i = 0; i < tailCount; i++)
            {
                ItemType item = result[i];
                item.SetTo(array[i + count]);
                result[i] = item;
            }
            return result;
        }

        public static ArrayType TailAverageConditional<ArrayType>(IList<T> array, int count, ArrayType result)
            where ArrayType : IList<T>
        {
            int tailCount = array.Count - count;
            if (result.Count != tailCount)
                throw new ArgumentException($"result.Count ({result.Count}) != array.Length ({array.Count}) - count ({count})");
            for (int i = 0; i < tailCount; i++)
            {
                result[i] = array[i + count];
            }
            return result;
        }

        // VMP ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        public static ArrayType ArrayAverageLogarithm<ArrayType, ItemType>(IList<ItemType> head, int count, IList<ItemType> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>
        {
            return ArrayAverageConditional(head, count, tail, result);
        }

        public static ArrayType ArrayAverageLogarithm<ArrayType, ItemType>(IList<T> head, int count, IList<ItemType> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>, HasPoint<T>
        {
            return ArrayAverageConditional(head, count, tail, result);
        }

        public static ArrayType ArrayAverageLogarithm<ArrayType, ItemType>(IList<ItemType> head, int count, IList<T> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>, HasPoint<T>
        {
            return ArrayAverageConditional(head, count, tail, result);
        }

        public static ArrayType ArrayAverageLogarithm<ArrayType, ItemType>(IList<T> head, int count, IList<T> tail, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : HasPoint<T>
        {
            return ArrayAverageConditional<ArrayType, ItemType>(head, count, tail, result);
        }

        public static ArrayType HeadAverageLogarithm<ArrayType, ItemType>(IList<ItemType> array, int count, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>
        {
            return HeadAverageConditional(array, count, result);
        }

        public static ArrayType TailAverageLogarithm<ArrayType, ItemType>(IList<ItemType> array, int count, ArrayType result)
            where ArrayType : IList<ItemType>
            where ItemType : SettableTo<ItemType>
        {
            return TailAverageConditional(array, count, result);
        }

        public static ArrayType TailAverageLogarithm<ArrayType>(IList<T> array, int count, ArrayType result)
            where ArrayType : IList<T>
        {
            return TailAverageConditional<ArrayType>(array, count, result);
        }

    }
}
