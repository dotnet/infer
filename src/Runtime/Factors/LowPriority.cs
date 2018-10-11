// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    [Hidden]
    public static class LowPriority
    {
        public static T Forward<T>(T value)
        {
            return value;
        }

        public static T Backward<T>(T value)
        {
            return value;
        }

        [ParameterNames("first", "value", "second")]
        public static T SequentialCopy<T>(T value, out T second)
        {
            second = value;
            return value;
        }

        [ParameterNames("first", "value", "second")]
        public static T SequentialCut<T>(T value, out T second)
        {
            second = value;
            return value;
        }
    }

    [FactorMethod(typeof(LowPriority), "SequentialCut<>")]
    [Quality(QualityBand.Experimental)]
    public static class SequentialCutOp<T>
    {
        public static TDist FirstAverageConditional<TDist>([IsReturned] TDist value)
            where TDist : IDistribution<T>
        {
            return value;
        }

        public static T FirstAverageConditional([IsReturned] T value)
        {
            return value;
        }

        public static TDist SecondAverageConditional<TDist>([IsReturned] TDist value)
            where TDist : IDistribution<T>
        {
            return value;
        }

        public static TDist ValueAverageConditional<TDist>([IsReturned] TDist first)
            where TDist : IDistribution<T>
        {
            return first;
        }
    }

    [FactorMethod(typeof(LowPriority), "SequentialCopy<>")]
    [Quality(QualityBand.Preview)]
    public static class SequentialCopyOp
    {
        [SkipIfAllUniform]
        public static double LogEvidenceRatio<T>(T value, T first, T second)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, ICloneable
        {
            T valueTimesFirst = (T)value.Clone();
            valueTimesFirst.SetToProduct(value, first);
            return value.GetLogAverageOf(second) - valueTimesFirst.GetLogAverageOf(second);
        }

        [SkipIfAllUniform]
        public static T FirstAverageConditional<T>(T value, [NoInit] T second, T result)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(value, second);
            return result;
        }

        [SkipIfAllUniform]
        public static T SecondAverageConditional<T>(T value, [RequiredArgument] T first, T result)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(value, first);
            return result;
        }

        [SkipIfAllUniform]
        public static T ValueAverageConditional<T>(T first, T second, T result)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(first, second);
            return result;
        }
    }

    [FactorMethod(typeof(LowPriority), "Forward<>")]
    [Quality(QualityBand.Preview)]
    public static class LowPriorityForwardOp
    {
        public static T ForwardAverageConditional<T>([NoInit] T value)
        {
            return value;
        }

        public static T ValueAverageConditional<T>([IsReturned] T forward)
        {
            return forward;
        }
    }

    [FactorMethod(typeof(LowPriority), "Backward<>")]
    [Quality(QualityBand.Preview)]
    public static class LowPriorityBackwardOp
    {
        public static T BackwardAverageConditional<T>([IsReturned] T value)
        {
            return value;
        }

        public static T ValueAverageConditional<T>([NoInit, IsReturned] T backward)
        {
            return backward;
        }

        [Skip]
        public static double LogEvidenceRatio<T>(T backward)
        {
            return 0.0;
        }
    }
}
