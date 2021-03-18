// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    [FactorMethod(typeof(Factor), "SumExcept")]
    [Quality(QualityBand.Experimental)]
    public static class SumExceptOp
    {
        [Skip]
        public static double LogEvidenceRatio(Gaussian sumExcept)
        {
            return 0.0;
        }

        public static Gaussian SumExceptAverageConditional([SkipIfAllUniform] IReadOnlyList<Gaussian> array, int index)
        {
            if(array.Count == 2)
            {
                return array[1 - index];
            }
            double mean = 0;
            double variance = 0;
            for (int i = 0; i < array.Count; i++)
            {
                if (i == index)
                    continue;
                if (array[i].Precision == 0)
                    return array[i];
                double mean1;
                double variance1;
                array[i].GetMeanAndVariance(out mean1, out variance1);
                mean += mean1;
                variance += variance1;
            }
            return Gaussian.FromMeanAndVariance(mean, variance);
        }

        public static ArrayType ArrayAverageConditional<ArrayType>([SkipIfUniform] Gaussian sumExcept, IReadOnlyList<Gaussian> array, int index, ArrayType result)
            where ArrayType : IList<Gaussian>
        {
            if (sumExcept.Precision == 0 || array.Count <= 2)
            {
                for (int i = 0; i < result.Count; i++)
                {
                    if (i == index)
                    {
                        result[i] = Gaussian.Uniform();
                    }
                    else
                    {
                        result[i] = sumExcept;
                    }
                }
                return result;
            }
            double sumMean, sumVariance;
            sumExcept.GetMeanAndVarianceImproper(out sumMean, out sumVariance);
            double[] means = new double[array.Count];
            double[] variances = new double[array.Count];
            for (int i = 0; i < array.Count; i++)
            {
                // could generalize this to be predicate(i)
                if (i == index)
                    continue;
                double mean1, variance1;
                array[i].GetMeanAndVarianceImproper(out mean1, out variance1);
                means[i] = mean1;
                variances[i] = variance1;
            }
            double[] meanPrevious = new double[array.Count];
            double[] variancePrevious = new double[array.Count];
            for (int i = 1; i < array.Count; i++)
            {
                // i == index doesn't matter since it will have mean=0, variance=0
                meanPrevious[i] = meanPrevious[i - 1] + means[i - 1];
                variancePrevious[i] = variancePrevious[i - 1] + variances[i - 1];
            }
            double meanNext = 0;
            double varianceNext = 0;
            for (int i = array.Count - 1; i >= 0; i--)
            {
                if (i == index)
                {
                    result[i] = Gaussian.Uniform();
                    continue;
                }
                double sumOtherMeans = meanPrevious[i] + meanNext;
                double sumOtherVariances = variancePrevious[i] + varianceNext;
                result[i] = Gaussian.FromMeanAndVariance(sumMean - sumOtherMeans, sumVariance + sumOtherVariances);
                meanNext += means[i];
                varianceNext += variances[i];
            }
            return result;
        }
    }
}
