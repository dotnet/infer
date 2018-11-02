// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    [FactorMethod(typeof(Factor), "Plus", typeof(Vector), typeof(Vector), Default = true)]
    [Buffers("SumMean", "SumVariance", "aMean", "aVariance", "bMean", "bVariance")]
    [Quality(QualityBand.Experimental)]
    public static class PlusVectorOp
    {
        [Skip]
        public static PositiveDefiniteMatrix SumVarianceInit([IgnoreDependency] VectorGaussian sum)
        {
            return new PositiveDefiniteMatrix(sum.Dimension, sum.Dimension);
        }

        [Fresh]
        public static PositiveDefiniteMatrix SumVariance([Proper] VectorGaussian sum, PositiveDefiniteMatrix result)
        {
            return sum.GetVariance(result);
        }

        [Skip]
        public static Vector SumMeanInit([IgnoreDependency] VectorGaussian sum)
        {
            return Vector.Zero(sum.Dimension);
        }

        [Fresh]
        public static Vector SumMean([Proper] VectorGaussian sum, PositiveDefiniteMatrix sumVariance, Vector result)
        {
            return sum.GetMean(result, sumVariance);
        }

        [Skip]
        public static PositiveDefiniteMatrix AVarianceInit([IgnoreDependency] VectorGaussian a)
        {
            return new PositiveDefiniteMatrix(a.Dimension, a.Dimension);
        }

        [Fresh]
        public static PositiveDefiniteMatrix AVariance([Proper] VectorGaussian a, PositiveDefiniteMatrix result)
        {
            return a.GetVariance(result);
        }

        [Skip]
        public static Vector AMeanInit([IgnoreDependency] VectorGaussian a)
        {
            return Vector.Zero(a.Dimension);
        }

        [Fresh]
        public static Vector AMean([Proper] VectorGaussian a, PositiveDefiniteMatrix aVariance, Vector result)
        {
            return a.GetMean(result, aVariance);
        }

        [Skip]
        public static PositiveDefiniteMatrix BVarianceInit([IgnoreDependency] VectorGaussian b)
        {
            return new PositiveDefiniteMatrix(b.Dimension, b.Dimension);
        }

        [Fresh]
        public static PositiveDefiniteMatrix BVariance([Proper] VectorGaussian b, PositiveDefiniteMatrix result)
        {
            return b.GetVariance(result);
        }

        [Skip]
        public static Vector BMeanInit([IgnoreDependency] VectorGaussian b)
        {
            return Vector.Zero(b.Dimension);
        }

        [Fresh]
        public static Vector BMean([Proper] VectorGaussian b, PositiveDefiniteMatrix bVariance, Vector result)
        {
            return b.GetMean(result, bVariance);
        }

        public static VectorGaussian SumConditional(Vector a, Vector b, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetToSum(a, b);
            result.Point = result.MeanTimesPrecision;
            return result;
        }

        public static VectorGaussian AConditional(Vector sum, Vector b, VectorGaussian result)
        {
            result.MeanTimesPrecision.SetToDifference(sum, b);
            result.Point = result.MeanTimesPrecision;
            return result;
        }

        public static VectorGaussian SumAverageConditional([SkipIfUniform] VectorGaussian a, Vector b, VectorGaussian result)
        {
            if (a.IsPointMass)
            {
                return SumConditional(a.Point, b, result);
            }
            result.Precision.SetTo(a.Precision);
            result.MeanTimesPrecision.SetToProduct(a.Precision, b);
            result.MeanTimesPrecision.SetToSum(result.MeanTimesPrecision, a.MeanTimesPrecision);
            return result;
        }

        public static VectorGaussian SumAverageConditional(Vector a, [SkipIfUniform] VectorGaussian b, VectorGaussian result)
        {
            return SumAverageConditional(b, a, result);
        }

        public static VectorGaussian SumAverageConditional(
            Vector aMean, PositiveDefiniteMatrix aVariance,
            Vector bMean, PositiveDefiniteMatrix bVariance,
            VectorGaussian result)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            int dimension = result.Dimension;

            var sumMean = Vector.Zero(dimension);
            var sumVariance = PositiveDefiniteMatrix.IdentityScaledBy(dimension, 0);
            sumMean.SetToSum(aMean, bMean);
            sumVariance.SetToSum(aVariance, bVariance);

            result.SetMeanAndVariance(sumMean, sumVariance);
            return result;
        }

        public static VectorGaussian AAverageConditional(
            Vector sumMean, PositiveDefiniteMatrix sumVariance,
            Vector bMean, PositiveDefiniteMatrix bVariance,
            VectorGaussian result)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            int dimension = result.Dimension;

            var aMean = Vector.Zero(dimension);
            var aVariance = PositiveDefiniteMatrix.IdentityScaledBy(dimension, 0);
            aMean.SetToDifference(sumMean, bMean);
            aVariance.SetToSum(sumVariance, bVariance);

            result.SetMeanAndVariance(aMean, aVariance);
            return result;
        }

        public static VectorGaussian BAverageConditional(
            Vector sumMean, PositiveDefiniteMatrix sumVariance,
            Vector aMean, PositiveDefiniteMatrix aVariance,
            VectorGaussian result)
        {
            return AAverageConditional(sumMean, sumVariance, aMean, aVariance, result);
        }

        public static VectorGaussian AAverageConditional([SkipIfUniform] VectorGaussian sum, Vector b, VectorGaussian result)
        {
            if (sum.IsPointMass)
                return AConditional(sum.Point, b, result);
            result.Precision.SetTo(sum.Precision);
            result.MeanTimesPrecision.SetToProduct(sum.Precision, b);
            result.MeanTimesPrecision.SetToDifference(sum.MeanTimesPrecision, result.MeanTimesPrecision);
            return result;
        }

        public static VectorGaussian BAverageConditional([SkipIfUniform] VectorGaussian sum, Vector a, VectorGaussian result)
        {
            return AAverageConditional(sum, a, result);
        }

        [Skip]
        public static double LogEvidenceRatio(VectorGaussian sum)
        {
            return 0.0;
        }

        public static double LogAverageFactor(Vector sum, Vector a, Vector b)
        {
            Vector aPlusb = a + b;
            return sum.Equals(aPlusb) ? 0.0 : double.NegativeInfinity;
        }

        public static double LogEvidenceRatio(Vector sum, Vector a, Vector b)
        {
            return LogAverageFactor(sum, a, b);
        }

        public static double LogAverageFactor(Vector sum, Vector aMean, PositiveDefiniteMatrix aVariance, Vector bMean, PositiveDefiniteMatrix bVariance)
        {
            VectorGaussian toSum = new VectorGaussian(sum.Count);
            toSum = SumAverageConditional(aMean, aVariance, bMean, bVariance, toSum);
            return toSum.GetLogProb(sum);
        }

        public static double LogEvidenceRatio(Vector sum, Vector aMean, PositiveDefiniteMatrix aVariance, Vector bMean, PositiveDefiniteMatrix bVariance)
        {
            return LogAverageFactor(sum, aMean, aVariance, bMean, bVariance);
        }

        // VMP /////////////////////////////////////////////////////////////////////////////////////////////////////////

        public static VectorGaussian SumAverageLogarithm([SkipIfUniform] VectorGaussian a, Vector b, VectorGaussian result)
        {
            return SumAverageConditional(a, b, result);
        }

        public static VectorGaussian SumAverageLogarithm(Vector a, [SkipIfUniform] VectorGaussian b, VectorGaussian result)
        {
            return SumAverageLogarithm(b, a, result);
        }

        public static VectorGaussian SumAverageLogarithm(
            Vector aMean, PositiveDefiniteMatrix aVariance,
            Vector bMean, PositiveDefiniteMatrix bVariance,
            VectorGaussian result)
        {
            return SumAverageConditional(aMean, aVariance, bMean, bVariance, result);
        }

        public static VectorGaussian AAverageLogarithm([SkipIfUniform] VectorGaussian sum, Vector bMean, VectorGaussian result)
        {
            return AAverageConditional(sum, bMean, result);
        }

        public static VectorGaussian BAverageLogarithm([SkipIfUniform] VectorGaussian sum, Vector aMean, VectorGaussian result)
        {
            return AAverageLogarithm(sum, aMean, result);
        }

        public static double AverageLogFactor(Vector sum, Vector a, Vector b)
        {
            return LogAverageFactor(sum, a, b);
        }

        [Skip]
        public static double AverageLogFactor(VectorGaussian sum)
        {
            return 0.0;
        }

        [Skip]
        public static double AverageLogFactor(Vector sum, VectorGaussian a)
        {
            return 0.0;
        }

        [Skip]
        public static double AverageLogFactor(Vector sum, Vector a, VectorGaussian b)
        {
            return 0.0;
        }
    }
}
