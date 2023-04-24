// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGammaFromShapeAndRateOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "TruncatedGammaFromShapeAndRate")]
    [Quality(QualityBand.Experimental)]
    public static class TruncatedGammaFromShapeAndRateOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGammaFromShapeAndRateOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGamma)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGamma sample)
        {
            return 0.0;
        }

        public static Gamma RateAverageConditional(double sample, double shape, double lowerBound, double upperBound)
        {
            // f(x,a,b,L) = b^a x^(a-1) exp(-bx) / GammaUpper(a,L*b)
            // if a = 2: b^2 x exp(-bx) / (bL + 1) / exp(-bL) = b^2 / (bL + 1) x exp(-b(x - L))
            if (shape != 1) throw new NotImplementedException();
            if (!double.IsPositiveInfinity(upperBound)) throw new NotImplementedException();
            // when shape=1 and upperBound=infinity, the factor reduces to:  f(x,b) = b exp(-b(x-L))
            return Gamma.FromShapeAndRate(shape + 1, sample - lowerBound);
        }

        public static Gamma RateAverageLogarithm(double sample, double shape, double lowerBound, double upperBound)
        {
            return RateAverageConditional(sample, shape, lowerBound, upperBound);
        }

        public static TruncatedGamma SampleAverageConditional(double shape, double rate, double lowerBound, double upperBound)
        {
            return new TruncatedGamma(shape, 1 / rate, lowerBound, upperBound);
        }

        public static TruncatedGamma SampleAverageLogarithm(double shape, double rate, double lowerBound, double upperBound)
        {
            return SampleAverageConditional(shape, rate, lowerBound, upperBound);
        }

        public static TruncatedGamma SampleAverageConditional(double shape, Gamma rate, double lowerBound, double upperBound)
        {
            throw new NotImplementedException();
        }

        public static TruncatedGamma SampleAverageLogarithm(double shape, Gamma rate, double lowerBound, double upperBound)
        {
            return SampleAverageConditional(shape, rate, lowerBound, upperBound);
        }

        public static Gamma RateAverageConditional(TruncatedGamma sample, double shape, double lowerBound, double upperBound)
        {
            if (sample.IsPointMass) return RateAverageConditional(sample.Point, shape, lowerBound, upperBound);
            if (shape != 1) throw new NotImplementedException();
            if (!double.IsPositiveInfinity(upperBound)) throw new NotImplementedException();
            // when shape=1 and upperBound=infinity, the factor reduces to:  f(x,b) = b exp(-b(x-L))
            if (!sample.Gamma.IsUniform()) throw new NotImplementedException();
            if (!double.IsPositiveInfinity(sample.UpperBound)) throw new NotImplementedException();
            if (sample.LowerBound <= lowerBound) return Gamma.Uniform();
            // when the message from x is uniform with a lower bound of L2 > L, then f(b) = int_L2^inf b exp(-b(x-L)) dx = exp(-b(L2-L))
            return Gamma.FromShapeAndRate(shape, sample.LowerBound - lowerBound);
        }

        public static Gamma RateAverageLogarithm(TruncatedGamma sample, double shape, double lowerBound, double upperBound)
        {
            return RateAverageConditional(sample, shape, lowerBound, upperBound);
        }
    }
}
