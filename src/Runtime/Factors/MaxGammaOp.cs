// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGammaOp"]/doc/*'/>
    [FactorMethod(new string[] { "max", "a", "b" }, typeof(Math), "Max", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Preview)]
    public static class MaxGammaOp
    {
        public static double LogEvidenceRatio(double max, Gamma a, double b)
        {
            return MaxAverageConditional(a, b).GetLogProb(max);
        }

        public static double LogEvidenceRatio(double max, double a, Gamma b)
        {
            return LogEvidenceRatio(max, b, a);
        }

        [Skip]
        public static double LogEvidenceRatio(TruncatedGamma max)
        {
            return 0.0;
        }

        public static TruncatedGamma MaxAverageConditional(Gamma a, double b)
        {
            return new TruncatedGamma(a, b, double.PositiveInfinity);
        }

        public static TruncatedGamma MaxAverageConditional(double a, Gamma b)
        {
            return MaxAverageConditional(b, a);
        }

        public static Gamma AAverageConditional([SkipIfUniform] TruncatedGamma max, Gamma a, double b)
        {
            if (max.IsUniform()) return Gamma.Uniform();
            if (a.IsPointMass) return max.Gamma;
            if (max.IsPointMass)
                return Gamma.PointMass(max.Point);
            TruncatedGamma aTruncated = new TruncatedGamma(a, 0, b);
            var product = max * aTruncated;
            var projected = product.ToGamma();
            return projected / a;
        }

        public static Gamma BAverageConditional([SkipIfUniform] TruncatedGamma max, double a, Gamma b)
        {
            return AAverageConditional(max, b, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGammaOp_PointMax"]/doc/*'/>
    [FactorMethod(new string[] { "max", "a", "b" }, typeof(Math), "Max", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class MaxGammaOp_PointMax
    {
        public static Gamma AAverageConditional([SkipIfUniform] TruncatedGamma max, double b)
        {
            if (!max.IsPointMass)
                throw new ArgumentException($"{nameof(max)} is not a point mass: {max}", nameof(max));
            return Gamma.PointMass(max.Point);
        }

        public static Gamma BAverageConditional([SkipIfUniform] TruncatedGamma max, double a)
        {
            return AAverageConditional(max, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxTruncatedGammaOp"]/doc/*'/>
    [FactorMethod(new string[] { "max", "a", "b" }, typeof(Math), "Max", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class MaxTruncatedGammaOp
    {
        public static double LogEvidenceRatio(double max, TruncatedGamma a, double b)
        {
            return MaxAverageConditional(a, b).GetLogProb(max);
        }

        public static double LogEvidenceRatio(double max, double a, TruncatedGamma b)
        {
            return LogEvidenceRatio(max, b, a);
        }

        public static TruncatedGamma MaxAverageConditional(TruncatedGamma a, double b)
        {
            return new TruncatedGamma(a.Gamma, Math.Max(b, a.LowerBound), a.UpperBound);
        }

        public static TruncatedGamma MaxAverageConditional(double a, TruncatedGamma b)
        {
            return MaxAverageConditional(b, a);
        }

        public static TruncatedGamma AAverageConditional([SkipIfUniform] TruncatedGamma max, double b)
        {
            return max;
        }

        public static TruncatedGamma BAverageConditional([SkipIfUniform] TruncatedGamma max, double a)
        {
            return AAverageConditional(max, a);
        }
    }
}
