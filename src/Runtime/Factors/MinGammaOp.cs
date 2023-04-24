// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MinGammaOp"]/doc/*'/>
    [FactorMethod(new string[] { "min", "a", "b" }, typeof(Math), "Min", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class MinGammaOp
    {
        public static double LogEvidenceRatio(double min, double a, Gamma b)
        {
            TruncatedGamma to_min = MinAverageConditional(a, b);
            return to_min.GetLogProb(min);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MinGammaOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGamma,Gamma,double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGamma min, Gamma a, double b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MinGammaOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGamma,double,Gamma)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGamma min, double a, Gamma b)
        {
            return 0.0;
        }

        public static TruncatedGamma MinAverageConditional(Gamma a, double b)
        {
            return new TruncatedGamma(a, 0, b);
        }

        public static TruncatedGamma MinAverageConditional(double a, Gamma b)
        {
            return MinAverageConditional(b, a);
        }

        public static Gamma AAverageConditional([SkipIfUniform] TruncatedGamma min, double b)
        {
            if (min.IsUniform()) return Gamma.Uniform();
            if (min.LowerBound == 0 && min.UpperBound == b) return min.Gamma;
            if (!min.IsPointMass)
                throw new ArgumentException("min is not a point mass");
            return Gamma.PointMass(min.Point);
        }

        public static Gamma BAverageConditional([SkipIfUniform] TruncatedGamma min, double a)
        {
            return AAverageConditional(min, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MinTruncatedGammaOp"]/doc/*'/>
    [FactorMethod(new string[] { "min", "a", "b" }, typeof(Math), "Min", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class MinTruncatedGammaOp
    {
        public static double LogEvidenceRatio(double min, double a, TruncatedGamma b)
        {
            TruncatedGamma to_min = MinAverageConditional(a, b);
            return to_min.GetLogProb(min);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MinGammaOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGamma,TruncatedGamma,double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGamma min, TruncatedGamma a, double b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MinGammaOp"]/message_doc[@name="LogEvidenceRatio(TruncatedGamma,double,TruncatedGamma)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(TruncatedGamma min, double a, TruncatedGamma b)
        {
            return 0.0;
        }

        public static TruncatedGamma MinAverageConditional(TruncatedGamma a, double b)
        {
            return new TruncatedGamma(a.Gamma, a.LowerBound, Math.Min(b, a.UpperBound));
        }

        public static TruncatedGamma MinAverageConditional(double a, TruncatedGamma b)
        {
            return MinAverageConditional(b, a);
        }

        public static TruncatedGamma AAverageConditional([SkipIfUniform] TruncatedGamma min, double b)
        {
            if (min.IsUniform()) return TruncatedGamma.Uniform();
            if (min.UpperBound == b) return new TruncatedGamma(min.Gamma, min.LowerBound, double.PositiveInfinity);
            if (!min.IsPointMass)
                throw new ArgumentException("min is not a point mass");
            return TruncatedGamma.PointMass(min.Point);
        }

        public static TruncatedGamma BAverageConditional([SkipIfUniform] TruncatedGamma min, double a)
        {
            return AAverageConditional(min, a);
        }
    }
}
