// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    using System;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/doc/*'/>
    [FactorMethod(typeof(Beta), "Sample", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class BetaOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="LogAverageFactor(Beta, double, double, Beta)"]/*'/>
        public static double LogAverageFactor(Beta sample, double trueCount, double falseCount, [Fresh] Beta to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="LogEvidenceRatio(Beta, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Beta sample, double trueCount, double falseCount)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="AverageLogFactor(Beta, double, double, Beta)"]/*'/>
        public static double AverageLogFactor(Beta sample, double trueCount, double falseCount, [Fresh] Beta to_sample)
        {
            return to_sample.GetAverageLog(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double trueCount, double falseCount)
        {
            return SampleAverageConditional(trueCount, falseCount).GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double trueCount, double falseCount)
        {
            return LogAverageFactor(sample, trueCount, falseCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double trueCount, double falseCount)
        {
            return LogAverageFactor(sample, trueCount, falseCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Beta SampleAverageLogarithm(double trueCount, double falseCount)
        {
            return new Beta(trueCount, falseCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Beta SampleAverageConditional(double trueCount, double falseCount)
        {
            return new Beta(trueCount, falseCount);
        }

        const string TrueCountMustBeOneMessage = "falseCount is Gamma and trueCount is not 1";
        const string FalseCountMustBeOneMessage = "trueCount is Gamma and falseCount is not 1";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="AverageLogFactor(double, Gamma, double)"]/*'/>
        public static double AverageLogFactor(double sample, Gamma trueCount, double falseCount)
        {
            if (trueCount.IsPointMass)
            {
                return LogEvidenceRatio(sample, trueCount.Point, falseCount);
            }
            else if (falseCount == 1)
            {
                // The factor is f(x, a) = a x^(a-1)
                // whose logarithm is log(a) + (a-1)*log(x)
                return trueCount.GetMeanLog() + (trueCount.GetMean() - 1) * Math.Log(sample);
            }
            else
            {
                throw new NotSupportedException(FalseCountMustBeOneMessage);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="LogEvidenceRatio(double, Gamma, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, Gamma trueCount, double falseCount)
        {
            if (trueCount.IsPointMass)
            {
                return LogEvidenceRatio(sample, trueCount.Point, falseCount);
            }
            else if (falseCount == 1)
            {
                // The factor is f(x, a) = a x^(a-1)
                // f(x, a) Ga(a; s, r) = a^s exp(-r*a + (a-1)*log(x)) r^s / Gamma(s)
                // Z = Gamma(s+1)/Gamma(s) * r^s / (r - log(x))^(s+1) / x
                return Math.Log(trueCount.Shape) - Math.Log(sample) + trueCount.Shape * Math.Log(trueCount.Rate) - (trueCount.Shape + 1) * Math.Log(trueCount.Rate - Math.Log(sample));
            }
            else
            {
                throw new NotSupportedException(FalseCountMustBeOneMessage);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="AverageLogFactor(double, double, Gamma)"]/*'/>
        public static double AverageLogFactor(double sample, double trueCount, Gamma falseCount)
        {
            if (falseCount.IsPointMass)
            {
                return LogEvidenceRatio(sample, trueCount, falseCount.Point);
            }
            else if (trueCount == 1)
            {
                // The factor is f(x, b) = b (1-x)^(b-1)
                return falseCount.GetMeanLog() + (falseCount.GetMean() - 1) * Math.Log(1 - sample);
            }
            else
            {
                throw new NotSupportedException(TrueCountMustBeOneMessage);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="LogEvidenceRatio(double, double, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double sample, double trueCount, Gamma falseCount)
        {
            if (falseCount.IsPointMass)
            {
                return LogEvidenceRatio(sample, trueCount, falseCount.Point);
            }
            else if (trueCount == 1)
            {
                // The factor is f(x, b) = b (1-x)^(b-1)
                return Math.Log(falseCount.Shape) - Math.Log(1 - sample) + falseCount.Shape * Math.Log(falseCount.Rate) - (falseCount.Shape + 1) * Math.Log(falseCount.Rate - Math.Log(1 - sample));
            }
            else
            {
                throw new NotSupportedException(TrueCountMustBeOneMessage);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="TrueCountAverageConditional(double, double)"]/*'/>
        public static Gamma TrueCountAverageConditional(double sample, double falseCount)
        {
            return TrueCountAverageLogarithm(sample, falseCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="TrueCountAverageLogarithm(double, double)"]/*'/>
        public static Gamma TrueCountAverageLogarithm(double sample, double falseCount)
        {
            if (falseCount == 1)
            {
                return Gamma.FromShapeAndRate(2, -Math.Log(sample));
            }
            else
            {
                throw new NotSupportedException(FalseCountMustBeOneMessage);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="FalseCountAverageConditional(double, double)"]/*'/>
        public static Gamma FalseCountAverageConditional(double sample, double trueCount)
        {
            return FalseCountAverageLogarithm(sample, trueCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaOp"]/message_doc[@name="FalseCountAverageLogarithm(double, double)"]/*'/>
        public static Gamma FalseCountAverageLogarithm(double sample, double trueCount)
        {
            if (trueCount == 1)
            {
                return Gamma.FromShapeAndRate(2, -Math.Log(1 - sample));
            }
            else
            {
                throw new NotSupportedException(TrueCountMustBeOneMessage);
            }
        }
    }
}
