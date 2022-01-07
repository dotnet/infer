// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/doc/*'/>
    [FactorMethod(new string[] { "sample", "mean", "variance" }, typeof(Beta), "SampleFromMeanAndVariance")]
    [Quality(QualityBand.Stable)]
    public static class BetaFromMeanAndVarianceOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(Beta, double, double, Beta)"]/*'/>
        public static double LogAverageFactor(Beta sample, double mean, double variance, [Fresh] Beta to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(Beta, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Beta sample, double mean, double variance)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(Beta, double, double, Beta)"]/*'/>
        public static double AverageLogFactor(Beta sample, double mean, double variance, [Fresh] Beta to_sample)
        {
            return to_sample.GetAverageLog(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double mean, double variance)
        {
            return SampleAverageConditional(mean, variance).GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Beta SampleAverageLogarithm(double mean, double variance)
        {
            return Beta.FromMeanAndVariance(mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Beta SampleAverageConditional(double mean, double variance)
        {
            return Beta.FromMeanAndVariance(mean, variance);
        }
    }
}
