// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "AreEqual", typeof(string), typeof(string))]
    [Quality(QualityBand.Experimental)]
    public static class StringsAreEqualOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(StringDistribution, StringDistribution)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(StringDistribution str1, StringDistribution str2)
        {
            return Bernoulli.FromLogOdds(MMath.LogitFromLog(str1.GetLogAverageOf(str2)));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/message_doc[@name="Str1AverageConditional(StringDistribution, Bernoulli, StringDistribution)"]/*'/>
        public static StringDistribution Str1AverageConditional(
            StringDistribution str2, [SkipIfUniform] Bernoulli areEqual, StringDistribution result)
        {
            StringDistribution uniform = StringDistribution.Any();
            double probNotEqual = areEqual.GetProbFalse();
            if (probNotEqual > 0.5)
            {
                throw new NotImplementedException("Non-equality case is not yet supported.");
            }

            double logWeight1 = Math.Log(1 - (2 * probNotEqual));
            double logWeight2 = Math.Log(probNotEqual) + uniform.GetLogNormalizer();
            result.SetToSumLog(logWeight1, str2, logWeight2, uniform);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/message_doc[@name="Str2AverageConditional(StringDistribution, Bernoulli, StringDistribution)"]/*'/>
        public static StringDistribution Str2AverageConditional(
            StringDistribution str1, [SkipIfUniform] Bernoulli areEqual, StringDistribution result)
        {
            return Str1AverageConditional(str1, areEqual, result);
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, StringDistribution, StringDistribution)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, StringDistribution str1, StringDistribution str2)
        {
            return LogAverageFactor(areEqual, str1, str2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringsAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, StringDistribution, StringDistribution)"]/*'/>
        public static double LogAverageFactor(bool areEqual, StringDistribution str1, StringDistribution str2)
        {
            Bernoulli messageToAreEqual = AreEqualAverageConditional(str1, str2);
            return messageToAreEqual.GetLogProb(areEqual);
        }

        #endregion
    }
}
