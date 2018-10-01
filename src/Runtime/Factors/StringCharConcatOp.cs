// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Concat", typeof(string), typeof(char))]
    [Quality(QualityBand.Experimental)]
    public static class StringCharConcatOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/message_doc[@name="ConcatAverageConditional(StringDistribution, DiscreteChar)"]/*'/>
        public static StringDistribution ConcatAverageConditional(StringDistribution str, DiscreteChar ch)
        {
            return StringConcatOp.ConcatAverageConditional(str, StringDistribution.Char(ch));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/message_doc[@name="StrAverageConditional(StringDistribution, DiscreteChar)"]/*'/>
        public static StringDistribution StrAverageConditional(StringDistribution concat, DiscreteChar ch)
        {
            return StringConcatOp.Str1AverageConditional(concat, StringDistribution.Char(ch));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/message_doc[@name="ChAverageConditional(StringDistribution, StringDistribution)"]/*'/>
        public static DiscreteChar ChAverageConditional(StringDistribution concat, StringDistribution str)
        {
            var result = StringConcatOp.Str2AverageConditional(concat, str);
            return SingleOp.CharacterAverageConditional(result);
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/message_doc[@name="LogAverageFactor(StringDistribution, StringDistribution, DiscreteChar)"]/*'/>
        public static double LogAverageFactor(StringDistribution concat, StringDistribution str, DiscreteChar ch)
        {
            return StringConcatOp.LogAverageFactor(concat, str, StringDistribution.Char(ch));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/message_doc[@name="LogEvidenceRatio(StringDistribution, StringDistribution, DiscreteChar)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution concat, StringDistribution str, DiscreteChar ch)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringCharConcatOp"]/message_doc[@name="LogEvidenceRatio(String, StringDistribution, DiscreteChar)"]/*'/>
        public static double LogEvidenceRatio(string concat, StringDistribution str, DiscreteChar ch)
        {
            return StringConcatOp.LogEvidenceRatio(concat, str, StringDistribution.Char(ch));
        }

        #endregion
    }
}
