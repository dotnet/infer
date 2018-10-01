// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Concat", typeof(char), typeof(string))]
    [Quality(QualityBand.Experimental)]
    public static class CharStringConcatOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/message_doc[@name="ConcatAverageConditional(DiscreteChar, StringDistribution)"]/*'/>
        public static StringDistribution ConcatAverageConditional(DiscreteChar ch, StringDistribution str)
        {
            return StringConcatOp.ConcatAverageConditional(StringDistribution.Char(ch), str);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/message_doc[@name="StrAverageConditional(StringDistribution, DiscreteChar)"]/*'/>
        public static StringDistribution StrAverageConditional(StringDistribution concat, DiscreteChar ch)
        {
            return StringConcatOp.Str2AverageConditional(concat, StringDistribution.Char(ch));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/message_doc[@name="ChAverageConditional(StringDistribution, StringDistribution)"]/*'/>
        public static DiscreteChar ChAverageConditional(StringDistribution concat, StringDistribution str)
        {
            var result = StringConcatOp.Str1AverageConditional(concat, str);
            return SingleOp.CharacterAverageConditional(result);
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/message_doc[@name="LogAverageFactor(StringDistribution, DiscreteChar, StringDistribution)"]/*'/>
        public static double LogAverageFactor(StringDistribution concat, DiscreteChar ch, StringDistribution str)
        {
            return StringConcatOp.LogAverageFactor(concat, StringDistribution.Char(ch), str);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/message_doc[@name="LogEvidenceRatio(StringDistribution, DiscreteChar, StringDistribution)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution concat, DiscreteChar ch, StringDistribution str)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharStringConcatOp"]/message_doc[@name="LogEvidenceRatio(String, DiscreteChar, StringDistribution)"]/*'/>
        public static double LogEvidenceRatio(string concat, DiscreteChar ch, StringDistribution str)
        {
            return StringConcatOp.LogEvidenceRatio(concat, StringDistribution.Char(ch), str);
        }

        #endregion
    }
}
