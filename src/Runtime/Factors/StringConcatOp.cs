// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Distributions;
    using Distributions.Automata;
    using Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Concat", typeof(string), typeof(string))]
    [Quality(QualityBand.Preview)]
    public static class StringConcatOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/message_doc[@name="ConcatAverageConditional(StringDistribution, StringDistribution)"]/*'/>
        public static StringDistribution ConcatAverageConditional(StringDistribution str1, StringDistribution str2)
        {
            Argument.CheckIfNotNull(str1, "str1");
            Argument.CheckIfNotNull(str2, "str2");
            
            return str1 + str2;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/message_doc[@name="Str1AverageConditional(StringDistribution, StringDistribution)"]/*'/>
        public static StringDistribution Str1AverageConditional(StringDistribution concat, StringDistribution str2)
        {
            Argument.CheckIfNotNull(concat, "concat");
            Argument.CheckIfNotNull(str2, "str2");
            
            StringTransducer transducer = StringTransducer.Copy();
            transducer.AppendInPlace(StringTransducer.Consume(str2.ToAutomaton()));
            return StringDistribution.FromWeightFunction(transducer.ProjectSource(concat.ToAutomaton()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/message_doc[@name="Str2AverageConditional(StringDistribution, StringDistribution)"]/*'/>
        public static StringDistribution Str2AverageConditional(StringDistribution concat, StringDistribution str1)
        {
            Argument.CheckIfNotNull(concat, "concat");
            Argument.CheckIfNotNull(str1, "str1");

            StringTransducer transducer = StringTransducer.Consume(str1.ToAutomaton());
            transducer.AppendInPlace(StringTransducer.Copy());
            return StringDistribution.FromWeightFunction(transducer.ProjectSource(concat.ToAutomaton()));
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/message_doc[@name="LogAverageFactor(StringDistribution, StringDistribution, StringDistribution)"]/*'/>
        public static double LogAverageFactor(StringDistribution concat, StringDistribution str1, StringDistribution str2)
        {
            Argument.CheckIfNotNull(concat, "concat");
            
            StringDistribution messageToConcat = ConcatAverageConditional(str1, str2);
            return messageToConcat.GetLogAverageOf(concat);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/message_doc[@name="LogEvidenceRatio(StringDistribution, StringDistribution, StringDistribution)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution concat, StringDistribution str1, StringDistribution str2)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="StringConcatOp"]/message_doc[@name="LogEvidenceRatio(String, StringDistribution, StringDistribution)"]/*'/>
        public static double LogEvidenceRatio(string concat, StringDistribution str1, StringDistribution str2)
        {
            Argument.CheckIfNotNull(concat, "concat");
            
            return LogAverageFactor(StringDistribution.String(concat), str1, str2);
        }

        #endregion
    }
}
