// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Distributions;
    using Distributions.Automata;
    using Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Substring")]
    [Quality(QualityBand.Experimental)]
    public static class SubstringOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="SubAverageConditional(StringDistribution, int, int)"]/*'/>
        public static StringDistribution SubAverageConditional(StringDistribution str, int start, int length)
        {
            return SubAverageConditional(str, start, length, length);
        }

        public static StringDistribution SubAverageConditional(StringDistribution str, int start, int minLength, int maxLength)
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(minLength >= 0, "minLength", "Min length must be non-negative.");
            Argument.CheckIfInRange(maxLength >= 0, "maxLength", "Max length must be non-negative.");

            if (str.IsPointMass)
            {
                var strPoint = str.Point;
                var alts = new HashSet<string>();
                for (int length = minLength; length <= maxLength; length++)
                {
                    var s = strPoint.Substring(start, Math.Min(length, strPoint.Length));
                    alts.Add(s);
                }
                return StringDistribution.OneOf(alts);
            }

            var anyChar = StringAutomaton.ConstantOnElement(1.0, ImmutableDiscreteChar.Any());
            var transducer = StringTransducer.Consume(StringAutomaton.Repeat(anyChar, minTimes: start, maxTimes: start));
            transducer.AppendInPlace(StringTransducer.Copy(StringAutomaton.Repeat(anyChar, minTimes: minLength, maxTimes: maxLength)));
            transducer.AppendInPlace(StringTransducer.Consume(StringAutomaton.Constant(1.0)));

            return StringDistribution.FromWeightFunction(transducer.ProjectSource(str.ToAutomaton()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="SubAverageConditional(String, int, int)"]/*'/>
        public static StringDistribution SubAverageConditional(string str, int start, int length)
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");

            if (start + length > str.Length)
            {
                return StringDistribution.Zero();
            }
            
            return StringDistribution.String(str.Substring(start, length));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="StrAverageConditional(String, int, int)"]/*'/>
        public static StringDistribution StrAverageConditional(string sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");
            
            if (sub.Length != length)
            {
                return StringDistribution.Zero();
            }

            StringDistribution result = StringDistribution.Any(minLength: start, maxLength: start);
            result.AppendInPlace(sub);
            result.AppendInPlace(StringDistribution.Any());
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="StrAverageConditional(StringDistribution, int, int)"]/*'/>
        public static StringDistribution StrAverageConditional(StringDistribution sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");

            var anyChar = StringAutomaton.ConstantOnElement(1.0, ImmutableDiscreteChar.Any());
            var transducer = StringTransducer.Produce(StringAutomaton.Repeat(anyChar, minTimes: start, maxTimes: start));
            transducer.AppendInPlace(StringTransducer.Copy(StringAutomaton.Repeat(anyChar, minTimes: length, maxTimes: length)));
            transducer.AppendInPlace(StringTransducer.Produce(StringAutomaton.Constant(1.0)));

            return StringDistribution.FromWeightFunction(transducer.ProjectSource(sub.ToAutomaton()));
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="LogAverageFactor(StringDistribution, StringDistribution, int, int)"]/*'/>
        public static double LogAverageFactor(StringDistribution str, StringDistribution sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            
            StringDistribution messageToSub = SubAverageConditional(str, start, length);
            return messageToSub.GetLogAverageOf(sub);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="LogEvidenceRatio(StringDistribution, StringDistribution, int, int)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution str, StringDistribution sub, int start, int length)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SubstringOp"]/message_doc[@name="LogEvidenceRatio(StringDistribution, String, int, int)"]/*'/>
        public static double LogEvidenceRatio(StringDistribution str, string sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            
            return LogAverageFactor(str, StringDistribution.String(sub), start, length);
        }

        #endregion
    }
}
