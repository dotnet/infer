// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Not")]
    [Quality(QualityBand.Mature)]
    public static class BooleanNotOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogAverageFactor(bool, bool)"]/*'/>
        public static double LogAverageFactor(bool not, bool b)
        {
            return (not == Factor.Not(b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogEvidenceRatio(bool, bool)"]/*'/>
        public static double LogEvidenceRatio(bool not, bool b)
        {
            return LogAverageFactor(not, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="AverageLogFactor(bool, bool)"]/*'/>
        public static double AverageLogFactor(bool not, bool b)
        {
            return LogAverageFactor(not, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="NotAverageConditional(Bernoulli)"]/*'/>
        public static Bernoulli NotAverageConditional([SkipIfUniform] Bernoulli b)
        {
            return Bernoulli.FromLogOdds(-b.LogOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="BAverageConditional(Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli not)
        {
            return Bernoulli.FromLogOdds(-not.LogOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="BAverageConditional(bool)"]/*'/>
        public static Bernoulli BAverageConditional(bool not)
        {
            return Bernoulli.PointMass(!not);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli not, Bernoulli b, [Fresh] Bernoulli to_not)
        {
            return to_not.GetLogAverageOf(not);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogAverageFactor(Bernoulli, bool)"]/*'/>
        public static double LogAverageFactor(Bernoulli not, bool b)
        {
            return LogAverageFactor(b, not);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool not, Bernoulli b)
        {
            return not ? b.GetLogProbFalse() : b.GetLogProbTrue();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli not)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool not, Bernoulli b)
        {
            return LogAverageFactor(not, b);
        }


        //- VMP ---------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="NotAverageLogarithm(Bernoulli)"]/*'/>
        public static Bernoulli NotAverageLogarithm([SkipIfUniform] Bernoulli b)
        {
            return NotAverageConditional(b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="BAverageLogarithm(Bernoulli)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli not)
        {
            return BAverageConditional(not);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="BAverageLogarithm(bool)"]/*'/>
        public static Bernoulli BAverageLogarithm(bool not)
        {
            return BAverageConditional(not);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanNotOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }
    }
}
