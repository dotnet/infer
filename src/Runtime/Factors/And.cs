// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "And")]
    [Quality(QualityBand.Mature)]
    public static class BooleanAndOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(bool, bool, bool)"]/*'/>
        public static double LogAverageFactor(bool and, bool a, bool b)
        {
            return (and == Factor.And(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogEvidenceRatio(bool, bool, bool)"]/*'/>
        public static double LogEvidenceRatio(bool and, bool a, bool b)
        {
            return LogAverageFactor(and, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AverageLogFactor(bool, bool, bool)"]/*'/>
        public static double AverageLogFactor(bool and, bool a, bool b)
        {
            return LogAverageFactor(and, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(Bernoulli, bool, bool)"]/*'/>
        public static double LogAverageFactor(Bernoulli and, bool a, bool b)
        {
            return and.GetLogProb(Factor.And(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AndAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AndAverageConditional(Bernoulli A, Bernoulli B)
        {
            return Bernoulli.FromLogOdds(-Bernoulli.Or(-A.LogOdds, -B.LogOdds));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AndAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli AndAverageConditional(bool A, Bernoulli B)
        {
            if (A)
                return B;
            else
                return Bernoulli.PointMass(false);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AndAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli AndAverageConditional(Bernoulli A, bool B)
        {
            return AndAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli and, Bernoulli B)
        {
            if (B.IsPointMass)
                return AAverageConditional(and, B.Point);
            return Bernoulli.FromLogOdds(-Bernoulli.Gate(-and.LogOdds, -B.LogOdds));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli AAverageConditional(bool and, Bernoulli B)
        {
            if (B.IsPointMass)
                return AAverageConditional(and, B.Point);
            return AAverageConditional(Bernoulli.PointMass(and), B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli and, bool B)
        {
            if (and.IsPointMass)
                return AAverageConditional(and.Point, B);
            return Bernoulli.FromLogOdds(B ? and.LogOdds : 0);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli AAverageConditional(bool and, bool B)
        {
            if (B)
                return Bernoulli.PointMass(and);
            else if (!and)
                return Bernoulli.Uniform();
            else
                throw new AllZeroException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli and, Bernoulli A)
        {
            return AAverageConditional(and, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional(bool and, Bernoulli A)
        {
            return AAverageConditional(and, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli and, bool A)
        {
            return AAverageConditional(and, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli BAverageConditional(bool and, bool A)
        {
            return AAverageConditional(and, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli and, Bernoulli a, Bernoulli b)
        {
            Bernoulli to_and = AndAverageConditional(a, b);
            return to_and.GetLogAverageOf(and);
        }

#pragma warning disable 1591
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(Bernoulli, bool, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli and, bool a, Bernoulli b)
        {
            Bernoulli to_and = AndAverageConditional(a, b);
            return to_and.GetLogAverageOf(and);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli, bool)"]/*'/>
        public static double LogAverageFactor(Bernoulli and, Bernoulli a, bool b)
        {
            Bernoulli to_and = AndAverageConditional(a, b);
            return to_and.GetLogAverageOf(and);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool and, Bernoulli a, Bernoulli b)
        {
            Bernoulli to_and = AndAverageConditional(a, b);
            return to_and.GetLogProb(and);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli, bool)"]/*'/>
        public static double LogAverageFactor(bool and, Bernoulli a, bool b)
        {
            Bernoulli to_and = AndAverageConditional(a, b);
            return to_and.GetLogProb(and);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogAverageFactor(bool, bool, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool and, bool a, Bernoulli b)
        {
            return LogAverageFactor(and, b, a);
        }
#pragma warning restore 1591

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli and)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool and, Bernoulli a, Bernoulli b)
        {
            return LogAverageFactor(and, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli, bool)"]/*'/>
        public static double LogEvidenceRatio(bool and, Bernoulli a, bool b)
        {
            return LogAverageFactor(and, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="LogEvidenceRatio(bool, bool, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool and, bool a, Bernoulli b)
        {
            return LogEvidenceRatio(and, b, a);
        }

        //-- VMP ---------------------------------------------------------------------------------------------

        private const string NotSupportedMessage = "Variational Message Passing does not support an And factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AndAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AndAverageLogarithm(Bernoulli A, Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return AndAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AndAverageLogarithm(bool, Bernoulli)"]/*'/>
        public static Bernoulli AndAverageLogarithm(bool A, Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return AndAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AndAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli AndAverageLogarithm(Bernoulli A, bool B)
        {
            // same as BP if you use John Winn's rule.
            return AndAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli and, Bernoulli B)
        {
            // when 'and' is marginalized, the factor is proportional to exp(A*B*and.LogOdds)
            return Bernoulli.FromLogOdds(and.LogOdds * B.GetProbTrue());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli and, bool B)
        {
            return Bernoulli.FromLogOdds(B ? and.LogOdds : 0.0);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageLogarithm(bool, Bernoulli)"]/*'/>
        [NotSupported(BooleanAndOp.NotSupportedMessage)]
        public static Bernoulli AAverageLogarithm(bool and, Bernoulli B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="AAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli AAverageLogarithm(bool and, bool B)
        {
            return AAverageConditional(and, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli and, Bernoulli A)
        {
            return AAverageLogarithm(and, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli and, bool A)
        {
            return AAverageLogarithm(and, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageLogarithm(bool, Bernoulli)"]/*'/>
        [NotSupported(BooleanAndOp.NotSupportedMessage)]
        public static Bernoulli BAverageLogarithm(bool and, Bernoulli A)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAndOp"]/message_doc[@name="BAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli BAverageLogarithm(bool and, bool A)
        {
            return AAverageLogarithm(and, A);
        }
    }
}
