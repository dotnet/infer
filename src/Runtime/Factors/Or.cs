// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Or")]
    [Quality(QualityBand.Mature)]
    public static class BooleanOrOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogAverageFactor(bool, bool, bool)"]/*'/>
        public static double LogAverageFactor(bool or, bool a, bool b)
        {
            return (or == Factor.Or(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogEvidenceRatio(bool, bool, bool)"]/*'/>
        public static double LogEvidenceRatio(bool or, bool a, bool b)
        {
            return LogAverageFactor(or, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AverageLogFactor(bool, bool, bool)"]/*'/>
        public static double AverageLogFactor(bool or, bool a, bool b)
        {
            return LogAverageFactor(or, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogAverageFactor(Bernoulli, bool, bool)"]/*'/>
        public static double LogAverageFactor(Bernoulli or, bool a, bool b)
        {
            return or.GetLogProb(Factor.Or(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="OrAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli OrAverageConditional(Bernoulli A, Bernoulli B)
        {
            return Bernoulli.FromLogOdds(Bernoulli.Or(A.LogOdds, B.LogOdds));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="OrAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli OrAverageConditional(bool A, Bernoulli B)
        {
            if (A)
                return Bernoulli.PointMass(true);
            else
                return B;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="OrAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli OrAverageConditional(Bernoulli A, bool B)
        {
            return OrAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli or, Bernoulli B)
        {
            if (B.IsPointMass)
                return AAverageConditional(or, B.Point);
            return Bernoulli.FromLogOdds(Bernoulli.Gate(or.LogOdds, B.LogOdds));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli AAverageConditional(bool or, Bernoulli B)
        {
            if (B.IsPointMass)
                return AAverageConditional(or, B.Point);
            return AAverageConditional(Bernoulli.PointMass(or), B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli or, bool B)
        {
            if (or.IsPointMass)
                return AAverageConditional(or.Point, B);
            return Bernoulli.FromLogOdds(B ? 0 : or.LogOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli AAverageConditional(bool or, bool B)
        {
            if (!B)
                return Bernoulli.PointMass(or);
            else if (or)
                return Bernoulli.Uniform();
            else
                throw new AllZeroException();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli or, Bernoulli A)
        {
            return AAverageConditional(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli or, bool A)
        {
            return AAverageConditional(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional(bool or, Bernoulli A)
        {
            return AAverageConditional(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli BAverageConditional(bool or, bool A)
        {
            return AAverageConditional(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli or, [Fresh] Bernoulli to_or)
        {
            return to_or.GetLogAverageOf(or);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool or, Bernoulli a, Bernoulli b)
        {
            Bernoulli to_or = OrAverageConditional(a, b);
            return to_or.GetLogProb(or);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli, bool)"]/*'/>
        public static double LogAverageFactor(bool or, Bernoulli a, bool b)
        {
            Bernoulli to_or = OrAverageConditional(a, b);
            return to_or.GetLogProb(or);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogAverageFactor(bool, bool, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool or, bool a, Bernoulli b)
        {
            return LogAverageFactor(or, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli or)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool or, Bernoulli a, Bernoulli b)
        {
            return LogAverageFactor(or, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli, bool)"]/*'/>
        public static double LogEvidenceRatio(bool or, Bernoulli a, bool b)
        {
            return LogAverageFactor(or, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="LogEvidenceRatio(bool, bool, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool or, bool a, Bernoulli b)
        {
            return LogEvidenceRatio(or, b, a);
        }

        //-- VMP ---------------------------------------------------------------------------------------

        private const string NotSupportedMessage = "Variational Message Passing does not support an Or factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="OrAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli OrAverageLogarithm(Bernoulli A, Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return OrAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="OrAverageLogarithm(bool, Bernoulli)"]/*'/>
        public static Bernoulli OrAverageLogarithm(bool A, Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return OrAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="OrAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli OrAverageLogarithm(Bernoulli A, bool B)
        {
            // same as BP if you use John Winn's rule.
            return OrAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli or, Bernoulli B)
        {
            // when 'or' is marginalized, the factor is proportional to exp((A|B)*or.LogOdds)
            return Bernoulli.FromLogOdds(or.LogOdds * B.GetProbFalse());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli or, bool B)
        {
            return Bernoulli.FromLogOdds(B ? 0.0 : or.LogOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageLogarithm(bool, Bernoulli)"]/*'/>
        [NotSupported(BooleanOrOp.NotSupportedMessage)]
        public static Bernoulli AAverageLogarithm(bool or, Bernoulli B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="AAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli AAverageLogarithm(bool or, bool B)
        {
            return AAverageConditional(or, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli or, Bernoulli A)
        {
            return AAverageLogarithm(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli or, bool A)
        {
            return AAverageLogarithm(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageLogarithm(bool, Bernoulli)"]/*'/>
        public static Bernoulli BAverageLogarithm(bool or, Bernoulli A)
        {
            return AAverageLogarithm(or, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanOrOp"]/message_doc[@name="BAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli BAverageLogarithm(bool or, bool A)
        {
            return AAverageLogarithm(or, A);
        }
    }
}
