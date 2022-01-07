// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/doc/*'/>
    /// <remarks>This factor is symmetric among all three arguments.</remarks>
    [FactorMethod(typeof(Factor), "AreEqual", typeof(bool), typeof(bool))]
    [Quality(QualityBand.Mature)]
    public static class BooleanAreEqualOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, bool, bool)"]/*'/>
        public static double LogAverageFactor(bool areEqual, bool a, bool b)
        {
            return (areEqual == Factor.AreEqual(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, bool, bool)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, bool a, bool b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AverageLogFactor(bool, bool, bool)"]/*'/>
        public static double AverageLogFactor(bool areEqual, bool a, bool b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogAverageFactor(Bernoulli, bool, bool)"]/*'/>
        public static double LogAverageFactor(Bernoulli areEqual, bool a, bool b)
        {
            return areEqual.GetLogProb(Factor.AreEqual(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AreEqualAverageConditional([SkipIfUniform] Bernoulli A, [SkipIfUniform] Bernoulli B)
        {
            return Bernoulli.FromLogOdds(Bernoulli.LogitProbEqual(A.LogOdds, B.LogOdds));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(bool A, [SkipIfUniform] Bernoulli B)
        {
            return Bernoulli.FromLogOdds(A ? B.LogOdds : -B.LogOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli AreEqualAverageConditional([SkipIfUniform] Bernoulli A, bool B)
        {
            return AreEqualAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(bool A, bool B)
        {
            return Bernoulli.PointMass(Factor.AreEqual(A, B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli AAverageConditional(bool areEqual, [SkipIfUniform] Bernoulli B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli areEqual, bool B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli AAverageConditional(bool areEqual, bool B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageConditional(bool, Bernoulli)"]/*'/>
        public static Bernoulli BAverageConditional(bool areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageConditional(Bernoulli, bool)"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli areEqual, bool A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageConditional(bool, bool)"]/*'/>
        public static Bernoulli BAverageConditional(bool areEqual, bool A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli areEqual, [Fresh] Bernoulli to_areEqual)
        {
            return to_areEqual.GetLogAverageOf(areEqual);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli, Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool areEqual, Bernoulli A, Bernoulli B, [Fresh] Bernoulli to_A)
        {
            //Bernoulli to_A = AAverageConditional(areEqual, B);
            return A.GetLogAverageOf(to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, Bernoulli, bool, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool areEqual, Bernoulli A, bool B, [Fresh] Bernoulli to_A)
        {
            //Bernoulli toA = AAverageConditional(areEqual, B);
            return A.GetLogAverageOf(to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, bool, Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(bool areEqual, bool A, Bernoulli B, [Fresh] Bernoulli to_B)
        {
            return LogAverageFactor(areEqual, B, A, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli, Bernoulli, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, Bernoulli A, Bernoulli B, [Fresh] Bernoulli to_A)
        {
            return LogAverageFactor(areEqual, A, B, to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, bool, Bernoulli, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, bool A, Bernoulli B, [Fresh] Bernoulli to_B)
        {
            return LogAverageFactor(areEqual, A, B, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, Bernoulli, bool, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, Bernoulli A, bool B, [Fresh] Bernoulli to_A)
        {
            return LogEvidenceRatio(areEqual, B, A, to_A);
        }

        //- VMP -------------------------------------------------------------------------------------------------------------

        private const string NotSupportedMessage = "Variational Message Passing does not support an AreEqual factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm([SkipIfUniform] Bernoulli A, [SkipIfUniform] Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageLogarithm(bool, Bernoulli)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(bool A, [SkipIfUniform] Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm([SkipIfUniform] Bernoulli A, bool B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AreEqualAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(bool A, bool B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli B)
        {
            if (areEqual.IsPointMass)
                return AAverageLogarithm(areEqual.Point, B);
            // when AreEqual is marginalized, the factor is proportional to exp((A==B)*areEqual.LogOdds)
            return Bernoulli.FromLogOdds(areEqual.LogOdds * (2 * B.GetProbTrue() - 1));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, bool B)
        {
            return Bernoulli.FromLogOdds(B ? areEqual.LogOdds : -areEqual.LogOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageLogarithm(bool, Bernoulli)"]/*'/>
        [NotSupported(BooleanAreEqualOp.NotSupportedMessage)]
        public static Bernoulli AAverageLogarithm(bool areEqual, [SkipIfUniform] Bernoulli B)
        {
            if (B.IsPointMass)
                return AAverageLogarithm(areEqual, B.Point);
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="AAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli AAverageLogarithm(bool areEqual, bool B)
        {
            return AAverageConditional(areEqual, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageLogarithm(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, bool)"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, bool A)
        {
            return AAverageLogarithm(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageLogarithm(bool, Bernoulli)"]/*'/>
        [NotSupported(BooleanAreEqualOp.NotSupportedMessage)]
        public static Bernoulli BAverageLogarithm(bool areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageLogarithm(areEqual, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BooleanAreEqualOp"]/message_doc[@name="BAverageLogarithm(bool, bool)"]/*'/>
        public static Bernoulli BAverageLogarithm(bool areEqual, bool A)
        {
            return AAverageLogarithm(areEqual, A);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "AreEqual", typeof(int), typeof(int))]
    [Quality(QualityBand.Mature)]
    public static class DiscreteAreEqualOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, int, int)"]/*'/>
        public static double LogAverageFactor(bool areEqual, int a, int b)
        {
            return (areEqual == Factor.AreEqual(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, int, int)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, int a, int b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AverageLogFactor(bool, int, int)"]/*'/>
        public static double AverageLogFactor(bool areEqual, int a, int b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(Bernoulli, int, int)"]/*'/>
        public static double LogAverageFactor(Bernoulli areEqual, int a, int b)
        {
            return areEqual.GetLogProb(Factor.AreEqual(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(Discrete, Discrete)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(Discrete A, Discrete B)
        {
            return Bernoulli.FromLogOdds(MMath.Logit(A.ProbEqual(B)));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(int, Discrete)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(int A, Discrete B)
        {
            return Bernoulli.FromLogOdds(MMath.Logit(B[A]));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AreEqualAverageConditional(Discrete, int)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(Discrete A, int B)
        {
            return AreEqualAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageConditional(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageConditional([SkipIfUniform] Bernoulli areEqual, Discrete B, Discrete result)
        {
            if (areEqual.IsPointMass)
                return AAverageConditional(areEqual.Point, B, result);
            if (result == default(Discrete))
                result = Distributions.Discrete.Uniform(B.Dimension, B.Sparsity);
            double p = areEqual.GetProbTrue();
            Vector probs = result.GetWorkspace();
            probs = B.GetProbs(probs);
            probs.SetToProduct(probs, 2.0 * p - 1.0);
            probs.SetToSum(probs, 1.0 - p);
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageConditional(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete AAverageConditional([SkipIfUniform] Bernoulli areEqual, int B, Discrete result)
        {
            if (areEqual.IsPointMass)
                return AAverageConditional(areEqual.Point, B, result);
            Vector probs = result.GetWorkspace();
            double p = areEqual.GetProbTrue();
            probs.SetAllElementsTo(1 - p);
            probs[B] = p;
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageConditional(bool, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageConditional(bool areEqual, Discrete B, Discrete result)
        {
            if (B.IsPointMass)
                return AAverageConditional(areEqual, B.Point, result);
            if (result == default(Discrete))
                result = Distributions.Discrete.Uniform(B.Dimension, B.Sparsity);
            if (areEqual)
                result.SetTo(B);
            else
            {
                Vector probs = result.GetWorkspace();
                probs = B.GetProbs(probs);
                probs.SetToDifference(1.0, probs);
                result.SetProbs(probs);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageConditional(bool, int, Discrete)"]/*'/>
        public static Discrete AAverageConditional(bool areEqual, int B, Discrete result)
        {
            if (areEqual)
                result.Point = B;
            else if (result.Dimension == 2)
                result.Point = 1 - B;
            else
            {
                Vector probs = result.GetWorkspace();
                probs.SetAllElementsTo(1);
                probs[B] = 0;
                result.SetProbs(probs);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageConditional(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageConditional([SkipIfUniform] Bernoulli areEqual, Discrete A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageConditional(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete BAverageConditional([SkipIfUniform] Bernoulli areEqual, int A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageConditional(bool, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageConditional(bool areEqual, Discrete A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageConditional(bool, int, Discrete)"]/*'/>
        public static Discrete BAverageConditional(bool areEqual, int A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli areEqual, [Fresh] Bernoulli to_areEqual)
        {
            return to_areEqual.GetLogAverageOf(areEqual);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(bool areEqual, Discrete A, Discrete B)
        {
            Bernoulli to_areEqual = AreEqualAverageConditional(A, B);
            return to_areEqual.GetLogProb(areEqual);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(bool areEqual, Discrete A, Discrete B, [Fresh] Discrete to_A)
        {
            return A.GetLogAverageOf(to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, int, Discrete)"]/*'/>
        public static double LogAverageFactor(bool areEqual, int A, Discrete B)
        {
            Bernoulli to_areEqual = AreEqualAverageConditional(A, B);
            return to_areEqual.GetLogProb(areEqual);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, int, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(bool areEqual, int A, Discrete B, [Fresh] Discrete to_B)
        {
            return B.GetLogAverageOf(to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, int)"]/*'/>
        public static double LogAverageFactor(bool areEqual, Discrete A, int B)
        {
            Bernoulli to_areEqual = AreEqualAverageConditional(A, B);
            return to_areEqual.GetLogProb(areEqual);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, int, Discrete)"]/*'/>
        public static double LogAverageFactor(bool areEqual, Discrete A, int B, [Fresh] Discrete to_A)
        {
            return A.GetLogAverageOf(to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0.0;
        }

        //public static double LogEvidenceRatio(bool areEqual, Discrete A, Discrete B) { return LogAverageFactor(areEqual, A, B); }
        //public static double LogEvidenceRatio(bool areEqual, int A, Discrete B) { return LogAverageFactor(areEqual, A, B); }
        //public static double LogEvidenceRatio(bool areEqual, Discrete A, int B) { return LogAverageFactor(areEqual, A, B); }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, Discrete, Discrete, Discrete)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, Discrete A, Discrete B, [Fresh] Discrete to_A)
        {
            return LogAverageFactor(areEqual, A, B, to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, int, Discrete, Discrete)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, int A, Discrete B, [Fresh] Discrete to_B)
        {
            return LogAverageFactor(areEqual, A, B, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="LogEvidenceRatio(bool, Discrete, int, Discrete)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, Discrete A, int B, [Fresh] Discrete to_A)
        {
            return LogAverageFactor(areEqual, A, B, to_A);
        }

        //-- VMP ----------------------------------------------------------------------------------------

        private const string NotSupportedMessage = "Variational Message Passing does not support an AreEqual factor with fixed output.";

        /// <summary>
        /// Evidence message for VMP.
        /// </summary>
        /// <returns>Zero</returns>
        /// <remarks><para>
        /// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
        /// </para></remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(Discrete A, Discrete B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AreEqualAverageLogarithm(Discrete, Discrete)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(int A, Discrete B)
        {
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AreEqualAverageLogarithm(int, Discrete)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(Discrete A, int B)
        {
            return AreEqualAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, Discrete B, Discrete result)
        {
            if (areEqual.IsPointMass)
                return AAverageLogarithm(areEqual.Point, B, result);
            if (result == default(Discrete))
                result = Discrete.Uniform(B.Dimension, B.Sparsity);
            // when AreEqual is marginalized, the factor is proportional to exp((A==B)*areEqual.LogOdds)
            Vector probs = result.GetWorkspace();
            probs = B.GetProbs(probs);
            probs.SetToFunction(probs, x => Math.Exp(x * areEqual.LogOdds));
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, int B, Discrete result)
        {
            return AAverageConditional(areEqual, B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageLogarithm(bool, Discrete, Discrete)"]/*'/>
        [NotSupported(DiscreteAreEqualOp.NotSupportedMessage)]
        public static Discrete AAverageLogarithm(bool areEqual, Discrete B, Discrete result)
        {
            if (B.IsPointMass)
                return AAverageLogarithm(areEqual, B.Point, result);
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="AAverageLogarithm(bool, int, Discrete)"]/*'/>
        public static Discrete AAverageLogarithm(bool areEqual, int B, Discrete result)
        {
            return AAverageConditional(areEqual, B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, Discrete A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, int A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageLogarithm(bool, Discrete, Discrete)"]/*'/>
        [NotSupported(DiscreteAreEqualOp.NotSupportedMessage)]
        public static Discrete BAverageLogarithm(bool areEqual, Discrete A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteAreEqualOp"]/message_doc[@name="BAverageLogarithm(bool, int, Discrete)"]/*'/>
        public static Discrete BAverageLogarithm(bool areEqual, int A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }
    }
}
