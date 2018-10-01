// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "AllTrue")]
    [Quality(QualityBand.Mature)]
    public static class AllTrueOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="LogAverageFactor(bool, IList{bool})"]/*'/>
        public static double LogAverageFactor(bool allTrue, IList<bool> array)
        {
            return (allTrue == Factor.AllTrue(array)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="LogEvidenceRatio(bool, IList{bool})"]/*'/>
        public static double LogEvidenceRatio(bool allTrue, IList<bool> array)
        {
            return LogAverageFactor(allTrue, array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="AverageLogFactor(bool, IList{bool})"]/*'/>
        public static double AverageLogFactor(bool allTrue, IList<bool> array)
        {
            return LogAverageFactor(allTrue, array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Bernoulli allTrue, [Fresh] Bernoulli to_allTrue)
        {
            return to_allTrue.GetLogAverageOf(allTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="LogAverageFactor(bool, IList{Bernoulli})"]/*'/>
        public static double LogAverageFactor(bool allTrue, [SkipIfAnyUniform] IList<Bernoulli> array)
        {
            Bernoulli to_allTrue = AllTrueAverageConditional(array);
            return to_allTrue.GetLogProb(allTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli allTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="LogEvidenceRatio(bool, IList{Bernoulli})"]/*'/>
        public static double LogEvidenceRatio(bool allTrue, [SkipIfAnyUniform] IList<Bernoulli> array)
        {
            return LogAverageFactor(allTrue, array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="AllTrueAverageConditional(IList{Bernoulli})"]/*'/>
        public static Bernoulli AllTrueAverageConditional(IList<Bernoulli> array)
        {
            double logOdds = Double.NegativeInfinity;
            for (int i = 0; i < array.Count; i++)
            {
                logOdds = Bernoulli.Or(logOdds, -array[i].LogOdds);
            }
            return Bernoulli.FromLogOdds(-logOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="AllTrueAverageConditional(IList{bool})"]/*'/>
        public static Bernoulli AllTrueAverageConditional(IList<bool> array)
        {
            foreach (bool b in array)
                if (!b)
                    return Bernoulli.PointMass(false);
            return Bernoulli.PointMass(true);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="ArrayAverageConditional{BernoulliList}(Bernoulli, IList{Bernoulli}, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the resulting array.</typeparam>
        public static BernoulliList ArrayAverageConditional<BernoulliList>(
            [SkipIfUniform] Bernoulli allTrue, IList<Bernoulli> array, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            if (result.Count == 0)
            {
            }
            else if (result.Count == 1)
                result[0] = allTrue;
            else if (result.Count == 2)
            {
                result[0] = BooleanAndOp.AAverageConditional(allTrue, array[1]);
                result[1] = BooleanAndOp.BAverageConditional(allTrue, array[0]);
            }
            else
            {
                // result.Count >= 3
                double notallTruePrevious = Double.NegativeInfinity;
                double[] notallTrueNext = new double[result.Count];
                notallTrueNext[notallTrueNext.Length - 1] = Double.NegativeInfinity;
                for (int i = notallTrueNext.Length - 2; i >= 0; i--)
                {
                    notallTrueNext[i] = Bernoulli.Or(-array[i + 1].LogOdds, notallTrueNext[i + 1]);
                }
                for (int i = 0; i < result.Count; i++)
                {
                    double notallTrueExcept = Bernoulli.Or(notallTruePrevious, notallTrueNext[i]);
                    result[i] = Bernoulli.FromLogOdds(-Bernoulli.Gate(-allTrue.LogOdds, notallTrueExcept));
                    notallTruePrevious = Bernoulli.Or(notallTruePrevious, -array[i].LogOdds);
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="ArrayAverageConditional{BernoulliList}(bool, IList{Bernoulli}, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the resulting array.</typeparam>
        public static BernoulliList ArrayAverageConditional<BernoulliList>(bool allTrue, IList<Bernoulli> array, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            return ArrayAverageConditional(Bernoulli.PointMass(allTrue), array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="ArrayAverageConditional{BernoulliList}(Bernoulli, IList{bool}, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the resulting array.</typeparam>
        public static BernoulliList ArrayAverageConditional<BernoulliList>([SkipIfUniform] Bernoulli allTrue, IList<bool> array, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            if (result.Count == 0)
            {
            }
            else if (result.Count == 1)
                result[0] = allTrue;
            else if (result.Count == 2)
            {
                result[0] = BooleanAndOp.AAverageConditional(allTrue, array[1]);
                result[1] = BooleanAndOp.BAverageConditional(allTrue, array[0]);
            }
            else
            {
                // result.Count >= 3
                int trueCount = 0;
                int firstFalseIndex = -1;
                for (int i = 0; i < array.Count; i++)
                {
                    if (array[i])
                        trueCount++;
                    else if (firstFalseIndex < 0)
                        firstFalseIndex = i;
                }
                if (trueCount == array.Count)
                {
                    for (int i = 0; i < result.Count; i++)
                        result[i] = BooleanAndOp.AAverageConditional(allTrue, true);
                }
                else
                {
                    for (int i = 0; i < result.Count; i++)
                        result[i] = BooleanAndOp.AAverageConditional(allTrue, false);
                    if (trueCount == result.Count - 1)
                        result[firstFalseIndex] = BooleanAndOp.AAverageConditional(allTrue, true);
                }
            }
            return result;
        }

        // VMP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        private const string NotSupportedMessage = "Variational Message Passing does not support an AllTrue factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="AllTrueAverageLogarithm(IList{Bernoulli})"]/*'/>
        public static Bernoulli AllTrueAverageLogarithm(IList<Bernoulli> array)
        {
            return AllTrueAverageConditional(array);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="ArrayAverageLogarithm{BernoulliList}(Bernoulli, IList{Bernoulli}, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the resulting array.</typeparam>
        public static BernoulliList ArrayAverageLogarithm<BernoulliList>([SkipIfUniform] Bernoulli allTrue, IList<Bernoulli> array, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            // when 'allTrue' is marginalized, the factor is proportional to exp(allTrue.LogOdds*prod_i a[i])
            // therefore we maintain the value of prod_i E(a[i]) as we update each a[i]'s distribution
            double prodProbTrue = 1.0;
            for (int i = 0; i < array.Count; i++)
            {
                prodProbTrue *= array[i].GetProbTrue();
            }
            for (int i = 0; i < result.Count; i++)
            {
                prodProbTrue /= array[i].GetProbTrue();
                Bernoulli ratio = array[i] / result[i];
                result[i] = Bernoulli.FromLogOdds(allTrue.LogOdds * prodProbTrue);
                Bernoulli newMarginal = ratio * result[i];
                prodProbTrue *= newMarginal.GetProbTrue();
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="ArrayAverageLogarithm{BernoulliList}(bool, IList{Bernoulli}, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the resulting array.</typeparam>
        [NotSupported(AllTrueOp.NotSupportedMessage)]
        public static BernoulliList ArrayAverageLogarithm<BernoulliList>(bool allTrue, [Stochastic] IList<Bernoulli> array, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            throw new NotSupportedException(NotSupportedMessage);
            //return ArrayAverageLogarithm(Bernoulli.PointMass(allTrue), array, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="AllTrueOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }
    }
}
