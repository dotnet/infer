// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/doc/*'/>
    /// </summary>
    [FactorMethod(typeof(SparseBernoulliList), "Sample", typeof(ISparseList<double>))]
    [Quality(QualityBand.Stable)]
    public class SparseBernoulliFromBetaOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogAverageFactor(ISparseList{bool}, ISparseList{double})"]/*'/>
        public static double LogAverageFactor(ISparseList<bool> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogAverageFactor(SparseBernoulliList, SparseBernoulliList)"]/*'/>
        public static double LogAverageFactor(SparseBernoulliList sample, [Fresh] SparseBernoulliList to_sample)
        {
            Func<Bernoulli, Bernoulli, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample, to_sample).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogAverageFactor(ISparseList{bool}, SparseBetaList)"]/*'/>
        public static double LogAverageFactor(ISparseList<bool> sample, SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(SparseBernoulliList, ISparseList{double})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(SparseBernoulliList sample, ISparseList<double> probTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(ISparseList{bool}, SparseBetaList)"]/*'/>
        public static double LogEvidenceRatio(ISparseList<bool> sample, SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.LogEvidenceRatio;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(ISparseList{bool}, ISparseList{double})"]/*'/>
        public static double LogEvidenceRatio(ISparseList<bool> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.LogEvidenceRatio;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(SparseBernoulliList, SparseBetaList)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(SparseBernoulliList sample, SparseBetaList probTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="SampleConditional(ISparseList{double}, SparseBernoulliList)"]/*'/>
        public static SparseBernoulliList SampleConditional(ISparseList<double> probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleConditional(pt));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="SampleAverageConditional(ISparseList{double}, SparseBernoulliList)"]/*'/>
        public static SparseBernoulliList SampleAverageConditional(ISparseList<double> probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageConditional(pt));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="ProbTrueConditional(ISparseList{bool}, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueConditional(ISparseList<bool> sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueConditional(s));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageConditional(ISparseList{bool}, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageConditional(ISparseList<bool> sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueAverageConditional(s));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="SampleAverageConditional(SparseBetaList, SparseBernoulliList)"]/*'/>
        public static SparseBernoulliList SampleAverageConditional([SkipIfUniform] SparseBetaList probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageConditional(pt));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageConditional(SparseBernoulliList, SparseBetaList, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageConditional([SkipIfUniform] SparseBernoulliList sample, SparseBetaList probTrue, SparseBetaList result)
        {
            result.SetToFunction(sample, probTrue, (s, pt) => BernoulliFromBetaOp.ProbTrueAverageConditional(s, pt));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(SparseBernoulliList, SparseBetaList)"]/*'/>
        public static double AverageLogFactor(SparseBernoulliList sample, [Proper] SparseBetaList probTrue)
        {
            Func<Bernoulli, Beta, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(SparseBernoulliList, ISparseList{double})"]/*'/>
        public static double AverageLogFactor(SparseBernoulliList sample, ISparseList<double> probTrue)
        {
            Func<Bernoulli, double, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(ISparseList{bool}, SparseBetaList)"]/*'/>
        public static double AverageLogFactor(ISparseList<bool> sample, [Proper] SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(ISparseList{bool}, ISparseList{double})"]/*'/>
        public static double AverageLogFactor(ISparseList<bool> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="SampleAverageLogarithm(ISparseList{double}, SparseBernoulliList)"]/*'/>
        public static SparseBernoulliList SampleAverageLogarithm(ISparseList<double> probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageLogarithm(pt));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="SampleAverageLogarithm(SparseBetaList, SparseBernoulliList)"]/*'/>
        public static SparseBernoulliList SampleAverageLogarithm([SkipIfUniform] SparseBetaList probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageLogarithm(pt));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageLogarithm(ISparseList{bool}, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageLogarithm(ISparseList<bool> sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueAverageLogarithm(s));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="SparseBernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageLogarithm(SparseBernoulliList, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageLogarithm(SparseBernoulliList sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueAverageLogarithm(s));
            return result;
        }
    }
}
