// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/doc/*'/>
    /// </summary>
    [FactorMethod(typeof(BernoulliIntegerSubset), "Sample", typeof(ISparseList<double>))]
    [Quality(QualityBand.Preview)]
    public class BernoulliIntegerSubsetFromBeta
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogAverageFactor(IList{int}, ISparseList{double})"]/*'/>
        public static double LogAverageFactor(IList<int> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(BernoulliIntegerSubset.SubsetToList(sample, probTrue.Count), probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogAverageFactor(BernoulliIntegerSubset, BernoulliIntegerSubset)"]/*'/>
        public static double LogAverageFactor(BernoulliIntegerSubset sample, [Fresh] BernoulliIntegerSubset to_sample)
        {
            Func<Bernoulli, Bernoulli, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample.SparseBernoulliList, to_sample.SparseBernoulliList).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogAverageFactor(IList{int}, SparseBetaList)"]/*'/>
        public static double LogAverageFactor(IList<int> sample, SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(BernoulliIntegerSubset.SubsetToList(sample, probTrue.Count), probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogEvidenceRatio(BernoulliIntegerSubset, ISparseList{double})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(BernoulliIntegerSubset sample, ISparseList<double> probTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogEvidenceRatio(IList{int}, SparseBetaList)"]/*'/>
        public static double LogEvidenceRatio(IList<int> sample, SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.LogEvidenceRatio;
            return f.Map(BernoulliIntegerSubset.SubsetToList(sample, probTrue.Count), probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogEvidenceRatio(IList{int}, ISparseList{double})"]/*'/>
        public static double LogEvidenceRatio(IList<int> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.LogEvidenceRatio;
            return f.Map(BernoulliIntegerSubset.SubsetToList(sample, probTrue.Count), probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="LogEvidenceRatio(BernoulliIntegerSubset, SparseBetaList)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(BernoulliIntegerSubset sample, SparseBetaList probTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="SampleConditional(ISparseList{double}, BernoulliIntegerSubset)"]/*'/>
        public static BernoulliIntegerSubset SampleConditional(ISparseList<double> probTrue, BernoulliIntegerSubset result)
        {
            Func<double, Bernoulli> f = BernoulliFromBetaOp.SampleConditional;
            result.SetTo(f.Map(probTrue));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="SampleAverageConditional(ISparseList{double}, BernoulliIntegerSubset)"]/*'/>
        public static BernoulliIntegerSubset SampleAverageConditional(ISparseList<double> probTrue, BernoulliIntegerSubset result)
        {
            Func<double, Bernoulli> f = BernoulliFromBetaOp.SampleAverageConditional;
            result.SetTo(f.Map(probTrue));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="ProbTrueConditional(IList{int}, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueConditional(IList<int> sample, SparseBetaList result)
        {
            result.SetToFunction(BernoulliIntegerSubset.SubsetToList(sample, result.Count), s => BernoulliFromBetaOp.ProbTrueConditional(s));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="ProbTrueAverageConditional(IList{int}, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageConditional(IList<int> sample, SparseBetaList result)
        {
            result.SetToFunction(BernoulliIntegerSubset.SubsetToList(sample, result.Count), s => BernoulliFromBetaOp.ProbTrueAverageConditional(s));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="SampleAverageConditional(SparseBetaList, BernoulliIntegerSubset)"]/*'/>
        public static BernoulliIntegerSubset SampleAverageConditional([SkipIfUniform] SparseBetaList probTrue, BernoulliIntegerSubset result)
        {
            Func<Beta, Bernoulli> f = BernoulliFromBetaOp.SampleAverageConditional;
            result.SetTo(f.Map(probTrue));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="ProbTrueAverageConditional(BernoulliIntegerSubset, SparseBetaList, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageConditional([SkipIfUniform] BernoulliIntegerSubset sample, SparseBetaList probTrue, SparseBetaList result)
        {
            Func<Bernoulli, Beta, Beta> f = BernoulliFromBetaOp.ProbTrueAverageConditional;
            result.SetTo(f.Map(sample.SparseBernoulliList, probTrue));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="AverageLogFactor(BernoulliIntegerSubset, SparseBetaList)"]/*'/>
        public static double AverageLogFactor(BernoulliIntegerSubset sample, [Proper] SparseBetaList probTrue)
        {
            Func<Bernoulli, Beta, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample.SparseBernoulliList, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="AverageLogFactor(BernoulliIntegerSubset, ISparseList{double})"]/*'/>
        public static double AverageLogFactor(BernoulliIntegerSubset sample, ISparseList<double> probTrue)
        {
            Func<Bernoulli, double, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample.SparseBernoulliList, probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="AverageLogFactor(IList{int}, SparseBetaList)"]/*'/>
        public static double AverageLogFactor(IList<int> sample, [Proper] SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(BernoulliIntegerSubset.SubsetToList(sample, probTrue.Count), probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="AverageLogFactor(IList{int}, ISparseList{double})"]/*'/>
        public static double AverageLogFactor(IList<int> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(BernoulliIntegerSubset.SubsetToList(sample, probTrue.Count), probTrue).EnumerableSum(x => x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="SampleAverageLogarithm(ISparseList{double}, BernoulliIntegerSubset)"]/*'/>
        public static BernoulliIntegerSubset SampleAverageLogarithm(ISparseList<double> probTrue, BernoulliIntegerSubset result)
        {
            Func<double, Bernoulli> f = BernoulliFromBetaOp.SampleAverageLogarithm;
            result.SetTo(f.Map(probTrue));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="SampleAverageLogarithm(SparseBetaList, BernoulliIntegerSubset)"]/*'/>
        public static BernoulliIntegerSubset SampleAverageLogarithm([SkipIfUniform] SparseBetaList probTrue, BernoulliIntegerSubset result)
        {
            Func<Beta, Bernoulli> f = BernoulliFromBetaOp.SampleAverageLogarithm;
            result.SetTo(f.Map(probTrue));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="ProbTrueAverageLogarithm(IList{int}, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageLogarithm(IList<int> sample, SparseBetaList result)
        {
            result.SetToFunction(BernoulliIntegerSubset.SubsetToList(sample, result.Count), s => BernoulliFromBetaOp.ProbTrueAverageLogarithm(s));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliIntegerSubsetFromBeta"]/message_doc[@name="ProbTrueAverageLogarithm(BernoulliIntegerSubset, SparseBetaList)"]/*'/>
        public static SparseBetaList ProbTrueAverageLogarithm(BernoulliIntegerSubset sample, SparseBetaList result)
        {
            result.SetToFunction(sample.SparseBernoulliList, s => BernoulliFromBetaOp.ProbTrueAverageLogarithm(s));
            return result;
        }
    }
}
