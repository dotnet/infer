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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/doc/*'/>
    /// <remarks>The factor is f(sample,p,n) = n!/prod_k sample[k]!  prod_k p[k]^sample[k]</remarks>
    [FactorMethod(new string[] { "sample", "trialCount", "p" }, typeof(Rand), "Multinomial", typeof(int), typeof(Vector))]
    [FactorMethod(typeof(Factor), "MultinomialList", typeof(int), typeof(Vector))]
    [Buffers("MeanLog")]
    [Quality(QualityBand.Preview)]
    public static class MultinomialOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="LogEvidenceRatio(IList{Discrete})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(IList<Discrete> sample)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="LogAverageFactor(IList{int}, int, IList{double})"]/*'/>
        public static double LogAverageFactor(IList<int> sample, int trialCount, IList<double> p)
        {
            double result = MMath.GammaLn(trialCount + 1);
            for (int i = 0; i < sample.Count; i++)
            {
                result += sample[i] * Math.Log(p[i]) - MMath.GammaLn(sample[i] + 1);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="LogAverageFactor(IList{int}, int, Dirichlet)"]/*'/>
        public static double LogAverageFactor(IList<int> sample, int trialCount, Dirichlet p)
        {
            double result = MMath.GammaLn(trialCount + 1);
            for (int i = 0; i < sample.Count; i++)
            {
                result += MMath.GammaLn(sample[i] + p.PseudoCount[i]) + MMath.GammaLn(p.PseudoCount[i])
                          - MMath.GammaLn(sample[i] + 1);
            }
            result += MMath.GammaLn(p.TotalCount) - MMath.GammaLn(p.TotalCount + trialCount);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="LogEvidenceRatio(IList{int}, int, Dirichlet)"]/*'/>
        public static double LogEvidenceRatio(IList<int> sample, int trialCount, Dirichlet p)
        {
            return LogAverageFactor(sample, trialCount, p);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="LogEvidenceRatio(IList{int}, int, IList{double})"]/*'/>
        public static double LogEvidenceRatio(IList<int> sample, int trialCount, IList<double> p)
        {
            return LogAverageFactor(sample, trialCount, p);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="PAverageConditional(IList{int}, int)"]/*'/>
        public static Dirichlet PAverageConditional(IList<int> sample, int trialCount)
        {
            return PAverageLogarithm(sample, trialCount);
        }

        //-- VMP ------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="AverageLogFactor(IList{int}, int, IList{double})"]/*'/>
        public static double AverageLogFactor(IList<int> sample, int trialCount, IList<double> p)
        {
            return LogAverageFactor(sample, trialCount, p);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="MeanLog(Dirichlet)"]/*'/>
        public static Vector MeanLog(Dirichlet p)
        {
            return p.GetMeanLog();
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="AverageLogFactor(IList{int}, int, Dirichlet, Vector)"]/*'/>
        public static double AverageLogFactor(IList<int> sample, int trialCount, Dirichlet p, [Fresh] Vector MeanLog)
        {
            double result = MMath.GammaLn(trialCount + 1);
            if (true)
            {
                result += sample.Inner(MeanLog);
                result -= sample.ListSum(x => MMath.GammaLn(x + 1));
            }
            else
            {
                for (int i = 0; i < sample.Count; i++)
                {
                    result += sample[i] * MeanLog[i] - MMath.GammaLn(sample[i] + 1);
                }
            }
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MultinomialOp"]/message_doc[@name="PAverageLogarithm(IList{int}, int)"]/*'/>
        public static Dirichlet PAverageLogarithm(IList<int> sample, int trialCount)
        {
            // This method demonstrates how to write an operator method using extension methods,
            // so that it is efficient for sparse and dense lists.
            //
            // The vector returned here will be sparse if the list is sparse.
            var counts = sample.ListSelect(x => x + 1.0).ToVector();
            return new Dirichlet(counts);
        }
    }
}
