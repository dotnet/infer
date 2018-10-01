// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/doc/*'/>
    /// <exclude/>
    [FactorMethod(new string[] { "sample", "index", "probTrue" }, typeof(Factor), "BernoulliFromDiscrete")]
    public static class BernoulliFromDiscreteOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, double[])"]/*'/>
        public static double LogAverageFactor(bool sample, Discrete index, double[] probTrue)
        {
            double p = 0;
            for (int i = 0; i < index.Dimension; i++)
            {
                p += probTrue[i] * index[i];
            }
            if (!sample)
                p = 1 - p;
            return Math.Log(p);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="LogEvidenceRatio(bool, Discrete, double[])"]/*'/>
        public static double LogEvidenceRatio(bool sample, Discrete index, double[] probTrue)
        {
            return LogAverageFactor(sample, index, probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="SampleConditional(int, double[])"]/*'/>
        public static Bernoulli SampleConditional(int index, double[] ProbTrue)
        {
            Bernoulli result = new Bernoulli();
            result.SetProbTrue(ProbTrue[index]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="SampleAverageConditional(int, double[])"]/*'/>
        public static Bernoulli SampleAverageConditional(int index, double[] ProbTrue)
        {
            return SampleConditional(index, ProbTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="SampleAverageLogarithm(int, double[])"]/*'/>
        public static Bernoulli SampleAverageLogarithm(int index, double[] ProbTrue)
        {
            return SampleConditional(index, ProbTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="IndexConditional(bool, double[], Discrete)"]/*'/>
        public static Discrete IndexConditional(bool sample, double[] ProbTrue, Discrete result)
        {
            if (result == default(Discrete))
                result = Discrete.Uniform(ProbTrue.Length);
            Vector prob = result.GetWorkspace();
            if (sample)
            {
                prob.SetTo(ProbTrue);
            }
            else
            {
                prob.SetTo(ProbTrue);
                prob.SetToDifference(1.0, prob);
            }
            result.SetProbs(prob);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="IndexAverageConditional(bool, double[], Discrete)"]/*'/>
        public static Discrete IndexAverageConditional(bool sample, double[] ProbTrue, Discrete result)
        {
            return IndexConditional(sample, ProbTrue, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="IndexAverageLogarithm(bool, double[], Discrete)"]/*'/>
        public static Discrete IndexAverageLogarithm(bool sample, double[] ProbTrue, Discrete result)
        {
            return IndexConditional(sample, ProbTrue, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="SampleAverageConditional(Discrete, double[])"]/*'/>
        public static Bernoulli SampleAverageConditional(Discrete index, double[] ProbTrue)
        {
            Bernoulli result = new Bernoulli();
            // E[X] = sum_Y p(Y) ProbTrue[Y]
            double p = 0;
            for (int i = 0; i < index.Dimension; i++)
            {
                p += ProbTrue[i] * index[i];
            }
            result.SetProbTrue(p);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="SampleAverageLogarithm(Discrete, double[])"]/*'/>
        public static Bernoulli SampleAverageLogarithm(Discrete index, double[] ProbTrue)
        {
            Bernoulli result = new Bernoulli();
            // E[sum_k I(Y=k) (X*log(ProbTrue[k]) + (1-X)*log(1-ProbTrue[k]))]
            // = X*(sum_k p(Y=k) log(ProbTrue[k])) + (1-X)*(sum_k p(Y=k) log(1-ProbTrue[k]))
            // p(X=true) =propto prod_k ProbTrue[k]^p(Y=k)
            // log(p(X=true)/p(X=false)) = sum_k p(Y=k) log(ProbTrue[k]/(1-ProbTrue[k]))
            double s = 0;
            for (int i = 0; i < index.Dimension; i++)
            {
                s += index[i] * MMath.Logit(ProbTrue[i]);
            }
            result.LogOdds = s;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="IndexAverageConditional(Bernoulli, double[], Discrete)"]/*'/>
        public static Discrete IndexAverageConditional([SkipIfUniform] Bernoulli sample, double[] ProbTrue, Discrete result)
        {
            if (result == default(Discrete))
                result = Discrete.Uniform(ProbTrue.Length);
            // p(Y) = ProbTrue[Y]*p(X=true) + (1-ProbTrue[Y])*p(X=false)
            Vector probs = result.GetWorkspace();
            double p = sample.GetProbTrue();
            probs.SetTo(ProbTrue);
            probs.SetToProduct(probs, 2.0 * p - 1.0);
            probs.SetToSum(probs, 1.0 - p);
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromDiscreteOp"]/message_doc[@name="IndexAverageLogarithm(Bernoulli, double[], Discrete)"]/*'/>
        public static Discrete IndexAverageLogarithm(Bernoulli sample, double[] ProbTrue, Discrete result)
        {
            if (result == default(Discrete))
                result = Discrete.Uniform(ProbTrue.Length);
            // E[sum_k I(Y=k) (X*log(ProbTrue[k]) + (1-X)*log(1-ProbTrue[k]))]
            // = sum_k I(Y=k) (p(X=true)*log(ProbTrue[k]) + p(X=false)*log(1-ProbTrue[k]))
            // p(Y=k) =propto ProbTrue[k]^p(X=true) (1-ProbTrue[k])^p(X=false)
            Vector probs = result.GetWorkspace();
            double p = sample.GetProbTrue();
            probs.SetTo(ProbTrue);
            probs.SetToFunction(probs, x => Math.Pow(x, p) * Math.Pow(1.0 - x, 1.0 - p));
            result.SetProbs(probs);
            return result;
        }
    }
}
