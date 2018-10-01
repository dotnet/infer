// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define FAST

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/doc/*'/>
    /// <exclude/>
    [FactorMethod(new string[] { "Sample", "Choice", "ProbTrue0", "ProbTrue1" }, typeof(Factor), "BernoulliFromBoolean", typeof(bool), typeof(double), typeof(double))]
    public static class BernoulliFromBooleanOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="SampleConditional(bool, double, double)"]/*'/>
        public static Bernoulli SampleConditional(bool choice, double probTrue0, double probTrue1)
        {
            Bernoulli result = new Bernoulli();
            result.SetProbTrue(choice ? probTrue1 : probTrue0);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="SampleAverageConditional(bool, double, double)"]/*'/>
        public static Bernoulli SampleAverageConditional(bool choice, double probTrue0, double probTrue1)
        {
            return SampleConditional(choice, probTrue0, probTrue1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="SampleAverageLogarithm(bool, double, double)"]/*'/>
        public static Bernoulli SampleAverageLogarithm(bool choice, double probTrue0, double probTrue1)
        {
            return SampleConditional(choice, probTrue0, probTrue1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="ChoiceConditional(bool, double, double)"]/*'/>
        public static Bernoulli ChoiceConditional(bool sample, double probTrue0, double probTrue1)
        {
            Bernoulli result = new Bernoulli();
            if (probTrue0 == 0 || probTrue1 == 0)
                throw new ArgumentException("probTrue is zero");
            if (sample)
            {
                double sum = probTrue0 + probTrue1;
                if (sum == 0.0)
                    throw new AllZeroException();
                else
                    result.SetProbTrue(probTrue1 / sum);
            }
            else
            {
                double sum = 2 - probTrue1 - probTrue0;
                if (sum == 0.0)
                    throw new AllZeroException();
                else
                    result.SetProbTrue((1 - probTrue1) / sum);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="ChoiceAverageConditional(bool, double, double)"]/*'/>
        public static Bernoulli ChoiceAverageConditional(bool sample, double probTrue0, double probTrue1)
        {
            return ChoiceConditional(sample, probTrue0, probTrue1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="ChoiceAverageLogarithm(bool, double, double)"]/*'/>
        public static Bernoulli ChoiceAverageLogarithm(bool sample, double probTrue0, double probTrue1)
        {
            return ChoiceConditional(sample, probTrue0, probTrue1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="SampleAverageConditional(Bernoulli, double, double)"]/*'/>
        public static Bernoulli SampleAverageConditional(Bernoulli choice, double probTrue0, double probTrue1)
        {
            Bernoulli result = new Bernoulli();
            if (choice.IsPointMass)
                return SampleConditional(choice.Point, probTrue0, probTrue1);
#if FAST
            result.SetProbTrue(choice.GetProbFalse() * probTrue0 + choice.GetProbTrue() * probTrue1);
#else
    // This method is more numerically stable but slower.
    // let oX = log(p(X)/(1-p(X))
    // let oY = log(p(Y)/(1-p(Y))
    // oX = log( (TT*sigma(oY) + TF*sigma(-oY))/(FT*sigma(oY) + FF*sigma(-oY)) )
    //    = log( (TT*exp(oY) + TF)/(FT*exp(oY) + FF) )
    //    = log( (exp(oY) + TF/TT)/(exp(oY) + FF/FT) ) + log(TT/FT)
    // ay = log(TF/TT)
    // by = log(FF/FT)
    // offset = log(TT/FT)
            if (probTrue0 == 0 || probTrue1 == 0) throw new ArgumentException("probTrue is zero");
            double ay = Math.Log(probTrue0 / probTrue1);
            double by = Math.Log((1 - probTrue0) / (1 - probTrue1));
            double offset = MMath.Logit(probTrue1);
            result.LogOdds = MMath.DiffLogSumExp(choice.LogOdds, ay, by) + offset;
#endif
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="ChoiceAverageConditional(Bernoulli, double, double)"]/*'/>
        public static Bernoulli ChoiceAverageConditional([SkipIfUniform] Bernoulli sample, double probTrue0, double probTrue1)
        {
            Bernoulli result = new Bernoulli();
            if (sample.IsPointMass)
                return ChoiceConditional(sample.Point, probTrue0, probTrue1);
#if FAST
            double p1 = sample.GetProbFalse() * (1 - probTrue1) + sample.GetProbTrue() * probTrue1;
            double p0 = sample.GetProbFalse() * (1 - probTrue0) + sample.GetProbTrue() * probTrue0;
            double sum = p0 + p1;
            if (sum == 0.0)
                throw new AllZeroException();
            else
                result.SetProbTrue(p1 / sum);
#else
    // This method is more numerically stable but slower.
    // ax = log(FT/TT)
    // bx = log(FF/TF)
    // offset = log(TT/TF)
            if (probTrue0 == 0 || probTrue1 == 0) throw new ArgumentException("probTrue is zero");
            double ax = -MMath.Logit(probTrue1);
            double bx = -MMath.Logit(probTrue0);
            double offset = Math.Log(probTrue1 / probTrue0);
            result.LogOdds = MMath.DiffLogSumExp(sample.LogOdds, ax, bx) + offset;
#endif
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="SampleAverageLogarithm(Bernoulli, double, double)"]/*'/>
        public static Bernoulli SampleAverageLogarithm(Bernoulli choice, double probTrue0, double probTrue1)
        {
            Bernoulli result = new Bernoulli();
            if (choice.IsPointMass)
                return SampleConditional(choice.Point, probTrue0, probTrue1);
            // log(p(X=true)/p(X=false)) = sum_k p(Y=k) log(ProbTrue[k]/(1-ProbTrue[k]))
            result.LogOdds = choice.GetProbFalse() * MMath.Logit(probTrue0) + choice.GetProbTrue() * MMath.Logit(probTrue1);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanOp"]/message_doc[@name="ChoiceAverageLogarithm(Bernoulli, double, double)"]/*'/>
        public static Bernoulli ChoiceAverageLogarithm(Bernoulli sample, double probTrue0, double probTrue1)
        {
            Bernoulli result = new Bernoulli();
            if (sample.IsPointMass)
                return ChoiceConditional(sample.Point, probTrue0, probTrue1);
            // p(Y=k) =propto ProbTrue[k]^p(X=true) (1-ProbTrue[k])^p(X=false)
            // log(p(Y=true)/p(Y=false)) = p(X=true)*log(ProbTrue[1]/ProbTrue[0]) + p(X=false)*log((1-ProbTrue[1])/(1-ProbTrue[0]))
            //                           = p(X=false)*(log(ProbTrue[0]/(1-ProbTrue[0]) - log(ProbTrue[1]/(1-ProbTrue[1]))) + log(ProbTrue[1]/ProbTrue[0])
            if (probTrue0 == 0 || probTrue1 == 0)
                throw new ArgumentException("probTrue is zero");
            result.LogOdds = sample.GetProbTrue() * Math.Log(probTrue1 / probTrue0) + sample.GetProbFalse() * Math.Log((1 - probTrue1) / (1 - probTrue0));
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanArray"]/doc/*'/>
    [FactorMethod(new string[] { "Sample", "Choice", "ProbTrue" }, typeof(Factor), "BernoulliFromBoolean", typeof(bool), typeof(double[]))]
    public static class BernoulliFromBooleanArray
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanArray"]/message_doc[@name="SampleAverageConditional(Bernoulli, double[])"]/*'/>
        public static Bernoulli SampleAverageConditional(Bernoulli choice, double[] probTrue)
        {
            return BernoulliFromBooleanOp.SampleAverageConditional(choice, probTrue[0], probTrue[1]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanArray"]/message_doc[@name="SampleAverageConditional(bool, double[])"]/*'/>
        public static Bernoulli SampleAverageConditional(bool choice, double[] probTrue)
        {
            return SampleAverageConditional(Bernoulli.PointMass(choice), probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanArray"]/message_doc[@name="ChoiceAverageConditional(Bernoulli, double[])"]/*'/>
        public static Bernoulli ChoiceAverageConditional(Bernoulli sample, double[] probTrue)
        {
            return BernoulliFromBooleanOp.ChoiceAverageConditional(sample, probTrue[0], probTrue[1]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanArray"]/message_doc[@name="SampleAverageLogarithm(Bernoulli, double[])"]/*'/>
        public static Bernoulli SampleAverageLogarithm(Bernoulli choice, double[] probTrue)
        {
            return BernoulliFromBooleanOp.SampleAverageLogarithm(choice, probTrue[0], probTrue[1]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBooleanArray"]/message_doc[@name="ChoiceAverageLogarithm(Bernoulli, double[])"]/*'/>
        public static Bernoulli ChoiceAverageLogarithm(Bernoulli sample, double[] probTrue)
        {
            return BernoulliFromBooleanOp.ChoiceAverageLogarithm(sample, probTrue[0], probTrue[1]);
        }
    }
}
