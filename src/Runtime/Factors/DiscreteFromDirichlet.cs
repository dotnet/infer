// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/doc/*'/>
    [FactorMethod(typeof(Discrete), "Sample", typeof(Vector))]
    [FactorMethod(new String[] { "sample", "probs" }, typeof(Factor), "Discrete", typeof(Vector))]
    [FactorMethod(typeof(EnumSupport), "DiscreteEnum<>")]
    [Quality(QualityBand.Mature)]
    public static class DiscreteFromDirichletOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogAverageFactor(Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(Discrete sample, [Fresh] Discrete to_sample)
        {
            return sample.GetLogAverageOf(to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogAverageFactor(int, Dirichlet)"]/*'/>
        public static double LogAverageFactor(int sample, Dirichlet probs)
        {
            Discrete to_sample = SampleAverageConditional(probs, Discrete.Uniform(probs.Dimension, probs.Sparsity));
            return to_sample.GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogAverageFactor(int, Vector)"]/*'/>
        public static double LogAverageFactor(int sample, Vector probs)
        {
            return Math.Log(probs[sample]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogEvidenceRatio(int, Dirichlet)"]/*'/>
        public static double LogEvidenceRatio(int sample, Dirichlet probs)
        {
            return LogAverageFactor(sample, probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogEvidenceRatio(int, Vector)"]/*'/>
        public static double LogEvidenceRatio(int sample, Vector probs)
        {
            return LogAverageFactor(sample, probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogEvidenceRatio(Discrete, Dirichlet)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Discrete sample, Dirichlet probs)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="LogEvidenceRatio(Discrete, Vector)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Discrete sample, Vector probs)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleConditional(Vector, Discrete)"]/*'/>
        public static Discrete SampleConditional(Vector probs, Discrete result)
        {
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageConditional(Vector, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(Vector probs, Discrete result)
        {
            return SampleConditional(probs, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageLogarithm(Vector, Discrete)"]/*'/>
        public static Discrete SampleAverageLogarithm(Vector probs, Discrete result)
        {
            return SampleConditional(probs, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageConditionalInit(Dirichlet)"]/*'/>
        [Skip]
        public static Discrete SampleAverageConditionalInit([IgnoreDependency] Dirichlet probs)
        {
            return Discrete.Uniform(probs.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageLogarithmInit(Dirichlet)"]/*'/>
        [Skip]
        public static Discrete SampleAverageLogarithmInit([IgnoreDependency] Dirichlet probs)
        {
            return Discrete.Uniform(probs.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageConditionalInit(Vector)"]/*'/>
        [Skip]
        public static Discrete SampleAverageConditionalInit([IgnoreDependency] Vector probs)
        {
            return Discrete.Uniform(probs.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageLogarithmInit(Vector)"]/*'/>
        [Skip]
        public static Discrete SampleAverageLogarithmInit([IgnoreDependency] Vector probs)
        {
            return Discrete.Uniform(probs.Count);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="ProbsConditional(int, Dirichlet)"]/*'/>
        public static Dirichlet ProbsConditional(int sample, Dirichlet result)
        {
            result.TotalCount = result.Dimension + 1;
            result.PseudoCount.SetAllElementsTo(1);
            result.PseudoCount[sample] = 2;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="ProbsAverageConditional(int, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageConditional(int sample, Dirichlet result)
        {
            return ProbsConditional(sample, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="ProbsAverageLogarithm(int, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageLogarithm(int sample, Dirichlet result)
        {
            return ProbsConditional(sample, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageConditional(Dirichlet, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional([SkipIfUniform] Dirichlet probs, Discrete result)
        {
            result.SetProbs(probs.GetMean(result.GetWorkspace()));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="SampleAverageLogarithm(Dirichlet, Discrete)"]/*'/>
        public static Discrete SampleAverageLogarithm([SkipIfUniform] Dirichlet probs, Discrete result)
        {
            // E[sum_k I(X=k) log(P[k])] = sum_k I(X=k) E[log(P[k])]
            Vector p = probs.GetMeanLog(result.GetWorkspace());
            double max = p.Max();
            p.SetToFunction(p, x => Math.Exp(x - max));
            result.SetProbs(p);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="ProbsAverageLogarithm(Discrete, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageLogarithm(Discrete sample, Dirichlet result)
        {
            // E[sum_k I(X=k) log(P[k])] = sum_k p(X=k) log(P[k])
            result.TotalCount = result.Dimension + 1;
            result.PseudoCount.SetAllElementsTo(1);
            result.PseudoCount.SetToSum(result.PseudoCount, sample.GetProbs());
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="ProbsAverageConditional(Discrete, Vector, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageConditional([SkipIfUniform] Discrete sample, Vector probs, Dirichlet result)
        {
            Vector sampleProbs = sample.GetProbs();
            double Z = sampleProbs.Inner(probs);
            Vector dLogP = Vector.Zero(probs.Count);
            Vector ddLogP = Vector.Zero(probs.Count);
            double sampleN = sampleProbs[probs.Count - 1];
            double probN = probs[probs.Count - 1];
            dLogP.SetToFunction(sampleProbs, sampleProb => (sampleProb - sampleN) / Z);
            ddLogP.SetToFunction(dLogP, x => -x * x);
            result.SetDerivatives(probs, dLogP, ddLogP, !Dirichlet.AllowImproperSum);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="ProbsAverageConditional(Discrete, Dirichlet, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageConditional([SkipIfUniform] Discrete sample, [NoInit] Dirichlet probs, Dirichlet result)
        {
            if (probs.IsPointMass)
                return ProbsAverageConditional(sample, probs.Point, result);
            if (sample.IsPointMass)
                return ProbsConditional(sample.Point, result);
            // Z = sum_x q(x) int_p f(x,p)*q(p) = sum_x q(x) E[p[x]]
            Vector sampleProbs = sample.GetProbs();
            double Z = sampleProbs.Inner(probs.PseudoCount);
            double invZ = 1.0 / Z;
            // the posterior is a mixture of Dirichlets having the following form:
            // sum_x q(x) (alpha(x)/sum_i alpha(i)) Dirichlet(p; alpha(x)+1, alpha(not x)+0)
            // where the mixture weights are w(x) =propto q(x) alpha(x)
            //                               w[i] = sample[i]*probs.PseudoCount[i]/Z
            // The posterior mean of probs(x) = (w(x) + alpha(x))/(1 + sum_x alpha(x))
            double invTotalCountPlus1 = 1.0 / (probs.TotalCount + 1);
            Vector m = Vector.Zero(sample.Dimension, sample.Sparsity);
            m.SetToFunction(sampleProbs, probs.PseudoCount, (x, y) => (x * invZ + 1.0) * y * invTotalCountPlus1);
            if (!Dirichlet.AllowImproperSum)
            {
                // To get the correct mean, we need (probs.PseudoCount[i] + delta[i]) to be proportional to m[i].
                // If we set delta[argmin] = 0, then we just solve the equation 
                //   (probs.PseudoCount[i] + delta[i])/probs.PseudoCount[argmin] = m[i]/m[argmin]
                // for delta[i].
                int argmin = sampleProbs.IndexOfMinimum();
                Debug.Assert(argmin != -1);
                double newTotalCount = probs.PseudoCount[argmin] / m[argmin];
                double argMinValue = sampleProbs[argmin];
                result.PseudoCount.SetToFunction(m, probs.PseudoCount, (x, y) => 1.0 + (x * newTotalCount) - y);
                result.PseudoCount.SetToFunction(result.PseudoCount, sampleProbs, (x, y) => (y == argMinValue) ? 1.0 : x);
                result.TotalCount = result.PseudoCount.Sum(); // result.Dimension + newTotalCount - probs.TotalCount;
                return result;
            }
            else
            {
                // The posterior meanSquare of probs(x) = (2 w(x) + alpha(x))/(2 + sum_x alpha(x)) * (1 + alpha(x))/(1 + sum_x alpha(x))
                double invTotalCountPlus2 = 1.0 / (2 + probs.TotalCount);
                Vector m2 = Vector.Zero(sample.Dimension, sample.Sparsity);
                m2.SetToFunction(sampleProbs, probs.PseudoCount, (x, y) => (2.0 * x * invZ + 1.0) * y * invTotalCountPlus2 * (1.0 + y) * invTotalCountPlus1);
                result.SetMeanAndMeanSquare(m, m2);
                result.SetToRatio(result, probs);
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="AverageLogFactor(Discrete, Dirichlet)"]/*'/>
        public static double AverageLogFactor(Discrete sample, [Proper] Dirichlet probs)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, probs);

            if (sample.Dimension != probs.Dimension)
                throw new ArgumentException("sample.Dimension (" + sample.Dimension + ") != probs.Dimension (" + probs.Dimension + ")");
            Vector sampleProbs = sample.GetProbs();
            Vector pSuffStats = probs.GetMeanLog();
            // avoid multiplication of 0*log(0)
            foreach (int i in sampleProbs.IndexOfAll(v => v == 0.0))
                pSuffStats[i] = 0.0;
            double total = Vector.InnerProduct(sampleProbs, pSuffStats);
            return total;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="AverageLogFactor(Discrete, Vector)"]/*'/>
        public static double AverageLogFactor(Discrete sample, Vector probs)
        {
            return AverageLogFactor(sample, Dirichlet.PointMass(probs));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="AverageLogFactor(int, Dirichlet)"]/*'/>
        public static double AverageLogFactor(int sample, [Proper] Dirichlet probs)
        {
            return probs.GetMeanLogAt(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDirichletOp"]/message_doc[@name="AverageLogFactor(int, Vector)"]/*'/>
        public static double AverageLogFactor(int sample, Vector probs)
        {
            return Math.Log(probs[sample]);
        }
    }
}
