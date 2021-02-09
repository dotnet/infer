// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Distributions;
    using Math;
    using Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/doc/*'/>
    [FactorMethod(typeof(Bernoulli), "Sample", typeof(double))]
    [FactorMethod(new String[] { "Sample", "ProbTrue" }, typeof(Factor), "Bernoulli")]
    [Quality(QualityBand.Mature)]
    public static class BernoulliFromBetaOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogAverageFactor(bool, double)"]/*'/>
        public static double LogAverageFactor(bool sample, double probTrue)
        {
            return sample ? Math.Log(probTrue) : Math.Log(1 - probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli sample, [Fresh] Bernoulli to_sample)
        {
            return sample.GetLogAverageOf(to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogAverageFactor(bool, Beta)"]/*'/>
        public static double LogAverageFactor(bool sample, Beta probTrue)
        {
            Bernoulli to_sample = SampleAverageConditional(probTrue);
            return to_sample.GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli sample, double probTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(bool, Beta)"]/*'/>
        public static double LogEvidenceRatio(bool sample, Beta probTrue)
        {
            return LogAverageFactor(sample, probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(bool, double)"]/*'/>
        public static double LogEvidenceRatio(bool sample, double probTrue)
        {
            return LogAverageFactor(sample, probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli, Beta)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli sample, Beta probTrue)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="SampleConditional(double)"]/*'/>
        public static Bernoulli SampleConditional(double probTrue)
        {
            return new Bernoulli(probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="SampleAverageConditional(double)"]/*'/>
        public static Bernoulli SampleAverageConditional(double probTrue)
        {
            return SampleConditional(probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="ProbTrueConditional(bool)"]/*'/>
        public static Beta ProbTrueConditional(bool sample)
        {
            if (sample)
            {
                return new Beta(2, 1);
            }
            else
            {
                return new Beta(1, 2);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageConditional(bool)"]/*'/>
        public static Beta ProbTrueAverageConditional(bool sample)
        {
            return ProbTrueConditional(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="SampleAverageConditional(Beta)"]/*'/>
        public static Bernoulli SampleAverageConditional([SkipIfUniform] Beta probTrue)
        {
            if (probTrue.IsPointMass)
            {
                return new Bernoulli(probTrue.Point);
            }
            else if (!probTrue.IsProper())
                throw new ImproperMessageException(probTrue);
            else
            {
                // p(x=true) = trueCount/total
                // p(x=false) = falseCount/total
                // log(p(x=true)/p(x=false)) = log(trueCount/falseCount)
                return Bernoulli.FromLogOdds(Math.Log(probTrue.TrueCount / probTrue.FalseCount));
            }
        }

        public static Beta ProbTrueAverageConditional([SkipIfUniform] Bernoulli sample, double probTrue)
        {
            // f(x,p) = q(T) p + q(F) (1-p)
            // dlogf/dp = (q(T) - q(F))/f(x,p)
            // ddlogf/dp^2 = -(q(T) - q(F))^2/f(x,p)^2
            double qT = sample.GetProbTrue();
            double qF = sample.GetProbFalse();
            double pTT = qT * probTrue;
            double probFalse = 1 - probTrue;
            double pFF = qF * probFalse;
            double Z = pTT + pFF;
            double dlogp = (qT - qF) / Z;
            double ddlogp = -dlogp * dlogp;
            return Beta.FromDerivatives(probTrue, dlogp, ddlogp, !Beta.AllowImproperSum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageConditional(Bernoulli, Beta)"]/*'/>
        public static Beta ProbTrueAverageConditional([SkipIfUniform] Bernoulli sample, Beta probTrue)
        {
            // this code is similar to DiscreteFromDirichletOp.ProbsAverageConditional()
            if (probTrue.IsPointMass)
            {
                return ProbTrueAverageConditional(sample, probTrue.Point);
            }
            if (sample.IsPointMass)
            {
                // shortcut
                return ProbTrueConditional(sample.Point);
            }
            if (!probTrue.IsProper())
                throw new ImproperMessageException(probTrue);
            // q(x) is the distribution stored in this.X.
            // q(p) is the distribution stored in this.P.
            // f(x,p) is the factor.
            // Z = sum_x q(x) int_p f(x,p)*q(p) = q(false)*E[1-p] + q(true)*E[p]
            // Ef[p] = 1/Z sum_x q(x) int_p p*f(x,p)*q(p) = 1/Z (q(false)*E[p(1-p)] + q(true)*E[p^2])
            // Ef[p^2] = 1/Z sum_x q(x) int_p p^2*f(x,p)*q(p) = 1/Z (q(false)*E[p^2(1-p)] + q(true)*E[p^3])
            // var_f(p) = Ef[p^2] - Ef[p]^2
            double mo = probTrue.GetMean();
            double m2o = probTrue.GetMeanSquare();
            double pT = sample.GetProbTrue();
            double pF = sample.GetProbFalse();
            double Z = pF * (1 - mo) + pT * mo;
            double m = pF * (mo - m2o) + pT * m2o;
            m = m / Z;
            if (!Beta.AllowImproperSum)
            {
                if (pT < 0.5)
                {
                    double inc = probTrue.TotalCount * (mo / m - 1);
                    return new Beta(1, 1 + inc);
                }
                else
                {
                    double inc = probTrue.TotalCount * ((1 - mo) / (1 - m) - 1);
                    return new Beta(1 + inc, 1);
                }
            }
            else
            {
                double m3o = probTrue.GetMeanCube();
                double m2 = pF * (m2o - m3o) + pT * m3o;
                m2 = m2 / Z;
                Beta result = Beta.FromMeanAndVariance(m, m2 - m * m);
                result.SetToRatio(result, probTrue);
                return result;
            }
        }

        //-- VMP -------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(Bernoulli, Beta)"]/*'/>
        public static double AverageLogFactor(Bernoulli sample, [Proper] Beta probTrue)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, probTrue);
            double eLogP, eLog1MinusP;
            probTrue.GetMeanLogs(out eLogP, out eLog1MinusP);
            double p = sample.GetProbTrue();
            return p * eLogP + (1 - p) * eLog1MinusP;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(Bernoulli, double)"]/*'/>
        public static double AverageLogFactor(Bernoulli sample, double probTrue)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, probTrue);
            return AverageLogFactor(sample, Beta.PointMass(probTrue));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(bool, Beta)"]/*'/>
        public static double AverageLogFactor(bool sample, [Proper] Beta probTrue)
        {
            double eLogP, eLog1MinusP;
            probTrue.GetMeanLogs(out eLogP, out eLog1MinusP);
            return sample ? eLogP : eLog1MinusP;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="AverageLogFactor(bool, double)"]/*'/>
        public static double AverageLogFactor(bool sample, double probTrue)
        {
            return sample ? Math.Log(probTrue) : Math.Log(1 - probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="SampleAverageLogarithm(double)"]/*'/>
        public static Bernoulli SampleAverageLogarithm(double probTrue)
        {
            return SampleConditional(probTrue);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="SampleAverageLogarithm(Beta)"]/*'/>
        public static Bernoulli SampleAverageLogarithm([SkipIfUniform] Beta probTrue)
        {
            if (probTrue.IsPointMass)
            {
                return new Bernoulli(probTrue.Point);
            }
            else if (!probTrue.IsProper())
                throw new ImproperMessageException(probTrue);
            else
            {
                // E[x*log(p) + (1-x)*log(1-p)] = x*E[log(p)] + (1-x)*E[log(1-p)]
                // p(x=true) = exp(E[log(p)])/(exp(E[log(p)]) + exp(E[log(1-p)]))
                // log(p(x=true)/p(x=false)) = E[log(p)] - E[log(1-p)] = digamma(trueCount) - digamma(falseCount)
                return Bernoulli.FromLogOdds(MMath.Digamma(probTrue.TrueCount) - MMath.Digamma(probTrue.FalseCount));
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageLogarithm(bool)"]/*'/>
        public static Beta ProbTrueAverageLogarithm(bool sample)
        {
            return ProbTrueConditional(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromBetaOp"]/message_doc[@name="ProbTrueAverageLogarithm(Bernoulli)"]/*'/>
        public static Beta ProbTrueAverageLogarithm(Bernoulli sample)
        {
            // E[x*log(p) + (1-x)*log(1-p)] = E[x]*log(p) + (1-E[x])*log(1-p)
            double ex = sample.GetProbTrue();
            return new Beta(1 + ex, 2 - ex);
        }
    }
}
