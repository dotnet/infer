// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/doc/*'/>
    /// <remarks>The factor is f(sample,n,p) = choose(n,sample) p^sample (1-p)^(n-sample)</remarks>
    [FactorMethod(new string[] { "sample", "trialCount", "p" }, typeof(Rand), "Binomial", typeof(int), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class BinomialOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogAverageFactor(Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(Discrete sample, [Fresh] Discrete to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogAverageFactor(int, double, int)"]/*'/>
        public static double LogAverageFactor(int sample, double p, int trialCount)
        {
            return MMath.ChooseLn(trialCount, sample) + sample * Math.Log(p) + (trialCount - sample) * Math.Log(1 - p);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogAverageFactor(int, double, Discrete)"]/*'/>
        public static double LogAverageFactor(int sample, double p, Discrete trialCount)
        {
            double logZ = Double.NegativeInfinity;
            for (int n = 0; n < trialCount.Dimension; n++)
            {
                logZ = MMath.LogSumExp(logZ, trialCount.GetLogProb(n) + LogAverageFactor(sample, p, n));
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogAverageFactor(int, Beta, Discrete)"]/*'/>
        public static double LogAverageFactor(int sample, Beta p, Discrete trialCount)
        {
            double logZ = Double.NegativeInfinity;
            for (int n = 0; n < trialCount.Dimension; n++)
            {
                logZ = MMath.LogSumExp(logZ, trialCount.GetLogProb(n) + LogAverageFactor(sample, p, n));
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogAverageFactor(int, Beta, int)"]/*'/>
        public static double LogAverageFactor(int sample, Beta p, int trialCount)
        {
            if (trialCount < sample)
                return double.NegativeInfinity;
            double a = p.TrueCount;
            double b = p.FalseCount;
            return MMath.ChooseLn(trialCount, sample) +
                   MMath.GammaLn(a + sample) - MMath.GammaLn(a) +
                   MMath.GammaLn(b + trialCount - sample) - MMath.GammaLn(b) +
                   MMath.GammaLn(a + b) - MMath.GammaLn(a + b + trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogEvidenceRatio(Discrete)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Discrete sample)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogEvidenceRatio(int, double, Discrete)"]/*'/>
        public static double LogEvidenceRatio(int sample, double p, Discrete trialCount)
        {
            return LogAverageFactor(sample, p, trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogEvidenceRatio(int, Beta, Discrete)"]/*'/>
        public static double LogEvidenceRatio(int sample, Beta p, Discrete trialCount)
        {
            return LogAverageFactor(sample, p, trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogEvidenceRatio(int, double, int)"]/*'/>
        public static double LogEvidenceRatio(int sample, double p, int trialCount)
        {
            return LogAverageFactor(sample, p, trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="LogEvidenceRatio(int, Beta, int)"]/*'/>
        public static double LogEvidenceRatio(int sample, Beta p, int trialCount)
        {
            return LogAverageFactor(sample, p, trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditionalInit(int)"]/*'/>
        [Skip]
        public static Discrete SampleAverageConditionalInit(int trialCount)
        {
            return Discrete.Uniform(trialCount + 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditionalInit(Discrete)"]/*'/>
        [Skip]
        public static Discrete SampleAverageConditionalInit([IgnoreDependency] Discrete trialCount)
        {
            return Discrete.Uniform(trialCount.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditional(double, int, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(double p, int trialCount, Discrete result)
        {
            // result must range from 0 to n
            if (result.Dimension < trialCount + 1)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < n+1 (" + trialCount + "+1)");
            Vector probs = result.GetWorkspace();
            double logp = Math.Log(p);
            double log1minusp = Math.Log(1 - p);
            probs.SetAllElementsTo(0.0);
            for (int k = 0; k <= trialCount; k++)
                probs[k] = Math.Exp(MMath.ChooseLn(trialCount, k) + k * logp + (trialCount - k) * log1minusp);
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditional(Beta, int, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(Beta p, int trialCount, Discrete result)
        {
            if (p.IsPointMass)
                return SampleAverageConditional(p.Point, trialCount, result);
            // result must range from 0 to n
            if (result.Dimension < trialCount + 1)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < n+1 (" + trialCount + "+1)");
            Vector probs = result.GetWorkspace();
            double a = p.TrueCount;
            double b = p.FalseCount;
            probs.SetAllElementsTo(0.0);
            double max = double.NegativeInfinity;
            for (int k = 0; k <= trialCount; k++)
            {
                double logProb = MMath.GammaLn(a + k) - MMath.GammaLn(1 + k) + MMath.GammaLn(b + trialCount - k) - MMath.GammaLn(1 + trialCount - k);
                probs[k] = logProb;
                if (logProb > max)
                    max = logProb;
            }
            for (int k = 0; k <= trialCount; k++)
            {
                probs[k] = Math.Exp(probs[k] - max);
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditional(double, Poisson)"]/*'/>
        public static Poisson SampleAverageConditional(double p, Poisson trialCount)
        {
            if (trialCount.Precision != 1)
                throw new NotImplementedException("precision != 1 not implemented");
            // p(sample=k) = sum(n>=k) lambda^n/n! exp(-lambda) nchoosek(n,k) p^k (1-p)^(n-k)
            //             = (p/(1-p))^k 1/k! exp(-lambda) sum_(n>=k) 1/(n-k)! (1-p)^n lambda^n
            //             = (p/(1-p))^k 1/k! exp(-lambda) (1-p)^k lambda^k exp((1-p) lambda)
            //             = p^k lambda^k 1/k! exp(-p lambda)
            return new Poisson(p * trialCount.Rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageConditional(int, double, Poisson)"]/*'/>
        public static Poisson TrialCountAverageConditional(int sample, double p, Poisson trialCount)
        {
            if (trialCount.Precision != 1)
                throw new NotImplementedException("precision != 1 not implemented");
            // p(n) =propto lambda^n/n! exp(-lambda) nchoosek(n,x) p^x (1-p)^(n-x)
            //      =propto lambda^n (1-p)^n 1/(n-x)!
            //      = Po(n-x; lambda*(1-p))
            // E[n] = x + lambda*(1-p)
            return new Poisson(sample + trialCount.Rate * (1 - p)) / trialCount;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageConditional(Poisson, double)"]/*'/>
        public static Poisson TrialCountAverageConditional(Poisson sample, double p)
        {
            if (sample.Precision != 0)
                throw new NotImplementedException("precision != 0 not implemented");
            // sum(x<=n) r^x nchoosek(n,x) p^x (1-p)^(n-x)
            // =propto (1-p)^n sum(x<=n) r^x nchoosek(n,x) (p/(1-p))^x
            // =propto (1-p)^n (1 + r p/(1-p))^n
            // =propto (1-p + r p)^n
            return new Poisson(1 - p + sample.Rate * p, 0);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditional(double, Discrete, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(double p, Discrete trialCount, Discrete result)
        {
            if (trialCount.IsPointMass)
                return SampleAverageConditional(p, trialCount.Point, result);
            // result must range from 0 to nMax
            if (result.Dimension < trialCount.Dimension)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < n.Dimension (" + trialCount.Dimension + ")");
            Vector probs = result.GetWorkspace();
            double logp = Math.Log(p);
            double log1minusp = Math.Log(1 - p);
            // p(sample=k) = sum_(n>=k) p(n) nchoosek(n,k) p^k (1-p)^(n-k)
            //             = (p/(1-p))^k 1/k! sum_(n>=k) p(n) n!/(n-k)! (1-p)^n
            for (int k = 0; k < result.Dimension; k++)
            {
                double s = 0.0;
                for (int n = k; n < trialCount.Dimension; n++)
                {
                    s += trialCount[n] * Math.Exp(MMath.ChooseLn(n, k) + n * log1minusp);
                }
                probs[k] = Math.Exp(k * (logp - log1minusp)) * s;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="SampleAverageConditional(Beta, Discrete, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(Beta p, Discrete trialCount, Discrete result)
        {
            if (p.IsPointMass)
                return SampleAverageConditional(p.Point, trialCount, result);
            if (trialCount.IsPointMass)
                return SampleAverageConditional(p, trialCount.Point, result);
            // result must range from 0 to nMax
            if (result.Dimension < trialCount.Dimension)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < n.Dimension (" + trialCount.Dimension + ")");
            Vector probs = result.GetWorkspace();
            double a = p.TrueCount;
            double b = p.FalseCount;
            // p(sample=k) = sum_(n>=k) p(n) nchoosek(n,k) p^k (1-p)^(n-k)
            //             = (p/(1-p))^k 1/k! sum_(n>=k) p(n) n!/(n-k)! (1-p)^n
            for (int k = 0; k < result.Dimension; k++)
            {
                double s = 0.0;
                for (int n = k; n < trialCount.Dimension; n++)
                {
                    s += trialCount[n] * Math.Exp(MMath.ChooseLn(n, k) + MMath.GammaLn(b + n - k) - MMath.GammaLn(a + b + n));
                }
                probs[k] = Math.Exp(MMath.GammaLn(a + k)) * s;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageConditional(Discrete, double, Discrete)"]/*'/>
        public static Discrete TrialCountAverageConditional(Discrete sample, double p, Discrete result)
        {
            if (sample.IsPointMass)
                return TrialCountAverageConditional(sample.Point, p, result);
            // n must range from 0 to sampleMax
            if (result.Dimension < sample.Dimension)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < sample.Dimension (" + sample.Dimension + ")");
            Vector probs = result.GetWorkspace();
            double logp = Math.Log(p);
            double log1minusp = Math.Log(1 - p);
            // p(n) = sum_(k<=n) p(k) nchoosek(n,k) p^k (1-p)^(n-k)
            for (int n = 0; n < result.Dimension; n++)
            {
                double s = 0.0;
                for (int k = 0; k <= n; k++)
                {
                    s += sample[k] * Math.Exp(MMath.ChooseLn(n, k) + k * (logp - log1minusp));
                }
                probs[n] = Math.Exp(n * log1minusp) * s;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageConditional(Discrete, Beta, Discrete)"]/*'/>
        public static Discrete TrialCountAverageConditional(Discrete sample, Beta p, Discrete result)
        {
            if (p.IsPointMass)
                return TrialCountAverageConditional(sample, p.Point, result);
            if (sample.IsPointMass)
                return TrialCountAverageConditional(sample.Point, p, result);
            // n must range from 0 to sampleMax
            if (result.Dimension < sample.Dimension)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < sample.Dimension (" + sample.Dimension + ")");
            Vector probs = result.GetWorkspace();
            double a = p.TrueCount;
            double b = p.FalseCount;
            // p(n) = sum_(k<=n) p(k) nchoosek(n,k) p^k (1-p)^(n-k)
            for (int n = 0; n < result.Dimension; n++)
            {
                double s = 0.0;
                for (int k = 0; k <= n; k++)
                {
                    s += sample[k] * Math.Exp(MMath.ChooseLn(n, k) + MMath.GammaLn(a + k) + MMath.GammaLn(b + n - k));
                }
                probs[n] = Math.Exp(-MMath.GammaLn(a + b + n)) * s;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageConditional(int, double, Discrete)"]/*'/>
        public static Discrete TrialCountAverageConditional(int sample, double p, Discrete result)
        {
            Vector probs = result.GetWorkspace();
            double logp = Math.Log(p);
            double log1minusp = Math.Log(1 - p);
            // p(n) = nchoosek(n,k) p^k (1-p)^(n-k)
            for (int n = 0; n < result.Dimension; n++)
            {
                if (n < sample)
                    probs[n] = double.NegativeInfinity;
                else
                    probs[n] = MMath.ChooseLn(n, sample) + sample * logp + (n - sample) * log1minusp;
            }
            double max = probs.Max();
            probs.SetToFunction(probs, logprob => Math.Exp(logprob - max));
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageConditional(int, Beta, Discrete)"]/*'/>
        public static Discrete TrialCountAverageConditional(int sample, Beta p, Discrete result)
        {
            if (p.IsPointMass)
                return TrialCountAverageConditional(sample, p.Point, result);
            Vector probs = result.GetWorkspace();
            double a = p.TrueCount;
            double b = p.FalseCount;
            // p(n) = nchoosek(n,k) p^k (1-p)^(n-k)
            for (int n = 0; n < result.Dimension; n++)
            {
                if (n < sample)
                    probs[n] = double.NegativeInfinity;
                else
                    probs[n] = MMath.ChooseLn(n, sample) + MMath.GammaLn(b + n - sample) - MMath.GammaLn(a + b + n);
            }
            double max = probs.Max();
            probs.SetToFunction(probs, logp => Math.Exp(logp - max));
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageConditional(int, int)"]/*'/>
        public static Beta PAverageConditional(int sample, int trialCount)
        {
            if (sample > trialCount)
                return Beta.Uniform();
            // sample ranges from 0 to trialCount.
            return new Beta(sample + 1, trialCount - sample + 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageConditional(int, Discrete, Beta)"]/*'/>
        public static Beta PAverageConditional(int sample, Discrete trialCount, [NoInit] Beta p)
        {
            // Z = sum_n q(n) nchoosek(n,k) int p^k (1-p)^(n-k) p^(a-1) (1-p)^(b-1) dp
            //double logZ = Double.NegativeInfinity;
            BetaEstimator est = new BetaEstimator();
            double offset = 0;
            bool needOffset = true;
            for (int n = 0; n < trialCount.Dimension; n++)
            {
                double logWeight = trialCount.GetLogProb(n) + LogAverageFactor(sample, p, n);
                if (double.IsNegativeInfinity(logWeight))
                    continue;
                if (needOffset)
                {
                    offset = logWeight;
                    needOffset = false;
                }
                Beta post = new Beta(p.TrueCount + sample, p.FalseCount + n - sample);
                est.Add(post, Math.Exp(logWeight - offset));
            }
            Beta result = est.GetDistribution(new Beta());
            result.SetToRatio(result, p, true);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageConditional(Discrete, int)"]/*'/>
        public static Beta PAverageConditional(Discrete sample, int trialCount)
        {
            if (sample.IsPointMass)
                return PAverageConditional(sample.Point, trialCount);
            throw new NotSupportedException("Random sample is not yet implemented");
        }


        //-- VMP -----------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="AverageLogFactor(int, double, int)"]/*'/>
        public static double AverageLogFactor(int sample, double p, int trialCount)
        {
            return LogAverageFactor(sample, p, trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="AverageLogFactor(int, Beta, int)"]/*'/>
        public static double AverageLogFactor(int sample, Beta p, int trialCount)
        {
            double eLogP, eLogOneMinusP;
            p.GetMeanLogs(out eLogP, out eLogOneMinusP);
            return MMath.ChooseLn(trialCount, sample) + sample * eLogP + (trialCount - sample) * eLogOneMinusP;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="AverageLogFactor(int, Beta, Discrete)"]/*'/>
        public static double AverageLogFactor(int sample, Beta p, Discrete trialCount)
        {
            double logZ = 0;
            for (int n = 0; n < trialCount.Dimension; n++)
            {
                logZ += trialCount[n] * AverageLogFactor(sample, p, n);
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="AverageLogFactor(int, double, Discrete)"]/*'/>
        public static double AverageLogFactor(int sample, double p, Discrete trialCount)
        {
            double logZ = 0;
            for (int n = 0; n < trialCount.Dimension; n++)
            {
                logZ += trialCount[n] * AverageLogFactor(sample, p, n);
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageLogarithm(int, int)"]/*'/>
        public static Beta PAverageLogarithm(int sample, int trialCount)
        {
            return PAverageConditional(sample, trialCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageLogarithm(int, Discrete)"]/*'/>
        public static Beta PAverageLogarithm(int sample, Discrete trialCount)
        {
            double trialCountMean = trialCount.GetMean();
            if (trialCountMean < sample)
                return Beta.Uniform();
            // log f(sample,n,p) = sample*log(p) + (n-sample)*log(1-p) + const.
            return new Beta(sample + 1, trialCountMean - sample + 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageLogarithm(Discrete, int)"]/*'/>
        public static Beta PAverageLogarithm(Discrete sample, int trialCount)
        {
            // log f(sample,n,p) = sample*log(p) + (n-sample)*log(1-p) + const.
            double mSample = sample.GetMean();
            if (trialCount < mSample)
                return Beta.Uniform();
            return new Beta(mSample + 1, trialCount - mSample + 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="PAverageLogarithm(Discrete, Discrete)"]/*'/>
        public static Beta PAverageLogarithm(Discrete sample, Discrete trialCount)
        {
            // log f(sample,n,p) = sample*log(p) + (n-sample)*log(1-p) + const.
            double mSample = sample.GetMean();
            double trialCountMean = trialCount.GetMean();
            if (trialCountMean < mSample)
                return Beta.Uniform();
            return new Beta(mSample + 1, trialCountMean - mSample + 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageLogarithm(int, double, Discrete)"]/*'/>
        public static Discrete TrialCountAverageLogarithm(int sample, double p, Discrete result)
        {
            return TrialCountAverageConditional(sample, p, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BinomialOp"]/message_doc[@name="TrialCountAverageLogarithm(int, Beta, Discrete)"]/*'/>
        public static Discrete TrialCountAverageLogarithm(int sample, Beta p, Discrete result)
        {
            if (p.IsPointMass)
                return TrialCountAverageConditional(sample, p.Point, result);
            Vector probs = result.GetWorkspace();
            double eLogP, eLogOneMinusP;
            p.GetMeanLogs(out eLogP, out eLogOneMinusP);
            // p(n) = nchoosek(n,k) p^k (1-p)^(n-k)
            for (int n = 0; n < result.Dimension; n++)
            {
                if (n < sample)
                    probs[n] = double.NegativeInfinity;
                else
                    probs[n] = MMath.ChooseLn(n, sample) + sample * eLogP + (n - sample) * eLogOneMinusP;
            }
            double max = probs.Max();
            probs.SetToFunction(probs, logp => Math.Exp(logp - max));
            result.SetProbs(probs);
            return result;
        }
    }
}
