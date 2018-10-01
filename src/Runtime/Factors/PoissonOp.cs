// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Poisson", typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class PoissonOp
    {
        public static bool ForceProper;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogAverageFactor(int, double)"]/*'/>
        public static double LogAverageFactor(int sample, double mean)
        {
            return SampleAverageConditional(mean).GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogEvidenceRatio(int, double)"]/*'/>
        public static double LogEvidenceRatio(int sample, double mean)
        {
            return LogAverageFactor(sample, mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="AverageLogFactor(int, double)"]/*'/>
        public static double AverageLogFactor(int sample, double mean)
        {
            return LogAverageFactor(sample, mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogAverageFactor(Poisson, double, Poisson)"]/*'/>
        public static double LogAverageFactor(Poisson sample, double mean, [Fresh] Poisson to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogEvidenceRatio(Poisson, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Poisson sample, double mean)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogAverageFactor(int, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(int sample, [SkipIfUniform] Gamma mean, [Fresh] Gamma to_mean)
        {
            return to_mean.GetLogAverageOf(mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogEvidenceRatio(int, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio(int sample, Gamma mean, [Fresh] Gamma to_mean)
        {
            return LogAverageFactor(sample, mean, to_mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogAverageFactor(Poisson, Gamma)"]/*'/>
        public static double LogAverageFactor(Poisson sample, Gamma mean)
        {
            if (sample.IsUniform())
                return 0;
            if (sample.IsPointMass)
                return LogAverageFactor(sample.Point, mean, MeanAverageConditional(sample.Point));
            if (sample.Precision != 0)
                throw new NotImplementedException("sample.Precision != 0 is not implemented");
            return -mean.Shape * Math.Log(1 + (1 - sample.Rate) / mean.Rate) - sample.GetLogNormalizer();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="LogEvidenceRatio(Poisson, Gamma, Poisson)"]/*'/>
        public static double LogEvidenceRatio(Poisson sample, Gamma mean, Poisson to_sample)
        {
            return LogAverageFactor(sample, mean) - to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="MeanAverageConditional(int)"]/*'/>
        public static Gamma MeanAverageConditional(int sample)
        {
            // p(mean) = mean^sample exp(-mean)/Gamma(sample+1)
            return new Gamma(sample + 1, 1);
        }

        private const string NotSupportedMessage =
            "A Poisson factor with unobserved output is not yet implemented for Expectation Propagation.  Try using Variational Message Passing.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="MeanAverageConditional(Poisson, Gamma)"]/*'/>
        public static Gamma MeanAverageConditional([SkipIfUniform] Poisson sample, [Proper] Gamma mean)
        {
            if (sample.IsUniform())
                return Gamma.Uniform();
            if (sample.IsPointMass)
                return MeanAverageConditional(sample.Point);
            if (sample.Precision != 0)
                throw new NotImplementedException("sample.Precision != 0 is not implemented");
            // Z = int_m sum_x r^x m^x exp(-m)/x! q(m)
            //   = int_m exp(rm -m) q(m)
            //   = (1 + (1-r)/b)^(-a)
            // logZ = -a log(1 + (1-r)/b)
            // alpha = -b dlogZ/db
            //       = -a(1-r)/(b + (1-r))
            // beta = -b dalpha/db
            //      = -ba(1-r)/(b + (1-r))^2
            double omr = 1 - sample.Rate;
            double denom = 1 / (mean.Rate + omr);
            double alpha = -mean.Shape * omr * denom;
            double beta = mean.Rate * alpha * denom;
            return GaussianOp.GammaFromAlphaBeta(mean, alpha, beta, ForceProper);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="SampleAverageConditional(double)"]/*'/>
        public static Poisson SampleAverageConditional(double mean)
        {
            return new Poisson(mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="SampleAverageConditional(Poisson, Gamma)"]/*'/>
        public static Poisson SampleAverageConditional(Poisson sample, [Proper] Gamma mean)
        {
            if (mean.IsPointMass)
                return SampleAverageConditional(mean.Point);
            if (sample.IsPointMass)
                return new Poisson(mean.GetMean());
            if (sample.Precision != 0)
                throw new NotImplementedException("sample.Precision != 0 is not implemented");
            // posterior mean of x is r dlogZ/dr = ar/(b + 1-r)
            // want to choose m such that rm = above
            return new Poisson(mean.Shape / (mean.Rate + 1 - sample.Rate));
        }

        //-- VMP -------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="AverageLogFactor(int, Gamma)"]/*'/>
        public static double AverageLogFactor(int sample, [Proper] Gamma mean)
        {
            return sample * mean.GetMeanLog() - MMath.GammaLn(sample + 1) - mean.GetMean();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="AverageLogFactor(Poisson, Gamma)"]/*'/>
        public static double AverageLogFactor(Poisson sample, [Proper] Gamma mean)
        {
            return sample.GetMean() * mean.GetMeanLog() - sample.GetMeanLogFactorial() - mean.GetMean();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="AverageLogFactor(Poisson, double)"]/*'/>
        public static double AverageLogFactor(Poisson sample, double mean)
        {
            return sample.GetMean() * mean - sample.GetMeanLogFactorial() - mean;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="MeanAverageLogarithm(int)"]/*'/>
        public static Gamma MeanAverageLogarithm(int sample)
        {
            return MeanAverageConditional(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="MeanAverageLogarithm(Poisson)"]/*'/>
        public static Gamma MeanAverageLogarithm([Proper] Poisson sample)
        {
            // p(mean) = exp(E[sample]*log(mean) - mean - E[log(sample!)])
            return new Gamma(sample.GetMean() + 1, 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="SampleAverageLogarithm(Gamma)"]/*'/>
        public static Poisson SampleAverageLogarithm([Proper] Gamma mean)
        {
            return new Poisson(Math.Exp(mean.GetMeanLog()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PoissonOp"]/message_doc[@name="SampleAverageLogarithm(double)"]/*'/>
        public static Poisson SampleAverageLogarithm(double mean)
        {
            return SampleAverageConditional(mean);
        }
    }
}
