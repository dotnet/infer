// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/doc/*'/>
    /// <remarks>
    /// Performs KL minimization using gradient matching, a distributed gradient descent algorithm. 
    /// </remarks>
    [FactorMethod(typeof(Factor), "BernoulliFromLogOdds")]
    [Quality(QualityBand.Experimental)]
    public static class BernoulliFromLogOddsOp
    {
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogOddsAverageConditional(bool, Gaussian)"]/*'/>
        public static Gaussian LogOddsAverageConditional(bool sample, [SkipIfUniform] Gaussian logOdds)
        {
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            double s = sample ? 1 : -1;
            m *= s;
            // catch cases when sigma0 would evaluate to 0
            if (m + 1.5 * v < -38)  // this check catches sigma0=0 when v <= 200
            {
                double beta2 = Math.Exp(m + 1.5 * v);
                return Gaussian.FromMeanAndVariance(s * (m + v), v * (1 - v * beta2)) / logOdds;
            }
            else if (m + v < 0)  
            {
                // Factor out exp(m+v/2) in the following formulas:
                // sigma(m,v) = exp(m+v/2)(1-sigma(m+v,v))
                // sigma'(m,v) = d/dm sigma(m,v) = exp(m+v/2)(1-sigma(m+v,v) - sigma'(m+v,v))
                // sigma''(m,v) = d/dm sigma'(m,v) = exp(m+v/2)(1-sigma(m+v,v) - 2 sigma'(m+v,v) - sigma''(m+v,v))
                // This approach is always safe if sigma(-m-v,v)>0, which is guaranteed by m+v<0  
                double sigma0 = MMath.LogisticGaussian(-m - v, v);
                double sd = MMath.LogisticGaussianDerivative(m + v, v);
                double sigma1 = sigma0 - sd;
                double sigma2 = sigma1 - sd - MMath.LogisticGaussianDerivative2(m + v, v);
                double alpha = sigma1 / sigma0; // 1 - sd/sigma0
                if (Double.IsNaN(alpha))
                    throw new Exception("alpha is NaN");
                double beta = alpha * alpha - sigma2 / sigma0;
                if (Double.IsNaN(beta))
                    throw new Exception("beta is NaN");
                return GaussianOp.GaussianFromAlphaBeta(logOdds, s * alpha, beta, ForceProper);
            }
            else if (v > 1488 && m < 0)
            {
                double sigma0 = MMath.LogisticGaussianRatio(m, v, 0);
                double sigma1 = MMath.LogisticGaussianRatio(m, v, 1);
                double sigma2 = MMath.LogisticGaussianRatio(m, v, 2);
                double alpha, beta;
                alpha = sigma1 / sigma0;
                if (Double.IsNaN(alpha))
                    throw new Exception("alpha is NaN");
                beta = alpha * alpha - sigma2 / sigma0;
                if (Double.IsNaN(beta))
                    throw new Exception("beta is NaN");
                return GaussianOp.GaussianFromAlphaBeta(logOdds, s * alpha, beta, ForceProper);
            }
            else
            {
                // the following code only works when sigma0 > 0
                // sigm0=0 can only happen here if v > 1488
                double sigma0 = MMath.LogisticGaussian(m, v);
                double sigma1 = MMath.LogisticGaussianDerivative(m, v);
                double sigma2 = MMath.LogisticGaussianDerivative2(m, v);
                double alpha, beta;
                alpha = sigma1 / sigma0;
                if (Double.IsNaN(alpha))
                    throw new Exception("alpha is NaN");
                if (m + 2 * v < -19)
                {
                    beta = Math.Exp(3 * m + 2.5 * v) / (sigma0 * sigma0);
                }
                else
                {
                    //beta = (sigma1*sigma1 - sigma2*sigma0)/(sigma0*sigma0);
                    beta = alpha * alpha - sigma2 / sigma0;
                }
                if (Double.IsNaN(beta))
                    throw new Exception("beta is NaN");
                return GaussianOp.GaussianFromAlphaBeta(logOdds, s * alpha, beta, ForceProper);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogOddsAverageConditional(Bernoulli, Gaussian)"]/*'/>
        public static Gaussian LogOddsAverageConditional(Bernoulli sample, [SkipIfUniform] Gaussian logOdds)
        {
            Gaussian toLogOddsT = LogOddsAverageConditional(true, logOdds);
            double logWeightT = LogAverageFactor(true, logOdds) + sample.GetLogProbTrue();
            Gaussian toLogOddsF = LogOddsAverageConditional(false, logOdds);
            double logWeightF = LogAverageFactor(false, logOdds) + sample.GetLogProbFalse();
            double maxWeight = Math.Max(logWeightT, logWeightF);
            logWeightT -= maxWeight;
            logWeightF -= maxWeight;
            Gaussian result = new Gaussian();
            result.SetToSum(Math.Exp(logWeightT), toLogOddsT * logOdds, Math.Exp(logWeightF), toLogOddsF * logOdds);
            result.SetToRatio(result, logOdds, ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogAverageFactor(bool, double)"]/*'/>
        public static double LogAverageFactor(bool sample, double logOdds)
        {
            return MMath.LogisticLn(sample ? logOdds : -logOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogEvidenceRatio(bool, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(bool sample, Gaussian logOdds)
        {
            return LogAverageFactor(sample, logOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogEvidenceRatio(bool, double)"]/*'/>
        public static double LogEvidenceRatio(bool sample, double logOdds)
        {
            return LogAverageFactor(sample, logOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli sample)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="LogAverageFactor(bool, Gaussian)"]/*'/>
        public static double LogAverageFactor(bool sample, [Proper] Gaussian logOdds)
        {
            if (logOdds.IsPointMass)
                return LogAverageFactor(sample, logOdds.Point);
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            return Math.Log(MMath.LogisticGaussian(sample ? m : -m, v));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="SampleAverageConditional(Gaussian)"]/*'/>
        public static Bernoulli SampleAverageConditional([Proper] Gaussian logOdds)
        {
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            return new Bernoulli(MMath.LogisticGaussian(m, v));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="AverageLogFactor(bool, double)"]/*'/>
        public static double AverageLogFactor(bool sample, double logOdds)
        {
            if (sample)
                return MMath.LogisticLn(logOdds);
            else
                return MMath.LogisticLn(-logOdds);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="AverageLogFactor(Bernoulli, double)"]/*'/>
        public static double AverageLogFactor(Bernoulli sample, double logOdds)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, logOdds);
            // probTrue*log(sigma(logOdds)) + probFalse*log(sigma(-logOdds))
            // = -log(1+exp(-logOdds)) + probFalse*(-logOdds)
            // = probTrue*logOdds - log(1+exp(logOdds))
            if (logOdds >= 0)
            {
                double probFalse = sample.GetProbFalse();
                return -probFalse * logOdds - MMath.Log1PlusExp(-logOdds);
            }
            else
            {
                double probTrue = sample.GetProbTrue();
                return probTrue * logOdds - MMath.Log1PlusExp(logOdds);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="AverageLogFactor(bool, Gaussian)"]/*'/>
        public static double AverageLogFactor(bool sample, [Proper, SkipIfUniform] Gaussian logOdds)
        {
            // f(sample,logOdds) = exp(sample*logOdds)/(1 + exp(logOdds))
            // log f(sample,logOdds) = sample*logOdds - log(1 + exp(logOdds))
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            return (sample ? 1.0 : 0.0) * m - MMath.Log1PlusExpGaussian(m, v);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="AverageLogFactor(Bernoulli, Gaussian)"]/*'/>
        public static double AverageLogFactor(Bernoulli sample, [Proper, SkipIfUniform] Gaussian logOdds)
        {
            // f(sample,logOdds) = exp(sample*logOdds)/(1 + exp(logOdds))
            // log f(sample,logOdds) = sample*logOdds - log(1 + exp(logOdds))
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            return sample.GetProbTrue() * m - MMath.Log1PlusExpGaussian(m, v);
        }

        // Calculate \int f(x) dx over the whole real line. Uses a change of variable
        // x=cot(t) which maps the real line onto [0,pi] and use trapezium rule
        private static double ClenshawCurtisQuadrature(Converter<double, double> f, int CCFactor, int numIntervals)
        {
            double intervalWidth = Math.PI / (double)numIntervals;
            double sum = 0;
            for (double x = intervalWidth; x < Math.PI; x += intervalWidth)
            {
                double sinX = Math.Sin(x);
                sum += f(CCFactor / Math.Tan(x)) / (sinX * sinX);
            }
            return CCFactor * intervalWidth * sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="SampleAverageLogarithm(Gaussian)"]/*'/>
        public static Bernoulli SampleAverageLogarithm([Proper] Gaussian logOdds)
        {
            return Bernoulli.FromLogOdds(logOdds.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BernoulliFromLogOddsOp"]/message_doc[@name="SampleAverageLogarithm(double)"]/*'/>
        public static Bernoulli SampleAverageLogarithm(double logOdds)
        {
            return Bernoulli.FromLogOdds(logOdds);
        }

        /// <summary>
        /// Gradient matching VMP message from factor to logOdds variable
        /// </summary>
        /// <param name="sample">Constant value for 'sample'.</param>
        /// <param name="logOdds">Incoming message. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="to_LogOdds">Previous message sent, used for damping</param>
        /// <returns>The outgoing VMP message.</returns>
        /// <remarks><para>
        /// The outgoing message is the Gaussian approximation to the factor which results in the 
        /// same derivatives of the KL(q||p) divergence with respect to the parameters of the posterior
        /// as if the true factor had been used.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="logOdds"/> is not a proper distribution</exception>
        public static Gaussian LogOddsAverageLogarithm(bool sample, [Proper, SkipIfUniform] Gaussian logOdds, Gaussian to_LogOdds)
        {
            double m, v; // prior mean and variance
            double s = sample ? 1 : -1;
            logOdds.GetMeanAndVariance(out m, out v);
            // E = \int q log f dx
            // Match gradients
            double dEbydm = s * MMath.LogisticGaussian(-s * m, v);
            double dEbydv = -.5 * MMath.LogisticGaussianDerivative(s * m, v);
            double prec = -2.0 * dEbydv;
            double meanTimesPrec = m * prec + dEbydm;
            Gaussian result = Gaussian.FromNatural(meanTimesPrec, prec);
            double step = Rand.Double() * 0.5; // random damping helps convergence, especially with parallel updates
            if (step != 1.0)
            {
                result.Precision = step * result.Precision + (1 - step) * to_LogOdds.Precision;
                result.MeanTimesPrecision = step * result.MeanTimesPrecision + (1 - step) * to_LogOdds.MeanTimesPrecision;
            }
            return result;
        }

        /// <summary>
        /// Gradient matching VMP message from factor to logOdds variable
        /// </summary>
        /// <param name="sample">Incoming message from 'sample'.</param>
        /// <param name="logOdds">Incoming message. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="result">Previous message sent, used for damping</param>
        /// <returns>The outgoing VMP message.</returns>
        /// <remarks><para>
        /// The outgoing message is the Gaussian approximation to the factor which results in the 
        /// same derivatives of the KL(q||p) divergence with respect to the parameters of the posterior
        /// as if the true factor had been used.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="logOdds"/> is not a proper distribution</exception>
        public static Gaussian LogOddsAverageLogarithm(Bernoulli sample, [Proper, SkipIfUniform] Gaussian logOdds, Gaussian result)
        {
            if (sample.IsUniform())
                return Gaussian.Uniform();
            throw new NotImplementedException("BernoulliFromLogOdds with non-observed output is not yet implemented");
        }
    }
}
