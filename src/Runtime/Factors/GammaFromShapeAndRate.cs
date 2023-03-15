// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "GammaFromShapeAndRate", Default = false)]
    [Quality(QualityBand.Mature)]
    public class GammaFromShapeAndRateOp : GammaFromShapeAndRateOpBase
    {
        public static bool ForceProper = true;
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/doc/*'/>
    public class GammaFromShapeAndRateOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double shape, double rate)
        {
            return Gamma.GetLogProb(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double shape, double rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(Gamma sample, [Fresh] Gamma to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogEvidenceRatio(Gamma, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gamma sample, double shape, double rate)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(double, double, Gamma)"]/*'/>
        public static double LogAverageFactor(double sample, double shape, Gamma rate)
        {
            // f(y,a,b) = y^(a-1) b^a/Gamma(a) exp(-by) = y^(-2) Gamma(a+1)/Gamma(a) Ga(b; a+1, y)
            Gamma to_rate = RateAverageConditional(sample, shape);
            return rate.GetLogAverageOf(to_rate) - 2 * Math.Log(sample) + Math.Log(shape);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogEvidenceRatio(double, double, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double sample, double shape, Gamma rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Gamma SampleAverageConditional(double shape, double rate)
        {
            return Gamma.FromShapeAndRate(shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="RateAverageConditional(double, double)"]/*'/>
        public static Gamma RateAverageConditional(double sample, double shape)
        {
            return Gamma.FromShapeAndRate(shape + 1, sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(Gamma, double, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gamma sample, double shape, double rate)
        {
            return sample.GetLogAverageOf(Gamma.FromShapeAndRate(shape, rate));
        }

        private const string NotSupportedMessage = "Expectation Propagation does not support Gamma variables with stochastic shape";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(Gamma, Gamma, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor([SkipIfUniform] Gamma sample, [SkipIfUniform] Gamma shape, [SkipIfUniform] Gamma rate)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(Gamma, Gamma, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor([SkipIfUniform] Gamma sample, [SkipIfUniform] Gamma shape, double rate)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(double, Gamma, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor(double sample, [SkipIfUniform] Gamma shape, [SkipIfUniform] Gamma rate)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="LogAverageFactor(double, Gamma, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor(double sample, [SkipIfUniform] Gamma shape, double rate)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageConditional(Gamma, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma SampleAverageConditional([SkipIfUniform] Gamma shape, [SkipIfUniform] Gamma rate)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gamma SampleAverageConditionalInit()
        {
            return Gamma.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageConditional(Gamma, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma SampleAverageConditional([SkipIfUniform] Gamma shape, double rate)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="RateAverageConditional(Gamma, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma RateAverageConditional([SkipIfUniform] Gamma sample, [SkipIfUniform] Gamma shape)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="ShapeAverageConditional(Gamma, Gamma, double, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma ShapeAverageConditional([SkipIfUniform] Gamma sample, [SkipIfUniform] Gamma shape, double rate, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="ShapeAverageConditional(double, Gamma, Gamma, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma ShapeAverageConditional(double sample, [SkipIfUniform] Gamma shape, [SkipIfUniform] Gamma rate, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="ShapeAverageConditional(double, Gamma, double, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma ShapeAverageConditional(double sample, [SkipIfUniform] Gamma shape, double rate, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="ShapeAverageConditional(Gamma, Gamma, Gamma, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma ShapeAverageConditional([SkipIfUniform] Gamma sample, [SkipIfUniform] Gamma shape, [SkipIfUniform] Gamma rate, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        // VMP ----------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double shape, double rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="AverageLogFactor(Gamma, Gamma, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static double AverageLogFactor([Proper] Gamma sample, [Proper] Gamma shape, [Proper] Gamma rate)
        {
            double a = shape.GetMean();
            return (a - 1) * sample.GetMeanLog() + a * rate.GetMeanLog() - rate.GetMean() * sample.GetMean() - ELogGamma(shape);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="AverageLogFactor(Gamma, double, double)"]/*'/>
        public static double AverageLogFactor([Proper] Gamma sample, double shape, double rate)
        {
            Gamma to_sample = SampleAverageLogarithm(shape, rate);
            return sample.GetAverageLog(to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageLogarithm(Gamma, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma SampleAverageLogarithm([SkipIfUniform] Gamma shape, [SkipIfUniform] Gamma rate)
        {
            return Gamma.FromShapeAndRate(shape.GetMean(), rate.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageLogarithm(Gamma, double)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma SampleAverageLogarithm([SkipIfUniform] Gamma shape, double rate)
        {
            return Gamma.FromShapeAndRate(shape.GetMean(), rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageLogarithm(double, Gamma)"]/*'/>
        public static Gamma SampleAverageLogarithm(double shape, [SkipIfUniform] Gamma rate)
        {
            return Gamma.FromShapeAndRate(shape, rate.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Gamma SampleAverageLogarithm(double shape, double rate)
        {
            return Gamma.FromShapeAndRate(shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="RateAverageLogarithm(Gamma, Gamma)"]/*'/>
        public static Gamma RateAverageLogarithm([SkipIfUniform] Gamma sample, [SkipIfUniform] Gamma shape)
        {
            // factor = rate^shape exp(-y*rate)
            // log(factor) = shape*log(rate) - y*rate
            // E[log(factor)] = E[shape]*log(rate) - E[y]*rate
            return Gamma.FromShapeAndRate(shape.GetMean() + 1, sample.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="RateAverageLogarithm(double, Gamma)"]/*'/>
        public static Gamma RateAverageLogarithm(double sample, [SkipIfUniform] Gamma shape)
        {
            return Gamma.FromShapeAndRate(shape.GetMean() + 1, sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="RateAverageLogarithm(Gamma, double)"]/*'/>
        public static Gamma RateAverageLogarithm([SkipIfUniform] Gamma sample, double shape)
        {
            return Gamma.FromShapeAndRate(shape + 1, sample.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="RateAverageLogarithm(double, double)"]/*'/>
        public static Gamma RateAverageLogarithm(double sample, double shape)
        {
            return Gamma.FromShapeAndRate(shape + 1, sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOpBase"]/message_doc[@name="ShapeAverageLogarithm(Gamma, Gamma, Gamma, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma ShapeAverageLogarithm([SkipIfUniform] Gamma sample, [Proper] Gamma shape, [Proper] Gamma rate, Gamma to_shape)
        {
            //if (to_shape.IsUniform())
            //    to_shape.SetShapeAndRate(1, 1);
            //if (y.Rate == 0 || Double.IsInfinity(y.Rate))
            //    y.SetShapeAndRate(1, 1);
            //if (rate.Rate==0 || Double.IsInfinity(rate.Rate))
            //    rate.SetShapeAndRate(1, 1);
            double ElogYR = sample.GetMeanLog() + rate.GetMeanLog();
            double a, b;
            a = shape.Shape;
            b = shape.Rate;

            // Find required expectations using quadrature
            Vector gradElogGamma = CalculateDerivatives(shape);

            // Calculate gradients
            Vector gradS = -gradElogGamma;
            gradS[0] += ElogYR / b;
            gradS[1] += -a * ElogYR / (b * b);

            // Calculate the required message to match these gradients
            Gamma approximateFactor = NonconjugateProjection(shape, gradS);

            double damping = 0.0;
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_shape ^ damping);
        }

        /// <summary>
        /// Project the standard VMP message onto a gradient matched Gamma message. 
        /// </summary>
        /// <param name="context">Incoming message.</param>
        /// <param name="gradS">Gradient of S=int factor(x) p(x) dx</param>
        /// <returns>Projected gamma message</returns>
        internal static Gamma NonconjugateProjection(Gamma context, Vector gradS)
        {
            Matrix mat = new Matrix(2, 2);
            mat[0, 0] = MMath.Trigamma(context.Shape);
            mat[1, 0] = mat[0, 1] = -1 / context.Rate;
            mat[1, 1] = context.Shape / (context.Rate * context.Rate);
            Vector v = twoByTwoInverse(mat) * gradS;
            return Gamma.FromShapeAndRate(v[0] + 1, v[1]);
        }

        /// <summary>
        /// Two by two matrix inversion. 
        /// </summary>
        /// <param name="a">Matrix to invert</param>
        /// <returns>Inverted matrix</returns>
        internal static Matrix twoByTwoInverse(Matrix a)
        {
            Matrix result = new Matrix(2, 2);
            double det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0];
            result[0, 0] = a[1, 1] / det;
            result[0, 1] = -a[0, 1] / det;
            result[1, 0] = -a[1, 0] / det;
            result[1, 1] = a[0, 0] / det;
            return result;
        }

        /// <summary>
        /// Calculate derivatives of \int G(x;a,b) LogGamma(x) dx wrt (a,b)
        /// </summary>
        /// <param name="q">The Gamma distribution G(x;a,b).</param>
        /// <returns>A 2-vector containing derivatives of \int G(x;a,b) LogGamma(x) dx wrt (a,b).</returns>
        /// <remarks><para>
        ///  Calculates expectations in x=log(s) space using Gauss-Hermite quadrature. 
        ///  For each integral the behaviour as x->0 is subtracted from the integrand 
        ///  before performing quadrature to remove the singularity there. 
        /// </para></remarks>
        public static Vector CalculateDerivatives(Gamma q)
        {
            Vector gradElogGamma = Vector.Zero(2);
            // Get shape and scale of the distribution
            double a = q.Shape;
            double b = q.Rate;
            // Mean in the transformed domain
            double proposalMean = q.GetMeanLog();
            // Laplace approximation of variance in transformed domain 
            double proposalVariance = 1 / a;
            //double proposalVariance = Math.Exp(-proposalMean) / b; 

            // Quadrature coefficient
            int n = 11;
            Vector nodes = Vector.Zero(n);
            Vector weights = Vector.Zero(n);
            Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);

            double EXDigamma = 0;
            double ELogGam = 0;
            double ELogXLogGamma = 0;

            // Calculate expectations in x=log(s) space using Gauss-Hermite quadrature
            double logZ = MMath.GammaLn(a) - a * Math.Log(b);
            for (int i = 0; i < n; i++)
            {
                double x = nodes[i];
                double expx = Math.Exp(x);
                double p = a * x - b * expx - logZ - Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                p = Math.Exp(p) * weights[i];
                EXDigamma += p * (expx * MMath.Digamma(expx) + 1);
                ELogGam += p * (MMath.GammaLn(expx) + x);
                ELogXLogGamma += p * (x * MMath.GammaLn(expx) + x * x);
            }

            // Normalise and add removed components
            ELogGam = ELogGam - proposalMean;
            ELogXLogGamma = ELogXLogGamma - (MMath.Trigamma(a) + proposalMean * proposalMean);
            EXDigamma = EXDigamma - 1;

            // Calculate derivatives
            gradElogGamma[0] = ELogXLogGamma - proposalMean * ELogGam;
            gradElogGamma[1] = -1.0 / b * EXDigamma;
            //gradElogGamma[1] = (ELogGamma(q) - ELogGamma(new Gamma(a + 1, b))) * a / b; 
            return gradElogGamma;
        }

        public static Vector CalculateDerivativesTrapezoid(Gamma q)
        {
            Vector gradElogGamma = Vector.Zero(2);
            // Get shape and scale of the distribution
            double a = q.Shape;
            double b = q.Rate;
            double mean, variance;
            q.GetMeanAndVariance(out mean, out variance);
            double upperBound = 10;

            int n = 10000;
            double ELogGamma = 0, ELogXLogGamma = 0, ExDigamma = 0;
            double inc = upperBound / n;
            for (int i = 0; i < n; i++)
            {
                double x = inc * (i + 1);
                double logp = q.GetLogProb(x);
                double p = Math.Exp(logp);
                double f = p * MMath.GammaLn(x);
                ELogGamma += f;
                ELogXLogGamma += Math.Log(x) * f;
                ExDigamma += x * MMath.Digamma(x) * p;
            }
            ELogGamma *= inc;
            ELogXLogGamma *= inc;
            ExDigamma *= inc;
            gradElogGamma[0] = ELogXLogGamma + (Math.Log(b) - MMath.Digamma(a)) * ELogGamma;
            gradElogGamma[1] = -ExDigamma / b;
            return gradElogGamma;
        }

        public static Vector CalculateDerivativesNaive(Gamma q)
        {
            Vector gradElogGamma = Vector.Zero(2);
            // Get shape and scale of the distribution
            double a = q.Shape;
            double b = q.Rate;
            // Mean in the transformed domain
            double proposalMean = q.GetMeanLog();
            // Laplace approximation of variance in transformed domain 
            double proposalVariance = 1 / a;

            // Quadrature coefficient
            int n = 11;
            Vector nodes = Vector.Zero(n);
            Vector weights = Vector.Zero(n);
            Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);

            double EXDigamma = 0;
            double ELogGam = 0;
            double ELogXLogGamma = 0;

            // Calculate expectations in x=log(s) space using Gauss-Hermite quadrature
            double logZ = MMath.GammaLn(a) - a * Math.Log(b);
            for (int i = 0; i < n; i++)
            {
                double x = nodes[i];
                double expx = Math.Exp(x);
                double p = a * x - b * expx - logZ - Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                p = Math.Exp(p) * weights[i];
                EXDigamma += p * (expx * MMath.Digamma(expx));
                ELogGam += p * (MMath.GammaLn(expx));
                ELogXLogGamma += p * (x * MMath.GammaLn(expx));
            }

            // Calculate derivatives
            gradElogGamma[0] = ELogXLogGamma - proposalMean * ELogGam;
            gradElogGamma[1] = -1.0 / b * EXDigamma;
            //gradElogGamma[1] = (ELogGamma(q) - ELogGamma(new Gamma(a + 1, b))) * a / b; 
            return gradElogGamma;
        }

        /// <summary>
        /// Calculates \int G(x;a,b) LogGamma(x) dx
        /// </summary>
        /// <param name="q">G(x;a,b)</param>
        /// <returns>\int G(x;a,b) LogGamma(x) dx</returns>
        public static double ELogGamma(Gamma q)
        {
            if (q.IsPointMass)
                return MMath.GammaLn(q.Point);
            double a = q.Shape;
            double b = q.Rate;
            // Mean in the transformed domain
            double proposalMean = q.GetMeanLog();
            // Laplace approximation of variance in transformed domain 
            double proposalVariance = 1 / a;

            // Quadrature coefficient
            int n = 11;
            Vector nodes = Vector.Zero(n);
            Vector weights = Vector.Zero(n);
            Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);

            double ELogGamma = 0;
            double logZ = -a * Math.Log(b) + MMath.GammaLn(a);
            // Calculate expectations in x=log(s) space using Gauss-Hermite quadrature
            for (int i = 0; i < n; i++)
            {
                double x = nodes[i];
                double expx = Math.Exp(x);
                double p = a * x - b * expx - Gaussian.GetLogProb(x, proposalMean, proposalVariance) - logZ;
                p = Math.Exp(p + Math.Log(weights[i]));
                ELogGamma += p * (MMath.GammaLn(expx) + x);
            }

            // Add removed components
            return ELogGamma - proposalMean;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Slow"]/doc/*'/>
    [FactorMethod(typeof(Factor), "GammaFromShapeAndRate", Default = true)]
    [Quality(QualityBand.Stable)]
    public class GammaFromShapeAndRateOp_Slow : GammaFromShapeAndRateOpBase
    {
        public static int QuadratureNodeCount = 1000000;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Slow"]/message_doc[@name="LogAverageFactor(Gamma, double, Gamma)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gamma sample, double shape, [SkipIfUniform] Gamma rate)
        {
            if (rate.IsPointMass)
                return LogAverageFactor(sample, shape, rate.Point);
            if (sample.IsPointMass)
                return LogAverageFactor(sample.Point, shape, rate);
            double shape1 = shape + rate.Shape;
            double shape2 = AddShapesMinus1(shape, sample.Shape);
            double r, rmin, rmax;
            GetIntegrationBoundsForRate(sample, shape, rate, out r, out rmin, out rmax);
            if (r == 0)
                return double.PositiveInfinity;
            double logrmin = Math.Log(rmin);
            double logrmax = Math.Log(rmax);
            int n = QuadratureNodeCount;
            double inc = (logrmax - logrmin) / (n - 1);
            double logz = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                double logx = logrmin + i * inc;
                double x = Math.Exp(logx);
                double logp = shape1 * Math.Log(x) - shape2 * Math.Log(x + sample.Rate) - x * rate.Rate;
                logz = MMath.LogSumExp(logz, logp);
            }
            logz += Math.Log(inc) + MMath.GammaLn(shape2) - MMath.GammaLn(shape) - rate.GetLogNormalizer() - sample.GetLogNormalizer();
            return logz;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Slow"]/message_doc[@name="LogEvidenceRatio(Gamma, double, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma sample, double shape, [SkipIfUniform] Gamma rate, Gamma to_sample)
        {
            return LogAverageFactor(sample, shape, rate) - to_sample.GetLogAverageOf(sample);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Slow"]/message_doc[@name="SampleAverageConditional(Gamma, double, Gamma)"]/*'/>
        public static Gamma SampleAverageConditional([NoInit] Gamma sample, double shape, [SkipIfUniform] Gamma rate)
        {
            if (rate.IsPointMass)
                return SampleAverageConditional(shape, rate.Point);
            if (rate.Rate == 0)
                return Gamma.Uniform();
            if (sample.IsPointMass)
            {
                // This is a limiting case of EP where the message simply matches the derivatives of the factor at a point.
                // f(x) = int_r x^(s-1) r^s exp(-rx) p(r) dr
                //      = x^(s-1) / (x + rr)^(s+rs)
                // log f(x) = (s-1)*log(x) - (s+rs)*log(x+rr)
                // dlogf = (s-1)/x - (s+rs)/(x+rr)
                // ddlogf = -(s-1)/x^2 + (s+rs)/(x+rr)^2
                double x = sample.Point;
                double shape2 = shape + rate.Shape;
                double xrr = x + rate.Rate;
                double shape2OverXrr = shape2 / xrr;
                double shape2TimesXOverXrr = shape2 * x / xrr;
                if (x == 0)
                {
                    if (shape == 1)
                    {
                        double dlogf = -shape2OverXrr;
                        double xdlogf = -shape2TimesXOverXrr;
                        double xxddlogf = shape2TimesXOverXrr * x / xrr;
                        return Gamma.FromDerivatives(x, dlogf, xdlogf, xxddlogf, GammaFromShapeAndRateOp.ForceProper);
                    }
                    else
                    {
                        // a = -x*x*ddLogP
                        // b = a / x - dLogP
                        return Gamma.FromShapeAndRate(shape, shape2OverXrr);
                    }
                }
                else
                {
                    double shapeMinus1OverX = (shape - 1) / x;
                    double dlogf = shapeMinus1OverX - shape2OverXrr;
                    double xdlogf = shape - 1 - shape2TimesXOverXrr;
                    double xxddlogf = -(shape - 1) + shape2TimesXOverXrr * x / xrr;
                    return Gamma.FromDerivatives(x, dlogf, xdlogf, xxddlogf, GammaFromShapeAndRateOp.ForceProper);
                }
            }
            double sampleMean, sampleVariance;
            if (sample.Rate == 0)
            {
                // message to rate = int_x x^(ax-1) x^(s-1) r^s exp(-r*x) dx
                //                 = r^s / r^(ax-1+s) = r^(1-ax)
                double shape2 = AddShapesMinus1(shape, sample.Shape);
                Gamma ratePost = Gamma.FromShapeAndRate(rate.Shape + (1 - sample.Shape), rate.Rate);
                sampleMean = shape2 * ratePost.GetMeanInverse();
                sampleVariance = shape2 * (1 + shape2) * ratePost.GetMeanPower(-2) - sampleMean * sampleMean;
            }
            else if (true)
            {
                // quadrature over sample
                double y, ymin, ymax;
                GetIntegrationBoundsForSample(sample, shape, rate, out y, out ymin, out ymax);
                if (ymin == y || ymax == y)
                    return SampleAverageConditional(Gamma.PointMass(y), shape, rate);
                double logymin = Math.Log(ymin);
                double logymax = Math.Log(ymax);
                int n = QuadratureNodeCount;
                double inc = (logymax - logymin) / (n - 1);
                double shape1 = AddShapesMinus1(shape, sample.Shape);
                double shape2 = shape + rate.Shape;
                if (shape != 1)
                {
                    // Compute the output message directly, instead of using SetToRatio
                    // this is more accurate when the prior is sharp
                    double Z = 0;
                    // sum1 = Exf'
                    double sum1 = 0;
                    // sum2 = Ex^2f''
                    // f''/f = d(f'/f) + (f'/f)^2
                    double sum2 = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double logx = logymin + i * inc;
                        double x = Math.Exp(logx);
                        double diff = shape1 * Math.Log(x / y) - shape2 * Math.Log((x + rate.Rate) / (y + rate.Rate)) - sample.Rate * (x - y);
                        double f = Math.Exp(diff);
                        double q = x / (x + rate.Rate);
                        double xdlogf = shape - 1 - shape2 * q;
                        double xxddlogf = -(shape - 1) + shape2 * q * q + xdlogf * xdlogf;
                        Z += f;
                        sum1 += xdlogf * f;
                        sum2 += xxddlogf * f;
                    }
                    double alpha = sum1 / Z;
                    double beta = (sum1 + sum2) / Z - alpha * alpha;
                    return GaussianOp.GammaFromAlphaBeta(sample, alpha, beta, GammaFromShapeAndRateOp.ForceProper);
                }
                else
                {
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < n; i++)
                    {
                        double logx = logymin + i * inc;
                        double x = Math.Exp(logx);
                        //double shift = shape1*Math.Log(y) - shape2*Math.Log(y+rate.Rate) - y*sample.Rate;
                        //double logp = shape1*Math.Log(x) - shape2*Math.Log(x+rate.Rate) - x*sample.Rate;
                        double diff = shape1 * Math.Log(x / y) - shape2 * Math.Log((x + rate.Rate) / (y + rate.Rate)) - sample.Rate * (x - y);
                        //if (i == 0 || i == n-1) Console.WriteLine(logp-shift);
                        if ((i == 0 || i == n - 1) && (diff > -49))
                            throw new Exception("invalid integration bounds");
                        double p = Math.Exp(diff);
                        mva.Add(x, p);
                        if (double.IsNaN(mva.Variance))
                            throw new Exception();
                    }
                    sampleMean = mva.Mean;
                    sampleVariance = mva.Variance;
                }
            }
            else
            {
                // quadrature over rate
                // sampleMean = E[ shape2/(sample.Rate + r) ]
                // sampleVariance = var(shape2/(sample.Rate + r)) + E[ shape2/(sample.Rate+r)^2 ]
                //                = shape2^2*var(1/(sample.Rate + r)) + shape2*(var(1/(sample.Rate+r)) + (sampleMean/shape2)^2)
                double r, rmin, rmax;
                GetIntegrationBoundsForRate(sample, shape, rate, out r, out rmin, out rmax);
                double shape1 = shape + rate.Shape;
                double shape2 = AddShapesMinus1(shape, sample.Shape);
                double logrmin = Math.Log(rmin);
                double logrmax = Math.Log(rmax);
                int n = QuadratureNodeCount;
                double inc = (logrmax - logrmin) / (n - 1);
                MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                for (int i = 0; i < n; i++)
                {
                    double logx = logrmin + i * inc;
                    double x = Math.Exp(logx);
                    //double shift = (shape1*Math.Log(r) - r*rate.Rate) - shape2*Math.Log(r+sample.Rate);
                    //double logp = (shape1*Math.Log(x) - x*rate.Rate) - shape2*Math.Log(x+sample.Rate);
                    double diff = shape1 * Math.Log(x / r) - shape2 * Math.Log((x + sample.Rate) / (r + sample.Rate)) - rate.Rate * (x - r);
                    //if (i == 0 || i == n-1) Console.WriteLine(logp-shift);
                    if ((i == 0 || i == n - 1) && (diff > -49))
                        throw new Exception("invalid integration bounds");
                    double p = Math.Exp(diff);
                    double f = 1 / (x + sample.Rate);
                    mva.Add(f, p);
                }
                sampleMean = shape2 * mva.Mean;
                sampleVariance = shape2 * (1 + shape2) * mva.Variance + shape2 * mva.Mean * mva.Mean;
            }
            Gamma sampleMarginal = Gamma.FromMeanAndVariance(sampleMean, sampleVariance);
            Gamma result = new Gamma();
            result.SetToRatio(sampleMarginal, sample, GammaFromShapeAndRateOp.ForceProper);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException($"result is NaN.  sample={sample}, shape={shape}, rate={rate}");
            return result;
        }

        /// <summary>
        /// Get integration bounds for sample where rate is marginalized
        /// </summary>
        /// <param name="sample"></param>
        /// <param name="shape"></param>
        /// <param name="rate"></param>
        /// <param name="y">On output, the mode of the integrand</param>
        /// <param name="ymin">On output, the lower bound for integration</param>
        /// <param name="ymax">On output, the upper bound for integration</param>
        private static void GetIntegrationBoundsForSample(Gamma sample, double shape, Gamma rate, out double y, out double ymin, out double ymax)
        {
            // When we marginalize the factor over rate, we get a function of the same form as marginalizing over sample.
            // f(y,r) = y^(s-1) r^s exp(-r y)/Gamma(s)
            // q(r) f(y,r) = y^(s-1) r^(s+ar-1) exp(-r (y+br)) br^ar/Gamma(ar)/Gamma(s)
            // int over r = y^(s-1) / (y + br)^(s+ar) br^ar Gamma(s+ar)/Gamma(ar)/Gamma(s)
            // compare to r^s / (r + by)^(ay+s-1)
            GetIntegrationBoundsForRate(Gamma.FromShapeAndRate(rate.Shape + 2, rate.Rate), shape - 1, sample, out y, out ymin, out ymax);
        }

        /// <summary>
        /// Find the maximum of the function shape1*log(x) - shape2*log(x + yRate) - x*rateRate
        /// </summary>
        /// <param name="shape1"></param>
        /// <param name="shape2"></param>
        /// <param name="yRate">Must be &gt;= 0</param>
        /// <param name="rateRate"></param>
        /// <returns></returns>
        internal static double FindMaximum(double shape1, double shape2, double yRate, double rateRate)
        {
            if (shape1 < 0)
            {
                return 0;
            }
            else if (rateRate < 0)
            {
                return double.PositiveInfinity;
            }
            else if (shape2 == 0)
            {
                return shape1 / rateRate;
            }
            if (yRate < 0)
                throw new ArgumentOutOfRangeException(nameof(yRate), yRate, "yRate < 0");
            // f = shape1*log(x) - shape2*log(x+yRate) - x*rateRate
            // df = shape1/x - shape2/(x + yrate) - rateRate
            // df=0 when shape1*(x+yRate) - shape2*x - rateRate*x*(x+yRate) = 0
            // -rateRate*x^2 + (shape1-shape2-rateRate*yRate)*x + shape1*yRate = 0
            double a = -rateRate;
            double b = shape1 - shape2 - yRate * rateRate;
            double c = shape1 * yRate;
            if (a == 0)
            {
                if (b >= 0)
                    return double.PositiveInfinity;
                else
                    return -c / b;
            }
            if (b == 0)
            {
                return -Math.Sqrt(-a * c) / a;
            }
            // compute one of the roots
            double numer = (-b - Math.Sign(b) * Math.Sqrt(b * b - 4 * a * c)) / 2;
            double r0 = numer / a;
            if (r0 < 0)
            {
                // choose the other root
                // since a<0 and c>0, this must flip the sign
                // r0 = c / (a*r0)
                r0 = c / numer;
            }
            // compute the derivative wrt log(rs)
            double sum = r0 + yRate;
            double p = r0 / sum;
            double df = shape1 - shape2 * p - rateRate * r0;
            if (Math.Abs(df) > 1)
            {
                // take a Newton step for extra accuracy
                double ddf = shape2 * p * (p - 1) - rateRate * r0;
                r0 *= Math.Exp(-df / ddf);
            }
            if (double.IsNaN(r0))
                throw new InferRuntimeException($"result is NaN.  shape1={shape1}, shape2={shape2}, yRate={yRate}, rateRate={rateRate}");
            return r0;
        }

        /// <summary>
        /// Get integration bounds for rate where sample is marginalized.  The bounds are for an integral in the log-domain, but exponentiated.
        /// </summary>
        /// <param name="sample"></param>
        /// <param name="shape"></param>
        /// <param name="rate"></param>
        /// <param name="r">On output, the mode of the integrand</param>
        /// <param name="rmin">On output, the lower bound for integration</param>
        /// <param name="rmax">On output, the upper bound for integration</param>
        private static void GetIntegrationBoundsForRate(Gamma sample, double shape, Gamma rate, out double r, out double rmin, out double rmax)
        {
            if (sample.Rate < 0)
                throw new ArgumentException("sample.Rate <= 0");
            if (rate.Rate < 0)
                throw new ArgumentException("rate.Rate < 0");
            if (rate.Shape < 0)
                throw new ArgumentException("rate.Shape < 0");
            // this routine assumes integration is done in log(r), so the Jacobian log(r) is added, turning (s-1)*log(r) into s*log(r)
            double shape1 = shape + rate.Shape;
            double shape2 = AddShapesMinus1(shape, sample.Shape);
            // q(r) = r^(ar-1) exp(-br r)
            // q(y) = y^(ay-1) exp(-by y)
            // f(y,r) = y^(s-1) r^s exp(-ry)
            // q(y) f(y,r) = y^(ay+s-2) r^s exp(-y (r+by))
            // int over y = r^s / (r+by)^(ay+s-1)
            // logp = (s+ar)*log(r) - (ay+s-1)*log(r+by) - r*br
            //      = shape1*log(r) - shape2*log(r+by) - r*br
            // shape1*log(rmax) - shape2*log(rmax+by) - rmax*br < shape1*log(r) - shape2*log(r+by) - r*br - 50
            // this function has 1 stationary point and 0 or 1 inflection points
            r = FindMaximum(shape1, shape2, sample.Rate, rate.Rate);
            // Since the function is unimodal, we can guarantee that the integration includes all of the significant mass by ensuring
            // that the function value at the bounds is sufficiently small.  We take the value at the mode and subtract 50 in the log-domain.
            // Then we apply Newton's method to find where the function equals this value.
            // To find the roots of a 1D function:
            // Find the set of stationary points and inflection points.
            // Between neighboring stationary points, the function must be monotonic, and there must be at least one inflection point.
            // Inflection points have the maximum derivative and thus will take the smallest Newton step.
            // At each inflection point, the first step of Newton's method will either go left or right.
            // If it stays within the neighboring stationary points, run Newton's method to convergence. It must find a root.
            // If it goes beyond the neighboring stationary point, skip the search.
            // If the sequence begins or ends with a stationary point, start the search anywhere left or right of this point.
            // TODO: change this to use FindZeroes()
            if (shape1 == shape2 && sample.Rate == 0)
            {
                rmin = r;
                // -rmax*br = -50
                rmax = 50 / rate.Rate;
                return;
            }
            double bound = shape1 * Math.Log(r) - shape2 * Math.Log(r + sample.Rate) - r * rate.Rate - 50;
            bool hasInflection = (shape2 > shape1);
            if (shape1 < 0)
                throw new ArgumentException("shape1 < 0");
            if (hasInflection)
            {
                // start Newton's method at the inflection point
                // the inflection point satisfies:
                // ddf = -shape1 / r^2 + shape2 / (r + by)^2 = 0
                // -shape1 * (r+by)^2 + shape2 * r^2 = 0
                // which is a quadratic equation.
                double a = shape2 - shape1;
                double b = -shape1 * 2 * sample.Rate;
                double c = -shape1 * sample.Rate * sample.Rate;
                rmax = (-b - Math.Sign(b) * Math.Sqrt(b * b - 4 * a * c)) / (2 * a);
                if (rmax <= r)
                    throw new Exception("inflection point is less than the stationary point");
            }
            else
            {
                // start anywhere greater than the stationary point
                rmax = r * 1.1;
            }
            // Newton's method
            for (int iter = 0; iter < 200; iter++)
            {
                double oldrmax = rmax;
                // see factors/matlab/test_gamma_rate.m
                //double f = shape1*Math.Log(rmax) - shape2*Math.Log(rmax+sample.Rate) - rmax*rate.Rate - bound;
                double f = shape1 * Math.Log(rmax / r) - shape2 * Math.Log((rmax + sample.Rate) / (r + sample.Rate)) - (rmax - r) * rate.Rate + 50;
                //Console.WriteLine("{0}: {1}", iter, f);
                double df = shape1 / rmax - rate.Rate - shape2 / (rmax + sample.Rate);
                rmax = rmax - f / df;
                if (rmax < r)
                {
                    if (hasInflection)
                        rmax = r * 1.1;  // restart closer to the stationary point
                    else
                        throw new Exception("rmax < r");
                }
                if (MMath.AreEqual(rmax, r))
                    break;
                if (MMath.AreEqual(rmax, oldrmax))
                {
                    // make sure that f < 0
                    while (true)
                    {
                        f = shape1 * Math.Log(rmax / r) - shape2 * Math.Log((rmax + sample.Rate) / (r + sample.Rate)) - (rmax - r) * rate.Rate + 50;
                        if (f < 0)
                            break;
                        // get the next representable floating point number
                        rmax = MMath.NextDouble(rmax);
                    }
                    break;
                }
            }
            if (double.IsPositiveInfinity(rmax))
                throw new Exception("rmax is infinity");
            rmin = r * 0.9;
            // Newton's method in log(r)
            for (int iter = 0; iter < 200; iter++)
            {
                double old_rmin = rmin;
                if (rmin == 0)
                {
                    rmin = Math.Exp((bound + shape2 * Math.Log(sample.Rate)) / shape1);
                }
                double f = shape1 * Math.Log(rmin / r) - shape2 * Math.Log((rmin + sample.Rate) / (r + sample.Rate)) - (rmin - r) * rate.Rate + 50;
                //Console.WriteLine("{0}: {1} {2}", iter, f, rmin);
                double df = shape1 - rate.Rate * rmin - shape2 * rmin / (rmin + sample.Rate);
                rmin *= Math.Exp(-f / df);
                if (double.IsNaN(rmin))
                    throw new Exception("rmin is nan");
                if (rmin > r)
                    throw new Exception("rmin > r");
                if (MMath.AreEqual(rmin, r))
                    break;
                if (MMath.AreEqual(rmin, old_rmin))
                {
                    // make sure that f < 0
                    while (true)
                    {
                        f = shape1 * Math.Log(rmin / r) - shape2 * Math.Log((rmin + sample.Rate) / (r + sample.Rate)) - (rmin - r) * rate.Rate + 50;
                        if (f < 0)
                            break;
                        // get the previous representable floating point number
                        rmin = MMath.PreviousDouble(rmin);
                    }
                    break;
                }
            }
            if (rmin > rmax)
                throw new Exception(String.Format("Internal: rmin ({0}) > rmax ({1})", rmin, rmax));
            //Console.WriteLine("rmin = {0}, rmax = {1}", rmin, rmax);
        }

        internal static double AddShapesMinus1(double a1, double a2)
        {
            if (a1 < a2)
                return a1 + (a2 - 1);
            else
                return a2 + (a1 - 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Slow"]/message_doc[@name="RateAverageConditional(Gamma, double, Gamma)"]/*'/>
        public static Gamma RateAverageConditional([SkipIfUniform] Gamma sample, double shape, Gamma rate)
        {
            if (sample.IsPointMass)
                return RateAverageConditional(sample.Point, shape);
            if (sample.Rate == 0)
            {
                // message to rate = int_x x^(ax-1) x^(s-1) r^s exp(-r*x) dx
                //                 = r^s / r^(ax-1+s) = r^(1-ax)
                // if marginal ends up with shape <= 0, then it has no moments and we will throw an exception later.
                Gamma rateMarginal = Gamma.FromShapeAndRate(rate.Shape + 1 - sample.Shape, rate.Rate);
                Gamma result = new Gamma();
                result.SetToRatio(rateMarginal, rate, GammaFromShapeAndRateOp.ForceProper);
                if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                    throw new InferRuntimeException($"result is NaN.  sample={sample}, shape={shape}, rate={rate}");
                return result;
            }
            double shape2 = AddShapesMinus1(shape, sample.Shape);
            double r;
            if (rate.IsPointMass)
            {
                // f(r) = int_x p(x) x^(s-1) r^s exp(-r*x) dx
                //      = r^s / (r + xr)^(s+xs-1)
                // log f(r) = s*log(r) - (s+xs-1)*log(r+xr)
                // dlogf = s/r - (s+xs-1)/(r+xr)
                // ddlogf = -s/r^2 + (s+xs-1)/(r+xr)^2
                r = rate.Point;
                double r2 = r + sample.Rate;
                if (r == 0)
                {
                    // a = -r*r*ddLogP
                    // b = a / r - dLogP
                    return Gamma.FromShapeAndRate(shape + 1, shape2 / r2);
                }
                else
                {
                    double dlogf = shape / r - shape2 / r2;
                    double xdlogf = shape - shape2 * r / r2;
                    double xxddlogf = -shape + shape2 * (r / r2) * (r / r2);
                    return Gamma.FromDerivatives(r, dlogf, xdlogf, xxddlogf, GammaFromShapeAndRateOp.ForceProper);
                }
            }
            double shape1 = shape + rate.Shape;
            double rateMean, rateVariance;
            double rmin, rmax;
            GetIntegrationBoundsForRate(sample, shape, rate, out r, out rmin, out rmax);
            if (rmin == r || rmax == r)
                return RateAverageConditional(sample, shape, Gamma.PointMass(r));
            double logrmin = Math.Log(rmin);
            double logrmax = Math.Log(rmax);
            int n = QuadratureNodeCount;
            double inc = (logrmax - logrmin) / (n - 1);
            if (true)
            {
                // Compute the output message directly, instead of using SetToRatio
                // this is more accurate when the prior is sharp
                double Z = 0;
                // sum1 = Exf'
                double sum1 = 0;
                // sum2 = Ex^2f''
                // f''/f = d(f'/f) + (f'/f)^2
                double sum2 = 0;
                for (int i = 0; i < n; i++)
                {
                    double logx = logrmin + i * inc;
                    double x = Math.Exp(logx);
                    // log(x) = log(x) - x/r + x/r
                    // shape1*x/r + shape1*(log(x)-x/r) - rate.Rate*x
                    // r = shape1/rate.Rate gives:
                    // shape1*(log(x) - x/shape1*rate.Rate)
                    //diff = shape1 * Math.Log(x / r) - shape2 * Math.Log((x + sample.Rate) / (r + sample.Rate)) - rate.Rate * (x - r);
                    double diff = shape1 * Math.Log(x / r) - rate.Rate * (x - r) - shape2 * Math.Log((x + sample.Rate) / (r + sample.Rate));
                    if ((i == 0 || i == n - 1) && (diff > -49))
                        throw new Exception("invalid integration bounds");
                    double f = Math.Exp(diff);
                    double q = x / (x + sample.Rate);
                    double xdlogf = shape - shape2 * q;
                    double xxddlogf = -shape + shape2 * q * q + xdlogf * xdlogf;
                    Z += f;
                    sum1 += xdlogf * f;
                    sum2 += xxddlogf * f;
                }
                double alpha = sum1 / Z;
                double beta = (sum1 + sum2) / Z - alpha * alpha;
                return GaussianOp.GammaFromAlphaBeta(rate, alpha, beta, GammaFromShapeAndRateOp.ForceProper);
            }
            else
            {
                MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                for (int i = 0; i < n; i++)
                {
                    double logx = logrmin + i * inc;
                    double x = Math.Exp(logx);
                    //double shift = shape1 * Math.Log(r) - shape2 * Math.Log(r + sample.Rate) - r * rate.Rate;
                    //double logp = shape1 * Math.Log(x) - shape2 * Math.Log(x + sample.Rate) - x * rate.Rate;
                    double diff = shape1 * Math.Log(x / r) - shape2 * Math.Log((x + sample.Rate) / (r + sample.Rate)) - rate.Rate * (x - r);
                    if ((i == 0 || i == n - 1) && (diff > -49))
                        throw new Exception("invalid integration bounds");
                    double p = Math.Exp(diff);
                    mva.Add(x, p);
                }
                rateMean = mva.Mean;
                rateVariance = mva.Variance;
                Gamma rateMarginal = Gamma.FromMeanAndVariance(rateMean, rateVariance);
                Gamma result = new Gamma();
                result.SetToRatio(rateMarginal, rate, GammaFromShapeAndRateOp.ForceProper);
                if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                    throw new InferRuntimeException($"result is NaN.  sample={sample}, shape={shape}, rate={rate}");
                return result;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/doc/*'/>
    [FactorMethod(typeof(Factor), "GammaFromShapeAndRate", Default = false)]
    [Buffers("Q")]
    [Quality(QualityBand.Experimental)]
    public class GammaFromShapeAndRateOp_Laplace : GammaFromShapeAndRateOpBase
    {
        // derivatives of the factor marginalized over sample (=y)
        private static double[] dlogfs(double r, double shape, Gamma y)
        {
            if (y.IsPointMass)
            {
                // logf = s*log(r) - y*r
                double p = 1 / r;
                double p2 = p * p;
                double dlogf = shape * p - y.Point;
                double ddlogf = -shape * p2;
                double dddlogf = 2 * shape * p * p2;
                double d4logf = -6 * shape * p2 * p2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
            else
            {
                // logf = s*log(r) - (s+ya-1)*log(r + yb)
                double p = 1 / (r + y.Rate);
                double p2 = p * p;
                double r2 = r * r;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(y.Shape, shape);
                double dlogf = shape / r - shape2 * p;
                double ddlogf = -shape / r2 + shape2 * p2;
                double dddlogf = 2 * shape / (r * r2) - 2 * shape2 * p * p2;
                double d4logf = -6 * shape / (r2 * r2) + 6 * shape2 * p2 * p2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
        }

        // derivatives of the factor marginalized over sample (=y), multiplied by powers of r
        private static double[] xdlogfs(double r, double shape, Gamma y)
        {
            if (y.IsPointMass)
            {
                // logf = s*log(r) - y*r
                double dlogf = shape - y.Point * r;
                double ddlogf = -shape;
                double dddlogf = 2 * shape;
                double d4logf = -6 * shape;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
            else
            {
                // logf = s*log(r) - (s+ya-1)*log(r + yb)
                double p = r / (r + y.Rate);
                double p2 = p * p;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(y.Shape, shape);
                double dlogf = shape - shape2 * p;
                double ddlogf = -shape + shape2 * p2;
                double dddlogf = 2 * shape - 2 * shape2 * p * p2;
                double d4logf = -6 * shape + 6 * shape2 * p2 * p2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/message_doc[@name="QInit()"]/*'/>
        [Skip]
        public static Gamma QInit()
        {
            return Gamma.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/message_doc[@name="Q(Gamma, double, Gamma)"]/*'/>
        [Fresh]
        public static Gamma Q([NoInit] Gamma sample, double shape, [Proper] Gamma rate)
        {
            if (rate.IsPointMass)
                return rate;
            if (sample.IsPointMass)
                return GammaFromShapeAndRateOp.RateAverageConditional(sample.Point, shape) * rate;
            double shape1 = shape + rate.Shape;
            double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(shape, sample.Shape);
            double x = GammaFromShapeAndRateOp_Slow.FindMaximum(shape1, shape2, sample.Rate, rate.Rate);
            if (x == 0 || double.IsInfinity(x))
                return rate;
            double[] xdlogfss = xdlogfs(x, shape, sample);
            double xdlogf = xdlogfss[0];
            double xxddlogf = xdlogfss[1];
            return GammaFromDerivatives2(rate, x, xdlogf, xxddlogf);
        }

        // Same as Gamma.FromDerivatives multiplied by B
        internal static Gamma GammaFromDerivatives(Gamma B, double x, double dlogf, double ddlogf)
        {
            double b = B.Rate - (dlogf + x * ddlogf);
            double a = B.Shape - ddlogf * x * x;
            if (a <= 0)
                a = b * B.Shape / (B.Rate - dlogf);
            if (a <= 0 || b <= 0)
                throw new InferRuntimeException("a <= 0 || b <= 0");
            if (double.IsNaN(a) || double.IsNaN(b))
                throw new InferRuntimeException($"result is NaN.  B={B}, x={x}, dlogf={dlogf}, ddlogf={ddlogf}");
            return Gamma.FromShapeAndRate(a, b);
        }

        // Same as Gamma.FromDerivatives multiplied by B
        internal static Gamma GammaFromDerivatives2(Gamma B, double x, double xdlogf, double x2ddlogf)
        {
            double b = B.Rate - (xdlogf + x2ddlogf) / x;
            double a = B.Shape - x2ddlogf;
            if (a <= 0)
                a = b * B.Shape / (B.Rate - xdlogf / x);
            if (a <= 0 || b <= 0)
                throw new InferRuntimeException("a <= 0 || b <= 0");
            if (double.IsNaN(a) || double.IsNaN(b))
                throw new InferRuntimeException($"result is NaN.  B={B}, x={x}, xdlogf={xdlogf}, x2ddlogf={x2ddlogf}");
            return Gamma.FromShapeAndRate(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/message_doc[@name="LogAverageFactor(Gamma, double, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(Gamma sample, double shape, Gamma rate, Gamma q)
        {
            if (rate.IsPointMass)
                return GammaFromShapeAndRateOp.LogAverageFactor(sample, shape, rate.Point);
            if (sample.IsPointMass)
                return GammaFromShapeAndRateOp.LogAverageFactor(sample.Point, shape, rate);
            double x = q.GetMean();
            double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(sample.Shape, shape);
            double logf = shape * Math.Log(x) - shape2 * Math.Log(x + sample.Rate) +
              MMath.GammaLn(shape2) - MMath.GammaLn(shape) - sample.GetLogNormalizer();
            double logz = logf + rate.GetLogProb(x) - q.GetLogProb(x);
            return logz;
        }

        private static double LogAverageFactor_slow(Gamma sample, double shape, [Proper] Gamma rate)
        {
            return LogAverageFactor(sample, shape, rate, Q(sample, shape, rate));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/message_doc[@name="LogEvidenceRatio(Gamma, double, Gamma, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio(Gamma sample, double shape, Gamma rate, Gamma to_sample, Gamma q)
        {
            return LogAverageFactor(sample, shape, rate, q) - to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/message_doc[@name="RateAverageConditional(Gamma, double, Gamma, Gamma)"]/*'/>
        public static Gamma RateAverageConditional([SkipIfUniform] Gamma sample, double shape, [Proper] Gamma rate, Gamma q)
        {
            if (sample.IsPointMass)
                return GammaFromShapeAndRateOp.RateAverageConditional(sample.Point, shape);
            if (rate.IsPointMass)
                return GammaFromShapeAndRateOp_Slow.RateAverageConditional(sample, shape, rate);
            if (q.IsUniform())
            {
                q = Q(sample, shape, rate);
                if (q.IsUniform())
                    return q;
            }
            double x = q.GetMean();
            double[] xg = new double[] { x, x, 0, 0 };
            double rateMean, rateVariance;
            GaussianOp_Laplace.LaplaceMoments2(q, xg, xdlogfs(x, shape, sample), out rateMean, out rateVariance);
            Gamma rateMarginal = Gamma.FromMeanAndVariance(rateMean, rateVariance);
            Gamma result = new Gamma();
            result.SetToRatio(rateMarginal, rate, true);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException($"result is NaN.  sample={sample}, shape={shape}, rate={rate}, q={q}");
            return result;
        }

        // [NoInit] is needed here (and Q) to get a good schedule for Learners/Classifier
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndRateOp_Laplace"]/message_doc[@name="SampleAverageConditional(Gamma, double, Gamma, Gamma)"]/*'/>
        public static Gamma SampleAverageConditional([NoInit] Gamma sample, double shape, [SkipIfUniform] Gamma rate, Gamma q)
        {
            if (sample.IsPointMass)
                return GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
            if (rate.IsPointMass)
                return GammaFromShapeAndRateOp.SampleAverageConditional(shape, rate.Point);
            if (q.IsPointMass)
                throw new InferRuntimeException();
            double sampleMean, sampleVariance;
            if (sample.Rate < 1e-20)
                sample.Rate = 1e-20;
            bool useLaplaceForSample = true;
            if (useLaplaceForSample)
            {
                // Laplace for sample
                // if we get here, sample.Rate > 0
                // int Ga(x; s, r) Ga(r; a, b) dr = x^(s-1) / (x + b)^(s+a)
                Gamma temp = Gamma.FromShapeAndRate(rate.Shape + 2, rate.Rate);
                Gamma qy = Q(temp, shape - 1, sample);
                double x = qy.GetMean();
                double[] xg = new double[] { x, x, 0, 0 };
                GaussianOp_Laplace.LaplaceMoments2(qy, xg, xdlogfs(x, shape - 1, temp), out sampleMean, out sampleVariance);
            }
            else
            {
                // Laplace for rate
                // tends to be less accurate than above
                double x = q.GetMean();
                double x2 = x * x;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(sample.Shape, shape); // sample.Shape+shape-1
                if (sample.Rate < x)
                {
                    // another approach might be to write 1/(sample.Rate + r) = 1/r - sample.Rate/r/(sample.Rate + r)
                    //double a1 = q.Shape - x2*p2;
                    //double b1 = q.Rate + sample.Rate*p2;
                    double logz = LogAverageFactor_slow(sample, shape, rate) + sample.GetLogNormalizer();
                    Gamma sample1 = Gamma.FromShapeAndRate(sample.Shape + 1, sample.Rate);
                    double logz1 = LogAverageFactor_slow(sample1, shape, rate) + sample1.GetLogNormalizer();
                    double pMean = Math.Exp(logz1 - logz);
                    sampleMean = shape2 * pMean;
                    Gamma sample2 = Gamma.FromShapeAndRate(sample.Shape + 2, sample.Rate);
                    double logz2 = LogAverageFactor_slow(sample2, shape, rate) + sample2.GetLogNormalizer();
                    double pMean2 = Math.Exp(logz2 - logz);
                    double pVariance = pMean2 - pMean * pMean;
                    sampleVariance = shape2 * shape2 * pVariance + shape2 * (pVariance + pMean * pMean);
                }
                else
                {
                    // sampleMean = E[ shape2/(sample.Rate + r) ]
                    // sampleVariance = var(shape2/(sample.Rate + r)) + E[ shape2/(sample.Rate+r)^2 ]
                    //                = shape2^2*var(1/(sample.Rate + r)) + shape2*(var(1/(sample.Rate+r)) + (sampleMean/shape2)^2)
                    // Note: this is not a good approximation if sample.Rate is small
                    double p = x / (x + sample.Rate);
                    double p2 = p * p;
                    double[] xg = new double[] { p, -p2, 2 * p2 * p, -6 * p2 * p2 };
                    double pMean, pVariance;
                    GaussianOp_Laplace.LaplaceMoments2(q, xg, xdlogfs(x, shape, sample), out pMean, out pVariance);
                    sampleMean = shape2 * pMean;
                    sampleVariance = shape2 * shape2 * pVariance + shape2 * (pVariance + pMean * pMean);
                }
            }
            if (double.IsPositiveInfinity(sampleVariance))  // this can happen due to overflow
                throw new InferRuntimeException("posterior variance is infinite");
            Gamma sampleMarginal = Gamma.FromMeanAndVariance(sampleMean, sampleVariance);
            if (!sampleMarginal.IsProper())
                throw new ImproperMessageException(sampleMarginal);
            Gamma result = new Gamma();
            result.SetToRatio(sampleMarginal, sample, true);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException($"result is NaN.  sample={sample}, shape={shape}, rate={rate}, q={q}");
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/doc/*'/>
    [FactorMethod(new string[] { "sample", "mean", "variance" }, typeof(Gamma), "SampleFromMeanAndVariance")]
    [Quality(QualityBand.Stable)]
    public static class GammaFromMeanAndVarianceOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double mean, double variance)
        {
            return Gamma.FromShapeAndScale(mean, variance).GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/message_doc[@name="LogAverageFactor(Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(Gamma sample, [Fresh] Gamma to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Gamma SampleAverageLogarithm(double mean, double variance)
        {
            return Gamma.FromMeanAndVariance(mean, variance);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromMeanAndVarianceOp"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Gamma SampleAverageConditional(double mean, double variance)
        {
            return Gamma.FromMeanAndVariance(mean, variance);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/doc/*'/>
    [FactorMethod(typeof(Gamma), "Sample", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class GammaFromShapeAndScaleOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double shape, double scale)
        {
            return Gamma.FromShapeAndScale(shape, scale).GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double shape, double scale)
        {
            return LogAverageFactor(sample, shape, scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="LogEvidenceRatio(Gamma, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gamma sample, double shape, double scale)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double shape, double scale)
        {
            return LogAverageFactor(sample, shape, scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="AverageLogFactor(Gamma, double, double)"]/*'/>
        public static double AverageLogFactor([Proper] Gamma sample, double shape, double scale)
        {
            return GammaFromShapeAndRateOp.AverageLogFactor(sample, shape, 1 / scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="LogAverageFactor(Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(Gamma sample, [Fresh] Gamma to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Gamma SampleAverageLogarithm(double shape, double scale)
        {
            return Gamma.FromShapeAndScale(shape, scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaFromShapeAndScaleOp"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Gamma SampleAverageConditional(double shape, double scale)
        {
            return Gamma.FromShapeAndScale(shape, scale);
        }
    }
}
