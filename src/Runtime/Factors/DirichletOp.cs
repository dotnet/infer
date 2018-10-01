// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/doc/*'/>
    [FactorMethod(new string[] { "sample", "pseudoCounts" }, typeof(Dirichlet), "SampleFromPseudoCounts")]
    [Quality(QualityBand.Stable)]
    public static class DirichletFromPseudoCountsOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="LogAverageFactor(Dirichlet, Vector, Dirichlet)"]/*'/>
        public static double LogAverageFactor(Dirichlet sample, Vector pseudoCounts, [Fresh] Dirichlet to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="LogEvidenceRatio(Dirichlet, Vector)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Dirichlet sample, Vector pseudoCounts)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="AverageLogFactor(Dirichlet, Vector, Dirichlet)"]/*'/>
        public static double AverageLogFactor(Dirichlet sample, Vector pseudoCounts, [Fresh] Dirichlet to_sample)
        {
            return to_sample.GetAverageLog(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="LogAverageFactor(Vector, Vector)"]/*'/>
        public static double LogAverageFactor(Vector sample, Vector pseudoCounts)
        {
            return SampleAverageConditional(pseudoCounts).GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector)"]/*'/>
        public static double LogEvidenceRatio(Vector sample, Vector pseudoCounts)
        {
            return LogAverageFactor(sample, pseudoCounts);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="AverageLogFactor(Vector, Vector)"]/*'/>
        public static double AverageLogFactor(Vector sample, Vector pseudoCounts)
        {
            return LogAverageFactor(sample, pseudoCounts);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="SampleAverageConditional(Vector)"]/*'/>
        public static Dirichlet SampleAverageConditional(Vector pseudoCounts)
        {
            return new Dirichlet(pseudoCounts);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletFromPseudoCountsOp"]/message_doc[@name="SampleAverageLogarithm(Vector)"]/*'/>
        public static Dirichlet SampleAverageLogarithm(Vector pseudoCounts)
        {
            return new Dirichlet(pseudoCounts);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "DirichletSymmetric")]
    [Quality(QualityBand.Preview)]
    [Buffers("probMeanLog")]
    public static class DirichletSymmetricOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AverageLogFactor(Dirichlet, Gamma, Vector)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static double AverageLogFactor([Proper] Dirichlet prob, [SkipIfUniform] Gamma alpha, Vector probMeanLog)
        {
            if (alpha.IsPointMass)
                return AverageLogFactor(prob, alpha.Point, probMeanLog);
            double SumElogP = probMeanLog.Sum();
            double K = (double)probMeanLog.Count;
            double a = alpha.Shape;
            double b = alpha.Rate;
            double averageFactor = GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b / K));
            averageFactor -= K * GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b));
            averageFactor += (a / b - 1.0) * SumElogP;
            return averageFactor;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AverageLogFactor(Vector, Gamma)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static double AverageLogFactor(Vector prob, [SkipIfUniform] Gamma alpha)
        {
            Dirichlet d = Dirichlet.PointMass(prob);
            return AverageLogFactor(d, alpha, d.GetMeanLog());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AverageLogFactor(Vector, ConjugateDirichlet)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static double AverageLogFactor(Vector prob, [SkipIfUniform] ConjugateDirichlet alpha)
        {
            return (alpha.GetMean() - 1.0) * prob.Sum(Math.Log) - (prob.Count * alpha.GetMeanLogGamma(1.0) - alpha.GetMeanLogGamma(prob.Count));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AverageLogFactor(Dirichlet, ConjugateDirichlet)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static double AverageLogFactor([Proper] Dirichlet prob, ConjugateDirichlet alpha)
        {
            if (alpha.IsPointMass)
                return LogAverageFactor(prob, alpha.Point);
            return LogAverageFactor(prob, alpha.GetMode());
            //return (alpha.GetMean() - 1.0) * prob.GetMeanLog().Sum() - (prob.Dimension * alpha.GetMeanLogGamma(1.0) - alpha.GetMeanLogGamma(prob.Dimension));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AverageLogFactor(Dirichlet, double, Vector)"]/*'/>
        [Quality(QualityBand.Stable)]
        public static double AverageLogFactor([Proper] Dirichlet prob, double alpha, Vector probMeanLog)
        {
            double SumElogP = probMeanLog.Sum();
            double K = (double)probMeanLog.Count;
            return MMath.GammaLn(K * alpha) - K * MMath.GammaLn(alpha) + (alpha - 1.0) * SumElogP;
        }

        /// <summary>
        /// Returns the gradient and value of the KL divergence for this factor
        /// </summary>
        /// <param name="a2">Prior shape</param>
        /// <param name="b2">Prior rate</param>
        /// <param name="x">A vector of the variational posterior parameters. x[1]=log(shape), x[2]=log(rate)</param>
        /// <param name="SumElogP">Sum E[ log(prob_k) ]. Cached for efficiency</param>
        /// <param name="grad">Vector to fill with the gradient</param>
        /// <param name="K">Dimensionality</param>
        /// <returns>KL divergence</returns>
        private static double GradientAndValueAtPoint(double a2, double b2, Vector x, double SumElogP, Vector grad, double K)
        {
            double a = Math.Exp(x[0]);
            double b = Math.Exp(x[1]);
            double averageFactor = GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b / K));
            averageFactor -= K * GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b));
            averageFactor += (a / b - 1.0) * SumElogP;
            double kl_value = Math.Log(b) - MMath.GammaLn(a) + (a - 1) * MMath.Digamma(a) - a // entropy
                              - (a2 * Math.Log(b2) - MMath.GammaLn(a2) + (a2 - 1) * (MMath.Digamma(a) - Math.Log(b)) - b2 * a / b) // cross entropy
                              - averageFactor; // factor
            if (double.IsInfinity(kl_value) || double.IsNaN(kl_value))
                throw new InferRuntimeException("KL divergence became ill-defined.");
            if (grad != null)
            {
                var gradS = Vector.Zero(2);
                gradS += GammaFromShapeAndRateOp.CalculateDerivatives(Gamma.FromShapeAndRate(a, b / K));
                gradS[1] = gradS[1] / K;
                gradS -= K * GammaFromShapeAndRateOp.CalculateDerivatives(Gamma.FromShapeAndRate(a, b));
                gradS[0] += SumElogP / b;
                gradS[1] -= SumElogP * a / (b * b);
                grad[0] = (a - 1.0) * MMath.Trigamma(a) - 1.0 // entropy
                          - (a2 - 1) * MMath.Trigamma(a) + b2 / b // cross term
                          - gradS[0]; // factor
                grad[0] *= a; // chain rule
                grad[1] = 1.0 / b // entropy
                          + (a2 - 1) / b - b2 * a / (b * b) // cross term
                          - gradS[1]; // factor
                grad[1] *= b; // chain rule
            }

            return kl_value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbMeanLogInit(Vector)"]/*'/>
        public static Vector ProbMeanLogInit([IgnoreDependency] Vector prob)
        {
            return Vector.Copy(prob);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbMeanLogInit(Dirichlet)"]/*'/>
        public static Vector ProbMeanLogInit([IgnoreDependency] Dirichlet prob)
        {
            return Vector.Copy(prob.PseudoCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbMeanLog(Dirichlet, Vector)"]/*'/>
        [Fresh]
        public static Vector ProbMeanLog(Dirichlet prob, Vector result)
        {
            prob.GetMeanLog(result);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbMeanLog(Vector, Vector)"]/*'/>
        [Fresh]
        public static Vector ProbMeanLog(Vector prob, Vector result)
        {
            result.SetToFunction(prob, Math.Log);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AlphaAverageLogarithm(Dirichlet, Gamma, Vector, Gamma)"]/*'/>
        /// <remarks><para>
        /// Optimal message calculated by minimising local KL divergence using LBFGS. 
        /// </para></remarks>
        [Quality(QualityBand.Experimental)]
        public static Gamma AlphaAverageLogarithm([Proper] Dirichlet prob, [SkipIfUniform] Gamma alpha, Vector probMeanLog, Gamma to_Alpha)
        {
            if (alpha.IsPointMass)
                return Gamma.Uniform();
            var s = new BFGS();
            int K = probMeanLog.Count;
            var prior = alpha / to_Alpha;
            int evalCounter = 0;
            s.MaximumStep = 20;
            s.MaximumIterations = 100;
            s.Epsilon = 1e-5;
            s.convergenceCriteria = BFGS.ConvergenceCriteria.Objective;
            double SumElogP = probMeanLog.Sum();
            var z = Vector.FromArray(new double[] { Math.Log(alpha.Shape), Math.Log(alpha.Rate) });
            double startingValue = GradientAndValueAtPoint(prior.Shape, prior.Rate, z, SumElogP, null, (double)K);
            FunctionEval f = delegate(Vector y, ref Vector grad)
                {
                    evalCounter++;
                    return GradientAndValueAtPoint(prior.Shape, prior.Rate, y, SumElogP, grad, (double)K);
                };
            //DerivativeChecker.CheckDerivatives(f, z); 
            z = s.Run(z, 1.0, f);
            var result = Gamma.FromShapeAndRate(Math.Exp(z[0]), Math.Exp(z[1]));
            result.SetToRatio(result, prior);
            double endValue = GradientAndValueAtPoint(prior.Shape, prior.Rate, z, SumElogP, null, (double)K);
            //Console.WriteLine("Went from {0} to {1} in {2} steps, {3} evals", startingValue, endValue, s.IterationsPerformed, evalCounter);
            if (startingValue < endValue)
                Console.WriteLine("Warning: BFGS resulted in an increased objective function");
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AlphaAverageLogarithm(Dirichlet, ConjugateDirichlet, Vector)"]/*'/>
        /// <remarks><para>
        /// Optimal message calculated by minimising local KL divergence using LBFGS. 
        /// </para></remarks>
        [Quality(QualityBand.Experimental)]
        public static ConjugateDirichlet AlphaAverageLogarithm(
            [Proper, SkipIfUniform] Dirichlet prob, [SkipIfUniform] ConjugateDirichlet alpha, [SkipIfUniform] Vector probMeanLog)
        {
            // TODO: why is probMeanLog not set correctly? 
            if (alpha.IsPointMass)
                return ConjugateDirichlet.Uniform();
            var result = ConjugateDirichlet.FromNatural(0, -prob.GetMeanLog().Sum(), probMeanLog.Count, 1);
            //var result = ConjugateDirichlet.FromNatural(0, -probMeanLog.Sum(), probMeanLog.Count, 1);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AlphaAverageLogarithm(Vector, ConjugateDirichlet, Vector)"]/*'/>
        /// <remarks><para>
        /// Optimal message calculated by minimising local KL divergence using LBFGS. 
        /// </para></remarks>
        [Quality(QualityBand.Experimental)]
        public static ConjugateDirichlet AlphaAverageLogarithm([Proper] Vector prob, ConjugateDirichlet alpha, [SkipIfUniform] Vector probMeanLog)
        {
            // TODO: why is probMeanLog not set correctly? 
            if (alpha.IsPointMass)
                return ConjugateDirichlet.Uniform();
            var result = ConjugateDirichlet.FromNatural(0, -prob.Sum(Math.Log), probMeanLog.Count, 1);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AlphaAverageLogarithm(Vector, Gamma, Gamma)"]/*'/>
        /// <remarks><para>
        /// Optimal message calculated by minimising local KL divergence using LBFGS. 
        /// </para></remarks>
        [Quality(QualityBand.Experimental)]
        public static Gamma AlphaAverageLogarithm(Vector prob, [SkipIfUniform] Gamma alpha, Gamma to_Alpha)
        {
            Dirichlet d = Dirichlet.PointMass(prob);
            return AlphaAverageLogarithm(d, alpha, d.GetMeanLog(), to_Alpha);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageLogarithm(double, Dirichlet)"]/*'/>
        [Quality(QualityBand.Stable)]
        public static Dirichlet ProbAverageLogarithm([SkipIfUniform] double alpha, Dirichlet result)
        {
            result.PseudoCount.SetAllElementsTo(alpha);
            result.TotalCount = result.PseudoCount.Sum();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageLogarithmInit(int)"]/*'/>
        [Skip]
        public static Dirichlet ProbAverageLogarithmInit(int K)
        {
            return Dirichlet.Uniform(K);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageLogarithm(Gamma, Dirichlet)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Dirichlet ProbAverageLogarithm([SkipIfUniform] Gamma alpha, Dirichlet result)
        {
            result.PseudoCount.SetAllElementsTo(alpha.GetMean());
            result.TotalCount = result.PseudoCount.Sum();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageLogarithm(ConjugateDirichlet, Dirichlet)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Dirichlet ProbAverageLogarithm([SkipIfUniform] ConjugateDirichlet alpha, Dirichlet result)
        {
            if (!alpha.IsPointMass)
            {
                double mean, variance;
                alpha.SmartProposal(out mean, out variance);
                double alphaMode = Math.Exp(mean);
                if (double.IsNaN(alphaMode) || double.IsInfinity(alphaMode))
                    throw new InferRuntimeException("Nan message in ProbAverageLogarithm");
                //result.PseudoCount.SetAllElementsTo(alphaMode);
                result.PseudoCount.SetAllElementsTo(alpha.GetMean());
            }
            else
                result.PseudoCount.SetAllElementsTo(alpha.Point);
            result.TotalCount = result.PseudoCount.Sum();
            return result;
        }


        // --------------------- EP not supported ------------------------------ 

        private const string NotSupportedMessage = "Expectation Propagation does not currently support Dirichlet distributions with stochastic arguments";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="LogAverageFactor(Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor([SkipIfUniform] Dirichlet prob, [SkipIfUniform] Gamma alpha)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="LogAverageFactor(IList{double}, double)"]/*'/>
        public static double LogAverageFactor(IList<double> prob, double alpha)
        {
            int dim = prob.Count;
            double sum = MMath.GammaLn(dim * alpha) - dim * MMath.GammaLn(alpha);
            double alphaM1 = alpha - 1;
            for (int i = 0; i < dim; i++)
            {
                sum += alphaM1 * Math.Log(prob[i]);
            }
            return sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="LogEvidenceRatio(IList{double}, double)"]/*'/>
        public static double LogEvidenceRatio(IList<double> prob, double alpha)
        {
            return LogAverageFactor(prob, alpha);
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AverageLogFactor(IList{double}, double)"]/*'/>
        public static double AverageLogFactor(IList<double> prob, double alpha)
        {
            return LogAverageFactor(prob, alpha);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="LogAverageFactor(Dirichlet, double)"]/*'/>
        public static double LogAverageFactor(Dirichlet prob, double alpha)
        {
            int dim = prob.Dimension;
            double sum = MMath.GammaLn(dim * alpha) - dim * MMath.GammaLn(alpha) - prob.GetLogNormalizer() - MMath.GammaLn(prob.TotalCount + dim * alpha);
            for (int i = 0; i < dim; i++)
            {
                sum += MMath.GammaLn(prob.PseudoCount[i] + alpha);
            }
            return sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="LogEvidenceRatio(Dirichlet, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Dirichlet prob, double alpha)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageConditional(double, Dirichlet)"]/*'/>
        public static Dirichlet ProbAverageConditional([SkipIfUniform] double alpha, Dirichlet result)
        {
            result.PseudoCount.SetAllElementsTo(alpha);
            result.TotalCount = result.PseudoCount.Sum();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageConditionalInit(int)"]/*'/>
        [Skip]
        public static Dirichlet ProbAverageConditionalInit(int K)
        {
            return Dirichlet.Uniform(K);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="AlphaAverageConditional(Gamma, Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma AlphaAverageConditional([SkipIfUniform] Gamma alpha, [SkipIfUniform] Dirichlet prob, Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletSymmetricOp"]/message_doc[@name="ProbAverageConditional(Gamma, Dirichlet)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet ProbAverageConditional([SkipIfUniform] Gamma alpha, Dirichlet result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "DirichletFromMeanAndTotalCount")]
    [Quality(QualityBand.Preview)]
    public static class DirichletOp
    {
        /// <summary>
        /// How much damping to use to prevent improper messages. Higher values result in more damping. 
        /// </summary>
        public static double damping = 0.5;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="AverageLogFactor(Vector, Vector, double)"]/*'/>
        public static double AverageLogFactor(Vector prob, Vector mean, double totalCount)
        {
            return (new Dirichlet(mean * totalCount)).GetLogProb(prob);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="AverageLogFactor(Dirichlet, Dirichlet, Gamma)"]/*'/>
        public static double AverageLogFactor(Dirichlet prob, Dirichlet mean, [Proper] Gamma totalCount)
        {
            double totalCountMean = totalCount.GetMean();
            Vector meanMean = mean.GetMean();
            Vector probMeanLog = prob.GetMeanLog();
            double sum = probMeanLog.Inner(meanMean, x => totalCountMean * x - 1.0);
            return sum + GammaFromShapeAndRateOp.ELogGamma(totalCount) - EvidenceMessageExpectations(mean, totalCount).Sum();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="AverageLogFactor(Dirichlet, Vector, Gamma)"]/*'/>
        public static double AverageLogFactor(Dirichlet prob, Vector mean, [Proper] Gamma totalCount)
        {
            double totalCountMean = totalCount.GetMean();
            Vector probMeanLog = prob.GetMeanLog();
            double sum = GammaFromShapeAndRateOp.ELogGamma(totalCount);
            Gamma smk = new Gamma(totalCount);
            sum += probMeanLog.Inner(mean, x => totalCountMean * x - 1.0);
            sum += mean.Sum(x =>
                {
                    smk.Rate = totalCount.Rate / x;
                    return -GammaFromShapeAndRateOp.ELogGamma(smk);
                });
            return sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="AverageLogFactor(Vector, Vector, Gamma)"]/*'/>
        public static double AverageLogFactor(Vector prob, Vector mean, [Proper] Gamma totalCount)
        {
            return AverageLogFactor(Dirichlet.PointMass(prob), mean, totalCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="AverageLogFactor(Vector, Dirichlet, Gamma)"]/*'/>
        public static double AverageLogFactor(Vector prob, Dirichlet mean, [Proper] Gamma totalCount)
        {
            return AverageLogFactor(Dirichlet.PointMass(prob), mean, totalCount);
        }

        public static Dirichlet ProbAverageLogarithm(Dirichlet mean, [Proper] Gamma totalCount, Dirichlet result)
        {
            double totalCountMean = totalCount.GetMean();
            // result.PseduoCount = totalCount.GetMean() * mean.GetMean()
            mean.GetMean(result.PseudoCount);
            result.PseudoCount.Scale(totalCountMean);
            result.TotalCount = totalCountMean;
            return result;
        }

        public static Dirichlet ProbAverageLogarithm(Dirichlet mean, double totalCount, Dirichlet result)
        {
            mean.GetMean(result.PseudoCount);
            result.PseudoCount.Scale(totalCount);
            result.TotalCount = totalCount;
            return result;
        }

        public static Dirichlet ProbAverageLogarithm(Vector mean, [Proper] Gamma totalCount, Dirichlet result)
        {
            double totalCountMean = totalCount.GetMean();
            result.PseudoCount.SetToProduct(mean, totalCountMean);
            result.TotalCount = totalCountMean;
            return result;
        }

        public static Dirichlet ProbAverageLogarithm(Vector mean, double totalCount, Dirichlet result)
        {
            result.PseudoCount.SetToProduct(mean, totalCount);
            result.TotalCount = totalCount;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="MeanAverageLogarithm(Dirichlet, Gamma, Dirichlet, Dirichlet)"]/*'/>
        public static Dirichlet MeanAverageLogarithm(Dirichlet mean, [Proper] Gamma totalCount, [SkipIfUniform] Dirichlet prob, Dirichlet to_mean)
        {
            Vector gradS = CalculateGradientForMean(mean.PseudoCount, totalCount, prob.GetMeanLog());
            // Project onto Dirichlet, efficient matrix inversion (see TM's Dirichlet fitting paper)
            int K = mean.Dimension;
            Vector q = Vector.Zero(K);
            double gOverQ = 0, OneOverQ = 0;
            for (int k = 0; k < K; k++)
            {
                q[k] = MMath.Trigamma(mean.PseudoCount[k]);
                gOverQ += gradS[k] / q[k];
                OneOverQ += 1 / q[k];
            }
            double z = -MMath.Trigamma(mean.TotalCount);
            double b = gOverQ / (1 / z + OneOverQ);
            // Create new approximation and damp

            if (damping == 0.0)
            {
                to_mean.PseudoCount.SetToFunction(gradS, q, (x, y) => ((x - b) / y) + 1.0);
                to_mean.TotalCount = to_mean.PseudoCount.Sum();
                return to_mean;
            }
            else
            {
                var old_msg = (Dirichlet)to_mean.Clone();
                to_mean.PseudoCount.SetToFunction(gradS, q, (x, y) => ((x - b) / y) + 1.0);
                to_mean.TotalCount = to_mean.PseudoCount.Sum();
                return (to_mean ^ (1 - damping)) * (old_msg ^ damping);
            }
        }

        /// <summary>
        /// Helper function to calculate gradient of the KL divergence with respect to the mean of the Dirichlet. 
        /// </summary>
        /// <param name="meanPseudoCount">Pseudocount vector of the incoming message from 'mean'</param>
        /// <param name="totalCount">Incoming message from 'totalCount'</param>
        /// <param name="meanLogProb">E[log(prob)]</param>
        /// <returns>Gradient of the KL divergence with respect to the mean of the Dirichlet</returns>
        internal static Vector CalculateGradientForMean(Vector meanPseudoCount, Gamma totalCount, Vector meanLogProb)
        {
            // Compute required integrals
            double[] EELogGamma;
            double[] EELogMLogGamma;
            double[] EELogOneMinusMLogGamma;
            MeanMessageExpectations(
                meanPseudoCount,
                totalCount,
                out EELogGamma,
                out EELogMLogGamma,
                out EELogOneMinusMLogGamma);

            // Calculate gradients of ELogGamma(sm)
            int K = meanPseudoCount.Count;
            double meanTotalCount = meanPseudoCount.Sum();
            Vector ELogM = Vector.Zero(K);
            Vector B = Vector.Zero(K);
            Vector A = Vector.Zero(K);
            ELogM.SetToFunction(meanPseudoCount, MMath.Digamma);
            ELogM.SetToDifference(ELogM, MMath.Digamma(meanTotalCount));
            for (int k = 0; k < K; k++)
            {
                A[k] = EELogMLogGamma[k] - ELogM[k] * EELogGamma[k];
                double ELogOneMinusM = MMath.Digamma(meanTotalCount - meanPseudoCount[k])
                                       - MMath.Digamma(meanTotalCount);
                B[k] = EELogOneMinusMLogGamma[k] - ELogOneMinusM * EELogGamma[k];
            }
            Vector gradC = A - B + B.Sum();
            // Calculate gradients of analytic part
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += meanPseudoCount[k] * meanLogProb[k];
            Vector gradS = Vector.Constant(K, -sum / (meanTotalCount * meanTotalCount));
            for (int k = 0; k < K; k++)
                gradS[k] += meanLogProb[k] / meanTotalCount;
            gradS *= totalCount.GetMean();
            gradS -= gradC;
            return gradS;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="TotalCountAverageLogarithm(Vector, Gamma, Dirichlet, Gamma)"]/*'/>
        /// <remarks><para>
        /// The outgoing message here would not be Dirichlet distributed, so we use Nonconjugate VMP, which
        /// sends the approximate factor ensuring the gradient of the KL wrt to the variational parameters match. 
        /// </para></remarks>
        public static Gamma TotalCountAverageLogarithm(Vector mean, [Proper] Gamma totalCount, [SkipIfUniform] Dirichlet prob, Gamma to_totalCount)
        {
            double at = totalCount.Shape;
            double bt = totalCount.Rate;
            // Find required expectations using quadrature
            Vector gradElogGamma = GammaFromShapeAndRateOp.CalculateDerivatives(totalCount);
            Vector gradS = gradElogGamma;
            Gamma smk = new Gamma(totalCount);
            for (int k = 0; k < mean.Count; k++)
            {
                smk.Rate = totalCount.Rate / mean[k];
                gradS -= GammaFromShapeAndRateOp.CalculateDerivatives(smk);
            }
            // Analytic 
            double c = mean.Inner(prob.GetMeanLog());
            gradS[0] += c / bt;
            gradS[1] -= c * at / (bt * bt);
            Matrix mat = new Matrix(2, 2);
            mat[0, 0] = MMath.Trigamma(at);
            mat[1, 0] = mat[0, 1] = -1 / bt;
            mat[1, 1] = at / (bt * bt);
            Vector v = GammaFromShapeAndRateOp.twoByTwoInverse(mat) * gradS;
            Gamma approximateFactor = Gamma.FromShapeAndRate(v[0] + 1, v[1]);

            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_totalCount ^ damping);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="TotalCountAverageLogarithm(Dirichlet, Gamma, Dirichlet, Gamma)"]/*'/>
        /// <remarks><para>
        /// The outgoing message here would not be Dirichlet distributed, so we use Nonconjugate VMP, which
        /// sends the approximate factor ensuring the gradient of the KL wrt to the variational parameters match. 
        /// </para></remarks>
        public static Gamma TotalCountAverageLogarithm(Dirichlet mean, [Proper] Gamma totalCount, [SkipIfUniform] Dirichlet prob, Gamma to_totalCount)
        {
            Gamma approximateFactor = TotalCountAverageLogarithmHelper(
                mean.PseudoCount,
                totalCount,
                prob.GetMeanLog());
            double damping = 0.9;
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_totalCount ^ damping);
        }

        /// <summary>
        /// VMP message to 'totalCount'. This functionality is separated out to allow use by BetaOp. 
        /// </summary>
        /// <param name="meanPseudoCount">Pseudocount of incoming message from 'mean'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from 'totalCount'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="meanLogProb">E[log(prob)] from incoming message from 'prob'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <remarks><para>
        /// The outgoing message here would not be Dirichlet distributed, so we use Nonconjugate VMP, which
        /// sends the approximate factor ensuring the gradient of the KL wrt to the variational parameters match. 
        /// </para></remarks>
        internal static Gamma TotalCountAverageLogarithmHelper(Vector meanPseudoCount, Gamma totalCount, Vector meanLogProb)
        {
            double[] EELogGamma;
            double[] EELogSLogGamma;
            double[] EEMSDigamma;
            // 2D quadrature
            TotalCountMessageExpectations(
                meanPseudoCount,
                totalCount,
                out EELogGamma,
                out EELogSLogGamma,
                out EEMSDigamma);
            double at = totalCount.Shape;
            double bt = totalCount.Rate;
            // Find required expectations using quadrature
            Vector gradElogGamma = GammaFromShapeAndRateOp.CalculateDerivatives(totalCount);
            Vector gradS = gradElogGamma;
            Vector EM = Vector.Zero(meanPseudoCount.Count);
            EM.SetToProduct(meanPseudoCount, 1.0 / meanPseudoCount.Sum());
            double c = 0;
            for (int k = 0; k < meanPseudoCount.Count; k++)
            {
                gradS[0] -= EELogSLogGamma[k] - totalCount.GetMeanLog() * EELogGamma[k];
                gradS[1] -= -EEMSDigamma[k] / bt;
                c += EM[k] * meanLogProb[k];
            }
            // Analytic 
            gradS[0] += c / bt;
            gradS[1] -= c * at / (bt * bt);
            Matrix mat = new Matrix(2, 2);
            mat[0, 0] = MMath.Trigamma(at);
            mat[1, 0] = mat[0, 1] = -1 / bt;
            mat[1, 1] = at / (bt * bt);
            Vector v = GammaFromShapeAndRateOp.twoByTwoInverse(mat) * gradS;
            return Gamma.FromShapeAndRate(v[0] + 1, v[1]);
        }

        /// <summary>
        /// Perform the quadrature required for the Nonconjugate VMP message to 'mean'
        /// </summary>
        /// <param name="meanQPseudoCount">Incoming message from 'mean'.</param>
        /// <param name="totalCountQ">Incoming message from 'totalCount'.</param>
        /// <param name="EELogGamma">Array to be filled with E[LogGamma(s*m_k)].</param>
        /// <param name="EELogMLogGamma">Array to be filled with E[Log(m_k)*LogGamma(s*m_k)].</param>
        /// <param name="EELogOneMinusMLogGamma">Array to be filled with E[Log(1-m_k)*LogGamma(s*m_k)].</param>
        /// <remarks><para>
        /// All three arrays are calculated simultaneously for efficiency. The quadrature over 
        /// 'totalCount' (which is Gamma-distributed) is performed by a change of variable x=log(s)
        /// followed by Gauss-Hermite quadrature. The quadrature over m is performed using 
        /// Gauss-Legendre. 
        /// </para></remarks>
        public static void MeanMessageExpectations(
            Vector meanQPseudoCount,
            Gamma totalCountQ,
            out double[] EELogGamma,
            out double[] EELogMLogGamma,
            out double[] EELogOneMinusMLogGamma)
        {
            // Get shape and scale of the distribution
            double at, bt;
            at = totalCountQ.Shape;
            bt = totalCountQ.Rate;

            // Mean in the transformed domain
            double ELogS = totalCountQ.GetMeanLog();
            // Laplace approximation of variance in transformed domain 
            double proposalVariance = 1 / at;

            // Quadrature coefficient
            int nt = 32;
            Vector nodes = Vector.Zero(nt);
            Vector weights = Vector.Zero(nt);
            Vector expx = Vector.Zero(nt);
            if (!totalCountQ.IsPointMass)
            {
                Quadrature.GaussianNodesAndWeights(ELogS, proposalVariance, nodes, weights);
                // Precompute weights for each m slice
                for (int i = 0; i < nt; i++)
                {
                    double x = nodes[i];
                    expx[i] = Math.Exp(x);
                    double p = at * x - bt * expx[i] - Gaussian.GetLogProb(x, ELogS, proposalVariance);
                    weights[i] *= Math.Exp(p);
                }
            }

            int nm = 20;
            Vector mnodes = Vector.Zero(nm);
            Vector mweight = Vector.Zero(nm);
            Quadrature.UniformNodesAndWeights(0, 1, mnodes, mweight);
            int K = meanQPseudoCount.Count;
            Vector[] mweights = new Vector[K];
            Beta[] mkDist = new Beta[K];
            EELogGamma = new double[K];
            EELogMLogGamma = new double[K];
            EELogOneMinusMLogGamma = new double[K];
            double meanQTotalCount = meanQPseudoCount.Sum();
            for (int i = 0; i < K; i++)
            {
                mweights[i] = Vector.Copy(mweight);
                mkDist[i] = new Beta(meanQPseudoCount[i], meanQTotalCount - meanQPseudoCount[i]);
                EELogGamma[i] = 0;
                EELogMLogGamma[i] = 0;
                EELogOneMinusMLogGamma[i] = 0;
            }

            double ES = totalCountQ.GetMean();
            double ESLogS = ELogS * ES + 1 / bt;

            for (int j = 0; j < nm; j++)
            {
                double m = mnodes[j];
                double ELogGamma = 0;
                if (totalCountQ.IsPointMass)
                    ELogGamma = MMath.GammaLn(m * totalCountQ.Point);
                else
                {
                    // Calculate expectations in x=log(s) space using Gauss-Hermite quadrature
                    for (int i = 0; i < nt; i++)
                        ELogGamma += weights[i] * (MMath.GammaLn(m * expx[i]) + nodes[i]);
                    // Normalise and add removed components
                    double normalisation = Math.Pow(bt, at) / MMath.Gamma(at);
                    ELogGamma = normalisation * ELogGamma - ELogS;
                }

                double EELogMLogGammaTemp = Math.Log(m) * (ELogGamma + ELogS + Math.Log(m));
                double EELogOneMinusMLogGammaTemp = Math.Log(1 - m) *
                                                    (ELogGamma - (.5 * Math.Log(2 * Math.PI) - .5 * ELogS
                                                                  - .5 * Math.Log(m) + m * ESLogS + ES * m * Math.Log(m) - ES * m));
                for (int i = 0; i < K; i++)
                {
                    mweights[i][j] *= Math.Exp(mkDist[i].GetLogProb(m));
                    EELogGamma[i] += mweights[i][j] * ELogGamma;
                    EELogMLogGamma[i] += mweights[i][j] * EELogMLogGammaTemp;
                    EELogOneMinusMLogGamma[i] += mweights[i][j] * EELogOneMinusMLogGammaTemp;
                }
            }
            for (int i = 0; i < K; i++)
                AddAnalyticComponent(
                    mkDist[i],
                    ELogS,
                    ES,
                    ESLogS,
                    ref EELogMLogGamma[i],
                    ref EELogOneMinusMLogGamma[i]);
        }

        // Helper function to add the removed parts back (see note)
        private static void AddAnalyticComponent(
            Beta meanQ,
            double ELogS,
            double ES,
            double ESLogS,
            ref double EELogMLogGamma,
            ref double EELogOneMinusMLogGamma)
        {
            double ELogM, ELogOneMinusM;
            meanQ.GetMeanLogs(out ELogM, out ELogOneMinusM);
            double ELogMSquared = ELogM * ELogM
                                  + MMath.Trigamma(meanQ.TrueCount) - MMath.Trigamma(meanQ.TotalCount);
            EELogMLogGamma -= ELogS * ELogM + ELogMSquared;
            double Em = meanQ.GetMean();
            double am = meanQ.TrueCount;
            double bm = meanQ.FalseCount;
            double EmlogOneMinusM = Em * (MMath.Digamma(bm) - MMath.Digamma(am + bm + 1));
            double EmlogmlogOneMinusM = Em * ((MMath.Digamma(bm) - MMath.Digamma(am + bm + 1)) * (MMath.Digamma(am + 1)
                                                                                              - MMath.Digamma(am + bm + 1)) - MMath.Trigamma(am + bm + 1));
            double ELogMLogOneMinusM = ELogM * ELogOneMinusM - MMath.Trigamma(meanQ.TotalCount);
            EELogOneMinusMLogGamma += .5 * Math.Log(2 * Math.PI) * ELogOneMinusM - .5 * ELogS * ELogOneMinusM
                                      - .5 * ELogMLogOneMinusM + EmlogOneMinusM * ESLogS
                                      + ES * EmlogmlogOneMinusM - ES * EmlogOneMinusM;
        }

        /// <summary>
        /// Perform the quadrature required for the Nonconjugate VMP message to 'totalCount'
        /// </summary>
        /// <param name="meanQPseudoCount">Incoming message from 'mean'.</param>
        /// <param name="totalCountQ">Incoming message from 'totalCount'.</param>
        /// <param name="EELogGamma">Array to be filled with E[LogGamma(s*m_k)].</param>
        /// <param name="EELogSLogGamma">Array to be filled with E[Log(s)*LogGamma(s*m_k)].</param>
        /// <param name="EEMSDigamma">Array to be filled with E[s*m_k*Digamma(s*m_k)].</param>
        /// <remarks><para>
        /// All three arrays are calculated simultaneously for efficiency. The quadrature over 
        /// 'totalCount' (which is Gamma-distributed) is peformed by a change of variable x=log(s)
        /// followed by Gauss-Hermite quadrature. The quadrature over m is performed using 
        /// Gauss-Legendre. 
        /// </para></remarks>
        public static void TotalCountMessageExpectations(
            Vector meanQPseudoCount,
            Gamma totalCountQ,
            out double[] EELogGamma,
            out double[] EELogSLogGamma,
            out double[] EEMSDigamma)
        {
            // Get shape and rate of the distribution
            double at = totalCountQ.Shape, bt = totalCountQ.Rate;

            // Mean in the transformed domain
            double proposalMean = totalCountQ.GetMeanLog();
            // Laplace approximation of variance in transformed domain 
            double proposalVariance = 1 / at;

            // Quadrature coefficient
            int nt = 32;
            Vector nodes = Vector.Zero(nt);
            Vector weights = Vector.Zero(nt);
            Vector expx = Vector.Zero(nt);
            if (!totalCountQ.IsPointMass)
            {
                Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                // Precompute weights for each m slice
                for (int i = 0; i < nt; i++)
                {
                    double x = nodes[i];
                    expx[i] = Math.Exp(x);
                    double p = at * x - bt * expx[i] - Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                    weights[i] *= Math.Exp(p);
                }
            }

            int nm = 20;
            Vector mnodes = Vector.Zero(nm);
            Vector mweight = Vector.Zero(nm);
            Quadrature.UniformNodesAndWeights(0, 1, mnodes, mweight);
            int K = meanQPseudoCount.Count;
            Vector[] mweights = new Vector[K];
            Beta[] mkDist = new Beta[K];
            EELogGamma = new double[K];
            EELogSLogGamma = new double[K];
            EEMSDigamma = new double[K];
            double meanQTotalCount = meanQPseudoCount.Sum();
            for (int i = 0; i < K; i++)
            {
                mweights[i] = Vector.Copy(mweight);
                mkDist[i] = new Beta(meanQPseudoCount[i], meanQTotalCount - meanQPseudoCount[i]);
                EELogGamma[i] = 0;
                EELogSLogGamma[i] = 0;
                EEMSDigamma[i] = 0;
            }

            for (int j = 0; j < nm; j++)
            {
                double m = mnodes[j];
                double ESDigamma = 0;
                double ELogGamma = 0;
                double ELogSLogGamma = 0;
                if (totalCountQ.IsPointMass)
                {
                    ESDigamma = totalCountQ.Point * MMath.Digamma(m * totalCountQ.Point);
                    ELogGamma = MMath.GammaLn(m * totalCountQ.Point);
                    ELogSLogGamma = Math.Log(totalCountQ.Point) * ELogGamma;
                }
                else
                {
                    // Calculate expectations in x=log(s) space using Gauss-Hermite quadrature
                    for (int i = 0; i < nt; i++)
                    {
                        double x = nodes[i];
                        ELogGamma += weights[i] * (MMath.GammaLn(m * expx[i]) + x);
                        ESDigamma += weights[i] * (expx[i] * MMath.Digamma(m * expx[i]) + 1);
                        ELogSLogGamma += weights[i] * (x * MMath.GammaLn(m * expx[i]) + x * x + x * Math.Log(m));
                    }
                    // Normalise and add removed components
                    double normalisation = Math.Pow(bt, at) / MMath.Gamma(at);
                    ELogGamma = normalisation * ELogGamma - proposalMean;
                    ELogSLogGamma = normalisation * ELogSLogGamma
                                    - (MMath.Trigamma(at) + proposalMean * proposalMean + Math.Log(m) * proposalMean);
                    ESDigamma = normalisation * ESDigamma - 1;
                }
                for (int i = 0; i < K; i++)
                {
                    mweights[i][j] *= Math.Exp(mkDist[i].GetLogProb(m));
                    EELogGamma[i] += mweights[i][j] * ELogGamma;
                    EELogSLogGamma[i] += mweights[i][j] * ELogSLogGamma;
                    EEMSDigamma[i] += mweights[i][j] * m * ESDigamma;
                }
            }
        }

        /// <summary>
        /// Perform the quadrature required for the VMP evidence message
        /// </summary>
        /// <param name="meanQ">Incoming message from m='mean'.</param>
        /// <param name="totalCountQ">Incoming message from s='totalCount'.</param>
        /// <returns>Vector of E[ LogGamma(s*m_k)].</returns>
        /// <remarks><para>
        /// The quadrature over 'totalCount' (which is Gamma-distributed) is 
        /// peformed by a change of variable x=log(s) followed by Gauss-Hermite 
        /// quadrature. The quadrature over m is performed using Gauss-Legendre. 
        /// </para></remarks>
        public static Vector EvidenceMessageExpectations(
            Dirichlet meanQ,
            Gamma totalCountQ)
        {
            // Get shape and scale of the distribution
            double at, bt;
            totalCountQ.GetShapeAndScale(out at, out bt);
            bt = 1 / bt; // want rate not scale

            // Mean in the transformed domain
            double proposalMean = totalCountQ.GetMeanLog();
            // Laplace approximation of variance in transformed domain 
            double proposalVariance = 1 / at;

            // Quadrature coefficient
            int nt = 32;
            Vector nodes = Vector.Zero(nt);
            Vector weights = Vector.Zero(nt);
            Vector expx = Vector.Zero(nt);
            if (!totalCountQ.IsPointMass)
            {
                Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                // Precompute weights for each m slice
                for (int i = 0; i < nt; i++)
                {
                    double x = nodes[i];
                    expx[i] = Math.Exp(x);
                    double p = at * x - bt * expx[i] - Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                    weights[i] *= Math.Exp(p);
                }
            }

            int nm = 20;
            Vector mnodes = Vector.Zero(nm);
            Vector mweight = Vector.Zero(nm);
            Quadrature.UniformNodesAndWeights(0, 1, mnodes, mweight);
            int K = meanQ.Dimension;
            Vector[] mweights = new Vector[K];
            Beta[] mkDist = new Beta[K];
            double[] EELogGamma = new double[K];
            for (int i = 0; i < K; i++)
            {
                mweights[i] = Vector.Copy(mweight);
                mkDist[i] = new Beta(meanQ.PseudoCount[i], meanQ.TotalCount - meanQ.PseudoCount[i]);
                EELogGamma[i] = 0;
            }

            for (int j = 0; j < nm; j++)
            {
                double m = mnodes[j];
                double ELogGamma = 0;
                if (totalCountQ.IsPointMass)
                    ELogGamma = MMath.GammaLn(m * totalCountQ.Point);
                else
                {
                    // Calculate expectations in x=log(s) space using Gauss-Hermite quadrature
                    for (int i = 0; i < nt; i++)
                    {
                        double x = nodes[i];
                        ELogGamma += weights[i] * (MMath.GammaLn(m * expx[i]) + x);
                    }
                    // Normalise and add removed components
                    double normalisation = Math.Pow(bt, at) / MMath.Gamma(at);
                    ELogGamma = normalisation * ELogGamma - proposalMean;
                }
                for (int i = 0; i < K; i++)
                {
                    mweights[i][j] *= Math.Exp(mkDist[i].GetLogProb(m));
                    EELogGamma[i] += mweights[i][j] * ELogGamma;
                }
            }
            return Vector.FromArray(EELogGamma);
        }


        // ------------------------------------ EP ----------------------------------

        private const string NotSupportedMessage = "Expectation Propagation does not currently support Dirichlet distributions with stochastic arguments";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, double)"]/*'/>
        public static double LogAverageFactor(Vector prob, Vector mean, double totalCount)
        {
            var temp = mean.Clone();
            mean.SetToProduct(mean, totalCount);
            var d = new Dirichlet(temp);
            return d.GetLogProb(prob);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogEvidenceRatio(Vector, Vector, double)"]/*'/>
        public static double LogEvidenceRatio(Vector prob, Vector mean, double totalCount)
        {
            return LogAverageFactor(prob, mean, totalCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Dirichlet, Vector, Dirichlet, double)"]/*'/>
        public static double LogAverageFactor(Dirichlet prob, Vector mean, Dirichlet to_prob, double totalCount)
        {
            return to_prob.GetLogAverageOf(prob);
        }

        public static Dirichlet ProbAverageConditional(Vector mean, double totalCount, Dirichlet result)
        {
            return ProbAverageLogarithm(mean, totalCount, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Dirichlet, Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor([SkipIfUniform] Dirichlet prob, [SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Dirichlet, Vector, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor([SkipIfUniform] Dirichlet prob, Vector mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Vector, Vector, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor(Vector prob, Vector mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Dirichlet, Dirichlet, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor([SkipIfUniform] Dirichlet prob, [SkipIfUniform] Dirichlet mean, double totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Vector, Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor(Vector prob, [SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="LogAverageFactor(Vector, Dirichlet, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double LogAverageFactor(Vector prob, [SkipIfUniform] Dirichlet mean, double totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="ProbAverageConditional(Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet ProbAverageConditional([SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="ProbAverageConditional(Dirichlet, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet ProbAverageConditional([SkipIfUniform] Dirichlet mean, double totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="ProbAverageConditional(Vector, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet ProbAverageConditional(Vector mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="ProbAverageConditional(Vector, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet ProbAverageConditional(Vector mean, double totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="MeanAverageConditional(Dirichlet, Gamma, Vector, Dirichlet)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet MeanAverageConditional([SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount, Vector prob, [SkipIfUniform] Dirichlet result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="MeanAverageConditional(Dirichlet, double, Vector, Dirichlet)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet MeanAverageConditional([SkipIfUniform] Dirichlet mean, double totalCount, Vector prob, [SkipIfUniform] Dirichlet result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="MeanAverageConditional(Dirichlet, double, Dirichlet, Dirichlet)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet MeanAverageConditional([SkipIfUniform] Dirichlet mean, double totalCount, [SkipIfUniform] Dirichlet prob, [SkipIfUniform] Dirichlet result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="MeanAverageConditional(Dirichlet, Gamma, Dirichlet, Dirichlet)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Dirichlet MeanAverageConditional(
            [SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Dirichlet prob, [SkipIfUniform] Dirichlet result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="TotalCountAverageConditional(Vector, Gamma, Vector, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional(Vector mean, [SkipIfUniform] Gamma totalCount, Vector prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="TotalCountAverageConditional(Vector, Gamma, Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional(Vector mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Dirichlet prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="TotalCountAverageConditional(Dirichlet, Gamma, Vector, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional([SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount, Vector prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DirichletOp"]/message_doc[@name="TotalCountAverageConditional(Dirichlet, Gamma, Dirichlet, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional(
            [SkipIfUniform] Dirichlet mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Dirichlet prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }
}
