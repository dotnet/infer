// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Tests
{
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif


    public class LaplaceTests
    {

        // Fails without sequential due to oscillation
        internal void NoisyCoinTest3()
        {
            int n = 5;
            double priorVar = 10;
            Variable<double> bias = Variable.GaussianFromMeanAndVariance(0, priorVar).Named("bias");
            Range r = new Range(n).Named("r");
            var flips = Variable.Array<bool>(r).Named("flips");
            double noiseVar = 0.1;
            double noiseStd = System.Math.Sqrt(noiseVar);
            double scale = System.Math.PI / System.Math.Sqrt(3) / noiseStd;
            bool useLogistic = false;
            using (Variable.ForEach(r))
            {
                if (!useLogistic)
                {
                    var noisyBias = Variable.GaussianFromMeanAndVariance(bias, noiseVar).Named("noisyBias");
                    flips[r] = (noisyBias > 0);
                }
                else
                {
                    // Bernoulli(Logistic(x)) is approximately the same as (N(x, pi^2/3) > 0)
                    flips[r] = Variable.Bernoulli(Variable.Logistic(bias * scale));
                }
            }
            flips.ObservedValue = Util.ArrayInit(n, i => false);
            r.AddAttribute(new Sequential());

            var engine = new InferenceEngine();
            // PointEstimate leads to same result as Laplace
            //bias.AddAttribute(new PointEstimate());
            if (false)
            {
                // VMP is stable and accurate for any n, but it is slow
                // n = 5:
                // VMP SJ98 = Gaussian(-2.602, 0.6334)  after 370 iterations
                // VMP JJ96 = Gaussian(-2.211, 0.1542)  after 214 iterations
                // Exact = Gaussian(-2.75, 3.428)
                engine.Algorithm = new VariationalMessagePassing();
                engine.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
                LogisticOp_SJ99.global_step = 0.4;
            }
            engine.ShowProgress = false;
            Gaussian biasExpected = new Gaussian(-2.884, 2);
            Gaussian biasPrev = Gaussian.Uniform();
            double diff = 0;
            for (int iter = 1; iter < 10; iter++)
            {
                engine.NumberOfIterations = iter;
                Gaussian biasPost = engine.Infer<Gaussian>(bias);
                diff = biasPost.MaxDiff(biasPrev);
                Console.WriteLine("{0}: {1} diff={2}", iter, biasPost, diff.ToString("g2"));
                biasPrev = biasPost;
            }
            //Assert.True(diff < 0.1);

            if (false)
            {
                // modified EP
                // unstable with parallel updates
                double mx = 0;
                double vx = priorVar;
                Gaussian[] msg = new Gaussian[n];
                double[] alphaPrev = new double[n];
                double mxPrev = 0;
                for (int iter = 0; iter < 40; iter++)
                {
                    double sumAlpha = 0;
                    double sumBeta = 0;
                    double sumPrec = 0;
                    double sumMeanTimesPrec = 0;
                    // q acts like damping rate, but constant q isn't great
                    double q = 1.1;
                    for (int i = 0; i < n; i++)
                    {
                        Gaussian x = Gaussian.FromMeanAndVariance(mx, vx);
                        Gaussian p = x / msg[i];
                        Gaussian noisyp = GaussianFromMeanAndVarianceOp.SampleAverageConditional(p, noiseVar);
                        Gaussian b = IsPositiveOp.XAverageConditional(false, noisyp);
                        Gaussian noisyb = GaussianFromMeanAndVarianceOp.MeanAverageConditional(b, noiseVar);
                        double beta = 1 / (noisyb.GetVariance() + p.GetVariance());
                        double alpha = beta * (noisyb.GetMean() - p.GetMean());
                        sumAlpha += alpha;
                        sumBeta += beta;
                        double a = q * mx + (1 - q) * p.GetMean();
                        double c = q / noisyb.Precision + (1 - q) / beta;
                        //noisyb.MeanTimesPrecision = noisyb.Precision * (mx + alpha / noisyb.Precision);
                        noisyb.MeanTimesPrecision = noisyb.Precision * (a + alpha * c);
                        msg[i] = noisyb;
                        sumPrec += noisyb.Precision;
                        sumMeanTimesPrec += noisyb.MeanTimesPrecision;
                        //Console.WriteLine("beta={0} taur={1}", beta, noisyp.Precision);
                        double dalpha = (alpha - alphaPrev[i]) / (mx - mxPrev);
                        double newQ = -c * dalpha;
                        Console.WriteLine("alpha = {0} newQ = {1}", alpha, newQ);
                        alphaPrev[i] = alpha;
                    }
                    mxPrev = mx;
                    double vr = 1 / sumBeta;
                    double mr = mx + vr * sumAlpha;
                    //vx = 1 / (1 / vr + 1 / priorVar);
                    //mx = vx * (mr / vr + 0);
                    vx = 1 / (sumPrec + 1 / priorVar);
                    mx = vx * (sumMeanTimesPrec + 0);
                    Gaussian xPost = Gaussian.FromMeanAndVariance(mx, vx);
                    Console.WriteLine("{0}: {1}", iter, xPost);
                }
            }

            if (true)
            {
                // Exact
                // Exact = Gaussian(-2.755, 3.418) for n=5
                double xmin = -20;
                double xmax = 1;
                int nsamples = 1000;
                double inc = (xmax - xmin) / (nsamples - 1);
                double b = 1 / System.Math.Sqrt(noiseVar);
                Gaussian prior = Gaussian.FromMeanAndVariance(0, priorVar);
                GaussianEstimator est = new GaussianEstimator();
                for (int s = 0; s < nsamples; s++)
                {
                    double x = xmin + s * inc;
                    double logp = prior.GetLogProb(x);
                    for (int i = 0; i < n; i++)
                    {
                        if (useLogistic)
                            logp += MMath.LogisticLn(-x * scale);
                        else
                            logp += MMath.NormalCdfLn(-x * b);
                    }
                    est.Add(x, System.Math.Exp(logp));
                    //Console.WriteLine("{0}: {1}", x.ToString("g4"), logp.ToString("g4"));
                }
                Gaussian xPost = est.GetDistribution(new Gaussian());
                Console.WriteLine("Exact = {0}", xPost);
            }

            if (false)
            {
                // Laplace is stable and fast for any n, even using parallel updates
                // but it is inaccurate - we could make it more accurate using higher derivatives
                // Laplace = Gaussian(-0.9198, 1.055)
                double mx = 0;
                double vx = priorVar;
                for (int iter = 0; iter < 10; iter++)
                {
                    double sumAlpha = 0;
                    double sumBeta = 0;
                    for (int i = 0; i < n; i++)
                    {
                        Gaussian p = Gaussian.FromMeanAndVariance(mx, 0);
                        Gaussian noisyp = GaussianFromMeanAndVarianceOp.SampleAverageConditional(p, noiseVar);
                        Gaussian b = IsPositiveOp.XAverageConditional(false, noisyp);
                        Gaussian noisyb = GaussianFromMeanAndVarianceOp.MeanAverageConditional(b, noiseVar);
                        double beta = 1 / (noisyb.GetVariance() + p.GetVariance());
                        double alpha = beta * (noisyb.GetMean() - p.GetMean());
                        sumAlpha += alpha;
                        sumBeta += beta;
                    }
                    double vr = 1 / sumBeta;
                    double mr = mx + vr * sumAlpha;
                    vx = 1 / (1 / vr + 1 / priorVar);
                    mx = vx * (mr / vr + 0);
                    Gaussian xPost = Gaussian.FromMeanAndVariance(mx, vx);
                    Console.WriteLine("{0}: {1}", iter, xPost);
                }
            }

            if (false)
            {
                // GAMP
                // will oscillate if n is large enough
                // result is Gaussian(-2.84, 2.258)
                double mx = 0;
                double vx = priorVar;
                double[] alpha = Util.ArrayInit(n, i => 0.0);
                double[] beta = Util.ArrayInit(n, i => 0.0);
                for (int iter = 0; iter < 50; iter++)
                {
                    double sumAlpha = 0;
                    double sumBeta = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double vp = vx;
                        double mp = mx - vp * alpha[i];
                        Gaussian p = Gaussian.FromMeanAndVariance(mp, vp);
                        Gaussian noisyp = GaussianFromMeanAndVarianceOp.SampleAverageConditional(p, noiseVar);
                        Gaussian b = IsPositiveOp.XAverageConditional(false, noisyp);
                        Gaussian noisyb = GaussianFromMeanAndVarianceOp.MeanAverageConditional(b, noiseVar);
                        beta[i] = 1 / (noisyb.GetVariance() + vp);
                        alpha[i] = beta[i] * (noisyb.GetMean() - mp);
                        sumAlpha += alpha[i];
                        sumBeta += beta[i];
                    }
                    double vr = 1 / sumBeta;
                    double mr = mx + vr * sumAlpha;
                    vx = 1 / (1 / vr + 1 / priorVar);
                    mx = vx * (mr / vr + 0);
                    Gaussian xPost = Gaussian.FromMeanAndVariance(mx, vx);
                    Console.WriteLine("{0}: {1}", iter, xPost);
                }
            }
        }

        internal void LearningAGaussianEP()
        {
            double meanMean = 0;
            double meanVariance = 1;
            double a = 100;
            double b = 100;
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(meanMean, meanVariance).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(a, 1 / b).Named("precision");
            Variable<int> xCount = Variable.New<int>().Named("xCount");
            Range item = new Range(xCount).Named("item");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(item);

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            Rand.Restart(0);
            double[] data = new double[1];
            for (int i = 0; i < data.Length; i++)
                data[i] = Rand.Normal(0, 1);
            data[0] = 5;
            x.ObservedValue = data;
            xCount.ObservedValue = data.Length;

            var gibbs = new GibbsSampling();
            gibbs.DefaultNumberOfIterations = 20000;
            engine.Algorithm = gibbs;
            if (true)
            {
                // check accuracy of constraints
                mean.AddAttribute(QueryTypes.Samples);
                precision.AddAttribute(QueryTypes.Samples);
                mean.AddAttribute(QueryTypes.Marginal);
                precision.AddAttribute(QueryTypes.Marginal);
                engine.NumberOfIterations = 4000000;
                gibbs.BurnIn = 1000;
                IList<double> meanSamples = engine.Infer<IList<double>>(mean, QueryTypes.Samples);
                IList<double> precisionSamples = engine.Infer<IList<double>>(precision, QueryTypes.Samples);
                GaussianEstimator meanEst = new GaussianEstimator();
                GaussianEstimator mrEst = new GaussianEstimator();
                GaussianEstimator mr2Est = new GaussianEstimator();
                GaussianEstimator m2rEst = new GaussianEstimator();
                GammaEstimator precEst = new GammaEstimator();
                for (int i = 0; i < meanSamples.Count; i++)
                {
                    meanEst.Add(meanSamples[i]);
                    precEst.Add(precisionSamples[i]);
                    mrEst.Add(meanSamples[i] * precisionSamples[i]);
                    mr2Est.Add(meanSamples[i] * precisionSamples[i] * precisionSamples[i]);
                    m2rEst.Add(meanSamples[i] * meanSamples[i] * precisionSamples[i]);
                }
                Console.WriteLine("mean = {0}", meanEst.GetDistribution(new Gaussian()));
                double x0 = data[0];
                double precisionMean = a / b;
                double k = meanVariance * precisionMean;
                double newMeanMean = meanEst.mva.Mean;
                double newMeanVariance = meanEst.mva.Variance;
                double newPrecisionMean = precEst.mva.Mean;
                double newPrecisionVariance = precEst.mva.Variance;
                double Erm = mrEst.mva.Mean;
                double covrm = Erm - newMeanMean * newPrecisionMean;
                double newMrVariance = mrEst.mva.Variance;
                double Er2m = mr2Est.mva.Mean;
                double alpha = newPrecisionMean * x0 - Erm;
                double newMeanMean2 = meanMean + meanVariance * alpha;
                double diff = x0 - newMeanMean;
                double alpha2 = newPrecisionMean * diff;
                double newMeanMean3 = (meanMean + meanVariance * newPrecisionMean * x0) / (1 + meanVariance * newPrecisionMean);
                // E[r^2(x-m)^2] = E[r^2]*x^2 - 2*x*E[r^2 m] + E[r^2 m^2]
                double Er2 = (newPrecisionMean * newPrecisionMean + newPrecisionVariance);
                double Em2 = newMeanMean * newMeanMean + newMeanVariance;
                double Er2m2 = newMrVariance + Erm * Erm;
                double Er2xm2 = Er2 * x0 * x0 - 2 * x0 * Er2m + Er2m2;
                double Exm2 = diff * diff + newMeanVariance;
                double pi = meanVariance * newPrecisionMean - meanVariance * meanVariance * newPrecisionVariance / (1 + meanVariance * newPrecisionMean);
                double newMeanMean4 = (meanMean + pi * x0) / (1 + pi);
                // this estimate is unstable since the rhs involves newMeanVariance
                double newMeanVariance2 = meanVariance - meanVariance * meanVariance * (alpha * alpha + newPrecisionMean - Er2xm2);
                //newMeanVariance2 = (meanVariance - meanVariance * meanVariance * (alpha * alpha + newPrecisionMean - (Er2 * x0 * x0 - 2 * x0 * newMr2Mean + (newMrMean * newMrMean + 0)))) / 1;
                //double newMeanVariance3 = meanVariance - meanVariance * meanVariance * (alpha * alpha + newPrecisionMean - Er2 * Exm2);
                //double newMeanVariance3 = meanVariance - meanVariance * meanVariance * (newPrecisionMean - newPrecisionVariance*diff*diff - newMeanVariance*newPrecisionMean*newPrecisionMean);
                //double newMeanVariance3 = (meanVariance - meanVariance * meanVariance * (newPrecisionMean - newPrecisionVariance * diff * diff)) / (1 - meanVariance * meanVariance * newPrecisionMean * newPrecisionMean);
                // this is worse than above
                //newMeanVariance2 = (meanVariance - meanVariance * meanVariance * (newPrecisionMean - newPrecisionVariance * diff * diff)) / (1 - meanVariance * meanVariance * Er2);
                double newMeanVariance4 = (meanVariance - meanVariance * meanVariance * (alpha * alpha + newPrecisionMean) + meanVariance * meanVariance * (
                    Er2 * x0 * x0 - 2 * x0 * Er2m
                    + Er2 * newMeanMean * newMeanMean - 2 * newPrecisionMean * newPrecisionMean * newMeanMean * newMeanMean + 2 * Erm * Erm))
                    / (1 - meanVariance * meanVariance * Er2);
                double newMeanVariance5 = meanVariance * (1 + covrm * diff) / (1 + meanVariance * newPrecisionMean);
                double Er2m_approx = Er2 * newMeanMean + 2 * newPrecisionMean * covrm;
                double Er2m_approx2 = newPrecisionMean * Erm + newPrecisionVariance * x0 - covrm / meanVariance;
                //double Erm_approx2 = (meanMean * newPrecisionMean + meanVariance * (Er2*diff + 2*newPrecisionMean*newPrecisionMean*newMeanMean)) / (1 + 2 * meanVariance * newPrecisionMean);
                double covrm_approx = meanVariance * newPrecisionVariance * diff / (1 + meanVariance * newPrecisionMean);
                double Erm2 = m2rEst.mva.Mean;
                double Erm2_approx = newMeanMean * Erm + newMeanVariance * newPrecisionMean + newMeanMean * covrm;
                double delta_rm2 = Erm2 - Erm2_approx;
                double delta_r2m = Er2m - Er2m_approx;
                //- newMeanVariance * newPrecisionVariance - 2 * covrm * covrm;
                double delta_rm2_approx2 = -newMeanMean * covrm + meanVariance * (
                    (Er2m_approx - newPrecisionMean * Erm - newPrecisionMean * covrm) * x0 - newPrecisionVariance * Em2 + covrm * covrm)
                    / (1 + meanVariance * newPrecisionMean);
                double delta_rm2_approx = -meanVariance * newMeanVariance * newPrecisionVariance / (1 + meanVariance * newPrecisionMean);
                double Er2m2_approx = Er2 * Em2 + 4 * newPrecisionMean * newMeanMean * covrm + 2 * covrm * covrm + 2 * newPrecisionMean * delta_rm2 + 2 * newMeanMean * delta_r2m;
                double delta_r2m2 = Er2m2 - Er2m2_approx;
                //double Er2m2_approx = Er2 * Em2 - 2 * newPrecisionMean * newPrecisionMean * newMeanMean * newMeanMean + 2 * Erm * Erm;
                double Er2m2_approx2 = Er2 * Em2 - 2 * Er2 * newMeanMean * newMeanMean + 2 * Er2m * newMeanMean;
                double delta_rm2_approx3 = meanVariance * (delta_r2m * diff - delta_r2m2 - newMeanVariance * newPrecisionVariance) / (1 + meanVariance * newPrecisionMean);
                double newMeanMean5 = newMeanMean4 + meanVariance * meanVariance * delta_r2m / (1 + meanVariance * newPrecisionMean) / (1 + pi);
                double covrm_approx2 = meanVariance * (newPrecisionVariance * diff - delta_r2m) / (1 + meanVariance * newPrecisionMean);
                double newMeanVariance6 = meanVariance * (1 + covrm * diff - delta_rm2) / (1 + meanVariance * newPrecisionMean);
                double Er2m2_approx3 = Er2 * Em2 - 2 * newPrecisionMean * newPrecisionMean * Em2 + 2 * newPrecisionMean * Erm2;
                double Er2m2_approx4 = Er2 * Em2 - 2 * newPrecisionMean * newPrecisionMean * Em2 + 2 * newPrecisionMean * Erm2;
                double Er2m2_approx5 = Er2 * Em2 - 2 * Er2 * newMeanMean * newMeanMean + 2 * Er2m_approx * newMeanMean;
                double Er2m2_approx6 = (Er2 * Em2 - 2 * newPrecisionMean * newPrecisionMean * (Em2 - meanVariance) + 2 * newPrecisionMean *
                    (newMeanMean * Erm + meanVariance * (Er2m - newPrecisionMean * Erm) * x0 + meanVariance * Erm * Erm)) / (1 + meanVariance);
                double Er2m2_approx7 = 2 * newMeanMean * (Er2m - Erm * newPrecisionMean) + 2 * newPrecisionMean * (Erm2 - Erm * newMeanMean)
                    + newMeanVariance * newPrecisionVariance - newMeanVariance * newPrecisionMean * newPrecisionMean - newPrecisionVariance * newMeanMean * newMeanMean + newPrecisionMean * newPrecisionMean * newMeanMean * newMeanMean;
                double Er2m2_approx8 = Er2 * Em2 + 2 * newPrecisionMean * (Erm2 - Erm * newMeanMean - newPrecisionMean * newMeanVariance)
                    + 2 * newMeanMean * (Er2m - Erm * newPrecisionMean - newPrecisionVariance * newMeanMean) + 2 * covrm * covrm;
                // this is bad for small vm
                //double Er2xm2_approx = (newMeanVariance - meanVariance) / (meanVariance * meanVariance) + newPrecisionMean + alpha * alpha;
                //double Er2xm2_approx = 
                double newPrecisionMean2 = (a + 0.5 / (k + 1)) / (b + 0.5 * diff * diff / (k + 1) / (k + 1));
                double newPrecisionVariance2 = newPrecisionMean2 / (b + 0.5 * diff * diff / (k + 1) / (k + 1));
                newPrecisionVariance2 = newPrecisionMean / (b + 0.5 * (diff * diff + newMeanVariance));
                double newPrecisionMean3 = (a + 0.5 + covrm * diff) / (b + 0.5 * (diff * diff + newMeanVariance));
                double newPrecisionMean4 = (a - 0.5 + newMeanVariance / meanVariance) / (b + 0.5 * (diff * diff - newMeanVariance));
                double newPrecisionMean5 = (a + 0.5 + covrm * diff - 0.5 * delta_rm2) / (b + 0.5 * (diff * diff + newMeanVariance));
                double newPrecisionVariance3 = newPrecisionMean * (1 + covrm * diff) / (b + 0.5 * (diff * diff + newMeanVariance));
                //double newPrecisionVariance4 = (a + 1.5) / b * newPrecisionMean - 0.5 / b * Er2xm2_approx - newPrecisionMean*newPrecisionMean;
                double newPrecisionVariance4 = (newPrecisionMean + newPrecisionMean * covrm * diff - covrm * covrm) / (b + 0.5 * (diff * diff + newMeanVariance));
                double newPrecisionVariance5 = (newPrecisionMean - 0.5 * newPrecisionMean * delta_rm2
                    + newPrecisionMean * covrm * diff + diff * delta_r2m - covrm * covrm - 0.5 * delta_r2m2)
                    / (b + 0.5 * (diff * diff + newMeanVariance));
                double newPrecisionVariance6 = (newPrecisionMean - 0.5 * newPrecisionMean * delta_rm2 + diff * delta_r2m / (1 + meanVariance * newPrecisionMean)
                    - covrm * covrm - 0.5 * delta_r2m2)
                    / (b + 0.5 * (diff * diff + newMeanVariance) - diff * diff * meanVariance * newPrecisionMean / (1 + meanVariance * newPrecisionMean));
                double newPrecisionVariance7 = (newPrecisionMean - 0.5 * newPrecisionMean * delta_rm2 +
                    (covrm + diff) * delta_r2m / (1 + meanVariance * newPrecisionMean) - 0.5 * delta_r2m2 * 0)
                    / (b + 0.5 * (diff * diff + newMeanVariance) - diff * diff * meanVariance * newPrecisionMean / (1 + meanVariance * newPrecisionMean)
                    + meanVariance * diff * covrm / (1 + meanVariance * newPrecisionMean));
                double newPrecisionVariance8 = (newPrecisionMean - 0.5 * newPrecisionMean * delta_rm2 +
                    diff * (newPrecisionMean * covrm + delta_r2m * 0) + meanVariance * delta_r2m * 0 / (1 + meanVariance * newPrecisionMean) - 0.5 * delta_r2m2 * 0)
                    / (b + 0.5 * (diff * diff + newMeanVariance) + meanVariance * diff * covrm / (1 + meanVariance * newPrecisionMean));
                Console.WriteLine("   m = {0} {1} {2} {3}", newMeanMean2, newMeanMean3, newMeanMean4, newMeanMean5);
                Console.WriteLine("   v = {0} {1} {2} {3}", newMeanVariance2, newMeanVariance4, newMeanVariance5, newMeanVariance6);
                //Console.WriteLine("mr^2 = {0} {1}", Er2m, Er2m_approx);
                Console.WriteLine("delta(r^2m) = {0} {1}", delta_r2m, Er2m_approx2 - Er2m_approx);
                //Console.WriteLine("m^2r^2 = {0} {1}", Er2m2, Er2m2_approx);
                Console.WriteLine("delta(rm^2) = {0} {1} {2} {3}", delta_rm2, delta_rm2_approx, delta_rm2_approx2, delta_rm2_approx3);
                Console.WriteLine("delta(r^2m^2) = {0}", delta_r2m2);
                Console.WriteLine("cov(m^2,r^2) = {0} {1} {2} {3} {4} {5} {6} {7} {8}", Er2m2 - Er2 * Em2, Er2m2_approx - Er2 * Em2, Er2m2_approx2 - Er2 * Em2,
                    Er2m2_approx3 - Er2 * Em2, Er2m2_approx4 - Er2 * Em2, Er2m2_approx5 - Er2 * Em2, Er2m2_approx6 - Er2 * Em2,
                    Er2m2_approx7 - Er2 * Em2, Er2m2_approx8 - Er2 * Em2);
                //Console.WriteLine("E[r^2(x-m)^2] = {0} {1}", Er2xm2, Er2xm2_approx);
                Console.WriteLine("mean*precision = {0}", mrEst.GetDistribution(new Gaussian()));
                Console.WriteLine("cov(r,m) = {0} {1} {2}", covrm, covrm_approx, covrm_approx2);
                Console.WriteLine("precision = {0}", precEst.GetDistribution(new Gamma()));
                Console.WriteLine("   m = {0} {1} {2} {3} {4}", newPrecisionMean, newPrecisionMean2, newPrecisionMean3, newPrecisionMean4, newPrecisionMean5);
                Console.WriteLine("   v = {0} {1} {2} {3} {4} {5} {6} {7}", newPrecisionVariance, newPrecisionVariance2,
                    newPrecisionVariance3, newPrecisionVariance4, newPrecisionVariance5, newPrecisionVariance6, newPrecisionVariance7, newPrecisionVariance8);
                return;
            }
            Gaussian meanExpected = engine.Infer<Gaussian>(mean);
            Gamma precisionExpected = engine.Infer<Gamma>(precision);

            engine.NumberOfIterations = 50;
            engine.Algorithm = new ExpectationPropagation();
            //engine.Algorithm = new VariationalMessagePassing();
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("GaussianOp:");
            Console.WriteLine("mean = {0} (should be {1})", meanActual, meanExpected);
            Console.WriteLine("precision = {0} (should be {1})", precisionActual, precisionExpected);
            Console.WriteLine("Laplace:");
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Laplace));
            meanActual = engine.Infer<Gaussian>(mean);
            precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("mean = {0} (should be {1})", meanActual, meanExpected);
            Console.WriteLine("precision = {0} (should be {1})", precisionActual, precisionExpected);
            Console.WriteLine("new method:");
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Test));
            meanActual = engine.Infer<Gaussian>(mean);
            precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("mean = {0} (should be {1})", meanActual, meanExpected);
            Console.WriteLine("precision = {0} (should be {1})", precisionActual, precisionExpected);
        }

        internal void GaussianSharedMeanTest()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<int> xCount = Variable.New<int>().Named("xCount");
            Range item = new Range(xCount).Named("item");
            VariableArray<double> precision = Variable.Array<double>(item).Named("precision");
            precision[item] = Variable.GammaFromShapeAndRate(1, 1).ForEach(item);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(mean, precision[item]);

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            //engine.Compiler.GivePriorityTo(typeof(GaussianOp_Slow));
            Rand.Restart(1);
            double[] data = new double[10];
            for (int i = 0; i < data.Length; i++)
                data[i] = Rand.Normal(0, 1);
            x.ObservedValue = data;
            xCount.ObservedValue = data.Length;

            var gibbs = new GibbsSampling();
            gibbs.DefaultNumberOfIterations = 20000;
            engine.Algorithm = gibbs;
            Gaussian precisionExpected = engine.Infer<Gaussian>(mean);

            engine.Algorithm = new ExpectationPropagation();
            //engine.Algorithm = new VariationalMessagePassing();
            for (int iter = 1; iter < 50; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0} {1}", iter, engine.Infer(mean));
            }
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Console.WriteLine("GaussianOp:");
            Console.WriteLine("mean = {0} (should be {1})", meanActual, precisionExpected);
            Console.WriteLine("Laplace:");
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Laplace));
            meanActual = engine.Infer<Gaussian>(mean);
            Console.WriteLine("mean = {0} (should be {1})", meanActual, precisionExpected);
            Console.WriteLine("new method:");
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Test));
            meanActual = engine.Infer<Gaussian>(mean);
            Console.WriteLine("mean = {0} (should be {1})", meanActual, precisionExpected);
        }

        internal void GaussianSharedPrecisionTest()
        {
            Variable<double> precision = Variable.GammaFromShapeAndRate(1, 1).Named("precision");
            Variable<int> xCount = Variable.New<int>().Named("xCount");
            Range item = new Range(xCount).Named("item");
            VariableArray<double> mean = Variable.Array<double>(item).Named("mean");
            mean[item] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(item);
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(mean[item], precision);

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            Rand.Restart(1);
            double[] data = new double[100];
            for (int i = 0; i < data.Length; i++)
                data[i] = Rand.Normal(0, 1);
            x.ObservedValue = data;
            xCount.ObservedValue = data.Length;

            var gibbs = new GibbsSampling();
            gibbs.DefaultNumberOfIterations = 20000;
            engine.Algorithm = gibbs;
            Gamma precisionExpected = engine.Infer<Gamma>(precision);
            GaussianOp.ForceProper = false;

            engine.Algorithm = new ExpectationPropagation();
            //engine.Algorithm = new VariationalMessagePassing();
            for (int iter = 1; iter < 50; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0} {1}", iter, engine.Infer(precision));
            }
            Gamma precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("GaussianOp:");
            Console.WriteLine("precision = {0} (should be {1})", precisionActual, precisionExpected);
            Console.WriteLine("variance = {0} (should be {1})", precisionActual.GetVariance(), precisionExpected.GetVariance());
            Console.WriteLine("Laplace:");
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Laplace));
            precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("precision = {0} (should be {1})", precisionActual, precisionExpected);
            Console.WriteLine("variance = {0} (should be {1})", precisionActual.GetVariance(), precisionExpected.GetVariance());
            Console.WriteLine("new method:");
            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Test));
            precisionActual = engine.Infer<Gamma>(precision);
            Console.WriteLine("precision = {0} (should be {1})", precisionActual, precisionExpected);
            Console.WriteLine("variance = {0} (should be {1})", precisionActual.GetVariance(), precisionExpected.GetVariance());
        }

        [FactorMethod(typeof(Factor), "Gaussian")]
        [Quality(QualityBand.Experimental)]
        public static class GaussianOp_Test
        {
            public static bool ForceProper = false;

            public static Gaussian MeanAverageConditional_Slow(double sample, Gaussian mean, Gamma precision)
            {
                double precisionMean = precision.GetMean();
                double meanMean, meanVariance;
                mean.GetMeanAndVariance(out meanMean, out meanVariance);
                double newMeanMean = meanMean;
                double newMeanVariance = meanVariance;
                double newPrecisionVariance = 0;
                double covrm = 0;
                double tolerance = 1e-8;
                for (int i = 0; i < 100; i++)
                {
                    double oldMeanMean = newMeanMean;
                    double oldMeanVariance = newMeanVariance;
                    double diff = sample - newMeanMean;
                    double newPrecisionMean = (precision.Shape + 0.5 + covrm * diff) / (precision.Rate + 0.5 * (diff * diff + newMeanVariance));
                    if (newPrecisionMean < 0)
                        throw new Exception();
                    double k = meanVariance * newPrecisionMean;
                    double pi = meanVariance * newPrecisionMean - meanVariance * meanVariance * newPrecisionVariance / (1 + k);
                    newMeanMean = (meanMean + pi * sample) / (1 + pi);
                    covrm = meanVariance * newPrecisionVariance * diff / (1 + k);
                    newMeanVariance = meanVariance * (1 + covrm * diff) / (1 + k);
                    if (newMeanVariance < 0)
                        throw new Exception();
                    newPrecisionVariance = newPrecisionMean * (1 + covrm * diff)
                        / (precision.Rate + 0.5 * (diff * diff + newMeanVariance) + meanVariance * covrm * diff / (1 + k));
                    if (newPrecisionVariance < 0)
                        throw new Exception();
                    //Console.WriteLine("{0} ({1}, {2}) ({3}, {4})", i, newMeanMean, newMeanVariance, newPrecisionMean, newPrecisionVariance);
                    if (MMath.AbsDiff(newMeanMean, oldMeanMean) < tolerance && MMath.AbsDiff(newMeanVariance, oldMeanVariance, 1e-10) < tolerance)
                        break;
                }
                Gaussian result = Gaussian.FromMeanAndVariance(newMeanMean, newMeanVariance);
                result.SetToRatio(result, mean, ForceProper);
                return result;
            }

            public static Gaussian MeanAverageConditional(double sample, Gaussian mean, Gamma precision, Gamma to_precision)
            {
                double meanMean, meanVariance;
                mean.GetMeanAndVariance(out meanMean, out meanVariance);
                Gamma precisionPost = precision * to_precision;
                double newPrecisionMean, newPrecisionVariance;
                precisionPost.GetMeanAndVariance(out newPrecisionMean, out newPrecisionVariance);

                double k = meanVariance * newPrecisionMean;
                double pi = k - meanVariance * meanVariance * newPrecisionVariance / (1 + k);
                double newMeanMean = (meanMean + pi * sample) / (1 + pi);
                double diff = sample - newMeanMean;
                double covrm = meanVariance * newPrecisionVariance * diff / (1 + k);
                double newMeanVariance = meanVariance * (1 + covrm * diff) / (1 + k);
                if (newMeanVariance < 0)
                    throw new Exception();
                Gaussian result = Gaussian.FromMeanAndVariance(newMeanMean, newMeanVariance);
                result.SetToRatio(result, mean, ForceProper);
                return result;
            }

            public static Gamma PrecisionAverageConditional(double sample, Gaussian mean, Gamma precision, Gaussian to_mean, Gamma to_precision)
            {
                double meanMean, meanVariance;
                mean.GetMeanAndVariance(out meanMean, out meanVariance);
                Gaussian meanPost = mean * to_mean;
                double newMeanMean, newMeanVariance;
                meanPost.GetMeanAndVariance(out newMeanMean, out newMeanVariance);
                Gamma precisionPost = precision * to_precision;
                double newPrecisionMean, newPrecisionVariance;
                precisionPost.GetMeanAndVariance(out newPrecisionMean, out newPrecisionVariance);

                //Console.WriteLine("newPrecisionMean = {0}, newPrecisionVariance = {1}", newPrecisionMean, newPrecisionVariance);
                double diff = sample - newMeanMean;
                double k = meanVariance * newPrecisionMean;
                double covrm = meanVariance * newPrecisionVariance * diff / (1 + k);
                newPrecisionMean = (precision.Shape + 0.5 + covrm * diff) / (precision.Rate + 0.5 * (diff * diff + newMeanVariance));
                if (newPrecisionMean < 0)
                    throw new Exception();
                newPrecisionVariance = newPrecisionMean * (1 + covrm * diff)
                    / (precision.Rate + 0.5 * (diff * diff + newMeanVariance / (1 + k)) + meanVariance * covrm * diff / (1 + k));
                if (newPrecisionVariance < 0)
                    throw new Exception();
                //Console.WriteLine("newPrecisionMean = {0}, newPrecisionVariance = {1}", newPrecisionMean, newPrecisionVariance);
                Gamma precMarginal = Gamma.FromMeanAndVariance(newPrecisionMean, newPrecisionVariance);
                Gamma result = new Gamma();
                result.SetToRatio(precMarginal, precision, ForceProper);
                return result;
            }
        }

        // Test GaussianOp_Laplace
        internal void GaussianPrecisionTest()
        {
            if (false)
            {
                Gamma to_precision = Gamma.FromShapeAndRate(122676.37264992672, 340603969.95259166);
                to_precision = Gamma.Uniform();
                // fails because a single quadrature node (the first one) gets all the mass
                //Console.WriteLine(GaussianOp.PrecisionAverageConditional(Gaussian.FromNatural(0.48989640228448528, 0.000000074827705636805314), Gaussian.FromNatural(67.293178500755857, 0.27970616812644788), Gamma.FromShapeAndRate(3.0693761223141034, 209.64245128846574), to_precision));
                Console.WriteLine(GaussianOp_Laplace.Q(Gaussian.PointMass(2.424), Gaussian.FromNatural(0, 1.7027721333788637), Gamma.FromShapeAndRate(1.0, 0.44045822995611544),
                                                       Gamma.Uniform()));
                Console.WriteLine(GaussianOp_Laplace.PrecisionAverageConditional(Gaussian.FromNatural(1.1546406519734849, 0.18248079449015167), Gaussian.PointMass(0),
                                                                                 new Gamma(1, 1), Gamma.FromShapeAndRate(0.20358996624813053, 0.58600373610663059)));
                Console.WriteLine(GaussianOp_Laplace.SampleAverageConditional(Gaussian.FromNatural(1.2525727229679782, 0.27878308545210362), Gaussian.PointMass(0),
                                                                              Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(0.44966183478609556, 0.83573466672245378)));
                Console.WriteLine(GaussianOp_Laplace.SampleAverageConditional(Gaussian.FromNatural(1.0008312247412725, 0.035195985080939886), Gaussian.PointMass(0),
                                                                              Gamma.FromShapeAndRate(1.0, 1.1859890443593031),
                                                                              Gamma.FromShapeAndRate(0.46155516707638966, 1.0296186547344195)));
                Console.WriteLine(GaussianOp_Laplace.SampleAverageConditional(Gaussian.FromNatural(0.0, 1.0538107263976093), Gaussian.PointMass(2.4244775414356945),
                                                                              Gamma.FromShapeAndRate(1.0, 0.44045822995611544),
                                                                              Gamma.FromShapeAndRate(0.59074455350400923, 0.54971937277405969)));
            }

            double mm = 1, vm = 2;
            double a = 3, b = 1;
            bool extremeCheck = false;
            if (extremeCheck)
            {
                Gaussian xPrior = Gaussian.FromNatural(-2.3874057896477092, 0.0070584383295080044);
                Gaussian meanPrior = Gaussian.FromNatural(1.3999879871144227, 0.547354438587195);
                mm = xPrior.GetMean() - meanPrior.GetMean();
                vm = xPrior.GetVariance() + meanPrior.GetVariance();
                Console.WriteLine("mm={0} vm={1}", mm, vm);
            }
            if (false)
            {
                Gaussian xPrior = Gaussian.FromNatural(1.1546406519734849, 0.18248079449015167);
                mm = xPrior.GetMean();
                vm = xPrior.GetVariance();
                a = 1;
                b = 1;
            }
            if (false)
            {
                // BugsRats case (has infinite variance if x is unobserved)
                mm = 0;
                vm = 1e4;
                a = 1e-3;
                b = 1e-3;
            }
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(mm, vm).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndRate(a, b).Named("precision");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision).Named("x");
            x.ObservedValue = 0;
            //Variable.ConstrainEqualRandom(x, new Gaussian(4, 5));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GaussianOp));
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            Gaussian xExpected = engine.Infer<Gaussian>(x);
            Gaussian meanExpected = engine.Infer<Gaussian>(mean);
            Gamma precExpected = engine.Infer<Gamma>(precision);
            double evExpected = engine.Infer<Bernoulli>(evidence).LogOdds;

            engine.Compiler.GivePriorityTo(typeof(GaussianOp_Laplace));
            //engine.Compiler.GivePriorityTo(typeof(ExpOp3));
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Gaussian meanActual = engine.Infer<Gaussian>(mean);
            Gamma precActual = engine.Infer<Gamma>(precision);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("mean = {0} should be {1}", meanActual, meanExpected);
            Console.WriteLine("precision = {0} should be {1}", precActual, precExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
        }

        internal void ProductTest()
        {
            // exact evidence here is  -2.834075861624064
            // exact posterior for A is N( 1.440115377416470, 0.826021607530110 )
            double ma = 1, va = 2;
            double mb = 3, vb = 4;
            double mx = 5, vx = 6;
            if (false)
            {
                // a good product operator should give good results for the case b = N(0,1)
                // note: the true posterior is bimodal, but it should pick one of them (as if there were an implicit ConstrainPositive)
                mb = 0;
                vb = 1;
                va = 100000;
                vx /= 100;
            }
            if (false)
            {
                // bad case for VMP
                ma = 0.1;
                va = 1;
                mb = 0;
                vb = 1;
                mx = 0.5;
                vx = 1e-4;
            }
            if (false)
            {
                ma = -0.002463;
                va = 0.5;
                mb = 0;
                vb = 1;
                mx = 0.6041;
                vx = 1;
            }
            if (true)
            {
                // bad case for Laplace
                ma = 0.0006117;
                va = 0.761;
                mb = 0.1059;
                vb = 3.803;
                mx = -1.93;
                vx = 5.229;
            }
            //vb = 1;
            //vx = 1;
            //vx = 100000;
            //va = 1e-4;
            //vb = 1e-4;
            //va /= 100;
            //vb /= 10;
            //ma = 0;
            //mx = Math.Sqrt(vx + va*vb);
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            Variable<double> a = Variable.GaussianFromMeanAndVariance(ma, va).Named("a");
            Variable<double> b = Variable.GaussianFromMeanAndVariance(mb, vb).Named("b");
            Variable<double> x = a * b;
            x.Name = "x";
            Variable.ConstrainEqualRandom(x, new Gaussian(mx, vx));
            block.CloseBlock();
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Slow));
            //GaussianProductOp_Slow.QuadratureNodeCount = 100;
            Gaussian aExpected = engine.Infer<Gaussian>(a);
            Gaussian bExpected = engine.Infer<Gaussian>(b);
            Gaussian xExpected = engine.Infer<Gaussian>(x);
            double evExpected = engine.Infer<Bernoulli>(evidence).LogOdds;
            //engine.Algorithm = new ExpectationPropagation();
            //engine.Algorithm = new VariationalMessagePassing();
            //a.AddAttribute(new PointEstimate());
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Laplace));
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0}: {1} {2}", iter, engine.Infer<Gaussian>(a), engine.Infer<Gaussian>(b));
            }
            // must figure out why Laplace (modified) is worse than Laplace2
            //GaussianProductOp_Laplace.modified = false;
            //engine.Algorithm = new GibbsSampling();
            Gaussian aActual = engine.Infer<Gaussian>(a);
            Gaussian bActual = engine.Infer<Gaussian>(b);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("a = {0} should be {1}", aActual, aExpected);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            //Console.WriteLine(aExpected / new Gaussian(ma, va));
        }

        // This is a test of the product factor under EP, using a realistic yet simple model
        internal void NoisyRegressionTest()
        {
            Rand.Restart(10);
            int d = 1;
            Gaussian wPrior = new Gaussian(0, 1);
            Gaussian xPrior = new Gaussian(0, 1);
            double[] wTrue = Util.ArrayInit(d, i => wPrior.Sample());
            double yVariance = 1;
            double xVariance = 1;
            double[] yData;
            double[][] xNoisyData;
            int n = 10;
            NoisyRegressionData(n, wTrue, xPrior, yVariance, xVariance, out yData, out xNoisyData);
            IList<Gaussian> wExpected = NoisyRegressionSampler(wPrior, xPrior, yVariance, xVariance, yData, xNoisyData);

            Range item = new Range(yData.Length).Named("item");
            Range dim = new Range(d).Named("dim");
            var w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.Random(wPrior).ForEach(dim);
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.Random(xPrior).ForEach(item, dim);
            var xNoisy = Variable.Array(Variable.Array<double>(dim), item).Named("xNoisy");
            xNoisy[item][dim] = Variable.GaussianFromMeanAndVariance(x[item][dim], xVariance);
            var y = Variable.Array<double>(item).Named("y");
            using (Variable.ForEach(item))
            {
                var products = Variable.Array<double>(dim).Named("products");
                products[dim] = x[item][dim] * w[dim];
                var sum = Variable.Sum(products).Named("sum");
                y[item] = Variable.GaussianFromMeanAndVariance(sum, yVariance);
            }
            xNoisy.ObservedValue = xNoisyData;
            y.ObservedValue = yData;
            w.AddAttribute(new PointEstimate());

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            Type[] operators = new Type[] { typeof(GaussianProductOp), typeof(GaussianProductOp_Laplace), typeof(GaussianProductOp_Laplace2), typeof(GaussianProductOp_LaplaceProp), typeof(GaussianProductOp_SHG09), typeof(GaussianProductOp_EM) };
            foreach (Type type in operators)
            {
                Console.WriteLine(type.Name);
                engine.Compiler.GivePriorityTo(type);
                // SHG does poorly for d=5, n=100 - perhaps a local optimum? try initializing there
                var wActual = engine.Infer<IList<Gaussian>>(w);
                Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", StringUtil.ToString(wExpected)));
                Console.WriteLine("error = {0}", ((Diffable)wExpected).MaxDiff(wActual));
                Console.WriteLine();
            }
            Console.WriteLine(StringUtil.JoinColumns("wTrue = ", StringUtil.ToString(wTrue)));
        }
        private IList<Gaussian> NoisyRegressionSampler(Gaussian wPrior, Gaussian xPrior, double yVariance, double xVariance, double[] y, double[][] xNoisy)
        {
            Console.WriteLine("Sampling");
            int n = xNoisy.Length;
            int d = xNoisy[0].Length;
            GaussianEstimator[] est = Util.ArrayInit(d, i => new GaussianEstimator());
            double[] w = new double[d];
            double[][] x = Util.ArrayInit(n, i => new double[d]);
            for (int iter = 0; iter < 100000; iter++)
            {
                if (iter > 0 && iter % 10000 == 0)
                    Console.WriteLine("iter = {0}", iter);
                // sample x
                for (int i = 0; i < n; i++)
                {
                    Gaussian yB = new Gaussian(y[i], yVariance);
                    double sum = 0;
                    for (int k = 0; k < d; k++)
                    {
                        sum += x[i][k] * w[k];
                    }
                    for (int k = 0; k < d; k++)
                    {
                        double partialSum = sum - x[i][k] * w[k];
                        Gaussian sumB = DoublePlusOp.AAverageConditional(yB, partialSum);
                        Gaussian productB = GaussianProductOp.AAverageConditional(sumB, w[k]);
                        Gaussian xNoisyB = new Gaussian(xNoisy[i][k], xVariance);
                        Gaussian xPost = xPrior * productB * xNoisyB;
                        x[i][k] = xPost.Sample();
                        sum = partialSum + x[i][k] * w[k];
                    }
                }
                // sample w
                for (int j = 0; j < d; j++)
                {
                    Gaussian wPost = wPrior;
                    for (int i = 0; i < n; i++)
                    {
                        Gaussian yB = new Gaussian(y[i], yVariance);
                        double sum = 0;
                        for (int k = 0; k < d; k++)
                        {
                            sum += x[i][k] * w[k];
                        }
                        Gaussian sumB = DoublePlusOp.AAverageConditional(yB, sum - x[i][j] * w[j]);
                        Gaussian productB = GaussianProductOp.AAverageConditional(sumB, x[i][j]);
                        wPost *= productB;
                    }
                    w[j] = wPost.Sample();
                    if (iter > 1000)
                    {
                        est[j].Add(wPost);
                    }
                }
            }
            return new GaussianArray(d, i => est[i].GetDistribution(new Gaussian()));
        }
        private void NoisyRegressionData(int nSamples, double[] wTrue, Gaussian xPrior, double yVariance, double xVariance, out double[] y, out double[][] xNoisy)
        {
            y = new double[nSamples];
            xNoisy = new double[nSamples][];
            int d = wTrue.Length;
            for (int i = 0; i < nSamples; i++)
            {
                xNoisy[i] = new double[d];
                double sum = 0;
                for (int k = 0; k < d; k++)
                {
                    double x = xPrior.Sample();
                    sum += x * wTrue[k];
                    xNoisy[i][k] = new Gaussian(x, xVariance).Sample();
                }
                y[i] = new Gaussian(sum, yVariance).Sample();
            }
        }

        private Gaussian[] PlusProduct(int N, double trueMean, double trueVariance)
        {
            Gaussian g = Gaussian.FromMeanAndVariance(trueMean, trueVariance);
            double[] sample = Util.ArrayInit(N, i => g.Sample());

            var mean = Variable.Random(Gaussian.FromMeanAndVariance(0, 1));
            var sigma = Variable.Random(Gaussian.FromMeanAndVariance(1, 1e8));

            Range n = new Range(N);
            n.AddAttribute(new Sequential());
            var x = Variable.Array<double>(n);

            using (Variable.ForEach(n))
            {
                x[n] = (mean + sigma * Variable.GaussianFromMeanAndPrecision(0, 1));
            }

            x.ObservedValue = sample;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_LaplaceProp));
            //engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_Slow));
            engine.ShowProgress = false;
            for (int iter = 1; iter <= 5; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("{0}: {1} {2}", iter, engine.Infer(mean), engine.Infer(sigma));
            }

            return new Gaussian[] { engine.Infer<Gaussian>(mean), engine.Infer<Gaussian>(sigma) };
        }

        [Fact]
        public void PlusProductTest()
        {
            Rand.Restart(0);
            double trueMean = 1.2;
            double trueVariance = 2.3;
            int N = 10000;
            var result = PlusProduct(N, trueMean, trueVariance);

            Console.WriteLine("True mean = {0}, Inferred = {1}", trueMean, result[0]);
            Console.WriteLine("True sigma = {0}, Inferred = {1}", System.Math.Sqrt(trueVariance), result[1]);

            Assert.Equal(trueMean, result[0].GetMean(), 1e-2);
            Assert.Equal(System.Math.Sqrt(trueVariance), result[1].GetMean(), 1e-2);
        }

        private Gaussian[] PlusProductHierarchy(
          int N1, int N2, double trueMeanMean, double trueMeanVariance, double trueSigmaMean, double trueSigmaVariance)
        {
            // Generate data from ground truth
            double[][] sample = new double[N1][];

            Gaussian trueMeanDist = Gaussian.FromMeanAndVariance(trueMeanMean, trueMeanVariance);
            Gaussian trueSigmaDist = Gaussian.FromMeanAndVariance(trueSigmaMean, trueSigmaVariance);

            for (int i = 0; i < N1; i++)
            {
                double mean = trueMeanDist.Sample();
                double sigma = trueSigmaDist.Sample();
                Gaussian g = Gaussian.FromMeanAndVariance(mean, sigma * sigma);
                sample[i] = Util.ArrayInit(N2, j => g.Sample());
            }

            // The model
            double broadVariance = 10e4;
            var meanmean = Variable.Random(Gaussian.FromMeanAndVariance(0, broadVariance)).Named("meanmean");
            var meansigma = Variable.Random(Gaussian.FromMeanAndVariance(1, broadVariance)).Named("meansigma");
            var sigmamean = Variable.Random(Gaussian.FromMeanAndVariance(1, broadVariance)).Named("sigmamean");
            var sigmasigma = Variable.Random(Gaussian.FromMeanAndVariance(1, broadVariance)).Named("sigmasigma");
            Range n1 = new Range(N1).Named("n1");
            var N2ofN1 = Variable.Array<int>(n1).Named("N2ofN1");
            Range n2 = new Range(N2ofN1[n1]).Named("n2");
            //var x = Variable.Array(Variable.Array<int>(n2), n1);
            var x = Variable.Array(Variable.Array<double>(n2), n1).Named("x");

            using (Variable.ForEach(n1))
            {
                var mean = meanmean + meansigma * Variable.GaussianFromMeanAndPrecision(0, 1);
                mean.Name = "mean";
                var sigma = sigmamean + sigmasigma * Variable.GaussianFromMeanAndPrecision(0, 1);
                sigma.Name = "sigma";
                Variable.ConstrainPositive(sigma);

                using (Variable.ForEach(n2))
                {
                    //var logLambda = mean + sigma * Variable.GaussianFromMeanAndPrecision(0, 1);
                    //var lambda = Variable.Exp(logLambda);
                    //x[n1][n2] = Variable.Poisson(lambda);
                    x[n1][n2] = mean + sigma * Variable.GaussianFromMeanAndPrecision(0, 1);
                }
            }

            N2ofN1.ObservedValue = sample.Select(arr => arr.Length).ToArray();
            x.ObservedValue = sample;

            InferenceEngine engine = new InferenceEngine();
            //sigmamean.ObservedValue = trueSigmaMean;
            //sigmasigma.ObservedValue = Math.Sqrt(trueSigmaVariance);
            n1.AddAttribute(new Sequential());
            n2.AddAttribute(new Sequential());
            //engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_LaplaceProp));
            engine.Compiler.FreeMemory = false;

            for (int iter = 1; iter < 50; iter++)
            {
                engine.NumberOfIterations = iter;
                //Console.WriteLine("{0} {1}", engine.Infer<Gaussian>(meansigma), engine.Infer<Gaussian>(meanmean));
                Console.WriteLine("{0} {1} {2} {3}", engine.Infer(meanmean), engine.Infer(meansigma), engine.Infer(sigmamean), engine.Infer(sigmasigma));
            }

            return new Gaussian[]
                {
                engine.Infer<Gaussian>(meanmean), 
                engine.Infer<Gaussian>(meansigma), 
                engine.Infer<Gaussian>(sigmamean), 
                    engine.Infer<Gaussian>(sigmasigma)
                };
        }

        [Fact]
        public void PlusProductHierarchyTest()
        {
            Rand.Restart(0);
            double trueMeanMean = 1.2;
            double trueMeanVariance = 2.3;
            double trueSigmaMean = 4.1;
            double trueSigmaVariance = 0.5;

            int N1 = 100;
            int N2 = 1000;
            var result = PlusProductHierarchy(N1, N2, trueMeanMean, trueMeanVariance, trueSigmaMean, trueSigmaVariance);

            Console.WriteLine("True mean mean = {0}, Inferred = {1}", trueMeanMean, result[0]);
            Console.WriteLine("True mean sigma = {0}, Inferred = {1}", System.Math.Sqrt(trueMeanVariance), result[1]);
            Console.WriteLine("True sigma mean = {0}, Inferred = {1}", trueSigmaMean, result[2]);
            Console.WriteLine("True sigma = {0}, Inferred = {1}", System.Math.Sqrt(trueSigmaVariance), result[3]);

            Assert.True(System.Math.Abs(trueMeanMean - result[0].GetMean()) <= 0.3);
            Assert.True(System.Math.Abs(System.Math.Sqrt(trueMeanVariance) - result[1].GetMean()) <= 0.2);
            Assert.True(System.Math.Abs(trueSigmaMean - result[2].GetMean()) <= 0.2);
            Assert.True(System.Math.Abs(System.Math.Sqrt(trueSigmaVariance) - result[3].GetMean()) <= 0.1);
        }

        [Fact]
        public void ExpOp_FindMaximumTest()
        {
            Gamma exp = Gamma.FromShapeAndRate(2, 0.80037777777777785);
            Gaussian d = Gaussian.FromNatural(-0.42046067000973919, 0.00047516684017256684);
            double x = ExpOp_Slow.FindMaximum(exp, d);
            Assert.True(!double.IsNaN(x));
            Gaussian dMsg = ExpOp_Slow.DAverageConditional(exp, d);
            Assert.True(!double.IsNaN(dMsg.MeanTimesPrecision));
        }

        internal void ExpTest2()
        {
            double mx = 0, vx = 1000;
            double a = 1.5, b = 0.5;
            Variable<double> x = Variable.GaussianFromMeanAndVariance(mx, vx).Named("x");
            Variable<int> count = Variable.Observed(0).Named("count");
            Range item = new Range(count).Named("i");
            using (Variable.ForEach(item))
            {
                Variable<double> y = Variable.Exp(x).Named("y");
                Variable.ConstrainEqualRandom(y, Gamma.FromShapeAndRate(a, b));
            }

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            ExpOp.QuadratureNodeCount = 50;
            Console.WriteLine("ExpOp_Slow");
            engine.Compiler.GivePriorityTo(typeof(ExpOp_Slow));
            engine.NumberOfIterations = 10;
            int maxCount = 100;
            Gaussian[] exact = new Gaussian[maxCount];
            for (int n = 1; n <= maxCount; n++)
            {
                count.ObservedValue = n;
                exact[n - 1] = engine.Infer<Gaussian>(x);
            }

            Dictionary<string, object> dict = new Dictionary<string, object>();
            Type[] operators = new Type[] { typeof(ExpOp_LaplaceProp), typeof(ExpOp_Laplace), typeof(ExpOp_Laplace3), typeof(ExpOp), typeof(ExpOp_BFGS) };
            foreach (Type type in operators)
            {
                Console.WriteLine("{0}", type.Name);
                if (type.Name == "ExpOp_BFGS")
                {
                    engine.Algorithm = new VariationalMessagePassing();
                    x.InitialiseTo(Gaussian.PointMass(0.0));
                }
                engine.Compiler.GivePriorityTo(type);
                engine.NumberOfIterations = 100;
                double[] error = new double[maxCount];
                for (int n = 1; n <= maxCount; n++)
                {
                    count.ObservedValue = n;
                    Gaussian xExpected = exact[n - 1];
                    Gaussian xActual = new Gaussian();
                    try
                    {
                        xActual = engine.Infer<Gaussian>(x);
                    }
                    catch
                    {
                    }
                    Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                    error[n - 1] = xExpected.MaxDiff(xActual);
                }
                dict[type.Name] = error;
            }
            if (true)
            {
                engine.Algorithm = new VariationalMessagePassing();
                x.InitialiseTo(Gaussian.PointMass(0.0));
                Console.WriteLine("VMP");
                double[] error = new double[maxCount];
                for (int n = 1; n <= maxCount; n++)
                {
                    count.ObservedValue = n;
                    Gaussian xExpected = exact[n - 1];
                    Gaussian xActual = engine.Infer<Gaussian>(x);
                    Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                    error[n - 1] = xExpected.MaxDiff(xActual);
                }
                dict["VMP"] = error;
            }
            //TODO: change path for cross platform using
            using (var writer = new MatlabWriter(@"..\..\..\Tests\ExpTest.mat"))
            {
                writer.Write("maxdiff", dict);
            }
        }

        // Test different operators for Math.Exp
        internal void ExpTest()
        {
            double mx = 1, vx = 2;
            double a = 3, b = 4;
            // breaks with:
            // a = 0.5; b = 0.2;
            if (false)
            {
                // causes Laplace to give negative variance
                a = 1.01;
                mx = -10;
                vx = 200;
                mx = 1;
                vx = 10;
            }
            //vx = 1e-1;
            //vx = 1e-2;
            //b = 1e-5;
            //b = 1;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);
            Gaussian xPrior = Gaussian.FromMeanAndVariance(mx, vx);
            Variable<double> x = Variable.Random(xPrior).Named("x");
            Variable<double> y = Variable.Exp(x).Named("y");
            Gamma yPrior = Gamma.FromShapeAndRate(a, b);
            Variable.ConstrainEqualRandom(y, yPrior);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            ExpOp.QuadratureNodeCount = 50;
            Gaussian xExpected = engine.Infer<Gaussian>(x);
            Gamma yExpected = engine.Infer<Gamma>(y);
            double evExpected = engine.Infer<Bernoulli>(evidence).LogOdds;

            if (true)
            {
                ExpApprox2(yPrior, xPrior);
            }

            if (false)
            {
                ExpApprox(mx, vx, a, b, xExpected);
            }
            if (false)
            {
                double xhat = ExpOp_Laplace.X(yPrior, xPrior);
                double logZ = ExpOp_Laplace.LogAverageFactor(yPrior, xPrior, xhat);
                double xhat2 = ExpOp_Laplace.X(Gamma.FromShapeAndRate(a + 1, b), xPrior);
                double logZeX = ExpOp_Laplace.LogAverageFactor(Gamma.FromShapeAndRate(a + 1, b), xPrior, xhat2) + System.Math.Log(a / b);
                double expxApprox = System.Math.Exp(logZeX - logZ);
                Console.WriteLine("expxApprox = {0}", expxApprox);
            }

            Type[] operators = new Type[] { typeof(ExpOp_LaplaceProp), typeof(ExpOp_Laplace), typeof(ExpOp_Laplace3), typeof(ExpOp3), typeof(ExpOp_BFGS) };
            foreach (Type type in operators)
            {
                Console.WriteLine("{0}", type.Name);
                engine.Compiler.GivePriorityTo(type);
                if (type.Name == typeof(ExpOp_BFGS).Name)
                    engine.Algorithm = new VariationalMessagePassing();
                Gaussian xActual = engine.Infer<Gaussian>(x);
                Gamma yActual = engine.Infer<Gamma>(y);
                double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            }
        }

        private void ExpApprox(double mx, double vx, double a, double b, Gaussian xExpected)
        {
            double m, v;
            xExpected.GetMeanAndVariance(out m, out v);
            double pseudoPrecision = (a - 1) - (m - mx - 1) / vx;
            double expxTrue = (pseudoPrecision - 1 / vx) / b;
            double[] xs = EpTests.linspace(-10, 10, 10000);
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mva3 = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mva4 = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mvaExp = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mvaxExp = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mvaExp3 = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mvax2Exp = new MeanVarianceAccumulator();
            MeanVarianceAccumulator mvaxExp2 = new MeanVarianceAccumulator();
            for (int i = 0; i < xs.Length; i++)
            {
                double logp = Gaussian.GetLogProb(xs[i], mx, vx) + (a - 1) * xs[i] - b * System.Math.Exp(xs[i]);
                double p = System.Math.Exp(logp);
                mva.Add(xs[i], p);
                double diff = xs[i] - m;
                mva3.Add(diff * diff * diff, p);
                mva4.Add(diff * diff * diff * diff, p);
                mvaExp.Add(System.Math.Exp(xs[i]), p);
                double expdiff = System.Math.Exp(xs[i]) - expxTrue;
                mvaxExp.Add(diff * expdiff, p);
                mvaExp3.Add(expdiff * expdiff * expdiff, p);
                mvax2Exp.Add(diff * diff * expdiff, p);
                mvaxExp2.Add(diff * expdiff * expdiff, p);
            }
            double m3 = mva3.Mean;
            double m4 = mva4.Mean;
            double expxMean = mvaExp.Mean;
            double expxVariance = mvaExp.Variance;
            double m11 = mvaxExp.Mean;
            double expxM3 = mvaExp3.Mean;
            double m21 = mvax2Exp.Mean;
            double m12 = mvaxExp2.Mean;
            Console.WriteLine("E[e^x] = {0} {1}", expxTrue, expxMean);
            Console.WriteLine("m = {0}, v = {1}, m4[x] = {2}", mva.Mean, mva.Variance, m4);
            Console.WriteLine("v = {0} {1}", v, vx * (1 - b * m11));
            double gr = -expxMean * 0.5 / (pseudoPrecision * pseudoPrecision);
            double grr = expxMean * 0.25 / (pseudoPrecision * pseudoPrecision * pseudoPrecision * pseudoPrecision) + expxMean / (pseudoPrecision * pseudoPrecision * pseudoPrecision);
            double denom = 1 + b * vx * expxMean - b * gr;
            Console.WriteLine("m11 = {0} {1} {2}", m11, vx * (expxMean - b * expxVariance), vx * expxMean / denom);
            double m12_approx = -vx * expxVariance / denom - vx * expxMean * (vx * expxMean - gr - b * vx * expxVariance + b * (vx * gr - grr) * m11 / vx) / (denom * denom);
            m12_approx = vx * expxMean * (-2 * vx * expxMean + 2 * gr + b * vx * expxVariance - b * (vx * gr - grr) * m11 / vx) / (denom * denom);
            m12_approx = vx * expxMean * ((-2 * vx * expxMean + 2 * gr) * denom + b * vx * vx * expxMean * expxMean - 2 * b * vx * expxMean * gr + b * expxMean * grr) / (denom * denom * denom);
            //m12_approx = b*vx * expxMean*expxMean * (vx*vx*expxMean - 2*vx*gr + grr) / (denom * denom * denom);
            Console.WriteLine("m12 = {0} {1} {2}", m12, vx * (2 * expxVariance - b * expxM3), m12_approx);
            double m21_approx = vx * m11 / denom - vx * expxMean * (b * vx * m11 - b * vx * gr + b * b * (vx * gr - grr) * m11) / (denom * denom);
            m21_approx = vx * vx * expxMean * (1 - b * b * gr * gr + b * b * expxMean * grr) / (denom * denom * denom);
            //m21_approx = 
            Console.WriteLine("m21 = {0} {1} {2}", m21, vx * (m11 - b * m12), m21_approx);
            double m3_approx = vx * vx * vx * (-b * expxMean + 3 * b * b * expxVariance - b * b * b * expxM3);
            Console.WriteLine("m3[x] = {0} {1} {2}", m3, -vx * b * m21, m3_approx);
            Console.WriteLine("m3[e^x] = {0} {1}", expxM3, 0);
            double m1 = mx + vx * (a - 1 - b * System.Math.Exp(m + v / 2));
            double pv1 = vx / (1 + vx * b * System.Math.Exp(m + v / 2));
            ExpIteration(mx, vx, a, b, ref m1, ref pv1);
            double m2 = m1, pv2 = pv1;
            ExpIteration(mx, vx, a + 1, b, ref m2, ref pv2);
            Console.WriteLine("pseudo-variance = {0} {1}", 1 / pseudoPrecision, pv1);
            double v1 = vx + vx * vx * (-b * System.Math.Exp(m1 + pv1 / 2)
                        + b * b * System.Math.Exp(2 * m1 + pv1) * (System.Math.Exp(pv1) - 1));
            double e2e = (1 / pv2 - 1 / vx) / b;
            double expx = (1 / pv1 - 1 / vx) / b;
            double expxApproxVariance = System.Math.Exp(m + v / 2);
            double c = v / 2 + m3 / 6 + m4 / 24;
            double expxTaylor = System.Math.Exp(m) * (1 + c);
            // delta + 0.5*delta^2 = 0.5*v + m3/6 + m4/24
            double delta = -1 + System.Math.Sqrt(1 + 2 * c);
            double expxApprox = System.Math.Exp(m + delta);
            expxApprox = System.Math.Exp(mx + vx / 2) * (1 + vx * (a - 1 - b * expx));
            Console.WriteLine("expx = {0} {1} {2} {3} {4}", expxTrue, expx, expxApproxVariance, expxTaylor, expxApprox);
            double exp2x = e2e * expx;
            v1 = vx + vx * vx * (-b * expx + b * b * (exp2x - expx * expx));
            Console.WriteLine("m1 = {0}, v1 = {1}", m1, v1);
        }

        private void ExpIteration(double mx, double vx, double a, double b, ref double m, ref double pv)
        {
            for (int iter = 0; iter < 50; iter++)
            {
                m = -pv / 2 + System.Math.Log((mx + vx * (a - 1) - m) / (vx * b));
                //m = mx + vx * (a - 1 - 1 / pv) + 1;
                pv = vx / (1 + vx * b * System.Math.Exp(m + pv / 2));
                //Console.WriteLine("{0}: m={1}, pv={2}", iter, m, pv);
            }
        }

        internal static void ExpApprox_NonconjugateVmp(Gamma yPrior, Gaussian xPrior, out double logZ, out Gaussian q)
        {
            double a = yPrior.Shape;
            double b = yPrior.Rate;
            Gaussian g = Gaussian.Uniform();
            q = xPrior * g;
            logZ = double.NaN;
            Matrix qMoments = new Matrix(2, 2);
            Vector fMoments = Vector.Zero(2);
            for (int iter = 0; iter < 20; iter++)
            {
                double mq, vq;
                q.GetMeanAndVariance(out mq, out vq);
                double mq2 = mq * mq;
                double Ex2 = mq2 + vq;
                double Ex3 = mq2 * mq + 3 * mq * vq;
                double Ex4 = mq2 * mq2 + 6 * mq2 * vq + 3 * vq * vq;
                double Eexpx = System.Math.Exp(mq + vq / 2);
                double Exexpx = (mq + vq) * Eexpx;
                double Ex2expx = ((mq + vq) * (mq + vq) + vq) * Eexpx;
                double Ef = (a - 1) * mq - b * Eexpx;
                double Exf = (a - 1) * Ex2 - b * Exexpx;
                double Ex2f = (a - 1) * Ex3 - b * Ex2expx;
                logZ = xPrior.GetLogAverageOf(g) + Ef - q.GetAverageLog(g) - yPrior.GetLogNormalizer();
                qMoments[0, 0] = vq;
                qMoments[0, 1] = Ex3 - Ex2 * mq;
                qMoments[1, 0] = qMoments[0, 1];
                qMoments[1, 1] = Ex4 - Ex2 * Ex2;
                fMoments[0] = Exf - Ef * mq;
                fMoments[1] = Ex2f - Ef * Ex2;
                (new LuDecomposition(qMoments)).Solve(fMoments);
                g.MeanTimesPrecision = fMoments[0];
                g.Precision = -2 * fMoments[1];
                q = xPrior * g;
                Console.WriteLine("{0} {1}", iter, q);
            }
        }

        private static void ExpApprox2(Gamma yPrior, Gaussian xPrior)
        {
            double a = yPrior.Shape;
            double b = yPrior.Rate;
            double logs = 0;
            Gaussian g = Gaussian.Uniform();
            Gaussian q = xPrior * g;
            double logZ = double.NaN;
            Matrix qMoments = new Matrix(3, 3);
            Vector fMoments = Vector.Zero(3);
            for (int iter = 0; iter < 20; iter++)
            {
                double mq, vq;
                q.GetMeanAndVariance(out mq, out vq);
                double mq2 = mq * mq;
                double Ex2 = mq2 + vq;
                double Ex3 = mq2 * mq + 3 * mq * vq;
                double Ex4 = mq2 * mq2 + 6 * mq2 * vq + 3 * vq * vq;
                double Eexpx = System.Math.Exp(mq + vq / 2);
                double Exexpx = (mq + vq) * Eexpx;
                double Ex2expx = ((mq + vq) * (mq + vq) + vq) * Eexpx;
                double Ef = (a - 1) * mq - b * Eexpx;
                double Exf = (a - 1) * Ex2 - b * Exexpx;
                double Ex2f = (a - 1) * Ex3 - b * Ex2expx;
                qMoments[0, 0] = 1;
                qMoments[0, 1] = mq;
                qMoments[0, 2] = Ex2;
                qMoments[1, 0] = qMoments[0, 1];
                qMoments[1, 1] = qMoments[0, 2];
                qMoments[1, 2] = Ex3;
                qMoments[2, 0] = qMoments[0, 2];
                qMoments[2, 1] = qMoments[1, 2];
                qMoments[2, 2] = Ex4;
                fMoments[0] = Ef;
                fMoments[1] = Exf;
                fMoments[2] = Ex2f;
                (new LuDecomposition(qMoments)).Solve(fMoments);
                g.MeanTimesPrecision = fMoments[1];
                g.Precision = -2 * fMoments[2];
                logs = fMoments[0] + MMath.LnSqrt2PI - 0.5 * System.Math.Log(g.Precision) + 0.5 * g.MeanTimesPrecision * g.MeanTimesPrecision / g.Precision - yPrior.GetLogNormalizer();
                q = xPrior * g;
                logZ = xPrior.GetLogAverageOf(g) + logs;
                Console.WriteLine("{0} {1} {2}", iter, q, logZ);
            }
        }

        internal static void ExpApprox3(Gamma yPrior, Gaussian xPrior)
        {
            double a = yPrior.Shape;
            double b = yPrior.Rate;
            double logs = 0;
            Gaussian ft = Gaussian.Uniform();
            Gaussian q = xPrior * ft;
            double logZ = double.NaN;
            Matrix qMoments = new Matrix(3, 3);
            Vector theta = Vector.Zero(3);
            for (int iter = 0; iter < 20; iter++)
            {
                double mq, vq;
                q.GetMeanAndVariance(out mq, out vq);
                double mq2 = mq * mq;
                double Ex2 = mq2 + vq;
                double Ex3 = mq2 * mq + 3 * mq * vq;
                double Ex4 = mq2 * mq2 + 6 * mq2 * vq + 3 * vq * vq;
                double Ex5 = mq2 * mq2 * mq + 10 * mq2 * mq * vq + 15 * mq * vq*vq;
                double Ex6 = mq2 * mq2 * mq2 + 15 * mq2 * mq2 * vq + 45 * mq2 * vq * vq + 15 * vq * vq * vq;
                double Eexpx = System.Math.Exp(mq + vq / 2);
                double Exexpx = (mq + vq) * Eexpx;
                double Ex2expx = ((mq + vq) * (mq + vq) + vq) * Eexpx;
                double Ex3expx = ((mq + vq) * (mq + vq) * (mq+vq) + 3*vq * (mq + vq))*Eexpx;
                double Ex4expx = ((mq + vq) * (mq + vq) * (mq + vq)*(mq+vq) + 6 * vq * (mq + vq)*(mq+vq) + 3*vq*vq) * Eexpx;
                double Ef = (a - 1) * mq - b * Eexpx;
                double Exf = (a - 1) * Ex2 - b * Exexpx;
                double Ex2f = (a - 1) * Ex3 - b * Ex2expx;
                double Ex3f = (a - 1) * Ex4 - b * Ex3expx;
                double Ex4f = (a - 1) * Ex5 - b * Ex4expx;
                double Eexp2x = System.Math.Exp(2 * mq + 2 * vq);
                double Exexp2x = (mq + 2*vq) * Eexp2x;
                double Ex2exp2x = mq * Exexp2x + vq * (Eexp2x + 2*Exexp2x);
                double Ef2 = (a - 1) * (a - 1) * Ex2 - 2 * (a - 1) * b * Exexpx + b * b * Eexp2x;
                double Exf2 = (a - 1) * (a - 1) * Ex3 - 2 * (a - 1) * b * Ex2expx + b * b * Exexp2x;
                double Ex2f2 = (a - 1) * (a - 1) * Ex4 - 2 * (a - 1) * b * Ex3expx + b * b * Ex2exp2x;
                double Efft = Ef * theta[0] + Exf * theta[1] + Ex2f * theta[2];
                double Exfft = Exf * theta[0] + Ex2f * theta[1] + Ex3f * theta[2];
                double Ex2fft = Ex2f * theta[0] + Ex3f * theta[1] + Ex4f * theta[2];
                double Eft2 = theta[0] * theta[0] + 2 * mq * theta[0] * theta[1] 
                    + Ex2 * (theta[1] * theta[1] + 2 * theta[0] * theta[2])
                    + 2 * Ex3 * theta[1] * theta[2] + Ex4 * theta[2] * theta[2];
                double Exft2 = mq* theta[0] * theta[0] + 2 * Ex2 * theta[0] * theta[1]
                    + Ex3 * (theta[1] * theta[1] + 2 * theta[0] * theta[2])
                    + 2 * Ex4 * theta[1] * theta[2] + Ex5 * theta[2] * theta[2];
                double Ex2ft2 = Ex2*theta[0] * theta[0] + 2 * Ex3 * theta[0] * theta[1]
                    + Ex4 * (theta[1] * theta[1] + 2 * theta[0] * theta[2])
                    + 2 * Ex5 * theta[1] * theta[2] + Ex6 * theta[2] * theta[2];
                double Esqdiff = Ef2 - 2 * Efft + Eft2;
                double Exsqdiff = Exf2 - 2 * Exfft + Exft2;
                double Ex2sqdiff = Ex2f2 - 2 * Ex2fft + Ex2ft2;
                qMoments[0, 0] = 1;
                qMoments[0, 1] = mq;
                qMoments[0, 2] = Ex2;
                qMoments[1, 0] = qMoments[0, 1];
                qMoments[1, 1] = qMoments[0, 2];
                qMoments[1, 2] = Ex3;
                qMoments[2, 0] = qMoments[0, 2];
                qMoments[2, 1] = qMoments[1, 2];
                qMoments[2, 2] = Ex4;
                theta[0] = Ef - Esqdiff;
                theta[1] = Exf - Exsqdiff;
                theta[2] = Ex2f - Ex2sqdiff;
                (new LuDecomposition(qMoments)).Solve(theta);
                theta.Scale(2);
                ft.MeanTimesPrecision = theta[1];
                ft.Precision = -2 * theta[2];
                logs = theta[0] + MMath.LnSqrt2PI - 0.5 * System.Math.Log(ft.Precision) + 0.5 * ft.MeanTimesPrecision * ft.MeanTimesPrecision / ft.Precision - yPrior.GetLogNormalizer();
                q = xPrior * ft;
                logZ = xPrior.GetLogAverageOf(ft) + logs;
                Console.WriteLine("{0} {1} {2}", iter, q, logZ);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/doc/*'/>
        [FactorMethod(typeof(System.Math), "Exp", typeof(double))]
        [Buffers("c")]
        [Quality(QualityBand.Experimental)]
        public static class ExpOp_Laplace3
        {
            public static bool modified = true;

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/message_doc[@name="CInit()"]/*'/>
            [Skip]
            public static double CInit()
            {
                return 0.0;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/message_doc[@name="C(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static double C([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
            {
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogh = b * ehat / (d.Precision + c + b * ehat);
                double ddlogh = dlogh * (1 - dlogh);
                return 0.5 * ddlogh;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
            {
                if (exp.IsPointMass)
                    return ExpOp.DAverageConditional(exp.Point);
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogf = (a - 1) - b * ehat;
                double ddlogf = -b * ehat;
                if (modified)
                {
                    double h = d.Precision + c - ddlogf;
                    double dlogh = b * ehat / h;
                    double ddlogh = dlogh * (1 - dlogh);
                    dlogf -= 0.5 * dlogh;
                    ddlogf -= 0.5 * ddlogh;
                }
                double r = System.Math.Max(0, -ddlogf);
                return Gaussian.FromNatural(r * dhat + dlogf, r);
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d, double c)
            {
                if (d.IsPointMass)
                    return Gamma.PointMass(System.Math.Exp(d.Point));
                if (exp.IsPointMass)
                    return Gamma.Uniform();
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogf_diff = -ehat;
                double ddlogf_diff = 0;
                double ddlogfx = -ehat;
                if (modified)
                {
                    double dlogh = ehat / (d.Precision + c + b * ehat);
                    double ddlogh = -dlogh * dlogh;
                    double ddloghx = dlogh * (1 - b * dlogh);
                    dlogf_diff -= 0.5 * dlogh;
                    ddlogf_diff -= 0.5 * ddlogh;
                    ddlogfx -= 0.5 * ddloghx;
                }
                double dlogz_diff = dlogf_diff;
                double dx = ddlogfx / (d.Precision - ddlogf_diff + a / (b * b));
                double ddlogz_diff = ddlogf_diff + ddlogfx * dx;
                double m = -dlogz_diff;
                double v = ddlogz_diff;
                Gamma result = Gamma.FromMeanAndVariance(m, v);
                result.SetToRatio(result, exp, true);
                return result;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/message_doc[@name="LogAverageFactor(Gamma, Gaussian, Gaussian)"]/*'/>
            public static double LogAverageFactor([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
            {
                Gaussian dPost = d * to_d;
                double x = dPost.GetMean();
                double expx = System.Math.Exp(x);
                double a = exp.Shape;
                double b = exp.Rate;
                double v = dPost.GetVariance();
                return exp.GetLogProb(expx) + d.GetLogProb(x) + MMath.LnSqrt2PI + 0.5 * System.Math.Log(v);
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3"]/message_doc[@name="LogEvidenceRatio(Gamma, Gaussian, Gaussian, Gamma)"]/*'/>
            public static double LogEvidenceRatio([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, Gamma to_exp)
            {
                return LogAverageFactor(exp, d, to_d) - to_exp.GetLogAverageOf(exp);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3_Slow"]/doc/*'/>
        [FactorMethod(typeof(System.Math), "Exp", typeof(double))]
        [Buffers("logh")]
        [Quality(QualityBand.Experimental)]
        internal static class ExpOp_Laplace3_Slow
        {
            public static bool modified = true;
            public const int n = 100;
            public const double xmin = -10;
            public const double xmax = 5;
            private const double inc = (xmax - xmin) / (n - 1);
            private static Matrix dd = null;

            private static DenseVector Aux([SkipIfUniform] Gamma exp, [Proper] Gaussian d, DenseVector logh, DenseVector aux)
            {
                double b = exp.Rate;
                double dPrec = d.Precision;
                DenseVector target = DenseVector.Zero(n);
                for (int i = 0; i < n; i++)
                {
                    double x = xmin + i * inc;
                    target[i] = dPrec + b * System.Math.Exp(x) + 0.5 * aux[i];
                }
                DenseVector numer = DenseVector.Zero(n);
                numer.SetToProduct(dd, logh);
                for (int i = 0; i < n; i++)
                {
                    double c = 0.5 / target[i];
                    if (target[i] < 0)
                        throw new Exception();
                    double g = (logh[i] - System.Math.Log(target[i]) + c * aux[i]) * c;
                    aux[i] = (numer[i] + g) / (1 + c * c);
                }
                return aux;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3_Slow"]/message_doc[@name="LoghInit()"]/*'/>
            public static double[] LoghInit()
            {
                return new double[n];
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3_Slow"]/message_doc[@name="Logh(Gamma, Gaussian, double[])"]/*'/>
            public static double[] Logh([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double[] result)
            {
                if (dd == null)
                    dd = Deriv2Matrix(n, 1 / (inc * inc));
                double b = exp.Rate;
                double dPrec = d.Precision;
                DenseVector aux = DenseVector.Zero(n);
                PositiveDefiniteMatrix denom = new PositiveDefiniteMatrix(n, n);
                denom.SetToOuterTranspose(dd);
                for (int i = 0; i < n; i++)
                {
                    denom[i, i] += 1;
                }
                DenseVector logh = DenseVector.Zero(n);
                for (int i = 0; i < n; i++)
                {
                    double x = xmin + i * inc;
                    logh[i] = System.Math.Log(dPrec + b * System.Math.Exp(x));
                }
                for (int iter = 0; iter < 100; iter++)
                {
                    if (iter == 0)
                        aux.SetToProduct(dd, logh);
                    aux = Aux(exp, d, logh, aux);
                    logh.SetToProduct(aux, dd);
                    for (int i = 0; i < n; i++)
                    {
                        double x = xmin + i * inc;
                        double target = dPrec + b * System.Math.Exp(x) + 0.5 * aux[i];
                        if (target < 0)
                            throw new Exception();
                        logh[i] += System.Math.Log(target);
                    }
                    logh.PredivideBy(denom);
                }
                for (int i = 0; i < n; i++)
                {
                    result[i] = logh[i];
                }
                return result;
            }

            public static Matrix Deriv2Matrix(int n, double scale)
            {
                Matrix dd = new Matrix(n, n);
                for (int i = 1; i < (n - 1); i++)
                {
                    dd[i, i] = -2 * scale;
                    dd[i, i + 1] = 1 * scale;
                    dd[i, i - 1] = 1 * scale;
                }
                for (int j = 0; j < n; j++)
                {
                    dd[0, j] = 2 * dd[1, j] - dd[2, j];
                    dd[n - 1, j] = 2 * dd[n - 2, j] - dd[n - 3, j];
                }
                return dd;
            }

            public static double Interpolate(double[] y, double x)
            {
                double ifrac = (x - xmin) / inc;
                int i = (int)System.Math.Floor(ifrac);
                // interpolate along the line joining (i*inc, y[i]) and ((i+1)*inc, y[i+1])
                //double slope = (y[i+1]-y[i])/inc;
                //return y[i] + slope*(ifrac - i)*inc;
                double w = (ifrac - i);
                return y[i + 1] * w + y[i] * (1 - w);
            }

            public static void GetDerivatives(double[] y, double x, out double deriv, out double deriv2)
            {
                double y0 = Interpolate(y, x);
                double yPlus = Interpolate(y, x + inc);
                double yMinus = Interpolate(y, x - inc);
                deriv = (yPlus - yMinus) / (2 * inc);
                deriv2 = (yPlus + yMinus - 2 * y0) / (inc * inc);
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3_Slow"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, Gaussian, double[])"]/*'/>
            public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double[] logh)
            {
                if (exp.IsPointMass)
                    return ExpOp.DAverageConditional(exp.Point);
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogf = (a - 1) - b * ehat;
                double ddlogf = -b * ehat;
                if (modified)
                {
                    double dlogh, ddlogh;
                    GetDerivatives(logh, dhat, out dlogh, out ddlogh);
                    dlogf -= 0.5 * dlogh;
                    ddlogf -= 0.5 * ddlogh;
                }
                double r = System.Math.Max(0, -ddlogf);
                return Gaussian.FromNatural(r * dhat + dlogf, r);
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace3_Slow"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, Gaussian)"]/*'/>
            public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d)
            {
                if (d.IsPointMass)
                    return Gamma.PointMass(System.Math.Exp(d.Point));
                if (exp.IsPointMass)
                    return Gamma.Uniform();
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogf_diff = -ehat;
                double ddlogf_diff = 0;
                double ddlogfx = -ehat;
                if (modified)
                {
                    double c = 0;
                    double dlogh = ehat / (d.Precision + c + b * ehat);
                    double ddlogh = -dlogh * dlogh;
                    double ddloghx = dlogh * (1 - b * dlogh);
                    dlogf_diff -= 0.5 * dlogh;
                    ddlogf_diff -= 0.5 * ddlogh;
                    ddlogfx -= 0.5 * ddloghx;
                }
                double dlogz_diff = dlogf_diff;
                double dx = ddlogfx / (d.Precision - ddlogf_diff + a / (b * b));
                double ddlogz_diff = ddlogf_diff + ddlogfx * dx;
                double m = -dlogz_diff;
                double v = ddlogz_diff;
                Gamma result = Gamma.FromMeanAndVariance(m, v);
                result.SetToRatio(result, exp, true);
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace2_Slow"]/doc/*'/>
        [FactorMethod(typeof(System.Math), "Exp", typeof(double))]
        [Buffers("c")]
        [Quality(QualityBand.Experimental)]
        internal static class ExpOp_Laplace2_Slow
        {
            public static bool modified = true;

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace2_Slow"]/message_doc[@name="CInit()"]/*'/>
            [Skip]
            public static double CInit()
            {
                return 0.0;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace2_Slow"]/message_doc[@name="C(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static double C([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
            {
                //return 0.0;
                Gaussian dPost = d * to_d;
                double h = dPost.Precision;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double eh = System.Math.Exp(dhat + 0.5 / h);
                double gx = (a - 1) - b * eh;
                double gh = 0.5 / (h * h) * b * eh;
                double gxx = -b * eh;
                double gxh = gh;
                double ghh = -gh * (2 / h + 0.5 / (h * h));
                double gxxx = gxx;
                double gxxxx = gxx;
                h = d.Precision + c - gxx;
                double dh = -gxxx;
                double ddh = -gxxxx;
                double dlogh = dh / h;
                double ddlogh = ddh / h - dlogh * dlogh;
                double beta2 = 2 * gxh * dh + ghh * (dh * dh) + gh * ddh - 0.5 * ddlogh;
                return -beta2;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace2_Slow"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
            {
                if (exp.IsPointMass)
                    return ExpOp.DAverageConditional(exp.Point);
                //c = 0;
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double h = dPost.Precision;
                for (int i = 0; i < 10; i++)
                {
                    h = d.Precision + c + b * System.Math.Exp(dhat + 0.5 / h);
                }
                double eh = System.Math.Exp(dhat + 0.5 / h);
                double gx = (a - 1) - b * eh;
                double gh = 0.5 / (h * h) * b * eh;
                double gxx = -b * eh;
                double gxh = gh;
                double ghh = -gh * (2 / h + 0.5 / (h * h));
                double gxxx = gxx;
                double gxxxx = gxx;
                //h = d.Precision + c - gxx;
                Console.WriteLine("h = {0}, {1}", h, dPost.Precision);
                double dh = -gxxx;
                double ddh = -gxxxx;
                double dlogh = dh / h;
                double ddlogh = ddh / h - dlogh * dlogh;
                double dlogz = gx + gh * dh - 0.5 * dlogh;
                double beta = gxx + 2 * gxh * dh + ghh * (dh * dh) + gh * ddh - 0.5 * ddlogh;
                double r = System.Math.Max(0, -beta);
                return Gaussian.FromNatural(r * dhat + dlogz, r);
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace2_Slow"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d, double c)
            {
                if (d.IsPointMass)
                    return Gamma.PointMass(System.Math.Exp(d.Point));
                if (exp.IsPointMass)
                    return Gamma.Uniform();
                Gaussian dPost = d * to_d;
                double dhat = dPost.GetMean();
                double ehat = System.Math.Exp(dhat);
                double a = exp.Shape;
                double b = exp.Rate;
                double h = dPost.Precision;
                double eh = System.Math.Exp(dhat + 0.5 / h);
                double gx = (a - 1) - b * eh;
                double gh = 0.5 / (h * h) * b * eh;
                double gxx = -b * eh;
                double gxh = gh;
                double ghh = -gh * (2 / h + 0.5 / (h * h));
                double gxxx = gxx;
                double gxxxx = gxx;
                //h = d.Precision + c - gxx;
                Console.WriteLine("h = {0}, {1}", h, dPost.Precision);
                double dh = -gxxx;
                double ddh = -gxxxx;
                double dlogh = dh / h;
                double ddlogh = ddh / h - dlogh * dlogh;
                double gb = a / b - eh;
                double gb_diff = -eh;
                double gxb = -eh;
                double ghb = 0.5 / (h * h) * eh;
                double gbb = -a / (b * b);
                double gxxb = gxb;
                double gxxbb = 0;
                double hb = -gxxb;
                double hbb = -gxxbb;
                double dhb = hb;
                double dloghb = hb / h;
                double ddloghbb = hbb / h - dloghb * dloghb;
                double ddloghb = dhb / h - dloghb * dlogh;
                double dlogz_diff = gb_diff + gh * hb - 0.5 * dloghb;
                double alpha = gxb + ghb * dh + gxh * hb + ghh * dh * hb + gh * dhb - 0.5 * ddloghb;
                //double ddlogz = gbb + 2*ghb*hb - 0.5*ddloghbb + alpha*alpha/h;
                double ddlogz_diff = 2 * ghb * hb - 0.5 * ddloghbb + alpha * alpha / h;
                //double dlogz_diff = dlogz - a/b;
                //double ddlogz_diff = ddlogz + a/(b*b);
                double m = -dlogz_diff;
                double v = ddlogz_diff;
                Gamma result = Gamma.FromMeanAndVariance(m, v);
                result.SetToRatio(result, exp, true);
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp3"]/doc/*'/>
        [FactorMethod(typeof(System.Math), "Exp", typeof(double))]
        [Buffers("c")]
        [Quality(QualityBand.Experimental)]
        public static class ExpOp3
        {
            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp3"]/message_doc[@name="CInit()"]/*'/>
            [Skip]
            public static double CInit()
            {
                return 0.0;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp3"]/message_doc[@name="C(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static double C([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
            {
                Gaussian dPost = d * to_d;
                double m, v;
                dPost.GetMeanAndVariance(out m, out v);
                double a = exp.Shape;
                double b = exp.Rate;
                return -0.5 * b * b * System.Math.Exp(2 * m + v) / (1 / (v * v) + 0.5 * b * System.Math.Exp(m + v / 2));
                //return -0.5*b*b*Math.Exp(2*m+v)*v*v;
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp3"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, Gaussian, double)"]/*'/>
            public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
            {
                if (exp.IsPointMass)
                    return ExpOp.DAverageConditional(exp.Point);
                Gaussian dPost = d * to_d;
                double m, v;
                dPost.GetMeanAndVariance(out m, out v);
                double ee = System.Math.Exp(m + v / 2);
                //ee = Math.Exp(m)*(1 + v/2);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogz = (a - 1) - b * ee;
                double t = -b * ee - c;
                //double ddlogz = b*b*Math.Exp(2*m+v)*MMath.ExpMinus1(v) - b*ee;
                double ddlogz = t / (1 - t / d.Precision);
                //double t = d.Precision*ddlogz/(d.Precision + ddlogz);
                //Console.WriteLine("dPost = {0}, t = {1}, postvar = {2}", dPost, t, (1 + ddlogz/d.Precision)/d.Precision);
                double r = System.Math.Max(0, -t);
                return Gaussian.FromNatural(r * m + dlogz, r);
            }

            /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp3"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, Gaussian)"]/*'/>
            public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d)
            {
                if (d.IsPointMass)
                    return Gamma.PointMass(System.Math.Exp(d.Point));
                if (exp.IsPointMass)
                    return Gamma.Uniform();
                Gaussian dPost = d * to_d;
                double m, v;
                dPost.GetMeanAndVariance(out m, out v);
                double ee = System.Math.Exp(m + v / 2);
                double a = exp.Shape;
                double b = exp.Rate;
                double dlogz_diff = -ee;
                double ddlogz_diff = System.Math.Exp(2 * m + v) * MMath.ExpMinus1(v);
                double me = -dlogz_diff;
                double ve = ddlogz_diff;
                Gamma result = Gamma.FromMeanAndVariance(me, ve);
                result.SetToRatio(result, exp, true);
                return result;
            }
        }

    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
