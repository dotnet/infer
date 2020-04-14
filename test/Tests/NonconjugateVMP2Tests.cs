// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using System.IO;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    public static class ZipExtension
    {
        public static IEnumerable<TResult> Zip<TFirst, TSecond, TResult>(
            this IEnumerable<TFirst> first,
            IEnumerable<TSecond> second,
            Func<TFirst, TSecond, TResult> resultSelector)
        {
            using (IEnumerator<TFirst> e1 = first.GetEnumerator())
            using (IEnumerator<TSecond> e2 = second.GetEnumerator())
                while (e1.MoveNext() && e2.MoveNext())
                    yield return resultSelector(e1.Current, e2.Current);
        }
    }

    
    public class NonconjugateVMP2Tests
    {
        // generates plots for NIPS 2011 paper
        internal void SimpleLogistic()
        {
            //TODO: change path for cross platform using
            string baseDir = @"..\..\..\boundtests\";
            Directory.CreateDirectory(baseDir);
            // model is just \sigma(x)N(x;m,v)
            var prior = Variable.Observed<Gaussian>(new Gaussian());
            var x = Variable<double>.Random(prior);
            var s = Variable.Bernoulli(Variable.Logistic(x));
            s.ObservedValue = true;
            var ies = new Dictionary<string, InferenceEngine>();
            ies["true"] = new InferenceEngine();
            ies["ncvmp"] = new InferenceEngine(new VariationalMessagePassing());
            ies["ncvmp"].Compiler.GivePriorityTo(typeof (LogisticOp));
            ies["ncvmp_sj"] = new InferenceEngine(new VariationalMessagePassing());
            ies["ncvmp_sj"].Compiler.GivePriorityTo(typeof (LogisticOp_SJ99));
            ies["vmp_jj"] = new InferenceEngine(new VariationalMessagePassing());
            ies["vmp_jj"].Compiler.GivePriorityTo(typeof (LogisticOp_JJ96));
            foreach (var kvp in ies)
                kvp.Value.ShowProgress = false;
            double m, v;
            // m = -10, v=16
            var meanFile = new StreamWriter(baseDir + "meanVaryMean.txt");
            var varFile = new StreamWriter(baseDir + "varVaryMean.txt");
            //meanFile.WriteLine("m " + ies.Keys.Aggregate((p, q) => p + " " + q));
            //varFile.WriteLine("m " + ies.Keys.Aggregate((p, q) => p + " " + q));
            v = 10;
            for (int i = 0; i < 100; i++)
            {
                m = -20.0 + (40.0/100.0)*i;
                prior.ObservedValue = Gaussian.FromMeanAndVariance(m, v);
                meanFile.Write("{0:G6} ", m);
                varFile.Write("{0:G6} ", m);
                foreach (var kvp in ies)
                {
                    Console.WriteLine(".");
                    var res = kvp.Value.Infer<Gaussian>(x);
                    meanFile.Write("{0:G6} ", res.GetMean());
                    varFile.Write("{0:G6} ", res.GetVariance());
                }
                meanFile.WriteLine();
                varFile.WriteLine();
            }
            meanFile.Close();
            varFile.Close();

            meanFile = new StreamWriter(baseDir + "meanVaryVar.txt");
            varFile = new StreamWriter(baseDir + "varVaryVar.txt");
            //Console.WriteLine("v " + ies.Keys.Aggregate((p, q) => p + " " + q));
            m = 0;
            for (int i = 0; i < 100; i++)
            {
                v = 0.1 + (20.0/100.0)*i;
                prior.ObservedValue = Gaussian.FromMeanAndVariance(m, v);
                meanFile.Write("{0:G6} ", v);
                varFile.Write("{0:G6} ", v);
                foreach (var kvp in ies)
                {
                    Console.WriteLine(".");
                    var res = kvp.Value.Infer<Gaussian>(x);
                    meanFile.Write("{0:G6} ", res.GetMean());
                    varFile.Write("{0:G6} ", res.GetVariance());
                }
                meanFile.WriteLine();
                varFile.WriteLine();
            }
            meanFile.Close();
            varFile.Close();
        }


        public static double SliceSampleUnivariate(double init_x, Converter<double, double> log_prob, double init_width = 1, double lower_bound = double.NegativeInfinity,
                                                   double upper_bound = double.PositiveInfinity)
        {
            double logy = log_prob(init_x) - (-System.Math.Log(Rand.Double()));
            double u = init_width*Rand.Double();
            double L = init_x - u;
            double R = init_x + (init_width - u);

            while (true)
            {
                if (L <= lower_bound || log_prob(L) < logy)
                    break;
                L = L - init_width;
            }

            while (true)
            {
                if (R >= upper_bound || log_prob(R) < logy)
                    break;
                R = R + init_width;
            }

            L = System.Math.Max(L, lower_bound);
            R = System.Math.Min(R, upper_bound);
            double x1;
            while (true)
            {
                x1 = L + (R - L)*Rand.Double();
                if (log_prob(x1) >= logy)
                    break;
                if (x1 > init_x)
                    R = x1;
                else
                    L = x1;
            }
            //Console.WriteLine(x1); 
            return x1;
        }


        public static double[] SliceSamplingHelper(Converter<double, double> f, double x, double lower_bound, double upper_bound,
                                                   int burnin = 1000, int thin = 10, int nsamples = 1000, double init_width = 1.0)
        {
            for (int i = 0; i < burnin; i++)
                x = SliceSampleUnivariate(x, f, lower_bound: lower_bound, upper_bound: upper_bound, init_width: init_width);
            var samples = new double[nsamples];
            for (int i = 0; i < nsamples; i++)
            {
                for (int j = 0; j < thin; j++)
                    x = SliceSampleUnivariate(x, f, lower_bound: lower_bound, upper_bound: upper_bound, init_width: init_width);
                samples[i] = x;
            }
            return samples;
        }

        public static double[][] SliceSamplingHelper(Converter<double[], double> f, double[] x, double[] lower_bound, double[] upper_bound,
                                                     int burnin = 1000, int thin = 10, int nsamples = 1000)
        {
            for (int i = 0; i < burnin; i++)
                for (int j = 0; j < x.Length; j++)
                {
                    x[j] = SliceSampleUnivariate(x[j], o =>
                        {
                            x[j] = o;
                            return f(x);
                        }, lower_bound: lower_bound[j], upper_bound: upper_bound[j]);
                }

            var samples = new double[nsamples][];
            for (int i = 0; i < nsamples; i++)
            {
                for (int j = 0; j < thin; j++)
                    for (int k = 0; k < x.Length; k++)
                    {
                        x[k] = SliceSampleUnivariate(x[k], o =>
                            {
                                x[k] = o;
                                return f(x);
                            }, lower_bound: lower_bound[k], upper_bound: upper_bound[k]);
                    }
                samples[i] = x.Select(o => o).ToArray();
            }
            return samples;
        }

        internal static void TestSliceSampler()
        {
            double x = 1.0;
            int N = 1000;
            var samples = new double[N];
            for (int i = 0; i < N; i++)
            {
                x = SliceSampleUnivariate(x, j => Gaussian.FromMeanAndVariance(3, 1).GetLogProb(j));
                samples[i] = x;
            }
            double mean = samples.Sum()/N;
            Console.WriteLine("Mean= " + mean);
            Console.WriteLine("Var= " + (samples.Sum(i => i*i)/N - mean*mean));
        }

        public double ConjugateDirichletExpectationBySliceSampling(ConjugateDirichlet t, Converter<double, double> f, double x = double.NaN, int N = 10000, int thin = 10,
                                                                   int burnin = 10000)
        {
            if (double.IsNaN(x))
                x = t.GammaApproximation().GetMean();
            for (int i = 0; i < burnin; i++)
                x = SliceSampleUnivariate(x, j => t.GetUnnormalisedLogProb(j), lower_bound: 0.0, upper_bound: double.PositiveInfinity);
            var samples = new double[N];
            for (int i = 0; i < N; i++)
            {
                for (int k = 0; k < thin; k++)
                    x = SliceSampleUnivariate(x, j => t.GetUnnormalisedLogProb(j), lower_bound: 0.0, upper_bound: double.PositiveInfinity);
                samples[i] = f(x);
            }
            double mcMean = samples.Sum()/N;
            //double mcVar = (samples.Sum(i => i * i) / N - mcMean * mcMean);
            return mcMean;
        }

        //[Fact]
        internal void ConjugateDirichletQuadrature()
        {
            double scale = .01;
            var shapeRange = new double[] {0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100};
            double D = 4;
            double K = 1;
            double n = shapeRange.Length;
            Console.WriteLine("----------------- E[x] ------------------");
            Console.WriteLine("Shape True MC GH LA Asym");
            for (int i = 0; i < shapeRange.Length; i++)
                //Parallel.For(0, shapeRange.Length, i =>
            {
                var t = new ConjugateDirichlet(shapeRange[i], scale, D, K);
                var s = new Gamma(shapeRange[i], scale);
                Console.Write("{0} {1} {2} ", shapeRange[i], s.GetMean(), ConjugateDirichletExpectationBySliceSampling(t, x => x));
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadrature;
                Console.Write(t.GetMean() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
                Console.Write(t.GetMean() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
                Console.Write(t.GetMean() + " ");
                Console.WriteLine();
            } //);

            Console.WriteLine("----------------- E[log(x)] ------------------");
            Console.WriteLine("Shape True MC GH CC Asym");
            for (int i = 0; i < shapeRange.Length; i++)
                //Parallel.For(0, shapeRange.Length, i =>
            {
                var t = new ConjugateDirichlet(shapeRange[i], scale, D, K);
                var s = new Gamma(shapeRange[i], scale);
                Console.Write("{0} {1} {2} ", shapeRange[i], s.GetMeanLog(), ConjugateDirichletExpectationBySliceSampling(t, x => System.Math.Log(x)));
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadrature;
                Console.Write(t.GetMeanLog() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
                Console.Write(t.GetMeanLog() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
                Console.Write(t.GetMeanLog() + " ");
                Console.WriteLine();
            } //);

            Console.WriteLine("----------------- E[log Gamma(x)] ------------------");
            Console.WriteLine("Shape True MC GH CC Asym");
            for (int i = 0; i < shapeRange.Length; i++)
                //Parallel.For(0, shapeRange.Length, i =>
            {
                var t = new ConjugateDirichlet(shapeRange[i], scale, D, K);
                var s = new Gamma(shapeRange[i], scale);
                Console.Write("{0} {1} {2} ", shapeRange[i], GammaFromShapeAndRateOp.ELogGamma(s), ConjugateDirichletExpectationBySliceSampling(t, x => MMath.GammaLn(x)));
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadrature;
                Console.Write(t.GetMeanLogGamma(1.0) + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
                Console.Write(t.GetMeanLogGamma(1.0) + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
                Console.Write(t.GetMeanLogGamma(1.0) + " ");
                Console.WriteLine();
            } //);

            Console.WriteLine("----------------- E[log Gamma(4*x)] ------------------");
            Console.WriteLine("Shape True MC GH CC Asym");
            double factor = 4.0;
            for (int i = 0; i < shapeRange.Length; i++)
                //Parallel.For(0, shapeRange.Length, i =>
            {
                var t = new ConjugateDirichlet(shapeRange[i], scale, D, K);
                var s = new Gamma(shapeRange[i], scale*factor);
                Console.Write("{0} {1} {2} ", shapeRange[i], GammaFromShapeAndRateOp.ELogGamma(s),
                              ConjugateDirichletExpectationBySliceSampling(t, x => MMath.GammaLn(x*factor)));
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadrature;
                Console.Write(t.GetMeanLogGamma(factor) + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
                Console.Write(t.GetMeanLogGamma(factor) + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
                Console.Write(t.GetMeanLogGamma(factor) + " ");
                Console.WriteLine();
            } //);

            //double shape = 1, scale = 1;
            //var scaleRange = new double[] { 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100 };
            //double n = scaleRange.Length;
            //Console.WriteLine("Scale MC Quad True");
            //for (int i = 0; i < scaleRange.Length; i++)
            ////Parallel.For(0, shapeRange.Length, i =>
            //{
            //    var t = new ConjugateDirichlet(shape, scaleRange[i], 0, 0);
            //    var s = new Gamma(shape, scaleRange[i]);
            //    ConjugateDirichlet.clenshawCurtisQuadrature = false; 
            //    Console.Write("{0} {1} {2} {3} ", scaleRange[i], ConjugateDirichletMeanBySliceSampling(t), t.GetMean(), s.GetMean());
            //    ConjugateDirichlet.clenshawCurtisQuadrature = true;
            //    Console.WriteLine(t.GetMean()); 
            //}//);
        }

        internal void ConjugateDirichletQuadratureExtreme()
        {
            Console.WriteLine("D Truth Asym Quad QuadLaplace");
            var drange = new int[] {10, 20, 50, 100, 200, 500, 1000, 2000}; //, 10000 };
            foreach (var d in drange)
            {
                var t = new ConjugateDirichlet(1, 1.0/2.5e5, d /* 1e3 */, 5);
                Converter<double, double> f = o => t.GetUnnormalisedLogProb(System.Math.Exp(o)) + o;
                double initx, dummy;
                t.SmartProposal(out initx, out dummy);
                var samples = SliceSamplingHelper(f, initx, double.NegativeInfinity, double.PositiveInfinity);
                Console.Write(d + " ");
                Console.Write(samples.Sum(o => System.Math.Exp(o)) / samples.Length + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
                Console.Write(t.GetMean() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadrature;
                Console.Write(t.GetMean() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
                Console.Write(t.GetMean() + " ");
                Console.WriteLine();
            }

            Console.WriteLine("Rate Truth Asym Quad QuadLaplace");
            var rrange = new double[] {1, 10, 1e2, 1e3, 1e4, 1e5, 1e6};
            foreach (var r in rrange)
            {
                var t = new ConjugateDirichlet(1, 1.0/r, 10, 5);
                if (!t.IsProper())
                    continue;
                Converter<double, double> f = o => t.GetUnnormalisedLogProb(System.Math.Exp(o)) + o;
                double initx, dummy;
                t.SmartProposal(out initx, out dummy);
                var samples = SliceSamplingHelper(f, initx, double.NegativeInfinity, double.PositiveInfinity);
                Console.Write(r + " ");
                Console.Write(samples.Sum(o => System.Math.Exp(o)) / samples.Length + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
                Console.Write(t.GetMean() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadrature;
                Console.Write(t.GetMean() + " ");
                ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
                Console.Write(t.GetMean() + " ");
                Console.WriteLine();
            }

            Console.WriteLine();
        }

        private double ConjugateDirichletTestManual(Gamma prior, int[][] data, int n, int K, int D)
        {
            double alpha = 1.0;
            var counts = new SparseVector[K];
            for (int k = 0; k < K; k++)
            {
                counts[k] = SparseVector.Constant(D, 0.0);
                foreach (var word in data[k])
                    counts[k][word]++;
            }
            for (int i = 0; i < 500; i++)
            {
                var q = counts.Select(c => new Dirichlet(SparseVector.Constant(D, alpha) + c)).ToArray();
                var logp = q.Select(o => o.GetMeanLog()).ToArray();
                var sum = logp.Select(o => o.Sum()).Sum();

                Converter<double, double> l = a => K*(MMath.GammaLn(D*a) - D*MMath.GammaLn(a)) + (a - 1.0)*sum + prior.GetLogProb(a);
                Converter<double, double> lprime = a => K*(D*MMath.Digamma(D*a) - D*MMath.Digamma(a)) + sum + (prior.Shape - 1)/a - prior.Rate;
                Converter<double, double> lh = a => K*(D*D*MMath.Trigamma(D*a) - D*MMath.Trigamma(a)) - (prior.Shape - 1)/(a*a);
                Converter<double, double> f = loga => l(System.Math.Exp(loga)); // +loga;
                Converter<double, double> fprime = loga => System.Math.Exp(loga) * lprime(System.Math.Exp(loga)); // +1;
                Converter<double, double> fh = loga => System.Math.Exp(2.0* loga) * lh(System.Math.Exp(loga)) + System.Math.Exp(loga) * lprime(System.Math.Exp(loga));

                //foreach (var o in Enumerable.Range(-15, 15))
                //{
                //    Console.WriteLine((o / Math.Log(10.0)) + " " + f(o) + " " + fprime(o) + " " + fh(o));
                //}

                //alpha = Math.Exp(ConjugateDirichlet.Newton1DMaximise(Math.Log(alpha), f, fprime, fh)); 
                var alphaMarg = ConjugateDirichlet.FromShapeAndRate(prior.Shape, prior.Rate);
                for (int k = 0; k < K; k++)
                    alphaMarg *= DirichletSymmetricOp.AlphaAverageLogarithm(q[k], new ConjugateDirichlet(), logp[k]);
                var alphaQuadrature = alphaMarg.GetMean();
                var samples = SliceSamplingHelper(o => f(o) + o, System.Math.Log(alpha), double.NegativeInfinity, double.PositiveInfinity);
                alpha = samples.Sum(o => System.Math.Exp(o)) / samples.Length;
                double L = l(alpha);
                for (int k = 0; k < K; k++)
                {
                    L += logp[k].Inner(counts[k]);
                    L -= q[k].GetAverageLog(q[k]);
                }
                Console.WriteLine(i + " " + L + " " + alpha + " " + alphaQuadrature);
            }
            return alpha;
        }

        private double ConjugateDirichletTestSliceSample(Gamma prior, int[][] data, Range n, Range k, int D)
        {
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable.Observed(.01);
            var p = Variable.Array<Vector>(k);
            p[k] = Variable.DirichletSymmetric(D, alpha).ForEach(k);
            p.SetSparsity(Sparsity.Sparse);
            var d = Variable.Array(Variable.Array<int>(n), k);
            d[k][n] = Variable.Discrete(p[k]).ForEach(n);
            d.ObservedValue = data;
            modelBlock.CloseBlock();
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 1;
            Converter<double, double> f = o =>
                {
                    alpha.ObservedValue = o;
                    return prior.GetLogProb(o) + ie.Infer<Bernoulli>(ev).LogOdds;
                };

            Converter<double, double> flog = o => f(System.Math.Exp(o)) + o;
            int K = k.SizeAsInt;
            var counts = new SparseVector[K];
            for (int j = 0; j < k.SizeAsInt; j++)
            {
                counts[j] = SparseVector.Constant(D, 0.0);
                foreach (var word in data[j])
                    counts[j][word]++;
            }

            Func<double, SparseVector, double> trueZonePlate = (a, c) =>
                                                               MMath.GammaLn(D*a) - D*MMath.GammaLn(a) + c.Sum(o => MMath.GammaLn(o + a)) - MMath.GammaLn(c.Sum() + D*a);
            Converter<double, double> trueZ = a => counts.Sum(o => trueZonePlate(a, o));
            Converter<double, double> trueZwithPrior = a => prior.GetLogProb(a) + trueZ(a);
            Converter<double, double> trueZtransformed = l => trueZwithPrior(System.Math.Exp(l)) + l;

            //foreach (var o in Enumerable.Range(-40, 40))
            //{
            //    double res = double.NaN;
            //    try
            //    {
            //        res = flog(o);
            //    }
            //    catch { }
            //    Console.WriteLine((o / Math.Log(10.0)) + " " + res + " " + trueZtransformed(o));
            //}


            var samples = SliceSamplingHelper(trueZtransformed, 0, /*-15.0*Math.Log(10)*/ double.NegativeInfinity, double.PositiveInfinity, burnin: 100, thin: 1, nsamples: 100);

            double mcMean = samples.Sum(o => System.Math.Exp(o)) / samples.Length;
            double mcVari = samples.Sum(o => System.Math.Exp(2.0* o)) / samples.Length - mcMean* mcMean;
            Console.WriteLine("Mean " + mcMean + " std " + System.Math.Sqrt(mcVari));
            return mcMean;
        }

        [Fact]
        public void ConjugateDirichletTest()
        {
            Rand.Restart(12347);
            int D = 4, K = 5, N = 100;
            var gammaPrior = Gamma.FromShapeAndRate(1, 1);
            var priorAlpha = ConjugateDirichlet.FromShapeAndRate(gammaPrior.Shape, gammaPrior.Rate);
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable<double>.Random(priorAlpha);
            var k = new Range(K);
            var n = new Range(N);
            var p = Variable.Array<Vector>(k);
            p[k] = Variable.DirichletSymmetric(D, alpha).ForEach(k);
            var d = Variable.Array(Variable.Array<int>(n), k);
            d[k][n] = Variable.Discrete(p[k]).ForEach(n);

            modelBlock.CloseBlock();

            var ie = new InferenceEngine(new VariationalMessagePassing());
            // all values at 0, alpha should be small (got 0.01358)
            d.ObservedValue = Enumerable.Range(0, K).Select(_ => Enumerable.Range(0, N).Select(j => 0).ToArray()).ToArray();
            var post = ie.Infer<ConjugateDirichlet>(alpha);
            Console.WriteLine(post);
            //Console.WriteLine(ConjugateDirichletTestSliceSample(gammaPrior, d.ObservedValue, n, k, D));
            double mcAnswer = 0.0147279164611893;
            Assert.True(MMath.AbsDiff(post.GetMean(), mcAnswer) < 0.05);
            // random values, alpha should be large (got 6.237)
            d.ObservedValue = Enumerable.Range(0, K).Select(_ => Enumerable.Range(0, N).Select(j => Rand.Int(D)).ToArray()).ToArray();
            post = ie.Infer<ConjugateDirichlet>(alpha);
            Console.WriteLine(post);
            //Console.WriteLine(ConjugateDirichletTestSliceSample(gammaPrior, d.ObservedValue, n, k, D));
            mcAnswer = 5.89876812486751;
            Assert.True(MMath.AbsDiff(post.GetMean(), mcAnswer) < 0.1);
            // uniform probabilities, alpha should be close to one (got 1.036)
            d.ObservedValue = Enumerable.Range(0, K).Select(_ =>
                {
                    var prob = Dirichlet.Uniform(D).Sample();
                    return Enumerable.Range(0, N).Select(j => Rand.Sample(prob)).ToArray();
                }).ToArray();
            post = ie.Infer<ConjugateDirichlet>(alpha);
            Console.WriteLine(post);
            // Console.WriteLine(ConjugateDirichletTestSliceSample(gammaPrior, d.ObservedValue, n, k, D));
            mcAnswer = 0.821993443689866;
            Assert.True(MMath.AbsDiff(post.GetMean(), mcAnswer) < 0.05);
        }

        [Fact]
        public void ConjugateDirichletTest2()
        {
            Rand.Restart(12347);
            //int D = 1000, K = 5, N = 10000, numNonSparse = 10;
            int D = 10, K = 4, N = 4, numNonSparse = 2;
            var gammaPrior = Gamma.FromShapeAndRate(1, 1);
            var priorAlpha = ConjugateDirichlet.FromShapeAndRate(gammaPrior.Shape, gammaPrior.Rate);
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable<double>.Random(priorAlpha);
            var k = new Range(K);
            var n = new Range(N);
            var p = Variable.Array<Vector>(k);
            p.SetSparsity(Sparsity.Sparse);
            p[k] = Variable.DirichletSymmetric(D, alpha).ForEach(k);
            var d = Variable.Array(Variable.Array<int>(n), k);
            d[k][n] = Variable.Discrete(p[k]).ForEach(n);

            modelBlock.CloseBlock();

            var ie = new InferenceEngine(new VariationalMessagePassing());
            ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.GaussHermiteQuadratureLaplace;
            // all values at 0, alpha should be small (got 0.01358)
            //d.ObservedValue = Enumerable.Range(0, K).Select(_ => Enumerable.Range(0, N).Select(j => 0).ToArray()).ToArray();
            d.ObservedValue = Enumerable.Range(0, K).Select(_ =>
                {
                    var indices = Rand.Perm(D).Take(numNonSparse).ToArray();
                    return Enumerable.Range(0, N).Select(j => indices[j%numNonSparse]).ToArray();
                }).ToArray();

            //var post = ie.Infer<ConjugateDirichlet>(alpha);
            // ConjugateDirichlet.approximationMethod = ConjugateDirichlet.ApproximationMethod.Asymptotic;
            var ca = ie.GetCompiledInferenceAlgorithm(ev, alpha);
            for (int i = 0; i < 100; i++)
            {
                ca.Update(1);
                var marg = ca.Marginal<ConjugateDirichlet>(alpha.NameInGeneratedCode);
                Console.WriteLine(ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds + " " + marg + " mode:" + marg.GetMode());
            }
            var post = ca.Marginal<ConjugateDirichlet>(alpha.NameInGeneratedCode);

            Console.WriteLine(post);
            double m, v;
            post.SmartProposal(out m, out v);
            Console.WriteLine(m);

            ConjugateDirichletTestManual(gammaPrior, d.ObservedValue, N, K, D);

            Console.WriteLine(ConjugateDirichletTestSliceSample(gammaPrior, d.ObservedValue, n, k, D));

            foreach (var o in Enumerable.Range(-40, 40))
            {
                double res = double.NaN;
                try
                {
                    res = post.GetUnnormalisedLogProb(System.Math.Exp(o)) + o;
                }
                catch
                {
                }
                Console.WriteLine((o / System.Math.Log(10)) + " " + res);
            }
        }

        public double[] ConjugateDirichletTestPointmassHelper(Gamma gammaPrior, Vector obs, double[] range)
        {
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var priorAlpha = ConjugateDirichlet.FromShapeAndRate(gammaPrior.Shape, gammaPrior.Rate);
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable<double>.Random(priorAlpha);
            var p = Variable.DirichletSymmetric(obs.Count, alpha);
            modelBlock.CloseBlock();
            p.ObservedValue = obs;
            Converter<double, double> logPost = o =>
                {
                    alpha.ObservedValue = System.Math.Exp((double) o);
                    return ie.Infer<Bernoulli>(ev).LogOdds + (double) o;
                };
            var samples = SliceSamplingHelper(logPost, 0, double.NegativeInfinity, double.PositiveInfinity);
            var mean = samples.Sum(o => System.Math.Exp(o)) / samples.Length;
            Console.WriteLine("MC mean: " + mean);
            return range.Select(o => logPost(o)).ToArray();
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void ConjugateDirichletTestPointmass()
        {
            Rand.Restart(12347);
            int D = 10000;
            var gammaPrior = Gamma.FromShapeAndRate(.1, .1);
            var priorAlpha = ConjugateDirichlet.FromShapeAndRate(gammaPrior.Shape, gammaPrior.Rate);
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable<double>.Random(priorAlpha);
            var p = Variable.DirichletSymmetric(D, alpha);

            modelBlock.CloseBlock();

            var ie = new InferenceEngine(new VariationalMessagePassing());
            p.ObservedValue = Dirichlet.Symmetric(D, .1).Sample();
            if (p.ObservedValue.Any(o => o == 0.0))
                throw new NotSupportedException("Can't deal with 0 probs");

            //Console.WriteLine(ConjugateDirichletTestSliceSample(gammaPrior, d.ObservedValue, n, k, D));
            var post = ie.Infer<ConjugateDirichlet>(alpha);
            Console.WriteLine(post);
            var range = Enumerable.Range(-20, 40).Select(o => (double) o);
            var res = range.Select(o => post.GetUnnormalisedLogProb(System.Math.Exp(o)) + o).ToArray();
            var res2 = ConjugateDirichletTestPointmassHelper(gammaPrior, p.ObservedValue, range.ToArray());
            var diff = res.Zip(res2, (a, b) => a - b).ToArray();
            Assert.True(diff.All(o => (o - diff[0]) < 1e-6));
        }

        private double ConjugateDirichletTestSliceSample2(Gamma prior, Dirichlet[] likelihood, Range k, int D)
        {
            int burnin = 100;
            int thin = 1;
            int N = 1000;
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable.Observed(1.0);
            var p = Variable.Array<Vector>(k);
            p[k] = Variable.DirichletSymmetric(D, alpha).ForEach(k);
            var pMessage = Variable.Observed(likelihood, k);
            Variable.ConstrainEqualRandom(p[k], pMessage[k]);
            modelBlock.CloseBlock();
            var ie = new InferenceEngine();
            double x = 1.0;
            Converter<double, double> f = o =>
                {
                    alpha.ObservedValue = o;
                    return prior.GetLogProb(o) + ie.Infer<Bernoulli>(ev).LogOdds;
                };
            for (int i = 0; i < burnin; i++)
                x = SliceSampleUnivariate(x, f, lower_bound: 0.0, upper_bound: double.PositiveInfinity);
            var samples = new double[N];
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < thin; j++)
                    x = SliceSampleUnivariate(x, f, lower_bound: 0.0, upper_bound: double.PositiveInfinity);
                Console.Write(".");
                if (i%50 == 0) Console.WriteLine();
                samples[i] = x;
            }
            double mcMean = samples.Sum()/N;
            return mcMean;
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void ConjugateDirichletExtremeTest()
        {
            Rand.Restart(12347);
            int D = 1000, K = 5;
            int nonsparse = 10;
            var gammaPrior = Gamma.FromShapeAndRate(1, 1);
            var priorAlpha = ConjugateDirichlet.FromShapeAndRate(gammaPrior.Shape, gammaPrior.Rate);
            var ev = Variable.Bernoulli(0.5);
            var modelBlock = Variable.If(ev);
            var alpha = Variable<double>.Random(priorAlpha);
            var k = new Range(K);
            var p = Variable.Array<Vector>(k);
            p[k] = Variable.DirichletSymmetric(D, alpha).ForEach(k);
            var likelihood = Enumerable.Range(0, K).Select(_ =>
                {
                    //var res = Vector.Zero(D); 
                    var res = Vector.Constant(D, 0.001);
                    var perm = Rand.Perm(D);
                    for (int i = 0; i < nonsparse; i++)
                    {
                        res[perm[i]] = 1000;
                    }
                    return new Dirichlet(res);
                }).ToArray();
            var pMessage = Variable.Observed(likelihood, k);
            Variable.ConstrainEqualRandom(p[k], pMessage[k]);
            modelBlock.CloseBlock();

            var ie = new InferenceEngine(new VariationalMessagePassing());
            // all values at 0, alpha should be small (got 0.01358)
            var post = ie.Infer<ConjugateDirichlet>(alpha);
            Console.WriteLine(post);
            Console.WriteLine(ConjugateDirichletTestSliceSample2(gammaPrior, likelihood, k, D));
        }

        [Fact]
        public void SoftmaxEvidence()
        {
            int K = 3, obs = 1;
            var k = new Range(K);
            var xPrior = Enumerable.Range(0, K).Select(c => Gaussian.FromMeanAndPrecision(0, 1)).ToArray();
            //xPrior[0] = Gaussian.PointMass(0);
            var xPriorVar = Variable.Observed(xPrior, k);
            var ev = Variable.Bernoulli(0.5).Named("evidence");
            VariableArray<double> x;
            using (Variable.If(ev))
            {
                x = Variable.Array<double>(k);
                x[k] = Variable<double>.Random(xPriorVar[k]);
                var s = Variable.Discrete(Variable.Softmax(x));
                s.ObservedValue = obs;
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            // SaulJordanSoftmaxOp_LBFGS does not work will unrolling because it assumes the array is updated in parallel
            ie.Compiler.UnrollLoops = false;
            //ie.Compiler.GivePriorityTo(typeof(SoftmaxOp_Bouchard)); 
            //ie.Compiler.GivePriorityTo(typeof(SaulJordanSoftmaxOp_NCVMP_Sparse)); 
            double evidence = ie.Infer<Bernoulli>(ev).LogOdds;
            var logOdds = ie.Infer<Gaussian[]>(x);
            Vector ms, vs;
            SoftmaxOp_BL06_LBFGS.GetMeanAndVariance(logOdds, out ms, out vs);
            var a = SoftmaxOp_KM11_LBFGS.AInit(logOdds);
            for (int i = 0; i < 20; i++) // actually converges after 8 or its
                a = SoftmaxOp_KM11_LBFGS.A(logOdds, a);
            double logSumExp = SoftmaxOp_KM11_LBFGS.ExpectationLogSumExp_Helper(ms, vs, a);
            double halfSumA2v = .5*(a*a*vs).Sum();
            double myEv = ms[obs] - (logSumExp + halfSumA2v)
                          + logOdds.Zip(xPrior, (c, d) => c.GetAverageLog(d)).Aggregate((c, d) => c + d)
                          - logOdds.Select(c => c.GetAverageLog(c)).Aggregate((c, d) => c + d);
            Console.WriteLine("Log odds : " + logOdds.Select(c => c.ToString()).Aggregate((c, d) => c + ", " + d));
            Console.WriteLine("Lower bound calculated by Infer.NET : " + evidence);
            Console.WriteLine("Lower bound calculated by hand : " + myEv);
            Assert.True(MMath.AbsDiff(evidence, myEv) < 1e-10);
            // SJ LBFGS: -1.13738827734361
            // Blei LBFGS -1.27810506273509
            // SJ NCVMP: -1.13738827734363
            // Product of logistic: -1.36026138352298
            // Bouchard: -1.9479
            // As expected SJ is tighter than Blei
        }

        // Sometimes undeterministically fails. Usually it happens one or two times during CompilerOptions run.
        // Sets of compiler options on which this test fails differ from run to run.
        [Fact]
        [Trait("Category", "OpenBug")]
        public void SoftmaxEvidenceReducesToLogistic()
        {
            Func<Gaussian, object, double> f_softmax = (prior, factor) =>
                {
                    var k = new Range(2);
                    var xPrior = new Gaussian[] {Gaussian.PointMass(0), prior};
                    var xPriorVar = Variable.Observed(xPrior, k);
                    var ev = Variable.Bernoulli(0.5).Named("evidence");
                    VariableArray<double> x;
                    using (Variable.If(ev))
                    {
                        x = Variable.Array<double>(k);
                        x[k] = Variable<double>.Random(xPriorVar[k]);
                        var s = Variable.Discrete(Variable.Softmax(x));
                        s.ObservedValue = 1;
                    }
                    var ie = new InferenceEngine(new VariationalMessagePassing());
                    ie.Compiler.GivePriorityTo(factor);
                    ie.ShowProgress = false;
                    // SaulJordanSoftmaxOp_LBFGS does not work will unrolling because it assumes the array is updated in parallel
                    ie.Compiler.UnrollLoops = false;
                    return ie.Infer<Bernoulli>(ev).LogOdds; //, ie.Infer<Gaussian[]>(x)[1]); 
                };

            Func<Gaussian, object, double> f_logistic = (prior, factor) =>
                {
                    var xPrior = new Gaussian[] {Gaussian.PointMass(0), prior};
                    var xPriorVar = Variable.Observed(xPrior);
                    var ev = Variable.Bernoulli(0.5).Named("evidence");
                    Variable<double> x;
                    using (Variable.If(ev))
                    {
                        x = Variable<double>.Random(prior);
                        var s = Variable.Bernoulli(Variable.Logistic(x));
                        s.ObservedValue = true;
                    }
                    var ie = new InferenceEngine(new VariationalMessagePassing());
                    ie.Compiler.GivePriorityTo(factor);
                    ie.ShowProgress = false;
                    // SaulJordanSoftmaxOp_LBFGS does not work will unrolling because it assumes the array is updated in parallel
                    ie.Compiler.UnrollLoops = false;
                    return ie.Infer<Bernoulli>(ev).LogOdds;
                };

            var softmaxFactors = new[]
                {
                    typeof (SoftmaxOp_KM11_Sparse), typeof (SoftmaxOp_KM11_LBFGS), typeof (SoftmaxOp_KM11_LBFGS_Sparse),
                    typeof (SoftmaxOp_KM11)
                };
            var logisticFactor = typeof (LogisticOp_SJ99);

            foreach (var p in new Gaussian[] {Gaussian.FromMeanAndPrecision(0, 1), Gaussian.FromMeanAndPrecision(1, 1), Gaussian.FromMeanAndVariance(2, 10)})
            {
                Console.WriteLine("Prior=" + p);

                double evLogistic = f_logistic(p, logisticFactor);
                foreach (var f in softmaxFactors)
                {
                    double evSoftmax = f_softmax(p, f);
                    Console.WriteLine($"{StringUtil.TypeToString(f)} ev: {evSoftmax}, logistic ev: {evLogistic}");
                    Assert.True(MMath.AbsDiff(evSoftmax, evLogistic) < 1.0e-5);
                }
                //Console.WriteLine("Softmax x: {0}, logistic x: {1}", softmaxResult.Item2, logisticResult.Item2);
            }
            //Console.WriteLine("x : " + ie.Infer(x));
            //Console.WriteLine("Lower bound calculated by Infer.NET : " + evidence);
            //Assert.True(MMath.AbsDiff(evidence, -0.701188555472956) < 1e-10); // "true" value from logistic version
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 429
#endif

        [Fact]
        public void LogisticEvidence()
        {
            var xPrior = Gaussian.FromMeanAndPrecision(1, 1);
            var ev = Variable.Bernoulli(0.5).Named("evidence");
            Variable<double> x;
            using (Variable.If(ev))
            {
                x = Variable<double>.Random(Variable.Observed(xPrior));
                var s = false
                            ? Variable<bool>.Factor(Factor.BernoulliFromLogOdds, x)
                            : Variable.Bernoulli(Variable.Logistic(x));
                s.ObservedValue = true;
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            //ie.Compiler.GivePriorityTo(typeof(LogisticOp));
            //ie.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
            ie.Compiler.GivePriorityTo(typeof (LogisticOp_SJ99));
            //ie.Compiler.GivePriorityTo(typeof(BernoulliFromLogOddsOp));
            //ie.NumberOfIterations = 1000;
            //var ieEP = new InferenceEngine(new ExpectationPropagation());
            double evidence = ie.Infer<Bernoulli>(ev).LogOdds;
            var logOdds = ie.Infer<Gaussian>(x);
            //Console.WriteLine(logOdds);
            //Console.WriteLine("EP : " + ieEP.Infer(x));
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v); //Gaussian(0.4144, 0.8047)
            double myEv = m - MMath.Log1PlusExpGaussian(m, v) + logOdds.GetAverageLog(xPrior) - logOdds.GetAverageLog(logOdds);
            Console.WriteLine("Log odds : " + logOdds);
            Console.WriteLine("Lower bound calculated by Infer.NET : " + evidence);
            Console.WriteLine("Lower bound calculated by hand : " + myEv);
            //Assert.True(MMath.AbsDiff(evidence, myEv) < 1e-10);
            Console.WriteLine("True evidence: " + MMath.LogisticGaussian(xPrior.GetMean(), xPrior.GetVariance()));
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 429
#endif

        [Fact]
        public void LogisticEvidenceMultipleUses()
        {
            var xPrior = Gaussian.FromMeanAndPrecision(0, 1);
            var ev = Variable.Bernoulli(0.5).Named("evidence");
            Variable<double> x;
            using (Variable.If(ev))
            {
                x = Variable<double>.Random(Variable.Observed(xPrior));
                var l = Variable.Logistic(x);
                var s = Variable.Bernoulli(l);
                s.ObservedValue = true;
                var t = Variable.Bernoulli(l);
                t.ObservedValue = true;
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            ie.Compiler.GivePriorityTo(typeof (LogisticOp));
            double evidence = ie.Infer<Bernoulli>(ev).LogOdds;
            var logOdds = ie.Infer<Gaussian>(x);
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            double myEv = 2*(m - MMath.Log1PlusExpGaussian(m, v)) + logOdds.GetAverageLog(xPrior) - logOdds.GetAverageLog(logOdds);
            Console.WriteLine("Log odds : " + logOdds);
            Console.WriteLine("Lower bound calculated by Infer.NET : " + evidence);
            Console.WriteLine("Lower bound calculated by hand : " + myEv);
            Assert.True(MMath.AbsDiff(evidence, myEv) < 1e-10);
            Console.WriteLine("True evidence: " + MMath.LogisticGaussian(xPrior.GetMean(), xPrior.GetVariance()));
        }

        [Fact]
        public void LogisticEvidenceDeterministicParent()
        {
            var xPrior = Gaussian.FromMeanAndPrecision(0, 1);
            var ev = Variable.Bernoulli(0.5).Named("evidence");
            Variable<double> x;
            using (Variable.If(ev))
            {
                x = Variable<double>.Random(Variable.Observed(xPrior));
                var l = Variable.Logistic(x*3);
                var s = Variable.Bernoulli(l);
                s.ObservedValue = true;
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            ie.Compiler.GivePriorityTo(typeof (LogisticOp));
            double evidence = ie.Infer<Bernoulli>(ev).LogOdds;
            var logOdds = ie.Infer<Gaussian>(x);
            double m, v;
            logOdds.GetMeanAndVariance(out m, out v);
            double myEv = 3*m - MMath.Log1PlusExpGaussian(3*m, 9*v) + logOdds.GetAverageLog(xPrior) - logOdds.GetAverageLog(logOdds);
            Console.WriteLine("Log odds : " + logOdds);
            Console.WriteLine("Lower bound calculated by Infer.NET : " + evidence);
            Console.WriteLine("Lower bound calculated by hand : " + myEv);
            Assert.True(MMath.AbsDiff(evidence, myEv) < 1e-10);
            Console.WriteLine("True evidence: " + MMath.LogisticGaussian(xPrior.GetMean(), xPrior.GetVariance()));
        }

        // This tests the initialization of cluster means rather than assignments.
        // Fails if the initialisation for means is not used.
        [Fact]
        public void MixtureOfMultivariateGaussiansWithHyperLearning()
        {
            // Define a range for the number of mixture components
            Range k = new Range(2).Named("k");

            // Mixture component means
            VariableArray<Vector> means = Variable.Array<Vector>(k).Named("means");
            means[k] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(2), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01)).ForEach(k);

            // Mixture component precisions
            VariableArray<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");
            precs[k] = Variable.WishartFromShapeAndScale(3.0, PositiveDefiniteMatrix.IdentityScaledBy(2, 1)).ForEach(k);

            // Mixture weights 
            //Variable<Vector> weights = Variable.Dirichlet(k, new double[] { 1, 1 }).Named("weights");
            var alpha = Variable.GammaFromShapeAndRate(1, 2).Named("alpha");
            //alpha.ObservedValue = 1;
            //var alpha = Variable.GammaFromMeanAndVariance(1, .1); 
            var weights = Variable.DirichletSymmetric(k, alpha).Named("weights");
            // Create a variable array which will hold the data
            Range n = new Range(300).Named("n");
            VariableArray<Vector> data = Variable.Array<Vector>(n).Named("x");
            // Create latent indicator variable for each data point
            VariableArray<int> z = Variable.Array<int>(n).Named("z");

            // The mixture of Gaussians model
            using (Variable.ForEach(n))
            {
                z[n] = Variable.Discrete(weights).Named("z");
                using (Variable.Switch(z[n]))
                {
                    data[n] = Variable.VectorGaussianFromMeanAndPrecision(means[z[n]], precs[z[n]]);
                }
            }

            // Attach some generated data
            double truePi = 0.6;
            data.ObservedValue = (new MixtureTests()).GenerateData(n.SizeAsInt, truePi);

            // Initialise messages randomly so as to break symmetry
            Discrete[] zinit = new Discrete[n.SizeAsInt];
            for (int i = 0; i < zinit.Length; i++)
                zinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
            //z.InitialiseTo(Distribution<int>.Array(zinit));
            var temp = Vector.Zero(2);
            VectorGaussian[] mInit = Util.ArrayInit(k.SizeAsInt, i =>
            {
                Rand.NormalP(Vector.Zero(2), PositiveDefiniteMatrix.Identity(2), temp);
                return new VectorGaussian(temp, PositiveDefiniteMatrix.Identity(2));
            });
            means.InitialiseTo(Distribution<Vector>.Array(mInit));

            // The inference
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            //ie.Compiler.GenerateInMemory = false;
            //ie.NumberOfIterations = 200;
            Dirichlet wDist = ie.Infer<Dirichlet>(weights);
            Vector wEstMean = wDist.GetMean();

            object meansActual = ie.Infer(means);
            Console.WriteLine("means = ");
            Console.WriteLine(meansActual);
            Wishart[] precsActual = ie.Infer<Wishart[]>(precs);
            for (int i = 0; i < precsActual.Length; i++)
            {
                Console.WriteLine(StringUtil.JoinColumns("precs[", i, "] = ", precsActual[i].GetMean()));
            }
            Console.WriteLine("w = {0} should be {1}", wEstMean, Vector.FromArray(truePi, 1 - truePi));
            Console.WriteLine("alpha = " + ie.Infer(alpha));
            //Console.WriteLine(StringUtil.JoinColumns("z = ", ie.Infer(z)));
            Assert.True(
                MMath.AbsDiff(wEstMean[0], truePi) < 0.05 ||
                MMath.AbsDiff(wEstMean[1], truePi) < 0.05);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void ElogGammaTest()
        {
            double a = 4.0, b = 6.0;
            var y = Vector.Zero(2);
            y[0] = a;
            y[1] = b;
            double truth = 0.4764435;
            double res = GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b));
            Console.WriteLine("{0} should be {1}", res, truth);
            Assert.True(System.Math.Abs(truth - res) < 1e-3);

            a = 150;
            b = 300;
            truth = 0.5766655;
            res = GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b));
            Console.WriteLine("{0} should be {1}", res, truth);
            Assert.True(System.Math.Abs(truth - res) < 1e-3);

            a = 150.0;
            b = 300.0/1000.0;
            truth = 2607.153;
            res = GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(a, b));
            Console.WriteLine("{0} should be {1}", res, truth);
            Assert.True(MMath.AbsDiff(truth - res, 1e-4) < 1e-3);
        }

        [Fact]
        public void TestGammaOpSpecialFunctions()
        {
            // Results from Matlab
            var a_range = new double[] {0.1, 1, 10};
            var b_range = new double[] {0.1, 1, 10};
            var res1 = new double[,]
                {
                    {-98.1128745656953, -101.619052916491, -101.463901369094},
                    {23.083823898752, -1.07591234606867, -1.68177226593525},
                    {46.0084163325605, 2.24986111100127, -0.06164419898917}
                };
            var res2 = new double[,]
                {
                    {2.73149117235933, 1.01046822799348, 0.100323253944879},
                    {-268.807503321411, 0.209471882085483, 0.103017290294049},
                    {-4649.33539610706, -23.0083294831535, 0.0533941295190074}
                };
            for (int ia = 0; ia < 3; ia++)
                for (int ib = 0; ib < 3; ib++)
                {
                    double a = a_range[ia];
                    double b = b_range[ib];
                    Gamma g = Gamma.FromShapeAndRate(a, b);
                    Vector grad = Factors.GammaFromShapeAndRateOp.CalculateDerivatives(g);
                    double err, relativeErr;
                    err = System.Math.Abs(grad[0] - res1[ia, ib]);
                    relativeErr = err/res1[ia, ib];
                    Assert.True(relativeErr < 1E-10);
                    Console.WriteLine("a=" + a + " b=" + b + " rel err in dSbyda=" + relativeErr);
                    err = System.Math.Abs(grad[1] - res2[ia, ib]);
                    relativeErr = err/res2[ia, ib];
                    Assert.True(relativeErr < 1E-10);
                    Console.WriteLine("a=" + a + " b=" + b + " rel err in dSbydb=" + relativeErr);
                }
        }


        internal void DirichletFactorTest3()
        {
            int K = 3;
            Range dim = new Range(K);
            //Variable<Vector> mean = Variable.Constant(Vector.Constant(K,1.0/K)).Named("mean");
            Variable<double> totalCount = Variable.GammaFromShapeAndRate(1, K).Named("totalCount");
            Variable<Vector> p = Variable.DirichletSymmetric(K, totalCount).Named("p");
            var d = Variable.Discrete(p);
            d.ObservedValue = 1;
            //Variable<Vector> p2 = DirichletSymmetric(totalCount, K).Named("p2");
            //var d2 = Variable.Discrete(p2);
            //d2.ObservedValue = 0; 
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(totalCount));
        }

        // TODO: add assertion
        internal void CopyTruncatedGaussianTest()
        {
            // truncating a Gaussian with precision=0 gives an exponential distribution
            double a = .1;
            var x = Variable.Random(new TruncatedGaussian(Gaussian.FromNatural(-a, 0), 0, double.PositiveInfinity));
            var y = Variable.Copy(x);
            Variable.ConstrainEqualRandom(y, new Gaussian(-3, 10));
            y.AddAttribute(new MarginalPrototype(new Gaussian()));
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(y));
        }

        internal void TestSparseSoftmax()
        {
            double a = 2;
            int K = 1000, N = 50000;
            // generate data
            var trueX = Vector.Zero(K);
            trueX.SetToFunction(trueX, t => Rand.Gamma(1)/a);
            //var trueP = Dirichlet.Sample(Vector.Constant(K, .5), Vector.Zero(K));
            var trueP = MMath.Softmax(trueX.ToArray());
            var data = new int[N];
            for (int i = 0; i < N; i++)
                data[i] = Rand.Sample(trueP);
            // model
            var k = new Range(K);
            var n = new Range(N);
            var x = Variable.Array<double>(k);
            x[k] = Variable.Random(new TruncatedGaussian(Gaussian.FromNatural(-a, 0), 0, double.PositiveInfinity)).ForEach(k);
            var y = Variable.Array<double>(k);
            y[k] = Variable.Copy(x[k]);
            y.AddAttribute(new MarginalPrototype(new Gaussian()));
            var p = Variable.Softmax(y);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowTimings = true;
            //Console.WriteLine("------- Using Blei 2006 bound and LBFGS ---------");
            //SoftmaxOp.UseBlei06Bound = true;
            Console.WriteLine("RMSE: " + RMSE(ie.Infer<DistributionArray<Gaussian>>(y), trueX));
            //ie.Reset();
            //Console.WriteLine("------- Using ProductOfLogistic bound ---------");
            //Console.WriteLine("RMSE: " + RMSE(ie.Infer<DistributionArray<Gaussian>>(y), trueX));
        }

        //[Fact]
        internal void TestSparseSoftmax2()
        {
            Rand.Restart(12345);
            double a = 2;
            int K = 20, N = 5000;
            // generate data
            var trueX = Vector.Zero(K);
            trueX.SetToFunction(trueX, t => Rand.Gamma(1)/a);
            //var trueP = Dirichlet.Sample(Vector.Constant(K, .5), Vector.Zero(K));
            var trueP = MMath.Softmax(trueX.ToArray());
            var data = new int[N];
            for (int i = 0; i < N; i++)
                data[i] = Rand.Sample(trueP);
            // model
            var k = new Range(K);
            var n = new Range(N);
            var x = Variable.Array<double>(k);
            var av = Variable.GammaFromShapeAndRate(.1, .1);
            x[k] = Variable.GammaFromShapeAndRate(1, av).ForEach(k);
            var p = Variable.Softmax(x);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowTimings = true;
            Console.WriteLine("------- Using Blei 2006 bound and LBFGS ---------");
            ie.Compiler.GivePriorityTo(typeof (SoftmaxOp_KM11_LBFGS));
            var post = ie.Infer<DistributionArray<Gamma>>(x);
            var postMean = Vector.Zero(K);
            for (int i = 0; i < K; i++)
            {
                postMean[i] = post[i].GetMean();
                Console.WriteLine("{0} should be {1}", postMean[i], trueX[i]);
            }
            double corr = postMean.Inner(trueX) / System.Math.Sqrt(postMean.Sum(z => z* z) * trueX.Sum(z => z* z));
            Console.WriteLine("Corr: " + corr);
            Console.WriteLine("RMSE: " + RMSE(post, trueX));
            //ie.Reset();
            //Console.WriteLine("------- Using old bound ---------");
            //SoftmaxOp.UseBlei06Bound = false;
            //Console.WriteLine("RMSE: " + RMSE(ie.Infer<DistributionArray<Gaussian>>(y), trueX));
        }


        internal void TestSparseSoftmax3()
        {
            var data = new int[] {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3};
            int K = 4, N = data.Length;
            // model
            var k = new Range(K);
            var n = new Range(N);
            var x = Variable.Array<double>(k);
            x[k] = Variable.GammaFromShapeAndRate(1.4, 1.5).ForEach(k);
            var p = Variable.Softmax(x);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowTimings = true;
            Console.WriteLine("------- Using Blei 2006 bound and LBFGS ---------");
            var post = ie.Infer<DistributionArray<Gamma>>(x);
            Console.WriteLine(post);
        }

        // Gamma-distributed Dirichlet totalCount
        internal void TestLearningDirichletHyper()
        {
            var a = Variable.GammaFromShapeAndRate(1, 1);
            var data = new int[] {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3};
            int K = 4, N = data.Length;
            // model
            var k = new Range(K);
            var n = new Range(N);
            var p = Variable.DirichletSymmetric(K, a);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowTimings = true;
            var post = ie.Infer<Gamma>(a);
            Console.WriteLine(post);
        }

        internal void TestLearningDirichletHyper2()
        {
            var a = Variable.GammaFromShapeAndRate(1, 1);
            double trueA = .1;
            int K = 1000, N = 1000;
            var data = new int[N];
            var trueP = Dirichlet.Sample(Vector.Constant(K, trueA), Vector.Zero(K));
            for (int i = 0; i < N; i++)
                data[i] = Rand.Sample(trueP);
            // model
            var k = new Range(K);
            var n = new Range(N);
            var p = Variable.DirichletSymmetric(K, a);
            var d = Variable.Array<int>(n);
            d[n] = Variable.Discrete(p).ForEach(n);
            d.ObservedValue = data;
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowTimings = true;
            var post = ie.Infer<Gamma>(a);
            Console.WriteLine(post);
            double m, v;
            post.GetMeanAndVariance(out m, out v);
            Console.WriteLine("Mean {0} sd {1}", m, System.Math.Sqrt(v));
        }


        //public void TestGIG()
        //{
        //    var x = Variable<double>.Random(new GeneralisedInverseGaussian(10.1,3,3));
        //    var y = Variable<double>.Random(new GeneralisedInverseGaussian(1.1, 3, 3));
        //    var z = x - y;
        //    z.AddAttribute(new MarginalPrototype(new Gamma()));
        //    Variable.ConstrainEqualRandom(z, new Gamma(1, 1)); 
        //    var ie = new InferenceEngine(new VariationalMessagePassing());
        //    Console.WriteLine(ie.Infer(x));
        //}

        public static double RMSE(DistributionArray<Gaussian> inferred, Vector truth)
        {
            int K = truth.Count;
            double sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += (inferred[k].GetMean() - truth[k])*(inferred[k].GetMean() - truth[k]);
            }
            return System.Math.Sqrt(sum / (double) K);
        }

        public static double RMSE(DistributionArray<Gamma> inferred, Vector truth)
        {
            int K = truth.Count;
            double sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += (inferred[k].GetMean() - truth[k])*(inferred[k].GetMean() - truth[k]);
            }
            return System.Math.Sqrt(sum / (double) K);
        }

        // Gamma plus is not implemented yet
        internal void TestGammaPlus()
        {
            var x = Variable.GammaFromMeanAndVariance(1, 2);
            var y = Variable.GammaFromMeanAndVariance(1, 1);
            var z = x + y;
            z.AddAttribute(new MarginalPrototype(new Gamma()));
            var zLikelihood = new Gamma();
            zLikelihood.SetMeanAndVariance(10, 1);
            Variable.ConstrainEqualRandom(z, zLikelihood);
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var post = ie.Infer<Gamma>(x);
            double m, v;
            post.GetMeanAndVariance(out m, out v);
            Console.WriteLine("Mean {0} variance {1}", m, v);
        }

        internal void HeteroRegressionTest()
        {
            var data = Enumerable.Range(0, 40).Select(o => Vector.FromArray(1.0, ((double) o - 20.0)/10.0)).ToArray();
            var wTrue = Vector.FromArray(2.0, 1.0);
            var whTrue = Vector.FromArray(0.0, -1.0);
            Console.WriteLine("wTrue = {0}", wTrue);
            Console.WriteLine("whTrue = {0}", whTrue);
            int n = data.Length;
            var yobs = new double[n];
            for (int i = 0; i < n; i++)
            {
                yobs[i] = data[i].Inner(wTrue) + Rand.Normal()* System.Math.Exp(-0.5* whTrue.Inner(data[i]));
                //Console.WriteLine(data[i][1] + " " + yobs[i]);
            }

            Range rows = new Range(n);

            VariableArray<Vector> x = Variable.Constant(data, rows).Named("x");

            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(new double[] {0, 0}),
                PositiveDefiniteMatrix.Identity(2)).Named("w");

            Variable<Vector> wh = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(new double[] {0, 0}),
                PositiveDefiniteMatrix.Identity(2)).Named("wh");

            VariableArray<double> lognoisePrecision = Variable.Array<double>(rows);
            //lognoisePrecision[rows] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[rows], wh), 1.0);
            lognoisePrecision[rows] = Variable.InnerProduct(x[rows], wh);
            VariableArray<double> noisePrecision = Variable.Array<double>(rows);
            noisePrecision[rows] = Variable<double>.Factor(System.Math.Exp, lognoisePrecision[rows]);
            VariableArray<double> y = Variable.Array<double>(rows);
            //y[rows] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[rows], w), noisePrecision[rows]);
            y[rows] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(x[rows], w), noisePrecision[rows]);
            y.ObservedValue = yobs;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            //InferenceEngine engine = new InferenceEngine();
            VectorGaussian postW = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Posterior over the weights: " + Environment.NewLine + postW);

            VectorGaussian postWH = engine.Infer<VectorGaussian>(wh);
            Console.WriteLine("Posterior over the hetero weights: " + Environment.NewLine + postWH);

            var pred = new Gaussian[n];
            for (int i = 0; i < n; i++)
            {
                pred[i] = GaussianOp.SampleAverageLogarithm(
                    InnerProductOp.InnerProductAverageLogarithm((DenseVector)postW.GetMean(), postW.GetVariance(), data[i]),
                    ExpOp.ExpAverageLogarithm(
                        InnerProductOp.InnerProductAverageLogarithm((DenseVector)postWH.GetMean(), postWH.GetVariance(), data[i])));
            }
        }

        // not the same model as above, but fully conjugate
        internal void HeteroRegressionTest2()
        {
            var data = Enumerable.Range(0, 40).Select(o => Vector.FromArray(1.0, ((double) o - 20.0)/10.0)).ToArray();
            var wTrue = Vector.FromArray(2.0, 1.0);
            var whTrue = Vector.FromArray(0.0, -1.0);
            int n = data.Length;
            var yobs = new double[n];
            for (int i = 0; i < n; i++)
            {
                yobs[i] = data[i].Inner(wTrue) + Rand.Normal()* System.Math.Exp(-0.5* whTrue.Inner(data[i]));
                //Console.WriteLine(data[i][1] + " " + yobs[i]);
            }

            Range rows = new Range(n);

            VariableArray<Vector> x = Variable.Constant(data, rows).Named("x");

            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(new double[] {0, 0}),
                PositiveDefiniteMatrix.Identity(2)).Named("w");

            Variable<Vector> wh = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(new double[] {1, 0}),
                PositiveDefiniteMatrix.Identity(2)).Named("wh");

            VariableArray<double> heteroNoiseScale = Variable.Array<double>(rows);
            //lognoisePrecision[rows] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[rows], wh), 1.0);
            heteroNoiseScale[rows] = Variable.InnerProduct(x[rows], wh);
            var heteroNoise = Variable.Array<double>(rows);
            heteroNoise[rows] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(rows);
            // Variable.ConstrainPositive(heteroNoise[rows]);
            VariableArray<double> y = Variable.Array<double>(rows);
            double fixedNoisePrec = 10.0;
            y[rows] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(x[rows], w) +
                                                            heteroNoiseScale[rows]*heteroNoise[rows], fixedNoisePrec);
            y.ObservedValue = yobs;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            //InferenceEngine engine = new InferenceEngine();

            //wh.InitialiseTo(VectorGaussian.FromMeanAndPrecision(Vector.FromArray(1,0),PositiveDefiniteMatrix.Identity(2))); 

            VectorGaussian postW = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Posterior over the weights: " + Environment.NewLine + postW);

            VectorGaussian postWH = engine.Infer<VectorGaussian>(wh);
            Console.WriteLine("Posterior over the hetero weights: " + Environment.NewLine + postWH);

            var pred = new Gaussian[n];
            for (int i = 0; i < n; i++)
            {
                pred[i] = GaussianOp.SampleAverageConditional(
                    DoublePlusVmpOp.SumAverageLogarithm(
                        InnerProductOp.InnerProductAverageLogarithm((DenseVector)postW.GetMean(), postW.GetVariance(), data[i]),
                        GaussianProductVmpOp.ProductAverageLogarithm(
                            InnerProductOp.InnerProductAverageLogarithm((DenseVector)postWH.GetMean(), postWH.GetVariance(), data[i]),
                            Gaussian.FromMeanAndPrecision(0, 1))),
                    fixedNoisePrec);
            }
        }

        public double[] GammaFitMC(Gamma shapePrior, Gamma ratePrior, double[] data)
        {
            Rand.Restart(21983);
            Converter<double[], double> f =
                x => shapePrior.GetLogProb(x[0]) + ratePrior.GetLogProb(x[1]) + data.Select(o => Gamma.FromShapeAndRate(x[0], x[1]).GetLogProb(o)).Sum();
            var samples = SliceSamplingHelper(f, new double[] {1, 1}, lower_bound: new double[] {0, 0},
                                              upper_bound: new double[] {double.PositiveInfinity, double.PositiveInfinity});
            var means = samples.Aggregate((p, q) => new double[] {p[0] + q[0], p[1] + q[1]});
            means[0] /= (double) samples.Length;
            means[1] /= (double) samples.Length;
            var variances = samples.Aggregate((p, q) => new double[] {p[0] + q[0]*q[0], p[1] + q[1]*q[1]});
            variances[0] /= (double) samples.Length;
            variances[1] /= (double) samples.Length;
            variances[0] -= means[0]*means[0];
            variances[1] -= means[1]*means[1];
            variances[0] = System.Math.Sqrt(variances[0]);
            variances[1] = System.Math.Sqrt(variances[1]);
            double cross = samples.Aggregate(0.0, (p, q) => p + q[0]*q[1])/(double) samples.Length - means[0]*means[1];
            double corr = cross/(variances[0]*variances[1]);

            return new double[] {means[0], variances[0]};
        }

        private static double studentTlogProb(double x, double a, double b, double m)
        {
            double diff = x - m;
            return a* System.Math.Log(b) - (a + .5)* System.Math.Log(b + .5* diff * diff) - .5* System.Math.Log(2* System.Math.PI) + MMath.GammaLn(a + .5) - MMath.GammaLn(a);
        }

        public double[] StudentTFitMC(Gamma shapePrior, Gamma ratePrior, Gaussian meanPrior, double[] data)
        {
            Rand.Restart(21983);
            Converter<double[], double> f =
                x => shapePrior.GetLogProb(x[0]) + ratePrior.GetLogProb(x[1]) + meanPrior.GetLogProb(x[2]) + data.Select(o => studentTlogProb(o, x[0], x[1], x[2])).Sum();
            var samples = SliceSamplingHelper(f, new double[] {1, 1, 0}, lower_bound: new double[] {0, 0, double.NegativeInfinity},
                                              upper_bound: new double[] {double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity}, burnin: 10000);
            var vecSamples = samples.Select(o => Vector.FromArray(o)).ToArray();
            var means = vecSamples.Aggregate((p, q) => p + q);
            means.Scale(1.0/samples.Length);
            var variances = vecSamples.Aggregate((p, q) => p + q*q);
            variances.Scale(1.0/samples.Length);
            variances -= means*means;
            variances.SetToFunction(variances, System.Math.Sqrt);

            return new double[] {means[0], variances[0]};
        }

        internal void GammaFitVaryS()
        {
            var srange = new double[] {0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0};
            var res = new Gamma[srange.Length];
            var resTrue = new double[srange.Length][];
            double trueShape = 5.0, trueRate = 1.0;
            var trueGamma = Gamma.FromShapeAndRate(trueShape, trueRate);
            int N = 20;
            var data = Enumerable.Range(0, N).Select(_ => trueGamma.Sample()).ToArray();
            var shape = Variable.GammaFromShapeAndRate(1, 1).Named("shape");
            var rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            var n = new Range(N).Named("n");
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GammaFromShapeAndRate(shape, rate).ForEach(n);
            x.ObservedValue = data;
            //GammaFitMC(Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), data); 
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            for (int i = 0; i < srange.Length; i++)
            {
                trueGamma = Gamma.FromShapeAndRate(srange[i], trueRate);
                x.ObservedValue = Enumerable.Range(0, N).Select(_ => trueGamma.Sample()).ToArray();
                resTrue[i] = GammaFitMC(Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), x.ObservedValue);
                res[i] = ie.Infer<Gamma>(shape);
                Console.WriteLine(srange[i] + " " + res[i].GetMean() + " " + System.Math.Sqrt(res[i].GetVariance()));
            }
        }

        internal void StudentTFitVaryS()
        {
            var srange = new double[] {0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0};
            var res = new Gamma[srange.Length];
            var resTrue = new double[srange.Length][];
            double trueShape = 5.0, trueRate = 1.0;
            var trueGamma = Gamma.FromShapeAndRate(trueShape, trueRate);
            int N = 100;
            var data = Enumerable.Range(0, N).Select(_ => Gaussian.FromMeanAndPrecision(0, trueGamma.Sample()).Sample()).ToArray();
            var shape = Variable.GammaFromShapeAndRate(1, 1).Named("shape");
            var rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1).Named("mean");
            var n = new Range(N).Named("n");
            var precs = Variable.Array<double>(n).Named("precs");
            precs[n] = Variable.GammaFromShapeAndRate(shape, rate).ForEach(n);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(mean, precs[n]);
            x.ObservedValue = data;
            //GammaFitMC(Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), data); 
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 300;
            for (int i = 0; i < srange.Length; i++)
            {
                trueGamma = Gamma.FromShapeAndRate(srange[i], trueRate);
                x.ObservedValue = Enumerable.Range(0, N).Select(_ => Gaussian.FromMeanAndPrecision(0, trueGamma.Sample()).Sample()).ToArray();
                resTrue[i] = StudentTFitMC(Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), Gaussian.FromMeanAndPrecision(0, 1), x.ObservedValue);
                res[i] = ie.Infer<Gamma>(shape);
                Console.WriteLine(srange[i] + " " + res[i].GetMean() + " " + System.Math.Sqrt(res[i].GetVariance()) + " " + resTrue[i][0] + " " + System.Math.Sqrt(resTrue[i][1]));
            }
        }

        // test accuracy of learning Gamma shape with VMP
        internal void GammaFit()
        {
            var nrange = new int[] {20, 50, 100, 150, 200, 250};
            var res = new Gamma[nrange.Length];
            var resTrue = new double[nrange.Length][];
            double trueShape = 5.0, trueRate = 1.0;
            var trueGamma = Gamma.FromShapeAndRate(trueShape, trueRate);
            int N = 50;
            var Nvar = Variable.Observed(N);
            var data = Enumerable.Range(0, N).Select(_ => trueGamma.Sample()).ToArray();
            var shape = Variable.GammaFromShapeAndRate(1, 1).Named("shape");
            var rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            var n = new Range(Nvar).Named("n");
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GammaFromShapeAndRate(shape, rate).ForEach(n);
            x.ObservedValue = data;
            //GammaFitMC(Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), data); 
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            for (int i = 0; i < nrange.Length; i++)
            {
                Rand.Restart(1);
                Nvar.ObservedValue = nrange[i];
                x.ObservedValue = Enumerable.Range(0, nrange[i]).Select(_ => trueGamma.Sample()).ToArray();

                resTrue[i] = GammaFitMC(Gamma.FromShapeAndRate(1, 1), Gamma.FromShapeAndRate(1, 1), x.ObservedValue);
                res[i] = ie.Infer<Gamma>(shape);
                Console.WriteLine("{0} actual = {1} expected = {2}", nrange[i], res[i], resTrue[i][0]);
            }
        }

        [Fact]
        public void GammaShapeTest()
        {
            double[] xdata = new double[] {1, 2, 3};

            Gamma shapePrior = Gamma.FromShapeAndRate(2, 3);
            var shape = Variable.Random(shapePrior).Named("shape");
            var rate = 1.0;
            var tau = Variable.GammaFromShapeAndRate(shape, rate).Named("tau");
            var mean = 2.0;
            var n = new Range(xdata.Length).Named("n");
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(mean, tau).ForEach(n);
            x.ObservedValue = xdata;

            // compute the exact moments of shape
            Gamma tau_B = Gamma.Uniform();
            for (int i = 0; i < xdata.Length; i++)
            {
                tau_B.SetToProduct(tau_B, GaussianOp.PrecisionAverageLogarithm(xdata[i], mean));
            }
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            for (int i = 0; i < 1000; i++)
            {
                double s = 1e-2 + i*1e-2;
                Gamma tau_F = Gamma.FromShapeAndRate(s, rate);
                double logProb = tau_F.GetLogAverageOf(tau_B) + shapePrior.GetLogProb(s);
                mva.Add(s, System.Math.Exp(logProb));
            }
            Gamma shapeExpected = Gamma.FromMeanAndVariance(mva.Mean, mva.Variance);

            var engine = new InferenceEngine(new VariationalMessagePassing());
            Gamma shapeActual = engine.Infer<Gamma>(shape);
            Console.WriteLine("shape = {0} should be {1}", shapeActual, shapeExpected);
            Assert.True(shapeExpected.MaxDiff(shapeActual) < 0.5);
        }

        internal void StudentTTruncated()
        {
            var prec = Variable.GammaFromShapeAndRate(1, 1);
            var x = Variable.GaussianFromMeanAndPrecision(0, prec);
            Variable.ConstrainPositive(x);
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(x));
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 429
#endif

        [Fact]
        public void LogisticRegressionTest()
        {
            int P = 8, N = 30;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(P).Named("dim");

            // weight vector
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.GaussianFromMeanAndPrecision(1.2, 0.4).ForEach(dim);
            Range item = new Range(N).Named("item");

            // Covariates
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(2.0, 0.5).ForEach(item, dim);

            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = w[dim]*x[item][dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);

            // Observed labels
            var y = Variable.Array<bool>(item).Named("y");
            y[item] = Variable.BernoulliFromLogOdds(sum[item]);
            block.CloseBlock();

            InferenceEngine ieJJ = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine ieSJ = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine ieKL = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine ieEP = new InferenceEngine(new ExpectationPropagation());
            int iterations = 100;
            ieEP.ShowProgress = false;
            ieEP.ShowTimings = true;
            ieEP.NumberOfIterations = iterations;
            ieJJ.Compiler.GivePriorityTo(typeof (LogisticOp_JJ96));
            ieJJ.ShowProgress = false;
            ieJJ.ShowTimings = true;
            ieJJ.NumberOfIterations = iterations;
            ieSJ.Compiler.GivePriorityTo(typeof (LogisticOp_SJ99));
            ieSJ.ShowProgress = false;
            ieSJ.ShowTimings = true;
            ieSJ.NumberOfIterations = iterations;
            // Fails with unrolling because FastSumOp.ArrayAverageLogarithm assumes the array is updated in parallel
            ieSJ.Compiler.UnrollLoops = false;
            ieKL.Compiler.GivePriorityTo(typeof (LogisticOp));
            ieKL.ShowProgress = false;
            ieKL.ShowTimings = true;
            ieKL.NumberOfIterations = iterations;
            int repeats = 10;
            double[] resultsJaakkola = new double[10];
            double[] resultsSaul = new double[10];
            double[] resultsKL = new double[10];
            double[] resultsEP = new double[10];
            double[] evidenceJaakkola = new double[10];
            double[] evidenceSaul = new double[10];
            double[] evidenceKL = new double[10];
            double[] evidenceEP = new double[10];
            for (int seed = 0; seed < repeats; seed++)
            {
                // Set random set
                Rand.Restart(seed);
                double[][] xTrue = new double[N][]; // data drawn from the model
                double[] wTrue = new double[P]; // "true" weights
                bool[] yObs = new bool[N]; // observed data
                double[] g = new double[N];
                for (int i = 0; i < P; i++)
                    wTrue[i] = Rand.Normal(0, 1);

                // sample data from the model
                for (int j = 0; j < N; j++)
                {
                    xTrue[j] = new double[P];
                    g[j] = 0;
                    for (int i = 0; i < P; i++)
                    {
                        double mean = Rand.Normal(0, 1);
                        double std = Rand.Gamma(1);
                        xTrue[j][i] = Rand.Normal(mean, std);
                        g[j] += xTrue[j][i]*wTrue[i];
                    }
                    yObs[j] = (Rand.Binomial(1, MMath.Logistic(g[j])) == 1);
                }

                // set observed values
                y.ObservedValue = yObs;
                x.ObservedValue = xTrue;

                double G_mean, G_var;
                DistributionArray<Gaussian> G;
                // run inference
                Console.WriteLine("Jaakkola");
                G = ieJJ.Infer<DistributionArray<Gaussian>>(w);
                evidenceJaakkola[seed] = ieJJ.Infer<Bernoulli>(evidence).LogOdds;
                resultsJaakkola[seed] = 0.0;
                for (int i = 0; i < P; i++)
                {
                    G[i].GetMeanAndVariance(out G_mean, out G_var);
                    resultsJaakkola[seed] += G[i].GetLogProb(wTrue[i]);
                    //Console.WriteLine("True w: " + wTrue[i] + " inferred: " + G_mean + " +/- " + Math.Sqrt(G_var));
                }
                //Console.WriteLine("log P(W_true | D) = " + resultsJaakkola[seed]);

                Console.WriteLine("Saul");
                G = ieSJ.Infer<DistributionArray<Gaussian>>(w);
                evidenceSaul[seed] = ieSJ.Infer<Bernoulli>(evidence).LogOdds;
                resultsSaul[seed] = 0.0;
                for (int i = 0; i < P; i++)
                {
                    G[i].GetMeanAndVariance(out G_mean, out G_var);
                    resultsSaul[seed] += G[i].GetLogProb(wTrue[i]);
                    //Console.WriteLine("True w: " + wTrue[i] + " inferred: " + G_mean + " +/- " + Math.Sqrt(G_var));
                }
                //Console.WriteLine("log P(W_true | D) = " + resultsSaul[seed]);

                Console.WriteLine("KL");
                G = ieKL.Infer<DistributionArray<Gaussian>>(w);
                evidenceKL[seed] = ieKL.Infer<Bernoulli>(evidence).LogOdds;
                resultsKL[seed] = 0.0;
                for (int i = 0; i < P; i++)
                {
                    G[i].GetMeanAndVariance(out G_mean, out G_var);
                    resultsKL[seed] += G[i].GetLogProb(wTrue[i]);
                    //Console.WriteLine("True w: " + wTrue[i] + " inferred: " + G_mean + " +/- " + Math.Sqrt(G_var));
                }
                //Console.WriteLine("log P(W_true | D) = " + resultsKL[seed]);

                Console.WriteLine("EP");
                G = ieEP.Infer<DistributionArray<Gaussian>>(w);
                evidenceEP[seed] = ieEP.Infer<Bernoulli>(evidence).LogOdds;
                resultsEP[seed] = 0.0;
                for (int i = 0; i < P; i++)
                {
                    G[i].GetMeanAndVariance(out G_mean, out G_var);
                    resultsEP[seed] += G[i].GetLogProb(wTrue[i]);
                    //Console.WriteLine("True w: " + wTrue[i] + " inferred: " + G_mean + " +/- " + Math.Sqrt(G_var));
                }
                //Console.WriteLine("log P(W_true | D) = " + resultsEP[seed]);
            }
            Console.WriteLine("log P(W_true | D)");
            Console.WriteLine("Jaakk:\t\tSaul:\t\tKL:\t\tEP:\t\tWinner");
            for (int seed = 0; seed < repeats; seed++)
            {
                Console.Write("{0:F}\t\t{1:F}\t\t{2:F}\t\t{3:F}\t\t", resultsJaakkola[seed], resultsSaul[seed], resultsKL[seed], resultsEP[seed]);
                string best = "KL(q||p)";
                double bestResult = resultsKL[seed];
                if (resultsJaakkola[seed] > bestResult)
                {
                    best = "Jaakkola";
                    bestResult = resultsJaakkola[seed];
                }
                if (resultsSaul[seed] > bestResult)
                {
                    best = "Saul";
                    bestResult = resultsSaul[seed];
                }
                if (resultsEP[seed] > bestResult)
                {
                    best = "EP";
                    bestResult = resultsEP[seed];
                }
                Console.WriteLine(best);
            }
            Console.WriteLine("Evidence");
            Console.WriteLine("Jaakk:\t\tSaul:\t\tKL:\t\tEP:\t\tWinner");
            for (int seed = 0; seed < repeats; seed++)
            {
                Console.Write("{0:F}\t\t{1:F}\t\t{2:F}\t\t{3:F}\t\t", evidenceJaakkola[seed], evidenceSaul[seed], evidenceKL[seed], evidenceEP[seed]);
                string best = "KL(q||p)";
                double bestResult = evidenceKL[seed];
                if (evidenceJaakkola[seed] > bestResult)
                {
                    best = "Jaakkola";
                    bestResult = evidenceJaakkola[seed];
                }
                if (evidenceSaul[seed] > bestResult)
                {
                    best = "Saul";
                    bestResult = evidenceSaul[seed];
                }
                if (false && evidenceEP[seed] > bestResult)
                {
                    best = "EP";
                    bestResult = evidenceEP[seed];
                }
                Console.WriteLine(best);
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 429
#endif

        //[Fact]
        internal void SpikeAndSlabTest()
        {
            int P = 8;

            Range dim = new Range(P).Named("dim");

            // weight vector
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            VariableArray<bool> probOn = Variable.Array<bool>(dim).Named("p");
            var noisyW = Variable.Array<double>(dim);
            using (Variable.ForEach(dim))
            {
                probOn[dim] = Variable.Bernoulli(0.5);
                using (Variable.If(probOn[dim]))
                {
                    w[dim] = Variable.GaussianFromMeanAndPrecision(0, 1);
                }
                using (Variable.IfNot(probOn[dim]))
                {
                    w[dim] = Variable.GaussianFromMeanAndPrecision(0, 10000 /*double.PositiveInfinity*/);
                }
            }
            noisyW[dim] = Variable.GaussianFromMeanAndPrecision(w[dim], 10);
            noisyW.ObservedValue = Enumerable.Range(0, P).Select(_ => Rand.Double() > .5 ? Rand.Normal() : 0.0).ToArray();

            var vmp = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(vmp.Infer(probOn));
        }


        internal void TestExp0()
        {
            var x = Variable.GaussianFromMeanAndVariance(0, 100).Named("x");
            var y = Variable.Exp(x).Named("y");
            Variable.ConstrainEqualRandom(y, new Gamma(1, 1));
            var ie = new InferenceEngine(new VariationalMessagePassing());
            // Gaussian(-4.151, 8.638e-08)
            Console.WriteLine(ie.Infer(x));
            var b = new BernoulliEstimator();
            b.NProbTrue = 10;
            b.N = 100;
            b.GetDistribution(new Bernoulli());
        }

        //[Fact]
        internal void TestExp()
        {
            var x = Variable.GaussianFromMeanAndPrecision(0, .1).Named("x");
            var g = Variable.Exp(x).Named("g");
            //x.AddAttribute(new LikelihoodPrototype(new NonconjugateGaussian()));
            var y = Variable.Poisson(g).Named("y");
            y.ObservedValue = 1;
            var engine = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(engine.Infer(x));
        }

        //[Fact]
        internal void TestExpVMP()
        {
            var evidence = Variable.Bernoulli(.5).Named("evidence");
            Variable<double> x, g;
            Variable<int> y;
            using (Variable.If(evidence))
            {
                x = Variable.GaussianFromMeanAndPrecision(0, .1).Named("x");
                g = Variable.Exp(x).Named("g");
                y = Variable.Poisson(g).Named("y");
                y.ObservedValue = 0;
            }
            var engine = new InferenceEngine(new VariationalMessagePassing());
            // EP: Gaussian(-0.1193, 0.4993)
            // VMP damping .1: Gaussian(-0.1229, 0.4729)  -1.14115770200788
            // VMP damping 1: Gaussian(-0.1213, 0.4714)   -23145482.8394575
            // Double loop VMP  -23145482.8394571
            engine.ShowTimings = true;
            Console.WriteLine("With VMP and BFGS: " + engine.Infer(x));
            engine = new InferenceEngine(new ExpectationPropagation());
            engine.ShowTimings = true;
            Console.WriteLine("With EP: " + engine.Infer(x));

            var x3 = Variable.GaussianFromMeanAndPrecision(0, .1);
            var x2 = Variable.Copy(x3);
            x2.AddAttribute(new MarginalPrototype(new NonconjugateGaussian()));
            var g2 = Variable.Exp(x2).Named("g");
            var y2 = Variable.Poisson(g2).Named("y");
            y2.ObservedValue = 0;
            engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowTimings = true;
            Console.WriteLine("With NCVMP: " + engine.Infer(x3));
            engine.ShowProgress = false;
            for (int i = 0; i < 100; i++)
            {
                engine.Algorithm = new VariationalMessagePassing();
                engine.NumberOfIterations = 10*(i + 1);
                Console.WriteLine(10*(i + 1) + " iterations, x=" + engine.Infer(x) + " logEv=" + engine.Infer<Bernoulli>(evidence).LogOdds);
            }
        }
    }
}
