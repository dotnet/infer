// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class NonconjugateVmpTests
    {
        // Test BernoulliFromLogOdds using direct KL minimisation 
        [Fact]
        public void BernoulliFromLogOddsUnivariate()
        {
            Rand.Restart(12347);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            Variable<bool> s = Variable.BernoulliFromLogOdds(x).Named("s");
            s.ObservedValue = true;
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof (LogisticOp));
            ie.Compiler.GivePriorityTo(typeof (BernoulliFromLogOddsOp));
            Gaussian xActual = ie.Infer<Gaussian>(x);
            Console.WriteLine("x = {0}", xActual);
            double m, v;
            xActual.GetMeanAndVariance(out m, out v);
            double matlabM = 0.413126805979683;
            double matlabV = 0.828868291887001;
            Gaussian xExpected = new Gaussian(matlabM, matlabV);
            double relErr = System.Math.Abs((m - matlabM)/ matlabM);
            Console.WriteLine("Posterior mean is {0} should be {1} (relErr = {2})", m, matlabM, relErr);
            Assert.True(relErr < 1e-6);
            relErr = System.Math.Abs((v - matlabV)/ matlabV);
            Console.WriteLine("Posterior variance is {0} should be {1} (relErr = {2})", v, matlabV, relErr);
            Assert.True(relErr < 1e-6);
        }


        [Fact]
        public void BetaFromMeanAndTotalCountTest()
        {
            var meanExpected = new Beta(4.673119273418562, 2.781360122279706);
            Variable<double> mean = Variable.Beta(1, 1).Named("mean");
            mean.InitialiseTo(meanExpected);
            Variable<double> totalCount = Variable.Constant<double>(5).Named("totalCount");
            Variable<double> p = BetaFromMeanAndTotalCount(mean, totalCount).Named("p");
            p.ObservedValue = 0.7;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("mean = {0}", engine.Infer(mean));
            }
            Beta meanActual = engine.Infer<Beta>(mean);
            Console.WriteLine("mean = {0} (should be {1})", meanActual, meanExpected);
            Assert.True(meanExpected.MaxDiff(meanActual) < 2e-4);
        }

        // Should be moved into Variable.cs once fully tested. 
        /// <summary>
        /// Creates a Beta-distributed random variable from its mean and totalCount parameters.
        /// </summary>
        /// <param name="mean">The mean parameter of the Beta distribution</param>
        /// <param name="totalCount">The totalCount parameter of the Beta distribution</param>
        /// <returns>Beta-distributed random variable</returns>
        public static Variable<double> BetaFromMeanAndTotalCount(Variable<double> mean, Variable<double> totalCount)
        {
            return Variable<double>.Factor(Factor.BetaFromMeanAndTotalCount, mean, totalCount);
        }

        [Fact]
        public void DirichletFromMeanAndTotalCountTest()
        {
            var meanExpected = new Dirichlet(4.673119273418562, 2.781360122279706);
            Variable<Vector> mean = Variable.Dirichlet(new double[] {1, 1}).Named("mean");
            Variable<double> totalCount = Variable.Constant<double>(5).Named("totalCount");
            Variable<Vector> p = DirichletFromMeanAndTotalCount(mean, totalCount, 2).Named("p");
            p.ObservedValue = Vector.FromArray(new double[] {0.7, 0.3});
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;
            for (int iter = 1; iter < 20; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine(engine.Infer(mean));
            }
            Dirichlet meanActual = engine.Infer<Dirichlet>(mean);
            Console.WriteLine(StringUtil.JoinColumns("mean = ", meanActual, " should be ", meanExpected));
            Assert.True(meanExpected.MaxDiff(meanActual) < 3e-4);
        }

        // Fails if N is too large.  Probably due to inaccuracy in the quadrature when incoming mean is sharp.
        internal void DirichletFactorTest2()
        {
            Rand.Restart(0);
            int N = 10;
            var truth = new Dirichlet(5, 2, 1);
            var data = Util.ArrayInit(N, i => truth.Sample());

            Variable<Vector> mean = Variable.Dirichlet(new double[] {1, 1, 1}).Named("mean");
            mean.InitialiseTo(truth);
            //mean.ObservedValue = truth.GetMean();
            //Variable<Vector> mean = Variable.Constant<Vector>(Vector.FromArray(new double[] { 5, 2, 1 })).Named("mean");
            Variable<double> totalCount = Variable.GammaFromShapeAndRate(1, 1).Named("totalCount");
            totalCount.ObservedValue = truth.TotalCount;
            Range samples = new Range(N).Named("N");
            var p = Variable.Array<Vector>(samples).Named("p");
            p[samples] = DirichletFromMeanAndTotalCount(mean, totalCount, 3).ForEach(samples);
            p.ObservedValue = data;
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;
            for (int iter = 1; iter < 30; iter++)
            {
                engine.NumberOfIterations = iter;
                Console.WriteLine("[{0}] {1}", iter, engine.Infer(mean));
            }
            Dirichlet meanPosterior = engine.Infer<Dirichlet>(mean);
            var meanMean = meanPosterior.GetMean();
            var truthMean = truth.GetMean();
            for (int i = 0; i < 3; i++)
                Console.WriteLine("Mean[{0}]={1:g4}, should be {2:g4}",
                                  i, meanMean[i], truthMean[i]);
            Gamma totalCountPosterior = engine.Infer<Gamma>(totalCount);
            Console.WriteLine("Totalcount posterior=" + totalCountPosterior + " should be " + truth.TotalCount);
        }


        [Fact]
        public void DirichletOpQuadratureTest()
        {
            var matlabResults
                = new double[]
                    {
                        0.625872049875551,
                        0.866057568760984,
                        -0.266065360660541,
                        -1.227320719860393,
                        1.280900246404125
                    };
            var matlabResults2
                = new double[]
                    {
                        0.843302107208523,
                        0.610546297106219,
                        -2.182481855300747,
                        -0.254011377373013,
                        -0.217430057568389
                    };
            double am = 2;
            double bm = 1;
            double at = 3;
            double bt = 1;
            Dirichlet meanQ = new Dirichlet(new double[] {am, bm});
            Gamma totalCountQ = new Gamma(at, 1/bt);
            double[] EELogGamma;
            double[] EELogMLogGamma;
            double[] EELogOneMinusMLogGamma;
            double[] EELogSLogGamma;
            double[] EEMSDigamma;
            DirichletOp.MeanMessageExpectations(
                meanQ.PseudoCount,
                totalCountQ,
                out EELogGamma,
                out EELogMLogGamma,
                out EELogOneMinusMLogGamma);

            Console.WriteLine(System.Math.Abs(EELogGamma[0] - matlabResults[0]));
            Console.WriteLine(System.Math.Abs(EELogMLogGamma[0] - matlabResults[2]));
            Console.WriteLine(System.Math.Abs(EELogOneMinusMLogGamma[0] - matlabResults[3]));

            Console.WriteLine(System.Math.Abs(EELogGamma[1] - matlabResults2[0]));
            Console.WriteLine(System.Math.Abs(EELogMLogGamma[1] - matlabResults2[2]));
            Console.WriteLine(System.Math.Abs(EELogOneMinusMLogGamma[1] - matlabResults2[3]));

            DirichletOp.TotalCountMessageExpectations(
                meanQ.PseudoCount,
                totalCountQ,
                out EELogGamma,
                out EELogSLogGamma,
                out EEMSDigamma);

            Console.WriteLine(System.Math.Abs(EELogGamma[0] - matlabResults[0]));
            Console.WriteLine(System.Math.Abs(EELogSLogGamma[0] - matlabResults[1]));
            Console.WriteLine(System.Math.Abs(EEMSDigamma[0] - matlabResults[4]));

            Console.WriteLine(System.Math.Abs(EELogGamma[1] - matlabResults2[0]));
            Console.WriteLine(System.Math.Abs(EELogSLogGamma[1] - matlabResults2[1]));
            Console.WriteLine(System.Math.Abs(EEMSDigamma[1] - matlabResults2[4]));
        }

        // Should be moved into Variable.cs once fully tested. 
        /// <summary>
        /// Creates a Dirichlet-distributed random variable from its mean and totalCount parameters.
        /// </summary>
        /// <param name="mean">The mean parameter of the Dirichlet distribution</param>
        /// <param name="totalCount">The totalCount parameter of the Dirichlet distribution</param>
        /// <returns>Beta-distributed random variable</returns>
        public static Variable<Vector> DirichletFromMeanAndTotalCount(
            Variable<Vector> mean,
            Variable<double> totalCount,
            int K)
        {
            return Variable<Vector>.Factor(Factor.DirichletFromMeanAndTotalCount, mean, totalCount)
                                   .Attrib(new MarginalPrototype(Dirichlet.Uniform(K)));
        }

        internal void LogTest()
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            Variable<double> x;
            using (Variable.If(evidence))
            {
                x = Variable.GammaFromShapeAndRate(1, 1).Named("x");
                var y = Variable.Log(x).Named("y");
                Variable.ConstrainEqualRandom(y, new Gaussian(2, 0.1));
            }
            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(x));
            Console.WriteLine(ie.Infer<Bernoulli>(evidence).LogOdds);
            ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(x));
            Console.WriteLine(ie.Infer<Bernoulli>(evidence).LogOdds);
        }


        [Fact]
        public void PoissonExpTest2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> x = Variable.GaussianFromMeanAndVariance(1.2, 3.4).Named("x");

            Rand.Restart(12347);
            int n = 10;
            int N = 10;
            int[] data = new int[N];
            for (int i = 0; i < N; i++)
                data[i] = Rand.Binomial(n, 1.0/(double) n);
            data = new int[] {5, 6, 7};
            Range item = new Range(data.Length).Named("item");
            VariableArray<double> ex = Variable.Array<double>(item).Named("ex");
            ex[item] = Variable.Exp(x).ForEach(item);
            VariableArray<int> y = Variable.Array<int>(item).Named("y");
            y[item] = Variable.Poisson(ex[item]);
            block.CloseBlock();
            y.ObservedValue = data;

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            var ca = ie.GetCompiledInferenceAlgorithm(evidence, x);

            double oldLogEvidence = double.NegativeInfinity;
            for (int i = 0; i < 1000; i++)
            {
                ca.Update(1);
                double logEvidence1 = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;
                Console.WriteLine(logEvidence1);
                if (i > 20 && System.Math.Abs(logEvidence1 - oldLogEvidence) < 1e-10)
                    break;
                oldLogEvidence = logEvidence1;
            }
            Gaussian xExpected = new Gaussian(1.755071011884509, 0.055154577283323);
            Gaussian xActual = ca.Marginal<Gaussian>(x.NameInGeneratedCode);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-3);
        }


        internal void FitNegativeBinomial()
        {
            // generate data from the model
            double r = 2;
            double p = 0.3;
            int[] data = new int[] {1, 4, 5, 14, 0, 3, 2, 18, 0, 1, 8, 1, 4, 3, 6, 4, 9, 5, 1, 10, 5, 9, 2, 3, 3, 9, 14, 3, 5, 12};
            int N = data.Length;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var rate = Variable.GammaFromShapeAndRate(1, 1).Named("rate");
            //var shape = Variable.GammaFromShapeAndRate(1,1).Named("shape");
            var shape = Variable.GammaFromShapeAndRate(1, 1).Named("shape");
            var lambda = Variable.GammaFromShapeAndRate(shape, rate).Named("lambda");
            Range Nrange = new Range(N);
            var y = Variable.Array<int>(Nrange).Named("y");
            y[Nrange] = Variable.Poisson(lambda).ForEach(Nrange);
            y.ObservedValue = data;
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowFactorGraph = true;
            var ca = ie.GetCompiledInferenceAlgorithm(evidence, rate, shape);
            ca.Reset();
            double oldLogEvidence = double.NegativeInfinity;
            for (int i = 0; i < 1000; i++)
            {
                ca.Update(1);
                double logEvidence1 = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;
                Console.WriteLine(logEvidence1);
                if (i > 20 && System.Math.Abs(logEvidence1 - oldLogEvidence) < 0.01)
                    break;
                oldLogEvidence = logEvidence1;
            }
            Gamma shapePost = ca.Marginal<Gamma>(shape.NameInGeneratedCode);
            Gamma ratePost = ca.Marginal<Gamma>(rate.NameInGeneratedCode);
            double mean, variance;
            shapePost.GetMeanAndVariance(out mean, out variance);
            Console.WriteLine("shape = " + mean + " +/- " + System.Math.Sqrt(variance) + " true= " + r);
            ratePost.GetMeanAndVariance(out mean, out variance);
            Console.WriteLine("rate = " + mean + " +/- " + System.Math.Sqrt(variance) + " true= " + p/(1 - p));
        }


        internal void FitNegativeBinomialExpGaussian()
        {
            // generate data from the model
            double r = 2;
            double p = 0.3;
            int[] data = new int[] {1, 4, 5, 14, 0, 3, 2, 18, 0, 1, 8, 1, 4, 3, 6, 4, 9, 5, 1, 10, 5, 9, 2, 3, 3, 9, 14, 3, 5, 12};
            int N = data.Length;

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1);
            var prec = Variable.GammaFromShapeAndRate(1, 1);
            var logLambda = Variable.GaussianFromMeanAndPrecision(mean, prec).Named("logLambda");
            var lambda = Variable.Exp(logLambda).Named("lambda");
            Range Nrange = new Range(N);
            var y = Variable.Array<int>(Nrange).Named("y");
            y[Nrange] = Variable.Poisson(lambda).ForEach(Nrange);
            y.ObservedValue = data;
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowFactorGraph = true;
            var ca = ie.GetCompiledInferenceAlgorithm(evidence, lambda);
            ca.Reset();

            double oldLogEvidence = double.NegativeInfinity;
            for (int i = 0; i < 1000; i++)
            {
                ca.Update(1);
                double logEvidence1 = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;
                Console.WriteLine(logEvidence1);
                if (i > 20 && System.Math.Abs(logEvidence1 - oldLogEvidence) < 0.01)
                    break;
                oldLogEvidence = logEvidence1;
            }
            Gamma lambdaPost = ca.Marginal<Gamma>(lambda.NameInGeneratedCode);
            Console.WriteLine("shape = " + lambdaPost.Shape + " true= " + r);
            Console.WriteLine("rate = " + lambdaPost.Rate + " true= " + p/(1 - p));
        }


        //[Fact]
        internal void GaussianTimesBetaTest2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            //Variable<double> x = Variable.GaussianFromMeanAndVariance(1.2, 3.4).Named("x");
            //var x = Variable.Constant<double>(1.2).Named("x"); 
            var s = Variable.Beta(5.6, 4.8).Named("s");
            //var s = Variable.GaussianFromMeanAndPrecision(0, 1).Named("s"); 
            Variable<double> y = 1.2*s;
            y.Name = "y";
            Variable.ConstrainEqualRandom(y, new Gaussian(2.7, 0.001));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var ca = engine.GetCompiledInferenceAlgorithm(evidence, s);
            ca.Reset();

            double oldLogEvidence = double.NegativeInfinity;
            for (int i = 0; i < 1000; i++)
            {
                ca.Update(1);
                double logEvidence1 = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;
                Console.WriteLine(logEvidence1);
                if (i > 20 && System.Math.Abs(logEvidence1 - oldLogEvidence) < 0.01)
                    break;
                oldLogEvidence = logEvidence1;
            }

            //Gaussian xActual = ca.Marginal<Gaussian>(x);
            Beta sActual = ca.Marginal<Beta>(s.NameInGeneratedCode);
            //Console.WriteLine("x = {0}", xActual);
            Console.WriteLine("s = {0}", sActual);
        }


        internal void BetaRegression()
        {
            int P = 8;
            double[] b = new double[P];
            for (int p = 0; p < P; p++)
                b[p] = Rand.Beta(1, 1);
            int N = 100;
            double[][] X = new double[N][];
            //Gaussian[][] softX = new Gaussian[N][]; 
            double[] Y = new double[N];
            for (int n = 0; n < N; n++)
            {
                X[n] = new double[P];
                //softX[n] = new Gaussian[P]; 
                Y[n] = 0;
                for (int p = 0; p < P; p++)
                {
                    X[n][p] = Rand.Normal();
                    //softX[n][p] = new Gaussian(X[n][p], 1e-4); 
                    Y[n] += X[n][p]*b[p];
                }
            }

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range dim = new Range(P).Named("P");
            Range item = new Range(N).Named("N");
            VariableArray<double> w = Variable.Array<double>(dim).Named("w");
            w[dim] = Variable.Beta(1, 1).ForEach(dim);
            var x = Variable.Array(Variable.Array<double>(dim), item).Named("x");
            var softXvar = Variable.Array(Variable.Array<double>(dim), item).Named("softx");
            softXvar.ObservedValue = X;
            x[item][dim] = Variable.GaussianFromMeanAndPrecision(softXvar[item][dim], 1e4);
            var wx = Variable.Array(Variable.Array<double>(dim), item).Named("wx");
            wx[item][dim] = x[item][dim]*w[dim];
            var sum = Variable.Array<double>(item).Named("sum");
            sum[item] = Variable.Sum(wx[item]);
            var prec = Variable.GammaFromShapeAndRate(.1, .1).Named("Noise");
            var y = Variable.Array<double>(item).Named("y");
            y[item] = Variable.GaussianFromMeanAndPrecision(sum[item], prec);
            block.CloseBlock();
            y.ObservedValue = Y;

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            var ca = engine.GetCompiledInferenceAlgorithm(evidence, w);
            ca.Reset();

            double oldLogEvidence = double.NegativeInfinity;
            for (int i = 0; i < 1000; i++)
            {
                ca.Update(1);
                double logEvidence1 = ca.Marginal<Bernoulli>(evidence.NameInGeneratedCode).LogOdds;
                Console.WriteLine(logEvidence1);
                if (i > 20 && System.Math.Abs(logEvidence1 - oldLogEvidence) < 0.01)
                    break;
                oldLogEvidence = logEvidence1;
            }

            DistributionArray<Beta> wInferred = ca.Marginal<DistributionArray<Beta>>(w.NameInGeneratedCode);
            for (int p = 0; p < P; p++)
                Console.WriteLine("w[{0}] = {1} +/- {2} should be {3}",
                                  p, wInferred[p].GetMean(), System.Math.Sqrt(wInferred[p].GetVariance()), b[p]);
        }
    }
}