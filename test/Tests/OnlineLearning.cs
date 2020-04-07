// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System.IO;
using Assert = Xunit.Assert;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /* public void Run()
        {
                int T = 100;
                int midPoint = (int)(Math.Floor((double)T * 0.5));

                Variable<Gamma> precprior = Variable.Given(Gamma.FromMeanAndVariance(1, 100));
                Variable<double> prec = Variable.Random<double, Gamma>(precprior).Named("prec");

                Variable<Gaussian> mprior = Variable.Given(new Gaussian(0, 10));
                Variable<double> m = Variable.GaussianFromMeanAndPrecision(Variable.Random<double, Gaussian>(mprior), prec).Named("m");

                Variable<Gaussian> xObsDist = Variable.Given(new Gaussian(0, 1));
                Variable.ConstrainEqualRandom<double, Gaussian>(m, xObsDist);

                InferenceEngine ie = new InferenceEngine();
                for (int t = 0; t < T; t++)
                {
                        double data = Gaussian.Sample((t < midPoint) ? 0 : 10, 1);
                        xObsDist.Value = new Gaussian(data, 1);

                        Gaussian mpost = ie.Infer<Gaussian>(m);
                        Gamma precpost = ie.Infer<Gamma>(prec);
                        precprior.Value = precpost;
                        mprior.Value = mpost;
                        Console.Write(mprior.Value.GetMean() + "  ");
                }
        }*/

    
    public class OnlineLearning
    {
        public VectorGaussian BatchRegression(Vector[] xdata, double[] ydata)
        {
            int ndim = xdata[0].Count;
            Range row = new Range(xdata.Length);

            VariableArray<Vector> x = Variable.Observed(xdata, row).Named("x");

            // Set a prior on the weights and sample from it
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(Vector.Constant(ndim, 0.0), PositiveDefiniteMatrix.Identity(ndim)).Named("w");

            // Multiply to determine obeservations
            VariableArray<double> y = Variable.Array<double>(row);
            y[row] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[row], w), 1.0);
            y.ObservedValue = ydata;

            InferenceEngine engine = new InferenceEngine();
            VectorGaussian postW = engine.Infer<VectorGaussian>(w);
            return postW;
        }

        public VectorGaussian OnlineRegression(Vector[] xdata, double[] ydata)
        {
            int ndim = xdata[0].Count;
            Variable<Vector> x = Variable.New<Vector>().Named("x");
            Variable<VectorGaussian> wPrior = Variable.New<VectorGaussian>().Named("wPrior");
            Variable<Vector> w = Variable.Random<Vector, VectorGaussian>(wPrior).Named("w");
            Variable<double> y = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x, w), 1.0);
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPost = new VectorGaussian(Vector.Constant(ndim, 0.0), PositiveDefiniteMatrix.Identity(ndim));
            for (int i = 0; i < ydata.Length; i++)
            {
                x.ObservedValue = xdata[i];
                y.ObservedValue = ydata[i];
                wPrior.ObservedValue = wPost;
                wPost = engine.Infer<VectorGaussian>(w);
            }
            return wPost;
        }

        internal void OnlineRegression()
        {
            int n = 100;
            double[] ydata = new double[n];
            Vector[] xdata = new Vector[n];
            Vector wTrue = Vector.FromArray(2.0, 3.0);
            for (int i = 0; i < n; i++)
            {
                xdata[i] = Vector.FromArray(Rand.Normal(), Rand.Normal());
                ydata[i] = xdata[i].Inner(wTrue);
            }
            VectorGaussian postW = BatchRegression(xdata, ydata);
            Console.WriteLine("Batch posterior over the weights: " + Environment.NewLine + postW);
            VectorGaussian postW2 = OnlineRegression(xdata, ydata);
            Console.WriteLine("Online posterior over the weights: " + Environment.NewLine + postW2);
            Assert.True(postW.MaxDiff(postW2) < 1e-10);
        }


        //Testing online learning :)
        //    [Fact]
        internal void OnlineGaussian()
        {
            // make ddata
            double psi;
            int T = 1;
            double[] data = CreateData("Parabola", T, out psi);
            Console.WriteLine("psi = {0}", psi);
            for (int i = 0; i < T; i++)
            {
                Console.Write(" " + data[i]);
            }
            Console.WriteLine();

            InferenceEngine ie = new InferenceEngine();
            ie.ShowFactorGraph = true;
            Variable<double> m;
            Variable<Gaussian> mprior = Variable.Observed(new Gaussian(data[0], 10 + psi));
            //   m = Variable.GaussianFromMeanAndVariance(Variable.Random<double, Gaussian>(mprior), psi).Named("m");
            m = Variable.Random<double, Gaussian>(mprior).Named("m");
            Variable<Gaussian> xObsDist = Variable.Observed(new Gaussian(data[0], 1));
            Variable.ConstrainEqualRandom<double, Gaussian>(m, xObsDist);
            for (int t = 0; t < T; t++)
            {
                xObsDist.ObservedValue = new Gaussian(data[t], 1);
                Gaussian mpost = ie.Infer<Gaussian>(m);
                double mpost_mean, mpost_var;
                mpost.GetMeanAndVariance(out mpost_mean, out mpost_var);
                mprior.ObservedValue = new Gaussian(mpost_mean, mpost_var + psi);
                // mprior.Value = ie.Infer<Gaussian>(m);
                Console.WriteLine(mpost);
            }
        }

        // illustrates how to do online learning (or train vs. test) when using a non-conjugate prior
        [Fact]
        public void OnlineLearningWithCompoundGammaPrecisionTest()
        {
            OnlineLearningWithCompoundGammaPrecision(new ExpectationPropagation());
            OnlineLearningWithCompoundGammaPrecision(new VariationalMessagePassing());
        }
        private void OnlineLearningWithCompoundGammaPrecision(IAlgorithm algorithm)
        {
            Variable<int> nItems = Variable.New<int>().Named("nItems");
            Range item = new Range(nItems).Named("item");
            Variable<double> shape = Variable.Observed(1.0).Named("shape");
            Gamma ratePrior = Gamma.FromShapeAndRate(1, 1);
            Variable<double> rate = Variable<double>.Random(ratePrior).Named("rate");
            Variable<double> prec = Variable.GammaFromShapeAndRate(shape, rate).Named("prec");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(0, prec).ForEach(item);
            Variable<Gamma> precMessage = Variable.Observed<Gamma>(Gamma.Uniform()).Named("precMessage");
            Variable.ConstrainEqualRandom(prec, precMessage);
            prec.AddAttribute(QueryTypes.Marginal);
            prec.AddAttribute(QueryTypes.MarginalDividedByPrior);
            InferenceEngine engine = new InferenceEngine(algorithm);
            engine.ModelName = "GaussianWithCompoundGammaPrecision";
            engine.ShowProgress = false;

            // inference on a single batch
            double[] data = { 2, 3, 4, 5 };
            x.ObservedValue = data;
            nItems.ObservedValue = data.Length;
            Gamma precExpected = engine.Infer<Gamma>(prec);

            // online learning in mini-batches
            int batchSize = 1;
            double[][] dataBatches = new double[data.Length / batchSize][];
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                dataBatches[batch] = data.Skip(batch * batchSize).Take(batchSize).ToArray();
            }
            Gamma precMarginal = Gamma.Uniform();
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                nItems.ObservedValue = dataBatches[batch].Length;
                x.ObservedValue = dataBatches[batch];
                precMarginal = engine.Infer<Gamma>(prec);
                Console.WriteLine("prec after batch {0} = {1}", batch, precMarginal);
                precMessage.ObservedValue = engine.Infer<Gamma>(prec, QueryTypes.MarginalDividedByPrior);
            }
            // the answers should be identical for this simple model
            Console.WriteLine("prec = {0} should be {1}", precMarginal, precExpected);
            Assert.True(precExpected.MaxDiff(precMarginal) < 1e-10);
        }

        // illustrates how to do online learning (or train vs. test) when using a non-conjugate prior
        [Fact]
        public void OnlineLearningWithLaplacianMean()
        {
            Variable<int> nItems = Variable.New<int>().Named("nItems");
            Range item = new Range(nItems).Named("item");
            Variable<double> variance = Variable.GammaFromShapeAndRate(1.0, 1.0).Named("variance");
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0.0, variance).Named("mean");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(mean, 1.0).ForEach(item);

            Variable<Gaussian> meanMessage = Variable.Observed<Gaussian>(Gaussian.Uniform()).Named("meanMessage");
            Variable.ConstrainEqualRandom(mean, meanMessage);
            mean.AddAttribute(QueryTypes.Marginal);
            mean.AddAttribute(QueryTypes.MarginalDividedByPrior);
            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "OnlineLearningWithLaplacianMean";
            engine.ShowProgress = false;

            // inference on a single batch
            double[] data = { 2, 3, 4, 5 };
            x.ObservedValue = data;
            nItems.ObservedValue = data.Length;
            Gaussian meanExpected = engine.Infer<Gaussian>(mean);

            // online learning in mini-batches
            int batchSize = 1;
            double[][] dataBatches = new double[data.Length / batchSize][];
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                dataBatches[batch] = data.Skip(batch * batchSize).Take(batchSize).ToArray();
            }
            Gaussian meanMarginal = Gaussian.Uniform();
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                nItems.ObservedValue = dataBatches[batch].Length;
                x.ObservedValue = dataBatches[batch];
                meanMarginal = engine.Infer<Gaussian>(mean);
                Console.WriteLine("mean after batch {0} = {1}", batch, meanMarginal);
                meanMessage.ObservedValue = engine.Infer<Gaussian>(mean, QueryTypes.MarginalDividedByPrior);
            }
            // the answers should be identical for this simple model
            Console.WriteLine("mean = {0} should be {1}", meanMarginal, meanExpected);
            Assert.True(meanExpected.MaxDiff(meanMarginal) < 1e-10);
        }

        // models dynamics of precision of the means
        //        [Fact]
        internal void OnlineGaussianWithPriorOnStateTransitionVariance()
        {
            // make ddata
            double psi;
            int T = 10;
            double[] data = CreateData("Parabola", T, out psi);
            Console.WriteLine("psi = {0}", psi);
            for (int i = 0; i < T; i++)
            {
                Console.Write(" " + data[i]);
            }
            Console.WriteLine();
            double[] learnedMeans = new double[T];
            double[] learnedPrecs = new double[T];

            InferenceEngine ie = new InferenceEngine();
            Variable<Gamma> precprior = Variable.Observed(Gamma.FromMeanAndVariance(1, 100));
            Variable<double> prec = Variable.Random<double, Gamma>(precprior).Named("prec");

            Variable<Gaussian> mprior = Variable.Observed(new Gaussian(0, 10));
            Variable<double> m = Variable.GaussianFromMeanAndPrecision(Variable.Random<double, Gaussian>(mprior), prec).Named("m");

            Variable<Gaussian> xObsDist = Variable.Observed(new Gaussian(data[0], 1));
            Variable.ConstrainEqualRandom<double, Gaussian>(m, xObsDist);


            for (int t = 0; t < T; t++)
            {
                xObsDist.ObservedValue = new Gaussian(data[t], 1);

                Gaussian mpost = ie.Infer<Gaussian>(m);
                Gamma precpost = ie.Infer<Gamma>(prec);
                precprior.ObservedValue = precpost;
                mprior.ObservedValue = mpost;
                learnedMeans[t] = mprior.ObservedValue.GetMean();
                learnedPrecs[t] = precprior.ObservedValue.GetMean();
            }
            Console.WriteLine("[ " + StringUtil.CollectionToString(learnedMeans, " ") + " ]");
            Console.WriteLine("[ " + StringUtil.CollectionToString(learnedPrecs, " ") + " ]");
        }

        //  [Fact]
        internal void OnlineMixtureGaussianPriorOnStateTransitionWithInertia()
        {
            // make ddata
            double psi;
            int T = 100;
            double[] data = CreateData("ParabolaIParabola", T, out psi);
            int K = 2;
            double[][] learnedMeans = new double[K][];
            double[][] learnedMeansPrec = new double[K][];
            double[][] learnedPrecs = new double[K][];
            for (int i = 0; i < K; i++)
            {
                learnedMeans[i] = new double[T];
                learnedMeansPrec[i] = new double[T];
                learnedPrecs[i] = new double[T];
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            // ie.ShowFactorGraph = true;

            Gaussian[] mpost = new Gaussian[K];
            Gaussian[] mVelocitypost = new Gaussian[K];
            Gamma[] precpost = new Gamma[K];
            Variable<Gamma>[] precprior = new Variable<Gamma>[K];
            Variable<Gaussian>[] mprior = new Variable<Gaussian>[K];
            Variable<Gaussian>[] mVelocityprior = new Variable<Gaussian>[K];
            Variable<double>[] prec = new Variable<double>[K];
            Variable<double>[] m = new Variable<double>[K];
            Variable<double>[] mVelocity = new Variable<double>[K];
            for (int i = 0; i < K; i++)
            {
                precprior[i] = Variable.Observed(Gamma.FromMeanAndVariance(1, 1)).Named("precprior" + i);
                prec[i] = Variable.Random<double, Gamma>(precprior[i]).Named("prec" + i);

                mVelocityprior[i] = Variable.Observed(new Gaussian(1, 1)).Named("mVeloctiyprior" + i);
                Variable<double> mVelocityPrev = Variable.Random<double, Gaussian>(mVelocityprior[i]);
                mVelocity[i] = Variable.GaussianFromMeanAndPrecision(mVelocityPrev, prec[i]).Named("mVelocity" + i);

                mprior[i] = Variable.Observed(new Gaussian(i, 10)).Named("mprior" + i);
                Variable<double> mprev = Variable.Random<double, Gaussian>(mprior[i]);
                m[i] = mprev + mVelocityPrev;
            }

            Variable<Gaussian> xObsDist = Variable.Observed(new Gaussian(data[0], 1)).Named("xObsDist");
            Vector probs = Vector.Constant(K, 1.0/(double) K);
            Variable<int> c = Variable.Discrete(probs).Named("c");
            for (int i = 0; i < K; i++)
            {
                using (Variable.Case(c, i))
                {
                    Variable.ConstrainEqualRandom<double, Gaussian>(m[i], xObsDist);
                }
            }
            Vector cpost = Vector.Zero(K);
            for (int t = 0; t < T; t++)
            {
                xObsDist.ObservedValue = new Gaussian(data[t], 1);
                //  cpost = ie.Infer<Discrete>(c).GetProbs(); ;
                for (int i = 0; i < K; i++)
                {
                    mpost[i] = ie.Infer<Gaussian>(m[i]);
                    precpost[i] = ie.Infer<Gamma>(prec[i]);
                    mVelocitypost[i] = ie.Infer<Gaussian>(mVelocity[i]);
                }
                for (int i = 0; i < K; i++)
                {
                    precprior[i].ObservedValue = precpost[i];
                    mprior[i].ObservedValue = mpost[i];
                    mVelocityprior[i].ObservedValue = mVelocitypost[i];

                    learnedMeans[i][t] = mprior[i].ObservedValue.GetMean();
                    learnedMeansPrec[i][t] = mprior[i].ObservedValue.GetVariance();
                    learnedPrecs[i][t] = precprior[i].ObservedValue.GetMean();
                }
            }
            for (int i = 0; i < K; i++)
            {
                Console.WriteLine("[ " + StringUtil.CollectionToString(learnedMeans[i], " ") + " ]");
            }
            WriteToFile("learnedMeans", data, learnedMeans, learnedPrecs);
            /* for (int i = 0; i < K; i++)
             {
                     Console.WriteLine("[ " + StringUtil.CollectionToString(learnedMeansPrec[i], " ") + " ]");

             }
             Console.WriteLine("" + cpost.ToString());
             Console.WriteLine("first component: mean = {0}, meanprec = {1}  second component: mean = {2}, meanprec = {3} ", learnedMeans[0][0], learnedMeansPrec[0][0], learnedMeans[1][0], learnedMeansPrec[1][0]);
             Console.WriteLine("psi = {0}, estimated = {1}, {2}",  psi, learnedPrecs[0][0],learnedPrecs[1][0]  );
             */
        }

        //[Fact]
        internal void OnlineMixtureGaussian_MeanAndVarianceDynamics()
        {
            bool varianceDynamics = true;

            // make ddata
            double psi;
            int T = 500;
            double[] data = CreateData("ParabolaIParabola", T, out psi);
            int K = 2;
            double[][] learnedMeans = new double[K][];
            double[][] learnedMeansPrec = new double[K][];
            double[][] learnedPrecs = new double[K][];
            for (int i = 0; i < K; i++)
            {
                learnedMeans[i] = new double[T];
                learnedMeansPrec[i] = new double[T];
                learnedPrecs[i] = new double[T];
            }

            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            // ie.ShowFactorGraph = true;

            Gaussian[] mpost = new Gaussian[K];
            Gamma[] precpost = new Gamma[K];
            Variable<Gaussian>[] mprior = new Variable<Gaussian>[K];
            Variable<double>[] m = new Variable<double>[K];
            Variable<Gamma>[] precprior = new Variable<Gamma>[K];
            Variable<double>[] prec = new Variable<double>[K];
            if (varianceDynamics)
            {
                for (int i = 0; i < K; i++)
                {
                    // note that for VMP posintmass prec will give different result
                    precprior[i] = Variable.Observed(Gamma.FromMeanAndVariance(1, 5)).Named("precprior" + i);
                    prec[i] = Variable.Random<double, Gamma>(precprior[i]).Named("prec" + i);
                }
            }
            for (int i = 0; i < K; i++)
            {
                mprior[i] = Variable.Observed(new Gaussian(i, 10)).Named("mprior" + i);
                Variable<double> mprev = Variable.Random<double, Gaussian>(mprior[i]);
                if (varianceDynamics)
                {
                    m[i] = Variable.GaussianFromMeanAndPrecision(mprev, prec[i]).Named("m" + i);
                }
                else
                {
                    m[i] = Variable.GaussianFromMeanAndPrecision(mprev, psi).Named("m" + i);
                }
            }
            Variable<Gaussian> xObsDist = Variable.Observed(new Gaussian(data[0], 1)).Named("xObsDist");
            Vector probs = Vector.Constant(K, 1.0/(double) K);
            Variable<int> c = Variable.Discrete(probs).Named("c");
            for (int i = 0; i < K; i++)
            {
                using (Variable.Case(c, i))
                {
                    Variable.ConstrainEqualRandom<double, Gaussian>(m[i], xObsDist);
                }
            }
            Vector cpost = Vector.Zero(K);
            for (int t = 0; t < T; t++)
            {
                xObsDist.ObservedValue = new Gaussian(data[t], 1);
                // cpost = ie.Infer<Discrete>(c).GetProbs(); ;
                for (int i = 0; i < K; i++)
                {
                    mpost[i] = ie.Infer<Gaussian>(m[i]);
                    learnedMeans[i][t] = mprior[i].ObservedValue.GetMean();
                    if (varianceDynamics)
                    {
                        precpost[i] = ie.Infer<Gamma>(prec[i]);
                    }
                }
                for (int i = 0; i < K; i++)
                {
                    mprior[i].ObservedValue = mpost[i];
                    learnedMeans[i][t] = mprior[i].ObservedValue.GetMean();
                    learnedMeansPrec[i][t] = mprior[i].ObservedValue.GetVariance();
                    if (varianceDynamics)
                    {
                        precprior[i].ObservedValue = precpost[i];
                        learnedPrecs[i][t] = precprior[i].ObservedValue.GetMean();
                    }
                }
            }
            for (int i = 0; i < K; i++)
            {
                Console.WriteLine("[ " + StringUtil.CollectionToString(learnedMeans[i], " ") + " ]");
            }
            WriteToFile("learnedMeans", data, learnedMeans, learnedPrecs);
        }

        public double[] ChooseSamples(double[][] trueData)
        {
            int T = trueData[0].Length;
            int nClass = trueData.Length;
            double[] data = new double[T];
            int classIndx;
            Vector prob = Vector.Constant(nClass, 1.0/(double) nClass);
            for (int t = 0; t < T; t++)
            {
                classIndx = Discrete.Sample(prob);
                // classIndx = t % 2;
                data[t] = trueData[classIndx][t];
            }
            return data;
        }

        public double[] CreateData2(String s, int T, out double psi)
        {
            double[] x = Linspace(-5, 5, T);
            int nClass = 2;
            double[][] trueData = new double[nClass][];
            for (int i = 0; i < nClass; i++)
            {
                trueData[i] = new double[T];
            }

            if (s.Equals("ParabolaIParabola"))
            {
                for (int t = 0; t < T; t++)
                {
                    trueData[0][t] = 2 + x[t]*x[t];
                    trueData[1][t] = 6 - x[t]*x[t];
                }
            }
            else if (s.Equals("ParabolaSine"))
            {
                for (int t = 0; t < T; t++)
                {
                    trueData[0][t] = 2 + x[t]*x[t];
                    trueData[1][t] = 2* System.Math.Sin(2* x[t] * System.Math.PI/4.0);
                }
            }
            psi = EstimateDataVariance(trueData[0]);
            for (int i = 1; i < nClass; i++)
            {
                psi = System.Math.Max(EstimateDataVariance(trueData[i]), psi);
            }
            return ChooseSamples(trueData);
        }

        // Parabola, StationaryShift, ParabolaIParabola,ParabolaSine 
        public double[] CreateData(String s, int T, out double psi)
        {
            if (s.Equals("ParabolaIParabola") || (s.Equals("ParabolaSine")))
                return CreateData2(s, T, out psi);
            else if (s.Equals("Parabola") || (s.Equals("StationaryShift")))
            {
                double[] data = new double[T];
                psi = 0;
                if (s.Equals("Parabola"))
                {
                    data = Linspace(-5, 5, T);
                    for (int i = 0; i < T; i++)
                    {
                        data[i] = 2 + data[i]*data[i];
                    }
                }
                else if (s.Equals("StationaryShift"))
                {
                    int midPoint = (int) (System.Math.Floor((double) T*0.5));
                    for (int i = 0; i < T; i++)
                    {
                        data[i] = (i < midPoint) ? 0 : 10;
                    }
                }
                psi = EstimateDataVariance(data);
                return data;
            }
            else
            {
                Console.WriteLine("WARNING: Method for generating data of the type {0} not specified", s);
                psi = Double.NaN;
                return new double[T];
            }
        }


        public double EstimateDataVariance(double[] data)
        {
            double psi = 0;
            double diff;
            for (int i = 1; i < data.Length; i++)
            {
                diff = data[i] - data[i - 1];
                psi = psi + diff*diff;
            }
            psi = (psi > 0) ? psi/(data.Length - 1) : .00001;
            return psi;
        }

        public double[] SampleFrom(double[][] trueData, int nClass, int T)
        {
            Console.WriteLine("T = {0} nClass = {1} rank =  {2} Length = {3} ", T, nClass, trueData.Rank, trueData.Length);
            double[] data = new double[T];
            int[] swapInd = new int[T];

            int classIndx;
            for (int t = 0; t < T; t++)
            {
                classIndx = Discrete.Sample(
                    Vector.Constant(nClass, 1.0/(double) nClass));
                data[t] = trueData[classIndx][t];
            }

            return data;
        }


        public double[] Linspace(double d1, double d2, int n1)
        {
            double n = n1*1.0;
            double[] linspaced = new double[n1];

            for (int i = 0; i < n1 - 1; i++)
            {
                linspaced[i] = d1 + i*(d2 - d1)/(n - 1);
            }
            linspaced[n1 - 1] = d2;
            return linspaced;
        }


        /* Gaussian[] mpost = new Gaussian[K];
                Gamma[] precpost = new Gamma[K];
                Variable<Gamma>[] precprior = new Variable<Gamma>[K];
                Variable<Gaussian>[] mprior = new Variable<Gaussian>[K]; 
                Variable<double>[] prec = new Variable<double>[K];
                Variable<double>[] m = new Variable<double>[K];
                for (int i = 0; i < K; i++)
                {
                        precprior[i] = Variable.Given(Gamma.FromMeanAndVariance(1.0/psi, 0)).Named("precprior" + i);
                        prec[i] = Variable.Random<double, Gamma>(precprior[i]).Named("prec" + i);              
                        mprior[i] = Variable.Given(new Gaussian(i, 10)).Named("mprior" + i);
                        m[i] = Variable.GaussianFromMeanAndPrecision(Variable.Random<double, Gaussian>(mprior[i]), prec[i]).Named("m" + i);
             }*/


        [Fact]
        public void OnePointMixtureGaussianTest2()
        {
            double psi = 0.34006664952384;
            int K = 2;

            Variable<double>[] m = new Variable<double>[K];
            Variable<Gamma>[] precprior = new Variable<Gamma>[K];
            Variable<Gaussian>[] mprior = new Variable<Gaussian>[K];
            Variable<double>[] prec = new Variable<double>[K];
            for (int i = 0; i < K; i++)
            {
                precprior[i] = Variable.Observed(Gamma.FromMeanAndVariance(1.0/psi, 0)).Named("precprior" + i);
                prec[i] = Variable.Random<double, Gamma>(precprior[i]).Named("prec" + i);
                mprior[i] = Variable.Observed(new Gaussian(i, 10)).Named("mprior" + i);
                m[i] = Variable.GaussianFromMeanAndPrecision(Variable.Random<double, Gaussian>(mprior[i]), prec[i]).Named("m" + i);
            }
            Vector probs = Vector.Constant(K, 1.0/(double) K);
            Variable<int> c = Variable.Discrete(probs).Named("c");
            for (int i = 0; i < K; i++)
            {
                using (Variable.Case(c, i))
                {
                    Variable.ConstrainEqualRandom<double, Gaussian>(m[i], Variable.Constant(new Gaussian(27, 1)));
                }
            }

            InferenceEngine ie = new InferenceEngine();
            Discrete cpost = Discrete.Uniform(K);
            cpost = ie.Infer<Discrete>(c);
            double error = cpost.MaxDiff(new Discrete(new double[] {0.08811694213437, 0.91188305786563}));
            Console.WriteLine("cpost's error = {0}", error);
            Assert.True(error < 1e-5);

            Gaussian[] mExpected = new Gaussian[K];
            mExpected[0] = new Gaussian(2.16935642753181, 58.21069036896901);
            mExpected[1] = new Gaussian(22.61823462278782, 46.90331161710029);

            Gaussian[] mpost = new Gaussian[K];
            for (int i = 0; i < K; i++)
            {
                mpost[i] = ie.Infer<Gaussian>(m[i]);
                error = mpost[i].MaxDiff(mExpected[i]);
                Console.WriteLine("mpost[{0}] = {1} (should be {2}) and  error = {3}", i, mpost[i], mExpected[i], error);
                // Assert.True(error < 1e-5);
            }
        }

        private static void WriteToFile(String filename, double[] x, double[][] learnedMeans, double[][] learnedPrecs)
        {
            FileInfo fi = new FileInfo(@"c:\" + filename + ".txt");
            DirectoryInfo di = fi.Directory;
            int K = learnedMeans.Length;
            int T = learnedMeans[0].Length;
            StreamWriter swr = fi.CreateText();
            try
            {
                for (int t = 0; t < T; t++)
                {
                    swr.Write(x[t] + " ");
                }
                swr.WriteLine();
                for (int k = 0; k < K; k++)
                {
                    for (int t = 0; t < T; t++)
                    {
                        swr.Write(learnedMeans[k][t] + " ");
                    }
                    swr.WriteLine();
                }
                for (int k = 0; k < K; k++)
                {
                    for (int t = 0; t < T; t++)
                    {
                        swr.Write(learnedPrecs[k][t] + " ");
                    }
                    swr.WriteLine();
                }
            }
            finally
            {
                swr.Flush();
                swr.Close();
            }
            Console.WriteLine("The directory '{0}' contains the file just created", di.FullName);
        }

        [Fact]
        public void OnePointMixtureGaussianTest3()
        {
            int K = 2;

            InferenceEngine ie = new InferenceEngine(); //new VariationalMessagePassing());

            Variable<Gaussian>[] mprior = new Variable<Gaussian>[K];
            Variable<double>[] m = new Variable<double>[K];
            Variable<Gamma>[] precprior = new Variable<Gamma>[K];
            Variable<double>[] prec = new Variable<double>[K];
            for (int i = 0; i < K; i++)
            {
                precprior[i] = Variable.Observed(Gamma.FromMeanAndVariance(1, 1)).Named("precprior" + i);
                prec[i] = Variable.Random<double, Gamma>(precprior[i]).Named("prec" + i);
            }
            for (int i = 0; i < K; i++)
            {
                mprior[i] = Variable.Observed(new Gaussian(i, 10)).Named("mprior" + i);
            }
            Vector probs = Vector.Constant(K, 1.0/(double) K);
            Variable<int> c = Variable.Discrete(probs).Named("c");
            for (int i = 0; i < K; i++)
            {
                using (Variable.Case(c, i))
                {
                    Variable<double> mprev = Variable.Random<double, Gaussian>(mprior[i]);
                    m[i] = Variable.GaussianFromMeanAndPrecision(mprev, prec[i]).Named("m" + i);
                    Variable.ConstrainEqualRandom<double, Gaussian>(m[i], Variable.Constant(new Gaussian(27, 1)));
                }
            }
            Console.WriteLine(ie.Infer<Discrete>(c));
            for (int i = 0; i < K; i++)
            {
                Console.WriteLine(ie.Infer<Gaussian>(m[i]));
                Console.WriteLine(ie.Infer<Gamma>(prec[i]));
            }
        }
    }
}