// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using System.IO;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System.Globalization;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class BioTests
    {
        internal void MarkusTruncated()
        {
            int nData = 1000;
            double[] data = new double[nData];
            bool[] trunc = new bool[nData];
            Rand.Restart(1234);

            int nTrunc = 0;
            for (int i = 0; i < nData; i++)
            {
                data[i] = Rand.Normal(1.0, 1);
                if (data[i] < 0)
                {
                    trunc[i] = true;
                    nTrunc++;
                }
                else
                    trunc[i] = false;
            }
            Console.WriteLine("{0} values truncated.", nTrunc);
            Range N = new Range(nData);
            VariableArray<double> obsData = Variable.Array<double>(N).Named("obsData");
            VariableArray<bool> truncInd = Variable.Array<bool>(N).Named("truncInd");

            var ev = Variable.Bernoulli(0.5).Named("ev");
            var modelBlock = Variable.If(ev);
            //VariableArray<double> dummy = Variable.Array<double>(N).Named("dummy");
            obsData.ObservedValue = data;
            truncInd.ObservedValue = trunc;
            Variable<double> prec = Variable.GammaFromShapeAndScale(2.0, 0.5).Named("prec");
            //Variable<double> variance = Variable.GammaFromShapeAndScale(2.0, 0.5).Named("prec");
            Variable<double> meanPrior = Variable.New<double>().Named("meanPrior");
            meanPrior.ObservedValue = 0.0;
            Variable<double> mean = Variable.GaussianFromMeanAndPrecision(meanPrior, 0.01*prec).Named("mean");
            //Variable<double> mean = Variable.GaussianFromMeanAndPrecision(0.0, 0.01);
            using (Variable.ForEach(N))
            {
                //dummy[N] = Variable.GaussianFromMeanAndPrecision(mean, prec);
                using (Variable.IfNot(truncInd[N]))
                {
                    obsData[N] = Variable.GaussianFromMeanAndPrecision(mean, prec);
                    //obsData[N] = Variable.GaussianFromMeanAndVariance(mean, variance);
                }
                using (Variable.If(truncInd[N]))
                {
                    var dummy = Variable.GaussianFromMeanAndPrecision(-mean, prec);
                    //var dummy = Variable.GaussianFromMeanAndVariance(mean, variance);
                    Variable.ConstrainPositive(dummy /*[N]*/);
                }
            }
            modelBlock.CloseBlock();
            //var ie = new InferenceEngine(new ExpectationPropagation());
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 100;
            var meanPost = ie.Infer<Gaussian>(mean);
            Console.WriteLine("Posterior over the mean = {0}", meanPost);
            var precPost = ie.Infer<Gamma>(prec);
            Console.WriteLine("Posterior over the prec = {0}", precPost);
            Console.WriteLine("Log marginal likelihood = {0}", ie.Infer<Bernoulli>(ev).LogOdds);
        }


        public static VariableArray<Vector> CPT1(Range rY, Range rS)
        {
            int dimS = rS.SizeAsInt;
            int dimY = rY.SizeAsInt;
            double pc = 1/(double) (dimS*dimY);

            var priorObs = new Dirichlet[dimS];
            for (int i = 0; i < dimS; i++)
                priorObs[i] = Dirichlet.Symmetric(dimS, pc);

            var priorVar = Variable.Array<Dirichlet>(rS);
            priorVar.ObservedValue = priorObs;

            var cpt = Variable.Array<Vector>(rS);
            cpt[rS] = Variable<Vector>.Random(priorVar[rS]);
            cpt.SetValueRange(rY);

            return cpt;
        }


        public static VariableArray<VariableArray<Vector>, Vector[][]> CPT2(Range rY, Range rS0, Range rS1)
        {
            int dimS0 = rS0.SizeAsInt;
            int dimS1 = rS1.SizeAsInt;
            int dimY = rY.SizeAsInt;
            double pc = 1/(double) (dimS0*dimS1*dimY);


            var priorObs = new Dirichlet[dimS0][];
            for (int i0 = 0; i0 < dimS0; i0++)
            {
                priorObs[i0] = new Dirichlet[dimS1];
                for (int i1 = 0; i1 < dimS1; i1++)
                    priorObs[i0][i1] = Dirichlet.Symmetric(dimY, pc);
            }

            var priorVar = Variable.Array(Variable.Array<Dirichlet>(rS1), rS0);
            priorVar.ObservedValue = priorObs;

            var cpt = Variable.Array(Variable.Array<Vector>(rS1), rS0);
            cpt[rS0][rS1] = Variable<Vector>.Random(priorVar[rS0][rS1]);
            cpt.SetValueRange(rY);
            return cpt;
        }


        internal void NevenaRunArrayIntervention()
        {
            int S1 = 0, S2 = 1, Y1 = 2, Y2 = 3;
            int numVar = 4;
            int[] dimVar = Enumerable.Range(0, numVar).Select(o => 2).ToArray();

            // Data
            int numData = 100;
            int[][] dataX = new int[numVar][];
            bool[][] dataI = new bool[numVar][];
            int[] observed = new int[] {Y1, Y2};
            int[][] parents = new int[numVar][];
            parents[S1] = new int[0] {};
            parents[S2] = new int[0] {};
            parents[Y1] = new int[] {S2};
            parents[Y2] = new int[] {S1, S2};

            // here data iid (not sampled from dag)
            var coin = new Discrete(new double[2] {0, 1});

            for (int j = 0; j < numVar; j++)
            {
                dataX[j] = new int[numData];
                dataI[j] = new bool[numData];
                for (int i = 0; i < numData; i++)
                {
                    dataX[j][i] = coin.Sample();
                    dataI[j][i] = false;
                }
            }

            // Model
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock H1 = Variable.If(evidence);

            Range N = new Range(numData).Named("N");
            Range[] rVariables = new Range[numVar];
            for (int d = 0; d < numVar; d++)
                rVariables[d] = new Range(dimVar[d]);

            // Six different priors, depending on #parents and intervention
            var priorX0 = new Variable<Vector>[numVar];
            var priorX1 = new VariableArray<Vector>[numVar];
            var priorX2 = new VariableArray<VariableArray<Vector>, Vector[][]>[numVar];

            var priorXI0 = new Variable<Vector>[numVar];
            var priorXI1 = new VariableArray<Vector>[numVar];
            var priorXI2 = new VariableArray<VariableArray<Vector>, Vector[][]>[numVar];

            for (int d = 0; d < numVar; d++)
            {
                int numPar = parents[d].Length;
                if (numPar == 0)
                {
                    priorX0[d] = Variable.DirichletSymmetric(rVariables[d], 1/(double) dimVar[d]).Named("priorX0" + d);
                    priorXI0[d] = Variable.DirichletSymmetric(rVariables[d], 1/(double) dimVar[d]).Named("priorXI0" + d);
                }
                if (numPar == 1)
                {
                    priorX1[d] = CPT1(rVariables[d], rVariables[parents[d][0]]).Named("priorX1" + d);
                    priorXI1[d] = CPT1(rVariables[d], rVariables[parents[d][0]]).Named("priorXI1" + d);
                }
                if (numPar == 2)
                {
                    priorX2[d] = CPT2(rVariables[d], rVariables[parents[d][0]], rVariables[parents[d][1]]).Named("cptX2" + d);
                    priorXI2[d] = CPT2(rVariables[d], rVariables[parents[d][0]], rVariables[parents[d][1]]).Named("cptXI2" + d);
                }
            }

            // Variable arrays
            VariableArray<int>[] X = new VariableArray<int>[numVar];
            VariableArray<bool>[] I = new VariableArray<bool>[numVar];
            for (int d = 0; d < numVar; d++)
            {
                I[d] = Variable.Array<bool>(N).Named("I" + d);
                I[d].ObservedValue = dataI[d];

                X[d] = Variable.Array<int>(N).Named("X" + d);
                X[d].SetValueRange(rVariables[d]);

                int ind = Array.IndexOf(observed, d);
                if (ind > -1)
                    X[d].ObservedValue = dataX[d];
            }

            for (int d = 0; d < numVar; d++)
            {
                int numPar = parents[d].Length;

                if (numPar == 0)
                {
                    using (Variable.ForEach(N))
                    {
                        using (Variable.IfNot(I[d][N]))
                            X[d][N] = Variable.Discrete(priorX0[d]);
                        using (Variable.If(I[d][N]))
                            X[d][N] = Variable.Discrete(priorXI0[d]);
                    }
                }

                if (numPar == 1)
                {
                    int par = parents[d][0];
                    using (Variable.ForEach(N))
                    {
                        using (Variable.Switch(X[par][N]))
                        {
                            using (Variable.IfNot(I[d][N]))
                                X[d][N] = Variable.Discrete(priorX1[d][X[par][N]]);
                            using (Variable.If(I[d][N]))
                                X[d][N] = Variable.Discrete(priorXI1[d][X[par][N]]);
                        }
                    }
                }

                if (numPar == 2)
                {
                    using (Variable.ForEach(N))
                    {
                        using (Variable.Switch(X[parents[d][0]][N]))
                        using (Variable.Switch(X[parents[d][1]][N]))
                        {
                            using (Variable.IfNot(I[d][N]))
                                X[d][N] = Variable.Discrete(priorX2[d][X[parents[d][0]][N]][X[parents[d][1]][N]]);
                            using (Variable.If(I[d][N]))
                                X[d][N] = Variable.Discrete(priorXI2[d][X[parents[d][0]][N]][X[parents[d][1]][N]]);
                        }
                    }
                }
            }
            H1.CloseBlock();

            // Inference engine
            var ieVMP = new InferenceEngine(new VariationalMessagePassing());
            ieVMP.NumberOfIterations = 100;
            double eviVMP = ieVMP.Infer<Bernoulli>(evidence).LogOdds;

            var ieEP = new InferenceEngine(new ExpectationPropagation());
            ieEP.NumberOfIterations = 100;
            double eviEP = ieEP.Infer<Bernoulli>(evidence).LogOdds;

            Console.WriteLine(eviVMP + " " + eviEP);
        }

        internal void SoftmaxGWAS()
        {
            var v = Vector.Zero(1);
            var x1 = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(1), PositiveDefiniteMatrix.Identity(1));
            x1.ObservedValue = v;
            var x2 = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(1), PositiveDefiniteMatrix.Identity(1));
            v[0] = 1;
            x2.ObservedValue = v;
            Console.WriteLine(x1.ObservedValue);
            Console.WriteLine(x2.ObservedValue);

            int numExposures = 10;
            int numClasses = 5;
            int numSamples = 1000;
            var c = new Range(numClasses);
            var n = new Range(numSamples);
            // model
            var Bexposures = Variable.Array<Vector>(c);
            Bexposures[c] = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(numExposures),
                PositiveDefiniteMatrix.Identity(numExposures))
                                    .ForEach(c);
            var exposures = Variable.Array<Vector>(n);
            var Bgenotype = Variable.Array<double>(c);
            Bgenotype[c] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(c);
            var genotypes = Variable.Array<double>(n);
            var mean = Variable.Array<double>(c);
            mean[c] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(c);
            var g = Variable.Array(Variable.Array<double>(c), n);
            g[n][c] = Bgenotype[c]*genotypes[n] + Variable.InnerProduct(Bexposures[c], exposures[n]) + mean[c];
            var classProbs = Variable.Array<Vector>(n);
            classProbs[n] = Variable.Softmax(g[n]);
            var classAssignments = Variable.Array<int>(n);
            classAssignments[n] = Variable.Discrete(classProbs[n]);
            // generate data
            var exposuresObs = new Vector[numSamples];
            var genotypeObs = new double[numSamples];
            var classObs = new int[numSamples];
            var trueBexposures = new Vector[numClasses];
            var trueBgenotype = new double[numClasses];
            Rand.Restart(123);
            for (int i = 0; i < numClasses; i++)
            {
                trueBexposures[i] = VectorGaussian.SampleFromMeanAndVariance(Vector.Zero(numExposures), PositiveDefiniteMatrix.Identity(numExposures));
                trueBgenotype[i] = Rand.Normal();
            }
            for (int i = 0; i < numSamples; i++)
            {
                exposuresObs[i] = VectorGaussian.SampleFromMeanAndVariance(Vector.Zero(numExposures), PositiveDefiniteMatrix.Identity(numExposures));
                genotypeObs[i] = Rand.Double() < .5 ? 0.0 : 1.0;
                var gt = new double[numClasses];
                for (int j = 0; j < numClasses; j++)
                {
                    gt[j] = genotypeObs[i]*trueBgenotype[j] + exposuresObs[i].Inner(trueBexposures[j]);
                }
                var p = MMath.Softmax(gt);
                classObs[i] = Discrete.Sample(p);
            }
            Rand.Restart(DateTime.Now.Millisecond);
            exposures.ObservedValue = exposuresObs;
            genotypes.ObservedValue = genotypeObs;
            classAssignments.ObservedValue = classObs;
            // inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var bPost = ie.Infer<Gaussian[]>(Bgenotype);
            for (int i = 0; i < numClasses; i++)
            {
                Console.WriteLine("Inferred: " + bPost[i] + " truth: " + trueBgenotype[i]);
            }
        }

        public static double[,] ReadCSV(
            string ifn) // The file name
        {
            // File is assumed to have a header row, followed by
            // tab or comma separated label, clicks, exams
            double[,] M = null;
            int Nrows, Ncols;
            Nrows = 1;
            Ncols = 0;
            string myStr;
            char[] sep = {'\t', ','};
            for (int pass = 0; pass < 2; pass++)
            {
                if (1 == pass)
                {
                    M = new double[Nrows,Ncols];
                    Nrows = 0;
                }
                using (var mySR = new StreamReader(ifn))
                {
                    mySR.ReadLine(); // Skip over header line
                    //overwrite number seperator to use US-style(. for decimal separation)
                    NumberFormatInfo nfi = new CultureInfo("en-US", false).NumberFormat;
                    while ((myStr = mySR.ReadLine()) != null)
                    {
                        string[] mySplitStr = myStr.Split(sep);
                        Ncols = mySplitStr.Length;
                        if (1 == pass)
                        {
                            //parse the doubles into the array.
                            for (int i = 0; i < Ncols; i++)
                            {
                                string str = mySplitStr[i];
                                M[Nrows, i] = Double.Parse(str, nfi);
                            }
                        }


                        Nrows++;
                    }
                }
            }
            return M;
        }


        public static DistributionArray2D<Gaussian, double> RandomGaussianArray(int C, int d)
        {
            Gaussian[,] array = new Gaussian[C,d];
            for (int i = 0; i < C; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    array[i, j] = Gaussian.FromMeanAndVariance(Rand.Normal(), 1);
                }
            }
            return (DistributionArray2D<Gaussian, double>) Distribution<double>.Array(array);
        }


        internal static IDistribution<Vector[]> RandomGaussianVectorArray(int N, int C)
        {
            VectorGaussian[] array = new VectorGaussian[N];
            for (int i = 0; i < N; i++)
            {
                Vector mean = Vector.Zero(C);
                for (int j = 0; j < C; j++) mean[j] = Rand.Normal();
                array[i] = new VectorGaussian(mean, PositiveDefiniteMatrix.Identity(C));
            }
            return Distribution<Vector>.Array(array);
        }

        internal static void WriteMatrix(VectorGaussian[] matrix, string filename)
        {
            StreamWriter sw = new StreamWriter(filename);
            foreach (VectorGaussian vg in matrix)
            {
                Vector v = vg.GetMean();
                for (int i = 0; i < v.Count; i++) sw.Write(v[i] + " ");
                sw.WriteLine();
            }
            sw.Close();
        }

        internal static void WriteMatrix(Gaussian[,] matrix, string filename)
        {
            StreamWriter sw = new StreamWriter(filename);
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++) sw.Write(matrix[i, j].GetMean() + " ");
                sw.WriteLine();
            }
            sw.Close();
        }

        internal static void WriteMatrix(double[,] matrix, string filename)
        {
            StreamWriter sw = new StreamWriter(filename);
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++) sw.Write(matrix[i, j] + " ");
                sw.WriteLine();
            }
            sw.Close();
        }

        internal void ProbabilisticIndexMap()
        {
            //TODO: change the path for cross platform using
            double[,] dataIn = MatlabReader.ReadMatrix(new double[10,6400*3], @"c:\temp\pim\chand.txt", ' ');
            Vector[,] pixData = new Vector[10,6400];
            for (int i = 0; i < pixData.GetLength(0); i++)
            {
                int ct = 0;
                for (int j = 0; j < pixData.GetLength(1); j++)
                {
                    pixData[i, j] = Vector.FromArray(dataIn[i, ct++], dataIn[i, ct++], dataIn[i, ct++]);
                }
            }
            Range images = new Range(pixData.GetLength(0));
            Range pixels = new Range(pixData.GetLength(1));
            VariableArray2D<Vector> pixelData = Variable.Constant(pixData, images, pixels);

            // For each image we have a palette of L multivariate Gaussians
            Range L = new Range(2);
            VariableArray2D<Vector> means = Variable.Array<Vector>(images, L).Named("means");
            means[images, L] = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.FromArray(0.5, 0.5, 0.5),
                PositiveDefiniteMatrix.Identity(3)).ForEach(images, L);
            VariableArray2D<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(images, L).Named("precs");
            precs[images, L] = Variable.WishartFromShapeAndScale(1.0, PositiveDefiniteMatrix.Identity(3)).ForEach(images, L);

            // Across all pixels we have a 
            VariableArray<Vector> pi = Variable.Array<Vector>(pixels);
            pi[pixels] = Variable.Dirichlet(L, new double[] {1.1, 1.0}).ForEach(pixels);
            // For each pixel of each image we have a discrete indicator
            VariableArray2D<int> ind = Variable.Array<int>(images, pixels).Named("ind");
            ind[images, pixels] = Variable.Discrete(pi[pixels]).ForEach(images);

            using (Variable.ForEach(pixels))
            {
                using (Variable.ForEach(images))
                {
                    using (Variable.Switch(ind[images, pixels]))
                    {
                        pixelData[images, pixels] = Variable.VectorGaussianFromMeanAndPrecision(means[images, ind[images, pixels]],
                                                                                                precs[images, ind[images, pixels]]);
                    }
                }
            }
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowProgress = true;
            ie.NumberOfIterations = 5;
            Console.WriteLine("Dist over L: " + ie.Infer(pi));
        }

        internal void ProbabilisticIndexMapNoGate()
        {
            //TODO: change path for cross platform using
            double[,] pixData = MatlabReader.ReadMatrix(new double[10,6400], @"c:\temp\pim\chand2.txt", ' ');
            Range images = new Range(pixData.GetLength(0));
            Range pixels = new Range(pixData.GetLength(1));
            VariableArray2D<double> pixelData = Variable.Constant(pixData, images, pixels);
            //pixelData.QuoteInMSL = false;

            // For each image we have a palette of L multivariate Gaussians
            VariableArray<double> means = Variable.Array<double>(images).Named("means");
            means[images] = Variable.GaussianFromMeanAndPrecision(0.5, 1).ForEach(images);
            VariableArray<double> precs = Variable.Array<double>(images).Named("precs");
            precs[images] = Variable.GammaFromShapeAndScale(1.0, 1.0).ForEach(images);

            // Across all pixels we have a 
            VariableArray<Vector> pi = Variable.Array<Vector>(pixels).Named("pi");
            Dirichlet[] dinit = new Dirichlet[pixels.SizeAsInt];
            for (int i = 0; i < dinit.Length; i++)
            {
                double d = Rand.Double();
                dinit[i] = new Dirichlet(1.0 + d/10, 1.0 - d/10);
            }
            pi[pixels] = Variable.Dirichlet(new double[] {1.0}).ForEach(pixels);
            // For each pixel of each image we have a discrete indicator
            VariableArray2D<int> ind = Variable.Array<int>(images, pixels).Named("ind");
            ind[images, pixels] = Variable.Discrete(pi[pixels]).ForEach(images);

            using (Variable.ForEach(pixels))
            {
                using (Variable.ForEach(images))
                {
                    pixelData[images, pixels] = Variable.GaussianFromMeanAndPrecision(means[images], //10);
                                                                                      precs[images]);
                    Variable.ConstrainEqualRandom(ind[images, pixels], Discrete.Uniform(1));
                }
            }
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ModelName = "PIM_NoGate";
            ie.NumberOfIterations = 8;
            ie.ShowTimings = true;
            DistributionArray<Dirichlet> piDist = ie.Infer<DistributionArray<Dirichlet>>(pi);
            //Console.WriteLine("Dist over pi: " + ie.Infer(pi));
            //TODO: change path for cross platform using
            WriteMatrix(piDist.ToArray(), @"C:\temp\pim\results.txt");
        }

        internal void JudgementModel()
        {
            int[,] Rdata = new int[,] {{0, 1}, {0, 1}, {0, 1}, {1, 0}};
            Range judges = new Range(Rdata.GetLength(0));
            Range docs = new Range(Rdata.GetLength(1));
            int numberOfLevels = 2;
            Vector counts = Vector.Constant(numberOfLevels, 1.0);
            Variable<Vector> Qprior = Variable.Dirichlet(counts);
            VariableArray<int> Q = Variable.Array<int>(docs);
            Q[docs] = Variable.Discrete(Qprior).ForEach(docs);
            Vector[] alpha = new Vector[numberOfLevels];
            VariableArray<Vector>[] B = new VariableArray<Vector>[numberOfLevels];
            for (int i = 0; i < alpha.Length; i++)
            {
                alpha[i] = Vector.Zero(numberOfLevels);
                alpha[i].SetAllElementsTo(1); // the off-diagonal pseudocount
                alpha[i][i] = 2; // the diagonal pseudocount
                B[i] = Variable.Array<Vector>(judges);
                B[i][judges] = Variable.Dirichlet(alpha[i]).ForEach(judges);
            }
            VariableArray2D<int> R = Variable.Constant(Rdata, judges, docs);
            using (Variable.ForEach(docs))
            {
                for (int i = 0; i < numberOfLevels; i++)
                {
                    using (Variable.Case(Q[docs], i))
                    {
                        // TODO: ask infer.net team how to make this sparse
                        R[judges, docs] = Variable.Discrete(B[i][judges]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            for (int i = 0; i < numberOfLevels; i++)
            {
                Console.WriteLine("Dist over B[" + i + "]:\n" + engine.Infer(B[i]));
            }
            Console.WriteLine("Dist over Q:\n" + engine.Infer(Q));
        }

        internal void JudgementModelSparse()
        {
            int[,] Rdata = new int[,] {{0, 1}, {0, 1}, {0, 1}, {1, 0}}; //, { -1, 0 }, { 0, -1 } };
            double[,] observed = new double[Rdata.GetLength(0),Rdata.GetLength(1)];
            for (int i = 0; i < Rdata.GetLength(0); i++)
                for (int j = 0; j < Rdata.GetLength(1); j++)
                {
                    observed[i, j] = (Rdata[i, j] == -1) ? 0 : 1;
                    Rdata[i, j] = System.Math.Max(Rdata[i, j], 0);
                }
            Range judges = new Range(Rdata.GetLength(0));
            Range docs = new Range(Rdata.GetLength(1));
            int numberOfLevels = 2;
            Vector counts = Vector.Constant(numberOfLevels, 1.0);
            Variable<Vector> Qprior = Variable.Dirichlet(counts);
            VariableArray<int> Q = Variable.Array<int>(docs);
            Q[docs] = Variable.Discrete(Qprior).ForEach(docs);
            Vector[] alpha = new Vector[numberOfLevels];
            VariableArray<Vector>[] B = new VariableArray<Vector>[numberOfLevels];
            for (int i = 0; i < alpha.Length; i++)
            {
                alpha[i] = Vector.Zero(numberOfLevels);
                alpha[i].SetAllElementsTo(1); // the off-diagonal pseudocount
                alpha[i][i] = 2; // the diagonal pseudocount
                B[i] = Variable.Array<Vector>(judges);
                B[i][judges] = Variable.Dirichlet(alpha[i]).ForEach(judges);
            }
            VariableArray2D<int> R = Variable.Constant(Rdata, judges, docs);
            VariableArray2D<double> obs = Variable.Constant(observed, judges, docs);
            VariableArray2D<bool> obsVar = Variable.Array<bool>(judges, docs);
            obsVar[judges, docs] = Variable.Bernoulli(obs[judges, docs]);
            //Variable.ConstrainEqual(obs[judges, docs], obsVar[judges, docs]);
            for (int i = 0; i < numberOfLevels; i++)
            {
                using (Variable.ForEach(docs))
                {
                    using (Variable.ForEach(judges))
                    {
                        using (Variable.Case(Q[docs], i))
                        {
                            using (Variable.If(obsVar[judges, docs]))
                            {
                                R[judges, docs] = Variable.Discrete(B[i][judges]);
                            }
                        }
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            for (int i = 0; i < numberOfLevels; i++)
            {
                Console.WriteLine("Dist over B[" + i + "]:\n" + engine.Infer(B[i]));
            }
            Console.WriteLine("Dist over Q:\n" + engine.Infer(Q));
        }

        internal void JudgementModelSparse2()
        {
            int numberOfLevels = 2;
            int[,] Rdata = new int[,] {{0, 1}, {0, 1}, {0, 1}, {1, 0}}; //, { -1, 0 }, { 0, -1 } };
            Discrete[,] Rsoft = new Discrete[Rdata.GetLength(0),Rdata.GetLength(1)];
            for (int i = 0; i < Rdata.GetLength(0); i++)
                for (int j = 0; j < Rdata.GetLength(1); j++)
                {
                    int r = Rdata[i, j];
                    Rsoft[i, j] = (r == -1) ? Discrete.Uniform(numberOfLevels) : Discrete.PointMass(r, numberOfLevels);
                }
            Range judges = new Range(Rdata.GetLength(0));
            Range docs = new Range(Rdata.GetLength(1));
            Vector counts = Vector.Constant(numberOfLevels, 1.0);
            Variable<Vector> Qprior = Variable.Dirichlet(counts);
            VariableArray<int> Q = Variable.Array<int>(docs);
            Q[docs] = Variable.Discrete(Qprior).ForEach(docs);
            Vector[] alpha = new Vector[numberOfLevels];
            VariableArray<Vector>[] B = new VariableArray<Vector>[numberOfLevels];
            for (int i = 0; i < alpha.Length; i++)
            {
                alpha[i] = Vector.Zero(numberOfLevels);
                alpha[i].SetAllElementsTo(1); // the off-diagonal pseudocount
                alpha[i][i] = 2; // the diagonal pseudocount
                B[i] = Variable.Array<Vector>(judges);
                B[i][judges] = Variable.Dirichlet(alpha[i]).ForEach(judges);
            }
            VariableArray2D<int> R = Variable.Array<int>(judges, docs);
            VariableArray2D<Discrete> RsoftConst = Variable.Constant(Rsoft, judges, docs);
            for (int i = 0; i < numberOfLevels; i++)
            {
                using (Variable.ForEach(docs))
                {
                    using (Variable.Case(Q[docs], i))
                    {
                        R[judges, docs] = Variable.Discrete(B[i][judges]);
                        Variable.ConstrainEqualRandom<int, Discrete>(R[judges, docs], RsoftConst[judges, docs]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            for (int i = 0; i < numberOfLevels; i++)
            {
                Console.WriteLine("Dist over B[" + i + "]:\n" + engine.Infer(B[i]));
            }
            Console.WriteLine("Dist over Q:\n" + engine.Infer(Q));
        }

        private static void WriteMatrix(Dirichlet[] matrix, string filename)
        {
            StreamWriter sw = new StreamWriter(filename);
            for (int i = 0; i < matrix.Length; i++)
            {
                sw.WriteLine(matrix[i].GetMean());
            }
            sw.Close();
        }
    }
}