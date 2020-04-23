// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using System.IO;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class MulticlassRegressionTests
    {
        public int MulticlassRegression(Vector[] xObs, int[] yObs, int C, out VectorGaussian[] bPost, out Gaussian[] meanPost, out double lowerBound, object softmaxOperator,
                                        bool trackLowerBound = false)
        {
            int N = xObs.Length;
            int K = xObs[0].Count;
            var c = new Range(C).Named("c");
            var n = new Range(N).Named("n");
            Variable<bool> ev = null;
            IfBlock model = null;
            if (trackLowerBound)
            {
                ev = Variable.Bernoulli(0.5).Named("evidence");
                model = Variable.If(ev);
            }
            // model
            var B = Variable.Array<Vector>(c).Named("coefficients");
            B[c] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(K), PositiveDefiniteMatrix.Identity(K)).ForEach(c);
            var m = Variable.Array<double>(c).Named("mean");
            m[c] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(c);
            Variable.ConstrainEqualRandom(B[C - 1], VectorGaussian.PointMass(Vector.Zero(K)));
            Variable.ConstrainEqualRandom(m[C - 1], Gaussian.PointMass(0));
            var x = Variable.Array<Vector>(n);
            x.ObservedValue = xObs;
            var yData = Variable.Array<int>(n);
            yData.ObservedValue = yObs;
            var g = Variable.Array(Variable.Array<double>(c), n);
            g[n][c] = Variable.InnerProduct(B[c], x[n]) + m[c];
            var p = Variable.Array<Vector>(n);
            p[n] = Variable.Softmax(g[n]);
            using (Variable.ForEach(n))
                yData[n] = Variable.Discrete(p[n]);
            if (trackLowerBound)
                model.CloseBlock();
            // inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(softmaxOperator);
            var ca = ie.GetCompiledInferenceAlgorithm(ev, B, m);
            var start = DateTime.Now;
            var initBound = double.NegativeInfinity;
            int i = 0;
            lowerBound = 0;
            for (i = 1; i <= 50; i++)
            {
                ca.Update(1);
                lowerBound = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
                Console.WriteLine(i + "," + lowerBound + "," + (DateTime.Now - start).TotalMilliseconds);
                if (System.Math.Abs(initBound - lowerBound) < 1e-2)
                    break;
                initBound = lowerBound;
                start = DateTime.Now;
            }

            bPost = ca.Marginal<VectorGaussian[]>(B.NameInGeneratedCode);
            meanPost = ca.Marginal<Gaussian[]>(m.NameInGeneratedCode);
            return i;
        }

        public double MulticlassRegressionSynthetic(int numSamples, object softmaxOperator, out int iterations, out double lowerBound, double noiseVar = 0.0)
        {
            int numFeatures = 6;
            int numClasses = 4;
            var features = new Vector[numSamples];
            var counts = new int[numSamples];
            var coefficients = new Vector[numClasses];
            var mean = Vector.Zero(numClasses);

            for (int i = 0; i < numClasses - 1; i++)
            {
                mean[i] = Rand.Normal();
                coefficients[i] = Vector.Zero(numFeatures);
                Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), coefficients[i]);
            }
            mean[numClasses - 1] = 0;
            coefficients[numClasses - 1] = Vector.Zero(numFeatures);
            var noiseDistribution = new VectorGaussian(Vector.Zero(numClasses), PositiveDefiniteMatrix.IdentityScaledBy(numClasses, noiseVar));
            for (int i = 0; i < numSamples; i++)
            {
                features[i] = Vector.Zero(numFeatures);
                Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), features[i]);
                var temp = Vector.FromArray(coefficients.Select(o => o.Inner(features[i])).ToArray());
                if (noiseVar != 0.0)
                    temp += noiseDistribution.Sample();
                var p = MMath.Softmax(temp + mean);
                counts[i] = Rand.Sample(p);
            }

            Rand.Restart(DateTime.Now.Millisecond);
            VectorGaussian[] bPost;
            Gaussian[] meanPost;
            iterations = MulticlassRegression(features, counts, numClasses, out bPost, out meanPost, out lowerBound, softmaxOperator, true);
            var bMeans = bPost.Select(o => o.GetMean()).ToArray();
            var bVars = bPost.Select(o => o.GetVariance()).ToArray();
            double error = 0;
            Console.WriteLine("Coefficients -------------- ");
            for (int i = 0; i < numClasses; i++)
            {
                error += (bMeans[i] - coefficients[i]).Sum(o => o*o);
                Console.WriteLine("True " + coefficients[i]);
                Console.WriteLine("Inferred " + bMeans[i]);
            }
            Console.WriteLine("Mean -------------- ");
            Console.WriteLine("True " + mean);
            Console.WriteLine("Inferred " + Vector.FromArray(meanPost.Select(o => o.GetMean()).ToArray()));

            error = System.Math.Sqrt(error / (numClasses * numFeatures));
            Console.WriteLine(numSamples + " " + error);
            return error;
        }

        private object[] softmaxOperators = new object[]
            {
                typeof (SoftmaxOp_KM11_Sparse2),
                //typeof(ProductOfLogisticsSoftmaxOp),
                typeof (SoftmaxOp_KM11),
                typeof (SoftmaxOp_Bouchard),
                typeof (SoftmaxOp_BL06),
                typeof (SoftmaxOp_Bohning),
                typeof (SoftmaxOp_Taylor)
            };


        public void TestMulticlassRegressionSampleSize()
        {
            var sampleSize = new int[] {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 1000, 2000};
            var results = new double[sampleSize.Length][];
            var iterations = new int[sampleSize.Length][];
            var lowerBounds = new double[sampleSize.Length][];
            //for (int i = 0; i < sampleSize.Length; i++)
            Parallel.For(0, sampleSize.Length, i =>
                {
                    results[i] = new double[softmaxOperators.Length];
                    iterations[i] = new int[softmaxOperators.Length];
                    lowerBounds[i] = new double[softmaxOperators.Length];
                    for (int j = 0; j < softmaxOperators.Length; j++)
                    {
                        Rand.Restart(i);
                        results[i][j] = MulticlassRegressionSynthetic(sampleSize[i], softmaxOperators[j], out iterations[i][j], out lowerBounds[i][j]);
                    }
                });
            Console.WriteLine("N," + softmaxOperators.Select(o => o.ToString()).Aggregate((p, q) => p + "," + q));
            for (int i = 0; i < sampleSize.Length; i++)
            {
                Console.WriteLine(sampleSize[i] + "," + results[i].Select(o => o.ToString(CultureInfo.InvariantCulture)).Aggregate((p, q) => p + "," + q));
            }
            Console.WriteLine();
            Console.WriteLine("N," + softmaxOperators.Select(o => o.ToString()).Aggregate((p, q) => p + "," + q));
            for (int i = 0; i < sampleSize.Length; i++)
            {
                Console.WriteLine(sampleSize[i] + "," + iterations[i].Select(o => o.ToString(CultureInfo.InvariantCulture)).Aggregate((p, q) => p + "," + q));
            }
        }

        public static void WriteToFile<T>(string fileName, T[][] results, string[] rownames)
        {
            using (var sw = new StreamWriter(fileName))
            {
                for (int i = 0; i < rownames.Length; i++)
                {
                    sw.WriteLine(rownames[i] + "," + results[i].Select(o => o.ToString()).Aggregate((p, q) => p + "," + q));
                }
            }
        }

        public void TestMulticlassRegressionSampleSize(int repeats, string baseDir)
        {
            Directory.CreateDirectory(baseDir);
            var sampleSize = new int[] {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 1000, 2000};
            var results = Enumerable.Range(0, softmaxOperators.Length)
                                    .Select(o => Enumerable.Range(0, sampleSize.Length)
                                                           .Select(p => Enumerable.Range(0, repeats)
                                                                                  .Select(q => 0.0).ToArray()).ToArray()).ToArray();
            var iterations = Enumerable.Range(0, softmaxOperators.Length)
                                       .Select(o => Enumerable.Range(0, sampleSize.Length)
                                                              .Select(p => Enumerable.Range(0, repeats)
                                                                                     .Select(q => 0).ToArray()).ToArray()).ToArray();
            var lowerBound = Enumerable.Range(0, softmaxOperators.Length)
                                       .Select(o => Enumerable.Range(0, sampleSize.Length)
                                                              .Select(p => Enumerable.Range(0, repeats)
                                                                                     .Select(q => 0.0).ToArray()).ToArray()).ToArray();
            //for (int i = 0; i < sampleSize.Length; i++)
            Parallel.For(0, sampleSize.Length, i =>
                {
                    for (int r = 0; r < repeats; r++)
                    {
                        for (int j = 0; j < softmaxOperators.Length; j++)
                        {
                            Rand.Restart(i*repeats + r);
                            results[j][i][r] = MulticlassRegressionSynthetic(sampleSize[i], softmaxOperators[j], out iterations[j][i][r], out lowerBound[j][i][r]);
                        }
                    }
                });
            for (int j = 0; j < softmaxOperators.Length; j++)
            {
                var opName = softmaxOperators[j].ToString().Split('.').Last();
                WriteToFile(baseDir + "multinomial_error_" + opName + ".csv",
                            results[j], sampleSize.Select(o => o.ToString(CultureInfo.InvariantCulture)).ToArray());
                WriteToFile(baseDir + "multinomial_iterations_" + opName + ".csv",
                            iterations[j], sampleSize.Select(o => o.ToString(CultureInfo.InvariantCulture)).ToArray());
                WriteToFile(baseDir + "multinomial_lowerBound_" + opName + ".csv",
                            lowerBound[j], sampleSize.Select(o => o.ToString(CultureInfo.InvariantCulture)).ToArray());
            }
        }

        public void TestMulticlassRegressionNoiseVariance(int repeats, string baseDir)
        {
            Directory.CreateDirectory(baseDir);
            var noiseVariance = new double[] {1e-3, 1e-2, 1e-1, .5, 1.0, 2.0, 3.0, 5.0};
            var results = Enumerable.Range(0, softmaxOperators.Length)
                                    .Select(o => Enumerable.Range(0, noiseVariance.Length)
                                                           .Select(p => Enumerable.Range(0, repeats)
                                                                                  .Select(q => 0.0).ToArray()).ToArray()).ToArray();
            var iterations = Enumerable.Range(0, softmaxOperators.Length)
                                       .Select(o => Enumerable.Range(0, noiseVariance.Length)
                                                              .Select(p => Enumerable.Range(0, repeats)
                                                                                     .Select(q => 0).ToArray()).ToArray()).ToArray();
            var lowerBound = Enumerable.Range(0, softmaxOperators.Length)
                                       .Select(o => Enumerable.Range(0, noiseVariance.Length)
                                                              .Select(p => Enumerable.Range(0, repeats)
                                                                                     .Select(q => 0.0).ToArray()).ToArray()).ToArray();
            //for (int i = 0; i < sampleSize.Length; i++)
            Parallel.For(0, noiseVariance.Length, i =>
                {
                    for (int r = 0; r < repeats; r++)
                    {
                        for (int j = 0; j < softmaxOperators.Length; j++)
                        {
                            Rand.Restart(i*repeats + r);
                            results[j][i][r] = MulticlassRegressionSynthetic(200, softmaxOperators[j], out iterations[j][i][r], out lowerBound[j][i][r], noiseVariance[i]);
                        }
                    }
                });
            for (int j = 0; j < softmaxOperators.Length; j++)
            {
                var opName = softmaxOperators[j].ToString().Split('.').Last();
                WriteToFile(baseDir + "multinomial_error_" + opName + ".csv",
                            results[j], noiseVariance.Select(o => o.ToString(CultureInfo.InvariantCulture)).ToArray());
                WriteToFile(baseDir + "multinomial_iterations_" + opName + ".csv",
                            iterations[j], noiseVariance.Select(o => o.ToString(CultureInfo.InvariantCulture)).ToArray());
                WriteToFile(baseDir + "multinomial_lowerBound_" + opName + ".csv",
                            lowerBound[j], noiseVariance.Select(o => o.ToString(CultureInfo.InvariantCulture)).ToArray());
            }
        }

        public void TestMulticlassRegressionNoiseVariance()
        {
            SoftmaxOp_KM11.useBounds = true;
            var noiseVariance = new double[] {0.0, 1e-3, 1e-2, 1e-1, .5, 1.0, 2.0, 3.0, 5.0};
            var results = new double[noiseVariance.Length][];
            var iterations = new int[noiseVariance.Length][];
            var lowerBound = new double[noiseVariance.Length][];
            //for (int i = 0; i < noiseVariance.Length; i++)
            Parallel.For(0, noiseVariance.Length, i =>
                {
                    results[i] = new double[softmaxOperators.Length];
                    iterations[i] = new int[softmaxOperators.Length];
                    for (int j = 0; j < softmaxOperators.Length; j++)
                    {
                        results[i][j] = MulticlassRegressionSynthetic(100, softmaxOperators[j], out iterations[i][j], out lowerBound[i][j], noiseVariance[i]);
                    }
                });
            Console.WriteLine("noise," + softmaxOperators.Select(o => o.ToString()).Aggregate((p, q) => p + "," + q));
            for (int i = 0; i < noiseVariance.Length; i++)
            {
                Console.WriteLine(noiseVariance[i] + "," + results[i].Select(o => o.ToString(CultureInfo.InvariantCulture)).Aggregate((p, q) => p + "," + q));
            }
            Console.WriteLine();
            Console.WriteLine("N," + softmaxOperators.Select(o => o.ToString()).Aggregate((p, q) => p + "," + q));
            for (int i = 0; i < noiseVariance.Length; i++)
            {
                Console.WriteLine(noiseVariance[i] + "," + iterations[i].Select(o => o.ToString(CultureInfo.InvariantCulture)).Aggregate((p, q) => p + "," + q));
            }
        }


        public void IndexOfMaximumDiverges()
        {
            int numberOfFeatures = 10;
            int numberOfFolders = 3;
            int numEmails = 1000;
            var featuresCountsObs = Enumerable.Range(0, numEmails).Select(o => Rand.Binomial(numberOfFeatures, 5.0/numberOfFeatures)).ToArray();
            var featureIndicesObs = featuresCountsObs.Select(o => Rand.Perm(numberOfFeatures).ToList().GetRange(0, o).ToArray()).ToArray();
            //var trueFeatureWeights = Enumerable.Range(0, numberOfFolders).Select(p=>Enumerable.Range(0, numberOfFeatures).Select(o => Rand.Normal()).ToArray()).ToArray(); 
            //var folders = featureIndicesObs.Select(fi=>fi.Select(p=>trueFeatureWeights.Select(q=>q[p]).Sum()
            // random data for now!
            var folders = Enumerable.Range(0, numEmails).Select(o => Rand.Int(numberOfFolders)).ToArray();

            Range numberOfFeaturesRange = new Range(numberOfFeatures).Named("NumberOfFeaturesRange");
            numberOfFeaturesRange.AddAttribute(new Sequential()); // This requires new build of Infer.NET

            // creat a range for the number of classes
            Range numberOfClases = new Range(numberOfFolders).Named("NumberOfClassesRange");

            // Model the total number of items
            var numberOfItems = Variable.New<int>().Named("numberOfItems");
            numberOfItems.ObservedValue = numEmails;
            Range numberOfItemsRange = new Range(numberOfItems).Named("numberOfItemsRange");

            numberOfItemsRange.AddAttribute(new Sequential());

            // Model the number features present in each item in each class
            var featureCounts = Variable.Array<int>(numberOfItemsRange).Named("featureCounts");
            Range featureCountItemRange = new Range(featureCounts[numberOfItemsRange]).Named("featureItemCountRange");
            featureCounts.ObservedValue = featuresCountsObs;

            // Model the features we observe
            var featureIndicies = Variable.Array(Variable.Array<int>(featureCountItemRange), numberOfItemsRange).Named("featureIndicies");
            featureIndicies.ObservedValue = featureIndicesObs;
            // Setup the priors
            var FeatureWeights = Variable.Array(Variable.Array<double>(numberOfFeaturesRange), numberOfClases).Named("FeatureWeights");
            FeatureWeights[numberOfClases][numberOfFeaturesRange] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(numberOfClases, numberOfFeaturesRange);

            // Setup the label value (Folder)
            var folderValue = Variable.Array<int>(numberOfItemsRange).Named("folderValue");
            folderValue.ObservedValue = folders;

            var sparseWeightVector =
                Variable.Array(Variable.Array(Variable.Array<double>(featureCountItemRange), numberOfClases), numberOfItemsRange).Named("sparseWeightVector");
            ;
            sparseWeightVector[numberOfItemsRange][numberOfClases] = Variable.Subarray<double>(FeatureWeights[numberOfClases], featureIndicies[numberOfItemsRange]);
            var scoresWithNoise = Variable.Array(Variable.Array<double>(numberOfClases), numberOfItemsRange).Named("scoresWithNoise");
            scoresWithNoise[numberOfItemsRange][numberOfClases] = Variable.GaussianFromMeanAndVariance(Variable.Sum(sparseWeightVector[numberOfItemsRange][numberOfClases]), 1);
            folderValue[numberOfItemsRange] = Variable<int>.Factor(MMath.IndexOfMaximumDouble, scoresWithNoise[numberOfItemsRange]);
            folderValue.AddAttribute(new MarginalPrototype(Discrete.Uniform(numberOfClases.SizeAsInt)));

            var ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(FeatureWeights));
        }
    }
}