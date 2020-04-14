// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using System.IO;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;

    public class MultinomialRegressionBlog
    {
        /// <summary>
        /// Build and run a multinomial regression model. 
        /// </summary>
        /// <param name="xObs">An array of vectors of observed inputs.
        /// The length of the array is the number of samples, and the
        /// length of the vectors is the number of input features. </param>
        /// <param name="yObs">An array of array of counts, where the first index is the sample,
        /// and the second index is the class. </param>
        /// <param name="bPost">The returned posterior over the coefficients.</param>
        /// <param name="meanPost">The returned posterior over the means.</param>
        public void MultinomialRegression(
            Vector[] xObs, int[][] yObs, out VectorGaussian[] bPost, out Gaussian[] meanPost)
        {
            int C = yObs[0].Length;
            int N = xObs.Length;
            int K = xObs[0].Count;
            var c = new Range(C).Named("c");
            var n = new Range(N).Named("n");

            // model
            var B = Variable.Array<Vector>(c).Named("coefficients");
            B[c] = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(K), PositiveDefiniteMatrix.Identity(K)).ForEach(c);
            var m = Variable.Array<double>(c).Named("mean");
            m[c] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(c);
            Variable.ConstrainEqualRandom(B[C - 1], VectorGaussian.PointMass(Vector.Zero(K)));
            Variable.ConstrainEqualRandom(m[C - 1], Gaussian.PointMass(0));
            var x = Variable.Array<Vector>(n);
            x.ObservedValue = xObs;
            var yData = Variable.Array(Variable.Array<int>(c), n);
            yData.ObservedValue = yObs;
            var trialsCount = Variable.Array<int>(n);
            trialsCount.ObservedValue = yObs.Select(o => o.Sum()).ToArray();
            var g = Variable.Array(Variable.Array<double>(c), n);
            g[n][c] = Variable.InnerProduct(B[c], x[n]) + m[c];
            var p = Variable.Array<Vector>(n);
            p[n] = Variable.Softmax(g[n]);
            using (Variable.ForEach(n))
                yData[n] = Variable.Multinomial(trialsCount[n], p[n]);

            // inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.Compiler.GivePriorityTo(typeof (SoftmaxOp_KM11));
            bPost = ie.Infer<VectorGaussian[]>(B);
            meanPost = ie.Infer<Gaussian[]>(m);
        }

        /// <summary>
        /// For the multinomial regression model: generate synthetic data,
        /// infer the model parameters and calculate the RMSE between the true
        /// and mean inferred coefficients. 
        /// </summary>
        /// <param name="numSamples">Number of samples</param>
        /// <param name="numFeatures">Number of input features</param>
        /// <param name="numClasses">Number of classes</param>
        /// <param name="countPerSample">Total count per sample</param>
        /// <returns>RMSE between the true and mean inferred coefficients</returns>
        public double MultinomialRegressionSynthetic(
            int numSamples, int numFeatures, int numClasses, int countPerSample)
        {
            var features = new Vector[numSamples];
            var counts = new int[numSamples][];
            var coefficients = new Vector[numClasses];
            var mean = Vector.Zero(numClasses);
            Rand.Restart(1);
            for (int i = 0; i < numClasses - 1; i++)
            {
                mean[i] = Rand.Normal();
                coefficients[i] = Vector.Zero(numFeatures);
                Rand.Normal(
                    Vector.Zero(numFeatures),
                    PositiveDefiniteMatrix.Identity(numFeatures), coefficients[i]);
            }
            mean[numClasses - 1] = 0;
            coefficients[numClasses - 1] = Vector.Zero(numFeatures);
            for (int i = 0; i < numSamples; i++)
            {
                features[i] = Vector.Zero(numFeatures);
                Rand.Normal(
                    Vector.Zero(numFeatures),
                    PositiveDefiniteMatrix.Identity(numFeatures), features[i]);
                var temp = Vector.FromArray(coefficients.Select(o => o.Inner(features[i])).ToArray());
                var p = MMath.Softmax(temp + mean);
                counts[i] = Rand.Multinomial(countPerSample, p);
            }
            Rand.Restart(DateTime.Now.Millisecond);
            VectorGaussian[] bPost;
            Gaussian[] meanPost;
            MultinomialRegression(features, counts, out bPost, out meanPost);
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
            Console.WriteLine(
                "Inferred " + Vector.FromArray(meanPost.Select(o => o.GetMean()).ToArray()));
            error = System.Math.Sqrt(error / (numClasses * numFeatures));
            Console.WriteLine(numSamples + " " + error);
            return error;
        }

        /// <summary>
        /// Run the synthetic data experiment on a number of different sample sizes. 
        /// </summary>
        /// <param name="numFeatures">Number of input features</param>
        /// <param name="numClasses">Number of classes</param>
        /// <param name="totalCount">Total count per individual</param>
        public void TestMultinomialRegressionSampleSize(
            int numFeatures, int numClasses, int totalCount)
        {
            var sampleSize = new int[]
                {
                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 1500, 2000
                };
            var results = new double[sampleSize.Length];
            for (int i = 0; i < sampleSize.Length; i++)
            {
                results[i] = MultinomialRegressionSynthetic(
                    sampleSize[i], numFeatures, numClasses, totalCount);
            }
            for (int i = 0; i < sampleSize.Length; i++)
            {
                Console.WriteLine(sampleSize[i] + " " + results[i]);
            }
        }
    }

    
    public class MultinomialRegressionTests
    {
        /// <summary>
        /// Build and run a multinomial regression model. 
        /// </summary>
        /// <param name="xObs">An array of vectors of observed inputs.
        /// The length of the array is the number of samples, and the
        /// length of the vectors is the number of input features. </param>
        /// <param name="yObs">An array of array of counts, where the first index is the sample,
        /// and the second index is the class. </param>
        /// <param name="weightsPost">The returned posterior over the coefficients.</param>
        /// <param name="biasPost">The returned posterior over the means.</param>
        public double MultinomialRegression(Vector[] xObs, int[][] yObs, out IList<VectorGaussian> weightsPost, out IList<Gaussian> biasPost, bool trackLowerBound = true)
        {
            int nClasses = yObs[0].Length;
            int nPoints = xObs.Length;
            int dimension = xObs[0].Count;
            var c = new Range(nClasses).Named("c");
            var n = new Range(nPoints).Named("n");
            Variable<bool> ev = null;
            IfBlock model = null;
            if (trackLowerBound)
            {
                ev = Variable.Bernoulli(0.5).Named("evidence");
                model = Variable.If(ev);
            }
            // model
            var weights = Variable.Array<Vector>(c).Named("weights");
            weights[c] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(dimension), PositiveDefiniteMatrix.Identity(dimension)).ForEach(c);
            var bias = Variable.Array<double>(c).Named("bias");
            bias[c] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(c);
            Variable.ConstrainEqualRandom(weights[nClasses - 1], VectorGaussian.PointMass(Vector.Zero(dimension)));
            Variable.ConstrainEqualRandom(bias[nClasses - 1], Gaussian.PointMass(0));
            var x = Variable.Array<Vector>(n).Named("x");
            x.ObservedValue = xObs;
            var yData = Variable.Array(Variable.Array<int>(c), n).Named("y");
            yData.ObservedValue = yObs;
            var trialCounts = Variable.Array<int>(n).Named("trialCounts");
            trialCounts.ObservedValue = yObs.Select(o => o.Sum()).ToArray();
            var g = Variable.Array(Variable.Array<double>(c), n).Named("g");
            g[n][c] = Variable.InnerProduct(weights[c], x[n]) + bias[c];
            var p = Variable.Array<Vector>(n).Named("p");
            p[n] = Variable.Softmax(g[n]);
            using (Variable.ForEach(n))
                yData[n] = Variable.Multinomial(trialCounts[n], p[n]);
            if (trackLowerBound)
                model.CloseBlock();
            // inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            //ie.Compiler.GivePriorityTo(typeof(SaulJordanSoftmaxOp_LBFGS));
            //ie.Compiler.GivePriorityTo(typeof(ProductOfLogisticsSoftmaxOp));
            //ie.Compiler.GivePriorityTo(typeof(SaulJordanSoftmaxOp_NCVMP));
            //ie.Compiler.GivePriorityTo(typeof(SoftmaxOp_Bouchard));
            //ie.Compiler.GivePriorityTo(typeof(Blei06SoftmaxOp_NCVMP));
            ie.OptimiseForVariables = (trackLowerBound ? new IVariable[] {weights, bias, ev} : new IVariable[] {weights, bias});
            if (trackLowerBound) ie.ShowProgress = false;
            for (int iter = 1; iter <= 50; iter++)
            {
                ie.NumberOfIterations = iter;
                if (trackLowerBound)
                    Console.WriteLine(ie.Infer<Bernoulli>(ev).LogOdds);
            }
            weightsPost = ie.Infer<IList<VectorGaussian>>(weights);
            biasPost = ie.Infer<IList<Gaussian>>(bias);
            return trackLowerBound ? ie.Infer<Bernoulli>(ev).LogOdds : 0.0;
        }

        /// <summary>
        /// For the multinomial regression model: generate synthetic data,
        /// infer the model parameters and calculate the RMSE between the true
        /// and mean inferred coefficients. 
        /// </summary>
        /// <param name="numSamples">Number of samples</param>
        /// <param name="numFeatures">Number of input features</param>
        /// <param name="numClasses">Number of classes</param>
        /// <param name="countPerSample">Total count per sample</param>
        /// <returns>RMSE between the true and mean inferred coefficients</returns>
        public double MultinomialRegressionSynthetic(
            int numSamples, int numFeatures, int numClasses, int countPerSample, double noiseVar = 0.0)
        {
            var features = new Vector[numSamples];
            var counts = new int[numSamples][];
            var coefficients = new Vector[numClasses];
            var bias = Vector.Zero(numClasses);
            Rand.Restart(1);
            for (int i = 0; i < numClasses - 1; i++)
            {
                bias[i] = Rand.Normal();
                coefficients[i] = Vector.Zero(numFeatures);
                Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), coefficients[i]);
            }
            bias[numClasses - 1] = 0;
            coefficients[numClasses - 1] = Vector.Zero(numFeatures);
            var noiseDistribution = new VectorGaussian(Vector.Zero(numClasses), PositiveDefiniteMatrix.IdentityScaledBy(numClasses, noiseVar));
            for (int i = 0; i < numSamples; i++)
            {
                features[i] = Vector.Zero(numFeatures);
                Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), features[i]);
                var temp = Vector.FromArray(coefficients.Select(o => o.Inner(features[i])).ToArray());
                if (noiseVar != 0.0)
                    temp += noiseDistribution.Sample();
                var p = MMath.Softmax(temp + bias);
                counts[i] = Rand.Multinomial(countPerSample, p);
            }

            IList<VectorGaussian> weightsPost;
            IList<Gaussian> biasPost;
            bool trackLowerBound = true;
            double ev = MultinomialRegression(features, counts, out weightsPost, out biasPost, trackLowerBound);
            if (trackLowerBound) Console.WriteLine("Log lower bound= " + ev);
            double error = 0;
            Console.WriteLine("Weights -------------- ");
            for (int i = 0; i < numClasses; i++)
            {
                var bMean = weightsPost[i].GetMean();
                error += (bMean - coefficients[i]).Sum(o => o*o);
                Console.WriteLine("Class " + i + " True " + coefficients[i]);
                Console.WriteLine("Class " + i + " Inferred " + bMean);
            }
            error = System.Math.Sqrt(error / (numClasses * numFeatures));
            Console.WriteLine("RMSE " + error);
            Console.WriteLine("Bias -------------- ");
            Console.WriteLine("True " + bias);
            Console.WriteLine("Inferred " + Vector.FromArray(biasPost.Select(o => o.GetMean()).ToArray()));
            return error;
        }

        [Fact]
        public void MultinomialRegressionTest()
        {
            double error = MultinomialRegressionSynthetic(100, 4, 3, 10);
            Assert.True(error < 0.13);
        }

        /// <summary>
        /// Run the synthetic data experiment on a number of different sample sizes. 
        /// </summary>
        /// <param name="numFeatures">Number of input features</param>
        /// <param name="numClasses">Number of classes</param>
        /// <param name="totalCount">Total count per individual</param>
        internal void TestMultinomialRegressionSampleSize(
            int numFeatures, int numClasses, int totalCount)
        {
            var sampleSize = new int[]
                {
                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 1500, 2000
                };
            var results = new double[sampleSize.Length];
            for (int i = 0; i < sampleSize.Length; i++)
            {
                results[i] = MultinomialRegressionSynthetic(sampleSize[i], numFeatures, numClasses, totalCount);
            }
            for (int i = 0; i < sampleSize.Length; i++)
            {
                Console.WriteLine(sampleSize[i] + " " + results[i]);
            }
        }

        internal void TestMultinomialRegressionNoiseVariance()
        {
            var noiseVariance = new double[] {0.0, 1e-3, 1e-2, 1e-1, .5, 1.0, 2.0, 3.0, 5.0};
            var results = new double[noiseVariance.Length];
            for (int i = 0; i < noiseVariance.Length; i++)
            {
                results[i] = MultinomialRegressionSynthetic(100, 10, 3, 10, noiseVariance[i]);
            }
            for (int i = 0; i < noiseVariance.Length; i++)
            {
                Console.WriteLine(noiseVariance[i] + " " + results[i]);
            }
        }

        internal void MultinomialRegression454()
        {
            char[] sep = {'\t', ','};
            var x = new List<Vector>();
            var y = new List<int[]>();
            int numFeatures = 6;
            int numClasses = 8;
            var key = new Dictionary<string, int> {{"A", 0}, {"G", 1}, {"T", 2}, {"C", 3}};
            //TODO: change path
            using (var mySR = new StreamReader(@"C:\Users\v-daknow\Documents\svn\svn\projects\454\pickles\C2_errorMatrix.txt"))
            {
                mySR.ReadLine(); // Skip over header line
                string myStr;
                while ((myStr = mySR.ReadLine()) != null)
                {
                    var mySplitStr = myStr.Split(sep);
                    var newx = Vector.Zero(numFeatures);
                    var newy = new int[numClasses];
                    if (mySplitStr[1] != "C") // take C as baseline
                        newx[key[mySplitStr[1]]] = 1.0;
                    newx[3] = Double.Parse(mySplitStr[2]);
                    newx[4] = Double.Parse(mySplitStr[3]);
                    newx[5] = Double.Parse(mySplitStr[4]);
                    for (int i = 0; i < numClasses; i++)
                    {
                        newy[i] = Int32.Parse(mySplitStr[5 + i]);
                    }
                    x.Add(newx);
                    y.Add(newy);
                }
            }
            var xObs = x.ToArray();
            var yObs = y.ToArray();
            var means = new double[3];
            var stds = new double[3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < xObs.Length; j++)
                {
                    means[i] += xObs[j][3 + i];
                    stds[i] += xObs[j][3 + i]*xObs[j][3 + i];
                }
                means[i] /= xObs.Length;
                stds[i] = System.Math.Sqrt(stds[i] / xObs.Length - means[i] * means[i]);
                for (int j = 0; j < xObs.Length; j++)
                {
                    xObs[j][3 + i] = (xObs[j][3 + i] - means[i])/stds[i];
                }
            }

            IList<VectorGaussian> bPost;
            IList<Gaussian> meanPost;
            MultinomialRegression(xObs, yObs, out bPost, out meanPost);
            var bMeans = bPost.Select(o => o.GetMean()).ToArray();
            var bVars = bPost.Select(o => o.GetVariance()).ToArray();

            Console.WriteLine("Class Mean A G T Coverage Homo Qual");
            var classNames = "A G T C N d i correct".Split(' ');
            for (int i = 0; i < numClasses; i++)
            {
                Console.Write(classNames[i] + " " + meanPost[i].GetMean() + " ");
                Console.WriteLine(bMeans[i]);
            }
            Console.WriteLine("Variances----------------------");
            Console.WriteLine("Class Mean A G T Coverage Homo Qual");

            for (int i = 0; i < numClasses; i++)
            {
                Console.Write(classNames[i] + " " + System.Math.Sqrt(meanPost[i].GetVariance()) + " ");
                for (int j = 0; j < numFeatures; j++)
                {
                    Console.Write(System.Math.Sqrt(bVars[i][j, j]) + " ");
                }

                Console.WriteLine();
            }
        }
    }
}