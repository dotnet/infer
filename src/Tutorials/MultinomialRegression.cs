// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "Multinomial regression")]
    public class MultinomialRegression
    {
        /// <summary>
        /// For the multinomial regression model: generate synthetic data,
        /// infer the model parameters and calculate the RMSE between the true
        /// and mean inferred coefficients. 
        /// </summary>
        public void Run()
        {
            // This example requires VMP
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.VariationalMessagePassing))
            {
                Console.WriteLine("This example only runs with Variational Message Passing");
                return;
            }

            int numSamples = 1000;
            int numFeatures = 6;
            int numClasses = 4;
            int countPerSample = 10;
            var features = new Vector[numSamples];
            var counts = new int[numSamples][];
            var coefficients = new Vector[numClasses];
            var mean = Vector.Zero(numClasses);
            Rand.Restart(1);
            for (int i = 0; i < numClasses - 1; i++)
            {
                mean[i] = Rand.Normal();
                coefficients[i] = Vector.Zero(numFeatures);
                Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), coefficients[i]);
            }

            mean[numClasses - 1] = 0;
            coefficients[numClasses - 1] = Vector.Zero(numFeatures);
            for (int i = 0; i < numSamples; i++)
            {
                features[i] = Vector.Zero(numFeatures);
                Rand.Normal(Vector.Zero(numFeatures), PositiveDefiniteMatrix.Identity(numFeatures), features[i]);
                var temp = Vector.FromArray(coefficients.Select(o => o.Inner(features[i])).ToArray());
                var p = MMath.Softmax(temp + mean);
                counts[i] = Rand.Multinomial(countPerSample, p);
            }

            Rand.Restart(DateTime.Now.Millisecond);
            VectorGaussian[] bPost;
            Gaussian[] meanPost;
            MultinomialRegressionModel(features, counts, out bPost, out meanPost);
            var bMeans = bPost.Select(o => o.GetMean()).ToArray();
            var bVars = bPost.Select(o => o.GetVariance()).ToArray();
            double error = 0;
            Console.WriteLine("Coefficients -------------- ");
            for (int i = 0; i < numClasses; i++)
            {
                error += (bMeans[i] - coefficients[i]).Sum(o => o * o);
                Console.WriteLine("Class {0} True {1}", i, coefficients[i]);
                Console.WriteLine("Class {0} Inferred {1}", i, bMeans[i]);
            }

            Console.WriteLine("Mean True " + mean);
            Console.WriteLine("Mean Inferred " + Vector.FromArray(meanPost.Select(o => o.GetMean()).ToArray()));
            error = System.Math.Sqrt(error / (numClasses * numFeatures));
            Console.WriteLine("RMSE is {0}", error);
        }

        /// <summary>
        /// Build and run a multinomial regression model. 
        /// </summary>
        /// <param name="xObs">An array of vectors of observed inputs.
        /// The length of the array is the number of samples, and the
        ///   length of the vectors is the number of input features. </param>
        /// <param name="yObs">An array of array of counts, where the first index is the sample,
        /// and the second index is the class.  </param>
        /// <param name="bPost">The returned posterior over the coefficients.</param>
        /// <param name="meanPost">The returned posterior over the means.</param>
        public void MultinomialRegressionModel(
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
            {
                yData[n] = Variable.Multinomial(trialsCount[n], p[n]);
            }

            // inference
            var ie = new InferenceEngine();
            bPost = ie.Infer<VectorGaussian[]>(B);
            meanPost = ie.Infer<Gaussian[]>(m);
        }
    }
}
