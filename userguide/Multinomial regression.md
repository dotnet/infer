---
layout: default
---

[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md) : [The softamx factor](The softmax factor.md)

## Multinomial regression

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace MultinomialRegressionBlog
{
  class MultinomialRegression
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
      ie.Compiler.GivePriorityTo(typeof(SaulJordanSoftmaxOp_NCVMP));
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
        error += (bMeans[i] - coefficients[i]).Sum(o => o * o);
        Console.WriteLine("True " + coefficients[i]);
        Console.WriteLine("Inferred " + bMeans[i]);
      }
      Console.WriteLine("Mean -------------- ");
      Console.WriteLine("True " + mean);
      Console.WriteLine(
        "Inferred " + Vector.FromArray(meanPost.Select(o => o.GetMean()).ToArray()));
      error = Math.Sqrt(error / (numClasses * numFeatures));
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
      var sampleSize = new int[] {
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 1500, 2000 };
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
}
```