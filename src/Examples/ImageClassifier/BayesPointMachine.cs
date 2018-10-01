// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace ImageClassifier
{
    /// <summary>
    /// Implements classification using a probit Bayes Point Machine.
    /// </summary>
    public class BayesPointMachine
    {
        Variable<int> nTrain, nTest;
        VariableArray<bool> trainingLabels;
        VariableArray<bool> testLabels;
        VariableArray<Vector> trainingItems, testItems;
        Variable<Vector> weights;
        Variable<VectorGaussian> weightPosterior;
        InferenceEngine trainEngine, testEngine;
        bool singleModel = false;

        public BayesPointMachine(int nFeatures, double noise)
        {
            // Training model
            nTrain = Variable.New<int>().Named("nTrain");
            Range trainItem = new Range(nTrain).Named("trainItem");
            trainingLabels = Variable.Array<bool>(trainItem).Named("trainingLabels");
            trainingItems = Variable.Array<Vector>(trainItem).Named("trainingItems");
            weights = Variable.Random(new VectorGaussian(Vector.Zero(nFeatures), PositiveDefiniteMatrix.Identity(nFeatures))).Named("weights");
            trainingLabels[trainItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(weights, trainingItems[trainItem]), noise));

            // Testing model
            nTest = Variable.New<int>().Named("nTest");
            Range testItem = new Range(nTest).Named("testItem");
            testItems = Variable.Array<Vector>(testItem).Named("testItems");
            testLabels = Variable.Array<bool>(testItem).Named("testLabels");
            if (singleModel)
            {
                testLabels[testItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(weights, testItems[testItem]), noise));

                testEngine = new InferenceEngine();
                testEngine.NumberOfIterations = 2;
            }
            else
            {
                weightPosterior = Variable.New<VectorGaussian>().Named("weightPosterior");
                Variable<Vector> testWeights = Variable<Vector>.Random(weightPosterior);
                testLabels[testItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(testWeights, testItems[testItem]), noise));

                trainEngine = new InferenceEngine();
                trainEngine.ShowProgress = false;
                trainEngine.NumberOfIterations = 5;
                testEngine = new InferenceEngine();
                testEngine.ShowProgress = false;
                testEngine.NumberOfIterations = 1;
            }
        }

        public void Train(Vector[] data, bool[] labels)
        {
            nTrain.ObservedValue = data.Length;
            trainingItems.ObservedValue = data;
            trainingLabels.ObservedValue = labels;
            if (!singleModel)
                weightPosterior.ObservedValue = trainEngine.Infer<VectorGaussian>(weights);
        }

        public double[] Test(Vector[] data)
        {
            nTest.ObservedValue = data.Length;
            testItems.ObservedValue = data;
            Bernoulli[] labelPosteriors = testEngine.Infer<Bernoulli[]>(testLabels);
            double[] probs = new double[data.Length];
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] = labelPosteriors[i].GetProbTrue();
            }

            return probs;
        }
    }
}
