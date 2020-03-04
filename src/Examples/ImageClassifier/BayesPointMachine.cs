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
        InferenceEngine engine;
        /// <summary>
        /// If true, train and test data will be part of a single model.  This is less efficient, but simpler.
        /// </summary>
        readonly bool singleModel = false;

        public BayesPointMachine(int nFeatures, double noise)
        {
            // Training model
            nTrain = Variable.Observed(default(int)).Named(nameof(nTrain));
            Range trainItem = new Range(nTrain);
            trainItem.Name = nameof(trainItem);
            trainingLabels = Variable.Observed(default(bool[]), trainItem).Named(nameof(trainingLabels));
            trainingItems = Variable.Observed(default(Vector[]), trainItem).Named(nameof(trainingItems));
            weights = Variable.Random(new VectorGaussian(Vector.Zero(nFeatures), PositiveDefiniteMatrix.Identity(nFeatures))).Named(nameof(weights));
            trainingLabels[trainItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(weights, trainingItems[trainItem]), noise));

            // Testing model
            nTest = Variable.Observed(default(int)).Named(nameof(nTest));
            Range testItem = new Range(nTest);
            testItem.Name = nameof(testItem);
            testItems = Variable.Observed(default(Vector[]), testItem).Named(nameof(testItems));
            testLabels = Variable.Array<bool>(testItem).Named(nameof(testLabels));
            engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.WriteSourceFiles = false;
            engine.NumberOfIterations = 5;
            if (singleModel)
            {
                testLabels[testItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(weights, testItems[testItem]), noise));
            }
            else
            {
                weightPosterior = Variable.Observed(default(VectorGaussian)).Named(nameof(weightPosterior));
                Variable<Vector> testWeights = Variable<Vector>.Random(weightPosterior);
                testLabels[testItem] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(testWeights, testItems[testItem]), noise));

                // Force compilation of the training model.
                engine.GetCompiledInferenceAlgorithm(weights);
            }
            // Force compilation of the testing model.
            // This also defines the variables to be inferred, obviating OptimizeForVariables.
            // This requires observed variables to have values, but they can be null.
            engine.GetCompiledInferenceAlgorithm(testLabels);
        }

        public void Train(Vector[] data, bool[] labels)
        {
            nTrain.ObservedValue = data.Length;
            trainingItems.ObservedValue = data;
            trainingLabels.ObservedValue = labels;
            if (!singleModel)
                weightPosterior.ObservedValue = engine.Infer<VectorGaussian>(weights);
        }

        public double[] Test(Vector[] data)
        {
            nTest.ObservedValue = data.Length;
            testItems.ObservedValue = data;
            Bernoulli[] labelPosteriors = engine.Infer<Bernoulli[]>(testLabels);
            double[] probs = new double[data.Length];
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] = labelPosteriors[i].GetProbTrue();
            }

            return probs;
        }
    }
}
