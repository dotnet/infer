// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// Bayesian PCA Model
    /// </summary>
    public class BayesianPCAModel
    {
        // Inference engine
        public InferenceEngine engine;

        // Model variables
        public Variable<int> observationCount;
        public Variable<int> featureCount;
        public Variable<int> componentCount;
        public VariableArray2D<double> data;
        public VariableArray2D<double> W;
        public VariableArray2D<Gaussian> initW;
        public VariableArray2D<double> Z;
        public VariableArray2D<double> T;
        public VariableArray2D<double> U;
        public VariableArray<double> mu;
        public VariableArray<double> pi;
        public VariableArray<double> alpha;

        // Priors - these are declared as distribution variables
        // so that we can set them at run-time. They are variables
        // from the perspective of the 'Random' factor which takes
        // a distribution as an argument.
        public Variable<Gamma> priorAlpha;
        public Variable<Gaussian> priorMu;
        public Variable<Gamma> priorPi;

        // Model ranges
        public Range observation;
        public Range feature;
        public Range component;

        /// <summary>
        /// Model constructor
        /// </summary>
        public BayesianPCAModel()
        {
            // The various dimensions will be set externally...
            observationCount = Variable.New<int>().Named(nameof(observationCount));
            featureCount = Variable.New<int>().Named(nameof(featureCount));
            componentCount = Variable.New<int>().Named(nameof(componentCount));
            observation = new Range(observationCount).Named(nameof(observation));
            feature = new Range(featureCount).Named(nameof(feature));
            component = new Range(componentCount).Named(nameof(component));

            // ... as will the data
            data = Variable.Array<double>(observation, feature).Named(nameof(data));

            // ... and the priors
            priorAlpha = Variable.New<Gamma>().Named(nameof(priorAlpha));
            priorMu = Variable.New<Gaussian>().Named(nameof(priorMu));
            priorPi = Variable.New<Gamma>().Named(nameof(priorPi));

            // Mixing matrix. Each row is drawn from a Gaussian with zero mean and
            // a precision which will be learnt. This is a form of Automatic
            // Relevance Determination (ARD). The larger the precisions become, the
            // less important that row in the mixing matrix is in explaining the data
            alpha = Variable.Array<double>(component).Named(nameof(alpha));
            W = Variable.Array<double>(component, feature).Named(nameof(W));
            alpha[component] = Variable<double>.Random(priorAlpha).ForEach(component);
            W[component, feature] = Variable.GaussianFromMeanAndPrecision(0, alpha[component]).ForEach(feature);
            // Initialize the W marginal to break symmetry
            initW = Variable.Array<Gaussian>(component, feature).Named(nameof(initW));
            W[component, feature].InitialiseTo(initW[component, feature]);

            // Latent variables are drawn from a standard Gaussian
            Z = Variable.Array<double>(observation, component).Named(nameof(Z));
            Z[observation, component] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(observation, component);

            // Multiply the latent variables with the mixing matrix...
            T = Variable.MatrixMultiply(Z, W).Named(nameof(T));

            // ... add in a bias ...
            mu = Variable.Array<double>(feature).Named(nameof(mu));
            mu[feature] = Variable<double>.Random(priorMu).ForEach(feature);
            U = Variable.Array<double>(observation, feature).Named(nameof(U));
            U[observation, feature] = T[observation, feature] + mu[feature];

            // ... and add in some observation noise ...
            pi = Variable.Array<double>(feature).Named(nameof(pi));
            pi[feature] = Variable<double>.Random(priorPi).ForEach(feature);

            // ... to give the likelihood of observing the data
            data[observation, feature] = Variable.GaussianFromMeanAndPrecision(U[observation, feature], pi[feature]);

            // Inference engine
            engine = new InferenceEngine();
        }
    }

    /// <summary>
    /// Run a Bayesian PCA example
    /// </summary>
    [Example("Applications", "A Bayesian Principal Components Analysis example")]
    public class BayesianPCA
    {
        public void Run()
        {
            BayesianPCAModel bpca = new BayesianPCAModel();
            if (!(bpca.engine.Algorithm is Algorithms.VariationalMessagePassing))
            {
                Console.WriteLine("This example only runs with Variational Message Passing");
                return;
            }

            // Set a stable random number seed for repeatable runs
            Rand.Restart(12347);
            double[,] data = generateData(1000);

            // Set the data
            bpca.data.ObservedValue = data;

            // Set the dimensions
            bpca.observationCount.ObservedValue = data.GetLength(0);
            bpca.featureCount.ObservedValue = data.GetLength(1);
            bpca.componentCount.ObservedValue = 6;

            // Set the priors
            bpca.priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 0.01);
            bpca.priorPi.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);
            bpca.priorAlpha.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);

            // Set the initialization
            bpca.initW.ObservedValue = randomGaussianArray(bpca.componentCount.ObservedValue, bpca.featureCount.ObservedValue);

            // Infer the marginals
            bpca.engine.NumberOfIterations = 200;
            var inferredW = bpca.engine.Infer<IArray2D<Gaussian>>(bpca.W);
            var inferredMu = bpca.engine.Infer<IReadOnlyList<Gaussian>>(bpca.mu);
            var inferredPi = bpca.engine.Infer<IReadOnlyList<Gamma>>(bpca.pi);

            // Print out the results
            Console.WriteLine("Inferred W:");
            printMatrixToConsole(inferredW);
            Console.Write("Mean absolute means of rows in W: ");
            printVectorToConsole(meanAbsoluteRowMeans(inferredW));
            Console.Write("    True bias: ");
            printVectorToConsole(trueMu);
            Console.Write("Inferred bias: ");
            printVectorToConsole(inferredMu.Select(d => d.GetMean()));
            Console.Write("    True noise:");
            printVectorToConsole(truePi);
            Console.Write("Inferred noise:");
            printVectorToConsole(inferredPi.Select(d => d.GetMean()));
            Console.WriteLine();
        }

        /// <summary>
        /// True W. Inference will find a different basis
        /// </summary>
        static double[,] trueW =
        {
          { -0.30, 0.40, 0.20, -0.15, 0.20, -0.25, -0.50, -0.10, -0.25, 0.10 },
          { -0.10, -0.20, 0.40, 0.50, 0.15, -0.35, 0.05, 0.20, 0.20, -0.15 },
          { 0.15, 0.05, 0.15, -0.10, -0.15, 0.25, -0.10, 0.15, -0.30, -0.55 },
        };

        /// <summary>
        /// True bias
        /// </summary>
        static double[] trueMu = { -0.95, 0.75, -0.20, 0.20, 0.30, -0.35, 0.65, 0.20, 0.25, 0.40 };

        /// <summary>
        /// True observation noise
        /// </summary>
        static double[] truePi = { 8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0 };

        /// <summary>
        /// Generate data from the true model
        /// </summary>
        /// <param name="numObs">Number of observations</param>
        static double[,] generateData(int numObs)
        {
            int numComp = trueW.GetLength(0);
            int numFeat = trueW.GetLength(1);
            double[,] data = new double[numObs, numFeat];
            Matrix WMat = new Matrix(trueW);
            Vector z = Vector.Zero(numComp);
            for (int i = 0; i < numObs; i++)
            {
                // Sample scores from standard Gaussian
                for (int j = 0; j < numComp; j++)
                {
                    z[j] = Gaussian.Sample(0.0, 1.0);
                }

                // Mix the components with the true mixture matrix
                Vector t = z * WMat;
                for (int j = 0; j < numFeat; j++)
                {
                    // Add in the bias
                    double u = t[j] + trueMu[j];

                    // ... and the noise
                    data[i, j] = Gaussian.Sample(u, truePi[j]);
                }
            }
            return data;
        }

        /// <summary>
        /// Create an array of Gaussian distributions with random mean and unit variance
        /// </summary>
        /// <param name="row">Number of rows</param>
        /// <param name="col">Number of columns</param>
        /// <returns>The array as a distribution over a 2-D double array domain</returns>
        private static Gaussian[,] randomGaussianArray(int row, int col)
        {
            return Util.ArrayInit(row, col, (i, j) => Gaussian.FromMeanAndVariance(Rand.Normal(), 1));
        }

        /// <summary>
        /// Mean absolute row means
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        private static double[] meanAbsoluteRowMeans(IArray2D<Gaussian> matrix)
        {
            double[] mam = new double[matrix.GetLength(0)];
            double mult = 1.0 / ((double)matrix.GetLength(1));
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                double sum = 0.0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += System.Math.Abs(matrix[i, j].GetMean());
                }
                mam[i] = mult * sum;
            }
            return mam;
        }

        /// <summary>
        /// Print the means of a 2-D array of Gaussians to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printMatrixToConsole(IArray2D<Gaussian> matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    Console.Write("{0,5:0.00}\t", matrix[i, j].GetMean());
                }
                Console.WriteLine("");
            }
        }

        /// <summary>
        /// Print a 1-D double array to the console
        /// </summary>
        /// <param name="vector"></param>
        private static void printVectorToConsole(IEnumerable<double> vector)
        {
            foreach (var item in vector)
            {
                Console.Write("{0,5:0.00}\t", item);
            }
            Console.WriteLine("");
        }
    }
}
