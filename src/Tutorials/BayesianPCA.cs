// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// Bayesian PCA Model
    /// </summary>
    public class BayesianPCAModel
    {
        // Inference engine
        public InferenceEngine engine = null;

        // Model variables
        public Variable<int> vN = null;
        public Variable<int> vD = null;
        public Variable<int> vM = null;
        public VariableArray2D<double> vData = null;
        public VariableArray2D<double> vW = null;
        public VariableArray2D<double> vZ = null;
        public VariableArray2D<double> vT = null;
        public VariableArray2D<double> vU = null;
        public VariableArray<double> vMu = null;
        public VariableArray<double> vPi = null;
        public VariableArray<double> vAlpha = null;

        // Priors - these are declared as distribution variables
        // so that we can set them at run-time. They are variables
        // from the perspective of the 'Random' factor which takes
        // a distribution as an argument.
        public Variable<Gamma> priorAlpha = null;
        public Variable<Gaussian> priorMu = null;
        public Variable<Gamma> priorPi = null;

        // Model ranges
        public Range rN = null;
        public Range rD = null;
        public Range rM = null;

        /// <summary>
        /// Model constructor
        /// </summary>
        public BayesianPCAModel()
        {
            // The various dimensions will be set externally...
            vN = Variable.New<int>().Named("NumObs");
            vD = Variable.New<int>().Named("NumFeats");
            vM = Variable.New<int>().Named("MaxComponents");
            rN = new Range(vN).Named("N");
            rD = new Range(vD).Named("D");
            rM = new Range(vM).Named("M");

            // ... as will the data
            vData = Variable.Array<double>(rN, rD).Named("data");

            // ... and the priors
            priorAlpha = Variable.New<Gamma>().Named("PriorAlpha");
            priorMu = Variable.New<Gaussian>().Named("PriorMu");
            priorPi = Variable.New<Gamma>().Named("PriorPi");

            // Mixing matrix. Each row is drawn from a Gaussian with zero mean and
            // a precision which will be learnt. This is a form of Automatic
            // Relevance Determination (ARD). The larger the precisions become, the
            // less important that row in the mixing matrix is in explaining the data
            vAlpha = Variable.Array<double>(rM).Named("Alpha");
            vW = Variable.Array<double>(rM, rD).Named("W");
            vAlpha[rM] = Variable.Random<double, Gamma>(priorAlpha).ForEach(rM);
            vW[rM, rD] = Variable.GaussianFromMeanAndPrecision(0, vAlpha[rM]).ForEach(rD);

            // Latent variables are drawn from a standard Gaussian
            vZ = Variable.Array<double>(rN, rM).Named("Z");
            vZ[rN, rM] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(rN, rM);

            // Multiply the latent variables with the mixing matrix...
            vT = Variable.MatrixMultiply(vZ, vW).Named("T");

            // ... add in a bias ...
            vMu = Variable.Array<double>(rD).Named("mu");
            vMu[rD] = Variable.Random<double, Gaussian>(priorMu).ForEach(rD);
            vU = Variable.Array<double>(rN, rD).Named("U");
            vU[rN, rD] = vT[rN, rD] + vMu[rD];

            // ... and add in some observation noise ...
            vPi = Variable.Array<double>(rD).Named("pi");
            vPi[rD] = Variable.Random<double, Gamma>(priorPi).ForEach(rD);

            // ... to give the likelihood of observing the data
            vData[rN, rD] = Variable.GaussianFromMeanAndPrecision(vU[rN, rD], vPi[rD]);

            // Inference engine
            engine = new InferenceEngine();
            return;
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
            bpca.vData.ObservedValue = data;

            // Set the dimensions
            bpca.vN.ObservedValue = data.GetLength(0);
            bpca.vD.ObservedValue = data.GetLength(1);
            bpca.vM.ObservedValue = 6;

            // Set the priors
            bpca.priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 0.01);
            bpca.priorPi.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);
            bpca.priorAlpha.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);

            // Initialize the W marginal to break symmetry
            bpca.vW.InitialiseTo(randomGaussianArray(bpca.vM.ObservedValue, bpca.vD.ObservedValue));

            // Infer the marginals
            bpca.engine.NumberOfIterations = 200;
            Gaussian[,] inferredW = bpca.engine.Infer<Gaussian[,]>(bpca.vW);
            Gaussian[] inferredMu = bpca.engine.Infer<Gaussian[]>(bpca.vMu);
            Gamma[] inferredPi = bpca.engine.Infer<Gamma[]>(bpca.vPi);

            // Print out the results
            Console.WriteLine("Inferred W:");
            printMatrixToConsole(inferredW);
            Console.Write("Mean absolute means of rows in W: ");
            printVectorToConsole(meanAbsoluteRowMeans(inferredW));
            Console.Write("    True bias: ");
            printVectorToConsole(trueMu);
            Console.Write("Inferred bias: ");
            printVectorToConsole(inferredMu);
            Console.Write("    True noise:");
            printVectorToConsole(truePi);
            Console.Write("Inferred noise:");
            printVectorToConsole(inferredPi);
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
        private static IDistribution<double[,]> randomGaussianArray(int row, int col)
        {
            Gaussian[,] array = new Gaussian[row, col];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    array[i, j] = Gaussian.FromMeanAndVariance(Rand.Normal(), 1);
                }
            }

            return Distribution<double>.Array(array);
        }

        /// <summary>
        /// Mean absolute row means
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        private static double[] meanAbsoluteRowMeans(Gaussian[,] matrix)
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
        private static void printMatrixToConsole(Gaussian[,] matrix)
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
        /// Print a 2-D double array to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printMatrixToConsole(double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    Console.Write("{0,5:0.00}\t", matrix[i, j]);
                }

                Console.WriteLine("");
            }
        }

        /// <summary>
        /// Print the means of a 1-D array of Gaussians to the console
        /// </summary>
        /// <param name="vector"></param>
        private static void printVectorToConsole(Gaussian[] vector)
        {
            for (int i = 0; i < vector.GetLength(0); i++)
            {
                Console.Write("{0,5:0.00}\t", vector[i].GetMean());
            }

            Console.WriteLine("");
        }

        /// <summary>
        /// Print the means of a 1-D array of Gammas to the console
        /// </summary>
        /// <param name="vector"></param>
        private static void printVectorToConsole(Gamma[] vector)
        {
            for (int i = 0; i < vector.GetLength(0); i++)
            {
                Console.Write("{0,5:0.00}\t", vector[i].GetMean());
            }

            Console.WriteLine("");
        }

        /// <summary>
        /// Print a 1-D double array to the console
        /// </summary>
        /// <param name="vector"></param>
        private static void printVectorToConsole(double[] vector)
        {
            for (int i = 0; i < vector.GetLength(0); i++)
            {
                Console.Write("{0,5:0.00}\t", vector[i]);
            }

            Console.WriteLine("");
        }
    }
}
