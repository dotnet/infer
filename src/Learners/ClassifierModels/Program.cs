// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Compiler.Visualizers;

    /// <summary>
    /// A program which uses Infer.NET to compile various inference algorithms for Bayes point machine classifiers.
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// The entry point for the program.
        /// </summary>
        /// <param name="args">The array of command-line arguments.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("Usage: {0} <generated_source_folder>", Environment.GetCommandLineArgs()[0]);
                Environment.Exit(1);
            }

            bool debug = false;
            if (debug)
            {
                Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));
                var logWriter = new System.IO.StreamWriter("debug.txt");
                Trace.Listeners.Add(new TextWriterTraceListener(logWriter));
                Debug.AutoFlush = true;
                Trace.AutoFlush = true;
#if NETFRAMEWORK || WINDOWS
                Models.InferenceEngine.Visualizer = new WindowsVisualizer();
#endif
                Models.InferenceEngine.DefaultEngine.Compiler.BrowserMode = Compiler.BrowserMode.Always;
            }

            // The folder to drop the generated inference algorithms to
            string generatedSourceFolder = args[0];

            // Generate all training algorithms
            foreach (bool computeModelEvidence in new[] { false, true })
            {
                foreach (bool useCompoundWeightPriorDistributions in new[] { false, true })
                {
                    foreach (var trainingAlgorithmFactory in AlgorithmFactories.TrainingAlgorithmFactories)
                    {
                        trainingAlgorithmFactory(generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistributions);
                    }
                }
                AlgorithmFactories.CreateDenseBinaryVectorTrainingAlgorithm(generatedSourceFolder, computeModelEvidence, false);
            }

            // Generate all prediction algorithms
            foreach (var predictionAlgorithmFactory in AlgorithmFactories.PredictionAlgorithmFactories)
            {
                predictionAlgorithmFactory(generatedSourceFolder);
            }
        }
    }
}