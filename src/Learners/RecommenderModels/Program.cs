// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Diagnostics;
    using System.IO;

    /// <summary>
    /// The program.
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
                Console.OutputEncoding = System.Text.Encoding.Unicode;
                Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));
                var logWriter = new StreamWriter("debug.txt");
                Trace.Listeners.Add(new TextWriterTraceListener(logWriter));
                Debug.AutoFlush = true;
                Trace.AutoFlush = true;
                //InferenceEngine.DefaultEngine.BrowserMode = BrowserMode.Always;
            }

            string generatedSourceFolder = args[0];
            AlgorithmFactories.GenerateCommunityTrainingAlgorithm(generatedSourceFolder);
            AlgorithmFactories.GenerateRatingPredictionAlgorithm(generatedSourceFolder);
        }
    }
}
