// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.TestApp
{
    using System;
    using System.Diagnostics;
    using System.IO;

    using Tests;

    /// <summary>
    /// This program allows to run tests in a customized way from a console application.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The entry point for the program.
        /// </summary>
        /// <param name="args">The array of command-line arguments.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.Unicode;
            var coreAssemblyInfo = FileVersionInfo.GetVersionInfo(typeof(object).Assembly.Location);
            Console.WriteLine($".NET version {Environment.Version} mscorlib {coreAssemblyInfo.ProductVersion}");
            Console.WriteLine(Environment.Is64BitProcess ? "64-bit" : "32-bit");
            Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));
            var logWriter = new StreamWriter("debug.txt");
            Trace.Listeners.Add(new TextWriterTraceListener(logWriter));
            Debug.AutoFlush = true;
            Trace.AutoFlush = true;

            // Tests for the Bayes Point machine classifier
            var classifierTests = new BayesPointMachineClassifierTests();

            classifierTests.GaussianSparseBinaryStandardCustomSerializationRegressionTest();
            ////classifierTests.DenseBinaryNativeConstantZeroFeatureTest();
            ////classifierTests.SparseBinaryNativeConstantZeroFeatureTest();
            ////classifierTests.DenseMulticlassNativeConstantZeroFeatureTest();
            ////classifierTests.SparseMulticlassNativeConstantZeroFeatureTest();

            ////classifierTests.DenseBinaryNativeRareFeatureTest();
            ////classifierTests.SparseBinaryNativeRareFeatureTest();
            ////classifierTests.DenseMulticlassNativeRareFeatureTest();
            ////classifierTests.SparseMulticlassNativeRareFeatureTest();
            
            // Tests for the Matchbox recommender
            var recommenderTests = new MatchboxRecommenderTests();
            ////recommenderTests.UserItemFeaturesRegressionTest();
            ////recommenderTests.StandardDataFormatCustomSerializationTest();

#if NETFRAMEWORK
            logWriter.Dispose();
#endif
        }
    }
}
