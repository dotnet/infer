// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Concurrent;
using System.Reflection;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Microsoft.ML.Probabilistic.Tests;
using Microsoft.ML.Probabilistic.Tests.Core;
using Microsoft.ML.Probabilistic.Tests.CodeModel;
using Microsoft.ML.Probabilistic.Tests.CodeCompilerTests;


#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

[assembly: AssemblyVersion("1.0.0.0")]
namespace TestApp
{
    internal static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        private static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.Unicode;
            var coreAssemblyInfo = FileVersionInfo.GetVersionInfo(typeof(object).Assembly.Location);
            Console.WriteLine($".NET version {Environment.Version} mscorlib {coreAssemblyInfo.ProductVersion}");
            Console.WriteLine(Environment.Is64BitProcess ? "64-bit" : "32-bit");
            Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));
            var logWriter = new StreamWriter("debug.txt");
            Trace.Listeners.Add(new TextWriterTraceListener(logWriter));
#if NETFRAMEWORK || WINDOWS
            InferenceEngine.Visualizer = new WindowsVisualizer();
#endif
            Debug.AutoFlush = true;
            Trace.AutoFlush = true;
            InferenceEngine.DefaultEngine.Compiler.RecommendedQuality = QualityBand.Experimental;
            InferenceEngine.DefaultEngine.Compiler.AddComments = false;
            //InferenceEngine.DefaultEngine.Compiler.CompilerChoice = Microsoft.ML.Probabilistic.Compiler.CompilerChoice.Roslyn;
            //InferenceEngine.DefaultEngine.Compiler.GenerateInMemory = false;
            InferenceEngine.DefaultEngine.Compiler.WriteSourceFiles = true;
            InferenceEngine.DefaultEngine.Compiler.IncludeDebugInformation = true;
            //InferenceEngine.DefaultEngine.Compiler.OptimiseInferenceCode = false;
            //InferenceEngine.DefaultEngine.Compiler.FreeMemory = false;
            //InferenceEngine.DefaultEngine.Compiler.ReturnCopies = false;
            //InferenceEngine.DefaultEngine.Compiler.UnrollLoops = false;
            //InferenceEngine.DefaultEngine.Compiler.UseParallelForLoops = false;
            //InferenceEngine.DefaultEngine.ShowTimings = true;
            //InferenceEngine.DefaultEngine.ShowProgress = false;
            //InferenceEngine.DefaultEngine.ShowFactorGraph = true;
            //InferenceEngine.DefaultEngine.SaveFactorGraphToFolder = @".";
            //InferenceEngine.DefaultEngine.Compiler.ShowProgress = true;
            //InferenceEngine.DefaultEngine.ShowMsl = true;
            //InferenceEngine.DefaultEngine.ShowSchedule = true;
            //InferenceEngine.DefaultEngine.Compiler.CatchExceptions = true;
            //InferenceEngine.DefaultEngine.Compiler.UseSerialSchedules = false;
            //InferenceEngine.DefaultEngine.Compiler.UseExperimentalSerialSchedules = true;
            //InferenceEngine.DefaultEngine.Compiler.AllowSerialInitialisers = true;
            //InferenceEngine.DefaultEngine.Compiler.UseLocals = false;
            TestUtils.SetDebugOptions();
            TestUtils.SetBrowserMode(BrowserMode.OnError);
            //TestUtils.SetBrowserMode(BrowserMode.Always);
            //TestUtils.SetBrowserMode(BrowserMode.WriteFiles);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            bool runAllTests = false;
            if (runAllTests)
            {
                // Run all tests (need to run in 64-bit else OutOfMemory due to loading many DLLs)
                // This is useful when looking for failures due to certain compiler options.
                //Console.WriteLine(StringUtil.VerboseToString(TestUtils.GetTestResultPaths()));
                //string path = @"C:\Users\minka\Depots\mlp\infernet\Infer2\TestResults\minka_MSRC-MINKA3 2013-04-11 14_36_55.trx";
                InferenceEngine.DefaultEngine.Compiler.RecommendedQuality = QualityBand.Preview;
                InferenceEngine.DefaultEngine.Compiler.GenerateInMemory = true;
                InferenceEngine.DefaultEngine.Compiler.WriteSourceFiles = false;
                TestUtils.TestAllCompilerOptions();
                //TestUtils.TestAllCompilerOptions(path);
                //TestUtils.RunAllTests(path);
                //using(TextWriter writer = new StreamWriter(@"..\..\RunAllTests.cs")) {
                //  TestUtils.WriteCodeToRunAllTests(writer, path);
                //}
                //TestUtils.CheckTransformNames();
            }
            bool showFactorManager = false;
            if (showFactorManager)
            {
                InferenceEngine.ShowFactorManager(true);
            }
#if NETFRAMEWORK
            logWriter.Dispose();
#endif
            watch.Stop();
            Console.WriteLine("elapsed time = {0}ms", watch.ElapsedMilliseconds);
        }
    }
}

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
