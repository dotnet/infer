// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// The program.
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// The default config file.
        /// </summary>
        private const string DefaultConfigFile = "Config.xml";

        private static bool HasFailed = false;

        #region Event handler lists

        /// <summary>
        /// The list of handlers for <see cref="RecommenderRun.Started"/> event.
        /// </summary>
        private static List<EventHandler> runStartedHandlers;

        /// <summary>
        /// The list of handlers for <see cref="RecommenderRun.Completed"/> event.
        /// </summary>
        private static List<EventHandler<RecommenderRunCompletedEventArgs>> runCompletedHandlers;

        /// <summary>
        /// The list of handlers for <see cref="RecommenderRun.FoldProcessed"/> event.
        /// </summary>
        private static List<EventHandler<RecommenderRunFoldProcessedEventArgs>> foldProcessedHandlers;

        /// <summary>
        /// The list of handlers for <see cref="RecommenderRun.Interrupted"/> event.
        /// </summary>
        private static List<EventHandler<RecommenderRunInterruptedEventArgs>> runInterruptedHandlers;

        #endregion

        /// <summary>
        /// The entry point of the application.
        /// </summary>
        /// <param name="args">The list of command line arguments.</param>
        public static int Main(string[] args)
        {
            try
            {
                string configFile;
                if (args.Length > 0)
                {
                    configFile = args[0];
                }
                else
                {
                    configFile = DefaultConfigFile;
                    Console.WriteLine($"Using default config file {configFile}");
                }

                RegisterTestRunHandlers();
                
                EvaluatorConfiguration configuration = EvaluatorConfiguration.LoadFromFile(configFile);

                foreach (var runConfiguration in configuration.Runs)
                {
                    var testRun = runConfiguration.Create();
                    SubscribeTestRunHandlers(testRun);
                    testRun.Execute();
                }
            }
            catch (Exception e)
            {
                HasFailed = true;
                PrintErrorMessage(e);
            }
            int exitCode = HasFailed ? 1 : 0;
            return exitCode;
        }

        #region Console output formatting

        /// <summary>
        /// Prints the error message associated with a given exception.
        /// </summary>
        /// <param name="exception">The exception.</param>
        private static void PrintErrorMessage(Exception exception)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(exception.Message);
            if (exception.InnerException != null)
            {
                Console.WriteLine(exception.InnerException.Message);
            }

            Console.ResetColor();
        }

        /// <summary>
        /// Prints a given value with a given caption to console.
        /// </summary>
        /// <param name="caption">The caption for the value.</param>
        /// <param name="value">The value.</param>
        /// <param name="startNewLine">Indicates whether the new line should be started after printing the value.</param>
        private static void WriteValueWithCaption(string caption, object value, bool startNewLine = true)
        {
            const int Padding = 20;
            
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.Write(caption.PadRight(Padding));
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write(value);
            if (startNewLine)
            {
                Console.WriteLine();    
            }

            Console.ResetColor();
        }

        #endregion

        #region Handler registration

        /// <summary>
        /// Fills the handler lists.
        /// </summary>
        private static void RegisterTestRunHandlers()
        {
            //// Add new handlers in this function

            runStartedHandlers = new List<EventHandler>
                {
                    ConsoleOutputRecommenderRunStartedHandler
                };

            foldProcessedHandlers = new List<EventHandler<RecommenderRunFoldProcessedEventArgs>>
                {
                    ConsoleOutputRecommenderRunFoldProcessedHandler
                };

            runCompletedHandlers = new List<EventHandler<RecommenderRunCompletedEventArgs>>
                {
                    ConsoleOutputRecommenderRunCompletedHandler
                };

            runInterruptedHandlers = new List<EventHandler<RecommenderRunInterruptedEventArgs>>
                {
                    ConsoleOutputRecommenderRunInterruptedHandler,
                    FlagRaisingRecommenderRunInterruptedHandler
                };
        }

        /// <summary>
        /// Subscribes event handlers to a given test run.
        /// </summary>
        /// <param name="testRun">The test run to subscribe handlers to.</param>
        private static void SubscribeTestRunHandlers(RecommenderRun testRun)
        {
            foreach (var runStartedHandler in runStartedHandlers)
            {
                testRun.Started += runStartedHandler;
            }

            foreach (var foldProcessedHandler in foldProcessedHandlers)
            {
                testRun.FoldProcessed += foldProcessedHandler;
            }

            foreach (var runCompletedHandler in runCompletedHandlers)
            {
                testRun.Completed += runCompletedHandler;
            }

            foreach (var runInterruptedHandler in runInterruptedHandlers)
            {
                testRun.Interrupted += runInterruptedHandler;
            }
        }

        #endregion

        #region Console-printing handlers

        /// <summary>
        /// <see cref="RecommenderRun.Started"/> handler which prints results to console.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The arguments of the event.</param>
        private static void ConsoleOutputRecommenderRunStartedHandler(object sender, EventArgs e)
        {
            var testRun = (RecommenderRun)sender;

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(testRun.Name);

            WriteValueWithCaption("Folds:", testRun.FoldCount);
            WriteValueWithCaption("Progress:", string.Empty, false);
        }

        /// <summary>
        /// <see cref="RecommenderRun.FoldProcessed"/> handler which prints results to console.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The arguments of the event.</param>
        private static void ConsoleOutputRecommenderRunFoldProcessedHandler(object sender, RecommenderRunFoldProcessedEventArgs e)
        {
            var testRun = (RecommenderRun)sender;

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("+");

            if (e.FoldNumber == testRun.FoldCount - 1)
            {
                Console.WriteLine();
            }
        }

        /// <summary>
        /// <see cref="RecommenderRun.Completed"/> handler which prints results to console.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The arguments of the event.</param>
        private static void ConsoleOutputRecommenderRunCompletedHandler(object sender, RecommenderRunCompletedEventArgs e)
        {
            WriteValueWithCaption("Total time:", e.TotalTime);
            WriteValueWithCaption("Training time:", e.TrainingTime);
            WriteValueWithCaption("Prediction time:", e.PredictionTime);
            WriteValueWithCaption("Evaluation time:", e.EvaluationTime);

            const int MetricNamePadding = 25;
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("Metrics:");
            foreach (KeyValuePair<string, MetricValueDistribution> nameValue in e.Metrics)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("\t{0}", nameValue.Key.PadRight(MetricNamePadding));
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine("\t{0:0.000} ± {1:0.000}", nameValue.Value.Mean, nameValue.Value.StdDev);
            }

            Console.WriteLine();
            Console.ResetColor();
        }

        /// <summary>
        /// <see cref="RecommenderRun.Interrupted"/> handler which prints results to console.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The arguments of the event.</param>
        private static void ConsoleOutputRecommenderRunInterruptedHandler(object sender, RecommenderRunInterruptedEventArgs e)
        {
            Console.WriteLine();
            PrintErrorMessage(e.Exception);
            Console.WriteLine();
        }

        /// <summary>
        /// <see cref="RecommenderRun.Interrupted"/> handler which sets ErrorHappened flag to true.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The arguments of the event.</param>
        private static void FlagRaisingRecommenderRunInterruptedHandler(object sender, RecommenderRunInterruptedEventArgs e)
        {
            HasFailed = true;
        }

        #endregion
    }
}
