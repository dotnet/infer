using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    public class ExamplesBrowser
    {
        /// <summary>
        /// Pop up a window for running examples, when targeting Windows, then exit the application.  Otherwise do nothing.
        /// </summary>
        public static void RunBrowser() // Must not be called "Run"
        {
#if NETFRAMEWORK || WINDOWS
            InferenceEngine.Visualizer = new Compiler.Visualizers.WindowsVisualizer();
            // Show all tutorials, in a browser
            IAlgorithm[] algs = InferenceEngine.GetBuiltInAlgorithms();

            // Avoid max product in the examples browser, as none of the examples apply.
            List<IAlgorithm> algList = new List<IAlgorithm>(algs);
            algList.RemoveAll(alg => alg is MaxProductBeliefPropagation);
            ExamplesViewer tview = new ExamplesViewer(typeof(ExamplesBrowser), algList.ToArray());
            tview.RunBrowser();
            Environment.Exit(0);
#endif
        }
    }
}
