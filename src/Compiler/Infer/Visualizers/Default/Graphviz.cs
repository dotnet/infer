// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews;
using System;
using System.Diagnostics;
using System.IO;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Class for visualizing graphs using Graphviz "dot.exe" program.
    /// </summary>
    internal class Graphviz
    {
        public static void ShowGraph(GraphWriter graph, string title = "Graph")
        {
            string fileName = title + DateTime.Now.ToString("_MM_dd_yy_HH_mm_ss_ff");
            string dotFileName = fileName + ".gv";
            string svgFileName = fileName + ".svg";

            graph.Write(dotFileName);

            try
            {
                Process.Start("dot", "-Tsvg -o" + "\"" + svgFileName + "\" \"" + dotFileName + "\"").WaitForExit();
            }
            catch (Exception e)
            {
                Console.WriteLine("Problem with converting DOT to SVG");
                Console.WriteLine($"Exception message: \"{e.Message}\"\n");
                Console.WriteLine("If \"dot\" program is not installed, install Graphviz\nand add a path to \"dot\" to the PATH\n");
                Console.WriteLine($"DOT file is saved to \"{Path.GetFullPath(dotFileName)}\"\n");

                return;
            }

            try
            {
                var p = new Process();
                p.StartInfo = new ProcessStartInfo(svgFileName);
                p.StartInfo.UseShellExecute = true;
                p.Start();
            }
            catch
            {
                Console.WriteLine("Can't open SVG file using shell\n");
                Console.WriteLine($"DOT file is saved to \"{Path.GetFullPath(dotFileName)}\"\n");
                Console.WriteLine($"SVG file is saved to \"{Path.GetFullPath(svgFileName)}\"\n");
            }
        }
    }
}
