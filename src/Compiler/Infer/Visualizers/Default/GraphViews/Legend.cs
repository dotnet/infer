// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    /// <summary>
    /// Holds info about one line type in the legend.
    /// </summary>
    internal class LegendItem
    {
        public int LineWidth { get; set; }
        public Color LineColor { get; set; }
        public EdgeStyle Style { get; set; }
        public string Label { get; set; }
    }



    /// <summary>
    /// Holds info about the legend for dependency graph.
    /// </summary>
    internal class Legend
    {
        /// <summary>
        /// The list of legend items.
        /// </summary>
        public List<LegendItem> Items { get; private set; }

        /// <summary>
        /// Width of the legend.
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// Hieght of the legend.
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// Top Y-coordinate of the legend.
        /// </summary>
        public int Top { get; set; }

        /// <summary>
        /// Left X-coordinate of the legend.
        /// </summary>
        public int Left { get; set; }

        /// <summary>
        /// Background color of the legend.
        /// </summary>
        public Color BackColor { get; set; }

        public Legend(int count)
        {
            Items = new List<LegendItem>(count);
        }
    }
}
