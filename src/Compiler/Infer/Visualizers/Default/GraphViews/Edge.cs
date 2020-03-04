// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    internal enum ArrowheadStyle
    {
        Normal, None
    }

    internal enum EdgeStyle
    {
        Normal, Dashed
    }

    /// <summary>
    /// Extention for getting string values of styles.
    /// </summary>
    internal static class StyleStringValues
    {
        public static string GetStringValue(this EdgeStyle style)
        {
            switch (style)
            {
                case EdgeStyle.Normal:
                    return "solid";
                case EdgeStyle.Dashed:
                    return "dashed";
                default:
                    return "";
            }
        }

        public static string GetStringValue(this ArrowheadStyle style)
        {
            switch (style)
            {
                case ArrowheadStyle.None:
                    return "none";
                case ArrowheadStyle.Normal:
                    return "normal";
                default:
                    return "";
            }
        }
    }

    /// <summary>
    /// Holds info about edge attributes.
    /// </summary>
    internal class Edge
    {
        public object UserData { get; set; }

        public Node Source { get; private set; }
        public Node Target { get; private set; }

        public string Label { get; set; }
        public Color FontColor { get; set; }
        public int FontSize { get; set; }

        public Color Color { get; set; }
        public int Width { get; set; }

        public ArrowheadStyle ArrowheadAtSource { get; set; } = ArrowheadStyle.None;
        public ArrowheadStyle ArrowheadAtTarget { get; set; } = ArrowheadStyle.Normal;

        public EdgeStyle Style { get; set; }
        public bool Reverse { get; set; }

        public Edge(Node source, Node target)
        {
            Source = source;
            Target = target;
        }
    }
}
