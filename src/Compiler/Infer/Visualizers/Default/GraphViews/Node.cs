// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    internal enum ShapeStyle
    {
        Box, Ellipse, None, Empty
    }

    /// <summary>
    /// Extention for getting string values of shape styles.
    /// </summary>
    internal static class ShapeStyleStringValues
    {
        public static string GetStringValue(this ShapeStyle style)
        {
            switch (style)
            {
                case ShapeStyle.Box:
                    return "box";
                case ShapeStyle.Ellipse:
                    return "ellipse";
                case ShapeStyle.None:
                    return "none";
                default:
                    return "";
            }
        }
    }

    /// <summary>
    /// Holds info about node attributes.
    /// </summary>
    internal class Node
    {
        public string ID { get; private set; }
        public object UserData { get; set; }

        public string Label { get; set; }
        public Color FontColor { get; set; } = Color.Black;
        public int FontSize { get; set; }

        public Color FillColor { get; set; }
        public Color BorderColor { get; set; }
        public ShapeStyle Shape { get; set; }
        public int BorderWidth { get; set; }

        public Node(string id)
        {
            ID = id;
        }
    }
}
