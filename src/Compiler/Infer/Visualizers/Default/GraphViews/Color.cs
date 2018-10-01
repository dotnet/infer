// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    internal enum Color { Empty, Red, Green, Blue, Purple, LightGray, Black, White }

    internal static class ColorStringHexValues
    {
        public static string GetHexString(this Color c)
        {
            switch(c)
            {
                case Color.Red:
                    return "#FF0000";
                case Color.Green:
                    return "#00FF00";
                case Color.Blue:
                    return "#0000FF";
                case Color.Purple:
                    return "#800080";
                case Color.LightGray:
                    return "#D3D3D3";
                case Color.Black:
                    return "#000000";
                case Color.White:
                    return "#FFFFFF";
                case Color.Empty:
                default:
                    return "";
            }
        }
    }
}
