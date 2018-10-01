// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic;
    using System;

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Class used in MSL only.
    /// </summary>
    /// <exclude/>
    public static class InferNet
    {
        /// <summary>
        /// For use in MSL only.
        /// </summary>
        /// <param name="obj"></param>
        public static void Infer(object obj)
        {
        }

        public static void Infer(object obj, string name)
        {
        }

        public static void Infer(object obj, string name, QueryType query)
        {
        }

        /*public class Power : IDisposable
        {
                double power;

                public Power(double p)
                {
                        power = p;
                }

                public void Dispose()
                {
                }
        }*/
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}