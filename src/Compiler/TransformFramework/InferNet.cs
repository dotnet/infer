// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <summary>
    /// Class used in MSL only.
    /// </summary>
    /// <exclude/>
    public static class InferNet
    {
        /// <summary>
        /// Used in MSL to indicate that a variable will be inferred.
        /// </summary>
        /// <param name="obj">A variable reference expression</param>
        public static void Infer<T>(T obj)
        {
        }

        /// <summary>
        /// Used in MSL to indicate that a variable will be inferred under a specific name.
        /// </summary>
        /// <param name="obj">A variable reference expression</param>
        /// <param name="name">The external name of the variable</param>
        public static void Infer<T>(T obj, string name)
        {
        }

        /// <summary>
        /// Used in MSL to indicate that a variable will be inferred under a specific name and query type.
        /// </summary>
        /// <param name="obj">A variable reference expression</param>
        /// <param name="name">The external name of the variable</param>
        /// <param name="query">The query type</param>
        public static void Infer<T>(T obj, string name, QueryType query)
        {
        }

        /// <summary>
        /// Used in MSL to indicate that the loop counter is increasing in the currently executing loop.
        /// </summary>
        /// <param name="loopCounter"></param>
        /// <returns></returns>
        public static bool IsIncreasing(int loopCounter)
        {
            return true;
        }
    }
}