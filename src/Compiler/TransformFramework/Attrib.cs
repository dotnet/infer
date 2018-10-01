// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// For use in MSL only, to add attributes to variables.
    /// </summary>
    internal static class Attrib
    {
        /// <summary>
        /// Adds an attribute to a variable.
        /// </summary>
        /// <typeparam name="T">The type of the variable</typeparam>
        /// <param name="variable">The variable</param>
        /// <param name="attr">The attribute to add</param>
        /// <returns>The variable</returns>
        public static T Var<T>(T variable, ICompilerAttribute attr)
        {
            return variable;
        }

        /// <summary>
        /// Specifies the initial marginal of a variable.
        /// </summary>
        /// <param name="var">The variable whose initial marginal is to be set</param>
        /// <param name="initialMessages">The initial marginal (or array of marginals)</param>
        public static void InitialiseTo(object var, object initialMessages)
        {
        }
    }
}