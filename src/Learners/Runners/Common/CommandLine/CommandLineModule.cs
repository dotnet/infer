// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    /// <summary>
    /// Module that can be invoked from the command line.
    /// </summary>
    public abstract class CommandLineModule
    {
        /// <summary>
        /// Overridden in the derived classes to perform module operations.
        /// If the specified command line is invalid, a module should print an error message and return false.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public abstract bool Run(string[] args, string usagePrefix);
    }
}
