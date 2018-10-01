// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    /// <summary>
    /// Represents the type of a command-line parameter.
    /// </summary>
    public enum CommandLineParameterType
    {
        /// <summary>
        /// The parameter is required.
        /// </summary>
        Required,

        /// <summary>
        /// The parameter is optional.
        /// </summary>
        Optional
    }
}