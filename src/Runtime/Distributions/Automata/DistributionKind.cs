// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// Type of distribution.
    /// </summary>
    public enum DistributionKind
    {
        /// <summary>
        /// All sequences in the language have the same probability.
        /// </summary>
        UniformOverValue,

        /// <summary>
        /// All sequence lengths have the same probability. For a given sequence length, all sequences have the same probability.
        /// </summary>
        UniformOverLengthThenValue
    }
}
