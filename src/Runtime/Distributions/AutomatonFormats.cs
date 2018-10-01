// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using Microsoft.ML.Probabilistic.Distributions.Automata;

    /// <summary>
    /// A collection of sequence distribution formats.
    /// </summary>
    public static class SequenceDistributionFormats
    {
        /// <summary>
        /// Initializes static members of the <see cref="SequenceDistributionFormats"/> class.
        /// </summary>
        static SequenceDistributionFormats()
        {
            GraphViz = new SequenceDistributionFormatPointMassAsAutomaton(AutomatonFormats.GraphViz);
            Regexp = new SequenceDistributionFormatPointMassAsString(AutomatonFormats.Regexp);
            Friendly = new SequenceDistributionFormatPointMassAsString(AutomatonFormats.Friendly);
        }
        
        /// <summary>
        /// Gets a format for converting a sequence distribution to a GraphViz representation of the underlying automaton.
        /// </summary>
        public static ISequenceDistributionFormat GraphViz { get; private set; }

        /// <summary>
        /// Gets a format for converting a sequence distribution to a regular expression representing its support.
        /// </summary>
        public static ISequenceDistributionFormat Regexp { get; private set; }

        /// <summary>
        /// Gets a format for converting a sequence distribution to a friendly regular expression like string representing its support.
        /// </summary>
        public static ISequenceDistributionFormat Friendly { get; private set; }
    }
}