// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// A collection of automaton formats.
    /// </summary>
    public static class AutomatonFormats
    {
        /// <summary>
        /// Initializes static members of the <see cref="AutomatonFormats"/> class.
        /// </summary>
        static AutomatonFormats()
        {
            GraphViz = new GraphVizAutomatonFormat();
            Regexp = new RegexpAutomatonFormat(
                new RegexpFormattingSettings(
                    putOptionalInSquareBrackets: false,
                    showAnyElementAsQuestionMark: false,
                    ignoreElementDistributionDetails: false,
                    truncationLength: int.MaxValue,
                    escapeCharacters: true,
                    useLazyQuantifier: true));
            Friendly = new RegexpAutomatonFormat(
                new RegexpFormattingSettings(
                    putOptionalInSquareBrackets: true,
                    showAnyElementAsQuestionMark: true,
                    ignoreElementDistributionDetails: true,
                    truncationLength: 500,
                    escapeCharacters: false,
                    useLazyQuantifier: false));
        }

        /// <summary>
        /// Gets a format for converting an automaton to a GraphViz representation.
        /// </summary>
        public static GraphVizAutomatonFormat GraphViz { get; }

        /// <summary>
        /// Gets a format for converting an automaton to a regular expression representing its support.
        /// </summary>
        public static RegexpAutomatonFormat Regexp { get; }

        /// <summary>
        /// Gets a format for converting an automaton to a friendly regular expression like string representing its support.
        /// </summary>
        public static RegexpAutomatonFormat Friendly { get; }
    }
}
