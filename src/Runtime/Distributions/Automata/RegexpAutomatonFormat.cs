// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using System.Text;
    using System.Text.RegularExpressions;

    /// <summary>
    /// Converts a given automaton to a regular expression
    /// (or a regular expression-like string) corresponding to the support of the automaton.
    /// </summary>
    public class RegexpAutomatonFormat : IAutomatonFormat
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RegexpAutomatonFormat"/> class.
        /// </summary>
        /// <param name="formattingSettings">The formatting settings used for conversion from an automaton to a string.</param>
        public RegexpAutomatonFormat(RegexpFormattingSettings formattingSettings)
        {
            Argument.CheckIfNotNull(formattingSettings, "formattingSettings");
            
            this.FormattingSettings = formattingSettings;
        }

        /// <summary>
        /// Gets the formatting settings used for conversion.
        /// </summary>
        public RegexpFormattingSettings FormattingSettings { get; private set; }

        /// <summary>
        /// Converts a given automaton to a regular expression corresponding to the support of the function.
        /// </summary>
        /// <typeparam name="TSequence">The type of sequences <paramref name="automaton"/> is defined on.</typeparam>
        /// <typeparam name="TElement">The type of sequence elements of <paramref name="automaton"/>.</typeparam>
        /// <typeparam name="TElementDistribution">The type of distributions over sequence elements of <paramref name="automaton"/>.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate instances of <typeparamref name="TSequence"/>.</typeparam>
        /// <typeparam name="TAutomaton">The concrete type of <paramref name="automaton"/>.</typeparam>
        /// <param name="automaton">The automaton to convert to a string.</param>
        /// <returns>The string representation of <paramref name="automaton"/>.</returns>
        public string ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>(
            Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> automaton)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>,
                SettableToPartialUniform<TElementDistribution>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        {
            var regex = RegexpTreeBuilder.BuildRegexp(automaton);
            var builder = new StringBuilder();
            bool noTruncation = regex.AppendToString(builder,this.FormattingSettings);
            if (!noTruncation) builder.Append(" ... long regex truncated");
            return builder.ToString();
        }

        /// <summary>
        /// Escapes a raw string, such that potential special characters can be represented in a way that the target can handle.
        /// </summary>
        /// <param name="rawString">Raw generated string.</param>
        /// <returns>An escaped string.</returns>
        public string Escape(string rawString)
        {
            return this.FormattingSettings.EscapeCharacters ? Regex.Escape(rawString) : rawString;
        }
    }
}