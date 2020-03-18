// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;
    using System.Text;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Converts a given automaton to its <a href="http://www.graphviz.org/">GraphViz</a> description.
    /// </summary>
    public class GraphVizAutomatonFormat : IAutomatonFormat
    {
        /// <summary>
        /// Converts a given automaton to a GraphViz representation.
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
            var graphVizCode = new StringBuilder();

            graphVizCode.AppendLine("digraph finite_state_machine {");
            graphVizCode.AppendLine("  rankdir=LR;");

            // Specify states
            foreach (var state in automaton.States)
            {
                string shape = automaton.Start.Index == state.Index ? "doublecircle" : "circle";
                graphVizCode.AppendFormat("  node [shape = {0}; label = \"{1}\\nE={2:G5}\"]; N{1}", shape, state.Index, state.EndWeight.Value);
                graphVizCode.AppendLine();
            }

            // Specify transitions
            foreach (var state in automaton.States)
            {
                foreach (var transition in state.Transitions)
                {
                    string transitionLabel;
                    if (transition.IsEpsilon)
                    {
                        transitionLabel = "eps";
                    }
                    else if (transition.ElementDistribution.Value.IsPointMass)
                    {
                        transitionLabel = EscapeLabel(transition.ElementDistribution.Value.Point.ToString());
                    }
                    else
                    {
                        transitionLabel = EscapeLabel(transition.ElementDistribution.ToString());
                    }

                    string label = string.Format("W={0:G5}\\n{1}", transition.Weight.Value, transitionLabel);
                    if (transition.Group != 0)
                    {
                        label = string.Format("{0}\\n#{1}", label, transition.Group);
                    }

                    graphVizCode.AppendFormat("  N{0} -> N{1} [ label = \"{2}\" ];", state.Index, transition.DestinationStateIndex, label);
                    graphVizCode.AppendLine();
                }
            }

            graphVizCode.AppendLine("}");

            return graphVizCode.ToString();
        }

        public string Escape(string rawString)
        {
            return rawString;
        }

        /// <summary>
        /// Escapes an edge label text.
        /// </summary>
        /// <param name="label">A raw edge label text.</param>
        /// <returns>The escaped edge label text.</returns>
        private static string EscapeLabel(string label)
        {
            const string CharsToEscape = "\"\\";

            var result = new StringBuilder();
            for (int i = 0; i < label.Length; ++i)
            {
                if (CharsToEscape.IndexOf(label[i]) != -1)
                {
                    result.Append('\\');
                }

                result.Append(label[i]);
            }

            return result.ToString();
        }
    }
}