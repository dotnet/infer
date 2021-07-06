// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// An implementation of <see cref="AutomatonBasedSequenceDistributionFormatBase"/>, which works on a given distribution as follows:
    /// <list type="bullet">
    /// <item><description>
    /// If the distribution is not a point mass, applies the underlying automaton format to the probability function.
    /// </description></item>
    /// <item><description>
    /// If the distribution is a point mass, applies the underlying automaton format to the automaton representation of the point.
    /// </description></item>
    /// </list>
    /// </summary>
    public class SequenceDistributionFormatPointMassAsAutomaton : AutomatonBasedSequenceDistributionFormatBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SequenceDistributionFormatPointMassAsAutomaton"/> class.
        /// </summary>
        /// <param name="automatonFormat">The underlying automaton format. </param>
        public SequenceDistributionFormatPointMassAsAutomaton(IAutomatonFormat automatonFormat)
            : base(automatonFormat)
        {
        }

        /// <summary>
        /// Converts a point mass distribution to a string by first converting the point to an automaton function and
        /// then using the underlying automaton function format on that function.
        /// </summary>
        /// <typeparam name="TSequence">The type of sequences <paramref name="sequenceDistribution"/> is defined on.</typeparam>
        /// <typeparam name="TElement">The type of sequence elements of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TElementDistribution">The type of distributions over sequence elements of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate instances of <typeparamref name="TSequence"/>.</typeparam>
        /// <typeparam name="TAutomaton">The type of automata used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TWeightFunction">The type of weight functions used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TWeightFunctionFactory">The type of weight function factory used by <paramref name="sequenceDistribution"/>.</typeparam>
        /// <typeparam name="TSequenceDistribution">The concrete type of <paramref name="sequenceDistribution"/>.</typeparam>
        /// <param name="sequenceDistribution">The sequence distribution to convert to string.</param>
        /// <returns>The string representation of the <paramref name="sequenceDistribution"/>.</returns>
        protected override string ConvertPointMassToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TSequenceDistribution>(
            TSequenceDistribution sequenceDistribution)
        {
            return this.AutomatonFormat.ConvertToString(sequenceDistribution.ToAutomaton());
        }
    }
}