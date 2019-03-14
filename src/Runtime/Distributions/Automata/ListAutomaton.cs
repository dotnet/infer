// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A base class for automata defined on types implementing <see cref="IList{T}"/>.
    /// </summary>
    /// <typeparam name="TList">The type of a list the automaton is defined on.</typeparam>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over a list element.</typeparam>
    /// <typeparam name="TThis">The type of a concrete list automaton class.</typeparam>
    public abstract class ListAutomaton<TList, TElement, TElementDistribution, TThis>
        : Automaton<TList, TElement, TElementDistribution, ListManipulator<TList, TElement>, TThis>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TThis : ListAutomaton<TList, TElement, TElementDistribution, TThis>, new()
    {
        protected ListAutomaton()
        {
        }
    }

    /// <summary>
    /// An automaton defined on types implementing <see cref="IList{T}"/>.
    /// </summary>
    /// <typeparam name="TList">The type of a list the automaton is defined on.</typeparam>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over a list element.</typeparam>
    public class ListAutomaton<TList, TElement, TElementDistribution>
        : ListAutomaton<TList, TElement, TElementDistribution, ListAutomaton<TList, TElement, TElementDistribution>>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
    {
        public ListAutomaton()
        {
        }

        /// <summary>
        /// Computes a set of outgoing transitions from a given state of the determinization result.
        /// </summary>
        /// <param name="sourceState">The source state of the determinized automaton represented as 
        /// a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton.</param>
        /// <returns>
        /// A collection of (element distribution, weight, weighted state set) triples corresponding to outgoing transitions from <paramref name="sourceState"/>.
        /// The first two elements of a tuple define the element distribution and the weight of a transition.
        /// The third element defines the outgoing state.
        /// </returns>
        protected override IEnumerable<(TElementDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState)
        {
            throw new NotImplementedException("Determinization is not yet supported for this type of automata.");
        }
    }

    /// <summary>
    /// An automaton defined on generic lists.
    /// </summary>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over a list element.</typeparam>
    public class ListAutomaton<TElement, TElementDistribution>
        : ListAutomaton<List<TElement>, TElement, TElementDistribution, ListAutomaton<TElement, TElementDistribution>>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>,
        CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
    {
        public ListAutomaton()
        {
        }

        /// <summary>
        /// Computes a set of outgoing transitions from a given state of the determinization result.
        /// </summary>
        /// <param name="sourceState">The source state of the determinized automaton represented as 
        /// a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton.</param>
        /// <returns>
        /// A collection of (element distribution, weight, weighted state set) triples corresponding to outgoing transitions from <paramref name="sourceState"/>.
        /// The first two elements of a tuple define the element distribution and the weight of a transition.
        /// The third element defines the outgoing state.
        /// </returns>
        protected override IEnumerable<(TElementDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState)
        {
            throw new NotImplementedException();
        }
    }
}
