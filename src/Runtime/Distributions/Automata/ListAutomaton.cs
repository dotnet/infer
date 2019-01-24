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
        protected override List<(TElementDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
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
        protected override List<(TElementDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState)
        {
            throw new NotImplementedException();

            //// Build a list of elements, with probabilities
            //var elementLists = new Dictionary<TElement, List<TransitionElement>>();
            //var uniformList = new List<TransitionElement>();
            //foreach (KeyValuePair<int, Weight> stateIdWeight in sourceState)
            //{
            //    var state = this.States[stateIdWeight.Key];
            //    for (int i = 0; i < state.TransitionCount; ++i)
            //    {
            //        AddTransitionElements(state.GetTransition(i), stateIdWeight.Value, elementLists, uniformList);
            //    }
            //}

            //// Produce an outgoing transition for each unique subset of overlapping segments
            //var results = new List<Tuple<TElementDistribution, Weight, Determinization.WeightedStateSet>>();
  
            //foreach (var kvp in elementLists)
            //{
            //    AddResult(results, kvp.Value);
            //}
            //AddResult(results, uniformList);
            //return results;
        }

        private static void AddResult(List<Tuple<TElementDistribution, Weight, Determinization.WeightedStateSet>> results, 
             List<TransitionElement> transitionElements)
        {
            if (transitionElements.Count == 0) return;
            const double LogEps = -30; // Don't add transitions with log-weight less than this as they have been produced by numerical inaccuracies
            
            var elementStatesWeights = new Determinization.WeightedStateSet();
            var elementStateWeightSum = Weight.Zero;
            foreach (var element in transitionElements)
            {
                Weight prevStateWeight;
                if (!elementStatesWeights.TryGetWeight(element.destIndex, out prevStateWeight))
                {
                    prevStateWeight = Weight.Zero;
                }

                elementStatesWeights[element.destIndex] = Weight.Sum(prevStateWeight, element.weight);
                elementStateWeightSum = Weight.Sum(elementStateWeightSum, element.weight);
            }

            var destinationState = new Determinization.WeightedStateSet();
            foreach (KeyValuePair<int, Weight> stateIdWithWeight in elementStatesWeights)
            {
                if (stateIdWithWeight.Value.LogValue > LogEps)
                {
                    Weight stateWeight = Weight.Product(stateIdWithWeight.Value, Weight.Inverse(elementStateWeightSum));
                    destinationState.Add(stateIdWithWeight.Key, stateWeight);
                }
            }

            Weight transitionWeight = Weight.Product(Weight.FromValue(1), elementStateWeightSum);
            results.Add(Tuple.Create(transitionElements[0].distribution,transitionWeight, destinationState));
        }

        /// <summary>
        /// Given a transition and the residual weight of its source state, adds weighted non-zero probability elements
        /// associated with the transition to the list.
        /// </summary>
        /// <param name="transition">The transition.</param>
        /// <param name="sourceStateResidualWeight">The logarithm of the residual weight of the source state of the transition.</param>
        /// <param name="elements">The list for storing transition elements.</param>
        /// <param name="uniformList">The list for storing transition elements for uniform.</param>
        private static void AddTransitionElements(
            Transition transition, Weight sourceStateResidualWeight, 
            Dictionary<TElement, List<TransitionElement>> elements, List<TransitionElement> uniformList)
        {
            var dist = transition.ElementDistribution.Value;
            Weight weightBase = Weight.Product(transition.Weight, sourceStateResidualWeight);
            if (dist.IsPointMass)
            {
                var pt = dist.Point;
                // todo: enumerate distribution
                if (!elements.ContainsKey(pt)) elements[pt] = new List<TransitionElement>();
                elements[pt].Add(new TransitionElement(transition.DestinationStateIndex, weightBase, dist));
            }
            else
            {
                uniformList.Add(new TransitionElement(transition.DestinationStateIndex, weightBase, dist));
            }
        }

        private class TransitionElement
        {
            internal int destIndex;
            internal Weight weight;
            internal TElementDistribution distribution;

            internal TransitionElement(int destIndex, Weight weight, TElementDistribution distribution)
            {
                this.destIndex = destIndex;
                this.distribution = distribution;
                this.weight = weight;
            }
        }
    }
}
