// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    using Math;

    /// <content>
    /// Extracts groups from a loop-free automaton.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        private static class GroupExtractor
        {           
            internal static Dictionary<int, TThis> ExtractGroups(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton)
            {
                var order = ComputeTopologicalOrderAndGroupSubgraphs(automaton, out var subGraphs);
                return BuildSubautomata(automaton.States, order, subGraphs);
            }

            private static Dictionary<int, TThis> BuildSubautomata(
                IReadOnlyList<State> states,
                IReadOnlyList<State> topologicalOrder,
                Dictionary<int, HashSet<int>> groupSubGraphs) => groupSubGraphs.ToDictionary(g => g.Key, g => BuildSubautomaton(states, topologicalOrder, g.Key, g.Value));

            private static TThis BuildSubautomaton(IReadOnlyList<State> states, IReadOnlyList<State> topologicalOrder, int group, HashSet<int> subgraph)
            {
                var weightsFromRoot = ComputeWeightsFromRoot(states.Count, topologicalOrder, group);
                var weightsToEnd = ComputeWeightsToEnd(states.Count, topologicalOrder, group);
                var subautomaton = new TThis();
                var stateMapping = subgraph.ToDictionary(x => x, _ => subautomaton.AddState());
                var hasNoIncomingTransitions = new HashSet<int>(subgraph);

                // copy the automaton and find states without incoming transitions.
                foreach (var stateIndex in subgraph)
                {
                    var newSourceState = stateMapping[stateIndex];

                    for (int i = 0; i < states[stateIndex].TransitionCount; i++)
                    {
                        var transition = states[stateIndex].GetTransition(i);
                        if (transition.Group != group) continue;
                        hasNoIncomingTransitions.Remove(transition.DestinationStateIndex);
                        newSourceState.AddTransition(
                            transition.ElementDistribution,
                            transition.Weight,
                            stateMapping[transition.DestinationStateIndex]);
                    }
                }

                var correctionFactor = Weight.Zero;
                
                // mark start and end states, modulo paths bypassing the automaton.
                foreach (var stateIndex in subgraph)
                {
                    var newSourceState = stateMapping[stateIndex];

                    // consider start states
                    var weightFromRoot = newSourceState.TransitionCount > 0 ? weightsFromRoot[stateIndex] : Weight.Zero;
                    if (!weightFromRoot.IsZero)
                    {
                        subautomaton.Start.AddEpsilonTransition(weightFromRoot, newSourceState);
                    }

                    // consider end states
                    var weightToEnd = !hasNoIncomingTransitions.Contains(stateIndex) ? weightsToEnd[stateIndex] : Weight.Zero;
                    if (!weightToEnd.IsZero)
                    {
                        newSourceState.SetEndWeight(weightToEnd);
                    }

                    correctionFactor = Weight.Sum(correctionFactor, Weight.Product(weightFromRoot, weightToEnd));
                }

                if (!correctionFactor.IsZero) throw new Exception("Write a unit test for this case. Code should be fine.");
                var epsilonWeight = Weight.AbsoluteDifference(weightsToEnd[topologicalOrder[0].Index], correctionFactor);
                subautomaton.Start.SetEndWeight(epsilonWeight);

                return subautomaton;
            }

            /// <summary>
            /// Creates a zero-initialized array of weights. It is important, that Weight.Zero is not the default value, so we need to explicitly set it.
            /// </summary>
            private static Weight[] CreateZeroWeights(int nStates)
            {
                var weights = new Weight[nStates];
                for (var j = 0; j < nStates; j++)
                {
                    weights[j] = Weight.Zero;
                }

                return weights;
            }

            private static List<State> ComputeTopologicalOrderAndGroupSubgraphs(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton, out Dictionary<int, HashSet<int>> groupSubGraphs)
            {
                var topologicalOrder = new Stack<int>();
                var temporary = new BitArray(automaton.States.Count);
                var permanent = new BitArray(automaton.States.Count);
                groupSubGraphs = new Dictionary<int, HashSet<int>>();

                VisitNode(automaton.States, automaton.Start.Index, temporary, permanent, groupSubGraphs, topologicalOrder);
                return topologicalOrder.Select(idx => automaton.States[idx]).ToList();
            }

            private static void VisitNode(IReadOnlyList<State> states, int stateIdx, BitArray temporary, BitArray permanent, Dictionary<int, HashSet<int>> groupSubGraphs, Stack<int> topologicalOrder)
            {
                if (temporary[stateIdx])
                {
                    throw new InvalidOperationException("the automaton is not a DAG");
                }

                if (permanent[stateIdx]) return;

                var state = states[stateIdx];

                temporary[stateIdx] = true;
                for (var i = 0; i < state.TransitionCount; i++)
                {
                    var transition = state.GetTransition(i);
                    var group = transition.Group;
                    if (group != 0)
                    {
                        HashSet<int> groupSubGraph;
                        if (!groupSubGraphs.TryGetValue(group, out groupSubGraph))
                        {
                            groupSubGraph = new HashSet<int>();
                            groupSubGraphs.Add(group, groupSubGraph);
                        }

                        groupSubGraph.Add(stateIdx);
                        groupSubGraph.Add(transition.DestinationStateIndex);
                    }

                    VisitNode(states, transition.DestinationStateIndex, temporary, permanent, groupSubGraphs, topologicalOrder);
                }
                permanent[stateIdx] = true;
                temporary[stateIdx] = false;
                topologicalOrder.Push(stateIdx);
            }



            /// <summary>
            /// For each state of the component, computes the total weight of all paths starting at that state.
            /// Ending weights are taken into account.
            /// </summary>
            /// <remarks>The weights are computed using dynamic programming, going up from leafs to the root.</remarks>
            private static Weight[] ComputeWeightsToEnd(int nStates, IReadOnlyList<State> topologicalOrder, int group)
            {
                var weights = CreateZeroWeights(nStates);
                // Iterate in the reverse topological order
                for (var stateIndex = topologicalOrder.Count - 1; stateIndex >= 0; stateIndex--)
                {
                    var state = topologicalOrder[stateIndex];
                    // Aggregate weights of all the outgoing transitions from this state
                    var weightToAdd = state.EndWeight;
                    for (var transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                    {
                        var transition = state.GetTransition(transitionIndex);

                        if (transition.Group == group) continue;

                        weightToAdd = Weight.Sum(
                            weightToAdd,
                            Weight.Product(transition.Weight, weights[transition.DestinationStateIndex]));
                    }

                    weights[state.Index] = weightToAdd;
                }

                return weights;
            }


            /// <summary>
            /// For each state of the component, computes the total weight of all paths starting at the root
            /// and ending at that state. Ending weights are not taken into account.
            /// </summary>
            /// <remarks>The weights are computed using dynamic programming, going down from the root to leafs.</remarks>
            private static Weight[] ComputeWeightsFromRoot(int nStates, IReadOnlyList<State> topologicalOrder, int group)
            {
                var weights = CreateZeroWeights(nStates);
                weights[topologicalOrder[0].Index] = Weight.One;

                // Iterate in the topological order
                for (var i = 0; i < topologicalOrder.Count; i++)
                {
                    var srcState = topologicalOrder[i];
                    var srcWeight = weights[srcState.Index];
                    if (srcWeight.IsZero)
                    {
                        continue;
                    }

                    // Aggregate weights of all the outgoing transitions from this state
                    for (var transitionIndex = 0; transitionIndex < srcState.TransitionCount; transitionIndex++)
                    {
                        var transition = srcState.GetTransition(transitionIndex);

                        if (transition.Group == group) continue;

                        var destWeight = weights[transition.DestinationStateIndex];
                        var weight = Weight.Sum(destWeight, Weight.Product(srcWeight, transition.Weight));

                        weights[transition.DestinationStateIndex] = weight;
                    }
                }

                return weights;
            }
        }
    }
}
