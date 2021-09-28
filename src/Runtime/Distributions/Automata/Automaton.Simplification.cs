// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Collections;


    /// <content>
    /// Contains classes and methods for automata simplification.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Attempts to simplify the structure of the automaton, reducing the number of states and transitions.
        /// </summary>
        /// <param name="result">Result automaton. Simplified automaton, if the operation was successful, current automaton otherwise.</param>
        /// <returns><see langword="true"/> if the simplification was successful, <see langword="false"/> otherwise.</returns>
        public bool Simplify(out TThis result)
        {
            var builder = Builder.FromAutomaton(this);
            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            if (simplification.Simplify())
            {
                result = WithData(builder.GetData());
                return true;
            }

            result = (TThis)this;
            return false;
        }

        /// <summary>
        /// Optimizes the automaton by removing all states which can't reach end states.
        /// </summary>
        /// <returns>Result automaton. Simplified automaton, if there were states to be removed, current automaton otherwise.</returns>
        public TThis RemoveDeadStates()
        {
            var builder = Builder.FromAutomaton(this);
            var initialStatesCount = builder.StatesCount;
            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            simplification.RemoveDeadStates();
            if (builder.StatesCount != initialStatesCount)
                return WithData(builder.GetData());
            else
                return (TThis)this;
        }

        /// <summary>
        /// Helper class which which implements automaton simplification.
        /// </summary>
        public class Simplification
        {
            /// <summary>
            /// Automaton builder which contains automaton during simplification procedure.
            /// </summary>
            private readonly Builder builder;

            /// <summary>
            /// Gets or sets a value for truncating small weights.
            /// If non-null, any transition whose weight falls below this value in a normalized
            /// automaton will be removed.
            /// </summary>
            private readonly double? pruneStatesWithLogEndWeightLessThan;

            /// <summary>
            /// Initializes new instance of <see cref="Simplification"/> class.
            /// </summary>
            public Simplification(Builder builder, double? pruneStatesWithLogEndWeightLessThan)
            {
                this.builder = builder;
                this.pruneStatesWithLogEndWeightLessThan = pruneStatesWithLogEndWeightLessThan;
            }

            /// <summary>
            /// Attempts to simplify the automaton using <see cref="Simplify"/> if the number of states
            /// exceeds <see cref="MaxStateCountBeforeSimplification"/>.
            /// </summary>
            public bool SimplifyIfNeeded()
            {
                if (this.builder.StatesCount > MaxStateCountBeforeSimplification ||
                    this.pruneStatesWithLogEndWeightLessThan != null)
                {
                    return this.Simplify();
                }

                return false;
            }

            /// <summary>
            /// Attempts to simplify the structure of the automaton, reducing the number of states and transitions.
            /// </summary>
            /// <remarks>
            /// Only generalized tree part of automaton is simplified. Generalized tree is a tree with self-loops
            /// allowed. Any non-trivial loops (consisting of more than 1 state) are left untouched.
            /// 
            /// The simplification procedure works as follows:
            ///   * If a pair of states has more than one transition between them, the transitions get merged.
            ///   * A part of the automaton that is a tree is found.
            ///     For example in this automaton:
            ///       a--->b--->c--->d---\
            ///       v              ^   |
            ///       e--->f--->g     \--/
            ///       |          
            ///       v
            ///       h--->i--->j-->k
            ///            ^    |
            ///             \--/
            ///     Nodes "a" to "h" form the tree. Nodes "i" to "k" are not part of the tree.
            ///     Note 1: h has child nodes which form loop (i, j) but is still considered a part of tree.
            ///     Because path leading to it from root has no loops.
            ///     Note 2: d is also considered to be a part of tree. Self-loops are allowed.
            ///   * After that, states in the tree part of automaton are recursively merged if they are compatible.
            ///     Two states are considered compatible if path from root to them has exactly same element
            ///     distributions and groups on transitions, and they have compatible self-loops. Weights in
            ///     transitions from root can be different - weights will be adjusted as necessary.
            ///     For example if in previous automaton transitions (a-b) and (a-e) have the same
            ///     element distribution then result will look like this:
            ///
            ///            a--->b--->c--->d---\
            ///            | \            ^   |
            ///       e    |   f--->g      \--/
            ///            |          
            ///            v
            ///            h--->i--->j-->k
            ///                 ^    |
            ///                  \--/
            ///   * If pruneStatesWithLogEndWeightLessThan is not null then log normalizer and graph
            ///     condensation computed. All states which do not have high enough end probability are deleted
            /// </remarks>
            public bool Simplify()
            {
                var initialStatesCount = builder.StatesCount;
                var initialTransitionsCount = builder.TransitionsCount;

                this.MergeParallelTransitions();
                this.MergeTrees();

                if (this.pruneStatesWithLogEndWeightLessThan != null)
                {
                    this.RemoveLowWeightEndStates();
                }

                return builder.StatesCount != initialStatesCount ||
                       builder.TransitionsCount != initialTransitionsCount;
            }

            /// <summary>
            /// Merges outgoing transitions with the same destination state.
            /// </summary>
            public void MergeParallelTransitions()
            {
                for (var stateIndex = 0; stateIndex < this.builder.StatesCount; ++stateIndex)
                {
                    var state = this.builder[stateIndex];
                    for (var iterator1 = state.TransitionIterator; iterator1.Ok; iterator1.Next())
                    {
                        var transition1 = iterator1.Value;
                        var iterator2 = iterator1;
                        iterator2.Next();
                        for (; iterator2.Ok; iterator2.Next())
                        {
                            var transition2 = iterator2.Value;
                            var mergedTransition = TryMergeTransitions(transition1, transition2);
                            if (mergedTransition != null)
                            {
                                iterator1.Value = mergedTransition.Value;
                                transition1 = mergedTransition.Value;
                                iterator2.Remove();
                            }

                        }
                    }
                }

                Transition? TryMergeTransitions(Transition transition1, Transition transition2)
                {
                    if (transition1.DestinationStateIndex != transition2.DestinationStateIndex ||
                        transition1.Group != transition2.Group)
                    {
                        return null;
                    }

                    if (transition1.IsEpsilon && transition2.IsEpsilon)
                    {
                        return transition1.With(weight: transition1.Weight + transition2.Weight);
                    }
                    
                    if (!transition1.IsEpsilon && !transition2.IsEpsilon)
                    {
                        TElementDistribution newElementDistribution;
                        if (transition1.Weight.IsInfinity && transition2.Weight.IsInfinity)
                        {
                            newElementDistribution = transition1.ElementDistribution.Value.Sum(1.0,
                                transition2.ElementDistribution.Value,
                                1.0);
                        }
                        else if (transition1.Weight > transition2.Weight)
                        {
                            newElementDistribution = transition1.ElementDistribution.Value.Sum(1.0,
                                transition2.ElementDistribution.Value,
                                (transition2.Weight / transition1.Weight).Value);
                        }
                        else
                        {
                            newElementDistribution = transition1.ElementDistribution.Value.Sum((transition1.Weight / transition2.Weight).Value,
                                transition2.ElementDistribution.Value,
                                1.0);
                        }

                        return new Transition(
                            newElementDistribution,
                            transition1.Weight + transition2.Weight,
                            transition1.DestinationStateIndex,
                            transition1.Group);
                    }

                    return null;
                }
            }

            public void MergeTrees()
            {
                var builder = this.builder;
                var isRemovedNode = new bool[builder.StatesCount];
                var isTreeNode = FindTreeNodes();

                var stack = new Stack<int>();
                stack.Push(builder.StartStateIndex);

                while (stack.Count > 0)
                {
                    var stateIndex = stack.Pop();
                    var state = builder[stateIndex];

                    // Transitions to non-tree nodes and self-loops should be ignored
                    bool IsMergeableTransition(Transition t) =>
                        isTreeNode[t.DestinationStateIndex] && t.DestinationStateIndex != stateIndex;

                    for (var iterator1 = state.TransitionIterator; iterator1.Ok; iterator1.Next())
                    {
                        var transition1 = iterator1.Value;

                        // ignore non-tree nodes and self-loops
                        if (!IsMergeableTransition(transition1))
                        {
                            continue;
                        }

                        // If it is an epsilon transition then try to merge with current state first
                        // Note: group doesn't matter for epsilon transitions (in generalized trees)
                        if (transition1.IsEpsilon &&
                            CanMergeStates(stateIndex, transition1.DestinationStateIndex))
                        {
                            // All transitions from transition1.DestinationStateIndex will be inserted
                            // into current state. And will be iterated by iterator1 without special treatment.
                            MergeStates(stateIndex, transition1.DestinationStateIndex, transition1.Weight);
                            isRemovedNode[transition1.DestinationStateIndex] = true;
                            iterator1.Remove();
                            continue;
                        }

                        // Try to find transitions with which this one can be merged
                        var iterator2 = iterator1;
                        iterator2.Next();
                        for (; iterator2.Ok; iterator2.Next())
                        {
                            var transition2 = iterator2.Value;

                            Debug.Assert(
                                transition1.DestinationStateIndex != transition2.DestinationStateIndex,
                                "Parallel transitions must be merged earlier by MergeParallelTransitions()");

                            // ignore non-tree nodes and self-loops
                            if (IsMergeableTransition(transition2) &&
                                CanMergeDestinations(transition1, transition2))
                            {
                                MergeStates(
                                    transition1.DestinationStateIndex,
                                    transition2.DestinationStateIndex,
                                    transition2.Weight * Weight.Inverse(transition1.Weight));
                                isRemovedNode[transition2.DestinationStateIndex] = true;
                                iterator2.Remove();
                            }
                        }

                        stack.Push(transition1.DestinationStateIndex);
                    }
                }

                builder.RemoveStates(isRemovedNode, true);
                return;

                // Returns a boolean array in which for each automaton state a "isTree" flag is stored.
                // State is considered to be tree node if its in degree = 1 and it's parent is also a tree node.
                bool[] FindTreeNodes()
                {
                    var inDegree = new int[builder.StatesCount];
                    for (var i = 0; i < builder.StatesCount; ++i)
                    {
                        for (var iterator = builder[i].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            var destinationIndex = iterator.Value.DestinationStateIndex;
                            // Ignore self-loops
                            if (destinationIndex != i)
                            {
                                ++inDegree[destinationIndex];
                            }
                        }
                    }

                    var result = new bool[builder.StatesCount];

                    var treeSearchStack = new Stack<int>();
                    treeSearchStack.Push(builder.StartStateIndex);
                    while (treeSearchStack.Count > 0)
                    {
                        var stateIndex = treeSearchStack.Pop();
                        result[stateIndex] = true;
                        for (var iterator = builder[stateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            var destinationIndex = iterator.Value.DestinationStateIndex;
                            if (destinationIndex != stateIndex && inDegree[destinationIndex] == 1)
                            {
                                treeSearchStack.Push(destinationIndex);
                            }
                        }
                    }

                    return result;
                }

                bool CanMergeStates(int stateIndex1, int stateIndex2)
                {
                    var selfLoop1 = TryFindSelfLoop(stateIndex1);
                    var selfLoop2 = TryFindSelfLoop(stateIndex2);

                    // Can merge only if both destination states don't have self-loops
                    // or these loops are exactly the same.
                    return
                        (!selfLoop1.HasValue && !selfLoop2.HasValue)
                        || (selfLoop1.HasValue &&
                            selfLoop2.HasValue &&
                            selfLoop1.Value.Group == selfLoop2.Value.Group &&
                            selfLoop1.Value.Weight == selfLoop2.Value.Weight &&
                            EqualDistributions(selfLoop1.Value.ElementDistribution, selfLoop2.Value.ElementDistribution));
                }

                bool CanMergeDestinations(Transition transition1, Transition transition2)
                {
                    // Check that group and element distribution match
                    if (transition1.Group != transition2.Group ||
                        !EqualDistributions(transition1.ElementDistribution, transition2.ElementDistribution))
                    {
                        return false;
                    }

                    return CanMergeStates(transition1.DestinationStateIndex, transition2.DestinationStateIndex);
                }

                // Compares element distributions in transition. Epsilon transitions are considered equal.
                bool EqualDistributions(Option<TElementDistribution> dist1, Option<TElementDistribution> dist2) =>
                    dist1.HasValue == dist2.HasValue &&
                    (!dist1.HasValue || dist1.Value.Equals(dist2.Value));

                // Finds transition which points to state itself
                // It is assumed that there's only one such transition
                Transition? TryFindSelfLoop(int stateIndex)
                {
                    for (var iterator = builder[stateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        if (iterator.Value.DestinationStateIndex == stateIndex)
                        {
                            return iterator.Value;
                        }
                    }

                    return null;
                }

                // Adds EndWeight and all transitions from state2 into state1.
                // All state2 weights are multiplied by state2WeightMultiplier
                void MergeStates(int state1Index, int state2Index, Weight state2WeightMultiplier)
                {
                    var state1 = builder[state1Index];
                    var state2 = builder[state2Index];

                    // sum end weights
                    if (!state2.EndWeight.IsZero)
                    {
                        var state2EndWeight = state2WeightMultiplier * state2.EndWeight;
                        state1.SetEndWeight(state1.EndWeight + state2EndWeight);
                    }

                    // Copy all transitions
                    for (var iterator = state2.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        if (transition.DestinationStateIndex != state2Index)
                        {
                            // Self-loop is not copied: it is already present in state1 and is absolutely
                            // compatible: it has the same distribution and weight
                            transition = transition.With(weight: transition.Weight * state2WeightMultiplier);
                            state1.AddTransition(transition);
                        }
                    }

                }
            }

            public void RemoveLowWeightEndStates()
            {
                // Note: no production work-load currently uses PruneStatesWithLogEndWeightLessThan
                // So having no implementation is not big deal
                // TODO:
                //   - set automaton to epsilon-closure
                //   - compute Condensation & logNormalizer
                //   - remove all end states with WeightFromRoot * EndWeight < PruneStatesWithLogEndWeightLessThan - LogNormalizer
                //   - remove dead states
                //   - write lots of unit-tests
            }

            /// <summary>
            /// Optimizes the automaton by removing all states which can't reach end states.
            /// </summary>
            public bool RemoveDeadStates()
            {
                var builder = this.builder;
                var (edgesStart, edges) = BuildReversedGraph();

                //// Now run a depth-first search to label all reachable nodes
                var (stack, visited) = PreallocatedAutomataObjects.LeaseRemoveDeadStatesState(builder.StatesCount);

                for (var i = 0; i < builder.StatesCount; ++i)
                {
                    if (!visited[i] && builder[i].CanEnd)
                    {
                        visited[i] = true;
                        stack.Push(i);
                        while (stack.Count != 0)
                        {
                            var stateIndex = stack.Pop();
                            for (var j = edgesStart[stateIndex]; j < edgesStart[stateIndex + 1]; ++j)
                            {
                                var destinationIndex = edges[j];
                                if (!visited[destinationIndex])
                                {
                                    visited[destinationIndex] = true;
                                    stack.Push(destinationIndex);
                                }
                            }
                        }
                    }
                }

                if (!visited[builder.StartStateIndex])
                {
                    builder.Clear();
                    builder.StartStateIndex = builder.AddState().Index;
                    return true;
                }

                return this.builder.RemoveStates(visited, false) > 0;

                (int[] edgesStart, int[] edges) BuildReversedGraph()
                {
                    // [edgesStart[i]; edgesStart[i+1]) is a range `edges` array which corresponds to incoming edges for state `i`
                    // edges1 is incoming edges for state. Represented as index of source state
                    var (edgesStart1, edges1) = PreallocatedAutomataObjects.LeaseBuildReversedGraphState(
                        builder.StatesCount + 1, builder.TransitionsCount);

                    // first populate edges
                    for (var i = 0; i < builder.StatesCount; ++i)
                    {
                        for (var iterator = builder[i].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            ++edgesStart1[iterator.Value.DestinationStateIndex];
                        }
                    }

                    // calculate commutative sums. Now edgesStart[i] contains end of range
                    for (var i = 0; i < builder.StatesCount; ++i)
                    {
                        edgesStart1[i + 1] += edgesStart1[i];
                    }

                    // Fill ranges and adjust start indices. Now edgesStart[i] contains begining of the range
                    for (var i = 0; i < builder.StatesCount; ++i)
                    {
                        for (var iterator = builder[i].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            var index = --edgesStart1[iterator.Value.DestinationStateIndex];
                            edges1[index] = i;
                        }
                    }

                    return (edgesStart1, edges1);
                }
            }
        }
    }
}
