// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
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
        public bool Simplify()
        {
            var builder = Builder.FromAutomaton(this);
            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            if (simplification.Simplify())
            {
                this.Data = builder.GetData();
                return true;
            }

            return false;
        }

        /// <summary>
        /// Optimizes the automaton by removing all states which can't reach end states.
        /// </summary>
        public bool RemoveDeadStates()
        {
            var builder = Builder.FromAutomaton(this);
            var initialStatesCount = builder.StatesCount;
            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            simplification.RemoveDeadStates();
            if (builder.StatesCount != initialStatesCount)
            {
                this.Data = builder.GetData();
                return true;
            }

            return false;
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
                        transition1.Weight = Weight.Sum(transition1.Weight, transition2.Weight);
                        return transition1;
                    }
                    
                    if (!transition1.IsEpsilon && !transition2.IsEpsilon)
                    {
                        var newElementDistribution = new TElementDistribution();
                        if (transition1.Weight.IsInfinity && transition2.Weight.IsInfinity)
                        {
                            newElementDistribution.SetToSum(
                                1.0,
                                transition1.ElementDistribution.Value,
                                1.0,
                                transition2.ElementDistribution.Value);
                        }
                        else
                        {
                            newElementDistribution.SetToSum(
                                transition1.Weight.Value,
                                transition1.ElementDistribution.Value,
                                transition2.Weight.Value,
                                transition2.ElementDistribution.Value);
                        }

                        return new Transition(
                            newElementDistribution,
                            Weight.Sum(transition1.Weight, transition2.Weight),
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
                    for (var iterator1 = state.TransitionIterator; iterator1.Ok; iterator1.Next())
                    {
                        var transition1 = iterator1.Value;

                        // ignore non-tree nodes and self-loops
                        if (!isTreeNode[transition1.DestinationStateIndex] ||
                            transition1.DestinationStateIndex == stateIndex)
                        {
                            continue;
                        }

                        var iterator2 = iterator1;
                        iterator2.Next();
                        for (; iterator2.Ok; iterator2.Next())
                        {
                            var transition2 = iterator2.Value;

                            Debug.Assert(
                                transition1.DestinationStateIndex != transition2.DestinationStateIndex,
                                "Parallel transitions must be merged earlier by MergeParallelTransitions()");

                            // ignore non-tree nodes and self-loops
                            if (isTreeNode[transition2.DestinationStateIndex] &&
                                transition2.DestinationStateIndex != stateIndex &&
                                CanMerge(transition1, transition2))
                            {
                                MergeStates(
                                    transition1.DestinationStateIndex,
                                    transition2.DestinationStateIndex,
                                    Weight.Product(transition2.Weight, Weight.Inverse(transition1.Weight)));
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

                bool CanMerge(Transition transition1, Transition transition2)
                {
                    // Check that group and element distribution match
                    if (transition1.Group != transition2.Group ||
                        !EqualDistributions(transition1.ElementDistribution, transition2.ElementDistribution))
                    {
                        return false;
                    }

                    var selfLoop1 = TryFindSelfLoop(transition1.DestinationStateIndex);
                    var selfLoop2 = TryFindSelfLoop(transition2.DestinationStateIndex);

                    // Can merge only if both destination states don't have self-loops
                    // or these loops are exactly the same.
                    return
                        selfLoop1.HasValue == selfLoop2.HasValue
                        && (!selfLoop1.HasValue
                            || (selfLoop1.Value.Group == selfLoop2.Value.Group &&
                                selfLoop1.Value.Weight == selfLoop2.Value.Weight &&
                                EqualDistributions(selfLoop1.Value.ElementDistribution, selfLoop2.Value.ElementDistribution)));
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
                        var state2EndWeight = Weight.Product(state2WeightMultiplier, state2.EndWeight);
                        state1.SetEndWeight(Weight.Sum(state1.EndWeight, state2EndWeight));
                    }

                    // Copy all transitions except self-loop.
                    // Self-loop doesn't need to be copied because two states can be merged only if they have
                    // identical self-loops. Thus if there is self-loop in state2 then it is also already
                    // present in state1.
                    for (var iterator = state2.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        if (transition.DestinationStateIndex != state2Index)
                        {
                            transition.Weight = Weight.Product(transition.Weight, state2WeightMultiplier);
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
            /// <remarks>
            /// This function looks a lot like <see cref="ComputeEndStateReachability"/>. But unfortunately operates on
            /// a different graph representation.
            /// </remarks>
            public bool RemoveDeadStates()
            {
                var builder = this.builder;
                var (edgesStart, edges) = BuildReversedGraph();

                //// Now run a depth-first search to label all reachable nodes
                var stack = new Stack<int>();
                var visited = new bool[builder.StatesCount];
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
                    var edgesStart1 = new int[builder.StatesCount + 1];

                    // incoming edges for state. Represented as index of source state
                    var edges1 = new int[builder.TransitionsCount];

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
