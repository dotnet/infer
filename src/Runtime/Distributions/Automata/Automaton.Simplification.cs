// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Core.Collections;
    using Microsoft.ML.Probabilistic.Utilities;

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
            var simplification = new Simplification(builder, this.PruneTransitionsWithLogWeightLessThan);
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
            var simplification = new Simplification(builder, this.PruneTransitionsWithLogWeightLessThan);
            if (simplification.RemoveDeadStates())
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
            private readonly double? pruneTransitionsWithLogWeightLessThan;

            /// <summary>
            /// Initializes new instance of <see cref="Simplification"/> class.
            /// </summary>
            public Simplification(Builder builder, double? pruneTransitionsWithLogWeightLessThan)
            {
                this.builder = builder;
                this.pruneTransitionsWithLogWeightLessThan = pruneTransitionsWithLogWeightLessThan;
            }

            /// <summary>
            /// Attempts to simplify the automaton using <see cref="Simplify"/> if the number of states
            /// exceeds <see cref="MaxStateCountBeforeSimplification"/>.
            /// </summary>
            public bool SimplifyIfNeeded()
            {
                if (this.builder.StatesCount > MaxStateCountBeforeSimplification ||
                    this.pruneTransitionsWithLogWeightLessThan != null)
                {
                    return this.Simplify();
                }

                return false;
            }

            /// <summary>
            /// Attempts to simplify the structure of the automaton, reducing the number of states and transitions.
            /// </summary>
            /// <remarks>
            /// <para>
            /// The simplification procedure works as follows:
            /// <list type="number">
            /// <item><description>
            /// If a pair of states has more than one transition between them, the transitions get merged.
            /// </description></item>
            /// <item><description>
            /// A part of the automaton that is a tree is found.
            /// </description></item>
            /// <item><description>
            /// States and transitions that don't belong to the found tree part are simply copied to the result.
            /// </description></item>
            /// <item><description>
            /// The found tree part is rebuild from scratch. The new tree is essentially a trie:
            /// for example, if the original tree has two paths accepting <c>"abc"</c> and one path accepting <c>"ab"</c>,
            /// the resulting tree has a single path accepting both <c>"ab"</c> and <c>"abc"</c>.
            /// </description></item>
            /// </list>
            /// </para>
            /// <para>The simplification procedure doesn't support automata with non-trivial loops.</para>
            /// </remarks>
            public bool Simplify()
            {
                this.MergeParallelTransitions();

                if (AutomatonHasNonTrivialLoops())
                {
                    return false; // TODO: make this stuff work with non-trivial loops
                }

                var generalizedTreeNodes = this.FindGeneralizedTrees();
                var sequenceToLogWeight = this.BuildAcceptedSequenceList(generalizedTreeNodes);
                this.builder.RemoveStates(generalizedTreeNodes, true);

                var firstNonCopiedStateIndex = this.builder.StatesCount;

                // Before we rebuild the tree part, we prune out the low probability sequences
                if (this.pruneTransitionsWithLogWeightLessThan != null)
                {
                    double logNorm = this.builder.GetAutomaton().GetLogNormalizer();
                    if (!double.IsInfinity(logNorm))
                    {
                        sequenceToLogWeight = sequenceToLogWeight
                            .Where(s => s.Weight.LogValue - logNorm >= this.pruneTransitionsWithLogWeightLessThan.Value)
                            .ToList();
                    }
                }

                foreach (var weightedSequence in sequenceToLogWeight)
                {
                    this.AddGeneralizedSequence(firstNonCopiedStateIndex, weightedSequence.Sequence, weightedSequence.Weight);
                }

                return true;
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
                var visited = new bool[builder.StatesCount];
                var (edgesStart, edges) = BuildReversedGraph();

                for (var i = 0; i < builder.StatesCount; ++i)
                {
                    if (builder[i].CanEnd)
                    {
                        Traverse(i);
                    }
                }

                return this.builder.RemoveStates(visited, false) > 0;

                void Traverse(int index)
                {
                    if (!visited[index])
                    {
                        visited[index] = true;
                        for (var i = edgesStart[index]; i < edgesStart[index + 1]; ++i)
                        {
                            Traverse(edges[i]);
                        }
                    }
                }

                (int[] edgesStart, int[] edges) BuildReversedGraph()
                {
                    // [edgesStart[i]; edgesStart[i+1]) is a range `edges` array which corresponds to incoming edges for state `i`
                    var edgesStart1 = new int[builder.StatesCount + 1];

                    // incomming edges for state. Represented as index of source state
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

            /// <summary>
            /// Removes transitions in the automaton whose log weight is less than the specified threshold.
            /// </summary>
            /// <remarks>
            /// Any states which are unreachable in the resulting automaton are also removed.
            /// </remarks>
            /// <param name="logWeightThreshold">The smallest log weight that a transition can have and not be removed.</param>
            public void RemoveTransitionsWithSmallWeights(double logWeightThreshold)
            {
                for (var i = 0; i < this.builder.StatesCount; ++i)
                {
                    var state = this.builder[i];
                    for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        if (iterator.Value.Weight.LogValue < logWeightThreshold)
                        {
                            iterator.Remove();
                        }
                    }
                }

                // Calculate reachable states and remove unreachable
                var visited = new bool[this.builder.StatesCount];
                Traverse(this.builder.StartStateIndex);
                this.builder.RemoveStates(visited, false);

                void Traverse(int stateIndex)
                {
                    if (!visited[stateIndex])
                    {
                        visited[stateIndex] = true;
                        var state = this.builder[stateIndex];
                        for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            Traverse(iterator.Value.DestinationStateIndex);
                        }
                    }
                }
            }

            /// <summary>
            /// Recursively checks if the automaton has non-trivial loops
            /// (i.e. loops consisting of more than one transition).
            /// </summary>
            /// <returns>
            /// <see langword="true"/> if a non-trivial loop has been found,
            /// <see langword="false"/> otherwise.
            /// </returns>
            private bool AutomatonHasNonTrivialLoops()
            {
                var stateInStack = new ArrayDictionary<bool>(this.builder.StatesCount);
                return HasNonTrivialLoops(this.builder.StartStateIndex);

                bool HasNonTrivialLoops(int stateIndex)
                {
                    if (stateInStack.TryGetValue(stateIndex, out var inStack))
                    {
                        return inStack;
                    }

                    stateInStack[stateIndex] = true;
                    for (var iterator = this.builder[stateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        if (transition.DestinationStateIndex != stateIndex)
                        {
                            if (HasNonTrivialLoops(transition.DestinationStateIndex))
                            {
                                return true;
                            }
                        }
                    }

                    stateInStack[stateIndex] = false;
                    return false;
                }
            }

            /// <summary>
            /// Labels each state with a value indicating whether the automaton having that state as the start state is a
            /// generalized tree (i.e. a tree with self-loops), which has single path from root leading to it
            /// (excluding the self-loops).
            /// </summary>
            /// <returns>A dictionary mapping state indices to the computed labels.</returns>
            private bool[] FindGeneralizedTrees()
            {
                var inDegree = CalcInDegree();
                var isGeneralizedTree = new bool[this.builder.StatesCount];
                Traverse(this.builder.StartStateIndex);
                return isGeneralizedTree;

                bool Traverse(int currentStateIndex)
                {
                    if (inDegree[currentStateIndex] > 1)
                    {
                        // Valid states can't have more than one incoming edge. Having more than one edge
                        // means that there's more than one path from root or that there's a loop somewhere.
                        // Both these cases are not considered generalized trees.
                        return false;
                    }

                    var currentIsGeneralizedTree = true;
                    for (var iterator = this.builder[currentStateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;

                        // Self-loops are allowed
                        if (transition.DestinationStateIndex != currentStateIndex)
                        {
                            currentIsGeneralizedTree &= Traverse(transition.DestinationStateIndex);
                        }
                    }

                    // if current node is not generalized tree then all transitions from it are also not trees
                    if (!currentIsGeneralizedTree)
                    {
                        for (var iterator = this.builder[currentStateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            MarkAsNonTree(iterator.Value.DestinationStateIndex);
                        }
                    }

                    isGeneralizedTree[currentStateIndex] = currentIsGeneralizedTree;
                    return currentIsGeneralizedTree;
                }

                void MarkAsNonTree(int currentStateIndex)
                {
                    if (isGeneralizedTree[currentStateIndex])
                    {
                        isGeneralizedTree[currentStateIndex] = false;
                        for (var iterator = this.builder[currentStateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            MarkAsNonTree(iterator.Value.DestinationStateIndex);
                        }
                    }
                }

                int[] CalcInDegree()
                {
                    var result = new int[this.builder.StatesCount];
                    for (var i = 0; i < this.builder.StatesCount; ++i)
                    {
                        for (var iterator = this.builder[i].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            // do not count self-loops
                            var destination = iterator.Value.DestinationStateIndex;
                            if (destination != i)
                            {
                                ++result[destination];
                            }
                        }
                    }

                    return result;
                }
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
                            if (transition1.DestinationStateIndex == transition2.DestinationStateIndex && transition1.Group == transition2.Group)
                            {
                                var removeTransition2 = false;
                                if (transition1.IsEpsilon && transition2.IsEpsilon)
                                {
                                    transition1.Weight = Weight.Sum(transition1.Weight, transition2.Weight);
                                    iterator1.Value = transition1;
                                    removeTransition2 = true;
                                }
                                else if (!transition1.IsEpsilon && !transition2.IsEpsilon)
                                {
                                    var newElementDistribution = new TElementDistribution();
                                    if (double.IsInfinity(transition1.Weight.Value) && double.IsInfinity(transition2.Weight.Value))
                                    {
                                        newElementDistribution.SetToSum(
                                            1.0, transition1.ElementDistribution.Value, 1.0, transition2.ElementDistribution.Value);
                                    }
                                    else
                                    {
                                        newElementDistribution.SetToSum(
                                            transition1.Weight.Value, transition1.ElementDistribution.Value, transition2.Weight.Value, transition2.ElementDistribution.Value);
                                    }

                                    transition1.ElementDistribution = newElementDistribution;
                                    transition1.Weight = Weight.Sum(transition1.Weight, transition2.Weight);
                                    iterator1.Value = transition1;
                                    removeTransition2 = true;
                                }

                                if (removeTransition2)
                                {
                                    iterator2.Remove();
                                }
                            }
                        }
                    }
                }
            }

            /// <summary>
            /// Builds a complete list of generalized sequences accepted by the simplifiable part of the automaton.
            /// </summary>
            /// <param name="generalizedTreeNodes">The state labels obtained from <see cref="FindGeneralizedTrees"/>.</param>
            /// <returns>The list of generalized sequences accepted by the simplifiable part of the automaton.</returns>
            private List<WeightedSequence> BuildAcceptedSequenceList(bool[] generalizedTreeNodes)
            {
                var sequenceToWeight = new List<WeightedSequence>();
                this.DoBuildAcceptedSequenceList(
                    this.builder.StartStateIndex,
                    generalizedTreeNodes,
                    sequenceToWeight,
                    new List<GeneralizedElement>(),
                    Weight.One);
                return sequenceToWeight;
            }


            private class StackItem
            {
            }

            private class ElementItem : StackItem
            {
                public readonly GeneralizedElement? Element;

                public ElementItem(GeneralizedElement? element) => this.Element = element;

                public override string ToString() => this.Element.ToString();
            }

            private class StateWeight : StackItem
            {
                public readonly int StateIndex;
                public readonly Weight Weight;

                public StateWeight(int stateIndexIndex, Weight weight)
                {
                    this.StateIndex = stateIndexIndex;
                    this.Weight = weight;
                }

                public override string ToString() => $"StateIndex: {this.StateIndex}, Weight: {Weight}";
            }

            /// <summary>
            /// Recursively builds a complete list of generalized sequences accepted by the simplifiable part of the automaton.
            /// </summary>
            /// <param name="stateIndex">The currently traversed state.</param>
            /// <param name="generalizedTreeNodes">The state labels obtained from <see cref="FindGeneralizedTrees"/>.</param>
            /// <param name="weightedSequences">The sequence list being built.</param>
            /// <param name="currentSequenceElements">The list of elements of the sequence currently being built.</param>
            /// <param name="currentWeight">The weight of the sequence currently being built.</param>
            private void DoBuildAcceptedSequenceList(
                int stateIndex,
                bool[] generalizedTreeNodes,
                List<WeightedSequence> weightedSequences,
                List<GeneralizedElement> currentSequenceElements,
                Weight currentWeight)
            {
                var stack = new Stack<StackItem>();
                stack.Push(new StateWeight(stateIndex, currentWeight));

                while (stack.Count > 0)
                {
                    var stackItem = stack.Pop();

                    if (stackItem is ElementItem elementItem)
                    {
                        if (elementItem.Element != null)
                            currentSequenceElements.Add(elementItem.Element.Value);
                        else
                            currentSequenceElements.RemoveAt(currentSequenceElements.Count - 1);
                        continue;
                    }

                    var stateAndWeight = stackItem as StateWeight;

                    stateIndex = stateAndWeight.StateIndex;
                    var state = this.builder[stateIndex];
                    currentWeight = stateAndWeight.Weight;

                    // Find a non-epsilon self-loop if there is one
                    Transition? selfLoop = null;
                    for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        if (transition.DestinationStateIndex == stateIndex)
                        {
                            Debug.Assert(
                                selfLoop == null,
                                "Multiple self-loops should have been merged by MergeParallelTransitions()");
                            selfLoop = transition;
                        }
                    }

                    // Push the found self-loop to the end of the current sequence
                    if (selfLoop != null)
                    {
                        currentSequenceElements.Add(new GeneralizedElement(
                            selfLoop.Value.ElementDistribution, selfLoop.Value.Group, selfLoop.Value.Weight));
                        stack.Push(new ElementItem(null));
                    }

                    // Can this state produce a sequence?
                    if (state.CanEnd && generalizedTreeNodes[stateIndex])
                    {
                        var sequence = new GeneralizedSequence(currentSequenceElements);
                        // TODO: use immutable data structure instead of copying sequences
                        weightedSequences.Add(new WeightedSequence(sequence, Weight.Product(currentWeight, state.EndWeight)));
                    }

                    // Traverse the outgoing transitions
                    for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        // Skip self-loops & disallowed states
                        if (transition.DestinationStateIndex == stateIndex ||
                            !generalizedTreeNodes[transition.DestinationStateIndex])
                        {
                            continue;
                        }

                        if (!transition.IsEpsilon)
                        {
                            // Non-epsilon transitions contribute to the sequence
                            stack.Push(new ElementItem(null));
                        }

                        stack.Push(
                            new StateWeight(
                                transition.DestinationStateIndex,
                                Weight.Product(currentWeight, transition.Weight)));

                        if (!transition.IsEpsilon)
                        {
                            stack.Push(
                                new ElementItem(
                                    new GeneralizedElement(transition.ElementDistribution, transition.Group, null)));
                        }
                    }
                }
            }

            /// <summary>
            /// Increases the value of this automaton on <paramref name="sequence"/> by <paramref name="weight"/>.
            /// </summary>
            /// <param name="firstAllowedStateIndex">The minimum index of an existing state that can be used for the sequence.</param>
            /// <param name="sequence">The generalized sequence.</param>
            /// <param name="weight">The weight of the sequence.</param>
            /// <remarks>
            /// This function attempts to add as few new states and transitions as possible.
            /// Its implementation is conceptually similar to adding string to a trie.
            /// </remarks>
            private void AddGeneralizedSequence(int firstAllowedStateIndex, Simplification.GeneralizedSequence sequence, Weight weight)
            {
                // First, try to add at start state
                var isFreshStartState = !this.builder.Start.HasTransitions && this.builder.Start.EndWeight.IsZero;
                if (this.DoAddGeneralizedSequence(this.builder.Start.Index, isFreshStartState, false, firstAllowedStateIndex, 0, sequence, weight))
                {
                    return;
                }

                // Branch the start state
                var builder = this.builder;
                var oldStart = builder.Start;
                var start = builder.AddState();
                builder.StartStateIndex = start.Index;
                var otherBranch = builder.AddState();
                builder.Start.AddEpsilonTransition(Weight.One, oldStart.Index);
                builder.Start.AddEpsilonTransition(Weight.One, otherBranch.Index);

                // This should always work
                var success = this.DoAddGeneralizedSequence(otherBranch.Index, true, false, firstAllowedStateIndex, 0, sequence, weight);
                Debug.Assert(success, "This call must always succeed.");
            }

            /// <summary>
            /// Recursively increases the value of this automaton on <paramref name="sequence"/> by <paramref name="weight"/>.
            /// </summary>
            /// <param name="stateIndex">Index of currently traversed state.</param>
            /// <param name="isNewState">Indicates whether state <paramref name="stateIndex"/> was just created.</param>
            /// <param name="selfLoopAlreadyMatched">Indicates whether self-loop on state <paramref name="stateIndex"/> was just matched.</param>
            /// <param name="firstAllowedStateIndex">The minimum index of an existing state that can be used for the sequence.</param>
            /// <param name="currentSequencePos">The current position in the generalized sequence.</param>
            /// <param name="sequence">The generalized sequence.</param>
            /// <param name="weight">The weight of the sequence.</param>
            /// <returns>
            /// <see langword="true"/> if the subsequence starting at <paramref name="currentSequencePos"/> has been successfully merged in,
            /// <see langword="false"/> otherwise.
            /// </returns>
            /// <remarks>
            /// This function attempts to add as few new states and transitions as possible.
            /// Its implementation is conceptually similar to adding string to a trie.
            /// </remarks>
            private bool DoAddGeneralizedSequence(
                int stateIndex,
                bool isNewState,
                bool selfLoopAlreadyMatched,
                int firstAllowedStateIndex,
                int currentSequencePos,
                GeneralizedSequence sequence,
                Weight weight)
            {
                bool success;
                var builder = this.builder;
                var state = builder[stateIndex];

                if (currentSequencePos == sequence.Count)
                {
                    if (!selfLoopAlreadyMatched)
                    {
                        // We can't finish in a state with a self-loop
                        for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            if (iterator.Value.DestinationStateIndex == state.Index)
                            {
                                return false;
                            }
                        }
                    }

                    state.SetEndWeight(Weight.Sum(state.EndWeight, weight));
                    return true;
                }

                var element = sequence[currentSequencePos];

                // Treat self-loops elements separately
                if (element.LoopWeight.HasValue)
                {
                    if (selfLoopAlreadyMatched)
                    {
                        // Previous element was also a self-loop, we should try to find an espilon transition
                        for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            var transition = iterator.Value;
                            if (transition.DestinationStateIndex != state.Index &&
                                transition.IsEpsilon &&
                                transition.DestinationStateIndex >= firstAllowedStateIndex)
                            {
                                if (this.DoAddGeneralizedSequence(
                                    transition.DestinationStateIndex,
                                    false,
                                    false,
                                    firstAllowedStateIndex,
                                    currentSequencePos,
                                    sequence,
                                    Weight.Product(weight, Weight.Inverse(transition.Weight))))
                                {
                                    return true;
                                }
                            }
                        }

                        // Epsilon transition not found, let's create a new one
                        var destination = state.AddEpsilonTransition(Weight.One);
                        success = this.DoAddGeneralizedSequence(
                            destination.Index, true, false, firstAllowedStateIndex, currentSequencePos, sequence, weight);
                        Debug.Assert(success, "This call must always succeed.");
                        return true;
                    }

                    // Find a matching self-loop
                    for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;

                        if (transition.IsEpsilon && transition.DestinationStateIndex != state.Index && transition.DestinationStateIndex >= firstAllowedStateIndex)
                        {
                            // Try this epsilon transition
                            if (this.DoAddGeneralizedSequence(
                                transition.DestinationStateIndex, false, false, firstAllowedStateIndex, currentSequencePos, sequence, weight))
                            {
                                return true;
                            }
                        }

                        // Is it a self-loop?
                        if (transition.DestinationStateIndex == state.Index)
                        {
                            // Do self-loops match?
                            if ((transition.Weight == element.LoopWeight.Value) &&
                                (element.Group == transition.Group) &&
                                ((transition.IsEpsilon && element.IsEpsilonSelfLoop) || (!transition.IsEpsilon && !element.IsEpsilonSelfLoop && transition.ElementDistribution.Equals(element.ElementDistribution))))
                            {
                                // Skip the element in the sequence, remain in the same state
                                success = this.DoAddGeneralizedSequence(
                                    stateIndex, false, true, firstAllowedStateIndex, currentSequencePos + 1, sequence, weight);
                                Debug.Assert(success, "This call must always succeed.");
                                return true;
                            }

                            // StateIndex also has a self-loop, but the two doesn't match
                            return false;
                        }
                    }

                    if (!isNewState)
                    {
                        // Can't add self-loop to an existing state, it will change the language accepted by the state
                        return false;
                    }

                    // Add a new self-loop
                    state.AddTransition(element.ElementDistribution, element.LoopWeight.Value, stateIndex, element.Group);
                    success = this.DoAddGeneralizedSequence(stateIndex, false, true, firstAllowedStateIndex, currentSequencePos + 1, sequence, weight);
                    Debug.Assert(success, "This call must always succeed.");
                    return true;
                }

                // Try to find a transition for the element
                for (var iterator = state.TransitionIterator; iterator.Ok; iterator.Next())
                {
                    var transition = iterator.Value;

                    if (transition.IsEpsilon && transition.DestinationStateIndex != state.Index && transition.DestinationStateIndex >= firstAllowedStateIndex)
                    {
                        // Try this epsilon transition
                        if (this.DoAddGeneralizedSequence(
                            transition.DestinationStateIndex, false, false, firstAllowedStateIndex, currentSequencePos, sequence, weight))
                        {
                            return true;
                        }
                    }

                    // Is it a self-loop?
                    if (transition.DestinationStateIndex == state.Index)
                    {
                        if (selfLoopAlreadyMatched)
                        {
                            // The self-loop was checked or added by the caller
                            continue;
                        }

                        // Can't go through an existing self-loop, it will allow undesired sequences to be accepted
                        return false;
                    }

                    if (transition.DestinationStateIndex < firstAllowedStateIndex ||
                        element.Group != transition.Group ||
                        !element.ElementDistribution.Equals(transition.ElementDistribution))
                    {
                        continue;
                    }

                    // Skip the element in the sequence, move to the destination state
                    // Weight of the existing transition must be taken into account
                    // This case can fail if the next element is a self-loop and the destination state already has a different one
                    if (this.DoAddGeneralizedSequence(
                        transition.DestinationStateIndex,
                        false,
                        false,
                        firstAllowedStateIndex,
                        currentSequencePos + 1,
                        sequence,
                        Weight.Product(weight, Weight.Inverse(transition.Weight))))
                    {
                        return true;
                    }
                }

                // Add a new transition
                var newChild = state.AddTransition(element.ElementDistribution, Weight.One, null, element.Group);
                success = this.DoAddGeneralizedSequence(
                    newChild.Index, true, false, firstAllowedStateIndex, currentSequencePos + 1, sequence, weight);
                Debug.Assert(success, "This call must always succeed.");
                return true;
            }

            /// <summary>
            /// Represents an element of a generalized sequence,
            /// i.e. a distribution over a single symbol or a weighted self-loop.
            /// </summary>
            public struct GeneralizedElement
            {
                /// <summary>
                /// Initializes a new instance of the <see cref="GeneralizedElement"/> struct.
                /// </summary>
                /// <param name="elementDistribution">The element distribution associated with the generalized element.</param>
                /// <param name="group">The group associated with the generalized element.</param>
                /// <param name="loopWeight">
                /// The loop weight associated with the generalized element, <see langword="null"/> if the element does not represent a self-loop.
                /// </param>
                public GeneralizedElement(Option<TElementDistribution> elementDistribution, int group, Weight? loopWeight)
                    : this()
                {
                    Debug.Assert(
                        elementDistribution.HasValue || loopWeight.HasValue,
                        "Epsilon elements are only allowed in combination with self-loops.");

                    this.ElementDistribution = elementDistribution;
                    this.Group = group;
                    this.LoopWeight = loopWeight;
                }

                /// <summary>
                /// Gets the element distribution associated with the generalized element.
                /// </summary>
                public Option<TElementDistribution> ElementDistribution { get; }

                /// <summary>
                /// Gets a value indicating whether this element corresponds to an epsilon self-loop.
                /// </summary>
                public bool IsEpsilonSelfLoop => !this.ElementDistribution.HasValue && this.LoopWeight.HasValue;

                /// <summary>
                /// Gets the group associated with the generalized element.
                /// </summary>
                public int Group { get; }

                /// <summary>
                /// Gets the loop weight associated with the generalized element,
                /// <see langword="null"/> if the element does not represent a self-loop.
                /// </summary>
                public Weight? LoopWeight { get; }

                /// <summary>
                /// Gets the string representation of the generalized element.
                /// </summary>
                /// <returns>The string representation of the generalized element.</returns>
                public override string ToString()
                {
                    var elementDistributionAsString =
                        this.ElementDistribution.Value.IsPointMass
                            ? this.ElementDistribution.Value.Point.ToString()
                            : this.ElementDistribution.ToString();
                    var groupString = this.Group == 0 ? string.Empty : $"#{this.Group}";
                    return
                        this.LoopWeight.HasValue
                            ? $"{groupString}{elementDistributionAsString}*({this.LoopWeight.Value})"
                            : $"{groupString}{elementDistributionAsString}";
                }
            }

            /// <summary>
            /// Represents a sequence of generalized elements.
            /// </summary>
            public class GeneralizedSequence
            {
                /// <summary>
                /// The sequence elements.
                /// </summary>
                private readonly List<GeneralizedElement> elements;

                /// <summary>
                /// Initializes a new instance of the <see cref="GeneralizedSequence"/> class.
                /// </summary>
                /// <param name="elements">The sequence elements.</param>
                public GeneralizedSequence(IEnumerable<GeneralizedElement> elements)
                {
                    this.elements = new List<GeneralizedElement>(elements);
                }

                /// <summary>
                /// Gets the number of elements in the sequence.
                /// </summary>
                public int Count => this.elements.Count;

                /// <summary>
                /// Gets the sequence element with the specified index.
                /// </summary>
                /// <param name="index">The element index.</param>
                /// <returns>The element at the given index.</returns>
                public GeneralizedElement this[int index] => this.elements[index];

                /// <summary>
                /// Gets the string representation of the sequence.
                /// </summary>
                /// <returns>The string representation of the sequence.</returns>
                public override string ToString()
                {
                    var stringBuilder = new StringBuilder();
                    foreach (var element in this.elements)
                    {
                        stringBuilder.Append(element);
                    }

                    return stringBuilder.ToString();
                }
            }

            public struct WeightedSequence
            {
                /// <summary>
                /// Initializes a new instance of the <see cref="WeightedSequence"/> struct.
                /// </summary>
                /// <param name="sequence">The <see cref="GeneralizedSequence"/></param>
                /// <param name="weight">The <see cref="Weight"/> for the specified sequence</param>
                public WeightedSequence(GeneralizedSequence sequence, Weight weight)
                    : this()
                {
                    this.Sequence = sequence;
                    this.Weight = weight;
                }

                /// <summary>
                /// Gets or sets the <see cref="GeneralizedSequence"/>.
                /// </summary>
                public readonly GeneralizedSequence Sequence;

                /// <summary>
                /// Gets or sets the predicted probability.
                /// </summary>
                public readonly Weight Weight;

                /// <summary>
                /// Gets the string representation of this <see cref="WeightedSequence"/>.
                /// </summary>
                /// <returns>The string representation of the <see cref="WeightedSequence"/>.</returns>
                public override string ToString() => $"[{this.Sequence},{this.Weight}]";

                /// <summary>
                /// Checks if this object is equal to <paramref name="obj"/>.
                /// </summary>
                /// <param name="obj">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if this object is equal to <paramref name="obj"/>,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public override bool Equals(object obj) =>
                    obj is WeightedSequence weightedSequence &&
                    object.Equals(this.Sequence, weightedSequence.Sequence) &&
                    object.Equals(this.Weight, weightedSequence.Weight);

                /// <summary>
                /// Computes the hash code of this object.
                /// </summary>
                /// <returns>The computed hash code.</returns>
                public override int GetHashCode()
                {
                    int result = Hash.Start;

                    result = Hash.Combine(result, this.Sequence.GetHashCode());
                    result = Hash.Combine(result, this.Weight.GetHashCode());

                    return result;
                }
            }
        }
    }
}
