// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains classes and methods for automata simplification.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Tests whether the automaton is deterministic,
        /// i.e. it's epsilon free and for every state and every element there is at most one transition that allows for that element.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the automaton is deterministic,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public bool IsDeterministic()
        {
            //// We can't track whether the automaton is deterministic while adding/updating transitions
            //// because element distributions are not immutable.
            
            if (!this.IsEpsilonFree)
            {
                return false;
            }
            
            for (int stateId = 0; stateId < this.States.Count; ++stateId)
            {
                var state = this.States[stateId];
                
                // There should be no epsilon transitions
                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                {
                    if (state.GetTransition(transitionIndex).IsEpsilon)
                    {
                        return false;
                    }
                }
                
                // Element distributions should not intersect
                for (int transitionIndex1 = 0; transitionIndex1 < state.TransitionCount; ++transitionIndex1)
                {
                    var transition1 = state.GetTransition(transitionIndex1);
                    for (int transitionIndex2 = transitionIndex1 + 1; transitionIndex2 < state.TransitionCount; ++transitionIndex2)
                    {
                        var transition2 = state.GetTransition(transitionIndex2);
                        double logProductNormalizer = transition1.ElementDistribution.Value.GetLogAverageOf(transition2.ElementDistribution.Value);
                        if (!double.IsNegativeInfinity(logProductNormalizer))
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }
        
        /// <summary>
        /// Attempts to determinize the automaton,
        /// i.e. modify it such that for every state and every element there is at most one transition that allows for that element,
        /// and there are no epsilon transitions.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the determinization attempt was successful and the automaton is now deterministic,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>See <a href="http://www.cs.nyu.edu/~mohri/pub/hwa.pdf"/> for algorithm details.</remarks>
        public bool TryDeterminize()
        {
            // We'd like to break if the determinized automaton is much larger than the original one,
            // or the original automaton is not determinizable at all.
            int maxStatesBeforeStop = Math.Min(this.States.Count * 3, MaxStateCount);
            return this.TryDeterminize(maxStatesBeforeStop);
        }

        /// <summary>
        /// Attempts to determinize the automaton,
        /// i.e. modify it such that for every state and every element there is at most one transition that allows for that element,
        /// and there are no epsilon transitions.
        /// </summary>
        /// <param name="maxStatesBeforeStop">
        /// The maximum number of states the resulting automaton can have. If the number of states exceeds the value
        /// of this parameter during determinization, the process is aborted.
        /// </param>
        /// <returns>
        /// <see langword="true"/> if the determinization attempt was successful and the automaton is now deterministic,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>See <a href="http://www.cs.nyu.edu/~mohri/pub/hwa.pdf"/> for algorithm details.</remarks>
        public bool TryDeterminize(int maxStatesBeforeStop)
        {
            Argument.CheckIfInRange(
                maxStatesBeforeStop > 0 && maxStatesBeforeStop <= MaxStateCount,
                "maxStatesBeforeStop",
                "The maximum number of states must be positive and not greater than the maximum number of states allowed in an automaton.");

            this.MakeEpsilonFree(); // Deterministic automata cannot have epsilon-transitions
            ////using (var writer = new System.IO.StreamWriter(@"\GraphViz\graphviz-2.38\release\bin\epsfree.txt"))
            ////{
            ////    writer.WriteLine(this.ToString(AutomatonFormats.GraphViz));
            ////}

            if (this.UsesGroups())
            {
                // Determinization will result in lost of group information, which we cannot allow
                return false;
            }

            // Weighted state set is a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton..
            // Such pairs correspond to states of the resulting automaton.
            var weightedStateSetQueue = new Queue<Determinization.WeightedStateSet>();
            var weightedStateSetToNewState = new Dictionary<Determinization.WeightedStateSet, State>();
            var result = Zero();

            var startWeightedStateSet = new Determinization.WeightedStateSet { { this.Start.Index, Weight.One } };
            weightedStateSetQueue.Enqueue(startWeightedStateSet);
            weightedStateSetToNewState.Add(startWeightedStateSet, result.Start);
            result.Start.SetEndWeight(this.Start.EndWeight);

            while (weightedStateSetQueue.Count > 0)
            {
                // Take one unprocessed state of the resulting automaton
                Determinization.WeightedStateSet currentWeightedStateSet = weightedStateSetQueue.Dequeue();
                var currentState = weightedStateSetToNewState[currentWeightedStateSet];

                // Find out what transitions we should add for this state
                IEnumerable<Tuple<TElementDistribution, Weight, Determinization.WeightedStateSet>> outgoingTransitionInfos =
                    this.GetOutgoingTransitionsForDeterminization(currentWeightedStateSet);

                // For each transition to add
                foreach (Tuple<TElementDistribution, Weight, Determinization.WeightedStateSet> outgoingTransitionInfo in outgoingTransitionInfos)
                {
                    TElementDistribution elementDistribution = outgoingTransitionInfo.Item1;
                    Weight weight = outgoingTransitionInfo.Item2;
                    Determinization.WeightedStateSet destWeightedStateSet = outgoingTransitionInfo.Item3;

                    State destinationState;
                    if (!weightedStateSetToNewState.TryGetValue(destWeightedStateSet, out destinationState))
                    {
                        if (result.States.Count == maxStatesBeforeStop)
                        {
                            // Too many states, determinization attempt failed
                            return false;
                        }

                        // Add new state to the result
                        destinationState = result.AddState();
                        weightedStateSetToNewState.Add(destWeightedStateSet, destinationState);
                        weightedStateSetQueue.Enqueue(destWeightedStateSet);

                        // Compute its ending weight
                        destinationState.SetEndWeight(Weight.Zero);
                        foreach (KeyValuePair<int, Weight> stateIdWithWeight in destWeightedStateSet)
                        {
                            destinationState.SetEndWeight(Weight.Sum(
                                destinationState.EndWeight,
                                Weight.Product(stateIdWithWeight.Value, this.States[stateIdWithWeight.Key].EndWeight)));
                        }
                    }

                    // Add transition to the destination state
                    currentState.AddTransition(elementDistribution, weight, destinationState);
                }
            }

            ////using (var writer = new System.IO.StreamWriter(@"\GraphViz\graphviz-2.38\release\bin\detpremerge.txt"))
            ////{
            ////    writer.WriteLine(result.ToString(AutomatonFormats.GraphViz));
            ////}

            result.MergeParallelTransitions(); // Determinization produces a separate transition for each segment
            result.PruneTransitionsWithLogWeightLessThan = this.PruneTransitionsWithLogWeightLessThan;
            result.LogValueOverride = this.LogValueOverride;
            // Determinization was successful, we can replace the current automaton with its deterministic version
            this.SwapWith(result);
            return true;
        }

        /// <summary>
        /// Attempts to simplify the automaton using <see cref="Simplify"/> if the number of states
        /// exceeds <see cref="MaxStateCountBeforeSimplification"/>.
        /// </summary>
        public void SimplifyIfNeeded()
        {
            if (this.States.Count > MaxStateCountBeforeSimplification || this.PruneTransitionsWithLogWeightLessThan != null)
            {
                ////Console.WriteLine(this.ToString(AutomatonFormats.GraphViz));
                this.Simplify();
                ////Console.WriteLine(this.ToString(AutomatonFormats.GraphViz));
            }
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
        public void Simplify()
        {
            this.MergeParallelTransitions();

            if (this.HasNonTrivialLoops())
            {
                return; // TODO: make this stuff work with non-trivial loops
            }

            ArrayDictionary<bool> stateLabels = this.LabelStatesForSimplification();

            TThis result = this.CopyNonSimplifiable(stateLabels);
            int firstNonCopiedStateIndex = result.States.Count;

            IEnumerable<Simplification.WeightedSequence> sequenceToLogWeight = this.BuildAcceptedSequenceList(stateLabels);

            // Before we rebuild the tree part, we prune out the low probability sequences
            if (this.PruneTransitionsWithLogWeightLessThan != null)
            {
                double logNorm = this.GetLogNormalizer();
                if (!double.IsInfinity(logNorm))
                {
                    sequenceToLogWeight = sequenceToLogWeight.Where(
                        s => s.Weight.LogValue - logNorm >= this.PruneTransitionsWithLogWeightLessThan.Value).ToList();
                }
            }

            foreach (Simplification.WeightedSequence weightedSequence in sequenceToLogWeight)
            {
                result.AddGeneralizedSequence(firstNonCopiedStateIndex, weightedSequence.Sequence, weightedSequence.Weight);
            }

            this.SwapWith(result);
        }
        
        /// <summary>
        /// Optimizes the automaton by removing all states unreachable from the end state.
        /// </summary>
        public void RemoveDeadStates()
        {
            if (this.IsCanonicZero())
            {
                return;
            }
            
            bool[] isEndStateReachable = this.ComputeEndStateReachability();
            RemoveStates(isEndStateReachable, MaxDeadStateCount);
        }

        /// <summary>
        /// Optimizes the automaton by removing all states unreachable from the start state.
        /// </summary>
        public void RemoveOrphanStates()
        {
            if (this.IsCanonicZero())
            {
                return;
            }

            bool[] isStateReachable = this.ComputeStartStateReachability();
            RemoveStates(isStateReachable, 0);
        }

        /// <summary>
        /// Removes a set of states from the automaton where the set is defined by
        /// the indices of the false elements in the supplied bool array.
        /// </summary>
        /// <param name="statesToKeep">The bool array specifying states to keep</param>
        /// <param name="minStatesToActuallyRemove">If the number of stats to remove is less than this value, the removal will not be done.</param>
        private void RemoveStates(bool[] statesToKeep, int minStatesToActuallyRemove)
        {
            int[] oldToNewStateIdMapping = new int[this.States.Count];
            int newStateId = 0;
            int deadStateCount = 0;
            for (int stateId = 0; stateId < this.States.Count; ++stateId)
            {
                if (statesToKeep[stateId])
                {
                    oldToNewStateIdMapping[stateId] = newStateId++;
                }
                else
                {
                    oldToNewStateIdMapping[stateId] = -1;
                    ++deadStateCount;
                }
            }

            if (oldToNewStateIdMapping[this.Start.Index] == -1)
            {
                // Cannot reach any end state from the start state => the automaton is zero everywhere
                this.SetToZero();
                return;
            }

            if (deadStateCount <= minStatesToActuallyRemove)
            {
                // Not enough dead states => no additional work needs to be done
                return;
            }

            TThis funcWithoutStates = Zero();
            funcWithoutStates.AddStates(newStateId - 1);
            funcWithoutStates.Start = funcWithoutStates.States[oldToNewStateIdMapping[this.Start.Index]];
            funcWithoutStates.LogValueOverride = this.LogValueOverride;
            funcWithoutStates.PruneTransitionsWithLogWeightLessThan = this.PruneTransitionsWithLogWeightLessThan;
            for (int i = 0; i < this.States.Count; ++i)
            {
                if (oldToNewStateIdMapping[i] == -1)
                {
                    continue;
                }

                State oldState = this.States[i];
                State newState = funcWithoutStates.States[oldToNewStateIdMapping[i]];
                newState.SetEndWeight(oldState.EndWeight);
                for (int transitionIndex = 0; transitionIndex < oldState.TransitionCount; ++transitionIndex)
                {
                    Transition transition = oldState.GetTransition(transitionIndex);
                    int newDestStateId = oldToNewStateIdMapping[transition.DestinationStateIndex];
                    if (newDestStateId != -1)
                    {
                        State newDestState = funcWithoutStates.States[newDestStateId];
                        newState.AddTransition(transition.ElementDistribution, transition.Weight, newDestState, transition.Group);
                    }
                }
            }

            this.SwapWith(funcWithoutStates);
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
            foreach (var state in this.States)
            {
                for (int i = state.TransitionCount-1; i >=0; i--)
                {
                    if (state.GetTransition(i).Weight.LogValue < logWeightThreshold)
                    {
                        state.RemoveTransition(i);
                    }
                }
            }
            RemoveOrphanStates();
        }
        
        /// <summary>
        /// Overridden in the derived classes to compute a set of outgoing transitions
        /// from a given state of the determinization result.
        /// </summary>
        /// <param name="sourceState">The source state of the determinized automaton represented as 
        /// a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton.</param>
        /// <returns>
        /// A collection of (element distribution, weight, weighted state set) triples corresponding to outgoing transitions from <paramref name="sourceState"/>.
        /// The first two elements of a tuple define the element distribution and the weight of a transition.
        /// The third element defines the outgoing state.
        /// </returns>
        protected abstract IEnumerable<Tuple<TElementDistribution, Weight, Determinization.WeightedStateSet>> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState);

        /// <summary>
        /// Labels each state with a value indicating whether the automaton having that state as the start state is a
        /// generalized tree (i.e. a tree with self-loops), which is also unreachable from previously traversed states.
        /// </summary>
        /// <returns>A dictionary mapping state indices to the computed labels.</returns>
        private ArrayDictionary<bool> LabelStatesForSimplification()
        {
            var result = new ArrayDictionary<bool>();
            this.DoLabelStatesForSimplification(this.Start, result);
            return result;
        }

        /// <summary>
        /// Recursively labels each state with a value indicating whether the automaton having that state as the start state
        /// is a generalized tree (i.e. a tree with self-loops), which is also unreachable from previously traversed states.
        /// </summary>
        /// <param name="currentState">The currently traversed state.</param>
        /// <param name="stateLabels">A dictionary mapping state indices to the computed labels.</param>
        /// <returns>
        /// <see langword="true"/> if the automaton having <paramref name="currentState"/> having that state as the start state
        /// is a generalized tree and it was the first visit to it, <see langword="false"/> otherwise.
        /// </returns>
        private bool DoLabelStatesForSimplification(State currentState, ArrayDictionary<bool> stateLabels)
        {
            if (stateLabels.ContainsKey(currentState.Index))
            {
                // This is not the first visit to the state
                return false;
            }

            stateLabels.Add(currentState.Index, true);

            bool isGeneralizedTree = true;
            for (int i = 0; i < currentState.TransitionCount; ++i)
            {
                Transition transition = currentState.GetTransition(i);

                // Self-loops are allowed
                if (transition.DestinationStateIndex != currentState.Index)
                {
                    isGeneralizedTree &= this.DoLabelStatesForSimplification(this.States[transition.DestinationStateIndex], stateLabels);
                }
            }

            // It was the first visit to the state
            stateLabels[currentState.Index] = isGeneralizedTree;
            return isGeneralizedTree;
        }

        /// <summary>
        /// Merges outgoing transitions with the same destination state.
        /// </summary>
        private void MergeParallelTransitions()
        {
            for (int stateIndex = 0; stateIndex < this.States.Count; ++stateIndex)
            {
                State state = this.States[stateIndex];
                for (int transitionIndex1 = 0; transitionIndex1 < state.TransitionCount; ++transitionIndex1)
                {
                    Transition transition1 = state.GetTransition(transitionIndex1);
                    for (int transitionIndex2 = transitionIndex1 + 1; transitionIndex2 < state.TransitionCount; ++transitionIndex2)
                    {
                        Transition transition2 = state.GetTransition(transitionIndex2);
                        if (transition1.DestinationStateIndex == transition2.DestinationStateIndex && transition1.Group == transition2.Group)
                        {
                            bool removeTransition2 = false;
                            if (transition1.IsEpsilon && transition2.IsEpsilon)
                            {
                                transition1.Weight = Weight.Sum(transition1.Weight, transition2.Weight);
                                state.SetTransition(transitionIndex1, transition1);
                                removeTransition2 = true;
                            }
                            else if (!transition1.IsEpsilon && !transition2.IsEpsilon)
                            {
                                var newElementDistribution = new TElementDistribution();
                                if (double.IsInfinity(transition1.Weight.Value) && double.IsInfinity(transition2.Weight.Value))
                                {
                                    newElementDistribution.SetToSum(1.0, transition1.ElementDistribution.Value, 1.0, transition2.ElementDistribution.Value);
                                }
                                else
                                {
                                    newElementDistribution.SetToSum(transition1.Weight.Value, transition1.ElementDistribution.Value, transition2.Weight.Value, transition2.ElementDistribution.Value);
                                }

                                transition1.ElementDistribution = newElementDistribution;
                                transition1.Weight = Weight.Sum(transition1.Weight, transition2.Weight);
                                state.SetTransition(transitionIndex1, transition1);
                                removeTransition2 = true;
                            }

                            if (removeTransition2)
                            {
                                state.RemoveTransition(transitionIndex2);
                                --transitionIndex2;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Creates a copy of the non-simplifiable part of the automaton (states labeled with
        /// <see langword="false"/> by <see cref="LabelStatesForSimplification"/> and their children).
        /// </summary>
        /// <param name="stateLabels">The state labels obtained from <see cref="LabelStatesForSimplification"/>.</param>
        /// <returns>The copied part of the automaton.</returns>
        private TThis CopyNonSimplifiable(ArrayDictionary<bool> stateLabels)
        {
            TThis result = Zero();
            if (!stateLabels[this.Start.Index])
            {
                var copiedStateCache = new ArrayDictionary<State>();
                result.Start = result.DoCopyNonSimplifiable(this.Start, stateLabels, true, copiedStateCache);
            }

            return result;
        }

        /// <summary>
        /// Recursively creates a copy of the non-simplifiable part of the automaton
        /// (states labeled with <see langword="false"/> by <see cref="LabelStatesForSimplification"/>).
        /// </summary>
        /// <param name="stateToCopy">The currently traversed state that needs to be copied.</param>
        /// <param name="stateLabels">The state labels obtained from <see cref="LabelStatesForSimplification"/>.</param>
        /// <param name="lookAtLabels">Whether or not labels should be ignored because one of the ancestors was labeled with <see langword="false"/>.</param>
        /// <param name="copiedStateCache">Cache of the state copies to avoid creating redundant states when traversing diamond-like structures.</param>
        /// <returns>The copied part of the automaton.</returns>
        private State DoCopyNonSimplifiable(
            State stateToCopy, ArrayDictionary<bool> stateLabels, bool lookAtLabels, ArrayDictionary<State> copiedStateCache)
        {
            Debug.Assert(!lookAtLabels || !stateLabels[stateToCopy.Index], "States that are not supposed to be copied should not be visited.");

            State copiedState;
            if (copiedStateCache.TryGetValue(stateToCopy.Index, out copiedState))
            {
                return copiedState;
            }

            copiedState = this.AddState();
            copiedState.SetEndWeight(stateToCopy.EndWeight);
            copiedStateCache.Add(stateToCopy.Index, copiedState);

            for (int i = 0; i < stateToCopy.TransitionCount; ++i)
            {
                Transition transitionToCopy = stateToCopy.GetTransition(i);
                State destStateToCopy = stateToCopy.Owner.States[transitionToCopy.DestinationStateIndex];
                if (!lookAtLabels || !stateLabels[destStateToCopy.Index])
                {
                    State copiedDestState = this.DoCopyNonSimplifiable(destStateToCopy, stateLabels, false, copiedStateCache);
                    copiedState.AddTransition(transitionToCopy.ElementDistribution, transitionToCopy.Weight, copiedDestState, transitionToCopy.Group);
                }
            }

            return copiedState;
        }

        /// <summary>
        /// Builds a complete list of generalized sequences accepted by the simplifiable part of the automaton.
        /// </summary>
        /// <param name="stateLabels">The state labels obtained from <see cref="LabelStatesForSimplification"/>.</param>
        /// <returns>The list of generalized sequences accepted by the simplifiable part of the automaton.</returns>
        private List<Simplification.WeightedSequence> BuildAcceptedSequenceList(ArrayDictionary<bool> stateLabels)
        {
            var sequenceToWeight = new List<Simplification.WeightedSequence>();
            this.DoBuildAcceptedSequenceList(this.Start, stateLabels, sequenceToWeight, new List<Simplification.GeneralizedElement>(), Weight.One);
            return sequenceToWeight;
        }


        private class StackItem
        {
        }

        private class ElementItem : StackItem
        {
            public readonly Simplification.GeneralizedElement? Element;

            public ElementItem(Simplification.GeneralizedElement? element)
            {
                this.Element = element;
            }
            public override string ToString()
            {
                return Element.ToString();
            }
        }

        private class StateWeight : StackItem
        {
            public StateWeight(State state, Weight weight)
            {
                this.State = state;
                this.Weight = weight;
            }

            public readonly State State;
            public readonly Weight Weight;

            public override string ToString()
            {
                return $"State: {State}, Weight: {Weight}";
            }
        }

        /// <summary>
        /// Recursively builds a complete list of generalized sequences accepted by the simplifiable part of the automaton.
        /// </summary>
        /// <param name="state">The currently traversed state.</param>
        /// <param name="stateLabels">The state labels obtained from <see cref="LabelStatesForSimplification"/>.</param>
        /// <param name="weightedSequences">The sequence list being built.</param>
        /// <param name="currentSequenceElements">The list of elements of the sequence currently being built.</param>
        /// <param name="currentWeight">The weight of the sequence currently being built.</param>
        private void DoBuildAcceptedSequenceList(
            State state,
            ArrayDictionary<bool> stateLabels,
            List<Simplification.WeightedSequence> weightedSequences,
            List<Simplification.GeneralizedElement> currentSequenceElements,
            Weight currentWeight)
        {
            var stack = new Stack<StackItem>();
            stack.Push(new StateWeight(state, currentWeight));

            while (stack.Count > 0)
            {
                var stackItem = stack.Pop();
                var elementItem = stackItem as ElementItem;

                if (elementItem != null)
                {
                    if (elementItem.Element != null)
                        currentSequenceElements.Add(elementItem.Element.Value);
                    else
                        currentSequenceElements.RemoveAt(currentSequenceElements.Count - 1);
                    continue;
                }

                var stateAndWeight = stackItem as StateWeight;

                state = stateAndWeight.State;
                currentWeight = stateAndWeight.Weight;

                // Find a non-epsilon self-loop if there is one
                var selfLoopIndex = -1;
                for (var i = 0; i < state.TransitionCount; ++i)
                {
                    var transition = state.GetTransition(i);
                    if (transition.DestinationStateIndex != state.Index) continue;
                    if (selfLoopIndex == -1)
                    {
                        selfLoopIndex = i;
                    }
                    else
                    {
                        Debug.Fail("Multiple self-loops should have been merged by MergeParallelTransitions()");
                    }
                }

                // Push the found self-loop to the end of the current sequence
                if (selfLoopIndex != -1)
                {
                    var transition = state.GetTransition(selfLoopIndex);
                    currentSequenceElements.Add(new Simplification.GeneralizedElement(
                        transition.ElementDistribution, transition.Group, transition.Weight));
                    stack.Push(new ElementItem(null));
                }

                // Can this state produce a sequence?
                if (state.CanEnd && stateLabels[state.Index])
                {
                    var sequence = new Simplification.GeneralizedSequence(currentSequenceElements);
                        // TODO: use immutable data structure instead of copying sequences
                    weightedSequences.Add(new Simplification.WeightedSequence(sequence, Weight.Product(currentWeight, state.EndWeight)));
                }

                // Traverse the outgoing transitions
                for (var i = 0; i < state.TransitionCount; ++i)
                {
                    var transition = state.GetTransition(i);

                    // Skip self-loops & disallowed states
                    if (transition.DestinationStateIndex == state.Index ||
                        !stateLabels[transition.DestinationStateIndex])
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
                            States[transition.DestinationStateIndex],
                            Weight.Product(currentWeight, transition.Weight)));

                    if (!transition.IsEpsilon)
                    {
                        stack.Push(
                            new ElementItem(new Simplification.GeneralizedElement(transition.ElementDistribution,
                                transition.Group, null)));
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
            bool isFreshStartState = this.IsCanonicZero();
            if (this.DoAddGeneralizedSequence(this.Start, isFreshStartState, false, firstAllowedStateIndex, 0, sequence, weight))
            {
                return;
            }

            // Branch the start state
            State oldStart = this.Start;
            this.Start = this.AddState();
            State otherBranch = this.AddState();
            this.Start.AddEpsilonTransition(Weight.One, oldStart);
            this.Start.AddEpsilonTransition(Weight.One, otherBranch);

            // This should always work
            bool success = this.DoAddGeneralizedSequence(otherBranch, true, false, firstAllowedStateIndex, 0, sequence, weight);
            Debug.Assert(success, "This call must always succeed.");
        }

        /// <summary>
        /// Recursively increases the value of this automaton on <paramref name="sequence"/> by <paramref name="weight"/>.
        /// </summary>
        /// <param name="state">The currently traversed state.</param>
        /// <param name="isNewState">Indicates whether <paramref name="state"/> was just created.</param>
        /// <param name="selfLoopAlreadyMatched">Indicates whether self-loop on <paramref name="state"/> was just matched.</param>
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
            State state,
            bool isNewState,
            bool selfLoopAlreadyMatched,
            int firstAllowedStateIndex,
            int currentSequencePos,
            Simplification.GeneralizedSequence sequence,
            Weight weight)
        {
            bool success;

            if (currentSequencePos == sequence.Count)
            {
                if (!selfLoopAlreadyMatched)
                {
                    // We can't finish in a state with a self-loop
                    for (int i = 0; i < state.TransitionCount; ++i)
                    {
                        Transition transition = state.GetTransition(i);
                        if (transition.DestinationStateIndex == state.Index)
                        {
                            return false;
                        }
                    }
                }

                state.SetEndWeight(Weight.Sum(state.EndWeight, weight));
                return true;
            }

            Simplification.GeneralizedElement element = sequence[currentSequencePos];

            // Treat self-loops elements separately
            if (element.LoopWeight.HasValue)
            {
                if (selfLoopAlreadyMatched)
                {
                    // Previous element was also a self-loop, we should try to find an espilon transition
                    for (int i = 0; i < state.TransitionCount; ++i)
                    {
                        Transition transition = state.GetTransition(i);
                        if (transition.DestinationStateIndex != state.Index && transition.IsEpsilon && transition.DestinationStateIndex >= firstAllowedStateIndex)
                        {
                            if (this.DoAddGeneralizedSequence(
                                this.States[transition.DestinationStateIndex],
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
                    State destination = state.AddEpsilonTransition(Weight.One);
                    success = this.DoAddGeneralizedSequence(destination, true, false, firstAllowedStateIndex, currentSequencePos, sequence, weight);
                    Debug.Assert(success, "This call must always succeed.");
                    return true;
                }

                // Find a matching self-loop
                for (int i = 0; i < state.TransitionCount; ++i)
                {
                    Transition transition = state.GetTransition(i);

                    if (transition.IsEpsilon && transition.DestinationStateIndex != state.Index && transition.DestinationStateIndex >= firstAllowedStateIndex)
                    {
                        // Try this epsilon transition
                        if (this.DoAddGeneralizedSequence(
                            this.States[transition.DestinationStateIndex], false, false, firstAllowedStateIndex, currentSequencePos, sequence, weight))
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
                            success = this.DoAddGeneralizedSequence(state, false, true, firstAllowedStateIndex, currentSequencePos + 1, sequence, weight);
                            Debug.Assert(success, "This call must always succeed.");
                            return true;
                        }

                        // State also has a self-loop, but the two doesn't match
                        return false;
                    }
                }

                if (!isNewState)
                {
                    // Can't add self-loop to an existing state, it will change the language accepted by the state
                    return false;
                }

                // Add a new self-loop
                state.AddTransition(element.ElementDistribution, element.LoopWeight.Value, state, element.Group);
                success = this.DoAddGeneralizedSequence(state, false, true, firstAllowedStateIndex, currentSequencePos + 1, sequence, weight);
                Debug.Assert(success, "This call must always succeed.");
                return true;
            }

            // Try to find a transition for the element
            for (int i = 0; i < state.TransitionCount; ++i)
            {
                Transition transition = state.GetTransition(i);

                if (transition.IsEpsilon && transition.DestinationStateIndex != state.Index && transition.DestinationStateIndex >= firstAllowedStateIndex)
                {
                    // Try this epsilon transition
                    if (this.DoAddGeneralizedSequence(
                        this.States[transition.DestinationStateIndex], false, false, firstAllowedStateIndex, currentSequencePos, sequence, weight))
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
                    this.States[transition.DestinationStateIndex],
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
            State newChild = state.AddTransition(element.ElementDistribution, Weight.One, default(State), element.Group);
            success = this.DoAddGeneralizedSequence(newChild, true, false, firstAllowedStateIndex, currentSequencePos + 1, sequence, weight);
            Debug.Assert(success, "This call must always succeed.");
            return true;
        }

        /// <summary>
        /// Groups together helper classes used for automata determinization.
        /// </summary>
        protected static class Determinization
        {
            /// <summary>
            /// Represents a state of the resulting automaton in the power set construction.
            /// It is essentially a set of (stateId, weight) pairs of the source automaton, where each state id is unique.
            /// Supports a quick lookup of the weight by state id.
            /// </summary>
            public class WeightedStateSet : IEnumerable<KeyValuePair<int, Weight>>
            {
                /// <summary>
                /// A mapping from state ids to weights.
                /// </summary>
                private readonly Dictionary<int, Weight> stateIdToWeight;

                /// <summary>
                /// Initializes a new instance of the <see cref="WeightedStateSet"/> class.
                /// </summary>
                public WeightedStateSet()
                {
                    this.stateIdToWeight = new Dictionary<int, Weight>();
                }

                /// <summary>
                /// Initializes a new instance of the <see cref="WeightedStateSet"/> class.
                /// </summary>
                /// <param name="stateIdToWeight">A collection of (stateId, weight) pairs.
                /// </param>
                public WeightedStateSet(IEnumerable<KeyValuePair<int, Weight>> stateIdToWeight)
                {
                    this.stateIdToWeight = stateIdToWeight.ToDictionary(kv => kv.Key, kv => kv.Value);
                }

                /// <summary>
                /// Gets or sets the weight for a given state id.
                /// </summary>
                /// <param name="stateId">The state id.</param>
                /// <returns>The weight.</returns>
                public Weight this[int stateId]
                {
                    get { return this.stateIdToWeight[stateId]; }
                    set { this.stateIdToWeight[stateId] = value; }
                }

                /// <summary>
                /// Adds a given state id and a weight to the set.
                /// </summary>
                /// <param name="stateId">The state id.</param>
                /// <param name="weight">The weight.</param>
                public void Add(int stateId, Weight weight)
                {
                    this.stateIdToWeight.Add(stateId, weight);
                }

                /// <summary>
                /// Attempts to retrieve the weight corresponding to a given state id from the set.
                /// </summary>
                /// <param name="stateId">The state id.</param>
                /// <param name="weight">When the method returns, contains the retrieved weight.</param>
                /// <returns>
                /// <see langword="true"/> if the given state id was present in the set,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public bool TryGetWeight(int stateId, out Weight weight)
                {
                    return this.stateIdToWeight.TryGetValue(stateId, out weight);
                }

                /// <summary>
                /// Checks whether the state with a given id is present in the set.
                /// </summary>
                /// <param name="stateId">The state id,</param>
                /// <returns>
                /// <see langword="true"/> if the given state id was present in the set,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public bool ContainsState(int stateId)
                {
                    return this.stateIdToWeight.ContainsKey(stateId);
                }

                /// <summary>
                /// Checks whether this object is equal to a given one.
                /// </summary>
                /// <param name="obj">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if the objects are equal,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public override bool Equals(object obj)
                {
                    if (obj == null || obj.GetType() != typeof(WeightedStateSet))
                    {
                        return false;
                    }

                    var other = (WeightedStateSet)obj;

                    if (this.stateIdToWeight.Count != other.stateIdToWeight.Count)
                    {
                        return false;
                    }

                    foreach (KeyValuePair<int, Weight> pair in this.stateIdToWeight)
                    {
                        // TODO: Should we allow for some tolerance? But what about hashing then?
                        Weight otherWeight;
                        if (!other.stateIdToWeight.TryGetValue(pair.Key, out otherWeight) || otherWeight != pair.Value)
                        {
                            return false;
                        }
                    }

                    return true;
                }

                /// <summary>
                /// Computes the hash code of this instance.
                /// </summary>
                /// <returns>The computed hash code.</returns>
                public override int GetHashCode()
                {
                    int result = 0;
                    foreach (KeyValuePair<int, Weight> pair in this.stateIdToWeight)
                    {
                        int pairHash = Hash.Start;
                        pairHash = Hash.Combine(pairHash, pair.Key.GetHashCode());
                        pairHash = Hash.Combine(pairHash, pair.Value.GetHashCode());

                        // Use commutative hashing combination because dictionaries are not ordered
                        result ^= pairHash;
                    }

                    return result;
                }

                /// <summary>
                /// Returns a string representation of the instance.
                /// </summary>
                /// <returns>A string representation of the instance.</returns>
                public override string ToString()
                {
                    StringBuilder builder = new StringBuilder();
                    foreach (var kvp in this.stateIdToWeight)
                    {
                        builder.AppendLine(kvp.ToString());
                    }

                    return builder.ToString();
                }

                #region IEnumerable implementation

                /// <summary>
                /// Gets the enumerator.
                /// </summary>
                /// <returns>
                /// The enumerator.
                /// </returns>
                public IEnumerator<KeyValuePair<int, Weight>> GetEnumerator()
                {
                    return this.stateIdToWeight.GetEnumerator();
                }

                /// <summary>
                /// Gets the enumerator.
                /// </summary>
                /// <returns>
                /// The enumerator.
                /// </returns>
                IEnumerator IEnumerable.GetEnumerator()
                {
                    return this.GetEnumerator();
                }

                #endregion
            }
        }

        /// <summary>
        /// Groups together helper classes used for automata simplification.
        /// </summary>
        private static class Simplification
        {
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
                public int Group { get; private set; }

                /// <summary>
                /// Gets the loop weight associated with the generalized element,
                /// <see langword="null"/> if the element does not represent a self-loop.
                /// </summary>
                public Weight? LoopWeight { get; private set; }

                /// <summary>
                /// Gets the string representation of the generalized element.
                /// </summary>
                /// <returns>The string representation of the generalized element.</returns>
                public override string ToString()
                {
                    string elementDistributionAsString = this.ElementDistribution.Value.IsPointMass ? this.ElementDistribution.Value.Point.ToString() : this.ElementDistribution.ToString();
                    string groupString = this.Group == 0 ? string.Empty : string.Format("#{0}", this.Group);
                    if (this.LoopWeight.HasValue)
                    {
                        return string.Format("{0}{1}*({2})", groupString, elementDistributionAsString, this.LoopWeight.Value);
                    }

                    return string.Format("{0}{1}", groupString, elementDistributionAsString);
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
                public int Count
                {
                    get { return this.elements.Count; }
                }

                /// <summary>
                /// Gets the sequence element with the specified index.
                /// </summary>
                /// <param name="index">The element index.</param>
                /// <returns>The element at the given index.</returns>
                public GeneralizedElement this[int index]
                {
                    get { return this.elements[index]; }
                }

                /// <summary>
                /// Gets the string representation of the sequence.
                /// </summary>
                /// <returns>The string representation of the sequence.</returns>
                public override string ToString()
                {
                    var stringBuilder = new StringBuilder();
                    foreach (GeneralizedElement element in this.elements)
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
                public override string ToString()
                {
                    return $"[{this.Sequence},{this.Weight}]";
                }

                /// <summary>
                /// Checks if this object is equal to <paramref name="obj"/>.
                /// </summary>
                /// <param name="obj">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if this object is equal to <paramref name="obj"/>,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public override bool Equals(object obj)
                {
                    if (obj is WeightedSequence weightedSequence)
                    {
                        return object.Equals(this.Sequence, weightedSequence.Sequence) && object.Equals(this.Weight, weightedSequence.Weight);
                    }

                    return false;
                }

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
