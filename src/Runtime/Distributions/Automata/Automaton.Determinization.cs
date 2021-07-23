﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains classes and methods for automata simplification.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <inheritdoc cref="TryDeterminize(out TThis)" path="/summary"/>
        /// <returns>Result automaton. Determinized automaton, if the operation was successful, current automaton or
        /// an equvalent automaton without epsilon transitions otherwise.</returns>
        public TThis TryDeterminize()
        {
            TryDeterminize(out TThis result);
            return result;
        }

        /// <summary>
        /// Attempts to determinize the automaton,
        /// i.e. modify it such that for every state and every element there is at most one transition that allows for that element,
        /// and there are no epsilon transitions.
        /// </summary>
        /// <param name="result">Result automaton. Determinized automaton, if the operation was successful, current automaton or
        /// an equvalent automaton without epsilon transitions otherwise.</param>
        /// <returns>
        /// <see langword="true"/> if the determinization attempt was successful and the automaton is now deterministic,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>See <a href="http://www.cs.nyu.edu/~mohri/pub/hwa.pdf"/> for algorithm details.</remarks>
        public bool TryDeterminize(out TThis result)
        {
            if (this.Data.IsDeterminized != null)
            {
                result = (TThis)this;
                return this.Data.IsDeterminized == true;
            }

            int maxStatesBeforeStop = Math.Min(this.States.Count * 3, MaxStateCount);

            var epsilonClosure = GetEpsilonClosure(); // Deterministic automata cannot have epsilon-transitions

            if (this.UsesGroups)
            {
                // Determinization will result in lost of group information, which we cannot allow
                this.Data = this.Data.With(isDeterminized: false);
                result = WithData(epsilonClosure.Data.With(isDeterminized: false));
                return false;
            }

            var builder = new Builder();
            builder.Start.SetEndWeight(epsilonClosure.Start.EndWeight);

            var weightedStateSetStack = new Stack<(bool enter, Determinization.WeightedStateSet set)>();
            var enqueuedWeightedStateSetStack = new Stack<(bool enter, Determinization.WeightedStateSet set)>();
            var weightedStateSetToNewState = new Dictionary<Determinization.WeightedStateSet, int>();
            // This hash set is used to track sets currently in path from root. If we've found a set of states
            // that we have already seen during current path from root, but weights are different, that means
            // we've found a non-converging loop - infinite number of weighed sets will be generated if
            // we continue traversal and determinization will fail. For performance reasons we want to fail
            // fast if such loop is found.
            var stateSetsInPath = new Dictionary<Determinization.WeightedStateSet, Determinization.WeightedStateSet>(
                Determinization.WeightedStateSetOnlyStateComparer.Instance);
            
            var startWeightedStateSet = new Determinization.WeightedStateSet(epsilonClosure.Start.Index);
            weightedStateSetStack.Push((true, startWeightedStateSet));
            weightedStateSetToNewState.Add(startWeightedStateSet, builder.StartStateIndex);

            while (weightedStateSetStack.Count > 0)
            {
                // Take one unprocessed state of the resulting automaton
                var (enter, currentWeightedStateSet) = weightedStateSetStack.Pop();

                if (enter)
                {
                    if (currentWeightedStateSet.Count > 1)
                    {
                        // Only sets with more than 1 state can lead to infinite loops with different weights.
                        // Because if there's only 1 state, than it's weight is always Weight.One.
                        if (!stateSetsInPath.ContainsKey(currentWeightedStateSet))
                        {
                            stateSetsInPath.Add(currentWeightedStateSet, currentWeightedStateSet);
                        }

                        weightedStateSetStack.Push((false, currentWeightedStateSet));
                    }

                    if (!EnqueueOutgoingTransitions(currentWeightedStateSet))
                    {
                        this.Data = this.Data.With(isDeterminized: false);
                        result = WithData(epsilonClosure.Data.With(isDeterminized: false));
                        return false;
                    }
                }
                else
                {
                    stateSetsInPath.Remove(currentWeightedStateSet);
                }
            }

            var simplification = new Simplification(builder, epsilonClosure.PruneStatesWithLogEndWeightLessThan);
            simplification.MergeParallelTransitions(); // Determinization produces a separate transition for each segment

            result = WithData(builder.GetData(true));
            return true;

            bool EnqueueOutgoingTransitions(Determinization.WeightedStateSet currentWeightedStateSet)
            {
                var currentStateIndex = weightedStateSetToNewState[currentWeightedStateSet];
                var currentState = builder[currentStateIndex];

                // Common special-case: definitely deterministic transitions from single state.
                // In this case no complicated determinization procedure is needed.
                if (currentWeightedStateSet.Count == 1 &&
                    AllDestinationsAreSame(currentWeightedStateSet[0].Index))
                {
                    Debug.Assert(currentWeightedStateSet[0].Weight == Weight.One);

                    var sourceState = epsilonClosure.States[currentWeightedStateSet[0].Index];
                    foreach (var transition in sourceState.Transitions)
                    {
                        var destinationStates = new Determinization.WeightedStateSet(transition.DestinationStateIndex);
                        var outgoingTransitionInfo = new Determinization.OutgoingTransition(
                            transition.ElementDistribution.Value, transition.Weight, destinationStates);
                        if (!TryAddTransition(enqueuedWeightedStateSetStack, outgoingTransitionInfo, currentState))
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    // Find out what transitions we should add for this state
                    var outgoingTransitions =
                        epsilonClosure.GetOutgoingTransitionsForDeterminization(currentWeightedStateSet);
                    foreach (var outgoingTransition in outgoingTransitions)
                    {
                        if (!TryAddTransition(enqueuedWeightedStateSetStack, outgoingTransition, currentState))
                        {
                            return false;
                        }
                    }
                }

                while (enqueuedWeightedStateSetStack.Count > 0)
                {
                    weightedStateSetStack.Push(enqueuedWeightedStateSetStack.Pop());
                }

                return true;
            }

            // Checks that all transitions from state end up in the same destination. This is used
            // as a very fast "is deterministic" check, that doesn't care about distributions.
            // State can have deterministic transitions with different destinations. This case will be
            // handled by slow path.
            bool AllDestinationsAreSame(int stateIndex)
            {
                var transitions = epsilonClosure.States[stateIndex].Transitions;
                if (transitions.Count <= 1)
                {
                    return true;
                }

                var destination = transitions[0].DestinationStateIndex;
                for (var i = 1; i < transitions.Count; ++i)
                {
                    if (transitions[i].DestinationStateIndex != destination)
                    {
                        return false;
                    }
                }

                return true;
            }

            // Adds transition from currentState into state corresponding to weighted state set from
            // outgoingTransitionInfo. If that state does not exist yet it is created and is put into stack
            // for further processing. This function returns false if determinization has failed.
            // That can happen because of 2 ressons:
            // - Too many states were created and its not feasible to continue trying to determinize
            //   automaton further
            // - An infinite loop with not converging weights was found. It leads to infinite number of states.
            //   So determinization is aborted early.
            bool TryAddTransition(
                Stack<(bool enter, Determinization.WeightedStateSet set)> destinationStack,
                Determinization.OutgoingTransition transition,
                Builder.StateBuilder currentState)
            {
                var destinations = transition.Destinations;
                if (!weightedStateSetToNewState.TryGetValue(destinations, out var destinationStateIndex))
                {
                    if (builder.StatesCount == maxStatesBeforeStop)
                    {
                        // Too many states, determinization attempt failed
                        return false;
                    }

                    var visitedWeightedStateSet = default(Determinization.WeightedStateSet);
                    var sameSetVisited =
                        destinations.Count > 1 &&
                        stateSetsInPath.TryGetValue(destinations, out visitedWeightedStateSet);

                    if (sameSetVisited && !destinations.Equals(visitedWeightedStateSet))
                    {
                        // We arrived into the same state set as before, but with different weights.
                        // This is an infinite non-converging loop. Determinization has failed
                        return false;
                    }

                    // Add new state to the result
                    var destinationState = builder.AddState();
                    weightedStateSetToNewState.Add(destinations, destinationState.Index);
                    destinationStack.Push((true, destinations));

                    if (destinations.Count > 1 && !sameSetVisited)
                    {
                        destinationStack.Push((false, destinations));
                    }

                    // Compute its ending weight
                    destinationState.SetEndWeight(Weight.Zero);
                    for (var i = 0; i < destinations.Count; ++i)
                    {
                        var weightedState = destinations[i];
                        var addedWeight = weightedState.Weight * epsilonClosure.States[weightedState.Index].EndWeight;
                        destinationState.SetEndWeight(destinationState.EndWeight + addedWeight);
                    }

                    destinationStateIndex = destinationState.Index;
                }

                // Add transition to the destination state
                currentState.AddTransition(transition.ElementDistribution, transition.Weight, destinationStateIndex);
                return true;
            }
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
        protected abstract IEnumerable<Determinization.OutgoingTransition> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState);

        /// <summary>
        /// Groups together helper classes used for automata determinization.
        /// </summary>
        protected static class Determinization
        {
            public struct OutgoingTransition
            {
                public TElementDistribution ElementDistribution { get; }
                public Weight Weight { get; }
                public WeightedStateSet Destinations { get; }

                public OutgoingTransition(
                    TElementDistribution elementDistribution, Weight weight, WeightedStateSet destinations)
                {
                    this.ElementDistribution = elementDistribution;
                    this.Weight = weight;
                    this.Destinations = destinations;
                }
            }

            /// <summary>
            /// Represents
            /// </summary>
            public struct WeightedState : IComparable, IComparable<WeightedState>
            {
                /// <summary>
                /// Index of the state.
                /// </summary>
                public int Index { get; }

                /// <summary>
                /// High bits of the state weight. Only these bits are used when comparing weighted states
                /// for equality or when calculating hash code.
                /// </summary>
                /// <remarks>
                /// Using high bits for hash code allows to have "fuzzy" matching of WeightedStateSets.
                /// Sometimes there's more than one way to get to the same weighted state set in automaton,
                /// but due to using floating point numbers calculated weights are slightly off. Using
                /// only high 32 bits for comparison means that 20 bits of mantissa are used. Which means that
                /// difference between weights (in log space) is no more than ~~ 10e-6 which is a sufficiently
                /// good precision for all practical purposes.
                /// </remarks>
                public int WeightHighBits { get; }

                public Weight Weight { get; }

                public WeightedState(int index, Weight weight)
                {
                    this.Index = index;
                    this.WeightHighBits = (int)(BitConverter.DoubleToInt64Bits(weight.LogValue) >> 32);
                    this.Weight = weight;
                }

                public int CompareTo(object obj)
                {
                    return obj is WeightedState that
                        ? this.CompareTo(that)
                        : throw new InvalidOperationException(
                            "WeightedState can be compared only to another WeightedState");
                }

                public int CompareTo(WeightedState that) => Index.CompareTo(that.Index);

                public override int GetHashCode() => (Index ^ WeightHighBits).GetHashCode();

                public bool Equals(int index, Weight weight) =>
                    this.Index == index && this.Weight == weight;

                public override string ToString() => $"({this.Index}, {this.Weight})";
            }

            /// <summary>
            /// Represents a state of the resulting automaton in the power set construction.
            /// It is essentially a set of (stateId, weight) pairs of the source automaton, where each state id is unique.
            /// Supports a quick lookup of the weight by state id.
            /// </summary>
            public struct WeightedStateSet
            {
                /// <summary>
                /// A mapping from state ids to weights. This array is sorted by state Id.
                /// </summary>
                private readonly ReadOnlyArray<WeightedState> weightedStates;

                private readonly int singleStateIndex;

                public WeightedStateSet(int stateIndex)
                {
                    this.weightedStates = default(ReadOnlyArray<WeightedState>);
                    this.singleStateIndex = stateIndex;
                }

                public WeightedStateSet(ReadOnlyArray<WeightedState> weightedStates)
                {
                    Debug.Assert(weightedStates.Count > 0);
                    Debug.Assert(IsSorted(weightedStates));
                    if (weightedStates.Count == 1)
                    {
                        Debug.Assert(weightedStates[0].Weight == Weight.One);
                        this.weightedStates = default(ReadOnlyArray<WeightedState>);
                        this.singleStateIndex = weightedStates[0].Index;
                    }
                    else
                    {
                        this.weightedStates = weightedStates;
                        this.singleStateIndex = 0; // <- value doesn't matter, but silences the compiler
                    }
                }

                public int Count =>
                    (this.weightedStates == default(ReadOnlyArray<WeightedState>))
                        ? 1
                        : this.weightedStates.Count;

                public WeightedState this[int index] =>
                    (this.weightedStates == default(ReadOnlyArray<WeightedState>))
                        ? new WeightedState(this.singleStateIndex, Weight.One)
                        : this.weightedStates[index];

                /// <summary>
                /// Checks whether this object is equal to a given one.
                /// </summary>
                /// <param name="obj">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if the objects are equal,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public override bool Equals(object obj) => obj is WeightedStateSet that && this.Equals(that);

                /// <summary>
                /// Checks whether this object is equal to a given one.
                /// </summary>
                /// <param name="that">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if the objects are equal,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public bool Equals(WeightedStateSet that)
                {
                    if (this.Count != that.Count)
                    {
                        return false;
                    }

                    for (var i = 0; i < this.Count; ++i)
                    {
                        var state1 = this[i];
                        var state2 = that[i];
                        if (state1.Index != state2.Index || state1.WeightHighBits != state2.WeightHighBits)
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
                /// <remarks>Only state ids</remarks>
                public override int GetHashCode()
                {
                    var result = this[0].GetHashCode();
                    for (var i = 1; i < this.Count; ++i)
                    {
                        result = Hash.Combine(result, this[i].GetHashCode());
                    }

                    return result;
                }

                /// <summary>
                /// Returns a string representation of the instance.
                /// </summary>
                /// <returns>A string representation of the instance.</returns>
                public override string ToString() => string.Join(", ", weightedStates);

                /// <summary>
                /// Turns weighted state set into an array. This is convenient for writing LINQ queries
                /// in tests.
                /// </summary>
                public WeightedState[] ToArray()
                {
                    var result = new WeightedState[this.Count];
                    for (var i = 0; i < this.Count; ++i)
                    {
                        result[i] = this[i];
                    }

                    return result;
                }

                /// <summary>
                /// Checks weather states array is sorted in ascending order by Index.
                /// </summary>
                private static bool IsSorted(ReadOnlyArray<WeightedState> array)
                {
                    for (var i = 1; i < array.Count; ++i)
                    {
                        if (array[i].Index <= array[i - 1].Index)
                        {
                            return false;
                        }
                    }

                    return true;
                }
            }

            /// <summary>
            /// Builder for weighted sets.
            /// </summary>
            public struct WeightedStateSetBuilder
            {
                private List<WeightedState> weightedStates;

                public static WeightedStateSetBuilder Create() =>
                    new WeightedStateSetBuilder()
                    {
                        weightedStates = new List<WeightedState>(1),
                    };

                public void Add(int index, Weight weight) =>
                    this.weightedStates.Add(new WeightedState(index, weight));

                /// <summary>
                /// Removes state that was previously added into this builder
                /// </summary>
                public void Remove(int index, Weight weight)
                {
                    // This algorithm is linear, in theory we could store all states in HashSet<> and make
                    // this algorithm amortized O(1). In practice we still have O(n) algorithm when
                    // constructing the final WeightedStateSet, so it's fine to have O(n) algorithm in here
                    // even in the worst case, because runtime is anyway O(n).
                    // In average case (small number of states in set) linear search has way better
                    // constant costs.
                    var lastStateIndex = this.weightedStates.Count - 1;
                    var removedStateIndex = lastStateIndex;
                    while (!this.weightedStates[removedStateIndex].Equals(index, weight))
                    {
                        --removedStateIndex;
                    }

                    this.weightedStates.RemoveAt(removedStateIndex);
                }

                public int StatesCount() => this.weightedStates.Count;

                /// <summary>
                /// Builds a <see cref="WeightedStateSet"/> with normalized weights. Normalizer (max weight
                /// pre-normalization) is also returned.
                ///
                /// Internal state of builder is not mutated, it can be modified by Add()/Remove() calls
                /// to obtain new WeightedStateSets.
                /// </summary>
                public (WeightedStateSet, Weight) Get()
                {
                    Debug.Assert(this.weightedStates.Count > 0);

                    if (this.weightedStates.Count == 1)
                    {
                        // Happy path that requires no additional storage or sorting
                        var state = this.weightedStates[0];
                        return (new WeightedStateSet(state.Index), state.Weight);
                    }

                    var weightsClone = new List<WeightedState>(this.weightedStates);
                    weightsClone.Sort();
                    MergeRepeatedEntries(weightsClone);
                    var maxWeight = NormalizeWeights(weightsClone);

                    return (new WeightedStateSet(weightsClone.ToReadOnlyArray()), maxWeight);
                }

                private static void MergeRepeatedEntries(List<WeightedState> weightedStates)
                {
                    // Merge repeated entries for the same state
                    var dst = 0;
                    for (var i = 1; i < weightedStates.Count; ++i)
                    {
                        var srcState = weightedStates[i];
                        var dstState = weightedStates[dst];
                        if (srcState.Index == dstState.Index)
                        {
                            weightedStates[dst] =
                                new WeightedState(dstState.Index, dstState.Weight + srcState.Weight);
                        }
                        else
                        {
                            ++dst;
                            if (dst != i)
                            {
                                weightedStates[dst] = srcState;
                            }
                        }
                    }

                    // truncate excess
                    ++dst;
                    if (dst < weightedStates.Count)
                    {
                        weightedStates.RemoveRange(dst, weightedStates.Count - dst);
                    }
                }

                private static Weight NormalizeWeights(List<WeightedState> weightedStates)
                {
                    var maxWeight = weightedStates[0].Weight;
                    for (var i = 1; i < weightedStates.Count; ++i)
                    {
                        if (weightedStates[i].Weight > maxWeight)
                        {
                            maxWeight = weightedStates[i].Weight;
                        }
                    }

                    var normalizer = Weight.Inverse(maxWeight);

                    for (var i = 0; i < weightedStates.Count; ++i)
                    {
                        var state = weightedStates[i];
                        weightedStates[i] = new WeightedState(state.Index, state.Weight * normalizer);
                    }

                    return maxWeight;
                }
            }

            public class WeightedStateSetOnlyStateComparer : IEqualityComparer<WeightedStateSet>
            {
                public static readonly WeightedStateSetOnlyStateComparer Instance =
                    new WeightedStateSetOnlyStateComparer();

                public bool Equals(WeightedStateSet x, WeightedStateSet y)
                {
                    if (x.Count != y.Count)
                    {
                        return false;
                    }

                    for (var i = 0; i < x.Count; ++i)
                    {
                        if (x[i].Index != y[i].Index)
                        {
                            return false;
                        }
                    }

                    return true;
                }

                public int GetHashCode(WeightedStateSet set)
                {
                    var result = set[0].Index.GetHashCode();
                    for (var i = 1; i < set.Count; ++i)
                    {
                        result = Hash.Combine(result, set[i].Index.GetHashCode());
                    }

                    return result;
                }
            }
        }
    }
}
