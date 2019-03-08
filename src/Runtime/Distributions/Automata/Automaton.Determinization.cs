// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains classes and methods for automata simplification.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
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

            if (this.UsesGroups)
            {
                // Determinization will result in lost of group information, which we cannot allow
                return false;
            }

            // Weighted state set is a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton..
            // Such pairs correspond to states of the resulting automaton.
            var weightedStateSetQueue = new Queue<Determinization.WeightedStateSet>();
            var weightedStateSetToNewState = new Dictionary<Determinization.WeightedStateSet, int>();
            var builder = new Builder();

            var startWeightedStateSet = new Determinization.WeightedStateSet { { this.Start.Index, Weight.One } };
            weightedStateSetQueue.Enqueue(startWeightedStateSet);
            weightedStateSetToNewState.Add(startWeightedStateSet, builder.StartStateIndex);
            builder.Start.SetEndWeight(this.Start.EndWeight);

            while (weightedStateSetQueue.Count > 0)
            {
                // Take one unprocessed state of the resulting automaton
                Determinization.WeightedStateSet currentWeightedStateSet = weightedStateSetQueue.Dequeue();
                var currentStateIndex = weightedStateSetToNewState[currentWeightedStateSet];
                var currentState = builder[currentStateIndex];

                // Find out what transitions we should add for this state
                var outgoingTransitionInfos = this.GetOutgoingTransitionsForDeterminization(currentWeightedStateSet);

                // For each transition to add
                foreach ((TElementDistribution, Weight, Determinization.WeightedStateSet) outgoingTransitionInfo in outgoingTransitionInfos)
                {
                    TElementDistribution elementDistribution = outgoingTransitionInfo.Item1;
                    Weight weight = outgoingTransitionInfo.Item2;
                    Determinization.WeightedStateSet destWeightedStateSet = outgoingTransitionInfo.Item3;

                    int destinationStateIndex;
                    if (!weightedStateSetToNewState.TryGetValue(destWeightedStateSet, out destinationStateIndex))
                    {
                        if (builder.StatesCount == maxStatesBeforeStop)
                        {
                            // Too many states, determinization attempt failed
                            return false;
                        }

                        // Add new state to the result
                        var destinationState = builder.AddState();
                        weightedStateSetToNewState.Add(destWeightedStateSet, destinationState.Index);
                        weightedStateSetQueue.Enqueue(destWeightedStateSet);

                        // Compute its ending weight
                        destinationState.SetEndWeight(Weight.Zero);
                        foreach (KeyValuePair<int, Weight> stateIdWithWeight in destWeightedStateSet)
                        {
                            destinationState.SetEndWeight(Weight.Sum(
                                destinationState.EndWeight,
                                Weight.Product(stateIdWithWeight.Value, this.States[stateIdWithWeight.Key].EndWeight)));
                        }

                        destinationStateIndex = destinationState.Index;
                    }

                    // Add transition to the destination state
                    currentState.AddTransition(elementDistribution, weight, destinationStateIndex);
                }
            }

            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            simplification.MergeParallelTransitions(); // Determinization produces a separate transition for each segment

            var result = builder.GetAutomaton();
            result.PruneStatesWithLogEndWeightLessThan = this.PruneStatesWithLogEndWeightLessThan;
            result.LogValueOverride = this.LogValueOverride;
            this.SwapWith(result);

            return true;
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
        protected abstract List<(TElementDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState);

        
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
                public WeightedStateSet() =>
                    this.stateIdToWeight = new Dictionary<int, Weight>();
                
                /// <summary>
                /// Initializes a new instance of the <see cref="WeightedStateSet"/> class.
                /// </summary>
                /// <param name="stateIdToWeight">A collection of (stateId, weight) pairs.
                /// </param>
                public WeightedStateSet(IEnumerable<KeyValuePair<int, Weight>> stateIdToWeight) =>
                    this.stateIdToWeight = stateIdToWeight.ToDictionary(kv => kv.Key, kv => kv.Value);

                /// <summary>
                /// Gets or sets the weight for a given state id.
                /// </summary>
                /// <param name="stateId">The state id.</param>
                /// <returns>The weight.</returns>
                public Weight this[int stateId]
                {
                    get => this.stateIdToWeight[stateId];
                    set => this.stateIdToWeight[stateId] = value;
                }

                /// <summary>
                /// Adds a given state id and a weight to the set.
                /// </summary>
                /// <param name="stateId">The state id.</param>
                /// <param name="weight">The weight.</param>
                public void Add(int stateId, Weight weight) =>
                    this.stateIdToWeight.Add(stateId, weight);

                /// <summary>
                /// Attempts to retrieve the weight corresponding to a given state id from the set.
                /// </summary>
                /// <param name="stateId">The state id.</param>
                /// <param name="weight">When the method returns, contains the retrieved weight.</param>
                /// <returns>
                /// <see langword="true"/> if the given state id was present in the set,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public bool TryGetWeight(int stateId, out Weight weight) =>
                    this.stateIdToWeight.TryGetValue(stateId, out weight);

                /// <summary>
                /// Checks whether the state with a given id is present in the set.
                /// </summary>
                /// <param name="stateId">The state id,</param>
                /// <returns>
                /// <see langword="true"/> if the given state id was present in the set,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public bool ContainsState(int stateId) =>
                    this.stateIdToWeight.ContainsKey(stateId);

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
                public IEnumerator<KeyValuePair<int, Weight>> GetEnumerator() =>
                    this.stateIdToWeight.GetEnumerator();

                /// <summary>
                /// Gets the enumerator.
                /// </summary>
                /// <returns>
                /// The enumerator.
                /// </returns>
                IEnumerator IEnumerable.GetEnumerator() =>
                    this.GetEnumerator();

                #endregion
            }
        }
    }
}
