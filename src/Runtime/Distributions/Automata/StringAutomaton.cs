// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a weighted finite state automaton defined on <see cref="string"/>.
    /// </summary>
    [Serializable]
    public class StringAutomaton : Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>
    {
        public StringAutomaton()
        {
        }

        /// <summary>
        /// Computes a set of outgoing transitions from a given state of the determinization result.
        /// </summary>
        /// <param name="sourceStateSet">The source state of the determinized automaton represented as 
        /// a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton.</param>
        /// <returns>
        /// A collection of (element distribution, weight, weighted state set) triples corresponding to outgoing
        /// transitions from <paramref name="sourceStateSet"/>.
        /// The first two elements of a tuple define the element distribution and the weight of a transition.
        /// The third element defines the outgoing state.
        /// </returns>
        protected override IEnumerable<Determinization.OutgoingTransition> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceStateSet)
        {
            // Build a list of numbered non-zero probability character segment bounds (they are numbered here due to perf. reasons)
            var segmentBounds = new List<TransitionCharSegmentBound>();
            for (var i = 0; i < sourceStateSet.Count; ++i)
            {
                var sourceState = sourceStateSet[i];
                var state = this.States[sourceState.Index];
                foreach (var transition in state.Transitions)
                {
                    AddTransitionCharSegmentBounds(transition, sourceState.Weight, segmentBounds);
                }
            }

            segmentBounds.Sort();

            // Produce an outgoing transition for each unique subset of overlapping segments
            var currentSegmentTotal = WeightSum.Zero();

            var currentSegmentStateWeights = new Dictionary<int, WeightSum>();
            var currentSegmentStart = (int)char.MinValue;
            var destinationStateSetBuilder = Determinization.WeightedStateSetBuilder.Create();
            foreach (var segmentBound in segmentBounds)
            {
                if (currentSegmentTotal.Count != 0 && currentSegmentStart < segmentBound.Bound)
                {
                    // Flush previous segment
                    var segmentEnd = (char)(segmentBound.Bound - 1);
                    var segmentLength = segmentEnd - currentSegmentStart + 1;
                    var elementDist = DiscreteChar.InRange((char)currentSegmentStart, segmentEnd);
                    var invTotalWeight = Weight.Inverse(currentSegmentTotal.Sum);

                    destinationStateSetBuilder.Reset();
                    foreach (var stateIdWithWeight in currentSegmentStateWeights)
                    {
                        var stateWeight = stateIdWithWeight.Value.Sum * invTotalWeight;
                        destinationStateSetBuilder.Add(stateIdWithWeight.Key, stateWeight);
                    }

                    var (destinationStateSet, destinationStateSetWeight) = destinationStateSetBuilder.Get();

                    var transitionWeight = Weight.Product(
                        Weight.FromValue(segmentLength),
                        currentSegmentTotal.Sum,
                        destinationStateSetWeight);
                    yield return new Determinization.OutgoingTransition(
                        elementDist, transitionWeight, destinationStateSet);
                }

                // Update current segment
                currentSegmentStart = segmentBound.Bound;

                if (segmentBound.IsStart)
                {
                    currentSegmentTotal += segmentBound.Weight;
                    if (currentSegmentStateWeights.TryGetValue(segmentBound.DestinationStateId, out var stateWeight))
                    {
                        currentSegmentStateWeights[segmentBound.DestinationStateId] =
                            stateWeight + segmentBound.Weight;
                    }
                    else
                    {
                        currentSegmentStateWeights[segmentBound.DestinationStateId] = new WeightSum(segmentBound.Weight);
                    }
                }
                else
                {
                    Debug.Assert(currentSegmentStateWeights.ContainsKey(segmentBound.DestinationStateId), "We shouldn't exit a state we didn't enter.");
                    Debug.Assert(!segmentBound.Weight.IsInfinity);
                    currentSegmentTotal -= segmentBound.Weight;

                    var prevStateWeight = currentSegmentStateWeights[segmentBound.DestinationStateId];
                    var newStateWeight = prevStateWeight - segmentBound.Weight;
                    if (newStateWeight.Count == 0)
                    {
                        currentSegmentStateWeights.Remove(segmentBound.DestinationStateId);
                    }
                    else
                    {
                        currentSegmentStateWeights[segmentBound.DestinationStateId] = newStateWeight;
                    }
                }
            }
        }

        /// <summary>
        /// Given a transition and the residual weight of its source state, adds weighted non-zero probability character segments
        /// associated with the transition to the list.
        /// </summary>
        /// <param name="transition">The transition.</param>
        /// <param name="sourceStateResidualWeight">The logarithm of the residual weight of the source state of the transition.</param>
        /// <param name="bounds">The list for storing numbered segment bounds.</param>
        private static void AddTransitionCharSegmentBounds(
            Transition transition, Weight sourceStateResidualWeight, List<TransitionCharSegmentBound> bounds)
        {
            var distribution = transition.ElementDistribution.Value;
            var ranges = distribution.Ranges;
            var commonValueStart = (int)char.MinValue;
            var commonValue = distribution.ProbabilityOutsideRanges;
            var weightBase = transition.Weight * sourceStateResidualWeight;

            void AddEndPoints(int start, int end, int destinationIndex, Weight weight)
            {
                bounds.Add(new TransitionCharSegmentBound(start, destinationIndex, weight * weightBase, true));
                bounds.Add(new TransitionCharSegmentBound(end, destinationIndex, weight * weightBase, false));
            }

            foreach (var range in ranges)
            {
                if (range.StartInclusive > commonValueStart && !commonValue.IsZero)
                {
                    AddEndPoints(commonValueStart, range.StartInclusive, transition.DestinationStateIndex, commonValue);
                }

                // Add segment endpoints
                var pieceValue = range.Probability;
                if (!pieceValue.IsZero)
                {
                    AddEndPoints(range.StartInclusive, range.EndExclusive, transition.DestinationStateIndex, pieceValue);
                }

                commonValueStart = range.EndExclusive;
            }

            if (!commonValue.IsZero && (ranges.Count == 0 || ranges[ranges.Count - 1].EndExclusive != DiscreteChar.CharRangeEndExclusive))
            {
                AddEndPoints(commonValueStart, char.MaxValue + 1, transition.DestinationStateIndex, commonValue);
            }
        }

        /// <summary>
        /// Represents the start or the end of a segment of characters associated with a transition.
        /// </summary>
        private struct TransitionCharSegmentBound : IComparable<TransitionCharSegmentBound>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="TransitionCharSegmentBound"/> struct.
            /// </summary>
            /// <param name="bound">The bound (segment start or segment end + 1).</param>
            /// <param name="destinationStateId">The destination state id of the transition that produced the segment.</param>
            /// <param name="weight">The weight of the segment.</param>
            /// <param name="isStart">Whether this instance represents the start of the segment.</param>
            public TransitionCharSegmentBound(int bound, int destinationStateId, Weight weight, bool isStart)
                : this()
            {
                this.Bound = bound;
                this.DestinationStateId = destinationStateId;
                this.Weight = weight;
                this.IsStart = isStart;
            }

            /// <summary>
            /// Gets the bound (segment start or segment end + 1).
            /// </summary>
            public int Bound { get; private set; }

            /// <summary>
            /// Gets the destination state id of the transition that produced this segment.
            /// </summary>
            public int DestinationStateId { get; private set; }

            /// <summary>
            /// Gets the weight of the segment.
            /// </summary>
            /// <remarks>
            /// The weight of the segment is defined as the product of the following:
            /// the weight of the transition that produced the segment,
            /// the value of the element distribution on the segment,
            /// and the residual weight of the source state of the transition.
            /// </remarks>
            public Weight Weight { get; private set; }

            /// <summary>
            /// Gets a value indicating whether this instance represents the start of the segment.
            /// </summary>
            public bool IsStart { get; private set; }

            /// <summary>
            /// Compares this instance to a given one.
            /// </summary>
            /// <param name="other">The other instance.</param>
            /// <returns>The comparison result.</returns>
            /// <remarks>
            /// Instances are first compared by <see cref="Bound"/>, and then by <see cref="IsStart"/> (start goes before end).
            /// </remarks>
            public int CompareTo(TransitionCharSegmentBound other)
            {
                if (this.Bound != other.Bound)
                {
                    return this.Bound.CompareTo(other.Bound);
                }

                return other.IsStart.CompareTo(this.IsStart);
            }

            /// <summary>
            /// Returns a string representation of this instance.
            /// </summary>
            /// <returns>String representation of this instance.</returns>
            public override string ToString()
            {
                return String.Format(
                    "Bound: {0}, Dest: {1}, Weight: {2}, {3}",
                    this.Bound,
                    this.DestinationStateId,
                    this.Weight,
                    this.IsStart ? "Start" : "End");
            }

            /// <summary>
            /// Overrides equals.
            /// </summary>
            /// <param name="obj">An object.</param>
            /// <returns>True if equal, false otherwise.</returns>
            public override bool Equals(object obj)
            {
                return
                    obj is TransitionCharSegmentBound that &&
                    this.DestinationStateId == that.DestinationStateId &&
                    this.Bound == that.Bound &&
                    this.IsStart == that.IsStart &&
                    this.Weight == that.Weight;
            }

            /// <summary>
            /// Get a hash code for the instance
            /// </summary>
            /// <returns>The hash code.</returns>
            public override int GetHashCode()
            {
                // Make hash code from the main distinguishers. Could
                // consider adding in IsStart here as well, but currently
                // only start segments are put in a hashset/dictionary.
                return Hash.Combine(
                    this.DestinationStateId.GetHashCode(),
                    this.Bound.GetHashCode());
            }
        }

        /// <summary>
        /// Reads an automaton from.
        /// </summary>
        public static StringAutomaton Read(BinaryReader reader) => 
            Read(reader.ReadDouble, reader.ReadInt32, () => DiscreteChar.Read(reader.ReadInt32, reader.ReadDouble));

        /// <summary>
        /// Writes the current automaton.
        /// </summary>
        public void Write(BinaryWriter writer) => 
            Write(writer.Write, writer.Write, c => c.Write(writer.Write, writer.Write));
    }
}
