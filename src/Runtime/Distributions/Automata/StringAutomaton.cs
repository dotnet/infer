// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.IO;

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
            var destinationStateSetBuilder = Determinization.WeightedStateSetBuilder.Create();

            var segmentBounds = CollectCharSegmentBounds(sourceStateSet);
            var currentSegmentStart = 0;

            foreach (var segmentBound in segmentBounds)
            {
                if (currentSegmentStart != segmentBound.Position &&
                    destinationStateSetBuilder.StatesCount() > 0)
                {
                    // Flush previous segment
                    var (destinationStateSet, destinationStateSetWeight) = destinationStateSetBuilder.Get();
                    var segmentLength = segmentBound.Position - currentSegmentStart;
                    yield return new Determinization.OutgoingTransition(
                        DiscreteChar.InRange((char)currentSegmentStart, (char)(segmentBound.Position - 1)),
                        Weight.FromValue(segmentLength) * destinationStateSetWeight,
                        destinationStateSet);

                }

                // Update current segment
                currentSegmentStart = segmentBound.Position;
                if (segmentBound.IsEnd)
                {
                    destinationStateSetBuilder.Remove(segmentBound.DestinationStateId, segmentBound.Weight);
                }
                else
                {
                    destinationStateSetBuilder.Add(segmentBound.DestinationStateId, segmentBound.Weight);
                }
            }
        }

        private List<TransitionCharSegmentBound> CollectCharSegmentBounds(
            Determinization.WeightedStateSet sourceStateSet)
        {
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
            return segmentBounds;
        }

        /// <summary>
        /// Given a transition and the residual weight of its source state, adds weighted non-zero
        /// probability character ranges asociated with the transition to the list.
        /// </summary>
        /// <param name="transition">The transition.</param>
        /// <param name="sourceStateResidualWeight">The logarithm of the residual weight of the source state of the transition.</param>
        /// <param name="segmentBounds">The list for storing weighted ranges.</param>
        private static void AddTransitionCharSegmentBounds(
            Transition transition, Weight sourceStateResidualWeight, List<TransitionCharSegmentBound> segmentBounds)
        {
            var distribution = transition.ElementDistribution.Value;
            var ranges = distribution.Ranges;
            var weightBase = transition.Weight * sourceStateResidualWeight;

            foreach (var range in ranges)
            {
                var fullWeight = range.Probability * weightBase;
                segmentBounds.Add(new TransitionCharSegmentBound(
                    range.StartInclusive, transition.DestinationStateIndex, fullWeight, false));
                segmentBounds.Add(new TransitionCharSegmentBound(
                    range.EndExclusive, transition.DestinationStateIndex, fullWeight, true));
            }
        }

        /// <summary>
        /// Represents the start or the end of a segment of characters associated with a transition.
        /// </summary>
        private struct TransitionCharSegmentBound : IComparable<TransitionCharSegmentBound>
        {
            private const int IsEndFlag = 1 << 31;
            private const int DestinationIdMask = 0x7fffffff;

            /// <summary>
            /// Low 31 bits encode state id. High (sign) bit encodes "isEnd"
            /// </summary>
            /// <remarks>
            /// Such encoding is used to fit whole struct into 16 bytes.
            /// </remarks>
            private readonly int destinationStateId;

            /// <summary>
            /// Initializes a new instance of the <see cref="TransitionCharSegmentBound"/> struct.
            /// </summary>
            /// <param name="position">The bound (segment start or segment end + 1).</param>
            /// <param name="destinationStateId">The destination state id of the transition that produced the segment.</param>
            /// <param name="weight">The weight of the segment.</param>
            /// <param name="isEnd">Whether this instance represents the start of the segment.</param>
            public TransitionCharSegmentBound(int position, int destinationStateId, Weight weight, bool isEnd)
                : this()
            {
                this.Position = position;
                this.destinationStateId = destinationStateId | (isEnd ? IsEndFlag : 0);
                this.Weight = weight;
            }

            /// <summary>
            /// Gets the bound (segment start or segment end + 1).
            /// </summary>
            public int Position { get; }

            /// <summary>
            /// Gets the destination state id of the transition that produced this segment.
            /// </summary>
            public int DestinationStateId => this.destinationStateId & DestinationIdMask;

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
            public bool IsEnd => (this.destinationStateId & IsEndFlag) != 0;

            /// <summary>
            /// Compares this instance to a given one.
            /// </summary>
            /// <param name="that">The other instance.</param>
            /// <returns>The that result.</returns>
            /// <remarks>
            /// Instances are first compared by <see cref="Position"/>, and then by <see cref="IsEnd"/>
            /// (end goes before start).
            /// </remarks>
            public int CompareTo(TransitionCharSegmentBound that) =>
                (Bound: this.Position, this.destinationStateId).CompareTo((that.Position, that.destinationStateId));

            /// <summary>
            /// Returns a string representation of this instance.
            /// </summary>
            /// <returns>String representation of this instance.</returns>
            public override string ToString() =>
                string.Format(
                    "Bound: {0}, Dest: {1}, Weight: {2}, {3}",
                    this.Position,
                    this.DestinationStateId,
                    this.Weight,
                    this.IsEnd ? "End" : "Start");
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
