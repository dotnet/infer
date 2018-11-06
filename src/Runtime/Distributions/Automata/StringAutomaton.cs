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
        /// Constructor used during deserialization by Newtonsoft.Json and BinaryFormatter .
        /// </summary>
        protected StringAutomaton(SerializationInfo info, StreamingContext context)
            : base(info, context)
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
        protected override IEnumerable<Tuple<DiscreteChar, Weight, Determinization.WeightedStateSet>> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState)
        {
            const double LogEps = -35; // Don't add transitions with log-weight less than this as they have been produced by numerical inaccuracies
            
            // Build a list of numbered non-zero probability character segment bounds (they are numbered here due to perf. reasons)
            var segmentBounds = new List<Tuple<int, TransitionCharSegmentBound>>();
            foreach (KeyValuePair<int, Weight> stateIdWeight in sourceState)
            {

                var state = this.States[stateIdWeight.Key];
                for (int i = 0; i < state.TransitionCount; ++i)
                {
                    AddTransitionCharSegmentBounds(state.GetTransition(i), stateIdWeight.Value, segmentBounds);
                }
            }

            // Sort segment bounds left-to-right, start-to-end
            var sortedIndexedSegmentBounds = segmentBounds.OrderBy(tup => tup.Item2);

            // Produce an outgoing transition for each unique subset of overlapping segments
            var result = new List<Tuple<DiscreteChar, Weight, Determinization.WeightedStateSet>>();
            Weight currentSegmentStateWeightSum = Weight.Zero;
            var currentSegmentStateWeights = segmentBounds.Select(b => b.Item2.DestinationStateId).Distinct().ToDictionary(d => d, d => Weight.Zero);
            var activeSegments = new HashSet<TransitionCharSegmentBound>();
            int currentSegmentStart = char.MinValue;
            foreach (var tup in sortedIndexedSegmentBounds)
            {
                TransitionCharSegmentBound segmentBound = tup.Item2;

                if (currentSegmentStateWeightSum.LogValue > LogEps && currentSegmentStart < segmentBound.Bound)
                {
                    // Flush previous segment
                    char segmentEnd = (char)(segmentBound.Bound - 1);
                    int segmentLength = segmentEnd - currentSegmentStart + 1;
                    DiscreteChar elementDist = DiscreteChar.InRange((char)currentSegmentStart, segmentEnd);

                    var destinationState = new Determinization.WeightedStateSet();
                    foreach (KeyValuePair<int, Weight> stateIdWithWeight in currentSegmentStateWeights)
                    {
                        if (stateIdWithWeight.Value.LogValue > LogEps)
                        {
                            Weight stateWeight = Weight.Product(stateIdWithWeight.Value, Weight.Inverse(currentSegmentStateWeightSum));
                            destinationState.Add(stateIdWithWeight.Key, stateWeight);
                        }
                    }

                    Weight transitionWeight = Weight.Product(Weight.FromValue(segmentLength), currentSegmentStateWeightSum);
                    result.Add(Tuple.Create(elementDist, transitionWeight, destinationState));
                }

                // Update current segment
                currentSegmentStart = segmentBound.Bound;

                if (segmentBound.IsStart)
                {
                    activeSegments.Add(segmentBound);
                    currentSegmentStateWeightSum = Weight.Sum(currentSegmentStateWeightSum, segmentBound.Weight);
                    currentSegmentStateWeights[segmentBound.DestinationStateId] = Weight.Sum(currentSegmentStateWeights[segmentBound.DestinationStateId], segmentBound.Weight);
                }
                else
                {
                    Debug.Assert(currentSegmentStateWeights.ContainsKey(segmentBound.DestinationStateId), "We shouldn't exit a state we didn't enter.");
                    activeSegments.Remove(segmentBounds[tup.Item1 - 1].Item2);  // End follows start in original.
                    if (double.IsInfinity(segmentBound.Weight.Value))
                    {
                        // Cannot subtract because of the infinities involved.
                        currentSegmentStateWeightSum = activeSegments.Select(sb => sb.Weight).Aggregate(Weight.Zero, (acc, w) => Weight.Sum(acc, w));
                        currentSegmentStateWeights[segmentBound.DestinationStateId] =
                            activeSegments.Where(sb => sb.DestinationStateId == segmentBound.DestinationStateId).Select(sb => sb.Weight).Aggregate(Weight.Zero, (acc, w) => Weight.Sum(acc, w));
                    }
                    else
                    {
                        currentSegmentStateWeightSum = activeSegments.Count == 0 ? Weight.Zero : Weight.AbsoluteDifference(currentSegmentStateWeightSum, segmentBound.Weight);

                        Weight prevStateWeight = currentSegmentStateWeights[segmentBound.DestinationStateId];
                        currentSegmentStateWeights[segmentBound.DestinationStateId] = Weight.AbsoluteDifference(
                            prevStateWeight, segmentBound.Weight);
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Given a transition and the residual weight of its source state, adds weighted non-zero probability character segments
        /// associated with the transition to the list.
        /// </summary>
        /// <param name="transition">The transition.</param>
        /// <param name="sourceStateResidualWeight">The logarithm of the residual weight of the source state of the transition.</param>
        /// <param name="bounds">The list for storing numbered segment bounds.</param>
        private static void AddTransitionCharSegmentBounds(
            Transition transition, Weight sourceStateResidualWeight, List<Tuple<int, TransitionCharSegmentBound>> bounds)
        {
            var probs = (PiecewiseVector)transition.ElementDistribution.Value.GetProbs();
            int commonValueStart = char.MinValue;
            Weight commonValue = Weight.FromValue(probs.CommonValue);
            Weight weightBase = Weight.Product(transition.Weight, sourceStateResidualWeight);
            TransitionCharSegmentBound newSegmentBound;

            ////if (double.IsInfinity(weightBase.Value))
            ////{
            ////    Console.WriteLine("Weight base infinity");
            ////}

            for (int i = 0; i < probs.Pieces.Count; ++i)
            {
                ConstantVector piece = probs.Pieces[i];
                if (piece.Start > commonValueStart && !commonValue.IsZero)
                {
                    // Add endpoints for the common value
                    Weight segmentWeight = Weight.Product(commonValue, weightBase);
                    newSegmentBound = new TransitionCharSegmentBound(commonValueStart, transition.DestinationStateIndex, segmentWeight, true);
                    bounds.Add(new Tuple<int, TransitionCharSegmentBound>(bounds.Count, newSegmentBound));
                    newSegmentBound = new TransitionCharSegmentBound(piece.Start, transition.DestinationStateIndex, segmentWeight, false);
                    bounds.Add(new Tuple<int, TransitionCharSegmentBound>(bounds.Count, newSegmentBound));
                }

                // Add segment endpoints
                Weight pieceValue = Weight.FromValue(piece.Value);
                if (!pieceValue.IsZero)
                {
                    Weight segmentWeight = Weight.Product(pieceValue, weightBase);
                    newSegmentBound = new TransitionCharSegmentBound(piece.Start, transition.DestinationStateIndex, segmentWeight, true);
                    bounds.Add(new Tuple<int, TransitionCharSegmentBound>(bounds.Count, newSegmentBound));
                    newSegmentBound = new TransitionCharSegmentBound(piece.End + 1, transition.DestinationStateIndex, segmentWeight, false);
                    bounds.Add(new Tuple<int, TransitionCharSegmentBound>(bounds.Count,newSegmentBound));    
                }
                
                commonValueStart = piece.End + 1;
            }

            if (!commonValue.IsZero && (probs.Pieces.Count == 0 || probs.Pieces[probs.Pieces.Count - 1].End != char.MaxValue))
            {
                // Add endpoints for the last common value segment
                Weight segmentWeight = Weight.Product(commonValue, weightBase);
                newSegmentBound = new TransitionCharSegmentBound(commonValueStart, transition.DestinationStateIndex, segmentWeight, true);
                bounds.Add(new Tuple<int, TransitionCharSegmentBound>(bounds.Count, newSegmentBound));
                newSegmentBound = new TransitionCharSegmentBound(char.MaxValue + 1, transition.DestinationStateIndex, segmentWeight, false);
                bounds.Add(new Tuple<int, TransitionCharSegmentBound>(bounds.Count, newSegmentBound));
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
