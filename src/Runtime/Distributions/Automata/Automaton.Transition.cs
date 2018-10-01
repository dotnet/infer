// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent a transition in an automaton.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : class, IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Represents a transition in an automaton.
        /// </summary>
        /// <remarks>
        /// Whenever a transition is being cloned,
        /// the element distribution is not being cloned, so the two copies share the same element distribution instance.
        /// Therefore, a manual change to an element distribution can affect multiple functions and should be avoided.
        /// </remarks>
        [Serializable]
        [DataContract]
        public struct Transition
        {
            //// This class has been made inner so that the user doesn't have to deal with a lot of generic parameters on it.
            
            /// <summary>
            /// Initializes a new instance of the <see cref="Transition"/> struct.
            /// </summary>
            /// <param name="elementDistribution">The element distribution associated with the transition, or <see langword="null"/> if this is an epsilon transition.</param>
            /// <param name="weight">The weight associated with the transition.</param>
            /// <param name="destinationStateIndex">The index of the destination state of the transition.</param>
            /// <param name="group">The group this transition belongs to.</param>
            [Construction("ElementDistribution", "Weight", "DestinationStateIndex", "Group")]
            public Transition(TElementDistribution elementDistribution, Weight weight, int destinationStateIndex, byte group = 0)
                : this()
            {
                Argument.CheckIfInRange(destinationStateIndex >= 0, "destinationStateIndex", "A destination state index cannot be negative.");

                this.ElementDistribution = elementDistribution;
                this.DestinationStateIndex = destinationStateIndex;
                this.Weight = weight;
                this.Group = group;
            }

            /// <summary>
            /// Gets or sets the destination state index.
            /// </summary>
            [DataMember]
            public int DestinationStateIndex { get; set; }

            /// <summary>
            /// Gets or sets the element distribution for this transition.
            /// </summary>
            [DataMember]
            public TElementDistribution ElementDistribution { get; set; }

            /// <summary>
            /// Gets or sets the weight associated with this transition.
            /// </summary>
            [DataMember]
            public Weight Weight { get; set; }
            
            /// <summary>
            /// Gets or sets the group this transition belongs to.
            /// </summary>
            [DataMember]
            public byte Group { get; set; }

            /// <summary>
            /// Gets a value indicating whether this transition is an epsilon transition.
            /// </summary>
            public bool IsEpsilon
            {
                get { return this.ElementDistribution == null; }
            }

            /// <summary>
            /// Replaces the configuration of this transition with the configuration of a given transition.
            /// </summary>
            /// <param name="transition">
            /// The transition which configuration would be used to replace the configuration of the current transition.
            /// </param>
            public void SetTo(Transition transition)
            {
                this.ElementDistribution = transition.ElementDistribution;
                this.Weight = transition.Weight;
                this.DestinationStateIndex = transition.DestinationStateIndex;
                this.Group = transition.Group;
            }

            /// <summary>
            /// Gets the string representation of this transition.
            /// </summary>
            /// <returns>The string representation of this transition.</returns>
            public override string ToString()
            {
                var sb = new StringBuilder();

                if (this.Group != 0)
                {
                    sb.Append("#" + this.Group);
                }

                sb.Append('[');
                sb.Append(this.ElementDistribution == null ? "eps" : this.ElementDistribution.ToString());
                sb.Append(']');
                sb.Append(" " + this.Weight.Value);
                sb.Append(" -> " + this.DestinationStateIndex);

                return sb.ToString();
            }

            /// <summary>
            /// Writes the transition.
            /// </summary>
            public void Write(Action<int> writeInt32, Action<double> writeDouble, Action<TElementDistribution> writeElementDistribution)
            {
                writeInt32(this.DestinationStateIndex);
                var hasElementDistribution = this.ElementDistribution != null;

                var groupAndHasElementDistribution = this.Group << 1 | (hasElementDistribution ? 1 : 0);
                writeInt32(groupAndHasElementDistribution);

                if (hasElementDistribution)
                {
                    writeElementDistribution(this.ElementDistribution);
                }

                this.Weight.Write(writeDouble);
            }

            /// <summary>
            /// Reads a transition.
            /// </summary>
            public static Transition Read(Func<int> readInt32, Func<double> readDouble, Func<TElementDistribution> readElementDistribution)
            {
                var res = new Transition();
                res.DestinationStateIndex = readInt32();
                var groupAndHasElementDistribution = readInt32();
                if ((groupAndHasElementDistribution & 0x1) == 0x1)
                {
                    res.ElementDistribution = readElementDistribution();
                }
                res.Group = (byte)(groupAndHasElementDistribution >> 1);
                res.Weight = Weight.Read(readDouble);
                return res;
            }
        }
    }
}
