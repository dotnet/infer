// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent a transition in an automaton.#
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
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
        public readonly struct Transition
        {
            //// This class has been made inner so that the user doesn't have to deal with a lot of generic parameters on it.

            /// Note the order of fields. This struct is densely packed and should take only
            /// 24 bytes to store all its fields in case of StringAutomaton.Transition.
            /// Think wisely before you decide to reorder fields or change their types.

            [DataMember]
            private readonly int destinationStateIndex;

            [DataMember]
            private readonly short group;

            [DataMember]
            private readonly short hasElementDistribution;

            [DataMember]
            private readonly TElementDistribution elementDistribution;

            [DataMember]
            private readonly Weight weight;

            /// <summary>
            /// Initializes a new instance of the <see cref="Transition"/> struct.
            /// </summary>
            /// <param name="elementDistribution">The element distribution associated with the transition, or <see langword="null"/> if this is an epsilon transition.</param>
            /// <param name="weight">The weight associated with the transition.</param>
            /// <param name="destinationStateIndex">The index of the destination state of the transition.</param>
            /// <param name="group">The group this transition belongs to.</param>
            [Construction("ElementDistribution", "Weight", "DestinationStateIndex", "Group")]
            public Transition(Option<TElementDistribution> elementDistribution, Weight weight, int destinationStateIndex, int group = 0)
                : this()
            {
                Argument.CheckIfInRange(destinationStateIndex >= 0, nameof(destinationStateIndex), "A destination state index cannot be negative.");

                this.hasElementDistribution = (short)(elementDistribution.HasValue ? 1 : 0);
                if (elementDistribution.HasValue)
                {
                    this.elementDistribution = elementDistribution.Value;
                }

                this.destinationStateIndex = destinationStateIndex;
                this.weight = weight;
                this.group = (short)group;
            }

            public Transition With(Option<TElementDistribution>? elementDistribution = null, Weight? weight = null, int? destinationStateIndex = null, int? group = null) =>
                new Transition(
                    elementDistribution ?? this.ElementDistribution,
                    weight ?? this.Weight,
                    destinationStateIndex ?? this.DestinationStateIndex,
                    group ?? this.Group);

            /// <summary>
            /// Gets the destination state index.
            /// </summary>
            public int DestinationStateIndex
            {
                get => this.destinationStateIndex;
            }

            /// <summary>
            /// Gets or sets the group this transition belongs to.
            /// </summary>
            public int Group
            {
                get => this.group;
            }

            /// <summary>
            /// Gets the element distribution for this transition.
            /// </summary>
            public Option<TElementDistribution> ElementDistribution
            {
                get => this.hasElementDistribution == 0 ? Option.None : Option.Some(this.elementDistribution);
            }

            /// <summary>
            /// Gets a value indicating whether this transition is an epsilon transition.
            /// </summary>
            public bool IsEpsilon => this.hasElementDistribution == 0;

            /// <summary>
            /// Gets or sets the weight associated with this transition.
            /// </summary>
            public Weight Weight
            {
                get => this.weight;
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
                sb.Append(this.ElementDistribution.HasValue ? this.ElementDistribution.ToString() : "eps");
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

                var groupAndHasElementDistribution = this.Group << 1 | (this.hasElementDistribution != 0 ? 1 : 0);
                writeInt32(groupAndHasElementDistribution);

                if (this.hasElementDistribution != 0)
                {
                    writeElementDistribution(this.elementDistribution);
                }

                this.Weight.Write(writeDouble);
            }

            /// <summary>
            /// Reads a transition.
            /// </summary>
            public static Transition Read(Func<int> readInt32, Func<double> readDouble, Func<TElementDistribution> readElementDistribution)
            {
                var destinationStateIndex = readInt32();
                var groupAndHasElementDistribution = readInt32();
                Option<TElementDistribution> elementDistribution = Option.None;
                if ((groupAndHasElementDistribution & 0x1) == 0x1)
                {
                    elementDistribution = readElementDistribution();
                }
                var group = groupAndHasElementDistribution >> 1;
                var weight = Weight.Read(readDouble);
                return new Transition(elementDistribution, weight, destinationStateIndex, group);
            }
        }
    }
}
