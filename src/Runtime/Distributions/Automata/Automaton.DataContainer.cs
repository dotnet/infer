// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Collections;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Immutable container for automaton data - states and transitions.
        /// </summary>
        [Serializable]
        public struct DataContainer : ISerializable
        {
            /// <summary>
            /// Stores
            /// </summary>
            private readonly Flags flags;

            /// <summary>
            /// Index of start state of automaton.
            /// </summary>
            public readonly int StartStateIndex;
            
            /// <summary>
            /// All automaton states.
            /// </summary>
            public readonly ReadOnlyArray<StateData> States;

            /// <summary>
            /// All automaton transitions. Transitions for the same state are stored as a contiguous block
            /// inside this array.
            /// </summary>
            public readonly ReadOnlyArray<Transition> Transitions;

            /// <summary>
            /// Gets value indicating whether this automaton is epsilon-free.
            /// </summary>
            public bool IsEpsilonFree => (this.flags & Flags.IsEpsilonFree) != 0;

            /// <summary>
            /// Get value indicating whether this automaton uses groups.
            /// </summary>
            public bool UsesGroups => (this.flags & Flags.UsesGroups) != 0;

            /// <summary>
            /// Initializes instance of <see cref="DataContainer"/>.
            /// </summary>
            [Construction("StartStateIndex", "IsEpsilonFree", "UsesGroups", "States", "Transitions")]
            public DataContainer(
                int startStateIndex,
                bool isEpsilonFree,
                bool usesGroups,
                ReadOnlyArray<StateData> states,
                ReadOnlyArray<Transition> transitions)
            {
                this.flags =
                    (isEpsilonFree ? Flags.IsEpsilonFree : 0) |
                    (usesGroups ? Flags.UsesGroups : 0);
                this.StartStateIndex = startStateIndex;
                this.States = states;
                this.Transitions = transitions;
            }

            /// <summary>
            /// Returns true if indices assigned to given states and their transitions are consistent with each other.
            /// </summary>
            public bool IsConsistent()
            {
                if (this.StartStateIndex < 0 || this.StartStateIndex >= this.States.Count)
                {
                    return false;
                }

                var isEpsilonFree = true;

                foreach (var state in this.States)
                {
                    if (state.FirstTransition < 0 || state.LastTransition > this.Transitions.Count)
                    {
                        return false;
                    }

                    for (var i = state.FirstTransition; i < state.LastTransition; ++i)
                    {
                        var transition = this.Transitions[i];
                        if (transition.DestinationStateIndex < 0 ||
                            transition.DestinationStateIndex >= this.States.Count)
                        {
                            return false;
                        }

                        if (transition.IsEpsilon)
                        {
                            isEpsilonFree = false;
                        }
                    }
                }

                return this.IsEpsilonFree == isEpsilonFree;
            }

            #region Serialization

            /// <summary>
            /// Constructor used by Json and BinaryFormatter serializers. Informally needed to be
            /// implemented for <see cref="ISerializable"/> interface.
            /// </summary>
            internal DataContainer(SerializationInfo info, StreamingContext context)
            {
                this.flags = (Flags)info.GetValue(nameof(this.flags), typeof(Flags));
                this.StartStateIndex = (int)info.GetValue(nameof(this.StartStateIndex), typeof(int));
                this.States = (StateData[])info.GetValue(nameof(this.States), typeof(StateData[]));
                this.Transitions = (Transition[])info.GetValue(nameof(this.Transitions), typeof(Transition[]));

                if (!IsConsistent())
                {
                    throw new Exception("Deserialized automaton is inconsistent!");
                }
            }

            /// <inheritdoc/>
            void ISerializable.GetObjectData(SerializationInfo info, StreamingContext context)
            {
                info.AddValue(nameof(this.States), this.States.CloneArray());
                info.AddValue(nameof(this.Transitions), this.Transitions.CloneArray());
                info.AddValue(nameof(this.StartStateIndex), this.StartStateIndex);
                info.AddValue(nameof(this.flags), this.flags);
            }

            #endregion

            [Flags]
            private enum Flags
            {
                IsEpsilonFree = 0x1,
                UsesGroups = 0x2,
            }
        }
    }
}
