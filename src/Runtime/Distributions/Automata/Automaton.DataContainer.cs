// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Diagnostics;
    using System.Runtime.Serialization;
    
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Serialization;

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
            /// Gets value indicating whether this automaton is determinized
            /// </summary>
            /// <remarks>
            /// Null value means that this property is unknown.
            /// False value means that this automaton can not be determinized
            /// </remarks>
            public bool? IsDeterminized =>
                (this.flags & Flags.DeterminizationStateKnown) != 0
                    ? (this.flags & Flags.IsDeterminized) != 0
                    : (bool?)null;

            /// <summary>
            /// Gets value indicating whether this automaton is zero
            /// </summary>
            /// <remarks>
            /// Null value means that this property is unknown.
            /// </remarks>
            public bool? IsZero =>
                ((this.flags & Flags.IsZeroStateKnown) != 0)
                    ? (this.flags & Flags.IsZero) != 0
                    : (bool?)null;

            /// <summary>
            /// Initializes instance of <see cref="DataContainer"/>.
            /// </summary>
            [Construction("StartStateIndex", "IsEpsilonFree", "UsesGroups", "DeterminizationState", "IsZeroState", "States", "Transitions")]
            public DataContainer(
                int startStateIndex,
                ReadOnlyArray<StateData> states,
                ReadOnlyArray<Transition> transitions,
                bool isEpsilonFree,
                bool usesGroups,
                bool? isDeterminized,
                bool? isZero)
            {
                this.flags =
                    (isEpsilonFree ? Flags.IsEpsilonFree : 0) |
                    (usesGroups ? Flags.UsesGroups : 0) |
                    (isDeterminized.HasValue ? Flags.DeterminizationStateKnown : 0) |
                    (isDeterminized == true ? Flags.IsDeterminized : 0) |
                    (isZero.HasValue ? Flags.IsZeroStateKnown : 0) |
                    (isZero == true ? Flags.IsZero : 0);
                this.StartStateIndex = startStateIndex;
                this.States = states;
                this.Transitions = transitions;
            }

            public DataContainer With(
                bool? isDetermenized = null,
                bool? isZero= null)
            {
                // Can't overwrite known properties
                Debug.Assert(isDetermenized.HasValue != this.IsDeterminized.HasValue || isDetermenized == this.IsDeterminized);
                Debug.Assert(isZero.HasValue != this.IsZero.HasValue || isZero == this.IsZero);

                return new DataContainer(
                    this.StartStateIndex,
                    this.States,
                    this.Transitions,
                    this.IsEpsilonFree,
                    this.UsesGroups,
                    isDetermenized ?? this.IsDeterminized,
                    isZero ?? this.IsZero);
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
                    var lastTransitionIndex = state.FirstTransitionIndex + state.TransitionsCount;
                    if (state.FirstTransitionIndex < 0 || lastTransitionIndex > this.Transitions.Count)
                    {
                        return false;
                    }

                    for (var i = state.FirstTransitionIndex; i < lastTransitionIndex; ++i)
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
                DeterminizationStateKnown = 0x4,
                IsDeterminized = 0x8,
                IsZeroStateKnown = 0x10,
                IsZero = 0x20,
            }
        }
    }
}
