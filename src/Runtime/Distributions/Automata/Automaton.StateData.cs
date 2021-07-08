// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Serialization;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Represents a state of an automaton that is stored in the Automaton.states. This is an internal representation
        /// of the state. <see cref="State"/> struct should be used in public APIs.
        /// </summary>
        [Serializable]
        [DataContract]
        public readonly struct StateData
        {
            /// <summary>
            /// Gets or sets index of the first transition from this state in <see cref="DataContainer.Transitions"/>
            /// array. All transitions for the same state are stored as a contiguous block.
            /// </summary>
            /// <remarks>
            /// During automaton construction <see cref="Automaton{TSequence,TElement,TElementDistribution,TSequenceManipulator,TThis}.Builder"/>
            /// stores transitions as linked-list instead of contiguous block. So, during construction
            /// this property contains index of the head of the linked-list of transitions.
            /// </remarks>
            [DataMember]
            public int FirstTransitionIndex { get; }

            /// <summary>
            /// Gets or sets count of transition in <see cref="DataContainer.Transitions"/> after
            /// <see cref="FirstTransitionIndex"/> which belong to this state. All transitions for
            /// the same state are stored as a contiguous block.
            /// </summary>
            [DataMember]
            public int TransitionsCount { get; }

            /// <summary>
            /// Gets or sets ending weight of the state.
            /// </summary>
            [DataMember]
            public Weight EndWeight { get; }

            /// <summary>
            /// Initializes a new instance of the <see cref="StateData"/> struct.
            /// </summary>
            [Construction("FirstTransitionIndex", "TransitionsCount", "EndWeight")]
            public StateData(int firstTransitionIndex, int transitionsCount, Weight endWeight)
            {
                this.FirstTransitionIndex = firstTransitionIndex;
                this.TransitionsCount = transitionsCount;
                this.EndWeight = endWeight;
            }

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            internal bool CanEnd => !this.EndWeight.IsZero;
        }
    }
}
