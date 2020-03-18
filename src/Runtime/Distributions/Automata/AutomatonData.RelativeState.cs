// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Serialization;

    public abstract partial class AutomatonData<TElement, TElementDistribution>
    {
        /// <summary>
        /// Represents a state of an automaton that is stored in the Automaton.states. This is an internal representation
        /// of the state. TODO
        /// </summary>
        [Serializable]
        [DataContract]
        public class RelativeState
        {
            [DataMember]
            public ImmutableArraySegment<Transition> RelativeTransitions { get; }

            /// <summary>
            /// Gets or sets ending weight of the state.
            /// </summary>
            [DataMember]
            public Weight EndWeight { get; }

            /// <summary>
            /// Initializes a new instance of the <see cref="RelativeState"/> struct.
            /// </summary>
            [Construction("RelativeTransitions", "EndWeight")]
            public RelativeState(ImmutableArraySegment<Transition> relativeTransitions, Weight endWeight)
            {
                this.RelativeTransitions = relativeTransitions;
                this.EndWeight = endWeight;
            }

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            internal bool CanEnd => !this.EndWeight.IsZero;
        }
    }
}
