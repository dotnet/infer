// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Represents a collection of automaton states for use in public APIs
        /// </summary>
        /// <remarks>
        /// Is a thin wrapper around Automaton.stateData. Wraps each <see cref="StateData"/> into <see cref="State"/> on demand.
        /// </remarks>
        public struct StateCollection : IReadOnlyList<State>
        {
            /// <summary>
            /// Owner automaton of all states in collection.
            /// </summary>
            private readonly Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> owner;

            /// <summary>
            /// Cached value of owner.statesData. Cached for performance reasons.
            /// </summary>
            private readonly List<StateData> statesData;

            /// <summary>
            /// Initializes instance of <see cref="StateCollection"/>.
            /// </summary>
            internal StateCollection(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> owner, List<StateData> states)
            {
                this.owner = owner;
                this.statesData = owner.statesData;
            }

            /// <summary>
            /// Gets state by its index.
            /// </summary>
            public State this[int index] => new State(this.owner, index, this.statesData[index]);

            /// <summary>
            /// Gets number of states in collection.
            /// </summary>
            public int Count => this.statesData.Count;

            /// <summary>
            /// Returns enumerator over all states in collection.
            /// </summary>
            public IEnumerator<State> GetEnumerator()
            {
                var owner = this.owner;
                return this.statesData.Select((data, index) => new State(owner, index, data)).GetEnumerator();
            }

            /// <summary>
            /// Returns enumerator over all states in collection.
            /// </summary>
            IEnumerator IEnumerable.GetEnumerator()
            {
                return this.GetEnumerator();
            }
        }
    }
}
