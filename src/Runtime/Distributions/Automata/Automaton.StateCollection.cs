// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Collections;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Represents a collection of automaton states for use in public APIs
        /// </summary>
        /// <remarks>
        /// TODO xmldoc
        /// Is a thin wrapper around Automaton.data. Wraps each see cref="StateData"/> into <see cref="State"/> on demand.
        /// </remarks>
        public struct StateCollection : IReadOnlyList<State>
        {
            /// <summary>
            /// Cached value of this.owner.Data.states. Cached for performance.
            /// </summary>
            internal readonly ImmutableArray<RelativeState> states;

            /// <summary>
            /// Initializes instance of <see cref="StateCollection"/>.
            /// </summary>
            internal StateCollection(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> owner)
            {
                this.states = owner.Data.States;
            }

            #region IReadOnlyList<State> methods

            /// <inheritdoc/>
            public State this[int index] => new State(this.states[index], index);

            /// <inheritdoc/>
            public int Count => this.states.Count;

            /// <summary>
            /// Returns enumerator over all states.
            /// </summary>
            /// <remarks>
            /// This is value-type non-virtual version of enumerator that is used by compiler in foreach loops.
            /// </remarks>
            public StateEnumerator GetEnumerator() => new StateEnumerator(this);

            /// <inheritdoc/>
            IEnumerator<State> IEnumerable<State>.GetEnumerator() => new StateEnumerator(this);

            /// <inheritdoc/>
            IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();

            #endregion

            /// <summary>
            /// Enumerator over states in <see cref="StateCollection"/>.
            /// </summary>
            public struct StateEnumerator : IEnumerator<State>
            {
                /// <summary>
                /// Collection being enumerated.
                /// </summary>
                private readonly StateCollection collection;

                /// <summary>
                /// Index of current state.
                /// </summary>
                private int index;

                public StateEnumerator(StateCollection collection)
                {
                    this.collection = collection;
                    this.index = -1;
                }

                /// <inheritdoc/>
                public void Dispose()
                {
                }

                /// <inheritdoc/>
                public bool MoveNext()
                {
                    ++this.index;
                    return this.index < this.collection.Count;
                }

                /// <inheritdoc/>
                public State Current => this.collection[this.index];

                /// <inheritdoc/>
                object IEnumerator.Current => this.Current;

                /// <inheritdoc/>
                void IEnumerator.Reset()
                {
                    this.index = -1;
                }
            }
        }
    }
}
