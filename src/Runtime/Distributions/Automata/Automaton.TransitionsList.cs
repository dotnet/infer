// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Collections;

    /// <content>
    /// TODO
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        public struct TransitionsList : IReadOnlyList<Transition>
        {
            private readonly int baseStateIndex;
            private readonly ImmutableArray<Transition> relativeTransitions;
            private readonly int firstTransitionIndex;
            private readonly int transitionsCount;

            public TransitionsList(
                int baseStateIndex,
                ImmutableArray<Transition> relativeTransitions,
                int firstTransitionIndex,
                int transitionsCount)
            {
                this.baseStateIndex = baseStateIndex;
                this.relativeTransitions = relativeTransitions;
                this.firstTransitionIndex = firstTransitionIndex;
                this.transitionsCount = transitionsCount;
            }

            /// <inheritdoc/>
            public Transition this[int index]
            {
                get
                {
                    var result = this.relativeTransitions[this.firstTransitionIndex + index];
                    result.DestinationStateIndex += this.baseStateIndex;
                    return result;
                }
            }

            /// <inheritdoc/>
            public int Count => this.transitionsCount;

            public TransitionsEnumerator GetEnumerator() =>
                new TransitionsEnumerator(this);

            /// <inheritdoc/>
            IEnumerator<Transition> IEnumerable<Transition>.GetEnumerator() =>
                this.GetEnumerator();

            /// <inheritdoc/>
            IEnumerator IEnumerable.GetEnumerator() =>
                this.GetEnumerator();

            public struct TransitionsEnumerator : IEnumerator<Transition>
            {
                private readonly int baseStateIndex;
                private readonly ImmutableArray<Transition> relativeTransitions;
                private readonly int endIndex;
                private int pointer;

                /// <summary>
                /// Initializes a new instance of <see cref="ImmutableArraySegment{T}"/> structure.
                /// </summary>
                internal TransitionsEnumerator(TransitionsList list)
                {
                    this.baseStateIndex = list.baseStateIndex;
                    this.relativeTransitions = list.relativeTransitions;
                    this.endIndex = list.firstTransitionIndex + list.transitionsCount;
                    this.pointer = list.firstTransitionIndex - 1;
                }

                /// <inheritdoc/>
                public void Dispose()
                {
                }

                /// <inheritdoc/>
                public bool MoveNext() => ++this.pointer < this.endIndex;

                /// <inheritdoc/>
                public Transition Current
                {
                    get
                    {
                        var result = this.relativeTransitions[this.pointer];
                        result.DestinationStateIndex += this.baseStateIndex;
                        return result;
                    }
                }

                /// <inheritdoc/>
                object IEnumerator.Current => this.Current;

                /// <inheritdoc/>
                void IEnumerator.Reset() => throw new NotSupportedException();

                // TODO: rethink!
                public bool IsLast => this.pointer >= this.endIndex - 1;
            }
        }
    }
}
