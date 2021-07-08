// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Represents a reference to a state of automaton for exposure in public API.
        /// </summary>
        /// <remarks>
        /// Acts as a "fat reference" to state in automaton. In addition to reference to actual
        /// StateData it carries 2 additional properties for convenience: <see cref="Index"/>
        /// of the state and full <see cref="transitions"/> table.
        /// </remarks>
        public readonly struct State : IEquatable<State>
        {
            private readonly ReadOnlyArray<StateData> states;

            private readonly ReadOnlyArray<Transition> transitions;

            /// <summary>
            /// Initializes a new instance of <see cref="State"/> class. Used internally by automaton implementation
            /// to wrap StateData for use in public Automaton APIs.
            /// </summary>
            internal State(
                ReadOnlyArray<StateData> states,
                ReadOnlyArray<Transition> transitions,
                int index)
            {
                this.states = states;
                this.transitions = transitions;
                this.Index = index;
            }

            /// <summary>
            /// Gets the index of the state.
            /// </summary>
            public int Index { get; }

            /// <summary>
            /// Gets the ending weight of the state.
            /// </summary>
            public Weight EndWeight => this.Data.EndWeight;

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            public bool CanEnd => this.Data.CanEnd;

            public ReadOnlyArraySegment<Transition> Transitions =>
                new ReadOnlyArraySegment<Transition>(
                    this.transitions,
                    this.Data.FirstTransitionIndex,
                    this.Data.TransitionsCount);

            internal StateData Data => this.states[this.Index];

            /// <summary>
            /// Compares 2 states for equality.
            /// </summary>
            public static bool operator ==(State a, State b) => a.Index == b.Index;

            /// <summary>
            /// Compares 2 states for inequality.
            /// </summary>
            public static bool operator !=(State a, State b) => !(a == b);

            /// <summary>
            /// Compares 2 states for equality.
            /// </summary>
            public bool Equals(State that) => this == that;

            /// <inheritdoc/>
            public override bool Equals(object obj) => obj is State that && this.Equals(that);

            /// <inheritdoc/>
            public override int GetHashCode() => this.Data.GetHashCode();

            /// <summary>
            /// Returns a string that represents the state.
            /// </summary>
            /// <returns>A string that represents the state.</returns>
            public override string ToString()
            {
                const string TransitionSeparator = ",";

                var sb = new StringBuilder();

                var firstTransition = true;
                foreach (var transition in this.Transitions)
                {
                    if (firstTransition)
                    {
                        firstTransition = false;
                    }
                    else
                    {
                        sb.Append(TransitionSeparator);
                    }

                    sb.Append(transition.ToString());
                }

                if (this.CanEnd)
                {
                    if (!firstTransition)
                    {
                        sb.Append(TransitionSeparator);
                    }

                    sb.Append(this.EndWeight.Value + " -> END");
                }

                return sb.ToString();
            }

            #region Serialization

            public void Write(Action<double> writeDouble, Action<int> writeInt32, Action<TElementDistribution> writeElementDistribution)
            {
                this.EndWeight.Write(writeDouble);
                writeInt32(this.Index);
                writeInt32(this.Transitions.Count);
                foreach (var transition in this.Transitions)
                {
                    transition.Write(writeInt32, writeDouble, writeElementDistribution);
                }
            }

            /// <summary>
            /// Reads state and appends it into Automaton builder. Returns index in the serialized data.
            /// If <paramref name="checkIndex"/> is true, will throw exception if serialized index
            /// does not match index in deserialized states array. This check is bypassed only when
            /// start state is serialized second time.
            /// </summary>
            public static int ReadTo(
                ref Builder builder,
                Func<int> readInt32,
                Func<double> readDouble,
                Func<TElementDistribution> readElementDistribution,
                bool checkIndex = false)
            {
                var endWeight = Weight.Read(readDouble);
                // Note: index is serialized for compatibility with old binary serializations
                var index = readInt32();

                if (checkIndex && index != builder.StatesCount)
                {
                    throw new Exception("Index in serialized data does not match index in deserialized array");
                }

                var state = builder.AddState();
                state.SetEndWeight(endWeight);

                var transitionCount = readInt32();
                for (var i = 0; i < transitionCount; i++)
                {
                    state.AddTransition(Transition.Read(readInt32, readDouble, readElementDistribution));
                }

                return index;
            }

            #endregion
        }
    }
}
