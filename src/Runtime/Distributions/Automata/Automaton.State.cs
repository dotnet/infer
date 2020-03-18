// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Text;


    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Represents a reference to a state of automaton for exposure in public API.
        /// </summary>
        public struct State
        {
            /// <summary>
            /// Initializes a new instance of <see cref="State"/> class. Used internally by automaton implementation
            /// to wrap StateData for use in public Automaton APIs.
            /// </summary>
            internal State(RelativeState data, int index)
            {
                this.Data = data;
                this.Index = index;
            }

            /// <summary>
            /// Gets the index of the state.
            /// </summary>
            public int Index { get; }

            internal RelativeState Data { get; }

            /// <summary>
            /// Gets the ending weight of the state.
            /// </summary>
            public Weight EndWeight => this.Data.EndWeight;

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            public bool CanEnd => this.Data.CanEnd;

            public TransitionsList Transitions =>
                new TransitionsList(
                    this.Index,
                    this.Data.RelativeTransitions.BaseArray,
                    this.Data.RelativeTransitions.BaseIndex,
                    this.Data.RelativeTransitions.Count);

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
                writeInt32(this.Transitions.Count);
                foreach (var transition in this.Transitions)
                {
                    transition.Write(writeInt32, writeDouble, writeElementDistribution);
                }
            }

            /// <summary>
            /// Reads state and appends it into Automaton builder.
            /// </summary>
            public static void ReadTo(
                ref Builder builder,
                Func<int> readInt32,
                Func<double> readDouble,
                Func<TElementDistribution> readElementDistribution)
            {
                var endWeight = Weight.Read(readDouble);

                var state = builder.AddState();
                state.SetEndWeight(endWeight);

                var transitionCount = readInt32();
                for (var i = 0; i < transitionCount; i++)
                {
                    state.AddTransition(Transition.Read(readInt32, readDouble, readElementDistribution));
                }
            }

            #endregion
        }
    }
}
