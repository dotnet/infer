// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Represents a state of an automaton that is stored in the Automaton.statesData. This is an internal representation
        /// of the state. <see cref="State"/> struct should be used in public APIs.
        /// </summary>
        [Serializable]
        [DataContract]
        internal class StateData
        {
            /// <summary>
            /// The default capacity of the <see cref="transitions"/>.
            /// </summary>
            private const int DefaultTransitionArrayCapacity = 1;

            /// <summary>
            /// The array of outgoing transitions.
            /// </summary>
            /// <remarks>
            /// We don't use <see cref="List{T}"/> here for performance reasons.
            /// </remarks>
            [DataMember]
            private Transition[] transitions = new Transition[DefaultTransitionArrayCapacity];

            /// <summary>
            /// The number of outgoing transitions from the state.
            /// </summary>
            [DataMember]
            private int transitionCount;

            /// <summary>
            /// Initializes a new instance of the <see cref="StateData"/> class.
            /// </summary>
            public StateData() => this.EndWeight = Weight.Zero;

            /// <summary>
            /// Initializes a new instance of the <see cref="StateData"/> class.
            /// </summary>
            /// <param name="transitions">The outgoing transitions.</param>
            /// <param name="endWeight">The ending weight of the state.</param>
            [Construction("GetTransitions", "EndWeight")]
            public StateData(IEnumerable<Transition> transitions, Weight endWeight)
                : this()
            {
                Argument.CheckIfNotNull(transitions, "transitions");

                this.EndWeight = endWeight;

                foreach (var transition in transitions)
                {
                    this.AddTransition(transition);
                }
            }

            /// <summary>
            /// Gets or sets the ending weight of the state.
            /// </summary>
            [DataMember]
            public Weight EndWeight { get; set; }

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            public bool CanEnd => !this.EndWeight.IsZero;

            /// <summary>
            /// Gets the number of outgoing transitions.
            /// </summary>
            public int TransitionCount => this.transitionCount;

            /// <summary>
            /// Creates the copy of the array of outgoing transitions. Used by quoting.
            /// </summary>
            /// <returns>The copy of the array of outgoing transitions.</returns>
            public Transition[] GetTransitions()
            {
                var result = new Transition[this.transitionCount];
                Array.Copy(this.transitions, result, this.transitionCount);
                return result;
            }

            /// <summary>
            /// Adds a transition to the current state.
            /// </summary>
            /// <param name="transition">The transition to add.</param>
            /// <returns>The destination state of the added transition.</returns>
            public void AddTransition(Transition transition)
            {
                if (this.transitionCount == this.transitions.Length)
                {
                    var newTransitions = new Transition[this.transitionCount * 2];
                    Array.Copy(this.transitions, newTransitions, this.transitionCount);
                    this.transitions = newTransitions;
                }

                this.transitions[this.transitionCount++] = transition;
            }

            /// <summary>
            /// Gets the transition at a specified index.
            /// </summary>
            /// <param name="index">The index of the transition.</param>
            /// <returns>The transition.</returns>
            public Transition GetTransition(int index)
            {
                Debug.Assert(index >= 0 && index < this.transitionCount, nameof(index), "An invalid transition index given.");
                return this.transitions[index];
            }

            /// <summary>
            /// Replaces the transition at a given index with a given transition.
            /// </summary>
            /// <param name="index">The index of the transition to replace.</param>
            /// <param name="updatedTransition">The transition to replace with.</param>
            public void SetTransition(int index, Transition updatedTransition) =>
                this.transitions[index] = updatedTransition;

            /// <summary>
            /// Removes the transition with a given index.
            /// </summary>
            /// <param name="index">The index of the transition to remove.</param>
            public void RemoveTransition(int index)
            {
                Argument.CheckIfInRange(index >= 0 && index < this.transitionCount, "index", "An invalid transition index given.");
                this.transitions[index] = this.transitions[--this.transitionCount];
            }
        }
    }
}
