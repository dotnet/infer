// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent the epsilon closure of a state of an automaton.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Represents the epsilon closure of a state.
        /// </summary>
        public class EpsilonClosure
        {
            /// <summary>
            /// The default capacity of <see cref="weightedStates"/>.
            /// </summary>
            private const int DefaultStateListCapacity = 5;

            /// <summary>
            /// The list of the states in the closure.
            /// </summary>
            private readonly List<Pair<State, Weight>> weightedStates = new List<Pair<State, Weight>>(DefaultStateListCapacity);

            /// <summary>
            /// Initializes a new instance of the <see cref="EpsilonClosure"/> class.
            /// </summary>
            /// <param name="state">The state, which epsilon closure this instance will represent.</param>
            internal EpsilonClosure(State state)
            {
                Argument.CheckIfValid(!state.IsNull, nameof(state));

                // Optimize for a very common case: a single-node closure
                bool singleNodeClosure = true;
                Weight selfLoopWeight = Weight.Zero;
                for (int i = 0; i < state.TransitionCount; ++i)
                {
                    Transition transition = state.GetTransition(i);
                    if (transition.IsEpsilon)
                    {
                        if (transition.DestinationStateIndex != state.Index)
                        {
                            singleNodeClosure = false;
                            break;
                        }

                        selfLoopWeight = Weight.Sum(selfLoopWeight, transition.Weight);
                    }
                }

                if (singleNodeClosure)
                {
                    Weight stateWeight = Weight.ApproximateClosure(selfLoopWeight);
                    this.weightedStates.Add(Pair.Create(state, stateWeight));
                    this.EndWeight = Weight.Product(stateWeight, state.EndWeight);
                }
                else
                {
                    Condensation condensation = state.Owner.ComputeCondensation(state, tr => tr.IsEpsilon, true);
                    for (int i = 0; i < condensation.ComponentCount; ++i)
                    {
                        StronglyConnectedComponent component = condensation.GetComponent(i);
                        for (int j = 0; j < component.Size; ++j)
                        {
                            State componentState = component.GetStateByIndex(j);
                            this.weightedStates.Add(Pair.Create(componentState, condensation.GetWeightFromRoot(componentState)));
                        }
                    }

                    this.EndWeight = condensation.GetWeightToEnd(state);
                }
            }

            /// <summary>
            /// Gets the total weight for ending inside the closure,
            /// if following epsilon transitions only
            /// </summary>
            public Weight EndWeight { get; private set; }

            /// <summary>
            /// Gets the number of states in the closure.
            /// </summary>
            public int Size
            {
                get { return this.weightedStates.Count; }
            }

            /// <summary>
            /// Gets a state by its index.
            /// </summary>
            /// <param name="index">The index. Must be non-negative and less than <see cref="Size"/>.</param>
            /// <returns>The state with the specified index.</returns>
            public State GetStateByIndex(int index)
            {
                Argument.CheckIfInRange(index >= 0 && index < this.weightedStates.Count, "index", "An invalid closure state index given.");

                return this.weightedStates[index].First;
            }

            /// <summary>
            /// Gets the total weight of all the paths from the root to the state with a given index,
            /// if following epsilon transitions only.
            /// </summary>
            /// <param name="index">The index. Must be non-negative and less than <see cref="Size"/>.</param>
            /// <returns>The weight.</returns>
            public Weight GetStateWeightByIndex(int index)
            {
                Argument.CheckIfInRange(index >= 0 && index < this.weightedStates.Count, "index", "An invalid closure state index given.");

                return this.weightedStates[index].Second;
            }
        }
    }
}
