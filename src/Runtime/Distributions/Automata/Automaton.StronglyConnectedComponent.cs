// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent the condensation of an automaton graph.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Represents a strongly connected component of an automaton graph.
        /// </summary>
        public class StronglyConnectedComponent
        {
            /// <summary>
            /// The transition filter used to build the condensation this component belongs to.
            /// </summary>
            private readonly Func<Transition, bool> transitionFilter;

            /// <summary>
            /// The list of states in the component.
            /// </summary>
            private readonly List<State> statesInComponent;

            /// <summary>
            /// Specifies whether <see cref="Weight.ApproximateClosure"/> should be used
            /// instead of <see cref="Weight.Closure"/> in semiring computations.
            /// </summary>
            private readonly bool useApproximateClosure;

            /// <summary>
            /// The mappings from state indices to indices in the <see cref="statesInComponent"/> list.
            /// </summary>
            private Dictionary<int, int> stateIdToIndexInComponent;

            /// <summary>
            /// The table of total weights between pairs of component nodes.
            /// </summary>
            private Weight[,] pairwiseWeights;

            /// <summary>
            /// An optimized version of <see cref="pairwiseWeights"/> in case the component consists of a single state.
            /// </summary>
            private Weight? singleStatePairwiseWeight;

            /// <summary>
            /// Initializes a new instance of the <see cref="StronglyConnectedComponent"/> class.
            /// </summary>
            /// <param name="transitionFilter">The transition filter used to build the condensation this component belongs to.</param>
            /// <param name="statesInComponent">The list of states in the component.</param>
            /// <param name="useApproximateClosure">
            /// Specifies whether <see cref="Weight.ApproximateClosure"/> should be used
            /// instead of <see cref="Weight.Closure"/> in semiring computations.
            /// </param>
            internal StronglyConnectedComponent(
                Func<Transition, bool> transitionFilter,
                List<State> statesInComponent,
                bool useApproximateClosure)
            {
                Debug.Assert(
                    statesInComponent.Count > 0,
                    "There must be at least one state in the strongly connected component.");
                Debug.Assert(
                    statesInComponent.All(s => s != null && ReferenceEquals(s.Owner, statesInComponent[0].Owner)),
                    "All the states must be valid and belong to the same automaton.");

                this.transitionFilter = transitionFilter;
                this.statesInComponent = statesInComponent;
                this.useApproximateClosure = useApproximateClosure;
            }

            /// <summary>
            /// Gets the number of states in the component.
            /// </summary>
            public int Size
            {
                get { return this.statesInComponent.Count; }
            }

            /// <summary>
            /// Gets the state by its index in the component.
            /// </summary>
            /// <param name="indexInComponent">The index of the state in the component. Must be non-negative and less than <see cref="Size"/>.</param>
            /// <returns>The state corresponding to the specified index.</returns>
            public State GetStateByIndex(int indexInComponent)
            {
                Argument.CheckIfInRange(indexInComponent >= 0 && indexInComponent < this.Size, "index", "The given index is out of range.");

                return this.statesInComponent[indexInComponent];
            }

            /// <summary>
            /// Checks whether the given state belongs to this component.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>
            /// <see langword="true"/> if <paramref name="state"/> belongs to the component,
            /// <see langword="false"/> otherwise.
            /// </returns>
            public bool HasState(State state)
            {
                Argument.CheckIfValid(!state.IsNull, nameof(state));

                return this.GetIndexByState(state) != -1;
            }

            /// <summary>
            /// Attempts to retrieve the index of a given state in the component.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>
            /// The index of <paramref name="state"/> in the component, or -1 if it does not belong to the component.
            /// </returns>
            public int GetIndexByState(State state)
            {
                Argument.CheckIfValid(!state.IsNull, nameof(state));
                Argument.CheckIfValid(ReferenceEquals(state.Owner, this.statesInComponent[0].Owner), "state", "The given state belongs to other automaton.");

                if (this.statesInComponent.Count == 1)
                {
                    return this.statesInComponent[0].Index == state.Index ? 0 : -1;
                }

                if (this.stateIdToIndexInComponent == null)
                {
                    this.stateIdToIndexInComponent = new Dictionary<int, int>(this.statesInComponent.Count);
                    for (int i = 0; i < this.statesInComponent.Count; ++i)
                    {
                        this.stateIdToIndexInComponent.Add(this.statesInComponent[i].Index, i);
                    }
                }

                int stateIndex;
                return this.stateIdToIndexInComponent.TryGetValue(state.Index, out stateIndex) ? stateIndex : -1;
            }

            /// <summary>
            /// Gets the total weight between two given states in the component.
            /// </summary>
            /// <param name="srcStateIndexInComponent">The index of the source state in the component.</param>
            /// <param name="destStateIndexInComponent">The index of the destination state in the component.</param>
            /// <returns>The total weight between the given states in the component.</returns>
            public Weight GetWeight(int srcStateIndexInComponent, int destStateIndexInComponent)
            {
                Argument.CheckIfInRange(
                    srcStateIndexInComponent >= 0 && srcStateIndexInComponent < this.Size,
                    "srcStateIndexInComponent",
                    "The given index is out of range.");
                Argument.CheckIfInRange(
                    destStateIndexInComponent >= 0 && destStateIndexInComponent < this.Size,
                    "destStateIndexInComponent",
                    "The given index is out of range.");

                if (this.Size == 1)
                {
                    if (!this.singleStatePairwiseWeight.HasValue)
                    {
                        // Optimize for a common case
                        State state = this.statesInComponent[0];
                        this.singleStatePairwiseWeight = Weight.Zero;
                        for (int i = 0; i < state.TransitionCount; ++i)
                        {
                            Transition transition = state.GetTransition(i);
                            if (this.transitionFilter(transition) && transition.DestinationStateIndex == state.Index)
                            {
                                this.singleStatePairwiseWeight = Weight.Sum(
                                    this.singleStatePairwiseWeight.Value, transition.Weight);
                            }
                        }

                        this.singleStatePairwiseWeight =
                            this.useApproximateClosure
                                ? Weight.ApproximateClosure(this.singleStatePairwiseWeight.Value)
                                : Weight.Closure(this.singleStatePairwiseWeight.Value);
                    }

                    return this.singleStatePairwiseWeight.Value;
                }

                if (this.pairwiseWeights == null)
                {
                    this.ComputePairwiseWeightsMatrix();
                }

                return this.pairwiseWeights[srcStateIndexInComponent, destStateIndexInComponent];
            }

            /// <summary>
            /// Computes the total weights between each pair of states in the component
            /// using the <a href="http://www.cs.nyu.edu/~mohri/pub/hwa.pdf">generalized Floyd's algorithm</a>.
            /// </summary>
            private void ComputePairwiseWeightsMatrix()
            {
                this.pairwiseWeights = Util.ArrayInit(this.Size, this.Size, (i, j) => Weight.Zero);
                for (int srcStateIndexInComponent = 0; srcStateIndexInComponent < this.Size; ++srcStateIndexInComponent)
                {
                    State state = this.statesInComponent[srcStateIndexInComponent];
                    for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                    {
                        Transition transition = state.GetTransition(transitionIndex);
                        State destState = state.Owner.States[transition.DestinationStateIndex];
                        int destStateIndexInComponent;
                        if (this.transitionFilter(transition) && (destStateIndexInComponent = this.GetIndexByState(destState)) != -1)
                        {
                            this.pairwiseWeights[srcStateIndexInComponent, destStateIndexInComponent] = Weight.Sum(
                                this.pairwiseWeights[srcStateIndexInComponent, destStateIndexInComponent], transition.Weight);
                        }
                    }
                }

                for (int k = 0; k < this.Size; ++k)
                {
                    Weight loopWeight =
                        this.useApproximateClosure ? Weight.ApproximateClosure(this.pairwiseWeights[k, k]) : Weight.Closure(this.pairwiseWeights[k, k]);
                    
                    for (int i = 0; i < this.Size; ++i)
                    {
                        if (i == k || this.pairwiseWeights[i, k].IsZero)
                        {
                            continue;
                        }

                        for (int j = 0; j < this.Size; ++j)
                        {
                            if (j == k || this.pairwiseWeights[k, j].IsZero)
                            {
                                continue;
                            }

                            Weight additionalWeight = Weight.Product(
                                this.pairwiseWeights[i, k], loopWeight, this.pairwiseWeights[k, j]);
                            this.pairwiseWeights[i, j] = Weight.Sum(this.pairwiseWeights[i, j], additionalWeight);
                        }
                    }

                    for (int i = 0; i < this.Size; ++i)
                    {
                        this.pairwiseWeights[i, k] = Weight.Product(this.pairwiseWeights[i, k], loopWeight);
                        this.pairwiseWeights[k, i] = Weight.Product(this.pairwiseWeights[k, i], loopWeight);
                    }

                    this.pairwiseWeights[k, k] = loopWeight;
                }
            }
        }
    }
}
