// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent a condensation of the automaton graph.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Computes a condensation of the underlying automaton graph.
        /// </summary>
        /// <param name="root">The root of the condensation.</param>
        /// <returns>The computed condensation.</returns>
        public Condensation ComputeCondensation(State root)
        {
            return this.ComputeCondensation(root, transition => true, false);
        }

        /// <summary>
        /// Computes a condensation of the underlying automaton graph.
        /// </summary>
        /// <param name="root">The root of the condensation.</param>
        /// <param name="transitionFilter">
        /// A function specifying whether the transition should be treated as an edge
        /// of the automaton graph while building the condensation.
        /// </param>
        /// <param name="useApproximateClosure">
        /// Specifies whether <see cref="Weight.ApproximateClosure"/> should be used
        /// instead of <see cref="Weight.Closure"/> in semiring computations.
        /// </param>
        /// <returns>The computed condensation.</returns>
        public Condensation ComputeCondensation(State root, Func<Transition, bool> transitionFilter, bool useApproximateClosure)
        {
            Argument.CheckIfValid(!root.IsNull, nameof(root));
            Argument.CheckIfNotNull(transitionFilter, nameof(transitionFilter));
            Argument.CheckIfValid(ReferenceEquals(root.Owner, this), nameof(root), "The given node belongs to a different automaton.");

            return new Condensation(root, transitionFilter, useApproximateClosure);
        }
        
        /// <summary>
        /// Represents the <a href="http://en.wikipedia.org/wiki/Condensation_(graph_theory)">condensation</a>
        /// of an automaton graph.
        /// </summary>
        public class Condensation
        {
            /// <summary>
            /// A function specifying whether the transition should be treated as an edge
            /// of the automaton graph while building the condensation.
            /// </summary>
            private readonly Func<Transition, bool> transitionFilter;

            /// <summary>
            /// Specifies whether <see cref="Weight.ApproximateClosure"/> should be used
            /// instead of <see cref="Weight.Closure"/> in semiring computations.
            /// </summary>
            private readonly bool useApproximateClosure;

            /// <summary>
            /// The list of the strongly connected components of the condensation.
            /// Components are stored in the reverse topological order.
            /// </summary>
            private readonly IList<StronglyConnectedComponent> components;

            /// <summary>
            /// The dictionary containing information associated with every state of the condensation.
            /// </summary>
            private readonly Dictionary<int, CondensationStateInfo> stateIdToInfo;

            /// <summary>
            /// Specifies whether the total weights of all paths starting from the states of the component
            /// have been computed.
            /// </summary>
            private bool weightsToEndComputed;

            /// <summary>
            /// Specifies whether the total weights of all paths starting in the root and ending in the states of the component
            /// have been computed.
            /// </summary>
            private bool weightsFromRootComputed;

            /// <summary>
            /// Initializes a new instance of the <see cref="Condensation"/> class.
            /// </summary>
            /// <param name="root">The root of the condensation DAG.</param>
            /// <param name="transitionFilter">
            /// A function specifying whether the transition should be treated as an edge
            /// of the automaton graph while building the condensation.
            /// </param>
            /// <param name="useApproximateClosure">
            /// Specifies whether <see cref="Weight.ApproximateClosure"/> should be used
            /// instead of <see cref="Weight.Closure"/> in semiring computations.
            /// </param>
            internal Condensation(State root, Func<Transition, bool> transitionFilter, bool useApproximateClosure)
            {
                Debug.Assert(!root.IsNull, "A valid root node must be provided.");
                Debug.Assert(transitionFilter != null, "A valid transition filter must be provided.");
                
                this.Root = root;
                this.transitionFilter = transitionFilter;
                this.useApproximateClosure = useApproximateClosure;

                this.components = new List<StronglyConnectedComponent>();
                var stateIdStack = new Stack<State>();
                var stateIdToTarjanInfo = new Dictionary<int, TarjanStateInfo>();
                int traversalIndex = 0;
                this.FindStronglyConnectedComponents(this.Root, ref traversalIndex, stateIdToTarjanInfo, stateIdStack);

                this.stateIdToInfo = new Dictionary<int, CondensationStateInfo>(stateIdToTarjanInfo.Count);
                for (int i = 0; i < this.components.Count; ++i)
                {
                    StronglyConnectedComponent component = this.components[i];
                    for (int j = 0; j < component.Size; ++j)
                    {
                        this.stateIdToInfo.Add(component.GetStateByIndex(j).Index, CondensationStateInfo.Default);
                    }
                }
            }

            /// <summary>
            /// Gets the root of the condensation's DAG.
            /// </summary>
            public State Root { get; private set; }

            /// <summary>
            /// Gets the number of strongly connected components in the condensation.
            /// </summary>
            public int ComponentCount
            {
                get { return this.components.Count; }
            }

            /// <summary>
            /// Gets the strongly connected component by its index.
            /// Component indices are assigned in the reverse topological order (i.e. <see cref="Root"/> is in the last component).
            /// </summary>
            /// <param name="index">The index. Must be non-negative and less than <see cref="ComponentCount"/>.</param>
            /// <returns>The strongly connected component with the given index.</returns>
            public StronglyConnectedComponent GetComponent(int index)
            {
                return this.components[index];
            }

            /// <summary>
            /// Gets the total weight of all paths starting at a given state. 
            /// Ending weights are taken into account.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>The computed total weight.</returns>
            public Weight GetWeightToEnd(State state)
            {
                Argument.CheckIfValid(!state.IsNull, nameof(state));
                Argument.CheckIfValid(ReferenceEquals(state.Owner, this.Root.Owner), "state", "The given state belongs to a different automaton.");

                if (!this.weightsToEndComputed)
                {
                    this.ComputeWeightsToEnd();
                }

                CondensationStateInfo info;
                if (!this.stateIdToInfo.TryGetValue(state.Index, out info))
                {
                    return Weight.Zero;
                }

                return info.WeightToEnd;
            }

            /// <summary>
            /// Gets the total weight of all paths starting at the root of the condensation
            /// and ending in a given state. Ending weights are not taken into account.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>The computed total weight.</returns>
            public Weight GetWeightFromRoot(State state)
            {
                Argument.CheckIfValid(!state.IsNull, nameof(state));
                Argument.CheckIfValid(ReferenceEquals(state.Owner, this.Root.Owner), "state", "The given state belongs to a different automaton.");

                if (!this.weightsFromRootComputed)
                {
                    this.ComputeWeightsFromRoot();
                }

                CondensationStateInfo info;
                if (!this.stateIdToInfo.TryGetValue(state.Index, out info))
                {
                    return Weight.Zero;
                }

                return info.WeightFromRoot;
            }

            /// <summary>
            /// Implements <a href="http://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm">Tarjan's algorithm</a>
            /// for finding the strongly connected components of the automaton graph.
            /// </summary>
            /// <param name="currentState">The state currently being traversed.</param>
            /// <param name="traversalIndex">The traversal index (as defined by the Tarjan's algorithm).</param>
            /// <param name="stateIdToStateInfo">A dictionary mapping state indices to info records maintained by the Tarjan's algorithm.</param>
            /// <param name="stateIdStack">The traversal stack (as defined by the Tarjan's algorithm).</param>
            private void FindStronglyConnectedComponents(
                State currentState,
                ref int traversalIndex,
                Dictionary<int, TarjanStateInfo> stateIdToStateInfo,
                Stack<State> stateIdStack)
            {
                Debug.Assert(!stateIdToStateInfo.ContainsKey(currentState.Index), "Visited states must not be revisited.");

                var stateInfo = new TarjanStateInfo(traversalIndex);
                stateIdToStateInfo.Add(currentState.Index, stateInfo);
                ++traversalIndex;

                stateIdStack.Push(currentState);
                stateInfo.InStack = true;

                for (int transitionIndex = 0; transitionIndex < currentState.TransitionCount; ++transitionIndex)
                {
                    Transition transition = currentState.GetTransition(transitionIndex);
                    if (!this.transitionFilter(transition))
                    {
                        continue;
                    }

                    TarjanStateInfo destinationStateInfo;
                    if (!stateIdToStateInfo.TryGetValue(transition.DestinationStateIndex, out destinationStateInfo))
                    {
                        this.FindStronglyConnectedComponents(
                            this.Root.Owner.States[transition.DestinationStateIndex], ref traversalIndex, stateIdToStateInfo, stateIdStack);
                        stateInfo.Lowlink = Math.Min(stateInfo.Lowlink, stateIdToStateInfo[transition.DestinationStateIndex].Lowlink);
                    }
                    else if (destinationStateInfo.InStack)
                    {
                        stateInfo.Lowlink = Math.Min(stateInfo.Lowlink, destinationStateInfo.TraversalIndex);
                    }
                }

                if (stateInfo.Lowlink == stateInfo.TraversalIndex)
                {
                    var statesInComponent = new List<State>();
                    State state;
                    do
                    {
                        state = stateIdStack.Pop();
                        stateIdToStateInfo[state.Index].InStack = false;
                        statesInComponent.Add(state);
                    }
                    while (state.Index != currentState.Index);

                    this.components.Add(new StronglyConnectedComponent(this.transitionFilter, statesInComponent, this.useApproximateClosure));
                }
            }

            /// <summary>
            /// For each state of the component, computes the total weight of all paths starting at that state.
            /// Ending weights are taken into account.
            /// </summary>
            /// <remarks>The weights are computed using dynamic programming, going up from leafs to the root.</remarks>
            private void ComputeWeightsToEnd()
            {
                // Iterate in the reverse topological order
                for (int currentComponentIndex = 0; currentComponentIndex < this.components.Count; ++currentComponentIndex)
                {
                    StronglyConnectedComponent currentComponent = this.components[currentComponentIndex];

                    // Update end weights in this component based on outgoing transitions to downward components
                    for (int stateIndex = 0; stateIndex < currentComponent.Size; ++stateIndex)
                    {
                        State state = currentComponent.GetStateByIndex(stateIndex);

                        // Aggregate weights of all the outgoing transitions from this state
                        Weight weightToAdd = state.EndWeight;
                        for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                        {
                            Transition transition = state.GetTransition(transitionIndex);
                            State destState = state.Owner.States[transition.DestinationStateIndex];
                            if (this.transitionFilter(transition) && !currentComponent.HasState(destState))
                            {
                                weightToAdd = Weight.Sum(
                                    weightToAdd,
                                    Weight.Product(transition.Weight, this.stateIdToInfo[transition.DestinationStateIndex].WeightToEnd));
                            }
                        }

                        // We can go from any state of the component to the current state
                        if (!weightToAdd.IsZero)
                        {
                            for (int updatedStateIndex = 0; updatedStateIndex < currentComponent.Size; ++updatedStateIndex)
                            {
                                State updatedState = currentComponent.GetStateByIndex(updatedStateIndex);
                                CondensationStateInfo updatedStateInfo = this.stateIdToInfo[updatedState.Index];
                                updatedStateInfo.WeightToEnd = Weight.Sum(
                                    updatedStateInfo.WeightToEnd,
                                    Weight.Product(currentComponent.GetWeight(updatedStateIndex, stateIndex), weightToAdd));
                                this.stateIdToInfo[updatedState.Index] = updatedStateInfo;
                            }
                        }
                    }
                }

                this.weightsToEndComputed = true;
            }

            /// <summary>
            /// For each state of the component, computes the total weight of all paths starting at the root
            /// and ending at that state. Ending weights are not taken into account.
            /// </summary>
            /// <remarks>The weights are computed using dynamic programming, going down from the root to leafs.</remarks>
            private void ComputeWeightsFromRoot()
            {
                CondensationStateInfo rootInfo = this.stateIdToInfo[this.Root.Index];
                rootInfo.UpwardWeightFromRoot = Weight.One;
                this.stateIdToInfo[this.Root.Index] = rootInfo;

                // Iterate in the topological order
                for (int currentComponentIndex = this.components.Count - 1; currentComponentIndex >= 0; --currentComponentIndex)
                {
                    StronglyConnectedComponent currentComponent = this.components[currentComponentIndex];

                    // Propagate weights inside the component
                    for (int srcStateIndex = 0; srcStateIndex < currentComponent.Size; ++srcStateIndex)
                    {
                        State srcState = currentComponent.GetStateByIndex(srcStateIndex);
                        CondensationStateInfo srcStateInfo = this.stateIdToInfo[srcState.Index];
                        if (srcStateInfo.UpwardWeightFromRoot.IsZero)
                        {
                            continue;
                        }

                        for (int destStateIndex = 0; destStateIndex < currentComponent.Size; ++destStateIndex)
                        {
                            State destState = currentComponent.GetStateByIndex(destStateIndex);
                            CondensationStateInfo destStateInfo = this.stateIdToInfo[destState.Index];
                            destStateInfo.WeightFromRoot = Weight.Sum(
                                destStateInfo.WeightFromRoot,
                                Weight.Product(srcStateInfo.UpwardWeightFromRoot, currentComponent.GetWeight(srcStateIndex, destStateIndex)));
                            this.stateIdToInfo[destState.Index] = destStateInfo;
                        }
                    }

                    // Compute weight contributions to downward components
                    for (int srcStateIndex = 0; srcStateIndex < currentComponent.Size; ++srcStateIndex)
                    {
                        State srcState = currentComponent.GetStateByIndex(srcStateIndex);
                        CondensationStateInfo srcStateInfo = this.stateIdToInfo[srcState.Index];
                        if (srcStateInfo.WeightFromRoot.IsZero)
                        {
                            continue;
                        }

                        // Aggregate weights of all the outgoing transitions from this state
                        for (int transitionIndex = 0; transitionIndex < srcState.TransitionCount; ++transitionIndex)
                        {
                            Transition transition = srcState.GetTransition(transitionIndex);
                            State destState = srcState.Owner.States[transition.DestinationStateIndex];
                            if (this.transitionFilter(transition) && !currentComponent.HasState(destState))
                            {
                                CondensationStateInfo destStateInfo = this.stateIdToInfo[destState.Index];
                                destStateInfo.UpwardWeightFromRoot = Weight.Sum(
                                    destStateInfo.UpwardWeightFromRoot,
                                    Weight.Product(srcStateInfo.WeightFromRoot, transition.Weight));
                                this.stateIdToInfo[transition.DestinationStateIndex] = destStateInfo;
                            }
                        }
                    }
                }

                this.weightsFromRootComputed = true;
            }

            /// <summary>
            /// Stores the information associated with a state of the condensation.
            /// </summary>
            private struct CondensationStateInfo
            {
                /// <summary>
                /// Gets the default state info with all the weights set to 0.
                /// </summary>
                public static CondensationStateInfo Default
                {
                    get
                    {
                        return new CondensationStateInfo
                        {
                            WeightToEnd = Weight.Zero,
                            WeightFromRoot = Weight.Zero,
                            UpwardWeightFromRoot = Weight.Zero
                        };
                    }
                }

                /// <summary>
                /// Gets or sets the total weight of all paths starting at the state.
                /// Ending weights are taken into account.
                /// </summary>
                public Weight WeightToEnd { get; set; }

                /// <summary>
                /// Gets or sets the total weight of all paths from the root of the condensation
                /// to the state. Ending weights are not taken into account.
                /// </summary>
                public Weight WeightFromRoot { get; set; }

                /// <summary>
                /// Gets or sets the total weight of all paths from the root of the condensation
                /// to the state that don't go through any other states of the same strongly connected component.
                /// Ending weights are not taken into account.
                /// </summary>
                public Weight UpwardWeightFromRoot { get; set; }
            }

            /// <summary>
            /// Stores the information maintained by the Tarjan's algorithm.
            /// </summary>
            private class TarjanStateInfo
            {
                /// <summary>
                /// Initializes a new instance of the <see cref="TarjanStateInfo"/> class.
                /// </summary>
                /// <param name="traversalIndex">The current traversal index.</param>
                public TarjanStateInfo(int traversalIndex)
                {
                    this.TraversalIndex = traversalIndex;
                    this.Lowlink = traversalIndex;
                }

                /// <summary>
                /// Gets or sets a value indicating whether the state is currently in stack.
                /// </summary>
                public bool InStack { get; set; }

                /// <summary>
                /// Gets the traversal index of the state.
                /// </summary>
                public int TraversalIndex { get; private set; }

                /// <summary>
                /// Gets or sets the lowlink of the state.
                /// </summary>
                public int Lowlink { get; set; }
            }
        }
    }
}
