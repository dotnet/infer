// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent a condensation of the automaton graph.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
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
            Argument.CheckIfNotNull(transitionFilter, nameof(transitionFilter));

            return new Condensation(this, root, transitionFilter, useApproximateClosure);
        }
        
        /// <summary>
        /// Represents the <a href="http://en.wikipedia.org/wiki/Condensation_(graph_theory)">condensation</a>
        /// of an automaton graph.
        /// </summary>
        public class Condensation
        {
            /// <summary>
            /// Automaton to which <see cref="Root"/> belongs.
            /// </summary>
            private readonly Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton;

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
            /// <param name="automaton">The automaton.</param>
            /// <param name="root">The root of the condensation DAG.</param>
            /// <param name="transitionFilter">
            /// A function specifying whether the transition should be treated as an edge
            /// of the automaton graph while building the condensation.
            /// </param>
            /// <param name="useApproximateClosure">
            /// Specifies whether <see cref="Weight.ApproximateClosure"/> should be used
            /// instead of <see cref="Weight.Closure"/> in semiring computations.
            /// </param>
            internal Condensation(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton,
                State root,
                Func<Transition, bool> transitionFilter, bool useApproximateClosure)
            {
                Debug.Assert(transitionFilter != null, "A valid transition filter must be provided.");

                this.automaton = automaton;
                this.Root = root;
                this.transitionFilter = transitionFilter;
                this.useApproximateClosure = useApproximateClosure;

                this.components = this.FindStronglyConnectedComponents();

                this.stateIdToInfo = new Dictionary<int, CondensationStateInfo>();
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
            public int ComponentCount => this.components.Count;

            public int TotalStatesCount { get; private set; }

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
            /// <param name="stateIndex">The state Index.</param>
            /// <returns>The computed total weight.</returns>
            public Weight GetWeightToEnd(int stateIndex)
            {
                if (!this.weightsToEndComputed)
                {
                    this.ComputeWeightsToEnd();
                }

                return
                    this.stateIdToInfo.TryGetValue(stateIndex, out var info)
                        ? info.WeightToEnd
                        : Weight.Zero;
            }

            /// <summary>
            /// Gets the total weight of all paths starting at the root of the condensation
            /// and ending in a given state. Ending weights are not taken into account.
            /// </summary>
            /// <param name="state">The state.</param>
            /// <returns>The computed total weight.</returns>
            public Weight GetWeightFromRoot(State state)
            {
                if (!this.weightsFromRootComputed)
                {
                    this.ComputeWeightsFromRoot();
                }

                if (!this.stateIdToInfo.TryGetValue(state.Index, out CondensationStateInfo info))
                {
                    return Weight.Zero;
                }

                return info.WeightFromRoot;
            }

            /// <summary>
            /// Implements <a href="http://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm">Tarjan's algorithm</a>
            /// for finding the strongly connected components of the automaton graph.
            /// </summary>
            private List<StronglyConnectedComponent> FindStronglyConnectedComponents()
            {
                var components = new List<StronglyConnectedComponent>();

                var states = this.automaton.States;
                var stateIdStack = new Stack<int>();
                var stateIdToStateInfo = new Dictionary<int, TarjanStateInfo>();
                int traversalIndex = 0;

                var traversalStack = new Stack<(TarjanStateInfo, int stateIndex, int currentTransitionIndex)>();

                traversalStack.Push((null, this.Root.Index, 0));
                while (traversalStack.Count > 0)
                {
                    var (stateInfo, currentStateIndex, currentTransitionIndex) = traversalStack.Pop();
                    var currentState = states[currentStateIndex];
                    var transitions = currentState.Transitions;

                    if (stateInfo == null)
                    {
                        // Entered into processing of this state for the first time: create Tarjan info for it
                        // and push the state onto Tarjan stack.
                        stateInfo = new TarjanStateInfo(traversalIndex);
                        stateIdToStateInfo.Add(currentStateIndex, stateInfo);
                        ++traversalIndex;
                        stateIdStack.Push(currentStateIndex);
                        stateInfo.InStack = true;
                    }
                    else
                    {
                        // Just returned from recursion into 'currentTransitionIndex': update Lowlink
                        // and advance to next transition.
                        var lastDestination = transitions[currentTransitionIndex].DestinationStateIndex;
                        stateInfo.Lowlink = Math.Min(stateInfo.Lowlink, stateIdToStateInfo[lastDestination].Lowlink);
                        ++currentTransitionIndex;
                    }

                    for (; currentTransitionIndex < transitions.Count; ++currentTransitionIndex)
                    {
                        var transition = transitions[currentTransitionIndex];
                        if (!this.transitionFilter(transition))
                        {
                            continue;
                        }

                        var destinationStateIndex = transition.DestinationStateIndex;
                        if (!stateIdToStateInfo.TryGetValue(destinationStateIndex, out var destinationStateInfo))
                        {
                            // Return to this (state/transition) after destinationState is processed.
                            // Processing will resume from currentTransitionIndex.
                            traversalStack.Push((stateInfo, currentStateIndex, currentTransitionIndex));
                            // Process destination state.
                            traversalStack.Push((null, destinationStateIndex, 0));
                            // Do not process other transitions until destination state is processed.
                            break;
                        }
                        
                        if (destinationStateInfo.InStack)
                        {
                            stateInfo.Lowlink = Math.Min(stateInfo.Lowlink, destinationStateInfo.TraversalIndex);
                        }
                    }

                    if (currentTransitionIndex < transitions.Count)
                    {
                        // Not all states processed: continue processing according to stack. Current state
                        // is already pushed onto it -> will be processed again.
                        continue;
                    }

                    if (stateInfo.Lowlink == stateInfo.TraversalIndex)
                    {
                        var statesInComponent = new List<State>();
                        int stateIndex;
                        do
                        {
                            stateIndex = stateIdStack.Pop();
                            stateIdToStateInfo[stateIndex].InStack = false;
                            statesInComponent.Add(states[stateIndex]);
                        } while (stateIndex != currentStateIndex);

                        components.Add(new StronglyConnectedComponent(
                            this.automaton, this.transitionFilter, statesInComponent, this.useApproximateClosure));
                        this.TotalStatesCount += statesInComponent.Count;
                    }
                }

                return components;
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
                        foreach (var transition in state.Transitions)
                        {
                            State destState = this.automaton.States[transition.DestinationStateIndex];
                            if (this.transitionFilter(transition) && !currentComponent.HasState(destState))
                            {
                                weightToAdd += transition.Weight * this.stateIdToInfo[transition.DestinationStateIndex].WeightToEnd;
                            }
                        }

                        // We can go from any state of the component to the current state
                        if (!weightToAdd.IsZero)
                        {
                            for (int updatedStateIndex = 0; updatedStateIndex < currentComponent.Size; ++updatedStateIndex)
                            {
                                State updatedState = currentComponent.GetStateByIndex(updatedStateIndex);
                                CondensationStateInfo updatedStateInfo = this.stateIdToInfo[updatedState.Index];
                                updatedStateInfo.WeightToEnd +=
                                    currentComponent.GetWeight(updatedStateIndex, stateIndex) * weightToAdd;
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
                            destStateInfo.WeightFromRoot +=
                                srcStateInfo.UpwardWeightFromRoot * currentComponent.GetWeight(srcStateIndex, destStateIndex);
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
                        foreach (var transition in srcState.Transitions)
                        {
                            State destState = this.automaton.States[transition.DestinationStateIndex];
                            if (this.transitionFilter(transition) && !currentComponent.HasState(destState))
                            {
                                CondensationStateInfo destStateInfo = this.stateIdToInfo[destState.Index];
                                destStateInfo.UpwardWeightFromRoot += srcStateInfo.WeightFromRoot * transition.Weight;
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
                public static CondensationStateInfo Default =>
                    new CondensationStateInfo
                    {
                        WeightToEnd = Weight.Zero,
                        WeightFromRoot = Weight.Zero,
                        UpwardWeightFromRoot = Weight.Zero
                    };

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
                public int TraversalIndex { get; }

                /// <summary>
                /// Gets or sets the lowlink of the state.
                /// </summary>
                public int Lowlink { get; set; }
            }
        }
    }
}
