﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Utilities;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Helper class which is used to construct new automaton interactively using the builder pattern.
        /// </summary>
        public class Builder
        {
            /// <summary>
            /// Containers which are reused between builders
            /// </summary>
            /// <remarks>
            /// StringDistribution operations create enormous amounts of automata objects.
            /// Reusing containers allows to avoid a lot of intermediate allocations.
            ///
            /// See also <see cref="PreallocatedAutomataObjects"/> class and comments on it.
            /// This field can't go into that class because it depends on generic parameters.
            /// </remarks>
            [ThreadStatic]
            private static (List<LinkedStateData>, List<LinkedTransitionNode>) preallocatedContainers;

            /// <summary>
            /// States created so far.
            /// </summary>
            private List<LinkedStateData> states;

            /// <summary>
            /// Transitions created so far.
            /// </summary>
            /// <remarks>
            /// Unlike in <see cref="StateCollection.transitions"/>, transitions
            /// for single entity are not represented by contiguous segment of array, but rather as a linked
            /// list. It is done this way, because transitions can be added at any moment and inserting
            /// transition into a middle of array is not feasible.
            /// </remarks>
            private List<LinkedTransitionNode> transitions;

            /// <summary>
            /// Number of transitions marked as removed. We maintain this count to calculate finaly transitions
            /// array size without need to traverse all transitions.
            /// </summary>
            private int numRemovedTransitions = 0;

            /// <summary>
            /// Cached value of <see cref="MaxStateCount"/>. Getting MaxStateCount involves checking
            /// thread static variable value and a comparison. Caching this property values is a little
            /// faster.
            /// </summary>
            private readonly int maxStateCount;

            /// <summary>
            /// Creates a new empty <see cref="Builder"/>.
            /// </summary>
            public Builder(int startStateCount = 1)
            {
                var preallocated = preallocatedContainers;

                this.states = preallocated.Item1 ?? new List<LinkedStateData>();
                this.transitions = preallocated.Item2 ?? new List<LinkedTransitionNode>();


                // Generally only one builder is used at once.
                // In case two builders are created, ensure that another builder won't use
                // the same containers.
                preallocatedContainers = default;

                this.maxStateCount = MaxStateCount;
                this.AddStates(startStateCount);
            }

            #region Properties

            /// <summary>
            /// Index of the start state.
            /// </summary>
            public int StartStateIndex { get; set; }

            /// <summary>
            /// Number of states which were created so far.
            /// </summary>
            public int StatesCount => this.states.Count;

            /// <summary>
            /// Number of transitions which were created so far.
            /// </summary>
            public int TransitionsCount => this.transitions.Count - this.numRemovedTransitions;

            /// <summary>
            /// Returns <see cref="StateBuilder"/> object for specified state.
            /// </summary>
            public StateBuilder this[int index] => new StateBuilder(this, index);

            /// <summary>
            /// Returns <see cref="StateBuilder"/> object for start state.
            /// </summary>
            public StateBuilder Start => this[this.StartStateIndex];

            #endregion

            # region Factory methods

            /// <summary>
            /// Factory method which creates <see cref="Builder"/> which contains all states and transitions
            /// from given <paramref name="automaton"/>.
            /// </summary>
            public static Builder FromAutomaton(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton)
            {
                var result = new Builder(0);
                result.AddStates(automaton.States);
                result.StartStateIndex = automaton.Start.Index;
                return result;
            }

            /// <summary>
            /// Factory method which creates <see cref="Builder"/> and inserts into it an automaton
            /// matching given <paramref name="sequence"/> with some <paramref name="weight"/>.
            /// </summary>
            public static Builder ConstantOn(Weight weight, TSequence sequence)
            {
                var result = new Builder();
                result.Start.AddTransitionsForSequence(sequence).SetEndWeight(weight);
                return result;
            }

            #endregion

            #region States manipulation (AddState and RemoveState methods state family)

            /// <summary>
            /// Adds new state and returns <see cref="StateBuilder"/> for it.
            /// </summary>
            public StateBuilder AddState()
            {
                if (this.states.Count >= this.maxStateCount)
                {
                    throw new AutomatonTooLargeException(this.maxStateCount);
                }

                var index = this.states.Count;
                this.states.Add(
                    new LinkedStateData
                    {
                        FirstTransitionIndex = -1,
                        LastTransitionIndex = -1,
                        EndWeight = Weight.Zero,
                    });
                return new StateBuilder(this, index);
            }

            /// <summary>
            /// Adds <paramref name="count"/> new states/
            /// </summary>
            public void AddStates(int count)
            {
                for (var i = 0; i < count; ++i)
                {
                    this.AddState();
                }
            }

            /// <summary>
            /// Adds all states and transitions from some <see cref="StateCollection"/>.
            /// </summary>
            public void AddStates(StateCollection states)
            {
                var oldStateCount = this.states.Count;
                foreach (var state in states)
                {
                    var stateBuilder = this.AddState();
                    stateBuilder.SetEndWeight(state.EndWeight);
                    foreach (var transition in state.Transitions)
                    {
                        var updatedTransition = transition.With(destinationStateIndex: transition.DestinationStateIndex + oldStateCount);
                        stateBuilder.AddTransition(updatedTransition);
                    }
                }
            }

            /// <summary>
            /// Removes state with given <paramref name="stateIndex"/> and all incoming transitions.
            /// </summary>
            public void RemoveState(int stateIndex)
            {
                // Caller must ensure that it doesn't try to delete start state. Because it will lead to
                // invalid automaton.
                Debug.Assert(stateIndex != this.StartStateIndex);

                // After state is removed, all its transitions will be dead
                for (var iterator = this[stateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                {
                    iterator.Remove();
                }

                this.states.RemoveAt(stateIndex);

                for (var i = 0; i < this.states.Count; ++i)
                {
                    for (var iterator = this[i].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        if (transition.DestinationStateIndex > stateIndex)
                        {
                            transition = transition.With(destinationStateIndex: transition.DestinationStateIndex - 1);
                            iterator.Value = transition;
                        }
                        else if (transition.DestinationStateIndex == stateIndex)
                        {
                            iterator.Remove();
                        }
                    }
                }
            }

            /// <summary>
            /// Removes a set of states from the automaton where the set is defined by labels not matching
            /// the <paramref name="removeLabel"/>.
            /// </summary>
            /// <param name="labels">State labels</param>
            /// <param name="removeLabel">Label which marks states which should be deleted</param>
            public int RemoveStates(bool[] labels, bool removeLabel)
            {
                var oldToNewStateIdMapping = PreallocatedAutomataObjects.LeaseRemoveStatesState(this.states.Count);

                var newStateId = 0;
                var deadStateCount = 0;
                for (var stateId = 0; stateId < this.states.Count; ++stateId)
                {
                    if (labels[stateId] != removeLabel)
                    {
                        oldToNewStateIdMapping[stateId] = newStateId++;
                    }
                    else
                    {
                        oldToNewStateIdMapping[stateId] = -1;
                        ++deadStateCount;
                    }
                }

                if (deadStateCount == 0)
                {
                    return 0;
                }

                this.StartStateIndex = oldToNewStateIdMapping[this.StartStateIndex];

                // Caller must ensure that it doesn't try to delete start state. Because it will lead to
                // invalid automaton.
                Debug.Assert(this.StartStateIndex != -1);

                for (var i = 0; i < this.states.Count; ++i)
                {
                    var newId = oldToNewStateIdMapping[i];
                    if (newId == -1)
                    {
                        // remove all transitions
                        for (var iterator = this[i].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            iterator.Remove();
                        }

                        continue;
                    }

                    Debug.Assert(newId <= i);

                    this.states[newId] = this.states[i];

                    // Remap transitions
                    for (var iterator = this[newId].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        var destinationStateIndex = oldToNewStateIdMapping[transition.DestinationStateIndex];
                        if (destinationStateIndex == -1)
                        {
                            iterator.Remove();
                        }
                        else
                        {
                            iterator.Value = transition.With(destinationStateIndex: destinationStateIndex);
                        }
                    }
                }

                this.states.RemoveRange(newStateId, this.states.Count - newStateId);

                return deadStateCount;
            }


            /// <summary>
            /// Creates an automaton <c>f'(s) = sum_{tu=s} f(t)g(u)</c>, where <c>f(t)</c> is the current
            /// automaton (in builder) and <c>g(u)</c> is the given automaton.
            /// The resulting automaton is also known as the Cauchy product of two automata.
            /// </summary>
            /// <param name="automaton">Given automaton.</param>
            /// <param name="group">If non-zero, all transitions in the appended part will be put
            /// into the specified group.</param>
            /// <param name="avoidEpsilonTransitions">When set to <see langword="true"/> (default), and
            /// at least one of the following
            /// <list type="bullet">
            /// <item>None of the end states of the current automaton have any outgoing transitions</item>
            /// <item>The start state of the given <paramref name="automaton"/> has no incoming transitions</item>
            /// </list>
            /// is true, no epsilon transitions will be used to concatenate the automata. Otherwise,
            /// epsilon transitions will be used</param>
            /// <returns>A pair of boolean values. The first indicates whether adding new epsilon transitions was avoided.
            /// The second indicates, whether the determinization state of the concatenated automata was preserved, i.e.
            /// whether both the current and the given automata being determinized implies that the result automaton is
            /// determinized as well.</returns>
            public (bool avoidedEpsilonTransitions, bool preservedDeterminizationState) Append(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton,
                int group = 0,
                bool avoidEpsilonTransitions = true)
            {
                var oldStateCount = this.states.Count;

                foreach (var state in automaton.States)
                {
                    var stateBuilder = this.AddState();
                    stateBuilder.SetEndWeight(state.EndWeight);
                    foreach (var transition in state.Transitions)
                    {
                        var updatedTransition = transition.With(destinationStateIndex: transition.DestinationStateIndex + oldStateCount);
                        if (group != 0)
                        {
                            updatedTransition = updatedTransition.With(group: group);
                        }

                        stateBuilder.AddTransition(updatedTransition);
                    }
                }

                var secondStartState = this[oldStateCount + automaton.Start.Index];


                bool allOldEndStatesHaveNoOutgoingTransitions = AllOldEndStatesHaveNoOutgoingTransitions();
                bool secondStartStateHasIncomingTransitions = SecondStartStateHasIncomingTransitions();
                bool canMergeEndAndStart = allOldEndStatesHaveNoOutgoingTransitions || !secondStartStateHasIncomingTransitions;
                bool willAvoidEpsilonTransitions = avoidEpsilonTransitions && canMergeEndAndStart;
                bool preservedDeterminization = avoidEpsilonTransitions && allOldEndStatesHaveNoOutgoingTransitions && !secondStartStateHasIncomingTransitions;
                if (willAvoidEpsilonTransitions)
                {
                    // Remove start state of appended automaton and copy all its transitions to previous end states
                    for (var i = 0; i < oldStateCount; ++i)
                    {
                        var endState = this[i];
                        if (!endState.CanEnd)
                        {
                            continue;
                        }

                        for (var iterator = secondStartState.TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            var transition = iterator.Value;

                            transition = transition.With(
                                weight: transition.DestinationStateIndex == secondStartState.Index ? (Weight?)null : transition.Weight * endState.EndWeight,
                                destinationStateIndex: transition.DestinationStateIndex == secondStartState.Index ? (int?)endState.Index : null,
                                group: group != 0 ? (int?)group : null);

                            endState.AddTransition(transition);
                        }

                        endState.SetEndWeight(endState.EndWeight * secondStartState.EndWeight);
                    }

                    this.RemoveState(secondStartState.Index);
                }
                else
                {
                    // Just connect all end states with start state of appended automaton
                    for (var i = 0; i < oldStateCount; i++)
                    {
                        var state = this[i];
                        if (state.CanEnd)
                        {
                            state.AddEpsilonTransition(state.EndWeight, secondStartState.Index, group);
                            state.SetEndWeight(Weight.Zero);
                        }
                    }
                }

                return (willAvoidEpsilonTransitions, preservedDeterminization);

                bool AllOldEndStatesHaveNoOutgoingTransitions()
                {
                    for (var i = 0; i < oldStateCount; ++i)
                    {
                        var state = this.states[i];
                        if (!state.EndWeight.IsZero && state.FirstTransitionIndex != -1)
                        {
                            return false;
                        }
                    }

                    return true;
                }

                bool SecondStartStateHasIncomingTransitions()
                {
                    foreach (var transition in automaton.Data.Transitions)
                    {
                        if (transition.DestinationStateIndex == automaton.Data.StartStateIndex)
                        {
                            return true;
                        }
                    }

                    return false;
                }
            }

            #endregion

            #region Result getters

            /// <summary>
            /// Builds new automaton object. Builder must not be used after this method is called
            /// </summary>
            public TThis GetAutomaton()
            {
                var result = Util.New<TThis>();
                result.Data = this.GetData();
                return result;
            }

            /// <summary>
            /// Stores built automaton in pre-allocated <see cref="Automaton{TSequence,TElement,TElementDistribution,TSequenceManipulator,TThis}"/> object.
            /// </summary>
            /// <remarks>
            /// Builder must not be used after this method is called.
            /// </remarks>
            public DataContainer GetData(bool? isDeterminized = null)
            {
                if (this.StartStateIndex < 0 || this.StartStateIndex >= this.states.Count)
                {
                    throw new InvalidOperationException(
                        $"Built automaton must have a valid start state. " +
                        $"StartStateIndex = {this.StartStateIndex}, states.Count = {this.states.Count}");
                }

                if (this.states == null)
                {
                    throw new InvalidOperationException("GetData() can be called only once");
                }

                var hasEpsilonTransitions = false;
                var usesGroups = false;
                var hasOnlyForwardTransitions = true;

                var resultStates = ReadOnlyArray.CreateBuilder<StateData>(this.states.Count);
                var resultTransitions = ReadOnlyArray.CreateBuilder<Transition>(this.transitions.Count - this.numRemovedTransitions);
                var nextResultTransitionIndex = 0;

                for (var i = 0; i < resultStates.Count; ++i)
                {
                    var firstResultTransitionIndex = nextResultTransitionIndex;
                    var transitionIndex = this.states[i].FirstTransitionIndex;
                    while (transitionIndex != -1)
                    {
                        var node = this.transitions[transitionIndex];
                        var transition = node.Transition;
                        Debug.Assert(
                            transition.DestinationStateIndex < resultStates.Count,
                            "Destination indexes must be in valid range");
                        resultTransitions[nextResultTransitionIndex] = transition;
                        ++nextResultTransitionIndex;
                        hasEpsilonTransitions = hasEpsilonTransitions || transition.IsEpsilon;
                        usesGroups = usesGroups || (transition.Group != 0);

                        if (transition.DestinationStateIndex <= i)
                        {
                            hasOnlyForwardTransitions = false;
                        }

                        transitionIndex = node.Next;
                    }

                    resultStates[i] = new StateData(
                        firstResultTransitionIndex,
                        nextResultTransitionIndex - firstResultTransitionIndex,
                        this.states[i].EndWeight);
                }

                Debug.Assert(
                    nextResultTransitionIndex == resultTransitions.Count,
                    "number of copied transitions must match result array size");

                // Detect two very common automata shapes
                var isEnumerable = hasOnlyForwardTransitions ? true : (bool?)null;

                // Return lists into the preallocated cache. This will make this object unusable
                // from now on.
                this.states.Clear();
                this.transitions.Clear();
                preallocatedContainers = (this.states, this.transitions);
                this.states = null;
                this.transitions = null;

                return new DataContainer(
                    this.StartStateIndex,
                    resultStates.MoveToImmutable(),
                    resultTransitions.MoveToImmutable(),
                    !hasEpsilonTransitions,
                    usesGroups,
                    isDeterminized,
                    isZero: null,
                    isEnumerable: isEnumerable);
            }

            #endregion

            #region Helpers

            /// <summary>
            /// Removes all states and transitions from builder returning it to empty state.
            /// </summary>
            public void Clear()
            {
                this.states.Clear();
                this.transitions.Clear();
                this.numRemovedTransitions = 0;
                this.StartStateIndex = 0;
            }

            #endregion

            /// <summary>
            /// Builder struct for a state
            /// </summary>
            /// <remarks>
            /// Implemented as value type to minimize gc.
            /// </remarks>
            public struct StateBuilder
            {
                /// <summary>
                /// Builder for whole automaton.
                /// </summary>
                private readonly Builder builder;

                /// <summary>
                /// Index of current state.
                /// </summary>
                public int Index { get; }

                /// <summary>
                /// Gets a value indicating whether the ending weight of this state is greater than zero.
                /// </summary>
                public bool CanEnd => !this.builder.states[this.Index].EndWeight.IsZero;

                /// <summary>
                /// Gets or sets the ending weight of the state.
                /// </summary>
                public Weight EndWeight => this.builder.states[this.Index].EndWeight;

                /// <summary>
                /// Gets a value indicating whether this state is a start state for any transitions.
                /// </summary>
                public bool HasTransitions => this.builder[this.Index].TransitionIterator.Ok;

                /// <summary>
                /// Gets <see cref="TransitionIterator"/> over transitions of this state.
                /// </summary>
                public TransitionIterator TransitionIterator =>
                    new TransitionIterator(this.builder, this.Index, this.builder.states[this.Index].FirstTransitionIndex);

                /// <summary>
                /// Initializes a new instance of <see cref="StateBuilder"/> struct.
                /// </summary>
                internal StateBuilder(Builder builder, int index)
                {
                    this.builder = builder;
                    this.Index = index;
                }

                /// <summary>
                /// Sets a new end weight for this state.
                /// </summary>
                public StateBuilder SetEndWeight(Weight weight)
                {
                    var state = this.builder.states[this.Index];
                    state.EndWeight = weight;
                    this.builder.states[this.Index] = state;
                    return this;
                }

                #region AddTransition variants

                /// <summary>
                /// Adds a transition to the current state.
                /// </summary>
                /// <remarks>
                /// Transitions are stored in linked list over an array. This function updates this linked
                /// list and pointers into it. It involves a lot of bookkeeping.
                /// </remarks>
                public StateBuilder AddTransition(Transition transition)
                {
                    var state = this.builder.states[this.Index];
                    var transitionIndex = this.builder.transitions.Count;
                    this.builder.transitions.Add(
                        new LinkedTransitionNode
                        {
                            Transition = transition,
                            Next = -1,
                            Prev = state.LastTransitionIndex,
                        });

                    if (state.LastTransitionIndex != -1)
                    {
                        // update "next" field in old tail
                        var oldTail = this.builder.transitions[state.LastTransitionIndex];
                        oldTail.Next = transitionIndex;
                        this.builder.transitions[state.LastTransitionIndex] = oldTail;
                    }
                    else
                    {
                        state.FirstTransitionIndex = transitionIndex;
                    }

                    state.LastTransitionIndex = transitionIndex;
                    this.builder.states[this.Index] = state;

                    return new StateBuilder(this.builder, transition.DestinationStateIndex);
                }

                /// <summary>
                /// Adds a transition to the current state.
                /// </summary>
                /// <param name="elementDistribution">
                /// The element distribution associated with the transition.
                /// If the value of this parameter is <see langword="null"/>, an epsilon transition will be created.
                /// </param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="destinationStateIndex">
                /// The destination state of the added transition.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The destination state of the added transition.</returns>
                public StateBuilder AddTransition(
                    Option<TElementDistribution> elementDistribution,
                    Weight weight,
                    int? destinationStateIndex = null,
                    int group = 0)
                {
                    if (destinationStateIndex == null)
                    {
                        destinationStateIndex = this.builder.AddState().Index;
                    }

                    return this.AddTransition(
                        new Transition(elementDistribution, weight, destinationStateIndex.Value, group));
                }

                /// <summary>
                /// Adds a transition labeled with a given element to the current state.
                /// </summary>
                /// <param name="element">The element.</param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="destinationStateIndex">
                /// The destination state of the added transition.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The destination state of the added transition.</returns>
                public StateBuilder AddTransition(
                    TElement element,
                    Weight weight,
                    int? destinationStateIndex = null,
                    int group = 0)
                {
                    return this.AddTransition(
                        ElementDistributionFactory.CreatePointMass(element), weight, destinationStateIndex, group);
                }

                /// <summary>
                /// Adds an epsilon transition to the current state.
                /// </summary>
                /// <param name="weight">The transition weight.</param>
                /// <param name="destinationStateIndex">
                /// The destination state of the added transition.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The destination state of the added transition.</returns>
                public StateBuilder AddEpsilonTransition(
                    Weight weight, int? destinationStateIndex = null, int group = 0)
                {
                    return this.AddTransition(Option.None, weight, destinationStateIndex, group);
                }

                /// <summary>
                /// Adds a self-transition labeled with a given element to the current state.
                /// </summary>
                /// <param name="element">The element.</param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The current state.</returns>
                public StateBuilder AddSelfTransition(TElement element, Weight weight, int group = 0)
                {
                    return this.AddTransition(element, weight, this.Index, group);
                }

                /// <summary>
                /// Adds a self-transition to the current state.
                /// </summary>
                /// <param name="elementDistribution">
                /// The element distribution associated with the transition.
                /// If the value of this parameter is <see langword="null"/>, an epsilon transition will be created.
                /// </param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The current state.</returns>
                public StateBuilder AddSelfTransition(
                    Option<TElementDistribution> elementDistribution, Weight weight, byte group = 0)
                {
                    return this.AddTransition(elementDistribution, weight, this.Index, group);
                }


                /// <summary>
                /// Adds a series of transitions labeled with the elements of a given sequence to the current state,
                /// as well as the intermediate states. All the added transitions have unit weight.
                /// </summary>
                /// <param name="sequence">The sequence.</param>
                /// <param name="destinationStateIndex">
                /// The last state in the transition series.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.
                /// </param>
                /// <param name="group">The group of the added transitions.</param>
                /// <returns>The last state in the added transition series.</returns>
                public StateBuilder AddTransitionsForSequence(
                    TSequence sequence,
                    int? destinationStateIndex = null,
                    int group = 0)
                {
                    var currentState = this;
                    using (var enumerator = sequence.GetEnumerator())
                    {
                        var moveNext = enumerator.MoveNext();
                        while (moveNext)
                        {
                            var element = enumerator.Current;
                            moveNext = enumerator.MoveNext();
                            currentState = currentState.AddTransition(
                                element, Weight.One, moveNext ? null : destinationStateIndex, group);
                        }
                    }

                    return currentState;
                }

                #endregion
            }

            /// <summary>
            /// Helper struct for iterating over currently constructed list of transitions for state.
            /// Unlike standard enumerator pattern through this iterator elements can be changed and removed.
            /// </summary>
            /// <remarks>
            /// Implemented as a value type to minimize amount of GC.
            /// </remarks>
            /// <remarks>
            /// Implementation does not follow common due to three reasons.
            /// - Items are mutable through this iterator (unlike enumerators)
            /// - Implementation-wise initializing iterator in "valid" state is easier than in the one which
            ///   requires MoveNext() call
            /// - Iterator with this interface can conveniently be used in a for-loop unlike traditional
            ///   enumerators which will require 3 lines in a while-loop
            /// </remarks>
            public struct TransitionIterator
            {
                /// <summary>
                /// Automaton builder which contains transitions array.
                /// </summary>
                private readonly Builder builder;

                /// <summary>
                /// Index of state to which this transition belongs.
                /// </summary>
                private int stateIndex;

                /// <summary>
                /// Index of current transition in this.builder.transitions array.
                /// </summary>
                private int index;

                /// <summary>
                /// Initializes new instance of <see cref="TransitionIterator"/> struct.
                /// </summary>
                public TransitionIterator(Builder builder, int stateIndex, int index)
                {
                    this.builder = builder;
                    this.stateIndex = stateIndex;
                    this.index = index;
                }

                /// <summary>
                /// Gets or sets current transition value.
                /// </summary>
                public Transition Value
                {
                    get => this.builder.transitions[this.index].Transition;
                    set
                    {
                        var node = this.builder.transitions[this.index];
                        node.Transition = value;
                        this.builder.transitions[this.index] = node;
                    }
                }

                /// <summary>
                /// Marks current transition as removed. This transition will not be visible during further
                /// iterations.
                /// </summary>
                /// <remarks>
                /// <see cref="Remove()"/> can be called only once until <see cref="Next()"/> call.
                /// </remarks>
                public void Remove()
                {
                    var state = this.builder.states[this.stateIndex];
                    var node = this.builder.transitions[this.index];

                    // unlink from previous node
                    if (node.Prev != -1)
                    {
                        var prevNode = this.builder.transitions[node.Prev];
                        prevNode.Next = node.Next;
                        this.builder.transitions[node.Prev] = prevNode;
                    }

                    // unlink from next node
                    if (node.Next != -1)
                    {
                        var nextNode = this.builder.transitions[node.Next];
                        nextNode.Prev = node.Prev;
                        this.builder.transitions[node.Next] = nextNode;
                    }

                    // update references in state
                    if (state.FirstTransitionIndex == this.index || state.LastTransitionIndex == this.index)
                    {
                        if (state.FirstTransitionIndex == this.index)
                        {
                            state.FirstTransitionIndex = node.Next;
                        }

                        if (state.LastTransitionIndex == this.index)
                        {
                            state.LastTransitionIndex = node.Prev;
                        }

                        this.builder.states[this.stateIndex] = state;
                    }

                    ++this.builder.numRemovedTransitions;
                }

                /// <summary>
                /// Gets a value indicating whether iterator is dereferenceable - its Value can be get or set.
                /// Once iteration is finished property will become false.
                /// </summary>
                public bool Ok => this.index != -1;

                /// <summary>
                /// Moves iterator to next transition in list.
                /// </summary>
                public void Next()
                {
                    this.index = this.builder.transitions[this.index].Next;
                }

            }

            /// <summary>
            /// Version of <see cref="StateData"/> that is used during automaton constructions. Unlike
            /// regular <see cref="StateData"/> transitions are stored as a linked list over
            /// <see cref="Builder.transitions"/> array.
            /// </summary>
            private struct LinkedStateData
            {
                /// <summary>
                /// Index of the head of transitions list in <see cref="Builder.transitions"/>.
                /// </summary>
                public int FirstTransitionIndex { get; internal set; }

                /// <summary>
                /// Index of the tail of transitions list in <see cref="Builder.transitions"/>.
                /// </summary>
                public int LastTransitionIndex { get; internal set; }

                /// <summary>
                /// Ending weight of the state.
                /// </summary>
                public Weight EndWeight { get; internal set; }
            }

            /// <summary>
            /// Linked list node for representing transitions for state.
            /// </summary>
            /// <remarks>
            /// Fields are mutable, because they are changed during automaton construction.
            /// </remarks>
            private struct LinkedTransitionNode
            {
                /// <summary>
                /// Stored transition.
                /// </summary>
                public Transition Transition;

                /// <summary>
                /// Index of next transition in list.
                /// </summary>
                public int Next;

                /// <summary>
                /// Index of previous transition in list.
                /// </summary>
                public int Prev;
            }
        }
    }
}
