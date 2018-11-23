// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;
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
        /// Represents a reference to a state of automaton for exposure in public API.
        /// </summary>
        /// <remarks>
        /// Acts as a "fat reference" to state in automaton. In addition to reference to actual StateData it carries
        /// 2 additional properties for convinience: <see cref="Owner"/> automaton and <see cref="Index"/> of the state.
        /// We don't store them in <see cref="StateData"/> to save some memoty. C# compiler and .NET jitter are good
        /// at optimizing wrapping where it is not needed.
        /// </remarks>
        public struct State : IEquatable<State>
        {
            internal readonly StateData Data;

            /// <summary>
            /// Initializes a new instance of <see cref="State"/> class. Used internally by automaton implementation
            /// to wrap StateData for use in public Automaton APIs.
            /// </summary>
            internal State(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> owner, int index, StateData data)
            {
                this.Owner = owner;
                this.Index = index;
                this.Data = data;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="State"/> class. Created state does not belong
            /// to any automaton and has to be added to some automaton explicitly via Automaton.AddStates.
            /// </summary>
            /// <param name="index">The index of the state.</param>
            /// <param name="transitions">The outgoing transitions.</param>
            /// <param name="endWeight">The ending weight of the state.</param>
            [Construction("Index", "GetTransitions", "EndWeight")]
            public State(int index, IEnumerable<Transition> transitions, Weight endWeight)
                : this()
            {
                Argument.CheckIfInRange(index >= 0, "index", "State index must be non-negative.");
                this.Index = index;
                this.Data = new StateData(transitions, endWeight);
            }

            /// <summary>
            /// Returns where this State represents some valid state in automaton.
            /// </summary>
            public bool IsNull => this.Data == null;

            /// <summary>
            /// Automaton to which this state belongs.
            /// </summary>
            public Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> Owner { get; }

            /// <summary>
            /// Gets the index of the state.
            /// </summary>
            public int Index { get; }

            /// <summary>
            /// Gets or sets the ending weight of the state.
            /// </summary>
            /// <remarks>
            /// C# compiler disallows to use property setter if it sees that <see cref="States"/> instance is a temporary.
            /// It is not smart enough to understand that property setter actually changes something behind a reference.
            /// To overcome this issue special <see cref="SetEndWeight"/> method is added calling which is equivalent
            /// to calling property setter but is not rejected by compiler.
            /// </remarks>
            public Weight EndWeight => this.Data.EndWeight;

            /// <summary>
            /// Sets the <see cref="EndWeight"/> property of State.
            ///
            /// Because <see cref="State"/> is a struct, trying to set <see cref="EndWeight"/> on it
            /// (if property setter was provided) would result in compilation error. Compiler isn't
            /// smart enough to see that setting property just updates the value in referenced <see cref="Data"/>.
            /// Having a method call doesn't create this problem.
            /// </summary>
            /// <param name="weight">New end weight.</param>
            public void SetEndWeight(Weight weight)
            {
                this.Data.EndWeight = weight;
            }

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            public bool CanEnd => this.Data.CanEnd;

            /// <summary>
            /// Gets the number of outgoing transitions.
            /// </summary>
            public int TransitionCount => this.Data.TransitionCount;

            /// <summary>
            /// Creates the copy of the array of outgoing transitions. Used by quoting.
            /// </summary>
            /// <returns>The copy of the array of outgoing transitions.</returns>
            public Transition[] GetTransitions() => this.Data.GetTransitions();

            /// <summary>
            /// Compares 2 states for equality.
            /// </summary>
            public static bool operator ==(State a, State b) => a.Data == b.Data;

            /// <summary>
            /// Compares 2 states for inequality.
            /// </summary>
            public static bool operator !=(State a, State b) => !(a == b);

            /// <summary>
            /// Compares 2 states for equality.
            /// </summary>
            public bool Equals(State that) => this == that;

            /// <summary>
            /// Compares 2 states for equality.
            /// </summary>
            public override bool Equals(object obj) => obj is State that && this.Equals(that);

            /// <summary>
            /// Returns HashCode of this state.
            /// </summary>
            public override int GetHashCode() => this.Data?.GetHashCode() ?? 0;

            /// <summary>
            /// Adds a series of transitions labeled with the elements of a given sequence to the current state,
            /// as well as the intermediate states. All the added transitions have unit weight.
            /// </summary>
            /// <param name="sequence">The sequence.</param>
            /// <param name="destinationState">
            /// The last state in the transition series.
            /// If the value of this parameter is <see langword="null"/>, a new state will be created.
            /// </param>
            /// <param name="group">The group of the added transitions.</param>
            /// <returns>The last state in the added transition series.</returns>
            public State AddTransitionsForSequence(TSequence sequence, State destinationState = default(State), int group = 0)
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
                            element, Weight.One, moveNext ? default(State) : destinationState, group);
                    }
                }

                return currentState;
            }

            /// <summary>
            /// Adds a transition labeled with a given element to the current state.
            /// </summary>
            /// <param name="element">The element.</param>
            /// <param name="weight">The transition weight.</param>
            /// <param name="destinationState">
            /// The destination state of the added transition.
            /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
            /// <param name="group">The group of the added transition.</param>
            /// <returns>The destination state of the added transition.</returns>
            public State AddTransition(TElement element, Weight weight, State destinationState = default(State), int group = 0)
            {
                return this.AddTransition(new TElementDistribution { Point = element }, weight, destinationState, group);
            }

            /// <summary>
            /// Adds an epsilon transition to the current state.
            /// </summary>
            /// <param name="weight">The transition weight.</param>
            /// <param name="destinationState">
            /// The destination state of the added transition.
            /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
            /// <param name="group">The group of the added transition.</param>
            /// <returns>The destination state of the added transition.</returns>
            public State AddEpsilonTransition(Weight weight, State destinationState = default(State), int group = 0)
            {
                return this.AddTransition(Option.None, weight, destinationState, group);
            }

            /// <summary>
            /// Adds a transition to the current state.
            /// </summary>
            /// <param name="elementDistribution">
            /// The element distribution associated with the transition.
            /// If the value of this parameter is <see langword="null"/>, an epsilon transition will be created.
            /// </param>
            /// <param name="weight">The transition weight.</param>
            /// <param name="destinationState">
            /// The destination state of the added transition.
            /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
            /// <param name="group">The group of the added transition.</param>
            /// <returns>The destination state of the added transition.</returns>
            public State AddTransition(Option<TElementDistribution> elementDistribution, Weight weight, State destinationState = default(State), int group = 0)
            {
                if (destinationState.IsNull)
                {
                    destinationState = this.Owner.AddState();
                }
                else if (!ReferenceEquals(destinationState.Owner, this.Owner))
                {
                    throw new ArgumentException("The given state belongs to another automaton.");
                }

                this.AddTransition(new Transition(elementDistribution, weight, destinationState.Index, group));
                return destinationState;
            }

            /// <summary>
            /// Adds a transition to the current state.
            /// </summary>
            /// <param name="transition">The transition to add.</param>
            /// <returns>The destination state of the added transition.</returns>
            public State AddTransition(Transition transition)
            {
                Argument.CheckIfValid(this.Owner == null || transition.DestinationStateIndex < this.Owner.statesData.Count, "transition", "The destination state index is not valid.");
                
                this.Data.AddTransition(transition);
                if (this.Owner.isEpsilonFree == true)
                {
                    this.Owner.isEpsilonFree = !transition.IsEpsilon;
                }

                return this.Owner.States[transition.DestinationStateIndex];
            }

            /// <summary>
            /// Adds a self-transition labeled with a given element to the current state.
            /// </summary>
            /// <param name="element">The element.</param>
            /// <param name="weight">The transition weight.</param>
            /// <param name="group">The group of the added transition.</param>
            /// <returns>The current state.</returns>
            public State AddSelfTransition(TElement element, Weight weight, int group = 0)
            {
                return this.AddTransition(element, weight, this, group);
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
            public State AddSelfTransition(Option<TElementDistribution> elementDistribution, Weight weight, byte group = 0)
            {
                return this.AddTransition(elementDistribution, weight, this, group);
            }

            /// <summary>
            /// Gets the transition at a specified index.
            /// </summary>
            /// <param name="index">The index of the transition.</param>
            /// <returns>The transition.</returns>
            public Transition GetTransition(int index) => this.Data.GetTransition(index);

            /// <summary>
            /// Replaces the transition at a given index with a given transition.
            /// </summary>
            /// <param name="index">The index of the transition to replace.</param>
            /// <param name="updatedTransition">The transition to replace with.</param>
            public void SetTransition(int index, Transition updatedTransition)
            {
                Argument.CheckIfInRange(index >= 0 && index < this.TransitionCount, "index", "An invalid transition index given.");
                Argument.CheckIfValid(updatedTransition.DestinationStateIndex < this.Owner.statesData.Count, "updatedTransition", "The destination state index is not valid.");

                if (updatedTransition.IsEpsilon)
                {
                    this.Owner.isEpsilonFree = false;
                }
                else
                {
                    this.Owner.isEpsilonFree = null;
                }

                this.Data.SetTransition(index, updatedTransition);
            }

            /// <summary>
            /// Removes the transition with a given index.
            /// </summary>
            /// <param name="index">The index of the transition to remove.</param>
            public void RemoveTransition(int index) => this.Data.RemoveTransition(index);

            /// <summary>
            /// Returns a string that represents the state.
            /// </summary>
            /// <returns>A string that represents the state.</returns>
            public override string ToString()
            {
                const string StartStateMarker = "START ->";
                const string TransitionSeparator = ",";

                var sb = new StringBuilder();

                bool isStartState = this.Owner != null && this.Owner.Start == this;
                if (isStartState)
                {
                    sb.Append(StartStateMarker);
                }

                bool firstTransition = true;
                foreach (Transition transition in this.GetTransitions())
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

                if (CanEnd)
                {
                    if (!firstTransition) sb.Append(TransitionSeparator);
                    sb.Append(this.EndWeight.Value + " -> END");
                }

                return sb.ToString();
            }

            /// <summary>
            /// Computes the logarithm of the value of the automaton
            /// having this state as the start state on a given sequence.
            /// </summary>
            /// <param name="sequence">The sequence.</param>
            /// <returns>The logarithm of the value on the sequence.</returns>
            public double GetLogValue(TSequence sequence)
            {
                var valueCache = new Dictionary<(int, int), Weight>();
                return this.DoGetValue(sequence, 0, valueCache).LogValue;
            }

            /// <summary>
            /// Gets whether the automaton having this state as the start state is zero everywhere.
            /// </summary>
            /// <returns>A value indicating whether the automaton having this state as the start state is zero everywhere.</returns>
            public bool IsZero()
            {
                var visitedStates = new BitArray(this.Owner.States.Count, false);
                return this.DoIsZero(visitedStates);
            }

            /// <summary>
            /// Gets whether the automaton having this state as the start state has non-trivial loops.
            /// </summary>
            /// <returns>A value indicating whether the automaton having this state as the start state is zero everywhere.</returns>
            public bool HasNonTrivialLoops()
            {
                return this.DoHasNonTrivialLoops(new ArrayDictionary<bool>(this.Owner.States.Count));
            }

            /// <summary>
            /// Gets the epsilon closure of this state.
            /// </summary>
            /// <returns>The epsilon closure of this state.</returns>
            public EpsilonClosure GetEpsilonClosure()
            {
                return new EpsilonClosure(this);
            }

            #region Helpers

            /// <summary>
            /// Recursively checks if the automaton has non-trivial loops
            /// (i.e. loops consisting of more than one transition).
            /// </summary>
            /// <param name="stateInStack">
            /// A dictionary, storing for each state whether it has already been visited, and,
            /// if so, whether the state still is on the traversal stack.</param>
            /// <returns>
            /// <see langword="true"/> if a non-trivial loop has been found,
            /// <see langword="false"/> otherwise.
            /// </returns>
            private bool DoHasNonTrivialLoops(ArrayDictionary<bool> stateInStack)
            {
                bool inStack;
                if (stateInStack.TryGetValue(this.Index, out inStack))
                {
                    return inStack;
                }

                stateInStack.Add(this.Index, true);

                for (int i = 0; i < this.TransitionCount; ++i)
                {
                    var transition = this.GetTransition(i);
                    if (transition.DestinationStateIndex != this.Index)
                    {
                        var destState = this.Owner.States[transition.DestinationStateIndex];
                        if (destState.DoHasNonTrivialLoops(stateInStack))
                        {
                            return true;
                        }
                    }
                }

                stateInStack[this.Index] = false;
                return false;
            }

            /// <summary>
            /// Recursively checks if the automaton is zero.
            /// </summary>
            /// <param name="visitedStates">For each state stores whether it has been already visited.</param>
            /// <returns>
            /// <see langword="false"/> if an accepting path has been found,
            /// <see langword="true"/> otherwise.
            /// </returns>
            private bool DoIsZero(BitArray visitedStates)
            {
                if (visitedStates[this.Index])
                {
                    return true;
                }

                visitedStates[this.Index] = true;

                var isZero = !this.CanEnd;
                var transitionIndex = 0;
                while (isZero && transitionIndex < this.TransitionCount)
                {
                    var transition = this.GetTransition(transitionIndex);
                    if (!transition.Weight.IsZero)
                    {
                        var destState = this.Owner.States[transition.DestinationStateIndex];
                        isZero = destState.DoIsZero(visitedStates);
                    }

                    ++transitionIndex;
                }

                return isZero;
            }

            /// <summary>
            /// Recursively computes the value of the automaton on a given sequence.
            /// </summary>
            /// <param name="sequence">The sequence to compute the value on.</param>
            /// <param name="sequencePosition">The current position in the sequence.</param>
            /// <param name="valueCache">A lookup table for memoization.</param>
            /// <returns>The value computed from the current state.</returns>
            private Weight DoGetValue(
                TSequence sequence, int sequencePosition, Dictionary<(int, int), Weight> valueCache)
            {
                var stateIndexPair = (this.Index, sequencePosition);
                if (valueCache.TryGetValue(stateIndexPair, out var cachedValue))
                {
                    return cachedValue;
                }

                var closure = this.GetEpsilonClosure();

                var value = Weight.Zero;
                var count = SequenceManipulator.GetLength(sequence);
                var isCurrent = sequencePosition < count;
                if (isCurrent)
                {
                    var element = SequenceManipulator.GetElement(sequence, sequencePosition);
                    for (var closureStateIndex = 0; closureStateIndex < closure.Size; ++closureStateIndex)
                    {
                        var closureState = closure.GetStateByIndex(closureStateIndex);
                        var closureStateWeight = closure.GetStateWeightByIndex(closureStateIndex);

                        for (int transitionIndex = 0; transitionIndex < closureState.TransitionCount; transitionIndex++)
                        {
                            var transition = closureState.GetTransition(transitionIndex);
                            if (transition.IsEpsilon)
                            {
                                continue; // The destination is a part of the closure anyway
                            }

                            var destState = this.Owner.States[transition.DestinationStateIndex];
                            var distWeight = Weight.FromLogValue(transition.ElementDistribution.Value.GetLogProb(element));
                            if (!distWeight.IsZero && !transition.Weight.IsZero)
                            {
                                var destValue = destState.DoGetValue(sequence, sequencePosition + 1, valueCache);
                                if (!destValue.IsZero)
                                {
                                    value = Weight.Sum(
                                        value,
                                        Weight.Product(closureStateWeight, transition.Weight, distWeight, destValue));
                                }
                            }
                        }
                    }
                }
                else
                {
                    value = closure.EndWeight;
                }

                valueCache.Add(stateIndexPair, value);
                return value;
            }

            #endregion

            /// <summary>
            /// Whether there are incoming transitions to this state
            /// </summary>
            public bool HasIncomingTransitions
            {
                get
                {
                    var this_ = this;
                    return this.Owner.States.Any(
                        state => state.GetTransitions().Any(
                            transition => transition.DestinationStateIndex == this_.Index));
                }
            }


            /// <summary>
            /// Writes the automaton state.
            /// </summary>
            public void Write(Action<int> writeInt32, Action<double> writeDouble, Action<TElementDistribution> writeElementDistribution)
            {
                this.EndWeight.Write(writeDouble);
                writeInt32(this.Index);
                writeInt32(this.TransitionCount);
                for (var i = 0; i < TransitionCount; i++)
                {
                    GetTransition(i).Write(writeInt32, writeDouble, writeElementDistribution);
                }
            }

            /// <summary>
            /// Reads the automaton state.
            /// </summary>
            public static State Read(Func<int> readInt32, Func<double> readDouble, Func<TElementDistribution> readElementDistribution)
            {
                var endWeight = Weight.Read(readDouble);
                // Note: index is serialized for compatibility with old binary serializations
                var index = readInt32();
                var transitionCount = readInt32();
                var transitions = new Transition[transitionCount];
                for (var i = 0; i < transitionCount; i++)
                {
                    transitions[i] = Transition.Read(readInt32, readDouble, readElementDistribution);
                }

                return new State(index, transitions, endWeight);
            }
        }
    }
}
