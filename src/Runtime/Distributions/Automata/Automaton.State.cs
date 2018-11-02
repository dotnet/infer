// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains the class used to represent a state of an automaton.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : class, IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        /// <summary>
        /// Represents a state of an automaton.
        /// </summary>
        [Serializable]
        [DataContract(IsReference = true)]
        public class State
        {
            //// This class has been made inner so that the user doesn't have to deal with a lot of generic parameters on it.

            /// <summary>
            /// The default capacity of the <see cref="transitions"/>.
            /// </summary>
            private const int DefaultTransitionArrayCapacity = 3;

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
            /// Initializes a new instance of the <see cref="State"/> class.
            /// </summary>
            public State()
            {
                this.EndWeight = Weight.Zero;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="State"/> class.
            /// </summary>
            /// <param name="index">The index of the state.</param>
            /// <param name="transitions">The outgoing transitions.</param>
            /// <param name="endWeight">The ending weight of the state.</param>
            [Construction("Index", "GetTransitions", "EndWeight")]
            public State(int index, IEnumerable<Transition> transitions, Weight endWeight)
                : this()
            {
                Argument.CheckIfInRange(index >= 0, "index", "State index must be non-negative.");
                Argument.CheckIfNotNull(transitions, "transitions");

                this.Index = index;
                this.EndWeight = endWeight;

                foreach (var transition in transitions)
                {
                    this.DoAddTransition(transition);
                }
            }

            /// <summary>
            /// Gets the automaton which owns the state.
            /// </summary>
            /// <remarks>
            /// Owner is not serialized to avoid circular references. It has to be restored manually upon deserialization.
            /// BinaryFormatter and Newtonsoft.Json handle circular references differently by default.
            /// At the same time [DataMember] does the right thing, because IsReference=true property on State DataContract
            /// makes DataContractSerializer handle circular references just fine.
            /// </remarks>
            [DataMember]
            [NonSerializedProperty]
            public TThis Owner { get; internal set; }

            /// <summary>
            /// Helper method for Newtonsoft.Json to skip serialization of <see cref="Owner"/> property.
            /// </summary>
            public bool ShouldSerializeOwner() => false;

            /// <summary>
            /// Gets the index of the state.
            /// </summary>
            [DataMember]
            public int Index { get; internal set; } // TODO: setter of this property is needed only for the state removal procedure

            /// <summary>
            /// Gets or sets the ending weight of the state.
            /// </summary>
            [DataMember]
            public Weight EndWeight { get; set; }
            
            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            public bool CanEnd
            {
                get { return !this.EndWeight.IsZero; }
            }

            /// <summary>
            /// Gets the number of outgoing transitions.
            /// </summary>
            public int TransitionCount
            {
                get { return this.transitionCount; }
            }

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
            public State AddTransitionsForSequence(TSequence sequence, State destinationState = null, int group = 0)
            {
                State currentState = this;
                IEnumerator<TElement> enumerator = sequence.GetEnumerator();
                bool moveNext = enumerator.MoveNext();
                while (moveNext)
                {
                    TElement element = enumerator.Current;
                    moveNext = enumerator.MoveNext();
                    currentState = currentState.AddTransition(element, Weight.One, moveNext ? null : destinationState, group);
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
            public State AddTransition(TElement element, Weight weight, State destinationState = null, int group = 0)
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
            public State AddEpsilonTransition(Weight weight, State destinationState = null, int group = 0)
            {
                return this.AddTransition(null, weight, destinationState, group);
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
            public State AddTransition(TElementDistribution elementDistribution, Weight weight, State destinationState = null, int group = 0)
            {
                if (destinationState == null)
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
                Argument.CheckIfValid(this.Owner == null || transition.DestinationStateIndex < this.Owner.states.Count, "transition", "The destination state index is not valid.");

                this.DoAddTransition(transition);
                if (this.Owner.isEpsilonFree==true) this.Owner.isEpsilonFree = !transition.IsEpsilon;
                return this.Owner.states[transition.DestinationStateIndex];
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
            public State AddSelfTransition(TElementDistribution elementDistribution, Weight weight, byte group = 0)
            {
                return this.AddTransition(elementDistribution, weight, this, group);
            }

            /// <summary>
            /// Gets the transition at a specified index.
            /// </summary>
            /// <param name="index">The index of the transition.</param>
            /// <returns>The transition.</returns>
            public Transition GetTransition(int index)
            {
                //Argument.CheckIfInRange(index >= 0 && index < this.transitionCount, "index", "An invalid transition index given.");
                return this.transitions[index];
            }

            /// <summary>
            /// Replaces the transition at a given index with a given transition.
            /// </summary>
            /// <param name="index">The index of the transition to replace.</param>
            /// <param name="updatedTransition">The transition to replace with.</param>
            public void SetTransition(int index, Transition updatedTransition)
            {
                Argument.CheckIfInRange(index >= 0 && index < this.transitionCount, "index", "An invalid transition index given.");
                Argument.CheckIfValid(updatedTransition.DestinationStateIndex < this.Owner.states.Count, "updatedTransition", "The destination state index is not valid.");

                if (updatedTransition.IsEpsilon) {
                    this.Owner.isEpsilonFree = false;
                }
                else
                {
                    this.Owner.isEpsilonFree = null;
                }
                this.transitions[index] = updatedTransition;
            }

            /// <summary>
            /// Removes the transition with a given index.
            /// </summary>
            /// <param name="index">The index of the transition to remove.</param>
            public void RemoveTransition(int index)
            {
                Argument.CheckIfInRange(index >= 0 && index < this.transitionCount, "index", "An invalid transition index given.");
                this.transitions[index] = this.transitions[--this.transitionCount];
            }

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
                    sb.Append(this.EndWeight.Value+" -> END");
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
                var valueCache = new Dictionary<IntPair, Weight>(IntPair.DefaultEqualityComparer);
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
            /// Adds a given transition to the transition array, increasing the size of the array if necessary.
            /// </summary>
            /// <param name="transition">The transition to add.</param>
            private void DoAddTransition(Transition transition)
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

                for (int i = 0; i < this.transitionCount; ++i)
                {
                    if (this.transitions[i].DestinationStateIndex != this.Index)
                    {
                        State destState = this.Owner.States[this.transitions[i].DestinationStateIndex];
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

                bool isZero = !this.CanEnd;
                int transitionIndex = 0;
                while (isZero && transitionIndex < this.transitionCount)
                {
                    if (!this.transitions[transitionIndex].Weight.IsZero)
                    {
                        State destState = this.Owner.States[this.transitions[transitionIndex].DestinationStateIndex];
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
                TSequence sequence, int sequencePosition, Dictionary<IntPair, Weight> valueCache)
            {
                var stateIndexPair = new IntPair(this.Index, sequencePosition);
                Weight cachedValue;
                if (valueCache.TryGetValue(stateIndexPair, out cachedValue))
                {
                    return cachedValue;
                }

                EpsilonClosure closure = this.GetEpsilonClosure();

                Weight value = Weight.Zero;
                int count = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>.SequenceManipulator.GetLength(sequence);
                bool isCurrent = sequencePosition < count;
                if (isCurrent)
                {
                    TElement element = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>.SequenceManipulator.GetElement(sequence, sequencePosition);
                    for (int closureStateIndex = 0; closureStateIndex < closure.Size; ++closureStateIndex)
                    {
                        State closureState = closure.GetStateByIndex(closureStateIndex);
                        Weight closureStateWeight = closure.GetStateWeightByIndex(closureStateIndex);

                        for (int transitionIndex = 0; transitionIndex < closureState.transitionCount; transitionIndex++)
                        {
                            Transition transition = closureState.transitions[transitionIndex];
                            if (transition.IsEpsilon)
                            {
                                continue; // The destination is a part of the closure anyway
                            }

                            State destState = this.Owner.states[transition.DestinationStateIndex];
                            Weight distWeight = Weight.FromLogValue(transition.ElementDistribution.GetLogProb(element));
                            if (!distWeight.IsZero && !transition.Weight.IsZero)
                            {
                                Weight destValue = destState.DoGetValue(sequence, sequencePosition + 1, valueCache);
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
                    foreach (var state in Owner.States)
                    {
                        var trans = state.GetTransitions();
                        for (int i = 0; i < trans.Length; i++)
                        {
                            if (trans[i].DestinationStateIndex == Index) return true;
                        }
                    }
                    return false;
                }
            }


            /// <summary>
            /// Writes the automaton state.
            /// </summary>
            public void Write(Action<int> writeInt32, Action<double> writeDouble, Action<TElementDistribution> writeElementDistribution)
            {
                this.EndWeight.Write(writeDouble);
                writeInt32(this.Index);
                writeInt32(this.transitionCount);
                for (var i = 0; i < transitionCount; i++)
                {
                    transitions[i].Write(writeInt32, writeDouble, writeElementDistribution);
                }
            }

            /// <summary>
            /// Reads the automaton state.
            /// </summary>
            public static State Read(Func<int> readInt32, Func<double> readDouble, Func<TElementDistribution> readElementDistribution)
            {
                var res = new State();
                res.EndWeight = Weight.Read(readDouble);
                res.Index = readInt32();
                var transitionCount = readInt32();

                var transitionLength = res.transitions.Length;
                while (transitionLength < transitionCount)
                {
                    transitionLength <<= 1;
                }
                
                var transitions = transitionLength == res.transitions.Length ? res.transitions : new Transition[transitionLength];
                for (var i = 0; i < transitionCount; i++)
                {
                    transitions[i] = Transition.Read(readInt32, readDouble, readElementDistribution);
                }

                res.transitionCount = transitionCount;
                res.transitions = transitions;

                return res;
            }
        }
    }
}
