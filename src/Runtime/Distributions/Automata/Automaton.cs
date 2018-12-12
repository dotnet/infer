// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Collections.Specialized;
    using System.Diagnostics;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Abstract base class for a weighted finite state automaton.
    /// It can be viewed as a function that maps arbitrary sequences of elements to real values.
    /// </summary>
    /// <remarks>
    /// An automaton represented by this class has the following properties:
    /// <list type="bullet">
    /// <item><description>
    /// Its states and transitions form a directed graph rooted at the <see cref="Start"/>;
    /// </description></item>
    /// <item><description>
    /// Each state has an associated cost for being an accepting ("end") state;
    /// </description></item>
    /// <item><description>
    /// Each transition has an associated cost for using it;
    /// </description></item>
    /// <item><description>
    /// A transition can have an associated distribution over elements. In that case an additional cost is being
    /// paid for using that transition with a specific element. The cost is equal to the log-probability of that element
    /// under the distribution;
    /// </description></item>
    /// <item><description>
    /// If a transition doesn't have an associated distribution over elements, it is an epsilon transition
    /// which can be used with no element only.
    /// </description></item>
    /// </list>
    /// </remarks>
    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The type of a sequence element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
    /// <typeparam name="TThis">The type of a concrete automaton class.</typeparam>
    [Quality(QualityBand.Experimental)]
    [Serializable]
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> : ISerializable
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        #region Fields & constants

        /// <summary>
        /// The maximum number of states an automaton can have.
        /// </summary>
        private static int maxStateCount = 50000;

        /// <summary>
        /// Whether to use the Regex builder for the ToString method.
        /// </summary>
        private static bool useRegexForToString = false;

        /// <summary>
        /// The maximum number of states an automaton can have
        /// before an attempt to simplify it will be made.
        /// </summary>
        private static int maxStateCountBeforeSimplification = 200;

        /// <summary>
        /// The maximum number of "dead" states (states from which an end state cannot be reached)
        /// an automaton can have before the dead state removal procedure will be run.
        /// </summary>
        private static int maxDeadStateCount = 0;

        //// When adding a new field, make sure to update the SwapWith() function implementation

        /// <summary>
        /// The collection of states.
        /// </summary>
        [DataMember]
        private List<StateData> statesData = new List<StateData>();

        /// <summary>
        /// The start state.
        /// </summary>
        [DataMember]
        private int startStateIndex;

        /// <summary>
        /// Whether the automaton is free of epsilon transition.
        /// If the value of this field is null, it means that the presence of epsilon transitions is unknown.
        /// </summary>
        [DataMember]
        private bool? isEpsilonFree; // TODO: Isn't it always known?

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Automaton{TSequence,TElement,TElementDistribution,TSequenceManipulator,TThis}"/>
        /// class by setting it to be zero everywhere.
        /// </summary>
        protected Automaton()
        {
            // Zero by default
            this.SetToZero();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the sequence manipulator.
        /// </summary>
        public static TSequenceManipulator SequenceManipulator { get; } =
            new TSequenceManipulator();

        /// <summary>
        /// Gets or sets a value that, if not null, will be returned when computing the log value of any sequence
        /// which is in the support of this automaton (i.e. has non-zero value under the automaton).
        /// </summary>
        /// <remarks>
        /// This should only be set non-null if this distribution is improper and there is 
        /// a need to override the actual automaton value. 
        /// </remarks>
        [DataMember]
        public double? LogValueOverride
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets a value for truncating small weights.
        /// If non-null, any transition whose weight falls below this value in a normalized
        /// automaton will be removed following a product operation.
        /// </summary>
        /// <remarks>
        /// TODO: We need to develop more elegant automaton approximation methods, this is a simple placeholder for those.
        /// </remarks>
        [DataMember]
        public double? PruneTransitionsWithLogWeightLessThan
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the maximum number of states an automaton can have.
        /// </summary>
        public static int MaxStateCount
        {
            get => maxStateCount;

            set
            {
                Argument.CheckIfInRange(value > 0, nameof(value), "The maximum number of states must be positive.");
                maxStateCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the maximum number of states an automaton can have
        /// before an attempt to simplify it will be made.
        /// </summary>
        public static int MaxStateCountBeforeSimplification
        {
            get => maxStateCountBeforeSimplification;

            set
            {
                Argument.CheckIfInRange(value > 0, nameof(value), "The maximum number of states before simplification must be positive.");
                maxStateCountBeforeSimplification = value;
            }
        }

        /// <summary>
        /// Gets or sets the maximum number of "dead" states (states from which an end state cannot be reached)
        /// an automaton can have before the dead state removal procedure will be run.
        /// </summary>
        public static int MaxDeadStateCount
        {
            get => maxDeadStateCount;

            set
            {
                Argument.CheckIfInRange(value >= 0, nameof(value), "The maximum number of dead states should be non-negative.");
                maxDeadStateCount = value;
            }
        }

        /// <summary>
        /// Gets the collection of the states of the automaton.
        /// </summary>
        public StateCollection States => new StateCollection(this);

        /// <summary>
        /// Gets or sets the start state of the automaton.
        /// </summary>
        /// <remarks>
        /// Only a state from <see cref="States"/> can be specified as the value of this property. 
        /// </remarks>
        public State Start
        {
            get => new State(this, this.startStateIndex, this.statesData[this.startStateIndex]);

            set
            {
                Argument.CheckIfValid(!value.IsNull, nameof(value));
                Argument.CheckIfValid(ReferenceEquals(value.Owner, this), nameof(value), "The given state does not belong to this automaton.");
                this.startStateIndex = value.Index;
            }
        }

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates an automaton from a given array of states and a start state.
        /// Used by quoting to embed automata in the generated inference code.
        /// </summary>
        /// <remarks>
        /// Only the index of the <paramref name="startState"/> will be used to determine the start state,
        /// not the object itself.
        /// </remarks>
        /// <param name="states">The array of states.</param>
        /// <param name="startState">The start state.</param>
        /// <returns>The created automaton.</returns>
        [Construction("GetStates", "Start")]
        public static TThis FromStates(IEnumerable<State> states, State startState)
        {
            Argument.CheckIfNotNull(states, nameof(states));
            Argument.CheckIfValid(!startState.IsNull, nameof(startState));

            CheckStateConsistency(states, startState);

            var result = new TThis();
            result.SetStates(states.Select(state => state.Data), startState.Index);
            return result;
        }

        /// <summary>
        /// Creates an automaton which maps every sequence to zero.
        /// </summary>
        /// <returns>The created automaton.</returns>
        public static TThis Zero()
        {
            var result = new TThis();
            return result;
        }

        /// <summary>
        /// Creates an automaton which maps every sequence to a given value.
        /// </summary>
        /// <param name="value">The value to map every sequence to.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Constant(double value)
        {
            return Constant(value, Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Creates an automaton which maps every allowed sequence to a given value and maps all other sequences to zero.
        /// A sequence is allowed if all its elements have non-zero probability under a given distribution.
        /// </summary>
        /// <param name="value">The value to map every sequence to.</param>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Constant(double value, TElementDistribution allowedElements)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            return ConstantLog(Math.Log(value), allowedElements);
        }

        /// <summary>
        /// Creates an automaton which maps every sequence to a given value.
        /// </summary>
        /// <param name="logValue">The logarithm of the value to map every sequence to.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantLog(double logValue)
        {
            return ConstantLog(logValue, Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Creates an automaton which maps every allowed sequence to a given value and maps all other sequences to zero.
        /// A sequence is allowed if all its elements have non-zero probability under a given distribution.
        /// </summary>
        /// <param name="logValue">The logarithm of the value to map every sequence to.</param>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantLog(double logValue, TElementDistribution allowedElements)
        {
            TThis result = new TThis();
            result.SetToConstantLog(logValue, allowedElements);
            return result;
        }

        /// <summary>
        /// Creates an automaton which has a given value on the sequence consisting of a given element only
        /// and is zero everywhere else.
        /// </summary>
        /// <param name="logValue">The logarithm of the value of the automaton on the sequence consisting of the given element only.</param>
        /// <param name="element">The element.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnElementLog(double logValue, TElement element)
        {
            var allowedElements = new TElementDistribution
            {
                Point = element
            };
            return ConstantOnElementLog(logValue, allowedElements);
        }

        /// <summary>
        /// Creates an automaton which has a given value on the sequence consisting of a single element from a set of allowed elements
        /// and is zero everywhere else. An element is allowed if it has non-zero probability under a given distribution.
        /// </summary>
        /// <param name="logValue">The logarithm of the value of the automaton on the sequence consisting of the given element only.</param>
        /// <param name="allowedElements">The distribution representing allowed elements.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnElementLog(double logValue, TElementDistribution allowedElements)
        {
            Argument.CheckIfNotNull(allowedElements, "allowedElements");

            TThis result = Zero();
            if (!double.IsNegativeInfinity(logValue))
            {
                allowedElements = Distribution.CreatePartialUniform(allowedElements);
                State finish = result.Start.AddTransition(allowedElements, Weight.FromLogValue(-allowedElements.GetLogAverageOf(allowedElements)));
                finish.SetEndWeight(Weight.FromLogValue(logValue));
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton which has a given value on a given sequence and is zero everywhere else.
        /// </summary>
        /// <param name="logValue">The logarithm of the value of the automaton on the given sequence.</param>
        /// <param name="sequence">The sequence.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnLog(double logValue, TSequence sequence)
        {
            Argument.CheckIfNotNull(sequence, "sequence");
            return ConstantOnLog(logValue, new[] { sequence });
        }

        /// <summary>
        /// Creates an automaton which has a given value on given sequences and is zero everywhere else.
        /// </summary>
        /// <param name="logValue">The logarithm of the value of the automaton on the given sequences.</param>
        /// <param name="sequences">The sequences.</param>
        /// <remarks>
        /// If the same sequence is specified multiple times, the value of the created automaton on that sequence
        /// will be scaled by the number of the sequence occurrences.
        /// </remarks>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnLog(double logValue, params TSequence[] sequences)
        {
            return ConstantOnLog(logValue, (IEnumerable<TSequence>)sequences);
        }

        /// <summary>
        /// Creates an automaton which has a given value on given sequences and is zero everywhere else.
        /// </summary>
        /// <param name="logValue">The logarithm of the value of the automaton on the given sequences.</param>
        /// <param name="sequences">The sequences.</param>
        /// <remarks>
        /// If the same sequence is specified multiple times, the value of the created automaton on that sequence
        /// will be scaled by the number of the sequence occurrences.
        /// </remarks>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnLog(double logValue, IEnumerable<TSequence> sequences)
        {
            Argument.CheckIfNotNull(sequences, "sequences");

            TThis result = Zero();
            if (!double.IsNegativeInfinity(logValue))
            {
                foreach (TSequence sequence in sequences)
                {
                    State sequenceEndState = result.Start.AddTransitionsForSequence(sequence);
                    sequenceEndState.SetEndWeight(Weight.FromLogValue(logValue));
                }
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton which has a given value on the sequence consisting of a given element only
        /// and is zero everywhere else.
        /// </summary>
        /// <param name="value">The value of the automaton on the sequence consisting of the given element only.</param>
        /// <param name="element">The element.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnElement(double value, TElement element)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            return ConstantOnElementLog(Math.Log(value), element);
        }

        /// <summary>
        /// Creates an automaton which has a given value on the sequence consisting of a single element from a set of allowed elements
        /// and is zero everywhere else. An element is allowed if it has non-zero probability under a given distribution.
        /// </summary>
        /// <param name="value">The value of the automaton on the sequence consisting of the given element only.</param>
        /// <param name="allowedElements">The distribution representing allowed elements.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOnElement(double value, TElementDistribution allowedElements)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            return ConstantOnElementLog(Math.Log(value), allowedElements);
        }

        /// <summary>
        /// Creates an automaton which has a given value on a given sequence and is zero everywhere else.
        /// </summary>
        /// <param name="value">The value of the automaton on the given sequence.</param>
        /// <param name="sequence">The sequence.</param>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOn(double value, TSequence sequence)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            return ConstantOnLog(Math.Log(value), sequence);
        }

        /// <summary>
        /// Creates an automaton which has a given value on the empty sequence and is zero everywhere else.
        /// </summary>
        /// <param name="value">The value of the automaton on the empty sequence.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Empty(double value = 1.0)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            TThis result = Zero();
            result.Start.SetEndWeight(Weight.FromLogValue(Math.Log(value)));
            return result;
        }

        /// <summary>
        /// Creates an automaton which has a given value on given sequences and is zero everywhere else.
        /// </summary>
        /// <param name="value">The value of the automaton on the given sequences.</param>
        /// <param name="sequences">The sequences.</param>
        /// <remarks>
        /// If the same sequence is specified multiple times, the value of the created automaton on that sequence
        /// will be scaled by the number of the sequence occurrences.
        /// </remarks>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOn(double value, params TSequence[] sequences)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            return ConstantOnLog(Math.Log(value), sequences);
        }

        /// <summary>
        /// Creates an automaton which has a given value on given sequences and is zero everywhere else.
        /// </summary>
        /// <param name="value">The value of the automaton on the given sequences.</param>
        /// <param name="sequences">The sequences.</param>
        /// <remarks>
        /// If the same sequence is specified multiple times, the value of the created automaton on that sequence
        /// will be scaled by the number of the sequence occurrences.
        /// </remarks>
        /// <returns>The created automaton.</returns>
        public static TThis ConstantOn(double value, IEnumerable<TSequence> sequences)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            return ConstantOnLog(Math.Log(value), sequences);
        }

        /// <summary>
        /// Creates an automaton which is a weighted sum of a given pair of automata.
        /// </summary>
        /// <param name="weight1">The weight of the first automaton.</param>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="weight2">The weight of the second automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <returns>The created automaton.</returns>
        public static TThis WeightedSum(double weight1, TThis automaton1, double weight2, TThis automaton2)
        {
            if (weight1 < 0 || weight2 < 0)
            {
                throw new NotImplementedException("Negative weights are not yet supported.");
            }

            return WeightedSumLog(Math.Log(weight1), automaton1, Math.Log(weight2), automaton2);
        }

        /// <summary>
        /// Creates an automaton which is a weighted sum of a given pair of automata.
        /// </summary>
        /// <param name="logWeight1">The logarithm of the weight of the first automaton.</param>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="logWeight2">The logarithm of the weight of the second automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <returns>The created automaton.</returns>
        public static TThis WeightedSumLog(double logWeight1, TThis automaton1, double logWeight2, TThis automaton2)
        {
            var result = new TThis();
            result.SetToSumLog(logWeight1, automaton1, logWeight2, automaton2);
            return result;
        }

        /// <summary>
        /// Creates an automaton which is a sum of given automata.
        /// </summary>
        /// <param name="automata">The automata to sum.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Sum(params TThis[] automata)
        {
            return Sum((IEnumerable<TThis>)automata);
        }

        /// <summary>
        /// Creates an automaton which is a sum of given automata.
        /// </summary>
        /// <param name="automata">The automata to sum.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Sum(IEnumerable<TThis> automata)
        {
            Argument.CheckIfNotNull(automata, "automata");

            TThis result = Zero();
            foreach (TThis automaton in automata)
            {
                if (automaton.IsCanonicZero())
                {
                    continue;
                }

                int index = result.statesData.Count;
                result.AddStates(automaton.statesData);
                result.Start.AddEpsilonTransition(Weight.One, result.States[index + automaton.Start.Index]);
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton which is the product of given automata.
        /// </summary>
        /// <param name="automata">The automata to multiply.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Product(params TThis[] automata)
        {
            return Product((IEnumerable<TThis>)automata);
        }

        /// <summary>
        /// Creates an automaton which is the product of given automata.
        /// </summary>
        /// <param name="automata">The automata to multiply.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Product(IEnumerable<TThis> automata)
        {
            Argument.CheckIfNotNull(automata, "automata");

            // TODO: can we adjust the multiplication order to improve performance?
            TThis result = Constant(1.0);
            foreach (TThis automaton in automata)
            {
                if (automaton.IsCanonicZero())
                {
                    result.SetToZero();
                    return result;
                }

                result.SetToProduct(result, automaton);
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton which has given values on given sequences and is zero everywhere else.
        /// </summary>
        /// <param name="sequenceToValue">The collection of pairs of a sequence and the automaton value on that sequence.</param>
        /// <remarks>
        /// If the same sequence is presented in the collection of pairs multiple times,
        /// the value of the automaton on that sequence will be equal to the sum of the values in the collection.
        /// </remarks>
        /// <returns>The created automaton.</returns>
        public static TThis FromValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceToValue)
        {
            Argument.CheckIfNotNull(sequenceToValue, "sequenceToValue");

            TThis result = Zero();
            foreach (KeyValuePair<TSequence, double> sequenceWithValue in sequenceToValue)
            {
                result = Sum(result, ConstantOn(sequenceWithValue.Value, sequenceWithValue.Key));
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton <c>g(s) = sum_k v(k) sum_{t1 t2 ... tk = s} f(t1)f(t2)...f(tk)</c>,
        /// where <c>f(t)</c> is the given automaton, and <c>v(k)</c> is a weight function given as a vector.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <param name="repetitionNumberWeights">The weight vector.</param>
        /// <returns>The created automaton.</returns>
        /// <remarks>
        /// The result is a weighted sum of Cauchy products of the given automaton with itself,
        /// each product having a different number of factors.
        /// </remarks>
        public static TThis Repeat(TThis automaton, Vector repetitionNumberWeights)
        {
            Argument.CheckIfNotNull(automaton, "automaton");
            Argument.CheckIfNotNull(repetitionNumberWeights, "repetitionNumberWeights");

            List<ValueAtIndex<double>> nonZeroRepetitionWeights = repetitionNumberWeights.FindAll(w => w > 0).ToList();

            if (automaton.IsCanonicZero() || nonZeroRepetitionWeights.Count == 0)
            {
                return Zero();
            }

            int maxTimes = nonZeroRepetitionWeights[nonZeroRepetitionWeights.Count - 1].Index;
            TThis result = ConstantOn(1.0, SequenceManipulator.ToSequence(new TElement[0]));

            // Build a list of all intermediate end states with their target ending weights while adding repetitions
            var endStatesWithTargetWeights = new List<(State, Weight)>();
            int prevStateCount = 0;
            for (int i = 0; i <= maxTimes; ++i)
            {
                // Remember added ending states
                if (repetitionNumberWeights[i] > 0)
                {
                    for (int j = prevStateCount; j < result.statesData.Count; ++j)
                    {
                        if (result.statesData[j].CanEnd)
                        {
                            endStatesWithTargetWeights.Add(ValueTuple.Create(
                                result.States[j],
                                Weight.Product(Weight.FromValue(repetitionNumberWeights[i]), result.statesData[j].EndWeight)));
                        }
                    }
                }

                // Add one more repetition
                if (i != maxTimes)
                {
                    prevStateCount = result.statesData.Count;
                    result.AppendInPlaceNoOptimizations(automaton);
                }
            }

            // Set target ending weights
            for (int i = 0; i < endStatesWithTargetWeights.Count; ++i)
            {
                var (state, weight) = endStatesWithTargetWeights[i];
                state.SetEndWeight(weight);
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton <c>g(s) = sum_{k=Kmin}^{Kmax} sum_{t1 t2 ... tk = s} f(t1)f(t2)...f(tk)</c>,
        /// where <c>f(t)</c> is the given automaton, and <c>Kmin</c> and <c>Kmax</c> are the minimum
        /// and the maximum number of factors in a sum term.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <param name="minTimes">The minimum number of factors in a sum term. Defaults to 1.</param>
        /// <param name="maxTimes">An optional maximum number of factors in a sum term.</param>
        /// <returns>The created automaton.</returns>
        /// <remarks>
        /// The result is the sum of Cauchy products of the given automaton with itself,
        /// each product having a different number of factors.
        /// </remarks>
        public static TThis Repeat(TThis automaton, int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfNotNull(automaton, "automaton");
            Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of automaton repetitions must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || maxTimes.Value >= minTimes, "The maximum number of automaton repetitions must not be less than the minimum number.");

            if (automaton.IsCanonicZero())
            {
                return Zero();
            }

            if (maxTimes.HasValue)
            {
                PiecewiseVector repetitionNumberWeights = PiecewiseVector.Zero(maxTimes.Value + 1);
                repetitionNumberWeights.Pieces.Add(new ConstantVector(minTimes, maxTimes.Value, 1));
                return Repeat(automaton, repetitionNumberWeights);
            }

            TSequence emptySequence = SequenceManipulator.ToSequence(new TElement[0]);

            TThis result = ConstantOn(1.0, emptySequence);
            for (int i = 0; i < minTimes; ++i)
            {
                result.AppendInPlace(automaton);
            }

            TThis optionalPart = automaton.Clone();
            for (int i = 0; i < optionalPart.statesData.Count; ++i)
            {
                if (optionalPart.statesData[i].CanEnd)
                {
                    optionalPart.States[i].AddEpsilonTransition(optionalPart.States[i].EndWeight, optionalPart.Start);
                }
            }

            result.AppendInPlace(Sum(optionalPart, ConstantOn(1.0, emptySequence)));

            return result;
        }

        #endregion

        #region ToString

        /// <summary>
        /// Returns a string that represents the automaton.
        /// </summary>
        /// <returns>A string that represents the automaton.</returns>
        public override string ToString()
        {
            if (useRegexForToString)
            {
                return this.ToString(AutomatonFormats.Friendly);
            }
            else
            {
                if (this is StringAutomaton)
                {
                    Action<DiscreteChar, StringBuilder> appendElement = (eltDist, stringBuilder) => eltDist.AppendRegex(stringBuilder, true);
                    return this.ToString(appendElement as Action<TElementDistribution, StringBuilder>);
                }
                else
                {
                    return this.ToString((Action<TElementDistribution, StringBuilder>)null);
                }
            }
        }

        /// <summary>
        /// Returns a string that represents the automaton.
        /// </summary>
        /// <param name="appendElement">Optional method for appending at the element distribution level.</param>
        /// <returns>A string that represents the automaton.</returns>
        public string ToString(Action<TElementDistribution, StringBuilder> appendElement)
        {
            if (useRegexForToString)
            {
                return this.ToString();
            }
            else
            {
                StringBuilder builder = new StringBuilder();
                this.AppendString(builder, new HashSet<int>(), this.Start.Index, appendElement);
                return builder.ToString();
            }
        }

        /// <summary>
        /// Returns a string that represents the automaton.
        /// </summary>
        /// <param name="format">The format.</param>
        /// <returns>A string that represents the automaton.</returns>
        public string ToString(IAutomatonFormat format)
        {
            Argument.CheckIfNotNull(format, "format");

            return format.ConvertToString(this);
        }

        #endregion

        #region Group support

        /// <summary>
        /// Determines whether this automaton has the specified group.
        /// </summary>
        /// <param name="group">The specified group.</param>
        /// <returns>True if it the automaton has this group, false otherwise.</returns>
        public bool HasGroup(int group)
        {
            for (int stateIndex = 0; stateIndex < this.statesData.Count; stateIndex++)
            {
                var state = this.statesData[stateIndex];
                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; transitionIndex++)
                {
                    Transition transition = state.GetTransition(transitionIndex);
                    if (transition.Group == group)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Determines whether this automaton has groups.
        /// </summary>
        /// <returns>True if it the automaton has groups, false otherwise.</returns>
        public bool UsesGroups()
        {
            for (int stateIndex = 0; stateIndex < this.statesData.Count; stateIndex++)
            {
                var state = this.statesData[stateIndex];
                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; transitionIndex++)
                {
                    Transition transition = state.GetTransition(transitionIndex);
                    if (transition.Group != 0)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        public Dictionary<int, TThis> GetGroups() => GroupExtractor.ExtractGroups(this);

        /// <summary>
        /// Clears the group for all transitions.
        /// </summary>
        public void ClearGroups()
        {
            this.SetGroup(0);
        }

        /// <summary>
        /// Sets all transitions to have the specified group.
        /// </summary>
        /// <param name="group">The specified group.</param>
        public void SetGroup(int group)
        {
            for (int stateIndex = 0; stateIndex < this.statesData.Count; stateIndex++)
            {
                var state = this.statesData[stateIndex];
                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; transitionIndex++)
                {
                    Transition transition = state.GetTransition(transitionIndex);
                    transition.Group = group;
                    state.SetTransition(transitionIndex, transition);
                }
            }
        }

        #endregion

        #region Normalization

        /// <summary>
        /// Computes the logarithm of the normalizer (sum of values of the automaton on all sequences).
        /// </summary>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <remarks>Returns <see cref="double.PositiveInfinity"/> if the sum diverges.</remarks>
        public double GetLogNormalizer()
        {
            return this.DoGetLogNormalizer(false);
        }

        /// <summary>
        /// Normalizes the automaton so that the sum of its values over all possible sequences equals to one
        /// and returns the logarithm of the normalizer.
        /// </summary>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the automaton cannot be normalized (i.e. if the normalizer is zero or positive infinity).</exception>
        /// <remarks>The only automaton which cannot be normalized, but has a finite normalizer, is zero.</remarks>
        public double NormalizeValues()
        {
            double result;
            if (!this.TryNormalizeValues(out result))
            {
                throw new InvalidOperationException("This automaton cannot be normalized.");
            }

            return result;
        }

        /// <summary>
        /// Attempts to normalize the automaton so that sum of its values on all possible sequences equals to one (if it is possible).
        /// </summary>
        /// <param name="logNormalizer">When the function returns, contains the logarithm of the normalizer.</param>
        /// <returns><see langword="true"/> if the automaton was successfully normalized, <see langword="false"/> otherwise.</returns>
        public bool TryNormalizeValues(out double logNormalizer)
        {
            logNormalizer = this.DoGetLogNormalizer(true);
            return !double.IsInfinity(logNormalizer);
        }

        /// <summary>
        /// Attempts to normalize the automaton so that sum of its values on all possible sequences equals to one (if it is possible).
        /// </summary>
        /// <returns><see langword="true"/> if the automaton was successfully normalized, <see langword="false"/> otherwise.</returns>>
        public bool TryNormalizeValues()
        {
            double logNormalizer;
            return this.TryNormalizeValues(out logNormalizer);
        }

        #endregion

        #region Operations

        /// <summary>
        /// Checks whether the automaton is zero on all sequences.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the automaton is zero on all sequences,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>
        /// The time complexity of this function is not constant,
        /// so it should not be used when treating zero specially for performance reasons.
        /// Use <see cref="IsCanonicZero"/> instead.
        /// </remarks>
        public bool IsZero()
        {
            return this.IsCanonicZero() || this.Start.IsZero();
        }

        /// <summary>
        /// Checks whether the automaton is a canonic representation of zero,
        /// as produced by <see cref="SetToZero"/> and <see cref="Zero"/>.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the automaton is a canonic representation of zero,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>
        /// The time complexity of this function is O(1), so it can be used to treat zero specially in performance-critical code.
        /// All the operations on automata resulting in zero produce the canonic representation.
        /// </remarks>
        public bool IsCanonicZero()
        {
            return this.Start.TransitionCount == 0 && this.Start.EndWeight.IsZero;
        }

        /// <summary>
        /// Checks whether the automaton is a canonic representation of a constant,
        /// as produced by <see cref="SetToConstantLog(double)"/>.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the automaton is a canonic representation of the constant,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>
        /// The time complexity of this function is O(1), so it can be used to treat constants specially in performance-critical code.
        /// </remarks>
        public bool IsCanonicConstant()
        {
            if (this.statesData.Count != 1 || this.Start.TransitionCount != 1 || !this.Start.CanEnd)
            {
                return false;
            }

            var transitionDistribution = this.Start.GetTransition(0).ElementDistribution;
            return transitionDistribution.HasValue && transitionDistribution.Value.IsUniform();
        }

        /// <summary>
        /// Checks whether the automaton has non-trivial loops (i.e. loops consisting of more than one edge).
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the automaton has non-trivial loops,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public bool HasNonTrivialLoops()
        {
            return this.Start.HasNonTrivialLoops();
        }

        /// <summary>
        /// Sets the automaton to be constant on the support of a given automaton.
        /// </summary>
        /// <param name="value">The desired value on the support of <paramref name="automaton"/>.</param>
        /// <param name="automaton">The automaton.</param>
        /// <remarks>This function will fail if the <paramref name="automaton"/> cannot be determinized.</remarks>
        public void SetToConstantOnSupportOf(double value, TThis automaton)
        {
            if (value < 0)
            {
                throw new NotSupportedException("Negative values are not yet supported.");
            }

            this.SetToConstantOnSupportOfLog(Math.Log(value), automaton);
        }

        /// <summary>
        /// Sets the automaton to be constant on the support of a given automaton.
        /// </summary>
        /// <param name="logValue">The logarithm of the desired value on the support of <paramref name="automaton"/>.</param>
        /// <param name="automaton">The automaton.</param>
        /// <remarks>This function will fail if the <paramref name="automaton"/> cannot be determinized.</remarks>
        public void SetToConstantOnSupportOfLog(double logValue, TThis automaton)
        {
            Argument.CheckIfNotNull(automaton, "automaton");
            Argument.CheckIfValid(!double.IsNaN(logValue), "logValue", "A valid value must be provided.");

            Weight value = Weight.FromLogValue(logValue);
            if (value.IsZero)
            {
                this.SetToZero();
                return;
            }

            TThis result = automaton.Clone();
            if (!result.TryDeterminize())
            {
                throw new NotImplementedException("Not yet supported for non-determinizable automata.");
            }

            for (int stateId = 0; stateId < result.States.Count; ++stateId)
            {
                var state = result.States[stateId];
                if (state.CanEnd)
                {
                    // Make all accepting states contibute the desired value to the result
                    state.SetEndWeight(value);
                }

                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                {
                    // Make all non-zero transition weights unit
                    var transition = state.GetTransition(transitionIndex);
                    if (!transition.Weight.IsZero)
                    {
                        if (transition.IsEpsilon)
                        {
                            transition.Weight = Weight.One;
                        }
                        else
                        {
                            transition.ElementDistribution = Distribution.CreatePartialUniform(transition.ElementDistribution.Value);
                            transition.Weight = Weight.FromLogValue(-transition.ElementDistribution.Value.GetLogAverageOf(transition.ElementDistribution.Value));
                        }

                        state.SetTransition(transitionIndex, transition);
                    }
                }
            }

            this.SwapWith(result);
        }

        /// <summary>
        /// Creates a copy of the automaton.
        /// </summary>
        /// <returns>The created copy.</returns>
        public TThis Clone()
        {
            var result = new TThis();
            result.SetTo((TThis)this);
            return result;
        }

        /// <summary>
        /// Creates an automaton <c>f'(s) = f(reverse(s))</c>, 
        /// where <c>reverse</c> is a sequence reverse function.
        /// </summary>
        /// <returns>The created automaton.</returns>
        public TThis Reverse()
        {
            var result = Zero();

            // Result already has 1 state, we add the remaining Count-1 states
            result.AddStates(this.statesData.Count - 1);

            // And the new start state
            result.Start = result.AddState();

            // The start state in the original automaton is going to be the one and only end state in result
            result.States[this.Start.Index].SetEndWeight(Weight.One);

            for (int i = 0; i < this.statesData.Count; ++i)
            {
                var oldState = this.statesData[i];
                for (int j = 0; j < this.statesData[i].TransitionCount; ++j)
                {
                    // Result has original transitions reversed
                    var oldTransition = oldState.GetTransition(j);
                    result.States[oldTransition.DestinationStateIndex].AddTransition(
                        oldTransition.ElementDistribution, oldTransition.Weight, result.States[i]);
                }

                // End states of the original automaton are the new start states
                if (oldState.CanEnd)
                {
                    result.Start.AddEpsilonTransition(oldState.EndWeight, result.States[i]);
                }
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton <c>f'(st) = f(s)</c>, where <c>f(s)</c> is the current automaton
        /// and <c>t</c> is the given sequence.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="group">The group.</param>
        /// <returns>The created automaton.</returns>
        public TThis Append(TSequence sequence, int group = 0)
        {
            return this.Append(ConstantOn(1.0, sequence), group);
        }

        /// <summary>
        /// Creates an automaton <c>f'(s) = sum_{tu=s} f(t)g(u)</c>, where <c>f(t)</c> is the current automaton
        /// and <c>g(u)</c> is the given automaton.
        /// The resulting automaton is also known as the Cauchy product of two automata.
        /// </summary>
        /// <param name="automaton">The automaton to append.</param>
        /// <param name="group">The group.</param>
        /// <returns>The created automaton.</returns>
        public TThis Append(TThis automaton, int group = 0)
        {
            TThis result = this.Clone();
            result.AppendInPlace(automaton, group);
            return result;
        }

        /// <summary>
        /// Replaces the current automaton with an automaton <c>f'(st) = f(s)</c>, where <c>f(s)</c> is the current automaton
        /// and <c>t</c> is the given sequence.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="group">The group.</param>
        public void AppendInPlace(TSequence sequence, int group = 0)
        {
            this.AppendInPlace(ConstantOn(1.0, sequence), group);
        }

        /// <summary>
        /// Replaces the current automaton with an automaton <c>f'(s) = sum_{tu=s} f(t)g(u)</c>,
        /// where <c>f(t)</c> is the current automaton and <c>g(u)</c> is the given automaton.
        /// The resulting automaton is also known as the Cauchy product of two automata.
        /// </summary>
        /// <param name="automaton">The automaton to append.</param>
        /// <param name="group">The group.</param>
        public void AppendInPlace(TThis automaton, int group = 0)
        {
            Argument.CheckIfNotNull(automaton, "automaton");

            if (this.IsCanonicZero())
            {
                return;
            }

            if (automaton.IsCanonicZero())
            {
                this.SetToZero();
                return;
            }

            if (ReferenceEquals(automaton, this))
            {
                automaton = automaton.Clone();
            }

            // Append the states of the second automaton
            var endStates = this.States.Where(nd => nd.CanEnd).ToList();
            int stateCount = this.statesData.Count;

            this.AddStates(automaton.statesData, group);
            var secondStartState = this.States[stateCount + automaton.Start.Index];

            // todo: make efficient
            bool startIncoming = automaton.Start.HasIncomingTransitions;
            if (!startIncoming || endStates.All(endState => (endState.TransitionCount == 0)))
            {
                foreach (var endState in endStates)
                {
                    for (int transitionIndex = 0; transitionIndex < secondStartState.TransitionCount; transitionIndex++)
                    {
                        var transition = secondStartState.GetTransition(transitionIndex);

                        if (group != 0)
                        {
                            transition.Group = group;
                        }

                        if (transition.DestinationStateIndex == secondStartState.Index)
                        {
                            transition.DestinationStateIndex = endState.Index;
                        }
                        else
                        {
                            transition.Weight = Weight.Product(transition.Weight, endState.EndWeight);
                        }

                        endState.Data.AddTransition(transition);
                    }

                    endState.SetEndWeight(Weight.Product(endState.EndWeight, secondStartState.EndWeight));
                }

                this.RemoveState(secondStartState.Index);
                return;
            }

            for (int i = 0; i < endStates.Count; i++)
            {
                State state = endStates[i];
                state.AddEpsilonTransition(state.EndWeight, secondStartState, group);
                state.SetEndWeight(Weight.Zero);
            }
        }

        /// <summary>
        /// Computes the product of the current automaton and a given one.
        /// </summary>
        /// <param name="automaton">The automaton to compute the product with.</param>
        /// <param name="tryDeterminize">Whether to try to determinize the result.</param>
        /// <returns>The computed product.</returns>
        public TThis Product(TThis automaton, bool tryDeterminize = true)
        {
            Argument.CheckIfNotNull(automaton, "automaton");

            var result = new TThis();
            result.SetToProduct((TThis)this, automaton, tryDeterminize);
            return result;
        }

        /// <summary>
        /// Replaces the current automaton by the product of a given pair of automata.
        /// </summary>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        public void SetToProduct(TThis automaton1, TThis automaton2)
        {
            this.SetToProduct(automaton1, automaton2, true);
        }

        /// <summary>
        /// Replaces the current automaton by the product of a given pair of automata.
        /// </summary>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <param name="tryDeterminize">When to try to dterminize the product.</param>
        public void SetToProduct(TThis automaton1, TThis automaton2, bool tryDeterminize)
        {
            Argument.CheckIfNotNull(automaton1, "automaton1");
            Argument.CheckIfNotNull(automaton2, "automaton2");

            TThis result = Zero();
            if (!automaton1.IsCanonicZero() && !automaton2.IsCanonicZero())
            {
                if (automaton1.UsesGroups())
                {
                    // We cannot swap automaton 1 and automaton 2 as groups from first are used.
                    if (!automaton2.IsEpsilonFree)
                    {
                        automaton2.MakeEpsilonFree();
                    }
                }
                else
                {
                    // The second argument of BuildProduct must be epsilon-free
                    if (!automaton1.IsEpsilonFree && !automaton2.IsEpsilonFree)
                    {
                        automaton2.MakeEpsilonFree();
                    }
                    else if (automaton1.IsEpsilonFree && !automaton2.IsEpsilonFree)
                    {
                        Util.Swap(ref automaton1, ref automaton2);
                    }
                }

                var stateCache = new Dictionary<(int, int), State>(automaton1.States.Count + automaton2.States.Count);
                result.Start = result.BuildProduct(automaton1.Start, automaton2.Start, stateCache);

                result.RemoveDeadStates(); // Product can potentially create dead states
                result.PruneTransitionsWithLogWeightLessThan = this.MergePruningWeights(automaton1, automaton2);
                result.SimplifyIfNeeded();
                result.LogValueOverride = this.MergeLogValueOverrides(automaton1, automaton2);

                if (this is StringAutomaton && tryDeterminize)
                {
                    result.TryDeterminize();
                }
            }

            this.SwapWith(result);
        }

        /// <summary>
        /// Merges the pruning thresholds for two automata.
        /// </summary>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <returns>The merged pruning threshold.</returns>
        private double? MergePruningWeights(TThis automaton1, TThis automaton2)
        {
            if (automaton1.PruneTransitionsWithLogWeightLessThan == null)
            {
                return automaton2.PruneTransitionsWithLogWeightLessThan;
            }

            if (automaton2.PruneTransitionsWithLogWeightLessThan == null)
            {
                return automaton1.PruneTransitionsWithLogWeightLessThan;
            }

            return Math.Min(automaton1.PruneTransitionsWithLogWeightLessThan.Value, automaton2.PruneTransitionsWithLogWeightLessThan.Value);
        }

        /// <summary>
        /// Merges the log value overrides for two automata.
        /// </summary>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <returns>The merged log value override.</returns>
        private double? MergeLogValueOverrides(TThis automaton1, TThis automaton2)
        {
            if ((automaton1.LogValueOverride == null) && (automaton2.LogValueOverride == null))
            {
                return null;
            }

            double logNorm1 = automaton1.GetLogNormalizer();
            double logNorm2 = automaton2.GetLogNormalizer();
            if (double.IsPositiveInfinity(logNorm1) && double.IsPositiveInfinity(logNorm2))
            {
                if (automaton1.LogValueOverride == null)
                {
                    return automaton2.LogValueOverride;
                }

                if (automaton2.LogValueOverride == null)
                {
                    return automaton1.LogValueOverride;
                }

                return Math.Max(automaton1.LogValueOverride.Value, automaton2.LogValueOverride.Value);
            }

            return null;

            //// todo: is this the correct merging logic here?
        }

        /// <summary>
        /// Computes the sum of the current automaton and a given automaton.
        /// </summary>
        /// <param name="automaton">The automaton to compute the sum with.</param>
        /// <returns>The computed sum.</returns>
        public TThis Sum(TThis automaton)
        {
            Argument.CheckIfNotNull(automaton, "automaton");

            var result = new TThis();
            result.SetToSumLog(0.0, (TThis)this, 0.0, automaton);
            return result;
        }

        /// <summary>
        /// Replaces the current automaton by the weighted sum of a given pair of automata.
        /// </summary>
        /// <param name="weight1">The weight of the first automaton.</param>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="weight2">The weight of the second automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        public void SetToSum(double weight1, TThis automaton1, double weight2, TThis automaton2)
        {
            if (weight1 < 0 || weight2 < 0)
            {
                throw new NotImplementedException("Negative weights are not yet supported.");
            }

            this.SetToSumLog(Math.Log(weight1), automaton1, Math.Log(weight2), automaton2);
        }

        /// <summary>
        /// Replaces the current automaton by the weighted sum of a given pair of automata.
        /// </summary>
        /// <param name="logWeight1">The logarithm of the weight of the first automaton.</param>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="logWeight2">The logarithm of the weight of the second automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        public void SetToSumLog(double logWeight1, TThis automaton1, double logWeight2, TThis automaton2)
        {
            Argument.CheckIfNotNull(automaton1, "automaton1");
            Argument.CheckIfNotNull(automaton2, "automaton2");
            Argument.CheckIfValid(
                !double.IsPositiveInfinity(logWeight1) && !double.IsPositiveInfinity(logWeight2),
                "Weights must not be infinite.");

            TThis result = Zero();

            bool hasFirstTerm = !automaton1.IsCanonicZero() && !double.IsNegativeInfinity(logWeight1);
            bool hasSecondTerm = !automaton2.IsCanonicZero() && !double.IsNegativeInfinity(logWeight2);
            if (hasFirstTerm || hasSecondTerm)
            {
                if (hasFirstTerm)
                {
                    result.AddStates(automaton1.statesData);
                    result.Start.AddEpsilonTransition(Weight.FromLogValue(logWeight1), result.States[1 + automaton1.Start.Index]);
                }

                if (hasSecondTerm)
                {
                    int cnt = result.statesData.Count;
                    result.AddStates(automaton2.statesData);
                    result.Start.AddEpsilonTransition(Weight.FromLogValue(logWeight2), result.States[cnt + automaton2.Start.Index]);
                }
            }

            result.SimplifyIfNeeded();
            this.SwapWith(result);
        }

        /// <summary>
        /// Scales the automaton and returns the result.
        /// </summary>
        /// <param name="scale">The scale.</param>
        /// <returns>The scaled automaton.</returns>
        public TThis Scale(double scale)
        {
            if (scale < 0)
            {
                throw new NotImplementedException("Negative scale is not yet supported.");
            }

            return this.ScaleLog(Math.Log(scale));
        }

        /// <summary>
        /// Scales the automaton and returns the result.
        /// </summary>
        /// <param name="logScale">The logarithm of the scale.</param>
        /// <returns>The scaled automaton.</returns>
        public TThis ScaleLog(double logScale)
        {
            var result = new TThis();
            result.SetToScaleLog((TThis)this, logScale);
            return result;
        }

        /// <summary>
        /// Replaces the current automaton with a given automaton scaled by a given value.
        /// </summary>
        /// <param name="automaton">The automaton to scale.</param>
        /// <param name="scale">The scale.</param>
        public void SetToScale(TThis automaton, double scale)
        {
            if (scale < 0)
            {
                throw new NotImplementedException("Negative scale is not yet supported.");
            }

            this.SetToScaleLog(automaton, Math.Log(scale));
        }

        /// <summary>
        /// Replaces the current automaton with a given automaton scaled by a given value.
        /// </summary>
        /// <param name="automaton">The automaton to scale.</param>
        /// <param name="logScale">The logarithm of the scale.</param>
        public void SetToScaleLog(TThis automaton, double logScale)
        {
            Argument.CheckIfNotNull(automaton, "automaton");
            this.SetToSumLog(logScale, automaton, double.NegativeInfinity, Zero());
        }

        /// <summary>
        /// Replaces the current automaton with an automaton which is zero everywhere.
        /// </summary>
        public void SetToZero()
        {
            this.statesData.Clear();
            this.startStateIndex = this.AddState().Index;
            this.isEpsilonFree = true;
        }

        /// <summary>
        /// Replaces the current automaton with an automaton which maps every sequence to a given value.
        /// </summary>
        /// <param name="value">The value to map every sequence to.</param>
        public void SetToConstant(double value)
        {
            this.SetToConstant(value, Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Replaces the current automaton with an automaton which maps every allowed sequence to
        /// a given value and maps all other sequences to zero.
        /// A sequence is allowed if all its elements have non-zero probability under a given distribution.
        /// </summary>
        /// <param name="value">The value to map every sequence to.</param>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        public void SetToConstant(double value, TElementDistribution allowedElements)
        {
            if (value < 0)
            {
                throw new NotImplementedException("Negative values are not yet supported.");
            }

            this.SetToConstantLog(Math.Log(value), allowedElements);
        }

        /// <summary>
        /// Replaces the current automaton with an automaton which maps every sequence to a given value.
        /// </summary>
        /// <param name="logValue">The logarithm of the value to map every sequence to.</param>
        public void SetToConstantLog(double logValue)
        {
            this.SetToConstantLog(logValue, Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Replaces the current automaton with an automaton which maps every allowed sequence to
        /// a given value and maps all other sequences to zero.
        /// A sequence is allowed if all its elements have non-zero probability under a given distribution.
        /// </summary>
        /// <param name="logValue">The logarithm of the value to map every sequence to.</param>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        public void SetToConstantLog(double logValue, TElementDistribution allowedElements)
        {
            Argument.CheckIfNotNull(allowedElements, "allowedElements");

            allowedElements = Distribution.CreatePartialUniform(allowedElements);
            this.SetToZero();
            if (!double.IsNegativeInfinity(logValue))
            {
                this.Start.SetEndWeight(Weight.FromLogValue(logValue));
                this.Start.AddTransition(allowedElements, Weight.FromLogValue(-allowedElements.GetLogAverageOf(allowedElements)), this.Start);
            }
        }

        /// <summary>
        /// Replaces the current automaton with a copy of a given automaton.
        /// </summary>
        /// <param name="automaton">The automaton to replace the current automaton with.</param>
        public void SetTo(TThis automaton)
        {
            Argument.CheckIfNotNull(automaton, "automaton");

            if (!ReferenceEquals(this, automaton))
            {
                this.SetStates(automaton.statesData, automaton.Start.Index);
                this.isEpsilonFree = automaton.isEpsilonFree;
                this.LogValueOverride = automaton.LogValueOverride;
                this.PruneTransitionsWithLogWeightLessThan = automaton.PruneTransitionsWithLogWeightLessThan;
            }
        }

        /// <summary>
        /// Replaces the current automaton by an automaton obtained by performing a transition transformation
        /// on a source automaton.
        /// </summary>
        /// <typeparam name="TSrcSequence">The type of a source automaton sequence.</typeparam>
        /// <typeparam name="TSrcElement">The type of a source automaton sequence element.</typeparam>
        /// <typeparam name="TSrcElementDistribution">The type of a distribution over source automaton sequence elements.</typeparam>
        /// <typeparam name="TSrcSequenceManipulator">The type providing ways to manipulate source automaton sequences.</typeparam>
        /// <typeparam name="TSrcAutomaton">The type of a source automaton.</typeparam>
        /// <param name="sourceAutomaton">The source automaton.</param>
        /// <param name="transitionTransform">The transition transformation.</param>
        public void SetToFunction<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>(
            Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton> sourceAutomaton,
            Func<Option<TSrcElementDistribution>, Weight, int, Tuple<Option<TElementDistribution>, Weight>> transitionTransform)
            where TSrcElementDistribution : IDistribution<TSrcElement>, CanGetLogAverageOf<TSrcElementDistribution>, SettableToProduct<TSrcElementDistribution>, SettableToWeightedSumExact<TSrcElementDistribution>, SettableToPartialUniform<TSrcElementDistribution>, new()
            where TSrcSequence : class, IEnumerable<TSrcElement>
            where TSrcSequenceManipulator : ISequenceManipulator<TSrcSequence, TSrcElement>, new()
            where TSrcAutomaton : Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>, new()
        {
            Argument.CheckIfNotNull(sourceAutomaton, "sourceAutomaton");
            Argument.CheckIfNotNull(transitionTransform, "transitionTransform");

            // Add states
            this.SetToZero();
            this.AddStates(sourceAutomaton.statesData.Count - 1);

            // Copy state parameters and transitions
            for (int stateIndex = 0; stateIndex < sourceAutomaton.statesData.Count; stateIndex++)
            {
                var thisState = this.States[stateIndex];
                var otherState = sourceAutomaton.States[stateIndex];

                thisState.SetEndWeight(otherState.EndWeight);
                if (otherState == sourceAutomaton.Start)
                {
                    this.Start = thisState;
                }

                for (int transitionIndex = 0; transitionIndex < otherState.TransitionCount; transitionIndex++)
                {
                    var otherTransition = otherState.GetTransition(transitionIndex);
                    var transformedTransition = transitionTransform(otherTransition.ElementDistribution, otherTransition.Weight, otherTransition.Group);
                    this.States[stateIndex].AddTransition(
                        transformedTransition.Item1,
                        transformedTransition.Item2,
                        this.States[otherTransition.DestinationStateIndex],
                        otherTransition.Group);
                }
            }
        }

        /// <summary>
        /// Computes the logarithm of the value of the automaton on a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to compute the value on.</param>
        /// <returns>The logarithm of the value.</returns>
        public double GetLogValue(TSequence sequence)
        {
            Argument.CheckIfNotNull(sequence, "sequence");
            var value = this.Start.GetLogValue(sequence);
            if (!double.IsNegativeInfinity(value) && this.LogValueOverride.HasValue)
            {
                return this.LogValueOverride.Value;
            }

            return value;
        }

        /// <summary>
        /// Computes the value of the automaton on a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to compute the value on.</param>
        /// <returns>The computed value.</returns>
        public double GetValue(TSequence sequence)
        {
            return Math.Exp(this.GetLogValue(sequence));
        }

        /// <summary>
        /// Attempts to find the only sequence which has non-zero value under the automaton.
        /// </summary>
        /// <returns>
        /// The only sequence having non-zero value, if found.
        /// <see langword="null"/>, if the automaton is zero everywhere or is non-zero on more than one sequence.
        /// </returns>
        public TSequence TryComputePoint()
        {
            bool[] endNodeReachability = this.ComputeEndStateReachability();
            if (!endNodeReachability[this.Start.Index])
            {
                return null;
            }

            var point = new List<TElement>();
            int? pointLength = null;
            var stateDepth = new ArrayDictionary<int>(this.statesData.Count);
            bool isPoint = this.TryComputePointDfs(this.Start, 0, stateDepth, endNodeReachability, point, ref pointLength);
            return isPoint && pointLength.HasValue ? SequenceManipulator.ToSequence(point) : null;
        }

        /// <summary>
        /// Enumerates support of this automaton when possible.
        /// Only point mass element distributions are supported.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <exception cref="AutomatonEnumerationCountException">Thrown if enumeration is too large.</exception>
        /// <returns>The sequences in the support of this automaton</returns>
        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true)
        {
            if (tryDeterminize && this is StringAutomaton)
            {
                this.TryDeterminize();
            }

            // Lazily return sequences until the count is exceeded.
            var enumeration = this.EnumerateSupport(
                new Stack<TElement>(),
                new ArrayDictionary<bool>(),
                this.Start.Index);

            if (!tryDeterminize) enumeration = enumeration.Distinct();
            var result = enumeration.Select(
                (seq, idx) =>
                {
                    if (idx < maxCount)
                    {
                        return seq;
                    }
                    else
                    {
                        throw new AutomatonEnumerationCountException(maxCount);
                    }
                });

            return result;
        }

        /// <summary>
        /// Tries to enumerate support of this automaton.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="result">The sequences in the support of this automaton</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <exception cref="AutomatonException">General automaton exception.</exception>
        /// <returns>True if successful, false otherwise</returns>
        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true)
        {
            if (tryDeterminize && this is StringAutomaton)
            {
                this.TryDeterminize();
            }


            result = this.EnumerateSupport(new Stack<TElement>(), new ArrayDictionary<bool>(), this.Start.Index);
            if (!tryDeterminize) result = result.Distinct();
            result = result.Take(maxCount + 1).ToList();
            return result.Count() <= maxCount;
        }

        /// <summary>
        /// Enumerates paths through this automaton.
        /// </summary>
        /// <returns>The paths through this automaton, with their log weights</returns>
        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths()
        {
            return this.EnumeratePaths(new Stack<TElementDistribution>(), new ArrayDictionary<bool>(), Weight.One, this.Start.Index);
        }

        #endregion

        #region Equality

        /// <summary>
        /// Checks if <paramref name="obj"/> is an automaton that defines the same weighted regular language.
        /// </summary>
        /// <param name="obj">The object to compare this automaton with.</param>
        /// <returns><see langword="true"/> if this automaton is equal to <paramref name="obj"/>, false otherwise.</returns>
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            double logSimilarity = GetLogSimilarity((TThis)this, (TThis)obj);
            const double LogSimilarityThreshold = -30; // Can't be -inf due to numerical instabilities in GetLogSimilarity()
            return logSimilarity < LogSimilarityThreshold;
        }

        /// <summary>
        /// Gets the hash code of this automaton.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            TThis thisAutomaton = (TThis)this;
            TThis theConverger = GetConverger(thisAutomaton);
            thisAutomaton = thisAutomaton.Product(theConverger);
            double logNorm = thisAutomaton.Product(thisAutomaton).GetLogNormalizer();

            // Very similar automata will have almost identical norms and, thus, will almost always be hashed to the same bucket
            return (BitConverter.DoubleToInt64Bits(logNorm) >> 31).GetHashCode();
        }

        #endregion

        #region Manipulating structure

        /// <summary>
        /// Creates a deep copy of the state collection. Used by quoting.
        /// </summary>
        /// <returns>The created state collection copy.</returns>
        public State[] GetStates()
        {
            // FIXME: discuss what it is supposed to do
            // if needed - implement real full automaton deep-copy
            return this.States.ToArray();
        }

        /// <summary>
        /// Adds a new state to the automaton.
        /// </summary>
        /// <returns>The added state.</returns>
        /// <remarks>Indices of the added states are guaranteed to be increasing consecutive.</remarks>
        public State AddState()
        {
            if (this.statesData.Count >= maxStateCount)
            {
                throw new AutomatonTooLargeException(MaxStateCount);
            }

            var index = this.statesData.Count;
            var stateImpl = new StateData();
            this.statesData.Add(stateImpl);

            return new State(this, index, stateImpl);
        }

        /// <summary>
        /// Adds a given number of states to the automaton.
        /// </summary>
        /// <param name="stateCount">The number of states to add.</param>
        public void AddStates(int stateCount)
        {
            Argument.CheckIfInRange(stateCount >= 0, "stateCount", "The number of states to add must not be negative.");

            for (int i = 0; i < stateCount; i++)
            {
                this.AddState();
            }
        }

        /// <summary>
        /// Gets a value indicating whether this automaton is epsilon-free.
        /// </summary>
        public bool IsEpsilonFree
        {
            get
            {
                if (this.isEpsilonFree == null)
                {
                    this.isEpsilonFree = true;
                    foreach (var state in this.statesData)
                    {
                        for (int i = 0; i < state.TransitionCount; i++)
                        {
                            if (state.GetTransition(i).IsEpsilon)
                            {
                                this.isEpsilonFree = false;
                                break;
                            }
                        }

                        if (this.isEpsilonFree == false)
                        {
                            break;
                        }
                    }
                }

                return this.isEpsilonFree.Value;
            }
        }

        /// <summary>
        /// Replaces the current automaton with an equal automaton that has no epsilon transitions.
        /// </summary>
        public void MakeEpsilonFree()
        {
            if (this.IsEpsilonFree)
            {
                return;
            }

            this.SetToEpsilonClosureOf((TThis)this);
        }

        /// <summary>
        /// Replaces the current automaton with an epsilon closure of a given automaton.
        /// </summary>
        /// <remarks>
        /// The resulting automaton will be equal to the given one, but may have a simpler structure.
        /// </remarks>
        /// <param name="automaton">The automaton which epsilon closure will be used to replace the current automaton.</param>
        public void SetToEpsilonClosureOf(TThis automaton)
        {
            Argument.CheckIfNotNull(automaton, "automaton");

            if (automaton.IsEpsilonFree)
            {
                this.SetTo(automaton);
                return;
            }

            TThis result = Zero();
            result.Start = result.BuildEpsilonClosure(automaton.Start, new ArrayDictionary<State>(automaton.States.Count));
            result.LogValueOverride = automaton.LogValueOverride;
            result.PruneTransitionsWithLogWeightLessThan = automaton.PruneTransitionsWithLogWeightLessThan;
            result.isEpsilonFree = true;
            this.SwapWith(result);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Gets a value indicating how close two given automata are
        /// in terms of values they assign to sequences.
        /// </summary>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <returns>The logarithm of a non-negative value, which is close to zero if the two automata assign similar values to all sequences.</returns>
        internal static double GetLogSimilarity(TThis automaton1, TThis automaton2)
        {
            if (automaton1.IsZero())
            {
                return automaton2.IsZero() ? double.NegativeInfinity : 1;
            }

            TThis theConverger = GetConverger(automaton1, automaton2);
            var automaton1conv = automaton1.Product(theConverger);
            var automaton2conv = automaton2.Product(theConverger);

            TThis product12 = automaton1conv.Product(automaton2conv);
            if (product12.IsZero())
            {
                return 1;
            }

            TThis squared1 = automaton1conv.Product(automaton1conv);
            TThis squared2 = automaton2conv.Product(automaton2conv);

            double logNormSquared1 = squared1.GetLogNormalizer();
            double logNormSquared2 = squared2.GetLogNormalizer();
            double logNormProduct12 = product12.GetLogNormalizer();

            Debug.Assert(
                !double.IsPositiveInfinity(logNormSquared1) &&
                !double.IsPositiveInfinity(logNormSquared2) &&
                !double.IsPositiveInfinity(logNormProduct12),
                "Covergence enforcing failed (probably due to presense of non-trivial loops).");

            double term1 = MMath.LogSumExp(logNormSquared1, logNormSquared2);
            double term2 = logNormProduct12 + MMath.Ln2;
            double result = MMath.LogDifferenceOfExp(Math.Max(term1, term2), Math.Min(term1, term2)); // To avoid NaN due to numerical instabilities
            Debug.Assert(!double.IsNaN(result), "Log-similarity must be a valid number.");
            return result;
        }

        /// <summary>
        /// Gets an automaton such that every given automaton, if multiplied by it, becomes normalizable.
        /// </summary>
        /// <param name="automata">The automata.</param>
        /// <returns>An automaton, product with which will make every given automaton normalizable.</returns>
        public static TThis GetConverger(params TThis[] automata)
        {
            // TODO: This method might not work in the presense of non-trivial loops.

            // Find the maximum total weight of outgoing transitions
            double maxLogTransitionWeightSum = double.NegativeInfinity;
            for (int automatonIndex = 0; automatonIndex < automata.Length; ++automatonIndex)
            {
                TThis automaton = automata[automatonIndex];

                for (int stateIndex = 0; stateIndex < automaton.statesData.Count; ++stateIndex)
                {
                    State state = automaton.States[stateIndex];
                    Weight transitionWeightSum = Weight.Zero;
                    for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                    {
                        Transition transition = state.GetTransition(transitionIndex);
                        transitionWeightSum = Weight.Sum(transitionWeightSum, transition.Weight);
                    }

                    maxLogTransitionWeightSum = Math.Max(maxLogTransitionWeightSum, transitionWeightSum.LogValue);
                }
            }

            if (maxLogTransitionWeightSum < 0)
            {
                // Should be normalizable just fine
                return Constant(1.0);
            }

            // An automaton, product with which will make every self-loop converging
            var uniformDist = new TElementDistribution();
            uniformDist.SetToUniform();
            TThis theConverger = Zero();
            Weight transitionWeight = Weight.Product(
                Weight.FromLogValue(-uniformDist.GetLogAverageOf(uniformDist)),
                Weight.FromLogValue(-maxLogTransitionWeightSum),
                Weight.FromValue(0.99));
            theConverger.Start.AddSelfTransition(uniformDist, transitionWeight);
            theConverger.Start.SetEndWeight(Weight.One);

            return theConverger;
        }

        /// <summary>
        /// Checks if indices assigned to given states and their transitions are consistent with each other.
        /// </summary>
        /// <param name="states">The collection of states to check.</param>
        /// <param name="startState">The start state to check.</param>
        private static void CheckStateConsistency(IEnumerable<State> states, State startState)
        {
            State[] stateArray = states.ToArray();
            for (int stateIndex = 0; stateIndex < stateArray.Length; ++stateIndex)
            {
                var state = stateArray[stateIndex];
                if (state.Index != stateIndex)
                {
                    throw new ArgumentException("State indices must be consequent zero-based.");
                }

                for (int transitionIndex = 0; transitionIndex < stateArray[stateIndex].TransitionCount; ++transitionIndex)
                {
                    var transition = state.GetTransition(transitionIndex);
                    if (transition.DestinationStateIndex < 0 || transition.DestinationStateIndex >= stateArray.Length)
                    {
                        throw new ArgumentException("Transition destination indices must point to a valid state.");
                    }
                }
            }

            if (startState.Index >= stateArray.Length)
            {
                throw new ArgumentException("Start state has an invalid index.");
            }
        }

        /// <summary>
        /// Performs depth-first traversal of a given graph.
        /// </summary>
        /// <param name="currentVertex">The index of the currently traversed vertex.</param>
        /// <param name="visitedVertices">An array to keep track of the visited vertices.</param>
        /// <param name="edgeDestinationIndices">An array containing destination indices of the graph edges.</param>
        /// <param name="edgeArrayStarts">An array</param>
        private static void LabelReachableNodesDfs(
            int currentVertex, bool[] visitedVertices, int[] edgeDestinationIndices, int[] edgeArrayStarts)
        {
            Debug.Assert(!visitedVertices[currentVertex], "Visited vertices must not be revisited.");
            visitedVertices[currentVertex] = true;

            for (int edgeIndex = edgeArrayStarts[currentVertex]; edgeIndex < edgeArrayStarts[currentVertex + 1]; ++edgeIndex)
            {
                int destVertexIndex = edgeDestinationIndices[edgeIndex];
                if (!visitedVertices[destVertexIndex])
                {
                    LabelReachableNodesDfs(destVertexIndex, visitedVertices, edgeDestinationIndices, edgeArrayStarts);
                }
            }
        }

        /// <summary>
        /// A version of <see cref="AppendInPlace(TThis, int)"/> that is guaranteed to preserve
        /// the states of both the original automaton and the automaton being appended in the result.
        /// </summary>
        /// <param name="automaton">The automaton to append.</param>
        /// <remarks>
        /// Useful for implementing functions like <see cref="Repeat(TThis, Vector)"/>,
        /// where on-the-fly result optimization creates unnecessary complications.
        /// </remarks>
        private void AppendInPlaceNoOptimizations(TThis automaton)
        {
            if (ReferenceEquals(automaton, this))
            {
                automaton = automaton.Clone();
            }

            int stateCount = this.statesData.Count;
            var endStates = this.States.Where(nd => nd.CanEnd).ToList();

            this.AddStates(automaton.statesData);
            var secondStartState = this.States[stateCount + automaton.Start.Index];

            for (int i = 0; i < endStates.Count; i++)
            {
                var state = endStates[i];
                state.AddEpsilonTransition(state.EndWeight, secondStartState);
                state.SetEndWeight(Weight.Zero);
            }
        }

        /// <summary>
        /// For each state computes whether any state with non-zero ending weight can be reached from it.
        /// </summary>
        /// <returns>An array mapping state indices to end state reachability.</returns>
        private bool[] ComputeEndStateReachability()
        {
            //// First, build a reversed graph

            int[] edgePlacementIndices = new int[this.statesData.Count + 1];
            for (int i = 0; i < this.statesData.Count; ++i)
            {
                var state = this.statesData[i];
                for (int j = 0; j < state.TransitionCount; ++j)
                {
                    var transition = state.GetTransition(j);
                    if (!transition.Weight.IsZero)
                    {
                        ++edgePlacementIndices[transition.DestinationStateIndex + 1];
                    }
                }
            }

            // The element of edgePlacementIndices at index i+1 contains a count of the number of edges 
            // going into the i'th state (the indegree of the state).
            // Convert this into a cumulative count (which will be used to give a unique index to each edge).
            for (int i = 1; i < edgePlacementIndices.Length; ++i)
            {
                edgePlacementIndices[i] += edgePlacementIndices[i - 1];
            }

            int[] edgeArrayStarts = (int[])edgePlacementIndices.Clone();
            int totalEdgeCount = edgePlacementIndices[this.statesData.Count];
            int[] edgeDestinationIndices = new int[totalEdgeCount];
            for (int i = 0; i < this.statesData.Count; ++i)
            {
                var state = this.statesData[i];
                for (int j = 0; j < state.TransitionCount; ++j)
                {
                    var transition = state.GetTransition(j);
                    if (!transition.Weight.IsZero)
                    {
                        // The unique index for this edge
                        int edgePlacementIndex = edgePlacementIndices[transition.DestinationStateIndex]++;

                        // The source index for the edge (which is the destination edge in the reversed graph)
                        edgeDestinationIndices[edgePlacementIndex] = i;
                    }
                }
            }

            //// Now run a depth-first search to label all reachable nodes

            bool[] visitedNodes = new bool[this.statesData.Count];
            for (int i = 0; i < this.statesData.Count; ++i)
            {
                if (!visitedNodes[i] && this.statesData[i].CanEnd)
                {
                    LabelReachableNodesDfs(i, visitedNodes, edgeDestinationIndices, edgeArrayStarts);
                }
            }

            return visitedNodes;
        }

        /// <summary>
        /// For each state computes whether it can be reached from the start state.
        /// </summary>
        /// <returns>An array mapping state indices to start state reachability.</returns>
        private bool[] ComputeStartStateReachability()
        {
            //// First, build a reversed graph

            int[] edgePlacementIndices = new int[this.statesData.Count + 1];
            for (int i = 0; i < this.statesData.Count; ++i)
            {
                var state = this.statesData[i];
                for (int j = 0; j < state.TransitionCount; ++j)
                {
                    var transition = state.GetTransition(j);
                    if (!transition.Weight.IsZero)
                    {
                        ++edgePlacementIndices[i + 1];
                    }
                }
            }

            // The element of edgePlacementIndices at index i+1 contains a count of the number of edges 
            // going out of the i'th state (the outdegree of the state).
            // Convert this into a cumulative count (which will be used to give a unique index to each edge).
            for (int i = 1; i < edgePlacementIndices.Length; ++i)
            {
                edgePlacementIndices[i] += edgePlacementIndices[i - 1];
            }

            int[] edgeArrayStarts = (int[])edgePlacementIndices.Clone();
            int totalEdgeCount = edgePlacementIndices[this.statesData.Count];
            int[] edgeDestinationIndices = new int[totalEdgeCount];
            for (int i = 0; i < this.statesData.Count; ++i)
            {
                var state = this.statesData[i];
                for (int j = 0; j < state.TransitionCount; ++j)
                {
                    var transition = state.GetTransition(j);
                    if (!transition.Weight.IsZero)
                    {
                        // The unique index for this edge
                        int edgePlacementIndex = edgePlacementIndices[i]++;

                        // The destination index for the edge 
                        edgeDestinationIndices[edgePlacementIndex] = transition.DestinationStateIndex;
                    }
                }
            }

            //// Now run a depth-first search to label all reachable nodes
            bool[] visitedNodes = new bool[this.statesData.Count];
            LabelReachableNodesDfs(this.Start.Index, visitedNodes, edgeDestinationIndices, edgeArrayStarts);
            return visitedNodes;
        }

        /// <summary>
        /// Computes the logarithm of the normalizer of the automaton, normalizing it afterwards if requested.
        /// </summary>
        /// <param name="normalize">Specifies whether the automaton must be normalized after computing the normalizer.</param>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <remarks>The automaton is normalized only if the normalizer has a finite non-zero value.</remarks>
        private double DoGetLogNormalizer(bool normalize)
        {
            // TODO: apply to 'this' instead?
            TThis noEpsilonTransitions = Zero();
            noEpsilonTransitions.SetToEpsilonClosureOf((TThis)this); // To get rid of infinite weight closures

            Condensation condensation = noEpsilonTransitions.ComputeCondensation(noEpsilonTransitions.Start, tr => true, false);
            double logNormalizer = condensation.GetWeightToEnd(noEpsilonTransitions.Start).LogValue;
            if (normalize)
            {
                if (!double.IsInfinity(logNormalizer))
                {
                    noEpsilonTransitions.PushWeights(condensation);
                    this.SwapWith(noEpsilonTransitions);
                }
            }

            return logNormalizer;
        }

        /// <summary>
        /// Re-distributes weights of the states and transitions so that the underlying automaton becomes stochastic
        /// (i.e. sum of weights of all the outgoing transitions and the ending weight is 1 for every node).
        /// </summary>
        /// <param name="condensation">A condensation of the automaton.</param>
        private void PushWeights(Condensation condensation)
        {
            for (int i = 0; i < this.statesData.Count; ++i)
            {
                var state = this.States[i];
                Weight weightToEnd = condensation.GetWeightToEnd(state);
                if (weightToEnd.IsZero)
                {
                    continue; // End states cannot be reached from this state, no point in normalization
                }

                Weight weightToEndInv = Weight.Inverse(weightToEnd);
                for (int j = 0; j < state.TransitionCount; ++j)
                {
                    Transition transition = state.GetTransition(j);
                    transition.Weight = Weight.Product(
                        transition.Weight,
                        condensation.GetWeightToEnd(this.States[transition.DestinationStateIndex]),
                        weightToEndInv);
                    state.SetTransition(j, transition);
                }

                state.SetEndWeight(Weight.Product(state.EndWeight, weightToEndInv));
            }
        }

        /// <summary>
        /// Recursively looks for the only sequence which has non-zero value under the automaton.
        /// </summary>
        /// <param name="currentState">The state currently being traversed.</param>
        /// <param name="currentSequencePos">The current position in the sequence.</param>
        /// <param name="stateSequencePosCache">A lookup table for memoization.</param>
        /// <param name="isEndNodeReachable">End node reachability table to avoid dead branches.</param>
        /// <param name="point">The candidate sequence.</param>
        /// <param name="pointLength">The length of the candidate sequence
        /// or <see langword="null"/> if the length isn't known yet.</param>
        /// <returns>
        /// <see langword="true"/> if the sequence was found, <see langword="false"/> otherwise.
        /// </returns>
        private bool TryComputePointDfs(
            State currentState,
            int currentSequencePos,
            ArrayDictionary<int> stateSequencePosCache,
            bool[] isEndNodeReachable,
            List<TElement> point,
            ref int? pointLength)
        {
            Debug.Assert(isEndNodeReachable[currentState.Index], "Dead branches must not be visited.");

            int cachedStateDepth;
            if (stateSequencePosCache.TryGetValue(currentState.Index, out cachedStateDepth))
            {
                // If we've already been in this state, we must be at the same sequence pos
                return currentSequencePos == cachedStateDepth;
            }

            stateSequencePosCache.Add(currentState.Index, currentSequencePos);

            // Can we stop in this state?
            if (currentState.CanEnd)
            {
                // Is this a suffix or a prefix of the point already found?
                if (pointLength.HasValue)
                {
                    if (currentSequencePos != pointLength.Value)
                    {
                        return false;
                    }
                }
                else
                {
                    // Now we know the length of the sequence
                    pointLength = currentSequencePos;
                }
            }

            for (int i = 0; i < currentState.TransitionCount; ++i)
            {
                Transition transition = currentState.GetTransition(i);

                State destState = this.States[transition.DestinationStateIndex];
                if (!isEndNodeReachable[destState.Index])
                {
                    continue; // Only walk through the accepting part of the automaton
                }

                if (transition.IsEpsilon)
                {
                    // Move to the next state, keep the sequence position
                    if (!this.TryComputePointDfs(destState, currentSequencePos, stateSequencePosCache, isEndNodeReachable, point, ref pointLength))
                    {
                        return false;
                    }
                }
                else if (!transition.ElementDistribution.Value.IsPointMass)
                {
                    return false;
                }
                else
                {
                    TElement element = transition.ElementDistribution.Value.Point;
                    if (currentSequencePos == point.Count)
                    {
                        // It is the first time at this sequence position
                        point.Add(element);
                    }
                    else if (!point[currentSequencePos].Equals(element))
                    {
                        // This is not the first time at this sequence position, and the elements are different
                        return false;
                    }

                    // Look at the next sequence position
                    if (!this.TryComputePointDfs(destState, currentSequencePos + 1, stateSequencePosCache, isEndNodeReachable, point, ref pointLength))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Recursively builds an automaton representing the product of two given automata.
        /// The second automaton must be epsilon-free.
        /// </summary>
        /// <param name="state1">The currently traversed state of the first automaton.</param>
        /// <param name="state2">The currently traversed state of the second automaton.</param>
        /// <param name="productStateCache">
        /// The mapping from a pair of argument state indices to an index of the corresponding product state.
        /// </param>
        /// <returns>The index of the product state corresponding to the given pair of state.</returns>
        private State BuildProduct(
            State state1,
            State state2,
            Dictionary<(int, int), State> productStateCache)
        {
            Debug.Assert(state1 != null && state2 != null, "Valid states must be provided.");
            Debug.Assert(!ReferenceEquals(state1.Owner, this) && !ReferenceEquals(state2.Owner, this), "Cannot build product in place.");
            Debug.Assert(state2.Owner.IsEpsilonFree, "The second argument of the product operation must be epsilon-free.");

            // State already exists, return its index
            var statePair = (state1.Index, state2.Index);
            State productState;
            if (productStateCache.TryGetValue(statePair, out productState))
            {
                return productState;
            }

            // Create a new state
            productState = this.AddState();
            productStateCache.Add(statePair, productState);

            // Iterate over transitions in state1
            for (int transition1Index = 0; transition1Index < state1.TransitionCount; transition1Index++)
            {
                Transition transition1 = state1.GetTransition(transition1Index);
                State destState1 = state1.Owner.States[transition1.DestinationStateIndex];

                if (transition1.IsEpsilon)
                {
                    // Epsilon transition case
                    State destProductState = this.BuildProduct(destState1, state2, productStateCache);
                    productState.AddEpsilonTransition(transition1.Weight, destProductState, transition1.Group);
                    continue;
                }

                // Iterate over transitions in state2
                for (int transition2Index = 0; transition2Index < state2.TransitionCount; transition2Index++)
                {
                    Transition transition2 = state2.GetTransition(transition2Index);
                    Debug.Assert(!transition2.IsEpsilon, "The second argument of the product operation must be epsilon-free.");
                    State destState2 = state2.Owner.States[transition2.DestinationStateIndex];

                    TElementDistribution product;
                    double productLogNormalizer = Distribution<TElement>.GetLogAverageOf(
                        transition1.ElementDistribution.Value, transition2.ElementDistribution.Value, out product);
                    ////if (product is StringDistribution)
                    ////{
                    ////    Console.WriteLine(transition1.ElementDistribution+" x "+transition2.ElementDistribution+" = "+product+" "+productLogNormalizer+" "+transition1.ElementDistribution.Equals(transition1.ElementDistribution)+" "+transition1.ElementDistribution.Equals(transition2.ElementDistribution));
                    ////}
                    if (double.IsNegativeInfinity(productLogNormalizer))
                    {
                        continue;
                    }

                    Weight productWeight = Weight.Product(transition1.Weight, transition2.Weight, Weight.FromLogValue(productLogNormalizer));
                    State destProductState = this.BuildProduct(destState1, destState2, productStateCache);
                    productState.AddTransition(product, productWeight, destProductState, transition1.Group);
                }
            }

            productState.SetEndWeight(Weight.Product(state1.EndWeight, state2.EndWeight));
            return productState;
        }

        /// <summary>
        /// Recursively builds an automaton representing the epsilon closure of a given automaton.
        /// </summary>
        /// <param name="state">The currently traversed state of the source automaton.</param>
        /// <param name="oldToNewState">
        /// The mapping from the indices of the states of the source automaton to the states of the closure.
        /// </param>
        /// <returns>The state representing the closure of <paramref name="state"/>.</returns>
        private State BuildEpsilonClosure(State state, ArrayDictionary<State> oldToNewState)
        {
            Debug.Assert(state != null, "A valid state must be provided.");
            Debug.Assert(!ReferenceEquals(state.Owner, this), "In-place epsilon closure building is not supported.");

            State resultState;
            if (oldToNewState.TryGetValue(state.Index, out resultState))
            {
                return resultState;
            }

            resultState = this.AddState();
            oldToNewState.Add(state.Index, resultState);

            EpsilonClosure closure = state.GetEpsilonClosure();
            resultState.SetEndWeight(closure.EndWeight);
            for (int stateIndex = 0; stateIndex < closure.Size; ++stateIndex)
            {
                State closureState = closure.GetStateByIndex(stateIndex);
                Weight closureStateWeight = closure.GetStateWeightByIndex(stateIndex);
                for (int transitionIndex = 0; transitionIndex < closureState.TransitionCount; ++transitionIndex)
                {
                    Transition transition = closureState.GetTransition(transitionIndex);
                    if (!transition.IsEpsilon)
                    {
                        State destState = state.Owner.States[transition.DestinationStateIndex];
                        State closureDestState = this.BuildEpsilonClosure(destState, oldToNewState);
                        resultState.AddTransition(
                            transition.ElementDistribution, Weight.Product(transition.Weight, closureStateWeight), closureDestState, transition.Group);
                    }
                }
            }

            return resultState;
        }

        /// <summary>
        /// Swaps the current automaton with a given one.
        /// </summary>
        /// <param name="automaton">The automaton to swap the current one with.</param>
        private void SwapWith(TThis automaton)
        {
            Debug.Assert(automaton != null, "A valid automaton must be provided.");

            // Swap contents
            Util.Swap(ref this.statesData, ref automaton.statesData);
            Util.Swap(ref this.startStateIndex, ref automaton.startStateIndex);
            Util.Swap(ref this.isEpsilonFree, ref automaton.isEpsilonFree);
            var dummy = this.LogValueOverride;
            this.LogValueOverride = automaton.LogValueOverride;
            automaton.LogValueOverride = dummy;
            dummy = this.PruneTransitionsWithLogWeightLessThan;
            this.PruneTransitionsWithLogWeightLessThan = automaton.PruneTransitionsWithLogWeightLessThan;
            automaton.PruneTransitionsWithLogWeightLessThan = dummy;
        }

        /// <summary>
        /// Replaces the states of this automaton with a given collection of states.
        /// </summary>
        /// <param name="newStates">The states to replace the existing states with.</param>
        /// <param name="newStartStateIndex">The index of the new start state.</param>
        private void SetStates(IEnumerable<StateData> newStates, int newStartStateIndex)
        {
            this.statesData.Clear();
            this.AddStates(newStates);
            this.Start = this.States[newStartStateIndex];
        }

        /// <summary>
        /// Adds the states in a given collection to the automaton together with their transitions,
        /// but without attaching any of them to any of the existing states.
        /// </summary>
        /// <param name="statesToAdd">The states to add.</param>
        /// <param name="group">The group for the transitions of the states being added.</param>
        private void AddStates(IEnumerable<StateData> statesToAdd, int group = 0)
        {
            Debug.Assert(statesToAdd != null, "A valid state collection must be provided.");

            int startIndex = this.statesData.Count;
            var statesToAddList = statesToAdd as IList<StateData> ?? statesToAdd.ToList();

            // Add states
            for (int i = 0; i < statesToAddList.Count; ++i)
            {
                State newState = this.AddState();
                newState.SetEndWeight(statesToAddList[i].EndWeight);

                Debug.Assert(newState.Index == i + startIndex, "State indices must always be consequent.");
            }

            // Add transitions
            for (int i = 0; i < statesToAddList.Count; ++i)
            {
                var stateToAdd = statesToAddList[i];
                for (int transitionIndex = 0; transitionIndex < stateToAdd.TransitionCount; transitionIndex++)
                {
                    Transition transitionToAdd = stateToAdd.GetTransition(transitionIndex);
                    Debug.Assert(transitionToAdd.DestinationStateIndex < statesToAddList.Count, "Self-inconsistent collection of states provided.");
                    this.States[i + startIndex].AddTransition(
                        transitionToAdd.ElementDistribution,
                        transitionToAdd.Weight,
                        this.States[transitionToAdd.DestinationStateIndex + startIndex],
                        group != 0 ? group : transitionToAdd.Group);
                }
            }
        }

        /// <summary>
        /// Removes the state with a given index from the automaton.
        /// </summary>
        /// <param name="index">The index of the state to remove.</param>
        /// <param name="replaceIndex">
        /// If specified, all the transitions to the state being removed
        /// will be redirected to the state with this index.
        /// </param>
        /// <remarks>
        /// The automaton representation we currently use does not allow for fast state removal.
        /// Ideally we should get rid of this function completely.
        /// </remarks>
        private void RemoveState(int index, int? replaceIndex = null)
        {
            //// TODO: see remarks

            Debug.Assert(index >= 0 && index < this.statesData.Count, "An invalid state index provided.");
            Debug.Assert(index != this.Start.Index, "Cannot remove the start state.");
            Debug.Assert(
                !replaceIndex.HasValue || (replaceIndex.Value >= 0 && replaceIndex.Value < this.statesData.Count),
                "An invalid replace index provided.");
            Debug.Assert(!replaceIndex.HasValue || replaceIndex.Value != index, "Replace index must point to a different state.");

            this.statesData.RemoveAt(index);
            int stateCount = this.statesData.Count;
            for (int i = 0; i < stateCount; i++)
            {
                StateData state = this.statesData[i];
                for (int j = state.TransitionCount - 1; j >= 0; j--)
                {
                    Transition transition = state.GetTransition(j);
                    if (transition.DestinationStateIndex == index)
                    {
                        if (!replaceIndex.HasValue)
                        {
                            state.RemoveTransition(j);
                            continue;
                        }

                        transition.DestinationStateIndex = replaceIndex.Value;
                    }

                    if (transition.DestinationStateIndex > index)
                    {
                        transition.DestinationStateIndex = transition.DestinationStateIndex - 1;
                    }

                    state.SetTransition(j, transition);
                }
            }
        }

        /// <summary>
        /// Appends a string representing the automaton from the current state.
        /// </summary>
        /// <param name="builder">The string builder.</param>
        /// <param name="visitedStates">The set of states that has already been visited.</param>
        /// <param name="stateIndex">The index of the current state.</param>
        /// <param name="appendRegex">Optional method for appending at the element distribution level.</param>
        private void AppendString(StringBuilder builder, HashSet<int> visitedStates, int stateIndex, Action<TElementDistribution, StringBuilder> appendRegex = null)
        {
            if (visitedStates.Contains(stateIndex))
            {
                builder.Append('➥');
                return;
            }

            visitedStates.Add(stateIndex);

            var currentState = this.States[stateIndex];
            var transitions = currentState.GetTransitions().Where(t => !t.Weight.IsZero);
            var selfTransitions = transitions.Where(t => t.DestinationStateIndex == stateIndex);
            int selfTransitionCount = selfTransitions.Count();
            var nonSelfTransitions = transitions.Where(t => t.DestinationStateIndex != stateIndex);
            int nonSelfTransitionCount = nonSelfTransitions.Count();

            if (currentState.CanEnd && nonSelfTransitionCount > 0)
            {
                builder.Append('▪');
            }

            if (selfTransitionCount > 1)
            {
                builder.Append('[');
            }

            int transIdx = 0;
            foreach (var transition in selfTransitions)
            {
                if (!transition.IsEpsilon)
                {
                    if (appendRegex != null)
                    {
                        appendRegex(transition.ElementDistribution.Value, builder);
                    }
                    else
                    {
                        builder.Append(transition.ElementDistribution);
                    }
                }

                if (++transIdx < selfTransitionCount)
                {
                    builder.Append('|');
                }
            }

            if (selfTransitionCount > 1)
            {
                builder.Append(']');
            }

            if (selfTransitionCount > 0)
            {
                builder.Append('*');
            }

            if (nonSelfTransitionCount > 1)
            {
                builder.Append('[');
            }

            transIdx = 0;
            foreach (var transition in nonSelfTransitions)
            {
                if (!transition.IsEpsilon)
                {
                    if (appendRegex != null)
                    {
                        appendRegex(transition.ElementDistribution.Value, builder);
                    }
                    else
                    {
                        builder.Append(transition.ElementDistribution);
                    }
                }

                this.AppendString(builder, visitedStates, transition.DestinationStateIndex, appendRegex);
                if (++transIdx < nonSelfTransitionCount)
                {
                    builder.Append('|');
                }
            }

            if (nonSelfTransitionCount > 1)
            {
                builder.Append(']');
            }
        }

        /// <summary>
        /// Recursively enumerate support of this automaton
        /// </summary>
        /// <param name="prefix">The prefix at this point</param>
        /// <param name="visitedStates">The states visited at this point</param>
        /// <param name="stateIndex">The index of the next state to process</param>
        /// <returns>The strings supporting this automaton</returns>
        private IEnumerable<TSequence> EnumerateSupport(Stack<TElement> prefix, ArrayDictionary<bool> visitedStates, int stateIndex)
        {
            if (visitedStates.ContainsKey(stateIndex) && visitedStates[stateIndex])
            {
                throw new NotSupportedException("Infinite loops cannot be enumerated");
            }

            var currentState = this.States[stateIndex];
            if (currentState.CanEnd)
            {
                yield return SequenceManipulator.ToSequence(prefix.Reverse());
            }

            visitedStates[stateIndex] = true;
            for (int i = 0; i < currentState.TransitionCount; ++i)
            {
                var transition = currentState.GetTransition(i);

                if (transition.Weight.IsZero)
                {
                    continue;
                }

                if (transition.IsEpsilon)
                {
                    foreach (var support in this.EnumerateSupport(prefix, visitedStates, transition.DestinationStateIndex))
                    {
                        yield return support;
                    }
                }
                else if (transition.ElementDistribution.Value.IsPointMass)
                {
                    prefix.Push(transition.ElementDistribution.Value.Point);
                    foreach (var support in this.EnumerateSupport(prefix, visitedStates, transition.DestinationStateIndex))
                    {
                        yield return support;
                    }

                    prefix.Pop();
                }
                else
                {
                    if (!(transition.ElementDistribution.Value is CanEnumerateSupport<TElement> supportEnumerator))
                    {
                        throw new NotImplementedException("Only point mass element distributions or distributions for which we can enumerate support are currently implemented");
                    }

                    foreach (var elt in supportEnumerator.EnumerateSupport())
                    {
                        prefix.Push(elt);
                        foreach (var support in this.EnumerateSupport(prefix, visitedStates, transition.DestinationStateIndex))
                        {
                            yield return support;
                        }

                        prefix.Pop();
                    }
                }
            }

            visitedStates[stateIndex] = false;
        }

        /// <summary>
        /// Recursively enumerate paths through this automaton
        /// </summary>
        /// <param name="prefix">The prefix at this point</param>
        /// <param name="visitedStates">The states visited at this point</param>
        /// <param name="weight">The weight of the path to this point</param>
        /// <param name="stateIndex">The index of the next state to process</param>
        /// <returns>The paths through this automaton, with their log weights</returns>
        private IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths(Stack<TElementDistribution> prefix, ArrayDictionary<bool> visitedStates, Weight weight, int stateIndex)
        {
            var currentState = this.States[stateIndex];
            if (currentState.CanEnd)
            {
                var newWeight = Weight.Product(weight, currentState.EndWeight);
                yield return new Tuple<List<TElementDistribution>, double>(prefix.Reverse().ToList(), newWeight.LogValue);
            }

            visitedStates[stateIndex] = true;
            for (int i = 0; i < currentState.TransitionCount; ++i)
            {
                var transition = currentState.GetTransition(i);

                if (transition.Weight.IsZero)
                {
                    continue;
                }

                if (visitedStates.ContainsKey(transition.DestinationStateIndex) && visitedStates[transition.DestinationStateIndex])
                {
                    // throw new NotSupportedException("Loops cannot be enumerated");
                    // Ignore loops
                    // todo: do something else?
                    continue;
                }

                if (!transition.IsEpsilon)
                {
                    prefix.Push(transition.ElementDistribution.Value);
                }

                foreach (var support in this.EnumeratePaths(prefix, visitedStates, Weight.Product(weight, transition.Weight), transition.DestinationStateIndex))
                {
                    yield return support;
                }

                if (!transition.IsEpsilon)
                {
                    prefix.Pop();
                }
            }

            visitedStates[stateIndex] = false;
        }

        /// <summary>
        /// Temporarily allow for large automata.
        /// </summary>
        public class UnlimitedStatesComputation : IDisposable
        {
            private readonly int originalMaxStateCount;

            /// <summary>
            /// Initilizes a new instance of the <see cref="UnlimitedStatesComputation"/> class.
            /// </summary>
            public UnlimitedStatesComputation()
            {
                originalMaxStateCount = StringAutomaton.MaxStateCount;
                StringAutomaton.MaxStateCount = int.MaxValue;
            }

            /// <summary>
            /// Verifies that the resulting automaton is within the range for MaxStateCount.
            /// </summary>
            public void CheckStateCount(TThis automaton)
            {
                if(automaton.States.Count > originalMaxStateCount) throw new AutomatonTooLargeException(originalMaxStateCount);
            }

            public void Dispose()
            {
                StringAutomaton.MaxStateCount = originalMaxStateCount;
            }
        }
        #endregion

        #region Serialization

        /// <summary>
        /// Constructor used during deserialization by Newtonsoft.Json and BinaryFormatter .
        /// </summary>
        /// <remarks>
        /// To be (de)serializable, classes inherited from Automaton must constructor with the same signature.
        /// </remarks>
        protected Automaton(SerializationInfo info, StreamingContext context)
        {
            this.statesData = (List<StateData>)info.GetValue(nameof(this.statesData), typeof(List<StateData>));
            this.startStateIndex = (int)info.GetValue(nameof(this.startStateIndex), typeof(int));
            this.isEpsilonFree = (bool?)info.GetValue(nameof(this.isEpsilonFree), typeof(bool?));
        }

        void ISerializable.GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue(nameof(this.statesData), this.statesData);
            info.AddValue(nameof(this.startStateIndex), this.startStateIndex);
            info.AddValue(nameof(this.isEpsilonFree), this.isEpsilonFree);
        }

        /// <summary>
        /// Writes the current automaton.
        /// </summary>
        public void Write(Action<double> writeDouble, Action<int> writeInt32, Action<TElementDistribution> writeElementDistribution)
        {
            var propertyMask = new BitVector32();
            var idx = 0;
            propertyMask[1 << idx++] = this.isEpsilonFree.HasValue;
            propertyMask[1 << idx++] = this.isEpsilonFree.HasValue && this.isEpsilonFree.Value;
            propertyMask[1 << idx++] = this.LogValueOverride.HasValue;
            propertyMask[1 << idx++] = this.PruneTransitionsWithLogWeightLessThan.HasValue;
            propertyMask[1 << idx++] = !this.Start.IsNull;

            writeInt32(propertyMask.Data);

            if (this.LogValueOverride.HasValue)
            {
                writeDouble(this.LogValueOverride.Value);
            }

            if (this.PruneTransitionsWithLogWeightLessThan.HasValue)
            {
                writeDouble(this.PruneTransitionsWithLogWeightLessThan.Value);
            }

            if (!this.Start.IsNull)
            {
                this.Start.Write(writeInt32, writeDouble, writeElementDistribution);
            }

            writeInt32(this.statesData.Count);
            foreach (var state in this.States)
            {
                state.Write(writeInt32, writeDouble, writeElementDistribution);
            }
        }

        /// <summary>
        /// Reads an automaton from.
        /// </summary>
        public static TThis Read(Func<double> readDouble, Func<int> readInt32, Func<TElementDistribution> readElementDistribution)
        {
            var propertyMask = new BitVector32(readInt32());
            var res = new TThis();
            var idx = 0;
            var hasEpsilonFree = propertyMask[1 << idx++];
            var isEpsilonFree = propertyMask[1 << idx++];
            var hasLogValueOverride = propertyMask[1 << idx++];
            var hasPruneTransitions = propertyMask[1 << idx++];
            var hasStartState = propertyMask[1 << idx++];

            res.isEpsilonFree = hasEpsilonFree ? (bool?)isEpsilonFree : null;

            if (hasLogValueOverride)
            {
                res.LogValueOverride = readDouble();
            }

            if (hasPruneTransitions)
            {
                res.PruneTransitionsWithLogWeightLessThan = readDouble();
            }

            var startState =
                hasStartState
                    ? State.Read(readInt32, readDouble, readElementDistribution)
                    : default(State);

            var numStates = readInt32();
            res.statesData.Clear();
            res.AddStates(numStates);
            for (var i = 0; i < numStates; i++)
            {
                res.statesData[i] = State.Read(readInt32, readDouble, readElementDistribution).Data;
            }

            if (hasStartState)
            {
                res.startStateIndex = startState.Index;
            }

            return res;
        }
        #endregion
    }
}