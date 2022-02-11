// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
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
    /// 
    /// This class is constant and thread-safe (though, not truly immutable, as some properties are evaluated lazily
    /// and stored afterwards) and all implementations must preserve this property - some algorithms used
    /// to implement this class rely on it. For the same reason, implementations are not allowed to add new data fields.
    /// </remarks>
    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The immutable type of a sequence element.</typeparam>
    /// <typeparam name="TElementDistribution">The immutable type of a distribution over sequence elements.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
    /// <typeparam name="TThis">The type of a concrete automaton class.</typeparam>
    [Quality(QualityBand.Experimental)]
    [DataContract]
    [Serializable]
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> :
        WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>.IWeightFunction<TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TThis : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>, new()
    {
        #region Fields & constants

        private static readonly TThis ZeroAutomaton = new TThis() { Data = DataContainer.ZeroData };

        /// <summary>
        /// The maximum number of states an automaton can have.
        /// </summary>
        private static int maxStateCount = 300_000;

        /// <summary>
        /// The maximum number of states an automaton can have in current thread.
        /// </summary>
        /// <remarks>
        /// If non-zero, this value overrides the default <see cref="GlobalMaxStateCount"/>.
        /// This value is used by <see cref="UnlimitedStatesComputation"/> to temporary increase
        /// the state count limit in a thread-safe manner.
        /// </remarks>
        [ThreadStatic]
        private static int threadMaxStateCountOverride;

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

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Automaton{TSequence,TElement,TElementDistribution,TSequenceManipulator,TThis}"/>
        /// class by setting it to be zero everywhere.
        /// </summary>
        protected Automaton()
        {
            // Zero by default
            Data = DataContainer.ZeroData;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Immutable container for automaton data - states and transitions.
        /// </summary>
        [DataMember]
        public DataContainer Data { get; protected set; }

        /// <summary>
        /// Gets the sequence manipulator.
        /// </summary>
        public static TSequenceManipulator SequenceManipulator { get; } =
            new TSequenceManipulator();

        /// <summary>
        /// Gets an instance of <typeparamref name="TElementDistribution"/> that can be used as a factory
        /// to create zero, uniform, and pointmass distributions.
        /// </summary>
        internal static TElementDistribution ElementDistributionFactory { get; } = new TElementDistribution();

        /// <summary>
        /// Gets a value that, if not null, will be returned when computing the log value of any sequence
        /// which is in the support of this automaton (i.e. has non-zero value under the automaton).
        /// </summary>
        /// <remarks>
        /// This should only be non-null if the distribution this automaton represents is improper and there is 
        /// a need to override the actual automaton value. 
        /// </remarks>
        public double? LogValueOverride 
        {
            get;
            // Setter should only ever be used in factory methods
            // TODO: replace with `init` after switching to C# 9.0+
            private set;
        }

        /// <summary>
        /// Gets a value for truncating small weights.
        /// If non-null, any transition whose weight falls below this value in a normalized
        /// automaton will be removed following a product operation.
        /// </summary>
        /// <remarks>
        /// TODO: We need to develop more elegant automaton approximation methods, this is a simple placeholder for those.
        /// </remarks>
        public double? PruneStatesWithLogEndWeightLessThan
        {
            get;
            // Setter should only ever be used in factory methods
            // TODO: replace with `init` after switching to C# 9.0+
            private set;
        }

        /// <summary>
        /// Gets or sets the maximum number of states an automaton can have. This setting is shared
        /// by all threads of the program.
        /// </summary>
        /// <remarks>
        /// This value can only be set because it is intended to be a program-level setting.
        /// For inspecting the value use the <see cref="MaxStateCount"/>.
        /// </remarks>
        public static int GlobalMaxStateCount
        {
            set
            {
                Argument.CheckIfInRange(value > 0, nameof(value), "The maximum number of states must be positive.");
                maxStateCount = value;
            }
        }

        public static int MaxStateCount
        {
            get => threadMaxStateCountOverride != 0 ? threadMaxStateCountOverride : maxStateCount;
            internal set => threadMaxStateCountOverride = value;
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
        /// Gets the start state of the automaton.
        /// </summary>
        /// <remarks>
        /// Only a state from <see cref="States"/> can be specified as the value of this property. 
        /// </remarks>
        public State Start => this.States[this.Data.StartStateIndex];

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates an automaton from a given array of states and a start state.
        /// Used by quoting to embed automata in the generated inference code.
        /// </summary>
        /// <param name="data">Quoted automaton state.</param>
        /// <returns>
        /// The created automaton.
        /// </returns>
        [Construction("Data")]
        public static TThis FromData(DataContainer data)
        {
            if (!data.IsConsistent())
            {
                throw new ArgumentException("Quoted data is not consistent");
            }

            return new TThis { Data = data };
        }

        /// <summary>
        /// Creates an automaton which maps every sequence to zero.
        /// </summary>
        /// <returns>The created automaton.</returns>
        public static TThis Zero() => ZeroAutomaton;

        /// <summary>
        /// Creates an automaton which maps every sequence to a given value.
        /// </summary>
        /// <param name="value">The value to map every sequence to.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Constant(double value)
        {
            return Constant(value, ElementDistributionFactory.CreateUniform());
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
            return ConstantLog(logValue, ElementDistributionFactory.CreateUniform());
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
            Argument.CheckIfNotNull(allowedElements, nameof(allowedElements));

            allowedElements = allowedElements.CreatePartialUniform();
            var builder = new Builder();
            if (!double.IsNegativeInfinity(logValue))
            {
                builder.Start.SetEndWeight(Weight.FromLogValue(logValue));
                builder.Start.AddTransition(allowedElements, Weight.FromLogValue(-allowedElements.GetLogAverageOf(allowedElements)), builder.StartStateIndex);
            }

            var result = Util.New<TThis>();
            result.Data = builder.GetData(true);
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
            var allowedElements = ElementDistributionFactory.CreatePointMass(element);
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
            Argument.CheckIfNotNull(allowedElements, nameof(allowedElements));

            var builder = new Builder();
            if (!double.IsNegativeInfinity(logValue))
            {
                allowedElements = allowedElements.CreatePartialUniform();
                var finish = builder.Start.AddTransition(allowedElements, Weight.FromLogValue(-allowedElements.GetLogAverageOf(allowedElements)));
                finish.SetEndWeight(Weight.FromLogValue(logValue));
            }

            var result = Util.New<TThis>();
            result.Data = builder.GetData(true);
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
            Argument.CheckIfNotNull(sequence, nameof(sequence));
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

            var builder = new Builder();
            int sequenceCount = 0;
            if (!double.IsNegativeInfinity(logValue))
            {
                foreach (var sequence in sequences)
                {
                    var sequenceEndState = builder.Start.AddTransitionsForSequence(sequence);
                    sequenceEndState.SetEndWeight(Weight.FromLogValue(logValue));
                    ++sequenceCount;
                }
            }

            var result = Util.New<TThis>();
            result.Data = builder.GetData(sequenceCount <= 1 ? (bool?)true : null);
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

            var result = new Builder();
            result.Start.SetEndWeight(Weight.FromLogValue(Math.Log(value)));
            return result.GetAutomaton();
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
            Argument.CheckIfNotNull(automaton1, nameof(automaton1));
            Argument.CheckIfNotNull(automaton2, nameof(automaton2));
            return automaton1.SumLog(logWeight1, automaton2, logWeight2);
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

            var result = new Builder();
            foreach (var automaton in automata)
            {
                if (automaton.IsCanonicZero())
                {
                    continue;
                }

                int index = result.StatesCount;
                result.AddStates(automaton.States);
                result.Start.AddEpsilonTransition(Weight.One, index + automaton.Start.Index);
            }

            return result.GetAutomaton();
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
                    result = Zero();
                    return result;
                }

                result = result.Product(automaton);
            }

            return result;
        }

        /// <summary>
        /// Creates an automaton which is the concatenation of given automata.
        /// </summary>
        /// <param name="automata">The automata to multiply.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Concatenate(params TThis[] automata)
        {
            return Concatenate((IEnumerable<TThis>)automata);
        }

        /// <summary>
        /// Creates an automaton which is the concatenation of given automata.
        /// </summary>
        /// <param name="automata">The automata to multiply.</param>
        /// <returns>The created automaton.</returns>
        public static TThis Concatenate(IEnumerable<TThis> automata)
        {
            Argument.CheckIfNotNull(automata, "automata");

            var builder = new Builder(1);
            builder.Start.SetEndWeight(Weight.One);

            foreach (var automaton in automata)
            {
                builder.Append(automaton);
            }

            return builder.GetAutomaton();
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
            Argument.CheckIfNotNull(sequenceToValue, nameof(sequenceToValue));

            return Sum(sequenceToValue.Select(kvp => ConstantOn(kvp.Value, kvp.Key)));
        }

        /// <summary>
        /// Creates an automaton which has given values on given sequences and is zero everywhere else.
        /// </summary>
        /// <param name="sequenceToLogValue">The collection of pairs of a sequence and the logarithm of the automaton value on that sequence.</param>
        /// <remarks>
        /// If the same sequence is presented in the collection of pairs multiple times,
        /// the value of the automaton on that sequence will be equal to the sum of the values in the collection.
        /// </remarks>
        /// <returns>The created automaton.</returns>
        public static TThis FromLogValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceToLogValue)
        {
            Argument.CheckIfNotNull(sequenceToLogValue, nameof(sequenceToLogValue));

            return Sum(sequenceToLogValue.Select(kvp => ConstantOnLog(kvp.Value, kvp.Key)));
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
            var result = Builder.ConstantOn(Weight.One, SequenceManipulator.ToSequence(new TElement[0]));

            // Build a list of all intermediate end states with their target ending weights while adding repetitions
            var endStatesWithTargetWeights = new List<(int, Weight)>();
            int prevStateCount = 0;
            for (int i = 0; i <= maxTimes; ++i)
            {
                // Remember added ending states
                if (repetitionNumberWeights[i] > 0)
                {
                    for (int j = prevStateCount; j < result.StatesCount; ++j)
                    {
                        var state = result[j];
                        if (state.CanEnd)
                        {
                            endStatesWithTargetWeights.Add(ValueTuple.Create(
                                state.Index,
                                Weight.FromValue(repetitionNumberWeights[i]) * state.EndWeight));
                        }
                    }
                }

                // Add one more repetition
                if (i != maxTimes)
                {
                    prevStateCount = result.StatesCount;
                    result.Append(automaton, avoidEpsilonTransitions: false);
                }
            }

            // Set target ending weights
            for (int i = 0; i < endStatesWithTargetWeights.Count; ++i)
            {
                var (stateIndex, weight) = endStatesWithTargetWeights[i];
                result[stateIndex].SetEndWeight(weight);
            }

            return result.GetAutomaton();
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

            var result = Builder.ConstantOn(Weight.One, emptySequence);
            for (int i = 0; i < minTimes; ++i)
            {
                result.Append(automaton);
            }

            var optionalPart = Builder.FromAutomaton(automaton);
            for (var i = 0; i < optionalPart.StatesCount; ++i)
            {
                var state = optionalPart[i];
                if (state.CanEnd)
                {
                    state.AddEpsilonTransition(state.EndWeight, optionalPart.StartStateIndex);
                }
            }

            result.Append(Sum(optionalPart.GetAutomaton(), ConstantOn(1.0, emptySequence)));

            return result.GetAutomaton();
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

            var builder = new StringBuilder();
            if (this.LogValueOverride.HasValue)
            {
                builder.Append(this.LogValueOverride);
                builder.Append(":");
            }

            var visitedStates = new HashSet<int>();
            var stack = new Stack<(string prefix, Option<TElementDistribution> prefixDistribution, int state)>();
            stack.Push((string.Empty, Option.None, Start.Index));

            while (stack.Count > 0)
            {
                var (prefix, prefixDistribution, stateIndex) = stack.Pop();

                builder.Append(prefix);
                if (prefixDistribution.HasValue)
                {
                    if (appendElement != null)
                    {
                        appendElement(prefixDistribution.Value, builder);
                    }
                    else
                    {
                        builder.Append(prefixDistribution);
                    }
                }

                if (stateIndex == -1)
                {
                    continue;
                }

                if (visitedStates.Contains(stateIndex))
                {
                    builder.Append('➥');
                    continue;
                }

                visitedStates.Add(stateIndex);

                var state = this.States[stateIndex];
                var transitions = state.Transitions.Where(t => !t.Weight.IsZero);
                var selfTransitions = transitions.Where(t => t.DestinationStateIndex == stateIndex);
                var selfTransitionCount = selfTransitions.Count();
                var nonSelfTransitions = transitions.Where(t => t.DestinationStateIndex != stateIndex).ToArray();
                var nonSelfTransitionCount = nonSelfTransitions.Count();

                if (state.CanEnd && nonSelfTransitionCount > 0)
                {
                    builder.Append('▪');
                }

                if (selfTransitionCount > 1)
                {
                    builder.Append('[');
                }

                var transIdx = 0;
                foreach (var transition in selfTransitions)
                {
                    if (!transition.IsEpsilon)
                    {
                        if (appendElement != null)
                        {
                            appendElement(transition.ElementDistribution.Value, builder);
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
                    stack.Push(("]", Option.None, -1));
                }

                // Iterate transitions in reverse order, because after destinations are pushed to stack,
                // order will be reversed again.
                for (transIdx = nonSelfTransitions.Length - 1; transIdx >= 0; --transIdx)
                {
                    var transition = nonSelfTransitions[transIdx];
                    var transitionPrefix = transIdx == 0 ? string.Empty : "|";
                    stack.Push((transitionPrefix, transition.ElementDistribution, transition.DestinationStateIndex));
                }
            }

            return builder.ToString();
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
            for (int stateIndex = 0; stateIndex < this.States.Count; stateIndex++)
            {
                var state = this.States[stateIndex];
                foreach (var transition in state.Transitions)
                {
                    if (transition.Group == group)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        public Dictionary<int, TThis> GetGroups() => GroupExtractor.ExtractGroups(this);

        /// <summary>
        /// Creates a copy of the current automaton with the group for all transitions cleared.
        /// </summary>
        public TThis WithGroupsClear() => WithGroup(0);

        /// <summary>
        /// Creates a copy of the current automaton with all transitions set to have the specified group.
        /// </summary>
        /// <param name="group">The specified group.</param>
        public TThis WithGroup(int group)
        {
            if (group == 0 && !this.UsesGroups)
            {
                return (TThis)this;
            }

            var builder = Builder.FromAutomaton(this);
            for (var i = 0; i < builder.StatesCount; ++i)
            {
                for (var iterator = builder[i].TransitionIterator; iterator.Ok; iterator.Next())
                {
                    iterator.Value = iterator.Value.With(group: group);
                }
            }

            return WithData(builder.GetData());
        }

        #endregion

        #region Normalization

        /// <summary>
        /// Computes the logarithm of the normalizer (sum of values of the automaton on all sequences).
        /// </summary>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <remarks>Returns <see cref="double.PositiveInfinity"/> if the sum diverges.</remarks>
        public double GetLogNormalizer() => DoGetLogNormalizer(false, out var _);

        /// <summary>
        /// Normalizes the automaton so that the sum of its values over all possible sequences equals to one
        /// and returns the logarithm of the normalizer.
        /// </summary>
        /// <returns>Will be set to the normalized automaton if the normalization was successful, or to the current automaton otherwise.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the automaton cannot be normalized (i.e. if the normalizer is zero or positive infinity).</exception>
        /// <remarks>The only automaton which cannot be normalized, but has a finite normalizer, is zero.</remarks>
        public TThis NormalizeValues()
        {
            if (!TryNormalizeValues(out var result))
            {
                throw new InvalidOperationException("This automaton cannot be normalized.");
            }

            return result;
        }

        /// <summary>
        /// Attempts to normalize the automaton so that sum of its values on all possible sequences equals to one (if it is possible)
        /// and returns the result in an out parameter.
        /// If successful, produces a stochastic automaton,
        /// i.e. an automaton, in which the sum of weights of all the outgoing transitions and the ending weight is 1 for every node.
        /// </summary>
        /// <param name="normalizedAutomaton">Will be set to the normalized automaton if the normalization was successful, or to the current automaton otherwise.</param>
        /// <param name="logNormalizer">When the function returns, contains the logarithm of the normalizer.</param>
        /// <returns><see langword="true"/> if the normalization was successful, <see langword="false"/> otherwise.</returns>
        public bool TryNormalizeValues(out TThis normalizedAutomaton, out double logNormalizer)
        {
            logNormalizer = DoGetLogNormalizer(true, out normalizedAutomaton);
            return !double.IsInfinity(logNormalizer);
        }

        /// <summary>
        /// Attempts to normalize the automaton so that sum of its values on all possible sequences equals to one (if it is possible).
        /// </summary>
        /// <param name="result">Will be set to the normalized automaton if the normalization was successful, or to the current automaton otherwise.</param>
        /// <returns><see langword="true"/> if the automaton was successfully normalized, <see langword="false"/> otherwise.</returns>>
        public bool TryNormalizeValues(out TThis result)
        {
            return TryNormalizeValues(out result, out var _);
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
            // Return cached value if available
            if (this.Data.IsZero.HasValue)
            {
                return this.Data.IsZero.Value;
            }

            // Calculate and cache whether this automaton is zero
            var isZero = DoIsZero();
            this.Data = this.Data.With(isZero: isZero);
            return isZero;

            bool DoIsZero()
            {
                var visited = new BitArray(this.States.Count, false);
                var stack = new Stack<int>();
                stack.Push(this.Start.Index);
                visited[this.Start.Index] = true;

                while (stack.Count > 0)
                {
                    var stateIndex = stack.Pop();

                    var state = this.States[stateIndex];
                    if (state.CanEnd)
                    {
                        return false;
                    }

                    foreach (var transition in state.Transitions)
                    {
                        if (!visited[transition.DestinationStateIndex] && !transition.Weight.IsZero)
                        {
                            stack.Push(transition.DestinationStateIndex);
                            visited[transition.DestinationStateIndex] = true;
                        }
                    }
                }

                return true;
            }
        }

        /// <summary>
        /// Checks whether the automaton is a canonic representation of zero,
        /// as produced by <see cref="Zero"/>.
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
            return this.Start.Transitions.Count == 0 && this.Start.EndWeight.IsZero;
        }

        /// <summary>
        /// Checks whether the automaton is a canonic representation of a constant,
        /// as produced by <see cref="ConstantLog(double)"/>.
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
            if (this.States.Count != 1 || this.Start.Transitions.Count != 1 || !this.Start.CanEnd)
            {
                return false;
            }

            var transitionDistribution = this.Start.Transitions[0].ElementDistribution;
            return transitionDistribution.HasValue && transitionDistribution.Value.IsUniform();
        }

        /// <summary>
        /// Creates an automaton constant on the support of the current automaton.
        /// </summary>
        /// <param name="value">The desired value on the support of the current automaton.</param>
        /// <returns>Resulting automaton.</returns>
        /// <remarks>This function will fail if the current automaton cannot be determinized.</remarks>
        public TThis ConstantOnSupport(double value)
        {
            if (value < 0)
            {
                throw new NotSupportedException("Negative values are not yet supported.");
            }

            return ConstantOnSupportLog(Math.Log(value));
        }

        /// <summary>
        /// Creates an automaton constant on the support of the current automaton.
        /// </summary>
        /// <param name="logValue">The logarithm of the desired value on the support of the current automaton.</param>
        /// <returns>Resulting automaton.</returns>
        /// <exception cref="NotImplementedException">Thrown if if the current automaton cannot be determinized.</exception>
        public TThis ConstantOnSupportLog(double logValue)
        {
            var success = TryGetConstantOnSupportLog(logValue, out var automaton);
            if (!success)
            {
                throw new NotImplementedException("Not yet supported for non-determinizable automata.");
            }

            return automaton;
        }

        /// <summary>
        /// Tries to create an automaton constant on the support of the current automaton.
        /// </summary>
        /// <param name="logValue">The logarithm of the desired value on the support of <paramref name="automaton"/>.</param>
        /// <param name="automaton">The resulting automaton.</param>
        /// <returns>True if successful, false otherwise.</returns>
        /// <remarks>This function will return false if the current automaton cannot be determinized.</remarks>
        public bool TryGetConstantOnSupportLog(double logValue, out TThis automaton)
        {
            Argument.CheckIfValid(!double.IsNaN(logValue), "logValue", "A valid value must be provided.");

            Weight value = Weight.FromLogValue(logValue);
            if (value.IsZero)
            {
                automaton = Zero();
                return true;
            }

            if (!TryDeterminize(out var determinizedAutomaton))
            {
                automaton = determinizedAutomaton;
                return false;
            }

            var result = Builder.FromAutomaton(determinizedAutomaton);

            for (int stateId = 0; stateId < result.StatesCount; ++stateId)
            {
                var state = result[stateId];
                if (state.CanEnd)
                {
                    // Make all accepting states contribute the desired value to the result
                    state.SetEndWeight(value);
                }

                for (var transitionIterator = state.TransitionIterator; transitionIterator.Ok; transitionIterator.Next())
                {
                    var transition = transitionIterator.Value;
                    if (!transition.Weight.IsZero)
                    {
                        if (transition.IsEpsilon)
                        {
                            transition = transition.With(weight: Weight.One);
                        }
                        else
                        {
                            transition = transition.With(
                                elementDistribution: transition.ElementDistribution.Value.CreatePartialUniform(),
                                weight: Weight.FromLogValue(-transition.ElementDistribution.Value.GetLogAverageOf(transition.ElementDistribution.Value)));
                        }

                        transitionIterator.Value = transition;
                    }
                }
            }

            automaton = Util.New<TThis>();
            automaton.Data = result.GetData(true);
            return true;
        }

        /// <summary>
        /// Creates a copy of the automaton.
        /// </summary>
        /// <returns>The created copy.</returns>
        public TThis Clone() => (TThis)this; // Automata are constant.

        /// <summary>
        /// Creates an automaton <c>f'(s) = f(reverse(s))</c>, 
        /// where <c>reverse</c> is a sequence reverse function.
        /// </summary>
        /// <returns>The created automaton.</returns>
        public TThis Reverse()
        {
            var result = new Builder();

            // Result already has 1 state, we add the remaining Count-1 states
            result.AddStates(this.States.Count - 1);

            // And the new start state
            result.StartStateIndex = result.AddState().Index;

            // The start state in the original automaton is going to be the one and only end state in result
            result[this.Start.Index].SetEndWeight(Weight.One);

            for (int i = 0; i < this.States.Count; ++i)
            {
                var oldState = this.States[i];
                foreach (var oldTransition in oldState.Transitions)
                {
                    // Result has original transitions reversed
                    result[oldTransition.DestinationStateIndex].AddTransition(
                        oldTransition.ElementDistribution, oldTransition.Weight, i);
                }

                // End states of the original automaton are the new start states
                if (oldState.CanEnd)
                {
                    result.Start.AddEpsilonTransition(oldState.EndWeight, i);
                }
            }

            return result.GetAutomaton();
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
            Argument.CheckIfNotNull(automaton, nameof(automaton));

            if (this.IsCanonicZero())
            {
                return (TThis)this;
            }

            if (automaton.IsCanonicZero())
            {
                return WithData(DataContainer.ZeroData);
            }

            var builder = Builder.FromAutomaton(this);
            var appendResult = builder.Append(automaton, group);

            var determinizationState =
                appendResult.preservedDeterminizationState && group == 0 && this.Data.IsDeterminized == true && automaton.Data.IsDeterminized == true
                ? (bool?)true
                : group != 0 || this.Data.IsDeterminized == false || automaton.Data.IsDeterminized == false ? (bool?)false : null;

            return WithData(builder.GetData(determinizationState));
        }

        /// <summary>
        /// Computes the product of the current automaton and a given one.
        /// </summary>
        /// <param name="automaton">The automaton to compute the product with.</param>
        /// <returns>The computed product.</returns>
        public TThis Product(TThis automaton)
        {
            return Product(automaton, true);
        }

        /// <summary>
        /// Computes the product of the current automaton and a given one.
        /// </summary>
        /// <param name="automaton">The automaton to compute the product with.</param>
        /// <param name="tryDeterminize">Whether to try to determinize the result.</param>
        /// <returns>The computed product.</returns>
        public TThis Product(TThis automaton, bool tryDeterminize)
        {
            Argument.CheckIfNotNull(automaton, nameof(automaton));

            var automaton1 = (TThis)this;
            var automaton2 = automaton;

            if (automaton1.IsCanonicZero() || automaton2.IsCanonicZero())
            {
                return Zero();
            }

            if (automaton1.UsesGroups)
            {
                // We cannot swap automaton 1 and automaton 2 as groups from first are used.
                if (!automaton2.IsEpsilonFree)
                {
                    automaton2 = automaton2.GetEpsilonClosure();
                }
            }
            else
            {
                // The second argument of BuildProduct must be epsilon-free
                if (!automaton1.IsEpsilonFree && !automaton2.IsEpsilonFree)
                {
                    automaton2 = automaton2.GetEpsilonClosure();
                }
                else if (automaton1.IsEpsilonFree && !automaton2.IsEpsilonFree)
                {
                    Util.Swap(ref automaton1, ref automaton2);
                }
            }

            var builder = new Builder(0);
            var (productStateCache, stack) = PreallocatedAutomataObjects.LeaseProductState();

            // Creates product state and schedules product computation for it.
            // If computation is already scheduled or done the state index is simply taken from cache
            int CreateProductState(State state1, State state2)
            {
                var destPair = new IntPair(state1.Index, state2.Index);
                if (!productStateCache.TryGetValue(destPair, out var productStateIndex))
                {
                    var productState = builder.AddState();
                    productState.SetEndWeight(state1.EndWeight * state2.EndWeight);
                    stack.Push((state1.Index, state2.Index, productState.Index));
                    productStateCache.Add(destPair, productState.Index);
                    productStateIndex = productState.Index;
                }

                return productStateIndex;
            }

            // Populate the stack with start product state
            builder.StartStateIndex = CreateProductState(automaton1.Start, automaton2.Start);

            while (stack.Count > 0)
            {
                var (state1Index, state2Index, productStateIndex) = stack.Pop();

                var state1 = automaton1.States[state1Index];
                var state2 = automaton2.States[state2Index];
                var productState = builder[productStateIndex];

                // Iterate over transitions in state1
                foreach (var transition1 in state1.Transitions)
                {
                    var destState1 = automaton1.States[transition1.DestinationStateIndex];

                    if (transition1.IsEpsilon)
                    {
                        // Epsilon transition case
                        var destProductStateIndex = CreateProductState(destState1, state2);
                        productState.AddEpsilonTransition(transition1.Weight, destProductStateIndex, transition1.Group);
                        continue;
                    }

                    // Iterate over transitions in state2
                    foreach (var transition2 in state2.Transitions)
                    {
                        Debug.Assert(
                            !transition2.IsEpsilon,
                            "The second argument of the product operation must be epsilon-free.");
                        var destState2 = automaton2.States[transition2.DestinationStateIndex];
                        var productLogNormalizer = transition1.ElementDistribution.Value.GetLogAverageOf(transition2.ElementDistribution.Value);
                        if (double.IsNegativeInfinity(productLogNormalizer))
                        {
                            continue;
                        }

                        var product = transition1.ElementDistribution.Value.Multiply(transition2.ElementDistribution.Value);
                        var productWeight = Weight.Product(
                            transition1.Weight, transition2.Weight, Weight.FromLogValue(productLogNormalizer));
                        var destProductStateIndex = CreateProductState(destState1, destState2);
                        productState.AddTransition(product, productWeight, destProductStateIndex, transition1.Group);
                    }
                }
            }

            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            simplification.RemoveDeadStates(); // Product can potentially create dead states
            simplification.SimplifyIfNeeded();

            var bothInputsDeterminized = automaton1.Data.IsDeterminized == true && automaton2.Data.IsDeterminized == true;
            var determinizationState = bothInputsDeterminized ? (bool?)true : null;

            var result = Util.New<TThis>();
            result.Data = builder.GetData(determinizationState);
            if (determinizationState != true && result is StringAutomaton && tryDeterminize)
            {
                result = result.TryDeterminize();
            }

            return result;
        }

        /// <summary>
        /// Merges the pruning thresholds for two automata.
        /// </summary>
        /// <param name="automaton1">The first automaton.</param>
        /// <param name="automaton2">The second automaton.</param>
        /// <returns>The merged pruning threshold.</returns>
        private double? MergePruningWeights(TThis automaton1, TThis automaton2)
        {
            if (automaton1.PruneStatesWithLogEndWeightLessThan == null)
            {
                return automaton2.PruneStatesWithLogEndWeightLessThan;
            }

            if (automaton2.PruneStatesWithLogEndWeightLessThan == null)
            {
                return automaton1.PruneStatesWithLogEndWeightLessThan;
            }

            return Math.Min(automaton1.PruneStatesWithLogEndWeightLessThan.Value, automaton2.PruneStatesWithLogEndWeightLessThan.Value);
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
            return SumLog(0.0, automaton, 0.0);
        }

        /// <summary>
        /// Computes the weighted sum of the current automaton and a given automaton.
        /// </summary>
        /// <param name="weightThis">The weight of the current automaton.</param>
        /// <param name="other">The given automaton.</param>
        /// <param name="weightOther">The weight of <paramref name="other"/>.</param>
        public TThis Sum(double weightThis, TThis other, double weightOther)
        {
            if (weightThis < 0 || weightOther < 0)
            {
                throw new NotImplementedException("Negative weights are not yet supported.");
            }

            return SumLog(Math.Log(weightThis), other, Math.Log(weightOther));
        }

        /// <summary>
        /// Computes the weighted sum of the current automaton and a given automaton.
        /// </summary>
        /// <param name="logWeightThis">The logarithm of the weight of the current automaton.</param>
        /// <param name="other">The given automaton.</param>
        /// <param name="logWeightOther">The logarithm of the weight of <paramref name="other"/>.</param>
        public TThis SumLog(double logWeightThis, TThis other, double logWeightOther)
        {
            Argument.CheckIfNotNull(other, nameof(other));
            Argument.CheckIfValid(
                !double.IsPositiveInfinity(logWeightThis) && !double.IsPositiveInfinity(logWeightOther),
                "Weights must not be infinite.");

            var result = new Builder();

            bool hasFirstTerm = !IsCanonicZero() && !double.IsNegativeInfinity(logWeightThis);
            bool hasSecondTerm = !other.IsCanonicZero() && !double.IsNegativeInfinity(logWeightOther);
            if (hasFirstTerm || hasSecondTerm)
            {
                if (hasFirstTerm)
                {
                    result.AddStates(States);
                    result.Start.AddEpsilonTransition(Weight.FromLogValue(logWeightThis), 1 + Start.Index);
                }

                if (hasSecondTerm)
                {
                    int cnt = result.StatesCount;
                    result.AddStates(other.States);
                    result.Start.AddEpsilonTransition(Weight.FromLogValue(logWeightOther), cnt + other.Start.Index);
                }
            }

            var simplification = new Simplification(result, this.PruneStatesWithLogEndWeightLessThan);
            simplification.SimplifyIfNeeded();

            return result.GetAutomaton();
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

            return ScaleLog(Math.Log(scale));
        }

        /// <summary>
        /// Scales the automaton and returns the result.
        /// </summary>
        /// <param name="logScale">The logarithm of the scale.</param>
        /// <returns>The scaled automaton.</returns>
        public TThis ScaleLog(double logScale)
        {
            return SumLog(logScale, Zero(), double.NegativeInfinity);
        }

        /// <summary>
        /// Creates an automaton obtained by performing a transition transformation
        /// on the current automaton.
        /// </summary>
        /// <typeparam name="TDstSequence">The type of a created automaton sequence.</typeparam>
        /// <typeparam name="TDstElement">The type of a created automaton sequence element.</typeparam>
        /// <typeparam name="TDstElementDistribution">The type of a distribution over created automaton sequence elements.</typeparam>
        /// <typeparam name="TDstSequenceManipulator">The type providing ways to manipulate created automaton sequences.</typeparam>
        /// <typeparam name="TDstAutomaton">The type of a created automaton.</typeparam>
        /// <param name="transitionTransform">The transition transformation.</param>
        public TDstAutomaton ApplyFunction<TDstSequence, TDstElement, TDstElementDistribution, TDstSequenceManipulator, TDstAutomaton>(
            Func<Option<TElementDistribution>, Weight, int, ValueTuple<Option<TDstElementDistribution>, Weight>> transitionTransform)
            where TDstElementDistribution : IImmutableDistribution<TDstElement, TDstElementDistribution>, CanGetLogAverageOf<TDstElementDistribution>, CanComputeProduct<TDstElementDistribution>, CanCreatePartialUniform<TDstElementDistribution>, SummableExactly<TDstElementDistribution>, new()
            where TDstSequence : class, IEnumerable<TDstElement>
            where TDstSequenceManipulator : ISequenceManipulator<TDstSequence, TDstElement>, new()
            where TDstAutomaton : Automaton<TDstSequence, TDstElement, TDstElementDistribution, TDstSequenceManipulator, TDstAutomaton>, new()
        {
            Argument.CheckIfNotNull(transitionTransform, nameof(transitionTransform));

            var builder = new Automaton<TDstSequence, TDstElement, TDstElementDistribution, TDstSequenceManipulator, TDstAutomaton>.Builder();

            // Add states
            builder.AddStates(States.Count - 1);

            // Copy state parameters and transitions
            for (int stateIndex = 0; stateIndex < States.Count; stateIndex++)
            {
                var thisState = builder[stateIndex];
                var otherState = States[stateIndex];

                thisState.SetEndWeight(otherState.EndWeight);
                if (otherState == Start)
                {
                    builder.StartStateIndex = thisState.Index;
                }

                foreach (var otherTransition in otherState.Transitions)
                {
                    var transformedTransition = transitionTransform(otherTransition.ElementDistribution, otherTransition.Weight, otherTransition.Group);
                    builder[stateIndex].AddTransition(
                        transformedTransition.Item1,
                        transformedTransition.Item2,
                        otherTransition.DestinationStateIndex,
                        otherTransition.Group);
                }
            }

            return builder.GetAutomaton();
        }

        /// <summary>
        /// Computes the logarithm of the value of the automaton on a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to compute the value on.</param>
        /// <returns>The logarithm of the value.</returns>
        /// <remarks></remarks>
        /// <remarks>Recursive implementation would be simpler but prone to stack overflows with large automata</remarks>
        public double GetLogValue(TSequence sequence)
        {
            Argument.CheckIfNotNull(sequence, nameof(sequence));

            if (Data.IsDeterminized == true)
                return GetLogValueDeterministic();
            else if (Data.IsEpsilonFree)
                return GetLogValueEpsilonFree();
            else
                return GetLogValueGeneral();

            double GetLogValueDeterministic()
            {
                Weight acc = Weight.One;
                State currentState = Start;
                foreach (TElement element in sequence)
                {
                    bool foundTransition = false;
                    foreach (Transition transition in currentState.Transitions)
                    {
                        var distWeight = Weight.FromLogValue(transition.ElementDistribution.Value.GetLogProb(element));
                        if (!distWeight.IsZero)
                        {
                            if (transition.Weight.IsZero)
                                return double.NegativeInfinity;

                            acc = Weight.Product(distWeight, transition.Weight, acc);
                            currentState = States[transition.DestinationStateIndex];
                            foundTransition = true;
                            break;
                        }
                    }

                    if (!foundTransition)
                        return double.NegativeInfinity;
                }
                if (currentState.CanEnd)
                {
                    acc *= currentState.EndWeight;
                    return
                        !acc.IsZero && this.LogValueOverride.HasValue
                            ? this.LogValueOverride.Value
                            : acc.LogValue;
                }
                else
                    return double.NegativeInfinity;
            }

            // A simplified version of the general algorithm below.
            // Skips allocation and computation of epsilon closures, which is not needed
            // in epsilon-free automaton case.
            double GetLogValueEpsilonFree()
            {
                var sequenceLength = SequenceManipulator.GetLength(sequence);
                var (operationsStack, valuesStack, valueCache) = PreallocatedAutomataObjects.LeaseGetLogValueState();
                operationsStack.Push((this.Start.Index, 0, Weight.One, -1));

                while (operationsStack.Count > 0)
                {
                    var (stateIndex, sequencePos, multiplier, sumUntil) = operationsStack.Pop();
                    var statePosPair = new IntPair(stateIndex, sequencePos);

                    if (sumUntil < 0)
                    {
                        if (valueCache.TryGetValue(statePosPair, out var cachedValue))
                        {
                            valuesStack.Push(cachedValue * multiplier);
                        }
                        else
                        {
                            var state = this.States[stateIndex];

                            if (sequencePos == sequenceLength)
                            {
                                // We are at the end of sequence. So put an answer on stack
                                valuesStack.Push(state.EndWeight * multiplier);
                                valueCache.Add(statePosPair, state.EndWeight);
                            }
                            else
                            {
                                // schedule second part of computation - sum values for all transitions
                                // Note: it is put on stack before operations for any transitions, that
                                // means that it will be executed after them
                                operationsStack.Push((stateIndex, sequencePos, multiplier, valuesStack.Count));

                                var element = SequenceManipulator.GetElement(sequence, sequencePos);
                                foreach (var transition in state.Transitions)
                                {
                                    var destStateIndex = transition.DestinationStateIndex;
                                    var distWeight = Weight.FromLogValue(transition.ElementDistribution.Value.GetLogProb(element));
                                    if (!distWeight.IsZero && !transition.Weight.IsZero)
                                    {
                                        var weightMul = Weight.Product(transition.Weight, distWeight);
                                        operationsStack.Push((destStateIndex, sequencePos + 1, weightMul, -1));
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // All transitions value from this state are already calculated, sum them and store on stack and in cache
                        var sum = Weight.Zero;
                        while (valuesStack.Count > sumUntil)
                        {
                            sum += valuesStack.Pop();
                        }

                        valuesStack.Push(multiplier * sum);
                        valueCache.Add(statePosPair, sum);
                    }
                }

                Debug.Assert(valuesStack.Count == 1);
                var result = valuesStack.Pop();
                return
                    !result.IsZero && this.LogValueOverride.HasValue
                        ? this.LogValueOverride.Value
                        : result.LogValue;
            }

            // This algorithm is unwinding of trivial recursion with cache. It encodes recursion through explicit
            // stack to avoid stack-overflow. It makes calculation not straightforward. Recursive algorithm that
            // we want to encode looks like this:
            //    GetLogValue(state, sequencePos) =
            //       closure = EpsilonClosure(state)
            //       if sequencePos == sequenceLength:
            //           return closure.EndWeight
            //       else:
            //           return sum([GetLogValue(transition.Destination, sequencePos+1) * transitionWeight'
            //                          for transition in closure.Transitions])
            //           where transitionWeight' = transition.Weight * closureStateWeight * transition.ElementDistribution.GetLogWeight(sequence[sequencePos])
            //
            // To encode that with explicit stack we split each GetLogValue() call into 2 parts which are put
            // on operations stack:
            // a) (stateIndex, sequencePos, mul, -1) -
            //    schedules computation for each transition from this state and (b) operation after them
            // b) (stateIndex, sequencePos, mul, sumUntil) -
            //    sums values of transitions from this state. They will be on top of the stack
            // As an optimization, value calculated by (b) is cached, so if (a) notices that result for
            // (stateIndex, sequencePos) is already calculated it doesn't schedule any extra work
            double GetLogValueGeneral()
            {
                var sequenceLength = SequenceManipulator.GetLength(sequence);
                var (operationsStack, valuesStack, valueCache) =
                    PreallocatedAutomataObjects.LeaseGetLogValueState();
                operationsStack.Push((this.Start.Index, 0, Weight.One, -1));

                while (operationsStack.Count > 0)
                {
                    var (stateIndex, sequencePos, multiplier, sumUntil) = operationsStack.Pop();
                    var statePosPair = new IntPair(stateIndex, sequencePos);

                    if (sumUntil < 0)
                    {
                        if (valueCache.TryGetValue(statePosPair, out var cachedValue))
                        {
                            valuesStack.Push(cachedValue * multiplier);
                        }
                        else
                        {
                            var closure = new EpsilonClosure(this, this.States[stateIndex]);

                            if (sequencePos == sequenceLength)
                            {
                                // We are at the end of sequence. So put an answer on stack
                                valuesStack.Push(closure.EndWeight * multiplier);
                                valueCache.Add(statePosPair, closure.EndWeight);
                            }
                            else
                            {
                                // schedule second part of computation - sum values for all transitions
                                // Note: it is put on stack before operations for any transitions, that
                                // means that it will be executed after them
                                operationsStack.Push((stateIndex, sequencePos, multiplier, valuesStack.Count));

                                var element = SequenceManipulator.GetElement(sequence, sequencePos);
                                for (var closureStateIndex = 0; closureStateIndex < closure.Size; ++closureStateIndex)
                                {
                                    var closureState = closure.GetStateByIndex(closureStateIndex);
                                    var closureStateWeight = closure.GetStateWeightByIndex(closureStateIndex);
                                    foreach (var transition in closureState.Transitions)
                                    {
                                        if (transition.IsEpsilon)
                                        {
                                            continue; // The destination is a part of the closure anyway
                                        }

                                        var destStateIndex = transition.DestinationStateIndex;
                                        var distWeight = Weight.FromLogValue(transition.ElementDistribution.Value.GetLogProb(element));
                                        if (!distWeight.IsZero && !transition.Weight.IsZero)
                                        {
                                            var weightMul = Weight.Product(closureStateWeight, transition.Weight, distWeight);
                                            operationsStack.Push((destStateIndex, sequencePos + 1, weightMul, -1));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // All transitions value from this state are already calculated, sum them and store on stack and in cache
                        var sum = Weight.Zero;
                        while (valuesStack.Count > sumUntil)
                        {
                            sum += valuesStack.Pop();
                        }

                        valuesStack.Push(multiplier * sum);
                        valueCache.Add(statePosPair, sum);
                    }
                }

                Debug.Assert(valuesStack.Count == 1);
                var result = valuesStack.Pop();
                return
                    !result.IsZero && this.LogValueOverride.HasValue
                        ? this.LogValueOverride.Value
                        : result.LogValue;
            }
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
        /// <remarks>Recursive implementation would be simpler but prone to stack overflows with large automata</remarks>
        public TSequence TryComputePoint()
        {
            // Need to get at least 2 support strings to be sure that this is not a point
            var enumerated = this.TryEnumerateSupport(2, out var support);
            return enumerated && support.Count() == 1
                ? support.Single()
                : null;
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
        /// Checks if <paramref name="other"/> is an automaton that defines the same weighted regular language.
        /// </summary>
        /// <param name="other">The automaton to compare this automaton with.</param>
        /// <returns><see langword="true"/> if this automaton is equal to <paramref name="other"/>, false otherwise.</returns>
        public bool Equals(TThis other)
        {
            double logSimilarity = GetLogSimilarity((TThis)this, other);
            const double LogSimilarityThreshold = -30; // Can't be -inf due to numerical instabilities in GetLogSimilarity()
            return logSimilarity < LogSimilarityThreshold;
        }

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

            return Equals((TThis)obj);
        }

        /// <summary>
        /// Gets the hash code of this automaton.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            TThis thisAutomaton = (TThis)this;
            // TODO: make consistent with Equals
            // now the hashcode depends on the maximum sum of outgoing transitions over states
            // (through GetConverger), which can differ for equal automata.
            TThis theConverger = GetConverger(thisAutomaton);
            thisAutomaton = thisAutomaton.Product(theConverger);
            double logNorm = thisAutomaton.Product(thisAutomaton).GetLogNormalizer();

            // Very similar automata will have almost identical norms and, thus, will almost always be hashed to the same bucket
            return (BitConverter.DoubleToInt64Bits(logNorm) >> 31).GetHashCode();
        }

        #endregion

        #region Manipulating structure

        /// <summary>
        /// Gets a value indicating whether this automaton is epsilon-free.
        /// </summary>
        public bool IsEpsilonFree => this.Data.IsEpsilonFree;

        /// <summary>
        /// Determines whether this automaton has groups.
        /// </summary>
        /// <value>True if it the automaton has groups, false otherwise.</value>
        public bool UsesGroups => this.Data.UsesGroups;

        /// <summary>
        /// Tests whether the automaton is deterministic,
        /// i.e. it's epsilon free and for every state and every element there is at most one transition that allows for that element.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the automaton is deterministic,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public bool IsDeterministic()
        {
            if (Data.IsDeterminized.HasValue)
            {
                return Data.IsDeterminized.Value;
            }
            
            if (!this.IsEpsilonFree)
            {
                // Should not set isDeterminized to false here,
                // only when we determine that the automaton is not determinizable
                return false;
            }
            
            for (int stateId = 0; stateId < this.States.Count; ++stateId)
            {
                var state = this.States[stateId];

                // There should be no epsilon transitions
                foreach (var transition in state.Transitions)
                {
                    if (transition.IsEpsilon)
                    {
                        // Should not set isDeterminized to false here,
                        // only when we determine that the automaton is not determinizable
                        return false;
                    }
                }
                
                // Element distributions should not intersect
                var transitions = state.Transitions;
                for (int transitionIndex1 = 0; transitionIndex1 < transitions.Count; ++transitionIndex1)
                {
                    var transition1 = transitions[transitionIndex1];
                    for (int transitionIndex2 = transitionIndex1 + 1; transitionIndex2 < transitions.Count; ++transitionIndex2)
                    {
                        var transition2 = transitions[transitionIndex2];
                        double logProductNormalizer = transition1.ElementDistribution.Value.GetLogAverageOf(transition2.ElementDistribution.Value);
                        if (!double.IsNegativeInfinity(logProductNormalizer))
                        {
                            // Should not set isDeterminized to false here,
                            // only when we determine that the automaton is not determinizable
                            return false;
                        }
                    }
                }
            }

            Data = Data.With(isDeterminized: true);
            return true;
        }

        /// <summary>
        /// Returns an epsilon closure of the current automaton.
        /// </summary>
        /// <remarks>
        /// The resulting automaton will be equal to the current one, but have no epsilon transitions.
        /// </remarks>
        public TThis GetEpsilonClosure()
        {
            if (IsEpsilonFree)
            {
                return (TThis)this;
            }

            var builder = new Builder(0);
            var oldToNewState = new int?[States.Count];
            var stack = new Stack<int>();

            // Enqueues state for processing if necessary and returns its index in new automaton
            int Enqueue(int oldStateIndex)
            {
                if (oldToNewState[oldStateIndex] != null)
                {
                    return oldToNewState[oldStateIndex].Value;
                }

                var newStateIndex = builder.AddState().Index;
                oldToNewState[oldStateIndex] = newStateIndex;
                stack.Push(oldStateIndex);
                return newStateIndex;
            }

            builder.StartStateIndex = Enqueue(Start.Index);
            while (stack.Count > 0)
            {
                var oldStateIndex = stack.Pop();
                var closure = new EpsilonClosure(this, States[oldStateIndex]);
                var resultState = builder[oldToNewState[oldStateIndex].Value];

                resultState.SetEndWeight(closure.EndWeight);
                for (var stateIndex = 0; stateIndex < closure.Size; ++stateIndex)
                {
                    var closureState = closure.GetStateByIndex(stateIndex);
                    var closureStateWeight = closure.GetStateWeightByIndex(stateIndex);
                    foreach (var transition in closureState.Transitions)
                    {
                        if (transition.IsEpsilon)
                        {
                            continue;
                        }

                        var closureDestStateIndex = Enqueue(transition.DestinationStateIndex);
                        resultState.AddTransition(
                            transition.ElementDistribution,
                            transition.Weight * closureStateWeight,
                            closureDestStateIndex,
                            transition.Group);
                    }
                }
            }

            var result = WithData(builder.GetData());

            return result;
        }

        #endregion

        #region Helpers

        protected TThis WithData(DataContainer data)
        {
            var result = Util.New<TThis>();
            result.Data = data;
            result.LogValueOverride = LogValueOverride;
            result.PruneStatesWithLogEndWeightLessThan = PruneStatesWithLogEndWeightLessThan;
            return result;
        }

        /// <summary>
        /// Creates a copy of the current automaton with a different value of <see cref="LogValueOverride"/>.
        /// </summary>
        /// <param name="logValueOverride">New <see cref="LogValueOverride"/>.</param>
        /// <returns>The created automaton.</returns>
        public TThis WithLogValueOverride(double? logValueOverride)
        {
            var result = Util.New<TThis>();
            result.Data = Data;
            result.LogValueOverride = logValueOverride;
            result.PruneStatesWithLogEndWeightLessThan = PruneStatesWithLogEndWeightLessThan;
            return result;
        }

        /// <summary>
        /// Creates a copy of the current automaton with a different value of <see cref="PruneStatesWithLogEndWeightLessThan"/>.
        /// </summary>
        /// <param name="pruneStatesWithLogEndWeightLessThan">New <see cref="PruneStatesWithLogEndWeightLessThan"/>.</param>
        /// <returns>The created automaton.</returns>
        public TThis WithPruneStatesWithLogEndWeightLessThan(double? pruneStatesWithLogEndWeightLessThan)
        {
            var result = Util.New<TThis>();
            result.Data = Data;
            result.LogValueOverride = LogValueOverride;
            result.PruneStatesWithLogEndWeightLessThan = pruneStatesWithLogEndWeightLessThan;
            return result;
        }

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

            TThis theConverger = GetConverger(new TThis[] {automaton1, automaton2});
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
        /// <param name="decayWeight">The decay weight.</param>
        /// <returns>An automaton, product with which will make every given automaton normalizable.</returns>
        public static TThis GetConverger(TThis automata, double decayWeight = 0.99)
        {
            return GetConverger(new TThis[] {automata}, decayWeight);
        }

        /// <summary>
            /// Gets an automaton such that every given automaton, if multiplied by it, becomes normalizable.
            /// </summary>
            /// <param name="automata">The automata.</param>
            /// <param name="decayWeight">The decay weight.</param>
            /// <returns>An automaton, product with which will make every given automaton normalizable.</returns>
            public static TThis GetConverger(TThis[] automata, double decayWeight = 0.99)
        {
            // TODO: This method might not work in the presense of non-trivial loops.

            // Find the maximum total weight of outgoing transitions
            double maxLogTransitionWeightSum = double.NegativeInfinity;
            for (int automatonIndex = 0; automatonIndex < automata.Length; ++automatonIndex)
            {
                TThis automaton = automata[automatonIndex];

                for (int stateIndex = 0; stateIndex < automaton.States.Count; ++stateIndex)
                {
                    State state = automaton.States[stateIndex];
                    Weight transitionWeightSum = Weight.Zero;
                    foreach (var transition in state.Transitions)
                    {
                        transitionWeightSum += transition.Weight;
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
            var uniformDist = ElementDistributionFactory.CreateUniform();
            var theConverger = new Builder();
            Weight transitionWeight = Weight.Product(
                Weight.FromLogValue(-uniformDist.GetLogAverageOf(uniformDist)),
                Weight.FromLogValue(-maxLogTransitionWeightSum),
                Weight.FromValue(decayWeight));
            theConverger.Start.AddSelfTransition(uniformDist, transitionWeight);
            theConverger.Start.SetEndWeight(Weight.One);

            return theConverger.GetAutomaton();
        }

        /// <summary>
        /// Computes the logarithm of the normalizer of the automaton, normalizing it afterwards if requested.
        /// </summary>
        /// <param name="normalize">Specifies whether a normalized automaton must be computed after computing the normalizer.</param>
        /// <param name="normalizedAutomaton">Will be set to normalized automaton if <paramref name="normalize"/> is <see langword="true"/>
        /// and the normalizer has a finite non-zero value. ot to the current automaton otherwise.</param>
        /// <returns>The logarithm of the normalizer and condensation of automaton.</returns>
        private double DoGetLogNormalizer(bool normalize, out TThis normalizedAutomaton)
        {
            TThis noEpsilonTransitions = GetEpsilonClosure();

            using (var condensation = noEpsilonTransitions.ComputeCondensation(noEpsilonTransitions.Start, tr => true, false))
            {
                double logNormalizer = condensation.GetWeightToEnd(noEpsilonTransitions.Start.Index).LogValue;
                normalizedAutomaton = (TThis)this;
                if (normalize)
                {
                    if (!double.IsInfinity(logNormalizer))
                    {
                        normalizedAutomaton = WithData(PushWeights());
                    }
                }

                return logNormalizer;

                // Re-distributes weights of the states and transitions so that the underlying automaton becomes stochastic
                // (i.e. sum of weights of all the outgoing transitions and the ending weight is 1 for every node).
                DataContainer PushWeights()
                {
                    var builder = Builder.FromAutomaton(noEpsilonTransitions);
                    for (int i = 0; i < builder.StatesCount; ++i)
                    {
                        var state = builder[i];
                        var weightToEnd = condensation.GetWeightToEnd(i);
                        if (weightToEnd.IsZero)
                        {
                            continue; // End states cannot be reached from this state, no point in normalization
                        }

                        var weightToEndInv = Weight.Inverse(weightToEnd);
                        for (var transitionIterator = state.TransitionIterator; transitionIterator.Ok; transitionIterator.Next())
                        {
                            var transition = transitionIterator.Value;
                            transitionIterator.Value = transition.With(
                                weight: Weight.Product(
                                    transition.Weight,
                                    condensation.GetWeightToEnd(transition.DestinationStateIndex),
                                    weightToEndInv));
                        }

                        state.SetEndWeight(state.EndWeight * weightToEndInv);
                    }

                    return builder.GetData();
                }
            }
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
                var newWeight = weight * currentState.EndWeight;
                yield return new Tuple<List<TElementDistribution>, double>(prefix.Reverse().ToList(), newWeight.LogValue);
            }

            visitedStates[stateIndex] = true;
            foreach (var transition in currentState.Transitions)
            {
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

                foreach (var support in this.EnumeratePaths(prefix, visitedStates, weight * transition.Weight, transition.DestinationStateIndex))
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
            private readonly int originalThreadMaxStateCount;

            /// <summary>
            /// Initializes a new instance of the <see cref="UnlimitedStatesComputation"/> class.
            /// </summary>
            public UnlimitedStatesComputation()
            {
                this.originalThreadMaxStateCount = threadMaxStateCountOverride;
                threadMaxStateCountOverride = int.MaxValue;
            }

            /// <summary>
            /// Verifies that the resulting automaton is within the range for MaxStateCount.
            /// </summary>
            public void CheckStateCount(TThis automaton)
            {
                var limit = this.originalThreadMaxStateCount != 0
                    ? this.originalThreadMaxStateCount
                    : maxStateCount;
                if (automaton.States.Count > limit)
                {
                    throw new AutomatonTooLargeException(limit);
                }
            }

            public void Dispose()
            {
                threadMaxStateCountOverride = this.originalThreadMaxStateCount;
                PreallocatedAutomataObjects.FreeAllObjects();
            }
        }
        #endregion


        #region IWeightFunction implementation

        /// <inheritdoc/>
        public TThis AsAutomaton() => (TThis)this;

        /// <inheritdoc/>
        public TSequence Point
        { 
            get => TryComputePoint() ?? throw new InvalidOperationException("This automaton is zero everywhere or is non-zero on more than one sequence.");
        }

        /// <inheritdoc/>
        public bool IsPointMass => TryEnumerateSupport(2, out var support) && support.Count() == 1;

        /// <inheritdoc/>
        public bool UsesAutomatonRepresentation => true;

        /// <inheritdoc/>
        public TThis Repeat(int minTimes = 1, int? maxTimes = null) => Repeat((TThis)this, minTimes, maxTimes);

        /// <inheritdoc/>
        public double MaxDiff(TThis that) => Math.Exp(GetLogSimilarity((TThis)this, that));

        /// <inheritdoc/>
        public TThis NormalizeStructure() => (TThis)this;

        #endregion

        #region Serialization

        /// <summary>
        /// Writes the current automaton.
        /// </summary>
        /// <remarks>
        /// Serialization format is a bit unnatural, but we do it for compatibility with old serialized data.
        /// So we don't have to maintain 2 versions of deserialization.
        /// </remarks>
        public void Write(Action<double> writeDouble, Action<int> writeInt32, Action<TElementDistribution> writeElementDistribution)
        {
            var propertyMask = new BitVector32();
            var idx = 0;
            propertyMask[1 << idx++] = true; // isEpsilonFree is always known
            propertyMask[1 << idx++] = this.Data.IsEpsilonFree;
            propertyMask[1 << idx++] = this.LogValueOverride.HasValue;
            propertyMask[1 << idx++] = this.PruneStatesWithLogEndWeightLessThan.HasValue;
            propertyMask[1 << idx++] = true; // start state is always serialized
            propertyMask[1 << idx++] = this.Data.IsDeterminized.HasValue;
            propertyMask[1 << idx++] = this.Data.IsDeterminized.HasValue ? this.Data.IsDeterminized.Value : false;

            writeInt32(propertyMask.Data);

            if (this.LogValueOverride.HasValue)
            {
                writeDouble(this.LogValueOverride.Value);
            }

            if (this.PruneStatesWithLogEndWeightLessThan.HasValue)
            {
                writeDouble(this.PruneStatesWithLogEndWeightLessThan.Value);
            }

            // This state is serialized only for its index.
            this.Start.Write(writeDouble, writeInt32, writeElementDistribution);

            writeInt32(this.States.Count);
            foreach (var state in this.States)
            {
                state.Write(writeDouble, writeInt32, writeElementDistribution);
            }
        }

        /// <summary>
        /// Reads an automaton from.
        /// </summary>
        /// <remarks>
        /// Serialization format is a bit unnatural, but we do it for compatibility with old serialized data.
        /// So we don't have to maintain 2 versions of deserialization.
        /// </remarks>
        public static TThis Read(Func<double> readDouble, Func<int> readInt32, Func<TElementDistribution> readElementDistribution)
        {
            var propertyMask = new BitVector32(readInt32());
            var idx = 0;
            // Serialized "isEpsilonFree" is not used. Will be taken from builder anyway
            var hasEpsilonFreeIgnored = propertyMask[1 << idx++];
            var isEpsilonFreeIgnored = propertyMask[1 << idx++];
            var hasLogValueOverride = propertyMask[1 << idx++];
            var hasPruneWeights = propertyMask[1 << idx++];
            var hasStartState = propertyMask[1 << idx++];
            var hasIsDeterminizedValue = propertyMask[1 << idx++];
            var isDeterminized = propertyMask[1 << idx++];
            double? logValueOverride = null, pruneStatesWithLogEndWeightLessThan = null;

            if (hasLogValueOverride)
            {
                logValueOverride = readDouble();
            }

            if (hasPruneWeights)
            {
                pruneStatesWithLogEndWeightLessThan = readDouble();
            }

            var builder = new Builder(0);

            if (hasStartState)
            {
                // Start state is also present in the list of all states, so read it into temp builder where
                // it will get index 0. But keep real deserialized start state index to be used in real builder
                var tempBuilder = new Builder(0);
                builder.StartStateIndex =
                    State.ReadTo(ref tempBuilder, readInt32, readDouble, readElementDistribution, checkIndex: false);
            }

            var numStates = readInt32();

            for (var i = 0; i < numStates; i++)
            {
                State.ReadTo(ref builder, readInt32, readDouble, readElementDistribution);
            }

            return new TThis()
            { 
                Data = builder.GetData(hasIsDeterminizedValue ? (bool?)isDeterminized : null),
                LogValueOverride = logValueOverride,
                PruneStatesWithLogEndWeightLessThan = pruneStatesWithLogEndWeightLessThan
            };
        }
        #endregion
    }
}

