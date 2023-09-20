// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;

    using Distributions.Automata;
    using Factors.Attributes;
    using Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Utilities;

    /// <summary>
    /// A base class for implementations of distributions over sequences.
    /// </summary>
    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The type of a sequence element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
    /// <typeparam name="TAutomaton">The type of a weighted finite state automaton that can be used to represent every valid function mapping sequences to weights.</typeparam>
    /// <typeparam name="TWeightFunction">The type of an underlying function mapping sequences to weights.</typeparam>
    /// <typeparam name="TWeightFunctionFactory">The type of the factory for <typeparamref name="TWeightFunction"/>.</typeparam>
    /// <typeparam name="TThis">The type of a concrete distribution class.</typeparam>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public abstract class SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis> :
        IDistribution<TSequence>,
        SettableTo<TThis>,
        SettableToProduct<TThis>,
        SettableToRatio<TThis>,
        SettableToPower<TThis>,
        CanGetLogAverageOf<TThis>,
        CanGetLogAverageOfPower<TThis>,
        CanGetAverageLog<TThis>,
        SettableToWeightedSumExact<TThis>,
        SettableToPartialUniform<TThis>,
        CanGetLogNormalizer,
        Sampleable<TSequence>
        where TSequence : class, IEnumerable<TElement>
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, Sampleable<TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TWeightFunction : WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.IWeightFunction<TWeightFunction>, new()
        where TWeightFunctionFactory : WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.IWeightFunctionFactory<TWeightFunction>, new()
        where TThis : SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis>, new()
    {
        #region Fields & constants

        /// <summary>
        /// A sequence manipulator.
        /// </summary>
        private static readonly TSequenceManipulator SequenceManipulator = new TSequenceManipulator();

        private static readonly TWeightFunctionFactory WeightFunctionFactory = new TWeightFunctionFactory();

        /// <summary>
        /// A function mapping sequences to weights (non-normalized probabilities).
        /// </summary>
        [DataMember]
        private TWeightFunction sequenceToWeight = default;

        /// <summary>
        /// Specifies whether the <see cref="sequenceToWeight"/> is normalized.
        /// </summary>
        [DataMember]
        private bool isNormalized;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TAutomaton,TWeightFunction,TWeightFunctionFactory,TThis}"/> class
        /// with a null weight function.  The weight function must be set by the subclass constructor or factory method.
        /// </summary>
        protected SequenceDistribution()
        {
        }

        #endregion

        #region Properties

        private static TElementDistribution ElementDistributionFactory =>
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ElementDistributionFactory;

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public TSequence Point
        {
            get => sequenceToWeight.Point;

            set
            {
                Argument.CheckIfNotNull(value, "value", "Point mass must not be null.");
                
                sequenceToWeight = WeightFunctionFactory.PointMass(value);
                this.isNormalized = true;
            }
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution represents a point mass.
        /// </summary>
        public bool IsPointMass => sequenceToWeight.IsPointMass;

        /// <summary>
        /// Gets a value indicating whether the current distribution is represented as an automaton internally.
        /// </summary>
        public bool UsesAutomatonRepresentation => sequenceToWeight.UsesAutomatonRepresentation;

        /// <summary>
        /// Gets a value indicating whether the current distribution
        /// puts all probability mass on the empty sequence.
        /// </summary>
        public bool IsEmpty
        {
            get { return this.IsPointMass && SequenceManipulator.GetLength(this.Point) == 0; }
        }

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates a point mass distribution.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns>The created point mass distribution.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static TThis PointMass(TSequence point)
        {
            Argument.CheckIfNotNull(point, "point", "Point mass must not be null.");

            var result = Util.New<TThis>();
            result.Point = point;
            return result;
        }

        /// <summary>
        /// Creates an improper distribution which assigns the probability of 1 to every sequence.
        /// </summary>
        /// <returns>The created uniform distribution.</returns>
        [Construction(UseWhen = "IsUniform")]
        [Skip]
        public static TThis Uniform()
        {
            var result = Util.New<TThis>();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates an improper distribution which assigns zero probability to every sequence.
        /// </summary>
        /// <returns>The created zero distribution.</returns>
        [Construction(UseWhen = "IsZero")]
        public static TThis Zero()
        {
            var result = Util.New<TThis>();
            result.SetToZero();
            return result;
        }

        /// <summary>
        /// Creates a distribution from a given weight (non-normalized probability) function.
        /// </summary>
        /// <param name="weightFunction">The weight function specifying the distribution.</param>
        /// <returns>The created distribution.</returns>
        [Construction(nameof(GetWeightFunction))]
        public static TThis FromWeightFunction(TWeightFunction weightFunction)
        {
            Argument.CheckIfNotNull(weightFunction, nameof(weightFunction));

            var result = Util.New<TThis>();
            result.SetWeightFunction(weightFunction);
            return result;
        }

        /// <inheritdoc cref="FromWeightFunction(TWeightFunction)"/>
        [Construction(nameof(ToAutomaton), UseWhen = nameof(UsesAutomatonRepresentation))]
        public static TThis FromWeightFunction(TAutomaton weightFunction)
            => FromWeightFunction(WeightFunctionFactory.FromAutomaton(weightFunction));

        /// <summary>
        /// Creates a distribution which puts all probability mass on the empty sequence.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsEmpty")]
        public static TThis Empty()
        {
            return PointMass(SequenceManipulator.ToSequence(new TElement[0]));
        }

        /// <summary>
        /// Creates a distribution over sequences of length 1 induced by a given distribution over sequence elements.
        /// </summary>
        /// <param name="elementDistribution">The distribution over sequence elements.</param>
        /// <returns>The created distribution.</returns>
        /// <remarks>
        /// The distribution created by this method can differ from the result of
        /// <see cref="Repeat(TElementDistribution, int, int?, DistributionKind)"/> with both min and max number of times to repeat set to 1 since the latter always creates a partial uniform distribution.
        /// </remarks>
        public static TThis SingleElement(TElementDistribution elementDistribution)
        {
            var func = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Builder();
            var end = func.Start.AddTransition(elementDistribution, Weight.One);
            end.SetEndWeight(Weight.One);
            return FromWeightFunction(func.GetAutomaton());
        }

        /// <summary>
        /// Creates a distribution which puts all probability mass on a sequence containing only a given element.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <returns>The created distribution.</returns>
        public static TThis SingleElement(TElement element)
        {
            var sequence = SequenceManipulator.ToSequence(new List<TElement> { element });
            return PointMass(sequence);
        }

        /// <summary>
        /// Creates a distribution over sequences induced by a given list of distributions over sequence elements
        /// where the sequence can optionally end at any length, and the last element can optionally repeat without limit.
        /// </summary>
        /// <param name="elementDistributions">Enumerable of distributions over sequence elements and the transition weights.</param>
        /// <param name="allowEarlyEnd">Allow the sequence to end at any point.</param>
        /// <param name="repeatLastElement">Repeat the last element.</param>
        /// <returns>The created distribution.</returns>
        public static TThis Concatenate(IEnumerable<TElementDistribution> elementDistributions, bool allowEarlyEnd = false, bool repeatLastElement = false)
        {
            var result = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Builder();
            var last = result.Start;
            var elementDistributionArray = elementDistributions.ToArray();
            for (var i = 0; i < elementDistributionArray.Length - 1; i++)
            {
                last = last.AddTransition(elementDistributionArray[i], Weight.One);
                if (allowEarlyEnd)
                {
                    last.SetEndWeight(Weight.One);
                }
            }

            var lastElement = elementDistributionArray[elementDistributionArray.Length - 1];
            if (repeatLastElement)
            {
                last.AddSelfTransition(lastElement, Weight.One);
            }
            else
            {
                last = last.AddTransition(lastElement, Weight.One);
            }

            last.SetEndWeight(Weight.One);
            return FromWeightFunction(result.GetAutomaton());
        }

        /// <summary>
        /// Creates a distribution which is a uniform mixture of a given set of distributions.
        /// </summary>
        /// <param name="distributions">The set of distributions to create a mixture from.</param>
        /// <param name="cloneIfSizeOne">If the set has one element, whether to return that element or to clone it.</param>
        /// <param name="skipNormalization">Allow normalization to be skipped as it can be expensive</param>
        /// <returns>The created mixture distribution.</returns>
        public static TThis OneOf(IEnumerable<TThis> distributions, bool cloneIfSizeOne = false, bool skipNormalization = false)
        {
            Argument.CheckIfNotNull(distributions, "distributions");

            var enumerable = distributions as IReadOnlyList<TThis> ?? distributions.ToList();
            if (enumerable.Count == 1)
            {
                if (!cloneIfSizeOne)
                {
                    return enumerable[0];
                }

                return enumerable[0].Clone();
            }

            if (!skipNormalization)
            {
                foreach (var d in enumerable)
                    d.EnsureNormalized();
            }

            var probFunctions = enumerable.Select(d => d.sequenceToWeight);
            var result = Util.New<TThis>();
            result.sequenceToWeight = WeightFunctionFactory.Sum(probFunctions).NormalizeStructure();
            return result;
        }

        /// <summary>
        /// Creates a distribution which is a uniform mixture of a given set of distributions.
        /// </summary>
        /// <param name="distributions">The set of distributions to create a mixture from.</param>
        /// <returns>The created mixture distribution.</returns>
        public static TThis OneOf(params TThis[] distributions)
        {
            return OneOf((IEnumerable<TThis>)distributions);
        }

        /// <summary>
        /// Creates a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        /// <returns>The created mixture distribution.</returns>
        public static TThis OneOf(double weight1, TThis dist1, double weight2, TThis dist2)
        {
            var result = Util.New<TThis>();
            result.SetToSum(weight1, dist1, weight2, dist2);
            return result;
        }

        /// <summary>
        /// Creates a distribution which assigns specified probabilities to given sequences.
        /// Probabilities do not have to be normalized.
        /// </summary>
        /// <param name="sequenceProbPairs">A list of (sequence, probability) pairs.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(IEnumerable<KeyValuePair<TSequence, double>> sequenceProbPairs)
        {
            var result = Util.New<TThis>();
            result.sequenceToWeight = WeightFunctionFactory.FromValues(sequenceProbPairs).NormalizeStructure();
            return result;
        }

        /// <summary>
        /// Creates a distribution which is uniform over a given set of sequences.
        /// </summary>
        /// <param name="sequences">The set of sequences to create a distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(IEnumerable<TSequence> sequences)
        {
            Argument.CheckIfNotNull(sequences, nameof(sequences));

            return OneOf(sequences.Select(s => new KeyValuePair<TSequence, double>(s, 1.0)));
        }

        /// <summary>
        /// Creates a distribution which is uniform over a given set of sequences.
        /// </summary>
        /// <param name="sequences">The set of sequences to create a distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(params TSequence[] sequences)
        {
            return OneOf((IEnumerable<TSequence>)sequences);
        }

        /// <summary>
        /// Creates a uniform distribution over sequences of length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible sequence length.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Any(int minLength = 0, int? maxLength = null)
        {
            return Repeat(ElementDistributionFactory.CreateUniform(), minLength, maxLength); ;
        }

        /// <summary>
        /// Creates a uniform distribution over sequences of length within the given bounds.
        /// Sequence elements are restricted to be equal to a given element.
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <param name="minTimes">The minimum possible sequence length. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <param name="uniformity">The type of uniformity.</param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TElement element, int minTimes = 1, int? maxTimes = null, DistributionKind uniformity = DistributionKind.UniformOverValue)
        {
            var elementDistribution = ElementDistributionFactory.CreatePointMass(element);
            return Repeat(elementDistribution, minTimes, maxTimes, uniformity);
        }

        /// <summary>
        /// Creates a uniform distribution over sequences containing N repeats of the 
        /// specified sequence, where n lies within the given bounds.
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the number of repeats, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="sequence">The sequence to repeat.</param>
        /// <param name="minTimes">The minimum possible sequence length. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TSequence sequence, int minTimes = 1, int? maxTimes = null)
        {
            return Repeat(PointMass(sequence), minTimes, maxTimes);
        }

        /// <summary>
        /// Creates a uniform distribution over sequences of length within the given bounds.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <param name="minTimes">The minimum possible sequence length. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <param name="uniformity">The type of uniformity</param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TElementDistribution allowedElements, int minTimes = 1, int? maxTimes = null, DistributionKind uniformity = DistributionKind.UniformOverValue)
        {
            Argument.CheckIfNotNull(allowedElements, nameof(allowedElements));
            Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of times to repeat must be non-negative.");
            Argument.CheckIfInRange(!maxTimes.HasValue || maxTimes.Value >= 0, "maxTimes", "The maximum number of times to repeat must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || minTimes <= maxTimes.Value, "The minimum length cannot be greater than the maximum length.");
            Argument.CheckIfValid(uniformity != DistributionKind.UniformOverLengthThenValue || maxTimes.HasValue, "Maximum length must be set for uniform over lengths then values option.");

            //// TODO: delegate to Repeat(TThis, int, int?)

            if (maxTimes.HasValue)
            {
                if (minTimes == 0 && maxTimes.Value == 0)
                {
                    return Empty();
                }

                if (allowedElements.IsPointMass && (minTimes == maxTimes.Value))
                {
                    return PointMass(SequenceManipulator.ToSequence(Enumerable.Repeat(allowedElements.Point, minTimes)));
                }
            }

            allowedElements = allowedElements.CreatePartialUniform();
            double distLogNormalizer = -allowedElements.GetLogAverageOf(allowedElements);
            var weight = uniformity == DistributionKind.UniformOverLengthThenValue
                ? Weight.One
                : Weight.FromLogValue(distLogNormalizer);

            var func = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Builder();
            var state = func.Start;

            int iterationBound = maxTimes.HasValue ? maxTimes.Value : minTimes;
            var currentDistLogNormalizer = distLogNormalizer;
            for (int i = 0; i <= iterationBound; i++)
            {
                bool isLengthAllowed = i >= minTimes;
                state.SetEndWeight(isLengthAllowed ? Weight.One : Weight.Zero);
                if (i < iterationBound)
                {
                    state = state.AddTransition(allowedElements, weight); // todo: clone set?    
                }
            }

            if (!maxTimes.HasValue)
            {
                state.AddSelfTransition(allowedElements, weight);
            }

            return FromWeightFunction(func.GetAutomaton());
        }

        /// <summary>
        /// <para>
        /// Creates a distribution by applying <see cref="Automaton{TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis}.Repeat(TThis, int, int?)"/>
        /// to the weight function of a given distribution,
        /// which is additionally scaled by the inverse of <see cref="GetLogAverageOf"/> with itself.
        /// So, if the given distribution is partial uniform, the result will be partial uniform over the repetitions of
        /// sequences covered by the distribution.
        /// </para>
        /// <para>
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </para>
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="minTimes">The minimum number of repetitions. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TThis dist, int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfNotNull(dist, nameof(dist));
            Argument.CheckIfInRange(minTimes >= 0, nameof(minTimes), "The minimum number of repetitions must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || maxTimes.Value >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

            double logNormalizer = -dist.GetLogAverageOf(dist);

            return FromWeightFunction(dist.sequenceToWeight.ScaleLog(logNormalizer).Repeat(minTimes, maxTimes));
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TThis, int, int?)"/> with the minimum number of repetitions set to 0.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TThis dist, int? maxTimes = null)
        {
            return Repeat(dist, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TThis, int, int?)"/> with the minimum number of repetitions set to 0.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TSequence sequence, int? maxTimes = null)
        {
            return Repeat(sequence, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElement, int, int?, DistributionKind)"/> with the minimum number of repetitions set to 0
        /// and uniformity kind set to <see cref="DistributionKind.UniformOverValue"/>.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TElement element, int? maxTimes = null)
        {
            return Repeat(element, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElementDistribution, int, int?, DistributionKind)"/> with the minimum number of repetitions set to 0
        /// and uniformity kind set to <see cref="DistributionKind.UniformOverValue"/>.
        /// </summary>
        /// <param name="allowedElements">The allowed sequence elements.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TElementDistribution allowedElements, int? maxTimes = null)
        {
            return Repeat(allowedElements, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TThis, int, int?)"/> with the minimum number of repetitions set to 1.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TThis dist, int? maxTimes = null)
        {
            return Repeat(dist, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TThis, int, int?)"/> with the minimum number of repetitions set to 1.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TSequence sequence, int? maxTimes = null)
        {
            return Repeat(sequence, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElement, int, int?, DistributionKind)"/> with the minimum number of repetitions set to 1
        /// and uniformity kind set to <see cref="DistributionKind.UniformOverValue"/>.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TElement element, int? maxTimes = null)
        {
            return Repeat(element, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElementDistribution, int, int?, DistributionKind)"/> with the minimum number of repetitions set to 0
        /// and uniformity kind set to <see cref="DistributionKind.UniformOverValue"/>.
        /// </summary>
        /// <param name="allowedElements">The allowed sequence elements.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TElementDistribution allowedElements, int? maxTimes = null)
        {
            return Repeat(allowedElements, maxTimes: maxTimes);
        }

        /// <summary>
        /// Creates a mixture of a given distribution and a point mass representing an empty sequence.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="prob">The probability of the component corresponding to <paramref name="dist"/>.</param>
        /// <returns>The created mixture.</returns>
        public static TThis Optional(TThis dist, double prob = 0.5)
        {
            return OneOf(prob, dist, 1 - prob, Empty());
        }

        /// <summary>
        /// Creates a mixture of a point mass on the given sequence and a point mass representing an empty sequence.
        /// </summary>
        /// <param name="sequence">The given sequence.</param>
        /// <param name="prob">The probability of the component corresponding to <paramref name="sequence"/>.</param>
        /// <returns>The created mixture.</returns>
        public static TThis Optional(TSequence sequence, double prob = 0.5)
        {
            return OneOf(prob, PointMass(sequence), 1 - prob, Empty());
        }

        #endregion

        #region Manipulation

        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution and a given element.
        /// </summary>
        /// <param name="element">The element to append.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append the given element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences and the element.</returns>
        public TThis Append(TElement element, int group = 0)
        {
            return this.Append(SingleElement(element), group);
        }
        
        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution
        /// and elements from a given distribution.
        /// </summary>
        /// <param name="elementDistribution">The distribution to generate the elements from.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// 2) Sample a random element from <paramref name="elementDistribution"/>.
        /// </description></item>
        /// <item><description>
        /// 3) Append the sampled element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences and elements.</returns>
        public TThis Append(TElementDistribution elementDistribution, int group = 0)
        {
            return this.Append(SingleElement(elementDistribution), group);
        }

        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution and a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append <paramref name="sequence"/> to it and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences.</returns>
        public TThis Append(TSequence sequence, int group = 0)
        {
            return this.Append(PointMass(sequence), group);
        }

        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution
        /// and sequences from a given distribution.
        /// </summary>
        /// <param name="dist">The distribution over the sequences to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Sample a random sequence from <paramref name="dist"/>.
        /// </description></item>
        /// <item><description>
        /// Output the concatenation of the sampled pair of sequences.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences.</returns>
        public TThis Append(TThis dist, int group = 0)
        {
            Argument.CheckIfNotNull(dist, "dist");
            
            TThis result = this.Clone();
            result.AppendInPlace(dist, group);
            return result;
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and a given element.
        /// </summary>
        /// <param name="element">The element to append.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append the given element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TElement element, int group = 0)
        {
            this.AppendInPlace(SingleElement(element), group);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and elements from a given distribution.
        /// </summary>
        /// <param name="elementDistribution">The distribution to generate the elements from.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Sample a random element from <paramref name="elementDistribution"/>.
        /// </description></item>
        /// <item><description>
        /// Append the sampled element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TElementDistribution elementDistribution, int group = 0)
        {
            Argument.CheckIfValid(elementDistribution != null, nameof(elementDistribution));
            
            this.AppendInPlace(SingleElement(elementDistribution), group);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append <paramref name="sequence"/> to it and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TSequence sequence, int group = 0)
        {
            Argument.CheckIfNotNull(sequence, "sequence");
            
            this.AppendInPlace(PointMass(sequence), group);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and sequences from a given distribution.
        /// </summary>
        /// <param name="dist">The distribution over the sequences to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Sample a random sequence from <paramref name="dist"/>.
        /// </description></item>
        /// <item><description>
        /// Output the concatenation of the sampled pair of sequences.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TThis dist, int group = 0)
        {
            Argument.CheckIfNotNull(dist, nameof(dist));

            SetWeightFunction(sequenceToWeight.Append(dist.sequenceToWeight, group));
        }

        /// <inheritdoc cref="SetWeightFunction(TWeightFunction,bool)"/>
        public void SetWeightFunction(TAutomaton weightFunction, bool normalizeStructure = true)
            => SetWeightFunction(WeightFunctionFactory.FromAutomaton(weightFunction), normalizeStructure);

        /// <summary>
        /// Replaces the weight function of the current distribution with a given one.
        /// </summary>
        /// <param name="weightFunction">The weight function to replace the current one with.</param>
        /// <param name="normalizeStructure">Whether to normalize the structure of the distribution</param>
        public void SetWeightFunction(TWeightFunction weightFunction, bool normalizeStructure = true)
        {
            Argument.CheckIfNotNull(weightFunction, nameof(weightFunction));

            this.sequenceToWeight = weightFunction;
            this.isNormalized = false;

            if (normalizeStructure)
            {
                this.NormalizeStructure();
            }
        }

        /// <summary>
        /// Returns an automaton representing the underlying weight function.
        /// </summary>
        /// <remarks>
        /// The returned weight function might differ from the probability function by an arbitrary scale factor.
        /// To ensure that the weight function is normalized, use <see cref="ToNormalizedAutomaton"/>.
        /// </remarks>
        /// <returns>An automaton representing the underlying weight function.</returns>
        public TAutomaton ToAutomaton()
        {
            return sequenceToWeight.AsAutomaton();
        }

        /// <summary>
        /// Returns an automaton representing the underlying weight function.
        /// </summary>
        /// <remarks>
        /// The returned weight function is guaranteed to be normalized.
        /// If you don't care about an arbitrary scale factor, 
        /// consider using <see cref="ToAutomaton"/> which has less overhead.
        /// </remarks>
        /// <returns>An automaton representing the underlying weight function.</returns>
        public TAutomaton ToNormalizedAutomaton()
        {
            this.EnsureNormalized();
            return this.ToAutomaton();
        }

        /// <summary>
        /// Returns the underlying weight function.
        /// </summary>
        /// <remarks>
        /// The returned weight function might differ from the probability function by an arbitrary scale factor.
        /// To ensure that the weight function is normalized, use <see cref="GetNormalizedWeightFunction"/>.
        /// </remarks>
        /// <returns>The underlying weight function.</returns>
        public TWeightFunction GetWeightFunction() => sequenceToWeight;


        /// <summary>
        /// Returns the underlying weight function.
        /// </summary>
        /// <remarks>
        /// The returned weight function is guaranteed to be normalized.
        /// If you don't care about an arbitrary scale factor, 
        /// consider using <see cref="GetWeightFunction"/> which has less overhead.
        /// </remarks>
        /// <returns>The underlying weight function.</returns>
        public TWeightFunction GetNormalizedWeightFunction()
        {
            EnsureNormalized();
            return sequenceToWeight;
        }

        /// <summary>
        /// Enumerates support of this distribution when possible.
        /// Only point mass elements are supported.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="tryDeterminize">Try to determinize if this distribution uses
        /// a string automaton representation.</param>
        /// <exception cref="AutomatonException">Thrown if enumeration is too large.</exception>
        /// <returns>The strings supporting this distribution</returns>
        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true)
        {
            if (tryDeterminize)
                TryDeterminizeBeforeSupportEnumeration();

            return this.sequenceToWeight.EnumerateSupport(maxCount);
        }

        /// <summary>
        /// Enumerates support of this distribution when possible.
        /// Only point mass elements are supported.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="result">The strings supporting this distribution.</param>
        /// <param name="tryDeterminize">Try to determinize if this distribution uses
        /// a string automaton representation.</param>
        /// <exception cref="AutomatonException">Thrown if enumeration is too large.</exception>
        /// <returns>True if successful, false otherwise.</returns>
        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true)
        {
            if (tryDeterminize)
                TryDeterminizeBeforeSupportEnumeration();

            return this.sequenceToWeight.TryEnumerateSupport(maxCount, out result);
        }

        /// <summary>
        /// Enumerates components of this distribution.
        /// </summary>
        /// <remarks>
        /// Any sequence distribution can be represented as a mixture of 'simple' sequence
        /// distributions where each element is independent.  This method enumerates through
        /// the components of this distribution treated as such a mixture.  Each component
        /// is returned as a list of the independent element distributions for each element.
        /// 
        /// For some distributions, the number of components may be very large.
        /// </remarks>
        /// <returns>For each component, the list of element distributions and the log mixture weight for that component.</returns>
        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumerateComponents()
        {
            return this.sequenceToWeight.EnumeratePaths();
        }

        /// <summary>
        /// Tries to replace the internal representation of the current distribution with an equivalent
        /// deterministic automaton, unless the current representation is guaranteed to produce
        /// a deterministic automaton on conversion.
        /// </summary>
        /// <returns><see langword="true"/> if the internal representation of the current distribution
        /// was successfully replaced with a deterministic automaton or is guaranteed to produce one
        /// on direct conversion, <see langword="false"/> otherwise.</returns>
        public bool TryDeterminize()
        {
            if (IsPointMass || (this is StringDistribution && !UsesAutomatonRepresentation))
            {
                return true;
            }

            var determinizationSuccess = ToNormalizedAutomaton().TryDeterminize(out var determinizationResult);
            SetWeightFunction(determinizationResult, false);
            return determinizationSuccess;
        }

        /// <summary>
        /// Sets a value that, if not null, will be returned when computing the log probability of any sequence
        /// which is in the support of this distribution.
        /// </summary>
        /// <param name="logValueOverride">A non-null value that should be returned when computing the log probability of any sequence
        /// which is in the support of this distribution, or <see langword="null"/> to clear any existing override.</param>
        public void SetLogValueOverride(double? logValueOverride)
        {
            // No need to clear logValueOverride fof non-automaton representations
            if (logValueOverride.HasValue || UsesAutomatonRepresentation)
            {
                var workspace = ToAutomaton().WithLogValueOverride(logValueOverride);
                // Normalizing structure can be useful only when logValueOverride == null
                SetWeightFunction(workspace, !logValueOverride.HasValue);
            }
        }

        /// <summary>
        /// Computes a distribution <c>g(b) = sum_a f(a) T(a, b)</c>, where <c>f(a)</c> is the current distribution and <c>T(a, b)</c> is a given transducer.
        /// </summary>
        /// <param name="transducer">The transducer to project on.</param>
        /// <returns>The projection.</returns>
        public TThis ApplyTransducer<TTransducer>(TTransducer transducer) where TTransducer : Transducer<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TTransducer>, new()
            => FromWeightFunction(IsPointMass ? transducer.ProjectSource(Point) : transducer.ProjectSource(ToAutomaton()));

        #endregion

        #region ToString

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <returns>
        /// A string that represents the distribution.
        /// </returns>
        public override string ToString()
        {
            return this.ToString((Action<TElementDistribution, StringBuilder>)null);
        }

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <param name="appendRegex">Optional method for appending at the element distribution level.</param>
        /// <returns>A string that represents the automaton.</returns>
        protected string ToString(Action<TElementDistribution, StringBuilder> appendRegex) => sequenceToWeight.ToString(appendRegex);

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <param name="format">The format.</param>
        /// <returns>A string that represents the distribution.</returns>
        public string ToString(ISequenceDistributionFormat format)
        {
            Argument.CheckIfNotNull(format, nameof(format));
            return format.ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis>((TThis)this);
        }

        #endregion

        #region Groups

        public bool HasGroup(int group)
        {
            return sequenceToWeight.HasGroup(group);
        }
   
        public Dictionary<int, TThis> GetGroups()
        {
            // TODO: get rid of groups or do something about groups + point mass combo
            return this.sequenceToWeight.GetGroups().ToDictionary(x => x.Key, x => FromWeightFunction(x.Value));
        }

        #endregion

        #region IDistribution implementation

        /// <summary>
        /// Replaces the current distribution by a copy of the given distribution.
        /// </summary>
        /// <param name="that">The distribution to set the current distribution to.</param>
        public void SetTo(TThis that)
        {
            Argument.CheckIfNotNull(that, nameof(that));
            
            if (ReferenceEquals(this, that))
            {
                return;
            }

            this.sequenceToWeight = that.sequenceToWeight;
            this.isNormalized = that.isNormalized;
        }

        /// <summary>
        /// Replaces the current distribution by an improper zero distribution.
        /// </summary>
        public void SetToZero()
        {
            this.sequenceToWeight = WeightFunctionFactory.Zero();
            this.isNormalized = true;
        }

        /// <summary>
        /// Returns a product of the current distribution and a given one.
        /// </summary>
        /// <param name="that">The distribution to compute the product with.</param>
        /// <returns>The product.</returns>
        public TThis Product(TThis that)
        {
            Argument.CheckIfNotNull(that, "that");

            var result = Util.New<TThis>();
            result.SetToProduct((TThis)this, that);
            return result;
        }

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions.
        /// </summary>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        public void SetToProduct(TThis dist1, TThis dist2)
        {
            Argument.CheckIfNotNull(dist1, "dist1");
            Argument.CheckIfNotNull(dist2, "dist2");
            
            this.DoSetToProduct(dist1, dist2, computeLogNormalizer: false);
        }

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions
        /// Returns the logarithm of the normalizer for the product (as returned by <see cref="GetLogAverageOf"/>).
        /// </summary>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        /// <param name="tryDeterminize">Whether to try to determinize the result.</param>
        /// <returns>The logarithm of the normalizer for the product.</returns>
        public double SetToProductAndReturnLogNormalizer(TThis dist1, TThis dist2, bool tryDeterminize = true)
        {
            Argument.CheckIfNotNull(dist1, "dist1");
            Argument.CheckIfNotNull(dist2, "dist2");
            
            double? logNormalizer = this.DoSetToProduct(dist1, dist2, computeLogNormalizer: true, tryDeterminize : tryDeterminize);
            Debug.Assert(logNormalizer.HasValue, "Log-normalizer wasn't computed.");
            return logNormalizer.Value;
        }

        /// <summary>
        /// Returns the logarithm of the probability that the current distribution would draw the same sample
        /// as a given one.
        /// </summary>
        /// <param name="that">The given distribution.</param>
        /// <returns>The logarithm of the probability that distributions would draw the same sample.</returns>
        public double GetLogAverageOf(TThis that)
        {
            Argument.CheckIfNotNull(that, nameof(that));

            var result = Util.New<TThis>();
            return result.SetToProductAndReturnLogNormalizer((TThis)this, that, false);
        }

        /// <summary>
        /// Computes the log-integral of one distribution times another raised to a power.
        /// </summary>
        /// <param name="that">The other distribution</param>
        /// <param name="power">The exponent</param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x) * Math.Pow(that.Evaluate(x), power))</c></returns>
        /// <remarks>
        /// <para>
        /// This is not the same as GetLogAverageOf(that^power) because it includes the normalization constant of that.
        /// </para>
        /// <para>Powers other than 1 are not currently supported.</para>
        /// </remarks>
        public double GetLogAverageOfPower(TThis that, double power)
        {
            if (power == 1.0)
            {
                return this.GetLogAverageOf(that);
            }
            
            throw new NotImplementedException("GetLogAverageOfPower() is not implemented for non-unit power.");
        }

        /// <summary>
        /// Computes the expected logarithm of a given distribution under this distribution.
        /// Not currently supported.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(TThis that)
        {
            throw new NotSupportedException("GetAverageLog() is not supported for sequence distributions.");
        }

        /// <summary>
        /// Returns the logarithm of the normalizer of the exponential family representation of this distribution.
        /// Normalizer of an improper distribution is defined to be 1.
        /// </summary>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <remarks>Getting the normalizer is currently supported for improper distributions only.</remarks>
        public double GetLogNormalizer()
        {
            if (!this.IsProper())
            {
                return 0;
            }

            throw new NotSupportedException("GetLogNormalizer() is not supported for proper distributions.");
        }

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        public void SetToSum(double weight1, TThis dist1, double weight2, TThis dist2)
        {
            Argument.CheckIfValid(!double.IsNaN(weight1), "weight1", "A weight must be a valid number.");
            Argument.CheckIfValid(!double.IsNaN(weight2), "weight2", "A weight must be a valid number.");
            Argument.CheckIfValid(weight1 + weight2 >= 0, "The sum of the weights must not be negative.");
            
            if (weight1 < 0 || weight2 < 0)
            {
                throw new NotImplementedException("Negative weights are not yet supported.");
            }

            this.SetToSumLog(Math.Log(weight1), dist1, Math.Log(weight2), dist2);
        }

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="logWeight1">The logarithm of the weight of the first distribution.</param>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="logWeight2">The logarithm of the weight of the second distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        public void SetToSumLog(double logWeight1, TThis dist1, double logWeight2, TThis dist2)
        {
            Argument.CheckIfNotNull(dist1, "dist1");
            Argument.CheckIfNotNull(dist2, "dist2");
            Argument.CheckIfValid(!double.IsPositiveInfinity(logWeight1) || !double.IsPositiveInfinity(logWeight2), "Both weights are infinite.");
            
            if (double.IsNegativeInfinity(logWeight1) && double.IsNegativeInfinity(logWeight2))
            {
                this.SetToUniform();
                return;
            }

            if (double.IsPositiveInfinity(logWeight1) || double.IsNegativeInfinity(logWeight2))
            {
                this.SetTo(dist1);
                return;
            }

            if (double.IsPositiveInfinity(logWeight2) || double.IsNegativeInfinity(logWeight1))
            {
                this.SetTo(dist2);
                return;
            }

            double weightSum = MMath.LogSumExp(logWeight1, logWeight2);
            logWeight1 -= weightSum;
            logWeight2 -= weightSum;

            dist1.EnsureNormalized();
            dist2.EnsureNormalized();

            SetWeightFunction(dist1.sequenceToWeight.SumLog(logWeight1, dist2.sequenceToWeight, logWeight2));
        }

        /// <summary>
        /// Replaces the current distribution with a given distribution raised to a given power.
        /// </summary>
        /// <param name="that">The distribution to raise to the power.</param>
        /// <param name="power">The power.</param>
        /// <remarks>Only 0 and 1 are currently supported as powers.</remarks>
        public void SetToPower(TThis that, double power)
        {
            if (power == 0.0)
            {
                this.SetToUniform();
                return;
            }
            
            if (power != 1.0)
            {
                throw new NotSupportedException("SetToPower() is not supported for powers other than 0 and 1.");
            }
        }

        /// <summary>
        /// Replaces the current distribution with the ratio of a given pair of distributions. Not currently supported.
        /// </summary>
        /// <param name="numerator">The numerator in the ratio.</param>
        /// <param name="denominator">The denominator in the ratio.</param>
        /// <param name="forceProper">Specifies whether the ratio must be proper.</param>
        public void SetToRatio(TThis numerator, TThis denominator, bool forceProper)
        {
            throw new NotSupportedException("SetToRatio() is not supported for sequence distributions.");
        }

        /// <summary>
        /// Creates a copy of the current distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        object ICloneable.Clone()
        {
            return this.Clone();
        }

        /// <summary>
        /// Creates a copy of the current distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public TThis Clone()
        {
            var result = Util.New<TThis>();
            result.SetTo((TThis)this);
            return result;
        }

        /// <summary>
        /// Gets a value indicating how close this distribution is to a given one
        /// in terms of probabilities they assign to sequences.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>A non-negative value, which is close to zero if the two distribution assign similar values to all sequences.</returns>
        public double MaxDiff(object that)
        {
            TThis thatDistribution = that as TThis;
            if (thatDistribution == null)
            {
                return double.PositiveInfinity;
            }

            EnsureNormalized();
            thatDistribution.EnsureNormalized();
            
            return sequenceToWeight.MaxDiff(thatDistribution.sequenceToWeight);
        }

        /// <summary>
        /// Gets the logarithm of the probability of a given sequence under this distribution.
        /// If the distribution is improper, returns the logarithm of the value of the underlying unnormalized weight function.
        /// </summary>
        /// <param name="sequence">The sequence to get the probability for.</param>
        /// <returns>The logarithm of the probability of the sequence.</returns>
        public double GetLogProb(TSequence sequence)
        {
            Argument.CheckIfNotNull(sequence, "sequence");

            this.EnsureNormalized();
            return this.sequenceToWeight.GetLogValue(sequence);
        }

        /// <summary>
        /// Returns true if the given sequence is in the support of this distribution (i.e. has non-zero probability).
        /// </summary>
        /// <param name="sequence">The sequence to check.</param>
        /// <returns>True if the sequence has non-zero probability under this distribiton, false otherwise.</returns>
        public bool Contains(TSequence sequence)
        {
            Argument.CheckIfNotNull(sequence, "sequence");

            // todo: stop GetLogValue early as soon as we know the weight is not neg infinity.
            var logWeight = this.sequenceToWeight.GetLogValue(sequence);
            return !double.IsNegativeInfinity(logWeight);
        }

        /// <summary>
        /// Gets the probability of a given sequence under this distribution.
        /// If the distribution is improper, returns the value of the underlying unnormalized weight function.
        /// </summary>
        /// <param name="sequence">The sequence to get the probability for.</param>
        /// <returns>The probability of the sequence.</returns>
        public double GetProb(TSequence sequence)
        {
            return Math.Exp(this.GetLogProb(sequence));
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <param name="result">A pre-allocated storage for the sample (will be ignored).</param>
        /// <returns>The drawn sample.</returns>
        public TSequence Sample(TSequence result)
        {
            return this.Sample();
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <returns>The drawn sample.</returns>
        public TSequence Sample()
        {
            if (this.IsPointMass)
            {
                return this.Point;
            }

            TThis properDist = (TThis)this;
            if (!this.IsProper())
            {
                var weightFunction = this.ToAutomaton();
                var theConverger =
                    Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.GetConverger(weightFunction);
                var properWeightFunction = weightFunction.Product(theConverger);
                properDist = FromWeightFunction(properWeightFunction);
            }

            return SampleFromProperDistribution(properDist);
        }

        /// <summary>
        /// Draws a sample from the proper distribution.
        /// </summary>
        /// <returns>The drawn sample.</returns>
        private static TSequence SampleFromProperDistribution(TThis dist)
        {
            dist.EnsureNormalized();

            var sampledElements = new List<TElement>();
            var automaton = dist.sequenceToWeight.AsAutomaton();
            var currentState = automaton.Start;
            while (true)
            {
                double logSample = Math.Log(Rand.Double());
                Weight probSum = Weight.Zero;
                foreach (var transition in currentState.Transitions)
                {
                    probSum += transition.Weight;
                    if (logSample < probSum.LogValue)
                    {
                        if (!transition.IsEpsilon)
                        {
                            sampledElements.Add(transition.ElementDistribution.Sample());
                        }

                        currentState = automaton.States[transition.DestinationStateIndex];
                        break;
                    }
                }

                if (logSample >= probSum.LogValue)
                {
                    Debug.Assert(!currentState.EndWeight.IsZero, "This state must have a non-zero ending probability.");
                    return SequenceManipulator.ToSequence(sampledElements);
                }
            }
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns the probability of 1 to every sequence, or a different value if <see cref="SetLogValueOverride"/> was previously called.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        public void SetToUniformOf(TElementDistribution allowedElements)
        {
            this.SetToUniformOf(allowedElements, 0.0);
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns the probability of 1 to every sequence, or a different value if <see cref="SetLogValueOverride"/> was previously called.
        /// </summary>
        public void SetToUniform()
        {
            this.SetToUniformOf(ElementDistributionFactory.CreateUniform());
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        /// <remarks>This function will fail if the weight function of this distribution
        /// is represented by an automaton that cannot be determinized.</remarks>
        public void SetToPartialUniform()
        {
            this.SetToPartialUniformOf((TThis)this);
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="dist">The distribution which support will be used to setup the current distribution.</param>
        /// <remarks>This function will fail if the weight function of <paramref name="dist"/>
        /// is represented by an automaton that cannot be determinized.</remarks>
        public void SetToPartialUniformOf(TThis dist)
        {
            var resultWeightFunction = WeightFunctionFactory.ConstantOnSupportOfLog(0.0, dist.GetWeightFunction());
            SetWeightFunction(resultWeightFunction, false);
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution is uniform over its support, <see langword="false"/> otherwise.</returns>
        public bool IsPartialUniform()
        {
            if (this.IsPointMass)
            {
                return true;
            }

            if (IsCanonicUniform())
            {
                // This should be much faster for distributions which are canonic uniform.
                return true;
            }

            TThis partialUniform = this.Clone();
            partialUniform.SetToPartialUniform();
            return this.Equals(partialUniform);
        }

        /// <summary>
        /// Checks whether the current distribution is proper.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is proper, <see langword="false"/> otherwise.</returns>
        public bool IsProper()
        {
            return !double.IsInfinity(this.sequenceToWeight.GetLogNormalizer()); // TODO: cache?
        }
        
        /// <summary>
        /// Checks whether the current distribution is uniform over all possible sequences.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is uniform over all possible sequences, <see langword="false"/> otherwise.</returns>
        public bool IsUniform()
        {
            if (this.IsPointMass)
            {
                return false;
            }

            if (IsCanonicUniform())
            {
                // This should be much faster for distributions which are canonic uniform.
                return true;
            }

            TThis canonicUniform = Uniform();
            // todo: check if the below works if the distribution is not normalised
            return this.Equals(canonicUniform);
        }

        /// <summary>
        /// Returns true if this distribution has identical structure to the distribution returned by Uniform().
        /// If it returns true, the distribution is uniform.  If it returns false, the distribution may or may not
        /// be uniform.
        /// </summary>
        /// <returns>True if the distribution is canonically uniform</returns>
        private bool IsCanonicUniform()
        {
            return this.ToAutomaton().IsCanonicConstant();
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution
        /// is an improper distribution which assigns zero probability to every sequence.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the current distribution
        /// is an improper distribution that assigns zero probability to every sequence, <see langword="false"/> otherwise.
        /// </returns>
        public bool IsZero()
        {
            return this.sequenceToWeight.IsZero();
        }

        /// <summary>
        /// Converges an improper sequence distribution
        /// </summary>
        /// <param name="dist">The original distribution.</param>
        /// <param name="decayWeight">The decay weight.</param>
        /// <returns>The converged distribution.</returns>
        public static TThis Converge(TThis dist, double decayWeight = 0.99)
        {
            var converger =
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>
                    .GetConverger(new TAutomaton[]
                    {
                        dist.sequenceToWeight.AsAutomaton()
                    }, decayWeight);
            return dist.Product(FromWeightFunction(converger));
        }

        /// <summary>
        /// Checks if <paramref name="obj"/> equals to this distribution (i.e. represents the same distribution over sequences).
        /// </summary>
        /// <param name="obj">The object to compare this distribution with.</param>
        /// <returns><see langword="true"/> if this distribution is equal to <paramref name="obj"/>, false otherwise.</returns>
        public override bool Equals(object obj)
        {
            if (obj == null || obj.GetType() != typeof(TThis))
            {
                return false;
            }

            TThis that = (TThis)obj;
            EnsureNormalized();
            that.EnsureNormalized();
            return sequenceToWeight.Equals(that.sequenceToWeight);
        }

        /// <summary>
        /// Gets the hash code of this distribution.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            EnsureNormalized();
            return sequenceToWeight.GetHashCode();
        }

        #endregion

        #region Helpers

        #region Helpers for producing improper distributions

        /// <summary>
        /// Creates an improper distribution which assigns a given probability to every sequence.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <param name="uniformLogProb">The logarithm of the probability assigned to every allowed sequence.</param>
        /// <returns>The created distribution.</returns>
        protected static TThis UniformOf(TElementDistribution allowedElements, double uniformLogProb)
        {
            var result = Util.New<TThis>();
            result.SetToUniformOf(allowedElements, uniformLogProb);
            return result;
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns a given probability to every sequence.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <param name="uniformLogProb">The logarithm of the probability assigned to every allowed sequence.</param>
        protected void SetToUniformOf(TElementDistribution allowedElements, double uniformLogProb)
        {
            Argument.CheckIfNotNull(allowedElements, nameof(allowedElements));

            var weightFunction = WeightFunctionFactory.ConstantLog(uniformLogProb, allowedElements);
            SetWeightFunction(weightFunction, false);
        }

        #endregion

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions,
        /// optionally normalizing the result.
        /// </summary>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        /// <param name="computeLogNormalizer">Specifies whether the product normalizer should be computed and used to normalize the product.</param>
        /// <param name="tryDeterminize">Try to determinize the product.</param>
        /// <returns>The logarithm of the normalizer for the product if requested, <see langword="null"/> otherwise.</returns>
        private double? DoSetToProduct(TThis dist1, TThis dist2, bool computeLogNormalizer, bool tryDeterminize = true)
        {
            Debug.Assert(dist1 != null && dist2 != null, "Valid distributions must be provided.");

            if (computeLogNormalizer)
            {
                dist1.EnsureNormalized();
                dist2.EnsureNormalized();    
            }

            var product = dist1.sequenceToWeight.Product(dist2.sequenceToWeight);

            double? logNormalizer = null;
            bool didNormalize = false;
            if (computeLogNormalizer)
            {
                if (product.IsZero())
                {
                    logNormalizer = double.NegativeInfinity; // Normalizer of zero is defined to be -inf, not zero
                }
                else
                {
                    double computedLogNormalizer;

                    // todo: consider moving the following logic into the Automaton class
                    if (!product.TryNormalizeValues(out product, out computedLogNormalizer))
                    {
                        computedLogNormalizer = 0;
                    }

                    if (dist1.UsesAutomatonRepresentation)
                    {
                        var logValueOverride1 = dist1.sequenceToWeight.AsAutomaton().LogValueOverride;
                        if (logValueOverride1 != null)
                        {
                            computedLogNormalizer = logValueOverride1.Value;
                        }
                    }

                    if (dist2.UsesAutomatonRepresentation)
                    {
                        var logValueOverride2 = dist2.sequenceToWeight.AsAutomaton().LogValueOverride;
                        if (logValueOverride2 != null)
                        {
                            computedLogNormalizer = Math.Min(computedLogNormalizer, logValueOverride2.Value);
                        }
                    }

                    logNormalizer = computedLogNormalizer;    
                }

                didNormalize = true;
            }

            SetWeightFunction(product);
            isNormalized = didNormalize;

            return logNormalizer;
        }
        
        /// <summary>
        /// Checks if the distribution uses groups.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution uses groups, <see langword="false"/> otherwise.</returns>
        private bool UsesGroups()
        {
            return this.sequenceToWeight.UsesGroups;
        }

        /// <summary>
        /// Normalizes the underlying weight function (if there is one), if it hasn't been normalized before.
        /// If the distribution is improper, does nothing but marks it as normalized.
        /// </summary>
        private void EnsureNormalized()
        {
            if (!this.isNormalized)
            {
                sequenceToWeight.TryNormalizeValues(out sequenceToWeight, out _);
                this.isNormalized = true;
            }
        }
        
        /// <summary>
        /// Modifies the distribution to be in normalized form e.g. using special
        /// case structures for point masses.
        /// </summary>
        private void NormalizeStructure()
        {
            sequenceToWeight = sequenceToWeight.NormalizeStructure();
        }

        private void TryDeterminizeBeforeSupportEnumeration()
        {
            if (sequenceToWeight.UsesAutomatonRepresentation
                    && sequenceToWeight.AsAutomaton() is StringAutomaton sa
                    && sa.Data.IsEnumerable != false)
            {
                // Determinization of automaton may fail if distribution is not normalized.
                this.EnsureNormalized();
                var determinizedAutomaton = sequenceToWeight.AsAutomaton().TryDeterminize();
                // Sometimes automaton is not determinized, but is still made epsilon-free,
                // which is a nice optimization to keep.
                sequenceToWeight = WeightFunctionFactory.FromAutomaton(determinizedAutomaton);
            }
        }

        #endregion

    }
}