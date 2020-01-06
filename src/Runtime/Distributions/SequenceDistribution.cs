// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices.ComTypes;

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
    /// <typeparam name="TWeightFunction">The type of an underlying function mapping sequences to weights. Currently must be a weighted finite state automaton.</typeparam>
    /// <typeparam name="TThis">The type of a concrete distribution class.</typeparam>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public abstract class SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis> :
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
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
        where TWeightFunction : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>, new()
        where TThis : SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis>, new()
    {
        #region Fields & constants

        /// <summary>
        /// A sequence manipulator.
        /// </summary>
        private static readonly TSequenceManipulator SequenceManipulator = new TSequenceManipulator();

        /// <summary>
        /// A function mapping sequences to weights (non-normalized probabilities).
        /// </summary>
        [DataMember]
        private TWeightFunction sequenceToWeight;

        /// <summary>
        /// Specifies whether the <see cref="sequenceToWeight"/> is normalized.
        /// </summary>
        [DataMember]
        private bool isNormalized;

        /// <summary>
        /// If the distribution is a point mass, stores the point. Otherwise it is set to <see langword="null"/>.
        /// </summary>
        [DataMember]
        private TSequence point;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}"/> class
        /// with a null weight function.  The workspace must be set by the subclass constructor or factory method.
        /// </summary>
        protected SequenceDistribution()
        {
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public TSequence Point
        {
            get
            {
                if (this.point == null)
                {
                    throw new InvalidOperationException("This distribution is not a point mass.");
                }

                return this.point;
            }

            set
            {
                Argument.CheckIfNotNull(value, "value", "Point mass must not be null.");
                
                this.sequenceToWeight = null;
                this.point = value;
                this.isNormalized = true;
            }
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution represents a point mass.
        /// </summary>
        public bool IsPointMass
        {
            get { return this.point != null; }
        }

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

            return new TThis { Point = point };
        }

        /// <summary>
        /// Creates an improper distribution which assigns the probability of 1 to every sequence.
        /// </summary>
        /// <returns>The created uniform distribution.</returns>
        [Construction(UseWhen = "IsUniform")]
        [Skip]
        public static TThis Uniform()
        {
            var result = new TThis();
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
            var result = new TThis();
            result.SetToZero();
            return result;
        }

        /// <summary>
        /// Creates a distribution from a given weight (non-normalized probability) function.
        /// </summary>
        /// <param name="sequenceToWeight">The weight function specifying the distribution.</param>
        /// <returns>The created distribution.</returns>
        [Construction("GetWorkspaceOrPoint")]
        public static TThis FromWeightFunction(TWeightFunction sequenceToWeight)
        {
            Argument.CheckIfNotNull(sequenceToWeight, "sequenceToWeight");

            return FromWorkspace(sequenceToWeight.Clone());
        }

        /// <summary>
        /// Creates a distribution which will use a given weight function as a workspace.
        /// Any modifications to the workspace after the distribution has been created
        /// would put the distribution into an invalid state.
        /// </summary>
        /// <param name="workspace">The workspace to create the distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis FromWorkspace(TWeightFunction workspace)
        {
            Argument.CheckIfNotNull(workspace, "workspace");

            var result = new TThis();
            result.SetWorkspace(workspace);
            return result;
        }

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
            var func = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Builder();
            var end = func.Start.AddTransition(elementDistribution, Weight.One);
            end.SetEndWeight(Weight.One);
            return FromWorkspace(func.GetAutomaton());
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
        /// Creates a distribution over sequences induced by a given list of distributions over sequence elements.
        /// </summary>
        /// <param name="sequence">Enumerable of distributions over sequence elements.</param>
        /// <returns>The created distribution.</returns>
        public static TThis Concatenate(IEnumerable<TElementDistribution> sequence)
        {
            var result = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Builder();
            var last = result.Start;
            foreach (var elem in sequence)
            {
                last = last.AddTransition(elem, Weight.One);
            }
            
            last.SetEndWeight(Weight.One);
            return FromWorkspace(result.GetAutomaton());
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

            var enumerable = distributions as IList<TThis> ?? distributions.ToList();
            if (enumerable.Count == 1)
            {
                if (!cloneIfSizeOne)
                {
                    return enumerable[0];
                }

                return enumerable[0].Clone();
            }

            var probFunctions = enumerable.Select(d => skipNormalization ? d.GetWorkspaceOrPoint() : d.GetNormalizedWorkspaceOrPoint());

            return FromWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Sum(probFunctions));
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
            var result = new TThis();
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
            return FromWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.FromValues(sequenceProbPairs));
        }

        /// <summary>
        /// Creates a distribution which is uniform over a given set of sequences.
        /// </summary>
        /// <param name="sequences">The set of sequences to create a distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(IEnumerable<TSequence> sequences)
        {
            Argument.CheckIfNotNull(sequences, "sequences");

            var enumerable = sequences as IList<TSequence> ?? sequences.ToList();
            return OneOf(enumerable.Select(PointMass), false, true);
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
            return Repeat(Distribution.CreateUniform<TElementDistribution>(), minLength, maxLength);
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
            var elementDistribution = new TElementDistribution { Point = element };
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

            allowedElements = Distribution.CreatePartialUniform(allowedElements);
            double distLogNormalizer = -allowedElements.GetLogAverageOf(allowedElements);
            var weight = uniformity == DistributionKind.UniformOverLengthThenValue
                ? Weight.One
                : Weight.FromLogValue(distLogNormalizer);

            var func = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Builder();
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

            return FromWorkspace(func.GetAutomaton());
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
            Argument.CheckIfNotNull(dist, "dist");
            Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of repetitions must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || maxTimes.Value >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

            if (dist.IsPointMass && maxTimes.HasValue && minTimes == maxTimes)
            {
                var newSequenceElements = new List<TElement>(SequenceManipulator.GetLength(dist.Point) * minTimes);
                for (int i = 0; i < minTimes; ++i)
                {
                    newSequenceElements.AddRange(dist.Point);
                }

                return PointMass(SequenceManipulator.ToSequence(newSequenceElements));
            }

            double logNormalizer = -dist.GetLogAverageOf(dist);
            return FromWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Repeat(
                dist.GetNormalizedWorkspaceOrPoint().ScaleLog(logNormalizer), minTimes, maxTimes));
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
            Argument.CheckIfNotNull(dist, "dist");

            if (this.IsPointMass && dist.IsPointMass && group == 0)
            {
                this.Point = SequenceManipulator.Concat(this.point, dist.point);
                return;
            }

            var workspace = this.GetWorkspaceOrPoint();
            if (dist.IsPointMass)
            {
                workspace.AppendInPlace(dist.Point, group);
            }
            else
            {
                workspace.AppendInPlace(dist.sequenceToWeight, group);
            }

            this.SetWorkspace(workspace);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution induced by a given weight function.
        /// </summary>
        /// <param name="newSequenceToWeight">The function mapping sequences to weights.</param>
        public void SetWeightFunction(TWeightFunction newSequenceToWeight)
        {
            Argument.CheckIfNotNull(newSequenceToWeight, "newSequenceToWeight");

            this.SetWorkspace(newSequenceToWeight.Clone());
        }

        /// <summary>
        /// Replaces the workspace (weight function) of the current distribution with a given one.
        /// </summary>
        /// <param name="workspace">The workspace to replace the current one with.</param>
        /// <param name="normalizeStructure">Whether to normalize the structure of the distribution</param>
        /// <remarks>
        /// If the given workspace represents a point mass, the distribution would be converted to a point mass
        /// and the workspace would be set to <see langword="null"/>.
        /// Any modifications of the workspace will put the distribution into an undefined state.
        /// </remarks>
        public void SetWorkspace(TWeightFunction workspace, bool normalizeStructure=true)
        {
            Argument.CheckIfNotNull(workspace, "workspace");

            this.point = null;
            this.sequenceToWeight = workspace;
            this.isNormalized = false;

            if (normalizeStructure)
            {
                this.NormalizeStructure();
            }
        }

        /// <summary>
        /// Returns the underlying weight function, or, if the distribution is a point mass,
        /// a functional representation of the corresponding point.
        /// Any modifications of the returned function will put the distribution into an undefined state.
        /// </summary>
        /// <remarks>
        /// The returned weight function might differ from the probability function by an arbitrary scale factor.
        /// To ensure that the workspace is normalized, use <see cref="GetNormalizedWorkspaceOrPoint"/>.
        /// </remarks>
        /// <returns>The underlying weight function or a functional representation of the point.</returns>
        public TWeightFunction GetWorkspaceOrPoint()
        {
            return this.IsPointMass
                ? Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.ConstantOn(1.0, this.point)
                : this.sequenceToWeight;
        }

        /// <summary>
        /// Returns the underlying weight function, or, if the distribution is a point mass,
        /// a functional representation of the corresponding point.
        /// Any modifications of the returned function will put the distribution into an undefined state.
        /// </summary>
        /// <remarks>
        /// The returned weight function is guaranteed to be normalized.
        /// If you don't care about an arbitrary scale factor, 
        /// consider using <see cref="GetWorkspaceOrPoint"/> which has less overhead.
        /// </remarks>
        /// <returns>The underlying weight function or a functional representation of the point.</returns>
        public TWeightFunction GetNormalizedWorkspaceOrPoint()
        {
            this.EnsureNormalized();
            return this.GetWorkspaceOrPoint();
        }

        /// <summary>
        /// Enumerates support of this distribution when possible.
        /// Only point mass elements are supported.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <exception cref="AutomatonException">Thrown if enumeration is too large.</exception>
        /// <returns>The strings supporting this distribution</returns>
        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true)
        {
            if (this.IsPointMass)
            {
                return new List<TSequence>(new[] { this.Point });
            }

            this.EnsureNormalized();
            return this.sequenceToWeight.EnumerateSupport(maxCount, tryDeterminize);
        }

        /// <summary>
        /// Enumerates support of this distribution when possible.
        /// Only point mass elements are supported.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="result">The strings supporting this distribution.</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <exception cref="AutomatonException">Thrown if enumeration is too large.</exception>
        /// <returns>True if successful, false otherwise.</returns>
        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true)
        {
            if (this.IsPointMass)
            {
                result = new List<TSequence>(new[] { this.Point });
                return true;
            }

            this.EnsureNormalized();
            return this.sequenceToWeight.TryEnumerateSupport(maxCount, out result, tryDeterminize);
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
            if (this.IsPointMass)
            {
                var singleton = new List<Tuple<List<TElementDistribution>, double>>
                {
                   new Tuple<List<TElementDistribution>, double>(this.Point.Select(el => new TElementDistribution { Point = el }).ToList(),0)
                };

                return singleton;
            }

            return this.sequenceToWeight.EnumeratePaths();
        }

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
        protected string ToString(Action<TElementDistribution, StringBuilder> appendRegex)
        {
            if (this.IsPointMass)
            {
                return this.Point.ToString();
            }
            else
            {
                return this.GetWorkspaceOrPoint().ToString(appendRegex);
            }
        }

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <param name="format">The format.</param>
        /// <returns>A string that represents the distribution.</returns>
        public string ToString(ISequenceDistributionFormat format)
        {
            Argument.CheckIfNotNull(format, "format");
            return format.ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis>((TThis)this);
        }

        #endregion

        #region Groups

        public bool HasGroup(int group)
        {
            if (this.IsPointMass)
            {
                return false; // TODO: get rid of groups or do something about groups + point mass combo
            }
            
            return this.sequenceToWeight.HasGroup(group);
        }
   
        public Dictionary<int, TThis> GetGroups()
        {
            if (this.IsPointMass)
            {
                return new Dictionary<int, TThis>(); // TODO: get rid of groups or do something about groups + point mass combo
            }

            return this.sequenceToWeight.GetGroups().ToDictionary(x => x.Key, x => FromWorkspace(x.Value));
        }

        #endregion

        #region IDistribution implementation

        /// <summary>
        /// Replaces the current distribution by a copy of the given distribution.
        /// </summary>
        /// <param name="that">The distribution to set the current distribution to.</param>
        public void SetTo(TThis that)
        {
            Argument.CheckIfNotNull(that, "that");
            
            if (ReferenceEquals(this, that))
            {
                return;
            }

            this.point = that.point;
            if (this.point == null)
            {
                this.SetWeightFunction(that.sequenceToWeight);
            }
            else
            {
                this.sequenceToWeight = null;
            }

            this.isNormalized = that.isNormalized;
        }

        /// <summary>
        /// Replaces the current distribution by an improper zero distribution.
        /// </summary>
        public void SetToZero()
        {
            this.point = null;
            this.sequenceToWeight = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Zero();
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
            
            var auto = new TThis();
            auto.SetToProduct((TThis)this, that);
            return auto;
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
            Argument.CheckIfNotNull(that, "that");
            
            if (that.IsPointMass)
            {
                return this.GetLogProb(that.point);
            }

            if (this.IsPointMass)
            {
                return that.GetLogProb(this.point);
            }

            var temp = new TThis();
            return temp.SetToProductAndReturnLogNormalizer((TThis)this, that, false);
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

            this.SetWorkspace(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.WeightedSumLog(
                    logWeight1,
                    dist1.GetNormalizedWorkspaceOrPoint(),
                    logWeight2,
                    dist2.GetNormalizedWorkspaceOrPoint()));
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
            var result = new TThis();
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
            
            return Math.Exp(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.GetLogSimilarity(
                this.GetNormalizedWorkspaceOrPoint(), thatDistribution.GetNormalizedWorkspaceOrPoint()));
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
            
            if (this.IsPointMass)
            {
                if (SequenceManipulator.SequencesAreEqual(sequence, this.point))
                {
                    return 0;
                }

                return double.NegativeInfinity;
            }

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

            if (this.IsPointMass)
            {
                return SequenceManipulator.SequencesAreEqual(sequence, this.point);
            }

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
                var weightFunction = this.GetWorkspaceOrPoint();
                var theConverger =
                    Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.GetConverger(weightFunction);
                var properWeightFunction = weightFunction.Product(theConverger);
                properDist = this.Clone();
                properDist.SetWeightFunction(properWeightFunction);
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
            var currentState = dist.sequenceToWeight.Start;
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
                            sampledElements.Add(transition.ElementDistribution.Value.Sample());
                        }

                        currentState = dist.sequenceToWeight.States[transition.DestinationStateIndex];
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
        /// Replaces the current distribution with an improper distribution which assigns the probability of 1 to every sequence.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        public void SetToUniformOf(TElementDistribution allowedElements)
        {
            this.SetToUniformOf(allowedElements, 0.0);
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns the probability of 1 to every sequence.
        /// </summary>
        public void SetToUniform()
        {
            this.SetToUniformOf(Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        /// <remarks>This function will fail if the workspace of this distribution cannot be determinized.</remarks>
        public void SetToPartialUniform()
        {
            this.SetToPartialUniformOf((TThis)this);
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="dist">The distribution which support will be used to setup the current distribution.</param>
        /// <remarks>This function will fail if the workspace of <paramref name="dist"/> cannot be determinized.</remarks>
        public void SetToPartialUniformOf(TThis dist)
        {
            if (dist.IsPointMass)
            {
                this.Point = dist.Point;
                return;
            }
            
            var resultWorkspace = new TWeightFunction();
            resultWorkspace.SetToConstantOnSupportOfLog(0.0, dist.GetWorkspaceOrPoint());
            this.SetWorkspace(resultWorkspace, false);
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution is uniform over its support, <see langword="false"/> otherwise.</returns>
        public bool IsPartialUniform()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Checks whether the current distribution is proper.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is proper, <see langword="false"/> otherwise.</returns>
        public bool IsProper()
        {
            if (this.IsPointMass)
            {
                return true;
            }

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
            return this.GetWorkspaceOrPoint().IsCanonicConstant();
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
            return !this.IsPointMass && this.sequenceToWeight.IsZero();
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
            if (this.IsPointMass)
            {
                if (!that.IsPointMass)
                {
                    return false;
                }

                // todo: use sequence manipulator equality
                return Equals(this.Point, that.Point);
            }
            else
            {
                if (that.IsPointMass)
                {
                    return false;
                }
            }

            return this.GetNormalizedWorkspaceOrPoint().Equals(that.GetNormalizedWorkspaceOrPoint());
        }

        /// <summary>
        /// Gets the hash code of this distribution.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            return this.GetNormalizedWorkspaceOrPoint().GetHashCode();
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
            var result = new TThis();
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
            Argument.CheckIfNotNull(allowedElements, "allowedElements");

            this.SetWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.ConstantLog(uniformLogProb, allowedElements), false);
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
            
            if (dist1.IsPointMass)
            {
                var pt = dist1.Point;
                bool containsPoint = false;
                double? logNorm = null;
                if (computeLogNormalizer)
                {
                    logNorm = dist2.GetLogProb(pt);
                    containsPoint = !double.IsNegativeInfinity(logNorm.Value);
                }
                else
                {
                    containsPoint = dist2.Contains(pt);
                }
                
                if (containsPoint)
                {
                    this.SetTo(dist1);
                }
                else
                {
                    this.SetToZero();
                }

                return logNorm;
            }

            if (dist2.IsPointMass)
            {
                var pt = dist2.Point;
                bool containsPoint = false;
                double? logNorm = null;

                if (computeLogNormalizer)
                {
                    logNorm = dist1.GetLogProb(pt);
                    containsPoint = !double.IsNegativeInfinity(logNorm.Value);
                } else
                {
                    containsPoint = dist1.Contains(pt);
                }
                
                if (!containsPoint)
                {
                    this.SetToZero();
                    return logNorm;
                }

                if (!dist1.UsesGroups())
                {
                    this.Point = pt;
                    return logNorm;
                }
            }

            if (computeLogNormalizer)
            {
                dist1.EnsureNormalized();
                dist2.EnsureNormalized();    
            }
            
            var weightFunction1 = dist1.GetWorkspaceOrPoint();
            var weightFunction2 = dist2.GetWorkspaceOrPoint();
            var product = weightFunction1.Product(weightFunction2, tryDeterminize);
            this.SetWorkspace(product);

            double? logNormalizer = null;
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
                    if (!product.TryNormalizeValues(out computedLogNormalizer))
                    {
                        computedLogNormalizer = 0;
                    }

                    if (weightFunction1.LogValueOverride != null)
                    {
                        computedLogNormalizer = weightFunction1.LogValueOverride.Value;
                    }

                    if (weightFunction2.LogValueOverride != null)
                    {
                        computedLogNormalizer = Math.Min(computedLogNormalizer, weightFunction2.LogValueOverride.Value);
                    }

                    logNormalizer = computedLogNormalizer;    
                }
                
                this.isNormalized = true;
            }
            
            return logNormalizer;
        }
        
        /// <summary>
        /// Checks if the distribution uses groups.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution uses groups, <see langword="false"/> otherwise.</returns>
        private bool UsesGroups()
        {
            if (this.IsPointMass)
            {
                return false;
            }

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
                this.sequenceToWeight.TryNormalizeValues();
                this.isNormalized = true;
            }
        }
        
        /// <summary>
        /// Modifies the distribution to be in normalized form e.g. using special
        /// case structures for point masses.
        /// </summary>
        private void NormalizeStructure()
        {
            if (this.UsesGroups())
            {
                return; // todo: remove groups
            }

            var pt = this.TryComputePoint();
            if (pt != null)
            {
                this.Point = pt;
            }
        }

        /// <summary>
        /// Returns a point mass represented by the current distribution, or <see langword="null"/>,
        /// if it doesn't represent a point mass.
        /// </summary>
        /// <returns>The point mass represented by the distribution, or <see langword="null"/>.</returns>
        private TSequence TryComputePoint()
        {
            if (this.IsPointMass)
            {
                return this.point;
            }

            return this.sequenceToWeight.TryComputePoint();
        }

        #endregion    

        /// <summary>
        /// Whether sequenceToWeight needs to be serialized.
        /// </summary>
        public bool ShouldSerializesequenceToWeight() => point == null;

    }
}