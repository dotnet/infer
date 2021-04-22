// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;

    /// <summary>
    /// An implementation of <see cref="IWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}"/>
    /// that automatically chooses between <see cref="PointMassWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}"/>,
    /// <see cref="DictionaryWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}"/>, and
    /// <see cref="Automaton{TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis}"/> representation.
    /// </summary>
    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The type of a sequence element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
    /// <typeparam name="TPointMass">The type used for representing point mass weight functions.</typeparam>
    /// <typeparam name="TDictionary">The type used for weight functions represented as dictionaries.</typeparam>
    /// <typeparam name="TAutomaton">The type of a weighted finite state automaton, that can be used to
    /// represent all weight functions.</typeparam>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public struct MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> :
        IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TPointMass : PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TPointMass>, new()
        where TDictionary : DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
    {
        private static TSequenceManipulator SequenceManipulator =>
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.SequenceManipulator;

        private static readonly TAutomaton ZeroAutomaton =
            Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Zero();

        private const int MaxDictionarySize = 128;

        /// <summary>
        /// A function mapping sequences to weights.
        /// Can only be of one of the following types: TPointMass, TDictionary, TAutomaton, or <see langword="null"/>.
        /// <see langword="null"/> should be interpreted as zero function.
        /// </summary>
        [DataMember]
        private IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> weightFunction;

        /// <summary>
        /// An <see cref="IWeightFunctionFactory{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction}"/>
        /// implementation for <see cref="MultiRepresentationWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton}"/>.
        /// </summary>
        public class Factory : IWeightFunctionFactory<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>>
        {
            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> ConstantLog(double logValue, TElementDistribution allowedElements)
            {
                if (allowedElements is CanEnumerateSupport<TElement> supportEnumerator)
                {
                    var possiblyTruncatedSupport = supportEnumerator.EnumerateSupport().Take(MaxDictionarySize + 1).ToList();
                    if (possiblyTruncatedSupport.Count <= MaxDictionarySize)
                    {
                        var weight = Weight.FromLogValue(logValue);
                        return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(
                            possiblyTruncatedSupport.Select(elem => new KeyValuePair<TSequence, Weight>(SequenceManipulator.ToSequence(new[] { elem }), weight))));
                    }
                }
                var automaton = new TAutomaton();
                automaton.SetToConstantLog(logValue, allowedElements);
                return FromAutomaton(automaton);
            }

            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> ConstantOnSupportOfLog(double logValue, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
            {
                if (weightFunction.TryEnumerateSupport(MaxDictionarySize, out var support, false))
                {
                    var weight = Weight.FromLogValue(logValue);
                    return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(
                        support.Select(sequence => new KeyValuePair<TSequence, Weight>(sequence, weight))));
                }
                var automaton = new TAutomaton();
                automaton.SetToConstantOnSupportOfLog(logValue, weightFunction.AsAutomaton());
                return FromAutomaton(automaton);
            }

            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromAutomaton(TAutomaton automaton) =>
                MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>.FromAutomaton(automaton);

            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs)
            {
                var collection = sequenceWeightPairs as ICollection<KeyValuePair<TSequence, double>> ?? sequenceWeightPairs.ToList();
                if (collection.Count == 1 && collection.Single().Value == 1.0)
                {
                    return FromPoint(collection.Single().Key);
                }
                else
                {
                    if (collection.Count <= MaxDictionarySize)
                    {
                        return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromValues(sequenceWeightPairs));
                    }
                    else
                        return FromAutomaton(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.FromValues(collection));
                }
            }

            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> PointMass(TSequence point) =>
                FromPoint(point);

            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Sum(IEnumerable<MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>> weightFunctions)
            {
                var dictionary = new Dictionary<TSequence, Weight>(MaxDictionarySize, SequenceManipulator.SequenceEqualityComparer);
                bool resultFitsDictionary = true;
                foreach (var weightFunction in weightFunctions)
                {
                    if (weightFunction.IsCanonicZero())
                        continue;
                    if (weightFunction.weightFunction is TPointMass pointMass)
                    {
                        if (dictionary.TryGetValue(pointMass.Point, out Weight oldWeight))
                            dictionary[pointMass.Point] = oldWeight + Weight.One;
                        else if (dictionary.Count < MaxDictionarySize)
                            dictionary.Add(pointMass.Point, Weight.One);
                        else
                        {
                            resultFitsDictionary = false;
                            break;
                        }
                    }
                    else if (weightFunction.weightFunction is TDictionary wfDictionary)
                    {
                        foreach (var kvp in wfDictionary.Dictionary)
                        {
                            if (dictionary.TryGetValue(kvp.Key, out Weight oldWeight))
                                dictionary[kvp.Key] = oldWeight + kvp.Value;
                            else if (dictionary.Count < MaxDictionarySize)
                                dictionary.Add(kvp.Key, kvp.Value);
                            else
                            {
                                resultFitsDictionary = false;
                                break;
                            }
                        }
                        if (!resultFitsDictionary)
                            break;
                    }
                    else
                    {
                        resultFitsDictionary = false;
                        break;
                    }
                }

                if (resultFitsDictionary)
                {
                    if (dictionary.Count == 0)
                        return MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>.Zero();
                    if (dictionary.Count == 1)
                    {
                        var singleKvp = dictionary.Single();
                        if (singleKvp.Value.LogValue == 0.0)
                            return FromPoint(singleKvp.Key);
                    }
                    return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(dictionary));
                }

                var automaton = new TAutomaton();
                automaton.SetToSum(weightFunctions.Select(wf => wf.AsAutomaton()));
                return FromAutomaton(automaton);
            }

            public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Zero() => MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>.Zero();
        }

        #region Factory Methods

        /// <summary>
        /// Creates a weight function which is zero everywhere.
        /// </summary>
        /// <returns>The created weight function</returns>
        [Construction(UseWhen = nameof(IsCanonicZero))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Zero() =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>();

        /// <summary>
        /// Creates a <see cref="MultiRepresentationWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton}"/>
        /// represented as <see cref="PointMassWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}"/>.
        /// </summary>
        /// <param name="pointMass">The point mass weight function.</param>
        /// <returns>The created weight function.</returns>
        [Construction(nameof(AsPointMass), UseWhen = nameof(IsPointMass))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromPointMass(TPointMass pointMass) =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>() { weightFunction = pointMass };

        /// <summary>
        /// Creates a <see cref="MultiRepresentationWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton}"/>
        /// represented as <see cref="DictionaryWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis}"/>.
        /// </summary>
        /// <param name="dictionary">The dictionary weight function.</param>
        /// <returns>The created weight function.</returns>
        [Construction(nameof(AsDictionary), UseWhen = nameof(IsDictionary))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromDictionary(TDictionary dictionary) =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>() { weightFunction = dictionary };

        /// <summary>
        /// Creates a <see cref="MultiRepresentationWeightFunction{TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton}"/>
        /// represented as <see cref="Automaton{TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis}"/>.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <returns>The created weight function.</returns>
        [Construction(nameof(AsAutomaton), UseWhen = nameof(IsAutomaton))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromAutomaton(TAutomaton automaton) =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>() { weightFunction = automaton };

        /// <summary>
        /// Creates a point mass weight function.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns>The created point mass weight function.</returns>
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromPoint(TSequence point) =>
            FromPointMass(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TPointMass>.FromPoint(point));

        #endregion

        public TSequence Point
        {
            get
            {
                if (weightFunction is TPointMass pointMassWeightFunction)
                {
                    return pointMassWeightFunction.Point;
                }

                throw new InvalidOperationException("This distribution is not a point mass.");
            }
        }

        public bool IsPointMass => weightFunction is TPointMass;

        /// <summary>
        /// Gets a value indicating whether the weight function uses a dictionary internal representation.
        /// </summary>
        public bool IsDictionary => weightFunction is TDictionary;

        /// <summary>
        /// Gets a value indicating whether the weight function uses an automaton internal representation.
        /// </summary>
        public bool IsAutomaton => weightFunction is TAutomaton;

        /// <summary>
        /// Returns a point mass representation of the current weight function, if it is being used,
        /// or <see langword="null"/> otherwise.
        /// </summary>
        public TPointMass AsPointMass() => weightFunction as TPointMass;

        /// <summary>
        /// Returns a dictionary representation of the current weight function, if it is being used,
        /// or <see langword="null"/> otherwise.
        /// </summary>
        public TDictionary AsDictionary() => weightFunction as TDictionary;

        public TAutomaton AsAutomaton() => weightFunction?.AsAutomaton() ?? new TAutomaton();

        public bool UsesAutomatonRepresentation => weightFunction is TAutomaton;

        public bool UsesGroups => weightFunction.UsesGroups;

        public bool HasGroup(int group) => weightFunction.HasGroup(group);

        public Dictionary<int, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>> GetGroups()
        {
            return weightFunction is TAutomaton automaton
                ? automaton.GetGroups().ToDictionary(
                    kvp => kvp.Key,
                    kvp => new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>()
                    {
                        weightFunction = kvp.Value
                    })
                : new Dictionary<int, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>>(); // TODO: get rid of groups or do something about groups + point mass combo
        }

        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true) => weightFunction?.EnumerateSupport(maxCount, tryDeterminize) ?? Enumerable.Empty<TSequence>();

        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true)
        {
            if (weightFunction == null)
            {
                result = Enumerable.Empty<TSequence>();
                return true;
            }
            else
                return weightFunction.TryEnumerateSupport(maxCount, out result, tryDeterminize);
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> NormalizeStructure()
        {
            switch (weightFunction)
            {
                case TDictionary dictionary:
                    var filteredTruncated = dictionary.Dictionary.Where(kvp => !kvp.Value.IsZero).Take(2).ToList();
                    if (filteredTruncated.Count == 1)
                    {
                        return FromPoint(filteredTruncated.Single().Key);
                    }
                    else
                    {
                        return FromDictionary(dictionary.NormalizeStructure());
                    }
                case TAutomaton automaton:
                    if (!automaton.UsesGroups)
                    {
                        if (automaton.LogValueOverride == null && automaton.TryEnumerateSupport(MaxDictionarySize, out var support, false, 4 * MaxDictionarySize, true))
                        {
                            var list = support.ToList();
                            if (list.Count == 1)
                                return FromPoint(list[0]);
                            else
                            {
                                // Create a dictionary only if we expect it to be smaller than the automaton.
                                // Approximation uses sizes corresponding to a string automaton, which is the most used one.
                                // We don't require this comparison to be always precise - most of the times is good enough.
                                var dictSizeApprox = list.Sum(el => SequenceManipulator.GetLength(el)) * sizeof(char) + (24 + 8 + sizeof(double)) * list.Count;
                                var automatonSizeAprox =
                                    24 // header
                                    + 16 + 2 * sizeof(double) // 2 double? fields
                                                              // Data Container
                                    + 2 * sizeof(int) // Flags and StartStateIndex
                                    + 2 * 24 // Headers of the states and transitions arrays
                                    + automaton.Data.States.Count * (2 * sizeof(int) + sizeof(double)) // states
                                    + automaton.Data.Transitions.Count * 24 // 24 is the size of one transition w/o storage for discrete char
                                    + automaton.Data.Transitions.Count(tr => !tr.IsEpsilon) * 80;
                                    // 40 is the size of a DiscreteChar filled with nulls;
                                    // another 40 is the size of an array with a single char range.
                                    // Any specific DiscreteChar can be larger or can be cached.
                                    // 40 seems an ok approximation for the average case.
                                if (dictSizeApprox < automatonSizeAprox)
                                    return FromDictionary(
                                        DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(
                                            list.Select(seq => new KeyValuePair<TSequence, Weight>(seq, Weight.FromLogValue(automaton.GetLogValue(seq))))));
                            }
                        }
                        // TryEnumerateSupport(..., maxTraversedPaths, ...) is allowed to quit early
                        // on complex automata, so we need to explicitly check for point mass
                        var point = automaton.TryComputePoint();
                        if (point != null)
                        {
                            return FromPoint(point);
                        }
                    }
                    break;
            }
            
            return Clone(); // TODO: replace with `this` after making automata immutable
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Repeat(int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfInRange(minTimes >= 0, nameof(minTimes), "The minimum number of repetitions must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || maxTimes.Value >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

            if (weightFunction == null)
                return Zero();
            if (weightFunction is TPointMass pointMass && maxTimes.HasValue && maxTimes - minTimes < MaxDictionarySize)
            {
                var newSequenceElements = new List<TElement>(SequenceManipulator.GetLength(pointMass.Point) * maxTimes.Value);
                for (int i = 0; i < minTimes; ++i)
                {
                    newSequenceElements.AddRange(pointMass.Point);
                }
                if (minTimes == maxTimes)
                {
                    return FromPoint(SequenceManipulator.ToSequence(newSequenceElements));
                }
                else
                {
                    Weight uniformWeight = Weight.FromValue(1.0 / (maxTimes.Value - minTimes));
                    Dictionary<TSequence, Weight> dict = new Dictionary<TSequence, Weight>(maxTimes.Value - minTimes + 1);
                    dict.Add(SequenceManipulator.ToSequence(newSequenceElements), uniformWeight);
                    for (int i = minTimes + 1; i <= maxTimes.Value; ++i)
                    {
                        newSequenceElements.AddRange(pointMass.Point);
                        dict.Add(SequenceManipulator.ToSequence(newSequenceElements), uniformWeight);
                    }
                    return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(dict));
                }
            }
            if (weightFunction is TDictionary dictionary && maxTimes.HasValue)
            {
                var resultSupportSize = ResultSupportSize(dictionary.Dictionary.Count, minTimes, maxTimes.Value);
                if (resultSupportSize <= MaxDictionarySize)
                    return FromDictionary(dictionary.Repeat(minTimes, maxTimes.Value, (int)resultSupportSize + 1));
            }

            return FromAutomaton(AsAutomaton().Repeat(minTimes, maxTimes));

            double ResultSupportSize(int sourceSupportSize, int minReps, int maxReps)
            {
                return Math.Pow(sourceSupportSize, minReps) * (1 - Math.Pow(sourceSupportSize, maxReps - minReps + 1)) / (1 - sourceSupportSize);
            }
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> ScaleLog(double logScale)
        {
            switch (weightFunction)
            {
                case null:
                    return Zero();
                case TPointMass pointMass:
                    return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(
                        new[] { new KeyValuePair<TSequence, Weight>(pointMass.Point, Weight.FromLogValue(logScale)) }));
                case TDictionary dictionary:
                    return FromDictionary(dictionary.ScaleLog(logScale));
                case TAutomaton automaton:
                    return FromAutomaton(automaton.ScaleLog(logScale));
                default:
                    throw new InvalidOperationException("Current function has an invalid type");
            }
        }

        public bool TryNormalizeValues(out MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> normalizedFunction, out double logNormalizer)
        {
            bool result;
            switch (weightFunction)
            {
                case null:
                    normalizedFunction = new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>();
                    logNormalizer = double.NegativeInfinity;
                    result = false;
                    break;
                case TPointMass pointMass:
                    result = pointMass.TryNormalizeValues(out var normalizedPointMass, out logNormalizer);
                    normalizedFunction = FromPointMass(normalizedPointMass);
                    break;
                case TDictionary dictionary:
                    result = dictionary.TryNormalizeValues(out var normalizedDictionary, out logNormalizer);
                    normalizedFunction = FromDictionary(normalizedDictionary);
                    break;
                case TAutomaton automaton:
                    result = automaton.TryNormalizeValues(out var normalizedAutomaton, out logNormalizer);
                    normalizedFunction = FromAutomaton(normalizedAutomaton);
                    break;
                default:
                    throw new InvalidOperationException("Current function has an invalid type");
            }
            return result;
        }

        public double GetLogValue(TSequence sequence) => weightFunction?.GetLogValue(sequence) ?? double.NegativeInfinity;

        public bool IsZero() => weightFunction?.IsZero() ?? true;

        /// <summary>
        /// Checks whether the weight function is a canonic representation of zero,
        /// as produced by <see cref="Zero"/>.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the weight function is a canonic representation of zero,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>
        /// The time complexity of this function is O(1), so it can be used to treat zero specially in performance-critical code.
        /// All the operations on automata resulting in zero produce the canonic representation.
        /// </remarks>
        public bool IsCanonicZero() => weightFunction == null;

        public double MaxDiff(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> that)
        {
            if (IsCanonicZero())
                return that.IsZero() ? 0.0 : Math.E;
            if (that.IsCanonicZero())
                return IsZero() ? 0.0 : Math.E;

            switch (weightFunction)
            {
                case TAutomaton automaton:
                    return automaton.MaxDiff(that.AsAutomaton());
                case TDictionary dictionary:
                    switch (that.weightFunction)
                    {
                        case TAutomaton otherAutomaton:
                            return AsAutomaton().MaxDiff(otherAutomaton);
                        case TDictionary otherDictionary:
                            return dictionary.MaxDiff(otherDictionary);
                        case TPointMass otherPointMass:
                            return dictionary.MaxDiff(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(otherPointMass.Point));
                        default:
                            throw new InvalidOperationException("Other function has an invalid type");
                    }
                case TPointMass pointMass:
                    switch (that.weightFunction)
                    {
                        case TAutomaton otherAutomaton:
                            return AsAutomaton().MaxDiff(otherAutomaton);
                        case TDictionary otherDictionary:
                            return otherDictionary.MaxDiff(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(pointMass.Point));
                        case TPointMass otherPointMass:
                            return pointMass.MaxDiff(otherPointMass);
                        default:
                            throw new InvalidOperationException("Other function has an invalid type");
                    }
                default:
                    throw new InvalidOperationException("Current function has an invalid type");
            }
        }

        public double GetLogNormalizer() => weightFunction?.GetLogNormalizer() ?? double.NegativeInfinity;

        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths() => weightFunction?.EnumeratePaths() ?? Enumerable.Empty<Tuple<List<TElementDistribution>, double>>();

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Append(TSequence sequence, int group = 0)
        {
            switch (weightFunction)
            {
                case null:
                    return Zero();
                case TPointMass pointMass:
                    return group == 0 ? FromPointMass(pointMass.Append(sequence)) : FromAutomaton(pointMass.AsAutomaton().Append(sequence, group));
                case TDictionary dictionary:
                    return group == 0 ? FromDictionary(dictionary.Append(sequence)) : FromAutomaton(dictionary.AsAutomaton().Append(sequence, group));
                case TAutomaton automaton:
                    return FromAutomaton(automaton.Append(sequence, group));
                default:
                    throw new InvalidOperationException("Current function has an invalid type");
            }
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Append(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction, int group = 0)
        {
            if (this.weightFunction == null || weightFunction.weightFunction == null)
                return Zero();

            if (group == 0)
            {
                if (weightFunction.weightFunction is TPointMass otherPointMass)
                {
                    if (this.weightFunction is TPointMass thisPointMass)
                        return FromPointMass(thisPointMass.Append(otherPointMass.Point));
                    if (this.weightFunction is TDictionary thisDictionary)
                        return FromDictionary(thisDictionary.Append(otherPointMass.Point));
                }

                if (weightFunction.weightFunction is TDictionary otherDictionary)
                {
                    if (this.weightFunction is TPointMass thisPointMass)
                        return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(thisPointMass.Point).Append(otherDictionary));
                    if (this.weightFunction is TDictionary thisDictionary && thisDictionary.Dictionary.Count * otherDictionary.Dictionary.Count <= MaxDictionarySize)
                        return FromDictionary(thisDictionary.Append(otherDictionary));
                }
            }

            if (weightFunction.weightFunction is TPointMass pointMass)
                return FromAutomaton(this.weightFunction.AsAutomaton().Append(pointMass.Point, group));

            return FromAutomaton(this.weightFunction.AsAutomaton().Append(weightFunction.weightFunction.AsAutomaton(), group));
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Sum(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            if (weightFunction.IsCanonicZero())
                return Clone(); // TODO: return `this` when automata become immutable
            if (IsCanonicZero())
                return weightFunction.Clone(); // TODO: return weightFunction when automata become immutable

            if (weightFunction.weightFunction is TAutomaton otherAutomaton)
                return FromAutomaton(AsAutomaton().Sum(otherAutomaton));
            if (this.weightFunction is TAutomaton thisAutomaton)
                return FromAutomaton(thisAutomaton.Sum(weightFunction.AsAutomaton()));

            // Now both weight functions are either point masses or dictionaries
            var thisDictionary = this.weightFunction as TDictionary ?? DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(((TPointMass)this.weightFunction).Point);
            var otherDictionary = weightFunction.weightFunction as TDictionary ?? DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(((TPointMass)weightFunction.weightFunction).Point);

            var resultDictionary = thisDictionary.Sum(otherDictionary);

            if (resultDictionary.Dictionary.Count <= MaxDictionarySize)
                return FromDictionary(resultDictionary);
            else
                return FromAutomaton(resultDictionary.AsAutomaton());
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Sum(double weight1, double weight2, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            Argument.CheckIfInRange(weight1 >= 0, nameof(weight1), "Negative weights are not supported.");
            Argument.CheckIfInRange(weight2 >= 0, nameof(weight2), "Negative weights are not supported.");

            return SumLog(Math.Log(weight1), Math.Log(weight2), weightFunction);
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> SumLog(double logWeight1, double logWeight2, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            if (weightFunction.IsCanonicZero() || double.IsNegativeInfinity(logWeight2))
                return ScaleLog(logWeight1);
            if (IsCanonicZero() || double.IsNegativeInfinity(logWeight1))
                return weightFunction.ScaleLog(logWeight2);

            if (weightFunction.weightFunction is TAutomaton otherAutomaton)
                return FromAutomaton(AsAutomaton().SumLog(logWeight1, logWeight2, otherAutomaton));
            if (this.weightFunction is TAutomaton thisAutomaton)
                return FromAutomaton(thisAutomaton.SumLog(logWeight1, logWeight2, weightFunction.AsAutomaton()));

            // Now both weight functions are either point masses or dictionaries
            var thisDictionary = this.weightFunction as TDictionary ?? DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(((TPointMass)this.weightFunction).Point);
            var otherDictionary = weightFunction.weightFunction as TDictionary ?? DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromPoint(((TPointMass)weightFunction.weightFunction).Point);

            var resultDictionary = thisDictionary.SumLog(logWeight1, logWeight2, otherDictionary);

            if (resultDictionary.Dictionary.Count <= MaxDictionarySize)
                return FromDictionary(resultDictionary);
            else
                return FromAutomaton(resultDictionary.AsAutomaton());
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Product(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            if (IsCanonicZero() || weightFunction.IsCanonicZero())
                return Zero();

            TPointMass pointMass = null;
            IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> other = null;
            if (this.weightFunction is TPointMass thisPointMass)
            {
                pointMass = thisPointMass;
                other = weightFunction.weightFunction;
            }
            else if (weightFunction.weightFunction is TPointMass otherPointMass)
            {
                pointMass = otherPointMass;
                other = this.weightFunction;
            }
            if (pointMass != null && !other.UsesGroups)
            {
                var logValue = other.GetLogValue(pointMass.Point);
                if (double.IsNegativeInfinity(logValue))
                    return Zero();
                else if (logValue == 0.0)
                    return FromPointMass(pointMass);
                else
                    return FromDictionary(
                        DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(
                            new[] { new KeyValuePair<TSequence, Weight>(pointMass.Point, Weight.FromLogValue(logValue)) }));
            }

            TDictionary dictionary = null;
            if (this.weightFunction is TDictionary thisDictionary)
            {
                if (weightFunction.weightFunction is TDictionary secondDictionary)
                    return FromDictionary(thisDictionary.Product(secondDictionary));
                dictionary = thisDictionary;
                other = weightFunction.weightFunction;
            }
            else if (weightFunction.weightFunction is TDictionary otherDictionary)
            {
                dictionary = otherDictionary;
                other = this.weightFunction;
            }

            if (dictionary != null && !other.UsesGroups)
            {
                var resultList = new List<KeyValuePair<TSequence, Weight>>(dictionary.Dictionary.Count);
                foreach (var kvp in dictionary.Dictionary)
                {
                    if (!kvp.Value.IsZero)
                    {
                        var otherLogValue = other.GetLogValue(kvp.Key);
                        if (!double.IsNegativeInfinity(otherLogValue))
                            resultList.Add(new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * Weight.FromLogValue(otherLogValue)));
                    }
                }
                if (resultList.Count == 0)
                    return Zero();
                else if (resultList.Count == 1 && resultList[0].Value.LogValue == 0.0)
                    return FromPoint(resultList[0].Key);
                else
                    return FromDictionary(
                        DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(resultList));
            }

            return FromAutomaton(AsAutomaton().Product(weightFunction.AsAutomaton()));
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Clone()
        {
            // TODO: remove when automata become immutable
            if (weightFunction is TAutomaton automaton)
                return FromAutomaton(automaton.Clone());

            return this;
        }

        public bool Equals(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> other)
        {
            if (IsCanonicZero())
                return other.IsZero();
            if (other.IsCanonicZero())
                return IsZero();

            switch (weightFunction)
            {
                case TAutomaton automaton:
                    return automaton.Equals(other.AsAutomaton());
                case TDictionary dictionary:
                    switch (other.weightFunction)
                    {
                        case TAutomaton otherAutomaton:
                            return AsAutomaton().Equals(otherAutomaton);
                        case TDictionary otherDictionary:
                            return dictionary.Equals(otherDictionary);
                        case TPointMass otherPointMass:
                            if (dictionary.Dictionary.Count != 1)
                                return false;
                            var singleKvp = dictionary.Dictionary.Single();
                            return SequenceManipulator.SequenceEqualityComparer.Equals(singleKvp.Key, otherPointMass.Point) && singleKvp.Value.LogValue == 0.0;
                        default:
                            throw new InvalidOperationException("Other function has an invalid type");
                    }
                case TPointMass pointMass:
                    switch (other.weightFunction)
                    {
                        case TAutomaton otherAutomaton:
                            return AsAutomaton().Equals(otherAutomaton);
                        case TDictionary otherDictionary:
                            if (otherDictionary.Dictionary.Count != 1)
                                return false;
                            var singleKvp = otherDictionary.Dictionary.Single();
                            return SequenceManipulator.SequenceEqualityComparer.Equals(pointMass.Point, singleKvp.Key) && singleKvp.Value.LogValue == 0.0;
                        case TPointMass otherPointMass:
                            return pointMass.Equals(otherPointMass);
                        default:
                            throw new InvalidOperationException("Other function has an invalid type");
                    }
                default:
                    throw new InvalidOperationException("Current function has an invalid type");
            }
        }

        public override bool Equals(object obj)
        {
            if (obj == null || typeof(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>) != obj.GetType())
            {
                return false;
            }

            return Equals((MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>)obj);
        }

        public override int GetHashCode() => (weightFunction ?? ZeroAutomaton).GetHashCode();

        public override string ToString() => (weightFunction ?? ZeroAutomaton).ToString();

        public string ToString(Action<TElementDistribution, StringBuilder> appendElement) => (weightFunction ?? ZeroAutomaton).ToString(appendElement);
    }
}
