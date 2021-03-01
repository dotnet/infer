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

        private const int MaxDictionarySize = 20;

        /// <summary>
        /// A function mapping sequences to weights.
        /// Can only be of one of the following types: TPointMass, TDictionary, TAutomaton, or <see langword="null"/>.
        /// <see langword="null"/> should be interpreted as zero function.
        /// </summary>
        [DataMember]
        private IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> weightFunction;

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
                    if (weightFunction.IsCanonicalZero())
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

        [Construction(UseWhen = nameof(IsCanonicalZero))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Zero() =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>();

        [Construction(nameof(AsPointMass), UseWhen = nameof(IsPointMass))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromPointMass(TPointMass pointMass) =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>() { weightFunction = pointMass };

        [Construction(nameof(AsDictionary), UseWhen = nameof(IsDictionary))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromDictionary(TDictionary dictionary) =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>() { weightFunction = dictionary };

        [Construction(nameof(AsAutomaton), UseWhen = nameof(IsAutomaton))]
        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromAutomaton(TAutomaton automaton) =>
            new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>() { weightFunction = automaton };

        public static MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> FromPoint(TSequence point) =>
            FromPointMass(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TPointMass>.FromPoint(point));

        #endregion

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
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

        /// <summary>
        /// Gets a value indicating whether the weight function uses a point mass internal representation.
        /// </summary>
        public bool IsPointMass => weightFunction is TPointMass;

        /// <summary>
        /// Gets a value indicating whether the weight function uses a dictionary internal representation.
        /// </summary>
        public bool IsDictionary => weightFunction is TDictionary;

        /// <summary>
        /// Gets a value indicating whether the weight function uses an automaton internal representation.
        /// </summary>
        public bool IsAutomaton => weightFunction is TAutomaton;

        public TPointMass AsPointMass() => weightFunction as TPointMass;

        public TDictionary AsDictionary() => weightFunction as TDictionary;

        public TAutomaton AsAutomaton() => weightFunction?.AsAutomaton() ?? new TAutomaton();

        public bool UsesAutomatonRepresentation => weightFunction is TAutomaton;

        /// <summary>
        /// Checks if the weight function uses groups.
        /// </summary>
        /// <returns><see langword="true"/> if the weight function uses groups, <see langword="false"/> otherwise.</returns>
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

        /// <summary>
        /// Enumerates support of this distribution when possible.
        /// Only point mass elements are supported.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <exception cref="AutomatonException">Thrown if enumeration is too large.</exception>
        /// <returns>The strings supporting this distribution</returns>
        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true) => weightFunction?.EnumerateSupport(maxCount, tryDeterminize) ?? Enumerable.Empty<TSequence>();

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
            if (weightFunction == null)
            {
                result = Enumerable.Empty<TSequence>();
                return true;
            }
            else
                return weightFunction.TryEnumerateSupport(maxCount, out result, tryDeterminize);
        }

        /// <summary>
        /// Returns the weight function converted to the normalized form e.g. using special
        /// case structures for point masses and functions with small support.
        /// </summary>
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
                    if (!automaton.UsesGroups && automaton.TryEnumerateSupport(MaxDictionarySize, out var support, false))
                    {
                        // TODO: compute values along with support
                        var list = support.Select(seq => new KeyValuePair<TSequence, Weight>(seq, Weight.FromLogValue(automaton.GetLogValue(seq)))).ToList();
                        if (list.Count == 1)
                        {
                            return FromPoint(list.First().Key);
                        }
                        else
                        {
                            return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(list));
                        }
                    }
                    break;
            }
            
            return Clone(); // TODO: replace with `this` after making automata immutable
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Repeat(int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of repetitions must be non-negative.");
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
                {
                    var dictAsList = dictionary.Dictionary.ToList();
                    var currentRepsEnumerable = dictAsList.AsEnumerable();
                    for (int i = 1; i < minTimes; ++i)
                        currentRepsEnumerable = currentRepsEnumerable.SelectMany(kvp => dictAsList.Select(skvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, skvp.Key), kvp.Value * skvp.Value)));
                    var resultList = new List<KeyValuePair<TSequence, Weight>>((int)resultSupportSize + 1);
                    resultList.AddRange(currentRepsEnumerable);
                    int lastRepStart = 0;
                    for (int i = minTimes; i < maxTimes; ++i)
                    {
                        int curRepStart = resultList.Count;
                        for (int j = lastRepStart; j < curRepStart; ++j)
                        {
                            var kvp = resultList[j];
                            foreach (var skvp in dictAsList)
                                resultList.Add(new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, skvp.Key), kvp.Value * skvp.Value));
                        }
                        lastRepStart = curRepStart;
                    }
                    return FromDictionary(DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TDictionary>.FromDistinctWeights(resultList));
                }
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

        public bool IsCanonicalZero() => weightFunction == null;

        public double MaxDiff(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> that)
        {
            if (IsCanonicalZero())
                return that.IsZero() ? 0.0 : Math.E;
            if (that.IsCanonicalZero())
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
            if (weightFunction.IsCanonicalZero())
                return Clone(); // TODO: return `this` when automata become immutable
            if (IsCanonicalZero())
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
            if (weightFunction.IsCanonicalZero() || double.IsNegativeInfinity(logWeight2))
                return ScaleLog(logWeight1);
            if (IsCanonicalZero() || double.IsNegativeInfinity(logWeight1))
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
            if (IsCanonicalZero() || weightFunction.IsCanonicalZero())
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
                if (logValue == 0.0)
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
            if (IsCanonicalZero())
                return other.IsZero();
            if (other.IsCanonicalZero())
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

        public override int GetHashCode() => AsAutomaton().GetHashCode();

        public override string ToString() => weightFunction?.ToString() ?? "{null}";
    }
}
