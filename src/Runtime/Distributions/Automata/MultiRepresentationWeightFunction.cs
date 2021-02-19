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
        IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>>,
        SettableTo<MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>>,
        SettableTo<TAutomaton>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TPointMass : PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>, new()
        where TDictionary : DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>, new()
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

            set
            {
                Argument.CheckIfNotNull(value, nameof(value), "Point mass must not be null.");

                weightFunction = new TPointMass() { Point = value };
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

        public void SetTo(TPointMass pointMass)
        {
            Argument.CheckIfNotNull(pointMass, nameof(pointMass), "Point mass must not be null.");
            weightFunction = pointMass;
        }

        public void SetTo(TDictionary dictionary)
        {
            Argument.CheckIfNotNull(dictionary, nameof(dictionary), "Dictionary mass must not be null.");
            weightFunction = dictionary;
        }

        public void SetTo(TAutomaton automaton)
        {
            Argument.CheckIfNotNull(automaton, nameof(automaton), "Automaton mass must not be null.");
            weightFunction = automaton;
        }

        public void SetTo(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> other)
        {
            switch (other.weightFunction)
            {
                case null:
                    weightFunction = null;
                    break;
                case TPointMass pointMass:
                    Point = pointMass.Point;
                    break;
                case TDictionary dictionary:
                    weightFunction = new TDictionary();
                    ((TDictionary)weightFunction).SetWeights(dictionary.Dictionary);
                    break;
                case TAutomaton automaton:
                    weightFunction = automaton.Clone();
                    break;
                default:
                    throw new ArgumentException("Argument's weight function has an invalid type", nameof(other));
            }
        }

        public void SetToZero()
        {
            weightFunction = null;
        }

        public void SetValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceProbPairs)
        {
            Argument.CheckIfNotNull(sequenceProbPairs, nameof(sequenceProbPairs));
            var collection = sequenceProbPairs as ICollection<KeyValuePair<TSequence, double>> ?? sequenceProbPairs.ToList();
            if (collection.Count == 1 && collection.Single().Value == 1.0)
            {
                weightFunction = new TPointMass() { Point = collection.Single().Key };
            }
            else
            {
                if (collection.Count <= MaxDictionarySize)
                {
                    weightFunction = new TDictionary();
                    weightFunction.SetValues(collection);
                }
                else
                    weightFunction = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.FromValues(collection);
            }
        }

        //unused
        //public void SetToRepeat(IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> weightFunction, int minTimes = 1, int? maxTimes = null)
        //{
        //    Argument.CheckIfNotNull(weightFunction, nameof(weightFunction));
        //    Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of repetitions must be non-negative.");
        //    Argument.CheckIfValid(!maxTimes.HasValue || maxTimes.Value >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

        //    if (weightFunction is TPointMass pointMass && maxTimes.HasValue && maxTimes - minTimes < MaxDictionarySize)
        //    {
        //        var newSequenceElements = new List<TElement>(SequenceManipulator.GetLength(pointMass.Point) * minTimes);
        //        for (int i = 0; i < minTimes; ++i)
        //        {
        //            newSequenceElements.AddRange(pointMass.Point);
        //        }
        //        if (minTimes == maxTimes)
        //        {
        //            this.weightFunction = new TPointMass() { Point = SequenceManipulator.ToSequence(newSequenceElements) };
        //        }
        //        else
        //        {
        //            double uniformProb = 1.0 / (maxTimes.Value - minTimes);
        //            Dictionary<TSequence, double> dict = new Dictionary<TSequence, double>(maxTimes.Value - minTimes + 1);
        //            dict.Add(SequenceManipulator.ToSequence(newSequenceElements), uniformProb);
        //            for (int i = minTimes + 1; i <= maxTimes.Value; ++i)
        //            {
        //                dict.Add(SequenceManipulator.ToSequence(newSequenceElements), uniformProb);
        //                newSequenceElements.AddRange(pointMass.Point);
        //            }
        //            this.weightFunction = new TDictionary() { Dictionary = dict };
        //        }
        //    }
        //    else if (false && weightFunction is TDictionary dictionary && maxTimes.HasValue && Math.Pow(dictionary.Dictionary.Count, maxTimes.Value - minTimes) <= MaxDictionarySize)
        //    {
        //        // TODO
        //    }
        //    else
        //    {
        //        double logNormalizer = 1.0;// -weightFunction.GetLogAverageOf(weightFunction);
        //        this.weightFunction = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Repeat(
        //            weightFunction.AsAutomaton().ScaleLog(logNormalizer), minTimes, maxTimes); // AsAutomaton - get normalized automaton - TODO
        //    }
        //}

        /// <summary>
        /// Checks if the weight function uses groups.
        /// </summary>
        /// <returns><see langword="true"/> if the weight function uses groups, <see langword="false"/> otherwise.</returns>
        public bool UsesGroups => weightFunction is TAutomaton automaton && automaton.UsesGroups;

        public bool HasGroup(int group)
        {
            // TODO: get rid of groups or do something about groups + non-automaton weight function combo
            return weightFunction is TAutomaton automaton && automaton.HasGroup(group);
        }

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

        //public TAutomaton GetAutomaton() => weightFunction.AsAutomaton();

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

        ///// <summary>
        ///// Enumerates components of the distribution represented by this weight function.
        ///// </summary>
        ///// <remarks>
        ///// Any sequence distribution can be represented as a mixture of 'simple' sequence
        ///// distributions where each element is independent.  This method enumerates through
        ///// the components of this distribution treated as such a mixture.  Each component
        ///// is returned as a list of the independent element distributions for each element.
        ///// 
        ///// For some distributions, the number of components may be very large.
        ///// </remarks>
        ///// <returns>For each component, the list of element distributions and the log mixture weight for that component.</returns>
        //public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumerateComponents()
        //{
        //    //unused
        //    if (weightFunction is TPointMass pointMass)
        //    {
        //        var singleton = new List<Tuple<List<TElementDistribution>, double>>
        //        {
        //           new Tuple<List<TElementDistribution>, double>(pointMass.Point.Select(el => new TElementDistribution { Point = el }).ToList(),0)
        //        };

        //        return singleton;
        //    }

        //    // TODO: a special case for dictionaries

        //    return weightFunction.AsAutomaton().EnumeratePaths();
        //}

        ///// <summary>
        ///// Attempts to normalize the weight function so that sum of its values on all possible sequences equals to one (if it is possible).
        ///// </summary>
        ///// <returns><see langword="true"/> if the weight function was successfully normalized, <see langword="false"/> otherwise.</returns>>
        //public bool TryNormalizeValues() => weightFunction.TryNormalizeValues(out _);

        /// <summary>
        /// Modifies the weight function to be in normalized form e.g. using special
        /// case structures for point masses and fucntions with small support.
        /// </summary>
        public void NormalizeStructure()
        {
            switch (weightFunction)
            {
                case TDictionary dictionary:
                    var filteredTruncated = dictionary.Dictionary.Where(kvp => !kvp.Value.IsZero).Take(2).ToList();
                    if (filteredTruncated.Count == 1/* && dictionary.Dictionary.Single().Value.LogValue == 0.0*/)
                    {
                        weightFunction = new TPointMass() { Point = filteredTruncated.Single().Key };
                    }
                    else
                    {
                        weightFunction.NormalizeStructure();
                    }
                    break;
                case TAutomaton automaton:
                    if (!automaton.UsesGroups && automaton.TryEnumerateSupport(MaxDictionarySize, out var support, false))
                    {
                        // TODO: compute values along with support
                        var list = support.Select(seq => new KeyValuePair<TSequence, Weight>(seq, Weight.FromLogValue(automaton.GetLogValue(seq)))).ToList();
                        if (list.Count == 1/* && list.Single().Value.LogValue == 0.0*/)
                        {
                            weightFunction = new TPointMass() { Point = list.First().Key };
                        }
                        else
                        {
                            weightFunction = new TDictionary();
                            ((TDictionary)weightFunction).SetWeights(list);
                        }
                    }
                    break;
            }
        }

        public void SetToSum(IEnumerable<MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>> weightFunctions)
        {
            // TODO
            var automaton = new TAutomaton();
            automaton.SetToSum(weightFunctions.Select(wf => wf.AsAutomaton()));
            weightFunction = automaton;
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Repeat(int minTimes = 1, int? maxTimes = null)
        {
            // TODO
            var result = new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>();
            result.weightFunction = AsAutomaton().Repeat(minTimes, maxTimes);
            return result;
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> ScaleLog(double logScale)
        {
            // TODO
            var result = new MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>();
            result.weightFunction = AsAutomaton().ScaleLog(logScale);
            return result;
        }

        public bool TryNormalizeValues(out double logNormalizer)
        {
            if (weightFunction == null)
            {
                logNormalizer = double.NegativeInfinity;
                return false;
            }
            return weightFunction.TryNormalizeValues(out logNormalizer);
        }

        public double GetLogValue(TSequence sequence) => weightFunction?.GetLogValue(sequence) ?? double.NegativeInfinity;

        public bool IsZero() => weightFunction?.IsZero() ?? true;

        public bool IsCanonicalZero() => weightFunction == null;

        public double MaxDiff(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> that)
        {
            // TODO
            return AsAutomaton().MaxDiff(that.AsAutomaton());
        }

        public void SetToConstantOnSupportOfLog(double logValue, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            // TODO
            var automaton = new TAutomaton();
            automaton.SetToConstantOnSupportOfLog(logValue, weightFunction.AsAutomaton());
            this.weightFunction = automaton;
        }

        public double GetLogNormalizer() => weightFunction?.GetLogNormalizer() ?? double.NegativeInfinity;

        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths()
        {
            if (weightFunction is TPointMass pointMass)
            {
                var singleton = new List<Tuple<List<TElementDistribution>, double>>
                    {
                       new Tuple<List<TElementDistribution>, double>(pointMass.Point.Select(el => new TElementDistribution { Point = el }).ToList(), 0)
                    };

                return singleton;
            }

            // TODO: a special case for dictionaries

            return AsAutomaton().EnumeratePaths();
        }

        public void SetToConstantLog(double logValue, TElementDistribution allowedElements)
        {
            // TODO
            weightFunction = new TAutomaton();
            weightFunction.SetToConstantLog(logValue, allowedElements);
        }

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

                // TODO: if (weightFunction.weightFunction is TDictionary otherDictionary)
            }

            return FromAutomaton(this.weightFunction.AsAutomaton().Append(weightFunction.weightFunction.AsAutomaton(), group));
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Sum(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            // TODO
            return FromAutomaton(AsAutomaton().Sum(weightFunction.AsAutomaton()));
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Sum(double weight1, double weight2, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            // TODO
            var automaton = new TAutomaton();
            automaton.SetToSum(weight1, AsAutomaton(), weight2, weightFunction.AsAutomaton());
            return FromAutomaton(automaton);
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> SumLog(double logWeight1, double logWeight2, MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            // TODO
            var automaton = new TAutomaton();
            automaton.SetToSumLog(logWeight1, AsAutomaton(), logWeight2, weightFunction.AsAutomaton());
            return FromAutomaton(automaton);
        }

        public MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> Product(MultiRepresentationWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> weightFunction)
        {
            // TODO
            return FromAutomaton(AsAutomaton().Product(weightFunction.AsAutomaton()));
        }
    }
}
