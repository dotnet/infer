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
    public class DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis> : IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TThis : DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>, new()
    {
        protected static TSequenceManipulator SequenceManipulator =>
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.SequenceManipulator;

        [DataMember]
        // Should only ever be set in factory methods.
        // TODO: use truly immutable storage,
        // replace with init-only property after switching to C# 9.0+
        protected IReadOnlyDictionary<TSequence, Weight> dictionary;

        #region Constructors

        protected DictionaryWeightFunction(IReadOnlyDictionary<TSequence, Weight> dictionary)
        {
            this.dictionary = dictionary;
        }

        public DictionaryWeightFunction() : this(new Dictionary<TSequence, Weight>(SequenceManipulator.SequenceEqualityComparer)) { }

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates a sequence to weight dictionary using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection contains multiple entries for the same sequence, the weights for that sequence are summed.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        public static TThis FromWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            var result = new TThis();
            result.SetWeights(sequenceWeightPairs);
            return result;

        }

        /// <summary>
        /// Creates a sequence to weight dictionary using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection is expected to not contain multiple entries for the same sequence.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        [Construction(nameof(Dictionary))]
        public static TThis FromDistinctWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            var result = new TThis();
            result.SetDistinctWeights(sequenceWeightPairs);
            return result;
        }

        /// <summary>
        /// Creates a sequence to weight dictionary using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection contains multiple entries for the same sequence, the weights for that sequence are summed.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        public static TThis FromValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs) =>
            FromWeights(sequenceWeightPairs.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromValue(kvp.Value))));

        /// <summary>
        /// Creates a sequence to weight dictionary using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection is expected to not contain multiple entries for the same sequence.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        public static TThis FromDistinctValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs) =>
            FromDistinctWeights(sequenceWeightPairs.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromValue(kvp.Value))));

        public static TThis FromPoint(TSequence point)
        {
            var result = new TThis();
            result.SetDistinctWeights(new[] { new KeyValuePair<TSequence, Weight>(point, Weight.One) });
            return result;
        }

        #endregion

        public IReadOnlyDictionary<TSequence, Weight> Dictionary => dictionary;

        public TSequence Point =>
            Dictionary.Count == 1 ? Dictionary.Single().Key : throw new InvalidOperationException("This weight function is zero everywhere or is non-zero on more than one sequence.");

        public bool IsPointMass => Dictionary.Count == 1;

        public bool UsesAutomatonRepresentation => false;

        public bool UsesGroups => false;

        public virtual TAutomaton AsAutomaton()
        {
            var result = new Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Builder();
            foreach (var entry in Dictionary)
            {
                if (!entry.Value.IsZero)
                {
                    var sequenceStartState = result.AddState();
                    var sequenceEndState = sequenceStartState.AddTransitionsForSequence(entry.Key);
                    sequenceEndState.SetEndWeight(Weight.One);
                    result.Start.AddEpsilonTransition(entry.Value, sequenceStartState.Index);
                }
            }

            return result.GetAutomaton();
        }

        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true)
        {
            if (maxCount < Dictionary.Count)
                throw new AutomatonEnumerationCountException(maxCount);
            return Dictionary.Keys;
        }

        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true)
        {
            if (maxCount < Dictionary.Count)
            {
                result = null;
                return false;
            }
            result = Dictionary.Keys;
            return true;
        }

        public TThis Repeat(int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfNotNull(maxTimes, nameof(maxTimes), "Can not represent unlimited repetitions as a finite dictionary.");

            return Repeat(minTimes, maxTimes.Value, 0);
        }
        public TThis Repeat(int minTimes, int maxTimes, int expectedResultSupportSize)
        {
            Argument.CheckIfInRange(minTimes >= 0, nameof(minTimes), "The minimum number of repetitions must be non-negative.");
            Argument.CheckIfValid(maxTimes >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

            var dictAsList = Dictionary.ToList();
            var currentRepsEnumerable = dictAsList.AsEnumerable();
            for (int i = 1; i < minTimes; ++i)
                currentRepsEnumerable = currentRepsEnumerable.SelectMany(kvp => dictAsList.Select(skvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, skvp.Key), kvp.Value * skvp.Value)));
            var resultList = expectedResultSupportSize > 0 ? new List<KeyValuePair<TSequence, Weight>>(expectedResultSupportSize) : new List<KeyValuePair<TSequence, Weight>>();
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
            return FromDistinctWeights(resultList);
        }

        public TThis ScaleLog(double logScale)
        {
            var scale = Weight.FromLogValue(logScale);
            return FromDistinctWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * scale)));
        }

        public Dictionary<int, TThis> GetGroups() => new Dictionary<int, TThis>();

        protected double GetLogSimilarity(TThis that)
        {
            var thisIsZero = IsZero();
            var thatIsZero = that.IsZero();
            if (thisIsZero)
            {
                return thatIsZero ? double.NegativeInfinity : 1.0;
            }
            if (thatIsZero)
                return 1.0;

            double logNormProduct = GetLogNormalizerOfProduct(that);
            if (double.IsNegativeInfinity(logNormProduct))
                return 1.0;

            double logNormThisSquared = GetLogNormalizerOfSquare();
            double logNormThatSquared = that.GetLogNormalizerOfSquare();

            double term1 = MMath.LogSumExp(logNormThisSquared, logNormThatSquared);
            double term2 = logNormProduct + MMath.Ln2;
            double result = MMath.LogDifferenceOfExp(Math.Max(term1, term2), Math.Min(term1, term2)); // To avoid NaN due to numerical instabilities
            System.Diagnostics.Debug.Assert(!double.IsNaN(result), "Log-similarity must be a valid number.");
            return result;
        }

        public double MaxDiff(TThis that) => Math.Exp(GetLogSimilarity(that));

        public virtual bool TryNormalizeValues(out TThis normalizedFunction, out double logNormalizer)
        {
            double logNormalizerLocal = GetLogNormalizer();
            logNormalizer = logNormalizerLocal;
            if (double.IsNaN(logNormalizerLocal) || double.IsInfinity(logNormalizerLocal))
            {
                normalizedFunction = (TThis)this;
                return false;
            }
            if (logNormalizerLocal == 0.0)
            {
                normalizedFunction = (TThis)this;
                return true;
            }
            normalizedFunction = FromDistinctWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromLogValue(kvp.Value.LogValue - logNormalizerLocal))));
            return true;
        }

        public double GetLogNormalizer()
        {
            if (Dictionary == null || Dictionary.Count == 0)
                return double.NegativeInfinity;

            return MMath.LogSumExp(Dictionary.Values.Select(v => v.LogValue));
        }

        protected double GetLogNormalizerOfSquare()
        {
            if (Dictionary == null || Dictionary.Count == 0)
                return double.NegativeInfinity;
            else
                return MMath.LogSumExp(Dictionary.Values.Select(v => 2 * v.LogValue));
        }

        protected virtual double GetLogNormalizerOfProduct(TThis weightFunction)
        {
            IReadOnlyDictionary<TSequence, Weight> dict1, dict2;
            if (Dictionary.Count <= weightFunction.Dictionary.Count)
            {
                dict1 = Dictionary;
                dict2 = weightFunction.Dictionary;
            }
            else
            {
                dict1 = weightFunction.Dictionary;
                dict2 = Dictionary;
            }
            Weight normalizer = Weight.Zero;
            foreach (var kvp in dict1)
                if (dict2.TryGetValue(kvp.Key, out Weight weight2))
                    normalizer += kvp.Value * weight2;

            return normalizer.LogValue;
        }

        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths() =>
            Dictionary.Select(kvp => new Tuple<List<TElementDistribution>, double>(kvp.Key.Select(el => new TElementDistribution { Point = el }).ToList(), kvp.Value.LogValue));

        /// <summary>
        /// Replaces the internal sequence to weight dictionary with a new one using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection contains multiple entries for the same sequence, the weights for that sequence are summed.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        /// <remarks>Should only ever be called in factory methods.</remarks>
        protected virtual void SetWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            var newDictionary = new Dictionary<TSequence, Weight>(SequenceManipulator.SequenceEqualityComparer);
            FillDictionary(newDictionary, sequenceWeightPairs);
            dictionary = newDictionary;
        }

        /// <summary>
        /// Replaces the internal sequence to weight dictionary with a new one using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection is expected to not contain multiple entries for the same sequence.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        /// <remarks>Should only ever be called in factory methods.</remarks>
        protected virtual void SetDistinctWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            dictionary = sequenceWeightPairs.ToDictionary(kvp => kvp.Key, kvp => kvp.Value, SequenceManipulator.SequenceEqualityComparer);
        }

        /// <summary>
        /// Fills the supplied empty sequence to weight dictionary with values from the given collection.
        /// If the supplied collection contains multiple entries for the same sequence, the weights for that sequence are summed.
        /// </summary>
        /// <param name="dictionaryToFill">The target empty dictionary.</param>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        protected static void FillDictionary(IDictionary<TSequence, Weight> dictionaryToFill, IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            foreach (var kvp in sequenceWeightPairs)
            {
                if (dictionaryToFill.TryGetValue(kvp.Key, out Weight prob))
                    dictionaryToFill[kvp.Key] = prob + kvp.Value;
                else
                    dictionaryToFill.Add(kvp.Key, kvp.Value);
            }
        }

        public double GetLogValue(TSequence sequence)
        {
            if (Dictionary.TryGetValue(sequence, out Weight prob))
                return prob.LogValue;

            return double.NegativeInfinity;
        }

        public bool IsZero() => Dictionary.All(kvp => kvp.Value.IsZero);

        public bool HasGroup(int group) => false;

        public TThis NormalizeStructure() => Dictionary.Values.Any(val => val.IsZero) ? FromDistinctWeights(Dictionary.Where(kvp => !kvp.Value.IsZero)) : (TThis)this;

        public TThis Append(TSequence sequence, int group = 0)
        {
            Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

            return FromDistinctWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, sequence), kvp.Value)));
        }

        public TThis Append(TThis weightFunction, int group = 0)
        {
            Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

            return FromWeights(Dictionary.SelectMany(kvp => weightFunction.Dictionary.Select(skvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, skvp.Key), kvp.Value * skvp.Value))));
        }

        public TThis Sum(TThis weightFunction) =>
            FromWeights(Dictionary.Concat(weightFunction.Dictionary));

        public TThis Sum(double weight1, double weight2, TThis weightFunction)
        {
            Argument.CheckIfInRange(weight1 >= 0, nameof(weight1), "Negative weights are not supported.");
            Argument.CheckIfInRange(weight2 >= 0, nameof(weight2), "Negative weights are not supported.");

            return SumLog(Math.Log(weight1), Math.Log(weight2), weightFunction);
        }

        public TThis SumLog(double logWeight1, double logWeight2, TThis weightFunction)
        {
            var scale1 = Weight.FromLogValue(logWeight1);
            var scale2 = Weight.FromLogValue(logWeight2);

            return FromWeights(
                Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * scale1))
                .Concat(weightFunction.Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * scale2))));
        }

        public virtual TThis Product(TThis weightFunction)
        {
            IReadOnlyDictionary<TSequence, Weight> dict1, dict2;
            if (Dictionary.Count <= weightFunction.Dictionary.Count)
            {
                dict1 = Dictionary;
                dict2 = weightFunction.Dictionary;
            }
            else
            {
                dict1 = weightFunction.Dictionary;
                dict2 = Dictionary;
            }
            var resultList = new List<KeyValuePair<TSequence, Weight>>(dict1.Count);
            foreach (var kvp in dict1)
                if (dict2.TryGetValue(kvp.Key, out Weight weight2))
                    resultList.Add(new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * weight2));

            return FromDistinctWeights(resultList);
        }

        public TThis Clone() => (TThis)this; // This type is immutable.

        public bool Equals(TThis other)
        {
            if (other == null || other.Dictionary.Count != Dictionary.Count)
                return false;

            // Consistently with Automaton.Equals()
            double logSimilarity = GetLogSimilarity(other);
            const double LogSimilarityThreshold = -30;
            return logSimilarity < LogSimilarityThreshold;
        }

        public override bool Equals(object obj)
        {
            if (obj == null || typeof(TThis) != obj.GetType())
            {
                return false;
            }

            return Equals((TThis)obj);
        }

        public override int GetHashCode()
        {
            // Consistently with Automaton.GetHashCode()
            return (BitConverter.DoubleToInt64Bits(GetLogNormalizerOfSquare()) >> 31).GetHashCode();
        }

        public override string ToString()
        {
            return $"[{string.Join("|", Dictionary.Keys)}]";
        }
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class StringDictionaryWeightFunction : DictionaryWeightFunction<string, char, DiscreteChar, StringManipulator, StringAutomaton, StringDictionaryWeightFunction>
    {
        public StringDictionaryWeightFunction() : base(new SortedList<string, Weight>()) { }
        private StringDictionaryWeightFunction(SortedList<string, Weight> sortedList) : base(sortedList) { }

        protected override void SetWeights(IEnumerable<KeyValuePair<string, Weight>> sequenceWeightPairs)
        {
            var newDictionary = new SortedList<string, Weight>();
            FillDictionary(newDictionary, sequenceWeightPairs);
            dictionary = newDictionary;
        }

        protected override void SetDistinctWeights(IEnumerable<KeyValuePair<string, Weight>> sequenceWeightPairs)
        {
            var dict = sequenceWeightPairs as IDictionary<string, Weight> ?? sequenceWeightPairs.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            dictionary = new SortedList<string, Weight>(dict);
        }

        public override StringDictionaryWeightFunction Product(StringDictionaryWeightFunction weightFunction)
        {
            var dict1 = (SortedList<string, Weight>)Dictionary;
            var dict2 = (SortedList<string, Weight>)weightFunction.Dictionary;
            var resultList = new SortedList<string, Weight>(Math.Min(dict1.Count, dict2.Count));
            int idx1 = 0, idx2 = 0;
            while (idx1 < dict1.Count && idx2 < dict2.Count)
            {
                int comparisonResult = dict1.Keys[idx1].CompareTo(dict2.Keys[idx2]);
                if (comparisonResult < 0)
                    ++idx1;
                else if (comparisonResult > 0)
                    ++idx2;
                else
                {
                    resultList.Add(dict1.Keys[idx1], dict1.Values[idx1] * dict2.Values[idx2]);
                    ++idx1;
                    ++idx2;
                }
            }
            resultList.TrimExcess();

            return new StringDictionaryWeightFunction(resultList);
        }

        protected override double GetLogNormalizerOfProduct(StringDictionaryWeightFunction weightFunction)
        {
            var dict1 = (SortedList<string, Weight>)Dictionary;
            var dict2 = (SortedList<string, Weight>)weightFunction.Dictionary;
            Weight normalizer = Weight.Zero;
            int idx1 = 0, idx2 = 0;
            while (idx1 < dict1.Count && idx2 < dict2.Count)
            {
                int comparisonResult = dict1.Keys[idx1].CompareTo(dict2.Keys[idx2]);
                if (comparisonResult < 0)
                    ++idx1;
                else if (comparisonResult > 0)
                    ++idx2;
                else
                {
                    normalizer += dict1.Values[idx1] * dict2.Values[idx2];
                    ++idx1;
                    ++idx2;
                }
            }

            return normalizer.LogValue;
        }

        public override StringAutomaton AsAutomaton()
        {
            var dict = (SortedList<string, Weight>)Dictionary;
            if (dict.Count == 0)
                return Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>.Zero();

            var result = new Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>.Builder();
            var end = result
                .AddState()
                .SetEndWeight(Weight.One);
            var sharedPrefixWithPreviousLength = new int[dict.Count];
            sharedPrefixWithPreviousLength[0] = 0;
            for (int i = 1; i < dict.Count; ++i)
                sharedPrefixWithPreviousLength[i] = GetSharedPrefixLength(dict.Keys[i - 1], dict.Keys[i]);

            int processedEntriesCount = 0;
            do
            {
                processedEntriesCount = ProcessEntries(processedEntriesCount, result.Start, 0, Weight.One);
            }
            while (processedEntriesCount < dict.Count);

            return Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>.FromData(result.GetData(true));

            int GetSharedPrefixLength(string x, string y)
            {
                var shorterLength = Math.Min(x.Length, y.Length);
                int idx = 0;
                while (idx < shorterLength && x[idx] == y[idx])
                    ++idx;
                return idx;
            }

            int ProcessEntries(int startIdx, Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>.Builder.StateBuilder currentState, int currentSharedPrefixLength, Weight weightNormalizer)
            {
                int nextStartIdx = startIdx + 1;
                int nextSharedPrefixLength = int.MaxValue;
                while (nextStartIdx < sharedPrefixWithPreviousLength.Length && sharedPrefixWithPreviousLength[nextStartIdx] > currentSharedPrefixLength)
                {
                    nextSharedPrefixLength = Math.Min(nextSharedPrefixLength, sharedPrefixWithPreviousLength[nextStartIdx]);
                    ++nextStartIdx;
                }

                if (nextStartIdx == startIdx + 1)
                {
                    string currentString = dict.Keys[startIdx];
                    var weight = dict.Values[startIdx] * weightNormalizer;
                    if (currentString.Length == currentSharedPrefixLength)
                    {
                        currentState.SetEndWeight(weight);
                    }
                    else if (currentString.Length == currentSharedPrefixLength + 1)
                    {
                        currentState.AddTransition(currentString[currentSharedPrefixLength], weight, end.Index);
                    }
                    else
                    {
                        currentState = currentState.AddTransition(currentString[currentSharedPrefixLength], weight);
                        for (int i = currentSharedPrefixLength + 1; i < currentString.Length - 1; ++i)
                        {
                            currentState = currentState.AddTransition(currentString[i], Weight.One);
                        }
                        currentState.AddTransition(currentString[currentString.Length - 1], Weight.One, end.Index);
                    }
                }
                else
                {
                    var batchTotalWeight = Weight.FromLogValue(MMath.LogSumExp(Enumerable.Range(startIdx, nextStartIdx - startIdx).Select(i => dict.Values[i].LogValue)));
                    var weight = batchTotalWeight * weightNormalizer;
                    var childBatchWeightNormalizer = Weight.Inverse(batchTotalWeight);

                    string firstString = dict.Keys[startIdx];
                    currentState = currentState.AddTransition(firstString[currentSharedPrefixLength], weight);
                    for (int i = currentSharedPrefixLength + 1; i < nextSharedPrefixLength; ++i)
                    {
                        currentState = currentState.AddTransition(firstString[i], Weight.One);
                    }
                    int idx = startIdx;
                    do
                    {
                        idx = ProcessEntries(idx, currentState, nextSharedPrefixLength, childBatchWeightNormalizer);
                    }
                    while (idx < nextStartIdx);
                }

                return nextStartIdx;
            }
        }
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListDictionaryWeightFunction<TList, TElement, TElementDistribution> : DictionaryWeightFunction<TList, TElement, TElementDistribution, ListManipulator<TList, TElement>, ListAutomaton<TList, TElement, TElementDistribution>, ListDictionaryWeightFunction<TList, TElement, TElementDistribution>>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListDictionaryWeightFunction<TElement, TElementDistribution> : DictionaryWeightFunction<List<TElement>, TElement, TElementDistribution, ListManipulator<List<TElement>, TElement>, ListAutomaton<TElement, TElementDistribution>, ListDictionaryWeightFunction<TElement, TElementDistribution>>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }
}
