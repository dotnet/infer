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

    public partial class WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>
    {
        /// <summary>
        /// An implementation of <see cref="IWeightFunction{TThis}"/>
        /// that stores the entire support of the weight function along with the associated weights in a dictionary.
        /// </summary>
        /// <typeparam name="TThis">The type of a concrete dictionary weight function class.</typeparam>
        [Serializable]
        [DataContract]
        [Quality(QualityBand.Experimental)]
        public class DictionaryWeightFunction<TThis> : IWeightFunction<TThis>
        where TThis : DictionaryWeightFunction<TThis>, new()
        {
            protected static TSequenceManipulator SequenceManipulator =>
                    Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.SequenceManipulator;

            protected static TElementDistribution ElementDistributionFactory =>
                    Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ElementDistributionFactory;

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
                var result = Util.New<TThis>();
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
                var result = Util.New<TThis>();
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

            /// <summary>
            /// Creates a sequence to weight dictionary with <paramref name="point"/> being its only key which corresponds to a unit weight.
            /// </summary>
            /// <param name="point">The only sequence contained in the dictionary.</param>
            public static TThis FromPoint(TSequence point)
            {
                var result = Util.New<TThis>();
                result.SetDistinctWeights(new[] { new KeyValuePair<TSequence, Weight>(point, Weight.One) });
                return result;
            }

            #endregion

            /// <summary>
            /// A dictionary containing the entire support of the current weight function and the corresponding weights.
            /// </summary>
            public IReadOnlyDictionary<TSequence, Weight> Dictionary => dictionary;

            /// <inheritdoc/>
            public TSequence Point =>
                Dictionary.Count == 1 ? Dictionary.Single().Key : throw new InvalidOperationException("This weight function is zero everywhere or is non-zero on more than one sequence.");

            /// <inheritdoc/>
            public bool IsPointMass => Dictionary.Count == 1;

            /// <inheritdoc/>
            public bool UsesAutomatonRepresentation => false;

            /// <inheritdoc/>
            public bool UsesGroups => false;

            /// <inheritdoc/>
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

            /// <inheritdoc/>
            public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000)
            {
                if (maxCount < Dictionary.Count)
                    throw new AutomatonEnumerationCountException(maxCount);
                return Dictionary.Keys;
            }

            /// <inheritdoc/>
            public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result)
            {
                if (maxCount < Dictionary.Count)
                {
                    result = null;
                    return false;
                }
                result = Dictionary.Keys;
                return true;
            }

            /// <inheritdoc/>
            public TThis Repeat(int minTimes = 1, int? maxTimes = null)
            {
                Argument.CheckIfNotNull(maxTimes, nameof(maxTimes), "Can not represent unlimited repetitions as a finite dictionary.");

                return Repeat(minTimes, maxTimes.Value, 0);
            }

            /// <summary>
            /// Creates a weight function <c>g(s) = sum_{k=Kmin}^{Kmax} sum_{t1 t2 ... tk = s} f(t1)f(t2)...f(tk)</c>,
            /// where <c>f(t)</c> is the current weight function, and <c>Kmin</c> and <c>Kmax</c> are the minimum
            /// and the maximum number of factors in a sum term.
            /// </summary>
            /// <param name="minTimes">The minimum number of factors in a sum term. Defaults to 1.</param>
            /// <param name="maxTimes">The maximum number of factors in a sum term.</param>
            /// <param name="expectedResultSupportSize">Predicted size of the support of the resulting weight function.
            /// Does not have to be correct, but when it is, the performance is slightly improved.</param>
            /// <returns>The created weight function.</returns>
            public TThis Repeat(int minTimes, int maxTimes, int expectedResultSupportSize)
            {
                Argument.CheckIfInRange(minTimes >= 0, nameof(minTimes), "The minimum number of repetitions must be non-negative.");
                Argument.CheckIfValid(maxTimes >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

                if (maxTimes == 0)
                    return FromPoint(SequenceManipulator.ToSequence(Enumerable.Empty<TElement>()));

                var resultList = expectedResultSupportSize > 0 ? new List<KeyValuePair<TSequence, Weight>>(expectedResultSupportSize) : new List<KeyValuePair<TSequence, Weight>>();
                int lastRepStart = 0;
                if (minTimes == 0)
                {
                    resultList.Add(new KeyValuePair<TSequence, Weight>(SequenceManipulator.ToSequence(Enumerable.Empty<TElement>()), Weight.One));
                    lastRepStart = 1;
                    minTimes = 1;
                }
                var dictAsList = Dictionary.ToList();
                var currentRepsEnumerable = dictAsList.AsEnumerable();
                for (int i = 1; i < minTimes; ++i)
                    currentRepsEnumerable = currentRepsEnumerable.SelectMany(kvp => dictAsList.Select(skvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, skvp.Key), kvp.Value * skvp.Value)));
                resultList.AddRange(currentRepsEnumerable);
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
                return FromWeights(resultList);
            }

            /// <inheritdoc/>
            public TThis ScaleLog(double logScale)
            {
                var scale = Weight.FromLogValue(logScale);
                return FromDistinctWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * scale)));
            }

            /// <inheritdoc/>
            public Dictionary<int, TThis> GetGroups() => new Dictionary<int, TThis>();

            /// <summary>
            /// Gets a value indicating how close this weight function is to a given one
            /// in terms of weights they assign to sequences.
            /// </summary>
            /// <param name="that">The other weight function.</param>
            /// <returns>The logarithm of a non-negative value, which is close to zero if the two automata assign similar values to all sequences.</returns>
            protected double GetLogSimilarity(TThis that)
            {
                // Consistently with Automaton.GetLogSimilarity
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

            /// <inheritdoc/>
            public double MaxDiff(TThis that) => Math.Exp(GetLogSimilarity(that));

            /// <inheritdoc/>
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

            /// <inheritdoc/>
            public double GetLogNormalizer()
            {
                if (Dictionary == null || Dictionary.Count == 0)
                    return double.NegativeInfinity;

                return MMath.LogSumExp(Dictionary.Values.Select(v => v.LogValue));
            }

            /// <summary>
            /// Computes the logarithm of the normalizer (sum of values of the weight function on all sequences)
            /// of the square of the current weight function.
            /// </summary>
            /// <returns>The logarithm of the normalizer.</returns>
            protected double GetLogNormalizerOfSquare()
            {
                if (Dictionary == null || Dictionary.Count == 0)
                    return double.NegativeInfinity;
                else
                    return MMath.LogSumExp(Dictionary.Values.Select(v => 2 * v.LogValue));
            }

            /// <summary>
            /// Computes the logarithm of the normalizer (sum of values of the weight function on all sequences)
            /// of the product of the current and the given weight functions.
            /// </summary>
            /// <param name="weightFunction">The given weight function.</param>
            /// <returns>The logarithm of the normalizer.</returns>
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
                Dictionary.Select(kvp => new Tuple<List<TElementDistribution>, double>(kvp.Key.Select(el => ElementDistributionFactory.CreatePointMass(el)).ToList(), kvp.Value.LogValue));

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

            /// <inheritdoc/>
            public double GetLogValue(TSequence sequence)
            {
                if (Dictionary.TryGetValue(sequence, out Weight prob))
                    return prob.LogValue;

                return double.NegativeInfinity;
            }

            /// <inheritdoc/>
            public bool IsZero() => Dictionary.All(kvp => kvp.Value.IsZero);

            /// <inheritdoc/>
            public bool HasGroup(int group) => false;

            /// <inheritdoc/>
            public TThis NormalizeStructure() => Dictionary.Values.Any(val => val.IsZero) ? FromDistinctWeights(Dictionary.Where(kvp => !kvp.Value.IsZero)) : (TThis)this;

            /// <inheritdoc/>
            public TThis Append(TSequence sequence, int group = 0)
            {
                Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

                return FromDistinctWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, sequence), kvp.Value)));
            }

            /// <inheritdoc/>
            public TThis Append(TThis weightFunction, int group = 0)
            {
                Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

                return FromWeights(Dictionary.SelectMany(kvp => weightFunction.Dictionary.Select(skvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, skvp.Key), kvp.Value * skvp.Value))));
            }

            /// <inheritdoc/>
            public TThis Sum(TThis weightFunction) =>
                FromWeights(Dictionary.Concat(weightFunction.Dictionary));

            /// <inheritdoc/>
            public TThis Sum(double weight1, TThis weightFunction, double weight2)
            {
                Argument.CheckIfInRange(weight1 >= 0, nameof(weight1), "Negative weights are not supported.");
                Argument.CheckIfInRange(weight2 >= 0, nameof(weight2), "Negative weights are not supported.");

                return SumLog(Math.Log(weight1), weightFunction, Math.Log(weight2));
            }

            /// <inheritdoc/>
            public TThis SumLog(double logWeight1, TThis weightFunction, double logWeight2)
            {
                var scale1 = Weight.FromLogValue(logWeight1);
                var scale2 = Weight.FromLogValue(logWeight2);

                return FromWeights(
                    Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * scale1))
                    .Concat(weightFunction.Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, kvp.Value * scale2))));
            }

            /// <inheritdoc/>
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

            /// <inheritdoc/>
            public bool Equals(TThis other)
            {
                if (other == null || other.Dictionary.Count != Dictionary.Count)
                    return false;

                // Consistently with Automaton.Equals()
                double logSimilarity = GetLogSimilarity(other);
                const double LogSimilarityThreshold = -30;
                return logSimilarity < LogSimilarityThreshold;
            }

            /// <inheritdoc/>
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

            public string ToString(Action<TElementDistribution, StringBuilder> appendElement)
            {
                if (appendElement == null)
                    return ToString();

                var sb = new StringBuilder();
                if (Dictionary.Count > 1)
                    sb.Append('[');

                string prefix = string.Empty;
                foreach (var sequence in Dictionary.Keys)
                {
                    sb.Append(prefix);
                    foreach (var element in sequence)
                        appendElement(ElementDistributionFactory.CreatePointMass(element), sb);

                    prefix = "|";
                }

                if (Dictionary.Count > 1)
                    sb.Append(']');

                return sb.ToString();
            }
        }
    }

    /// <summary>
    /// <see cref="WeightFunctions{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton}.DictionaryWeightFunction{TThis}"/>
    /// specialized for strings.
    /// </summary>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class StringDictionaryWeightFunction : WeightFunctions<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.DictionaryWeightFunction<StringDictionaryWeightFunction>
    {
        public StringDictionaryWeightFunction() : base(new SortedList<string, Weight>(StringComparer.Ordinal)) { }
        private StringDictionaryWeightFunction(SortedList<string, Weight> sortedList) : base(sortedList) { }

        /// <inheritdoc/>
        protected override void SetWeights(IEnumerable<KeyValuePair<string, Weight>> sequenceWeightPairs)
        {
            var newDictionary = new SortedList<string, Weight>(StringComparer.Ordinal);
            FillDictionary(newDictionary, sequenceWeightPairs);
            dictionary = newDictionary;
        }

        /// <inheritdoc/>
        protected override void SetDistinctWeights(IEnumerable<KeyValuePair<string, Weight>> sequenceWeightPairs)
        {
            var dict = sequenceWeightPairs as IDictionary<string, Weight> ?? sequenceWeightPairs.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            dictionary = new SortedList<string, Weight>(dict, StringComparer.Ordinal);
        }

        /// <inheritdoc/>
        public override StringDictionaryWeightFunction Product(StringDictionaryWeightFunction weightFunction)
        {
            var dict1 = (SortedList<string, Weight>)Dictionary;
            var dict2 = (SortedList<string, Weight>)weightFunction.Dictionary;
            var resultList = new SortedList<string, Weight>(Math.Min(dict1.Count, dict2.Count), StringComparer.Ordinal);
            int idx1 = 0, idx2 = 0;
            while (idx1 < dict1.Count && idx2 < dict2.Count)
            {
                int comparisonResult = string.CompareOrdinal(dict1.Keys[idx1], dict2.Keys[idx2]);
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

        /// <inheritdoc/>
        protected override double GetLogNormalizerOfProduct(StringDictionaryWeightFunction weightFunction)
        {
            var dict1 = (SortedList<string, Weight>)Dictionary;
            var dict2 = (SortedList<string, Weight>)weightFunction.Dictionary;
            Weight normalizer = Weight.Zero;
            int idx1 = 0, idx2 = 0;
            while (idx1 < dict1.Count && idx2 < dict2.Count)
            {
                int comparisonResult = string.CompareOrdinal(dict1.Keys[idx1], dict2.Keys[idx2]);
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

        /// <summary>
        /// Converts the current <see cref="StringDictionaryWeightFunction"/> to a deterministic automaton,
        /// compressing shared suffixes. When the current <see cref="StringDictionaryWeightFunction"/> is
        /// normalized, the resulting automaton is stochastic.
        /// </summary>
        /// <returns>A deterministic <see cref="StringAutomaton"/>.</returns>
        public override StringAutomaton AsAutomaton()
        {
            var dict = (SortedList<string, Weight>)Dictionary;
            if (dict.Count == 0)
                return Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.Zero();
            if (dict.Count == 1)
                return Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.ConstantOnLog(dict.Values[0].LogValue, dict.Keys[0]);

            var result = new Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.Builder();
            var end = result
                .AddState()
                .SetEndWeight(Weight.One);
            var sharedPrefixWithPreviousLength = new int[dict.Count];
            sharedPrefixWithPreviousLength[0] = 0;
            for (int i = 1; i < dict.Count; ++i)
                sharedPrefixWithPreviousLength[i] = GetSharedPrefixLength(dict.Keys[i - 1], dict.Keys[i]);

            var suffixInfo = new (int suffixLength, int prefixEndStateIndex, char suffixTransitionChar, Weight suffixTransitionWeight)[dict.Count];

            var transitionsToMerge = new Dictionary<int, Dictionary<int, List<(char character, Weight weight)>>>();

            int processedEntriesCount = 0;
            do
            {
                processedEntriesCount = ProcessPrefixes(processedEntriesCount, result.Start, 0, Weight.One);
            }
            while (processedEntriesCount < dict.Count);

            var reversedStringSortIndex = Enumerable.Range(0, dict.Count).ToArray();
            // Ordinal sorting of reversed suffixes.
            Array.Sort(reversedStringSortIndex, new Comparison<int>((x, y) =>
            {
                int lenX = suffixInfo[x].suffixLength;
                int lenY = suffixInfo[y].suffixLength;
                if (lenX == 0 || lenY == 0)
                    return lenX - lenY;

                var shorterLength = Math.Min(lenX, lenY);
                var strX = dict.Keys[x];
                var strY = dict.Keys[y];
                int idx = 1;
                while (idx <= shorterLength && strX[strX.Length - idx] == strY[strY.Length - idx])
                    ++idx;
                if (idx > shorterLength)
                    return lenX - lenY;
                return strX[strX.Length - idx].CompareTo(strY[strY.Length - idx]);
            }));

            int curOriginalIdx = 0;
            int nextOriginalIdx = reversedStringSortIndex[0];
            int sharedPostfixWithPrevLen = 0;
            int sharedPostfixWithNextLen = 0;
            int firstSharedStateIdx = end.Index;
            for (int i = 0; i < dict.Count - 1; ++i)
            {
                curOriginalIdx = nextOriginalIdx;
                sharedPostfixWithPrevLen = sharedPostfixWithNextLen;
                nextOriginalIdx = reversedStringSortIndex[i + 1];
                sharedPostfixWithNextLen = GetSharedSuffixLength(dict.Keys[curOriginalIdx], suffixInfo[curOriginalIdx].suffixLength, dict.Keys[nextOriginalIdx], suffixInfo[nextOriginalIdx].suffixLength);
                firstSharedStateIdx = ProcessSuffix(curOriginalIdx, sharedPostfixWithPrevLen, firstSharedStateIdx, sharedPostfixWithNextLen);
            }
            ProcessSuffix(reversedStringSortIndex[dict.Count - 1], sharedPostfixWithNextLen, firstSharedStateIdx, 0);
            
            // Creating merged transitions
            foreach (var sourceStateAndDict in transitionsToMerge)
            {
                var currentState = result[sourceStateAndDict.Key];
                foreach (var destStateAndDistr in sourceStateAndDict.Value)
                {
                    var distr = destStateAndDistr.Value;
                    distr.Sort(new Comparison<(char character, Weight weight)>((x, y) => x.character.CompareTo(y.character)));
                    var ranges = new List<ImmutableDiscreteChar.CharRange>(distr.Count);
                    int currentStart = distr[0].character;
                    int currentEnd = currentStart + 1;
                    Weight currentWeight = distr[0].weight;
                    for (int i = 1; i < distr.Count; ++i)
                    {
                        (char character, Weight weight) = distr[i];
                        if (weight == currentWeight && character == currentEnd)
                            ++currentEnd;
                        else
                        {
                            ranges.Add(new ImmutableDiscreteChar.CharRange(currentStart, currentEnd, currentWeight));
                            currentStart = character;
                            currentEnd = currentStart + 1;
                            currentWeight = weight;
                        }
                    }
                    ranges.Add(new ImmutableDiscreteChar.CharRange(currentStart, currentEnd, currentWeight));
                    var weightNormalizer = Weight.FromLogValue(MMath.LogSumExp(ranges.Select(r => r.Probability.LogValue + Math.Log(r.EndExclusive - r.StartInclusive))));
                    currentState.AddTransition(new Collections.Option<ImmutableDiscreteChar>(ImmutableDiscreteChar.Create(ranges)), weightNormalizer, destStateAndDistr.Key);
                }
            }

            return Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.FromData(result.GetData(true));

            int GetSharedPrefixLength(string x, string y)
            {
                var shorterLength = Math.Min(x.Length, y.Length);
                int idx = 0;
                while (idx < shorterLength && x[idx] == y[idx])
                    ++idx;
                return idx;
            }

            // Length of a suffix shared by x.Substring(x.Length - suffixLengthX) and y.Substring(y.Length - suffixLengthY)
            int GetSharedSuffixLength(string x, int suffixLengthX, string y, int suffixLengthY)
            {
                var shorterLength = Math.Min(suffixLengthX, suffixLengthY);
                int idx = 1;
                while (idx <= shorterLength && x[x.Length - idx] == y[y.Length - idx])
                    ++idx;
                return idx - 1;
            }

            // Recursively creates a part of the result automaton that corresponds to shared prefixes and populates suffixInfo.
            // Every call is responsible for processing substrings [currentSharedPrefixLength..] of strings in dict.Keys starting at index
            // startIdx that have at least one additional shared character. Newly created part of the automaton will be attached to currentState.
            //
            // sharedPrefixWithPreviousLength must be populated prior to calling this. Deterministicity of the resulting automaton is guaranteed
            // by the alphabetical sorting of string in dict.Keys, which ensures that strings that share a prefix will fall into the same batch.
            //
            // When weightNormalizer equals to the inverse of the total weight of the currently processed batch, the newly created part of the automaton
            // will be stochastic.
            //
            // Returns the index of the first string that doesn't share at least (currentSharedPrefixLength + 1) first chars with dict.Keys[startIdx].
            // If that index lies within the batch of strings being processed, i.e. sharing a prefix of length currentSharedPrefixLength,
            // this function should then be called again for that index.
            int ProcessPrefixes(int startIdx, Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.Builder.StateBuilder currentState, int currentSharedPrefixLength, Weight weightNormalizer)
            {
                int nextStartIdx = startIdx + 1; // Starting index of the next batch
                int nextSharedPrefixLength = int.MaxValue; // Length of the prefix shared by all string in the current batch
                while (nextStartIdx < sharedPrefixWithPreviousLength.Length && sharedPrefixWithPreviousLength[nextStartIdx] > currentSharedPrefixLength)
                {
                    nextSharedPrefixLength = Math.Min(nextSharedPrefixLength, sharedPrefixWithPreviousLength[nextStartIdx]);
                    ++nextStartIdx;
                }

                if (nextStartIdx == startIdx + 1)
                {
                    // Current batch consists of a single string. End recursion.
                    string currentString = dict.Keys[startIdx];
                    var weight = dict.Values[startIdx] * weightNormalizer;
                    if (currentString.Length == currentSharedPrefixLength)
                    {
                        // String ends on the current state, which is not the shared end state, because
                        // there're other stings, for which the current string is a prefix.
                        currentState.SetEndWeight(weight);
                    }
                    else if (currentString.Length == currentSharedPrefixLength + 1)
                    {
                        // Just one more character, so the corresponding transition should lead to the end state.
                        AddOrPrepareForMergeTransition(currentState, currentString[currentSharedPrefixLength], weight, end.Index);
                    }
                    else
                    {
                        // Multiple characters left, populate an entry in suffixInfo.
                        suffixInfo[startIdx] = (currentString.Length - currentSharedPrefixLength - 1, currentState.Index, currentString[currentSharedPrefixLength], weight);
                    }
                }
                else
                {
                    // Multiple entries in the batch.
                    // Append a sequence corresponding to shared prefix and make recursive calls at the split.
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
                        idx = ProcessPrefixes(idx, currentState, nextSharedPrefixLength, childBatchWeightNormalizer);
                    }
                    while (idx < nextStartIdx);
                }

                return nextStartIdx;
            }

            // Adds states and transitions corresponding to the suffix of dict.Keys[idx] to the result automaton.
            // Reuses part of the suffix shared with previously processed suffixes.
            // Returns the index of the first state that can be reused by the next suffix.
            // Numbers of characters shared with previous and next suffixes are supplied as parameters.
            int ProcessSuffix(int idx, int sharedSuffixWithPrevLength, int firstStateOfSuffixSharedWithPrevIdx, int sharedSuffixWithNextLength)
            {
                (int suffixLength, int prefixEndStateIndex, char suffixTransitionChar, Weight suffixTransitionWeight) = suffixInfo[idx];
                int firstStateOfSuffixSharedWithNextIdx = end.Index;
                if (suffixLength != 0)
                {
                    var currentState = result[prefixEndStateIndex];
                    int transitionsUntilSuffixSharedWithNext = suffixLength - sharedSuffixWithNextLength;
                    if (suffixLength == sharedSuffixWithPrevLength)
                    {
                        // The first transition leads to the shared part.
                        currentState = AddOrPrepareForMergeTransition(currentState, suffixTransitionChar, suffixTransitionWeight, firstStateOfSuffixSharedWithPrevIdx);
                    }
                    else
                    {
                        // Adding states and transitions not shared with previously processed suffix.
                        currentState = currentState.AddTransition(suffixTransitionChar, suffixTransitionWeight);
                        var currentString = dict.Keys[idx];
                        for (int i = currentString.Length - suffixLength; i < currentString.Length - sharedSuffixWithPrevLength - 1; ++i)
                        {
                            if (transitionsUntilSuffixSharedWithNext == 0)
                                firstStateOfSuffixSharedWithNextIdx = currentState.Index;

                            currentState = currentState.AddTransition(currentString[i], Weight.One);
                            --transitionsUntilSuffixSharedWithNext;
                        }
                        if (transitionsUntilSuffixSharedWithNext == 0)
                            firstStateOfSuffixSharedWithNextIdx = currentState.Index;
                        currentState = AddOrPrepareForMergeTransition(currentState, currentString[currentString.Length - sharedSuffixWithPrevLength - 1], Weight.One, firstStateOfSuffixSharedWithPrevIdx);
                        --transitionsUntilSuffixSharedWithNext;
                    }
                    if (transitionsUntilSuffixSharedWithNext >= 0)
                    {
                        // If we didn't encounter the state, from which the part shared with the next suffix starts yet,
                        // find it within the part shared with the previous suffix.
                        for (int i = 0; i < transitionsUntilSuffixSharedWithNext; ++i)
                            currentState = result[currentState.TransitionIterator.Value.DestinationStateIndex];

                        firstStateOfSuffixSharedWithNextIdx = currentState.Index;
                    }
                }

                return firstStateOfSuffixSharedWithNextIdx;
            }

            Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.Builder.StateBuilder AddOrPrepareForMergeTransition(Automaton<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.Builder.StateBuilder currentState, char transitionChar, Weight weight, int destinationStateIndex)
            {
                if (transitionsToMerge.TryGetValue(currentState.Index, out var stateDict))
                {
                    if (stateDict.TryGetValue(destinationStateIndex, out var distr))
                    {
                        distr.Add((transitionChar, weight));
                        return result[destinationStateIndex];
                    }
                }
                if (currentState.HasTransitions)
                {
                    for (var iterator = currentState.TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var currentTransition = iterator.Value;
                        if (currentTransition.DestinationStateIndex == destinationStateIndex)
                        {
                            if (stateDict == null)
                            {
                                stateDict = new Dictionary<int, List<(char character, Weight weight)>>();
                                transitionsToMerge.Add(currentState.Index, stateDict);
                            }
                            var distr = new List<(char character, Weight weight)>();
                            distr.Add((currentTransition.ElementDistribution.Value.Point, currentTransition.Weight));
                            distr.Add((transitionChar, weight));
                            stateDict.Add(destinationStateIndex, distr);
                            iterator.Remove();
                            return result[destinationStateIndex];
                        }
                    }
                }

                return currentState.AddTransition(transitionChar, weight, destinationStateIndex);
            }
        }
    }

    /// <summary>
    /// A <see cref="WeightFunctions{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton}.DictionaryWeightFunction{TThis}"/>
    /// defined on types implementing <see cref="IList{T}"/>.
    /// </summary>
    /// <typeparam name="TList">The type of a list the automaton is defined on.</typeparam>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over a list element.</typeparam>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListDictionaryWeightFunction<TList, TElement, TElementDistribution> :
        WeightFunctions<
            TList,
            TElement,
            TElementDistribution,
            ListManipulator<TList, TElement>,
            ListAutomaton<TList, TElement, TElementDistribution>>
        .DictionaryWeightFunction<ListDictionaryWeightFunction<TList, TElement, TElementDistribution>>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, Sampleable<TElement>, new()
    {
    }

    /// <summary>
    /// A <see cref="WeightFunctions{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton}.DictionaryWeightFunction{TThis}"/>
    /// defined on generic lists.
    /// </summary>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over a list element.</typeparam>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListDictionaryWeightFunction<TElement, TElementDistribution> :
        WeightFunctions<
            List<TElement>,
            TElement,
            TElementDistribution,
            ListManipulator<List<TElement>, TElement>,
            ListAutomaton<TElement, TElementDistribution>>
        .DictionaryWeightFunction<ListDictionaryWeightFunction<TElement, TElementDistribution>>
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, Sampleable<TElement>, new()
    {
    }
}
