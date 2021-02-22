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
        protected IDictionary<TSequence, Weight> dictionary;

        #region Constructors

        protected DictionaryWeightFunction(IDictionary<TSequence, Weight> dictionary)
        {
            this.dictionary = dictionary;
        }

        public DictionaryWeightFunction() : this(new Dictionary<TSequence, Weight>(SequenceManipulator.SequenceEqualityComparer)) { }

        #endregion

        #region Factory methods

        [Construction(nameof(Dictionary))]
        public static TThis FromWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            var result = new TThis();
            result.SetWeights(sequenceWeightPairs);
            return result;
        }

        #endregion

        public IDictionary<TSequence, Weight> Dictionary
        {
            get => dictionary;
            //set
            //{
            //    Argument.CheckIfNotNull(value, nameof(value), "Dictionary must not be null.");
                
            //    dictionary = new Dictionary<TSequence, double>(value, SequenceManipulator.SequenceEqualityComparer);
            //}
        }

        public TSequence Point 
        {
            get => Dictionary.Count == 1 ? Dictionary.Single().Key : throw new InvalidOperationException("This weight function is zero everywhere or is non-zero on more than one sequence.");
            set
        {
                if (Dictionary.Count != 1 || !SequenceManipulator.SequenceEqualityComparer.Equals(value, Dictionary.Single().Key))
                {
                    Dictionary.Clear();
                    Dictionary.Add(value, Weight.One);
                }
            }
        }

        public bool IsPointMass => Dictionary.Count == 1;

        public bool UsesAutomatonRepresentation => false;

        public bool UsesGroups => false;

        public TAutomaton AsAutomaton()
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

        public void SetToSum(IEnumerable<TThis> weightFunctions)
        {
            throw new NotImplementedException();
        }

        public TThis Repeat(int minTimes = 1, int? maxTimes = null)
        {
            throw new NotImplementedException();
        }

        public TThis ScaleLog(double logScale)
        {
            throw new NotImplementedException();
        }

        public Dictionary<int, TThis> GetGroups() => new Dictionary<int, TThis>();

        public double MaxDiff(TThis that)
        {
            throw new NotImplementedException();
        }

        public void SetToConstantOnSupportOfLog(double logValue, TThis weightFunction)
        {
            throw new NotImplementedException();
        }

        public virtual bool TryNormalizeValues(out TThis normalizedFunction, out double logNormalizer)
        {
            if (Dictionary.Count == 0)
            {
                normalizedFunction = FromWeights(Dictionary); // TODO: replace with `this` after making this type immutable
                logNormalizer = double.NegativeInfinity;
                return false;
            }
            double logNormalizerLocal = MMath.LogSumExp(Dictionary.Values.Select(v => v.LogValue));
            logNormalizer = logNormalizerLocal;
            if (double.IsNaN(logNormalizerLocal) || double.IsInfinity(logNormalizerLocal))
            {
                normalizedFunction = FromWeights(Dictionary); // TODO: replace with `this` after making this type immutable
                return false;
            }
            if (logNormalizerLocal == 0.0)
            {
                normalizedFunction = FromWeights(Dictionary); // TODO: replace with `this` after making this type immutable
                return true;
            }
            // TODO: do it faster
            normalizedFunction = FromWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromLogValue(kvp.Value.LogValue - logNormalizerLocal))).ToList());
            return true;
        }

        public double GetLogNormalizer()
        {
            if (Dictionary == null || Dictionary.Count == 0)
                return double.NegativeInfinity;

            return MMath.LogSumExp(Dictionary.Values.Select(v => v.LogValue));
        }

        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths()
        {
            throw new NotImplementedException();
        }

        public void SetValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs)
        {
            SetWeights(sequenceWeightPairs.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromValue(kvp.Value))));
        }

        public void SetWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            Dictionary.Clear();
            foreach (var kvp in sequenceWeightPairs)
            {
                if (Dictionary.TryGetValue(kvp.Key, out Weight prob))
                    Dictionary[kvp.Key] = prob * kvp.Value;
                else
                    Dictionary.Add(kvp);
            }
        }

        public double GetLogValue(TSequence sequence)
        {
            if (Dictionary.TryGetValue(sequence, out Weight prob))
                return prob.LogValue;

            return double.NegativeInfinity;
        }

        public bool IsZero() => Dictionary.All(kvp => kvp.Value.IsZero);

        public void SetToZero()
        {
            throw new NotImplementedException();
        }

        public void SetToConstantLog(double logValue, TElementDistribution allowedElements)
        {
            throw new NotImplementedException();
        }

        public bool HasGroup(int group)
        {
            throw new NotImplementedException();
        }

        public TThis NormalizeStructure() => FromWeights(Dictionary.Where(kvp => !kvp.Value.IsZero)); // TODO: return `this` when normalization is not required after making this type immutable

        public TThis Append(TSequence sequence, int group = 0)
        {
            Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

            return FromWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, sequence), kvp.Value)));
        }

        public TThis Append(TThis weightFunction, int group = 0)
        {
            throw new NotImplementedException();
        }

        public TThis Sum(TThis weightFunction)
        {
            throw new NotImplementedException();
        }

        public TThis Sum(double weight1, double weight2, TThis weightFunction)
        {
            throw new NotImplementedException();
        }

        public TThis SumLog(double logWeight1, double logWeight2, TThis weightFunction)
        {
            throw new NotImplementedException();
        }

        public TThis Product(TThis weightFunction)
        {
            throw new NotImplementedException();
        }
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class StringDictionaryWeightFunction : DictionaryWeightFunction<string, char, DiscreteChar, StringManipulator, StringAutomaton, StringDictionaryWeightFunction>
    {
        public StringDictionaryWeightFunction() : base(new SortedList<string, Weight>()) { }

        //public override IDictionary<string, double> Dictionary
        //{
        //    get => dictionary;
        //    set
        //    {
        //        Argument.CheckIfNotNull(value, nameof(value), "Dictionary must not be null.");

        //        dictionary = new SortedList<string, double>(value);
        //    }
        //}

        //public override void AppendInPlace(string sequence, int group = 0)
        //{
        //    Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");
        //    var list = (SortedList<string, double>)Dictionary;
        //    for (int i = 0; i < list.Keys.Count; ++i)
        //        list.Keys[i] += sequence; // Can't do this - colletion is read-only
        //}

        //public override bool TryNormalizeValues(out double logNormalizer)
        //{
        //    if (!(Dictionary is SortedList<string, double> list) || list.Count == 0)
        //    {
        //        logNormalizer = double.NegativeInfinity;
        //        return false;
        //    }
        //    double normalizer = list.Values.Sum();
        //    logNormalizer = Math.Log(normalizer);
        //    if (normalizer == 0.0 || double.IsNaN(normalizer) || double.IsInfinity(normalizer))
        //        return false;
        //    if (normalizer == 1.0)
        //        return true;

        //    for (int i = 0; i < list.Values.Count; ++i)
        //        list.Values[i] /= normalizer; // Can't do this - colletion is read-only
        //    return true;
        //}
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
