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
    public class DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton> : IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton, TDictionary>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TPointMass : PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>, new()
        where TDictionary : DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>, new()
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
        public static TDictionary FromWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            var result = new TDictionary();
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

        public void SetToSum(IEnumerable<TDictionary> weightFunctions)
        {
            throw new NotImplementedException();
        }

        public void SetToProduct(TDictionary a, TDictionary b)
        {
            throw new NotImplementedException();
        }

        public void SetToSum(double weight1, TDictionary value1, double weight2, TDictionary value2)
        {
            throw new NotImplementedException();
        }

        public void SetToSumLog(double logWeight1, TDictionary weightFunction1, double logWeight2, TDictionary weightFunction2)
        {
            throw new NotImplementedException();
        }

        public TDictionary Repeat(int minTimes = 1, int? maxTimes = null)
        {
            throw new NotImplementedException();
        }

        public TDictionary ScaleLog(double logScale)
        {
            throw new NotImplementedException();
        }

        public Dictionary<int, TDictionary> GetGroups() => new Dictionary<int, TDictionary>();

        public double MaxDiff(TDictionary that)
        {
            throw new NotImplementedException();
        }

        public void SetToConstantOnSupportOfLog(double logValue, TDictionary weightFunction)
        {
            throw new NotImplementedException();
        }

        public virtual bool TryNormalizeValues(out double logNormalizer)
        {
            if (Dictionary.Count == 0)
            {
                logNormalizer = double.NegativeInfinity;
                return false;
            }
            double logNormalizerLocal = MMath.LogSumExp(Dictionary.Values.Select(v => v.LogValue));
            logNormalizer = logNormalizerLocal;
            if (double.IsNaN(logNormalizerLocal) || double.IsInfinity(logNormalizerLocal))
                return false;
            if (logNormalizerLocal == 0.0)
                return true;
            // TODO: do it faster
            SetWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromLogValue(kvp.Value.LogValue - logNormalizerLocal))).ToList());
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

        public void NormalizeStructure()
        {
            if (Dictionary.Any(kvp => kvp.Value.IsZero))
                SetWeights(Dictionary.Where(kvp => !kvp.Value.IsZero).ToList());
        }

        public TDictionary Append(TSequence sequence, int group = 0)
        {
            Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

            return FromWeights(Dictionary.Select(kvp => new KeyValuePair<TSequence, Weight>(SequenceManipulator.Concat(kvp.Key, sequence), kvp.Value)));
        }

        public TDictionary Append(TDictionary weightFunction, int group = 0)
        {
            throw new NotImplementedException();
        }
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class StringDictionaryWeightFunction : DictionaryWeightFunction<string, char, DiscreteChar, StringManipulator, StringPointMassWeightFunction, StringDictionaryWeightFunction, StringAutomaton>
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
    public class ListDictionaryWeightFunction<TList, TElement, TElementDistribution> : DictionaryWeightFunction<TList, TElement, TElementDistribution, ListManipulator<TList, TElement>, ListPointMassWeightFunction<TList, TElement, TElementDistribution>, ListDictionaryWeightFunction<TList, TElement, TElementDistribution>, ListAutomaton<TList, TElement, TElementDistribution>>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListDictionaryWeightFunction<TElement, TElementDistribution> : DictionaryWeightFunction<List<TElement>, TElement, TElementDistribution, ListManipulator<List<TElement>, TElement>, ListPointMassWeightFunction<TElement, TElementDistribution>, ListDictionaryWeightFunction<TElement, TElementDistribution>, ListAutomaton<TElement, TElementDistribution>>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }
}
