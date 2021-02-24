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
        // TODO: use truly immutable storage.
        protected IReadOnlyDictionary<TSequence, Weight> dictionary;

        #region Constructors

        protected DictionaryWeightFunction(IReadOnlyDictionary<TSequence, Weight> dictionary)
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

        public static TThis FromValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs) =>
            FromWeights(sequenceWeightPairs.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromValue(kvp.Value))));

        #endregion

        public IReadOnlyDictionary<TSequence, Weight> Dictionary
        {
            get => dictionary;
            //protected set
            //{
            //    Argument.CheckIfNotNull(value, nameof(value), "Dictionary must not be null.");

            //    dictionary = new Dictionary<TSequence, Weight>(value, SequenceManipulator.SequenceEqualityComparer);
            //}
        }

        public TSequence Point 
        {
            get => Dictionary.Count == 1 ? Dictionary.Single().Key : throw new InvalidOperationException("This weight function is zero everywhere or is non-zero on more than one sequence.");
            //set
            //{
            //    if (Dictionary.Count != 1 || !SequenceManipulator.SequenceEqualityComparer.Equals(value, Dictionary.Single().Key))
            //    {
            //        Dictionary.Clear();
            //        Dictionary.Add(value, Weight.One);
            //    }
            //}
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
            // TODO: implement properly.
            return Math.Exp(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.GetLogSimilarity(
                AsAutomaton(), that.AsAutomaton()));
        }

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

        //public void SetValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs)
        //{
        //    SetWeights(sequenceWeightPairs.Select(kvp => new KeyValuePair<TSequence, Weight>(kvp.Key, Weight.FromValue(kvp.Value))));
        //}

        /// <summary>
        /// Replaces the internal sequence to weight dictionary with a new one using the supplied <paramref name="sequenceWeightPairs"/>.
        /// If the supplied collection contains multiple entries for the same sequence, the weights for that sequence are summed.
        /// </summary>
        /// <param name="sequenceWeightPairs">The collection of pairs of a sequence and the weight on that sequence.</param>
        protected virtual void SetWeights(IEnumerable<KeyValuePair<TSequence, Weight>> sequenceWeightPairs)
        {
            var newDictionary = new Dictionary<TSequence, Weight>(SequenceManipulator.SequenceEqualityComparer);
            FillDictionary(newDictionary, sequenceWeightPairs);
            dictionary = newDictionary;
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

        public TThis NormalizeStructure() => Dictionary.Values.Any(val => val.IsZero) ? FromWeights(Dictionary.Where(kvp => !kvp.Value.IsZero)) : (TThis)this;

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

        public TThis Clone() => (TThis)this; // This type is immutable.
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class StringDictionaryWeightFunction : DictionaryWeightFunction<string, char, DiscreteChar, StringManipulator, StringAutomaton, StringDictionaryWeightFunction>
    {
        public StringDictionaryWeightFunction() : base(new SortedList<string, Weight>()) { }

        protected override void SetWeights(IEnumerable<KeyValuePair<string, Weight>> sequenceWeightPairs)
        {
            var newDictionary = new SortedList<string, Weight>();
            FillDictionary(newDictionary, sequenceWeightPairs);
            dictionary = newDictionary;
        }

        //public override IDictionary<string, Weight> Dictionary
        //{
        //    get => dictionary;
        //    protected set
        //    {
        //        Argument.CheckIfNotNull(value, nameof(value), "Dictionary must not be null.");

        //        dictionary = new SortedList<string, Weight>(value);
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
