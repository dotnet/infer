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

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis> : IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TThis : PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>, new()
    {
        private static TSequenceManipulator SequenceManipulator =>
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.SequenceManipulator;

        private static TSequence EmptySequence = SequenceManipulator.ToSequence(Enumerable.Empty<TElement>());

        [DataMember]
        private TSequence point = EmptySequence;

        #region Factory methods

        [Construction(nameof(Point))]
        public static TThis FromPoint(TSequence point) => new TThis() { Point = point };

        #endregion

        public TSequence Point
        {
            get => point;
            // Setter should only ever be used in factory methods
            // TODO: replace with `init` after switching to C# 9.0+
            private set
            {
                Argument.CheckIfNotNull(value, nameof(value), "Point mass must not be null.");
                point = value;
            }
        }

        public bool IsPointMass => true;

        public bool UsesAutomatonRepresentation => false;

        public bool UsesGroups => false;

        public TAutomaton AsAutomaton() => Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ConstantOn(1.0, point);

        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true)
        {
            return new List<TSequence>(new[] { point });
        }

        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true)
        {
            result = new List<TSequence>(new[] { point });
            return true;
        }

        public double GetLogValue(TSequence sequence) => SequenceManipulator.SequenceEqualityComparer.Equals(point, sequence) ? 0.0 : double.NegativeInfinity;

        public TThis Repeat(int minTimes = 1, int? maxTimes = null)
        {
            throw new NotSupportedException($"{nameof(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>)} is not closed under repetition.");
        }

        public TThis ScaleLog(double logScale)
        {
            throw new NotSupportedException($"{nameof(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>)} is not closed under scaling.");
        }

        public Dictionary<int, TThis> GetGroups() => new Dictionary<int, TThis>();

        public double MaxDiff(TThis that) => SequenceManipulator.SequenceEqualityComparer.Equals(point, that.point) ? 0.0 : Math.E;

        public bool TryNormalizeValues(out TThis normalizedFunction, out double logNormalizer)
        {
            normalizedFunction = (TThis)this;
            logNormalizer = 0.0;
            return true;
        }

        public double GetLogNormalizer() => 0;

        public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths() =>
            new List<Tuple<List<TElementDistribution>, double>> { new Tuple<List<TElementDistribution>, double>(Point.Select(el => new TElementDistribution { Point = el }).ToList(), 0) };

        public bool IsZero() => false;

        public bool HasGroup(int group) => false;

        public TThis NormalizeStructure() => (TThis)this;

        public TThis Append(TSequence sequence, int group = 0)
        {
            Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

            return FromPoint(SequenceManipulator.Concat(point, sequence));
        }

        public TThis Append(TThis weightFunction, int group = 0)
        {
            Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

            return FromPoint(SequenceManipulator.Concat(point, weightFunction.Point));
        }

        public TThis Sum(TThis weightFunction)
        {
            throw new NotSupportedException($"{nameof(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>)} is not closed under summation.");
        }

        public TThis Sum(double weight1, double weight2, TThis weightFunction)
        {
            throw new NotSupportedException($"{nameof(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>)} is not closed under summation.");
        }

        public TThis SumLog(double logWeight1, double logWeight2, TThis weightFunction)
        {
            throw new NotSupportedException($"{nameof(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>)} is not closed under summation.");
        }

        public TThis Product(TThis weightFunction)
        {
            if (point == weightFunction.point)
                return (TThis)this;
            else
                throw new NotSupportedException($"Can not create a zero {nameof(PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>)}.");
        }

        public TThis Clone() => (TThis)this; // This type is immutable.

        public bool Equals(TThis other) => SequenceManipulator.SequenceEqualityComparer.Equals(point, other?.point);

        public override bool Equals(object obj)
        {
            if (obj == null || typeof(TThis) != obj.GetType())
            {
                return false;
            }

            return Equals((TThis)obj);
        }

        public override int GetHashCode() => 0; // Consistently with Automaton.GetHashCode()

        public override string ToString() => point.ToString(); // Point can not be null

        public string ToString(Action<TElementDistribution, StringBuilder> appendElement)
        {
            if (appendElement == null)
                return ToString();

            var sb = new StringBuilder();
            foreach (var element in point)
                appendElement(new TElementDistribution() { Point = element }, sb);

            return sb.ToString();
        }
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class StringPointMassWeightFunction : PointMassWeightFunction<string, char, DiscreteChar, StringManipulator, StringAutomaton, StringPointMassWeightFunction>
    {
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListPointMassWeightFunction<TList, TElement, TElementDistribution> : PointMassWeightFunction<TList, TElement, TElementDistribution, ListManipulator<TList, TElement>, ListAutomaton<TList, TElement, TElementDistribution>, ListPointMassWeightFunction<TList, TElement, TElementDistribution>>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }

    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public class ListPointMassWeightFunction<TElement, TElementDistribution> : PointMassWeightFunction<List<TElement>, TElement, TElementDistribution, ListManipulator<List<TElement>, TElement>, ListAutomaton<TElement, TElementDistribution>, ListPointMassWeightFunction<TElement, TElementDistribution>>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }
}
