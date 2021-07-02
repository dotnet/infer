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
        /// that represents a point mass weight function, aka a weight function that maps one sequence to 1.0 and all other sequences to zero.
        /// </summary>
        [Serializable]
        [DataContract]
        [Quality(QualityBand.Experimental)]
        public sealed class PointMassWeightFunction : IWeightFunction<PointMassWeightFunction>
        {
            private static TSequenceManipulator SequenceManipulator =>
                    Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.SequenceManipulator;

            private static TElementDistribution ElementDistributionFactory =>
                    Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ElementDistributionFactory;

            private static TSequence EmptySequence = SequenceManipulator.ToSequence(Enumerable.Empty<TElement>());

            [DataMember]
            private TSequence point = EmptySequence;

            #region Factory methods

            /// <summary>
            /// Creates a point mass weight function.
            /// </summary>
            /// <param name="point">The point.</param>
            /// <returns>The created point mass weight function.</returns>
            [Construction(nameof(Point))]
            public static PointMassWeightFunction FromPoint(TSequence point) => new PointMassWeightFunction() { Point = point };

            #endregion

            /// <summary>
            /// Gets the only point to which the current function assigns a non-zero weight.
            /// </summary>
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

            /// <inheritdoc/>
            public bool IsPointMass => true;

            /// <inheritdoc/>
            public bool UsesAutomatonRepresentation => false;

            /// <inheritdoc/>
            public bool UsesGroups => false;

            /// <inheritdoc/>
            public TAutomaton AsAutomaton() => Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ConstantOn(1.0, point);

            /// <inheritdoc/>
            public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000)
            {
                return new List<TSequence>(new[] { point });
            }

            /// <inheritdoc/>
            public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result)
            {
                result = new List<TSequence>(new[] { point });
                return true;
            }

            /// <inheritdoc/>
            public double GetLogValue(TSequence sequence) => SequenceManipulator.SequenceEqualityComparer.Equals(point, sequence) ? 0.0 : double.NegativeInfinity;

            /// <inheritdoc/>
            /// <remarks>Not supported for <see cref="PointMassWeightFunction"/>.</remarks>
            public PointMassWeightFunction Repeat(int minTimes = 1, int? maxTimes = null)
            {
                throw new NotSupportedException($"{nameof(PointMassWeightFunction)} is not closed under repetition.");
            }

            /// <inheritdoc/>
            /// <remarks>Not supported for <see cref="PointMassWeightFunction"/>.</remarks>
            public PointMassWeightFunction ScaleLog(double logScale)
            {
                throw new NotSupportedException($"{nameof(PointMassWeightFunction)} is not closed under scaling.");
            }

            /// <inheritdoc/>
            public Dictionary<int, PointMassWeightFunction> GetGroups() => new Dictionary<int, PointMassWeightFunction>();

            /// <inheritdoc/>
            public double MaxDiff(PointMassWeightFunction that) => SequenceManipulator.SequenceEqualityComparer.Equals(point, that.point) ? 0.0 : Math.E;

            /// <inheritdoc/>
            public bool TryNormalizeValues(out PointMassWeightFunction normalizedFunction, out double logNormalizer)
            {
                normalizedFunction = this;
                logNormalizer = 0.0;
                return true;
            }

            /// <inheritdoc/>
            public double GetLogNormalizer() => 0;

            /// <inheritdoc/>
            public IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths() =>
                new List<Tuple<List<TElementDistribution>, double>> { new Tuple<List<TElementDistribution>, double>(Point.Select(el => ElementDistributionFactory.CreatePointMass(el)).ToList(), 0) };

            /// <inheritdoc/>
            public bool IsZero() => false;

            /// <inheritdoc/>
            public bool HasGroup(int group) => false;

            /// <inheritdoc/>
            public PointMassWeightFunction NormalizeStructure() => this;

            /// <inheritdoc/>
            /// <remarks><see cref="PointMassWeightFunction"/> only supports case
            /// <c><paramref name="group"/> == 0</c>.</remarks>
            public PointMassWeightFunction Append(TSequence sequence, int group = 0)
            {
                Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

                return FromPoint(SequenceManipulator.Concat(point, sequence));
            }

            /// <inheritdoc/>
            /// <remarks><see cref="PointMassWeightFunction"/> only supports case
            /// <c><paramref name="group"/> == 0</c>.</remarks>
            public PointMassWeightFunction Append(PointMassWeightFunction weightFunction, int group = 0)
            {
                Argument.CheckIfValid(group == 0, nameof(group), "Groups are not supported.");

                return FromPoint(SequenceManipulator.Concat(point, weightFunction.Point));
            }

            /// <inheritdoc/>
            /// <remarks>Not supported for <see cref="PointMassWeightFunction"/>.</remarks>
            public PointMassWeightFunction Sum(PointMassWeightFunction weightFunction)
            {
                throw new NotSupportedException($"{nameof(PointMassWeightFunction)} is not closed under summation.");
            }

            /// <inheritdoc/>
            /// <remarks>Not supported for <see cref="PointMassWeightFunction"/>.</remarks>
            public PointMassWeightFunction Sum(double weight1, PointMassWeightFunction weightFunction, double weight2)
            {
                throw new NotSupportedException($"{nameof(PointMassWeightFunction)} is not closed under summation.");
            }

            /// <inheritdoc/>
            /// <remarks>Not supported for <see cref="PointMassWeightFunction"/>.</remarks>
            public PointMassWeightFunction SumLog(double logWeight1, PointMassWeightFunction weightFunction, double logWeight2)
            {
                throw new NotSupportedException($"{nameof(PointMassWeightFunction)} is not closed under summation.");
            }

            /// <inheritdoc/>
            /// <remarks><see cref="PointMassWeightFunction"/> only supports case
            /// <c><paramref name="weightFunction"/> == <see langword="this"/></c>.</remarks>
            public PointMassWeightFunction Product(PointMassWeightFunction weightFunction)
            {
                if (point == weightFunction.point)
                    return this;
                else
                    throw new NotSupportedException($"Can not create a zero {nameof(PointMassWeightFunction)}.");
            }

            /// <inheritdoc/>
            public bool Equals(PointMassWeightFunction other) => SequenceManipulator.SequenceEqualityComparer.Equals(point, other?.point);

            /// <inheritdoc/>
            public override bool Equals(object obj)
            {
                if (obj == null || typeof(PointMassWeightFunction) != obj.GetType())
                {
                    return false;
                }

                return Equals((PointMassWeightFunction)obj);
            }

            public override int GetHashCode() => 0; // Consistently with Automaton.GetHashCode()

            public override string ToString() => point.ToString(); // Point can not be null

            /// <inheritdoc/>
            public string ToString(Action<TElementDistribution, StringBuilder> appendElement)
            {
                if (appendElement == null)
                    return ToString();

                var sb = new StringBuilder();
                foreach (var element in point)
                    appendElement(ElementDistributionFactory.CreatePointMass(element), sb);

                return sb.ToString();
            }
        }
    }
}
