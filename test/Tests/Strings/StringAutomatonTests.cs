// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Xunit;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;

    /// <summary>
    /// Tests functionality specific to <see cref="StringAutomaton"/>.
    /// </summary>
    public class StringAutomatonTests
    {
        #region Determinization

        /// <summary>
        /// Tests building lists of non-overlapping outgoing transitions when determinizing
        /// weighted finite-state automata operating on strings.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GetOutgoingTransitionsForDeterminization1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition(DiscreteChar.Uniform(), Weight.FromValue(2));

            var wrapper = new StringAutomatonWrapper(builder);
            
            var outgoingTransitions =
                wrapper.GetOutgoingTransitionsForDeterminization(0, Weight.FromValue(3));
            var expectedOutgoingTransitions = new[]
            {
                ValueTuple.Create(
                    DiscreteChar.Uniform(),
                    Weight.FromValue(2),
                    new[] {(1, Weight.FromValue(1))})
            };

            AssertCollectionsEqual(expectedOutgoingTransitions, outgoingTransitions, TransitionInfoEqualityComparer.Instance);
        }

        /// <summary>
        /// Tests building lists of non-overlapping outgoing transitions when determinizing
        /// weighted finite-state automata operating on strings.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GetOutgoingTransitionsForDeterminization2()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', 'z'), Weight.FromValue(2));
            builder.Start.AddTransition(DiscreteChar.UniformInRanges('a', 'z', 'A', 'Z'), Weight.FromValue(3));

            var wrapper = new StringAutomatonWrapper(builder);
            
            var outgoingTransitions =
                wrapper.GetOutgoingTransitionsForDeterminization(0, Weight.FromValue(5));
            var expectedOutgoingTransitions = new[]
            {
                ValueTuple.Create(
                    DiscreteChar.UniformInRange('A', 'Z'),
                    Weight.FromValue(1.5),
                    new[] {(2, Weight.FromValue(1))}),
                ValueTuple.Create(
                    DiscreteChar.UniformInRange('a', 'z'),
                    Weight.FromValue(3.5),
                    new[] {(1, Weight.FromValue(1)), (2, Weight.FromValue(0.75))}),
            };

            AssertCollectionsEqual(expectedOutgoingTransitions, outgoingTransitions, TransitionInfoEqualityComparer.Instance);
        }

        /// <summary>
        /// Tests building lists of non-overlapping outgoing transitions when determinizing
        /// weighted finite-state automata operating on strings.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GetOutgoingTransitionsForDeterminization3()
        {
            var builder = new StringAutomaton.Builder();

            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', 'b'), Weight.FromValue(2));
            builder.Start.AddTransition(DiscreteChar.UniformInRanges('b', 'd'), Weight.FromValue(3));
            builder.Start.AddTransition(DiscreteChar.UniformInRanges('e', 'g'), Weight.FromValue(4));
            builder.Start.AddTransition(DiscreteChar.UniformInRanges(char.MinValue, 'a'), Weight.FromValue(5));

            var wrapper = new StringAutomatonWrapper(builder);

            var outgoingTransitions =
                wrapper.GetOutgoingTransitionsForDeterminization(0, Weight.FromValue(6));
            var expectedOutgoingTransitions = new[]
            {
                ValueTuple.Create(
                    DiscreteChar.UniformInRange(char.MinValue, (char) ('a' - 1)),
                    Weight.FromValue(5.0 * 97.0 / 98.0),
                    new[] {(4, Weight.FromValue(1))}),
                ValueTuple.Create(
                    DiscreteChar.PointMass('a'),
                    Weight.FromValue((5.0 / 98.0) + 1.0),
                    new[]
                    {
                        (1, Weight.FromValue(1.0 / ((30.0 / 98.0) + 1.0))),
                        (4, Weight.FromValue((5.0 / 98.0) / ((30.0 / 98.0) + 1.0)))
                    }),
                ValueTuple.Create(
                    DiscreteChar.PointMass('b'),
                    Weight.FromValue(2.0),
                    new[] {(1, Weight.FromValue(0.5)), (2, Weight.FromValue(0.5))}),
                ValueTuple.Create(
                    DiscreteChar.UniformInRange('c', 'd'),
                    Weight.FromValue(2.0),
                    new[] {(2, Weight.FromValue(1.0))}),
                ValueTuple.Create(
                    DiscreteChar.UniformInRange('e', 'g'),
                    Weight.FromValue(4.0),
                    new[] {(3, Weight.FromValue(1.0))}),
            };

            AssertCollectionsEqual(expectedOutgoingTransitions, outgoingTransitions, TransitionInfoEqualityComparer.Instance);
        }

        /// <summary>
        /// Tests building lists of non-overlapping outgoing transitions when determinizing
        /// weighted finite-state automata operating on strings.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GetOutgoingTransitionsForDeterminization4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition(DiscreteChar.UniformInRange(char.MinValue, char.MaxValue), Weight.FromValue(2));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', char.MaxValue), Weight.FromValue(3));
            builder.Start.AddTransition(DiscreteChar.UniformInRanges('z', char.MaxValue), Weight.FromValue(4));

            var wrapper = new StringAutomatonWrapper(builder);

            var outgoingTransitions =
                wrapper.GetOutgoingTransitionsForDeterminization(0, Weight.FromValue(5));

            double transition1Segment1Weight = 2.0 * 'a' / (char.MaxValue + 1.0);
            double transition1Segment2Weight = 2.0 * ('z' - 'a') / (char.MaxValue + 1.0);
            double transition1Segment3Weight = 2.0 * (char.MaxValue - 'z' + 1.0) / (char.MaxValue + 1.0);
            double transition2Segment1Weight = 3.0 * ('z' - 'a') / (char.MaxValue - 'a' + 1.0);
            double transition2Segment2Weight = 3.0 * (char.MaxValue - 'z' + 1.0) / (char.MaxValue - 'a' + 1.0);
            double transition3Segment1Weight = 4.0;
            var expectedOutgoingTransitions = new[]
            {
                ValueTuple.Create(
                    DiscreteChar.UniformInRange(char.MinValue, (char)('a' - 1)),
                    Weight.FromValue(transition1Segment1Weight),
                    new[] {(1, Weight.FromValue(1))}),
                ValueTuple.Create(
                    DiscreteChar.UniformInRange('a', (char)('z' - 1)),
                    Weight.FromValue(transition1Segment2Weight + transition2Segment1Weight),
                    new[]
                    {
                        (1, Weight.FromValue(transition1Segment2Weight / (transition1Segment2Weight + transition2Segment1Weight))),
                        (2, Weight.FromValue(transition2Segment1Weight / (transition1Segment2Weight + transition2Segment1Weight))),
                    }),
                ValueTuple.Create(
                    DiscreteChar.UniformInRange('z', char.MaxValue),
                    Weight.FromValue(transition1Segment3Weight + transition2Segment2Weight + transition3Segment1Weight),
                    new[]
                    {
                        (1, Weight.FromValue(transition1Segment3Weight / (transition1Segment3Weight + transition2Segment2Weight + transition3Segment1Weight))),
                        (2, Weight.FromValue(transition2Segment2Weight / (transition1Segment3Weight + transition2Segment2Weight + transition3Segment1Weight))),
                        (3, Weight.FromValue(transition3Segment1Weight / (transition1Segment3Weight + transition2Segment2Weight + transition3Segment1Weight))),
                    }),
            };

            AssertCollectionsEqual(expectedOutgoingTransitions, outgoingTransitions, TransitionInfoEqualityComparer.Instance);
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void GetOutgoingTrainsitionsForDeterminization5()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('A', Weight.FromValue(2.49999999999995));
            builder.Start.AddTransition('A', Weight.FromValue(4.49999999999959));
            builder.Start.AddTransition('D', Weight.FromValue(2.49999999999996));
            builder.Start.AddTransition('D', Weight.FromValue(4.49999999999966));
            builder.Start.AddTransition('K', Weight.FromValue(5.00000001783332));
            builder.Start.AddTransition('M', Weight.FromValue(2.49999999999996));
            builder.Start.AddTransition('M', Weight.FromValue(2.49999999999991));
            builder.Start.AddTransition('N', Weight.FromValue(2.49999999999996));
            builder.Start.AddTransition('N', Weight.FromValue(2.49999999999994));

            var wrapper = new StringAutomatonWrapper(builder);

            var outgoingTransitions =
                wrapper.GetOutgoingTransitionsForDeterminization(0, Weight.FromValue(1)).ToArray();

            Assert.Equal(5, outgoingTransitions.Length);
            Assert.True(outgoingTransitions.All(ot => ot.Item1.IsPointMass));
        }
        #endregion

        #region Helper methods

        /// <summary>
        /// Asserts that expected and actual collections are equal with respect to a given comparer.
        /// </summary>
        /// <typeparam name="T">The type of a collection element.</typeparam>
        /// <param name="expected">The expected collection.</param>
        /// <param name="actual">The actual collection.</param>
        /// <param name="comparer">The comparer.</param>
        private static void AssertCollectionsEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual, IEqualityComparer<T> comparer)
        {
            bool areEqual = Enumerable.SequenceEqual(expected, actual, comparer);
            Assert.True(areEqual, "Given collections are not equal.");
        }

        #endregion

        #region Nested classes

        #region Comparers

        /// <summary>
        /// A base class for all equality comparer implementations in this file.
        /// </summary>
        /// <typeparam name="T">The type of objects being compared.</typeparam>
        /// <typeparam name="TThis">The type of derived class.</typeparam>
        private abstract class EqualityComparerBase<T, TThis> : IEqualityComparer<T>
            where TThis : EqualityComparerBase<T, TThis>, new()
        {
            /// <summary>
            /// An instance of the comparer.
            /// </summary>
            public static readonly TThis Instance = new TThis();

            /// <summary>
            /// Overridden in the derived classes to check whether two objects are equal.
            /// </summary>
            /// <param name="x">The first object.</param>
            /// <param name="y">The second object.</param>
            /// <returns>
            /// <see langword="true"/> if the objects are equal,
            /// <see langword="false"/> otherwise.
            /// </returns>
            public abstract bool Equals(T x, T y);

            /// <summary>
            /// Computes the hash code of an object. Not supported by implementations of this class.
            /// </summary>
            /// <param name="obj">The object.</param>
            /// <returns>The computed hash code.</returns>
            public int GetHashCode(T obj)
            {
                throw new InvalidOperationException("This method should not be called.");
            }
        }

        /// <summary>
        /// A comparer for <see cref="Weight"/> that allows for some tolerance.
        /// </summary>
        private class WeightEqualityComparer :
            EqualityComparerBase<Weight, WeightEqualityComparer>
        {
            /// <summary>
            /// Checks whether two objects are equal.
            /// </summary>
            /// <param name="x">The first object.</param>
            /// <param name="y">The second object.</param>
            /// <returns>
            /// <see langword="true"/> if the objects are equal,
            /// <see langword="false"/> otherwise.
            /// </returns>
            public override bool Equals(Weight x, Weight y)
            {
                return Math.Abs(x.LogValue - y.LogValue) < 1e-8;
            }
        }

        /// <summary>
        /// A comparer for weighted states that allows for some tolerance when comparing weights.
        /// </summary>
        private class WeightedStateEqualityComparer :
            EqualityComparerBase<(int, Weight), WeightedStateEqualityComparer>
        {
            /// <summary>
            /// Checks whether two objects are equal.
            /// </summary>
            /// <param name="x">The first object.</param>
            /// <param name="y">The second object.</param>
            /// <returns>
            /// <see langword="true"/> if the objects are equal,
            /// <see langword="false"/> otherwise.
            /// </returns>
            public override bool Equals((int, Weight) x, (int, Weight) y)
            {
                return x.Item1 == y.Item1 && WeightEqualityComparer.Instance.Equals(x.Item2, y.Item2);
            }
        }

        /// <summary>
        /// A comparer for transition information that allows for some tolerance when comparing weights.
        /// </summary>
        private class TransitionInfoEqualityComparer :
            EqualityComparerBase<ValueTuple<DiscreteChar, Weight, (int, Weight)[]>, TransitionInfoEqualityComparer>
        {
            /// <summary>
            /// Checks whether two objects are equal.
            /// </summary>
            /// <param name="x">The first object.</param>
            /// <param name="y">The second object.</param>
            /// <returns>
            /// <see langword="true"/> if the objects are equal,
            /// <see langword="false"/> otherwise.
            /// </returns>
            public override bool Equals(
                ValueTuple<DiscreteChar, Weight, (int, Weight)[]> x,
                ValueTuple<DiscreteChar, Weight, (int, Weight)[]> y)
            {
                return
                    object.Equals(x.Item1, y.Item1) &&
                    WeightEqualityComparer.Instance.Equals(x.Item2, y.Item2) &&
                    x.Item3.SequenceEqual(y.Item3, WeightedStateEqualityComparer.Instance);
            }
        }

        #endregion

        /// <summary>
        /// Makes possible to call protected methods of <see cref="StringAutomaton"/>.
        /// </summary>
        private class StringAutomatonWrapper : StringAutomaton
        {
            public StringAutomatonWrapper(StringAutomaton.Builder builder)
            {
                this.Data = builder.GetData();
            }

            /// <summary>
            /// Wraps the protected method with the same name, providing access.
            /// </summary>
            /// <param name="sourceState">The source state.</param>
            /// <returns>The produced transitions.</returns>
            /// <remarks>See the doc of the original method.</remarks>
            public IEnumerable<(DiscreteChar, Weight, (int, Weight)[])>
                GetOutgoingTransitionsForDeterminization(int sourceState, Weight sourceWeight)
            {
                var weightedStateSetBuilder = Determinization.WeightedStateSetBuilder.Create();
                weightedStateSetBuilder.Add(sourceState, sourceWeight);

                var (weightedStateSet, weight) = weightedStateSetBuilder.Get();
                var result = base.GetOutgoingTransitionsForDeterminization(weightedStateSet);
                return result.Select(t => (
                    t.Item1,
                    t.Item2 * weight,
                    t.Item3.ToArray().Select(state => (state.Index, state.Weight)).ToArray()));
            }
        }
        #endregion
    }
}
