// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;

    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;

    /// <summary>
    /// Contains a set of tests for automata with non-trivial loops,
    /// i.e. loops consisting of more than one transition.
    /// </summary>
    public class LoopyAutomatonTests
    {
        #region Repeat

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat1()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "abc");
            automaton.AppendInPlace(StringAutomaton.Constant(2.0, DiscreteChar.Upper()));
            StringInferenceTestUtilities.TestValue(automaton, 0.0, string.Empty, "ab", "abcab", "XYZ");
            StringInferenceTestUtilities.TestValue(automaton, 2.0, "abc", "abcX", "abcXXYZ");

            StringAutomaton loopyAutomaton = StringAutomaton.Repeat(automaton);
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 0.0, string.Empty, "ab", "abcab", "XYZ");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 2.0, "abc", "abcA");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 4.0, "abcabc", "abcabcX", "abcABCabc", "abcXabcYZ");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 8.0, "abcabcabc", "abcXabcYabcZZ");
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat2()
        {
            StringAutomaton automaton = StringAutomaton.Constant(1.0, DiscreteChar.Lower());
            StringInferenceTestUtilities.TestValue(automaton, 1.0, string.Empty, "ab", "abcab");

            automaton = StringAutomaton.Repeat(automaton);
            
            // Can't use StringInferenceTestUtilities.TestValue here since the value is not infinite in log-domain
            // due to approximate closure computations for epsilon-loops
            Assert.True(automaton.GetValue(string.Empty) > 1000);
            Assert.True(automaton.GetValue("ab") > 1000);
            Assert.True(automaton.GetValue("abcab") > 1000);
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat3()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "a");
            automaton.AppendInPlace(StringAutomaton.ConstantOn(1.0, "a"));
            automaton = StringAutomaton.Sum(automaton, StringAutomaton.ConstantOn(1.0, "a"));
            StringInferenceTestUtilities.TestValue(automaton, 0.0, string.Empty, "aaa", "ab", "X");
            StringInferenceTestUtilities.TestValue(automaton, 1.0, "a", "aa");

            // Yep, it turns out you can compute the Fibonacci sequence with an automaton
            StringAutomaton loopyAutomaton = StringAutomaton.Repeat(automaton);
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 0.0, string.Empty, "ab");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 1.0, "a");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 2.0, "aa");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 3.0, "aaa");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 5.0, "aaaa");
            StringInferenceTestUtilities.TestValue(loopyAutomaton, 8.0, "aaaaa");
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat4()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "aa");
            automaton = StringAutomaton.Repeat(automaton, minTimes: 0);

            StringInferenceTestUtilities.TestValue(automaton, 0.0, "a", "aaa");
            StringInferenceTestUtilities.TestValue(automaton, 1.0, string.Empty);
            StringInferenceTestUtilities.TestValue(automaton, 2.0, "aa");
            StringInferenceTestUtilities.TestValue(automaton, 4.0, "aaaa");
        }

        #endregion

        #region Point mass detection

        /// <summary>
        /// Tests whether a singleton support is being recognized correctly in the presence of an epsilon loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMassDetectionWithEpsilonLoop()
        {
            var f = new StringAutomaton.Builder();
            AddEpsilonLoop(f.Start, 5, 0.5);
            f.Start.AddTransitionsForSequence("abc").SetEndWeight(Weight.One);
            Assert.Equal("abc", f.GetAutomaton().TryComputePoint());
        }

        /// <summary>
        /// Tests whether a singleton support is being recognized correctly in the presence of a zero-weight loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMassDetectionWithDeadLoop()
        {
            var f = new StringAutomaton.Builder();
            f.Start.AddTransition('a', Weight.FromValue(0.5)).AddTransition('b', Weight.Zero, f.Start.Index);
            f.Start.AddTransitionsForSequence("abc").SetEndWeight(Weight.One);
            Assert.Equal("abc", f.GetAutomaton().TryComputePoint());
        }

        /// <summary>
        /// Tests whether a singleton support is being recognized correctly in the presence of a loop on a zero-weight path.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMassDetectionLoopInDeadEnd()
        {
            var f = new StringAutomaton.Builder();
            f.Start.AddTransition('a', Weight.FromValue(0.5)).AddSelfTransition('a', Weight.FromValue(0.5)).AddTransition('b', Weight.One);
            f.Start.AddTransition('b', Weight.FromValue(0.5)).SetEndWeight(Weight.One);
            Assert.Equal("b", f.GetAutomaton().TryComputePoint());
        }

        /// <summary>
        /// Tests whether a non-singleton support is being recognized correctly in the presence of a non-trivial loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NoPoint1()
        {
            var f = new StringAutomaton.Builder();
            f.Start.AddTransition('a', Weight.FromValue(0.5)).AddTransition('b', Weight.FromValue(0.5), f.Start.Index).SetEndWeight(Weight.One);
            Assert.Null(f.GetAutomaton().TryComputePoint());
        }

        /// <summary>
        /// Tests whether a non-singleton mass support is being recognized correctly in the presence of a non-trivial loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NoPoint2()
        {
            var f = new StringAutomaton.Builder();
            var state = f.Start.AddTransition('a', Weight.FromValue(0.5));
            state.SetEndWeight(Weight.One);
            state.AddTransition('b', Weight.FromValue(0.5), f.Start.Index);
            Assert.Null(f.GetAutomaton().TryComputePoint());
        }

        /// <summary>
        /// Tests whether a non-singleton support is being recognized correctly for an automaton with an empty support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NoPointZero()
        {
            var f = StringAutomaton.Zero();
            Assert.Null(f.TryComputePoint());
        }

        #endregion

        #region Zero detection

        /// <summary>
        /// Tests whether an empty support is being recognized correctly in the presence of a non-trivial epsilon loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ZeroDetectionWithEpsilonLoop1()
        {
            var builder = new StringAutomaton.Builder();
            AddEpsilonLoop(builder.Start, 5, 0);

            var f = builder.GetAutomaton();
            Assert.False(f.IsCanonicZero());
            Assert.True(f.IsZero());
        }

        /// <summary>
        /// Tests whether an empty support is being recognized correctly in the presence of a non-trivial epsilon loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ZeroDetectionWithEpsilonLoop2()
        {
            var builder = new StringAutomaton.Builder();
            AddEpsilonLoop(builder.Start, 5, 2.0);
            builder.Start.AddTransition('a', Weight.One);

            var f = builder.GetAutomaton();
            Assert.False(f.IsCanonicZero());
            Assert.True(f.IsZero());
        }

        /// <summary>
        /// Tests whether an empty support is being recognized correctly in the presence of a zero-weight loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ZeroDetectionWithDeadSelfLoop()
        {
            var f = new StringAutomaton.Builder();
            f.Start.AddSelfTransition('x', Weight.Zero);
            f.Start.AddTransition('y', Weight.Zero).SetEndWeight(Weight.One);
            Assert.True(f.GetAutomaton().IsZero());
        }
        
        #endregion

        #region Arithmetic with loops

        /// <summary>
        /// Tests arithmetic operations on automata with non-trivial loops.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void LoopyArithmetic()
        {
            var builder1 = new StringAutomaton.Builder();
            builder1.Start
                .AddTransition('a', Weight.FromValue(4.0))
                .AddTransition('b', Weight.One, builder1.Start.Index)
                .SetEndWeight(Weight.One);
            var automaton1 = builder1.GetAutomaton();

            var builder2 = new StringAutomaton.Builder();
            builder2.Start
                .AddSelfTransition('a', Weight.FromValue(2))
                .AddSelfTransition('b', Weight.FromValue(3))
                .SetEndWeight(Weight.One);
            var automaton2 = builder2.GetAutomaton();

            StringAutomaton sum = automaton1.Sum(automaton2);
            StringInferenceTestUtilities.TestValue(sum, 2.0, string.Empty);
            StringInferenceTestUtilities.TestValue(sum, 4.0 + 6.0, "ab");
            StringInferenceTestUtilities.TestValue(sum, 16.0 + 36.0, "abab");
            StringInferenceTestUtilities.TestValue(sum, 12.0, "aab", "aba", "baa");
            StringInferenceTestUtilities.TestValue(sum, 18.0, "abb", "bab", "bba");
            StringInferenceTestUtilities.TestValue(sum, 8.0, "aaa");
            StringInferenceTestUtilities.TestValue(sum, 27.0, "bbb");

            StringAutomaton product = automaton1.Product(automaton2);
            StringInferenceTestUtilities.TestValue(product, 1.0, string.Empty);
            StringInferenceTestUtilities.TestValue(product, 4 * 6, "ab");
            StringInferenceTestUtilities.TestValue(product, 16 * 36, "abab");
            StringInferenceTestUtilities.TestValue(product, 0.0, "aba", "bbb", "a", "b");

            product.SetToProduct(product, product);
            StringInferenceTestUtilities.TestValue(product, 1.0, string.Empty);
            StringInferenceTestUtilities.TestValue(product, 4 * 4 * 6 * 6, "ab");
            StringInferenceTestUtilities.TestValue(product, 16 * 16 * 36 * 36, "abab");
            StringInferenceTestUtilities.TestValue(product, 0.0, "aba", "bbb", "a", "b");
        }

        #endregion

        #region Normalization

        /// <summary>
        /// Tests computing the automaton normalizer.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerSimple1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddSelfTransition('a', Weight.FromValue(0.7));
            builder.Start.SetEndWeight(Weight.FromValue(0.3));
            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);
            Assert.Equal(0.0, automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests computing the automaton normalizer.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerSimple2()
        {
            var builder = new StringAutomaton.Builder();
            var state = builder.Start;
            state.SetEndWeight(Weight.FromValue(0.1));
            state.AddSelfTransition('a', Weight.FromValue(0.7));
            state = state.AddTransition('b', Weight.FromValue(0.2));
            state.AddSelfTransition('a', Weight.FromValue(0.4));
            state.SetEndWeight(Weight.FromValue(0.6));
            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);
            Assert.Equal(0.0, automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests computing the automaton normalizer.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerSimple3()
        {
            var builder = new StringAutomaton.Builder();

            builder.Start.AddSelfTransition('a', Weight.FromValue(0.7));
            builder.Start.SetEndWeight(Weight.FromValue(0.1));

            var state1 = builder.Start.AddTransition('b', Weight.FromValue(0.15));
            state1.AddSelfTransition('a', Weight.FromValue(0.4));
            state1.SetEndWeight(Weight.FromValue(0.6));

            var state2 = builder.Start.AddTransition('c', Weight.FromValue(0.05));
            state2.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);
            Assert.Equal(0.0, automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests computing the automaton normalizer in the presence of a non-trivial loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerWithNonTrivialLoop1()
        {
            var builder = new StringAutomaton.Builder();

            var state = builder.Start.AddTransition('a', Weight.FromValue(0.9));
            state.AddTransition('a', Weight.FromValue(0.1)).SetEndWeight(Weight.One);
            state = state.AddTransition('a', Weight.FromValue(0.9));
            state.AddTransition('a', Weight.FromValue(0.1)).SetEndWeight(Weight.One);
            state = state.AddTransition('a', Weight.FromValue(0.9), builder.Start.Index);
            state.AddTransition('a', Weight.FromValue(0.1)).SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);
            Assert.Equal(0.0, automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests computing the automaton normalizer in the presence of a non-trivial loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerWithNonTrivialLoop2()
        {
            var builder = new StringAutomaton.Builder();

            var endState = builder.Start.AddTransition('a', Weight.FromValue(2.0));
            endState.SetEndWeight(Weight.FromValue(5.0));
            endState.AddTransition('b', Weight.FromValue(0.25), builder.Start.Index);
            endState.AddTransition('c', Weight.FromValue(0.2), builder.Start.Index);

            var automaton = builder.GetAutomaton();

            Assert.Equal(Math.Log(100.0), automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(Math.Log(100.0), GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(Math.Log(100.0), GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests normalizing an automaton in the presence of a non-trivial loop.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeValuesWithNonTrivialLoop()
        {
            var builder = new StringAutomaton.Builder();

            var endState = builder.Start.AddTransition('a', Weight.FromValue(2.0));
            endState.SetEndWeight(Weight.FromValue(5.0));
            endState.AddTransition('b', Weight.FromValue(0.1), builder.Start.Index);
            endState.AddTransition('c', Weight.FromValue(0.05), builder.Start.Index);
            endState.AddSelfTransition('!', Weight.FromValue(0.5));

            var automaton = builder.GetAutomaton();

            var normalizedAutomaton = automaton.Clone();
            double logNormalizer = normalizedAutomaton.NormalizeValues();
            
            Assert.Equal(Math.Log(50.0), logNormalizer, 1e-6);
            Assert.Equal(Math.Log(50.0), GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(Math.Log(50.0), GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
            AssertStochastic(normalizedAutomaton);
            foreach (var str in new[] { "a!!", "abaca", "a!ba!!ca!!!!" })
            {
                Assert.False(double.IsNegativeInfinity(automaton.GetLogValue(str)));
                Assert.Equal(automaton.GetLogValue(str), normalizedAutomaton.GetLogValue(str) + logNormalizer, 1e-6);
            }
        }

        /// <summary>
        /// Tests computing the automaton normalizer in the presence of several non-trivial loops.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerWithManyNonTrivialLoops1()
        {
            var builder = new StringAutomaton.Builder();

            AddEpsilonLoop(builder.Start, 3, 0.2);
            AddEpsilonLoop(builder.Start, 5, 0.3);
            builder.Start.SetEndWeight(Weight.FromValue(0.1));
            var nextState = builder.Start.AddTransition('a', Weight.FromValue(0.4));
            nextState.SetEndWeight(Weight.FromValue(0.6));
            AddEpsilonLoop(nextState, 0, 0.3);
            nextState = nextState.AddTransition('b', Weight.FromValue(0.1));
            AddEpsilonLoop(nextState, 1, 0.9);
            nextState.SetEndWeight(Weight.FromValue(0.1));

            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);
            Assert.Equal(0.0, automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests computing the automaton normalizer in the presence of several non-trivial loops.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ComputeNormalizerWithManyNonTrivialLoops2()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(6);

            builder[0].AddEpsilonTransition(Weight.FromValue(0.2), 1);
            builder[0].AddEpsilonTransition(Weight.FromValue(0.5), 3);
            builder[0].SetEndWeight(Weight.FromValue(0.3));
            builder[1].AddEpsilonTransition(Weight.FromValue(0.8), 0);
            builder[1].AddEpsilonTransition(Weight.FromValue(0.1), 2);
            builder[1].SetEndWeight(Weight.FromValue(0.1));
            builder[2].SetEndWeight(Weight.FromValue(1.0));
            builder[3].AddEpsilonTransition(Weight.FromValue(0.2), 4);
            builder[3].AddEpsilonTransition(Weight.FromValue(0.1), 5);
            builder[3].SetEndWeight(Weight.FromValue(0.7));
            builder[4].AddEpsilonTransition(Weight.FromValue(0.5), 2);
            builder[4].AddEpsilonTransition(Weight.FromValue(0.5), 6);
            builder[4].SetEndWeight(Weight.FromValue(0.0));
            builder[5].AddEpsilonTransition(Weight.FromValue(0.1), 3);
            builder[5].AddEpsilonTransition(Weight.FromValue(0.9), 6);
            builder[5].SetEndWeight(Weight.Zero);
            builder[6].SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);
            Assert.Equal(0.0, automaton.GetLogNormalizer(), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValue(automaton), 1e-6);
            Assert.Equal(0.0, GetLogNormalizerByGetValueWithTransducers(automaton), 1e-6);
        }

        /// <summary>
        /// Tests the behavior of automaton normalization on a non-normalizable automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NonNormalizableLoop1()
        {
            var builder = new StringAutomaton.Builder();

            var endState = builder.Start.AddTransition('a', Weight.FromValue(3.5));
            endState.SetEndWeight(Weight.FromValue(5.0));
            endState.AddTransition('b', Weight.FromValue(0.1), builder.Start.Index);
            endState.AddTransition('c', Weight.FromValue(0.05), builder.Start.Index);
            endState.AddSelfTransition('!', Weight.FromValue(0.5));

            var automaton = builder.GetAutomaton();

            StringAutomaton copyOfAutomaton = automaton.Clone();
            Assert.Throws<InvalidOperationException>(() => copyOfAutomaton.NormalizeValues());
            Assert.False(copyOfAutomaton.TryNormalizeValues());
            ////Assert.Equal(f, copyOfF); // TODO: fix equality first
        }

        /// <summary>
        /// Tests the behavior of automaton normalization on a non-normalizable automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NonNormalizableLoop2()
        {
            var builder = new StringAutomaton.Builder();

            var endState = builder.Start.AddTransition('a', Weight.FromValue(2.0));
            endState.SetEndWeight(Weight.FromValue(5.0));
            endState.AddTransition('b', Weight.FromValue(0.1), builder.Start.Index);
            endState.AddTransition('c', Weight.FromValue(0.05), builder.Start.Index);
            endState.AddSelfTransition('!', Weight.FromValue(0.75));

            var automaton = builder.GetAutomaton();

            StringAutomaton copyOfAutomaton = automaton.Clone();
            Assert.Throws<InvalidOperationException>(() => copyOfAutomaton.NormalizeValues());
            Assert.False(copyOfAutomaton.TryNormalizeValues());
            ////Assert.Equal(f, copyOfF); // TODO: fix equality first
        }

        /// <summary>
        /// Tests the behavior of automaton normalization on a non-normalizable automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NonNormalizableLoop3()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.FromValue(2.0), builder.Start.Index);
            builder.Start.SetEndWeight(Weight.FromValue(5.0));

            var automaton = builder.GetAutomaton();

            StringAutomaton copyOfAutomaton = automaton.Clone();
            Assert.Throws<InvalidOperationException>(() => automaton.NormalizeValues());
            Assert.False(copyOfAutomaton.TryNormalizeValues());
            ////Assert.Equal(f, copyOfF); // TODO: fix equality first
        }

        /// <summary>
        /// Tests the behavior of automaton normalization on a non-normalizable automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NonNormalizableLoop4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddSelfTransition('a', Weight.FromValue(0.1));
            var branch1 = builder.Start.AddTransition('a', Weight.FromValue(2.0));
            branch1.AddSelfTransition('a', Weight.FromValue(2.0));
            branch1.SetEndWeight(Weight.One);
            var branch2 = builder.Start.AddTransition('a', Weight.FromValue(2.0));
            branch2.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            StringAutomaton copyOfAutomaton = automaton.Clone();
            Assert.Throws<InvalidOperationException>(() => automaton.NormalizeValues());
            Assert.False(copyOfAutomaton.TryNormalizeValues());
            ////Assert.Equal(f, copyOfF); // TODO: fix equality first
        }

        /// <summary>
        /// Tests the normalization of an automaton with infinite values.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeWithInfiniteEpsilon1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddSelfTransition(Option.None, Weight.FromValue(3)).SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            // The automaton takes an infinite value on "a", and yet the normalization must work
            Assert.True(automaton.TryNormalizeValues());
            StringInferenceTestUtilities.TestValue(automaton, 1, "a");
            StringInferenceTestUtilities.TestValue(automaton, 0, "b");
        }

        /// <summary>
        /// Tests the normalization of an automaton with infinite values.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeWithInfiniteEpsilon2()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddSelfTransition(Option.None, Weight.FromValue(2)).SetEndWeight(Weight.One);
            builder.Start.AddTransition('b', Weight.One).AddSelfTransition(Option.None, Weight.FromValue(1)).SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            // "a" branch infinitely dominates over the "b" branch
            Assert.True(automaton.TryNormalizeValues());
            StringInferenceTestUtilities.TestValue(automaton, 1, "a");
            Assert.True(automaton.GetValue("b") < 1e-50);
        }

        #endregion

        #region Epsilon closure

        /// <summary>
        /// Tests computing the epsilon closure of an automaton with non-trivial loops.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void LoopyEpsilonClosure1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddEpsilonTransition(Weight.FromValue(0.5), builder.Start.Index);
            var nextState = builder.Start.AddEpsilonTransition(Weight.FromValue(0.4));
            nextState.AddEpsilonTransition(Weight.One).AddEpsilonTransition(Weight.One, builder.Start.Index);
            builder.Start.SetEndWeight(Weight.FromValue(0.1));

            var automaton = builder.GetAutomaton();

            AssertStochastic(automaton);

            StringAutomaton.EpsilonClosure startClosure = new Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>.EpsilonClosure(automaton, automaton.Start);
            Assert.Equal(3, startClosure.Size);
            Assert.Equal(0.0, startClosure.EndWeight.LogValue, 1e-8);

            for (int i = 0; i < startClosure.Size; ++i)
            {
                Weight weight = startClosure.GetStateWeightByIndex(i);
                var state = startClosure.GetStateByIndex(i);
                double expectedWeight = state.Index == automaton.Start.Index ? 10 : 4;
                Assert.Equal(expectedWeight, weight.Value, 1e-8);
            }
        }

        #endregion

        #region ToString with loops

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToStringWithLoops1()
        {
            var builder = new StringAutomaton.Builder();
            var middleNode = builder.Start.AddTransition('a', Weight.One);
            middleNode.AddTransitionsForSequence("bbb", builder.Start.Index);
            middleNode.AddTransition('c', Weight.One, builder.Start.Index);
            builder.Start.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.Equal("(a(c|bbb))*", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("(a(c|bbb))*?", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToStringWithLoops2()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddSelfTransition('a', Weight.One);
            builder.Start.AddSelfTransition('b', Weight.One);
            builder.Start.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.Equal("(a|b)*", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("(a|b)*?", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")] // TODO: extract longest common suffix when 'or'-ing concatenations
        [Trait("Category", "StringInference")]
        public void ConvertToStringWithLoops3()
        {
            var builder = new StringAutomaton.Builder();
            var state = builder.Start.AddTransition('x', Weight.One);
            builder.Start.AddTransition('y', Weight.One, state.Index);
            state.AddSelfTransition('a', Weight.One);
            state.AddSelfTransition('b', Weight.One);
            state.SetEndWeight(Weight.One);
            state.AddTransitionsForSequence("zzz").SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.Equal("(x|y)(a|b)*[zzz]", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("(x|y)(a|b)*(|zzz)", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToStringWithLoops4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransitionsForSequence("xyz", builder.Start.Index);
            builder.Start.AddTransition('!', Weight.One).SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            Assert.Equal("(xyz)*!", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("(xyz)*?!", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToStringWithDeadTransitions1()
        {
            var builder = new StringAutomaton.Builder();
            var state = builder.Start.AddTransition('x', Weight.One);
            builder.Start.AddTransition('y', Weight.Zero, state.Index);
            state.AddSelfTransition('a', Weight.One);
            state.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.Equal("xa*", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("xa*?", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToStringWithDeadTransitions2()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddSelfTransition('x', Weight.Zero);
            builder.Start.AddTransition('y', Weight.Zero).SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.Equal("Ø", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("Ø", automaton.ToString(AutomatonFormats.Regexp));
        }
        
        #endregion

        #region Helpers

        /// <summary>
        /// Computes the normalizer of an automaton by replacing every transition with an epsilon transition
        /// and computing the value of the automaton on an empty sequence.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <returns>The logarithm of the normalizer.</returns>
        private static double GetLogNormalizerByGetValue(StringAutomaton automaton)
        {
            var epsilonAutomaton = new StringAutomaton();
            epsilonAutomaton.SetToFunction(automaton, (dist, weight, group) => ValueTuple.Create<Option<DiscreteChar>, Weight>(Option.None, weight)); // Convert all the edges to epsilon edges
            return epsilonAutomaton.GetLogValue(string.Empty); // Now this will be exactly the normalizer
        }

        /// <summary>
        /// Computes the normalizer via transducers.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <returns>The logarithm of the normalizer.</returns>
        private static double GetLogNormalizerByGetValueWithTransducers(StringAutomaton automaton)
        {
            var one = StringAutomaton.Constant(1.0);
            StringTransducer transducer = StringTransducer.Consume(automaton);
            transducer.AppendInPlace(StringTransducer.Produce(one));
            return transducer.ProjectSource(one).GetLogValue("an arbitrary string"); // Now this will be exactly the normalizer
        }

        /// <summary>
        /// Tests if the weights of all outgoing transitions sum to one for each state of a given automaton.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        private static void AssertStochastic(StringAutomaton automaton)
        {
            StringAutomaton automatonClone = automaton.Clone();
            automatonClone.RemoveDeadStates();

            for (int i = 0; i < automatonClone.States.Count; ++i)
            {
                Weight weightSum = automatonClone.States[i].EndWeight;
                for (int j = 0; j < automatonClone.States[i].Transitions.Count; ++j)
                {
                    weightSum += automatonClone.States[i].Transitions[j].Weight;
                }

                Assert.Equal(0.0, weightSum.LogValue, 1e-6);
            }
        }

        /// <summary>
        /// Adds an epsilon-loop with a specified number of intermediate states to a given state.
        /// </summary>
        /// <param name="state">The state.</param>
        /// <param name="loopSize">The number of intermediate states in the loop.</param>
        /// <param name="loopWeight">The weight of the loop.</param>
        private static void AddEpsilonLoop(StringAutomaton.Builder.StateBuilder state, int loopSize, double loopWeight)
        {
            var currentState = state;
            for (int i = 0; i <= loopSize; ++i)
            {
                currentState = currentState.AddEpsilonTransition(
                    i == 0 ? Weight.FromValue(loopWeight) : Weight.One,
                    i == loopSize ? (int?)state.Index : null);
            }
        }

        #endregion
    }
}
