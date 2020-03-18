// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text.RegularExpressions;

    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Tests for weighted finite state automata.
    /// </summary>
    public class AutomatonTests
    {
        /// <summary>
        /// Tests cloning an automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Clone()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();
            var clone = automaton.Clone();

            Assert.Equal(automaton, clone);
        }

        /// <summary>
        /// Tests the behavior of an automaton that is zero everywhere.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Zero()
        {
            StringAutomaton zero1 = StringAutomaton.Zero();
            Assert.True(zero1.IsZero());
            Assert.True(zero1.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero1, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero2 = StringAutomaton.ConstantOn(0.0, "a", "bcd");
            Assert.True(zero2.IsZero());
            Assert.True(zero2.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero2, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero3 = StringAutomaton.Constant(0.0);
            Assert.True(zero3.IsZero());
            Assert.True(zero3.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero3, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero4 =
                StringAutomaton
                    .Constant(2.0, DiscreteChar.Lower())
                    .Product(
                        StringAutomaton
                            .Constant(3.0, DiscreteChar.Upper())
                            .Append(StringAutomaton.ConstantOnElement(1.5, DiscreteChar.Digit())));
            Assert.True(zero4.IsZero());
            Assert.True(zero4.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero4, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero5 = StringAutomaton.WeightedSum(
                1.2, StringAutomaton.Zero(), 0.0, StringAutomaton.ConstantOn(1.3, "abc"));
            Assert.True(zero5.IsZero());
            Assert.True(zero5.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero5, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero6 = StringAutomaton.ConstantOn(1.0, "a", "b").Append(StringAutomaton.Constant(0.0));
            Assert.True(zero6.IsZero());
            Assert.True(zero6.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero6, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero7 = StringAutomaton.Constant(0.0).Append(StringAutomaton.Constant(1.0));
            Assert.True(zero7.IsZero());
            Assert.True(zero7.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero7, 0.0, "abc", "ab", "a", string.Empty);

            var zero8Builder = new StringAutomaton.Builder();
            zero8Builder.Start.AddTransition('a', Weight.One);
            var zero8 = zero8Builder.GetAutomaton();
            Assert.True(zero8.IsZero());
            Assert.False(zero8.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero8, 0.0, "abc", "ab", "a", string.Empty);

            var zero9Builder = new StringAutomaton.Builder();
            zero9Builder.Start.AddTransition('a', Weight.One);
            zero9Builder.Start.AddTransition('b', Weight.One, zero9Builder.StartStateIndex);
            zero9Builder.Start.AddTransitionsForSequence("abc");
            var zero9 = zero9Builder.GetAutomaton();
            Assert.True(zero9.IsZero());
            Assert.False(zero9.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero9, 0.0, "abc", "ab", "a", string.Empty);

            StringAutomaton zero10 = StringAutomaton.Constant(1.0);
            zero10.SetToZero();
            Assert.True(zero10.IsZero());
            Assert.True(zero10.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(zero10, 0.0, "abc", "ab", "a", string.Empty);
        }

        /// <summary>
        /// Tests the behavior of automata that are constant and have infinite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Constant()
        {
            StringAutomaton constant1 = StringAutomaton.Constant(2.0);
            Assert.False(constant1.IsZero());
            StringInferenceTestUtilities.TestValue(constant1, 2.0, string.Empty, "a", "aejdfiejmbcr");

            StringAutomaton constant2 = StringAutomaton.Constant(3.0, DiscreteChar.Digit());
            Assert.False(constant2.IsZero());
            StringInferenceTestUtilities.TestValue(constant2, 3.0, string.Empty, "12", "999666999");
            StringInferenceTestUtilities.TestValue(constant2, 0.0, "a", "1a", "a23", "232c34fr4");
        }

        /// <summary>
        /// Tests the behavior of automata that are constant and have finite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConstantOn()
        {
            StringAutomaton constant1 = StringAutomaton.ConstantOn(3.0, "a", "abc");
            Assert.False(constant1.IsZero());
            StringInferenceTestUtilities.TestValue(constant1, 3.0, "a");
            StringInferenceTestUtilities.TestValue(constant1, 0.0, "ab", string.Empty);

            StringAutomaton constant2 = StringAutomaton.ConstantOnElement(3.0, 'a');
            Assert.False(constant2.IsZero());
            StringInferenceTestUtilities.TestValue(constant2, 3.0, "a");
            StringInferenceTestUtilities.TestValue(constant2, 0.0, "abc", "ab", string.Empty);

            StringAutomaton constant3 = StringAutomaton.ConstantOn(3.0, "a");
            Assert.False(constant3.IsZero());
            StringInferenceTestUtilities.TestValue(constant3, 3.0, "a");
            StringInferenceTestUtilities.TestValue(constant3, 0.0, "abc", "ab", string.Empty);

            StringAutomaton constant4 = StringAutomaton.ConstantOnElement(3.0, DiscreteChar.Digit());
            Assert.False(constant4.IsZero());
            StringInferenceTestUtilities.TestValue(constant4, 3.0, "1", "2", "0", "9");
            StringInferenceTestUtilities.TestValue(constant4, 0.0, "11", "99", "abc", "ab", string.Empty);
        }

        /// <summary>
        /// Tests summing automata.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Sum()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(1.0, "a", "abc");
            StringAutomaton automaton2 = StringAutomaton.ConstantOn(1.0, "a", "bcd");
            StringAutomaton automaton3 = StringAutomaton.Constant(1.0);
            StringAutomaton automaton4 = StringAutomaton.Zero();
            StringAutomaton sum = StringAutomaton.Sum(automaton1, automaton2, automaton3, automaton4);
            Assert.False(sum.IsZero());
            StringInferenceTestUtilities.TestValue(sum, 3, "a");
            StringInferenceTestUtilities.TestValue(sum, 2, "abc", "bcd");
            StringInferenceTestUtilities.TestValue(sum, 1, "ab", string.Empty);
        }

        /// <summary>
        /// Tests summing automata that are constant and have infinite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SumOfConstants()
        {
            StringAutomaton automaton1 = StringAutomaton.Constant(1.0, DiscreteChar.Lower());
            StringAutomaton automaton2 = StringAutomaton.Constant(1.0, DiscreteChar.Upper());
            StringAutomaton sum = automaton1.Sum(automaton2);
            Assert.False(sum.IsZero());
            StringInferenceTestUtilities.TestValue(sum, 1.0, "a", "abc");
            StringInferenceTestUtilities.TestValue(sum, 1.0, "A", "ABC");
            StringInferenceTestUtilities.TestValue(sum, 2.0, string.Empty);
            StringInferenceTestUtilities.TestValue(sum, 0.0, "abcABC", "ABCabc");
        }

        /// <summary>
        /// Tests the weighted sum of automata.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void WeightedSum()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(0.5, "a", "abc");
            StringAutomaton automaton2 = StringAutomaton.ConstantOn(1.5, "a", "bcd");
            StringAutomaton weightedSum = StringAutomaton.WeightedSum(2.0, automaton1, 3.0, automaton2);
            Assert.False(weightedSum.IsZero());
            StringInferenceTestUtilities.TestValue(weightedSum, 5.5, "a");
            StringInferenceTestUtilities.TestValue(weightedSum, 1.0, "abc");
            StringInferenceTestUtilities.TestValue(weightedSum, 4.5, "bcd");
            StringInferenceTestUtilities.TestValue(weightedSum, 0.0, "ab", string.Empty);

            weightedSum.SetToSum(1.0, weightedSum, 1.0, automaton2);
            Assert.False(weightedSum.IsZero());
            StringInferenceTestUtilities.TestValue(weightedSum, 7.0, "a");
            StringInferenceTestUtilities.TestValue(weightedSum, 1.0, "abc");
            StringInferenceTestUtilities.TestValue(weightedSum, 6.0, "bcd");
            StringInferenceTestUtilities.TestValue(weightedSum, 0.0, "ab", string.Empty);
        }

        /// <summary>
        /// Tests the weighted sum of automata when one of the arguments is zero everywhere.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void WeightedSumWithZero()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(1.0, "a", "abc");
            StringAutomaton automaton2 = StringAutomaton.Zero();

            StringAutomaton weightedSum1 = StringAutomaton.WeightedSum(2.0, automaton1, 3.0, automaton2);
            Assert.False(weightedSum1.IsZero());
            StringInferenceTestUtilities.TestValue(weightedSum1, 2.0, "a", "abc");
            StringInferenceTestUtilities.TestValue(weightedSum1, 0.0, "bcd", "ab", string.Empty);

            StringAutomaton weightedSum2 = StringAutomaton.WeightedSum(3.0, automaton2, 2.0, automaton1);
            Assert.False(weightedSum2.IsZero());
            StringInferenceTestUtilities.TestValue(weightedSum2, 2.0, "a", "abc");
            StringInferenceTestUtilities.TestValue(weightedSum2, 0.0, "bcd", "ab", string.Empty);
        }

        /// <summary>
        /// Tests automaton scaling.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Scale()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "abc");
            StringAutomaton scaledAutomaton = automaton.Scale(2.0);
            Assert.False(scaledAutomaton.IsZero());
            StringInferenceTestUtilities.TestValue(scaledAutomaton, 4.0, "a");
            StringInferenceTestUtilities.TestValue(scaledAutomaton, 4.0, "abc");
            StringInferenceTestUtilities.TestValue(scaledAutomaton, 0.0, "bcd", "ab", string.Empty);
        }

        /// <summary>
        /// Tests automaton scaling when the scale is zero.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ZeroScale()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "abc");
            StringAutomaton scaledAutomaton = automaton.Scale(0.0);
            Assert.True(scaledAutomaton.IsZero());
            Assert.True(scaledAutomaton.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(scaledAutomaton, 0.0, "a", "abc", string.Empty);
        }

        /// <summary>
        /// Tests creating an automaton from a dictionary mapping sequences to values.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void FromValues()
        {
            var values = new Dictionary<string, double> { { "abc", 3.0 }, { "a", 1.0 }, { "bc", 2.0 } };
            StringAutomaton automaton = StringAutomaton.FromValues(values);
            StringInferenceTestUtilities.TestValue(automaton, 3.0, "abc");
            StringInferenceTestUtilities.TestValue(automaton, 1.0, "a");
            StringInferenceTestUtilities.TestValue(automaton, 2.0, "bc");
            StringInferenceTestUtilities.TestValue(automaton, 0.0, "ab", "b", "c", string.Empty);
        }

        /// <summary>
        /// Tests automata product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product1()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(2.0, "a", "abc", "d");
            automaton1 = automaton1.Sum(StringAutomaton.ConstantOn(1.0, "a"));
            StringAutomaton automaton2 = StringAutomaton.ConstantOn(3.0, "a", "abc");
            StringAutomaton product = automaton1.Product(automaton2);
            Assert.False(product.IsZero());
            StringInferenceTestUtilities.TestValue(product, 9.0, "a");
            StringInferenceTestUtilities.TestValue(product, 6.0, "abc");
            StringInferenceTestUtilities.TestValue(product, 0.0, "d", string.Empty);

            product.SetToProduct(product, product);
            Assert.False(product.IsZero());
            StringInferenceTestUtilities.TestValue(product, 81.0, "a");
            StringInferenceTestUtilities.TestValue(product, 36.0, "abc");
            StringInferenceTestUtilities.TestValue(product, 0.0, "d", string.Empty);
        }

        /// <summary>
        /// Tests automata product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product2()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "a", "b").Scale(2.0);
            automaton.SetToProduct(automaton, automaton);
            Assert.False(automaton.IsZero());
            StringInferenceTestUtilities.TestValue(automaton, 4.0, "a", "b");
            StringInferenceTestUtilities.TestValue(automaton, 0.0, "ab", "aa", "bb", string.Empty);
        }

        /// <summary>
        /// Tests automata product when one of the arguments is constant and has infinite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProductWithConstant()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(2.0, "a", "abc", "d");
            automaton1 = automaton1.Sum(StringAutomaton.ConstantOn(1.0, "a"));
            StringAutomaton automaton2 = StringAutomaton.Constant(3.0);
            StringAutomaton product = automaton1.Product(automaton2);
            Assert.False(product.IsZero());
            StringInferenceTestUtilities.TestValue(product, 9.0, "a");
            StringInferenceTestUtilities.TestValue(product, 6.0, "abc", "d");
            StringInferenceTestUtilities.TestValue(product, 0.0, string.Empty, "ab", "ad", "abcd");
        }

        /// <summary>
        /// Tests product of two constant automata with infinite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConstantTimesConstant()
        {
            StringAutomaton automaton1 = StringAutomaton.Constant(2.0);
            StringAutomaton automaton2 = StringAutomaton.Constant(3.0);
            StringAutomaton product = automaton1.Product(automaton2);
            Assert.False(product.IsZero());
            StringInferenceTestUtilities.TestValue(product, 6.0, "a", "abc", string.Empty);
        }

        /// <summary>
        /// Tests product of two automata with non-intersecting support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ZeroProduct()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(2.0, "a", "abc", "d");
            StringAutomaton automaton2 = StringAutomaton.ConstantOn(3.0, "ab", "da");
            StringAutomaton product = automaton1.Product(automaton2);
            Assert.True(product.IsZero());
            Assert.True(product.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(product, 0.0, "a", "ab", "abc", "d", "da", string.Empty);
        }

        /// <summary>
        /// Tests automata product when one of the arguments is zero.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProductWithZero()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "abc", "d");

            StringAutomaton product1 = automaton.Product(StringAutomaton.Zero());
            Assert.True(product1.IsZero());
            Assert.True(product1.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(product1, 0.0, "a", "abc", "d", string.Empty);

            StringAutomaton product2 = StringAutomaton.Zero().Product(automaton);
            Assert.True(product2.IsZero());
            Assert.True(product2.IsCanonicZero());
            StringInferenceTestUtilities.TestValue(product2, 0.0, "a", "abc", "d", string.Empty);
        }

        /// <summary>
        /// Tests that a product of a pair of automata has no redundant branches that don't contribute into the automaton value.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProductNoDeadBranches()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddTransition('b', Weight.One).SetEndWeight(Weight.One);
            builder.Start.AddTransition('a', Weight.One).AddTransition('c', Weight.One).SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            StringAutomaton automatonSqr = automaton.Product(automaton);
            Assert.Equal(4, automatonSqr.States.Count);
        }

        /// <summary>
        /// Tests automaton appending, also known as the Cauchy product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Append()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(2.0, "a", "ab", "c");
            StringAutomaton automaton2 = StringAutomaton.ConstantOn(3.0, "b", string.Empty, "c");
            StringAutomaton concat = automaton1.Append(automaton2);
            Assert.False(concat.IsZero());
            StringInferenceTestUtilities.TestValue(concat, 12.0, "ab");
            StringInferenceTestUtilities.TestValue(concat, 6.0, "a", "ac", "abb", "abc", "cb", "c", "cc");
            StringInferenceTestUtilities.TestValue(concat, 0.0, "b", string.Empty);

            concat.AppendInPlace(concat);
            Assert.False(concat.IsZero());
            StringInferenceTestUtilities.TestValue(concat, 72.0, "aba", "abc", "ababb", "acc");
            StringInferenceTestUtilities.TestValue(concat, 108.0, "abcc");
            StringInferenceTestUtilities.TestValue(concat, 36.0, "aa", "aac");
            StringInferenceTestUtilities.TestValue(concat, 0.0, "b", string.Empty);
        }

        /// <summary>
        /// Tests appending to an automaton with a singleton support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AppendToPoint()
        {
            StringAutomaton automaton1 = StringAutomaton.ConstantOn(2.0, "a");
            StringAutomaton automaton2 = StringAutomaton.ConstantOn(3.0, "b", string.Empty);
            StringAutomaton concat = automaton1.Append(automaton2);
            Assert.False(concat.IsZero());
            StringInferenceTestUtilities.TestValue(concat, 6.0, "ab", "a");
            StringInferenceTestUtilities.TestValue(concat, 0.0, "b", string.Empty);
        }

        /// <summary>
        /// Tests automaton reversal on an automaton with finite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Reverse1()
        {
            StringAutomaton automaton = StringAutomaton.FromValues(new Dictionary<string, double> { { "abc", 1.0 }, { "abd", 2.0 }, { "bc", 3.0 } });
            StringAutomaton expectedReverse = StringAutomaton.FromValues(new Dictionary<string, double> { { "cba", 1.0 }, { "dba", 2.0 }, { "cb", 3.0 } });
            
            StringAutomaton reverse1 = automaton.Reverse();
            Assert.Equal(expectedReverse, reverse1);

            bool determinized = automaton.TryDeterminize();
            Assert.True(determinized);
            StringAutomaton reverse2 = automaton.Reverse();
            Assert.Equal(expectedReverse, reverse2);
        }

        /// <summary>
        /// Tests automaton reversal on an automaton with infinite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Reverse2()
        {
            var builder = new StringAutomaton.Builder();
            var otherState = builder.Start.AddSelfTransition('a', Weight.FromValue(0.5)).AddTransition('b', Weight.FromValue(0.7));
            builder.Start.SetEndWeight(Weight.FromValue(0.3));
            otherState.SetEndWeight(Weight.FromValue(0.8));
            var automaton = builder.GetAutomaton();

            StringAutomaton reverse = automaton.Reverse();
            StringInferenceTestUtilities.TestValue(reverse, 0.7 * 0.8, "b");
            StringInferenceTestUtilities.TestValue(reverse, 0.5 * 0.5 * 0.7 * 0.8, "baa");
            StringInferenceTestUtilities.TestValue(reverse, 0.5 * 0.5 * 0.5 * 0.3, "aaa");
            StringInferenceTestUtilities.TestValue(reverse, 0.3, string.Empty);
            StringInferenceTestUtilities.TestValue(reverse, 0.0, "bb", "ab", "baab");
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat1()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "ab", "cd");
            automaton = StringAutomaton.Repeat(automaton, 1, 3);
            StringInferenceTestUtilities.TestValue(automaton, 0.0, string.Empty, "aa", "cb", "abababab", "abcdabcd");
            StringInferenceTestUtilities.TestValue(automaton, 2.0, "ab", "cd");
            StringInferenceTestUtilities.TestValue(automaton, 4.0, "abab", "abcd", "cdab", "cdcd");
            StringInferenceTestUtilities.TestValue(automaton, 8.0, "ababab", "cdcdcd", "cdabcd", "cdcdab");
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat2()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "aa");
            automaton = StringAutomaton.Repeat(automaton, 1, 2);
            StringInferenceTestUtilities.TestValue(automaton, 0.0, string.Empty, "aaaaa", "ab");
            StringInferenceTestUtilities.TestValue(automaton, 2.0, "a");
            StringInferenceTestUtilities.TestValue(automaton, 6.0, "aa");
            StringInferenceTestUtilities.TestValue(automaton, 8.0, "aaa");
            StringInferenceTestUtilities.TestValue(automaton, 4.0, "aaaa");
        }
            
        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat3()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", string.Empty);
            automaton = StringAutomaton.Repeat(automaton, 1, 3);
            StringInferenceTestUtilities.TestValue(automaton, 0.0, "aaaa", "ab");
            StringInferenceTestUtilities.TestValue(automaton, 14.0, string.Empty); // 'eps', 'eps eps', 'eps eps eps'
            StringInferenceTestUtilities.TestValue(automaton, 34.0, "a"); // 'a', 'eps a', 'a eps', 'eps eps a', 'eps a eps', 'a eps eps'
            StringInferenceTestUtilities.TestValue(automaton, 28.0, "aa"); // 'a a', 'eps a a', 'a eps a', 'a a eps'
            StringInferenceTestUtilities.TestValue(automaton, 8.0, "aaa"); // 'a a a'
        }
            
        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat4()
        {
            StringAutomaton automaton = StringAutomaton.Constant(2.0, DiscreteChar.Digit());
            automaton = StringAutomaton.Repeat(automaton, 1, 3);
            StringInferenceTestUtilities.TestValue(automaton, 0.0, "a");
            StringInferenceTestUtilities.TestValue(automaton, 14.0, string.Empty); // 'eps', 'eps eps', 'eps eps eps'
            StringInferenceTestUtilities.TestValue(automaton, 34.0, "1"); // '1', 'eps 1', '1 eps', 'eps eps 1', 'eps 1 eps', '1 eps eps'
            StringInferenceTestUtilities.TestValue(automaton, 62, "88"); // '8 8', 'eps 8 8', '8 eps 8', '8 8 eps', '88', 'eps 88', '88 eps', '88 eps eps', 'eps 88 eps', 'eps eps 88'
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat5()
        {
            StringAutomaton automaton = StringAutomaton.Constant(2.0, DiscreteChar.Digit());
            var repeatedAutomaton = StringAutomaton.Repeat(automaton, 1, 1);
            Assert.Equal(automaton, repeatedAutomaton);
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat6()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a");
            automaton = StringAutomaton.Repeat(automaton, minTimes: 3);
            
            StringInferenceTestUtilities.TestValue(automaton, 0.0, string.Empty, "a", "aa", "aab");
            StringInferenceTestUtilities.TestValue(automaton, 8.0, "aaa");
            StringInferenceTestUtilities.TestValue(automaton, 16.0, "aaaa");
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat7()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a");
            automaton = StringAutomaton.Repeat(automaton, minTimes: 0, maxTimes: 2);

            StringInferenceTestUtilities.TestValue(automaton, 1.0, string.Empty);
            StringInferenceTestUtilities.TestValue(automaton, 2.0, "a");
            StringInferenceTestUtilities.TestValue(automaton, 4.0, "aa");
            StringInferenceTestUtilities.TestValue(automaton, 0.0, "aaa", "b");
        }

        /// <summary>
        /// Tests automaton repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat8()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(2.0, "a", "b");
            automaton = StringAutomaton.Repeat(automaton, minTimes: 2, maxTimes: 2);

            StringInferenceTestUtilities.TestValue(automaton, 0.0, "a", "b", "aaa", string.Empty);
            StringInferenceTestUtilities.TestValue(automaton, 4.0, "aa", "ab", "ba", "bb");   
        }

        /// <summary>
        /// Tests automaton value normalization when the support is finite.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeValuesFinite()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "a", "bc", "d");
            double logNormalizer;
            Assert.Equal(Math.Log(3.0), automaton.GetLogNormalizer(), 1e-8);
            Assert.True(automaton.TryNormalizeValues(out logNormalizer));
            Assert.Equal(Math.Log(3.0), logNormalizer, 1e-8);
            StringInferenceTestUtilities.TestValue(automaton, 1.0 / 3.0, "a", "bc", "d");
        }

        /// <summary>
        /// Tests automaton value normalization when the automaton has a branch which doesn't contribute into its value.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeValuesDeadBranch()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransitionsForSequence("abc");
            builder.Start.AddTransitionsForSequence("def").SetEndWeight(Weight.FromValue(4.0));
            var automaton = builder.GetAutomaton();

            double logNormalizer;
            Assert.Equal(Math.Log(4.0), automaton.GetLogNormalizer(), 1e-8);
            Assert.True(automaton.TryNormalizeValues(out logNormalizer));
            Assert.Equal(Math.Log(4.0), logNormalizer, 1e-8);
            StringInferenceTestUtilities.TestValue(automaton, 1.0, "def");
            StringInferenceTestUtilities.TestValue(automaton, 0.0, "abc");
        }

        /// <summary>
        /// Tests automaton value normalization when the support is infinite.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeValuesInfinite()
        {
            const double TransitionProbability = 0.7;
            const double EndWeight = 10.0;

            var builder = new StringAutomaton.Builder();
            builder.Start.AddSelfTransition('a', Weight.FromValue(TransitionProbability));
            builder.Start.SetEndWeight(Weight.FromValue(EndWeight));
            var automaton = builder.GetAutomaton();

            double logNormalizer = automaton.GetLogNormalizer();
            Assert.Equal(Math.Log(EndWeight / (1 - TransitionProbability)), logNormalizer, 1e-8);
            Assert.Equal(logNormalizer, automaton.NormalizeValues());
            StringInferenceTestUtilities.TestValue(automaton, 1 - TransitionProbability, string.Empty);
            StringInferenceTestUtilities.TestValue(automaton, TransitionProbability * (1 - TransitionProbability), "a");
            StringInferenceTestUtilities.TestValue(automaton, TransitionProbability * TransitionProbability * (1 - TransitionProbability), "aa");
        }

        /// <summary>
        /// Tests automaton value normalization when the automaton is zero.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeValuesZero()
        {
            StringAutomaton automaton = StringAutomaton.Zero();
            Assert.False(automaton.TryNormalizeValues());
            Assert.Equal(double.NegativeInfinity, automaton.GetLogNormalizer());
        }

        /// <summary>
        /// Tests automaton value normalization behavior when the automaton cannot be normalized.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NormalizeNonNormalizable()
        {
            TestNonNormalizable(StringAutomaton.Zero(), true);
            TestNonNormalizable(StringAutomaton.Constant(1.01), false);
            TestNonNormalizable(StringAutomaton.ConstantLog(Math.Log(0.87)), false);
            TestNonNormalizable(StringAutomaton.ConstantLog(Math.Log(0.01), DiscreteChar.PointMass('a')), false);

            var builder = new StringAutomaton.Builder();
            builder.Start.AddSelfTransition('a', Weight.FromValue(1.01));
            builder.Start.SetEndWeight(Weight.One);
            TestNonNormalizable(builder.GetAutomaton(), false);
        }

        /// <summary>
        /// Tests that taking the epsilon closure of an automaton doesn't change its values.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void EpsilonClosure1()
        {
            var automaton = StringAutomaton
                .ConstantOn(1.0, "abc", "def")
                .Scale(3.0)
                .Sum(StringAutomaton.Constant(2.0, DiscreteChar.Lower()))
                .Scale(0.5);

            // Make sure it contains epsilon transitions
            Assert.Contains(automaton.States, s => s.Transitions.Any(t => t.IsEpsilon));

            // Test the original automaton, its epsilon closure and the closure of the closure
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 2.5, "abc", "def");
                StringInferenceTestUtilities.TestValue(automaton, 1.0, "abcd", "ab", string.Empty);

                automaton.MakeEpsilonFree();

                // Make sure it doesn't contain epsilon transitions
                Assert.True(automaton.States.All(s => s.Transitions.All(t => !t.IsEpsilon)));
            }
        }

        /// <summary>
        /// Tests that computing the epsilon closure doesn't have a complexity exponential in the size of the support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void EpsilonClosure2()
        {
            var builder = new StringAutomaton.Builder();
            var state = builder.Start;
            const int ChainLength = 16;
            for (int i = 0; i < ChainLength; ++i)
            {
                var nextState = state.AddEpsilonTransition(Weight.One);
                state.AddEpsilonTransition(Weight.One, nextState.Index);
                state = nextState;
            }

            state.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();
            var closure = new Automaton<string, char, DiscreteChar, StringManipulator, StringAutomaton>.EpsilonClosure(automaton, automaton.Start);
            
            Assert.Equal(ChainLength + 1, closure.Size);
            Assert.Equal(Math.Log(1 << ChainLength), closure.EndWeight.LogValue);
        }

        /// <summary>
        /// Tests computing the value of an automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GetLogValue()
        {
            const int UniformCount = 10;
            const int LetterCount = 20;

            StringAutomaton automaton = StringAutomaton.Constant(1.0);
            for (int i = 0; i < UniformCount - 1; ++i)
            {
                automaton.AppendInPlace(StringAutomaton.Constant(1.0));
            }

            var logValue = automaton.GetLogValue(new string(Enumerable.Repeat('a', LetterCount).ToArray()));
            var expectedLogValue = Math.Log(StringInferenceTestUtilities.Partitions(LetterCount, UniformCount));

            Assert.Equal(expectedLogValue, logValue, 1e-8);
        }

        /// <summary>
        /// Tests whether <see cref="StringAutomaton.TryComputePoint"/> fails due to a stack overflow
        /// when an automaton becomes sufficiently large.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void TryComputePointLargeAutomaton()
        {
            using (var unlimited = new StringAutomaton.UnlimitedStatesComputation())
            {
                const int StateCount = 100_000;

                var builder = new StringAutomaton.Builder();
                var state = builder.Start;

                for (var i = 1; i < StateCount; ++i)
                {
                    state = state.AddTransition('a', Weight.One);
                }

                state.SetEndWeight(Weight.One);

                var automaton = builder.GetAutomaton();
                var point = new string('a', StateCount - 1);

                Assert.True(automaton.TryComputePoint() == point);
                StringInferenceTestUtilities.TestValue(automaton, 1.0, point);
            }
        }

        /// <summary>
        /// Tests whether <see cref="StringAutomaton.Product"/> fails due to a stack overflow
        /// when an automaton becomes sufficiently large.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SetToProductLargeAutomaton()
        {
            using (var unlimited = new StringAutomaton.UnlimitedStatesComputation())
            {
                const int StateCount = 100_000;

                var builder = new StringAutomaton.Builder();
                var state = builder.Start;

                for (var i = 1; i < StateCount; ++i)
                {
                    state = state.AddTransition('a', Weight.One);
                }

                state.SetEndWeight(Weight.One);

                var automaton1 = builder.GetAutomaton();
                var automaton2 = builder.GetAutomaton();
                var point = new string('a', StateCount - 1);

                var productAutomaton = StringAutomaton.Product(automaton1, automaton2);
                StringInferenceTestUtilities.TestValue(productAutomaton, 1.0, point);
            }
        }

        /// <summary>
        /// Tests whether <see cref="StringAutomaton.GetLogNormalizer"/> fails due to a stack overflow
        /// when an automaton becomes sufficiently large.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GetLogNormalizerLargeAutomaton()
        {
            using (var unlimited = new StringAutomaton.UnlimitedStatesComputation())
            {
                const int StateCount = 100_000;

                var builder = new StringAutomaton.Builder();
                var state = builder.Start;

                for (var i = 1; i < StateCount; ++i)
                {
                    state = state.AddTransition('a', Weight.One);
                }

                state.SetEndWeight(Weight.One);

                var logNormalizer = builder.GetAutomaton().GetLogNormalizer();

                Assert.Equal(0.0, logNormalizer);
            }
        }

        /// <summary>
        /// Tests whether <see cref="StringAutomaton.IsZero"/> fails due to a stack overflow
        /// when an automaton becomes sufficiently large.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void IsZeroLargeAutomaton()
        {
            using (var unlimited = new StringAutomaton.UnlimitedStatesComputation())
            {
                var zeroAutomaton = MakeAutomaton(Weight.Zero);
                var nonZeroAutomaton = MakeAutomaton(Weight.One);

                Assert.True(zeroAutomaton.IsZero());
                Assert.False(nonZeroAutomaton.IsZero());
            }

            StringAutomaton MakeAutomaton(Weight endWeight)
            {
                const int StateCount = 100_000;

                var builder = new StringAutomaton.Builder();
                var state = builder.Start;

                for (var i = 1; i < StateCount; ++i)
                {
                    state = state.AddTransition('a', Weight.One);
                }

                state.SetEndWeight(endWeight);

                return builder.GetAutomaton();
            }
        }

        /// <summary>
        /// Tests whether StringAutomaton.UnlimitedStatesComputation.CheckStateCount() works as expected
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CheckStateCount()
        {
            using (var unlimited = new StringAutomaton.UnlimitedStatesComputation())
            {
                var builder = new StringAutomaton.Builder();
                var state = builder.Start;

                for (var i = 1; i < 200000; ++i)
                {
                    state = state.AddTransition('a', Weight.One);
                }

                var automaton = builder.GetAutomaton();

                // Fine, because 200k < default limit
                unlimited.CheckStateCount(automaton);

                for (var i = 1; i < 200000; ++i)
                {
                    state = state.AddTransition('a', Weight.One);
                }

                automaton = builder.GetAutomaton();

                // Not fine anymore, automaton (with 400k states) is over the default limit
                Assert.Throws<AutomatonTooLargeException>(() => unlimited.CheckStateCount(automaton));
            }
        }

        /// <summary>
        /// Tests creating an automaton from state and transition lists.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void QuotingTest()
        {
            /*
            // Should work 1
            var automaton1 = StringAutomaton.FromData(
                new StringAutomaton.DataContainer(
                    0,
                    ImmutableArray.Create(
                        new StringAutomaton.StateData(0, 1, Weight.One),
                        new StringAutomaton.StateData(1, 0, Weight.One)),
                    ImmutableArray.Create(
                        new StringAutomaton.Transition(DiscreteChar.PointMass('a'), Weight.One, 1)),
                    isEpsilonFree: true,
                    usesGroups: false,
                    isDeterminized: null,
                    isZero: null,
                    isEnumerable: null));

            StringInferenceTestUtilities.TestValue(automaton1, 1.0, string.Empty, "a");
            StringInferenceTestUtilities.TestValue(automaton1, 0.0, "b");

            // Should work 2
            var automaton2 = StringAutomaton.FromData(
                new StringAutomaton.DataContainer(
                    0,
                    ImmutableArray.Create(new StringAutomaton.StateData(0, 0, Weight.Zero)),
                    ImmutableArray<StringAutomaton.Transition>.Empty,
                    isEpsilonFree: true,
                    usesGroups: false,
                    isDeterminized: true,
                    isZero: true,
                    isEnumerable: true));
            Assert.True(automaton2.IsZero());

            // Bad start state index
            Assert.Throws<ArgumentException>(
                () => StringAutomaton.FromData(
                    new StringAutomaton.DataContainer(
                        0,
                        ImmutableArray<StringAutomaton.StateData>.Empty,
                        ImmutableArray<StringAutomaton.Transition>.Empty,
                        isEpsilonFree: true,
                        usesGroups: false,
                        isDeterminized: false,
                        isZero: true,
                        isEnumerable: false)));

            // automaton is actually epsilon-free, but data says that it is
            Assert.Throws<ArgumentException>(
                () => StringAutomaton.FromData(
                    new StringAutomaton.DataContainer(
                        0,
                        ImmutableArray.Create(new StringAutomaton.StateData(0, 0, Weight.Zero)),
                        ImmutableArray<StringAutomaton.Transition>.Empty,
                        isEpsilonFree: false,
                        usesGroups: false,
                        isDeterminized: null,
                        isZero: null,
                        isEnumerable: null)));

            // automaton is not epsilon-free
            Assert.Throws<ArgumentException>(
                () => StringAutomaton.FromData(
                    new StringAutomaton.DataContainer(
                        0,
                        ImmutableArray.Create(new StringAutomaton.StateData(0, 1, Weight.Zero)),
                        ImmutableArray.Create(new StringAutomaton.Transition(Option.None, Weight.One, 1)),
                        isEpsilonFree: false,
                        usesGroups: false,
                        isDeterminized: null,
                        isZero: null,
                        isEnumerable: null)));

            // Incorrect transition index
            Assert.Throws<ArgumentException>(
                () => StringAutomaton.FromData(
                    new StringAutomaton.DataContainer(
                        0,
                        ImmutableArray.Create(new StringAutomaton.StateData(0, 1, Weight.One)),
                        ImmutableArray.Create(new StringAutomaton.Transition(Option.None, Weight.One, 2)),
                        true,
                        false,
                        isDeterminized: null,
                        isZero: null,
                        isEnumerable: null)));
             */
        }

        #region ToString tests

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToString1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddSelfTransition('b', Weight.One).AddTransition('c', Weight.One).SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            Assert.Equal("ab*c", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("ab*?c", automaton.ToString(AutomatonFormats.Regexp));
        }
            
        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")] // TODO: extract longest common suffix when 'or'-ing concatenations
        [Trait("Category", "StringInference")]
        public void ConvertToString2()
        {
            var builder = new StringAutomaton.Builder();
            var middleState = builder.Start.AddTransition('b', Weight.One).AddTransition('c', Weight.One);
            builder.Start.AddTransition('a', Weight.One, middleState.Index);
            middleState.AddTransition('d', Weight.One).SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            Assert.Equal("(bc|a)d", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("(bc|a)d", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToString3()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransitionsForSequence("hello").SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            Assert.Equal("hello", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("hello", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToString4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransitionsForSequence("hello").SetEndWeight(Weight.Zero);
            var automaton = builder.GetAutomaton();
            Assert.Equal("Ø", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("Ø", automaton.ToString(AutomatonFormats.Regexp));
        }
            
        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToString5()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransitionsForSequence("hello").SetEndWeight(Weight.One);
            builder.Start.AddEpsilonTransition(Weight.One).AddTransitionsForSequence("hi").SetEndWeight(Weight.One);
            builder.Start.AddEpsilonTransition(Weight.One).AddTransitionsForSequence("hey").SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            Assert.Equal("hey|hi|hello", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("hey|hi|hello", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")] // TODO: represent character distributions in a nice way, merge star concatenations if possible
        [Trait("Category", "StringInference")]
        public void ConvertToString6()
        {
            StringAutomaton automaton = StringAutomaton.Constant(1.0, DiscreteChar.Lower());
            automaton.AppendInPlace(StringAutomaton.Constant(2.0));
            Assert.Equal("?*", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("[a-z]*.*", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToString7()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "abc");
            automaton.SetToSum(1.0, automaton, 1.0, automaton);
            Assert.Equal("abc", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("abc", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")] // TODO: represent character distributions in a nice way
        [Trait("Category", "StringInference")]
        public void ConvertToString8()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOnElement(1.0, 'a');
            automaton.AppendInPlace(StringAutomaton.ConstantOnElement(2.0, DiscreteChar.Upper()));
            automaton.AppendInPlace(StringAutomaton.ConstantOnElement(1.0, 'b'));
            Assert.Equal("a?b", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("a[A-Z]b", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")] // TODO: represent character distributions in a nice way, somehow merge element sets when friendly format is requested
        [Trait("Category", "StringInference")]
        public void ConvertToString9()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Digit());
            automaton.AppendInPlace(StringAutomaton.ConstantOnElement(2.0, DiscreteChar.Upper()));
            automaton.AppendInPlace(StringAutomaton.Sum(
                StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Lower()),
                StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Digit())));
            Assert.Equal("???", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("[0-9][A-Z]([a-z]|[0-9])", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ConvertToString10()
        {
            StringAutomaton automaton = StringAutomaton.ConstantOn(1.0, "ab");
            automaton.AppendInPlace(StringAutomaton.Sum(
                StringAutomaton.ConstantOn(1.0, "cd"),
                StringAutomaton.ConstantOn(1.0, "ef"),
                StringAutomaton.ConstantOn(1.0, string.Empty)));
            
            Assert.Equal("ab[ef|cd]", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("ab(|ef|cd)", automaton.ToString(AutomatonFormats.Regexp));
        }

        /// <summary>
        /// Tests converting an automaton to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "StringInference")]
        public void ConvertToString11()
        {
            StringAutomaton automaton = StringAutomaton.Repeat(StringAutomaton.ConstantOn(1.0, "a", "b"), minTimes: 0);

            Assert.Equal("[b|a|(b|a)*]", automaton.ToString(AutomatonFormats.Friendly));
            Assert.Equal("b||a|(b|a)*", automaton.ToString(AutomatonFormats.Regexp));
        }


        [Fact]
        [Trait("Category", "StringInference")]
        public void RegexForAutomatonWithNonTrivialComponent()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(7);

            builder[0].AddTransition('a', Weight.FromValue(1), 1);
            builder[1].AddTransition('b', Weight.FromValue(1), 1);
            builder[1].AddTransition('c', Weight.FromValue(1), 1);
            builder[1].AddTransition('d', Weight.FromValue(1), 2);
            builder[2].AddTransition('e', Weight.FromValue(1), 3);
            builder[3].AddTransition('f', Weight.FromValue(1), 3);
            builder[3].AddTransition('g', Weight.FromValue(1), 4);
            builder[3].AddTransition('h', Weight.FromValue(1), 5);
            builder[4].AddTransition('i', Weight.FromValue(1), 7);
            builder[4].AddTransition('j', Weight.FromValue(1), 5);
            builder[5].AddTransition('k', Weight.FromValue(1), 6);
            builder[5].AddTransition('l', Weight.FromValue(1), 6);
            builder[6].AddTransition('m', Weight.FromValue(1), 3);
            builder[6].AddTransition('n', Weight.FromValue(1), 1);
            builder[7].SetEndWeight(Weight.FromValue(1));

            var automaton = builder.GetAutomaton();

            var distribution = StringDistribution.FromWorkspace(automaton);
            var regexPattern = distribution.ToRegex();

            var inputs = new string[]
            {
                "adegi",
                "abbcdegi",
                "abcbdeffgi",
                "acdefhknbbcdefffgi",
                "abcbcbdegjlmffhknbbbcdeffgjlmffgi"
            };

            foreach (var input in inputs)
            {
                var match = Regex.Match(input, regexPattern);
                Assert.True(match.Success);
                Assert.Equal(input, match.Value);
            }
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void RegexForAutomatonWithNonTrivialComponent2()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(7);

            builder[0].AddTransition('a', Weight.FromValue(1), 1);
            builder[1].AddTransition('b', Weight.FromValue(1), 1);
            builder[1].AddTransition('c', Weight.FromValue(1), 1);
            builder[1].AddTransition('d', Weight.FromValue(1), 2);
            builder[2].AddTransition('e', Weight.FromValue(1), 3);
            builder[3].AddTransition('f', Weight.FromValue(1), 3);
            builder[3].AddTransition('g', Weight.FromValue(1), 4);
            builder[3].AddTransition('h', Weight.FromValue(1), 5);
            builder[4].AddTransition('i', Weight.FromValue(1), 7);
            builder[4].AddTransition('j', Weight.FromValue(1), 5);
            builder[5].AddTransition('k', Weight.FromValue(1), 6);
            builder[5].AddTransition('l', Weight.FromValue(1), 6);
            builder[6].AddTransition('m', Weight.FromValue(1), 3);
            builder[6].AddTransition('n', Weight.FromValue(1), 1);

            builder[0].AddTransition('o', Weight.FromValue(1), 6);
            builder[1].AddTransition('p', Weight.FromValue(1), 7);
            builder[6].AddTransition('q', Weight.FromValue(1), 7);
            builder[7].SetEndWeight(Weight.FromValue(1));

            var automaton = builder.GetAutomaton();

            var distribution = StringDistribution.FromWorkspace(automaton);
            var regexPattern = distribution.ToRegex();

            var inputs = new string[]
            {
                "adegi",
                "abbcdegi",
                "abcbdeffgi",
                "acdefhknbbcdefffgi",
                "abcbcbdegjlmffhknbbbcdeffgjlmffgi"
            };

            foreach (var input in inputs)
            {
                var match = Regex.Match(input, regexPattern);
                Assert.True(match.Success);
                Assert.Equal(input, match.Value);
            }
        }

        #endregion

        #region Equality

        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality1()
        {
            StringAutomaton automaton1 =
                StringAutomaton.FromValues(new Dictionary<string, double> { { "abc", 1.0 }, { "ab", 2.0 }, { "bc", 1.5 } });
            StringAutomaton automaton2 =
                StringAutomaton.ConstantOn(0.5, "ab")
                                       .Sum(StringAutomaton.ConstantOn(0.5, "ab", "abc"))
                                       .Scale(2)
                                       .Sum(StringAutomaton.ConstantOn(1.5, "bc"));
            AssertEquals(automaton1, automaton1);
            AssertEquals(automaton2, automaton2);
            AssertEquals(automaton1, automaton2);
            AssertNotEquals(automaton1, automaton1.Product(automaton2));
            AssertEquals(automaton1.Product(automaton2), automaton2.Product(automaton1));
            AssertEquals(automaton1.Sum(automaton2), automaton2.Sum(automaton1));
            AssertNotEquals(automaton1, automaton2.Sum(StringAutomaton.ConstantOn(0.01, "ab")));
            AssertNotEquals(automaton1, automaton2.Sum(StringAutomaton.ConstantOn(0.01, "ad")));
        }

        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality2()
        {
            StringAutomaton automaton1 =
                StringAutomaton.Constant(1.5, DiscreteChar.Lower())
                                       .Sum(StringAutomaton.Constant(0.5, DiscreteChar.Lower()))
                                       .Scale(0.75);
            StringAutomaton automaton2 = StringAutomaton.Constant(1.5, DiscreteChar.Lower());
            AssertEquals(automaton1, automaton1);
            AssertEquals(automaton2, automaton2);
            AssertEquals(automaton1, automaton2);
            AssertNotEquals(automaton1, automaton1.Product(automaton2));
            AssertNotEquals(automaton1, automaton1.Product(automaton2).Scale(1.0 / (1.5 * 1.5)));
            AssertEquals(automaton1.Product(automaton2), automaton2.Product(automaton1));
            AssertEquals(automaton1.Sum(automaton2), automaton2.Sum(automaton1));
            AssertNotEquals(automaton1, automaton2.Scale(0.99));
            AssertNotEquals(automaton1, automaton2.Sum(StringAutomaton.Constant(0.01, DiscreteChar.Lower())));
        }

        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality3()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.SetEndWeight(Weight.One);
            builder.Start.AddSelfTransition(DiscreteChar.Lower(), Weight.FromLogValue(26 + 1e-3));
            var automaton = builder.GetAutomaton();

            AssertEquals(automaton, automaton);
            AssertNotEquals(automaton, automaton.Product(automaton));
        }

        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.SetEndWeight(Weight.One);
            builder.Start.AddSelfTransition('a', Weight.FromLogValue(1.0 - 1e-3));
            var automaton = builder.GetAutomaton();

            AssertEquals(automaton, automaton);
            AssertNotEquals(automaton, automaton.Product(automaton));
        }
        
        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality5()
        {
            AssertEquals(StringAutomaton.Zero(), StringAutomaton.Zero());
            AssertNotEquals(StringAutomaton.Zero(), StringAutomaton.Constant(1.0));

            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One);
            var automaton = builder.GetAutomaton();

            AssertEquals(StringAutomaton.Zero(), automaton);
        }

        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality6()
        {
            StringAutomaton func1 = StringAutomaton.Constant(1.0, DiscreteChar.OneOf('a', 'b'));

            var func2Builder = new StringAutomaton.Builder();
            func2Builder.Start.AddSelfTransition('a', Weight.One).AddSelfTransition('b', Weight.One).SetEndWeight(Weight.One);
            var func2 = func2Builder.GetAutomaton();

            AssertEquals(func1, func2);
        }

        /// <summary>
        /// Tests the automata equality operator.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Equality7()
        {
            var func1Builder = new StringAutomaton.Builder();
            func1Builder.Start.AddSelfTransition(DiscreteChar.PointMass('a'), Weight.One).SetEndWeight(Weight.One);
            var func1 = func1Builder.GetAutomaton();

            var func2Builder = new StringAutomaton.Builder();
            func2Builder.Start
                .AddEpsilonTransition(Weight.One)
                .AddTransition(DiscreteChar.PointMass('a'), Weight.One, func2Builder.StartStateIndex)
                .SetEndWeight(Weight.One);
            var func2 = func2Builder.GetAutomaton();

            AssertEquals(func1, func2);
        }

        #endregion

        #region Simplification

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddTransition('b', Weight.One).SetEndWeight(
                Weight.FromValue(2.0));
            builder.Start.AddTransition('a', Weight.One).AddSelfTransition('d', Weight.One).AddTransition('c', Weight.One).SetEndWeight(
                Weight.FromValue(3.0));
            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 2.0, "ab");
                StringInferenceTestUtilities.TestValue(automaton, 3.0, "adc", "adddc", "ac");
                StringInferenceTestUtilities.TestValue(automaton, 0.0, "adb", "ad", string.Empty);

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify2()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddSelfTransition('d', Weight.One).AddTransition('c', Weight.One).SetEndWeight(
                Weight.FromValue(3.0));
            builder.Start.AddTransition('a', Weight.One).AddTransition('b', Weight.One).SetEndWeight(
                Weight.FromValue(2.0));
            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
        {
                StringInferenceTestUtilities.TestValue(automaton, 2.0, "ab");
                StringInferenceTestUtilities.TestValue(automaton, 3.0, "adc", "adddc", "ac");
                StringInferenceTestUtilities.TestValue(automaton, 0.0, "adb", "ad", string.Empty);

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify3()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.One).AddTransition('d', Weight.One).AddTransition('c', Weight.One).SetEndWeight(
                Weight.FromValue(2.0));
            builder.Start.AddTransition('a', Weight.One).AddSelfTransition('d', Weight.One).AddTransition('c', Weight.One).SetEndWeight(
                Weight.FromValue(3.0));
            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
        {
                StringInferenceTestUtilities.TestValue(automaton, 5.0, "adc");
                StringInferenceTestUtilities.TestValue(automaton, 3.0, "addc", "adddc", "ac");
            
                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddEpsilonTransition(Weight.One).AddSelfTransition('a', Weight.One)
                           .AddEpsilonTransition(Weight.One).AddSelfTransition('b', Weight.One).SetEndWeight(Weight.FromValue(2.0));
            builder.Start.AddEpsilonTransition(Weight.One).AddSelfTransition('a', Weight.One)
                           .AddEpsilonTransition(Weight.One).AddSelfTransition('c', Weight.One).SetEndWeight(Weight.FromValue(3.0));
            builder.Start.AddSelfTransition('x', Weight.One);
            builder.Start.SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 2.0, "aabb", "abb", "xxaab", "xbb");
                StringInferenceTestUtilities.TestValue(automaton, 3.0, "aacc", "acc", "xxaac", "xcc");
                StringInferenceTestUtilities.TestValue(automaton, 5.0, "xxaa", "xa", "a");
                StringInferenceTestUtilities.TestValue(automaton, 6.0, "xx", "x", string.Empty);

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify5()
        {
            var builder = new StringAutomaton.Builder();
            const string AcceptedSequence = "aaaaaaaaaa";
            var state = builder.Start;
            for (int i = 0; i < AcceptedSequence.Length; ++i)
            {
                var nextState = state.AddTransition(AcceptedSequence[i], Weight.One);
                state.AddTransition(AcceptedSequence[i], Weight.One, nextState.Index);
                state = nextState;
            }

            state.SetEndWeight(Weight.One);

            const int AdditionalSequenceCount = 5;
            for (int i = 0; i < AdditionalSequenceCount; ++i)
            {
                builder.Start.AddTransitionsForSequence(AcceptedSequence).SetEndWeight(Weight.One);
            }

            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, (1 << AcceptedSequence.Length) + AdditionalSequenceCount, AcceptedSequence);

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify6()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(5);
            builder[0].AddTransition('a', Weight.FromValue(1.0), 1);
            builder[0].AddTransition('a', Weight.FromValue(4.0), 4);
            builder[1].AddTransition('a', Weight.FromValue(2.0), 2);
            builder[2].AddTransition('a', Weight.FromValue(3.0), 3);
            builder[2].AddTransition('a', Weight.FromValue(6.0), 5);
            builder[4].AddTransition('a', Weight.FromValue(5.0), 2);

            builder[3].SetEndWeight(Weight.FromValue(2.0));
            builder[5].SetEndWeight(Weight.FromValue(3.0));

            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 528.0, "aaa");
                StringInferenceTestUtilities.TestValue(automaton, 0.0, "a", "aa", "aaaa", string.Empty);

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify7()
        {
            DiscreteChar lowerEnglish = DiscreteChar.UniformInRange('a', 'z');
            DiscreteChar upperEnglish = DiscreteChar.UniformInRange('A', 'Z');

            var builder = new StringAutomaton.Builder();
            var branch1 = builder.Start.AddEpsilonTransition(Weight.FromValue(0.5)).AddTransition('a', Weight.FromValue(1.0 / 3.0)).AddTransition('B', Weight.FromValue(1.0 / 4.0));
            branch1.SetEndWeight(Weight.FromValue(3.0));
            branch1.AddTransition('X', Weight.FromValue(1.0 / 6.0)).SetEndWeight(Weight.FromValue(5.0));
            branch1.AddEpsilonTransition(Weight.FromValue(1.0 / 8.0)).SetEndWeight(Weight.FromValue(7.0));
            var branch2 = builder.Start.AddTransition(lowerEnglish, Weight.FromValue(2.0));
            branch2.SetEndWeight(Weight.FromValue(4.0));
            branch2.AddTransition(upperEnglish, Weight.FromValue(3.0), branch2.Index);
            branch2.AddTransition('X', Weight.FromValue(4.0)).SetEndWeight(Weight.FromValue(5.0));
            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 8.0 / 26.0, "a");
                StringInferenceTestUtilities.TestValue(automaton, 2.0 * 3.0 * 4.0 / (26.0 * 26.0), "aA");
                StringInferenceTestUtilities.TestValue(automaton, 72.0 / (26.0 * 26.0 * 26.0), "aBC");
                StringInferenceTestUtilities.TestValue(automaton, (24.0 / (26.0 * 26.0)) + (1.0 / 8.0) + (7.0 / 192.0), "aB");
                StringInferenceTestUtilities.TestValue(automaton, (120.0 / (26.0 * 26.0)) + (72.0 / (26.0 * 26.0 * 26.0)) + (5.0 / 144.0), "aBX");

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify8()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(10);

            builder[0].AddTransition('a', Weight.One, 0);
            builder[0].AddTransition('a', Weight.One, 1);
            builder[0].AddTransition('b', Weight.One, 2);
            builder[0].AddTransition('a', Weight.One, 4);
            builder[0].AddEpsilonTransition(Weight.One, 7);
            builder[0].AddTransition('b', Weight.One, 8);
            builder[0].AddTransition('c', Weight.One, 10);
            builder[0].AddTransition('c', Weight.One, 10);
            builder[1].AddTransition('b', Weight.One, 2);
            builder[2].AddTransition('a', Weight.One, 3);
            builder[4].AddTransition('a', Weight.One, 5);
            builder[5].AddTransition('b', Weight.One, 6);
            builder[5].AddTransition('a', Weight.One, 6);
            builder[7].AddTransition('b', Weight.One, 5);
            builder[8].AddTransition('b', Weight.One, 8);
            builder[8].AddTransition('a', Weight.One, 9);

            builder[3].SetEndWeight(Weight.FromValue(0.1));
            builder[6].SetEndWeight(Weight.FromValue(0.2));
            builder[9].SetEndWeight(Weight.FromValue(0.3));
            builder[10].SetEndWeight(Weight.FromValue(0.4));

            var automaton = builder.GetAutomaton();

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 0.1 + 0.1 + 0.2 + 0.3, "aba", "aaba", "aaaba");
                StringInferenceTestUtilities.TestValue(automaton, 0.1 + 0.2 + 0.3, "ba");
                StringInferenceTestUtilities.TestValue(automaton, 0.3, "abba");
                StringInferenceTestUtilities.TestValue(automaton, 0.2, "abb");
                StringInferenceTestUtilities.TestValue(automaton, 0.4 + 0.4, "c");

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests the automaton simplification heuristic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Simplify9()
        {
            StringAutomaton automaton = StringAutomaton.Sum(
                StringAutomaton.Constant(1.0, DiscreteChar.PointMass('a')).Append("b"),
                StringAutomaton.Empty(),
                StringAutomaton.Constant(1.0, DiscreteChar.PointMass('b')),
                StringAutomaton.ConstantOn(1.0, "abc"),
                StringAutomaton.ConstantOn(1.0, "abd"),
                StringAutomaton.ConstantOn(1.0, "ab"));

            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 1.0, "abc", "abd", "aab", "aaab", "bb", "bbbb");
                StringInferenceTestUtilities.TestValue(automaton, 2.0, "ab", "b", string.Empty);
                StringInferenceTestUtilities.TestValue(automaton, 0.0, "a", "aabc", "aabd", "ba", "bc");

                automaton.Simplify();
            }
        }

        /// <summary>
        /// Tests that merging parallel transitions doesn't fail when transition weights are too large
        /// to fit in double in non-log space.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MergeParallelTransitionsWithHighTransitionsWeightsDoesNotThrow()
        {
            var builder = new StringAutomaton.Builder();
            // Add 2 identical transitions from start state to second
            builder.Start.AddTransition(DiscreteChar.Any(), Weight.FromLogValue(1e5));
            builder.Start.AddTransition(DiscreteChar.Any(), Weight.FromLogValue(1e5), 1);
            var automaton = builder.GetAutomaton();
            automaton.Simplify();
        }

        #endregion

        #region Determinization

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.FromValue(2)).AddTransition('b', Weight.FromValue(3)).SetEndWeight(Weight.FromValue(4));
            builder.Start.AddTransition('a', Weight.FromValue(5)).AddTransition('c', Weight.FromValue(6)).SetEndWeight(Weight.FromValue(7));
            builder.Start.SetEndWeight(Weight.FromValue(17));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 17, string.Empty);
                StringInferenceTestUtilities.TestValue(automaton, 24, "ab");
                StringInferenceTestUtilities.TestValue(automaton, 210, "ac");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize2()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', 'z'), Weight.FromValue(2)).AddTransition('b', Weight.FromValue(3)).SetEndWeight(Weight.FromValue(4));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', 'z'), Weight.FromValue(5)).AddTransition('c', Weight.FromValue(6)).SetEndWeight(Weight.FromValue(7));
            builder.Start.SetEndWeight(Weight.FromValue(17));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 17, string.Empty);
                StringInferenceTestUtilities.TestValue(automaton, 24 / 26.0, "ab", "nb", "zb");
                StringInferenceTestUtilities.TestValue(automaton, 210 / 26.0, "ac", "nc", "zc");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize3()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition(DiscreteChar.Uniform(), Weight.FromValue(2)).AddTransition('b', Weight.FromValue(3)).SetEndWeight(Weight.FromValue(4));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', 'z'), Weight.FromValue(5)).AddTransition('c', Weight.FromValue(6)).SetEndWeight(Weight.FromValue(7));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('x', 'z'), Weight.FromValue(8)).AddTransition('d', Weight.FromValue(9)).SetEndWeight(Weight.FromValue(10));
            builder.Start.SetEndWeight(Weight.FromValue(17));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 17, string.Empty);
                StringInferenceTestUtilities.TestValue(
                    automaton, 24 / 65536.0, "ab", "nb", "zb", char.MinValue + "b", char.MaxValue + "b", (char)('a' - 1) + "b", (char)('a' + 1) + "b");
                StringInferenceTestUtilities.TestValue(automaton, 210 / 26.0, "ac", "nc", "zc");
                StringInferenceTestUtilities.TestValue(automaton, 720 / 3.0, "xd", "yd", "zd");
                StringInferenceTestUtilities.TestValue(
                    automaton, 0.0, (char)('a' - 1) + "c", (char)('z' + 1) + "c", "wd", (char)('z' + 1) + "d");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize4()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.FromValue(2)).AddSelfTransition('b', Weight.FromValue(0.5)).AddTransition('c', Weight.FromValue(3.0)).SetEndWeight(Weight.FromValue(4));
            builder.Start.AddTransition('a', Weight.FromValue(5)).AddSelfTransition('b', Weight.FromValue(0.5)).AddTransition('d', Weight.FromValue(6.0)).SetEndWeight(Weight.FromValue(7));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 24, "ac");
                StringInferenceTestUtilities.TestValue(automaton, 12, "abc");
                StringInferenceTestUtilities.TestValue(automaton, 6, "abbc");
                StringInferenceTestUtilities.TestValue(automaton, 210, "ad");
                StringInferenceTestUtilities.TestValue(automaton, 105, "abd");
                StringInferenceTestUtilities.TestValue(automaton, 52.5, "abbd");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize5()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.FromValue(2)).AddSelfTransition('b', Weight.FromValue(0.5)).AddTransition('c', Weight.FromValue(3.0)).SetEndWeight(Weight.FromValue(4));
            builder.Start.AddTransition('a', Weight.FromValue(5)).AddSelfTransition('b', Weight.FromValue(0.5)).AddTransition('c', Weight.FromValue(6.0)).SetEndWeight(Weight.FromValue(7));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 234, "ac");
                StringInferenceTestUtilities.TestValue(automaton, 117, "abc");
                StringInferenceTestUtilities.TestValue(automaton, 58.5, "abbc");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize6()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition(DiscreteChar.UniformInRange('a', 'c'), Weight.FromValue(2)).SetEndWeight(Weight.FromValue(3.0));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('b', 'c'), Weight.FromValue(4)).SetEndWeight(Weight.FromValue(5.0));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('b', 'd'), Weight.FromValue(6)).SetEndWeight(Weight.FromValue(7.0));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('d', 'd'), Weight.FromValue(8)).SetEndWeight(Weight.FromValue(9.0));
            builder.Start.AddTransition(DiscreteChar.UniformInRange('d', 'e'), Weight.FromValue(10)).SetEndWeight(Weight.FromValue(11.0));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 2, "a");
                StringInferenceTestUtilities.TestValue(automaton, 26, "b", "c");
                StringInferenceTestUtilities.TestValue(automaton, 141, "d");
                StringInferenceTestUtilities.TestValue(automaton, 55, "e");
                StringInferenceTestUtilities.TestValue(automaton, 0, ((char)('a' - 1)).ToString(), "f");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize7()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(4);

            builder[0].AddTransition('a', Weight.FromValue(3), 1);
            builder[0].AddTransition('a', Weight.FromValue(2), 2);
            builder[0].AddTransition('b', Weight.FromValue(1), 2);
            builder[1].AddTransition('b', Weight.FromValue(4), 3);
            builder[2].AddTransition(DiscreteChar.UniformOver('b', 'c'), Weight.FromValue(10), 3);
            builder[2].AddTransition('b', Weight.FromValue(6), 4);
            builder[3].AddTransition('c', Weight.FromValue(7), 4);

            builder[2].SetEndWeight(Weight.FromValue(0.5));
            builder[3].SetEndWeight(Weight.FromValue(1));
            builder[4].SetEndWeight(Weight.FromValue(2));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 1, "a");
                StringInferenceTestUtilities.TestValue(automaton, 0.5, "b");
                StringInferenceTestUtilities.TestValue(automaton, 46, "ab");
                StringInferenceTestUtilities.TestValue(automaton, 17, "bb");
                StringInferenceTestUtilities.TestValue(automaton, 10, "ac");
                StringInferenceTestUtilities.TestValue(automaton, 308, "abc");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize8()
        {
            var builder = new StringAutomaton.Builder();
            var state = builder.Start;

            const int AcceptedSequenceLength = 20;
            for (int i = 0; i < AcceptedSequenceLength; ++i)
            {
                var nextState = state.AddTransition(DiscreteChar.Uniform(), Weight.One);
                state.AddTransition(DiscreteChar.Uniform(), Weight.One, nextState.Index);
                state = nextState;
            }

            state.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());
            
            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            double logValue = AcceptedSequenceLength * (MMath.Ln2 - Math.Log(char.MaxValue + 1.0));
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestLogValue(automaton, logValue, new string('a', AcceptedSequenceLength), new string('b', AcceptedSequenceLength));
                StringInferenceTestUtilities.TestValue(automaton, 0, "a");
                
                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests automata determinization.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize9()
        {
            var builder = new StringAutomaton.Builder();

            const int TransitionsPerCharacter = 3;
            for (int i = 0; i < TransitionsPerCharacter; ++i)
            {
                builder.Start.AddTransition('a', Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition('b', Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition('d', Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition('e', Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition('g', Weight.One).SetEndWeight(Weight.One);    
            }

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic() || TransitionsPerCharacter <= 1);

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, TransitionsPerCharacter, "a", "b", "d", "e", "g");
                StringInferenceTestUtilities.TestValue(automaton, 0, "c", "f", "h", "i");

                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests determinization of an automaton with an infinite weight.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize10()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(4);

            builder[0].AddTransition('a', Weight.FromValue(3), 1);
            builder[0].AddTransition('a', Weight.Infinity, 2);
            builder[0].AddTransition('b', Weight.FromValue(1), 2);
            builder[1].AddTransition('b', Weight.FromValue(4), 3);
            builder[2].AddTransition(DiscreteChar.UniformOver('b', 'c'), Weight.FromValue(10), 3);
            builder[2].AddTransition('b', Weight.FromValue(6), 4);
            builder[3].AddTransition('c', Weight.FromValue(7), 4);

            builder[2].SetEndWeight(Weight.FromValue(0.5));
            builder[3].SetEndWeight(Weight.FromValue(1));
            builder[4].SetEndWeight(Weight.FromValue(2));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                Assert.True(double.IsInfinity(automaton.GetLogValue("a")));
                StringInferenceTestUtilities.TestValue(automaton, 0.5, "b");
                Assert.True(double.IsInfinity(automaton.GetLogValue("ab")));
                StringInferenceTestUtilities.TestValue(automaton, 17, "bb");
                Assert.True(double.IsInfinity(automaton.GetLogValue("ac")));
                Assert.True(double.IsInfinity(automaton.GetLogValue("abc")));
                bool determinized = automaton.TryDeterminize();
                Assert.True(determinized);
                Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests determinization of an automaton with hugely different weights
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Determinize11()
        {
            var builder = new StringAutomaton.Builder();

            builder.Start
                .AddTransition(DiscreteChar.InRange('a', 'c'), Weight.FromLogValue(-1000))
                .AddTransition(DiscreteChar.PointMass('x'), Weight.One)
                .SetEndWeight(Weight.One);
            builder.Start
                .AddTransition(DiscreteChar.PointMass('b'), Weight.One)
                .SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());
            var determinized = automaton.TryDeterminize();

            Assert.True(determinized);
            Assert.True(automaton.IsDeterministic());

            // "cx" vanishes from automaton language with naive implementation of segments overlap
            // due to numerical errors
            var dist = StringDistribution.FromWeightFunction(automaton);
            StringInferenceTestUtilities.TestIfIncludes(dist, "ax", "bx", "cx", "b");
        }


        /// <summary>
        /// Tests whether the inability to determinize an automaton is handled correctly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void NonDeterminizable1()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start.AddTransition('a', Weight.FromValue(2)).AddSelfTransition('b', Weight.FromValue(0.5)).AddTransition('c', Weight.FromValue(3.0)).SetEndWeight(Weight.FromValue(4));
            builder.Start.AddTransition('a', Weight.FromValue(5)).AddSelfTransition('b', Weight.FromValue(0.1)).AddTransition('c', Weight.FromValue(6.0)).SetEndWeight(Weight.FromValue(7));

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic());

            // Test: original automaton, automaton after determinization attempt failed (should not change)
            for (int i = 0; i < 2; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, 24 + 210, "ac");
                StringInferenceTestUtilities.TestValue(automaton, 12 + 21, "abc");
                StringInferenceTestUtilities.TestValue(automaton, 6 + 2.1, "abbc");

                bool determinized = automaton.TryDeterminize();
                Assert.False(determinized);

                Assert.False(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests that determinization doesn't produce anything unexpected.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CountDeterminizedStates()
        {
            Assert.Equal(2, CountStates("a"));
            Assert.Equal(4, CountStates("a", "ba"));
            Assert.Equal(5, CountStates("a", "ba", "bb"));
            Assert.Equal(6, CountStates("a", "ba", "bb", "d"));
        }

        //[Fact]
        [Trait("Category", "StringInference")]
        internal void DeterminizeList()
        {
            var builder = new Automaton<List<string>, string, StringDistribution, ListManipulator<List<string>, string>, ListAutomaton<string, StringDistribution>>.Builder();

            const int TransitionsPerCharacter = 3;
            for (int i = 0; i < TransitionsPerCharacter; ++i)
            {
                builder.Start.AddTransition("a", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("b", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("d", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("e", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("g", Weight.One).SetEndWeight(Weight.One);
            }

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic() || TransitionsPerCharacter <= 1);

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
                StringInferenceTestUtilities.TestValue(automaton, TransitionsPerCharacter, new List<string>{"a"}, new List<string>{"b"});
                StringInferenceTestUtilities.TestValue(automaton, 0, new List<string>{ "c"});

                Console.WriteLine("automaton=" +automaton.ToString(AutomatonFormats.Regexp)+" "+automaton.States.Count);
                bool determinized = automaton.TryDeterminize();
                
                Assert.True(determinized);

                Assert.True(automaton.IsDeterministic());
            }
        }


        //[Fact]
        [Trait("Category", "StringInference")]
        internal void DeterminizeList2()
        {
            var builder = new Automaton<List<string>, string, StringDistribution, ListManipulator<List<string>, string>, ListAutomaton<string, StringDistribution>>.Builder();
            var unif = StringDistribution.Uniform();
            var scaledUniform = StringDistribution.FromWorkspace(unif.GetWorkspaceOrPoint().Scale(1.0 / 1000));
   
            const int TransitionsPerCharacter = 3;
            for (int i = 0; i < TransitionsPerCharacter; ++i)
            {
                builder.Start.AddTransition("a", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("b", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("d", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition("e", Weight.One).SetEndWeight(Weight.One);
                builder.Start.AddTransition(scaledUniform, Weight.One).SetEndWeight(Weight.One);
            }

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.IsDeterministic() || TransitionsPerCharacter <= 1);

            // Test: original automaton, determinized automaton, determinization of the determinized automaton (shouldn't break anything)
            for (int i = 0; i < 3; ++i)
            {
               // StringInferenceTestUtilities.TestValue(automaton, TransitionsPerCharacter, new List<string> { "a" }, new List<string> { "b" });
                //StringInferenceTestUtilities.TestValue(automaton, 0, new List<string> { "c" });

                Console.WriteLine("automaton=" + automaton.ToString(AutomatonFormats.Regexp) + " " + automaton.States.Count);
                bool determinized = automaton.TryDeterminize();

                //Assert.True(determinized);

                //Assert.True(automaton.IsDeterministic());
            }
        }

        /// <summary>
        /// Tests enumeration of support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void EnumerateSupport()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(6);

            builder[0].AddTransition(DiscreteChar.UniformOver('a', 'b'), Weight.FromValue(1), 1);
            builder[0].AddTransition(DiscreteChar.UniformOver('c', 'd'), Weight.FromValue(1), 2);
            builder[2].AddTransition(DiscreteChar.UniformOver('e', 'f'), Weight.FromValue(1), 3);
            builder[2].AddTransition(DiscreteChar.UniformOver('g', 'h'), Weight.FromValue(1), 4);
            builder[2].AddEpsilonTransition(Weight.FromValue(1), 4);
            builder[4].AddTransition(DiscreteChar.UniformOver('i', 'j'), Weight.FromValue(1), 5);
            builder[4].AddTransition(DiscreteChar.UniformOver('k', 'l'), Weight.FromValue(1), 5);

            builder[0].SetEndWeight(Weight.FromValue(1));
            builder[1].SetEndWeight(Weight.FromValue(1));
            builder[3].SetEndWeight(Weight.FromValue(1));
            builder[5].SetEndWeight(Weight.FromValue(1));
            builder[6].SetEndWeight(Weight.FromValue(1));

            var automaton = builder.GetAutomaton();

            var expectedSupport = new HashSet<string>
            {
                "",
                "a", "b", 
                "ce", "cf",
                "cgi", "cgj", "cgk", "cgl",
                "chi", "chj", "chk", "chl",
                "ci", "cj", "ck", "cl",
                "de", "df",
                "dgi", "dgj", "dgk", "dgl",
                "dhi", "dhj", "dhk", "dhl",
                "di", "dj", "dk", "dl"
            };

            var calculatedSupport1= new HashSet<string>(automaton.EnumerateSupport(tryDeterminize: false));
            Assert.True(calculatedSupport1.SetEquals(expectedSupport));

            var calculatedSupport2 = new HashSet<string>(automaton.EnumerateSupport(tryDeterminize: true));
            Assert.True(calculatedSupport2.SetEquals(expectedSupport));
        }

        /// <summary>
        /// Tests enumeration of support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void EnumerateSupportThrowsOnLoop()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start
                .AddTransition('a', Weight.One)
                .AddTransition('a', Weight.One, 0)
                .SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.Throws<NotSupportedException>(() => automaton.EnumerateSupport().ToList());
        }

        /// <summary>
        /// Tests enumeration of support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void TryEnumerateSupportReturnsFalseOnLoop()
        {
            var builder = new StringAutomaton.Builder();
            builder.Start
                .AddTransition('a', Weight.One)
                .AddTransition('a', Weight.One, 0)
                .SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            Assert.False(automaton.TryEnumerateSupport(Int32.MaxValue, out _));
        }

        [Trait("Category", "BadTest")] // Performance tests which look for exact timings are likely to fail on the build machine
        [Fact]
        public void TryEnumerateSupportPerfTest()
        {
            var builder = new StringAutomaton.Builder();
            builder.AddStates(6);

            builder[0].AddTransition(DiscreteChar.UniformOver('a', 'b'), Weight.FromValue(1), 1);
            builder[0].AddTransition(DiscreteChar.UniformOver('c', 'd'), Weight.FromValue(1), 2);
            builder[2].AddTransition(DiscreteChar.UniformOver('e', 'f'), Weight.FromValue(1), 3);
            builder[2].AddTransition(DiscreteChar.UniformOver('g', 'h'), Weight.FromValue(1), 4);
            builder[4].AddTransition(DiscreteChar.UniformOver('i', 'j'), Weight.FromValue(1), 5);
            builder[4].AddTransition(DiscreteChar.UniformOver('k', 'l'), Weight.FromValue(1), 5);

            builder[1].SetEndWeight(Weight.FromValue(1));
            builder[3].SetEndWeight(Weight.FromValue(1));
            builder[5].SetEndWeight(Weight.FromValue(1));
            builder[6].SetEndWeight(Weight.FromValue(1));

            var s = builder[5];
            for (var i = 0; i < 100; ++i) s = s.AddTransition('a', Weight.One);
            s.SetEndWeight(Weight.One);

            var automaton = builder.GetAutomaton();

            int numPasses = 10000;
            Stopwatch watch = new Stopwatch();
            IEnumerable<string> support = null;

            // Call once in order to prepare for test (Determinize etc.)
            automaton.TryEnumerateSupport(10, out support);
            
            // Method 1: Enumerate support and catch exception:
            watch.Restart();
            for (int i = 0; i < numPasses; i++)
            {
                try
                {
                    support = automaton.EnumerateSupport(10).ToList();
                }
                catch (AutomatonEnumerationCountException)
                {
                }
            }

            watch.Stop();
            long time1 = watch.ElapsedMilliseconds;

            // Method 2: call TryEnumerateSupport
            watch.Restart();
            for (int i = 0; i < numPasses; i++)
            {
               automaton.TryEnumerateSupport(10, out support);
            }

            watch.Stop();
            long time2 = watch.ElapsedMilliseconds;

            // Second method should be substantially faster.
            Console.WriteLine("{0}, {1}", time1, time2);
            Assert.True(time2 * 1.25 < time1);
        }


        #endregion

        #region SetToConstantOnSupportOf

        /// <summary>
        /// Tests setting an automaton to be constant on the support of another automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SetToConstantOnSupportOf1()
        {
            StringAutomaton nonUniform = StringAutomaton.ConstantOn(1.0, "a", "a", "a", "b", "b", "c");
            StringAutomaton uniform = StringAutomaton.ConstantOn(1.0, "a", "b", "c");

            Assert.False(nonUniform.Equals(uniform));

            nonUniform.SetToConstantOnSupportOf(1.0, nonUniform);

            Assert.True(nonUniform.Equals(uniform));
        }

        /// <summary>
        /// Tests setting an automaton to be constant on the support of another automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SetToConstantOnSupportOf2()
        {
            StringAutomaton nonUniform = StringAutomaton.ConstantOn(2.0, string.Empty, "a", "aa");
            nonUniform.AppendInPlace(StringAutomaton.ConstantOn(3.0, string.Empty, "a", "aa"));
            nonUniform = nonUniform.Sum(StringAutomaton.ConstantOn(5.0, string.Empty, "a", "aa"));

            StringAutomaton uniform = StringAutomaton.ConstantOn(1.0, string.Empty, "a", "aa", "aaa", "aaaa");

            Assert.False(nonUniform.Equals(uniform));

            nonUniform.SetToConstantOnSupportOf(1.0, nonUniform);

            Assert.True(nonUniform.Equals(uniform));
        }

        /// <summary>
        /// Tests setting an automaton to be constant on the support of another automaton.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SetToConstantOnSupportOf3()
        {
            StringAutomaton constantOnChar = StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Lower());
            
            StringAutomaton nonUniform = StringAutomaton.Repeat(constantOnChar, 1, 3);
            nonUniform = nonUniform.Sum(StringAutomaton.Repeat(constantOnChar, 0, 2));
            nonUniform = nonUniform.Sum(StringAutomaton.Repeat(StringAutomaton.ConstantOn(1.0, "a"), 0, 3));
            nonUniform = nonUniform.Product(nonUniform);

            StringAutomaton uniform = StringAutomaton.Repeat(constantOnChar, 0, 3);

            Assert.False(nonUniform.Equals(uniform));

            nonUniform.SetToConstantOnSupportOf(1.0, nonUniform);

            Assert.True(nonUniform.Equals(uniform));
        }

        /// <summary>
        /// Tests that SetToConstantSupportOf() Doesn't throw low probability transitions out.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SetToConstantSupportOfWithLowProbabilityTransition1()
        {
            var builder = new StringAutomaton.Builder(1);
            builder.Start
                .AddTransition('a', Weight.FromValue(5e-40))
                .SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            automaton.SetToConstantOnSupportOfLog(0.0, automaton);
            Assert.Equal(new[] { "a" }, automaton.EnumerateSupport().ToArray());
        }

        /// <summary>
        /// Tests that SetToConstantSupportOf() Doesn't throw low probability transitions out.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SetToConstantSupportOfWithMultipleLowProbabilityTransitions()
        {
            var builder = new StringAutomaton.Builder(1);
            builder.Start
                .AddTransition(DiscreteChar.OneOf('a', 'b'), Weight.One)
                .SetEndWeight(Weight.One);
            builder.Start
                .AddTransition(DiscreteChar.OneOf('a', 'b'), Weight.FromLogValue(-1000))
                .AddTransition('c', Weight.One)
                .SetEndWeight(Weight.One);
            builder.Start
                .AddTransition('c', Weight.FromLogValue(-10000))
                .SetEndWeight(Weight.One);
            var automaton = builder.GetAutomaton();
            var tmp = automaton.Clone();
            tmp.TryDeterminize();
            automaton.SetToConstantOnSupportOfLog(0.0, automaton);
            var support = automaton.EnumerateSupport().OrderBy(s => s).ToArray();
            Assert.Equal(new[] {"a", "ac", "b", "bc", "c"}, support);
        }


        #endregion

        #region Helpers

        /// <summary>
        /// Tests the behavior of the normalization operation on a non-normalizable automaton.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <param name="zeroNormalizer">Specifies whether the automaton normalizer is zero.</param>
        private static void TestNonNormalizable(StringAutomaton automaton, bool zeroNormalizer)
        {
            StringAutomaton automatonBefore = automaton.Clone();
            Assert.Equal(zeroNormalizer ? double.NegativeInfinity : double.PositiveInfinity, automaton.GetLogNormalizer());
            Assert.False(automaton.TryNormalizeValues());
            Assert.Throws<InvalidOperationException>(() => automaton.NormalizeValues());
            Assert.Equal(automaton, automatonBefore); // Failed normalization should not affect the automaton
        }

        /// <summary>
        /// Tests the behavior of the equality operator on two equal objects.
        /// </summary>
        /// <param name="a">The first object.</param>
        /// <param name="b">The second object.</param>
        private static void AssertEquals(object a, object b)
        {
            Assert.True(a.Equals(b));
            Assert.True(b.Equals(a));
            Assert.True(a.GetHashCode() == b.GetHashCode());
        }

        /// <summary>
        /// Tests the behavior of the equality operator on two different objects.
        /// </summary>
        /// <param name="a">The first object.</param>
        /// <param name="b">The second object.</param>
        private static void AssertNotEquals(object a, object b)
        {
            Assert.False(a.Equals(b));
            Assert.False(b.Equals(a));
        }

        /// <summary>
        /// Count the number of states in a determinized version of automaton that assigns one to the supplied strings
        /// </summary>
        /// <param name="values">The strings.</param>
        /// <returns>The number of states in the determinized automaton.</returns>
        private static int CountStates(params string[] values)
        {
            var distribution = StringDistribution.OneOf(values);
            var workspace = distribution.GetWorkspaceOrPoint();
            workspace.TryDeterminize();
            return workspace.States.Count;
        }
        
        #endregion
    }
}
