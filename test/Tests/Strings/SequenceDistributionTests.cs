// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Automata;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Tests for sequence distributions.
    /// </summary>
    public class SequenceDistributionTests
    {
        #region Distribution operation tests

        /// <summary>
        /// Tests distribution concatenation.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Concat()
        {
            var hello = StringDistribution.String("hello");
            var world = StringDistribution.String("world");
            var helloworld = hello + world;
            StringInferenceTestUtilities.TestProbability(helloworld, 1.0, "helloworld");
            StringInferenceTestUtilities.TestProbability(helloworld, 0.0, string.Empty, "hello", "world");

            var any = StringDistribution.Any();
            var hellostar = hello + any;
            StringInferenceTestUtilities.TestProbability(hellostar, 1.0, "helloworld", "hello", "hello world");
            StringInferenceTestUtilities.TestProbability(hellostar, 0.0, "hhelloworld", string.Empty, "hell", "hhello");

            var starhello = any + hello;
            StringInferenceTestUtilities.TestProbability(starhello, 1.0, "hhello", "hello", "well hello");
            StringInferenceTestUtilities.TestProbability(starhello, 0.0, "hhelloworld", string.Empty, "hell", "hello!!");

            var starhellostar = any + hello + any;
            StringInferenceTestUtilities.TestProbability(starhellostar, 1.0, "hello!!", "hhelloworld", "hhello", "hello", "well hello");
            StringInferenceTestUtilities.TestProbability(starhellostar, 0.0, string.Empty, "hell");

            var hellostarworld = hello + any + world;
            StringInferenceTestUtilities.TestProbability(hellostarworld, 1.0, "hello world", "helloworld", "hello uncertain world");
            StringInferenceTestUtilities.TestProbability(hellostarworld, 0.0, "hello", "world", "hello world!!");
        }

        /// <summary>
        /// Digits test.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Digits()
        {
            var uniformOverValue = StringDistribution.Digits(2, 3);
            var uniformOverLengthThenValue =
                StringDistribution.Digits(2, 3, DistributionKind.UniformOverLengthThenValue);

            var lp12A = uniformOverValue.GetLogProb("12");
            var lp13A = uniformOverValue.GetLogProb("123");
            var lp12B = uniformOverLengthThenValue.GetLogProb("12");
            var lp13B = uniformOverLengthThenValue.GetLogProb("123");

            Assert.Equal(1.0 / 1100, Math.Exp(lp12A), 1e-10);
            Assert.Equal(1.0 / 1100, Math.Exp(lp13A), 1e-10);
            Assert.Equal(0.5 / 100, Math.Exp(lp12B), 1e-10);
            Assert.Equal(0.5 / 1000, Math.Exp(lp13B), 1e-10);
        }

        /// <summary>
        /// Tests distribution product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product1()
        {
            var hello = StringDistribution.String("hello");
            var world = StringDistribution.String("world");
            var helloworld = StringDistribution.String("hello world");
            var subs = StringDistribution.OneOf("hello", "world", "hello world");
            var any = StringDistribution.Any();

            string[] interestingStrings = new[] { "hello", "world", "hello world", "hell", "ello", string.Empty };

            StringInferenceTestUtilities.TestProduct(subs, subs, subs, interestingStrings);
            StringInferenceTestUtilities.TestProduct(hello, any, hello, interestingStrings);
            StringInferenceTestUtilities.TestProduct(any, hello, hello, interestingStrings);
            StringInferenceTestUtilities.TestProduct(world, any, world, interestingStrings);
            StringInferenceTestUtilities.TestProduct(any, world, world, interestingStrings);
            StringInferenceTestUtilities.TestProduct(any, any, any, interestingStrings);
            StringInferenceTestUtilities.TestProduct(hello, world, StringDistribution.Zero());
            StringInferenceTestUtilities.TestProduct(helloworld, world, StringDistribution.Zero());
        }

        /// <summary>
        /// Tests distribution product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product2()
        {
            var ab = StringDistribution.ZeroOrMore(DiscreteChar.OneOf('a', 'b'));
            var a = StringDistribution.ZeroOrMore('a');
            var prod = ab.Product(a);
            StringInferenceTestUtilities.TestProbability(prod, 1.0, string.Empty, "a", "aa", "aaa");
            StringInferenceTestUtilities.TestProbability(prod, 0.0, "b", "bb", "ab", "ba");
        }

        /// <summary>
        /// Tests distribution product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product3()
        {
            StringAutomaton weights1 = StringAutomaton.Sum(
                StringAutomaton.ConstantOn(1.0, "a"),
                StringAutomaton.ConstantOn(2.0, "b"),
                StringAutomaton.ConstantOn(4.0, "c"));
            StringAutomaton weights2 = StringAutomaton.Sum(
                StringAutomaton.ConstantOn(2.0, "a"),
                StringAutomaton.ConstantOn(5.0, "b"),
                StringAutomaton.ConstantOn(7.0, "c"));
            StringDistribution dist1 = StringDistribution.FromWeightFunction(weights1);
            StringDistribution dist2 = StringDistribution.FromWeightFunction(weights2);
            StringDistribution product = dist1.Product(dist2);

            StringInferenceTestUtilities.TestProbability(product, 2.0 / 40.0, "a");
            StringInferenceTestUtilities.TestProbability(product, 10.0 / 40.0, "b");
            StringInferenceTestUtilities.TestProbability(product, 28.0 / 40.0, "c");
        }

        /// <summary>
        /// Tests distribution product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product4()
        {
            StringInferenceTestUtilities.TestProduct(StringDistribution.Zero(), StringDistribution.OneOf("a", "c"), StringDistribution.Zero(), "a", "c", string.Empty);
            StringInferenceTestUtilities.TestProduct(StringDistribution.Zero(), StringDistribution.String("a"), StringDistribution.Zero(), "a", "c", string.Empty);
            StringInferenceTestUtilities.TestProduct(StringDistribution.OneOf("a", "c"), StringDistribution.Zero(), StringDistribution.Zero(), "a", "c", string.Empty);
            StringInferenceTestUtilities.TestProduct(StringDistribution.String("a"), StringDistribution.Zero(), StringDistribution.Zero(), "a", "c", string.Empty);
            StringInferenceTestUtilities.TestProduct(StringDistribution.Zero(), StringDistribution.Zero(), StringDistribution.Zero(), "a", "c", string.Empty);
            StringInferenceTestUtilities.TestProduct(StringDistribution.Zero(), StringDistribution.Any(), StringDistribution.Zero(), "a", "c", string.Empty);
            StringInferenceTestUtilities.TestProduct(StringDistribution.Any(), StringDistribution.Zero(), StringDistribution.Zero(), "a", "c", string.Empty);
        }

        /// <summary>
        /// Tests distribution product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Product5()
        {
            var uniformThenChar = StringDistribution.Any() + StringDistribution.Char('a');
            var charThenUniform = StringDistribution.Char('a') + StringDistribution.Any();
            var expectedProduct = StringDistribution.OneOf(
                StringDistribution.Char('a') + StringDistribution.Any() + StringDistribution.Char('a'),
                StringDistribution.Char('a'));

            StringInferenceTestUtilities.TestProduct(uniformThenChar, charThenUniform, expectedProduct, string.Empty, "a", "aba", "abaa", "abd");
        }

        /// <summary>
        /// Tests that a group in the left-hand-side automaton shows up in the product.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ProductWithGroups()
        {
            StringDistribution lhsWithoutGroup = StringDistribution.String("ab");

            // add a group to first transition of the start state
            var weightFunctionBuilder = StringAutomaton.Builder.FromAutomaton(lhsWithoutGroup.GetWorkspaceOrPoint());
            var transitionIterator = weightFunctionBuilder.Start.TransitionIterator;
            var transitionWithGroup = transitionIterator.Value;
            transitionWithGroup.Group = 1;
            transitionIterator.Value = transitionWithGroup;

            StringDistribution lhs = StringDistribution.FromWeightFunction(weightFunctionBuilder.GetAutomaton());
            StringDistribution rhs = StringDistribution.OneOf("ab", "ac");
            Assert.True(lhs.GetWorkspaceOrPoint().HasGroup(1));
            Assert.False(rhs.GetWorkspaceOrPoint().UsesGroups);
            var result = StringDistribution.Zero();
            result.SetToProduct(lhs, rhs);
            Assert.True(result.GetWorkspaceOrPoint().HasGroup(1));
        }

        /// <summary>
        /// Tests that a point mass support is being detected correctly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMassDetection()
        {
            StringDistribution s1 = StringDistribution.OneOf("hello", "world", "people");
            StringDistribution s2 = StringDistribution.OneOf("greetings", "people", "animals");
            StringDistribution point1 = s1.Product(s2);
            Assert.True(point1.IsPointMass);
            Assert.Equal("people", point1.Point);

            StringDistribution point2 = StringDistribution.OneOf(new Dictionary<string, double> { { "a", 3.0 }, { "b", 0.0 } });
            Assert.True(point2.IsPointMass);
            Assert.Equal("a", point2.Point);

            StringDistribution point3 = StringDistribution.CaseInvariant("123");
            Assert.True(point3.IsPointMass);
            Assert.Equal("123", point3.Point);

            StringDistribution point4 = StringDistribution.Char('Z');
            Assert.True(point4.IsPointMass);
            Assert.Equal("Z", point4.Point);

            StringDistribution point5 = StringDistribution.OneOf(1.0, StringDistribution.String("!"), 0.0, StringDistribution.Any());
            Assert.True(point5.IsPointMass);
            Assert.Equal("!", point5.Point);

            StringDistribution point6 = StringDistribution.Repeat('@', minTimes: 3, maxTimes: 3);
            Assert.True(point6.IsPointMass);
            Assert.Equal("@@@", point6.Point);

            StringDistribution point7 = StringDistribution.String("hello").Append(StringDistribution.String(" world"));
            Assert.True(point7.IsPointMass);
            Assert.Equal("hello world", point7.Point);

            string point = string.Empty;
            StringAutomaton point8Automaton = StringAutomaton.Empty();
            for (int i = 0; i < 22; ++i)
            {
                const string PointElement = "a";
                point8Automaton.AppendInPlace(StringAutomaton.ConstantOn(1.0, PointElement, PointElement));
                point += PointElement;
            }

            StringDistribution point8 = StringDistribution.FromWeightFunction(point8Automaton);
            Assert.True(point8.IsPointMass);
            Assert.Equal(point, point8.Point);
        }

        /// <summary>
        /// Tests that an empty support is being detected correctly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ZeroDetection()
        {
            StringDistribution dist1 = StringDistribution.OneOf(1.0, StringDistribution.Zero(), 0.0, StringDistribution.Any());
            Assert.True(dist1.IsZero());
            StringInferenceTestUtilities.TestProbability(dist1, 0.0, string.Empty, "a", "bc");

            StringDistribution dist2 = StringDistribution.Capitalized(2, 4).Product(StringDistribution.Any(minLength: 5, maxLength: 7));
            Assert.True(dist2.IsZero());
            StringInferenceTestUtilities.TestProbability(dist2, 0.0, string.Empty, "Abc", "Abcdef");

            StringDistribution dist3 = StringDistribution.Digits(minLength: 3, maxLength: 3).Product(StringDistribution.String("12a"));
            Assert.True(dist3.IsZero());
            StringInferenceTestUtilities.TestProbability(dist3, 0.0, string.Empty, "12a", "1", "2", "666");

            StringDistribution dist4 = StringDistribution.Any(minLength: 1, maxLength: 2).Product(StringDistribution.Any(minLength: 2, maxLength: 3).Product(StringDistribution.Any(minLength: 3, maxLength: 4)));
            Assert.True(dist4.IsZero());
            StringInferenceTestUtilities.TestProbability(dist4, 0.0, string.Empty, "a", "ab", "abc", "abcd");

            StringDistribution dist5 = StringDistribution.Any().Append(StringDistribution.Zero());
            Assert.True(dist5.IsZero());
            StringInferenceTestUtilities.TestProbability(dist5, 0.0, string.Empty, "a", "bc");

            StringDistribution dist6 = StringDistribution.Zero().Append(StringDistribution.OneOf("abc", "def"));
            Assert.True(dist6.IsZero());
            StringInferenceTestUtilities.TestProbability(dist6, 0.0, string.Empty, "a", "bc");
        }

        /// <summary>
        /// Tests that a uniform distribution is being detected correctly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void UniformDetection()
        {
            StringDistribution dist1 = StringDistribution.OneOf(0.0, StringDistribution.Zero(), 1.0, StringDistribution.Any());
            Assert.True(dist1.IsUniform());
            StringInferenceTestUtilities.TestProbability(dist1, 1.0, string.Empty, "a", "bc");

            StringDistribution dist2 = StringDistribution.OneOf(0.3, StringDistribution.Any(), 0.7, StringDistribution.Any());
            Assert.True(dist2.IsUniform());
            StringInferenceTestUtilities.TestProbability(dist1, 1.0, string.Empty, "a", "bc");

            StringDistribution dist3 =
                StringDistribution.OneOf(1.0, StringDistribution.Any(), 2.0, StringDistribution.OneOf(0.1, StringDistribution.Any(), 0.2, StringDistribution.Any()));
            Assert.True(dist3.IsUniform());
            StringInferenceTestUtilities.TestProbability(dist3, 1.0, string.Empty, "a", "bc");
        }

        /// <summary>
        /// Tests a mixture of point masses.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MixtureOfPointMasses()
        {
            var mixture = StringDistribution.OneOf(StringDistribution.String("a"), StringDistribution.String("b"));
            StringInferenceTestUtilities.TestProbability(mixture, 0.5, "a", "b");
        }

        /// <summary>
        /// Tests a mixture of mixtures.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MixtureOfMixtures()
        {
            var dist1 = StringDistribution.OneOf("a", "b");
            StringInferenceTestUtilities.TestProbability(dist1, 0.5, "a", "b");

            var dist2 = StringDistribution.OneOf("c", "d");
            StringInferenceTestUtilities.TestProbability(dist2, 0.5, "c", "d");

            var mixture = StringDistribution.OneOf(dist1, dist2);
            StringInferenceTestUtilities.TestProbability(mixture, 0.25, "a", "b", "c", "d");
        }

        /// <summary>
        /// Tests a mixture of mixtures.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Mixture1()
        {
            var dist1 = StringDistribution.OneOf("a", "b");
            var dist2 = StringDistribution.OneOf("c", "d", "e");
            var mixture = StringDistribution.OneOf(dist1, dist2);

            StringInferenceTestUtilities.TestProbability(mixture, 0.5 / 2, "a", "b");
            StringInferenceTestUtilities.TestProbability(mixture, 0.5 / 3, "c", "d", "e");
        }

        /// <summary>
        /// Tests a mixture.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Mixture2()
        {
            var dist1 = StringDistribution.Any();
            var dist2 = StringDistribution.OneOf("c", "d", "e");
            var mixture = StringDistribution.OneOf(dist1, dist2);
            Assert.False(mixture.IsProper());
            StringInferenceTestUtilities.TestIfIncludes(mixture, "a", "b", "c", "d", "e");
        }

        /// <summary>
        /// Tests a mixture.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Mixture3()
        {
            var unifMix = StringDistribution.Zero();
            Assert.False(unifMix.IsProper());
            unifMix.SetToSum(0.5, StringDistribution.Any(), 0.5, StringDistribution.String("hello"));
            StringInferenceTestUtilities.TestProbability(unifMix, 1.0, "hello");
            StringInferenceTestUtilities.TestProbability(unifMix, 0.5, string.Empty, "something else");
        }

        /// <summary>
        /// Tests replacing a point mass distribution with a uniform one.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMassToUniform()
        {
            var dist = StringDistribution.String("1337");
            Assert.False(dist.IsUniform());
            dist.SetToUniform();
            Assert.True(dist.IsUniform());
        }

        /// <summary>
        /// Tests appending a point mass.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AppendPointMass()
        {
            var dist = StringDistribution.OneOf("13", "313");
            dist.AppendInPlace(StringDistribution.String("37"));
            StringInferenceTestUtilities.TestProbability(dist, 0.5, "1337", "31337");
        }

        /// <summary>
        /// Tests appending to a point mass.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AppendToPointMass()
        {
            var dist1 = StringDistribution.String("x");
            var dist2 = StringDistribution.OneOf("y", "z");
            dist1.AppendInPlace(dist2);
            StringInferenceTestUtilities.TestProbability(dist1, 0.5, "xy", "xz");
        }

        /// <summary>
        /// Tests appending in-place.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AppendInPlace()
        {
            var dist = StringDistribution.OneOf("x", "y");
            StringInferenceTestUtilities.TestProbability(dist, 0.5, "x", "y");
            dist.AppendInPlace(StringDistribution.OneOf("z", "w"));
            StringInferenceTestUtilities.TestProbability(dist, 0.25, "xz", "xw", "yz", "yw");
        }

        /// <summary>
        /// Tests appending point mass to a uniform distribution.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AppendPointMassUniform()
        {
            var unifPlusH = StringDistribution.Any() + StringDistribution.String("h");
            Assert.False(unifPlusH.IsProper());
            StringInferenceTestUtilities.TestProbability(unifPlusH, 1.0, "h", "hh", "advahbdkjshbfjlhh");
            StringInferenceTestUtilities.TestProbability(unifPlusH, 0.0, string.Empty, "jam");
        }

        /// <summary>
        /// Tests repeating a point mass.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void RepeatPointMass()
        {
            var dist1 = StringDistribution.Repeat(StringDistribution.PointMass("ab"), minTimes: 2, maxTimes: 2);
            Assert.Equal(StringDistribution.PointMass("abab"), dist1);

            var dist2 = StringDistribution.Repeat(StringDistribution.PointMass("ab"), minTimes: 1, maxTimes: 3);
            Assert.True(dist2.IsProper());
            StringInferenceTestUtilities.TestProbability(dist2, 1.0 / 3.0, "ab", "abab", "ababab");

            var dist3 = StringDistribution.Repeat(StringDistribution.PointMass("ab"), minTimes: 0, maxTimes: 0);
            Assert.Equal(StringDistribution.PointMass(string.Empty), dist3);
        }

        /// <summary>
        /// Tests distribution repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat1()
        {
            var dist = StringDistribution.Repeat(StringDistribution.OneOf("ab", "cd"), minTimes: 1, maxTimes: 3);
            Assert.True(dist.IsProper());
            StringInferenceTestUtilities.TestProbability(dist, StringInferenceTestUtilities.StringUniformProbability(1, 3, 2), "ab", "cd", "abab", "abcd", "cdabcd", "cdcdcd");
        }

        /// <summary>
        /// Tests distribution repetition.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Repeat2()
        {
            var baseDist = StringDistribution.OneOf("a", "b");
            var dist1 = StringDistribution.Repeat(baseDist, minTimes: 1, maxTimes: 3);
            var dist2 = StringDistribution.Repeat(DiscreteChar.OneOf('a', 'b'), minTimes: 1, maxTimes: 3);
            Assert.Equal(dist2, dist1);
        }

        /// <summary>
        /// Tests converting a point mass distribution to a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMassToString()
        {
            StringDistribution point = StringDistribution.PointMass("ab\"");
            Assert.Equal("ab\"", point.ToString(SequenceDistributionFormats.Friendly));
            Assert.Equal("ab\"", point.ToString(SequenceDistributionFormats.Regexp));
            Assert.Equal(
@"digraph finite_state_machine {"                        + Environment.NewLine +   
@"  rankdir=LR;"                                         + Environment.NewLine +
@"  node [shape = doublecircle; label = ""0\nE=0""]; N0" + Environment.NewLine +
@"  node [shape = circle; label = ""1\nE=0""]; N1"       + Environment.NewLine +
@"  node [shape = circle; label = ""2\nE=0""]; N2"       + Environment.NewLine +
@"  node [shape = circle; label = ""3\nE=1""]; N3"       + Environment.NewLine +
@"  N0 -> N1 [ label = ""W=1\na"" ];"                    + Environment.NewLine +
@"  N1 -> N2 [ label = ""W=1\nb"" ];"                    + Environment.NewLine +
@"  N2 -> N3 [ label = ""W=1\n\"""" ];"                  + Environment.NewLine +
@"}"                                                     + Environment.NewLine
,
                point.ToString(SequenceDistributionFormats.GraphViz));
        }

        /// <summary>
        /// Tests that converting a mixture to a string preserves the order of the mixture components.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MixtureToStringPreservesOrder()
        {
            StringDistribution mixture = StringDistribution.OneOf("John", "Tom", "Sam");
            Assert.Equal("[John|Tom|Sam]", mixture.ToString());
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void EnumerateSupport()
        {
            Assert.Equal(string.Empty, StringDistribution.Empty().EnumerateSupport().Single());
            Assert.Equal("12345", StringDistribution.String("12345").EnumerateSupport().Single());

            var array = new[] { "aa", "aabb", "aac" };
            Assert.Equal(array, StringDistribution.OneOf(array).EnumerateSupport().ToArray());

            Assert.Equal(10, StringDistribution.Repeat('z', 1, 10).EnumerateSupport().Count());

            var diamondResults = new[] { "abd", "acd" };
            var diamond = StringDistribution.String("a") + StringDistribution.OneOf(StringDistribution.String("b"), StringDistribution.String("c")) + StringDistribution.String("d");
            Assert.Equal(diamondResults, diamond.EnumerateSupport().ToArray());
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void EnumerateSupportThrowsOnLoops()

        {
            Assert.Throws<NotSupportedException>(() =>
            {

                StringDistribution.OneOf(StringDistribution.String("zzz"), StringDistribution.Repeat('z')).EnumerateSupport().ToArray();

            });
        }

        #region Sampling tests

        /// <summary>
        /// Tests sampling from a distribution with a finite support.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SampleFiniteSupport()
        {
            Rand.Restart(69);

            StringDistribution dist = StringDistribution.OneOf("a", "ab").Append(StringDistribution.OneOf("c", "bc"));
            const int SampleCount = 10000;
            int[] sampleCounts = new int[3];
            for (int i = 0; i < SampleCount; ++i)
            {
                string sample = dist.Sample();
                int sampleIndex = sample == "ac" ? 0 : sample == "abc" ? 1 : 2;
                ++sampleCounts[sampleIndex];
            }

            Assert.Equal(0.25, sampleCounts[0] / (double)SampleCount, 1e-2);
            Assert.Equal(0.5, sampleCounts[1] / (double)SampleCount, 1e-2);
            Assert.Equal(0.25, sampleCounts[2] / (double)SampleCount, 1e-2);
        }

        /// <summary>
        /// Tests sampling from a geometric distribution.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SampleGeometric()
        {
            Rand.Restart(96);

            const double StoppingProbability = 0.7;

            // The length of sequences sampled from this distribution must follow a geometric distribution
            var builder = new StringAutomaton.Builder();
            builder.StartStateIndex = builder.AddState().Index;
            builder.Start.SetEndWeight(Weight.FromValue(StoppingProbability));
            builder.Start.AddTransition('a', Weight.FromValue(1 - StoppingProbability), builder.Start.Index);
            StringDistribution dist = StringDistribution.FromWeightFunction(builder.GetAutomaton());

            var acc = new MeanVarianceAccumulator();
            const int SampleCount = 30000;
            for (int i = 0; i < SampleCount; ++i)
            {
                string sample = dist.Sample();
                acc.Add(sample.Length);
            }

            const double ExpectedMean = (1.0 - StoppingProbability) / StoppingProbability;
            const double ExpectedVariance = (1.0 - StoppingProbability) / (StoppingProbability * StoppingProbability);

            Assert.Equal(ExpectedMean, acc.Mean, 1e-2);
            Assert.Equal(ExpectedVariance, acc.Variance, 1e-2);
        }

        /// <summary>
        /// Tests the behavior of sampling from an improper distribution.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        [Trait("Category", "OpenBug")]
        public void SampleImproper()
        {
            Rand.Restart(96);

            const double probChar = (double)1 / (char.MaxValue + 1);
            const double StoppingProbability = probChar * 0.99;
            var dist = StringDistribution.Any();

            // The length of sequences sampled from this distribution will follow a geometric distribution
            var acc = new MeanVarianceAccumulator();
            const int SampleCount = 30000;
            for (int i = 0; i < SampleCount; ++i)
            {
                string sample = dist.Sample();
                acc.Add(sample.Length);
            }

            const double ExpectedMean = (1.0 - StoppingProbability) / StoppingProbability;
            const double ExpectedVariance = (1.0 - StoppingProbability) / (StoppingProbability * StoppingProbability);

            Assert.Equal(ExpectedMean, acc.Mean, 1e-2);
            Assert.Equal(ExpectedVariance, acc.Variance, 1e-2);
        }

        #endregion

        #endregion

        #region Factory method tests

        /// <summary>
        /// Tests <see cref="StringDistribution.Zero"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Zero()
        {
            var zero = StringDistribution.Zero();
            Assert.False(zero.IsUniform());
            Assert.False(zero.IsPointMass);
            Assert.False(zero.IsProper());
            StringInferenceTestUtilities.TestProbability(zero, 0.0, "hello", "!", string.Empty);
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.Any"/> and <see cref="StringDistribution.Uniform"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Uniform()
        {
            var unif1 = StringDistribution.Any();
            var unif2 = StringDistribution.Uniform();
            Assert.True(unif1.IsUniform());
            Assert.True(unif2.IsUniform());
            Assert.False(unif1.IsProper());
            Assert.False(unif2.IsProper());
            StringInferenceTestUtilities.TestProbability(unif1, 1.0, "hello", string.Empty);
            StringInferenceTestUtilities.TestProbability(unif2, 1.0, "hello", string.Empty);
        }

        /// <summary>
        /// Tests <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}.ZeroOrMore(TThis, int?)"/>
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void UniformOf()
        {
            var unif1 = StringDistribution.ZeroOrMore(DiscreteChar.Lower());
            Assert.False(unif1.IsUniform());
            Assert.False(unif1.IsProper());
            StringInferenceTestUtilities.TestProbability(unif1, 1.0, "hello", "a", string.Empty);
            StringInferenceTestUtilities.TestProbability(unif1, 0.0, "123", "!", "Abc");

            // Test if non-uniform element distribution does not affect the outcome
            Vector probs = DiscreteChar.Digit().GetProbs();
            probs['1'] = 0;
            probs['2'] = 0.3;
            probs['3'] = 0.0001;
            var unif2 = StringDistribution.ZeroOrMore(DiscreteChar.FromVector(probs));
            StringInferenceTestUtilities.TestProbability(unif2, 1.0, "0", "234", string.Empty);
            StringInferenceTestUtilities.TestProbability(unif2, 0.0, "1", "231", "!", "Abc");
        }

        /// <summary>
        /// Tests <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}.OneOf(IEnumerable{KeyValuePair{TSequence, double}})"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void FromProbabilities()
        {
            var probabilities = new Dictionary<string, double> { { "abc", 3.0 }, { "b", 2.0 }, { "bc", 5.0 } };
            var dist = StringDistribution.OneOf(probabilities);
            StringInferenceTestUtilities.TestProbability(dist, 0.3, "abc");
            StringInferenceTestUtilities.TestProbability(dist, 0.2, "b");
            StringInferenceTestUtilities.TestProbability(dist, 0.5, "bc");
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.Empty"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Empty()
        {
            var empty = StringDistribution.Empty();
            StringInferenceTestUtilities.TestProbability(empty, 1.0, string.Empty);
            StringInferenceTestUtilities.TestProbability(empty, 0.0, "something");
            Assert.True(empty.IsPointMass);
            Assert.Equal(string.Empty, empty.Point);
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.PointMass"/> and <see cref="StringDistribution.String"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void PointMass()
        {
            const string Point = "hello world";
            StringDistribution pointMass1 = StringDistribution.String(Point);
            StringDistribution pointMass2 = StringDistribution.PointMass(Point);
            Assert.True(pointMass1.IsPointMass);
            Assert.True(pointMass2.IsPointMass);
            Assert.Equal(pointMass1.Point, Point);
            Assert.Equal(pointMass2.Point, Point);
            StringInferenceTestUtilities.TestProbability(pointMass1, 1.0, Point);
            StringInferenceTestUtilities.TestProbability(pointMass2, 1.0, Point);
            StringInferenceTestUtilities.TestProbability(pointMass1, 0.0, string.Empty, "x", "hello", "world");
            StringInferenceTestUtilities.TestProbability(pointMass2, 0.0, string.Empty, "x", "hello", "world");
        }

        /// <summary>
        /// Tests <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}.OneOf(TThis[])"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void OneOf()
        {
            StringDistribution uniformOver1 = StringDistribution.OneOf("hello", "world", "hi");
            StringInferenceTestUtilities.TestProbability(uniformOver1, 1.0 / 3.0, "hello", "world", "hi");
            StringInferenceTestUtilities.TestProbability(uniformOver1, 0.0, string.Empty, "x", "hello world", "h");

            StringDistribution uniformOver2 = StringDistribution.OneOf(string.Empty, "a", "aa");
            StringInferenceTestUtilities.TestProbability(uniformOver2, 1.0 / 3.0, string.Empty, "a", "aa");
            StringInferenceTestUtilities.TestProbability(uniformOver2, 0.0, "x", "hello world", "h", "aaa");
        }

        /// <summary>
        /// Test factory methods that create distributions with string length constraints.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void LengthBounds()
        {
            var lengthDist1 = StringDistribution.Any(minLength: 1, maxLength: 3);
            Assert.True(lengthDist1.IsProper());
            StringInferenceTestUtilities.TestProbability(lengthDist1, StringInferenceTestUtilities.StringUniformProbability(1, 3, 65536), "a", "aa", "aaa");
            StringInferenceTestUtilities.TestProbability(lengthDist1, 0.0, string.Empty, "aaaa");

            var lengthDist2 = StringDistribution.Repeat(DiscreteChar.OneOf('a', 'b'), minTimes: 1, maxTimes: 3);
            Assert.True(lengthDist2.IsProper());
            StringInferenceTestUtilities.TestProbability(lengthDist2, StringInferenceTestUtilities.StringUniformProbability(1, 3, 2), "a", "ab", "aba");
            StringInferenceTestUtilities.TestProbability(lengthDist2, 0.0, string.Empty, "aaaa", "abab", "cc");

            var lengthDist3 = StringDistribution.Repeat(DiscreteChar.OneOf('a', 'b'), minTimes: 2, maxTimes: 2);
            Assert.True(lengthDist3.IsProper());
            StringInferenceTestUtilities.TestProbability(lengthDist3, StringInferenceTestUtilities.StringUniformProbability(2, 2, 2), "aa", "ab", "ba", "bb");
            StringInferenceTestUtilities.TestProbability(lengthDist3, 0.0, string.Empty, "a", "abab", "cc");

            var minLengthDist = StringDistribution.Any(minLength: 2);
            Assert.False(minLengthDist.IsProper());
            StringInferenceTestUtilities.TestProbability(minLengthDist, 1.0, "aa", "123", "@*(@*&(@)");
            StringInferenceTestUtilities.TestProbability(minLengthDist, 0.0, string.Empty, "a", "!");

            var maxLengthDist = StringDistribution.ZeroOrMore(DiscreteChar.Digit(), maxTimes: 3);
            Assert.True(maxLengthDist.IsProper());
            StringInferenceTestUtilities.TestProbability(maxLengthDist, StringInferenceTestUtilities.StringUniformProbability(0, 3, 10), string.Empty, "1", "32", "432");
            StringInferenceTestUtilities.TestProbability(maxLengthDist, 0.0, "abc", "1234");
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.Char(DiscreteChar)"/>, <see cref="StringDistribution.Char(char)"/>, and
        /// <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}.SingleElement(TElementDistribution)"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Char()
        {
            var charDist1 = StringDistribution.Char('a');
            StringInferenceTestUtilities.TestProbability(charDist1, 1.0, "a");
            StringInferenceTestUtilities.TestProbability(charDist1, 0.0, "aa", string.Empty);

            var charDist2 = StringDistribution.Char(DiscreteChar.InRange('a', 'c'));
            StringInferenceTestUtilities.TestProbability(charDist2, 1.0 / 3.0, "a", "b", "c");
            StringInferenceTestUtilities.TestProbability(charDist2, 0.0, "ab", string.Empty);

            Vector charProbs3 = PiecewiseVector.Zero(char.MaxValue + 1);
            charProbs3['a'] = 0.1;
            charProbs3['b'] = 0.9;
            var charDist3 = StringDistribution.SingleElement(DiscreteChar.FromVector(charProbs3));
            StringInferenceTestUtilities.TestProbability(charDist3, 0.1, "a");
            StringInferenceTestUtilities.TestProbability(charDist3, 0.9, "b");
            StringInferenceTestUtilities.TestProbability(charDist3, 0.0, "c", "ab", string.Empty);
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.CaseInvariant"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CaseInvariant()
        {
            var caseInvariantDist = StringDistribution.CaseInvariant("0aBC");
            Assert.True(caseInvariantDist.IsProper());
            StringInferenceTestUtilities.TestProbability(caseInvariantDist, 1.0 / 8.0, "0aBC", "0abc", "0Abc");
            StringInferenceTestUtilities.TestProbability(caseInvariantDist, 0.0, "0aB", string.Empty);
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.Lower"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Lower()
        {
            int lowercaseCharacterCount = DiscreteChar.Lower().GetProbs().Count(p => p > 0);

            var lowercaseAutomaton1 = StringDistribution.Lower(minLength: 1, maxLength: 2);
            Assert.True(lowercaseAutomaton1.IsProper());
            StringInferenceTestUtilities.TestProbability(lowercaseAutomaton1, StringInferenceTestUtilities.StringUniformProbability(1, 2, lowercaseCharacterCount), "a", "bc");
            StringInferenceTestUtilities.TestProbability(lowercaseAutomaton1, 0.0, "abc", "BC", "A", string.Empty);

            var lowercaseAutomaton2 = StringDistribution.Lower(minLength: 2);
            Assert.False(lowercaseAutomaton2.IsProper());
            StringInferenceTestUtilities.TestProbability(lowercaseAutomaton2, 1.0, "bc", "abvhrbfijbr");
            StringInferenceTestUtilities.TestProbability(lowercaseAutomaton2, 0.0, "a", "BC", "adasdADNdej", string.Empty);
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.Upper"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Upper()
        {
            int uppercaseCharacterCount = DiscreteChar.Upper().GetProbs().Count(p => p > 0);

            var uppercaseAutomaton1 = StringDistribution.Upper(minLength: 1, maxLength: 2);
            Assert.True(uppercaseAutomaton1.IsProper());
            StringInferenceTestUtilities.TestProbability(uppercaseAutomaton1, StringInferenceTestUtilities.StringUniformProbability(1, 2, uppercaseCharacterCount), "A", "BC");
            StringInferenceTestUtilities.TestProbability(uppercaseAutomaton1, 0.0, "ABC", "bc", "a", string.Empty);

            var uppercaseAutomaton2 = StringDistribution.Upper(minLength: 2);
            Assert.False(uppercaseAutomaton2.IsProper());
            StringInferenceTestUtilities.TestProbability(uppercaseAutomaton2, 1.0, "BC", "HFJLHFLJN");
            StringInferenceTestUtilities.TestProbability(uppercaseAutomaton2, 0.0, "A", "bc", "JDFJjjlkJ", string.Empty);
        }

        /// <summary>
        /// Tests <see cref="StringDistribution.Capitalized"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void Capitalized()
        {
            int lowercaseCharacterCount = DiscreteChar.Lower().GetProbs().Count(p => p > 0);
            int uppercaseCharacterCount = DiscreteChar.Upper().GetProbs().Count(p => p > 0);

            var capitalizedAutomaton1 = StringDistribution.Capitalized(minLength: 3, maxLength: 5);
            Assert.True(capitalizedAutomaton1.IsProper());
            StringInferenceTestUtilities.TestProbability(
                capitalizedAutomaton1,
                StringInferenceTestUtilities.StringUniformProbability(2, 4, lowercaseCharacterCount) / uppercaseCharacterCount,
                "Abc",
                "Bcde",
                "Abcde");
            StringInferenceTestUtilities.TestProbability(capitalizedAutomaton1, 0.0, "A", "abc", "Ab", "Abcdef", string.Empty);

            var capitalizedAutomaton2 = StringDistribution.Capitalized(minLength: 3);
            Assert.False(capitalizedAutomaton2.IsProper());
            StringInferenceTestUtilities.TestProbability(capitalizedAutomaton2, 1.0, "Abc", "Bcde", "Abcde", "Abfjrhfjlrl");
            StringInferenceTestUtilities.TestProbability(capitalizedAutomaton2, 0.0, "A", "abc", "Ab", string.Empty);
        }

        #endregion
    }
}
