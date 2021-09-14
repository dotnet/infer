// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Models;

    /// <summary>
    /// Tests for <see cref="StringConcatOp"/>.
    /// </summary>
    public class StringConcatOpTests
    {
        /// <summary>
        /// The tolerance used when comparing probabilities.
        /// </summary>
        private const double ValueEps = 1e-14;

        /// <summary>
        /// Tests message operators for the string concatenation factor directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageOperatorsTest()
        {
            //// Messages to concatenation results

            StringDistribution concat1 = StringConcatOp.ConcatAverageConditional(StringDistribution.String("ab"), StringDistribution.String("cd"));
            Assert.True(concat1.IsPointMass && concat1.Point == "abcd");

            StringDistribution concat2 = StringConcatOp.ConcatAverageConditional(StringDistribution.Upper(), StringDistribution.String("cd"));
            StringInferenceTestUtilities.TestIfIncludes(concat2, "Acd", "ABcd");
            StringInferenceTestUtilities.TestIfExcludes(concat2, "cd", "Abcd", "ABc");

            StringDistribution concat3 = StringConcatOp.ConcatAverageConditional(StringDistribution.OneOf("a", "ab"), StringDistribution.OneOf("b", string.Empty));
            StringInferenceTestUtilities.TestProbability(concat3, 0.5, "ab");
            StringInferenceTestUtilities.TestProbability(concat3, 0.25, "a", "abb");
            
            //// Messages to the first argument

            StringDistribution str11 = StringConcatOp.Str1AverageConditional(StringDistribution.String("abc"), StringDistribution.OneOf("abc", "bc"));
            StringInferenceTestUtilities.TestProbability(str11, 0.5, string.Empty, "a");
            
            StringDistribution str12 = StringConcatOp.Str1AverageConditional(StringDistribution.String("abc"), StringDistribution.OneOf("abc", "bc", "a", "b"));
            StringInferenceTestUtilities.TestProbability(str12, 0.5, string.Empty, "a");
            
            StringDistribution str13 = StringConcatOp.Str1AverageConditional(StringDistribution.OneOf("aa", "aaa"), StringDistribution.OneOf("a", "aa"));
            StringInferenceTestUtilities.TestProbability(str13, 0.5, "a");
            StringInferenceTestUtilities.TestProbability(str13, 0.25, string.Empty, "aa");
            
            StringDistribution str14 = StringConcatOp.Str1AverageConditional(StringDistribution.OneOf("abc", "abd"), StringDistribution.OneOf("bc", "d"));
            StringInferenceTestUtilities.TestProbability(str14, 0.5, "a", "ab");
            
            StringDistribution str15 = StringConcatOp.Str1AverageConditional(StringDistribution.OneOf("abc", "abd"), StringDistribution.OneOf("bc", "bd"));
            Assert.True(str15.IsPointMass && str15.Point == "a");

            //// Messages to the second argument

            StringDistribution str21 = StringConcatOp.Str2AverageConditional(StringDistribution.String("abc"), StringDistribution.OneOf("abc", "ab"));
            StringInferenceTestUtilities.TestProbability(str21, 0.5, string.Empty, "c");
            
            StringDistribution str22 = StringConcatOp.Str2AverageConditional(StringDistribution.String("abc"), StringDistribution.OneOf("abc", "ab", "c", "b"));
            StringInferenceTestUtilities.TestProbability(str22, 0.5, string.Empty, "c");
            
            StringDistribution str23 = StringConcatOp.Str2AverageConditional(StringDistribution.OneOf("aa", "aaa"), StringDistribution.OneOf("a", "aa"));
            StringInferenceTestUtilities.TestProbability(str23, 0.5, "a");
            StringInferenceTestUtilities.TestProbability(str23, 0.25, string.Empty, "aa");
            
            StringDistribution str24 = StringConcatOp.Str2AverageConditional(StringDistribution.OneOf("abc", "dbc"), StringDistribution.OneOf("ab", "d"));
            StringInferenceTestUtilities.TestProbability(str24, 0.5, "c", "bc");
            
            StringDistribution str25 = StringConcatOp.Str2AverageConditional(StringDistribution.OneOf("abc", "dbc"), StringDistribution.OneOf("ab", "db"));
            Assert.True(str25.IsPointMass && str25.Point == "c");

            //// Evidence messages

            const double EvidenceEps = 1e-6;
            Assert.Equal(0.0, StringConcatOp.LogEvidenceRatio(StringDistribution.Any(minLength: 1, maxLength: 5), StringDistribution.Capitalized(minLength: 2, maxLength: 2), StringDistribution.Upper()), EvidenceEps);
            Assert.Equal(
                Math.Log(1.0 / 3.0),
                StringConcatOp.LogEvidenceRatio("aaba", StringDistribution.OneOf("a", "aa"), StringDistribution.OneOf("a", "ba", "aba")),
                EvidenceEps);
            Assert.True(double.IsNegativeInfinity(
                StringConcatOp.LogEvidenceRatio("aaba", StringDistribution.OneOf("a", "aa"), StringDistribution.OneOf("a", "bd", "abd"))));

            //// Incompatible message parameters

            Assert.True(StringConcatOp.Str1AverageConditional(StringDistribution.OneOf("abc", "abd"), StringDistribution.OneOf("ab", "b")).IsZero());
            Assert.True(StringConcatOp.Str2AverageConditional(StringDistribution.OneOf("abc", "abd"), StringDistribution.OneOf("bc", "b")).IsZero());
        }

        /// <summary>
        /// Tests string concatenation factor in forward direction with uncertain inputs.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferConcatenationFromArgumentsTest()
        {
            var str1 = Variable.Random(StringDistribution.OneOf("No man ", "Man "));
            var str2 = Variable.Random(StringDistribution.OneOf("is an island", "is not an island"));
            var s = str1 + str2;
            var engine = new InferenceEngine();
            var posteriorOverS = engine.Infer<StringDistribution>(s);
            StringInferenceTestUtilities.TestProbability(
                posteriorOverS, 0.25, "No man is an island", "Man is an island", "Man is not an island", "No man is not an island");
            StringInferenceTestUtilities.TestProbability(posteriorOverS, 0.0, "No man ", "Man ", "is an island", "is not an island");
        }

        /// <summary>
        /// Tests string concatenation factor in backward direction with observed output.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferArgumentsFromConcatenationTest1()
        {
            var str1 = Variable.StringUpper(minLength: 0);
            var str2 = Variable.StringLower(minLength: 0);
            var s = str1 + str2;
            s.ObservedValue = "ABc";
            var engine = new InferenceEngine();
            var posteriorOverStr1 = engine.Infer<StringDistribution>(str1);
            Assert.True(posteriorOverStr1.IsPointMass);
            Assert.Equal("AB", posteriorOverStr1.Point);
        }

        /// <summary>
        /// Tests string concatenation factor in backward direction with observed output.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferArgumentsFromConcatenationTest2()
        {
            var str1 = Variable.StringUpper(minLength: 0);
            var str2 = Variable.StringLower(minLength: 0);
            var str3 = Variable.StringUpper(minLength: 0);
            var s = str1 + str2 + str3;
            s.ObservedValue = "ABC";
            var engine = new InferenceEngine();

            var posteriorOverStr1 = engine.Infer<StringDistribution>(str1);
            StringInferenceTestUtilities.TestIfIncludes(posteriorOverStr1, string.Empty, "A", "AB", "ABC");
            StringInferenceTestUtilities.TestIfExcludes(posteriorOverStr1, "B", "BC", "C");

            var posteriorOverStr2 = engine.Infer<StringDistribution>(str2);
            Assert.True(posteriorOverStr2.IsPointMass);
            Assert.Equal(string.Empty, posteriorOverStr2.Point);

            var posteriorOverStr3 = engine.Infer<StringDistribution>(str3);
            StringInferenceTestUtilities.TestIfIncludes(posteriorOverStr3, "ABC", "BC", "C", string.Empty);
            StringInferenceTestUtilities.TestIfExcludes(posteriorOverStr3, "A", "AB", "B");
        }

        /// <summary>
        /// Tests string concatenation factor in backward direction with uncertain output.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferArgumentsFromConcatenationTest3()
        {
            var str1 = Variable.StringUpper(minLength: 0);
            var str2 = Variable.StringLower(minLength: 0);
            var str3 = Variable.StringUpper(minLength: 1, maxLength: 1);
            var s = str1 + str2 + str3;
            Variable.ConstrainEqualRandom(s, StringDistribution.OneOf("AB", "aB"));
            var engine = new InferenceEngine();

            var posteriorOverStr1 = engine.Infer<StringDistribution>(str1);
            StringInferenceTestUtilities.TestIfIncludes(posteriorOverStr1, string.Empty, "A");
            StringInferenceTestUtilities.TestIfExcludes(posteriorOverStr1, "AB", "B");

            var posteriorOverStr2 = engine.Infer<StringDistribution>(str2);
            StringInferenceTestUtilities.TestIfIncludes(posteriorOverStr2, string.Empty, "a");

            var posteriorOverStr3 = engine.Infer<StringDistribution>(str3);
            Assert.True(posteriorOverStr3.IsPointMass);
            Assert.Equal("B", posteriorOverStr3.Point);
        }

        /// <summary>
        /// Tests string concatenation factor with gates, factor output is observed.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GatedModelObservedOutputTest1()
        {
            Variable<bool> selector = Variable.Bernoulli(0.5);
            Variable<string> concat = Variable.New<string>();
            
            using (Variable.If(selector))
            {
                Variable<string> arg1 = Variable.Random(StringDistribution.OneOf("ab", "abc"));
                Variable<string> arg2 = Variable.Random(StringDistribution.OneOf("cd", "d"));
                concat.SetTo(arg1 + arg2);
            }
            
            using (Variable.IfNot(selector))
            {
                Variable<string> arg1 = "ab";
                Variable<string> arg2 = Variable.Random(StringDistribution.OneOf("cd", "d", "e"));
                concat.SetTo(arg1 + arg2);
            }

            concat.ObservedValue = "abcd";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(0.6, selectorPosterior.GetProbTrue(), ValueEps);
        }

        /// <summary>
        /// Tests string concatenation factor with gates, factor output is observed.
        /// First gate is very unlikely because of <see cref="StringDistribution.Lower"/>.
        /// </summary>
        /// <remarks>This test fails because it reflects the way we want gates with uniform internals to behave, not the way they currently do.</remarks>
        [Fact]
        [Trait("Category", "BadTest")]
        [Trait("Category", "StringInference")]
        public void GatedModelObservedOutputTest2()
        {
            Variable<bool> selector = Variable.Bernoulli(0.5);
            Variable<string> concat = Variable.New<string>();
            
            using (Variable.If(selector))
            {
                Variable<string> arg1 = Variable.StringLower();
                Variable<string> arg2 = Variable.Random(StringDistribution.OneOf("cd", "d"));
                concat.SetTo(arg1 + arg2);
            }
            
            using (Variable.IfNot(selector))
            {
                Variable<string> arg1 = "ab";
                Variable<string> arg2 = Variable.Random(StringDistribution.OneOf("cd", "d", "e"));
                concat.SetTo(arg1 + arg2);
            }

            concat.ObservedValue = "abcd";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(0.0, selectorPosterior.GetProbTrue());
        }

        /// <summary>
        /// Tests string concatenation factor with gates, factor output is uncertain.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GatedModelUncertainOutputTest()
        {
            Variable<bool> selector = Variable.Bernoulli(0.5);
            Variable<string> concat = Variable.New<string>();
            
            using (Variable.If(selector))
            {
                Variable<string> arg1 = Variable.Random(StringDistribution.OneOf("ab", "abc"));
                Variable<string> arg2 = Variable.Random(StringDistribution.OneOf("cd", "d"));
                concat.SetTo(arg1 + arg2);
            }
            
            using (Variable.IfNot(selector))
            {
                Variable<string> arg1 = "ab";
                Variable<string> arg2 = Variable.Random(StringDistribution.OneOf("cd", "d", "e"));
                concat.SetTo(arg1 + arg2);
            }

            Variable.ConstrainEqualRandom(concat, StringDistribution.OneOf("abcd", "abd"));

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(9.0 / 17.0, selectorPosterior.GetProbTrue());
        }
    }
}
