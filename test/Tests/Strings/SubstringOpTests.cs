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
    /// Tests for <see cref="SubstringOp"/>.
    /// </summary>
    public class SubstringOpTests
    {
        /// <summary>
        /// A tolerance for comparing log-probabilities.
        /// </summary>
        private const double LogProbEps = 1e-8;
        
        /// <summary>
        /// Tests message operators directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageOperatorsTest()
        {
            //// Messages to substring

            StringDistribution substr1 = SubstringOp.SubAverageConditional("abc", 1, 1);
            Assert.True(substr1.IsPointMass && substr1.Point == "b");

            StringDistribution substr2 = SubstringOp.SubAverageConditional("abc", 0, 3);
            Assert.True(substr2.IsPointMass && substr2.Point == "abc");

            StringDistribution substr3 = SubstringOp.SubAverageConditional(StringDistribution.String("abc"), 1, 1);
            Assert.True(substr3.IsPointMass && substr3.Point == "b");

            StringDistribution substr4 = SubstringOp.SubAverageConditional(StringDistribution.String("abc"), 0, 3);
            Assert.True(substr4.IsPointMass && substr4.Point == "abc");

            StringDistribution substr5 = SubstringOp.SubAverageConditional(StringDistribution.Any(), 0, 2);
            StringInferenceTestUtilities.TestIfIncludes(substr5, "ab", "  ", "17");
            StringInferenceTestUtilities.TestIfExcludes(substr5, "abb", " ", "177", string.Empty);

            StringDistribution substr6 = SubstringOp.SubAverageConditional(StringDistribution.OneOf("abc", "abd", "de"), 0, 2);
            StringInferenceTestUtilities.TestProbability(substr6, 2.0 / 3.0, "ab");
            StringInferenceTestUtilities.TestProbability(substr6, 1.0 / 3.0, "de");

            StringDistribution substr7 = SubstringOp.SubAverageConditional(StringDistribution.OneOf("abc", "abd", "de", "d", "c", string.Empty), 0, 2);
            StringInferenceTestUtilities.TestProbability(substr7, 2.0 / 3.0, "ab");
            StringInferenceTestUtilities.TestProbability(substr7, 1.0 / 3.0, "de");

            //// Messages to string

            StringDistribution str1 = SubstringOp.StrAverageConditional("sss", 1, 3);
            StringInferenceTestUtilities.TestIfIncludes(str1, "asss", "asssa", "bsssa");
            StringInferenceTestUtilities.TestIfExcludes(str1, "sss", "assa", "basssa", string.Empty);

            StringDistribution str2 = SubstringOp.StrAverageConditional("sss", 0, 3);
            StringInferenceTestUtilities.TestIfIncludes(str2, "sss", "sssa", "sssab");
            StringInferenceTestUtilities.TestIfExcludes(str2, "asss", "basssa", "ssa", string.Empty);

            StringDistribution str3 = SubstringOp.StrAverageConditional(StringDistribution.String("sss"), 1, 3);
            StringInferenceTestUtilities.TestIfIncludes(str3, "asss", "asssa", "bsssa");
            StringInferenceTestUtilities.TestIfExcludes(str3, "sss", "assa", "basssa", string.Empty);

            StringDistribution str4 = SubstringOp.StrAverageConditional(StringDistribution.String("sss"), 0, 3);
            StringInferenceTestUtilities.TestIfIncludes(str4, "sss", "sssa", "sssab");
            StringInferenceTestUtilities.TestIfExcludes(str4, "asss", "basssa", "ssa", string.Empty);

            StringDistribution str5 = SubstringOp.StrAverageConditional(StringDistribution.Capitalized(minLength: 3, maxLength: 3), 0, 3);
            StringInferenceTestUtilities.TestIfIncludes(str5, "Bbb", "Baba", "Bbb ab");
            StringInferenceTestUtilities.TestIfExcludes(str5, "BAba", "aaaB", "ABab", "Bb ab", string.Empty);

            StringDistribution str6 = SubstringOp.StrAverageConditional(StringDistribution.Upper(minLength: 0, maxLength: 5), 0, 2);
            StringInferenceTestUtilities.TestIfIncludes(str6, "BBb", "BAba", "BBB ab");
            StringInferenceTestUtilities.TestIfExcludes(str6, "Baba", "aaaB", "aBab", "bb ab", string.Empty);

            //// Evidence messages

            const double EvidenceEps = 1e-6;
            Assert.Equal(0.0, SubstringOp.LogEvidenceRatio(StringDistribution.Any(minLength: 1, maxLength: 5), StringDistribution.Capitalized(2, 2), 1, 2), EvidenceEps);
            Assert.Equal(Math.Log(2.0 / 3.0), SubstringOp.LogEvidenceRatio(StringDistribution.OneOf("baa", "bab", "baab"), "aa", 1, 2), EvidenceEps);
            Assert.True(double.IsNegativeInfinity(SubstringOp.LogEvidenceRatio(StringDistribution.Any(minLength: 1, maxLength: 5), "string", 1, 3)));
            Assert.True(double.IsNegativeInfinity(SubstringOp.LogEvidenceRatio(StringDistribution.Any(minLength: 1, maxLength: 2), "str", 1, 3)));

            //// Incompatible message parameters

            Assert.True(SubstringOp.SubAverageConditional("abc", 1, 3).IsZero());
            Assert.True(SubstringOp.SubAverageConditional(StringDistribution.Any(minLength: 1, maxLength: 2), 1, 3).IsZero());
            Assert.True(SubstringOp.StrAverageConditional("abc", 1, 4).IsZero());
            Assert.True(SubstringOp.StrAverageConditional(StringDistribution.Any(minLength: 1, maxLength: 2), 1, 3).IsZero());
        }

        /// <summary>
        /// Tests substring factor in forward direction.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferSubstringFromStringTest()
        {
            Variable<string> str = Variable.Observed("Hello");
            Variable<string> substr = Variable.Substring(str, 0, 2);
            
            var engine = new InferenceEngine();
            var substrPosterior = engine.Infer<StringDistribution>(substr);
            
            StringInferenceTestUtilities.TestLogProbability(substrPosterior, 0, "He");

            Variable<string> str2 = Variable.Random(StringDistribution.Any());
            Variable<string> substr2 = Variable.Substring(str2, 0, 2);

            var substr2Posterior = engine.Infer<StringDistribution>(substr2);

            StringInferenceTestUtilities.TestIfIncludes(substr2Posterior, "ab", "  ", "dd");
            StringInferenceTestUtilities.TestIfExcludes(substr2Posterior, "abc", " ", string.Empty);
        }

        /// <summary>
        /// Tests substring factor in backward direction.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferStringFromSubstringTest()
        {
            var str = Variable.StringUniform();
            var substr = Variable.Substring(str, 0, 2);
            substr.ObservedValue = "He";

            var engine = new InferenceEngine();
            var strPosterior = engine.Infer<StringDistribution>(str);
            
            StringInferenceTestUtilities.TestIfIncludes(strPosterior, "Hello", "Hell", "He", "He is great");
            StringInferenceTestUtilities.TestIfExcludes(strPosterior, "abc", " ", "H", string.Empty);
        }

        /// <summary>
        /// Tests substring factor with gates, factor output is observed.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GatedModelObservedOutputTest1()
        {
            Variable<bool> selector = Variable.Bernoulli(0.5);
            Variable<string> substr = Variable.New<string>();
            
            using (Variable.If(selector))
            {
                Variable<string> str = Variable.Random(StringDistribution.OneOf("bcad", "bacd", "bca"));
                substr.SetTo(Variable.Substring(str, 0, 2));
            }
            
            using (Variable.IfNot(selector))
            {
                Variable<string> str = Variable.Random(StringDistribution.OneOf("dbc", "abcd", "abcc"));
                substr.SetTo(Variable.Substring(str, 1, 2));
            }

            substr.ObservedValue = "bc";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(0.4, selectorPosterior.GetProbTrue(), LogProbEps);
        }

        /// <summary>
        /// Tests substring factor with gates, factor output is observed.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "StringInference")]
        public void GatedModelObservedOutputTest2()
        {
            Variable<bool> selector = Variable.Bernoulli(0.5);
            Variable<string> substr = Variable.New<string>();
            
            using (Variable.If(selector))
            {
                Variable<string> str = Variable.Random(StringDistribution.OneOf("dbc", "abcd", "abcc"));
                substr.SetTo(Variable.Substring(str, 1, 2));
            }
            
            using (Variable.IfNot(selector))
            {
                Variable<string> str = Variable.StringLower();
                substr.SetTo(Variable.Substring(str, 0, 2));
            }

            substr.ObservedValue = "bc";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(1.0 / (1.0 + (1.0 / (26 * 26))), selectorPosterior.GetProbTrue(), LogProbEps);
        }

        /// <summary>
        /// Tests substring factor with gates, factor output is observed.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void GatedModelUncertainOutputTest()
        {
            Variable<bool> selector = Variable.Bernoulli(0.5);
            Variable<string> substr = Variable.New<string>();
            
            using (Variable.If(selector))
            {
                Variable<string> str = Variable.Random(StringDistribution.OneOf("bcad", "bacd", "bca"));
                substr.SetTo(Variable.Substring(str, 0, 2));
            }
            
            using (Variable.IfNot(selector))
            {
                Variable<string> str = Variable.Random(StringDistribution.OneOf("dbc", "abdd", "a"));
                substr.SetTo(Variable.Substring(str, 1, 2));
            }

            Variable.ConstrainEqualRandom(substr, StringDistribution.OneOf("bc", "ba"));

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(2.0 / 3.0, selectorPosterior.GetProbTrue(), LogProbEps);
        }
    }
}
