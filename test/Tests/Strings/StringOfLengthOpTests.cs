// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;

    /// <summary>
    /// Tests for <see cref="StringOfLengthOp"/>.
    /// </summary>
    public class StringOfLengthOpTests
    {
        /// <summary>
        /// Tests message operators directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageOpsTest()
        {
            const double Eps = 1e-6;
            
            StringDistribution str1 = StringOfLengthOp.StrAverageConditional(DiscreteChar.Letter(), 10);
            Assert.Equal(StringDistribution.Repeat(ImmutableDiscreteChar.Letter(), 10, 10), str1);

            StringDistribution str2 = StringOfLengthOp.StrAverageConditional(
                DiscreteChar.PointMass('a'), Discrete.UniformInRange(5, 2, 4));
            Assert.Equal(StringDistribution.OneOf("aa", "aaa", "aaaa"), str2);

            StringDistribution str3 = StringOfLengthOp.StrAverageConditional(
                DiscreteChar.OneOf('a', 'b'), new Discrete(0.1, 0.0, 0.6, 0.3));
            StringInferenceTestUtilities.TestProbability(str3, 0.1, string.Empty);
            StringInferenceTestUtilities.TestProbability(str3, 0.6 / 4, "aa", "ab", "ba", "bb");
            StringInferenceTestUtilities.TestProbability(str3, 0.3 / 8, "aaa", "bbb", "abb", "bab");

            Discrete length1 = StringOfLengthOp.LengthAverageConditional(
                StringDistribution.OneOf("aa", "bbb"), DiscreteChar.PointMass('a'), Discrete.Uniform(10));
            Assert.Equal(Discrete.PointMass(2, 10), length1);

            Discrete length2 = StringOfLengthOp.LengthAverageConditional(
                StringDistribution.OneOf("aab", "ab", "b", "bc"), DiscreteChar.OneOf('a', 'b'), Discrete.Uniform(10));
            Assert.Equal(4.0 / 7.0, length2[1], Eps);
            Assert.Equal(2.0 / 7.0, length2[2], Eps);
            Assert.Equal(1.0 / 7.0, length2[3], Eps);
        }
    }
}
