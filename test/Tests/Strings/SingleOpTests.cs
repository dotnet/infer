// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System.Linq;
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Tests for <see cref="SingleOp"/>.
    /// </summary>
    public class SingleOpTests
    {
        /// <summary>
        /// A tolerance for comparing probability values.
        /// </summary>
        private const double ProbEps = 1e-8;
        
        /// <summary>
        /// Tests message operators directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageOperatorsTest()
        {
            int lowercaseCharacterCount = DiscreteChar.Lower().GetProbs().Count(p => p > 0);
            int uppercaseCharacterCount = DiscreteChar.Upper().GetProbs().Count(p => p > 0);

            StringDistribution strDist1 = StringDistribution.OneOf("a", "b", "abc", "ab", "bcd", "d", string.Empty);
            DiscreteChar charDist1 = SingleOp.CharacterAverageConditional(strDist1);
            Assert.Equal(1.0 / 3.0, charDist1['a'], ProbEps);
            Assert.Equal(1.0 / 3.0, charDist1['b'], ProbEps);
            Assert.Equal(1.0 / 3.0, charDist1['d'], ProbEps);

            StringDistribution strDist2 = StringDistribution.OneOf(strDist1, StringDistribution.OneOf("b", "d"));
            DiscreteChar charDist2 = SingleOp.CharacterAverageConditional(strDist2);
            Assert.Equal(1.0 / 10.0, charDist2['a'], ProbEps);
            Assert.Equal(4.5 / 10.0, charDist2['b'], ProbEps);
            Assert.Equal(4.5 / 10.0, charDist2['d'], ProbEps);

            StringDistribution strDist3 = StringDistribution.Letters(minLength: 0);
            DiscreteChar charDist3 = SingleOp.CharacterAverageConditional(strDist3);
            Assert.Equal(1.0 / (lowercaseCharacterCount + uppercaseCharacterCount), charDist3['a'], ProbEps);
            Assert.Equal(1.0 / (lowercaseCharacterCount + uppercaseCharacterCount), charDist3['B'], ProbEps);
            Assert.Equal(1.0 / (lowercaseCharacterCount + uppercaseCharacterCount), charDist3['d'], ProbEps);

            StringDistribution strDist4 = StringDistribution.OneOf(strDist3, StringDistribution.Lower(minLength: 0));
            DiscreteChar charDist4 = SingleOp.CharacterAverageConditional(strDist4);
            Assert.Equal(2.0 / (2 * lowercaseCharacterCount + uppercaseCharacterCount), charDist4['a'], ProbEps);
            Assert.Equal(1.0 / (2 * lowercaseCharacterCount + uppercaseCharacterCount), charDist4['B'], ProbEps);
            Assert.Equal(2.0 / (2 * lowercaseCharacterCount + uppercaseCharacterCount), charDist4['d'], ProbEps);

            StringDistribution strDist5 = StringDistribution.String("XX").Append(strDist4);
            Assert.Throws<AllZeroException>(() => SingleOp.CharacterAverageConditional(strDist5));
        }
    }
}
