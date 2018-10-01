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
    /// Tests for <see cref="StringFromArrayOp"/>.
    /// </summary>
    public class StringFromArrayOpTests
    {
        /// <summary>
        /// A tolerance for comparing probability values.
        /// </summary>
        private const double Eps = 1e-6;
        
        /// <summary>
        /// Tests message operators directly.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void MessageOperatorsTest()
        {
            var str1 = StringFromArrayOp.StrAverageConditional(new[] { DiscreteChar.PointMass('a'), DiscreteChar.OneOf('b', 'c'), DiscreteChar.OneOf('d', 'e') });
            Assert.Equal(StringDistribution.OneOf("abd", "abe", "acd", "ace"), str1, Eps);

            var str2 = StringFromArrayOp.StrAverageConditional(new DiscreteChar[0]);
            Assert.Equal(StringDistribution.Empty(), str2, Eps);

            var chars1 = StringFromArrayOp.CharactersAverageConditional(
                StringDistribution.OneOf("ab", "cd"),
                new[] { DiscreteChar.PointMass('a'), DiscreteChar.Uniform() },
                new DiscreteChar[2]);
            Assert.Equal(DiscreteChar.OneOf('a', 'c'), chars1[0], Eps);
            Assert.Equal(DiscreteChar.PointMass('b'), chars1[1], Eps);

            var chars2 = StringFromArrayOp.CharactersAverageConditional(
                StringDistribution.OneOf("ab", "ac"),
                new[] { DiscreteChar.Uniform(), DiscreteChar.Uniform() },
                new DiscreteChar[2]);
            Assert.Equal(DiscreteChar.PointMass('a'), chars2[0], Eps);
            Assert.Equal(DiscreteChar.OneOf('b', 'c'), chars2[1], Eps);

            var chars3 = StringFromArrayOp.CharactersAverageConditional(
                StringDistribution.OneOf("ab", "ac", "bc"),
                new[] { DiscreteChar.Uniform(), DiscreteChar.Uniform() },
                new DiscreteChar[2]);
            Assert.Equal(2.0 / 3.0, chars3[0]['a'], Eps);
            Assert.Equal(1.0 / 3.0, chars3[0]['b'], Eps);
            Assert.Equal(1.0 / 3.0, chars3[1]['b'], Eps);
            Assert.Equal(2.0 / 3.0, chars3[1]['c'], Eps);

            var chars4 = StringFromArrayOp.CharactersAverageConditional(
                StringDistribution.OneOf("ab", "cde"),
                new[] { DiscreteChar.Uniform(), DiscreteChar.Uniform() },
                new DiscreteChar[2]);
            Assert.Equal(DiscreteChar.PointMass('a'), chars4[0], Eps);
            Assert.Equal(DiscreteChar.PointMass('b'), chars4[1], Eps);

            var chars5 = StringFromArrayOp.CharactersAverageConditional(
                StringDistribution.OneOf("ab", "cb", "ae", "ax"),
                new[] { DiscreteChar.PointMass('a'), DiscreteChar.PointMass('b') },
                new DiscreteChar[2]);
            Assert.Equal(DiscreteChar.OneOf('a', 'c'), chars5[0], Eps);
            Assert.Equal(DiscreteChar.OneOf('b', 'e', 'x'), chars5[1], Eps);

            var chars6 = StringFromArrayOp.CharactersAverageConditional(
                StringDistribution.OneOf("abcd", "accd", "acce"),
                new[] { DiscreteChar.Uniform(), DiscreteChar.Uniform(), DiscreteChar.Uniform(), DiscreteChar.PointMass('d') },
                new DiscreteChar[4]);
            Assert.Equal(DiscreteChar.PointMass('a'), chars6[0], Eps);
            Assert.Equal(DiscreteChar.OneOf('b', 'c'), chars6[1], Eps);
            Assert.Equal(DiscreteChar.PointMass('c'), chars6[2], Eps);
            Assert.Equal(2.0 / 3.0, chars6[3]['d'], Eps);
            Assert.Equal(1.0 / 3.0, chars6[3]['e'], Eps);
        }
    }
}
