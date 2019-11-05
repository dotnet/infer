// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions;
    using Assert = Xunit.Assert;

    /// <summary>
    /// Tests for <see cref="DiscreteChar"/>.
    /// </summary>
    public class DiscreteCharTest
    {
        /// <summary>
        /// Runs a set of common distribution tests for <see cref="DiscreteChar"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CharDistribution()
        {
            var rng = DiscreteChar.UniformInRanges("bdgi");
            var unif = DiscreteChar.Uniform();
            var mix = new DiscreteChar();
            mix.SetToSum(0.8, rng, 0.2, unif);

            DistributionTests.DistributionTest(unif, mix, false);
            DistributionTests.PointMassTest(mix, 'b');
            DistributionTests.UniformTest(rng, 'b');
        }

        /// <summary>
        /// Tests the support of the standard character distribution.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void CommonChars()
        {
            TestSupport("digit", DiscreteChar.Digit(), "0123456789", "Ab !Ј");
            TestSupport("lower", DiscreteChar.Lower(), "abcdefghixyz", "ABC0123, ");
            TestSupport("upper", DiscreteChar.Upper(), "ABCDEFGHUXYZ", "abc0123, ");
            TestSupport("letter", DiscreteChar.Letter(), "aBcDeFgGhxyzXYZ", "0123! ,");
            TestSupport("letterOrDigit", DiscreteChar.LetterOrDigit(), "abcABC0123xyzXYZ789", " !Ј$,");
            TestSupport("wordChar", DiscreteChar.WordChar(), "abc_ABC_0123s", " !:.,");
            TestSupport("whitespace", DiscreteChar.Whitespace(), " \t", "abcABC0123,:!");
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void BadRanges()
        {
            try
            {
                var a = DiscreteChar.UniformInRanges("aавz");
            }
            catch
            {
                return;
            }

            Assert.True(false);
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void SampleFromUniformCharDistribution()
        {
            // Make test deterministic
            Rand.Restart(7);

            // 10 chars in distribution
            const int numChars = 10;
            const int numSamples = 100000;
            var dist = DiscreteChar.UniformInRanges("aj");

            var hist = Vector.Zero(numChars);
            for (var i = 0; i < numSamples; ++i)
            {
                hist[dist.Sample() - 'a'] += 1;
            }

            hist = hist * (1.0 / numSamples);
            var unif = Vector.Constant(numChars, 1.0 / numChars);
            var maxDiff = hist.MaxDiff(unif);

            Assert.True(maxDiff < 0.01);
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void ComplementWorks()
        {
            TestComplement(DiscreteChar.PointMass('\0'));
            TestComplement(DiscreteChar.PointMass('a'));
            TestComplement(DiscreteChar.PointMass(char.MaxValue));

            var a = DiscreteChar.PointMass('a');
            var b = DiscreteChar.PointMass('b');

            var ab = default(DiscreteChar);
            ab.SetToSum(1, a, 2, b);

            // 2 subsequent ranges
            Assert.Equal(2, ab.Ranges.Count);
            TestComplement(ab);

            void TestComplement(DiscreteChar dist)
            {
                var uniformDist = dist.Clone();
                uniformDist.SetToPartialUniform();

                var complement = dist.Complement();

                // complement should always be partial uniform
                Assert.True(complement.IsPartialUniform());

                // overlap is zero
                Assert.True(double.IsNegativeInfinity(dist.GetLogAverageOf(complement)));
                Assert.True(double.IsNegativeInfinity(uniformDist.GetLogAverageOf(complement)));

                // union is covers the whole range
                var sum = default(DiscreteChar);
                sum.SetToSum(1, dist, 1, complement);
                sum.SetToPartialUniform();
                Assert.True(sum.IsUniform());

                // Doing complement again will cover the same set of characters
                var complement2 = complement.Complement();
                Assert.Equal(uniformDist, complement2);
            }
        }

        /// <summary>
        /// Tests the support of a character distribution.
        /// </summary>
        /// <param name="distributionName">The name of the distribution.</param>
        /// <param name="distribution">The distribution.</param>
        /// <param name="included">A list of characters that must be included in the support of the distribution.</param>
        /// <param name="excluded">A list of characters that must not be included in the support of the distribution.</param>
        private static void TestSupport(
            string distributionName,
            DiscreteChar distribution,
            IEnumerable<char> included,
            IEnumerable<char> excluded)
        {
            Console.WriteLine(distributionName.PadLeft(12) + ":" + distribution);
            
            foreach (var ch in included)
            {
                Assert.True(!double.IsNegativeInfinity(distribution.GetLogProb(ch)), distribution + " should contain " + ch);
            }
            
            foreach (var ch in excluded)
            {
                Assert.True(double.IsNegativeInfinity(distribution.GetLogProb(ch)), distribution + " should not contain " + ch);
            }
        }
    }
}