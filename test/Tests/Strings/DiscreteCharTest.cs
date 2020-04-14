// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    /// <summary>
    /// Tests for <see cref="DiscreteChar"/>.
    /// </summary>
    public class DiscreteCharTest
    {
        const double Eps = 1e-10;

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

            Xunit.Assert.True(false);
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

            Xunit.Assert.True(maxDiff < 0.01);
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
            Xunit.Assert.Equal(2, ab.Ranges.Count);
            TestComplement(ab);

            void TestComplement(DiscreteChar dist)
            {
                var uniformDist = dist.Clone();
                uniformDist.SetToPartialUniform();

                var complement = dist.Complement();

                // complement should always be partial uniform
                Xunit.Assert.True(complement.IsPartialUniform());

                // overlap is zero
                Xunit.Assert.True(double.IsNegativeInfinity(dist.GetLogAverageOf(complement)));
                Xunit.Assert.True(double.IsNegativeInfinity(uniformDist.GetLogAverageOf(complement)));

                // union is covers the whole range
                var sum = default(DiscreteChar);
                sum.SetToSum(1, dist, 1, complement);
                sum.SetToPartialUniform();
                Xunit.Assert.True(sum.IsUniform());

                // Doing complement again will cover the same set of characters
                var complement2 = complement.Complement();
                Xunit.Assert.Equal(uniformDist, complement2);
            }
        }

        [Fact]
        public void PartialUniformWithLogProbabilityOverride()
        {
            var dist = DiscreteChar.LetterOrDigit();
            var probLetter = Math.Exp(dist.GetLogProb('j'));
            var probNumber = Math.Exp(dist.GetLogProb('5'));

            var logProbabilityOverride = Math.Log(0.7);
            var scaledDist = DiscreteChar.Uniform();
            scaledDist.SetToPartialUniformOf(dist, logProbabilityOverride);
            var scaledLogProbLetter = scaledDist.GetLogProb('j');
            var scaledLogProbNumber = scaledDist.GetLogProb('5');

            Assert.Equal(scaledLogProbLetter, logProbabilityOverride, Eps);
            Assert.Equal(scaledLogProbNumber, logProbabilityOverride, Eps);

            // Check that cache has not been compromised.
            Assert.Equal(probLetter, Math.Exp(dist.GetLogProb('j')), Eps);
            Assert.Equal(probNumber, Math.Exp(dist.GetLogProb('5')), Eps);

            // Check that an exception is thrown if a bad maximumProbability is passed down.
            Xunit.Assert.Throws<ArgumentException>(() =>
            {
                var badDist = DiscreteChar.Uniform();
                badDist.SetToPartialUniformOf(dist, Math.Log(1.2));
            });
        }

        [Fact]
        public void BroadAndNarrow()
        {
            var dist1 = DiscreteChar.Digit();
            Xunit.Assert.True(dist1.IsBroad);

            var dist2 = DiscreteChar.OneOf('1', '3', '5', '6');
            Xunit.Assert.False(dist2.IsBroad);
        }

        [Fact]
        public void HasLogOverride()
        {
            var dist1 = DiscreteChar.LetterOrDigit();
            Xunit.Assert.False(dist1.HasLogProbabilityOverride);

            dist1.SetToPartialUniformOf(dist1, Math.Log(0.9));
            Xunit.Assert.True(dist1.HasLogProbabilityOverride);
        }

        [Fact]
        public void ProductWithLogOverrideBroad()
        {
            for (var i = 0; i < 2; i++)
            {
                var dist1 = DiscreteChar.LetterOrDigit();
                var dist2 = DiscreteChar.Digit();

                var logOverrideProbability = Math.Log(0.9);
                dist1.SetToPartialUniformOf(dist1, logOverrideProbability);
                Xunit.Assert.True(dist1.HasLogProbabilityOverride);
                Xunit.Assert.True(dist2.IsBroad);

                if (i == 1)
                {
                    Util.Swap(ref dist1, ref dist2);
                }

                var dist3 = DiscreteChar.Uniform();
                dist3.SetToProduct(dist1, dist2);

                Xunit.Assert.True(dist3.HasLogProbabilityOverride);
                Assert.Equal(logOverrideProbability, dist3.GetLogProb('5'), Eps);
                Xunit.Assert.True(double.IsNegativeInfinity(dist3.GetLogProb('a')));
            }
        }

        [Fact]
        public void ProductWithLogOverrideNarrow()
        {
            for (var i = 0; i < 2; i++)
            {
                var dist1 = DiscreteChar.LetterOrDigit();
                var dist2 = DiscreteChar.OneOf('1', '3', '5', '6');

                var logOverrideProbability = Math.Log(0.9);
                dist1.SetToPartialUniformOf(dist1, logOverrideProbability);
                Xunit.Assert.True(dist1.HasLogProbabilityOverride);
                Xunit.Assert.False(dist2.IsBroad);

                if (i == 1)
                {
                    Util.Swap(ref dist1, ref dist2);
                }

                var dist3 = DiscreteChar.Uniform();
                dist3.SetToProduct(dist1, dist2);

                Xunit.Assert.False(dist3.HasLogProbabilityOverride);
                Assert.Equal(Math.Log(0.25), dist3.GetLogProb('5'), Eps);
                Xunit.Assert.True(double.IsNegativeInfinity(dist3.GetLogProb('a')));
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
                Xunit.Assert.True(!double.IsNegativeInfinity(distribution.GetLogProb(ch)), distribution + " should contain " + ch);
            }
            
            foreach (var ch in excluded)
            {
                Xunit.Assert.True(double.IsNegativeInfinity(distribution.GetLogProb(ch)), distribution + " should not contain " + ch);
            }
        }
    }
}