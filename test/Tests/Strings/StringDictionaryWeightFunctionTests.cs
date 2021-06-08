// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System.Collections.Generic;
    using System.Linq;

    using Xunit;
    using Microsoft.ML.Probabilistic.Distributions.Automata;

    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Tests functionality specific to <see cref="StringDictionaryWeightFunction"/>.
    /// </summary>
    public class StringDictionaryWeightFunctionTests
    {
        static readonly Dictionary<string, double> ZeroDistributionTable = new Dictionary<string, double>();
        static readonly Dictionary<string, double> NormalizedPointMassDistributionTable = new Dictionary<string, double>()
        {
            ["a"] = 1.0
        };
        static readonly Dictionary<string, double> NonNormalizedPointMassDistributionTable = new Dictionary<string, double>()
        {
            ["a"] = 0.5
        };
        static readonly Dictionary<string, double> NormalizedComplexDistributionTable = new Dictionary<string, double>()
        {
            ["aaabbbccc"] = 0.25,
            ["aaadccc"] = 0.125,
            ["aaabfff"] = 0.125,
            ["bbbdccc"] = 0.125,
            ["bbbghfff"] = 0.25,
            ["zzzzzzz"] = 0.125
        };
        static readonly Dictionary<string, double> NonNormalizedComplexDistributionTable = new Dictionary<string, double>()
        {
            ["aaabbbccc"] = 0.125,
            ["aaadccc"] = 0.25,
            ["aaabfff"] = 0.125,
            ["bbbdccc"] = 0.25,
            ["bbbghfff"] = 0.125,
            ["zzzzzzz"] = 0.125,
            ["ggghhjjj"] = 0.25,
            ["ggglljjj"] = 0.125
        };

        /// <summary>
        /// Tests that the automaton returned by <see cref="StringDictionaryWeightFunction.AsAutomaton"/>
        /// defines the same weight function as the instance of <see cref="StringDictionaryWeightFunction"/>
        /// it was called on.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AsAutomatonCorrectness()
        {
            Assert.True(StringDictionaryWeightFunction.FromValues(ZeroDistributionTable).AsAutomaton().IsZero());
            AssertAutomatonEqualsTable(
                NormalizedPointMassDistributionTable,
                StringDictionaryWeightFunction.FromValues(NormalizedPointMassDistributionTable).AsAutomaton());
            AssertAutomatonEqualsTable(
                NonNormalizedPointMassDistributionTable,
                StringDictionaryWeightFunction.FromValues(NonNormalizedPointMassDistributionTable).AsAutomaton());
            AssertAutomatonEqualsTable(
                NormalizedComplexDistributionTable,
                StringDictionaryWeightFunction.FromValues(NormalizedComplexDistributionTable).AsAutomaton());
            AssertAutomatonEqualsTable(
                NonNormalizedComplexDistributionTable,
                StringDictionaryWeightFunction.FromValues(NonNormalizedComplexDistributionTable).AsAutomaton());
        }

        /// <summary>
        /// Tests that the automaton returned by <see cref="StringDictionaryWeightFunction.AsAutomaton"/>
        /// compresses shared prefixes and suffixes and is deterministic.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void AsAutomatonCompression()
        {
            var normalizedAutomaton = StringDictionaryWeightFunction.FromValues(NormalizedComplexDistributionTable).AsAutomaton();
            Assert.True(normalizedAutomaton.IsDeterministic());
            Assert.True(normalizedAutomaton.States.Count <= 23);
            var nonNormalizedAutomaton = StringDictionaryWeightFunction.FromValues(NonNormalizedComplexDistributionTable).AsAutomaton();
            Assert.True(nonNormalizedAutomaton.IsDeterministic());
            Assert.True(nonNormalizedAutomaton.States.Count <= 31);
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void RepeatCorrectness()
        {
            var sourceDict = new Dictionary<string, double>()
            {
                ["a"] = 0.5,
                ["aa"] = 0.5
            };
            var wf = StringDictionaryWeightFunction.FromDistinctValues(sourceDict);
            var repeatedWf = wf.Repeat(0, 3);
            var expectedResultDict = new Dictionary<string, double>()
            {
                [string.Empty] = 1.0,
                ["a"] = 0.5,
                ["aa"] = 0.75, // aa, a+a
                ["aaa"] = 0.625, // aa+a, a+aa, a+a+a
                ["aaaa"] = 0.625, // aa+aa, aa+a+a, a+aa+a, a+a+aa
                ["aaaaa"] = 0.375, // aa+aa+a, aa+a+aa, a+aa+aa
                ["aaaaaa"] = 0.125 // aa+aa+aa
            };
            Assert.Equal(expectedResultDict.Count, repeatedWf.Dictionary.Count);
            foreach (var kvp in expectedResultDict)
            {
                Assert.True(repeatedWf.Dictionary.TryGetValue(kvp.Key, out Weight weight));
                Assert.Equal(kvp.Value, weight.Value, MMath.Ulp1);
            }
        }

        static void AssertAutomatonEqualsTable(Dictionary<string, double> table, StringAutomaton automaton)
        {
            Assert.True(automaton.TryEnumerateSupport(table.Count, out var support));
            var automatonTable = support.ToDictionary(s => s, s => automaton.GetValue(s));
            Assert.Equal(table.Count, automatonTable.Count);
            foreach (var kvp in table)
            {
                Assert.True(automatonTable.TryGetValue(kvp.Key, out var value));
                Assert.Equal(kvp.Value, value, 1e-15);
            }
        }
    }
}
