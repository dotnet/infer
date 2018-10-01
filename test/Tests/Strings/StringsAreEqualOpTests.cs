// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Models;
    
    /// <summary>
    /// Tests for <see cref="StringsAreEqualOp"/>.
    /// </summary>
    public class StringsAreEqualOpTests
    {
        /// <summary>
        /// Tests string equality factor with uncertain arguments.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferEqualityFromArgumentsTest()
        {
            Variable<string> str1 = Variable.Random(StringDistribution.OneOf("a", "b"));
            Variable<string> str2 = Variable.Random(StringDistribution.OneOf("a", "b", "c"));
            Variable<bool> equal = str1 == str2;

            var engine = new InferenceEngine();
            
            var equalPosterior = engine.Infer<Bernoulli>(equal);
            Assert.Equal(1.0 / 3.0, equalPosterior.GetProbTrue(), 1e-6);
        }

        /// <summary>
        /// Constrains the arguments of AreEqual by observing the result to be <see langword="true"/>.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void InferArgumentsFromEqualityTest()
        {
            Variable<string> str1 = Variable.Random(StringDistribution.OneOf("a", "b"));
            Variable<string> str2 = Variable.Random(StringDistribution.OneOf("a", "b", "c"));
            Variable<bool> equal = str1 == str2;
            equal.ObservedValue = true;

            var engine = new InferenceEngine();
            var str2Posterior = engine.Infer<StringDistribution>(str2);
            StringInferenceTestUtilities.TestProbability(str2Posterior, 0.5, "a", "b");
        }

        /// <summary>
        /// Constrains the arguments of AreEqual by observing the result to be <see langword="false"/>.
        /// Automata should be able to represent languages like 'all except "a" and "b"' to handle this case.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "StringInference")]
        public void InferArgumentsFromInequalityTest()
        {
            Variable<string> str1 = Variable.Random(StringDistribution.OneOf("a", "b"));
            Variable<string> str2 = Variable.Random(StringDistribution.OneOf("a", "b", "c"));
            Variable<bool> equal = str1 == str2;
            equal.ObservedValue = false;

            var engine = new InferenceEngine();
            
            var str2Posterior = engine.Infer<StringDistribution>(str2);
            StringInferenceTestUtilities.TestProbability(str2Posterior, 0.25, "a", "b");
            StringInferenceTestUtilities.TestProbability(str2Posterior, 0.5, "c");
        }
    }
}
