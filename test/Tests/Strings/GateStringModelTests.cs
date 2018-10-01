// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using Xunit;
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;

    /// <summary>
    /// A number of tests for string models with gates.
    /// </summary>
    public class GateStringModelTests
    {
        /// <summary>
        /// Defines a mixture of "a" and "b".
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SimpleGatedModelTest1()
        {
            const double SelectorProbabilityTrue = 0.3;

            Variable<string> str  = Variable.New<string>().Named("str");
            Variable<bool> selector = Variable.Bernoulli(SelectorProbabilityTrue).Named("selector");
            
            using (Variable.If(selector))
            {
                str.SetTo("a");
            }
            
            using (Variable.IfNot(selector))
            {
                str.SetTo("b");
            }

            var engine = new InferenceEngine();
            var strPosterior = engine.Infer<StringDistribution>(str);
            StringInferenceTestUtilities.TestProbability(strPosterior, SelectorProbabilityTrue, "a");
            StringInferenceTestUtilities.TestProbability(strPosterior, 1 - SelectorProbabilityTrue, "b");
        }

        /// <summary>
        /// Another way to define a mixture of "a" and "b".
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SimpleGatedModelTest2()
        {
            const double SelectorProbabilityTrue = 0.3;

            Variable<string> str = Variable.Random(StringDistribution.OneOf("a", "b", "c")).Named("str");
            Variable<bool> selector = Variable.Bernoulli(SelectorProbabilityTrue).Named("selector");
            
            using (Variable.If(selector))
            {
                Variable.ConstrainEqual(str, "a");
            }
            
            using (Variable.IfNot(selector))
            {
                Variable.ConstrainEqual(str, "b");
            }

            var engine = new InferenceEngine();
            var strPosterior = engine.Infer<StringDistribution>(str);
            
            StringInferenceTestUtilities.TestProbability(strPosterior, SelectorProbabilityTrue, "a");
            StringInferenceTestUtilities.TestProbability(strPosterior, 1 - SelectorProbabilityTrue, "b");
        }

        /// <summary>
        /// The value "a" should be slightly more probable.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void SimpleGatedModelTest3()
        {
            const double SelectorProbabilityTrue = 0.3;
            string[] options = new[] { "a", "b", "c" };

            Variable<string> str = Variable.Random(StringDistribution.OneOf(options)).Named("str");
            Variable<bool> selector = Variable.Bernoulli(SelectorProbabilityTrue).Named("selector");
            using (Variable.If(selector))
            {
                Variable.ConstrainEqual(str, options[0]);
            }

            var engine = new InferenceEngine();
            var strPosterior = engine.Infer<StringDistribution>(str);

            double normalizer = (SelectorProbabilityTrue / options.Length) + (1 - SelectorProbabilityTrue);
            double nonFirstOptionProbability = (1 - SelectorProbabilityTrue) / (options.Length * normalizer);
            double firstOptionProbability = nonFirstOptionProbability + (SelectorProbabilityTrue / (options.Length * normalizer));

            for (int i = 0; i < options.Length; ++i)
            {
                double expectedProbability = i == 0 ? firstOptionProbability : nonFirstOptionProbability;
                StringInferenceTestUtilities.TestProbability(strPosterior, expectedProbability, options[i]);
            }
        }

        /// <summary>
        /// The first branch is much more likely given the observation.
        /// </summary>
        /// <remarks>This test fails because it reflects the way we want gates with uniform internals to behave, not the way they currently do.</remarks>
        [Fact]
        [Trait("Category", "BadTest")]
        [Trait("Category", "StringInference")]
        public void SimpleGatedModelTest4()
        {
            const double SelectorProbabilityTrue = 0.3;
            const string PreferredOption = "a";

            Variable<string> str = Variable.StringUniform().Named("str");
            Variable<bool> selector = Variable.Bernoulli(SelectorProbabilityTrue).Named("selector");
            
            using (Variable.If(selector))
            {
                Variable.ConstrainEqual(str, PreferredOption);
            }
            
            using (Variable.IfNot(selector))
            {
                Variable.ConstrainEqualRandom(str, StringDistribution.Any());
            }

            str.ObservedValue = PreferredOption;

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(1.0, selectorPosterior.GetProbTrue(), 1e-6);
        }

        /// <summary>
        /// Tests the model where one of the branches is impossible given the data.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ImpossibleBranchTest1()
        {
            Variable<string> str = Variable.New<string>().Named("str");
            Variable<bool> selector = Variable.Bernoulli(0.5).Named("selector");
            
            using (Variable.If(selector))
            {
                var str1 = Variable.StringUniform();
                var str2 = Variable.StringUniform();
                str.SetTo(str1 + " " + str2);
            }
            
            using (Variable.IfNot(selector))
            {
                var str1 = Variable.StringUniform();
                var str2 = Variable.StringUniform();
                str.SetTo(str1 + "-" + str2);
            }

            str.ObservedValue = "ab-c";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Bernoulli>(selector);

            Assert.Equal(1.0, selectorPosterior.GetProbFalse());
        }

        /// <summary>
        /// Tests the model where one of the branches is impossible given the data.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ImpossibleBranchTest2()
        {
            Variable<string> subStr = Variable.New<string>().Named("subStr");
            Variable<int> selector = Variable.DiscreteUniform(2).Named("selector");
            
            using (Variable.Case(selector, 0))
            {
                var str = Variable.StringUniform();
                subStr.SetTo(Variable.Substring(str, 2, 2));
            }
            
            using (Variable.Case(selector, 1))
            {
                var str = Variable.StringUniform();
                subStr.SetTo(Variable.Substring(str, 2, 3));
            }

            subStr.ObservedValue = "ab";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Discrete>(selector);

            Assert.Equal(1.0, selectorPosterior[0]);
        }

        /// <summary>
        /// Tests the model where one of the branches is impossible given the data.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ImpossibleBranchTest3()
        {
            Variable<string> str1 = Variable.StringLower().Named("str1");
            Variable<string> str2 = Variable.StringLower().Named("str2");
            Variable<string> text = Variable.New<string>().Named("text");
            
            Variable<int> selector = Variable.DiscreteUniform(2).Named("selector");
            using (Variable.Case(selector, 0))
            {
                text.SetTo(str1 + Variable.Constant(" ") + str2);
            }

            using (Variable.Case(selector, 1))
            {
                text.SetTo(str1);
            }

            text.ObservedValue = "abc";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Discrete>(selector);
            var str1Posterior = engine.Infer<StringDistribution>(str1);
            var str2Posterior = engine.Infer<StringDistribution>(str2);

            Assert.True(selectorPosterior.IsPointMass && selectorPosterior.Point == 1);
            Assert.True(str1Posterior.IsPointMass && str1Posterior.Point == "abc");
            Assert.Equal(StringDistribution.Lower(), str2Posterior);
        }

        /// <summary>
        /// Tests the model where one of the branches is impossible given the data.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ImpossibleBranchTest4()
        {
            Variable<string> str1 = Variable.StringLower().Named("str1");
            Variable<string> str2 = Variable.StringLower().Named("str2");
            Variable<string> str3 = Variable.StringLower().Named("str3");
            Variable<string> text = Variable.New<string>().Named("text");

            Variable<int> selector = Variable.DiscreteUniform(3).Named("selector");
            using (Variable.Case(selector, 0))
            {
                text.SetTo(str1 + Variable.Constant(" ") + str2 + Variable.Constant(" ") + str3);
            }

            using (Variable.Case(selector, 1))
            {
                text.SetTo(str1 + Variable.Constant(" ") + str3);
            }

            using (Variable.Case(selector, 2))
            {
                text.SetTo(str1);
            }

            text.ObservedValue = "abc def";

            var engine = new InferenceEngine();
            var selectorPosterior = engine.Infer<Discrete>(selector);
            var str1Posterior = engine.Infer<StringDistribution>(str1);
            var str2Posterior = engine.Infer<StringDistribution>(str2);
            var str3Posterior = engine.Infer<StringDistribution>(str3);

            Assert.True(selectorPosterior.IsPointMass && selectorPosterior.Point == 1);
            Assert.True(str1Posterior.IsPointMass && str1Posterior.Point == "abc");
            Assert.Equal(StringDistribution.Lower(), str2Posterior);
            Assert.True(str3Posterior.IsPointMass && str3Posterior.Point == "def");
        }
    }
}
