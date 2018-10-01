// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using Xunit;
    using Assert = AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;

    /// <summary>
    /// Tests <see cref="Variable.GetItem(Variable{string}, Variable{int})"/>
    /// </summary>
    public class StringGetItemOpTests
    {
        /// <summary>
        /// A tolerance for comparing probability values.
        /// </summary>
        private const double ProbEps = 1e-6;
        
        /// <summary>
        /// Tests a model that infers individual characters from a string.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void ItemFromString()
        {
            Variable<string> str1 = Variable.StringOfLength(2, DiscreteChar.Digit());
            Variable<string> str2 = Variable.StringOfLength(3, DiscreteChar.Letter());
            Variable<string> str = str1 + str2;
            
            var engine = new InferenceEngine();

            Assert.Equal(DiscreteChar.Digit(), engine.Infer<DiscreteChar>(Variable.GetItem(str, 0)), ProbEps);
            Assert.Equal(DiscreteChar.Digit(), engine.Infer<DiscreteChar>(Variable.GetItem(str, 1)), ProbEps);
            Assert.Equal(DiscreteChar.Letter(), engine.Infer<DiscreteChar>(Variable.GetItem(str, 2)), ProbEps);
            Assert.Equal(DiscreteChar.Letter(), engine.Infer<DiscreteChar>(Variable.GetItem(str, 3)), ProbEps);
            Assert.Equal(DiscreteChar.Letter(), engine.Infer<DiscreteChar>(Variable.GetItem(str, 4)), ProbEps);
            Assert.Throws<AllZeroException>(() => engine.Infer<DiscreteChar>(Variable.GetItem(str, 5)));
        }

        /// <summary>
        /// Tests a model that infers a string from individual characters.
        /// </summary>
        [Fact]
        [Trait("Category", "StringInference")]
        public void StringFromItems()
        {
            Variable<string> str = Variable.StringOfLength(2);
            Variable<char> char1 = Variable.GetItem(str, 0);
            Variable<char> char2 = Variable.GetItem(str, 1);

            var engine = new InferenceEngine();
            char1.ObservedValue = 'a';
            char2.ObservedValue = 'b';

            Assert.Equal(StringDistribution.PointMass("ab"), engine.Infer<StringDistribution>(str), ProbEps);

            char1.ClearObservedValue();

            Assert.Equal(
                StringDistribution.Char(DiscreteChar.Any()) + StringDistribution.Char('b'),
                engine.Infer<StringDistribution>(str),
                ProbEps);
        }
    }
}
