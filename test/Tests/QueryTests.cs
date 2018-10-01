// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Compiler;
    using Assert = Xunit.Assert;

    public class TwoCoinsQueryTests
    {
        static TwoCoinsQueryTests()
        {
            // TODO fix hack around circular dependency
            Csoft.RoslynDeclarationProvider = RoslynDeclarationProvider.Instance;
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ExecuteQueryTest()
        {
            Csoft.ExecuteQuery(InferenceEngine.DefaultEngine, QueryPosterior);
            Csoft.ExecuteQuery(InferenceEngine.DefaultEngine, QueryWithObservation);
        }

        private void QueryPosterior()
        {
            TwoCoinsModel();
            var marginal = Csoft.Infer<Bernoulli>(BothHeads).GetProbTrue();
            Assert.Equal(0.25, marginal);
        }

        private void QueryWithObservation()
        {
            TwoCoinsModel();
            Csoft.Observe(BothHeads, false);
            var marginal = Csoft.Infer<Bernoulli>(FirstCoinHeads).GetProbTrue();
            Assert.Equal(1/3.0, marginal);
        }

        protected bool FirstCoinHeads { get; set; }
        protected bool SecondCoinHeads { get; set; }
        protected bool BothHeads { get; set; }

        [ModelMethod]
        private void TwoCoinsModel()
        {
            FirstCoinHeads = Factor.Bernoulli(0.5);
            SecondCoinHeads = Factor.Bernoulli(0.5);
            BothHeads = FirstCoinHeads & SecondCoinHeads;
        }
    }

    public class TwoCoinsQueryTests2
    {
        static TwoCoinsQueryTests2()
        {
            // TODO fix hack around circular dependency
            Csoft.RoslynDeclarationProvider = RoslynDeclarationProvider.Instance;
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ExecuteQueryTest()
        {
            var compiled = Csoft.GenerateSubclass<TwoCoinsQueryTests2>(InferenceEngine.DefaultEngine);
            Assert.Equal(0.5, compiled.QueryPosterior(true).GetProbTrue());
            Assert.Equal(0.0, compiled.QueryPosterior(false).GetProbTrue());
        }

        public virtual Bernoulli QueryPosterior(bool firstCoinHeads)
        {
            TwoCoinsModel(firstCoinHeads);
            return Csoft.Infer<Bernoulli>(BothHeads);
        }

        protected bool SecondCoinHeads { get; set; }
        protected bool BothHeads { get; set; }

        [ModelMethod]
        private void TwoCoinsModel(bool firstCoinHeads)
        {
            SecondCoinHeads = Factor.Bernoulli(0.5);
            BothHeads = firstCoinHeads & SecondCoinHeads;
        }
    }
}