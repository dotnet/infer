// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using SparseElement = Microsoft.ML.Probabilistic.Collections.ValueAtIndex<double>;

namespace Microsoft.ML.Probabilistic.Tests.Distributions
{
    /// <summary>
    /// Summary description for SparseTests
    /// </summary>
    public class SparseDistributionTests
    {
        [Fact]
        public void SparseDiscreteNormalise()
        {
            double commonValue = 1.0;
            double nonCommonValue = 10.0;
            Discrete d = new Discrete(
                SparseVector.FromSparseValues(100, 1.0, new List<SparseElement>
                    {
                        new SparseElement(20, nonCommonValue),
                        new SparseElement(55, nonCommonValue)
                    }));

            Vector v = d.GetProbs();
            Assert.Equal(Sparsity.Sparse, d.Sparsity);
            double sum = (v.Count - 2)*commonValue + 2*nonCommonValue;
            SparseVector sv = (SparseVector) v;
            Assert.Equal(sv.CommonValue, commonValue/sum);
            Assert.Equal(2, sv.SparseValues.Count);
        }

        [Fact]
        public void SparseDiscreteProduct()
        {
            double commonValue = 1.0;
            double nonCommonValue1 = 10.0;
            double nonCommonValue2 = 30.0;
            Discrete d1 = new Discrete(
                SparseVector.FromSparseValues(100, commonValue, new List<SparseElement>
                    {
                        new SparseElement(20, nonCommonValue1),
                        new SparseElement(55, nonCommonValue1)
                    }));

            Discrete d2 = new Discrete(
                SparseVector.FromSparseValues(100, commonValue, new List<SparseElement>
                    {
                        new SparseElement(25, nonCommonValue2),
                        new SparseElement(55, nonCommonValue2)
                    }));

            Discrete d = d1*d2;

            Assert.Equal(Sparsity.Sparse, d.Sparsity);
            SparseVector sv = (SparseVector) (d.GetProbs());
            Assert.Equal(3, sv.SparseValues.Count);
        }

        [Fact]
        public void SparseDiscreteRatio()
        {
            double commonValue = 1.0;
            double nonCommonValue1 = 10.0;
            double nonCommonValue2 = 30.0;
            Discrete d1 = new Discrete(
                SparseVector.FromSparseValues(100, commonValue, new List<SparseElement>
                    {
                        new SparseElement(20, nonCommonValue1),
                        new SparseElement(55, nonCommonValue1)
                    }));

            Discrete d2 = new Discrete(
                SparseVector.FromSparseValues(100, commonValue, new List<SparseElement>
                    {
                        new SparseElement(25, nonCommonValue2),
                        new SparseElement(55, nonCommonValue2)
                    }));

            Discrete d = d1/d2;

            Assert.Equal(Sparsity.Sparse, d.Sparsity);
            SparseVector sv = (SparseVector) (d.GetProbs());
            Assert.Equal(3, sv.SparseValues.Count);
        }

        [Fact]
        public void SparseDirichletProduct()
        {
            double commonPseudoCount = 1.0;
            double nonCommonPseudoCount1 = 10.0;
            double nonCommonPseudoCount2 = 30.0;
            Dirichlet d1 = new Dirichlet(
                SparseVector.FromSparseValues(100, commonPseudoCount, new List<SparseElement>
                    {
                        new SparseElement(20, nonCommonPseudoCount1),
                        new SparseElement(55, nonCommonPseudoCount1)
                    }));

            Dirichlet d2 = new Dirichlet(
                SparseVector.FromSparseValues(100, commonPseudoCount, new List<SparseElement>
                    {
                        new SparseElement(25, nonCommonPseudoCount2),
                        new SparseElement(55, nonCommonPseudoCount2)
                    }));

            Dirichlet d = d1*d2;

            Assert.Equal(Sparsity.Sparse, d.Sparsity);
            SparseVector sv = (SparseVector) (d.PseudoCount);
            Assert.Equal(3, sv.SparseValues.Count);
        }

        [Fact]
        public void SparseDirichletRatio()
        {
            double commonPseudoCount = 1.0;
            double nonCommonPseudoCount1 = 10.0;
            double nonCommonPseudoCount2 = 30.0;
            Dirichlet d1 = new Dirichlet(
                SparseVector.FromSparseValues(100, commonPseudoCount, new List<SparseElement>
                    {
                        new SparseElement(20, nonCommonPseudoCount1),
                        new SparseElement(55, nonCommonPseudoCount1)
                    }));

            Dirichlet d2 = new Dirichlet(
                SparseVector.FromSparseValues(100, commonPseudoCount, new List<SparseElement>
                    {
                        new SparseElement(25, nonCommonPseudoCount2),
                        new SparseElement(55, nonCommonPseudoCount2)
                    }));

            Dirichlet d = d1/d2;

            Assert.Equal(Sparsity.Sparse, d.Sparsity);
            SparseVector sv = (SparseVector) (d.PseudoCount);
            Assert.Equal(3, sv.SparseValues.Count);
        }

        [Fact]
        public void BernoulliIntegerSubsetArithmetic()
        {
            double tolerance = 1e-10;
            var commonValue1 = new Bernoulli(0.1);
            var commonValue2 = new Bernoulli(0.2);
            var specialValue1 = new Bernoulli(0.7);
            var specialValue2 = new Bernoulli(0.8);
            var specialValue3 = new Bernoulli(0.9);

            var listSize = 100;
            var sparseBernoulliList1 = SparseBernoulliList.Constant(listSize, commonValue1);
            var sparseBernoulliList2 = SparseBernoulliList.Constant(listSize, commonValue2);
            sparseBernoulliList1[20] = specialValue1;
            sparseBernoulliList1[55] = specialValue2;
            sparseBernoulliList2[25] = specialValue2;
            sparseBernoulliList2[55] = specialValue3;
            var bernoulliIntegerSubset1 = BernoulliIntegerSubset.FromSparseList(sparseBernoulliList1);
            var bernoulliIntegerSubset2 = BernoulliIntegerSubset.FromSparseList(sparseBernoulliList2);

            // Product
            var product = bernoulliIntegerSubset1 * bernoulliIntegerSubset2;
            Assert.Equal(3, product.SparseBernoulliList.SparseValues.Count);
            Assert.Equal(commonValue1 * commonValue2, product.SparseBernoulliList.CommonValue);
            Assert.Equal(specialValue1 * commonValue2, product.SparseBernoulliList[20]);
            Assert.Equal(commonValue1 * specialValue2, product.SparseBernoulliList[25]);
            Assert.Equal(specialValue2 * specialValue3, product.SparseBernoulliList[55]);

            // Ratio
            var ratio = bernoulliIntegerSubset1 / bernoulliIntegerSubset2;
            Assert.Equal(2, ratio.SparseBernoulliList.SparseValues.Count);
            Assert.Equal((commonValue1 / commonValue2).GetProbTrue(), ratio.SparseBernoulliList.CommonValue.GetProbTrue(), tolerance);
            Assert.Equal((specialValue1 / commonValue2).GetProbTrue(), ratio.SparseBernoulliList[20].GetProbTrue(), tolerance);
            Assert.Equal((commonValue1 / specialValue2).GetProbTrue(), ratio.SparseBernoulliList[25].GetProbTrue(), tolerance);
            Assert.Equal((specialValue2 / specialValue3).GetProbTrue(), ratio.SparseBernoulliList[55].GetProbTrue(), tolerance);

            // Power
            var exponent = 1.2;
            var power = bernoulliIntegerSubset1 ^ exponent;
            Assert.Equal(2, power.SparseBernoulliList.SparseValues.Count);
            Assert.Equal(commonValue1 ^ exponent, power.SparseBernoulliList.CommonValue);
            Assert.Equal(specialValue1 ^ exponent, power.SparseBernoulliList[20]);
            Assert.Equal(specialValue2 ^ exponent, power.SparseBernoulliList[55]);
        }

        [Fact]
        public void SparseBernoulliListArithmetic()
        {
            double tolerance = 1e-10;
            var commonValue1 = new Bernoulli(0.1);
            var commonValue2 = new Bernoulli(0.2);
            var specialValue1 = new Bernoulli(0.7);
            var specialValue2 = new Bernoulli(0.8);
            var specialValue3 = new Bernoulli(0.9);

            var listSize = 100;
            var sparseBernoulliList1 = SparseBernoulliList.Constant(listSize, commonValue1);
            var sparseBernoulliList2 = SparseBernoulliList.Constant(listSize, commonValue2);
            sparseBernoulliList1[20] = specialValue1;
            sparseBernoulliList1[55] = specialValue2;
            sparseBernoulliList2[25] = specialValue2;
            sparseBernoulliList2[55] = specialValue3;

            // Product
            var product = sparseBernoulliList1 * sparseBernoulliList2;
            Assert.Equal(3, product.SparseValues.Count);
            Assert.Equal(commonValue1 * commonValue2, product.CommonValue);
            Assert.Equal(specialValue1 * commonValue2, product[20]);
            Assert.Equal(commonValue1 * specialValue2, product[25]);
            Assert.Equal(specialValue2 * specialValue3, product[55]);

            // Ratio
            var ratio = sparseBernoulliList1 / sparseBernoulliList2;
            Assert.Equal(2, ratio.SparseValues.Count);
            Assert.Equal((commonValue1 / commonValue2).GetProbTrue(), ratio.CommonValue.GetProbTrue(), tolerance);
            Assert.Equal((specialValue1 / commonValue2).GetProbTrue(), ratio[20].GetProbTrue(), tolerance);
            Assert.Equal((commonValue1 / specialValue2).GetProbTrue(), ratio[25].GetProbTrue(), tolerance);
            Assert.Equal((specialValue2 / specialValue3).GetProbTrue(), ratio[55].GetProbTrue(), tolerance);

            // Power
            var exponent = 1.2;
            var power = sparseBernoulliList1 ^ exponent;
            Assert.Equal(2, power.SparseValues.Count);
            Assert.Equal(commonValue1 ^ exponent, power.CommonValue);
            Assert.Equal(specialValue1 ^ exponent, power[20]);
            Assert.Equal(specialValue2 ^ exponent, power[55]);
        }

        [Fact]
        public void SparseDistributionListTolerances()
        {
            var bern = SparseBernoulliList.FromSize(1);
            var beta = SparseBetaList.FromSize(1);
            var gamma = SparseGammaList.FromSize(1);
            var gauss = SparseGaussianList.FromSize(1);
            double origBernDefaultTolerance = SparseBernoulliList.DefaultTolerance;
            double origBetaDefaultTolerance = SparseBetaList.DefaultTolerance;
            double origGammaDefaultTolerance = SparseGammaList.DefaultTolerance;
            double origGaussDefaultTolerance = SparseGaussianList.DefaultTolerance;
            Assert.Equal(origBernDefaultTolerance, bern.Tolerance);
            Assert.Equal(origBetaDefaultTolerance, beta.Tolerance);
            Assert.Equal(origGammaDefaultTolerance, gamma.Tolerance);
            Assert.Equal(origGaussDefaultTolerance, gauss.Tolerance);

            double newBernDefaultTolerance = 0.1;
            double newBetaDefaultTolerance = 0.2;
            double newGammaDefaultTolerance = 0.3;
            double newGaussDefaultTolerance = 0.4;

            try
            {
                // Checks that we can maintain different default tolerances
                // on different specializations of the generic base class
                SparseBernoulliList.DefaultTolerance = newBernDefaultTolerance;
                SparseBetaList.DefaultTolerance = newBetaDefaultTolerance;
                SparseGammaList.DefaultTolerance = newGammaDefaultTolerance;
                SparseGaussianList.DefaultTolerance = newGaussDefaultTolerance;

                Assert.Equal(newBernDefaultTolerance, SparseBernoulliList.DefaultTolerance);
                Assert.Equal(newBetaDefaultTolerance, SparseBetaList.DefaultTolerance);
                Assert.Equal(newGammaDefaultTolerance, SparseGammaList.DefaultTolerance);
                Assert.Equal(newGaussDefaultTolerance, SparseGaussianList.DefaultTolerance);

                // Now check that the default tolerance gets picked up by the factory methods.
                bern = SparseBernoulliList.FromSize(1);
                beta = SparseBetaList.FromSize(1);
                gamma = SparseGammaList.FromSize(1);
                gauss = SparseGaussianList.FromSize(1);

                Assert.Equal(newBernDefaultTolerance, bern.Tolerance);
                Assert.Equal(newBetaDefaultTolerance, beta.Tolerance);
                Assert.Equal(newGammaDefaultTolerance, gamma.Tolerance);
                Assert.Equal(newGaussDefaultTolerance, gauss.Tolerance);
            }

            finally
            {
                // Now revert back so that we don't spoil the other tests
                SparseBernoulliList.DefaultTolerance = origBernDefaultTolerance;
                SparseBetaList.DefaultTolerance = origBetaDefaultTolerance;
                SparseGammaList.DefaultTolerance = origGammaDefaultTolerance;
                SparseGaussianList.DefaultTolerance = origGaussDefaultTolerance;
            }
        }
    }
}