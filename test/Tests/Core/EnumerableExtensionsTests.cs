using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests.Core
{
    public class EnumerableExtensionsTests
    {
        [Fact]
        public void SumUInt_EmptyArray_ReturnsZero()
        {
            Assert.Equal(0U, new uint[0].Sum());
        }

        [Fact]
        public void SumULong_EmptyArray_ReturnsZero()
        {
            Assert.Equal(0UL, new ulong[0].Sum());
        }

        /// <summary>
        /// This also tests DiscreteEstimator.Add(DiscreteEstimator)
        /// </summary>
        [Fact]
        public void TakeRandom_HasCorrectDistribution()
        {
            Rand.Restart(0);
            int universe = 10;
            int count = 5;
            var combinedEstimator = ParallelEnumerable.Range(0, 10000).Select(block =>
            {
                DiscreteEstimator discreteEstimator = new DiscreteEstimator(universe);
                HashSet<int> set = new HashSet<int>();
                foreach (var value in Enumerable.Range(0, universe).TakeRandom(count))
                {
                    Assert.DoesNotContain(value, set);
                    set.Add(value);
                    discreteEstimator.Add(value);
                }
                return discreteEstimator;
            }).Aggregate((a, b) => { a.Add(b); return a; });
            var dist = combinedEstimator.GetDistribution(Discrete.Uniform(universe));
            for (int i = 0; i < universe; i++)
            {
                Assert.True(dist[i] > 0.08 && dist[i] < 0.12);
            }
        }
    }
}
