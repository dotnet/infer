// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;
using Xunit;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class QuadratureTests
    {
        [Fact]
        public void UniformQuadrature()
        {
            for (int count = 2; count <= 20; count++)
            {
                Vector nodes = Vector.Zero(count);
                Vector weights = Vector.Zero(count);
                Quadrature.UniformNodesAndWeights(0, 1, nodes, weights);
                double result = (weights*(nodes ^ 3.0)).Sum();
                Assert.True(MMath.AbsDiff(1.0/4, result, 1e-10) < 1e-10);
            }
        }

        [Fact]
        public void UniformQuadrature2()
        {
            for (int count = 3; count <= 20; count++)
            {
                Vector nodes = Vector.Zero(count);
                Vector weights = Vector.Zero(count);
                Quadrature.UniformNodesAndWeights(0, 1, nodes, weights);
                double result = (weights*(nodes ^ 5.0)).Sum();
                Assert.True(MMath.AbsDiff(1.0/6, result, 1e-10) < 1e-10);
            }
        }

        [Fact]
        public void GammaQuadrature()
        {
            Vector nodes = Vector.Zero(15);
            Vector logWeights = Vector.Zero(15);
            Quadrature.GammaNodesAndWeights(2, 3, nodes, logWeights);
            Vector weights = Vector.Zero(logWeights.Count);
            weights.SetToFunction(logWeights, System.Math.Exp);
            double result = (weights*nodes*nodes).Sum();
            Assert.True(MMath.AbsDiff(4.0/3, result, 1e-4) < 1e-4);
        }
    }
}