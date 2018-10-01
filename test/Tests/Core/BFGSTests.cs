// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests.Core
{
    
    public class BFGSTests
    {
        private static double rosenbrock(double x, double y, out double gradx, out double grady)
        {
            gradx = -2.0*(1.0 - x) - 400.0*x*(y - x*x);
            grady = 200.0*(y - x*x);
            return (1.0 - x)*(1.0 - x) + 100.0*(y - x*x)*(y - x*x);
        }

        private static double f(Vector x, ref Vector grad)
        {
            double gradx, grady;
            double result = rosenbrock(x[0], x[1], out gradx, out grady);
            if (grad != null)
            {
                grad[0] = gradx;
                grad[1] = grady;
            }
            return result;
        }

        [Fact]
        public void CompareToSparse()
        {
            var dim = 10;
            var lbfgs = new LBFGS(3);
            var x = Vector.Zero(dim);
            x = lbfgs.Run(x, 1.0, f);
            Console.WriteLine(x);

            SparseVector y = SparseVector.Zero(dim);
            y = (SparseVector) lbfgs.Run(y, 1.0, f);
            Console.WriteLine(y);
            Assert.True(y.SparseCount == 2);
            Assert.True(y == x);

            SparseVector[] z = new SparseVector[2];
            z[0] = SparseVector.Zero(dim/2);
            z[1] = SparseVector.Zero(dim/2);
            var lbfgsa = new LBFGSArray(3);
            var result = lbfgsa.Run(z, 1.0, delegate(Vector[] o, ref Vector[] grad)
                {
                    Vector gradx = Vector.Zero(2);
                    double res = f(Vector.FromArray(new double[] {o[0][0], o[1][0]}), ref gradx);
                    if (grad != null)
                    {
                        grad[0][0] = gradx[0];
                        grad[1][0] = gradx[1];
                    }
                    return res;
                });
            Assert.True(MMath.AbsDiff(result[0][0], x[0]) < 1e-5);
            Assert.True(MMath.AbsDiff(result[1][0], x[1]) < 1e-5);
            Assert.True((result[0] as SparseVector).SparseCount == 1);
        }
    }
}