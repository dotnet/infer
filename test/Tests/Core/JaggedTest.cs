// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;

namespace Microsoft.ML.Probabilistic.Tests.Core
{
    /// <summary>
    /// Summary description for JaggedTest
    /// </summary>
    public class JaggedTest
    {
        public int[,][][,][] jagged = null;
        private int[] arrExpected = null;

        public JaggedTest()
        {
            List<int> listExpected = new List<int>();
            jagged = new int[2,3][][,][];
            for (int i = 0; i < jagged.GetLength(0); i++)
            {
                for (int j = 0; j < jagged.GetLength(1); j++)
                {
                    jagged[i, j] = new int[i + j + 1][,][];
                    for (int k = 0; k < jagged[i, j].Length; k++)
                    {
                        jagged[i, j][k] = new int[k + 1,k + 2][];
                        for (int l = 0; l < jagged[i, j][k].GetLength(0); l++)
                            for (int m = 0; m < jagged[i, j][k].GetLength(1); m++)
                            {
                                jagged[i, j][k][l, m] = new int[i + 1];
                                for (int n = 0; n < jagged[i, j][k][l, m].Length; n++)
                                {
                                    jagged[i, j][k][l, m][n] = i + j + k + l + m + n;
                                    listExpected.Add(jagged[i, j][k][l, m][n]);
                                }
                            }
                    }
                }
            }
            arrExpected = listExpected.ToArray();
        }

        [Fact]
        public void JaggedRanks()
        {
            int[] ranks = JaggedArray.GetRanks(jagged.GetType(), typeof (int));
            Assert.Equal(4, ranks.Length);
            Assert.Equal(2, ranks[0]);
            Assert.Equal(1, ranks[1]);
            Assert.Equal(2, ranks[2]);
            Assert.Equal(1, ranks[3]);
        }

        [Fact]
        public void JaggedTypes()
        {
            Type[] types = JaggedArray.GetTypes(jagged.GetType(), typeof (int));
            Assert.Equal(5, types.Length);
            Assert.Equal(types[0], jagged.GetType());
            Assert.Equal(types[1], jagged[0, 0].GetType());
            Assert.Equal(types[2], jagged[0, 0][0].GetType());
            Assert.Equal(types[3], jagged[0, 0][0][0, 0].GetType());
            Assert.Equal(types[4], jagged[0, 0][0][0, 0][0].GetType());
        }

        [Fact]
        public void JaggedIterator()
        {
            List<int> listActual = new List<int>();
            foreach (int i in JaggedArray.ElementIterator(jagged, typeof (int)))
                listActual.Add(i);
            int[] arrActual = listActual.ToArray();
            Assert.Equal(arrActual.Length, arrExpected.Length);
            for (int i = 0; i < arrActual.Length; i++)
                Assert.Equal(arrActual[i], arrExpected[i]);
        }

        [Fact]
        public void JaggedIntIterator()
        {
            List<int> listActual = new List<int>();
            foreach (int i in JaggedArray.ElementIterator<int>(jagged))
                listActual.Add(i);
            int[] arrActual = listActual.ToArray();
            Assert.Equal(arrActual.Length, arrExpected.Length);
            for (int i = 0; i < arrActual.Length; i++)
                Assert.Equal(arrActual[i], arrExpected[i]);
        }

        [Fact]
        public void JaggedConvertToNewType()
        {
            Array jaggedGaussian = JaggedArray.ConvertToNew(
                jagged, typeof (int), typeof (Gaussian),
                delegate(object elt) { return new Gaussian(0.5, 1.0); });

            Assert.Equal(
                JaggedArray.GetLength(jaggedGaussian, typeof (Gaussian)),
                JaggedArray.GetLength(jagged, typeof (int)));

            foreach (Gaussian g in JaggedArray.ElementIterator(jaggedGaussian, typeof (Gaussian)))
                Assert.Equal(0.5, g.GetMean());
        }

        [Fact]
        public void JaggedConvertToNewTypeGeneric()
        {
            Array jaggedGaussian = JaggedArray.ConvertToNew<int, Gaussian>(
                jagged, delegate(int elt) { return new Gaussian(0.5, 1.0); });

            Assert.Equal(
                JaggedArray.GetLength(jaggedGaussian, typeof (Gaussian)),
                JaggedArray.GetLength(jagged, typeof (int)));

            foreach (Gaussian g in JaggedArray.ElementIterator<Gaussian>(jaggedGaussian))
                Assert.Equal(0.5, g.GetMean());
        }

        [Fact]
        public void JaggedSetValues()
        {
            Array jaggedGaussian = JaggedArray.ConvertToNew(
                jagged, typeof (int), typeof (Gaussian),
                delegate(object elt) { return new Gaussian(0.0, 1.0); });

            Assert.Equal(
                JaggedArray.GetLength(jaggedGaussian, typeof (Gaussian)),
                JaggedArray.GetLength(jagged, typeof (int)));

            JaggedArray.ConvertElements(
                jaggedGaussian, typeof (Gaussian),
                delegate(object elt) { return Gaussian.FromMeanAndPrecision(1.0, 2.0); });

            foreach (Gaussian g in JaggedArray.ElementIterator(jaggedGaussian, typeof (Gaussian)))
                Assert.Equal(1.0, g.GetMean());
        }

        [Fact]
        public void JaggedSetValues2()
        {
            Array jaggedGaussian = JaggedArray.ConvertToNew(
                jagged, typeof (int), typeof (Gaussian),
                delegate(object elt) { return new Gaussian(1.0, 1.0); });

            Assert.Equal(
                JaggedArray.GetLength(jaggedGaussian, typeof (Gaussian)),
                JaggedArray.GetLength(jagged, typeof (int)));

            JaggedArray.ConvertElements2(
                jaggedGaussian, jagged, typeof (Gaussian),
                delegate(object elt1, object elt2)
                    {
                        Gaussian g = (Gaussian) elt1;
                        double d = (int) elt2;
                        return Gaussian.FromMeanAndPrecision(d + g.GetMean(), 2.0);
                    });

            int i = 0;
            foreach (Gaussian g in JaggedArray.ElementIterator(jaggedGaussian, typeof (Gaussian)))
                Assert.Equal(g.GetMean(), 1.0 + arrExpected[i++]);
        }

        [Fact]
        public void JaggedVisitor()
        {
            Array jaggedDir = JaggedArray.ConvertToNew(
                jagged, typeof (int), typeof (Dirichlet),
                delegate(object elt) { return new Dirichlet(2.0, 3.0, 4.0); });

            double[] newPseudoCount = {3.0, 4.0, 5.0};
            JaggedArray.VisitElements(
                jaggedDir, typeof (Dirichlet),
                delegate(object elt) { ((Dirichlet) elt).PseudoCount.SetTo(newPseudoCount); });

            foreach (Dirichlet d in JaggedArray.ElementIterator(jaggedDir, typeof (Dirichlet)))
                Assert.Equal(3.0, d.PseudoCount[0]);
        }

        [Fact]
        public void JaggedVisitor2()
        {
            Dirichlet d1 = new Dirichlet(2.0, 3.0, 4.0);
            Dirichlet d2 = new Dirichlet(3.0, 4.0, 5.0);
            Dirichlet d3 = d1*d2;

            Array jaggedDir1 = JaggedArray.ConvertToNew(
                jagged, typeof (int), typeof (Dirichlet),
                delegate(object elt) { return new Dirichlet(d1); });

            Array jaggedDir2 = JaggedArray.ConvertToNew(
                jagged, typeof (int), typeof (Dirichlet),
                delegate(object elt) { return new Dirichlet(d2); });

            JaggedArray.VisitElements2(
                jaggedDir1, jaggedDir2, typeof (Dirichlet), typeof (Dirichlet),
                delegate(object elt1, object elt2) { ((Dirichlet) elt1).PseudoCount.SetTo((((Dirichlet) elt1)*((Dirichlet) elt2)).PseudoCount); });

            foreach (Dirichlet d in JaggedArray.ElementIterator(jaggedDir1, typeof (Dirichlet)))
                Assert.Equal(d.PseudoCount, d3.PseudoCount);
        }
    }
}