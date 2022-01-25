// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class MaxProductTests
    {
        [Fact]
        public void BasicTest()
        {
            var a = Variable.Discrete(0.1, 0.8, 0.1).Named("a");
            var b = Variable.Discrete(0.1, 0.2, 0.7).Named("b");
            Variable.ConstrainEqual(a, b);

            InferenceEngine ie = new InferenceEngine {Algorithm = new MaxProductBeliefPropagation()};
            Console.WriteLine("Dist over a = " + ie.Infer(a));
            Console.WriteLine("Dist over b = " + ie.Infer(b));
        }

        [Fact]
        public void PottsTest()
        {
            var a = Variable.Discrete(0.1, 0.8, 0.1).Named("a");
            var b = Variable.Discrete(0.1, 0.2, 0.7).Named("b");
            Variable.Potts(a, b, 2.5);
            InferenceEngine ie = new InferenceEngine {Algorithm = new MaxProductBeliefPropagation()};
            Console.WriteLine("Dist over a = " + ie.Infer(a));
            Console.WriteLine("Dist over b = " + ie.Infer(b));
        }

        [Fact]
        public void PottsTestCSharp4Grid()
        {
            // A 4-connected grid using C# loops to define variables and connectivity (unrolled)
            int sideLengthI = 2;
            int sideLengthJ = 3;
            Variable<bool>[,] pixels = new Variable<bool>[sideLengthI,sideLengthJ];
            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    if ((i + j)%3 == 0)
                    {
                        pixels[i, j] = Variable.Bernoulli(.8).Named(String.Format("x_{0}{1}", i, j));
                    }
                    else if ((i + j)%3 == 1)
                    {
                        pixels[i, j] = Variable.Bernoulli(.1).Named(String.Format("x_{0}{1}", i, j));
                    }
                    else if ((i + j)%3 == 2)
                    {
                        pixels[i, j] = Variable.Bernoulli(.4).Named(String.Format("x_{0}{1}", i, j));
                    }
                    else
                    {
                        throw new Exception("Should not reach here.");
                    }
                }
            }

            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    Variable<bool> variableIJ = pixels[i, j];
                    if (j < sideLengthJ - 1)
                    {
                        Variable<bool> rightNeighbor = pixels[i, j + 1];
                        Variable.Potts(variableIJ, rightNeighbor, .5);
                    }

                    if (i < sideLengthI - 1)
                    {
                        Variable<bool> belowNeighbor = pixels[i + 1, j];
                        Variable.Potts(variableIJ, belowNeighbor, .5);
                    }
                }
            }

            InferenceEngine ie = new InferenceEngine {Algorithm = new MaxProductBeliefPropagation()};
            // These don't work?
            //InferenceEngine ie = new InferenceEngine { Algorithm = new VariationalMessagePassing() };
            //InferenceEngine ie = new InferenceEngine { Algorithm = new VariationalMessagePassing() };

            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    Console.WriteLine("Dist over pixels[" + i + ", " + j + "] = " + ie.Infer(pixels[i, j]));
                }
            }
        }

        [Fact]
        public void Chain3gramTest()
        {
            int N = 5;
            Variable<bool>[] x = new Variable<bool>[N];

            for (int i = 0; i < N; i++)
            {
                x[i] = Variable.New<bool>().Named(String.Format("x{0}", i));
                if (i <= 1)
                {
                    x[i].SetTo(Variable.Bernoulli(.5));
                }
                else
                {
                    using (Variable.If(x[i - 2]))
                    {
                        using (Variable.If(x[i - 1]))
                        {
                            x[i].SetTo(Variable.Bernoulli(.7));
                        }
                        using (Variable.IfNot(x[i - 1]))
                        {
                            x[i].SetTo(Variable.Bernoulli(.3));
                        }
                    }
                    using (Variable.IfNot(x[i - 2]))
                    {
                        using (Variable.If(x[i - 1]))
                        {
                            x[i].SetTo(Variable.Bernoulli(.4));
                        }
                        using (Variable.IfNot(x[i - 1]))
                        {
                            x[i].SetTo(Variable.Bernoulli(.6));
                        }
                    }
                }
            }

            InferenceEngine ie = new InferenceEngine {Algorithm = new ExpectationPropagation()};
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("dist[" + i + "]=" + ie.Infer(x[i]));
            }
        }

        [Fact]
        public void SumChainTest()
        {
            int N = 5;
            Variable<double>[] x = new Variable<double>[N];

            for (int i = 0; i < N; i++)
            {
                x[i] = Variable.New<double>().Named(String.Format("x{0}", i));
                if (i <= 1)
                {
                    x[i].SetTo(Variable.GaussianFromMeanAndPrecision(0, 1));
                }
                else
                {
                    x[i].SetTo(x[i - 2] + x[i - 1]);
                }
            }
            x[N - 1].ObservedValue = 1;

            InferenceEngine ie = new InferenceEngine {Algorithm = new ExpectationPropagation()};
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("dist[" + i + "]=" + ie.Infer(x[i]));
            }
        }

        [Fact]
        public void BernoulliChainTest()
        {
            int N = 10;
            Variable<bool>[] x = new Variable<bool>[N];
            Beta b = Beta.Uniform();
            for (int i = 0; i < N; i++)
            {
                x[i] = Variable.Bernoulli(b.Sample());
                if (i == 0) continue;
                Variable.Potts(x[i], x[i - 1], 1.0);
            }

            InferenceEngine ie = new InferenceEngine {Algorithm = new MaxProductBeliefPropagation()};
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("dist[" + i + "]=" + ie.Infer(x[i]));
            }
        }

        [Fact]
        public void LinearTest()
        {
            int K = 7, N = 10;
            Variable<int>[] x = new Variable<int>[N];
            Dirichlet d = Dirichlet.Uniform(K);
            for (int i = 0; i < N; i++)
            {
                x[i] = Variable.Discrete(d.Sample());
                if (i == 0) continue;
                Variable.Linear(x[i], x[i - 1], 1.0);
            }

            InferenceEngine ie = new InferenceEngine {Algorithm = new MaxProductBeliefPropagation()};
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("dist[" + i + "]=" + ie.Infer<UnnormalizedDiscrete>(x[i]).ToString("f4"));
            }
        }

        [Fact]
        public void TruncLinearTest()
        {
            int K = 7, N = 10;
            Variable<int>[] x = new Variable<int>[N];
            Dirichlet d = Dirichlet.Uniform(K);
            for (int i = 0; i < N; i++)
            {
                x[i] = Variable.Discrete(d.Sample());
                if (i == 0) continue;
                Variable.LinearTrunc(x[i], x[i - 1], 1.0, 1.0);
            }

            InferenceEngine ie = new InferenceEngine {Algorithm = new MaxProductBeliefPropagation()};
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("dist[" + i + "]=" + ie.Infer<UnnormalizedDiscrete>(x[i]).ToString("f4"));
            }
        }

        [Fact]
        public void PottsGridTest()
        {
            int size = 10;
            Range rows = new Range(size).Named("rows");
            Range cols = new Range(size).Named("cols");

            var states = Variable.Array<bool>(rows, cols).Named("states");
            Bernoulli[,] unary = new Bernoulli[size,size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    var xdist = System.Math.Abs(i - size/2)/((double) size);
                    var ydist = System.Math.Abs(j - size/2)/((double) size);
                    bool inrect = (xdist < 0.2) && (ydist < 0.2);
                    unary[i, j] = new Bernoulli(inrect ? 0.8 : 0.2);
                }
            }

            var p = Variable.Observed(unary, rows, cols);
            p.Name = nameof(p);
            states[rows, cols] = Variable.Random<bool, Bernoulli>(p[rows, cols]);

            double logCost = 0;
            using (ForEachBlock rowBlock = Variable.ForEach(rows))
            {
                using (ForEachBlock colBlock = Variable.ForEach(cols))
                {
                    using (Variable.If(rowBlock.Index >= 1))
                    {
                        Variable.Potts(states[rowBlock.Index, colBlock.Index],
                                       states[rowBlock.Index + -1, colBlock.Index], logCost);
                    }

                    using (Variable.If(colBlock.Index >= 1))
                    {
                        Variable.Potts(states[rowBlock.Index, colBlock.Index],
                                       states[rowBlock.Index, colBlock.Index + -1], logCost);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new MaxProductBeliefPropagation();
            engine.ShowTimings = true;
            engine.NumberOfIterations = 20;
            var result = engine.Infer<Bernoulli[,]>(states);
            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    Console.Write("{0:f5} ", result[i, j].GetProbTrue());
                }
                Console.WriteLine();
            }
        }
    }
}