// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;
using Xunit;
using Microsoft.ML.Probabilistic.Algorithms;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    /// <summary>
    /// Tests of EqualityPropagationTransform
    /// </summary>
    public class ConstrainEqualTests
    {
        [Fact]
        public void ConstrainEqualCaseArrayTest()
        {
            Range i = new Range(4).Named("i");
            VariableArray<double> x = Variable.Array<double>(i).Named("x");
            VariableArray<int> c = Variable.Array<int>(i).Named("c");
            using (Variable.ForEach(i))
            {
                c[i] = Variable.Discrete(new double[] {0.5, 0.5});
                using (Variable.Case(c[i], 0))
                {
                    x[i] = Variable.GaussianFromMeanAndVariance(1, 1);
                }
                using (Variable.Case(c[i], 1))
                {
                    x[i] = Variable.GaussianFromMeanAndVariance(2, 1);
                }
            }

            VariableArray<double> data = Variable.Observed(new double[] {0.9, 1.1, 1.9, 2.1}, i).Named("data");
            Variable.ConstrainEqual(x[i], data[i]);
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            Discrete[] cExpectedArray = new Discrete[]
                {
                    new Discrete(0.6457, 0.3543),
                    new Discrete(0.5987, 0.4013), new Discrete(0.4013, 0.5987),
                    new Discrete(0.3543, 0.6457)
                };
            IDistribution<int[]> cExpected = Distribution<int>.Array(cExpectedArray);
            object cActual = engine.Infer(c);
            Console.WriteLine(StringUtil.JoinColumns("c = ", cActual, " should be ", cExpected));
            Assert.True(cExpected.MaxDiff(cActual) < 1e-4);
        }

        [Fact]
        public void ConstrainEqualCycleTest()
        {
            double xPrior = 0.1;
            double yPrior = 0.2;
            double zPrior = 0.3;
            double wPrior = 0.4;
            var x = Variable.Bernoulli(xPrior).Named("x");
            var y = Variable.Bernoulli(yPrior).Named("y");
            var z = Variable.Bernoulli(zPrior).Named("z");
            var w = Variable.Bernoulli(wPrior).Named("w");
            Variable.ConstrainEqual(x, y);
            Variable.ConstrainEqual(y, z);
            Variable.ConstrainEqual(x, z);
            Variable.ConstrainEqual(z, w);
            w.ObservedValue = true;
            InferenceEngine engine = new InferenceEngine();
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            Bernoulli xExpected = new Bernoulli(1);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
        }

        [Fact]
        public void ConstrainEqualManyToOneTest()
        {
            double bPrior = 0.1;
            double cPrior = 0.2;
            var c = Variable.Bernoulli(cPrior).Named("c");
            Range i = new Range(2).Named("i");
            var bools = Variable.Array<bool>(i).Named("bools");
            using (Variable.ForEach(i))
            {
                bools[i] = Variable.Bernoulli(bPrior);
                Variable.ConstrainEqual(bools[i], c);
            }
            bools.ObservedValue = new bool[] {true, true};
            InferenceEngine engine = new InferenceEngine();
            Bernoulli cActual = engine.Infer<Bernoulli>(c);
            Bernoulli cExpected = new Bernoulli(1);
            Console.WriteLine("c = {0} should be {1}", cActual, cExpected);
            Assert.True(cExpected.MaxDiff(cActual) < 1e-10);
        }

        [Fact]
        public void ConstrainEqualIntGateExitTest()
        {
            double bPrior = 0.1;
            var b = Variable.Bernoulli(bPrior).Named("b");
            var x = Variable.New<int>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Discrete(0.2, 0.8));
                Variable.ConstrainEqual(x, 0);
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Discrete(0.3, 0.7));
                Variable.ConstrainEqual(x, 1);
            }
            InferenceEngine engine = new InferenceEngine();
            Discrete xActual = engine.Infer<Discrete>(x);
            double evExpected = bPrior*0.2 + (1 - bPrior)*0.7;
            double xExpected0 = (bPrior*0.2 + (1 - bPrior)*0.0)/evExpected;
            Discrete xExpected = new Discrete(xExpected0, 1 - xExpected0);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        [Fact]
        public void ConstrainEqualGateExitTest()
        {
            double bPrior = 0.1;
            var b = Variable.Bernoulli(bPrior).Named("b");
            var x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                if (true)
                {
                    x.SetTo(Variable.Bernoulli(0.2));
                    Variable.ConstrainEqual(x, true);
                }
                else
                {
                    // this is what the above code should transform into
                    var y = Variable.Bernoulli(0.2);
                    Variable.ConstrainEqual(y, true);
                    x.SetTo(Variable.Copy(Variable.Constant(true)));
                }
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(0.3));
            }
            InferenceEngine engine = new InferenceEngine();
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evExpected = bPrior*0.2 + (1 - bPrior);
            Bernoulli xExpected = new Bernoulli((bPrior*0.2 + (1 - bPrior)*0.3)/evExpected);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        [Fact]
        public void ConstrainEqualGateExitTest2()
        {
            double bPrior = 0.1;
            var b = Variable.Bernoulli(bPrior).Named("b");
            var x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(0.2));
                Variable.ConstrainEqual(x, true);
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(0.3));
            }
            Variable.ConstrainEqual(b, true);
            bPrior = 1;
            InferenceEngine engine = new InferenceEngine();
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evExpected = bPrior*0.2 + (1 - bPrior);
            Bernoulli xExpected = new Bernoulli((bPrior*0.2 + (1 - bPrior)*0.3)/evExpected);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        [Fact]
        public void ConstrainEqualGateExitTest3()
        {
            double bPrior = 0.1;
            var b = Variable.Bernoulli(bPrior).Named("b");
            var x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(0.2));
                Variable.ConstrainEqual(x, true);
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(0.3));
                Variable.ConstrainEqual(x, false);
            }
            Variable.ConstrainEqual(b, true);
            bPrior = 1;
            InferenceEngine engine = new InferenceEngine();
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evExpected = bPrior*0.2 + (1 - bPrior);
            Bernoulli xExpected = new Bernoulli((bPrior*0.2 + (1 - bPrior)*0.3)/evExpected);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }

        // TODO: x is marked stoc when it should not be
        [Fact]
        public void ConstrainEqualGateExitTest4()
        {
            double bPrior = 0.1;
            var b = Variable.Bernoulli(bPrior).Named("b");
            var x = Variable.New<bool>().Named("x");
            using (Variable.If(b))
            {
                x.SetTo(Variable.Bernoulli(0.2));
            }
            using (Variable.IfNot(b))
            {
                x.SetTo(Variable.Bernoulli(0.3));
            }
            Variable.ConstrainEqual(x, true);
            InferenceEngine engine = new InferenceEngine();
            Bernoulli xActual = engine.Infer<Bernoulli>(x);
            double evExpected = bPrior*0.2 + (1 - bPrior);
            Bernoulli xExpected = new Bernoulli(1);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-10);
        }
    }
}