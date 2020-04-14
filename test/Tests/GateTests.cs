// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Math;
using Xunit;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Algorithms;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class GateTests
    {
        // this model doesn't compile, requiring Switch(index)
        internal void FakeSwitchTest2()
        {
            Vector probs = Vector.FromArray(0.1, 0.5, 0.4);
            int size = probs.Count;
            Range item = new Range(size).Named("item");
            VariableArray<bool> bools = Variable.Array<bool>(item).Named("bools");
            using (ForEachBlock fb = Variable.ForEach(item))
            {
                bools[item] = Variable.Bernoulli(0.3);
                Variable<int> index = Variable.Discrete(probs);
                index.SetValueRange(item);
                using (Variable.If(index == fb.Index))
                {
                    Variable.ConstrainTrue(bools[index]);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            double[] boolsPost = new double[probs.Count];
            boolsPost[0] = 0.1 + 0.5*0.3 + 0.4*0.3;
            boolsPost[1] = 0.1*0.3 + 0.5 + 0.4*0.3;
            boolsPost[2] = 0.1*0.3 + 0.5*0.3 + 0.4;
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = engine.Infer(bools);
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        // Fails due to unsupported MSL
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void FakeSwitchTest()
        {
            Vector probs = Vector.FromArray(0.1, 0.5, 0.4);

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(FakeSwitchModel, probs.Count, probs);
            ca.Execute(20);

            double[] boolsPost = new double[probs.Count];
            boolsPost[0] = 0.1 + 0.5*0.3 + 0.4*0.3;
            boolsPost[1] = 0.1*0.3 + 0.5 + 0.4*0.3;
            boolsPost[2] = 0.1*0.3 + 0.5*0.3 + 0.4;
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = ca.Marginal("bools");
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        // This model looks like a 'switch' block but it isn't one, since 'index' is a local variable of the loop.
        private void FakeSwitchModel(int size, Vector probs)
        {
            bool[] bools = new bool[size];
            for (int i = 0; i < size; i++)
            {
                bools[i] = Factor.Bernoulli(0.3);
                int index = Factor.Discrete(probs);
                if (index == i)
                {
                    Constrain.Equal(true, bools[index]);
                }
            }
            InferNet.Infer(bools, nameof(bools));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void SwitchWithReplicateTest()
        {
            // same as SwitchTest but with bools[i] instead of bools[index] inside the 'if'
            Vector probs = Vector.FromArray(0.1, 0.5, 0.4);
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(SwitchWithReplicateModel, probs.Count, probs);
            ca.Execute(20);

            double[] boolsPost = new double[probs.Count];
            boolsPost[0] = 0.1 + 0.5*0.3 + 0.4*0.3;
            boolsPost[1] = 0.1*0.3 + 0.5 + 0.4*0.3;
            boolsPost[2] = 0.1*0.3 + 0.5*0.3 + 0.4;
            Bernoulli[] boolsExpectedArray = new Bernoulli[boolsPost.Length];
            for (int i = 0; i < boolsExpectedArray.Length; i++)
            {
                boolsExpectedArray[i] = new Bernoulli(boolsPost[i]);
            }
            Diffable boolsExpected = Distribution<bool>.Array(boolsExpectedArray);
            object boolsActual = ca.Marginal("bools");
            Console.WriteLine(StringUtil.JoinColumns("bools = ", boolsActual, " should be ", boolsExpected));
            Assert.True(boolsExpected.MaxDiff(boolsActual) < 1e-10);
        }

        private void SwitchWithReplicateModel(int size, Vector probs)
        {
            bool[] bools = new bool[size];
            for (int i = 0; i < size; i++)
            {
                bools[i] = Factor.Bernoulli(0.3);
            }
            int index = Factor.Discrete(probs);
            // must use a different index name or declaration provider may get confused
            for (int i2 = 0; i2 < size; i2++)
            {
                if (index == i2)
                {
                    Constrain.Equal(true, bools[i2]);
                }
            }
            InferNet.Infer(bools, nameof(bools));
        }

        internal void SwitchWithReplicateModel(int size, int size2, Vector probs)
        {
            bool[] bools = new bool[size];
            int[] indices = new int[size];
            for (int j = 0; j < size2; j++)
            {
                for (int i = 0; i < size; i++)
                {
                    indices[i] = Factor.Discrete(probs);
                    if (indices[i] == i)
                    {
                        Constrain.Equal(true, bools[i]);
                    }
                }
            }
            InferNet.Infer(indices, nameof(indices));
            InferNet.Infer(bools, nameof(bools));
        }

        /// <summary>
        /// Fails because we attempt to take a product of different point masses during inference.
        /// The correct implementation would not run inference inside a gate that is known to be off.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void GatedAllZeroTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedAllZeroModel);
            ca.Execute(20);

            Bernoulli bExpected = new Bernoulli(0);
            Bernoulli bActual = ca.Marginal<Bernoulli>("b");
            Console.WriteLine("b = {0} (should be {1})", bActual, bExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-10);

            Console.WriteLine("x = {0}", ca.Marginal<Bernoulli>("x"));
        }

        private void GatedAllZeroModel()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (b)
            {
                Constrain.Equal(true, x);
                Constrain.Equal(false, x);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }


        [Fact]
        [Trait("Category", "CsoftModel")]
        public void SwitchManyTest()
        {
            Vector probs = Vector.FromArray(0.1, 0.5, 0.4);
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(SwitchManyConstraintModel, probs);
            ca.Execute(20);

            Console.WriteLine(ca.Marginal("i"));
            Console.WriteLine(ca.Marginal("x"));
            // TODO: assertion
        }

        private void SwitchManyConstraintModel(Vector probs)
        {
            bool[] x = new bool[3];
            for (int j = 0; j < 3; j++)
            {
                x[j] = Factor.Bernoulli(0.4);
            }
            int i = Factor.Discrete(probs);
            for (int k = 0; k < 3; k++)
            {
                if (i == k)
                {
                    Constrain.EqualRandom(x[i], new Bernoulli(0.2));
                }
            }
            InferNet.Infer(i, nameof(i));
            InferNet.Infer(x, nameof(x));
        }


        [Fact]
        [Trait("Category", "CsoftModel")]
        public void SwitchMany3Test()
        {
            Vector probs = Vector.FromArray(0.1, 0.5, 0.4);
            //Vector probs2 = Vector.FromArray(0.2, 0.3, 0.4, 0.1);
            Vector probs2 = Vector.FromArray(0.0, 1.0, 0.0, 0.0);
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(SwitchManyConstraint3Model, probs, probs2);
            ca.Execute(20);

            Console.WriteLine(ca.Marginal("i"));
            Console.WriteLine(ca.Marginal("i2"));
            Console.WriteLine(ca.Marginal("x"));
            // TODO: assertion
        }

        private void SwitchManyConstraint3Model(Vector probs, Vector probs2)
        {
            bool[,] x = new bool[3,4];
            for (int j = 0; j < 3; j++)
            {
                for (int j2 = 0; j2 < 4; j2++)
                {
                    x[j, j2] = Factor.Bernoulli(0.4);
                }
            }
            int i = Factor.Discrete(probs);
            int i2 = Factor.Discrete(probs2);
            for (int k = 0; k < 3; k++)
            {
                if (i == k)
                {
                    for (int k2 = 0; k2 < 4; k2++)
                    {
                        if (i2 == k2)
                        {
                            Constrain.EqualRandom(x[i, i2], new Bernoulli(0.2));
                        }
                    }
                }
            }
            InferNet.Infer(i, nameof(i));
            InferNet.Infer(i2, nameof(i2));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        public void RandomIndexObservedArrayInt()
        {
            var observedArray = Variable.Observed(new int[] {1, 2, 3});
            var randomIndex = Variable.DiscreteUniform(observedArray.Range);

            var element = Variable.New<int>().Named("element");
            observedArray.SetValueRange(new Range(10));
            using (Variable.Switch(randomIndex))
            {
                element.SetTo(observedArray[randomIndex]);
            }

            InferenceEngine engine = new InferenceEngine {};
            Discrete elementExpected = new Discrete(0, 1.0/3, 1.0/3, 1.0/3, 0, 0, 0, 0, 0, 0);
            var elementActual = engine.Infer(element);
            Console.WriteLine("element = {0} should be {1}", elementActual, elementExpected);
            Assert.True(elementExpected.MaxDiff(elementActual) < 1e-6);
        }

        [Fact]
        public void RandomIndexObservedArrayDouble()
        {
            var observedArray = Variable.Observed(new double[] {1.0, 2.0, 3.0});
            var randomIndex = Variable.DiscreteUniform(observedArray.Range);

            var element = Variable.New<double>().Named("element");
            using (Variable.Switch(randomIndex))
            {
                element.SetTo(observedArray[randomIndex]);
            }

            InferenceEngine engine = new InferenceEngine {};
            var elementActual = engine.Infer(element);
            double mExpected = 2.0;
            double m2Expected = (1*1 + 2*2 + 3*3)/3.0;
            Gaussian elementExpected = Gaussian.FromMeanAndVariance(mExpected, m2Expected - mExpected*mExpected);
            Console.WriteLine("element = {0} should be {1}", elementActual, elementExpected);
            Assert.True(elementExpected.MaxDiff(elementActual) < 1e-4);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ArrayOfGatedConstraintTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(ArrayOfGatedConstraintModel);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 0.2;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double Z = priorB*sumXCondT + (1 - priorB);
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB))/Z;

            DistributionArray<Bernoulli> bDist = ca.Marginal<DistributionArray<Bernoulli>>("b");
            DistributionArray<Bernoulli> xDist = ca.Marginal<DistributionArray<Bernoulli>>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            for (int i = 0; i < bDist.Count; i++)
            {
                Assert.True(System.Math.Abs(bDist[i].GetProbTrue() - postB) < 1e-4);
                Assert.True(System.Math.Abs(xDist[i].GetProbTrue() - postX) < 1e-4);
            }
        }

        private void ArrayOfGatedConstraintModel()
        {
            bool[] b = new bool[3];
            bool[] x = new bool[3];
            for (int i = 0; i < 3; i++)
            {
                b[i] = Factor.Bernoulli(0.1);
                x[i] = Factor.Bernoulli(0.4);
                if (b[i])
                {
                    Constrain.EqualRandom(x[i], new Bernoulli(0.2));
                }
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstraintTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstraint);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 0.2;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double Z = priorB*sumXCondT + (1 - priorB);
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB))/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstraintVmpTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstraint);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 0.2;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double Z = priorB*sumXCondT + (1 - priorB);
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB))/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            //Assert.True(Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            //Assert.True(Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedConstraint()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (b)
            {
                Constrain.EqualRandom(x, new Bernoulli(0.2));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedMultipleConstraintTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedMultipleConstraintModel);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 0.2;
            double pXCondT2 = 0.3;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = priorX*pXCondT*pXCondT2 + (1 - priorX)*(1 - pXCondT)*(1 - pXCondT2);
            double Z = priorB*sumXCondT + (1 - priorB);
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT*pXCondT2 + (1 - priorB))/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedMultipleConstraintModel()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (b)
            {
                Constrain.EqualRandom(x, new Bernoulli(0.2));
                Constrain.EqualRandom(x, new Bernoulli(0.3));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedMultipleConstraint2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedMultipleConstraint2Model);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 0.2;
            double pXCondT2 = 0.3;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = priorX*pXCondT*pXCondT2 + (1 - priorX)*(1 - pXCondT)*(1 - pXCondT2);
            double Z = priorB*sumXCondT + (1 - priorB);
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT*pXCondT2 + (1 - priorB))/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Gaussian xDist = ca.Marginal<Gaussian>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            //Assert.True(Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            //Assert.True(Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedMultipleConstraint2Model()
        {
            bool b = Factor.Bernoulli(0.1);
            double x = Factor.Gaussian(0.2, 2.2);
            if (b)
            {
                bool h = Factor.IsPositive(x);
                Constrain.Equal(true, h);
                double OneMinusX = Factor.Difference(1.0, x);
                bool h2 = Factor.IsPositive(OneMinusX);
                Constrain.Equal(true, h2);
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstraint2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstraint2);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double pXCondT = 0.2;
            double pXCondF = 0.3;
            // p(x,b) =propto p0(b) p0(x) f(x)^b = (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double Z = priorB*(priorX*pXCondT + (1 - priorX)*(1 - pXCondT)) + (1 - priorB)*(priorX*pXCondF + (1 - priorX)*(1 - pXCondF));
            double postB = priorB*(priorX*pXCondT + (1 - priorX)*(1 - pXCondT))/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB)*pXCondF)/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedConstraint2()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (b)
            {
                Constrain.EqualRandom(x, new Bernoulli(0.2));
            }
            else
            {
                Constrain.EqualRandom(x, new Bernoulli(0.3));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstraint3Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstraint3Model);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.4;
            double priorY = 0.7;
            double pXCondT = 0.2;
            double pXCondF = 0.3;
            double pYCondF = 0.6;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) (py)^y (1-py)^(1-y) [(pT)^x (1-pT)^(1-x)]^b
            //                [(pF)^x (1-pF)^(1-x) (pF)^y (1-pF)^(1-y)]^(1-b)
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double sumYCondF = priorY*pYCondF + (1 - priorY)*(1 - pYCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF*sumYCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB)*pXCondF*sumYCondF)/Z;
            double postY = priorY*(priorB*sumXCondT + (1 - priorB)*sumXCondF*pYCondF)/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Bernoulli yDist = ca.Marginal<Bernoulli>("y");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Console.WriteLine("y = {0} (should be {1})", yDist, postY);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
            Assert.True(System.Math.Abs(yDist.GetProbTrue() - postY) < 1e-4);
        }

        private void GatedConstraint3Model()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            bool y = Factor.Bernoulli(0.7);
            if (b)
            {
                Constrain.EqualRandom(x, new Bernoulli(0.2));
            }
            else
            {
                Constrain.EqualRandom(x, new Bernoulli(0.3));
                Constrain.EqualRandom(y, new Bernoulli(0.6));
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(y, nameof(y));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstantTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstant);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0;
            double pXCondF = 1;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) 
            //                [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB)*pXCondF)/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedConstant()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x;
            if (b)
            {
                x = false;
            }
            else
            {
                x = true;
            }
            Constrain.EqualRandom(x, new Bernoulli(0.2));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGatedConstantTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGatedConstant);
            ca.Execute(20);

            double priorA = 0.9;
            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0;
            double pXCondF = 1;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) 
            //                [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double sumCondT = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double Z = priorA*sumCondT + (1 - priorA);
            double postA = priorA*sumCondT/Z;
            double postB = priorB*sumXCondT/sumCondT;
            double postX = priorX*(priorB*pXCondT + (1 - priorB)*pXCondF)/sumCondT;

            Bernoulli aDist = ca.Marginal<Bernoulli>("a");
            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("a = {0} (should be {1})", aDist, postA);
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - postA) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void NestedGatedConstant()
        {
            bool a = Factor.Bernoulli(0.9);
            if (a)
            {
                bool b = Factor.Bernoulli(0.1);
                bool x;
                if (b)
                {
                    x = false;
                }
                else
                {
                    x = true;
                }
                Constrain.EqualRandom(x, new Bernoulli(0.2));
                InferNet.Infer(b, nameof(b));
                InferNet.Infer(x, nameof(x));
            }
            InferNet.Infer(a, nameof(a));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedConstantOrRandomTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedConstantOrRandom);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.3;
            double pXCondF = 0;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x) 
            //                [(pT)^x (1-pT)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT + (1 - priorX)*(1 - pXCondT);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT + (1 - priorB)*pXCondF)/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GatedConstantOrRandom()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x;
            if (b)
            {
                x = Factor.Bernoulli(0.3);
            }
            else
            {
                x = Factor.Random(Bernoulli.PointMass(false));
                //x = false; - not legal in MSL
            }
            Constrain.EqualRandom(x, new Bernoulli(0.2));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GateExitConstraintTestMsl()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GateExitConstraintModel);
            ca.Execute(20);

            double priorB = 0.1;
            double priorX = 0.2;
            double pXCondT = 0.3;
            double pXCondT2 = 0.6;
            double pXCondF = 0.4;
            // p(x,b) =propto (pb)^b (1-pb)^(1-b) (px)^x (1-px)^(1-x)
            //                [(pT)^x (1-pT)^(1-x) (pT2)^x (1-pT2)^(1-x)]^b [(pF)^x (1-pF)^(1-x)]^(1-b)
            double sumXCondT = priorX*pXCondT*pXCondT2 + (1 - priorX)*(1 - pXCondT)*(1 - pXCondT2);
            double sumXCondF = priorX*pXCondF + (1 - priorX)*(1 - pXCondF);
            double Z = priorB*sumXCondT + (1 - priorB)*sumXCondF;
            double postB = priorB*sumXCondT/Z;
            double postX = priorX*(priorB*pXCondT*pXCondT2 + (1 - priorB)*pXCondF)/Z;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, postB);
            Console.WriteLine("x = {0} (should be {1})", xDist, postX);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - postB) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - postX) < 1e-4);
        }

        private void GateExitConstraintModel()
        {
            bool b = Factor.Bernoulli(0.1);
            bool x;
            if (b)
            {
                x = Factor.Bernoulli(0.3);
                Constrain.EqualRandom(x, new Bernoulli(0.6));
            }
            else
            {
                x = Factor.Bernoulli(0.4);
            }
            Constrain.EqualRandom(x, new Bernoulli(0.2));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        //public void GateExitConstraintTransformed()
        //{
        //  bool b = Factor.Bernoulli(0.1);
        //  if (b_cases0) {
        //    bool x_cond_b0 = Factor.Bernoulli(0.3);
        //    Constrain.EqualRandom(x_cond_b0, new Bernoulli(0.6));
        //  }
        //  if (b_cases1) {
        //    bool x_cond_b1 = Factor.Bernoulli(0.4);
        //  }
        //  bool x = Gate.Exit(b_cases0, x_cond_b0, b_cases1, x_cond_b1);
        //  Constrain.EqualRandom(x, new Bernoulli(0.2));
        //  InferNet.Infer(b, nameof(b));
        //  InferNet.Infer(x, nameof(x));
        //}

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void GatedDeclarationTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(GatedDeclarationModel);
            ca.Execute(20);

            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondT = 0.2;
            // p(b) =propto (pb)^b (1-pb)^(1-b) [sum_x (px)^x (1-px)^(1-x) (pT)^x (1-pT)^(1-x)]^b
            double sumXCondT = xPrior*xCondT + (1 - xPrior)*(1 - xCondT);
            double Z = bPrior*sumXCondT + (1 - bPrior);
            double bPost = bPrior*sumXCondT/Z;
            double xPost = xPrior*xCondT/sumXCondT;

            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        private void GatedDeclarationModel()
        {
            bool b = Factor.Bernoulli(0.1);
            if (b)
            {
                bool x = Factor.Bernoulli(0.4);
                Constrain.EqualRandom(x, new Bernoulli(0.2));
                InferNet.Infer(x, nameof(x));
            }
            InferNet.Infer(b, nameof(b));
        }

        internal void GateEnterArray2()
        {
            bool b = Factor.Bernoulli(0.1);
            bool[] x = new bool[2];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = Factor.Bernoulli(0.2);
            }
            if (b)
            {
                bool y = x[0];
                Constrain.EqualRandom(y, new Bernoulli(0.3));
            }
            InferNet.Infer(x, nameof(x));
            InferNet.Infer(b, nameof(b));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGate1Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGate1Model);
            ca.Execute(20);

            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondTT = 0.2;
            double sumXCondTT = xPrior*xCondTT + (1 - xPrior)*(1 - xCondTT);
            double Z = aPrior*bPrior*sumXCondTT + (1 - aPrior)*bPrior + aPrior*(1 - bPrior) + (1 - aPrior)*(1 - bPrior);
            double aPost = aPrior*(bPrior*sumXCondTT + (1 - bPrior))/Z;
            double bPost = bPrior*(aPrior*sumXCondTT + (1 - aPrior))/Z;
            double xPost = xPrior*(aPrior*bPrior*xCondTT + (1 - aPrior)*bPrior + aPrior*(1 - bPrior) + (1 - aPrior)*(1 - bPrior))/Z;

            Bernoulli aDist = ca.Marginal<Bernoulli>("a");
            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("a = {0} (should be {1})", aDist, aPost);
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - aPost) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        private void NestedGate1Model()
        {
            bool a = Factor.Bernoulli(0.9);
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (a)
            {
                if (b)
                {
                    Constrain.EqualRandom(x, new Bernoulli(0.2));
                }
            }
            InferNet.Infer(a, nameof(a));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGate2Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGate2Model);
            ca.Execute(20);

            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondTT = 0.2;
            double sumXCondTT = xPrior*xCondTT + (1 - xPrior)*(1 - xCondTT);
            double Z = aPrior*(bPrior*sumXCondTT + (1 - bPrior)) + (1 - aPrior);
            double aPost = aPrior*(bPrior*sumXCondTT + (1 - bPrior))/Z;
            double bPost = bPrior*sumXCondTT/(bPrior*sumXCondTT + (1 - bPrior));
            double xPost = xPrior*(aPrior*bPrior*xCondTT + aPrior*(1 - bPrior) + (1 - aPrior))/Z;

            Bernoulli aDist = ca.Marginal<Bernoulli>("a");
            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("a = {0} (should be {1})", aDist, aPost);
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - aPost) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        private void NestedGate2Model()
        {
            bool a = Factor.Bernoulli(0.9);
            bool x = Factor.Bernoulli(0.4);
            if (a)
            {
                bool b = Factor.Bernoulli(0.1);
                if (b)
                {
                    Constrain.EqualRandom(x, new Bernoulli(0.2));
                }
                InferNet.Infer(b, nameof(b));
            }
            InferNet.Infer(a, nameof(a));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGate3Test()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGate3Model);
            ca.Execute(20);

            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.4;
            double xCondTT = 0.2;
            double sumXCondTT = xPrior*xCondTT + (1 - xPrior)*(1 - xCondTT);
            double Z = aPrior*(bPrior*sumXCondTT + (1 - bPrior)) + (1 - aPrior);
            double aPost = aPrior*(bPrior*sumXCondTT + (1 - bPrior))/Z;
            double bPost = bPrior*sumXCondTT/(bPrior*sumXCondTT + (1 - bPrior));
            //double xPost = xPrior * (aPrior * bPrior * xCondTT + aPrior * (1 - bPrior) + (1 - aPrior)) / Z;
            double xPost = xPrior*xCondTT/sumXCondTT;

            Bernoulli aDist = ca.Marginal<Bernoulli>("a");
            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("a = {0} (should be {1})", aDist, aPost);
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - aPost) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        private void NestedGate3Model()
        {
            bool a = Factor.Bernoulli(0.9);
            if (a)
            {
                bool b = Factor.Bernoulli(0.1);
                if (b)
                {
                    bool x = Factor.Bernoulli(0.4);
                    InferNet.Infer(x, nameof(x));
                    Constrain.EqualRandom(x, new Bernoulli(0.2));
                }
                InferNet.Infer(b, nameof(b));
            }
            InferNet.Infer(a, nameof(a));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGatesInnerLoopTest()
        {
            double xCondTT1 = 0.2, xCondTT2 = 0.3;
            Bernoulli[] priors = new Bernoulli[] {new Bernoulli(xCondTT1), new Bernoulli(xCondTT2)};

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGatesInnerLoopModel, priors);
            ca.Execute(20);

            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.4;
            double sumXCondTT = xPrior*xCondTT1*xCondTT2 + (1 - xPrior)*(1 - xCondTT1)*(1 - xCondTT2);
            double Z = aPrior*bPrior*sumXCondTT + (1 - aPrior)*bPrior + aPrior*(1 - bPrior) + (1 - aPrior)*(1 - bPrior);
            double aPost = aPrior*(bPrior*sumXCondTT + (1 - bPrior))/Z;
            double bPost = bPrior*(aPrior*sumXCondTT + (1 - aPrior))/Z;
            double xPost = xPrior*(aPrior*bPrior*xCondTT1*xCondTT2 + (1 - aPrior)*bPrior + aPrior*(1 - bPrior) + (1 - aPrior)*(1 - bPrior))/Z;

            Bernoulli aDist = ca.Marginal<Bernoulli>("a");
            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("a = {0} (should be {1})", aDist, aPost);
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - aPost) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        private void NestedGatesInnerLoopModel(Bernoulli[] priors)
        {
            bool a = Factor.Bernoulli(0.9);
            bool b = Factor.Bernoulli(0.1);
            bool x = Factor.Bernoulli(0.4);
            if (a)
            {
                if (b)
                {
                    for (int i = 0; i < priors.Length; i++)
                    {
                        Constrain.EqualRandom(x, priors[i]);
                    }
                }
            }
            InferNet.Infer(a, nameof(a));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGatesInnerLoop2Test()
        {
            double xCondTT1 = 0.2, xCondTT2 = 0.3;
            Bernoulli[] priors = new Bernoulli[] {new Bernoulli(xCondTT1), new Bernoulli(xCondTT2)};

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGatesInnerLoopModel2, priors);
            ca.Execute(20);
            // This is a compiler test only  
        }

        private void NestedGatesInnerLoopModel2(Bernoulli[] priors)
        {
            bool a = Factor.Bernoulli(0.9);
            bool b = Factor.Bernoulli(0.1);
            bool[] x = new bool[priors.Length];
            for (int j = 0; j < priors.Length; j++)
            {
                x[j] = Factor.Bernoulli(0.4);
            }
            if (a)
            {
                if (b)
                {
                    for (int i = 0; i < priors.Length; i++)
                    {
                        Constrain.EqualRandom(x[i], priors[i]);
                    }
                }
            }
            InferNet.Infer(a, nameof(a));
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(x, nameof(x));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void NestedGateExitTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(NestedGateExitModel);
            ca.Execute(20);

            double aPrior = 0.9;
            double bPrior = 0.1;
            double xPrior = 0.2;
            double xCondTT = 0.4;
            double xCondTF = 0.3;
            double sumXCondTT = xPrior*xCondTT + (1 - xPrior)*(1 - xCondTT);
            double sumXCondTF = xPrior*xCondTF + (1 - xPrior)*(1 - xCondTF);
            double sumXCondT = (bPrior*sumXCondTT + (1 - bPrior)*sumXCondTF);
            double Z = aPrior*sumXCondT + (1 - aPrior);
            double aPost = aPrior*sumXCondT/Z;
            double bPost = bPrior*sumXCondTT/sumXCondT;
            //double xPost = xPrior * (aPrior * bPrior * xCondTT + aPrior * (1 - bPrior) + (1 - aPrior)) / Z;
            double xPost = xPrior*(bPrior*xCondTT + (1 - bPrior)*xCondTF)/sumXCondT;

            Bernoulli aDist = ca.Marginal<Bernoulli>("a");
            Bernoulli bDist = ca.Marginal<Bernoulli>("b");
            Bernoulli xDist = ca.Marginal<Bernoulli>("x");
            Console.WriteLine("a = {0} (should be {1})", aDist, aPost);
            Console.WriteLine("b = {0} (should be {1})", bDist, bPost);
            Console.WriteLine("x = {0} (should be {1})", xDist, xPost);
            Assert.True(System.Math.Abs(aDist.GetProbTrue() - aPost) < 1e-4);
            Assert.True(System.Math.Abs(bDist.GetProbTrue() - bPost) < 1e-4);
            Assert.True(System.Math.Abs(xDist.GetProbTrue() - xPost) < 1e-4);
        }

        private void NestedGateExitModel()
        {
            bool a = Factor.Bernoulli(0.9);
            if (a)
            {
                bool b = Factor.Bernoulli(0.1);
                bool x;
                if (b)
                {
                    x = Factor.Bernoulli(0.4);
                }
                else
                {
                    x = Factor.Bernoulli(0.3);
                }
                Constrain.EqualRandom(x, new Bernoulli(0.2));
                InferNet.Infer(x, nameof(x));
                InferNet.Infer(b, nameof(b));
            }
            InferNet.Infer(a, nameof(a));
        }

        internal void SimpleGateTest()
        {
            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(SimpleGateModel);
            ca.Execute(20);
        }

        private void SimpleGateModel()
        {
            double y;
            Gamma gamma = new Gamma(1, 1);
            double z = Factor.Random(gamma);
            bool b = Factor.Bernoulli(0.5);
            if (b)
            {
                double x = Factor.Gaussian(10, 2);
                InferNet.Infer(x, nameof(x));
                y = Factor.Gaussian(x, z);
            }
            else
            {
                y = Factor.Gaussian(0, 1);
                //y = Factor.Gaussian(0, 1);
            }
            InferNet.Infer(y, nameof(y));
            InferNet.Infer(b, nameof(b));
        }

        internal void NestedGateTest()
        {
            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(NestedGateModel);
            ca.Execute(20);
        }

        private void NestedGateModel()
        {
            bool a = Factor.Bernoulli(0.5);
            double z = Factor.Gaussian(0, 1);
            if (a)
            {
                double y;
                bool b = Factor.Bernoulli(0.5);
                if (b)
                {
                    double x = Factor.Gaussian(10, 2);
                    InferNet.Infer(x, nameof(x));
                    y = Factor.Gaussian(x, z);
                }
                else
                {
                    y = Factor.Gaussian(0, 1);
                }
                InferNet.Infer(y, nameof(y));
                InferNet.Infer(b, nameof(b));
            }
            InferNet.Infer(a, nameof(a));
        }

        internal void GatedConditionTest()
        {
            InferenceEngine engine = new InferenceEngine();
            var ca = engine.Compiler.Compile(GatedCondition);
            ca.Execute(20);
        }

        private void GatedCondition()
        {
            bool a = Factor.Bernoulli(0.5);

            bool b;
            if (a)
            {
                double y;
                b = Factor.Bernoulli(0.5);
                if (b)
                {
                    y = Factor.Gaussian(10, 1);
                }
                else
                {
                    y = Factor.Gaussian(0, 1);
                }
                InferNet.Infer(y, nameof(y));
                InferNet.Infer(b, nameof(b));
            }
            else
            {
                b = Factor.Bernoulli(1);
            }
            InferNet.Infer(a, nameof(a));
        }


        [Fact]
        public void CasesOpSparseVectorTest()
        {
            var probs = new Discrete(Vector.Constant(20, 1e-1, Sparsity.ApproximateWithTolerance(1e-3)));
            int nCases = probs.Dimension;
            var cases = new List<Bernoulli>(Util.ArrayInit(nCases, i => new Bernoulli(1e-4)));
            cases[0] = new Bernoulli(0.99);
            var result = IntCasesOp.IAverageConditional(cases, probs);
            var sparseProbs = (ApproximateSparseVector)result.GetProbs();
            Assert.Equal(1, sparseProbs.SparseCount);
        }
    }
}