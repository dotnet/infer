// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tests
{

    public class ErrorTests
    {
        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TestLoopSizeError()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(LoopSizeError);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 1
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 1);
            }
        }

        private void LoopSizeError()
        {
            double[] array = new double[4]; // invalid number of dims
            for (int l = 0; l < array.GetLength(0); l++)
            {
                array[l] = Factor.Random(Gamma.FromShapeAndScale(1, 1));
            }
            InferNet.Infer(array, nameof(array));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TestLoopDecrementError()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(LoopDecrementError);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 1
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 1);
            }
        }

        private void LoopDecrementError()
        {
            double[] array = new double[4]; // invalid number of dims
            for (int l = 0; l < 4; l--)
            {
                array[l] = Factor.Random(Gamma.FromShapeAndScale(1, 1));
            }
            InferNet.Infer(array, nameof(array));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TestLoopStartError()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(LoopStartError);
                ca.Execute(50);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 1
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 1);
            }
        }

        private void LoopStartError()
        {
            double[] array = new double[4];
            for (int l = 1; l < 4; l++) // loop starts at 1 instead of 0
            {
                array[l] = Factor.Random(Gamma.FromShapeAndScale(1, 1));
            }
            InferNet.Infer(array, nameof(array));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TestArrayDimError()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(ArrayDimError);
            });
        }

        private void ArrayDimError()
        {
            double[,,] array3D = new double[1, 1, 1]; // invalid number of dims
            array3D[0, 0, 0] = Factor.Gaussian(0, 1);
            InferNet.Infer(array3D, nameof(array3D));
        }

        [Fact]
        [Trait("Category", "BadTest")]
        [Trait("Category", "CsoftModel")]
        public void TestRedefineParameterError()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(RedefineParameterError, 10);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 1
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 1);
            }
        }


        private void RedefineParameterError(int m)
        {
            m = 5;
            InferNet.Infer(m, nameof(m));
        }

        [Fact]
        [Trait("Category", "BadTest")]
        [Trait("Category", "CsoftModel")]
        public void TestRedefineParameterElementError()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(RedefineParameterElementError, new int[] { 10 });
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 1
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 1);
            }
        }


        private void RedefineParameterElementError(int[] m)
        {
            for (int i = 0; i < m.Length; i++)
            {
                m[i] = 5;
            }
            InferNet.Infer(m, nameof(m));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void TestDoubleDefinitionError()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(DoubleDefinitionModel);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 1
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 1);
            }
        }


        private void DoubleDefinitionModel()
        {
            Gaussian xPrior = new Gaussian(0, 1);
            double x = Factor.Random(xPrior);
            Gaussian xPrior2 = new Gaussian(1, 1);
            x = Factor.Random(xPrior2);
            InferNet.Infer(x, nameof(x));
        }

        internal void ElementRedefinitionError(int m, int[] marr)
        {
            double[] darr = new double[10];
            darr[0] = Factor.Random(new Gaussian(0, 1));
            for (int k = 0; k < 10; k++)
            {
                darr[k] = Factor.Random(Gamma.FromShapeAndScale(1, 1)); //cannot re-define
            }
            InferNet.Infer(darr, nameof(darr));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void TestDefinitionInLoop()
        {
            try
            {
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
                var ca = engine.Compiler.Compile(DefinitionInLoopErrors);
                ca.Execute(1);
                Assert.True(false, "Did not throw exception");
            }
            catch (CompilationFailedException tfe)
            {
                // Number of errors in MSL should be 2
                int numberOfErrors = tfe.Results.ErrorCount;
                Console.WriteLine("Detected " + numberOfErrors + " errors");
                Assert.True(numberOfErrors == 2);
            }
        }

        /// <summary>
        /// Model which breaks many of the constraints of MSL, to test the corresponding error messages.
        /// </summary>
        private void DefinitionInLoopErrors()
        {
            double d = Factor.Random(new Gaussian(0, 1));
            bool[] b = new bool[10];
            for (int m2 = 0; m2 < 12; m2++)
            {
                d = Factor.Random(new Gaussian(0, m2)); // cannot redefine, not indexed by m2
                for (int m = 0; m < 10; m++)
                {
                    b[m] = Factor.Bernoulli(0.5);
                }
            }
            InferNet.Infer(b, nameof(b));
            InferNet.Infer(d, nameof(d));
        }

        //public void ModelWithMoreErrors()
        //{
        //    //  TODO: use same variable as index in two loops
        //    Factor.Random(new Gaussian(0, 1)); //methods with return values must have them assigned to something
        //    double d = Factor.Random(new Gaussian(0, 1));
        //    double x = Factor.Random(new Gaussian(d, 1)); // invalid dependency on random variable
        //}
    }
}