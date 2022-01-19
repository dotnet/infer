// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using GaussianArray2D = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray2D<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;

namespace Microsoft.ML.Probabilistic.Tests
{
    public delegate void ModelDefinitionMethod();

    /// <summary>
    /// Summary description for MarginalPrototypeTests
    /// </summary>
    public class MarginalPrototypeMslTests
    {
        private ITypeDeclaration declaringType;

        public MarginalPrototypeMslTests()
        {
            // because this is in the constructor, every test in this class must have CsoftModel
            declaringType = RoslynDeclarationProvider.Instance.GetTypeDeclaration(this.GetType(), true);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Logical()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(LogicalModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));

            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
            Bernoulli dMarg = ca.Marginal<Bernoulli>("d");
            Bernoulli eMarg = ca.Marginal<Bernoulli>("e");
        }

        private void LogicalModel()
        {
            bool a = Factor.Random(new Bernoulli(0.5));
            bool b = Factor.Random(new Bernoulli(0.5));
            bool c = Factor.And(a, b);
            bool d = Factor.Or(a, b);
            bool e = Factor.Not(a);
            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
            InferNet.Infer(e, nameof(e));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_AreEqualInt()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(AreEqualIntModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
        }

        private void AreEqualIntModel()
        {
            int a = Factor.Random(new Discrete(0.1, 0.3, 0.6));
            int b = Factor.Random(new Discrete(0.2, 0.4, 0.4));
            bool c = Factor.AreEqual(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_AreEqualBool()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(AreEqualBoolModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));

            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
        }

        private void AreEqualBoolModel()
        {
            bool a = Factor.Random(new Bernoulli(0.5));
            bool b = Factor.Random(new Bernoulli(0.5));
            bool c = Factor.AreEqual(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_BernoulliFromBeta()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(BernoulliModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
        }

        private void BernoulliModel()
        {
            double a = Factor.Random(Beta.FromMeanAndVariance(0.5, 0.1));
            bool c = Factor.Bernoulli(a);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Discrete()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(DiscreteModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Discrete cMarg = ca.Marginal<Discrete>("c");
        }

        private void DiscreteModel()
        {
            Vector a = Factor.Random(new Dirichlet(0.1, 1.2, 2.3));
            int c = Factor.Discrete(a);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        [Trait("Category", "CsoftModel")]
        public void MP_DiscreteFromDiscrete()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(DiscreteFromDiscreteModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Discrete cMarg = ca.Marginal<Discrete>("c");
        }

        private void DiscreteFromDiscreteModel()
        {
            Matrix m = new Matrix(new double[,] { { 0.1, 0.3, 0.6 }, { 0.2, 0.4, 0.4 } });
            int s = Factor.Random(new Discrete(0.3, 0.7));
            int c = Factor.Discrete(s, m);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Exp()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(ExpModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gamma cMarg = ca.Marginal<Gamma>("c");
        }

        private void ExpModel()
        {
            double a = Factor.Random(Gaussian.FromMeanAndVariance(1.0, 2.0));
            double c = System.Math.Exp(a);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Gaussian()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GaussianModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
        }

        private void GaussianModel()
        {
            double m = Factor.Random(Gaussian.FromMeanAndVariance(1.0, 2.0));
            double p = Factor.Random(Gamma.FromMeanAndVariance(2.0, 1.0));
            double c = Factor.Gaussian(m, p);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemBernoulli()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemBernoulliModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
            DistributionArray<Bernoulli> dMarg = ca.Marginal<DistributionArray<Bernoulli>>("d");
        }

        private void GetItemBernoulliModel()
        {
            bool[] array = new bool[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(new Bernoulli(0.5));
            }
            bool c = Collection.GetItem(array, 1);
            bool[] d = new bool[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });
            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemPoisson()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemPoissonModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Poisson cMarg = ca.Marginal<Poisson>("c");
            DistributionArray<Poisson> dMarg = ca.Marginal<DistributionArray<Poisson>>("d");
        }

        private void GetItemPoissonModel()
        {
            int[] array = new int[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(new Poisson(1.0));
            }
            int c = Collection.GetItem(array, 1);
            int[] d = new int[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemDiscrete()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemDiscreteModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Discrete cMarg = ca.Marginal<Discrete>("c");
            DistributionArray<Discrete> dMarg = ca.Marginal<DistributionArray<Discrete>>("d");
        }

        private void GetItemDiscreteModel()
        {
            int[] array = new int[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(new Discrete(0.3, 0.3, 0.4));
            }
            int c = Collection.GetItem(array, 1);
            int[] d = new int[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemBeta()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemBetaModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Beta cMarg = ca.Marginal<Beta>("c");
            DistributionArray<Beta> dMarg = ca.Marginal<DistributionArray<Beta>>("d");
        }

        private void GetItemBetaModel()
        {
            double[] array = new double[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(Beta.FromMeanAndVariance(0.5, 1.0));
            }
            double c = Collection.GetItem(array, 1);
            double[] d = new double[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemGamma()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemGammaModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gamma cMarg = ca.Marginal<Gamma>("c");
            DistributionArray<Gamma> dMarg = ca.Marginal<DistributionArray<Gamma>>("d");
        }

        private void GetItemGammaModel()
        {
            double[] array = new double[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(Gamma.FromMeanAndVariance(1.0, 1.0));
            }
            double c = Collection.GetItem(array, 1);
            double[] d = new double[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemGaussian()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemGaussianModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
            DistributionArray<Gaussian> dMarg = ca.Marginal<DistributionArray<Gaussian>>("d");
        }

        private void GetItemGaussianModel()
        {
            double[] array = new double[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(Gaussian.FromMeanAndVariance(1.0, 1.0));
            }
            double c = Collection.GetItem(array, 1);
            double[] d = new double[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemDirichlet()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemDirichletModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Dirichlet cMarg = ca.Marginal<Dirichlet>("c");
            DistributionArray<Dirichlet> dMarg = ca.Marginal<DistributionArray<Dirichlet>>("d");
        }

        private void GetItemDirichletModel()
        {
            Vector[] array = new Vector[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(Dirichlet.Uniform(3));
            }
            Vector c = Collection.GetItem(array, 1);
            Vector[] d = new Vector[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemVectorGaussian()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemVectorGaussianModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            VectorGaussian cMarg = ca.Marginal<VectorGaussian>("c");
            DistributionArray<VectorGaussian> dMarg = ca.Marginal<DistributionArray<VectorGaussian>>("d");
        }

        private void GetItemVectorGaussianModel()
        {
            Vector[] array = new Vector[3];
            Vector mean = Vector.FromArray(0.1, 0.2);
            PositiveDefiniteMatrix prec = PositiveDefiniteMatrix.Identity(2);
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(new VectorGaussian(mean, prec));
            }
            Vector c = Collection.GetItem(array, 1);
            Vector[] d = new Vector[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemWishart()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemWishartModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Wishart cMarg = ca.Marginal<Wishart>("c");
            DistributionArray<Wishart> dMarg = ca.Marginal<DistributionArray<Wishart>>("d");
        }

        private void GetItemWishartModel()
        {
            PositiveDefiniteMatrix[] array = new PositiveDefiniteMatrix[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(new Wishart(3, 1.0, 1.0));
            }
            PositiveDefiniteMatrix c = Collection.GetItem(array, 1);
            PositiveDefiniteMatrix[] d = new PositiveDefiniteMatrix[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_GetItemSparseGP()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(GetItemSparseGPModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            SparseGP cMarg = ca.Marginal<SparseGP>("c");
            DistributionArray<SparseGP> dMarg = ca.Marginal<DistributionArray<SparseGP>>("d");
        }

        private void GetItemSparseGPModel()
        {
            // Basis Vector
            Vector[] basis = new Vector[]
                {
                    Vector.FromArray(new double[2] {0.2, 0.2}),
                    Vector.FromArray(new double[2] {0.2, 0.8}),
                    Vector.FromArray(new double[2] {0.8, 0.2}),
                    Vector.FromArray(new double[2] {0.8, 0.8})
                };
            // The kernel
            double[] kp = new double[2];
            kp[0] = 0.0;
            kp[1] = 0.0;
            IKernelFunction kf = new NNKernel(kp, 0.0);
            // The fixed parameters
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            SparseGP sgp = new SparseGP(sgpf);

            IFunction[] array = new IFunction[3];
            for (int i = 0; i < 3; i++)
            {
                array[i] = Factor.Random(sgp);
            }
            IFunction c = Collection.GetItem(array, 1);
            IFunction[] d = new IFunction[2];
            if (true) // preserve the previous assignment
                d = Collection.GetItems(array, new int[] { 0, 2 });

            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_InnerProduct()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            ModelDefinitionMethod meth = new ModelDefinitionMethod(InnerProductModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
        }

        private void InnerProductModel()
        {
            Vector a = Factor.Random(new VectorGaussian(Vector.FromArray(0.1, 0.2), PositiveDefiniteMatrix.Identity(2)));
            Vector b = Factor.Random(new VectorGaussian(Vector.FromArray(0.3, 0.4), PositiveDefiniteMatrix.Identity(2)));
            double c = Vector.InnerProduct(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_IsBetween()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(IsBetweenModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
        }

        private void IsBetweenModel()
        {
            double a = Factor.Random(Gaussian.FromMeanAndPrecision(0.1, 0.2));
            double b = Factor.Random(Gaussian.FromMeanAndPrecision(0.3, 0.4));
            double x = Factor.Random(Gaussian.FromMeanAndPrecision(0.2, 0.5));
            bool c = Factor.IsBetween(x, a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_IsPositive()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(IsPositiveModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Bernoulli cMarg = ca.Marginal<Bernoulli>("c");
        }

        private void IsPositiveModel()
        {
            double a = Factor.Random(Gaussian.FromMeanAndPrecision(1.2, 2.3));
            bool c = Factor.IsPositive(a);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_MatrixMultiply()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            ModelDefinitionMethod meth = new ModelDefinitionMethod(MatrixMultiplyModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            GaussianArray2D cMarg = ca.Marginal<GaussianArray2D>("c");
        }

        private void MatrixMultiplyModel()
        {
            double[,] a = new double[3, 2];
            double[,] b = new double[2, 3];
            double[,] c = new double[3, 3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    a[i, j] = Factor.Random(Gaussian.FromMeanAndPrecision(1.2, 3.4));
                    b[j, i] = Factor.Random(Gaussian.FromMeanAndPrecision(4.3, 2.1));
                }
            }
            c = Factor.MatrixMultiply(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_MatrixVectorProduct()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(MatrixVectorProductModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            VectorGaussian cMarg = ca.Marginal<VectorGaussian>("c");
        }

        private void MatrixVectorProductModel()
        {
            Matrix a = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Vector m = Vector.FromArray(0.1, 1.2, 2.3);
            Vector b = Factor.Random(VectorGaussian.FromMeanAndPrecision(m, PositiveDefiniteMatrix.Identity(3)));
            Vector c = Factor.Product(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Max()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(MaxModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
        }

        private void MaxModel()
        {
            double a = Factor.Random(new Gaussian(0.1, 0.2));
            double b = Factor.Random(new Gaussian(0.3, 0.4));
            double c = System.Math.Max(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Plus()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(PlusModel);
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
            Gaussian dMarg = ca.Marginal<Gaussian>("d");
        }

        private void PlusModel()
        {
            double a = Factor.Random(new Gaussian(0.1, 0.2));
            double b = Factor.Random(new Gaussian(0.3, 0.4));
            double c = Factor.Plus(a, b);
            double d = Factor.Difference(a, b);
            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Poisson()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            ModelDefinitionMethod meth = new ModelDefinitionMethod(PoissonModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Poisson cMarg = ca.Marginal<Poisson>("c");
        }

        private void PoissonModel()
        {
            double a = Factor.Random(new Gamma(1.0, 1.0));
            int c = Factor.Poisson(a);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Product()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(ProductModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
            Gaussian dMarg = ca.Marginal<Gaussian>("d");
        }

        private void ProductModel()
        {
            double a = Factor.Random(new Gaussian(0.1, 0.2));
            double b = Factor.Random(new Gaussian(0.3, 0.0));
            double c = Factor.Product(a, b);
            double d = Factor.Ratio(a, b);
            InferNet.Infer(c, nameof(c));
            InferNet.Infer(d, nameof(d));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_SparseGP()
        {
            InferenceEngine engine = new InferenceEngine();
            ModelDefinitionMethod meth = new ModelDefinitionMethod(SparseGPModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
        }

        private void SparseGPModel()
        {
            // Basis Vector
            Vector[] basis = new Vector[]
                {
                    Vector.FromArray(new double[2] {0.2, 0.2}),
                    Vector.FromArray(new double[2] {0.2, 0.8}),
                    Vector.FromArray(new double[2] {0.8, 0.2}),
                    Vector.FromArray(new double[2] {0.8, 0.8})
                };
            // The kernel
            double[] kp = new double[2];
            kp[0] = 0.0;
            kp[1] = 0.0;
            IKernelFunction kf = new NNKernel(kp, 0.0);
            // The fixed parameters
            SparseGPFixed sgpf = new SparseGPFixed(kf, basis);
            SparseGP agp = new SparseGP(sgpf);
            IFunction a = Factor.Random(agp);
            Vector b = Vector.FromArray(0.1, 0.2);
            double c = Factor.FunctionEvaluate(a, b);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_Sum()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            Gaussian[] aPrior = new Gaussian[2];
            aPrior[0] = Gaussian.FromMeanAndPrecision(0.1, 0.2);
            aPrior[1] = Gaussian.FromMeanAndPrecision(0.3, 0.4);
            var ca = engine.Compiler.Compile(SumModel, aPrior);
            ca.Execute(1);
            Gaussian cMarg = ca.Marginal<Gaussian>("c");
        }

        private void SumModel(Gaussian[] aPrior)
        {
            double[] a = new double[2];
            for (int i = 0; i < 2; i++)
            {
                a[i] = Factor.Random(aPrior[i]);
            }
            double c = Factor.Sum(a);
            InferNet.Infer(c, nameof(c));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MP_VectorGaussian()
        {
            // Need VMP as EP doesn't support random mean
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            ModelDefinitionMethod meth = new ModelDefinitionMethod(VectorGaussianModel);
            var ca = engine.Compiler.CompileWithoutParams(
                declaringType, meth.Method, new AttributeRegistry<object, ICompilerAttribute>(true));
            ca.Execute(1);
            VectorGaussian cMarg = ca.Marginal<VectorGaussian>("c");
        }

        private void VectorGaussianModel()
        {
            Vector mm = Vector.FromArray(0.1, 0.2);
            Vector m = Factor.Random(VectorGaussian.FromMeanAndPrecision(mm, PositiveDefiniteMatrix.Identity(2)));
            PositiveDefiniteMatrix p = Factor.Random(Wishart.FromShapeAndRate(2, 1.0, 1.0));
            Vector c = Factor.VectorGaussian(m, p);
            InferNet.Infer(c, nameof(c));
        }
    }
}