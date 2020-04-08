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
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Summary description for MarginalPrototypeTests
    /// </summary>
    public class MarginalPrototypeTests
    {
        /// <summary>
        /// Tests that MarginalPrototype is inferred correctly when using DirichletUniform in a Switch
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MP_Switch()
        {
            var T_val_range = new Range(2);
            Range T_cat_range = new Range(2);
            var T_cat_val_p = Variable.Array<Vector>(T_cat_range).Named("T_cat_val_p");
            using (Variable.ForEach(T_cat_range))
            {
              T_cat_val_p[T_cat_range] = Variable.DirichletUniform(T_val_range);
            }
            var v3 = Variable.DiscreteUniform(T_cat_range);
            var v1 = Variable.New<Vector>();
            using (Variable.Switch(v3))
            {
                v1.SetTo(Variable.Copy(T_cat_val_p[v3]));
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(v1));
        }

        [Fact]
        public void BinomialMarginalPrototype()
        {
            Variable<int> x = Variable.Binomial(4, 0.1);
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(x));
        }

        [Fact]
        public void BinomialMarginalPrototype2()
        {
          Variable<int> n = Variable.New<int>().Named("n");
          //Variable<int> x = Variable.Binomial(n, 0.1);
          n.ObservedValue = 4;
          var x = Variable.New<int>();
          var b = Variable.Observed(false);
          using (Variable.If(b))
          {          
            x.SetTo(Variable.DiscreteUniform(n + 1));
          }
          using (Variable.IfNot(b))
          {
            x.SetTo(Variable.Binomial(n, 0.1));
          }
          InferenceEngine ie = new InferenceEngine();
          Console.WriteLine(ie.Infer(x));
        }

        // Fails because we lack an appropriate distribution type
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MultinomialMarginalPrototype()
        {
            VariableArray<int> x = Variable.Multinomial(1000, Vector.FromArray(new double[] {0.1, 0.9}));
            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(x));
        }

        [Fact]
        public void ObservedVariableInMarginalPrototype()
        {
            var nDimensions = Variable.New<int>().Named("nDimensions");
            var d = new Range(nDimensions).Named("d");
            var mean = Variable.Observed(Vector.Zero(1)).Named("mean");
            mean.SetValueRange(d);
            var precision = Variable.Observed(PositiveDefiniteMatrix.Identity(1)).Named("precision");
            var x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");

            mean.ObservedValue = Vector.Zero(1);
            nDimensions.ObservedValue = 1;
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer<VectorGaussian>(x));
        }

        [Fact]
        public void MP_VectorGaussian2()
        {
            var nDimensions = Variable.New<int>().Named("nDimensions");
            var d = new Range(nDimensions).Named("d");
            var meanPrior = Variable.New<VectorGaussian>().Named("meanPrior");
            var mean = Variable<Vector>.Random(meanPrior).Named("mean");
            mean.SetValueRange(d);
            mean.AddAttribute(new MarginalPrototype(new VectorGaussian(1)));
            var precision = Variable.Observed(PositiveDefiniteMatrix.Identity(1)).Named("precision");
            var x = Variable.VectorGaussianFromMeanAndPrecision(mean, precision).Named("x");

            mean.ObservedValue = Vector.Zero(1);
            meanPrior.ObservedValue = VectorGaussian.FromMeanAndVariance(0, 1);
            nDimensions.ObservedValue = 1;
            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer<VectorGaussian>(x));
        }

        [Fact]
        public void LocalArrayMarginalPrototypeTest()
        {
            var TA_size = Variable.New<int>().Named("TA_size");
            var TA_range = new Range(TA_size).Named("TA_range");
            var TB_size = Variable.New<int>().Named("TB_size");
            var TB_range = new Range(TB_size).Named("TB_range");
            var TB_B = Variable.Array<int>(TB_range).Named("TB_B");
            TB_B.SetValueRange(TA_range);

            using (Variable.ForEach(TB_range))
            {
                var TA_range_clone = TA_range.Clone();
                var array = Variable.Array<double>(TA_range_clone).Named("array");
                using (Variable.ForEach(TA_range_clone))
                {
                    array[TA_range_clone].SetTo(Variable.Constant(1.0));
                }
                var vector = Variable.Vector(array).Named("vector");
                TB_B[TB_range].SetTo(Variable.Discrete(TA_range, vector));
            }
            var indices = Variable.Array<int>(new Range(2).Named("indices_range")).Named("indices");
            indices.ObservedValue = new int[] { 0, 1 };
            var TB_B_Subarray = Variable.Subarray(TB_B, indices).Named("TB_B_Subarray");
            TB_B_Subarray.ObservedValue = new int[] { 0, 1 };

            TA_size.ObservedValue = 3;
            TB_size.ObservedValue = 3;

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new IVariable[] { TB_B };
            Console.WriteLine("Z=" + engine.Infer(TB_B));
        }

        [Fact]
        public void LocalArrayMarginalPrototypeTest2()
        {
            var Topics_size = Variable.New<int>().Named("Topics_size");
            var Topics_range = new Range(Topics_size).Named("Topics_range");
            var Docs_size = Variable.New<int>().Named("Docs_size");
            var Docs_range = new Range(Docs_size).Named("Docs_range");
            ;
            var Docs_DocTopicDist = Variable.Array<Vector>(Docs_range).Named("Docs_DocTopicDist");

            //  "Cannot index Dirichlet.Uniform(v5.Length) by vint__1[index2] since v5 has an implicit dependency on Docs_range. Try making the dependency explicit by putting v5 into an array indexed by Docs_range"
            using (Variable.ForEach(Docs_range))
            {
                var v5 = Variable.Array<double>(Topics_range).Named("v5"); // the problematic array
                var v6 = Variable.Constant<double>(5);
                var v7 = Variable.Constant<double>(5.1);
                var v8 = Variable.Constant<double>(5.2);
                var v9 = Variable.Constant<double>(5.3);
                var v10 = Variable.Constant<double>(5.4);
                var v11 = Variable.Constant<double>(5.5);
                var v12 = Variable.Constant<double>(5.6);
                v5[Variable.Constant<int>(0)] = v6;
                v5[Variable.Constant<int>(1)] = v7;
                v5[Variable.Constant<int>(2)] = v8;
                v5[Variable.Constant<int>(3)] = v9;
                v5[Variable.Constant<int>(4)] = v10;
                v5[Variable.Constant<int>(5)] = v11;
                v5[Variable.Constant<int>(6)] = v12;
                var v13 = v5;
                Docs_DocTopicDist[Docs_range] = Variable.Dirichlet(Topics_range, Variable.Vector(v13));
            }

            var Occs_size = Variable.New<int>();
            var Occs_range = new Range(Occs_size);
            var Occs_DocID = Variable.Array<int>(Occs_range);
            Occs_DocID.SetValueRange(Docs_range);

            var Occs_OccTopic = Variable.Array<int>(Occs_range);
            Occs_OccTopic.SetValueRange(Topics_range);

            using (Variable.ForEach(Occs_range))
            {
                var v3 = Occs_DocID[Occs_range];
                var v4 = Docs_DocTopicDist[v3];
                Occs_OccTopic[Occs_range] = Variable.Discrete(Topics_range, v4);
            }

            Topics_size.ObservedValue = 7;
            Docs_size.ObservedValue = 3;
            Occs_size.ObservedValue = 4;
            Occs_DocID.ObservedValue = new int[] { 0, 1, 2, 2 };

            InferenceEngine engine = new InferenceEngine();
            engine.OptimiseForVariables = new IVariable[] { Docs_DocTopicDist, Occs_OccTopic };
            Console.WriteLine("Z=" + engine.Infer(Docs_DocTopicDist) + engine.Infer(Occs_OccTopic));
        }

        [Fact]
        public void MarginalPrototypeOfConstantInGateTest()
        {
            Range item = new Range(2).Named("i");
            var bools = Variable.Array<bool>(item).Named("bools");
            bools.ObservedValue = Util.ArrayInit(item.SizeAsInt, i => false);
            var Response = Variable.Array<int>(item).Named("response");
            Response.ObservedValue = Util.ArrayInit(item.SizeAsInt, i => 0);
            var correct = Variable.Array<bool>(item).Named("correct");
            using (Variable.ForEach(item))
            {
                correct[item] = Variable.Bernoulli(0.1);
                using (Variable.If(bools[item]))
                {
                    using (Variable.If(correct[item]))
                        Response[item] = Variable.Constant(1);
                    using (Variable.IfNot(correct[item]))
                        Response[item] =
                            Variable.DiscreteUniform(4) + Variable.Constant(1);
                }
                using (Variable.IfNot(bools[item]))
                {
                    Response[item] = Variable.Constant(0);
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(correct));
            Console.WriteLine(engine.Infer(Response));
        }

        /// <summary>
        /// Fails because StocAnalysis cannot determine a MarginalPrototype that covers both cases of the branch
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void MarginalPrototypeOfConstantInGateTest2()
        {
            Range r = new Range(3);
            var skills = Variable.Array<double>(r).Named("skills");
            var results = Variable.Array<int>(r).Named("results");
            using (Variable.ForEach(r))
            {
                skills[r] = Variable.GaussianFromMeanAndPrecision(25.5, 0.05);
                var b = (skills[r] - 25) > 0;
                using (Variable.If(b))
                {
                    results[r] = 0;
                }
                using (Variable.IfNot(b))
                {
                    results[r] = 1;
                }
            }
            int[] observedResults = new int[3] { 0, 0, 0 };
            Variable.ConstrainEqual(results, observedResults);

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(skills));
        }
    }
}