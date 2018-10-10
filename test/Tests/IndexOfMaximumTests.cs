// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class IndexOfMaximumTests
    {
        public Gaussian[] IndexOfMaximumObservedIndexExplicit(int N, int index, out double logEvidence)
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Enumerable.Range(0, N).Select(o => Variable.GaussianFromMeanAndPrecision(0, 1)).ToArray();
            for (int i = 0; i < N; i++)
            {
                if (i != index)
                    Variable.ConstrainPositive(x[index] - x[i]);
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            var toInfer = x.Select(o => (IVariable)o).ToList();
            toInfer.Add(ev);
            var ca = ie.GetCompiledInferenceAlgorithm(toInfer.ToArray());
            ca.Execute(10);
            logEvidence = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
            return x.Select(o => ca.Marginal<Gaussian>(o.NameInGeneratedCode)).ToArray();
        }

        public Gaussian[] IndexOfMaximumExplicit(Discrete y, out double logEvidence)
        {
            int N = y.Dimension;
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Enumerable.Range(0, N).Select(o => Variable.GaussianFromMeanAndPrecision(0, 1)).ToArray();
            var yVar = Variable<int>.Random(y).Named("y");
            for (int index = 0; index < N; index++)
            {
                using (Variable.Case(yVar, index))
                {
                    for (int i = 0; i < N; i++)
                    {
                        if (i != index)
                            Variable.ConstrainPositive(x[index] - x[i]);
                    }
                }
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            var toInfer = x.Select(o => (IVariable)o).ToList();
            toInfer.Add(ev);
            var ca = ie.GetCompiledInferenceAlgorithm(toInfer.ToArray());
            ca.Execute(10);
            logEvidence = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
            return x.Select(o => ca.Marginal<Gaussian>(o.NameInGeneratedCode)).ToArray();
        }

        public Gaussian[] IndexOfMaximumObservedIndexFactor(int N, int index, out double logEvidence)
        {
            var n = new Range(N);
            var ev = Variable.Bernoulli(0.5);
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var p = IndexOfMaximum(x);
            p.ObservedValue = index;
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        [Fact]
        public void IndexOfMaximumObservedIndexTest()
        {
            int N = 3;
            int index = 0;
            double expEv, facEv;
            var exp = IndexOfMaximumObservedIndexExplicit(N, index, out expEv);
            var fac = IndexOfMaximumObservedIndexFactor(N, index, out facEv);
            for (int i = 0; i < N; i++)
            {
                Console.WriteLine("exp: " + exp[i] + " fac: " + fac[i]);
                Assert.True(exp[i].MaxDiff(fac[i]) < 1e-8);
            }
            Assert.True(MMath.AbsDiff(expEv, facEv) < 1e-8);
        }


        public Gaussian[] IndexOfMaximumFactorGate(Discrete y, out double logEvidence)
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            int N = y.Dimension;
            var n = new Range(N).Named("n");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var yVar = Variable<int>.Random(y).Named("y");
            for (int index = 0; index < N; index++)
            {
                using (Variable.Case(yVar, index))
                {
                    //var temp = Variable.Observed<int>(index).Named("temp"+index) ;
                    //temp.SetTo(Variable<int>.Factor(MMath.IndexOfMaximumDouble, x).Named("fac"+index));
                    var temp = IndexOfMaximum(x).Named("temp" + index);
                    temp.ObservedValue = index;
                }
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            ie.ModelName = "FactorGate";
            ie.OptimiseForVariables = new List<IVariable>() { x, ev };
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        public Gaussian[] IndexOfMaximumFactorGate2(Discrete y, out double logEvidence)
        {
            var ev = Variable.Bernoulli(0.5).Named("ev");
            int N = y.Dimension;
            var n = new Range(N).Named("n");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var yVar = Variable<int>.Random(y).Named("y");
            var indices = Variable.Observed(new int[] { 0, 1, 2 }, n);
            yVar.SetValueRange(n);
            using (Variable.Switch(yVar))
            {
                indices[yVar] = IndexOfMaximum(x).Named("temp");
            }
            block.CloseBlock();
            var ie = new InferenceEngine();
            //ie.NumberOfIterations = 2;
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        // This factor is not included as a Variable.IndexOfMaximum shortcut 
        // because you generally get better results with the expanded version, when part of a larger model.
        // Eventually the factor should be deprecated.
        public static Variable<int> IndexOfMaximum(VariableArray<double> array)
        {
            var p = Variable<int>.Factor(MMath.IndexOfMaximumDouble, array).Named("p");
            p.SetValueRange(array.Range);
            return p;
        }

        public Gaussian[] IndexOfMaximumFactorCA(Discrete y, out double logEvidence)
        {
            int N = y.Dimension;
            var n = new Range(N);
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var p = IndexOfMaximum(x);
            Variable.ConstrainEqualRandom(p, y);
            block.CloseBlock();
            var ie = new InferenceEngine();
            ie.ModelName = "IndexOfMaximumCA";
            //ie.NumberOfIterations = 2;
            var toinfer = new List<IVariable>();
            toinfer.Add(x);
            toinfer.Add(ev);
            ie.OptimiseForVariables = toinfer;

            var ca = ie.GetCompiledInferenceAlgorithm(x, ev);
            ca.Reset();
            Gaussian[] xPost = null;
            logEvidence = 0;
            for (int i = 0; i < 10; i++)
            {
                ca.Update(1);
                logEvidence = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
                xPost = ca.Marginal<Gaussian[]>(x.NameInGeneratedCode);
            }
            return xPost;
        }

        public Gaussian[] IndexOfMaximumFactorIE(Discrete y, out double logEvidence)
        {
            int N = y.Dimension;
            var n = new Range(N);
            var ev = Variable.Bernoulli(0.5).Named("ev");
            var block = Variable.If(ev);
            var x = Variable.Array<double>(n).Named("x");
            x[n] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(n);
            var p = IndexOfMaximum(x).Named("p");
            Variable.ConstrainEqualRandom(p, y);
            block.CloseBlock();
            var ie = new InferenceEngine();
            ie.ModelName = "IndexOfMaximumIE";
            var toinfer = new List<IVariable>();
            toinfer.Add(x);
            toinfer.Add(ev);
            ie.OptimiseForVariables = toinfer;
            logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            return ie.Infer<Gaussian[]>(x);
        }

        [Fact]
        public void IndexOfMaximumTest()
        {
            var y = new Discrete(0.1, 0.4, 0.5);
            double expEv, gateEv, facEv;
            Console.WriteLine("explicit");
            var exp = IndexOfMaximumExplicit(y, out expEv);
            Console.WriteLine("gate");
            var gate = IndexOfMaximumFactorGate(y, out gateEv);
            Console.WriteLine("compiled alg");
            var facCA = IndexOfMaximumFactorCA(y, out facEv);
            Console.WriteLine("engine");
            var facIE = IndexOfMaximumFactorIE(y, out facEv);
            for (int i = 0; i < y.Dimension; i++)
            {
                Console.WriteLine("exp: " + exp[i] + " facCA: " + facCA[i] + " fac: " + facIE[i] + " gate: " + gate[i]);
                Assert.True(exp[i].MaxDiff(facCA[i]) < 1e-8);
                Assert.True(exp[i].MaxDiff(gate[i]) < 1e-8);
                Assert.True(exp[i].MaxDiff(facIE[i]) < 1e-8);
            }
            Assert.True(MMath.AbsDiff(expEv, facEv) < 1e-8);
            Assert.True(MMath.AbsDiff(expEv, gateEv) < 1e-8);
        }

        [Fact]
        public void IndexOfMaximumFastTest()
        {
            int n = 5;
            Range item = new Range(n).Named("item");
            var priors = Variable<Gaussian>.Array(item);
            priors.ObservedValue = Util.ArrayInit(n, i => Gaussian.FromMeanAndVariance(i * 0.5, i));
            var x = Variable.Array<double>(item).Named("x");
            x[item] = Variable<double>.Random(priors[item]);
            var y = Variable<int>.Factor(MMath.IndexOfMaximumDouble, x);
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;
            string format = "f4";
            var yActual = engine.Infer<Discrete>(y);
            Console.WriteLine("Quadratic: {0}", yActual.ToString(format));

            // Monte Carlo estimate
            Rand.Restart(0);
            DiscreteEstimator est = new DiscreteEstimator(n);
            for (int iter = 0; iter < 100000; iter++)
            {
                double[] samples = Util.ArrayInit(n, i => priors.ObservedValue[i].Sample());
                int argmax = MMath.IndexOfMaximumDouble(samples);
                est.Add(argmax);
            }
            var yExpected = est.GetDistribution(Discrete.Uniform(n));
            Console.WriteLine("Sampling:  {0}", yExpected.ToString(format));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-2);

            engine.Compiler.GivePriorityTo(typeof(IndexOfMaximumOp_Fast));
            yActual = engine.Infer<Discrete>(y);
            Console.WriteLine("Linear:    {0}", yActual.ToString(format));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-2);

            bool compareApproximation = false;
            if (compareApproximation)
            {
                var yPost2 = IndexOfMaximumOp_Fast.IndexOfMaximumDoubleAverageConditional(priors.ObservedValue, Discrete.Uniform(n));
                Console.WriteLine(yPost2);
                var yPost3 = IndexOfMaximumOp_Fast.IndexOfMaximumDoubleAverageConditional2(priors.ObservedValue, Discrete.Uniform(n));
                Console.WriteLine(yPost3);
            }
        }
    }
}
