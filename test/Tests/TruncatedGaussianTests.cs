// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Xunit;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Summary description for TruncatedGaussianTests
    /// </summary>

    public class TruncatedGaussianTests
    {
        [Fact]
        public void TruncatedGaussianNormaliser()
        {
            double a = 0, b = 2;
            var g = new TruncatedGaussian(3, 1, a, b);
            double Z = Quadrature.AdaptiveTrapeziumRule(x => System.Math.Exp(g.GetLogProb(x)), 32, a, b, 1e-10, 10000);
            Assert.True((1.0 - Z) < 1e-4);
        }

        [Fact]
        public void TruncatedGaussianConstrainPositiveEvidence()
        {
            var ev = Variable.Bernoulli(0.5);
            var prior = new Gaussian(0, 1);
            using (Variable.If(ev))
            {
                var x = Variable.Random(prior);
                Variable.ConstrainPositive(x);
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            double trueZ = -System.Math.Log(2.0);
            double logEvidence = ie.Infer<Bernoulli>(ev).LogOdds;
            Console.WriteLine(logEvidence + " should be " + trueZ);
            Assert.True(System.Math.Abs(trueZ - logEvidence) < 1e-10);
        }

        [Fact]
        public void TruncatedGaussianEntersGate()
        {
            var x = Variable.TruncatedGaussian(0, 1, double.NegativeInfinity, double.PositiveInfinity);
            var l = Variable.Bernoulli(0.9).Named("l");
            using (Variable.If(l))
            {
                var True = Variable.IsPositive(x).Named("VarTrue");
                True.ObservedValue = true;
            }
            using (Variable.IfNot(l))
            {
                var False = Variable.IsPositive(x).Named("VarFalse");
                False.ObservedValue = false;
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(x));
        }

        [Fact]
        public void TruncatedGaussianEntersGate2()
        {
            var x = Variable.GaussianFromMeanAndPrecision(0, 1).Named("x");
            var xCopy = Variable.Copy(x);
            xCopy.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            var l = Variable.Bernoulli(0.9).Named("l");
            using (Variable.If(l))
            {
                var True = Variable.IsPositive(xCopy).Named("VarTrue");
                True.ObservedValue = true;
            }
            using (Variable.IfNot(l))
            {
                var False = Variable.IsPositive(xCopy).Named("VarFalse");
                False.ObservedValue = false;
            }
            var ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(x));
        }

        [Fact]
        public void TruncatedGaussianConstrainBetween()
        {
            double lowerBound = 0, upperBound = 1;

            var x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable.ConstrainBetween(x, lowerBound, upperBound);
            var ie = new InferenceEngine();
            var epPost = ie.Infer<Gaussian>(x);
            Console.WriteLine(epPost);

            x = Variable.TruncatedGaussian(0, 1, lowerBound, upperBound);
            var y = Variable.Copy(x);
            y.AddAttribute(new MarginalPrototype(new Gaussian()));
            ie = new InferenceEngine(new VariationalMessagePassing());
            var vmpPost = ie.Infer<Gaussian>(y);
            Assert.True(vmpPost.MaxDiff(epPost) < 1e-8);
            Console.WriteLine(vmpPost);

            x = Variable.GaussianFromMeanAndVariance(0, 1);
            y = Variable.Copy(x);
            y.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainBetween(y, lowerBound, upperBound);
            vmpPost = ie.Infer<Gaussian>(x);
            Assert.True(vmpPost.MaxDiff(epPost) < 1e-8);
            Console.WriteLine(vmpPost);

            x = Variable.TruncatedGaussian(0, 1, double.NegativeInfinity, double.PositiveInfinity);
            Variable.ConstrainBetween(x, lowerBound, upperBound);
            y = Variable.Copy(x);
            y.AddAttribute(new MarginalPrototype(new Gaussian()));
            vmpPost = ie.Infer<Gaussian>(y);
            Assert.True(vmpPost.MaxDiff(epPost) < 1e-8);
            Console.WriteLine(vmpPost);

            // Don't support this because can't handle IsBetween=False; 
            //x = Variable.TruncatedGaussian(0, 1, double.NegativeInfinity, double.PositiveInfinity);
            //var b= Variable.IsBetween(x, lowerBound, upperBound);
            //b.ObservedValue = true; 
            //y = Variable.Copy(x);
            //y.AddAttribute(new MarginalPrototype(new Gaussian()));
            //vmpPost = ie.Infer<Gaussian>(y);
            //Assert.True(vmpPost.MaxDiff(epPost) < 1e-8);
            //Console.WriteLine(vmpPost);

            x = Variable.TruncatedGaussian(0, double.PositiveInfinity, lowerBound, upperBound);
            y = Variable.Copy(x);
            y.AddAttribute(new MarginalPrototype(new Gaussian()));
            Variable.ConstrainEqualRandom(y, new Gaussian(0, 1));
            vmpPost = ie.Infer<Gaussian>(y);
            Assert.True(vmpPost.MaxDiff(epPost) < 1e-8);
            Console.WriteLine(vmpPost);

        }

        [Fact]
        public void TruncatedGaussianIsPositive()
        {
            var x = Variable.TruncatedGaussian(0, 1, double.NegativeInfinity, double.PositiveInfinity);
            Variable.ConstrainPositive(x);
            var y = Variable.Copy(x);
            y.AddAttribute(new MarginalPrototype(new Gaussian()));
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var vmpPost = ie.Infer<Gaussian>(y);
            Console.WriteLine("VMP: " + vmpPost);

            x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
            y = Variable.Copy(x).Named("y");
            y.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
            Variable.ConstrainPositive(y);
            Console.WriteLine("VMP: " + ie.Infer(x));

            x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable.ConstrainPositive(x);
            var vmp2Post = ie.Infer<Gaussian>(x);
            Console.WriteLine("VMP2: " + vmp2Post);

            x = Variable.GaussianFromMeanAndVariance(0, 1);
            var b = Variable.IsPositive(x);
            b.ObservedValue = true;
            vmp2Post = ie.Infer<Gaussian>(x);
            Console.WriteLine("VMP2: " + vmp2Post);

            x = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable.ConstrainPositive(x);
            ie = new InferenceEngine(new ExpectationPropagation());
            var epPost = ie.Infer<Gaussian>(x);
            Console.WriteLine("EP: " + epPost);

            Assert.True(vmpPost.MaxDiff(epPost) < 1e-8);
        }

        [Fact]
        public void ConstrainPositive_Throws()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                var x = Variable.GaussianFromMeanAndVariance(0, 1);
                var x2 = Variable.Copy(x);
                Variable.ConstrainPositive(x2);
                var ie = new InferenceEngine(new VariationalMessagePassing());
                var vmp2Post = ie.Infer<Gaussian>(x);
                Console.WriteLine("VMP2: " + vmp2Post);
            });
        }

        [Fact]
        public void IsGreaterThan()
        {
            var x2 = Variable.GaussianFromMeanAndVariance(0, 1);
            var y2 = Variable.GaussianFromMeanAndVariance(0, 1);
            Variable.ConstrainPositive(x2 - y2);
            var ie2 = new InferenceEngine(new ExpectationPropagation());
            var epPost = ie2.Infer<Gaussian>(x2);
            Console.WriteLine("EP: " + epPost);
        }

        [Fact]
        public void ConstrainGreaterThan_Throws()
        {
            Assert.Throws<CompilationFailedException>(() =>
            {
                // This will throw the appropriate exception- we don't currently know how to 
                // constrain derived variables in VMP
                var x = Variable.GaussianFromMeanAndVariance(0, 1);
                var y = Variable.GaussianFromMeanAndVariance(0, 1);
                var z = x - y;
                var w = Variable.Copy(z);
                w.AddAttribute(new MarginalPrototype(new TruncatedGaussian()));
                Variable.ConstrainPositive(w);
                var ie = new InferenceEngine(new VariationalMessagePassing());
                var vmpPost = ie.Infer<Gaussian>(x);
                Console.WriteLine("VMP: " + vmpPost);
            });
        }
    }
}
