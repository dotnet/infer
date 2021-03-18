// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class VmpGateTests
    {
        // demonstrates the effect of changing the extent of a switch block
        internal void SwitchBoundaryExample()
        {
            for (int trial = 0; trial < 2; trial++)
            {
                Range c = new Range(2).Named("c");
                VariableArray<double> m = Variable.Constant(new double[] {0.0, 3.0}, c).Named("m");
                Variable<double> y = Variable.New<double>().Named("y");
                Variable<int> z = Variable.Discrete(c, 0.5, 0.5).Named("z");
                SwitchBlock block = Variable.Switch(z);
                y.SetTo(Variable.GaussianFromMeanAndPrecision(m[z], 1.0));
                if (trial == 0) block.CloseBlock(); // smallest possible switch block (least accurate)
                Variable<double> y2 = Variable.GaussianFromMeanAndPrecision(y, 1.0).Named("y2");
                Variable<double> y3 = Variable.GaussianFromMeanAndPrecision(y2, 1.0).Named("y3");
                y3.ObservedValue = 3.0;
                if (trial == 1) block.CloseBlock(); // largest possible switch block (most accurate)
                InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
                Console.WriteLine(engine.Infer(z));
                // trial==0: Discrete(0.1107 0.8893)
                // trial==1: Discrete(0.1824 0.8176)
            }
        }

        [Fact]
        public void DifferentSizedDirichletInGate()
        {
            var N = new Range(2);
            var numCats = Variable.Constant(new int[] {2, 3}, N);
            var catRange = new Range(numCats[N]);
            var probs = Variable.Array<Vector>(N);
            probs[N] = Variable.DirichletUniform(catRange);
            var b = Variable.Bernoulli(.5);
            using (Variable.If(b))
            {
                var data = Variable.Array<int>(N);
                data[N] = Variable.Discrete(probs[N]);
                data.ObservedValue = new int[] {1, 2};
            }

            var ie = new InferenceEngine(new VariationalMessagePassing());

            Console.WriteLine(ie.Infer(probs));
        }

        [Fact]
        public void IfObservedThenIfRandomDerivedElseStochastic()
        {
            Variable<bool> y = Variable.New<bool>().Named("y");
            double cPrior = 0.1;
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            double yPrior1 = 0.2;
            double yPrior2 = 0.3;
            double yPriorF = 0.4;
            double yLike = 0.45;
            Variable<bool> obs = Variable.New<bool>().Named("obs");
            using (Variable.If(obs))
            {
                using (Variable.If(c))
                {
                    y.SetTo(Variable.Bernoulli(yPrior1));
                }
                using (Variable.IfNot(c))
                {
                    y.SetTo(Variable.Bernoulli(yPrior2));
                }
            }
            using (Variable.IfNot(obs))
            {
                y.SetTo(Variable.Bernoulli(yPriorF));
            }
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            for (int iter = 0; iter < 2; iter++)
            {
                double yPost;
                if (iter == 0)
                {
                    obs.ObservedValue = true;
                    double yPriorT = cPrior*yPrior1 + (1 - cPrior)*yPrior2;
                    double z = yLike*yPriorT + (1 - yLike)*(1 - yPriorT);
                    yPost = yLike*yPriorT/z;
                }
                else
                {
                    obs.ObservedValue = false;
                    double z = yLike*yPriorF + (1 - yLike)*(1 - yPriorF);
                    yPost = yLike*yPriorF/z;
                }
                Bernoulli yExpected = new Bernoulli(yPost);
                Bernoulli yActual = engine.Infer<Bernoulli>(y);
                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            }
        }

        [Fact]
        public void IfObservedThenDerivedElseStochasticPerArrayElement()
        {
            Range item = new Range(2).Named("item");
            VariableArray<bool> isDerived = Variable.Array<bool>(item).Named("isDerived");
            VariableArray<bool> y = Variable.Array<bool>(item).Named("y");
            double xPrior = 0.1;
            double yPrior = 0.2;
            double yCondT = 0.3;
            double yCondF = 0.4;
            double yLike = 0.45;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            using (Variable.ForEach(item))
            {
                using (Variable.If(isDerived[item]))
                {
                    y[item].SetTo(!x);
                    Variable.ConstrainEqualRandom(y[item], new Bernoulli(yCondT));
                }
                using (Variable.IfNot(isDerived[item]))
                {
                    y[item].SetTo(Variable.Bernoulli(yPrior));
                    Variable.ConstrainEqualRandom(y[item], new Bernoulli(yCondF));
                }
                Variable.ConstrainEqualRandom(y[item], new Bernoulli(yLike));
            }
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            isDerived.ObservedValue = new bool[] {true, false};
            double z1 = (1 - xPrior)*yCondT*yLike + xPrior*(1 - yCondT)*(1 - yLike);
            Bernoulli yExpected1 = new Bernoulli((1 - xPrior)*yCondT*yLike/z1);
            double z0 = yPrior*yCondF*yLike + (1 - yPrior)*(1 - yCondF)*(1 - yLike);
            Bernoulli yExpected0 = new Bernoulli(yPrior*yCondF*yLike/z0);
            IDistribution<bool[]> yExpected = Distribution<bool>.Array(new Bernoulli[] {yExpected1, yExpected0});
            object yActual = engine.Infer(y);
            Console.WriteLine(StringUtil.JoinColumns("y = ", yActual, " should be ", yExpected));
            Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
        }

        [Fact]
        public void IfObservedThenDerivedElseStochastic()
        {
            Variable<bool> isDerived = Variable.New<bool>().Named("isDerived");
            Variable<bool> y = Variable.New<bool>().Named("y");
            double xPrior = 0.1;
            double yPrior = 0.2;
            double yCondT = 0.3;
            double yCondF = 0.4;
            double yLike = 0.45;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            using (Variable.If(isDerived))
            {
                y.SetTo(!x);
                Variable.ConstrainEqualRandom(y, new Bernoulli(yCondT));
            }
            using (Variable.IfNot(isDerived))
            {
                y.SetTo(Variable.Bernoulli(yPrior));
                Variable.ConstrainEqualRandom(y, new Bernoulli(yCondF));
            }
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            for (int trial = 0; trial < 2; trial++)
            {
                Bernoulli yExpected;
                if (trial == 0)
                {
                    isDerived.ObservedValue = true;
                    double z = (1 - xPrior)*yCondT*yLike + xPrior*(1 - yCondT)*(1 - yLike);
                    yExpected = new Bernoulli((1 - xPrior)*yCondT*yLike/z);
                }
                else
                {
                    isDerived.ObservedValue = false;
                    double z = yPrior*yCondF*yLike + (1 - yPrior)*(1 - yCondF)*(1 - yLike);
                    yExpected = new Bernoulli(yPrior*yCondF*yLike/z);
                }
                Bernoulli yActual = engine.Infer<Bernoulli>(y);
                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            }
        }

        [Fact]
        public void IfRandomThenDerivedElseStochastic()
        {
            Variable<double> pDerived = Variable.New<double>().Named("pDerived");
            Variable<bool> isDerived = Variable.Bernoulli(pDerived).Named("isDerived");
            Variable<bool> y = Variable.New<bool>().Named("y");
            double xPrior = 0.1;
            double yPrior = 0.2;
            double yCondT = 0.3;
            double yCondF = 0.4;
            double yLike = 0.45;
            Variable<bool> x = Variable.Bernoulli(xPrior).Named("x");
            using (Variable.If(isDerived))
            {
                y.SetTo(!x);
                Variable.ConstrainEqualRandom(y, new Bernoulli(yCondT));
            }
            using (Variable.IfNot(isDerived))
            {
                y.SetTo(Variable.Bernoulli(yPrior));
                Variable.ConstrainEqualRandom(y, new Bernoulli(yCondF));
            }
            Variable.ConstrainEqualRandom(y, new Bernoulli(yLike));
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            for (int trial = 0; trial < 2; trial++)
            {
                Bernoulli yExpected;
                if (trial == 0)
                {
                    pDerived.ObservedValue = 1.0;
                    double z = (1 - xPrior)*yCondT*yLike + xPrior*(1 - yCondT)*(1 - yLike);
                    yExpected = new Bernoulli((1 - xPrior)*yCondT*yLike/z);
                }
                else
                {
                    pDerived.ObservedValue = 0.0;
                    double z = yPrior*yCondF*yLike + (1 - yPrior)*(1 - yCondF)*(1 - yLike);
                    yExpected = new Bernoulli(yPrior*yCondF*yLike/z);
                }
                Bernoulli yActual = engine.Infer<Bernoulli>(y);
                Console.WriteLine("y = {0} should be {1}", yActual, yExpected);
                Assert.True(yExpected.MaxDiff(yActual) < 1e-10);
            }
        }

        [Fact]
        public void MixtureOfManyBernoullis3()
        {
            double p1 = 0.1, p2 = 0.2, px = 0.3;
            double cPrior = 0.4;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range item = new Range(2).Named("item");
            VariableArray<double> probs = Variable.Constant(new double[] {p1, p2}, item);
            VariableArray<bool> h = Variable.Array<bool>(item).Named("h");
            h[item] = Variable.Bernoulli(probs[item]);
            Variable<int> i = Variable.Discrete(item, new double[] {cPrior, 1 - cPrior}).Named("i");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Switch(i))
            {
                x.SetTo(Variable.Copy(h[i]));
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(px));
            block.CloseBlock();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(vmp);
            double xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double xExpected = 0.107480286770265;
            double evExpected = -0.468610820117876;
            Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
            Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
        }

        [Fact]
        public void MixtureOfBernoullis3()
        {
            double p1 = 0.1, p2 = 0.2, px = 0.3;
            double cPrior = 0.4;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            Variable<bool> h1 = Variable.Bernoulli(p1).Named("h1");
            Variable<bool> h2 = Variable.Bernoulli(p2).Named("h2");
            using (Variable.If(c))
            {
                x.SetTo(Variable.Copy(h1));
            }
            using (Variable.IfNot(c))
            {
                x.SetTo(Variable.Copy(h2));
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(px));
            block.CloseBlock();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(vmp);
            double xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
            double cActual = engine.Infer<Bernoulli>(c).GetProbTrue();
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            double xExpected;
            double cExpected;
            double evExpected;
            if (true)
            {
                // exact result
                double sumCT = p1*px + (1 - p1)*(1 - px);
                double sumCF = p2*px + (1 - p2)*(1 - px);
                double Z = cPrior*sumCT + (1 - cPrior)*sumCF;
                cExpected = cPrior*sumCT/Z;
                Console.WriteLine("exact c = {0}", cExpected);
                xExpected = (cPrior*p1*px + (1 - cPrior)*p2*px)/Z;
                Console.WriteLine("exact x = {0}", xExpected);
                evExpected = System.Math.Log(Z);
                Console.WriteLine("exact evidence = {0}", evExpected);
            }
            if (true)
            {
                // VMP by hand:
                // p(h1,h2,c) = fc(c) f1(h1) (fx(h1))^c f2(h2) (fx(h2))^(1-c)
                // q(h1) = f1(h1) * fx(h1)^q(c=T)
                // q(h2) = f2(h2) * fx(h2)^q(c=F)
                // q(c) = fc(c)*Bernoulli(c;q1/(q1+q2))
                // where q1 = px^q(h1=T)*(1-px)^q(h1=F)
                //       q2 = px^q(h2=T)*(1-px)^q(h2=F)
                Bernoulli qh1 = new Bernoulli();
                Bernoulli qh2 = new Bernoulli();
                Bernoulli qc = new Bernoulli(cPrior);
                Bernoulli f1 = new Bernoulli(p1);
                Bernoulli f2 = new Bernoulli(p2);
                Bernoulli fx = new Bernoulli(px);
                Bernoulli fc = new Bernoulli(cPrior);
                for (int iter = 0; iter < 100; iter++)
                {
                    qh1 = f1*(fx ^ qc.GetProbTrue());
                    qh2 = f2*(fx ^ qc.GetProbFalse());
                    double q1 = qh1.GetAverageLog(fx);
                    double q2 = qh2.GetAverageLog(fx);
                    qc = fc*Bernoulli.FromLogOdds(q1 - q2);
                }
                xExpected = qc.GetProbTrue()*qh1.GetProbTrue() + qc.GetProbFalse()*qh2.GetProbTrue();
                cExpected = qc.GetProbTrue();
                evExpected = qc.GetAverageLog(fc) - qc.GetAverageLog(qc);
                evExpected += qh1.GetAverageLog(f1) - qh1.GetAverageLog(qh1);
                evExpected += qh2.GetAverageLog(f2) - qh2.GetAverageLog(qh2);
                evExpected += qc.GetProbTrue()*qh1.GetAverageLog(fx) + qc.GetProbFalse()*qh2.GetAverageLog(fx);
                // xExpected = 0.107480286770265
                // evExpected = -0.468610820117876
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(cActual, cExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
        }

        [Fact]
        public void MixtureOfBernoullis2()
        {
            double p1 = 0.1, p2 = 0.2, px = 0.3, p3 = 0.6, p4 = 0.7;
            double cPrior = 0.4;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            bool innerConstraint = true;
            using (Variable.If(c))
            {
                x.SetTo(Variable.Bernoulli(p1));
                if (innerConstraint)
                {
                    Variable.ConstrainEqualRandom(x, new Bernoulli(p3));
                }
            }
            using (Variable.IfNot(c))
            {
                x.SetTo(Variable.Bernoulli(p2));
                if (innerConstraint)
                {
                    Variable.ConstrainEqualRandom(x, new Bernoulli(p4));
                }
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(px));
            block.CloseBlock();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(vmp);
            double xActual, xExpected;
            double cActual, cExpected;
            double evActual, evExpected;
            if (true)
            {
                double Z;
                if (innerConstraint)
                {
                    Z = px*(p1*p3*cPrior + p2*p4*(1 - cPrior)) + (1 - px)*((1 - p1)*(1 - p3)*cPrior + (1 - p2)*(1 - p4)*(1 - cPrior));
                    xExpected = px*(p1*p3*cPrior + p2*p4*(1 - cPrior))/Z;
                    cExpected = cPrior*(p1*p3*px + (1 - p1)*(1 - p3)*(1 - px))/Z;
                    evExpected = System.Math.Log(Z);
                }
                else
                {
                    Z = px*(p1*cPrior + p2*(1 - cPrior)) + (1 - px)*((1 - p1)*cPrior + (1 - p2)*(1 - cPrior));
                    xExpected = px*(p1*cPrior + p2*(1 - cPrior))/Z;
                    cExpected = cPrior*(p1*px + (1 - p1)*(1 - px))/Z;
                    evExpected = System.Math.Log(Z);
                }
                Console.WriteLine("exact c = {0}", cExpected);
                Console.WriteLine("exact x = {0}", xExpected);
                Console.WriteLine("exact evidence = {0}", evExpected);
            }
            if (true)
            {
                // VMP by hand:
                // p(h1,h2,c) = fc(c) (fx(h1) f1(h1) f3(h1))^c (fx(h2) f2(h2) f4(h2))^(1-c)
                // q(h1) = fx(h1) f1(h1) f3(h1)
                // q(c) = fc(c) Bernoulli(c;q1/(q1+q2))
                // where q1 = exp(sum_h1 q(h1) log(fx(h1) f1(h1) f3(h1)))
                Bernoulli f1 = new Bernoulli(p1);
                Bernoulli f2 = new Bernoulli(p2);
                Bernoulli f3 = new Bernoulli(p3);
                Bernoulli f4 = new Bernoulli(p4);
                Bernoulli fx = new Bernoulli(px);
                Bernoulli fc = new Bernoulli(cPrior);
                Bernoulli qh1 = new Bernoulli();
                Bernoulli qh2 = new Bernoulli();
                Bernoulli qc = fc;
                for (int iter = 0; iter < 1; iter++)
                {
                    qh1 = fx*f1;
                    qh2 = fx*f2;
                    if (innerConstraint)
                    {
                        qh1 *= f3;
                        qh2 *= f4;
                    }
                    double q1 = qh1.GetAverageLog(fx) + qh1.GetAverageLog(f1) - qh1.GetAverageLog(qh1);
                    double q2 = qh2.GetAverageLog(fx) + qh2.GetAverageLog(f2) - qh2.GetAverageLog(qh2);
                    if (innerConstraint)
                    {
                        q1 += qh1.GetAverageLog(f3);
                        q2 += qh2.GetAverageLog(f4);
                    }
                    qc = fc*Bernoulli.FromLogOdds(q1 - q2);
                }
                xExpected = qc.GetProbTrue()*qh1.GetProbTrue() + qc.GetProbFalse()*qh2.GetProbTrue();
                cExpected = qc.GetProbTrue();
                evExpected = qc.GetProbTrue()*(qh1.GetAverageLog(fx) + qh1.GetAverageLog(f1) - qh1.GetAverageLog(qh1));
                evExpected += qc.GetProbFalse()*(qh2.GetAverageLog(fx) + qh2.GetAverageLog(f2) - qh2.GetAverageLog(qh2));
                if (innerConstraint)
                {
                    evExpected += qc.GetProbTrue()*qh1.GetAverageLog(f3);
                    evExpected += qc.GetProbFalse()*qh2.GetAverageLog(f4);
                }
                evExpected += qc.GetAverageLog(fc) - qc.GetAverageLog(qc);
                // cExpected = 0.461538461538461
                // xExpected = 0.138461538461538
                // evExpected = -1.45243416362444
                cActual = engine.Infer<Bernoulli>(c).GetProbTrue();
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(cActual, cExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
            if (true)
            {
                // VMP by hand:
                // p(x,c) = fc(c) fx(x) (f1(x) f3(x))^c (f2(x) f4(x))^(1-c)
                // q(x) = fx(x) * (f1(x) f3(x))^q(c=T) * (f2(x) f4(x))^q(c=F)
                // q(c) = fc(c)*Bernoulli(c;q1/(q1+q2))
                // where q1 = p1^q(x=T)*(1-p1)^q(x=F) * p3^q(x=T)*(1-p3)^q(x=F)
                //       q2 = p2^q(x=T)*(1-p2)^q(x=F) * p4^q(x=T)*(1-p4)^q(x=F)
                Bernoulli qx = new Bernoulli();
                Bernoulli qc = new Bernoulli(cPrior);
                Bernoulli f1 = new Bernoulli(p1);
                Bernoulli f2 = new Bernoulli(p2);
                Bernoulli f3 = new Bernoulli(p3);
                Bernoulli f4 = new Bernoulli(p4);
                Bernoulli fx = new Bernoulli(px);
                Bernoulli fc = new Bernoulli(cPrior);
                for (int iter = 0; iter < 100; iter++)
                {
                    qx = fx*(f1 ^ qc.GetProbTrue())*(f2 ^ qc.GetProbFalse());
                    if (innerConstraint) qx *= (f3 ^ qc.GetProbTrue())*(f4 ^ qc.GetProbFalse());
                    double q1 = qx.GetAverageLog(f1);
                    double q2 = qx.GetAverageLog(f2);
                    if (innerConstraint)
                    {
                        q1 += qx.GetAverageLog(f3);
                        q2 += qx.GetAverageLog(f4);
                    }
                    qc = fc*Bernoulli.FromLogOdds(q1 - q2);
                }
                cExpected = qc.GetProbTrue();
                xExpected = qx.GetProbTrue();
                evExpected = qc.GetAverageLog(fc) - qc.GetAverageLog(qc) + qx.GetAverageLog(fx) - qx.GetAverageLog(qx);
                evExpected += qc.GetProbTrue()*qx.GetAverageLog(f1) + qc.GetProbFalse()*qx.GetAverageLog(f2);
                if (innerConstraint)
                {
                    evExpected += qc.GetProbTrue()*qx.GetAverageLog(f3) + qc.GetProbFalse()*qx.GetAverageLog(f4);
                }
                // xExpected = 0.12297872337406
                // evExpected = -1.47320387037043
                vmp.UseGateExitRandom = true;
                engine.Algorithm = vmp;
                cActual = engine.Infer<Bernoulli>(c).GetProbTrue();
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(cActual, cExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
        }

        [Fact]
        public void MixtureOfManyBernoullis2()
        {
            double p1 = 0.1, p2 = 0.2, px = 0.3, p3 = 0.6, p4 = 0.7;
            double cPrior = 0.4;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Range r = new Range(2);
            VariableArray<double> probs = Variable.Constant(new double[] {p1, p2}, r);
            VariableArray<Bernoulli> berns = Variable.Constant(new Bernoulli[] {new Bernoulli(p3), new Bernoulli(p4)}, r);
            Variable<int> i = Variable.Discrete(r, new double[] {cPrior, 1 - cPrior}).Named("i");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Switch(i))
            {
                x.SetTo(Variable.Bernoulli(probs[i]));
                Variable.ConstrainEqualRandom<bool, Bernoulli>(x, berns[i]);
            }
            Variable.ConstrainEqualRandom(x, new Bernoulli(px));
            block.CloseBlock();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(vmp);
            double xActual, xExpected;
            double evActual, evExpected;
            if (true)
            {
                xExpected = 0.138461538461538;
                evExpected = -1.45243416362444;
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
            if (true)
            {
                xExpected = 0.12297872337406;
                evExpected = -1.47320387037043;
                vmp.UseGateExitRandom = true;
                engine.Algorithm = vmp;
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
        }

        [Fact]
        public void MixtureOfBernoullis()
        {
            double p1 = 0.1, p2 = 0.2;
            double cPrior = 0.4;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<bool> c = Variable.Bernoulli(cPrior).Named("c");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.If(c))
            {
                x.SetTo(Variable.Bernoulli(p1));
            }
            using (Variable.IfNot(c))
            {
                x.SetTo(Variable.Bernoulli(p2));
            }
            block.CloseBlock();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(vmp);
            double xActual, xExpected;
            double cActual, cExpected;
            double evActual, evExpected;
            if (true)
            {
                cExpected = cPrior;
                xExpected = p1*cPrior + p2*(1 - cPrior);
                evExpected = 0;
                cActual = engine.Infer<Bernoulli>(c).GetProbTrue();
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(cActual, cExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-5) < 1e-10);
            }
            if (true)
            {
                // VMP by hand:
                // p(x,c) = fc(c) (f1(x))^c (f2(x))^(1-c)
                // q(x) = f1(x)^q(c=T) * f2(x)^q(c=F)
                // q(c) = fc(c)*Bernoulli(c;q1/(q1+q2))
                // where q1 = p1^q(x=T)*(1-p1)^q(x=F)
                //       q2 = p2^q(x=T)*(1-p2)^q(x=F)
                Bernoulli f1 = new Bernoulli(p1);
                Bernoulli f2 = new Bernoulli(p2);
                Bernoulli fc = new Bernoulli(cPrior);
                Bernoulli qx = new Bernoulli();
                Bernoulli qc = fc;
                for (int iter = 0; iter < 100; iter++)
                {
                    qx = (f1 ^ qc.GetProbTrue())*(f2 ^ qc.GetProbFalse());
                    double q1 = qx.GetAverageLog(f1);
                    double q2 = qx.GetAverageLog(f2);
                    qc = fc*Bernoulli.FromLogOdds(q1 - q2);
                }
                cExpected = qc.GetProbTrue();
                xExpected = qx.GetProbTrue();
                double evGate = qc.GetProbTrue()*qx.GetAverageLog(f1) + qc.GetProbFalse()*qx.GetAverageLog(f2);
                evExpected = qc.GetAverageLog(fc) + evGate - qc.GetAverageLog(qc) - qx.GetAverageLog(qx);
                // xExpected = 0.153240878485712
                // evExpected = -0.00987955117486039
                vmp.UseGateExitRandom = true;
                engine.Algorithm = vmp;
                cActual = engine.Infer<Bernoulli>(c).GetProbTrue();
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("c = {0} (should be {1})", cActual, cExpected);
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(cActual, cExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
        }

        [Fact]
        public void MixtureOfManyBernoullis()
        {
            double p1 = 0.1, p2 = 0.2;
            double cPrior = 0.4;
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            VariableArray<double> p = Variable.Constant(new double[] {p1, p2});
            Variable<int> i = Variable.Discrete(p.Range, new double[] {cPrior, 1 - cPrior}).Named("i");
            Variable<bool> x = Variable.New<bool>().Named("x");
            using (Variable.Switch(i))
            {
                x.SetTo(Variable.Bernoulli(p[i]));
            }
            block.CloseBlock();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            InferenceEngine engine = new InferenceEngine(vmp);
            double xActual, xExpected;
            double evActual, evExpected;
            if (true)
            {
                xExpected = p1*cPrior + p2*(1 - cPrior);
                evExpected = 0;
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-5) < 1e-10);
            }
            if (true)
            {
                xExpected = 0.153240878485712;
                evExpected = -0.00987955117486039;
                vmp.UseGateExitRandom = true;
                engine.Algorithm = vmp;
                xActual = engine.Infer<Bernoulli>(x).GetProbTrue();
                evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("x = {0} (should be {1})", xActual, xExpected);
                Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
                Assert.True(MMath.AbsDiff(xActual, xExpected, 1e-8) < 1e-10);
                Assert.True(MMath.AbsDiff(evActual, evExpected, 1e-8) < 1e-10);
            }
        }


#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        [Fact]
        public void MixtureOf2GaussiansUsingDiscrete()
        {
            double data = 27;
            double psi = 1;
            //ie.ShowFactorGraph = true;
            // ie.ShowMsl = true;

            int K = 2;

            Vector probs = Vector.Zero(K);
            probs.SetAllElementsTo(1.0/(double) K);
            Variable<double> m0 = Variable.GaussianFromMeanAndVariance(0, 10 + psi).Named("m0");
            Variable<double> m1 = Variable.GaussianFromMeanAndVariance(1, 10 + psi).Named("m1");
#if false
             Variable<int> c = Variable.Discrete(probs).Named("c");

             using (Variable.Case(c, 0))
             {
                            Variable.ConstrainEqualRandom(m0, new Gaussian(data,1));
               //  Variable.ConstrainEqual(data, Variable.GaussianFromMeanAndVariance(m0, 1));
             }
             using (Variable.Case(c, 1))
             {
                            Variable.ConstrainEqualRandom(m1, new Gaussian(data,1));
               //  Variable.ConstrainEqual(data, Variable.GaussianFromMeanAndVariance(m1, 1));
             }
#else
            Variable<bool> c = Variable.Bernoulli(.5).Named("c");
            using (Variable.If(c))
            {
                //Variable.ConstrainEqualRandom(m0, new Gaussian(data,1));
                Variable.ConstrainEqual(data, Variable.GaussianFromMeanAndVariance(m0, 1));
            }
            using (Variable.IfNot(c))
            {
                //Variable.ConstrainEqualRandom(m1, new Gaussian(data,1));
                Variable.ConstrainEqual(data, Variable.GaussianFromMeanAndVariance(m1, 1));
            }
#endif
            InferenceEngine ie;
            if (false)
            {
                Console.WriteLine("EP:");
                ie = new InferenceEngine();
                Console.WriteLine(ie.Infer(c));
                Console.WriteLine(ie.Infer<Gaussian>(m0));
                Console.WriteLine(ie.Infer<Gaussian>(m1));
                //Assert.True(ie.Infer<Gaussian>(m0).MaxDiff(new Gaussian(2.46821689630930, 64.99070368978387)) < 1e-5);
                //Assert.True(ie.Infer<Gaussian>(m1).MaxDiff(new Gaussian(22.45653187762808, 52.92015271444372)) < 1e-5);
                //evidence : -30.91515015941819
            }
            Console.WriteLine("VMP:");
            ie = new InferenceEngine(new VariationalMessagePassing());
            ie.NumberOfIterations = 60;
            Console.WriteLine(ie.Infer(c));
            Console.WriteLine(ie.Infer<Gaussian>(m0));
            Console.WriteLine(ie.Infer<Gaussian>(m1));
            //Assert.True(ie.Infer<Gaussian>(m0).MaxDiff(new Gaussian(0.29318854886476, 10.88055281342547)) < 1e-5);
            //Assert.True(ie.Infer<Gaussian>(m1).MaxDiff(new Gaussian(24.83134937151420, 0.91750603512861)) < 1e-5);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        [Fact]
        public void Mixture1a()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<double> x = Variable.Constant(7.0).Named("data");
            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1}).Named("D");

            Variable<double> mean1 = Variable.Constant(5.5).Named("mean1");
            Variable.ConstrainEqualRandom(mean1, new Gaussian(10, 100));
            Variable<double> mean2 = Variable.Constant(8.0).Named("mean2");
            Variable.ConstrainEqualRandom(mean2, new Gaussian(10, 100));


            Variable<int> c = Variable.Discrete(D).Named("c");
            using (Variable.Case(c, 0))
            {
                // These two lines are equivalent.
                Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndPrecision(mean1, 1.0));
                //Variable.ConstrainEqualRandom(x, new Gaussian(mean1.Value, 1.0));
            }
            using (Variable.Case(c, 1))
            {
                // These two lines are equivalent.
                Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndPrecision(mean2, 1.0));
                //Variable.ConstrainEqualRandom(x, new Gaussian(mean2.Value, 1.0));
            }
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(D));
            double[] DVibesResult = new double[] {-1.15070203880043, -0.67501576717763};
            VmpTests.TestDirichletMoments(ie, D, DVibesResult);
            double[] cVibesResult = new double[] {0.24961241199438, 0.75038758800562};
            VmpTests.TestDiscrete(ie, c, cVibesResult);


            Dirichlet dPost = ie.Infer<Dirichlet>(D);
            Discrete cPost = ie.Infer<Discrete>(c);
            double evMean0 = new Gaussian(10, 100).GetLogProb(mean1.ObservedValue);
            double evMean1 = new Gaussian(10, 100).GetLogProb(mean2.ObservedValue);
            double sumCond0 = new Gaussian(mean1.ObservedValue, 1).GetLogProb(x.ObservedValue);
            double sumCond1 = new Gaussian(mean2.ObservedValue, 1).GetLogProb(x.ObservedValue);
            double evCase = cPost[0]*sumCond0 + cPost[1]*sumCond1;
            double evDPrior = dPost.GetAverageLog(new Dirichlet(1.0, 1.0));
            double evcD = cPost[0]*dPost.GetMeanLogAt(0) + cPost[1]*dPost.GetMeanLogAt(1);
            double evidenceExpected = evMean0 + evMean1 + evCase + evDPrior + evcD - cPost.GetAverageLog(cPost) - dPost.GetAverageLog(dPost);
            Console.WriteLine("evidence should be {0}", evidenceExpected);
            VmpTests.TestEvidence(ie, evidence, evidenceExpected);
            // Vibes bound = -8.4529705
        }


        //need to check about vibes 
        [Fact]
        public void Mixture1a2()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<double> x = Variable.Constant(7.0).Named("x");
            Variable<double> mean1 = Variable.Constant(5.5).Named("mean1");
            Variable.ConstrainEqualRandom(mean1, new Gaussian(10, 100));
            Variable<double> mean2 = Variable.Constant(8.0).Named("mean2");
            Variable.ConstrainEqualRandom(mean2, new Gaussian(10, 100));

            Variable<Vector> D = Variable.Constant(Vector.FromArray(new double[] {0.5, 0.5})).Named("D");
            Dirichlet dPrior = new Dirichlet(new double[] {1, 1});
            Variable.ConstrainEqualRandom(D, dPrior);
            Variable<int> c = Variable.Discrete(D).Named("c");
            using (Variable.Case(c, 0))
            {
                // These two lines are not equivalent wrt evidence.
                //Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndPrecision(mean1, 1.0));
                Variable.ConstrainEqualRandom(x, new Gaussian(mean1.ObservedValue, 1.0));
            }
            using (Variable.Case(c, 1))
            {
                // These two lines are not equivalent wrt evidence.
                //Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndPrecision(mean2, 1.0));
                Variable.ConstrainEqualRandom(x, new Gaussian(mean2.ObservedValue, 1.0));
            }

            block.CloseBlock();


            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());

            double[] cVibesResult = new double[] {0.34864513533395, 0.65135486466605};
            VmpTests.TestDiscrete(ie, c, cVibesResult);

            Discrete cPost = ie.Infer<Discrete>(c);
            double evMean0 = new Gaussian(10, 100).GetLogProb(5.5);
            double evMean1 = new Gaussian(10, 100).GetLogProb(8);
            double sumCond0 = new Gaussian(5.5, 1).GetLogProb(x.ObservedValue);
            double sumCond1 = new Gaussian(8, 1).GetLogProb(x.ObservedValue);
            double evCase = cPost[0]*sumCond0 + cPost[1]*sumCond1;
            double evcPrior = -MMath.Ln2;
            double evdPrior = dPrior.GetLogProb(D.ObservedValue);
            double evidenceExpected = evMean0 + evMean1 + evCase + evcPrior + evdPrior - cPost.GetAverageLog(cPost);
            Console.WriteLine("evidence should be {0}", evidenceExpected);
            VmpTests.TestEvidence(ie, evidence, evidenceExpected);
        }

        [Fact]
        public void Mixture1InferMeans()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<double> x = Variable.Constant(7.0).Named("data");
            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1}).Named("D");


            Variable<double> mprior1 = Variable.Constant(6.85).Named("mprior1");
            Variable.ConstrainEqualRandom(mprior1, new Gaussian(3, .5));
            Variable<double> mprior2 = Variable.Constant(7.5).Named("mprior2");
            Variable.ConstrainEqualRandom(mprior2, new Gaussian(3, .5));

            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(mprior1, 10).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(mprior2, 10).Named("mean2");

            Variable<int> c = Variable.Discrete(D).Named("c");
            using (Variable.Case(c, 0))
            {
                Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndPrecision(mean1, 10.0));
            }
            using (Variable.Case(c, 1))
            {
                Variable.ConstrainEqual(x, Variable.GaussianFromMeanAndPrecision(mean2, 10.0));
            }
            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(D));
            double[] DVibesResult = new double[] {-0.59815485386865, -1.28362541751857};
            VmpTests.TestDirichletMoments(ie, D, DVibesResult);

            double[] cVibesResult = new double[] {0.85482703617024, 0.14517296382976};
            VmpTests.TestDiscrete(ie, c, cVibesResult);

            double[][] mVibesResult = new double[2][];
            mVibesResult[0] = new double[] {6.91912992582332, 47.92827231320828};
            mVibesResult[1] = new double[] {7.43661526755562, 55.39056969115254};
            VmpTests.TestGaussianMoments(ie, mean1, mVibesResult[0][0], mVibesResult[0][1]);
            VmpTests.TestGaussianMoments(ie, mean2, mVibesResult[1][0], mVibesResult[1][1]);

            VmpTests.TestEvidence(ie, evidence, -36.972706);
        }


        [Fact]
        public void Mixture1InferMeansPartiallyObs()
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            int T = 3;
            double[] xobs = new double[] {6.5, 9, 2};
            Variable<double>[] x = new Variable<double>[T];
            for (int t = 0; t < T; t++)
            {
                x[t] = Variable.Constant(xobs[t]).Named("x" + t);
            }

            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1}).Named("D");

            Variable<double> mprior1 = Variable.Constant(6.0).Named("mprior1");
            Variable.ConstrainEqualRandom(mprior1, new Gaussian(3, .5));
            Variable<double> mprior2 = Variable.Constant(7.0).Named("mprior2");
            Variable.ConstrainEqualRandom(mprior2, new Gaussian(3, .5));

            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(mprior1, 100).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(mprior2, 100).Named("mean2");

            Variable<int>[] c = new Variable<int>[T];

            for (int t = 0; t < T; t++)
            {
                c[t] = Variable.Discrete(D).Named("c" + t);
                using (Variable.Case(c[t], 0))
                {
                    Variable.ConstrainEqual(x[t], Variable.GaussianFromMeanAndPrecision(mean1, 10.0));
                }
                using (Variable.Case(c[t], 1))
                {
                    Variable.ConstrainEqual(x[t], Variable.GaussianFromMeanAndPrecision(mean2, 10.0));
                }
            }
            Variable.ConstrainEqual(c[1], 1);
            Variable.ConstrainEqual(c[2], 0);

            block.CloseBlock();
            InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
            Console.WriteLine(ie.Infer(D));
            double[] DVibesResult = new double[] {-1.01215180188572, -0.62950363620104};
            VmpTests.TestDirichletMoments(ie, D, DVibesResult);

            double[] cVibesResult = new double[] {0.11428070563954, 0.88571929436046};
            VmpTests.TestDiscrete(ie, c[0], cVibesResult);

            double[][] mVibesResult = new double[2][];
            mVibesResult[0] = new double[] {5.64524378476754, 31.87777482330109};
            mVibesResult[1] = new double[] {7.13100940918764, 50.85970865271186};
            VmpTests.TestGaussianMoments(ie, mean1, mVibesResult[0][0], mVibesResult[0][1]);
            VmpTests.TestGaussianMoments(ie, mean2, mVibesResult[1][0], mVibesResult[1][1]);

            VmpTests.TestEvidence(ie, evidence, -121.00304);
        }


        internal void Mixture1InferMeansPartiallyObsGetItem()
        {
            int T = 3;
            double[] xobs = new double[] {6.5, 9, 2};

            Range item = new Range(T).Named("item");
            VariableArray<double> x = Variable.Constant<double>(xobs, item).Named("x");
            Variable<Vector> D = Variable.Dirichlet(new double[] {1, 1}).Named("D");

            Variable<double> mprior1 = Variable.Constant(6.0).Named("mprior1");
            Variable.ConstrainEqualRandom(mprior1, new Gaussian(3, .5));
            Variable<double> mprior2 = Variable.Constant(7.0).Named("mprior2");
            Variable.ConstrainEqualRandom(mprior2, new Gaussian(3, .5));

            Variable<double> mean1 = Variable.GaussianFromMeanAndPrecision(mprior1, 100).Named("mean1");
            Variable<double> mean2 = Variable.GaussianFromMeanAndPrecision(mprior2, 100).Named("mean2");

            VariableArray<int> c = Variable.Array<int>(item);
            c[item] = Variable.Discrete(D).Named("c").ForEach(item);

            using (Variable.ForEach(item))
            {
                using (Variable.Case(c[item], 0))
                {
                    Variable.ConstrainEqual(x[item], Variable.GaussianFromMeanAndPrecision(mean1, 10.0));
                }
                using (Variable.Case(c[item], 1))
                {
                    Variable.ConstrainEqual(x[item], Variable.GaussianFromMeanAndPrecision(mean2, 10.0));
                }
            }

            /*     Given<int[]>  index = Variable.Given<int[]>().Named("index");
                     Range itemObs = new Range("itemObs", 2);
                     VariableArray<int> h = Variable.GetItems(c, index, itemObs).Named("h");
                     h.Attrib(new MarginalPrototype(new Discrete(2)));
           
                     Variable.ConstrainEqual(h[itemObs], Variable.Constant(new int[] { 1, 0 }, itemObs)[itemObs]);
                     index.Value = new int[] { 1, 2 };
        */

            Variable<int> index = Variable.New<int>().Named("index");
            Variable<int> h = Variable<int>.Factor(Collection.GetItem, c, index).Named("h").Attrib(new MarginalPrototype(Discrete.Uniform(2)));
            Variable.ConstrainEqual(h, 1);
            index.ObservedValue = 1;

            Variable<int> index2 = Variable.New<int>().Named("index2");
            Variable<int> h2 = Variable<int>.Factor(Collection.GetItem, c, index2).Named("h2").Attrib(new MarginalPrototype(Discrete.Uniform(2)));
            Variable.ConstrainEqual(h2, 0);
            index2.ObservedValue = 2;


            InferenceEngine ie = new InferenceEngine();
            Console.WriteLine(ie.Infer(D));
            Console.WriteLine(ie.Infer(c));
            Console.WriteLine(ie.Infer(mean1));
            Console.WriteLine(ie.Infer(mean2));
            Console.ReadKey();
        }

        private static void Clutter2(Gaussian meanPrior, Gaussian noiseDist, double x1, double x2)
        {
            double prec = 1;
            double mixWeight = 0.5;
            bool evidence = Factor.Random(new Bernoulli(0.5));
            if (evidence)
            {
                double mean = Factor.Random(meanPrior);
                Attrib.InitialiseTo(mean, new Gaussian(2, 1));
                bool b1 = Factor.Bernoulli(mixWeight);
                //double x1;
                if (b1)
                {
                    x1 = Factor.Gaussian(mean, prec);
                }
                else
                {
                    x1 = Factor.Random(noiseDist);
                }
                //Constrain.Equal(x1, data1);
                bool b2 = Factor.Bernoulli(mixWeight);
                //double x2;
                if (b2)
                {
                    x2 = Factor.Gaussian(mean, prec);
                }
                else
                {
                    x2 = Factor.Random(noiseDist);
                }
                //Constrain.Equal(x2, data2);
                InferNet.Infer(mean, nameof(mean));
                InferNet.Infer(b1, nameof(b1));
                InferNet.Infer(b2, nameof(b2));
            }
            InferNet.Infer(evidence, nameof(evidence));
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ClutterVmpTest()
        {
            InferenceEngine engine = new InferenceEngine();
            VariationalMessagePassing vmp = new VariationalMessagePassing();
            vmp.UseGateExitRandom = true;
            engine.Algorithm = vmp;
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(Clutter2, new Gaussian(0, 100), new Gaussian(0, 10), 0.1, 2.3);
            ca.Execute(20);

            Gaussian meanExpected = new Gaussian(1.31257, 0.892128);
            Gaussian meanActual = ca.Marginal<Gaussian>("mean");
            Console.WriteLine("mean={0} (expected {1})", meanActual, meanExpected);
            Console.WriteLine("b1=" + ca.Marginal("b1"));
            Console.WriteLine("b2=" + ca.Marginal("b2"));

            double evidenceExpected = -6.0229;
            double evidenceActual = ca.Marginal<Bernoulli>("evidence").LogOdds;
            Console.WriteLine("evidence={0} (expected {1})", evidenceActual, evidenceExpected);
            Assert.True(MMath.AbsDiff(evidenceActual, evidenceExpected, 1e-4) < 1e-4);

            Assert.True(meanActual.MaxDiff(meanExpected) < 1e-4);
        }

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void MixtureOfTwoGaussiansTest()
        {
            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            engine.Algorithm = new VariationalMessagePassing();
            var ca = engine.Compiler.Compile(MixtureOfTwoGaussiansModel, 0.1, 9.3);
            ca.Execute(20);
            Console.WriteLine("mean1=" + ca.Marginal("mean1"));
            Console.WriteLine("mean2=" + ca.Marginal("mean2"));
        }


        private void MixtureOfTwoGaussiansModel(double x1, double x2)
        {
            double mean1 = Factor.Gaussian(0.1, 0.0001);
            double mean2 = Factor.Gaussian(10.2, 0.0001);
            double prec = 1;
            double mixWeight = 0.5;
            bool b1 = Factor.Bernoulli(mixWeight);
            if (b1)
            {
                x1 = Factor.Gaussian(mean1, prec);
            }
            else
            {
                x1 = Factor.Gaussian(mean2, prec);
            }
            bool b2 = Factor.Bernoulli(mixWeight);
            if (b2)
            {
                x2 = Factor.Gaussian(mean1, prec);
            }
            else
            {
                x2 = Factor.Gaussian(mean2, prec);
            }
            InferNet.Infer(mean1, nameof(mean1));
            InferNet.Infer(mean2, nameof(mean2));
            InferNet.Infer(b1, nameof(b1));
            InferNet.Infer(b2, nameof(b2));
        }
    }
}