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
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    public class StudentIsPositiveTests
    {
        /// <summary>
        /// Test modified EP updates
        /// </summary>
        internal void StudentIsPositiveTest()
        {
            double shape = 1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);
            // mean=-1 causes improper messages
            double mean = -1;
            double evExpected;
            Gaussian xExpected = StudentIsPositiveExact(mean, precPrior, out evExpected);

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> prec = Variable.Random(precPrior).Named("prec");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, prec).Named("x");
            Variable.ConstrainPositive(x);
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            x.AddAttribute(new TraceMessages());
            //x.InitialiseTo(Gaussian.FromMeanAndVariance(-3.719, 4.836));
            //GaussianOp.ForceProper = false;
            //GaussianOp.modified = true;
            //engine.Compiler.GivePriorityTo(typeof(GaussianOp_Laplace));
            //engine.Compiler.GivePriorityTo(typeof(GaussianOp_Slow));
            GaussianOp_Laplace.modified = true;
            GaussianOp_Laplace.modified2 = true;
            Console.WriteLine("x = {0} should be {1}", engine.Infer(x), xExpected);
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
        }

        /// <summary>
        /// Test a difficult case
        /// </summary>
        internal void StudentIsPositiveTest5()
        {
            // depending on the exact setting of priors, the messages will alternate between proper and improper
            Gamma precPrior = new Gamma(5, 0.2);
            // mean=-1 causes improper messages
            var mean = Variable.Random(new Gaussian(-0.9, 0.25)).Named("mean");
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> prec = Variable.Random(precPrior).Named("prec");
            Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, prec).Named("x");
            Variable<bool> y = Variable.IsPositive(x);
            Variable.ConstrainEqualRandom(y, new Bernoulli(0.8889));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            x.AddAttribute(new TraceMessages());
            Console.WriteLine(engine.Infer(x));
            //Console.WriteLine("x = {0} should be {1}", engine.Infer(x), xExpected);
            double evActual = System.Math.Exp(engine.Infer<Bernoulli>(evidence).LogOdds);
            //Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
        }

        internal void StudentIsPositiveTest4()
        {
            double shape = 1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);
            // mean=-1 causes improper messages
            double mean = -1;
            Gaussian meanPrior = Gaussian.PointMass(mean);
            double evExpected;
            Gaussian xExpected = StudentIsPositiveExact(mean, precPrior, out evExpected);

            GaussianOp.ForceProper = false;
            GaussianOp_Laplace.modified = true;
            GaussianOp_Laplace.modified2 = true;
            Gaussian xF = Gaussian.Uniform();
            Gaussian xB = Gaussian.Uniform();
            Gamma q = GaussianOp_Laplace.QInit();
            double r0 = 0.38;
            r0 = 0.1;
            for (int iter = 0; iter < 20; iter++)
            {
                q = GaussianOp_Laplace.Q(xB, meanPrior, precPrior, q);
                //xF = GaussianOp_Laplace.SampleAverageConditional(xB, meanPrior, precPrior, q);
                xF = Gaussian.FromMeanAndPrecision(mean, r0);
                xB = IsPositiveOp.XAverageConditional(true, xF);
                Console.WriteLine("xF = {0} xB = {1}", xF, xB);
            }
            Console.WriteLine("x = {0} should be {1}", xF * xB, xExpected);

            double[] precs = EpTests.linspace(1e-3, 5, 100);
            double[] evTrue = new double[precs.Length];
            double[] evApprox = new double[precs.Length];
            double[] evApprox2 = new double[precs.Length];
            //r0 = q.GetMean();
            double sum = 0, sum2 = 0;
            for (int i = 0; i < precs.Length; i++)
            {
                double r = precs[i];
                Gaussian xFt = Gaussian.FromMeanAndPrecision(mean, r);
                evTrue[i] = IsPositiveOp.LogAverageFactor(true, xFt) + precPrior.GetLogProb(r);
                evApprox[i] = IsPositiveOp.LogAverageFactor(true, xF) + precPrior.GetLogProb(r) + xB.GetLogAverageOf(xFt) - xB.GetLogAverageOf(xF);
                evApprox2[i] = IsPositiveOp.LogAverageFactor(true, xF) + precPrior.GetLogProb(r0) + q.GetLogProb(r) - q.GetLogProb(r0);
                sum += System.Math.Exp(evApprox[i]);
                sum2 += System.Math.Exp(evApprox2[i]);
            }
            Console.WriteLine("r0 = {0}: {1} {2} {3}", r0, sum, sum2, q.GetVariance() + System.Math.Pow(r0 - q.GetMean(), 2));
            //TODO: change path for cross platform using
            using (var writer = new MatlabWriter(@"..\..\..\Tests\student.mat"))
            {
                writer.Write("z", evTrue);
                writer.Write("z2", evApprox);
                writer.Write("z3", evApprox2);
                writer.Write("precs", precs);
            }
        }

        private Gaussian StudentIsPositiveExact(double mean, Gamma precPrior, out double evidence)
        {
            // importance sampling for true answer
            GaussianEstimator est = new GaussianEstimator();
            int nSamples = 1000000;
            evidence = 0;
            for (int iter = 0; iter < nSamples; iter++)
            {
                double precSample = precPrior.Sample();
                Gaussian xPrior = Gaussian.FromMeanAndPrecision(mean, precSample);
                double logWeight = IsPositiveOp.LogAverageFactor(true, xPrior);
                evidence += System.Math.Exp(logWeight);
                double xSample = xPrior.Sample();
                if (xSample > 0)
                    est.Add(xSample);
            }
            evidence /= nSamples;
            return est.GetDistribution(new Gaussian());
        }

        internal void StudentIsPositiveTest2()
        {
            GaussianOp.ForceProper = false;
            double shape = 1;
            double mean = -1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);
            Gaussian meanPrior = Gaussian.PointMass(mean);
            double evExpected;
            Gaussian xExpected = StudentIsPositiveExact(mean, precPrior, out evExpected);

            Gaussian xF2 = Gaussian.FromMeanAndVariance(-1, 1);
            // the energy has a stationary point here (min in both dimensions), even though xF0 is improper
            Gaussian xB0 = new Gaussian(2, 1);
            xF2 = Gaussian.FromMeanAndVariance(-4.552, 6.484);
            //xB0 = new Gaussian(1.832, 0.9502);
            //xB0 = new Gaussian(1.792, 1.558);
            //xB0 = new Gaussian(1.71, 1.558);
            //xB0 = new Gaussian(1.792, 1.5);
            Gaussian xF0 = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
            //Console.WriteLine("xB0 = {0} xF0 = {1}", xB0, xF0);
            //Console.WriteLine(xF0*xB0);
            //Console.WriteLine(xF2*xB0);

            xF2 = new Gaussian(0.8651, 1.173);
            xB0 = new Gaussian(-4, 2);
            xB0 = new Gaussian(7, 7);
            if (false)
            {
                xF2 = new Gaussian(mean, 1);
                double[] xs = EpTests.linspace(0, 100, 1000);
                double[] logTrue = Util.ArrayInit(xs.Length, i => GaussianOp.LogAverageFactor(xs[i], mean, precPrior));
                Normalize(logTrue);
                xF2 = FindxF4(xs, logTrue, xF2);
                xF2 = Gaussian.FromNatural(-0.85, 0);
                xB0 = IsPositiveOp.XAverageConditional(true, xF2);
                Console.WriteLine("xF = {0} xB = {1}", xF2, xB0);
                Console.WriteLine("x = {0} should be {1}", xF2 * xB0, xExpected);
                Console.WriteLine("proj[T*xB] = {0}", GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior) * xB0);
                double ev = System.Math.Exp(IsPositiveOp.LogAverageFactor(true, xF2) + GaussianOp_Slow.LogAverageFactor(xB0, meanPrior, precPrior) - xF2.GetLogAverageOf(xB0));
                Console.WriteLine("evidence = {0} should be {1}", ev, evExpected);
                return;
            }
            if (false)
            {
                xF2 = new Gaussian(mean, 1);
                xF2 = FindxF3(xExpected, evExpected, meanPrior, precPrior, xF2);
                xB0 = IsPositiveOp.XAverageConditional(true, xF2);
                Console.WriteLine("xF = {0} xB = {1}", xF2, xB0);
                Console.WriteLine("x = {0} should be {1}", xF2 * xB0, xExpected);
                //double ev = Math.Exp(IsPositiveOp.LogAverageFactor(true, xF2) + GaussianOp.LogAverageFactor_slow(xB0, meanPrior, precPrior) - xF2.GetLogAverageOf(xB0));
                //Console.WriteLine("evidence = {0} should be {1}", ev, evExpected);
                return;
            }
            if (false)
            {
                xF2 = new Gaussian(-2, 10);
                xF2 = FindxF2(meanPrior, precPrior, xF2);
                xB0 = IsPositiveOp.XAverageConditional(true, xF2);
                xF0 = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
                Console.WriteLine("xB = {0}", xB0);
                Console.WriteLine("xF = {0} should be {1}", xF0, xF2);
                return;
            }
            if (false)
            {
                xF2 = new Gaussian(-3998, 4000);
                xF2 = new Gaussian(0.8651, 1.173);
                xB0 = new Gaussian(-4, 2);
                xB0 = new Gaussian(2000, 1e-5);
                xB0 = FindxB(xB0, meanPrior, precPrior, xF2);
                xF0 = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
                Console.WriteLine("xB = {0}", xB0);
                Console.WriteLine("xF = {0} should be {1}", xF0, xF2);
                return;
            }
            if (false)
            {
                //xF2 = new Gaussian(-7, 10);
                //xF2 = new Gaussian(-50, 52);
                xB0 = new Gaussian(-1.966, 5.506e-08);
                //xF2 = new Gaussian(-3998, 4000);
                xF0 = FindxF(xB0, meanPrior, precPrior, xF2);
                Gaussian xB2 = IsPositiveOp.XAverageConditional(true, xF0);
                Console.WriteLine("xF = {0}", xF0);
                Console.WriteLine("xB = {0} should be {1}", xB2, xB0);
                return;
            }
            if (true)
            {
                xF0 = new Gaussian(-3.397e+08, 5.64e+08);
                xF0 = new Gaussian(-2.373e+04, 2.8e+04);
                xB0 = new Gaussian(2.359, 1.392);
                xF0 = Gaussian.FromNatural(-0.84, 0);
                //xF0 = Gaussian.FromNatural(-0.7, 0);
                for (int iter = 0; iter < 10; iter++)
                {
                    xB0 = FindxB(xB0, meanPrior, precPrior, xF0);
                    Gaussian xFt = GaussianOp_Slow.SampleAverageConditional(xB0, meanPrior, precPrior);
                    Console.WriteLine("xB = {0}", xB0);
                    Console.WriteLine("xF = {0} should be {1}", xFt, xF0);
                    xF0 = FindxF0(xB0, meanPrior, precPrior, xF0);
                    Gaussian xBt = IsPositiveOp.XAverageConditional(true, xF0);
                    Console.WriteLine("xF = {0}", xF0);
                    Console.WriteLine("xB = {0} should be {1}", xBt, xB0);
                }
                Console.WriteLine("x = {0} should be {1}", xF0 * xB0, xExpected);
                double ev = System.Math.Exp(IsPositiveOp.LogAverageFactor(true, xF0) + GaussianOp_Slow.LogAverageFactor(xB0, meanPrior, precPrior) - xF0.GetLogAverageOf(xB0));
                Console.WriteLine("evidence = {0} should be {1}", ev, evExpected);
                return;
            }

            //var precs = EpTests.linspace(1e-6, 1e-5, 200);
            var precs = EpTests.linspace(xB0.Precision / 11, xB0.Precision, 100);
            //var precs = EpTests.linspace(xF0.Precision/20, xF0.Precision/3, 100);
            precs = EpTests.linspace(1e-9, 1e-5, 100);
            //precs = new double[] { xB0.Precision };
            var ms = EpTests.linspace(xB0.GetMean() - 1, xB0.GetMean() + 1, 100);
            //var ms = EpTests.linspace(xF0.GetMean()-1, xF0.GetMean()+1, 100);
            //precs = EpTests.linspace(1.0/10, 1.0/8, 200);
            ms = EpTests.linspace(2000, 4000, 100);
            //ms = new double[] { xB0.GetMean() };
            Matrix result = new Matrix(precs.Length, ms.Length);
            Matrix result2 = new Matrix(precs.Length, ms.Length);
            //ms = new double[] { 0.7 };
            for (int j = 0; j < ms.Length; j++)
            {
                double maxZ = double.NegativeInfinity;
                double minZ = double.PositiveInfinity;
                Gaussian maxxF = Gaussian.Uniform();
                Gaussian minxF = Gaussian.Uniform();
                Gaussian maxxB = Gaussian.Uniform();
                Gaussian minxB = Gaussian.Uniform();
                Vector v = Vector.Zero(3);
                for (int i = 0; i < precs.Length; i++)
                {
                    Gaussian xF = Gaussian.FromMeanAndPrecision(ms[j], precs[i]);
                    xF = xF2;
                    Gaussian xB = IsPositiveOp.XAverageConditional(true, xF);
                    xB = Gaussian.FromMeanAndPrecision(ms[j], precs[i]);
                    //xB = xB0;
                    v[0] = IsPositiveOp.LogAverageFactor(true, xF);
                    v[1] = GaussianOp.LogAverageFactor_slow(xB, meanPrior, precPrior);
                    //v[1] = GaussianOp_Slow.LogAverageFactor(xB, meanPrior, precPrior);
                    v[2] = -xF.GetLogAverageOf(xB);
                    double logZ = v.Sum();
                    double Z = logZ;
                    if (Z > maxZ)
                    {
                        maxZ = Z;
                        maxxF = xF;
                        maxxB = xB;
                    }
                    if (Z < minZ)
                    {
                        minZ = Z;
                        minxF = xF;
                        minxB = xB;
                    }
                    result[i, j] = Z;
                    result2[i, j] = IsPositiveOp.LogAverageFactor(true, xF) + xF0.GetLogAverageOf(xB) - xF.GetLogAverageOf(xB);
                    //Gaussian xF3 = GaussianOp.SampleAverageConditional_slower(xB, meanPrior, precPrior);
                    //result[i, j] = Math.Pow(xF3.Precision - xF.Precision, 2);
                    //result2[i, j] = Math.Pow((xF2*xB).Precision - (xF*xB).Precision, 2);
                    //result2[i, j] = -xF.GetLogAverageOf(xB);
                    //Gaussian xF2 = GaussianOp.SampleAverageConditional_slow(xB, Gaussian.PointMass(0), precPrior);
                    Gaussian xMarginal = xF * xB;
                    //Console.WriteLine("xF = {0} Z = {1} x = {2}", xF, Z.ToString("g4"), xMarginal);
                }
                double delta = v[1] - v[2];
                //Console.WriteLine("xF = {0} xB = {1} maxZ = {2} x = {3}", maxxF, maxxB, maxZ.ToString("g4"), maxxF*maxxB);
                //Console.WriteLine("xF = {0} maxZ = {1} delta = {2}", maxxF, maxZ.ToString("g4"), delta.ToString("g4"));
                Console.WriteLine("xF = {0} xB = {1} minZ = {2} x = {3}", minxF, minxB, minZ.ToString("g4"), minxF * minxB);
            }
            //TODO: change path for cross platform using
            using (var writer = new MatlabWriter(@"..\..\..\Tests\student.mat"))
            {
                writer.Write("z", result);
                writer.Write("z2", result2);
                writer.Write("precs", precs);
                writer.Write("ms", ms);
            }
        }

        public static Gaussian FindxF4(double[] xs, double[] logTrue, Gaussian xF)
        {
            double[] logApprox = new double[xs.Length];
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xFt = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                for (int i = 0; i < xs.Length; i++)
                {
                    logApprox[i] = xFt.GetLogProb(xs[i]);
                }
                Normalize(logApprox);
                double sum = 0;
                for (int i = 0; i < xs.Length; i++)
                {
                    sum += System.Math.Abs(System.Math.Exp(logApprox[i]) - System.Math.Exp(logTrue[i]));
                    //sum += Math.Pow(Math.Exp(logApprox[i]) - Math.Exp(logTrue[i]), 2);
                    //sum += Math.Pow(Math.Exp(logApprox[i]/2) - Math.Exp(logTrue[i]/2), 2);
                    //sum += Math.Exp(logApprox[i])*(logApprox[i] - logTrue[i]);
                    //sum += Math.Exp(logTrue[i])*(logTrue[i] - logApprox[i]);
                }
                return sum;
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }
        private static void Normalize(double[] logProb)
        {
            double logSum = MMath.LogSumExp(logProb);
            for (int i = 0; i < logProb.Length; i++)
            {
                logProb[i] -= logSum;
            }
        }

        public static Gaussian FindxF3(Gaussian xExpected, double evExpected, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xFt = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                Gaussian xB = IsPositiveOp.XAverageConditional(true, xFt);
                Gaussian xM = xFt * xB;
                //return KlDiv(xExpected, xM);
                return KlDiv(xM, xExpected);
                //Gaussian xF2 = GaussianOp.SampleAverageConditional_slow(xB, meanPrior, precPrior);
                //Gaussian xF2 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
                //Gaussian xM2 = xF2*xB;
                //double ev1 = IsPositiveOp.LogAverageFactor(true, xFt);
                //double ev2 = GaussianOp.LogAverageFactor_slow(xB, meanPrior, precPrior) - xFt.GetLogAverageOf(xB);
                //double ev = ev1 + ev2;
                //return xExpected.MaxDiff(xM);
                //return Math.Pow(xExpected.GetMean() - xM.GetMean(), 2) + Math.Pow(ev - Math.Log(evExpected), 2);
                //return 100*Math.Pow(xM.GetMean() - xM2.GetMean(), 2) -ev;
                //return 100*Math.Pow(ev2, 2) + Math.Pow(ev - Math.Log(evExpected), 2);
                //return 100*Math.Pow(ev2, 2) + Math.Pow(xM2.GetMean() - xM.GetMean(), 2);
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static double KlDiv(Gaussian p, Gaussian q)
        {
            // E[log p] = -0.5*log(vp) - 0.5*vp/vp
            // E[log q] = -0.5*log(vq) - 0.5*((mp-mq)^2 + vp)/vq
            double delta = p.GetMean() - q.GetMean();
            return 0.5 * System.Math.Log(p.Precision / q.Precision) - 0.5 + 0.5 * (delta * delta + p.GetVariance()) * q.Precision;
        }
        public static double MeanError(Gaussian p, Gaussian q)
        {
            double delta = p.GetMean() - q.GetMean();
            return 0.5 * delta * delta;
        }

        public static Gaussian FindxF2(Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xFt = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                Gaussian xB = IsPositiveOp.XAverageConditional(true, xFt);
                Gaussian xF2 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
                return xFt.MaxDiff(xF2);
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static Gaussian FindxB(Gaussian xB, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Gaussian xB3 = IsPositiveOp.XAverageConditional(true, xF);
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xB2 = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                //Gaussian xF2 = GaussianOp.SampleAverageConditional_slow(xB2, meanPrior, precPrior);
                Gaussian xF2 = GaussianOp_Slow.SampleAverageConditional(xB2, meanPrior, precPrior);
                //Assert.True(xF2.MaxDiff(xF3) < 1e-10);
                //return Math.Pow((xF*xB2).GetMean() - (xF2*xB2).GetMean(), 2) + Math.Pow((xF*xB2).GetVariance() - (xF2*xB2).GetVariance(), 2);
                //return KlDiv(xF2*xB2, xF*xB2) + KlDiv(xF*xB3, xF*xB2);
                //return KlDiv(xF2*xB2, xF*xB2) + Math.Pow((xF*xB3).GetMean() - (xF*xB2).GetMean(),2);
                return MeanError(xF2 * xB2, xF * xB2) + KlDiv(xF * xB3, xF * xB2);
                //return xF.MaxDiff(xF2);
                //Gaussian q = new Gaussian(0, 0.1);
                //return Math.Pow((xF*q).GetMean() - (xF2*q).GetMean(), 2) + Math.Pow((xF*q).GetVariance() - (xF2*q).GetVariance(), 2);
            };

            double m = xB.GetMean();
            double p = xB.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static Gaussian FindxF(Gaussian xB, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Gaussian xF3 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
            Func<Vector, double> func = delegate (Vector x2)
            {
                Gaussian xF2 = Gaussian.FromMeanAndPrecision(x2[0], System.Math.Exp(x2[1]));
                Gaussian xB2 = IsPositiveOp.XAverageConditional(true, xF2);
                //return (xF2*xB2).MaxDiff(xF2*xB) + (xF3*xB).MaxDiff(xF2*xB);
                //return KlDiv(xF2*xB2, xF2*xB) + KlDiv(xF3*xB, xF2*xB);
                //return KlDiv(xF3*xB, xF2*xB) + Math.Pow((xF2*xB2).GetMean() - (xF2*xB).GetMean(),2);
                return KlDiv(xF2 * xB2, xF2 * xB) + MeanError(xF3 * xB, xF2 * xB);
            };

            double m = xF.GetMean();
            double p = xF.Precision;
            Vector x = Vector.FromArray(m, System.Math.Log(p));
            Minimize2(func, x);
            //MinimizePowell(func, x);
            return Gaussian.FromMeanAndPrecision(x[0], System.Math.Exp(x[1]));
        }

        public static Gaussian FindxF0(Gaussian xB, Gaussian meanPrior, Gamma precPrior, Gaussian xF)
        {
            Gaussian xF3 = GaussianOp_Slow.SampleAverageConditional(xB, meanPrior, precPrior);
            Func<double, double> func = delegate (double tau2)
            {
                Gaussian xF2 = Gaussian.FromNatural(tau2, 0);
                if (tau2 >= 0)
                    return double.PositiveInfinity;
                Gaussian xB2 = IsPositiveOp.XAverageConditional(true, xF2);
                //return (xF2*xB2).MaxDiff(xF2*xB) + (xF3*xB).MaxDiff(xF2*xB);
                //return KlDiv(xF2*xB2, xF2*xB) + KlDiv(xF3*xB, xF2*xB);
                //return KlDiv(xF3*xB, xF2*xB) + Math.Pow((xF2*xB2).GetMean() - (xF2*xB).GetMean(), 2);
                return KlDiv(xF2 * xB2, xF2 * xB) + MeanError(xF3 * xB, xF2 * xB);
            };

            double tau = xF.MeanTimesPrecision;
            double fmin;
            tau = Minimize(func, tau, out fmin);
            //MinimizePowell(func, x);
            return Gaussian.FromNatural(tau, 0);
        }

        internal static void Minimize2a(Func<Vector, double> func, Vector x)
        {
            Vector temp = Vector.Copy(x);
            Func<double, double> func1 = delegate (double x1)
            {
                temp[1] = x1;
                return func(temp);
            };
            Func<double, double> func0 = delegate (double x0)
            {
                temp[0] = x0;
                double fmin0;
                Minimize(func1, x[1], out fmin0);
                return fmin0;
            };
            double fTol = 1e-15;
            double delta = 100;
            int maxIter = 100;
            int iter;
            double fmin = func(x);
            for (iter = 0; iter < maxIter; iter++)
            {
                double oldmin = fmin;
                double newx = MinimizeBrent(func0, x[0] - 2 * delta, x[0], x[0] + 2 * delta, out fmin);
                if (fmin > oldmin)
                    throw new Exception("objective increased");
                delta = newx - x[0];
                x[0] = newx;
                temp[0] = newx;
                x[1] = Minimize(func1, x[1], out fmin);
                Console.WriteLine("x = {0} f = {1}", x, fmin);
                if (MMath.AbsDiff(fmin, oldmin, 1e-14) < fTol)
                    break;
            }
            if (iter == maxIter)
                throw new Exception("exceeded maximum number of iterations");
        }
        private static void Minimize2(Func<Vector, double> func, Vector x)
        {
            Vector temp = Vector.Copy(x);
            Func<double, double> func1 = delegate (double x1)
            {
                temp[1] = x1;
                return func(temp);
            };
            Func<double, double> func0 = delegate (double x0)
            {
                temp[0] = x0;
                double fmin0;
                Minimize(func1, x[1], out fmin0);
                return fmin0;
            };
            double xTol = 1e-10;
            double delta = 1;
            int maxIter = 100;
            int iter;
            double fmin = func(x);
            for (iter = 0; iter < maxIter; iter++)
            {
                double oldmin = fmin;
                Console.WriteLine("x={0} f={1} delta={2}", x, fmin, delta);
                if (delta < xTol)
                    break;
                bool changed = false;
                double f1 = func0(x[0] - delta);
                if (f1 < fmin)
                {
                    x[0] -= delta;
                    changed = true;
                }
                else
                {
                    double f2 = func0(x[0] + delta);
                    if (f2 < fmin)
                    {
                        x[0] += delta;
                        changed = true;
                    }
                    else
                    {
                        delta /= 2;
                    }
                }
                if (changed)
                {
                    temp[0] = x[0];
                    x[1] = Minimize(func1, x[1], out fmin);
                    delta *= 2;
                }
            }
        }

        public static double Minimize(Func<double, double> func, double x, out double fmin)
        {
            double fTol = 1e-15;
            double delta = 1;
            int maxIter = 100;
            int iter;
            fmin = func(x);
            for (iter = 0; iter < maxIter; iter++)
            {
                double oldmin = fmin;
                double newx = MinimizeBrent(func, x - 2 * delta, x, x + 2 * delta, out fmin);
                if (fmin > oldmin)
                    throw new Exception("objective increased");
                delta = newx - x;
                x = newx;
                if (MMath.AbsDiff(fmin, oldmin, 1e-14) < fTol)
                    break;
            }
            if (iter == maxIter)
                throw new Exception("exceeded maximum number of iterations");
            return x;
        }

        /* Minimize a multidimensional scalar function starting at x.
         * Modifies x to be the minimum.
         */
        internal static void MinimizePowell(Func<Vector, double> func, Vector x)
        {
            double fTol = 1e-15;
            Vector old_x = Vector.Copy(x);
            Vector ext_x = Vector.Copy(x);
            int d = x.Count;
            /* Initialize the directions to the unit vectors */
            Vector[] dirs = Util.ArrayInit(d, i => Vector.FromArray(Util.ArrayInit(d, j => (i == j) ? 1.0 : 0.0)));
            double fmin = func(x);
            int maxIter = 100;
            int iter;
            for (iter = 0; iter < maxIter; iter++)
            {
                double fx = fmin;
                int i_max = 0;
                double delta_max = 0;
                /* Minimize along each direction, remembering the direction of greatest
                 * function decrease.
                 */
                for (int i = 0; i < d; i++)
                {
                    double old_min = fmin;
                    Vector dir = dirs[i];
                    double a = MinimizeLine(func, x, dir, out fmin);
                    dir.Scale(a);
                    if (fmin > old_min)
                        throw new Exception("objective increased");
                    double delta = System.Math.Abs(old_min - fmin);
                    if (delta > delta_max)
                    {
                        delta_max = delta;
                        i_max = i;
                    }
                }
                if (MMath.AbsDiff(fx, fmin, 1e-14) < fTol)
                    break;
                /* Construct new direction from old_x to x. */
                Vector dir2 = x - old_x;
                old_x.SetTo(x);
                /* And extrapolate it. */
                ext_x.SetTo(x);
                x.SetToSum(x, dir2);
                /* Good extrapolation? */
                double fex = func(x);
                x.SetTo(ext_x);
                if (fex < fx)
                {
                    double t = fx - fmin - delta_max;
                    double delta = fx - fex;
                    t = 2 * (fx - 2 * fmin + fex) * t * t - delta_max * delta * delta;
                    if (t < 0)
                    {
                        double a = MinimizeLine(func, x, dir2, out fmin);
                        dir2.Scale(a);
                        /* Replace i_max with the new dir. */
                        dirs[i_max] = dir2;
                    }
                }
                Console.WriteLine("x = {0} f = {1}", x, fmin);
            }
            if (iter == maxIter)
                throw new Exception("exceeded maximum number of iterations");
        }

        /* Modifies x to be the minimum of f along the direction dir. */
        public static double MinimizeLine(Func<Vector, double> func, Vector x, Vector dir, out double fmin)
        {
            Vector temp = Vector.Zero(x.Count);
            Func<double, double> lineFunc = delegate (double u)
            {
                temp.SetToSum(1.0, x, u, dir);
                return func(temp);
            };
            double a = MinimizeBrent(lineFunc, -2, 0, 2, out fmin);
            if (lineFunc(a) > lineFunc(0))
                throw new Exception("objective increased");
            x.SetToSum(1.0, x, a, dir);
            return a;
        }

        /* Minimize the scalar function f in the interval [a,b] via Brent's method.
         * Requires a < b.
         * Modifies *fmin_return to be the ordinate of the minimum.
         * Returns the abscissa of the minimum.
         * Algorithm taken from Numerical Recipes and Matlab optimization toolbox.
         */
        public static double MinimizeBrent(Func<double, double> func, double min, double x, double max, out double fmin)
        {
            double tol = 1e-3;
            double d = 0, e = 0;
            double v, w;
            double fx, fv, fw;
            double u, fu;
            int iter;
            const double cgold = 0.38196601125011;
            const double zeps = 1e-10;

            //double x = min + cgold*(max-min); /* golden section to get third point */
            w = v = x;
            fx = func(x);
            fv = fx;
            fw = fx;

            int maxIter = 100;
            for (iter = 0; iter < maxIter; iter++)
            {
                bool golden_section_step = true;
                double xm = (min + max) / 2;
                double tol1 = zeps * System.Math.Abs(x) + tol / 3;
                double tol2 = 2 * tol1;
                if (System.Math.Abs(x - xm) <= (tol2 - (max - min) / 2))
                    break;

                /* Construct a trial parabolic fit */
                if (System.Math.Abs(e) > tol1)
                {
                    double r = (x - w) * (fx - fv);
                    double q = (x - v) * (fx - fw);
                    double p = (x - v) * q - (x - w) * r;
                    q = 2 * (q - r);
                    if (q > 0)
                        p = -p;
                    q = System.Math.Abs(q);
                    r = e;
                    e = d;
                    /* Is the parabola acceptable? */
                    if ((System.Math.Abs(p) < System.Math.Abs(0.5 * q * r)) && (p > q * (min - x)) && (p < q * (max - x)))
                    {
                        /* Yes, take the parabolic step */
                        d = p / q;
                        u = x + d;
                        if ((u - min < tol2) || (max - u < tol2))
                        {
                            d = tol1;
                            if (xm < x)
                                d = -d;
                        }
                        golden_section_step = false;
                    }
                }

                if (golden_section_step)
                {
                    /* Take the golden section step */
                    if (x >= xm)
                        e = min - x;
                    else
                        e = max - x;
                    d = cgold * e;
                }

                /* Evaluate f at x+d as long as fabs(d) > tol1 */
                if (System.Math.Abs(d) >= tol1)
                    u = d;
                else if (d >= 0)
                    u = tol1;
                else
                    u = -tol1;
                u += x;
                fu = func(u);

                if (fu <= fx)
                {
                    if (u >= x)
                        min = x;
                    else
                        max = x;
                    v = w;
                    w = x;
                    x = u;
                    fv = fw;
                    fw = fx;
                    fx = fu;
                }
                else
                {
                    if (u < x)
                        min = u;
                    else
                        max = u;
                    if (fu <= fw || w == x)
                    {
                        v = w;
                        w = u;
                        fv = fw;
                        fw = fu;
                    }
                    else if (fu <= fv || v == x || v == w)
                    {
                        v = u;
                        fv = fu;
                    }
                }
            }
            if (iter == maxIter)
            {
                throw new Exception("exceeded maximum number of iterations");
            }
            fmin = fx;
            return x;
        }


        // this is a very simple derivative-free optimizer
        internal static void Minimize(Func<Vector, double> func, Vector x, int maxIter = 1000, double xTol = 1e-10)
        {
            Vector delta = Vector.Constant(x.Count, 1);
            Vector temp = Vector.Zero(x.Count);
            for (int iter = 0; iter < maxIter; iter++)
            {
                bool changed = false;
                for (int i = 0; i < x.Count; i++)
                {
                    //if (i == 0) continue;
                    double f = func(x);
                    Console.WriteLine("x={0} f={1} delta={2}", x, f, delta);
                    while (delta[i] > xTol)
                    {
                        temp.SetTo(x);
                        temp[i] = x[i] - delta[i];
                        double f1 = func(temp);
                        if (f1 < f)
                        {
                            x[i] -= delta[i];
                            changed = true;
                            break;
                        }
                        temp[i] = x[i] + delta[i];
                        double f2 = func(temp);
                        if (f2 < f)
                        {
                            x[i] += delta[i];
                            changed = true;
                            break;
                        }
                        delta[i] /= 2;
                    }
                    delta[i] *= 2;
                }
                if (!changed)
                    break;
            }
        }

        internal void StudentIsPositiveTest3()
        {
            double shape = 1;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, shape);

            Gaussian meanPrior = Gaussian.PointMass(0);
            Gaussian xB = Gaussian.Uniform();
            Gaussian xF = GaussianOp.SampleAverageConditional_slow(xB, meanPrior, precPrior);
            for (int iter = 0; iter < 100; iter++)
            {
                xB = IsPositiveOp.XAverageConditional(true, xF);
                xF = GetConstrainedMessage(xB, meanPrior, precPrior, xF);
            }
            Console.WriteLine("xF = {0} x = {1}", xF, xB * xF);
        }
        private static Gaussian GetConstrainedMessage(Gaussian sample, Gaussian mean, Gamma precision, Gaussian to_sample)
        {
            for (int iter = 0; iter < 100; iter++)
            {
                Gaussian old = to_sample;
                to_sample = GetConstrainedMessage1(sample, mean, precision, to_sample);
                if (old.MaxDiff(to_sample) < 1e-10)
                    break;
            }
            return to_sample;
        }
        private static Gaussian GetConstrainedMessage1(Gaussian sample, Gaussian mean, Gamma precision, Gaussian to_sample)
        {
            Gaussian sampleMarginal = sample * to_sample;
            double m1, v1;
            to_sample.GetMeanAndVariance(out m1, out v1);
            double m, v;
            sampleMarginal.GetMeanAndVariance(out m, out v);
            double moment2 = m * m + v;
            // vq < moment2 implies 1/vq > 1/moment2
            // implies 1/v2 > 1/moment2 - to_sample.Precision
            double v2max = 1 / (1 / moment2 - to_sample.Precision);
            double v2min = 1e-2;
            double[] v2s = EpTests.linspace(v2min, v2max, 100);
            double p2min = 1 / moment2 - to_sample.Precision;
            if (p2min < 0.0)
                return to_sample;
            double p2max = sample.Precision * 10;
            double[] p2s = EpTests.linspace(p2min, p2max, 100);
            Gaussian bestResult = to_sample;
            double bestScore = double.PositiveInfinity;
            for (int i = 0; i < p2s.Length; i++)
            {
                double p2 = p2s[i];
                double vq = 1 / (to_sample.Precision + p2);
                double m2 = (System.Math.Sqrt(moment2 - vq) / vq - to_sample.MeanTimesPrecision) / p2;
                // check
                double mq = vq * (to_sample.MeanTimesPrecision + m2 * p2);
                Assert.True(MMath.AbsDiff(mq * mq + vq, moment2) < 1e-10);
                Gaussian sample2 = Gaussian.FromMeanAndPrecision(m2, p2);
                Gaussian result = GaussianOp.SampleAverageConditional_slow(sample2, mean, precision);
                double score = System.Math.Abs(result.MeanTimesPrecision);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestResult = result;
                }
            }
            return bestResult;
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
