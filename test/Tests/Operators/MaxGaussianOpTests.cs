using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class MaxGaussianOpTests
    {
        [Fact]
        public void MaxTest()
        {
            Gaussian actual, expected;
            actual = MaxGaussianOp.MaxAverageConditional(Gaussian.FromNatural(6053.7946407740192, 2593.4559834344436), Gaussian.FromNatural(-1.57090676324773, 1.3751262174888785), Gaussian.FromNatural(214384.78500926663, 96523.508973471908));
            Assert.False(actual.IsProper());
            actual = MaxGaussianOp.MaxAverageConditional(Gaussian.FromNatural(146.31976467723146, 979.371757950659), Gaussian.FromNatural(0.075442729439046508, 0.086399540048904114), Gaussian.PointMass(0));
            Assert.False(actual.IsProper());

            actual = MaxGaussianOp.MaxAverageConditional(Gaussian.Uniform(), Gaussian.PointMass(0), new Gaussian(-7.357e+09, 9.75));
            expected = Gaussian.PointMass(0);
            Assert.True(expected.MaxDiff(actual) < 1e-4);

            Gaussian max;
            max = new Gaussian(4, 5);
            actual = MaxGaussianOp.MaxAverageConditional(max, new Gaussian(0, 1), new Gaussian(2, 3));
            actual *= max;
            expected = new Gaussian(2.720481395499785, 1.781481142817509);
            //expected = MaxPosterior(max, new Gaussian(0, 1), new Gaussian(2, 3));
            Assert.True(expected.MaxDiff(actual) < 1e-4);

            max = new Gaussian();
            max.MeanTimesPrecision = 0.2;
            max.Precision = 1e-10;
            actual = MaxGaussianOp.MaxAverageConditional(max, new Gaussian(0, 1), new Gaussian(0, 1));
            actual *= max;
            expected = new Gaussian(0.702106815765215, 0.697676918460236);
            Assert.True(expected.MaxDiff(actual) < 1e-4);
        }

        [Fact]
        public void MaxTest2()
        {
            Assert.False(double.IsNaN(MaxGaussianOp.AAverageConditional(Gaussian.PointMass(0), Gaussian.FromNatural(2.9785077299969009E+293, 8.7780555996490138E-16), Gaussian.PointMass(0)).MeanTimesPrecision));

            foreach (double max in new[] { 0.0, 2.0 })
            {
                foreach (double aPrecision in new[] { 1.0/177, 8.7780555996490138E-16 })
                {
                    double oldm = double.NaN;
                    double oldv = double.NaN;
                    for (int i = 0; i < 300; i++)
                    {
                        Gaussian a = Gaussian.FromNatural(System.Math.Pow(10, i), aPrecision);
                        Gaussian to_a = MaxGaussianOp.AAverageConditional(max, a, 0);
                        Gaussian to_b = MaxGaussianOp.BAverageConditional(max, 0, a);
                        Assert.Equal(to_a, to_b);
                        if (max == 0)
                        {
                            Gaussian to_a2 = IsPositiveOp.XAverageConditional(false, a);
                            double error = System.Math.Max(MMath.AbsDiff(to_a.MeanTimesPrecision, to_a2.MeanTimesPrecision, double.Epsilon),
                                MMath.AbsDiff(to_a.Precision, to_a2.Precision, double.Epsilon));
                            Trace.WriteLine($"{i} {a} {to_a} {to_a2} {error}");
                            //Assert.True(error < 1e-15);
                        }
                        //else Trace.WriteLine($"{a} {to_a}");
                        double m, v;
                        to_a.GetMeanAndVariance(out m, out v);
                        if (!double.IsNaN(oldm))
                        {
                            Assert.True(v <= oldv);
                            double olddiff = System.Math.Abs(max - oldm);
                            double diff = System.Math.Abs(max - m);
                            Assert.True(diff <= olddiff);
                        }
                        oldm = m;
                        oldv = v;
                    }
                }
            }
        }

        [Fact]
        public void Max_MaxPointMassTest()
        {
            Gaussian a = new Gaussian(1, 2);
            Gaussian b = new Gaussian(4, 5);
            Max_MaxPointMass(a, b);
            Max_MaxPointMass(a, Gaussian.PointMass(2));
            Max_MaxPointMass(a, Gaussian.PointMass(3));
            Max_MaxPointMass(Gaussian.PointMass(2), b);
            Max_MaxPointMass(Gaussian.PointMass(3), b);
            Max_MaxPointMass(Gaussian.PointMass(3), Gaussian.PointMass(2));
            Max_MaxPointMass(Gaussian.PointMass(2), Gaussian.PointMass(3));
        }

        private void Max_MaxPointMass(Gaussian a, Gaussian b)
        {
            double point = 3;
            Gaussian toPoint = MaxGaussianOp.MaxAverageConditional(Gaussian.PointMass(point), a, b);
            //Console.WriteLine($"{point} {toPoint} {toPoint.MeanTimesPrecision} {toPoint.Precision}");
            double oldDiff = double.PositiveInfinity;
            for (int i = 5; i < 100; i++)
            {
                Gaussian max = Gaussian.FromMeanAndPrecision(point, System.Math.Pow(10, i));
                Gaussian to_max = MaxGaussianOp.MaxAverageConditional(max, a, b);
                double diff = toPoint.MaxDiff(to_max);
                //Console.WriteLine($"{max} {to_max} {to_max.MeanTimesPrecision} {to_max.Precision} {diff}");
                if (diff < 1e-14) diff = 0;
                Assert.True(diff <= oldDiff);
                oldDiff = diff;
            }
        }

        [Fact]
        public void Max_APointMassTest()
        {
            //MaxGaussianOp.ForceProper = false;
            Gaussian max = new Gaussian(4, 5);
            Gaussian b = new Gaussian(1, 2);
            Max_APointMass(new Gaussian(4, 1e-0), Gaussian.PointMass(0));
            Max_APointMass(Gaussian.PointMass(11), Gaussian.PointMass(0));
            Max_APointMass(max, b);
            Max_APointMass(max, Gaussian.PointMass(2));
            Max_APointMass(max, Gaussian.PointMass(3));
            Max_APointMass(max, Gaussian.PointMass(4));
            Max_APointMass(Gaussian.PointMass(3), b);
            Max_APointMass(Gaussian.PointMass(4), b);
            Max_APointMass(Gaussian.PointMass(3), Gaussian.PointMass(3));
        }

        private void Max_APointMass(Gaussian max, Gaussian b)
        {
            double point = 3;
            Gaussian toPoint = MaxGaussianOp.AAverageConditional(max, Gaussian.PointMass(point), b);
            //Console.WriteLine($"{point} {toPoint} {toPoint.MeanTimesPrecision:g17} {toPoint.Precision:g17}");
            Gaussian toUniform = MaxGaussianOp.AAverageConditional(max, Gaussian.Uniform(), b);
            if (max.IsPointMass && b.IsPointMass)
            {
                if (max.Point > b.Point)
                {
                    Assert.Equal(toUniform, max);
                }
                else
                {
                    Assert.Equal(toUniform, Gaussian.Uniform());
                }
            }
            double oldDiff = double.PositiveInfinity;
            for (int i = 3; i < 100; i++)
            {
                Gaussian a = Gaussian.FromMeanAndPrecision(point, System.Math.Pow(10, i));
                Gaussian to_a = MaxGaussianOp.AAverageConditional(max, a, b);
                double diff = toPoint.MaxDiff(to_a);
                //Console.WriteLine($"{i} {a} {to_a} {to_a.MeanTimesPrecision:g17} {to_a.Precision:g17} {diff:g17}");
                if (diff < 1e-14) diff = 0;
                Assert.True(diff <= oldDiff);
                oldDiff = diff;
            }
            oldDiff = double.PositiveInfinity;
            for (int i = 3; i < 100; i++)
            {
                Gaussian a = Gaussian.FromMeanAndPrecision(point, System.Math.Pow(10, -i));
                Gaussian to_a = MaxGaussianOp.AAverageConditional(max, a, b);
                double diff = toUniform.MaxDiff(to_a);
                //Console.WriteLine($"{i} {a} {to_a} {to_a.MeanTimesPrecision:g17} {to_a.Precision:g17} {diff:g17}");
                if (diff < 1e-14) diff = 0;
                Assert.True(diff <= oldDiff);
                oldDiff = diff;
            }
        }

        public static Gaussian MaxAPosterior(Gaussian max, Gaussian a, Gaussian b)
        {
            int n = 10000000;
            GaussianEstimator est = new GaussianEstimator();
            for (int i = 0; i < n; i++)
            {
                double aSample = a.Sample();
                double bSample = b.Sample();
                double logProb = max.GetLogProb(System.Math.Max(aSample, bSample));
                double weight = System.Math.Exp(logProb);
                est.Add(aSample, weight);
            }
            return est.GetDistribution(new Gaussian());
        }

        public static Gaussian MaxPosterior(Gaussian max, Gaussian a, Gaussian b)
        {
            int n = 10000000;
            GaussianEstimator est = new GaussianEstimator();
            for (int i = 0; i < n; i++)
            {
                double aSample = a.Sample();
                double bSample = b.Sample();
                double maxSample = System.Math.Max(aSample, bSample);
                double logProb = max.GetLogProb(maxSample);
                double weight = System.Math.Exp(logProb);
                est.Add(maxSample, weight);
            }
            return est.GetDistribution(new Gaussian());
        }
    }
}
