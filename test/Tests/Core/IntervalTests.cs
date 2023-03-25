namespace Microsoft.ML.Probabilistic.Tests.Core
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;
    using Xunit;
    using Assert = AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    public class IntervalTests
    {
        /*
    using FsCheck;
        [Fact]
        public void Interval_ValueEquality_Fuzzing()
        {
            Prop
               .ForAll(Arb.Generate<double>().Where(x => !double.IsNaN(x)).Three().ToArbitrary(),
                   input =>
                   {
                       var lower = Math.Min(input.Item1, input.Item2);
                       var upper = Math.Max(input.Item1, input.Item2);
                       Interval interval1 = new Interval(lower, upper);
                       Interval interval2 = new Interval(lower, upper);

                       Assert.True(interval1 == interval2, "==");
                       Assert.False(interval1 != interval2, "!=");
                       Assert.True(interval1.Equals(interval2), "Equals(Interval)");
                       Assert.True(interval1.Equals((object)interval2), "Equals(object)");
                       Assert.True(interval1.GetHashCode() == interval2.GetHashCode(), "Hashcode");

                       var lower2 = Math.Min(lower, input.Item3);
                       var upper2 = Math.Max(upper, input.Item3);
                       if (lower2 != lower || upper2 != upper)
                       {
                           Interval interval3 = new Interval(lower2, upper2);

                           Assert.False(interval1 == interval3, "== (non equal)");
                           Assert.True(interval1 != interval3, "!= (non equal)");
                           Assert.False(interval1.Equals(interval3), "Equals(Interval) (non equal)");
                           Assert.False(interval1.Equals((object)interval3), "Equals(object) (non equal)");
                       }
                   })
               .Check(500);
        }
        */

        [Fact]
        public void Interval_FindMaximumTest()
        {
            int dim = 5;
            double fTolerance = 1e-1;
            Region bounds = new Region(dim);
            bounds.Lower.SetAllElementsTo(-10);
            bounds.Upper.SetAllElementsTo(100);
            int getBoundCount = 0;
            double funcSquareMagnitude(Vector t) => 1 - t.Inner(t);
            (Interval, object) getBoundSquareMagnitude(Region region) { getBoundCount++; return (1 - SquareMagnitude(region), null); };
            (Interval, object) getBoundSquareMagnitudeUncertain(Region region) { getBoundCount++; return (1 - addNoise(SquareMagnitude(region), fTolerance / 2), null); };
            double funcLogDelta(Vector t) => t.EqualsAll(0) ? 0 : double.NegativeInfinity;
            (Interval, object) getBoundLogDelta(Region region) { getBoundCount++; return (LogDelta(region), null); };
            //void reportProgress(double value, int dummy)
            //{
            //    Trace.WriteLine($"Best value so far = {value}");
            //}
            foreach (var (func, getBound) in new (Func<Vector, double>, Func<Region, (Interval, object)>)[] {
                (funcSquareMagnitude, getBoundSquareMagnitude),
                (funcSquareMagnitude, getBoundSquareMagnitudeUncertain),
                (funcLogDelta, getBoundLogDelta),
            })
            {
                //Interval.Debug = true;
                (Vector argmax, _, _) = Interval.FindMaximum(bounds, getBound, fTolerance);
                // FindMaximum2 with fTolerance=1e-1:
                // dim=1: getBound count = 16
                // dim=2: getBound count = 79
                // dim=3: getBound count = 137
                // dim=4: getBound count = 193
                // dim=5: getBound count = 255
                // Findmaximum with tight bounds:
                // dim=1: getBound count = 81
                // dim=2: getBound count = 169
                // dim=3: getBound count = 265
                // dim=4: getBound count = 383
                // dim=5: getBound count = 994
                // dim=6: getBound count = 741
                // Findmaximum with loose bounds:
                // dim=1:
                // getBound count = 81
                // with reduceRegion:
                // getBound count = 135
                // dim=2:
                // without reduceRegion:
                // Interval.FindMaximum upperBoundCount = 263
                // getBound count = 656
                // with reduceRegion:
                // Interval.FindMaximum upperBoundCount = 41
                // getBound count = 3661
                // shaveFraction = 0.2
                // getBound count = 2373
                // shaveFraction = 0.3
                // getBound count = 1921
                // dim=4:
                // getBound count = 10526
                // with reduceRegion:
                // getBound count = 11422195
                // dim=5:
                // getBound count = 41011
                // with reduceRegion:
                // getBound count = 336511
                //Trace.WriteLine(argmax);
                bool showCount = false;
                if (showCount)
                {
                    Trace.WriteLine($"argmax = {argmax} getBound count = {getBoundCount}");
                }
                //Assert.IsTrue(argmax.MaxDiff(Vector.Zero(dim)) < xTolerance);
                double fMax = func(Vector.Zero(dim));
                double fArgmax = func(argmax);
                Assert.True(fMax - fArgmax < fTolerance);
            }

            Interval addNoise(Interval interval, double error)
                => interval + new Interval(-1, 1) * error;
        }

        public Interval LogDelta(Region a)
        {
            if (a.Contains(Vector.Zero(a.Lower.Count)))
            {
                if (a.Lower.EqualsAll(0) && a.Upper.EqualsAll(0)) return Interval.Point(0);
                else return new Interval(double.NegativeInfinity, 0);
            }
            else return Interval.Point(double.NegativeInfinity);
        }

        public Interval Inner(Region a, Region b)
        {
            Interval sum = Interval.Point(0);
            int count = a.Lower.Count;
            for (int i = 0; i < count; i++)
            {
                sum += new Interval(a.Lower[i], a.Upper[i]) * new Interval(b.Lower[i], b.Upper[i]);
            }
            return sum;
        }

        public Interval SquareMagnitude(Region a)
        {
            Interval sum = Interval.Point(0);
            int count = a.Lower.Count;
            for (int i = 0; i < count; i++)
            {
                sum += Interval.FromRegion(a, i).Square();
            }
            return sum;
        }

        [Fact]
        public void Interval_ProductTest()
        {
            Interval a = new Interval(0, double.PositiveInfinity);
            Assert.Equal(Interval.Point(0), a * 0);
            Interval product = a * a;
            Assert.True(product.Equals(a));
        }

        [Fact]
        public void Interval_RatioTest()
        {
            Assert.Equal(new Interval(2, 20), 2 / new Interval(0.1, 1));
            Assert.Equal(new Interval(double.NegativeInfinity, double.PositiveInfinity), 2 / new Interval(-1, 1));
            Assert.Equal(new Interval(double.NegativeInfinity, -2), 2 / new Interval(-1, 0));
        }

        [Fact]
        public void Interval_WeightedAverageInfinityTest3()
        {
            var weights = new Interval[11]
            {
                new Interval(0, 0),
                new Interval(0, 0.024442968509074754),
                new Interval(0, 0.39031113890339741),
                new Interval(0, 0.04888450920478498),
                new Interval(0, 0.17124394442263921),
                new Interval(0, 0.72151854952689276),
                new Interval(0, 0.013296703296703297),
                new Interval(0, 0.67072247389511974),
                new Interval(0, 0.011958932475538581),
                new Interval(0, 0.6663240796559613),
                new Interval(0, 0.64938199242720529),
            };

            var values = new double[11]
            {
                double.NegativeInfinity,
                -94.45578055768344,
                -165.78102928705744,
                -208.72818901393529,
                -163.41981127674782,
                -161.19298287663963,
                -119.60330578512396,
                -150.60167581087643,
                -248.51254973982248,
                -167.05951160045367,
                -168.26996283517371
            };

            var average = Interval.WeightedAverage(values, weights);
            Assert.True(average.LowerBound == values[8]);
            Assert.True(average.UpperBound == values[1]);
        }

        [Fact]
        public void Interval_WeightedAverageInfinityTest()
        {
            for (int n = 1; n <= 4; n++)
            {
                Interval_WeightedAverageInfinityTester(n);
            }
        }

        internal void Interval_WeightedAverageInfinityTester(int count)
        {
            double value = -100;
            Interval[] weights = Util.ArrayInit(count, i => Interval.Point(i));
            double[] values = Util.ArrayInit(count, i => (i == 0) ? double.NegativeInfinity : value);
            Interval expected = (count == 1) ? Interval.Point(double.NegativeInfinity) : Interval.Point(value);
            Interval actual = Interval.WeightedAverage(values, weights);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Interval_WeightedAverageInfinityTest2()
        {
            for (int n = 1; n <= 4; n++)
            {
                Interval_WeightedAverageInfinityTester2(n);
            }
        }

        internal void Interval_WeightedAverageInfinityTester2(int count)
        {
            Interval[] weights = Util.ArrayInit(count, i => new Interval(0, 0.1));
            double[] values = Util.ArrayInit(count, i => double.NegativeInfinity);
            Interval expected = Interval.Point(double.NegativeInfinity);
            Interval actual = Interval.WeightedAverage(values, weights);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Interval_WeightedAverageTest()
        {
            for (int n = 1; n <= 10; n++)
            {
                Interval_WeightedAverageTester(n);
            }
        }

        internal void Interval_WeightedAverageTester(int count)
        {
            Interval[] weights = Util.ArrayInit(count, i => new Interval(0.1, 1.0));
            double[] values = Util.ArrayInit(count, i => i + 1.0);
            Interval expected = WeightedAverageBrute(weights, values);
            Interval actual = Interval.WeightedAverage(values, weights);
            //Console.WriteLine($"WeightedAverage = {actual} should be {expected}");
            Assert.True(actual.Contains(expected));
        }

        public static Interval WeightedAverageBrute(IReadOnlyList<Interval> weights, IReadOnlyList<double> values)
        {
            Interval? union = null;
            for (int iter = 0; iter < 100000; iter++)
            {
                // draw a random vector of weights
                double numer = 0;
                double denom = 0;
                for (int i = 0; i < weights.Count; i++)
                {
                    double weight = weights[i].Sample();
                    denom += weight;
                    numer += weight * values[i];
                }
                double weightedAverage = numer / denom;
                if (union == null) union = Interval.Point(weightedAverage);
                else union = union.Value.Union(weightedAverage);
            }
            return union.Value;
        }

        [Fact]
        public void Interval_WeightedAverage_ElementsWithZeroWeightCanBeAnyTest()
        {
            Interval[] values = new double[] { 1, 2, 3, 4, 5 }.Select(Value).ToArray();
            Interval[] values2 = new double[] { 0, 2, 3, 4, 0 }.Select(Value).ToArray();
            Interval[] weights = new double[] { 0, 1, 2, 3, 0 }.Select(Weight).ToArray();

            Interval avg = Interval.WeightedAverage(values, weights);
            Interval avg_excl = Interval.WeightedAverage(values2, weights);

            Assert.Equal(avg, avg_excl);

            Interval Value(double v) => new Interval(v, v * 2);
            Interval Weight(double w) => new Interval(Math.Min(w, 1), w);
        }

        [Fact]
        public void Interval_WeightedAverage_ElementsWithZeroWeightCanBeExcludedTest()
        {
            Interval[] values = new double[] { 1, 2, 3, 4, 5 }.Select(Value).ToArray();
            Interval[] weights = new double[] { 0, 1, 2, 3, 0 }.Select(Weight).ToArray();

            Interval[] values_excl = new double[] { 2, 3, 4 }.Select(Value).ToArray();
            Interval[] weights_excl = new double[] { 1, 2, 3 }.Select(Weight).ToArray();

            Interval avg = Interval.WeightedAverage(values, weights);
            Interval avg_excl = Interval.WeightedAverage(values_excl, weights_excl);

            Assert.Equal(avg, avg_excl);

            Interval Value(double v) => new Interval(v, v * 2);
            Interval Weight(double w) => new Interval(Math.Min(w, 1), w);
        }

        [Fact]
        public void Interval_IntegrateTest()
        {
            Interval interval = new Interval(0, 1);
            // integral of exp over [a,b] is exp(b)-exp(a)
            double expected = Math.Exp(interval.UpperBound) - Math.Exp(interval.LowerBound);
            Interval actual = interval.Integrate(1e-3, BoundExp);
            Console.WriteLine($"actual = {actual} should be {expected}");
            Assert.True(actual.Contains(expected));
        }

        Interval BoundExp(Interval x)
        {
            return new Interval(Math.Exp(x.LowerBound), Math.Exp(x.UpperBound));
        }

        [Fact]
        public void Interval_IntegrateGammaLowerTest()
        {
            Interval interval = new Interval(0, 1);
            double shape = 3;
            double expected = MMath.GammaLower(shape, interval.UpperBound) * MMath.Gamma(shape);
            Interval actual = interval.Integrate(1e-3, x => BoundGammaDensity(x, shape));
            Console.WriteLine($"actual = {actual} should be {expected}");
            Assert.True(actual.Contains(expected));
        }

        [Fact]
        public void Interval_IntegrateGammaUpperTest()
        {
            Interval interval = new Interval(0, 1);
            double shape = -10;
            double expected = MMath.GammaUpper(shape, Math.Pow(1 - interval.LowerBound, -0.5), regularized: false);
            Interval actual = interval.Integrate(1e-3, x => BoundGammaDensityTransformed(x, shape));
            Console.WriteLine($"actual = {actual} should be {expected}");
            Assert.True(actual.Contains(expected));
        }

        Interval BoundGammaDensity(Interval x, double shape)
        {
            double shapeMinus1 = shape - 1;
            double logp(double t) => ((t > double.MaxValue) ? 0 : shapeMinus1 * Math.Log(t)) - t;
            double left = logp(x.LowerBound);
            double right = logp(x.UpperBound);
            double lowerBound = Math.Min(left, right);
            double upperBound = x.Contains(shapeMinus1) ? shapeMinus1 * (Math.Log(shapeMinus1) - 1) : Math.Max(left, right);
            return new Interval(lowerBound, upperBound).Exp();
        }

        Interval BoundGammaDensityTransformed(Interval p, double shape)
        {
            // Pareto transformation:
            // p = 1 - x^-2  (x >= 1)
            // x = 1/sqrt(1-p)
            // dp = 2 x^-3 dx
            // int_1^inf f(x) dx => int_0^1 f(1/sqrt(1-p))/sqrt(1-p)^3/2 dp
            var x = 1.0 / (1 - p).Sqrt();
            return BoundGammaDensity(x, shape + 3) / 2;
        }

        [Fact]
        public void Interval_GetExpectationTest()
        {
            Gaussian dist = new Gaussian(1, 2);
            double expected = Math.Exp(dist.GetLogAverageOf(new Gaussian(0, 1)));
            Interval actual = Interval.GetExpectation(1e-3, default(CancellationToken), dist, BoundGaussian);
            Console.WriteLine($"actual = {actual} should be {expected}");
            Assert.True(actual.Contains(expected));
        }

        [Fact]
        public void Interval_GetExpectation2Test()
        {
            Gaussian dist = new Gaussian(1, 2);
            double expected = Math.Exp(dist.GetLogAverageOf(new Gaussian(0, 1)));
            Interval actual = Interval.GetExpectation(
                1000, default, Interval.Point(-10), Interval.Point(10),
                EstimatedDistribution.NoError(dist),
                BoundGaussian);
            Console.WriteLine($"actual = {actual} should be {expected}");
            Assert.True(actual.Contains(expected));
        }

        /// <summary>
        /// Computes bounds of N(x;0,1) over an interval <paramref name="x"/>.
        /// </summary>
        Interval BoundGaussian(Interval x)
        {
            Interval abs = x.Abs();
            return new Interval(Math.Exp(-abs.UpperBound * abs.UpperBound / 2) * MMath.InvSqrt2PI, Math.Exp(-abs.LowerBound * abs.LowerBound / 2) * MMath.InvSqrt2PI);
        }

        private static Interval GetEstimate(Func<int, Interval> boundFunc)
        {
            int count = 30;
            while (true)
            {
                Interval interval = boundFunc(count);
                double lowerBound = interval.LowerBound;
                double upperBound = interval.UpperBound;
                if (double.IsNaN(lowerBound)) throw new Exception("lowerBound is NaN");
                if (double.IsNaN(upperBound)) throw new Exception("upperBound is NaN");
                if (lowerBound > upperBound) throw new Exception("lowerBound > upperBound");
                double error = interval.RelativeError();
                if (count > 1000)
                    Trace.WriteLine($"count = {count}");
                if (error < 1e-2 || count >= 100000)
                    return interval;
                count *= 2;
            }
        }

        [Fact]
        public void Interval_ApplySquareTest()
        {
            Interval input = new Interval(-3, 2);
            Interval outputBad = BoundSquareBadly(input);
            Interval output = input.Apply(1e-1, BoundSquareBadly);
            Console.WriteLine($"output = {output}, relative error = {output.RelativeError()}, bad output = {outputBad}, relative error = {outputBad.RelativeError()}");
            Assert.True(outputBad.Contains(output));
        }

        Interval BoundSquareBadly(Interval x)
        {
            return x * x;
        }

        internal void Interval_CircleParabolaTest()
        {
            Interval x = new Interval(0.1, 1);
            x = new Interval(-1, -0.1);
            //x = new Interval(double.NegativeInfinity, double.PositiveInfinity);
            x = new Interval(-0.9999, 0.9999);
            Interval y1B = new Interval(double.NegativeInfinity, double.PositiveInfinity);
            Interval y2B = new Interval(double.NegativeInfinity, double.PositiveInfinity);
            Interval yyB = new Interval(double.NegativeInfinity, double.PositiveInfinity);
            var yF = x.Square();
            Interval yB = new Interval(double.NegativeInfinity, double.PositiveInfinity);
            for (int i = 0; i < 200; i++)
            {
                var y2F = yF.Intersect(y1B);//.Intersect(y2B);
                yyB = 1 - y2F;
                var y1F = yF.Intersect(y2B);
                y1B = yyB.Max(0).SquareInv(y1F);
                var yyF = y1F.Square();
                y2B = 1 - yyF;
                yB = y2B.Intersect(y1B);
                var y = yF.Intersect(yB);
                if (y.LowerBound >= y.UpperBound)
                {
                    Trace.WriteLine($"no solution");
                    break;
                }
                Trace.WriteLine($"y = {y}");
            }
            x = yB.Max(0).SquareInv(x);
            Trace.WriteLine($"x = {x}");
        }
    }
}
