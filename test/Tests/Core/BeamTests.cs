namespace Microsoft.ML.Probabilistic.Tests.Core
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;
    using Xunit;
    using Assert = AssertHelper;

    public class BeamTests
    {
        [Fact]
        public void Beam_GetExpectation2Test()
        {
            var distribution = new Gaussian(1, 2*2).Truncate(-3, 3);
            double expected = 1;
            double maximumError = 1e-2;
            Func<Interval, Beam> func = x => new Beam(1, 0);
            foreach (Interval left in new[] { Interval.Point(0), new Interval(0, 2) })
            {
                foreach (Interval right in new[] { Interval.Point(2), new Interval(0, 2) })
                {
                    var beam = Beam.GetExpectation(maximumError, default(CancellationToken), left, right, distribution, true, func);
                    //var beam = Beam.GetExpectation2(maximumError, default(CancellationToken), left, right, distribution, func);
                    var actual = beam.GetOutputInterval(left, right);
                    Trace.WriteLine($"actual={actual} expected={expected}");
                    if (left.Width() == 0 && right.Width() == 0)
                        Assert.True(actual.Width() <= maximumError);
                    Assert.True(actual.Contains(expected));
                    if(left.UpperBound >= right.LowerBound)
                    {
                        Assert.True(actual.Contains(Math.Min(left.UpperBound, right.UpperBound)));
                        Assert.True(actual.Contains(Math.Max(left.LowerBound, right.LowerBound)));
                    }
                }
            }
        }

        [Fact]
        public void Beam_GetExpectationTest()
        {
            var distribution = new Gaussian(1, 0.2*0.2).Truncate(-3, 3);
            double expected = 1;
            double maximumError = 1e-2;
            Func<Interval, Beam> func = x => new Beam(1, 0);
            var beam = Beam.GetExpectation(maximumError, default(CancellationToken), distribution, func);
            var actual = beam.Offset;
            //Trace.WriteLine($"actual={actual} expected={expected}");
            Assert.True(actual.Width() <= maximumError);
            Assert.True(actual.Contains(expected));
        }

        [Fact]
        public void Beam_IntegrateTest()
        {
            Interval x = new Interval(0, 1);
            // 4/(1 + x^2)
            double expected = Math.PI;
            Func<Interval, Interval> func = y => 4 / (1 + y.Square());
            func = delegate (Interval y)
            {
                Beam denom = Beam.Square(y) + 1;
                var denomInterval = denom.GetOutputInterval(y);
                Beam inv = Beam.Reciprocal(denomInterval) * 4;
                Beam beam = inv.Compose(denom);
                return beam.Integrate(y).GetOutputInterval() / y.Width();
            };
            double maximumError = 1e-2;
            var actual = x.Integrate(maximumError, func);
            //Trace.WriteLine($"actual={actual} expected={expected}");
            Assert.True(actual.Width() <= maximumError);
            Assert.True(actual.Contains(expected));
        }

        [Fact]
        public void Beam_WeightedAverageTest()
        {
            for (int w2Lower = -1; w2Lower < 2; w2Lower++)
            {
                for (int k = 0; k < 10; k++)
                {
                    for (int j = 0; j < 10; j++)
                    {
                        for (int i = 0; i < 10; i++)
                        {
                            Interval w1 = new Interval(0, i / 10.0);
                            Interval v1 = -0.5 + new Interval(0, k / 10.0);
                            Interval w2 = w2Lower + new Interval(0, j / 10.0);
                            Interval v2 = -0.25 + new Interval(0, j / 10.0);
                            Beam beam = Beam.WeightedAverage(w1, v1, w2, v2);
                            //Trace.WriteLine(beam);
                            CheckBeam(beam, RegionFromIntervals(w1, v1, w2, v2), v => MMath.WeightedAverage(Math.Max(0, v[0]), v[1], Math.Max(0, v[2]), v[3]));
                        }
                    }
                }
            }
        }

        [Fact]
        public void Beam_ProductTest()
        {
            for (int i = 0; i < 10; i++)
            {
                Interval x = new Interval(-0.5, -0.5 + i / 10.0);
                Interval y = new Interval(-0.5, -0.5 + i / 10.0);
                Beam beam = Beam.Product(x, y);
                Trace.WriteLine(beam);
                CheckBeam(beam, RegionFromIntervals(x, y), v => v[0] * v[1]);
            }
        }

        public static Region RegionFromIntervals(params Interval[] intervals)
        {
            Region region = new Region(intervals.Length);
            for (int i = 0; i < intervals.Length; i++)
            {
                region.Lower[i] = intervals[i].LowerBound;
                region.Upper[i] = intervals[i].UpperBound;
            }
            return region;
        }

        public static double[] linspace(double min, double max, int count)
        {
            if (count < 2)
                throw new ArgumentException("count < 2");
            double inc = (max - min) / (count - 1);
            return Util.ArrayInit(count, i => (min + i * inc));
        }

        [Fact]
        public void Beam_GetProbLessThanTest()
        {
            var distribution = new Gaussian(1, 0.2*0.2);
            Interval input = new Interval(double.NegativeInfinity, -1);
            //input = new Interval(double.MinValue, -1);
            Beam beam = Beam.GetProbLessThan(distribution, 100, input, out Interval output);
            Assert.True(output.LowerBound >= 0);
            CheckBeam(x => Beam.GetProbLessThan(distribution, 100, x, out Interval ignore), distribution.GetProbLessThan, 1);
        }

        [Fact]
        public void Beam_NormalCdfTest()
        {
            CheckBeam(Beam.NormalCdf, MMath.NormalCdf);
        }

        [Fact]
        public void Beam_AbsTest()
        {
            CheckBeam(Beam.Abs, Math.Abs);
        }

        [Fact]
        public void Beam_MaxTest()
        {
            double minimum = 0;
            CheckBeam(x => Beam.Max(x, minimum), x => Math.Max(x, minimum));
        }

        [Fact]
        public void Beam_MinTest()
        {
            double maximum = 0;
            CheckBeam(x => Beam.Min(x, maximum), x => Math.Min(x, maximum));
        }

        [Fact]
        public void Beam_SquareTest()
        {
            CheckBeam(Beam.Square, x => x * x);
        }

        [Fact]
        public void Beam_ReciprocalTest()
        {
            CheckBeam(Beam.Reciprocal, x => 1 / x, 1);
        }

        [Fact]
        public void Beam_ExpTest()
        {
            CheckBeam(Beam.Exp, Math.Exp);
        }

        private static Beam BadExp(Interval input)
        {
            return new Beam(
                0,
                new Interval(Math.Exp(input.LowerBound), Math.Exp(input.UpperBound))
            );
        }

        private void CheckBeam(Func<Interval, Beam> getBeam, Func<double, double> func)
        {
            CheckBeam(getBeam, func, -0.5);
            CheckBeam(getBeam, func, 1);
        }

        private void CheckBeam(Func<Interval, Beam> getBeam, Func<double, double> func, double lowerBound)
        {
            for (int i = 0; i < 10; i++)
            {
                Interval x = new Interval(lowerBound, lowerBound + i / 10.0);
                Beam beam = getBeam(x);
                Trace.WriteLine(beam);
                CheckBeam(beam, x.ToRegion(), v => func(v[0]));
            }
        }

        private void CheckBeam(Beam beam, Region region, Func<Vector, double> func)
        {
            Rand.Restart(0);
            for (int i = 0; i < 100; i++)
            {
                Vector input = region.Sample();
                double output = func(input);
                Interval interval = beam.GetOutputInterval(input);
                Assert.True(interval.LowerBound <= interval.UpperBound);
                Assert.True(interval.Contains(output));
            }
        }
    }
}
