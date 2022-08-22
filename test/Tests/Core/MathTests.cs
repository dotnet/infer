// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class MathTests
    {
        [Fact]
        public void WeightedAverageTest()
        {
            Assert.Equal(Environment.Is64BitProcess ? 3.86361619394904E-311 : 3.86361619394162E-311, MMath.WeightedAverage(0.82912896852490248, 2.5484859206000203E-311, 3.50752234977395E-313, 31.087830618727477));
            Assert.Equal(MMath.WeightedAverage(0.1, double.MinValue, 0.01, double.MinValue), double.MinValue);
            Assert.Equal(MMath.WeightedAverage(0.1, -double.Epsilon, double.MaxValue, -double.Epsilon), -double.Epsilon);
            Assert.Equal(MMath.WeightedAverage(1e-100, 2e-250, 1e-100, 4e-250), MMath.Average(2e-250, 4e-250));
            Assert.Equal(MMath.WeightedAverage(1e100, 2e250, 1e100, 4e250), MMath.Average(2e250, 4e250));
            Assert.Equal(MMath.WeightedAverage(0, 0, 0.1, -double.Epsilon), -double.Epsilon);
            Assert.Equal(MMath.WeightedAverage(0.1, -double.Epsilon, 0, double.NegativeInfinity), -double.Epsilon);
            Assert.False(double.IsNaN(MMath.WeightedAverage(1.7976931348623157E+308, double.NegativeInfinity, 4.94065645841247E-324, double.NegativeInfinity)));
            Assert.False(double.IsNaN(MMath.WeightedAverage(0.01, double.NegativeInfinity, double.MaxValue, double.MaxValue)));
            Assert.False(double.IsNaN(MMath.WeightedAverage(0.01, double.NegativeInfinity, double.Epsilon, double.NegativeInfinity)));
            Assert.Equal(double.MaxValue, MMath.WeightedAverage(double.MaxValue, double.MaxValue, double.MaxValue, double.MaxValue));
            const int limit = 2_000_000;
            int count = 0;
            Parallel.ForEach(OperatorTests.DoublesAtLeastZero(), wa =>
            {
                Parallel.ForEach(OperatorTests.DoublesAtLeastZero(), wb =>
                {
                    if (count > limit) return;
                    Trace.WriteLine($"wa = {wa}, wb = {wb}");
                    foreach (var a in OperatorTests.Doubles())
                    {
                        if (count > limit) break;
                        foreach (var b in OperatorTests.Doubles())
                        {
                            if (count > limit) break;
                            if (double.IsNaN(a + b)) continue;
                            double midpoint = MMath.WeightedAverage(wa, a, wb, b);
                            Assert.True(midpoint >= System.Math.Min(a, b), $"Failed assertion: MMath.WeightedAverage({wa:r}, {a:r}, {wb:r}, {b:r}) {midpoint} >= {System.Math.Min(a, b)}");
                            Assert.True(midpoint <= System.Math.Max(a, b), $"Failed assertion: MMath.WeightedAverage({wa:r}, {a:r}, {wb:r}, {b:r}) {midpoint} <= {System.Math.Max(a, b)}");
                            if (wa == wb) Assert.Equal(MMath.Average(a, b), midpoint);
                            Interlocked.Add(ref count, 1);
                        }
                    }
                });
            });
        }

        [Fact]
        public void WeightedAverage_IsMonotonic()
        {
            // Checking all cases takes a long time.
            const int limit = 2_000_000;
            int count = 0;
            Parallel.ForEach(OperatorTests.DoublesAtLeastZero(), wa =>
            {
                Parallel.ForEach(OperatorTests.DoublesAtLeastZero(), wb =>
                {
                    Parallel.ForEach(OperatorTests.Doubles(), a =>
                    {
                        if (count > limit) return;
                        Trace.WriteLine($"wa = {wa}, wb = {wb}, a = {a}");
                        foreach (var b in OperatorTests.Doubles())
                        {
                            if (count > limit) break;
                            if (double.IsNaN(a + b)) continue;
                            double midpoint = MMath.WeightedAverage(wa, a, wb, b);
                            foreach (var inc in OperatorTests.DoublesAtLeastZero())
                            {
                                if (count > limit) break;
                                var a2 = a + inc;
                                if (a2 == a) continue;
                                if (double.IsNaN(a2 + b)) continue;
                                double midpoint2 = MMath.WeightedAverage(wa, a2, wb, b);
                                Assert.True(midpoint2 >= midpoint, $"Failed assertion: {midpoint2} >= {midpoint}, wa={wa:r}, a={a:r}, a2={a2:r}, wb={wb:r}, b={b:r}");
                                Interlocked.Add(ref count, 1);
                            }
                        }
                    });
                });
            });
        }

        /// <summary>
        /// Tests a specific case that previously failed.
        /// </summary>
        [Fact]
        public void WeightedAverage_IsMonotonic2()
        {
            double wa = double.MaxValue;
            double a = 1;
            double a2 = 1.0000000000000011;
            double wb = 0.0010000000000000002;
            double b = double.MinValue;
            double midpoint = MMath.WeightedAverage(wa, a, wb, b);
            double midpoint2 = MMath.WeightedAverage(wa, a2, wb, b);
            Assert.True(midpoint2 >= midpoint, $"Failed assertion: {midpoint2} >= {midpoint}, wa={wa:r}, a={a:r}, a2={a2:r}, wb={wb:r}, b={b:r}");
        }

        /// <summary>
        /// Tests a specific case that previously failed.
        /// </summary>
        [Fact]
        public void WeightedAverage_IsMonotonic3()
        {
            double wa = 1E-05;
            double a = -1.7976931348623157E+308;
            double a2 = -1.7976931348623155E+308;
            double wb = 1.0;
            double b = -1E+287;
            double midpoint = MMath.WeightedAverage(wa, a, wb, b);
            double midpoint2 = MMath.WeightedAverage(wa, a2, wb, b);
            Assert.True(midpoint2 >= midpoint, $"Failed assertion: {midpoint2} >= {midpoint}, wa={wa:r}, a={a:r}, a2={a2:r}, wb={wb:r}, b={b:r}");
        }

        [Fact]
        public void MeanAccumulator_Add_ZeroWeight()
        {
            MeanAccumulator mva = new MeanAccumulator();
            mva.Add(4.5, 0.0);
            Assert.Equal(4.5, mva.Mean);
            mva.Add(4.5);
            mva.Add(double.PositiveInfinity, 0.0);
            Assert.Equal(4.5, mva.Mean);
        }

        [Fact]
        public void ToStringExactTest()
        {
            Assert.Equal("0", MMath.ToStringExact(0));
            Assert.Equal("NaN", MMath.ToStringExact(double.NaN));
            Assert.Equal(double.MaxValue, double.Parse(MMath.ToStringExact(double.MaxValue)));
            Assert.Equal(double.MinValue, double.Parse(MMath.ToStringExact(double.MinValue)));
            Assert.Equal("10.5", MMath.ToStringExact(10.5));
            Assert.Equal(10.05, double.Parse(MMath.ToStringExact(10.05)));
            Assert.Equal("0.100000000000000002505909183520875968569614680770370524992534231990046604318405148467630281218195010089496230627027825414891031146499880413081224609160619018271942662793458427551041478278701507022263926060379361392435977509403014386614147912551359088259101734169222292122040491862182202915561954185941852588326204092831631787205015401996986616948980410676557942431921652541808732242554300585073938340203330993157646467433638479065531661724812599598594906293782493759617177861888792970476530542335134710418229637566637950767497147854236589795152044892049176025289756709261767081824924720105632337755616538050643653812583050224659631159300563236507929025398878153811554013986009587978081167432804936359631140419153283449560376539011485874652862548828125e-299", MMath.ToStringExact(1e-300));
            Assert.Equal("0.988131291682493088353137585736442744730119605228649528851171365001351014540417503730599672723271984759593129390891435461853313420711879592797549592021563756252601426380622809055691634335697964207377437272113997461446100012774818307129968774624946794546339230280063430770796148252477131182342053317113373536374079120621249863890543182984910658610913088802254960259419999083863978818160833126649049514295738029453560318710477223100269607052986944038758053621421498340666445368950667144166486387218476578691673612021202301233961950615668455463665849580996504946155275185449574931216955640746893939906729403594535543517025132110239826300978220290207572547633450191167477946719798732961988232841140527418055848553508913045817507736501283943653106689453125e-322", MMath.ToStringExact(1e-322));
            Assert.Equal("0.4940656458412465441765687928682213723650598026143247644255856825006755072702087518652998363616359923797965646954457177309266567103559397963987747960107818781263007131903114045278458171678489821036887186360569987307230500063874091535649843873124733972731696151400317153853980741262385655911710266585566867681870395603106249319452715914924553293054565444011274801297099995419319894090804165633245247571478690147267801593552386115501348035264934720193790268107107491703332226844753335720832431936092382893458368060106011506169809753078342277318329247904982524730776375927247874656084778203734469699533647017972677717585125660551199131504891101451037862738167250955837389733598993664809941164205702637090279242767544565229087538682506419718265533447265625e-323", MMath.ToStringExact(double.Epsilon));
            Assert.Equal(1e-300, double.Parse(MMath.ToStringExact(1e-300)));
            Assert.Equal(1e-322, double.Parse(MMath.ToStringExact(1e-322)));
            Assert.Equal(double.Epsilon, double.Parse(MMath.ToStringExact(double.Epsilon)));
        }

        [Fact]
        public void DigammaInvTest()
        {
            for (int i = 0; i < 1000; i++)
            {
                double y = -3 + i * 0.01;
                double x = MMath.DigammaInv(y);
                double y2 = MMath.Digamma(x);
                double error = MMath.AbsDiff(y, y2, 1e-8);
                Assert.True(error < 1e-8);
            }
        }

        /// <summary>
        /// Test LogisticGaussian with large variance
        /// </summary>
        [Fact]
        public void LogisticGaussianTest2()
        {
            for (int i = 4; i < 100; i++)
            {
                double v = System.Math.Pow(10, i);
                double f = MMath.LogisticGaussian(1, v);
                double err = System.Math.Abs(f - 0.5);
                //Console.WriteLine("{0}: {1} {2}", i, f, err);
                Assert.True(err < 2e-2 / i);
            }
        }

        [Fact]
        public void NormalCdfRatioTest()
        {
            double r0 = MMath.Sqrt2PI / 2;
            Assert.Equal(r0, MMath.NormalCdfRatio(0));
            Assert.Equal(r0, MMath.NormalCdfRatio(6.6243372842224754E-170));
            Assert.Equal(r0, MMath.NormalCdfRatio(-6.6243372842224754E-170));
        }

        [Fact]
        public void NormalCdfIntegralTest()
        {
            Assert.True(0 <= NormalCdfIntegral(190187183095334850882507750944849586799124505055478568794871547478488387682304.0, -190187183095334850882507750944849586799124505055478568794871547478488387682304.0, -1, 0.817880416082724044547388352452631856079457366800004151664125953519049673808376291470533145141236089924006896061006277409614237094627499958581030715374379576478204968748786874650796450332240045653919846557755590765736997127532958984375e-78).Mantissa);
            Assert.True(0 <= NormalCdfIntegral(213393529.2046706974506378173828125, -213393529.2046706974506378173828125, -1, 0.72893668811495072384656764856902984306419313043079455383121967315673828125e-9).Mantissa);
            Assert.True(0 < NormalCdfIntegral(-0.421468532207607216033551367218024097383022308349609375, 0.42146843802130329326161017888807691633701324462890625, -0.99999999999999989, 0.62292398855983019004972723654291189010479001808562316000461578369140625e-8).Mantissa);

            // Checking all cases takes a long time.
            const int limit = 2_000_000;
            int count = 0;
            Parallel.ForEach(OperatorTests.Doubles(), x =>
            {
                if (count > limit) return;
                foreach (var y in OperatorTests.Doubles())
                {
                    if (count > limit) break;
                    foreach (var r in OperatorTests.Doubles().Where(d => d >= -1 && d <= 1))
                    {
                        if (count > limit) break;
                        MMath.NormalCdfIntegral(x, y, r);
                        Interlocked.Add(ref count, 1);
                    }
                }
            });
        }

        // Same as MMath.NormalCdfIntegral but avoids inconsistent values of r and sqrtomr2 when using arbitrary precision.
        public static ExtendedDouble NormalCdfIntegral(double x, double y, double r, double sqrtomr2)
        {
            if (sqrtomr2 < 0.618)
            {
                // In this regime, it is more accurate to compute r from sqrtomr2.
                // See NormalCdfRatioLn
                r = System.Math.Sign(r) * System.Math.Sqrt((1 - sqrtomr2) * (1 + sqrtomr2));
            }
            else
            {
                // In this regime, it is more accurate to compute sqrtomr2 from r.
                double omr2 = 1 - r * r;
                sqrtomr2 = System.Math.Sqrt(omr2);
            }
            return MMath.NormalCdfIntegral(x, y, r, sqrtomr2);
        }

        [Fact]
        public void NormalCdf2Test4()
        {
            for (int j = 8; j <= 17; j++)
            {
                double r = -1 + System.Math.Pow(10, -j);
                double expected = System.Math.Log(NormalCdfZero(r));
                for (int i = 5; i < 100; i++)
                {
                    double x = -System.Math.Pow(10, -i);
                    double y = -x;
                    double actual = MMath.NormalCdfLn(x, y, r);
                    //double actual = Math.Log(NormalCdfConFrac3(x, y, r));
                    double error;
                    if (actual == expected) error = 0;
                    else error = System.Math.Abs(actual - expected);
                    //Console.WriteLine($"NormalCdfLn({x},{y},{r}) = {actual}, error = {error}");
                    Assert.True(error < 1e-9);
                }
            }
        }

        // returns NormalCdf(0,0,r)
        public static double NormalCdfZero(double r)
        {
            return (System.Math.PI / 2 + System.Math.Asin(r)) / (2 * System.Math.PI);
        }

        [Fact]
        public void SoftmaxTest()
        {
            // Check sparse versus dense.
            // Check situation where some entries are positive or negative infinity
            // Check effects of different common values
            double trueCV = -1;

            double[][] array = new double[][]
                {
                    new double[] {trueCV, trueCV, 5, 4, trueCV, -3},
                    new double[] {trueCV, trueCV, double.NegativeInfinity, 5, trueCV, 7},
                    new double[] {trueCV, trueCV, double.PositiveInfinity, 5, trueCV, double.PositiveInfinity},
                    new double[] {trueCV, trueCV, double.NegativeInfinity, 5, trueCV, double.PositiveInfinity},
                    new double[]
                        {double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity}
                };

            double[] forcedCVArr = new double[]
                {
                    trueCV, 0, double.PositiveInfinity, double.NegativeInfinity
                };

            double[][] expected = new double[][]
                {
                    new double[] {0.001802, 0.001802, 0.7269, 0.2674, 0.001802, 0.0002439},
                    new double[] {0.0002952, 0.0002952, 0, 0.1191, 0.0002952, 0.88},
                    new double[] {0, 0, 0.5, 0, 0, 0.5},
                    new double[] {0, 0, 0, 0, 0, 1},
                    new double[] {0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667}
                };

            for (int pass = 0; pass < 2; pass++)
            {
                for (int i = 0; i < array.Length; i++)
                {
                    if (pass == 0)
                    {
                        Vector sma = MMath.Softmax(new List<double>(array[i]));
                        for (int e = 0; e < sma.Count; e++)
                            Assert.Equal(expected[i][e], sma[e], 1e-4);
                        Console.WriteLine(sma);
                    }
                    else
                    {
                        for (int k = 0; k < forcedCVArr.Length; k++)
                        {
                            var lst = SparseList<double>.Constant(array[i].Length, forcedCVArr[k]);
                            for (int j = 0; j < array[i].Length; j++)
                                if (array[i][j] != forcedCVArr[k])
                                    lst.SparseValues.Add(new ValueAtIndex<double>(j, array[i][j]));
                            Vector sma = MMath.Softmax(lst);
                            for (int e = 0; e < sma.Count; e++)
                                Assert.Equal(expected[i][e], sma[e], 1e-4);
                            Console.WriteLine(sma);
                        }
                    }
                }
            }
        }

        [Fact]
        public void MedianTest()
        {
            Assert.Equal(double.MaxValue, MMath.Median(new[] { double.MaxValue }));
            Assert.Equal(double.MaxValue, MMath.Median(new[] { double.MaxValue, double.MaxValue }));
            Assert.Equal(3, MMath.Median(new[] { double.MaxValue, 3, double.MinValue }));
        }

        internal static void GammaSpeedTest()
        {
            Stopwatch watch = new Stopwatch();
            MMath.GammaLn(0.1);
            MMath.GammaLn(3);
            MMath.GammaLn(20);
            for (int i = 0; i < 100; i++)
            {
                double x = (i + 1) * 0.1;
                watch.Restart();
                for (int trial = 0; trial < 1000; trial++)
                {
                    MMath.GammaLn(x);
                }
                watch.Stop();
                Console.WriteLine($"{x} {watch.ElapsedTicks}");
            }
        }
    }
}
