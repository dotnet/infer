// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Core.Maths;
using System.IO;

namespace Microsoft.ML.Probabilistic.Tests
{

    public class SpecialFunctionsTests
    {
        public const double TOLERANCE = 1e-15;

        public delegate double MathFcn(double arg);
        public delegate double MathFcn2(double arg1, double arg2);
        public delegate double MathFcn3(double arg1, double arg2, double arg3);
        public delegate double MathFcn4(double arg1, double arg2, double arg3, double arg4);

        public struct Test
        {
            public MathFcn fcn;
            public double[,] pairs;
        };

        private static string ToStringExact(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x) || x == 0) return x.ToString(System.Globalization.CultureInfo.InvariantCulture);
            long bits = BitConverter.DoubleToInt64Bits(x);
            ulong fraction = Convert.ToUInt64(bits & 0x000fffffffffffff);
            short exponent = Convert.ToInt16((bits & 0x7ff0000000000000) >> 52);
            if (exponent == 0)
            {
                // subnormal number
                exponent = -1022 - 52;
            }
            else
            {
                // normal number
                fraction += 0x0010000000000000;
                exponent = Convert.ToInt16(exponent - 1023 - 52);
            }
            while ((fraction & 1) == 0)
            {
                fraction >>= 1;
                exponent++;
            }
            string sign = (x >= 0) ? "" : "-";
            System.Numerics.BigInteger big;
            if (exponent >= 0)
            {
                big = System.Numerics.BigInteger.Pow(2, exponent) * fraction;
                return $"{sign}{big}";
            }
            else
            {
                // Rewrite 2^-4 as 5^4 * 10^-4
                big = System.Numerics.BigInteger.Pow(5, -exponent) * fraction;
                // At this point, we could output the big integer with an "E"{exponent} suffix.  
                // However, double.Parse does not correctly parse such strings.
                // Instead we insert a decimal point and eliminate the "E" suffix if possible.
                int digitCount = big.ToString().Length;
                if (digitCount < -exponent)
                {
                    return $"{sign}0.{big}e{exponent + digitCount}";
                }
                else
                {
                    System.Numerics.BigInteger pow10 = System.Numerics.BigInteger.Pow(10, -exponent);
                    System.Numerics.BigInteger integerPart = big / pow10;
                    System.Numerics.BigInteger fractionalPart = big - integerPart * pow10;
                    string zeros = new string('0', -exponent);
                    return $"{sign}{integerPart}.{fractionalPart.ToString(zeros)}";
                }
            }
        }

        [Fact]
        public void GammaSpecialFunctionsTest()
        {
            double[,] BesselI_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "BesselI.csv"));
            CheckFunctionValues("BesselI", MMath.BesselI, BesselI_pairs);

            double[,] Gamma_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Gamma.csv"));
            CheckFunctionValues("Gamma", MMath.Gamma, Gamma_pairs);

            double[,] GammaLn_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "GammaLn.csv"));
            CheckFunctionValues("GammaLn", new MathFcn(MMath.GammaLn), GammaLn_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
digamma(mpf('9.5'))
            */
            double[,] digamma_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Digamma.csv"));
            CheckFunctionValues("Digamma", new MathFcn(MMath.Digamma), digamma_pairs);

            double[,] trigamma_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Trigamma.csv"));
            CheckFunctionValues("Trigamma", new MathFcn(MMath.Trigamma), trigamma_pairs);

            double[,] tetragamma_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Tetragamma.csv"));
            CheckFunctionValues("Tetragamma", MMath.Tetragamma, tetragamma_pairs);
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

        [Fact]
        public void SpecialFunctionsTest()
        {
            double[,] logistic_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Logistic.csv"));

            CheckFunctionValues("Logistic", MMath.Logistic, logistic_pairs);

            double[,] logisticln_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "LogisticLn.csv"));
            CheckFunctionValues("LogisticLn", MMath.LogisticLn, logisticln_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
log(1+mpf('1e-3'))
             */
            double[,] log1plus_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Log1Plus.csv"));
            CheckFunctionValues("Log1Plus", MMath.Log1Plus, log1plus_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
log(1-exp(mpf('-3')))
            */
            double[,] log1MinusExp_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "Log1MinusExp.csv"));
            CheckFunctionValues("Log1MinusExp", MMath.Log1MinusExp, log1MinusExp_pairs);

            double[,] expminus1_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "ExpMinus1.csv"));
            CheckFunctionValues("ExpMinus1", MMath.ExpMinus1, expminus1_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
x = mpf("1e-4")
((exp(x)-1)/x - 1)/x - 0.5
            */
            double[,] expMinus1RatioMinus1RatioMinusHalf_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "ExpMinus1RatioMinus1RatioMinusHalf.csv"));
            CheckFunctionValues("ExpMinus1RatioMinus1RatioMinusHalf", MMath.ExpMinus1RatioMinus1RatioMinusHalf, expMinus1RatioMinus1RatioMinusHalf_pairs);

            double[,] logexpminus1_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "LogExpMinus1.csv"));
            CheckFunctionValues("LogExpMinus1", MMath.LogExpMinus1, logexpminus1_pairs);

            double[,] logsumexp_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "LogSumExp.csv"));
            CheckFunctionValues("LogSumExp", MMath.LogSumExp, logsumexp_pairs);
        }

        [Fact]
        public void NormalCdfTest()
        {
            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
ncdf(-12.2)
            */
            // In wolfram alpha: (not always accurate)
            // http://www.wolframalpha.com/input/?i=erfc%2813%2Fsqrt%282%29%29%2F2
            // In xcas: (not always accurate)
            // Digits := 30
            // phi(x) := evalf(erfc(-x/sqrt(2))/2);
            double[,] normcdf_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdf.csv"));
            CheckFunctionValues("NormalCdf", new MathFcn(MMath.NormalCdf), normcdf_pairs);

            Assert.Equal(0.5, MMath.NormalCdf(0));
            Assert.True(MMath.NormalCdf(7.9) <= 1);
            Assert.True(MMath.NormalCdf(-40) >= 0);
            Assert.True(MMath.NormalCdf(-80) >= 0);

            double[,] erfc_pairs = new double[normcdf_pairs.GetLength(0), normcdf_pairs.GetLength(1)];
            for (int i = 0; i < normcdf_pairs.GetLength(0); i++)
            {
                double input = normcdf_pairs[i, 0];
                double output = normcdf_pairs[i, 1];
                erfc_pairs[i, 0] = -input / MMath.Sqrt2;
                erfc_pairs[i, 1] = 2 * output;
            }
            CheckFunctionValues("Erfc", new MathFcn(MMath.Erfc), erfc_pairs);

            double[,] normcdfinv_pairs = new double[normcdf_pairs.GetLength(0), 2];
            for (int i = 0; i < normcdfinv_pairs.GetLength(0); i++)
            {
                double input = normcdf_pairs[i, 0];
                double output = normcdf_pairs[i, 1];
                if (!(Double.IsPositiveInfinity(input) || Double.IsNegativeInfinity(input)) && (input <= -10 || input >= 6))
                {
                    normcdfinv_pairs[i, 0] = 0.5;
                    normcdfinv_pairs[i, 1] = 0;
                }
                else
                {
                    normcdfinv_pairs[i, 0] = output;
                    normcdfinv_pairs[i, 1] = input;
                }
            }
            CheckFunctionValues("NormalCdfInv", new MathFcn(MMath.NormalCdfInv), normcdfinv_pairs);

            double[,] normcdfln_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfLn.csv"));
            CheckFunctionValues("NormalCdfLn", new MathFcn(MMath.NormalCdfLn), normcdfln_pairs);

            double[,] normcdflogit_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfLogit.csv"));
            CheckFunctionValues("NormalCdfLogit", new MathFcn(MMath.NormalCdfLogit), normcdflogit_pairs);
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
        public void LogisticGaussianTest()
        {
            double[,] logisticGaussian_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "logisticGaussian.csv"));
            CheckFunctionValues("logisticGaussian", new MathFcn2(MMath.LogisticGaussian), logisticGaussian_pairs);

            double[,] logisticDerivGaussian_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "logisticGaussianDeriv.csv"));
            CheckFunctionValues("logisticGaussianDeriv", MMath.LogisticGaussianDerivative, logisticDerivGaussian_pairs);

            double[,] logisticDeriv2Gaussian_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "logisticGaussianDeriv2.csv"));
            CheckFunctionValues("logisticGaussianDeriv2", MMath.LogisticGaussianDerivative2, logisticDeriv2Gaussian_pairs);
        }

        [Fact]
        public void GammaUpperTest()
        {
            double[,] gammaLower_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "GammaLower.csv"));
            CheckFunctionValues("GammaLower", MMath.GammaLower, gammaLower_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
gammainc(mpf('1'),mpf('1'),mpf('inf'),regularized=True)
            */
            double[,] gammaUpperRegularized_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "GammaUpperRegularized.csv"));
            CheckFunctionValues("GammaUpperRegularized", (a,x) => MMath.GammaUpper(a, x, true), gammaUpperRegularized_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
gammainc(mpf('1'),mpf('1'),mpf('inf'),regularized=True)
            */
            double[,] gammaUpper_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "GammaUpper.csv"));
            CheckFunctionValues("GammaUpper", (a, x) => MMath.GammaUpper(a, x, false), gammaUpper_pairs);
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
        public void NormalCdfMomentRatioTest()
        {
            /* Anaconda Python script to generate a true value (must not be indented):
from mpmath import *
mp.dps = 100; mp.pretty = True
x = mpf('-2'); n=300;
exp(x*x/4)*pcfu(0.5+n,-x)
             */
            double[,] normcdfMomentRatio_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfMomentRatio.csv"));
            CheckFunctionValues("NormalCdfMomentRatio", delegate (double n, double x)
            {
                return MMath.NormalCdfMomentRatio((int)n, x);
            }, normcdfMomentRatio_pairs);
        }

#if false
    public static void NormalCdfRatioTest3()
    {
        // x=-2,-4,-8 are good positions for Taylor expansion
        double x = -4;
        for (int i = 0; i < 100; i++) {
            // x=-10: nBase=20,30 is unusually good, nBase=50 doesn't work as well
            // x=-20: nBase=50 works well
            // larger x allows larger nBase
            // x=-8: nBase=100
            // x=-6: nBase=10 works well
            // x=-5: poor
            // x=-4: nBase=40
            // x=-3: poor results for any nBase
            // x=-2: nBase=100 works well
            double r2 = MMath.NormalCdfMomentRatioRecurrence2(i, x, 100);
            double r = MMath.NormalCdfMomentRatioConFrac(i, x);
            //Console.WriteLine("recurrence = {0}, confrac = {1}", r1, r2);
            //Console.WriteLine("ncdfmr = {0}, ncdfmr/(i-1)!! = {1}", r, r/DoubleFact(i-1));
            //Console.WriteLine("ncdfmr = {0}, ncdfmr*i! = {1}", r, r*MMath.Gamma(i+1));
            Console.WriteLine("error = {0}", Math.Abs(r-r2)/Math.Abs(r));
        }
    }
    public static void NormalCdfRatioTest2()
    {
        // larger n requires more terms in the Taylor expansion
        // need nTerms=60 to get n=19, x=-4 from x=-8
        // nTerms=40 can do it in 2 steps, even though the 60 values for x=-6 are not accurate
        // (n=19,x=-4) in 4 steps: nTerms=30
        // (n=19,x=-4) in 1 step from x=-6: nTerms=100
        // thus it makes a difference where you start from
        double start = -6;
        double step = 2;
        double[][] table = NormalCdfRatioTable2(1, 20, 100, start, step);
        double x = start;
        for (int i = 0; i < table.Length; i++) {
            for (int j = 0; j < table[i].Length; j++) {
                double r = MMath.NormalCdfMomentRatioConFrac(j, x);
                Console.WriteLine("({0},{1}) error = {2}", j, x, Math.Abs(table[i][j]-r)/r);
            }
            x += step;
        }
    }
    public static double[][] NormalCdfRatioTable2(int nSteps, int nMax, int nTerms, double start, double step)
    {
        double[][] table = new double[nSteps+1][];
        double[] derivs = new double[nMax+nSteps*nTerms];
        for (int n = 0; n < derivs.Length; n++) {
            //Console.WriteLine("ncdfmr({0}) = {1}", n, NormalCdfMomentRatioConFrac(n, start));
            derivs[n] = MMath.NormalCdfMomentRatioConFrac(n, start);
            //Console.WriteLine("derivs[{0}] = {1}", n, derivs[n]);
        }
        table[0] = derivs;
        for (int i = 0; i < nSteps; i++) {
            int nMax2 = derivs.Length - nTerms;
            Console.WriteLine("{0} nMax2 = {1}", i, nMax2);
            double[] newDerivs = new double[nMax2];
            for (int n = 0; n < nMax2; n++) {
                newDerivs[n] = MMath.NormalCdfMomentRatioTaylor(n, step, derivs);
            }
            table[i+1] = newDerivs;
            derivs = newDerivs;
        }
        return table;
    }
    public static double[][] NormalCdfRatioTable(int nSteps, int nMax, int nTerms, double step)
    {
        double[][] table = new double[nSteps+1][];
        double[] derivs = new double[nMax+nSteps*nTerms];
        derivs[0] = 0.5*MMath.Sqrt2PI;
        derivs[1] = 1;
        for (int n = 2; n < derivs.Length; n++) {
            derivs[n] = derivs[n-2]/n;
        }
        table[0] = derivs;
        for (int i = 0; i < nSteps; i++) {
            int nMax2 = derivs.Length - nTerms;
            Console.WriteLine("{0} nMax2 = {1}", i, nMax2);
            double[] newDerivs = new double[nMax2];
            for (int n = 0; n < nMax2; n++) {
                newDerivs[n] = MMath.NormalCdfMomentRatioTaylor(n, step, derivs);
            }
            table[i+1] = newDerivs;
            derivs = newDerivs;
        }
        return table;
    }
#endif
        // this test shows that a Taylor expansion at zero is not reliable for x less than -0.5
        internal void NormalCdfMomentRatioTaylorZeroTest()
        {
            double x = -1;
            for (int n = 0; n < 20; n++)
            {
                Console.WriteLine("{0} {1} {2}", n, MMath.NormalCdfMomentRatio(n, x), NormalCdfMomentRatioTaylorZero(n, x));
            }
        }

        public static double NormalCdfMomentRatioTaylorZero(int n, double x)
        {
            double sum = 0;
            double Reven = MMath.Sqrt2PI / 2;
            double Rodd = 1;
            if (n % 2 == 0)
            {
                // Reven = (1*3*...*(n-1))/n! = 1/(2*4*...*n)
                // Rodd = (2*4*...*n)/n! = 1/(1*3*...*(n-1))
                int h = n / 2;
                for (int i = 0; i < h; i++)
                {
                    Reven /= 2 * (i + 1);
                    Rodd /= 2 * i + 1;
                }
                Rodd *= x;
            }
            else
            {
                // Reven = (1*3*...*n)/n! = 1/(2*4*...*(n-1)) 
                // Rodd = (2*4*...*(n-1))/n! = 1/(1*3*...*n)
                int h = (n - 1) / 2;
                for (int i = 0; i < h; i++)
                {
                    Reven /= 2 * (i + 1);
                    Rodd /= 2 * i + 3;
                }
                Reven *= x;
            }
            double x2 = x * x;
            double sumOld = 0;
            for (int i = 0; i < 100; i++)
            {
                if ((i + n) % 2 == 0)
                {
                    sum += Reven;
                    Reven *= x2 * (i + n + 1) / ((i + 1) * (i + 2));
                }
                else
                {
                    sum += Rodd;
                    Rodd *= x2 * (i + n + 1) / ((i + 1) * (i + 2));
                }
                if (sum == sumOld)
                {
                    Console.WriteLine("{0} terms", i);
                    break;
                }
                sumOld = sum;
            }
            return sum;
        }

        internal static void ConFracTest()
        {
            // for x=-1, n=0: 300, n=100: 200
            // for x=-2, n=0: 110, n=100: 270
            // for x=-3, n=0: 50, n=100: 170
            // for x=-4, n=0: 40, n=100: 120
            // for x=-5, n=0: 25, n=100: 100
            // for x=-6, n=0: 20, n=100: 80, n=200: 100
            // for x=-10, n=0: 10, n=100: 40, n=200: 60
            // for x=-20, n=0: 10, n=100: 20
            // for x=-40, the number of terms remains around 10
            double x = -10;
            for (int i = 0; i < 200; i++)
            {
                int n = NormalCdfMomentRatioConFracCount(i, x);
                Console.WriteLine("{0}, {1}", i, n);
            }
        }

        public static int NormalCdfMomentRatioConFracCount(int n, double x)
        {
            if (x >= 0)
                throw new ArgumentException("x >= 0", nameof(x));
            double invX = 1 / x;
            double invX2 = invX * invX;
            double numer = -invX;
            double numerPrev = 0;
            double denom = 1;
            double denomPrev = 1;
            double a = invX2;
            for (int i = 1; i <= n; i++)
            {
                numer *= -invX;
                double denomNew = denom + a * denomPrev;
                a += invX2;
                denomPrev = denom;
                denom = denomNew;
            }
            double rprev = 0;
            for (int i = 0; i < 1000; i++)
            {
                double numerNew = numer + a * numerPrev;
                double denomNew = denom + a * denomPrev;
                a += invX2;
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                double r = numer / denom;
                if (r == rprev)
                    return i;
                rprev = r;
            }
            return Int32.MaxValue;
        }

        public static int NormalCdfRatioConFracCount(double x, double expected)
        {
            Assert.True(x < 0);
            double invX = 1 / x;
            double invX2 = invX * invX;
            // Continued fraction approach
            // can also be used for x > -6, but prohibitively many terms would be required.
            double numer = -invX;
            double numerPrev = 0;
            double denom = 1;
            double denomPrev = 1;
            double a = invX2;
            double rprev = 0;
            for (int i = 1; i < 100; i++)
            {
                double numerNew = numer + a * numerPrev;
                double denomNew = denom + a * denomPrev;
                a += invX2;
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                double r = numer / denom;
                if (r == expected)
                    return i;
                if (r == rprev)
                    return i;
                rprev = r;
            }
            Console.WriteLine("actual = {0} expected = {1}", numer / denom, expected);
            return 50;
        }

        internal void NormalCdfSpeedTest()
        {
            // current results:
            // NormalCdf(0.5): 28ms
            // NormalCdf2: 1974ms
            MMath.NormalCdf(0.5);
            double x = 1.5;
            MMath.NormalCdf(x, 0.5, 0.95);
            Stopwatch watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < 100000; i++)
            {
                MMath.NormalCdf(0.5);
            }
            watch.Stop();
            Console.WriteLine("NormalCdf(0.5): " + watch.ElapsedMilliseconds + "ms");
            watch.Restart();
            for (int i = 0; i < 100000; i++)
            {
                NormalCdf_Quadrature(x, 0.5, 0.1);
            }
            watch.Stop();
            Console.WriteLine("NormalCdf2Quad: " + watch.ElapsedMilliseconds + "ms");
            watch.Restart();
            for (int i = 0; i < 100000; i++)
            {
                NormalCdfAlt(x, 0.5, 0.1);
            }
            watch.Stop();
            Console.WriteLine("NormalCdf2Alt: " + watch.ElapsedMilliseconds + "ms");
            watch.Restart();
            for (int i = 0; i < 100000; i++)
            {
                MMath.NormalCdf(x, 0.5, 0.95);
            }
            watch.Stop();
            Console.WriteLine("NormalCdf2: " + watch.ElapsedMilliseconds + "ms");
        }

        /// <summary>
        /// Used to tune MMath.NormalCdfLn.  The best tuning minimizes the number of messages printed.
        /// </summary>
        internal void NormalCdf2Test2()
        {
            // Call both routines now to speed up later calls.
            MMath.NormalCdf(-2, -2, -0.5);
            NormalCdf_Quadrature(-2, -2, -0.5);
            Stopwatch watch = new Stopwatch();
            double xmin = 0.1;
            double xmax = 0.1;
            double n = 20;
            double xinc = (xmax - xmin) / (n - 1);
            for (int xi = 0; xi < n; xi++)
            {
                if (xinc == 0 && xi > 0) break;
                double x = xmin + xi * xinc;
                double ymin = -System.Math.Abs(x)*10;
                double ymax = -ymin;
                double yinc = (ymax - ymin) / (n - 1);
                for (int yi = 0; yi < n; yi++)
                {
                    double y = ymin + yi * yinc;
                    double rmin = -0.999999;
                    double rmax = -0.000001;
                    rmin = MMath.NextDouble(-1);
                    rmax = -1 + 1e-6;
                    //rmax = -0.5;
                    //rmax = 0.1;
                    //rmax = -0.58;
                    //rmax = -0.9;
                    //rmin = -0.5;
                    rmax = 1;
                    double rinc = (rmax - rmin) / (n - 1);
                    for (int ri = 0; ri < n; ri++)
                    {
                        double r = rmin + ri * rinc;
                        string good = "good";
                        watch.Restart();
                        double result1 = double.NaN;
                        try
                        {
                            result1 = MMath.NormalCdfIntegral(x, y, r);
                        }
                        catch
                        {
                            good = "bad";
                            //throw;
                        }
                        watch.Stop();
                        long ticks = watch.ElapsedTicks;
                        watch.Restart();
                        double result2 = double.NaN;
                        try
                        {
                            result2 = NormalCdfLn_Quadrature(x, y, r);
                        }
                        catch
                        {
                        }
                        long ticks2 = watch.ElapsedTicks;
                        bool overtime = ticks > 10 * ticks2;
                        if (double.IsNaN(result1) /*|| overtime*/)
                            Trace.WriteLine($"({x:r},{y:r},{r:r},{x-r*y}): {good} {ticks} {ticks2} {result1} {result2}");
                    }
                }
            }
        }

        internal void NormalCdf2SpeedTest()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            double result = 0;
            double result2 = 0;
            for (int i = 0; i < 10000; i++)
            {
                result = NormalCdfAlt(0.5, 0.5, 0.1);
                result2 += result;
            }
            watch.Stop();
            long ticks = watch.ElapsedTicks;
            Console.WriteLine("{0} {1} {2}", result, result2, ticks);
        }

        // Used to debug MMath.NormalCdf
        internal void NormalCdf2Test3()
        {
            double x, y, r;
            bool first = true;
            if (first)
            {
                // x=-2, y=-10, r=0.9 is dominated by additive part of numerator - poor convergence
                // -2,-2,-0.5
                x = -1.0058535005109381;
                y = -0.11890687017604007;
                r = -0.79846947062734286;
                x = -63;
                y = 63;
                r = -0.4637494637494638;

                x = -1.0329769464004883E-08;
                y = 1.0329769464004876E-08;
                r = -0.99999999999999512;

                x = 0;
                y = 0;
                r = -0.6;

                x = -1.15950886531361;
                y = 0.989626418003324;
                r = -0.626095038754337;

                x = -1.5;
                y = 1.5;
                r = -0.49;

                x = -1.6450031341281908;
                y = 1.2645625117080999;
                r = -0.054054238344620031;

                x = -0.5;
                y = -0.5;
                r = 0.001;

                Console.WriteLine(1 - r * r);

                Console.WriteLine("NormalCdfBrute: {0}", NormalCdfBrute(0, x, y, r));
                Console.WriteLine("NormalCdf_Quadrature: {0}", NormalCdf_Quadrature(x, y, r));
                //Console.WriteLine("{0}", NormalCdfAlt2(x, y, r));
                //Console.WriteLine("NormalCdfAlt: {0}", NormalCdfAlt(x, y, r));
                //Console.WriteLine("NormalCdfTaylor: {0}", MMath.NormalCdfRatioTaylor(x, y, r));
                //Console.WriteLine("NormalCdfConFrac3: {0}", NormalCdfConFrac3(x, y, r));
                //Console.WriteLine("NormalCdfConFrac4: {0}", NormalCdfConFrac4(x, y, r));
                //Console.WriteLine("NormalCdfConFrac5: {0}", NormalCdfConFrac5(x, y, r));
                Console.WriteLine("MMath.NormalCdf: {0}", MMath.NormalCdf(x, y, r));
                Console.WriteLine("MMath.NormalCdfLn: {0}", MMath.NormalCdfLn(x, y, r));
                for (int i = 1; i < 50; i++)
                {
                    //Console.WriteLine("{0}: {1}", i, NormalCdfBrute(i, x, y, r));
                }
                //x = 0;
                //y = 0;
                //r2 = -0.7;
                //r2 = -0.999;
            }
            else
            {
                // x=-2, y=-10, r=0.9 is dominated by additive part of numerator - poor convergence
                // -2,-2,-0.5
                x = -0.1;
                y = -0.1;
                r = -0.999999;
                Console.WriteLine("{0}", MMath.NormalCdfLn(x, y, r));
                //x = 0;
                //y = 0;
                //r2 = -0.7;
                //r2 = -0.999;
            }
        }

        [Fact]
        public void NormalCdf2Test()
        {
            double x = 1.2, y = 1.3;
            double xy1 = System.Math.Max(0.0, MMath.NormalCdf(x) + MMath.NormalCdf(y) - 1);

            Assert.True(0 < MMath.NormalCdf(6.8419544775976187E-08, -5.2647906596206016E-08, -1, 3.1873689658872377E-10).Mantissa);

            // In sage:
            // integral(1/(2*pi*sqrt(1-t*t))*exp(-(x*x+y*y-2*t*x*y)/(2*(1-t*t))),t,-1,r).n(digits=200);
            double[,] normalcdf2_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdf2.csv"));
            CheckFunctionValues("NormalCdf2", new MathFcn3(MMath.NormalCdf), normalcdf2_pairs);

            // using wolfram alpha:  (for cases where r=-1 returns zero)
            // log(integrate(1/(2*pi*sqrt(1-t*t))*exp(-(2*0.1*0.1 - 2*t*0.1*0.1)/(2*(1-t*t))),t,-1,-0.9999))
            // log(integrate(1/(2*pi*sqrt(1-t*t))*exp(-(2*1.5*1.5 + 2*t*1.5*1.5)/(2*(1-t*t))),t,-1,-0.5))
            double[,] normalcdfln2_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfLn2.csv"));
            CheckFunctionValues("NormalCdfLn2", new MathFcn3(MMath.NormalCdfLn), normalcdfln2_pairs);

            double[,] normalcdfRatioLn2_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfRatioLn2.csv"));
            CheckFunctionValues("NormalCdfRatioLn2", new MathFcn4(MMath.NormalCdfRatioLn), normalcdfRatioLn2_pairs);

            double[,] normalcdfIntegral_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfIntegral.csv"));
            CheckFunctionValues("NormalCdfIntegral", new MathFcn3(MMath.NormalCdfIntegral), normalcdfIntegral_pairs);

            double[,] normalcdfIntegralRatio_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "NormalCdfIntegralRatio.csv"));
            CheckFunctionValues("NormalCdfIntegralRatio", new MathFcn3(MMath.NormalCdfIntegralRatio), normalcdfIntegralRatio_pairs);
        }

        [Fact]
        public void NormalCdfIntegralTest()
        {
            Assert.True(0 <= MMath.NormalCdfIntegral(1.9018718309533485E+77, -1.9018718309533485E+77, -1, 8.17880416082724E-79).Mantissa);
            Assert.True(0 <= MMath.NormalCdfIntegral(213393529.2046707, -213393529.2046707, -1, 7.2893668811495072E-10).Mantissa);
            Assert.True(0 < MMath.NormalCdfIntegral(-0.42146853220760722, 0.42146843802130329, -0.99999999999999989, 6.2292398855983019E-09).Mantissa);
 
            Parallel.ForEach (OperatorTests.Doubles(), x =>
            {
                foreach (var y in OperatorTests.Doubles())
                {
                    foreach (var r in OperatorTests.Doubles().Where(d => d >= -1 && d <= 1))
                    {
                        MMath.NormalCdfIntegral(x, y, r);
                    }
                }
            });
        }

        internal void NormalCdfIntegralTest2()
        {
            double x = 0.0093132267868981222;
            double y = -0.0093132247056551785;
            double r = -1;
            y = -2499147.006377392;
            x = 2499147.273918618;
            //MMath.TraceConFrac = true;
            //MMath.TraceConFrac2 = true;
            for (int i = 0; i < 100; i++)
            {
                //x = 2.1 * (i + 1);
                //y = -2 * (i + 1);
                //x = -2 * (i + 1);
                //y = 2.1 * (i + 1);
                //x = -System.Math.Pow(10, -i);
                //y = -x * 1.1;
                x = -0.33333333333333331;
                y = -1.5;
                r = 0.16666666666666666;
                x = -0.4999;
                y = 0.5;
                x = -0.1;
                y = 0.5;
                r = -0.1;

                x = -824.43680216388009;
                y = -23300.713731480908;
                r = -0.99915764591723821;
                x = -0.94102098773740084;
                x = 1 + i * 0.01;
                y = 2;
                r = 1;

                x = 0.021034851174404436;
                y = -0.37961242087533614;
                //x = -0.02;
                //y += -1;
                //x -= -1;
                r = -1 + System.Math.Pow(10, -i);

                //x = i * 0.01;
                //y = -1;
                //r = -1 + 1e-8;

                // 1.81377005549484E-40 with exponent
                // flipped is 1.70330340479022E-40
                //x = -1;
                //y = -8.9473684210526319;
                //x = System.Math.Pow(10, -i);
                //y = x;
                //r = -0.999999999999999;

                //x = -0.94102098773740084;
                //y = -1.2461486442846208;
                //r = 0.5240076921033775;

                x = 790.80368892437889;
                y = -1081776354979.6719;
                y = -System.Math.Pow(10, i);
                r = -0.94587440643473975;

                x = -39062.492380206008;
                y = 39062.501110681893;
                r = -0.99999983334056686;

                //x = -2;
                //y = 1.5789473684210522;
                //r = -0.78947368421052622;

                //x = -1.1;
                //y = -1.1;
                //r = 0.052631578947368474;

                //x = 0.001;
                //y = -0.0016842105263157896;
                //r = -0.4;

                //x = 0.1;
                //x = 2000;
                //y = -2000;
                //r = -0.99999999999999989;

                x = double.MinValue;
                y = double.MinValue;
                r = 0.1;


                Trace.WriteLine($"(x,y,r) = {x:r}, {y:r}, {r:r}");

                double intZOverZ;
                try
                {
                    intZOverZ = MMath.NormalCdfIntegralRatio(x, y, r);
                }
                catch
                {
                    intZOverZ = double.NaN;
                }
                Trace.WriteLine($"intZOverZ = {intZOverZ:r}");

                double intZ0 = NormalCdfIntegralBasic(x, y, r);
                double intZ1 = 0; // NormalCdfIntegralFlip(x, y, r);
                double intZr = 0;// NormalCdfIntegralBasic2(x, y, r);
                ExtendedDouble intZ;
                double sqrtomr2 = System.Math.Sqrt((1 - r) * (1 + r));
                try
                {
                    intZ = MMath.NormalCdfIntegral(x, y, r, sqrtomr2);
                }
                catch
                {
                    intZ = ExtendedDouble.NaN();
                }
                //double intZ = intZ0;
                Trace.WriteLine($"intZ = {intZ:r} {intZ.ToDouble():r} {intZ0:r} {intZ1:r} {intZr:r}");
                if (intZ.Mantissa < 0) throw new Exception();
                //double intZ2 = NormalCdfIntegralBasic(y, x, r);
                //Trace.WriteLine($"intZ2 = {intZ2} {r*intZ}");
                double Z = MMath.NormalCdf(x, y, r);
                if (Z < 0) throw new Exception();
            }
        }

        private double NormalCdfIntegralFlip(double x, double y, double r)
        {
            double logProbX = Gaussian.GetLogProb(x, 0, 1);
            return -MMath.NormalCdfIntegral(x, -y, -r) + x * MMath.NormalCdf(x) + System.Math.Exp(logProbX);
        }

        private double NormalCdfIntegralTaylor(double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            double sqrtomr2 = System.Math.Sqrt(omr2);
            double ymrx = y / sqrtomr2;
            double dx0 = MMath.NormalCdf(0, y, r);
            double ddx0 = System.Math.Exp(Gaussian.GetLogProb(0, 0, 1) + MMath.NormalCdfLn(ymrx));
            // \phi_{xx} &= -x \phi_x - r \phi_r
            double dddx0 = -r * System.Math.Exp(Gaussian.GetLogProb(0, 0, 1) + Gaussian.GetLogProb(ymrx, 0, 1));
            Trace.WriteLine($"dx0 = {dx0} {ddx0} {dddx0}");
            return MMath.NormalCdfIntegral(0, y, r) + x * dx0 + 0.5 * x * x * ddx0 + 1.0 / 6 * x * x * x * dddx0;
        }

        private double NormalCdfIntegralBasic2(double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            double sqrtomr2 = System.Math.Sqrt(omr2);
            double ymrx = (y - r * x) / sqrtomr2;
            double xmry = (x - r * y) / sqrtomr2;
            double func(double t)
            {
                return (y - r * x + r * t) * System.Math.Exp(Gaussian.GetLogProb(t, x, 1) + MMath.NormalCdfLn(ymrx + r * t / sqrtomr2));
            }
            func(0);
            double func2(double t)
            {
                double ymrxt = ymrx + r * t / sqrtomr2;
                return sqrtomr2 * System.Math.Exp(Gaussian.GetLogProb(t, x, 1) + Gaussian.GetLogProb(ymrxt, 0, 1)) * (MMath.NormalCdfMomentRatio(1, ymrxt) - 1);
            }
            func2(0);
            //return -MMath.NormalCdf(x, y, r) * (y / r - x) + Integrate(func2) / r;
            double func3(double t)
            {
                double xmryt = xmry + r * t / sqrtomr2;
                return sqrtomr2 * System.Math.Exp(Gaussian.GetLogProb(t, y, 1) + Gaussian.GetLogProb(xmryt, 0, 1)) * MMath.NormalCdfMomentRatio(1, xmryt);
            }
            //double Z = MMath.NormalCdf(x, y, r, out double exponent);
            double Z3 = Integrate(func3);
            //return System.Math.Exp(exponent)*(-Z * (y / r - x) - omr2 / r * MMath.NormalCdfRatio(xmry)) + Z3/r;
            return Z3;
        }

        private static double Integrate(Func<double, double> func)
        {
            double sum = 0;
            var ts = EpTests.linspace(0, 1, 100000);
            double inc = ts[1] - ts[0];
            for (int i = 0; i < ts.Length; i++)
            {
                double t = ts[i];
                double term = func(t);
                if (i == 0 || i == ts.Length - 1) term /= 2;
                sum += term * inc;
            }
            return sum;
        }

        private double NormalCdfIntegralBasic(double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            double sqrtomr2 = System.Math.Sqrt(omr2);
            double ymrx = (y - r * x) / sqrtomr2;
            double xmry = (x - r * y) / sqrtomr2;
            // should use this whenever x > 0 and Rymrx >= Rxmry (y-r*x >= x-r*y implies y*(1+r) >= x*(1+r) therefore y >= x)
            // we need a special routine to compute 2nd half without cancellation and without dividing by phir
            // what about x > y > 0?
            //double t = MMath.NormalCdfIntegral(-x, y, -r) + x * MMath.NormalCdf(y) + r * System.Math.Exp(Gaussian.GetLogProb(y, 0, 1));
            //Console.WriteLine(t);
            double phix = System.Math.Exp(Gaussian.GetLogProb(x, 0, 1) + MMath.NormalCdfLn(ymrx));
            double phiy = System.Math.Exp(Gaussian.GetLogProb(y, 0, 1) + MMath.NormalCdfLn(xmry));
            //Trace.WriteLine($"phix = {phix} phiy = {phiy}");
            return x * MMath.NormalCdf(x, y, r) + phix + r * phiy;
            //return y * MMath.NormalCdf(x, y, r) + r * System.Math.Exp(Gaussian.GetLogProb(x, 0, 1) + MMath.NormalCdfLn(ymrx)) + System.Math.Exp(Gaussian.GetLogProb(y, 0, 1) + MMath.NormalCdfLn(xmry));
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

        // computes phi_n
        public static double NormalCdfBrute(int n, double x, double y, double r)
        {
            int nSamples = 1000000;
            double tMax = 20;
            double inc = tMax / nSamples;
            double sum = 0;
            double f0 = 0;
            double omr2 = 1 - r * r;
            double s = System.Math.Sqrt(omr2);
            for (int i = 0; i < nSamples; i++)
            {
                double t = (i + 1) * inc;
                double diff1 = t - x;
                double diffy = y - r * (x - t);
                double f = System.Math.Pow(t, n) * System.Math.Exp(-0.5 * diff1 * diff1 + MMath.NormalCdfLn(diffy / s));
                if (i == 0)
                    f0 = f;
                if (i == nSamples - 1)
                {
                    if (f > f0 * 1e-20)
                        throw new Exception();
                }
                sum += f;
            }
            return sum * inc / MMath.Sqrt2PI;
        }

        public static double NormalCdfAlt(double x, double y, double r)
        {
            return NormalCdfConFrac5(x, y, r);
        }

        public static double NormalCdfAlt2(double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            double diff = (y - r * x) / System.Math.Sqrt(omr2);
            double logOffset = MMath.NormalCdfLn(x) + MMath.NormalCdfLn(diff);
            double psi = NormalCdfBrute2(0, x, y, r);
            Console.WriteLine("psi = {0}", psi);
            bool verbose = false;
            if (verbose)
            {
                double special = System.Math.Exp(MMath.NormalCdfLn(x) + Gaussian.GetLogProb(y, r * x, omr2));
                double psi2 = omr2 / r * (NormalCdfMomentDy(0, x, y, r) - special) / (r * x - y);
                Console.WriteLine("{0} approx {1}", psi, psi2);
                double psi1 = NormalCdfBrute2(1, x, y, r);
                Console.WriteLine("{0} {1}", r * psi1, (r * x - y) * psi + omr2 / r * (special - NormalCdfMomentDy(0, x, y, r)));
                Console.WriteLine("{0} {1}", NormalCdfMomentDy(0, x, y, r) + r / omr2 * ((y - r * x) * psi + r * psi1), special);
            }
            return System.Math.Exp(logOffset) + r * psi;
        }

        // computes psi_n
        public static double NormalCdfBrute2(int n, double x, double y, double r)
        {
            int nSamples = 1000000;
            double tMax = 20;
            double inc = tMax / nSamples;
            double sum = 0;
            double f0 = 0;
            double omr2 = 1 - r * r;
            double s = System.Math.Sqrt(omr2);
            for (int i = 0; i < nSamples; i++)
            {
                double t = (i + 1) * inc;
                double diff = (y - r * (x - t)) / s;
                double f = System.Math.Pow(t, n) * System.Math.Exp(-0.5 * diff * diff + MMath.NormalCdfLn(x - t));
                if (i == 0)
                    f0 = f;
                if (i == nSamples - 1)
                {
                    if (f > f0 * 1e-20)
                        throw new Exception();
                }
                sum += f;
            }
            return sum * inc / MMath.Sqrt2PI / s;
        }

        public static double NormalCdfMomentDyBrute(int n, double x, double y, double r)
        {
            int nSamples = 10000;
            double tMax = 20;
            double inc = tMax / nSamples;
            double sum = 0;
            double f0 = 0;
            double omr2 = 1 - r * r;
            double s = System.Math.Sqrt(omr2);
            for (int i = 0; i < nSamples; i++)
            {
                double t = i * inc;
                double y2 = x - r * y - t;
                //double f = Math.Pow(t, n) / (2 * Math.PI * s) * Math.Exp(-0.5 * (t - x) * (t - x) - 0.5 * y2 * y2 / omr2);
                double f = System.Math.Pow(t, n) / (2 * System.Math.PI * s) * System.Math.Exp(-0.5 * y * y - 0.5 * y2 * y2 / omr2);
                if (i == 1)
                    f0 = f;
                if (i == nSamples - 1)
                {
                    if (f > f0 * 1e-20)
                        throw new Exception();
                }
                sum += f;
            }
            return sum * inc;
        }

        // result is divided by n!! (odd n)
        public static double NormalCdfMomentDy2(int n, double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            if (omr2 == 0)
            {
                double lfact;
                if (n % 2 == 1)
                    lfact = DoubleFactorialLn(n);
                else
                    lfact = DoubleFactorialLn(n - 1);
                return System.Math.Pow(x - r * y, n) * System.Math.Exp(Gaussian.GetLogProb(y, 0, 1) - lfact);
            }
            else
            {
                double diff = (x - r * y) / System.Math.Sqrt(omr2);
                double lfact;
                if (n % 2 == 1)
                    lfact = DoubleFactorialLn(n - 1);   // n!/n!!
                else
                    lfact = DoubleFactorialLn(n) - System.Math.Log(n + 1);   // n!/(n+1)!!
                return System.Math.Exp(lfact + Gaussian.GetLogProb(y, 0, 1)
                    + Gaussian.GetLogProb(diff, 0, 1)
                    + 0.5 * n * System.Math.Log(omr2)) * MMath.NormalCdfMomentRatio(n, diff);
            }
        }

        /// <summary>
        /// Computes <c>log(n!!)</c> where n!! = n(n-2)(n-4)...2 (if n even) or n(n-2)...1 (if n odd)
        /// </summary>
        /// <param name="n">An integer &gt;= 0</param>
        /// <returns></returns>
        public static double DoubleFactorialLn(int n)
        {
            if (n < 0)
                throw new ArgumentException("n < 0");
            else if (n == 0)
                return 0;
            else if (n % 2 == 0)
            {
                int h = n / 2;
                return h * System.Math.Log(2) + MMath.GammaLn(h + 1);
            }
            else
            {
                int h = (n + 1) / 2;
                return h * System.Math.Log(2) + MMath.GammaLn(h + 0.5) - MMath.GammaLn(0.5);
            }
        }

        public static double NormalCdfMomentDy(int n, double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            if (omr2 == 0)
            {
                return System.Math.Pow(x - r * y, n) * System.Math.Exp(Gaussian.GetLogProb(y, 0, 1));
            }
            else
            {
                double diff = (x - r * y) / System.Math.Sqrt(omr2);
                return System.Math.Exp(MMath.GammaLn(n + 1) + Gaussian.GetLogProb(y, 0, 1)
                    + Gaussian.GetLogProb(diff, 0, 1)
                    + 0.5 * n * System.Math.Log(omr2)) * MMath.NormalCdfMomentRatio(n, diff);
            }
        }

        public static double NormalCdfMomentDyRatio(int n, double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            if (omr2 == 0)
            {
                throw new ArgumentException();
            }
            else
            {
                double diff = (x - r * y) / System.Math.Sqrt(omr2);
                //return Math.Exp(MMath.GammaLn(n + 1) + 0.5 * n * Math.Log(omr2)) * MMath.NormalCdfMomentRatio(n, diff);
                return System.Math.Exp(MMath.GammaLn(n + 1) + 0.5 * n * System.Math.Log(omr2) + System.Math.Log(MMath.NormalCdfMomentRatio(n, diff)));
            }
        }

        public static double NormalCdfDx(double x, double y, double r)
        {
            double omr2 = 1 - r * r;
            if (omr2 == 0)
                return NormalCdfMomentDy(0, y, x, r);
            else
                return System.Math.Exp(Gaussian.GetLogProb(x, 0, 1)
                    + MMath.NormalCdfLn((y - r * x) / System.Math.Sqrt(omr2)));
        }

#if false
        public void NormalCdfMomentRatioSequenceTest()
        {
            int n = 0;
            double x = -1.1;
            var iter = MMath.NormalCdfMomentRatioSequence(n,x);
            for (int i = 0; i < 20; i++)
            {
                iter.MoveNext();
                double r1 = MMath.NormalCdfMomentRatio(n+i,x);
                double r2 = iter.Current;
                Console.WriteLine("{0}: {1} {2} {3}", i, r1, r2, Math.Abs(r1-r2)/Math.Abs(r1));
            }
        }
#endif

        public static IEnumerator<double> NormalCdfMomentRatioSequence(double x)
        {
            if (x > -1)
            {
                double rPrev = MMath.NormalCdfRatio(x);
                yield return rPrev;
                double r = x * rPrev + 1;
                yield return r;
                for (int i = 1; ; i++)
                {
                    double rNew = (x * r + rPrev) / (i + 1);
                    rPrev = r;
                    r = rNew;
                    yield return r;
                }
            }
            else
            {
                int tableSize = 10;
                // rtable[tableStart-i] = R_i
                double[] rtable = new double[tableSize];
                int tableStart = -1;
                for (int i = 0; ; i++)
                {
                    if (i > tableStart)
                    {
                        // build the table
                        tableStart = i + tableSize - 1;
                        rtable[0] = MMath.NormalCdfMomentRatio(tableStart, x);
                        rtable[1] = MMath.NormalCdfMomentRatio(tableStart - 1, x);
                        for (int j = 2; j < tableSize; j++)
                        {
                            int n = tableStart - j + 1;
                            rtable[j] = (n + 1) * rtable[j - 2] - x * rtable[j - 1];
                        }
                    }
                    yield return rtable[tableStart - i];
                }
            }
        }

        // r psi_0
        public static double NormalCdfConFrac5(double x, double y, double r)
        {
            //if (r * (y - r * x) < 0)
            //    throw new ArgumentException("r*(y - r*x) < 0");
            //if (x - r * y > 0)
            //    throw new ArgumentException("x - r*y > 0");
            double omr2 = 1 - r * r;
            double sqrtomr2 = System.Math.Sqrt(omr2);
            //double scale = Math.Exp(Gaussian.GetLogProb(x, 0, 1) + Gaussian.GetLogProb(y, r * x, omr2));
            double rxmy = r * x - y;
            double diff = (x - r * y) / sqrtomr2;
            double logProbDiff = Gaussian.GetLogProb(diff, 0, 1);
            double logScale = Gaussian.GetLogProb(y, 0, 1) + logProbDiff;
            double scale = System.Math.Exp(logScale) * omr2;
            if (scale == 0)
                return scale;
            double omsomr2 = MMath.OneMinusSqrtOneMinus(r * r);
            double delta = (r * y - x * omsomr2) / sqrtomr2;
            var RdiffIter = NormalCdfMomentRatioSequence(diff);
            RdiffIter.MoveNext();
            //double Rdiff = MMath.NormalCdfRatio(diff);
            double Rdiff = RdiffIter.Current;
            double Rx = MMath.NormalCdfRatio(x);
            double offset = Rx * MMath.NormalCdfRatio(-rxmy / sqrtomr2) * scale / omr2;
            double numer;
            if (System.Math.Abs(delta) > 0.5)
                // for r =approx 0 this becomes inaccurate due to cancellation
                numer = scale * (Rx / sqrtomr2 - Rdiff);
            else
                numer = scale * (MMath.NormalCdfRatioDiff(diff, delta) + omsomr2 * Rdiff) / sqrtomr2;
            double numerPrev = 0;
            double denom = rxmy;
            double denomPrev = 1;
            double rOld = 0;
            double result = 0;
            double cEven = scale;
            double cOdd = cEven * r * sqrtomr2;
            double cIncr = r * r * omr2;
            for (int i = 1; i < 1000; i++)
            {
                double numerNew, denomNew;
                double c;
                if (i % 2 == 1)
                {
                    if (i > 1)
                        cOdd *= (i - 1) * cIncr;
                    c = cOdd;
                }
                else
                {
                    cEven *= i * cIncr;
                    c = cEven;
                }
                RdiffIter.MoveNext();
                c *= RdiffIter.Current;
                if (i % 2 == 1)
                {
                    numerNew = rxmy * numer + omr2 * numerPrev - c;
                    denomNew = rxmy * denom + omr2 * denomPrev;
                }
                else
                {
                    numerNew = (rxmy * numer + omr2 * i * numerPrev - c) / (i + 1);
                    denomNew = (rxmy * denom + omr2 * i * denomPrev) / (i + 1);
                }
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                if (i % 2 == 1)
                {
                    result = -numer / denom;
                    Console.WriteLine($"iter {i}: {result:r} {c:g4}");
                    if (double.IsInfinity(result) || double.IsNaN(result))
                        throw new Exception($"NormalCdfConFrac5 not converging for x={x} y={y} r={r}");
                    if (result == rOld)
                    {
                        return offset + result;
                    }
                    rOld = result;
                }
            }
            throw new Exception($"NormalCdfConFrac5 not converging for x={x} y={y} r={r}");
        }

        // r psi_0
        public static double NormalCdfConFrac4(double x, double y, double r)
        {
            if (r * (y - r * x) < 0)
                throw new ArgumentException("r*(y - r*x) < 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            double omr2 = 1 - r * r;
            double rxmy = r * x - y;
            double numer = omr2 * (System.Math.Exp(MMath.NormalCdfLn(x) + Gaussian.GetLogProb(y, r * x, omr2)) - NormalCdfMomentDy(0, x, y, r));
            double numerPrev = 0;
            double denom = rxmy;
            double denomPrev = 1;
            double rOld = 0;
            double result = 0;
            for (int i = 1; i < 1000; i++)
            {
                double numerNew = rxmy * numer + omr2 * (i * numerPrev - System.Math.Pow(r, i) * NormalCdfMomentDy(i, x, y, r));
                double denomNew = rxmy * denom + omr2 * i * denomPrev;
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                result = -numer / denom;
                Console.WriteLine("iter {0}: {1}", i, result.ToString("r"));
                if (double.IsInfinity(result) || double.IsNaN(result))
                    throw new Exception(string.Format("NormalCdfConFrac4 not converging for x={0} y={1} r={2}", x, y, r));
                if (result == rOld)
                    return result;
                rOld = result;
            }
            return result;
            throw new Exception("not converging");
        }

        // requires x < 0, r <= 0, and x-r*y <= 0 (or equivalently y < -x).
        public static double NormalCdfConFrac3(double x, double y, double r)
        {
            if (x > 0)
                throw new ArgumentException("x >= 0");
            if (r > 0)
                throw new ArgumentException("r > 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            double numer = NormalCdfDx(x, y, r) + r * NormalCdfMomentDy(0, x, y, r);
            double numerPrev = 0;
            double denom = x;
            double denomPrev = 1;
            double rprev = 0;
            for (int i = 1; i < 1000; i++)
            {
                double numerNew = x * numer + i * numerPrev + r * NormalCdfMomentDy(i, x, y, r);
                double denomNew = x * denom + i * denomPrev;
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                if (i % 2 == 1)
                {
                    //Console.WriteLine("denom/dfact = {0}", denom / dfact);
                    double result = -numer / denom;
                    Console.WriteLine("iter {0}: {1}", i, result);
                    if (double.IsInfinity(result) || double.IsNaN(result))
                        throw new Exception();
                    if (MMath.AreEqual(result, rprev))
                        return result;
                    rprev = result;
                }
            }
            throw new Exception(string.Format("NormalCdfConFrac3 not converging for x={0} y={1} r={2}", x, y, r));
        }


        // Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale
        // This version works best for small r^2
        // We need x <= 0 and (y - r*x) <= 0
        private static double NormalCdfRatioConFrac3b(double x, double y, double r, double scale)
        {
            if (scale == 0)
                return scale;
            //if (r * (y - r * x) < 0)
            //    throw new ArgumentException("r*(y - r*x) < 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            if (x > 0)
                throw new ArgumentException("x > 0");
            double omr2 = 1 - r * r;
            double sqrtomr2 = System.Math.Sqrt(omr2);
            double rxmy = r * x - y;
            double ymrx = -rxmy / sqrtomr2;
            if (ymrx > 0)
                throw new ArgumentException("ymrx > 0");
            double offset = MMath.NormalCdfRatio(x) * MMath.NormalCdfRatio(ymrx) * scale;
            double omsomr2 = MMath.OneMinusSqrtOneMinus(r * r);
            double delta = (r * y - x * omsomr2) / sqrtomr2;
            double diff = (x - r * y) / sqrtomr2;
            double Rdiff = MMath.NormalCdfRatio(diff);
            //var RdiffIter = MMath.NormalCdfMomentRatioSequence(0, diff);
            //RdiffIter.MoveNext();
            //double Rdiff = RdiffIter.Current;
            double scale2 = scale * omr2;
            double numer;
            if (System.Math.Abs(delta) > 0.5)
                // for r =approx 0 this becomes inaccurate due to cancellation
                numer = scale2 * (MMath.NormalCdfRatio(x) / sqrtomr2 - Rdiff);
            else
                numer = scale2 * (MMath.NormalCdfRatioDiff(diff, delta) + omsomr2 * Rdiff) / sqrtomr2;
            double numerPrev = 0;
            double denom = rxmy;
            double denomPrev = 1;
            double rOld = 0;
            double result = 0;
            double cEven = scale2;
            double cIncr = r * sqrtomr2;
            double cOdd = cEven * cIncr;
            cIncr *= cIncr;
            for (int i = 1; i < 10000; i++)
            {
                double numerNew, denomNew;
                //RdiffIter.MoveNext();
                double c = MMath.NormalCdfMomentRatio(i, diff);
                //double c = RdiffIter.Current;
                if (i % 2 == 1)
                {
                    if (i > 1)
                        cOdd *= (i - 1) * cIncr;
                    c *= cOdd;
                    numerNew = rxmy * numer + omr2 * numerPrev - c;
                    denomNew = rxmy * denom + omr2 * denomPrev;
                }
                else
                {
                    cEven *= i * cIncr;
                    c *= cEven;
                    numerNew = (rxmy * numer + omr2 * i * numerPrev - c) / (i + 1);
                    denomNew = (rxmy * denom + omr2 * i * denomPrev) / (i + 1);
                }
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                if (i % 2 == 1)
                {
                    result = -numer / denom;
                    Console.WriteLine($"iter {i.ToString().PadLeft(3)}: {result.ToString("r").PadRight(24)} {numer.ToString("r").PadRight(24)} {denom.ToString("r").PadRight(24)} {c}");
                    if (double.IsInfinity(result) || double.IsNaN(result))
                        throw new Exception(string.Format("NormalCdfRatioConFrac3 not converging for x={0} y={1} r={2} scale={3}", x, y, r, scale));
                    //if (result == rOld)
                    //    return result + offset;
                    rOld = result;
                }
            }
            throw new Exception(string.Format("NormalCdfRatioConFrac3 not converging for x={0} y={1} r={2} scale={3}", x, y, r, scale));
        }

        /// <summary>
        /// Computes the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.  Must be finite.</param>
        /// <param name="y">Second upper limit.  Must be finite.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <returns><c>phi(x,y,r)</c></returns>
        /// <remarks>
        /// The double integral is transformed into a single integral which is approximated by quadrature.
        /// Reference: 
        /// "Numerical Computation of Rectangular Bivariate and Trivariate Normal and t Probabilities"
        /// Alan Genz, Statistics and Computing, 14 (2004), pp. 151-160
        /// http://www.math.wsu.edu/faculty/genz/genzhome/research.html
        /// </remarks>
        public static double NormalCdf_Quadrature(double x, double y, double r)
        {
            double absr = System.Math.Abs(r);
            Vector nodes, weights;
            int count = 20;
            if (absr < 0.3)
                count = 6;
            else if (absr < 0.75)
                count = 12;
            nodes = Vector.Zero(count);
            weights = Vector.Zero(count);
            double result = 0.0;
            if (absr < 0.925)
            {
                // use equation (3)
                double asinr = System.Math.Asin(r);
                Quadrature.UniformNodesAndWeights(0, asinr, nodes, weights);
                double sq = 0.5 * (x * x + y * y), xy = x * y;
                for (int i = 0; i < nodes.Count; i++)
                {
                    double sin = System.Math.Sin(nodes[i]);
                    double cos2 = 1 - sin * sin;
                    result += weights[i] * System.Math.Exp((xy * sin - sq) / cos2);
                }
                result /= 2 * System.Math.PI;
                result += MMath.NormalCdf(x, y, 0);
            }
            else
            {
                double sy = (r < 0) ? -y : y;
                if (absr < 1)
                {
                    // use equation (6) modified by (7)
                    // quadrature part
                    double cos2asinr = (1 - r) * (1 + r), sqrt1mrr = System.Math.Sqrt(cos2asinr);
                    Quadrature.UniformNodesAndWeights(0, sqrt1mrr, nodes, weights);
                    double sxy = x * sy;
                    double diff2 = (x - sy) * (x - sy);
                    double c = (4 - sxy) / 8, d = (12 - sxy) / 16;
                    for (int i = 0; i < nodes.Count; i++)
                    {
                        double cos2 = nodes[i] * nodes[i];
                        double sin = System.Math.Sqrt(1 - cos2);
                        double series = 1 + c * cos2 * (1 + d * cos2);
                        double exponent = -0.5 * (diff2 / cos2 + sxy);
                        double f = System.Math.Exp(-0.5 * sxy * (1 - sin) / (1 + sin)) / sin;
                        result += weights[i] * System.Math.Exp(exponent) * (f - series);
                    }
                    // Taylor expansion part
                    double exponentr = -0.5 * (diff2 / cos2asinr + sxy);
                    double absdiff = System.Math.Sqrt(diff2);
                    if (exponentr > -800)
                    {
                        // avoid 0*Inf problems
                        result += sqrt1mrr * System.Math.Exp(exponentr) * (1 - c * (diff2 - cos2asinr) * (1 - d * diff2 / 5) / 3 + c * d * cos2asinr * cos2asinr / 5);
                        // for large absdiff, NormalCdfLn(-absdiff / sqrt1mrr) =approx -0.5*diff2/cos2asinr
                        // so (-0.5*sxy + NormalCdfLn) =approx exponentr
                        result -= System.Math.Exp(-0.5 * sxy + MMath.NormalCdfLn(-absdiff / sqrt1mrr)) * absdiff * (1 - c * diff2 * (1 - d * diff2 / 5) / 3) * MMath.Sqrt2PI;
                    }
                    result /= -2 * System.Math.PI;
                }
                if (r > 0)
                {
                    // exact value for r=1
                    result += MMath.NormalCdf(x, y, 1);
                }
                else
                {
                    // exact value for r=-1
                    result = -result;
                    result += MMath.NormalCdf(x, y, -1);
                }
            }
            if (result < 0)
                result = 0.0;
            else if (result > 1)
                result = 1.0;
            return result;
        }

        private static double NormalCdfLn_Quadrature(double x, double y, double r)
        {
            double absr = System.Math.Abs(r);
            Vector nodes, weights;
            int count = 20;
            if (absr < 0.3)
                count = 6;
            else if (absr < 0.75)
                count = 12;
            nodes = Vector.Zero(count);
            weights = Vector.Zero(count);
            // hasInfiniteLimit is true if NormalCdf(x,y,-1) is 0
            bool hasInfiniteLimit = false;
            if (r < -0.5)
            {
                if (x > 0)
                {
                    // NormalCdf(y) <= NormalCdf(-x)  iff y <= -x
                    if (y < 0)
                        hasInfiniteLimit = (y <= -x);
                }
                else
                {
                    // NormalCdf(x) <= NormalCdf(-y) iff x <= -y
                    if (y > 0)
                        hasInfiniteLimit = (x <= -y);
                    else
                        hasInfiniteLimit = true;
                }
            }
            if (absr < 0.925 && !hasInfiniteLimit)
            {
                // use equation (3)
                double asinr = System.Math.Asin(r);
                Quadrature.UniformNodesAndWeights(0, asinr, nodes, weights);
                double sq = 0.5 * (x * x + y * y), xy = x * y;
                double logResult = double.NegativeInfinity;
                bool useLogWeights = true;
                if (useLogWeights)
                {
                    for (int i = 0; i < nodes.Count; i++)
                    {
                        double sin = System.Math.Sin(nodes[i]);
                        double cos2 = 1 - sin * sin;
                        logResult = MMath.LogSumExp(logResult, System.Math.Log(System.Math.Abs(weights[i])) + (xy * sin - sq) / cos2);
                    }
                    logResult -= 2 * MMath.LnSqrt2PI;
                }
                else
                {
                    double result = 0.0;
                    for (int i = 0; i < nodes.Count; i++)
                    {
                        double sin = System.Math.Sin(nodes[i]);
                        double cos2 = 1 - sin * sin;
                        result += weights[i] * System.Math.Exp((xy * sin - sq) / cos2);
                    }
                    result /= 2 * System.Math.PI;
                    logResult = System.Math.Log(System.Math.Abs(result));
                }
                double r0 = MMath.NormalCdfLn(x, y, 0);
                if (asinr > 0)
                    return MMath.LogSumExp(r0, logResult);
                else
                    return MMath.LogDifferenceOfExp(r0, logResult);
            }
            else
            {
                double result = 0.0;
                double sy = (r < 0) ? -y : y;
                if (absr < 1)
                {
                    // use equation (6) modified by (7)
                    // quadrature part
                    double cos2asinr = (1 - r) * (1 + r), sqrt1mrr = System.Math.Sqrt(cos2asinr);
                    Quadrature.UniformNodesAndWeights(0, sqrt1mrr, nodes, weights);
                    double sxy = x * sy;
                    double diff2 = (x - sy) * (x - sy);
                    double c = (4 - sxy) / 8, d = (12 - sxy) / 16;
                    for (int i = 0; i < nodes.Count; i++)
                    {
                        double cos2 = nodes[i] * nodes[i];
                        double sin = System.Math.Sqrt(1 - cos2);
                        double series = 1 + c * cos2 * (1 + d * cos2);
                        double exponent = -0.5 * (diff2 / cos2 + sxy);
                        double f = System.Math.Exp(-0.5 * sxy * (1 - sin) / (1 + sin)) / sin;
                        result += weights[i] * System.Math.Exp(exponent) * (f - series);
                    }
                    // Taylor expansion part
                    double exponentr = -0.5 * (diff2 / cos2asinr + sxy);
                    double absdiff = System.Math.Sqrt(diff2);
                    if (exponentr > -800)
                    {
                        double taylor = sqrt1mrr * (1 - c * (diff2 - cos2asinr) * (1 - d * diff2 / 5) / 3 + c * d * cos2asinr * cos2asinr / 5);
                        // avoid 0*Inf problems
                        //result -= Math.Exp(-0.5*sxy + NormalCdfLn(-absdiff/sqrt1mrr))*absdiff*(1 - c*diff2*(1 - d*diff2/5)/3)*Sqrt2PI;
                        taylor -= MMath.NormalCdfRatio(-absdiff / sqrt1mrr) * absdiff * (1 - c * diff2 * (1 - d * diff2 / 5) / 3);
                        result += System.Math.Exp(exponentr) * taylor;
                    }
                    result /= -2 * System.Math.PI;
                }
                if (r > 0)
                {
                    // result += NormalCdf(x, y, 1);
                    double r1 = MMath.NormalCdfLn(x, y, 1);
                    if (result > 0)
                    {
                        result = System.Math.Log(result);
                        return MMath.LogSumExp(result, r1);
                    }
                    else
                    {
                        return MMath.LogDifferenceOfExp(r1, System.Math.Log(-result));
                    }
                }
                else
                {
                    // return NormalCdf(x, y, -1) - result;
                    double r1 = MMath.NormalCdfLn(x, y, -1);
                    if (result > 0)
                    {
                        return MMath.LogDifferenceOfExp(r1, System.Math.Log(result));
                    }
                    else
                    {
                        return MMath.LogSumExp(r1, System.Math.Log(-result));
                    }
                }
            }
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

        /// <summary>
        /// Tests the BetaCDF calculation. The test points are designed to get good coverage and to exercise
        /// every condition in BetaCDF.
        /// </summary>
        [Fact]
        public void BetaCdfTest()
        {
            // This ground truth comes from R. Each row is of the form:
            // (x, a, b, pbeta(x, a, b))
            double[,] groundTruth = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "BetaCdf.csv"));

            MathFcn3 fun = new MathFcn3((x, a, b) => (new Beta(a, b)).GetProbLessThan(x));
            CheckFunctionValues("BetaCdf", fun, groundTruth, 1e-9);
        }

        [Fact]
        public void UlpTest()
        {
            double[,] Ulp_pairs = ReadPairs(Path.Combine("Data", "SpecialFunctionsValues", "ulp.csv"));
            CheckFunctionValues("ulp", MMath.Ulp, Ulp_pairs);
        }

        [Fact]
        public void MedianTest()
        {
            Assert.Equal(double.MaxValue, MMath.Median(new[] { double.MaxValue }));
            Assert.Equal(double.MaxValue, MMath.Median(new[] { double.MaxValue, double.MaxValue }));
            Assert.Equal(3, MMath.Median(new[] { double.MaxValue, 3, double.MinValue }));
        }

        private static double[,] ReadPairs(string filepath)
        {
            string[] lines = File.ReadAllLines(filepath);
            int argsNo = lines[0].Count(c => c == ',') / 2;
            int length = lines.Length - 1;
            int width = argsNo + 1;
            double[,] pairs = new double[length, width];
            var cells = lines.Skip(1).SelectMany(s => s.Split(',').Skip(argsNo).Select(c => double.Parse(c, System.Globalization.CultureInfo.InvariantCulture))).GetEnumerator();
            for (int i = 0; i < length; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    cells.MoveNext();
                    pairs[i, j] = cells.Current;
                }
            }
            return pairs;
        }

        private static void WritePairs(double[,] pairs, string folder, string name)
        {
            using (var file = new StreamWriter(Path.Combine(folder, $"{name}.csv"), false))
            {
                var len = pairs.GetLength(0);
                var width = pairs.GetLength(1);
                var argsNo = width - 1;

                for (int i = 0; i < argsNo; i++)
                    file.Write($"arg{i}rounded,");
                for (int i = 0; i < argsNo; i++)
                    file.Write($"arg{i}exact,");
                file.WriteLine("expectedresult");

                var jagged = new double[len][];
                for (int i = 0; i < len; ++i)
                {
                    jagged[i] = new double[width];
                    for (int j = 0; j < width; ++j)
                        jagged[i][j] = pairs[i, j];
                }
                Array.Sort(jagged, new ArrayComparer());

                for (int i = 0; i < len; ++i)
                {
                    for (int j = 0; j < argsNo; ++j)
                        file.Write($"{jagged[i][j].ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                    for (int j = 0; j < argsNo; ++j)
                        file.Write($"{ToStringExact(jagged[i][j])},");
                    file.WriteLine(ToStringExact(jagged[i][width - 1]));
                }
            }
        }

        private static void CheckFunctionValues(string name, MathFcn fcn, double[,] pairs, double assertTolerance = 1e-11)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private static void CheckFunctionValues(string name, MathFcn2 fcn, double[,] pairs, double assertTolerance = 1e-11)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private static void CheckFunctionValues(string name, MathFcn3 fcn, double[,] pairs, double assertTolerance = 1e-11)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private static void CheckFunctionValues(string name, MathFcn4 fcn, double[,] pairs, double assertTolerance = 1e-11)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private class ArrayComparer : IComparer<double[]>
        {
            public int Compare(double[] x, double[] y)
            {
                var len = System.Math.Min(x.Length, y.Length);
                for (int i = 0; i< len; ++i)
                {
                    if (x[i] < y[i])
                        return -1;
                    if (x[i] > y[i])
                        return 1;

                    // Making NaNs the largest
                    var xiIsNaN = double.IsNaN(x[i]);
                    var yiIsNaN = double.IsNaN(y[i]);
                    if (xiIsNaN && !yiIsNaN)
                        return 1;
                    if (!xiIsNaN && yiIsNaN)
                        return -1;
                }
                return 0;
            }
        }

        private static void CheckFunctionValues(string name, Delegate fcn, double[,] pairs, double assertTolerance)
        {
            Vector x = Vector.Zero(pairs.GetLength(1) - 1);
            object[] args = new object[x.Count];
            for (int i = 0; i < pairs.GetLength(0); i++)
            {
                for (int k = 0; k < x.Count; k++)
                {
                    x[k] = pairs[i, k];
                    args[k] = x[k];
                }
                bool showTiming = false;
                if (showTiming)
                {
                    Stopwatch watch = Stopwatch.StartNew();
                    int repetitionCount = 100000;
                    for (int repetition = 0; repetition < repetitionCount; repetition++)
                    {
                        Util.DynamicInvoke(fcn, args);
                    }
                    watch.Stop();
                    Trace.WriteLine($"  ({watch.ElapsedTicks} ticks for {repetitionCount} calls)");
                }
                double fx = pairs[i, x.Count];
                double result = (double)Util.DynamicInvoke(fcn, args);
                if (!double.IsNaN(result) && System.Math.Sign(result) != System.Math.Sign(fx) && fx != 0)
                {
                    string strMsg = $"{name}({x:r})\t has wrong sign (result = {result:r})";
                    Trace.WriteLine(strMsg);
                    Assert.True(false, strMsg);
                }
                double err = System.Math.Abs(result - fx);
                if (fx != 0)
                    err /= System.Math.Abs(fx);
                if (Double.IsInfinity(fx))
                {
                    err = (result == fx) ? 0 : Double.PositiveInfinity;
                }
                if (Double.IsNaN(fx))
                {
                    err = Double.IsNaN(result) ? 0 : Double.NaN;
                }
                if (!IsErrorSignificant(TOLERANCE, err))
                {
                    Trace.WriteLine($"{name}({x:r})\t ok");
                }
                else
                {
                    string strMsg = $"{name}({x:r})\t wrong by {err.ToString("g2")} (result = {result:r})";
                    Trace.WriteLine(strMsg);
                    if (IsErrorSignificant(assertTolerance, err) || double.IsNaN(err))
                        Assert.True(false, strMsg);
                }
            }
        }

        public static bool IsErrorSignificant(double assertTolerance, double err)
        {
            return err > assertTolerance;
        }
    }
}