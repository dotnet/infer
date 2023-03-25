// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{

    public class SpecialFunctionsTests
    {
        public const double TOLERANCE = 1e-15;
        const double defaultAssertTolerance = 1e-11;

        public delegate double MathFcn(double arg);
        public delegate double MathFcn2(double arg1, double arg2);
        public delegate double MathFcn3(double arg1, double arg2, double arg3);
        public delegate double MathFcn4(double arg1, double arg2, double arg3, double arg4);

        private const string PairFileArgColumnNamePrefix = "arg";
        private const string PairFileExpectedValueColumnName = "expectedresult";
        private const char PairFileSeparator = ',';

        public struct Test
        {
            public MathFcn fcn;
            public double[,] pairs;
        };

        [Fact]
        public void GammaSpecialFunctionsTest()
        {
            double[,] BesselI_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "BesselI.csv"));
            CheckFunctionValues("BesselI", MMath.BesselI, BesselI_pairs);

            double[,] Gamma_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Gamma.csv"));
            CheckFunctionValues("Gamma", MMath.Gamma, Gamma_pairs);

            double[,] GammaLn_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "GammaLn.csv"));
            CheckFunctionValues("GammaLn", new MathFcn(MMath.GammaLn), GammaLn_pairs);

            double[,] GammaLnSeries_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "GammaLnSeries.csv"));
            CheckFunctionValues("GammaLnSeries", MMath.GammaLnSeries, GammaLnSeries_pairs);

            double[,] digamma_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Digamma.csv"));
            CheckFunctionValues("Digamma", new MathFcn(MMath.Digamma), digamma_pairs);

            double[,] trigamma_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Trigamma.csv"));
            CheckFunctionValues("Trigamma", new MathFcn(MMath.Trigamma), trigamma_pairs);

            double[,] tetragamma_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Tetragamma.csv"));
            CheckFunctionValues("Tetragamma", MMath.Tetragamma, tetragamma_pairs);
        }

        [Fact]
        public void SpecialFunctionsTest()
        {
            double[,] logistic_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Logistic.csv"));
            CheckFunctionValues("Logistic", MMath.Logistic, logistic_pairs);

            double[,] logisticln_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "LogisticLn.csv"));
            CheckFunctionValues("LogisticLn", MMath.LogisticLn, logisticln_pairs);

            double[,] log1plus_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Log1Plus.csv"));
            CheckFunctionValues("Log1Plus", MMath.Log1Plus, log1plus_pairs);


            double[,] xMinusLog1Plus_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "XMinusLog1Plus.csv"));
            CheckFunctionValues("XMinusLog1Plus", MMath.XMinusLog1Plus, xMinusLog1Plus_pairs);
            double[,] log1MinusExp_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "Log1MinusExp.csv"));
            CheckFunctionValues("Log1MinusExp", MMath.Log1MinusExp, log1MinusExp_pairs);

            double[,] expminus1_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "ExpMinus1.csv"));
            CheckFunctionValues("ExpMinus1", MMath.ExpMinus1, expminus1_pairs);

            double[,] expMinus1RatioMinus1RatioMinusHalf_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "ExpMinus1RatioMinus1RatioMinusHalf.csv"));
            CheckFunctionValues("ExpMinus1RatioMinus1RatioMinusHalf", MMath.ExpMinus1RatioMinus1RatioMinusHalf, expMinus1RatioMinus1RatioMinusHalf_pairs);

            double[,] logexpminus1_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "LogExpMinus1.csv"));
            CheckFunctionValues("LogExpMinus1", MMath.LogExpMinus1, logexpminus1_pairs);

            double[,] logsumexp_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "LogSumExp.csv"));
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
            double[,] normcdf_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdf.csv"));
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

            double[,] normcdfln_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfLn.csv"));
            CheckFunctionValues("NormalCdfLn", new MathFcn(MMath.NormalCdfLn), normcdfln_pairs);

            double[,] normcdflogit_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfLogit.csv"));
            CheckFunctionValues("NormalCdfLogit", new MathFcn(MMath.NormalCdfLogit), normcdflogit_pairs);
        }

        [Fact]
        public void LogisticGaussianTest()
        {
            double[,] logisticGaussian_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "logisticGaussian.csv"));
            CheckFunctionValues("logisticGaussian", new MathFcn2(MMath.LogisticGaussian), logisticGaussian_pairs);

            double[,] logisticDerivGaussian_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "logisticGaussianDeriv.csv"));
            CheckFunctionValues("logisticGaussianDeriv", MMath.LogisticGaussianDerivative, logisticDerivGaussian_pairs);

            double[,] logisticDeriv2Gaussian_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "logisticGaussianDeriv2.csv"));
            CheckFunctionValues("logisticGaussianDeriv2", MMath.LogisticGaussianDerivative2, logisticDeriv2Gaussian_pairs);
        }

        [Fact]
        public void GammaUpperTest()
        {
            double[,] gammaUpperScale_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "GammaUpperScale.csv"));
            CheckFunctionValues(nameof(MMath.GammaUpperScale), MMath.GammaUpperScale, gammaUpperScale_pairs);

            double[,] gammaLower_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "GammaLower.csv"));
            CheckFunctionValues(nameof(MMath.GammaLower), MMath.GammaLower, gammaLower_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
gammainc(mpf('1'),mpf('1'),mpf('inf'),regularized=True)
            */
            double[,] gammaUpperRegularized_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "GammaUpperRegularized.csv"));
            CheckFunctionValues("GammaUpperRegularized", (a, x) => MMath.GammaUpper(a, x, true), gammaUpperRegularized_pairs);

            /* In python mpmath:
from mpmath import *
mp.dps = 500
mp.pretty = True
gammainc(mpf('1'),mpf('1'),mpf('inf'),regularized=False)
            */
            double[,] gammaUpper_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "GammaUpper.csv"));
            CheckFunctionValues("GammaUpper", (a, x) => MMath.GammaUpper(a, x, false), gammaUpper_pairs);
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
            double[,] normcdfMomentRatio_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfMomentRatio.csv"));
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

        [Fact]
        public void NormalCdf2Test()
        {
            double x = 1.2, y = 1.3;
            double xy1 = System.Math.Max(0.0, MMath.NormalCdf(x) + MMath.NormalCdf(y) - 1);

            Assert.True(0 < MMath.NormalCdf(6.8419544775976187E-08, -5.2647906596206016E-08, -1, 3.1873689658872377E-10).Mantissa);

            // In sage:
            // integral(1/(2*pi*sqrt(1-t*t))*exp(-(x*x+y*y-2*t*x*y)/(2*(1-t*t))),t,-1,r).n(digits=200);
            double[,] normalcdf2_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdf2.csv"));
            CheckFunctionValues("NormalCdf2", new MathFcn3(MMath.NormalCdf), normalcdf2_pairs);

            // using wolfram alpha:  (for cases where r=-1 returns zero)
            // log(integrate(1/(2*pi*sqrt(1-t*t))*exp(-(2*0.1*0.1 - 2*t*0.1*0.1)/(2*(1-t*t))),t,-1,-0.9999))
            // log(integrate(1/(2*pi*sqrt(1-t*t))*exp(-(2*1.5*1.5 + 2*t*1.5*1.5)/(2*(1-t*t))),t,-1,-0.5))
            double[,] normalcdfln2_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfLn2.csv"));
            CheckFunctionValues("NormalCdfLn2", new MathFcn3(MMath.NormalCdfLn), normalcdfln2_pairs);

            double[,] normalcdfRatioLn2_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfRatioLn2.csv"));
            CheckFunctionValues("NormalCdfRatioLn2", new MathFcn4(NormalCdfRatioLn), normalcdfRatioLn2_pairs);

            // The true values are computed using
            // x * MMath.NormalCdf(x, y, r) + System.Math.Exp(Gaussian.GetLogProb(x, 0, 1) + MMath.NormalCdfLn(ymrx)) + r * System.Math.Exp(Gaussian.GetLogProb(y, 0, 1) + MMath.NormalCdfLn(xmry))
            // where
            // ymrx = (y - r * x) / sqrt(1-r*r)
            // xmry = (x - r * y) / sqrt(1-r*r)
            double[,] normalcdfIntegral_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfIntegral.csv"));
            CheckFunctionValues("NormalCdfIntegral", new MathFcn3(MMath.NormalCdfIntegral), normalcdfIntegral_pairs);

            double[,] normalcdfIntegralRatio_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "NormalCdfIntegralRatio.csv"));
            CheckFunctionValues("NormalCdfIntegralRatio", new MathFcn3(MMath.NormalCdfIntegralRatio), normalcdfIntegralRatio_pairs);
        }

        // Same as MMath.NormalCdfRatioLn but avoids inconsistent values of r and sqrtomr2 when using arbitrary precision.
        private static double NormalCdfRatioLn(double x, double y, double r, double sqrtomr2)
        {
            if (sqrtomr2 < 0.618)
            {
                // In this regime, it is more accurate to compute r from sqrtomr2.
                // Proof:  
                // In the presence of roundoff, sqrt(1 - sqrtomr2^2) becomes 
                // sqrt(1 - sqrtomr2^2*(1+eps)) 
                // = sqrt(1 - sqrtomr2^2 - sqrtomr2^2*eps)
                // =approx sqrt(1 - sqrtomr2^2) - sqrtomr2^2*eps*0.5/sqrt(1-sqrtomr2^2)
                // The error is below machine precision when
                // sqrtomr2^2/sqrt(1-sqrtomr2^2) < 1
                // which is equivalent to sqrtomr2 < (sqrt(5)-1)/2 =approx 0.618
                double omr2 = sqrtomr2 * sqrtomr2;
                r = System.Math.Sign(r) * System.Math.Sqrt(1 - omr2);
            }
            else
            {
                // In this regime, it is more accurate to compute sqrtomr2 from r.
                double omr2 = 1 - r * r;
                sqrtomr2 = System.Math.Sqrt(omr2);
            }
            return MMath.NormalCdfRatioLn(x, y, r, sqrtomr2);
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

        /// <summary>
        /// Tests the BetaCDF calculation. The test points are designed to get good coverage and to exercise
        /// every condition in BetaCDF.
        /// </summary>
        [Fact]
        public void BetaCdfTest()
        {
            // This ground truth comes from R. Each row is of the form:
            // (x, a, b, pbeta(x, a, b))
            double[,] groundTruth = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "BetaCdf.csv"));

            MathFcn3 fun = new MathFcn3((x, a, b) => (new Beta(a, b)).GetProbLessThan(x));
            CheckFunctionValues("BetaCdf", fun, groundTruth, 1e-9);
        }

        [Fact]
        public void UlpTest()
        {
            double[,] Ulp_pairs = ReadPairs(Path.Combine(TestUtils.DataFolderPath, "SpecialFunctionsValues", "ulp.csv"));
            CheckFunctionValues("ulp", MMath.Ulp, Ulp_pairs);
        }

        /// <summary>
        /// Reads a 2D array of doubles from a csv file.
        /// Checks that the header of the file has the form
        /// arg0,arg1,...,argN,expectedresult
        /// and that the rest of the file conforms to it.
        /// </summary>
        private static double[,] ReadPairs(string filepath)
        {
            string[] lines = File.ReadAllLines(filepath);
            string[] header = lines[0].Split(PairFileSeparator).Select(s => s.Trim()).ToArray();
            for (int i = 0; i < header.Length - 1; ++i)
                if (header[i] != $"{PairFileArgColumnNamePrefix}{i}")
                    throw new InvalidDataException($"The header of the file {filepath} has an incorrect element at position {i}\nExpected: {PairFileArgColumnNamePrefix}{i}\nActual: {header[i]} ");
            if (header[header.Length - 1] != PairFileExpectedValueColumnName)
                throw new InvalidDataException($"The header of the file {filepath} has an incorrect last element\nExpected: {PairFileExpectedValueColumnName}\nActual: {header[header.Length - 1]} ");
            int argsNo = header.Length - 1;
            int length = lines.Length - 1;
            int width = argsNo + 1;
            double[,] pairs = new double[length, width];
            for (int i = 0; i < length; ++i)
            {
                string[] line = lines[i + 1].Split(PairFileSeparator);
                if (line.Length != width)
                    throw new InvalidDataException($"The number of entries on the line {i + 1} of the file {filepath} is inconsistent with the file's header.");
                for (int j = 0; j < width; ++j)
                {
                    if (!DoubleTryParseWithWorkarounds(line[j], out pairs[i, j]))
                        throw new InvalidDataException($"Failed to parse the entry at position {j} on the line {i + 1} of the file {filepath}.");
                }
            }
            return pairs;
        }

        // These get incorrectly rounded when parsed on .NET Framework x64 and .NET Core 2.1
        private static readonly Dictionary<string, double> doubleParsingWorkarounds = new Dictionary<string, double>()
            {
                { "1.0866893952407441142985209453839598005754913416137e-317", 1.0866894829774883E-317 },
                { "1.4752977335476395893266937547311965076180770936655e-317", 1.4752978048452125E-317 },
                { "9.9999999999985400000000001172379999999931533008e+2628", double.PositiveInfinity },
                { "8.7432346725418748345517984418356955202593669773535e+22005185347451822441642474609178574152257326040675780012584300835562556718334196807739458376343563329726769290143343263935463229609569787933711840058557961298006731067257214226525645302856902417504745728399459717322818291964340458508973669264873826129339400419536238302065289447890487684206500524608861", double.PositiveInfinity }
            };

        private static bool DoubleTryParseWithWorkarounds(string s, out double result)
        {
            return doubleParsingWorkarounds.TryGetValue(s, out result)
                || double.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out result);
        }

        /// <summary>
        /// Writes a 2D array of doubles into a csv file using MMath.ToStringExact.
        /// Add a call to this in CheckFunctionValues to save all arguments and expected values
        /// used in tests.
        /// </summary>
        /// <param name="pairs">2D array of doubles to write</param>
        /// <param name="folder">Folder where the resulting csv will be saved</param>
        /// <param name="name">Name of the saved file without extension</param>
        private static void WritePairs(double[,] pairs, string folder, string name)
        {
            using (var file = new StreamWriter(Path.Combine(folder, $"{name}.csv"), false))
            {
                var len = pairs.GetLength(0);
                var width = pairs.GetLength(1);
                var argsNo = width - 1;

                for (int i = 0; i < argsNo; i++)
                    file.Write($"{PairFileArgColumnNamePrefix}{i}{PairFileSeparator}");
                file.WriteLine(PairFileExpectedValueColumnName);

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
                        file.Write($"{MMath.ToStringExact(jagged[i][j])}{PairFileSeparator}");
                    file.WriteLine(MMath.ToStringExact(jagged[i][width - 1]));
                }
            }
        }

        private static void CheckFunctionValues(string name, MathFcn fcn, double[,] pairs, double assertTolerance = defaultAssertTolerance)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private static void CheckFunctionValues(string name, MathFcn2 fcn, double[,] pairs, double assertTolerance = defaultAssertTolerance)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private static void CheckFunctionValues(string name, MathFcn3 fcn, double[,] pairs, double assertTolerance = defaultAssertTolerance)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private static void CheckFunctionValues(string name, MathFcn4 fcn, double[,] pairs, double assertTolerance = defaultAssertTolerance)
        {
            CheckFunctionValues(name, (Delegate)fcn, pairs, assertTolerance);
        }

        private class ArrayComparer : IComparer<double[]>
        {
            public int Compare(double[] x, double[] y)
            {
                var len = System.Math.Min(x.Length, y.Length);
                for (int i = 0; i < len; ++i)
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
                if (!double.IsNaN(result) && System.Math.Sign(result) != System.Math.Sign(fx) && fx != 0 && result != 0)
                {
                    string strMsg = $"{name}({x:g17})\t has wrong sign (result = {result:g17})";
                    Trace.WriteLine(strMsg);
                    Assert.True(false, strMsg);
                }
                double err = MMath.AbsDiff(result, fx, 1e14*double.Epsilon);
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
                    Trace.WriteLine($"{name}({x:g17})\t ok");
                }
                else
                {
                    string strMsg = $"{name}({x:g17})\t wrong by {err.ToString("g2")} (result = {result:g17})";
                    Trace.WriteLine(strMsg);
                    if (IsErrorSignificant(assertTolerance, err) || double.IsNaN(err))
                        Assert.True(false, strMsg);
                }
            }
        }

        public static bool IsErrorSignificant(double assertTolerance, double err)
        {
            return double.IsNaN(err) || err > assertTolerance;
        }
    }
}