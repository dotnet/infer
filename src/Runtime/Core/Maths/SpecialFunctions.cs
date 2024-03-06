// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Math
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Numerics;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions; // for Gaussian.GetLogProb
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// This class provides mathematical constants and special functions, 
    /// analogous to System.Math.
    /// It cannot be instantiated and consists of static members only.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In order to provide the highest accuracy, some routines return their results in log form or logit form.
    /// These transformations expand the domain to cover the full range of double-precision values, ensuring 
    /// all bits of the representation are utilized.  A good example of this is the NormalCdf function, whose 
    /// value lies between 0 and 1.  Numbers between 0 and 1 use only a small fraction of the capacity of a 
    /// double-precision number.  The function NormalCdfLogit transforms the result p according to log(p/(1-p)), 
    /// providing full use of the range from -Infinity to Infinity and (potentially) much higher precision.
    /// </para><para>
    /// To get maximal use out of these transformations, you want to stay in the expanded form as long as 
    /// possible.  Every time you transform into a smaller domain, you lose precision.  Thus helper functions 
    /// are provided which allow you to perform common tasks directly in the log form and logit form. 
    /// For logs, you have addition.  For logit, you have averaging. 
    /// </para>
    /// </remarks>
    public static class MMath
    {
        #region Bessel functions

        /// <summary>
        /// Modified Bessel function of the first kind
        /// </summary>
        /// <param name="a">Order parameter.  Any real number except a negative integer.</param>
        /// <param name="x">Argument of the Bessel function.  Non-negative real number.</param>
        /// <remarks>
        /// Reference:
        /// "A short note on parameter approximation for von Mises-Fisher distributions, And a fast implementation of Is(x)"
        /// Suvrit Sra
        /// Computational Statistics, 2011
        /// http://people.kyb.tuebingen.mpg.de/suvrit/papers/vmfFinal.pdf
        /// </remarks>
        /// <returns>BesselI(a,x)</returns>
        public static double BesselI(double a, double x)
        {
            if (x < 0)
                throw new ArgumentException("x (" + x + ") cannot be negative");
            if (a < 0 && a == Math.Floor(a))
                throw new ArgumentException("Order parameter a=" + a + " cannot be a negative integer");
            // http://functions.wolfram.com/Bessel-TypeFunctions/BesselI/02/
            double xh = 0.5 * x;
            double term = Math.Pow(xh, a) / MMath.Gamma(a + 1);
            double xh2 = xh * xh;
            double sum = 0;
            for (int k = 0; k < 1000; k++)
            {
                double oldsum = sum;
                sum += term;
                term *= xh2 / ((k + 1 + a) * (k + 1));
                if (AreEqual(oldsum, sum))
                {
                    break;
                }
            }
            return sum;
        }

        #endregion

        #region Beta functions
        /// <summary>
        /// Power series for incomplete beta function obtained by replacing the second term in the integrand.
        /// BPSER method described in Didonato and Morris.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="a">The true count for the Beta.</param>
        /// <param name="b">The false count for the Beta.</param>
        /// <param name="epsilon">A tolerance for terminating the series calculation.</param>
        /// <returns></returns>
        private static double BPSer(double x, double a, double b, double epsilon)
        {
            double coeff = 1.0;
            double result = 1.0 / a;

            for (int j = 1; ; j++)
            {
                coeff *= (1.0 - b / j) * x;
                double term = coeff / (a + j);
                result += term;

                if (Math.Abs(term) < epsilon * Math.Abs(result))
                {
                    break;
                }
            }

            return Math.Exp(Math.Log(result) + a * Math.Log(x) - Distributions.Beta.BetaLn(a, b));
        }

        /// <summary>
        /// BUP method described in Didonato and Morris.
        /// </summary>
        /// <param name="x">Value.</param>
        /// <param name="a">True count.</param>
        /// <param name="b">False count.</param>
        /// <param name="n">True count increment.</param>
        /// <param name="epsilon">Tolerance.</param>
        /// <returns></returns>
        private static double BUp(double x, double a, double b, int n, double epsilon)
        {
            double y = 1.0 - x;
            double d = 1.0;
            double xPow = 1.0;
            double sumh = 1.0;

            // i must be greater than iCheck to check for convergence
            int iCheck = 0;
            if (b > 1)
            {
                iCheck = (int)Math.Floor((b - 1) * x / y - a);
            }

            for (int i = 1; i < n; i++)
            {
                int i1 = i - 1;
                d = (a + b + i1) * d / (a + 1 + i1);
                xPow *= x;
                double h = d * xPow;
                sumh += h;

                if (i > iCheck && h <= epsilon * sumh)
                    break;
            }

            return Math.Exp(Math.Log(sumh) + a * Math.Log(x) + b * Math.Log(y) - Math.Log(a) - Distributions.Beta.BetaLn(a, b));
        }

        /// <summary>
        /// BGRAT method described in Didonato and Morris.
        /// </summary>
        /// <param name="x">Value.</param>
        /// <param name="a">True count.</param>
        /// <param name="b">False count.</param>
        /// <param name="epsilon">Tolerance.</param>
        /// <returns></returns>
        private static double BGRat(double x, double a, double b, double epsilon)
        {
            List<double> p = new List<double>();
            List<double> twoNPlus1Factorial = new List<double>();
            p.Add(1.0);
            twoNPlus1Factorial.Add(1);
            double T = a + 0.5 * (b - 1);
            double u = -T * Math.Log(x);
            double lnHOfBU = -u + b * Math.Log(u) - MMath.GammaLn(b);
            double M = Math.Exp(lnHOfBU + MMath.GammaLn(a + b) - MMath.GammaLn(a) - (b * Math.Log(T)));
            double oneOver4TSq = 0.25 / (T * T);
            // Starting point for J
            double J = Math.Exp(Math.Log(GammaUpper(b, u)) - lnHOfBU);
            double result = J;

            double bPlus2n = b;
            double bPlus2nPlus1 = b + 1;
            double lnXOver2 = 0.5 * Math.Log(x);
            double lnXOver2Sq = lnXOver2 * lnXOver2;
            double lnXTerm = 1.0;

            for (int n = 1; ; n++)
            {
                J = oneOver4TSq * ((bPlus2n * bPlus2nPlus1 * J) + ((bPlus2nPlus1 + u) * lnXTerm));
                bPlus2n += 2;
                bPlus2nPlus1 += 2;
                lnXTerm *= lnXOver2Sq;
                // This can overflow, which will result in infinity. This is OK, because
                // it will mean only that some of the terms added to currP will be zero.
                twoNPlus1Factorial.Add(twoNPlus1Factorial.Last() * (2 * n) * (2 * n + 1));

                double currP = 0.0;
                for (int m = 1; m <= n; m++)
                {
                    currP += p[n - m] * (m * b - n) / twoNPlus1Factorial[m];
                }

                currP /= n;

                double term = J * currP;
                result += term;

                if (Math.Abs(term) <= epsilon * Math.Abs(result))
                {
                    break;
                }

                if (n > 100)
                {
                    throw new Exception("BGRat series not converging");
                }

                p.Add(currP);
            }

            return M * result;
        }

        /// <summary>
        /// The classic incomplete beta function continued fraction.
        /// </summary>
        private class BetaConFracClassic : ContinuedFraction
        {
            /// <summary>
            /// True count.
            /// </summary>
            public double A
            {
                get;
                set;
            }

            /// <summary>
            /// False count.
            /// </summary>
            public double B
            {
                get;
                set;
            }

            /// <summary>
            /// The value.
            /// </summary>
            public double X
            {
                get;
                set;
            }

            /// <summary>
            /// 1.0 minus the value.
            /// </summary>
            public double Y
            {
                get
                {
                    return 1.0 - this.X;
                }
            }

            /// <summary>
            /// The probability of true.
            /// </summary>
            public double P
            {
                get
                {
                    return this.A / (this.A + this.B);
                }
            }

            /// <summary>
            /// Evaluate the incomplete beta function.
            /// </summary>
            /// <param name="epsilon">The convergence tolerance.</param>
            /// <returns>The function value.</returns>
            public virtual double EvaluateIncompleteBeta(double epsilon)
            {
                return Math.Exp(
                   Math.Log(base.Evaluate(epsilon)) +
                   this.A * Math.Log(this.X) +
                   this.B * Math.Log(this.Y) -
                   this.A -
                   Distributions.Beta.BetaLn(this.A, this.B));
            }

            /// <summary>
            /// Gets the numerator of the nth term of the continued fraction.
            /// </summary>
            /// <param name="n">The term index.</param>
            /// <returns>The numerator of the nth term.</returns>
            public override double GetNumerator(int n)
            {
                if (n == 1)
                {
                    return 1.0;
                }
                else
                {
                    int m = (n - 1) / 2;
                    double aPlus2m = this.A + 2 * m;
                    if ((n % 2) == 0)
                    {
                        double num = -(this.A + m) * (this.A + this.B + m) * this.X;
                        double den = aPlus2m * (aPlus2m + 1.0);
                        return num / den;
                    }
                    else
                    {
                        double num = m * (this.B - m) * this.X;
                        double den = (aPlus2m - 1) * aPlus2m;
                        return num / den;
                    }
                }
            }

            /// <summary>
            /// Gets the deniminator of the nth term of the continued fraction.
            /// </summary>
            /// <param name="n">The term index.</param>
            /// <returns>The denominator of the nth term.</returns>
            public override double GetDenominator(int n)
            {
                return 1.0;
            }
        }

        /// <summary>
        /// Incomplete beta function continued fraction according to Didonato and Morris.
        /// </summary>
        private class BetaConFracDidonatoMorris : BetaConFracClassic
        {
            /// <summary>
            /// Evaluate the incomplete beta function.
            /// </summary>
            /// <param name="epsilon">The convergence tolerance.</param>
            /// <returns>The function value.</returns>
            public override double EvaluateIncompleteBeta(double epsilon)
            {
                if (this.X > this.P)
                    throw new InvalidOperationException(@"x must be greater than a/(a+b)");

                return Math.Exp(
                    Math.Log(base.Evaluate(epsilon)) +
                    this.A * Math.Log(this.X) +
                    this.B * Math.Log(this.Y) -
                    Distributions.Beta.BetaLn(this.A, this.B));
            }

            /// <summary>
            /// Gets the numerator of the nth term of the continued fraction.
            /// </summary>
            /// <param name="n">The term index.</param>
            /// <returns>The numerator of the nth term.</returns>
            public override double GetNumerator(int n)
            {
                if (n == 1)
                {
                    return 1.0;
                }
                else
                {
                    int m = n - 1;
                    double am1 = this.A + m - 1;
                    double a2m1 = am1 + m;
                    double abm1 = am1 + this.B;
                    return am1 * abm1 * m * (this.B - m) * this.X * this.X / (a2m1 * a2m1);
                }
            }

            /// <summary>
            /// Gets the deniminator of the nth term of the continued fraction.
            /// </summary>
            /// <param name="n">The term index.</param>
            /// <returns>The denominator of the nth term.</returns>
            public override double GetDenominator(int n)
            {
                int m = n - 1;
                double lambda = this.A - (this.A + this.B) * this.X;
                double a2mMinus1 = this.A + (2 * m) - 1.0;
                double a2mPlus1 = a2mMinus1 + 2.0;
                return
                    m +
                    (m * (this.B - m) * this.X / a2mMinus1) +
                    ((this.A + m) / a2mPlus1) * (lambda + 1.0 + m * (1.0 + this.Y));
            }
        }

        /// <summary>
        /// BFRAC method using continued fraction described in Didonato and Morris.
        /// </summary>
        /// <param name="x">Value.</param>
        /// <param name="a">True count.</param>
        /// <param name="b">False count.</param>
        /// <param name="maxIterations">Maximum number of iterations.</param>
        /// <param name="epsilon">Tolerance.</param>
        /// <param name="useDM">Whether to use the Didonato-Morris (default) or the classic continued fraction.</param>
        /// <returns></returns>
        private static double BFrac(double x, double a, double b, double epsilon, double maxIterations = 60, bool useDM = true)
        {
            BetaConFracClassic conFrac = useDM ? new BetaConFracDidonatoMorris() : new BetaConFracClassic();
            conFrac.X = x;
            conFrac.A = a;
            conFrac.B = b;
            return conFrac.EvaluateIncompleteBeta(epsilon);
        }

        /// <summary>
        /// BASYM method described in Didonato and Morris.
        /// </summary>
        /// <param name="x">Value.</param>
        /// <param name="a">True count.</param>
        /// <param name="b">False count.</param>
        /// <param name="epsilon">Tolerance.</param>
        /// <returns></returns>
        private static double BAsym(double x, double a, double b, double epsilon)
        {
            double p = a / (a + b);
            double q = 1.0 - p;
            double y = 1.0 - x;
            double lambda = (a + b) * (p - x);
            double zsq = -(a * Math.Log(x / p) + b * Math.Log(y / q)); // should be >= 0
            double z = Math.Sqrt(zsq);
            double twoPowMinus15 = Math.Pow(2, -1.5);
            double zsqrt2 = z * Math.Sqrt(2.0);
            double LPrev = 0.25 * Math.Sqrt(Math.PI) * Math.Exp(zsq) * MMath.Erfc(z);
            double LCurr = twoPowMinus15;
            double U = Math.Exp(GammaLnSeries(a + b) - GammaLnSeries(a) - GammaLnSeries(b));
            double mult = 2 * U * Math.Exp(-zsq) / Math.Sqrt(Math.PI);
            double betaGamma = (a < b) ? Math.Sqrt(q / a) : Math.Sqrt(p / b);
            List<double> aTerm = new List<double>();
            List<double> cTerm = new List<double>();
            List<double> eTerm = new List<double>();
            double aFunc(int n)
            {
                double neg = (n % 2) == 0 ? 1.0 : -1.0;
                return (a <= b) ?
                    2.0 * q * (1.0 + neg * Math.Pow(a / b, n + 1)) / (2.0 + n) :
                    2.0 * p * (neg + Math.Pow(b / a, n + 1)) / (2.0 + n);
            }

            // Assumes aTerm has been populated up to n
            double bFunc(int n, double r)
            {
                if (n == 0)
                {
                    return 1.0;
                }
                else if (n == 1)
                {
                    return r * aTerm[1];
                }
                else
                {
                    List<double> bTerm = new List<double>();
                    bTerm.Add(1.0);
                    bTerm.Add(r * aTerm[1]);
                    for (int j = 2; j <= n; j++)
                    {
                        bTerm.Add(r * aTerm[j] + Enumerable.Range(1, j - 1).Sum(i => ((j - i) * r - i) * bTerm[i] * aTerm[j - i]) / j);
                    }

                    return bTerm[n];
                }
            }

            // Assumes aTerm has been populated up to n
            double cFunc(int n) => bFunc(n - 1, -n / 2.0) / n;
            aTerm.Add(aFunc(0));
            aTerm.Add(aFunc(1));
            cTerm.Add(cFunc(1));
            cTerm.Add(cFunc(2));
            eTerm.Add(1.0);
            eTerm.Add(-eTerm[0] * cTerm[1]);

            double result = LPrev + eTerm[1] * LCurr * betaGamma;
            double powBetaGamma = betaGamma;
            double powZSqrt2 = 1.0;

            for (int n = 2; ; n++)
            {
                powZSqrt2 *= zsqrt2;
                double L = (twoPowMinus15 * powZSqrt2) + (n - 1) * LPrev;
                LPrev = LCurr;
                LCurr = L;

                powBetaGamma *= betaGamma;
                aTerm.Add(aFunc(n));
                cTerm.Add(cFunc(n + 1));
                eTerm.Add(-Enumerable.Range(0, n).Sum(i => eTerm[i] * cTerm[n - i]));
                double term = eTerm[n] * LCurr * powBetaGamma;
                result += term;

                if (aTerm.Last() != 0 && Math.Abs(term) <= epsilon * Math.Abs(result))
                {
                    break;
                }

                if (n > 100)
                {
                    throw new Exception("BASym series not converging");
                }
            }

            return mult * result;
        }

        private const double DefaultBetaEpsilon = 1e-15;

        /// <summary>
        /// Computes the regularized incomplete beta function: int_0^x t^(a-1) (1-t)^(b-1) dt / Beta(a,b)
        /// </summary>
        /// <param name="x">The first argument - any real number between 0 and 1.</param>
        /// <param name="a">The second argument - any real number greater than 0.</param>
        /// <param name="b">The third argument - any real number greater than 0.</param>
        /// <param name="epsilon">A tolerance for terminating the series calculation.</param>
        /// <returns>The incomplete beta function at (<paramref name="x"/>, <paramref name="a"/>, <paramref name="b"/>).</returns>
        /// <remarks>The beta function is obtained by setting x to 1.</remarks>
        public static double Beta(double x, double a, double b, double epsilon = DefaultBetaEpsilon)
        {
            double result = 0.0;
            bool swap = false;

            if (Math.Min(a, b) <= 1)
            {
                swap = x > 0.5;
                if (swap)
                {
                    // Swap a for b, x for y
                    double c = a;
                    a = b;
                    b = c;
                    x = 1.0 - x;
                }

                double y = 1.0 - x;
                double p = a / (a + b);
                double q = 1.0 - p;
                double maxAB = Math.Max(a, b);
                double minPt2B = Math.Min(0.2, b);
                double bx = b * x;
                double xPowA = Math.Pow(x, a);
                double bxPowA = Math.Pow(bx, a);

                // Conditions
                bool maxAB_LE_1 = maxAB <= 1;
                bool a_LE_MinPt2B = a < minPt2B;
                bool xPowA_LE_Pt9 = xPowA <= 0.9;
                bool x_GE_Pt1 = x >= 0.1;
                bool x_GE_Pt3 = x >= 0.3;
                bool b_LE_1 = b <= 1;
                bool b_LE_15 = b <= 15;
                bool bxPowA_LE_Pt7 = bxPowA <= 0.7;

                if ((maxAB_LE_1 && !a_LE_MinPt2B) ||
                    (maxAB_LE_1 && a_LE_MinPt2B && xPowA_LE_Pt9) ||
                    (!maxAB_LE_1 && b_LE_1) ||
                    (!maxAB_LE_1 && !b_LE_1 && !x_GE_Pt1 && bxPowA_LE_Pt7))
                {
                    result = BPSer(x, a, b, epsilon);
                }
                else if (
                    (maxAB_LE_1 && a_LE_MinPt2B && !xPowA_LE_Pt9 && x_GE_Pt3) ||
                    (!maxAB_LE_1 && !b_LE_1 && x_GE_Pt3))
                {
                    result = BPSer(y, b, a, epsilon);
                    if (!swap)
                    {
                        result = 1.0 - result;
                    }

                    swap = false;
                }
                else if (
                    (!maxAB_LE_1 && !b_LE_15 && x_GE_Pt1 && !x_GE_Pt3) ||
                    (!maxAB_LE_1 && !b_LE_15 && !x_GE_Pt1 && !bxPowA_LE_Pt7))
                {
                    result = BGRat(y, b, a, 15.0 * epsilon);
                    if (!swap)
                    {
                        result = 1.0 - result;
                    }

                    swap = false;
                }
                else if (
                    (!maxAB_LE_1 && !b_LE_1 && x_GE_Pt1 && !x_GE_Pt3 && b_LE_15) ||
                    (!maxAB_LE_1 && !b_LE_1 && !x_GE_Pt1 && !bxPowA_LE_Pt7 && b_LE_15) ||
                    (maxAB_LE_1 && a_LE_MinPt2B && !xPowA_LE_Pt9 && !x_GE_Pt3))
                {
                    int n = 20;
                    double w0 = BUp(y, b, a, n, epsilon);
                    result = w0 + BGRat(y, b + n, a, 15.0 * epsilon);
                    if (!swap)
                    {
                        result = 1.0 - result;
                    }

                    swap = false;
                }
                else
                {
                    throw new InvalidOperationException(String.Format("Unexpected condition: a = {0}, b = {1}, x = {2}", a, b, x));
                }
            }
            else
            {
                double p = a / (a + b);
                swap = x > p;
                if (swap)
                {
                    // Swap a for b, x for y
                    double c = a;
                    a = b;
                    b = c;
                    x = 1.0 - x;
                }

                double y = 1.0 - x;
                p = a / (a + b);
                double q = 1.0 - p;
                double bx = b * x;
                int n = (int)Math.Floor(b);
                if (n == b)
                {
                    n -= 1;
                }

                double bbar = b - n;

                // Conditions
                bool b_LT_40 = b < 40;
                bool a_GT_15 = a > 15;
                bool a_LE_100 = a <= 100;
                bool b_LE_100 = b <= 100;
                bool x_LE_Pt7 = x <= 0.7;
                bool x_LT_Pt97p = x < 0.97 * p;
                bool y_GT_1Pt03q = y > 1.03 * q;
                bool bx_LE_Pt7 = bx <= 0.7;
                bool a_LE_b = a <= b;

                if (b_LT_40 && bx_LE_Pt7)
                {
                    result = BPSer(x, a, b, epsilon);
                }
                else if (b_LT_40 && !bx_LE_Pt7 && x_LE_Pt7)
                {
                    result = BUp(y, bbar, a, n, epsilon) + BPSer(x, a, bbar, epsilon);
                }
                else if (b_LT_40 && !x_LE_Pt7 && a_GT_15)
                {
                    result = BUp(y, bbar, a, n, epsilon) + BGRat(x, a, bbar, 15.0 * epsilon);
                }
                else if (b_LT_40 && !x_LE_Pt7 && !a_GT_15)
                {
                    int m = 20;
                    result = BUp(y, bbar, a, n, epsilon) + BUp(x, a, bbar, m, epsilon) + BGRat(x, a + m, bbar, 15.0 * epsilon);
                }
                else if (
                    (!b_LT_40 && a_LE_b && a_LE_100) ||
                    (a_LE_b && !a_LE_100 && x_LT_Pt97p) ||
                    (!b_LT_40 && !a_LE_b && b_LE_100) ||
                    (!a_LE_b && b > 100 && y_GT_1Pt03q))
                {
                    result = BFrac(x, a, b, 15.0 * epsilon);
                }
                else if (
                    (a_LE_b && !a_LE_100 && !x_LT_Pt97p) ||
                    (!a_LE_b && !b_LE_100 && !y_GT_1Pt03q))
                {
                    result = BAsym(x, a, b, 100.0 * epsilon);
                }
                else
                {
                    throw new InvalidOperationException(String.Format("Unexpected condition: a = {0}, b = {1}, x = {2}", a, b, x));
                }
            }

            if (swap)
            {
                result = 1.0 - result;
            }

            return result;
        }

        #endregion

        #region Gamma functions

        /// <summary>
        /// Evaluates Gamma(x), defined as the integral from 0 to x of t^(x-1)*exp(-t) dt.
        /// </summary>
        /// <param name="x">Any real value.</param>
        /// <returns>Gamma(x).</returns>
        public static double Gamma(double x)
        {
            if (x < 0)
            {
                // this test also catches -inf
                if (Math.Floor(x) == x)
                {
                    return Double.NaN;
                }

                // Gamma(z)*Gamma(-z) = -pi/z/sin(pi*z)
                return -Math.PI / (x * Math.Sin(Math.PI * x) * Gamma(-x));
            }

            if (x <= GammaSmallX)
            {
                return 1 / x + GammaSeries(x);
            }

            if (x > 180)
            {
                return Double.PositiveInfinity;
            }

            return Math.Exp(GammaLn(x));
        }

        private const double GammaSmallX = 1e-3;

        /// <summary>
        /// Computes the natural logarithm of the Gamma function.
        /// </summary>
        /// <param name="x">A real value >= 0.</param>
        /// <returns>ln(Gamma(x)).</returns>
        /// <remarks>This function provides higher accuracy than <c>Math.Log(Gamma(x))</c>, which may fail for large x.</remarks>
        public static double GammaLn(double x)
        {
            if (Double.IsPositiveInfinity(x) || (x == 0))
            {
                return Double.PositiveInfinity;
            }

            /* Negative values */
            if (x < 0)
            {
                // answer might be complex
                return Double.NaN;
            }

            if (x < 6)
            {
                double result = 0;
                while (x < 1.5)
                {
                    result -= Math.Log(x);
                    x++;
                }
                while (x > 2.5)
                {
                    x--;
                    result += Math.Log(x);
                }
                // 1.5 <= x <= 2.5
                // Use Taylor series at x=2
                double dx = x - 2;
                double sum =
                    // Truncated series 1: Gammaln at 2
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 1.1921401405860912e-09*dx**25 when dx >= 0
                    // which is at most Ulp(0.42278433509846713*dx)/2 when 0 <= dx <= 0.4914357448219413
                    dx * (0.42278433509846713 +
                    dx * (0.3224670334241132 +
                    dx * (-0.067352301053198102 +
                    dx * (0.020580808427784546 +
                    dx * (-0.0073855510286739857 +
                    dx * (0.0028905103307415234 +
                    dx * (-0.001192753911703261 +
                    dx * (0.00050966952474304245 +
                    dx * (-0.00022315475845357939 +
                    dx * (9.9457512781808531e-05 +
                    dx * (-4.4926236738133142e-05 +
                    dx * (2.0507212775670691e-05 +
                    dx * (-9.4394882752683967e-06 +
                    dx * (4.3748667899074882e-06 +
                    dx * (-2.0392157538013662e-06 +
                    dx * (9.5514121304074194e-07 +
                    dx * (-4.4924691987645662e-07 +
                    dx * (2.1207184805554665e-07 +
                    dx * (-1.0043224823968099e-07 +
                    dx * (4.7698101693639804e-08 +
                    dx * (-2.2711094608943164e-08 +
                    dx * (1.0838659214896955e-08 +
                    dx * (-5.1834750419700466e-09 +
                    dx * 2.4836745438024785e-09)))))))))))))))))))))))
                    ;
                result += sum;
                return result;
            }
            else // x >= 6
            {
                double sum = LnSqrt2PI;
                while (x < GammaLnLargeX)
                {
                    sum -= Math.Log(x);
                    x++;
                }
                // x >= GammaLnLargeX
                // Use asymptotic series
                return GammaLnSeries(x) + (x - 0.5) * Math.Log(x) - x + sum;
            }
        }

        /// <summary>
        /// Computes the logarithm of the Pochhammer function: GammaLn(x + n) - GammaLn(x)
        /// </summary>
        /// <param name="x">A real number &gt; 0</param>
        /// <param name="n">If zero, result is 0.</param>
        /// <returns></returns>
        public static double RisingFactorialLn(double x, double n)
        {
            // To ensure that the result increases with n, we group terms in n.
            if (x <= 0) throw new ArgumentOutOfRangeException(nameof(x), x, "x <= 0");
            else if (n == 0) return 0;
            else if (x < Ulp1 && x + n < Ulp1)
            {
                // GammaLn(x) = -log(x)
                return -Log1Plus(n / x);
            }
            else if (Math.Abs(n / x) < CbrtUlp1)
            {
                // x >= 1e-6 ensures that Tetragamma doesn't overflow
                // For small x, Digamma(x) = -1/x, Trigamma(x) = 1/x^2, Tetragamma(x) = -1/x^3
                // To ignore the next term, we need 1/x to dominate n^3/x^4, i.e. 1 >> n^3/x^3
                return n * (MMath.Digamma(x) + 0.5 * n * (MMath.Trigamma(x) + n / 3 * MMath.Tetragamma(x)));
            }
            else if (x > GammaLnLargeX && x + n > GammaLnLargeX)
            {
                double nOverX = n / x;
                if (Math.Abs(nOverX) < SqrtUlp1)
                    // log(1 + x) = x - 0.5*x^2  when x^2 << 1
                    return n * (Log1Plus(nOverX) - 0.5 * (1 - 0.5 / x) * nOverX + (GammaLnSeries(x + n) - GammaLnSeries(x)) / n - 0.5 / x + Math.Log(x));
                else
                    return n * ((1 + (x - 0.5) / n) * Log1Plus(nOverX) + (GammaLnSeries(x + n) - GammaLnSeries(x)) / n - 1 + Math.Log(x));
            }
            else return MMath.GammaLn(x + n) - MMath.GammaLn(x);
        }

        /// <summary>
        /// Computes the logarithm of the Pochhammer function, divided by n: (GammaLn(x + n) - GammaLn(x))/n
        /// </summary>
        /// <param name="x">A real number &gt; 0</param>
        /// <param name="n">If zero, result is 0.</param>
        /// <returns></returns>
        public static double RisingFactorialLnOverN(double x, double n)
        {
            // To ensure that the result increases with n, we group terms in n.
            if (x <= 0) throw new ArgumentOutOfRangeException(nameof(x), x, "x <= 0");
            else if (n == 0) return 0;
            else if (x < Ulp1 && x + n < Ulp1)
            {
                // GammaLn(x) = -log(x)
                return -Log1Plus(n / x) / n;
            }
            else if (Math.Abs(n / x) < CbrtUlp1)
            {
                // x >= 1e-6 ensures that Tetragamma doesn't overflow
                // For small x, Digamma(x) = -1/x, Trigamma(x) = 1/x^2, Tetragamma(x) = -1/x^3
                // To ignore the next term, we need 1/x to dominate n^3/x^4, i.e. 1 >> n^3/x^3
                return MMath.Digamma(x) + 0.5 * n * (MMath.Trigamma(x) + n / 3 * MMath.Tetragamma(x));
            }
            else if (x > GammaLnLargeX && x + n > GammaLnLargeX)
            {
                double nOverX = n / x;
                if (Math.Abs(nOverX) < SqrtUlp1)
                    // log(1 + x) = x - 0.5*x^2  when x^2 << 1
                    return Log1Plus(nOverX) - 0.5 * (1 - 0.5 / x) * nOverX + (GammaLnSeries(x + n) - GammaLnSeries(x)) / n - 0.5 / x + Math.Log(x);
                else
                    return (1 + (x - 0.5) / n) * Log1Plus(nOverX) + (GammaLnSeries(x + n) - GammaLnSeries(x)) / n - 1 + Math.Log(x);
            }
            else return (MMath.GammaLn(x + n) - MMath.GammaLn(x)) / n;
        }

        const int DigammaTableLength = 100;
        private static readonly Lazy<double[]> DigammaTable = new Lazy<double[]>(MakeDigammaTable);

        private static double[] MakeDigammaTable()
        {
            double[] table = new double[DigammaTableLength];
            table[0] = double.NegativeInfinity;
            table[1] = Digamma1;
            for (int i = 2; i < table.Length; i++)
                table[i] = table[i - 1] + 1.0 / (i - 1);
            return table;
        }

        /// <summary>
        /// Evaluates Digamma(x), the derivative of ln(Gamma(x)).
        /// </summary>
        /// <param name="x">Any real value.</param>
        /// <returns>Digamma(x).</returns>
        public static double Digamma(double x)
        {
            /* Negative values */
            /* Use the reflection formula (Jeffrey 11.1.6):
             * digamma(-x) = digamma(x+1) + pi*cot(pi*x)
             *
             * This is related to the identity
             * digamma(-x) = digamma(x+1) - digamma(z) + digamma(1-z)
             * where z is the fractional part of x
             * For example:
             * digamma(-3.1) = 1/3.1 + 1/2.1 + 1/1.1 + 1/0.1 + digamma(1-0.1)
             *                 = digamma(4.1) - digamma(0.1) + digamma(1-0.1)
             * Then we use
             * digamma(1-z) - digamma(z) = pi*cot(pi*z)
             */
            if (x < 0)
            {
                // this test also catches -inf
                if (Math.Floor(x) == x)
                {
                    return Double.NaN;
                }

                return (Digamma(1 - x) - Math.PI / Math.Tan(Math.PI * x));
            }

            /* Lookup table for when x is an integer */
            int xAsInt = (int)x;
            if ((xAsInt == x) && (xAsInt < DigammaTableLength))
            {
                return DigammaTable.Value[xAsInt];
            }

            if (x <= 2.5)
            {
                double result2 = 1 + Digamma1;
                while (x < 1.5)
                {
                    result2 -= 1 / x;
                    x++;
                }
                // 1.5 <= x <= 2.5
                // Use Taylor series at x=2
                double dx = x - 2;
                double sum2 =
                    // Truncated series 3: Digamma at 2
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 3.7253340247884573e-09*dx**27 when dx >= 0
                    // which is at most Ulp(1)/2 when 0 <= dx <= 0.52634237250421645
                    dx * (0.64493406684822641 +
                    dx * (-0.20205690315959429 +
                    dx * (0.082323233711138186 +
                    dx * (-0.036927755143369927 +
                    dx * (0.01734306198444914 +
                    dx * (-0.0083492773819228271 +
                    dx * (0.0040773561979443396 +
                    dx * (-0.0020083928260822143 +
                    dx * (0.00099457512781808526 +
                    dx * (-0.00049418860411946453 +
                    dx * (0.00024608655330804832 +
                    dx * (-0.00012271334757848915 +
                    dx * (6.1248135058704828e-05 +
                    dx * (-3.0588236307020493e-05 +
                    dx * (1.5282259408651871e-05 +
                    dx * (-7.6371976378997626e-06 +
                    dx * (3.8172932649998402e-06 +
                    dx * (-1.908212716553939e-06 +
                    dx * (9.5396203387279621e-07 +
                    dx * (-4.7693298678780645e-07 +
                    dx * (2.38450502727733e-07 +
                    dx * (-1.1921992596531106e-07 +
                    dx * (5.960818905125948e-08 +
                    dx * (-2.9803503514652279e-08 +
                    dx * (1.4901554828365043e-08 +
                    dx * -7.4507117898354301e-09)))))))))))))))))))))))))
                    ;
                result2 += sum2;
                return result2;
            }

            const double c_digamma_large = 8;

            /* Reduce to digamma(X + N) where (X + N) >= C */
            double result = 0;
            while (x < c_digamma_large)
            {
                result -= 1 / x;
                ++x;
            }

            /* X >= L. Use de Moivre's expansion. */
            // This expansion can be computed in Maple via asympt(Psi(x),x)
            double invX = 1 / x;
            result += Math.Log(x) - 0.5 * invX;
            double invX2 = invX * invX;
            double sum =
                // Truncated series 4: Digamma asymptotic
                // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                // Error is at most 0.44325980392156861*invX2**8 when invX2 >= 0
                // which is at most Ulp(1)/2 when 0 <= invX2 <= 0.011216154551723961
                invX2 * (1.0 / 12.0 +
                invX2 * (-1.0 / 120.0 +
                invX2 * (1.0 / 252.0 +
                invX2 * (-1.0 / 240.0 +
                invX2 * (1.0 / 132.0 +
                invX2 * (-691.0 / 32760.0 +
                invX2 * (1.0 / 12.0
                )))))))
                ;
            result -= sum;
            return result;
        }

        /// <summary>
        /// Evaluates Trigamma(x), the derivative of Digamma(x).
        /// </summary>
        /// <param name="x">Any real value.</param>
        /// <returns>Trigamma(x).</returns>
        public static double Trigamma(double x)
        {
            double result;

            /* Negative values */
            /* Use the derivative of the digamma reflection formula:
             * -trigamma(-x) = trigamma(x+1) - (pi*csc(pi*x))^2
             */
            if (x < 0)
            {
                // this test also catches -inf
                if (Math.Floor(x) == x)
                {
                    return Double.NaN;
                }

                result = Math.PI / Math.Sin(Math.PI * x);
                return (-Trigamma(1 - x) + result * result);
            }

            // The threshold for applying de Moivre's expansion for the trigamma function.
            const double c_trigamma_large = 8;
            const double c_trigamma_small = 1e-4;

            /* Shift the argument and use Taylor series at 1 if argument <= small */
            if (x <= c_trigamma_small)
            {
                return 1.0 / (x * x) +
                    // Truncated series 5: Trigamma at 1
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 3.2469697011334144*x**2 when x >= 0
                    // which is at most Ulp(1e8)/2 when 0 <= x <= 5.8474429974674805e-05
                    1.6449340668482264 +
                    x * -2.4041138063191885
                    ;
            }

            result = 0.0;

            /* Reduce to trigamma(x+n) where ( X + N ) >= L */
            while (x < c_trigamma_large)
            {
                result += 1 / (x * x);
                ++x;
            }

            /* X >= L.    Apply asymptotic formula. */
            // This expansion can be computed in Maple via asympt(Psi(1,x),x)
            double invX2 = 1 / (x * x);
            result += 0.5 * invX2;
            double sum =
                    // Truncated series 6: Trigamma asymptotic
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 54.971177944862156*invX2**9 when invX2 >= 0
                    // which is at most Ulp(1)/2 when 0 <= invX2 <= 0.010812334306109761
                    invX2 * (1.0 / 6.0 +
                    invX2 * (-1.0 / 30.0 +
                    invX2 * (1.0 / 42.0 +
                    invX2 * (-1.0 / 30.0 +
                    invX2 * (5.0 / 66.0 +
                    invX2 * (-691.0 / 2730.0 +
                    invX2 * (7.0 / 6.0 +
                    invX2 * (-3617.0 / 510.0
                    ))))))))
                    ;
            result += (1 + sum) / x;
            return result;
        }

        /// <summary>
        ///  Evaluates Tetragamma, the forth derivative of logGamma(x)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Tetragamma(double x)
        {
            if (x < 0)
                return double.NaN;

            // The threshold for applying de Moivre's expansion for the quadgamma function.
            const double c_tetragamma_large = 12,
                         c_tetragamma_small = 1e-4;
            /* Use Taylor series if argument <= small */
            if (x < c_tetragamma_small)
            {
                return -2 / (x * x * x) +
                    // Truncated series 7: Tetragamma at 1
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 12.44313306172044*x**2 when x >= 0
                    // which is at most Ulp(2e12)/2 when 0 <= x <= 0.0042243047356963293
                    -2.4041138063191885 +
                    x * 6.4939394022668289
                    ;
            }
            double result = 0;
            /* Reduce to Tetragamma(x+n) where ( X + N ) >= L */
            while (x < c_tetragamma_large)
            {
                result -= 2 / (x * x * x);
                x++;
            }
            /* X >= L.    Apply asymptotic formula. */
            // This expansion can be computed in Maple via asympt(Psi(2,x),x) or found in
            // Milton Abramowitz and Irene A. Stegun, Handbook of Mathematical Functions, Section 6.4
            double invX2 = 1 / (x * x);
            result += -invX2 / x;
            double sum =
                // Truncated series 8: Tetragamma asymptotic
                // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                // Error is at most 120.56666666666666*invX2**9 when invX2 >= 0
                // which is at most Ulp(0.00057870370370370367)/2 when 0 <= invX2 <= 0.0043280597593257338
                invX2 * (-1.0 +
                invX2 * (-1.0 / 2.0 +
                invX2 * (1.0 / 6.0 +
                invX2 * (-1.0 / 6.0 +
                invX2 * (3.0 / 10.0 +
                invX2 * (-5.0 / 6.0 +
                invX2 * (691.0 / 210.0 +
                invX2 * (-35.0 / 2.0
                ))))))))
                ;
            result += sum;
            return result;
        }

        /// <summary>
        /// Computes the natural logarithm of the multivariate Gamma function.
        /// </summary>
        /// <param name="x">A real value >= 0.</param>
        /// <param name="d">The dimension, an integer > 0.</param>
        /// <returns>ln(Gamma_d(x))</returns>
        /// <remarks>The <a href="http://en.wikipedia.org/wiki/Multivariate_gamma_function">multivariate Gamma function</a> 
        /// is defined as Gamma_d(x) = pi^(d*(d-1)/4)*prod_(i=1..d) Gamma(x + (1-i)/2)</remarks>
        public static double GammaLn(double x, double d)
        {
            double result = d * (d - 1) / 4 * LnPI;
            for (int i = 0; i < d; i++)
            {
                result += GammaLn(x - 0.5 * i);
            }
            return result;
        }

        /// <summary>
        /// Derivative of the natural logarithm of the multivariate Gamma function.
        /// </summary>
        /// <param name="x">A real value >= 0</param>
        /// <param name="d">The dimension, an integer > 0</param>
        /// <returns>digamma_d(x)</returns>
        public static double Digamma(double x, double d)
        {
            double result = 0;
            for (int i = 0; i < d; i++)
            {
                result += Digamma(x - 0.5 * i);
            }
            return result;
        }

        /// <summary>
        /// Second derivative of the natural logarithm of the multivariate Gamma function.
        /// </summary>
        /// <param name="x">A real value >= 0</param>
        /// <param name="d">The dimension, an integer > 0</param>
        /// <returns>trigamma_d(x)</returns>
        public static double Trigamma(double x, double d)
        {
            double result = 0;
            for (int i = 0; i < d; i++)
            {
                result += Trigamma(x - 0.5 * i);
            }
            return result;
        }

        /// <summary>
        /// Evaluates the natural logarithm of Gamma(n+1)/(Gamma(k+1)*Gamma(n-k+1))
        /// </summary>
        /// <param name="n"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static double ChooseLn(double n, double k)
        {
            if (k <= -1 || k >= n + 1)
                return Double.NegativeInfinity;
            return GammaLn(n + 1) - GammaLn(k + 1) - GammaLn(n - k + 1);
        }

        /// <summary>
        /// Evaluates the natural logarithm of Gamma(n+1)/(prod_i Gamma(k[i]+1))
        /// </summary>
        /// <param name="n"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static double ChooseLn(double n, double[] k)
        {
            double result = GammaLn(n + 1);
            for (int i = 0; i < k.Length; i++)
            {
                result -= GammaLn(k[i] + 1);
            }
            return result;
        }

        /// <summary>
        /// Computes GammaLower(a, r*u) - GammaLower(a, r*l) to high accuracy.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="rate"></param>
        /// <param name="lowerBound">Any real number.</param>
        /// <param name="upperBound">Any real number.</param>
        /// <param name="regularized">If true, result is divided by <c>MMath.Gamma(shape)</c></param>
        /// <returns></returns>
        public static double GammaProbBetween(double shape, double rate, double lowerBound, double upperBound, bool regularized = true)
        {
            lowerBound = Math.Max(0, lowerBound);
            if (lowerBound >= upperBound) return 0;
            double rl = rate * lowerBound;
            double ru = rate * upperBound;
            double logRate = Math.Log(rate);
            double logl = Math.Log(lowerBound);
            double logrl = logRate + logl;
            // Use the criterion from Gautschi (1979) to determine whether GammaLower(a,x) or GammaUpper(a,x) is smaller.
            bool lowerIsSmaller;
            if (rl > 0.25)
                lowerIsSmaller = (shape > rl + 0.25);
            else
                lowerIsSmaller = (shape > -MMath.Ln2 / logrl);
            if (!lowerIsSmaller)
            {
                double smallShape = -1e-16 / logrl;
                if (ru < 1e-16 && Math.Abs(shape) < smallShape)
                {
                    double logu = Math.Log(upperBound);
                    double result = logu - logl;
                    return regularized ? result * shape : result;
                }
                else
                {
                    double lower;
                    if (MMath.AreEqual(rl, 0))
                    {
                        //lower = -logrl; when shape is small
                        lower = Math.Exp(MMath.LogGammaUpper(shape, 0, logrl));
                        if (regularized) lower *= shape;
                    }
                    else
                    {
                        lower = MMath.GammaUpper(shape, rl, regularized);
                    }
                    double upper;
                    if (MMath.AreEqual(ru, 0))
                    {
                        //upper = -logru; when shape is small
                        double logu = Math.Log(upperBound);
                        double logru = logRate + logu;
                        upper = Math.Exp(MMath.LogGammaUpper(shape, 0, logru));
                        if (regularized) upper *= shape;
                    }
                    else
                    {
                        upper = MMath.GammaUpper(shape, ru, regularized);
                    }
                    // For this difference to be non-negative, we need GammaUpper to be non-increasing
                    // This is inaccurate when lowerBound is close to upperBound.  In that case, use a Taylor expansion of lowerBound around upperBound.
                    return Math.Max(0, lower - upper);
                }
            }
            else
            {
                double diff = MMath.GammaLower(shape, ru) - MMath.GammaLower(shape, rl);
                diff = Math.Max(0, diff);
                return regularized ? diff : (MMath.Gamma(shape) * diff);
            }
        }

        /// <summary>
        /// Compute log(int_x^inf t^(a-1) exp(-b*t) dt)
        /// </summary>
        /// <param name="a">The shape parameter.</param>
        /// <param name="logb">The logarithm of the rate parameter.</param>
        /// <param name="logx">The logarithm of the threshold.</param>
        /// <returns></returns>
        public static double LogGammaUpper(double a, double logb, double logx)
        {
            double bx = Math.Exp(logb + logx);
            if (bx - a > 1e16)
            {
                return a * logx - bx - Math.Log(bx - a + 1);
            }
            else if (AreEqual(bx, 0) && !AreEqual(logx, 0))
            {
                double b = Math.Exp(logb);
                return Math.Log(GammaUpperSeries1(a, b, logx) + Math.Exp(LogGammaUpper(a, logb, 0)));
            }
            else if (a < 0 && bx < 1)
            {
                if (Math.Exp(a * (logb + logx)) > 1e16)
                {
                    // In this case, GammaUpper(a, bx, false) == -bx^a/a
                    return a * logx - Math.Log(-a);
                }
                else
                {
                    return Math.Log(GammaUpperSeries1(a, bx, logb + logx, false) + GammaUpperConFrac(a, 1, false)) - a * logb;
                }
            }
            else
            {
                return Math.Log(GammaUpper(a, bx, false)) - a * logb;
            }
        }

        /// <summary>
        /// Compute the regularized upper incomplete Gamma function: int_x^inf t^(a-1) exp(-t) dt / Gamma(a)
        /// </summary>
        /// <param name="a">The shape parameter.  Must be &gt; 0 if regularized is true or x is 0.</param>
        /// <param name="x">The lower bound of the integral, &gt;= 0</param>
        /// <param name="regularized">If true, result is divided by Gamma(a)</param>
        /// <returns></returns>
        public static double GammaUpper(double a, double x, bool regularized = true)
        {
            // special cases:
            // GammaUpper(1,x) = exp(-x)
            // GammaUpper(a,0) = 1
            // GammaUpper(a,x) = GammaUpper(a-1,x) + x^(a-1) exp(-x) / Gamma(a)
            if (x < 0)
                throw new ArgumentException($"x ({x}) < 0");
            if (!regularized)
            {
                if (a < 1 && x >= 0.1) return GammaUpperConFrac2(a, x, regularized);
                else if (a <= GammaSmallX)
                {
                    if (x < 1)
                    {
                        // This case is needed by TruncatedGamma_GetMeanPower_WithinBounds
                        double logx = Math.Log(x);
                        return GammaUpperSeries1(a, x, logx, regularized) + GammaUpperConFrac2(a, 1, regularized);
                    }
                    return GammaUpperSeries(a, x, regularized);
                }
                else
                {
                    double regularizedResult = GammaUpper(a, x, true);
                    return MMath.AreEqual(regularizedResult, 0) ? 0 : Gamma(a) * regularizedResult;
                }
            }
            if (a <= 0)
                throw new ArgumentException($"a ({a}) <= 0");
            if (x == 0) return 1; // avoid 0/0
            // Use the criterion from Gautschi (1979) to determine whether GammaLower(a,x) or GammaUpper(a,x) is smaller.
            bool lowerIsSmaller;
            if (x > 0.25)
                lowerIsSmaller = (a > x + 0.25);
            else
                lowerIsSmaller = (a > -MMath.Ln2 / Math.Log(x));
            if (lowerIsSmaller)
            {
                if (x < 0.5 * a + 67)
                    return 1 - GammaLowerSeries(a, x);
                else // x > 67, a > 67, a > x + 0.25
                    return GammaAsympt(a, x, true);
            }
            else if ((a > 20) && x < 2.35 * a)
                return GammaAsympt(a, x, true);
            else if (x > 1.5)
                return GammaUpperConFrac(a, x);
            else if (a <= Ulp1)
            {
                // Gamma(a) = 1/a for a <= 1e-16
                return a * GammaUpperSeries(a, x, false);
            }
            else
                return GammaUpperSeries(a, x);
        }

        // Origin: James McCaffrey, http://msdn.microsoft.com/en-us/magazine/dn520240.aspx
        /// <summary>
        /// Compute the regularized lower incomplete Gamma function: int_0^x t^(a-1) exp(-t) dt / Gamma(a)
        /// </summary>
        /// <param name="a">The shape parameter, &gt; 0</param>
        /// <param name="x">The upper bound of the integral, &gt;= 0</param>
        /// <returns></returns>
        public static double GammaLower(double a, double x)
        {
            if (x < 0)
                throw new ArgumentException($"x ({x}) < 0");
            if (a <= 0)
                throw new ArgumentException($"a ({a}) <= 0");
            if (x == 0) return 0; // avoid 0/0
            // Use the criterion from Gautschi (1979) to determine whether GammaLower(a,x) or GammaUpper(a,x) is smaller.
            bool lowerIsSmaller;
            if (x > 0.25)
                lowerIsSmaller = (a > x + 0.25);
            else
                lowerIsSmaller = (a > -MMath.Ln2 / Math.Log(x));
            if (lowerIsSmaller)
            {
                if (x < 0.5 * a + 67)
                    return GammaLowerSeries(a, x);
                else // x > 67, a > 67, a > x + 0.25
                    return GammaAsympt(a, x, false);
            }
            else if ((a > 20) && x < 2.35 * a)
                return GammaAsympt(a, x, false);
            else if (x > 1.5)
                return 1 - GammaUpperConFrac(a, x);
            else
                return 1 - GammaUpperSeries(a, x);
        }

        /// <summary>
        /// Computes <c>x - log(1+x)</c> to high accuracy.
        /// </summary>
        /// <param name="x">Any real number &gt;= -1</param>
        /// <returns>A real number &gt;= 0</returns>
        internal static double XMinusLog1Plus(double x)
        {
            if (Math.Abs(x) < 1e-1)
            {
                return
                    // Truncated series 12: x - log(1 + x)
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 0.055555555555555552*x**18 when x >= 0
                    // which is at most Ulp(0.5*x*x)/2 when 0 <= x <= 0.11547242760872471
                    x * x * (1.0 / 2.0 +
                    x * (-1.0 / 3.0 +
                    x * (1.0 / 4.0 +
                    x * (-1.0 / 5.0 +
                    x * (1.0 / 6.0 +
                    x * (-1.0 / 7.0 +
                    x * (1.0 / 8.0 +
                    x * (-1.0 / 9.0 +
                    x * (1.0 / 10.0 +
                    x * (-1.0 / 11.0 +
                    x * (1.0 / 12.0 +
                    x * (-1.0 / 13.0 +
                    x * (1.0 / 14.0 +
                    x * (-1.0 / 15.0 +
                    x * (1.0 / 16.0 +
                    x * (-1.0 / 17.0
                    ))))))))))))))))
                    ;
            }
            else if (x >= -1)
            {
                return x - MMath.Log1Plus(x);
            }
            else throw new ArgumentOutOfRangeException(nameof(x), x, "x < -1");
        }

        // Reference:
        // "Computation of the incomplete gamma function ratios and their inverse"
        // Armido R DiDonato and Alfred H Morris, Jr.   
        // ACM Transactions on Mathematical Software (TOMS)
        // Volume 12 Issue 4, Dec. 1986  
        // http://dl.acm.org/citation.cfm?id=23109
        // and section 8.3 of
        // "Numerical Methods for Special Functions"
        // Amparo Gil, Javier Segura, Nico M. Temme, 2007
        /// <summary>
        /// Compute the regularized lower incomplete Gamma function: <c>int_0^x t^(a-1) exp(-t) dt / Gamma(a)</c>
        /// </summary>
        /// <param name="a">Must be &gt; 20</param>
        /// <param name="x">A real number &gt;= 0</param>
        /// <param name="upper">If true, compute the upper incomplete Gamma function</param>
        /// <returns></returns>
        private static double GammaAsympt(double a, double x, bool upper)
        {
            if (a <= 20)
                throw new Exception("a <= 20");
            double xOverAMinus1 = (x - a) / a;
            double phi = XMinusLog1Plus(xOverAMinus1);
            // phi >= 0
            double y = a * phi;
            double z = Math.Sqrt(2 * phi);
            if (x <= a)
                z *= -1;
            int length = GammaLowerAsympt_C0.Length;
            double bEven = GammaLowerAsympt_C0[length - 1];
            double sum = bEven;
            double bOdd = GammaLowerAsympt_C0[length - 2];
            sum = z * sum + bOdd;
            for (int i = length - 3; i >= 0; i -= 2)
            {
                bEven = bEven * (i + 2) / a + GammaLowerAsympt_C0[i];
                sum = z * sum + bEven;
                if (i > 0)
                {
                    bOdd = bOdd * (i + 1) / a + GammaLowerAsympt_C0[i - 1];
                    sum = z * sum + bOdd;
                }
            }
            sum *= a / (a + bOdd);
            if (x <= a)
                sum *= -1;
            double result = 0.5 * Erfc(Math.Sqrt(y)) + sum * Math.Exp(-y) / (Math.Sqrt(a) * MMath.Sqrt2PI);
            return ((x > a) == upper) ? result : 1 - result;
        }

        // Reference: Appendix F of
        // "Computation of the incomplete gamma function ratios and their inverse"
        // Armido R DiDonato and Alfred H Morris, Jr.   
        // ACM Transactions on Mathematical Software (TOMS)
        // Volume 12 Issue 4, Dec. 1986  
        // http://dl.acm.org/citation.cfm?id=23109
        private static readonly double[] GammaLowerAsympt_C0 =
        {
            -.333333333333333333333333333333E+00,
            .833333333333333333333333333333E-01,
            -.148148148148148148148148148148E-01,
            .115740740740740740740740740741E-02,
            .352733686067019400352733686067E-03,
            -.178755144032921810699588477356E-03,
            .391926317852243778169704095630E-04,
            -.218544851067999216147364295512E-05,
            -.185406221071515996070179883623E-05,
            .829671134095308600501624213166E-06,
            -.176659527368260793043600542457E-06,
            .670785354340149858036939710030E-08,
            .102618097842403080425739573227E-07,
            -.438203601845335318655297462245E-08,
            .914769958223679023418248817633E-09,
            -.255141939949462497668779537994E-10,
            -.583077213255042506746408945040E-10,
            .243619480206674162436940696708E-10,
            -.502766928011417558909054985926E-11,
            .110043920319561347708374174497E-12,
            .337176326240098537882769884169E-12,
            -.139238872241816206591936618490E-12,
            .285348938070474432039669099053E-13,
            -.513911183424257261899064580300E-15,
            -.197522882943494428353962401581E-14,
            .809952115670456133407115668703E-15,
            -.165225312163981618191514820265E-15,
            .253054300974788842327061090060E-17,
            .116869397385595765888230876508E-16
        };

        // Origin: James McCaffrey, http://msdn.microsoft.com/en-us/magazine/dn520240.aspx
        /// <summary>
        /// Compute the regularized lower incomplete Gamma function by a series expansion
        /// </summary>
        /// <param name="a">The shape parameter, &gt; 0</param>
        /// <param name="x">The lower bound of the integral, &gt;= 0</param>
        /// <returns></returns>
        private static double GammaLowerSeries(double a, double x)
        {
            // the series is: x^a exp(-x) sum_{k=0}^inf x^k / Gamma(a+k+1)
            // if x < a+1, then the terms decrease monotonically.
            // to have a reasonably fast convergence, we want x < 0.5*a + 67
            double denom = a;
            double delta = 1.0 / a;
            double sum = delta;
            double scale = GammaUpperScale(a, x);
            for (int i = 0; i < 1000; i++)
            {
                ++denom;
                delta *= x / denom;
                double oldSum = sum;
                sum += delta;
                if (AreEqual(sum, oldSum))
                    return sum * scale;
            }
            throw new Exception($"GammaLowerSeries not converging for a={a:g17} x={x:g17}");
        }

        /// <summary>
        /// Compute int_x^1 t^(a-1) sum_{k=0}^inf (-t)^k/k! dt by series expansion.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="x"></param>
        /// <param name="logx"></param>
        /// <param name="regularized"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static double GammaUpperSeries1(double a, double x, double logx, bool regularized)
        {
            // The starting point is exp(-t) = sum_{k=0}^inf (-t)^k/k!
            // Substituting gives sum_{k=0}^inf int_x^1 t^(a-1) (-t)^k/k! dt
            // = sum_{k=0}^inf (-1)^k/k! int_x^1 t^(a+k-1) dt
            // = sum_{k=0}^inf (-1)^k/k! ((1 - x^(a+k))/(a+k) 1(a+k != 0) -log(x) 1(a+k == 0))
            // = (1 - x^a)/a - (1 - x^(a+1))/(a+1) + 1/2*(1 - x^(a+2))/(a+2) ...
            // Combining consecutive terms gives:
            // ((1 - x^(a+k))/(a+k) - (1 - x^(a+k+1))/(a+k+1)/(k+1))/k!
            // = ((k+1)*(a+k+1)*(1 - x^(a+k)) - (a+k)*(1 - x^(a+k+1)))/k!/(k+1)/(a+k)/(a+k+1)
            // = ((k+1)*(a+k+1) - (k+1)*(a+k+1)*x^(a+k) - (a+k) + (a+k)*x^(a+k+1))/k!/(k+1)/(a+k)/(a+k+1)
            // = (k*(a+k+1) + 1 + ((a+k)*x - (a+k+1)*(k+1))*x^(a+k))/k!/(k+1)/(a+k)/(a+k+1)
            // the coefficient of x^(a+k) tends toward -1/k!/(a+k)
            // k=0 gives (1 + (a*x - a - 1)*x^a)/a/(a+1)  if a != 0 and a != -1
            double alogx = a * logx;
            double xPowerA = Math.Exp(alogx);
            if (xPowerA > double.MaxValue && a < 0) return xPowerA;
            double sum;
            if (Math.Abs(alogx) < 1e-16)
            {
                // abs(x) < 1e-16 implies ExpMinus1(x) == x
                sum = -logx;
            }
            else
            {
                double xaMinus1 = MMath.ExpMinus1(alogx);
                sum = -xaMinus1 / a;
                //sum = (1 - xPowerA) / a;
            }
            double sign = 1;
            double factorial = 1;
            for (int i = 1; i < 1000; i++)
            {
                xPowerA *= x;
                sign *= -1;
                factorial *= i;
                double sumOld = sum;
                double term;
                if (a + i == 0)
                {
                    term = -logx;
                }
                else
                {
                    term = (1 - xPowerA) / (a + i);
                }
                sum += sign * term / factorial;
                if (AreEqual(sum, sumOld))
                {
                    return regularized ? sum / MMath.Gamma(a) : sum;
                }
            }
            throw new Exception($"GammaUpperSeries1 not converging for a={a:g17} x={x:g17}");
        }

        /// <summary>
        /// Compute int_x^1 t^(a-1) sum_{k=0}^inf (-b*t)^k/k! dt by series expansion.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logx"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static double GammaUpperSeries1(double a, double b, double logx)
        {
            // The starting point is exp(-b*t) = sum_{k=0}^inf (-b*t)^k/k!
            // Substituting gives sum_{k=0}^inf int_x^1 t^(a-1) (-b*t)^k/k! dt
            // = sum_{k=0}^inf (-b)^k/k! int_x^1 t^(a+k-1) dt
            // = sum_{k=0}^inf (-b)^k/k! ((1 - x^(a+k))/(a+k) 1(a+k != 0) -log(x) 1(a+k == 0))
            // = (1 - x^a)/a - b*(1 - x^(a+1))/(a+1) + b^2/2*(1 - x^(a+2))/(a+2) ...
            if (logx >= 0) return 0;
            double alogx = a * logx;
            double xPowerA = Math.Exp(alogx);
            if (xPowerA > double.MaxValue && a < 0) return xPowerA;
            double sum;
            if (Math.Abs(alogx) < 1e-16)
            {
                // abs(x) < 1e-16 implies ExpMinus1(x) == x
                sum = -logx;
            }
            else
            {
                double xaMinus1 = MMath.ExpMinus1(alogx);
                sum = -xaMinus1 / a;
                //sum = (1 - xPowerA) / a;
            }
            double sign = 1;
            double factorial = 1;
            double x = Math.Exp(logx);
            for (int i = 1; i < 1000; i++)
            {
                xPowerA *= x;
                sign *= -b;
                factorial *= i;
                double sumOld = sum;
                double term;
                if (a + i == 0)
                {
                    term = -Math.Log(x);
                }
                else
                {
                    term = (1 - xPowerA) / (a + i);
                }
                sum += sign * term / factorial;
                if (AreEqual(sum, sumOld))
                {
                    return sum;
                }
            }
            throw new Exception($"GammaUpperSeries1 not converging for a={a:g17} x={x:g17}");
        }

        /// <summary>
        /// Compute the upper incomplete Gamma function by a series expansion
        /// </summary>
        /// <param name="a">The shape parameter, &gt; 0</param>
        /// <param name="x">The lower bound of the integral, &gt;= 0</param>
        /// <param name="regularized">If true, result is divided by Gamma(a)</param>
        /// <returns></returns>
        private static double GammaUpperSeries(double a, double x, bool regularized = true)
        {
            // this series should only be applied when x is small
            // the regularized series is: 1 - x^a/Gamma(a) sum_{k=0}^inf (-x)^k /(k! (a+k))
            // = (1 - 1/Gamma(a+1)) + (1 - x^a)/Gamma(a+1) - x^a/Gamma(a) sum_{k=1}^inf (-x)^k/(k! (a+k))
            // The unregularized series is:
            // = (Gamma(a) - 1/a) + (1 - x^a)/a - x^a sum_{k=1}^inf (-x)^k/(k! (a+k))
            double logx = Math.Log(x);
            double alogx = a * logx;
            double xaMinus1 = MMath.ExpMinus1(alogx);
            double offset, term;
            if (regularized)
            {
                double aReciprocalFactorial, aReciprocalFactorialMinus1;
                if (a > 0.3)
                {
                    aReciprocalFactorial = 1 / MMath.Gamma(a + 1);
                    aReciprocalFactorialMinus1 = aReciprocalFactorial - 1;
                }
                else
                {
                    aReciprocalFactorialMinus1 = ReciprocalFactorialMinus1(a);
                    aReciprocalFactorial = 1 + aReciprocalFactorialMinus1;
                }
                // offset = 1 - x^a/Gamma(a+1)
                offset = -xaMinus1 * aReciprocalFactorial - aReciprocalFactorialMinus1;
                double scale = 1 - offset;
                term = x * a * scale;
            }
            else
            {
                if (Math.Abs(alogx) <= Ulp1) offset = GammaSeries(a) - logx;
                else offset = GammaSeries(a) - xaMinus1 / a;
                double scale = 1 + xaMinus1;
                term = x * scale;
            }
            // Unfortunately this computation is not monotonic in x.
            // Kahan summation can improve the accuracy, but it remains non-monotonic.
            double sum = offset + term / (a + 1);
            for (int i = 1; i < 1000; i++)
            {
                term *= -x / (i + 1);
                double sumOld = sum;
                sum += term / (a + i + 1);
                ////Trace.WriteLine($"{i}: {sum}");
                if (AreEqual(sum, sumOld))
                {
                    return sum;
                }
            }
            throw new Exception($"GammaUpperSeries not converging for a={a:g17} x={x:g17} regularized={regularized}");
        }

        /// <summary>
        /// Compute <c>Gamma(x) - 1/x</c> to high accuracy
        /// </summary>
        /// <param name="x">A real number &gt;= 0</param>
        /// <returns></returns>
        private static double GammaSeries(double x)
        {
            if (x > GammaSmallX)
                return MMath.Gamma(x) - 1 / x;
            else
                return
                    // Truncated series 19: Gamma(x) - 1/x
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 0.99810569378312897*x**7 when x >= 0
                    // which is at most Ulp(0.57721566490153287)/2 when 0 <= x <= 0.0048617874947108133
                    -0.57721566490153287 +
                    x * (0.9890559953279725 +
                    x * (-0.90747907608088629 +
                    x * (0.98172808683440016 +
                    x * (-0.98199506890314525 +
                    x * (0.9931491146212762 +
                    x * -0.99600176044243149)))))
                    ;
        }

        /// <summary>
        /// Computes <c>1/Gamma(x+1) - 1</c> to high accuracy
        /// </summary>
        /// <param name="x">A real number &gt;= 0</param>
        /// <returns></returns>
        public static double ReciprocalFactorialMinus1(double x)
        {
            if (x > 0.3)
                return 1 / MMath.Gamma(x + 1) - 1;

            return
                // Truncated series 18: Reciprocal factorial minus 1
                // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                x * (0.577215664901532860606512090082 +
                x * (-0.655878071520253881077019515145 +
                x * (-0.0420026350340952355290039348754 +
                x * (0.166538611382291489501700795102 +
                x * (-0.0421977345555443367482083012891 +
                x * (-0.00962197152787697356211492167231 +
                x * (0.00721894324666309954239501034044 +
                x * (-0.00116516759185906511211397108402 +
                x * (-0.000215241674114950972815729963027 +
                x * (0.000128050282388116186153198626338 +
                x * (-0.0000201348547807882386556893913662 +
                x * (-0.00000125049348214267065734535945301 +
                x * (0.00000113302723198169588237412961344 +
                x * (-0.000000205633841697760710345015364430 +
                x * (0.00000000611609510448141581786252410878 +
                x * 0.00000000500200764446922293005568923640)))))))))))))))
                ;
        }

        /// <summary>
        /// Computes <c>x^a e^(-x)/Gamma(a)</c> to high accuracy.
        /// </summary>
        /// <param name="a">A positive real number</param>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double GammaUpperScale(double a, double x)
        {
            return Math.Exp(GammaUpperLogScale(a, x));
        }

        /// <summary>
        /// Computes <c>log(x^a e^(-x)/Gamma(a))</c> to high accuracy.
        /// </summary>
        /// <param name="a">A positive real number</param>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double GammaUpperLogScale(double a, double x)
        {
            if (double.IsPositiveInfinity(x) || double.IsPositiveInfinity(a))
                return double.NegativeInfinity;
            if (a < 10)
            {
                return a * Math.Log(x) - x - GammaLn(a);
            }
            else
            {
                // Result is inaccurate for a=100, x=3
                double xOverAMinus1 = (x - a) / a;
                double phi = XMinusLog1Plus(xOverAMinus1);
                return 0.5 * Math.Log(a) - MMath.LnSqrt2PI - GammaLnSeries(a) - a * phi;
            }
        }

        // Origin: James McCaffrey, http://msdn.microsoft.com/en-us/magazine/dn520240.aspx
        /// <summary>
        /// Compute the regularized upper incomplete Gamma function by a continued fraction
        /// </summary>
        /// <param name="a">A real number.  Must be &gt; 0 if regularized is true.</param>
        /// <param name="x">A real number &gt;= 1.1</param>
        /// <param name="regularized">If true, result is divided by Gamma(a)</param>
        /// <returns></returns>
        private static double GammaUpperConFrac(double a, double x, bool regularized = true)
        {
            double scale = regularized ? GammaUpperScale(a, x) : Math.Exp(a * Math.Log(x) - x);
            if (scale == 0)
                return scale;
            if (x > double.MaxValue) return 0.0;
            // the confrac coefficients are:
            // a_i = -i*(i-a)
            // b_i = x+1-a+2*i
            // the confrac is evaluated using Lentz's algorithm
            double b = x - a + 1.0;
            const double tiny = 1e-30;
            double c = 1.0 / tiny;
            double d = b;
            double h = scale / d;
            for (int i = 1; i < 1000; ++i)
            {
                double an = -i * (i - a);
                b += 2.0;
                d = an / d + b;
                if (Math.Abs(d) < tiny)
                    d = tiny;
                c = b + an / c;
                if (Math.Abs(c) < tiny)
                    c = tiny;
                double del = c / d;
                double oldH = h;
                h *= del;
                ////Trace.WriteLine($"h = {h:g17} del = {del:g17}");
                if (AreEqual(h, oldH))
                    return h;
            }
            throw new Exception($"GammaUpperConFrac not converging for a={a:g17} x={x:g17}");
        }

        /// <summary>
        /// Compute the regularized upper incomplete Gamma function by a continued fraction
        /// </summary>
        /// <param name="a">A real number.  Must be &gt; 0 if regularized is true.</param>
        /// <param name="x">A real number &gt;= 1.1</param>
        /// <param name="regularized">If true, result is divided by Gamma(a)</param>
        /// <returns></returns>
        private static double GammaUpperConFrac2(double a, double x, bool regularized = true)
        {
            // Origin: Gautschi (1979)
            double scale = regularized ? GammaUpperScale(a, x) : Math.Exp(a * Math.Log(x) - x);
            ////Trace.WriteLine($"{scale:g17}");
            if (scale == 0)
                return scale;
            if (x > double.MaxValue) return 0.0;
            double p = 0;
            double q = (x - 1 - a) * (x + 1 - a);
            double r = 4 * (x + 1 - a);
            double s = 1 - a;
            double rho = 0;
            double t = scale / (x + 1 - a);
            double sum = t;
            for (int i = 1; i < 1000; ++i)
            {
                p += s; // p = -i*(a-i)
                q += r; // q = (x + 2*i - 1 - a) * (x + 2*i + 1 - a)
                r += 8; // r = 4 * (x + 2*i + 1 - a)
                s += 2; // s = 2*i + 1 - a
                double tau = p * (1 + rho);
                rho = tau / (q - tau);
                if (AreEqual(rho, 0))
                    return sum;
                t *= rho;
                double oldSum = sum;
                sum += t;
                ////Trace.WriteLine($"sum={sum} t={t} rho={rho} p={p} q={q} r={r} s={s}");
                if (AreEqual(sum, oldSum))
                    return sum;
            }
            throw new Exception($"GammaUpperConFrac not converging for a={a:g17} x={x:g17}");
        }

        /// <summary>
        /// Computes <c>GammaLn(x) - (x-0.5)*log(x) + x - 0.5*log(2*pi)</c> for x &gt;= 10
        /// </summary>
        /// <param name="x">A real number &gt;= 10</param>
        /// <returns></returns>
        internal static double GammaLnSeries(double x)
        {
            // GammaLnSeries(10) = 0.008330563433362871
            if (x < GammaLnLargeX)
            {
                return MMath.GammaLn(x) - (x - 0.5) * Math.Log(x) + x - LnSqrt2PI;
            }
            else
            {
                // the series is:  sum_{i=1}^inf B_{2i} / (2i*(2i-1)*x^(2i-1))
                double invX = 1.0 / x;
                double invX2 = invX * invX;
                double sum = invX * (
                    // Truncated series 9: GammaLn asymptotic
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 0.029550653594771242*invX2**7 when invX2 >= 0
                    // which is at most Ulp(0.083333333333333329)/2 when 0 <= invX2 <= 0.0060966962219634185
                    1.0 / 12.0 +
                    invX2 * (-1.0 / 360.0 +
                    invX2 * (1.0 / 1260.0 +
                    invX2 * (-1.0 / 1680.0 +
                    invX2 * (1.0 / 1188.0 +
                    invX2 * (-691.0 / 360360.0 +
                    invX2 * (1.0 / 156.0
                    ))))))
                    );
                return sum;
            }
        }

        /// <summary>
        /// Computes the inverse of the digamma function, i.e.
        /// <c>Digamma(DigammaInv(y)) == y</c>
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        internal static double DigammaInv(double y)
        {
            // Newton iteration to solve digamma(x)-y = 0
            double x;
            if (y <= -2.22)
                x = -1 / (y - Digamma1);
            else
                x = Math.Exp(y) + 0.5;

            // should never need more than 5 iterations
            int maxIter = 5;
            for (int iter = 0; iter < maxIter; iter++)
            {
                x -= (Digamma(x) - y) / Trigamma(x);
            }
            return x;
        }

        #endregion

        #region Erf functions

        /// <summary>
        /// Computes the complementary error function. This function is defined by 2/sqrt(pi) * integral from x to infinity of exp (-t^2) dt.
        /// </summary>
        /// <param name="x">Any real value.</param>
        /// <returns>The complementary error function at x.</returns>
        public static double Erfc(double x)
        {
            return 2 * NormalCdf(-MMath.Sqrt2 * x);
        }

        /// <summary>
        /// Computes the inverse of the complementary error function, i.e.
        /// <c>erfcinv(erfc(x)) == x</c>.
        /// </summary>
        /// <param name="y">A real number between 0 and 2.</param>
        /// <returns>A number x such that <c>erfc(x) == y</c>.</returns>
        public static double ErfcInv(double y)
        {
            const double small = 0.0485, large = 1.9515;
            // check for boundary cases
            if (y < 0 || y > 2)
                throw new ArgumentOutOfRangeException(nameof(y),
                                                      y,
                                                      "function not defined outside [0,2].");

            if (y == 0)
                return double.PositiveInfinity;
            else if (y == 1)
                return 0.0;
            else if (y == 2)
                return double.NegativeInfinity;

            // stores the result
            double x = 0.0;

            // This function uses a polynomial approximation followed by a few steps of Halley's rational method.
            // Origin: unknown

            // Rational approximation for the central region
            if (y >= small && y <= large)
            {
                double q = y - 1.0;
                double r = q * q;
                x = (((((0.01370600482778535 * r - 0.3051415712357203) * r + 1.524304069216834) * r - 3.057303267970988) * r + 2.710410832036097) * r - 0.8862269264526915) * q /
                    (((((-0.05319931523264068 * r + 0.6311946752267222) * r - 2.432796560310728) * r + 4.175081992982483) * r - 3.320170388221430) * r + 1.0);
            }
            // Rational approximation for the lower region
            else if (y < small)
            {
                double q = Math.Sqrt(-2.0 * Math.Log(y / 2.0));
                x = (((((0.005504751339936943 * q + 0.2279687217114118) * q + 1.697592457770869) * q + 1.802933168781950) * q + -3.093354679843504) * q - 2.077595676404383) /
                    ((((0.007784695709041462 * q + 0.3224671290700398) * q + 2.445134137142996) * q + 3.754408661907416) * q + 1.0);
            }
            // Rational approximation for the upper region
            else if (y > large)
            {
                double q = Math.Sqrt(-2.0 * Math.Log(1 - y / 2.0));
                x = -(((((0.005504751339936943 * q + 0.2279687217114118) * q + 1.697592457770869) * q + 1.802933168781950) * q + -3.093354679843504) * q - 2.077595676404383) /
                    ((((0.007784695709041462 * q + 0.3224671290700398) * q + 2.445134137142996) * q + 3.754408661907416) * q + 1.0);
            }

            // Iterate Halley's rational method to increase precision (but only up to the precision of Erfc)
            for (int iter = 0; iter < 4; iter++)
            {
                double oldx = x;
                double u = (Erfc(x) - y) / (-2.0 / Math.Sqrt(Math.PI) * Math.Exp(-x * x));
                x = x - u / (1.0 + x * u);
                if (AreEqual(x, oldx))
                    break;
            }

            // done!
            return x;
        }

        /// <summary>
        /// Computes <c>ln(NormalCdf(x)/N(x;0,1))</c> to high accuracy.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns></returns>
        public static double NormalCdfRatioLn(double x)
        {
            if (x > 0)
            {
                return LogDifferenceOfExp(MMath.LnSqrt2PI + 0.5 * x * x, NormalCdfRatioLn(-x));
            }
            else
            {
                return Math.Log(NormalCdfRatio(x));
            }
        }

        /// <summary>
        /// Computes <c>NormalCdf(x)/N(x;0,1)</c> to high accuracy.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns></returns>
        public static double NormalCdfRatio(double x)
        {
            if (x > 0)
            {
                return Sqrt2PI * Math.Exp(0.5 * x * x) - NormalCdfRatio(-x);
            }
            else if (x > -6)
            {
                // Taylor expansion
                // works only for |x| < 17
                Assert.IsTrue(x > -17);
                Assert.IsTrue(x <= 0);
                int j = (int)System.Math.Floor(0.5 - x);
                if (j >= c_normcdf_table.Length)
                    j = c_normcdf_table.Length - 1;
                double y = c_normcdf_table[j];
                return y + NormalCdfRatioDiff_Simple(-j, x + j, y);
            }
            else
            {
                if (double.IsNaN(x))
                    return x;
                Assert.IsTrue(x <= -6);
                double invX = 1 / x;
                double invX2 = invX * invX;
                // Continued fraction approach
                // can also be used for x > -6, but prohibitively many terms would be required.
                double numer = -invX;
                double numerPrev = 0;
                double denom = 1;
                double denomPrev = 1;
                double a = invX2;
                int n;
                if (x < -80)
                    n = 5;
                else if (x < -15)
                    n = 10;
                else
                    n = 20;
                for (int i = 1; i < n; i++)
                {
                    double numerNew = numer + a * numerPrev;
                    double denomNew = denom + a * denomPrev;
                    a += invX2;
                    numerPrev = numer;
                    numer = numerNew;
                    denomPrev = denom;
                    denom = denomNew;
                }
                return numer / denom;
            }
        }

        /// <summary>
        /// Computes <c>NormalCdfRatio(x+delta)-NormalCdfRatio(x)</c> given <c>NormalCdfRatio(x)</c>.
        /// Is inaccurate for x &lt; -17.  For more accuracy, use NormalCdfRatioDiff.
        /// </summary>
        /// <param name="x">Any real number</param>
        /// <param name="delta">A real number with absolute value less than or equal to 0.5</param>
        /// <param name="y">NormalCdfRatio(x)</param>
        /// <returns></returns>
        private static double NormalCdfRatioDiff_Simple(double x, double delta, double y)
        {
            if (Math.Abs(delta) > 0.5)
                throw new ArgumentException("delta > 0.5");
            // Algorithm: Taylor expansion of NormalCdfRatio based on upward recurrence.
            // Based on Marsaglia's algorithm for cPhi.
            // Reference:
            // "Evaluating the Normal Distribution"
            // Journal of Statistical Software
            // Volume 11, 2004, Issue 5  
            // http://www.jstatsoft.org/v11/i05
            // Reven = deriv(R,i)(z)/i! for even i
            // Rodd = deriv(R,i+1)(z)/(i+1)! for even i
            // deriv(R,i+2)(z) = z*deriv(R,i+1)(z) + (i+1)*deriv(R,i)(z)
            // deriv(R,i+2)(z)/(i+2)! = (z*deriv(R,i+1)(z)/(i+1)! + deriv(R,i)(z)/i!)/(i+2)
            double Reven = y;
            double Rodd = Reven * x + 1;
            double delta2 = delta * delta;
            double sum = delta * Rodd;
            double oldSum;
            double pwr = 1;
            for (int i = 2; i < 1000; i += 2)
            {
                Reven = (Reven + x * Rodd) / i;
                Rodd = (Rodd + x * Reven) / (i + 1);
                pwr *= delta2;
                oldSum = sum;
                sum += pwr * (Reven + delta * Rodd);
                if (AreEqual(sum, oldSum)) break;
            }
            return sum;
        }

        /// <summary>
        /// Computes <c>NormalCdfRatio(x+delta)-NormalCdfRatio(x)</c> 
        /// </summary>
        /// <param name="x">A real number less than 37</param>
        /// <param name="delta">A finite real number with absolute value less than 9.9, or less than 70% of the absolute value of x.</param>
        /// <param name="startingIndex">The first moment to use in the power series.  Used to skip leading terms.  For example, 2 will skip NormalCdfMomentRatio(1, x).</param>
        /// <returns></returns>
        public static double NormalCdfRatioDiff(double x, double delta, int startingIndex = 1)
        {
            if (startingIndex < 1) throw new ArgumentOutOfRangeException(nameof(startingIndex), "startingIndex < 1");
            // NormalCdfRatio will overflow for large x
            if (x >= 37) throw new ArgumentOutOfRangeException(nameof(x), x, "x >= 37");
            if (double.IsInfinity(delta)) throw new ArgumentOutOfRangeException(nameof(delta), "delta is infinite");
            // This code is adapted from NormalCdfMomentRatioTaylor
            double sum = 0;
            double term = delta;
            var iter = NormalCdfMomentRatioSequence(startingIndex, x);
            for (int i = 0; i < 1000; i++)
            {
                double sumOld = sum;
                iter.MoveNext();
                double deriv = iter.Current;
                if (deriv == 0 || term == 0) return sum;
                sum += deriv * term;
                if (double.IsNaN(sum)) throw new Exception($"sum is NaN for x={x}, delta={delta}");
                //Console.WriteLine("{0}: {1}", i, sum);
                if (AreEqual(sum, sumOld)) return sum;
                term *= delta;
            }
            throw new Exception($"Not converging for x={x}, delta={delta}");
        }

        /// <summary>
        /// Computes <c>NormalCdfMomentRatio(n, x+delta)-NormalCdfMomentRatio(n, x)</c>
        /// </summary>
        /// <param name="n">A non-negative integer</param>
        /// <param name="x">Any real number</param>
        /// <param name="delta">A finite real number with absolute value less than 9.9, or less than 70% of the absolute value of x.</param>
        /// <param name="startingIndex">The first moment to use in the power series.  Used to skip leading terms.  For example, 2 will skip NormalCdfMomentRatio(n+1, x).</param>
        /// <returns></returns>
        public static double NormalCdfMomentRatioDiff(int n, double x, double delta, int startingIndex = 1)
        {
            if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "n < 0");
            if (startingIndex < 1) throw new ArgumentOutOfRangeException(nameof(startingIndex), "startingIndex < 1");
            if (double.IsInfinity(delta)) throw new ArgumentOutOfRangeException(nameof(delta), "delta is infinite");
            // n=1: delta*2!/1!*R(2,x) + delta^2*3!/2!*R(3,x) + ...
            // n=2: delta*3!/1!*R(3,x) + delta^2*4!/2!*R(4,x) + ...
            double sum = 0;
            double term = delta * Gamma(n + 1 + startingIndex) / Gamma(1 + startingIndex);
            var iter = NormalCdfMomentRatioSequence(n + startingIndex, x);
            for (int i = startingIndex - 1; i < 1000; i++)
            {
                double sumOld = sum;
                iter.MoveNext();
                double deriv = iter.Current;
                if (deriv == 0 || term == 0) return sum;
                sum += deriv * term;
                //Console.WriteLine("{0}: {1}", i, sum);
                if (AreEqual(sum, sumOld))
                    return sum;
                term *= delta * (n + i + 2) / (i + 2);
            }
            throw new Exception($"Not converging for n={n}, x={x}, delta={delta}");
        }

        /// <summary>
        /// Computes the cumulative Gaussian distribution, defined as the
        /// integral from -infinity to x of N(t;0,1) dt.  
        /// For example, <c>NormalCdf(0) == 0.5</c>.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns>The cumulative Gaussian distribution at <paramref name="x"/>.</returns>
        public static double NormalCdf(double x)
        {
            if (double.IsNaN(x))
                return x;
            else if (x > 10)
                return 1.0;
            else if (x < -40)
                return 0.0;
            else if (false && x > -4)
            {
                // Marsaglia's algorithm:
                // NormalCdf(x) = 0.5 + exp(-x^2/2)/sqrt(2*pi)*s
                // where s = x + x^3/3 + x^5/(3*5) + x^7/(3*5*7) + ...
                // http://www.jstatsoft.org/v11/i04/v11i04.pdf
                // Not accurate for x < -4 or x > 10.
                Assert.IsTrue(x >= -4);
                Assert.IsTrue(x <= 10);
                double s = x, t = 0, b = x, x2 = x * x;
                //int count = 0;
                for (int i = 3; s != t; i += 2)
                {
                    t = s;
                    b *= x2 / i;
                    s += b;
                    //count++;
                }
                //Console.WriteLine("count = {0}", count);
                double result = .5 + s * Math.Exp(-.5 * x2) * InvSqrt2PI;
                if (result < 0)
                    result = 0.0;
                else if (result > 1)
                    result = 1.0;
                return result;
            }
            else if (x > 0)
            {
                return 1 - NormalCdf(-x);
            }
            else
            {
                // we could use GammaUpper to compute this via:
                // return 0.5 * GammaUpper(0.5, x * x / 2);
                // however, this is less accurate because GammaUpper uses the Gamma function 
                // which only gets 14 digits of accuracy.
                double s = NormalCdfRatio(x);
                s *= Math.Exp(-0.5 * x * x) / Sqrt2PI;
                return s;
            }
        }

        /// <summary>
        /// The natural logarithm of the cumulative Gaussian distribution.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns>ln(NormalCdf(x)).</returns>
        /// <remarks>This function provides higher accuracy than <c>Math.Log(NormalCdf(x))</c>, which can fail for x &lt; -7.</remarks>
        public static double NormalCdfLn(double x)
        {
            const double large = -10;
            if (x > 4)
            {
                // log(NormalCdf(x)) = log(1-NormalCdf(-x))
                double y = NormalCdf(-x);
                // log(1-y)
                return
                    // Truncated series 11: log(1 - x)
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    y * (-1.0 +
                    y * (-1.0 / 2.0 +
                    y * (-1.0 / 3.0 +
                    y * -1.0 / 4.0)))
                    ;
            }
            else if (x >= large)
            {
                return Math.Log(NormalCdf(x));
            }
            else
            {
                // x < large
                double z = 1 / (x * x);
                double s;
                if (x > -16)
                    // This truncated series provides a good approximation
                    // for x <= -10
                    s =
                        // Truncated series 16: normcdfln asymptotic
                        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                        z * (-1.0 +
                        z * (5.0 / 2.0 +
                        z * (-37.0 / 3.0 +
                        z * (353.0 / 4.0 +
                        z * (-4081.0 / 5.0 +
                        z * (55205.0 / 6.0 +
                        z * (-854197.0 / 7.0 +
                        z * (14876033.0 / 8.0 +
                        z * (-288018721.0 / 9.0 +
                        z * (1227782785.0 / 2.0 +
                        z * (-142882295557.0 / 11.0 +
                        z * (3606682364513.0 / 12.0 +
                        z * (-98158402127761.0 / 13.0 +
                        z * (2865624738913445.0 / 14.0 +
                        z * (-89338394736560917.0 / 15.0
                        )))))))))))))))
                        ;
                else
                    // This truncated series provides a good approximation
                    // for x <= -16
                    s =
                        // Truncated series 16: normcdfln asymptotic
                        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                        z * (-1.0 +
                        z * (5.0 / 2.0 +
                        z * (-37.0 / 3.0 +
                        z * (353.0 / 4.0 +
                        z * (-4081.0 / 5.0 +
                        z * (55205.0 / 6.0 +
                        z * (-854197.0 / 7.0 +
                        z * (14876033.0 / 8.0
                        ))))))))
                        ;
                return s - LnSqrt2PI - 0.5 * x * x - Math.Log(-x);
            }
        }

        /// <summary>
        /// The log-odds of the cumulative Gaussian distribution.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns>ln(NormalCdf(x)/(1-NormalCdf(x))).</returns>
        public static double NormalCdfLogit(double x)
        {
            return NormalCdfLn(x) - NormalCdfLn(-x);
        }

        /// <summary>
        /// Computes the inverse of the cumulative Gaussian distribution,
        /// i.e. <c>NormalCdf(NormalCdfInv(p)) == p</c>.
        /// For example, <c>NormalCdfInv(0.5) == 0</c>.
        /// This is also known as the Gaussian quantile function.  
        /// </summary>
        /// <param name="p">A real number in [0,1].</param>
        /// <returns>A number x such that <c>NormalCdf(x) == p</c>.</returns>
        public static double NormalCdfInv(double p)
        {
            return -Sqrt2 * ErfcInv(2 * p);
        }

        #endregion

        #region NormalCdf functions

        const int NormalCdfMomentRatioTableSize = 200;

        // [0] contains moments for x=-2
        // [1] contains moments for x=-3, etc.
        private static readonly Lazy<double[][]> NormalCdfMomentRatioTable = new Lazy<double[][]>(MakeNormalCdfMomentRatioTable);

        private static double[][] MakeNormalCdfMomentRatioTable()
        {
            return Util.ArrayInit(7, index =>
            {
                double[] derivs = new double[NormalCdfMomentRatioTableSize];
                double x0 = -index - 2;
                var iter = NormalCdfMomentRatioSequence(0, x0, true);
                for (int i = 0; i < NormalCdfMomentRatioTableSize; i++)
                {
                    iter.MoveNext();
                    derivs[i] = iter.Current;
                }
                return derivs;
            });
        }

        const int NormalCdfMomentRatioMaxTerms = 60;

        /// <summary>
        /// Computes int_0^infinity t^n N(t;x,1) dt / (n! N(x;0,1))
        /// </summary>
        /// <param name="n">The exponent</param>
        /// <param name="x">Any real number</param>
        /// <returns></returns>
        public static double NormalCdfMomentRatio(int n, double x)
        {
            if (x >= -0.5)
                return NormalCdfMomentRatioRecurrence(n, x);
            else if (n <= NormalCdfMomentRatioTableSize - NormalCdfMomentRatioMaxTerms && x > -8)
            {
                int index = (int)(-x - 1.5); // index ranges from 0 to 6
                double x0 = -index - 2;
                return NormalCdfMomentRatioTaylor(n, x - x0, NormalCdfMomentRatioTable.Value[index]);
            }
            else if (x > -2)
            {
                // x is between -2 and -1
                // use Taylor from -2
                double x0 = -2;
                return NormalCdfMomentRatioTaylor(n, x - x0, x0);
            }
            else
                return NormalCdfMomentRatioConFrac(n, x);
        }

        /// <summary>
        /// Computes <c>int_0^infinity t^n N(t;x,1) dt / (n! N(x;0,1))</c>
        /// </summary>
        /// <param name="n">The exponent</param>
        /// <param name="x">A real number &gt; -1</param>
        /// <returns></returns>
        private static double NormalCdfMomentRatioRecurrence(int n, double x)
        {
            double rPrev = MMath.NormalCdfRatio(x);
            if (n == 0)
                return rPrev;
            double r = x * rPrev + 1;
            for (int i = 1; i < n; i++)
            {
                double rNew = (x * r + rPrev) / (i + 1);
                rPrev = r;
                r = rNew;
            }
            return r;
        }

        /// <summary>
        /// Compute <c>int_0^infinity t^n N(t;x,1) dt / (n! N(x;0,1))</c> by Taylor expansion around <c>x0</c>
        /// </summary>
        /// <param name="n"></param>
        /// <param name="delta"></param>
        /// <param name="x0"></param>
        /// <returns></returns>
        private static double NormalCdfMomentRatioTaylor(int n, double delta, double x0)
        {
            double sum = 0;
            double term = 1;
            var iter = NormalCdfMomentRatioSequence(n, x0);
            for (int i = n; i < n + 1000; i++)
            {
                double sumOld = sum;
                iter.MoveNext();
                double deriv = iter.Current;
                sum += deriv * term;
                //Console.WriteLine("{0}: {1}", i, sum);
                if (AreEqual(sum, sumOld))
                    return sum;
                term *= delta * (i + 1) / (i - n + 1);
            }
            throw new Exception("Not converging for n=" + n + ",delta=" + delta);
        }

        /// <summary>
        /// Compute <c>int_0^infinity t^n N(t;x,1) dt / (n! N(x;0,1))</c> by Taylor expansion given pre-computed derivatives
        /// </summary>
        /// <param name="n"></param>
        /// <param name="delta"></param>
        /// <param name="derivs"></param>
        /// <returns></returns>
        private static double NormalCdfMomentRatioTaylor(int n, double delta, double[] derivs)
        {
            // derivs[i] is already divided by i!, and the result is divided by n!
            double sum = 0;
            double term = 1;
            for (int i = n; i < derivs.Length; i++)
            {
                double sumOld = sum;
                sum += derivs[i] * term;
                //Console.WriteLine("{0}: {1}", i, sum);
                if (AreEqual(sum, sumOld))
                    return sum;
                term *= delta * (i + 1) / (i - n + 1);
            }
            throw new Exception("Not converging for n=" + n + ",delta=" + delta);
        }

        /// <summary>
        /// Computes int_0^infinity t^n N(t;x,1) dt / (n! N(x;0,1))
        /// </summary>
        /// <param name="n">The exponent</param>
        /// <param name="x">A real number &lt;= -1</param>
        /// <returns></returns>
        private static double NormalCdfMomentRatioConFrac(int n, double x)
        {
            double invX = 1 / x;
            double invX2 = invX * invX;
            double numer = -invX;
            double numerPrev = 0;
            // denom and denomPrev are always > 0
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
            if (numer == 0) return 0;
            // a = (n+1)*invX2
            double rOld = 0;
            const double smallestNormalized = 1e-308;
            const double smallestNormalizedOverEpsilon = smallestNormalized / double.Epsilon;
            // the number of iterations required may grow with n, so we need to explicitly test for convergence.
            for (int i = 0; i < 10000; i++)
            {
                double numerNew = numer + a * numerPrev;
                double denomNew = denom + a * denomPrev;
                a += invX2;
                // Rescale to avoid overflow or underflow.
                // Let c=1e-308 be the smallest normalized number.
                // We want to stay above this magnitude if possible.
                // Thus we want to ensure that after rescaling, (|numerPrev|<c)==(|numer|<c)
                // We know that |numerNew| > |numer| > 0 at this point.
                // Thus we need |numer|*s >= c or |numerNew|*s < c, the latter only if |numerNew|/denomNew < double.Epsilon.
                // Thus s >= c/|numer| and s >= c/double.Epsilon/denomNew.
                // c/|numer| > c/double.Epsilon/denomNew when double.Epsilon*denomNew > |numer|
                double s;
                if (double.Epsilon * denomNew > Math.Abs(numer))
                    s = smallestNormalized / Math.Abs(numer);
                else
                    s = smallestNormalizedOverEpsilon / denomNew;
                numerPrev = numer * s;
                numer = numerNew * s;
                denomPrev = denom * s;
                denom = denomNew * s;
                double r = numer / denom;
                if (AreEqual(r, rOld))
                    return r;
                rOld = r;
            }
            throw new Exception($"Not converging for n={n},x={x:g17}");
        }

        /// <summary>
        /// Computes NormalCdf(x) - NormalCdf(y) to high accuracy.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>The difference.</returns>
        public static ExtendedDouble NormalCdfDiff(double x, double y)
        {
            double delta = x - y;
            if (delta <= 0)
            {
                return ExtendedDouble.Zero();
            }
            if (x > -y)
            {
                //   NormalCdf(x) - NormalCdf(y)
                // = (1 - NormalCdf(-x)) - (1 - NormalCdf(-y))
                // = NormalCdf(-y) - NormalCdf(-x)
                return NormalCdfDiff(-y, -x);
            }
            // we know that x <= -y so x + y <= 0
            if (delta > 1)
            {
                return NormalCdfExtended(x) - NormalCdfExtended(y);
                //double quadDiff = (x + y) * delta / 2;
                //return NormalCdfRatio(x) - NormalCdfRatio(x - delta) * Math.Exp(quadDiff);
            }
            else
            {
                //   NormalCdf(x) - NormalCdf(x-delta)
                // = N(x;0,1) R(x) - N(x-delta;0,1) R(x-delta)
                // = N(x;0,1) (R(x) - N(x-delta;0,1)/N(x;0,1) R(x-delta))
                // = N(x;0,1) (R(x) - R(x-delta) + (1 - N(x-delta;0,1)/N(x;0,1)) R(x))
                // N(x-delta;0,1)/N(x;0,1) = exp(-(x-delta)^2/2 + x^2/2) = exp((x+x-delta)*delta/2)
                double Rx = NormalCdfRatio(y);
                double Rdiff = NormalCdfRatioDiff(y, delta);
                // both Rx and Rdiff are bounded.
                double OneMinusProbRatio = -ExpMinus1((x + y) * delta / 2);
                return new ExtendedDouble(Rdiff + OneMinusProbRatio * Rx, Gaussian.GetLogProb(x, 0, 1));
            }
        }

        /// <summary>
        /// Computes the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <returns><c>phi(x,y,r)</c></returns>
        /// <remarks>
        /// The cumulative bivariate normal distribution is defined as
        /// <c>int_(-inf)^x int_(-inf)^y N([x;y],[0;0],[1 r; r 1]) dx dy</c>
        /// where <c>N([x;y],[0;0],[1 r; r 1]) = exp(-0.5*(x^2+y^2-2*x*y*r)/(1-r^2))/(2*pi*sqrt(1-r^2))</c>.
        /// </remarks>
        public static double NormalCdf(double x, double y, double r)
        {
            return NormalCdfExtended(x, y, r).ToDouble();
        }

        /// <summary>
        /// Computes the cumulative Gaussian distribution, defined as the
        /// integral from -infinity to x of N(t;0,1) dt.  
        /// For example, <c>NormalCdf(0) == 0.5</c>.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns>The cumulative Gaussian distribution at <paramref name="x"/>.</returns>
        public static ExtendedDouble NormalCdfExtended(double x)
        {
            if (x < 0)
            {
                return new ExtendedDouble(NormalCdfRatio(x), Gaussian.GetLogProb(x, 0, 1));
            }
            else
            {
                return new ExtendedDouble(NormalCdf(x), 0);
            }
        }

        /// <summary>
        /// Computes the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <returns><c>phi(x,y,r)</c></returns>
        /// <remarks>
        /// The cumulative bivariate normal distribution is defined as
        /// <c>int_(-inf)^x int_(-inf)^y N([x;y],[0;0],[1 r; r 1]) dx dy</c>
        /// where <c>N([x;y],[0;0],[1 r; r 1]) = exp(-0.5*(x^2+y^2-2*x*y*r)/(1-r^2))/(2*pi*sqrt(1-r^2))</c>.
        /// </remarks>
        public static ExtendedDouble NormalCdfExtended(double x, double y, double r)
        {
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r  
            double sqrtomr2 = Math.Sqrt(omr2);
            return NormalCdf(x, y, r, sqrtomr2);
        }

        public static ExtendedDouble NormalCdf(double x, double y, double r, double sqrtomr2)
        {
            if (Double.IsNegativeInfinity(x) || Double.IsNegativeInfinity(y))
            {
                return ExtendedDouble.Zero();
            }
            else if (Double.IsPositiveInfinity(x))
            {
                return NormalCdfExtended(y);
            }
            else if (Double.IsPositiveInfinity(y))
            {
                return NormalCdfExtended(x);
            }
            else if (r == 0)
            {
                return NormalCdfExtended(x) * NormalCdfExtended(y);
            }
            else if (sqrtomr2 == 0)
            {
                if (r == 1)
                {
                    return NormalCdfExtended(Math.Min(x, y));
                }
                else // (r == -1)
                {
                    return NormalCdfDiff(x, -y);
                }
            }
            // at this point, both x and y are finite.
            // swap to ensure |x| > |y|
            if (Math.Abs(y) > Math.Abs(x))
            {
                double t = x;
                x = y;
                y = t;
            }
            // ensure x <= 0
            if (x > 0)
            {
                if (x + y > 1e-4)
                {
                    // phi(x,y,r) = phi(inf,y,r) - phi(-x,y,-r)
                    // This is only safe when phi(y) - phi(-x) is sufficiently large.  This difference can be approximated by N(x;0,1)*(y+x).
                    // Thus we need (y+x) to be sufficiently large.
                    return (NormalCdfExtended(y) - NormalCdf(-x, y, -r, sqrtomr2)).Max(0);
                }
                else if (x - r * y > 0)
                {
                    // phi(x,y,r) = phi(-x,-y,r) + phi(x,y,-1)
                    // recursive call has x-ry < 0
                    return NormalCdf(-x, -y, r, sqrtomr2) + NormalCdf(x, y, -1, 0);
                }
            }
            // avoid the problematic region
            if (r > NormalCdf_Helper_maxR && x > NormalCdf_Helper_maxX)
            {
                // phi(x,y,r) = phi(x,inf,r) - phi(x,-y,-r)
                return (NormalCdfExtended(x) - NormalCdf(x, -y, -r, sqrtomr2)).Max(0);
            }
            return NormalCdf_Helper(x, y, r, sqrtomr2);
        }

        /// <summary>
        /// Computes the natural logarithm of the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <returns><c>ln(phi(x,y,r))</c></returns>
        public static double NormalCdfLn(double x, double y, double r)
        {
            return NormalCdfExtended(x, y, r).Log();
        }

        /// <summary>
        /// Computes the natural logarithm of the cumulative bivariate normal distribution, 
        /// minus the log-density of the bivariate normal distribution at x and y,
        /// plus 0.5*log(1-r*r).
        /// </summary>
        /// <param name="x">First upper limit. Must be finite.</param>
        /// <param name="y">Second upper limit. Must be finite.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <param name="sqrtomr2">sqrt(1-r*r)</param>
        /// <returns><c>ln(phi(x,y,r)/N([x;y],[0;0],[1 r; r 1])</c></returns>
        public static double NormalCdfRatioLn(double x, double y, double r, double sqrtomr2)
        {
            // The log-density of the bivariate normal distribution at x and y can be written in two equivalent ways:
            // Gaussian.GetLogProb(x,0,1)+Gaussian.GetLogProb(y-r*x,0,1-r*r)
            // Gaussian.GetLogProb(y,0,1)+Gaussian.GetLogProb(x-r*y,0,1-r*r)
            if (Double.IsNegativeInfinity(x) || Double.IsNegativeInfinity(y))
            {
                throw new NotImplementedException();
            }
            else if (Double.IsPositiveInfinity(x) || Double.IsPositiveInfinity(y))
            {
                return double.PositiveInfinity;
            }
            else if (r == 0)
            {
                return NormalCdfRatioLn(x) + NormalCdfRatioLn(y);
            }
            else if (r == 1)
            {
                return NormalCdfRatioLn(Math.Min(x, y)) + MMath.LnSqrt2PI;
            }
            else if (r == -1 && sqrtomr2 == 0)
            {
                // In this case, we should subtract log N(y;0,1)
                bool shouldThrow = true;
                if (shouldThrow)
                    throw new NotImplementedException();
                if (x > 0)
                {
                    if (y > 0)
                    {
                        // 1-NormalCdf(-x) + 1-NormalCdf(-y)-1 = 1 - NormalCdf(-x) - NormalCdf(-y)
                        return Log1MinusExp(LogSumExp(NormalCdfLn(-x), NormalCdfLn(-y)));
                    }
                    else
                    {
                        // 1-NormalCdf(-x) + NormalCdf(y) - 1 = NormalCdf(y) - NormalCdf(-x)
                        double nclx = NormalCdfLn(-x);
                        double diff = NormalCdfLn(y) - nclx;
                        if (diff < 0)
                            return double.NegativeInfinity;
                        return LogExpMinus1(diff) + nclx;
                    }
                }
                else
                {
                    if (y > 0)
                    {
                        // NormalCdf(x) - NormalCdf(-y)
                        if (x < -y)
                            return double.NegativeInfinity;
                        double ncly = NormalCdfLn(-y);
                        double diff = NormalCdfLn(x) - ncly;
                        if (diff < 0)
                            return double.NegativeInfinity;
                        return LogExpMinus1(diff) + ncly;
                    }
                    else
                    {
                        // x < 0 and y < 0
                        return double.NegativeInfinity;
                    }
                }
            }
            // at this point, both x and y are finite.
            // swap to ensure |x| > |y|
            if (Math.Abs(y) > Math.Abs(x))
            {
                double t = x;
                x = y;
                y = t;
            }
            double logOffset = double.NegativeInfinity;
            double scale = 1;
            double omr2 = sqrtomr2 * sqrtomr2;
            // ensure x <= 0
            if (x > 0)
            {
                // phi(x,y,r) = phi(inf,y,r) - phi(-x,y,-r)
                double xmry = GetXMinusRY(x, y, r, omr2) / sqrtomr2;
                logOffset = MMath.NormalCdfRatioLn(y) - Gaussian.GetLogProb(xmry, 0, 1);
                scale = -1;
                x = -x;
                r = -r;
            }
            // ensure r <= 0
            if (r > 0)
            {
                // phi(x,y,r) = phi(x,inf,r) - phi(x,-y,-r)
                double ymrx = GetXMinusRY(y, x, r, omr2) / sqrtomr2;
                double logOffset2 = MMath.NormalCdfRatioLn(x) - Gaussian.GetLogProb(ymrx, 0, 1);
                if (scale == 1)
                    logOffset = logOffset2;
                else
                {
                    // the difference here must always be positive since y > -x
                    // offset -= offset2;
                    // logOffset = log(exp(logOffset) - exp(logOffset2))
                    logOffset = MMath.LogDifferenceOfExp(logOffset, logOffset2);
                }
                scale *= -1;
                y = -y;
                r = -r;
            }
            double logResult = NormalCdf_Helper(x, y, r, sqrtomr2, true).Log();
            if (scale == -1)
                return MMath.LogDifferenceOfExp(logOffset, logResult);
            else
                return MMath.LogSumExp(logOffset, logResult);
        }

        private const double NormalCdf_Helper_maxR = 0.5;
        private const double NormalCdf_Helper_maxX = -1.6;

        // factor out the dominant terms and then call confrac
        private static ExtendedDouble NormalCdf_Helper(double x, double y, double r, double sqrtomr2, bool ratio = false)
        {
            double exponent;
            if (ratio)
                exponent = 0;
            else
                exponent = Gaussian.GetLogProb(x, 0, 1);
            double omr2 = sqrtomr2 * sqrtomr2;
            double scale;
            double ymrx = GetXMinusRY(y, x, r, omr2) / sqrtomr2;
            if (ymrx < 0)
            {
                // since phi(ymrx) will be small, we factor N(ymrx;0,1) out of the confrac
                if (!ratio)
                    exponent += Gaussian.GetLogProb(ymrx, 0, 1);
                scale = 1;
            }
            else
            {
                // leave N(ymrx;0,1) in the confrac
                double logProb = Gaussian.GetLogProb(ymrx, 0, 1);
                if (ratio)
                    exponent -= logProb;
                scale = Math.Exp(logProb);
            }
            // This threshold is set using SpecialFunctionsTests.NormalCdf2Test2
            // For debugging, see SpecialFunctionsTests.NormalCdf2Test3
            if (x < -2 || (x < -1 && r <= -0.48) || (r == -1 && x > 2))
            {
                // For x == -1.3, this should not be used when -0.2 < r < 0.87
                // For x == -1.2, this should not be used when -0.3 < r < 0.96
                // For x == -1.1, this should not be used when -0.35 < r < 0.98
                // For x == -1.0, this should not be used
                return new ExtendedDouble(NormalCdfRatioConFrac(x, y, r, scale, sqrtomr2, false, true), exponent);
            }
            else if ((omr2 > 1 - 0.51 * 0.51) || (omr2 > 1 - 0.6 * 0.6 && x > -1))
            {
                // Empirical results from SpecialFunctionsTests.NormalCdf2Test2:
                // Should never be used for x < -2
                // For x == -2, this should not be used when omr2 < 1 - 0.51 * 0.51
                // For x == -1.8, this should not be used when omr2 < 1 - 0.52 * 0.52
                // For x == -1.6, this should not be used when omr2 < 1 - 0.54 * 0.54
                // For x == -1, this should not be used when omr2 < 1 - 0.59 * 0.59
                return new ExtendedDouble(NormalCdfRatioTaylor(x, y, r, sqrtomr2) * scale, exponent);
                //return NormalCdfRatioConFrac3(x, y, r, scale);
            }
            else // (x >= -1 || (x >= -2 && r > -0.48)) && (omr2 <= 1 - 0.51 * 0.51) && (omr2 < 1 - 0.6 * 0.6 || x <= -1)
            {    // which implies ((x >= -1 && omr2 <= 1 - 0.51 * 0.51) || (x >= -2 && r >= 0.51)) && (omr2 < 1 - 0.6 * 0.6 || x <= -1)
                // For x <= -1.6, this always works
                // For x == -1.5, this should not be used when r > -0.02
                // For x == -1.4, this should not be used when r > -0.16
                // For x == -1.3, this should not be used when r > -0.27
                // For x == -1.2, this should not be used when omr2 > 1 - 0.37 * 0.37
                // For x == -1.1, this should not be used when omr2 > 1 - 0.4 * 0.4
                // For x == -1.0, this should not be used when r > -0.39
                // For x == -0.9, this should not be used when r > -0.43
                // For x == -0.8, this should not be used when r > -0.45
                // For x == -0.7, this should not be used when r > -0.46
                // For x == -0.6, this should not be used when omr2 > 1 - 0.52 * 0.52
                // For x == -0.5, this should not be used when omr2 > 1 - 0.53 * 0.53
                // For x == -0.4, this should not be used when omr2 > 1 - 0.53 * 0.53
                // For x == -0.3, this should not be used when omr2 > 1 - 0.54 * 0.54
                // For x == -0.1, this should not be used when r > -0.57
                // For x == -0.01, this should not be used when r > -0.59
                //return NormalCdfRatioConFrac2b(x, y, r, scale, sqrtomr2);
                // For x == -2.0, this always works
                // For x == -1.0, this should not be used when -0.1 < r < 0.7
                return new ExtendedDouble(NormalCdfRatioConFrac(x, y, r, scale, sqrtomr2, false), exponent);
            }
        }

        // Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1)
        // requires -1 < x < 0, abs(r) <= 0.6, and x-r*y <= 0 (or equivalently y < -x).
        // Uses Taylor series at r=0.
        internal static double NormalCdfRatioTaylor(double x, double y, double r, double sqrtomr2)
        {
            if (Math.Abs(x) > 5 || Math.Abs(y) > 5) throw new ArgumentOutOfRangeException();
            // First term of the Taylor series
            double sum = MMath.NormalCdfRatio(x) * MMath.NormalCdfRatio(y);
            double Halfx2y2 = (x * x + y * y) / 2;
            double xy = x * y;
            // Q = NormalCdf(x,y,r)/dNormalCdf(x,y,r)
            // where dNormalCdf(x,y,r) = N(x;0,1) N(y; rx, 1-r^2)
            List<double> Qderivs = new List<double>();
            Qderivs.Add(sum);
            List<double> logphiDerivs = new List<double>();
            double dlogphi = xy;
            logphiDerivs.Add(dlogphi);
            // dQ(0) = 1 - Q(0) d(log(dNormalCdf))
            double Qderiv = 1 - sum * dlogphi;
            Qderivs.Add(Qderiv);
            double rPowerN = r;
            // Second term of the Taylor series
            sum += Qderiv * rPowerN;
            double sumOld = sum;
            int maxIterations = 1000;
            for (int n = 2; n <= maxIterations; n++)
            {
                //Console.WriteLine($"n = {n - 1} sum = {sum:g17}");
                double dlogphiOverFactorial;
                if (n % 2 == 0) dlogphiOverFactorial = 1.0 / n - Halfx2y2;
                else dlogphiOverFactorial = xy;
                logphiDerivs.Add(dlogphiOverFactorial);
                // ddQ = -dQ d(log(dNormalCdf)) - Q dd(log(dNormalCdf)), and so on
                double QderivOverFactorial = 0;
                for (int i = 0; i < n; i++)
                {
                    QderivOverFactorial -= Qderivs[i] * logphiDerivs[n - i - 1] * (n - i) / n;
                }
                Qderivs.Add(QderivOverFactorial);
                rPowerN *= r;
                sum += QderivOverFactorial * rPowerN;
                if ((sum > double.MaxValue) || double.IsNaN(sum) || n >= maxIterations)
                    throw new Exception($"NormalCdfRatioTaylor not converging for x={x:g17}, y={y:g17}, r={r:g17}");
                if (AreEqual(sum, sumOld)) break;
                sumOld = sum;
            }
            return sum / sqrtomr2;
        }

        // Helper function for NormalCdfRatioConFrac
        // Returns (phix + r*phiy)/phir/sqrt(1-r^2)
        private static double NormalCdfRatioConFracNumer(double x, double y, double r, double scale, double sqrtomr2, double xmry, double Rxmry)
        {
            double rPlus1 = GetRPlus1(r, sqrtomr2);
            double omr2 = sqrtomr2 * sqrtomr2;
            if (double.IsInfinity(Rxmry)) throw new ArgumentOutOfRangeException(nameof(Rxmry), Rxmry, "Rxmry is infinite");
            // delta = ymrx - xmry
            double delta = AreEqual(x, y) ? 0 : rPlus1 * (y - x) / sqrtomr2;
            double numer;
            if (Math.Abs(delta) > 0.5)
            {
                /*
from mpmath import *
mp.dps = 500; mp.pretty = True;
vx = mpf('9.9999999999999885E+19');
vl = mpf('1.493910049357609');
vu = mpf('2.0207845396013928');
r = -vx/sqrt(vx+vu)/sqrt(vx+vl);
yl = mpf('-2.2188760015237346E-06');
yu = mpf('2.2178897550353896E-06');
(yu-r*yl)/sqrt(1-r*r)
sqrt(1-r*r)
sqrtomr2 = mpf('0.0018747493845240961');
rr = mpf('-0.99999824265582826');
                 */
                numer = scale * r * Rxmry;
                double ymrx = GetXMinusRY(y, x, r, omr2) / sqrtomr2;
                if (scale == 1)
                    numer += MMath.NormalCdfRatio(ymrx);
                else // this assumes scale = N((y-rx)/sqrt(1-r^2);0,1)
                    numer += MMath.NormalCdf(ymrx);
            }
            else
            {
                numer = scale * (NormalCdfRatioDiff(xmry, delta) + rPlus1 * Rxmry);
            }
            return numer;
        }

        internal static double GetRPlus1(double r, double sqrtomr2)
        {
            if (r >= 0) return r + 1;
            else return sqrtomr2 * sqrtomr2 / (1 - r); // good test for Loki
        }

        internal static double GetXMinusRY(double x, double y, double r, double omr2)
        {
            double xPlusy = x + y;
            if (r < -0.75 && Math.Abs(xPlusy) < 0.5 * Math.Abs(y))
            {
                // x-r*y has error eps*abs(r*y)
                // x+y-(r+1)*y has error eps*abs(x+y)+eps*(r+1)*abs(y)
                // difference is eps*(abs(r)-abs(r+1))*abs(y) which is significant when r near -1
                // we want abs(r)-abs(r+1) > 0.5 > abs((x+y)/y)
                // -r -(r+1) = -2r-1 > 0.5
                // -1.5 > 2r
                // -0.75 > r
                double rPlus1 = omr2 / (1 - r);
                return xPlusy - rPlus1 * y;
            }
            else if (r > 0.75 && Math.Abs(x - y) < 0.5 * Math.Abs(y))
            {
                double omr = omr2 / (1 + r);
                return x - y + omr * y;
            }
            else if (AreEqual(x, r * y))
            {
                return 0;
            }
            else
            {
                return x - r * y;
            }
        }

        internal static bool TraceConFrac;

        /// <summary>
        /// Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale.
        /// If r == -1, then divides by N(min(x,y);0,1) instead.
        /// If integral==true, returns NormalCdfIntegral divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="r"></param>
        /// <param name="scale"></param>
        /// <param name="sqrtomr2">sqrt(1-r*r)</param>
        /// <param name="integral"></param>
        /// <param name="smallX"></param>
        /// <returns></returns>
        private static double NormalCdfRatioConFrac(double x, double y, double r, double scale, double sqrtomr2, bool integral = false, bool smallX = false)
        {
            if (smallX)
            {
                //if (x >= -1)
                //    throw new ArgumentException("x >= -1");
                if (!integral)
                {
                    if (x - r * y > 0)
                        throw new ArgumentException("x - r*y > 0");
                }
            }
            else if (!integral)
            {
                //if (x > 0)
                //    throw new ArgumentOutOfRangeException(nameof(x), x, "x >= 0");
            }
            else if (double.IsPositiveInfinity(x)) throw new ArgumentOutOfRangeException(nameof(x), x, "x is infinity");
            else if (double.IsNegativeInfinity(x)) return 0;
            bool rIsMinus1 = AreEqual(sqrtomr2, 0) && AreEqual(r, -1);
            double xPlusy = x + y;
            if (xPlusy <= 0 && rIsMinus1) return 0;
            //double xmry0 = x - r * y;
            //if (xmry0 > 1e-2 || (!rIsMinus1 && xmry0 > 0))
            //    throw new ArgumentException("x - r*y > 0");
            double omr2 = sqrtomr2 * sqrtomr2;
            double rPlus1 = GetRPlus1(r, sqrtomr2);
            double numer;
            IEnumerator<double> RxmryIter;
            double probYScaled;
            // If true, use a recurrence for (numer - A2) instead of numer
            bool shiftNumer = false;
            double numerPrevPlusC = double.NaN;
            // Direct computation of numer on later iterations, useful for debugging
            //double numer3;
            if (rIsMinus1)
            {
                // numer = probYScaled * scale * (N(x;0,1) - N(y;0,1))/N(y;0,1)
                // C = probYScaled * scale * r * x * xPlusy
                if (y <= x)
                {
                    // normalize by N(y;0,1)
                    double delta = xPlusy * (y - x) / 2;
                    numer = ExpMinus1(delta) * scale;
                    probYScaled = 1;
                    if (integral && delta > -1)
                    {
                        // Avoid cancellation in numerPrev + C  = C - numer
                        // exp(delta)-1 + x*(x+y)
                        // = exp(delta)-1-delta + (x+y)*((y-x)/2 + x)
                        // = exp(delta)-1-delta + (x+y)*(y+x)/2
                        double expMinus1RatioMinus1RatioMinusHalf = MMath.ExpMinus1RatioMinus1RatioMinusHalf(delta);
                        double expMinus1RatioMinus1 = delta * (0.5 + expMinus1RatioMinus1RatioMinusHalf);
                        numerPrevPlusC = -scale * (delta * expMinus1RatioMinus1 + xPlusy * xPlusy / 2);
                        // delta + (x+y)^2 + x*(x+y) = (x+y)*((y-x)/2 + x+y+x) = (x+y)*(x+y)*3/2
                        //numer3 = -scale * (x / 3 * delta * expMinus1RatioMinus1 + x * xPlusy * xPlusy / 2);
                        // (x+y)^2*(x*(x+y) + 3/2 + 3/2*x^2)
                        //numer4 = -scale / 3 * ((x * x + 3) * delta * expMinus1RatioMinus1 + xPlusy * xPlusy * (x * x * 3 / 2 + 3.0 / 2 + x * xPlusy));
                        shiftNumer = true;
                    }
                }
                else
                {
                    // normalize by N(x;0,1)
                    double delta = xPlusy * (x - y) / 2;
                    double probYScaledMinus1 = ExpMinus1(delta);
                    numer = -probYScaledMinus1 * scale;
                    // N(y;0,1)/N(x;0,1)
                    probYScaled = probYScaledMinus1 + 1;
                    if (integral && delta > -1)
                    {
                        // Avoid cancellation in numerPrev + C = C - numer
                        double expMinus1RatioMinus1RatioMinusHalf = MMath.ExpMinus1RatioMinus1RatioMinusHalf(delta);
                        double expMinus1RatioMinus1 = delta * (0.5 + expMinus1RatioMinus1RatioMinusHalf);
                        numerPrevPlusC = -scale * (-delta * expMinus1RatioMinus1 + xPlusy * (xPlusy / 2 + x * probYScaledMinus1));
                        shiftNumer = true;
                    }
                }
                RxmryIter = null;
            }
            else
            {
                double xmry = GetXMinusRY(x, y, r, omr2) / sqrtomr2;
                RxmryIter = NormalCdfMomentRatioSequence(0, xmry);
                RxmryIter.MoveNext();
                double Rxmry = RxmryIter.Current;
                numer = NormalCdfRatioConFracNumer(x, y, r, scale, sqrtomr2, xmry, Rxmry);
                probYScaled = 1;
                if (integral && (xmry < 0 || r < -0.99))
                {
                    double ymrx = GetXMinusRY(y, x, r, omr2) / sqrtomr2;
                    if (ymrx < 0 || r < -0.99)
                    {
                        double R1ymrx = NormalCdfMomentRatio(1, ymrx);
                        //double R2ymrx = NormalCdfMomentRatio(2, ymrx) * 2;
                        double R1xmry = NormalCdfMomentRatio(1, xmry);
                        double R2xmry = NormalCdfMomentRatio(2, xmry) * 2;
                        //double R3xmry = NormalCdfMomentRatio(3, xmry) * 6;
                        double delta = AreEqual(x, y) ? 0 : rPlus1 * (y - x) / sqrtomr2;
                        if (Math.Abs(delta) <= 0.5)
                        {
                            // numer = scale * (NormalCdfRatioDiff(xmry, delta) + (1 + r) * Rxmry)
                            //       = scale * (delta*R1xmry + 0.5*delta*delta*R2xmry + ... + (1 + r) * Rxmry)
                            // replace Rxmry = R2xmry - xmry*R1xmry
                            // numer = scale * ((delta - (1+r)*xmry)*R1xmry + (0.5*delta*delta + 1+r)*R2xmry + ...)
                            // numerPrevPlusC = -numer + scale * r * sqrtomr2 * R1xmry * x
                            //                = scale * ((x * r * sqrtomr2 - delta + (1+r)*xmry)*R1xmry - (0.5*delta*delta + 1+r)*R2xmry - ...)
                            // c1 = x * r * sqrtomr2 - delta + (1+r)*xmry 
                            // = (x * r * (1-r^2) - (1+r)*(y-x) + (1+r)*(x-r*y)) / sqrtomr2
                            // = (x * r * (1-r) - (y-x) + (x-r*y)) * (1+r) / sqrtomr2
                            // = (x * (2+r-r^2) - (1+r)*y) * (1+r) / sqrtomr2
                            // = (x * (2-r)*(1+r) - (1+r)*y) * (1+r) / sqrtomr2
                            // (1+r)/sqrtomr2 = sqrtomr2/(1-r)
                            double omr = omr2 / rPlus1;
                            double c1 = sqrtomr2 * rPlus1 * (x + (x - y) / omr);
                            if (AreEqual(R1xmry, 0)) c1 = 0; // avoid inf * 0
                            double c2 = 0.5 * delta * delta + rPlus1;
                            double c3 = NormalCdfRatioDiff(xmry, delta, 3) * delta * delta;
                            numerPrevPlusC = scale * (c1 * R1xmry - c2 * R2xmry - c3);
                            //numer3 = scale / 3 * x * (c1 * R1xmry - c2 * R2xmry - c3 + r * omr2 * R2xmry);
                            //numer4 = scale / 3 * ((3 + x * x) * c1 * R1xmry + x * r * sqrtomr2 * omr2 * R3xmry - (3 * c2 + x * x * (c2 - r * omr2)) * R2xmry - (3 + x * x) * c3);
                            //Trace.WriteLine($"numerPrevPlusC = {numerPrevPlusC:g17} numer4 = {numer4:g17}");
                            //shiftNumer = omr2 * x * x > 100;
                            shiftNumer = true;
                        }
                        else if (scale == 1 && xmry < -1 && ymrx < -1)
                        {
                            // In this regime, Rymrx =approx -1/ymrx, Rxmry =approx -1/xmry, R1xmry =approx 1/xmry^2
                            // R1(ymrx)/ymrx + r*R1(xmry)/xmry = (R2(ymrx)+(1-R1(ymrx))/ymrx)/ymrx^2 + r*(R2(xmry)+(1-R1(xmry))/xmry)/xmry^2
                            // 1/ymrx^3 + r/xmry^3 = (xmry^3 + r*ymrx^3)/(xmry^3*ymrx^3)
                            // numerPrevPlusC = -numer + scale * r * sqrtomr2 * R1xmry * x
                            // = scale * (-Rymrx -r * Rxmry + r * sqrtomr2 * R1xmry * x)
                            // = scale * ((1-R1(ymrx))/ymrx + r * (1-R1(xmry))/xmry + r * sqrtomr2 * (R2(xmry)-R(xmry))/xmry * x)
                            // = scale * sqrtomr2 * ((1-R1(ymrx))/(y-r*x) + r * (1-R1(xmry))/(x-r*y) + r * (R2(xmry)+(1-R1(xmry))/xmry)/xmry * x)
                            // = scale * sqrtomr2 * (u - R1(ymrx)/(y-r*x) - r * R1(xmry)/(x-r*y) + r * (R2(xmry)-R1(xmry)/xmry)/xmry * x)
                            // u = 1/(y-rx) + r/(x-ry) + rx(1-r^2)/(x-ry)^2
                            // = ((x-ry)^2 + r(y-rx)(x-ry) + rx(1-r^2)(y-rx))/(x-ry)^2/(y-rx)
                            // = (1-r^2)^2 x^2 / (x-ry)^2 / (y-rx)
                            if (x < double.MinValue || x > double.MaxValue)
                            {
                                numerPrevPlusC = 0;
                            }
                            else
                            {
                                double xOverxmry = x / xmry;
                                double u = sqrtomr2 * xOverxmry * xOverxmry / ymrx;
                                numerPrevPlusC = scale * sqrtomr2 * (u - R1ymrx / (y - r * x) - r * R1xmry / (x - r * y) + r * (R2xmry - R1xmry / xmry) * xOverxmry);
                            }
                            shiftNumer = true;
                        }
                    }
                }
            }
            double numerPrev = 0;
            // denom is shifted by 1 after scaling
            double denom = 0;
            double denomPrev = 0;
            double resultPrev = 0;
            // This confrac converges faster as r -> -1
            double cOdd, cEven;
            if (rIsMinus1)
            {
                cEven = scale * r;
                // cOdd = xPlusy^i / i!! for odd i
                // cEven = xPlusy^i / (i-1)!! for even i
                cOdd = cEven * xPlusy;
            }
            else
            {
                cEven = scale * r;
                cOdd = cEven * sqrtomr2;
            }
            double xPlusy2 = xPlusy * xPlusy;
            // SmallX variables
            double invX2 = 1 / (x * x);
            double nOverX2 = invX2;
            double b = scale * r;
            if (integral)
            {
                if (shiftNumer)
                {
                    if (smallX)
                        numer = -numerPrevPlusC * invX2;
                    else
                        numer = -numerPrevPlusC * x;
                    numerPrev = 0; // unused
                }
                else
                {
                    numerPrev = -numer;
                    numer = 0;
                }
            }
            if (smallX && !integral)
            {
                numer /= x;
                b /= x;
            }
            double bIncr = sqrtomr2 / x;
            const int iterationCount = 10000;
            for (int i = 1; i <= iterationCount; i++)
            {
                double numerNew, denomNew;
                double c;
                if (rIsMinus1)
                {
                    c = probYScaled;
                }
                else
                {
                    //double c = MMath.NormalCdfMomentRatio(i, diff);
                    RxmryIter.MoveNext();
                    c = RxmryIter.Current;
                }
                if (smallX)
                {
                    // In this version, numer and denom are scaled by 1/x^n and then shifted.
                    b *= bIncr * i;
                    if (shiftNumer)
                    {
                        if (i == 1)
                            numerNew = 0;
                        else
                            numerNew = numer + nOverX2 * (numerPrev + numerPrevPlusC * invX2) + c * b;
                    }
                    else
                        numerNew = numer + nOverX2 * numerPrev + c * b;
                    denomNew = denom + nOverX2 * (denomPrev - 1);
                    nOverX2 += invX2;
                }
                else
                {
                    // In this version, numer and denom are scaled by 1/n!! and then shifted.
                    if (integral) c *= x;
                    if (i % 2 == 1)
                    {
                        if (i > 1)
                        {
                            if (rIsMinus1)
                                cOdd *= xPlusy2 / i;
                            else
                                cOdd *= (i - 1) * omr2;
                        }
                        if (shiftNumer)
                        {
                            if (i == 1)
                                numerNew = 0;
                            else
                            {
                                c *= cOdd;
                                numerNew = x * numer + c + numerPrev + numerPrevPlusC * x * x;
                            }
                        }
                        else
                        {
                            c *= cOdd;
                            numerNew = x * numer + c + numerPrev;
                        }
                        denomNew = x * denom + denomPrev - x * x;
                    }
                    else
                    {
                        if (rIsMinus1)
                            cEven *= xPlusy2 / (i - 1);
                        else
                            cEven *= i * omr2;
                        c *= cEven;
                        numerNew = (x * numer + c + i * numerPrev) / (i + 1);
                        denomNew = (x * denom + i * denomPrev) / (i + 1);
                    }
                }
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                if (i % 2 == 1)
                {
                    double numer2;
                    if (shiftNumer)
                    {
                        if (smallX)
                            numer2 = numer + numerPrevPlusC * invX2;
                        else
                            numer2 = numer + numerPrevPlusC;
                    }
                    else numer2 = numer;
                    double result = numer2 / (denom - 1);
                    if (TraceConFrac)
                        Trace.WriteLine($"iter {i}: result={result:g17} c={c:g17} cOdd={cOdd:g17} numer={numer:g17} numer2={numer2:g17} denom={denom:g17} numerPrev={numerPrev:g17}");
                    if ((result > double.MaxValue) || double.IsNaN(result) || result < 0 || i >= iterationCount - 1)
                        throw new Exception($"NormalCdfRatioConFrac2 not converging for x={x:g17} y={y:g17} r={r:g17} sqrtomr2={sqrtomr2:g17} scale={scale:g17}");
                    if (AreEqual(result, resultPrev) || AbsDiff(result, resultPrev, 0) < NormalCdfRatioConfracTolerance)
                        break;
                    resultPrev = result;
                }
            }
            return resultPrev;
        }

        // Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale
        // requires x <= 0, r <= 0, and x-r*y <= 0 (or equivalently y < -x).
        // This version works best for -1 < x <= 0.
        private static double NormalCdfRatioConFrac2b(double x, double y, double r, double scale, double sqrtomr2)
        {
            if (x > 0)
                throw new ArgumentException("x >= 0");
            //if (r > 0)
            //    throw new ArgumentException("r > 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            double omr2 = sqrtomr2 * sqrtomr2;
            double xmry = GetXMinusRY(x, y, r, omr2) / sqrtomr2;
            var RxmryIter = NormalCdfMomentRatioSequence(0, xmry);
            RxmryIter.MoveNext();
            double Rdiff = RxmryIter.Current;
            double numer = NormalCdfRatioConFracNumer(x, y, r, scale, sqrtomr2, xmry, Rdiff);
            double numerPrev = 0;
            // Negating here avoids negation in the loop.
            double denom = -x;
            double denomPrev = -1;
            double resultPrev = 0;
            double rDy = scale * r;
            for (int i = 1; i <= 1001; i++)
            {
                RxmryIter.MoveNext();
                double c = RxmryIter.Current;
                rDy *= sqrtomr2 * i;
                double numerNew = x * numer + i * numerPrev + c * rDy;
                double denomNew = x * denom + i * denomPrev;
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                if (i % 2 == 1)
                {
                    double result = numer / denom;
                    if (TraceConFrac)
                        Trace.WriteLine($"iter {i}: result={result:g17} c={c:g4} numer={numer:g17} denom={denom:g17} numerPrev={numerPrev:g17}");
                    if ((result > double.MaxValue) || double.IsNaN(result) || result < 0 || i >= 1000)
                        throw new Exception($"NormalCdfRatioConFrac2b not converging for x={x:g17} y={y:g17} r={r:g17} scale={scale}");
                    if (AreEqual(result, resultPrev))
                        break;
                    resultPrev = result;
                }
            }
            return resultPrev;
        }

        public static double NormalCdfIntegral(double x, double y, double r)
        {
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r  
            double sqrtomr2 = Math.Sqrt(omr2);
            return NormalCdfIntegral(x, y, r, sqrtomr2).ToDouble();
        }

        /// <summary>
        /// Computes the integral of the cumulative bivariate normal distribution wrt x,
        /// from -infinity to <paramref name="x"/>.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <param name="sqrtomr2">sqrt(1-r*r)</param>
        /// <returns>r such that r*exp(exponent) is the integral</returns>
        public static ExtendedDouble NormalCdfIntegral(double x, double y, double r, double sqrtomr2)
        {
            if (x < double.MinValue || y < double.MinValue)
            {
                return ExtendedDouble.Zero();
            }
            if (x > double.MaxValue)
            {
                return ExtendedDouble.PositiveInfinity();
            }
            if (AreEqual(r, 0) || y > double.MaxValue)
            {
                var result = NormalCdf(x, y, r, sqrtomr2);
                if (x > 0)
                {
                    // (x*phi(x) + N(x))/phi(x) = x + N(x)/phi(x)
                    result *= x + 1 / NormalCdfRatio(x);
                }
                else
                {
                    // x + 1/R(x) = (x*R(x)+1)/R(x)
                    result *= NormalCdfMomentRatio(1, x) / NormalCdfRatio(x);
                }
                return result;
            }
            double logProbX = Gaussian.GetLogProb(x, 0, 1);
            double logProbY = Gaussian.GetLogProb(y, 0, 1);
            if (AreEqual(sqrtomr2, 0) && AreEqual(r, 1))
            {
                // NormalCdf(x,y,1) = NormalCdf(min(x,y))
                // If x <= y then NormalCdfIntegral(x,y,1) = NormalCdfIntegral(x)  (xmry <= 0, ymrx >= 0)
                // Otherwise NormalCdfIntegral(x,y,1) = NormalCdfIntegral(y) + (x-y)*NormalCdf(y)  (xmry > 0, ymrx < 0)
                if (x <= y)
                {
                    return new ExtendedDouble(NormalCdfMomentRatio(1, x), logProbX);
                }
                else // y < x
                {
                    return new ExtendedDouble(NormalCdfMomentRatio(1, y) + (x - y) * NormalCdfRatio(y), logProbY);
                }
            }
            double rPlus1 = GetRPlus1(r, sqrtomr2);
            double omr2 = sqrtomr2 * sqrtomr2;
            double xmry = GetXMinusRY(x, y, r, omr2) / sqrtomr2;
            if ((x >= 0 && r >= 0) || (x > -1.2 && (r > -0.5 || (y > -100 && 1 + r > 1e-12))))
            {
                var result = NormalCdf(x, y, r, sqrtomr2) * x;
                double ymrx = GetXMinusRY(y, x, r, omr2) / sqrtomr2;
                if (ymrx < 0.1 && xmry < 0.1)
                {
                    double exponent2;
                    if (Math.Abs(x) >= Math.Abs(y))
                    {
                        exponent2 = logProbX + Gaussian.GetLogProb(ymrx, 0, 1);
                    }
                    else
                    {
                        // compute the exponent to match NormalCdf
                        exponent2 = logProbY + Gaussian.GetLogProb(xmry, 0, 1);
                    }
                    // R(ymrx) + r*R(xmry)
                    double result2 = NormalCdfRatioConFracNumer(x, y, r, 1, sqrtomr2, xmry, NormalCdfRatio(xmry));
                    return result + new ExtendedDouble(result2, exponent2);
                }
                else
                {
                    var phiy = NormalCdfExtended(ymrx).MultiplyExp(logProbX);
                    var phix = NormalCdfExtended(xmry).MultiplyExp(logProbY);
                    result += phiy;
                    result += phix * r;
                }
                // This is accurate when x >= 0 and r >= 0
                //return x * MMath.NormalCdf(x, y, r) + System.Math.Exp(Gaussian.GetLogProb(x, 0, 1) + MMath.NormalCdfLn(ymrx)) + r * System.Math.Exp(Gaussian.GetLogProb(y, 0, 1) + MMath.NormalCdfLn(xmry));
                return result;
            }
            // (x < 0 || r < 0) && (x <= -1.2 || Math.Abs(r) >= 0.9)
            // which implies x <= -1.2 || ((x < 0 || r < 0) && Math.Abs(r) >= 0.9)
            bool rIsMinus1 = AreEqual(sqrtomr2, 0) && AreEqual(r, -1);
            double xPlusy = x + y;
            if (rIsMinus1 && xPlusy <= 0)
            {
                return ExtendedDouble.Zero();
            }
            if (AreEqual(sqrtomr2, 0) && ((rIsMinus1 && xPlusy <= 1) || (AreEqual(r, 1) && x - y <= 1)))
            {
                double exponent = (y <= x) ? logProbY : logProbX;
                return new ExtendedDouble(NormalCdfRatioConFrac(x, y, r, 1, sqrtomr2, true), exponent);
            }
            else
            {
                if (xmry > 0)
                {
                    // ensure x-ry <= 0 
                    if (y > -x)
                    {
                        // recursive call has x-ry < 0
                        var result = -NormalCdfIntegral(-x, -y, r, sqrtomr2);
                        if (y - r * x > 0 && !rIsMinus1)
                        {
                            // Apply the identity NormalCdfIntegral(x,y,r) = 
                            // NormalCdfIntegral(x,y,-1) - NormalCdfIntegral(-x,-y,r)
                            return NormalCdfIntegral(x, y, -1, 0) + result;
                        }
                        else
                        {
                            // Apply the identity NormalCdfIntegral(x,y,r) = 
                            // -NormalCdfIntegral(-x,-y,r) + x * (NormalCdf(y) - NormalCdf(-x)) + N(x;0,1) + r * N(y;0,1)
                            var Z = NormalCdfDiff(y, -x);
                            if (x > double.MaxValue)
                            {
                                return new ExtendedDouble(AreEqual(Z.Mantissa, 0) ? 0 : x, 0);
                            }
                            // logProbX - logProbY = -x^2/2 + y^2/2 = (y+x)*(y-x)/2
                            ExtendedDouble n;
                            if (logProbX > logProbY || (logProbX == logProbY && x < y))
                            {
                                n = new ExtendedDouble(rPlus1 + r * ExpMinus1(xPlusy * (x - y) / 2), logProbX);
                            }
                            else
                            {
                                n = new ExtendedDouble(rPlus1 + ExpMinus1(xPlusy * (y - x) / 2), logProbY);
                            }
                            result += Z * x;
                            return result + n;
                        }
                    }
                    else // y <= -x
                    {
                        // Apply the identity NormalCdfIntegral(x,y,r) = 
                        // NormalCdfIntegral(-x,y,-r) + x * NormalCdf(y) + r * N(y;0,1)
                        // recursive call has x-ry < 0
                        var result = NormalCdfIntegral(-x, y, -r, sqrtomr2);
                        var Z = NormalCdfExtended(y);
                        result += Z * x;
                        return result + new ExtendedDouble(r, logProbY);
                        // This transformation doesn't help
                        //return -NormalCdfIntegral(x, -y, -r) + x * NormalCdf(x) + Math.Exp(logProbX);
                    }
                }
                double exponent = logProbX;
                double scale;
                double ymrx = GetXMinusRY(y, x, r, omr2) / sqrtomr2;
                if (ymrx < 0)
                {
                    // since phi(ymrx) will be small, we factor N(ymrx;0,1) out of the confrac
                    if (Math.Abs(x) >= Math.Abs(y))
                    {
                        exponent += Gaussian.GetLogProb(ymrx, 0, 1);
                    }
                    else
                    {
                        // compute the exponent in a different way, to match NormalCdf
                        exponent = logProbY + Gaussian.GetLogProb(xmry, 0, 1);
                    }
                    scale = 1;
                }
                else
                {
                    // leave N(ymrx;0,1) in the confrac
                    double logProb = Gaussian.GetLogProb(ymrx, 0, 1);
                    scale = Math.Exp(logProb);
                }
                if (x < -2)
                {
                    // should not be used when x-r*y > 0
                    // or |r|=1
                    return new ExtendedDouble(NormalCdfRatioConFrac(x, y, r, scale, sqrtomr2, true, true), exponent);
                }
                else
                {
                    // should not be used when x-r*y > 0
                    // or when x > -1.2 and y > -100 and r > -0.5
                    // experimental results:
                    // should not be used when x == -10 and y > 10
                    // should not be used when x == -2 and y > 2
                    // should not be used when x == -1.5 and y > 1.5
                    // should not be used when x == -1.1 and (y > 1.1 or r > 0)
                    // should not be used when x == -1 and (y > 1 or r > -0.13)
                    // (if y > -x then range of safe r decreases, and r near -1 becomes unsafe)
                    // should not be used when x == -0.1 and (y > 0.1 or r > -0.41)
                    // should not be used when x == -0.01 and (y > 0.01 or r > -0.42)
                    // should not be used when x == -0.001 and (y > 0.001 or r > -0.42)
                    // should not be used when x == 0.1 and (y > -0.1 or (y > -100 and -0.43 < r))
                    // (this includes x-ry > 0 but also some cases where x-ry<0)
                    // should not be used when x == 1 and (y > -1 or -0.4 < r)
                    // should not be used when x == 2 and (y > -2 or -0.4 < r)
                    // should not be used when x == 10 and (y > -10 or -0.2 < r)
                    return new ExtendedDouble(NormalCdfRatioConFrac(x, y, r, scale, sqrtomr2, true), exponent);
                }
            }
        }

        /// <summary>
        /// Computes the integral of the cumulative bivariate normal distribution wrt x, divided by the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <returns></returns>
        public static double NormalCdfIntegralRatio(double x, double y, double r)
        {
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r  
            double sqrtomr2 = Math.Sqrt(omr2);
            return NormalCdfIntegralRatio(x, y, r, sqrtomr2);
        }

        /// <summary>
        /// Computes the integral of the cumulative bivariate normal distribution wrt x, divided by the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <param name="sqrtomr2">sqrt(1-r*r)</param>
        /// <returns></returns>
        public static double NormalCdfIntegralRatio(double x, double y, double r, double sqrtomr2)
        {
            var intZ = NormalCdfIntegral(x, y, r, sqrtomr2);
            if (AreEqual(intZ.Mantissa, 0)) return 0;
            var Z = NormalCdf(x, y, r, sqrtomr2);
            return (intZ / Z).ToDouble();
        }

        /// <summary>
        /// Computes <c>1-sqrt(1-x)</c> to high accuracy.
        /// </summary>
        /// <param name="x">A real number between 0 and 1</param>
        /// <returns></returns>
        public static double OneMinusSqrtOneMinus(double x)
        {
            if (x > 1e-4)
                return 1 - Math.Sqrt(1 - x);
            else
                return
                    // Truncated series 17: 1 - sqrt(1 - x)
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    x * (1.0 / 2.0 +
                    x * (1.0 / 8.0 +
                    x * (1.0 / 16.0 +
                    x * 5.0 / 128.0)))
                    ;
        }

        /// <summary>
        /// Returns NormalCdfMomentRatio(i,x) for i=n,n+1,n+2,...
        /// </summary>
        /// <param name="n">A starting index &gt;= 0</param>
        /// <param name="x">A real number</param>
        /// <param name="useConFrac">If true, do not use the lookup table</param>
        /// <returns></returns>
        private static IEnumerator<double> NormalCdfMomentRatioSequence(int n, double x, bool useConFrac = false)
        {
            if (n < 0)
                throw new ArgumentException("n < 0");
            if (x > -1)
            {
                // Use the upward recurrence.
                double rPrev = MMath.NormalCdfRatio(x);
                if (n == 0)
                    yield return rPrev;
                double r = x * rPrev + 1;
                if (n <= 1)
                    yield return r;
                for (int i = 1; ; i++)
                {
                    double rNew = (x * r + rPrev) / (i + 1);
                    rPrev = r;
                    r = rNew;
                    if (n <= i + 1)
                        yield return r;
                }
            }
            else
            {
                // Use the downward recurrence.
                // Each batch of tableSize items is generated in advance.
                // If tableSize is larger than 50, the results will lose accuracy.
                // If tableSize is too small, then the recurrence is used less often, requiring more computation.
                int maxTableSize = 30;
                // rtable[tableStart-i] = R_i
                double[] rtable = new double[maxTableSize];
                int tableStart = -1;
                int tableSize = 2;
                for (int i = n; ; i++)
                {
                    if (i > tableStart)
                    {
                        // build the table
                        tableStart = i + tableSize - 1;
                        if (useConFrac)
                        {
                            rtable[0] = MMath.NormalCdfMomentRatioConFrac(tableStart, x);
                            rtable[1] = MMath.NormalCdfMomentRatioConFrac(tableStart - 1, x);
                        }
                        else
                        {
                            rtable[0] = MMath.NormalCdfMomentRatio(tableStart, x);
                            rtable[1] = MMath.NormalCdfMomentRatio(tableStart - 1, x);
                        }
                        for (int j = 2; j < tableSize; j++)
                        {
                            int nj = tableStart - j + 1;
                            if (rtable[j - 2] == 0 && rtable[j - 1] == 0)
                            {
                                rtable[j] = MMath.NormalCdfMomentRatio(nj - 1, x);
                            }
                            else
                            {
                                rtable[j] = (nj + 1) * rtable[j - 2] - x * rtable[j - 1];
                            }
                        }
                        // Increase tableSize up to the maximum.
                        tableSize = Math.Min(maxTableSize, 2 * tableSize);
                    }
                    yield return rtable[tableStart - i];
                }
            }
        }

        #endregion

        #region Logistic functions

        /// <summary>
        /// Computes the logistic function 1/(1+exp(-x)).
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <returns>1/(1+exp(-x)).</returns>
        public static double Logistic(double x)
        {
            if (x > 0.0)
                return 1.0 / (1 + Math.Exp(-x));
            else
            {
                double y = Math.Exp(x);
                return y / (1.0 + y);
            }
        }

        /// <summary>
        /// Compute the natural logarithm of the logistic function, i.e. -log(1+exp(-x)).
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <returns>-log(1+exp(-x)).</returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of <c>-log(1+exp(-x))</c>, 
        /// which can fail for x &lt; -50 and x > 36.</remarks>
        public static double LogisticLn(double x)
        {
            return -Log1PlusExp(-x);
#if false
            const double small = -50;
            const double large = 12.5;
            if (x < small) return x;
            if (x < large) return -Math.Log(1 + Math.Exp(-x));
            // x >= large
            // use the Taylor series for -log(1+x) around x=0
            // Maple command: series(-log(1+x),x);
            double e = Math.Exp(-x);
            return -e * (1 - e * 0.5);
#endif
        }

        /// <summary>
        /// Computes the natural logarithm of 1+x.
        /// </summary>
        /// <param name="x">A real number in the range -1 &lt;= x &lt;= Inf, or NaN.</param>
        /// <returns>log(1+x), which is always >= 0.</returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of <c>log(1+x)</c>,
        /// particularly when <paramref name="x"/> is small.
        /// </remarks>
        public static double Log1Plus(double x)
        {
            Assert.IsTrue(Double.IsNaN(x) || x >= -1);
            if (x > -1e-3 && x < 6e-2)
            {
                return
                    // Truncated series 10: log(1 + x)
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 0.076923076923076927*x**13 when x >= 0
                    // which is at most Ulp(1*x)/2 when 0 <= x <= 0.057980167435117531
                    x * (1.0 +
                    x * (-1.0 / 2.0 +
                    x * (1.0 / 3.0 +
                    x * (-1.0 / 4.0 +
                    x * (1.0 / 5.0 +
                    x * (-1.0 / 6.0 +
                    x * (1.0 / 7.0 +
                    x * (-1.0 / 8.0 +
                    x * (1.0 / 9.0 +
                    x * (-1.0 / 10.0 +
                    x * (1.0 / 11.0 +
                    x * (-1.0 / 12.0
                    ))))))))))))
                    ;
            }
            else
            {
                return Math.Log(1 + x);
            }
        }

        /// <summary>
        /// Computes log(numerator/denominator) to high accuracy.
        /// </summary>
        /// <param name="numerator">Any positive real number.</param>
        /// <param name="denominator">Any positive real number.</param>
        /// <returns>log(numerator/denominator)</returns>
        public static double LogRatio(double numerator, double denominator)
        {
            Assert.IsTrue(numerator > 0 && denominator > 0);
            double delta = (numerator - denominator) / denominator;
            if (delta > -1e-3 && delta < 5e-4)
                return Log1Plus(delta);
            else
                return Math.Log(numerator / denominator);
        }

        /// <summary>
        /// Computes log(1 + exp(x)) to high accuracy.
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <returns>log(1+exp(x)), which is always >= 0.</returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of <c>log(1+exp(x))</c>,
        /// particularly when x &lt; -36 or x > 50.</remarks>
        public static double Log1PlusExp(double x)
        {
            // The error of approximating log(1+exp(x)) by x is:
            //   log(1+exp(x))-x = log(1+exp(-x)) <= exp(-x)
            // Thus we should use the approximation when exp(-x) <= ulp(x)/2
            // ulp(1) < ulp(x)  therefore x > -log(ulp(1)/2)
            if (x > -logHalfUlpPrev1)
                return x;
            else
                return Log1Plus(Math.Exp(x));
        }

        /// <summary>
        /// Computes log(1 - exp(x)) to high accuracy.
        /// </summary>
        /// <param name="x">A non-positive real number: -Inf &lt;= x &lt;= 0, or NaN.</param>
        /// <returns>log(1-exp(x)), which is always &lt;= 0.</returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of <c>log(1-exp(x))</c>,
        /// particularly when x &lt; -5 or x > -1e-5.</remarks>
        public static double Log1MinusExp(double x)
        {
            if (x > 0)
                throw new ArgumentException("x (" + x + ") > 0");
            if (x < -3.5)
            {
                double expx = Math.Exp(x);
                return
                    // Truncated series 11: log(1 - x)
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    expx * (-1.0 +
                    expx * (-1.0 / 2.0 +
                    expx * (-1.0 / 3.0 +
                    expx * (-1.0 / 4.0 +
                    expx * (-1.0 / 5.0 +
                    expx * (-1.0 / 6.0 +
                    expx * (-1.0 / 7.0 +
                    expx * (-1.0 / 8.0 +
                    expx * (-1.0 / 9.0 +
                    expx * -1.0 / 10.0)))))))))
                    ;
            }
            else
            {
                return Math.Log(-ExpMinus1(x));
            }
        }

        /// <summary>
        /// Computes the exponential of x and subtracts 1.
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <returns>exp(x)-1</returns>
        /// <remarks>
        /// This function is more accurate than a direct evaluation of <c>exp(x)-1</c> when x is small.
        /// It is the inverse function to Log1Plus: <c>ExpMinus1(Log1Plus(x)) == x</c>.
        /// </remarks>
        public static double ExpMinus1(double x)
        {
            if (Math.Abs(x) < 55e-3)
            {
                return
                    // Truncated series 13: exp(x) - 1
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    x * (1.0 +
                    x * (1.0 / 2.0 +
                    x * (1.0 / 6.0 +
                    x * (1.0 / 24.0 +
                    x * (1.0 / 120.0 +
                    x * (1.0 / 720.0 +
                    x * (1.0 / 5040.0 +
                    x * (1.0 / 40320.0
                    ))))))));
            }
            else if (x == Ln2)
            {
                // Math.Exp(Ln2) is not required to be exactly 2, so we force it.
                return 1;
            }
            else
            {
                return Math.Exp(x) - 1.0;
            }
        }

        /// <summary>
        /// Computes ((exp(x)-1)/x - 1)/x - 0.5
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <returns>((exp(x)-1)/x - 1)/x - 0.5</returns>
        public static double ExpMinus1RatioMinus1RatioMinusHalf(double x)
        {
            if (Math.Abs(x) < 6e-1)
            {
                return
                    // Truncated series 14: ((exp(x) - 1) / x - 1) / x - 0.5
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    x * (1.0 / 6.0 +
                    x * (1.0 / 24.0 +
                    x * (1.0 / 120.0 +
                    x * (1.0 / 720.0 +
                    x * (1.0 / 5040.0 +
                    x * (1.0 / 40320.0 +
                    x * (1.0 / 362880.0 +
                    x * (1.0 / 3628800.0 +
                    x * (1.0 / 39916800.0 +
                    x * (1.0 / 479001600.0 +
                    x * (1.0 / 6227020800.0 +
                    x * 1.0 / 87178291200.0)))))))))))
                    ;
            }
            else if (double.IsPositiveInfinity(x))
            {
                return x;
            }
            else
            {
                return ((Math.Exp(x) - 1) / x - 1) / x - 0.5;
            }
        }

        /// <summary>
        /// Computes <c>log(exp(x)-1)</c> for non-negative x.
        /// </summary>
        /// <param name="x">A non-negative real number: 0 &lt;= x &lt;= Inf, or NaN.</param>
        /// <returns><c>log(exp(x)-1)</c></returns>
        /// <remarks>
        /// This function is more accurate than a direct evaluation of <c>log(exp(x)-1)</c> when x &lt; 1e-3
        /// or x > 50.
        /// It is the inverse function to Log1PlusExp: <c>LogExpMinus1(Log1PlusExp(x)) == x</c>.
        /// </remarks>
        public static double LogExpMinus1(double x)
        {
            if (x < 1e-3)
            {
                return Math.Log(x) +
                    // Truncated series 15: log(exp(x) - 1) / x
                    // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
                    // Error is at most 0.00034722222222222224*x**4 when x >= 0
                    // which is at most Ulp(6.9077552789821368)/2 when 0 <= x <= 0.0012190876039316647
                    x * (1.0 / 2.0 +
                    x * (1.0 / 24.0
                    ))
                    ;
            }
            else if (x > 50)
            {
                return x;
            }
            else
            {
                return Math.Log(ExpMinus1(x));
            }
        }

        /// <summary>
        /// Computes log(exp(x) - exp(y)) to high accuracy.
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.  Must be greater or equal to y.</param>
        /// <param name="y">Any real number from -Inf to Inf, or NaN.  Must be less or equal to x.</param>
        /// <returns></returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of <c>log(exp(x)-exp(y))</c>.</remarks>
        public static double LogDifferenceOfExp(double x, double y)
        {
            if (x == y)
                return Double.NegativeInfinity;
            if (Double.IsNegativeInfinity(y))
                return x;
            return x + MMath.Log1MinusExp(y - x);
        }

        /// <summary>
        /// Computes exp(x)-exp(y) to high accuracy.
        /// </summary>
        /// <param name="x">Any real number</param>
        /// <param name="y">Any real number</param>
        /// <returns>exp(x)-exp(y)</returns>
        public static double DifferenceOfExp(double x, double y)
        {
            if (x == y)
                return 0.0;
            else if (x > y)
                return Math.Exp(x + MMath.Log1MinusExp(y - x));
            else if (double.IsNaN(x) || double.IsNaN(y))
                return double.NaN;
            else
                return -DifferenceOfExp(y, x);
        }

        /// <summary>
        /// Computes log(exp(x) + exp(y)) to high accuracy.
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <param name="y">Any real number from -Inf to Inf, or NaN.</param>
        /// <returns>log(exp(x)+exp(y)), which is always >= max(x,y).</returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of <c>log(exp(x)+exp(y))</c>.</remarks>
        public static double LogSumExp(double x, double y)
        {
            if (double.IsNegativeInfinity(x))
            {
                return y;
            }

            if (double.IsNegativeInfinity(y))
            {
                return x;
            }

            double delta = Math.Abs(x - y);
            double max = Math.Max(x, y); // also 0.5*(x+y+delta)
            // if x = y = Inf or -Inf, delta will be NaN.
            // return max to also catch the case where x = NaN or y = NaN.
            if (Double.IsNaN(delta))
                return max;
            return max + Log1PlusExp(-delta);
        }

        /// <summary>
        /// Returns the log of the sum of exponentials of a list of doubles
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static double LogSumExp(IEnumerable<double> list)
        {
            IReadOnlyCollection<double> items = list as IReadOnlyCollection<double> ?? list.ToReadOnlyList();

            if (items.Count == 0)
                return Double.NegativeInfinity; // log(0)

            double max = Max(items);
            if (Double.IsNegativeInfinity(max))
                return Double.NegativeInfinity; // log(0)
            if (Double.IsPositiveInfinity(max))
                return Double.PositiveInfinity;

            // at this point, max is finite
            IEnumerator<double> iter = items.GetEnumerator();
            iter.MoveNext();
            double Z = Math.Exp(iter.Current - max);
            while (iter.MoveNext())
            {
                Z += Math.Exp(iter.Current - max);
            }

            return Math.Log(Z) + max;
        }

        /// <summary>
        /// Returns the log of the sum of exponentials of a list of doubles
        /// </summary>
        /// <param name="list"></param>    
        /// <returns></returns>
        public static double LogSumExpSparse(IEnumerable<double> list)
        {
            IReadOnlyCollection<double> items = list as IReadOnlyCollection<double> ?? list.ToReadOnlyList();

            if (items.Count == 0)
                return Double.NegativeInfinity; // log(0)

            if (!(items is ISparseEnumerable<double>))
                return LogSumExp(items);

            double max = items.EnumerableReduce(double.NegativeInfinity,
                                               (res, i) => Math.Max(res, i), (res, i, count) => Math.Max(res, i));
            var iter = (items as ISparseEnumerable<double>).GetSparseEnumerator();
            if ((!iter.MoveNext() && iter.CommonValueCount == 0) || Double.IsNegativeInfinity(max))
                return Double.NegativeInfinity; // log(0)
            if (Double.IsPositiveInfinity(max))
                return Double.PositiveInfinity;
            // at this point, max is finite
            double Z = Math.Exp(iter.Current - max);
            while (iter.MoveNext())
            {
                Z += Math.Exp(iter.Current - max);
            }
            Z += Math.Exp(iter.CommonValue - max) * iter.CommonValueCount;
            return Math.Log(Z) + max;
        }

        /// <summary>
        /// Computes log(exp(x)+exp(a))-log(exp(x)+exp(b)) to high accuracy.
        /// </summary>
        /// <param name="x">Any real number from -Inf to Inf, or NaN.</param>
        /// <param name="a">A finite real number.</param>
        /// <param name="b">A finite real number.</param>
        /// <returns>log(exp(x)+exp(a))-log(exp(x)+exp(b))</returns>
        /// <remarks>This function provides higher accuracy than a direct evaluation of 
        /// <c>LogSumExp(x,a)-LogSumExp(x,b)</c>, particularly when x is large.
        /// </remarks>
        public static double DiffLogSumExp(double x, double a, double b)
        {
            Assert.IsTrue(!Double.IsInfinity(a));
            Assert.IsTrue(!Double.IsInfinity(b));
            if (x > a && x > b)
            {
                return Log1PlusExp(a - x) - Log1PlusExp(b - x);
            }
            // x cannot be Inf here so this difference is safe.
            // if a and b could be infinite, we would need extra checks.
            return LogSumExp(x, a) - LogSumExp(x, b);
        }

        /// <summary>
        /// Computes the log-odds function log(p/(1-p)).
        /// </summary>
        /// <param name="p">Any number between 0 and 1, inclusive.</param>
        /// <returns>log(p/(1-p))</returns>
        /// <remarks>This function is the inverse of the logistic function, 
        /// i.e. <c>Logistic(Logit(p)) == p.</c></remarks>
        public static double Logit(double p)
        {
            if (p >= 1.0)
                return Double.PositiveInfinity;
            else if (p <= 0.0)
                return Double.NegativeInfinity;
            else
                return Math.Log(p / (1 - p));
        }

        /// <summary>
        /// Compute log(p/(1-p)) from log(p).
        /// </summary>
        /// <param name="logp">Any number between -infinity and 0, inclusive.</param>
        /// <returns>log(exp(logp)/(1-exp(logp))) = -log(exp(-logp)-1).</returns>
        public static double LogitFromLog(double logp)
        {
            return -LogExpMinus1(-logp);
        }

        /// <summary>
        /// Exponentiate array elements and normalize to sum to 1.
        /// </summary>
        /// <param name="x">May be +/-infinity</param>
        /// <returns>A Vector p where <c>p[k] = exp(x[k])/sum_j exp(x[j])</c></returns>
        /// <remarks>Sparse lists and vectors are handled efficiently</remarks>
        public static Vector Softmax(IList<double> x)
        {
            int n = x.Count;
            Vector p;

            bool isSparse = x.IsSparse();
            ISparseList<double> isl = x as ISparseList<double>;

            if (n == 0)
                return Vector.Zero(0, isSparse ? Sparsity.Sparse : Sparsity.Dense); // Zero length

            double max = x.EnumerableReduce(
                double.NegativeInfinity,
                (acc, d) => Math.Max(acc, d),
                (acc, d, cnt) => Math.Max(acc, d));

            if (double.IsNegativeInfinity(max))
            {
                p = Vector.Constant(n, 1.0 / n, isSparse ? Sparsity.Sparse : Sparsity.Dense);
                return p;
            }
            if (double.IsPositiveInfinity(max))
            {
                double count = x.EnumerableReduce(
                    0.0, (acc, d) => double.IsPositiveInfinity(d) ? acc + 1.0 : acc,
                    (acc, d, cnt) => double.IsPositiveInfinity(d) ? acc + cnt : acc);
                double y = 1.0 / count;
                if (isSparse)
                {
                    ISparseEnumerator<double> sen = isl.GetSparseEnumerator();
                    if (double.IsPositiveInfinity(sen.CommonValue))
                    {
                        p = SparseVector.Constant(n, y);
                        while (sen.MoveNext())
                            if (!Double.IsPositiveInfinity(sen.Current))
                                p[sen.CurrentIndex] = 0.0;
                    }
                    else
                    {
                        p = SparseVector.Constant(n, 0.0);
                        while (sen.MoveNext())
                            if (Double.IsPositiveInfinity(sen.Current))
                                p[sen.CurrentIndex] = y;
                    }
                }
                else
                {
                    p = Vector.Zero(n);
                    for (int i = 0; i < n; i++)
                        if (double.IsPositiveInfinity(x[i]))
                            p[i] = y;
                }
                return p;
            }
            // at this point, max is finite
            double z = 0;
            double expX;
            if (isSparse)
            {
                ISparseEnumerator<double> sen = isl.GetSparseEnumerator();
                double expCV = Math.Exp(sen.CommonValue - max);
                p = SparseVector.Constant(n, expCV);
                while (sen.MoveNext())
                {
                    expX = Math.Exp(sen.Current - max);
                    z += expX;
                    p[sen.CurrentIndex] = expX;
                }
                // Following must occur at end of loop so CommonValueCount is
                // correctly set. Also we must check for > 0 because common value
                // may be infinite
                if (sen.CommonValueCount > 0)
                    z += sen.CommonValueCount * expCV;
            }
            else
            {
                p = Vector.Zero(n);
                for (int i = 0; i < n; i++)
                {
                    expX = Math.Exp(x[i] - max);
                    z += expX;
                    p[i] = expX;
                }
            }
            double invZ = 1.0 / z;
            p.Scale(invZ);
            return p;
        }

        #endregion

        #region LogisticGaussian functions

        /// <summary>
        /// Evaluates E[log(1+exp(x))] under a Gaussian distribution with specified mean and variance.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <returns></returns>
        public static double Log1PlusExpGaussian(double mean, double variance)
        {
            double[] nodes = new double[11];
            double[] weights = new double[11];
            Quadrature.GaussianNodesAndWeights(mean, variance, nodes, weights);
            double z = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                double x = nodes[i];
                double f = MMath.Log1PlusExp(x);
                z += weights[i] * f;
            }
            return z;
        }

        // Integrate using quadrature nodes and weights
        private static double Integrate(Converter<double, double> f, Vector nodes, Vector weights)
        {
            return weights.Inner(nodes, x => f(x));
        }

        /// <summary>
        /// Specifies the number of quadrature nodes that should be used when doing
        /// Gauss Hermite quadrature for direct KL minimisation 
        /// </summary>
        private const int LogisticGaussianQuadratureNodeCount = 50;

        /// <summary>
        /// For integrals with variance greater than this Clenshaw curtis quadrature will be used
        /// instead of Gauss-Hermite quadrature. 
        /// </summary>
        private const double LogisticGaussianVarianceThreshold = 2.0;

        // Use the large variance approximation to sigma(m,v) = \int N(x;m,v) logistic(x)
        private static void BigvProposal(double m, double v, out double mf, out double vf)
        {
            double v2 = v + Math.PI * Math.PI / 3.0;
            double Z = MMath.NormalCdf(m / Math.Sqrt(v2));
            double s1 = Math.Exp(Gaussian.GetLogProb(0, m, v2));
            double s2 = -s1 * m / v2;
            mf = m + v * s1 / Z;
            double Ex2 = v + m * m + 2 * m * v * s1 / Z + v * v * s2 / Z;
            vf = Ex2 - mf * mf;
        }

        /// <summary>
        /// Calculate sigma(m,v) = \int N(x;m,v) logistic(x) dx
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>The value of this special function.</returns>
        /// <remarks><para>
        /// Note <c>1-LogisticGaussian(m,v) = LogisticGaussian(-m,v)</c> which is more accurate.
        /// </para><para>
        /// For large v we can use the big v approximation <c>\sigma(m,v)=normcdf(m/sqrt(v+pi^2/3))</c>.
        /// For small and moderate v we use Gauss-Hermite quadrature.
        /// For moderate v we first find the mode of the (log concave) function since this may be quite far from m.
        /// </para></remarks>
        public static double LogisticGaussian(double mean, double variance)
        {
            double halfVariance = 0.5 * variance;

            // use the upper bound exp(m+v/2) to prune cases that must be zero or one
            if (mean + halfVariance < log0)
                return 0.0;
            if (-mean + halfVariance < logHalfUlpPrev1)
                return 1.0;

            // use the upper bound 0.5 exp(-0.5 m^2/v) to prune cases that must be zero or one
            double q = -0.5 * mean * mean / variance - MMath.Ln2;
            if (mean <= 0 && mean + variance >= 0 && q < log0)
                return 0.0;
            if (mean >= 0 && variance - mean >= 0 && q < logHalfUlpPrev1)
                return 1.0;
            // sigma(|m|,v) <= 0.5 + |m| sigma'(0,v)
            // sigma'(0,v) <= N(0;0,v+8/pi)
            //double d0Upper = MMath.InvSqrt2PI / Math.Sqrt(variance + 8 / Math.PI);
            if (mean * mean / (variance + 8 / Math.PI) < logisticGaussianSeriesApproximmationThreshold)
            {
                double deriv = LogisticGaussianDerivative(0, variance);
                return 0.5 + mean * deriv;
            }

            // Handle tail cases using the following exact formulas:
            // sigma(m,v) = 1 - exp(-m+v/2) + exp(-2m+2v) - exp(-3m+9v/2) sigma(m-3v,v)
            if (2 * (variance - mean) < logHalfUlpPrev1)
                return 1.0 - Math.Exp(halfVariance - mean);
            if (-3 * mean + 9 * halfVariance < logHalfUlpPrev1)
                return 1.0 - Math.Exp(halfVariance - mean) + Math.Exp(2 * (variance - mean));
            // sigma(m,v) = exp(m+v/2) - exp(2m+2v) + exp(3m + 9v/2) (1 - sigma(m+3v,v))
            if (mean + 1.5 * variance < logHalfUlpPrev1)
                return Math.Exp(mean + halfVariance);
            if (2 * mean + 4 * variance < logHalfUlpPrev1)
                return Math.Exp(mean + halfVariance) * (1 - Math.Exp(mean + 1.5 * variance));

            if (variance > LogisticGaussianVarianceThreshold)
            {
                double sqrtv = System.Math.Sqrt(variance);
                double meanInvSqrtV = mean / sqrtv;
                double shift = (mean < 0) ? Gaussian.GetLogProb(0, meanInvSqrtV, 1) : 0;
                double f(double x)
                {
                    double diff = x - meanInvSqrtV;
                    return Math.Exp(MMath.LogisticLn(x * sqrtv) - diff * diff / 2 - MMath.LnSqrt2PI - shift);
                }
                double upperBound = mean + Math.Sqrt(variance);
                double scale = Math.Max(upperBound, 10) / sqrtv;
                return new ExtendedDouble(Quadrature.AdaptiveExpSinh(f, scale, logisticGaussianQuadratureRelativeTolerance / 2) +
                    Quadrature.AdaptiveExpSinh(x => f(-x), scale, logisticGaussianQuadratureRelativeTolerance / 2), shift).ToDouble();
            }
            else
            {
                Vector nodes = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                Vector weights = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                double m_p, v_p;
                BigvProposal(mean, variance, out m_p, out v_p);
                Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                double weightedIntegrand(double z)
                {
                    return Math.Exp(MMath.LogisticLn(z) + Gaussian.GetLogProb(z, mean, variance) - Gaussian.GetLogProb(z, m_p, v_p));
                }
                return Integrate(weightedIntegrand, nodes, weights);
            }
        }

        /// <summary>
        /// Calculate <c>\sigma'(m,v)=\int N(x;m,v)logistic'(x) dx</c>
        /// </summary>
        /// <param name="mean">Mean.</param>
        /// <param name="variance">Variance.</param>
        /// <returns>The value of this special function.</returns>
        /// <remarks><para>
        /// For large v we can use the big v approximation <c>\sigma'(m,v)=N(m,0,v+pi^2/3)</c>.
        /// For small and moderate v we use Gauss-Hermite quadrature.
        /// For moderate v we first find the mode of the (log concave) function since this may be quite far from m.
        /// </para></remarks>
        public static double LogisticGaussianDerivative(double mean, double variance)
        {
            double halfVariance = 0.5 * variance;
            mean = Math.Abs(mean);

            // use the upper bound exp(-|m|+v/2) to prune cases that must be zero
            if (-mean + halfVariance < log0)
                return 0.0;

            // use the upper bound 0.5 exp(-0.5 m^2/v) to prune cases that must be zero
            double q = -0.5 * mean * mean / variance - MMath.Ln2;
            if (mean <= variance && q < log0)
                return 0.0;
            if (double.IsPositiveInfinity(variance))
                return 0.0;

            // Handle the tail cases using the following exact formula:
            // sigma'(m,v) = exp(-m+v/2) -2 exp(-2m+2v) +3 exp(-3m+9v/2) sigma(m-3v,v) - exp(-3m+9v/2) sigma'(m-3v,v)
            if (-mean + 1.5 * variance < logHalfUlpPrev1)
                return Math.Exp(halfVariance - mean);
            if (-2 * mean + 4 * variance < logHalfUlpPrev1)
                return Math.Exp(halfVariance - mean) - 2 * Math.Exp(2 * (variance - mean));

            if (variance > LogisticGaussianVarianceThreshold)
            {
                double shift = Gaussian.GetLogProb(0, mean, variance);
                double f(double x)
                {
                    return Math.Exp(MMath.LogisticLn(x) + MMath.LogisticLn(-x) + Gaussian.GetLogProb(x, mean, variance) - shift);
                }
                return new ExtendedDouble(Quadrature.AdaptiveClenshawCurtis(f, 10, 32, logisticGaussianDerivativeQuadratureRelativeTolerance), shift).ToDouble();
            }
            else
            {
                Vector nodes = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                Vector weights = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                double m_p, v_p;
                BigvProposal(mean, variance, out m_p, out v_p);
                Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                double weightedIntegrand(double z)
                {
                    return Math.Exp(MMath.LogisticLn(z) + MMath.LogisticLn(-z) + Gaussian.GetLogProb(z, mean, variance) - Gaussian.GetLogProb(z, m_p, v_p));
                }
                return Integrate(weightedIntegrand, nodes, weights);
            }
        }


        /// <summary>
        /// Calculate <c>\sigma''(m,v)=\int N(x;m,v)logistic''(x) dx</c>
        /// </summary>
        /// <param name="mean">Mean.</param>
        /// <param name="variance">Variance.</param>
        /// <returns>The value of this special function.</returns>
        /// <remarks><para>
        /// For large v we can use the big v approximation <c>\sigma'(m,v)=-m/(v+pi^2/3)*N(m,0,v+pi^2/3)</c>.
        /// For small and moderate v we use Gauss-Hermite quadrature.
        /// The function is multimodal so mode finding is difficult and probably won't help.
        /// </para></remarks>
        public static double LogisticGaussianDerivative2(double mean, double variance)
        {
            if (double.IsPositiveInfinity(variance))
                return 0.0;
            if (mean == 0)
                return 0;

            double halfVariance = 0.5 * variance;

            // use the upper bound exp(-|m|+v/2) to prune cases that must be zero
            if (-Math.Abs(mean) + halfVariance < log0)
                return 0.0;

            // use the upper bound 0.5 exp(-0.5 m^2/v) to prune cases that must be zero
            double q = -0.5 * mean * mean / variance - MMath.Ln2;
            if (Math.Abs(mean) <= variance && q < log0)
                return 0.0;

            // Handle the tail cases using the following exact formulas:
            // sigma''(m,v) = -exp(-m+v/2) +4 exp(-2m+2v) -9 exp(-3m+9v/2) sigma(m-3v,v) +6 exp(-3m+9v/2) sigma'(m-3v,v) - exp(-3m+9v/2) sigma''(m-3v,v)
            if (-mean + 1.5 * variance < logHalfUlpPrev1)
                return -Math.Exp(halfVariance - mean);
            if (-2 * mean + 4 * variance < logHalfUlpPrev1)
                return -Math.Exp(halfVariance - mean) + 4 * Math.Exp(2 * (variance - mean));
            // sigma''(m,v) = exp(m+v/2) -4 exp(2m+2v) +9 exp(3m + 9v/2) (1 - sigma(m+3v,v)) - 6 exp(3m+9v/2) sigma'(m+3v,v) - exp(3m + 9v/2) sigma''(m+3v,v)
            if (mean + 1.5 * variance < logHalfUlpPrev1)
                return Math.Exp(mean + halfVariance);
            if (2 * mean + 4 * variance < logHalfUlpPrev1)
                return Math.Exp(mean + halfVariance) * (1 - 4 * Math.Exp(mean + 1.5 * variance));

            if (variance > LogisticGaussianVarianceThreshold)
            {
                double shift = Gaussian.GetLogProb(0, mean, variance);
                double f(double x)
                {
                    double logSigma = MMath.LogisticLn(x);
                    double log1MinusSigma = MMath.LogisticLn(-x);
                    double OneMinus2Sigma = -Math.Tanh(x / 2);
                    return OneMinus2Sigma * Math.Exp(logSigma + log1MinusSigma + Gaussian.GetLogProb(x, mean, variance) - shift);
                }
                return new ExtendedDouble(Quadrature.AdaptiveClenshawCurtis(f, 10, 32, logisticGaussianDerivativeQuadratureRelativeTolerance), shift).ToDouble();
            }
            else
            {
                Vector nodes = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                Vector weights = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                double m_p, v_p;
                BigvProposal(mean, variance, out m_p, out v_p);
                Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                double weightedIntegrand(double z)
                {
                    double logSigma = MMath.LogisticLn(z);
                    double log1MinusSigma = MMath.LogisticLn(-z);
                    double OneMinus2Sigma = -Math.Tanh(z / 2);
                    return OneMinus2Sigma * Math.Exp(logSigma + log1MinusSigma + Gaussian.GetLogProb(z, mean, variance) - Gaussian.GetLogProb(z, m_p, v_p));
                }
                return Integrate(weightedIntegrand, nodes, weights);
            }
        }

        /// <summary>
        /// Calculate (kth derivative of LogisticGaussian)*exp(0.5*mean^2/variance)
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static double LogisticGaussianRatio(double mean, double variance, int k)
        {
            if (k < 0 || k > 2) throw new ArgumentException("invalid k (" + k + ")");
            double a = mean / variance;
            // int 0.5 cosh(x(m/v+1/2))/cosh(x/2) N(x;0,v) dx
            double f(double x)
            {
                double logSigma = MMath.LogisticLn(x);
                double extra = 0;
                double s = 1;
                if (k > 0) extra += MMath.LogisticLn(-x);
                if (k > 1) s = -Math.Tanh(x / 2);
                return s * Math.Exp(logSigma + extra + x * a + Gaussian.GetLogProb(x, 0, variance));
            }
            double upperBound = (Math.Abs(a + 0.5) - 0.5) * variance + Math.Sqrt(variance);
            upperBound = Math.Max(upperBound, 10);
            return Quadrature.AdaptiveClenshawCurtis(f, upperBound, 32, 1e-10);
        }

        #endregion

        #region Min-Max functions

        /// <summary>
        /// Returns the maximum of a list of doubles
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static double Max(IEnumerable<double> list)
        {
            IEnumerator<double> iter = list.GetEnumerator();
            if (!iter.MoveNext())
                throw new InvalidOperationException("Sequence contains no elements");
            double Z = iter.Current;
            while (iter.MoveNext())
            {
                Z = Math.Max(Z, iter.Current);
            }
            return Z;
        }

        /// <summary>
        /// Returns the minimum of a list of doubles
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static double Min(IEnumerable<double> list)
        {
            IEnumerator<double> iter = list.GetEnumerator();
            if (!iter.MoveNext())
                throw new InvalidOperationException("Sequence contains no elements");
            double Z = iter.Current;
            while (iter.MoveNext())
            {
                Z = Math.Min(Z, iter.Current);
            }
            return Z;
        }

        /// <summary>
        /// Returns the index of the minimum element, or -1 if empty.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <returns></returns>
        public static int IndexOfMinimum<T>(IEnumerable<T> list)
            where T : IComparable<T>
        {
            IEnumerator<T> iter = list.GetEnumerator();
            if (!iter.MoveNext())
                return -1;
            T min = iter.Current;
            int indexOfMinimum = 0;
            for (int i = 1; iter.MoveNext(); i++)
            {
                T item = iter.Current;
                if (min.CompareTo(item) > 0)
                {
                    min = item;
                    indexOfMinimum = i;
                }
            }
            return indexOfMinimum;
        }

        /// <summary>
        /// Returns the index of the maximum element, or -1 if empty.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <returns></returns>
        public static int IndexOfMaximum<T>(IEnumerable<T> list)
            where T : IComparable<T>
        {
            IEnumerator<T> iter = list.GetEnumerator();
            if (!iter.MoveNext())
                return -1;
            T max = iter.Current;
            int indexOfMaximum = 0;
            for (int i = 1; iter.MoveNext(); i++)
            {
                T item = iter.Current;
                if (max.CompareTo(item) < 0)
                {
                    max = item;
                    indexOfMaximum = i;
                }
            }
            return indexOfMaximum;
        }

        /// <summary>
        /// Returns the index of the maximum element, or -1 if empty.
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static int IndexOfMaximumDouble(IList<double> list)
        {
            return IndexOfMaximum<double>(list);
        }

        #endregion

        /// <summary>
        /// Returns the median of the array elements.
        /// </summary>
        /// <param name="array"></param>
        /// <returns>The median ignoring NaNs.</returns>
        public static double Median(double[] array)
        {
            return Median(array, 0, array.Length);
        }

        /// <summary>
        /// Returns the median of elements in a subrange of an array.
        /// </summary>
        /// <param name="array"></param>
        /// <param name="start">Starting index of the range.</param>
        /// <param name="length">The number of elements in the range.</param>
        /// <returns>The median of array[start:(start+length-1)], ignoring NaNs.</returns>
        public static double Median(double[] array, int start, int length)
        {
            if (length <= 0)
                throw new InvalidOperationException("Sequence contains no elements");
            // this can be done in O(n) time, but here we take a slower shortcut.
            // Array.Sort does not sort NaNs reliably, so we must extract them first.
            double[] a = RemoveNaNs(array, start, length);
            length = a.Length;
            if (length == 0)
                return Double.NaN;
            Array.Sort(a);
            int middle = length / 2;
            if (length % 2 == 0)
            {
                // average the two middle elements
                return Average(a[middle - 1], a[middle]);
            }
            else
            {
                return a[middle];
            }
        }

        /// <summary>
        /// Given an array, returns a new array with all NANs removed.
        /// </summary>
        /// <param name="array">The source array</param>
        /// <param name="start">The start index in the source array</param>
        /// <param name="length">How many items to look at in the source array</param>
        /// <returns></returns>
        public static double[] RemoveNaNs(double[] array, int start, int length)
        {
            int count = 0;
            for (int i = 0; i < length; i++)
            {
                if (!Double.IsNaN(array[i + start]))
                    count++;
            }
            double[] result = new double[count];
            if (count == length)
            {
                Array.Copy(array, start, result, 0, length);
            }
            else
            {
                count = 0;
                for (int i = 0; i < length; i++)
                {
                    if (!Double.IsNaN(array[i + start]))
                        result[count++] = array[i + start];
                }
            }
            return result;
        }

        /// <summary>
        /// Returns the relative distance between two numbers.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="rel">An offset to avoid division by zero.</param>
        /// <returns><c>abs(x - y)/(min(abs(x),abs(y)) + rel)</c>. 
        /// Matching infinities give zero.  Any NaN gives NaN.
        /// </returns>
        /// <remarks>
        /// This routine is often used to measure the error of y in estimating x.
        /// </remarks>
        public static double AbsDiff(double x, double y, double rel)
        {
            if (x == y)
                return 0; // catches infinities
            return Math.Abs(x - y) / (Math.Min(Math.Abs(x), Math.Abs(y)) + rel);
        }

        /// <summary>
        /// Returns the relative distance between two numbers.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="rel">An offset to avoid division by zero.</param>
        /// <returns><c>abs(x - y)/(min(abs(x),abs(y)) + rel)</c>. 
        /// Matching infinities or NaNs give zero.
        /// </returns>
        /// <remarks>
        /// This routine is often used to measure the error of y in estimating x.
        /// </remarks>
        public static double AbsDiffAllowingNaNs(double x, double y, double rel)
        {
            if (double.IsNaN(x)) return double.IsNaN(y) ? 0 : double.PositiveInfinity;
            if (double.IsNaN(y)) return double.PositiveInfinity;
            return AbsDiff(x, y, rel);
        }

        /// <summary>
        /// Returns the distance between two numbers.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns><c>abs(x - y)</c>. 
        /// Matching infinities give zero.  Any NaN gives NaN.
        /// </returns>
        public static double AbsDiff(double x, double y)
        {
            if (x == y)
                return 0; // catches infinities
            return Math.Abs(x - y);
        }

        /// <summary>
        /// Returns the distance between two numbers.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns><c>abs(x - y)</c>. 
        /// Matching infinities or NaNs give zero.
        /// </returns>
        public static double AbsDiffAllowingNaNs(double x, double y)
        {
            if (double.IsNaN(x)) return double.IsNaN(y) ? 0 : double.PositiveInfinity;
            if (double.IsNaN(y)) return double.PositiveInfinity;
            return AbsDiff(x, y);
        }

        /// <summary>
        /// Returns true if two numbers are equal when represented in double precision.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static bool AreEqual(double x, double y)
        {
            // These casts force the runtime to represent in double precision instead of a higher internal precision.
            // See section I.12.1.3 of the ECMA specification.
            return (double)x == (double)y;
            // The below approach does not work when x or y is infinity.  For example, try running GaussianTest in 32-bit Release mode.
            // Also, the ECMA specification allows differences larger than double.Epsilon, as long as the internal precision is higher than double precision.
            ////return AbsDiff(x, y) < double.Epsilon;
        }

        /// <summary>
        /// Returns the positive distance between a value and the next representable value that is larger in magnitude.
        /// </summary>
        /// <param name="value">Any double-precision value.</param>
        /// <returns></returns>
        public static double Ulp(double value)
        {
            if (double.IsNaN(value)) return value;
            value = Math.Abs(value);
            if (double.IsPositiveInfinity(value)) return 0;
            long bits = BitConverter.DoubleToInt64Bits(value);
            double nextValue = BitConverter.Int64BitsToDouble(bits + 1);
            return nextValue - value;
        }

        /// <summary>
        /// Returns the smallest double precision number greater than value, if one exists.  Otherwise returns value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double NextDouble(double value)
        {
            if (value < 0) return -PreviousDouble(-value);
            value = Math.Abs(value); // needed to handle -0
            if (double.IsNaN(value)) return value;
            if (double.IsPositiveInfinity(value)) return value;
            long bits = BitConverter.DoubleToInt64Bits(value);
            return BitConverter.Int64BitsToDouble(bits + 1);
        }

        /// <summary>
        /// Returns the smallest double precision number such that when value is subtracted from it,
        /// the result is strictly greater than zero, if one exists.  Otherwise returns value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double NextDoubleWithPositiveDifference(double value)
        {
            // This relies on denormalization being enabled in .NET.
            return NextDouble(value);
        }

        /// <summary>
        /// Returns the largest double precision number less than value, if one exists.  Otherwise returns value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double PreviousDouble(double value)
        {
            if (value <= 0) return -NextDouble(-value);
            if (double.IsNaN(value)) return value;
            if (double.IsNegativeInfinity(value)) return value;
            long bits = BitConverter.DoubleToInt64Bits(value);
            return BitConverter.Int64BitsToDouble(bits - 1);
        }

        /// <summary>
        /// Returns the biggest double precision number such that when it is subtracted from value,
        /// the result is strictly greater than zero, if one exists.  Otherwise returns value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double PreviousDoubleWithPositiveDifference(double value)
        {
            // This relies on denormalization being enabled in .NET.
            return PreviousDouble(value);
        }

        /// <summary>
        /// Returns the largest value such that value * denominator &lt;= numerator.
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <returns></returns>
        internal static double LargestDoubleRatio(double numerator, double denominator)
        {
            if (denominator < 0) return LargestDoubleRatio(-numerator, -denominator);
            if (denominator == 0)
            {
                if (double.IsNaN(numerator)) return double.PositiveInfinity;
                else if (numerator >= 0)
                    return double.MaxValue;
                else
                    return double.NaN;
            }
            // denominator > 0
            if (double.IsPositiveInfinity(numerator)) return numerator;
            if (double.IsPositiveInfinity(denominator))
            {
                if (double.IsNaN(numerator)) return 0;
                else return PreviousDouble(0);
            }
            double lowerBound, upperBound;
            if (denominator >= 1)
            {
                if (double.IsNegativeInfinity(numerator))
                {
                    upperBound = NextDouble(numerator) / denominator;
                    if (AreEqual(upperBound * denominator, numerator)) return upperBound;
                    else return PreviousDouble(upperBound);
                }
                // ratio cannot be infinite since numerator is not infinite.
                double ratio = numerator / denominator;
                lowerBound = PreviousDouble(ratio);
                upperBound = NextDouble(ratio);
            }
            else // 0 < denominator < 1
            {
                // avoid infinite bounds
                if (numerator == double.Epsilon) lowerBound = numerator / denominator / 2; // cannot overflow
                else if (numerator == 0) lowerBound = 0;
                else lowerBound = (double)Math.Max(double.MinValue, Math.Min(double.MaxValue, PreviousDouble(numerator) / denominator));
                if (numerator == -double.Epsilon) upperBound = numerator / denominator / 2; // cannot overflow
                else upperBound = (double)Math.Min(double.MaxValue, NextDouble(numerator) / denominator);
                if (double.IsNegativeInfinity(upperBound)) return upperBound; // must have ratio < -1 and denominator > 1
            }
            int iterCount = 0;
            while (true)
            {
                iterCount++;
                double value = (double)Average(lowerBound, upperBound);
                if (value < lowerBound || value > upperBound) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, denominator={denominator:g17}, ratio={numerator:g17}");
                if ((double)(value * denominator) <= numerator)
                {
                    double value2 = NextDouble(value);
                    if (value2 == value || (double)(value2 * denominator) > numerator)
                    {
                        // Used for performance debugging
                        //if (iterCount > 100)
                        //    throw new Exception();
                        return value;
                    }
                    else
                    {
                        // value is too low
                        lowerBound = value2;
                        if (lowerBound > upperBound || double.IsNaN(lowerBound)) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, denominator={denominator:g17}, ratio={numerator:g17}");
                    }
                }
                else
                {
                    // value is too high
                    upperBound = PreviousDouble(value);
                    if (lowerBound > upperBound || double.IsNaN(upperBound)) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, denominator={denominator:g17}, ratio={numerator:g17}");
                }
            }
        }

        /// <summary>
        /// Returns the largest value such that value/denominator &lt;= ratio.
        /// </summary>
        /// <param name="ratio"></param>
        /// <param name="denominator"></param>
        /// <returns></returns>
        internal static double LargestDoubleProduct(double ratio, double denominator)
        {
            if (denominator < 0) return LargestDoubleProduct(-ratio, -denominator);
            if (denominator == 0)
            {
                if (double.IsNaN(ratio)) return 0;
                else if (double.IsPositiveInfinity(ratio))
                    return double.PositiveInfinity;
                else if (double.IsNegativeInfinity(ratio))
                    return PreviousDouble(0);
                else
                    return double.NaN;
            }
            // denominator > 0
            if (double.IsPositiveInfinity(denominator))
            {
                if (double.IsNaN(ratio)) return denominator;
                else return double.MaxValue;
            }
            if (double.IsPositiveInfinity(ratio)) return ratio;
            double lowerBound, upperBound;
            if (denominator <= 1)
            {
                if (double.IsNegativeInfinity(ratio))
                {
                    upperBound = denominator * NextDouble(ratio);
                    if (AreEqual(upperBound / denominator, ratio)) return upperBound;
                    else return PreviousDouble(upperBound);
                }
                // product cannot be infinite since ratio is not infinite.
                double product = denominator * ratio;
                lowerBound = PreviousDouble(product);
                upperBound = NextDouble(product);
            }
            else // 1 < denominator <= double.MaxValue
            {
                // avoid infinite bounds
                if (ratio == double.Epsilon) lowerBound = denominator * ratio / 2; // cannot overflow
                else if (ratio == 0) lowerBound = 0;
                else lowerBound = (double)Math.Max(double.MinValue, Math.Min(double.MaxValue, denominator * PreviousDouble(ratio)));
                if (ratio == -double.Epsilon) upperBound = denominator * ratio / 2; // cannot overflow
                else upperBound = (double)Math.Min(double.MaxValue, denominator * NextDouble(ratio));
                if (double.IsNegativeInfinity(upperBound)) return upperBound; // must have ratio < -1 and denominator > 1
            }
            int iterCount = 0;
            while (true)
            {
                iterCount++;
                double value = (double)Average(lowerBound, upperBound);
                if (value < lowerBound || value > upperBound) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, denominator={denominator:g17}, ratio={ratio:g17}");
                if ((double)(value / denominator) <= ratio)
                {
                    double value2 = NextDouble(value);
                    if (value2 == value || (double)(value2 / denominator) > ratio)
                    {
                        // Used for performance debugging
                        //if (iterCount > 100)
                        //    throw new Exception();
                        return value;
                    }
                    else
                    {
                        // value is too low
                        lowerBound = value2;
                        if (lowerBound > upperBound || double.IsNaN(lowerBound)) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, denominator={denominator:g17}, ratio={ratio:g17}");
                    }
                }
                else
                {
                    // value is too high
                    upperBound = PreviousDouble(value);
                    if (lowerBound > upperBound || double.IsNaN(upperBound)) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, denominator={denominator:g17}, ratio={ratio:g17}");
                }
            }
        }

        /// <summary>
        /// Returns the largest value such that value - b &lt;= sum.
        /// </summary>
        /// <param name="sum"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        internal static double LargestDoubleSum(double b, double sum)
        {
            if (double.IsPositiveInfinity(b))
            {
                if (double.IsNaN(sum)) return double.PositiveInfinity;
                else return double.MaxValue;
            }
            if (double.IsNegativeInfinity(b))
            {
                if (double.IsNaN(sum)) return double.NegativeInfinity;
                else return double.PositiveInfinity;
            }
            if (double.IsPositiveInfinity(sum)) return sum;
            double lowerBound = PreviousDoubleWithPositiveDifference(b + sum);
            double upperBound;
            if (Math.Abs(sum) > Math.Abs(b))
            {
                // Cast to double to prevent upperBound from landing between double.MaxValue and infinity on a 32-bit platform.
                upperBound = (double)(b + NextDoubleWithPositiveDifference(sum));
            }
            else
            {
                upperBound = (double)(NextDoubleWithPositiveDifference(b) + sum);
            }
            int iterCount = 0;
            while (true)
            {
                iterCount++;
                double value = (double)Average(lowerBound, upperBound);
                //double value = RepresentationMidpoint(lowerBound, upperBound);
                if (value < lowerBound || value > upperBound) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, b={b:g17}, sum={sum:g17}");
                if ((double)(value - b) <= sum)
                {
                    double value2 = NextDouble(value);
                    if (value2 == value || (double)(value2 - b) > sum)
                    {
                        // Used for performance debugging
                        //if (iterCount > 100)
                        //    throw new Exception();
                        return value;
                    }
                    else
                    {
                        // value is too low
                        lowerBound = value2;
                        if (lowerBound > upperBound || double.IsNaN(lowerBound)) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, b={b:g17}, sum={sum:g17}");
                    }
                }
                else
                {
                    // value is too high
                    upperBound = PreviousDouble(value);
                    if (lowerBound > upperBound || double.IsNaN(upperBound)) throw new Exception($"value={value:g17}, lowerBound={lowerBound:g17}, upperBound={upperBound:g17}, b={b:g17}, sum={sum:g17}");
                }
            }
        }

        /// <summary>
        /// Returns (a+b)/2, avoiding overflow.  The result is guaranteed to be between a and b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double Average(double a, double b)
        {
            // This version avoids underflow but may overflow.
            double midpoint = (a + b) / 2;
            if (double.IsInfinity(midpoint))
            {
                // This version avoids overflow but may underflow.
                // Luckily, if we are in this branch, it cannot underflow.
                midpoint = 0.5 * a + 0.5 * b;
            }
            return midpoint;
        }

        /// <summary>
        /// Returns (a+b)/2, avoiding overflow.  The result is guaranteed to be between a and b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static long Average(long a, long b)
        {
            return a / 2 + b / 2 + ((a % 2) + (b % 2)) / 2;
        }

        /// <summary>
        /// Returns (weight1*value1 + weight2*value2)/(weight1 + weight2), avoiding underflow and overflow.  
        /// The result is guaranteed to be between value1 and value2, monotonic in value1 and value2,
        /// and equal to Average(value1, value2) when weight1 == weight2.
        /// </summary>
        /// <param name="weight1">Any number &gt;=0</param>
        /// <param name="value1">Any number</param>
        /// <param name="weight2">Any number &gt;=0</param>
        /// <param name="value2">Any number</param>
        /// <returns></returns>
        public static double WeightedAverage(double weight1, double value1, double weight2, double value2)
        {
            const double ScaleBM1024 = 5.5626846462680035E-309; // Math.ScaleB(1,-1024)
            if (weight1 < weight2)
            {
                return WeightedAverage(weight2, value2, weight1, value1);
            }
            // weight1 >= weight2
            if (MMath.AreEqual(weight2, 0) || weight1 > double.MaxValue)
            {
                if (MMath.AreEqual(weight1, 0) || weight2 > double.MaxValue)
                {
                    return MMath.Average(value1, value2);
                }
                else
                {
                    return value1;
                }
            }
            // Since overflow is easily detected, we start by scaling to avoid underflow.
            // Normalize by weight2 to avoid underflow.
            // weight1/weight2 >= 1
            double ratio = weight1 / weight2;
            double result;
            if (ratio > double.MaxValue)
            {
                // Instead of scaling by 1/weight2, choose scale so that scale*(weight1+weight2) <= double.MaxValue.
                // We know that (weight1+weight2) == weight1
                // weight2 < 1
                const double nextBelowOne = double.MaxValue * ScaleBM1024; // nextBelowOne < 1
                if (weight1 < 1)
                {
                    double scale = nextBelowOne / weight1; // scale >= 1 but scale*weight2 << 1
                    double scaleWeight2Value2 = scale * weight2 * value2; // cannot overflow
                    result = (nextBelowOne * value1 + scaleWeight2Value2) / nextBelowOne; // cannot overflow
                }
                else
                {
                    double scale = double.MaxValue / weight1; // scale >= 1 but scale*weight2 < 1
                    // Below is equivalent to:
                    // result = (scale * weight1 * value1 + scale * weight2 * value2) / (scale * weight1 + scale * weight2);
                    double scaleWeight2Value2 = scale * weight2 * value2; // cannot overflow
                    result = (double.MaxValue * value1 + scaleWeight2Value2) / double.MaxValue;
                    if (double.IsNaN(result) || double.IsInfinity(result))
                    {
                        // Overflow happened.  Scale down to avoid overflow.
                        // We scale by a power of 2 to ensure that the result is rounded the same way as above.
                        // Otherwise, the function would not always be monotonic in value1 and value2.
                        // IsInfinity(result) implies value1 > 1 so the first term cannot underflow.
                        // This cannot overflow, because the second term's magnitude is less than 1.
                        result = (nextBelowOne * value1 + scaleWeight2Value2 * ScaleBM1024) / nextBelowOne;
                        if (double.IsNaN(result)) return value1 + value2;
                    }
                }
            }
            else
            {
                result = (ratio * value1 + value2) / (ratio + 1);
                // IsNaN(result) happens in 2 ways:
                // 1. denominator is infinity and numerator is +/-infinity
                // 2. (weight1/weight2)*value1 + value2 is NaN
                //    a. (weight1/weight2)*value1=inf and value2=-inf
                //    b. (weight1/weight2)*value1 is NaN.  This implies (weight1/weight2)=inf and value1=0.
                // In all cases, we must have weight1/weight2 > 1.
                if (double.IsNaN(result) || double.IsInfinity(result))
                {
                    // Overflow happened.  Scale down to avoid overflow.
                    // We scale by a power of 2 to ensure that the result is rounded the same way as above.
                    // ratio * ScaleBM1024 < 1 therefore ratio * ScaleBM1024 * value1 cannot overflow.
                    // value2 * ScaleBM1024 < 1 therefore the sum cannot overflow.
                    // Both terms in the denominator are < 1 therefore the denominator cannot overflow.
                    result = (ratio * ScaleBM1024 * value1 + value2 * ScaleBM1024) / (ratio * ScaleBM1024 + ScaleBM1024);
                }
            }
            result = Math.Min(result, Math.Max(value1, value2));
            result = Math.Max(result, Math.Min(value1, value2));
            return result;
        }

        private static double RepresentationMidpoint(double lower, double upper)
        {
            if (lower == 0)
            {
                if (upper < 0) return -RepresentationMidpoint(-lower, -upper);
                else if (upper == 0) return lower;
                // fall through
            }
            else if (lower < 0)
            {
                if (upper <= 0) return -RepresentationMidpoint(-lower, -upper);
                else return 0; // upper > 0
            }
            else if (upper < 0) return 0; // lower > 0
            // must have lower >= 0, upper >= 0
            long lowerBits = BitConverter.DoubleToInt64Bits(lower);
            long upperBits = BitConverter.DoubleToInt64Bits(upper);
            long midpoint = MMath.Average(lowerBits, upperBits);
            return BitConverter.Int64BitsToDouble(midpoint);
        }

        /// <summary>
        /// Returns a decimal string that exactly equals a double-precision number, unlike double.ToString which always returns a rounded result.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static string ToStringExact(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x) || x == 0) return x.ToString(System.Globalization.CultureInfo.InvariantCulture);
            long bits = BitConverter.DoubleToInt64Bits(x);
            ulong fraction = Convert.ToUInt64(bits & 0x000fffffffffffff); // 52 bits
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
            BigInteger big;
            if (exponent >= 0)
            {
                big = BigInteger.Pow(2, exponent) * fraction;
                return $"{sign}{big}";
            }
            else
            {
                // Rewrite 2^-4 as 5^4 * 10^-4
                big = BigInteger.Pow(5, -exponent) * fraction;
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
                    BigInteger pow10 = BigInteger.Pow(10, -exponent);
                    BigInteger integerPart = big / pow10;
                    BigInteger fractionalPart = big - integerPart * pow10;
                    string zeros = new string('0', -exponent);
                    return $"{sign}{integerPart}.{fractionalPart.ToString(zeros)}";
                }
            }
        }

        #region Enumerations and constants

        /// <summary>
        /// The Euler-Mascheroni Constant.
        /// </summary>
        public const double EulerGamma = 0.57721566490153286060651209;

        /// <summary>
        /// Ulp(1.0) == NextDouble(1.0) - 1.0
        /// </summary>
        public static readonly double Ulp1 = Ulp(1.0);

        /// <summary>
        /// Math.Sqrt(Ulp(1.0))
        /// </summary>
        public static readonly double SqrtUlp1 = Math.Sqrt(Ulp1);

        /// <summary>
        /// Math.Pow(Ulp(1.0), 1.0 / 3)
        /// </summary>
        public static readonly double CbrtUlp1 = Math.Pow(Ulp1, 1.0 / 3);

        /// <summary>
        /// Digamma(1)
        /// </summary>
        public const double Digamma1 = -EulerGamma;

        private const double TOLERANCE = 1.0e-7;

        private const double GammaLnLargeX = 10;

        /// <summary>
        /// Math.Sqrt(2*Math.PI)
        /// </summary>
        public const double Sqrt2PI = 2.5066282746310005;

        /// <summary>
        /// 1.0/Math.Sqrt(2*Math.PI)
        /// </summary>
        public const double InvSqrt2PI = 0.398942280401432677939946;

        /// <summary>
        /// Math.Log(Math.Sqrt(2*Math.PI)).
        /// </summary>
        public const double LnSqrt2PI = 0.91893853320467274178;

        /// <summary>
        /// Math.Log(Math.PI)
        /// </summary>
        public const double LnPI = 1.14472988584940;

        /// <summary>
        /// Math.Log(2)
        /// </summary>
        public const double Ln2 = 0.6931471805599453094172321214581765680;

        /// <summary>
        /// Math.Sqrt(2)
        /// </summary>
        public const double Sqrt2 = 1.41421356237309504880;

        /// <summary>
        /// Math.Sqrt(3)
        /// </summary>
        public const double Sqrt3 = 1.7320508075688772935274463415059;

        /// <summary>
        /// Math.Sqrt(0.5)
        /// </summary>
        public const double SqrtHalf = 0.70710678118654752440084436210485;

        /// <summary>
        /// Zeta(2) = Trigamma(1) = pi^2/6.
        /// </summary>
        public const double Zeta2 = 1.644934066848226436472415;

        /// <summary>
        /// Tetragamma(1) = -2 Zeta(3)
        /// </summary>
        private const double M2Zeta3 = -2.404113806319188570799476;

        /// <summary>
        /// Zeta4 = pi^4/90 = polygamma(3,1)/6
        /// </summary>
        private const double Zeta4 = 1.0823232337111381915;

        /// <summary>
        /// NormCdf(x)/NormPdf(x) for x = 0, -1, -2, -3, ..., -16
        /// </summary>
        private static readonly double[] c_normcdf_table =
            {
                Sqrt2PI/2, 0.655679542418798471543871, .421369229288054473, 0.30459029871010329573361254651, .236652382913560671,
                0.1928081047153157648774657, .162377660896867462, 0.140104183453050241599534, .123131963257932296, 0.109787282578308291230, .0990285964717319214,
                0.09017567550106468227978, .0827662865013691773, 0.076475761016248502993495, .0710695805388521071, 0.0663742358232501735, .0622586659950261958
            };

        private const double Ln2SqrtPi = 1.265512123484645396488945797;

        private static readonly double[] c_erfc_table =
            // Taylor series (not as good as Chebyshev)
            // Maple command for Taylor series: asympt(subs(x=2*y-2,log(erfc(x)*(1+x/2))+x*x),y);
            //{ - Ln2SqrtPi, 1, 3.0 / 8, 1.0 / 12, -11.0 / 128, -23.0 / 160, -47.0 / 512, 53.0 / 1792, 2177.0/16384, 2249.0/18432 };
            // Chebyshev series
            // Maple command: chebyshev( subs(x=2/y-2,log(erfc(x)*(1+x/2))+x*x), y=0..1, 1e-6);
            //{ -1.26551220, 1.000023566, 0.3740932172, 0.0967773480, -0.1862670280, 0.2788294034, -1.135160988, 1.488487954, -0.8221427183, 0.1708715269 };
            // from Ralf (better than above)
            { -1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277 };

        // Maple command: chebyshev( subs(x=2/y-2,log(erfc(x)*(1+x/2))+x*x), y=0..1, 1e-8);
        // more accurate, but more expensive
        //{ -1.265512122, 0.9999999460, 0.3750003113, 0.08331553737, -0.0857441378, -0.142407635, -0.1269326291, 0.295066080, -0.9476268748, 2.769131338, -3.938783592, 2.943051024, -1.142311460, 0.1837542133 };

        private static readonly double NormalCdfRatioConfracTolerance = Ulp1 * 1024;

        // Math.Exp(log0) == 0
        private static readonly double log0 = Math.Log(double.Epsilon) - Ln2;
        // 1-Math.Exp(logHalfUlpPrev1) == 1
        private static readonly double logHalfUlpPrev1 = Math.Log((1.0 - PreviousDouble(1.0)) / 2);
        private static readonly double logisticGaussianQuadratureRelativeTolerance = 512 * Ulp1;
        private static readonly double logisticGaussianDerivativeQuadratureRelativeTolerance = 1024 * 512 * Ulp1;
        private static readonly double logisticGaussianSeriesApproximmationThreshold = 1e-8;

        #endregion
    }
}