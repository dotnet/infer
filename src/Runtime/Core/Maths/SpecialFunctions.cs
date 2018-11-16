// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions; // for Gaussian.GetLogProb

namespace Microsoft.ML.Probabilistic.Math
{
    using System;
    using Microsoft.ML.Probabilistic.Collections;

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
            List<int> twoNPlus1Factorial = new List<int>();
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
            Func<int, double> aFunc = n =>
                {
                    double neg = (n % 2) == 0 ? 1.0 : -1.0;
                    return (a <= b) ?
                        2.0 * q * (1.0 + neg * Math.Pow(a / b, n + 1)) / (2.0 + n) :
                        2.0 * p * (neg + Math.Pow(b / a, n + 1)) / (2.0 + n);
                };

            // Assumes aTerm has been populated up to n
            Func<int, double, double> bFunc = (n, r) =>
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
                };

            // Assumes aTerm has been populated up to n
            Func<int, double> cFunc = n => bFunc(n - 1, -n / 2.0) / n;
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

        /// <summary>
        /// Computes the regularized incomplete beta function: int_0^x t^(a-1) (1-t)^(b-1) dt / Beta(a,b)
        /// </summary>
        /// <param name="x">The first argument - any real number between 0 and 1.</param>
        /// <param name="a">The second argument - any real number greater than 0.</param>
        /// <param name="b">The third argument - any real number greater than 0.</param>
        /// <param name="epsilon">A tolerance for terminating the series calculation.</param>
        /// <returns>The incomplete beta function at (<paramref name="x"/>, <paramref name="a"/>, <paramref name="b"/>).</returns>
        /// <remarks>The beta function is obtained by setting x to 1.</remarks>
        public static double Beta(double x, double a, double b, double epsilon = 1e-15)
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
            /* Negative values */
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

            if (x > 180)
            {
                return Double.PositiveInfinity;
            }

            return Math.Exp(GammaLn(x));
        }

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
                // Reference: https://dlmf.nist.gov/5.7#E3
                double dx = x - 2;
                double sum = 0;
                for (int i = gammaTaylorCoefficients.Length - 1; i >= 0; i--)
                {
                    sum = dx * (gammaTaylorCoefficients[i] + sum);
                }
                sum = dx * (1 + Digamma1 + sum);
                result += sum;
                return result;
            }
            else // x >= 6
            {
                double sum = LnSqrt2PI;
                while (x < 10)
                {
                    sum -= Math.Log(x);
                    x++;
                }
                // x >= 10
                // Use asymptotic series
                return GammaLnSeries(x) + (x - 0.5) * Math.Log(x) - x + sum;
            }
        }

        /* Python code to generate this table (must not be indented):
for k in range(2,26):
		print("            %0.20g," % ((-1)**k*(zeta(k)-1)/k))
         */
        private static readonly double[] gammaTaylorCoefficients =
        {
            0.32246703342411320303,
            -0.06735230105319810201,
            0.020580808427784546416,
            -0.0073855510286739856768,
            0.0028905103307415229257,
            -0.0011927539117032610189,
            0.00050966952474304234172,
            -0.00022315475845357938579,
            9.945751278180853098e-05,
            -4.4926236738133142046e-05,
            2.0507212775670691067e-05,
            -9.4394882752683967152e-06,
            4.3748667899074873274e-06,
            -2.0392157538013666132e-06,
            9.551412130407419353e-07,
            -4.4924691987645661855e-07,
            2.1207184805554664645e-07,
            -1.0043224823968100408e-07,
            4.7698101693639803983e-08,
            -2.2711094608943166813e-08,
            1.0838659214896952939e-08,
            -5.1834750419700474714e-09,
            2.4836745438024780616e-09,
            -1.1921401405860913615e-09,
            5.7313672416788612175e-10,
        };

        private static double[] DigammaLookup;

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
            const int tableLength = 100;
            int xAsInt = (int)x;
            if ((xAsInt == x) && (xAsInt < tableLength))
            {
                if (DigammaLookup == null)
                {
                    double[] table = new double[tableLength];
                    table[0] = double.NegativeInfinity;
                    table[1] = Digamma1;
                    for (int i = 2; i < table.Length; i++)
                        table[i] = table[i - 1] + 1.0 / (i - 1);
                    // This is thread-safe because read/write to a reference type is atomic. See
                    // http://msdn.microsoft.com/en-us/library/aa691278%28VS.71%29.aspx
                    DigammaLookup = table;
                }
                return DigammaLookup[xAsInt];
            }

            // The threshold for applying de Moivre's expansion for the digamma function.
            const double c_digamma_small = 1e-6;

            /* Use Taylor series if argument <= S */
            if (x <= c_digamma_small)
            {
                return (Digamma1 - 1 / x + Zeta2 * x);
            }

            if (x <= 2.5)
            {
                double result2 = 1 + Digamma1;
                while (x < 1.5)
                {
                    result2 -= 1 / x;
                    x++;
                }
                double dx = x - 2;
                double sum2 = 0;
                for (int i = gammaTaylorCoefficients.Length - 1; i >= 0; i--)
                {
                    sum2 = dx * (gammaTaylorCoefficients[i] * (i + 2) + sum2);
                }
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
            double sum = 0;
            for (int i = c_digamma_series.Length - 1; i >= 0; i--)
            {
                sum = invX2 * (c_digamma_series[i] + sum);
            }
            result -= sum;
            return result;
        }

        /// <summary>
        /// Coefficients of de Moivre's expansion for the digamma function.
        /// Each coefficient is B_{2j}/(2j) where B_{2j} are the Bernoulli numbers, starting from j=1
        /// </summary>
        private static readonly double[] c_digamma_series =
        {
            1.0/12, -1.0/120, 1.0/252, -1.0/240, 1.0/132,
            -691.0/32760, 1.0/12, /* -3617.0/8160, 43867.0/14364, -174611.0/6600 */
        };

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

            /* Use Taylor series if argument <= small */
            if (x <= c_trigamma_small)
            {
                return (1.0 / (x * x) + Zeta2 + M2Zeta3 * x);
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
            double sum = 0;
            for (int i = c_trigamma_series.Length - 1; i >= 0; i--)
            {
                sum = invX2 * (c_trigamma_series[i] + sum);
            }
            result += (1 + sum) / x;
            return result;
        }

        /// <summary>
        /// Coefficients of de Moivre's expansion for the trigamma function.
        /// Each coefficient is B_{2j} where B_{2j} are the Bernoulli numbers, starting from j=1
        /// </summary>
        private static readonly double[] c_trigamma_series = { 1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730, 7.0 / 6, -3617.0 / 510 };

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
                return -2 / (x * x * x) + M2Zeta3 + 6 * Zeta4 * x;
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
            double sum = 0;
            for (int i = c_tetragamma_series.Length - 1; i >= 0; i--)
            {
                sum = invX2 * (c_tetragamma_series[i] + sum);
            }
            result += sum;
            return result;
        }

        /// <summary>
        /// Coefficients of de Moivre's expansion for the quadgamma function.
        /// Each coefficient is -(2j+1) B_{2j} where B_{2j} are the Bernoulli numbers, starting from j=0
        /// </summary>
        private static readonly double[] c_tetragamma_series = { -1, -.5, +1 / 6.0, -1 / 6.0, +3 / 10.0, -5 / 6.0, 691.0 / 210, -35.0 / 2 };

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
            if (k < 0 || k > n)
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
        /// Compute the regularized upper incomplete Gamma function: int_x^inf t^(a-1) exp(-t) dt / Gamma(a)
        /// </summary>
        /// <param name="a">The shape parameter, &gt; 0</param>
        /// <param name="x">The lower bound of the integral, &gt;= 0</param>
        /// <returns></returns>
        public static double GammaUpper(double a, double x)
        {
            // special cases:
            // GammaUpper(1,x) = exp(-x)
            // GammaUpper(a,0) = 1
            // GammaUpper(a,x) = GammaUpper(a-1,x) + x^(a-1) exp(-x) / Gamma(a)
            if (x < 0)
                throw new ArgumentException($"x ({x}) < 0");
            if (a <= 0)
                throw new ArgumentException($"a ({a}) <= 0");
            if (x == 0) return 1; // avoid 0/0
            // Use the criterion from Gautschi (1979) to determine whether GammaLower(a,x) or GammaUpper(a,x) is smaller.
            // useLower = true means that GammaLower is smaller.
            bool useLower;
            if (x > 0.25)
                useLower = (a > x + 0.25);
            else
                useLower = (a > -MMath.Ln2 / Math.Log(x));
            if (useLower)
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
            // useLower = true means that GammaLower is smaller.
            bool useLower;
            if (x > 0.25)
                useLower = (a > x + 0.25);
            else
                useLower = (a > -MMath.Ln2 / Math.Log(x));
            if (useLower)
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
        /// <param name="x"></param>
        /// <param name="upper">If true, compute the upper incomplete Gamma function</param>
        /// <returns></returns>
        private static double GammaAsympt(double a, double x, bool upper)
        {
            if (a <= 20)
                throw new Exception("a <= 20");
            double xOverAMinus1 = (x - a) / a;
            double phi = xOverAMinus1 - MMath.Log1Plus(xOverAMinus1);
            double y = a * phi;
            double z = Math.Sqrt(2 * phi);
            if (x <= a)
                z *= -1;
            double[] b = new double[GammaLowerAsympt_C0.Length];
            b[b.Length - 1] = GammaLowerAsympt_C0[b.Length - 1];
            double sum = b[b.Length - 1];
            b[b.Length - 2] = GammaLowerAsympt_C0[b.Length - 2];
            sum = z * sum + b[b.Length - 2];
            for (int i = b.Length - 3; i >= 0; i--)
            {
                b[i] = b[i + 2] * (i + 2) / a + GammaLowerAsympt_C0[i];
                sum = z * sum + b[i];
            }
            sum *= a / (a + b[1]);
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
        private static double[] GammaLowerAsympt_C0 =
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
            throw new Exception(string.Format("GammaLowerSeries not converging for a={0} x={1}", a, x));
        }

        /// <summary>
        /// Compute the regularized upper incomplete Gamma function by a series expansion
        /// </summary>
        /// <param name="a">The shape parameter, &gt; 0</param>
        /// <param name="x">The lower bound of the integral, &gt;= 0</param>
        /// <returns></returns>
        private static double GammaUpperSeries(double a, double x)
        {
            // this series should only be applied when x is small
            // the series is: 1 - x^a sum_{k=0}^inf (-x)^k /(k! Gamma(a+k+1))
            // = (1 - 1/Gamma(a+1)) + (1 - x^a)/Gamma(a+1) - x^a sum_{k=1}^inf (-x)^k/(k! Gamma(a+k+1))
            double xaMinus1 = MMath.ExpMinus1(a * Math.Log(x));
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
            double offset = -xaMinus1 * aReciprocalFactorial - aReciprocalFactorialMinus1;
            double scale = 1 - offset;
            double term = x / (a + 1) * a;
            double sum = term;
            for (int i = 1; i < 1000; i++)
            {
                term *= -(a + i) * x / ((a + i + 1) * (i + 1));
                double sumOld = sum;
                sum += term;
                //Console.WriteLine("{0}: {1}", i, sum);
                if (AreEqual(sum, sumOld))
                    return scale * sum + offset;
            }
            throw new Exception(string.Format("GammaUpperSeries not converging for a={0} x={1}", a, x));
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
            double sum = 0;
            double term = x;
            for (int i = 0; i < reciprocalFactorialMinus1Coeffs.Length; i++)
            {
                sum += reciprocalFactorialMinus1Coeffs[i] * term;
                term *= x;
            }
            return sum;
        }

        /* using http://sagecell.sagemath.org/ (must not be indented)
var('x');
f = 1/gamma(x+1)-1
[c[0].n(100) for c in f.taylor(x,0,16).coefficients()]
         */
        private static readonly double[] reciprocalFactorialMinus1Coeffs =
        {
            0.57721566490153286060651209008,
            -0.65587807152025388107701951515,
            -0.042002635034095235529003934875,
            0.16653861138229148950170079510,
            -0.042197734555544336748208301289,
            -0.0096219715278769735621149216721,
            0.0072189432466630995423950103404,
            -0.0011651675918590651121139710839,
            -0.00021524167411495097281572996312,
            0.00012805028238811618615319862640,
            -0.000020134854780788238655689391375,
            -1.2504934821426706573453594884e-6,
            1.1330272319816958823741296196e-6,
            -2.0563384169776071034501526118e-7,
            6.1160951044814158178625767024e-9,
            5.0020076444692229300557063872e-9
        };

        /// <summary>
        /// Computes <c>x^a e^(-x)/Gamma(a)</c> to high accuracy.
        /// </summary>
        /// <param name="a">A positive real number</param>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double GammaUpperScale(double a, double x)
        {
            if (double.IsPositiveInfinity(x) || double.IsPositiveInfinity(a))
                return 0;
            double scale;
            if (a < 10)
                scale = Math.Exp(a * Math.Log(x) - x - GammaLn(a));
            else
            {
                double xia = x / a;
                double phi = xia - 1 - Math.Log(xia);
                scale = Math.Exp(0.5 * Math.Log(a) - MMath.LnSqrt2PI - GammaLnSeries(a) - a * phi);
            }
            return scale;
        }

        // Origin: James McCaffrey, http://msdn.microsoft.com/en-us/magazine/dn520240.aspx
        /// <summary>
        /// Compute the regularized upper incomplete Gamma function by a continued fraction
        /// </summary>
        /// <param name="a">A real number &gt; 0</param>
        /// <param name="x">A real number &gt;= 1.1</param>
        /// <returns></returns>
        private static double GammaUpperConFrac(double a, double x)
        {
            double scale = GammaUpperScale(a, x);
            if (scale == 0)
                return scale;
            // the confrac coefficients are:
            // a_i = -i*(i-a)
            // b_i = x+1-a+2*i
            // the confrac is evaluated using Lentz's algorithm
            double b = x + 1.0 - a;
            const double tiny = 1e-30;
            double c = 1.0 / tiny;
            double d = 1.0 / b;
            double h = d * scale;
            for (int i = 1; i < 1000; ++i)
            {
                double an = -i * (i - a);
                b += 2.0;
                d = an * d + b;
                if (Math.Abs(d) < tiny)
                    d = tiny;
                c = b + an / c;
                if (Math.Abs(c) < tiny)
                    c = tiny;
                d = 1.0 / d;
                double del = d * c;
                double oldH = h;
                h *= del;
                if (AreEqual(h, oldH))
                    return h;
            }
            throw new Exception(string.Format("GammaUpperConFrac not converging for a={0} x={1}", a, x));
        }

        /// <summary>
        /// Computes <c>GammaLn(x) - (x-0.5)*log(x) + x - 0.5*log(2*pi)</c> for x &gt;= 10
        /// </summary>
        /// <param name="x">A real number &gt;= 10</param>
        /// <returns></returns>
        private static double GammaLnSeries(double x)
        {
            // GammaLnSeries(10) = 0.008330563433362871
            if (x < 10)
            {
                return MMath.GammaLn(x) - (x - 0.5) * Math.Log(x) + x - LnSqrt2PI;
            }
            else
            {
                // the series is:  sum_{i=1}^inf B_{2i} / (2i*(2i-1)*x^(2i-1))
                double sum = 0;
                double term = 1.0 / x;
                double delta = term * term;
                for (int i = 0; i < c_gammaln_series.Length; i++)
                {
                    sum += c_gammaln_series[i] * term;
                    term *= delta;
                }
                return sum;
            }
        }

        private static double[] c_gammaln_series =
        {
            1.0 / (6 * 2), -1.0 / (30 * 4 * 3), 1.0 / (42 * 6 * 5), -1.0 / (30 * 8 * 7),
            5.0/(66*10*9), -691.0/(2730*12*11), 7.0/(6*14*13)
        };

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
                int j = (int)(-x + 0.5);
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
            double oldSum = 0;
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
        /// <param name="x">Any real number</param>
        /// <param name="delta">A finite real number with absolute value less than 9.9, or less than 70% of the absolute value of x.</param>
        /// <param name="startingIndex">The first moment to use in the power series.  Used to skip leading terms.  For example, 2 will skip NormalCdfMomentRatio(1, x).</param>
        /// <returns></returns>
        public static double NormalCdfRatioDiff(double x, double delta, int startingIndex = 1)
        {
            if (startingIndex < 1) throw new ArgumentOutOfRangeException(nameof(startingIndex), "startingIndex < 1");
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
                if (double.IsNaN(sum)) throw new Exception();
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
            const double large = -8;
            if (x > 4)
            {
                // log(NormalCdf(x)) = log(1-NormalCdf(-x))
                double y = NormalCdf(-x);
                // log(1-y)
                return -y * (1 + y * (0.5 + y * (1.0 / 3 + y * (0.25))));
            }
            else if (x >= large)
            {
                return Math.Log(NormalCdf(x));
            }
            else
            {
                // x < large
                double z = 1 / (x * x);
                // asymptotic series for log(normcdf)
                // Maple command: subs(x=-x,asympt(log(normcdf(-x)),x));
                double s = z * (c_normcdfln_series[0] + z *
                              (c_normcdfln_series[1] + z *
                               (c_normcdfln_series[2] + z *
                                (c_normcdfln_series[3] + z *
                                 (c_normcdfln_series[4] + z *
                                  (c_normcdfln_series[5] + z *
                                   c_normcdfln_series[6]))))));
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

        // [0] contains moments for x=-2
        // [1] contains moments for x=-3, etc.
        private static double[][] NormalCdfMomentRatioTable = new double[7][];

        /// <summary>
        /// Computes int_0^infinity t^n N(t;x,1) dt / (n! N(x;0,1))
        /// </summary>
        /// <param name="n">The exponent</param>
        /// <param name="x">Any real number</param>
        /// <returns></returns>
        public static double NormalCdfMomentRatio(int n, double x)
        {
            const int tableSize = 200;
            const int maxTerms = 60;
            if (x >= -0.5)
                return NormalCdfMomentRatioRecurrence(n, x);
            else if (n <= tableSize - maxTerms && x > -8)
            {
                int index = (int)(-x - 1.5); // index ranges from 0 to 6
                double x0 = -index - 2;
                if (NormalCdfMomentRatioTable[index] == null)
                {
                    double[] derivs = new double[tableSize];
                    // this must not try to use the lookup table, since we are building it
                    var iter = NormalCdfMomentRatioSequence(0, x0, true);
                    for (int i = 0; i < derivs.Length; i++)
                    {
                        iter.MoveNext();
                        derivs[i] = iter.Current;
                    }
                    NormalCdfMomentRatioTable[index] = derivs;
                }
                return NormalCdfMomentRatioTaylor(n, x - x0, NormalCdfMomentRatioTable[index]);
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
            for (int i = 0; i < 1000; i++)
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
            throw new Exception($"Not converging for n={n},x={x:r}");
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
            if (Double.IsNegativeInfinity(x) || Double.IsNegativeInfinity(y))
            {
                return 0.0;
            }
            else if (Double.IsPositiveInfinity(x))
            {
                return NormalCdf(y);
            }
            else if (Double.IsPositiveInfinity(y))
            {
                return NormalCdf(x);
            }
            else if (r == 0)
            {
                return NormalCdf(x) * NormalCdf(y);
            }
            else if (r == 1)
            {
                return NormalCdf(Math.Min(x, y));
            }
            else if (r == -1)
            {
                return Math.Max(0.0, NormalCdf(x) + NormalCdf(y) - 1);
            }
            // at this point, both x and y are finite.
            // swap to ensure |x| > |y|
            if (Math.Abs(y) > Math.Abs(x))
            {
                double t = x;
                x = y;
                y = t;
            }
            double offset = 0;
            double scale = 1;
            // ensure x <= 0
            if (x > 0)
            {
                // phi(x,y,r) = phi(inf,y,r) - phi(-x,y,-r)
                offset = MMath.NormalCdf(y);
                scale = -1;
                x = -x;
                r = -r;
            }
            // ensure r <= 0
            if (r > 0)
            {
                // phi(x,y,r) = phi(x,inf,r) - phi(x,-y,-r)
                offset += scale * MMath.NormalCdf(x);
                scale *= -1;
                y = -y;
                r = -r;
            }
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r            
            double ymrx = (y - r * x) / Math.Sqrt(omr2);
            double exponent;
            double result = NormalCdf_Helper(x, y, r, omr2, ymrx, out exponent);
            return offset + scale * result * Math.Exp(exponent);
        }

        // factor out the dominant terms and then call confrac
        private static double NormalCdf_Helper(double x, double y, double r, double omr2, double ymrx, out double exponent)
        {
            exponent = Gaussian.GetLogProb(x, 0, 1);
            double scale;
            if (ymrx < 0)
            {
                // since phi(ymrx) will be small, we factor N(ymrx;0,1) out of the confrac
                exponent += Gaussian.GetLogProb(ymrx, 0, 1);
                scale = 1;
            }
            else
            {
                // leave N(ymrx;0,1) in the confrac
                scale = Math.Exp(Gaussian.GetLogProb(ymrx, 0, 1));
            }
            // For debugging, see SpecialFunctionsTests.NormalCdf2Test3
            if (x < -1.7)
            {
                exponent -= Math.Log(-x);
                return NormalCdfRatioConFrac(x, y, r, scale);
            }
            else if (omr2 > 0.75 || (omr2 > 1 - 0.56 * 0.56 && x > -0.5))
            {
                return NormalCdfRatioTaylor(x, y, r) * scale;
                //return NormalCdfRatioConFrac3(x, y, r, scale);
            }
            else
            {
                return NormalCdfRatioConFrac2b(x, y, r, scale);
            }
        }

        // Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1)
        // requires -1 < x < 0, abs(r) <= 0.6, and x-r*y <= 0 (or equivalently y < -x).
        // Uses Taylor series at r=0.
        private static double NormalCdfRatioTaylor(double x, double y, double r)
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
            for (int n = 2; n <= 100; n++)
            {
                if (n == 100) throw new Exception($"not converging for x={x}, y={y}, r={r}");
                //Console.WriteLine($"n = {n - 1} sum = {sum:r}");
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
                if ((sum > double.MaxValue) || double.IsNaN(sum))
                    throw new Exception($"not converging for x={x}, y={y}, r={r}");
                if (AreEqual(sum, sumOld)) break;
                sumOld = sum;
            }
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r            
            return sum / Math.Sqrt(omr2);
        }

        // Returns NormalCdf divided by N(x;0,1)/(-x) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale
        // requires x < -1, r <= 0, and x-r*y <= 0 (or equivalently y < -x).
        private static double NormalCdfRatioConFrac(double x, double y, double r, double scale)
        {
            if (x >= -1)
                throw new ArgumentException("x >= -1");
            if (r > 0)
                throw new ArgumentException("r > 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            if (scale == 0)
                return scale;
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r
            double sqrtomr2 = Math.Sqrt(omr2);
            double diff = (x - r * y) / sqrtomr2;
            var RdiffIter = NormalCdfMomentRatioSequence(0, diff);
            RdiffIter.MoveNext();
            double Rdiff = RdiffIter.Current;
            double numer = NormalCdfRatioConFracNumer(x, y, r, scale, sqrtomr2, diff, Rdiff);
            double numerPrev = 0;
            double denom = 1;
            double denomPrev = denom;
            double resultPrev = 0;
            double result = 0;
            double invX2 = 1 / (x * x);
            double a = invX2;
            double b = scale * r;
            double bIncr = sqrtomr2 / x;
            for (int i = 1; i < 1000; i++)
            {
                b *= bIncr * i;
                RdiffIter.MoveNext();
                //double c = b * MMath.NormalCdfMomentRatio(i, diff);
                double c = b * RdiffIter.Current;
                double numerNew = numer + a * numerPrev + c;
                double denomNew = denom + a * denomPrev;
                a += invX2;
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                result = numer / denom;
                //Console.WriteLine("iter {0}: {1} {2}", i, result.ToString("r"), c.ToString("g4"));
                if ((result > double.MaxValue) || double.IsNaN(result) || result < 0)
                    throw new Exception($"not converging for x={x:r}, y={y:r}, r={r:r}");
                if (AreEqual(result, resultPrev))
                    return result;
                resultPrev = result;
            }
            throw new Exception($"not converging for x={x:r}, y={y:r}, r={r:r}");
        }

        // Helper function for NormalCdfRatioConFrac
        private static double NormalCdfRatioConFracNumer(double x, double y, double r, double scale, double sqrtomr2, double diff, double Rdiff)
        {
            double delta = (1 + r) * (y - x) / sqrtomr2;
            double numer;
            if (Math.Abs(delta) > 0.5)
            {
                numer = scale * r * Rdiff;
                double diffy = (y - r * x) / sqrtomr2;
                if (scale == 1)
                    numer += MMath.NormalCdfRatio(diffy);
                else // this assumes scale = N((y-rx)/sqrt(1-r^2);0,1)
                    numer += MMath.NormalCdf(diffy);
            }
            else
            {
                numer = scale * (NormalCdfRatioDiff(diff, delta) + (1 + r) * Rdiff);
            }
            return numer;
        }

        // Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale
        // requires x <= 0, r <= 0, and x-r*y <= 0 (or equivalently y < -x).
        // This version works best for -1 < x <= 0.
        private static double NormalCdfRatioConFrac2(double x, double y, double r, double scale)
        {
            if (x > 0)
                throw new ArgumentException("x >= 0");
            if (r > 0)
                throw new ArgumentException("r > 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            if (scale == 0)
                return scale;
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r
            double sqrtomr2 = Math.Sqrt(omr2);
            double diff = (x - r * y) / sqrtomr2;
            var RdiffIter = NormalCdfMomentRatioSequence(0, diff);
            RdiffIter.MoveNext();
            double Rdiff = RdiffIter.Current;
            double numer = NormalCdfRatioConFracNumer(x, y, r, scale, sqrtomr2, diff, Rdiff);
            double numerPrev = 0;
            double denom = -x;
            double denomPrev = -1;
            double resultPrev = 0;
            double cEven = scale * r;
            double cOdd = cEven * sqrtomr2;
            for (int i = 1; i < 1000; i++)
            {
                double numerNew, denomNew;
                //double c = MMath.NormalCdfMomentRatio(i, diff);
                RdiffIter.MoveNext();
                double c = RdiffIter.Current;
                if (i % 2 == 1)
                {
                    if (i > 1)
                        cOdd *= (i - 1) * omr2;
                    c *= cOdd;
                    numerNew = x * numer + numerPrev + c;
                    denomNew = x * denom + denomPrev;
                }
                else
                {
                    cEven *= i * omr2;
                    c *= cEven;
                    numerNew = (x * numer + i * numerPrev + c) / (i + 1);
                    denomNew = (x * denom + i * denomPrev) / (i + 1);
                }
                numerPrev = numer;
                numer = numerNew;
                denomPrev = denom;
                denom = denomNew;
                if (i % 2 == 1)
                {
                    double result = numer / denom;
                    //Console.WriteLine($"iter {i}: result={result:r} c={c:g4} numer={numer:r} denom={denom:r} numerPrev={numerPrev:r}");
                    if ((result > double.MaxValue) || double.IsNaN(result) || result < 0)
                        throw new Exception($"NormalCdfRatioConFrac2 not converging for x={x} y={y} r={r} scale={scale}");
                    if (AreEqual(result, resultPrev))
                        return result;
                    resultPrev = result;
                }
            }
            throw new Exception($"NormalCdfRatioConFrac2 not converging for x={x} y={y} r={r} scale={scale}");
        }

        // Returns NormalCdf divided by N(x;0,1) N((y-rx)/sqrt(1-r^2);0,1), multiplied by scale
        // requires x <= 0, r <= 0, and x-r*y <= 0 (or equivalently y < -x).
        // This version works best for -1 < x <= 0.
        private static double NormalCdfRatioConFrac2b(double x, double y, double r, double scale)
        {
            if (x > 0)
                throw new ArgumentException("x >= 0");
            if (r > 0)
                throw new ArgumentException("r > 0");
            if (x - r * y > 0)
                throw new ArgumentException("x - r*y > 0");
            if (scale == 0)
                return scale;
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r
            double sqrtomr2 = Math.Sqrt(omr2);
            double diff = (x - r * y) / sqrtomr2;
            var RdiffIter = NormalCdfMomentRatioSequence(0, diff);
            RdiffIter.MoveNext();
            double Rdiff = RdiffIter.Current;
            double numer = NormalCdfRatioConFracNumer(x, y, r, scale, sqrtomr2, diff, Rdiff);
            double numerPrev = 0;
            double denom = -x;
            double denomPrev = -1;
            double resultPrev = 0;
            double rDy = scale * r;
            for (int i = 1; i < 1000; i++)
            {
                RdiffIter.MoveNext();
                double c = RdiffIter.Current;
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
                    //Console.WriteLine($"iter {i}: result={result:r} c={c:g4} numer={numer:r} denom={denom:r} numerPrev={numerPrev:r}");
                    if ((result > double.MaxValue) || double.IsNaN(result) || result < 0)
                        throw new Exception($"NormalCdfRatioConFrac2b not converging for x={x} y={y} r={r} scale={scale}");
                    if (AreEqual(result, resultPrev))
                        return result;
                    resultPrev = result;
                }
            }
            throw new Exception($"NormalCdfRatioConFrac2b not converging for x={x} y={y} r={r} scale={scale}");
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
                return x * (1.0 / 2 + x * (1.0 / 8 + x * (1.0 / 16 + x * 5.0 / 128)));
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

        /// <summary>
        /// Computes the natural logarithm of the cumulative bivariate normal distribution.
        /// </summary>
        /// <param name="x">First upper limit.</param>
        /// <param name="y">Second upper limit.</param>
        /// <param name="r">Correlation coefficient.</param>
        /// <returns><c>ln(phi(x,y,r))</c></returns>
        public static double NormalCdfLn(double x, double y, double r)
        {
            if (Double.IsNegativeInfinity(x) || Double.IsNegativeInfinity(y))
            {
                return Double.NegativeInfinity;
            }
            else if (Double.IsPositiveInfinity(x))
            {
                return NormalCdfLn(y);
            }
            else if (Double.IsPositiveInfinity(y))
            {
                return NormalCdfLn(x);
            }
            else if (r == 0)
            {
                return NormalCdfLn(x) + NormalCdfLn(y);
            }
            else if (r == 1)
            {
                return NormalCdfLn(Math.Min(x, y));
            }
            else if (r == -1)
            {
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
            // ensure x <= 0
            if (x > 0)
            {
                // phi(x,y,r) = phi(inf,y,r) - phi(-x,y,-r)
                logOffset = MMath.NormalCdfLn(y);
                scale = -1;
                x = -x;
                r = -r;
            }
            // ensure r <= 0
            if (r > 0)
            {
                // phi(x,y,r) = phi(x,inf,r) - phi(x,-y,-r)
                double logOffset2 = MMath.NormalCdfLn(x);
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
            double omr2 = (1 - r) * (1 + r); // more accurate than 1-r*r            
            double ymrx = (y - r * x) / Math.Sqrt(omr2);
            double exponent;
            double result = NormalCdf_Helper(x, y, r, omr2, ymrx, out exponent);
            double logResult = exponent + Math.Log(result);
            if (scale == -1)
                return MMath.LogDifferenceOfExp(logOffset, logResult);
            else
                return MMath.LogSumExp(logOffset, logResult);
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
            if (x > -1e-3 && x < 2e-3)
            {
                // use the Taylor series for log(1+x) around x=0
                // Maple command: series(log(1+x),x);
                return x * (1 - x * (0.5 - x * (1.0 / 3 - x * (0.25 - x * (1.0 / 5)))));
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
            if (x > 50)
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
        /// particularly when x &lt; -7.5 or x > -1e-5.</remarks>
        public static double Log1MinusExp(double x)
        {
            if (x > 0)
                throw new ArgumentException("x (" + x + ") > 0");
            if (x < -7.5)
            {
                double y = Math.Exp(x);
                return -y * (1 + y * (0.5 + y * (1.0 / 3 + y * (0.25))));
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
            if (Math.Abs(x) < 2e-3)
            {
                return x * (1 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24))));
            }
            else
            {
                return Math.Exp(x) - 1.0;
            }
        }

        /// <summary>
        /// Computes ((exp(x)-1)/x - 1)/x - 0.5
        /// </summary>
        /// <param name="x">Any real number from 0 to Inf, or NaN.</param>
        /// <returns>((exp(x)-1)/x - 1)/x - 0.5</returns>
        public static double ExpMinus1RatioMinus1RatioMinusHalf(double x)
        {
            if (x < 0) throw new ArgumentOutOfRangeException(nameof(x), "x < 0");
            if (Math.Abs(x) < 6e-1)
            {
                return x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 +
                    x * (1.0 / 5040 + x * (1.0 / 40320 + x * (1.0 / 362880 + x * (1.0 / 3628800 +
                    x * (1.0 / 39916800 + x * (1.0 / 479001600 + x * (1.0 / 6227020800 + x * (1.0 / 87178291200))))))))))));
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
                return Math.Log(x) + x * (0.5 + x * (1.0 / 24 + x * (-1.0 / 2880)));
            }
            else if (x > 50)
            {
                return x;
            }
            else
            {
                return Math.Log(Math.Exp(x) - 1.0);
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
            double max = Max(list);
            IEnumerator<double> iter = list.GetEnumerator();
            if (!iter.MoveNext() || Double.IsNegativeInfinity(max))
                return Double.NegativeInfinity; // log(0)
            if (Double.IsPositiveInfinity(max))
                return Double.PositiveInfinity;
            // at this point, max is finite
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
            if (!(list is ISparseEnumerable<double>))
                return LogSumExp(list);
            double max = list.EnumerableReduce(double.NegativeInfinity,
                                               (res, i) => Math.Max(res, i), (res, i, count) => Math.Max(res, i));
            var iter = (list as ISparseEnumerable<double>).GetSparseEnumerator();
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
        private static double integrate(Converter<double, double> f, Vector nodes, Vector weights)
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

        // Math.Exp(-745.14) == 0
        private const double log0 = -745.14;
        // 1-Math.Exp(-38) == 1
        private const double logEpsilon = -38;

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
            if (-mean + halfVariance < logEpsilon)
                return 1.0;

            // use the upper bound 0.5 exp(-0.5 m^2/v) to prune cases that must be zero or one
            double q = -0.5 * mean * mean / variance - MMath.Ln2;
            if (mean <= 0 && mean + variance >= 0 && q < log0)
                return 0.0;
            if (mean >= 0 && variance - mean >= 0 && q < logEpsilon)
                return 1.0;
            // sigma(|m|,v) <= 0.5 + |m| sigma'(0,v)
            // sigma'(0,v) <= N(0;0,v+8/pi)
            double d0Upper = MMath.InvSqrt2PI / Math.Sqrt(variance + 8 / Math.PI);
            if (mean * mean / (variance + 8 / Math.PI) < 2e-20 * Math.PI)
            {
                double deriv = LogisticGaussianDerivative(mean, variance);
                return 0.5 + mean * deriv;
            }

            // Handle tail cases using the following exact formulas:
            // sigma(m,v) = 1 - exp(-m+v/2) + exp(-2m+2v) - exp(-3m+9v/2) sigma(m-3v,v)
            if (-mean + variance < logEpsilon)
                return 1.0 - Math.Exp(halfVariance - mean);
            if (-3 * mean + 9 * halfVariance < logEpsilon)
                return 1.0 - Math.Exp(halfVariance - mean) + Math.Exp(2 * (variance - mean));
            // sigma(m,v) = exp(m+v/2) - exp(2m+2v) + exp(3m + 9v/2) (1 - sigma(m+3v,v))
            if (mean + 1.5 * variance < logEpsilon)
                return Math.Exp(mean + halfVariance);
            if (2 * mean + 4 * variance < logEpsilon)
                return Math.Exp(mean + halfVariance) * (1 - Math.Exp(mean + 1.5 * variance));

            if (variance > LogisticGaussianVarianceThreshold)
            {
                Converter<double, double> f = delegate (double x)
                {
                    return Math.Exp(MMath.LogisticLn(x) + Gaussian.GetLogProb(x, mean, variance));
                };
                double upperBound = mean + Math.Sqrt(variance);
                upperBound = Math.Max(upperBound, 10);
                return Quadrature.AdaptiveClenshawCurtis(f, upperBound, 32, 1e-10);
            }
            else
            {
                Vector nodes = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                Vector weights = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                double m_p, v_p;
                BigvProposal(mean, variance, out m_p, out v_p);
                Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                Converter<double, double> weightedIntegrand =
                    delegate (double z)
                    {
                        return Math.Exp(MMath.LogisticLn(z) + Gaussian.GetLogProb(z, mean, variance) - Gaussian.GetLogProb(z, m_p, v_p));
                    };
                return integrate(weightedIntegrand, nodes, weights);
            }
            /*
else {
double s = Math.Sqrt(v);
// Region for which posterior is not moved much by the logistic function
// so can use Gauss-Hermite quadrature on the input Gaussian
if ((m/s > 5.0) || (m < -15.0-25.0/4.0*s)) {
    Vector nodes = Vector.Zero(QuadratureNodeCount);
    Vector weights = Vector.Zero(QuadratureNodeCount);
    Quadrature.GaussianNodesAndWeights(m, v, nodes, weights);
    return integrate(MMath.Logistic, nodes, weights);
}
    // Region where variance is not big enough for the big v approximation to
    // apply, but need to find the mode before doing quadrature
else if (m < 20.0 - 60.0/11.0 * s) {
    // Newton-Raphson to quickly find the mode of the Converter<double,double> N(x;m,v)logistic(x)
    double x = Math.Max(m, 0.0); // initialise x
    double tolerance = s/10; // the tolerance can be bigger at higher input variance
    int max_iterations=10;
    bool success=false;
    double h=1; // the "Hessian", here just the second derivative
    for (int i=0; i<max_iterations; i++) {
            double l = MMath.Logistic(x);
            double xOld = x;
            double g = -(x-m)/v + 1-l; // gradient
            h = -1.0/v -  l * (1.0-l); // second derivative
            x = x - g / h; // Newton step
            if (Math.Abs(x - xOld) < tolerance) {
                    success=true;
                    break;
            }
    }
    if (!success)
            Console.WriteLine("Warning: mini-newton did not converge");
    double m_p=x;
    double v_p=-1.0/h; // ala Laplace approximation
    // Gauss-Hermite quadrature
    Vector nodes = Vector.Zero(QuadratureNodeCount);
    Vector weights = Vector.Zero(QuadratureNodeCount);
    Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
    Converter<double,double> weightedIntegrand = delegate(double z)
    {
            return Math.Exp(MMath.LogisticLn(z) + Gaussian.GetLogProb(z, m, v)-Gaussian.GetLogProb(z, m_p, v_p));
    };
    return integrate(weightedIntegrand, nodes, weights);
} else {
    // Big v approximation
    return MMath.NormalCdf(m / Math.Sqrt(v + Math.PI * Math.PI / 3.0));
}

}*/
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
            if (-mean + 1.5 * variance < logEpsilon)
                return Math.Exp(halfVariance - mean);
            if (-2 * mean + 4 * variance < logEpsilon)
                return Math.Exp(halfVariance - mean) - 2 * Math.Exp(2 * (variance - mean));

            if (variance > LogisticGaussianVarianceThreshold)
            {
                Converter<double, double> f = delegate (double x)
                {
                    return Math.Exp(MMath.LogisticLn(x) + MMath.LogisticLn(-x) + Gaussian.GetLogProb(x, mean, variance));
                };
                return Quadrature.AdaptiveClenshawCurtis(f, 10, 32, 1e-10);
            }
            else
            {
                Vector nodes = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                Vector weights = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                double m_p, v_p;
                BigvProposal(mean, variance, out m_p, out v_p);
                Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                Converter<double, double> weightedIntegrand =
                    delegate (double z)
                    {
                        return Math.Exp(MMath.LogisticLn(z) + MMath.LogisticLn(-z) + Gaussian.GetLogProb(z, mean, variance) - Gaussian.GetLogProb(z, m_p, v_p));
                    };
                return integrate(weightedIntegrand, nodes, weights);
            }

            /*
            double s = Math.Sqrt(v);
            // Region where bigv approximation does badly
            if ((Math.Abs(m)+5.0)/s>4.0) {
                    // Newton-Raphson to quickly find the mode of the Converter<double,double> xN(x;m,v)logistic(x)
                    double x = 0.0; // initialise x
                    double tolerance = s / 10.0;// the tolerance can be bigger at higher input variance
                    int max_iterations=10;
                    bool success=false;
                    double h=1;
                    for (int i=0; i<max_iterations; i++) {
                            double l = MMath.Logistic(x);
                            double xOld = x;
                            double g = -(x - m) / v + 1.0 - 2.0 * l;// gradient
                            h = -1.0 / v - 2.0 * l * (1.0 - l);// second derivative
                            x = x - g / h;// Newton step
                            if (Math.Abs(x - xOld) < tolerance) {
                                    success=true;
                                    break;
                            }
                    }
                    if (!success)
                            Console.WriteLine("Warning: mini-newton did not converge");
                    double m_p=x;
                    // Here we make the "proposal" distribution a bit wider since
                    // there are problems with skewed integrands not being covered
                    // well otherwise
                    double v_p = -4.0 / h; // ala Laplace approximation
                    // Gauss-Hermite quadrature
                    Vector nodes = Vector.Zero(QuadratureNodeCount);
                    Vector weights = Vector.Zero(QuadratureNodeCount);
                    Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                    Converter<double,double> weightedIntegrand = delegate(double z)
                    {
                            return LogisticPrime(z)*Math.Exp(Gaussian.GetLogProb(z, m, v)-Gaussian.GetLogProb(z, m_p, v_p));
                    };
                    return integrate(weightedIntegrand, nodes, weights);
            } else {
                    // Big variance approximation
                    return Math.Exp(Gaussian.GetLogProb(m, 0, v + Math.PI * Math.PI / 3.0));
            } */
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
            if (-mean + 1.5 * variance < logEpsilon)
                return -Math.Exp(halfVariance - mean);
            if (-2 * mean + 4 * variance < logEpsilon)
                return -Math.Exp(halfVariance - mean) + 4 * Math.Exp(2 * (variance - mean));
            // sigma''(m,v) = exp(m+v/2) -4 exp(2m+2v) +9 exp(3m + 9v/2) (1 - sigma(m+3v,v)) - 6 exp(3m+9v/2) sigma'(m+3v,v) - exp(3m + 9v/2) sigma''(m+3v,v)
            if (mean + 1.5 * variance < logEpsilon)
                return Math.Exp(mean + halfVariance);
            if (2 * mean + 4 * variance < logEpsilon)
                return Math.Exp(mean + halfVariance) * (1 - 4 * Math.Exp(mean + 1.5 * variance));

            if (variance > LogisticGaussianVarianceThreshold)
            {
                Converter<double, double> f = delegate (double x)
                    {
                        double logSigma = MMath.LogisticLn(x);
                        double log1MinusSigma = MMath.LogisticLn(-x);
                        double OneMinus2Sigma = -Math.Tanh(x / 2);
                        return OneMinus2Sigma * Math.Exp(logSigma + log1MinusSigma + Gaussian.GetLogProb(x, mean, variance));
                    };
                return Quadrature.AdaptiveClenshawCurtis(f, 10, 32, 1e-10);
            }
            else
            {
                Vector nodes = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                Vector weights = Vector.Zero(LogisticGaussianQuadratureNodeCount);
                double m_p, v_p;
                BigvProposal(mean, variance, out m_p, out v_p);
                Quadrature.GaussianNodesAndWeights(m_p, v_p, nodes, weights);
                Converter<double, double> weightedIntegrand = delegate (double z)
                    {
                        double logSigma = MMath.LogisticLn(z);
                        double log1MinusSigma = MMath.LogisticLn(-z);
                        double OneMinus2Sigma = -Math.Tanh(z / 2);
                        return OneMinus2Sigma * Math.Exp(logSigma + log1MinusSigma + Gaussian.GetLogProb(z, mean, variance) - Gaussian.GetLogProb(z, m_p, v_p));
                    };
                return integrate(weightedIntegrand, nodes, weights);
            }

            /*
            double s = Math.Sqrt(v);
            // Region where bigv approximation does badly
            if (Math.Abs(m)/v > 1.0) {
                    // Gauss-Hermite quadrature
                    Vector nodes = Vector.Zero(QuadratureNodeCount);
                    Vector weights = Vector.Zero(QuadratureNodeCount);
                    Quadrature.GaussianNodesAndWeights(m, v, nodes, weights);
                    Converter<double,double> logisticPrimePrime = delegate(double z)
                    {
                            double l = MMath.Logistic(z);
                            return l * (1.0 - l) * (1.0 - 2.0 * l);
                    };
                    return integrate(logisticPrimePrime, nodes, weights);
            } else {
                    // Big variance approximation
                    double v2 = v + Math.PI * Math.PI / 3.0;
                    return -m / v2 * Math.Exp(Gaussian.GetLogProb(m, 0, v2));
            } */
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
            Converter<double, double> f = delegate (double x)
            {
                double logSigma = MMath.LogisticLn(x);
                double extra = 0;
                double s = 1;
                if (k > 0) extra += MMath.LogisticLn(-x);
                if (k > 1) s = -Math.Tanh(x / 2);
                return s * Math.Exp(logSigma + extra + x * a + Gaussian.GetLogProb(x, 0, variance));
            };
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
                return Double.NaN;
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
                return Double.NaN;
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
        public static int IndexOfMinimum<T>(IList<T> list)
            where T : IComparable<T>
        {
            if (list.Count == 0)
                return -1;
            T min = list[0];
            int pos = 0;
            for (int i = 1; i < list.Count; i++)
            {
                if (min.CompareTo(list[i]) > 0)
                {
                    min = list[i];
                    pos = i;
                }
            }
            return pos;
        }

        /// <summary>
        /// Returns the index of the maximum element, or -1 if empty.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <returns></returns>
        public static int IndexOfMaximum<T>(IList<T> list)
            where T : IComparable<T>
        {
            if (list.Count == 0)
                return -1;
            T max = list[0];
            int pos = 0;
            for (int i = 1; i < list.Count; i++)
            {
                if (max.CompareTo(list[i]) < 0)
                {
                    max = list[i];
                    pos = i;
                }
            }
            return pos;
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
                return Double.NaN;
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
        /// <returns><c>abs(x - y)/(abs(x) + rel)</c>. 
        /// Matching infinities give zero.
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
        /// Returns the distance between two numbers.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns><c>abs(x - y)</c>. 
        /// Matching infinities give zero.
        /// </returns>
        public static double AbsDiff(double x, double y)
        {
            if (x == y)
                return 0; // catches infinities
            return Math.Abs(x - y);
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
        /// Returns the largest value such that value/denominator &lt;= ratio.
        /// </summary>
        /// <param name="denominator"></param>
        /// <param name="ratio"></param>
        /// <returns></returns>
        internal static double LargestDoubleProduct(double denominator, double ratio)
        {
            if (denominator < 0) return LargestDoubleProduct(-denominator, -ratio);
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
            if (double.IsPositiveInfinity(denominator))
            {
                if (double.IsNaN(ratio)) return denominator;
                else return double.MaxValue;
            }
            if (double.IsPositiveInfinity(ratio)) return ratio;
            // denominator > 0
            // avoid infinite bounds
            double lowerBound = (double)Math.Max(double.MinValue, denominator * PreviousDouble(ratio));
            if (lowerBound == 0 && ratio < 0) lowerBound = -denominator; // must have ratio > -1
            if (double.IsPositiveInfinity(lowerBound)) lowerBound = denominator; // must have ratio > 1
            // subnormal numbers are linearly spaced, which can lead to lowerBound being too large.  Set lowerBound to zero to avoid this.
            const double maxSubnormal = 2.3e-308;
            if (lowerBound > 0 && lowerBound < maxSubnormal) lowerBound = 0;
            double upperBound = (double)Math.Min(double.MaxValue, denominator * NextDouble(ratio));
            if (upperBound == 0 && ratio > 0) upperBound = denominator; // must have ratio < 1
            if (double.IsNegativeInfinity(upperBound)) return upperBound; // must have ratio < -1 and denominator > 1
            if (upperBound < 0 && upperBound > -maxSubnormal) upperBound = 0;
            if (double.IsNegativeInfinity(ratio))
            {
                if (AreEqual(upperBound / denominator, ratio)) return upperBound;
                else return PreviousDouble(upperBound);
            }
            while (true)
            {
                double value = (double)Average(lowerBound, upperBound);
                if (value < lowerBound || value > upperBound) throw new Exception($"value={value:r}, lowerBound={lowerBound:r}, upperBound={upperBound:r}, denominator={denominator:r}, ratio={ratio:r}");
                if ((double)(value / denominator) <= ratio)
                {
                    double value2 = NextDouble(value);
                    if (value2 == value || (double)(value2 / denominator) > ratio)
                    {
                        return value;
                    }
                    else
                    {
                        // value is too low
                        lowerBound = value2;
                        if (lowerBound > upperBound || double.IsNaN(lowerBound)) throw new Exception($"value={value:r}, lowerBound={lowerBound:r}, upperBound={upperBound:r}, denominator={denominator:r}, ratio={ratio:r}");
                    }
                }
                else
                {
                    // value is too high
                    upperBound = PreviousDouble(value);
                    if (lowerBound > upperBound || double.IsNaN(upperBound)) throw new Exception($"value={value:r}, lowerBound={lowerBound:r}, upperBound={upperBound:r}, denominator={denominator:r}, ratio={ratio:r}");
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
            double lowerBound = PreviousDouble(b + sum);
            double upperBound;
            if (Math.Abs(sum) > Math.Abs(b))
            {
                upperBound = b + NextDouble(sum);
            }
            else
            {
                upperBound = NextDouble(b) + sum;
            }
            long iterCount = 0;
            while (true)
            {
                iterCount++;
                double value = Average(lowerBound, upperBound);
                //double value = RepresentationMidpoint(lowerBound, upperBound);
                if (value < lowerBound || value > upperBound) throw new Exception($"value={value:r}, lowerBound={lowerBound:r}, upperBound={upperBound:r}, b={b:r}, sum={sum:r}");
                if (value - b <= sum)
                {
                    double value2 = NextDouble(value);
                    if (value2 == value || value2 - b > sum)
                    {
                        //if (iterCount > 100)
                        //    throw new Exception();
                        return value;
                    }
                    else
                    {
                        // value is too low
                        lowerBound = value2;
                        if (lowerBound > upperBound || double.IsNaN(lowerBound)) throw new Exception($"value={value:r}, lowerBound={lowerBound:r}, upperBound={upperBound:r}, b={b:r}, sum={sum:r}");
                    }
                }
                else
                {
                    // value is too high
                    upperBound = PreviousDouble(value);
                    if (lowerBound > upperBound || double.IsNaN(upperBound)) throw new Exception($"value={value:r}, lowerBound={lowerBound:r}, upperBound={upperBound:r}, b={b:r}, sum={sum:r}");
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
            double midpoint = (a + b) / 2;
            if (double.IsInfinity(midpoint)) midpoint = 0.5 * a + 0.5 * b;
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


        #region Enumerations and constants

        /// <summary>
        /// The Euler-Mascheroni Constant.
        /// </summary>
        public const double EulerGamma = 0.57721566490153286060651209;

        /// <summary>
        /// Digamma(1)
        /// </summary>
        public const double Digamma1 = -EulerGamma;

        private const double TOLERANCE = 1.0e-7;

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
        /// Coefficients of the asymptotic expansion of NormalCdfLn.
        /// </summary>
        private static readonly double[] c_normcdfln_series =
            {
                -1, 5.0/2, -37.0/3, 353.0/4, -4081.0/5, 55205.0/6, -854197.0/7
            };

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

        #endregion
    }
}