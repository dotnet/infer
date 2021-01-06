// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Math;
using System.Diagnostics;

namespace Microsoft.ML.Probabilistic.Tests
{
    // See https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/preprocessor-directives/preprocessor-if
#if NETCOREAPP3_1
    public class FloatTests
    {
        internal void QuadraticFormulaTest()
        {
            float a = 1e-8f;
            float b = 1f;
            float c = -1f;
            float rootF = QuadraticFormula(a, b, c);
            float rootF2 = QuadraticFormulaMuller(a, b, c);
            double root = QuadraticFormula((double)a, b, c);
            Console.WriteLine($"{rootF} {rootF2} {(float)root}");
        }

        internal static float QuadraticFormula(float a, float b, float c)
        {
            float ac4 = a * c * 4;
            float bSqr = b * b;
            float discrim = bSqr - ac4;
            float sqrtDiscrim = System.MathF.Sqrt(discrim);
            float numer = sqrtDiscrim - b;
            return numer / (2 * a);
        }

        internal static float QuadraticFormulaMuller(float a, float b, float c)
        {
            float ac4 = a * c * 4;
            float bSqr = b * b;
            float discrim = bSqr - ac4;
            float sqrtDiscrim = System.MathF.Sqrt(discrim);
            float denom = -sqrtDiscrim - b;
            return 2 * c / denom;
        }

        internal static double QuadraticFormula(double a, double b, double c)
        {
            double ac4 = a * c * 4;
            double bSqr = b * b;
            double discrim = bSqr - ac4;
            double sqrtDiscrim = System.Math.Sqrt(discrim);
            double numer = sqrtDiscrim - b;
            return numer / (2 * a);
        }

        /// <summary>
        /// Tests whether a function decreases for positive x.
        /// In this case, XPlus1TimesXMinus1 sometimes decreases for positive x.
        /// </summary>
        internal void XPlus1TimesXMinus1Test()
        {
            float x = 0;
            while (x < 1)
            {
                float next = NextFloat(x);
                float fx = XPlus1TimesXMinus1(x);
                float fnext = XPlus1TimesXMinus1(next);
                if (fnext < fx)
                {
                    throw new Exception($"fnext < fx");
                }
                x = next;
            }
        }

        private static float XPlus1TimesXMinus1(float x)
        {
            return (x + 1) * (x - 1);
        }

        internal void X2MinusXSquaredTest()
        {
            float x = 0;
            while (x < 1)
            {
                float next = NextFloat(x);
                float fx = X2MinusXSquared(x);
                float fnext = X2MinusXSquared(next);
                if (fnext < fx)
                {
                    throw new Exception($"fnext < fx");
                }
                x = next;
            }
        }

        private static float X2MinusXSquared(float x)
        {
            return (float)(2 * x - (float)(x * x));
            // Below does not work
            //return x * (float)(2 - x);
        }

        internal void XPlusInvXTest()
        {
            float x = float.Epsilon;
            //x = 0.707169652f;
            //x = 0.707169831f;
            //x = 1e-39f;
            //x = NextFloat(x);
            while (x < 1)
            {
                float next = NextFloat(x);
                float fx = XPlusInvX(x);
                float fnext = XPlusInvX(next);
                float invX = (float)(1f / x);
                float invNext = (float)(1f / next);
                //if(!float.IsInfinity(invX) && invX == invNext)
                //    Trace.WriteLine($"x = {x:r} 1/x = {invX:r} 1/next = {invNext:r} fx = {fx:r} fnext = {fnext:r}");
                if (fnext > fx)
                {
                    Trace.WriteLine($"x = {x:r} 1/x = {invX:r} 1/next = {invNext:r} fx = {fx:r} fnext = {fnext:r}");
                    //Trace.WriteLine("fnext > fx");
                    throw new Exception($"fnext > fx");
                }
                x = next;
            }
        }

        private static float XPlusInvX(float x)
        {
            float invX = (float)(1f / x);
            // Below is monotonic for x >= 1
            //return x + invX;
            // Below is monotonic for 0 < x < 1
            float invInvX = (float)(1f / invX);
            return invInvX + invX;
        }

        /// <summary>
        /// Tests whether two expressions are equal for all floating-point values.
        /// </summary>
        internal void EqualTest()
        {
            float x = float.Epsilon;
            while(x < float.PositiveInfinity)
            {
                if ((float)(x + x + x) + x != 4 * x) throw new Exception();
                //if (x + x + x + x + x + x != 6 * x) throw new Exception();
                //if ((x + x) + (x + x) + (x + x) != 6 * x) throw new Exception();
                x = NextFloat(x);
            }
        }

        internal void SincTest()
        {
            float x = float.Epsilon;
            x = 0.0007f;
            x = 0.04f;
            double ulp1 = 5.9604644775390625e-8;
            float threshold = (float)System.Math.Sqrt(System.Math.Sqrt(60 * ulp1));
            Console.WriteLine($"0.5 ulp threshold is {threshold:r}");
            int equalCount = 0;
            int lowCount = 0;
            int highCount = 0;
            while (x < threshold)
            {
                float taylor = 1.0f - (x * x) * (1.0f / 6.0f);
                //float taylor = 1 - (x * x) / 6;
                double xD = x;
                double sincD = System.Math.Sin(xD) / xD;
                float sinc = (float)sincD;
                // First error occurs at x=0.0007324219 0.9999999 0.99999994
                if (taylor == sinc) equalCount++;
                else if (NextFloat(taylor) == sinc) lowCount++;
                else if (PreviousFloat(taylor) != sinc) highCount++;
                else throw new Exception($"x={ToStringExact(x)} {ToStringExact(taylor)} {ToStringExact((float)sincD)}");
                x = NextFloat(x);
            }
            // at the 0.5 ulp threshold, the truncated Taylor series is too low 50% of the time.
            // above this threshold, the truncated Taylor series is too low >50% of the time, even though the error is within 1 ulp.
            int total = equalCount + lowCount + highCount;
            Console.WriteLine($"Low = {100.0 * lowCount / total:f1}% equal = {100.0 * equalCount / total:f1}% high = {100.0 * highCount / total:f1}%");
        }

        public static float NextFloat(float value)
        {
            return MathF.BitIncrement(value);
            //if (value < 0) return -PreviousFloat(-value);
            //value = System.Math.Abs(value); // needed to handle -0
            //if (float.IsNaN(value)) return value;
            //if (float.IsPositiveInfinity(value)) return value;
            //int bits = BitConverter.SingleToInt32Bits(value);
            //return BitConverter.Int32BitsToSingle(bits + 1);
        }

        public static float PreviousFloat(float value)
        {
            return MathF.BitDecrement(value);
            //if (value <= 0) return -NextFloat(-value);
            //if (float.IsNaN(value)) return value;
            //if (float.IsNegativeInfinity(value)) return value;
            //int bits = BitConverter.SingleToInt32Bits(value);
            //return BitConverter.Int32BitsToSingle(bits - 1);
        }

        [Fact]
        public void ToStringExactFloatTest()
        {
            Assert.Equal("0", ToStringExact(0f));
            Assert.Equal("NaN", ToStringExact(float.NaN));
            Assert.Equal(float.MaxValue, float.Parse(ToStringExact(float.MaxValue)));
            Assert.Equal(float.MinValue, float.Parse(ToStringExact(float.MinValue)));
            Assert.Equal("10.5", ToStringExact(10.5f));
            Assert.Equal(10.05f, float.Parse(ToStringExact(10.05f)));
            Assert.Equal(2e-38f, float.Parse(ToStringExact(2e-38f)));
            Assert.Equal(1e-44f, float.Parse(ToStringExact(1e-44f)));
            Assert.Equal(float.Epsilon, float.Parse(ToStringExact(float.Epsilon)));
        }

        /// <summary>
        /// Returns a decimal string that exactly equals a single-precision number, unlike float.ToString which always returns a rounded result.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static string ToStringExact(float x)
        {
            if (float.IsNaN(x) || float.IsInfinity(x) || x == 0) return x.ToString(System.Globalization.CultureInfo.InvariantCulture);
            // BitConverter.SingleToInt32Bits does not exist in .NET Standard 2.0
            int bits = BitConverter.SingleToInt32Bits(x);
            uint fraction = Convert.ToUInt32(bits & 0x007fffff); // 23 bits
            short exponent = Convert.ToInt16((bits & 0x7f800000) >> 23);
            if (exponent == 0)
            {
                // subnormal number
                exponent = -126 - 23;
            }
            else
            {
                // normal number
                fraction += 0x00800000;
                exponent = Convert.ToInt16(exponent - 127 - 23);
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
                // However, float.Parse does not correctly parse such strings.
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
    }
#endif
        }
