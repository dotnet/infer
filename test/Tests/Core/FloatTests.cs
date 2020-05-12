﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{
    // See https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/preprocessor-directives/preprocessor-if
#if NETCOREAPP3_1
    public class FloatTests
    {
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
            if (value < 0) return -PreviousFloat(-value);
            value = System.Math.Abs(value); // needed to handle -0
            if (float.IsNaN(value)) return value;
            if (float.IsPositiveInfinity(value)) return value;
            int bits = BitConverter.SingleToInt32Bits(value);
            return BitConverter.Int32BitsToSingle(bits + 1);
        }

        public static float PreviousFloat(float value)
        {
            if (value <= 0) return -NextFloat(-value);
            if (float.IsNaN(value)) return value;
            if (float.IsNegativeInfinity(value)) return value;
            int bits = BitConverter.SingleToInt32Bits(value);
            return BitConverter.Int32BitsToSingle(bits - 1);
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
