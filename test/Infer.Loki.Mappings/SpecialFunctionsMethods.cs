using Loki.Mapping;
using Loki.Mapping.Methods;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.MPFR;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Mappings
{
    public static class SpecialFunctionsMethods
    {
        //[DllImport(MPFRLibrary.FileName, CallingConvention = CallingConvention.Cdecl)]
        //public static extern int mpfr_gamma_inc([In, Out] mpfr_struct rop, [In, Out] mpfr_struct op, [In, Out] mpfr_struct op2, int rnd);

        public static readonly BigFloat Sqrt2;
        public static readonly BigFloat Sqrt2PI;
        public static readonly BigFloat LnSqrt2PI;

        static SpecialFunctionsMethods()
        {
            Sqrt2 = BigFloatFactory.Create(2);
            Sqrt2.Sqrt();

            Sqrt2PI = BigFloatFactory.Empty();
            Sqrt2PI.ConstPi();
            Sqrt2PI.Mul2(1);
            Sqrt2PI.Sqrt();

            LnSqrt2PI = BigFloatFactory.Create(Sqrt2PI);
            LnSqrt2PI.Log();
        }

        public static BigFloat Gamma(BigFloat x)
        {
            if (x.IsNegative())
            {
                if (x.IsInteger())
                    return BigFloatFactory.NaN;
                //-Math.PI / (x * Math.Sin(Math.PI * x) * Gamma(-x));
                var result = BigFloatFactory.Empty();
                result.ConstPi();
                result.Neg();
                using (var acc = BigFloatFactory.Empty())
                using (var acc2 = BigFloatFactory.Create(x))
                {
                    acc.ConstPi();
                    acc.Mul(x);
                    acc.Sin();
                    acc.Mul(x);
                    acc2.Neg();
                    using (var gammamx = Gamma(acc2))
                        acc.Mul(gammamx);
                    result.Div(acc);
                }
                return result;
            }
            else
            {
                var result = BigFloatFactory.Empty();
                BigFloat.Gamma(result, x);
                return result;
            }
        }

        public static BigFloat GammaLn(BigFloat x)
        {
            if (DoubleMethods.IsPositiveInfinity(x) || x.IsZero())
                return BigFloatFactory.PositiveInfinity;
            if (x.IsNegative())
                return BigFloatFactory.NaN;
            var result = BigFloatFactory.Empty();
            BigFloat.Lngamma(result, x);
            return result;
        }

        // RisingFactorialLnOverN

        public static BigFloat Digamma(BigFloat x)
        {
            var result = BigFloatFactory.Empty();
            BigFloat.Digamma(result, x);
            return result;
        }


        private static BigFloat EvaluateSeries(BigFloat x, BigFloat[] coefficients)
        {
            BigFloat result = BigFloatFactory.Create(coefficients[coefficients.Length - 1]);
            for (int i = coefficients.Length - 2; i >= 0; --i)
            {
                result.Mul(x);
                result.Add(coefficients[i]);
            }
            return result;
        }

        private static int FindConvergencePoint(BigFloat x, BigFloat[] coefficients)
        {
            using (var sum = BigFloatFactory.Create(0.0))
            using (var oldSum = BigFloatFactory.Create(0.0))
            using (var term = BigFloatFactory.Create(1.0))
            using (var tmp = BigFloatFactory.Empty())
            {
                for (int i = 0; i < coefficients.Length; ++i)
                {
                    BigFloat.Mul(tmp, term, coefficients[i]);
                    sum.Add(tmp);
                    if (!coefficients[i].IsZero() && sum.IsEqual(oldSum))
                        return i;
                    term.Mul(x);
                    BigFloat.Set(oldSum, sum);
                }
            }
            return -1;
        }

        public static void FindThresholds()
        {
            using (var tmp1 = BigFloatFactory.Empty())
            using (var invX2 = BigFloatFactory.Empty())
            {
                for (int i = -30; i >= -100; --i)
                {
                    tmp1.Set(i);
                    tmp1.Sqr();
                    //BigFloat.Sqr(tmp1, c_trigamma_large);
                    BigFloat.Div(invX2, 1.0, tmp1);
                    int conv = FindConvergencePoint(invX2, normalCdfLnAsymptotic);
                    Console.WriteLine($"Convergence point at {i}: {conv}");
                    if (conv > 0)
                    {
                        using (var result = EvaluateSeries(invX2, normalCdfLnAsymptotic.Take(conv).ToArray()))
                        using (var x = BigFloatFactory.Create(i))
                        using (var f = NormalCdf(x))
                        {
                            result.Sub(LnSqrt2PI);
                            tmp1.Div2(1);
                            result.Sub(tmp1);
                            BigFloat.Neg(tmp1, x);
                            tmp1.Log();
                            result.Sub(tmp1);
                            Console.WriteLine($"Polynomial: {result}");
                            f.Log();
                            Console.WriteLine($"Algebraic:  {f}");
                            f.Sub(result);
                            f.Abs();
                            Console.WriteLine($"Diff:       {f}");
                        }
                    }
                }
            }
        }

        // The threshold for applying de Moivre's expansion for the trigamma function.
        private static readonly BigFloat c_trigamma_large = BigFloatFactory.Create("18");
        private static readonly BigFloat c_trigamma_small = BigFloatFactory.Create("1e-4");
        // Truncated series 5: Trigamma at 1
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] trigammaAt1Coefficients = new BigFloat[]
        {
            BigFloatFactory.Create("1.6449340668482264364724151666460251892189499012068"),
            BigFloatFactory.Create("-2.4041138063191885707994763230228999815299725846810"),
            BigFloatFactory.Create("3.2469697011334145745480110896235037083242528557562"),
            BigFloatFactory.Create("-4.1477110205734797053254619458281366722283236780077"),
            BigFloatFactory.Create("5.0867153099222456985725896489546026395090874501643"),
            BigFloatFactory.Create("-6.0500956642915369610387852990987805575991813633914"),
            BigFloatFactory.Create("7.0285414933856103756507966695605672568127255345489"),
            BigFloatFactory.Create("-8.0160671426086577153428221538592964838848468111591"),
            BigFloatFactory.Create("9.0089511761503627680343136301028711530541757840803"),
            BigFloatFactory.Create("-10.004941886041194645587022825264699364686064357582")
        };
        // Truncated series 6: Trigamma asymptotic
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] trigammaAsymptotic = new BigFloat[]
        {
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("0.16666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-0.033333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("0.023809523809523809523809523809523809523809523809524"),
            BigFloatFactory.Create("-0.033333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("0.075757575757575757575757575757575757575757575757576"),
            BigFloatFactory.Create("-0.25311355311355311355311355311355311355311355311355"),
            BigFloatFactory.Create("1.1666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-7.0921568627450980392156862745098039215686274509804"),
            BigFloatFactory.Create("54.971177944862155388471177944862155388471177944862"),
            BigFloatFactory.Create("-529.12424242424242424242424242424242424242424242424"),
            BigFloatFactory.Create("6192.1231884057971014492753623188405797101449275362"),
            BigFloatFactory.Create("-86580.253113553113553113553113553113553113553113553"),
            BigFloatFactory.Create("1425517.1666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-27298231.067816091954022988505747126436781609195402"),
            BigFloatFactory.Create("601580873.90064236838430386817483591677140064236838"),
            BigFloatFactory.Create("-15116315767.092156862745098039215686274509803921569"),
            BigFloatFactory.Create("429614643061.16666666666666666666666666666666666667"),
            BigFloatFactory.Create("-13711655205088.332772159087948561632772159087948562"),
            BigFloatFactory.Create("488332318973593.16666666666666666666666666666666667"),
            BigFloatFactory.Create("-19296579341940068.148632668144863266814486326681449"),
            BigFloatFactory.Create("841693047573682615.00055370985603543743078626799557"),
            BigFloatFactory.Create("-40338071854059455413.076811594202898550724637681159"),
            BigFloatFactory.Create("2115074863808199160560.1453900709219858156028368794"),
            BigFloatFactory.Create("-120866265222965259346027.31193708252531781943546649"),
            BigFloatFactory.Create("7500866746076964366855720.0757575757575757575757576"),
            BigFloatFactory.Create("-503877810148106891413789303.05220125786163522012579"),
            BigFloatFactory.Create("36528776484818123335110430842.971177944862155388471"),
            BigFloatFactory.Create("-2849876930245088222626914643291.0678160919540229885"),
            BigFloatFactory.Create("238654274996836276446459819192192.14971751412429379"),
            BigFloatFactory.Create("-21399949257225333665810744765191097.392674151161724"),
            BigFloatFactory.Create("2050097572347809756992173309567231025.1666666666667")
        };

        /// <summary>
        /// Evaluates Trigamma(x), the derivative of Digamma(x).
        /// </summary>
        /// <param name="x">Any real value.</param>
        /// <returns>Trigamma(x).</returns>
        public static BigFloat Trigamma(BigFloat x)
        {
            BigFloat result;

            /* Negative values */
            /* Use the derivative of the digamma reflection formula:
             * -trigamma(-x) = trigamma(x+1) - (pi*csc(pi*x))^2
             */
            if (x.IsNegative())
            {
                if (x.IsInteger() || x.IsInf())
                {
                    return BigFloatFactory.NaN;
                }
                result = BigFloatFactory.Empty();
                result.ConstPi();
                using (var tmp1 = BigFloatFactory.Create(x))
                {
                    tmp1.Mul(result);
                    tmp1.Sin();
                    result.Div(tmp1);
                    result.Sqr();
                    tmp1.Set(1.0);
                    tmp1.Sub(x);
                    using (var tmp2 = Trigamma(tmp1))
                        result.Sub(tmp2);
                }
                // result = Math.PI / Math.Sin(Math.PI * x);
                //return (-Trigamma(1 - x) + result * result);
                return result;
            }


            /* Shift the argument and use Taylor series at 1 if argument <= small */
            if (x.IsLesserOrEqual(c_trigamma_small))
            {
                using (var tmp1 = BigFloatFactory.Create(x))
                using (var series = EvaluateSeries(x, trigammaAt1Coefficients))
                {
                    result = BigFloatFactory.Create(1.0);
                    tmp1.Sqr();
                    result.Div(tmp1);
                    result.Add(series);
                }
                return result;
            }

            using (var localX = BigFloatFactory.Create(x))
            using (var tmp1 = BigFloatFactory.Empty())
            using (var invX2 = BigFloatFactory.Empty())
            {
                result = BigFloatFactory.Create(0.0);

                /* Reduce to trigamma(x+n) where ( X + N ) >= L */
                while (localX.IsLesser(c_trigamma_large))
                {
                    BigFloat.Sqr(tmp1, localX);
                    BigFloat.Div(invX2, 1.0, tmp1);
                    result.Add(invX2);
                    localX.Add(1.0);
                }
                /* X >= L.    Apply asymptotic formula. */
                // This expansion can be computed in Maple via asympt(Psi(1,x),x)
                BigFloat.Sqr(tmp1, localX);
                BigFloat.Div(invX2, 1.0, tmp1);
                BigFloat.Div2(tmp1, invX2, 1);
                result.Add(tmp1);
                using (var sum = EvaluateSeries(invX2, trigammaAsymptotic))
                    BigFloat.Add(tmp1, sum, 1.0);
                tmp1.Div(localX);
                result.Add(tmp1);
            }
            return result;
        }

        // The threshold for applying de Moivre's expansion for the tetragamma function.
        private static readonly BigFloat c_tetragamma_large = BigFloatFactory.Create("18");
        private static readonly BigFloat c_tetragamma_small = BigFloatFactory.Create("1e-4");
        // Truncated series 7: Tetragamma at 1
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] tetragammaAt1Coefficients = new BigFloat[]
        {
            BigFloatFactory.Create("-2.4041138063191885707994763230228999815299725846810"),
            BigFloatFactory.Create("6.4939394022668291490960221792470074166485057115124"),
            BigFloatFactory.Create("-12.443133061720439115976385837484410016684971034023"),
            BigFloatFactory.Create("20.346861239688982794290358595818410558036349800657"),
            BigFloatFactory.Create("-30.250478321457684805193926495493902787995906816957"),
            BigFloatFactory.Create("42.171248960313662253904780017363403540876353207294"),
            BigFloatFactory.Create("-56.112469998260604007399755077015075387193927678114"),
            BigFloatFactory.Create("72.071609409202902144274509040822969224433406272642"),
            BigFloatFactory.Create("-90.044476974370751810283205427382294282174579218239"),
            BigFloatFactory.Create("110.02706952086388531285017978525136380564576973038"),
            BigFloatFactory.Create("-132.01619816188036056737124242147917623428431397821")
        };
        // Truncated series 8: Tetragamma asymptotic
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] tetragammaAsymptotic = new BigFloat[]
        {
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("-1.0000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.50000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("0.16666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-0.16666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("0.30000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.83333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("3.2904761904761904761904761904761904761904761904762"),
            BigFloatFactory.Create("-17.500000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("120.56666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-1044.4523809523809523809523809523809523809523809524"),
            BigFloatFactory.Create("11111.609090909090909090909090909090909090909090909"),
            BigFloatFactory.Create("-142418.83333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("2164506.3278388278388278388278388278388278388278388"),
            BigFloatFactory.Create("-38488963.500000000000000000000000000000000000000000"),
            BigFloatFactory.Create("791648700.96666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-18649007090.919913419913419913419913419913419913420"),
            BigFloatFactory.Create("498838420314.04117647058823529411764705882352941176"),
            BigFloatFactory.Create("-15036512507140.833333333333333333333333333333333333"),
            BigFloatFactory.Create("507331242588268.31256988625409678041256988625409678"),
            BigFloatFactory.Create("-19044960439970133.500000000000000000000000000000000"),
            BigFloatFactory.Create("791159753019542794.09393939393939393939393939393939"),
            BigFloatFactory.Create("-36192801045668352445.023809523809523809523809523810"),
            BigFloatFactory.Create("1815213233432675493588.4565217391304347826086956522"),
            BigFloatFactory.Create("-99408518598985360546326.833333333333333333333333333"),
            BigFloatFactory.Create("5922446995925297707955338.2849170437405731523378582"),
            BigFloatFactory.Create("-382544204049925182709641723.86363636363636363636364"),
            BigFloatFactory.Create("26705523937849665244930833061.766666666666666666667"),
            BigFloatFactory.Create("-2009082706664996783431073696363.4147869674185463659"),
            BigFloatFactory.Create("162442985023970028689734134667590.86551724137931034"),
            BigFloatFactory.Create("-14080602224813340310341129332339336.833333333333333"),
            BigFloatFactory.Create("1305396904690745353614455430676656940.9531232208652")
        };
        /// <summary>
        ///  Evaluates Tetragamma, the forth derivative of logGamma(x)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static BigFloat Tetragamma(BigFloat x)
        {
            if (x.IsNegative())
                return BigFloatFactory.NaN;

            BigFloat result;
            /* Use Taylor series if argument <= small */
            if (x.IsLesser(c_tetragamma_small))
            {
                using (var tmp1 = BigFloatFactory.Create(x))
                using (var series = EvaluateSeries(x, tetragammaAt1Coefficients))
                {
                    result = BigFloatFactory.Create(-2.0);
                    tmp1.Sqr();
                    tmp1.Mul(x);
                    result.Div(tmp1);
                    result.Add(series);
                }
                return result;
            }


            using (var localX = BigFloatFactory.Create(x))
            using (var tmp1 = BigFloatFactory.Empty())
            using (var invX2 = BigFloatFactory.Empty())
            {
                result = BigFloatFactory.Create(0.0);
                /* Reduce to Tetragamma(x+n) where ( X + N ) >= L */
                while (localX.IsLesser(c_tetragamma_large))
                {
                    BigFloat.Sqr(tmp1, localX);
                    tmp1.Mul(localX);
                    BigFloat.Div(invX2, 2.0, tmp1);
                    result.Sub(invX2);
                    localX.Add(1.0);
                }
                /* X >= L.    Apply asymptotic formula. */
                // This expansion can be computed in Maple via asympt(Psi(2,x),x) or found in
                // Milton Abramowitz and Irene A. Stegun, Handbook of Mathematical Functions, Section 6.4

                BigFloat.Sqr(tmp1, localX);
                BigFloat.Div(invX2, 1.0, tmp1);
                BigFloat.Div(tmp1, invX2, localX);
                result.Sub(tmp1);
                using (var sum = EvaluateSeries(invX2, tetragammaAsymptotic))
                    result.Add(sum);
            }
            return result;
        }

        //public static BigFloat GammaUpper(BigFloat a, BigFloat x, bool regularized = true)
        //{
        //    var result = BigFloatFactory.Empty();
        //    var an = new Math.Mpfr.Native.mpfr_t();
        //    var xn = new Math.Mpfr.Native.mpfr_t();
        //    var resultn = new Math.Mpfr.Native.mpfr_t();
        //    Math.Mpfr.Native.mpfr_lib.mpfr_inits2((uint)BigFloatFactory.FloatingPointPrecision, an, xn, resultn);
        //    try
        //    {
        //        CrossLibSet(an, a);
        //        CrossLibSet(xn, x);
        //        Math.Mpfr.Native.mpfr_lib.mpfr_gamma_inc(resultn, xn, an, Math.Mpfr.Native.mpfr_rnd_t.MPFR_RNDN);
        //        var vbuf = Math.Gmp.Native.gmp_lib.allocate(1 + BigFloatFactory.FloatingPointPrecision / 4 + (BigFloatFactory.FloatingPointPrecision % 4 != 0 ? 1 : 0));
        //        try
        //        {
        //            var buf = new Math.Gmp.Native.char_ptr(vbuf.ToIntPtr());
        //            var expBuf = new Math.Mpfr.Native.mpfr_exp_t(0);
        //            Math.Mpfr.Native.mpfr_lib.mpfr_get_str(buf, ref expBuf, 16, 0, resultn, Math.Mpfr.Native.mpfr_rnd_t.MPFR_RNDN);
        //            var resultString = buf.ToString();
        //            int signPosShift = resultString[0] == '+' || resultString[0] == '-' ? 1 : 0;
        //            if (resultString[signPosShift] != '@')
        //            {
        //                resultString = $"{resultString.Substring(0, signPosShift + 1)}.{resultString.Substring(signPosShift + 1)}@{(int)expBuf}";
        //            }
        //            result.Set(resultString, 16);
        //        }
        //        finally
        //        {
        //            Math.Gmp.Native.gmp_lib.free(vbuf);
        //        }
        //    }
        //    finally
        //    {
        //        Math.Mpfr.Native.mpfr_lib.mpfr_clears(an, xn, resultn);
        //    }
        //    //Math.Mpfr.Native.mpfr_lib.mpfr_gamma_inc()
        //    //mpfr_gamma_inc(result.Value, x.Value, a.Value, BigFloat.GetRounding(null));
        //    if (regularized)
        //    {
        //        using (var gammaa = Gamma(x))
        //            result.Div(gammaa);
        //    }
        //    return result;

        //    void CrossLibSet(Math.Mpfr.Native.mpfr_t rop, BigFloat op)
        //    {
        //        var astr = new Math.Gmp.Native.char_ptr(op.ToString("b16d0", System.Globalization.CultureInfo.InvariantCulture));
        //        try
        //        {
        //            Math.Mpfr.Native.mpfr_lib.mpfr_set_str(rop, astr, 16, Math.Mpfr.Native.mpfr_rnd_t.MPFR_RNDN);
        //        }
        //        finally
        //        {
        //            Math.Gmp.Native.gmp_lib.free(astr);
        //        }
        //    }
        //}

        //public static BigFloat GammaLower(BigFloat a, BigFloat x)
        //{
        //    var result = BigFloatFactory.Create(1.0);
        //    using (var upper = GammaUpper(a, x))
        //        result.Sub(upper);
        //    return result;
        //}

        public static BigFloat ReciprocalFactorialMinus1(BigFloat x)
        {
            var result = BigFloatFactory.Create(1.0);
            using (var tmp = BigFloatFactory.Create(1.0))
            {
                tmp.Add(x);
                result.Div(tmp);
                tmp.Set(1.0);
                result.Sub(tmp);
            }
            return result;
        }

        public static BigFloat Log1Plus(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.Log1p();
            return result;
        }

        private static readonly BigFloat log1MinusMethodThreshold = BigFloatFactory.Create("-3.5");
        public static BigFloat Log1MinusExp(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            if (result.IsLesser(log1MinusMethodThreshold))
            {
                result.Exp();
                result.Neg();
                result.Log1p();
            }
            else
            {
                result.Expm1();
                result.Neg();
                result.Log();
            }
            return result;
        }

        public static BigFloat ExpMinus1(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.Expm1();
            return result;
        }

        public static BigFloat ExpMinus1RatioMinus1RatioMinusHalf(BigFloat x)
        {
            if (x.IsInf() && x.IsPositive())
            {
                return BigFloatFactory.Create(x);
            }
            else
            {
                var result = BigFloatFactory.Create(x);
                result.Expm1();
                result.Div(x);
                result.Sub(1);
                result.Div(x);
                result.Sub(0.5);
                return result;
            }
        }

        public static BigFloat NormalCdf(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.Neg();
            result.Div(Sqrt2);
            result.Erfc();
            result.Div2(1);
            return result;
        }


        private static readonly BigFloat normalCdfLnThreshold1 = BigFloatFactory.Create("4");
        private static readonly BigFloat normalCdfLnThreshold2 = BigFloatFactory.Create("-50");
        // Truncated series 16: normcdfln asymptotic
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] normalCdfLnAsymptotic = new BigFloat[]
        {
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("-1.0000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("2.5000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-12.333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("88.250000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-816.20000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("9200.8333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("-122028.14285714285714285714285714285714285714285714"),
            BigFloatFactory.Create("1859504.1250000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-32002080.111111111111111111111111111111111111111111"),
            BigFloatFactory.Create("613891392.50000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-12989299596.090909090909090909090909090909090909091"),
            BigFloatFactory.Create("300556863709.41666666666666666666666666666666666667"),
            BigFloatFactory.Create("-7550646317520.0769230769230769230769230769230769231"),
            BigFloatFactory.Create("204687481350960.35714285714285714285714285714285714"),
            BigFloatFactory.Create("-5955892982437394.4666666666666666666666666666666667"),
            BigFloatFactory.Create("185158929516994912.06250000000000000000000000000000"),
            BigFloatFactory.Create("-6125200081143892800.0588235294117647058823529411765"),
            BigFloatFactory.Create("214837724609518838186.94444444444444444444444444444")
        };

        public static BigFloat NormalCdfLn(BigFloat x)
        {
            if (x.IsGreater(normalCdfLnThreshold1))
            {
                using (var z = BigFloatFactory.Create(x))
                {
                    z.Neg();
                    var result = NormalCdf(z);
                    // maybe replace with series
                    result.Neg();
                    result.Log1p();
                    return result;
                }
            }
            else if (x.IsGreaterOrEqual(normalCdfLnThreshold2))
            {
                var result = NormalCdf(x);
                result.Log();
                return result;
            }
            else
            {
                using (var z = BigFloatFactory.Create(1))
                using (var xx = BigFloatFactory.Create(x))
                {
                    // x < large
                    xx.Sqr();
                    z.Div(xx);
                    var result = EvaluateSeries(z, normalCdfLnAsymptotic);
                    result.Sub(LnSqrt2PI);
                    xx.Div2(1);
                    result.Sub(xx);
                    BigFloat.Neg(xx, x);
                    xx.Log();
                    result.Sub(xx);
                    return result;
                }
            }
        }

        public static BigFloat Erfc(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.Erfc();
            return result;
        }


        private static readonly BigFloat normalCdfRatioThreshold = BigFloatFactory.Create("-25");
        private static readonly BigFloat[] c_normcdf_table = new BigFloat[]
        {
            BigFloatFactory.Create("1.2533141373155002512078826424055226265034933703050"),
            BigFloatFactory.Create("0.65567954241879847154387123073081128339928233287046"),
            BigFloatFactory.Create("0.42136922928805447322493433354238497871759897424685"),
            BigFloatFactory.Create("0.30459029871010329573361254651572220194332086785732"),
            BigFloatFactory.Create("0.23665238291356067062398593643584354772280595719945"),
            BigFloatFactory.Create("0.19280810471531576487746572791751625149030275528504"),
            BigFloatFactory.Create("0.16237766089686746181568210281899300101285429948633"),
            BigFloatFactory.Create("0.14010418345305024159953452179636098824179029701342"),
            BigFloatFactory.Create("0.12313196325793229628218074351719907628055418406307"),
            BigFloatFactory.Create("0.10978728257830829123063783305609744024963272432146"),
            BigFloatFactory.Create("0.099028596471731921395337188595310578345522351804893"),
            BigFloatFactory.Create("0.090175675501064682279780356185873435376447672142213"),
            BigFloatFactory.Create("0.082766286501369177252265050041905901860174647936978"),
            BigFloatFactory.Create("0.076475761016248502993495194221514297839495267927670"),
            BigFloatFactory.Create("0.071069580538852107090596841578057621441250984768689"),
            BigFloatFactory.Create("0.066374235823250173591323880429856263938548730308721"),
            BigFloatFactory.Create("0.062258665995026195776685945013859824034284602258714"),
            BigFloatFactory.Create("0.058622064980015943875453424458613461051690664823031"),
            BigFloatFactory.Create("0.055385651470100734467246738955231191017362356305507"),
            BigFloatFactory.Create("0.052486980219676364236804604231478581750669351644688"),
            BigFloatFactory.Create("0.049875925981836783658240561473547677421984200939185"),
            BigFloatFactory.Create("0.047511794276278113026555416422393514742536362543443"),
            BigFloatFactory.Create("0.045361207289993100875851687673213152771439836060349"),
            BigFloatFactory.Create("0.043396533095512704064299446607278252292001462420503"),
            BigFloatFactory.Create("0.041594702232575505419657616279389108630393708182485"),
            BigFloatFactory.Create("0.039936304769535592528778701538208913507390239835584"),
            BigFloatFactory.Create("0.038404893342102127679827361148942634536063966746361"),
            BigFloatFactory.Create("0.036986439428385819875150275693382405885634992108882"),
            BigFloatFactory.Create("0.035668904990075763502562777837627528340021232014842"),
            BigFloatFactory.Create("0.034441901929091245587507704942245227872609616802230")
        };
        /// <summary>
        /// Computes <c>NormalCdf(x)/N(x;0,1)</c> to high accuracy.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns></returns>
        public static BigFloat NormalCdfRatio(BigFloat x)
        {
            if (x.IsPositive() && !x.IsZero())
            {
                // Sqrt2PI * Math.Exp(0.5 * x * x) - NormalCdfRatio(-x);
                var result = BigFloatFactory.Create(x);
                result.Sqr();
                result.Mul2(-1);
                result.Exp();
                result.Mul(Sqrt2PI);
                using (var mx = BigFloatFactory.Create(x))
                {
                    mx.Neg();
                    using (var ncdfrmx = NormalCdfRatio(mx))
                        result.Sub(ncdfrmx);
                }
                return result;
            }
            else if (x.IsGreater(normalCdfRatioThreshold))
            {
                // Taylor expansion
                using (var tmp1 = BigFloatFactory.Create(x))
                using (var tmp2 = BigFloatFactory.Create(x))
                {
                    tmp1.Neg();
                    tmp1.Add(0.5);
                    tmp1.Floor();
                    var j = tmp1.ToInt64();
                    if (j >= c_normcdf_table.Length)
                        j = c_normcdf_table.Length - 1;
                    var result = BigFloatFactory.Create(c_normcdf_table[j]);
                    tmp2.Add(j);
                    tmp1.Set(-j);
                    using (var y = NormalCdfRatioDiff_Simple(tmp1, tmp2, result))
                        result.Add(y);
                    return result;
                }
            }
            else
            {
                // Continued fraction approach
                if (x.IsNan())
                    return x;
                using (var invX = BigFloatFactory.Create(1))
                using (var invX2 = BigFloatFactory.Empty())
                using (var numer = BigFloatFactory.Empty())
                using (var numerPrev = BigFloatFactory.Create(0))
                using (var denom = BigFloatFactory.Create(1))
                using (var denomPrev = BigFloatFactory.Create(1))
                using (var a = BigFloatFactory.Empty())
                using (var numerNew = BigFloatFactory.Empty())
                using (var denomNew = BigFloatFactory.Empty())
                {
                    invX.Div(x);
                    BigFloat.Sqr(invX2, invX);
                    BigFloat.Neg(numer, invX);
                    BigFloat.Set(a, invX2);
                    for (int i = 1; i < 22; i++)
                    {
                        BigFloat.Fma(numerNew, a, numerPrev, numer);
                        BigFloat.Fma(denomNew, a, denomPrev, denom);
                        a.Add(invX2);
                        BigFloat.Swap(numerPrev, numer);
                        BigFloat.Swap(numer, numerNew);
                        BigFloat.Swap(denomPrev, denom);
                        BigFloat.Swap(denom, denomNew);
                    }
                    var result = BigFloatFactory.Create(numer);
                    result.Div(denom);
                    return result;
                }
            }
        }

        /// <summary>
        /// BigFloat adaptation of <see cref="Microsoft.ML.Probabilistic.Math.MMath.NormalCdfRatioDiff_Simple"/>
        /// </summary>
        private static BigFloat NormalCdfRatioDiff_Simple(BigFloat x, BigFloat delta, BigFloat y)
        {
            using (var Reven = BigFloatFactory.Create(y))
            using (var Rodd = BigFloatFactory.Create(Reven))
            using (var delta2 = BigFloatFactory.Create(delta))
            using (var oldSum = BigFloatFactory.Empty())
            using (var tmp = BigFloatFactory.Empty())
            using (var pwr = BigFloatFactory.Create(1))
            {
                Rodd.Mul(x);
                Rodd.Add(1);
                delta2.Sqr();
                var sum = BigFloatFactory.Create(delta);
                sum.Mul(Rodd);
                for (long i = 2; i < 10000; i += 2)
                {
                    BigFloat.Fma(tmp, x, Rodd, Reven);
                    BigFloat.Div(Reven, tmp, i);
                    BigFloat.Fma(tmp, x, Reven, Rodd);
                    BigFloat.Div(Rodd, tmp, i + 1);
                    pwr.Mul(delta2);
                    BigFloat.Swap(oldSum, sum);
                    BigFloat.Fma(tmp, delta, Rodd, Reven);
                    tmp.Mul(pwr);
                    BigFloat.Add(sum, oldSum, tmp);
                    if (BigFloat.Equal(sum, oldSum))
                        break;
                }
                return sum;
            }
        }

        #region BinaryRepresentation

        public static BigFloat NextBigFloat(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.NextAbove();
            return result;
        }

        public static BigFloat PreviousBigFloat(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.NextBelow();
            return result;
        }

        #endregion
    }
}
