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
        public static readonly BigFloat Sqrt2;
        public static readonly BigFloat Sqrt2PI;
        public static readonly BigFloat InvSqrt2PI;
        public static readonly BigFloat LnSqrt2PI;
        public static readonly BigFloat Ln2;
        public static readonly BigFloat DefaultBetaEpsilon = BigFloatFactory.Create("1e-38");
        public static readonly BigFloat Ulp1;
        public static readonly BigFloat LogisticGaussianVarianceThreshold = BigFloatFactory.NegativeInfinity;
        public static readonly BigFloat LogisticGaussianSeriesApproximmationThreshold = BigFloatFactory.Create("1e-18");
        public const int AdaptiveQuadratureMaxNodes = 1000000;
        public static readonly BigFloat LogisticGaussianQuadratureRelativeTolerance = BigFloatFactory.Create("1e-38");

        static SpecialFunctionsMethods()
        {
            Sqrt2 = BigFloatFactory.Create(2);
            Sqrt2.Sqrt();

            Sqrt2PI = BigFloatFactory.Empty();
            Sqrt2PI.ConstPi();
            Sqrt2PI.Mul2(1);
            Sqrt2PI.Sqrt();

            InvSqrt2PI = BigFloatFactory.Create(1);
            InvSqrt2PI.Div(Sqrt2PI);

            LnSqrt2PI = BigFloatFactory.Create(Sqrt2PI);
            LnSqrt2PI.Log();

            Ln2 = BigFloatFactory.Create(2);
            Ln2.Log();

            Ulp1 = BigFloatFactory.Create(1);
            Ulp1.NextAbove();
            Ulp1.Sub(1);
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
            using (var tmp1 = BigFloatFactory.Create("0.6"))
            using (var invX = BigFloatFactory.Create(1))
            using (var invX2 = BigFloatFactory.Empty())
            {
                //invX.Div(tmp1);
                //int conv = FindConvergencePoint(expMinus1RatioMinus1RatioMinusHalfThresholdPos, expMinus1RatioMinus1RatioMinusHalfAt0);
                //Console.WriteLine($"Convergence point at {tmp1}: {conv}");
                //var result = EvaluateSeries(tmp1, gammaMinusReciprocalAt0.Take(conv).ToArray());
                //var f = BigFloatFactory.Create(tmp1);
                //f.Gamma();
                //f.Sub(invX);
                //Console.WriteLine($"Polynomial: {result}");
                //Console.WriteLine($"Algebraic:  {f}");
                //f.Sub(result);
                //f.Abs();
                //Console.WriteLine($"Diff:       {f}");
                //for (int i = 10; i < 70; ++i)
                //{
                //    BigFloat.Set(tmp1, i);
                //    BigFloat.Set(invX, 1);
                //    invX.Div(tmp1);
                //    BigFloat.Sqr(invX2, invX);
                //    int conv = FindConvergencePoint(invX2, gammaLnAsymptotic);
                //    Console.WriteLine($"Convergence point at {i}: {conv}");
                //    if (conv > 0)
                //    {
                //        using (var result = EvaluateSeries(invX2, gammaLnAsymptotic.Take(conv).ToArray()))
                //        using (var logx = BigFloatFactory.Create(tmp1))
                //        using (var tmp = BigFloatFactory.Create(tmp1))
                //        using (var f = GammaLn(tmp1))
                //        {
                //            result.Mul(invX);
                //            logx.Log();
                //            tmp.Sub(0.5);
                //            tmp.Mul(logx);
                //            f.Sub(tmp);
                //            f.Add(tmp1);
                //            f.Sub(LnSqrt2PI);
                //            Console.WriteLine($"Polynomial: {result}");
                //            Console.WriteLine($"Algebraic:  {f}");
                //            f.Sub(result);
                //            f.Abs();
                //            Console.WriteLine($"Diff:       {f}");
                //        }
                //    }
                //}
                for (int i = 0; i < 11; ++i)
                {
                    int conv = FindConvergencePoint(tmp1, expMinus1RatioMinus1RatioMinusHalfAt0);
                    Console.WriteLine($"Convergence point at {tmp1}: {conv}");
                    //if (conv < 0)
                    //    conv = xMinusLog1PlusAt0.Length;
                    if (conv > 0)
                    {
                        using (var result = EvaluateSeries(tmp1, expMinus1RatioMinus1RatioMinusHalfAt0.Take(conv).ToArray()))
                        using (var x = BigFloatFactory.Create(tmp1))
                        using (var f = BigFloatFactory.Create(tmp1))
                        {
                            f.Expm1();
                            f.Div(x);
                            f.Sub(1);
                            f.Div(x);
                            f.Sub(0.5);
                            Console.WriteLine($"Polynomial: {result}");
                            Console.WriteLine($"Algebraic:  {f}");
                            f.Sub(result);
                            f.Abs();
                            Console.WriteLine($"Diff:       {f}");
                        }
                    }
                    tmp1.Div2(1);
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

        private static readonly BigFloat reciprocalFactorialMinus1Threshold = BigFloatFactory.Create("0.025");
        // Truncated series 18: Reciprocal factorial minus 1
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] reciprocalFactorialMinus1At0 = new BigFloat[]
        {
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("0.57721566490153286060651209008240243104215933593992"),
            BigFloatFactory.Create("-0.65587807152025388107701951514539048127976638047858"),
            BigFloatFactory.Create("-0.042002635034095235529003934875429818711394500401106"),
            BigFloatFactory.Create("0.16653861138229148950170079510210523571778150224717"),
            BigFloatFactory.Create("-0.042197734555544336748208301289187391301652684189823"),
            BigFloatFactory.Create("-0.0096219715278769735621149216723481989753629422521130"),
            BigFloatFactory.Create("0.0072189432466630995423950103404465727099048008802383"),
            BigFloatFactory.Create("-0.0011651675918590651121139710840183886668093337953841"),
            BigFloatFactory.Create("-0.00021524167411495097281572996305364780647824192337834"),
            BigFloatFactory.Create("0.00012805028238811618615319862632816432339489209969368"),
            BigFloatFactory.Create("-0.000020134854780788238655689391421021818382294833297979"),
            BigFloatFactory.Create("-0.0000012504934821426706573453594738330922423226556211540"),
            BigFloatFactory.Create("0.0000011330272319816958823741296203307449433240048386211"),
            BigFloatFactory.Create("-0.00000020563384169776071034501541300205728365125790262934"),
            BigFloatFactory.Create("0.0000000061160951044814158178624986828553428672758657197123"),
            BigFloatFactory.Create("0.0000000050020076444692229300556650480599913030446127424945"),
            BigFloatFactory.Create("-0.0000000011812745704870201445881265654365055777387595049326"),
            BigFloatFactory.Create("0.00000000010434267116911005104915403323122501914007098231258"),
            BigFloatFactory.Create("0.0000000000077822634399050712540499373113607772260680861813929"),
            BigFloatFactory.Create("-0.0000000000036968056186422057081878158780857662365709634513610"),
            BigFloatFactory.Create("0.00000000000051003702874544759790154813228632318027268860697076")
        };

        /// <summary>
        /// Computes <c>1/Gamma(x+1) - 1</c> to high accuracy
        /// </summary>
        /// <param name="x">A real number &gt;= 0</param>
        public static BigFloat ReciprocalFactorialMinus1(BigFloat x)
        {
            if (x.IsGreater(reciprocalFactorialMinus1Threshold))
            {
                using (var denom = BigFloatFactory.Create(x))
                {
                    denom.Add(1);
                    denom.Gamma();
                    var result = BigFloatFactory.Create(1);
                    result.Div(denom);
                    result.Sub(1);
                    return result;
                }
            }
            else
            {
                return EvaluateSeries(x, reciprocalFactorialMinus1At0);
            }
        }

        public static BigFloat Log1Plus(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.Log1p();
            return result;
        }

        public static BigFloat Log1PlusExp(BigFloat x)
        {
            var result = BigFloatFactory.Create(x);
            result.Exp();
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


        private static readonly BigFloat expMinus1RatioMinus1RatioMinusHalfThresholdNeg = BigFloatFactory.Create("-0.075");
        private static readonly BigFloat expMinus1RatioMinus1RatioMinusHalfThresholdPos = BigFloatFactory.Create("0.075");
        // Truncated series 12: x - log(1 + x)
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] expMinus1RatioMinus1RatioMinusHalfAt0 = new BigFloat[]
        {
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("0.16666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("0.041666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("0.0083333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("0.0013888888888888888888888888888888888888888888888889"),
            BigFloatFactory.Create("0.00019841269841269841269841269841269841269841269841270"),
            BigFloatFactory.Create("0.000024801587301587301587301587301587301587301587301587"),
            BigFloatFactory.Create("0.0000027557319223985890652557319223985890652557319223986"),
            BigFloatFactory.Create("0.00000027557319223985890652557319223985890652557319223986"),
            BigFloatFactory.Create("0.000000025052108385441718775052108385441718775052108385442"),
            BigFloatFactory.Create("0.0000000020876756987868098979210090321201432312543423654535"),
            BigFloatFactory.Create("0.00000000016059043836821614599392377170154947932725710503488"),
            BigFloatFactory.Create("0.000000000011470745597729724713851697978682105666232650359634"),
            BigFloatFactory.Create("0.00000000000076471637318198164759011319857880704441551002397563"),
            BigFloatFactory.Create("0.000000000000047794773323873852974382074911175440275969376498477"),
            BigFloatFactory.Create("0.0000000000000028114572543455207631989455830103200162334927352045"),
            BigFloatFactory.Create("1.5619206968586226462216364350057333423519404084470e-16"),
            BigFloatFactory.Create("8.2206352466243297169559812368722807492207389918261e-18"),
            BigFloatFactory.Create("4.1103176233121648584779906184361403746103694959131e-19")
        };

        public static BigFloat ExpMinus1RatioMinus1RatioMinusHalf(BigFloat x)
        {
            if (x.IsLesser(expMinus1RatioMinus1RatioMinusHalfThresholdPos) && x.IsGreater(expMinus1RatioMinus1RatioMinusHalfThresholdNeg))
                return EvaluateSeries(x, expMinus1RatioMinus1RatioMinusHalfAt0);
            else if (x.IsInf() && x.IsPositive())
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

        private static readonly BigFloat xMinusLog1PlusThresholdNeg = BigFloatFactory.Create("-0.025");
        private static readonly BigFloat xMinusLog1PlusThresholdPos = BigFloatFactory.Create("0.025");
        // Truncated series 12: x - log(1 + x)
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] xMinusLog1PlusAt0 = new BigFloat[]
        {
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("0"),
            BigFloatFactory.Create("0.50000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.33333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("0.25000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.20000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("0.16666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-0.14285714285714285714285714285714285714285714285714"),
            BigFloatFactory.Create("0.12500000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.11111111111111111111111111111111111111111111111111"),
            BigFloatFactory.Create("0.10000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.090909090909090909090909090909090909090909090909091"),
            BigFloatFactory.Create("0.083333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("-0.076923076923076923076923076923076923076923076923077"),
            BigFloatFactory.Create("0.071428571428571428571428571428571428571428571428571"),
            BigFloatFactory.Create("-0.066666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("0.062500000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.058823529411764705882352941176470588235294117647059"),
            BigFloatFactory.Create("0.055555555555555555555555555555555555555555555555556"),
            BigFloatFactory.Create("-0.052631578947368421052631578947368421052631578947368"),
            BigFloatFactory.Create("0.050000000000000000000000000000000000000000000000000"),
            BigFloatFactory.Create("-0.047619047619047619047619047619047619047619047619048"),
            BigFloatFactory.Create("0.045454545454545454545454545454545454545454545454545"),
            BigFloatFactory.Create("-0.043478260869565217391304347826086956521739130434783"),
            BigFloatFactory.Create("0.041666666666666666666666666666666666666666666666667"),
            BigFloatFactory.Create("-0.040000000000000000000000000000000000000000000000000")
        };

        /// <summary>
        /// Computes <c>x - log(1+x)</c> to high accuracy.
        /// </summary>
        /// <param name="x">Any real number &gt;= -1</param>
        /// <returns>A real number &gt;= 0</returns>
        public static BigFloat XMinusLog1Plus(BigFloat x)
        {
            if (x.IsGreater(xMinusLog1PlusThresholdNeg) && x.IsLesser(xMinusLog1PlusThresholdPos))
            {
                return EvaluateSeries(x, xMinusLog1PlusAt0);
            }
            else
            {
                var result = Log1Plus(x);
                result.Neg();
                result.Add(x);
                return result;
            }
        }

        private static readonly BigFloat gammaLnSeriesThreshold = BigFloatFactory.Create("16");
        // Truncated series 9: GammaLn asymptotic
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] gammaLnAsymptotic = new BigFloat[]
        {
            BigFloatFactory.Create("0.083333333333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("-0.0027777777777777777777777777777777777777777777777778"),
            BigFloatFactory.Create("0.00079365079365079365079365079365079365079365079365079"),
            BigFloatFactory.Create("-0.00059523809523809523809523809523809523809523809523809"),
            BigFloatFactory.Create("0.00084175084175084175084175084175084175084175084175084"),
            BigFloatFactory.Create("-0.0019175269175269175269175269175269175269175269175269"),
            BigFloatFactory.Create("0.0064102564102564102564102564102564102564102564102564"),
            BigFloatFactory.Create("-0.029550653594771241830065359477124183006535947712418"),
            BigFloatFactory.Create("0.17964437236883057316493849001588939669435025472177"),
            BigFloatFactory.Create("-1.3924322169059011164274322169059011164274322169059"),
            BigFloatFactory.Create("13.402864044168391994478951000690131124913733609386"),
            BigFloatFactory.Create("-156.84828462600201730636513245208897382810426288687"),
            BigFloatFactory.Create("2193.1033333333333333333333333333333333333333333333"),
            BigFloatFactory.Create("-36108.771253724989357173265219242230736483610046828"),
            BigFloatFactory.Create("691472.26885131306710839525077567346755333407168780"),
            BigFloatFactory.Create("-15238221.539407416192283364958886780518659076533839"),
            BigFloatFactory.Create("382900751.39141414141414141414141414141414141414141"),
            BigFloatFactory.Create("-10882266035.784391089015149165525105374729434879811"),
            BigFloatFactory.Create("347320283765.00225225225225225225225225225225225225"),
            BigFloatFactory.Create("-12369602142269.274454251710349271324881080978641954"),
            BigFloatFactory.Create("488788064793079.33507581516251802290210847053890567"),
            BigFloatFactory.Create("-21320333960919373.896975058982136838557465453319852"),
            BigFloatFactory.Create("1021775296525700077.5652876280535855003940110323089"),
            BigFloatFactory.Create("-53575472173300203610.827709191969204484849040543659"),
            BigFloatFactory.Create("3061578263704883415043.1510513296227581941867656153"),
            BigFloatFactory.Create("-189999174263992040502937.14293069429029473424589962"),
            BigFloatFactory.Create("12763374033828834149234951.377697825976541633608830"),
            BigFloatFactory.Create("-925284717612041630723024234.83476227795193312434692"),
            BigFloatFactory.Create("72188225951856102978360501873.016379224898404202597"),
            BigFloatFactory.Create("-6045183405995856967743148238754.5472860661443959672"),
            BigFloatFactory.Create("542067047157009454519347781482610.00136612021857923")
        };

        /// <summary>
        /// Computes <c>GammaLn(x) - (x-0.5)*log(x) + x - 0.5*log(2*pi)</c> for x &gt;= 10
        /// </summary>
        /// <param name="x">A real number &gt;= 10</param>
        /// <returns></returns>
        private static BigFloat GammaLnSeries(BigFloat x)
        {
            if (x.IsLesser(gammaLnSeriesThreshold))
            {
                using (var logx = BigFloatFactory.Create(x))
                using (var tmp = BigFloatFactory.Create(x))
                {
                    var result = GammaLn(x);
                    logx.Log();
                    tmp.Sub(0.5);
                    tmp.Mul(logx);
                    result.Sub(tmp);
                    result.Add(x);
                    result.Sub(LnSqrt2PI);
                    return result;
                }
            }
            else
            {
                using (var invX = BigFloatFactory.Create(1))
                using (var invX2 = BigFloatFactory.Empty())
                {
                    invX.Div(x);
                    BigFloat.Sqr(invX2, invX);
                    var result = EvaluateSeries(invX2, gammaLnAsymptotic);
                    result.Mul(invX);
                    return result;
                }
            }
        }

        /// <summary>
        /// Computes <c>log(x^a e^(-x)/Gamma(a))</c> to high accuracy.
        /// </summary>
        /// <param name="a">A positive real number</param>
        /// <param name="x"></param>
        /// <returns></returns>
        public static BigFloat GammaUpperLogScale(BigFloat a, BigFloat x)
        {
            if (DoubleMethods.IsPositiveInfinity(x) || DoubleMethods.IsPositiveInfinity(a))
                return BigFloatFactory.NegativeInfinity;
            if (a.IsLesser(gammaLnSeriesThreshold))
            {
                using (var gammaLnA = GammaLn(a))
                {
                    var result = BigFloatFactory.Create(x);
                    result.Log();
                    result.Mul(a);
                    result.Sub(x);
                    result.Sub(gammaLnA);
                    return result;
                }
            }
            else
            {
                using (var tmp = BigFloatFactory.Create(x))
                {
                    tmp.Sub(a);
                    tmp.Div(a);
                    using (var phi = XMinusLog1Plus(tmp))
                    {
                        var result = GammaLnSeries(a);
                        result.Neg();
                        BigFloat.Set(tmp, a);
                        tmp.Log();
                        tmp.Div2(1);
                        tmp.Sub(LnSqrt2PI);
                        result.Add(tmp);
                        phi.Mul(a);
                        result.Sub(phi);
                        return result;// 0.5 * Math.Log(a) - MMath.LnSqrt2PI - GammaLnSeries(a) - a * phi;
                    }
                }
            }
        }

        private static readonly BigFloat gammaMinusReciprocalThreshold = BigFloatFactory.Create("0.0075");
        // Truncated series 19: Gamma(x) - 1/x
        // Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py
        private static readonly BigFloat[] gammaMinusReciprocalAt0 = new BigFloat[]
        {
            BigFloatFactory.Create("-0.57721566490153286060651209008240243104215933593992"),
            BigFloatFactory.Create("0.98905599532797255539539565150063470793918352072821"),
            BigFloatFactory.Create("-0.90747907608088628901656016735627511492861144907256"),
            BigFloatFactory.Create("0.98172808683440018733638029402185085036057367972347"),
            BigFloatFactory.Create("-0.98199506890314520210470141379137467551742650714720"),
            BigFloatFactory.Create("0.99314911462127619315386725332865849803749075523943"),
            BigFloatFactory.Create("-0.99600176044243153397007841966456668673529880955458"),
            BigFloatFactory.Create("0.99810569378312892197857540308836723752396852479018"),
            BigFloatFactory.Create("-0.99902526762195486779467805964888808853230396352566"),
            BigFloatFactory.Create("0.99951565607277744106705087759437019443450329799460"),
            BigFloatFactory.Create("-0.99975659750860128702584244914060923599695138562883"),
            BigFloatFactory.Create("0.99987827131513327572617164259000321938762910895432"),
            BigFloatFactory.Create("-0.99993906420644431683585223136895513185794350282804"),
            BigFloatFactory.Create("0.99996951776348210449861140509195350726552804247988"),
            BigFloatFactory.Create("-0.99998475269937704874370963172444753832608332577145"),
            BigFloatFactory.Create("0.99999237447907321585539509450510782583381634469466"),
            BigFloatFactory.Create("-0.99999618658947331202896495779561431380201731243263"),
            BigFloatFactory.Create("0.99999809308113089205186619151459489773169557198830"),
            BigFloatFactory.Create("-0.99999904646891115771748687947054372632469616324955")
        };

        /// <summary>
        /// Compute <c>Gamma(x) - 1/x</c> to high accuracy
        /// </summary>
        /// <param name="x">A real number &gt;= 0</param>
        /// <returns></returns>
        public static BigFloat GammaSeries(BigFloat x)
        {
            if (x.IsGreater(gammaMinusReciprocalThreshold))
            {
                using (var invX = BigFloatFactory.Create(1))
                {
                    invX.Div(x);
                    var result = BigFloatFactory.Create(x);
                    result.Gamma();
                    result.Sub(invX);
                    return result;
                }
            }
            else
                return EvaluateSeries(x, gammaMinusReciprocalAt0);
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

        /// <summary>
        /// Returns the positive distance between a value and the next representable value that is larger in magnitude.
        /// </summary>
        /// <param name="value">Any double-precision value.</param>
        /// <returns></returns>
        public static BigFloat Ulp(BigFloat value)
        {
            if (value.IsNan())
                return BigFloatFactory.Create(value);
            if (value.IsInf())
                return BigFloatFactory.Create(0);
            using (var absValue = BigFloatFactory.Create(value))
            {
                absValue.Abs();
                var result = NextBigFloat(absValue);
                result.Sub(absValue);
                return result;
            }
        }

        #endregion
    }
}
