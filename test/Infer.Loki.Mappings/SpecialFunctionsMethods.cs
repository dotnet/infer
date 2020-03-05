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
        [DllImport(MPFRLibrary.FileName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mpfr_gamma_inc([In, Out] mpfr_struct rop, [In, Out] mpfr_struct op, [In, Out] mpfr_struct op2, int rnd);

        public static BigFloat Gamma(BigFloat x)
        {
            var result = BigFloatFactory.Empty();
            BigFloat.Gamma(result, x);
            return result;
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

        // Trigamma

        // Tetragamma

        public static BigFloat GammaUpper(BigFloat a, BigFloat x, bool regularized = true)
        {
            var result = BigFloatFactory.Empty();
            mpfr_gamma_inc(result.Value, x.Value, a.Value, BigFloat.GetRounding(null));
            if (regularized)
            {
                using (var gammaa = Gamma(x))
                    result.Div(gammaa);
            }
            return result;
        }

        public static BigFloat GammaLower(BigFloat a, BigFloat x)
        {
            var result = BigFloatFactory.Create(1.0);
            using (var upper = GammaUpper(a, x))
                result.Sub(upper);
            return result;
        }

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
