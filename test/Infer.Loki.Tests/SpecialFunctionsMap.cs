using Loki.Mapping;
using Loki.Mapping.Maps;
using System;
using System.Numerics.MPFR;
using static Infer.Loki.Mappings.SpecialFunctionsMethods;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Math;

namespace Infer.Loki.Tests
{
    class SpecialFunctionsMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Gamma, Gamma);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.GammaLn, GammaLn);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Digamma, Digamma);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Trigamma, Trigamma);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Tetragamma, Tetragamma);
            //mappers.MethodMapper.CreateMap<Func<double, double, bool, double>, Func<BigFloat, BigFloat, bool, BigFloat>>(MMath.GammaUpper, GammaUpper);
            //mappers.MethodMapper.CreateMap<Func<double, double, double>, Func<BigFloat, BigFloat, BigFloat>>(MMath.GammaLower, GammaLower);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.ReciprocalFactorialMinus1, ReciprocalFactorialMinus1);

            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Log1Plus, Log1Plus);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Log1MinusExp, Log1MinusExp);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.ExpMinus1, ExpMinus1);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.ExpMinus1RatioMinus1RatioMinusHalf, ExpMinus1RatioMinus1RatioMinusHalf);

            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Erfc, Erfc);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NormalCdf, NormalCdf);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NormalCdfLn, NormalCdfLn);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NormalCdfRatio, NormalCdfRatio);

            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NextDouble, NextBigFloat);
            mappers.MethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.PreviousDouble, PreviousBigFloat);

            mappers.MemberMapper.CreateMap("Microsoft.ML.Probabilistic.Math.MMath.Sqrt2", () => Sqrt2);
            mappers.MemberMapper.CreateMap("Microsoft.ML.Probabilistic.Math.MMath.Sqrt2PI", () => Sqrt2PI);
            mappers.MemberMapper.CreateMap("Microsoft.ML.Probabilistic.Math.MMath.LnSqrt2PI", () => LnSqrt2PI);
        }
    }
}
