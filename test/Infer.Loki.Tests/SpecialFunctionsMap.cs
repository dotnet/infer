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
using System.Reflection;
using Infer.Loki.Mappings;

namespace Infer.Loki.Tests
{
    class SpecialFunctionsMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Gamma, Gamma);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.GammaLn, GammaLn);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Digamma, Digamma);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Trigamma, Trigamma);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Tetragamma, Tetragamma);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.ReciprocalFactorialMinus1, ReciprocalFactorialMinus1);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double, double>, Func<BigFloat, BigFloat, BigFloat>>(MMath.GammaUpperLogScale, GammaUpperLogScale);

            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Log1Plus, Log1Plus);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Log1PlusExp, Log1PlusExp);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Log1MinusExp, Log1MinusExp);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.LogExpMinus1, LogExpMinus1);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.ExpMinus1, ExpMinus1);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.ExpMinus1RatioMinus1RatioMinusHalf, ExpMinus1RatioMinus1RatioMinusHalf);

            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Erfc, Erfc);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NormalCdf, NormalCdf);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NormalCdfLn, NormalCdfLn);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NormalCdfRatio, NormalCdfRatio);

            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.NextDouble, NextBigFloat);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.PreviousDouble, PreviousBigFloat);
            mappers.FuelBasedMethodMapper.CreateMap<Func<double, double>, Func<BigFloat, BigFloat>>(MMath.Ulp, Ulp);

            mappers.FuelBasedMethodMapper.CreateMap
            (
                typeof(MMath).GetMethod(
                    "GammaSeries",
                    BindingFlags.Static | BindingFlags.NonPublic,
                    null,
                    new Type[] { typeof(double) },
                    null),
                new Func<BigFloat, BigFloat>(GammaSeries).Method
            );

            mappers.FuelBasedMethodMapper.CreateMap
            (
                typeof(MMath).GetMethod(
                    "GammaLnSeries",
                    BindingFlags.Static | BindingFlags.NonPublic,
                    null,
                    new Type[] { typeof(double) },
                    null),
                new Func<BigFloat, BigFloat>(GammaLnSeries).Method
            );

            mappers.FuelBasedMethodMapper.CreateMap
            (
                typeof(MMath).GetMethod(
                    "XMinusLog1Plus",
                    BindingFlags.Static | BindingFlags.NonPublic,
                    null,
                    new Type[] { typeof(double) },
                    null),
                new Func<BigFloat, BigFloat>(XMinusLog1Plus).Method
            );

            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField(nameof(MMath.Sqrt2)),
                typeof(SpecialFunctionsMethods).GetField(nameof(Sqrt2)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField(nameof(MMath.Sqrt2PI)),
                typeof(SpecialFunctionsMethods).GetField(nameof(Sqrt2PI)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField(nameof(MMath.InvSqrt2PI)),
                typeof(SpecialFunctionsMethods).GetField(nameof(InvSqrt2PI)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField(nameof(MMath.LnSqrt2PI)),
                typeof(SpecialFunctionsMethods).GetField(nameof(LnSqrt2PI)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField(nameof(MMath.Ln2)),
                typeof(SpecialFunctionsMethods).GetField(nameof(Ln2)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField("DefaultBetaEpsilon", BindingFlags.NonPublic | BindingFlags.Static),
                typeof(SpecialFunctionsMethods).GetField(nameof(DefaultBetaEpsilon)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField("LogisticGaussianVarianceThreshold", BindingFlags.NonPublic | BindingFlags.Static),
                typeof(SpecialFunctionsMethods).GetField(nameof(LogisticGaussianVarianceThreshold)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(Quadrature).GetField("AdaptiveQuadratureMaxNodes", BindingFlags.NonPublic | BindingFlags.Static),
                typeof(SpecialFunctionsMethods).GetField(nameof(AdaptiveQuadratureMaxNodes)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField("logisticGaussianSeriesApproximmationThreshold", BindingFlags.NonPublic | BindingFlags.Static),
                typeof(SpecialFunctionsMethods).GetField(nameof(LogisticGaussianSeriesApproximmationThreshold)));
            mappers.FuelBasedMemberMapper.CreateMap(
                typeof(MMath).GetField("NormalCdfMomentRatioMaxTerms", BindingFlags.NonPublic | BindingFlags.Static),
                typeof(SpecialFunctionsMethods).GetField(nameof(NormalCdfMomentRatioMaxTerms)));
        }
    }
}
