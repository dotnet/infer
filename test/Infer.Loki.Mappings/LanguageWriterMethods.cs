using Loki.Mapping;
using Loki.Mapping.Methods;
using Loki.Shared;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics.MPFR;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Mappings
{
    public static class LanguageWriterMethods
    {
        public static long GetNegativeZeroBits()
        {
            return -9223372036854775808; // actual value. Doesn't really matter as we aren't supposed to call this method anyway.
        }

        public static bool IsNegativeZero(object _, DoubleWithTransformingPrecision x)
        {
            return x.InternalValue.IsZero() && x.InternalValue.IsNegative();
        }

        private static readonly string MapDispatcherFullName = $"global::Loki.Shared.MapDispatcher";
        private static readonly string formatString = $"b{BigFloatFactory.FloatingPointBase}d0";
        private static readonly string DoubleWithTransformingPrecisionFullName = $"global::{typeof(DoubleWithTransformingPrecision).FullName}";
        private static readonly MethodInfo OriginalAppendLiteralExpressionMethod =
            typeof(Microsoft.ML.Probabilistic.Compiler.Quoter).Assembly
                    .GetType("Microsoft.ML.Probabilistic.Compiler.LanguageWriter")
                    .GetMethod(
                        "AppendLiteralExpression",
                        BindingFlags.Instance | BindingFlags.NonPublic,
                        null,
                        new Type[] { typeof(StringBuilder), typeof(ILiteralExpression) },
                        null);


        public static void AppendLiteralExpression(object self, StringBuilder sb, ILiteralExpression ile)
        {
            if (ile.Value == null)
            {
                sb.Append("null");
                return;
            }

            Type t = ile.GetExpressionType();
            if (t == typeof(DoubleWithTransformingPrecision))
            {
                BigFloat bf = ((DoubleWithTransformingPrecision)ile.Value).InternalValue;
                if (DoubleMethods.IsNegativeInfinity(bf))
                {
                    sb.Append(MapDispatcherFullName + ".System_Double_NegativeInfinity");
                }
                else if (DoubleMethods.IsPositiveInfinity(bf))
                {
                    sb.Append(MapDispatcherFullName + ".System_Double_PositiveInfinity");
                }
                else if (DoubleMethods.IsNaN(bf))
                {
                    sb.Append(MapDispatcherFullName + ".System_Double_NaN");
                }
                else
                {
                    string s = bf.ToString(formatString, CultureInfo.InvariantCulture);
                    sb.Append($"new {DoubleWithTransformingPrecisionFullName}(\"{s}\")");
                }
            }
            else
            {
                OriginalAppendLiteralExpressionMethod.Invoke(self, new object[] { sb, ile });
            }
        }
    }
}
