using Infer.Loki.Mappings;
using Loki.Mapping;
using Loki.Mapping.Maps;
using Loki.Shared;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Tests
{
    class LanguageWriterMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.DirectMethodMapper.CreateMap(
                typeof(Microsoft.ML.Probabilistic.Compiler.Quoter).Assembly
                    .GetType("Microsoft.ML.Probabilistic.Compiler.CSharpWriter")
                    .GetMethod(
                        "GetNegativeZeroBits",
                        BindingFlags.Static | BindingFlags.NonPublic,
                        null,
                        Type.EmptyTypes,
                        null),
                new Func<ulong, long>(LanguageWriterMethods.GetNegativeZeroBits).Method);
            mappers.DirectMethodMapper.CreateMap(
                typeof(Microsoft.ML.Probabilistic.Compiler.Quoter).Assembly
                    .GetType("Microsoft.ML.Probabilistic.Compiler.CSharpWriter")
                    .GetMethod(
                        "IsNegativeZero",
                        BindingFlags.Instance | BindingFlags.NonPublic,
                        null,
                        new Type[] { typeof(double) },
                        null),
                new Func<ulong, object, DoubleWithTransformingPrecision, bool>(LanguageWriterMethods.IsNegativeZero).Method);
            mappers.DirectMethodMapper.CreateMap(
                typeof(Microsoft.ML.Probabilistic.Compiler.Quoter).Assembly
                    .GetType("Microsoft.ML.Probabilistic.Compiler.LanguageWriter")
                    .GetMethod(
                        "AppendLiteralExpression",
                        BindingFlags.Instance | BindingFlags.NonPublic,
                        null,
                        new Type[] { typeof(StringBuilder), typeof(ILiteralExpression) },
                        null),
                new Action<ulong, object, StringBuilder, ILiteralExpression>(LanguageWriterMethods.AppendLiteralExpression).Method);
            mappers.DirectMethodMapper.CreateMap(
                typeof(Microsoft.ML.Probabilistic.Compiler.Quoter).Assembly
                    .GetType("Microsoft.ML.Probabilistic.Compiler.CSharpWriter")
                    .GetMethod(
                        "AppendLiteralExpression",
                        BindingFlags.Instance | BindingFlags.NonPublic,
                        null,
                        new Type[] { typeof(StringBuilder), typeof(ILiteralExpression) },
                        null),
                new Action<ulong, object, StringBuilder, ILiteralExpression>(LanguageWriterMethods.AppendLiteralExpression).Method);
        }
    }
}
