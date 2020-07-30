using Loki.Mapping;
using Loki.Shared;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Globalization;

namespace Infer.Loki.Mappings
{
    public static class ExpressionEvaluatorMethods
    {
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        public static IExpression Quote(object p)
        {
            if (p is DoubleWithTransformingPrecision dwtp)
            {
                return Builder.NewObject(
                    Builder.TypeRef(typeof(DoubleWithTransformingPrecision)),
                    Builder.LiteralExpr(dwtp.ToInvariantRoundTrippingString()));
            }
            else
                return ExpressionEvaluator.Quote(p);
        }
    }
}
