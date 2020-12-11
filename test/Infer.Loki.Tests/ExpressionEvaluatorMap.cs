using Infer.Loki.Mappings;
using Loki.Mapping;
using Loki.Mapping.Maps;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Tests
{
    class ExpressionEvaluatorMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.DirectMethodMapper.CreateMap<Func<object, IExpression>, Func<ulong, object, IExpression>>(ExpressionEvaluator.Quote, ExpressionEvaluatorMethods.Quote);
        }
    }
}
