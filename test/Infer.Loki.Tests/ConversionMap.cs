using Infer.Loki.Mappings;
using Loki.Mapping;
using Loki.Mapping.Maps;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Tests
{
    class ConversionMap : IMap
    {
        private delegate bool TryGetConversionDelegate(Type fromType, Type toType, out Conversion info);
        private delegate bool TryGetConversionDelegateWithOperationId(ulong operationId, Type fromType, Type toType, out Conversion info);

        public void MapAll(Mappers mappers)
        {
            mappers.DirectMethodMapper.CreateMap<Func<Type, Type, Converter>, Func<ulong, Type, Type, Converter>>(Conversion.GetPrimitiveConverter, ConversionMethods.GetPrimitiveConverter);
            mappers.DirectMethodMapper.CreateMap<TryGetConversionDelegate, TryGetConversionDelegateWithOperationId>(Conversion.TryGetPrimitiveConversion, ConversionMethods.TryGetPrimitiveConversion);
            mappers.DirectMethodMapper.CreateMap<TryGetConversionDelegate, TryGetConversionDelegateWithOperationId>(Conversion.TryGetConversion, ConversionMethods.TryGetConversion);
        }
    }
}
