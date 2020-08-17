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

        public void MapAll(Mappers mappers)
        {
            mappers.PermanentMethodMapper.CreateMap<Func<Type, Type, Converter>, Func<Type, Type, Converter>>(Conversion.GetPrimitiveConverter, ConversionMethods.GetPrimitiveConverter);
            mappers.PermanentMethodMapper.CreateMap<TryGetConversionDelegate, TryGetConversionDelegate>(Conversion.TryGetPrimitiveConversion, ConversionMethods.TryGetPrimitiveConversion);
            mappers.PermanentMethodMapper.CreateMap<TryGetConversionDelegate, TryGetConversionDelegate>(Conversion.TryGetConversion, ConversionMethods.TryGetConversion);
        }
    }
}
