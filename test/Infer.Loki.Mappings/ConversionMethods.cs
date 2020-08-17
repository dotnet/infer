using Loki.Mapping.Methods;
using Loki.Shared;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Mappings
{
    public static class ConversionMethods
    {
        public static Converter GetPrimitiveConverter(Type fromType, Type toType)
        {
            if (toType == typeof(DoubleWithTransformingPrecision))
            {
                if (fromType == typeof(DoubleWithTransformingPrecision))
                    return x => new DoubleWithTransformingPrecision((DoubleWithTransformingPrecision)x);
                else
                    return x => new DoubleWithTransformingPrecision(ConvertMethods.ToBigFloat(x));
            }
            else
                return Conversion.GetPrimitiveConverter(fromType, toType);
        }

        public static bool TryGetPrimitiveConversion(Type fromType, Type toType, out Conversion info)
        {
            var fromTypeIsDoubleWithTransformingPrecision = fromType == typeof(DoubleWithTransformingPrecision);
            var toTypeIsDoubleWithTransformingPrecision = toType == typeof(DoubleWithTransformingPrecision);
            if (fromTypeIsDoubleWithTransformingPrecision || toTypeIsDoubleWithTransformingPrecision)
            {
                info = new Conversion();
                if (fromTypeIsDoubleWithTransformingPrecision && toTypeIsDoubleWithTransformingPrecision)
                {
                    info.SubclassCount = 1000;
                    return true;
                }
                else
                {
                    info.Converter = GetPrimitiveConverter(fromType, toType);
                    // from now on, explicit is the default
                    info.IsExplicit = true;
                    TypeCode fromTypeCode = Type.GetTypeCode(fromType);
                    if (fromTypeCode == TypeCode.DateTime)
                    {
                        // DateTime can only be converted to itself or string
                        return false;
                    }
                    switch (fromTypeCode)
                    {
                        case TypeCode.Char: 
                            return false;
                        case TypeCode.Byte:
                        case TypeCode.SByte:
                        case TypeCode.UInt16:
                        case TypeCode.Int16:
                        case TypeCode.UInt32:
                        case TypeCode.Int32:
                        case TypeCode.UInt64:
                        case TypeCode.Int64:
                        case TypeCode.Single:
                            info.IsExplicit = false;
                            break;
                    }
                    if (info.IsExplicit && (fromTypeIsDoubleWithTransformingPrecision || fromType.IsPrimitive))
                    {
                        // wrap the converter with a compatibility check
                        Converter conv = info.Converter;
                        Converter back = GetPrimitiveConverter(toType, fromType);
                        info.Converter = fromValue =>
                        {
                            object toValue = conv(fromValue);
                            object backValue = back(toValue);
                            if (!backValue.Equals(fromValue))
                                throw new ArgumentException("The value " + fromValue + " does not convert to " + toValue.GetType().Name);
                            return toValue;
                        };
                    }
                    return true;
                }
            }
            else
                return Conversion.TryGetPrimitiveConversion(fromType, toType, out info);
        }

        public static bool TryGetConversion(Type fromType, Type toType, out Conversion info)
        {
            if (fromType == typeof(DoubleWithTransformingPrecision) && toType.IsPrimitive)
                return TryGetPrimitiveConversion(fromType, toType, out info);
            else
                return Conversion.TryGetConversion(fromType, toType, out info);
        }
    }
}
