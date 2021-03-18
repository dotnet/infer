// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Emit;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Reflection
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    public delegate object Converter(object toConvert);

    public struct Conversion
    {
        public const int SameTypeCodeSubclassCount = 1000;
        public const int OpImplicitSubclassCount = 2000;
        public const int ChangeRankSubclassCount = 3000;
        public const int SpecialImplicitSubclassCount = 10000;

        public Converter Converter;

        /// <summary>
        /// The number of subclass edges between the two types.
        /// </summary>
        /// <remarks>This is only valid if converter is null, i.e. no conversion is needed.  
        /// If the two types are the same, then SubclassCount == 0.  
        /// If one is a direct subclass of the other, SubclassCount == 1.
        /// If one is a subclass of a subclass of the other, SubclassCount == 2, and so on.
        /// </remarks>
        public int SubclassCount;

        /// <summary>
        /// True if the conversion is explicit.
        /// </summary>
        /// <remarks>Must be false if converter is null.
        /// An implicit conversion must always succeed and does not lose information.
        /// Otherwise, it is explicit.
        /// </remarks>
        public bool IsExplicit;

        public override string ToString()
        {
            return (IsExplicit ? "Explicit" : "Implicit") + " " + SubclassCount + ((Converter == null) ? "" : (" " + Converter.Method.Name));
        }

        /// <summary>
        /// True if A is a more specific conversion than B.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>True if A is a more specific conversion than B.</returns>
        /// <remarks>The following criteria are applied in order:
        /// 1. A null conversion versus a non-null conversion.
        /// 2. Among null conversions, the one crossing fewer subclass links.
        /// 3. An implicit conversion versus an explicit conversion.
        /// </remarks>
        public static bool operator <(Conversion a, Conversion b)
        {
            //if (a == null || b == null) return false;
            if (a.Converter == null)
            {
                if (b.Converter == null) return a.SubclassCount < b.SubclassCount;
                else return true;
            }
            if (b.Converter == null) return false;
            return !a.IsExplicit && b.IsExplicit;
        }

        public static bool operator >(Conversion a, Conversion b)
        {
            return (b < a);
        }

        /// <summary>
        /// Returns a numerical weight such that (a.GetWeight() &lt; b.GetWeight()) iff (a &lt; b)
        /// </summary>
        /// <returns></returns>
        public float GetWeight()
        {
            if (Converter == null)
            {
                return SubclassCount;
            }
            else
            {
                int maxSubclassCount = 100000;
                return maxSubclassCount*(IsExplicit ? 2 : 1);
            }
        }

        public static float GetWeight(IEnumerable<Conversion> array)
        {
            float weight = 0.0F;
            int maxLength = 1000;
            int index = maxLength;
            foreach (Conversion c in array) weight += c.GetWeight()*(index--);
            return weight;
        }

        public static Converter GetPrimitiveConverter(Type fromType, Type toType)
        {
            Debug.Assert(toType.IsPrimitive);
            TypeCode typeCode = Type.GetTypeCode(toType);
            string name = typeCode.ToString();
            Type[] actuals = new Type[] {fromType};
            MethodInfo method = typeof (Convert).GetMethod("To" + name, actuals);
            // CreateDelegate doesn't work since method is not of type Converter:
            //(Converter)Delegate.CreateDelegate(typeof(Converter), method);
            return delegate (object value)
            {
                return Util.Invoke(method, null, value);
            };
        }

        /// <summary>
        /// Get a Conversion structure to a primitive type.
        /// </summary>
        /// <param name="fromType">Any type.</param>
        /// <param name="toType">A primitive type.</param>
        /// <param name="info"></param>
        /// <returns>false if no conversion exists.</returns>
        public static bool TryGetPrimitiveConversion(Type fromType, Type toType, out Conversion info)
        {
            Debug.Assert(toType.IsPrimitive);
            info = new Conversion();
            // not needed: info.isExplicit = false;
            TypeCode fromTypeCode = Type.GetTypeCode(fromType);
            TypeCode toTypeCode = Type.GetTypeCode(toType);
            if (fromTypeCode == toTypeCode)
            {
                // fromType has the same TypeCode but is not assignable to toType.
                info.SubclassCount = SameTypeCodeSubclassCount;
                return true;
            }
            info.Converter = GetPrimitiveConverter(fromType, toType);
            // from now on, explicit is the default
            info.IsExplicit = true;
            // string and DateTime are not actually primitive types, but we leave them in anyway.
            if (toTypeCode == TypeCode.String)
            {
                // anything converts to string
                return true;
            }
            // string and object convert to anything, but not implicitly
            bool ok = (fromTypeCode == TypeCode.String || fromTypeCode == TypeCode.Object);
            if (fromTypeCode == TypeCode.DateTime || (toTypeCode == TypeCode.DateTime && !ok))
            {
                // DateTime can only be converted to itself or string
                return false;
            }
            // The primitive conversions are listed here:
            // ms-help://MS.VSCC.v80/MS.MSDN.v80/MS.NETDEVFX.v20.en/cpref2/html/T_System_Convert.htm
            // The implicit conversions are listed here:
            // ms-help://MS.VSCC.v80/MS.MSDN.v80/MS.NETDEVFX.v20.en/cpref10/html/M_System_Reflection_Binder_ChangeType_1_f31c470b.htm
            // Conversion from signed to unsigned cannot be implicit, since it may fail.
            switch (fromTypeCode)
            {
                case TypeCode.Char:
                    switch (toTypeCode)
                    {
                        case TypeCode.SByte:
                        case TypeCode.Byte:
                        case TypeCode.Int16:
                        case TypeCode.Int32:
                        case TypeCode.Int64:
                        case TypeCode.UInt16:
                        case TypeCode.UInt32:
                        case TypeCode.UInt64:
                            info.IsExplicit = false;
                            break;
                        case TypeCode.Boolean:
                        case TypeCode.Single:
                        case TypeCode.Double:
                        case TypeCode.Decimal:
                            return false;
                    }
                    break;
                case TypeCode.Byte: // unsigned
                    switch (toTypeCode)
                    {
                        case TypeCode.Char:
                        case TypeCode.Int16:
                        case TypeCode.Int32:
                        case TypeCode.Int64:
                        case TypeCode.UInt16:
                        case TypeCode.UInt32:
                        case TypeCode.UInt64:
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.SByte: // signed
                    switch (toTypeCode)
                    {
                        case TypeCode.Int16:
                        case TypeCode.Int32:
                        case TypeCode.Int64:
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.UInt16:
                    switch (toTypeCode)
                    {
                        case TypeCode.UInt32:
                        case TypeCode.Int32:
                        case TypeCode.UInt64:
                        case TypeCode.Int64:
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.Int16:
                    switch (toTypeCode)
                    {
                        case TypeCode.Int32:
                        case TypeCode.Int64:
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.UInt32:
                    switch (toTypeCode)
                    {
                        case TypeCode.UInt64:
                        case TypeCode.Int64:
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.Int32:
                    switch (toTypeCode)
                    {
                        case TypeCode.Int64:
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.UInt64:
                    switch (toTypeCode)
                    {
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.Int64:
                    switch (toTypeCode)
                    {
                        case TypeCode.Single:
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
                case TypeCode.Single:
                    switch (toTypeCode)
                    {
                        case TypeCode.Double:
                            info.IsExplicit = false;
                            break;
                    }
                    break;
            }
            if (info.IsExplicit && fromType.IsPrimitive)
            {
                // wrap the converter with a compatibility check
                Converter conv = info.Converter;
                Converter back = GetPrimitiveConverter(toType, fromType);
                info.Converter = delegate(object fromValue)
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

        /// <summary>
        /// Change array rank and convert elements.
        /// </summary>
        /// <param name="fromArray"></param>
        /// <param name="toRank">Can be smaller, larger, or equal to fromArray.Rank.</param>
        /// <param name="toElementType"></param>
        /// <param name="conv"></param>
        /// <returns>A new array of rank toRank with the same contents as fromArray.</returns>
        public static Array ChangeRank(Array fromArray, int toRank, Type toElementType, Converter conv)
        {
            int fromRank = fromArray.Rank;
            int[] lengths = new int[toRank];
            int minRank = System.Math.Min(fromRank, toRank);
            // if fromRank == 3 and toRank == 1 then lengths = fromArray[1:2] (exclude fromArray[0])
            for (int i = 0; i < minRank; i++) lengths[i] = fromArray.GetLength(fromRank - minRank + i);
            for (int i = minRank; i < toRank; i++) lengths[i] = 1;
            Array toArray = Array.CreateInstance(toElementType, lengths);
            if (toArray.Length != fromArray.Length)
            {
                throw new ArgumentException("The input array has true rank greater than " + toRank.ToString(CultureInfo.InvariantCulture));
            }
            int[] fromIndex = new int[fromRank];
            int[] toIndex = new int[toRank];
            if (fromRank == 1)
            {
                for (int i = 0; i < lengths[0]; i++)
                {
                    object item = fromArray.GetValue(i);
                    object value;
                    if (conv != null) value = conv(item);
                    else value = item;
                    toIndex[toRank - 1] = i;
                    toArray.SetValue(value, toIndex);
                }
            }
            else if (toRank == 1)
            {
                for (int i = 0; i < lengths[0]; i++)
                {
                    // here we assume index[>0]=0
                    fromIndex[fromRank - 1] = i;
                    object item = fromArray.GetValue(fromIndex);
                    object value;
                    if (conv != null) value = conv(item);
                    else value = item;
                    toArray.SetValue(value, i);
                }
            }
            else
            {
                throw new NotImplementedException();
            }
            return toArray;
        }

        public static bool IsNullable(Type type)
        {
            return !type.IsValueType || (type.IsGenericType && type.GetGenericTypeDefinition().Equals(typeof (Nullable<>)));
        }

        // must be kept in sync with Binding.TypesAssignableFrom
        private static bool IsAssignableFrom(Type toType, Type fromType, out int subclassCount)
        {
            subclassCount = 0;
            for (Type baseType = fromType; baseType != null; baseType = baseType.BaseType)
            {
                if (baseType.Equals(typeof (object)))
                {
                    break;
                }
                if (baseType.Equals(toType)) return true;
                subclassCount++;
            }
            Type[] faces = fromType.GetInterfaces();
            foreach (Type face in faces)
            {
                if (face.Equals(toType)) return true;
                subclassCount++;
            }
            // array covariance (C# 2.0 specification, sec 20.5.9)
            if (fromType.IsArray && fromType.GetArrayRank() == 1 && toType.IsGenericType && toType.GetGenericTypeDefinition().Equals(typeof(IList<>)))
            {
                Type fromElementType = fromType.GetElementType();
                Type toElementType = toType.GetGenericArguments()[0];
                bool ok = IsAssignableFrom(toElementType, fromElementType, out int elementSubclassCount);
                subclassCount += elementSubclassCount;
                return ok;
            }
            if (toType.IsAssignableFrom(fromType)) return true;
            return false;
        }

        /// <summary>
        /// Get a type converter.
        /// </summary>
        /// <param name="fromType">non-null.  May contain type parameters. Use typeof(Nullable) to convert from a null value.</param>
        /// <param name="toType">non-null.  May contain type parameters. May be typeof(void), for which no conversion is needed.</param>
        /// <param name="info"></param>
        /// <returns>null if no converter was found.</returns>
        public static bool TryGetConversion(Type fromType, Type toType, out Conversion info)
        {
            info = new Conversion();
            if (fromType == typeof (Nullable)) return IsNullable(toType);
            if (IsAssignableFrom(toType, fromType, out int subclassCount))
            {
                // toType is a superclass or an interface of fromType
                info.SubclassCount = subclassCount;
                return true;
            }
            if (toType == typeof (void))
            {
                return true;
            }
            if (fromType.Equals(typeof (object)))
            {
                info.IsExplicit = true;
                info.Converter = delegate(object value) { return ChangeType(value, toType); };
                return true;
            }
            // string -> enum conversion
            if (toType.IsEnum && fromType.Equals(typeof (string)))
            {
                info.IsExplicit = true;
                info.Converter = delegate(object fromString) { return Enum.Parse(toType, (string) fromString); };
                return true;
            }
            if (typeof (Delegate).IsAssignableFrom(toType))
            {
                // DelegateGroup or ComCallback -> Delegate conversion
                if (typeof(CanGetDelegate).IsAssignableFrom(fromType))
                {
                    info.IsExplicit = true;
                    info.Converter = delegate(object dg)
                        {
                            Delegate result = ((CanGetDelegate) dg).GetDelegate(toType);
                            if (result == null) throw new ArgumentException(String.Format("The {0} has no match for the signature of {1}", fromType.ToString(), toType.ToString()));
                            return result;
                        };
                    return true;
                }
            }
            // Matlab array up-conversion
            if (toType.IsArray)
            {
                int toRank = toType.GetArrayRank();
                Type toElementType = toType.GetElementType();
                if (fromType.IsArray)
                {
                    int fromRank = fromType.GetArrayRank();
                    Type fromElementType = fromType.GetElementType();
                    Conversion elementConversion;
                    if (!TryGetConversion(fromElementType, toElementType, out elementConversion))
                        return false;
                    return TryGetArrayConversion(fromRank, toRank, toElementType, elementConversion, out info);
                }
                else if (fromType.Equals(typeof (System.Reflection.Missing)))
                {
                    // convert to zero-length array
                    int[] lengths = new int[toRank];
                    for (int i = 0; i < toRank; i++) lengths[i] = 0;
                    info.SubclassCount = 1;
                    info.Converter = delegate(object missing) { return Array.CreateInstance(toElementType, lengths); };
                    return true;
                }
                else
                {
                    // convert a scalar to an array of given rank
                    Conversion elementConversion;
                    if (!TryGetConversion(fromType, toElementType, out elementConversion))
                        return false;
                    return TryGetArrayConversion(0, toRank, toElementType, elementConversion, out info);
                }
            }
            // check for custom conversions
            MemberInfo[] implicitsOnFromType = fromType.FindMembers(MemberTypes.Method, BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, Type.FilterName,
                                                          "op_Implicit");
            foreach (MemberInfo member in implicitsOnFromType)
            {
                MethodInfo method = (MethodInfo) member;
                if (method.ReturnType == toType)
                {
                    info.SubclassCount = OpImplicitSubclassCount;
                    info.Converter = delegate (object value)
                    {
                        return Util.Invoke(method, null, value);
                    };
                    return true;
                }
            }
            MemberInfo[] implicitsOnToType = toType.FindMembers(MemberTypes.Method, BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, Type.FilterName,
                                                          "op_Implicit");
            foreach (MemberInfo member in implicitsOnToType)
            {
                MethodInfo method = (MethodInfo)member;
                ParameterInfo[] parameters = method.GetParameters();
                if (parameters.Length == 1 && parameters[0].ParameterType == fromType)
                {
                    info.SubclassCount = OpImplicitSubclassCount;
                    info.Converter = delegate (object value)
                    {
                        return Util.Invoke(method, null, value);
                    };
                    return true;
                }
            }
            MemberInfo[] explicitsOnFromType = fromType.FindMembers(MemberTypes.Method, BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, Type.FilterName,
                                                          "op_Explicit");
            foreach (MemberInfo member in explicitsOnFromType)
            {
                MethodInfo method = (MethodInfo) member;
                if (method.ReturnType == toType)
                {
                    info.IsExplicit = true;
                    info.Converter = delegate (object value)
                    {
                        return Util.Invoke(method, null, value);
                    };
                    return true;
                }
            }
            MemberInfo[] explicitsOnToType = toType.FindMembers(MemberTypes.Method, BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, Type.FilterName,
                                                          "op_Explicit");

            foreach (MemberInfo member in explicitsOnToType)
            {
                MethodInfo method = (MethodInfo)member;
                ParameterInfo[] parameters = method.GetParameters();
                if (parameters.Length == 1 && parameters[0].ParameterType == fromType)
                {
                    info.IsExplicit = true;
                    info.Converter = delegate (object value)
                    {
                        return Util.Invoke(method, null, value);
                    };
                    return true;
                }
            }
            // lastly try the IConvertible interface
            if (toType.IsPrimitive)
            {
                return TryGetPrimitiveConversion(fromType, toType, out info);
            }
            return false;
        }

        public static object ChangeType(object value, Type toType)
        {
            Conversion info;
            Type type = value.GetType();
            // prevent an infinite loop
            if (type.Equals(typeof (object)))
            {
                throw new ArgumentException("Cannot convert from " + type.Name + " to " + toType.Name);
            }
            if (!TryGetConversion(type, toType, out info))
            {
                throw new ArgumentException("Cannot convert from " + type.Name + " to " + toType.Name);
            }
            Converter c = info.Converter;
            if (c != null) value = c(value);
            return value;
        }

        public static bool TryGetArrayConversion(int fromRank, int toRank, Type toElementType, Conversion elementConversion, out Conversion info)
        {
            Converter conv = elementConversion.Converter;
            info = elementConversion; // assumes info is a struct
            if (fromRank == 0)
            {
                // convert a scalar to an array of given rank
                int[] lengths = new int[toRank];
                for (int i = 0; i < toRank; i++) lengths[i] = 1;
                int[] index = new int[toRank];
                info.SubclassCount = ChangeRankSubclassCount;
                info.Converter = delegate(object item)
                    {
                        object value = item;
                        if (conv != null) value = conv(item);
                        Array a = Array.CreateInstance(toElementType, lengths);
                        a.SetValue(value, index);
                        return a;
                    };
                return true;
            }
            else if (toRank == fromRank)
            {
                if (conv == null) return true;
                info.Converter = delegate(object fromArray) { return ChangeRank((Array) fromArray, toRank, toElementType, conv); };
                return true;
            }
            else if (toRank == 1 || fromRank == 1)
            {
                info.SubclassCount = ChangeRankSubclassCount;
                info.Converter = delegate(object fromArray) { return ChangeRank((Array) fromArray, toRank, toElementType, conv); };
                return true;
            }
            return false;
        }

        private class ConvertedDelegate
        {
            public readonly Delegate InnerDelegate;
            public readonly Converter Converter;

            public ConvertedDelegate(Delegate inner, Converter converter)
            {
                this.InnerDelegate = inner;
                this.Converter = converter;
            }
        }

        /// <summary>
        /// Convert a weakly-typed delegate into a strongly-typed delegate.
        /// </summary>
        /// <param name="delegateType">The desired delegate type.</param>
        /// <param name="inner">A delegate with parameters (object[] args).
        /// The
        /// return type can be any type convertible to the return type of delegateType, or void if
        /// the delegateType is void.</param>
        /// <returns>A delegate of type delegateType.  The arguments of this delegate will be
        /// passed as (object[]) args to the innerMethod.</returns>
        public static Delegate ConvertDelegate(Type delegateType, Delegate inner)
        {
            // This code is based on:
            // http://blogs.msdn.com/joelpob/archive/2005/07/01/434728.aspx
            object target = inner;
            string methodName = "DynamicMethod"; //delegateType.ToString();
            MethodInfo signature = delegateType.GetMethod("Invoke");
            Type returnType = signature.ReturnType;
            Type innerReturnType = inner.Method.ReturnType;
            Conversion conv;
            if (!TryGetConversion(innerReturnType, returnType, out conv))
            {
                throw new ArgumentException("Return type of the innerMethod (" + innerReturnType.Name + ") cannot be converted to the delegate return type (" + returnType.Name +
                                            ")");
            }
            Converter c = conv.Converter;
            if (c != null)
            {
                target = new ConvertedDelegate(inner, c);
            }
            Type[] formals = Invoker.GetParameterTypes(signature);
            Type[] formalsWithTarget = formals;
            if (target != null)
            {
                formalsWithTarget = new Type[1 + formals.Length];
                formalsWithTarget[0] = target.GetType();
                formals.CopyTo(formalsWithTarget, 1);
            }
            DynamicMethod method = new DynamicMethod(methodName, returnType, formalsWithTarget, typeof (Conversion));
            ILGenerator il = method.GetILGenerator();
            // put the delegate parameters into an object[]
            LocalBuilder args = il.DeclareLocal(typeof (object[]));
            il.Emit(OpCodes.Ldc_I4, formals.Length);
            il.Emit(OpCodes.Newarr, typeof (object));
            il.Emit(OpCodes.Stloc, args);
            int offset = (target == null) ? 0 : 1;
            for (int i = 0; i < formals.Length; i++)
            {
                // args[i] = (arg i+1)
                il.Emit(OpCodes.Ldloc, args);
                il.Emit(OpCodes.Ldc_I4, i);
                il.Emit(OpCodes.Ldarg, i + offset);
                // box if necessary
                if (formals[i].IsValueType)
                {
                    il.Emit(OpCodes.Box, formals[i]);
                }
                il.Emit(OpCodes.Stelem_Ref);
            }
            // push the result converter on the stack
            if (c != null)
            {
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldfld, typeof (ConvertedDelegate).GetField("Converter"));
            }
            // call the inner delegate
            il.Emit(OpCodes.Ldarg_0);
            if (c != null)
            {
                il.Emit(OpCodes.Ldfld, typeof (ConvertedDelegate).GetField("InnerDelegate"));
            }
            il.Emit(OpCodes.Ldloc, args);
            il.Emit(OpCodes.Call, inner.GetType().GetMethod("Invoke"));
            // handle the return value
            if (innerReturnType != typeof (void))
            {
                if (returnType == typeof (void))
                {
                    il.Emit(OpCodes.Pop);
                }
                else
                {
                    // converter object is already on the stack
                    // calling c.Method directly does not work (access exception)
                    //il.Emit(OpCodes.Call, c.Method);
                    il.Emit(OpCodes.Call, c.GetType().GetMethod("Invoke"));
                    // Converter always returns object, so unbox if necessary
                    if (returnType.IsValueType)
                    {
                        il.Emit(OpCodes.Unbox_Any, returnType);
                    }
                }
            }
            il.Emit(OpCodes.Ret);
            return method.CreateDelegate(delegateType, target);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}