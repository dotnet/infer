// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections;
using System.Globalization;
using System.Text;
using System.Reflection;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Utilities
{
    /// <summary>
    /// Helpful methods for converting objects to strings.
    /// </summary>
    public static class StringUtil
    {
        public static string TypeToXmlString(Type type)
        {
            if (type == typeof (bool)) return "bool";
            else if (type == typeof (int)) return "int";
            else if (type == typeof (float)) return "float";
            else if (type == typeof (double)) return "double";
            string prefix = "";
            if (type.IsNested)
            {
                prefix = TypeToXmlString(type.DeclaringType) + ".";
            }
            if (type.IsGenericType)
            {
                StringBuilder s = new StringBuilder(prefix);
                int quotePos = type.Name.IndexOf('`');
                if (quotePos == -1)
                {
                    s.Append(type.Name);
                }
                else
                {
                    s.Append(type.Name.Remove(quotePos));
                }
                Type[] typeArguments = type.GetGenericArguments();
                if (
                    type.IsNested &&
                    type.DeclaringType.IsGenericType)
                {
                    // omit type parameters of the declaring type.
                    int parentTypeParameterCount = type.DeclaringType.GetGenericArguments().Length;
                    if (typeArguments.Length == parentTypeParameterCount)
                    {
                        // type is a non-generic nested type of a generic type.
                        return s.ToString();
                    }
                    Type[] nestedTypeArguments = new Type[typeArguments.Length - parentTypeParameterCount];
                    Array.Copy(typeArguments, parentTypeParameterCount, nestedTypeArguments, 0, nestedTypeArguments.Length);
                    typeArguments = nestedTypeArguments;
                }
                s.Append("_");
                AppendXmlTypes(s, typeArguments);
                s.Append("_");
                return s.ToString();
            }
            else if (type.IsArray)
            {
                string brackets = "[";
                int rank = type.GetArrayRank();
                for (int i = 1; i < rank; i++)
                {
                    brackets += ",";
                }
                brackets += "]";
                string s = TypeToXmlString(type.GetElementType());
                //if(startIndex == -1) startIndex = 0;
                // If the element type is an array, we must insert the outer brackets before the brackets
                // in the element type.  For example, the type "int[,,][]" has element type "int[]", so we
                // must insert "[,,]" in the middle.
                // Find the first bracket after the generic arguments and declaring types
                int startIndex = System.Math.Max(s.LastIndexOf('>'), s.LastIndexOf('.')) + 1;
                int pos = s.IndexOf('[', startIndex);
                if (pos == -1) return s + brackets;
                else return prefix + s.Substring(0, pos) + brackets + s.Substring(pos, s.Length - pos);
            }
            else return prefix + type.Name;
        }

        public static void AppendXmlTypes(StringBuilder s, Type[] types)
        {
            bool first = true;
            foreach (Type t in types)
            {
                if (!first) s.Append(",");
                else first = false;
                if (t != null) s.Append(TypeToXmlString(t));
            }
        }

        public static string GenericParameterAttributesToString(GenericParameterAttributes attributes)
        {
            StringBuilder s = new StringBuilder();
            bool addComma = false;
            if ((attributes & GenericParameterAttributes.NotNullableValueTypeConstraint) != 0)
            {
                if (addComma) s.Append(',');
                s.Append("struct");
                addComma = true;
            }
            if ((attributes & GenericParameterAttributes.ReferenceTypeConstraint) != 0)
            {
                if (addComma) s.Append(',');
                s.Append("class");
                addComma = true;
            }
            if ((attributes & GenericParameterAttributes.DefaultConstructorConstraint) != 0)
            {
                if (addComma) s.Append(',');
                s.Append("new()");
                addComma = true;
            }
            return s.ToString();
        }

        /// <summary>
        /// Get a C# style string describing a .NET type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string TypeToString(Type type)
        {
            return TypeToString(type, false);
        }

        public static string TypeToString(Type type, bool showConstraints)
        {
            return TypeToString(type, showConstraints, null);
        }

        public static string TypeToString(Type type, bool showConstraints, Set<Type> constrained)
        {
            if (type == typeof (bool)) return "bool";
            else if (type == typeof (int)) return "int";
            else if (type == typeof (float)) return "float";
            else if (type == typeof (double)) return "double";
            if (type.IsGenericParameter)
            {
                if (!showConstraints || (constrained != null && constrained.Contains(type))) return type.Name;
                Type[] constraints = type.GetGenericParameterConstraints();
                if (constraints.Length == 0) return type.Name;
                StringBuilder s = new StringBuilder(type.Name);
                s.Append("{");
                string attrStr = GenericParameterAttributesToString(type.GenericParameterAttributes);
                s.Append(attrStr);
                bool addComma = (attrStr.Length > 0);
                Set<Type> constrained2 = constrained;
                if (constrained2 == null) constrained2 = new Set<Type>();
                constrained2.Add(type);
                foreach (Type constraint in constraints)
                {
                    if (addComma) s.Append(',');
                    s.Append(TypeToString(constraint, showConstraints, constrained2));
                    addComma = true;
                }
                s.Append("}");
                return s.ToString();
            }
            string prefix = "";
            if (type.IsNested)
            {
                prefix = TypeToString(type.DeclaringType) + ".";
            }
            if (type.IsGenericType)
            {
                StringBuilder s = new StringBuilder(prefix);
                int quotePos = type.Name.IndexOf('`');
                if (quotePos == -1)
                {
                    s.Append(type.Name);
                }
                else
                {
                    s.Append(type.Name.Remove(quotePos));
                }
                Type[] typeArguments = type.GetGenericArguments();
                if (
                    type.IsNested &&
                    type.DeclaringType.IsGenericType)
                {
                    // omit type parameters of the declaring type.
                    int parentTypeParameterCount = type.DeclaringType.GetGenericArguments().Length;
                    if (typeArguments.Length == parentTypeParameterCount)
                    {
                        // type is a non-generic nested type of a generic type.
                        return s.ToString();
                    }
                    Type[] nestedTypeArguments = new Type[typeArguments.Length - parentTypeParameterCount];
                    Array.Copy(typeArguments, parentTypeParameterCount, nestedTypeArguments, 0, nestedTypeArguments.Length);
                    typeArguments = nestedTypeArguments;
                }
                s.Append("<");
                Set<Type> constrained2 = constrained;
                if (constrained2 == null) constrained2 = new Set<Type>();
                constrained2.Add(type);
                AppendTypes(s, typeArguments, showConstraints, constrained2);
                s.Append(">");
                return s.ToString();
            }
            else if (type.IsArray)
            {
                string brackets = "[";
                int rank = type.GetArrayRank();
                for (int i = 1; i < rank; i++)
                {
                    brackets += ",";
                }
                brackets += "]";
                string s = TypeToString(type.GetElementType(), showConstraints, constrained);
                //if(startIndex == -1) startIndex = 0;
                // If the element type is an array, we must insert the outer brackets before the brackets
                // in the element type.  For example, the type "int[,,][]" has element type "int[]", so we
                // must insert "[,,]" in the middle.
                // Find the first bracket after the generic arguments and declaring types
                int startIndex = System.Math.Max(s.LastIndexOf('>'), s.LastIndexOf('.')) + 1;
                int pos = s.IndexOf('[', startIndex);
                if (pos == -1) return s + brackets;
                else return prefix + s.Substring(0, pos) + brackets + s.Substring(pos, s.Length - pos);
            }
            else return prefix + type.Name;
        }

        /// <summary>
        /// Append type strings to a StringBuilder
        /// </summary>
        /// <param name="s"></param>
        /// <param name="types"></param>
        public static void AppendTypes(StringBuilder s, Type[] types)
        {
            AppendTypes(s, types, false, null);
        }

        /// <summary>
        /// Append type strings to a StringBuilder
        /// </summary>
        /// <param name="s"></param>
        /// <param name="types"></param>
        /// <param name="showConstraints"></param>
        /// <param name="constrained"></param>
        public static void AppendTypes(StringBuilder s, Type[] types, bool showConstraints, Set<Type> constrained)
        {
            bool first = true;
            foreach (Type t in types)
            {
                if (!first) s.Append(", ");
                else first = false;
                if (t != null) s.Append(TypeToString(t, showConstraints, constrained));
            }
        }

        /// <summary>
        /// Get a string of the form "typeName.methodName&amp;lt;types&amp;gt;", suitable
        /// for use as an XML element value.
        /// </summary>
        /// <param name="text">A string.</param>
        /// <returns>A valid XML element value.</returns>
        public static string EscapeXmlCharacters(string text)
        {
            return text.Replace("&", "&amp;")
                       .Replace("<", "&lt;")
                       .Replace(">", "&gt;");
        }

        /// <summary>
        /// Get a string of the form "typeName.methodName&lt;types&gt;".
        /// </summary>
        /// <param name="method"></param>
        /// <returns></returns>
        public static string MethodFullNameToString(MethodBase method)
        {
            return TypeToString(method.DeclaringType) + "." + MethodNameToString(method);
        }

        /// <summary>
        /// Get a string of the form "methodName&lt;types&gt;"
        /// </summary>
        /// <param name="method"></param>
        /// <returns></returns>
        public static string MethodNameToString(MethodBase method)
        {
            if (method.IsGenericMethod)
            {
                StringBuilder s = new StringBuilder(method.Name);
                s.Append("<");
                Type[] typeArguments = method.GetGenericArguments();
                AppendTypes(s, typeArguments);
                s.Append(">");
                return s.ToString();
            }
            else
            {
                return method.Name;
            }
        }

        /// <summary>
        /// Get a short string describing the signature of a method.
        /// </summary>
        /// <param name="method">The method to get the signature for.</param>
        /// <param name="useFullName">Specifies whether the name of the method should be prepended with the name of the class.</param>
        /// <param name="omitParameterNames">Specifies whether the parameter names should be omitted from the result.</param>
        /// <returns>A string of the form "methodName&lt;types&gt;(parameters)"</returns>
        /// <remarks>From the C# 3.0 specification sec 1.6.6: 
        /// The signature of a method consists of the name of the method, 
        /// the number of type parameters and the number, modifiers, and types of its parameters. 
        /// The signature of a method does not include the return type.</remarks>
        public static string MethodSignatureToString(MethodInfo method, bool useFullName = true, bool omitParameterNames = false)
        {
            StringBuilder s = new StringBuilder();
            s.Append(useFullName ? MethodFullNameToString(method) : MethodNameToString(method));
            s.Append("(");
            ParameterInfo[] parameters = method.GetParameters();
            AppendParameters(s, parameters, omitParameterNames);
            s.Append(")");
            return s.ToString();
        }

        public static void AppendParameters(StringBuilder s, IEnumerable<ParameterInfo> parameters, bool omitParameterNames = false)
        {
            bool firstTime = true;
            foreach (ParameterInfo parameter in parameters)
            {
                if (!firstTime)
                {
                    s.Append(", ");
                }

                string parameterTypeString;
                if (parameter.ParameterType.IsByRef)
                {
                    parameterTypeString = string.Format(
                        "{0} {1}", parameter.IsOut ? "out" : "ref", TypeToString(parameter.ParameterType.GetElementType()));
                }
                else
                {
                    parameterTypeString = TypeToString(parameter.ParameterType);
                }
                
                if (omitParameterNames)
                {
                    s.Append(parameterTypeString);
                }
                else
                {
                    s.AppendFormat("{0} {1}", parameterTypeString, parameter.Name);
                }

                firstTime = false;
            }
        }

        /// <summary>
        /// Get a string of list elements separated by a delimiter
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public static string CollectionToString<T>(IEnumerable<T> list, string delimiter)
        {
            if (list == null) return "";
            StringBuilder s = new StringBuilder();
            int i = 0;
            foreach (T item in list)
            {
                if (i > 0) s.Append(delimiter);
                //string rhs = item.ToString();
                string rhs = ToString(item);
                s.Append(rhs);
                i++;
            }
            return s.ToString();
        }

        /// <summary>
        /// Get a string listing all elements of an array on a separate line
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static string ArrayToString(Array a)
        {
            int[] dims = ArrayDimensions(a);
            int[] strides = ArrayStrides(dims);
            int[] lowerBounds = ArrayLowerBounds(a);
            return ArrayToString(a, strides, lowerBounds);
        }

        /// <summary>
        /// Get a string listing all elements of an array, one per line
        /// </summary>
        /// <param name="a"></param>
        /// <param name="strides"></param>
        /// <param name="lowerBounds"></param>
        /// <returns></returns>
        public static string ArrayToString(System.Collections.IEnumerable a, int[] strides, int[] lowerBounds)
        {
            int[] index = new int[strides.Length];
            StringBuilder s = new StringBuilder();
            int i = 0;
            foreach (object item in a)
            {
                if (i > 0) s.Append(Environment.NewLine);
                LinearIndexToMultidimensionalIndex(i++, strides, index, lowerBounds);
                string lhs = "[" + CollectionToString(index, ",") + "] ";
                string rhs = ToString(item);
                s.Append(JoinColumns(lhs, rhs));
            }
            if (i == 0) return "[]"; //"empty " + a.ToString();
            return s.ToString();
        }

        /// <summary>
        /// Get a verbose string describing an object, or invoke the object's custom ToString method if it exists
        /// </summary>
        /// <param name="o"></param>
        /// <returns></returns>
        public static string ToString(object o)
        {
            if (o == null) return "null";
            if (o is Type type1)
            {
                return TypeToString(type1);
            }
            else
            {
                Type type = o.GetType();
                if (type.IsArray)
                {
                    return ArrayToString((Array) o);
                }
                else
                {
                    MethodInfo method = type.GetMethod("ToString", Type.EmptyTypes);
                    if (method.DeclaringType == typeof (object))
                    {
                        // instead of calling object.ToString, provide more information
                        return VerboseToString(o);
                    }
                    else return o.ToString();
                }
            }
        }

        /// <summary>
        /// Get a string describing the contents of an object by enumerating its items and properties.
        /// </summary>
        /// <param name="o"></param>
        /// <returns></returns>
        public static string VerboseToString(object o)
        {
            // if the object is a dictionary, print it like one.
            // if the object is enumerable, print it like an array.
            // otherwise print its properties.
            if (o is IDictionary dictionary)
            {
                return DictionaryToString(dictionary, Environment.NewLine);
            }
            else if (o is IEnumerable enumerable)
            {
                return EnumerableToString(enumerable, Environment.NewLine);
            } 
            else if (IsIndexable(o))
            {
                return IndexerToString(o, Environment.NewLine);
            } 
            else return PropertiesToString(o, Environment.NewLine, TypeToString(o.GetType()));
        }

        /// <summary>
        /// Get a string listing the entries of a dictionary, one per line.
        /// </summary>
        /// <typeparam name="KeyType"></typeparam>
        /// <typeparam name="ValueType"></typeparam>
        /// <param name="dict"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public static string DictionaryToString<KeyType, ValueType>(IEnumerable<KeyValuePair<KeyType, ValueType>> dict, string delimiter)
        {
            StringBuilder s = new StringBuilder();
            int i = 0;
            foreach (KeyValuePair<KeyType, ValueType> entry in dict)
            {
                string lhs = "[" + entry.Key + "] ";
                string rhs = ToString(entry.Value);
                if (i > 0) s.Append(delimiter);
                s.Append(JoinColumns(lhs, rhs));
                i++;
            }
            if (i == 0) return "{}";
            return s.ToString();
        }

        /// <summary>
        /// Get a string listing the entries of a dictionary, one per line.
        /// </summary>
        /// <param name="dict"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public static string DictionaryToString(IDictionary dict, string delimiter)
        {
            StringBuilder s = new StringBuilder();
            int i = 0;
            foreach (DictionaryEntry entry in dict)
            {
                string lhs = "[" + entry.Key + "] ";
                string rhs = ToString(entry.Value);
                if (i > 0) s.Append(delimiter);
                s.Append(JoinColumns(lhs, rhs));
                i++;
            }
            if (i == 0) return "{}";
            return s.ToString();
        }

        /// <summary>
        /// Get a string listing the elements of an enumerable, one per line.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public static string EnumerableToString(IEnumerable a, string delimiter)
        {
            StringBuilder s = new StringBuilder();
            int i = 0;
            foreach (object item in a)
            {
                string lhs = "[" + i.ToString(CultureInfo.InvariantCulture) + "] ";
                string rhs = ToString(item);
                if (i > 0) s.Append(delimiter);
                s.Append(JoinColumns(lhs, rhs));
                i++;
            }
            if (i == 0) return "[]"; //"empty " + a.ToString();
            return s.ToString();
        }

        public static bool IsIndexable(object o)
        {
            Type[] types = {typeof (int)};
            try
            {
                Type type = o.GetType();
                PropertyInfo prop = type.GetProperty("Item", types);
                return (prop != null && prop.CanRead);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Get a string listing the elements of an indexable object, one per line
        /// </summary>
        /// <param name="o"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public static string IndexerToString(object o, string delimiter)
        {
            Type[] types = {typeof (int)};
            Type type = o.GetType();
            PropertyInfo prop = type.GetProperty("Item", types);
            object[] index = {0};
            StringBuilder s = new StringBuilder();
            int i = 0;
            while (true)
            {
                index[0] = i;
                try
                {
                    string rhs = ToString(prop.GetValue(o, index));
                    string lhs = "[" + i.ToString(CultureInfo.InvariantCulture) + "] ";
                    if (i > 0) s.Append(delimiter);
                    s.Append(JoinColumns(lhs, rhs));
                }
                catch
                {
                    break;
                }
                i++;
            }
            if (i == 0) return "[]"; //"empty " + o.ToString();
            return s.ToString();
        }

        public static string PropertiesToString(object o, string delimiter, string initial)
        {
            StringBuilder s = new StringBuilder(initial);
            Type type = o.GetType();
            int i = (initial.Length == 0) ? 0 : 1;
            FieldInfo[] fields = type.GetFields();
            foreach (FieldInfo field in fields)
            {
                try
                {
                    //string rhs = ToString(field.GetValue(o));
                    string rhs = field.GetValue(o).ToString();
                    if (i > 0) s.Append(delimiter);
                    s.Append(JoinColumns(field.Name, " = ", rhs));
                    i++;
                }
                catch
                {
                }
            }
            PropertyInfo[] props = type.GetProperties();
            foreach (PropertyInfo prop in props)
            {
                if (prop.CanRead && prop.GetIndexParameters().Length == 0)
                {
                    try
                    {
                        //string rhs = ToString(prop.GetValue(o, null));
                        string rhs = prop.GetValue(o, null).ToString();
                        if (i > 0) s.Append(delimiter);
                        s.Append(JoinColumns(prop.Name, " = ", rhs));
                        i++;
                    }
                    catch
                    {
                    }
                }
            }
            return s.ToString();
        }

        public static int Sum(ICollection<bool> list)
        {
            int sum = 0;
            foreach (bool item in list)
            {
                if (item) sum++;
            }
            return sum;
        }

        public static string[] Lines(string text)
        {
            return text.Split(new string[] {Environment.NewLine}, StringSplitOptions.None);
        }

        public static ICollection<int> StringLengths(ICollection<string> stringList)
        {
            int[] lengths = new int[stringList.Count];
            int i = 0;
            foreach (string line in stringList)
            {
                lengths[i++] = line.Length;
            }
            return lengths;
        }

        public static ICollection<int> ArrayLengths(ICollection<Array> arrayList)
        {
            int[] lengths = new int[arrayList.Count];
            int i = 0;
            foreach (Array a in arrayList)
            {
                lengths[i++] = a.Length;
            }
            return lengths;
        }

        public static int Max(ICollection<int> list)
        {
            int max = Int32.MinValue;
            foreach (int item in list)
            {
                max = System.Math.Max(max, item);
            }
            return max;
        }

        public static string JoinColumns(params object[] columns)
        {
            int ncols = columns.Length;
            string[][] lines = new string[ncols][];
            for (int i = 0; i < ncols; i++)
            {
                lines[i] = Lines(columns[i] == null ? "" : columns[i].ToString());
            }
            return JoinColumns(lines);
        }

        /// <summary>
        /// Create a string denoting a multi-line table with multiple columns.
        /// </summary>
        /// <param name="lines">lines[column][line] is a single text line in the column.</param>
        /// <returns></returns>
        public static string JoinColumns(params string[][] lines)
        {
            int ncols = lines.Length;
            // Determine the width of each column by taking a maximum over all lines.
            int[] widths = new int[ncols];
            for (int i = 0; i < ncols; i++)
            {
                widths[i] = Max(StringLengths(lines[i]));
            }
            // Determine the total number of rows in the table by taking a maximum over all columns.
#if false
            int nrows = Max(ArrayLengths(lines));
#else
            int[] lengths = new int[lines.Length];
            int count = 0;
            foreach (Array a in lines)
            {
                lengths[count++] = a.Length;
            }
            int nrows = Max(lengths);
#endif

            StringBuilder s = new StringBuilder();
            for (int i = 0; i < nrows; i++)
            {
                if (i > 0) s.AppendLine();
                for (int j = 0; j < ncols; j++)
                {
                    string line = (i < lines[j].Length) ? lines[j][i] : "";
                    if (j < ncols - 1) line = line.PadRight(widths[j]);
                    s.Append(line);
                }
            }
            return s.ToString();
        }

        public static int[] ArrayDimensions(Array a)
        {
            int[] dims = new int[a.Rank];
            for (int d = 0; d < dims.Length; d++) dims[d] = a.GetLength(d);
            return dims;
        }

        public static int[] ArrayLowerBounds(Array a)
        {
            return ArrayLowerBounds(a, out bool allZero);
        }

        public static int[] ArrayLowerBounds(Array a, out bool allZero)
        {
            allZero = true;
            int[] lowerBounds = new int[a.Rank];
            for (int d = 0; d < a.Rank; d++)
            {
                lowerBounds[d] = a.GetLowerBound(d);
                if (lowerBounds[d] != 0) allZero = false;
            }
            return lowerBounds;
        }

        public static int[] ArrayStrides(IList<int> dims)
        {
            int[] strides = new int[dims.Count];
            SetToArrayStrides(strides, dims, 1);
            return strides;
        }

        public static void SetToArrayStrides(IList<int> strides, IList<int> dims, int baseStride)
        {
            strides[dims.Count - 1] = baseStride;
            for (int d = dims.Count - 2; d >= 0; d--)
            {
                strides[d] = dims[d + 1]*strides[d + 1];
            }
        }

        public static void LinearIndexToMultidimensionalIndex(int index, int[] strides, int[] mIndex)
        {
            for (int d = 0; d < strides.Length; d++)
            {
                mIndex[d] = index/strides[d];
                index = index%strides[d];
            }
        }

        public static void LinearIndexToMultidimensionalIndex(int index, int[] strides, int[] mIndex, int[] lowerBounds)
        {
            for (int d = 0; d < strides.Length; d++)
            {
                mIndex[d] = index/strides[d] + lowerBounds[d];
                index = index%strides[d];
            }
        }

        public static int MultidimensionalIndexToLinearIndex(int[] index, int[] strides)
        {
            int i = 0;
            for (int d = 0; d < strides.Length; d++)
            {
                i += index[d]*strides[d];
            }
            return i;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#endif
}