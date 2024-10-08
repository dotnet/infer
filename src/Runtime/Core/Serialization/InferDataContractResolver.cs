// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Reflection;
    using System.Runtime.Serialization;
    using System.Xml;

    /// <summary>
    /// A data contract resolver that dynamically resolves known types.
    /// </summary>
    public class InferDataContractResolver : DataContractResolver
    {
        // URN prefix of namespaces generated by our contract resolver
        private const string nsPrefix = "urn:x-research.microsoft.com:Infer.NET:assemblyname:";

        private static Dictionary<string, Type> allowedTypes = GetAllowedTypes();

        private static Dictionary<string, Type> GetAllowedTypes()
        {
            // If this list isn't long enough then users can add to KnownTypes any types
            // they need. The exception thrown is very informative regarding which types are
            // allowed.
            // There are two KnownType mechanisms, the KnownTypeAttribute that can be placed
            // on types which declare the types as known when deserialising the given type.
            // The other KnownType mechanism is to explicitly add types to the
            // DataContractSerializer.KnownTypes collection, which whitelists the type for all
            // deserialization using that object.
            // AllowedType should only have open generic types or non-generic types (it should
            // not contain closed generic types).
            var seedAllowedTypes = new HashSet<Type>
            {
                typeof(Char),
                typeof(SortedList<,>),
                typeof(Double),
                typeof(String),
                typeof(Tuple<>),
                typeof(Tuple<,>),
                typeof(Tuple<,,>),
                typeof(Tuple<,,,>),
                typeof(Tuple<,,,,>),
                typeof(List<>),
                typeof(Stack<>),
                typeof(Int32),
                typeof(Int64),
                typeof(IEnumerable<>),
                typeof(Queue<>),
                typeof(IList<>),
                typeof(IReadOnlyList<>),
                typeof(Nullable<>),
                typeof(IEqualityComparer<>),
                typeof(IComparer<>),
                typeof(ICollection<>),
                typeof(Dictionary<,>),
                typeof(ValueTuple<,>),
                typeof(ValueTuple<,,,>),
                typeof(IReadOnlyDictionary<,>),
            };

            var allowedTypes = new HashSet<Type>();
            foreach (var allowedType in seedAllowedTypes)
            {
                RecursivelyAddTypes(allowedType, allowedTypes);
            }

            // Allow all types in Runtime
            var runtimeTypes = Assembly
                .GetExecutingAssembly()
                .GetTypes();

            foreach (var runtimeType in runtimeTypes)
            {
                RecursivelyAddTypes(runtimeType, allowedTypes);
            }

            return allowedTypes.ToDictionary(x => x.FullName);
        }

        private static void RecursivelyAddTypes(Type type, HashSet<Type> types)
        {
            // Skip closed generic types.
            if (type.IsGenericType && !type.IsGenericTypeDefinition)
            {
                return;
            }

            // Skip generics with type parameters (these are types that appear
            // inside a generic type definition).
            if (type.ContainsGenericParameters && !type.IsGenericTypeDefinition)
            {
                return;
            }

            // Avoid repeatedly exploring the same types.
            if (!types.Add(type))
            {
                return;
            }

            if (type.BaseType != null)
            {
                RecursivelyAddTypes(type.BaseType, types);
            }

            foreach (var property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                // We remove "by ref" because we don't care about this for serialization.
                var typeToAdd = property.PropertyType;

                if (typeToAdd.IsByRef)
                {
                    typeToAdd = typeToAdd.GetElementType();
                }

                RecursivelyAddTypes(typeToAdd, types);
            }

            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                // In .NET8.0 we now have certain compiler generated fields without a proper type.
                if (field.FieldType.FullName == null)
                {
                    continue;
                }

                RecursivelyAddTypes(field.FieldType, types);
            }

            foreach (var nestedType in type.GetNestedTypes(BindingFlags.Public | BindingFlags.NonPublic))
            {
                RecursivelyAddTypes(nestedType, types);
            }
        }

        public override bool TryResolveType(
            Type type,
            Type declaredType,
            DataContractResolver knownTypeResolver,
            out XmlDictionaryString typeName,
            out XmlDictionaryString typeNamespace)
        {
            if (knownTypeResolver.TryResolveType(type, declaredType, knownTypeResolver, out typeName, out typeNamespace))
            {
                return true;
            }

            var dict = new XmlDictionary(2);
            typeName = dict.Add(type.FullName); 
            typeNamespace = dict.Add(String.Concat(nsPrefix, type.Assembly.FullName));
            return true;
        }

        public override Type ResolveName(string typeName, string typeNamespace, Type declaredType, DataContractResolver knownTypeResolver)
        {
            var ty = knownTypeResolver.ResolveName(typeName, typeNamespace, declaredType, knownTypeResolver);

            if (ty != null) return ty;
            
            if (!typeNamespace.StartsWith(nsPrefix)) // Is this our namespace?
            {
                return null; // Cannot resolve.
            }

            // We limit the length of the type name we are willing to try to parse
            // for security reasons.
            if (typeName.Length > 10000)
            {
                throw new Exception("Type name exceeded maximum length");
            }

            return ConstructTypeFromString(typeName);
        }

        internal static Type ConstructTypeFromString(string typeString)
        {
            // We avoid loading a type from a string for security reasons before we
            // first parse it to ensure that it is constructed from approved types.
            var typeLayout = ParseTypeString(typeString);

            // We now construct the types entirely from the allowed list, using the 
            // type layout to direct how we combine types in the allowed list.
            // This means that even if the data is tampered with, a type is never
            // loaded -- types are only constructed as combinations of types in the
            // allowed list.
            return ConstructTypeFromAllowedlist(typeLayout);
        }

        /// <summary>
        /// Some information about the layout of a type.
        /// </summary>
        internal sealed class ParsedTypeString
        {
            /// <summary>
            /// The name of the type (without array marker or argument list).
            /// </summary>
            public string Name { get; }

            /// <summary>
            /// Gets the type arguments.
            /// </summary>
            public ParsedTypeString[] Arguments { get; }

            /// <summary>
            /// Gets the layout of the array. For example,
            /// int has a layout of {}.
            /// int[][,,][][] has a layout of {1,3,1,1}
            /// </summary>
            public int[] ArrayLayout { get; }

            /// <summary>
            /// Initializes an instance of <see cref="ParsedTypeString"/>.
            /// </summary>
            /// <param name="name">The name of the type (without array marker or argument list).</param>
            /// <param name="arguments">Gets the type arguments.</param>
            /// <param name="arrayLayout">Gets tha array layout.</param>
            public ParsedTypeString(
                string name,
                ParsedTypeString[] arguments,
                int[] arrayLayout)
            {
                Name  = name;
                Arguments = arguments;
                ArrayLayout = arrayLayout;
            }
        }

        /// <summary>
        /// Read the type name extension (this is extra information beyond the type name,
        /// array marker, and argument list, such as the assembly information).
        /// </summary>
        /// <param name="typeString">The type string.</param>
        /// <param name="cursor">The index in the type string to read from.</param>
        /// <returns>
        /// If there is no extension, then cursor is returned; if there is an extension
        /// then one after the end of the extension is returned.
        /// </returns>
        private static int ReadTypeNameExtension(string typeString, int cursor)
        {
            if (cursor == typeString.Length)
            {
                return cursor;
            }

            // The start of the extension is marked with a comma.
            if (typeString[cursor] != ',')
            {
                return cursor;
            }

            while (true)
            {

                // If we reached the end of the string there is nothing following to read.
                if (typeString.Length == cursor)
                {
                    return cursor;
                }

                // If we reached the typename list item delimiter there is nothing following to read.
                if (typeString[cursor] == ']')
                {
                    return cursor;
                }

                cursor++;
            }
        }

        /// <summary>
        /// Read the array marker if it exists.
        /// </summary>
        /// <param name="typeString">The type string.</param>
        /// <param name="cursor">The index in the type string to read from.</param>
        /// <returns>
        /// cursor:
        /// If there is no array marker, then cursor is returned; if there is an array marker
        /// then one after the end of the extension is returned.
        /// 
        /// arrayLayout:
        /// the layout of the array. For example,
        /// int has a layout of {}.
        /// int[][,,][][] has a layout of {1,3,1,1}
        /// </returns>
        private static (int cursor, int[] arrayLayout) ReadArrayMarker(string typeString, int cursor)
        {
            // An array marker is a sequence of "[]"s for each array level.
            // Each contains a number of commas indicating the number of dimensions
            // in that array level.
            // For example, int[][,,][,] is a 2-dimensional array of
            // 3-dimensional arrays, of one dimensional arrays of ints.
            var arrayLayout = new List<int>();
            while (true)
            {
                int arrayDimension;
                (cursor, arrayDimension) = ReadNextSingularArrayMarker(typeString, cursor);

                if (arrayDimension == 0)
                {
                    return (cursor, arrayLayout.ToArray());
                }

                arrayLayout.Add(arrayDimension);
            }
        }

        /// <summary>
        /// Read the singular array marker if it exists.
        /// </summary>
        /// <param name="typeString">The type string.</param>
        /// <param name="cursor">The location to read from in the type string.</param>
        /// <returns>
        /// cursor:
        /// If there is no array marker, then cursor is returned; if there is an array marker
        /// then one after the end of the extension is returned.
        /// 
        /// arrayDimension:
        /// the number of dimensions the array marker indicates.
        /// For example,
        /// int has dimension 0
        /// int[,,,] has dimension 4.
        /// </returns>
        private static (int cursor, int arrayDimension) ReadNextSingularArrayMarker(string typeString, int cursor)
        {
            if (cursor == typeString.Length)
            {
                return (cursor, arrayDimension: 0);
            }

            if (typeString[cursor] != '[')
            {
                return (cursor, arrayDimension: 0);
            }

            if (cursor + 1 == typeString.Length)
            {
                throw new InvalidOperationException("Invalid type string");
            }

            // The number of commas indicates the number of dimensions.
            int commaSkip = 0;
            while (cursor + commaSkip + 1 < typeString.Length
                && typeString[cursor + commaSkip + 1] == ',')
            {
                commaSkip++;
            }

            if (typeString[cursor + commaSkip + 1] == ']')
            {
                return (cursor + commaSkip + 2, arrayDimension: commaSkip + 1);
            }

            return (cursor, arrayDimension: 0);
        }

        /// <summary>
        /// Read the type name.
        /// </summary>
        /// <param name="typeString">The type string.</param>
        /// <param name="cursor">The index in the type string to read from.</param>
        /// <returns>
        /// cursor:
        /// One after the last index of the current type name.
        /// 
        /// isArray:
        /// The TypeId of the read type.
        /// </returns>
        private static (int cursor, ParsedTypeString typeId) ReadTypeName(string typeString, int cursor)
        {
            // The format of a type name is
            // Name[PossibleArgumentList][PossibleArrayMarker][PossibleExtension]
            var startOfTypeName = cursor;

            // First read the actual type name.
            while (true)
            {
                if (typeString.Length < cursor)
                {
                    throw new InvalidOperationException("invalid type name.");
                }

                // If we reached the end of the string there is nothing following to read.
                if (typeString.Length == cursor)
                {
                    break;
                }

                if (typeString[cursor] == '[' || typeString[cursor] == ',')
                {
                    break;
                }

                cursor++;
            }

            var name = typeString.Substring(startOfTypeName, cursor - startOfTypeName);

            ParsedTypeString[] list;
            (cursor, list) = ReadTypeList(typeString, cursor);

            int[] arrayLayout;
            (cursor, arrayLayout) = ReadArrayMarker(typeString, cursor);

            cursor = ReadTypeNameExtension(typeString, cursor);
            return (cursor, new ParsedTypeString(name, list, arrayLayout));
        }

        /// <summary>
        /// Read the type name name list if it is present.
        /// </summary>
        /// <param name="typeString">The type string.</param>
        /// <param name="cursor">The index in the type string to read from.</param>
        /// <returns>
        /// cursor:
        /// One after the last index of the type name list if it exists.
        /// Returns the orignal cursor otherwise.
        /// 
        /// types:
        /// The TypeIds of the read types in the list.
        /// </returns>
        private static (int cursor, ParsedTypeString[] types) ReadTypeList(string typeString, int cursor)
        {
            // Type lists are delimited with square brackets, as is each type in the list, and they are separated by commas.
            // For example, X`4[[A],[B],[C],[D]]
            var list = new List<ParsedTypeString>();

            if (typeString.Length == cursor)
            {
                return (cursor, new ParsedTypeString[0]);
            }

            // We check for the marker that is used for type lists and arrays.
            if (typeString[cursor] != '[')
            {
                return (cursor, new ParsedTypeString[0]); 
            }

            // We verify that this is not a marker of a single dimensional array.
            if (typeString[cursor + 1] == ']')
            {
                return (cursor, new ParsedTypeString[0]);
            }

            // We verify that this is not a marker of a multi-dimensional array.
            // (After this, it must be a type list marker.)
            if (typeString[cursor + 1] == ',')
            {
                return (cursor, new ParsedTypeString[0]);
            }

            cursor++;

            while (true)
            {
                if (typeString[cursor] != '[')
                {
                    throw new InvalidOperationException("invalid type list element.");
                }
                cursor++;

                ParsedTypeString item;
                (cursor, item) = ReadTypeName(typeString, cursor);
                list.Add(item);

                if (typeString.Length < cursor)
                {
                    throw new InvalidOperationException("invalid cursor.");
                }

                if (typeString.Length == cursor)
                {
                    throw new InvalidOperationException("truncated type string.");
                }

                // Check for the closing bracket of the type string.
                if (typeString[cursor] != ']')
                {
                    throw new InvalidOperationException("invalid type string.");
                }

                cursor++;

                // there is not another item.
                if (typeString[cursor] != ',')
                {
                    break;
                }

                cursor++;
            }

            if (typeString[cursor] != ']')
            {
                throw new InvalidOperationException("invalid type list termination.");
            }
            cursor++;

            return (cursor, list.ToArray());
        }

        /// <summary>
        /// Parse a type string into a type layout which gives basic information
        /// about the makeup of the type.
        /// </summary>
        /// <param name="typeString">The type string.</param>
        /// <returns>
        /// Basic information about how the type is constructed from other types.
        /// </returns>
        internal static ParsedTypeString ParseTypeString(string typeString)
        {
            var (endOfString, typeId) = ReadTypeName(typeString, cursor: 0);
            if (endOfString != typeString.Length)
            {
                throw new InvalidOperationException("Invalid type string");
            }

            return typeId;
        }

        private static string ConstructHelpfulExceptionMessage(string typeName) =>
            $"Type '{typeName}' is not allowed. Whitelist it by adding it " +
            $"to {nameof(DataContractSerializer)}.{nameof(DataContractSerializer.KnownTypes)}" +
            $" on the serializer you are using; or by declaring it using" +
            $" {nameof(KnownTypeAttribute)} on the ultimate target type you are deserializing into.";

        internal static Type ConstructTypeFromAllowedlist(ParsedTypeString type)
        {
            // To avoid any possibility that an unexpected type is extracted from the string
            // this method creates the type entirely from the allowedTypes list. That way even
            // if there is an error in parsing of the type it is not possible for any type not
            // in the allowed list to be created.
            if (type.ArrayLayout.Length != 0)
            {
                var itemType = new ParsedTypeString(type.Name, type.Arguments, arrayLayout: new int[0]);
                var resolvedItemType = ConstructTypeFromAllowedlist(itemType);

                for (int i = 0; i < type.ArrayLayout.Length; i++)
                {
                    if (type.ArrayLayout[i] == 1)
                    {
                        // MakeArrayType(1) creates an array of type T[*] instead of
                        // the desired T[].
                        resolvedItemType = resolvedItemType.MakeArrayType();
                    }
                    else
                    {
                        resolvedItemType = resolvedItemType.MakeArrayType(type.ArrayLayout[i]);
                    }
                }

                return resolvedItemType;
            }

            if (type.Arguments.Length == 0)
            {
                if (!allowedTypes.TryGetValue(type.Name, out var simpleType))
                {
                    throw new Exception(ConstructHelpfulExceptionMessage(type.Name));
                }

                return simpleType;
            }

            var genericDefinitionName = type.Name;
            if (!allowedTypes.TryGetValue(genericDefinitionName, out var allowedListEntry))
            {
                throw new Exception(ConstructHelpfulExceptionMessage(type.Name));
            }

            if (type.Arguments.Length != allowedListEntry.GetGenericArguments().Length)
            {
                throw new Exception($"Invalid operation: allowed list entry '{allowedListEntry.FullName}' does not match typeId '{type}' with regard to the number of generic arguments.");
            }

            var genericParameters = type
                .Arguments
                .Select(x => ConstructTypeFromAllowedlist(x))
                .ToArray();

            return allowedListEntry.MakeGenericType(genericParameters);
        }
    }
}