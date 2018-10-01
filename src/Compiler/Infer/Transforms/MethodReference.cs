// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Used to look up a method by its declaring type, its name, and its type parameters (if any).
    /// </summary>
    public class MethodReference
    {
        public Type Type;
        public string MethodName;
        public int TypeParameterCount;
        public Type[] TypeArguments;
        public Type[] ParameterTypes;
        protected Exception LastMatchException;

#if SUPPRESS_AMBIGUOUS_REFERENCE_WARNINGS
#pragma warning disable 419
#endif

        /// <summary>
        /// Construct a method reference from a name and optional types.
        /// </summary>
        /// <param name="type">The type providing the method.</param>
        /// <param name="methodName">The name must specify the number of generic type parameters via the syntax <c>&lt;&gt;</c> or <c>&lt;,&gt;</c> etc.</param>
        /// <param name="parameterTypes">An array of types, which may contain null as a wildcard.</param>
        /// <remarks>
        /// If the method is generic with <c>n</c> type parameters, then the first <c>n</c> parameterTypes are interpreted
        /// as type arguments.  These can be null to allow arbitrary type arguments.
        /// Examples:
        /// <list type="bullet">
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"Copy",typeof(Array),typeof(Array),typeof(Int32))</c>
        /// </term><description>
        /// This selects the first overload of the non-generic method <see cref="Array.Copy(Array, Array, int)"/>.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"Copy",null,null,typeof(Int32))</c>
        /// </term><description>
        /// This selects the same method as above, since the number of parameters and the type of the third parameter disambiguate the overload.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"Find")</c>
        /// </term><description>
        /// This throws an exception since <see cref="Array.Find{T}"/> is a generic method only.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"Find&lt;&gt;")</c>
        /// </term><description>
        /// This selects the generic method <see cref="Array.Find{T}"/>, without specifying any type arguments
        /// or parameter types.  This method is not overloaded so no disambiguation via parameter types is needed.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"Find&lt;&gt;",null)</c>
        /// </term><description>
        /// This is equivalent to the above.  Adding two more nulls (corresponding to the two parameters of Find) would also be equivalent.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"Find&lt;&gt;",typeof(int))</c>
        /// </term><description>
        /// This selects the generic method <see cref="Array.Find{T}"/>, with type argument set to <c>int</c>.
        /// No parameter types are specified.  This method is not overloaded so no disambiguation via parameter types is needed.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"FindIndex&lt;&gt;",typeof(int),typeof(int[]),typeof(Predicate&lt;int&gt;))</c>
        /// </term><description>
        /// This selects the first overload of the generic method <see cref="Array.FindIndex{T}(T[], Predicate{T})"/>, with type argument set to <c>int</c>.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"FindIndex&lt;&gt;",typeof(int),null,null)</c>
        /// </term><description>
        /// This selects the same method as above, since the number of parameters (two) disambiguates the overload.
        /// </description></item>
        /// <item><term>
        /// <c>new MethodReference(typeof(Array),"FindIndex&lt;&gt;",null,null,null)</c>
        /// </term><description>
        /// This selects the first overload of the generic method <see cref="Array.FindIndex{T}(T[], Predicate{T})"/>, without specializing on a type argument.
        /// </description></item>
        /// </list>
        /// </remarks>

#if SUPPRESS_AMBIGUOUS_REFERENCE_WARNINGS
#pragma warning restore 419
#endif

        public MethodReference(Type type, string methodName, params Type[] parameterTypes)
        {
            Type = type;
            MethodName = ParseGenericMethodName(methodName, out TypeParameterCount);
            if (parameterTypes.Length > 0)
            {
                if (TypeParameterCount > 0)
                {
                    this.TypeArguments = new Type[TypeParameterCount];
                    Array.Copy(parameterTypes, TypeArguments, TypeParameterCount);
                    int parameterCount = parameterTypes.Length - TypeParameterCount;
                    if (parameterCount > 0)
                    {
                        ParameterTypes = new Type[parameterCount];
                        Array.Copy(parameterTypes, TypeParameterCount, ParameterTypes, 0, ParameterTypes.Length);
                    }
                }
                else
                {
                    ParameterTypes = (Type[]) parameterTypes.Clone();
                }
            }
        }

        /// <summary>
        /// Convert the MethodReference into a concrete MethodInfo.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="MissingMethodException">No methods of <c>this.Type</c> match the specification.</exception>
        /// <exception cref="AmbiguousMatchException">More than one method matches (due to wildcards in the MethodReference).</exception>
        /// <exception cref="ArgumentException">A type argument does not satisfy the constraints of the generic method definition.</exception>
        public MethodInfo GetMethodInfo()
        {
            MethodInfo[] methods = Type.GetMethods();
            LastMatchException = null;
            methods = Array.FindAll<MethodInfo>(methods, this.MatchesMethod);
            if (methods.Length == 0)
            {
                if (LastMatchException == null) throw new MissingMethodException(this.ToString());
                else throw LastMatchException;
            }
            LastMatchException = null;
            if (methods.Length > 1)
            {
                if (ParameterTypes == null)
                {
                    // restrict to methods having zero arguments.
                    MethodInfo[] emptyMethods = Array.FindAll<MethodInfo>(methods, delegate(MethodInfo mmethod) { return (mmethod.GetParameters().Length == 0); });
                    if (emptyMethods.Length == 0) throw new AmbiguousMatchException(GetAmbiguousMatchString(methods));
                    else if (emptyMethods.Length == 1) methods = emptyMethods;
                    else throw new AmbiguousMatchException(GetAmbiguousMatchString(emptyMethods));
                }
                else
                {
                    throw new AmbiguousMatchException(GetAmbiguousMatchString(methods));
                }
            }
            MethodInfo method = methods[0];
            if (method.ContainsGenericParameters && TypeArguments != null)
            {
                // fill in generic type parameters
                method = MakeGenericMethod(method, TypeArguments);
            }
            return method;
        }

        protected string GetAmbiguousMatchString(MethodInfo[] methods)
        {
            StringBuilder s = new StringBuilder();
            s.AppendFormat("Looking for {0}, found:", this);
            foreach (MethodInfo method in methods)
            {
                s.AppendLine();
                s.Append(StringUtil.MethodSignatureToString(method));
            }
            return s.ToString();
        }

        /// <summary>
        /// Test if a MethodInfo satisfies the constraints of a MethodReference.
        /// </summary>
        /// <param name="method">A method to test.</param>
        /// <returns>True if the method satisfies the constraints.</returns>
        public bool MatchesMethod(MethodInfo method)
        {
            if (method.Name != MethodName) return false;
            if (method.ContainsGenericParameters)
            {
                if (Invoker.GenericParameterCount(method) != TypeParameterCount) return false;
                if (TypeArguments != null && ParameterTypes != null)
                {
                    // fill in generic type parameters in order to check parameter types.
                    try
                    {
                        method = MakeGenericMethod(method, TypeArguments);
                    }
                    catch (ArgumentException ex)
                    {
                        LastMatchException = ex;
                        return false;
                    }
                }
            }
            if (ParameterTypes != null)
            {
                ParameterInfo[] parameters = method.GetParameters();
                if (parameters.Length != ParameterTypes.Length) return false;
                for (int i = 0; i < parameters.Length; i++)
                {
                    if (ParameterTypes[i] == null) continue;
                    ParameterInfo parameter = parameters[i];
                    Type tp = parameter.ParameterType;
                    // Turn &type into type:
                    if (parameter.IsOut) tp = tp.GetElementType();
                    if (!tp.IsAssignableFrom(ParameterTypes[i])) return false;
                }
            }
            else
            {
                // if ParameterTypes == null, then the parameter types are not checked.  Thus all overloads
                // will be accepted.
            }
            return true;
        }

        /// <summary>
        /// Same as <see cref="MethodInfo.MakeGenericMethod"/>, but allows <c>null</c> as a wildcard.
        /// </summary>
        /// <param name="method">A generic method.</param>
        /// <param name="typeArguments">An array of types, some of which may be null.</param>
        /// <returns>A method with the non-null type arguments filled in.</returns>
        public static MethodInfo MakeGenericMethod(MethodInfo method, Type[] typeArguments)
        {
            bool anyChanged = false;
            int count = 0;
            Type[] args = method.GetGenericArguments();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].IsGenericParameter)
                {
                    if (typeArguments[count] != null)
                    {
                        args[i] = typeArguments[count];
                        anyChanged = true;
                    }
                    count++;
                }
            }
            if (count != typeArguments.Length)
                throw new ArgumentException("typeArguments.Length (" + typeArguments.Length + ") does not match the method signature: " + method);
            if (!anyChanged) return method;
            else return method.MakeGenericMethod(args);
        }

        /// <summary>
        /// Parse a method name of the form <c>fun</c> or <c>fun&lt;&gt;</c> or <c>fun&lt;,&gt;</c>, etc.
        /// </summary>
        /// <param name="methodName"></param>
        /// <param name="typeParameterCount"></param>
        /// <returns>The raw method name with generic suffix removed.</returns>
        public static string ParseGenericMethodName(string methodName, out int typeParameterCount)
        {
            if (methodName[methodName.Length - 1] != '>')
            {
                typeParameterCount = 0;
                return methodName;
            }
            else
            {
                int pos = methodName.IndexOf('<');
                typeParameterCount = methodName.Length - 1 - pos;
                return methodName.Substring(0, pos);
            }
        }

        public static MethodReference Parse(string s)
        {
            return Parse(s, "");
        }

        public static MethodReference Parse(string s, string namespaceName)
        {
            // parse the name into class.method
            bool isGeneric = (s[s.Length - 1] == '>');
            if (isGeneric)
            {
                throw new NotImplementedException();
            }
            int genericPos = s.IndexOf("<", StringComparison.InvariantCulture);
            if (genericPos == -1) genericPos = s.Length;
            int pos = s.Substring(0, genericPos).LastIndexOf(".", StringComparison.InvariantCulture);
            if (pos < 1 || pos == genericPos - 1) throw new ArgumentException(s + " is not of the form class.method");
            string typeName = s.Substring(0, pos);
            string methodName = s.Substring(pos + 1, genericPos - (pos + 1));
            Type type;
            try
            {
                type = Invoker.GetLoadedType(typeName);
            }
            catch
            {
                if (namespaceName == "") throw;
                else
                {
                    string fullName = namespaceName + "." + typeName;
                    type = Invoker.GetLoadedType(fullName);
                }
            }
            return new MethodReference(type, methodName);
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder(StringUtil.TypeToString(Type) + "." + MethodName);
            if (TypeArguments != null)
            {
                s.Append("<");
                StringUtil.AppendTypes(s, TypeArguments);
                s.Append(">");
            }
            else if (TypeParameterCount > 0)
            {
                s.Append("<");
                for (int i = 1; i < TypeParameterCount; i++) s.Append(",");
                s.Append(">");
            }
            if (ParameterTypes != null)
            {
                s.Append("(");
                StringUtil.AppendTypes(s, ParameterTypes);
                s.Append(")");
            }
            return s.ToString();
        }

        public static MethodReference FromFactorAttribute(FactorMethodAttribute attr)
        {
            return new MethodReference(attr.Type, attr.MethodName, attr.ParameterTypes);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}