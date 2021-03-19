// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

//#define PrintStatistics

using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;
using System.Reflection.Emit;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Reflection
{
    /// <summary>
    /// Represents the type parameters and argument conversions needed to invoke a method.
    /// </summary>
    /// <remarks><para>
    /// Given a generic method definition and a list of arguments, a Binding represents the 
    /// additional information needed to invoke the method with those arguments.  Specifically,
    /// it stores a list of type parameters (possibly empty if the method is not generic) and
    /// a list of conversions, one for each argument.  Note that Bindings are not unique, e.g.
    /// a type parameter might be instantiated in several different ways, yet still compatible
    /// with the given arguments.  Each of these choices is a different Binding.
    /// </para><para>
    /// This class allows you to find the Binding which is 'best' in the sense of being the most
    /// specific.  The &lt; operator can be used to compare the specificity of Bindings.
    /// </para></remarks>
    public class Binding
    {
        /// <summary>
        /// The type inferred for each type parameter.
        /// </summary>
        /// <remarks>All entries are non-null.  A missing entry indicates an unconstrained parameter.</remarks>
        public readonly Dictionary<Type, Type> Types;

        /// <summary>
        /// The conversion inferred for each method parameter position.
        /// </summary>
        public readonly Conversion[] Conversions;

        /// <summary>
        /// Array indexing depth needed to obtain a match.
        /// </summary>
        public uint Depth;

        public void SetTo(Binding info)
        {
            foreach (KeyValuePair<Type, Type> entry in info.Types) Types[entry.Key] = entry.Value;
            info.Conversions.CopyTo(Conversions, 0);
            Depth = info.Depth;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="info"></param>
        public Binding(Binding info)
        {
            Types = new Dictionary<Type, Type>();
            Conversions = new Conversion[info.Conversions.Length];
            SetTo(info);
        }

        /// <summary>
        /// Creare an empty Binding to a given method.
        /// </summary>
        /// <param name="method"></param>
        /// <remarks>The conversions are all set to null.</remarks>
        public Binding(MethodBase method)
        {
            Types = new Dictionary<Type, Type>();
            ParameterInfo[] parameters = method.GetParameters();
            Conversions = new Conversion[parameters.Length];
        }

        /// <summary>
        /// True if A is more specific than B.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>True if A is more specific than B.
        /// </returns>
        public static bool operator <(Binding a, Binding b)
        {
            if (a == null) return false;
            if (b == null) return true;
            if (a.Types == null || b.Types == null) return false;
            if (a.Conversions == null || b.Conversions == null) return false;
            if (a.Conversions.Length != b.Conversions.Length) return false;
            //if (a.Types.Count < b.Types.Count) return true;
            float aWeight = Conversion.GetWeight(a.Conversions);
            float bWeight = Conversion.GetWeight(b.Conversions);
            if (aWeight == bWeight) return a.Depth > b.Depth;
            return (aWeight < bWeight);
        }

        public static bool operator >(Binding a, Binding b)
        {
            return (b < a);
        }

        /// <summary>
        /// Apply the conversions to a set of arguments.
        /// </summary>
        /// <param name="args">An array of length Conversions.Length.  Modified to contain the converted values.</param>
        public void ConvertAll(object[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                Converter conv = Conversions[i].Converter;
                if (conv != null) args[i] = conv(args[i]);
            }
        }

        /// <summary>
        /// Find the most specific Binding for a generic method.
        /// </summary>
        /// <param name="method"></param>
        /// <param name="actuals">Types from which to infer type parameters.  actuals.Length == number of method parameters.  actuals[i] may be null to allow any type, or typeof(Nullable) to mean "any nullable type".</param>
        /// <param name="conversionOptions">Specifies which conversions are allowed.</param>
        /// <param name="exception">Exception created on failure</param>
        /// <returns>null if binding fails.</returns>
        public static Binding GetBestBinding(MethodBase method, Type[] actuals, ConversionOptions conversionOptions, out Exception exception)
        {
            List<Exception> errors = new List<Exception>();
            IEnumerator<Binding> iter = InferGenericParameters(method, actuals, conversionOptions, errors);
            if (!iter.MoveNext())
            {
                exception = errors[0];
                string message = exception.Message + " of method ";
                bool useShortString = true;
                if (useShortString)
                {
                    message += GetParameterMismatchString(method, actuals);
                }
                else
                {
                    message += method.DeclaringType.Name + "." + method.Name;
                    message += Environment.NewLine + GetParameterTableString(method, actuals);
                }
                exception = ChangeExceptionMessage(exception, message);
                return null;
            }
            else exception = null;
            // must make a copy since the iterator will modify current.
            Binding best = new Binding(iter.Current);
            if (!best.IsMinimal())
            {
                int count = 1;
                while (iter.MoveNext())
                {
                    count++;
                    if (iter.Current < best) best.SetTo(iter.Current);
                }
#if PrintStatistics
                Console.WriteLine("GetBestBinding: {0} bindings examined", count);
#endif
            }
            //if (iter.MoveNext()) throw new System.Reflection.AmbiguousMatchException();
            return best;
        }

        /// <summary>
        /// True if the binding has no explicit conversions (excluding the first) and at most one nonzero conversion.
        /// </summary>
        /// <returns></returns>
        private bool IsMinimal()
        {
            bool foundNonzero = false;
            for (int i = 0; i < Conversions.Length; i++)
            {
                if (Conversions[i].IsExplicit && (i > 0)) return false;
                if (Conversions[i].GetWeight() > 0.0)
                {
                    if (foundNonzero) return false;
                    else foundNonzero = true;
                }
            }
            return true;
        }

        private static Exception ChangeExceptionMessage(Exception exception, string message)
        {
            if (exception.InnerException == null)
            {
                exception = (Exception)Activator.CreateInstance(exception.GetType(), message);
            }
            else
            {
                // must only try this when InnerException!=null
                exception = (Exception)Activator.CreateInstance(exception.GetType(), message, exception.InnerException);
            }
            return exception;
        }

        private static string GetParameterTableString(MethodBase method, Type[] actuals)
        {
            ParameterInfo[] parameters = method.GetParameters();
            string[][] lines = new string[5][];
            lines[0] = new string[parameters.Length + 2];
            lines[1] = new string[] { " " };
            lines[2] = new string[parameters.Length + 2];
            lines[3] = new string[] { " " };
            lines[4] = new string[parameters.Length + 2];
            lines[0][0] = "Parameter";
            lines[0][1] = "---------";
            lines[2][0] = "Provided";
            lines[2][1] = "--------";
            lines[4][0] = "Expected";
            lines[4][1] = "--------";
            for (int i = 0; i < parameters.Length; i++)
            {
                lines[0][i + 2] = parameters[i].Name;
                lines[2][i + 2] = (i >= actuals.Length) ? "missing" : (actuals[i] == null) ? "null" : StringUtil.TypeToString(actuals[i]);
                lines[4][i + 2] = StringUtil.TypeToString(parameters[i].ParameterType);
            }
            return StringUtil.JoinColumns(lines);
        }

        private static string GetParameterMismatchString(MethodBase method, Type[] actuals)
        {
            StringBuilder sb = new StringBuilder(StringUtil.MethodFullNameToString(method));
            ParameterInfo[] parameters = method.GetParameters();
            if (parameters.Length == actuals.Length)
            {
                sb.Append("(");
                for (int i = 0; i < parameters.Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    ParameterInfo parameter = parameters[i];
                    sb.Append(StringUtil.TypeToString(parameter.ParameterType));
                    sb.Append(" ");
                    sb.Append(parameter.Name);
                    sb.Append(" = ");
                    sb.Append(actuals[i] == null ? "null" : StringUtil.TypeToString(actuals[i]));
                }
                sb.Append(")");
            }
            return sb.ToString();
        }


        /// <summary>
        /// Infer type parameters for a method call.
        /// </summary>
        /// <param name="method"></param>
        /// <param name="actuals">Types from which to infer type parameters.  actuals.Length == number of method parameters.  actuals[i] may be null to allow any type, or typeof(Nullable) to mean "any nullable type".</param>
        /// <param name="conversionOptions">Specifies which conversions are allowed</param>
        /// <param name="errors">A list which collects binding errors.</param>
        /// <returns>An iterator which yields all possible bindings.</returns>
        /// <remarks>Because it considers all possible bindings for each argument, this function can 
        /// infer type parameters in cases where the C# 2.0 specification (sec 20.6.4) cannot.</remarks>
        public static IEnumerator<Binding> InferGenericParameters(MethodBase method, Type[] actuals, ConversionOptions conversionOptions, IList<Exception> errors)
        {
            Type[] formals = Invoker.GetParameterTypes(method);
            Binding binding = new Binding(method);
            //return InferGenericParameters(formals, actuals, binding, 0, true);
            IEnumerator<Binding> iter = InferGenericParameters(formals, actuals, binding, errors, 0, 0, true, i => true, conversionOptions);
            if (method.IsGenericMethodDefinition)
            {
                return ApplyTypeConstraints(method, iter, errors);
            }
            else
            {
                return iter;
            }
        }

        /// <summary>
        /// Filter a stream of bindings via type constraints.
        /// </summary>
        /// <param name="method">A generic method which may have type constraints.</param>
        /// <param name="iter">A stream of Bindings.</param>
        /// <param name="errors">A list which collects binding errors.</param>
        /// <returns>A substream of Bindings which all satisfy the type constraints.
        /// Type parameters which were previously unknown may be filled in by the type constraints, possibly
        /// in multiple ways.  Thus a single Binding from <paramref name="iter"/> with an unknown type parameter
        /// may expand into many new Bindings, differing only in the instantiation of that parameter.
        /// Known type parameters are left unchanged.
        /// </returns>
        public static IEnumerator<Binding> ApplyTypeConstraints(MethodBase method, IEnumerator<Binding> iter, IList<Exception> errors)
        {
            int failCount = 0;
            while (iter.MoveNext())
            {
                Binding binding = iter.Current;
                bool found = false;
                IEnumerator<Binding> iter2 = ApplyTypeConstraints(binding, errors);
                while (iter2.MoveNext())
                {
                    yield return binding;
                    found = true;
                }
                if (!found) failCount++;
            }
#if PrintStatistics
            Console.WriteLine("ApplyTypeConstraints failCount = {0}", failCount);
#endif
        }

        /// <summary>
        /// Expand a binding via type constraints.
        /// </summary>
        /// <param name="binding">The binding to expand</param>
        /// <param name="errors">A list which collects binding errors.</param>
        /// <param name="doneParams">The set of parameters already processed.</param>
        /// <returns>A stream of Bindings which satisfy the constraints on typeParams[pos] and above.</returns>
        public static IEnumerator<Binding> ApplyTypeConstraints(Binding binding, IList<Exception> errors, ICollection<Type> doneParams = null)
        {
            Type typeParam = null;
            foreach (Type boundParam in binding.Types.Keys)
            {
                if (doneParams != null && doneParams.Contains(boundParam)) continue;
                typeParam = boundParam;
                break;
            }
            if (typeParam == null)
            {
                yield return binding;
                yield break;
            }
            IEnumerator<Binding> iter = ApplyTypeConstraints(typeParam, binding, errors);
            while (iter.MoveNext())
            {
                // recurse on the rest
                List<Type> doneParams2 = new List<Type>();
                if (doneParams != null) doneParams2.AddRange(doneParams);
                doneParams2.Add(typeParam);
                IEnumerator<Binding> iter2 = ApplyTypeConstraints(binding, errors, doneParams2);
                while (iter2.MoveNext()) yield return binding;
            }
        }

        public static IEnumerator<Binding> ApplyTypeConstraints(Type typeParam, Binding binding, IList<Exception> errors)
        {
            Type actual;
            if (!binding.Types.TryGetValue(typeParam, out actual))
            {
                yield return binding;
                yield break;
            }
            // check for special constraints first
            // list of special constraints:
            // ms-help://MS.VSCC.v80/MS.MSDN.v80/MS.VisualStudio.v80.en/dv_csref/html/141b003e-1ddb-4e1c-bcb2-e1c3870e6a51.htm
            GenericParameterAttributes attrs = typeParam.GenericParameterAttributes & GenericParameterAttributes.SpecialConstraintMask;
            if ((attrs & GenericParameterAttributes.ReferenceTypeConstraint) != 0)
            {
                if (actual.IsValueType)
                {
                    errors.Add(new ArgumentException(StringUtil.TypeToString(actual) + " is not a reference type, as required by the type constraint for " + typeParam));
                    yield break;
                }
            }
            if ((attrs & GenericParameterAttributes.NotNullableValueTypeConstraint) != 0)
            {
                if (Conversion.IsNullable(actual))
                {
                    errors.Add(new ArgumentException(StringUtil.TypeToString(actual) + " is nullable, which is prohibited by the type constraint for " + typeParam));
                    yield break;
                }
            }
            if ((attrs & GenericParameterAttributes.DefaultConstructorConstraint) != 0)
            {
                if (!actual.IsValueType && actual.GetConstructor(Type.EmptyTypes) == null)
                {
                    errors.Add(
                        new ArgumentException(StringUtil.TypeToString(actual) + " does not have a parameterless constructor, as required by the type constraint for " +
                                              typeParam));
                    yield break;
                }
            }
            // now check for subtype constraints
            Type[] constraints = typeParam.GetGenericParameterConstraints();
            IEnumerator<Binding> iter = ApplyTypeConstraints(actual, constraints, binding, errors, 0);
            while (iter.MoveNext()) yield return iter.Current;
        }

        /// <summary>
        /// Expand a binding via type constraints.
        /// </summary>
        /// <param name="actual">The type that needs to satisfy the constraints.</param>
        /// <param name="constraints">The constraints on the type parameter at binding.Types[pos].</param>
        /// <param name="binding">Any binding.</param>
        /// <param name="errors">A list which collects binding errors.</param>
        /// <param name="start">The index of the first constraint to process.</param>
        /// <returns>A stream of Bindings which satisfy constraints[start] and higher.</returns>
        public static IEnumerator<Binding> ApplyTypeConstraints(Type actual, Type[] constraints, Binding binding, IList<Exception> errors, int start)
        {
            if (start >= constraints.Length)
            {
                yield return binding;
                yield break;
            }
            Type formal = constraints[start];
            IEnumerator<Binding> iter = InferGenericParameters(formal, actual, binding, errors, -1, true, ConversionOptions.NoConversions);
            while (iter.MoveNext())
            {
                // recurse on the rest
                IEnumerator<Binding> iter2 = ApplyTypeConstraints(actual, constraints, binding, errors, start + 1);
                while (iter2.MoveNext())
                {
                    yield return binding;
                }
            }
        }

        /// <summary>
        /// Infer remaining type parameters for a method call or generic type.
        /// </summary>
        /// <param name="formals">Types with parameters to infer.</param>
        /// <param name="actuals">The corresponding types from which to infer parameters.  actuals.Length == formals.Length.  actuals[i] may be null to allow any type, or typeof(Nullable) to mean "any nullable type".</param>
        /// <param name="binding">The partial binding inferred from actuals[i &lt; start]. 
        /// Will be mutated and returned as elements of the stream.</param>
        /// <param name="errors">A list which collects binding errors.</param>
        /// <param name="start">The index into the formals array to begin processing.</param>
        /// <param name="position">The index into binding.Conversions to place the conversion.  If -1, only null conversions are allowed.  Ignored if isMethodCall is true.</param>
        /// <param name="isMethodCall">True if formals correspond to method parameters.  position is always taken equal to start.</param>
        /// <param name="allowSubtype">True if implicit subtype conversions are allowed for type i.</param>
        /// <param name="conversionOptions">Specifies which conversions are allowed.</param>
        /// <returns>A stream of all possible bindings which match formals[i] to actuals[i], for all i >= start, including conversions.</returns>
        /// <remarks>Each element of the stream is the same object as <paramref name="binding"/>, but modified to (possibly) 
        /// include more bindings.  <paramref name="binding"/> is returned to its original state at the end of the stream.</remarks>
        public static IEnumerator<Binding> InferGenericParameters(Type[] formals, Type[] actuals, Binding binding, IList<Exception> errors, int start, int position,
                                                                  bool isMethodCall, Func<int, bool> allowSubtype, ConversionOptions conversionOptions)
        {
            if (formals.Length != actuals.Length)
            {
                errors.Add(new ArgumentException(String.Format("formals.Length = {0}, actuals.Length = {1}", formals.Length, actuals.Length)));
                yield break;
            }
            if (start >= formals.Length)
            {
                yield return binding;
                yield break;
            }
            IEnumerator<Binding> iter = InferGenericParameters(formals[start], actuals[start], binding, errors, isMethodCall ? start : position, allowSubtype(start),
                                                               conversionOptions);
            while (iter.MoveNext())
            {
                // recurse on the rest
                IEnumerator<Binding> iter2 = InferGenericParameters(formals, actuals, binding, errors, start + 1, isMethodCall ? (start + 1) : position, isMethodCall,
                                                                    allowSubtype, conversionOptions);
                while (iter2.MoveNext())
                {
                    yield return binding;
                }
            }
        }

        /// <summary>
        /// Infer type parameter bindings.
        /// </summary>
        /// <param name="formal">The type with parameters to infer.</param>
        /// <param name="actual">A type from which to infer parameters.  May be null to allow any type, or typeof(Nullable) to mean "any nullable type".  May itself contain type parameters.</param>
        /// <param name="binding">Known bindings, which are taken as fixed.  Conversions[position] must be null.</param>
        /// <param name="errors">A list which collects binding errors.</param>
        /// <param name="position">If formal corresponds to a method parameter type, this is the index of the method parameter.  Otherwise it is -1 and no conversions are allowed.</param>
        /// <param name="allowSubtype">If true, actual can be a subtype of formal (implicit subtype conversions are allowed).</param>
        /// <param name="conversionOptions">Specifies which conversions are allowed.</param>
        /// <returns>A stream of all possible bindings which match formal to actual, 
        /// including conversions.</returns>
        /// <remarks><para>Each element of the stream is the same object as <paramref name="binding"/>, but modified to (possibly) 
        /// include more bindings.  <paramref name="binding"/> is returned to its original state at the end of the stream.
        /// </para><para>
        /// If <paramref name="formal"/> has no type parameters, then at most one Binding is returned.
        /// </para></remarks>
        public static IEnumerator<Binding> InferGenericParameters(Type formal, Type actual, Binding binding, IList<Exception> errors, int position, bool allowSubtype,
                                                                  ConversionOptions conversionOptions)
        {
            if (actual == null)
            {
                yield return binding;
                yield break;
            }
            bool isReplaced = false;
            if (formal.IsGenericParameter)
            {
                Type boundFormal;
                if (!binding.Types.TryGetValue(formal, out boundFormal))
                {
                    // formal is not bound
                    if (actual == typeof(Nullable))
                    {
                        // nothing to infer
                        yield return binding;
                    }
                    else
                    {
                        if (GenericParameterFactory.IsConstructedParameter(actual))
                        {
                            // pretend that actual has all constraints required by formal
                            binding.Types[formal] = formal;
                            binding.Types[actual] = formal;
                            yield return binding;
                            binding.Types.Remove(actual);
                        }
                        else if (allowSubtype)
                        {
                            // formal can be bound to any type which actual is assignable to.
                            foreach (Type alternate in TypesAssignableFrom(actual))
                            {
                                binding.Types[formal] = alternate;
                                bool checkConstraints = true;
                                if (checkConstraints)
                                {
                                    // check right away if this binding satisfies the constraints on formal.
                                    // this can sometimes improve performance.
                                    IEnumerator<Binding> iter2 = ApplyTypeConstraints(formal, binding, errors);
                                    while (iter2.MoveNext()) yield return iter2.Current;
                                }
                                else
                                {
                                    yield return binding;
                                }
                            }
                        }
                        else
                        {
                            binding.Types[formal] = actual;
                            yield return binding;
                        }
                        // restore previous state
                        binding.Types.Remove(formal);
                    }
                    yield break;
                }
                else
                {
                    formal = boundFormal;
                    isReplaced = true;
                    // fall through
                }
            }
            if (isReplaced || !formal.ContainsGenericParameters || actual == typeof(Nullable) || GenericParameterFactory.IsConstructedParameter(actual))
            {
                // no more substitutions are possible in formal.
                if (formal.Equals(actual) || GenericParameterFactory.IsConstructedParameter(actual))
                {
                    yield return binding;
                }
                else if (allowSubtype && conversionOptions.TryGetConversion(actual, formal, position, out Conversion conv))
                {
                    Conversion oldConversion = new Conversion();
                    if (position >= 0)
                    {
                        oldConversion = binding.Conversions[position];
                        binding.Conversions[position] = conv;
                    }
                    yield return binding;
                    // restore previous state
                    if (position >= 0) binding.Conversions[position] = oldConversion;
                }
                else
                {
                    errors.Add(new ArgumentException(StringUtil.TypeToString(actual) + " is not of type " + StringUtil.TypeToString(formal) + PositionString(position)));
                }
                yield break;
            }
            else if (formal.IsArray)
            {
                // formal is a generic array type such as T[,] or C<T>[]
                // for such types, ContainsGenericParameters is true but IsGenericType is false and GetGenericArguments is empty
                if (!actual.IsArray)
                {
                    errors.Add(new ArgumentException(StringUtil.TypeToString(actual) + " is not of type " + StringUtil.TypeToString(formal) + PositionString(position)));
                    yield break;
                }
                int rank = formal.GetArrayRank();
                if (actual.GetArrayRank() != rank)
                {
                    errors.Add(new ArgumentException(StringUtil.TypeToString(actual) + " is not of type " + StringUtil.TypeToString(formal) + PositionString(position)));
                    yield break;
                }
                Type formalElement = formal.GetElementType();
                Type actualElement = actual.GetElementType();
                binding.Depth++;
                IEnumerator<Binding> iter3 = InferGenericParameters(formalElement, actualElement, binding, errors, position, allowSubtype, conversionOptions);
                while (iter3.MoveNext())
                {
                    binding = iter3.Current;
                    Conversion elementConversion = default(Conversion);
                    if (position >= 0)
                    {
                        elementConversion = binding.Conversions[position];
                        if (elementConversion.Converter != null)
                        {
                            // if there is an element conversion, promote it to an array conversion
                            Type boundElementType = binding.Bind(formalElement);
                            if (!Conversion.TryGetArrayConversion(rank, rank, boundElementType, elementConversion, out binding.Conversions[position]))
                            {
                                continue;
                            }
                        }
                    }
                    yield return binding;
                    // restore previous state
                    if (position >= 0) binding.Conversions[position] = elementConversion;
                }
                binding.Depth--;
                yield break;
            }
            else
            {
                // TODO: need to check for generic pointer types
                // formal is a generic type such as C<T>.
                // we consider all supertypes of actual having the form C<U> and then try to match (T,U).
                Type formalGenericTypeDefinition = formal.GetGenericTypeDefinition();
                Type[] formalArgs = formal.GetGenericArguments();
                bool includeIList = formalGenericTypeDefinition.Equals(typeof(IList<>));
                IEnumerable<Type> choices;
                if (allowSubtype) choices = TypesAssignableFrom(actual, includeIList);
                else
                {
                    choices = new Type[] { actual };
                }
                Type[] formalDefArgs = formalGenericTypeDefinition.GetGenericArguments();
                bool parameterIsCovariant(int i) => (formalDefArgs[i].GenericParameterAttributes & GenericParameterAttributes.Covariant) != 0;
                int subclassCount = 0;
                bool anyMatch = false;
                List<Exception> innerErrors = new List<Exception>();
                binding.Depth++;
                foreach (Type parent in choices)
                {
                    if (!parent.IsGenericType || !formalGenericTypeDefinition.Equals(parent.GetGenericTypeDefinition()))
                    {
                        subclassCount++;
                        continue;
                    }
                    // both types have the same form (C<T> and C<U>)
                    Type[] actualArgs = parent.GetGenericArguments();
                    // conversions are not allowed when matching (T,U)
                    innerErrors.Clear();
                    IEnumerator<Binding> iter2 = InferGenericParameters(formalArgs, actualArgs, binding, innerErrors, 0, -2, false, parameterIsCovariant, ConversionOptions.NoConversions);
                    while (iter2.MoveNext())
                    {
                        anyMatch = true;
                        binding = iter2.Current;
                        if (position >= 0) binding.Conversions[position].SubclassCount += subclassCount;
                        yield return binding;
                        // restore previous state
                        if (position >= 0) binding.Conversions[position].SubclassCount -= subclassCount;
                    }
                    subclassCount++;
                }
                binding.Depth--;
                if (!anyMatch || innerErrors.Count > 0)
                {
                    string formalString = StringUtil.TypeToString(formal);
                    bool usePreciseMessage = false;
                    if (usePreciseMessage)
                    {
                        // this block tries to make the error message more precise.
                        // however it can significantly slow down the code due to exceptions being thrown.
                        try
                        {
                            Type boundFormal = binding.Bind(formal);
                            formalString = StringUtil.TypeToString(boundFormal) + " (" + formalString + ")";
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine("InferGenericParameters: {0}", ex.Message);
                        }
                    }
                    string errorString = StringUtil.TypeToString(actual) + " is not of type " + formalString + PositionString(position);
                    if (!anyMatch)
                    {
                        errors.Add(new ArgumentException(errorString));
                    }
                    else
                    {
                        foreach (Exception innerException in innerErrors)
                        {
                            errors.Add(new ArgumentException(errorString, innerException));
                        }
                    }
                }
            }
        }

        protected static string PositionString(int position)
        {
            if (position == -1) return " as required by the type constraint";
            else if (position >= 0) return " for argument " + (position + 1);
            else return "";
        }

        /// <summary>
        /// All base classes and interfaces of a given type.
        /// </summary>
        /// <param name="type">Any non-null type.  May be a type parameter.</param>
        /// <param name="includeIList">If true, include IList when <paramref name="type"/> is an array.</param>
        /// <returns>All base classes and interfaces of <paramref name="type"/>, most specific types first.</returns>
        public static IEnumerable<Type> TypesAssignableFrom(Type type, bool includeIList = true)
        {
            // must be kept in sync with Conversion.IsAssignableFrom
            bool isObject = false;
            for (Type baseType = type; baseType != null; baseType = baseType.BaseType)
            {
                if (baseType.Equals(typeof(object)))
                {
                    isObject = true;
                    break;
                }
                yield return baseType;
            }
            Type[] faces = type.GetInterfaces();
            foreach (Type face in faces)
            {
                yield return face;
            }
            // array covariance (C# 2.0 specification, sec 20.5.9)
            if (includeIList && type.IsArray && type.GetArrayRank() == 1)
            {
                IEnumerable<Type> elementSupertypes = TypesAssignableFrom(type.GetElementType());
                bool isFirst = true;
                foreach (Type super in elementSupertypes)
                {
                    if (isFirst)
                    {
                        isFirst = false;
                        continue;
                    }
                    yield return typeof(IList<>).MakeGenericType(super);
                }
            }
            if (isObject) yield return typeof(object);
        }

        /// <summary>
        /// Fill in some type parameters.
        /// </summary>
        /// <param name="type">A type which may have generic parameters.</param>
        /// <returns>A type with possibly fewer generic parameters.</returns>
        public Type Bind(Type type)
        {
            return ReplaceTypeParameters(type, Types);
        }

        /// <summary>
        /// Replace type parameters in a generic type.
        /// </summary>
        /// <param name="type">A type which may have generic parameters.</param>
        /// <param name="typeMap">A dictionary for mapping types</param>
        /// <returns>A type with possibly fewer generic parameters.</returns>
        public static Type ReplaceTypeParameters(Type type, IReadOnlyDictionary<Type, Type> typeMap)
        {
            if (type == null) return null;
            else if (!type.ContainsGenericParameters) return type;
            else if (type.IsGenericParameter)
            {
                if (typeMap.TryGetValue(type, out Type actual)) return actual;
                else return type;
            }
            else if (type.IsArray)
            {
                // a generic array type such as T[,]
                // for such types, ContainsGenericParameters is true but IsGenericType is false and GetGenericArguments is empty
                Type elementType = ReplaceTypeParameters(type.GetElementType(), typeMap);
                int rank = type.GetArrayRank();
                if (rank == 1)
                    return elementType.MakeArrayType();
                else
                    return elementType.MakeArrayType(rank);
            }
            else
            {
                Type[] args = type.GetGenericArguments();
                for (int i = 0; i < args.Length; i++)
                {
                    args[i] = ReplaceTypeParameters(args[i], typeMap);
                }
                return type.GetGenericTypeDefinition().MakeGenericType(args);
            }
        }

        /// <summary>
        /// Perform an action for each type parameter inside a type.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="action"></param>
        public static void ForEachTypeParameter(Type type, Action<Type> action)
        {
            if (type == null) return;
            else if (!type.ContainsGenericParameters) return;
            else if (type.IsGenericParameter)
            {
                action(type);
            }
            else if (type.IsArray)
            {
                // a generic array type such as T[,]
                // for such types, ContainsGenericParameters is true but IsGenericType is false and GetGenericArguments is empty
                ForEachTypeParameter(type.GetElementType(), action);
            }
            else
            {
                Type[] args = type.GetGenericArguments();
                for (int i = 0; i < args.Length; i++)
                {
                    ForEachTypeParameter(args[i], action);
                }
            }
        }

        /// <summary>
        /// Specialize a method on the inferred type parameters.
        /// </summary>
        /// <param name="method">A non-null generic or non-generic method.</param>
        /// <returns>A method with all type parameters of that method filled in.  Type parameters of an enclosing type will not be filled in.</returns>
        /// <remarks>
        /// Unknown type parameters are replaced with System.Object.
        /// </remarks>
        public MethodBase Bind(MethodBase method)
        {
            MethodInfo info = method as MethodInfo;
            if (info == null) return method;
            if (!info.ContainsGenericParameters) return method;
            Type[] args = Array.ConvertAll(info.GetGenericArguments(), delegate (Type arg)
            {
                if (!arg.IsGenericParameter) return arg;
                else
                {
                    // if the actuals contain constructed type parameters, then these may also be bound, so
                    // we need to call Bind(bound)
                    if (Types.TryGetValue(arg, out Type bound)) return Bind(bound);
                    else
                    {
                        Type[] constraints = arg.GetGenericParameterConstraints();
                        if (constraints.Length == 0) return typeof(object);
                        else return Bind(constraints[0]);
                    }
                }
            });
            if (args.Length == 0) return method;
            else
            {
                return info.GetGenericMethodDefinition().MakeGenericMethod(args);
            }
        }

        public override string ToString()
        {
            return $"Binding({StringUtil.DictionaryToString<Type,Type>(Types, Environment.NewLine)}, Conversions={StringUtil.ArrayToString(Conversions)}, Depth={Depth})";
        }

        /// <summary>
        /// Get a type which is the most specific of the input types.
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <returns></returns>
        public static Type IntersectTypes(Type t1, Type t2)
        {
            if (t1.IsAssignableFrom(t2)) return t2;
            if (t2.IsAssignableFrom(t1)) return t1;
            if (t1.IsGenericParameter)
            {
                if (t2.IsGenericParameter)
                {
                    if (t1.Name != t2.Name) return null;
                    if (GenericParameterFactory.IsConstructedParameter(t1)) return t1;
                    if (GenericParameterFactory.IsConstructedParameter(t2)) return t2;
                    // make a new generic parameter with the union of type constraints
                    //Type result = new Type
                    GenericParameterFactory.Constraints constraints1 =
                        GenericParameterFactory.Constraints.FromTypeParameter(t1);
                    GenericParameterFactory.Constraints constraints2 =
                        GenericParameterFactory.Constraints.FromTypeParameter(t2);
                    if (constraints1.IsAssignableFrom(constraints2)) return t2;
                    if (constraints2.IsAssignableFrom(constraints1)) return t1;
                    GenericParameterFactory.Constraints constraints = constraints1.Intersect(constraints2);
                    Type t3 = GenericParameterFactory.MakeGenericParameter(t1.Name, constraints);
                    return t3;
                }
                else
                {
                    if (GenericParameterFactory.IsConstructedParameter(t1)) return t1;
                    GenericParameterFactory.Constraints constraints1 =
                        GenericParameterFactory.Constraints.FromTypeParameter(t1);
                    GenericParameterFactory.Constraints constraints = new GenericParameterFactory.Constraints();
                    if (constraints1.baseTypeConstraint != null &&
                        IntersectTypes(constraints1.baseTypeConstraint, t2) != null)
                        return GenericParameterFactory.MakeGenericParameter(t1.Name, constraints);
                    foreach (Type face in constraints1.interfaceConstraints)
                    {
                        if (IntersectTypes(face, t2) != null)
                            return GenericParameterFactory.MakeGenericParameter(t1.Name, constraints);
                    }
                    return null;
                    //throw new NotImplementedException();
                }
            }
            else if (t2.IsGenericParameter)
            {
                return IntersectTypes(t2, t1);
            }
            else if (t1.IsGenericType && t2.IsGenericType && t1.GetGenericTypeDefinition().Equals(t2.GetGenericTypeDefinition()))
            {
                Type[] typeArgs = t1.GetGenericArguments();
                Type[] typeArgs2 = t2.GetGenericArguments();
                Type[] newTypeArgs = new Type[typeArgs.Length];
                for (int i = 0; i < typeArgs.Length; i++)
                {
                    newTypeArgs[i] = IntersectTypes(typeArgs[i], typeArgs2[i]);
                    if (newTypeArgs[i] == null) return null;
                }
                return t1.GetGenericTypeDefinition().MakeGenericType(newTypeArgs);
            }
            else
            {
                // not compatible
                return null;
            }
        }
    }

    public class ConversionOptions
    {
        public static ConversionOptions NoConversions = new ConversionOptions();
        public static ConversionOptions AllConversions = new ConversionOptions() { AllowImplicitConversions = true, AllowExplicitConversions = true };

        public bool AllowImplicitConversions;
        public bool AllowExplicitConversions;
        public Func<Type, Type, int, bool> IsImplicitConversion;

        public bool TryGetConversion(Type fromType, Type toType, int position, out Conversion conv)
        {
            if (IsImplicitConversion != null && IsImplicitConversion(fromType, toType, position))
            {
                conv = new Conversion
                {
                    SubclassCount = Conversion.SpecialImplicitSubclassCount
                };
                return true;
            }
            return Conversion.TryGetConversion(fromType, toType, out conv) &&
                   (conv.Converter == null ||
                    AllowExplicitConversions ||
                    (AllowImplicitConversions && !conv.IsExplicit));
        }
    }

    public static class GenericParameterFactory
    {
        public static readonly Type ThisType = typeof(SelfReference<>).GetGenericArguments()[0];

        /// <summary>
        /// Used in generic parameter constraints to refer to the type being defined.
        /// </summary>
        public static class SelfReference<ThisType>
        {
        }

        public class Constraints
        {
            public GenericParameterAttributes attributes;
            public Type baseTypeConstraint;
            public Set<Type> interfaceConstraints = new Set<Type>();

            public IEnumerable<Type> GetEnumerable()
            {
                if (baseTypeConstraint != null) yield return baseTypeConstraint;
                foreach (Type face in interfaceConstraints) yield return face;
            }

            public static Constraints FromTypeParameter(Type typeParam)
            {
                Dictionary<Type, Type> typeMap = new Dictionary<Type, Type>
                {
                    [typeParam] = ThisType
                };
                return FromTypeParameter(typeParam, typeMap);
            }

            public static Constraints FromTypeParameter(Type typeParam, IReadOnlyDictionary<Type, Type> typeMap)
            {
                Type[] constraints = typeParam.GetGenericParameterConstraints();
                constraints = Array.ConvertAll(constraints, delegate (Type tt) { return Binding.ReplaceTypeParameters(tt, typeMap); });
                Constraints result = new Constraints();
                result.attributes = typeParam.GenericParameterAttributes;
                foreach (Type c in constraints)
                {
                    if (c.IsInterface) result.interfaceConstraints.Add(c);
                    else result.baseTypeConstraint = c;
                }
                return result;
            }

            public Constraints Intersect(Constraints that)
            {
                Constraints result = new Constraints();
                result.attributes = attributes | that.attributes;
                bool intersectConstraints = false;
                if (intersectConstraints)
                {
                    if (baseTypeConstraint == null) result.baseTypeConstraint = that.baseTypeConstraint;
                    else if (that.baseTypeConstraint == null) result.baseTypeConstraint = baseTypeConstraint;
                    else
                    {
                        result.baseTypeConstraint = Binding.IntersectTypes(baseTypeConstraint, that.baseTypeConstraint);
                        if (result.baseTypeConstraint == null) return null;
                    }
                    Set<Type> skip = new Set<Type>();
                    foreach (Type constraint in interfaceConstraints)
                    {
                        bool found = false;
                        foreach (Type constraint2 in that.interfaceConstraints)
                        {
                            Type newConstraint = Binding.IntersectTypes(constraint, constraint2);
                            if (newConstraint != null)
                            {
                                result.interfaceConstraints.Add(newConstraint);
                                skip.Add(constraint2);
                                found = true;
                                break;
                            }
                        }
                        if (!found) result.interfaceConstraints.Add(constraint);
                    }
                    foreach (Type constraint2 in that.interfaceConstraints)
                    {
                        if (!skip.Contains(constraint2)) result.interfaceConstraints.Add(constraint2);
                    }
                }
                return result;
            }

            public static T[] ArrayJoin<T>(T[] array1, T[] array2)
            {
                T[] result = new T[array1.Length + array2.Length];
                array1.CopyTo(result, 0);
                array2.CopyTo(result, array1.Length);
                return result;
            }

            /// <summary>
            /// True if that is more constrained than this.
            /// </summary>
            /// <param name="that"></param>
            /// <returns></returns>
            public bool IsAssignableFrom(Constraints that)
            {
                if (attributes != that.attributes) return false;
                if (baseTypeConstraint != null)
                {
                    if (that.baseTypeConstraint != null)
                    {
                        if (!baseTypeConstraint.IsAssignableFrom(that.baseTypeConstraint)) return false;
                    }
                    else
                    {
                        return false;
                    }
                }
                return (interfaceConstraints <= that.interfaceConstraints);
            }

            public override bool Equals(object obj)
            {
                return (obj is Constraints that) && 
                    (attributes == that.attributes) &&
                    (baseTypeConstraint == that.baseTypeConstraint) &&
                    (interfaceConstraints == that.interfaceConstraints);
            }

            public override int GetHashCode()
            {
                int hash = Hash.Start;
                hash = Hash.Combine(hash, attributes.GetHashCode());
                hash = Hash.Combine(hash, (baseTypeConstraint == null) ? 0 : baseTypeConstraint.GetHashCode());
                hash = Hash.Combine(hash, interfaceConstraints.GetHashCode());
                return hash;
            }

            public override string ToString()
            {
                StringBuilder s = new StringBuilder();
                string attrStr = StringUtil.GenericParameterAttributesToString(attributes);
                s.Append(attrStr);
                bool addComma = (attrStr.Length > 0);
                if (baseTypeConstraint != null)
                {
                    if (addComma) s.Append(',');
                    s.Append(StringUtil.TypeToString(baseTypeConstraint, true));
                    addComma = true;
                }
                foreach (Type face in interfaceConstraints)
                {
                    if (addComma) s.Append(',');
                    s.Append(StringUtil.TypeToString(face, true));
                    addComma = true;
                }
                return s.ToString();
            }
        }

        public static Dictionary<Type, int> GetTypeParameterIndices(IEnumerable<Type> types)
        {
            Dictionary<Type, int> indexOfParam = new Dictionary<Type, int>();
            int count = 0;
            indexOfParam[ThisType] = count++;
            foreach (Type t in types)
            {
                ForEachTypeParameterIncludingConstraints(t, delegate (Type param)
                    {
                        if (!indexOfParam.ContainsKey(param))
                        {
                            indexOfParam[param] = count++;
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    });
            }
            return indexOfParam;
        }

        public static void ForEachTypeParameterIncludingConstraints(Type t, Predicate<Type> visit)
        {
            Binding.ForEachTypeParameter(t, delegate (Type param)
                {
                    if (visit(param))
                    {
                        Type[] constraints = param.GetGenericParameterConstraints();
                        foreach (Type constraint in constraints)
                        {
                            ForEachTypeParameterIncludingConstraints(constraint, visit);
                        }
                    }
                });
        }

        internal static Dictionary<KeyValuePair<string, Constraints>, Type> GenericParameterCache = new Dictionary<KeyValuePair<string, Constraints>, Type>();

        /// <summary>
        /// Make a generic type parameter with the given constraints.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="constraints"></param>
        /// <returns>A generic type parameter.</returns>
        /// <remarks>
        /// Algorithm:
        /// The type parameter is constructed by dynamically generating a type with a generic method.
        /// For example:
        /// <c>class Parent { void method&lt;U&gt;() where U : IList&lt;int&gt; {} }</c>
        /// If the constraints refer to other type parameters, placeholders are added to the parent type.
        /// For example:
        /// <c>class Parent&lt;T&gt; { void method&lt;U&gt;() where U : IList&lt;T&gt; {} }</c>
        /// The parent type is then specialized on the actual type parameters appearing in the constraints.
        /// The type parameter U is returned.
        /// </remarks>
        public static Type MakeGenericParameter(string name, Constraints constraints)
        {
            lock (GenericParameterCache)
            {
                KeyValuePair<string, Constraints> key = new KeyValuePair<string, Constraints>(name, constraints);
                if (GenericParameterCache.TryGetValue(key, out Type result)) return result;
                AssemblyName myAsmName = new AssemblyName("GenericParameterFactory");
                AssemblyBuilder myAssembly = AssemblyBuilder.DefineDynamicAssembly(myAsmName, AssemblyBuilderAccess.Run);
                ModuleBuilder myModule = myAssembly.DefineDynamicModule(myAsmName.Name);
                TypeBuilder parentType = myModule.DefineType("MakeGenericParameter", TypeAttributes.Public);
                //TypeBuilder myType = parentType.DefineNestedType("Child", TypeAttributes.NestedPublic);
                //GenericTypeParameterBuilder typeParam = myType.DefineGenericParameters(name)[0];

                if (true)
                {
                    // CreateType() seems to have a bug when the first method is generic.
                    // class type<t1,t2> { void method<t3>() {} }
                    // comes out as:
                    // class type<t3,t1> { void method<t2>() {} }
                    // therefore we create a dummy first method.
                    MethodBuilder dummy = parentType.DefineMethod("dummy", MethodAttributes.Public | MethodAttributes.Static);
                    dummy.SetParameters();
                    dummy.SetReturnType(typeof(void));
                    ILGenerator ilg = dummy.GetILGenerator();
                    ilg.Emit(OpCodes.Ret);
                }

                if (true)
                {
                    MethodBuilder myMethod = parentType.DefineMethod("method", MethodAttributes.Public | MethodAttributes.Static);
                    Dictionary<Type, int> indexOfParam = GetTypeParameterIndices(constraints.GetEnumerable());
                    string[] names = new string[indexOfParam.Count];
                    foreach (KeyValuePair<Type, int> entry in indexOfParam)
                    {
                        names[entry.Value] = entry.Key.Name;
                    }
                    names[0] = name;
                    GenericTypeParameterBuilder[] typeParams = myMethod.DefineGenericParameters(names);
                    Dictionary<Type, Type> typeMap = new Dictionary<Type, Type>();
                    foreach (KeyValuePair<Type, int> entry in indexOfParam)
                    {
                        typeMap[entry.Key] = typeParams[entry.Value];
                    }
                    // attach constraints to all typeParams
                    foreach (KeyValuePair<Type, int> entry in indexOfParam)
                    {
                        Constraints constraints2;
                        if (entry.Value == 0) constraints2 = constraints;
                        else constraints2 = Constraints.FromTypeParameter(entry.Key, typeMap);
                        Type baseTypeConstraint = Binding.ReplaceTypeParameters(constraints2.baseTypeConstraint, typeMap);
                        List<Type> interfaceConstraints = new List<Type>();
                        foreach (Type face in constraints2.interfaceConstraints)
                        {
                            interfaceConstraints.Add(Binding.ReplaceTypeParameters(face, typeMap));
                        }
                        if (baseTypeConstraint != null) typeParams[entry.Value].SetBaseTypeConstraint(baseTypeConstraint);
                        typeParams[entry.Value].SetInterfaceConstraints(interfaceConstraints.ToArray());
                    }
                    myMethod.SetParameters();
                    myMethod.SetReturnType(typeof(void));
                    ILGenerator ilg = myMethod.GetILGenerator();
                    ilg.Emit(OpCodes.Ret);
                }

                TypeInfo parentInfo = parentType.CreateTypeInfo();
                Type parent = parentInfo.AsType();
                //Type t = myType.CreateType();
                //result = t.GetGenericArguments()[0];
                MethodInfo m = parent.GetMethod("method");
                result = m.GetGenericArguments()[0];
                GenericParameterCache[key] = result;
                return result;
            }
        }

        public static bool IsConstructedParameter(Type typeParam)
        {
            return typeParam.IsGenericParameter && (typeParam.DeclaringType.Name == "MakeGenericParameter");
        }
    }
}