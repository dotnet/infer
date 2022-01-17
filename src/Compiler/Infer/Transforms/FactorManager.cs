// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Reflection;
    using System.Text;
    using Microsoft.ML.Probabilistic.Compiler.Reflection;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler.Attributes;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel;
    using Microsoft.ML.Probabilistic.Algorithms;

    internal class FactorManager
    {
        /// <summary>
        /// Cache of FactorInfo for each factor method.
        /// </summary>
        private static readonly Dictionary<MethodInfo, FactorInfo> InfoOfFactor = new Dictionary<MethodInfo, FactorInfo>();

        /// <summary>
        /// Cache of loaded operator types for each factor method.
        /// </summary>
        private static Dictionary<MethodInfo, List<Type>> OperatorsOfFactor;

        private static readonly List<object> DefaultPriorityList = new List<object>();

        /// <summary>
        /// A list of types, namespaces, modules, and assemblies.  Used to resolve ambiguous matches for message operators.
        /// </summary>
        public List<object> PriorityList;

        public bool Tracing = false;

        public FactorManager()
        {
            lock (padLock)
            {
                if (OperatorsOfFactor == null) BuildCache();
            }
            PriorityList = new List<object>(DefaultPriorityList);
        }

        public void SetTo(FactorManager that)
        {
            PriorityList.Clear();
            PriorityList.AddRange(that.PriorityList);
        }

        private static readonly object padLock = new object();

        public static void ClearMessageFcnCache()
        {
            lock (padLock)
            {
                foreach (FactorInfo info in InfoOfFactor.Values)
                {
                    lock (info.MessageFcns)
                    {
                        info.MessageFcns.Clear();
                    }
                }
            }
        }

        /// <summary>
        /// Resolve ambiguous matches for message operators in favor of the given container.  Accumulates with all previous calls.
        /// </summary>
        /// <param name="container">A Type, namespace string, Module, or Assembly.</param>
        public void GivePriorityTo(object container)
        {
            PriorityList.Insert(0, container);
        }

        /// <summary>
        /// Get the priority of the first matching position in the PriorityList, or 0 if no match.
        /// </summary>
        /// <param name="type"></param>
        /// <returns>PriorityList.Count if the type matches the first entry, PriorityList.Count-1 if the type matches the second entry, and so on down to 0 if no match.</returns>
        public int GetPriority(Type type)
        {
            int count = PriorityList.Count;
            for (int i = 0; i < count; i++)
            {
                if (ContainsType(PriorityList[i], type)) return count - i;
            }
            return 0;
        }

        public static bool ContainsType(object container, Type type)
        {
            if (container is Type containerType)
            {
                if (containerType.IsGenericTypeDefinition)
                    return type.IsGenericType && containerType.IsAssignableFrom(type.GetGenericTypeDefinition());
                else return containerType.IsAssignableFrom(type);
            }
            else if (container is Module) return type.Module.Equals(container);
            else if (container is Assembly) return type.Assembly.Equals(container);
            else if (container is string) return type.Namespace.Equals(container);
            else if (container == null) throw new NullReferenceException("container is null");
            else throw new ArgumentException("Unrecognized container: " + container);
        }

        /// <summary>
        /// Get metadata for a factor method.
        /// </summary>
        /// <param name="factor"></param>
        /// <returns></returns>
        public static FactorInfo GetFactorInfo(Delegate factor)
        {
            return GetFactorInfo(factor.Method);
        }

        /// <summary>
        /// Get metadata for a factor method.
        /// </summary>
        /// <param name="method"></param>
        /// <returns></returns>
        public static FactorInfo GetFactorInfo(MethodInfo method)
        {
            FactorInfo info;
            lock (padLock)
            {
                if (OperatorsOfFactor == null) BuildCache();
                if (!InfoOfFactor.TryGetValue(method, out info))
                {
                    info = new FactorInfo(method);
                    InfoOfFactor[method] = info;
                }
            }
            return info;
        }

        /// <summary>
        /// Get metadata for all factor methods with loaded message functions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<FactorInfo> GetFactorInfos()
        {
            lock (padLock)
            {
                if (OperatorsOfFactor == null) BuildCache();
            }
            foreach (MethodInfo factor in OperatorsOfFactor.Keys)
            {
                yield return GetFactorInfo(factor);
            }
        }

        /// <summary>
        /// Scan all loaded assemblies for message functions.
        /// </summary>
        private static void BuildCache()
        {
            var operatorsOfFactor = new Dictionary<MethodInfo, List<Type>>();
            AppDomain app = AppDomain.CurrentDomain;
            Assembly[] assemblies = app.GetAssemblies();
            foreach (Assembly assembly in assemblies)
            {
                object[] attrs = assembly.GetCustomAttributes(typeof(HasMessageFunctionsAttribute), true);
                if (attrs.Length > 0)
                {
                    // scan all types in the assembly
                    Type[] types = assembly.GetTypes();
                    foreach (Type type in types)
                    {
                        attrs = type.GetCustomAttributes(typeof(FactorMethodAttribute), true);
                        foreach (FactorMethodAttribute attr in attrs)
                        {
                            if (attr.Default) DefaultPriorityList.Add(type);
                            MethodReference mref = MethodReference.FromFactorAttribute(attr);
                            MethodInfo method;
                            try
                            {
                                method = mref.GetMethodInfo();
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine(ex);
                                continue;
                            }
                            //if (method.IsGenericMethod) method = method.GetGenericMethodDefinition();
                            if (!operatorsOfFactor.TryGetValue(method, out var operators))
                            {
                                operators = new List<Type>();
                                operatorsOfFactor[method] = operators;
                            }
                            operators.Add(type);
                        }
                    }
                }
            }
            OperatorsOfFactor = operatorsOfFactor;
        }

        /// <summary>
        /// Wraps an array-typed expression to indicate that all elements of the array are used except at the given index.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public static ListType AllExcept<ListType, IndexType>(ListType list, IndexType index)
        {
            return list;
        }

        public static object Any(params object[] values)
        {
            return values[0];
        }

        /// <summary>
        /// Wraps an array-typed expression to indicate that all elements of the array are non-uniform.
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static object All(object list)
        {
            return list;
        }

        public static object NoTrigger(object arg)
        {
            return arg;
        }

        public static object InducedSource(object arg)
        {
            return arg;
        }

        public static object InducedTarget(object arg)
        {
            return arg;
        }

        private static readonly Dictionary<MethodBase, DependencyInformation> DependencyInfoCache = new Dictionary<MethodBase, DependencyInformation>();

        /// <summary>
        /// Get a list of parameter expressions required by this method.
        /// </summary>
        /// <param name="method">An operator method.</param>
        /// <param name="factorInfo"></param>
        /// <param name="fcnInfo"></param>
        /// <returns>A DependencyInformation where only (Dependencies,Requirements,Triggers) fields are filled in.</returns>
        /// <remarks>
        /// Each dependency expression has one of the following forms:
        /// <list type="bullet">
        /// <item>parameter</item>
        /// <item>parameter[resultIndex]</item>
        /// <item>AllExcept(parameter,resultIndex)</item>
        /// <item>AnyItem(parameter)</item>
        /// <item>AnyItem(AllExcept(parameter,resultIndex))</item>
        /// <item>Any(expr,expr,...)</item>
        /// </list>
        /// </remarks>
        internal static DependencyInformation GetDependencyInfo(MethodBase method, FactorInfo factorInfo = null, MessageFcnInfo fcnInfo = null)
        {
            lock (padLock)
            {
                DependencyInformation info;
                // check the cache
                if (DependencyInfoCache.TryGetValue(method, out info))
                {
                    // return cached value
                    return info;
                }
                // not in cache
                info = new DependencyInformation();
#if TRACE_REFLECTION
            Console.WriteLine("FactorManager.GetDependencyInfo: reflecting on "+StringUtil.MethodFullNameToString(method));
#endif
                CodeBuilder Builder = CodeBuilder.Instance;
                ParameterInfo[] parameters = Util.GetParameters(method);
                ParameterInfo indexParameter = null;
                IExpression resultIndex = null;
                foreach (ParameterInfo parameter in parameters)
                {
                    if (parameter.Name == "resultIndex")
                    {
                        indexParameter = parameter;
                        resultIndex = Builder.ParamRef(Builder.Param(parameter.Name, parameter.ParameterType));
                    }
                }
                Dictionary<string, IExpression> dependencyExpr = new Dictionary<string, IExpression>();
                foreach (ParameterInfo parameter in parameters)
                {
                    IExpression paramRef = Builder.ParamRef(Builder.Param(parameter.Name, parameter.ParameterType));
                    IExpression dependency = paramRef;
                    bool isConstant = false;
                    bool allExceptIndex = false;
                    // Dependency attributes do not stack.  There are only 3 possibilities.
                    if (parameter.IsDefined(typeof(AllExceptIndexAttribute), false))
                    {
                        if (resultIndex == null)
                            throw new InferCompilerException(parameter.Name + " has AllExceptIndexAttribute but " + StringUtil.MethodNameToString(method) +
                                                           " has no resultIndex parameter");
                        dependency = Builder.StaticGenericMethod(
                            new Func<PlaceHolder, PlaceHolder, PlaceHolder>(FactorManager.AllExcept<PlaceHolder, PlaceHolder>),
                            new Type[] { parameter.ParameterType, indexParameter.ParameterType },
                            dependency, resultIndex);
                        allExceptIndex = true;
                    }
                    else if (parameter.IsDefined(typeof(MatchingIndexAttribute), false))
                    {
                        if (resultIndex == null)
                            throw new InferCompilerException(parameter.Name + " has MatchingIndexAttribute but " + StringUtil.MethodNameToString(method) +
                                                           " has no resultIndex parameter");
                        dependency = Builder.ArrayIndex(dependency, resultIndex);
                    }
#if false
                    if (parameter.IsDefined(typeof(InducedSourceAttribute), false))
                    {
                        // this must be the innermost wrapper on dependency
                        dependency = Builder.StaticMethod(new Func<object, object>(FactorManager.InducedSource), dependency);
                    }
                    if (parameter.IsDefined(typeof(InducedTargetAttribute), false))
                    {
                        dependency = Builder.StaticMethod(new Func<object, object>(FactorManager.InducedTarget), dependency);
                    }
#endif
                    if (factorInfo != null && fcnInfo != null)
                    {
                        if (fcnInfo.factorEdgeOfParameter.ContainsKey(parameter.Name))
                        {
                            FactorEdge edge = fcnInfo.factorEdgeOfParameter[parameter.Name];
                            string originalName = edge.ParameterName;
                            if (factorInfo.ParameterTypes.ContainsKey(originalName))
                            {
                                isConstant = factorInfo.ParameterTypes[originalName].IsAssignableFrom(parameter.ParameterType);
                            }
                            if (originalName == fcnInfo.TargetParameter && !allExceptIndex)
                            {
                                // messages to/from target must not have triggers
                                dependency = Builder.StaticMethod(new Func<object, object>(FactorManager.NoTrigger), dependency);
                            }
                        }
                    }
                    if (parameter.IsDefined(typeof(IgnoreDeclarationAttribute), false)) continue;
                    IStatement dependencySt = Builder.ExprStatement(dependency);
                    if ((parameter.Name == "resultIndex") ||
                        (parameter.Name == "result") ||
                        parameter.IsDefined(typeof(IgnoreDependencyAttribute), false) ||
                        parameter.IsOut)
                    {
                        info.Add(DependencyType.Declaration, dependencySt);
                        continue;
                    }
                    // Cancels
                    if (parameter.IsDefined(typeof(CancelsAttribute), false))
                    {
                        info.Add(DependencyType.Cancels, dependencySt);
                        info.Add(DependencyType.NoInit, dependencySt);
                    }
                    info.Add(DependencyType.Dependency, dependencySt);
                    dependencyExpr[parameter.Name] = dependency;

                    // Requirement attributes are more complex since they can stack.
                    if (parameter.IsDefined(typeof(SkipIfAnyUniformAttribute), false))
                    {
                        Type t = dependency.GetExpressionType();
                        IExpression requirement = Util.IsIList(t)
                                                  ? Builder.StaticMethod(new Func<object, object>(FactorManager.All), dependency)
                                                  : dependency;
                        info.Add(DependencyType.SkipIfUniform, Builder.ExprStatement(requirement));
                    }
                    else if (parameter.IsDefined(typeof(SkipIfAllUniformAttribute), false)
                             || parameter.IsDefined(typeof(SkipIfUniformAttribute), false)
                             || parameter.IsDefined(typeof(IsReturnedAttribute), false)
                             || parameter.IsDefined(typeof(IsReturnedInEveryElementAttribute), false)
                        )
                    {
                        info.Add(DependencyType.SkipIfUniform, dependencySt);
                    }
                    else
                    {
                        if (parameter.IsDefined(typeof(SkipIfMatchingIndexIsUniformAttribute), false))
                        {
                            if (resultIndex == null)
                                throw new InferCompilerException(parameter.Name + " has SkipIfMatchingIndexIsUniformAttribute but " + StringUtil.MethodNameToString(method) +
                                                               " has no resultIndex parameter");
                            IExpression requirement = Builder.ArrayIndex(paramRef, resultIndex);
                            info.Add(DependencyType.SkipIfUniform, Builder.ExprStatement(requirement));
                        }
                        if (parameter.IsDefined(typeof(SkipIfAnyExceptIndexIsUniformAttribute), false))
                        {
                            if (resultIndex == null)
                                throw new InferCompilerException(parameter.Name + " has SkipIfAnyExceptIndexIsUniformAttribute but " + StringUtil.MethodNameToString(method) +
                                                               " has no resultIndex parameter");
                            IExpression requirement = Builder.StaticGenericMethod(
                                new Func<PlaceHolder, PlaceHolder, PlaceHolder>(FactorManager.AllExcept<PlaceHolder, PlaceHolder>),
                                new Type[] { parameter.ParameterType, indexParameter.ParameterType },
                                paramRef, resultIndex);
                            requirement = Builder.StaticMethod(new Func<object, object>(FactorManager.All), requirement);
                            info.Add(DependencyType.SkipIfUniform, Builder.ExprStatement(requirement));
                        }
                        else if (parameter.IsDefined(typeof(SkipIfAllExceptIndexAreUniformAttribute), false))
                        {
                            if (resultIndex == null)
                                throw new InferCompilerException(parameter.Name + " has SkipIfAllExceptIndexAreUniformAttribute but " + StringUtil.MethodNameToString(method) +
                                                               " has no resultIndex parameter");
                            IExpression requirement = Builder.StaticGenericMethod(
                                new Func<PlaceHolder, PlaceHolder, PlaceHolder>(FactorManager.AllExcept<PlaceHolder, PlaceHolder>),
                                new Type[] { parameter.ParameterType, indexParameter.ParameterType },
                                paramRef, resultIndex);
                            info.Add(DependencyType.SkipIfUniform, Builder.ExprStatement(requirement));
                        }
                    }

                    // Required
                    if (isConstant
                        || parameter.IsDefined(typeof(RequiredArgumentAttribute), false)
                        || (parameter.IsDefined(typeof(ProperAttribute), false) && !UniformIsProper(parameter.ParameterType))
                        )
                    {
                        info.Add(DependencyType.Requirement, dependencySt);
                    }

                    // Triggers
                    if (parameter.IsDefined(typeof(TriggerAttribute), false)
                        || parameter.IsDefined(typeof(IsReturnedAttribute), false)
                        || parameter.IsDefined(typeof(IsReturnedInEveryElementAttribute), false)
                        )
                    {
                        // default case
                        info.Add(DependencyType.Trigger, dependencySt);
                    }
                    else
                    {
                        if (parameter.IsDefined(typeof(MatchingIndexTriggerAttribute), false))
                        {
                            if (resultIndex == null)
                                throw new InferCompilerException(parameter.Name + " has MatchingIndexTriggerAttribute but " + StringUtil.MethodNameToString(method) +
                                                               " has no resultIndex parameter");
                            IExpression trigger = Builder.ArrayIndex(paramRef, resultIndex);
                            info.Add(DependencyType.Trigger, Builder.ExprStatement(trigger));
                        }
                    }

                    // Fresh
                    if (parameter.IsDefined(typeof(FreshAttribute), false))
                    {
                        //info.Add(DependencyType.Fresh | DependencyType.Requirement, dependencySt);
                        info.Add(DependencyType.Fresh, dependencySt);
                    }

                    // NoInit
                    if (parameter.IsDefined(typeof(NoInitAttribute), false))
                    {
                        if (!info.HasDependency(DependencyType.Requirement, dependencySt))
                        {
                            info.Add(DependencyType.NoInit, dependencySt);
                        }
                    }

                    // Diode
                    if (parameter.IsDefined(typeof(DiodeAttribute), false))
                    {
                        info.Add(DependencyType.Diode, dependencySt);
                    }
                }
                object[] attrs = method.GetCustomAttributes(typeof(SkipIfAllUniformAttribute), true);
                if (method.IsDefined(typeof(MultiplyAllAttribute), true))
                {
                    // MultiplyAll implies SkipIfAllUniform
                    var list = new List<object>(attrs)
                    {
                        new SkipIfAllUniformAttribute()
                    };
                    attrs = list.ToArray();
                }
                foreach (SkipIfAllUniformAttribute attr in attrs)
                {
                    IEnumerable<string> parameterSet = dependencyExpr.Keys;
                    if (attr.ParameterNames != null)
                    {
                        parameterSet = attr.ParameterNames;
                    }
                    else if (info.HasAnyDependencyOfType(DependencyType.SkipIfUniform))
                    {
                        // SkipIfAllUniform is a redundant requirement
                        continue;
                    }
                    List<IExpression> messages = new List<IExpression>();
                    foreach (string parameterName in parameterSet)
                    {
                        IExpression dependency = dependencyExpr[parameterName];
                        Type t = dependency.GetExpressionType();
                        if (!t.IsPrimitive)
                        {
                            messages.Add(dependency);
                        }
                    }
                    IExpression requirement = Builder.StaticMethod(new Func<object[], object>(FactorManager.Any), messages.ToArray());
                    info.Add(DependencyType.SkipIfUniform, Builder.ExprStatement(requirement));
                }
                if (method.IsDefined(typeof(SkipAttribute), false)) info.IsUniform = true;
                if (method.IsDefined(typeof(FreshAttribute), false)) info.IsFresh = true;
                // add to cache
                DependencyInfoCache[method] = info;
                return info;
            }
        }

        private static bool UniformIsProper(Type type)
        {
            // In some cases, we could construct a uniform instance of type and check if it is proper.
            return type.Equals(typeof(Distributions.Bernoulli)) ||
                type.Equals(typeof(Distributions.Beta)) ||
                type.Equals(typeof(Distributions.Dirichlet)) ||
                type.Equals(typeof(Distributions.Discrete)) ||
                type.Equals(typeof(Distributions.DiscreteChar)) ||
                (type.IsGenericType && type.GetGenericTypeDefinition().Equals(typeof(Distributions.DiscreteEnum<>)));
        }

        /// <summary>
        /// Flip the case of the first letter of a string.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string FlipCapitalization(string s)
        {
            if (s.Length > 0 && char.IsLower(s[0])) return char.ToUpper(s[0]) + s.Substring(1);
            else return Uncapitalize(s);
        }

        /// <summary>
        /// Lowercase the first letter of a string.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string Uncapitalize(string s)
        {
            if (s.Length > 0 && char.IsUpper(s[0])) return char.ToLower(s[0]) + s.Substring(1);
            else return s;
        }

        public static MethodInfo GetPointMassMethod(Type distType, Type domainType)
        {
            return (MethodInfo)Invoker.GetBestMethod(
                distType, 
                "PointMass",
                BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy, 
                null,
                new Type[] { domainType }, 
                out Exception exception);
        }

        /// <summary>
        /// True if the type has a FactorMethodAttribute with Default=true
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool IsDefaultOperator(Type type)
        {
            object[] attrs = type.GetCustomAttributes(typeof(FactorMethodAttribute), false);
            foreach (FactorMethodAttribute attr in attrs)
            {
                if (attr.Default) return true;
            }
            return false;
        }

        internal class FactorInfo : ICompilerAttribute
        {
            private static readonly CodeBuilder Builder = CodeBuilder.Instance;

            private static readonly string epEvidenceMethodName = new ExpectationPropagation().GetEvidenceMethodName(new List<ICompilerAttribute>());

            public readonly MethodInfo Method;

            /// <summary>
            /// The name of the return value, followed by the names of the arguments.  For a void function, only the argument names.
            /// </summary>
            public readonly IReadOnlyList<string> ParameterNames;

            /// <summary>
            /// The declared type of a factor argument.
            /// </summary>
            /// <remarks>
            /// Use Type.IsArray to determine if an argument holds an array of messages versus a single message.
            /// </remarks>
            public readonly IReadOnlyDictionary<string, Type> ParameterTypes;

            /// <summary>
            /// Indicates if a factor is deterministic.
            /// </summary>
            /// <remarks>
            /// A factor is deterministic if its return value (and all 'out' parameters) is completely determined by its arguments.
            /// Non-deterministic factors must be annotated with the Stochastic attribute.
            /// </remarks>
            public readonly bool IsDeterministicFactor;

            /// <summary>
            /// True if the factor returns a composite array.
            /// </summary>
            public readonly bool ReturnsCompositeArray;

            /// <summary>
            /// Index of the IsReturnedInAllElements parameter, or -1 if none.
            /// </summary>
            public readonly int ReturnedInAllElementsParameterIndex = -1;

            /// <summary>
            /// True if the factor is a constraint, i.e. has a void return type.
            /// </summary>
            public bool IsVoid
            {
                get { return Method.ReturnType == typeof(void); }
            }

            /// <summary>
            /// Cache of MessageFcnInfo.  Need not be cleared if the PriorityList changes.
            /// </summary>
            public readonly Dictionary<string, MessageFcnInfo> MessageFcns = new Dictionary<string, MessageFcnInfo>();

            /// <summary>
            /// Create an empty FactorInfo structure.
            /// </summary>
            public FactorInfo(MethodInfo method)
            {
                this.Method = method;
                string[] names = null;
                object[] attrs = method.GetCustomAttributes(typeof(ParameterNamesAttribute), false);
                if (attrs.Length > 0)
                {
                    names = ((ParameterNamesAttribute)attrs[0]).Names;
                }
                this.IsDeterministicFactor = !method.IsDefined(typeof(Stochastic), false);
                this.ReturnsCompositeArray = method.IsDefined(typeof(ReturnsCompositeArrayAttribute), false);
                int extraParameterCount = 0;
                bool isVoid = (method.ReturnType == typeof(void));
                List<string> parameterNames = new List<string>();
                Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
                if (!isVoid)
                {
                    string resultField = (names == null) ? Uncapitalize(method.Name) : names[0];
                    parameterNames.Add(resultField);
                    parameterTypes[resultField] = method.ReturnType;
                    extraParameterCount++;
                }
                if (!method.IsStatic)
                {
                    string thisField = (names == null) ? "this" : names[0];
                    parameterNames.Add(thisField);
                    parameterTypes[thisField] = method.DeclaringType;
                    extraParameterCount++;
                }
                ParameterInfo[] parameters = method.GetParameters();
                if (names != null)
                {
                    int numberOfNamesExpected = parameters.Length + extraParameterCount;
                    if (names.Length != numberOfNamesExpected)
                        throw new InferCompilerException("Expected " + numberOfNamesExpected + " names in ParameterNames attribute but got " + names.Length + ", for method " +
                                                       StringUtil.MethodSignatureToString(method));
                }
                for (int i = 0; i < parameters.Length; i++)
                {
                    ParameterInfo parameter = parameters[i];
                    string field = (names == null) ? parameter.Name : names[i + extraParameterCount];
                    parameterNames.Add(field);
                    parameterTypes[field] = parameter.ParameterType;
                    if (parameter.IsDefined(typeof(IsReturnedInEveryElementAttribute), false))
                    {
                        this.ReturnedInAllElementsParameterIndex = i;
                    }
                }
                this.ParameterNames = parameterNames.AsReadOnly();
                this.ParameterTypes = parameterTypes;
            }

            protected bool MethodEquals(MethodInfo method)
            {
                if (method.IsGenericMethod) method = method.GetGenericMethodDefinition();
                MethodInfo factor = this.Method;
                if (factor.IsGenericMethod) factor = factor.GetGenericMethodDefinition();
                return (factor == method);
            }

            /// <summary>
            /// True if the factor is deterministic and all arguments to the factor are deterministic types.
            /// </summary>
            /// <param name="parameterTypes">A mapping from factor arguments to message types.  Missing entries imply no constraint.  Can be null.</param>
            /// <returns></returns>
            /// <remarks>
            /// If the result is true, then GetMessageFcnInfo will generally fail.  Thus it is good to
            /// call this function as a check before calling GetMessageFcnInfo.
            /// </remarks>
            public bool OutputIsDeterministic(IReadOnlyDictionary<string, Type> parameterTypes)
            {
                if (parameterTypes == null) return false;
                if (!IsDeterministicFactor) return false;
                for (int i = (IsVoid ? 0 : 1); i < ParameterNames.Count; i++)
                {
                    string field = ParameterNames[i];
                    Type actualType;
                    if (!parameterTypes.TryGetValue(field, out actualType)) return false;
                    Type formalType = ParameterTypes[field];
                    if (!formalType.IsAssignableFrom(actualType)) return false;
                }
                return true;
            }

            public MessageFcnInfo GetMessageFcnInfoFromFactor()
            {
                ParameterInfo[] parameters = Method.GetParameters();
                int offset = IsVoid ? 0 : 1;
                var factorEdgeOfParameter = new Dictionary<string, FactorEdge>();
                var dependencyInfo = new DependencyInformation();
                for (int i = 0; i < parameters.Length; i++)
                {
                    ParameterInfo parameter = parameters[i];
                    factorEdgeOfParameter[parameter.Name] = new FactorEdge(ParameterNames[i + offset]);
                    IExpression paramRef = Builder.ParamRef(Builder.Param(parameter.Name, parameter.ParameterType));
                    IStatement st = Builder.ExprStatement(paramRef);
                    dependencyInfo.Add(DependencyType.Dependency | DependencyType.Requirement, st);
                }
                return new MessageFcnInfo(Method, parameters, factorEdgeOfParameter)
                {
                    DependencyInfo = dependencyInfo
                };
            }

            /// <summary>
            /// Get metadata for an operator method.
            /// </summary>
            /// <param name="factorManager">Factor manager reference</param>
            /// <param name="methodSuffix">Cannot be null.</param>
            /// <param name="targetParameter">Can be null, to mean all parameters.</param>
            /// <param name="parameterTypes">A mapping from factor arguments to message types.  Missing entries imply no constraint.  Can be null.</param>
            /// <param name="isStochastic">A mapping from factor arguments to bool.  Missing entries imply no constraint.  Can be null.</param>
            /// <returns>A MessageFcnInfo</returns>
            /// <remarks>
            /// This routine caches its results so it tends to be faster than GetMessageFcnInfos.
            /// Besides the factor arguments, <paramref name="parameterTypes"/> should include an entry for the return type ("result").
            /// For composite array arguments, <paramref name="parameterTypes"/> should include an entry for the result index ("resultIndex"). 
            /// </remarks>
            /// <exception cref="ArgumentException">The best matching type parameters did not satisfy the constraints of the generic method.</exception>
            /// <exception cref="MissingMethodException">No match was found.</exception>
            /// <exception cref="NotSupportedException">The message function is not supported (as opposed to simply missing).</exception>
            /// <exception cref="AmbiguousMatchException">More than one method matches the given constraints.</exception>
            public MessageFcnInfo GetMessageFcnInfo(FactorManager factorManager, string methodSuffix, string targetParameter, IReadOnlyDictionary<string, Type> parameterTypes,
                                                    IReadOnlyDictionary<string, bool> isStochastic = null)
            {
                if (factorManager.Tracing)
                {
                    Trace.WriteLine("Entering GetMessageFcnInfo");
                    Trace.WriteLine(StringUtil.JoinColumns("parameterTypes:", StringUtil.VerboseToString(parameterTypes)));
                }
                lock (MessageFcns)
                {
                    // construct the methodKey
                    StringBuilder sb = new StringBuilder();
                    sb.Append(targetParameter);
                    sb.Append(methodSuffix);
                    foreach (KeyValuePair<string, Type> entry in parameterTypes)
                    {
                        sb.Append("(");
                        sb.Append(entry.Key);
                        sb.Append("=");
                        sb.Append(entry.Value);
                        sb.Append(")");
                    }
                    if (isStochastic != null)
                    {
                        foreach (KeyValuePair<string, bool> entry in isStochastic)
                        {
                            sb.Append("(");
                            sb.Append(entry.Key);
                            sb.Append("=");
                            sb.Append(entry.Value);
                            sb.Append(")");
                        }
                    }
                    //string methodKey = targetParameter + methodSuffix + "(" + StringUtil.DictionaryToString(parameterTypes, ",") + ")";
                    string methodKey = sb.ToString();
                    MessageFcnInfo info;
                    if (!MessageFcns.TryGetValue(methodKey, out info))
                    {
                        var operators = this.GetMessageOperators();
                        List<Exception> errors = new List<Exception>();
                        IEnumerable<MessageFunctionBinding> bindings = this.FindMessageFunctions(
                            operators, methodSuffix, targetParameter, GetTypeArguments(this.Method), parameterTypes, isStochastic, errors);
                        List<MessageFunctionBinding> bestBindings = new List<MessageFunctionBinding>();
                        Type declaringType = null;
                        int priority = 0;
                        bool foundMultipleDeclaringTypes = false;
                        foreach (MessageFunctionBinding b in bindings)
                        {
                            if (factorManager.Tracing)
                            {
                                Trace.WriteLine(b);
                            }
                            Type declaringType2 = b.DeclaringType;
                            int priority2 = factorManager.GetPriority(declaringType2);
                            if (bestBindings.Count == 0)
                            {
                                bestBindings.Add(b);
                                declaringType = declaringType2;
                                priority = priority2;
                            }
                            else
                            {
                                if (declaringType2 != declaringType) foundMultipleDeclaringTypes = true;
                                if (priority2 > priority || ((priority == priority2) && (b < bestBindings[0])))
                                {
                                    bestBindings.Clear();
                                    bestBindings.Add(b);
                                    declaringType = declaringType2;
                                    priority = priority2;
                                }
                                else if (priority2 == priority && !(b > bestBindings[0])) bestBindings.Add(b);
                            }
                        }
                        if (bestBindings.Count == 0)
                        {
                            // construct a composite exception
                            throw new ArgumentException((errors.Count == 1) ? errors[0].ToString() : StringUtil.VerboseToString(errors));
                        }
                        if (bestBindings.Count > 1)
                        {
                            // ambiguous match
                            StringBuilder s = new StringBuilder();
                            s.AppendFormat("Looking for {0}, found:", methodKey);
                            foreach (MessageFunctionBinding b in bestBindings)
                            {
                                s.AppendLine();
                                s.Append(StringUtil.MethodSignatureToString(b.MessageFcnInfo.Method));
                            }
                            throw new AmbiguousMatchException(s.ToString());
                        }
                        info = bestBindings[0].MessageFcnInfo;
                        // it is safe to cache this result if all matches occur in only one declaringType.
                        if (!foundMultipleDeclaringTypes) MessageFcns.Add(methodKey, info);
                    }

                    if (info.NotSupportedMessage != null)
                    {
                        string parameterString = (parameterTypes == null) ? string.Empty : (" using parameter types: " + StringUtil.DictionaryToString(parameterTypes, ","));
                        throw new NotSupportedException(info.NotSupportedMessage + parameterString);
                    }
                    return info;
                }
            }

            /// <summary>
            /// Get metadata for all operator methods.
            /// </summary>
            /// <returns>A stream of MessageFcnInfos.</returns>
            public IEnumerable<MessageFcnInfo> GetMessageFcnInfos()
            {
                return GetMessageFcnInfos(null, null, null);
            }

            /// <summary>
            /// Get metadata for multiple operator methods.
            /// </summary>
            /// <param name="methodSuffix">Can be null, to mean all suffixes.</param>
            /// <param name="targetParameter">Can be null, to mean all edges.</param>
            /// <param name="parameterTypes">A mapping from factor arguments to message types.  
            /// Missing entries imply no constraint.  
            /// If parameterTypes is null, the result may contain open type parameters.  
            /// Otherwise the result will not have open type parameters.</param>
            /// <returns>A non-empty stream of MessageFcnInfos that match the given constraints.</returns>
            /// <remarks>
            /// If targetKey is null and parameterTypes is empty, then all operator methods with the given suffix are returned.
            /// The results are not cached.  For fast retrieval of a single message function, use GetMessageFcnInfo instead.
            /// If no message functions match the constraints, one of the below exceptions is thrown.
            /// </remarks>
            /// <exception cref="ArgumentException">The best matching type parameters did not satisfy the constraints of the generic method.</exception>
            /// <exception cref="MissingMethodException">No match was found.</exception>
            /// <exception cref="NotSupportedException">The message function is not supported (as opposed to simply missing).</exception>
            public IEnumerable<MessageFcnInfo> GetMessageFcnInfos(string methodSuffix, string targetParameter, IReadOnlyDictionary<string, Type> parameterTypes)
            {
                var operators = this.GetMessageOperators();
                var errors = new List<Exception>();
                IEnumerable<MessageFunctionBinding> bindings = this.FindMessageFunctions(
                    operators, methodSuffix, targetParameter, GetTypeArguments(this.Method), parameterTypes, null, errors);
                var result = new List<MessageFcnInfo>();
                foreach (MessageFunctionBinding b in bindings)
                {
                    result.Add(b.MessageFcnInfo);
                }

                if (result.Count == 0)
                {
                    throw errors[0];
                }

                return result;
            }

            /// <summary>
            /// Gets the message operators of the method.
            /// </summary>
            /// <returns>The message operators of the method.</returns>
            /// <exception cref="MissingMethodException">The method has no message operators.</exception>
            private IEnumerable<Type> GetMessageOperators()
            {
                MethodInfo method = this.Method;

                // because this is an instance method, the constructor must have filled in OperatorsOfFactor.
                bool hasOperators = OperatorsOfFactor.TryGetValue(method, out var operators);
                if (method.IsGenericMethod && !method.IsGenericMethodDefinition)
                {
                    // if the method is generic, append the operators for its generic definition.
                    // i.e. if the method is GetItem<double> then append the operators for GetItem<T>.
                    method = method.GetGenericMethodDefinition();
                    bool hasOperators2 = OperatorsOfFactor.TryGetValue(method, out var operators2);
                    if (hasOperators2)
                    {
                        if (hasOperators)
                        {
                            operators = new List<Type>(operators);
                            operators.AddRange(operators2);
                        }
                        else
                        {
                            operators = operators2;
                        }
                    }
                }

                if (operators is null)
                {
                    throw new MissingMethodException("Factor " + StringUtil.MethodSignatureToString(this.Method) + " has no registered operators");
                }

                return operators;
            }

            /// <summary>
            /// A stream of MessageFcnInfos matching given constraints.
            /// </summary>
            /// <param name="operators">The types to scan for methods.</param>
            /// <param name="methodSuffix">Can be null, to mean all suffixes.</param>
            /// <param name="targetParameter">Can be null, to mean all parameters.</param>
            /// <param name="typeArguments">A mapping from factor type argument names to types. Missing entries imply no constraint.</param>
            /// <param name="parameterTypes">A mapping from factor parameter names to types.  
            /// Missing entries imply no constraint.  
            /// If parameterTypes is null, the result may contain type parameters of the message function.
            /// Otherwise the result will not have type parameters.</param>
            /// <param name="isStochastic"></param>
            /// <param name="errors">A list that collects matching errors.</param>
            /// <returns>A stream of MessageFcnInfos that match the given constraints.</returns>
            /// <remarks>
            /// If targetKey is null, methodSuffix is null, typeArguments is empty, and parameterTypes is empty, then all methods of
            /// the given operators are returned.
            /// </remarks>
            protected IEnumerable<MessageFunctionBinding> FindMessageFunctions(IEnumerable<Type> operators, string methodSuffix, string targetParameter,
                                                                               IReadOnlyDictionary<string, Type> typeArguments, IReadOnlyDictionary<string, Type> parameterTypes,
                                                                               IReadOnlyDictionary<string, bool> isStochastic, IList<Exception> errors)
            {
                bool tracing = false;
                Type resultType = null;
                if (parameterTypes != null && !parameterTypes.TryGetValue("result", out resultType))
                {
                    parameterTypes.TryGetValue("Result", out resultType);
                }
                bool found = false;
                // Collects error messages to print at the end if there is no match.
                const BindingFlags flags = BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy;
                foreach (Type ttype in operators)
                {
                    Type type = ttype;
                    // factorFields[new parameter name] = original parameter name
                    IDictionary<string, string> originalNameOfTarget = GetNameMapping(GetFactorMethodAttribute(type));
                    string methodName = null, methodNameFlipped = null;
                    string newTarget = targetParameter;
                    if (targetParameter != null)
                    {
                        // reverse map original field into newField
                        string fieldFlipped = FlipCapitalization(targetParameter);
                        foreach (KeyValuePair<string, string> entry in originalNameOfTarget)
                        {
                            if (entry.Value == targetParameter || entry.Value == fieldFlipped)
                            {
                                newTarget = entry.Key;
                                break;
                            }
                        }
                        if (methodSuffix != null)
                        {
                            methodName = newTarget + methodSuffix;
                            methodNameFlipped = FlipCapitalization(methodName);
                        }
                    }
                    int maxParameterCount = ParameterNames.Count; // the number of possible method parameters, updated below
                    Dictionary<string, FactorEdge> factorEdgeOfParameter = new Dictionary<string, FactorEdge>();
                    foreach (KeyValuePair<string, string> entry in originalNameOfTarget)
                    {
                        string newName = entry.Key;
                        string originalName = entry.Value;
                        factorEdgeOfParameter[newName] = new FactorEdge(originalName, false);
                        if (newName.Length == 0) continue;
                        newName = "to_" + newName;
                        factorEdgeOfParameter[newName] = new FactorEdge(originalName, true);
                        if (isStochastic != null)
                        {
                            newName = entry.Key + "_deriv";
                            factorEdgeOfParameter[newName] = new FactorEdge(originalName + "_deriv", false);
                        }
                    }
                    maxParameterCount += ParameterNames.Count;
                    object[] bufferAttrs = type.GetCustomAttributes(typeof(BuffersAttribute), true);
                    foreach (BuffersAttribute bufferAttr in bufferAttrs)
                    {
                        foreach (string bufferName in bufferAttr.BufferNames)
                        {
                            originalNameOfTarget[bufferName] = bufferName;
                            factorEdgeOfParameter[bufferName] = new FactorEdge(bufferName, false);
                            string flippedName = FlipCapitalization(bufferName);
                            originalNameOfTarget[flippedName] = bufferName;
                            factorEdgeOfParameter[flippedName] = new FactorEdge(bufferName, false);
                        }
                    }
                    if (type.IsGenericTypeDefinition)
                    {
                        try
                        {
                            // Bind generic type parameters whose names exactly match the type parameters of the factor.
                            type = MakeGenericType(type, typeArguments);
                        }
                        catch (ArgumentException ex)
                        {
                            errors.Add(ex);
                            continue;
                        }
                    }
                    MethodInfo[] methods = type.GetMethods(flags);
                    foreach (MethodInfo mmethod in methods)
                    {
                        MethodInfo method = mmethod;
                        if (HasAlternateFactorMethodAttribute(method))
                        {
                            if (tracing) Trace.WriteLine("HasAlternateFactorMethodAttribute: " + StringUtil.MethodSignatureToString(method));
                            continue;
                        }
                        string actualSuffix = methodSuffix;
                        string actualNewTarget = newTarget;
                        if (methodName != null)
                        {
                            if (method.Name != methodName && method.Name != methodNameFlipped) continue;
                        }
                        else if (methodSuffix != null)
                        {
                            if (!method.Name.EndsWith(methodSuffix)) continue;
                            // newTargetField == null
                            string methodPrefix = method.Name.Substring(0, method.Name.Length - methodSuffix.Length);
                            if (originalNameOfTarget.ContainsKey(methodPrefix)) actualNewTarget = methodPrefix;
                            else continue;
                        }
                        else
                        {
                            // methodSuffix == null
                            if (newTarget != null)
                            {
                                if (!method.Name.StartsWith(newTarget) && !method.Name.StartsWith(FlipCapitalization(newTarget))) continue;
                            }
                            else
                            {
                                foreach (string newFieldName in originalNameOfTarget.Keys)
                                {
                                    if (newFieldName.Length == 0) continue;
                                    if (method.Name.StartsWith(newFieldName))
                                    {
                                        if (actualNewTarget == null || newFieldName.Length > actualNewTarget.Length)
                                            actualNewTarget = newFieldName;
                                    }
                                }
                                if (actualNewTarget == null)
                                {
                                    actualNewTarget = string.Empty;
                                }
                            }
                            actualSuffix = method.Name.Substring(actualNewTarget.Length, method.Name.Length - actualNewTarget.Length);
                        }
                        if (tracing)
                        {
                            Trace.WriteLine(StringUtil.TypeToString(method.ReturnType) + " " + StringUtil.MethodSignatureToString(method));
                        }
                        // at this point, actualNewTargetField != null, actualSuffix != null
                        Assert.IsTrue(actualNewTarget != null);
                        Assert.IsTrue(actualSuffix != null);
                        ParameterInfo[] parameters = method.GetParameters();
                        Type returnType = resultType;
                        // if caller specified a type for resultIndex, require that it exists.
                        bool passResultIndex = Array.Exists(parameters, parameter => parameter.Name == "resultIndex" || parameter.Name == "ResultIndex");
                        var customParameterTypes = parameterTypes;
                        if (parameterTypes != null && parameterTypes.ContainsKey("resultIndex"))
                        {
                            if (!passResultIndex)
                            {
                                errors.Add(
                                    new ArgumentException("no resultIndex parameter found in method " + StringUtil.MethodFullNameToString(method))
                                    );
                                if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                                continue;
                            }
                        }
                        else if (passResultIndex && resultType != null)
                        {
                            // if the method has a resultIndex parameter that wasn't requested, change the result type
                            returnType = Util.GetElementType(resultType);
                            var parameterTypesWithResultIndex = new Dictionary<string, Type>();
                            parameterTypesWithResultIndex.AddRange(parameterTypes);
                            parameterTypesWithResultIndex["result"] = returnType;
                            parameterTypesWithResultIndex["resultIndex"] = typeof(int);
                            customParameterTypes = parameterTypesWithResultIndex;
                        }
                        // check that Stochastic attributes match desired
                        if (isStochastic != null)
                        {
                            string errorString = GetAttributesErrorString(parameters, isStochastic, factorEdgeOfParameter);
                            if (errorString != null)
                            {
                                errors.Add(new ArgumentException(errorString + " in method " + StringUtil.MethodFullNameToString(method)));
                                if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                                continue;
                            }
                        }
                        Type[] types = GetDesiredParameterTypes(parameters, customParameterTypes, factorEdgeOfParameter);
                        int returnValuePosition = -1;
                        if (targetParameter != null && targetParameter.Length == 0 && !IsVoid && method.Name == epEvidenceMethodName)
                        {
                            // This is an EP evidence method.
                            // Check that one of the parameters corresponds to the return value.
                            string returnValue = ParameterNames[0];
                            for (int i = 0; i < parameters.Length; i++)
                            {
                                var parameter = parameters[i];
                                if (factorEdgeOfParameter.TryGetValue(parameter.Name, out FactorEdge edge) &&
                                    edge.ParameterName == returnValue && !edge.IsOutgoingMessage)
                                {
                                    returnValuePosition = i;
                                }
                            }
                            if (returnValuePosition == -1)
                            {
                                errors.Add(new ArgumentException("'" + returnValue + "' is not an argument of " + StringUtil.MethodSignatureToString(method)));
                                if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                                continue;
                            }
                        }
                        // try to infer the method-specific type parameters
                        bool canConvertToPointMass(Type fromType, Type toType, int position)
                        {
                            // For an EP evidence method, do not allow point mass conversion of the return value.
                            if (position == returnValuePosition) return false;
                            bool isDomainType = Distributions.Distribution.IsDistributionType(toType) &&
                                                Distributions.Distribution.GetDomainType(toType).IsAssignableFrom(fromType);
                            if (isDomainType)
                            {
                                MethodInfo pointMassMethod = GetPointMassMethod(toType, fromType);
                                return (pointMassMethod != null);
                            }
                            else return false;
                        }
                        ConversionOptions conversionOptions = new ConversionOptions
                        {
                            AllowImplicitConversions = true,
                            IsImplicitConversion = canConvertToPointMass
                        };
                        Binding binding = Binding.GetBestBinding(method, types, conversionOptions, out Exception matchException);
                        if (binding == null)
                        {
                            errors.Add(matchException);
                            if (tracing) Trace.WriteLine(matchException);
                            continue;
                        }
                        //method = (MethodInfo)Invoker.GetBestMethod(new MethodBase[] { method }, null, types, out matchException);
                        //if (method == null) continue;
                        try
                        {
                            method = (MethodInfo)binding.Bind(method);
                        }
                        catch (Exception ex)
                        {
                            errors.Add(ex);
                            if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                            continue;
                        }

                        // check return type
                        if (returnType != null)
                        {
                            returnType = binding.Bind(returnType);
                            if ((returnType != null) && !returnType.IsAssignableFrom(method.ReturnType))
                            {
                                errors.Add(
                                    new ArgumentException(StringUtil.TypeToString(returnType) + " is not assignable from " + StringUtil.TypeToString(method.ReturnType)
                                                          + " for result of method " + StringUtil.MethodFullNameToString(method))
                                    );
                                if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                                continue;
                            }
                        }
                        // method has matched.  create a MessageFcnInfo.
                        MessageFcnInfo info = new MessageFcnInfo(method, parameters, factorEdgeOfParameter);
                        info.Suffix = actualSuffix;
                        info.TargetParameter = originalNameOfTarget.ContainsKey(actualNewTarget) ? originalNameOfTarget[actualNewTarget] : actualNewTarget;
                        try
                        {
                            info.Dependencies = GetDependencies(factorEdgeOfParameter, info.Method, out info.Requirements, out info.SkipIfAllUniform, out info.Triggers,
                                                                out info.NoTriggers);
                        }
                        catch (ArgumentException ex)
                        {
                            errors.Add(ex);
                            if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                            continue;
                        }
                        try
                        {
                            info.DependencyInfo = FactorManager.GetDependencyInfo(info.Method, this, info);
                            //info.DependencyInfo = GetDependencyInfo(info.Method, info);
                        }
                        catch (ArgumentException ex)
                        {
                            errors.Add(ex);
                            if (tracing) Trace.WriteLine(errors[errors.Count - 1]);
                            continue;
                        }
                        found = true;
                        var conversionWeight = 0.0F; // (float)binding.Types.Count/100;
                        foreach (Conversion c in binding.Conversions)
                        {
                            conversionWeight += c.GetWeight() + 1;
                        }
                        MessageFunctionBinding b = new MessageFunctionBinding
                        {
                            MessageFcnInfo = info,
                            ConversionWeight = conversionWeight,
                            DeclaringType = type
                        };
                        yield return b;
                    }
                }
                if (!found)
                {
                    if (errors.Count == 0)
                    {
                        string parameterString = (parameterTypes == null) ? string.Empty : (" using parameter types: " + StringUtil.DictionaryToString(parameterTypes, ","));
                        errors.Add(
                            new MissingMethodException(targetParameter + methodSuffix + " not found in " + StringUtil.CollectionToString(operators, ",") + parameterString)
                            );
                    }
                }
            }

            internal class MessageFunctionBinding
            {
                public MessageFcnInfo MessageFcnInfo;
                public float ConversionWeight;
                public Type DeclaringType;

                public static bool operator <(MessageFunctionBinding a, MessageFunctionBinding b)
                {
                    return (a.ConversionWeight < b.ConversionWeight);
                }

                public static bool operator >(MessageFunctionBinding a, MessageFunctionBinding b)
                {
                    return (b < a);
                }

                public override string ToString()
                {
                    return StringUtil.JoinColumns("MessageFunctionBinding(", ConversionWeight, ", ", MessageFcnInfo, ")");
                }
            }

            /// <summary>
            /// Get a mapping from type parameter names to type arguments.
            /// </summary>
            /// <param name="method"></param>
            /// <returns>empty if the method is a generic method definition.</returns>
            public static IReadOnlyDictionary<string, Type> GetTypeArguments(MethodInfo method)
            {
                Dictionary<string, Type> result = new Dictionary<string, Type>();
                if (method.IsGenericMethod && !method.IsGenericMethodDefinition)
                {
                    MethodInfo defn = method.GetGenericMethodDefinition();
                    Type[] argNames = defn.GetGenericArguments();
                    Type[] args = method.GetGenericArguments();
                    for (int i = 0; i < argNames.Length; i++)
                    {
                        result[argNames[i].Name] = args[i];
                    }
                }
                return result;
            }

            /// <summary>
            /// Get a mapping from type parameter names to type arguments.
            /// </summary>
            /// <param name="type"></param>
            /// <returns></returns>
            public static IDictionary<string, Type> GetTypeArguments(Type type)
            {
                Dictionary<string, Type> result = new Dictionary<string, Type>();
                if (type.IsGenericType)
                {
                    Type defn = type.GetGenericTypeDefinition();
                    Type[] argNames = defn.GetGenericArguments();
                    Type[] args = type.GetGenericArguments();
                    for (int i = 0; i < argNames.Length; i++)
                    {
                        result[argNames[i].Name] = args[i];
                    }
                }
                return result;
            }

            /// <summary>
            /// Same as <see cref="Type.MakeGenericType"/> except the typeArguments are taken by name from a dictionary.
            /// </summary>
            /// <param name="type"></param>
            /// <param name="typeArguments"></param>
            /// <returns>The instantiated Type.</returns>
            /// <exception cref="InvalidOperationException"><paramref name="type"/> is not a generic type definition.</exception>
            /// <exception cref="ArgumentException">A type argument does not satisfy the constraints of the generic type.</exception>
            public static Type MakeGenericType(Type type, IReadOnlyDictionary<string, Type> typeArguments)
            {
                Type[] args = type.GetGenericArguments();
                for (int i = 0; i < args.Length; i++)
                {
                    Type arg = args[i];
                    if (arg.IsGenericParameter && typeArguments.ContainsKey(arg.Name))
                        args[i] = typeArguments[arg.Name];
                }
                return type.MakeGenericType(args);
            }

            /// <summary>
            /// Same as <see cref="MethodInfo.MakeGenericMethod"/> except the typeArguments are taken by name from a dictionary.
            /// </summary>
            /// <param name="method">The method info</param>
            /// <param name="typeArguments">The type arguments</param>
            /// <returns></returns>
            public static MethodInfo MakeGenericMethod(MethodInfo method, IDictionary<string, Type> typeArguments)
            {
                Type[] args = Array.ConvertAll(method.GetGenericArguments(), delegate (Type arg)
                {
                    if (!arg.IsGenericParameter) return arg;
                    else if (typeArguments.TryGetValue(arg.Name, out Type bound)) return bound;
                    else return arg;
                });
                return method.MakeGenericMethod(args);
            }

            /// <summary>
            /// Map parameter names to types according to a dictionary.
            /// </summary>
            /// <param name="parameters"></param>
            /// <param name="parameterTypes">Mapping from original factor parameters to Types.  If null, assumed to be exactly the parameter types.</param>
            /// <param name="factorEdgeOfParameter">Mapping from new parameter names to original factor parameter names.</param>
            /// <returns>An array of types, in the same order as the parameters.</returns>
            /// <remarks>
            /// The result array will always have the same length as <paramref name="parameters"/>.
            /// It will contain null when the desired parameter type is unknown.
            /// </remarks>
            protected static Type[] GetDesiredParameterTypes(ParameterInfo[] parameters, IReadOnlyDictionary<string, Type> parameterTypes,
                                                             IReadOnlyDictionary<string, FactorEdge> factorEdgeOfParameter)
            {
                Type[] types = new Type[parameters.Length];
                for (int i = 0; i < parameters.Length; i++)
                {
                    string originalName = parameters[i].Name;
                    FactorEdge edge;
                    if ((factorEdgeOfParameter != null) &&
                        factorEdgeOfParameter.TryGetValue(originalName, out edge))
                    {
                        originalName = edge.ToString();
                    }
                    if (parameterTypes == null) types[i] = parameters[i].ParameterType;
                    else if (!parameterTypes.TryGetValue(originalName, out types[i]) &&
                             !parameterTypes.TryGetValue(FlipCapitalization(originalName), out types[i]))
                    {
                        // do nothing
                        // types[i] will be null.
                    }
                    else if (parameters[i].IsDefined(typeof(IndexedAttribute), false))
                    {
                        types[i] = Util.GetElementType(types[i]);
                    }
                }
                return types;
            }

            protected string GetAttributesErrorString(ParameterInfo[] parameters, IReadOnlyDictionary<string, bool> isStochastic, IReadOnlyDictionary<string, FactorEdge> factorEdgeOfParameter)
            {
                for (int i = 0; i < parameters.Length; i++)
                {
                    string originalName = parameters[i].Name;
                    if ((factorEdgeOfParameter != null) && factorEdgeOfParameter.TryGetValue(originalName, out FactorEdge edge))
                    {
                        originalName = edge.ToString();
                    }
                    bool wantStochastic;
                    if (isStochastic.TryGetValue(originalName, out wantStochastic) ||
                        isStochastic.TryGetValue(FlipCapitalization(originalName), out wantStochastic))
                    {
                        ParameterInfo parameter = parameters[i];
                        bool hasStochasticAttr = parameter.IsDefined(typeof(Stochastic), true);
                        if (hasStochasticAttr && !wantStochastic)
                        {
                            return "'" + originalName + "' must not be a derived variable";
                        }
                    }
                }
                return null;
            }

            /// <summary>
            /// Get the FactorMethodAttribute matching this factor
            /// </summary>
            /// <param name="op"></param>
            /// <returns></returns>
            public FactorMethodAttribute GetFactorMethodAttribute(MemberInfo op)
            {
                object[] attrs = op.GetCustomAttributes(typeof(FactorMethodAttribute), true);
                foreach (FactorMethodAttribute attr in attrs)
                {
                    MethodInfo method;
                    try
                    {
                        method = MethodReference.FromFactorAttribute(attr).GetMethodInfo();
                    }
                    catch (Exception)
                    {
                        continue;
                    }
                    if (this.MethodEquals(method))
                        return attr;
                }
                return null;
            }

            /// <summary>
            /// True if the member has a FactorMethodAttribute and none match this factor
            /// </summary>
            /// <param name="op"></param>
            /// <returns></returns>
            public bool HasAlternateFactorMethodAttribute(MemberInfo op)
            {
                object[] attrs = op.GetCustomAttributes(typeof(FactorMethodAttribute), true);
                foreach (FactorMethodAttribute attr in attrs)
                {
                    MethodInfo method;
                    try
                    {
                        method = MethodReference.FromFactorAttribute(attr).GetMethodInfo();
                    }
                    catch (Exception)
                    {
                        continue;
                    }
                    if (this.MethodEquals(method))
                        return false;
                }
                return attrs.Length > 0;
            }

            /// <summary>
            /// Get a mapping from the new parameter names used in the operator to the original parameter names of the factor.
            /// </summary>
            /// <param name="attr"></param>
            /// <returns></returns>
            protected IDictionary<string, string> GetNameMapping(FactorMethodAttribute attr)
            {
                string[] newParameterNames = attr.NewParameterNames;
                Dictionary<string, string> map = new Dictionary<string, string>
                {
                    // always include the empty string
                    [string.Empty] = string.Empty
                };
                if (newParameterNames == null)
                {
                    // map the fields to themselves
                    for (int i = 0; i < ParameterNames.Count; i++)
                    {
                        map[ParameterNames[i]] = ParameterNames[i];
                        map[FlipCapitalization(ParameterNames[i])] = ParameterNames[i];
                    }
                }
                else
                {
                    if (newParameterNames.Length != ParameterNames.Count)
                        throw new Exception("got " + newParameterNames.Length + " parameter names but expected " + ParameterNames.Count + " for " +
                                            StringUtil.MethodSignatureToString(Method));
                    for (int i = 0; i < ParameterNames.Count; i++)
                    {
                        map[newParameterNames[i]] = ParameterNames[i];
                        map[FlipCapitalization(newParameterNames[i])] = ParameterNames[i];
                    }
                }
                return map;
            }

            /// <summary>
            /// Get a list of parameter expressions required by this method.
            /// </summary>
            /// <param name="method">An operator method.</param>
            /// <param name="fcnInfo"></param>
            /// <returns>A DependencyInformation where only (Dependencies,Requirements,Triggers) fields are filled in.</returns>
            /// <remarks>
            /// Each dependency expression has one of the following forms:
            /// <list type="bullet">
            /// <item>parameter</item>
            /// <item>parameter[resultIndex]</item>
            /// <item>AllExcept(parameter,resultIndex)</item>
            /// <item>AnyItem(parameter)</item>
            /// <item>AnyItem(AllExcept(parameter,resultIndex))</item>
            /// <item>Any(expr,expr,...)</item>
            /// </list>
            /// </remarks>
            internal DependencyInformation GetDependencyInfo(MethodInfo method, MessageFcnInfo fcnInfo)
            {
                lock (padLock)
                {
                    DependencyInformation info = FactorManager.GetDependencyInfo(method, this, fcnInfo);
#if TRACE_REFLECTION
                    Console.WriteLine("FactorInfo.GetDependencyInfo: reflecting on "+StringUtil.MethodFullNameToString(method));
#endif
                    CodeBuilder Builder = CodeBuilder.Instance;
                    ParameterInfo[] parameters = method.GetParameters();
                    foreach (ParameterInfo parameter in parameters)
                    {
                        if (!fcnInfo.factorEdgeOfParameter.ContainsKey(parameter.Name)) continue;
                        string originalName = fcnInfo.factorEdgeOfParameter[parameter.Name].ParameterName;
                        if (!ParameterTypes.ContainsKey(originalName)) continue;
                        bool isConstant = this.ParameterTypes[originalName].IsAssignableFrom(parameter.ParameterType);
                        if (isConstant)
                        {
                            // constant arguments are automatically Required
                            IExpression paramRef = Builder.ParamRef(Builder.Param(parameter.Name, parameter.ParameterType));
                            info.Add(DependencyType.Requirement, Builder.ExprStatement(paramRef));
                        }
                    }
                    return info;
                }
            }

            /// <summary>
            /// Collect the FactorEdges that this message function depends on.
            /// </summary>
            /// <param name="factorEdgeOfParameter">A mapping from message function parameter names to factor parameter names.</param>
            /// 
            /// <param name="method">The message function.</param>
            /// <param name="requirements">On return, a list of FactorEdges that must be non-uniform.</param>
            /// <param name="skipIfAllUniform"></param>
            /// <param name="triggers"></param>
            /// <param name="noTriggers"></param>
            /// <returns>FactorEdges listed in the order of the parameters to the method.</returns>
            /// <remarks>
            /// Factor parameter dependencies are detected via the parameter names of the method.
            /// Array index dependencies are specified via parameter attributes.
            /// </remarks>
            private IReadOnlyList<FactorEdge> GetDependencies(
                IReadOnlyDictionary<string, FactorEdge> factorEdgeOfParameter,
                MethodInfo method,
                out IReadOnlyList<FactorEdge> requirements,
                out bool skipIfAllUniform,
                out IReadOnlyList<FactorEdge> triggers,
                out bool noTriggers)
            {
                var dependencies = new List<FactorEdge>();
                var requirementsList = new List<FactorEdge>();
                var triggersList = new List<FactorEdge>();
#if TRACE_REFLECTION
            Console.WriteLine("FactorInfo.GetDependencies: reflecting on "+StringUtil.MethodFullNameToString(method));
#endif
                ParameterInfo[] parameters = method.GetParameters();
                foreach (ParameterInfo parameter in parameters)
                {
                    string parameterName = parameter.Name;
                    bool match = factorEdgeOfParameter.ContainsKey(parameterName);
                    if (!match)
                    {
                        parameterName = FlipCapitalization(parameterName);
                        match = factorEdgeOfParameter.ContainsKey(parameterName);
                    }
                    if (match)
                    {
                        FactorEdge range = new FactorEdge(factorEdgeOfParameter[parameterName]);
                        // Dependency attributes do not stack.  There are only 3 possibilities.
                        if (parameter.GetCustomAttributes(typeof(AllExceptIndexAttribute), true).Length > 0)
                        {
                            range.ContainsIndex = false;
                        }
                        else if (parameter.GetCustomAttributes(typeof(MatchingIndexAttribute), true).Length > 0)
                        {
                            range.ContainsOthers = false;
                            range.MinCount = 1;
                        }
                        dependencies.Add(range);

                        // Requirement attributes are more complex since they can stack.
                        bool isConstant = this.ParameterTypes.ContainsKey(range.ParameterName) &&
                                          this.ParameterTypes[range.ParameterName].IsAssignableFrom(parameter.ParameterType);
                        FactorEdge requiredRange = new FactorEdge(range.ParameterName);
                        requiredRange.IsOutgoingMessage = range.IsOutgoingMessage;
                        if (isConstant
                            || parameter.GetCustomAttributes(typeof(SkipIfUniformAttribute), true).Length > 0
                            || parameter.GetCustomAttributes(typeof(ProperAttribute), true).Length > 0)
                        {
                            // default case
                        }
                        else
                        {
                            requiredRange.IsEmpty = true;
                            if (parameter.GetCustomAttributes(typeof(SkipIfAllUniformAttribute), true).Length > 0)
                            {
                                requiredRange.MinCount = 1;
                            }
                            if (parameter.GetCustomAttributes(typeof(SkipIfAllExceptIndexAreUniformAttribute), true).Length > 0)
                            {
                                requiredRange.ContainsOthers = true;
                            }
                            if (parameter.GetCustomAttributes(typeof(SkipIfAnyExceptIndexIsUniformAttribute), true).Length > 0)
                            {
                                requiredRange.ContainsAllOthers = true;
                            }
                            if (parameter.GetCustomAttributes(typeof(SkipIfMatchingIndexIsUniformAttribute), true).Length > 0)
                            {
                                requiredRange.ContainsIndex = true;
                            }
                        }
                        requiredRange = requiredRange.Intersect(range);
                        if (!requiredRange.IsEmpty) requirementsList.Add(requiredRange);

                        // Triggers
                        FactorEdge triggerRange = new FactorEdge(range.ParameterName);
                        requiredRange.IsOutgoingMessage = range.IsOutgoingMessage;
                        if (parameter.GetCustomAttributes(typeof(TriggerAttribute), true).Length > 0)
                        {
                            // default case
                        }
                        else
                        {
                            triggerRange.IsEmpty = true;
                            if (parameter.GetCustomAttributes(typeof(MatchingIndexTriggerAttribute), true).Length > 0)
                            {
                                triggerRange.ContainsIndex = true;
                            }
                        }
                        triggerRange = triggerRange.Intersect(range);
                        if (!triggerRange.IsEmpty) triggersList.Add(triggerRange);
                    }
                    else
                    {
                        parameterName = Uncapitalize(parameterName);
                        if (parameterName != "result" && parameterName != "resultIndex")
                        {
                            //throw new ArgumentException("Unrecognized argument named '" + parameterName + "' in operator method " + StringUtil.MethodFullNameToString(method));
                        }
                        // ignore extra parameters
                    }
                }
                skipIfAllUniform = (requirementsList.Count > 0) ||
                    (method.GetCustomAttributes(typeof(SkipIfAllUniformAttribute), true).Length > 0);
                noTriggers = (method.GetCustomAttributes(typeof(NoTriggersAttribute), true).Length > 0);
                Assert.IsTrue(dependencies.Count >= requirementsList.Count);
                requirements = requirementsList;
                triggers = triggersList;
                return dependencies;
            }

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            public override string ToString()
            {
                System.Text.StringBuilder s = new System.Text.StringBuilder();
                s.AppendFormat("{0}(", StringUtil.MethodFullNameToString(Method));
                bool firstTime = true;
                foreach (string field in ParameterNames)
                {
                    if (!firstTime) s.Append(", ");
                    s.AppendFormat("{0} {1}", StringUtil.TypeToString(ParameterTypes[field]), field);
                    firstTime = false;
                }
                s.Append(")");
                return s.ToString();
            }
        }
    }

    /// <summary>
    /// Represents a range of arguments to a factor.
    /// </summary>
    /// <remarks><para>
    /// Depending on which fields are non-null, this class represents a variety of different types of factor edge.
    /// The simplest type is the name of a parameter.  For an array parameter, this implies all elements of the array.
    /// The second type is "Param[i]", a specific element of an array parameter.
    /// The third type is "Param-[i]", which represents all elements of an array except one.
    /// </para></remarks>
    internal struct FactorEdge
    {
        /// <summary>
        /// The name of a parameter to the factor function, or its return value.  Note that the message "to_x" has ParameterName="x" and IsOutgoingMessage=true.
        /// </summary>
        public string ParameterName;

        /// <summary>
        /// True if the argument is of type "to_parameter".
        /// </summary>
        public bool IsOutgoingMessage;

        /// <summary>
        /// True if the range definitely includes items other than Index.
        /// </summary>
        private bool containsOthers;

        /// <summary>
        /// True if the range definitely includes the item at Index.
        /// </summary>
        private bool containsIndex;

        /// <summary>
        /// The minimum number of items in the range.
        /// </summary>
        /// <remarks>
        /// MinCount must be >= ContainsIndex + ContainsOthers.
        /// If ContainsOthers == false, MinCount can only be 0 or 1.
        /// MinCount can be 1 even if ContainsOthers == false and ContainsIndex == false.
        /// If ContainsOthers == true, MinCount == int.MaxValue means that all items other than Index are in the range.
        /// </remarks>
        public int MinCount;

        public bool IsEmpty
        {
            get { return (MinCount == 0); }
            set
            {
                if (value)
                {
                    ContainsOthers = false;
                    ContainsIndex = false;
                    MinCount = 0;
                }
                else
                {
                    throw new NotSupportedException();
                }
            }
        }

        /// <summary>
        /// True if the range definitely includes all items other than Index.
        /// </summary>
        public bool ContainsAllOthers
        {
            get { return (MinCount == int.MaxValue); }
            set
            {
                if (value)
                {
                    MinCount = int.MaxValue;
                    ContainsOthers = true;
                }
                else
                {
                    MinCount = (ContainsOthers ? 1 : 0) + (ContainsIndex ? 1 : 0);
                }
            }
        }

        /// <summary>
        /// True if the range definitely includes items other than Index.
        /// </summary>
        public bool ContainsOthers
        {
            get { return containsOthers; }
            set
            {
                containsOthers = value;
                RaiseMinCount();
            }
        }

        /// <summary>
        /// True if the range definitely includes the item at Index.
        /// </summary>
        public bool ContainsIndex
        {
            get { return containsIndex; }
            set
            {
                containsIndex = value;
                RaiseMinCount();
            }
        }

        public void RaiseMinCount()
        {
            int minMinCount = (ContainsOthers ? 1 : 0) + (ContainsIndex ? 1 : 0);
            if (MinCount < minMinCount) MinCount = minMinCount;
        }

        /// <summary>
        /// Create a range over all items of the parameter.
        /// </summary>
        /// <param name="parameterName"></param>
        public FactorEdge(string parameterName)
        {
            ParameterName = parameterName;
            IsOutgoingMessage = false;
            containsIndex = true;
            containsOthers = true;
            MinCount = int.MaxValue;
        }

        public FactorEdge(string parameterName, bool isOutgoingMessage)
            : this(parameterName)
        {
            this.IsOutgoingMessage = isOutgoingMessage;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public FactorEdge(FactorEdge that)
        {
            ParameterName = that.ParameterName;
            IsOutgoingMessage = that.IsOutgoingMessage;
            containsIndex = that.ContainsIndex;
            containsOthers = that.ContainsOthers;
            MinCount = that.MinCount;
        }

        public FactorEdge Intersect(FactorEdge that)
        {
            FactorEdge result = new FactorEdge(this);
            result.ContainsOthers = this.ContainsOthers && that.ContainsOthers;
            result.ContainsIndex = this.ContainsIndex && that.ContainsIndex;
            result.MinCount = Math.Min(this.MinCount, that.MinCount);
            return result;
        }

        public override string ToString()
        {
            string prefix = IsOutgoingMessage ? "to_" : string.Empty;
            prefix += ParameterName;
            if (ContainsIndex)
            {
                if (ContainsAllOthers)
                {
                    return prefix;
                }
                else if (!ContainsOthers)
                {
                    return prefix + "[" + IndexToString() + "]";
                }
                else
                {
                    return prefix + "[" + IndexToString() + "]+" + (MinCount - 1);
                }
            }
            else
            {
                if (ContainsAllOthers)
                {
                    return prefix + "-[" + IndexToString() + "]";
                }
                else if (!ContainsOthers)
                {
                    if (MinCount > 0) return prefix + "*" + MinCount;
                    else return string.Empty;
                }
                else
                {
                    return prefix + "-[" + IndexToString() + "]*" + MinCount;
                }
            }
        }

        public string IndexToString()
        {
            return "i";
        }
    }
}