// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler;
using System.Reflection;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Collections.ObjectModel;
using System.IO;
using System.Collections.Concurrent;
using System.Linq;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// An inference engine, used to perform inference tasks in Infer.NET.
    /// </summary>
    /// <remarks>
    /// The Debug class may be used to get debug messages for the inference engine.
    /// For example, use <code>Debug.Listeners.Add(new TextWriterTraceListener(Console.Out));</code>
    /// to get debug information for when compiled models and marginals are re-used.
    /// </remarks>
    public class InferenceEngine : SettableTo<InferenceEngine>
    {
        /// <summary>
        /// Bag of weak references to engine instances.  The weak references allow the instances
        /// to be garbage collected when they are no longer being used.
        /// </summary>
        private static readonly ConcurrentDictionary<WeakReference, EmptyStruct> allEngineInstances = new ConcurrentDictionary<WeakReference, EmptyStruct>();

        protected struct EmptyStruct
        {
        }

        /// <summary>
        /// Internal list of built-in algorithms
        /// </summary>
        private static readonly IAlgorithm[] algs =
            {
                new Algorithms.ExpectationPropagation(), new Algorithms.VariationalMessagePassing(),
                new Algorithms.GibbsSampling(), new Algorithms.MaxProductBeliefPropagation()
            };

        /// <summary>
        /// Default inference engine whose settings will be copied onto newly created engines.
        /// </summary>
        public static readonly InferenceEngine DefaultEngine = new InferenceEngine(false);

        private static bool IsUnitTest()
        {
            // FriendlyName works for VS 2005 but not VS 2008
            if (AppDomain.CurrentDomain.FriendlyName.Contains("UnitTest")) return true;
            Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
            foreach (Assembly assembly in assemblies)
            {
                if (assembly.GetName().Name == "Microsoft.VisualStudio.QualityTools.AgentObject") return true;
            }
            return false;
        }

        static InferenceEngine()
        {
            DefaultEngine.ResetOnObservedValueChanged = true;
        }

        /// <summary>
        /// The full name of the inference engine, including version
        /// </summary>
        public static string Name
        {
            get { return "Infer.NET " + Assembly.GetExecutingAssembly().GetName().Version; }
        }

        /// <summary>
        /// Model namespace, used when naming generated classes.
        /// </summary>
        public string ModelNamespace { get; set; } = ModelBuilder.ModelNamespace;

        /// <summary>
        /// Model name, used when naming generated classes.
        /// </summary>
        public string ModelName { get; set; } = "Model";

        /// <summary>
        /// Provides the implementation of ShowFactorGraph, ShowSchedule, and BrowserMode.
        /// </summary>
        public static Visualizer Visualizer { get; set; } = new DefaultVisualizer();

        /// <summary>
        /// The ModelBuilder used to construct MSL from in-memory graphs of Variables etc.
        /// </summary>
        private readonly ModelBuilder mb = new ModelBuilder();

        private readonly ConcurrentStack<CompiledAlgorithmInfo> compiledAlgorithms = new ConcurrentStack<CompiledAlgorithmInfo>();
        private readonly Dictionary<IVariable, CompiledAlgorithmInfo> compiledAlgorithmForVariable = new Dictionary<IVariable, CompiledAlgorithmInfo>();

        private class CompiledAlgorithmInfo
        {
            public IGeneratedAlgorithm exec;
            public List<Variable> observedVarsInOrder = new List<Variable>();
            public Set<Variable> observedVars = new Set<Variable>();

            public CompiledAlgorithmInfo(IGeneratedAlgorithm exec, IEnumerable<Variable> observedVars)
            {
                this.exec = exec;
                this.observedVarsInOrder.AddRange(observedVars);
                this.observedVars.AddRange(observedVars);
            }
        }

        /// <summary>
        /// The Compiler used to compile MSL into a compiled algorithm.
        /// </summary>
        protected ModelCompiler compiler;

        /// <summary>
        /// Creates an inference engine which uses the default inference algorithm
        /// (currently this is expectation propagation).
        /// </summary>
        public InferenceEngine()
            : this(true)
        {
            allEngineInstances.TryAdd(new WeakReference(this), new EmptyStruct());
        }

        /// <summary>
        /// Creates an inference engine which uses the specified inference algorithm.
        /// </summary>
        public InferenceEngine(IAlgorithm algorithm)
            : this()
        {
            this.Algorithm = algorithm;
        }

        /// <summary>
        /// Create a new ModelCompiler object
        /// </summary>
        private void CreateCompiler()
        {
            compiler = new ModelCompiler();
            compiler.ParametersChanged += delegate (object sender, EventArgs e) { InvalidateCompiledAlgorithms(); };
            compiler.Compiling += delegate (ModelCompiler sender, ModelCompiler.CompileEventArgs e) { if (ShowProgress) Console.Write("Compiling model..."); };
            compiler.Compiled += delegate (ModelCompiler sender, ModelCompiler.CompileEventArgs e)
                {
                    if (ShowWarnings && e.Warnings != null && (e.Warnings.Count > 0))
                    {
                        Console.WriteLine("compilation had " + e.Warnings.Count + " warning(s).");
                        int count = 1;
                        foreach (TransformError te in e.Warnings)
                        {
                            if (!te.IsWarning)
                                continue;
                            Console.WriteLine("  [" + count + "] " + te.ErrorText);
                            count++;
                        }
                    }
                    if (e.Exception != null)
                    {
                        if (ShowProgress) Console.WriteLine("compilation failed.");
                    }
                    else
                    {
                        if (ShowProgress) Console.WriteLine("done.");
                    }
                };
        }

        /// <summary>
        /// Creates an inference engine, optionally copying values from the default engine.
        /// </summary>
        /// <param name="copyValuesFromDefault"></param>
        internal InferenceEngine(bool copyValuesFromDefault)
        {
            CreateCompiler();
            if (copyValuesFromDefault)
            {
                lock (DefaultEngine)
                {
                    SetTo(DefaultEngine);
                }
            }
            else
            {
                Algorithm = algs[0];
            }
        }

        /// <summary>
        /// Get the abstract syntax tree for the generated code.
        /// </summary>
        /// <returns>A list of type declaration objects.</returns>
        public List<ITypeDeclaration> GetCodeToInfer(IVariable var)
        {
            if (!mb.variablesToInfer.Contains(var))
            {
                mb.Build(this, false, new IVariable[] { var });
            }
            return mb.GetGeneratedSyntax(this);
        }

        /// <summary>
        /// Compiles the last built model into a CompiledAlgorithm which implements
        /// the specified inference algorithm on the model.
        /// </summary>
        /// <returns></returns>
        private IGeneratedAlgorithm Compile()
        {
            mb.SetModelName(ModelNamespace, ModelName);
            if (ShowMsl) Console.WriteLine(mb.ModelString());
            if (ShowFactorGraph || SaveFactorGraphToFolder != null)
            {
                if (SaveFactorGraphToFolder != null && Visualizer?.GraphWriter != null)
                {
                    Directory.CreateDirectory(SaveFactorGraphToFolder);
                    Visualizer.GraphWriter.WriteGraph(mb, Path.Combine(SaveFactorGraphToFolder, ModelName));
                }
                if (ShowFactorGraph && Visualizer?.FactorGraphVisualizer != null)
                    Visualizer.FactorGraphVisualizer.VisualizeFactorGraph(mb);
            }
            Stopwatch s = null;
            if (ShowTimings)
            {
                s = new Stopwatch();
                s.Start();
            }
            IGeneratedAlgorithm compiledAlgorithm = Compiler.CompileWithoutParams(mb.modelType, null, mb.Attributes);
            if (ShowTimings)
            {
                s.Stop();
                Console.WriteLine("Compilation time was " + s.ElapsedMilliseconds + "ms.");
            }
            CompiledAlgorithmInfo info = new CompiledAlgorithmInfo(compiledAlgorithm, mb.observedVars);
            compiledAlgorithms.Push(info);
            foreach (IVariable v in mb.variablesToInfer)
            {
                compiledAlgorithmForVariable[v] = info;
            }
            SetObservedValues(info);
            return info.exec;
        }

        /// <summary>
        /// Infers the marginal distribution for the specified variable.
        /// </summary>
        /// <param name="var">The variable whose marginal is to be inferred</param>
        /// <returns>The marginal distribution (or an approximation to it)</returns>
        public object Infer(IVariable var)
        {
            IGeneratedAlgorithm ca = InferAll(false, var);
            return ca.Marginal(var.Name);
        }

        /// <summary>
        /// Performs an inference query for the specified variable, given a query type.
        /// </summary>
        /// <param name="var">The variable whose marginal is to be inferred</param>
        /// <param name="queryType">The type of query</param>
        /// <returns>The marginal distribution (or an approximation to it)</returns>
        public object Infer(IVariable var, QueryType queryType)
        {
            var ca = InferAll(false, var);
            return ca.Marginal(var.Name, queryType.Name);
        }

        /// <summary>
        /// Infers the marginal distribution for the specified variable.
        /// </summary>
        /// <typeparam name="TReturn">Desired return type which may be a distribution type or an array type if the argument is a VariableArray</typeparam>
        /// <param name="var">The variable whose marginal is to be inferred</param>
        /// <returns>The marginal distribution (or an approximation to it)</returns>
        public TReturn Infer<TReturn>(IVariable var)
        {
            IGeneratedAlgorithm ca = InferAll(false, var);
            return ca.Marginal<TReturn>(var.Name);
        }

        /// <summary>
        /// Infers the marginal distribution for the specified variable, and the specified
        /// query type
        /// </summary>
        /// <typeparam name="TReturn">Desired return type</typeparam>
        /// <param name="var">The variable whose marginal is to be inferred</param>
        /// <param name="queryType">The query type</param>
        /// <returns>The marginal distribution (or an approximation to it)</returns>
        public TReturn Infer<TReturn>(IVariable var, QueryType queryType)
        {
            // If asked for a non-default QueryType that is not an attribute of var,
            // this code will give an error message.
            // TM: This code previously recompiled the model if a new QueryType was requested.
            // This is bad because it leads to inconsistent inference results for different QueryTypes.
            // For example, if someone infers Samples and then infers Conditionals with a recompile in between.
            // Or if someone infers Marginal and then MarginalDividedByPrior with a recompile in between.
            var ca = InferAll(false, var);
            return ca.Marginal<TReturn>(var.Name, queryType.Name);
        }

        internal static T ConvertValueToDistribution<T>(object value)
        {
            if (value is T t) return t;
            Type toType = typeof(T);
            Type domainType = value.GetType();
            MethodInfo method = new Func<Converter<object, object>>(InferenceEngine.GetValueToDistributionConverter<object, object>).Method.GetGenericMethodDefinition();
            method = method.MakeGenericMethod(domainType, toType);
            Delegate converter = (Delegate)Util.Invoke(method, null);
            return (T)Util.DynamicInvoke(converter, value);
        }

        internal static TOutput[] ArrayConvertAll<TInput, TOutput>(Converter<TInput, TOutput> converter, TInput[] array)
        {
            return Array.ConvertAll(array, converter);
        }

        internal static Converter<TInput, TOutput> GetValueToDistributionConverter<TInput, TOutput>()
        {
            Exception exception = null;
            Type domainType = typeof(TInput);
            Type toType = typeof(TOutput);
            if (toType.IsArray)
            {
                Type toEltType = toType.GetElementType();
                if (domainType.IsArray)
                {
                    try
                    {
                        Type fromEltType = domainType.GetElementType();
                        // ArrayConvertAll<TInput,TOutput>(itemConverter, TInput[] array)
                        MethodInfo convertAll =
                            new Func<Converter<object, object>, object[], object[]>(InferenceEngine.ArrayConvertAll<object, object>).Method.GetGenericMethodDefinition();
                        convertAll = convertAll.MakeGenericMethod(fromEltType, toEltType);
                        MethodInfo thisMethod =
                            new Func<Converter<object, object>>(InferenceEngine.GetValueToDistributionConverter<object, object>).Method.GetGenericMethodDefinition();
                        thisMethod = thisMethod.MakeGenericMethod(fromEltType, toEltType);
                        object itemConverter = Util.Invoke(thisMethod, null);
                        return (Converter<TInput, TOutput>)Delegate.CreateDelegate(typeof(Converter<TInput, TOutput>), itemConverter, convertAll);
                    }
                    catch (Exception e)
                    {
                        // fall through to exception below
                        exception = e;
                    }
                }
                // fall through
            }
            else
            {
                Type hasPointType = typeof(HasPoint<>).MakeGenericType(domainType);
                if (hasPointType.IsAssignableFrom(toType))
                {
                    MethodInfo method =
                        (MethodInfo)
                        Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.GetBestMethod(toType, "PointMass",
                                                                BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy, null,
                                                                new Type[] { domainType }, out exception);
                    if (method != null)
                    {
                        return (Converter<TInput, TOutput>)Delegate.CreateDelegate(typeof(Converter<TInput, TOutput>), method);
                    }
                    else if (toType.IsGenericType && toType.GetGenericTypeDefinition().Equals(typeof(PointMass<>)))
                    {
                        MethodInfo method2 =
                            (MethodInfo)
                            Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.GetBestMethod(toType, "Create",
                                                                    BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy, null,
                                                                    new Type[] { domainType }, out exception);
                        return (Converter<TInput, TOutput>)Delegate.CreateDelegate(typeof(Converter<TInput, TOutput>), method2);
                    }
                    // fall through
                }
                else exception = new Exception(StringUtil.TypeToString(toType) + " does not implement " + StringUtil.TypeToString(hasPointType));
            }
            throw new ArgumentException("Cannot convert to distribution type " + StringUtil.TypeToString(toType) + " from type " + StringUtil.TypeToString(domainType) + ".",
                                        exception);
        }

        /// <summary>
        /// Attempts to convert the supplied object to the specified target type.
        /// Throws an ArgumentException if this is not possible.
        /// </summary>
        /// <remarks>
        /// Currently supports converting DistributionArray instances to .NET arrays and
        /// converting PointMass instances to distributions configured as point masses.
        /// </remarks>
        /// <typeparam name="T">The target type</typeparam>
        /// <param name="obj">The source object</param>
        /// <returns>The source object converted to type T</returns>
        internal static T ConvertDistributionToType<T>(object obj)
        {
            // Fast path if the object is already of the right type
            if (obj is T t) return t;

            // Conversion from PointMass to an instance of T set to a point mass, if T supports HasPoint.
            Type fromType = obj.GetType();
            if (fromType.IsGenericType && fromType.GetGenericTypeDefinition().Equals(typeof(PointMass<>)))
            {
                object value = fromType.GetProperty("Point").GetValue(obj, null);
                return ConvertValueToDistribution<T>(value);
            }

            Type toType = typeof(T);

            // Conversion from DistributionArray to dotNET array
            if (toType.IsArray)
            {
                try
                {
                    return Distribution.ToArray<T>(obj);
                }
                catch (Exception)
                {
                    // throw exception below instead
                }
            }
            return ConvertValueToDistribution<T>(obj);
            throw new ArgumentException("Cannot convert to type " + StringUtil.TypeToString(toType) + " from type " + StringUtil.TypeToString(fromType) + ".");
        }

        /// <summary>
        /// Computes the output message (message to the prior) for the specified variable.
        /// </summary>
        /// <typeparam name="Distribution">Desired distribution type</typeparam>
        /// <param name="var">The variable whose output message is to be inferred</param>
        /// <returns>The output message (or an approximation to it)</returns>
        public Distribution GetOutputMessage<Distribution>(IVariable var)
        {
            return Infer<Distribution>(var, QueryTypes.MarginalDividedByPrior);
        }

        private IList<IVariable> optimiseForVariables = null;

        /// <summary>
        /// The variables to optimize the engine to infer.
        /// If set to a list of variables, only the specified variables can be inferred by this engine. 
        /// If set to null, any variable can be inferred by this engine.
        /// </summary>
        /// <remarks>
        /// Setting this property to a list of variables can improve performance by removing redundent
        /// computation and storage needed to infer marginals for variables which are not on the list.</remarks>
        public IList<IVariable> OptimiseForVariables
        {
            get
            {
                if (optimiseForVariables == null) return null;
                // return a read only view of the internal list
                return new ReadOnlyCollection<IVariable>(optimiseForVariables);
            }

            set
            {
                if (value == null)
                {
                    optimiseForVariables = null;
                }
                else
                {
                    // make a copy of the passed in list
                    optimiseForVariables = new List<IVariable>(value);
                }
                InvalidateCompiledAlgorithms();
            }
        }

        protected IGeneratedAlgorithm InferAll(bool inferOnlySpecifiedVars, IVariable var)
        {
            IGeneratedAlgorithm ca = GetCompiledInferenceAlgorithm(inferOnlySpecifiedVars, var);
            Execute(ca);
            return ca;
        }

        protected IGeneratedAlgorithm InferAll(bool inferOnlySpecifiedVars, IEnumerable<IVariable> vars)
        {
            IGeneratedAlgorithm ca = GetCompiledInferenceAlgorithm(inferOnlySpecifiedVars, vars);
            Execute(ca);
            return ca;
        }

        private void SetObservedValues(CompiledAlgorithmInfo info)
        {
            foreach (Variable var in info.observedVarsInOrder)
            {
                info.exec.SetObservedValue(var.NameInGeneratedCode, ((HasObservedValue)var).ObservedValue);
            }
        }

        private void Execute(IGeneratedAlgorithm ca)
        {
            // If there is a message update listener, try to add in the engine to listen to messages.
            if (this.MessageUpdated != null)
            {
                DebuggingSupport.TryAddRemoveEventListenerDynamic(ca, OnMessageUpdated, add: true);
            }

            // Register the ProgressChanged handler only while doing inference within InferenceEngine.
            // We do not want the handler to run if the user accesses the GeneratedAlgorithms directly.
            ca.ProgressChanged += OnProgressChanged;
            try
            {
                Stopwatch s = null;
                if (ShowTimings)
                {
                    s = new Stopwatch();
                    s.Start();
                    FileStats.Clear();
                }
                if (ResetOnObservedValueChanged)
                    ca.Execute(NumberOfIterations);
                else
                    ca.Update(NumberOfIterations - ca.NumberOfIterationsDone);
                if (s != null)
                {
                    long elapsed = s.ElapsedMilliseconds;
                    Console.WriteLine("Inference time was {1}ms (max {0} iterations)",
                                      NumberOfIterations, elapsed);
                    if (FileStats.ReadCount > 0 || FileStats.WriteCount > 0)
                        Console.WriteLine("{0} file reads {1} file writes", FileStats.ReadCount, FileStats.WriteCount);
                }
            }
            finally
            {
                ca.ProgressChanged -= OnProgressChanged;
                if (this.MessageUpdated != null)
                {
                    DebuggingSupport.TryAddRemoveEventListenerDynamic(ca, OnMessageUpdated, add: false);
                }
            }
        }

        /// <summary>
        /// Returns a compiled algorithm which can later be used to infer marginal
        /// distributions for the specified variables.  This method allows more fine-grained
        /// control over the inference procedure.
        /// </summary>
        /// <remarks>This method should not be used unless fine-grained control over the
        /// inference is required.  Infer.NET will cache the last compiled algorithm
        /// and re-use it if possible.
        /// </remarks>
        /// <param name="vars">The variables whose marginals are to be computed by the returned algorithm.</param>
        /// <returns>An IGeneratedAlgorithm object</returns>
        public IGeneratedAlgorithm GetCompiledInferenceAlgorithm(params IVariable[] vars)
        {
            return GetCompiledInferenceAlgorithm(true, vars);
        }

        /// <summary>
        /// For advanced use. Returns all the model expressions that are relevant to
        /// inferring the set of variables provided.  This may be useful for constructing visualisations of the model.
        /// </summary>
        /// <remarks>
        /// The returned collection includes Variable and VariableArray objects which the engine has determined are
        /// relevant to inferring marginals over the variables provided.  This will at least include
        /// the provided variables, but may include other relevant variables as well.  It will also
        /// include MethodInvoke objects which act as priors, constraints or factors in the model.
        /// </remarks>
        /// <param name="vars">The variables to build a model for</param>
        /// <returns>A collection of model expressions</returns>
        public IReadOnlyCollection<IModelExpression> GetRelevantModelExpressions(params IVariable[] vars)
        {
            ModelBuilder mb2 = new ModelBuilder();
            mb2.Build(this, true, vars);
            return mb2.ModelExpressions;
        }

        internal IGeneratedAlgorithm GetCompiledInferenceAlgorithm(bool inferOnlySpecifiedVars, IVariable var)
        {
            // optimize the case of repeated inference on the same variable
            if (compiledAlgorithmForVariable.TryGetValue(var, out CompiledAlgorithmInfo info))
            {
                //SetObservedValues(info);
                return info.exec;
            }
            else
            {
                return BuildAndCompile(false, new IVariable[] { var });
            }
        }

        internal IGeneratedAlgorithm GetCompiledInferenceAlgorithm(bool inferOnlySpecifiedVars, IEnumerable<IVariable> vars)
        {
            // If a single compiledAlgorithm is available to infer all of the vars, then return it.
            // otherwise, build a new one.
            CompiledAlgorithmInfo info = null;
            foreach (IVariable var in vars)
            {
                CompiledAlgorithmInfo info2;
                if (!compiledAlgorithmForVariable.TryGetValue(var, out info2)) return BuildAndCompile(inferOnlySpecifiedVars, vars);
                if (info == null) info = info2;
                else if (!ReferenceEquals(info, info2)) return BuildAndCompile(inferOnlySpecifiedVars, vars);
            }
            if (info == null) throw new ArgumentException("Empty set of variables to infer");
            return info.exec;
        }

        private IGeneratedAlgorithm BuildAndCompile(bool inferOnlySpecifiedVars, IEnumerable<IVariable> vars)
        {
            if (optimiseForVariables != null)
            {
                foreach (IVariable v in vars)
                {
                    if (!optimiseForVariables.Contains(v))
                    {
                        throw new ArgumentException("Cannot call ML.Probabilistic() on variable '" + v.Name +
                                                    "' which is not in the OptimiseForVariables list. The list currently contains: " +
                                                    StringUtil.CollectionToString(OptimiseForVariables.ListSelect(x => "'" + x.Name + "'"), ",") + ".");
                    }
                }
                if (optimiseForVariables.Contains(null))
                    throw new ArgumentException("OptimiseForVariables contains a null variable");
                mb.Build(this, true, optimiseForVariables);
            }
            else
            {
                mb.Build(this, inferOnlySpecifiedVars, vars);
            }
            return Compile();
        }

        internal void OnProgressChanged(object sender, ProgressChangedEventArgs progress)
        {
            ProgressChanged?.Invoke(this, new InferenceProgressEventArgs() { Iteration = progress.Iteration, Algorithm = (IGeneratedAlgorithm)sender });
            if (!ShowProgress) return;
            int iteration = progress.Iteration + 1;
            if (iteration == 1) Console.WriteLine("Iterating: ");
            Console.Write(iteration % 10 == 0 ? "|" : ".");
            if ((iteration % 50 == 0) || (iteration == NumberOfIterations))
            {
                Console.WriteLine(" " + iteration);
            }
        }

        /// <summary>
        /// Event that is fired when the progress of inference changes, typically at the
        /// end of one iteration of the inference algorithm.
        /// </summary>
        public event InferenceProgressEventHandler ProgressChanged;

        internal void OnMessageUpdated(object sender, MessageUpdatedEventArgs messageEvent)
        {
            MessageUpdated?.Invoke(sender as IGeneratedAlgorithm, messageEvent);
        }

        /// <summary>
        /// Event that is fired when a message that has been marked with ListenToMessages has been updated.
        /// </summary>
        public event MessageUpdatedEventHandler MessageUpdated;

        /// <summary>
        /// Ensures that the last compiled algorithm will not be re-used.  This should be called
        /// whenever a change is made that requires recompiling (but not rebuilding) the model.
        /// </summary>
        internal void InvalidateCompiledAlgorithms()
        {
            compiledAlgorithmForVariable.Clear();
            compiledAlgorithms.Clear();
        }

        /// <summary>
        /// For message passing algorithms, reset all messages to their initial values.
        /// </summary>
        protected void Reset()
        {
            foreach (CompiledAlgorithmInfo info in compiledAlgorithms)
            {
                info.exec.Reset();
            }
        }

        /// <summary>
        /// If true (default), Infer resets messages to their initial values if an observed value has changed.
        /// </summary>
        public bool ResetOnObservedValueChanged { get; set; }

        internal static void InvalidateAllEngines(IModelExpression expr)
        {
            foreach (WeakReference weakRef in allEngineInstances.Keys)
            {
                if (weakRef.Target is InferenceEngine engine)
                {
                    var modelExpressions = engine.mb.ModelExpressions;
                    if (modelExpressions != null && modelExpressions.Contains(expr))
                    {
                        engine.mb.Reset(); // must rebuild the model
                        engine.InvalidateCompiledAlgorithms();
                    }
                }
                else
                {
                    // The engine has been freed, so we can remove it from the dictionary.
                    allEngineInstances.TryRemove(weakRef, out EmptyStruct value);
                }
            }
        }

        internal static void ObservedValueChanged(Variable var)
        {
            foreach (WeakReference weakRef in allEngineInstances.Keys)
            {
                if (weakRef.Target is InferenceEngine engine)
                {
                    foreach (CompiledAlgorithmInfo info in engine.compiledAlgorithms)
                    {
                        if (info.observedVars.Contains(var))
                        {
                            info.exec.SetObservedValue(var.NameInGeneratedCode, ((HasObservedValue)var).ObservedValue);
                        }
                    }
                }
                else
                {
                    // The engine has been freed, so we can remove it from the dictionary.
                    allEngineInstances.TryRemove(weakRef, out EmptyStruct value);
                }
            }
        }

        /// <summary>
        /// The model compiler that this inference engine uses.
        /// </summary>
        public ModelCompiler Compiler
        {
            get { return compiler; }
        }

        /// <summary>
        /// The default inference algorithm to use.  This can be overridden for individual
        /// variables or factors using the Algorithm attribute.
        /// </summary>
        public IAlgorithm Algorithm
        {
            get { return compiler.Algorithm; }
            set { compiler.Algorithm = value; }
        }

        private int numberOfIterations = -1;

        /// <summary>
        /// The number of iterations to use when executing the compiled inference algorithm.
        /// </summary>
        public int NumberOfIterations
        {
            get { return (numberOfIterations < 0) ? Algorithm.DefaultNumberOfIterations : numberOfIterations; }
            set { numberOfIterations = value; }
        }

        private bool showProgress = true;

        /// <summary>
        /// If true, prints progress information to the console during inference.
        /// </summary>
        public bool ShowProgress
        {
            get { return showProgress; }
            set { showProgress = value; }
        }

        private bool showTimings = false;

        /// <summary>
        /// If true, prints timing information to the console during inference. 
        /// </summary>
        public bool ShowTimings
        {
            get { return showTimings; }
            set { showTimings = value; }
        }

        private bool showMsl = false;

        /// <summary>
        /// If true, prints the model definition in Model Specification Language (MSL), prior
        /// to compiling the model. 
        /// </summary>
        public bool ShowMsl
        {
            get { return showMsl; }
            set { showMsl = value; }
        }

        /// <summary>
        /// If true, any warnings encountered during model compilation will be printed to the console. 
        /// </summary>
        public bool ShowWarnings
        {
            get { return compiler.ShowWarnings; }
            set { compiler.ShowWarnings = value; }
        }
        
        /// <summary>
        /// If true, displays the factor graph for the model, prior to compiling it.
        /// </summary>
        public bool ShowFactorGraph
        {
            get;
            set;
        }

        /// <summary>
        /// If not null, the factor graph will be saved (in DGML format) to a file in the specified folder (created if necessary) under the model name and the extension ".dgml"
        /// </summary>
        public string SaveFactorGraphToFolder
        {
            get;
            set;
        }

        /// <summary>
        /// If true, displays the schedule for the model, after the scheduler has run.
        /// </summary>
        public bool ShowSchedule
        {
            get { return Compiler.ShowSchedule; }
            set { Compiler.ShowSchedule = value; }
        }

        /// <summary>
        /// Configures this inference engine by copying the settings from the supplied inference engine.
        /// </summary>
        /// <param name="engine"></param>
        public void SetTo(InferenceEngine engine)
        {
            // note this does not copy events
            compiler.SetTo(engine.compiler);
            groups = new List<VariableGroup>();
            ModelName = engine.ModelName;
            numberOfIterations = engine.numberOfIterations;
            ShowFactorGraph = engine.ShowFactorGraph;
            SaveFactorGraphToFolder = engine.SaveFactorGraphToFolder;
            showMsl = engine.showMsl;
            showProgress = engine.showProgress;
            showTimings = engine.showTimings;
            ResetOnObservedValueChanged = engine.ResetOnObservedValueChanged;
        }
        
        /// <summary>
        /// Shows the factor manager, indicating which factors are available in Infer.NET and which
        /// are supported for each built-in inference algorithm.
        /// </summary>
        public static void ShowFactorManager(bool showMissingEvidences)
        {
            ShowFactorManager(showMissingEvidences, GetBuiltInAlgorithms());
        }

        /// <summary>
        /// Returns an array of the built-in inference algorithms.
        /// </summary>
        public static IAlgorithm[] GetBuiltInAlgorithms()
        {
            return algs;
        }

        /// <summary>
        /// Shows the factor manager, indicating which factors are available in Infer.NET and which
        /// are supported for the supplied list of inference algorithms.
        /// </summary>
        public static void ShowFactorManager(bool showMissingEvidences, params IAlgorithm[] algorithms)
        {
            if (Visualizer?.FactorManager != null)
                Visualizer.FactorManager.ShowFactorManager(showMissingEvidences, algorithms);
        }

        /// <summary>
        /// Variable groupings for the algorithm
        /// </summary>
        private List<VariableGroup> groups = new List<VariableGroup>();

        /// <summary>
        /// List of groups
        /// </summary>
        public IList<VariableGroup> Groups
        {
            get { return groups.AsReadOnly(); }
        }

        /// <summary>
        /// Add a variable group
        /// </summary>
        /// <param name="variables"></param>
        /// <returns></returns>
        public VariableGroup Group(params Variable[] variables)
        {
            VariableGroup vg = VariableGroup.FromVariables(variables);
            for (int i = 0; i < variables.Length; i++)
            {
                variables[i].AddAttribute(new Models.Attributes.GroupMember(vg, i == 0));
            }
            groups.Add(vg);
            return vg;
        }
    }
}