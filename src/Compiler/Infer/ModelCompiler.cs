// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Runtime.Serialization;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using System.IO;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// A model compiler takes a model specified in MSL and compiles it into the code
    /// required to perform inference in that model, as specified by the inference engine settings
    /// and model attributes.
    /// </summary>
    public class ModelCompiler
    {
        internal static bool UseTracingTransform = false;
        private readonly FactorManager factorManager = new FactorManager();
        private IAlgorithm algorithm;

        /// <summary>
        /// Assembly resolve event handler. Ref: James Margetson's Excel Add-in bug.
        /// AddIn cannot automatically resolve generated assembly.
        /// </summary>
        internal static Assembly ResolveEventHandler(object sender, ResolveEventArgs args)
        {
            var asms = AppDomain.CurrentDomain.GetAssemblies();
            foreach (Assembly asm in asms)
            {
                if (asm.FullName == args.Name)
                    return asm;
            }
            return null;
        }

        static ModelCompiler()
        {
            AppDomain.CurrentDomain.AssemblyResolve += ResolveEventHandler;
        }

        /// <summary>
        /// Creates a model compiler
        /// </summary>
        internal ModelCompiler()
        {
        }

        /// <summary>
        /// If true, all variables will implicitly have a TraceMessages attribute.
        /// </summary>
        public bool TraceAllMessages { get; set; }

        /// <summary>
        /// If true, a variable named "loggingAction" will be used to log method entry/exit.
        /// </summary>
        public bool Logging { get; set; }

        /// <summary>
        /// List of extra transforms to run before IterativeProcessTransform
        /// </summary>
        public IList<Func<ICodeTransform>> ExtraTransforms { get; set; }

        /// <summary>
        /// Declaration provider for model defined in MSL
        /// </summary>
        public IDeclarationProvider DeclarationProvider { get; set; }

        /// <summary>
        /// Controls if source code files are written to disk.  If true, source code files will be 
        /// written out to the GeneratedSourceFolder.
        /// </summary>
        public bool WriteSourceFiles { get; set; } = true;

        /// <summary>
        /// If true and WriteSourceFiles=true, existing source code files will be used instead of re-generated.  
        /// </summary>
        public bool UseExistingSourceFiles { get; set; }

        /// <summary>
        /// Controls if debug information is included in generated DLLs.  If true, debug information will 
        /// be included which allows stepping through the generated code in a debugger.
        /// </summary>
        public bool IncludeDebugInformation { get; set; } = false;

        /// <summary>
        /// If true, prints compilation progress information to the console during model compilation.
        /// </summary>
        public bool ShowProgress { get; set; }

        /// <summary>
        /// If true, compiler warnings are printed to the console.
        /// </summary>
        public bool ShowWarnings { get; set; } = true;

        /// <summary>
        /// If true, compiler exceptions are caught and displayed in the transform browser, rather than propagating to the caller.
        /// </summary>
        public bool CatchExceptions { get; set; }

        /// <summary>
        /// If true, compiler warnings are treated as errors.
        /// </summary>
        public bool TreatWarningsAsErrors { get; set; }

        /// <summary>
        /// If true, allow factor arguments marked 'stochastic' to be derived.  This can cause inference to diverge on some models.
        /// </summary>
        public bool AllowDerivedParents { get; set; }

        /// <summary>
        /// The inference algorithm to use.
        /// </summary>
        public IAlgorithm Algorithm
        {
            get { return algorithm; }
            set
            {
                algorithm = value;
                OnParametersChanged();
            }
        }

        /// <summary>
        /// Controls if inference assembly is generated in memory or on disk.
        /// </summary>
        /// <remarks>This is set to true by default. Set to false if you want to debug into the generated code</remarks>
        public bool GenerateInMemory { get; set; } = true;

        /// <summary>
        /// The path (absolute or relative) where source code files will be generated.
        /// </summary>
        public string GeneratedSourceFolder { get; set; } = "GeneratedSource";

        /// <summary>
        /// Selects the used compiler: Roslyn or CodeDom.Compiler. 
        /// </summary>
        /// <remarks>
        /// This is set to Auto by default, which means CodeDom.Compiler on .NET full / Windows, and Roslyn on .NET Core and Mono.
        /// When overriding, consider the following:
        /// <list type="bullet">
        /// <item><description>CodeDom.Compiler throws PlatformNotSupportedException on .NET Core.</description></item>
        /// <item><description>Roslyn takes a few seconds to initialize, which is not the case for CodeDom.Compiler.</description></item>
        /// <item><description>CodeDom.Compiler works very slowly on Mono.</description></item>
        /// </list>
        /// </remarks>
        public CompilerChoice CompilerChoice { get; set; } = CompilerChoice.Auto;

        private bool useParallelForLoops = false;

        /// <summary>
        /// If true, use Parallel.For() instead of top-level for loops in the generated code.  
        /// Requires the Microsoft Parallel Extensions to be installed.
        /// </summary>
        public bool UseParallelForLoops
        {
            get
            {
                return useParallelForLoops;
            }
            set
            {
                if (useParallelForLoops != value)
                {
                    useParallelForLoops = value;
                    OnParametersChanged();
                }
            }
        }

        private bool useSerialSchedules = true;

        /// <summary>
        /// Find serial schedules for graphs with offset indexing.
        /// </summary>
        public bool UseSerialSchedules
        {
            get { return useSerialSchedules; }
            set
            {
                if (useSerialSchedules != value)
                {
                    useSerialSchedules = value;
                    OnParametersChanged();
                }
            }
        }

        private bool useExperimentalSerialSchedules;

        public bool UseExperimentalSerialSchedules
        {
            get { return useExperimentalSerialSchedules; }
            set
            {
                if (useExperimentalSerialSchedules != value)
                {
                    useExperimentalSerialSchedules = value;
                    OnParametersChanged();
                }
            }
        }

        private bool allowSerialInitialisers = true;

        /// <summary>
        /// Experimental feature: Allow Sequential loops in the initialisation schedule.  Set this to false to work around scheduling problems.
        /// </summary>
        public bool AllowSerialInitialisers
        {
            get { return allowSerialInitialisers; }
            set
            {
                if (allowSerialInitialisers != value)
                {
                    allowSerialInitialisers = value;
                    OnParametersChanged();
                }
            }
        }

        private bool useSpecialFirstIteration = false;

        /// <summary>
        /// Experimental feature: generate a schedule using InitializeTo for the first iteration and another schedule ignoring InitializeTo for the remaining iterations.
        /// </summary>
        public bool UseSpecialFirstIteration
        {
            get { return useSpecialFirstIteration; }
            set
            {
                if (useSpecialFirstIteration != value)
                {
                    useSpecialFirstIteration = value;
                    OnParametersChanged();
                }
            }
        }

        private bool useLocals = false;

        /// <summary>
        /// If true and OptimiseInferenceCode=true, then the generated code is optimized by converting array elements to loop locals where possible.
        /// </summary>
        public bool UseLocals
        {
            get { return useLocals; }
            set
            {
                if (useLocals != value)
                {
                    useLocals = value;
                    OnParametersChanged();
                }
            }
        }

        private bool ignoreEqualObservedValuesForValueTypes = true;

        /// <summary>
        /// Let's you control what happens when you set an observed value which is equal
        /// to the old observed value (for value types). If this property is true, setting an equal
        /// value will be ignored.  If false, setting an equal value may cause inference to be re-run.
        /// </summary>
        public bool IgnoreEqualObservedValuesForValueTypes
        {
            get { return ignoreEqualObservedValuesForValueTypes; }
            set
            {
                if (ignoreEqualObservedValuesForValueTypes == value) return;
                ignoreEqualObservedValuesForValueTypes = value;
                OnParametersChanged();
            }
        }

        private bool ignoreEqualObservedValuesForReferenceTypes = false;

        /// <summary>
        /// Let's you control what happens when you set an observed value which is equal
        /// to the old observed value (for reference types). If this property is true, setting an equal
        /// value will be ignored.  If false, setting an equal value may cause inference to be re-run.
        /// </summary>
        /// <remarks>
        /// The default value of 'false' is safe, but may be inefficient if you often set observed values 
        /// to be the same value.
        /// </remarks>
        public bool IgnoreEqualObservedValuesForReferenceTypes
        {
            get { return ignoreEqualObservedValuesForReferenceTypes; }
            set
            {
                if (ignoreEqualObservedValuesForReferenceTypes == value) return;
                ignoreEqualObservedValuesForReferenceTypes = value;
                OnParametersChanged();
            }
        }

        private bool addComments = true;

        /// <summary>
        /// If true, comments will be added to the generated code.
        /// </summary>
        public bool AddComments
        {
            get { return addComments; }
            set
            {
                if (addComments != value)
                {
                    addComments = value;
                    OnParametersChanged();
                }
            }
        }

        private bool enforceTriggers = true;

        /// <summary>
        /// If false, trigger and fresh annotations will not be enforced during scheduling.  Useful for debugging infinite RepairSchedule loops.
        /// </summary>
        public bool EnforceTriggers
        {
            get { return enforceTriggers; }
            set
            {
                if (enforceTriggers != value)
                {
                    enforceTriggers = value;
                    OnParametersChanged();
                }
            }
        }

        private bool initialisationAffectsSchedule = false;

        /// <summary>
        /// If true, user-provided initializations will affect the iteration schedule.  This can sometimes improve the convergence rate.
        /// </summary>
        public bool InitialisationAffectsSchedule
        {
            get { return initialisationAffectsSchedule; }
            set
            {
                if (initialisationAffectsSchedule != value)
                {
                    initialisationAffectsSchedule = value;
                    OnParametersChanged();
                }
            }
        }

        private bool optimiseInferenceCode = true;

        /// <summary>
        /// Optimises generated code by removing redundant messages or operations.
        /// </summary>
        public bool OptimiseInferenceCode
        {
            get { return optimiseInferenceCode; }
            set
            {
                if (optimiseInferenceCode != value)
                {
                    optimiseInferenceCode = value;
                    OnParametersChanged();
                }
            }
        }

        private bool returnCopies = true;

        /// <summary>
        /// If true, code will be generated to return copies of the internal marginal distributions.  
        /// If this is not done, the returned marginals are volatile and may be modified in place when inference runs again.  
        /// Set to false to save memory/time.
        /// </summary>
        public bool ReturnCopies
        {
            get { return returnCopies; }
            set
            {
                if (returnCopies != value)
                {
                    returnCopies = value;
                    OnParametersChanged();
                }
            }
        }

        private bool unrollLoops = false;

        /// <summary>
        /// If true, all loops with constant bounds will be unrolled.
        /// </summary>
        public bool UnrollLoops
        {
            get { return unrollLoops; }
            set
            {
                if (unrollLoops != value)
                {
                    unrollLoops = value;
                    OnParametersChanged();
                }
            }
        }

        private bool freeMemory = true;

        /// <summary>
        /// Trade memory for time.
        /// If true, memory usage is reduced for increase in time.  Temporary storage will be freed when inference completes, requiring re-allocation every time inference is run.
        /// If false, memory usage is increased for reduction in time.  Temporary storage will be kept and re-used for later inference runs.
        /// </summary>
        public bool FreeMemory
        {
            get { return freeMemory; }
            set
            {
                if (freeMemory != value)
                {
                    freeMemory = value;
                    OnParametersChanged();
                }
            }
        }

        private QualityBand requiredQuality = QualityBand.Experimental;

        /// <summary>
        /// Sets the component quality band which is required for running inference. By default
        /// this is QualityBand.Experimental.
        /// </summary>
        /// <remarks>
        /// If the quality of any component is below this quality band, then an
        /// Infer.NET model compiler error is generated. This can be switched off
        /// by setting it to QualityBand.Unknown
        /// </remarks>
        public QualityBand RequiredQuality
        {
            get { return requiredQuality; }
            set
            {
                requiredQuality = value;
                if (recommendedQuality < requiredQuality) recommendedQuality = requiredQuality;
            }
        }

        private QualityBand recommendedQuality = QualityBand.Preview;

        /// <summary>
        /// Sets the quality band at which is recommended for running inference. By default
        /// this is QualityBand.Preview.
        /// </summary>
        /// <remarks>
        /// If the quality of any component is below this quality band, then an
        /// Infer.NET model compiler warning is generated. This can be switched off
        /// by setting it to QualityBand.Unknown
        /// </remarks>
        public QualityBand RecommendedQuality
        {
            get { return recommendedQuality; }
            set
            {
                recommendedQuality = value;
                if (requiredQuality > recommendedQuality) requiredQuality = recommendedQuality;
            }
        }

        /// <summary>
        /// Resolve ambiguous matches for message operators in favor of the given container.  Accumulates with all previous calls.
        /// </summary>
        /// <param name="container">A Type, namespace string, Module, or Assembly.</param>
        public void GivePriorityTo(object container)
        {
            factorManager.GivePriorityTo(container);
            OnParametersChanged();
        }

        /// <summary>
        /// Remove any priority of container given by previous calls to GivePriorityTo.
        /// </summary>
        /// <param name="container"></param>
        public void RemovePriority(object container)
        {
            factorManager.PriorityList.Remove(container);
            OnParametersChanged();
        }

        /// <summary>
        /// A list of message operator containers, highest priority first.
        /// </summary>
        public IList<object> PriorityList
        {
            get { return factorManager.PriorityList.AsReadOnly(); }
        }

        /// <summary>
        /// Controls when the model compiler browser is shown
        /// </summary>
        public BrowserMode BrowserMode { get; set; }
        
        /// <summary>
        /// If true, displays the schedule for the model, after the scheduler has run.
        /// </summary>
        public bool ShowSchedule { get; set; }

        /// <summary>
        /// Data passed to event handlers for the Compiling and Compiled events.
        /// </summary>
        public class CompileEventArgs : EventArgs
        {
            /// <summary>
            /// After compilation, the set of warnings.  Otherwise null.
            /// </summary>
            public ICollection<TransformError> Warnings;
            /// <summary>
            /// If compilation failed, the exception that was thrown.  Otherwise null.
            /// </summary>
            public Exception Exception;
        }

        public delegate void CompileEventHandler(ModelCompiler sender, CompileEventArgs e);

        /// <summary>
        /// Event raised before a model is compiled.
        /// </summary>
        public event CompileEventHandler Compiling;

        /// <summary>
        /// Event raised after a model is compiled or fails to compile.
        /// </summary>
        public event CompileEventHandler Compiled;

        /// <summary>
        /// Event raised when a compilation parameter is changed.
        /// </summary>
        public event EventHandler ParametersChanged;

        protected void OnParametersChanged()
        {
            ParametersChanged?.Invoke(this, new EventArgs());
        }

        // Model methods with up to ten parameters are supported directly.
        /// <exclude/>
        public delegate void ModelDefinitionMethod();

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T>(T a);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2>(T1 a, T2 b);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3>(T1 a, T2 b, T3 c);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4>(T1 a, T2 b, T3 c, T4 d);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4, T5>(T1 a, T2 b, T3 c, T4 d, T5 e);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4, T5, T6>(T1 a, T2 b, T3 c, T4 d, T5 e, T6 f);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7>(T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8>(T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8, T9>(T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i);

        /// <exclude/>
        public delegate void ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j);

        /// <exclude/>
        public IGeneratedAlgorithm Compile(ModelDefinitionMethod method)
        {
            return CompileWithParamArray(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1>(ModelDefinitionMethod<T1> method, T1 arg1)
        {
            return CompileWithParamArray(method.Method, arg1);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1>(ModelDefinitionMethod<T1> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2>(ModelDefinitionMethod<T1, T2> method, T1 arg1, T2 arg2)
        {
            return CompileWithParamArray(method.Method, arg1, arg2);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2>(ModelDefinitionMethod<T1, T2> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3>(ModelDefinitionMethod<T1, T2, T3> method, T1 arg1, T2 arg2, T3 arg3)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3>(ModelDefinitionMethod<T1, T2, T3> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4>(ModelDefinitionMethod<T1, T2, T3, T4> method, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4>(ModelDefinitionMethod<T1, T2, T3, T4> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5>(ModelDefinitionMethod<T1, T2, T3, T4, T5> method, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4, arg5);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5>(ModelDefinitionMethod<T1, T2, T3, T4, T5> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6> method, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4, arg5, arg6);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7> method, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5,
                                                                       T6 arg6, T7 arg7)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7, T8>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8> method, T1 arg1, T2 arg2, T3 arg3, T4 arg4,
                                                                           T5 arg5, T6 arg6, T7 arg7, T8 arg8)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7, T8>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7, T8, T9>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8, T9> method, T1 arg1, T2 arg2, T3 arg3,
                                                                               T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7, T8, T9>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8, T9> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> method, T1 arg1, T2 arg2,
                                                                                    T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
        {
            return CompileWithParamArray(method.Method, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
        }

        /// <exclude/>
        public IGeneratedAlgorithm Compile<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(ModelDefinitionMethod<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> method)
        {
            return CompileWithoutParams(method.Method);
        }

        /// <summary>
        /// Compiles the model defined in MSL by the specified method.  The model parameters are set
        /// to the specified values.  This method should only be used when the method has more than 10 parameters,
        /// otherwise one of the strongly typed <code>Compile</code> methods should be used.
        /// </summary>
        /// <param name="method"></param>
        /// <param name="varValues"></param>
        /// <returns>An instance of the compiled model, with all parameters set.</returns>
        public IGeneratedAlgorithm CompileWithParamArray(MethodBase method, params object[] varValues)
        {
            var ca = CompileWithoutParams(method);
            ParameterInfo[] pis = method.GetParameters();
            if (pis.Length != varValues.Length) throw new ArgumentException("Wrong number of variables passed, " + varValues.Length + "!=" + pis.Length);
            for (int i = 0; i < pis.Length; i++) ca.SetObservedValue(pis[i].Name, varValues[i]);
            return ca;
        }

        /// <summary>
        /// Compiles the model defined in MSL by the specified method.  The model parameters are not
        /// set - they must be set before the model is executed.
        /// </summary>
        /// <param name="method"></param>
        /// <returns>An instance of the compiled model, without any parameters set.</returns>
        public IGeneratedAlgorithm CompileWithoutParams(MethodBase method)
        {
            if (null == DeclarationProvider)
                throw new InferCompilerException("A type declaration provider has not been set for this instance of the model compiler");
            return CompileWithoutParams(DeclarationProvider.GetTypeDeclaration(method?.DeclaringType, true),
                                        method, new AttributeRegistry<object, ICompilerAttribute>(true));
        }

        /// <summary>
        /// Get the abstract syntax tree for the generated code.
        /// </summary>
        /// <param name="itd"></param>
        /// <param name="method"></param>
        /// <param name="inputAttributes"></param>
        /// <returns></returns>
        internal List<ITypeDeclaration> GetTransformedDeclaration(ITypeDeclaration itd, MethodBase method, AttributeRegistry<object, ICompilerAttribute> inputAttributes)
        {
            TransformerChain tc = ConstructTransformChain(method);
            try
            {
                Compiling?.Invoke(this, new CompileEventArgs());
                bool trackTransform = (BrowserMode != BrowserMode.Never);
                List<ITypeDeclaration> output = tc.TransformToDeclaration(itd, inputAttributes, trackTransform, ShowProgress, out List<TransformError> warnings, CatchExceptions, TreatWarningsAsErrors);
                OnCompiled(new CompileEventArgs()
                    {
                        Warnings = warnings
                    });
                if (BrowserMode == BrowserMode.Always)
                    ShowBrowser(tc, GeneratedSourceFolder, output[0].Name);
                else if(BrowserMode == BrowserMode.WriteFiles)
                    tc.WriteAllOutputs(Path.Combine(GeneratedSourceFolder, output[0].Name + "_Transforms"));
                if (ShowSchedule && InferenceEngine.Visualizer?.TaskGraphVisualizer != null)
                {
                    foreach (CodeTransformer ct in tc.transformers)
                    {
                        if (ct.Transform is DeadCodeTransform bst)
                        {
                            foreach (ITypeDeclaration itd2 in ct.transformMap.Values)
                            {
                                InferenceEngine.Visualizer.TaskGraphVisualizer.VisualizeTaskGraph(itd2, (BasicTransformContext)bst.Context);
                            }
                        }
                    }
                }
                return output;
            }
            catch (TransformFailedException ex)
            {
                OnCompiled(new CompileEventArgs() {Exception = ex});
                if (BrowserMode != BrowserMode.Never) ShowBrowser(tc, GeneratedSourceFolder, itd.Name);
                throw new CompilationFailedException(ex.Results, ex.Message);
            }
        }

        private void OnCompiled(CompileEventArgs e)
        {
            Compiled?.Invoke(this, e);
        }

        internal void ShowBrowser(TransformerChain tc, string folder, string name)
        {
            if (InferenceEngine.Visualizer?.TransformerChainVisualizer != null)
                InferenceEngine.Visualizer.TransformerChainVisualizer.VisualizeTransformerChain(tc, folder, name);
        }

        internal IGeneratedAlgorithm CompileWithoutParams(ITypeDeclaration itd, MethodBase method, AttributeRegistry<object, ICompilerAttribute> inputAttributes)
        {
            AttributeRegistry<object, ICompilerAttribute> outputAttributes = (AttributeRegistry<object, ICompilerAttribute>)inputAttributes.Clone();
            List<ITypeDeclaration> output = GetTransformedDeclaration(itd, method, outputAttributes);
            return CompileWithoutParams<IGeneratedAlgorithm>(output);
        }

        internal T CompileWithoutParams<T>(List<ITypeDeclaration> itds)
        {
            CodeCompiler cc = new CodeCompiler
            {
                GeneratedSourceFolder = GeneratedSourceFolder,
                writeSourceFiles = WriteSourceFiles,
                useExistingFiles = UseExistingSourceFiles,
                generateInMemory = GenerateInMemory,
                includeDebugInformation = IncludeDebugInformation,
                optimizeCode = !IncludeDebugInformation, // tie these together for now
                showProgress = ShowProgress,
                compilerChoice = CompilerChoice
            };
            CompilerResults cr = cc.WriteAndCompile(itds);
            // Examine the compilation results and stop if errors
            if (cr.Errors.Count > 0)
            {
                Console.WriteLine("Compilation failed with " + cr.Errors.Count + " error(s)");
                foreach (string err in cr.Errors) Console.WriteLine(err);
                throw new CompilationFailedException("Errors found when compiling generated code for: " + itds[0].Name);
            }
            Stopwatch watch = null;
            if (ShowProgress)
            {
                Console.Write("Creating compiled object ");
                watch = new Stopwatch();
                watch.Start();
            }
            try
            {
                Type inferenceType = CodeCompiler.GetCompiledType(cr, itds[0]);
                var iip = (T)Activator.CreateInstance(inferenceType);
                if (ShowProgress)
                {
                    watch.Stop();
                    Console.WriteLine("({0}ms)", watch.ElapsedMilliseconds);
                }
                if (ShowProgress) Console.WriteLine("Done compiling");
                return iip;
            }
            catch (System.Security.SecurityException sec_ex)
            {
                throw new System.Security.SecurityException(
                    sec_ex.Source + " has thrown a Security Exception." +
                    "This may be because you are running your application from a network drive." +
                    "Either copy your application or project to a local drive, " +
                    "or set the model compiler GenerateInMemory property to true.");
            }
        }


        /// <summary>
        /// Construct the transform chain for the given method
        /// </summary>
        /// <param name="method">The method</param>
        /// <returns></returns>
        private TransformerChain ConstructTransformChain(MethodBase method)
        {
            if (algorithm == null) throw new InferCompilerException("No algorithm was specified, please specify one before compiling.");
            TransformerChain tc = new TransformerChain();
            //if (args != null) tc.AddTransform(new ParameterInsertionTransform(method,args));
            if (UnrollLoops) tc.AddTransform(new LoopUnrollingTransform(method));
            bool useVariableTransform = !(algorithm is GibbsSampling);
            bool useDepthCloning = useVariableTransform;

            tc.AddTransform(new IsolateModelTransform(method));
            tc.AddTransform(new ExternalVariablesTransform());
            tc.AddTransform(new IntermediateVariableTransform());
            tc.AddTransform(new ModelAnalysisTransform());
            tc.AddTransform(new ConstantFoldingTransform());
            tc.AddTransform(new ArrayAnalysisTransform());
            tc.AddTransform(new EqualityPropagationTransform());
            tc.AddTransform(new StocAnalysisTransform(true));
            tc.AddTransform(new MarginalAnalysisTransform());

            tc.AddTransform(new GateTransform(algorithm));
            tc.AddTransform(new IndexingTransform());
            if (useDepthCloning)
            {
                // DepthCloningTransform needs two passes since the first pass may create new code that needs to be transformed.
                // See SwitchDeepArrayCopyTest.
                //tc.AddTransform(new DepthCloningTransform(false));
                // Unfortunately this breaks ArrayUsedAtManyDepths2.
                tc.AddTransform(new DepthCloningTransform(true));
                tc.AddTransform(new ReplicationTransform());
                // IfCutting must be between Depth and Replication because loops are added
                tc.AddTransform(new IfCuttingTransform());
            }
            tc.AddTransform(new DerivedVariableTransform());
            tc.AddTransform(new PowerTransform());
            tc.AddTransform(new ReplicationTransform());
            if (useVariableTransform)
            {
                tc.AddTransform(new VariableTransform(algorithm));
                if (useDepthCloning)
                    tc.AddTransform(new Channel2Transform(useDepthCloning, false));
            }
            // IfCutting must be after Variable because Variable factors send evidence
            tc.AddTransform(new IfCuttingTransform());
            if (useVariableTransform)
            {
                // must do Replication here since Channel2 could have created new loops (see ArrayUsedAtManyDepths3)
                if (useDepthCloning)
                    tc.AddTransform(new ReplicationTransform());
                tc.AddTransform(new Channel2Transform(useDepthCloning, true));
                tc.AddTransform(new PointMassAnalysisTransform());
            }
            else
                tc.AddTransform(new ChannelTransform(algorithm));
            if (algorithm is GibbsSampling)
                tc.AddTransform(new GroupTransform(algorithm));
            //   tc.AddTransform(new HybridAlgorithmTransform(engine.Algorithm));
            tc.AddTransform(new MessageTransform(this, algorithm, factorManager, this.AllowDerivedParents));
            tc.AddTransform(new IncrementTransform(this));
            if (OptimiseInferenceCode)
            {
                // LoopCutting must precede CopyPropagation because you can have situations such as:
                // for(N) {
                //   bool local = ...;
                //   array[N] = Copy(local);
                // }
                // SomeFactor(array);
                var lct = new LoopCuttingTransform(false);
                tc.AddTransform(lct);
                tc.AddTransform(lct); // run again to catch uses before declaration
                // TODO: fix CopyPropagation so it only needs to run once.
                tc.AddTransform(new CopyPropagationTransform());
                tc.AddTransform(new CopyPropagationTransform());
                tc.AddTransform(new CopyPropagationTransform());
                tc.AddTransform(new HoistingTransform(this));
                // LoopCutting must follow Hoisting since new hoist variables can be loop locals
            }
            var lct2 = new LoopCuttingTransform(true);
            tc.AddTransform(lct2);
            tc.AddTransform(lct2); // run again to catch uses before declaration
            if (OptimiseInferenceCode)
            {
                // must run after HoistingTransform
                tc.AddTransform(new LoopRemovalTransform());
            }
            tc.AddTransform(new DependencyAnalysisTransform());
            tc.AddTransform(new PruningTransform());
            tc.AddTransform(new IterationTransform(this));
            tc.AddTransform(new IncrementPruningTransform());
            tc.AddTransform(new InitializerTransform(this));
            if (UseSerialSchedules && !UseExperimentalSerialSchedules)
                tc.AddTransform(new ForwardBackwardTransform(this));
            tc.AddTransform(new SchedulingTransform(this));
            tc.AddTransform(new UniquenessTransform());
            tc.AddTransform(new DependencyPruningTransform());
            if (OptimiseInferenceCode)
                tc.AddTransform(new LoopReversalTransform());
            tc.AddTransform(new LocalAllocationTransform(this));
            tc.AddTransform(new DeadCodeTransform(this, true));
            // add any extra transforms provided by the user
            if (this.ExtraTransforms != null)
            {
                foreach (var transform in this.ExtraTransforms)
                    tc.AddTransform(transform());
            }
            tc.AddTransform(new IterativeProcessTransform(this, algorithm));
            // LoopMerging is required to support offset indexing (see GateModelTests.CaseLoopIndexTest2)
            tc.AddTransform(new LoopMergingTransform());
            tc.AddTransform(new IsIncreasingTransform());
            // Local is required for DistributedTests
            tc.AddTransform(new LocalTransform(this));
            if (OptimiseInferenceCode)
                tc.AddTransform(new DeadCode2Transform(this));
            tc.AddTransform(new ParallelScheduleTransform());
            // All messages after each iteration will be logged to csv files in a folder named with the model name.
            // Use MatlabWriter.WriteFromCsvFolder to convert these to a mat file.
            if (TraceAllMessages && UseTracingTransform)
                tc.AddTransform(new TracingTransform());
            bool useArraySizeTracing = false;
            if (useArraySizeTracing)
            {
                // This helps isolate memory performance issues.
                tc.AddTransform(new ArraySizeTracingTransform());
            }
            if (UseParallelForLoops)
                tc.AddTransform(new ParallelForTransform());
            tc.AddTransform(new LoggingTransform(this));
            return tc;
        }

        /// <summary>
        /// Configures this model compiler by copying settings from the supplied model compiler.
        /// </summary>
        /// <param name="compiler">The compiler to copy settings from</param>
        public void SetTo(ModelCompiler compiler)
        {
            CompilerChoice = compiler.CompilerChoice;
            ShowSchedule = compiler.ShowSchedule;
            BrowserMode = compiler.BrowserMode;
            WriteSourceFiles = compiler.WriteSourceFiles;
            GenerateInMemory = compiler.GenerateInMemory;
            GeneratedSourceFolder = compiler.GeneratedSourceFolder;
            UseParallelForLoops = compiler.UseParallelForLoops;
            UseSerialSchedules = compiler.UseSerialSchedules;
            UseExperimentalSerialSchedules = compiler.UseExperimentalSerialSchedules;
            UnrollLoops = compiler.UnrollLoops;
            OptimiseInferenceCode = compiler.OptimiseInferenceCode;
            IncludeDebugInformation = compiler.IncludeDebugInformation;
            returnCopies = compiler.returnCopies;
            FreeMemory = compiler.FreeMemory;
            RequiredQuality = compiler.RequiredQuality;
            RecommendedQuality = compiler.RecommendedQuality;
            ShowProgress = compiler.ShowProgress;
            ShowWarnings = compiler.ShowWarnings;
            CatchExceptions = compiler.CatchExceptions;
            Algorithm = compiler.Algorithm;
            IgnoreEqualObservedValuesForValueTypes = compiler.IgnoreEqualObservedValuesForValueTypes;
            IgnoreEqualObservedValuesForReferenceTypes = compiler.IgnoreEqualObservedValuesForReferenceTypes;
            AddComments = compiler.AddComments;
            AllowDerivedParents = compiler.AllowDerivedParents;
            AllowSerialInitialisers = compiler.AllowSerialInitialisers;
            UseExistingSourceFiles = compiler.UseExistingSourceFiles;
            ExtraTransforms = compiler.ExtraTransforms;
            TraceAllMessages = compiler.TraceAllMessages;
            EnforceTriggers = compiler.EnforceTriggers;
            UseLocals = compiler.UseLocals;
            factorManager.SetTo(compiler.factorManager);
        }
    }

    /// <summary>
    /// Exception thrown when Infer.NET model compilation encounters errors.
    /// </summary>
    [Serializable]
    public class CompilationFailedException : Exception
    {
        /// <summary>
        /// The errors and warnings thrown by the failed stage of compilation.
        /// </summary>
        public TransformResults Results;

        public CompilationFailedException(TransformResults tr, string msg)
            : base(msg)
        {
            Results = tr;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CompilationFailedException"/> class.
        /// </summary>
        public CompilationFailedException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CompilationFailedException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public CompilationFailedException(string message)
            : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CompilationFailedException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public CompilationFailedException(string message, Exception inner)
            : base(message, inner)
        {
        }

        // This constructor is needed for serialization.
        protected CompilationFailedException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }

    /// <summary>
    /// Controls when the model compiler browser is shown.
    /// </summary>
    public enum BrowserMode
    {
        /// <summary>
        /// Never show the browser
        /// </summary>
        Never,

        /// <summary>
        /// Show the browser only if an error occurs in compiling the model
        /// </summary>
        OnError,

        /// <summary>
        /// Always show the browser
        /// </summary>
        Always,

        /// <summary>
        /// Like OnError but also write transform outputs to files
        /// </summary>
        WriteFiles
    };

    public enum CompilerChoice
    {
        Auto,
        CodeDom,
        Roslyn
    }
}