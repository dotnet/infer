// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Utilities;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Turns a method into a class which implements IExecutableProcess.
    /// This involves:
    ///  - moving statements into methods which represent pieces of the schedule (these are called subroutines)
    ///  - creating an Execute method which invokes all of the subroutines
    ///  - converting local variables and method parameters into fields.
    ///  - adding methods to get marginals and outputs
    /// </summary>
    internal class IterativeProcessTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "IterativeProcessTransform";
            }
        }

        internal static bool debug = true;

        // The algorithm instance 
        public IAlgorithm algorithm;
        public ModelCompiler compiler;

        /// <summary>
        /// Variables that must become fields of the generated class.
        /// </summary>
        public Set<IVariableDeclaration> persistentVars = new Set<IVariableDeclaration>();

        /// <summary>
        /// Fields that will always be re-allocated whenever they change.
        /// </summary>
        private readonly Set<string> reallocatedVariables = new Set<string>();

        private readonly Dictionary<object, IFieldDeclaration> fieldDeclarations = new Dictionary<object, IFieldDeclaration>();
        private readonly Dictionary<object, IExpression> propertyReferences = new Dictionary<object, IExpression>();
        private readonly Dictionary<IParameterDeclaration, IList<IStatement>> propertySetterStatements = new Dictionary<IParameterDeclaration, IList<IStatement>>();
        private IList<IStatement> marginalMethodStmts, marginalQueryMethodStmts;
        private IList<IStatement> marginalTMethodStmts, marginalQueryTMethodStmts;
        private IExpression marginalVariableName, marginalQueryVariableName, marginalQuery;
        private IGenericParameter marginalType, marginalQueryType;
        private IExpression setObservedVariableName, setObservedValue, getObservedVariableName;
        private IList<IStatement> setObservedValueMethodStmts, getObservedValueMethodStmts;

        /// <summary>
        /// Reference to the numberOfIterationsDone field.
        /// </summary>
        private IExpression numberOfIterationsDone;

        private IMethodDeclaration progressChangedEventMethod;
        private LoopMergingInfo loopMergingInfo;
        private IndexedProperty<NodeIndex, bool> hasLoopAncestor;
        private bool isFirstConvergenceLoop;
        private bool isTopLevel;
        private Subroutine currentSubroutine;

        public IterativeProcessTransform(ModelCompiler compiler, IAlgorithm algorithm)
        {
            this.compiler = compiler;
            this.algorithm = algorithm;
        }

        public override ITypeDeclaration ConvertType(ITypeDeclaration itd)
        {
            var td = base.ConvertType(itd);
            // sort fields for easy differencing.
            List<IFieldDeclaration> fields = new List<IFieldDeclaration>(td.Fields);
            td.Fields.Clear();
            td.Fields.AddRange(fields.OrderBy(ifd => ifd.Name));
            List<IPropertyDeclaration> properties = new List<IPropertyDeclaration>(td.Properties);
            td.Properties.Clear();
            td.Properties.AddRange(properties.OrderBy(ipd => ipd.Name));
            List<IMethodDeclaration> methods = new List<IMethodDeclaration>(td.Methods);
            td.Methods.Clear();
            td.Methods.AddRange(methods.OrderBy(imd => imd.Name));
            return td;
        }

        public override void ConvertTypeProperties(ITypeDeclaration td, ITypeDeclaration itd)
        {
            base.ConvertTypeProperties(td, itd);
            if (compiler.AddComments)
            {
                DateTime now = DateTime.Now;
                td.Documentation = "<summary>" + Environment.NewLine +
                                   "Generated algorithm for performing inference." + Environment.NewLine +
                                   "</summary>" + Environment.NewLine + "<remarks>" + Environment.NewLine +
                                   "If you wish to use this class directly, you must perform the following steps:" + Environment.NewLine +
                                   "1) Create an instance of the class." + Environment.NewLine +
                                   "2) Set the value of any externally-set fields e.g. data, priors." + Environment.NewLine +
                                   "3) Call the Execute(numberOfIterations) method." + Environment.NewLine +
                                   "4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables." + Environment.NewLine + Environment.NewLine +
                                   "Generated by " + Models.InferenceEngine.Name + " at " + now.ToString("t") + " on " + now.ToString("D") + "." + Environment.NewLine +
                                   "</remarks>";
            }

            //IConstructorDeclaration constructor = Builder.ConstructorDecl(MethodVisibility.Public, td);
            //td.Methods.Add(constructor);
        }

        protected void MakeSetObservedValueMethod(ITypeDeclaration td)
        {
            // create the SetObservedValue method
            IParameterDeclaration variableNameDecl = Builder.Param("variableName", typeof(string));
            IParameterDeclaration valueDecl = Builder.Param("value", typeof(object));
            setObservedVariableName = Builder.ParamRef(variableNameDecl);
            setObservedValue = Builder.ParamRef(valueDecl);
            IMethodDeclaration setObservedValueMethod = Builder.MethodDecl(MethodVisibility.Public, "SetObservedValue", typeof(void), td, variableNameDecl, valueDecl);
            setObservedValueMethod.Documentation = "<summary>Set the observed value of the specified variable.</summary>";
            setObservedValueMethod.Documentation += Environment.NewLine + "<param name=\"variableName\">Variable name</param>";
            setObservedValueMethod.Documentation += Environment.NewLine + "<param name=\"value\">Observed value</param>";
            td.Methods.Add(setObservedValueMethod);
            setObservedValueMethodStmts = setObservedValueMethod.Body.Statements;
        }

        protected void MakeGetObservedValueMethod(ITypeDeclaration td)
        {
            // create the GetObservedValue method
            IParameterDeclaration variableNameDecl = Builder.Param("variableName", typeof(string));
            getObservedVariableName = Builder.ParamRef(variableNameDecl);
            IMethodDeclaration getObservedValueMethod = Builder.MethodDecl(MethodVisibility.Public, "GetObservedValue", typeof(object), td, variableNameDecl);
            getObservedValueMethod.Documentation = "<summary>Get the observed value of the specified variable.</summary>";
            getObservedValueMethod.Documentation += Environment.NewLine + "<param name=\"variableName\">Variable name</param>";
            td.Methods.Add(getObservedValueMethod);
            getObservedValueMethodStmts = getObservedValueMethod.Body.Statements;
        }

        /// <summary>
        /// Methods with the OperatorMethod attribute are converted and discarded.
        /// </summary>
        /// <param name="md"></param>
        /// <param name="imd"></param>
        /// <returns></returns>
        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd))
                return imd;
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            MakeGetObservedValueMethod(td);
            MakeSetObservedValueMethod(td);
            MakeNumberOfIterationsDoneProperty(td);
            IMethodDeclaration marginalMethod = MakeMarginalMethod(td);
            td.Methods.Add(marginalMethod);
            td.Methods.Add(MakeGenericMarginalMethod(marginalMethod));
            IMethodDeclaration marginalQueryMethod = MakeMarginalQueryMethod(marginalMethod);
            td.Methods.Add(marginalQueryMethod);
            td.Methods.Add(MakeGenericMarginalQueryMethod(marginalQueryMethod));
            isTopLevel = true;
            base.DoConvertMethod(md, imd);
            return null;
        }

        /// <summary>
        /// Convert method parameter into a class field.
        /// </summary>
        /// <param name="ipd"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        protected override IParameterDeclaration ConvertMethodParameter(IParameterDeclaration ipd, int index)
        {
            if (context.FindAncestor<IAnonymousMethodExpression>() != null)
                return ipd;
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ipd);
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            IFieldDeclaration fd = Builder.FieldDecl(ipd.Name + "_field", ipd.ParameterType.DotNetType, td);
            fd.Documentation = "Field backing the " + ipd.Name + " property";
            context.OutputAttributes.Set(fd, vi);
            context.AddMember(fd);
            td.Fields.Add(fd);
            IFieldReferenceExpression fre = Builder.FieldRefExpr(fd);
            IPropertyDeclaration prop = Builder.PropDecl(ipd.Name, ipd.ParameterType.DotNetType, td, MethodVisibility.Public, MethodVisibility.Public, out IExpression value);
            prop.Documentation = "The externally-specified value of '" + ipd.Name + "'";
            ((IMethodDeclaration)prop.GetMethod).Body.Statements.Add(Builder.Return(fre));
            IList<IStatement> setStmts = ((IMethodDeclaration)prop.SetMethod).Body.Statements;


            Type type = Builder.ToType(ipd.ParameterType);
            bool ignoreEqualObservedValues = type.IsValueType
                                                 ? compiler.IgnoreEqualObservedValuesForValueTypes
                                                 : compiler.IgnoreEqualObservedValuesForReferenceTypes;
            if (ignoreEqualObservedValues)
            {
                // Check if the new observed value is equal to the old value.
                IExpression condition;
                if (type.IsValueType && !HasOperator(type, "op_Inequality"))
                {
                    MethodInfo mi = (MethodInfo)Reflection.Invoker.GetBestMethod(type, "Equals",
                        BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod,
                        type, new Type[] { type }, out Exception exception);
                    condition = Builder.NotExpr(Builder.Method(fre, mi, value));
                }
                else
                {
                    // prefer to use ValueInequality to do the comparison since the 'Equals' method would cause boxing
                    condition = Builder.BinaryExpr(fre, BinaryOperator.ValueInequality, value);
                }
                IConditionStatement cs = Builder.CondStmt(condition, Builder.BlockStmt());
                setStmts.Add(cs);
                setStmts = cs.Then.Statements;
            }
            bool checkLengths = true;
            if (vi != null && vi.sizes.Count > 0 && checkLengths)
            {
                IExpression valueIsNotNull = Builder.BinaryExpr(value, BinaryOperator.ValueInequality, Builder.LiteralExpr(null));
                int rank = vi.sizes[0].Length;
                if (rank == 1)
                {
                    // 1D array
                    IExpression expectedLength = ConvertExpression(vi.sizes[0][0]);
                    IExpression valueLength = Builder.PropRefExpr(value, type, type.IsArray ? "Length" : "Count", typeof(int));
                    IExpression lengthsDiffer = Builder.BinaryExpr(valueLength, BinaryOperator.ValueInequality, expectedLength);
                    IExpression condition = Builder.BinaryExpr(valueIsNotNull, BinaryOperator.BooleanAnd, lengthsDiffer);
                    IConditionStatement cs = Builder.CondStmt(condition, Builder.BlockStmt());
                    IStatement throwStmt = Builder.ThrowStmt(Builder.NewObject(typeof(ArgumentException), Builder.Add(
                        Builder.LiteralExpr("Provided array of length "), valueLength, Builder.LiteralExpr(" when length "), expectedLength,
                        Builder.LiteralExpr(" was expected for variable '" + ipd.Name + "'"))));
                    cs.Then.Statements.Add(throwStmt);
                    setStmts.Add(cs);
                }
                else if (rank > 1)
                {
                    // nD array
                    IExpression valueLengthString = Builder.LiteralExpr("(");
                    IExpression expectedLengthString = Builder.LiteralExpr("(");
                    IExpression anyLengthsDiffer = null;
                    for (int i = 0; i < rank; i++)
                    {
                        IExpression expectedLength = ConvertExpression(vi.sizes[0][i]);
                        IExpression valueLength = Builder.Method(value, typeof(Array).GetMethod("GetLength"), Builder.LiteralExpr(i));
                        IExpression lengthsDiffer = Builder.BinaryExpr(valueLength, BinaryOperator.ValueInequality, expectedLength);
                        if (anyLengthsDiffer == null)
                            anyLengthsDiffer = lengthsDiffer;
                        else
                            anyLengthsDiffer = Builder.BinaryExpr(anyLengthsDiffer, BinaryOperator.BooleanOr, lengthsDiffer);
                        if (i > 0)
                        {
                            valueLengthString = Builder.Add(valueLengthString, Builder.LiteralExpr(","));
                            expectedLengthString = Builder.Add(expectedLengthString, Builder.LiteralExpr(","));
                        }
                        valueLengthString = Builder.Add(valueLengthString, valueLength);
                        expectedLengthString = Builder.Add(expectedLengthString, expectedLength);
                    }
                    valueLengthString = Builder.Add(valueLengthString, Builder.LiteralExpr(")"));
                    expectedLengthString = Builder.Add(expectedLengthString, Builder.LiteralExpr(")"));
                    IExpression condition = Builder.BinaryExpr(valueIsNotNull, BinaryOperator.BooleanAnd, anyLengthsDiffer);
                    IConditionStatement cs = Builder.CondStmt(condition, Builder.BlockStmt());
                    IStatement throwStmt = Builder.ThrowStmt(Builder.NewObject(typeof(ArgumentException), Builder.Add(
                        Builder.LiteralExpr("Provided array of size "), valueLengthString, Builder.LiteralExpr(" when size "), expectedLengthString,
                        Builder.LiteralExpr(" was expected"))));
                    cs.Then.Statements.Add(throwStmt);
                    setStmts.Add(cs);
                }
            }
            setStmts.Add(Builder.AssignStmt(fre, value));
            setStmts.Add(Builder.AssignStmt(numberOfIterationsDone, Builder.LiteralExpr(0)));
            propertySetterStatements[ipd] = setStmts;

            td.Properties.Add(prop);

            // add lines to SetObservedValue()
            IExpression pre = Builder.PropRefExpr(Builder.ThisRefExpr(), prop);
            propertyReferences[ipd] = pre;
            IConditionStatement cs2 = Builder.CondStmt(Builder.BinaryExpr(setObservedVariableName, BinaryOperator.ValueEquality, Builder.LiteralExpr(ipd.Name)),
                                                       Builder.BlockStmt());
            cs2.Then.Statements.Add(Builder.AssignStmt(pre, Builder.CastExpr(setObservedValue, ipd.ParameterType)));
            cs2.Then.Statements.Add(Builder.Return());
            setObservedValueMethodStmts.Add(cs2);

            // add lines to GetObservedValue()
            IConditionStatement cs3 = Builder.CondStmt(Builder.BinaryExpr(getObservedVariableName, BinaryOperator.ValueEquality, Builder.LiteralExpr(ipd.Name)),
                                                       Builder.BlockStmt());
            cs3.Then.Statements.Add(Builder.Return(pre));
            getObservedValueMethodStmts.Add(cs3);

            return null;
        }

        public static bool HasOperator(Type type, string name)
        {
            try
            {
                Reflection.Invoker.GetBestMethod(type, name, BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy,
                                                        null, new Type[] { type, type }, out Exception exception);
                return true;
            }
            catch (MissingMethodException)
            {
                return false;
            }
        }

        protected IPropertyDeclaration MakeNumberOfIterationsDoneProperty(ITypeDeclaration td)
        {
            IFieldDeclaration fd = Builder.FieldDecl("numberOfIterationsDone", typeof(int), td);
            fd.Documentation = "Field backing the NumberOfIterationsDone property";
            context.AddMember(fd);
            td.Fields.Add(fd);
            IFieldReferenceExpression fre = Builder.FieldRefExpr(fd);
            numberOfIterationsDone = fre;
            IPropertyDeclaration prop = Builder.PropDecl("NumberOfIterationsDone", typeof(int), td, MethodVisibility.Public);
            prop.Documentation = "The number of iterations done from the initial state";
            ((IMethodDeclaration)prop.GetMethod).Body.Statements.Add(Builder.Return(fre));
            td.Properties.Add(prop);
            return prop;
        }

        protected IMethodDeclaration MakeMarginalMethod(ITypeDeclaration td)
        {
            // create the Marginal method
            IParameterDeclaration variableNameDecl = Builder.Param("variableName", typeof(string));
            marginalVariableName = Builder.ParamRef(variableNameDecl);
            IMethodDeclaration method = Builder.MethodDecl(MethodVisibility.Public, "Marginal", typeof(object), td, variableNameDecl);
            method.Documentation = "<summary>Get the marginal distribution (computed up to this point) of a variable</summary>";
            method.Documentation += Environment.NewLine + "<param name=\"variableName\">Name of the variable in the generated code</param>";
            method.Documentation += Environment.NewLine + "<returns>The marginal distribution computed up to this point</returns>";
            method.Documentation += Environment.NewLine + "<remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>";
            marginalMethodStmts = method.Body.Statements;
            return method;
        }

        protected IMethodDeclaration MakeGenericMarginalMethod(IMethodDeclaration marginalMethod)
        {
            IParameterDeclaration variableNameDecl = Builder.Param("variableName", typeof(string));
            marginalType = Builder.GenericTypeParam("T");
            IMethodDeclaration method = Builder.GenericMethodDecl(MethodVisibility.Public, "Marginal", marginalType, marginalMethod.DeclaringType,
                                                                                 new IGenericParameter[] { marginalType }, variableNameDecl);
            method.Documentation = "<summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>";
            method.Documentation += Environment.NewLine + "<typeparam name=\"T\">The distribution type.</typeparam>";
            method.Documentation += Environment.NewLine + "<param name=\"variableName\">Name of the variable in the generated code</param>";
            method.Documentation += Environment.NewLine + "<returns>The marginal distribution computed up to this point</returns>";
            method.Documentation += Environment.NewLine + "<remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>";
            marginalTMethodStmts = method.Body.Statements;

            IExpression marginalObject = Builder.Method(Builder.ThisRefExpr(), marginalMethod, Builder.ParamRef(variableNameDecl));
            IExpression convertedMarginal = Builder.StaticGenericMethod(new Func<object, PlaceHolder>(Distribution.ChangeType<PlaceHolder>), new IType[] { marginalType },
                                                                        marginalObject);
            marginalTMethodStmts.Add(Builder.Return(convertedMarginal));
            return method;
        }

        protected IMethodDeclaration MakeMarginalQueryMethod(IMethodDeclaration marginalMethod)
        {
            // create the Marginal(query) method
            IParameterDeclaration variableNameDecl = Builder.Param("variableName", typeof(string));
            IParameterDeclaration queryDecl = Builder.Param("query", typeof(string));
            marginalQuery = Builder.ParamRef(queryDecl);
            marginalQueryVariableName = Builder.ParamRef(variableNameDecl);
            IMethodDeclaration method = Builder.MethodDecl(MethodVisibility.Public, "Marginal", typeof(object), marginalMethod.DeclaringType, variableNameDecl,
                                                                        queryDecl);
            method.Documentation = "<summary>Get the query-specific marginal distribution of a variable.</summary>";
            method.Documentation += Environment.NewLine + "<param name=\"variableName\">Name of the variable in the generated code</param>";
            method.Documentation += Environment.NewLine +
                                                 "<param name=\"query\">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>";
            method.Documentation += Environment.NewLine + "<remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>";
            marginalQueryMethodStmts = method.Body.Statements;
            IConditionStatement ics = Builder.CondStmt(Builder.BinaryExpr(marginalQuery, BinaryOperator.ValueEquality, Builder.LiteralExpr(QueryTypes.Marginal.Name)),
                                                       Builder.BlockStmt());
            ics.Then.Statements.Add(Builder.Return(Builder.Method(Builder.ThisRefExpr(), marginalMethod, marginalQueryVariableName)));
            marginalQueryMethodStmts.Add(ics);
            return method;
        }

        protected IMethodDeclaration MakeGenericMarginalQueryMethod(IMethodDeclaration marginalQueryMethod)
        {
            IParameterDeclaration variableNameDecl = Builder.Param("variableName", typeof(string));
            IParameterDeclaration queryDecl = Builder.Param("query", typeof(string));
            marginalQueryType = Builder.GenericTypeParam("T");
            IMethodDeclaration method = Builder.GenericMethodDecl(MethodVisibility.Public, "Marginal", marginalQueryType, marginalQueryMethod.DeclaringType,
                                                                                      new IGenericParameter[] { marginalQueryType }, variableNameDecl, queryDecl);
            method.Documentation = "<summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>";
            method.Documentation += Environment.NewLine + "<typeparam name=\"T\">The distribution type.</typeparam>";
            method.Documentation += Environment.NewLine + "<param name=\"variableName\">Name of the variable in the generated code</param>";
            method.Documentation += Environment.NewLine +
                                                 "<param name=\"query\">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>";
            method.Documentation += Environment.NewLine + "<remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>";
            marginalQueryTMethodStmts = method.Body.Statements;
            IExpression marginalObject = Builder.Method(Builder.ThisRefExpr(), marginalQueryMethod, Builder.ParamRef(variableNameDecl), Builder.ParamRef(queryDecl));
            IExpression convertedMarginal = Builder.StaticGenericMethod(new Func<object, PlaceHolder>(Distribution.ChangeType<PlaceHolder>), new IType[] { marginalQueryType },
                                                                        marginalObject);
            marginalQueryTMethodStmts.Add(Builder.Return(convertedMarginal));
            return method;
        }

        protected IMethodDeclaration MakeExecuteMethod(IMethodDeclaration executeMethod)
        {
            IParameterDeclaration numberOfIterationsDecl = Builder.Param("numberOfIterations", typeof(int));
            IMethodDeclaration method = Builder.MethodDecl(MethodVisibility.Public, "Execute", typeof(void), executeMethod.DeclaringType, numberOfIterationsDecl);
            method.Documentation = "<summary>Update all marginals, by iterating message-passing the given number of times</summary>";
            method.Documentation += Environment.NewLine + "<param name=\"numberOfIterations\">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>";
            IList<IStatement> stmts = method.Body.Statements;
            stmts.Add(Builder.ExprStatement(Builder.Method(Builder.ThisRefExpr(), executeMethod, Builder.ParamRef(numberOfIterationsDecl), Builder.LiteralExpr(true))));
            return method;
        }

        protected IMethodDeclaration MakeUpdateMethod(IMethodDeclaration executeMethod)
        {
            IParameterDeclaration additionalIterationsDecl = Builder.Param("additionalIterations", typeof(int));
            IMethodDeclaration method = Builder.MethodDecl(MethodVisibility.Public, "Update", typeof(void), executeMethod.DeclaringType, additionalIterationsDecl);
            method.Documentation = "<summary>Update all marginals, by iterating message-passing an additional number of times</summary>";
            method.Documentation += Environment.NewLine + "<param name=\"additionalIterations\">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>";
            IList<IStatement> stmts = method.Body.Statements;
            IExpression iters = Builder.BinaryExpr(numberOfIterationsDone, BinaryOperator.Add, Builder.ParamRef(additionalIterationsDecl));
            iters = Builder.CheckedExpr(iters);
            stmts.Add(Builder.ExprStatement(Builder.Method(Builder.ThisRefExpr(), executeMethod, iters, Builder.LiteralExpr(false))));
            return method;
        }

        protected IExpression TypeName(IExpression typeObject)
        {
            return Builder.StaticMethod(new Func<Type, string>(StringUtil.TypeToString), typeObject);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            td.Partial = true;

            getObservedValueMethodStmts.Add(
                Builder.ThrowStmt(Builder.NewObject(typeof(ArgumentException),
                                                    Builder.BinaryExpr(Builder.LiteralExpr("Not an observed variable name: "), BinaryOperator.Add, getObservedVariableName))));
            setObservedValueMethodStmts.Add(
                Builder.ThrowStmt(Builder.NewObject(typeof(ArgumentException),
                                                    Builder.BinaryExpr(Builder.LiteralExpr("Not an observed variable name: "), BinaryOperator.Add, setObservedVariableName))));

            // create the Execute method
            IParameterDeclaration numberOfIterationsDecl = Builder.Param("numberOfIterations", typeof(int));
            IExpression numberOfIterations = Builder.ParamRef(numberOfIterationsDecl);
            IParameterDeclaration initialiseDecl = Builder.Param("initialise", typeof(bool));
            IExpression initialise = Builder.ParamRef(initialiseDecl);
            IMethodDeclaration executeMethod = Builder.MethodDecl(MethodVisibility.Private, "Execute", typeof(void), td, numberOfIterationsDecl, initialiseDecl);
            executeMethod.Documentation = "<summary>Update all marginals, by iterating message passing the given number of times</summary>";
            executeMethod.Documentation += Environment.NewLine + "<param name=\"numberOfIterations\">The number of times to iterate each loop</param>";
            executeMethod.Documentation += Environment.NewLine +
                                           "<param name=\"initialise\">If true, messages that initialise loops are reset when observed values change</param>";
            td.Methods.Add(executeMethod);
            td.Interfaces.Add((ITypeReference)Builder.TypeRef(typeof(IGeneratedAlgorithm)));
            td.BaseType = null;
            IList<IStatement> executeStmts = executeMethod.Body.Statements;
            IExpression changedIters = Builder.BinaryExpr(numberOfIterations, BinaryOperator.ValueInequality, numberOfIterationsDone);
            IConditionStatement numIterCondStmt = Builder.CondStmt(changedIters, Builder.BlockStmt());
            propertySetterStatements[numberOfIterationsDecl] = numIterCondStmt.Then.Statements;
            executeStmts.Add(numIterCondStmt);
            // the below code must follow the above code in the output
            IExpression lessIters = Builder.BinaryExpr(numberOfIterations, BinaryOperator.LessThan, numberOfIterationsDone);
            IConditionStatement numIterDecreasedCondStmt = Builder.CondStmt(lessIters, Builder.BlockStmt());
            numIterDecreasedCondStmt.Then.Statements.Add(Builder.AssignStmt(numberOfIterationsDone, Builder.LiteralExpr(0)));
            numIterCondStmt.Then.Statements.Add(numIterDecreasedCondStmt);
            IParameterDeclaration numberOfIterationsDecreasedDecl = Builder.Param("numberOfIterationsDecreased", typeof(int));
            propertySetterStatements[numberOfIterationsDecreasedDecl] = numIterDecreasedCondStmt.Then.Statements;

            td.Methods.Add(MakeExecuteMethod(executeMethod));
            td.Methods.Add(MakeUpdateMethod(executeMethod));

            // create a dependency graph in which while-loops are single nodes
            // the graph will only contain read-after-write and write-after-write dependencies for now
            IndexedGraph dependencyGraph = new IndexedGraph();
            List<IStatement> nodes = new List<IStatement>();
            IndexedProperty<NodeIndex, Set<IParameterDeclaration>> parameterDependencies = dependencyGraph.CreateNodeData<Set<IParameterDeclaration>>(null);
            // stores virtual dependency edges along which parameterDeps should flow
            IndexedProperty<NodeIndex, Set<NodeIndex>> containerDependencies = dependencyGraph.CreateNodeData<Set<NodeIndex>>(null);
            Dictionary<IStatement, int> indexOfNode = new Dictionary<IStatement, int>(ReferenceEqualityComparer<IStatement>.Instance);
            Set<NodeIndex> inputNodes = new Set<NodeIndex>();
            Set<NodeIndex> loopNodes = new Set<NodeIndex>();
            int whileNodeIndex = -1;
            bool hasBackEdges = false;
            void addStatementToGraph(IStatement ist)
            {
                int targetIndex;
                if (whileNodeIndex == -1)
                {
                    nodes.Add(ist);
                    targetIndex = dependencyGraph.AddNode();
                }
                else
                {
                    // do not add a new node.  all dependencies on this statement will become dependencies on the while loop.
                    targetIndex = whileNodeIndex;
                }
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                if (di == null)
                {
                    context.Error("Dependency information not found for statement: " + ist);
                    di = new DependencyInformation();
                }
                if (whileNodeIndex == -1)
                {
                    // no need to create a copy
                    parameterDependencies[targetIndex] = di.ParameterDependencies;
                }
                else
                {
                    Set<IParameterDeclaration> parameterDeps = parameterDependencies[targetIndex];
                    parameterDeps.AddRange(di.ParameterDependencies);
                }
                Set<NodeIndex> sources = new Set<NodeIndex>(); // for fast checking of duplicate sources
                sources.Add(targetIndex); // avoid cyclic dependencies
                foreach (
                    IStatement source in
                        di.GetDependenciesOfType(DependencyType.Dependency | DependencyType.Declaration | DependencyType.Overwrite))
                {
                    int sourceIndex;
                    // we assume that the statements are already ordered properly to respect dependencies.
                    // if the source is not in indexOfNode, then it must be a cyclic dependency in this while loop.
                    // therefore we can ignore it.
                    if (indexOfNode.TryGetValue(source, out sourceIndex))
                    {
                        if (!sources.Contains(sourceIndex))
                        {
                            sources.Add(sourceIndex);
                            if (!dependencyGraph.ContainsEdge(sourceIndex, targetIndex))
                                dependencyGraph.AddEdge(sourceIndex, targetIndex);
                        }
                    }
                    else
                        sourceIndex = -1;
                    if (sourceIndex == -1)
                    {
                        hasBackEdges = true;
                        // add a dependency on the initializers of source
                        Stack<IStatement> todo = new Stack<IStatement>();
                        todo.Push(source);
                        while (todo.Count > 0)
                        {
                            IStatement source2 = todo.Pop();
                            DependencyInformation di2 = context.InputAttributes.Get<DependencyInformation>(source2);
                            foreach (IStatement init in di2.Overwrites)
                            {
                                if (indexOfNode.TryGetValue(init, out int initIndex))
                                {
                                    if (!sources.Contains(initIndex))
                                    {
                                        sources.Add(initIndex);
                                        dependencyGraph.AddEdge(initIndex, targetIndex);
                                    }
                                }
                                else
                                    todo.Push(init);
                            }
                        }
                    }
                }
                // If the statement has a condition that depends on a parameter, then the statement may not execute at all if the parameters change.
                // In this case, we need to re-execute the initializer statements.
                // Therefore the Initializer statements implicitly have a dependency on those parameters.
                // For example, suppose the statement is if(a) { if(b) { msg = ...; } }
                //   then parametersUsedInCondition = (a,b) and if any of these parameters change, we need to re-execute the previous assignment to msg.
                // Additionally, if the statement is an assignment whose lhs is indexed by a parameter, then we need to re-execute the initializer statements
                // when the parameter changes.  This case is also handled by GetConditionAndTargetIndexExpressions.
                Set<IParameterDeclaration> parametersUsedInCondition =
                    new Set<IParameterDeclaration>(ReferenceEqualityComparer<IParameterDeclaration>.Instance);
                parametersUsedInCondition.AddRange(Recognizer.GetConditionAndTargetIndexExpressions(ist)
                    .SelectMany(Recognizer.GetArgumentReferenceExpressions)
                    .Select(iare => iare.Parameter.Resolve()));
                if (parametersUsedInCondition.Count > 0 || di.HasAnyDependencyOfType(DependencyType.Container))
                {
                    // the statement has conditions that depend on parameters.  add these parameters as dependencies of the Initializers,
                    // because changing the parameter may require the message to be reset to its initializer.
                    foreach (IStatement sourceStmt in di.Overwrites)
                    {
                        NodeIndex source;
                        if (indexOfNode.TryGetValue(sourceStmt, out source))
                        {
                            parameterDependencies[source] = AddParameterDependencies(parameterDependencies[source], parametersUsedInCondition);
                            Set<NodeIndex> containerDeps = containerDependencies[source];
                            if (containerDeps == null)
                            {
                                containerDeps = new Set<EdgeIndex>();
                                containerDependencies[source] = containerDeps;
                            }
                            foreach (IStatement dependencyStmt in di.ContainerDependencies)
                            {
                                if (indexOfNode.TryGetValue(dependencyStmt, out int dependency))
                                    containerDeps.Add(dependency);
                            }
                        }
                    }
                }
                // the same statement may appear multiple times.  when looking up indexOfNode, we want to use the last occurrence of the statement.
                indexOfNode[ist] = targetIndex; // must do this at the end, in case the stmt depends on a previous occurrence of itself
            }
            DeadCodeTransform.ForEachStatement(inputs, delegate (IWhileStatement iws)
            {
                nodes.Add(iws);
                whileNodeIndex = dependencyGraph.AddNode();
                indexOfNode[iws] = whileNodeIndex;
                Set<IParameterDeclaration> parameterDeps = new Set<IParameterDeclaration>(ReferenceEqualityComparer<IParameterDeclaration>.Instance);
                parameterDependencies[whileNodeIndex] = parameterDeps;
                if (IsConvergenceLoop(iws))
                {
                    loopNodes.Add(whileNodeIndex);
                    parameterDeps.Add(numberOfIterationsDecl);
                }
            }, delegate (IWhileStatement iws)
            {
                whileNodeIndex = -1;
            },
            _ => { }, _ => { },
            addStatementToGraph);
            // includeBackEdges=true seems intuitive but it causes EndCoupledChainsTest2 to fail when Optimize=false.
            // In that model there are two iteration loops with a (false) write-after-read dependency between the loop initializers.
            bool includeBackEdges = false;
            if (includeBackEdges)
            {
                // add write-after-read dependencies
                // loop statements in their original order
                foreach (NodeIndex target in dependencyGraph.Nodes)
                {
                    IStatement ist = nodes[target];
                    if (ist is IWhileStatement)
                        continue;
                    foreach(NodeIndex source in DependencyGraph2.GetPreviousReaders(context, dependencyGraph, target, nodes, indexOfNode))
                    {
                        if (source > target)
                            throw new Exception("Internal: source statement follows target");
                        if (!dependencyGraph.ContainsEdge(source, target))
                        {
                            dependencyGraph.AddEdge(source, target);
                        }
                    }
                }
            }
            dependencyGraph.NodeCountIsConstant = true;
            dependencyGraph.IsReadOnly = true;

            hasLoopAncestor = dependencyGraph.CreateNodeData<bool>();
            Dictionary<IVariableDeclaration, NodeIndex> allocationOfVariable = new Dictionary<IVariableDeclaration, NodeIndex>();
            // loop statements in their original order
            foreach (NodeIndex target in dependencyGraph.Nodes)
            {
                foreach (NodeIndex source in dependencyGraph.SourcesOf(target))
                {
                    if (loopNodes.Contains(source) || hasLoopAncestor[source])
                    {
                        hasLoopAncestor[target] = true;
                    }
                }

                if (compiler.ReturnCopies)
                {
                    IStatement ist = nodes[target];
                    if (ist is IExpressionStatement ies)
                    {
                        IExpression expr = ies.Expression;
                        if (expr is IAssignExpression iae)
                        {
                            expr = iae.Target;
                        }
                        if (expr is IVariableDeclarationExpression)
                        {
                            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
                            if (context.InputAttributes.Has<IsInferred>(ivd))
                            {
                                allocationOfVariable.Add(ivd, target);
                            }
                        }
                    }
                }
            }

            foreach (NodeIndex loopNode in loopNodes)
            {
                IWhileStatement whileLoop = (IWhileStatement)nodes[loopNode];
                var newDeps = new[] { hasLoopAncestor[loopNode] ? numberOfIterationsDecl : numberOfIterationsDecreasedDecl };
                ForEachInitializerOutsideOfLoop(whileLoop, loopNode, indexOfNode, nodes, sourceIndex =>
                {
                    // a loop initializer must run when the number of iterations decreases.
                    parameterDependencies[sourceIndex] = AddParameterDependencies(parameterDependencies[sourceIndex], newDeps);
                    if (debug)
                        AddParameterDependencyMessage(nodes[sourceIndex], $"{parameterDependencies[sourceIndex]} initializer of whileLoop node {loopNode}, hasLoopAncestor = {hasLoopAncestor[loopNode]}");
                });
            }

            // for each node of the dependency graph, determine what method parameters it depends on.
            DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<int>(dependencyGraph.SourcesOf, dependencyGraph);
            bool parameterDepsChanged = false;
            dfs.FinishNode += delegate (NodeIndex target)
            {
                // the target inherits all the dependencies of the source.
                foreach (NodeIndex source in dependencyGraph.SourcesOf(target))
                {
                    parameterDepsChanged |= InheritMembers(parameterDependencies, target, source);
                    if (debug)
                        AddParameterDependencyMessage(nodes[target], $"{parameterDependencies[target]} inherited from {nodes[source]}");
                }
                if (containerDependencies[target] != null)
                {
                    foreach (NodeIndex source in containerDependencies[target])
                    {
                        parameterDepsChanged |= InheritMembers(parameterDependencies, target, source);
                    }
                }
            };
            if (!hasBackEdges)
                dfs.SearchFrom(dependencyGraph.Nodes);
            else
            {
                do
                {
                    // propagate new parameterDependencies
                    parameterDepsChanged = false;
                    dfs.Clear();
                    dfs.SearchFrom(dependencyGraph.Nodes);
                } while (parameterDepsChanged);
            }
            // we don't want to propagate along containerDependencies any more
            containerDependencies.Clear();

            if (compiler.FreeMemory)
            {
                // In order to free memory, we will not create subroutines based on their true parameter dependencies.
                // We only want to store the loop initializers in memory, so we will create one subroutine for the loop initializers and their ancestors, 
                // and another subroutine for everything else.
                // Disconnected parts of the graph will also get their own subroutine since this doesn't increase memory consumption and can save computation.
                Set<NodeIndex> loopInitializerAncestors = new Set<NodeIndex>();
                DepthFirstSearch<NodeIndex> dfsAncestors = new DepthFirstSearch<int>(dependencyGraph.SourcesOf, dependencyGraph);
                dfsAncestors.FinishNode += delegate (NodeIndex node)
                {
                    loopInitializerAncestors.Add(node);
                };
                foreach (NodeIndex loopNode in loopNodes)
                {
                    if (hasLoopAncestor[loopNode]) continue;
                    IWhileStatement whileLoop = (IWhileStatement)nodes[loopNode];
                    ForEachInitializerOutsideOfLoop(whileLoop, loopNode, indexOfNode, nodes, sourceIndex =>
                    {
                        // all ancestors of this statement must also be marked as initializers
                        dfsAncestors.SearchFrom(sourceIndex);
                    });
                }
                // find the set of persistent variables.
                foreach (NodeIndex node in dependencyGraph.Nodes)
                {
                    IStatement ist = nodes[node];
                    ForEachDeclaration(ist, delegate (IVariableDeclaration ivd)
                    {
                        // marginals are persistent.
                        // TODO: currently declarations inside conditions must be made persistent otherwise the code won't compile due to non-local references.
                        if (context.InputAttributes.Has<IsInferred>(ivd) || (ist is IConditionStatement))
                            persistentVars.Add(ivd);
                    });
                }
                void inheritDeps(Edge<EdgeIndex> edge)
                {
                    // the target of the edge is the source node in the dependency, thus we are propagating upwards.
                    // by propagating upwards, we force earlier statements to be included in the same subroutine.
                    // we must exclude initializers since these should not be in the same subroutine 
                    // (if they were in the same subroutine then incremental inference wouldn't be possible).
                    if (!loopInitializerAncestors.Contains(edge.Target))
                    {
                        parameterDepsChanged |= InheritMembers(parameterDependencies, edge.Target, edge.Source);
                        if (debug)
                            AddParameterDependencyMessage(nodes[edge.Target], $"{parameterDependencies[edge.Target]} upward inherited from {nodes[edge.Source]}");
                    }
                }
                dfs.TreeEdge += inheritDeps;
                dfs.CrossEdge += inheritDeps;
                // because we are propagating dependencies in both directions, we need to iterate until convergence
                do
                {
                    // propagate new parameterDependencies
                    parameterDepsChanged = false;
                    dfs.Clear();
                    dfs.SearchFrom(dependencyGraph.Nodes);
                } while (parameterDepsChanged);
            }

            // loop initializer statements inherit the parameter dependencies of the loops they initialize, as initParameter dependencies.
            var initParameters = dependencyGraph.CreateNodeData<Set<IParameterDeclaration>>(null);
            foreach (NodeIndex loopNode in loopNodes)
            {
                Set<IParameterDeclaration> parameterDeps = parameterDependencies[loopNode];
                IWhileStatement whileLoop = (IWhileStatement)nodes[loopNode];
                ForEachInitializerOutsideOfLoop(whileLoop, loopNode, indexOfNode, nodes, sourceIndex =>
                {
                    Set<IParameterDeclaration> initParams = initParameters[sourceIndex];
                    if (initParams == null)
                        initParams = new Set<IParameterDeclaration>(ReferenceEqualityComparer<IParameterDeclaration>.Instance);
                    initParams.AddRange(parameterDeps);
                    if (!hasLoopAncestor[loopNode])
                    {
                        // a loop initializer should not run only because the number of iterations increases.
                        initParams.Remove(numberOfIterationsDecl);
                        if (parameterDependencies[sourceIndex].Contains(numberOfIterationsDecl))
                            Error("loop initializer cannot depend on numberOfIterations");
                    }
                    // if the node already has a dependency on a parameter, then it doesn't need an init dependency on that parameter.
                    initParams.Remove(parameterDependencies[sourceIndex]);
                    initParameters[sourceIndex] = initParams;
                });
            }
            // for each node of the dependency graph, determine what method parameters it initializes.
            DepthFirstSearch<NodeIndex> dfs3 = new DepthFirstSearch<int>(dependencyGraph.SourcesOf, dependencyGraph);
            dfs3.FinishNode += delegate (NodeIndex target)
            {
                // the target inherits all the dependencies of the source.
                foreach (NodeIndex source in dependencyGraph.SourcesOf(target))
                {
                    parameterDepsChanged |= InheritMembers(initParameters, target, source, parameterDependencies[target]);
                }
            };
            if (!hasBackEdges)
                dfs3.SearchFrom(dependencyGraph.Nodes);
            else
            {
                do
                {
                    // propagate new parameterDependencies
                    parameterDepsChanged = false;
                    dfs3.Clear();
                    dfs3.SearchFrom(dependencyGraph.Nodes);
                } while (parameterDepsChanged);
            }
            dfs3 = null; // free memory
            // if a node depends on numberOfIterations, it doesn't need to depend on numberOfIterationsDecreased
            foreach (NodeIndex node in dependencyGraph.Nodes)
            {
                Set<IParameterDeclaration> parameterDeps = parameterDependencies[node];
                // in this case, it is safe to modify the set in place.
                if (parameterDeps != null && parameterDeps.Contains(numberOfIterationsDecl))
                    parameterDeps.Remove(numberOfIterationsDecreasedDecl);
                Set<IParameterDeclaration> initDeps = initParameters[node];
                if (initDeps != null && initDeps.Contains(numberOfIterationsDecl))
                    initDeps.Remove(numberOfIterationsDecreasedDecl);
                // check if an inferred message is always re-allocated
                if (compiler.ReturnCopies)
                {
                    IStatement ist = nodes[node];
                    if (ist is IExpressionStatement ies)
                    {
                        if (ies.Expression is IMethodInvokeExpression imie)
                        {
                            if (CodeRecognizer.IsInfer(imie))
                            {
                                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(imie.Arguments[0]);
                                if (ivd != null)
                                {
                                    NodeIndex allocation = allocationOfVariable[ivd];
                                    if (parameterDeps == parameterDependencies[allocation])
                                        reallocatedVariables.Add(ivd.Name);
                                }
                            }
                        }
                    }
                }
            }

            // use the parameter dependencies to define subroutines
            List<Subroutine> subroutines = new List<Subroutine>();
            Dictionary<SubroutineDependencies, int> indexOfSubroutine = new Dictionary<SubroutineDependencies, int>();
            Set<IParameterDeclaration> emptySet = new Set<IParameterDeclaration>(ReferenceEqualityComparer<IParameterDeclaration>.Instance);
            // loop statements in their original order
            foreach (NodeIndex target in dependencyGraph.Nodes)
            {
                SubroutineDependencies targetDeps = new SubroutineDependencies(parameterDependencies[target] ?? emptySet, initParameters[target] ?? emptySet);
                if (debug)
                    context.OutputAttributes.Set(nodes[target], targetDeps);
                Subroutine sub;
                int targetSubIndex;
                if (!indexOfSubroutine.TryGetValue(targetDeps, out targetSubIndex))
                {
                    //Console.WriteLine("Creating subroutine with dependencies "+targetDeps);
                    targetSubIndex = subroutines.Count;
                    indexOfSubroutine[targetDeps] = targetSubIndex;
                    sub = new Subroutine(targetDeps, td, subroutines.Count.ToString(CultureInfo.InvariantCulture));
                    subroutines.Add(sub);
                }
                else
                    sub = subroutines[targetSubIndex];
                sub.statements.Add(target);
                IStatement ist = nodes[target];
                if (loopNodes.Contains(target))
                    sub.containsConvergenceLoop = true;
                foreach (NodeIndex source in dependencyGraph.SourcesOf(target))
                {
                    SubroutineDependencies sourceDeps = new SubroutineDependencies(parameterDependencies[source] ?? emptySet, initParameters[source] ?? emptySet);
                    int sourceSubIndex = indexOfSubroutine[sourceDeps];
                    if (sourceSubIndex != targetSubIndex)
                    {
                        Subroutine sourceSub = subroutines[sourceSubIndex];
                        sourceSub.successors.Add(sub);
                        sub.hasLoopAncestor = sub.hasLoopAncestor || sourceSub.containsConvergenceLoop || sourceSub.hasLoopAncestor;
                        if (compiler.FreeMemory)
                        {
                            // if a variable is used in a different subroutine than its declaration, then it is persistent.
                            persistentVars.AddRange(Recognizer.GetTargetVariables(nodes[source]));
                        }
                    }
                }
            }
            indexOfSubroutine = null; // free memory

            // Compute subroutine depths.  The depth is computed as the finishing time in a dfs from the root.
            int depth = subroutines.Count;
            DepthFirstSearch<Subroutine> dfs2 = new DepthFirstSearch<Subroutine>(sub => sub.successors, new NodeDataDictionary<Subroutine>());
            dfs2.FinishNode += delegate (Subroutine sub)
            {
                sub.depth = --depth;
            };
            // Even though this may search multiple times, the state is persistent so each node is finished only once.
            dfs2.SearchFrom(subroutines);
            dfs2 = null; // free memory

            // create the ProgressChanged event
            IEventDeclaration progressChangedEvent = Builder.EventDecl("ProgressChanged", (ITypeReference)Builder.TypeRef(typeof(EventHandler<ProgressChangedEventArgs>)), td);
            progressChangedEvent.Documentation =
                "Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.";
            td.Events.Add(progressChangedEvent);
            // Build a wrapper function that allows clients to fire the event
            progressChangedEventMethod = Builder.FireEventDecl(MethodVisibility.Private, "OnProgressChanged", progressChangedEvent);
            td.Methods.Add(progressChangedEventMethod);

            IMethodDeclaration resetMethod = Builder.MethodDecl(MethodVisibility.Public, "Reset", typeof(void), td);
            resetMethod.Documentation = "<summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>";
            td.Methods.Add(resetMethod);
            IList<IStatement> resetStmts = resetMethod.Body.Statements;
            resetStmts.Add(Builder.ExprStatement(Builder.Method(Builder.ThisRefExpr(), executeMethod, Builder.LiteralExpr(0))));

            // Sort subroutines by depth.  This gives a valid order of execution.
            subroutines.Sort();
            // Create a method for each subroutine and call them from the Execute method.
            for (int i = 0; i < subroutines.Count; i++)
            {
                Subroutine sub = subroutines[i];
                // create a method for the subroutine
                IMethodDeclaration method = CreateSubroutine(sub, nodes, td);
                if (loopMergingInfo != null)
                    context.OutputAttributes.Set(method, loopMergingInfo);
                // call from the Execute method
                IMethodInvokeExpression imie;
                if (sub.containsConvergenceLoop)
                {
                    if (sub.isInitialised != null)
                        imie = Builder.Method(Builder.ThisRefExpr(), method, numberOfIterations, initialise);
                    else
                        imie = Builder.Method(Builder.ThisRefExpr(), method, numberOfIterations);
                }
                else
                {
                    if (sub.isInitialised != null)
                        imie = Builder.Method(Builder.ThisRefExpr(), method, initialise);
                    else
                        imie = Builder.Method(Builder.ThisRefExpr(), method);
                }
                executeStmts.Add(Builder.ExprStatement(imie));
                // invalidate the subroutine when any source parameter changes
                foreach (IParameterDeclaration ipd in sub.dependencies.parameters)
                {
                    if (!propertySetterStatements.ContainsKey(ipd))
                    {
                        Error("Unknown parameter: " + ipd);
                        continue;
                    }
                    IList<IStatement> propertyStmts = propertySetterStatements[ipd];
                    // setting isDone=false will force the subroutine to execute on the next run
                    propertyStmts.Add(Builder.AssignStmt(sub.isDone, Builder.LiteralExpr(false)));
                }
                foreach (IParameterDeclaration ipd in sub.dependencies.initParameters)
                {
                    if (!propertySetterStatements.ContainsKey(ipd))
                    {
                        Error("Unknown parameter: " + ipd);
                        continue;
                    }
                    IList<IStatement> propertyStmts = propertySetterStatements[ipd];
                    propertyStmts.Add(Builder.AssignStmt(sub.isInitialised, Builder.LiteralExpr(false)));
                }
            }

            // must do this at the end
            executeStmts.Add(Builder.AssignStmt(numberOfIterationsDone, numberOfIterations));
            marginalMethodStmts.Add(
                Builder.ThrowStmt(Builder.NewObject(typeof(ArgumentException),
                                                    Builder.BinaryExpr(Builder.LiteralExpr("This class was not built to infer "), BinaryOperator.Add, marginalVariableName))));
            marginalQueryMethodStmts.Add(Builder.ThrowStmt(Builder.NewObject(typeof(ArgumentException),
                                                                             Builder.Add(
                                                                                 Builder.LiteralExpr("This class was not built to infer '"), marginalQueryVariableName,
                                                                                 Builder.LiteralExpr("' with query '"), marginalQuery, Builder.LiteralExpr("'")))));
        }

        protected void AddLoopInitializers(IWhileStatement whileLoop, NodeIndex loopNode, List<NodeIndex> initializers,
            Dictionary<IStatement, int> indexOfNode, List<IStatement> nodes)
        {
            InitializerSet initSet = context.InputAttributes.Get<InitializerSet>(whileLoop);
            if (initSet != null)
            {
                foreach (IStatement init in initSet.initializers)
                {
                    if (!indexOfNode.ContainsKey(init))
                    {
                        // this error is usually due to a statement being transformed after DependencyAnalysis
                        Error("Initializer not found: " + init);
                        continue;
                    }
                    NodeIndex initIndex = indexOfNode[init];
                    if (initIndex == loopNode)
                    {
                        throw new Exception();
                    }
                    else
                    {
                        // not in the loop
                        initializers.Add(initIndex);
                        IStatement node = nodes[initIndex];
                        if (node is IWhileStatement iws)
                        {
                            AddLoopInitializers(iws, initIndex, initializers, indexOfNode, nodes);
                        }
                    }
                }
            }
        }

        protected void ForEachInitializerOutsideOfLoop(IWhileStatement whileLoop, NodeIndex loopNode,
            Dictionary<IStatement, int> indexOfNode, List<IStatement> nodes, Action<NodeIndex> action)
        {
            List<NodeIndex> initializers = new List<NodeIndex>();
            AddLoopInitializers(whileLoop, loopNode, initializers, indexOfNode, nodes);
            foreach (var sourceIndex in initializers)
            {
                action(sourceIndex);
            }
        }

        protected override IStatement ConvertWhile(IWhileStatement iws)
        {
            if (context.InputAttributes.Has<HasOffsetIndices>(iws))
            {
                // this is a pseudo-cycle caused by offset indices.
                // convert the body and remove the while loop.
                // this requires LoopMerging to merge the statements.
                IBlockStatement bs = ConvertBlock(iws.Body);
                context.AddStatementsAfterCurrent(bs.Statements);
                return null;
            }
            // replace the while loop with a 'for' loop
            IVariableDeclaration iteration = null;
            // check if the body already refers to an 'iteration' variable
            foreach (var ist in iws.Body.Statements)
            {
                if (context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist))
                {
                    IConditionStatement ics = (IConditionStatement)ist;
                    iteration = Recognizer.GetVariables(ics.Condition).First();
                    break;
                }
            }
            if (iteration == null)
                iteration = Builder.VarDecl("iteration", typeof(int));
            IForStatement fs = Builder.ForStmt(iteration,
                isFirstConvergenceLoop ? numberOfIterationsDone : Builder.LiteralExpr(0),
                currentSubroutine.numberOfIterations);
            context.SetPrimaryOutput(fs);
            ConvertStatements(fs.Body.Statements, iws.Body.Statements);
            IObjectCreateExpression progressEventArgs = Builder.NewObject(typeof(ProgressChangedEventArgs), Builder.VarRefExpr(iteration));
            IStatement progressChangedStmt = Builder.ExprStatement(Builder.Method(Builder.ThisRefExpr(), progressChangedEventMethod, progressEventArgs));
            fs.Body.Statements.Add(progressChangedStmt);
            loopMergingInfo.AddNode(progressChangedStmt);
            context.InputAttributes.CopyObjectAttributesTo(iws, context.OutputAttributes, fs);
            context.OutputAttributes.Set(fs, new ConvergenceLoop());
            return fs;
        }

        private bool IsConvergenceLoop(IStatement st)
        {
            bool hasConvergenceLoop = context.InputAttributes.Has<ConvergenceLoop>(st);
            if (hasConvergenceLoop) return true;
            return (st is IWhileStatement) && !context.InputAttributes.Has<HasOffsetIndices>(st);
        }

        internal IMethodDeclaration CreateSubroutine(Subroutine sub, List<IStatement> nodes, ITypeDeclaration td)
        {
            IMethodDeclaration method;
            if (sub.containsConvergenceLoop)
            {
                IParameterDeclaration numberOfIterationsDecl = Builder.Param("numberOfIterations", typeof(int));
                method = Builder.MethodDecl(MethodVisibility.Private, sub.Name, typeof(void), td, numberOfIterationsDecl);
                sub.numberOfIterations = Builder.ParamRef(numberOfIterationsDecl);
            }
            else
            {
                method = Builder.MethodDecl(MethodVisibility.Private, sub.Name, typeof(void), td);
                sub.numberOfIterations = Builder.LiteralExpr(1);
            }
            if (sub.dependencies.Count == 0)
            {
                method.Documentation = "<summary>Computations that do not depend on observed values</summary>";
            }
            else
            {
                string parameterString = StringUtil.CollectionToString(sub.dependencies.parameters.Select(ipd => ipd.Name).OrderBy(s => s), " and ");
                if (parameterString != "")
                {
                    parameterString = "depend on the observed value of " + parameterString;
                }
                string initString = StringUtil.CollectionToString(sub.dependencies.initParameters.Select(ipd => ipd.Name).OrderBy(s => s), " and ");
                if (initString != "")
                {
                    initString = "must reset on changes to " + initString;
                    if (parameterString == "")
                        parameterString = initString;
                    else
                        parameterString += " and " + initString;
                }
                method.Documentation = "<summary>Computations that " + parameterString + "</summary>";
            }
            if (sub.containsConvergenceLoop)
                method.Documentation += Environment.NewLine + "<param name=\"numberOfIterations\">The number of times to iterate each loop</param>";
            IExpression initialise = null;
            if (sub.isInitialised != null)
            {
                IParameterDeclaration initialiseDecl = Builder.Param("initialise", typeof(bool));
                method.Parameters.Add(initialiseDecl);
                initialise = Builder.ParamRef(initialiseDecl);
                method.Documentation += Environment.NewLine + "<param name=\"initialise\">If true, reset messages that initialise loops</param>";
            }
            td.Methods.Add(method);
            IList<IStatement> methodStmts = method.Body.Statements;
            OpenOutputBlock(methodStmts);
            IExpression quitExpr = sub.isDone;
            if (sub.isInitialised != null)
            {
                // quit if no parameter has changed AND (resumeLastRun OR is initialised)
                quitExpr = Builder.BinaryExpr(quitExpr, BinaryOperator.BooleanAnd,
                                              Builder.BinaryExpr(Builder.NotExpr(initialise), BinaryOperator.BooleanOr, sub.isInitialised));
            }
            IConditionStatement cs = Builder.CondStmt(quitExpr, Builder.BlockStmt());
            cs.Then.Statements.Add(Builder.Return());
            void addStmt(IStatement ist)
            {
                methodStmts.Add(ist);
                loopMergingInfo.AddNode(ist);
            }
            addStmt(cs);
            currentSubroutine = sub;
            foreach (NodeIndex nodeIndex in sub.statements)
            {
                IStatement ist = nodes[nodeIndex];
                if (IsConvergenceLoop(ist))
                {
                    Assert.IsTrue(sub.containsConvergenceLoop);
                    isFirstConvergenceLoop = !hasLoopAncestor[nodeIndex];
                }
                bool isWhileStatement = ist is IWhileStatement;
                isTopLevel = isWhileStatement;
                IStatement st = ConvertStatement(ist);
                if (st != null)
                {
                    methodStmts.Add(st);
                    if (!isWhileStatement && loopMergingInfo != null)
                    {
                        // update loopMergingInfo with the new statement
                        // no need to remove the old statement
                        int index = loopMergingInfo.GetIndexOf(ist);
                        loopMergingInfo.AddEquivalentStatement(st, index);
                    }
                }
                FinishConvertStatement();
            }
            var trueExpr = Builder.LiteralExpr(true);
            addStmt(Builder.AssignStmt(sub.isDone, trueExpr));
            if (sub.isInitialised != null)
                addStmt(Builder.AssignStmt(sub.isInitialised, trueExpr));
            CloseOutputBlock();
            return method;
        }

        /// <summary>
        /// Apply action to every variable declaration in the statement
        /// </summary>
        /// <param name="ist"></param>
        /// <param name="action"></param>
        private void ForEachDeclaration(IStatement ist, Action<IVariableDeclaration> action)
        {
            if (ist is IExpressionStatement ies)
            {
                IExpression expr = ies.Expression;
                if (expr is IAssignExpression)
                {
                    IAssignExpression iae = (IAssignExpression)ies.Expression;
                    expr = iae.Target;
                }
                if (expr is IVariableDeclarationExpression)
                {
                    IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
                    if (ivd != null)
                        action(ivd);
                }
            }
            else if (ist is IConditionStatement ics)
            {
                foreach (IStatement st in ics.Then.Statements)
                {
                    ForEachDeclaration(st, action);
                }
            }
            else if (ist is IForStatement ifs)
            {
                foreach (IStatement st in ifs.Body.Statements)
                {
                    ForEachDeclaration(st, action);
                }
            }
        }

        public class SubroutineDependencies : ICompilerAttribute
        {
            /// <summary>
            /// Parameters whose value this subroutine depends on.
            /// </summary>
            public Set<IParameterDeclaration> parameters;

            /// <summary>
            /// Parameters that this subroutine does not depend on but provides initializers for.
            /// </summary>
            /// <remarks>
            /// The intersection of initParameters and parameters should be empty.
            /// </remarks>
            public Set<IParameterDeclaration> initParameters;

            public SubroutineDependencies(Set<IParameterDeclaration> parameterDeps, Set<IParameterDeclaration> initDeps)
            {
                this.parameters = parameterDeps;
                this.initParameters = initDeps;
            }

            public int Count
            {
                get
                {
                    return parameters.Count + initParameters.Count;
                }
            }

            public override bool Equals(object obj)
            {
                if (!(obj is SubroutineDependencies))
                    return false;
                SubroutineDependencies that = (SubroutineDependencies)obj;
                return parameters.Equals(that.parameters) && initParameters.Equals(that.initParameters);
            }

            public override int GetHashCode()
            {
                int hash = Hash.Start;
                hash = Hash.Combine(hash, parameters.GetHashCode());
                hash = Hash.Combine(hash, initParameters.GetHashCode());
                return hash;
            }

            public override string ToString()
            {
                return "SubroutineDependencies(parameterDeps={" + StringUtil.CollectionToString(parameters.Select(ipd => ipd.Name).OrderBy(s => s), ",") + "}, initDeps={" +
                       StringUtil.CollectionToString(initParameters.Select(ipd => ipd.Name).OrderBy(s => s), ",") + "})";
            }
        }

        public class Subroutine : IComparable<Subroutine>
        {
            public SubroutineDependencies dependencies;

            /// <summary>
            /// Subroutines immediately following this one.
            /// </summary>
            public Set<Subroutine> successors = new Set<Subroutine>();

            /// <summary>
            /// The nodes in this subroutine.
            /// </summary>
            public List<NodeIndex> statements = new List<NodeIndex>();

            /// <summary>
            /// The position of this subroutine in topological order.  May not be unique.
            /// </summary>
            public int depth;

            /// <summary>
            /// Name of the subroutine.
            /// </summary>
            public string Name;

            /// <summary>
            /// Reference to the isDone flag for this subroutine.
            /// </summary>
            public IExpression isDone;

            /// <summary>
            /// Reference to the isInitialised flag for this subroutine.
            /// </summary>
            public IExpression isInitialised;

            /// <summary>
            /// True if the subroutine contains a convergence loop.
            /// </summary>
            public bool containsConvergenceLoop;

            /// <summary>
            /// Variable reference to the numberOfIterations method parameter.
            /// </summary>
            public IExpression numberOfIterations;

            /// <summary>
            /// True if the subroutine is a descendant of a convergence loop subroutine.
            /// </summary>
            public bool hasLoopAncestor;

            public Subroutine(SubroutineDependencies dependencies, ITypeDeclaration td, string uniqueSuffix)
            {
                this.dependencies = dependencies;
                if (dependencies.Count == 0)
                    Name = "Constant";
                else
                {
                    if (dependencies.parameters.Count > 0)
                        Name = "Changed_" + StringUtil.CollectionToString(dependencies.parameters.Select(ipd => ipd.Name).OrderBy(s => s), "_");
                    else
                        Name = "";
                    if (dependencies.initParameters.Count > 0)
                    {
                        if (Name.Length > 0)
                            Name += "_";
                        Name += "Init_" + StringUtil.CollectionToString(dependencies.initParameters.Select(ipd => ipd.Name).OrderBy(s => s), "_");
                    }
                    // C# limits names to 512 characters
                    if (Name.Length > 100)
                        Name = Name.Substring(0, 100) + uniqueSuffix;
                }
                IFieldDeclaration fd = Builder.FieldDecl(Name + "_isDone", typeof(bool), td);
                fd.Visibility = FieldVisibility.Public;
                fd.Documentation = "True if " + Name + " has executed. Set this to false to force re-execution of " + Name;
                td.Fields.Add(fd);
                isDone = Builder.FieldRefExpr(fd);
                if (dependencies.initParameters.Count > 0)
                {
                    IFieldDeclaration fd2 = Builder.FieldDecl(Name + "_isInitialised", typeof(bool), td);
                    fd2.Visibility = FieldVisibility.Public;
                    fd2.Documentation = "True if " + Name + " has performed initialisation. Set this to false to force re-execution of " + Name;
                    td.Fields.Add(fd2);
                    isInitialised = Builder.FieldRefExpr(fd2);
                }
            }

            public override string ToString()
            {
                return Name;
            }

            public int CompareTo(Subroutine other)
            {
                return depth.CompareTo(other.depth);
            }
        }

        internal class ParameterDependencyMessages : ICompilerAttribute
        {
            public readonly List<string> Messages = new List<string>();

            public void Add(string message)
            {
                Messages.Add(message);
            }

            public override string ToString()
            {
                return StringUtil.EnumerableToString(Messages, Environment.NewLine);
            }
        }

        private void AddParameterDependencyMessage(IStatement ist, string message)
        {
            ParameterDependencyMessages attr = context.InputAttributes.GetOrCreate(ist, () => new ParameterDependencyMessages());
            attr.Add(message);
        }

        Set<T> AddParameterDependencies<T>(Set<T> set, IEnumerable<T> itemsToAdd)
            where T : class
        {
            var result = Set<T>.FromEnumerable(ReferenceEqualityComparer<T>.Instance, set);
            result.AddRange(itemsToAdd);
            return result;
        }

        internal bool InheritMembers<T>(IndexedProperty<NodeIndex, Set<T>> setOf, NodeIndex target, NodeIndex source)
        {
            return InheritMembers(setOf, target, source, null);
        }

        /// <summary>
        /// Add all items in setOf[source] to setOf[target].  Returns true if any new items were added.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="setOf"></param>
        /// <param name="target"></param>
        /// <param name="source"></param>
        /// <param name="excludeFromTarget">Items that should not be added to setOf[target]</param>
        /// <returns></returns>
        internal bool InheritMembers<T>(IndexedProperty<NodeIndex, Set<T>> setOf, NodeIndex target, NodeIndex source, Set<T> excludeFromTarget)
        {
            Set<T> targetDeps = setOf[target];
            Set<T> sourceDeps = setOf[source];
            if (sourceDeps == null || object.ReferenceEquals(sourceDeps, targetDeps) || (targetDeps != null && targetDeps.ContainsAll(sourceDeps)))
                return false;
            else
            {
                Set<T> reducedSourceDeps;
                if (excludeFromTarget == null || !excludeFromTarget.ContainsAny(sourceDeps))
                {
                    reducedSourceDeps = sourceDeps;
                    // we know from the check above that targetDeps does not contain all sourceDeps
                    if (targetDeps != null && !reducedSourceDeps.ContainsAll(targetDeps))
                    {
                        // neither set is contained within the other.  must create a new set.
                        // do not mutate the existing set, since it may be shared between many nodes.
                        reducedSourceDeps += targetDeps;  // not AddRange
                    }
                }
                else
                {
                    reducedSourceDeps = sourceDeps - excludeFromTarget;
                    if (targetDeps != null)
                    {
                        if (targetDeps.ContainsAll(reducedSourceDeps))
                            return false;
                        reducedSourceDeps.AddRange(targetDeps);
                    }
                }
                setOf[target] = reducedSourceDeps;
                return true;
            }
        }

        protected override void ConvertStatements(IList<IStatement> outputs, IEnumerable<IStatement> inputs)
        {
            OpenOutputBlock(outputs);
            foreach (IStatement ist in inputs)
            {
                bool wasTopLevel = isTopLevel;
                bool isConvergenceLoop = ist is IWhileStatement;
                bool isFirstIterPost = context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist);
                if (!(isConvergenceLoop || isFirstIterPost))
                    isTopLevel = false;
                IStatement st = ConvertStatement(ist);
                isTopLevel = wasTopLevel;
                if (st != null)
                {
                    outputs.Add(st);
                    if (isTopLevel && !isConvergenceLoop && loopMergingInfo != null)
                    {
                        // update loopMergingInfo with the new statement
                        // no need to remove the old statement
                        int index = loopMergingInfo.GetIndexOf(ist);
                        loopMergingInfo.AddEquivalentStatement(st, index);
                    }
                }
                FinishConvertStatement();
            }
            CloseOutputBlock();
        }

        protected override IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            IStatement st = base.ConvertExpressionStatement(ies);
            if (st != null)
            {
                if (compiler.AddComments)
                {
                    DescriptionAttribute da = context.InputAttributes.Get<DescriptionAttribute>(ies);
                    if (da == null)
                        da = context.InputAttributes.Get<DescriptionAttribute>(ies.Expression);
                    if (da != null)
                        context.AddStatementBeforeCurrent(Builder.CommentStmt(da.Description));
                }
                //IExpressionStatement es = (IExpressionStatement)st;
                //CheckForMultiplyAll(es); // for AccumulationTransform
            }
            return st;
        }

        private void CheckForMultiplyAll(IExpressionStatement ies)
        {
            if (!context.InputAttributes.Has<MultiplyAllCompilerAttribute>(ies))
                return;
            if (context.OutputAttributes.Has<AccumulationInfo>(ies))
                return;
            //context.OutputAttributes.Remove<Factors.MultiplyAllAttribute>(ies);
            if (!(ies.Expression is IAssignExpression))
                return;
            IAssignExpression iae = (IAssignExpression)ies.Expression;
            IExpression target = iae.Target;
            if (!(iae.Expression is IMethodInvokeExpression))
                return;
            IMethodInvokeExpression imie = (IMethodInvokeExpression)iae.Expression;
            AccumulationInfo ai = new AccumulationInfo(target);
            ai.containers = new Containers(context);
            ai.accumulateMethod = (new Func<PlaceHolder, PlaceHolder, PlaceHolder>(ArrayHelper.SetToProductWith<PlaceHolder>));
            //.Method.GetGenericMethodDefinition().MakeGenericMethod(ai.type);
            ai.initializer = Builder.AssignStmt(target, Builder.StaticGenericMethod(
                new Func<PlaceHolder, PlaceHolder>(ArrayHelper.SetToUniform<PlaceHolder>),
                new Type[] { ai.type },
                target));
            foreach (IExpression arg in imie.Arguments)
            {
                if (arg.Equals(ai.accumulator))
                    continue;
                object ivd = Recognizer.GetVariableDeclaration(arg);
                if (ivd == null)
                    ivd = Recognizer.GetFieldReference(arg);
                if (ivd != null)
                {
                    List<AccumulationInfo> ais = context.OutputAttributes.GetAll<AccumulationInfo>(ivd);
                    foreach (AccumulationInfo ai2 in ais)
                    {
                        // for some reason, == doesn't work properly here
                        // TODO: == doesn't work properly between ThisReferenceExpressions - why?
                        if (ai2.accumulator.Equals(ai.accumulator))
                        {
                            // use the existing AccumulationInfo
                            context.OutputAttributes.Add(ies, ai2);
                            return;
                        }
                    }
                    context.OutputAttributes.Add(ivd, ai);
                }
                else
                {
                    if (!arg.GetExpressionType().Equals(ai.type))
                        throw new NotImplementedException("Accumulator type is " + ai.type + " but constant expression has type " + arg.GetExpressionType());
                    ai.statements.Add(ai.GetAccumulateStatement(context, ai.accumulator, arg));
                }
            }
            context.OutputAttributes.Set(ies, ai);
        }

        protected override IExpression ConvertArgumentRef(IArgumentReferenceExpression iare)
        {
            IParameterDeclaration ipd = iare.Parameter.Resolve();
            if (fieldDeclarations.TryGetValue(ipd, out IFieldDeclaration ifd))
                return Builder.FieldRefExpr(ifd);
            else if (propertyReferences.TryGetValue(ipd, out IExpression expr))
                return expr;
            else
                return iare;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            IVariableDeclaration ivd = ivre.Variable.Resolve();
            if (fieldDeclarations.TryGetValue(ivd, out IFieldDeclaration ifd))
                return Builder.FieldRefExpr(ifd);
            // If no field declaration is associated with the variable, leave it as a local variable
            // e.g. for index variables in loops.
            return ivre;
        }

        private void AddUse(IExpression messageExpr)
        {
            // increment usage counts
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(messageExpr);
            if (ivd != null)
            {
                MessageArrayInformation mai = context.InputAttributes.Get<MessageArrayInformation>(ivd);
                if (mai != null)
                    mai.useCount++;
            }
        }


        private void ProcessQualityBand(IExpression ie)
        {
            QualityBandCompilerAttribute qual = context.InputAttributes.Get<QualityBandCompilerAttribute>(ie);
            // Catch any quality attributes not handled by the transforms:
            if (qual == null)
            {
                if (ie is IVariableDeclarationExpression ivde)
                {
                    Type ty = ivde.GetExpressionType();
                    if (Distribution.HasDistributionType(ty))
                    {
                        // (1) If a distribution type, then get the quality whether unknown or known
                        qual = new QualityBandCompilerAttribute(Distribution.GetQualityBand(ty));
                    }
                    else
                    {
                        // (2) If a non-distribution type, only handle if there is a non-unknown quality
                        QualityBand qb = Quality.GetQualityBand(ty);
                        if (qb != QualityBand.Unknown)
                            qual = new QualityBandCompilerAttribute(qb);
                    }
                }
                else if (ie is IMethodInvokeExpression imie) 
                {
                    // (3) If a method reference, only handle if there is a non-unknown quality
                    IMethodReference imr = imie.Method.Method;
                    QualityBand qb = Quality.GetQualityBand(imr.MethodInfo);
                    if (qb != QualityBand.Unknown)
                        qual = new QualityBandCompilerAttribute(qb);
                }
            }

            if (qual != null)
            {
                if (qual.QualityBand < compiler.RequiredQuality)
                    Error(
                        String.Format("{0} has quality band {1} which is less than the required quality band ({2})",
                                      ie, qual.QualityBand, compiler.RequiredQuality));
                else if (qual.QualityBand < compiler.RecommendedQuality)
                    Warning(
                        String.Format("{0} has quality band {1} which is less than the recommended quality band ({2})",
                                      ie, qual.QualityBand, compiler.RecommendedQuality));
            }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            ProcessQualityBand(imie);
            foreach (var ivd in Recognizer.GetVariables(imie))
            {
                MessageArrayInformation mai = context.InputAttributes.Get<MessageArrayInformation>(ivd);
                if (mai != null)
                    mai.useCount++;
            }
            if (CodeRecognizer.IsInfer(imie))
                return ConvertInfer(imie);
            return base.ConvertMethodInvoke(imie);
        }

        private IExpression ConvertInfer(IMethodInvokeExpression imie)
        {
            string varName = (string)((ILiteralExpression)imie.Arguments[1]).Value;
            ExpressionEvaluator eval = new ExpressionEvaluator();
            QueryType query = (imie.Arguments.Count < 3) ? null : (QueryType)eval.Evaluate(imie.Arguments[2]);

            IExpression targetExpr = ConvertExpression(imie.Arguments[0]);
            if (targetExpr == null)
            {
                Error("Unhandled output expression: " + imie.Arguments[0]);
                return imie;
            }

            if (query == null)
            {
                CreateOutputMethod("Marginal", targetExpr);
            }
            else
            {
                string path = algorithm.GetQueryTypeBinding(query);
                if (path != "")
                {
                    CreateOutputMethodWithPath(varName, query.Name, path, targetExpr, query == QueryTypes.Marginal);
                }
                else
                    CreateOutputMethod(query.Name, targetExpr);
            }
            return null;

            void CreateOutputMethod(string suffix, IExpression expr)
            {
                bool isMarginal = (suffix != QueryTypes.MarginalDividedByPrior.Name);
                CreateOutputMethodWithPath(varName, suffix, null, expr, isMarginal);
            }
        }

        /// <summary>
        /// Returns an expression of type DistributionArray from an indexed expression whose target has type Distribution[]
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        private IExpression GetDistributionArrayWrapper(IExpression expr)
        {
            if (!(expr is IArrayIndexerExpression))
                return expr;
            IArrayIndexerExpression iaie = (IArrayIndexerExpression)expr;
            Type type = expr.GetExpressionType();
            int rank = iaie.Indices.Count;
            bool isDistribution = Distribution.IsDistributionType(type);
            if (isDistribution)
                type = Distribution.MakeDistributionArrayType(type, rank);
            else
                type = Util.MakeArrayType(type, rank);
            while (iaie.Target is IArrayIndexerExpression target)
            {
                iaie = target;
                rank = iaie.Indices.Count;
                if (isDistribution)
                    type = Distribution.MakeDistributionArrayType(type, rank);
                else
                    type = Util.MakeArrayType(type, rank);
            }
            // iaie.Target now has all indices stripped, so that it refers to the outermost array
            return GetDistributionArrayConversion(iaie.Target, type);
        }

        private IExpression GetDistributionArrayConversion(IExpression expr, Type newType)
        {
            Type type = expr.GetExpressionType();
            if (!type.IsArray)
                return expr;
            if (type == newType)
                return expr;
            Type elementType = type.GetElementType();
            if (!elementType.IsArray)
            {
                if (newType.IsArray)
                    return expr;
                else
                    return Builder.NewObject(newType, expr);
            }
            Type newElementType = Util.GetElementType(newType);
            Type[] typeArgs = new Type[] { elementType, newElementType };
            IAnonymousMethodExpression iame = Builder.AnonMethodExpr(typeof(Converter<,>).MakeGenericType(typeArgs));
            string name = CodeBuilder.MakeValid(StringUtil.TypeToString(elementType) + "item");
            IParameterDeclaration param = Builder.Param(name, elementType);
            iame.Parameters.Add(param);
            iame.Body = Builder.BlockStmt();
            iame.Body.Statements.Add(Builder.Return(GetDistributionArrayConversion(Builder.ParamRef(param), newElementType)));
            return Builder.NewObject(newType,
                                     Builder.StaticGenericMethod(
                                         new Func<PlaceHolder[], Converter<PlaceHolder, PlaceHolder>, PlaceHolder[]>(Array.ConvertAll<PlaceHolder, PlaceHolder>)
                                         , typeArgs, expr, iame)
                );
        }

        protected void AddMarginalMethod(string varName, IMethodDeclaration md)
        {
            IExpression variableName = marginalVariableName;
            IExpression condition = Builder.BinaryExpr(variableName, BinaryOperator.ValueEquality, Builder.LiteralExpr(varName));
            IConditionStatement cs = Builder.CondStmt(condition, Builder.BlockStmt());
            cs.Then.Statements.Add(Builder.Return(Builder.Method(Builder.ThisRefExpr(), md)));
            marginalMethodStmts.Add(cs);
        }

        protected void AddMarginalQueryMethod(string varName, string query, IMethodDeclaration md)
        {
            IExpression variableName = marginalQueryVariableName;
            IExpression equalsName = Builder.BinaryExpr(variableName, BinaryOperator.ValueEquality, Builder.LiteralExpr(varName));
            IExpression equalsQuery = Builder.BinaryExpr(marginalQuery, BinaryOperator.ValueEquality, Builder.LiteralExpr(query));
            IExpression condition = Builder.BinaryExpr(equalsName, BinaryOperator.BooleanAnd, equalsQuery);
            IConditionStatement cs = Builder.CondStmt(condition, Builder.BlockStmt());
            cs.Then.Statements.Add(Builder.Return(Builder.Method(Builder.ThisRefExpr(), md)));
            marginalQueryMethodStmts.Add(cs);
        }

        /// <summary>
        /// Create output method
        /// </summary>
        /// <param name="varName"></param>
        /// <param name="query"></param>
        /// <param name="expr"></param>
        /// <param name="path"></param>
        /// <param name="isMarginal"></param>
        /// <remarks>Marginal is returned as an array over distributions. Output is returned
        /// as a distribution array</remarks>
        protected void CreateOutputMethodWithPath(string varName, string query, string path, IExpression expr, bool isMarginal)
        {
            bool hasPath = !(path == null || path == "");

            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();

            Type type = expr.GetExpressionType();
            if (hasPath)
            {
                PropertyInfo pinfo = type.GetProperty(path);
                if (pinfo == null)
                {
                    Error("Cannot find property " + path);
                    return;
                }
                Type propertyType = pinfo.PropertyType;
                expr = Builder.PropRefExpr(expr, type, path, propertyType);
                type = propertyType;
            }

            // Create marginal method and add return statement.
            string name = GetUniqueMethodName(td, CodeBuilder.Capitalise(varName + query));
            IMethodDeclaration md = Builder.MethodDecl(MethodVisibility.Public, name, type, td);
            string queryDoc = query.ToString();
            if (queryDoc == QueryTypes.Marginal.Name) queryDoc = "marginal distribution";
            else if(queryDoc == QueryTypes.MarginalDividedByPrior.Name) queryDoc = "output message (the posterior divided by the prior)";
            md.Documentation = "<summary>" + Environment.NewLine +
                               "Returns the " + queryDoc + " for '" + varName + "' given by the current state of the" + Environment.NewLine +
                               "message passing algorithm." + Environment.NewLine +
                               "</summary>" + Environment.NewLine +
                               "<returns>The " + queryDoc + "</returns>";
            IBlockStatement body = md.Body;

            if (compiler.ReturnCopies && !type.IsValueType && !reallocatedVariables.Contains(Recognizer.GetFieldReference(expr).Name))
            {
                if (type.IsGenericType && type.GetGenericTypeDefinition().Equals(typeof(IList<>)))
                {
                    Type elementType = type.GetGenericArguments()[0];
                    Type listType = typeof(List<>).MakeGenericType(elementType);
                    expr = Builder.NewObject(listType, expr);
                }
                else
                {
                    expr = Builder.StaticGenericMethod(
                        new Func<PlaceHolder, PlaceHolder>(ArrayHelper.MakeCopy),
                        new Type[] { type },
                        expr);
                }
            }
            body.Statements.Add(Builder.Return(expr));
            context.AddMember(md);
            td.Methods.Add(md);

            if (isMarginal)
                AddMarginalMethod(varName, md); // must have path == ""
            else
                AddMarginalQueryMethod(varName, query, md);
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            bool isLoopVar = (Recognizer.GetAncestorIndexOfLoopBeingInitialized(context) != -1);
            if (isLoopVar)
                return base.ConvertVariableDeclExpr(ivde);
            bool isLhs = (context.FindAncestorIndex<IExpressionStatement>() == context.Depth - 2);
            ProcessQualityBand(ivde);
            IVariableDeclaration ivd = ivde.Variable;
            DescriptionAttribute da = context.GetAttribute<DescriptionAttribute>(ivd);
            if (compiler.FreeMemory && !persistentVars.Contains(ivd))
            {
                return ivde;
            }
            IFieldDeclaration fd = AddFieldDeclaration(ivd.Name, ivd.VariableType);
            if (da != null)
            {
                fd.Documentation = da.Description; //"The messages for '" + ivd.Name + "'";
            }
            fieldDeclarations[ivd] = fd;
            context.InputAttributes.CopyObjectAttributesTo(ivd, context.OutputAttributes, fd);
            if (isLhs)
                return null; // remove if only a declaration statement
            else
                return Builder.FieldRefExpr(fd);
        }

        protected IFieldDeclaration AddFieldDeclaration(string name, IType type)
        {
            name = GetUniqueFieldName(name);
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            IFieldDeclaration fd = Builder.FieldDecl(name, type, td);
            fd.Visibility = FieldVisibility.Public;
            context.AddMember(fd);
            td.Fields.Add(fd);
            return fd;
        }

        protected string GetUniqueFieldName(string basename)
        {
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            //            if (makeFirstLower) basename = Char.ToLower(basename[0]) + basename.Substring(1);
            int k = basename.IndexOf('`');
            if (k != -1)
                basename = basename.Substring(0, k);
            basename = CodeBuilder.MakeValid(basename);
            string s = basename;
            int i = 1;
            while (HasField(td, s))
            {
                s = basename + "_" + i;
                i++;
            }
            return s;
        }

        protected bool HasField(ITypeDeclaration td, string name)
        {
            foreach (IFieldDeclaration fd in td.Fields)
                if (fd.Name.Equals(name))
                    return true;
            return false;
        }

        protected string GetUniqueMethodName(ITypeDeclaration td, string basename)
        {
            int k = basename.IndexOf('`');
            if (k != -1)
                basename = basename.Substring(0, k);
            basename = CodeBuilder.MakeValid(basename);
            string s = basename;
            int i = 1;
            while (HasMethod(td, s))
            {
                s = basename + "_" + i;
                i++;
            }
            return s;
        }

        private bool HasMethod(ITypeDeclaration td, string name)
        {
            foreach (IMethodDeclaration imd in td.Methods)
            {
                if (imd.Name == name)
                    return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Let's you control how the model compiler detects
    /// whether an observed value has changed.
    /// </summary>
    public enum ObservedValueChangedBehaviour
    {
        WhenSet,

        /// Assumes observed value has changed whenever the ObservedValue property is set.
        WhenSetAndNotEqual, // Assumes observed value has changed when new ObservedValue != old ObservedValue,
        WhenSetAndRefTypeOrNotEqual,
        // Assumes observed value has changed whenever the ObservedValue property is set if it is a reference type or otherwise only if new ObservedValue != old ObservedValue,
    }

    /// <summary>
    /// Attached to a 'for' statement to indicate that it is an iteration loop.
    /// </summary>
    internal class ConvergenceLoop : ICompilerAttribute
    {
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}