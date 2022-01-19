// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using System.Reflection;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Builds an MSL model from an in-memory graph of model expression objects.
    /// </summary>
    internal class ModelBuilder
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        public ITypeDeclaration modelType;
        public AttributeRegistry<object, ICompilerAttribute> Attributes;
        private readonly Stack<IModelExpression> toSearch = new Stack<IModelExpression>();
        private readonly Set<IModelExpression> searched = new Set<IModelExpression>(ReferenceEqualityComparer<IModelExpression>.Instance);

        /// <summary>
        /// The set of condition variables used in 'IfNot' blocks.  Filled in during search.
        /// </summary>
        private readonly Set<Variable> negatedConditionVariables = new Set<Variable>(ReferenceEqualityComparer<Variable>.Instance);

        // optimizes ModelBuilder.Compile
        internal readonly List<Variable> observedVars = new List<Variable>();
        internal readonly Set<IVariable> variablesToInfer = new Set<IVariable>(ReferenceEqualityComparer<IVariable>.Instance);

        private IMethodDeclaration modelMethod;
        private Stack<IList<IStatement>> blockStack;
        private bool inferOnlySpecifiedVars = false;

        /// <summary>
        /// Mapping from constant values to their declarations in the generated code.
        /// </summary>
        private readonly Dictionary<object, IVariableDeclaration> constants = new Dictionary<object, IVariableDeclaration>();

        internal IReadOnlyCollection<IModelExpression> ModelExpressions { get; private set; }

        /// <summary>
        /// The default namespace for generated code.
        /// This should not be a child of any existing namespace, such as Microsoft.ML.Probabilistic, because of possible name conflicts.
        /// </summary>
        public const string ModelNamespace = "Models";

        public void Reset()
        {
            modelType = Builder.TypeDecl();
            modelType.Namespace = ModelNamespace;
            modelType.Owner = null;
            modelType.BaseType = null;
            modelType.Visibility = TypeVisibility.Public;
            modelMethod = Builder.MethodDecl(MethodVisibility.Public, "Model", typeof(void), modelType);
            IBlockStatement body = Builder.BlockStmt();
            modelMethod.Body = body;
            blockStack = new Stack<IList<IStatement>>();
            blockStack.Push(body.Statements);
            modelType.Methods.Add(modelMethod);
            Attributes = new AttributeRegistry<object, ICompilerAttribute>(true);
            searched.Clear();
            observedVars.Clear();
            variablesToInfer.Clear();
            constants.Clear();
            ModelExpressions = null;
        }

        /// <summary>
        /// Builds the model necessary to infer marginals for the supplied variables and algorithm.
        /// </summary>
        /// <param name="engine">The inference algorithm being used</param>
        /// <param name="inferOnlySpecifiedVars">If true, inference will be restricted to only the variables given.</param>
        /// <param name="vars">Variables to infer.</param>
        /// <returns></returns>
        /// <remarks>
        /// Algorithm: starting from the variables to infer, we search through the graph to build up a "searched set".
        /// Each Variable and MethodInvoke in this set has an associated timestamp.
        /// We sort by timestamp, and then generate code.
        /// </remarks>
        public ITypeDeclaration Build(InferenceEngine engine, bool inferOnlySpecifiedVars, IEnumerable<IVariable> vars)
        {
            List<IStatementBlock> openBlocks = StatementBlock.GetOpenBlocks();
            if (openBlocks.Count > 0)
            {
                throw new InvalidOperationException("The block " + openBlocks[0] + " has not been closed.");
            }
            Reset();
            this.inferOnlySpecifiedVars = inferOnlySpecifiedVars;
            variablesToInfer.AddRange(vars);
            foreach (IVariable var in vars) toSearch.Push(var);
            while (toSearch.Count > 0)
            {
                IModelExpression expr = toSearch.Pop();
                SearchExpressionUntyped(expr);
            }
            // lock in the set of model expressions.
            ModelExpressions = new List<IModelExpression>(searched);
            List<int> timestamps = new List<int>();
            List<IModelExpression> exprs = new List<IModelExpression>();
            foreach (IModelExpression expr in ModelExpressions)
            {
                if (expr is Variable var)
                {
                    exprs.Add(var);
                    timestamps.Add(var.timestamp);
                }
                else if (expr is MethodInvoke mi)
                {
                    exprs.Add(mi);
                    timestamps.Add(mi.timestamp);
                }
            }
            Sort(timestamps, exprs);
            foreach (IModelExpression expr in exprs)
            {
                BuildExpressionUntyped(expr);
            }
            foreach (IModelExpression expr in exprs)
            {
                FinishExpressionUntyped(expr, engine.Algorithm);
            }
            return modelType;
        }

        /// <summary>
        /// Sort a pair of collections according to the values in the first collection
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="keys"></param>
        /// <param name="items"></param>
        private static void Sort<T, U>(ICollection<T> keys, ICollection<U> items)
        {
            T[] keyArray = keys.ToArray();
            U[] itemArray = items.ToArray();
            Array.Sort(keyArray, itemArray);
            keys.Clear();
            keys.AddRange(keyArray);
            items.Clear();
            items.AddRange(itemArray);
        }

        /// <summary>
        /// Set modelType.Name to a valid identifier.
        /// </summary>
        /// <param name="namespaceName">The desired namespace.  Must be a valid identifier.</param>
        /// <param name="name">The desired name.  Need not be a valid identifier.</param>
        public void SetModelName(string namespaceName, string name)
        {
            modelType.Namespace = namespaceName;
            string validName = CodeBuilder.MakeValid(name);
            modelType.Name = validName;
        }

        /// <summary>
        /// Get the abstract syntax tree for the generated code.
        /// </summary>
        /// <param name="engine"></param>
        /// <returns></returns>
        public List<ITypeDeclaration> GetGeneratedSyntax(InferenceEngine engine)
        {
            SetModelName(engine.ModelNamespace, engine.ModelName);
            return engine.Compiler.GetTransformedDeclaration(modelType, null, Attributes);
        }

        private void AddStatement(IStatement ist)
        {
            blockStack.Peek().Add(ist);
        }

        protected void BuildExpressionUntyped(IModelExpression var)
        {
            if (var == null) throw new NullReferenceException("Model expression was null.");
            // Console.WriteLine("Building expression: "+var+" "+builtVars.ContainsKey(var));
            if (var is MethodInvoke methodInvoke)
            {
                BuildMethodInvoke(methodInvoke);
                return;
            }
            MethodInfo mb = new Action<IModelExpression<object>>(this.BuildExpression<object>).Method.GetGenericMethodDefinition();

            Type domainType = null;
            // Look through the interfaces for this model expression (is there a better way of doing this?).
            // We expect to find IModelExpression<> - we can then get the type parameter from this interface
            Type[] faces = var.GetType().GetInterfaces();
            foreach (Type face in faces)
            {
                if (face.IsGenericType && face.GetGenericTypeDefinition() == typeof(IModelExpression<>))
                {
                    domainType = face.GetGenericArguments()[0];
                    break;
                }
            }
            if (domainType == null) throw new ArgumentException("Expression: " + var + " does not implement IModelExpression<>");
            // Construct the BuildExpression method for this type.
            MethodInfo mi2 = mb.MakeGenericMethod(domainType);
            // Invoke the typed BuildExpression method. This will recurse into BuildExpressionUntyped
            // as necessary
            Util.Invoke(mi2, this, var);
        }

        protected void FinishExpressionUntyped(IModelExpression expr, IAlgorithm alg)
        {
            if (expr is MethodInvoke) return;
            MethodInfo mb = new Action<IModelExpression<object>, IAlgorithm>(this.FinishExpression<object>).Method.GetGenericMethodDefinition();
            Type domainType = null;
            // Look through the interfaces for this model expression (is there a better way of doing this?).
            // We expect to find IModelExpression<> - we can then get the type parameter from this interface
            Type[] faces = expr.GetType().GetInterfaces();
            foreach (Type face in faces)
            {
                if (face.IsGenericType && face.GetGenericTypeDefinition() == typeof(IModelExpression<>))
                {
                    domainType = face.GetGenericArguments()[0];
                    break;
                }
            }
            if (domainType == null) throw new ArgumentException("Expression: " + expr + " does not implement IModelExpression<>");
            MethodInfo mi2 = mb.MakeGenericMethod(domainType);
            Util.Invoke(mi2, this, expr, alg);
        }

        protected void SearchExpressionUntyped(IModelExpression expr)
        {
            if (expr == null) throw new NullReferenceException("Model expression was null.");
            // Console.WriteLine("Searching expression: "+var+" "+builtVars.ContainsKey(var));
            if (searched.Contains(expr)) return;
            if (expr is MethodInvoke methodInvoke)
            {
                SearchMethodInvoke(methodInvoke);
                return;
            }
            if (expr is Range range)
            {
                SearchRange(range);
                return;
            }
            MethodInfo mb = new Action<IModelExpression<object>>(this.SearchExpression<object>).Method.GetGenericMethodDefinition();
            Type domainType = null;
            // Look through the interfaces for this model expression (is there a better way of doing this?).
            // We expect to find IModelExpression<> - we can then get the type parameter from this interface
            Type[] faces = expr.GetType().GetInterfaces();
            foreach (Type face in faces)
            {
                if (face.IsGenericType && face.GetGenericTypeDefinition() == typeof(IModelExpression<>))
                {
                    domainType = face.GetGenericArguments()[0];
                    break;
                }
            }
            if (domainType == null) throw new ArgumentException("Expression: " + expr + " does not implement IModelExpression<>");
            MethodInfo mi2 = mb.MakeGenericMethod(domainType);
            Util.Invoke(mi2, this, expr);
        }

        private void SearchMethodInvoke(MethodInvoke method)
        {
            if (searched.Contains(method)) return;
            searched.Add(method);
            if (method.returnValue != null)
            {
                IModelExpression target = method.returnValue;
                if (method.method.DeclaringType == target.GetType() && method.method.Name == new Func<bool>(Variable<bool>.RemovedBySetTo).Method.Name)
                    throw new InvalidOperationException("Variable '" + target +
                                                        "' was consumed by variable.SetTo().  It can no longer be used or inferred.  Perhaps you meant Variable.ConstrainEqual instead of SetTo.");
            }
            if (method.returnValue != null) toSearch.Push(method.returnValue);
            foreach (IModelExpression arg in method.args) toSearch.Push(arg);
            SearchContainers(method.Containers);
        }

        /// <summary>
        /// Add a statement of the form x = f(...) to the MSL.
        /// </summary>
        /// <param name="method">Stores the method to call, the argument variables, and target variable.</param>
        /// <remarks>
        /// If any variable in the statement is an item variable, then we surround the statement with a loop over its range.
        /// Since there may be multiple item variables, and each item may depend on multiple ranges, we may end up with multiple loops.
        /// </remarks>
        private void BuildMethodInvoke(MethodInvoke method)
        {
            if (method.ReturnValue is Variable variable && variable.Inline) return;
            // Open containing blocks
            List<IStatementBlock> stBlocks = method.Containers;
            List<Range> localRanges = new List<Range>();
            // each argument of method puts a partial order on the ranges.
            // e.g.  array[i,j][k]  requires i < k, j < k  but says nothing about i and j
            // we assemble these constraints into a total order.
            Dictionary<Range, int> indexOfRange = new Dictionary<Range, int>();
            Dictionary<IModelExpression, List<List<Range>>> dict = MethodInvoke.GetRangeBrackets(method.ReturnValueAndArgs());
            foreach (IModelExpression arg in method.ReturnValueAndArgs())
            {
                MethodInvoke.ForEachRange(arg,
                                          delegate (Range r) { if (!localRanges.Contains(r)) localRanges.Add(r); });
            }
            ParameterInfo[] pis = method.method.GetParameters();
            for (int i = 0; i < pis.Length; i++)
            {
                IModelExpression arg = method.Arguments[i];
                ParameterInfo pi = pis[i];
                if (pi.IsOut && 
                    arg is HasObservedValue argHasObservedValue && 
                    argHasObservedValue.IsObserved)
                {
                    throw new NotImplementedException(string.Format("Out parameter '{0}' of {1} cannot be observed.  Use ConstrainEqual or observe a copy of the variable.", pi.Name, method));
                }
            }
            foreach (IStatementBlock b in method.Containers)
            {
                if (b is HasRange br)
                {
                    localRanges.Remove(br.Range);
                }
            }
            localRanges.Sort(delegate (Range a, Range b) { return MethodInvoke.CompareRanges(dict, a, b); });
            // convert from List<Range> to List<IStatementBlock>
            List<IStatementBlock> localRangeBlocks = new List<IStatementBlock>(localRanges.Select(r => r));
            BuildStatementBlocks(stBlocks, true);
            BuildStatementBlocks(localRangeBlocks, true);

            // Invoke method
            IExpression methodExpr = method.GetExpression();
            foreach (ICompilerAttribute attr in method.attributes) Attributes.Add(methodExpr, attr);
            IStatement st = Builder.ExprStatement(methodExpr);
            if (methodExpr is IAssignExpression && 
                method.ReturnValue is HasObservedValue hasObservedValue && 
                hasObservedValue.IsObserved)
            {
                Attributes.Set(st, new Constraint());
            }
            AddStatement(st);

            BuildStatementBlocks(localRangeBlocks, false);
            BuildStatementBlocks(stBlocks, false);
        }

        internal void BuildStatementBlocks(List<IStatementBlock> statementBlocks, bool open)
        {
            if (open)
            {
                // Build statements around method invocation
                foreach (IStatementBlock isb in statementBlocks)
                {
                    IStatement ist = isb.GetStatement(out IList<IStatement> innerBlock);
                    if (ist != null)
                    {
                        ModifyStatement(ist, isb);
                        AddStatement(ist);
                        blockStack.Push(innerBlock);
                    }
                    else blockStack.Push(blockStack.Peek());
                }
            }
            else
            {
                // Close 'if' statements
                for (int i = statementBlocks.Count - 1; i >= 0; i--)
                {
                    blockStack.Pop();
                }
            }
        }

        private void ModifyStatement(IStatement ist, IStatementBlock isb)
        {
            if (isb is IfBlock ib)
            {
                var condVar = ib.ConditionVariable;
                if (!negatedConditionVariables.Contains(condVar))
                {
                    if (condVar.definition != null)
                    {
                        MethodInvoke mi = condVar.definition;
                        if (mi.method.Equals(new Func<int, int, bool>(Factor.AreEqual).Method))
                        {
                            if (mi.Arguments[1] is Variable arg1)
                            {
                                if (arg1.IsObserved || arg1.IsLoopIndex)
                                {
                                    // convert 'if(vbool1)' into 'if(x==value)'  where value is observed (or a loop index) and vbool1 is never negated.
                                    // if vbool1 is negated, then we cannot make this substitution since we need to match the corresponding 'if(!vbool1)' condition.
                                    IConditionStatement ics = (IConditionStatement)ist;
                                    ics.Condition = Builder.BinaryExpr(mi.Arguments[0].GetExpression(), BinaryOperator.ValueEquality, arg1.GetExpression());
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Search a variable expression
        /// </summary>
        /// <typeparam name="T">Domain type of the variable expression</typeparam>
        /// <param name="var">The variable expression</param>
        /// 
        public void SearchExpression<T>(IModelExpression<T> var)
        {
            if (var is Variable<T> varT) SearchVariable<T>(varT);
            else throw new InferCompilerException("Unhandled model expression type: " + var.GetType());
        }

        /// <summary>
        /// Define a variable in the MSL.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        private void SearchVariable<T>(Variable<T> variable)
        {
            if (searched.Contains(variable))
                return;
            if (variable.IsBase && variable.NameInGeneratedCode != null)
            {
                CheckNameIsUnique(variable.NameInGeneratedCode);
            }
            if (variable.IsLoopIndex)
                return; // do nothing

            searched.Add(variable); // must do this first to prevent re-entry
            SearchContainers(variable.Containers);
            if (variable.IsArrayElement)
            {
                SearchItem(variable);
            }
            else // base variable
            {                
                if (variable.initialiseTo != null)
                {
                    toSearch.Push(variable.initialiseTo);
                }

                if (variable.initialiseBackwardTo != null)
                {
                    toSearch.Push(variable.initialiseBackwardTo);
                }

                variable.Inline = ShouldInline(variable);

                if (variable is IVariableArray iva)
                {
                    // Check for invalid ranges
                    GetJaggedArrayIndicesAndSizes(iva, out IList<IVariableDeclaration[]> jaggedIndexVars, out IList<IExpression[]> jaggedSizes);
                }

                if (variable.definition != null)
                {
                    SearchMethodInvoke(variable.definition);
                }

                SearchVariableAttributes(variable);
            }

            foreach (MethodInvoke condDef in variable.conditionalDefinitions.Values)
            {
                SearchMethodInvoke(condDef);
            }

            if (variable is HasItemVariables hasItemVariables)
            {
                ICollection<IVariable> items = hasItemVariables.GetItemsUntyped().Values;
                foreach (IVariable irv in items)
                {
                    toSearch.Push(irv);
                }
            }

            foreach (MethodInvoke constraint in variable.constraints)
                SearchMethodInvoke(constraint);
            foreach (MethodInvoke factor in variable.childFactors)
                SearchMethodInvoke(factor);
        }

        private bool ShouldInline<T>(Variable<T> variable)
        {
            if (variable.IsObserved && variable.IsReadOnly)
            {
                return ShouldInlineConstant(variable);
            }

            bool inline;
            if (variable.definition != null)
            {
                inline = variable.definition.CanBeInlined();
            }
            else
            {
                inline = (variable.conditionalDefinitions.Values.Count == 1);
                foreach (MethodInvoke condDef in variable.conditionalDefinitions.Values)
                {
                    inline = inline && condDef.CanBeInlined();
                }
                if (variable is HasItemVariables hasItemVariables)
                {
                    var items = hasItemVariables.GetItemsUntyped();
                    if (items.Count > 0)
                        inline = false;
                }
            }

            return inline;
        }

        private void SearchVariableAttributes(Variable variable)
        {
            foreach (var attr in variable.GetAttributes<ICompilerAttribute>())
            {
                if (attr is ValueRange vr)
                {
                    SearchRange(vr.Range);
                }
                else if (attr is DistributedCommunication dc)
                {
                    toSearch.Push(dc.arrayIndicesToSendExpression);
                    toSearch.Push(dc.arrayIndicesToReceiveExpression);
                    var attr2 = new DistributedCommunicationExpression(dc.arrayIndicesToSendExpression.GetExpression(), dc.arrayIndicesToReceiveExpression.GetExpression());
                    // find the base variable
                    Variable parent = variable;
                    while (parent.ArrayVariable != null)
                    {
                        parent = (Variable)parent.ArrayVariable;
                    }
                    var parentDecl = parent.GetDeclaration();
                    if (Attributes.Has<DistributedCommunicationExpression>(parentDecl))
                        throw new Exception($"{parent} has multiple DistributedCommunication attributes");
                    Attributes.Set(parentDecl, attr2);
                }
            }
        }

        private void CheckNameIsUnique(string name)
        {
            foreach (IModelExpression expr in searched)
            {
                if (name.Equals(expr.Name))
                {
                    throw new InferCompilerException("Model contains multiple items with the name '" + name + "'.  Names must be unique.");
                }
            }
        }

        private void FinishExpression<T>(IModelExpression<T> expr, IAlgorithm alg)
        {
            if (expr is Variable<T> variable)
                FinishVariable<T>(variable, alg);
            else
                throw new InferCompilerException("Unhandled model expression type: " + expr.GetType());
        }

        private void FinishVariable<T>(Variable<T> variable, IAlgorithm alg)
        {
            if (variable.IsLoopIndex) return; // do nothing
            if (variable.IsArrayElement) return;
            if (variable.Inline) return;

            object ivd = variable.GetDeclaration();
            bool doNotInfer = false;
            // Add attributes
            foreach (ICompilerAttribute attr in variable.GetAttributes<ICompilerAttribute>())
            {
                if (attr is DoNotInfer) doNotInfer = true;
                else Attributes.Add(ivd, attr);
            }
            foreach (IStatementBlock stBlock in variable.Containers)
            {
                if (stBlock is HasRange)
                {
                    doNotInfer = true;
                    break;
                }
            }
            List<IStatementBlock> stBlocks = new List<IStatementBlock>();
            stBlocks.AddRange(variable.Containers);
            // Add Infer statement 
            bool isConstant = (variable.IsBase && variable.IsReadOnly);
            if (!doNotInfer && ((!inferOnlySpecifiedVars && !isConstant) || variablesToInfer.Contains(variable)))
            {
                // If there has been no explicit indication of query types for inference, set the
                // default types
                List<QueryTypeCompilerAttribute> qtlist = Attributes.GetAll<QueryTypeCompilerAttribute>(ivd);
                if (qtlist.Count == 0)
                {
                    alg.ForEachDefaultQueryType(qt => Attributes.Add(ivd, new QueryTypeCompilerAttribute(qt)));
                    qtlist = Attributes.GetAll<QueryTypeCompilerAttribute>(ivd);
                }
                variablesToInfer.Add(variable);
                BuildStatementBlocks(stBlocks, true);
                IExpression varExpr = variable.GetExpression();
                IExpression varName = Builder.LiteralExpr(variable.NameInGeneratedCode);
                foreach (QueryTypeCompilerAttribute qt in qtlist)
                {
                    IExpression queryExpr = Builder.FieldRefExpr(Builder.TypeRefExpr(typeof(QueryTypes)), typeof(QueryTypes), qt.QueryType.Name);
                    // for a constant, we must get the variable reference, not the value
                    if (isConstant) varExpr = Builder.VarRefExpr((IVariableDeclaration)variable.GetDeclaration());
                    AddStatement(Builder.ExprStatement(
                        Builder.StaticMethod(new Action<object, string, QueryType>(InferNet.Infer), varExpr, varName, queryExpr)));
                }
                BuildStatementBlocks(stBlocks, false);
            }
        }

        /// <summary>
        /// Build a variable expression
        /// </summary>
        /// <typeparam name="T">Domain type of the variable expression</typeparam>
        /// <param name="expr">The variable expression</param>
        private void BuildExpression<T>(IModelExpression<T> expr)
        {
            if (expr is Variable<T> var) BuildVariable<T>(var);
            else throw new InferCompilerException("Unhandled model expression type: " + expr.GetType());
        }

        /// <summary>
        /// Define a variable in the MSL.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// 
        private void BuildVariable<T>(Variable<T> variable)
        {
            if (variable.IsLoopIndex) return; // do nothing

            if (!variable.IsObserved)
            {
                BuildRandVar(variable);
            }
            else if (!variable.IsReadOnly)
            {
                BuildGiven(variable);
            }
            else
            {
                BuildConstant(variable);
            }
        }

        private bool ShouldInlineConstant<T>(Variable<T> constant)
        {
            return Quoter.ShouldInlineType(typeof(T)) && !constant.IsDefined && !variablesToInfer.Contains(constant);
        }

        /// <summary>
        /// Define a constant in the MSL.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="constant"></param>
        private void BuildConstant<T>(Variable<T> constant)
        {
            if (constant.IsBase)
            {
                if (ShouldInlineConstant(constant))
                {
                    // do nothing.  the value will be put inline.
                }
                else
                {
                    // check if we have defined a constant with the same value already
                    IVariableDeclaration ivd;
                    bool useExisting = false;
                    object key = constant.ObservedValue;
                    if (key is null) key = new NullValue<T>();
                    if (constants.TryGetValue(key, out ivd))
                    {
                        // use the declaration of the existing constant
                        useExisting = constant.SetDeclaration(ivd);
                    }
                    if (!useExisting)
                    {
                        // create a new declaration
                        ivd = (IVariableDeclaration)constant.GetDeclaration();
                        var rhs = Quoter.Quote(constant.ObservedValue);
                        if (constant.ObservedValue == null) rhs = Builder.CastExpr(rhs, typeof(T));
                        AddStatement(Builder.AssignStmt(Builder.VarDeclExpr(ivd), rhs));
                        constants[key] = ivd;
                    }
                    FinishGivenOrConstant(constant, ivd);
                }
            }
            else if (constant.IsArrayElement)
            {
                // do nothing
            }
            else
            {
                throw new NotImplementedException("Unhandled constant type: " + constant);
            }
        }

        private class NullValue<T>
        {
            public override bool Equals(object obj)
            {
                return (obj is NullValue<T>);
            }

            public override int GetHashCode()
            {
                return GetType().GetHashCode();
            }
        }

        /// <summary>
        /// Search all variables referred to by an item variable.
        /// </summary>
        /// <param name="item"></param>
        /// 
        private void SearchItem(Variable item)
        {
            foreach (IModelExpression ind in item.indices) toSearch.Push(ind);
            toSearch.Push(item.ArrayVariable);
            if (item.definition != null) toSearch.Push(item.definition);
            foreach (MethodInvoke constraint in item.constraints) toSearch.Push(constraint);
            if (item.initialiseTo != null) toSearch.Push(item.initialiseTo);
            if (item.initialiseBackwardTo != null)
                toSearch.Push(item.initialiseBackwardTo);
        }

        /// <summary>
        /// Search all variables referred to by a Range.
        /// </summary>
        /// <param name="range"></param>
        private void SearchRange(Range range)
        {
            if (searched.Contains(range)) return;
            string name = ((IModelExpression)range).Name;
            CheckNameIsUnique(name);
            searched.Add(range);
            toSearch.Push(range.Size);
            SearchRangeAttributes(range);
        }

        private void SearchRangeAttributes(Range range)
        {
            IVariableDeclaration ivd = range.GetIndexDeclaration();
            foreach (ICompilerAttribute attr in range.GetAttributes<ICompilerAttribute>())
            {
                if (attr is ParallelSchedule ps)
                {
                    toSearch.Push(ps.scheduleExpression);
                    var attr2 = new ParallelScheduleExpression(ps.scheduleExpression.GetExpression());
                    Attributes.Set(ivd, attr2);
                }
                else if (attr is DistributedSchedule ds)
                {
                    toSearch.Push(ds.commExpression);
                    if (ds.scheduleExpression != null)
                        toSearch.Push(ds.scheduleExpression);
                    if (ds.schedulePerThreadExpression != null)
                        toSearch.Push(ds.schedulePerThreadExpression);
                    var attr2 = new DistributedScheduleExpression(ds.commExpression.GetExpression(), ds.scheduleExpression?.GetExpression(), ds.schedulePerThreadExpression?.GetExpression());
                    Attributes.Set(ivd, attr2);
                }
                else
                {
                    Attributes.Set(ivd, attr);
                }
            }
        }

        /// <summary>
        /// Build condition variable expressions associated with each condition block
        /// Build range variable expressions associated with each range
        /// </summary>
        /// <param name="containers">Containers - condition blocks or foreach block</param>
        /// 
        private void SearchContainers(IEnumerable<IStatementBlock> containers)
        {
            foreach (IStatementBlock sb in containers)
            {
                if (sb is ConditionBlock cb)
                {
                    if (cb is SwitchBlock swb)
                    {
                        SearchRange(swb.Range);
                    }
                    Variable condVar = cb.ConditionVariableUntyped;
                    if (cb is IfBlock ib)
                    {
                        if (ib.ConditionValue == false) negatedConditionVariables.Add(condVar);
                    }
                    toSearch.Push(condVar);
                }
                else if (sb is ForEachBlock fb)
                {
                    SearchRange(fb.Range);
                }
                else if (sb is RepeatBlock rb)
                {
                    toSearch.Push(rb.Count);
                }
            }
        }

        /// <summary>
        /// Add the definition of a random variable to the MSL, inside of the necessary containers.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// <remarks>
        /// A scalar variable is declared and defined in one line such as: <c>int x = factor(...);</c>.
        /// An array variable is first declared with an initializer such as: <c>int[] array = new int[4];</c>.
        /// Then it is defined either with a bulk factor such as: <c>array = factor(...);</c>,
        /// or it is defined via its item variable.
        /// An item variable is defined by 'for' loop whose body is: <c>array[i] = factor(...);</c>.
        /// </remarks>
        protected void BuildRandVar<T>(Variable<T> variable)
        {
            if (!variable.IsDefined) throw new InferCompilerException("Variable '" + variable + "' has no definition");
            if (variable.IsArrayElement)
            {
                for (int initType = 0; initType < 2; initType++)
                {
                    IModelExpression init = (initType == 0) ? variable.initialiseTo : variable.initialiseBackwardTo;
                    if (init != null)
                    {
                        IExpression initExpr = init.GetExpression();
                        // find the base variable
                        Variable parent = variable;
                        while (parent.ArrayVariable != null)
                        {
                            IVariableDeclaration[] indexVars = new IVariableDeclaration[parent.indices.Count];
                            for (int i = 0; i < indexVars.Length; i++)
                            {
                                IModelExpression expr = parent.indices[i];
                                if (!(expr is Range))
                                    throw new Exception(parent + ".InitializeTo is not allowed since the indices are not ranges");
                                indexVars[i] = ((Range)expr).GetIndexDeclaration();
                            }
                            initExpr = VariableInformation.MakePlaceHolderArrayCreate(initExpr, indexVars);
                            parent = (Variable)parent.ArrayVariable;
                        }
                        IVariableDeclaration parentDecl = (IVariableDeclaration)parent.GetDeclaration();
                        ICompilerAttribute attr;
                        if (initType == 0)
                            attr = new InitialiseTo(initExpr);
                        else attr = new InitialiseBackwardTo(initExpr);
                        Attributes.Set(parentDecl, attr);
                    }
                }
                return;
            }
            IVariableDeclaration ivd = (IVariableDeclaration)variable.GetDeclaration();
            if (variable.initialiseTo != null)
            {
                Attributes.Set(ivd, new InitialiseTo(variable.initialiseTo.GetExpression()));
            }
            if (variable.initialiseBackwardTo != null)
            {
                Attributes.Set(ivd, new InitialiseBackwardTo(variable.initialiseBackwardTo.GetExpression()));
            }
            List<IStatementBlock> stBlocks = new List<IStatementBlock>();
            stBlocks.AddRange(variable.Containers);

            IVariableDeclarationExpression ivde = Builder.VarDeclExpr(ivd);
            if (variable is IVariableArray iva)
            {
                IList<IStatement> sc = Builder.StmtCollection();
                GetJaggedArrayIndicesAndSizes(iva, out IList<IVariableDeclaration[]> jaggedIndexVars, out IList<IExpression[]> jaggedSizes);
                // check that containers are all unique and distinct from jaggedIndexVars
                Set<IVariableDeclaration> loopVars = new Set<IVariableDeclaration>();
                foreach (IStatementBlock stBlock in stBlocks)
                {
                    if (stBlock is ForEachBlock fb)
                    {
                        IVariableDeclaration loopVar = fb.Range.GetIndexDeclaration();
                        if (loopVars.Contains(loopVar))
                            throw new InvalidOperationException("Variable '" + ivd.Name + "' uses range '" + loopVar.Name + "' twice. Use a cloned range instead.");
                        loopVars.Add(loopVar);
                    }
                }
                foreach (IVariableDeclaration[] bracket in jaggedIndexVars)
                {
                    foreach (IVariableDeclaration indexVar in bracket)
                    {
                        if (loopVars.Contains(indexVar))
                            throw new InvalidOperationException("Variable '" + ivd.Name + "' uses range '" + indexVar.Name + "' twice. Use a cloned range instead.");
                    }
                }
                Builder.NewJaggedArray(sc, ivd, jaggedIndexVars, jaggedSizes);
                if (!variable.Inline)
                {
                    BuildStatementBlocks(stBlocks, true);
                    foreach (IStatement stmt in sc)
                    {
                        AddStatement(stmt);
                    }
                    BuildStatementBlocks(stBlocks, false);
                }
                ivde = null; // prevent re-declaration
            }
            if (ivde != null)
            {
                if (!variable.Inline)
                {
                    BuildStatementBlocks(stBlocks, true);
                    AddStatement(Builder.ExprStatement(ivde));
                    BuildStatementBlocks(stBlocks, false);
                }
                ivde = null;
            }
            if (ivde != null) throw new InferCompilerException("Variable '" + variable + "' has no definition");
        }

        protected void FinishRandVar<T>(Variable<T> variable, IAlgorithm alg)
        {
        }

        protected void GetJaggedArrayIndicesAndSizes(IVariableArray array, out IList<IVariableDeclaration[]> jaggedIndexVars, out IList<IExpression[]> jaggedSizes)
        {
            Set<IVariableDeclaration> allVars = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
            jaggedIndexVars = new List<IVariableDeclaration[]>();
            jaggedSizes = new List<IExpression[]>();
            while (true)
            {
                Type arrayType = array.GetExpression().GetExpressionType();
                Type elementType = Util.GetElementType(arrayType, out int rank);
                if (!arrayType.IsAssignableFrom(Util.MakeArrayType(elementType, rank)))
                {
                    break;
                }
                GetArrayIndicesAndSizes(array, out IVariableDeclaration[] indexVars, out IExpression[] sizes);
                foreach (IVariableDeclaration ivd in indexVars)
                {
                    if (allVars.Contains(ivd))
                        throw new CompilationFailedException("Array '" + array.Name + "' is indexed by range '" + ivd.Name +
                                                             "' on multiple dimensions, which is not allowed.  Use range cloning instead.");
                    allVars.Add(ivd);
                }
                jaggedIndexVars.Add(indexVars);
                jaggedSizes.Add(sizes);
                if (array is IVariableJaggedArray variableJaggedArray)
                {
                    IVariable itemPrototype = variableJaggedArray.ItemPrototype;
                    if (itemPrototype is IVariableArray variableArray) array = variableArray;
                    else break;
                }
                else break;
            }
        }

        protected void GetArrayIndicesAndSizes(IVariableArray array, out IVariableDeclaration[] indexVars, out IExpression[] sizes)
        {
            IList<Range> ranges = array.Ranges;
            sizes = new IExpression[ranges.Count];
            indexVars = new IVariableDeclaration[ranges.Count];
            int i = 0;
            foreach (Range r in ranges)
            {
                SearchRange(r);
                indexVars[i] = r.GetIndexDeclaration();
                sizes[i] = r.GetSizeExpression();
                i++;
            }
        }

        /// <summary>
        /// Define a given variable in the MSL.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="variable"></param>
        /// 
        protected void BuildGiven<T>(Variable<T> variable)
        {
            if (variable.IsBase)
            {
                IParameterDeclaration ipd = (IParameterDeclaration)variable.GetDeclaration();
                modelMethod.Parameters.Add(ipd);
                FinishGivenOrConstant(variable, ipd);
                // must do this after the sizes are built
                observedVars.Add(variable);
            }
            else if (variable.IsArrayElement)
            {
                // do nothing
            }
            else
            {
                throw new NotImplementedException("Unhandled given type: " + variable);
            }
        }

        private void FinishGivenOrConstant(Variable variable, object decl)
        {
            if (variable is IVariableArray iva)
            {
                // attach a VariableInformation attribute that holds the array sizes
                // this is needed because the array size never appears in the code
                GetJaggedArrayIndicesAndSizes(iva, out IList<IVariableDeclaration[]> jaggedIndexVars, out IList<IExpression[]> jaggedSizes);
                var vi = new VariableInformation(decl);
                vi.sizes.AddRange(jaggedSizes);
                vi.indexVars.AddRange(jaggedIndexVars);
                Attributes.Set(decl, vi);

                // see ModelTests.RangeWithConstantSizeTest
                bool addLengthConstraints = false;
                if (addLengthConstraints)
                {
                    IExpression array = variable.GetExpression();
                    IExpression valueIsNotNull = Builder.BinaryExpr(array, BinaryOperator.ValueInequality, Builder.LiteralExpr(null));
                    IConditionStatement cs = Builder.CondStmt(valueIsNotNull, Builder.BlockStmt());
                    var constraint = new Action<int, int>(Constrain.Equal);
                    Type arrayType = array.GetExpressionType();
                    int rank = vi.sizes[0].Length;
                    //Util.GetElementType(arrayType, out rank);
                    for (int i = 0; i < rank; i++)
                    {
                        IExpression lengthExpr;
                        if (rank == 1)
                        {
                            lengthExpr = Builder.PropRefExpr(array, arrayType, arrayType.IsArray ? "Length" : "Count", typeof(int));
                        }
                        else
                        {
                            lengthExpr = Builder.Method(array, typeof(Array).GetMethod("GetLength"), Builder.LiteralExpr(i));
                        }
                        cs.Then.Statements.Add(Builder.ExprStatement(
                            Builder.StaticMethod(constraint, lengthExpr, vi.sizes[0][i])));
                    }
                    AddStatement(cs);
                }
            }
        }

        public string ModelString()
        {
            return CodeCompiler.DeclarationToString(modelType);
        }
    }

    /// <summary>
    /// Attached to assignment statements to indicate that they are not really assignments but equality constraints between the left and right hand side.
    /// </summary>
    internal class Constraint : ICompilerAttribute
    {
    }
}