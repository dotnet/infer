// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Reflection;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Transform which:
    ///   Removes all methods other than the one containing the model.
    ///   Calls to InferNet.PreserveWhenCompiled are stripped out.
    ///   Switch cases with value less than zero are stripped out.
    ///   Evaluate some expressions that contain literals.
    ///   Processes and removes any Attrib static calls, replacing them with attributes on the appropriate code elements.
    ///   Replace variables in condition blocks with their conditioned values.
    ///   Attach index variable information to arrays (via VariableInformation attributes).
    ///   Attach Constraint attributes to assignments that are actually constraints.
    /// </summary>
    internal class ModelAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ModelAnalysisTransform"; }
        }

        private readonly ExpressionEvaluator evaluator = new ExpressionEvaluator();
        private readonly List<ConditionBinding> conditionContext = new List<ConditionBinding>();
        /// <summary>
        /// Used to track recursive calls to ConvertArrayCreate
        /// </summary>
        private bool convertingArrayCreate;
        private readonly Stack<IStatement> arrayCreateStmts = new Stack<IStatement>();

        /// <summary>
        /// Used to generate unique class names
        /// </summary>
        private static readonly Set<string> compiledClassNames = new Set<string>();
        private static readonly object compiledClassNamesLock = new object();

        /// <summary>
        /// Number of variables marked for inference
        /// </summary>
        private int inferCount = 0;

        public override void ConvertTypeProperties(ITypeDeclaration td, ITypeDeclaration itd)
        {
            base.ConvertTypeProperties(td, itd);
            // Set td.Name to a valid identifier that is unique from all previously generated classes, by adding an index as appropriate.
            string baseName = td.Name;
            int count = 0;
            lock (compiledClassNamesLock)
            {
                while (compiledClassNames.Contains(td.Namespace + "." + td.Name))
                {
                    td.Name = baseName + count;
                    count++;
                }
                compiledClassNames.Add(td.Namespace + "." + td.Name);
            }
        }

        protected override void ConvertNestedTypes(ITypeDeclaration td, ITypeDeclaration itd)
        {
            // remove all nested types            
        }

        protected override void ConvertProperties(ITypeDeclaration td, ITypeDeclaration itd)
        {
            // remove all properties
        }

        /// <summary>
        /// Analyses the method specified in MethodToTransform, if any.  Otherwise analyses all methods.
        /// </summary>
        /// <param name="imd"></param>
        /// <returns></returns>
        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            IMethodDeclaration imd2 = base.ConvertMethod(imd);
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            //td.Documentation = "model '"+imd.Name+"'";
            if (inferCount == 0)
            {
                Error("No variables were marked for inference, please mark some variables with InferNet.Infer(var).");
            }
            return imd2;
        }

        protected override IStatement ConvertSwitch(ISwitchStatement iss)
        {
            ISwitchStatement ss = Builder.SwitchStmt();
            context.SetPrimaryOutput(ss);
            ss.Expression = ConvertExpression(iss.Expression);
            foreach (ISwitchCase isc in iss.Cases)
            {
                if (isc is IConditionCase icc)
                {
                    IExpression cond = icc.Condition;
                    if ((cond is ILiteralExpression ile) && (ile.Value is int i) && (i < 0)) continue;
                }
                ConvertSwitchCase(ss.Cases, isc);
            }
            return ss;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IParameterDeclaration ipd = null;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            object decl = ivd;
            if (ivd == null)
            {
                ipd = Recognizer.GetParameterDeclaration(iae.Target);
                if (ipd == null)
                    return base.ConvertAssign(iae);
                decl = ipd;
            }
            if (iae.Target is IArrayIndexerExpression)
            {
                // Gather index variables from the left-hand side of the assignment
                VariableInformation vi = VariableInformation.GetVariableInformation(context, decl);
                try
                {
                    List<IVariableDeclaration[]> indVars = new List<IVariableDeclaration[]>();
                    Recognizer.AddIndexers(context, indVars, iae.Target);
                    int depth = Recognizer.GetIndexingDepth(iae.Target);
                    // if this statement is actually a constraint, then we don't need to enforce matching of index variables
                    bool isConstraint = context.InputAttributes.Has<Models.Constraint>(context.FindAncestor<IStatement>());
                    for (int i = 0; i < depth; i++)
                    {
                        vi.SetIndexVariablesAtDepth(i, indVars[i], allowMismatch: isConstraint);
                    }
                }
                catch (Exception ex)
                {
                    Error(ex.Message, ex);
                }
            }
            IAssignExpression ae = (IAssignExpression) base.ConvertAssign(iae);
            if (ipd == null)
            {
                // assignment to a local variable
                if (ae.Expression is IMethodInvokeExpression imie)
                {
                    // this unfortunately duplicates some of the work done by SetStoch and IsStoch.
                    FactorManager.FactorInfo info = CodeRecognizer.GetFactorInfo(context, imie);
                    if (info != null && info.IsDeterministicFactor && !context.InputAttributes.Has<DerivedVariable>(ivd))
                    {
                        context.InputAttributes.Set(ivd, new DerivedVariable());
                    }
                }
                if (ae.Expression is ILiteralExpression)
                {
                    bool isLoopInitializer = (Recognizer.GetAncestorIndexOfLoopBeingInitialized(context) != -1);
                    if (!isLoopInitializer)
                    {
                        Type valueType = ae.Expression.GetExpressionType();
                        if (Quoter.ShouldInlineType(valueType))
                        {
                            // inline all future occurrences of this variable with the rhs expression
                            conditionContext.Add(new ConditionBinding(ae.Target, ae.Expression));
                        }
                    }
                }
            }
            else
            {
                // assignment to a method parameter
                IStatement ist = context.FindAncestor<IStatement>();
                if (!context.InputAttributes.Has<Models.Constraint>(ist))
                {
                    // mark this statement as a constraint
                    context.OutputAttributes.Set(ist, new Models.Constraint());
                }
            }
            // a FactorAlgorithm attribute on a variable turns into an Algorithm attribute on its right hand side.
            var attr = context.InputAttributes.Get<FactorAlgorithm>(decl);
            if (attr != null)
            {
                context.OutputAttributes.Set(ae.Expression, new Algorithm(attr.algorithm));
            }
            context.InputAttributes.CopyObjectAttributesTo<GivePriorityTo>(decl, context.OutputAttributes, ae.Expression);
            return ae;
        }

        private void CheckMethodArgumentCount(IMethodInvokeExpression imie)
        {
            MethodInfo method = (MethodInfo)imie.Method.Method.MethodInfo;
            var parameters = method.GetParameters();
            if (parameters.Length != imie.Arguments.Count)
            {
                Error($"Method given {imie.Arguments.Count} argument(s) but expected {parameters.Length}");
            }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            CheckMethodArgumentCount(imie);
            if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, ICompilerAttribute, PlaceHolder>(Attrib.Var)))
            {
                IVariableReferenceExpression ivre = imie.Arguments[0] as IVariableReferenceExpression;
                IVariableDeclaration target = ivre.Variable.Resolve();
                IExpression expr = CodeRecognizer.RemoveCast(imie.Arguments[1]);
                AddAttribute(target, expr);
                return null;
            }
            else if (Recognizer.IsStaticMethod(imie, new Action<object, object>(Attrib.InitialiseTo)))
            {
                IVariableReferenceExpression ivre = CodeRecognizer.RemoveCast(imie.Arguments[0]) as IVariableReferenceExpression;
                IVariableDeclaration target = ivre.Variable.Resolve();
                context.OutputAttributes.Set(target, new InitialiseTo(imie.Arguments[1]));
                return null;
            }
            else if (CodeRecognizer.IsInfer(imie))
            {
                inferCount++;
                object decl = Recognizer.GetDeclaration(imie.Arguments[0]);
                if (decl != null && !context.InputAttributes.Has<IsInferred>(decl))
                    context.InputAttributes.Set(decl, new IsInferred());
                // the arguments must not be substituted for their values, so we don't call ConvertExpression
                var newArgs = imie.Arguments.Select(CodeRecognizer.RemoveCast);
                IMethodInvokeExpression infer = Builder.MethodInvkExpr();
                infer.Method = imie.Method;
                infer.Arguments.AddRange(newArgs);
                context.InputAttributes.CopyObjectAttributesTo(imie, context.OutputAttributes, infer);
                return infer;                
            }
            IExpression converted = base.ConvertMethodInvoke(imie);
            if (converted is IMethodInvokeExpression mie)
            {
                bool isAnd = Recognizer.IsStaticMethod(converted, new Func<bool, bool, bool>(Factors.Factor.And));
                bool isOr = Recognizer.IsStaticMethod(converted, new Func<bool, bool, bool>(Factors.Factor.Or));
                bool anyArgumentIsLiteral = mie.Arguments.Any(arg => arg is ILiteralExpression);
                if (anyArgumentIsLiteral)
                {
                    if (isAnd)
                    {
                        if (mie.Arguments.Any(arg => arg is ILiteralExpression ile && ile.Value.Equals(false)))
                            return Builder.LiteralExpr(false);
                        // any remaining literals must be true, and therefore can be ignored.
                        var reducedArguments = mie.Arguments.Where(arg => !(arg is ILiteralExpression));
                        if (reducedArguments.Count() == 1) return reducedArguments.First();
                        else return Builder.LiteralExpr(true);
                    }
                    else if (isOr)
                    {
                        if (mie.Arguments.Any(arg => arg is ILiteralExpression ile && ile.Value.Equals(true)))
                            return Builder.LiteralExpr(true);
                        // any remaining literals must be false, and therefore can be ignored.
                        var reducedArguments = mie.Arguments.Where(arg => !(arg is ILiteralExpression));
                        if (reducedArguments.Count() == 1) return reducedArguments.First();
                        else return Builder.LiteralExpr(false);
                    }
                    else if (Recognizer.IsStaticMethod(converted, new Func<bool, bool>(Factors.Factor.Not)))
                    {
                        bool allArgumentsAreLiteral = mie.Arguments.All(arg => arg is ILiteralExpression);
                        if (allArgumentsAreLiteral)
                        {
                            return Builder.LiteralExpr(evaluator.Evaluate(mie));
                        }
                    }
                }
                foreach (IExpression arg in mie.Arguments)
                {
                    if (arg is IAddressOutExpression iaoe)
                    {
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iaoe.Expression);
                        if (ivd != null)
                        {
                            FactorManager.FactorInfo info = CodeRecognizer.GetFactorInfo(context, mie);
                            if (info != null && info.IsDeterministicFactor && !context.InputAttributes.Has<DerivedVariable>(ivd))
                            {
                                context.InputAttributes.Set(ivd, new DerivedVariable());
                            }
                        }
                    }
                }
            }
            return converted;
        }

        protected override IExpression ConvertCastExpr(ICastExpression ice)
        {
            return CodeRecognizer.RemoveCast(base.ConvertCastExpr(ice));
        }

        private void AddAttribute(object target, IExpression attrExpr)
        {
            try
            {
                if (!(evaluator.Evaluate(attrExpr) is ICompilerAttribute value))
                {
                    throw new InvalidExpressionException("Expression must evaluate to an ICompilerAttribute");
                }
                // TODO: Fix this temporary hack to allow quoting of this expression later
                if (value is MarginalPrototype mp)
                {
                    mp.prototypeExpression = ((IObjectCreateExpression) attrExpr).Arguments[0];
                }
                object[] auas = value.GetType().GetCustomAttributes(typeof (AttributeUsageAttribute), true);
                if (auas.Length > 0)
                {
                    AttributeUsageAttribute aua = (AttributeUsageAttribute) auas[0];
                    if (!aua.AllowMultiple) Context.InputAttributes.RemoveOfType(target, value.GetType());
                }
                Context.InputAttributes.Set(target, value);
            }
            catch (Exception ex)
            {
                Error("Could not evaluate attribute " + attrExpr, ex);
            }
        }

        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            IArrayCreateExpression ace = (IArrayCreateExpression) base.ConvertArrayCreate(iace);
            IAssignExpression iae = context.FindAncestor<IAssignExpression>();
            if (iae == null) return ace;
            if (iae.Expression != iace) return ace;
            if (iace.Initializer != null)
            {
                var exprs = iace.Initializer.Expressions;
                bool expandInitializer = !AllElementsAreLiteral(exprs);
                if (expandInitializer)
                {
                    // convert the initializer to a list of assignment statements to literal indices
                    bool wasConvertingArrayCreate = convertingArrayCreate;
                    if (!wasConvertingArrayCreate)
                        arrayCreateStmts.Clear();
                    convertingArrayCreate = true;
                    IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                    // Sets the size of this variable at this array depth
                    int depth = Recognizer.GetIndexingDepth(iae.Target);
                    IExpression[] dimExprs = new IExpression[ace.Dimensions.Count];
                    for (int i = 0; i < dimExprs.Length; i++)
                        dimExprs[i] = ace.Dimensions[i];
                    List<IList<IExpression>> indices = Recognizer.GetIndices(iae.Target);
                    // for a multi-dimensional array, exprs will contain IBlockExpressions
                    // dimExprs must be ILiteralExpressions
                    int[] dims = Util.ArrayInit(dimExprs.Length, i => (int)((ILiteralExpression)dimExprs[i]).Value);
                    int[] strides = StringUtil.ArrayStrides(dims);
                    int elementCount = strides[0] * dims[0];
                    var target = Builder.JaggedArrayIndex(Builder.VarRefExpr(ivd), indices);
                    int[] mIndex = new int[dims.Length];
                    for (int linearIndex = elementCount - 1; linearIndex >= 0; linearIndex--)
                    {
                        StringUtil.LinearIndexToMultidimensionalIndex(linearIndex, strides, mIndex);
                        var indexExprs = Util.ArrayInit(mIndex.Length, i => Builder.LiteralExpr(mIndex[i]));
                        var lhs = Builder.ArrayIndex(target, indexExprs);
                        var expr = GetInitializerElement(exprs, mIndex);
                        var assignStmt = Builder.AssignStmt(lhs, expr);
                        var st = ConvertStatement(assignStmt);
                        arrayCreateStmts.Push(st);
                    }
                    ace.Initializer = null;
                    convertingArrayCreate = wasConvertingArrayCreate;
                    if (!wasConvertingArrayCreate)
                    {
                        context.AddStatementsAfterCurrent(arrayCreateStmts);
                    }
                }
            }
            return ace;
        }

        protected static bool AllElementsAreLiteral(IList<IExpression> exprs)
        {
            foreach (var expr in exprs)
            {
                if (expr is IBlockExpression ibe)
                {
                    if (!AllElementsAreLiteral(ibe.Expressions))
                        return false;
                }
                else if (!(expr is ILiteralExpression))
                {
                    return false;
                }
            }
            return true;
        }

        protected static IExpression GetInitializerElement(IList<IExpression> exprs, int[] mIndex, int dim = 0)
        {
            var expr = exprs[mIndex[dim]];
            if (dim == mIndex.Length - 1)
            {
                return expr;
            }
            else
            {
                var blockExpr = (IBlockExpression)expr;
                return GetInitializerElement(blockExpr.Expressions, mIndex, dim + 1);
            }
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            IConditionStatement cs = Builder.CondStmt();
            cs.Condition = ConvertExpression(ics.Condition);
            if (cs.Condition is ILiteralExpression ile)
            {
                bool value = (bool) ile.Value;
                if (value)
                {
                    if (ics.Then != null)
                    {
                        foreach (IStatement st in ics.Then.Statements)
                        {
                            IStatement ist = ConvertStatement(st);
                            if (ist != null) context.AddStatementBeforeCurrent(ist);
                        }
                    }
                }
                else
                {
                    if (ics.Else != null)
                    {
                        foreach (IStatement st in ics.Else.Statements)
                        {
                            IStatement ist = ConvertStatement(st);
                            if (ist != null) context.AddStatementBeforeCurrent(ist);
                        }
                    }
                }
                return null;
            }
            context.SetPrimaryOutput(cs);
            ConditionBinding binding = GateTransform.GetConditionBinding(cs.Condition, context, out IForStatement loop);
            int startIndex = conditionContext.Count;
            conditionContext.Add(binding);
            cs.Then = ConvertBlock(ics.Then);
            if (ics.Else != null)
            {
                conditionContext.RemoveRange(startIndex, conditionContext.Count - startIndex);
                binding = binding.FlipCondition();
                conditionContext.Add(binding);
                cs.Else = ConvertBlock(ics.Else);
            }
            conditionContext.RemoveRange(startIndex, conditionContext.Count - startIndex);
            if (cs.Then.Statements.Count == 0 && (cs.Else == null || cs.Else.Statements.Count == 0)) return null;
            return cs;
        }

        protected override IExpression DoConvertExpression(IExpression expr)
        {
            expr = base.DoConvertExpression(expr);
            if (expr != null)
            {
                foreach (ConditionBinding ci in conditionContext)
                {
                    // each lhs has already been replaced, so we only need to compare for equality
                    if (expr.Equals(ci.lhs)) return ci.rhs;
                }
            }
            return expr;
        }

        protected override IExpression ConvertUnary(IUnaryExpression iue)
        {
            iue = (IUnaryExpression)base.ConvertUnary(iue);
            if (iue.Operator == UnaryOperator.BooleanNot)
            {
                if (iue.Expression is ILiteralExpression expr)
                {
                    if (expr.Value is bool b)
                    {
                        return Builder.LiteralExpr(!b);
                    }
                }
                else if (iue.Expression is IUnaryExpression iue2)
                {
                    if (iue2.Operator == UnaryOperator.BooleanNot) // double negation
                        return iue2.Expression;
                }
                else if (iue.Expression is IBinaryExpression ibe)
                {
                    if (Recognizer.TryNegateOperator(ibe.Operator, out BinaryOperator negatedOp))
                    {
                        // replace !(i==0) with (i != 0)
                        return Builder.BinaryExpr(ibe.Left, negatedOp, ibe.Right);
                    }
                }
            }
            return iue;
        }

        protected override IExpression ConvertBinary(IBinaryExpression ibe)
        {
            ibe = (IBinaryExpression) base.ConvertBinary(ibe);
            if (ibe.Left is ILiteralExpression && ibe.Right is ILiteralExpression)
            {
                try
                {
                    return Builder.LiteralExpr(evaluator.Evaluate(ibe));
                }
                catch
                {
                }
            }
            return ibe;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}