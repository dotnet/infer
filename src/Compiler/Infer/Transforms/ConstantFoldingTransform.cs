// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Data;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Transform which:
    ///   Switch cases with value less than zero are stripped out.
    ///   Evaluate some expressions that contain literals.
    ///   Processes and removes any Attrib static calls, replacing them with attributes on the appropriate code elements.
    ///   Replace variables in condition blocks with their conditioned values.
    /// </summary>
    internal class ConstantFoldingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ConstantFoldingTransform"; }
        }

        private readonly ExpressionEvaluator evaluator = new ExpressionEvaluator();
        private readonly List<ConditionBinding> conditionContext = new List<ConditionBinding>();
        /// <summary>
        /// Used to track recursive calls to ConvertArrayCreate
        /// </summary>
        private bool convertingArrayCreate;
        private readonly Stack<IStatement> arrayCreateStmts = new Stack<IStatement>();

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
            if (ivd == null)
            {
                ipd = Recognizer.GetParameterDeclaration(iae.Target);
                if (ipd == null)
                    return base.ConvertAssign(iae);
            }
            IAssignExpression ae = (IAssignExpression)base.ConvertAssign(iae);
            if (ipd == null && !context.InputAttributes.Has<IsInferred>(ivd))
            {
                // assignment to a local variable
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
            }
            return ae;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            IExpression converted = base.ConvertMethodInvoke(imie);
            if (converted is IMethodInvokeExpression mie)
            {
                if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, ICompilerAttribute, PlaceHolder>(Attrib.Var)))
                {
                    IVariableReferenceExpression ivre = imie.Arguments[0] as IVariableReferenceExpression;
                    IVariableDeclaration target = ivre.Variable.Resolve();
                    IExpression expr = imie.Arguments[1];
                    AddAttribute(target, expr);
                    return null;
                }
                else if (Recognizer.IsStaticMethod(imie, new Action<object, object>(Attrib.InitialiseTo)))
                {
                    IVariableReferenceExpression ivre = imie.Arguments[0] as IVariableReferenceExpression;
                    IVariableDeclaration target = ivre.Variable.Resolve();
                    context.OutputAttributes.Set(target, new InitialiseTo(imie.Arguments[1]));
                    return null;
                }
                else if (CodeRecognizer.IsInfer(imie))
                {
                    // the arguments must not be substituted for their values, so we don't call ConvertExpression
                    IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(imie.Arguments[0]);
                    if (ivd != null)
                    {
                        var vi = VariableInformation.GetVariableInformation(context, ivd);
                        QueryType query = (imie.Arguments.Count < 3) ? null : (QueryType)evaluator.Evaluate(imie.Arguments[2]);
                        vi.NeedsMarginalDividedByPrior = (query == QueryTypes.MarginalDividedByPrior);
                    }
                    return imie;
                }
                bool anyArgumentIsLiteral = mie.Arguments.Any(arg => arg is ILiteralExpression);
                if (anyArgumentIsLiteral)
                {
                    if (Recognizer.IsStaticMethod(converted, new Func<bool, bool, bool>(Factors.Factor.And)))
                    {
                        if (mie.Arguments.Any(arg => arg is ILiteralExpression ile && ile.Value.Equals(false)))
                            return Builder.LiteralExpr(false);
                        // any remaining literals must be true, and therefore can be ignored.
                        var reducedArguments = mie.Arguments.Where(arg => !(arg is ILiteralExpression));
                        if (reducedArguments.Count() == 1) return reducedArguments.First();
                        else return Builder.LiteralExpr(true);
                    }
                    else if (Recognizer.IsStaticMethod(converted, new Func<bool, bool, bool>(Factors.Factor.Or)))
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
            }
            return converted;
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
                    mp.prototypeExpression = ((IObjectCreateExpression)attrExpr).Arguments[0];
                }
                object[] auas = value.GetType().GetCustomAttributes(typeof(AttributeUsageAttribute), true);
                if (auas.Length > 0)
                {
                    AttributeUsageAttribute aua = (AttributeUsageAttribute)auas[0];
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
            IArrayCreateExpression ace = (IArrayCreateExpression)base.ConvertArrayCreate(iace);
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

            bool AllElementsAreLiteral(IList<IExpression> exprs)
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

            IExpression GetInitializerElement(IList<IExpression> exprs, int[] mIndex, int dim = 0)
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
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            IConditionStatement cs = Builder.CondStmt();
            cs.Condition = ConvertExpression(ics.Condition);
            if (cs.Condition is ILiteralExpression ile)
            {
                bool value = (bool)ile.Value;
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
            if (ReferenceEquals(cs.Condition, ics.Condition) && ReferenceEquals(cs.Then, ics.Then) && ReferenceEquals(cs.Else, ics.Else))
                return ics;
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
            ibe = (IBinaryExpression)base.ConvertBinary(ibe);
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
}
