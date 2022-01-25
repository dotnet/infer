// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Optimises message passing by removing redundant messages and operations.
    /// </summary>
    internal class CopyPropagationTransform : ShallowCopyTransform
    {
        private readonly Stack<Type> formalTypeStack = new Stack<Type>();

        public override string Name
        {
            get { return "CopyPropagationTransform"; }
        }

        /// <summary>
        /// Run analysis on the supplied model and then transform it to
        /// remove redundant messages.
        /// </summary>
        /// <param name="itd"></param>
        /// <returns></returns>
        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            var analysis = new CopyAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            return base.Transform(itd);
        }

        /// <summary>
        /// Record the containers that a variable is created in, for use by HoistingTransform.
        /// </summary>
        /// <param name="ivde"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            context.InputAttributes.Remove<Containers>(ivde.Variable);
            context.InputAttributes.Set(ivde.Variable, new Containers(context));
            return ivde;
        }


        /// <summary>
        /// Remove assignments to variables which are copies of other variables.
        /// </summary>
        /// <param name="iae"></param>
        /// <returns></returns>
        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression copyExpr = GetCopyExpr(iae.Target);
            if (iae.Target is IVariableDeclarationExpression ivde)
                ConvertExpression(ivde);
            if (iae.Expression.Equals(copyExpr))
                return iae; // was 'null'
            var newTarget = ConvertIndices(iae.Target);
            // convert right hand side
            IExpression expr = ConvertExpression(iae.Expression);
            if (ReferenceEquals(newTarget, iae.Target) &&
                ReferenceEquals(expr, iae.Expression))
                return iae;
            IAssignExpression ae = Builder.AssignExpr();
            ae.Expression = expr;
            ae.Target = newTarget;
            context.InputAttributes.CopyObjectAttributesTo(iae, context.OutputAttributes, ae);
            return ae;
        }

        private IExpression ConvertIndices(IExpression expr)
        {
            if (!(expr is IArrayIndexerExpression iaie)) return expr;
            var newTarget = ConvertIndices(iaie.Target);
            formalTypeStack.Push(typeof(int));
            IList<IExpression> newIndices = ConvertCollection(iaie.Indices);
            formalTypeStack.Pop();
            if (ReferenceEquals(newTarget, iaie.Target) &&
                ReferenceEquals(newIndices, iaie.Indices))
                return iaie;
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            aie.Target = newTarget;
            aie.Indices.AddRange(newIndices);
            Context.InputAttributes.CopyObjectAttributesTo(iaie, context.OutputAttributes, aie);
            return aie;
        }

        /// <summary>
        /// Modify array index expressions which are copies of other variables.
        /// </summary>
        /// <param name="iaie"></param>
        /// <returns></returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            // should not convert if on LHS 
            IExpression copyExpr = GetCopyExpr(iaie);
            if (copyExpr != null)
                return copyExpr;
            formalTypeStack.Push(typeof(int));
            IList<IExpression> newIndices = ConvertCollection(iaie.Indices);
            formalTypeStack.Pop();
            Type arrayType;
            if (iaie.Indices.Count == 1) 
            {
                var elementType = FormalTypeStackApplies() ? formalTypeStack.Peek() : iaie.GetExpressionType();
                // This must be IReadOnlyList not IList to allow covariance.
                arrayType = typeof(IReadOnlyList<>).MakeGenericType(elementType);
            }
            else arrayType = iaie.Target.GetExpressionType();
            formalTypeStack.Push(arrayType);
            IExpression newTarget = ConvertExpression(iaie.Target);
            formalTypeStack.Pop();
            if (ReferenceEquals(newTarget, iaie.Target) &&
                ReferenceEquals(newIndices, iaie.Indices))
                return iaie;
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            aie.Target = newTarget;
            aie.Indices.AddRange(newIndices);
            Context.InputAttributes.CopyObjectAttributesTo(iaie, context.OutputAttributes, aie);
            // Since part of the expression has been replaced, check the whole again
            return ConvertArrayIndexer(aie);
        }

        /// <summary>
        /// Modify variable references that can be replaced by other variables.
        /// </summary>
        /// <param name="ivre"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            // should not convert if on LHS
            IExpression copyExpr = GetCopyExpr(ivre);
            if (copyExpr != null && CanBeReplacedBy(ivre, copyExpr.GetExpressionType()))
            {
                return copyExpr;
            }
            else
            {
                return ivre;
            }
        }

        private bool CanBeReplacedBy(IExpression expr, Type replacementType)
        {
            return (FormalTypeStackApplies() && formalTypeStack.Peek().IsAssignableFrom(replacementType)) ||
                 expr.GetExpressionType().IsAssignableFrom(replacementType);
        }

        private bool FormalTypeStackApplies()
        {
            if (formalTypeStack.Count == 0) return false;
            var previousElement = context.InputStack[context.Depth - 2].inputElement;
            return (previousElement is IMethodInvokeExpression) || (previousElement is IArrayIndexerExpression);
        }

        internal static MessageFcnInfo GetMessageFcnInfo(BasicTransformContext context, IMethodInvokeExpression imie)
        {
            MethodInfo method = (MethodInfo)imie.Method.Method.MethodInfo;
            MessageFcnInfo fcnInfo = context.InputAttributes.Get<MessageFcnInfo>(method);
            if (fcnInfo == null)
            {
                FactorManager.FactorInfo info = context.InputAttributes.Get<FactorManager.FactorInfo>(imie);
                if (info != null)
                {
                    fcnInfo = info.GetMessageFcnInfoFromFactor();
                }
                else
                {
                    ParameterInfo[] parameters = method.GetParameters();
                    fcnInfo = new MessageFcnInfo(method, parameters)
                    {
                        DependencyInfo = FactorManager.GetDependencyInfo(method)
                    };
                }
                //context.InputAttributes.Set(method, fcnInfo);
            }
            return fcnInfo;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(ArrayHelper.CopyStorage)))
                return imie;
            IMethodInvokeExpression mie = Builder.MethodInvkExpr();
            MessageFcnInfo fcnInfo = GetMessageFcnInfo(context, imie);
            var parameters = fcnInfo.Method.GetParameters();
            bool changed = false;
            var arguments = imie.Arguments.Select((arg, i) =>
            {
                if ((fcnInfo != null) && (i == fcnInfo.ResultParameterIndex))
                {
                    // if argument is 'result' argument, do not convert
                    return arg;
                }
                else
                {
                    formalTypeStack.Push(parameters[i].ParameterType);
                    var expr = ConvertExpression(arg);
                    formalTypeStack.Pop();
                    if (!ReferenceEquals(expr, arg))
                        changed = true;
                    return expr;
                }
            });
            mie.Arguments.AddRange(arguments);
            mie.Method = (IMethodReferenceExpression)ConvertExpression(imie.Method);
            if (ReferenceEquals(mie.Method, imie.Method) && !changed)
                return imie;
            context.InputAttributes.CopyObjectAttributesTo(imie, context.OutputAttributes, mie);
            return mie;
        }

        /// <summary>
        /// The expressions currently being converted.  Used to catch transformation cycles.
        /// </summary>
        private readonly Set<IExpression> expressions = new Set<IExpression>();

        /// <summary>
        /// Determines if the value of the supplied expression will always be equal
        /// to another expression, and if so returns the other expression.
        /// Otherwise returns null.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        private IExpression GetCopyExpr(IExpression expr)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd == null)
                return null;
            CopyOfAttribute coa = context.InputAttributes.Get<CopyOfAttribute>(ivd);
            if (coa == null)
                return null;

            if (expressions.Contains(expr))
            {
                Error("Copy cycle detected, starting from " + expr);
                return null;
            }

            IExpression newExpr = coa.Replace(context, expr);
            if (newExpr != null)
            {
                expressions.Add(expr);
                newExpr = DoConvertExpression(newExpr);
                expressions.Remove(expr);
            }
            return newExpr;
        }

        /// <summary>
        /// Analyses known copy expressions and attaches CopyOfAttribute where appropriate.
        /// </summary>
        private class CopyAnalysisTransform : ShallowCopyTransform
        {
            public override string Name
            {
                get { return "CopyAnalysisTransform"; }
            }

            protected override IExpression ConvertAssign(IAssignExpression iae)
            {
                foreach (IStatement stmt in context.FindAncestors<IStatement>())
                {
                    // an initializer statement may perform a copy, but it is not valid to replace the lhs
                    // in that case.
                    if (context.InputAttributes.Has<Initializer>(stmt))
                        return iae;
                }
                // Look for assignments where the right hand side is a SetTo call
                if (iae.Expression is IMethodInvokeExpression imie)
                {
                    bool isCopy = Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Clone.Copy));
                    bool isSetTo = Recognizer.IsStaticGenericMethod(imie, typeof(ArrayHelper), "SetTo");
                    bool isSetAllElementsTo = Recognizer.IsStaticGenericMethod(imie, typeof(ArrayHelper), "SetAllElementsTo");
                    bool isGetItemsPoint = Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsPointOp<>), "ItemsAverageConditional");
                    bool isGetJaggedItemsPoint = Recognizer.IsStaticGenericMethod(imie, typeof(GetJaggedItemsPointOp<>), "ItemsAverageConditional");
                    bool isGetDeepJaggedItemsPoint = Recognizer.IsStaticGenericMethod(imie, typeof(GetDeepJaggedItemsPointOp<>), "ItemsAverageConditional");
                    bool isGetItemsFromJaggedPoint = Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsFromJaggedPointOp<>), "ItemsAverageConditional");
                    bool isGetItemsFromDeepJaggedPoint = Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsFromDeepJaggedPointOp<>), "ItemsAverageConditional");
                    bool isGetJaggedItemsFromJaggedPoint = Recognizer.IsStaticGenericMethod(imie, typeof(GetJaggedItemsFromJaggedPointOp<>), "ItemsAverageConditional");
                    if (isCopy || isSetTo || isSetAllElementsTo || isGetItemsPoint ||
                        isGetJaggedItemsPoint || isGetDeepJaggedItemsPoint || isGetJaggedItemsFromJaggedPoint ||
                        isGetItemsFromJaggedPoint || isGetItemsFromDeepJaggedPoint)
                    {
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                        // Find the condition context
                        var ifs = context.FindAncestors<IConditionStatement>();
                        var condContext = new List<IConditionStatement>();
                        foreach (var ifSt in ifs)
                        {
                            if (!CodeRecognizer.IsStochastic(context, ifSt.Condition))
                                condContext.Add(ifSt);
                        }
                        var copyAttr = context.InputAttributes.GetOrCreate<CopyOfAttribute>(ivd, () => new CopyOfAttribute());
                        IExpression rhs;
                        if (isSetTo || isSetAllElementsTo)
                        {
                            // Mark as copy of the second argument
                            rhs = imie.Arguments[1];
                        }
                        else
                        {
                            // Mark as copy of the first argument
                            rhs = imie.Arguments[0];
                        }
                        InitialiseTo init = context.InputAttributes.Get<InitialiseTo>(ivd);
                        if (init != null)
                        {
                            IVariableDeclaration ivdRhs = Recognizer.GetVariableDeclaration(rhs);
                            InitialiseTo initRhs = (ivdRhs == null) ? null : context.InputAttributes.Get<InitialiseTo>(ivdRhs);
                            if (initRhs == null || !initRhs.initialMessagesExpression.Equals(init.initialMessagesExpression))
                            {
                                // Do not replace a variable with a unique initialiser
                                return iae;
                            }
                        }
                        var initBack = context.InputAttributes.Get<InitialiseBackwardTo>(ivd);
                        if (initBack != null && !(initBack.initialMessagesExpression is IArrayCreateExpression))
                        {
                            IVariableDeclaration ivdRhs = Recognizer.GetVariableDeclaration(rhs);
                            InitialiseBackwardTo initRhs = (ivdRhs == null) ? null : context.InputAttributes.Get<InitialiseBackwardTo>(ivdRhs);
                            if (initRhs == null || !initRhs.initialMessagesExpression.Equals(init.initialMessagesExpression))
                            {
                                // Do not replace a variable with a unique initialiser
                                return iae;
                            }
                        }
                        if (isCopy || isSetTo)
                        {
                            RemoveMatchingSuffixes(iae.Target, rhs, condContext, out IExpression lhsPrefix, out IExpression rhsPrefix);
                            copyAttr.copyMap[lhsPrefix] = new CopyOfAttribute.CopyContext { Expression = rhsPrefix, ConditionContext = condContext };
                        }
                        else if (isSetAllElementsTo)
                        {
                            copyAttr.copiedInEveryElementMap[iae.Target] = new CopyOfAttribute.CopyContext { Expression = rhs, ConditionContext = condContext };
                        }
                        else if (isGetItemsPoint || isGetJaggedItemsPoint || isGetDeepJaggedItemsPoint || isGetItemsFromJaggedPoint || isGetItemsFromDeepJaggedPoint || isGetJaggedItemsFromJaggedPoint)
                        {
                            var target = ((IArrayIndexerExpression)iae.Target).Target;
                            int inputDepth = imie.Arguments.Count - 3;
                            List<IExpression> indexExprs = new List<IExpression>();
                            for (int i = 0; i < inputDepth; i++)
                            {
                                indexExprs.Add(imie.Arguments[1 + i]);
                            }
                            int outputDepth;
                            if (isGetDeepJaggedItemsPoint)
                                outputDepth = 3;
                            else if (isGetJaggedItemsPoint || isGetJaggedItemsFromJaggedPoint)
                                outputDepth = 2;
                            else
                                outputDepth = 1;
                            copyAttr.copyAtIndexMap[target] = new CopyOfAttribute.CopyContext2
                            {
                                Depth = outputDepth,
                                ConditionContext = condContext,
                                ExpressionAtIndex = (lhsIndices) =>
                                {
                                    return Builder.JaggedArrayIndex(rhs, indexExprs.ListSelect(indexExpr =>
                                        new[] { Builder.JaggedArrayIndex(indexExpr, lhsIndices) }));
                                }
                            };
                        }
                        else
                            throw new NotImplementedException();
                    }
                }
                return iae;
            }

            private void RemoveMatchingSuffixes(IExpression lhs, IExpression rhs, List<IConditionStatement> condContext, out IExpression lhsPrefix, out IExpression rhsPrefix)
            {
                if ((lhs is IArrayIndexerExpression iaie1) &&
                    (rhs is IArrayIndexerExpression iaie2) &&
                    (iaie1.Indices.Count == iaie2.Indices.Count))
                {
                    bool allIndicesAreEqual = Enumerable.Range(0, iaie1.Indices.Count).All(i =>
                    {
                        IExpression index1 = iaie1.Indices[i];
                        IExpression index2 = iaie2.Indices[i];
                        // indices only match if they are loop variables, because then we know that this is the only assignment for the whole array.
                        // literal indices, on the other hand, can appear in multiple assignments, e.g. array[0] = Copy(x0[0]), array[1] = Copy(x1[0]).
                        return (index1 is IVariableReferenceExpression) &&
                            (index2 is IVariableReferenceExpression) &&
                            index1.Equals(index2) &&
                            !AnyConditionsDependOnLoopVariable(condContext, Recognizer.GetVariableDeclaration(index1));
                    });
                    if (allIndicesAreEqual)
                    {
                        // By removing suffixes we may substitute a collection for another collection with the same elements but a different collection type.
                        // CopyPropagationTransform must check that this is valid.
                        RemoveMatchingSuffixes(iaie1.Target, iaie2.Target, condContext, out lhsPrefix, out rhsPrefix);
                        return;
                    }
                }
                lhsPrefix = lhs;
                rhsPrefix = rhs;
            }

            private bool AnyConditionsDependOnLoopVariable(List<IConditionStatement> condContext, IVariableDeclaration find)
            {
                IForStatement ifs = Recognizer.GetLoopForVariable(context, find);
                if (ifs == null)
                    return false;
                return condContext.Any(ics => Recognizer.GetVariables(ics.Condition).Any(ivd =>
                {
                    Containers c = context.InputAttributes.Get<Containers>(ivd);
                    if (c == null)
                    {
                        context.Error($"Containers not found for '{ivd.Name}'.");
                        return false;
                    }
                    return c.Contains(ifs);
                }));
            }
        }

        /// <summary>
        /// Attribute used to mark variables which are just copies of other variables.
        /// </summary>
        private class CopyOfAttribute : ICompilerAttribute
        {
            internal Dictionary<IExpression, CopyContext> copyMap = new Dictionary<IExpression, CopyContext>();
            internal Dictionary<IExpression, CopyContext> copiedInEveryElementMap = new Dictionary<IExpression, CopyContext>();
            internal Dictionary<IExpression, CopyContext2> copyAtIndexMap = new Dictionary<IExpression, CopyContext2>();

            public IExpression Replace(BasicTransformContext context, IExpression expr)
            {
                if (copyMap.ContainsKey(expr))
                {
                    var cc = copyMap[expr];
                    if (cc.IsValidContext(context))
                        return cc.Expression;
                }
                if (expr is IArrayIndexerExpression iaie)
                {
                    if (copiedInEveryElementMap.ContainsKey(iaie.Target))
                    {
                        var cc = copiedInEveryElementMap[iaie.Target];
                        if (cc.IsValidContext(context))
                            return cc.Expression;
                    }
                    if (copyAtIndexMap.ContainsKey(iaie.Target))
                    {
                        var cc = copyAtIndexMap[iaie.Target];
                        if (cc.Depth == 1 && cc.IsValidContext(context))
                            return cc.ExpressionAtIndex(new[] { iaie.Indices });
                    }
                    if (iaie.Target is IArrayIndexerExpression iaie2)
                    {
                        if (copyAtIndexMap.ContainsKey(iaie2.Target))
                        {
                            var cc = copyAtIndexMap[iaie2.Target];
                            if (cc.Depth == 2 && cc.IsValidContext(context))
                                return cc.ExpressionAtIndex(new[] { iaie2.Indices, iaie.Indices });
                        }
                        if (iaie2.Target is IArrayIndexerExpression iaie3)
                        {
                            if (copyAtIndexMap.ContainsKey(iaie3.Target))
                            {
                                var cc = copyAtIndexMap[iaie3.Target];
                                if (cc.Depth == 3 && cc.IsValidContext(context))
                                    return cc.ExpressionAtIndex(new[] { iaie3.Indices, iaie2.Indices, iaie.Indices });
                            }
                        }
                    }
                }
                return null;
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                foreach (KeyValuePair<IExpression, CopyContext> kvp in copyMap)
                {
                    sb.Append(kvp.Key + " is copy of " + kvp.Value.Expression);
                }
                foreach (KeyValuePair<IExpression, CopyContext> kvp in copiedInEveryElementMap)
                {
                    sb.Append(kvp.Key + "[*] is copy of " + kvp.Value.Expression);
                }
                return sb.ToString();
            }

            internal class CopyContext
            {
                internal IExpression Expression { get; set; }
                internal IList<IConditionStatement> ConditionContext { get; set; }

                internal bool IsValidContext(BasicTransformContext context)
                {
                    var containers = new Containers(context);
                    foreach (var ics in ConditionContext)
                    {
                        if (!containers.Contains(ics))
                            return false;
                    }
                    return true;
                }
            }

            internal class CopyContext2
            {
                public int Depth;
                internal Func<IList<IList<IExpression>>, IExpression> ExpressionAtIndex { get; set; }
                internal IList<IConditionStatement> ConditionContext { get; set; }

                internal bool IsValidContext(BasicTransformContext context)
                {
                    var containers = new Containers(context);
                    foreach (var ics in ConditionContext)
                    {
                        if (!containers.Contains(ics))
                            return false;
                    }
                    return true;
                }
            }
        }
    }
}