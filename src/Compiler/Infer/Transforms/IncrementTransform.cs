// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Add increment statements to the code whenever we see certain patterns.
    /// </summary>
    internal class IncrementTransform : ShallowCopyTransform
    {
        private readonly IncrementAnalysisTransform analysis;
        protected Dictionary<object, List<Containers>> containersOfUpdate = new Dictionary<object, List<Containers>>();
        bool isOperatorStatement;

        public override string Name
        {
            get { return "IncrementTransform"; }
        }

        public IncrementTransform(ModelCompiler compiler)
        {
            analysis = new IncrementAnalysisTransform(compiler);
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            var itdOut = base.Transform(itd);
            return itdOut;
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            bool wasOperatorStatement = this.isOperatorStatement;
            if (context.InputAttributes.Has<OperatorStatement>(ist))
            {
                this.isOperatorStatement = true;
            }
            var result = base.DoConvertStatement(ist);
            this.isOperatorStatement = wasOperatorStatement;
            return result;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            object targetDecl = Recognizer.GetArrayDeclaration(iae.Target);
            IStatement increment;
            if (this.isOperatorStatement && targetDecl is IVariableDeclaration)
            {
                IVariableDeclaration ivd = (IVariableDeclaration)targetDecl;
                if (analysis.onUpdate.TryGetValue(targetDecl, out increment))
                {
                    IExpression incrExpr = ((IExpressionStatement)increment).Expression;
                    Containers containers = new Containers(context);
                    containers.AddContainersNeededForExpression(context, incrExpr);
                    containers = Containers.RemoveUnusedLoops(containers, context, incrExpr);
                    containers = Containers.RemoveStochasticConditionals(containers, context);
                    List<Containers> list;
                    if (!containersOfUpdate.TryGetValue(targetDecl, out list))
                    {
                        list = new List<Containers>();
                        containersOfUpdate[targetDecl] = list;
                    }
                    // have we already performed this update in these containers?
                    bool alreadyDone = false;
                    foreach (Containers prevContainers in list)
                    {
                        if (containers.Contains(prevContainers))
                        {
                            // prevContainers is more general, i.e. has fewer containers than 'containers'
                            alreadyDone = true;
                            break;
                        }
                    }
                    if (!alreadyDone)
                    {
                        list.Add(containers);
                        // must set this attribute before the statement is wrapped
                        context.OutputAttributes.Set(increment, new OperatorStatement());
                        int ancIndex = containers.GetMatchingAncestorIndex(context);
                        Containers missing = containers.GetContainersNotInContext(context, ancIndex);
                        increment = Containers.WrapWithContainers(increment, missing.outputs);
                        context.AddStatementAfterAncestorIndex(ancIndex, increment);
                    }
                }
                if (analysis.suppressUpdate.ContainsKey(ivd))
                {
                    foreach (IStatement ist in context.FindAncestors<IStatement>())
                    {
                        if (context.InputAttributes.Has<OperatorStatement>(ist))
                        {
                            var attr = analysis.suppressUpdate[ivd];
                            context.OutputAttributes.Set(ist, new HasIncrement(attr));
                            break;
                        }
                    }
                }
            }
            return base.ConvertAssign(iae);
        }
    }

    internal class IncrementAnalysisTransform : ShallowCopyTransform
    {
        private readonly ModelCompiler compiler;
        /// <summary>
        /// Maps from variable/field declaration to increment statement, or null if loop ended before variable was updated.
        /// </summary>
        public Dictionary<object, IStatement> onUpdate = new Dictionary<object, IStatement>();
        public Dictionary<IVariableDeclaration, IncrementStatement> suppressUpdate = new Dictionary<IVariableDeclaration, IncrementStatement>();

        public IncrementAnalysisTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        public override string Name
        {
            get { return "IncrementAnalysisTransform"; }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            // Insert an Increment statement to support Sequential loops
            // if we find the pattern:
            // forwardExpr = ReplicateOp_Divide.UsesAverageConditional(backwardExpr[index], marginalExpr, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index]);
            if (Recognizer.IsStaticMethod(imie, typeof(ReplicateOp_Divide), "UsesAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[1];
                IExpression indexExpr = imie.Arguments[2];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[3];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.DotNetType.GetMethod("MarginalIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression rhs = Builder.StaticGenericMethod(method, marginalExpr, forwardExpr, backwardExpr);
                    IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            // if we find the pattern:
            // forwardExpr = Replicate2BufferOp.UsesAverageConditional(backwardExpr, *, marginalExpr, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // marginalExpr = MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index]);
            if (Recognizer.IsStaticMethod(imie, typeof(Replicate2BufferOp), "UsesAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indexExpr = imie.Arguments[3];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[4];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.DotNetType.GetMethod("MarginalIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression rhs = Builder.StaticGenericMethod(method, marginalExpr, forwardExpr, Builder.ArrayIndex(backwardExpr, indexExpr));
                    IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            // if we find the pattern:
            // forwardExpr = JaggedSubarrayOp<>.ItemsAverageConditional(backwardExpr[index], *, marginalExpr, indices, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index], indices, index)
            if (Recognizer.IsStaticGenericMethod(imie, typeof(JaggedSubarrayOp<>), "ItemsAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indicesExpr = imie.Arguments[3];
                IExpression indexExpr = imie.Arguments[4];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[5];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    IExpression[] args2 = new IExpression[] { marginalExpr, forwardExpr, backwardExpr, indicesExpr, indexExpr };
                    Type[] argTypes = Array.ConvertAll(args2, e => e.GetExpressionType());
                    Exception exception;
                    MethodInfo method = (MethodInfo)Reflection.Invoker.GetBestMethod(itr.DotNetType, "MarginalIncrement",
                        BindingFlags.Static | BindingFlags.Public | BindingFlags.FlattenHierarchy, null, argTypes, out exception);
                    if (method == null)
                        Error("Cannot find a compatible MarginalIncrement method for JaggedSubarrayOp", exception);
                    else
                    {
                        IExpression rhs = Builder.StaticGenericMethod(method, args2);
                        IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                        var seq = context.InputAttributes.Get<Sequential>(indexVar);
                        var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                        context.OutputAttributes.Set(increment, attr);
                        onUpdate[backwardDecl] = increment;
                    }
                }
            }
            // if we find the pattern:
            // forwardExpr = JaggedSubarrayWithMarginalOp<>.ItemsAverageConditional(backwardExpr[index], *, marginalExpr, indices, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrementItems(backwardExpr[index], forwardExpr,  indices, index, marginalExpr)
            if (Recognizer.IsStaticGenericMethod(imie, typeof(JaggedSubarrayWithMarginalOp<>), "ItemsAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indicesExpr = imie.Arguments[3];
                IExpression indexExpr = imie.Arguments[4];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[5];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    IExpression[] args2 = new IExpression[] { backwardExpr, forwardExpr, indicesExpr, indexExpr, marginalExpr };
                    Type[] argTypes = Array.ConvertAll(args2, e => e.GetExpressionType());
                    Exception exception;
                    MethodInfo method = (MethodInfo)Reflection.Invoker.GetBestMethod(itr.DotNetType, "MarginalIncrementItems",
                        BindingFlags.Static | BindingFlags.Public | BindingFlags.FlattenHierarchy, null, argTypes, out exception);
                    if (method == null)
                        Error("Cannot find a compatible MarginalIncrementItems method for JaggedSubarrayWithMarginalOp", exception);
                    else
                    {
                        IExpression rhs = Builder.StaticGenericMethod(method, args2);
                        IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                        var seq = context.InputAttributes.Get<Sequential>(indexVar);
                        var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                        context.OutputAttributes.Set(increment, attr);
                        object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                        onUpdate[backwardDecl] = increment;
                        IVariableDeclaration marginalVar = Recognizer.GetVariableDeclaration(marginalExpr);
                        suppressUpdate[marginalVar] = attr;
                    }
                    var indicesVar = Recognizer.GetDeclaration(indicesExpr);
                    if (indicesVar != null)
                    {
                        DistributedCommunicationExpression dce = context.GetAttribute<DistributedCommunicationExpression>(indicesVar);
                        if (dce != null)
                        {
                            context.OutputAttributes.Set(imie, dce);
                        }
                    }
                }
            }
            // if we find the pattern:
            // backwardExpr = JaggedSubarrayWithMarginalOp<>.ArrayAverageConditional(forwardExpr, marginalExpr, backwardExpr)
            // then when forwardExpr is updated, we insert the following statement:
            // MarginalIncrementArray
            if (Recognizer.IsStaticGenericMethod(imie, typeof(JaggedSubarrayWithMarginalOp<>), "ArrayAverageConditional"))
            {
                IExpression forwardExpr = imie.Arguments[0];
                IExpression marginalExpr = imie.Arguments[1];
                IExpression backwardExpr = imie.Arguments[2];
                if (true)
                {
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    IExpression[] args2 = new IExpression[] { forwardExpr, backwardExpr, marginalExpr };
                    Type[] argTypes = Array.ConvertAll(args2, e => e.GetExpressionType());
                    Exception exception;
                    MethodInfo method = (MethodInfo)Reflection.Invoker.GetBestMethod(itr.DotNetType, "MarginalIncrementArray",
                        BindingFlags.Static | BindingFlags.Public | BindingFlags.FlattenHierarchy, null, argTypes, out exception);
                    if (method == null)
                        Error("Cannot find a compatible MarginalIncrementArray method for JaggedSubarrayWithMarginalOp", exception);
                    else
                    {
                        IExpression rhs = Builder.StaticGenericMethod(method, args2);
                        IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                        var attr = new IncrementStatement(null, false);
                        context.OutputAttributes.Set(increment, attr);
                        object forwardDecl = Recognizer.GetArrayDeclaration(forwardExpr);
                        onUpdate[forwardDecl] = increment;
                    }
                    IVariableDeclaration marginalVar = Recognizer.GetVariableDeclaration(marginalExpr);
                    // only suppress when we also have MarginalIncrementItems
                    //suppressUpdate.Add(marginalVar);
                }
            }
            // if we find the pattern:
            // forwardExpr = GetItemsOp<>.ItemsAverageConditional(backwardExpr[index], *, marginalExpr, indices, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index], indices, index)
            if (Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsOp<>), "ItemsAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetJaggedItemsOp<>), "ItemsAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetDeepJaggedItemsOp<>), "ItemsAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indicesExpr = imie.Arguments[3];
                IExpression indexExpr = imie.Arguments[4];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[5];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.GenericType.DotNetType.GetMethod("MarginalIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression rhs = Builder.StaticGenericMethod(method, marginalExpr, forwardExpr, backwardExpr, indicesExpr, indexExpr);
                    //IStatement increment = Builder.ExprStatement(rhs);
                    IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            // if we find the pattern:
            // forwardExpr = GetItemsOp<>.ItemsAverageConditional(backwardExpr[index], *, marginalExpr, indices, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index], indices, index)
            if (Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsWithDictionaryOp<>), "ItemsAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indicesExpr = imie.Arguments[3];
                IExpression dictExpr = imie.Arguments[4];
                IExpression indexExpr = imie.Arguments[5];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[6];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.GenericType.DotNetType.GetMethod("MarginalIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression rhs = Builder.StaticGenericMethod(method, marginalExpr, forwardExpr, backwardExpr, indicesExpr, dictExpr, indexExpr);
                    IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            // if we find the pattern:
            // partialExpr = GetItemsOp2<>.Partial(backwardExpr[index], to_array, indices, index, partialExpr);
            // then when backwardExpr is updated, we insert the following statement:
            // to_array[indices[index]] = ArrayIncrement(partialExpr, backwardExpr[index])
            if (Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsOp2<>), "Partial"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression to_array = imie.Arguments[1];
                IExpression indicesExpr = imie.Arguments[2];
                IExpression indexExpr = imie.Arguments[3];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[4];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.GenericType.DotNetType.GetMethod("ArrayIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression resultExpr = Builder.ArrayIndex(to_array, Builder.ArrayIndex(indicesExpr, indexExpr));
                    IExpression rhs = Builder.StaticGenericMethod(method, forwardExpr, backwardExpr, resultExpr);
                    IStatement increment = Builder.AssignStmt(resultExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            // if we find the pattern:
            // forwardExpr = GetItemsJaggedOp<>.ItemsAverageConditional(backwardExpr[index], *, marginalExpr, indices, indices2, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index], indices, indices2, index)
            if (Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsFromJaggedOp<>), "ItemsAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indicesExpr = imie.Arguments[3];
                IExpression indices2Expr = imie.Arguments[4];
                IExpression indexExpr = imie.Arguments[5];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[6];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.GenericType.DotNetType.GetMethod("MarginalIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression rhs = Builder.StaticGenericMethod(method, marginalExpr, forwardExpr, backwardExpr, indicesExpr, indices2Expr, indexExpr);
                    IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            // if we find the pattern:
            // forwardExpr = GetItemsDeepJaggedOp<>.ItemsAverageConditional(backwardExpr[index], *, marginalExpr, indices, indices2, indices3, index, forwardExpr)
            // then when backwardExpr is updated, we insert the following statement:
            // MarginalIncrement(marginalExpr, forwardExpr, backwardExpr[index], indices, indices2, indices3, index)
            if (Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsFromDeepJaggedOp<>), "ItemsAverageConditional"))
            {
                IExpression backwardExpr = imie.Arguments[0];
                object backwardDecl = Recognizer.GetArrayDeclaration(backwardExpr);
                IExpression marginalExpr = imie.Arguments[2];
                IExpression indicesExpr = imie.Arguments[3];
                IExpression indices2Expr = imie.Arguments[4];
                IExpression indices3Expr = imie.Arguments[5];
                IExpression indexExpr = imie.Arguments[6];
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexExpr);
                IExpression forwardExpr = imie.Arguments[7];
                if (indexVar != null && context.InputAttributes.Has<Sequential>(indexVar))
                {
                    if (!compiler.UseSerialSchedules)
                        context.Warning(indexVar.Name + " is marked Sequential but engine.Compiler.UseSerialSchedules = false");
                    IMethodReferenceExpression imre = imie.Method;
                    ITypeReference itr = ((ITypeReferenceExpression)imre.Target).Type;
                    MethodInfo method = itr.GenericType.DotNetType.GetMethod("MarginalIncrement");
                    method = method.MakeGenericMethod(imre.Method.MethodInfo.GetGenericArguments());
                    IExpression rhs = Builder.StaticGenericMethod(method, marginalExpr, forwardExpr, backwardExpr, indicesExpr, indices2Expr, indices3Expr, indexExpr);
                    IStatement increment = Builder.AssignStmt(marginalExpr, rhs);
                    var seq = context.InputAttributes.Get<Sequential>(indexVar);
                    var attr = new IncrementStatement(indexVar, seq.BackwardPass);
                    context.OutputAttributes.Set(increment, attr);
                    onUpdate[backwardDecl] = increment;
                }
            }
            return base.ConvertMethodInvoke(imie);
        }
    }

    /// <summary>
    /// Attached to a statement to indicate that it is an increment to a message.
    /// </summary>
    internal class IncrementStatement : ICompilerAttribute
    {
        internal IVariableDeclaration loopVar;
        internal bool Bidirectional;

        internal IncrementStatement(IVariableDeclaration loopVar, bool bidirectional)
        {
            this.loopVar = loopVar;
            this.Bidirectional = bidirectional;
        }

        public override string ToString()
        {
            string suffix = Bidirectional ? ", Bidirectional" : "";
            return $"IncrementStatement({loopVar?.Name}{suffix})";
        }
    }

    internal class HasIncrement : ICompilerAttribute
    {
        internal IncrementStatement incrementStatement;

        internal HasIncrement(IncrementStatement incrementStatement)
        {
            this.incrementStatement = incrementStatement;
        }
    }
}
