// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Attach size information to arrays (via VariableInformation attributes).
    /// </summary>
    internal class ArrayAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "ArrayAnalysisTransform";
            }
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            // Record the loop context that a variable is created in
            context.InputAttributes.Remove<LoopContext>(ivd);
            context.InputAttributes.Set(ivd, new LoopContext(context));
            context.InputAttributes.Remove<Containers>(ivd);
            context.InputAttributes.Set(ivd, new Containers(context));
            return ivd;
        }

        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            IArrayCreateExpression ace = (IArrayCreateExpression)base.ConvertArrayCreate(iace);
            IAssignExpression iae = context.FindAncestor<IAssignExpression>();
            if (iae == null)
                return ace;
            if (iae.Expression != iace)
                return ace;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            if (ivd == null)
                return ace;
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            int depth = Recognizer.GetIndexingDepth(iae.Target);
            IExpression[] dimExprs = new IExpression[ace.Dimensions.Count];
            for (int i = 0; i < dimExprs.Length; i++)
                dimExprs[i] = ace.Dimensions[i];
            bool targetHasLiteralIndices = false;
            List<IList<IExpression>> indices = Recognizer.GetIndices(iae.Target);
            foreach (IList<IExpression> bracket in indices)
            {
                foreach (IExpression index in bracket)
                {
                    if (index is ILiteralExpression)
                        targetHasLiteralIndices = true;
                }
            }

            if (!targetHasLiteralIndices)
            {
                try
                {
                    // Set the size of this array at the lhs indexing depth
                    if(vi.sizes.Count <= depth)
                        vi.SetSizesAtDepth(depth, dimExprs);
                }
                catch (Exception ex)
                {
                    Error(ex.Message, ex);
                }
            }
            else
            {
                // input is:  a[0][1] = new int[10];
                // output is: a_size[0][1] = 10;
                if (vi.sizes.Count < depth)
                    throw new Exception("missing size information for " + ivd);
                if (vi.sizes.Count == depth)
                    vi.sizes.Add(new IExpression[dimExprs.Length]);
                vi.DefineIndexVarsUpToDepth(context, depth - 1);
                for (int dim = 0; dim < dimExprs.Length; dim++)
                {
                    IExpression sizeExpr = vi.sizes[depth][dim];
                    if (sizeExpr != null)
                        continue;
                    // create a size variable using indexVars and sizes at lower depths
                    List<IExpression[]> sizes2 = new List<IExpression[]>();
                    List<IVariableDeclaration[]> indexVars2 = new List<IVariableDeclaration[]>();
                    for (int i = 0; i < depth; i++)
                    {
                        sizes2.Add(vi.sizes[i]);
                        if (i < depth - 1)
                            indexVars2.Add(vi.indexVars[i]);
                    }
                    Type tp = CodeBuilder.MakeJaggedArrayType(typeof(int), sizes2);
                    string name = VariableInformation.GenerateName(context, ivd.Name + "_size");
                    IVariableDeclaration sizeVar = Builder.VarDecl(name, tp);
                    VariableInformation viSize = VariableInformation.GetVariableInformation(context, sizeVar);
                    for (int i = 0; i < depth; i++)
                    {
                        viSize.SetSizesAtDepth(i, sizes2[i]);
                        if (i < depth - 1)
                            viSize.SetIndexVariablesAtDepth(i, indexVars2[i]);
                    }
                    IList<IStatement> stmts = Builder.StmtCollection();
                    Builder.NewJaggedArray(stmts, sizeVar, indexVars2, sizes2);
                    Containers containers = context.InputAttributes.Get<Containers>(ivd);
                    int ancIndex = containers.GetMatchingAncestorIndex(context);
                    context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
                    sizeExpr = Builder.JaggedArrayIndex(Builder.VarRefExpr(sizeVar), vi.GetIndexExpressions(context, depth));
                    vi.sizes[depth][dim] = sizeExpr;
                    var assignStmt = Builder.AssignStmt(Builder.JaggedArrayIndex(Builder.VarRefExpr(sizeVar), indices), dimExprs[dim]);
                    context.AddStatementAfterCurrent(assignStmt);
                }
            }
            if (iace.Initializer != null)
            {
                // array is being filled in by an initializer, rather than loops.
                // need to fill in the remaining sizes 
                vi.DefineSizesUpToDepth(context, vi.ArrayDepth);
            }
            return ace;
        }
    }
}
