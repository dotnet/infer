// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
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
    /// Cuts 'for' loops, so that each statement in the loop ends up in a loop by itself.
    /// Also converts declarations inside loops into top-level declarations.
    /// </summary>
    internal class LoopCuttingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "LoopCuttingTransform"; }
        }

        private bool convertingBrokenLoop;
        private readonly Dictionary<IVariableDeclaration, LoopVarInfo> loopVarInfos = new Dictionary<IVariableDeclaration, LoopVarInfo>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
        private readonly bool hoistAttributes;

        internal LoopCuttingTransform(bool hoistAttributes)
        {
            this.hoistAttributes = hoistAttributes;
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd)) return imd;
            return base.DoConvertMethod(md, imd);
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            bool wasBroken = convertingBrokenLoop;
            bool isBroken = ifs is IBrokenForStatement;
            if (isBroken) convertingBrokenLoop = true;
            ifs = (IForStatement) base.ConvertFor(ifs);
            convertingBrokenLoop = wasBroken;
            if (ifs == null) return ifs;
            if (isBroken)
            {
                if (hoistAttributes)
                {
                    IStatement st = ifs.Body.Statements[0];
                    context.InputAttributes.CopyObjectAttributesTo(st, context.OutputAttributes, ifs);
                }
                return ifs;
            }
            // We are not in an operator block, we need to cut the loop across each statement
            foreach (IStatement st in ifs.Body.Statements)
            {
                if (st is ICommentStatement)
                {
                    // we do not need to keep loops around comments
                    context.AddStatementBeforeCurrent(st);
                    continue;
                }
                // must not copy BrokenLoop from st to fs
                IForStatement fs = Builder.ForStmt();
                fs.Condition = ifs.Condition;
                fs.Increment = ifs.Increment;
                fs.Initializer = ifs.Initializer;
                fs.Body = Builder.BlockStmt();
                fs.Body.Statements.Add(st);
                context.AddStatementBeforeCurrent(fs);
                if(hoistAttributes)
                    context.InputAttributes.CopyObjectAttributesTo(st, context.OutputAttributes, fs);
                context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
            }
            return null;
        }

        // for debugging
        public void WriteAttributes(object ist)
        {
            List<ICompilerAttribute> attrs = context.InputAttributes.GetAll<ICompilerAttribute>(ist);
            if (attrs.Count > 0)
            {
                Console.WriteLine(ist);
                foreach (object attr in attrs)
                {
                    Console.WriteLine("  " + attr);
                }
            }
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            ics = (IConditionStatement) base.ConvertCondition(ics);
            if (ics == null) return ics;
            if ((ics.Else != null) && (ics.Else.Statements.Count > 0))
            {
                Error("Unexpected else clause");
                return ics;
            }
            if (convertingBrokenLoop)
            {
                if (hoistAttributes)
                {
                    IStatement st = ics.Then.Statements[0];
                    context.InputAttributes.CopyObjectAttributesTo(st, context.OutputAttributes, ics);
                }
                return ics;
            }
            // We are not in an operator block, we need to cut the if across each statement
            foreach (IStatement st in ics.Then.Statements)
            {
                if (st is ICommentStatement)
                {
                    // we do not need to keep ifs around comments
                    context.AddStatementBeforeCurrent(st);
                    continue;
                }
                IConditionStatement cs = Builder.CondStmt();
                cs.Condition = ConvertExpression(ics.Condition);
                cs.Then = Builder.BlockStmt();
                cs.Then.Statements.Add(st);
                context.AddStatementBeforeCurrent(cs);
                if (hoistAttributes)
                    context.InputAttributes.CopyObjectAttributesTo(st, context.OutputAttributes, cs);
                context.InputAttributes.CopyObjectAttributesTo(ics, context.OutputAttributes, cs);
            }
            return null;
        }

        /// <summary>
        /// Converts variable declarations inside loops into array declarations at the top level.
        /// </summary>
        /// <param name="ivde"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration ivd = ivde.Variable;
            bool isLoneDeclaration = (context.FindAncestorIndex<IExpressionStatement>() == context.Depth - 2);
            List<IForStatement> loops = context.FindAncestors<IForStatement>();
            if (loops.Count == 0)
            {
                List<IConditionStatement> ifs = context.FindAncestors<IConditionStatement>();
                if (ifs.Count == 0)
                    return ivde;
                // Add declaration outside the if
                IStatement outermostContainer = ifs[0];
                int ancIndex = context.GetAncestorIndex(outermostContainer);
                var defaultExpr = Builder.DefaultExpr(ivd.VariableType);
                var assignSt = Builder.AssignStmt(ivde, defaultExpr);
                context.OutputAttributes.Set(assignSt, new Initializer());
                context.AddStatementBeforeAncestorIndex(ancIndex, assignSt);
                if (isLoneDeclaration)
                {
                    return null;
                }
                else
                {
                    return Builder.VarRefExpr(ivd);
                }
            }
            // ignore declaration of a loop variable
            if (Recognizer.LoopVariable(loops[loops.Count - 1]) == ivd) return ivde;

            // Declaration is inside one or more loops, find their sizes and index variables
            Type type = Builder.ToType(ivd.VariableType);
            Type arrayType = type;
            LoopVarInfo lvi = new LoopVarInfo(loops);
            for (int i = 0; i < loops.Count; i++)
            {
                IForStatement loop = loops[i];
                lvi.indexVarRefs[i] = Builder.VarRefExpr(Recognizer.LoopVariable(loop));
                arrayType = Util.MakeArrayType(arrayType, 1);
            }
            Predicate<int> isPartitionedAtDepth = (depth => context.InputAttributes.Has<Partitioned>(Recognizer.GetVariableDeclaration(lvi.indexVarRefs[depth])));
            Type messageType = Distributions.Distribution.IsDistributionType(type)
                                   ? MessageTransform.GetDistributionType(arrayType, type, type, 0, loops.Count, isPartitionedAtDepth)
                                   : MessageTransform.GetArrayType(arrayType, type, 0, isPartitionedAtDepth);
            lvi.arrayvd = Builder.VarDecl(ivd.Name, messageType);
            loopVarInfos[ivd] = lvi;
            MessageArrayInformation mai = context.InputAttributes.Get<MessageArrayInformation>(ivd);
            if (mai != null) mai.loopVarInfo = lvi;
            context.InputAttributes.CopyObjectAttributesTo(ivd, context.OutputAttributes, lvi.arrayvd);
            context.InputAttributes.Remove<VariableInformation>(lvi.arrayvd);

            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            VariableInformation vi2 = VariableInformation.GetVariableInformation(context, lvi.arrayvd);
            vi2.IsStochastic = vi.IsStochastic;

            // Initialise the array to the appropriate sizes
            // TODO: change to work over loop brackets not loops
            IExpression expr = Builder.VarDeclExpr(lvi.arrayvd);
            for (int i = 0; i < loops.Count; i++)
            {
                IForStatement loop = loops[i];
                IExpression loopSize = Recognizer.LoopSizeExpression(loop);
                IExpressionStatement assignSt = Builder.AssignStmt(expr,
                                                                   MessageTransform.GetArrayCreateExpression(expr, expr.GetExpressionType(),
                                                                                                             new IExpression[] { loopSize }));
                context.OutputAttributes.Set(assignSt.Expression, new DescriptionAttribute("Create array for replicates of '" + ivd.Name + "'"));
                context.OutputAttributes.Set(assignSt, new Initializer());

                if (expr is IVariableDeclarationExpression) expr = Builder.VarRefExpr(lvi.arrayvd);
                expr = Builder.ArrayIndex(expr, lvi.indexVarRefs[i]);
                vi2.indexVars.Add(new IVariableDeclaration[] { Recognizer.GetVariableDeclaration(lvi.indexVarRefs[i]) });
                vi2.sizes.Add(new IExpression[] { loopSize });

                // Add declaration outside the loop
                int ancIndex = context.GetAncestorIndex(loop);
                context.AddStatementBeforeAncestorIndex(ancIndex, assignSt);
            }
            vi2.indexVars.AddRange(vi.indexVars);
            vi2.sizes.AddRange(vi.sizes);
            // If the variable declaration was a statement by itself, then return null (i.e. delete the statement).
            if (isLoneDeclaration) return null;
                // Return a reference to the newly created array
            else return expr;
        }

        /// <summary>
        /// Converts references to variables which were declared inside loops, by adding indexing as appropriate.
        /// </summary>
        /// <param name="ivre"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ivre);
            LoopVarInfo lvi;
            if (!loopVarInfos.TryGetValue(ivd, out lvi)) return ivre;
            IExpression expr = Builder.VarRefExpr(lvi.arrayvd);
            for (int i = 0; i < lvi.indexVarRefs.Length; i++) expr = Builder.ArrayIndex(expr, lvi.indexVarRefs[i]);
            context.InputAttributes.CopyObjectAttributesTo(ivre, context.OutputAttributes, expr);
            return expr;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            iae = (IAssignExpression) base.ConvertAssign(iae);
            if (iae.Target is IArrayIndexerExpression target && iae.Expression is IObjectCreateExpression ioce)
            {
                IExpression parent = target.Target;
                Type type = Builder.ToType(ioce.Type);
                if (MessageTransform.IsFileArray(type) && MessageTransform.IsFileArray(parent.GetExpressionType()) && ioce.Arguments.Count == 2)
                {
                    // change new FileArray(name, dimensions) into FileArray(parent, index, dimensions)
                    IList<IExpression> args = Builder.ExprCollection();
                    args.Add(target.Target);
                    args.AddRange(target.Indices);
                    args.Add(ioce.Arguments[1]);
                    return Builder.AssignExpr(target, Builder.NewObject(type, args));
                }
            }
            return iae;
        }
    }

    internal class LoopVarInfo
    {
        internal IVariableReferenceExpression[] indexVarRefs;
        internal IVariableDeclaration arrayvd;

        internal LoopVarInfo(List<IForStatement> loops)
        {
            indexVarRefs = new IVariableReferenceExpression[loops.Count];
        }

        public override string ToString()
        {
            return "LoopVarInfo(" + StringUtil.CollectionToString(indexVarRefs, ",") + ")";
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}