// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class LoopUnrollingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "LoopUnrollingTransform"; }
        }

        private List<ConditionBinding> iterationContext = new List<ConditionBinding>();

        private Dictionary<IVariableDeclaration, UnrolledVar> unrolledVars = new Dictionary<IVariableDeclaration, UnrolledVar>();

        private class UnrolledVar
        {
            public Set<IExpression> loopVars;
            public Dictionary<Set<ConditionBinding>, IVariableDeclaration> clones = new Dictionary<Set<ConditionBinding>, IVariableDeclaration>();
        }

        internal static ConditionBinding GetInitializerBinding(IForStatement ifs)
        {
            IStatement ist = ifs.Initializer;
            if (ist is IBlockStatement)
            {
                if (((IBlockStatement) ist).Statements.Count != 1) throw new Exception("Unhandled loop initializer: " + ist);
                ist = ((IBlockStatement) ist).Statements[0];
            }
            IExpressionStatement init = (IExpressionStatement) ist;
            IAssignExpression iae = (IAssignExpression) init.Expression;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            return new ConditionBinding(Builder.VarRefExpr(ivd), iae.Expression);
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            IExpression sizeExpr = Recognizer.LoopSizeExpression(ifs);
            if (sizeExpr is ILiteralExpression)
            {
                int size = (int) ((ILiteralExpression) sizeExpr).Value;
                if (size < 20)
                {
                    ConditionBinding binding = GetInitializerBinding(ifs);
                    if (binding.rhs is ILiteralExpression)
                    {
                        int start = (int) ((ILiteralExpression) binding.rhs).Value;
                        for (int i = start; i < size; i++)
                        {
                            iterationContext.Add(new ConditionBinding(binding.lhs, Builder.LiteralExpr(i)));
                            IBlockStatement body = ConvertBlock(ifs.Body);
                            context.AddStatementsBeforeCurrent(body.Statements);
                            iterationContext.RemoveAt(iterationContext.Count - 1);
                        }
                        return null;
                    }
                }
            }
            return base.ConvertFor(ifs);
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            if (iterationContext.Count == 0) return ivd;
            if (Recognizer.GetLoopForVariable(context, ivd) != null) return ivd;
            UnrolledVar v;
            if (!unrolledVars.TryGetValue(ivd, out v))
            {
                v = new UnrolledVar();
                unrolledVars[ivd] = v;
            }
            Set<ConditionBinding> bindings = new Set<ConditionBinding>();
            foreach (ConditionBinding binding in iterationContext)
            {
                // must clone since bindings are mutated
                bindings.Add((ConditionBinding) binding.Clone());
            }
            v.loopVars = new Set<IExpression>();
            StringBuilder sb = new StringBuilder(ivd.Name);
            foreach (ConditionBinding binding in bindings)
            {
                v.loopVars.Add(binding.lhs);
                sb.Append("_");
                sb.Append(binding);
            }
            // Note: cannot use VariableInformation to do cloning since this doesn't exist yet.
            IVariableDeclaration clone = Builder.VarDecl(CodeBuilder.MakeValid(sb.ToString()), ivd.VariableType);
            // copy attributes across
            context.InputAttributes.CopyObjectAttributesTo(ivd, context.OutputAttributes, clone);
            v.clones[bindings] = clone;
            return clone;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            foreach (ConditionBinding binding in iterationContext)
            {
                if (binding.lhs.Equals(ivre)) return binding.rhs;
            }
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ivre);
            UnrolledVar v;
            if (unrolledVars.TryGetValue(ivd, out v))
            {
                Set<ConditionBinding> bindings = new Set<ConditionBinding>();
                foreach (ConditionBinding binding in iterationContext)
                {
                    if (v.loopVars.Contains(binding.lhs)) bindings.Add(binding);
                }
                if (!v.clones.ContainsKey(bindings))
                {
                    Error(ivre + " not defined in context " + bindings);
                    return ivre;
                }
                IVariableDeclaration clone = v.clones[bindings];
                return Builder.VarRefExpr(clone);
            }
            return ivre;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            // set index variables on the converted target, using the unconverted lhs indices 
            // this code is copied from ModelAnalysisTransform
            IAssignExpression ae = (IAssignExpression) base.ConvertAssign(iae);
            IParameterDeclaration ipd = null;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ae.Target);
            if (ivd == null)
            {
                ipd = Recognizer.GetParameterDeclaration(ae.Target);
                if (ipd == null) return ae;
            }
            if (iae.Target is IArrayIndexerExpression)
            {
                // Gather index variables from the left-hand side of the assignment
                object decl = (ipd == null) ? (object) ivd : ipd;
                VariableInformation vi = VariableInformation.GetVariableInformation(context, decl);
                try
                {
                    List<IVariableDeclaration[]> indVars = new List<IVariableDeclaration[]>();
                    Recognizer.AddIndexers(context, indVars, iae.Target);
                    // Sets the size of this variable at this array depth
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
                if (!context.InputAttributes.Has<VariableInformation>(decl)) context.InputAttributes.Set(decl, vi);
            }
            return ae;
        }

        public MethodBase MethodToTransform;

        public LoopUnrollingTransform(MethodBase MethodToTransform)
        {
            this.MethodToTransform = MethodToTransform;
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            if (MethodToTransform != null)
            {
                MethodBase mb = Builder.ToMethod(imd);
                if (mb != MethodToTransform) return null;
            }
            return base.ConvertMethod(imd);
        }
    }
}