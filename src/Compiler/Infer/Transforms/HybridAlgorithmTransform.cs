// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class HybridAlgorithmTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "HybridAlgorithmTransform"; }
        }

        private IAlgorithm defaultAlg;
        protected ExpressionEvaluator evaluator = new ExpressionEvaluator();

        public HybridAlgorithmTransform(IAlgorithm defaultAlg)
        {
            this.defaultAlg = defaultAlg;
        }

        // Do shallow copying in most cases
        protected override IExpression DoConvertExpression(IExpression expr)
        {
            if (expr is IArrayIndexerExpression) return base.DoConvertExpression(expr);
            if (expr is IAssignExpression) return base.DoConvertExpression(expr);
            if (expr is IMethodInvokeExpression) return base.DoConvertExpression(expr);
            return expr;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IArrayIndexerExpression aie = (IArrayIndexerExpression) base.ConvertArrayIndexer(iaie);
            IExpression topExpr = context.FindAncestor<IAssignExpression>();
            bool isDef = false;
            if (topExpr == null)
            {
                topExpr = context.FindAncestor<IMethodInvokeExpression>();
                if (Recognizer.IsStaticMethod(topExpr, typeof (InferNet))) return aie;
            }
            else
            {
                isDef = ((IAssignExpression) topExpr).Target == iaie;
            }
            if (topExpr == null) return aie;
            // Find the variable algorithm
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iaie.Target);
            if (ivd == null)
            {
                //Error("Cannot find array indexer target variable: " + iaie.Target);
                return iaie;
            }
            IAlgorithm varAlg = GetAlgorithm(ivd);
            //Console.WriteLine("varAlg=" + varAlg + " " + ivd);
            // Find the factor algorithm
            IAlgorithm factorAlg = GetAlgorithm(topExpr);
            // If the algorithsm are the same, do nothing.
            if (varAlg.Equals(factorAlg)) return aie;

            // Otherwise, look up conversion operator
            ChannelInfo ci = context.InputAttributes.Get<ChannelInfo>(ivd);
            if (ci == null)
            {
                Error("Channel information not found for '" + ivd.Name + "'.");
                return aie;
            }
            Type chType = ci.varInfo.marginalPrototypeExpression.GetExpressionType();
            List<object> extraArgs = new List<object>();
            MethodReference opMethodRef = null;
            try
            {
                opMethodRef = factorAlg.GetAlgorithmConversionOperator(chType, varAlg, isDef, extraArgs);
            }
            catch (Exception ex)
            {
                Error("No conversion operator defined from " + factorAlg + " to " + varAlg, ex);
                return aie;
            }
            if (opMethodRef == null) return aie;
            // Create new channel
            DuplicateInfo dupInfo = context.InputAttributes.GetOrCreate<DuplicateInfo>(ivd, () => new DuplicateInfo());
            dupInfo.dupCount++;
            IVariableDeclaration vd = Builder.VarDecl(ivd.Name + "_dup" + dupInfo.dupCount, ivd.VariableType);
            Context.InputAttributes.CopyObjectAttributesTo(ivd, context.OutputAttributes, vd);
            Context.OutputAttributes.Remove<Algorithm>(vd);
            Context.OutputAttributes.Set(vd, new Algorithm(factorAlg));
            //ci.AddDeclaration(context);
            IExpression opMethodCall = null;
            IExpression[] args = new IExpression[extraArgs.Count + 1];
            IExpression lhs = Builder.ArrayIndex(Builder.VarRefExpr(vd), Builder.LiteralExpr(0));
            args[0] = aie;
            if (isDef)
            {
                args[0] = lhs;
                lhs = aie;
            }
            for (int i = 0; i < extraArgs.Count; i++) args[i + 1] = Builder.LiteralExpr(extraArgs[i]);
            if (opMethodRef.TypeParameterCount == 0)
            {
                opMethodCall = Builder.StaticMethod(opMethodRef.GetMethodInfo(), args);
            }
            else
            {
                // TODO: insert argument types
                opMethodCall = Builder.StaticGenericMethod(opMethodRef.GetMethodInfo(), args);
            }
            //context.AddExpression(Builder.AssignExpr(lhs,opMethodCall));
            return Builder.ArrayIndex(Builder.VarRefExpr(vd), Builder.LiteralExpr(0));
        }

        protected IAlgorithm GetAlgorithm(object obj)
        {
            if (!Context.InputAttributes.Has<Algorithm>(obj)) return defaultAlg;
            return Context.InputAttributes.Get<Algorithm>(obj).algorithm;
        }

        private class DuplicateInfo : ICompilerAttribute
        {
            internal int dupCount = 0;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}