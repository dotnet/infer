// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
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
    ///   Removes calls to InferNet.PreserveWhenCompiled.
    ///   Removes casts.
    ///   Attach index variable information to arrays (via VariableInformation attributes).
    ///   Attach Constraint attributes to assignments that are actually constraints.
    ///   Attach DerivedVariable attributes to children of deterministic factors. (should probably be a separate transform)
    /// </summary>
    internal class ModelAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ModelAnalysisTransform"; }
        }

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
            if (CodeRecognizer.IsInfer(imie))
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
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}