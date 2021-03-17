// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Attach ForwardPointMass attributes to variables.  Must be done after GateTransform and before MessageTransform.
    /// </summary>
    internal class PointMassAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "PointMassAnalysisTransform";
            }
        }

        protected HashSet<IVariableDeclaration> variablesDefinedNonPointMass = new HashSet<IVariableDeclaration>();
        protected Dictionary<IVariableDeclaration, List<IMethodInvokeExpression>> variablesDefinedPointMass = new Dictionary<IVariableDeclaration, List<IMethodInvokeExpression>>();

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            base.DoConvertMethodBody(outputs, inputs);
            if (context.Results.IsErrors())
                return;
            PostProcess();
        }

        /// <summary>
        /// Attach a ForwardPointMass attribute to every variable whose definitions are all point masses.
        /// </summary>
        protected void PostProcess()
        {
            foreach (var entry in variablesDefinedPointMass)
            {
                var targetVar = entry.Key;
                if (variablesDefinedNonPointMass.Contains(targetVar))
                    continue;
                foreach (var imie in entry.Value)
                {
                    context.InputAttributes.CopyObjectAttributesTo<ForwardPointMass>(targetVar, context.OutputAttributes, imie);
                }
            }
        }

        protected void ProcessDefinition(IExpression expr, IVariableDeclaration targetVar, bool isLhs)
        {
            bool targetIsPointMass = false;
            IMethodInvokeExpression imie = expr as IMethodInvokeExpression;
            if (imie != null)
            {
                // TODO: consider using a method attribute for this
                if (Recognizer.IsStaticGenericMethod(imie, new Models.FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.VariablePoint))
                    )
                {
                    targetIsPointMass = true;
                }
                else
                {
                    FactorManager.FactorInfo info = CodeRecognizer.GetFactorInfo(context, imie);
                    targetIsPointMass = info.IsDeterministicFactor && (
                        (info.ReturnedInAllElementsParameterIndex != -1 && ArgumentIsPointMass(imie.Arguments[info.ReturnedInAllElementsParameterIndex])) ||
                        imie.Arguments.All(ArgumentIsPointMass)
                        );
                }
                if (targetIsPointMass)
                {
                    // do this immediately so all uses are updated
                    if (!context.InputAttributes.Has<ForwardPointMass>(targetVar))
                        context.OutputAttributes.Set(targetVar, new ForwardPointMass());
                    // the rest is done later
                    List<IMethodInvokeExpression> list;
                    if (!variablesDefinedPointMass.TryGetValue(targetVar, out list))
                    {
                        list = new List<IMethodInvokeExpression>();
                        variablesDefinedPointMass.Add(targetVar, list);
                    }
                    // this code needs to be synchronized with MessageTransform.ConvertMethodInvoke
                    if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, int, PlaceHolder[]>(Clone.Replicate)) ||
                        Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItems)) ||
                        Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<IReadOnlyList<int>>, PlaceHolder[][]>(Collection.GetJaggedItems)) ||
                        Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<IReadOnlyList<IReadOnlyList<int>>>, PlaceHolder[][][]>(Collection.GetDeepJaggedItems)) ||
                        Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<IReadOnlyList<PlaceHolder>>, IReadOnlyList<int>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItemsFromJagged)) ||
                        Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<IReadOnlyList<IReadOnlyList<PlaceHolder>>>, IReadOnlyList<int>, IReadOnlyList<int>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItemsFromDeepJagged)) ||
                        Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<IReadOnlyList<PlaceHolder>>, IReadOnlyList<IReadOnlyList<int>>, IReadOnlyList<IReadOnlyList<int>>, PlaceHolder[][]>(Collection.GetJaggedItemsFromJagged))
                        )
                    {
                        list.Add(imie);
                    }
                }
            }
            if (!targetIsPointMass && !(expr is IArrayCreateExpression))
            {
                variablesDefinedNonPointMass.Add(targetVar);
                if (variablesDefinedPointMass.ContainsKey(targetVar))
                {
                    variablesDefinedPointMass.Remove(targetVar);
                    context.OutputAttributes.Remove<ForwardPointMass>(targetVar);
                }
            }
        }

        private bool ArgumentIsPointMass(IExpression arg)
        {
            bool IsOut = (arg is IAddressOutExpression);
            if (CodeRecognizer.IsStochastic(context, arg) && !IsOut)
            {
                IVariableDeclaration argVar = Recognizer.GetVariableDeclaration(arg);
                return (argVar != null) && context.InputAttributes.Has<ForwardPointMass>(argVar);
            }
            else
            {
                return true;
            }
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            // if all args to a deterministic method are non-stoch or ForwardPointMass, the output is ForwardPointMass
            IVariableDeclaration targetVar = Recognizer.GetVariableDeclaration(iae.Target);
            if (targetVar == null || variablesDefinedNonPointMass.Contains(targetVar))
                return base.ConvertAssign(iae);
            ProcessDefinition(iae.Expression, targetVar, isLhs: true);
            return base.ConvertAssign(iae);
        }

        protected override IExpression ConvertAddressOut(IAddressOutExpression iaoe)
        {
            IVariableDeclaration targetVar = Recognizer.GetVariableDeclaration(iaoe.Expression);
            if (targetVar == null || variablesDefinedNonPointMass.Contains(targetVar))
                return base.ConvertAddressOut(iaoe);
            IMethodInvokeExpression imie = context.FindAncestor<IMethodInvokeExpression>();
            ProcessDefinition(imie, targetVar, isLhs: false);
            return base.ConvertAddressOut(iaoe);
        }
    }
}
