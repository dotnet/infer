// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Sets vi.marginalPrototypeExpression for all variables and parameters.  Must run before GateTransform and IndexingTransform.
    /// </summary>
    internal class MarginalAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "MarginalAnalysisTransform";
            }
        }

        /// <summary>
        /// Sets the MarginalPrototype and DescriptionAttribute of all method parameters
        /// </summary>
        /// <param name="md"></param>
        /// <param name="imd"></param>
        /// <returns></returns>
        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            IMethodDeclaration method = base.DoConvertMethod(md, imd);
            foreach (IParameterDeclaration ipd in method.Parameters)
            {
                if (!context.InputAttributes.Has<DescriptionAttribute>(ipd))
                    context.OutputAttributes.Set(ipd, new DescriptionAttribute("The observed value of '" + ipd.Name + "'"));
                SetMarginalPrototype(ipd);
            }
            return method;
        }

        private void SetMarginalPrototype(object decl)
        {
            if (context.InputAttributes.Has<IsInferred>(decl))
            {
                VariableInformation vi = VariableInformation.GetVariableInformation(context, decl);
                if (vi.marginalPrototypeExpression == null)
                {
                    vi.DefineAllIndexVars(context);
                    MarginalPrototype mpa = Context.InputAttributes.Get<MarginalPrototype>(decl);
                    if (!vi.SetMarginalPrototypeFromAttribute(mpa, throwIfMissing: false) || InvalidMarginalPrototype(vi, decl))
                    {
                        mpa = new MarginalPrototype(null)
                        {
                            prototypeExpression = Builder.NewObject(typeof(PointMass<>).MakeGenericType(vi.varType), vi.GetExpression())
                        };
                        context.OutputAttributes.Set(decl, mpa);
                        vi.SetMarginalPrototypeFromAttribute(mpa);
                    }
                }
            }
        }

        private bool InvalidMarginalPrototype(VariableInformation vi, object decl)
        {
            Type marginalType = vi.marginalPrototypeExpression.GetExpressionType();
            if (marginalType == null) throw new InferCompilerException("Cannot determine type of marginal prototype expression: " + vi.marginalPrototypeExpression);
            Type domainType = Distribution.GetDomainType(marginalType);
            Type messageType = MessageTransform.GetDistributionType(vi.varType, domainType, marginalType, false);
            if (MessageTransform.IsPointMass(messageType))
            {
                context.OutputAttributes.Remove<MarginalPrototype>(decl);
                vi.marginalPrototypeExpression = null;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Sets the MarginalPrototype and DescriptionAttribute of all constant variables
        /// </summary>
        /// <param name="ivde"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration ivd = ivde.Variable;
            if (!CodeRecognizer.IsStochastic(context, ivd))
            {
                if (!context.InputAttributes.Has<DescriptionAttribute>(ivd))
                    context.OutputAttributes.Set(ivd, new DescriptionAttribute("The constant '" + ivd.Name + "'"));
                SetMarginalPrototype(ivd);
            }
            else
            {
                VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                // Ensure the marginal prototype is set.
                MarginalPrototype mpa = Context.InputAttributes.Get<MarginalPrototype>(ivd);
                try
                {
                    vi.SetMarginalPrototypeFromAttribute(mpa);
                }
                catch (ArgumentException ex)
                {
                    Error(ex.Message);
                }
            }
            return ivde;
        }
    }
}
