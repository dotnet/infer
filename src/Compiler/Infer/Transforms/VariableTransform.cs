// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Inserts a variable factor at every place that a variable is assigned to, cloning the variable into a definition, use, and marginal.
    /// </summary>
    internal class VariableTransform : ShallowCopyTransform
    {
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

        public override string Name
        {
            get { return "VariableTransform"; }
        }

        /// <summary>
        /// All variables that have been assigned to so far
        /// </summary>
        private Set<IVariableDeclaration> variablesAssigned = new Set<IVariableDeclaration>(new IdentityComparer<IVariableDeclaration>());

        private Set<IVariableDeclaration> variablesLackingVariableFactor = new Set<IVariableDeclaration>(new IdentityComparer<IVariableDeclaration>());

        private Set<IExpression> targetsOfCurrentAssignment;

        private Dictionary<IVariableDeclaration, IVariableDeclaration> useOfVariable =
            new Dictionary<IVariableDeclaration, IVariableDeclaration>(new IdentityComparer<IVariableDeclaration>());

        private Dictionary<IVariableDeclaration, IVariableDeclaration> marginalOfVariable =
            new Dictionary<IVariableDeclaration, IVariableDeclaration>(new IdentityComparer<IVariableDeclaration>());

        private IAlgorithm algorithmDefault;
        private VariableAnalysisTransform analysis;

        public VariableTransform(IAlgorithm algorithm)
        {
            this.algorithmDefault = algorithm;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            bool useJaggedSubarrayWithMarginal = (this.algorithmDefault is ExpectationPropagation);
            analysis = new VariableAnalysisTransform(useJaggedSubarrayWithMarginal);
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            var itdOut = base.Transform(itd);
            if (context.trackTransform)
            {
                IBlockStatement block = Builder.BlockStmt();
                block.Statements.Add(Builder.CommentStmt("variablesOmittingVariableFactor:"));
                foreach (var ivd in analysis.variablesExcludingVariableFactor)
                {
                    block.Statements.Add(Builder.CommentStmt(ivd.ToString()));
                }
                context.OutputAttributes.Add(itdOut, new DebugInfo()
                {
                    Transform = this,
                    Name = "analysis",
                    Value = block
                });
            }
            return itdOut;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            var oldTargets = targetsOfCurrentAssignment;
            targetsOfCurrentAssignment = new Set<IExpression>(new IdentityComparer<IExpression>());
            iae = (IAssignExpression)base.ConvertAssign(iae);
            targetsOfCurrentAssignment.Add(iae.Target);
            IExpression rhs = iae.Expression;
            bool shouldDelete = false;
            foreach (IExpression target in targetsOfCurrentAssignment)
                ProcessAssign(target, iae.Expression, ref shouldDelete);
            targetsOfCurrentAssignment = oldTargets;
            return shouldDelete ? null : iae;
        }

        protected void ProcessAssign(IExpression target, IExpression rhs, ref bool shouldDelete)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
            if (ivd == null)
                return;
            if (rhs is IArrayCreateExpression)
            {
                IArrayCreateExpression iace = (IArrayCreateExpression)rhs;
                bool zeroLength = iace.Dimensions.All(dimExpr =>
                    (dimExpr is ILiteralExpression) && ((ILiteralExpression)dimExpr).Value.Equals(0));
                if (!zeroLength && iace.Initializer == null)
                    return; // variable will have assignments to elements
            }
            bool firstTime = !variablesAssigned.Contains(ivd);
            variablesAssigned.Add(ivd);
            if (!CodeRecognizer.IsStochastic(context, ivd))
                return;
            // definition of a stochastic variable
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            IExpression lhs = target;
            if (lhs is IVariableDeclarationExpression)
                lhs = Builder.VarRefExpr(ivd);
            IExpression defExpr = lhs;
            Containers defContainers = context.InputAttributes.Get<Containers>(ivd);
            int ancIndex = defContainers.GetMatchingAncestorIndex(context);
            Containers missing = defContainers.GetContainersNotInContext(context, ancIndex);
            if (firstTime)
            {
                // Create a ChannelInfo attribute for use by later transforms, e.g. MessageTransform
                ChannelInfo defChannel = ChannelInfo.DefChannel(vi);
                defChannel.decl = ivd;
                context.OutputAttributes.Set(ivd, defChannel);
                SetMarginalPrototype(ivd);
            }
            bool isDerived = context.InputAttributes.Has<DerivedVariable>(ivd);
            IAlgorithm algorithm = this.algorithmDefault;
            Algorithm algAttr = context.InputAttributes.Get<Algorithm>(ivd);
            if (algAttr != null)
                algorithm = algAttr.algorithm;
            if (algorithm is VariationalMessagePassing && ((VariationalMessagePassing)algorithm).UseDerivMessages && isDerived && firstTime)
            {
                vi.DefineAllIndexVars(context);
                IList<IStatement> stmts = Builder.StmtCollection();
                IVariableDeclaration derivDecl = vi.DeriveIndexedVariable(stmts, context, ivd.Name + "_deriv");
                context.OutputAttributes.Set(ivd, new DerivMessage(derivDecl));
                ChannelInfo derivChannel = ChannelInfo.DefChannel(vi);
                derivChannel.decl = derivDecl;
                context.OutputAttributes.Set(derivChannel.decl, derivChannel);
                context.OutputAttributes.Set(derivChannel.decl, new DescriptionAttribute("deriv of '" + ivd.Name + "'"));
                // Add the declarations
                stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
            }
            bool isInferred = context.InputAttributes.Has<IsInferred>(ivd);
            bool isPointEstimate = context.InputAttributes.Has<PointEstimate>(ivd);
            if (this.analysis.variablesExcludingVariableFactor.Contains(ivd))
            {
                this.variablesLackingVariableFactor.Add(ivd);
                return;
            }
            if (isDerived && !isInferred && !isPointEstimate)
                return;

            IExpression useExpr2 = null;
            if (firstTime)
            {
                // create marginal and use channels
                vi.DefineAllIndexVars(context);
                IList<IStatement> stmts = Builder.StmtCollection();

                CreateMarginalChannel(ivd, vi, stmts);
                CreateUseChannel(ivd, vi, stmts);

                // Add the declarations
                stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
                context.InputAttributes.Set(useOfVariable[ivd], defContainers);
            }
            if (!useOfVariable.ContainsKey(ivd))
            {
                Error("cannot find use channel of " + ivd);
                return;
            }
            IExpression useExpr = Builder.ReplaceVariable(lhs, ivd, useOfVariable[ivd]);
            IExpression marginalExpr = Builder.ReplaceVariable(lhs, ivd, marginalOfVariable[ivd]);
            InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(ivd);
            Type[] genArgs = new Type[] { defExpr.GetExpressionType() };
            if (rhs is IMethodInvokeExpression)
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)rhs;
                if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Factor.Copy)) && ancIndex < context.InputStack.Count - 2)
                {
                    IExpression arg = imie.Arguments[0];
                    IVariableDeclaration ivd2 = Recognizer.GetVariableDeclaration(arg);
                    if (context.InputAttributes.Get<MarginalPrototype>(ivd) == context.InputAttributes.Get<MarginalPrototype>(ivd2))
                    {
                        // if a variable is a copy, use the original expression since it will give more precise dependencies.
                        defExpr = arg;
                        shouldDelete = true;
                        bool makeClone = false;
                        if (makeClone)
                        {
                            VariableInformation vi2 = VariableInformation.GetVariableInformation(context, ivd2);
                            IList<IStatement> stmts = Builder.StmtCollection();
                            List<IList<IExpression>> indices = Recognizer.GetIndices(defExpr);
                            IVariableDeclaration useDecl2 = vi2.DeriveIndexedVariable(stmts, context, ivd2.Name + "_use", indices);
                            useExpr2 = Builder.VarRefExpr(useDecl2);
                            Containers defContainers2 = context.InputAttributes.Get<Containers>(ivd2);
                            int ancIndex2 = defContainers2.GetMatchingAncestorIndex(context);
                            Containers missing2 = defContainers2.GetContainersNotInContext(context, ancIndex2);
                            stmts = Containers.WrapWithContainers(stmts, missing2.outputs);
                            context.AddStatementsBeforeAncestorIndex(ancIndex2, stmts);
                            context.InputAttributes.Set(useDecl2, defContainers2);

                            // TODO: call CreateUseChannel
                            ChannelInfo usageChannel = ChannelInfo.UseChannel(vi2);
                            usageChannel.decl = useDecl2;
                            context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, useDecl2);
                            context.InputAttributes.CopyObjectAttributesTo<DerivMessage>(vi.declaration, context.OutputAttributes, useDecl2);
                            context.OutputAttributes.Set(useDecl2, usageChannel);
                            //context.OutputAttributes.Set(useDecl2, new DescriptionAttribute("use of '" + ivd.Name + "'"));
                            context.OutputAttributes.Remove<InitialiseTo>(vi.declaration);
                            SetMarginalPrototype(useDecl2);

                            IExpression copyExpr = Builder.StaticGenericMethod(
                                new Func<PlaceHolder, PlaceHolder>(Factor.Copy), genArgs, useExpr2);
                            var copyStmt = Builder.AssignStmt(useExpr, copyExpr);
                            context.AddStatementAfterCurrent(copyStmt);
                        }
                    }
                }
            }

            // Add the variable factor
            IExpression variableFactorExpr;
            bool isGateExitRandom = context.InputAttributes.Has<VariationalMessagePassing.GateExitRandomVariable>(ivd);
            if (isGateExitRandom)
            {
                variableFactorExpr = Builder.StaticGenericMethod(
                    new Models.FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Gate.ExitingVariable),
                    genArgs, defExpr, marginalExpr);
            }
            else
            {
                Delegate d = algorithm.GetVariableFactor(isDerived, it != null);
                if (isPointEstimate)
                    d = new Models.FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Factor.VariablePoint);
                if (it == null)
                {
                    variableFactorExpr = Builder.StaticGenericMethod(d, genArgs, defExpr, marginalExpr);
                }
                else
                {
                    IExpression initExpr = Builder.ReplaceExpression(lhs, Builder.VarRefExpr(ivd), it.initialMessagesExpression);
                    variableFactorExpr = Builder.StaticGenericMethod(d, genArgs, defExpr, initExpr, marginalExpr);
                }
            }
            context.InputAttributes.CopyObjectAttributesTo<GivePriorityTo>(ivd, context.OutputAttributes, variableFactorExpr);
            context.InputAttributes.CopyObjectAttributesTo<Algorithm>(ivd, context.OutputAttributes, variableFactorExpr);
            context.OutputAttributes.Set(variableFactorExpr, new IsVariableFactor());
            var assignStmt = Builder.AssignStmt(useExpr2 == null ? useExpr : useExpr2, variableFactorExpr);
            context.AddStatementAfterCurrent(assignStmt);
        }

        private void CreateMarginalChannel(IVariableDeclaration ivd, VariableInformation vi, IList<IStatement> stmts)
        {
            IVariableDeclaration marginalDecl = vi.DeriveIndexedVariable(stmts, context, ivd.Name + "_marginal");
            marginalOfVariable[ivd] = marginalDecl;
            ChannelInfo marginalChannel = ChannelInfo.MarginalChannel(vi);
            marginalChannel.decl = marginalDecl;
            context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, marginalChannel.decl);
            context.OutputAttributes.Set(marginalChannel.decl, marginalChannel);
            context.OutputAttributes.Set(marginalChannel.decl, new DescriptionAttribute("marginal of '" + ivd.Name + "'"));
            SetMarginalPrototype(marginalDecl);
        }

        private void CreateUseChannel(IVariableDeclaration ivd, VariableInformation vi, IList<IStatement> stmts)
        {
            IVariableDeclaration useDecl = vi.DeriveIndexedVariable(stmts, context, ivd.Name + "_use");
            useOfVariable[ivd] = useDecl;
            ChannelInfo usageChannel = ChannelInfo.UseChannel(vi);
            usageChannel.decl = useDecl;
            context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, useDecl);
            context.InputAttributes.CopyObjectAttributesTo<DerivMessage>(vi.declaration, context.OutputAttributes, useDecl);
            context.OutputAttributes.Set(useDecl, usageChannel);
            context.OutputAttributes.Set(useDecl, new DescriptionAttribute("use of '" + ivd.Name + "'"));
            context.OutputAttributes.Remove<InitialiseTo>(vi.declaration);
            SetMarginalPrototype(useDecl);
        }

        private void SetMarginalPrototype(IVariableDeclaration ivd)
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

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            if (Recognizer.IsBeingMutated(context, ivre))
                return ivre;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ivre);
            if (ivd == null)
                return ivre;
            if (!variablesAssigned.Contains(ivd))
                Error(ivd + " is used before it is assigned to");
            IVariableDeclaration useDecl;
            if (useOfVariable.TryGetValue(ivd, out useDecl))
                return Builder.VarRefExpr(useDecl);
            return ivre;
        }

        // Record the containers that a variable is created in
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            context.InputAttributes.Remove<Containers>(ivde.Variable);
            context.InputAttributes.Set(ivde.Variable, new Containers(context));
            return ivde;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.IsInfer(imie))
                return ConvertInfer(imie);
            foreach (IExpression arg in imie.Arguments)
            {
                if (arg is IAddressOutExpression)
                {
                    IAddressOutExpression iaoe = (IAddressOutExpression)arg;
                    targetsOfCurrentAssignment.Add(iaoe.Expression);
                }
            }
            if (Recognizer.IsStaticGenericMethod(imie, new Func<IList<PlaceHolder>, int[][], PlaceHolder[][]>(Factor.JaggedSubarray)))
            {
                IExpression arrayExpr = imie.Arguments[0];
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(imie.Arguments[0]);
                if (ivd != null && (arrayExpr is IVariableReferenceExpression) && this.variablesLackingVariableFactor.Contains(ivd)
                    && !marginalOfVariable.ContainsKey(ivd))
                {
                    VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                    IList<IStatement> stmts = Builder.StmtCollection();
                    CreateMarginalChannel(ivd, vi, stmts);

                    Containers defContainers = context.InputAttributes.Get<Containers>(ivd);
                    int ancIndex = defContainers.GetMatchingAncestorIndex(context);
                    Containers missing = defContainers.GetContainersNotInContext(context, ancIndex);
                    stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                    context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);

                    // none of the arguments should need to be transformed
                    IExpression indicesExpr = imie.Arguments[1];
                    IExpression marginalExpr = Builder.VarRefExpr(marginalOfVariable[ivd]);
                    IMethodInvokeExpression mie = Builder.StaticGenericMethod(new Models.FuncOut<IList<PlaceHolder>, int[][], IList<PlaceHolder>, PlaceHolder[][]>(Factor.JaggedSubarrayWithMarginal),
                        new Type[] { Utilities.Util.GetElementType(arrayExpr.GetExpressionType()) },
                        arrayExpr, indicesExpr, marginalExpr);
                    return mie;
                }
            }
            return base.ConvertMethodInvoke(imie);
        }

        /// <summary>
        /// Modify the argument of Infer to be the marginal channel variable i.e. Infer(a) transforms to Infer(a_marginal)
        /// </summary>
        /// <param name="imie"></param>
        /// 
        /// <returns>The modified expression</returns>
        protected IExpression ConvertInfer(IMethodInvokeExpression imie)
        {
            IVariableReferenceExpression ivre = imie.Arguments[0] as IVariableReferenceExpression;
            if (ivre == null)
            {
                //Error("Argument to Infer() must be a variable reference, was " + imie.Arguments[0] + ".");
                return imie;
            }
            // Find expression for the marginal of interest
            IVariableDeclaration ivd = ivre.Variable.Resolve();
            IVariableDeclaration marginalDecl;
            ExpressionEvaluator eval = new ExpressionEvaluator();
            QueryType query = (imie.Arguments.Count < 3) ? null : (QueryType)eval.Evaluate(imie.Arguments[2]);
            bool isOutput = (query == QueryTypes.MarginalDividedByPrior);
            Dictionary<IVariableDeclaration, IVariableDeclaration> dict = isOutput ? useOfVariable : marginalOfVariable;
            if (!dict.TryGetValue(ivd, out marginalDecl))
            {
                return imie; // The argument is constant
            }
            IMethodInvokeExpression mie = Builder.MethodInvkExpr();
            mie.Method = imie.Method;
            mie.Arguments.Add(Builder.VarRefExpr(marginalDecl));
            for (int i = 1; i < imie.Arguments.Count; i++)
            {
                mie.Arguments.Add(imie.Arguments[i]);
            }
            // move the IsInferred attribute to the marginal channel
            //context.OutputAttributes.Remove<IsInferred>(ivd);
            if (!context.OutputAttributes.Has<IsInferred>(marginalDecl))
            {
                context.OutputAttributes.Set(marginalDecl, new IsInferred());
            }
            return mie;
        }
    }

    /// <summary>
    /// Determines which variables need a variable factor.
    /// </summary>
    internal class VariableAnalysisTransform : ShallowCopyTransform
    {
        public Set<IVariableDeclaration> variablesExcludingVariableFactor = new Set<IVariableDeclaration>(new IdentityComparer<IVariableDeclaration>());
        bool useJaggedSubarrayWithMarginal;

        public override string Name
        {
            get
            {
                return "VariableAnalysisTransform";
            }
        }

        internal VariableAnalysisTransform(bool useJaggedSubarrayWithMarginal)
        {
            this.useJaggedSubarrayWithMarginal = useJaggedSubarrayWithMarginal;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (useJaggedSubarrayWithMarginal && Recognizer.IsStaticGenericMethod(imie, new Func<IList<PlaceHolder>, int[][], PlaceHolder[][]>(Factor.JaggedSubarray)))
            {
                IExpression arrayExpr = imie.Arguments[0];
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(arrayExpr);
                // restrict to IVariableReferenceExpression for simplicity
                if (ivd != null && !context.InputAttributes.Has<PointEstimate>(ivd) && arrayExpr is IVariableReferenceExpression)
                {
                    variablesExcludingVariableFactor.Add(ivd);
                }
            }
            return base.ConvertMethodInvoke(imie);
        }
    }

    internal class DerivMessage : ICompilerAttribute
    {
        public IVariableDeclaration decl;

        public DerivMessage(IVariableDeclaration decl)
        {
            this.decl = decl;
        }
    }
}