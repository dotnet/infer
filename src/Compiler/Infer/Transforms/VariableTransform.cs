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
        private readonly Set<IVariableDeclaration> variablesAssigned = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);

        private readonly Set<IVariableDeclaration> variablesLackingVariableFactor = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);

        private Set<IExpression> targetsOfCurrentAssignment;

        private readonly Dictionary<object, IVariableDeclaration> useOfVariable =
            new Dictionary<object, IVariableDeclaration>(ReferenceEqualityComparer<object>.Instance);

        private readonly Dictionary<object, IVariableDeclaration> marginalOfVariable =
            new Dictionary<object, IVariableDeclaration>(ReferenceEqualityComparer<object>.Instance);

        private readonly ModelCompiler compiler;
        private readonly IAlgorithm algorithmDefault;
        private VariableAnalysisTransform analysis;

        public VariableTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
            this.algorithmDefault = compiler.Algorithm;
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
            targetsOfCurrentAssignment = new Set<IExpression>(ReferenceEqualityComparer<IExpression>.Instance);
            iae = (IAssignExpression)base.ConvertAssign(iae);
            targetsOfCurrentAssignment.Add(iae.Target);
            IExpression rhs = iae.Expression;
            bool shouldDelete = false;
            foreach (IExpression target in targetsOfCurrentAssignment)
                ProcessAssign(target, rhs, ref shouldDelete);
            targetsOfCurrentAssignment = oldTargets;
            return shouldDelete ? null : iae;
        }

        protected void ProcessAssign(IExpression target, IExpression rhs, ref bool shouldDelete)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
            if (ivd == null)
                return;
            if (rhs is IArrayCreateExpression iace)
            {
                bool zeroLength = iace.Dimensions.All(dimExpr =>
                    (dimExpr is ILiteralExpression ile) && ile.Value.Equals(0));
                if (!zeroLength && iace.Initializer == null)
                    return; // variable will have assignments to elements
            }
            bool firstTime = !variablesAssigned.Contains(ivd);
            variablesAssigned.Add(ivd);
            bool isInferred = context.InputAttributes.Has<IsInferred>(ivd);
            bool isStochastic = CodeRecognizer.IsStochastic(context, ivd);
            //if (!isStochastic) return;
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            if (!isStochastic && !vi.NeedsMarginalDividedByPrior) return;
            Containers defContainers = context.InputAttributes.Get<Containers>(ivd);
            int ancIndex = defContainers.GetMatchingAncestorIndex(context);
            Containers missing = defContainers.GetContainersNotInContext(context, ancIndex);
            // definition of a stochastic variable
            IExpression lhs = target;
            if (lhs is IVariableDeclarationExpression)
                lhs = Builder.VarRefExpr(ivd);
            IExpression defExpr = lhs;
            if (firstTime)
            {
                // Create a ChannelInfo attribute for use by later transforms, e.g. MessageTransform
                ChannelInfo defChannel = ChannelInfo.DefChannel(vi);
                defChannel.decl = ivd;
                context.OutputAttributes.Set(ivd, defChannel);
            }
            bool isDerived = context.InputAttributes.Has<DerivedVariable>(ivd);
            IAlgorithm algorithm = this.algorithmDefault;
            Algorithm algAttr = context.InputAttributes.Get<Algorithm>(ivd);
            if (algAttr != null)
                algorithm = algAttr.algorithm;
            if (algorithm is VariationalMessagePassing vmp && vmp.UseDerivMessages && isDerived && firstTime)
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
            bool hasPointEstimate = context.InputAttributes.Has<PointEstimate>(ivd);
            if (hasPointEstimate && firstTime)
            {
                if (!context.InputAttributes.Has<InitialiseTo>(ivd))
                {
                    Error($"{ivd.Name} has {nameof(PointEstimate)} but is not initialised");
                }
                if (!compiler.InitialisationAffectsSchedule)
                {
                    Error($"{ivd.Name} has {nameof(PointEstimate)} but {nameof(ModelCompiler.InitialisationAffectsSchedule)} is false");
                }
            }
            if (this.analysis.variablesExcludingVariableFactor.Contains(ivd))
            {
                this.variablesLackingVariableFactor.Add(ivd);
                // ivd will get a marginal channel in ConvertMethodInvoke
                useOfVariable[ivd] = ivd;
                return;
            }
            if (isDerived && !isInferred && !hasPointEstimate)
                return;

            IExpression useExpr2 = null;
            if (firstTime)
            {
                // create marginal and use channels
                vi.DefineAllIndexVars(context);
                IList<IStatement> stmts = Builder.StmtCollection();

                CreateMarginalChannel(ivd, vi, stmts);
                //if (isStochastic)
                {
                    CreateUseChannel(ivd, vi, stmts);
                    context.InputAttributes.Set(useOfVariable[ivd], defContainers);
                }

                // Add the declarations
                stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
            }
            if (isStochastic && !useOfVariable.ContainsKey(ivd))
            {
                Error("cannot find use channel of " + ivd);
                return;
            }
            IExpression marginalExpr = Builder.ReplaceVariable(lhs, ivd, marginalOfVariable[ivd]);
            IExpression useExpr = isStochastic || true ? Builder.ReplaceVariable(lhs, ivd, useOfVariable[ivd]) : marginalExpr;
            InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(ivd);
            Type[] genArgs = new Type[] { defExpr.GetExpressionType() };
            if (rhs is IMethodInvokeExpression imie && 
                Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Clone.Copy)) && 
                ancIndex < context.InputStack.Count - 2)
            {
                IExpression arg = imie.Arguments[0];
                IVariableDeclaration ivd2 = Recognizer.GetVariableDeclaration(arg);
                if (ivd2 != null && context.InputAttributes.Get<MarginalPrototype>(ivd) == context.InputAttributes.Get<MarginalPrototype>(ivd2))
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

                        IExpression copyExpr = Builder.StaticGenericMethod(
                            new Func<PlaceHolder, PlaceHolder>(Clone.Copy), genArgs, useExpr2);
                        var copyStmt = Builder.AssignStmt(useExpr, copyExpr);
                        context.AddStatementAfterCurrent(copyStmt);
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
                if (hasPointEstimate)
                {
                    d = new Models.FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.VariablePoint);
                }
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
            if (isStochastic)
                context.OutputAttributes.Set(variableFactorExpr, new IsVariableFactor());
            var assignStmt = Builder.AssignStmt(useExpr2 ?? useExpr, variableFactorExpr);
            context.AddStatementAfterCurrent(assignStmt);
        }

        private void CreateMarginalChannel(object decl, VariableInformation vi, IList<IStatement> stmts)
        {
            marginalOfVariable[decl] = CreateMarginalChannel(vi, stmts);
        }

        private IVariableDeclaration CreateMarginalChannel(VariableInformation vi, IList<IStatement> stmts)
        {
            IVariableDeclaration marginalDecl = vi.DeriveIndexedVariable(stmts, context, vi.Name + "_marginal");
            ChannelInfo marginalChannel = ChannelInfo.MarginalChannel(vi);
            marginalChannel.decl = marginalDecl;
            context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, marginalDecl);
            context.OutputAttributes.Set(marginalDecl, marginalChannel);
            context.OutputAttributes.Set(marginalDecl, new DescriptionAttribute("marginal of '" + vi.Name + "'"));
            // The following lines are needed for AddMarginalStatements
            VariableInformation marginalInformation = VariableInformation.GetVariableInformation(context, marginalDecl);
            marginalInformation.IsStochastic = true;
            return marginalDecl;
        }

        private void CreateUseChannel(object decl, VariableInformation vi, IList<IStatement> stmts)
        {
            useOfVariable[decl] = CreateUseChannel(vi, stmts);
        }

        private IVariableDeclaration CreateUseChannel(VariableInformation vi, IList<IStatement> stmts)
        { 
            IVariableDeclaration useDecl = vi.DeriveIndexedVariable(stmts, context, vi.Name + "_use");
            ChannelInfo usageChannel = ChannelInfo.UseChannel(vi);
            usageChannel.decl = useDecl;
            context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, useDecl);
            context.InputAttributes.CopyObjectAttributesTo<DerivMessage>(vi.declaration, context.OutputAttributes, useDecl);
            context.OutputAttributes.Set(useDecl, usageChannel);
            context.OutputAttributes.Set(useDecl, new DescriptionAttribute("use of '" + vi.Name + "'"));
            context.OutputAttributes.Remove<InitialiseTo>(vi.declaration);
            // The following lines are needed for AddMarginalStatements
            VariableInformation useInformation = VariableInformation.GetVariableInformation(context, useDecl);
            useInformation.IsStochastic = vi.IsStochastic;
            useInformation.NeedsMarginalDividedByPrior = vi.NeedsMarginalDividedByPrior;
            return useDecl;
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
                if (arg is IAddressOutExpression iaoe)
                {
                    if (targetsOfCurrentAssignment == null)
                    {
                        bool shouldDelete = false;
                        ProcessAssign(iaoe.Expression, null, ref shouldDelete);
                    }
                    else
                    {
                        targetsOfCurrentAssignment.Add(iaoe.Expression);
                    }
                }
            }
            if (Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, int[][], PlaceHolder[][]>(Collection.JaggedSubarray)))
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
                    IMethodInvokeExpression mie = Builder.StaticGenericMethod(new Models.FuncOut<IReadOnlyList<PlaceHolder>, int[][], IReadOnlyList<PlaceHolder>, PlaceHolder[][]>(Collection.JaggedSubarrayWithMarginal),
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
            IExpression arg = imie.Arguments[0];
            object decl = AddMarginalStatements(arg);
            //if (arg is IArgumentReferenceExpression iare) AddMarginalStatements(iare);
            //else if (arg is IVariableReferenceExpression ivre) decl = ivre.Variable.Resolve();
            //else return imie;
            // Find expression for the marginal of interest
            IVariableDeclaration marginalDecl;
            ExpressionEvaluator eval = new ExpressionEvaluator();
            QueryType query = (imie.Arguments.Count < 3) ? null : (QueryType)eval.Evaluate(imie.Arguments[2]);
            bool isOutput = (query == QueryTypes.MarginalDividedByPrior);
            if (!isOutput || !useOfVariable.TryGetValue(decl, out marginalDecl))
                marginalDecl = marginalOfVariable[decl];
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

        private object AddMarginalStatements(IExpression expr)
        {
            object decl = Recognizer.GetDeclaration(expr);
            if (marginalOfVariable.ContainsKey(decl)) return decl;

            VariableInformation vi = VariableInformation.GetVariableInformation(context, decl);
            vi.DefineAllIndexVars(context);
            IList<IStatement> stmts = Builder.StmtCollection();

            CreateMarginalChannel(decl, vi, stmts);
            CreateUseChannel(decl, vi, stmts);
            IVariableDeclaration useDecl = useOfVariable[decl];
            useOfVariable.Remove(decl);
            IExpression useExpr = Builder.VarRefExpr(useDecl);
            IVariableDeclaration marginalDecl = marginalOfVariable[decl];
            IExpression marginalExpr = Builder.VarRefExpr(marginalDecl);
            Type[] genArgs = new Type[] { expr.GetExpressionType() };
            IAlgorithm algorithm = this.algorithmDefault;
            Delegate d = algorithm.GetVariableFactor(true, false);
            IExpression variableFactorExpr = Builder.StaticGenericMethod(d, genArgs, expr, marginalExpr);
            //IExpression variableFactorExpr = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Factor.Copy), genArgs, expr);
            context.OutputAttributes.Set(variableFactorExpr, new IsVariableFactor());
            var assignStmt = Builder.AssignStmt(useExpr, variableFactorExpr);
            stmts.Add(assignStmt);

            context.AddStatementsBeforeCurrent(stmts);
            return decl;
        }
    }

    /// <summary>
    /// Determines which variables need a variable factor.
    /// </summary>
    internal class VariableAnalysisTransform : ShallowCopyTransform
    {
        public Set<IVariableDeclaration> variablesExcludingVariableFactor = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
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
            if (useJaggedSubarrayWithMarginal && Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, int[][], PlaceHolder[][]>(Collection.JaggedSubarray)))
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

    /// <summary>
    /// Attribute used to mark pseudo-factors corresponding to variables in the factor graph
    /// </summary>
    internal class IsVariableFactor : ICompilerAttribute
    {
    }
}