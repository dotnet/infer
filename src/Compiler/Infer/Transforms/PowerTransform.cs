// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Handles repeat blocks by inserting the appropriate power plate constructs.
    /// </summary>
    internal class PowerTransform : ShallowCopyTransform
    {
        Set<IVariableDeclaration> cutVariables = new Set<IVariableDeclaration>();

        public override string Name
        {
            get { return "PowerTransform"; }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (Recognizer.IsStaticGenericMethod(imie, new Models.FuncOut<object, object, object>(LowPriority.SequentialCut)))
            {
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(imie.Arguments[1]);
                if(ivd != null)
                    cutVariables.Add(ivd);
            }
            return base.ConvertMethodInvoke(imie);
        }

        protected override IStatement ConvertRepeat(IRepeatStatement irs)
        {
            var loopSize = irs.Count;
            if (!((loopSize is IVariableReferenceExpression) || (loopSize is IArrayIndexerExpression) ||
                  (loopSize is IArgumentReferenceExpression) || (loopSize is ILiteralExpression)
                  || (loopSize is IPropertyReferenceExpression)))
            {
                Error("Invalid expression type for the count of a repeat block (" + loopSize.GetType().Name + "): " + loopSize);
                return irs;
            }
            var body = ConvertBlock(irs.Body);
            context.AddStatementsAfterCurrent(body.Statements);
            return null;
        }

        // Record the loop context that a variable is created in
        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            //context.InputAttributes.Remove<RepeatContext>(ivde.Variable);
            // do not override a previously attached RepeatContext
            if (!context.InputAttributes.Has<RepeatContext>(ivd))
                context.InputAttributes.Set(ivd, new RepeatContext(context));
            context.InputAttributes.Remove<Containers>(ivd);
            context.InputAttributes.Set(ivd, new Containers(context));
            return ivd;
        }

        // Shallow copy variable references for efficiency
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            return ConvertWithReplication(ivre);
        }

        // Shallow copy array creations for efficiency
        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            return iace;
        }

        /// <summary>
        /// Converts an array index expression to replicate variables referenced in a loop.
        /// </summary>
        /// <param name="iaie">The array indexer expression to convert</param>
        /// <returns>The new expression</returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            return ConvertWithReplication(iaie);
        }

        protected IExpression ConvertWithReplication(IExpression expr)
        {
            IVariableDeclaration baseVar = Recognizer.GetVariableDeclaration(expr);
            // Check if this is an index local variable
            if (baseVar == null) return expr;
            // Check if the variable is stochastic
            if (!CodeRecognizer.IsStochastic(context, baseVar)) return expr;
            if (cutVariables.Contains(baseVar))
                return expr;

            // Get the repeat context for this variable
            RepeatContext lc = context.InputAttributes.Get<RepeatContext>(baseVar);
            if (lc == null)
            {
                Error("Repeat context not found for '" + baseVar.Name + "'.");
                return expr;
            }

            // Get the reference loop context for this expression
            var rlc = lc.GetReferenceRepeatContext(context);
            // If the reference is in the same loop context as the declaration, do nothing.
            if (rlc.repeats.Count == 0) return expr;

            // Determine if this expression is being defined (is on the LHS of an assignment)
            bool isDef = Recognizer.IsBeingMutated(context, expr);

            Containers containers = context.InputAttributes.Get<Containers>(baseVar);

            for (int currentRepeat = 0; currentRepeat < rlc.repeatCounts.Count; currentRepeat++)
            {
                IExpression repeatCount = rlc.repeatCounts[currentRepeat];
                IRepeatStatement repeat = rlc.repeats[currentRepeat];

                // must replicate across this loop.
                if (isDef)
                {
                    Error("Cannot re-define a variable in a repeat block.");
                    continue;
                }
                // are we replicating the argument of Gate.Cases?
                IMethodInvokeExpression imie = context.FindAncestor<IMethodInvokeExpression>();
                if (imie != null)
                {
                    if (Recognizer.IsStaticMethod(imie, typeof (Gate), "Cases"))
                        Error("'if(" + expr + ")' should be placed outside 'repeat(" + repeatCount + ")'");
                    if (Recognizer.IsStaticMethod(imie, typeof (Gate), "CasesInt"))
                        Error("'case(" + expr + ")' or 'switch(" + expr + ")' should be placed outside 'repeat(" + repeatCount + ")'");
                }

                VariableInformation varInfo = VariableInformation.GetVariableInformation(context, baseVar);
                IList<IStatement> stmts = Builder.StmtCollection();

                List<IList<IExpression>> inds = Recognizer.GetIndices(expr);
                IVariableDeclaration repVar = varInfo.DeriveIndexedVariable(stmts, context, VariableInformation.GenerateName(context, varInfo.Name + "_rpt"), inds);
                if (!context.InputAttributes.Has<DerivedVariable>(repVar))
                    context.OutputAttributes.Set(repVar, new DerivedVariable());
                if (context.InputAttributes.Has<ChannelInfo>(baseVar))
                {
                    VariableInformation repVarInfo = VariableInformation.GetVariableInformation(context, repVar);
                    ChannelInfo ci = ChannelInfo.UseChannel(repVarInfo);
                    ci.decl = repVar;
                    context.OutputAttributes.Set(repVar, ci);
                }
                // set the RepeatContext of repVar to include all repeats up to this one (so that it doesn't get Entered again)
                List<IRepeatStatement> repeats = new List<IRepeatStatement>(lc.repeats);
                for (int i = 0; i <= currentRepeat; i++)
                {
                    repeats.Add(rlc.repeats[i]);
                }
                context.OutputAttributes.Remove<RepeatContext>(repVar);
                context.OutputAttributes.Set(repVar, new RepeatContext(repeats));

                // Create replicate factor
                Type returnType = Builder.ToType(repVar.VariableType);
                IMethodInvokeExpression powerPlateMethod = Builder.StaticGenericMethod(
                    new Func<PlaceHolder, double, PlaceHolder>(PowerPlate.Enter),
                    new Type[] {returnType}, expr, repeatCount);

                IExpression assignExpression = Builder.AssignExpr(Builder.VarRefExpr(repVar), powerPlateMethod);
                // Copy attributes across from variable to replication expression
                context.InputAttributes.CopyObjectAttributesTo<Algorithm>(baseVar, context.OutputAttributes, powerPlateMethod);
                context.InputAttributes.CopyObjectAttributesTo<DivideMessages>(baseVar, context.OutputAttributes, powerPlateMethod);
                context.InputAttributes.CopyObjectAttributesTo<GivePriorityTo>(baseVar, context.OutputAttributes, powerPlateMethod);
                stmts.Add(Builder.ExprStatement(assignExpression));

                // add any containers missing from context.
                containers = new Containers(context);
                // remove inner repeats
                for (int i = currentRepeat + 1; i < rlc.repeatCounts.Count; i++)
                {
                    containers = containers.RemoveOneRepeat(rlc.repeats[i]);
                }
                context.OutputAttributes.Set(repVar, containers);
                containers = containers.RemoveOneRepeat(repeat);
                containers = Containers.RemoveUnusedLoops(containers, context, powerPlateMethod);
                if (context.InputAttributes.Has<DoNotSendEvidence>(baseVar)) containers = Containers.RemoveStochasticConditionals(containers, context);
                //Containers shouldBeEmpty = containers.GetContainersNotInContext(context, context.InputStack.Count);
                //if (shouldBeEmpty.inputs.Count > 0) { Error("Internal: Variable is out of scope"); return expr; }
                int ancIndex = containers.GetMatchingAncestorIndex(context);
                Containers missing = containers.GetContainersNotInContext(context, ancIndex);
                stmts = Containers.WrapWithContainers(stmts, missing.inputs);
                // must convert the output since it may contain 'if' conditions
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts, true);
                baseVar = repVar;
                expr = Builder.VarRefExpr(repVar);
            }

            return expr;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}