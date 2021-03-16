// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Removes increment statements that are not in while loops, or whose Cancels input is not in the same loop.
    /// On exit, there will be no increment statements outside of while loops, and every Cancels input will have a source statement.
    /// </summary>
    internal class IncrementPruningTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "IncrementPruningTransform";
            }
        }

        readonly Dictionary<IStatement, IStatement> replacements = new Dictionary<IStatement, IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
        readonly HashSet<IncrementStatement> incrementStatements = new HashSet<IncrementStatement>();
        readonly HashSet<IStatement> visitedStatements = new HashSet<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
        bool inWhileLoop;
        int ancestorIndexOfWhile;

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            replacements.Clear();
            base.DoConvertMethodBody(outputs, inputs);
        }

        protected override IStatement ConvertWhile(IWhileStatement iws)
        {
            bool wasInWhileLoop = this.inWhileLoop;
            if (!wasInWhileLoop)
            {
                // collect all statements in the body of this loop (and any nested loops).
                DeadCodeTransform.ForEachStatement(iws.Body.Statements,
                    _ =>
                    {
                    }, _ =>
                    {
                    },
                    _ =>
                    {
                    }, _ =>
                    {
                    },
                    ist =>
                {
                    visitedStatements.Add(ist);
                });
                // collect all IncrementStatement attributes in the body of this loop (and any nested loops).
                DeadCodeTransform.ForEachStatement(iws.Body.Statements,
                    _ =>
                    {
                    }, _ =>
                    {
                    },
                    _ =>
                    {
                    }, _ =>
                    {
                    },
                    ist =>
                    {
                        var incrementStatement = context.GetAttribute<IncrementStatement>(ist);
                        if (incrementStatement != null)
                        {
                            DependencyInformation di = context.GetAttribute<DependencyInformation>(ist);
                            if (di != null && !di.GetDependenciesOfType(DependencyType.Cancels).All(visitedStatements.Contains))
                            {
                                // Remove increment statements whose Cancels input is not available, implying that there is no purpose in carrying out the increment.
                                // This happens because IterationTransform ignores Cancels edges.
                                replacements[ist] = null;
                            }
                            else
                                incrementStatements.Add(incrementStatement);
                        }
                    });
                ancestorIndexOfWhile = context.Depth - 1;
            }
            this.inWhileLoop = true;
            var ws = base.ConvertWhile(iws);
            this.inWhileLoop = wasInWhileLoop;
            if (!wasInWhileLoop)
                incrementStatements.Clear();
            return ws;
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            if (ist is IWhileStatement iws)
                return ConvertWhile(iws);
            if (replacements.ContainsKey(ist))
                return null;
            bool isIncrement = context.InputAttributes.Has<IncrementStatement>(ist);
            if (this.inWhileLoop == false)
            {
                if (isIncrement)
                {
                    replacements[ist] = null;
                    return null;
                }
            }
            else if (incrementStatements.Count > 0)
            {
                var attr = context.GetAttribute<HasIncrement>(ist);
                if (attr != null && incrementStatements.Contains(attr.incrementStatement))
                {
                    // hoist this statement out of the while loop.
                    // TODO: hoisting ist is only valid if its requirements are met when hoisted.
                    context.AddStatementBeforeAncestorIndex(ancestorIndexOfWhile, ist);
                    return null;
                }
                else if (isIncrement)
                {
                    // Add to the initializerSet (assuming the non-increment statement is hoisted).
                    // This will only work correctly if the non-increment is in DependencyInformation.Initializers.
                    // This code currently does nothing since InitializerSet is only attached later by SchedulingTransform.
                    var iws2 = context.GetAncestor(ancestorIndexOfWhile);
                    var initializerSet = context.GetAttribute<InitializerSet>(iws2);
                    if (initializerSet != null)
                    {
                        initializerSet.initializers.Add(ist);
                    }
                }
            }
            DependencyInformation di = context.GetAttribute<DependencyInformation>(ist);
            if (di != null)
                di.Replace(replacements);
            visitedStatements.Add(ist);
            return base.DoConvertStatement(ist);
        }
    }
}
