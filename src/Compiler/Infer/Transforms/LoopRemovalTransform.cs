// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Removes loops whose body does not refer to the loop index.
    /// Requires LoopCuttingTransform to have been applied to the input.
    /// </summary>
    internal class LoopRemovalTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "LoopRemovalTransform"; }
        }

        Set<IVariableDeclaration> variablesInBody = new Set<IVariableDeclaration>();
        Set<IVariableDeclaration> variablesInConditions = new Set<IVariableDeclaration>();
        IList<IStatement> innermostConditionBlock;

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            // innermostConditionBlock will be set when transforming the body.
            innermostConditionBlock = null;
            Set<IVariableDeclaration> oldVariablesInBody = (Set<IVariableDeclaration>)variablesInBody.Clone();
            variablesInBody.Clear();
            Set<IVariableDeclaration> oldVariablesInConditions = (Set<IVariableDeclaration>)variablesInConditions.Clone();
            variablesInConditions.Clear();
            var newBody = ConvertBlock(ifs.Body);
            if (newBody.Statements.Count == 0) return null;
            var loopVar = Recognizer.LoopVariable(ifs);
            var loopSize = Recognizer.LoopSizeExpression(ifs);
            bool bodyContainsLoopVar = variablesInBody.Contains(loopVar);
            bool conditionsContainLoopVar = variablesInConditions.Contains(loopVar);
            variablesInBody.AddRange(oldVariablesInBody);
            variablesInConditions.AddRange(oldVariablesInConditions);
            if (!bodyContainsLoopVar && !conditionsContainLoopVar)
            {
                if (loopSize is ILiteralExpression)
                {
                    int loopSizeAsInt = (int)((ILiteralExpression)loopSize).Value;
                    if (loopSizeAsInt <= 0) return null;
                    else if (newBody.Statements.Count > 1) throw new InferCompilerException("newBody.Statements.Count > 1");
                    else return newBody.Statements[0];
                }
                else
                {
                    // Convert 
                    //   for (int i=0; i<loopSize; i++) { stmt; }
                    // into
                    //   if (loopSize>0) { stmt; }
                    // Record the variables used in the new condition.
                    AnalyzeCondition(loopSize);
                    var condition = Builder.BinaryExpr(loopSize, BinaryOperator.GreaterThan, Builder.LiteralExpr(0));
                    IConditionStatement ics = Builder.CondStmt(condition, newBody);
                    context.InputAttributes.CopyObjectAttributesTo(newBody.Statements[0], context.OutputAttributes, ics);
                    // if innermostConditionBlock is null after transforming the body, then this is the innermost.
                    if (innermostConditionBlock == null)
                        innermostConditionBlock = ics.Then.Statements;
                    return ics;
                }
            }
            // Transform the rest normally.
            IForStatement fs;
            if (!bodyContainsLoopVar && conditionsContainLoopVar)
            {
                fs = Builder.BrokenForStatement(ifs);
                // innermostConditionBlock should never be null at this point.
                innermostConditionBlock.Add(Recognizer.LoopBreakStatement(ifs));
            }
            else
            {
                fs = Builder.ForStmt(ifs);
            }
            context.SetPrimaryOutput(fs);
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            fs.Initializer = ConvertStatement(ifs.Initializer);
            fs.Condition = ConvertExpression(ifs.Condition);
            fs.Increment = ConvertStatement(ifs.Increment);
            fs.Body = newBody;
            ShallowCopy = wasCopy;
            if (fs.Body.Statements.Count == 0) return null;
            if (!ShallowCopy &&
                ReferenceEquals(fs.Body, ifs.Body) &&
                ReferenceEquals(fs.Initializer, ifs.Initializer) &&
                ReferenceEquals(fs.Condition, ifs.Condition) &&
                ReferenceEquals(fs.Increment, ifs.Increment)) return ifs;
            context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
            return fs;
        }

        private void AnalyzeCondition(IExpression condition)
        {
            Set<IVariableDeclaration> oldVariablesInBody = variablesInBody;
            variablesInBody = variablesInConditions;
            ConvertExpression(condition);
            variablesInBody = oldVariablesInBody;
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            // innermostConditionBlock will be set when transforming the body.
            innermostConditionBlock = null;
            AnalyzeCondition(ics.Condition);
            IConditionStatement cs = Builder.CondStmt();
            context.SetPrimaryOutput(cs);
            cs.Condition = ics.Condition;
            cs.Then = ConvertBlock(ics.Then);
            if (ics.Else != null) cs.Else = ConvertBlock(ics.Else);
            if (cs.Then.Statements.Count == 0 && (cs.Else == null || cs.Else.Statements.Count == 0)) return null;
            // if innermostConditionBlock is null after transforming the body, then this is the innermost.
            // we cannot return the original ics in this case since innermostConditionBlock can be modified by parents.
            if (innermostConditionBlock == null)
                innermostConditionBlock = cs.Then.Statements;
            else if (!ShallowCopy && ReferenceEquals(cs.Condition, ics.Condition) && ReferenceEquals(cs.Then, ics.Then) && ReferenceEquals(cs.Else, ics.Else))
                return ics;
            context.InputAttributes.CopyObjectAttributesTo(ics, context.OutputAttributes, cs);
            return cs;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            variablesInBody.Add(ivre.Variable.Variable);
            return base.ConvertVariableRefExpr(ivre);
        }
    }
}
