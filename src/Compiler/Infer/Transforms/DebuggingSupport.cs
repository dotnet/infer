// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using System;
    using System.Collections.Generic;
    using System.Text;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel;
    using Microsoft.ML.Probabilistic;

    /// <summary>
    /// Helper class for providing debugging functionality.
    /// </summary>
    internal class DebuggingSupport
    {
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Gets an expression for the string form of the supplied expression.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static IExpression GetExpressionTextExpression(IExpression expr)
        {
            string msgText = expr.ToString();
            if (msgText.StartsWith("this."))
                msgText = msgText.Substring(5);
            string comma = CSharpWriter.InsertSpaceAfterComma ? ", " : ",";
            if (expr is IArrayIndexerExpression)
            {
                List<IList<IExpression>> indices = CodeRecognizer.Instance.GetIndices(expr);
                List<IExpression> indexExprs = new List<IExpression>();
                bool foundVariableIndex = false;
                foreach (IList<IExpression> bracket in indices)
                {
                    StringBuilder bracketSb = new StringBuilder("[");
                    StringBuilder formatSb = new StringBuilder("[");
                    for (int i = 0; i < bracket.Count; i++)
                    {
                        if (i > 0)
                        {
                            bracketSb.Append(comma);
                            formatSb.Append(",");
                        }
                        IExpression indexExpr = bracket[i];
                        bracketSb.Append(indexExpr);
                        if (indexExpr is ILiteralExpression)
                        {
                            formatSb.Append(indexExpr);
                        }
                        else
                        {
                            foundVariableIndex = true;
                            int formatIndex = indexExprs.Count;
                            formatSb.Append("{");
                            formatSb.Append(formatIndex);
                            formatSb.Append("}");
                            indexExprs.Add(indexExpr);
                        }
                    }
                    bracketSb.Append("]");
                    formatSb.Append("]");
                    msgText = msgText.Replace(bracketSb.ToString(), formatSb.ToString());
                }
                if (foundVariableIndex)
                {
                    IArrayCreateExpression indexArray = Builder.ArrayCreateExpr(typeof (object), Builder.LiteralExpr(indexExprs.Count));
                    indexArray.Initializer = Builder.BlockExpr();
                    indexArray.Initializer.Expressions.AddRange(indexExprs);
                    return Builder.StaticMethod(new Func<string, object[], string>(string.Format), Builder.LiteralExpr(msgText), indexArray);
                }
            }
            IExpression textExpr = Builder.LiteralExpr(msgText);
            return textExpr;
        }

        /// <summary>
        /// Name of the generated event for message updates.
        /// </summary>
        internal const string MessageEventName = "MessageUpdated";

        /// <summary>
        /// Tries to dynamically add or remove an event handler to a generated algorithm instance.
        /// </summary>
        /// <param name="ca">The generated algorithm instance</param>
        /// <param name="d">The event handler to add or remove</param>
        /// <param name="add">If true will add, otherwise will remove</param>
        /// <returns>True if the event handler was added or removed successfully</returns>
        internal static bool TryAddRemoveEventListenerDynamic(IGeneratedAlgorithm ca, EventHandler<MessageUpdatedEventArgs> d, bool add)
        {
            var eventInfo = ca.GetType().GetEvent(MessageEventName);
            if (eventInfo == null)
                return false;
            if (add)
            {
                eventInfo.AddEventHandler(ca, d);
            }
            else
            {
                eventInfo.RemoveEventHandler(ca, d);
            }
            return true;
        }
    }
}