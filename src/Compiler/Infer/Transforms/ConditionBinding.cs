// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    internal class ConditionBinding : ICloneable
    {
        public readonly IExpression lhs, rhs;
        protected static CodeBuilder Builder = CodeBuilder.Instance;
        //protected static CodeRecognizer Recognizer = CodeRecognizer.Instance;
        public ConditionBinding(IExpression lhs, IExpression rhs)
        {
            this.lhs = lhs;
            this.rhs = rhs;
        }

        public ConditionBinding(IExpression condition)
        {
            lhs = condition;
            rhs = Builder.LiteralExpr(true);
            if (condition is IBinaryExpression)
            {
                IBinaryExpression ibe = (IBinaryExpression) condition;
                if ((ibe.Operator == BinaryOperator.IdentityEquality) || (ibe.Operator == BinaryOperator.ValueEquality))
                {
                    lhs = ibe.Left;
                    rhs = ibe.Right;
                }
            }
            else if (condition is IUnaryExpression)
            {
                IUnaryExpression iue = (IUnaryExpression) condition;
                if (iue.Operator == UnaryOperator.BooleanNot)
                {
                    lhs = iue.Expression;
                    rhs = Builder.LiteralExpr(false);
                }
            }
        }

        public IExpression GetExpression()
        {
            if (rhs is ILiteralExpression && rhs.GetExpressionType().Equals(typeof (bool)))
            {
                bool value = (bool) ((ILiteralExpression) rhs).Value;
                if (value) return lhs;
                else return Builder.UnaryExpr(UnaryOperator.BooleanNot, lhs);
            }
            else
            {
                return Builder.BinaryExpr(lhs, BinaryOperator.ValueEquality, rhs);
            }
        }

        public ConditionBinding FlipCondition()
        {
            if (rhs.GetExpressionType().Equals(typeof (bool)) && (rhs is ILiteralExpression))
            {
                IExpression rhs2 = Builder.LiteralExpr(!(bool) ((ILiteralExpression) rhs).Value);
                return new ConditionBinding(lhs, rhs2);
            }
            else
            {
                // condition must have been ValueEquality
                IExpression lhs2 = Builder.BinaryExpr(lhs, BinaryOperator.ValueEquality, rhs);
                IExpression rhs2 = Builder.LiteralExpr(false);
                return new ConditionBinding(lhs2, rhs2);
            }
        }

        public override bool Equals(object obj)
        {
            ConditionBinding that = obj as ConditionBinding;
            if (that == null) return false;
            return lhs.Equals(that.lhs) && rhs.Equals(that.rhs);
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, lhs.GetHashCode());
            hash = Hash.Combine(hash, rhs.GetHashCode());
            return hash;
        }

        public override string ToString()
        {
            if (rhs is ILiteralExpression && rhs.GetExpressionType().Equals(typeof (bool)))
            {
                string lhsString = lhs.ToString();
                bool value = (bool) ((ILiteralExpression) rhs).Value;
                if (value)
                    return lhsString;
                else
                {
                    if (lhs is IBinaryExpression)
                        lhsString = "(" + lhsString + ")";
                    return "!" + lhsString;
                }
            }
            else
            {
                return lhs + "=" + rhs;
            }
        }

        public static Set<ConditionBinding> Copy(IEnumerable<ConditionBinding> bindings)
        {
            Set<ConditionBinding> result = new Set<ConditionBinding>();
            foreach (ConditionBinding binding in bindings)
            {
                result.Add((ConditionBinding) binding.Clone());
            }
            return result;
        }

        public static Dictionary<IExpression, IExpression> ToDictionary(IEnumerable<ConditionBinding> bindings, bool boolVarsOnly)
        {
            Dictionary<IExpression, IExpression> result = new Dictionary<IExpression, IExpression>();
            foreach (ConditionBinding binding in bindings)
            {
                if (boolVarsOnly)
                {
                    bool isSimple = !(binding.lhs is IBinaryExpression);
                    if (!isSimple) continue;
                    bool isBool = binding.lhs.GetExpressionType().Equals(typeof (bool));
                    if (!isBool) continue;
                }
                result[binding.lhs] = binding.rhs;
            }
            return result;
        }

        public object Clone()
        {
            return new ConditionBinding(lhs, rhs);
        }
    }
}