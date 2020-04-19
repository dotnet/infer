// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Models
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class MethodInvoke : IModelExpression
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        // The factor or constraint method
        internal MethodInfo method;

        // The arguments
        internal List<IModelExpression> args = new List<IModelExpression>();

        // The return value (or null if void)
        internal IModelExpression returnValue;


        // The operator if this method was created from an operator
        internal Variable.Operator? op = null;

        // Attributes of the method invocation i.e. the factor or constraint.
        internal List<ICompilerAttribute> attributes = new List<ICompilerAttribute>();

        // The condition blocks this method is contained in
        private List<IStatementBlock> containers;

        internal List<IStatementBlock> Containers
        {
            get { return containers; }
        }

        // Provides global ordering for ModelBuilder
        internal readonly int timestamp;

        private static readonly GlobalCounter globalCounter = new GlobalCounter();

        internal static int GetTimestamp()
        {
            return globalCounter.GetNext();
        }

        internal MethodInvoke(MethodInfo method, params IModelExpression[] args)
            : this(StatementBlock.GetOpenBlocks(), method, args)
        {
        }

        internal MethodInvoke(IEnumerable<IStatementBlock> containers, MethodInfo method, params IModelExpression[] args)
        {
            this.timestamp = GetTimestamp();
            this.method = method;
            this.args.AddRange(args);
            this.containers = new List<IStatementBlock>(containers);
            foreach (IModelExpression arg in args)
            {
                if (ReferenceEquals(arg, null)) throw new ArgumentNullException();
                if (arg is Variable)
                {
                    Variable v = (Variable) arg;
                    if (v.IsObserved) continue;
                    foreach (ConditionBlock cb in v.GetContainers<ConditionBlock>())
                    {
                        if (!this.containers.Contains(cb))
                        {
                            throw new InvalidOperationException($"{arg} was created in condition {cb} and cannot be used outside.  " +
                                $"To give {arg} a conditional definition, use SetTo inside {cb} rather than assignment (=).  " +
                                $"If you are using GetCopyFor, make sure you call GetCopyFor outside of conflicting conditional statements.");
                        }
                    }
                }
            }
            foreach (ConditionBlock cb in StatementBlock.EnumerateBlocks<ConditionBlock>(containers))
            {
                cb.ConditionVariableUntyped.constraints.Add(this);
            }
        }

        /// <summary>
        /// The name of the method
        /// </summary>
        public string Name
        {
            get { return method.Name; }
        }

        /// <summary>
        /// The method arguments
        /// </summary>
        public List<IModelExpression> Arguments
        {
            get { return args; }
        }

        /// <summary>
        /// The expression the return value of the method will be assigned to.
        /// </summary>
        public IModelExpression ReturnValue
        {
            get { return returnValue; }
        }

        public void AddAttribute(ICompilerAttribute attr)
        {
            InferenceEngine.InvalidateAllEngines(this);
            attributes.Add(attr);
        }

        /// <summary>
        /// Inline method for adding an attribute to a method invoke.  This method
        /// returns the method invoke object, so that is can be used in an inline expression.
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        /// <returns>This object</returns>
        public MethodInvoke Attrib(ICompilerAttribute attr)
        {
            AddAttribute(attr);
            return this;
        }

        /// <summary>
        /// Get all attributes of this variable having type AttributeType.
        /// </summary>
        /// <typeparam name="AttributeType"></typeparam>
        /// <returns></returns>
        public IEnumerable<AttributeType> GetAttributes<AttributeType>() where AttributeType : ICompilerAttribute
        {
            // find the base variable
            foreach (ICompilerAttribute attr in attributes)
            {
                if (attr is AttributeType) yield return (AttributeType) attr;
            }
        }

        public IExpression GetExpression()
        {
            IExpression expr = GetMethodInvokeExpression();
            if (returnValue == null) return expr;
            expr = Builder.AssignExpr(returnValue.GetExpression(), expr);
            return expr;
        }

        /// <summary>
        /// True if the expression contains a loop index and all other variable references are givens.
        /// </summary>
        /// <returns></returns>
        internal bool CanBeInlined()
        {
            if (op == null) return false;
            bool hasLoopIndex = false;
            for (int i = 0; i < args.Count; i++)
            {
                if (args[i] is Variable<int>)
                {
                    Variable<int> v = (Variable<int>) args[i];
                    if (v.IsLoopIndex) hasLoopIndex = true;
                    else if (!v.IsObserved) return false;
                }
                else return false;
            }
            return hasLoopIndex;
        }

        internal IExpression GetMethodInvokeExpression(bool inline = false)
        {
            IExpression[] argExprs = new IExpression[args.Count];
            for (int i = 0; i < argExprs.Length; i++)
            {
                argExprs[i] = args[i].GetExpression();
            }
            if (inline || CanBeInlined())
            {
                if (op == Variable.Operator.Plus) return Builder.BinaryExpr(argExprs[0], BinaryOperator.Add, argExprs[1]);
                else if (op == Variable.Operator.Minus) return Builder.BinaryExpr(argExprs[0], BinaryOperator.Subtract, argExprs[1]);
                else if (op == Variable.Operator.LessThan) return Builder.BinaryExpr(argExprs[0], BinaryOperator.LessThan, argExprs[1]);
                else if (op == Variable.Operator.LessThanOrEqual) return Builder.BinaryExpr(argExprs[0], BinaryOperator.LessThanOrEqual, argExprs[1]);
                else if (op == Variable.Operator.GreaterThan) return Builder.BinaryExpr(argExprs[0], BinaryOperator.GreaterThan, argExprs[1]);
                else if (op == Variable.Operator.GreaterThanOrEqual) return Builder.BinaryExpr(argExprs[0], BinaryOperator.GreaterThanOrEqual, argExprs[1]);
                else if (op == Variable.Operator.Equal) return Builder.BinaryExpr(argExprs[0], BinaryOperator.ValueEquality, argExprs[1]);
                else if (op == Variable.Operator.NotEqual) return Builder.BinaryExpr(argExprs[0], BinaryOperator.ValueInequality, argExprs[1]);
            }
            IMethodInvokeExpression imie = null;
            if (method.IsGenericMethod && !method.ContainsGenericParameters)
            {
                imie = Builder.StaticGenericMethod(method, argExprs);
            }
            else
            {
                imie = Builder.StaticMethod(method, argExprs);
            }
            return imie;
        }


        public override string ToString()
        {
          StringBuilder sb = new StringBuilder(method.Name);
          sb.Append('(');
          bool isFirst = true;
          foreach (IModelExpression arg in args)
          {
            if (!isFirst)
              sb.Append(',');
            else
              isFirst = false;
            if(arg != null)
              sb.Append(arg.ToString());
          }
          sb.Append(')');
          return sb.ToString();
        }

        internal IEnumerable<IModelExpression> returnValueAndArgs()
        {
            if (returnValue != null) yield return returnValue;
            foreach (IModelExpression arg in args) yield return arg;
        }

        /// <summary>
        /// Get the set of ranges used as indices in the arguments of the MethodInvoke, that are not included in its ForEach containers.
        /// </summary>
        /// <returns></returns>
        internal Set<Range> GetLocalRangeSet()
        {
            Set<Range> ranges = new Set<Range>();
            foreach (IModelExpression arg in returnValueAndArgs()) ForEachRange(arg, ranges.Add);
            foreach (IStatementBlock b in containers)
            {
                if (b is HasRange)
                {
                    HasRange br = (HasRange) b;
                    ranges.Remove(br.Range);
                }
            }
            return ranges;
        }

        /// <summary>
        /// Get the set of ranges used as indices in the arguments of the MethodInvoke, that are not included in its ForEach containers.
        /// </summary>
        /// <returns></returns>
        internal List<Range> GetLocalRangeList()
        {
            List<Range> ranges = new List<Range>();
            foreach (IModelExpression arg in returnValueAndArgs())
            {
                ForEachRange(arg, delegate(Range r) { if (!ranges.Contains(r)) ranges.Add(r); });
            }
            foreach (IStatementBlock b in containers)
            {
                if (b is HasRange)
                {
                    HasRange br = (HasRange) b;
                    ranges.Remove(br.Range);
                }
            }
            return ranges;
        }

        internal static void ForEachRange(IModelExpression arg, Action<Range> action)
        {
            if (arg is Range)
            {
                action((Range) arg);
                return;
            }
            else if (arg is Variable)
            {
                Variable v = (Variable) arg;
                if (v.IsLoopIndex)
                {
                    action(v.loopRange);
                }
                if (v.IsArrayElement)
                {
                    ForEachRange(v.ArrayVariable, action);
                    // must add item indices after array's indices
                    foreach (IModelExpression expr in v.indices)
                    {
                        ForEachRange(expr, action);
                    }
                }
            }
        }

        /// <summary>
        /// Get a dictionary mapping all array indexer expressions (including sub-expressions) to a list of their Range indexes, in order.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        internal static Dictionary<IModelExpression, List<List<Range>>> GetRangeBrackets(IEnumerable<IModelExpression> args)
        {
            Dictionary<IModelExpression, List<List<Range>>> dict = new Dictionary<IModelExpression, List<List<Range>>>();
            foreach (IModelExpression arg in args)
            {
                List<List<Range>> brackets = GetRangeBrackets(arg, dict);
                dict[arg] = brackets;
            }
            return dict;
        }

        /// <summary>
        /// If arg is an array indexer expression, get a list of all Range indexes, in order.  Indexes that are not Ranges instead get their Ranges added to dict.
        /// </summary>
        /// <param name="arg"></param>
        /// <param name="dict"></param>
        /// <returns></returns>
        internal static List<List<Range>> GetRangeBrackets(IModelExpression arg, IDictionary<IModelExpression, List<List<Range>>> dict)
        {
            if (arg is Variable)
            {
                Variable v = (Variable) arg;
                if (v.IsArrayElement)
                {
                    List<List<Range>> brackets = GetRangeBrackets(v.ArrayVariable, dict);
                    List<Range> indices = new List<Range>();
                    // must add item indices after array's indices
                    foreach (IModelExpression expr in v.indices)
                    {
                        if (expr is Range) indices.Add((Range) expr);
                        else
                        {
                            List<List<Range>> argBrackets = GetRangeBrackets(expr, dict);
                            dict[expr] = argBrackets;
                        }
                    }
                    brackets.Add(indices);
                    return brackets;
                }
            }
            return new List<List<Range>>();
        }

        internal static int CompareRanges(IDictionary<IModelExpression, List<List<Range>>> dict, Range a, Range b)
        {
            foreach (List<List<Range>> brackets in dict.Values)
            {
                bool aInPreviousBracket = false;
                bool bInPreviousBracket = false;
                foreach (List<Range> bracket in brackets)
                {
                    bool aInThisBracket = false;
                    bool bInThisBracket = false;
                    foreach (Range range in bracket)
                    {
                        if (range == a) aInThisBracket = true;
                        if (range == b) bInThisBracket = true;
                    }
                    if (bInThisBracket && aInPreviousBracket && !bInPreviousBracket) return -1;
                    if (aInThisBracket && bInPreviousBracket && !aInPreviousBracket) return 1;
                    aInPreviousBracket = aInThisBracket;
                    bInPreviousBracket = bInThisBracket;
                }
            }
            return 0;
        }

        /// <summary>
        /// True if arg is indexed by at least the given ranges.
        /// </summary>
        /// <param name="arg"></param>
        /// <param name="ranges"></param>
        /// <returns></returns>
        internal static bool IsIndexedByAll(IModelExpression arg, ICollection<Range> ranges)
        {
            Set<Range> argRanges = new Set<Range>();
            ForEachRange(arg, argRanges.Add);
            foreach (Range r in ranges)
            {
                if (!argRanges.Contains(r)) return false;
            }
            return true;
        }

        /*internal string GetReturnValueName()
        {
                if (method == null) return "";
                if (op != null)
                {
                        return args[0].Name + " " + op + " " + args[1].Name;
                }
                string;
        }*/
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}