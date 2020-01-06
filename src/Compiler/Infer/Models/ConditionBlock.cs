// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Runtime.Serialization;

namespace Microsoft.ML.Probabilistic.Models
{
    internal interface IStatementBlock
    {
        /// <summary>
        /// Get a statement for the entire block, and a pointer to its body.
        /// </summary>
        /// <param name="innerBlock">On return, a pointer to the body of the block.</param>
        /// <returns></returns>
        IStatement GetStatement(out IList<IStatement> innerBlock);
    }

    /// <summary>
    /// Thrown when an empty block is closed.
    /// </summary>
    public class EmptyBlockException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EmptyBlockException"/> class.
        /// </summary>
        public EmptyBlockException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="EmptyBlockException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public EmptyBlockException(string message)
            : base(message)
        {
        }

        // This constructor is needed for serialization.
        protected EmptyBlockException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }

    /// <summary>
    /// Abstract base class for statement blocks
    /// </summary>
    public abstract class StatementBlock : IDisposable, IStatementBlock
    {
        /// <summary>
        /// A list of currently open blocks.  This is a thread-specific static variable and
        /// will have a different value for each thread.
        /// </summary>
        [ThreadStatic] private static List<IStatementBlock> openBlocks; // note cannot initalise thread-static variables here since it will only work in one thread

        internal static List<IStatementBlock> GetOpenBlocks()
        {
            if (openBlocks == null) openBlocks = new List<IStatementBlock>();
            return openBlocks;
        }

        internal static List<T> GetOpenBlocks<T>()
        {
            List<T> list = new List<T>();
            List<IStatementBlock> blocks = GetOpenBlocks();
            foreach (IStatementBlock sb in blocks)
            {
                if (sb is T) list.Add((T) sb);
            }
            return list;
        }

        internal static IEnumerable<T> EnumerateOpenBlocks<T>()
        {
            return EnumerateBlocks<T>(GetOpenBlocks());
        }

        internal static IEnumerable<T> EnumerateBlocks<T>(IEnumerable<IStatementBlock> blocks)
        {
            foreach (IStatementBlock sb in blocks)
            {
                if (sb is T) yield return (T) sb;
            }
        }

        /// <summary>
        /// Adds this block to a thread-specific list of open blocks.  
        /// </summary>
        internal virtual void OpenBlock()
        {
            GetOpenBlocks().Add(this);
        }

        /// <summary>
        /// Removes this block from a thread-specific list of open blocks.  
        /// If this block is not the final element of the list, gives an error.
        /// </summary>
        public void CloseBlock()
        {
            List<IStatementBlock> blocks = GetOpenBlocks();
            int k = blocks.IndexOf(this);
            if (k == -1)
            {
                throw new InvalidOperationException("Cannot close a block that is not open.");
            }
            if (k != blocks.Count - 1)
            {
                throw new InvalidOperationException("Blocks must be closed in the reverse order that they were opened.");
            }
            blocks.Remove(this);
        }

        /// <summary>
        /// Close blocks in order to recover from exceptions
        /// </summary>
        internal static void CloseAllBlocks()
        {
            List<IStatementBlock> blocks = new List<IStatementBlock>(StatementBlock.GetOpenBlocks());
            blocks.Reverse();
            foreach (StatementBlock block in blocks) block.CloseBlock();
        }

        /// <summary>
        /// Causes CloseBlock() to be called, so that this class can be used as the argument of a using() statement.
        /// </summary>
        /// <exclude/>
        public void Dispose()
        {
            CloseBlock();
        }

        /// <summary>
        /// Get a statement for the entire block, and a pointer to its body.
        /// </summary>
        /// <param name="innerBlock">On return, a pointer to the body of the block.</param>
        /// <returns></returns>
        internal abstract IStatement GetStatement(out IList<IStatement> innerBlock);

        IStatement IStatementBlock.GetStatement(out IList<IStatement> innerBlock)
        {
            return GetStatement(out innerBlock);
        }
    }

    /// <summary>
    /// Indicates that a StatementBlock has an associated range that it loops over.
    /// </summary>
    public interface HasRange
    {
        /// <summary>
        /// The Range being looped over.
        /// </summary>
        Range Range { get; }
    }

    /// <summary>
    /// 'For each' block
    /// </summary>
    public class ForEachBlock : StatementBlock, HasRange
    {
        /// <summary>
        /// Range associated with the 'for each' block
        /// </summary>
        protected Range range;

        /// <summary>
        /// Range associated with the 'for each' block
        /// </summary>
        public Range Range
        {
            get { return range; }
        }

        /// <summary>
        /// The index variable associated with the range
        /// </summary>
        public Variable<int> Index { get; private set; }

        /// <summary>
        /// Constructs 'for each' block from a range
        /// </summary>
        /// <param name="range">The range</param>
        public ForEachBlock(Range range)
        {
            this.range = range;
            OpenBlock();
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return "ForEach(" + range + ")";
        }

        internal static void CheckRangeCanBeOpened(Range range)
        {
            // check that all ranges in Range.Size are already opened.
            Set<Range> openRanges = new Set<Range>();
            foreach (HasRange fb in EnumerateOpenBlocks<HasRange>())
            {
                openRanges.Add(fb.Range);
            }
            if (openRanges.Contains(range))
            {
                throw new InvalidOperationException("Range '" + range + "' is already open in a ForEach or Switch block");
            }
            Models.MethodInvoke.ForEachRange(range.Size,
                                             delegate(Range r)
                                                 {
                                                     if (!openRanges.Contains(r))
                                                         throw new InvalidOperationException("Range '" + range + "' depends on range '" + r + "', but range '" + r +
                                                                                             "' is not open in a ForEach block.  Insert 'Variable.ForEach(" + r +
                                                                                             ")' around 'Variable.ForEach(" + range + ")'.");
                                                 });
        }

        /// <summary>
        /// Adds this block to a thread-specific list of open blocks.
        /// </summary>
        internal override void OpenBlock()
        {
            CheckRangeCanBeOpened(range);
            Index = new Variable<int>(range); // Needs to be here to prevent error when creating grid with .Index syntax

            base.OpenBlock();
        }

        /// <summary>
        /// Get a statement for the entire block, and a pointer to its body.
        /// </summary>
        /// <param name="innerBlock">On return, a pointer to the body of the block.</param>
        /// <returns></returns>
        internal override IStatement GetStatement(out IList<IStatement> innerBlock)
        {
            return range.GetStatement(out innerBlock);
        }
    }

    /// <summary>
    /// 'Repeat' block
    /// </summary>
    public class RepeatBlock : StatementBlock
    {
        private Variable<double> countVar;

        /// <summary>
        /// The variable that indicates the (possibly fractional) number of repeats. 
        /// </summary>
        public Variable<double> Count
        {
            get { return countVar; }
        }

        /// <summary>
        /// Constructs 'for each' block from a range
        /// </summary>
        /// <param name="count"></param>
        public RepeatBlock(Variable<double> count)
        {
            this.countVar = count;
            OpenBlock();
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return "Repeat(" + countVar + ")";
        }

        /// <summary>
        /// Get a statement for the entire block, and a pointer to its body.
        /// </summary>
        /// <param name="innerBlock">On return, a pointer to the body of the block.</param>
        /// <returns></returns>
        internal override IStatement GetStatement(out IList<IStatement> innerBlock)
        {
            IRepeatStatement rs = CodeBuilder.Instance.RepeatStmt(countVar.GetExpression());
            innerBlock = rs.Body.Statements;
            return rs;
        }
    }

    /// <summary>
    /// Base class for condition blocks
    /// </summary>
    public abstract class ConditionBlock : StatementBlock
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Adds this block to a thread-specific list of open blocks.
        /// </summary>
        internal override void OpenBlock()
        {
            foreach (ConditionBlock cb in EnumerateOpenBlocks<ConditionBlock>())
            {
                if (cb.ConditionVariableUntyped == ConditionVariableUntyped)
                {
                    throw new InvalidOperationException("Variable '" + ConditionVariableUntyped + "' is already being conditioned on.");
                }
            }
            base.OpenBlock();
        }

        internal static ConditionBlock GetConditionBlock(Variable conditionVar)
        {
            foreach (ConditionBlock cb in EnumerateOpenBlocks<ConditionBlock>())
            {
                if (cb.ConditionVariableUntyped == conditionVar) return cb;
            }
            return null;
        }

        internal abstract IExpression GetConditionExpression();

        /// <summary>
        /// The condition variable for this condition block.
        /// </summary>
        public abstract Variable ConditionVariableUntyped { get; }

        /// <summary>
        /// Gets a statement for the entire block, and a pointer to its body.
        /// </summary>
        /// <param name="innerBlock">On return, a pointer to the body of the block.</param>
        /// <returns></returns>
        internal override IStatement GetStatement(out IList<IStatement> innerBlock)
        {
            IConditionStatement cs = Builder.CondStmt();
            cs.Condition = GetConditionExpression();
            cs.Then = Builder.BlockStmt();
            innerBlock = cs.Then.Statements;
            return cs;
        }
    }

    /// <summary>
    /// Represents a conditional block in a model definition.  Anything defined inside 
    /// the block is placed inside a gate, whose condition is the condition of the block.
    /// </summary>
    public class ConditionBlock<T> : ConditionBlock
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        private readonly Variable<T> conditionVariable;
        private readonly T conditionValue;

        internal ConditionBlock(Variable<T> conditionVariable, T conditionValue)
            : this(conditionVariable, conditionValue, true)
        {
        }

        internal ConditionBlock(Variable<T> conditionVariable, T conditionValue, bool openBlock)
        {
            // check that all ranges in the conditionVariable are already opened.
            Set<Range> openRanges = new Set<Range>();
            foreach (HasRange fb in EnumerateOpenBlocks<HasRange>())
            {
                openRanges.Add(fb.Range);
            }
            Models.MethodInvoke.ForEachRange(conditionVariable,
                                             delegate(Range r)
                                                 {
                                                     if (!openRanges.Contains(r))
                                                         throw new InvalidOperationException(conditionVariable + " depends on range '" + r + "', but range '" + r +
                                                                                             "' is not open in a ForEach block.  Insert 'Variable.ForEach(" + r +
                                                                                             ")' around this block.");
                                                 });
            this.conditionVariable = conditionVariable;
            this.conditionValue = conditionValue;
            if (openBlock) OpenBlock();
        }

        /// <summary>
        /// The random variable which controls when this IfBlock is active.
        /// </summary>
        public Variable<T> ConditionVariable
        {
            get { return conditionVariable; }
        }

        /// <summary>
        /// The value of the condition variable which switches on this IfBlock.
        /// </summary>
        public T ConditionValue
        {
            get { return conditionValue; }
        }

        /// <summary>
        /// Equals override
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        /// <exclude/>
        public override bool Equals(object obj)
        {
            ConditionBlock<T> cb = obj as ConditionBlock<T>;
            if (cb == null) return false;
            if (!ReferenceEquals(conditionVariable, cb.conditionVariable)) return false;
            return conditionValue.Equals(cb.conditionValue);
        }

        /// <summary>
        /// Hash code override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int hash = conditionVariable.GetHashCode() + conditionValue.GetHashCode();
            return hash;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return GetType().Name + "(" + GetConditionExpression().ToString() + ")";
        }

        internal override IExpression GetConditionExpression()
        {
            return Builder.BinaryExpr(conditionVariable.GetExpression(), BinaryOperator.ValueEquality, Builder.LiteralExpr(conditionValue));
        }

        /// <summary>
        /// The condition variable for this condition block.
        /// </summary>
        public override Variable ConditionVariableUntyped
        {
            get { return conditionVariable; }
        }
    }

    /// <summary>
    /// An If block is a condition block with a binary condition.
    /// </summary>
    public class IfBlock : ConditionBlock<bool>
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        internal IfBlock(Variable<bool> conditionVariable, bool value)
            : base(conditionVariable, value)
        {
        }

        internal override IExpression GetConditionExpression()
        {
            IExpression expr = ConditionVariable.GetExpression();
            if (ConditionValue) return expr;
            else return Builder.NotExpr(expr);
        }
    }

    /// <summary>
    /// A case block is a condition block with a condition of the form (i==value) for integer i.
    /// </summary>
    public class CaseBlock : ConditionBlock<int>
    {
        internal CaseBlock(Variable<int> conditionVariable, int value)
            : base(conditionVariable, value)
        {
        }
    }

    /// <summary>
    /// A switch block is a condition block which acts like multiple case blocks ranging over the values
    /// of the integer condition variable.
    /// </summary>
    public class SwitchBlock : ConditionBlock<int>, HasRange
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        private Range range;

        internal SwitchBlock(Variable<int> conditionVariable, Range range)
            : base(conditionVariable, -1, false)
        {
            this.range = range;
            OpenBlock();
        }

        /// <summary>
        /// Adds this block to a thread-specific list of open blocks.
        /// </summary>
        internal override void OpenBlock()
        {
            ForEachBlock.CheckRangeCanBeOpened(range);
            base.OpenBlock();
        }

        /// <summary>
        /// Get switch block's range
        /// </summary>
        public Range Range
        {
            get { return range; }
        }

        internal override IExpression GetConditionExpression()
        {
            return Builder.BinaryExpr(ConditionVariable.GetExpression(), BinaryOperator.ValueEquality, range.GetExpression());
        }

        /// <summary>
        /// Gets a statement for the entire block, and a pointer to its body.
        /// </summary>
        /// <param name="innerBlock">On return, a pointer to the body of the block.</param>
        /// <returns></returns>
        internal override IStatement GetStatement(out IList<IStatement> innerBlock)
        {
            if (ConditionVariable.IsObserved)
            {
                innerBlock = null;
                return null;
            }
            IForStatement ifs = Builder.ForStmt(range.GetIndexDeclaration(), range.GetSizeExpression());
            ifs.Body.Statements.Add(base.GetStatement(out innerBlock));
            return ifs;
        }
    }
}