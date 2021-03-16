// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// A range of values from 0 to N-1. The size N may be an integer expression or constant.
    /// </summary>
    public class Range : IModelExpression, IStatementBlock
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Model expression for size of the range
        /// </summary>
        public IModelExpression<int> Size { get; private set; }

        /// <summary>
        /// Range from which this range was cloned, or null if none.
        /// </summary>
        public Range Parent { get; private set; }

        /// <summary>
        /// Name
        /// </summary>
        protected string name;

        /// <summary>
        /// Name of the range
        /// </summary>
        public string Name
        {
            get { return name; }
            set { name = value; }
        }

        /// <summary>
        /// Name used in generated code
        /// </summary>
        protected string nameInGeneratedCode;

        /// <summary>
        /// Name used in generated code
        /// </summary>
        internal string NameInGeneratedCode
        {
            get
            {
                if (nameInGeneratedCode == null) nameInGeneratedCode = CodeBuilder.MakeValid(Name);
                return nameInGeneratedCode;
            }
        }

        string IModelExpression.Name
        {
            get { return NameInGeneratedCode; }
        }

        /// <summary>
        /// The attributes associated with this Range.
        /// </summary>
        protected List<ICompilerAttribute> attributes = new List<ICompilerAttribute>();

        /// <summary>
        /// Inline method for adding an attribute to a range.  This method
        /// returns the range object, so that is can be used in an inline expression.
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        /// <returns>The range object</returns>
        public Range Attrib(ICompilerAttribute attr)
        {
            AddAttribute(attr);
            return this;
        }

        /// <summary>
        /// Adds an attribute to this range.  Attributes can be used
        /// to modify how inference is performed on the range.
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        public void AddAttribute(ICompilerAttribute attr)
        {
            InferenceEngine.InvalidateAllEngines(this);
            attributes.Add(attr);
        }

        /// <summary>
        /// Get all attributes of this range having type AttributeType.
        /// </summary>
        /// <typeparam name="AttributeType"></typeparam>
        /// <returns></returns>
        public IEnumerable<AttributeType> GetAttributes<AttributeType>() where AttributeType : ICompilerAttribute
        {
            foreach (ICompilerAttribute attr in attributes)
            {
                if (attr is AttributeType attributeType) yield return attributeType;
            }
        }

        /// <summary>
        /// Global counter used to generate variable names.
        /// </summary>
        private static readonly GlobalCounter globalCounter = new GlobalCounter();

        /// <summary>
        /// Constructs a range containing values from 0 to N-1.
        /// </summary>
        /// <param name="N">The number of elements in the range, including zero.</param>
        public Range(int N)
            : this(Variable.Constant(N))
        {
        }

        /// <summary>
        /// Constructs a range whose size is given by an integer-value expression.
        /// </summary>
        /// <param name="size">An expression giving the size of the range</param>
        public Range(IModelExpression<int> size)
        {
            this.name = $"index{globalCounter.GetNext()}";
            this.Size = size;
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="parent"></param>
        protected Range(Range parent)
            : this(parent.Size)
        {
            this.Parent = parent;
        }

        /// <summary>
        /// Create a copy of a range.  The copy can be used to index the same arrays as the original range.
        /// </summary>
        /// <returns></returns>
        public Range Clone()
        {
            return new Range(this);
        }

        /// <summary>
        /// Returns the size of the range as an integer.  This will fail if the size is not a constant,
        /// for example, if it is a Given value.
        /// </summary>
        public int SizeAsInt
        {
            get
            {
                if (Size is Variable<int> sizeVar)
                {
                    if (!(sizeVar.IsObserved && sizeVar.IsReadOnly))
                        throw new InvalidOperationException("The Range does not have constant size.  To use SizeAsInt, set IsReadOnly=true on the range size.");
                    return sizeVar.ObservedValue;
                }
                else throw new InvalidOperationException("The Range does not have constant size.  Set IsReadOnly=true on the range size.");
            }
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return Name;
            //return "Range " + Name + " (Size=" + Size + ")";
        }

        /// <summary>
        /// Inline method to name a range
        /// </summary>
        /// <param name="name">Name for the range</param>
        /// <returns>this</returns>
        public Range Named(string name)
        {
            this.name = name;
            return this;
        }

        internal IExpression GetSizeExpression()
        {
            return Size.GetExpression();
        }

        private IVariableDeclaration index;

        internal IVariableDeclaration GetIndexDeclaration()
        {
            if (index == null) index = Builder.VarDecl(NameInGeneratedCode, typeof(int));
            return index;
        }

        /// <summary>
        /// Gets the expression for the index variable
        /// </summary>
        /// <returns></returns>
        public IExpression GetExpression()
        {
            return Builder.VarRefExpr(GetIndexDeclaration());
        }


        internal static string ToString(IList<Range> ranges)
        {
            StringBuilder sb = new StringBuilder("[");
            foreach (Range r in ranges)
            {
                if (sb.Length > 1) sb.Append(",");
                sb.Append(r.Name);
            }
            sb.Append("]");
            return sb.ToString();
        }

        internal Range GetRoot()
        {
            Range root = this;
            while (root.Parent != null) root = root.Parent;
            return root;
        }

        /// <summary>
        /// Get an expression that evaluates to true when this loop counter is increasing in the currently executing loop.
        /// </summary>
        /// <returns></returns>
        public Variable<bool> IsIncreasing()
        {
            Variable<bool> v = new Variable<bool>();
            v.SetTo(new Func<int, bool>(Factors.InferNet.IsIncreasing).Method, this);
            v.Inline = true;
            return v;
        }

        private static Range ReplaceExpressions(Range r, Dictionary<IModelExpression, IModelExpression> replacements)
        {
            IModelExpression<int> newSize = (IModelExpression<int>)ReplaceExpressions(r.Size, replacements);
            if (ReferenceEquals(newSize, r.Size))
                return r;
            Range newRange = new Range(newSize);
            newRange.Parent = r;
            replacements.Add(r, newRange);
            return newRange;
        }

        private static IModelExpression ReplaceExpressions(IModelExpression expr, Dictionary<IModelExpression, IModelExpression> replacements)
        {
            if (replacements.ContainsKey(expr)) return replacements[expr];
            if (expr is Range range)
            {
                return ReplaceExpressions(range, replacements);
            }
            else if (expr is Variable v)
            {
                if (v.IsArrayElement)
                {
                    bool changed = false;
                    IVariableArray newArray = (IVariableArray)ReplaceExpressions(v.ArrayVariable, replacements);
                    if (!ReferenceEquals(newArray, v.ArrayVariable)) changed = true;
                    IModelExpression[] newIndices = new IModelExpression[v.indices.Count];
                    for (int i = 0; i < newIndices.Length; i++)
                    {
                        newIndices[i] = ReplaceExpressions(v.indices[i], replacements);
                        if (!ReferenceEquals(newIndices[i], v.indices[i])) changed = true;
                    }
                    if (changed)
                        return
                            (IModelExpression)
                            Invoker.InvokeMember(newArray.GetType(), "get_Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod, newArray, newIndices);
                }
            }
            return expr;
        }

        /// <summary>
        /// Construct a new Range in which all subranges and size expressions have been replaced according to given Dictionaries.
        /// </summary>
        /// <param name="rangeReplacements"></param>
        /// <param name="expressionReplacements">Modified on exit to contain newly created ranges</param>
        /// <returns></returns>
        internal Range Replace(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements)
        {
            if (rangeReplacements.ContainsKey(this)) return rangeReplacements[this];
            return ReplaceExpressions(this, expressionReplacements);
        }

        /// <summary>
        /// True if index is compatible with this range
        /// </summary>
        /// <param name="index">Index expression</param>
        /// <returns></returns>
        /// <exclude/>
        internal bool IsCompatibleWith(IModelExpression index)
        {
            if (index is Range range) return (range.GetRoot() == GetRoot());
            else if (index is Variable indexVar)
            {
                Range valueRange = indexVar.GetValueRange(false);
                if (valueRange == null) return true;
                return IsCompatibleWith(valueRange);
            }
            else
            {
                return true;
            }
        }

        /// <summary>
        /// Throws an exception if an index expression is not valid for subscripting an array.
        /// </summary>
        /// <param name="index">Index expression</param>
        /// <param name="array">Array that the expression is indexing</param>
        /// <exclude/>
        internal void CheckCompatible(IModelExpression index, IVariableArray array)
        {
            if (IsCompatibleWith(index)) return;
            string message = StringUtil.TypeToString(array.GetType()) + " " + array + " cannot be indexed by " + index + ".";
            if (index is Range)
            {
                string constructorName = "the constructor";
                message += " Perhaps you omitted " + index + " as an argument to " + constructorName + "?";
            }
            throw new ArgumentException(message, nameof(index));
        }

        /// <summary>
        /// Throws an exception if two index expression collections do not contain the same elements (regardless of order).
        /// </summary>
        /// <param name="set1">First set of index expressions</param>
        /// <param name="set2">Second set of index expressions</param>
        /// <exclude/>
        internal static void CheckCompatible(ICollection<IModelExpression> set1, ICollection<IModelExpression> set2)
        {
            if (set2.Count == 0)
            {
                if (set1.Count > 0)
                    throw new ArgumentException("The right-hand side is missing .ForEach(" + StringUtil.CollectionToString(set1, ",") + ")");
            }
            foreach (IModelExpression expr in set1)
            {
                if (!set2.Contains(expr))
                    throw new ArgumentException("The right-hand side indices " + Util.CollectionToString(set2) + " do not include the range '" + expr +
                                                "'.  Try adding .ForEach(" + expr + ")");
            }
            foreach (IModelExpression expr in set2)
            {
                if (!set1.Contains(expr))
                    throw new ArgumentException("The left-hand side indices " + Util.CollectionToString(set1) + " do not include the range '" + expr +
                                                "', which appears on the right-hand side (perhaps implicitly by an open ForEach block).");
            }
        }

        #region IStatementBlock Members

        /// <summary>
        /// Get 'for statement' for iterating over the range.
        /// </summary>
        /// <param name="innerBlock"></param>
        /// <returns></returns>
        internal IStatement GetStatement(out IList<IStatement> innerBlock)
        {
            IForStatement fs = Builder.ForStmt(GetIndexDeclaration(), GetSizeExpression());
            innerBlock = fs.Body.Statements;
            return fs;
        }

        IStatement IStatementBlock.GetStatement(out IList<IStatement> innerBlock)
        {
            return GetStatement(out innerBlock);
        }

        #endregion
    }
}