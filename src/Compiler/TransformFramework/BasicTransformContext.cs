// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    public class BasicTransformContext : StackContext, ICodeTransformContext
    {
        #region hidden for convenience

        protected List<ITypeDeclaration> typesToTransform = new List<ITypeDeclaration>();

        public List<ITypeDeclaration> TypesToTransform
        {
            get { return typesToTransform; }
        }

        // Following convenience method may be used by CSoft, but is not used by Infer.NET
        // Comment this out as part of the reorg that separates disassembly from the transform
        // chain
        //virtual public void AddTypeToTransform(Type t)
        //{
        //    t = DeclarationProvider.GetTopLevelType(t);
        //    ITypeDeclaration itd = DeclarationProvider.INSTANCE.GetTypeDeclaration(t, true);
        //    if (typesToTransform.Contains(itd)) return;
        //    typesToTransform.Add(itd);
        //}

        #endregion hidden for convenience

        /// <summary>
        /// Indicates if the association between input and output elements should be stored.
        /// </summary>
        public bool trackTransform = false;

        /// <summary>
        /// The output of each input element.  Collected only if trackTransform=true.  Used by DeclarationView.
        /// </summary>
        public List<KeyValuePair<object, TransformOutput>> outputOfElement = new List<KeyValuePair<object, TransformOutput>>();

        /// <summary>
        /// The output of each input element on the InputStack.  Updated only if trackTransform=true.  Used by DeclarationView.
        /// </summary>
        protected Stack<TransformOutput> OutputStack = new Stack<TransformOutput>();

        private AttributeRegistry<object, ICompilerAttribute> inputAttributes = new AttributeRegistry<object, ICompilerAttribute>(true);

        public AttributeRegistry<object, ICompilerAttribute> InputAttributes
        {
            get { return inputAttributes; }
            set { inputAttributes = value; }
        }

        public T GetAttribute<T>(object obj) where T : class, ICompilerAttribute
        {
            try
            {
                return InputAttributes.Get<T>(obj);
            }
            catch (Exception ex)
            {
                Error(ex.Message);
                return null;
            }
        }

        // IMPORTANT: input attributes and output attributes are currently the same!
        public AttributeRegistry<object, ICompilerAttribute> OutputAttributes
        {
            get { return inputAttributes; }
        }

        //  public AttributeRegistry<object, ICompilerAttribute> OutputAttributes = new AttributeRegistry<object, ICompilerAttribute>(true);


        protected TransformResults results = new TransformResults();

        public TransformResults Results
        {
            get { return results; }
            set { results = value; }
        }

        /// <summary>
        /// Opens the input object for transforming.
        /// </summary>
        /// <param name="inputItem">Input code element</param>
        protected virtual void Open(object inputItem)
        {
            TransformInfo ti = new TransformInfo(inputItem);
            InputStack.Add(ti);
        }

        protected virtual void OpenOutput()
        {
            if (trackTransform)
            {
                OutputStack.Push(new TransformOutput());
            }
        }

        /// <summary>
        /// Closes the current input object, indicating that its transformation has been completed.
        /// </summary>
        protected virtual void Close(object inputItem)
        {
            if (InputStack.Count == 0) throw new InvalidOperationException("Cannot close: stack is already empty.");
            TransformInfo ti = InputStack[InputStack.Count - 1];
            if (!object.ReferenceEquals(inputItem, ti.inputElement))
            {
                throw new InvalidOperationException("Cannot close: supplied object is not on top of the stack: " + inputItem + "," + ti.inputElement);
            }
            InputStack.RemoveAt(InputStack.Count - 1);
        }

        protected void CloseOutput(object inputItem)
        {
            if (trackTransform)
            {
                TransformOutput childOutput = OutputStack.Pop();
                if (childOutput != null)
                {
                    bool isShallow = (childOutput.outputElements.Count == 1 && ReferenceEquals(childOutput.outputElements[0], inputItem));
                    isShallow = false;
                    if (!isShallow)
                    {
                        outputOfElement.Add(new KeyValuePair<object, TransformOutput>(inputItem, childOutput));
                        if (OutputStack.Count > 0)
                        {
                            TransformOutput parentOutput = OutputStack.Pop();
                            if (parentOutput == null) parentOutput = new TransformOutput();
                            parentOutput.outputsOfChildren.Add(childOutput);
                            OutputStack.Push(parentOutput);
                        }
                    }
                }
            }
        }

        protected void AddOutput(object outputElement)
        {
            if (trackTransform)
            {
                TransformOutput output = OutputStack.Peek();
                if (output == null)
                {
                    //TransformInfo ti = InputStack[InputStack.Count - 1];
                    //if (ReferenceEquals(outputElement, ti.inputElement)) return;
                    OutputStack.Pop();
                    output = new TransformOutput();
                    OutputStack.Push(output);
                }
                // TODO: don't add dummy BlockStatements to output (they don't get TreeNodes anyway)
                output.outputElements.Add(outputElement);
            }
        }

        public override void AddStatementBeforeAncestorIndex(int ancInd, IStatement stmt, bool convertBeforeAdding = false)
        {
            AddOutput(stmt);
            base.AddStatementBeforeAncestorIndex(ancInd, stmt, convertBeforeAdding);
        }

        public override void AddStatementAfterAncestorIndex(int ancInd, IStatement statementToAdd, bool convertBeforeAdding = false)
        {
            AddOutput(statementToAdd);
            base.AddStatementAfterAncestorIndex(ancInd, statementToAdd, convertBeforeAdding);
        }

        public override void SetPrimaryOutput(object outputItem)
        {
            AddOutput(outputItem);
            base.SetPrimaryOutput(outputItem);
        }

        public virtual void OpenType(ITypeDeclaration itd)
        {
            Open(itd);
            OpenOutput();
        }

        public virtual void CloseType(ITypeDeclaration itd)
        {
            Close(itd);
            CloseOutput(itd);
        }

        public virtual void OpenMember(IMemberDeclaration imd)
        {
            Open(imd);
            OpenOutput();
        }

        public virtual void AddMember(IMemberDeclaration imd)
        {
            AddOutput(imd);
        }

        public virtual void CloseMember(IMemberDeclaration imd)
        {
            Close(imd);
            CloseOutput(imd);
        }

        public virtual void OpenStatement(IStatement istmt)
        {
            Open(istmt);
            OpenOutput();
        }

        public virtual void CloseStatement(IStatement istmt)
        {
            Close(istmt);
            CloseOutput(istmt);
        }

        // as an optimisation, we do not store the output for expressions.
        public virtual void OpenExpression(IExpression iexpr)
        {
            Open(iexpr);
        }

        public virtual void CloseExpression(IExpression iexpr)
        {
            Close(iexpr);
        }

        public void Warning(string msg)
        {
            Warning(msg, null);
        }

        public void Warning(string msg, Exception ex)
        {
            object inputElement = InputStack[InputStack.Count - 1].inputElement;
            object tag = FindAncestor<IStatement>();
            results.AddWarning(msg, ex, inputElement, tag);
        }

        public void Error(string msg)
        {
            Error(msg, null);
        }

        public void Error(string msg, Exception ex)
        {
            if (InputStack.Count == 0) results.AddError(msg, ex, "", "");
            else
            {
                object inputElement = InputStack[InputStack.Count - 1].inputElement;
                object tag = FindAncestor<IStatement>();
                if (inputElement == null) inputElement = tag;
                results.AddError(msg, ex, inputElement, tag);
            }
        }

        public void Error(string msg, Exception ex, object inputElement)
        {
            results.AddError(msg, ex, inputElement, inputElement);
        }

        public void FatalError(string msg)
        {
            throw new TransformFailedException(results, "Fatal error: " + msg + " [in " + GetContextString() + "]");
        }

        protected string GetContextString()
        {
            IMemberDeclaration imd = FindAncestor<IMemberDeclaration>();
            if (imd == null) return "";
            return imd.ToString();
        }
    }

    /// <summary>
    /// Class which maintains the transformation state of a code element
    /// </summary>
    public class TransformInfo
    {
        public object inputElement;
        internal List<AddAction> toAddBefore;
        internal List<AddAction> toAddAfterwards;

        /// <summary>
        /// Used by child statements to access the output of a container.  The full set of outputs is stored in a TransformOutput object on the OutputStack.
        /// </summary>
        protected object primaryOutput;

        public TransformInfo(object inputElement)
        {
            this.inputElement = inputElement;
        }

        public object PrimaryOutput
        {
            get { return primaryOutput; }
            set
            {
                if (primaryOutput != null) throw new InvalidOperationException("PrimaryOutput is already set for: " + inputElement);
                primaryOutput = value;
            }
        }

        public override string ToString()
        {
            return "TransInfo: " + inputElement;
        }

        internal void AddBefore(IStatement statementToAdd, bool convert)
        {
            if (toAddBefore == null) toAddBefore = new List<AddAction>();
            toAddBefore.Add(new AddAction() {Object = statementToAdd, DoConversion = convert});
        }

        internal void AddAfterwards(IStatement statementToAdd, bool convert)
        {
            if (toAddAfterwards == null) toAddAfterwards = new List<AddAction>();
            toAddAfterwards.Insert(0, new AddAction() {Object = statementToAdd, DoConversion = convert});
            //toAddAfterwards.Add(new AddAction() { Object=statementToAdd, DoConversion = convert });
        }

        internal void AddAfterwards(IEnumerable<IStatement> statementsToAdd, bool convert)
        {
            if (toAddAfterwards == null) toAddAfterwards = new List<AddAction>();
            int i = 0;
            foreach (IStatement statementToAdd in statementsToAdd)
            {
                toAddAfterwards.Insert(i++, new AddAction() {Object = statementToAdd, DoConversion = convert});
            }
            //toAddAfterwards.Add(new AddAction() { Object=statementToAdd, DoConversion = convert });
        }

        internal void Clear()
        {
            inputElement = null;
            primaryOutput = null;
            toAddBefore = null;
            toAddAfterwards = null;
        }
    }

    internal class AddAction
    {
        internal object Object { get; set; }
        internal bool DoConversion { get; set; }
    }

    /// <summary>
    /// Holds the output elements of a transformed code element.
    /// </summary>
    public class TransformOutput
    {
        public List<object> outputElements = new List<object>();
        public List<TransformOutput> outputsOfChildren = new List<TransformOutput>();

        public override string ToString()
        {
            if (outputElements.Count == 1) return "1 output";
            else return outputElements.Count + " outputs";
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}