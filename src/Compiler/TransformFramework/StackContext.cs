// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// A transformation context whose state at a point in the transformation is given by a stack.
    /// </summary>
    public class StackContext
    {
        /// <summary>
        /// Stack holding transformation info for elements on the path down the input tree to the 
        /// input cursor (the location currently being transformed).
        /// </summary>
        public List<TransformInfo> InputStack = new List<TransformInfo>();

        /// <summary>
        /// The depth of the current point in the transformation
        /// </summary>
        public int Depth
        {
            get { return InputStack.Count; }
        }

        # region Methods for finding ancestors

        /// <summary>
        /// Finds the closest ancestor of the specified type (or assignable to that type) in the input path.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T FindAncestor<T>() where T : class
        {
            int k = FindAncestorIndex<T>();
            if (k == -1) return null;
            return (T) InputStack[k].inputElement;
            //return (T)InputStack.FindLast(delegate(object obj) { return ((KeyValuePair<object,object>)obj). is T; });
        }

        /// <summary>
        /// Finds the closest ancestor of the specified type (or assignable to that type) in the input path, excluding the current item.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T FindAncestorNotSelf<T>() where T : class
        {
            int k = FindAncestorNotSelfIndex<T>();
            if (k == -1) return null;
            return (T) InputStack[k].inputElement;
            //return (T)InputStack.FindLast(delegate(object obj) { return ((KeyValuePair<object,object>)obj). is T; });
        }

        /// <summary>
        /// Finds the index of the closest ancestor of the specified type (or assignable to that type) in the input path, or -1 if none.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public int FindAncestorIndex<T>()
        {
            for (int i = InputStack.Count - 1; i >= 0; i--)
            {
                TransformInfo ti = InputStack[i];
                if (ti.inputElement is T) return i;
            }
            return -1;
        }

        /// <summary>
        /// Finds the index of the closest ancestor of the specified type (or assignable to that type) in the input path.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public int FindAncestorNotSelfIndex<T>()
        {
            for (int i = InputStack.Count - 2; i >= 0; i--)
            {
                TransformInfo ti = InputStack[i];
                if (ti.inputElement is T) return i;
            }
            return -1;
        }

        /// <summary>
        /// Finds the index of the highest ancestor of the specified type (or assignable to that type) in the input path.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public int FindTopAncestorIndex<T>()
        {
            for (int i = 0; i < InputStack.Count; i++)
            {
                TransformInfo ti = InputStack[i];
                if (ti.inputElement is T) return i;
            }
            return -1;
        }

        /// <summary>
        /// Returns the ancestor object at the specified index in the transformation stack.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public object GetAncestor(int index)
        {
            return InputStack[index].inputElement;
        }

        /// <summary>
        /// Returns the index in the transformation stack of the specified ancestor object,
        /// or -1 if the object is not an ancestor.
        /// </summary>
        /// <param name="ancestor"></param>
        /// <returns></returns>
        public int GetAncestorIndex(object ancestor)
        {
            for (int i = 0; i < InputStack.Count; i++) if (InputStack[i].inputElement == ancestor) return i;
            return -1;
        }

        /// <summary>
        /// Finds all ancestors assignable to type T, in high-to-low order.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public List<T> FindAncestors<T>()
        {
            List<T> list = new List<T>();
            foreach (TransformInfo ti in InputStack) if (ti.inputElement is T ancestor) list.Add(ancestor);
            return list;
        }

        /// <summary>
        /// Finds all ancestors assignable to type T, in high-to-low order, up to and *excluding* the specified ancestor index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public List<T> FindAncestors<T>(int ancIndex)
        {
            List<T> list = new List<T>();
            for (int i = 0; i < ancIndex; i++)
            {
                TransformInfo ti = InputStack[i];
                if (ti.inputElement is T ancestor) list.Add(ancestor);
            }
            return list;
        }

        /// <summary>
        /// Finds all ancestors assignable to type T, in high-to-low order, whose index is greater than the specified ancestor index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public List<T> FindAncestorsBelow<T>(int ancIndex)
        {
            List<T> list = new List<T>();
            for (int i = ancIndex + 1; i < InputStack.Count; i++)
            {
                TransformInfo ti = InputStack[i];
                if (ti.inputElement is T ancestor) list.Add(ancestor);
            }
            return list;
        }

        /// <summary>
        /// Finds the highest ancestor of the specified type (or assignable to that type) in the input path.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T FindTopAncestor<T>()
        {
            for (int i = 0; i < InputStack.Count; i++)
            {
                TransformInfo ti = InputStack[i];
                if (ti.inputElement is T ancestor) return ancestor;
            }
            return default(T);
        }

        # endregion Methods for finding ancestors

        # region Methods for finding output

        /// <summary>
        /// Returns the transformed output for the closest ancestor of the specified type.
        /// </summary>
        /// <typeparam name="TAnc"></typeparam>
        /// <typeparam name="TTransformed"></typeparam>
        /// <returns></returns>
        public TTransformed FindOutputForAncestor<TAnc, TTransformed>()
        {
            int i = FindAncestorIndex<TAnc>();
            return GetOutputForAncestorIndex<TTransformed>(i);
        }

        /// <summary>
        /// Returns the transformed output for the ancestor at the specified index.
        /// </summary>
        /// <typeparam name="TTransformed"></typeparam>
        /// <param name="i"></param>
        /// <returns></returns>
        public TTransformed GetOutputForAncestorIndex<TTransformed>(int i)
        {
            TransformInfo ti = InputStack[i];
            object obj = ti.PrimaryOutput;
            if (obj == null) return default(TTransformed);
            if (!(obj is TTransformed)) throw new InferCompilerException("Expected output to be of type " + typeof (TTransformed) + " but was of type " + obj.GetType());
            return (TTransformed) obj;
        }

        # endregion Methods for finding output

        # region Methods for adding statements

        /// <summary>
        /// Adds a statement immediately before the output element for the ancestor index given.
        /// </summary>
        /// <param name="ancInd"></param>
        /// <param name="stmt"></param>
        /// 
        /// <param name="convertBeforeAdding"></param>
        public virtual void AddStatementBeforeAncestorIndex(int ancInd, IStatement stmt, bool convertBeforeAdding = false)
        {
            InputStack[ancInd].AddBefore(stmt, convertBeforeAdding);
        }

        /// <summary>
        /// Adds statements immediately before the output element for the ancestor index given.
        /// </summary>
        /// <param name="ancInd"></param>
        /// <param name="stmts"></param>
        /// <param name="convertBeforeAdding"></param>
        public void AddStatementsBeforeAncestorIndex(int ancInd, IEnumerable<IStatement> stmts, bool convertBeforeAdding = false)
        {
            foreach (IStatement stmt in stmts)
            {
                AddStatementBeforeAncestorIndex(ancInd, stmt, convertBeforeAdding);
            }
        }

        /// <summary>
        /// Adds a statement immediately before the statement currently being processed.
        /// </summary>
        /// <param name="statementToAdd"></param>
        public virtual void AddStatementBeforeCurrent(IStatement statementToAdd)
        {
            AddStatementBeforeAncestorIndex(FindAncestorIndex<IStatement>(), statementToAdd);
        }

        public void AddStatementsBeforeCurrent(IEnumerable<IStatement> statementsToAdd)
        {
            AddStatementsBeforeAncestorIndex(FindAncestorIndex<IStatement>(), statementsToAdd);
        }

        //internal List<IStatement> statementsToAdd = new List<IStatement>();
        /// <summary>
        /// Adds a statement which will be inserted after the current statement has been transformed.
        /// </summary>
        /// <param name="statementToAdd"></param>
        public void AddStatementAfterCurrent(IStatement statementToAdd)
        {
            AddStatementAfterAncestorIndex(FindAncestorIndex<IStatement>(), statementToAdd);
        }

        public void AddStatementsAfterCurrent(IEnumerable<IStatement> statementsToAdd)
        {
            AddStatementsAfterAncestorIndex(FindAncestorIndex<IStatement>(), statementsToAdd);
        }


        /// <summary>
        /// Adds a statement after the statement which is currently being processed.
        /// </summary>
        /// <param name="ancestorStatement"></param>
        /// <param name="statementToAdd"></param>
        public void AddStatementAfter(IStatement ancestorStatement, IStatement statementToAdd)
        {
            AddStatementAfterAncestorIndex(GetAncestorIndex(ancestorStatement), statementToAdd);
        }

        /// <summary>
        /// Adds a collection of statements after the statement which is currently being processed. 
        /// </summary>
        /// <param name="ancestorStatement"></param>
        /// <param name="statementsToAdd"></param>
        public void AddStatementsAfter(IStatement ancestorStatement, IEnumerable<IStatement> statementsToAdd)
        {
            AddStatementsAfterAncestorIndex(GetAncestorIndex(ancestorStatement), statementsToAdd);
        }

        /// <summary>
        /// Adds a statement after the ancestor statement at the specified index, optionally converting
        /// the statement before it is added.
        /// </summary>
        public virtual void AddStatementAfterAncestorIndex(int ancestorIndex, IStatement statementToAdd, bool convertBeforeAdding = false)
        {
            TransformInfo ti = InputStack[ancestorIndex];
            ti.AddAfterwards(statementToAdd, convertBeforeAdding);
        }

        /// <summary>
        /// Adds mulitple statements after the ancestor statement at the specified index, optionally converting
        /// the statement before it is added.
        /// </summary>
        public void AddStatementsAfterAncestorIndex(int ancestorIndex, IEnumerable<IStatement> statementsToAdd, bool convertBeforeAdding = false)
        {
            TransformInfo ti = InputStack[ancestorIndex];
            ti.AddAfterwards(statementsToAdd, convertBeforeAdding);
            //foreach (IStatement statementToAdd in statementsToAdd) {
            //  AddStatementAfterAncestorIndex(ancInd, statementToAdd, convertBeforeAdding);
            //}
        }

        internal int ReferenceIndexOf(IList<IStatement> isc, IStatement st)
        {
            for (int i = 0; i < isc.Count; i++) if (object.ReferenceEquals(isc[i], st)) return i;
            return -1;
        }

        /// <summary>
        /// Sets the primary output of the last opened input object.  
        /// </summary>
        /// <param name="outputItem"></param>
        /// <remarks>
        /// This should be called when converting objects that contain other statements, before converting those statements.  
        /// For objects that do not contain statements, it is unnecessary.
        /// </remarks>
        public virtual void SetPrimaryOutput(object outputItem)
        {
            InputStack[InputStack.Count - 1].PrimaryOutput = outputItem;
        }

        # endregion Methods for adding statements
    }
}