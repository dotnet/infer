// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Wraps a stack to look like a list where you can only access index 0.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class StackAsList<T> : IList<T>
    {
        public Stack<T> Stack;

        public StackAsList(Stack<T> stack)
        {
            this.Stack = stack;
        }

        public StackAsList()
        {
            this.Stack = new Stack<T>();
        }

        public int IndexOf(T item)
        {
            throw new NotSupportedException();
        }

        public void Insert(int index, T item)
        {
            if (index == 0)
            {
                Add(item);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public void RemoveAt(int index)
        {
            if (index == 0)
            {
                Stack.Pop();
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public T this[int index]
        {
            get
            {
                if (index == 0)
                {
                    return Stack.Peek();
                }
                else
                {
                    throw new NotSupportedException();
                }
            }
            set { throw new NotSupportedException(); }
        }

        public void Add(T item)
        {
            Stack.Push(item);
        }

        public void Clear()
        {
            Stack.Clear();
        }

        public bool Contains(T item)
        {
            return Stack.Contains(item);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            Stack.CopyTo(array, arrayIndex);
        }

        public int Count
        {
            get { return Stack.Count; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(T item)
        {
            throw new NotSupportedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            return Stack.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return ((System.Collections.IEnumerable) Stack).GetEnumerator();
        }

        public override string ToString()
        {
            return StringUtil.EnumerableToString(this, Environment.NewLine);
        }
    }

    /// <summary>
    /// Wraps a Queue to look like a list where you can only access index 0.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class QueueAsList<T> : IList<T>
    {
        public Queue<T> Queue;

        public QueueAsList(Queue<T> queue)
        {
            this.Queue = queue;
        }

        public QueueAsList()
        {
            this.Queue = new Queue<T>();
        }

        public int IndexOf(T item)
        {
            throw new NotSupportedException();
        }

        public void Insert(int index, T item)
        {
            if (index == 0)
            {
                Add(item);
            }
            else
            {
                throw new Exception("The method or operation is not implemented.");
            }
        }

        public void RemoveAt(int index)
        {
            if (index == 0)
            {
                Queue.Dequeue();
            }
            else
            {
                throw new Exception("The method or operation is not implemented.");
            }
        }

        public T this[int index]
        {
            get
            {
                if (index == 0)
                {
                    return Queue.Peek();
                }
                else
                {
                    throw new Exception("The method or operation is not implemented.");
                }
            }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        public void Add(T item)
        {
            Queue.Enqueue(item);
        }

        public void Clear()
        {
            Queue.Clear();
        }

        public bool Contains(T item)
        {
            return Queue.Contains(item);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            Queue.CopyTo(array, arrayIndex);
        }

        public int Count
        {
            get { return Queue.Count; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(T item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public IEnumerator<T> GetEnumerator()
        {
            return Queue.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return ((System.Collections.IEnumerable) Queue).GetEnumerator();
        }

        public override string ToString()
        {
            return StringUtil.EnumerableToString(this, Environment.NewLine);
        }
    }
}