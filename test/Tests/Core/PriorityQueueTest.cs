// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class PriorityQueueTests
    {
        public class Test : IComparable<Test>
        {
            public int Value, Index;

            public int CompareTo(Test that)
            {
                return Value.CompareTo(that.Value);
            }

            public Test(int value)
            {
                Value = value;
            }

            public override string ToString()
            {
                return String.Format("({0},{1})", Index, Value);
            }
        }

        private void ExtractAll<T>(PriorityQueue<T> q)
        {
            while (q.Count > 0)
            {
                Console.WriteLine(" " + q.ExtractMinimum());
            }
        }

        [Fact]
        public void PriorityQueueTest()
        {
            int[] values = {44, 89, 3, 67, 12, 0, 56, 48, 23};
            Test[] array = new Test[values.Length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = new Test(values[i]);
            }
            PriorityQueue<Test> q = new PriorityQueue<Test>(array, delegate(Test t, int i) { t.Index = i; });
            TestUtils.PrintCollection(q.Items);
            q.RemoveAt(1);
            Console.WriteLine("q.RemoveAt(1):");
            TestUtils.PrintCollection(q.Items);

            Test item = new Test(22);
            q.Add(item);
            Console.WriteLine("q.Add(22):");
            TestUtils.PrintCollection(q.Items);
            item.Value = 24;
            q.Changed(item.Index);
            Console.WriteLine("changed to 24:");
            TestUtils.PrintCollection(q.Items);

            Console.WriteLine("extract all:");
            ExtractAll(q);
        }

        [Fact]
        public void PriorityQueueDequeTest()
        {
            int[] values = {44, 89, 3, 67, 12, 0, 56, 48, 23};
            Test[] array = new Test[values.Length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = new Test(values[i]);
            }
            PriorityQueue<Test> q = new PriorityQueue<Test>(array, delegate(Test t, int i) { t.Index = i; });
            while (q.Count > 0)
            {
                Console.WriteLine(q.ExtractMinimum());
            }
        }
    }
}