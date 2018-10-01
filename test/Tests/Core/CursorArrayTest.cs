// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;
using VectorArray = Microsoft.ML.Probabilistic.Collections.CursorArray<Microsoft.ML.Probabilistic.Math.DenseVector>;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class CursorArrayTests
    {
        [Fact]
        public void CursorArrayTest()
        {
            DenseVector v = DenseVector.Zero(4); // cursor
            VectorArray a = new VectorArray(v, 3, 2);
            int i;
            for (i = 0; i < a.Count; i++) a[i].SetAllElementsTo(i);
            for (i = 0; i < a.Count; i++)
            {
                Console.WriteLine("shape[{0}] = {1}", i, a[i]);
            }
            Console.WriteLine("foreach:");
            i = 0;
            foreach (Vector x in a)
            {
                Console.WriteLine("shape[{0}] = {1}", i++, x);
            }
            Console.WriteLine("");
            Console.WriteLine("shape[2,1] = {0}", a[2, 1]);
            Console.WriteLine("v.Start = {0}", v.Start);

            Console.WriteLine("FirstGetsSum():");
            Action action = delegate() { v.SetToSum(v, v); };
            a.ForEach(a, action);
            i = 0;
            foreach (Vector x in a)
            {
                Console.WriteLine("shape[{0}] = {1}", i++, x);
            }

            Console.WriteLine("");
            // a is now used as a cursor in a larger array
            CursorArray<VectorArray> aa = new CursorArray<VectorArray>(a, 2);
            // should be 4*3*2*2 = 48
            Console.WriteLine("total data count = {0}", ((ICursor) aa).Count);
            Console.WriteLine("v.SourceArray.Length = {0}", v.SourceArray.Length);
        }

        public struct VectorPair
        {
            public Vector a;
            public Vector b;

            public VectorPair(Vector a, Vector b)
            {
                this.a = a;
                this.b = b;
            }
        }

        [Fact]
        public void ParallelCursorArrayTest()
        {
            DenseVector v1 = DenseVector.Zero(1);
            DenseVector v2 = DenseVector.Zero(2);
            VectorArray va1 = new VectorArray(v1, 3, 2);
            VectorArray va2 = new VectorArray(v2, 3, 2);
            VectorPair cursor = new VectorPair(v1, v2);
            VectorArray[] arrays = new VectorArray[] {va1, va2};
            ParallelCursorArray<VectorPair, VectorArray> pa =
                new ParallelCursorArray<VectorPair, VectorArray>(cursor, arrays);

            int i;
            for (i = 0; i < pa.Count; i++)
            {
                pa[i].a.SetAllElementsTo(i);
                pa[i].b.SetAllElementsTo(-i);
            }
            i = 0;
            foreach (VectorPair p in pa)
            {
                Console.WriteLine("pa[{0}].shape = {1}, b = {2}", i++, p.a, p.b);
            }
            i = 0;
            foreach (Vector v in va2)
            {
                Console.WriteLine("va2[{0}] = {1}", i++, v);
            }
        }
    }
}