// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Compiler.Graphs;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class LabeledListTests
    {
        //public static void Main()
        //{
        //  Console.WriteLine("------------- LabeledArrayList ---------------");
        //  LabeledArrayListTest();

        //  Console.WriteLine("\n------------- LabeledList ---------------");
        //  LabeledListTest();

        //  Console.WriteLine("\n----------- LabeledDictionary -----------");
        //  LabeledDictionaryTest();
        //}

        [Fact]
        public void LabeledListTest()
        {
            LabeledList<string, string> llist = new LabeledList<string, string>("");
            llist.Add("a");
            Console.WriteLine(llist);
            llist.WithLabel("label1").Add("b");
            llist.WithLabel("label1").Add("c");
            llist.WithLabel("label2").Add("d");
            Console.WriteLine(llist);
            foreach (object value in llist)
            {
                Console.Write("{0} ", value);
            }
            Console.WriteLine("");
            Console.WriteLine("Count = {0}", llist.Count);
            Console.WriteLine("Contains(c) = {0}", llist.Contains("c"));
            llist.Remove("c");
            Console.WriteLine("Remove(c):");
            Console.WriteLine(llist);
        }
    }
}