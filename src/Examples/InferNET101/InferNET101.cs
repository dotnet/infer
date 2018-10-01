// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;

namespace InferNET101
{
    public static class InferNET101
    {
        static void Main()
        {
            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime1");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingSamples.RunCyclingTime1();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime2");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingSamples.RunCyclingTime2();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime3");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingSamples.RunCyclingTime3();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime4");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingSamples.RunCyclingTime4();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime5");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingSamples.RunCyclingTime5();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();
        }
    }
}
