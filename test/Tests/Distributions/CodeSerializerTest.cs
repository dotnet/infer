// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;

#if false

public class CodeSerializerTests
{
    [Fact]
    public void CodeSerializerTest()
    {
        CodeSerializer serializer = new CodeSerializer();
        Console.WriteLine(serializer.ToString(CodeSerializer.ToCodeExpression(4)));
        Console.WriteLine(serializer.ToString(CodeSerializer.ToCodeExpression("Smith")));
        Console.WriteLine(serializer.ToString(CodeSerializer.ToCodeExpression(new double[] { 1.1, 2.2, 3.3 })));

        serializer.AddDefinition("x", (double)4);
        serializer.AddDefinition("s", "Smith");
        object[] a = new object[5];
        a[0] = "text";
        a[1] = a[0];
        a[2] = a;
        object[,] a2 = new object[2, 2];
        a2[0, 0] = "text";
        a2[0, 1] = a[0];
        a2[1, 0] = a;
        a2[1, 1] = a2;
        a[3] = a2;
        serializer.AddDefinition("a", a);
        serializer.AddDefinition("a2", a2);
        //GaussianPlusOp op = new GaussianPlusOp(1,-1);
        //UnaryOp<Gaussian,double> op = new UnaryOp<Gaussian,double>(new Gaussian(0,1));
        //serializer.AddDefinition("op", op);
        Console.WriteLine(serializer);
    }
}
#endif