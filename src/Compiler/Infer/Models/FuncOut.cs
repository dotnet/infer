// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Models
{
#pragma warning disable 1591
    public delegate TResult FuncOut<in T1, TOut, out TResult>(T1 arg, out TOut output);

    public delegate TResult FuncOut<in T1, in T2, TOut, out TResult>(T1 arg, T2 arg2, out TOut output);

    public delegate TResult FuncOut2<in T1, TOut, TOut2, out TResult>(T1 arg, out TOut output, out TOut2 output2);

    public delegate TResult FuncOut2<in T1, in T2, TOut, TOut2, out TResult>(T1 arg, T2 arg2, out TOut output, out TOut2 output2);

    public delegate TResult FuncOut3<in T1, in T2, TOut, TOut2, TOut3, out TResult>(T1 arg, T2 arg2, out TOut output, out TOut2 output2, out TOut3 output3);

    public delegate TResult FuncOut3<in T1, in T2, in T3, TOut, TOut2, TOut3, out TResult>(T1 arg, T2 arg2, T3 arg3, out TOut output, out TOut2 output2, out TOut3 output3);

    public delegate TResult FuncOut3<in T1, in T2, in T3, in T4, TOut, TOut2, TOut3, out TResult>(T1 arg, T2 arg2, T3 arg3, T4 arg4, out TOut output, out TOut2 output2, out TOut3 output3);

    public delegate TResult FuncOut4<in T1, in T2, TOut, TOut2, TOut3, TOut4, out TResult>(T1 arg, T2 arg2, out TOut output, out TOut2 output2, out TOut3 output3, out TOut4 output4);

    public delegate TResult FuncOut4<in T1, in T2, in T3, TOut, TOut2, TOut3, TOut4, out TResult>(
        T1 arg, T2 arg2, T3 arg3, out TOut output, out TOut2 output2, out TOut3 output3, out TOut4 output4);
#pragma warning restore 1591

    /// <summary>
    /// Generic delegate with 2 out parameters
    /// </summary>
    /// <typeparam name="T1">Type of first argument</typeparam>
    /// <typeparam name="T2">Type of second argument</typeparam>
    /// <typeparam name="T3">Type of third argument</typeparam>
    /// <param name="arg1">First argument</param>
    /// <param name="arg2">Second argument</param>
    /// <param name="arg3">Third argument</param>
    public delegate void ActionOut2<in T1, T2, T3>(T1 arg1, out T2 arg2, out T3 arg3);

    /// <summary>
    /// Generic delegate with 2 out parameters
    /// </summary>
    /// <typeparam name="T1">Type of first argument</typeparam>
    /// <typeparam name="T2">Type of second argument</typeparam>
    /// <typeparam name="T3">Type of third argument</typeparam>
    /// <typeparam name="T4">Type of fourth argument</typeparam>
    /// <param name="arg1">First argument</param>
    /// <param name="arg2">Second argument</param>
    /// <param name="arg3">Third argument</param>
    /// <param name="arg4">Fourth argument</param>
    public delegate void ActionOut2<in T1, in T2, T3, T4>(T1 arg1, T2 arg2, out T3 arg3, out T4 arg4);
}
