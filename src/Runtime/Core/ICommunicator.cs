// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic
{
    public interface ICommunicator
    {
        /// <summary>
        /// Identifies the currently executing process within this communicator. 
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// The number of processes within this communicator. 
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Non-blocking send to a particular processor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="dest"></param>
        /// <param name="tag"></param>
        /// <returns></returns>
        ICommunicatorRequest ImmediateSend<T>(T value, int dest, int tag);
        ICommunicatorRequest ImmediateSend<T>(T[] values, int dest, int tag);

        /// <summary>
        /// Blocking receive from a particular processor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="tag"></param>
        /// <param name="value"></param>
        void Receive<T>(int source, int tag, out T value);

        /// <summary>
        /// Blocking receive an array from a particular processor.  Must have a matching array send.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="tag"></param>
        /// <param name="values">Must be allocated to the correct size.  Cannot be null.</param>
        void Receive<T>(int source, int tag, ref T[] values);

        /// <summary>
        /// Wait until all processes in the communicator have reached the same barrier. 
        /// </summary>
        void Barrier();

        /// <summary>
        /// Blocking send to a particular processor.  Will wait until the value is received.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="dest"></param>
        /// <param name="tag"></param>
        void Send<T>(T value, int dest, int tag);

        /// <summary>
        /// Blocking send array to a particular processor.  Will wait until the value is received.  Must be received by the array-specific receive method.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="values"></param>
        /// <param name="dest"></param>
        /// <param name="tag"></param>
        void Send<T>(T[] values, int dest, int tag);

        /// <summary>
        /// Gather the values from each process into an array of values at the root process. 
        /// On the root process, the pth element of the result will be equal to the value parameter of the process with rank p when this routine returns. 
        /// Other processes receive <c>null</c>.
        /// </summary>
        /// <typeparam name="T">Any serializable type.</typeparam>
        /// <param name="value">The value contributed by this process.</param>
        /// <param name="root">The rank of the process that will receive values for all of the processes in the communicator. </param>
        /// <returns></returns>
        T[] Gather<T>(T value, int root);

        /// <summary> 
        /// Similar to <see cref="Gather&lt;T&gt;(T,int)"/> but here all values are aggregated into one large array. 
        /// </summary> 
        /// <typeparam name="T">Any serializable type.</typeparam> 
        /// <param name="inValues">The values to be contributed by this process.</param> 
        /// <param name="root">The rank of the root node.</param> 
        /// <returns>The aggregated gathered values.</returns> 
        T[] GatherFlattened<T>(T[] inValues, int root);

        /// <summary>
        /// Broadcast a value from the root process to all other processes. 
        /// </summary>
        /// <typeparam name="T">Any serializable type.</typeparam>
        /// <param name="value">The value to be broadcast. At the root process, this value is read (but not written); at all other processes, this value will be replaced with the value at the root. </param>
        /// <param name="root">The rank of the process that is broadcasting the value out to all of the non-root processes. </param>
        void Broadcast<T>(ref T value, int root);

        /// <summary>
        /// Scatters an array of values by sending the ith value of the array to processor i. 
        /// </summary>
        /// <typeparam name="T">Any serializable type.</typeparam>
        /// <param name="values">An array of values of length Size, which is only significant at the root. The ith value of this array (at the root process) will be sent to the ith processor. </param>
        /// <param name="root">Rank of the "root" process, which will supply the array of values to be scattered. </param>
        /// <returns></returns>
        T Scatter<T>(T[] values, int root);

        /// <summary>
        /// Collective operation in which every process sends data to every other process. 
        /// Alltoall differs from Allgather&lt;T&gt;(T) in that a given process can send different data to all of the other processes, rather than contributing the same piece of data to all processes. 
        /// </summary>
        /// <typeparam name="T">Any serializable type.</typeparam>
        /// <param name="values">The array of values that will be sent to each process. The ith value in this array will be sent to the process with rank i.</param>
        /// <returns>An array of values received from all of the other processes. The jth value in this array will be the value sent to the calling process from the process with rank j.</returns>
        T[] Alltoall<T>(T[] values);

        /// <summary> 
        /// Collective operation in which every process sends data to every other process. <c>Alltoall</c> 
        /// differs from Allgather&lt;T&gt;(T) in that a given process can send different 
        /// data to all of the other processes, rather than contributing the same piece of data to all 
        /// processes.  
        /// </summary> 
        /// <typeparam name="T">Any serializable type.</typeparam> 
        /// <param name="inValues"> 
        ///   The array of values that will be sent to each process. sendCounts[i] worth of data will 
        ///   be sent to process i. 
        /// </param> 
        /// <param name="sendCounts"> 
        ///   The numbers of items to be sent to each process. 
        /// </param> 
        ///   The numbers of items to be received by each process. 
        /// <param name="recvCounts"> 
        /// </param> 
        /// <returns> 
        ///   The array of values received from all of the other processes. 
        /// </returns> 
        T[] AlltoallFlattened<T>(T[] inValues, int[] sendCounts, int[] recvCounts);

        /// <summary> 
        ///   <c>Allreduce</c> is a collective algorithm that combines the values stored by each process into a  
        ///   single value available to all processes. The values are combined in a user-defined way, specified via  
        ///   a delegate. If <c>value1</c>, <c>value2</c>, ..., <c>valueN</c> are the values provided by the  
        ///   N processes in the communicator, the result will be the value <c>value1 op value2 op ... op valueN</c>. 
        ///  
        ///   An <c>Allreduce</c> is equivalent to a Reduce&lt;T&gt;(T, MPI.ReductionOperation&lt;T&gt;, int)
        ///   followed by a Broadcast&lt;T&gt;(ref T, int)
        /// </summary> 
        /// <typeparam name="T">Any serializable type.</typeparam> 
        /// <param name="value">The local value that will be combined with the values provided by other processes.</param> 
        /// <param name="op"> 
        ///   The operation used to combine two values. This operation must be associative. 
        /// </param> 
        /// <returns>The result of the reduction. The same value will be returned to all processes.</returns> 
        T Allreduce<T>(T value, Func<T,T,T> op);

        /// <summary>
        ///   <c>Reduce</c> is a collective algorithm that combines the values stored by each process into a 
        ///   single value available at the designated <paramref name="root"/> process. The values are combined 
        ///   in a user-defined way, specified via a delegate. If <c>value1</c>, <c>value2</c>, ..., <c>valueN</c> 
        ///   are the values provided by the N processes in the communicator, the result will be the value 
        ///   <c>value1 op value2 op ... op valueN</c>. This result is only
        ///   available to the <paramref name="root"/> process. 
        /// </summary>
        /// <typeparam name="T">Any serializable type.</typeparam>
        /// <param name="value">The local value that will be combined with the values provided by other processes.</param>
        /// <param name="op">
        ///   The operation used to combine two values. This operation must be associative.
        /// </param>
        /// <param name="root">
        ///   The rank of the process that is the root of the reduction operation, which will receive the result
        ///   of the reduction operation.
        /// </param>
        /// <returns>
        ///   On the root, returns the result of the reduction operation. The other processes receive a default value.
        /// </returns>
        T Reduce<T>(T value, Func<T, T, T> op, int root);

        double PercentTimeSpentWaiting { get; }
    }

    public interface ICommunicatorRequest
    {
        void Wait();
        bool Test();
    }
}
