// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic
{
    public abstract class CommunicatorBase : ICommunicator
    {
        public abstract double PercentTimeSpentWaiting { get; }
        public abstract int Rank { get; }
        public abstract int Size { get; }
        public abstract ICommunicatorRequest ImmediateSend<T>(T value, int dest, int tag);
        public abstract ICommunicatorRequest ImmediateReceive<T>(int source, int tag, Action<T> action);
        public abstract ICommunicatorRequest ImmediateReceive<T>(int source, int tag, T[] values);
        public abstract void Receive<T>(int source, int tag, out T value);
        public abstract void Barrier();

        public virtual ICommunicatorRequest ImmediateSend<T>(T[] values, int dest, int tag)
        {
            return ImmediateSend<T[]>(values, dest, tag);
        }

        /// <summary>
        /// Blocking send of a single value.  Will wait until the value is received.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="dest"></param>
        /// <param name="tag"></param>
        public virtual void Send<T>(T value, int dest, int tag)
        {
            var request = ImmediateSend(value, dest, tag);
            request.Wait();
        }

        public virtual void Send<T>(T[] values, int dest, int tag)
        {
            Send<T[]>(values, dest, tag);
        }

        public virtual void Receive<T>(int source, int tag, ref T[] values)
        {
            T[] value;
            Receive(source, tag, out value);
            values = value;
        }

        public virtual T[] Gather<T>(T value, int root)
        {
            const int tag = 5;
            if (Rank == root)
            {
                T[] outValues = new T[Size];
                List<ICommunicatorRequest> requests = new List<ICommunicatorRequest>();
                for (int sender = 0; sender < Size; sender++)
                {
                    if (sender == root)
                    {
                        outValues[sender] = value;
                        continue;
                    }
                    // make a copy since 'sender' is being modified
                    int senderLocal = sender;
                    var request = ImmediateReceive<T>(sender, tag, receivedValue => { outValues[senderLocal] = receivedValue; });
                    requests.Add(request);
                }
                Communicator.Wait(requests);
                return outValues;
            }
            else
            {
                Send(value, root, tag);
                return null;
            }
        }

        public virtual T[] GatherFlattened<T>(T[] inValues, int root)
        {
            const int tag = 5;
            if (Rank == root)
            {
                T[][] outValues = new T[Size][];
                for (int sender = 0; sender < Size; sender++)
                {
                    if (sender == root)
                    {
                        outValues[sender] = inValues;
                        continue;
                    }
                    int length;
                    Receive(sender, tag, out length);
                    T[] valueFromSender = new T[length];
                    Receive(sender, tag, ref valueFromSender);
                    outValues[sender] = valueFromSender;
                }
                // concatenate the arrays
                return outValues.SelectMany(array => array).ToArray();
            }
            else
            {
                Send(inValues.Length, root, tag);
                Send(inValues, root, tag);
                return null;
            }
        }

        public virtual void Broadcast<T>(ref T value, int root)
        {
            const int tag = 7;
            if (Rank == root)
            {
                List<ICommunicatorRequest> requests = new List<ICommunicatorRequest>();
                for (int destination = 0; destination < Size; destination++)
                {
                    if (destination == root)
                        continue;
                    var request = ImmediateSend(value, destination, tag);
                    requests.Add(request);
                }
                Communicator.Wait(requests);
            }
            else
            {
                Receive(root, tag, out value);
            }
        }

        public virtual T Scatter<T>(T[] values, int root)
        {
            const int tag = 7;
            if (Rank == root)
            {
                List<ICommunicatorRequest> requests = new List<ICommunicatorRequest>();
                for (int destination = 0; destination < Size; destination++)
                {
                    if (destination == root)
                        continue;
                    var request = ImmediateSend(values[destination], destination, tag);
                    requests.Add(request);
                }
                Communicator.Wait(requests);
                return values[root];
            }
            else
            {
                T value;
                Receive(root, tag, out value);
                return value;
            }
        }

        public virtual T[] Alltoall<T>(T[] values)
        {
            const int tag = 11;
            T[] outValues = new T[Size];
            List<ICommunicatorRequest> requests = new List<ICommunicatorRequest>();
            for (int destination = 0; destination < Size; destination++)
            {
                if (destination == Rank)
                    continue;
                var request = ImmediateSend(values[destination], destination, tag);
                requests.Add(request);
            }
            for (int source = 0; source < Size; source++)
            {
                if (source == Rank)
                {
                    outValues[source] = values[source];
                    continue;
                }
                // make a copy since 'source' is being modified
                int sourceLocal = source;
                var request = ImmediateReceive<T>(source, tag, value => { outValues[sourceLocal] = value; });
                requests.Add(request);
            }
            Communicator.Wait(requests);
            return outValues;
        }

        protected virtual T[][] Alltoall<T>(T[][] values)
        {
            int[] sendCounts = values.Select(array => array.Length).ToArray();
            int[] recvCounts = Alltoall(sendCounts);
            const int tag = 11;
            T[][] outValues = new T[Size][];
            List<ICommunicatorRequest> requests = new List<ICommunicatorRequest>();
            for (int destination = 0; destination < Size; destination++)
            {
                if (destination == Rank)
                    continue;
                var request = ImmediateSend(values[destination], destination, tag);
                requests.Add(request);
            }
            for (int source = 0; source < Size; source++)
            {
                if (source == Rank)
                {
                    outValues[source] = values[source];
                    continue;
                }
                T[] value = new T[recvCounts[source]];
                outValues[source] = value;
                var request = ImmediateReceive(source, tag, value);
                requests.Add(request);
            }
            Communicator.Wait(requests);
            return outValues;
        }

        public virtual T[] AlltoallFlattened<T>(T[] inValues, int[] sendCounts, int[] recvCounts)
        {
            T[][] inValuesSplit = Util.ArrayInit(sendCounts.Length, i => new T[sendCounts[i]]);
            int position = 0;
            for (int i = 0; i < sendCounts.Length; i++)
            {
                for (int j = 0; j < sendCounts[i]; j++)
                {
                    inValuesSplit[i][j] = inValues[position++];
                }
            }
            T[][] outValuesSplit = Alltoall(inValuesSplit);
            T[] outValues = outValuesSplit.SelectMany(array => array).ToArray();
            return outValues;
        }

        public virtual T Reduce<T>(T value, Func<T, T, T> op, int root)
        {
            if (Size == 1)
                return value;
            T[] values = Gather(value, root);
            T result = default(T);
            if (values != null)
            {
                result = op(values[0], values[1]);
                for (int i = 2; i < values.Length; i++)
                {
                    result = op(result, values[i]);
                }
            }
            return result;
        }

        public virtual T Allreduce<T>(T value, Func<T, T, T> op)
        {
            if (Size == 1)
                return value;
            int root = Size / 2;
            T result = Reduce(value, op, root);
            Broadcast(ref result, root);
            return result;
        }
    }
}
