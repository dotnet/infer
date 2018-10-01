// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic
{
    using System;
    using System.Collections.Generic;
    using System.Threading;
    using Utilities;

    /// <summary>
    /// A mimic of the Communicator class in MPI.NET, that uses local threads only.
    /// </summary>
    public class ThreadCommunicator : CommunicatorBase
    {
        /// <summary>
        /// Shared state for threads using ThreadCommunicator.
        /// </summary>
        protected class SharedState : IDisposable
        {
            public readonly int NumberOfThreads;
            public readonly EventWaitHandle[] newMessage;
            /// <summary>
            /// [thread][sender][tag]
            /// </summary>
            public readonly Dictionary<int, Queue<object>>[][] inbox;
            public readonly Barrier barrier;

            public SharedState(int numberOfThreads)
            {
                this.NumberOfThreads = numberOfThreads;
                newMessage = new EventWaitHandle[numberOfThreads];
                inbox = new Dictionary<int, Queue<object>>[numberOfThreads][];
                for (int i = 0; i < numberOfThreads; i++)
                {
                    newMessage[i] = new AutoResetEvent(false);
                    inbox[i] = Util.ArrayInit(numberOfThreads, sender => new Dictionary<int, Queue<object>>());
                }
                this.barrier = new Barrier(numberOfThreads);
            }
            public void Dispose()
            {
                for (int i = 0; i < NumberOfThreads; i++)
                {
                    newMessage[i].Close();
                }
                this.barrier.Dispose();
            }
        }

        /// <summary>
        /// Execute an action in multiple threads, waiting for them all to finish
        /// </summary>
        /// <param name="numberOfThreads"></param>
        /// <param name="action"></param>
        public static void Run(int numberOfThreads, Action<ThreadCommunicator> action)
        {
            using (var env = new SharedState(numberOfThreads))
            {
                Thread[] threads = new Thread[env.NumberOfThreads];
                for (int i = 1; i < env.NumberOfThreads; i++)
                {
                    threads[i] = new Thread(c => action((ThreadCommunicator)c));
                    threads[i].Start(new ThreadCommunicator(env, i));
                }
                // rank 0 is the current thread
                action(new ThreadCommunicator(env, 0));
                for (int i = 1; i < env.NumberOfThreads; i++)
                {
                    threads[i].Join();
                }
            }
        }

        public override int Rank { get; }
        protected readonly SharedState shared;
        public override int Size { get { return shared.NumberOfThreads; } }
        public const int anySource = -2;
        public const int anyTag = -1;

        protected ThreadCommunicator(SharedState shared, int rank)
        {
            this.shared = shared;
            this.Rank = rank;
        }

        public override void Barrier()
        {
            shared.barrier.SignalAndWait();
        }

        protected class Request : ICommunicatorRequest
        {
            readonly SharedState shared;
            readonly int source, dest, tag;
            bool completed;

            public Request(SharedState shared, int source, int dest, int tag)
            {
                this.shared = shared;
                this.source = source;
                this.dest = dest;
                this.tag = tag;
            }

            public bool Test()
            {
                if (completed)
                    return true;
                Dictionary<int, Queue<object>> inboxOfDest = shared.inbox[dest][source];
                lock (inboxOfDest)
                {
                    Queue<object> messagesWithTag;
                    // send is complete when the destination inbox is empty.
                    completed = !inboxOfDest.TryGetValue(tag, out messagesWithTag) ||
                        (messagesWithTag.Count == 0);
                }
                return completed;
            }

            /// <summary>
            /// Wait for the request to complete
            /// </summary>
            public void Wait()
            {
                if (completed)
                    return;
                Dictionary<int, Queue<object>> inboxOfDest = shared.inbox[dest][source];
                while (true)
                {
                    lock (inboxOfDest)
                    {
                        Queue<object> messagesWithTag;
                        // send is complete when the destination inbox is empty.
                        if (!inboxOfDest.TryGetValue(tag, out messagesWithTag) ||
                            messagesWithTag.Count == 0)
                        {
                            completed = true;
                            return;
                        }
                        Monitor.Wait(inboxOfDest);
                    }
                }
            }
        }

        /// <summary>
        /// Non-blocking send of a single value. This routine will initiate communication and then return immediately with a Request object that can be used to query the status of the communication. 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="dest"></param>
        /// <param name="tag"></param>
        /// <returns></returns>
        /// <remarks>
        /// from the MPI.NET doc:
        /// it is crucial that outstanding communication requests be completed with a successful call to Wait or Test before the request object is lost.
        /// </remarks>
        public override ICommunicatorRequest ImmediateSend<T>(T value, int dest, int tag)
        {
            Dictionary<int, Queue<object>> inboxOfDest = shared.inbox[dest][Rank];
            lock (inboxOfDest)
            {
                Queue<object> messagesWithTag;
                if (!inboxOfDest.TryGetValue(tag, out messagesWithTag))
                {
                    messagesWithTag = new Queue<object>();
                    inboxOfDest[tag] = messagesWithTag;
                }
                messagesWithTag.Enqueue(value);
            }
            shared.newMessage[dest].Set();
            return new Request(shared, Rank, dest, tag);
        }

        /// <summary>
        /// Receive a message from another process, blocking until the message is received.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="tag"></param>
        /// <param name="value"></param>
        public override void Receive<T>(int source, int tag, out T value)
        {
            if (source == anySource)
                throw new NotImplementedException("source == anySource");
            if (tag == anyTag)
                throw new NotImplementedException("tag == anyTag");
            Dictionary<int, Queue<object>> myInbox = shared.inbox[Rank][source];
            while (true)
            {
                lock (myInbox)
                {
                    Queue<object> messagesWithTag;
                    if (myInbox.TryGetValue(tag, out messagesWithTag) &&
                          messagesWithTag.Count > 0)
                    {
                        value = (T)messagesWithTag.Dequeue();
                        Monitor.PulseAll(myInbox); // for Request.Wait()
                        return;
                    }
                }
                // wait until a new message is received
                shared.newMessage[Rank].WaitOne();
            }
        }

        protected class ReceiveRequest<T> : ICommunicatorRequest
        {
            readonly SharedState shared;
            readonly int source, dest, tag;
            readonly Action<T> action;
            bool completed;

            public ReceiveRequest(SharedState shared, int source, int dest, int tag, Action<T> action)
            {
                this.shared = shared;
                this.source = source;
                this.dest = dest;
                this.tag = tag;
                this.action = action;
            }

            /// <summary>
            /// Wait for the request to complete
            /// </summary>
            public void Wait()
            {
                if (completed)
                    return;
                while (true)
                {
                    if (Test())
                        return;
                    // wait until a new message is received
                    shared.newMessage[dest].WaitOne();
                }
            }

            public bool Test()
            {
                if (completed)
                    return true;
                Dictionary<int, Queue<object>> inboxOfDest = shared.inbox[dest][source];
                lock (inboxOfDest)
                {
                    Queue<object> messagesWithTag;
                    if (inboxOfDest.TryGetValue(tag, out messagesWithTag) &&
                        messagesWithTag.Count > 0)
                    {
                        var value = (T)messagesWithTag.Dequeue();
                        action(value);
                        completed = true;
                        Monitor.PulseAll(inboxOfDest); // for Request.Wait()
                    }
                }
                return completed;
            }
        }

        public override ICommunicatorRequest ImmediateReceive<T>(int source, int tag, T[] values)
        {
            if (source == anySource)
                throw new NotImplementedException("source == anySource");
            if (tag == anyTag)
                throw new NotImplementedException("tag == anyTag");
            return new ReceiveRequest<T[]>(shared, source, Rank, tag, array => { array.CopyTo(values, 0); });
        }

        public override ICommunicatorRequest ImmediateReceive<T>(int source, int tag, Action<T> action)
        {
            if (source == anySource)
                throw new NotImplementedException("source == anySource");
            if (tag == anyTag)
                throw new NotImplementedException("tag == anyTag");
            return new ReceiveRequest<T>(shared, source, Rank, tag, action);
        }

        // TODO
        public override double PercentTimeSpentWaiting => throw new NotImplementedException();
    }
}
