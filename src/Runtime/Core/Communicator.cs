// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic
{
    public static class Communicator
    {
        /// <summary>
        /// Determines whether any request has completed. If so, that request will be removed
        /// from the request list and returned. 
        /// </summary>
        /// <returns>
        ///   The first request that has completed, if any. Otherwise, returns <c>null</c> to
        ///   indicate that no request has completed.
        /// </returns>
        public static ICommunicatorRequest TestAny(List<ICommunicatorRequest> requests)
        {
            int n = requests.Count;
            for (int i = 0; i < n; ++i)
            {
                var req = requests[i];
                if (req.Test())
                {
                    requests.RemoveAt(i);
                    return req;
                }
            }

            return null;
        }

        /// <summary>
        /// Waits until any request has completed. That request will then be removed 
        /// from the request list and returned.
        /// </summary>
        /// <returns>The completed request, which has been removed from the request list.</returns>
        public static ICommunicatorRequest WaitAny(List<ICommunicatorRequest> requests)
        {
            if (requests.Count == 0)
                throw new ArgumentException("Cannot call WaitAny with an empty request list");

            while (true)
            {
                var req = TestAny(requests);
                if (req != null)
                    return req;
            }
        }

        public static void Wait(IEnumerable<ICommunicatorRequest> requests)
        {
            var list = requests.ToList();
            while (list.Count > 0)
            {
                WaitAny(list);
            }
        }

        public static IReadOnlyList<ICommunicatorRequest> SendSubarrays<T>(ICommunicator comm, int tag, IReadOnlyList<T> array, int[][] indices)
        {
            List<ICommunicatorRequest> requests = new List<ICommunicatorRequest>();
            for (int destination = 0; destination < comm.Size; destination++)
            {
                if (destination == comm.Rank)
                    continue;
                var request = comm.ImmediateSend(Factors.Collection.Subarray<T>(array, indices[destination]), destination, tag);
                requests.Add(request);
            }
            return requests;
        }

        public static void ReceiveSubarrays<T>(ICommunicator comm, int tag, IList<T> array, int[][] indices)
        {
            for (int source = 0; source < comm.Size; source++)
            {
                if (source == comm.Rank)
                    continue;
                T[] subarray;
                comm.Receive(source, tag, out subarray);
                IReadOnlyList<int> indicesOfSubarray = indices[source];
                for (int i = 0; i < indicesOfSubarray.Count; i++)
                {
                    int index = indicesOfSubarray[i];
                    array[index] = subarray[i];
                    //Debug.WriteLine($"process {comm.Rank} received {subarray[i]} for local index {index}");
                }
            }
        }

        public class TimingInfo
        {
            /// <summary>
            /// Accumulates the total communication time.  Useful for performance tuning.
            /// </summary>
            public Stopwatch watch = new Stopwatch();
            public long ItemsTransmitted;

            public override string ToString()
            {
                long milliseconds = watch.ElapsedMilliseconds;
                string suffix;
                if (milliseconds > ItemsTransmitted)
                    suffix = $" ({watch.ElapsedMilliseconds / System.Math.Max(1, ItemsTransmitted)} ms per item)";
                else
                    suffix = $" ({ItemsTransmitted / System.Math.Max(1, watch.ElapsedMilliseconds)} items per ms)";
                return $"transmitted {ItemsTransmitted} items in {watch.ElapsedMilliseconds} ms" + suffix;
            }
        }

        private static long AlltoallSubarraysItemsTransmitted;

        public static Dictionary<string, TimingInfo> AlltoallSubarraysTimingInfos = new Dictionary<string, TimingInfo>();
        public static Dictionary<string, TimingInfo> MultiplyAllTimingInfos = new Dictionary<string, TimingInfo>();

        public static string GetTimingString()
        {
            StringBuilder sb = new StringBuilder();
            bool firstTime = true;
            foreach (var entry in AlltoallSubarraysTimingInfos) {
                if (!firstTime) sb.AppendLine();
                sb.Append($"AlltoallSubarrays<{entry.Key}> {entry.Value}");
                firstTime = false;
            }
            foreach (var entry in MultiplyAllTimingInfos)
            {
                if (!firstTime)
                    sb.AppendLine();
                sb.Append($"MultiplyAll<{entry.Key}> {entry.Value}");
                firstTime = false;
            }
            return sb.ToString();
        }

        public static void AlltoallSubarrays<T>(ICommunicator comm, IList<T> array, IList<int>[] indicesToSend, IList<int>[] indicesToReceive)
        {
            string key = StringUtil.TypeToString(typeof(T));
            TimingInfo timingInfo;
            lock (AlltoallSubarraysTimingInfos)
            {
                if (!AlltoallSubarraysTimingInfos.TryGetValue(key, out timingInfo))
                {
                    timingInfo = new TimingInfo();
                    AlltoallSubarraysTimingInfos.Add(key, timingInfo);
                }
            }
            long previousItemsTransmitted = AlltoallSubarraysItemsTransmitted;
            timingInfo.watch.Start();
            AlltoallSubarrays(comm, array, indicesToSend, array, indicesToReceive);
            timingInfo.watch.Stop();
            timingInfo.ItemsTransmitted += AlltoallSubarraysItemsTransmitted - previousItemsTransmitted; 
        }

        public static void AlltoallSubarrays<T>(ICommunicator comm, IList<T> inputArray, ICollection<int>[] indicesToSend, IList<T> outputArray, ICollection<int>[] indicesToReceive)
        {
            AlltoallSubarrays(comm, inputArray.AsReadOnly(), indicesToSend, outputArray, indicesToReceive);
        }

        public static void AlltoallSubarrays<T>(ICommunicator comm, IReadOnlyList<T> inputArray, ICollection<int>[] indicesToSend, IList<T> outputArray, ICollection<int>[] indicesToReceive)
        {
            var sendCounts = indicesToSend.Select(indices => indices.Count).ToArray();
            var sendTotalCount = sendCounts.Sum();
            T[] subarrayToSend = new T[sendTotalCount];
            int position = 0;
            for (int recipient = 0; recipient < indicesToSend.Length; recipient++)
            {
                foreach(int index in indicesToSend[recipient])
                {
                    //if (comm.Rank == 0 && recipient == 1)
                    //    Trace.WriteLine($"sending {inputArray[index]}");
                    subarrayToSend[position++] = inputArray[index];
                }
            }
            var recvCounts = indicesToReceive.Select(indices => indices.Count).ToArray();
            var recvTotalCount = recvCounts.Sum();
            AlltoallSubarraysItemsTransmitted += sendTotalCount + recvTotalCount;
            T[] subarrayReceived = comm.AlltoallFlattened(subarrayToSend, sendCounts, recvCounts);
            int recvPosition = 0;
            for (int source = 0; source < comm.Size; source++)
            {
                var indicesOfSubarray = indicesToReceive[source];
                foreach(int index in indicesOfSubarray)
                {
                    outputArray[index] = subarrayReceived[recvPosition++];
                    //Trace.WriteLine($"process {comm.Rank} from {source} received {outputArray[index]} for local index {index} recvPosition = {recvPosition}");
                }
            }
        }

        private class Subarray<T> : ICollection<T>
        {
            readonly IReadOnlyList<T> array;
            readonly ICollection<int> indices;

            public Subarray(IReadOnlyList<T> array, ICollection<int> indices)
            {
                this.array = array;
                this.indices = indices;
            }

            public int Count
            {
                get
                {
                    return indices.Count;
                }
            }

            public bool IsReadOnly
            {
                get
                {
                    return false;
                }
            }

            public void Add(T item)
            {
                throw new NotImplementedException();
            }

            public void Clear()
            {
                throw new NotImplementedException();
            }

            public bool Contains(T item)
            {
                throw new NotImplementedException();
            }

            public void CopyTo(T[] array, int arrayIndex)
            {
                throw new NotImplementedException();
            }

            public bool Remove(T item)
            {
                throw new NotImplementedException();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            public IEnumerator<T> GetEnumerator()
            {
                foreach(int index in indices)
                {
                    yield return array[index];
                }
            }
        }

        /// <summary>
        /// Does not work with BinaryFormatter.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="comm"></param>
        /// <param name="inputArray"></param>
        /// <param name="indicesToSend"></param>
        /// <param name="outputArray"></param>
        /// <param name="indicesToReceive"></param>
        public static void AlltoallSubarrays2<T>(ICommunicator comm, IReadOnlyList<T> inputArray, ICollection<int>[] indicesToSend, IList<T> outputArray, ICollection<int>[] indicesToReceive)
        {
            var sendCounts = indicesToSend.Select(indices => indices.Count).ToArray();
            var subarrays = Util.ArrayInit(indicesToSend.Length, recipient => (ICollection<T>)new Subarray<T>(inputArray, indicesToSend[recipient]));
            var sendTotalCount = sendCounts.Sum();
            var recvCounts = indicesToReceive.Select(indices => indices.Count).ToArray();
            var recvTotalCount = recvCounts.Sum();
            AlltoallSubarraysItemsTransmitted += sendTotalCount + recvTotalCount;
            ICollection<T>[] subarraysReceived = comm.Alltoall(subarrays);
            for (int source = 0; source < comm.Size; source++)
            {
                var indicesOfSubarray = indicesToReceive[source];
                if (indicesOfSubarray.Count == 0)
                    continue;
                var iter = subarraysReceived[source].GetEnumerator();
                foreach (int index in indicesOfSubarray)
                {
                    if (!iter.MoveNext())
                        throw new Exception();
                    outputArray[index] = iter.Current;
                    //Trace.WriteLine($"process {comm.Rank} from {source} received {outputArray[index]} for local index {index} recvPosition = {recvPosition}");
                }
            }
        }

        public static T MultiplyAll<T>(ICommunicator comm, T value)
            where T : ICloneable, SettableToProduct<T>
        {
            string key = StringUtil.TypeToString(typeof(T));
            TimingInfo timingInfo;
            lock (MultiplyAllTimingInfos)
            {
                if (!MultiplyAllTimingInfos.TryGetValue(key, out timingInfo))
                {
                    timingInfo = new TimingInfo();
                    MultiplyAllTimingInfos.Add(key, timingInfo);
                }
            }
            timingInfo.watch.Start();
            var reduced = comm.Allreduce(value, (a, b) =>
            {
                T result = (T)a.Clone();
                result.SetToProduct(a, b);
                return result;
            });
            timingInfo.watch.Stop();
            timingInfo.ItemsTransmitted++;
            return reduced;
        }
    }
}
