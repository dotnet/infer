// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CrowdsourcingWithWords
{
    /// <summary>
    /// Data mapping class. This class manages the mapping between the data (which is
    /// in the form of task, worker ids, and labels) and the model data (which is in term of indices).
    /// </summary>
    public class DataMapping
    {
        /// <summary>
        /// The mapping from the worker index to the worker id.
        /// </summary>
        public string[] WorkerIndexToId;

        /// <summary>
        /// The mapping from the worker id to the worker index.
        /// </summary>
        public Dictionary<string, int> WorkerIdToIndex;

        /// <summary>
        /// The mapping from the community id to the community index.
        /// </summary>
        public Dictionary<string, int> CommunityIdToIndex;

        /// <summary>
        /// The mapping from the community index to the community id.
        /// </summary>
        public string[] CommunityIndexToId;

        /// <summary>
        /// The mapping from the task index to the task id.
        /// </summary>
        public string[] TaskIndexToId;

        /// <summary>
        /// The mapping from the task id to the task index.
        /// </summary>
        public Dictionary<string, int> TaskIdToIndex;

        /// <summary>
        /// The lower bound of the labels range.
        /// </summary>
        public int LabelMin;

        /// <summary>
        /// The upper bound of the labels range.
        /// </summary>
        public int LabelMax;

        /// <summary>
        /// The enumerable list of data.
        /// </summary>
        public IEnumerable<Datum> Data
        {
            get;
        }

        /// <summary>
        /// The filtered enumerable list of data with gold labels.
        /// </summary>
        public IEnumerable<Datum> DataWithGold
        {
            get;
        }

        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount => LabelMax - LabelMin + 1;

        /// <summary>
        /// The number of workers.
        /// </summary>
        public int WorkerCount => WorkerIndexToId.Length;

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount => TaskIndexToId.Length;

        /// <summary>
        /// Creates a data mapping.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="numCommunities">The number of communities.</param>
        /// <param name="labelMin">The lower bound of the labels range.</param>
        /// <param name="labelMax">The upper bound of the labels range.</param>
        public DataMapping(IEnumerable<Datum> data, int numCommunities = -1, int labelMin = int.MaxValue, int labelMax = int.MinValue)
        {
            WorkerIndexToId = data.Select(d => d.WorkerId).Distinct().ToArray();
            WorkerIdToIndex = WorkerIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
            TaskIndexToId = data.Select(d => d.TaskId).Distinct().ToArray();
            TaskIdToIndex = TaskIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
            var labels = data.SelectMany(d => d.GoldLabel.HasValue ? new[] { d.GoldLabel.Value, d.WorkerLabel } : new[] { d.WorkerLabel })
                .Distinct().OrderBy(lab => lab).ToArray();

            if (labelMin <= labelMax)
            {
                LabelMin = labelMin;
                LabelMax = labelMax;
            }
            else
            {
                LabelMin = labels.Min();
                LabelMax = labels.Max();
            }
            Data = data;

            if (numCommunities > 0)
            {
                CommunityIndexToId = Util.ArrayInit(numCommunities, comm => "Community" + comm);
                CommunityIdToIndex = CommunityIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
            }

            DataWithGold = data.Where(d => d.GoldLabel.HasValue);
        }

        /// <summary>
        /// Returns the matrix of the task indices (columns) of each worker (rows).
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The matrix of the task indices (columns) of each worker (rows).</returns>
        public int[][] GetTaskIndicesPerWorkerIndex(IEnumerable<Datum> data)
        {
            int[][] result = new int[WorkerCount][];
            for (int i = 0; i < WorkerCount; i++)
            {
                var wid = WorkerIndexToId[i];
                result[i] = data.Where(d => d.WorkerId == wid).Select(d => TaskIdToIndex[d.TaskId]).ToArray();
            }

            return result;
        }

        /// <summary>
        /// Returns the matrix of the labels (columns) of each worker (rows).
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The matrix of the labels (columns) of each worker (rows).</returns>
        public int[][] GetLabelsPerWorkerIndex(IEnumerable<Datum> data)
        {
            int[][] result = new int[WorkerCount][];
            for (int i = 0; i < WorkerCount; i++)
            {
                var wid = WorkerIndexToId[i];
                result[i] = data.Where(d => d.WorkerId == wid).Select(d => d.WorkerLabel - LabelMin).ToArray();
            }

            return result;
        }

        /// <summary>
        /// Returns the the gold labels of each task.
        /// </summary>
        /// <returns>The dictionary keyed by task id and the value is the gold label.</returns>
        public Dictionary<string, int?> GetGoldLabelsPerTaskId()
        {
            // Gold labels that are not consistent are returned as null
            // Labels are returned as indexed by task index
            return Data.GroupBy(d => d.TaskId).
              Select(t => t.GroupBy(d => d.GoldLabel).Where(d => d.Key.HasValue)).
              Where(gold_d => gold_d.Count() > 0).
              Select(gold_d =>
              {
                  int count = gold_d.Distinct().Count();
                  var datum = gold_d.First().First();
                  if (count == 1)
                  {
                      var gold = datum.GoldLabel;
                      if (gold != null)
                          gold = gold.Value - LabelMin;
                      return new Tuple<string, int?>(datum.TaskId, gold);
                  }
                  else if (count > 1)
                  {
                      throw new Exception($"Conflicting gold labels for task {gold_d.First().First().TaskId}");
                  }
                  else
                  { 
                      return new Tuple<string, int?>(datum.TaskId, null);
                  }
              }).ToDictionary(tup => tup.Item1, tup => tup.Item2);
        }

        public Dictionary<string, int?> GetRandomLabelPerTaskId(IList<Datum> data)
        {
            // Labels are returned as indexed by task index
            return data.GroupBy(d => d.TaskId).
              Select(collection =>
              {
                  int r = Rand.Int(0, collection.Count() - 1);
                  return new Tuple<string, int?>(collection.Key, collection.ToArray()[r].WorkerLabel);
              }).ToDictionary(tup => tup.Item1, tup => tup.Item2);
        }

        public List<Datum> BuildDataFromAssignedLabels(Dictionary<string, int?> AssignedLabels, IList<Datum> OriginalData)
        {
            List<Datum> data = new List<Datum>();
            string firstWorkerId = WorkerIndexToId[0];
            foreach (var entry in AssignedLabels)
            {
                var datum = new Datum();
                datum.TaskId = entry.Key;
                datum.GoldLabel = entry.Value;
                datum.WorkerLabel = (int)entry.Value;
                datum.WorkerId = firstWorkerId;
                datum.BodyText = OriginalData.First(d => d.TaskId == entry.Key).BodyText;

                data.Add(datum);
            }

            if (data.Count == 0)
                Console.WriteLine("*** Warning: There are no gold labels in the dataset ***");

            return data;
        }

        /// <summary>
        /// For each task, gets the majority vote label if it is unique.
        /// </summary>
        /// <returns>The list of majority vote labels.</returns>
        public int?[] GetMajorityVotesPerTaskIndex()
        {
            return Data.GroupBy(d => TaskIdToIndex[d.TaskId]).
              OrderBy(g => g.Key).
              Select(t => t.GroupBy(d => d.WorkerLabel - LabelMin).
                  Select(g => new { label = g.Key, count = g.Count() })).
                  Select(arr =>
                  {
                      int max = arr.Max(a => a.count);
                      int[] majorityLabs = arr.Where(a => a.count == max).Select(a => a.label).ToArray();
                      if (majorityLabs.Length == 1)
                          return (int?)majorityLabs[0];

                      //return random label;
                      return majorityLabs[0];

                  }).ToArray();
        }

        /// <summary>
        /// For each task Id, gets the majority vote label if it is unique.
        /// </summary>
        /// <returns>The dictionary of majority vote labels indexed by task id.</returns>
        public Dictionary<string, int?> GetMajorityVotesPerTaskId(IList<Datum> data)
        {
            Dictionary<string, int?> majorityVotesPerTaskId = new Dictionary<string, int?>();
            var majorityVotes = GetMajorityVotesPerTaskIndex();
            foreach (var d in data)
            {
                if (!majorityVotesPerTaskId.ContainsKey(d.TaskId))
                    majorityVotesPerTaskId[d.TaskId] = majorityVotes[TaskIdToIndex[d.TaskId]];
            }
            return majorityVotesPerTaskId;
        }
    }
}
