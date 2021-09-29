// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;

namespace Crowdsourcing
{
    /// <summary>
    /// Provides the functions to reproduce an iterative labelling selection process with `enums` for the various task selection methods (`TaskSelectionMethod`) and worker selection methods (`WorkerSelectionMethod`).
    /// </summary>
    public class ActiveLearning
    {
        /// <summary>
        /// Full list of simulated data from every task and worker
        /// It is used for initialising the BCC and the CBCC model.
        /// </summary>
        IList<Datum> PredictionData;

        /// <summary>
        /// Flag to indicate whether the model instance is CBCC (true) or BCC (false).
        /// </summary>
        bool IsCommunityModel;

        /// <summary>
        /// List of worker ids.
        /// </summary>
        string[] WorkerIds;

        /// <summary>
        /// List of task ids.
        /// </summary>
        string[] TaskIds;

        /// <summary>
        /// Model instance.
        /// </summary>
        BCC bcc;

        /// <summary>
        ///  Result instance for active learning.
        /// </summary>
        Results ActiveLearningResults;

        /// <summary>
        /// Result instance for batch training.
        /// </summary>
        Results BatchResults;

        /// <summary>
        /// Constructs an active learning instance with a specified data set and model instance.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="model">The model instance.</param>
        /// <param name="results">The results instance.</param>
        /// <param name="numCommunities">The number of communities (only for CBCC).</param>
        public ActiveLearning(IList<Datum> data, BCC model, Results results, int numCommunities)
        {
            this.bcc = model;
            CommunityModel communityModel = model as CommunityModel;
            IsCommunityModel = (communityModel != null);
            ActiveLearningResults = results;
            BatchResults = results;
            WorkerIds = ActiveLearningResults.Mapping.WorkerIdToIndex.Keys.ToArray();
            TaskIds = ActiveLearningResults.Mapping.TaskIdToIndex.Keys.ToArray();

            /// Builds the full matrix of data from every task and worker
            PredictionData = new List<Datum>();
            foreach (var workerId in WorkerIds)
            {
                foreach (var task in TaskIds)
                {
                    PredictionData.Add(new Datum
                    {
                        TaskId = task,
                        WorkerId = workerId,
                        WorkerLabel = 0,
                        GoldLabel = null
                    });
                }
            }
        }

        /// <summary>
        /// Updates the active learning results object.
        /// </summary>
        /// <param name="results">The new results</param>
        public void UpdateActiveLearningResults(Results results)
        {
            ActiveLearningResults = results;
        }

        /// <summary>
        /// Computes the entropy on the true label posterior distribution of the active learning results.
        /// </summary>
        /// <returns>A dictionary keyed by the TaskId and the value is the true label entropy.</returns>
        public Dictionary<string, ActiveLearningResult> EntropyTrueLabelPosterior()
        {
            return BatchResults.TrueLabel.ToDictionary(kvp => kvp.Key, kvp => new ActiveLearningResult
            {
                TaskId = kvp.Key,
                TaskValue = kvp.Value == null ? double.MaxValue : -kvp.Value.GetAverageLog(kvp.Value)
            });
        }

        /// <summary>
        /// Runs the standard active learning procedure on a model instance and an input data set.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="modelName">The model name.</param>
        /// <param name="runType">The model run type.</param>
        /// <param name="model">The model instance.</param>
        /// <param name="taskSelectionMethod">The method for selecting tasks (Random / Entropy).</param>
        /// <param name="workerSelectionMethod">The method for selecting workers (only Random is implemented).</param>
        /// <param name="resultsDir">The directory to save the log files.</param>
        /// <param name="communityCount">The number of communities (only for CBCC).</param>
        /// <param name="initialNumLabelsPerTask">The initial number of exploratory labels that are randomly selected for each task.</param>
        public static void RunActiveLearning(IList<Datum> data, string modelName, RunType runType, BCC model, TaskSelectionMethod taskSelectionMethod, WorkerSelectionMethod workerSelectionMethod, string resultsDir, int communityCount = -1, int initialNumLabelsPerTask = 1)
        {
            //Count elapsed time
            Stopwatch stopWatchTotal = new Stopwatch();
            stopWatchTotal.Start();
            int totalLabels = data.Count();

            // Dictionary keyed by task Id, with randomly order labelings
            var groupedRandomisedData =
                data.GroupBy(d => d.TaskId).
                Select(g =>
                {
                    var arr = g.ToArray();
                    int cnt = arr.Length;
                    var perm = Rand.Perm(cnt);
                    return new
                    {
                        key = g.Key,
                        arr = g.Select((t, i) => arr[perm[i]]).ToArray()
                    };
                }).ToDictionary(a => a.key, a => a.arr);

            // Dictionary keyed by task Id, with label counts
            Dictionary<string, int> totalCounts = groupedRandomisedData.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.Length);
            Dictionary<string, int> currentCounts = groupedRandomisedData.ToDictionary(kvp => kvp.Key, kvp => initialNumLabelsPerTask);

            // Keyed by task, value is a HashSet containing all the remaining workers with a label - workers are removed after adding a new datum 
            Dictionary<string, HashSet<string>> remainingWorkersPerTask = groupedRandomisedData.ToDictionary(kvp => kvp.Key, kvp => new HashSet<string>(kvp.Value.Select(dat => dat.WorkerId)));
            int numTaskIds = totalCounts.Count();
            int totalInstances = data.Count - initialNumLabelsPerTask * numTaskIds;
            string[] WorkerIds = data.Select(d => d.WorkerId).Distinct().ToArray();

            // Log structures
            List<double> accuracy = new List<double>();
            List<double> nlpd = new List<double>();
            List<double> avgRecall = new List<double>();
            List<ActiveLearningResult> taskValueList = new List<ActiveLearningResult>();
            int index = 0;

            Console.WriteLine("Active Learning: {0}", modelName);
            Console.WriteLine("\t\tAcc\tAvgRec");

            // Get initial data
            Results results = new Results();
            List<Datum> subData = null;
            subData = GetSubdata(groupedRandomisedData, currentCounts, remainingWorkersPerTask);
            var s = remainingWorkersPerTask.Select(w => w.Value.Count).Sum();
            List<Datum> nextData = null;
            int numIncremData = 3;
            ActiveLearning activeLearning = null;


            for (int iter = 0; iter < 500; iter++)
            {
                bool calculateAccuracy = true;
                ////bool doSnapShot = iter % 100 == 0; // Frequency of snapshots
                bool doSnapShot = true;
                if (subData != null || nextData != null)
                {
                    switch (runType)
                    {
                        case RunType.VoteDistribution:
                            results.RunMajorityVote(subData, calculateAccuracy, true);
                            break;
                        case RunType.MajorityVote:
                            results.RunMajorityVote(subData, calculateAccuracy, false);
                            break;
                        case RunType.DawidSkene:
                            results.RunDawidSkene(subData, calculateAccuracy);
                            break;
                        default: // Run BCC models
                            results.RunBCC(resultsDir + modelName, subData, data, model, Results.RunMode.ClearResults, calculateAccuracy, communityCount, false);
                            break;
                    }
                }

                if (activeLearning == null)
                {
                    activeLearning = new ActiveLearning(data, model, results, communityCount);
                }
                else
                {
                    activeLearning.UpdateActiveLearningResults(results);
                }

                // Select next task
                Dictionary<string, ActiveLearningResult> TaskValue = null;
                List<Tuple<string, string, ActiveLearningResult>> LabelValue = null;
                switch (taskSelectionMethod)
                {
                    case TaskSelectionMethod.EntropyTask:
                        TaskValue = activeLearning.EntropyTrueLabelPosterior();
                        break;
                    case TaskSelectionMethod.RandomTask:
                        TaskValue = data.GroupBy(d => d.TaskId).ToDictionary(a => a.Key, a => new ActiveLearningResult
                        {
                            TaskValue = Rand.Double()
                        });
                        break;
                    default: // Entropy task selection
                        TaskValue = activeLearning.EntropyTrueLabelPosterior();
                        break;
                }

                nextData = GetNextData(groupedRandomisedData, TaskValue, currentCounts, totalCounts, numIncremData);

                if (nextData == null || nextData.Count == 0)
                    break;

                index += nextData.Count;
                subData.AddRange(nextData);

                // Logs
                if (calculateAccuracy)
                {
                    accuracy.Add(results.Accuracy);
                    nlpd.Add(results.NegativeLogProb);
                    avgRecall.Add(results.AvgRecall);

                    if (TaskValue == null)
                    {
                        var sortedLabelValue = LabelValue.OrderByDescending(kvp => kvp.Item3.TaskValue).ToArray();
                        taskValueList.Add(sortedLabelValue.First().Item3);
                    }
                    else
                    {
                        taskValueList.Add(TaskValue[nextData.First().TaskId]);
                    }

                    if (doSnapShot)
                    {
                        Console.WriteLine("{0} of {1}:\t{2:0.000}\t{3:0.0000}", index, totalInstances, accuracy.Last(), avgRecall.Last());
                        DoSnapshot(accuracy, nlpd, avgRecall, taskValueList, results, modelName, "interim", resultsDir);
                    }
                }
            }
            stopWatchTotal.Stop();
            DoSnapshot(accuracy, nlpd, avgRecall, taskValueList, results, modelName, "final", resultsDir);
            Console.WriteLine("Elapsed time: {0}\n", stopWatchTotal.Elapsed);
        }

        /// <summary>
        /// Saves the results of the inference and the model's parameters on csv files.
        /// </summary>
        /// <param name="accuracy">The list of accuracies evaluated on the gold labels at each active learning round.</param>
        /// <param name="nlpd">The list of NLPD scores evaluated on the gold labels at each active learning round.</param>
        /// <param name="avgRecall">The list of average recalls evaluated on the gold labels at each active learning round.</param>
        /// <param name="taskValue">The list of utilities of the task selected at each active learning round.</param>
        /// <param name="results">The result instance.</param>
        /// <param name="modelName">The model name.</param>
        /// <param name="suffix">The suffix of the csv files.</param>
        /// <param name="resultsDir">The directory to store the csv files.</param>
        public static void DoSnapshot(List<double> accuracy, List<double> nlpd, List<double> avgRecall, List<ActiveLearningResult> taskValue, Results results, string modelName, string suffix, string resultsDir)
        {
            // Snapshot of accuracies, parameters and taskValues.
            using (StreamWriter writer = new StreamWriter(String.Format("{2}{0}_graph_{1}.csv", modelName, suffix, resultsDir)))
            {
                var accArr = accuracy.ToArray();
                var nlpdArr = nlpd.ToArray();
                var avgRec = avgRecall.ToArray();
                for (int i = 0; i < accArr.Length; i++)
                {
                    writer.WriteLine("{0:0.0000}", accArr[i]);
                    writer.WriteLine("{0:0.0000},{1:0.0000}", accArr[i], avgRec[i]);
                    writer.WriteLine("{0:0.0000},{1:0.0000}", accArr[i], nlpdArr[i]);
                }
            }

            using (StreamWriter writer = new StreamWriter(String.Format("{2}{0}_parameters_{1}.csv", modelName, suffix, resultsDir)))
            {
                results.WriteResults(writer, true, true, true);
            }

            using (StreamWriter writer = new StreamWriter(String.Format("{2}{0}_taskValue_{1}.csv", modelName, suffix, resultsDir)))
            {
                for (int i = 0; i < taskValue.Count; i++)
                {
                    writer.WriteLine(String.Format("{0}\t{1}\t{2:0.000}", taskValue[i].TaskId, taskValue[i].WorkerId, taskValue[i].TaskValue));
                }
            }
        }

        /// <summary>
        /// Returns a list of sub-data selected sequentially from the input data list.
        /// </summary>
        /// <param name="groupedRandomisedData">The randomised data.</param>
        /// <param name="currentCounts">The current data count per task.</param>
        /// <param name="workersPerTask">The dictionary keyed by taskId and the value is an hashset of workerId who have remaining labels for the tasks.</param>
        /// <returns>The list of sub-data.</returns>
        public static List<Datum> GetSubdata(Dictionary<string, Datum[]> groupedRandomisedData, Dictionary<string, int> currentCounts, Dictionary<string, HashSet<string>> workersPerTask)
        {
            var data = groupedRandomisedData.Select(g => g.Value.Take(currentCounts[g.Key])).SelectMany(d => d).ToList();
            foreach (Datum d in data)
            {
                workersPerTask[d.TaskId].Remove(d.WorkerId);
            }
            return data;
        }

        /// <summary>
        /// Return the list of sub-data for the task with the highest utility.
        /// </summary>
        /// <param name="groupedRandomisedData">The randomised data.</param>
        /// <param name="taskValue">The dictionary keyed by taskId and the value is an active learning result instance.</param>
        /// <param name="currentCounts">The current data count per task.</param>
        /// <param name="totalCounts">The total data count for all the tasks.</param>
        /// <param name="numIncremData">The number of data to be selected.</param>
        /// <returns>The list of sub-data.</returns>
        public static List<Datum> GetNextData(
            Dictionary<string, Datum[]> groupedRandomisedData,
            Dictionary<string, ActiveLearningResult> taskValue,
            Dictionary<string, int> currentCounts,
            Dictionary<string, int> totalCounts,
            int numIncremData)
        {
            List<Datum> data = new List<Datum>();

            var sortedTaskValues = taskValue.OrderByDescending(kvp => kvp.Value.TaskValue).ToArray();
            if (numIncremData > sortedTaskValues.Length)
                numIncremData = sortedTaskValues.Length;

            int numAdded = 0;
            for (; ; )
            {
                bool noMoreData = currentCounts.All(kvp => kvp.Value >= totalCounts[kvp.Key]);
                if (noMoreData)
                    break;

                for (int i = 0; i < sortedTaskValues.Length; i++)
                {
                    var task = sortedTaskValues[i].Key;
                    int index = currentCounts[task];
                    if (index >= totalCounts[task])
                        continue;
                    data.Add(groupedRandomisedData[task][index]);
                    currentCounts[task] = index + 1;
                    if (++numAdded >= numIncremData)
                        return data;
                }
            }
            return data;
        }

        /// <summary>
        /// Active learning results class with instances representing
        /// pairs of tasks and workers with their utility value.
        /// </summary>
        public class ActiveLearningResult
        {
            /// <summary>
            /// The task id.
            /// </summary>
            public string TaskId
            {
                get;
                set;
            }

            /// <summary>
            /// The worker id.
            /// </summary>
            public string WorkerId
            {
                get;
                set;
            }

            /// <summary>
            /// The utility of a label provided by the worker for the task.
            /// </summary>
            public double TaskValue
            {
                get;
                set;
            }
        }
    }

    /// <summary>
    /// Methods for selecting tasks.
    /// </summary>
    public enum TaskSelectionMethod
    {
        RandomTask,
        EntropyTask,
    }

    /// <summary>
    /// Methods for selecting workers
    /// </summary>
    public enum WorkerSelectionMethod
    {
        RandomWorker
    }
}

