// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

/* Community-Based Bayesian Aggregation for Crowdsoucing
* 
* Software to run the experiment presented in the paper "Community-Based Bayesian Aggregation Models for Crowdsourcing" by Venanzi et. al, WWW14
* To run it, you must create csv file with your data with the format <Worker id, Task id, worker's label, (optional) task's gold label>:
* 
* Example: {842,79185673,0,0
1258,79185673,0,0
1467,79185673,0,0
1674,79185673,0,0
662,79185673,0,0
708,79185673,0,0
1507,79185673,3,0
1701,79185724,4
38,79185724,3
703,79185724,1
353,79185724,1
165,79185724,0
1025,79185724,4
1638,79185724,4
782,79185900,1
1480,79185900,1}
* 
* You can download the original CF data set used in the paper from www.crowdscale.org
*/


using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GetAnotherLabel;

namespace WWWPaper
{
    using VectorGaussianArray = DistributionRefArray<VectorGaussian, Vector>;
    using VectorGaussianArrayArray = DistributionRefArray<DistributionRefArray<VectorGaussian, Vector>, Vector[]>;
    using DiscreteArray = DistributionRefArray<Discrete, int>;
    using System.IO;

    /// <summary>
    /// The class for the main program.
    /// </summary>
    class Program
    {
        /// <summary>
        /// The datasets.
        /// </summary>
        static string[] GoldDatasets = new string[] { "CF" };

        /// <summary>
        /// The number of communities of CBCC.
        /// </summary>
        static int[] NumCommunities = new int[] { 4 };

        /// <summary>
        /// Flag to run Dawid-Skene (you will also need to link to the Dawid-Skene C# code).
        /// </summary>
        static bool RunDawidSkene = false;

        /// <summary>
        /// The results directory.
        /// </summary>
        static string ResultsDir = @"Results\";

        /// <summary>
        /// Main method to run the crowdsourcing experiments presented in Venanzi et.al (WWW14).
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            int startIndex = 0;
            int endIndex = GoldDatasets.Length - 1;
            int whichModel = -1; // Default value to run all the models
            Directory.CreateDirectory(ResultsDir);

            // Experiment of Figure 5 and Table 2
            RunFullGold(startIndex, endIndex);

            // Experiment of Figure 4
            RunWWWExperiments(startIndex, endIndex, whichModel);

            // Experiment to find the number of communities
            FindNumCommunities(startIndex, endIndex, 10);
        }

        /// <summary>
        /// Runs the active learning experiment presented in Venanzi et.al (WWW14)
        /// for all the models with an array of data sets.
        /// </summary>
        /// <param name="startIndex">First instance of the data set array.</param>
        /// <param name="endIndex">Last instance of the data set array.</param>
        /// <param name="whichModel">Model to run.</param>
        static void RunWWWExperiments(int startIndex, int endIndex, int whichModel)
        {
            for (int ds = startIndex; ds <= endIndex; ds++)
            {
                switch (whichModel)
                {
                    case 1: RunWWWActiveLearning(GoldDatasets[ds], RunType.MajorityVote, TaskSelectionMethod.EntropyTask, null); break;
                    case 2:
                        if (RunDawidSkene)
                        {
                            RunWWWActiveLearning(GoldDatasets[ds], RunType.DawidSkene, TaskSelectionMethod.EntropyTask, null);
                        }
                        break;
                    case 3: RunWWWActiveLearning(GoldDatasets[ds], RunType.BCC, TaskSelectionMethod.EntropyTask, new BCC()); break;
                    case 4: RunWWWActiveLearning(GoldDatasets[ds], RunType.CBCC, TaskSelectionMethod.EntropyTask, new CBCC(), NumCommunities[ds]); break;
                    default: // Run all
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.MajorityVote, TaskSelectionMethod.EntropyTask, null);
                        if (RunDawidSkene)
                        {
                            RunWWWActiveLearning(GoldDatasets[ds], RunType.DawidSkene, TaskSelectionMethod.EntropyTask, null);
                        }
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.BCC, TaskSelectionMethod.EntropyTask, new BCC());
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.CBCC, TaskSelectionMethod.EntropyTask, new CBCC(), NumCommunities[ds]);
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.BCC, TaskSelectionMethod.EntropyTask, new CBCC());
                        break;
                }
            }
        }

        /// <summary>
        /// Runs the active learning experiment presented in Venanzi et.al (WWW14) on a single data set.
        /// </summary>
        /// <param name="dataSet">The data.</param>
        /// <param name="runType">The model run type.</param>
        /// <param name="taskSelectionMethod">The method for selecting tasks (Random / Entropy).</param>
        /// <param name="model">The model instance.</param>
        /// <param name="communityCount">The number of communities (only for CBCC).</param>
        static void RunWWWActiveLearning(string dataSet, RunType runType, TaskSelectionMethod taskSelectionMethod, BCC model, int communityCount = 4)
        {
            // Reset the random seed so results can be duplicated for the paper
            Rand.Restart(12347);
            var workerSelectionMetric = WorkerSelectionMethod.RandomWorker;
            var data = Datum.LoadData(@"Data\" + dataSet + ".csv");
            string modelName = GetModelName(dataSet, runType, taskSelectionMethod, workerSelectionMetric, communityCount);
            ActiveLearning.RunActiveLearning(data, modelName, runType, model, taskSelectionMethod, workerSelectionMetric, ResultsDir, communityCount);
        }

        /// <summary>
        /// Runs all the models on an array of full gold sets.
        /// </summary>
        /// <param name="startIndex">The first index of the gold set array.</param>
        /// <param name="endIndex">The fast index of the gold set array.</param>
        static void RunFullGold(int startIndex, int endIndex)
        {
            Console.Write("RunFullGolds: Running models");
            for (int ds = startIndex; ds <= endIndex; ds++)
            {
                RunGold(GoldDatasets[ds], RunType.MajorityVote, null); Console.Write(".");
                if (RunDawidSkene)
                {
                    RunGold(GoldDatasets[ds], RunType.DawidSkene, null);
                    Console.Write(".");
                }
                RunGold(GoldDatasets[ds], RunType.BCC, new BCC()); Console.Write(".");
                RunGold(GoldDatasets[ds], RunType.CBCC, new CBCC(), NumCommunities[ds]); Console.Write(".");
            }
            Console.Write("done\n");
        }

        /// <summary>
        /// Finds the optimal number of communities
        /// </summary>
        /// <param name="startIndex">The first index of the gold set array.</param>
        /// <param name="endIndex">The fast index of the gold set array.</param>
        /// <param name="communityUpperBound">The maximum number of communities</param>
        /// ///
        static void FindNumCommunities(int startIndex, int endIndex, int communityUpperBound = 10)
        {
            Console.WriteLine("Find community count: Running models");
            var modelEvidence = Util.ArrayInit<double>(communityUpperBound, endIndex + 1, (i,j) => 0.0);
            for (int ds = startIndex; ds <= endIndex; ds++)
            {
                Console.WriteLine("Dataset: " + GoldDatasets[ds]);
                for (int communityCount = 1; communityCount <= communityUpperBound; communityCount++)
                {
                    Results results = RunGold(GoldDatasets[ds], RunType.CBCC, new CBCC(), communityCount);
                    modelEvidence[communityCount-1, ds] = results.ModelEvidence.LogOdds;
                    Console.WriteLine("Community {0}: {1:0.0000}", communityCount, modelEvidence[communityCount - 1, ds]);
                }
            }
        }

        /// <summary>
        /// Runs a model with the full gold set.
        /// </summary>
        /// <param name="dataSet">The data.</param>
        /// <param name="runType">The model run type.</param>
        /// <param name="model">The model instance.</param>
        /// <param name="communityCount">The number of communities (only for CBCC).</param>
        /// <returns>The inference results</returns>
        static Results RunGold(string dataSet, RunType runType, BCC model, int communityCount = 3)
        {
            // Reset the random seed so results can be duplicated for the paper
            Rand.Restart(12347);
            var data = Datum.LoadData(@".\Data\" + dataSet + ".csv");
            int totalLabels = data.Count();

            string modelName = GetModelName(dataSet, runType, TaskSelectionMethod.EntropyTask, WorkerSelectionMethod.RandomWorker);
            Results results = new Results();

            switch (runType)
            {
                case RunType.VoteDistribution:
                    results.RunMajorityVote(data, true, true);
                    break;
                case RunType.MajorityVote:
                    results.RunMajorityVote(data, true, false);
                    break;
                case RunType.DawidSkene:
                    results.RunDawidSkene(data, true);
                    break;
                default:
                    results.RunBCC(modelName, data, data, model, Results.RunMode.ClearResults, false, communityCount, false, false);
                    break;
            }

            // Write the inference results on a csv file
            using (StreamWriter writer = new StreamWriter(ResultsDir + "endpoints.csv", true))
            {
                writer.WriteLine("{0}:,{1:0.000},{2:0.0000}", modelName, results.Accuracy, results.NegativeLogProb);
            }
            return results;
        }

        /// <summary>
        /// Returns the model name as a string.
        /// </summary>
        /// <param name="dataset">The name of the data set.</param>
        /// <param name="runType">The model run type.</param>
        /// <param name="taskSelectionMethod">The method for selecting tasks (Random / Entropy).</param>
        /// <param name="workerSelectionMethod">The method for selecting workers (only Random is implemented).</param>
        /// <param name="numCommunities">The number of communities (only for CBCC).</param>
        /// <returns>The model name</returns>
        public static string GetModelName(string dataset, RunType runType, TaskSelectionMethod taskSelectionMethod, WorkerSelectionMethod workerSelectionMethod, int numCommunities = -1)
        {
            return dataset + "_" + Enum.GetName(typeof(RunType), runType)
                + "_" + Enum.GetName(typeof(TaskSelectionMethod), taskSelectionMethod);
        }
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

    /// <summary>
    /// Options for which model to run.
    /// </summary>
    public enum RunType
    {
        /// <summary>
        /// The true label distribution
        /// as given by the normalised workers' label counts.
        /// </summary>
        VoteDistribution = 0,

        /// <summary>
        /// The true label is the majority label.
        /// </summary>
        MajorityVote = 1,

        /// <summary>
        /// The Dawid-Skene model.
        /// </summary>
        DawidSkene = 2,

        /// <summary>
        /// The BCC model.
        /// </summary>
        BCC = 3,

        /// <summary>
        /// The CBCC model.
        /// </summary>
        CBCC = 4,
    }

    /// <summary>
    /// Metrics for selecting tasks.
    /// </summary>
    public enum TaskSelectionMethod
    {
        RandomTask,
        EntropyTask,
    }

    /// <summary>
    /// Metrics for selecting workers
    /// </summary>
    public enum WorkerSelectionMethod
    {
        RandomWorker
    }

    /// <summary>
    /// Class of active learning functions
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
            CBCC communityModel = model as CBCC;
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
                            results.RunBCC(modelName, subData, data, model, Results.RunMode.ClearResults, calculateAccuracy, communityCount, false);
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
    }

    /// <summary>
    /// Results class containing posteriors and predictions.
    /// </summary>
    public class Results
    {
        /// <summary>
        /// The posterior of the true label for each task.
        /// </summary>
        public Dictionary<string, Discrete> TrueLabel
        {
            get;
            private set;
        }

        /// <summary>
        /// The predicted label for each task when doing simulations from the current
        /// model state. It avoids overwriting the true label posterior.
        /// </summary>
        public Dictionary<string, Discrete> LookAheadTrueLabel
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for the true label variable.
        /// </summary>
        public Dictionary<string, Discrete> TrueLabelConstraint
        {
            get;
            private set;
        }

        /// <summary>
        /// The probabilities that generate the true label of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior of the confusion matrix of each worker.
        /// </summary>
        public Dictionary<string, Dirichlet[]> WorkerConfusionMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The look-ahead posterior of the confusion matrix of each worker obtained after simulating
        /// a new label in look-ahead run mode.
        /// </summary>
        public Dictionary<string, Dirichlet[]> LookAheadWorkerConfusionMatrix
        {
            get;
            private set;
        }


        /// <summary>
        /// The predictive probabilities of the labels produced by each worker.
        /// </summary>
        public Dictionary<string, Dictionary<string, Discrete>> WorkerPrediction
        {
            get;
            private set;
        }

        /// <summary>
        /// The community membership probabilities of each worker.
        /// </summary>
        public Dictionary<string, Discrete> WorkerCommunity
        {
            get;
            private set;
        }

        /// <summary>
        /// The confusion matrix of each community.
        /// </summary>
        public Dirichlet[][] CommunityConfusionMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The score matrix of each community.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for worker confusion matrices
        /// int the community model.
        /// </summary>
        public Dictionary<string, VectorGaussian[]> WorkerScoreMatrixConstraint
        {
            get;
            private set;
        }

        /// <summary>
        /// The probabilities that generate the community memberships of all the workers.
        /// </summary>
        public Dirichlet CommunityProb
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for community membership.
        /// int the community model.
        /// </summary>
        public Dictionary<string, Discrete> CommunityConstraint
        {
            get;
            private set;
        }

        /// <summary>
        /// Model evidence.
        /// </summary>
        public Bernoulli ModelEvidence
        {
            get;
            private set;
        }

        /// <summary>
        /// The data mapping.
        /// </summary>
        public DataMapping Mapping
        {
            get;
            private set;
        }

        /// <summary>
        /// The gold labels of each task. The gold label type is nullable to
        /// support the (usual) situation where the is no labels.
        /// </summary>
        public Dictionary<string, int?> GoldLabels
        {
            get;
            private set;
        }

        /// <summary>
        /// The accuracy of the current true label predictions.
        /// </summary>
        public double Accuracy
        {
            get;
            private set;
        }

        /// <summary>
        /// The negative log probability density (NLPD) scores of the current true label predictions.
        /// </summary>
        public double NegativeLogProb
        {
            get;
            private set;
        }

        /// <summary>
        /// The average recall of the current true label predictions.
        /// </summary>
        public double AvgRecall
        {
            get;
            private set;
        }

        /// <summary>
        /// The confusion matrix of the predicted true labels against the gold labels
        /// The rows are the gold labels and the columns are the predicted labels.
        /// </summary>
        public double[,] ModelConfusionMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// Flags whether the model instance is CBCC (true) or BCC (false).
        /// </summary>
        public bool IsCommunityModel
        {
            get;
            private set;
        }

        /// <summary>
        /// The number of communities.
        /// </summary>
        public int CommunityCount
        {
            get;
            private set;
        }

        /// <summary>
        /// Runs the majority vote method on the data.
        /// </summary>
        /// <param name="data">The data</param>
        /// <param name="calculateAccuracy">Compute the accuracy (true).</param>
        /// <param name="useVoteDistribution">The true label is sampled from the vote distribution (true) or it is
        /// taken as the mode of the vote counts (false).
        /// In the latter case, ties are broken by sampling from the most voted classes.</param>
        /// <returns>The updated results</returns>
        public Results RunMajorityVote(IList<Datum> data, bool calculateAccuracy, bool useVoteDistribution)
        {
            var dataMapping = new DataMapping(data);
            Mapping = dataMapping;
            GoldLabels = Mapping.GetGoldLabelsPerTaskId();
            var inferredLabels = useVoteDistribution ? dataMapping.GetVoteDistribPerTaskIndex() : dataMapping.GetMajorityVotesPerTaskIndex().Select(mv => mv == null ? (Discrete)null : Discrete.PointMass(mv.Value, dataMapping.LabelCount)).ToArray();
            TrueLabel = inferredLabels.Select((lab, i) => new
            {
                key = dataMapping.TaskIndexToId[i],
                val = lab
            }).ToDictionary(a => a.key, a => a.val);

            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }
            return this;
        }

        /// <summary>
        /// Run Dawid-Skene on the data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="calculateAccuracy">Whether to calculate accuracy</param>
        /// <returns>A results instance</returns>
        public Results RunDawidSkene(IList<Datum> data, bool calculateAccuracy)
        {
            // If you want to run Dawid-Skene code, download his code, integrate it into
            // the project, and change false to true below.
#if false
            var dataMapping = new DataMapping(data);
            Mapping = dataMapping;
            var labelings = data.Select(d => new Labeling(d.WorkerId, d.TaskId, d.WorkerLabel.ToString(), d.GoldLabel.ToString())).ToList();
            DawidSkene ds = new DawidSkene(labelings, null, null);
            // The labels may be in a different order from our data labeling - we need to create a map.
            int[] labelIndexMap = new int[dataMapping.LabelCount];
            var dwLabels = ds.classes.Keys.ToArray();
            for (int i = 0; i < dataMapping.LabelCount; i++)
            {
                labelIndexMap[i] = Array.IndexOf(dwLabels, (i + dataMapping.LabelMin).ToString());
            }

            GoldLabels = Mapping.GetGoldLabelsPerTaskId().
                ToDictionary(kvp => kvp.Key, kvp => kvp.Value == null ? (int?)null : (int?)labelIndexMap[kvp.Value.Value]);

            ds.Estimate(10);

            var inferredLabels = ds.GetObjectClassProbabilities().Select(r => new Discrete(r)).ToArray();
            TrueLabel = inferredLabels.Select((lab, i) => new
            {
                key = dataMapping.TaskIndexToId[i],
                val = lab
            }).ToDictionary(a => a.key, a => a.val);

            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }

            return this;
#else
            throw new ApplicationException("To support Dawid-Skene, you must link to the C# version of their code");
#endif
        }

        /// <summary>
        /// The different modes in which the model can be run.
        /// </summary>
        public enum RunMode
        {
            /// <summary>
            /// Clears all posteriors
            /// </summary>
            ClearResults,
            /// <summary>
            /// Training from a batch of data - uses initial priors.
            /// </summary>
            BatchTraining,
            /// <summary>
            /// Online training from a batch of data - uses previous posteriors as priors.
            /// </summary>
            OnlineTraining,
            /// <summary>
            /// Online training where we don't update the posteriors
            /// </summary>
            LookAheadExperiment,
            /// <summary>
            /// Use communities as workers in a BCC
            /// </summary>
            LoadAndUseCommunityPriors,
            /// <summary>
            /// Prediction of worker labels
            /// </summary>
            Prediction,
        };

        /// <summary>
        /// The structure with the model parameters.
        /// </summary>
        [Serializable]
        public struct NonTaskWorkerParameters
        {
            public Dirichlet BackgroundLabelProb;
            public Dirichlet CommunityProb;
            public VectorGaussian[][] CommunityScoreMatrix;
        }

        /// <summary>
        /// Runs the BCC or CBCC model.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="data">The data that will be used for this run.</param>
        /// <param name="fullData">The full data set of data.</param>
        /// <param name="model">The model instance (BCC or CBCC).</param>
        /// <param name="mode">The mode (for example training, prediction, etc.).</param>
        /// <param name="calculateAccuracy">Whether to calculate accuracy.</param>
        /// <param name="numCommunities">The number of communities (community model only).</param>
        /// <param name="serialize">Whether to serialize all posteriors.</param>
        /// <param name="serializeCommunityPosteriors">Whether to serialize community posteriors.</param>
        public void RunBCC(string modelName, IList<Datum> data, IList<Datum> fullData, BCC model, RunMode mode, bool calculateAccuracy, int numCommunities = -1, bool serialize = false, bool serializeCommunityPosteriors = false)
        {
            CBCC communityModel = model as CBCC;
            IsCommunityModel = communityModel != null;

            if (this.Mapping == null)
            {
                this.Mapping = new DataMapping(fullData, numCommunities);
                this.GoldLabels = this.Mapping.GetGoldLabelsPerTaskId();
            }

            /// A new model is created if the label count or the task count has changed
            bool createModel = (Mapping.LabelCount != model.LabelCount) || (Mapping.TaskCount != model.TaskCount);

            if (IsCommunityModel)
            {

                /// Creates a new CBCC model instance
                CommunityCount = numCommunities;
                createModel = createModel || (numCommunities != communityModel.CommunityCount);
                if (createModel)
                {
                    communityModel.CreateModel(Mapping.TaskCount, Mapping.LabelCount, numCommunities);
                }

            }
            else if (createModel)
            {
                /// Creates a new BCC model instance
                model.CreateModel(Mapping.TaskCount, Mapping.LabelCount);
            }

            /// Selects the prior according to the run mode
            BCCPosteriors priors = null;
            switch (mode)
            {
                /// Use existing priors
                case RunMode.OnlineTraining:
                case RunMode.LookAheadExperiment:
                case RunMode.Prediction:
                    priors = ToPriors();
                    break;
                default:

                    /// Use default priors
                    ClearResults();
                    if (mode == RunMode.LoadAndUseCommunityPriors && IsCommunityModel)
                    {
                        priors = DeserializeCommunityPosteriors(modelName, numCommunities);
                    }
                    break;
            }

            /// Get data to observe
            var labelsPerWorkerIndex = Mapping.GetLabelsPerWorkerIndex(data);
            if (mode == RunMode.Prediction)
            {
                /// Signal prediction mode by setting all labels to null
                labelsPerWorkerIndex = labelsPerWorkerIndex.Select(arr => (int[])null).ToArray();
            }

            /// Run model inference
            BCCPosteriors posteriors = model.Infer(
                Mapping.GetTaskIndicesPerWorkerIndex(data),
                labelsPerWorkerIndex, priors);
            UpdateResults(posteriors, mode);

            /// Compute accuracy
            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }

            /// Serialize parameters
            if (serialize)
            {
                using (FileStream stream = new FileStream(modelName + ".xml", FileMode.Create))
                {
                    var serializer = new System.Xml.Serialization.XmlSerializer(IsCommunityModel ? typeof(CBCCPosteriors) : typeof(BCCPosteriors));
                    serializer.Serialize(stream, posteriors);
                }
            }

            if (serializeCommunityPosteriors && IsCommunityModel)
            {
                SerializeCommunityPosteriors(modelName);
            }
        }

        /// <summary>
        /// Serializes the posteriors on an xml file.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        void SerializeCommunityPosteriors(string modelName)
        {
            NonTaskWorkerParameters ntwp = new NonTaskWorkerParameters();
            ntwp.BackgroundLabelProb = BackgroundLabelProb;
            ntwp.CommunityProb = CommunityProb;
            ntwp.CommunityScoreMatrix = CommunityScoreMatrix;
            using (FileStream stream = new FileStream(modelName + "CommunityPriors.xml", FileMode.Create))
            {
                var serializer = new System.Xml.Serialization.XmlSerializer(typeof(NonTaskWorkerParameters));
                serializer.Serialize(stream, ntwp);
            }
        }

        /// <summary>
        /// Deserializes the parameters of CBCC from an xml file (used in the LoadAndUseCommunityPriors mode).
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="numCommunities">The number of communities.</param>
        /// <returns></returns>
        CBCCPosteriors DeserializeCommunityPosteriors(string modelName, int numCommunities)
        {
            CBCCPosteriors cbccPriors = new CBCCPosteriors();
            using (FileStream stream = new FileStream(modelName + "CommunityPriors.xml", FileMode.Open))
            {
                var serializer = new System.Xml.Serialization.XmlSerializer(typeof(NonTaskWorkerParameters));
                var ntwp = (NonTaskWorkerParameters)serializer.Deserialize(stream);

                if (ntwp.BackgroundLabelProb.Dimension != Mapping.LabelCount)
                {
                    throw new ApplicationException("Unexpected number of labels");
                }

                BackgroundLabelProb = ntwp.BackgroundLabelProb;
                cbccPriors.BackgroundLabelProb = ntwp.BackgroundLabelProb;
                if (ntwp.CommunityScoreMatrix.Length != numCommunities)
                {
                    throw new ApplicationException("Unexpected number of communities");
                }

                if (ntwp.CommunityScoreMatrix[0][0].Dimension != Mapping.LabelCount)
                {
                    throw new ApplicationException("Unexpected number of labels");
                }

                CommunityScoreMatrix = ntwp.CommunityScoreMatrix;
                cbccPriors.CommunityScoreMatrix = ntwp.CommunityScoreMatrix;

                if (ntwp.CommunityProb.Dimension != numCommunities)
                {
                    throw new ApplicationException("Unexpected number of communities");
                }

                CommunityProb = ntwp.CommunityProb;
                cbccPriors.CommunityProb = ntwp.CommunityProb;
            }

            return cbccPriors;
        }

        /// <summary>
        /// Resets all the parameters to the default values.
        /// </summary>
        void ClearResults()
        {
            BackgroundLabelProb = Dirichlet.Uniform(Mapping.LabelCount);
            WorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            WorkerPrediction = new Dictionary<string, Dictionary<String, Discrete>>();
            WorkerCommunity = new Dictionary<string, Discrete>();
            TrueLabel = new Dictionary<string, Discrete>();
            TrueLabelConstraint = new Dictionary<string, Discrete>();
            CommunityConfusionMatrix = null;
            WorkerScoreMatrixConstraint = new Dictionary<string, VectorGaussian[]>();
            CommunityProb = null;
            CommunityScoreMatrix = null;
            CommunityConstraint = new Dictionary<string, Discrete>();
            LookAheadTrueLabel = new Dictionary<string, Discrete>();
            LookAheadWorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            ModelEvidence = new Bernoulli(0.5);
        }

        /// <summary>
        /// Updates the results of with the new posteriors.
        /// </summary>
        /// <param name="posteriors">The posteriors.</param>
        /// <param name="mode">The mode (for example training, prediction, etc.).</param>
        void UpdateResults(BCCPosteriors posteriors, RunMode mode)
        {

            /// In the lookAheadExperiment mode, update only the LookAhead results
            if (mode == RunMode.LookAheadExperiment)
            {
                for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                {
                    LookAheadTrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                }
                for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                {
                    LookAheadWorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                }
            }

            /// In the prediction mode, update only the worker prediction results
            else if (mode == RunMode.Prediction)
            {
                for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                {
                    WorkerPrediction[Mapping.WorkerIndexToId[w]] = new Dictionary<string, Discrete>();
                    for (int tw = 0; tw < posteriors.WorkerPrediction[w].Length; tw++)
                    {
                        WorkerPrediction[Mapping.WorkerIndexToId[w]][Mapping.TaskIndexToId[tw]] = posteriors.WorkerPrediction[w][tw];
                    }
                }
            }
            else
            {
                /// In the all other modes, update all the results
                CBCCPosteriors cbccPosteriors = posteriors as CBCCPosteriors;
                bool isCommunityModel = cbccPosteriors != null;
                BackgroundLabelProb = posteriors.BackgroundLabelProb;
                for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                {
                    WorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                }
                for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                {
                    TrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                }
                for (int t = 0; t < posteriors.TrueLabelConstraint.Length; t++)
                {
                    TrueLabelConstraint[Mapping.TaskIndexToId[t]] = posteriors.TrueLabelConstraint[t];
                }

                if (isCommunityModel)
                {
                    CommunityConfusionMatrix = cbccPosteriors.CommunityConfusionMatrix;
                    for (int w = 0; w < cbccPosteriors.WorkerScoreMatrixConstraint.Length; w++)
                    {
                        WorkerScoreMatrixConstraint[Mapping.WorkerIndexToId[w]] = cbccPosteriors.WorkerScoreMatrixConstraint[w];
                        CommunityConstraint[Mapping.WorkerIndexToId[w]] = cbccPosteriors.WorkerCommunityConstraint[w];
                        WorkerCommunity[Mapping.WorkerIndexToId[w]] = cbccPosteriors.Community[w];
                    }

                    CommunityProb = cbccPosteriors.CommunityProb;
                    CommunityScoreMatrix = cbccPosteriors.CommunityScoreMatrix;
                }

                this.ModelEvidence = posteriors.Evidence;
            }
        }

        /// <summary>
        /// Loads the priors of BCC and CBCC.
        /// </summary>
        /// <returns>A BCC posterior instance with the loaded priors.</returns>
        BCCPosteriors ToPriors()
        {
            int numClasses = Mapping.LabelCount;
            int numTasks = Mapping.TaskCount;
            int numWorkers = Mapping.WorkerCount;
            CBCCPosteriors cbccPriors = new CBCCPosteriors();
            BCCPosteriors priors = IsCommunityModel ? cbccPriors : new BCCPosteriors();

            /// Loads the prior of the background probabilities of the tasks
            priors.BackgroundLabelProb = BackgroundLabelProb;

            /// Loads the prior of the confusion matrix of each worker
            priors.WorkerConfusionMatrix = Util.ArrayInit(numWorkers,
                w =>
                {
                    string wid = Mapping.WorkerIndexToId[w];
                    if (WorkerConfusionMatrix.ContainsKey(wid))
                        return Util.ArrayInit(numClasses, lab => WorkerConfusionMatrix[wid][lab]);
                    else
                        return Util.ArrayInit(numClasses, lab => Dirichlet.Uniform(numClasses));
                });

            /// Loads the true label constraint of each task
            priors.TrueLabelConstraint = Util.ArrayInit(numTasks,
                t =>
                {
                    string tid = Mapping.TaskIndexToId[t];
                    if (TrueLabelConstraint.ContainsKey(tid))
                        return TrueLabelConstraint[Mapping.TaskIndexToId[t]];
                    else
                        return Discrete.Uniform(numClasses);
                });

            /// Loads the priors of the parameters of CBCC
            if (IsCommunityModel)
            {
                cbccPriors.CommunityConfusionMatrix = CommunityConfusionMatrix;
                cbccPriors.WorkerScoreMatrixConstraint = Util.ArrayInit(numWorkers,
                w =>
                {
                    string wid = Mapping.WorkerIndexToId[w];
                    if (WorkerScoreMatrixConstraint.ContainsKey(wid))
                        return Util.ArrayInit(numClasses, lab => WorkerScoreMatrixConstraint[wid][lab]);
                    else
                        return Util.ArrayInit(numClasses, lab => VectorGaussian.Uniform(numClasses));
                });
                cbccPriors.CommunityProb = CommunityProb;
                cbccPriors.CommunityScoreMatrix = CommunityScoreMatrix;
                cbccPriors.WorkerCommunityConstraint = Util.ArrayInit(numWorkers,
                w =>
                {
                    string wid = Mapping.WorkerIndexToId[w];
                    if (CommunityConstraint.ContainsKey(wid))
                        return CommunityConstraint[wid];
                    else
                        return Discrete.Uniform(CommunityCount);
                });
            }

            priors.Evidence = ModelEvidence;

            return priors;
        }

        /// <summary>
        /// Updates the accuracy using the current results.
        /// </summary>
        private void UpdateAccuracy()
        {
            double nlpdThreshold = -Math.Log(0.001);
            int labelCount = TrueLabel.Where(kvp => kvp.Value != null).First().Value.Dimension;
            var confusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => 0.0);
            int correct = 0;
            double logProb = 0.0;

            int goldX = 0;
            foreach (var kvp in GoldLabels)
            {
                if (kvp.Value == null)
                    continue;

                // We have a gold label
                goldX++;

                var trueLabel = TrueLabel[kvp.Key];
                if (trueLabel == null)
                    continue;  // No inferred label

                var probs = trueLabel.GetProbs();
                var max = probs.Max();
                var predictedLabels = probs.Select((p, i) => new
                {
                    prob = p,
                    idx = i
                }).Where(a => a.prob == max).Select(a => a.idx).ToArray();
                int predictedLabel = predictedLabels.Length == 1 ? predictedLabels[0] : predictedLabels[Rand.Int(predictedLabels.Length)];
                int goldLabel = kvp.Value.Value;

                confusionMatrix[goldLabel, predictedLabel] = confusionMatrix[goldLabel, predictedLabel] + 1.0;

                if (goldLabel == predictedLabel)
                    correct++;

                var nlp = -trueLabel.GetLogProb(goldLabel);
                if (nlp > nlpdThreshold)
                    nlp = nlpdThreshold;
                logProb += nlp;
            }

            Accuracy = correct / (double)goldX;
            NegativeLogProb = logProb / (double)goldX;
            ModelConfusionMatrix = confusionMatrix;

            // Compute average recall
            double sumRec = 0;
            for (int i = 0; i < labelCount; i++)
            {
                double classSum = 0;
                for (int j = 0; j < labelCount; j++)
                {
                    classSum += confusionMatrix[i, j];
                }

                sumRec += confusionMatrix[i, i] / classSum;
            }
            AvgRecall = sumRec / labelCount;
        }

        /// <summary>
        /// Writes out the mean of an uncertain confusion matrix to a StreamWriter.
        /// The confusion matrix is passed as an of Dirichlets, one for each row
        /// of the confusion matrix (as given by the posteriors from the model).
        /// </summary>
        /// <param name="writer">A| StreamWriter instance.</param>
        /// <param name="worker">The worker id.</param>
        /// <param name="confusionMatrix">The confusion matrix</param>
        private static void WriteConfusionMatrix(StreamWriter writer, string worker, Dirichlet[] confusionMatrix)
        {
            int labelCount = confusionMatrix.Length;
            var meanConfusionMatrix = confusionMatrix.Select(cm => cm.GetMean()).ToArray();
            var printableConfusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => meanConfusionMatrix[i][j]);
            WriteWorkerConfusionMatrix(writer, worker, printableConfusionMatrix);
        }

        /// <summary>
        /// Writes the a confusion matrix to a stream writer.
        /// </summary>
        /// <param name="writer">A StreamWriter instance.</param>
        /// <param name="worker">The worker id.</param>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        private static void WriteWorkerConfusionMatrix(StreamWriter writer, string worker, double[,] confusionMatrix)
        {
            int labelCount = confusionMatrix.GetLength(0);
            writer.WriteLine(worker);
            for (int j = 0; j < labelCount; j++)
                writer.Write(",{0}", j);
            writer.WriteLine();

            for (int i = 0; i < labelCount; i++)
            {
                writer.Write(i);
                for (int j = 0; j < labelCount; j++)
                    writer.Write(",{0:0.0000}", confusionMatrix[i, j]);

                writer.WriteLine();
            }
        }

        /// <summary>
        /// Writes various results to a StreamWriter.
        /// </summary>
        /// <param name="writer">A StreamWriter instance.</param>
        /// <param name="writeCommunityParameters">Set true to write community parameters.</param>
        /// <param name="writeWorkerParameters">Set true to write worker parameters.</param>
        /// <param name="writeWorkerCommunities">Set true to write worker communities.</param>
        public void WriteResults(StreamWriter writer, bool writeCommunityParameters, bool writeWorkerParameters, bool writeWorkerCommunities)
        {
            this.WriteAccuracy(writer);

            if (writeCommunityParameters && this.CommunityConfusionMatrix != null)
            {
                for (int communityIndex = 0; communityIndex < this.CommunityConfusionMatrix.Length; communityIndex++)
                {
                    WriteConfusionMatrix(writer, "Community" + communityIndex, this.CommunityConfusionMatrix[communityIndex]);
                }
            }

            if (writeWorkerParameters && this.WorkerConfusionMatrix != null)
            {
                foreach (var kvp in this.WorkerConfusionMatrix)
                {
                    WriteConfusionMatrix(writer, kvp.Key, kvp.Value);
                }
            }

            if (writeWorkerCommunities && this.WorkerCommunity != null)
            {
                foreach (var kvp in this.WorkerCommunity)
                {
                    writer.WriteLine(string.Format("{0}:\t{1}", kvp.Key, kvp.Value));
                }
            }

            writer.WriteLine("Log Evidence = {0:0.0000}", ModelEvidence.LogOdds);
        }

        /// <summary>
        /// Writes the accuracy results on the StreamWriter.
        /// </summary>
        /// <param name="writer">The StreamWriter.</param>
        public void WriteAccuracy(StreamWriter writer)
        {
            writer.WriteLine("Accuracy = {0:0.000}", this.Accuracy);
            writer.WriteLine("Mean negative log prob density = {0:0.000}", this.NegativeLogProb);
            WriteWorkerConfusionMatrix(writer, "Model confusion matrix", this.ModelConfusionMatrix);
        }
    }

    /// <summary>
    /// This class represents a single datum, and has methods to read in data.
    /// </summary>
    public class Datum
    {
        /// <summary>
        /// The worker id.
        /// </summary>
        public string WorkerId;

        /// <summary>
        /// The task id.
        /// </summary>
        public string TaskId;

        /// <summary>
        /// The worker's label.
        /// </summary>
        public int WorkerLabel;

        /// <summary>
        /// The task's gold label (optional).
        /// </summary>
        public int? GoldLabel;

        /// <summary>
        /// Loads the data file in the format (worker id, task id, worker label, ?gold label).
        /// </summary>
        /// <param name="filename">The data file.</param>
        /// <returns>The list of parsed data.</returns>
        public static IList<Datum> LoadData(string filename)
        {
            var result = new List<Datum>();
            using (var reader = new StreamReader(filename))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var strarr = line.Split(',');
                    int length = strarr.Length;
                    //if (length < 3 || length > 4) //Filter bad entries!!
                    //    continue;

                    int workerLabel = int.Parse(strarr[2]);
                    //if (workerLabel < -4 || workerLabel > 4) //Filter bad entries!!
                    //    continue;

                    var datum = new Datum()
                    {
                        WorkerId = strarr[0],
                        TaskId = strarr[1],
                        WorkerLabel = workerLabel,
                    };

                    if (length == 4)
                        datum.GoldLabel = int.Parse(strarr[3]);
                    else
                        datum.GoldLabel = null;

                    result.Add(datum);
                }
            }

            return result;
        }
    }

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
            private set;
        }

        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount
        {
            get
            {
                return LabelMax - LabelMin + 1;
            }
        }

        /// <summary>
        /// The number of workers.
        /// </summary>
        public int WorkerCount
        {
            get
            {
                return WorkerIndexToId.Length;
            }
        }

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount
        {
            get
            {
                return TaskIndexToId.Length;
            }
        }

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
            var labels = data.Select(d => d.WorkerLabel).Distinct().OrderBy(lab => lab).ToArray();

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
              Select(t => t.GroupBy(d => d.GoldLabel).Where(d => d.Key != null)).
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
                  else
                  {
                      return new Tuple<string, int?>(datum.TaskId, (int?)null);
                  }
              }).ToDictionary(tup => tup.Item1, tup => tup.Item2);
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
                      else
                      {
                          return null;
                      }
                  }).ToArray();
        }

        /// <summary>
        /// For each task, gets the empirical label distribution.
        /// </summary>
        /// <returns></returns>
        public Discrete[] GetVoteDistribPerTaskIndex()
        {
            return Data.GroupBy(d => TaskIdToIndex[d.TaskId]).
              OrderBy(g => g.Key).
              Select(t => t.GroupBy(d => d.WorkerLabel - LabelMin).
                  Select(g => new
                  {
                      label = g.Key,
                      count = g.Count()
                  })).
                  Select(arr =>
                  {
                      Vector v = Vector.Zero(LabelCount);
                      foreach (var a in arr)
                          v[a.label] = (double)a.count;
                      return new Discrete(v);
                  }).ToArray();
        }
    }

    /// <summary>
    /// The BCC model class.
    /// </summary>
    public class BCC
    {
        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount
        {
            get
            {
                return c == null ? 0 : c.SizeAsInt;
            }
        }

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount
        {
            get
            {
                return n == null ? 0 : n.SizeAsInt;
            }
        }

        // Ranges
        protected Range n;
        protected Range k;
        protected Range c;
        protected Range kn;

        // Variables in the model
        protected Variable<int> WorkerCount;
        protected VariableArray<int> TrueLabel;
        protected VariableArray<int> WorkerTaskCount;
        protected VariableArray<VariableArray<int>, int[][]> WorkerTaskIndex;
        protected VariableArray<VariableArray<int>, int[][]> WorkerLabel;
        protected Variable<Vector> BackgroundLabelProb;
        protected VariableArray<VariableArray<Vector>, Vector[][]> WorkerConfusionMatrix;
        protected Variable<bool> Evidence;

        // Prior distributions
        protected Variable<Dirichlet> BackgroundLabelProbPrior;
        protected VariableArray<VariableArray<Dirichlet>, Dirichlet[][]> ConfusionMatrixPrior;
        protected VariableArray<Discrete> TrueLabelConstraint;
        protected Variable<Bernoulli> EvidencePrior;

        // Inference engine
        protected InferenceEngine Engine;

        // Hyperparameters and inference settings
        public double InitialWorkerBelief
        {
            get;
            set;
        }

        /// <summary>
        /// The number of inference iterations.
        /// </summary>
        public int NumberOfIterations
        {
            get;
            set;
        }

        /// <summary>
        /// Creates a BCC model instance.
        /// </summary>
        public BCC()
        {
            InitialWorkerBelief = 0.5;
            NumberOfIterations = 35;
            EvidencePrior = new Bernoulli(0.5);
        }

        /// <summary>
        /// Initializes the ranges, the generative process and the inference engine of the BCC model.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        public virtual void CreateModel(int taskCount, int labelCount)
        {
            Evidence = Variable<bool>.Random(this.EvidencePrior);
            var evidenceBlock = Variable.If(Evidence);
            DefineVariablesAndRanges(taskCount, labelCount);
            DefineGenerativeProcess();
            DefineInferenceEngine();
            evidenceBlock.CloseBlock();
        }

        /// <summary>
        /// Initializes the ranges of the variables.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        protected virtual void DefineVariablesAndRanges(int taskCount, int labelCount)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");
            n = new Range(taskCount).Named("n");
            c = new Range(labelCount).Named("c");
            k = new Range(WorkerCount).Named("k");

            // The tasks for each worker
            WorkerTaskCount = Variable.Array<int>(k).Named("WorkerTaskCount");
            kn = new Range(WorkerTaskCount[k]).Named("kn");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerTaskIndex");
            WorkerTaskIndex.SetValueRange(n);
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");

            // The background probability vector
            BackgroundLabelProbPrior = Variable.New<Dirichlet>().Named("BackgroundLabelProbPrior");
            BackgroundLabelProb = Variable<Vector>.Random(BackgroundLabelProbPrior).Named("BackgroundLabelProb");
            BackgroundLabelProb.SetValueRange(c);

            // The confusion matrices for each worker
            ConfusionMatrixPrior = Variable.Array(Variable.Array<Dirichlet>(c), k).Named("ConfusionMatrixPrior");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("ConfusionMatrix");
            WorkerConfusionMatrix[k][c] = Variable<Vector>.Random(ConfusionMatrixPrior[k][c]);
            WorkerConfusionMatrix.SetValueRange(c);

            // The unobserved 'true' label for each task
            TrueLabel = Variable.Array<int>(n).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("Truth");
            TrueLabelConstraint = Variable.Array<Discrete>(n).Named("TruthConstraint");
            // Constraint for online learning
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);
            Variable.ConstrainEqualRandom(TrueLabel[n], TrueLabelConstraint[n]);
            // The worker labels
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");
        }

        /// <summary>
        /// Defines the BCC generative process.
        /// </summary>
        protected virtual void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
            using (Variable.ForEach(k))
            {
                var trueLabel = Variable.Subarray(TrueLabel, WorkerTaskIndex[k]);
                trueLabel.SetValueRange(c);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][trueLabel[kn]]);
                    }
                }
            }
        }

        /// <summary>
        /// Initializes the BCC inference engine.
        /// </summary>
        protected virtual void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new ExpectationPropagation());
            Engine.Compiler.UseParallelForLoops = true;
            Engine.ShowProgress = false;
            Engine.Compiler.WriteSourceFiles = false;
        }

        /// <summary>
        /// Sets the priors of BCC.
        /// </summary>
        /// <param name="workerCount">The number of workers.</param>
        /// <param name="priors">The priors.</param>
        protected virtual void SetPriors(int workerCount, BCCPosteriors priors)
        {
            int numClasses = c.SizeAsInt;
            WorkerCount.ObservedValue = workerCount;
            if (priors == null)
            {
                BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(numClasses);
                var confusionMatrixPrior = GetConfusionMatrixPrior();
                ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(workerCount, worker => Util.ArrayInit(numClasses, lab => confusionMatrixPrior[lab]));
                TrueLabelConstraint.ObservedValue = Util.ArrayInit(TaskCount, t => Discrete.Uniform(numClasses));
            }
            else
            {
                BackgroundLabelProbPrior.ObservedValue = priors.BackgroundLabelProb;
                ConfusionMatrixPrior.ObservedValue = priors.WorkerConfusionMatrix;
                TrueLabelConstraint.ObservedValue = priors.TrueLabelConstraint;
            }
        }

        /// <summary>
        /// Attachs the data to the workers labels.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        protected virtual void AttachData(int[][] taskIndices, int[][] workerLabels)
        {
            AttachData(taskIndices, workerLabels, null);
        }

        /// <summary>
        /// Attachs the data to the workers labels with and sets the workers' confusion matrix priors.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="confusionMatrixPrior">The workers' confusion matrix priors.</param>
        protected virtual void AttachData(int[][] taskIndices, int[][] workerLabels, Dirichlet[][] confusionMatrixPrior)
        {
            int numClasses = c.SizeAsInt;
            WorkerCount.ObservedValue = taskIndices.Length;
            WorkerTaskCount.ObservedValue = taskIndices.Select(tasks => tasks.Length).ToArray();
            WorkerTaskIndex.ObservedValue = taskIndices;
            // Prediction mode is indicated by none of the workers having a label.
            // We can just look at the first one
            if (workerLabels[0] != null)
            {
                WorkerLabel.ObservedValue = workerLabels;
            }
            else
            {
                WorkerLabel.ClearObservedValue();
            }

            if (confusionMatrixPrior != null)
            {
                ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(confusionMatrixPrior.Length, worker => Util.ArrayInit(numClasses, lab => confusionMatrixPrior[worker][lab]));
            }
        }

        /// <summary>
        /// Infers the posteriors of BCC using the attached data and priors.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="priors">The priors of the BCC parameters.</param>
        /// <returns></returns>
        public virtual BCCPosteriors Infer(int[][] taskIndices, int[][] workerLabels, BCCPosteriors priors)
        {
            int workerCount = workerLabels.Length;
            SetPriors(workerCount, priors);
            AttachData(taskIndices, workerLabels, null);
            var result = new BCCPosteriors();
            Engine.NumberOfIterations = NumberOfIterations;
            result.Evidence = Engine.Infer<Bernoulli>(Evidence);
            result.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
            result.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
            result.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
            result.TrueLabelConstraint = Engine.Infer<Discrete[]>(TrueLabel, QueryTypes.MarginalDividedByPrior);

            // Prediction mode is indicated by none of the workers having a label.
            // We can just look at the first one
            if (workerLabels[0] == null)
            {
                result.WorkerPrediction = Engine.Infer<Discrete[][]>(WorkerLabel);
            }

            return result;
        }

        /// <summary>
        /// Returns the confusion matrix prior of each worker.
        /// </summary>
        /// <returns>The confusion matrix prior of each worker.</returns>
        public Dirichlet[] GetConfusionMatrixPrior()
        {
            var confusionMatrixPrior = new Dirichlet[LabelCount];
            for (int d = 0; d < LabelCount; d++)
            {
                confusionMatrixPrior[d] = new Dirichlet(Util.ArrayInit(LabelCount, i => i == d ? (InitialWorkerBelief / (1 - InitialWorkerBelief)) * (LabelCount - 1) : 1.0));
            }

            return confusionMatrixPrior;
        }
    }


    /// <summary>
    /// The BCC posteriors class.
    /// </summary>
    [Serializable]
    public class BCCPosteriors
    {
        /// <summary>
        /// The probabilities that generate the true labels of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb;

        /// <summary>
        /// The probabilities of the true label of each task.
        /// </summary>
        public Discrete[] TrueLabel;

        /// <summary>
        /// The Dirichlet parameters of the confusion matrix of each worker.
        /// </summary>
        public Dirichlet[][] WorkerConfusionMatrix;

        /// <summary>
        /// The predictive probabilities of the worker's labels.
        /// </summary>
        public Discrete[][] WorkerPrediction;

        /// <summary>
        /// The true label constraint used in online training.
        /// </summary>
        public Discrete[] TrueLabelConstraint;

        /// <summary>
        /// The model evidence.
        /// </summary>
        public Bernoulli Evidence;
    }

    /// <summary>
    /// The CBCC model class.
    /// </summary>
    public class CBCC : BCC
    {
        // Additional ranges
        protected Range m;

        // Additional variables
        protected VariableArray<int> Community;
        protected VariableArray<Discrete> CommunityInit;
        protected Variable<Vector> CommunityProb;
        protected VariableArray<VariableArray<Vector>, Vector[][]> ScoreMatrix;
        protected VariableArray<VariableArray<Vector>, Vector[][]> CommunityScoreMatrix;
        protected VariableArray<VariableArray<Vector>, Vector[][]> CommunityConfusionMatrix;
        protected Variable<PositiveDefiniteMatrix> NoiseMatrix = Variable.New<PositiveDefiniteMatrix>().Named("NoiseMatrix");

        // Additional priors
        protected VariableArray<Discrete> CommunityConstraint;
        protected VariableArray<VariableArray<VectorGaussian>, VectorGaussian[][]> ScoreMatrixConstraint;
        protected VariableArray<VariableArray<VectorGaussian>, VectorGaussian[][]> CommunityScoreMatrixPrior;
        protected Variable<Dirichlet> CommunityProbPrior;

        /// <summary>
        /// The noise precision that generates the workers score matrix from the communities score matrix.
        /// </summary>
        public double NoisePrecision
        {
            get;
            set;
        }

        /// <summary>
        /// The number of communities.
        /// </summary>
        public int CommunityCount
        {
            get;
            protected set;
        }

        /// <summary>
        /// The mean vector of the Gaussian distribution generating the community score matrices.
        /// </summary>
        public Tuple<double, double>[] ScoreMeanParameters
        {
            get;
            set;
        }

        /// <summary>
        /// The precision matrix of the Gaussian distribution generating the community score matrices.
        /// </summary>
        public double[] ScorePrecisionParameters
        {
            get;
            set;
        }

        /// <summary>
        /// The hyperparameter governing community membership.
        /// </summary>
        public double CommunityPseudoCount
        {
            get;
            set;
        }

        /// <summary>
        /// The prior for the score matrices.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrixPriorObserved
        {
            get;
            protected set;
        }

        /// <summary>
        /// The prior for community membership.
        /// </summary>
        public Dirichlet CommunityProbPriorObserved
        {
            get;
            protected set;
        }

        /// <summary>
        /// Creates a CBCC model instance.
        /// </summary>
        public CBCC()
            : base()
        {
            NoisePrecision = 5;
            CommunityCount = 3;
            CommunityPseudoCount = 10.0;
            ScoreMeanParameters = null;
            ScorePrecisionParameters = null;
        }

        /// <summary>
        /// Initializes the CBCC model.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        public override void CreateModel(int taskCount, int labelCount)
        {
            CreateModel(taskCount, labelCount, CommunityCount);
        }

        /// <summary>
        /// Initializes the CBCC model with a number of communities.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        /// <param name="communityCount">The number of communities.</param>
        public virtual void CreateModel(int taskCount, int labelCount, int communityCount)
        {
            Evidence = Variable<bool>.Random(this.EvidencePrior);
            var evidenceBlock = Variable.If(Evidence);
            CommunityCount = communityCount;
            CommunityProbPriorObserved = Dirichlet.Symmetric(communityCount, CommunityPseudoCount);
            DefineVariablesAndRanges(taskCount, labelCount);
            DefineGenerativeProcess();
            DefineInferenceEngine();
            evidenceBlock.CloseBlock();

            if (ScoreMeanParameters == null)
            {
                var scoreMatrixPrior = GetScoreMatrixPrior();
                CommunityScoreMatrixPriorObserved = Util.ArrayInit(CommunityCount, comm => Util.ArrayInit(labelCount, lab => new VectorGaussian(scoreMatrixPrior[lab])));
            }
            else
            {
                CommunityScoreMatrixPriorObserved = Util.ArrayInit(
                    CommunityCount,
                    comm => Util.ArrayInit(
                        labelCount, lab => VectorGaussian.FromMeanAndPrecision(
                            Vector.FromArray(
                            Util.ArrayInit(labelCount, lab1 => lab == lab1 ? ScoreMeanParameters[comm].Item1 : ScoreMeanParameters[comm].Item2)),
                            PositiveDefiniteMatrix.IdentityScaledBy(LabelCount, ScorePrecisionParameters[comm]))));
            }
        }

        /// <summary>
        /// Defines the variables and the ranges of CBCC.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        protected override void DefineVariablesAndRanges(int taskCount, int labelCount)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");
            m = new Range(CommunityCount).Named("m");
            n = new Range(taskCount).Named("n");
            c = new Range(labelCount).Named("c");
            k = new Range(WorkerCount).Named("k");

            // The tasks for each worker
            WorkerTaskCount = Variable.Array<int>(k).Named("WorkerTaskCount");
            kn = new Range(WorkerTaskCount[k]).Named("kn");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerTaskIndex");
            WorkerTaskIndex.SetValueRange(n);
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");

            // The background probability vector
            BackgroundLabelProbPrior = Variable.New<Dirichlet>().Named("BackgroundLabelProbPrior");
            BackgroundLabelProb = Variable<Vector>.Random(BackgroundLabelProbPrior).Named("BackgroundLabelProb");
            BackgroundLabelProb.SetValueRange(c);

            // Community membership
            CommunityProbPrior = Variable.New<Dirichlet>().Named("CommunityProbPrior");
            CommunityProb = Variable<Vector>.Random(CommunityProbPrior).Named("CommunityProb");
            CommunityProb.SetValueRange(m);
            Community = Variable.Array<int>(k).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("Community");
            CommunityConstraint = Variable.Array<Discrete>(k).Named("CommunityConstraint");
            Community[k] = Variable.Discrete(CommunityProb).ForEach(k);
            Variable.ConstrainEqualRandom(Community[k], CommunityConstraint[k]);
            // Initialiser to break symmetry for community membership
            CommunityInit = Variable.Array<Discrete>(k).Named("CommunityInit");
            Community[k].InitialiseTo(CommunityInit[k]);

            // Community parameters
            CommunityScoreMatrixPrior = Variable.Array(Variable.Array<VectorGaussian>(c), m).Named("CommunityScoreMatrixPrior");
            CommunityScoreMatrix = Variable.Array(Variable.Array<Vector>(c), m).Named("CommunityScoreMatrix");
            CommunityScoreMatrix[m][c] = Variable<Vector>.Random(CommunityScoreMatrixPrior[m][c]);
            CommunityConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), m).Named("CommunityConfusionMatrix");
            CommunityConfusionMatrix[m][c] = Variable.Softmax(CommunityScoreMatrix[m][c]);
            CommunityScoreMatrix.SetValueRange(c);

            // Parameters for each worker
            ScoreMatrix = Variable.Array(Variable.Array<Vector>(c), k).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("ScoreMatrix");
            ScoreMatrixConstraint = Variable.Array(Variable.Array<VectorGaussian>(c), k).Named("ScoreMatrixConstraint");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("ConfusionMatrix");

            // The unobserved 'true' label for each task
            TrueLabel = Variable.Array<int>(n).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("Truth");
            TrueLabelConstraint = Variable.Array<Discrete>(n).Named("TruthConstraint");
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);
            Variable.ConstrainEqualRandom(TrueLabel[n], TrueLabelConstraint[n]);

            // The labels given by the workers
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");
        }

        /// <summary>
        /// Defines the generative process of CBCC.
        /// </summary>
        protected override void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
            using (Variable.ForEach(k))
            {
                using (Variable.Switch(Community[k]))
                {
                    ScoreMatrix[k][c] = Variable.VectorGaussianFromMeanAndPrecision(CommunityScoreMatrix[Community[k]][c], NoiseMatrix);
                }

                Variable.ConstrainEqualRandom(ScoreMatrix[k][c], ScoreMatrixConstraint[k][c]);
                WorkerConfusionMatrix[k][c] = Variable.Softmax(ScoreMatrix[k][c]);
                var trueLabel = Variable.Subarray(TrueLabel, WorkerTaskIndex[k]);
                trueLabel.SetValueRange(c);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][trueLabel[kn]]);
                    }
                }
            }
        }

        /// <summary>
        /// Initializes the CBCC inference engine.
        /// </summary>
        protected override void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new VariationalMessagePassing());
            Engine.ShowProgress = false;
            Engine.Compiler.UseParallelForLoops = true;
            Engine.Compiler.GivePriorityTo(typeof(SoftmaxOp_BL06));
            Engine.Compiler.WriteSourceFiles = false;
        }

        /// <summary>
        /// Attachs the data to the workers labels.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        protected override void AttachData(int[][] taskIndices, int[][] workerLabels)
        {
            AttachData(taskIndices, workerLabels, null, null);
        }

        /// <summary>
        /// Attachs the data to the workers labels and sets the constraints on the community score matrices and
        /// the community memberships (used for online training).
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="scoreConstraint">The constraint of the community score matrices.</param>
        /// <param name="communityConstraint">The constraint of the workers community membership.</param>
        protected void AttachData(int[][] taskIndices, int[][] workerLabels, VectorGaussian[][] scoreConstraint, Discrete[] communityConstraint)
        {
            int communityCount = m.SizeAsInt;
            int workerCount = workerLabels.Length;
            int labelCount = c.SizeAsInt;
            base.AttachData(taskIndices, workerLabels);
            CommunityInit.ObservedValue = Util.ArrayInit(workerCount, worker => Discrete.PointMass(Rand.Int(communityCount), communityCount));

            if (scoreConstraint != null)
            {
                ScoreMatrixConstraint.ObservedValue = scoreConstraint;
            }
            else
            {
                ScoreMatrixConstraint.ObservedValue = Util.ArrayInit(workerCount, w => Util.ArrayInit(labelCount, lab => VectorGaussian.Uniform(labelCount)));
            }

            if (communityConstraint != null)
            {
                CommunityConstraint.ObservedValue = communityConstraint;
            }
            else
            {
                CommunityConstraint.ObservedValue = Util.ArrayInit(workerCount, w => Discrete.Uniform(communityCount));
            }
        }

        /// <summary>
        /// Sets the priors of CBCC.
        /// </summary>
        /// <param name="workerCount">The number of workers.</param>
        /// <param name="priors">The priors.</param>
        protected override void SetPriors(int workerCount, BCCPosteriors priors)
        {
            int communityCount = m.SizeAsInt;
            int labelCount = c.SizeAsInt;
            WorkerCount.ObservedValue = workerCount;
            NoiseMatrix.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(labelCount, NoisePrecision);
            CBCCPosteriors cbccPriors = (CBCCPosteriors)priors;

            if (cbccPriors == null || cbccPriors.BackgroundLabelProb == null)
                BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(labelCount);
            else
                BackgroundLabelProbPrior.ObservedValue = cbccPriors.BackgroundLabelProb;

            if (cbccPriors == null || cbccPriors.CommunityProb == null)
                CommunityProbPrior.ObservedValue = CommunityProbPriorObserved;
            else
                CommunityProbPrior.ObservedValue = cbccPriors.CommunityProb;

            if (cbccPriors == null || cbccPriors.CommunityScoreMatrix == null)
                CommunityScoreMatrixPrior.ObservedValue = CommunityScoreMatrixPriorObserved;
            else
                CommunityScoreMatrixPrior.ObservedValue = cbccPriors.CommunityScoreMatrix;

            if (cbccPriors == null || cbccPriors.TrueLabelConstraint == null)
                TrueLabelConstraint.ObservedValue = Util.ArrayInit(TaskCount, t => Discrete.Uniform(labelCount));
            else
                TrueLabelConstraint.ObservedValue = cbccPriors.TrueLabelConstraint;
        }

        /// <summary>
        /// Infers the posteriors of CBCC using the attached data.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="priors">The priors.</param>
        /// <returns></returns>
        public override BCCPosteriors Infer(int[][] taskIndices, int[][] workerLabels, BCCPosteriors priors)
        {
            var cbccPriors = (CBCCPosteriors)priors;
            VectorGaussian[][] scoreConstraint = (cbccPriors == null ? null : cbccPriors.WorkerScoreMatrixConstraint);
            Discrete[] communityConstraint = (cbccPriors == null ? null : cbccPriors.WorkerCommunityConstraint);
            SetPriors(workerLabels.Length, priors);
            AttachData(taskIndices, workerLabels, scoreConstraint, communityConstraint);
            var result = new CBCCPosteriors();
            Engine.NumberOfIterations = NumberOfIterations;
            result.Evidence = Engine.Infer<Bernoulli>(Evidence);
            result.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
            result.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
            result.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
            result.TrueLabelConstraint = Engine.Infer<Discrete[]>(TrueLabel, QueryTypes.MarginalDividedByPrior);
            result.CommunityScoreMatrix = Engine.Infer<VectorGaussian[][]>(CommunityScoreMatrix);
            result.CommunityConfusionMatrix = Engine.Infer<Dirichlet[][]>(CommunityConfusionMatrix);
            result.WorkerScoreMatrixConstraint = Engine.Infer<VectorGaussian[][]>(ScoreMatrix, QueryTypes.MarginalDividedByPrior);
            result.CommunityProb = Engine.Infer<Dirichlet>(CommunityProb);
            result.Community = Engine.Infer<Discrete[]>(Community);
            result.WorkerCommunityConstraint = Engine.Infer<Discrete[]>(Community, QueryTypes.MarginalDividedByPrior);
            return result;
        }

        /// <summary>
        /// Returns the community score matrix prior.
        /// </summary>
        /// <returns>The community score matrix prior.</returns>
        private VectorGaussian[] GetScoreMatrixPrior()
        {
            var dim = new Range(LabelCount);
            var mean = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(LabelCount), PositiveDefiniteMatrix.IdentityScaledBy(LabelCount, 1));
            var prec = Variable.WishartFromShapeAndRate(1.0, PositiveDefiniteMatrix.IdentityScaledBy(LabelCount, 1));
            var score = Variable.VectorGaussianFromMeanAndPrecision(mean, prec);
            var confusion = Variable.Softmax(score);
            confusion.SetValueRange(dim);
            var confusionConstraint = Variable.New<Dirichlet>();
            Variable.ConstrainEqualRandom(confusion, confusionConstraint);
            var engine = new InferenceEngine(new VariationalMessagePassing())
            {
                ShowProgress = false
            };

            engine.Compiler.WriteSourceFiles = false;
            var scorePrior = new VectorGaussian[LabelCount];
            for (int d = 0; d < LabelCount; d++)
            {
                confusionConstraint.ObservedValue = new Dirichlet(Util.ArrayInit(LabelCount, i => i == d ? (InitialWorkerBelief / (1 - InitialWorkerBelief)) * (LabelCount - 1) : 1.0));
                scorePrior[d] = engine.Infer<VectorGaussian>(score);
            }

            return scorePrior;
        }
    }

    /// <summary>
    /// CBCC posterior object.
    /// </summary>
    [Serializable]
    public class CBCCPosteriors : BCCPosteriors
    {
        /// <summary>
        /// The Dirichlet posteriors of the workers community membership.
        /// </summary>
        public Dirichlet CommunityProb;

        /// <summary>
        /// The posterior probabilities of the workers community membnerships.
        /// </summary>
        public Discrete[] Community;

        /// <summary>
        /// The Dirichlet posteriors of the community confusion matrix.
        /// </summary>
        public Dirichlet[][] CommunityConfusionMatrix;

        /// <summary>
        /// The Gaussian posteriors of the community score matrix.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrix;
        
        /// <summary>
        /// The Gaussian constraint of the community score matrix (used for online training).
        /// </summary>
        public VectorGaussian[][] WorkerScoreMatrixConstraint;

        /// <summary>
        /// Theconstraint of the workers community membership (used for online training).
        /// </summary>
        public Discrete[] WorkerCommunityConstraint;
    }
}

