// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

/* Community-Based Bayesian Aggregation for Crowdsoucing
* 
* Software to run the experiment presented in the paper "Community-Based Bayesian Aggregation Models for Crowdsourcing" by Venanzi et. al, WWW14
* To run it, you must create csv file with your data with the format <Worker id, Task id, worker's label, (optional) task's gold label>.
* See CF.csv for an example.
* You can download the original CF data set used in the paper from www.crowdscale.org
*/

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Crowdsourcing
{
    /// <summary>
    /// The class for the main program.
    /// </summary>
    class Crowdsourcing
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
                    case 4: RunWWWActiveLearning(GoldDatasets[ds], RunType.CBCC, TaskSelectionMethod.EntropyTask, new CommunityModel(), NumCommunities[ds]); break;
                    default: // Run all
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.MajorityVote, TaskSelectionMethod.EntropyTask, null);
                        if (RunDawidSkene)
                        {
                            RunWWWActiveLearning(GoldDatasets[ds], RunType.DawidSkene, TaskSelectionMethod.EntropyTask, null);
                        }
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.BCC, TaskSelectionMethod.EntropyTask, new BCC());
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.CBCC, TaskSelectionMethod.EntropyTask, new CommunityModel(), NumCommunities[ds]);
                        RunWWWActiveLearning(GoldDatasets[ds], RunType.BCC, TaskSelectionMethod.EntropyTask, new CommunityModel());
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
            var workerSelectionMethod = WorkerSelectionMethod.RandomWorker;
            var data = Datum.LoadData(@"Data\" + dataSet + ".csv");
            string modelName = GetModelName(dataSet, runType, taskSelectionMethod, workerSelectionMethod, communityCount);
            ActiveLearning.RunActiveLearning(data, modelName, runType, model, taskSelectionMethod, workerSelectionMethod, ResultsDir, communityCount);
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
                RunGold(GoldDatasets[ds], RunType.CBCC, new CommunityModel(), NumCommunities[ds]); Console.Write(".");
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
            var modelEvidence = Util.ArrayInit<double>(communityUpperBound, endIndex + 1, (i, j) => 0.0);
            for (int ds = startIndex; ds <= endIndex; ds++)
            {
                Console.WriteLine("Dataset: " + GoldDatasets[ds]);
                for (int communityCount = 1; communityCount <= communityUpperBound; communityCount++)
                {
                    Results results = RunGold(GoldDatasets[ds], RunType.CBCC, new CommunityModel(), communityCount);
                    modelEvidence[communityCount - 1, ds] = results.ModelEvidence.LogOdds;
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
                    results.RunBCC(ResultsDir + modelName, data, data, model, Results.RunMode.ClearResults, false, communityCount, false, false);
                    break;
            }

            // Write the inference results on a csv file
            using (StreamWriter writer = new StreamWriter(ResultsDir + "endpoints.csv", true))
            {
                writer.WriteLine("{0},{1:0.000},{2:0.0000}", modelName, results.Accuracy, results.NegativeLogProb);
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
}

