// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

/* Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning
* 
* Software to run the experiment presented in the paper "Language Understanding in the Wind: Combining Crowdsourcing and Machine Learning" by Simpsons et. al, WWW15
To run it:
- Replace <your-data-file> with a TSV with fields <WorkerId, TaskId, Worker label, Text, Gold label (optional)
- Replace <your-stop-words-file> with a TSV with the list of stop words, one for each line
*/

namespace BCCWordsRelease
{
    using EnglishStemmer;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Utilities;
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Text.RegularExpressions;

    /// <summary>
    /// This class runs the experiment presented in
    /// </summary>
    class WWW15_BCCWordsExperiment
    {
        /// <summary>
        /// Main method to run the crowdsourcing experiments presented in Simpsons et.al (WWW15).
        /// </summary>
        public static void Main()
        {
            var data = Datum.LoadData(@"<your-data-file>");

            // Run model and get results
            var VocabularyOnSubData = ResultsWords.BuildVocabularyOnSubdata((List<Datum>)data);

            BCCWords model = new BCCWords();
            ResultsWords resultsWords = new ResultsWords(data, VocabularyOnSubData);
            DataMappingWords mapping = resultsWords.Mapping as DataMappingWords;

            if (mapping != null)
            {
                resultsWords = new ResultsWords(data, VocabularyOnSubData);
                resultsWords.RunBCCWords("BCCwords", data, data, model, Results.RunMode.ClearResults, true);
            }

            using (var writer = new StreamWriter(Console.OpenStandardOutput()))
            {
                resultsWords.WriteResults(writer, false, false, false, true);
            }
        }
    }

    /// <summary>
    /// Results class containing posteriors and predictions of BCCWords.
    /// </summary>
    public class ResultsWords : Results
    {
        /// <summary>
        /// The posterior of the word probabilities for each true label.
        /// </summary>
        public Dirichlet[] ProbWords
        {
            get;
            private set;
        }

        /// <summary>
        /// The vocabulary
        /// </summary>
        public List<string> Vocabulary
        {
            get;
            set;
        }

        /// <summary>
        /// Creates an object for storing the inference results of BCCWords
        /// </summary>
        /// <param name="data">The data</param>
        /// <param name="vocabulary">The vocabulary</param>
        public ResultsWords(IList<Datum> data, List<string> vocabulary)
        {
            if (vocabulary == null)
            {
                // Build vocabulary
                Console.Write("Building vocabulary...");
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                string[] corpus = data.Select(d => d.BodyText).Distinct().ToArray();
                Vocabulary = BuildVocabularyFromCorpus(corpus);
                Console.WriteLine("done. Elapsed time: {0}", stopwatch.Elapsed);
            }

            // Build data mapping
            Vocabulary = vocabulary;
            this.Mapping = new DataMappingWords(data, vocabulary);
            this.GoldLabels = Mapping.GetGoldLabelsPerTaskId();
        }

        /// <summary>
        /// Runs the majority vote method on the data.
        /// </summary>
        /// <param name="modelName"></param>
        /// <param name="data">The data</param>
        /// <param name="mode"></param>
        /// <param name="calculateAccuracy">Compute the accuracy (true).</param>
        /// <param name="fullData"></param>
        /// <param name="model"></param>
        /// <param name="useMajorityVote"></param>
        /// <param name="useRandomLabel"></param>
        /// <returns>The updated results</returns>
        public void RunBCCWords(string modelName,
            IList<Datum> data,
            IList<Datum> fullData,
            BCCWords model,
            RunMode mode,
            bool calculateAccuracy,
            bool useMajorityVote = false,
            bool useRandomLabel = false)
        {
            DataMappingWords MappingWords = null;
            if (FullMapping == null)
                FullMapping = new DataMapping(fullData);

            if (Mapping == null)
            {
                // Build vocabulary
                Console.Write("Building vocabulary...");
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                string[] corpus = data.Select(d => d.BodyText).Distinct().ToArray();
                Vocabulary = BuildVocabularyFromCorpus(corpus);
                Console.WriteLine("done. Elapsed time: {0}", stopwatch.Elapsed);

                // Build data mapping
                this.Mapping = new DataMappingWords(data, MappingWords.Vocabulary);
                MappingWords = Mapping as DataMappingWords;
                this.GoldLabels = MappingWords.GetGoldLabelsPerTaskId();
            }

            MappingWords = Mapping as DataMappingWords;
            int[] trueLabels = null;
            if (useMajorityVote)
            {
                if (MappingWords != null)
                {
                    var majorityLabel = MappingWords.GetMajorityVotesPerTaskId(data);
                    trueLabels = Util.ArrayInit(FullMapping.TaskCount, i => majorityLabel.ContainsKey(Mapping.TaskIndexToId[i]) ? (int)majorityLabel[Mapping.TaskIndexToId[i]] : Rand.Int(Mapping.LabelMin, Mapping.LabelMax + 1));
                    data = MappingWords.BuildDataFromAssignedLabels(majorityLabel, data);
                }
            }

            if (useRandomLabel)
            {
                var randomLabels = MappingWords.GetRandomLabelPerTaskId(data);
                data = MappingWords.BuildDataFromAssignedLabels(randomLabels, data);
            }

            var labelsPerWorkerIndex = MappingWords.GetLabelsPerWorkerIndex(data);
            var taskIndicesPerWorkerIndex = MappingWords.GetTaskIndicesPerWorkerIndex(data);

            // Create model
            ClearResults();
            model.CreateModel(MappingWords.TaskCount, MappingWords.LabelCount, MappingWords.WordCount);

            // Run model inference
            BCCWordsPosteriors posteriors = model.InferPosteriors(labelsPerWorkerIndex, taskIndicesPerWorkerIndex, MappingWords.WordIndicesPerTaskIndex, MappingWords.WordCountsPerTaskIndex, trueLabels);

            // Update results
            UpdateResults(posteriors, mode);

            // Compute accuracy
            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }
        }

        /// <summary>
        /// Select high TFIDF terms
        /// </summary>
        /// <param name="corpus">array of terms</param>
        /// <param name="tfidf_threshold">TFIDF threshold</param>
        /// <returns></returns>
        private static List<string> BuildVocabularyFromCorpus(string[] corpus, double tfidf_threshold = 0.8)
        {
            List<string> vocabulary;
            double[][] inputs = TFIDFClass.Transform(corpus, out vocabulary, 0);
            inputs = TFIDFClass.Normalize(inputs);

            // Select high TF_IDF terms
            List<string> vocabularyTfidf = new List<string>();
            for (int index = 0; index < inputs.Length; index++)
            {
                var sortedTerms = inputs[index].Select((x, i) => new KeyValuePair<string, double>(vocabulary[i], x)).OrderByDescending(x => x.Value).ToList();
                vocabularyTfidf.AddRange(sortedTerms.Where(entry => entry.Value > tfidf_threshold).Select(k => k.Key).ToList());
            }
            return vocabulary.Distinct().ToList();
        }

        protected override void ClearResults()
        {
            BackgroundLabelProb = Dirichlet.Uniform(Mapping.LabelCount);
            WorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            WorkerPrediction = new Dictionary<string, Dictionary<String, Discrete>>();
            WorkerCommunity = new Dictionary<string, Discrete>();
            TrueLabel = new Dictionary<string, Discrete>();
            PredictedLabel = new Dictionary<string, int?>();
            TrueLabelConstraint = new Dictionary<string, Discrete>();
            CommunityConfusionMatrix = null;
            WorkerScoreMatrixConstraint = new Dictionary<string, VectorGaussian[]>();
            CommunityProb = null;
            CommunityScoreMatrix = null;
            CommunityConstraint = new Dictionary<string, Discrete>();
            LookAheadTrueLabel = new Dictionary<string, Discrete>();
            LookAheadWorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            ModelEvidence = new Bernoulli(0.5);
            ProbWords = null;
        }

        /// <summary>
        /// Writes various results to a StreamWriter.
        /// </summary>
        /// <param name="writer">A StreamWriter instance.</param>
        /// <param name="writeCommunityParameters">Set true to write community parameters.</param>
        /// <param name="writeWorkerParameters">Set true to write worker parameters.</param>
        /// <param name="writeWorkerCommunities">Set true to write worker communities.</param>
        /// <param name="writeProbWords">Set true to write word probabilities</param>
        /// <param name="topWords">Number of words to select</param>
        public void WriteResults(StreamWriter writer, bool writeCommunityParameters, bool writeWorkerParameters, bool writeWorkerCommunities, bool writeProbWords, int topWords = 30)
        {
            base.WriteResults(writer, writeCommunityParameters, writeWorkerCommunities, writeWorkerCommunities);
            DataMappingWords MappingWords = Mapping as DataMappingWords;
            if (writeProbWords && this.ProbWords != null)
            {
                int NumClasses = ProbWords.Length;
                for (int c = 0; c < NumClasses; c++)
                {
                    if (MappingWords != null && MappingWords.WorkerCount > 300) // Assume it's CF
                        writer.WriteLine("Class {0}", MappingWords.CFLabelName[c]);
                    else
                        if (MappingWords != null)
                        writer.WriteLine("Class {0}", MappingWords.SPLabelName[c]);

                    Vector probs = ProbWords[c].GetMean();
                    var probsDictionary = probs.Select((value, index) => new KeyValuePair<string, double>(MappingWords.Vocabulary[index], Math.Log(value))).OrderByDescending(x => x.Value).ToArray();

                    for (int w = 0; w < topWords; w++)
                    {
                        writer.WriteLine($"\t{probsDictionary[w].Key}: \t{probsDictionary[w].Value:0.000}");
                    }
                }
            }
        }

        /// <summary>
        /// Build a vocabulary of terms for a subset of text snippets extracted from the data
        /// </summary>
        /// <param name="data">the data</param>
        /// <returns></returns>
        public static List<string> BuildVocabularyOnSubdata(List<Datum> data)
        {
            Console.WriteLine("Building vocabulary");
            var subData = data.Where((k, i) => i < 20000).ToList();
            string[] corpus = subData.Select(d => d.BodyText).Distinct().ToArray();
            var vocabularyOnSubData = BuildVocabularyFromCorpus(corpus);
            return vocabularyOnSubData.GetRange(0, 300);
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
        /// The body text of the document (optional - only for text sentiment labelling tasks).
        /// </summary>
        public string BodyText;

        /// <summary>
        /// Loads the data file in the format (worker id, task id, worker label, ?gold label).
        /// </summary>
        /// <param name="filename">The data file.</param>
        /// <param name="maxLength"></param>
        /// <returns>The list of parsed data.</returns>
        public static IList<Datum> LoadData(string filename, int maxLength = short.MaxValue)
        {
            var result = new List<Datum>();
            using (var reader = new StreamReader(filename))
            {
                string line;
                while ((line = reader.ReadLine()) != null && result.Count < maxLength)
                {
                    var strarr = line.Split('\t');
                    int length = strarr.Length;

                    var datum = new Datum
                    {
                        WorkerId = strarr[0],
                        TaskId = strarr[1],
                        WorkerLabel = int.Parse(strarr[2]),
                        BodyText = strarr[3]
                    };

                    if (length >= 5)
                        datum.GoldLabel = int.Parse(strarr[4]);
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

            DataWithGold = data.Where(d => d.GoldLabel != null);
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
                datum.BodyText = OriginalData.Where(d => d.TaskId == entry.Key).First().BodyText;

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

    /// <summary>
    /// Data mapping class. This class manages the mapping between the data (which is
    /// in the form of task, worker ids, and labels) and the model data (which is in term of indices).
    /// </summary>
    public class DataMappingWords : DataMapping
    {
        /// <summary>
        /// The vocabulary
        /// </summary>
        public List<string> Vocabulary;

        /// <summary>
        /// The size of the vocabulary.
        /// </summary>
        public int WordCount
        {
            get
            {
                return Vocabulary.Count();
            }
        }

        public int[] WordCountsPerTaskIndex;

        public int[][] WordIndicesPerTaskIndex;

        public string[] CFLabelName = { "Negative", "Neutral", "Positive", "NotRelated", "Unknown" };
        public string[] SPLabelName = { "Negative", "Positive" };

        public DataMappingWords(
            IEnumerable<Datum> data,
            List<string> vocab,
            int[] wordCountPerTaskIndex = null,
            int[][] wordIndicesPerTaskIndex = null,
            bool buildFullMapping = false)
            : base(data)
        {
            Vocabulary = vocab;
            if (wordCountPerTaskIndex == null)
                GetWordIndicesAndCountsPerTaskIndex(data, out WordIndicesPerTaskIndex, out WordCountsPerTaskIndex);
            else
            {
                WordCountsPerTaskIndex = wordCountPerTaskIndex;
                WordIndicesPerTaskIndex = wordIndicesPerTaskIndex;
            }

            if (buildFullMapping) // Use task ids as worker ids
            {
                TaskIndexToId = data.Select(d => d.TaskId).Distinct().ToArray();
                TaskIdToIndex = TaskIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
            }
        }

        /// <summary>
        /// Returns the matrix of the task indices (columns) of each worker (rows).
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="wordIndicesPerTaskIndex">Matrix of word indices for each tash index</param>
        /// <param name="wordCountsPerTaskIndex">Matrix of word counts for each task index</param>
        /// <returns>The matrix of the word indices (columns) of each task (rows).</returns>
        public void GetWordIndicesAndCountsPerTaskIndex(IEnumerable<Datum> data, out int[][] wordIndicesPerTaskIndex, out int[] wordCountsPerTaskIndex)
        {
            wordIndicesPerTaskIndex = new int[TaskCount][];
            wordCountsPerTaskIndex = new int[TaskCount];
            string[] corpus = new string[TaskCount];

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

            foreach (var kvp in groupedRandomisedData)
            {
                corpus[TaskIdToIndex[kvp.Key]] = kvp.Value.First().BodyText;
            }

            wordIndicesPerTaskIndex = TFIDFClass.GetWordIndexStemmedDocs(corpus, Vocabulary);
            wordCountsPerTaskIndex = wordIndicesPerTaskIndex.Select(t => t.Length).ToArray();
        }
    }

    /// <summary>
    ///   Receiver Operating Characteristic (ROC) Curve
    /// </summary>
    /// <remarks>
    ///   In signal detection theory, a receiver operating characteristic (ROC), or simply
    ///   ROC curve, is a graphical plot of the sensitivity vs. (1 − specificity) for a 
    ///   binary classifier system as its discrimination threshold is varied. 
    ///  
    /// References: 
    ///   http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    ///   http://www.anaesthetist.com/mnm/stats/roc/Findex.htm
    ///   http://radiology.rsna.org/content/148/3/839.full.pdf
    /// </remarks>
    public class ReceiverOperatingCharacteristic
    {

        private double area;

        // The actual, measured data
        private double[] measurement;

        // The data, as predicted by a test
        private double[] prediction;


        // The real number of positives and negatives in the measured (actual) data
        private int positiveCount;
        private int negativeCount;

        // The values which represent positive and negative values in our
        //  measurement data (such as presence or absence of some disease)
        double dtrue;
        double dfalse;

        // The collection to hold our curve point information
        public PointCollection collection;

        /// <summary>
        ///   Constructs a new Receiver Operating Characteristic model
        /// </summary>
        /// <param name="measurement">An array of binary values. Tipically 0 and 1, or -1 and 1, indicating negative and positive cases, respectively.</param>
        /// <param name="prediction">An array of continuous values trying to approximate the measurement array.</param>
        public ReceiverOperatingCharacteristic(double[] measurement, double[] prediction)
        {
            this.measurement = measurement;
            this.prediction = prediction;

            // Determine which numbers correspont to each binary category
            dtrue = dfalse = measurement[0];
            for (int i = 1; i < measurement.Length; i++)
            {
                if (dtrue < measurement[i])
                    dtrue = measurement[i];
                if (dfalse > measurement[i])
                    dfalse = measurement[i];
            }

            // Count the real number of positive and negative cases
            for (int i = 0; i < measurement.Length; i++)
            {
                if (measurement[i] == dtrue)
                    this.positiveCount++;
            }

            // Negative cases is just the number of cases minus the number of positives
            this.negativeCount = this.measurement.Length - this.positiveCount;
        }

        #region Public Methods

        /// <summary>
        ///   Computes a ROC curve with 1/increment points
        /// </summary>
        /// <param name="increment">The increment over the previous point for each point in the curve.</param>
        public void Compute(double increment)
        {
            List<Point> points = new List<Point>();
            double cutoff;

            // Create the curve, computing a point for each cutoff value
            for (cutoff = dfalse; cutoff <= dtrue; cutoff += increment)
            {
                points.Add(ComputePoint(cutoff));
            }
            if (cutoff < dtrue) points.Add(ComputePoint(dtrue));

            // Sort the curve by descending specificity
            points.Sort((a, b) => a.Specificity.CompareTo(b.Specificity));

            // Create the point collection
            this.collection = new PointCollection(points.ToArray());

            // Calculate area and error associated with this curve
            this.area = calculateAreaUnderCurve();
            calculateStandardError();
        }


        Point ComputePoint(double threshold)
        {
            int truePositives = 0;
            int trueNegatives = 0;

            for (int i = 0; i < this.measurement.Length; i++)
            {
                bool measured = (this.measurement[i] == dtrue);
                bool predicted = (this.prediction[i] >= threshold);


                // If the prediction equals the true measured value
                if (predicted == measured)
                {
                    // We have a hit. Now we have to see
                    //  if it was a positive or negative hit
                    if (predicted)
                        truePositives++; // Positive hit
                    else trueNegatives++;// Negative hit
                }
            }



            // The other values can be computed from available variables
            int falsePositives = negativeCount - trueNegatives;
            int falseNegatives = positiveCount - truePositives;

            return new Point(this, threshold,
                truePositives, trueNegatives,
                falsePositives, falseNegatives);
        }
        #endregion


        #region Private Methods
        /// <summary>
        ///   Calculates the area under the ROC curve using the trapezium method
        /// </summary>
        private double calculateAreaUnderCurve()
        {
            double sum = 0.0;

            for (int i = 0; i < collection.Count - 1; i++)
            {
                // Obs: False Positive Rate = (1-specificity)
                var tpz = collection[i].Sensitivity + collection[i + 1].Sensitivity;
                tpz = tpz * (collection[i].FalsePositiveRate - collection[i + 1].FalsePositiveRate) / 2.0;
                sum += tpz;
            }
            return sum;
        }

        /// <summary>
        ///   Calculates the standard error associated with this curve
        /// </summary>
        private double calculateStandardError()
        {
            double A = area;

            // real positive cases
            int Na = positiveCount;

            // real negative cases
            int Nn = negativeCount;

            double Q1 = A / (2.0 - A);
            double Q2 = 2 * A * A / (1.0 + A);

            return Math.Sqrt((A * (1.0 - A) +
                (Na - 1.0) * (Q1 - A * A) +
                (Nn - 1.0) * (Q2 - A * A)) / (Na * Nn));
        }
        #endregion

        #region Nested Classes

        /// <summary>
        ///   Object to hold information about a Receiver Operating Characteristic Curve Point
        /// </summary>
        public class Point : ConfusionMatrix
        {

            // Discrimination threshold (cutoff value)
            private double cutoff;

            // Parent curve

            /// <summary>
            ///   Constructs a new Receiver Operating Characteristic point.
            /// </summary>
            internal Point(ReceiverOperatingCharacteristic curve, double cutoff,
                int truePositives, int trueNegatives, int falsePositives, int falseNegatives)
                : base(truePositives, trueNegatives, falsePositives, falseNegatives)
            {
                this.cutoff = cutoff;
            }


            /// <summary>
            ///   Gets the cutoff value (discrimination threshold) for this point.
            /// </summary>
            public double Cutoff
            {
                get { return cutoff; }
            }
        }

        /// <summary>
        ///   Represents a Collection of Receiver Operating Characteristic (ROC) Curve points.
        ///   This class cannot be instantiated.
        /// </summary>
        public class PointCollection : ReadOnlyCollection<Point>
        {
            internal PointCollection(Point[] points)
                : base(points)
            {
            }

        }
        #endregion
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
            protected set;
        }

        /// <summary>
        /// The predicted label for each task when doing simulations from the current
        /// model state. It avoids overwriting the true label posterior.
        /// </summary>
        public Dictionary<string, Discrete> LookAheadTrueLabel
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for the true label variable.
        /// </summary>
        public Dictionary<string, Discrete> TrueLabelConstraint
        {
            get;
            protected set;
        }

        /// <summary>
        /// The predicted label for each task
        /// </summary>
        public Dictionary<string, int?> PredictedLabel
        {
            get;
            protected set;
        }

        /// <summary>
        /// The probabilities that generate the true label of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior of the confusion matrix of each worker.
        /// </summary>
        public Dictionary<string, Dirichlet[]> WorkerConfusionMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The look-ahead posterior of the confusion matrix of each worker obtained after simulating
        /// a new label in look-ahead run mode.
        /// </summary>
        public Dictionary<string, Dirichlet[]> LookAheadWorkerConfusionMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The predictive probabilities of the labels produced by each worker.
        /// </summary>
        public Dictionary<string, Dictionary<string, Discrete>> WorkerPrediction
        {
            get;
            protected set;
        }

        /// <summary>
        /// The community membership probabilities of each worker.
        /// </summary>
        public Dictionary<string, Discrete> WorkerCommunity
        {
            get;
            protected set;
        }

        /// <summary>
        /// The confusion matrix of each community.
        /// </summary>
        public Dirichlet[][] CommunityConfusionMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The score matrix of each community.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for worker confusion matrices
        /// int the community model.
        /// </summary>
        public Dictionary<string, VectorGaussian[]> WorkerScoreMatrixConstraint
        {
            get;
            protected set;
        }

        /// <summary>
        /// The probabilities that generate the community memberships of all the workers.
        /// </summary>
        public Dirichlet CommunityProb
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for community membership.
        /// int the community model.
        /// </summary>
        public Dictionary<string, Discrete> CommunityConstraint
        {
            get;
            protected set;
        }

        /// <summary>
        /// Model evidence.
        /// </summary>
        public Bernoulli ModelEvidence
        {
            get;
            protected set;
        }

        /// <summary>
        /// The data mapping.
        /// </summary>
        public DataMapping Mapping
        {
            get;
            set;
        }

        /// <summary>
        /// The full data mapping.
        /// </summary>
        public DataMapping FullMapping
        {
            get;
            set;
        }

        /// <summary>
        /// The gold labels of each task. The gold label type is nullable to
        /// support the (usual) situation where the is no labels.
        /// </summary>
        public Dictionary<string, int?> GoldLabels
        {
            get;
            protected set;
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
        /// The accuracy of the worker labels.
        /// </summary>
        public double WorkerLabelAccuracy
        {
            get;
            protected set;
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
        /// The number of communities.
        /// </summary>
        public int CommunityCount
        {
            get;
            private set;
        }

        public ConfusionMatrix BynaryConfusionMatrix
        {
            get;
            private set;
        }

        public ReceiverOperatingCharacteristic RocCurve
        {
            get;
            private set;
        }

        public List<double> trueBinaryLabel;
        public List<double> probTrueBinaryLabel;

        public enum RunMode
        {
            ClearResults,
            BatchTraining,
            IncrementalExperiment,
            OnlineExperiment,
            LookAheadExperiment,
            LoadAndUseCommunityPriors,
            Prediction,
        };

        protected virtual void ClearResults()
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
            PredictedLabel = new Dictionary<string, int?>();
        }

        protected virtual void UpdateResults(BCCPosteriors posteriors, RunMode mode)
        {
            if (mode == RunMode.LookAheadExperiment)
            {
                if (posteriors.TrueLabel != null)
                {
                    for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                    {
                        LookAheadTrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                    }
                }

                if (posteriors.WorkerConfusionMatrix != null)
                {
                    for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                    {
                        LookAheadWorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                    }
                }

            }
            else if (mode == RunMode.Prediction)
            {
                if (posteriors.WorkerConfusionMatrix != null)
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
            }
            else
            {
                // Update results for BCC
                BackgroundLabelProb = posteriors.BackgroundLabelProb;
                if (posteriors.WorkerConfusionMatrix != null)
                {
                    for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                    {
                        WorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                    }
                }
                if (posteriors.TrueLabel != null)
                {
                    for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                    {
                        TrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                    }
                }
                if (posteriors.TrueLabelConstraint != null)
                {
                    for (int t = 0; t < posteriors.TrueLabelConstraint.Length; t++)
                    {
                        TrueLabelConstraint[Mapping.TaskIndexToId[t]] = posteriors.TrueLabelConstraint[t];
                    }
                }

                this.ModelEvidence = posteriors.Evidence;
            }
        }

        /// <summary>
        /// Updates the accuracy using the current results.
        /// </summary>
        protected virtual void UpdateAccuracy()
        {
            double nlpdThreshold = -Math.Log(0.001);
            int labelCount = TrueLabel.First(kvp => kvp.Value != null).Value.Dimension;
            var confusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => 0.0);
            int correct = 0;
            double logProb = 0.0;

            int goldX = 0;

            // Only for binary labels
            if (Mapping.LabelCount == 2)
            {
                trueBinaryLabel = new List<double>();
                probTrueBinaryLabel = new List<double>();
            }

            foreach (var kvp in GoldLabels)
            {
                if (kvp.Value == null)
                    continue;

                // We have a gold label
                goldX++;

                Discrete trueLabel = null;
                if (TrueLabel.ContainsKey(kvp.Key))
                    trueLabel = TrueLabel[kvp.Key];

                if (trueLabel == null)
                {
                    trueLabel = Discrete.Uniform(Mapping.LabelCount);
                    //continue;  // No inferred label
                }

                var probs = trueLabel.GetProbs();
                double max = probs.Max();
                var predictedLabels = probs.Select((p, i) => new
                {
                    prob = p,
                    idx = i
                }).Where(a => a.prob == max).Select(a => a.idx).ToArray();

                int predictedLabel = predictedLabels.Length == 1 ? predictedLabels[0] : predictedLabels[Rand.Int(predictedLabels.Length)];

                this.PredictedLabel[kvp.Key] = predictedLabel;

                int goldLabel = kvp.Value.Value;

                confusionMatrix[goldLabel, predictedLabel] = confusionMatrix[goldLabel, predictedLabel] + 1.0;

                var nlp = -trueLabel.GetLogProb(goldLabel);
                if (nlp > nlpdThreshold)
                    nlp = nlpdThreshold;
                logProb += nlp;

                if (trueBinaryLabel != null)
                {
                    trueBinaryLabel.Add(goldLabel);
                    probTrueBinaryLabel.Add(probs[goldLabel]);
                }
            }

            Accuracy = correct / (double)goldX;
            NegativeLogProb = logProb / goldX;
            ModelConfusionMatrix = confusionMatrix;

            // Average recall
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

            // WorkerLabelAccuracy: Perc. agreement between worker label and gold label
            int sumAcc = 0;
            var LabelSet = Mapping.DataWithGold;
            int numLabels = LabelSet.Count();
            foreach (var datum in LabelSet)
            {
                sumAcc += datum.WorkerLabel == datum.GoldLabel ? 1 : 0;
            }
            WorkerLabelAccuracy = sumAcc / (double)numLabels;

            if (trueBinaryLabel != null)
            {
                RocCurve = new ReceiverOperatingCharacteristic(trueBinaryLabel.ToArray(), probTrueBinaryLabel.ToArray());
                RocCurve.Compute(0.001);
                BynaryConfusionMatrix = new ConfusionMatrix((int)confusionMatrix[1, 1], (int)confusionMatrix[0, 0], (int)confusionMatrix[0, 1], (int)confusionMatrix[1, 0]);
            }
        }

        public static void WriteConfusionMatrix(StreamWriter writer, string worker, Dirichlet[] confusionMatrix)
        {
            int labelCount = confusionMatrix.Length;
            var meanConfusionMatrix = confusionMatrix.Select(cm => cm.GetMean()).ToArray();
            var printableConfusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => meanConfusionMatrix[i][j]);
            WriteWorkerConfusionMatrix(writer, worker, printableConfusionMatrix);
        }

        public static void WriteWorkerConfusionMatrix(StreamWriter writer, string worker, double[,] confusionMatrix)
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

        public virtual void WriteResults(StreamWriter writer, bool writeCommunityParameters, bool writeWorkerParameters, bool writeWorkerCommunities, IList<Datum> data = null)
        {
            if (writeCommunityParameters && this.CommunityConfusionMatrix != null)
            {
                for (int communityIndex = 0; communityIndex < this.CommunityConfusionMatrix.Length; communityIndex++)
                {
                    WriteConfusionMatrix(writer, "Community" + communityIndex, this.CommunityConfusionMatrix[communityIndex]);
                }
            }

            if (writeWorkerParameters && this.WorkerConfusionMatrix != null)
            {
                foreach (var kvp in this.WorkerConfusionMatrix.Distinct().Take(5))
                {
                    WriteConfusionMatrix(writer, kvp.Key, kvp.Value);
                }
            }

            if (this.TrueLabel != null)
            {
                foreach (var kvp in this.TrueLabel.OrderByDescending(kvp => kvp.Value.GetProbs()[1]))
                {
                    if (data != null)
                    {
                        var taskLabels = data.Where(d => d.TaskId == kvp.Key).Select(l => l.WorkerLabel);
                        var pos = taskLabels.Where(l => l == 1);
                        var neg = taskLabels.Where(l => l == 0);
                        int numPos = pos.Count();
                        int numNeg = neg.Count();
                        writer.WriteLine($"{kvp.Key}:\t{kvp.Value}\tnum_neg: {numNeg}\t num_pos: {numPos}");
                    }
                    else
                    {
                        writer.WriteLine($"{kvp.Key}:\t{kvp.Value}");
                    }
                }
            }
        }
    }

    /// <summary>
    /// The confusion matrix for the classified instances
    /// </summary>
    public class ConfusionMatrix
    {

        //  2x2 confusion matrix
        private int truePositives;
        private int trueNegatives;
        private int falsePositives;
        private int falseNegatives;

        /// <summary>
        ///   Constructs a new Confusion Matrix.
        /// </summary>
        public ConfusionMatrix(int truePositives, int trueNegatives,
            int falsePositives, int falseNegatives)
        {
            this.truePositives = truePositives;
            this.trueNegatives = trueNegatives;
            this.falsePositives = falsePositives;
            this.falseNegatives = falseNegatives;
        }

        /// <summary>
        ///   Sensitivity, also known as True Positive Rate
        /// </summary>
        /// <remarks>
        ///   Sensitivity = TPR = TP / (TP + FN)
        /// </remarks>
        public double Sensitivity => (double)truePositives / (truePositives + falseNegatives);

        /// <summary>
        ///   Specificity, also known as True Negative Rate
        /// </summary>
        /// <remarks>
        ///   Specificity = TNR = TN / (FP + TN)
        ///    or also as:  TNR = (1-False Positive Rate)
        /// </remarks>
        public double Specificity => (double)trueNegatives / (trueNegatives + falsePositives);

        /// <summary>
        ///   False Positive Rate, also known as false alarm rate.
        /// </summary>
        /// <remarks>
        ///   It can be calculated as: FPR = FP / (FP + TN)
        ///                or also as: FPR = (1-specifity)
        /// </remarks>
        public double FalsePositiveRate => (double)falsePositives / (falsePositives + trueNegatives);
    }

    /// <summary>
    /// The BCCWords model
    /// </summary>
    public class BCCWords : BCC
    {
        // Add extra ranges
        private Range w;
        private Range nw;

        // Model evidence
        private Variable<bool> evidence;

        // Additional variables for BCCWords
        private VariableArray<Vector> ProbWord;
        private VariableArray<int> WordCount;
        private VariableArray<VariableArray<int>, int[][]> Words;
        private Variable<Dirichlet> ProbWordPrior;

        public void CreateModel(int NumTasks, int NumClasses, int VocabSize, int numBatches = 3)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");

            // Set up inference engine
            Engine = new InferenceEngine(new VariationalMessagePassing())
            {
                ShowFactorGraph = false,
                ShowWarnings = true,
                ShowProgress = false
            };

            // Set engine flags
            Engine.Compiler.WriteSourceFiles = true;
            Engine.Compiler.UseParallelForLoops = true;

            evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            // Set up ranges
            n = new Range(NumTasks).Named("N");
            c = new Range(NumClasses).Named("C");
            k = new Range(WorkerCount).Named("K");
            WorkerTaskCount = Variable.Array<int>(k).Named("WorkerTaskCount");
            kn = new Range(WorkerTaskCount[k]).Named("KN");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(kn), k).Named("Task");
            WorkerTaskIndex.SetValueRange(n);

            // Initialise truth
            BackgroundLabelProbPrior = Variable.New<Dirichlet>().Named("TruthProbPrior");
            BackgroundLabelProb = Variable<Vector>.Random(BackgroundLabelProbPrior).Named("TruthProb");
            BackgroundLabelProb.SetValueRange(c);

            // Truth distributions
            TrueLabel = Variable.Array<int>(n).Named("Truth");
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);

            //VocabSize = Variable.New<int>();
            w = new Range(VocabSize).Named("W");
            ProbWord = Variable.Array<Vector>(c).Named("ProbWord");
            ProbWord.SetValueRange(w);
            WordCount = Variable.Array<int>(n).Named("WordCount");
            nw = new Range(WordCount[n]).Named("WN");
            Words = Variable.Array(Variable.Array<int>(nw), n).Named("Word");
            ProbWordPrior = Variable.New<Dirichlet>().Named("ProbWordPrior");
            ProbWord[c] = Variable<Vector>.Random(ProbWordPrior).ForEach(c);

            // Initialise user profiles
            ConfusionMatrixPrior = Variable.Array(Variable.Array<Dirichlet>(c), k).Named("WorkerConfusionMatrixPrior");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("WorkerConfusionMatrix");
            WorkerConfusionMatrix[k][c] = Variable<Vector>.Random(ConfusionMatrixPrior[k][c]);
            WorkerConfusionMatrix.SetValueRange(c);

            // Vote distributions
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");

            using (Variable.ForEach(k))
            {
                var trueLabel = Variable.Subarray(TrueLabel, WorkerTaskIndex[k]).Named("TrueLabelSubarray");
                trueLabel.SetValueRange(c);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][trueLabel[kn]]);
                    }
                }
            }

            // Words inference
            using (Variable.ForEach(n))
            {
                using (Variable.Switch(TrueLabel[n]))
                {
                    Words[n][nw] = Variable.Discrete(ProbWord[TrueLabel[n]]).ForEach(nw);
                }
            }
            block.CloseBlock();
        }

        private void ObserveCrowdLabels(int[][] workerLabel, int[][] workerTaskIndex)
        {
            BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(c.SizeAsInt);
            WorkerCount.ObservedValue = workerLabel.Length;
            WorkerLabel.ObservedValue = workerLabel;
            WorkerTaskCount.ObservedValue = workerTaskIndex.Select(tasks => tasks.Length).ToArray();
            WorkerTaskIndex.ObservedValue = workerTaskIndex;
            SetBiasedPriors(WorkerCount.ObservedValue);
        }

        private void ObserveWords(int[][] words, int[] wordCounts)
        {
            Words.ObservedValue = words;
            WordCount.ObservedValue = wordCounts;
        }

        private void ObserveTrueLabels(int[] trueLabels)
        {
            TrueLabel.ObservedValue = trueLabels;
        }

        public void SetBiasedPriors(int workerCount)
        {
            // uniform over true values
            BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(c.SizeAsInt);
            ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(workerCount, input => Util.ArrayInit(c.SizeAsInt, l => new Dirichlet(Util.ArrayInit(c.SizeAsInt, l1 => l1 == l ? 5.5 : 1))));
            ProbWordPrior.ObservedValue = Dirichlet.Symmetric(w.SizeAsInt, 1);
        }

        /* Inference */
public BCCWordsPosteriors InferPosteriors(
            int[][] workerLabel, int[][] workerTaskIndex, int[][] words, int[] wordCounts, int[] trueLabels = null,
            int numIterations = 35)
        {
            ObserveCrowdLabels(workerLabel, workerTaskIndex);

            ObserveWords(words, wordCounts);

            if (trueLabels != null)
            {
                ObserveTrueLabels(trueLabels);
            }

            BCCWordsPosteriors posteriors = new BCCWordsPosteriors();

            Console.WriteLine("\n***** BCC Words *****\n");
            for (int it = 1; it <= numIterations; it++)
            {
                Engine.NumberOfIterations = it;
                posteriors.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
                posteriors.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
                posteriors.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
                posteriors.ProbWordPosterior = Engine.Infer<Dirichlet[]>(ProbWord);
                Console.WriteLine("Iteration {0}:\t{1:0.0000}", it, posteriors.TrueLabel[0]);
            }

            posteriors.Evidence = Engine.Infer<Bernoulli>(evidence);
            return posteriors;
        }
    }

    /// <summary>
    /// BCCWords posterior object.
    /// </summary>
    [Serializable]
    public class BCCWordsPosteriors : BCCPosteriors
    {
        /// <summary>
        /// The Dirichlet posteriors of the word probabilities for each true label value.
        /// </summary>
        public Dirichlet[] ProbWordPosterior;

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
    /// The BCC model class.
    /// </summary>
    public class BCC
    {
        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount => c?.SizeAsInt ?? 0;

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount => n?.SizeAsInt ?? 0;

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
    /// Copyright (c) 2013 Kory Becker http://www.primaryobjects.com/kory-becker.aspx
    /// 
    /// Permission is hereby granted, free of charge, to any person obtaining
    /// a copy of this software and associated documentation files (the
    /// "Software"), to deal in the Software without restriction, including
    /// without limitation the rights to use, copy, modify, merge, publish,
    /// distribute, sublicense, and/or sell copies of the Software, and to
    /// permit persons to whom the Software is furnished to do so, subject to
    /// the following conditions:
    /// 
    /// The above copyright notice and this permission notice shall be
    /// included in all copies or substantial portions of the Software.
    /// 
    /// Description:
    /// Performs a TF*IDF (Term Frequency * Inverse Document Frequency) transformation on an array of documents.
    /// Each document string is transformed into an array of doubles, cooresponding to their associated TF*IDF values.
    /// 
    /// Usage:
    /// string[] documents = LoadYourDocuments();
    ///
    /// double[][] inputs = TFIDF.Transform(documents);
    /// inputs = TFIDF.Normalize(inputs);
    /// 
    /// </summary>
    public static class TFIDFClass
    {
        /// <summary>
        /// Document vocabulary, containing each word's IDF value.
        /// </summary>
        private static Dictionary<string, double> _vocabularyIDF = new Dictionary<string, double>();

        /// <summary>
        /// Transforms a list of documents into their associated TF*IDF values.
        /// If a vocabulary does not yet exist, one will be created, based upon the documents' words.
        /// </summary>
        /// <param name="documents">string[]</param>
        /// <param name="vocabulary">The vocabulary</param>
        /// <param name="vocabularyThreshold">Minimum number of occurences of the term within all documents</param>
        /// <returns>double[][]</returns>
        public static double[][] Transform(string[] documents, out List<string> vocabulary, int vocabularyThreshold = 3)
        {
            List<List<string>> stemmedDocs;

            // Get the vocabulary and stem the documents at the same time.
            vocabulary = GetVocabulary(documents, out stemmedDocs, vocabularyThreshold);

            if (_vocabularyIDF.Count == 0)
            {
                // Calculate the IDF for each vocabulary term.
                _vocabularyIDF = vocabulary.ToDictionary(term => term, term =>
                {
                    double numberOfDocsContainingTerm = stemmedDocs.Count(d => d.Contains(term));
                    return Math.Log(stemmedDocs.Count / (1 + numberOfDocsContainingTerm));
                });
            }

            // Transform each document into a vector of tfidf values.
            return TransformToTFIDFVectors(stemmedDocs, _vocabularyIDF);
        }

        /// <summary>
        /// Converts a list of stemmed documents (lists of stemmed words) and their associated vocabulary + idf values, into an array of TF*IDF values.
        /// </summary>
        /// <param name="stemmedDocs">List of List of string</param>
        /// <param name="vocabularyIDF">Dictionary of string, double (term, IDF)</param>
        /// <returns>double[][]</returns>
        private static double[][] TransformToTFIDFVectors(List<List<string>> stemmedDocs, Dictionary<string, double> vocabularyIDF)
        {
            // Transform each document into a vector of tfidf values.
            List<List<double>> vectors = new List<List<double>>();
            foreach (var doc in stemmedDocs)
            {
                List<double> vector = new List<double>();

                foreach (var vocab in vocabularyIDF)
                {
                    // Term frequency = count how many times the term appears in this document.
                    double tf = doc.Where(d => d == vocab.Key).Count();
                    double tfidf = tf * vocab.Value;

                    vector.Add(tfidf);
                }

                vectors.Add(vector);
            }

            return vectors.Select(v => v.ToArray()).ToArray();
        }

        /// <summary>
        /// Normalizes a TF*IDF array of vectors using L2-Norm.
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        /// <param name="vectors">double[][]</param>
        /// <returns>double[][]</returns>
        public static double[][] Normalize(double[][] vectors)
        {
            // Normalize the vectors using L2-Norm.
            List<double[]> normalizedVectors = new List<double[]>();
            foreach (var vector in vectors)
            {
                var normalized = Normalize(vector);
                normalizedVectors.Add(normalized);
            }

            return normalizedVectors.ToArray();
        }

        /// <summary>
        /// Normalizes a TF*IDF vector using L2-Norm.
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        /// <param name="vector">double[][]</param>
        /// <returns>double[][]</returns>
        public static double[] Normalize(double[] vector)
        {
            List<double> result = new List<double>();

            double sumSquared = 0;
            foreach (var value in vector)
            {
                sumSquared += value * value;
            }

            double SqrtSumSquared = Math.Sqrt(sumSquared);

            foreach (var value in vector)
            {
                // L2-norm: Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
                result.Add(value / SqrtSumSquared);
            }

            return result.ToArray();
        }

        #region Private Helpers

        /// <summary>
        /// Parses and tokenizes a list of documents, returning a vocabulary of words.
        /// </summary>
        /// <param name="docs">string[]</param>
        /// <param name="stemmedDocs">List of List of string</param>
        /// <param name="vocabularyThreshold"></param>
        /// <returns>Vocabulary (list of strings)</returns>
        private static List<string> GetVocabulary(string[] docs, out List<List<string>> stemmedDocs, int vocabularyThreshold)
        {
            List<string> vocabulary = new List<string>();
            Dictionary<string, int> wordCountList = new Dictionary<string, int>();
            stemmedDocs = new List<List<string>>();
            var stopWordsFile = File.ReadAllLines(@"<your-stop-words-file>");
            var stopWordsList = new List<string>(stopWordsFile).ToArray();
            int docIndex = 0;
            List<string> words = new List<string>();

            foreach (var doc in docs)
            {
                List<string> stemmedDoc = new List<string>();

                docIndex++;

                if (docIndex % 10000 == 0)
                {
                    Console.WriteLine("Processing " + docIndex + "/" + docs.Length);
                }

                string[] parts2 = Tokenize(doc.ToLower());

                foreach (string part in parts2)
                {
                    // Strip non-alphanumeric characters.
                    string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                    if (!stopWordsList.Contains(stripped.ToLower()))
                    {
                        try
                        {
                            var english = new EnglishWord(stripped);
                            string stem = english.Original;

                            words.Add(stem);

                            if (stem.Length > 0)
                            {
                                // Build the word count list.
                                if (wordCountList.ContainsKey(stem))
                                {
                                    wordCountList[stem]++;
                                }
                                else
                                {
                                    wordCountList.Add(stem, 0);
                                }

                                stemmedDoc.Add(stem);
                            }
                        }
                        catch
                        {
                            // ignored
                        }
                    }
                }

                stemmedDocs.Add(stemmedDoc);
            }

            // Get the top words.
            var vocabList = wordCountList.Where(w => w.Value >= vocabularyThreshold);
            foreach (var item in vocabList)
            {
                vocabulary.Add(item.Key);
            }

            return vocabulary;
        }

        public static int[][] GetWordIndexStemmedDocs(string[] docs, List<string> vocabulary)
        {
            List<int>[] wordIndex = Util.ArrayInit(docs.Length, d => new List<int>());

            int docIndex = 0;

            foreach (var doc in docs)
            {
                if (doc != null)
                {
                    string[] parts2 = Tokenize(doc.ToLower());

                    List<int> wordIndexDoc = new List<int>();
                    foreach (string part in parts2)
                    {
                        // Strip non-alphanumeric characters.
                        string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                        try
                        {
                            var english = new EnglishWord(stripped);
                            string stem = english.Stem;

                            if (vocabulary.Contains(stem))
                            {
                                wordIndexDoc.Add(vocabulary.IndexOf(stem));
                            }
                        }
                        catch
                        {
                            // ignored
                        }
                    }

                    wordIndex[docIndex] = (wordIndexDoc.Distinct().ToList());
                    docIndex++;
                }
            }

            return wordIndex.Select(list => list.Select(index => index).ToArray()).ToArray();
        }

        /// <summary>
        /// Tokenizes a string, returning its list of words.
        /// </summary>
        /// <param name="text">string</param>
        /// <returns>string[]</returns>
        private static string[] Tokenize(string text)
        {
            // Strip all HTML.
            text = Regex.Replace(text, "<[^<>]+>", "");

            // Strip numbers.
            text = Regex.Replace(text, "[0-9]+", "number");

            // Strip urls.
            text = Regex.Replace(text, @"(http|https)://[^\s]*", "httpaddr");

            // Strip email addresses.
            text = Regex.Replace(text, @"[^\s]+@[^\s]+", "emailaddr");

            // Strip dollar sign.
            text = Regex.Replace(text, "[$]+", "dollar");

            // Strip usernames.
            text = Regex.Replace(text, @"@[^\s]+", "username");

            // Tokenize and also get rid of any punctuation
            return text.Split(" @$/#.-:&*+=[]?!(){},''\">_<;%\\".ToCharArray());
        }

        #endregion
    }
}