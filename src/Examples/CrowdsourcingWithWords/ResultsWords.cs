// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace CrowdsourcingWithWords
{
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

		protected override void UpdateResults(BCCPosteriors posteriors, RunMode mode)
		{
			base.UpdateResults(posteriors, mode);
			var wordsPosteriors = posteriors as BCCWordsPosteriors;
			if (wordsPosteriors?.ProbWordPosterior != null)
			{
				this.ProbWords = wordsPosteriors.ProbWordPosterior;
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
			var filteredVocabulary = vocabularyTfidf.Distinct().ToList();
	        return filteredVocabulary.Count>=10 ? filteredVocabulary : vocabulary;
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
				var classifiedWords = new Dictionary<string, KeyValuePair<string, double>>();
                for (int c = 0; c < NumClasses; c++)
                {
	                string className = string.Empty;
	                if (MappingWords != null)
	                {
		                if (MappingWords.WorkerCount > 100) // Assume it's CF
						{
			                className = MappingWords.CFLabelName[c];
		                }
		                else
		                {
							className = MappingWords.SPLabelName[c];
						}
		                writer.WriteLine($"Class {className}");
	                }

                    Vector probs = ProbWords[c].GetMean();
                    var probsDictionary = probs.Select((value, index) => new KeyValuePair<string, double>(MappingWords.Vocabulary[index], Math.Log(value))).OrderByDescending(x => x.Value).ToArray();
	                topWords = Math.Min(topWords, probsDictionary.Length);
                    for (int w = 0; w < topWords; w++)
                    {
                        writer.WriteLine($"\t{probsDictionary[w].Key}: \t{probsDictionary[w].Value:0.000}");
	                    if (!string.IsNullOrEmpty(className))
	                    {
		                    KeyValuePair<string, double> classifiedWord ;
		                    if (!classifiedWords.TryGetValue(probsDictionary[w].Key,out classifiedWord)
								|| classifiedWord.Value< probsDictionary[w].Value)
		                    {
			                    classifiedWords[probsDictionary[w].Key] = new KeyValuePair<string, double>(className, probsDictionary[w].Value);
		                    }
	                    }
                    }
                }
	            writer.WriteLine();
				writer.WriteLine($"Main classes:");
				foreach (var wordByClass in classifiedWords.GroupBy(classified=>classified.Value.Key))
	            {
		            writer.WriteLine($"Class {wordByClass.Key}:");
		            foreach (var word in wordByClass.OrderByDescending(w=>w.Value.Value))
		            {
			            writer.WriteLine($"\t{word.Key}");
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
            if (vocabularyOnSubData.Count > 300)
                return vocabularyOnSubData.GetRange(0, 300);
            else
                return vocabularyOnSubData;
        }
    }
}
