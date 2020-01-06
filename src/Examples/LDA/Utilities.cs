// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using System.IO;
using System.Linq;

namespace LDAExample
{
    public class Utilities
    {
        /// <summary>
        /// Randomly create true theta and phi arrays
        /// </summary>
        /// <param name="numVocab">Vocabulary size</param>
        /// <param name="numTopics">Number of topics</param>
        /// <param name="numDocs">Number of documents</param>
        /// <param name="averageDocLength">Average document length</param>
        /// <param name="averageWordsPerTopic">Average number of words per topic</param>
        /// <param name="trueTheta">Theta array (output)</param>
        /// <param name="truePhi">Phi array (output)</param>
        public static void CreateTrueThetaAndPhi(
            int numVocab,
            int numTopics,
            int numDocs,
            int averageDocLength,
            int averageWordsPerTopic,
            out Dirichlet[] trueTheta,
            out Dirichlet[] truePhi)
        {
            truePhi = new Dirichlet[numTopics];
            for (int i = 0; i < numTopics; i++)
            {
                truePhi[i] = Dirichlet.Uniform(numVocab);
                truePhi[i].PseudoCount.SetAllElementsTo(0.0);

                // Draw the number of unique words in the topic.
                int numUniqueWordsPerTopic = Poisson.Sample((double)averageWordsPerTopic);
                if (numUniqueWordsPerTopic >= numVocab)
                {
                    numUniqueWordsPerTopic = numVocab;
                }

                if (numUniqueWordsPerTopic < 1)
                {
                    numUniqueWordsPerTopic = 1;
                }

                double expectedRepeatOfWordInTopic = 
                    ((double)numDocs) * averageDocLength / numUniqueWordsPerTopic;
                int[] shuffledWordIndices = Rand.Perm(numVocab);
                for (int j = 0; j < numUniqueWordsPerTopic; j++)
                {
                    int wordIndex = shuffledWordIndices[j];

                    // Draw the count for that word
                    int cnt = Poisson.Sample(expectedRepeatOfWordInTopic);
                    truePhi[i].PseudoCount[wordIndex] = cnt + 1.0;
                }
            }

            trueTheta = new Dirichlet[numDocs];
            for (int i = 0; i < numDocs; i++)
            {
                trueTheta[i] = Dirichlet.Uniform(numTopics);
                trueTheta[i].PseudoCount.SetAllElementsTo(0.0);

                // Draw the number of unique topics in the doc.
                int numUniqueTopicsPerDoc = Math.Min(1 + Poisson.Sample(1.0), numTopics);
                double expectedRepeatOfTopicInDoc = 
                    (double)averageDocLength / numUniqueTopicsPerDoc;
                int[] shuffledTopicIndices = Rand.Perm(numTopics);
                for (int j = 0; j < numUniqueTopicsPerDoc; j++)
                {
                    int topicIndex = shuffledTopicIndices[j];

                    // Draw the count for that topic
                    int cnt = Poisson.Sample(expectedRepeatOfTopicInDoc);
                    trueTheta[i].PseudoCount[topicIndex] = cnt + 1.0;
                }
            }
        }

        /// Generate LDA data - returns an array of dictionaries mapping unique word index
        /// to word count per document.
        /// <param name="trueTheta">Known Theta</param>
        /// <param name="truePhi">Known Phi</param>
        /// <param name="averageNumWords">Average number of words to sample per doc</param>
        /// <returns></returns>
        public static Dictionary<int, int>[] GenerateLDAData(Dirichlet[] trueTheta, Dirichlet[] truePhi, int averageNumWords)
        {
            int numVocab = truePhi[0].Dimension;
            int numTopics = truePhi.Length;
            int numDocs = trueTheta.Length;

            // Sample from the model
            Vector[] topicDist = new Vector[numDocs];
            Vector[] wordDist = new Vector[numTopics];
            for (int i = 0; i < numDocs; i++)
            {
                topicDist[i] = trueTheta[i].Sample();
            }

            for (int i = 0; i < numTopics; i++)
            {
                wordDist[i] = truePhi[i].Sample();
            }

            var wordCounts = new Dictionary<int, int>[numDocs];
            for (int i = 0; i < numDocs; i++)
            {
                int LengthOfDoc = Poisson.Sample((double)averageNumWords);

                var counts = new Dictionary<int, int>();
                for (int j = 0; j < LengthOfDoc; j++)
                {
                    int topic = Discrete.Sample(topicDist[i]);
                    int w = Discrete.Sample(wordDist[topic]);
                    if (!counts.ContainsKey(w))
                    {
                        counts.Add(w, 1);
                    }
                    else
                    {
                        counts[w] = counts[w] + 1;
                    }
                }

                wordCounts[i] = counts;
            }

            return wordCounts;
        }

        /// <summary>
        /// Calculate perplexity for test words
        /// </summary>
        /// <param name="predictiveDist">Predictive distribution for each document</param>
        /// <param name="testWordsPerDoc">Test words per document</param>
        /// <returns></returns>
        public static double Perplexity(Discrete[] predictiveDist, Dictionary<int, int>[] testWordsPerDoc)
        {
            double num = 0.0;
            double den = 0.0;
            int numDocs = predictiveDist.Length;
            for (int i = 0; i < numDocs; i++)
            {
                Discrete d = predictiveDist[i];
                var counts = testWordsPerDoc[i];
                foreach (KeyValuePair<int, int> kvp in counts)
                {
                    num += kvp.Value * d.GetLogProb(kvp.Key);
                    den += kvp.Value;
                }
            }

            return Math.Exp(-num / den);
        }

        /// <summary>
        /// Load data. Each line is of the form cnt,wrd1_index:count,wrd2_index:count,...
        /// </summary>
        /// <param name="fileName">The file name</param>
        /// <returns></returns>
        public static Dictionary<int, int>[] LoadWordCounts(string fileName)
        {
            return File.ReadLines(fileName).Select(str =>
            {
                string[] split = str.Split(' ', ':');
                int numUniqueTerms = int.Parse(split[0]);
                var dict = new Dictionary<int, int>();
                for (int i = 0; i < (split.Length - 1) / 2; i++)
                {
                    dict.Add(int.Parse(split[2 * i + 1]), int.Parse(split[2 * i + 2]));
                }
                return dict;
            }).ToArray();
        }

        /// <summary>
        /// Load the vocabulary
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public static Dictionary<int, string> LoadVocabulary(string fileName)
        {
            return File.ReadLines(fileName).Select((str, idx) => Tuple.Create(str, idx)).ToDictionary(tup => tup.Item2, tup => tup.Item1);
        }

        /// <summary>
        /// Get the vocabulary size for the data (max index + 1)
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static int GetVocabularySize(Dictionary<int, int>[] data)
        {
            int max = int.MinValue;
            foreach (Dictionary<int, int> dict in data)
            {
                foreach (int key in dict.Keys)
                {
                    if (key > max)
                    {
                        max = key;
                    }
                }
            }

            return max + 1;
        }
    }
}
