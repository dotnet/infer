// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
// #define blei_corpus
using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using System.Diagnostics;
using System.Linq;

namespace LDAExample
{
    class Program
    {
        /// <summary>
        /// Main test program for Infer.NET LDA models
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            Rand.Restart(5);
            Dictionary<int, string> vocabulary = null;
#if blei_corpus
            Dictionary<int, int>[] trainWordsInTrainDoc = Utilities.LoadWordCounts(@"..\..\..\Data\ap.txt");
            vocabulary = Utilities.LoadVocabulary(@"..\..\..\Data\apvocab.txt");
            Dictionary<int, int>[] testWordsInTrainDoc = trainWordsInTrainDoc;
            Dictionary<int, int>[] wordsInTestDoc = null;
            int sizeVocab = Utilities.GetVocabularySize(trainWordsInTrainDoc);
            int numTopics = 10;
            int numTrainDocs = trainWordsInTrainDoc.Length;
            Dirichlet[] trueThetaTest = null;
            Console.WriteLine("************************************");
            Console.WriteLine("Vocabulary size = " + sizeVocab);
            Console.WriteLine("Number of documents = " + numTrainDocs);
            Console.WriteLine("Number of topics = " + numTopics);
            Console.WriteLine("************************************");
            double alpha = 150 / numTopics;
            double beta = 0.1;
#else
            int numTopics = 5;
            int sizeVocab = 1000;
            int numTrainDocs = 500;
            int averageDocumentLength = 100;
            int averageWordsPerTopic = 10;
            int numTestDocs = numTopics * 5;
            int numDocs = numTrainDocs + numTestDocs;

            // Create the true model
            Dirichlet[] trueTheta, truePhi;
            Utilities.CreateTrueThetaAndPhi(sizeVocab, numTopics, numDocs, averageDocumentLength, averageWordsPerTopic, out trueTheta, out truePhi);

            // Split the documents between a train and test set
            Dirichlet[] trueThetaTrain = new Dirichlet[numTrainDocs];
            Dirichlet[] trueThetaTest = new Dirichlet[numTestDocs];
            int docx = 0;
            for (int i = 0; i < numTrainDocs; i++)
            {
                trueThetaTrain[i] = trueTheta[docx++];
            }

            for (int i = 0; i < numTestDocs; i++)
            {
                trueThetaTest[i] = trueTheta[docx++];
            }

            // Generate training and test data for the training documents
            Dictionary<int, int>[] trainWordsInTrainDoc = Utilities.GenerateLDAData(trueThetaTrain, truePhi, (int)(0.9 * averageDocumentLength));
            Dictionary<int, int>[] testWordsInTrainDoc = Utilities.GenerateLDAData(trueThetaTrain, truePhi, (int)(0.1 * averageDocumentLength));
            Dictionary<int, int>[] wordsInTestDoc = Utilities.GenerateLDAData(trueThetaTest, truePhi, averageDocumentLength);
            Console.WriteLine("************************************");
            Console.WriteLine("Vocabulary size = " + sizeVocab);
            Console.WriteLine("Number of topics = " + numTopics);
            Console.WriteLine("True average words per topic = " + averageWordsPerTopic);
            Console.WriteLine("Number of training documents = " + numTrainDocs);
            Console.WriteLine("Number of test documents = " + numTestDocs);
            Console.WriteLine("************************************");
            double alpha = 1.0;
            double beta = 0.1;
#endif

            for (int i = 0; i < 2; i++)
            {
                bool shared = i == 0;

                // if (!shared) continue; // Comment out this line to see full LDA models
                RunTest(
                    sizeVocab, 
                    numTopics, 
                    trainWordsInTrainDoc, 
                    testWordsInTrainDoc,
                    alpha,
                    beta, 
                    shared,
                    trueThetaTest,
                    wordsInTestDoc,
                    vocabulary);
            }

            Console.WriteLine("Done.  Press enter to exit.");
            Console.ReadLine();
        }

        /// <summary>
        /// Run a single test for a single model
        /// </summary>
        /// <param name="sizeVocab">Size of the vocabulary</param>
        /// <param name="numTopics">Number of topics</param>
        /// <param name="trainWordsInTrainDoc">Lists of words in training documents used for training</param>
        /// <param name="testWordsInTrainDoc">Lists of words in training documents used for testing</param>
        /// <param name="alpha">Background pseudo-counts for distributions over topics</param>
        /// <param name="beta">Background pseudo-counts for distributions over words</param>
        /// <param name="shared">If true, uses shared variable version of the model</param>
        /// <param name="trueThetaTest">The true topic distributions for the documents in the test set</param>
        /// <param name="wordsInTestDoc">Lists of words in test documents</param>
        /// <param name="vocabulary">Vocabulary</param>
        static void RunTest(
            int sizeVocab, 
            int numTopics,
            Dictionary<int, int>[] trainWordsInTrainDoc, 
            Dictionary<int, int>[] testWordsInTrainDoc,
            double alpha, 
            double beta, 
            bool shared,
            Dirichlet[] trueThetaTest, 
            Dictionary<int, int>[] wordsInTestDoc,
            Dictionary<int, string> vocabulary = null)
        {
            Stopwatch stopWatch = new Stopwatch();

            // Square root of number of documents is the optimal for memory
            int batchCount = (int)Math.Sqrt((double)trainWordsInTrainDoc.Length);
            Rand.Restart(5);
            ILDA model;
            LDAPredictionModel predictionModel;
            LDATopicInferenceModel topicInfModel;
            if (shared)
            {
                model = new LDAShared(batchCount, sizeVocab, numTopics);
                ((LDAShared)model).IterationsPerPass = Enumerable.Repeat(10, 5).ToArray();
            }
            else
            {
                model = new LDAModel(sizeVocab, numTopics);
                model.Engine.NumberOfIterations = 50;
            }

            Console.WriteLine("\n\n************************************");
            Console.WriteLine(String.Format("\nTraining {0}LDA model...\n", shared ? "batched " : "non-batched "));

            // Train the model - we will also get rough estimates of execution time and memory
            GC.Collect();
            PerformanceCounter memCounter = new PerformanceCounter("Memory", "Available MBytes");
            float preMem = memCounter.NextValue();
            stopWatch.Reset();
            stopWatch.Start();
            double logEvidence = model.Infer(trainWordsInTrainDoc, alpha, beta, out Dirichlet[] postTheta, out Dirichlet[] postPhi);
            stopWatch.Stop();
            float postMem = memCounter.NextValue();
            double approxMB = preMem - postMem;
            GC.KeepAlive(model); // Keep the model alive to this point (for the memory counter)
            Console.WriteLine(String.Format("Approximate memory usage: {0:F2} MB", approxMB));
            Console.WriteLine(String.Format("Approximate execution time (including model compilation): {0} seconds", stopWatch.ElapsedMilliseconds / 1000));

            // Calculate average log evidence over total training words
            int totalWords = trainWordsInTrainDoc.Sum(doc => doc.Sum(w => w.Value));
            Console.WriteLine("\nTotal number of training words = {0}", totalWords);
            Console.WriteLine(String.Format("Average log evidence of model: {0:F2}", logEvidence / (double)totalWords));

            if (vocabulary != null)
            {
                int numWordsToPrint = 20;

                // Print out the top n words for each topic
                for (int i = 0; i < postPhi.Length; i++)
                {
                    double[] pc = postPhi[i].PseudoCount.ToArray();
                    int[] wordIndices = new int[pc.Length];
                    for (int j = 0; j < wordIndices.Length; j++)
                    {
                        wordIndices[j] = j;
                    }

                    Array.Sort(pc, wordIndices);
                    Console.WriteLine("Top {0} words in topic {1}:", numWordsToPrint, i);
                    int idx = wordIndices.Length;
                    for (int j = 0; j < numWordsToPrint; j++)
                    {
                        Console.Write("\t{0}", vocabulary[wordIndices[--idx]]);
                    }

                    Console.WriteLine();
                }
            }

            if (testWordsInTrainDoc != null)
            {
                // Test on unseen words in training documents
                Console.WriteLine("\n\nCalculating perplexity on test words in training documents...");
                predictionModel = new LDAPredictionModel(sizeVocab, numTopics);
                predictionModel.Engine.NumberOfIterations = 5;
                var predDist = predictionModel.Predict(postTheta, postPhi);
                var perplexity = Utilities.Perplexity(predDist, testWordsInTrainDoc);
                Console.WriteLine(String.Format("\nPerplexity = {0:F3}", perplexity));
            }

            if (wordsInTestDoc != null)
            {
                // Test on unseen documents. Note that topic ids for the trained model will be a random
                // permutation of the topic ids for the ground truth
                Console.WriteLine("\n\nInferring topics for test documents...");
                topicInfModel = new LDATopicInferenceModel(sizeVocab, numTopics);
                topicInfModel.Engine.NumberOfIterations = 10;
                var inferredTopicDists = topicInfModel.InferTopic(alpha, postPhi, wordsInTestDoc);
                Dictionary<TopicPair, int> topicPairCounts = new Dictionary<TopicPair, int>();
                for (int i = 0; i < inferredTopicDists.Length; i++)
                {
                    int infTopic = inferredTopicDists[i].PseudoCount.IndexOfMaximum();
                    int trueTopic = trueThetaTest[i].PseudoCount.IndexOfMaximum();
                    TopicPair tp = new TopicPair() { InferredTopic = infTopic, TrueTopic = trueTopic };
                    if (!topicPairCounts.ContainsKey(tp))
                    {
                        topicPairCounts.Add(tp, 1);
                    }
                    else
                    {
                        topicPairCounts[tp] = topicPairCounts[tp] + 1;
                    }
                }

                var correctCount = CountCorrectTopicPredictions(topicPairCounts, numTopics);
                Console.WriteLine(String.Format("Maximum inferred topic matches maximum true topic {0} times out of {1}", correctCount, inferredTopicDists.Length));
                Console.WriteLine("\nThis uses a greedy algorithm to determine the mapping from inferred topic indices to true topic indices");
                Console.WriteLine("\n************************************");
            }
        }

        /// <summary>
        /// A topic pair
        /// </summary>
        public struct TopicPair
        {
            public int InferredTopic;
            public int TrueTopic;
        }

        /// <summary>
        /// Count the number of correct predictions of the best topic.
        /// This uses a simple greedy algorithm to determine the topic mapping (use with caution!)
        /// </summary>
        /// <param name="topicPairCounts">A dictionary mapping (inferred, true) pairs to counts</param>
        /// <param name="numTopics">The number of topics</param>
        /// <returns></returns>
        public static int CountCorrectTopicPredictions(Dictionary<TopicPair, int> topicPairCounts, int numTopics)
        {
            int[] topicMapping = new int[numTopics];
            for (int i = 0; i < numTopics; i++)
            {
                topicMapping[i] = -1;
            }

            // Sort by count
            List<KeyValuePair<TopicPair, int>> kvps = new List<KeyValuePair<TopicPair, int>>(topicPairCounts); 
            kvps.Sort( 
                delegate(KeyValuePair<TopicPair, int> kvp1, KeyValuePair<TopicPair, int> kvp2)
                {
                    return kvp2.Value.CompareTo(kvp1.Value); 
                }
            );

            int correctCount = 0;
            while (kvps.Count > 0)
            {
                KeyValuePair<TopicPair, int> kvpHead = kvps[0];
                int inferredTopic = kvpHead.Key.InferredTopic;
                int trueTopic = kvpHead.Key.TrueTopic;
                topicMapping[inferredTopic] = trueTopic;
                correctCount += kvpHead.Value;
                kvps.Remove(kvpHead);

                // Now delete anything in the list that has either of these
                for (int i = kvps.Count - 1; i >= 0; i--)
                {
                    KeyValuePair<TopicPair, int> kvp = kvps[i];
                    int infTop = kvp.Key.InferredTopic;
                    int trueTop = kvp.Key.TrueTopic;
                    if (infTop == inferredTopic || trueTop == trueTopic)
                    {
                        kvps.Remove(kvp);
                    }
                }
            }

            return correctCount;
        }
    }
}
