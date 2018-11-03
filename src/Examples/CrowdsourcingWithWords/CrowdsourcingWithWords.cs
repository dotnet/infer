// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

/* Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning
* 
* Software to run the experiment presented in the paper "Language Understanding in the Wind: Combining Crowdsourcing and Machine Learning" by Simpson et. al, WWW15
* To run it on your data:
* - Replace Data/labels.tsv with tab-separated fields <WorkerId, TaskId, Worker label, Text, Gold label (optional)>
* - Replace Data/stopwords.txt with the list of stop words, one for each line
*/

namespace CrowdsourcingWithWords
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    class CrowdsourcingWithWords
    {
        /// <summary>
        /// Main method to run the crowdsourcing experiments presented in Simpson et.al (WWW15).
        /// </summary>
        public static void Main()
        {
            var data = Datum.LoadData(Path.Combine("Data", "labels.tsv"));

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

            Console.WriteLine("Done.  Press enter to exit.");
            Console.ReadLine();
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
}