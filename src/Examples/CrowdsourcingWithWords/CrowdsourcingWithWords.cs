// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

/* Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning
* 
* Software to run the experiment presented in the paper "Language Understanding in the Wind: Combining Crowdsourcing and Machine Learning" by Simpson et. al, WWW15.
* The data in this example comes from https://data.world/crowdflower/weather-sentiment, licensed as "Public Domain".
* To run it on your data:
* - Replace Data/weatherTweets.tsv with tab-separated fields <WorkerId, TaskId, Worker label, Text, Gold label (optional)>
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
			var data = Datum.LoadData(Path.Combine("Data", "weatherTweets.tsv.gz"));

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
}