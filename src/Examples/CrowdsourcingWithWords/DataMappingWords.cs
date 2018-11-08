using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;

namespace CrowdsourcingWithWords
{
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
}