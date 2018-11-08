using System;
using Microsoft.ML.Probabilistic.Distributions;

namespace CrowdsourcingWithWords
{
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
}