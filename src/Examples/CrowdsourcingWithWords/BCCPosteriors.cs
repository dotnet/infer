using System;
using Microsoft.ML.Probabilistic.Distributions;

namespace CrowdsourcingWithWords
{
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
}