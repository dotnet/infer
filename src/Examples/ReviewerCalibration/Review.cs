using System;
using System.Collections.Generic;
using System.Text;

namespace ReviewerCalibration
{
    /// <summary>
    /// This class stores all the information about a review.
    /// </summary>
    public class Review
    {
        /// <summary>
        /// Reviewer of a submission.
        /// </summary>
        public Reviewer Reviewer { get; private set; }
        /// <summary>
        /// Submission that is reviewed.
        /// </summary>
        public Submission Submission { get; private set; }
        /// <summary>
        /// Recommendation of the submission by the reviewer - larger means better.
        /// </summary>
        public Recommendation Recommendation { get; private set; }
        /// <summary>
        /// Recommendation of the submission by the reviewer - larger means better.
        /// </summary>
        public Expertise Expertise { get; private set; }
        /// <summary>
        /// Non-default constructor for a review.
        /// </summary>
        /// <param name="reviewer">Reviewer of the submission.</param>
        /// <param name="submission">Submission that is reviewed.</param>
        /// <param name="recommendation">Recommendation of the reviewer for the submission.</param>
        /// <param name="expertise">Expertise of the reviewer for this recommendation.</param>
        public Review(Reviewer reviewer, Submission submission, int recommendation, int expertise)
        {
            Reviewer = reviewer;
            Submission = submission;
            switch (recommendation)
            {
                case 1: Recommendation = Recommendation.StrongReject; break;
                case 2: Recommendation = Recommendation.Reject; break;
                case 3: Recommendation = Recommendation.WeakReject; break;
                case 4: Recommendation = Recommendation.WeakAccept; break;
                case 5: Recommendation = Recommendation.Accept; break;
                case 6: Recommendation = Recommendation.StrongAccept; break;
                default:
                    throw new ArgumentException(
             "recommendation",
             "Recommendations have to be in the range {1,2,3,4,5,6}.");
            }
            switch (expertise)
            {
                case 1: Expertise = Expertise.InformedOutsider; break;
                case 2: Expertise = Expertise.Knowledgeable; break;
                case 3: Expertise = Expertise.Expert; break;
                default:
                    throw new ArgumentException(
             "expertise",
             "Expertise have to be in the range {1,2,3}.");
            }
        }
    }

    /// <summary>
    /// This class stores all the information about a reviewer.
    /// </summary>
    public class Reviewer
    {
        /// <summary>
        /// The name of the reviewer.
        /// </summary>
        public string Name { get; private set; }
        /// <summary>
        /// Non-default constructor based on a unique reviewer index.
        /// </summary>
        /// <param name="index">Internal index of the reviewer.</param>
        public Reviewer(int index)
        {
            Name = "Reviewer " + index;
        }
    }

    /// <summary>
    /// This class stores all the information about a submission.
    /// </summary>
    public class Submission
    {
        /// <summary>
        /// Title of the submission.
        /// </summary>
        public string Title { get; private set; }
        /// <summary>
        /// Non-default constructor based on a unique submission index.
        /// </summary>
        /// <param name="index">Internal index of the submission.</param>
        public Submission(int index)
        {
            Title = "Submission " + index;
        }
    }

    /// <summary>
    /// The type of recommendations that a reviewer can give for a submission.
    /// </summary>
    public enum Recommendation
    {
        StrongReject = 1,
        Reject = 2,
        WeakReject = 3,
        WeakAccept = 4,
        Accept = 5,
        StrongAccept = 6
    }

    /// <summary>
    /// The type of expertise levels of a reviewer in their review.
    /// </summary>
    public enum Expertise
    {
        InformedOutsider = 1,
        Knowledgeable = 2,
        Expert = 3
    }
}