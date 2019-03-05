using System;

namespace ReviewerCalibration
{
    public static class Program
    {
        static void Main(string[] args)
        {
            // Add your data here
            Review[] reviews = new Review[0];
            ReviewerCalibration reviewerCalibration = new ReviewerCalibration();
            var results = reviewerCalibration.Run(reviews);
            // Display the results here
        }
    }
}
