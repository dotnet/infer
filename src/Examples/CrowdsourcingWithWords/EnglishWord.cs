namespace CrowdsourcingWithWords
{
	/// <summary>
	/// This is a placeholder class.  Replace with your own stemmer.
	/// </summary>
	public class EnglishWord
	{
		public readonly string Original;
		public readonly string Stem;

		public EnglishWord(string word)
		{
			Original = word;
			Stem = word;
		}
	}
}