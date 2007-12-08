package mulan.evaluation;
/**
 * Simple aggregation class for both types of evaluation, 
 * example based and label based.
 */
public class Evaluation
{
	protected LabelBasedEvaluation labelBased;

	protected ExampleBasedEvaluation exampleBased;
	
	protected LabelRankingBasedEvaluation rankingBased;

	
	protected Evaluation(LabelBasedEvaluation labelBased,
			ExampleBasedEvaluation exampleBased,LabelRankingBasedEvaluation rankingBased )
	{
		this.labelBased = labelBased;
		this.exampleBased = exampleBased;
		this.rankingBased = rankingBased;
	}
	

	/**
	 * @return the labelBased
	 */
	public LabelBasedEvaluation getLabelBased()
	{
		return labelBased;
	}

	public ExampleBasedEvaluation getExampleBased()
	{
		return exampleBased;
	}
	
	public LabelRankingBasedEvaluation getRankingBased()
	{
		return rankingBased;
	}
	
        public String toString() {
            String description = "";

            description += "HammingLoss    : " + exampleBased.hammingLoss() + "\n";
            description += "SubsetAccuracy : " + exampleBased.subsetAccuracy() + "\n";
            description += "Ranking Based Measures\n";
            description += "One-error      : " + rankingBased.one_error() + "\n";
            description += "Coverage       : " + rankingBased.coverage() + "\n";
            description += "Ranking Loss   : " + rankingBased.rloss() + "\n";
            description += "Avg Precision  : " + rankingBased.avg_precision() + "\n";
            labelBased.setAveragingMethod(LabelBasedEvaluation.MICRO);
            description += "MICRO\n";
            description += "Precision : " + labelBased.precision() + "\n";
            description += "Recall    : " + labelBased.recall() + "\n";
            description += "F1        : " + labelBased.fmeasure() + "\n";
            labelBased.setAveragingMethod(LabelBasedEvaluation.MACRO);
            description += "MACRO\n";
            description += "Precision : " + labelBased.precision() + "\n";
            description += "Recall    : " + labelBased.recall() + "\n";
            description += "F1        : " + labelBased.fmeasure() + "\n";
            
            return description;
        }
}
