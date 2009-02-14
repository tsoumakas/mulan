package mulan.evaluation;

/**
 * Simple aggregation class which provides all possible evaluation measure types.
 * The evaluation is providing measures for particular multi-label learner type.
 * Only measures applicable to evaluated learner will be provided. 
 * Measures which are not applicable will be null. The proper measures are set by 
 * {@link Evaluator} based on predefined rules.
 * 
 * @see Evaluator
 * 
 * @author Jozef Vilcek
 */
public class Evaluation {
	
	private LabelBasedMeasures labelBasedMeasures;
	private ExampleBasedMeasures exampleBasedMeasures;
	private RankingMeasures rankingMeasures;

	
	public LabelBasedMeasures getLabelBasedMeasures() {
		return labelBasedMeasures;
	}

	protected void setLabelBasedMeasures(LabelBasedMeasures labelBasedMeasures) {
		this.labelBasedMeasures = labelBasedMeasures;
	}
	
	public ExampleBasedMeasures getExampleBasedMeasures() {
		return exampleBasedMeasures;
	}
	
	protected void setExampleBasedMeasures(ExampleBasedMeasures exampleBasedMeasures) {
		this.exampleBasedMeasures = exampleBasedMeasures;
	}

	public RankingMeasures getRankingMeasures() {
		return rankingMeasures;
	}
	
	protected void setRankingMeasures(RankingMeasures rankingMeasures) {
		this.rankingMeasures = rankingMeasures;
	}

//	public String toString() {
//		String description = "";
//
//		description += "HammingLoss    : " + exampleBased.hammingLoss() + "\n";
//		description += "SubsetAccuracy : " + exampleBased.subsetAccuracy() + "\n";
//		description += "Ranking Based Measures\n";
//		description += "One-error      : " + rankingBased.one_error() + "\n";
//		description += "Coverage       : " + rankingBased.coverage() + "\n";
//		description += "Ranking Loss   : " + rankingBased.rloss() + "\n";
//		description += "Avg Precision  : " + rankingBased.avg_precision() + "\n";
//		labelBased.setAveragingMethod(LabelBasedEvaluation.MICRO);
//		description += "MICRO\n";
//		description += "Precision : " + labelBased.precision() + "\n";
//		description += "Recall    : " + labelBased.recall() + "\n";
//		description += "F1        : " + labelBased.fmeasure() + "\n";
//		labelBased.setAveragingMethod(LabelBasedEvaluation.MACRO);
//		description += "MACRO\n";
//		description += "Precision : " + labelBased.precision() + "\n";
//		description += "Recall    : " + labelBased.recall() + "\n";
//		description += "F1        : " + labelBased.fmeasure() + "\n";
//
//		return description;
//	}
}
