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
	private RankingBasedMeasures rankingBasedMeasures;
    private ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures;

	
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

	public RankingBasedMeasures getRankingBasedMeasures() {
		return rankingBasedMeasures;
	}
	
	protected void setRankingBasedMeasures(RankingBasedMeasures rankingBasedMeasures) {
		this.rankingBasedMeasures = rankingBasedMeasures;
	}

	protected void setConfidenceLabelBasedMeasures(ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures) {
		this.confidenceLabelBasedMeasures = confidenceLabelBasedMeasures;
	}

	public ConfidenceLabelBasedMeasures getConfidenceLabelBasedMeasures() {
		return confidenceLabelBasedMeasures;
	}

    @Override
	public String toString() {
		String description = "";

        if (exampleBasedMeasures != null) {
//		description += "Average predicted labels: " + this.numPredictedLabels + "\n";
            description += "========Example Based Measures========\n";
            description += "HammingLoss    : " + exampleBasedMeasures.getHammingLoss() + "\n";
            description += "Accuracy       : " + exampleBasedMeasures.getAccuracy() + "\n";
            description += "Precision      : " + exampleBasedMeasures.getPrecision() + "\n";
            description += "Recall         : " + exampleBasedMeasures.getRecall() + "\n";
            description += "Fmeasure       : " + exampleBasedMeasures.getFMeasure() + "\n";
            description += "SubsetAccuracy : " + exampleBasedMeasures.getSubsetAccuracy() + "\n";
        }
        if (labelBasedMeasures != null) {
            description += "========Label Based Measures========\n";
            description += "MICRO\n";
            description += "Precision    : " + labelBasedMeasures.getPrecision(Averaging.MICRO) + "\n";
            description += "Recall       : " + labelBasedMeasures.getRecall(Averaging.MICRO) + "\n";
            description += "F1           : " + labelBasedMeasures.getFMeasure(Averaging.MICRO) + "\n";
            description += "MACRO\n";
            description += "Precision    : " + labelBasedMeasures.getPrecision(Averaging.MACRO) + "\n";
            description += "Recall       : " + labelBasedMeasures.getRecall(Averaging.MACRO) + "\n";
            description += "F1           : " + labelBasedMeasures.getFMeasure(Averaging.MACRO) + "\n";
        }
        if (confidenceLabelBasedMeasures != null) {
            description += "MICRO\n";
            description += "AUC          : " + confidenceLabelBasedMeasures.getAUC(Averaging.MICRO) + "\n";
            description += "MACRO\n";
            description += "AUC          : " + confidenceLabelBasedMeasures.getAUC(Averaging.MACRO) + "\n";
        }
        if (rankingBasedMeasures != null) {
            description += "========Ranking Based Measures========\n";
            description += "One-error    : " + rankingBasedMeasures.getOneError() + "\n";
            description += "Coverage     : " + rankingBasedMeasures.getCoverage() + "\n";
            description += "Ranking Loss : " + rankingBasedMeasures.getRankingLoss() + "\n";
            description += "AvgPrecision : " + rankingBasedMeasures.getAvgPrecision() + "\n";
        }
        /*
        description += "========Per Class Measures========\n";
		for (int i = 0; i < numLabels(); i++) {
			description += "Label " + i + " Accuracy   :" + labelAccuracy[i] + "\n";
			description += "Label " + i + " Precision  :" + labelPrecision[i] + "\n";
			description += "Label " + i + " Recall     :" + labelRecall[i] + "\n";
			description += "Label " + i + " F1         :" + labelFmeasure[i] + "\n";
		}
		*/
		return description;
	}
}
