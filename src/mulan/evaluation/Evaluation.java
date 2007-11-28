package mulan.evaluation;
/**
 * Simple aggregation class for both types of evaluation, 
 * example based and label based.
 */
public class Evaluation
{
	protected LabelBasedEvaluation labelBased;

	protected ExampleBasedEvaluation exampleBased;

	
	protected Evaluation(LabelBasedEvaluation labelBased,
			ExampleBasedEvaluation exampleBased)
	{
		this.labelBased = labelBased;
		this.exampleBased = exampleBased;
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
	
        public String toString() {
            String description = "";

            description += "HammingLoss    : " + exampleBased.hammingLoss() + "\n";
            description += "SubsetAccuracy : " + exampleBased.subsetAccuracy() + "\n";
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
