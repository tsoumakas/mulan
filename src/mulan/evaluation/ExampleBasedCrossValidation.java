package mulan.evaluation;

public class ExampleBasedCrossValidation extends ExampleBasedEvaluation
{
	protected ExampleBasedEvaluation[] folds;
	
	protected ExampleBasedCrossValidation(ExampleBasedEvaluation[] folds)
	{
		this(folds, 1.0);
	}
	protected ExampleBasedCrossValidation(ExampleBasedEvaluation[] folds, 
			double forgivenessRate)
	{
		super(forgivenessRate);
		this.folds = folds;
		computeMeasures();
	}
	
	protected void computeMeasures()
	{
		accuracy    = 0;
		recall      = 0;
		precision   = 0;
		fmeasure    = 0;
		hammingLoss = 0;
		subsetAccuracy = 0;
		
		for(int i = 0; i < folds.length; i++)
		{
			accuracy       += folds[i].accuracy;
			recall         += folds[i].recall;
			precision      += folds[i].precision;
			fmeasure       += folds[i].fmeasure;
			hammingLoss    += folds[i].hammingLoss;
			subsetAccuracy += folds[i].subsetAccuracy;
		}

		int n = folds.length;
		accuracy       /= n;
		recall         /= n;
		precision      /= n;
		fmeasure       /= n;
		hammingLoss    /= n;
		subsetAccuracy /= n;
		
	}
}
