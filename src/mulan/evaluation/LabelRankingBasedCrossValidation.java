package mulan.evaluation;

public class LabelRankingBasedCrossValidation extends
		LabelRankingBasedEvaluation {
protected LabelRankingBasedEvaluation[] folds;
	
	
	protected LabelRankingBasedCrossValidation(LabelRankingBasedEvaluation[] folds)
	{
		this(folds, 1.0);
	}
	protected LabelRankingBasedCrossValidation(LabelRankingBasedEvaluation[] folds, 
			double forgivenessRate)
	{
		super(forgivenessRate);
		this.folds = folds;
		computeMeasures();
	}
	
	protected void computeMeasures()
	{
		one_error   = 0;
		coverage    = 0;
		
		for(int i = 0; i < folds.length; i++)
		{
			one_error       += folds[i].one_error;
			coverage 		+= folds[i].coverage;
		}

		int n = folds.length;
		one_error       /= n;
		coverage		/= n;
	}
}

