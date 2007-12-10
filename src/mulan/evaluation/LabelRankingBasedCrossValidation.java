package mulan.evaluation;

/**
 * @author Eleftherios Spyromitros - Xioufis
 */

public class LabelRankingBasedCrossValidation extends
		LabelRankingBasedEvaluation {
protected LabelRankingBasedEvaluation[] folds;
	
	
	protected LabelRankingBasedCrossValidation(LabelRankingBasedEvaluation[] folds)
	{
		this.folds = folds;
		computeMeasures();
	}

	protected void computeMeasures()
	{
		one_error   = 0;
		coverage    = 0;
		rloss		= 0;
		avg_precision = 0;
		
		for(int i = 0; i < folds.length; i++)
		{
			one_error       += folds[i].one_error;
			coverage 		+= folds[i].coverage;
			rloss 			+= folds[i].rloss;
			avg_precision   += folds[i].avg_precision;
		}

		int n = folds.length;
		one_error       /= n;
		coverage		/= n;
		rloss			/= n;
		avg_precision   /= n;
	}
}

