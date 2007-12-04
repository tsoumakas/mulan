package mulan.evaluation;

public class LabelBasedCrossValidation extends LabelBasedEvaluation
{
	protected LabelBasedEvaluation[] folds;
	
	protected LabelBasedCrossValidation(LabelBasedEvaluation[] folds) throws Exception
	{
		this.folds = folds;
		computeMeasures();
	}
	
	protected void computeMeasures()
	{
		
		int numLabels  = folds[0].numLabels();
		labelAccuracy  = new double[numLabels];
		labelRecall    = new double[numLabels];  
		labelPrecision = new double[numLabels];
		labelFmeasure  = new double[numLabels];
		
		for(int i = 0; i < folds.length; i++)
		{
			accuracy[MICRO]  += folds[i].accuracy[MICRO];
			recall[MICRO]    += folds[i].recall[MICRO];
			precision[MICRO] += folds[i].precision[MICRO];
			fmeasure[MICRO]  += folds[i].fmeasure[MICRO];
			
			accuracy[MACRO]  += folds[i].accuracy[MACRO];
			recall[MACRO]    += folds[i].recall[MACRO];
			precision[MACRO] += folds[i].precision[MACRO];
			fmeasure[MACRO]  += folds[i].fmeasure[MACRO];
			
			for(int j = 0; j < numLabels; j++)
			{
				labelAccuracy[j]  += folds[i].accuracy(j);
				labelRecall[j]    += folds[i].recall(j);
				labelPrecision[j] += folds[i].precision(j);
				labelFmeasure[j]  += folds[i].precision(j);
			}
		}

		int n = folds.length;
		accuracy[MICRO]  /= n;
		recall[MICRO]    /= n;
		precision[MICRO] /= n;
		fmeasure[MICRO]  /= n;
		
		accuracy[MACRO]  /= n;
		recall[MACRO]    /= n;
		precision[MACRO] /= n;
		fmeasure[MACRO]  /= n;
		
		for(int i = 0; i < numLabels; i++)
		{
			labelAccuracy[i]  /= n;
			labelRecall[i]    /= n;
			labelPrecision[i] /= n;
			labelFmeasure[i]  /= n;
		}
	}
}
