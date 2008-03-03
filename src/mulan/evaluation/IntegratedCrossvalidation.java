package mulan.evaluation;

import java.util.*;

/**
 * CrossValidation - has identical semantics with Evaluation.
 * User is passed an instance of this class when calling
 * Evaluator.crossValidate() and friends.
 * 
 * @author lef
 */
public class IntegratedCrossvalidation extends IntegratedEvaluation {
	
	protected int numFolds;
	
	protected IntegratedEvaluation[] folds;
	
	// variables holding the standard deviation of each metric
	protected double std_one_error;
	protected double std_coverage;
	protected double std_rloss;
	protected double std_avg_precision;
	
	public IntegratedCrossvalidation(IntegratedEvaluation[] folds){
		this.folds = folds;
		computeMeasures();
	}

	public int numFolds() {
		return numFolds;
	}
	
	protected void computeMeasures()
	{
		//label based
		int numLabels  = folds[0].numLabels();
		labelAccuracy  = new double[numLabels];
		labelRecall    = new double[numLabels];  
		labelPrecision = new double[numLabels];
		labelFmeasure  = new double[numLabels];
		microRecall    = 0;
		microPrecision = 0;
		microFmeasure  = 0;
		microAccuracy  = 0;
		macroRecall    = 0;
		macroPrecision = 0;
		macroFmeasure  = 0;
		macroAccuracy  = 0;
		//example based
		accuracy       = 0;
		recall         = 0;
		precision      = 0;
		fmeasure       = 0;
		hammingLoss    = 0;
		subsetAccuracy = 0;
		//ranking based
		one_error   = 0;
		coverage    = 0;
		rloss		= 0;
		avg_precision = 0;
		
		for(int i = 0; i < folds.length; i++)
		{
			//label based
			for(int j = 0; j < numLabels; j++)
			{
				labelAccuracy[j]  += folds[i].accuracy(j);
				labelRecall[j]    += folds[i].recall(j);
				labelPrecision[j] += folds[i].precision(j);
				labelFmeasure[j]  += folds[i].precision(j);
			}
			microRecall    += folds[i].microRecall;
			microPrecision += folds[i].microPrecision;
			microFmeasure  += folds[i].microFmeasure;
			microAccuracy  += folds[i].microAccuracy;
			macroRecall    += folds[i].macroRecall;
			macroPrecision += folds[i].macroPrecision;
			macroFmeasure  += folds[i].macroFmeasure;
			macroAccuracy  += folds[i].macroAccuracy;
			//example based
			accuracy       += folds[i].accuracy;
			recall         += folds[i].recall;
			precision      += folds[i].precision;
			fmeasure       += folds[i].fmeasure;
			hammingLoss    += folds[i].hammingLoss;
			subsetAccuracy += folds[i].subsetAccuracy;
			//ranking based
			one_error       += folds[i].one_error;
			coverage 		+= folds[i].coverage;
			rloss 			+= folds[i].rloss;
			avg_precision   += folds[i].avg_precision;
		}

		int n = folds.length;
		//label-based
		for(int i = 0; i < numLabels; i++)
		{
			labelAccuracy[i]  /= n;
			labelRecall[i]    /= n;
			labelPrecision[i] /= n;
			labelFmeasure[i]  /= n;
		}
		microRecall    /= n;
		microPrecision /= n;
		microFmeasure  /= n;
		microAccuracy  /= n;
		macroRecall    /= n;
		macroPrecision /= n;
		macroFmeasure  /= n;
		macroAccuracy  /= n;
		//example based
		accuracy       /= n;
		recall         /= n;
		precision      /= n;
		fmeasure       /= n;
		hammingLoss    /= n;
		subsetAccuracy /= n;
		//ranking based
		one_error       /= n;
		coverage		/= n;
		rloss			/= n;
		avg_precision   /= n;
		
		std_one_error = 0;
		std_coverage = 0;
		std_rloss = 0;
		std_avg_precision = 0;

		// calculation of standard deviation
		// TODO: implement for other metrics too
		for(int i =0;i < folds.length;i++){
			std_one_error += Math.pow(folds[i].one_error - one_error,2);
			std_coverage += Math.pow(folds[i].coverage - coverage,2);
			std_rloss += Math.pow(folds[i].rloss - rloss,2);
			std_avg_precision += Math.pow(folds[i].avg_precision - avg_precision,2);
		}
		std_one_error = Math.pow(std_one_error/n, 0.5);
		std_coverage = Math.pow(std_coverage/n, 0.5);
		std_rloss = Math.pow(std_rloss/n, 0.5);
		std_avg_precision = Math.pow(std_avg_precision/n, 0.5);
	}
	
	public String toString() {
		String description = "";
		
		description += "========Cross Validation========\n";
		description += "========Example Based Measures========\n";
		description += "HammingLoss    : " + this.hammingLoss() + "\n";
		description += "Accuracy       : " + this.accuracy() + "\n";
		description += "Precision      : " + this.precision() + "\n";
		description += "Recall         : " + this.recall() + "\n";
		description += "Fmeasure       : " + this.fmeasure() + "\n";
		description += "SubsetAccuracy : " + this.subsetAccuracy() + "\n";
		description += "========Label Based Measures========\n";
		description += "MICRO\n";
		description += "Accuracy       : " + this.microAccuracy() + "\n";
		description += "Precision      : " + this.microPrecision() + "\n";
		description += "Recall         : " + this.microRecall() + "\n";
		description += "F1             : " + this.microFmeasure() + "\n";
		description += "MACRO\n";
		description += "Accuracy       : " + this.macroAccuracy() + "\n";
		description += "Precision      : " + this.macroPrecision() + "\n";
		description += "Recall         : " + this.macroRecall() + "\n";
		description += "F1             : " + this.macroFmeasure() + "\n";
		description += "========Ranking Based Measures========\n";
		description += "One-error      : " + this.one_error() + " +- " + std_one_error  + "\n";
		description += "Coverage       : " + this.coverage() + " +- " + std_coverage  + "\n";
		description += "Ranking Loss   : " + this.rloss() + " +- " + std_rloss  + "\n";
		description += "AvgPrecision   : " + this.avg_precision() + " +- " + std_avg_precision
				+ "\n";
		description += "========Per Class Measures========\n";
		int numLabels = folds[0].numLabels();
		for (int i = 0; i < numLabels; i++) {
			description += "Label " + i + " Accuracy   :" + labelAccuracy[i] + "\n";
			description += "Label " + i + " Precision  :" + labelPrecision[i] + "\n";
			description += "Label " + i + " Recall     :" + labelRecall[i] + "\n";
			description += "Label " + i + " F1         :" + labelFmeasure[i] + "\n";
		}
		return description;
	}
}
