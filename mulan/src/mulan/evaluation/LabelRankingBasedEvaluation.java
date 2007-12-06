package mulan.evaluation;

import weka.core.Utils;
import java.util.Arrays;

public class LabelRankingBasedEvaluation extends EvaluationBase {
	
	protected double one_error;
	protected double coverage;
	
	protected double forgivenessRate;

	
	protected LabelRankingBasedEvaluation(BinaryPrediction[][] predictions,
									 double forgivenessRate)
	{
		super(predictions);
		this.forgivenessRate = forgivenessRate;
		computeMeasures();
	}
	
	protected LabelRankingBasedEvaluation(BinaryPrediction[][] predictions)
	{
		this(predictions, 1.0);
	}
	
	protected LabelRankingBasedEvaluation(double forgivenessRate)
	{
		super(null);
		this.forgivenessRate = forgivenessRate;
	}
	
	protected void computeMeasures() // throws Exception
	{
		// Reset in case of multiple calls
		one_error = 0;
		coverage = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {
			// find the top ranked label for every instance
			int top_rated = 0; // index of top rated label
			for (int j = 1; j < numLabels; j++) {
				if (predictions[i][j].confidenceTrue > predictions[i][top_rated].confidenceTrue)
					top_rated = j;
			}

			// check if it is in the set of proper labels
			if (predictions[i][top_rated].actual != true) {
				one_error++;
			}
		}
		for (int i = 0; i < numInstances; i++){
			int how_deep=0;
			double ranks [] = new double[numLabels];
			int indexes [] = new int[numLabels];
			
			//copy the rankings into new array
			for (int j=0 ;j < numLabels; j++) {
				ranks[j] = predictions[i][j].confidenceTrue;
			}
			//sort the array of ranks
			indexes = Utils.stableSort(ranks);
			
			if (i % 100 == 0) {
				for (int k = 0; k < numLabels; k++) {
					System.out.print(ranks[k] + " ");
				}
				System.out.println();
				for (int k = 0; k < numLabels; k++) {
					System.out.print(indexes[k] + " ");
				}
				System.out.println();
			}
			
			for (int j=0 ;j < numLabels; j++) {
				if(predictions[i][j].actual == true && (numLabels - indexes[j]-1) > how_deep ){
					//find ranking of label j -> position of j's ranking in ranks
					//find j's ranking
				   how_deep = numLabels - indexes[j] - 1;
				}
					
			}
			
			coverage += how_deep;
			
		}

		// Set final values
		one_error /= numInstances;
		coverage /= numInstances;
	}

	public double one_error()
	{
		return one_error;
	}
	
	public double coverage()
	{
		return coverage;
	}
	
	@Override
	public double accuracy() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double fmeasure() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double precision() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double recall() {
		// TODO Auto-generated method stub
		return 0;
	}

}
