package mulan.evaluation;

import weka.core.Utils;
import java.util.ArrayList;

public class LabelRankingBasedEvaluation extends EvaluationBase {

	protected double one_error;
	protected double coverage;
	protected double rloss;
	protected double avg_precision;

	// apaiteitai apo tin cross-vallidation constructor xoris orismata
	protected LabelRankingBasedEvaluation() {
		super(null);
	}

	protected LabelRankingBasedEvaluation(BinaryPrediction[][] predictions) {
		super(predictions);
		// computeMeasures();
		compute_one_error();
		compute_coverage();
		compute_rloss();
		compute_avg_precision();
	}

	protected void computeMeasures() // throws Exception
	{
	}

	private void compute_one_error() {
		one_error = 0;

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

		one_error /= numInstances;
	}

	private void compute_coverage() {
		coverage = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {
			int how_deep = 0;
			double ranks[] = new double[numLabels];
			int indexes[] = new int[numLabels];

			// copy the rankings into new array
			for (int j = 0; j < numLabels; j++) {
				ranks[j] = predictions[i][j].confidenceTrue;
			}
			// sort the array of ranks
			indexes = Utils.stableSort(ranks);

			for (int j = 0; j < numLabels; j++) {
				if (predictions[i][j].actual == true) {
					// find the position of jth label in the sorted array.
					for (int k = 0; k < numLabels; k++) {
						if (indexes[k] == j) {
							if (how_deep < (numLabels - k - 1)) {
								how_deep = numLabels - k - 1;
							}
						}
					}
				}

			}
			coverage += how_deep;
		}

		coverage /= numInstances;

	}

	private void compute_rloss() {
		rloss = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {

			// indexes of true and false labels
			ArrayList<Integer> true_indexes = new ArrayList<Integer>();
			ArrayList<Integer> false_indexes = new ArrayList<Integer>();

			// xorizi se true kai false labels apothikeuontas ta indexes
			for (int j = 0; j < numLabels; j++) {
				if (predictions[i][j].actual == true) {
					true_indexes.add(j);
				} else {
					false_indexes.add(j);
				}
			}

			int rolp = 0; // reversed ordered label pairs

			for (int k = 0; k < true_indexes.size(); k++) {
				for (int l = 0; l < false_indexes.size(); l++) {
					if (predictions[i][true_indexes.get(k)].confidenceTrue <= predictions[i][false_indexes
							.get(l)].confidenceTrue) {
						rolp++;
					}
				}
			}
			rloss += (double) rolp
					/ (true_indexes.size() * false_indexes.size());
		}

		rloss /= numInstances;
	}

	private void compute_avg_precision() {
		avg_precision = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {

			double ranks[] = new double[numLabels];
			int indexes[] = new int[numLabels];

			// copy the rankings into new array
			for (int j = 0; j < numLabels; j++) {
				ranks[j] = predictions[i][j].confidenceTrue;
			}
			// sort the array of ranks
			indexes = Utils.stableSort(ranks);

			// indexes of true and false labels
			ArrayList<Integer> true_indexes = new ArrayList<Integer>();
			ArrayList<Integer> false_indexes = new ArrayList<Integer>();

			// xorizi se true kai false labels apothikeuontas ta indexes
			for (int j = 0; j < numLabels; j++) {
				if (predictions[i][j].actual == true) {
					true_indexes.add(j);
				} else {
					false_indexes.add(j);
				}
			}

			double rel_rankj = 0;

			for (int j : true_indexes) {
				int jrating = 0;
				int ranked_abovet = 0;

				// find rank of jth label in the array of ratings
				for (int k = 0; k < numLabels; k++) {
					if (indexes[k] == j) {
						jrating = k;
						break;
					}
				}
				// count the actually true above ranked labels
				for (int k = jrating + 1; k < numLabels; k++) {
					if (predictions[i][indexes[k]].actual == true) {
						ranked_abovet++;
					}
				}
				int jrank = numLabels - jrating;
				rel_rankj += (double) (ranked_abovet + 1) / jrank; //+1 to include the current label
			}

			// diairoume me to |Yi|
			rel_rankj /= true_indexes.size();

			avg_precision += rel_rankj;
		}
		avg_precision /= numInstances;
	}

	public double one_error() {
		return one_error;
	}

	public double coverage() {
		return coverage;
	}

	public double rloss() {
		return rloss;
	}

	public double avg_precision() {
		return avg_precision;
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
