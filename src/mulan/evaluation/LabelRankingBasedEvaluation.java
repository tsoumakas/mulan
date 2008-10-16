package mulan.evaluation;

import weka.core.Utils;
import java.util.ArrayList;

/**
 * Class implementing metrics which are defined based on the real-valued
 * function f <br>
 * which concern the ranking quality of proper labels of the instance.
 * 
 * @author Eleftherios Spyromitros - Xioufis
 */

public class LabelRankingBasedEvaluation extends EvaluationBase {

	protected double one_error;
	protected double coverage;
	protected double rloss;
	protected double avg_precision;

	/**
	 * This constructor is needed by LabelRankingBasedCrossvalidation class
	 */
	protected LabelRankingBasedEvaluation() {
		super(null);
	}

	protected LabelRankingBasedEvaluation(BinaryPrediction[][] predictions) {
		super(predictions);
		computeMeasures();
		//compute_one_error2();
		//compute_coverage();
		//compute_rloss();
		//compute_avg_precision();
	}

	protected void computeMeasures() // throws Exception
	{
		one_error = 0;
		coverage = 0;
		rloss = 0;
		avg_precision = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {
			double ranks[] = new double[numLabels];
			int sorted_ranks[] = new int[numLabels];

			// copy the rankings into new array
			for (int j = 0; j < numLabels; j++) {
				ranks[j] = predictions[i][j].confidenceTrue;
			}
			// sort the array of ranks
			sorted_ranks = Utils.stableSort(ranks);

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

			//======one error related============
			int top_rated = sorted_ranks[numLabels - 1];
			// check if the top rated label is in the set of proper labels
			if (predictions[i][top_rated].actual != true) {
				one_error++;
			}
			//======coverage related=============
			int how_deep = 0;
			for (int j = 0; j < numLabels; j++) {
				if (predictions[i][sorted_ranks[j]].actual == true) {
					how_deep = numLabels - j - 1;
					break;
				}
			}
			coverage += how_deep;

			//======ranking loss related=============
			int rolp = 0; // reversed ordered label pairs
			for (int k = 0; k < true_indexes.size(); k++) {
				for (int l = 0; l < false_indexes.size(); l++) {
					if (predictions[i][true_indexes.get(k)].confidenceTrue <= predictions[i][false_indexes
							.get(l)].confidenceTrue) {
						rolp++;
					}
				}
			}
			rloss += (double) rolp / (true_indexes.size() * false_indexes.size());

			//======average precision related related=============
			double rel_rankj = 0;

			for (int j : true_indexes) {
				int jrating = 0;
				int ranked_abovet = 0;

				// find rank of jth label in the array of ratings
				for (int k = 0; k < numLabels; k++) {
					if (sorted_ranks[k] == j) {
						jrating = k;
						break;
					}
				}
				// count the actually true above ranked labels
				for (int k = jrating + 1; k < numLabels; k++) {
					if (predictions[i][sorted_ranks[k]].actual == true) {
						ranked_abovet++;
					}
				}
				int jrank = numLabels - jrating;
				rel_rankj += (double) (ranked_abovet + 1) / jrank; //+1to include the current label
			}

			// diairoume me to |Yi|
			rel_rankj /= true_indexes.size();

			avg_precision += rel_rankj;
		}

		one_error /= numInstances;
		coverage /= numInstances;
		rloss /= numInstances;
		avg_precision /= numInstances;
	}

	/**
	 * One-error: evaluates how many times the top ranked label is not in the
	 * set of proper labels of the instance.<br>
	 * <br>
	 * The performance is perfect when one_error = 0
	 */
	protected void compute_one_error() {
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
		one_error /= numInstances;
	}

	protected void compute_one_error2() {
		one_error = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {
			double ranks[] = new double[numLabels];
			int sorted_ranks[] = new int[numLabels];

			// copy the rankings into new array
			for (int j = 0; j < numLabels; j++) {
				ranks[j] = predictions[i][j].confidenceTrue;
			}
			// sort the array of ranks
			sorted_ranks = Utils.stableSort(ranks);

			int top_rated = sorted_ranks[numLabels - 1];
			// check if the top rated label is in the set of proper labels
			if (predictions[i][top_rated].actual != true) {
				one_error++;
			}
		}
		one_error /= numInstances;
	}

	/**
	 * Coverage: evaluates how far we need, on the average, to go down to the
	 * list of labels in order to cover all the proper labels of the instance.<br>
	 * <br>
	 * The smaller the value of coverage, the better the performance.
	 */
	protected void compute_coverage() {
		coverage = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++) {
			int how_deep = 0; // to go down the sorted(based on ranking)list of labels

			double ranks[] = new double[numLabels];
			int indexes[] = new int[numLabels];

			// copy the rankings into new array
			for (int j = 0; j < numLabels; j++) {
				ranks[j] = predictions[i][j].confidenceTrue;
			}
			// sort the array of ranks
			indexes = Utils.stableSort(ranks);

			for (int j = 0; j < numLabels; j++) {
				if (predictions[i][indexes[j]].actual == true) {
					how_deep = numLabels - j - 1;
					break;
				}
			}
			coverage += how_deep;
		}
		coverage /= numInstances;
	}

	/**
	 * Ranking Loss: evaluates the average fraction of label pairs that are
	 * reversely ordered for the instance.<br>
	 * <br>
	 * The performance is perfect when rloss = 0. The smaller the value of
	 * rloss, the better the performance.
	 */
	protected void compute_rloss() {
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
			rloss += (double) rolp / (true_indexes.size() * false_indexes.size());
		}
		rloss /= numInstances;
	}

	/**
	 * average precision: evaluates the average fraction of labels ranked above
	 * a particular label y in Y which actually are in Y.<br>
	 * <br>
	 * The performance is perfect when avgprec = 1. The bigger the value of
	 * avgprec the better the performance.
	 */
	protected void compute_avg_precision() {
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
				rel_rankj += (double) (ranked_abovet + 1) / jrank; //+1to include the current label
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

	public String toString() {
		String description = "";

		description += "========Ranking Based Measures========\n";
		description += "One-error      : " + this.one_error() + "\n";
		description += "Coverage       : " + this.coverage() + "\n";
		description += "Ranking Loss   : " + this.rloss() + "\n";
		description += "Avg Precision  : " + this.avg_precision() + "\n";

		return description;
	}

}
