package mulan.evaluation;

import weka.core.Utils;
import java.util.ArrayList;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.FastVector;
import weka.core.Instances;

/**
 * The purpose of this class is to provide a single point of reference for the
 * calculation of all evaluation metrics
 * 
 * @author greg
 * @author lef
 * @version $Revision: 0.02 $ 
 */
public class IntegratedEvaluation {

	/**
	 * This is all the information needed to derive the measures and curves.
	 * 
	 * The predictions array contains one entry for each test example (1st
	 * dimension) and label (2nd dimension) containing a BinaryPrediction object
	 */
	protected BinaryPrediction[][] predictions;
	
	protected double numPredictedLabels;

	// Example based measures and parameters
	protected double hammingLoss;
	protected double subsetAccuracy;
	protected double accuracy;
	protected double recall;
	protected double precision;
	protected double fmeasure;
	protected double forgivenessRate = 1.0;

	// -- Measures per specific label
	protected double[] labelAccuracy;
	protected double[] labelRecall;
	protected double[] labelPrecision;
	protected double[] labelFmeasure;
        protected double[] labelAUC;

	// -- Micro and macro average measures
	// -- Note that accuracy is equivalent to hammingLoss
	protected double microRecall;
	protected double microPrecision;
	protected double microFmeasure;
        protected double microAUC;
	protected double macroRecall;
	protected double macroPrecision;
	protected double macroFmeasure;
        protected double macroAUC;

	// ranking measures 
	protected double one_error;
	protected double coverage;
	protected double rloss;
	protected double avg_precision;

	public IntegratedEvaluation(){}
	
	public IntegratedEvaluation(BinaryPrediction[][] predictions) {
		this.predictions = predictions;
		computeMeasures();
	}

	protected double computeFMeasure(double precision, double recall) {
		if (Utils.eq(precision + recall, 0))
			return 0;
		else
			return 2 * precision * recall / (precision + recall);
	}

        /**
         * @return the main predictions object  
         */
        public BinaryPrediction[][] getPredictions() 
        {
            return predictions;
        }
        
	public void setForgivenessRate(double rate) {
		forgivenessRate = rate;
	}

	public double getForgivenessRate() {
		return forgivenessRate;
	}

	/**
	 * @return size of the testset. (total number of predictions)
	 */
	protected int numInstances() {
		return predictions.length;
	}

	/**
	 * 
	 * @return total number of possible labels
	 */
	protected int numLabels() {
		return predictions[0].length;
	}

	protected void computeMeasures() //throws Exception
	{  
		int numLabels = numLabels();
		int numInstances = numInstances();
		
		numPredictedLabels = 0;

		// Reset measures in case of multiple calls
		// -- example-based 
		accuracy = 0;
		hammingLoss = 0;
		precision = 0;
		recall = 0;
		fmeasure = 0;
		subsetAccuracy = 0;

		// -- ranking
		one_error = 0;
		coverage = 0;
		rloss = 0;
		avg_precision = 0;
                int examplesToIgnore = 0;

		// label-based counters
                FastVector[] m_Predictions = new FastVector[numLabels]; //for AUC
		for (int j=0; j<numLabels; j++)
                    m_Predictions[j] = new FastVector();
                FastVector all_Predictions = new FastVector();
                double[] falsePositives = new double[numLabels];
		double[] truePositives = new double[numLabels];
		double[] falseNegatives = new double[numLabels];
		double[] trueNegatives = new double[numLabels];

		labelAccuracy = new double[numLabels];
		labelRecall = new double[numLabels];
		labelPrecision = new double[numLabels];
		labelFmeasure = new double[numLabels];
                labelAUC = new double[numLabels];

		for (int i = 0; i < numInstances; i++) {
			//Counter variables
			//Counters are doubles to avoid typecasting
			//when performing divisions. It makes the code a
			//little cleaner but:
			//TODO: run performance tests on counting with doubles
			
			for (int j = 0; j < numLabels; j++) {
				if (predictions[i][j].predicted == true) {
					numPredictedLabels++;
				}
			}

			// example-based counters
			double setUnion = 0; // |Y or Z|
			double setIntersection = 0; // |Y and Z|
			double labelPredicted = 0; // |Z|
			double labelActual = 0; // |Y|
			double symmetricDifference = 0; // |Y xor Z|
			boolean setsIdentical = true; // innocent until proven guilty

			// ranking counters
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

			// store the indexes of true and false labels separately
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


                        if (true_indexes.size() == 0 || false_indexes.size() == 0)
                            examplesToIgnore++;
                        else {
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

                            //======average precision related=============
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
                            // division with |Yi|
                            rel_rankj /= true_indexes.size();
                            avg_precision += rel_rankj;
                        }
                        
			//Do the counting
			for (int j = 0; j < numLabels; j++) {
                            boolean actual = predictions[i][j].actual;
                            boolean predicted = predictions[i][j].predicted;

                            // example-based counters
                            if (predicted != actual) {
                                    symmetricDifference++;
                                    if (setsIdentical)
                                            setsIdentical = false;
                            }

                            if (actual)
                                    labelActual++;
                            if (predicted)
                                    labelPredicted++;
                            if (predicted && actual)
                                    setIntersection++;
                            if (predicted || actual)
                                    setUnion++;

                            // label-based counters
                            int classValue;
                            double[] dist = new double[2];
                            dist[1] = predictions[i][j].confidenceTrue;
                            dist[0] = 1 - dist[1];
                            if (predictions[i][j].actual)
                                classValue = 1;
                            else
                                classValue = 0;
                            m_Predictions[j].addElement(new NominalPrediction(classValue, dist, 1));
                            all_Predictions.addElement(new NominalPrediction(classValue, dist, 1));

                            if (actual && predicted)
                                    truePositives[j]++;
                            else if (!actual && !predicted)
                                    trueNegatives[j]++;
                            else if (predicted)
                                    falsePositives[j]++;
                            else
                                    falseNegatives[j]++;
			}

			// example-based counters
			if (setsIdentical)
				subsetAccuracy++;

			if (Utils.eq(labelActual + labelPredicted, 0)) {
				accuracy += 1;
				recall += 1;
				precision += 1;
				fmeasure += 1;
			} else {
				if (Utils.eq(forgivenessRate, 1.0))
					accuracy += (setIntersection / setUnion);
				else
					accuracy += Math.pow(setIntersection / setUnion, forgivenessRate);

				if (labelPredicted > 0)
					precision += (setIntersection / labelPredicted);
				if (labelActual > 0)
					recall += (setIntersection / labelActual);
			}
			hammingLoss += (symmetricDifference / numLabels);
		}

		// Set final values for example-based measures
		hammingLoss /= numInstances;
		accuracy /= numInstances;
		precision /= numInstances;
		recall /= numInstances;
		subsetAccuracy /= numInstances;
		fmeasure = computeFMeasure(precision, recall);

		//Compute macro averaged label-based measures
		for (int i = 0; i < numLabels; i++) {
			labelAccuracy[i] = (truePositives[i] + trueNegatives[i]) / numInstances;

			labelRecall[i] = truePositives[i] + falseNegatives[i] == 0 ? 0 : truePositives[i]
					/ (truePositives[i] + falseNegatives[i]);

			labelPrecision[i] = truePositives[i] + falsePositives[i] == 0 ? 0 : truePositives[i]
					/ (truePositives[i] + falsePositives[i]);

			labelFmeasure[i] = computeFMeasure(labelPrecision[i], labelRecall[i]);
                        ThresholdCurve tc = new ThresholdCurve();
                        Instances result = tc.getCurve(m_Predictions[i], 1);
                        labelAUC[i] = ThresholdCurve.getROCArea(result);
		}
		macroRecall = Utils.mean(labelRecall);
		macroPrecision = Utils.mean(labelPrecision);
		macroFmeasure = Utils.mean(labelFmeasure);
                macroAUC = Utils.mean(labelAUC);
                
		//Compute micro averaged measures
		double tp = Utils.sum(truePositives);
		//double tn = Utils.sum(trueNegatives);
		double fp = Utils.sum(falsePositives);
		double fn = Utils.sum(falseNegatives);

		microRecall = tp + fn == 0 ? 0 : tp / (tp + fn);
		microPrecision = tp + fp == 0 ? 0 : tp / (tp + fp);
		microFmeasure = computeFMeasure(microPrecision, microRecall);
                ThresholdCurve tc = new ThresholdCurve();
                Instances result = tc.getCurve(all_Predictions, 1);
                microAUC = ThresholdCurve.getROCArea(result);                
                
		// Finalize computation of ranking measures
		one_error /= numInstances;
		coverage /= numInstances;
		rloss /= (numInstances-examplesToIgnore);
		avg_precision /= (numInstances-examplesToIgnore);
		
		numPredictedLabels /= numInstances;
	}

	// Methods used to obtain the calculated measures
	// -- example-based measures

	public double hammingLoss() {
		return hammingLoss;
	}

	public double accuracy() {
		return accuracy;
	}

	public double recall() {
		return recall;
	}

	public double precision() {
		return precision;
	}

	public double fmeasure() {
		return fmeasure;
	}

	public double subsetAccuracy() {
		return subsetAccuracy;
	}

	// -- label-based measures
	
	public double accuracy(int label)
	{
		return labelAccuracy[label];
	}

	public double recall(int label)
	{
		return labelRecall[label];
	}
	
	public double precision(int label)
	{
		return labelPrecision[label];
	}
	
	public double fmeasure(int label)
	{
		return labelFmeasure[label];
	}


	public double microFmeasure() {
		return microFmeasure;
	}

	public double microPrecision() {
		return microPrecision;
	}

	public double microRecall() {
		return microRecall;
	}

	public double microAUC() {
		return microAUC;
	}


	public double macroFmeasure() {
		return macroFmeasure;
	}

	public double macroPrecision() {
		return macroPrecision;
	}

	public double macroRecall() {
		return macroRecall;
	}
        
	public double macroAUC() {
		return macroAUC;
	}

	// -- ranking-based measures

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
	public String toString() {
		String description = "";
		
		description += "Average predicted labels: " + this.numPredictedLabels + "\n";
		description += "========Example Based Measures========\n";
		description += "HammingLoss  : " + hammingLoss() + "\n";
		description += "Accuracy     : " + accuracy() + "\n";
		description += "Precision    : " + precision() + "\n";
		description += "Recall       : " + recall() + "\n";
		description += "Fmeasure     : " + fmeasure() + "\n";
		description += "SubsetAccuracy : " + subsetAccuracy() + "\n";
		description += "========Label Based Measures========\n";
		description += "MICRO\n";
		description += "Precision    : " + microPrecision() + "\n";
		description += "Recall       : " + microRecall() + "\n";
		description += "F1           : " + microFmeasure() + "\n";
		description += "AUC          : " + microAUC() + "\n";
		description += "MACRO\n";
		description += "Precision    : " + macroPrecision() + "\n";
		description += "Recall       : " + macroRecall() + "\n";
		description += "F1           : " + macroFmeasure() + "\n";
		description += "AUC          : " + macroAUC() + "\n";
		description += "========Ranking Based Measures========\n";
		description += "One-error    : " + one_error() + "\n";
		description += "Coverage     : " + coverage() + "\n";
		description += "Ranking Loss : " + rloss() + "\n";
		description += "AvgPrecision : " + avg_precision() + "\n";
		description += "========Per Class Measures========\n";
		for (int i = 0; i < numLabels(); i++) {
			description += "Label " + i + " Accuracy   :" + labelAccuracy[i] + "\n";
			description += "Label " + i + " Precision  :" + labelPrecision[i] + "\n";
			description += "Label " + i + " Recall     :" + labelRecall[i] + "\n";
			description += "Label " + i + " F1         :" + labelFmeasure[i] + "\n";
		}
		
		return description;
	}

	//method for easier data extraction
	public String toExcel(){
		String output = "";
		
		output += hammingLoss()+ ";" + accuracy()+ ";" + precision()+ ";";
		output += recall() + ";" + fmeasure() + ";" + subsetAccuracy() + ";";
		output += microPrecision() + ";" + microRecall() + ";";
		output += microFmeasure() + ";" + microAUC() + ";" + macroPrecision() + ";";
		output += macroRecall() + ";" + macroFmeasure() + ";" + macroAUC() + ";" + one_error()+ ";";
		output += coverage() + ";" + rloss() + ";" + avg_precision();
		output += ";" + this.numPredictedLabels;
                /*
                for (int i = 0; i < numLabels(); i++) {
                    output += ";" + labelAccuracy[i];
		}
                */
		return output;
	}
	
}