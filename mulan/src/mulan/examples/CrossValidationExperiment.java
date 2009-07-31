package mulan.examples;

/**
 *
 * @author greg
 */

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HMC;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.IncludeLabelsClassifier;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.MultiClassLearner;
import mulan.classifier.transformation.MultiLabelStacking;
import mulan.core.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.transformations.multiclass.Copy;
import mulan.transformations.multiclass.Ignore;
import mulan.transformations.multiclass.MultiClassTransformation;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class CrossValidationExperiment {

	/**
	 * Creates a new instance of this class
	 */
	public CrossValidationExperiment() {
	}

	public static void main(String[] args) {
		String[] methodsToCompare = { "HOMER", "BR", "CLR", "MLkNN", "MC-Copy",
				"IncludeLabels", "MC-Ignore", "RAkEL", "LP", "MLStacking" };

		try {
			String path = Utils.getOption("path", args);
			String filestem = Utils.getOption("filestem", args);
			System.out.println("Loading the data set");
			MultiLabelInstances dataSet = new MultiLabelInstances(path
					+ filestem + ".arff", path + filestem + ".xml");

			Evaluator eval = new Evaluator();
			Evaluation results;

			int numFolds = 10;

			for (int i = 0; i < methodsToCompare.length; i++) {

				if (methodsToCompare[i].equals("BR")) {
					System.out.println(methodsToCompare[i]);
					Classifier brClassifier = new NaiveBayes();
					BinaryRelevance br = new BinaryRelevance(brClassifier);
					br.setDebug(true);
					results = eval.crossValidate(br, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("LP")) {
					System.out.println(methodsToCompare[i]);
					Classifier lpBaseClassifier = new J48();
					LabelPowerset lp = new LabelPowerset(lpBaseClassifier);
					lp.setDebug(true);
					results = eval.crossValidate(lp, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("CLR")) {
					System.out.println(methodsToCompare[i]);
					Classifier clrClassifier = new J48();
					CalibratedLabelRanking clr = new CalibratedLabelRanking(
							clrClassifier);
					clr.setDebug(true);
					results = eval.crossValidate(clr, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("RAkEL")) {
					System.out.println(methodsToCompare[i]);
					MultiLabelLearner lp = new LabelPowerset(new J48());
					RAkEL rakel = new RAkEL(lp);
					rakel.setDebug(true);
					results = eval.crossValidate(rakel, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("MC-Copy")) {
					System.out.println(methodsToCompare[i]);
					Classifier mclClassifier = new J48();
					MultiClassTransformation mcTrans = new Copy();
					MultiClassLearner mcl = new MultiClassLearner(
							mclClassifier, mcTrans);
					mcl.setDebug(true);
					results = eval.crossValidate(mcl, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("MC-Ignore")) {
					System.out.println(methodsToCompare[i]);
					Classifier mclClassifier = new J48();
					MultiClassTransformation mcTrans = new Ignore();
					MultiClassLearner mcl = new MultiClassLearner(
							mclClassifier, mcTrans);
					results = eval.crossValidate(mcl, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("IncludeLabels")) {
					System.out.println(methodsToCompare[i]);
					Classifier ilClassifier = new J48();
					IncludeLabelsClassifier il = new IncludeLabelsClassifier(
							ilClassifier);
					il.setDebug(true);
					results = eval.crossValidate(il, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("MLkNN")) {
					System.out.println(methodsToCompare[i]);
					int numOfNeighbors = 10;
					double smooth = 1.0;
					MLkNN mlknn = new MLkNN(numOfNeighbors, smooth);
					mlknn.setDebug(true);
					results = eval.crossValidate(mlknn, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("HMC")) {
					System.out.println(methodsToCompare[i]);
					Classifier baseClassifier = new J48();
					LabelPowerset lp = new LabelPowerset(baseClassifier);
					RAkEL rakel = new RAkEL(lp);
					HMC hmc = new HMC(rakel);
					results = eval.crossValidate(hmc, dataSet, numFolds);
					System.out.println(results.toString());
				}

				if (methodsToCompare[i].equals("HOMER")) {
					System.out.println(methodsToCompare[i]);
					Classifier baseClassifier = new SMO();
					CalibratedLabelRanking learner = new CalibratedLabelRanking(
							baseClassifier);
					learner.setDebug(true);
					HOMER homer = new HOMER(learner, 3, HierarchyBuilder.Method.Random);
					homer.setDebug(true);
					results = eval.crossValidate(homer, dataSet, numFolds);
					System.out.println(results.toString());
				}
				if (methodsToCompare[i].equals("MLStacking")) {
					System.out.println(methodsToCompare[i]);
					J48 baseClassifier = new J48();
					J48 metaClassifier = new J48();
					baseClassifier.setUseLaplace(true);
					metaClassifier.setUseLaplace(true);
					MultiLabelStacking mls = new MultiLabelStacking(
							baseClassifier, metaClassifier, 10);
					mls.setDebug(true);
					mls.setPhival(0.06);
					results = eval.crossValidate(mls, dataSet, numFolds);
					System.out.println(results.toString());
				}

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
