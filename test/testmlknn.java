import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import mulan.Statistics;
import mulan.classifier.MultiLabelKNN;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.LabelBasedEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

/**
 * Test class for the multi-label knn classifier
 * 
 * @author Eleftherios Spyromitros - Xioufis
 */

public class testmlknn {

	public testmlknn() {
	}

	public static void main(String[] args) throws Exception {

		String path = "E:/Documents and Settings/lefman/Desktop/my workspace/datasets/";
		String datastem = "yeast";

		FileReader frTrain = new FileReader(path + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + datastem + "-test.arff");
		Instances testData = new Instances(frTest);

		Instances allData = new Instances(trainData);
		for (int i = 0; i < testData.numInstances(); i++)
			allData.add(testData.instance(i));

		int numLabels = 14;
		// int numNeighbours = 2;

		for (int i = 7; i <= 7; i++) {
			System.out.println("Calculating mlknn output for " + i
					+ " neighbours");

			MultiLabelKNN mlknn = new MultiLabelKNN(numLabels, i, 1);
			mlknn.setdontnormalize(true);

			long start = System.currentTimeMillis();
			mlknn.buildClassifier(trainData);
			long end = System.currentTimeMillis();

			System.out.print("Buildclassifier Time: " + (end - start) + "\n");

			Evaluator eval;
			Evaluation results;
			eval = new Evaluator();
			//results = eval.crossValidate(mlknn, allData);

			start = System.currentTimeMillis();
			results = eval.evaluate(mlknn, testData);
			end = System.currentTimeMillis();
			
			System.out.print("Evaluation Time: " + (end - start) + "\n");

			System.out.println(results.toString());
			System.gc();

			// mlknn.output(); //outputs prior and conditional probabilities
		}
	}
}