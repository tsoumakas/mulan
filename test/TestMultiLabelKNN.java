import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

import mulan.classifier.*;
import mulan.evaluation.*;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import java.text.DecimalFormat;

/**
 * Test class for the multi-label knn classifier
 * 
 * @author Eleftherios Spyromitros - Xioufis
 */

public class TestMultiLabelKNN {

	public TestMultiLabelKNN() {
	}

	public static void main(String[] args) throws Exception {

		String path = "E:/Documents and Settings/lefman/Desktop/my workspace/datasets/";

		String datastem = "yeast";
		//String datastem = "scene";
		//String datastem = "genbase10";

		int numLabels = 14;
		
		boolean dontnormalize = false;

		if (args != null) {
			path = args[0];
			datastem = args[1];
			numLabels = Integer.parseInt(args[2]);
		}
		
		File file = new File(".//" + datastem + " results.txt");
		PrintWriter out = new PrintWriter(new FileWriter(file));

		FileReader frTrain = new FileReader(path + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + datastem + "-test.arff");
		Instances testData = new Instances(frTest);

		Instances allData = new Instances(trainData);
		for (int i = 0; i < testData.numInstances(); i++)
			allData.add(testData.instance(i));

		//names of the tested algorithms
		String[] algorithms = new String[3];
		algorithms[0] = "BRknn";
		algorithms[1] = "MLknn";
		algorithms[2] = "LPknn";

		//Statistics stats = new Statistics();
		//stats.calculateStats(allData, numLabels);
		//System.out.println(stats.toString());

		//test for values of k between 1 and 30
		for (int i = 1; i <= 1; i += 1) {

			BRknn br = new BRknn(numLabels, i);
			br.setDistanceWeighting(1);
			br.setDontNormalize(dontnormalize);
			//br.buildClassifier(trainData);

			Mlknn ml = new Mlknn(numLabels, i, 1);
			ml.setDontNormalize(dontnormalize);
			//ml.buildClassifier(trainData);

			IBk baseClassifier = new IBk(i);

			LabelPowersetClassifier lp = new LabelPowersetClassifier(Classifier
					.makeCopy(baseClassifier), numLabels);
			//lp.buildClassifier(trainData);

			/*//Binary Relevance Classifier 
			//System.out.println("BR");
			BinaryRelevanceClassifier br1 = new BinaryRelevanceClassifier(Classifier
					.makeCopy(baseClassifier), numLabels);
			br1.buildClassifier(trainData);*/
			
			/*=====================EVALUATION==========================
			Evaluator eval;
			IntegratedEvaluation[] results = new IntegratedEvaluation[4];
			eval = new Evaluator();
			//BR EVAULATION 
			long start = System.currentTimeMillis();
			results[0] = eval.evaluateAll(br, testData);
			long end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//System.out.println("Average labels predicted: " + (double) br.getSumedLabels() / trainData.numInstances());
			//BR2 EVAULATION 
			start = System.currentTimeMillis();
			results[1] = eval.evaluateAll(br1, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//System.out.println("Average labels predicted: " + (double) br.getSumedLabels() / trainData.numInstances());
			//ML EVAULATION 
			start = System.currentTimeMillis();
			results[2] = eval.evaluateAll(ml, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//System.out.println("Average labels predicted: " + (double) br.getSumedLabels() / trainData.numInstances());
			 LP EVAULATION 
			start = System.currentTimeMillis();
			results[3] = eval.evaluateAll(lp, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//System.out.println("Average labels predicted: " + (double) br.getSumedLabels() / trainData.numInstances());
			*/
			
			/* =====================CROSS-VALIDATION========================== */
			int numfolds = 10;
			int numsteps = 13;
			double start = 0.2;
			double increment = 0.05;

			DecimalFormat df = new DecimalFormat("0.00");

			Evaluator eval;
			IntegratedCrossvalidation[][] results = new IntegratedCrossvalidation[3][numsteps];
			eval = new Evaluator();

			results[0] = eval.crossvalidateOverThreshold(br, allData, start, increment, numsteps,
					numfolds);
			results[1] = eval.crossvalidateOverThreshold(ml, allData, start, increment, numsteps,
					numfolds);
			//LP can't be evaluated over threshold
			results[2][0] = eval.crossValidateAll(lp, allData, numfolds);

			for (int k = 0; k < algorithms.length; k++) {
				if (algorithms[k] != "LPknn") {
					for (int j = 0; j < numsteps; j++) {
						out.println(i + ";" + algorithms[k] + ";" + datastem + ";"
								+ results[k][j].toExcel() + ";" + df.format(start + increment * j));
					}
				} else {
					out.println(i + ";" + algorithms[k] + ";" + datastem + ";"
							+ results[k][0].toExcel() + ";" + "0,50");
				}
			}
			System.gc();
		}
	}
}