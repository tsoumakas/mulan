import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;

import mulan.Statistics;
import mulan.classifier.*;
import mulan.evaluation.Evaluator;
import mulan.evaluation.IntegratedCrossvalidation;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

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

		//String datastem = "yeast2";
		//String datastem = "scene";
		//String datastem = "genbase10";
		String datastem = "music0";

		//int numLabels = 14;
		//int numLabels = 6;
		//int numLabels = 27;
		int numLabels = 6;
		
		File file = new File(".//" + datastem + " results.txt");
		PrintWriter out = new PrintWriter(new FileWriter(file));

		FileReader frTrain = new FileReader(path + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + datastem + "-test.arff");
		Instances testData = new Instances(frTest);

		Instances allData = new Instances(trainData);
		for (int i = 0; i < testData.numInstances(); i++)
			allData.add(testData.instance(i));
		
		/*Statistics stats = new Statistics();
		stats.calculateStats(allData, numLabels);
		System.out.println(stats.toString());*/
		
		
		//names of the tested algorithms
		String[][] algorithms = {
				{"BRknn-r","normalized","WEIGHT_NONE"},
				{"BRknn2-r","normalized","WEIGHT_NONE"},
				{"BRknn3-r","normalized","WEIGHT_NONE"},
				{"LPknn-r","normalized","WEIGHT_NONE"},
				{"MLknn","normalized","WEIGHT_NONE"},
				{"MLknn","not-normalized","WEIGHT_NONE"},
				{"LPknn2","normalized","WEIGHT_NONE"},
				{"RAKELknn","normalized","WEIGHT_NONE"}
				};

		//test for values of k between 1 and 30
		for (int i = 10; i <= 10; i += 1) {

			BRknn br = new BRknn(numLabels, i);
			br.setDistanceWeighting(1);
			br.setDontNormalize(false);
			
			BRknn br2 = new BRknn(numLabels, i, 2);
			br2.setDistanceWeighting(1);
			br2.setDontNormalize(false);
			
			BRknn br3 = new BRknn(numLabels, i, 3);
			br2.setDistanceWeighting(1);
			br2.setDontNormalize(false);

			Mlknn ml = new Mlknn(numLabels, i, 1);
			ml.setDontNormalize(false);
						
			Mlknn ml2 = new Mlknn(numLabels, i, 1);
			ml2.setDontNormalize(true);

			IBk baseClassifier = new IBk(i);

			LabelPowersetClassifier lp = new LabelPowersetClassifier(Classifier
					.makeCopy(baseClassifier), numLabels);
			
			LPknn lp2 = new LPknn(numLabels, i);
			
			RAKELknn rakel = new RAKELknn(numLabels,i,10,4);

			//Binary Relevance Classifier 
			//System.out.println("BR");
			//BinaryRelevanceClassifier br1 = new BinaryRelevanceClassifier(Classifier
			//		.makeCopy(baseClassifier), numLabels);
			//br1.buildClassifier(trainData);
			
			//=====================EVALUATION==========================
			//br.buildClassifier(trainData);
			//br2.buildClassifier(trainData);
			//ml.buildClassifier(trainData);
			//ml2.buildClassifier(trainData);
			//lp.buildClassifier(trainData);
			//lp2.buildClassifier(trainData);
			
			/*Evaluator eval;
			IntegratedEvaluation[] results = new IntegratedEvaluation[6];
			eval = new Evaluator();
			//BR EVAULATION 
			long start = System.currentTimeMillis();
			results[0] = eval.evaluateAll(br, testData);
			long end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//BR2 EVAULATION 
			start = System.currentTimeMillis();
			results[1] = eval.evaluateAll(br2, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//ML EVAULATION 
			start = System.currentTimeMillis();
			results[2] = eval.evaluateAll(ml, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//ML2 EVAULATION 
			start = System.currentTimeMillis();
			results[3] = eval.evaluateAll(ml2, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//LP EVAULATION 
			start = System.currentTimeMillis();
			results[4] = eval.evaluateAll(lp, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			//LP2 EVAULATION 
			start = System.currentTimeMillis();
			results[5] = eval.evaluateAll(lp2, testData);
			end = System.currentTimeMillis();
			System.out.print("Evaluation Time: " + (end - start) + "\n");
			
			for (int k = 0; k <= 0; k++) {
						System.out.println(
								i 
								+ ";" + algorithms[k][0] 
						        + ";" + datastem 
						        + ";" + results[k].toExcel()
								+ ";" + "0,5"
								+ ";" + algorithms[k][1]
							    + ";" + algorithms[k][2]);
					}
			
			*/
			// =====================CROSS-VALIDATION==========================
			int numfolds = 10;
			int numsteps = 13;
			double start = 0.2;
			double increment = 0.05;

			DecimalFormat df = new DecimalFormat("0.00");

			Evaluator eval;
			IntegratedCrossvalidation[][] results = new IntegratedCrossvalidation[6][numsteps];
			eval = new Evaluator();

			results[0] = eval.crossvalidateOverThreshold(br, allData, start,
					increment, numsteps, numfolds);
			results[1] = eval.crossvalidateOverThresholdBR2(br2, allData, start,
					increment, numsteps, numfolds);
			//results[2] = eval.crossvalidateOverThreshold(ml, allData, start,
			//		increment, numsteps, numfolds);
			//results[3] = eval.crossvalidateOverThreshold(ml2, allData, start,
			//		increment, numsteps, numfolds);
			// LP can't be evaluated over threshold
			//results[4][0] = eval.crossValidateAll(lp, allData, numfolds);
			//results[5][0] = eval.crossValidateAll(lp2, allData, numfolds);
			
			//results[0][0] = eval.crossValidateAll(br, allData, numfolds);
			//results[1][0] = eval.crossValidateAll(br2, allData, numfolds);
			results[2][0] = eval.crossValidateAll(br3, allData, numfolds);
			results[3][0] = eval.crossValidateAll(lp, allData, numfolds);
			
			for (int k = 0; k <= 3; k++) {
				if (k<2) {
					for (int j = 0; j < numsteps; j++) {
						out.println(
								i 
								+ ";" + algorithms[k][0] 
						        + ";" + datastem 
						        + ";" + results[k][j].toExcel()
								+ ";" + df.format(start + increment * j)
								+ ";" + algorithms[k][1]
							    + ";" + algorithms[k][2]);
					}
				} else {
						out.println(
							i 
							+ ";" + algorithms[k][0] 
							+ ";" + datastem
							+ ";" + results[k][0].toExcel() 
							+ ";" + "0,50"
							+ ";" + algorithms[k][1]
							+ ";" + algorithms[k][2]);
				}
				out.flush();
			}	
		}
		out.close();
		System.gc();
	}
}