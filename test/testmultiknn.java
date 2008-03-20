import java.io.FileReader;

import mulan.classifier.MultiKnn;
import mulan.evaluation.Evaluator;
import mulan.evaluation.IntegratedCrossvalidation;
import weka.core.Instances;

/**
 * Test class for the multi-label knn classifier
 * 
 * @author Eleftherios Spyromitros - Xioufis
 */

public class testmultiknn {

	public testmultiknn() {
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
		
	   //Statistics stats = new Statistics();
       // stats.calculateStats(allData, numLabels);
       // System.out.println(stats.toString());

		for (int i = 10; i <= 10; i++) {
			System.out.println("Calculating mlknn output for " + i + " neighbours");

			MultiKnn mlknn = new MultiKnn(numLabels, trainData ,i);
			//mlknn.setDontNormalize(true);

			//long start = System.currentTimeMillis();
			//mlknn.buildClassifier(trainData);
			//long end = System.currentTimeMillis();

			//System.out.print("Buildclassifier Time: " + (end - start) + "\n");

			Evaluator eval;
			IntegratedCrossvalidation results;
			eval = new Evaluator();
			results = eval.crossValidateAll(mlknn, allData,10);

			//start = System.currentTimeMillis();
			//results = eval.evaluateAll(mlknn, testData);
			//end = System.currentTimeMillis();

			//System.out.print("Evaluation Time: " + (end - start) + "\n");
			System.out.println( (double) mlknn.sumedlabels / trainData.numInstances());
			System.out.println(results.toString());
			System.gc();

			//mlknn.output(); //outputs prior and conditional probabilities
		}
	}
}