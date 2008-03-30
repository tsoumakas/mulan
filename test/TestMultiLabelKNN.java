import java.io.FileReader;

import mulan.classifier.*;
import mulan.evaluation.*;
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
		String datastem = "yeast";
		//String datastem = "scene";
		//String datastem = "genbase10";

		FileReader frTrain = new FileReader(path + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + datastem + "-test.arff");
		Instances testData = new Instances(frTest);

		Instances allData = new Instances(trainData);
		for (int i = 0; i < testData.numInstances(); i++)
			allData.add(testData.instance(i));

		int numLabels = 14;
		//int numLabels2 = 6;
		
	   //Statistics stats = new Statistics();
       // stats.calculateStats(allData, numLabels);
       // System.out.println(stats.toString());

		for (int i = 1; i <= 30; i++) {
			//System.out.println("Calculating mlknn output for " + i + " neighbours");

			MultiLabelKNN BRknn = new BRknn(numLabels,i);
			BRknn.setDontNormalize(false);
			
			//MultiLabelKNN Mlknn = new Mlknn(numLabels2,i,1);
			//BRknn.setDontNormalize(false);

			//long start = System.currentTimeMillis();
			//mlknn.buildClassifier(trainData);
			//long end = System.currentTimeMillis();

		    //System.out.print("Buildclassifier Time: " + (end - start) + "\n");

			Evaluator eval;
			//IntegratedEvaluation results;
			IntegratedCrossvalidation results;
			eval = new Evaluator();
			results = eval.crossValidateAll(BRknn, allData,10);

			//start = System.currentTimeMillis();
			//results = eval.evaluateAll(mlknn, testData);
			//end = System.currentTimeMillis();

			//System.out.print("Evaluation Time: " + (end - start) + "\n");
			//System.out.println("Average labels predicted: " + (double) mlknn.sumedlabels / trainData.numInstances());
			//System.out.println(results.toString());
			
			System.out.println(i + ";" + "BRknn" + ";" + datastem + ";" + results.toExcel());
			System.gc();

			//mlknn.output(); //outputs prior and conditional probabilities
		}
	}
}