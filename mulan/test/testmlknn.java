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

public class testmlknn {

	public testmlknn() {
	}

	public static void main(String[] args) throws Exception {
		
		String path = "E:/Documents and Settings/lefman/Desktop/my workspace/datasets/";
		String datastem = "yeast";
		int numLabels = 14;

		FileReader frTrain = new FileReader(path + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + datastem + "-test.arff");
		Instances testData = new Instances(frTest);
		
		MultiLabelKNN mlknn = new MultiLabelKNN(14,2,1);
		
		mlknn.buildClassifier(trainData);	
		
		mlknn.output();
		
		
		/* // show multilabel statistics 
		//Instances allData = new Instances(trainData);
        //for (int i=0; i<testData.numInstances(); i++)
        //    allData.add(testData.instance(i));

        Statistics stats = new Statistics();
        stats.calculateStats(trainData, numLabels);
        System.out.println(stats.toString());
		 
		 /*  // Binary Relevance Classifier Evaluator eval; Evaluation results;
		 * System.out.println("BR"); BinaryRelevanceClassifier br = new
		 * BinaryRelevanceClassifier();
		 * br.setBaseClassifier(Classifier.makeCopy(baseClassifier));
		 * br.setNumLabels(numLabels); br.buildClassifier(trainData); eval = new
		 * Evaluator(); results = eval.evaluate(br, testData);
		 * results.getLabelBased().setAveragingMethod(LabelBasedEvaluation.MICRO);
		 * System.out.println("HammingLoss : " +
		 * results.getExampleBased().hammingLoss());
		 * System.out.println("Precision : " +
		 * results.getLabelBased().precision()); System.out.println("Recall : " +
		 * results.getLabelBased().recall()); System.out.println("F1 : " +
		 * results.getLabelBased().fmeasure());
		 * 
		 */

	}
}
