package mulan.evaluation;
import java.util.Random;

import mulan.classifier.AbstractMultiLabelClassifier;
import mulan.classifier.MultiLabelClassifier;
import mulan.classifier.Prediction;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;


/**
 * Evaluator - responsible for generating evaluation data
 * @author rofr
 *
 */
public class Evaluator
{
	public static final int DEFAULTFOLDS = 10;
	
	/**
	 * Seed to random number generator. Needed to reproduce crossvalidation randomization.
	 * Default is 1 
	 */
	protected int seed; 
	
	public Evaluator()
	{
		this(1);
	}
	
	public Evaluator(int seed)
	{
		this.seed = seed;
	}
	
	public Evaluation crossValidate(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		return crossValidate(classifier, dataset, DEFAULTFOLDS);
	}
	
	
	public Evaluation crossValidate(MultiLabelClassifier classifier, Instances dataset, int numFolds)
	throws Exception
	{
		if (numFolds == -1) numFolds = dataset.numInstances();
		LabelBasedEvaluation[]  labelBased = new LabelBasedEvaluation[numFolds]; 
		ExampleBasedEvaluation[]  exampleBased = new ExampleBasedEvaluation[numFolds];
		LabelRankingBasedEvaluation[] rankingBased=new LabelRankingBasedEvaluation[numFolds];
		Random random = new Random(seed);
		
		Instances workingSet = new Instances(dataset);
		workingSet.randomize(random);
		for(int i = 0; i < numFolds; i++)
		{
			Instances train = workingSet.trainCV(numFolds, i, random);  
			Instances test  = workingSet.testCV(numFolds, i);
			AbstractMultiLabelClassifier clone = 
				(AbstractMultiLabelClassifier) Classifier.makeCopy((Classifier) classifier);
			clone.buildClassifier(train);
			Evaluation evaluation = evaluate(clone, test);
			labelBased[i] = evaluation.getLabelBased();
			exampleBased[i] = evaluation.getExampleBased();
			rankingBased[i] = evaluation.getRankingBased();
		}
		
		return new CrossValidation(
				new LabelBasedCrossValidation(labelBased),
				new ExampleBasedCrossValidation(exampleBased),
				new LabelRankingBasedCrossValidation(rankingBased),
				numFolds); 

	}
	
	public IntegratedCrossvalidation crossValidateAll(MultiLabelClassifier classifier, Instances dataset, int numFolds)
	throws Exception
	{
		if (numFolds == -1) numFolds = dataset.numInstances();
		IntegratedEvaluation[] integrated=new IntegratedEvaluation[numFolds];
		Random random = new Random(seed);
		
		Instances workingSet = new Instances(dataset);
		workingSet.randomize(random);
		for(int i = 0; i < numFolds; i++)
		{
			Instances train = workingSet.trainCV(numFolds, i, random);  
			Instances test  = workingSet.testCV(numFolds, i);
			AbstractMultiLabelClassifier clone = 
				(AbstractMultiLabelClassifier) Classifier.makeCopy((Classifier) classifier);
			//long start = System.currentTimeMillis();
			clone.buildClassifier(train);
			//long end = System.currentTimeMillis();
			//System.out.print(i + "Buildclassifier Time: " + (end - start) + "\n");
			//start = System.currentTimeMillis();
			integrated[i] = evaluateAll(clone, test);
			//end = System.currentTimeMillis();
			//System.out.print(i + "Evaluation Time: " + (end - start) + "\n");
		}
		return new IntegratedCrossvalidation(integrated); 
	}
	
	public IntegratedCrossvalidation[] crossvalidateOverThreshold(
			BinaryPrediction[][][] predictions, Instances dataset, double start, double increment,
			int steps, int numFolds, boolean BR2) throws Exception {
		IntegratedCrossvalidation[] crossvalidations = new IntegratedCrossvalidation[steps];

		Random Rand = new Random(seed);
		double threshold = start;
		for (int i = 0; i < steps; i++) { //for every step
			crossvalidations[i] = new IntegratedCrossvalidation(numFolds);
			for (int l = 0; l < numFolds; l++) { //for every fold that has been evaluated
				//calculate the predictions based on threshold
				for (int j = 0; j < predictions[l].length; j++) {

					boolean flag = false;
					double[] confidences = new double[predictions[l][0].length];

					for (int k = 0; k < predictions[l][0].length; k++) {
						confidences[k] = predictions[l][j][k].confidenceTrue;
						if (predictions[l][j][k].confidenceTrue >= threshold) {
							predictions[l][j][k].predicted = true;
							flag = true;
						} else {
							predictions[l][j][k].predicted = false;
						}
					}
					//assign the class with the greater confidence
					if (flag == false && BR2 == true) {
						int index = RandomIndexOfMax(confidences, Rand);
						predictions[l][j][index].predicted = true;
					}
				}
				//assign the prediction to the l th fold of this step's crossvalidation
				crossvalidations[i].folds[l] = new IntegratedEvaluation(predictions[l]);
			}
			crossvalidations[i].computeMeasures();
			threshold += increment; //increase threshold for the next step
		}

		return crossvalidations;

	}

	public IntegratedCrossvalidation[] crossvalidateOverThreshold(MultiLabelClassifier classifier,
			Instances dataset, double start, double increment, int steps, int numFolds)
			throws Exception {
		//create a crossvalidation of the classifier in order to get predictions
		IntegratedCrossvalidation cv = crossValidateAll(classifier, dataset, numFolds);
		BinaryPrediction[][][] predictions2 = new BinaryPrediction[numFolds][][];
		for (int i = 0; i < numFolds; i++) {
			predictions2[i] = cv.folds[i].predictions;
		}

		return crossvalidateOverThreshold(predictions2, dataset, start, increment, steps, numFolds, false);
	}
	
	public IntegratedCrossvalidation[] crossvalidateOverThresholdBR2(MultiLabelClassifier classifier,
			Instances dataset, double start, double increment, int steps, int numFolds)
			throws Exception {
		//create a crossvalidation of the classifier in order to get predictions
		IntegratedCrossvalidation cv = crossValidateAll(classifier, dataset, numFolds);
		BinaryPrediction[][][] predictions2 = new BinaryPrediction[numFolds][][];
		for (int i = 0; i < numFolds; i++) {
			predictions2[i] = cv.folds[i].predictions;
		}

		return crossvalidateOverThreshold(predictions2, dataset, start, increment, steps, numFolds, true);
	}
	
	protected BinaryPrediction[][] getPredictions(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		BinaryPrediction[][] predictions = 
			new BinaryPrediction[dataset.numInstances()][classifier.getNumLabels()];
		
		for(int i = 0; i < dataset.numInstances(); i++)
		{
			Instance instance = dataset.instance(i);
			Prediction result = classifier.predict(instance);
			//System.out.println(java.util.Arrays.toString(result.getConfidences()));
			for(int j = 0; j < classifier.getNumLabels(); j++)
			{
				int classIdx = dataset.numAttributes() - classifier.getNumLabels() + j;
				String classValue = dataset.attribute(classIdx).value((int) instance.value(classIdx));
                                boolean actual = classValue.equals("1");
				predictions[i][j] = new BinaryPrediction(
							result.getPrediction(j), 
							actual, 
							result.getConfidence(j));
			}
		}
		return predictions;
	}
	

	public IntegratedEvaluation[] evaluateOverThreshold(BinaryPrediction[][] predictions,
											  Instances dataset,
											  double start,
											  double increment,
											  int steps)
	throws Exception
	{
		IntegratedEvaluation[] evaluations = new IntegratedEvaluation[steps];
		
		double threshold = start;
		for(int i = 0; i < steps; i++)
		{
			for(int j = 0; j < predictions.length; j++)
				for(int k = 0; k < predictions[0].length; k++)
					predictions[j][k].predicted = predictions[j][k].confidenceTrue >= threshold;
			threshold += increment;
			evaluations[i] = new IntegratedEvaluation(predictions);
		}
		
		return evaluations;
		
	}
	
	public IntegratedEvaluation[] evaluateOverThreshold(MultiLabelClassifier classifier, 
											  Instances dataset, 
											  double start, 
											  double increment, 
											  int steps)
	throws Exception
	{
		BinaryPrediction[][] predictions = getPredictions(classifier, dataset);
		return evaluateOverThreshold(predictions, dataset, start, increment, steps);
	}
	
	public Evaluation evaluate(BinaryPrediction[][] predictions)
	throws Exception
	{
		return new Evaluation(
				new LabelBasedEvaluation(predictions),
				new ExampleBasedEvaluation(predictions),
				new LabelRankingBasedEvaluation(predictions));
	}
	
	
	public Evaluation evaluate(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		BinaryPrediction[][] predictions = getPredictions(classifier, dataset);
		return evaluate(predictions);
	}
	
	public IntegratedEvaluation evaluateAll(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		BinaryPrediction[][] predictions = getPredictions(classifier, dataset);
		return new IntegratedEvaluation(predictions);
	}
	
	public ExampleBasedEvaluation evaluateExample(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		BinaryPrediction[][] predictions = getPredictions(classifier, dataset);
		return new ExampleBasedEvaluation(predictions);
	}
	
	public LabelRankingBasedEvaluation evaluateRanking(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		BinaryPrediction[][] predictions = getPredictions(classifier, dataset);
		return new LabelRankingBasedEvaluation(predictions);
	}
	
	public LabelBasedEvaluation evaluateLabel(MultiLabelClassifier classifier, Instances dataset)
	throws Exception
	{
		BinaryPrediction[][] predictions = getPredictions(classifier, dataset);
		return new LabelBasedEvaluation(predictions);
	}
	
	public int RandomIndexOfMax(double Array[], Random Rand) {
		double Max = Array[0];

		for (int i = 1; i < Array.length; i++)
			if (Array[i] > Max)
				Max = Array[i];

		int Count = 0;
		for (int i = 0; i < Array.length; i++)
			if (Array[i] == Max)
				Count++;

		int Choose = Rand.nextInt(Count) + 1;

		Count = 0;
		for (int i = 0; i < Array.length; i++) {
			if (Array[i] == Max)
				Count++;
			if (Count == Choose)
				return i;
		}

		return -1;
	}
	
}

