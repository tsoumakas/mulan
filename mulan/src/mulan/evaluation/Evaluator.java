package mulan.evaluation;
import java.util.Random;
import mulan.classifier.AbstractMultiLabelClassifier;
import mulan.classifier.MultiLabelClassifier;
import mulan.classifier.Prediction;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;


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
	

	public Evaluation[] evaluateOverThreshold(BinaryPrediction[][] predictions,
											  Instances dataset,
											  double start,
											  double increment,
											  int steps)
	throws Exception
	{
		Evaluation[] evaluations = new Evaluation[steps];
		
		double threshold = start;
		for(int i = 0; i < steps; i++)
		{
			for(int j = 0; j < predictions.length; j++)
				for(int k = 0; k < predictions[0].length; k++)
					predictions[j][k].predicted = predictions[j][k].confidenceTrue >= threshold;
			threshold += increment;
			evaluations[i] = new Evaluation(
					new LabelBasedEvaluation(predictions),
					new ExampleBasedEvaluation(predictions),
					new LabelRankingBasedEvaluation(predictions));
		}
		
		return evaluations;
		
	}
	
	public Evaluation[] evaluateOverThreshold(MultiLabelClassifier classifier, 
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
}

