package mulan.evaluation;

import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.core.data.MultiLabelInstances;
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
	
	/**
	 * Evaluates a {@link MultiLabelLearner} on given test data set.
	 * 
	 * @param learner the learner to be evaluated via cross-validation
	 * @param dataset the data set for cross-validation
	 * @param learner
	 * @param dataSet
	 * @return the evaluation result
	 * @throws IllegalArgumentException if either of input parameters is null.
	 * @throws Exception
	 */
	public Evaluation evaluate(MultiLabelLearner learner, Instances dataSet) throws Exception{
	
		if(learner == null){
			throw new IllegalArgumentException("Learner to be evaluated is null.");
		}
		if(dataSet == null){
			throw new IllegalArgumentException("TestDataSet for the evaluation is null.");
		}
	
		// collect output
		int numInstances = dataSet.numInstances();
		int numLabels = learner.getNumLabels();

        MultiLabelOutput[] output = new MultiLabelOutput[numInstances];
        boolean trueLabels[][] = new boolean[numInstances][numLabels];
        for (int instanceIndex=0; instanceIndex<numInstances; instanceIndex++) {
            Instance instance = dataSet.instance(instanceIndex);
            output[instanceIndex] = learner.makePrediction(instance);
            trueLabels[instanceIndex] = getTrueLabels(instance, numLabels);
        }
		Evaluation evaluation = new Evaluation();
        if (output[0].hasBipartition()) {
            ExampleBasedMeasures ebm = new ExampleBasedMeasures(output, trueLabels);
    		evaluation.setExampleBasedMeasures(ebm);
            LabelBasedMeasures lbm = new LabelBasedMeasures(output, trueLabels);
        	evaluation.setLabelBasedMeasures(lbm);
        }
        if (output[0].hasRanking()) {
            RankingBasedMeasures rbm = new RankingBasedMeasures(output, trueLabels);
            evaluation.setRankingBasedMeasures(rbm);
        }
        if (output[0].hasConfidences()) {
            ConfidenceLabelBasedMeasures clbm = new ConfidenceLabelBasedMeasures(output, trueLabels);
            evaluation.setConfidenceLabelBasedMeasures(clbm);
        }
		return evaluation;
	}
	
	private boolean[] getTrueLabels(Instance instance, int numLabels){
		
		boolean[] trueLabels = new boolean[numLabels];
		for(int labelIndex = 0; labelIndex < numLabels; labelIndex++)
		{
			int classIdx = instance.numAttributes() - numLabels + labelIndex;
			String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
            trueLabels[labelIndex] = classValue.equals("1");
		}
		
		return trueLabels;
	}
	
	/**
	 * Evaluates a {@link MultiLabelLearner} via cross-validation on given data set.
	 * The default number of folds {@link Evaluator#DEFAULTFOLDS} will be used. 
	 * 
	 * @param learner the learner to be evaluated via cross-validation
	 * @param mlDataSet the multi-label data set for cross-validation
	 * @return the evaluation result
	 * @throws IllegalArgumentException if either of input parameters is null.
	 * @throws Exception
	 */
	public Evaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances mlDataSet)
	throws Exception
	{
		return crossValidate(learner, mlDataSet, DEFAULTFOLDS);
	}
	
	/**
	 * Evaluates a {@link MultiLabelLearner} via cross-validation on given data set with
	 * defined number of folds. 
	 * The specified number of folds has to be at least two. 
	 * If negative value is specified, the used number of folds is equal to number 
	 * of instances in the data set. 
	 * 
	 * @param learner the learner to be evaluated via cross-validation
	 * @param mlDataSet the multi-label data set for cross-validation
	 * @param numFolds the number of folds to be used
	 * @return the evaluation result
	 * @throws IllegalArgumentException if either of learner or data set parameters is null
	 * @throws IllegalArgumentException if number of folds is invalid 
	 * @throws Exception
	 */
	public Evaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances mlDataSet, int numFolds)
	throws Exception
	{
		if(learner == null){
			throw new IllegalArgumentException("Learner to be evaluated is null.");
		}
		if(mlDataSet == null){
			throw new IllegalArgumentException("MutliLabelDataset for the evaluation is null.");
		}
		if(numFolds == 0 || numFolds == 1){
			throw new IllegalArgumentException("Number of folds must be at least two or higher.");
		}
		
		Instances workingSet = new Instances(mlDataSet.getDataSet());

		if (numFolds < 0) 
			numFolds = workingSet.numInstances();
        
		ExampleBasedMeasures[] ebm = new ExampleBasedMeasures[numFolds];
        LabelBasedMeasures[] lbm = new LabelBasedMeasures[numFolds];
        RankingBasedMeasures[] rbm = new RankingBasedMeasures[numFolds];
        ConfidenceLabelBasedMeasures[] clbm = new ConfidenceLabelBasedMeasures[numFolds];

		Random random = new Random(seed);
		workingSet.randomize(random);
		for (int i=0; i<numFolds; i++)
		{
			Instances train = workingSet.trainCV(numFolds, i, random);
			Instances test  = workingSet.testCV(numFolds, i);
			MultiLabelInstances mlTrain = new MultiLabelInstances(train, mlDataSet.getLabelsMetaData());
			MultiLabelLearner clone = learner.makeCopy();
			clone.build(mlTrain);
			Evaluation evaluation = evaluate(clone, test);
            ebm[i] = evaluation.getExampleBasedMeasures();
            lbm[i] = evaluation.getLabelBasedMeasures();
            rbm[i] = evaluation.getRankingBasedMeasures();
            clbm[i] = evaluation.getConfidenceLabelBasedMeasures();
        }
        ExampleBasedMeasures exampleBasedMeasures = new ExampleBasedMeasures(ebm);
        LabelBasedMeasures labelBasedMeasures = new LabelBasedMeasures(lbm);
        RankingBasedMeasures rankingBasedMeasures = new RankingBasedMeasures(rbm);
        ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures = new ConfidenceLabelBasedMeasures(clbm);
        Evaluation evaluation = new Evaluation();
        evaluation.setExampleBasedMeasures(exampleBasedMeasures);
        evaluation.setLabelBasedMeasures(labelBasedMeasures);
        evaluation.setRankingBasedMeasures(rankingBasedMeasures);
        evaluation.setConfidenceLabelBasedMeasures(confidenceLabelBasedMeasures);

        return evaluation;
    }
//		}
//		
//		return new CrossValidation(
//				new LabelBasedCrossValidation(labelBased),
//				new ExampleBasedCrossValidation(exampleBased),
//				new LabelRankingBasedCrossValidation(rankingBased),
//				numFolds); 
//
//	}
//	
//	public IntegratedCrossvalidation crossValidateAll(MultiLabelLearner learner, Instances dataset, int numFolds)
//	throws Exception
//	{
//		if (numFolds == -1) numFolds = dataset.numInstances();
//		IntegratedEvaluation[] integrated=new IntegratedEvaluation[numFolds];
//		Random random = new Random(seed);
//		
//		Instances workingSet = new Instances(dataset);
//		workingSet.randomize(random);
//		for(int i = 0; i < numFolds; i++)
//		{
//			Instances train = workingSet.trainCV(numFolds, i, random);  
//			Instances test  = workingSet.testCV(numFolds, i);
//			MultiLabelLearner clone = learner.makeCopy(learner);
//			//long start = System.currentTimeMillis();
//			clone.build(train);
//			//long end = System.currentTimeMillis();
//			//System.out.print(i + "Buildclassifier Time: " + (end - start) + "\n");
//			//start = System.currentTimeMillis();
//			integrated[i] = evaluateAll(clone, test);
//			//end = System.currentTimeMillis();
//			//System.out.print(i + "Evaluation Time: " + (end - start) + "\n");
//		}
//		return new IntegratedCrossvalidation(integrated); 
//	}
//	
//	public IntegratedCrossvalidation[] crossvalidateOverThreshold(
//			BinaryPrediction[][][] predictions, Instances dataset, double start, double increment,
//			int steps, int numFolds) throws Exception {
//		IntegratedCrossvalidation[] crossvalidations = new IntegratedCrossvalidation[steps];
//
//		double threshold = start;
//		for (int i = 0; i < steps; i++) { //for every step
//			crossvalidations[i] = new IntegratedCrossvalidation(numFolds);
//			for (int l = 0; l < numFolds; l++) { //for every fold that has been evaluated
//				//calculate the predictions based on threshold
//				for (int labelIndex = 0; labelIndex < predictions[l].length; labelIndex++) {
//
//					boolean flag = false;
//					double[] confidences = new double[predictions[l][0].length];
//
//					for (int k = 0; k < predictions[l][0].length; k++) {
//						confidences[k] = predictions[l][labelIndex][k].confidenceTrue;
//						if (predictions[l][labelIndex][k].confidenceTrue >= threshold) {
//							predictions[l][labelIndex][k].predicted = true;
//							flag = true;
//						} else {
//							predictions[l][labelIndex][k].predicted = false;
//						}
//					}
//					//assign the class with the greater confidence
//					if (flag == false) {
//						int index = Utils.maxIndex(confidences);
//						predictions[l][labelIndex][index].predicted = true;
//					}
//				}
//				//assign the prediction to the l th fold of this step's crossvalidation
//				crossvalidations[i].folds[l] = new IntegratedEvaluation(predictions[l]);
//			}
//			crossvalidations[i].computeMeasures();
//			threshold += increment; //increase threshold for the next step
//		}
//
//		return crossvalidations;
//
//	}
//
//	public IntegratedCrossvalidation[] crossvalidateOverThreshold(MultiLabelLearner learner,
//			Instances dataset, double start, double increment, int steps, int numFolds)
//			throws Exception {
//		//create a crossvalidation of the classifier in order to get predictions
//		IntegratedCrossvalidation cv = crossValidateAll(learner, dataset, numFolds);
//		BinaryPrediction[][][] predictions2 = new BinaryPrediction[numFolds][][];
//		for (int i = 0; i < numFolds; i++) {
//			predictions2[i] = cv.folds[i].predictions;
//		}
//
//return crossvalidateOverThreshold(predictions2, dataset, start, increment, steps,numFolds);
//}
//	
//	protected BinaryPrediction[][] getPredictions(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = 
//			new BinaryPrediction[dataset.numInstances()][learner.getNumLabels()];
//		
//		for(int i = 0; i < dataset.numInstances(); i++)
//		{
//			Instance instance = dataset.instance(i);
//			Prediction result = learner.predict(instance);
//			//System.out.println(java.util.Arrays.toString(result.getConfidences()));
//			for(int labelIndex = 0; labelIndex < learner.getNumLabels(); labelIndex++)
//			{
//				int classIdx = dataset.numAttributes() - learner.getNumLabels() + labelIndex;
//				String classValue = dataset.attribute(classIdx).value((int) instance.value(classIdx));
//                                boolean actual = classValue.equals("1");
//				predictions[i][labelIndex] = new BinaryPrediction(
//							result.getPrediction(labelIndex),
//							actual, 
//							result.getConfidence(labelIndex));
//			}
//		}
//		return predictions;
//	}
//	
//
//	public IntegratedEvaluation[] evaluateOverThreshold(BinaryPrediction[][] predictions,
//											  Instances dataset,
//											  double start,
//											  double increment,
//											  int steps)
//	throws Exception
//	{
//		IntegratedEvaluation[] evaluations = new IntegratedEvaluation[steps];
//		
//		double threshold = start;
//		for(int i = 0; i < steps; i++)
//		{
//			for(int labelIndex = 0; labelIndex < predictions.length; labelIndex++)
//				for(int k = 0; k < predictions[0].length; k++)
//					predictions[labelIndex][k].predicted = predictions[labelIndex][k].confidenceTrue >= threshold;
//			threshold += increment;
//			evaluations[i] = new IntegratedEvaluation(predictions);
//		}
//		
//		return evaluations;
//		
//	}
//	
//	public IntegratedEvaluation[] evaluateOverThreshold(MultiLabelLearner learner, 
//											  Instances dataset, 
//											  double start, 
//											  double increment, 
//											  int steps)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = getPredictions(learner, dataset);
//		return evaluateOverThreshold(predictions, dataset, start, increment, steps);
//	}
//	
//	public Evaluation evaluate(BinaryPrediction[][] predictions)
//	throws Exception
//	{
//		return new Evaluation(
//				new LabelBasedEvaluation(predictions),
//				new ExampleBasedEvaluation(predictions),
//				new LabelRankingBasedEvaluation(predictions));
//	}
//	
//	
//	public Evaluation evaluate(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = getPredictions(learner, dataset);
//		return evaluate(predictions);
//	}
//	
//	public IntegratedEvaluation evaluateAll(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = getPredictions(learner, dataset);
//		return new IntegratedEvaluation(predictions);
//	}
//	
//	public ExampleBasedEvaluation evaluateExample(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = getPredictions(learner, dataset);
//		return new ExampleBasedEvaluation(predictions);
//	}
//	
//	public LabelRankingBasedEvaluation evaluateRanking(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = getPredictions(learner, dataset);
//		return new LabelRankingBasedEvaluation(predictions);
//	}
//	
//	public LabelBasedEvaluation evaluateLabel(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		BinaryPrediction[][] predictions = getPredictions(learner, dataset);
//		return new LabelBasedEvaluation(predictions);
//	}

}

