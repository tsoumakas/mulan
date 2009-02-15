package mulan.evaluation;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mulan.classifier.Bipartition;
import mulan.classifier.BipartitionAndRanking;
import mulan.classifier.MultiLabelClassifier;
import mulan.classifier.MultiLabelClassifierAndRanker;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelRanker;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
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
	
	public Evaluation evaluate(MultiLabelLearner learner, Instances testDataSet) throws Exception{
	
		if(learner == null){
			throw new IllegalArgumentException("Learner to be evaluated is null.");
		}
		if(testDataSet == null){
			throw new IllegalArgumentException("TestDataSet for the evaluation is null.");
		}
	
		Evaluation evaluation;
		if(learner instanceof MultiLabelClassifier){
			evaluation = evaluateClassifier((MultiLabelClassifier)learner, testDataSet);
		} else if(learner instanceof MultiLabelClassifierAndRanker){
			evaluation = evaluateClassifierAndRanker((MultiLabelClassifierAndRanker)learner, testDataSet);
		} else if(learner instanceof MultiLabelRanker){
			evaluation = evaluateRanker((MultiLabelRanker)learner, testDataSet);
		} else{
			throw new IllegalArgumentException("The type of learner is not supported by the evaluator.");
		}
		
		return evaluation;
	}
	
	public Evaluation crossValidate(MultiLabelLearner learner, Instances dataSet, int numFolds) throws Exception{
		if(learner == null){
			throw new IllegalArgumentException("Learner to be evaluated is null.");
		}
		if(dataSet == null){
			throw new IllegalArgumentException("DataSet for the cross validation is null.");
		}
		if(numFolds < 2){
			throw new IllegalArgumentException("NumFolds parameter must be greater or equal to 2.");
		}
		
		Evaluation evaluation;
		if(learner instanceof MultiLabelClassifier){
			evaluation = crossValidateClassifier((MultiLabelClassifier)learner, dataSet, numFolds);
		} else if(learner instanceof MultiLabelClassifierAndRanker){
			evaluation = crossValidateClassifierAndRanker((MultiLabelClassifierAndRanker)learner, dataSet, numFolds);
		} else if(learner instanceof MultiLabelRanker){
			evaluation = crossValidateRanker((MultiLabelRanker)learner, dataSet, numFolds);
		} else{
			throw new IllegalArgumentException("The type of learner is not supported by the evaluator.");
		}
			
		return evaluation;
	}
	
	private Evaluation evaluateClassifier(MultiLabelClassifier learner, Instances testDataSet) throws Exception{
		
		List<ModelEvaluationDataPair<Bipartition>> result = collectClassifierOutput(learner, testDataSet); 
		ExampleBasedMeasures exampleBasedMeasures = new ExampleBasedMeasures(result);
		LabelBasedMeasures labelBasedMeasures = new LabelBasedMeasures(result);
		Evaluation evaluation = new Evaluation();
		evaluation.setExampleBasedMeasures(exampleBasedMeasures);
		evaluation.setLabelBasedMeasures(labelBasedMeasures);
		return evaluation;
	}
		
	private Evaluation evaluateRanker(MultiLabelRanker learner, Instances testDataSet){
		throw new NotImplementedException();
	}
	
	private Evaluation evaluateClassifierAndRanker(MultiLabelClassifierAndRanker learner, Instances testDataSet){
		List<ModelEvaluationDataPair<BipartitionAndRanking>> result = collectClassifierAndRankerOutput(learner, testDataSet);
		ExampleBasedMeasures exampleBasedMeasures = new ExampleBasedMeasures(result);
		LabelBasedMeasures labelBasedMeasures = new LabelBasedMeasures(result);
		Evaluation evaluation = new Evaluation();
		evaluation.setExampleBasedMeasures(exampleBasedMeasures);
		evaluation.setLabelBasedMeasures(labelBasedMeasures);
		return evaluation;

    }
	
	private Evaluation crossValidateClassifier(MultiLabelClassifier learner, Instances dataSet, int numFolds) throws Exception {
		
		Random random = new Random(seed);
        ExampleBasedMeasures[] ebm = new ExampleBasedMeasures[numFolds];
        LabelBasedMeasures[] lbm = new LabelBasedMeasures[numFolds];

		for(int fold = 0; fold < numFolds; fold++){
			Instances train = dataSet.trainCV(numFolds, fold, random);  
			Instances test  = dataSet.testCV(numFolds, fold);
			MultiLabelClassifier clone = (MultiLabelClassifier) learner.makeCopy(learner);
			clone.build(train);
            List<ModelEvaluationDataPair<Bipartition>> results = collectClassifierOutput(clone, test);
            ebm[fold] = new ExampleBasedMeasures(results);
            lbm[fold] = new LabelBasedMeasures(results);
		}
		
		ExampleBasedMeasures exampleBasedMeasures = new ExampleBasedMeasures(ebm);
		LabelBasedMeasures labelBasedMeasures = new LabelBasedMeasures(lbm);
		Evaluation evaluation = new Evaluation();
		evaluation.setExampleBasedMeasures(exampleBasedMeasures);
		evaluation.setLabelBasedMeasures(labelBasedMeasures);		
		return evaluation;
	}
	
	private Evaluation crossValidateRanker(MultiLabelRanker learner, Instances dataSet, int numFolds) throws Exception{
		throw new NotImplementedException();
	}
	
	private Evaluation crossValidateClassifierAndRanker(MultiLabelClassifierAndRanker learner, Instances dataSet, int numFolds) throws Exception{
		throw new NotImplementedException();
	}
	
	private List<ModelEvaluationDataPair<BipartitionAndRanking>> collectClassifierAndRankerOutput(MultiLabelClassifierAndRanker learner, Instances dataSet) throws Exception{
		int numInstances = dataSet.numInstances();
		int numLabels = learner.getNumLabels();
		List<ModelEvaluationDataPair<BipartitionAndRanking>> predictionData = new ArrayList<ModelEvaluationDataPair<BipartitionAndRanking>>(numInstances);
		for (int itemIndex = 0; itemIndex < numInstances; itemIndex++){
			Instance instance = dataSet.instance(itemIndex);
			BipartitionAndRanking prediction = learner.predictAndRank(instance);
			List<Boolean> trueLabels = getTrueLabels(instance, numLabels);
			ModelEvaluationDataPair<BipartitionAndRanking> result =
				new ModelEvaluationDataPair<BipartitionAndRanking>(prediction, trueLabels);
			predictionData.add(result);
		}
		
		return predictionData;
	}
	
	private List<Boolean> getTrueLabels(Instance instance, int numLabels){
		
		List<Boolean> trueLabels = new ArrayList<Boolean>();
		for(int j = 0; j < numLabels; j++)
		{
			int classIdx = instance.numAttributes() - numLabels + j;
			String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
            trueLabels.add(classValue.equals("1"));
		}
		
		return trueLabels;
	}
	
//	
//	public Evaluation crossValidate(MultiLabelLearner learner, Instances dataset)
//	throws Exception
//	{
//		return crossValidate(learner, dataset, DEFAULTFOLDS);
//	}
//	
//	
//	public Evaluation crossValidate(MultiLabelLearner learner, Instances dataset, int numFolds)
//	throws Exception
//	{
//		if (numFolds == -1) numFolds = dataset.numInstances();
//		LabelBasedEvaluation[]  labelBased = new LabelBasedEvaluation[numFolds]; 
//		ExampleBasedEvaluation[]  exampleBased = new ExampleBasedEvaluation[numFolds];
//		LabelRankingBasedEvaluation[] rankingBased=new LabelRankingBasedEvaluation[numFolds];
//		Random random = new Random(seed);
//		
//		Instances workingSet = new Instances(dataset);
//		workingSet.randomize(random);
//		for(int i = 0; i < numFolds; i++)
//		{
//			Instances train = workingSet.trainCV(numFolds, i, random);  
//			Instances test  = workingSet.testCV(numFolds, i);
//			MultiLabelLearner clone = learner.makeCopy(learner);
//			clone.build(train);
//			Evaluation evaluation = evaluate(clone, test);
//			labelBased[i] = evaluation.getLabelBased();
//			exampleBased[i] = evaluation.getExampleBased();
//			rankingBased[i] = evaluation.getRankingBased();
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
//				for (int j = 0; j < predictions[l].length; j++) {
//
//					boolean flag = false;
//					double[] confidences = new double[predictions[l][0].length];
//
//					for (int k = 0; k < predictions[l][0].length; k++) {
//						confidences[k] = predictions[l][j][k].confidenceTrue;
//						if (predictions[l][j][k].confidenceTrue >= threshold) {
//							predictions[l][j][k].predicted = true;
//							flag = true;
//						} else {
//							predictions[l][j][k].predicted = false;
//						}
//					}
//					//assign the class with the greater confidence
//					if (flag == false) {
//						int index = Utils.maxIndex(confidences);
//						predictions[l][j][index].predicted = true;
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
//			for(int j = 0; j < learner.getNumLabels(); j++)
//			{
//				int classIdx = dataset.numAttributes() - learner.getNumLabels() + j;
//				String classValue = dataset.attribute(classIdx).value((int) instance.value(classIdx));
//                                boolean actual = classValue.equals("1");
//				predictions[i][j] = new BinaryPrediction(
//							result.getPrediction(j), 
//							actual, 
//							result.getConfidence(j));
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
//			for(int j = 0; j < predictions.length; j++)
//				for(int k = 0; k < predictions[0].length; k++)
//					predictions[j][k].predicted = predictions[j][k].confidenceTrue >= threshold;
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

