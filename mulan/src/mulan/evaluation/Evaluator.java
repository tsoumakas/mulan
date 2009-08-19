/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Evaluator.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.evaluation;

import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;


/**
 * Evaluator - responsible for generating evaluation data
 * @author rofr
 * @author Grigorios Tsoumakas
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
	 * @param testSet
	 * @return the evaluation result
	 * @throws IllegalArgumentException if either of input parameters is null.
	 * @throws Exception
	 */
	public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances testSet) throws Exception{
	
		if(learner == null){
			throw new IllegalArgumentException("Learner to be evaluated is null.");
		}
		if(testSet == null){
			throw new IllegalArgumentException("TestDataSet for the evaluation is null.");
		}
	
		// collect output
        Instances testData = testSet.getDataSet();
		int numInstances = testData.numInstances();
		int numLabels = testSet.getNumLabels();

        MultiLabelOutput[] output = new MultiLabelOutput[numInstances];
        boolean trueLabels[][] = new boolean[numInstances][numLabels];

        // Create array of indexes of labels in the test set in prediction order
        int[] indices = testSet.getLabelIndices();
        for (int instanceIndex=0; instanceIndex<numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            output[instanceIndex] = learner.makePrediction(instance);
            trueLabels[instanceIndex] = getTrueLabels(instance, numLabels, indices);
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
        if (testSet.getLabelsMetaData().isHierarchy()) {
            HierarchicalMeasures hm = new HierarchicalMeasures(output, trueLabels, testSet.getLabelsMetaData());
            evaluation.setHierarchicalMeasures(hm);
        }

		return evaluation;
	}
	
	private boolean[] getTrueLabels(Instance instance, int numLabels, int[] labelIndices) {
		
		boolean[] trueLabels = new boolean[numLabels];
		for(int counter = 0; counter < numLabels; counter++)
		{
			int classIdx = labelIndices[counter];
			String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
            trueLabels[counter] = classValue.equals("1");
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
        HierarchicalMeasures[] hm = new HierarchicalMeasures[numFolds];

		Random random = new Random(seed);
		workingSet.randomize(random);
		for (int i=0; i<numFolds; i++)
		{
			Instances train = workingSet.trainCV(numFolds, i, random);
			Instances test  = workingSet.testCV(numFolds, i);
			MultiLabelInstances mlTrain = new MultiLabelInstances(train, mlDataSet.getLabelsMetaData());
			MultiLabelInstances mlTest = new MultiLabelInstances(test, mlDataSet.getLabelsMetaData());
			MultiLabelLearner clone = learner.makeCopy();
			clone.build(mlTrain);

            // Create array of indexes of labels in the test set in prediction order
			Evaluation evaluation = evaluate(clone, mlTest);
            ebm[i] = evaluation.getExampleBasedMeasures();
            lbm[i] = evaluation.getLabelBasedMeasures();
            rbm[i] = evaluation.getRankingBasedMeasures();
            clbm[i] = evaluation.getConfidenceLabelBasedMeasures();
            if (mlDataSet.getLabelsMetaData().isHierarchy())
               hm[i] = evaluation.getHierarchicalMeasures();
        }
        ExampleBasedMeasures exampleBasedMeasures = new ExampleBasedMeasures(ebm);
        LabelBasedMeasures labelBasedMeasures = new LabelBasedMeasures(lbm);
        RankingBasedMeasures rankingBasedMeasures = new RankingBasedMeasures(rbm);
        ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures = new ConfidenceLabelBasedMeasures(clbm);
        HierarchicalMeasures hierarchicalMeasures = null;
        if (mlDataSet.getLabelsMetaData().isHierarchy())
            hierarchicalMeasures = new HierarchicalMeasures(hm);
        Evaluation evaluation = new Evaluation();
        evaluation.setExampleBasedMeasures(exampleBasedMeasures);
        evaluation.setLabelBasedMeasures(labelBasedMeasures);
        evaluation.setRankingBasedMeasures(rankingBasedMeasures);
        evaluation.setConfidenceLabelBasedMeasures(confidenceLabelBasedMeasures);
        if (mlDataSet.getLabelsMetaData().isHierarchy())
            evaluation.setHierarchicalMeasures(hierarchicalMeasures);
        return evaluation;
    }

}

