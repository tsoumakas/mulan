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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import java.util.Set;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.*;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Evaluator - responsible for generating evaluation data
 * @author rofr
 * @author Grigorios Tsoumakas
 */
public class Evaluator {

    public static final int DEFAULTFOLDS = 10;
    /**
     * Seed to random number generator. Needed to reproduce crossvalidation randomization.
     * Default is 1
     */
    protected int seed;

    public Evaluator() {
        this(1);
    }

    public Evaluator(int seed) {
        this.seed = seed;
    }

    /**
     * Evaluates a {@link MultiLabelLearner} on given test data set using specified evaluation measures
     *
     * @param learner the learner to be evaluated via cross-validation
     * @param testSet the data set for cross-validation
     * @param measures the evaluation measures to compute
     * @throws IllegalArgumentException if an input parameter is null
     * @throws Exception
     */
    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances testSet, List<Measure> measures) throws IllegalArgumentException, Exception {

        if (measures == null) {
            throw new IllegalArgumentException("List of evaluation measures to compute is null.");
        }

        // reset measures
        for (Measure m : measures) {
            m.reset();
        }

        // collect output
        Instances testData = testSet.getDataSet();
        int numInstances = testData.numInstances();
        int numLabels = testSet.getNumLabels();

        MultiLabelOutput output = null;
        boolean trueLabels[] = new boolean[numLabels];

        Set<Measure> failed = new HashSet<Measure>();

        // Create array of indexes of labels in the test set in prediction order
        int[] indices = testSet.getLabelIndices();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            output = learner.makePrediction(instance);
            trueLabels = getTrueLabels(instance, numLabels, indices);
            Iterator<Measure> it = measures.iterator();
            while (it.hasNext()) {
                Measure m = it.next();
                if (!failed.contains(m)) {
                    try {
                        m.update(output, trueLabels);
                    } catch (Exception ex) {
                        failed.add(m);
                    }
                }
            }
        }

        return new Evaluation(measures);
    }

    /**
     * Evaluates a {@link MultiLabelLearner} on given test data set.
     *
     * @param learner the learner to be evaluated
     * @param dataset the data set for evaluation
     * @return the evaluation result
     * @throws IllegalArgumentException if either of input parameters is null.
     * @throws Exception
     */
    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances testSet) throws IllegalArgumentException, Exception {
        if (learner == null) {
            throw new IllegalArgumentException("Learner to be evaluated is null.");
        }
        if (testSet == null) {
            throw new IllegalArgumentException("TestDataSet for the evaluation is null.");
        }

        List<Measure> measures = new ArrayList<Measure>();

        MultiLabelOutput prediction = learner.makePrediction(testSet.getDataSet().instance(0));

        // add bipartition-based measures if applicable
        if (prediction.hasBipartition()) {
            // add example-based measures
            measures.add(new HammingLoss());
            measures.add(new SubsetAccuracy());
            measures.add(new ExampleBasedPrecision());
            measures.add(new ExampleBasedRecall());
            measures.add(new ExampleBasedFMeasure());
            measures.add(new ExampleBasedAccuracy());
            // add label-based measures
            int numOfLabels = testSet.getNumLabels();
            measures.add(new MicroPrecision(numOfLabels));
            measures.add(new MicroRecall(numOfLabels));
            measures.add(new MicroFMeasure(numOfLabels));
            measures.add(new MacroPrecision(numOfLabels));
            measures.add(new MacroRecall(numOfLabels));
            measures.add(new MacroFMeasure(numOfLabels));
        }
        // add ranking-based measures if applicable
        if (prediction.hasRanking()) {
            // add ranking based measures
            measures.add(new OneError());
            measures.add(new AveragePrecision());
            measures.add(new IsError());
            measures.add(new ErrorSetSize());
            measures.add(new Coverage());
            measures.add(new RankingLoss());
        }
        // add hierarchical measures if applicable
        if (testSet.getLabelsMetaData().isHierarchy()) {
            measures.add(new HierarchicalLoss(testSet));
        }
        return evaluate(learner, testSet, measures);
    }

    private boolean[] getTrueLabels(Instance instance, int numLabels, int[] labelIndices) {

        boolean[] trueLabels = new boolean[numLabels];
        for (int counter = 0; counter < numLabels; counter++) {
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
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances mlDataSet) throws Exception {
        return crossValidate(learner, mlDataSet, DEFAULTFOLDS);
    }

    /**
     * Evaluates a {@link MultiLabelLearner} via cross-validation on given data set.
     * The default number of folds {@link Evaluator#DEFAULTFOLDS} will be used.
     *
     * @param learner the learner to be evaluated via cross-validation
     * @param mlDataSet the multi-label data set for cross-validation
     * @param measures the evaluation measures to compute
     * @return the evaluation result
     * @throws IllegalArgumentException if either of input parameters is null.
     * @throws Exception
     */
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances mlDataSet, List<Measure> measures) throws Exception {
        return crossValidate(learner, mlDataSet, measures, DEFAULTFOLDS);
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
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances mlDataSet, int numFolds)
            throws Exception {
        if (learner == null) {
            throw new IllegalArgumentException("Learner to be evaluated is null.");
        }
        if (mlDataSet == null) {
            throw new IllegalArgumentException("MutliLabelDataset for the evaluation is null.");
        }
        if (numFolds == 0 || numFolds == 1) {
            throw new IllegalArgumentException("Number of folds must be at least two or higher.");
        }

        Instances workingSet = new Instances(mlDataSet.getDataSet());

        if (numFolds < 0) {
            numFolds = workingSet.numInstances();
        }

        Evaluation[] evaluation = new Evaluation[numFolds];

        Random random = new Random(seed);
        workingSet.randomize(random);
        for (int i = 0; i < numFolds; i++) {
            Instances train = workingSet.trainCV(numFolds, i, random);
            Instances test = workingSet.testCV(numFolds, i);
            MultiLabelInstances mlTrain = new MultiLabelInstances(train, mlDataSet.getLabelsMetaData());
            MultiLabelInstances mlTest = new MultiLabelInstances(test, mlDataSet.getLabelsMetaData());
            MultiLabelLearner clone = learner.makeCopy();
            clone.build(mlTrain);
            evaluation[i] = evaluate(clone, mlTest);
        }
        return new MultipleEvaluation(evaluation);
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
     * @param measures the evaluation measures to compute
     * @param numFolds the number of folds to be used
     * @return the evaluation result
     * @throws IllegalArgumentException if either of learner or data set parameters is null
     * @throws IllegalArgumentException if number of folds is invalid
     * @throws Exception
     */
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances mlDataSet, List<Measure> measures, int numFolds)
            throws Exception {
        if (learner == null) {
            throw new IllegalArgumentException("Learner to be evaluated is null.");
        }
        if (mlDataSet == null) {
            throw new IllegalArgumentException("MutliLabelDataset for the evaluation is null.");
        }
        if (numFolds == 0 || numFolds == 1) {
            throw new IllegalArgumentException("Number of folds must be at least two or higher.");
        }
        if (measures == null) {
            throw new IllegalArgumentException("List of evaluation measures to compute is null.");
        }

        Instances workingSet = new Instances(mlDataSet.getDataSet());

        if (numFolds < 0) {
            numFolds = workingSet.numInstances();
        }

        Evaluation[] evaluation = new Evaluation[numFolds];

        Random random = new Random(seed);
        workingSet.randomize(random);
        for (int i = 0; i < numFolds; i++) {
            Instances train = workingSet.trainCV(numFolds, i, random);
            Instances test = workingSet.testCV(numFolds, i);
            MultiLabelInstances mlTrain = new MultiLabelInstances(train, mlDataSet.getLabelsMetaData());
            MultiLabelInstances mlTest = new MultiLabelInstances(test, mlDataSet.getLabelsMetaData());
            MultiLabelLearner clone = learner.makeCopy();
            clone.build(mlTrain);
            evaluation[i] = evaluate(clone, mlTest, measures);
        }
        return new MultipleEvaluation(evaluation);
    }
}

