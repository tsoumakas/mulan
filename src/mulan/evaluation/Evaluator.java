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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
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
 * @version 2010.11.06
 */
public class Evaluator {

    // seed for reproduction of cross-validation results
    private int seed = 1;
    // when false divisions-by-zero are ignored in certain measures
    private boolean strict = true;

    /**
     * Sets the seed for reproduction of cross-validation results 
     * 
     * @param aSeed seed for reproduction of cross-validation results
     */
    public void setSeed(int aSeed) {
        seed = aSeed;
    }

    /**
     * Controls how divisions-by-zero are handled
     *
     * @param isStrict when false divisions-by-zero are ignored
     */
    public void setStrict(boolean isStrict) {
        strict = isStrict;
    }

    /**
     * Evaluates a {@link MultiLabelLearner} on given test data set using specified evaluation measures
     *
     * @param learner the learner to be evaluated via cross-validation
     * @param data the data set for cross-validation
     * @param measures the evaluation measures to compute
     * @return an Evaluation object
     * @throws IllegalArgumentException if an input parameter is null
     * @throws Exception
     */
    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException, Exception {
        checkLearner(learner);
        checkData(data);
        checkMeasures(measures);

        // reset measures
        for (Measure m : measures) {
            m.reset();
        }

        int numLabels = data.getNumLabels();
        int[] labelIndices = data.getLabelIndices();
        boolean[] trueLabels = new boolean[numLabels];
        Set<Measure> failed = new HashSet<Measure>();
        Instances testData = data.getDataSet();
        int numInstances = testData.numInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            if (data.hasMissingLabels(instance)) {
                continue;
            }
            MultiLabelOutput output = learner.makePrediction(instance);
            trueLabels = getTrueLabels(instance, numLabels, labelIndices);
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

    private void checkLearner(MultiLabelLearner learner) {
        if (learner == null) {
            throw new IllegalArgumentException("Learner to be evaluated is null.");
        }
    }

    private void checkData(MultiLabelInstances data) {
        if (data == null) {
            throw new IllegalArgumentException("Evaluation data object is null.");
        }
    }

    private void checkMeasures(List<Measure> measures) {
        if (measures == null) {
            throw new IllegalArgumentException("List of evaluation measures to compute is null.");
        }
    }

    private void checkFolds(int someFolds) {
        if (someFolds < 2) {
            throw new IllegalArgumentException("Number of folds must be at least two or higher.");
        }
    }

    /**
     * Evaluates a {@link MultiLabelLearner} on given test data set.
     *
     * @param learner the learner to be evaluated
     * @param data the data set for evaluation
     * @return the evaluation result
     * @throws IllegalArgumentException if either of input parameters is null.
     * @throws Exception
     */
    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data) throws IllegalArgumentException, Exception {
        checkLearner(learner);
        checkData(data);

        List<Measure> measures = prepareMeasures(learner, data);

        return evaluate(learner, data, measures);
    }

    private List<Measure> prepareMeasures(MultiLabelLearner learner, MultiLabelInstances data) {
        List<Measure> measures = new ArrayList<Measure>();

        MultiLabelOutput prediction;
        try {
            MultiLabelLearner copyOfLearner = learner.makeCopy();
            prediction = copyOfLearner.makePrediction(data.getDataSet().instance(0));
            // add bipartition-based measures if applicable
            if (prediction.hasBipartition()) {
                // add example-based measures
                measures.add(new HammingLoss());
                measures.add(new SubsetAccuracy());
                measures.add(new ExampleBasedPrecision(strict));
                measures.add(new ExampleBasedRecall(strict));
                measures.add(new ExampleBasedFMeasure(strict));
                measures.add(new ExampleBasedAccuracy(strict));
                // add label-based measures
                int numOfLabels = data.getNumLabels();
                measures.add(new MicroPrecision(numOfLabels));
                measures.add(new MicroRecall(numOfLabels));
                measures.add(new MicroFMeasure(numOfLabels));
                measures.add(new MacroPrecision(numOfLabels, strict));
                measures.add(new MacroRecall(numOfLabels, strict));
                measures.add(new MacroFMeasure(numOfLabels, strict));
            }
            // add ranking-based measures if applicable
            if (prediction.hasRanking()) {
                // add ranking based measures
                measures.add(new AveragePrecision());
                measures.add(new Coverage());
                measures.add(new OneError());
                measures.add(new IsError());
                measures.add(new ErrorSetSize());
                measures.add(new RankingLoss());
            }
            // add confidence measures if applicable
            if (prediction.hasConfidences()) {
                int numOfLabels = data.getNumLabels();
                measures.add(new MeanAveragePrecision(numOfLabels));
                measures.add(new MicroAUC(numOfLabels));
                measures.add(new MacroAUC(numOfLabels));
            }
            // add hierarchical measures if applicable
            if (data.getLabelsMetaData().isHierarchy()) {
                measures.add(new HierarchicalLoss(data));
            }
        } catch (Exception ex) {
            Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        }

        return measures;
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
     * Evaluates a {@link MultiLabelLearner} via cross-validation on given data 
     * set with defined number of folds and seed.
     *
     * @param learner the learner to be evaluated via cross-validation
     * @param data the multi-label data set for cross-validation
     * @param someFolds 
     * @return a {@link MultipleEvaluation} object holding the results
     */
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances data, int someFolds)
    {
        checkLearner(learner);
        checkData(data);
        checkFolds(someFolds);

        return innerCrossValidate(learner, data, false, null, someFolds);
    }

    /**
     * Evaluates a {@link MultiLabelLearner} via cross-validation on given data
     * set using given evaluation measures with defined number of folds and seed.
     *
     * @param learner the learner to be evaluated via cross-validation
     * @param data the multi-label data set for cross-validation
     * @param measures the evaluation measures to compute
     * @param someFolds 
     * @return a {@link MultipleEvaluation} object holding the results
     */
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures, int someFolds)
    {
        checkLearner(learner);
        checkData(data);
        checkMeasures(measures);

        return innerCrossValidate(learner, data, true, measures, someFolds);
    }

    private MultipleEvaluation innerCrossValidate(MultiLabelLearner learner, MultiLabelInstances data, boolean hasMeasures, List<Measure> measures, int someFolds) {
        Evaluation[] evaluation = new Evaluation[someFolds];

        Instances workingSet = new Instances(data.getDataSet());
        workingSet.randomize(new Random(seed));
        for (int i = 0; i < someFolds; i++) {
            System.out.println("Fold " + (i + 1) + "/" + someFolds);
            try {
                Instances train = workingSet.trainCV(someFolds, i);
                Instances test = workingSet.testCV(someFolds, i);
                MultiLabelInstances mlTrain = new MultiLabelInstances(train, data.getLabelsMetaData());
                MultiLabelInstances mlTest = new MultiLabelInstances(test, data.getLabelsMetaData());
                MultiLabelLearner clone = learner.makeCopy();
                clone.build(mlTrain);
                if (hasMeasures)
                    evaluation[i] = evaluate(clone, mlTest, measures);
                else
                    evaluation[i] = evaluate(clone, mlTest);
            } catch (Exception ex) {
                Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return new MultipleEvaluation(evaluation);
    }
}

