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
package mulan.evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.clus.ClusWrapperClassification;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.*;
import weka.core.Instance;
import weka.core.Instances;
import clus.Clus;

/**
 * Evaluator - responsible for generating evaluation data
 * 
 * @author rofr
 * @author Grigorios Tsoumakas
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2012.11.25
 */
public class Evaluator {

    // seed for reproduction of cross-validation results
    private int seed = 1;

    /**
     * Sets the seed for reproduction of cross-validation results
     * 
     * @param aSeed seed for reproduction of cross-validation results
     */
    public void setSeed(int aSeed) {
        seed = aSeed;
    }

    /**
     * Evaluates a {@link MultiLabelLearner} on given test data set using specified evaluation
     * measures
     * 
     * @param learner the learner to be evaluated
     * @param data the data set for evaluation
     * @param measures the evaluation measures to compute
     * @return an Evaluation object
     * @throws IllegalArgumentException if an input parameter is null
     * @throws Exception
     */
    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException,
            Exception {
        checkLearner(learner);
        checkData(data);
        checkMeasures(measures);

        // reset measures
        for (Measure m : measures) {
            m.reset();
        }

        int numLabels = data.getNumLabels();
        int[] labelIndices = data.getLabelIndices();
        GroundTruth truth;
        Set<Measure> failed = new HashSet<>();
        Instances testData = data.getDataSet();
        int numInstances = testData.numInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            if (data.hasMissingLabels(instance)) {
                continue; 
            }
            Instance labelsMissing = (Instance) instance.copy();
            labelsMissing.setDataset(instance.dataset());
            for (int i = 0; i < data.getNumLabels(); i++) {
                labelsMissing.setMissing(labelIndices[i]);
            }
            MultiLabelOutput output = learner.makePrediction(labelsMissing);
            if (output.hasPvalues()) {// check if we have regression outputs
                truth = new GroundTruth(getTrueScores(instance, numLabels, labelIndices));
            } else {
                truth = new GroundTruth(getTrueLabels(instance, numLabels, labelIndices));
            }
            Iterator<Measure> it = measures.iterator();
            while (it.hasNext()) {
                Measure m = it.next();
                if (!failed.contains(m)) {
                    try {
                        m.update(output, truth);
                    } catch (Exception ex) {
                        failed.add(m);
                    }
                }
            }
        }

        return new Evaluation(measures, data);
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
    public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data,
            MultiLabelInstances trainData) throws IllegalArgumentException, Exception {
        checkLearner(learner);
        checkData(data);

        List<Measure> measures = prepareMeasures(learner, data, trainData);

        if (learner instanceof ClusWrapperClassification) {
            return evaluate((ClusWrapperClassification) learner, data, measures);
        } else {
            return evaluate(learner, data, measures);
        }
    }

    private List<Measure> prepareMeasures(MultiLabelLearner learner, MultiLabelInstances data,
            MultiLabelInstances trainData) {
        List<Measure> measures = new ArrayList<>();

        MultiLabelOutput prediction;
        try {
            MultiLabelLearner copyOfLearner = learner.makeCopy();
            prediction = copyOfLearner.makePrediction(data.getDataSet().instance(0));
            int numOfLabels = data.getNumLabels();
            // add bipartition-based measures if applicable
            if (prediction.hasBipartition()) {
                // add example-based measures
                measures.add(new HammingLoss());
                measures.add(new SubsetAccuracy());
                measures.add(new ExampleBasedPrecision());
                measures.add(new ExampleBasedRecall());
                measures.add(new ExampleBasedFMeasure());
                measures.add(new ExampleBasedAccuracy());
                measures.add(new ExampleBasedSpecificity());
                // add label-based measures
                measures.add(new MicroPrecision(numOfLabels));
                measures.add(new MicroRecall(numOfLabels));
                measures.add(new MicroFMeasure(numOfLabels));
                measures.add(new MicroSpecificity(numOfLabels));
                measures.add(new MacroPrecision(numOfLabels));
                measures.add(new MacroRecall(numOfLabels));
                measures.add(new MacroFMeasure(numOfLabels));
                measures.add(new MacroSpecificity(numOfLabels));
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
                measures.add(new MeanAveragePrecision(numOfLabels));
                measures.add(new GeometricMeanAveragePrecision(numOfLabels));
                measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
                measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
                measures.add(new MicroAUC(numOfLabels));
                measures.add(new MacroAUC(numOfLabels));
            }
            // add hierarchical measures if applicable
            if (data.getLabelsMetaData().isHierarchy()) {
                measures.add(new HierarchicalLoss(data));
            }
            // add regression measures if applicable
            if (prediction.hasPvalues()) {
                measures.add(new AverageRMSE(numOfLabels));
                measures.add(new AverageRelativeRMSE(numOfLabels, trainData, data));
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

    private double[] getTrueScores(Instance instance, int numLabels, int[] labelIndices) {

        double[] trueScores = new double[numLabels];
        for (int counter = 0; counter < numLabels; counter++) {
            int classIdx = labelIndices[counter];
            double score;
            if (instance.isMissing(classIdx)) {// if target is missing
                score = Double.NaN; // make it equal to Double.Nan
            } else {
                score = instance.value(classIdx);
            }
            trueScores[counter] = score;
        }

        return trueScores;
    }

    /**
     * Evaluates a {@link MultiLabelLearner} via cross-validation on given data set with defined
     * number of folds and seed.
     * 
     * @param learner the learner to be evaluated via cross-validation
     * @param data the multi-label data set for cross-validation
     * @param someFolds
     * @return a {@link MultipleEvaluation} object holding the results
     */
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances data,
            int someFolds) {
        checkLearner(learner);
        checkData(data);
        checkFolds(someFolds);

        return innerCrossValidate(learner, data, false, null, someFolds);
    }

    /**
     * Evaluates a {@link MultiLabelLearner} via cross-validation on given data set using given
     * evaluation measures with defined number of folds and seed.
     * 
     * @param learner the learner to be evaluated via cross-validation
     * @param data the multi-label data set for cross-validation
     * @param measures the evaluation measures to compute
     * @param someFolds
     * @return a {@link MultipleEvaluation} object holding the results
     */
    public MultipleEvaluation crossValidate(MultiLabelLearner learner, MultiLabelInstances data,
            List<Measure> measures, int someFolds) {
        checkLearner(learner);
        checkData(data);
        checkMeasures(measures);

        return innerCrossValidate(learner, data, true, measures, someFolds);
    }

    private MultipleEvaluation innerCrossValidate(MultiLabelLearner learner,
            MultiLabelInstances data, boolean hasMeasures, List<Measure> measures, int someFolds) {
        Evaluation[] evaluation = new Evaluation[someFolds];

        Instances workingSet = new Instances(data.getDataSet());
        workingSet.randomize(new Random(seed));
        for (int i = 0; i < someFolds; i++) {
            System.out.println("Fold " + (i + 1) + "/" + someFolds);
            try {
                Instances train = workingSet.trainCV(someFolds, i);
                Instances test = workingSet.testCV(someFolds, i);
                MultiLabelInstances mlTrain = new MultiLabelInstances(train, data
                        .getLabelsMetaData());
                MultiLabelInstances mlTest = new MultiLabelInstances(test, data.getLabelsMetaData());
                MultiLabelLearner clone = learner.makeCopy();
                clone.build(mlTrain);
                if (hasMeasures) {
                    evaluation[i] = evaluate(clone, mlTest, measures);
                }
                else {
                    evaluation[i] = evaluate(clone, mlTest, mlTrain);
                }
            } catch (Exception ex) {
                Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        MultipleEvaluation me = new MultipleEvaluation(evaluation, data);
        me.calculateStatistics();
        return me;
    }

    /**
     * Evaluates a {@link ClusWrapperClassification} on given test data set using specified
     * evaluation measures
     * 
     * @param learner the learner to be evaluated
     * @param testData the data set for evaluation
     * @param measures the evaluation measures to compute
     * @return an Evaluation object
     * @throws IllegalArgumentException if an input parameter is null
     * @throws Exception
     */
    public Evaluation evaluate(ClusWrapperClassification learner, MultiLabelInstances testData, List<Measure> measures) throws IllegalArgumentException,
            Exception {

        boolean isEnsemble = learner.isEnsemble();
        if (isEnsemble) {
            throw new Exception("Evaluation of CLUS ensemble algorithms is not supported yet!");
        }
        boolean isRegression;
        MultiLabelOutput output = learner.makePrediction(testData.getDataSet().instance(0));
        if (output.hasPvalues()) {
            isRegression = true;
            throw new Exception("Evaluation of CLUS regression algorithms is not supported yet!");
        } else {
            isRegression = false;
        }

        String clusWorkingDir = learner.getClusWorkingDir();
        String datasetName = learner.getDatasetName();
        // write the supplied MultilabelInstances object in an arff formated file (accepted by CLUS)
        ClusWrapperClassification.makeClusCompliant(testData, clusWorkingDir + datasetName
                + "-test.arff");

        // call Clus.main to write the output files!
        if (!isEnsemble) {
            // the only argument passed to Clus is the settings file!
            String[] clusArgs = new String[1];
            clusArgs[0] = clusWorkingDir + datasetName + "-train.s";
            Clus.main(clusArgs);
        } else {
            String[] clusArgs = new String[2];
            clusArgs[0] = "-forest";
            clusArgs[1] = clusWorkingDir + datasetName + "-train.s";
            Clus.main(clusArgs);
        }

        // then parse the output files and finally update the measures!
        // open and load the test set predictions file, which is in arff format
        String predictionsFilePath = clusWorkingDir + datasetName + "-train.test.pred.arff";
        BufferedReader reader = new BufferedReader(new FileReader(predictionsFilePath));
        Instances predictionInstances = new Instances(reader);
        reader.close();

        checkLearner(learner);
        checkData(testData);
        checkMeasures(measures);

        // reset measures
        for (Measure m : measures) {
            m.reset();
        }

        int numLabels = testData.getNumLabels();
        Set<Measure> failed = new HashSet<>();
        Instances testDataset = testData.getDataSet();

        int numInstances = testDataset.numInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testDataset.instance(instanceIndex);
            if (testData.hasMissingLabels(instance)) {
                continue;
            }
            Instance labelsMissing = (Instance) instance.copy();
            labelsMissing.setDataset(instance.dataset());
            for (int i = 0; i < testData.getNumLabels(); i++) {
                labelsMissing.setMissing(testData.getLabelIndices()[i]);
            }
            // this part should be replaced
            GroundTruth truth;
            boolean[] trueLabels = new boolean[numLabels];
            double[] trueValues = new double[numLabels];

            // original way
            // output = learner.makePrediction(labelsMissing);
            // trueLabels = getTrueLabels(instance, numLabels, labelIndices);

            // clus way
            Instance predictionInstance = predictionInstances.instance(instanceIndex);
            double[] predictionsPerSample = new double[testData.getNumLabels()];
            int k = 0;
            for (int j = 0; j < predictionInstance.numValues() - 1; j++) {
                String pred = predictionInstance.toString(j);
                if (j < testData.getNumLabels()) {
                    trueValues[j] = Double.parseDouble(pred);
                    if (Double.parseDouble(pred) > 0.5) {
                        trueLabels[j] = true;
                    } else {
                        trueLabels[j] = false;
                    }
                }
                if (isEnsemble) {
                    if (j >= testData.getNumLabels() * 2) {
                        predictionsPerSample[k] = predictionInstance.value(j)
                                / (predictionInstance.value(j) + predictionInstance.value(j + 1));
                        j++;
                        k++;
                    }
                } else {
                    if (j >= (testData.getNumLabels() * 5 + 1)) {
                        predictionsPerSample[k] = predictionInstance.value(j)
                                / (predictionInstance.value(j) + predictionInstance.value(j + 1));
                        j++;
                        k++;
                    }
                }
            }

            if (!isRegression) {
                output = new MultiLabelOutput(predictionsPerSample, 0.5);
                truth = new GroundTruth(trueLabels);
            } else {
                output = new MultiLabelOutput(predictionsPerSample, true);
                truth = new GroundTruth(trueValues);
            }

            Iterator<Measure> it = measures.iterator();
            while (it.hasNext()) {
                Measure m = it.next();
                if (!failed.contains(m)) {
                    try {
                        m.update(output, truth);
                    } catch (Exception ex) {
                        failed.add(m);
                    }
                }
            }
        }

        return new Evaluation(measures, testData);
    }
}