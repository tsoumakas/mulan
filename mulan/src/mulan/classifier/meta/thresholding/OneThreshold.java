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
package mulan.classifier.meta.thresholding;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.core.MulanRuntimeException;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.BipartitionMeasureBase;
import mulan.evaluation.measure.HammingLoss;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

/**
 * <p> Class that estimates a single threshold for all labels and examples. For
 * more information, see <em>Read, Jesse, Pfahringer, Bernhard, Holmes, Geoff:
 * Multi-label Classification Using Ensembles of Pruned Sets. In: Data Mining,
 * 2008. ICDM '08. Eighth IEEE International Conference on, 995-1000, 2008.</em>
 * </p>
 *
 * @author Marios Ioannou
 * @author George Sakkas
 * @author Grigorios Tsoumakas
 * @version 2010.12.14
 */
public class OneThreshold extends MultiLabelMetaLearner {

    /**
     * final threshold value
     */
    private double threshold;
    /**
     * measure for auto-tuning the threshold
     */
    private BipartitionMeasureBase measure;
    /**
     * the folds of the cv to evaluate different thresholds
     */
    private int folds = 0;
    /**
     * copy of a clean multi-label learner to use at each fold
     */
    private MultiLabelLearner foldLearner;

    /**
     * Default constructor
     */
    public OneThreshold() {
        this(new BinaryRelevance(new J48()), new HammingLoss(), 3);
    }

    /**
     * @param baseLearner the underlying multi=label learner
     * @param aMeasure the measure to optimize
     * @param someFolds number of cross-validation folds
     */
    public OneThreshold(MultiLabelLearner baseLearner, BipartitionMeasureBase aMeasure, int someFolds) {
        super(baseLearner);
        if (someFolds < 2) {
            throw new IllegalArgumentException("folds should be more than 1");
        }
        measure = aMeasure;
        folds = someFolds;
        try {
            foldLearner = baseLearner.makeCopy();
        } catch (Exception ex) {
            Logger.getLogger(OneThreshold.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * @param baseLearner the underlying multi=label learner
     * @param aMeasure measure to optimize
     */
    public OneThreshold(MultiLabelLearner baseLearner, BipartitionMeasureBase aMeasure) {
        super(baseLearner);
        measure = aMeasure;
    }

    /**
     * Evaluates the performance of the learner on a data set according to a
     * bipartition measure for a range of thresholds
     *
     * @param data the test data to evaluate different thresholds
     * @param measure the evaluation is based on this parameter
     * @param min the minimum threshold
     * @param max the maximum threshold
     * @param the step to increase threshold from min to max
     * @return the optimal threshold
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private double computeThreshold(MultiLabelLearner learner, MultiLabelInstances data, BipartitionMeasureBase measure, double min, double step, double max) throws Exception {
        int numOfThresholds = (int) Math.rint((max - min) / step + 1);
        double[] performance = new double[numOfThresholds];
        BipartitionMeasureBase[] measureForThreshold = new BipartitionMeasureBase[numOfThresholds];
        for (int i = 0; i < numOfThresholds; i++) {
            measureForThreshold[i] = (BipartitionMeasureBase) measure.makeCopy();
            measureForThreshold[i].reset();
        }

        boolean[] thresholdHasProblem = new boolean[numOfThresholds];
        Arrays.fill(thresholdHasProblem, false);

        for (int j = 0; j < data.getNumInstances(); j++) {
            Instance instance = data.getDataSet().instance(j);

            if (data.hasMissingLabels(instance)) {
                continue;
            }

            MultiLabelOutput mlo = learner.makePrediction(instance);

            boolean[] trueLabels = new boolean[numLabels];
            for (int counter = 0; counter < numLabels; counter++) {
                int classIdx = labelIndices[counter];
                String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
                trueLabels[counter] = classValue.equals("1");
            }

            double[] confidences = mlo.getConfidences();


            int counter = 0;
            double currentThreshold = min;
            while (currentThreshold <= max) {
                boolean[] bipartition = new boolean[numLabels];
                for (int k = 0; k < numLabels; k++) {
                    if (confidences[k] >= currentThreshold) {
                        bipartition[k] = true;
                    }
                }
                try {
                    MultiLabelOutput temp = new MultiLabelOutput(bipartition);
                    measureForThreshold[counter].update(temp, new GroundTruth(trueLabels));
                } catch (MulanRuntimeException e) {
                    thresholdHasProblem[counter] = true;
                }
                currentThreshold += step;
                counter++;
            }
        }

        for (int i = 0; i < numOfThresholds; i++) {
            if (!thresholdHasProblem[i]) {
                performance[i] = Math.abs(measure.getIdealValue() - measureForThreshold[i].getValue());
            } else {
                performance[i] = Double.MAX_VALUE;
            }
        }

        return min + Utils.minIndex(performance) * step;
    }

    /**
     * Evaluates the measureForThreshold of different threshold values
     *
     * @param data the test data to evaluate different thresholds
     * @param measure the evaluation is based on this parameter
     * @return the sum of differences from the optimal value of the measure for
     * each instance and threshold
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private double computeThreshold(MultiLabelLearner learner, MultiLabelInstances data, BipartitionMeasureBase measure) throws Exception {
        double stage1 = computeThreshold(learner, data, measure, 0, 0.1, 1);
        debug("1st stage threshold = " + stage1);
        double stage2 = computeThreshold(learner, data, measure, stage1 - 0.05, 0.01, stage1 + 0.05);
        debug("2nd stage threshold = " + stage2);
        return stage2;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingData) throws Exception {
        baseLearner.build(trainingData);
        if (folds == 0) {
            threshold = computeThreshold(baseLearner, trainingData, measure);
        } else {
            LabelsMetaData labelsMetaData = trainingData.getLabelsMetaData();
            double[] thresholds = new double[folds];
            for (int f = 0; f < folds; f++) {
                Instances train = trainingData.getDataSet().trainCV(folds, f);
                MultiLabelInstances trainMulti = new MultiLabelInstances(train, labelsMetaData);
                Instances test = trainingData.getDataSet().testCV(folds, f);
                MultiLabelInstances testMulti = new MultiLabelInstances(test, labelsMetaData);
                MultiLabelLearner tempLearner = foldLearner.makeCopy();
                tempLearner.build(trainMulti);
                thresholds[f] = computeThreshold(tempLearner, testMulti, measure);
            }
            threshold = Utils.mean(thresholds);
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
        boolean[] predictedLabels;
        MultiLabelOutput mlo = baseLearner.makePrediction(instance);
        double[] confidences = mlo.getConfidences();
        predictedLabels = new boolean[numLabels];
        for (int i = 0; i < numLabels; i++) {
            if (confidences[i] >= threshold) {
                predictedLabels[i] = true;
            } else {
                predictedLabels[i] = false;
            }

        }
        MultiLabelOutput newOutput = new MultiLabelOutput(predictedLabels, mlo.getConfidences());
        return newOutput;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation info = new TechnicalInformation(Type.INPROCEEDINGS);
        info.setValue(Field.AUTHOR, "Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff");
        info.setValue(Field.YEAR, "2008");
        info.setValue(Field.TITLE, "Multi-label Classification Using Ensembles of Pruned Sets");
        info.setValue(Field.BOOKTITLE, "Data Mining, 2008. ICDM '08. Eighth IEEE International Conference on");
        info.setValue(Field.PAGES, "995-1000");
        info.setValue(Field.LOCATION, "Pisa, Italy");
        return info;
    }

    /**
     * Returns the calculated threshold
     *
     * @return the calculated threshold
     */
    public double getThreshold() {
        return threshold;
    }

}