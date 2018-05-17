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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.BinaryRelevance;
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
 * <p> Class that implements the SCut method (Score-based local optimization). 
 * It computes a separate threshold for each label based on improving a user 
 * defined performance measure.For more information, see <em> Yiming Yang: A 
 * study of thresholding strategies for text categorization. In: Proceedings of 
 * the 24th annual international ACM SIGIR conference on Research and 
 * development in information retrieval, 137 - 145, 2001.</em></p>
 *
 * @author Marios Ioannou
 * @author George Sakkas
 * @author Grigorios Tsoumakas
 * @version 2013.6.19
 */
public class SCut extends MultiLabelMetaLearner {

    /**
     * measure for auto-tuning the threshold
     */
    private BipartitionMeasureBase measure;
    /**
     * the folds of the cv to evaluate different thresholds
     */
    private int kFoldsCV;
    /**
     * one threshold for each label to consider relevant
     */
    private double[] thresholds;

    /**
     * Default constructor
     */
    public SCut() {
        this(new BinaryRelevance(new J48()), new HammingLoss(), 3);
    }

    /**
     * Constructor that initializes the learner with a base algorithm , Measure
     * and num of folds
     *
     * @param baseLearner the underlying multi-label learner
     * @param measure the measure for auto-tuning the threshold
     * @param folds the number of folds to split the dataset
     */
    public SCut(MultiLabelLearner baseLearner, BipartitionMeasureBase measure, int folds) {
        super(baseLearner);
        this.measure = measure;
        this.kFoldsCV = folds;
    }

    /**
     * Creates a new instance of SCut
     *
     * @param baseLearner the underlying multi-label learner
     * @param measure the measure for auto-tuning the threshold
     */
    public SCut(MultiLabelLearner baseLearner, BipartitionMeasureBase measure) {
        super(baseLearner);
        this.measure = measure;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Yiming Yang");
        result.setValue(Field.TITLE, "A study of thresholding strategies for text categorization");
        result.setValue(Field.BOOKTITLE, "Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval");
        result.setValue(Field.PAGES, "137 - 145");
        result.setValue(Field.LOCATION, "New Orleans, Louisiana, United States");
        result.setValue(Field.YEAR, "2001");

        return result;
    }

    /**
     * Evaluates the performance of different threshold values for each label
     *
     * @param baseRegressor the underlying multi-label learner
     * @param data the test data to evaluate different thresholds
     * @return one threshold for each label
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private double[] computeThresholds(MultiLabelLearner learner, MultiLabelInstances data) throws Exception {

        double[][] arraysOfConfidences = new double[data.getNumInstances()][numLabels];
        boolean[][] trueLabels = new boolean[data.getNumInstances()][numLabels];
        List<Double>[] conf = new ArrayList[numLabels];
        for (int l = 0; l < numLabels; l++) {
            conf[l] = new ArrayList();
        }
        //get the Confidences and TrueLabels from all instances
        for (int j = 0; j < data.getNumInstances(); j++) {
            try {
                arraysOfConfidences[j] = learner.makePrediction(data.getDataSet().instance(j)).getConfidences();
            } catch (Exception ex) {
                Logger.getLogger(SCut.class.getName()).log(Level.SEVERE, null, ex);
            }
            for (int l = 0; l < numLabels; l++) {
                int labelIndice = labelIndices[l];
                trueLabels[j][l] = data.getDataSet().attribute(labelIndice).value((int) data.getDataSet().instance(j).value(labelIndice)).equals("1");
                conf[l].add(arraysOfConfidences[j][l]);
            }
        }

        double[] currentThresholds = new double[numLabels];
        double[][] measureTable = new double[3][numLabels];
        // sorting the confidences and set initial threshohlds for all labels
        for (int l = 0; l < numLabels; l++) {
            Collections.sort(conf[l]);
            currentThresholds[l] = 0.5;
        }

        double counter = 0;
        double tempThreshold;
        int conv;
        int numOfThresholds = data.getNumInstances();
        double[] performance = new double[numOfThresholds];

        BipartitionMeasureBase[] measureForThreshold = new BipartitionMeasureBase[numOfThresholds];
        for (int i = 0; i < numOfThresholds; i++) {
            measureForThreshold[i] = (BipartitionMeasureBase) measure.makeCopy();
            measureForThreshold[i].reset();
        }

        do {
            System.arraycopy(measureTable[0], 0, measureTable[1], 0, numLabels);
            //for all labels computing the best thresholds
            for (int j = 0; j < numLabels; j++) {
                double score = 0;
                //get a measure for all Thresholds
                for (int l = numOfThresholds - 1; l >= 0; l--) //posa instances diladi tosa thresshold
                {
                    measureForThreshold[l].reset();
                    if (l == 0) {
                        currentThresholds[j] = conf[j].get(l);
                    } else {
                        currentThresholds[j] = (conf[j].get(l) + conf[j].get(l - 1)) / 2;
                    }
                    //get the predicted labels for all instances according to Thresholds
                    for (int k = 0; k < data.getNumInstances(); k++) {
                        boolean[] predictedLabels = new boolean[numLabels];
                        for (int x = 0; x < numLabels; x++) {
                            predictedLabels[x] = (arraysOfConfidences[k][x] >= currentThresholds[x]);
                        }
                        MultiLabelOutput temp = new MultiLabelOutput(predictedLabels);
                        measureForThreshold[l].update(temp, new GroundTruth(trueLabels[k]));
                    }
                    score += measureForThreshold[l].getValue();
                }
                for (int i = 0; i < numOfThresholds; i++) {
                    performance[i] = Math.abs(measure.getIdealValue() - measureForThreshold[i].getValue());
                }
                int t = Utils.minIndex(performance);
                if (t == 0) {
                    tempThreshold = conf[j].get(t);
                } else {
                    tempThreshold = (conf[j].get(t) + conf[j].get(t - 1)) / 2;
                }
                // get the curent measure
                measureTable[0][j] = score;
                currentThresholds[j] = tempThreshold;
                //get the first measure
                if (counter == 0) {
                    measureTable[2][j] = score;
                }
            }
            conv = 0;
            // find if the two last mesures of all labels are converge
            for (int l = 0; l < numLabels; l++) {
                //  (curent measure-old measure)/first measure
                if ((Math.abs((measureTable[0][l] - measureTable[1][l]))) / measureTable[2][l] < 0.001 && counter != 0) {
                    conv++;
                }
            }
            counter++;
        } while (conv != numLabels);

        return currentThresholds;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        if (kFoldsCV == 0) {
            baseLearner.build(trainingSet);
            thresholds = computeThresholds(baseLearner, trainingSet);
        } else {
            thresholds = new double[numLabels];
            double[] foldThresholds;
            for (int i = 0; i
                    < kFoldsCV; i++) {
                //Split data to train and test sets
                Instances train = trainingSet.getDataSet().trainCV(kFoldsCV, i);
                MultiLabelInstances mlTrain = new MultiLabelInstances(train, trainingSet.getLabelsMetaData());
                Instances test = trainingSet.getDataSet().testCV(kFoldsCV, i);
                MultiLabelInstances mlTest = new MultiLabelInstances(test, trainingSet.getLabelsMetaData());
                MultiLabelLearner learner = baseLearner.makeCopy();
                learner.build(mlTrain);
                foldThresholds =
                        computeThresholds(learner, mlTest);
                for (int j = 0; j
                        < numLabels; j++) {
                    thresholds[j] += foldThresholds[j];
                }

            }
            for (int j = 0; j < numLabels; j++) {
                thresholds[j] /= kFoldsCV;
            }

            baseLearner.build(trainingSet);
        }

    }

    @Override
    public MultiLabelOutput makePredictionInternal(
            Instance instance) throws Exception {

        MultiLabelOutput m = baseLearner.makePrediction(instance);
        double[] arrayOfConfidences = new double[numLabels];
        boolean[] predictedLabels = new boolean[numLabels];

        //Confidences higher than threshold set it as true label
        if (m.hasConfidences()) {
            arrayOfConfidences = m.getConfidences();
            for (int i = 0; i
                    < numLabels; i++) {
                if (arrayOfConfidences[i] >= thresholds[i]) {
                    predictedLabels[i] = true;
                } else {
                    predictedLabels[i] = false;
                }

            }
        }
        MultiLabelOutput final_mlo = new MultiLabelOutput(predictedLabels, arrayOfConfidences);
        return final_mlo;
    }

    /**
     * Method to obtain the computed thresholds per label
     *
     * @return the computed thresholds per label
     */
    public double[] getThresholds() {
        return thresholds;
    }
}