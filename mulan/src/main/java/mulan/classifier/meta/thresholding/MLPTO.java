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
import java.util.Comparator;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.ExampleBasedBipartitionMeasureBase;
import mulan.evaluation.measure.HammingLoss;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <p> Class that implements the Multi Label Probabilistic Threshold Optimizer
 * (MLTPTO). For more information, see <em> J.R. Quevedo, O. Luaces, A.
 * Bahamonde (2012). Multilabel classifiers with a probabilistic thresholding
 * strategy. Pattern Recognition. 45(2):876-883.</em></p>
 *
 * @author D. Toimil, J. R. Quevedo, O. Luaces
 * @version 2012.02.02
 */
public class MLPTO extends MultiLabelMetaLearner {

    private ExampleBasedBipartitionMeasureBase EBBM;

    /**
     * Default constructor
     */
    public MLPTO() {
        this(new BinaryRelevance(new J48()), new HammingLoss());
    }

    /**
     * @param baseLearner the underlying multi-label learner
     * @param EBBM the measure function to be optimized. The measure is
     * optimized minimizing the distance to its ideal value (using IdealValue()
     * method). For measures with 1 as ideal value, like F1 or Accuracy, this
     * algorithm searches for the highest value (the nearest to 1). For measures
     * with 0 as ideal value, like Hamming, this algorithm searches for the
     * lowest value (the nearest to 0).
     */
    public MLPTO(MultiLabelLearner baseLearner, ExampleBasedBipartitionMeasureBase EBBM) {
        super(baseLearner);
        this.EBBM = EBBM;
    }

    /**
     * Searches the number of labels to be selected that optimizes the loss
     * function
     *
     * @param orderedProbabilities a descending ordered array with the
     * probabilities of the labels
     * @param EBBM the measure function to be optimized.
     * @return the number of labels to optimize the loss function
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private int OptimizeFLoss(double[] orderedProbabilities, ExampleBasedBipartitionMeasureBase EBBM) throws Exception {
        int NLabels;
        int L = orderedProbabilities.length;
        double P;
        double[][] Pc = new double[L + 2][L + 1];

        // Calculate the C probabilities
        for (int a = -2; a < L; a++) {
            for (int From = L; From >= 0; From--) {
                if (a == -2) {
                    P = 0;
                } else if (From == L - 1 && a == -1) {
                    P = 1 - orderedProbabilities[L - 1];
                } else if (From == L - 1 && a == 0) {
                    P = orderedProbabilities[L - 1];
                } else if (a + 1 > (L - From)) {
                    P = 0;
                } else if ((From) == L && a == -1) {
                    P = 1;
                } else {
                    P = orderedProbabilities[From]
                            * Pc[a + 2 - 1][From + 1]
                            + (1 - orderedProbabilities[From])
                            * Pc[a + 2][From + 1];
                }
                Pc[a + 2][From] = P;
            }
        }

        double[][] Pa = new double[L + 2][L];

        // Calculate the A probabilities
        for (int a = -2; a < L; a++) {
            for (int To = 0; To < L; To++) {
                if (a == -2) {
                    P = 0;
                } else if (To == 0 && a == -1) {
                    P = 1 - orderedProbabilities[0];
                } else if (To == 0 && a == 0) {
                    P = orderedProbabilities[0];
                } else if (a > (To - 1 + 1)) {
                    P = 0;
                } else {
                    P = orderedProbabilities[To]
                            * Pa[a + 2 - 1][To - 1]
                            + (1 - orderedProbabilities[To])
                            * Pa[a + 2][To - 1];
                }
                Pa[a + 2][To] = P;
            }
        }

        // The search algorithm
        int BestR = 0;
        double BestLoss = Double.POSITIVE_INFINITY;

        for (int R = 1; R <= L; R++) {
            double TotalMeasure = 0;
            for (int a = 0; a <= R; a++) {
                int b = R - a;
                for (int c = 0; c <= (L - R); c++) {
                    // Distance from the measure value to the IdealValue
                    double TheMeasure = Math.abs(EBBM.getIdealValue() - CalculateMeasure(EBBM, a, b, c, L));
                    double TheMeasure1 = TheMeasure * Pc[c + 1][R] * Pa[a + 1][R - 1];
                    TotalMeasure = TotalMeasure + TheMeasure1;
                }
            }
            if (BestLoss > TotalMeasure) { // Minimize the distance to the IdealValue
                BestLoss = TotalMeasure;
                BestR = R;
            }
        }
        NLabels = BestR;
        return NLabels;
    }

    private double CalculateMeasure(ExampleBasedBipartitionMeasureBase EBBM, int a, int b, int c, int L) {
        // Create a bipartition experiment that agrees with a,b,c,L
        boolean bipartition[] = new boolean[L];
        boolean truth[] = new boolean[L];

        int p = 0;
        for (int i = 0; i < a; i++) // a (true positives)
        {
            bipartition[p] = true;
            truth[p] = true;
            p++;
        }
        for (int i = 0; i < b; i++) // b (false positives)
        {
            bipartition[p] = true;
            truth[p] = false;
            p++;
        }
        for (int i = 0; i < c; i++) // c (false negatives)
        {
            bipartition[p] = false;
            truth[p] = true;
            p++;
        }
        for (int i = 0; i < (L - a - b - c); i++) // d (true negatives)
        {
            bipartition[p] = false;
            truth[p] = false;
            p++;
        }

        // Creates a MultiLabelOutput in order to use update methodand then getVaule
        mulan.classifier.MultiLabelOutput MLO = new mulan.classifier.MultiLabelOutput(bipartition);
        EBBM.update(MLO, new GroundTruth(truth));
        double val = EBBM.getValue();
        EBBM.reset();
        return val;
    }

    /**
     * Calculates the threshold that optimizes the given loss function. That
     * threshold separates the N labels with more probability.
     *
     * @param confidences an array with the probabilities of each label
     * @return the optimal threshold for the given loss function
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private double calculateThreshold(double[] confidences) throws Exception {
        double newThreshold;

        double[] orderedConfidences = sort(confidences);
        int NLabels = OptimizeFLoss(orderedConfidences, EBBM);

        if (NLabels == orderedConfidences.length) {
            newThreshold = (orderedConfidences[NLabels - 1] + 0.0) / 2;
        } else {
            newThreshold = (orderedConfidences[NLabels - 1] + orderedConfidences[Math.min(NLabels, orderedConfidences.length - 1)]) / 2;
        }

        return newThreshold;
    }

    /**
     * Sorts the given vector in descending order
     *
     * @param vector the array to be ordered
     * @return the descending ordered vector
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private double[] sort(double[] vector) {
        ArrayList<Double> array = new ArrayList<>();
        Comparator comparator = Collections.reverseOrder();
        for (int i = 0; i < vector.length; i++) {
            array.add(vector[i]);
        }
        Collections.sort(array, comparator);
        double[] orderedVector = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            orderedVector[i] = array.get(i);
        }
        return orderedVector;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingData) throws Exception {
        baseLearner.build(trainingData);
    }

    /**
     * Produces the optimal bipartition output from a probabilistic multi label
     * output for a predefined loss function. This method evaluates the example
     * using the multi label base-learner to get the labels' probability. Then,
     * it calculates the threshold that optimizes the loss (as especified in the
     * constructor's param FLoss). Finally, this threshold is used to generate a
     * bipartite multi label output.
     *
     * @param instance Test example.
     */
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] predictedLabels;
        MultiLabelOutput mlo = baseLearner.makePrediction(instance);
        double[] confidences = mlo.getConfidences();
        double threshold = calculateThreshold(confidences);
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
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "J.R. Quevedo and O. Luaces and A. Bahamonde");
        result.setValue(Field.TITLE, "Multilabel classifiers with a probabilistic thresholding strategy");
        result.setValue(Field.JOURNAL, "Pattern Recognition");
        result.setValue(Field.VOLUME, "45");
        result.setValue(Field.NUMBER, "2");
        result.setValue(Field.YEAR, "2012");
        result.setValue(Field.ISSN, "0031-3203");
        result.setValue(Field.PAGES, "876-883");
        result.setValue(Field.PUBLISHER, "Elsevier");

        return result;
    }
}