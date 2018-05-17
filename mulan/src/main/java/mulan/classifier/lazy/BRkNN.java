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
package mulan.classifier.lazy;

import java.util.ArrayList;
import java.util.Random;
import mulan.classifier.MultiLabelOutput;
import mulan.core.Util;
import mulan.data.MultiLabelInstances;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * Simple BR implementation of the KNN algorithm.For more information, see<br>
 * <p>
 * Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas: An Empirical Study of Lazy Multilabel Classification Algorithms. In: Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008), 2008.
 * </p>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{EleftheriosSpyromitros2008,
 *    author = {Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas},
 *    booktitle = {Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008)},
 *    title = {An Empirical Study of Lazy Multilabel Classification Algorithms},
 *    year = {2008},
 *    location = {Syros, Greece}
 * }
 * </pre>
 * <br>
 <!-- technical-bibtex-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis 
 * @author Grigorios Tsoumakas
 * @version 2010.12.29
 */
public class BRkNN extends MultiLabelKNN {

    private Random random;
    /**
     * Stores the average number of labels among the knn for each instance Used
     * in BRkNN-b extension
     */
    private int avgPredictedLabels;
    /**
     * The value of kNN provided by the user. This may differ from
     * numOfNeighbors if cross-validation is being used.
     */
    private int cvMaxK;
    /**
     * Whether to select k by cross validation.
     */
    private boolean cvkSelection = false;

    /**
     * The two types of extensions
     */
    public enum ExtensionType {

        /**
         * Standard BR
         */
        NONE,
        /**
         * Predict top ranked label in case of empty prediction set
         */
        EXTA,
        /**
         * Predict top n ranked labels based on size of labelset in neighbors
         */
        EXTB
    };
    /**
     * The type of extension to be used
     */
    private ExtensionType extension = ExtensionType.NONE;

    /**
     * Default constructor
     */
    public BRkNN() {
        this(10, ExtensionType.NONE);
    }
    
    /**
     * A constructor that sets the number of neighbors
     *
     * @param numOfNeighbors the number of neighbors
     */
    public BRkNN(int numOfNeighbors) {
        this(numOfNeighbors, ExtensionType.NONE);
    }

    /**
     * Constructor giving the option to select an extension of the base version
     *
     * @param numOfNeighbors the number of neighbors
     * @param ext the extension to use (see {@link ExtensionType})
     */
    public BRkNN(int numOfNeighbors, ExtensionType ext) {
        super(numOfNeighbors);
        random = new Random(1);
        extension = ext;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas");
        result.setValue(Field.TITLE, "An Empirical Study of Lazy Multilabel Classification Algorithms");
        result.setValue(Field.BOOKTITLE, "Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008)");
        result.setValue(Field.LOCATION, "Syros, Greece");
        result.setValue(Field.YEAR, "2008");
        return result;
    }

    @Override
    protected void buildInternal(MultiLabelInstances aTrain) throws Exception {
        super.buildInternal(aTrain);

        if (cvkSelection == true) {
            crossValidate();
        }
    }

    /**
     *
     * @param flag
     *            if true the k is selected via cross-validation
     */
    public void setkSelectionViaCV(boolean flag) {
        cvkSelection = flag;
    }

    /**
     * Select the best value for k by hold-one-out cross-validation. Hamming
     * Loss is minimized
     *
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void crossValidate() throws Exception {
        try {
            // the performance for each different k
            double[] hammingLoss = new double[cvMaxK];

            for (int i = 0; i < cvMaxK; i++) {
                hammingLoss[i] = 0;
            }

            Instances dataSet = train;
            Instance instance; // the hold out instance
            Instances neighbours; // the neighboring instances
            double[] origDistances, convertedDistances;
            for (int i = 0; i < dataSet.numInstances(); i++) {
                if (getDebug() && (i % 50 == 0)) {
                    debug("Cross validating " + i + "/" + dataSet.numInstances() + "\r");
                }
                instance = dataSet.instance(i);
                neighbours = lnn.kNearestNeighbours(instance, cvMaxK);
                origDistances = lnn.getDistances();

                // gathering the true labels for the instance
                boolean[] trueLabels = new boolean[numLabels];
                for (int counter = 0; counter < numLabels; counter++) {
                    int classIdx = labelIndices[counter];
                    String classValue = instance.attribute(classIdx).value(
                            (int) instance.value(classIdx));
                    trueLabels[counter] = classValue.equals("1");
                }
                // calculate the performance metric for each different k
                for (int j = cvMaxK; j > 0; j--) {
                    convertedDistances = new double[origDistances.length];
                    System.arraycopy(origDistances, 0, convertedDistances, 0,
                            origDistances.length);
                    double[] confidences = this.getConfidences(neighbours,
                            convertedDistances);
                    boolean[] bipartition = null;

                    switch (extension) {
                        case NONE: // BRknn
                            MultiLabelOutput results;
                            results = new MultiLabelOutput(confidences, 0.5);
                            bipartition = results.getBipartition();
                            break;
                        case EXTA: // BRknn-a
                            bipartition = labelsFromConfidences2(confidences);
                            break;
                        case EXTB: // BRknn-b
                            bipartition = labelsFromConfidences3(confidences);
                            break;
                    }

                    double symmetricDifference = 0; // |Y xor Z|
                    for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
                        boolean actual = trueLabels[labelIndex];
                        boolean predicted = bipartition[labelIndex];

                        if (predicted != actual) {
                            symmetricDifference++;
                        }
                    }
                    hammingLoss[j - 1] += (symmetricDifference / numLabels);

                    neighbours = new IBk().pruneToK(neighbours,
                            convertedDistances, j - 1);
                }
            }

            // Display the results of the cross-validation
            if (getDebug()) {
                for (int i = cvMaxK; i > 0; i--) {
                    debug("Hold-one-out performance of " + (i) + " neighbors ");
                    debug("(Hamming Loss) = " + hammingLoss[i - 1] / dataSet.numInstances());
                }
            }

            // Check through the performance stats and select the best
            // k value (or the lowest k if more than one best)
            double[] searchStats = hammingLoss;

            double bestPerformance = Double.NaN;
            int bestK = 1;
            for (int i = 0; i < cvMaxK; i++) {
                if (Double.isNaN(bestPerformance) || (bestPerformance > searchStats[i])) {
                    bestPerformance = searchStats[i];
                    bestK = i + 1;
                }
            }
            numOfNeighbors = bestK;
            if (getDebug()) {
                System.err.println("Selected k = " + bestK);
            }

        } catch (Exception ex) {
            throw new Error("Couldn't optimize by cross-validation: " + ex.getMessage());
        }
    }

    /**
     * weka Ibk style prediction
     *
     * @throws Exception if nearest neighbours search fails
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        Instances knn = lnn.kNearestNeighbours(instance, numOfNeighbors);

        double[] distances = lnn.getDistances();
        double[] confidences = getConfidences(knn, distances);
        boolean[] bipartition;

        MultiLabelOutput results = null;
        switch (extension) {
            case NONE: // BRknn
                results = new MultiLabelOutput(confidences, 0.5);
                break;
            case EXTA: // BRknn-a
                bipartition = labelsFromConfidences2(confidences);
                results = new MultiLabelOutput(bipartition, confidences);
                break;
            case EXTB: // BRknn-b
                bipartition = labelsFromConfidences3(confidences);
                results = new MultiLabelOutput(bipartition, confidences);
                break;
        }
        return results;

    }

    /**
     * Calculates the confidences of the labels, based on the neighboring
     * instances
     *
     * @param neighbours
     *            the list of nearest neighboring instances
     * @param distances
     *            the distances of the neighbors
     * @return the confidences of the labels
     */
    private double[] getConfidences(Instances neighbours, double[] distances) {
        double total, weight;
        double neighborLabels = 0;
        double[] confidences = new double[numLabels];

        // Set up a correction to the estimator
        for (int i = 0; i < numLabels; i++) {
            confidences[i] = 1.0 / Math.max(1, train.numInstances());
        }
        total = (double) numLabels / Math.max(1, train.numInstances());

        for (int i = 0; i < neighbours.numInstances(); i++) {
            // Collect class counts
            Instance current = neighbours.instance(i);
            distances[i] = distances[i] * distances[i];
            distances[i] = Math.sqrt(distances[i] / (train.numAttributes() - numLabels));
            weight = 1.0;
            weight *= current.weight();

            for (int j = 0; j < numLabels; j++) {
                double value = Double.parseDouble(current.attribute(
                        labelIndices[j]).value(
                        (int) current.value(labelIndices[j])));
                if (Utils.eq(value, 1.0)) {
                    confidences[j] += weight;
                    neighborLabels += weight;
                }
            }
            total += weight;
        }

        avgPredictedLabels = (int) Math.round(neighborLabels / total);
        // Normalise distribution
        if (total > 0) {
            Utils.normalize(confidences, total);
        }
        return confidences;
    }

    /**
     * used for BRknn-a
     *
     * @param confidences the probabilities for each label
     * @return a bipartition
     */
    protected boolean[] labelsFromConfidences2(double[] confidences) {
        boolean[] bipartition = new boolean[numLabels];
        boolean flag = false; // check the case that no label is true

        for (int i = 0; i < numLabels; i++) {
            if (confidences[i] >= 0.5) {
                bipartition[i] = true;
                flag = true;
            }
        }
        // assign the class with the greater confidence
        if (flag == false) {
            int index = Util.RandomIndexOfMax(confidences, random);
            bipartition[index] = true;
        }
        return bipartition;
    }

    /**
     * used for BRkNN-b (break ties arbitrarily)
     *
     * @param confidences the probabilities for each label
     * @return a bipartition
     */
    protected boolean[] labelsFromConfidences3(double[] confidences) {
        boolean[] bipartition = new boolean[numLabels];

        int[] indices = Utils.stableSort(confidences);

        ArrayList<Integer> lastindices = new ArrayList<Integer>();

        int counter = 0;
        int i = numLabels - 1;

        while (i > 0) {
            if (confidences[indices[i]] > confidences[indices[numLabels - avgPredictedLabels]]) {
                bipartition[indices[i]] = true;
                counter++;
            } else if (confidences[indices[i]] == confidences[indices[numLabels - avgPredictedLabels]]) {
                lastindices.add(indices[i]);
            } else {
                break;
            }
            i--;
        }

        int size = lastindices.size();

        int j = avgPredictedLabels - counter;
        while (j > 0) {
            int next = random.nextInt(size);
            if (bipartition[lastindices.get(next)] != true) {
                bipartition[lastindices.get(next)] = true;
                j--;
            }
        }

        return bipartition;
    }

    /**
     * set the maximum number of neighbors to be evaluated via cross-validation
     *
     * @param cvMaxK  Maximum number of neighbors
     */
    public void setCvMaxK(int cvMaxK) {
        this.cvMaxK = cvMaxK;
    }

    public String globalInfo() {
        return "Simple BR implementation of the KNN algorithm." +
               "For more information, see\n\n" + 
                getTechnicalInformation().toString();
    }

}