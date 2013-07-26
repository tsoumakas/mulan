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

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

/**
 * <p>Class implementing the DML-kNN (Dependent Multi-Label k Nearest
 * Neighbours) algorithm. For more information, see <em>Zoulficar Younes, Fahed
 * Abdallah, Thierry Denceaux (2008). Multi-label classification algorithm
 * derived from k-nearest neighbor rule with label dependencies. In Proceedings
 * of 16th European Signal Processing Conference (EUSIPCO 2008), Lausanne,
 * Switzerland</em>.</p>
 *
 * @author Oscar Gabriel Reyes Pupo
 * @version 2012.11.24
 */
@SuppressWarnings("serial")
public class DMLkNN extends MultiLabelKNN {

    /**
     * Smoothing parameter controlling the strength of uniform prior <br>
     * (Default value is set to 1 which yields the Laplace smoothing).
     */
    protected double smooth;
    /**
     * A table holding the prior probability for an instance to belong in each
     * class
     */
    private double[] PriorProbabilities;
    /**
     * A table holding the prior probability for an instance not to belong in
     * each class
     */
    private double[] PriorNProbabilities;
    /**
     * A table holding the number of instances belonging to each class
     */
    private int[] C;
    /**
     * delta parameter (fuzzy parameter), can be calculated by cross validation
     */
    private int delta;
    /**
     * A table holding the number of instances not belonging to each class
     */
    private int[] NC;
    /**
     * A table holding the number of nearest neighbours belonging to each class
     * per training instance
     */
    private int[][] Ci;

    /**
     * @param numOfNeighbors : the number of neighbors
     * @param smooth : the smoothing factor
     */
    public DMLkNN(int numOfNeighbors, double smooth) {
        super(numOfNeighbors);
        this.smooth = smooth;
    }

    /**
     * The default constructor
     */
    public DMLkNN() {
        super();
        this.smooth = 1.0;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Zoulficar Younes, Fahed Abdallah, Thierry Denceaux");
        result.setValue(Field.TITLE, "Multi-label classification algorithm derived from k-nearest neighbor rule with label dependencies");
        result.setValue(Field.BOOKTITLE, "Proceedings of 16th European Signal Processing Conference (EUSIPCO 2008)");
        result.setValue(Field.LOCATION, "Lausanne, Switzerland");
        result.setValue(Field.YEAR, "2008");
        return result;
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        super.buildInternal(train);

        delta = 6;

        PriorProbabilities = new double[numLabels];
        PriorNProbabilities = new double[numLabels];
        C = new int[numLabels];
        NC = new int[numLabels];

        ComputePrior();

        Ci = new int[train.getNumInstances()][train.getNumLabels()];
        ComputeCountingMemberShip();

        if (getDebug()) {
            System.out.println("Computed Prior Probabilities");
            for (int i = 0; i < numLabels; i++) {
                System.out.println("Label " + (i + 1) + ": " + PriorProbabilities[i]);
            }
        }
    }

    private void ComputeCountingMemberShip() {

        Instances knn = null;

        for (int inst = 0; inst < train.numInstances(); inst++) {

            //find the k nearest neighbours
            try {
                knn = new Instances(lnn.kNearestNeighbours(train.instance(inst), numOfNeighbors));
            } catch (Exception ex) {
                Logger.getLogger(DMLkNN.class.getName()).log(Level.SEVERE, null, ex);
            }

            //Calculating the membership counting vector of the ith instance

            for (int i = 0; i < numLabels; i++) {
                int temp_Ci = 0;
                for (int k = 0; k < numOfNeighbors; k++) {
                    double value = Double.parseDouble(train.attribute(labelIndices[i]).value(
                            (int) knn.instance(k).value(labelIndices[i])));
                    if (Utils.eq(value, 1.0)) {
                        temp_Ci++;
                    }
                }
                Ci[inst][i] = temp_Ci;
            }
        }

    }

    /**
     * Computing Prior and PriorN Probabilities for each class of the training
     * set
     */
    private void ComputePrior() {
        for (int i = 0; i < numLabels; i++) {
            int temp_Ci = 0;
            for (int j = 0; j < train.numInstances(); j++) {
                double value = Double.parseDouble(train.attribute(labelIndices[i]).value(
                        (int) train.instance(j).value(labelIndices[i])));
                if (Utils.eq(value, 1.0)) {
                    temp_Ci++;
                }
            }
            PriorProbabilities[i] = (smooth + temp_Ci) / (smooth * 2 + train.numInstances());
            PriorNProbabilities[i] = 1 - PriorProbabilities[i];
            C[i] = temp_Ci;
            NC[i] = train.numInstances() - C[i];
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] confidences = new double[numLabels];
        boolean[] predictions = new boolean[numLabels];

        Instances knn = null;
        try {
            knn = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));
        } catch (Exception ex) {
            Logger.getLogger(DMLkNN.class.getName()).log(Level.SEVERE, null, ex);
        }

        //Calculating the membership counting vector of the query instance

        int Ct[] = new int[numLabels];

        for (int i = 0; i < numLabels; i++) {
            int temp_Ci = 0;
            for (int k = 0; k < numOfNeighbors; k++) {
                double value = Double.parseDouble(train.attribute(labelIndices[i]).value(
                        (int) knn.instance(k).value(labelIndices[i])));
                if (Utils.eq(value, 1.0)) {
                    temp_Ci++;
                }
            }
            Ct[i] = temp_Ci;
        }

        int V[] = new int[numLabels];
        int NV[] = new int[numLabels];

        //for each training instance

        for (int inst = 0; inst < train.numInstances(); inst++) {

            boolean forAll = true;

            for (int q = 0; q < numLabels; q++) {

                if (!(Ci[inst][q] >= (Ct[q] - delta) && Ci[inst][q] <= (Ct[q] + delta))) {
                    forAll = false;
                    break;

                }

            }

            if (forAll) {

                for (int q = 0; q < numLabels; q++) {

                    if (Ci[inst][q] == Ct[q]) {

                        double value = Double.parseDouble(train.attribute(labelIndices[q]).value(
                                (int) train.instance(inst).value(labelIndices[q])));

                        if (Utils.eq(value, 1.0)) {
                            V[q]++;
                        } else {
                            NV[q]++;
                        }

                    }
                }
            }
        }


        //Computing yt and rt
        for (int i = 0; i < numLabels; i++) {

            double Prob_in = PriorProbabilities[i] * (smooth + V[i]) / (smooth * numLabels + C[i]);
            double Prob_out = PriorNProbabilities[i] * (smooth + NV[i]) / (smooth * numLabels + NC[i]);

            if (Prob_in >= Prob_out) {
                predictions[i] = true;
            } else {
                predictions[i] = false;
            }

            // ranking function
            if ((Prob_in + Prob_out) == 0) {
                confidences[i] = Prob_in;
            } else {
                confidences[i] = Prob_in / (Prob_in + Prob_out);
            }

        }
        MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
        return mlo;
    }

    /**
     * Returns the value of the delta parameter
     * 
     * @return delta the value of the delta parameter
     */
    public int getDelta() {
        return delta;
    }

    /**
     * Sets the value of the delta parameter
     * 
     * @param delta a value for the delta parameter
     */
    public void setDelta(int delta) {
        this.delta = delta;
    }
}
