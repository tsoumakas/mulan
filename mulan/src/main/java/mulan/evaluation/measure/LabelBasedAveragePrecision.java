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
package mulan.evaluation.measure;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2010.12.04
 */
public abstract class LabelBasedAveragePrecision extends ConfidenceMeasureBase {

    /** the number of labels */
    protected int numOfLabels;
    /** collection that stores all predictions and ground truths */
    protected List<ConfidenceActual>[] confact;

    /**
     * Creates a new instance of this class
     *
     * @param numOfLabels the number of labels
     */
    public LabelBasedAveragePrecision(int numOfLabels) {
        this.numOfLabels = numOfLabels;
        confact = new ArrayList[numOfLabels];
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            confact[labelIndex] = new ArrayList<>();
        }
    }

    @Override
    protected void updateConfidence(double[] confidences, boolean[] truth) {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            boolean actual = truth[labelIndex];
            // boolean predicted = bipartition[labelIndex];
            double confidence = confidences[labelIndex];
            // another metric...
            // if (predicted) {
            confact[labelIndex].add(new ConfidenceActual(confidence, actual));
            // }
        }
    }

    @Override
    public void reset() {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            confact[labelIndex].clear();
        }
    }

    /**
     * Class that stores a confidence and a ground truth for one label/example
     */
    protected class ConfidenceActual implements Comparable, Serializable {

        private boolean actual;
        private double confidence;

        /**
         * Creates a new instance of this class
         *
         * @param confidence the confidence
         * @param actual the ground truth
         */
        public ConfidenceActual(double confidence, boolean actual) {
            this.actual = actual;
            this.confidence = confidence;
        }

        /**
         * Returns the ground truth
         *
         * @return the ground truht
         */
        public boolean getActual() {
            return actual;
        }

        /**
         * Returns the confidence
         *
         * @return the confidence
         */
        public double getConfidence() {
            return confidence;
        }

        @Override
        public int compareTo(Object o) {
            if (this.confidence > ((ConfidenceActual) o).confidence) {
                return 1;
            } else if (this.confidence < ((ConfidenceActual) o).confidence) {
                return -1;
            } else {
                return 0;
            }
        }
    }
}