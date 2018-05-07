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

import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;

/**
 * Implementation of the label-based macro precision measure.
 * 
 * @author Grigorios Tsoumakas
 * @version 2012.07.17
 */
public abstract class LabelBasedAUC extends ConfidenceMeasureBase {

    /** The number of labels */
    protected int numOfLabels;
    /** The predictions for each label */
    protected FastVector[] m_Predictions;
    /** The predictions for all labels */
    protected FastVector all_Predictions;

    /**
     * Creates a new instance of this class
     * 
     * @param numOfLabels the number of labels
     */
    public LabelBasedAUC(int numOfLabels) {
        this.numOfLabels = numOfLabels;
        m_Predictions = new FastVector[numOfLabels];
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            m_Predictions[labelIndex] = new FastVector();
        }
        all_Predictions = new FastVector();
    }

    @Override
    public void reset() {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            m_Predictions[labelIndex] = new FastVector();
        }
        all_Predictions = new FastVector();
    }

    @Override
    public double getIdealValue() {
        return 1;
    }
   
    @Override
    protected void updateConfidence(double[] confidences, boolean[] truth) {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {

            int classValue;
            boolean actual = truth[labelIndex];
            if (actual) {
                classValue = 1;
            } else {
                classValue = 0;
            }

            double[] dist = new double[2];
            dist[1] = confidences[labelIndex];
            dist[0] = 1 - dist[1];

            m_Predictions[labelIndex].addElement(new NominalPrediction(classValue, dist, 1));
            all_Predictions.addElement(new NominalPrediction(classValue, dist, 1));
        }
    }
}