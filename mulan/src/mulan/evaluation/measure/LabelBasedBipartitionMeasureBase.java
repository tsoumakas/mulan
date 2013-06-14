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

/**
 * Base class for label-based bipartition measures
 *
 * @author Grigorios Tsoumakas
 * @version 2012.05.29
 */
public abstract class LabelBasedBipartitionMeasureBase extends BipartitionMeasureBase {

    /**
     * the number of labels
     */
    protected int numOfLabels;
    /**
     * the number of false negative for each label
     */
    protected double[] falseNegatives;
    /**
     * the number of true positives for each label
     */
    protected double[] truePositives;
    /**
     * the number of false positives for each label
     */
    protected double[] falsePositives;
    /**
     * the number of true negatives for each label
     */
    protected double[] trueNegatives;

    /**
     * Creates a new instance of this class
     *
     * @param aNumOfLabels the number of labels
     */
    public LabelBasedBipartitionMeasureBase(int aNumOfLabels) {
        numOfLabels = aNumOfLabels;
        falseNegatives = new double[numOfLabels];
        truePositives = new double[numOfLabels];
        falsePositives = new double[numOfLabels];
        trueNegatives = new double[numOfLabels];
    }

    @Override
    public void updateBipartition(boolean[] bipartition, boolean[] truth) {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            boolean actual = truth[labelIndex];
            boolean predicted = bipartition[labelIndex];

            if (actual) {
                if (predicted) {
                    truePositives[labelIndex]++;
                } else {
                    falseNegatives[labelIndex]++;
                }
            } else {
                if (predicted) {
                    falsePositives[labelIndex]++;
                } else {
                    trueNegatives[labelIndex]++;
                }
            }
        }
    }

    @Override
    public void reset() {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            falseNegatives[labelIndex] = 0;
            truePositives[labelIndex] = 0;
            falsePositives[labelIndex] = 0;
            trueNegatives[labelIndex] = 0;
        }
    }
}