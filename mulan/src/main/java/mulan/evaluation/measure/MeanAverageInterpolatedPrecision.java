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

import java.util.Collections;
import weka.core.Utils;

/**
 * Implementation of MAiP (Mean Average Interpolated Precision)
 *
 * @author Fragkiskos Chatziasimidis
 * @author John Panagos
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2012.07.28
 */
public class MeanAverageInterpolatedPrecision extends LabelBasedAveragePrecision implements MacroAverageMeasure {

    /**
     * The number of recall levels for which MeanAverageInterpolatedPrecision is
     * calculated
     */
    private int numRecallLevels;

    /**
     * Constructor
     *
     * @param numOfLabels the number of labels
     * @param numRecallLevels the number of standard recall levels uniformly
     * distributed in [0,1]
     */
    public MeanAverageInterpolatedPrecision(int numOfLabels, int numRecallLevels) {
        super(numOfLabels);
        this.numRecallLevels = numRecallLevels;
    }

    @Override
    public String getName() {
        return "Mean Average Interpolated Precision";
    }

    @Override
    public double getValue() {
        // counts the number of labels with with no relevant examples (for which interpolated
        // average precision is NaN)
        int zeroRelevantCounter = 0;
        double miap = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            double iap = getValue(labelIndex);
            if (iap >= 0) {
                miap += iap;
            } else {
                zeroRelevantCounter++;
            }
        }
        // System.out.println("Labels with zero relevant examples: " + zeroRelevantCounter);
        return miap / (numOfLabels - zeroRelevantCounter);
    }

    /**
     * Returns the average interpolated precision for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the average interpolated precision for the given label
     */
    @Override
    public double getValue(int labelIndex) {
        double[] precisions = new double[confact[labelIndex].size()];
        double[] recalls = new double[confact[labelIndex].size()];
        double[] interpolatedPrecision = new double[numRecallLevels];
        Collections.sort(confact[labelIndex], Collections.reverseOrder());
        double retrievedCounter = 0;
        double relevantCounter = 0;
        double totalRelevantCounter;
        // calculate precision in all positions and count the total number of relevant instances
        for (int i = 0; i < confact[labelIndex].size(); i++) {
            retrievedCounter++;
            Boolean actual = confact[labelIndex].get(i).getActual();
            if (actual) {
                relevantCounter++;
            }
            precisions[i] = relevantCounter / retrievedCounter;
        }
        totalRelevantCounter = relevantCounter;
        if (totalRelevantCounter == 0) {
            return Double.NaN;
        }
        // calculate recall in all positions
        relevantCounter = 0;
        for (int i = 0; i < confact[labelIndex].size(); i++) {
            Boolean actual = confact[labelIndex].get(i).getActual();
            if (actual) {
                relevantCounter++;
            }
            recalls[i] = relevantCounter / totalRelevantCounter;
        }
        for (int i = 0; i < numRecallLevels; i++) {
            double recallLevel = (double) i / (numRecallLevels - 1);
            int pos = findRecallPosition(recallLevel, recalls);
            interpolatedPrecision[i] = highestPrecisionAfterPos(pos, precisions);
        }
        double averageInterpolatedPrecision = Utils.mean(interpolatedPrecision);

        return averageInterpolatedPrecision;
    }

    @Override
    public double getIdealValue() {
        return 1;
    }

    /**
     * Finds the highest precision after the given position in the list of
     * retrieved instances.
     *
     * @param pos
     * @param precisions
     * @return
     */
    private double highestPrecisionAfterPos(int pos, double[] precisions) {
        double maxPrecision = precisions[pos];
        for (int i = pos + 1; i < precisions.length; i++) {
            if (precisions[i] > maxPrecision) {
                maxPrecision = precisions[i];
            }
        }
        return maxPrecision;
    }

    /**
     * Finds the position in the list of retrieved instances where recall
     * becomes equal to or greater than the given value. It returns -1 in case
     * that there are no positive examples for a label.
     *
     * @param recallLevel
     * @param recalls
     * @return
     */
    private int findRecallPosition(double recallLevel, double[] recalls) {
        int pos = -1;
        for (int i = 0; i < recalls.length; i++) {
            if (recalls[i] >= recallLevel) {
                pos = i;
                break;
            }
        }
        return pos;
    }
}