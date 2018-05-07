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

/**
 * Implementation of MAP (Mean Average Precision)
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2010.12.10
 */
public class MeanAveragePrecision extends LabelBasedAveragePrecision implements MacroAverageMeasure {

    /**
     * Creates a new instance of this class
     *
     * @param numOfLabels the number of labels
     */
    public MeanAveragePrecision(int numOfLabels) {
        super(numOfLabels);
    }

    /**
     * Calculates map using multiple calls to {@link #getValue(int)}. If a label
     * has 0 relevant examples, then it is omitted from the average.
     */
    @Override
    public double getValue() {
        // counts the number of labels with with no relevant examples (for which average precision
        // is NaN)
        int zeroRelevantCounter = 0;
        double map = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            double ap = getValue(labelIndex);
            if (ap >= 0) {
                map += ap;
            } else {
                zeroRelevantCounter++;
            }
        }
        // System.out.println("Labels with zero relevant examples: " + zeroRelevantCounter);
        return map / (numOfLabels - zeroRelevantCounter);
    }

    /**
     * Returns the average precision for a label. If there are no relevant
     * examples for a given label, {@link Double#NaN} is returned.
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the average precision for the given label
     */
    @Override
    public double getValue(int labelIndex) {
        double ap = 0;
        Collections.sort(confact[labelIndex], Collections.reverseOrder());
        double retrievedCounter = 0, relevantCounter = 0;
        for (int i = 0; i < confact[labelIndex].size(); i++) {
            retrievedCounter++;
            Boolean actual = confact[labelIndex].get(i).getActual();
            if (actual) {
                relevantCounter++;
                ap += relevantCounter / retrievedCounter;
            }
        }
        ap /= relevantCounter;
        return ap;
    }

    @Override
    public String getName() {
        return "Mean Average Precision";
    }

    @Override
    public double getIdealValue() {
        return 1;
    }
}