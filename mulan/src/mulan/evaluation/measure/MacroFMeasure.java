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

/*
 *    MacroFMeasure.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import mulan.core.MulanRuntimeException;

/**
 * Implementation of the macro-averaged f measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class MacroFMeasure extends LabelBasedFMeasure {

    private boolean strict;

    /**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     * @param isStrict when false, divisions-by-zero are ignored
     */
    public MacroFMeasure(int numOfLabels, boolean isStrict) {
        super(numOfLabels);
        strict = isStrict;
    }

    /**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     * @param isStrict when false, divisions-by-zero are ignored
     * @param beta controls the combination of precision and recall
     */
    public MacroFMeasure(int numOfLabels, boolean isStrict, double beta) {
        super(numOfLabels, beta);
        strict = isStrict;
    }

    public String getName() {
        return "Macro-averaged F-Measure";
    }

    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            if (truePositives[labelIndex] + falsePositives[labelIndex] == 0) {
                if (strict) {
                    throw new MulanRuntimeException("None example predicted positive");
                } else {
                    continue;
                }
            }
            if (truePositives[labelIndex] + falseNegatives[labelIndex] == 0) {
                if (strict) {
                    throw new MulanRuntimeException("None example actually positive");
                } else {
                    continue;
                }
            }
            double precision = truePositives[labelIndex] / (truePositives[labelIndex] + falsePositives[labelIndex]);
            double recall = truePositives[labelIndex] / (truePositives[labelIndex] + falseNegatives[labelIndex]);
            sum += calculateFMeasure(precision, recall);
            count++;
        }
        return sum / count;
    }
}
