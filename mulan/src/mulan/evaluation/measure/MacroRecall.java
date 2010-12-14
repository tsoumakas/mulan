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
 *    MacroRecall.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import mulan.core.MulanRuntimeException;

/**
 * Implementation of the macro-averaged recall measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class MacroRecall extends LabelBasedRecall {

    private boolean strict;

    /**
     * Constructs a new object with given number of labels and strictness
     *
     * @param numOfLabels the number of labels
     * @param isStrict when false, divisions-by-zero are ignored
     */
    public MacroRecall(int numOfLabels, boolean isStrict) {
        super(numOfLabels);
        strict = isStrict;
    }

    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            if (truePositives[labelIndex] + falseNegatives[labelIndex] == 0) {
                if (strict) {
                    throw new MulanRuntimeException("None example actually positive");
                } 
            } else {
                sum += truePositives[labelIndex] / (truePositives[labelIndex] + falseNegatives[labelIndex]);
                count++;
            }
        }
        return sum / count;
    }

    public String getName() {
        return "Macro-averaged Recall";
    }
}
