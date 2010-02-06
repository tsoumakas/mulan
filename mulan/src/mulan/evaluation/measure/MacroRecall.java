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
import weka.core.Utils;

/**
 * Implementation of the macro-averaged recall measure.
 *
 * @author Grigorios Tsoumakas
 */
public class MacroRecall extends LabelBasedRecall {

    public MacroRecall(int numOfLabels) {
        super(numOfLabels);
    }

    public double getValue() {
        double[] labelRecall = new double[numOfLabels];
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            if (truePositives[labelIndex] + falseNegatives[labelIndex] == 0) {
                throw new MulanRuntimeException("None example actually positive");
            }
            labelRecall[labelIndex] = truePositives[labelIndex] / (truePositives[labelIndex] + falseNegatives[labelIndex]);
        }
        return Utils.mean(labelRecall);
    }

    public String getName() {
        return "Macro-averaged Recall";
    }
}
