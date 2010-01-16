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
 *    MacroPrecision.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation.measure;

import mulan.core.MulanRuntimeException;
import weka.core.Utils;

/**
 * Implementation of the macro-averaged precision measure.
 *
 * @author Grigorios Tsoumakas
 */
public class MacroPrecision extends LabelBasedPrecision {

    public MacroPrecision(int numOfLabels) {
        super(numOfLabels);
    }

    public double getValue() {
        double[] labelPrecision = new double[numOfLabels];
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            if (truePositives[labelIndex] + falsePositives[labelIndex] == 0) {
                throw new MulanRuntimeException("None example predicted positive");
            }
            labelPrecision[labelIndex] = truePositives[labelIndex] / (truePositives[labelIndex] + falsePositives[labelIndex]);
        }
        return Utils.mean(labelPrecision);
    }

    public String getName() {
        return "Macro-averaged Precision";
    }
}
