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
 *    MicroAUC.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

/**
 * Implementation of the micro-averaged AUC measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.10
 */
public class MicroAUC extends LabelBasedAUC {

    /**
     * Creates a new instance of this class
     * @param numOfLabels
     */
    public MicroAUC(int numOfLabels) {
        super(numOfLabels);
    }

    public String getName() {
        return "Micro-averaged AUC";
    }

    public double getValue() {
        ThresholdCurve tc = new ThresholdCurve();
        Instances result = tc.getCurve(all_Predictions, 1);
        return ThresholdCurve.getROCArea(result);
    }
}
