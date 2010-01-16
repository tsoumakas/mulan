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
 *    OneError.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation.measure;

import mulan.classifier.MultiLabelOutput;

/**
 * Implementation of the one-error measure. For a given example and prediction, 
 * one-error is 1 if the top ranked label is a relevant and 0 otherwise.
 * 
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 */
public class OneError extends ExampleBasedMeasure {

    public String getName() {
        return "One-Error";
    }

    /**
     * {@inheritDoc}<br/>
     * The computed value of one-error is from {0,1} set. The one-error is '1'
     * if the top ranked label is relevant.
     */
    public double updateInternal(MultiLabelOutput output, boolean[] trueLabels) {

        double oneError = 0;
        int[] ranks = output.getRanking();
        int numLabels = trueLabels.length;
        for (int topRated = 0; topRated < numLabels; topRated++) {
            if (ranks[topRated] == 1) {
                if (!trueLabels[topRated]) {
                    oneError++;
                    sum += oneError;
                }
                break;
            }
        }
        count++;
        return oneError;
    }

    @Override
    public double getIdealValue() {
        return 0;
    }
}
