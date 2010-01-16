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
 *    MeasureBase.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation.measure;

import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;

/**
 * 
 * @author Grigorios Tsoumakas
 */
public abstract class MeasureBase implements Measure {

    public final double update(MultiLabelOutput prediction, boolean[] truth) {
        if (prediction == null) {
            throw new ArgumentNullException("Prediction is null");
        }
        if (truth == null) {
            throw new ArgumentNullException("Ground truth is null");
        }
        if (truth.length < 2) {
            throw new IllegalArgumentException("Length of ground truth is less than 2");
        }

        return updateInternal(prediction, truth);
    }

    @Override
    public String toString() {
        double value = Double.NaN;
        try {
            value = getValue();
        } catch (Exception ex) {
        }
        return getName() + ": " + value;
    }

    protected abstract double updateInternal(MultiLabelOutput prediction, boolean[] truth);
}
