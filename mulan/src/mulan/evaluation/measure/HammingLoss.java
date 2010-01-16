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
 *    HammingLoss.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation.measure;

import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;

/**
 * Implementation of the Hamming loss function.
 * 
 * @author Grigorios Tsoumakas
 */
public class HammingLoss extends ExampleBasedMeasure {

    public String getName() {
        return "Hamming Loss";
    }

    public double getIdealValue() {
        return 0;
    }

    public double updateInternal(MultiLabelOutput prediction, boolean[] truth) {
        boolean[] bipartition = prediction.getBipartition();
        if (bipartition == null) {
            throw new ArgumentNullException("Bipartition is null");
        }
        if (bipartition.length != truth.length) {
            throw new IllegalArgumentException("The dimensions of the " +
                    "bipartition and the ground truth array do not match");
        }

        double symmetricDifference = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i] != truth[i]) {
                symmetricDifference++;
            }
        }
        double value = symmetricDifference/truth.length;

        sum += value;
        count++;

        return value;
    }

    public double getValue() {
        return sum / count;
    }
}
