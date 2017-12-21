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

import mulan.evaluation.loss.ConfidenceLossFunction;

/**
 * Base class for loss-based confidence measures
 * 
 * @author Christina Papagiannopoulou
 * @version 2013.6.13
 */
public abstract class LossBasedConfidenceMeasureBase extends ExampleBasedConfidenceMeasureBase {

    // a confidence loss function
    private final ConfidenceLossFunction loss;

    /**
     * Creates a loss-based confidence measure
     *
     * @param aLoss a confidence loss function
     */
    public LossBasedConfidenceMeasureBase(ConfidenceLossFunction aLoss) {
        loss = aLoss;
    }

    @Override
    public void updateConfidence(double[] confidences, boolean[] truth) {
        sum += loss.computeLoss(confidences, truth);
        count++;
    }

    @Override
    public String getName() {
        return loss.getName();
    }

    @Override
    public double getIdealValue() {
        return 0;
    }
}