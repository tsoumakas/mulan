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
package mulan.evaluation.loss;

import java.io.Serializable;

import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;

/**
 * Base class for confidence loss functions
 *
 * @author Christina Papagiannopoulou
 * @version 2013.6.13
 */
public abstract class ConfidenceLossFunctionBase implements ConfidenceLossFunction, Serializable {

    private void checkConfidences(double[] confidences) {
        if (confidences == null) {
            throw new ArgumentNullException("Confidences is null");
        }
    }

    private void checkLength(double[] confidences, boolean[] groundTruth) {
        if (confidences.length != groundTruth.length) {
            throw new IllegalArgumentException("The dimensions of the "
                    + "confidences and the ground truth array do not match");
        }
    }

    @Override
    public final double computeLoss(MultiLabelOutput prediction, boolean[] groundTruth) {
        double[] confidences = prediction.getConfidences();
        checkConfidences(confidences);
        checkLength(confidences, groundTruth);
        return computeLoss(confidences, groundTruth);
    }

    @Override
    abstract public double computeLoss(double[] confidences, boolean[] groundTruth);
}
