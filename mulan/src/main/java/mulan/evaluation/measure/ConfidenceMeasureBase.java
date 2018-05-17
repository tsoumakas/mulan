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

import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;

/**
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.12.04
 */
public abstract class ConfidenceMeasureBase extends ClassificationMeasureBase {

    @Override
    protected void updateInternal(MultiLabelOutput prediction, boolean[] truth) {
        double[] confidences = prediction.getConfidences();
        if (confidences == null) {
            throw new ArgumentNullException("Bipartition is null");
        }
        if (confidences.length != truth.length) {
            throw new IllegalArgumentException("The dimensions of the " +
                    "confidence array and the ground truth array do not match");
        }
        updateConfidence(confidences, truth);
    }

    /**
     * Updates the measure for a new example
     *
     * @param confidences the confidences output by the learner for the example
     * @param truth the ground truth of the example
     */
    abstract protected void updateConfidence(double[] confidences, boolean[] truth);
}