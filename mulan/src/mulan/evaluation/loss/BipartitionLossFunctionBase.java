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
 * Base class for bipartition loss functions
 *
 * @author GrigoriosTsoumakas
 * @version 2010.11.10
 */
public abstract class BipartitionLossFunctionBase implements BipartitionLossFunction, Serializable  {

    private void checkBipartition(boolean[] bipartition) {
        if (bipartition == null) {
            throw new ArgumentNullException("Bipartition is null");
        }
    }

    private void checkLength(boolean[] bipartition, boolean[] groundTruth) {
        if (bipartition.length != groundTruth.length) {
            throw new IllegalArgumentException("The dimensions of the " +
                    "bipartition and the ground truth array do not match");
        }
    }

    public final double computeLoss(MultiLabelOutput prediction, boolean[] groundTruth) {
        boolean[] bipartition = prediction.getBipartition();
        checkBipartition(bipartition);
        checkLength(bipartition, groundTruth);
        return computeLoss(bipartition, groundTruth);
    }

    abstract public double computeLoss(boolean[] bipartition, boolean[] groundTruth);
}