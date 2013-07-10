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
 * @version 2010.12.03
 */
public abstract class RankingMeasureBase extends ClassificationMeasureBase {
    /**
     * The current sum of the measure
     */
    protected double sum;
    /**
     * The number of validation examples processed
     */
    protected int count;

    @Override
    public void reset() {
        sum = 0;
        count = 0;
    }

    @Override
    public double getValue() {
        return sum / count;
    }

    @Override
    protected void updateInternal(MultiLabelOutput prediction, boolean[] truth) {
        int[] ranking = prediction.getRanking();
        if (ranking == null) {
            throw new ArgumentNullException("Bipartition is null");
        }
        if (ranking.length != truth.length) {
            throw new IllegalArgumentException("The dimensions of the " +
                    "bipartition and the ground truth array do not match");
        }
        updateRanking(ranking, truth);
    }

    /**
     * Updates the measure based on an example
     *
     * @param ranking the predicted ranking
     * @param truth the ground truth
     */
    protected abstract void updateRanking(int[] ranking, boolean[] truth);

}