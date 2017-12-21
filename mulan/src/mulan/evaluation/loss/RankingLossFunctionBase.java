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
 * Base class for ranking loss functions
 *
 * @author GrigoriosTsoumakas
 * @version 2010.11.10
 */
public abstract class RankingLossFunctionBase implements RankingLossFunction, Serializable  {

    private void checkRanking(int[] ranking) {
        if (ranking == null) {
            throw new ArgumentNullException("Ranking is null");
        }
    }

    private void checkLength(int[] ranking, boolean[] groundTruth) {
        if (ranking.length != groundTruth.length) {
            throw new IllegalArgumentException("The dimensions of the " +
                    "ranking and the ground truth array do not match");
        }
    }

    public final double computeLoss(MultiLabelOutput prediction, boolean[] groundTruth) {
        int[] ranking = prediction.getRanking();
        checkRanking(ranking);
        checkLength(ranking, groundTruth);
        return computeLoss(ranking, groundTruth);
    }

    abstract public double computeLoss(int[] ranking, boolean[] groundTruth);
}