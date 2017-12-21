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

import mulan.evaluation.loss.RankingLossFunction;

/**
 *
 * @author Grigorios Tsoumakas
 * @version 2010.11.10
 */
public abstract class LossBasedRankingMeasureBase extends RankingMeasureBase {

    // a ranking loss function
    private final RankingLossFunction loss;

    /**
     * Creates a loss-based ranking measure
     *
     * @param aLoss a ranking loss function
     */
    public LossBasedRankingMeasureBase(RankingLossFunction aLoss) {
        loss = aLoss;
    }

    @Override
    protected void updateRanking(int[] ranking, boolean[] truth) {
        sum += loss.computeLoss(ranking, truth);
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