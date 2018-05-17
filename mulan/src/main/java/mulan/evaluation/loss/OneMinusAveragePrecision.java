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

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of the average precision as loss function.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.12.14
 */
public class OneMinusAveragePrecision extends RankingLossFunctionBase {

    public String getName() {
        return "1 - AveragePrecision";
    }

    @Override
    public double computeLoss(int[] ranking, boolean[] trueLabels) {
        double avgP = 0;
        int numLabels = trueLabels.length;
        List<Integer> relevant = new ArrayList<Integer>();
        for (int index = 0; index < numLabels; index++) {
            if (trueLabels[index]) {
                relevant.add(index);
            }
        }

        if (!relevant.isEmpty()) {
            for (int r : relevant) {
                double rankedAbove = 0;
                for (int rr : relevant) {
                    if (ranking[rr] <= ranking[r]) {
                        rankedAbove++;
                    }
                }
                avgP += (rankedAbove / ranking[r]);
            }
            avgP /= relevant.size();
        }
        return 1 - avgP;
    }
}