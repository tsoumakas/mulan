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

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of the average precision measure. It evaluates the average
 * fraction of labels ranked above a particular relevant label, that are
 * actually relevant.
 * 
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class AveragePrecision extends RankingMeasureBase {

    @Override
    public String getName() {
        return "Average Precision";
    }

    @Override
    public double getIdealValue() {
        return 1;
    }

    @Override
    protected void updateRanking(int[] ranking, boolean[] trueLabels) {
        double avgP = 0;
        int numLabels = trueLabels.length;
        List<Integer> relevant = new ArrayList<>();
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
            sum += avgP;
            count++;
        }
    }

}